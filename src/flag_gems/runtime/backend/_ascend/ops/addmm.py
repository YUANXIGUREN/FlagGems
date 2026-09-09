# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._ascend import heuristics_config_utils as _hcu
from flag_gems.runtime.matmul_precision import should_use_fast_float32_matmul
from flag_gems.utils import broadcastable_to, libentry, libtuner

logger = logging.getLogger(__name__)


def classify_addmm_layout(
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
):
    if stride_ak == 1:
        a_layout = "a_row"
    elif stride_am == 1:
        a_layout = "a_column"
    else:
        return "general"

    if stride_bn == 1:
        b_layout = "b_row"
    elif stride_bk == 1 and stride_bn == K:
        b_layout = "b_k_contiguous"
    elif stride_bk == 1 and stride_bn > K:
        b_layout = "b_k_padded"
    else:
        return "general"
    return f"{a_layout}_{b_layout}"


def select_ascend_addmm_kernel(K, layout):
    del layout
    return "skinny_k" if K == 0 else "grouped_gemm"


def select_ascend_input_precision(M, N, K, fp32_operands, fast_enabled):
    hf32_safe_shape = M >= 4096 and N % 16 == 0 and K >= 16
    return (
        "hf32"
        if fp32_operands and fast_enabled and hf32_safe_shape
        else "ieee"
    )


_LAYOUT_IDS = {
    "general": 0,
    "a_row_b_row": 1,
    "a_row_b_k_contiguous": 2,
    "a_row_b_k_padded": 3,
    "a_column_b_row": 4,
    "a_column_b_k_contiguous": 5,
    "a_column_b_k_padded": 6,
}


_SKINNY_K_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16}, num_warps=8
    ),
]


def use_large_ascend_addmm_tiles(M, fp32_operands):
    return fp32_operands and M >= 4096


def prune_ascend_addmm_configs(configs, named_args, **kwargs):
    del kwargs
    fp32_operands = (
        named_args["A"].dtype == torch.float32
        and named_args["B"].dtype == torch.float32
    )
    if use_large_ascend_addmm_tiles(named_args["M"], fp32_operands):
        return configs
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] <= 128
        and config.kwargs["BLOCK_N"] <= 128
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K", "RHS_LAYOUT", "INPUT_PRECISION"],
    prune_configs_by={
        "early_config_prune": prune_ascend_addmm_configs,
    },
)
@triton.heuristics(_hcu.HEURISTICS_CONFIGS["mm"])
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    A,
    B,
    bias,
    C,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Visit neighboring M tiles before advancing N to improve B-tile reuse.
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    ram = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rbn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A += ram[:, None] * stride_am + rk[None, :] * stride_ak
    B += rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A, mask=(ram < M)[:, None], other=0.0)
            b = tl.load(B, mask=(rbn < N)[None, :], other=0.0)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(
                A,
                mask=(ram < M)[:, None] & (rk < k_remaining)[None, :],
                other=0.0,
            )
            b = tl.load(
                B,
                mask=(rk < k_remaining)[:, None] & (rbn < N)[None, :],
                other=0.0,
            )
        acc += tl.dot(
            a,
            b,
            out_dtype=dot_out_dtype,
            input_precision=INPUT_PRECISION,
        )
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    C += ram[:, None] * stride_cm + rbn[None, :] * stride_cn
    mask = (ram < M)[:, None] & (rbn < N)[None, :]
    if BETA_IS_ZERO:
        result = acc * alpha
    else:
        if BIAS_IS_VECTOR:
            # Load a 1-D bias once per output-column tile.
            bias_tile = tl.load(
                bias + stride_in * rbn,
                mask=rbn < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias_tile = tl.load(bias)
        else:
            bias += stride_im * ram[:, None] + stride_in * rbn[None, :]
            bias_tile = tl.load(bias, mask=mask, other=0.0)
        result = acc * alpha + bias_tile.to(acc.dtype) * beta
    tl.store(C, result.to(C.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_SKINNY_K_CONFIGS,
    key=["M", "N", "K", "RHS_LAYOUT"],
    warmup=5,
    rep=10,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_skinny_k_kernel(
    A,
    B,
    bias,
    C,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    RHS_LAYOUT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(0, BLOCK_K):
        a = tl.load(
            A + offs_m * stride_am + k * stride_ak,
            mask=(offs_m < M) & (k < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B + k * stride_bk + offs_n * stride_bn,
            mask=(k < K) & (offs_n < N),
            other=0.0,
        ).to(tl.float32)
        acc += a[:, None] * b[None, :]

    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if BETA_IS_ZERO:
        result = acc * alpha
    else:
        if BIAS_IS_VECTOR:
            bias_tile = tl.load(
                bias + offs_n * stride_in,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias_tile = tl.load(bias)
        else:
            bias_ptrs = (
                bias + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
            )
            bias_tile = tl.load(bias_ptrs, mask=mask, other=0.0)
        result = acc * alpha + bias_tile.to(acc.dtype) * beta
    output = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(output, result.to(C.dtype.element_ty), mask=mask)


def _launch_addmm(bias, mat1, mat2, out, alpha, beta):
    M, K = mat1.shape
    _, N = mat2.shape
    # Keep row- or column-contiguous views and materialize only general strides.
    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()

    # Keep vector/scalar bias compact; broadcast strides cover other valid shapes.
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == N
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        bias_stride_m = 0
        bias_stride_n = bias.stride(0)
    elif bias_is_scalar:
        bias_stride_m = 0
        bias_stride_n = 0
    else:
        bias = bias.broadcast_to(out.shape)
        bias_stride_m = bias.stride(0)
        bias_stride_n = bias.stride(1)
    layout = classify_addmm_layout(
        K,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
    )
    rhs_layout = _LAYOUT_IDS[layout]
    selected_kernel = select_ascend_addmm_kernel(K, layout)
    beta_is_zero = beta == 0
    if selected_kernel == "skinny_k":
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )
        with torch_device_fn.device(mat1.device):
            addmm_skinny_k_kernel[grid](
                mat1,
                mat2,
                bias,
                out,
                alpha,
                beta,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                bias_stride_m,
                bias_stride_n,
                out.stride(0),
                out.stride(1),
                BIAS_IS_VECTOR=bias_is_vector,
                BIAS_IS_SCALAR=bias_is_scalar,
                BETA_IS_ZERO=beta_is_zero,
                RHS_LAYOUT=rhs_layout,
            )
        return out

    dot_out_dtype = tl.float32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META.get("SPLIT_K", 1),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias_stride_m,
            bias_stride_n,
            out.stride(0),
            out.stride(1),
            dot_out_dtype=dot_out_dtype,
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
            BETA_IS_ZERO=beta_is_zero,
            RHS_LAYOUT=rhs_layout,
            INPUT_PRECISION=select_ascend_input_precision(
                M,
                N,
                K,
                mat1.dtype == torch.float32 and mat2.dtype == torch.float32,
                should_use_fast_float32_matmul("ascend", mat1, mat2),
            ),
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ASCEND ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M = mat1.shape[0]
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    return _launch_addmm(bias, mat1, mat2, out, alpha, beta)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_ASCEND ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    bias_c = bias.to(out_dtype)
    return addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)
