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
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.matmul_precision import should_use_fast_float32_matmul
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

from .tf32_cache import TF32RHSCache

logger = logging.getLogger(__name__)


EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "addmm_mthreads_expand.yaml")
)

_TF32_RHS_CACHE = TF32RHSCache()
_TF32_FP16_RHS_CACHE = TF32RHSCache()

_LAYOUT_IDS = {
    "skinny_k": 0,
    "rhs_row": 1,
    "rhs_k_compact": 2,
    "rhs_k_padded": 3,
    "general": 4,
}

_SKINNY_K_CONFIGS = [
    triton.Config(
        {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 256},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 2, "BLOCK_SIZE_N": 256},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64},
        num_stages=1,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128},
        num_stages=1,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64},
        num_stages=1,
        num_warps=8,
    ),
]


@libentry()
@triton.jit
def _round_to_tf32_copy_kernel(
    src,
    dst,
    n_elements,
    n_columns,
    stride_row,
    stride_column,
    BLOCK_SIZE: tl.constexpr,
    TRUNCATE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rows = offsets // n_columns
    columns = offsets - rows * n_columns
    values = tl.load(
        src + rows * stride_row + columns * stride_column,
        mask=mask,
        other=0.0,
    )
    if TRUNCATE:
        bits = values.to(tl.uint32, bitcast=True)
        values = (bits & 0xFFFFE000).to(tl.float32, bitcast=True)
    else:
        values = ext.round_to_tf32(values)
    tl.store(dst + offsets, values, mask=mask)


def round_to_tf32_copy(tensor):
    """Round and materialize a logical FP32 matrix with one Triton kernel."""

    assert tensor.dtype == torch.float32
    assert tensor.ndim == 2
    rounded = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    n_elements = tensor.numel()
    if n_elements == 0:
        return rounded
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    with torch_device_fn.device(tensor.device):
        _round_to_tf32_copy_kernel[grid](
            tensor,
            rounded,
            n_elements,
            tensor.shape[1],
            tensor.stride(0),
            tensor.stride(1),
            BLOCK_SIZE=1024,
            TRUNCATE=False,
        )
    return rounded


def _get_rounded_tf32_rhs(mat2):
    return _TF32_RHS_CACHE.get(mat2, round_to_tf32_copy)


def round_to_tf32_fp16_copy(tensor):
    """Materialize standard RNE-rounded TF32 operands in FP16 storage."""

    assert tensor.dtype == torch.float32
    assert tensor.ndim == 2
    rounded = torch.empty(
        tuple(tensor.shape),
        dtype=torch.float16,
        device=tensor.device,
    )
    n_elements = tensor.numel()
    if n_elements == 0:
        return rounded
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    with torch_device_fn.device(tensor.device):
        _round_to_tf32_copy_kernel[grid](
            tensor,
            rounded,
            n_elements,
            tensor.shape[1],
            tensor.stride(0),
            tensor.stride(1),
            BLOCK_SIZE=1024,
            TRUNCATE=False,
        )
    return rounded


def _get_tf32_fp16_rhs(mat2):
    return _TF32_FP16_RHS_CACHE.get(mat2, round_to_tf32_fp16_copy)


def classify_mthreads_addmm_layout(K, stride_bk, stride_bn):
    if K < 16:
        return "skinny_k"
    if stride_bn == 1:
        return "rhs_row"
    if stride_bk == 1 and stride_bn == K:
        return "rhs_k_compact"
    if stride_bk == 1:
        return "rhs_k_padded"
    return "general"


def select_mthreads_addmm_route(K, sqmma_compatible, promotes_to_fp32):
    if sqmma_compatible and not promotes_to_fp32:
        return "sqmma"
    if K < 16:
        return "skinny_k"
    return "pointer"


def should_inline_round_mthreads_addmm(is_fp32, fast_enabled, route):
    # Standard TF32 uses round-to-nearest-even.  Round pointer tiles explicitly
    # because this backend's bare ``input_precision="tf32"`` path truncates
    # mantissas and accumulates too much error across recurrent GraphCast steps.
    return is_fp32 and fast_enabled and route == "pointer"


def select_mthreads_pointer_precision(is_fp32, fast_enabled):
    # Match the platform's fast-FP32 contract directly in the Triton dot.
    # Unlike the rejected FP16 materialization, this keeps TF32's FP32
    # exponent range and leaves conversion to the MUSA matrix instruction.
    if is_fp32 and fast_enabled:
        return "tf32"
    return "ieee"


def can_use_mthreads_tf32_sqmma_contract(
    is_fp32,
    fast_enabled,
    grad_sensitive,
    M,
    N,
    K,
):
    return (
        is_fp32
        and fast_enabled
        and not grad_sensitive
        # FP16 storage narrows TF32's exponent range.  Keep SQMMA on the
        # Restrict FP16-backed SQMMA to GraphCast's measured N=512 classes and
        # use standard RNE conversion; the recurrent diagnostic is the final
        # guard against precision drift from the reduced exponent range.
        and M >= 4096
        and N == 512
        and K in (512, 1024)
    )


def can_use_mthreads_k4_tiled_contract(
    all_fp32,
    grad_sensitive,
    M,
    N,
    K,
    a_contiguous,
    stride_bk,
    stride_bn,
    bias_is_vector,
    out_contiguous,
):
    """Select the measured large-M, column-major K=4 Triton kernel."""

    return (
        all_fp32
        and not grad_sensitive
        and M >= 4096
        and N == 512
        and K == 4
        and a_contiguous
        and stride_bk == 1
        and stride_bn == 4
        and bias_is_vector
        and out_contiguous
    )


def is_supported_sqmma_layout(tensor):
    # Tensor descriptors consume row-major tensors directly.  Non-contiguous
    # inputs stay on the pointer kernel instead of invoking an ATen copy.
    return tensor.is_contiguous()


def is_sqmma_compatible(a, b, N, K):
    return (
        a.dim() == 2
        and b.dim() == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and a.shape[0] > 0
        and N > 0
        and K > 0
        and N % 8 == 0
        and K % 8 == 0
    )


def explain_mthreads_addmm_route(bias, mat1, mat2, out, out_dtype):
    M, K = mat1.shape
    _, N = mat2.shape
    fast_enabled = should_use_fast_float32_matmul(
        "mthreads", mat1, mat2
    )
    grad_sensitive = torch.is_grad_enabled() and (
        bias.requires_grad or mat1.requires_grad or mat2.requires_grad
    )
    if can_use_mthreads_k4_tiled_contract(
        bias.dtype == mat1.dtype == mat2.dtype == torch.float32,
        grad_sensitive,
        M,
        N,
        K,
        mat1.is_contiguous(),
        mat2.stride(0),
        mat2.stride(1),
        bias.ndim == 1 and bias.shape[0] == N,
        out is None or out.is_contiguous(),
    ):
        return "skinny_k4_tiled"
    if (
        bias.dtype == torch.float32
        and mat2.dtype == torch.float32
        and (out is None or out.is_contiguous())
        and can_use_mthreads_tf32_sqmma_contract(
            mat1.dtype == torch.float32,
            fast_enabled,
            grad_sensitive,
            M,
            N,
            K,
        )
    ):
        return "tf32_sqmma"
    promotes_to_fp32 = (
        out_dtype == torch.float32
        and mat1.dtype in (torch.float16, torch.bfloat16)
    )
    sqmma_compatible = (
        is_sqmma_compatible(mat1, mat2, N, K)
        and (out is None or out.is_contiguous())
    )
    return select_mthreads_addmm_route(
        K,
        sqmma_compatible,
        promotes_to_fp32,
    )


def _prepare_bias(bias, out):
    # Keep vector/scalar bias compact; broadcast strides cover other valid shapes.
    bias_is_vector = bias.ndim == 1 and bias.shape[0] == out.shape[1]
    bias_is_scalar = not bias_is_vector and bias.numel() == 1
    if bias_is_vector:
        return bias, 0, bias.stride(0), True, False
    if bias_is_scalar:
        return bias, 0, 0, False, True
    bias = bias.broadcast_to(out.shape)
    return bias, bias.stride(0), bias.stride(1), False, False


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16},
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 16},
            num_stages=1,
            num_warps=16,
        ),
        # Do not add the tempting 128x128x32/4-warp variant here: MUSA Triton
        # 3.6 compiles it, but produces a silently incorrect FP32 dot result
        # for large-M K=512 cases.  Both K=16 variants pass the full device
        # correctness and real-layout replay suites.
    ],
    key=["M", "N", "K", "stride_bk", "KERNEL_CLASS", "INPUT_PRECISION"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    HAS_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    ROUND_TF32_INPUTS: tl.constexpr,
    GROUP_M: tl.constexpr,
    KERNEL_CLASS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_K:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M)
                & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K)
                & (offs_n[None, :] < N),
                other=0.0,
            )
            if IS_FP64:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            if ROUND_TF32_INPUTS:
                a = ext.round_to_tf32(a)
                b = ext.round_to_tf32(b)
            if INPUT_PRECISION == "tf32x3":
                accumulator += tl.dot(a, b, input_precision="tf32x3")
            elif INPUT_PRECISION == "tf32":
                accumulator += tl.dot(a, b, input_precision="tf32")
            else:
                accumulator += tl.dot(a, b, input_precision="ieee")
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if BETA_IS_ZERO:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                i_ptr + stride_in * offs_cn,
                mask=offs_cn < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            i_ptrs = (
                i_ptr
                + stride_im * offs_cm[:, None]
                + stride_in * offs_cn[None, :]
            )
            bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        result = accumulator * alpha + bias.to(accumulator.dtype) * beta
    c = result.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@libtuner(
    configs=_SKINNY_K_CONFIGS,
    key=["M", "N", "K", "stride_am", "stride_bk", "INPUT_PRECISION"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_skinny_k_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    IS_FP64: tl.constexpr,
    GROUP_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.static_range(0, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr + offs_m * stride_am + k * stride_ak,
            mask=(offs_m < M) & (k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k * stride_bk + offs_n * stride_bn,
            mask=(k < K) & (offs_n < N),
            other=0.0,
        )
        accumulator += a[:, None] * b[None, :]

    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    if BETA_IS_ZERO:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                i_ptr + offs_n * stride_in,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            i_ptrs = (
                i_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
            )
            bias = tl.load(i_ptrs, mask=mask, other=0.0)
        result = accumulator * alpha + bias.to(accumulator.dtype) * beta
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, result.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_skinny_k4_tiled_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
):
    # A two-dimensional tile reuses each of the four A and B values instead
    # of reloading them independently for every flattened output element.
    pid = ext.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < M
    col_mask = cols < N

    a0 = tl.load(a_ptr + rows * stride_am, mask=row_mask, other=0.0)
    a1 = tl.load(
        a_ptr + rows * stride_am + stride_ak, mask=row_mask, other=0.0
    )
    a2 = tl.load(
        a_ptr + rows * stride_am + 2 * stride_ak, mask=row_mask, other=0.0
    )
    a3 = tl.load(
        a_ptr + rows * stride_am + 3 * stride_ak, mask=row_mask, other=0.0
    )
    b0 = tl.load(b_ptr + cols * stride_bn, mask=col_mask, other=0.0)
    b1 = tl.load(
        b_ptr + stride_bk + cols * stride_bn, mask=col_mask, other=0.0
    )
    b2 = tl.load(
        b_ptr + 2 * stride_bk + cols * stride_bn,
        mask=col_mask,
        other=0.0,
    )
    b3 = tl.load(
        b_ptr + 3 * stride_bk + cols * stride_bn,
        mask=col_mask,
        other=0.0,
    )
    accumulator = a0[:, None] * b0[None, :]
    accumulator += a1[:, None] * b1[None, :]
    accumulator += a2[:, None] * b2[None, :]
    accumulator += a3[:, None] * b3[None, :]
    result = alpha * accumulator
    if not BETA_IS_ZERO:
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        result += beta * bias[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        result.to(c_ptr.dtype.element_ty),
        mask=mask,
    )


def addmm_skinny_k4_tiled(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    """Execute the measured K=4 data-reuse kernel without vendor dispatch."""

    M, _ = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty((M, N), dtype=mat1.dtype, device=mat1.device)
    grid = (
        triton.cdiv(M, 32) * triton.cdiv(N, 64),
    )
    with torch_device_fn.device(mat1.device):
        addmm_skinny_k4_tiled_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64,
            BETA_IS_ZERO=beta == 0,
            num_warps=4,
        )
    return out


@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1, num_warps=16),
    ],
    key=["M", "N", "K", "stride_am", "stride_bk", "INPUT_PRECISION"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_skinny_k_flat_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    IS_FP64: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    rows = offsets // N
    columns = offsets - rows * N
    mask = offsets < M * N
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in tl.static_range(0, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr + rows * stride_am + k * stride_ak,
            mask=mask & (k < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k * stride_bk + columns * stride_bn,
            mask=mask & (k < K),
            other=0.0,
        )
        accumulator += a * b

    if BETA_IS_ZERO:
        result = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                i_ptr + columns * stride_in,
                mask=mask,
                other=0.0,
            )
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            bias = tl.load(
                i_ptr + rows * stride_im + columns * stride_in,
                mask=mask,
                other=0.0,
            )
        result = accumulator * alpha + bias.to(accumulator.dtype) * beta
    tl.store(
        c_ptr + rows * stride_cm + columns * stride_cn,
        result.to(c_ptr.dtype.element_ty),
        mask=mask,
    )


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_FMA")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    kernel_class = classify_mthreads_addmm_layout(
        K, mat2.stride(0), mat2.stride(1)
    )
    route = select_mthreads_addmm_route(K, False, False)
    fast_enabled = should_use_fast_float32_matmul(
        "mthreads", mat1, mat2
    )
    inline_round = should_inline_round_mthreads_addmm(
        mat1.dtype == torch.float32,
        fast_enabled,
        route,
    )
    input_precision = select_mthreads_pointer_precision(
        mat1.dtype == torch.float32,
        fast_enabled,
    )

    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    bias, bias_stride_m, bias_stride_n, bias_is_vector, bias_is_scalar = (
        _prepare_bias(bias, out)
    )

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if route == "skinny_k":
        skinny_grid = lambda META: (
            triton.cdiv(M * N, META["BLOCK_SIZE"]),
        )
        with torch_device_fn.device(mat1.device):
            addmm_skinny_k_flat_kernel[skinny_grid](
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
                BLOCK_SIZE_K=16,
                BIAS_IS_VECTOR=bias_is_vector,
                BIAS_IS_SCALAR=bias_is_scalar,
                BETA_IS_ZERO=beta == 0,
                IS_FP64=mat1.dtype == torch.float64,
                INPUT_PRECISION=input_precision,
            )
        return out

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
            BIAS_IS_VECTOR=bias_is_vector,
            BIAS_IS_SCALAR=bias_is_scalar,
            BETA_IS_ZERO=beta == 0,
            HAS_K=K > 0,
            ALLOW_TF32=inline_round,
            ROUND_TF32_INPUTS=inline_round,
            GROUP_M=8,
            KERNEL_CLASS=_LAYOUT_IDS[kernel_class],
            INPUT_PRECISION=input_precision,
            IS_FP64=mat1.dtype == torch.float64,
        )
    return out


def addmm_sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_SIZE_K"], nargs["BLOCK_SIZE_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]]


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
            pre_hook=addmm_sqmma_descriptor_pre_hook,
        ),
    ],
    key=["M", "N", "K"],
    strategy=["default", "default", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="addmm",
    flagtune_expand_op_name="addmm_sqmma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=addmm_sqmma_descriptor_pre_hook,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_sqmma_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    c_desc,
    M,
    N,
    K,
    alpha,
    beta,
    stride_im,
    stride_in,
    DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr,
    BIAS_IS_SCALAR: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_SIZE_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_SIZE_N).to(tl.int32)
    offs_k = 0
    offs_k = offs_k.to(tl.int32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if BETA_IS_ZERO:
        result = alpha * accumulator
    else:
        if BIAS_IS_VECTOR:
            bias = tl.load(
                bias_ptr + offs_n * stride_in,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
        elif BIAS_IS_SCALAR:
            bias = tl.load(bias_ptr)
        else:
            bias_ptrs = (
                bias_ptr
                + offs_m[:, None] * stride_im
                + offs_n[None, :] * stride_in
            )
            bias = tl.load(bias_ptrs, mask=mask, other=0.0)
        result = alpha * accumulator + beta * bias
    tl.store_tensor_descriptor(
        c_desc, [offs_am, offs_bn], result.to(c_desc.dtype)
    )


def addmm_sqmma(mat1, mat2, bias, elem_type, alpha, beta, M, N, K, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_SQMMA")
    device = mat1.device
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    assert mat1.is_contiguous() and mat2.is_contiguous()
    a_type = mat1.dtype
    b_type = mat2.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_type = a_type
    if out is None:
        out = torch.empty((M, N), dtype=c_type, device=device)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    bias, stride_im, stride_in, bias_is_vector, bias_is_scalar = _prepare_bias(
        bias, out
    )
    desc_a = TensorDescriptor.from_tensor(mat1, [1, 1])
    desc_b = TensorDescriptor.from_tensor(mat2, [1, 1])
    desc_c = TensorDescriptor.from_tensor(out, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        1,
        1,
    )
    addmm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        bias,
        desc_c,
        M,
        N,
        K,
        alpha,
        beta,
        stride_im,
        stride_in,
        str(a_type).split(".")[-1],
        BIAS_IS_VECTOR=bias_is_vector,
        BIAS_IS_SCALAR=bias_is_scalar,
        BETA_IS_ZERO=beta == 0,
    )
    return out


def addmm_tf32_sqmma(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    """Run fast FP32 AddMM with Triton conversion and SQMMA kernels."""

    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty(
            (M, N),
            dtype=torch.float32,
            device=mat1.device,
        )
    mat1_fp16 = round_to_tf32_fp16_copy(mat1)
    mat2_fp16 = _get_tf32_fp16_rhs(mat2)
    return addmm_sqmma(
        mat1_fp16,
        mat2_fp16,
        bias,
        torch.float16,
        alpha,
        beta,
        M,
        N,
        K,
        out=out,
    )


def _addmm_impl(bias, mat1, mat2, out, beta, alpha):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    a_dtype = mat1.dtype
    M, K = mat1.shape
    _, N = mat2.shape
    if out is not None:
        assert out.shape == (M, N), "Incompatible output shape"

    fast_enabled = should_use_fast_float32_matmul(
        "mthreads", mat1, mat2
    )
    grad_sensitive = torch.is_grad_enabled() and (
        bias.requires_grad or mat1.requires_grad or mat2.requires_grad
    )
    if can_use_mthreads_k4_tiled_contract(
        bias.dtype == mat1.dtype == mat2.dtype == torch.float32,
        grad_sensitive,
        M,
        N,
        K,
        mat1.is_contiguous(),
        mat2.stride(0),
        mat2.stride(1),
        bias.ndim == 1 and bias.shape[0] == N,
        out is None or out.is_contiguous(),
    ):
        return addmm_skinny_k4_tiled(
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=out,
        )
    if (
        bias.dtype == torch.float32
        and mat2.dtype == torch.float32
        and (out is None or out.is_contiguous())
        and can_use_mthreads_tf32_sqmma_contract(
            mat1.dtype == torch.float32,
            fast_enabled,
            grad_sensitive,
            M,
            N,
            K,
        )
    ):
        return addmm_tf32_sqmma(
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=out,
        )

    if (
        is_sqmma_compatible(mat1, mat2, N, K)
        and bias.dtype == a_dtype
        and (out is None or out.is_contiguous())
    ):
        return addmm_sqmma(
            mat1,
            mat2,
            bias,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
            out=out,
        )
    return addmm_fma(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM")
    return _addmm_impl(bias, mat1, mat2, None, beta, alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_MTHREADS ADDMM_OUT")
    return _addmm_impl(bias, mat1, mat2, out, beta, alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_MTHREADS ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            "mat1 and mat2 must have the same dtype, but got "
            f"{mat1.dtype} and {mat2.dtype}"
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
    M, K = mat1.shape
    _, N = mat2.shape
    a_dtype = mat1.dtype

    # Keep dtype promotion on FMA so FP32 output has no low-precision intermediate.
    if (
        out_dtype == mat1.dtype
        and out.is_contiguous()
        and is_sqmma_compatible(mat1, mat2, N, K)
    ):
        return addmm_sqmma(
            mat1,
            mat2,
            bias_c,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
            out=out,
        )
    return addmm_fma(bias_c, mat1, mat2, alpha=alpha, beta=beta, out=out)
