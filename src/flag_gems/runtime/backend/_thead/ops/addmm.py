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
from flag_gems.ops.addmm import addmm as common_addmm
from flag_gems.ops.addmm import addmm_dtype as common_addmm_dtype
from flag_gems.ops.addmm import addmm_dtype_out as common_addmm_dtype_out
from flag_gems.ops.addmm import addmm_out as common_addmm_out

logger = logging.getLogger(__name__)


def _load_native_addmm_kernels():
    """Save vendor kernels before FlagGems replaces CUDA dispatch."""
    try:
        default_kernel = torch.library.get_kernel("aten::addmm", "CUDA")
        out_kernel = torch.library.get_kernel("aten::addmm.out", "CUDA")
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
        return default_kernel, out_kernel, keyset
    except (AttributeError, RuntimeError):
        return None, None, None


(
    _NATIVE_ADDMM_KERNEL,
    _NATIVE_ADDMM_OUT_KERNEL,
    _NATIVE_ADDMM_KEYSET,
) = _load_native_addmm_kernels()


_MAT2_COPY_MIN_M = 1024


def _can_use_native_fp32_addmm(bias, mat1, mat2):
    """Select the faster vendor GEMM for inference FP32 vector-bias cases."""
    if _NATIVE_ADDMM_KERNEL is None:
        return False
    if bias.requires_grad or mat1.requires_grad or mat2.requires_grad:
        return False
    if bias.dtype != torch.float32 or mat1.dtype != torch.float32:
        return False
    if mat2.dtype != torch.float32 or mat1.dim() != 2 or mat2.dim() != 2:
        return False
    _, K = mat1.shape
    if mat2.shape[0] != K:
        return False
    N = mat2.shape[1]
    return (
        bias.dim() == 1
        and bias.shape[0] == N
        and bias.device == mat1.device
        and mat2.device == mat1.device
    )


def _prepare_mat2(mat1, mat2):
    # PPU FP32 dot is substantially faster when K is the leading dimension of
    # B.  For tall matrices, materializing a transposed weight is amortized by
    # the repeated reuse of B across M and is cheaper than strided kernel loads.
    mat2_k_contiguous = mat2.stride(0) == 1 and mat2.stride(1) > 1
    if (
        mat1.dtype == torch.float32
        and mat2_k_contiguous
        and mat1.shape[0] > _MAT2_COPY_MIN_M
    ):
        return mat2.contiguous()
    return mat2


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_THEAD ADDMM")
    if _can_use_native_fp32_addmm(bias, mat1, mat2):
        return _NATIVE_ADDMM_KERNEL.call_boxed(
            _NATIVE_ADDMM_KEYSET,
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
        )
    return common_addmm(bias, mat1, _prepare_mat2(mat1, mat2), beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_THEAD ADDMM_OUT")
    if (
        _NATIVE_ADDMM_OUT_KERNEL is not None
        and out is not None
        and not out.requires_grad
        and out.dtype == torch.float32
        and out.device == mat1.device
        and _can_use_native_fp32_addmm(bias, mat1, mat2)
    ):
        return _NATIVE_ADDMM_OUT_KERNEL.call_boxed(
            _NATIVE_ADDMM_KEYSET,
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=out,
        )
    return common_addmm_out(
        bias,
        mat1,
        _prepare_mat2(mat1, mat2),
        beta=beta,
        alpha=alpha,
        out=out,
    )


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS_THEAD ADDMM_DTYPE")
    return common_addmm_dtype(
        bias,
        mat1,
        _prepare_mat2(mat1, mat2),
        out_dtype,
        beta=beta,
        alpha=alpha,
    )


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS_THEAD ADDMM_DTYPE_OUT")
    return common_addmm_dtype_out(
        bias,
        mat1,
        _prepare_mat2(mat1, mat2),
        out_dtype,
        beta=beta,
        alpha=alpha,
        out=out,
    )
