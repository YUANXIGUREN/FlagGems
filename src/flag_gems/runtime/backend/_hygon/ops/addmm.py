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

"""Hygon AddMM inference routing.

GraphCast uses FP32 AddMM with a one-dimensional bias.  On Hygon, the native
BLAS implementation is substantially faster than the generic Triton kernel.
Capture the CUDA-dispatch implementation before FlagGems registers its own
kernel, then use it only for the validated, inference-safe subset.  All other
inputs continue to use the upstream generic implementation.
"""

import logging

import torch

from flag_gems.ops.addmm import addmm as _triton_addmm
from flag_gems.ops.addmm import addmm_out as _triton_addmm_out


logger = logging.getLogger(__name__)


def _load_native_addmm_kernels():
    """Capture Hygon CUDA kernels before FlagGems replaces CUDA dispatch."""

    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    try:
        default_kernel = torch.library.get_kernel("aten::addmm", "CUDA")
        out_kernel = torch.library.get_kernel("aten::addmm.out", "CUDA")
    except (AttributeError, RuntimeError):
        return None, None, None
    return default_kernel, out_kernel, keyset


(
    _NATIVE_ADDMM_KERNEL,
    _NATIVE_ADDMM_OUT_KERNEL,
    _NATIVE_ADDMM_KEYSET,
) = _load_native_addmm_kernels()

_NATIVE_ADDMM_MODE = (
    "captured_cuda" if _NATIVE_ADDMM_KERNEL is not None else "unavailable"
)


def _can_use_native_fp32_addmm(bias, mat1, mat2):
    """Return whether the call is in the validated GraphCast inference subset."""

    if _NATIVE_ADDMM_KERNEL is None:
        return False
    if torch.is_grad_enabled() and (
        bias.requires_grad or mat1.requires_grad or mat2.requires_grad
    ):
        return False
    if bias.dtype != torch.float32 or mat1.dtype != torch.float32:
        return False
    if mat2.dtype != torch.float32 or mat1.dim() != 2 or mat2.dim() != 2:
        return False
    if mat1.device.type != "cuda" or mat2.device != mat1.device:
        return False
    if bias.device != mat1.device or mat2.shape[0] != mat1.shape[1]:
        return False
    return bias.dim() == 1 and bias.shape[0] == mat2.shape[1]


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    if _can_use_native_fp32_addmm(bias, mat1, mat2):
        logger.debug("GEMS ADDMM")
        return _NATIVE_ADDMM_KERNEL.call_boxed(
            _NATIVE_ADDMM_KEYSET,
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
        )
    return _triton_addmm(bias, mat1, mat2, beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    if (
        _NATIVE_ADDMM_OUT_KERNEL is not None
        and out is not None
        and not out.requires_grad
        and out.dtype == torch.float32
        and out.device == mat1.device
        and out.shape == (mat1.shape[0], mat2.shape[1])
        and _can_use_native_fp32_addmm(bias, mat1, mat2)
    ):
        logger.debug("GEMS ADDMM_OUT")
        return _NATIVE_ADDMM_OUT_KERNEL.call_boxed(
            _NATIVE_ADDMM_KEYSET,
            bias,
            mat1,
            mat2,
            beta=beta,
            alpha=alpha,
            out=out,
        )
    return _triton_addmm_out(
        bias,
        mat1,
        mat2,
        beta=beta,
        alpha=alpha,
        out=out,
    )
