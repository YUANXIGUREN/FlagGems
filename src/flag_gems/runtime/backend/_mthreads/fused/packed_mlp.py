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

from flag_gems.fused.add_add_silu_addmm import (
    add_add_silu_addmm as _common_add_add_silu_addmm,
)
from flag_gems.fused.silu_addmm import silu_addmm as _common_silu_addmm
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._mthreads.ops.addmm import (
    _get_tf32_fp16_rhs,
    addmm_sqmma,
    can_use_mthreads_tf32_sqmma_contract,
)
from flag_gems.runtime.matmul_precision import should_use_fast_float32_matmul
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _silu_to_tf32_fp16_kernel(
    input,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
    activated = value / (1.0 + tl.exp(-value))
    bits = activated.to(tl.uint32, bitcast=True)
    packed = (bits & 0xFFFFE000).to(tl.float32, bitcast=True)
    tl.store(output + offsets, packed, mask=mask)


@libentry()
@triton.jit
def _add_add_silu_to_tf32_fp16_kernel(
    addend,
    residual_a,
    residual_b,
    output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(addend + offsets, mask=mask, other=0.0)
    value = value + tl.load(residual_a + offsets, mask=mask, other=0.0)
    value = value + tl.load(residual_b + offsets, mask=mask, other=0.0)
    value = value.to(tl.float32)
    activated = value / (1.0 + tl.exp(-value))
    bits = activated.to(tl.uint32, bitcast=True)
    packed = (bits & 0xFFFFE000).to(tl.float32, bitcast=True)
    tl.store(output + offsets, packed, mask=mask)


def _eligible(input, bias, mat2, residuals=()):
    tensors = (input, bias, mat2, *residuals)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        return False
    if input.ndim != 2 or mat2.ndim != 2:
        return False
    if input.shape[1] != mat2.shape[0]:
        return False
    M, K = input.shape
    _, N = mat2.shape
    if bias.ndim != 1 or bias.shape[0] != N:
        return False
    if any(item.shape != input.shape for item in residuals):
        return False
    if any(item.device != input.device for item in tensors):
        return False
    if any(item.dtype != torch.float32 for item in tensors):
        return False
    if not input.is_contiguous() or any(
        not item.is_contiguous() for item in residuals
    ):
        return False
    if torch.is_grad_enabled() and any(item.requires_grad for item in tensors):
        return False
    fast_enabled = should_use_fast_float32_matmul("mthreads", input, mat2)
    return can_use_mthreads_tf32_sqmma_contract(
        True,
        fast_enabled,
        False,
        M,
        N,
        K,
    )


def _launch_pack(input, residuals=()):
    packed = torch.empty(
        tuple(input.shape),
        dtype=torch.float16,
        device=input.device,
    )
    n_elements = input.numel()
    if n_elements == 0:
        return packed
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(input.device):
        if residuals:
            _add_add_silu_to_tf32_fp16_kernel[grid](
                input,
                residuals[0],
                residuals[1],
                packed,
                n_elements,
                BLOCK_SIZE=block_size,
            )
        else:
            _silu_to_tf32_fp16_kernel[grid](
                input,
                packed,
                n_elements,
                BLOCK_SIZE=block_size,
            )
    return packed


def _packed_addmm(input, bias, mat2, beta, alpha, residuals=()):
    M, K = input.shape
    _, N = mat2.shape
    packed_input = _launch_pack(input, residuals)
    packed_rhs = _get_tf32_fp16_rhs(mat2)
    output = torch.empty((M, N), dtype=torch.float32, device=input.device)
    return addmm_sqmma(
        packed_input,
        packed_rhs,
        bias,
        torch.float16,
        alpha,
        beta,
        M,
        N,
        K,
        out=output,
    )


def silu_addmm(input, bias, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS SILU_ADDMM_PACKED")
    if not _eligible(input, bias, mat2):
        return _common_silu_addmm(input, bias, mat2, beta=beta, alpha=alpha)
    return _packed_addmm(input, bias, mat2, beta, alpha)


def add_add_silu_addmm(
    addend,
    residual_a,
    residual_b,
    bias,
    mat2,
    *,
    beta=1,
    alpha=1,
):
    logger.debug("GEMS_MTHREADS ADD_ADD_SILU_ADDMM_PACKED")
    residuals = (residual_a, residual_b)
    if not _eligible(addend, bias, mat2, residuals):
        return _common_add_add_silu_addmm(
            addend,
            residual_a,
            residual_b,
            bias,
            mat2,
            beta=beta,
            alpha=alpha,
        )
    return _packed_addmm(addend, bias, mat2, beta, alpha, residuals)


__all__ = ["add_add_silu_addmm", "silu_addmm"]
