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

from flag_gems.ops.add import add as _common_add
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _collapsed_row_stride(shape, strides):
    """Return a constant physical row stride for a suffix-contiguous view."""

    if len(shape) < 2 or len(shape) != len(strides) or strides[-1] != 1:
        return None
    row_stride = strides[-2]
    if row_stride <= 0:
        return None
    expected = row_stride
    for dim in range(len(shape) - 2, -1, -1):
        if shape[dim] != 1 and strides[dim] != expected:
            return None
        expected *= shape[dim]
    return row_stride


def classify_ascend_add_layout(shape, strides_a, strides_b):
    if not shape:
        return "unsupported"

    expected = 1
    a_contiguous = True
    b_contiguous = True
    for dim in range(len(shape) - 1, -1, -1):
        if shape[dim] != 1:
            a_contiguous = a_contiguous and strides_a[dim] == expected
            b_contiguous = b_contiguous and strides_b[dim] == expected
        expected *= shape[dim]
    if a_contiguous and b_contiguous:
        return "contiguous"

    if shape[-1] > 32:
        return "unsupported"
    if _collapsed_row_stride(shape, strides_a) is None:
        return "unsupported"
    if _collapsed_row_stride(shape, strides_b) is None:
        return "unsupported"
    return "suffix_strided"


def select_ascend_add_route(
    tensor_pair,
    same_shape,
    is_fp32,
    grad_enabled,
    layout,
):
    if not tensor_pair or not same_shape or not is_fp32 or grad_enabled:
        return "common"
    if layout == "contiguous":
        return "flat"
    if layout == "suffix_strided":
        return "suffix_strided"
    return "common"


def _select_rows_per_program(suffix, n_rows):
    if suffix == 1:
        if n_rows < 256:
            return 8
        if n_rows < 8_192:
            return 32
        if n_rows <= 65_536:
            return 512
        return 8
    if suffix <= 16:
        if n_rows < 256:
            return 64
        if n_rows <= 65_536:
            return 32
        return 256
    return 2


def _select_flat_block_size(n_elements):
    if n_elements < 256:
        return 256
    if n_elements < 65_536:
        return 1024
    return 8192


def explain_ascend_add_route(A, B):
    tensor_pair = isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    if not tensor_pair:
        return "common"
    same_shape = A.shape == B.shape and A.device == B.device
    is_fp32 = A.dtype == torch.float32 and B.dtype == torch.float32
    grad_enabled = torch.is_grad_enabled() and (
        A.requires_grad or B.requires_grad
    )
    layout = (
        classify_ascend_add_layout(A.shape, A.stride(), B.stride())
        if same_shape
        else "unsupported"
    )
    return select_ascend_add_route(
        tensor_pair,
        same_shape,
        is_fp32,
        grad_enabled,
        layout,
    )


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def _flat_add_kernel(
    A,
    B,
    out,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(A + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, a + b * alpha, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha"])
def _suffix_add_kernel(
    A,
    B,
    out,
    alpha,
    n_rows,
    suffix,
    stride_a_row,
    stride_a_suffix,
    stride_b_row,
    stride_b_suffix,
    stride_out_row,
    stride_out_suffix,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SUFFIX: tl.constexpr,
):
    pid = ext.program_id(0)
    rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    columns = tl.arange(0, BLOCK_SUFFIX)
    mask = (rows < n_rows)[:, None] & (columns < suffix)[None, :]
    a_offsets = (
        rows[:, None] * stride_a_row
        + columns[None, :] * stride_a_suffix
    )
    b_offsets = (
        rows[:, None] * stride_b_row
        + columns[None, :] * stride_b_suffix
    )
    out_offsets = (
        rows[:, None] * stride_out_row
        + columns[None, :] * stride_out_suffix
    )
    a = tl.load(A + a_offsets, mask=mask, other=0.0)
    b = tl.load(B + b_offsets, mask=mask, other=0.0)
    tl.store(out + out_offsets, a + b * alpha, mask=mask)


def _launch_flat_add(A, B, alpha):
    out = torch.empty(A.shape, device=A.device, dtype=A.dtype)
    n_elements = A.numel()
    if n_elements == 0:
        return out
    block_size = _select_flat_block_size(n_elements)
    grid = (triton.cdiv(n_elements, block_size),)
    with torch_device_fn.device(A.device):
        _flat_add_kernel[grid](
            A,
            B,
            out,
            alpha,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    return out


def _launch_suffix_add(A, B, alpha):
    out = torch.empty(A.shape, device=A.device, dtype=A.dtype)
    suffix = A.shape[-1]
    n_rows = A.numel() // suffix
    if n_rows == 0:
        return out
    rows_per_program = _select_rows_per_program(suffix, n_rows)
    block_suffix = triton.next_power_of_2(suffix)
    stride_a_row = _collapsed_row_stride(A.shape, A.stride())
    stride_b_row = _collapsed_row_stride(B.shape, B.stride())
    grid = (triton.cdiv(n_rows, rows_per_program),)
    with torch_device_fn.device(A.device):
        _suffix_add_kernel[grid](
            A,
            B,
            out,
            alpha,
            n_rows,
            suffix,
            stride_a_row,
            A.stride(-1),
            stride_b_row,
            B.stride(-1),
            out.stride(-2),
            out.stride(-1),
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_SUFFIX=block_suffix,
        )
    return out


def add(A, B, *, alpha=1):
    route = explain_ascend_add_route(A, B)
    if route == "flat":
        logger.debug("GEMS_ASCEND ADD_FLAT")
        return _launch_flat_add(A, B, alpha)
    if route == "suffix_strided":
        logger.debug("GEMS_ASCEND ADD_SUFFIX")
        return _launch_suffix_add(A, B, alpha)
    return _common_add(A, B, alpha=alpha)
