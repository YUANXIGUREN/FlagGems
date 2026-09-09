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

import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.post_layernorm_residual import (
    post_layer_norm_residual as _common_post_layer_norm_residual,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit(do_not_specialize=["eps"])
def _post_layer_norm_residual_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    rows,
    width: tl.constexpr,
    eps,
    TILE_M: tl.constexpr,
):
    row_offsets = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    col_offsets = tl.arange(0, width)[None, :]
    row_mask = row_offsets < rows
    offsets = row_offsets[:, None] * width + col_offsets
    mask = row_mask[:, None]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / width
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / width
    rstd = tl.math.rsqrt(variance + eps)
    normalized = centered * rstd[:, None]

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + col_offsets).to(tl.float32)
        normalized *= weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_offsets).to(tl.float32)
        normalized += bias

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = normalized + residual
    tl.store(output_ptr + offsets, output, mask=mask)


def _can_use_graphcast_fp32_path(
    x, residual, normalized_shape, weight=None, bias=None
):
    if not isinstance(x, torch.Tensor) or not isinstance(residual, torch.Tensor):
        return False
    normalized_shape = tuple(normalized_shape)
    return (
        not torch.is_grad_enabled()
        and x.device.type == "musa"
        and x.ndim >= 2
        and x.shape == residual.shape
        and x.device == residual.device
        and x.dtype == residual.dtype == torch.float32
        and x.is_contiguous()
        and residual.is_contiguous()
        and normalized_shape == (512,)
        and x.shape[-1] == 512
        and x.numel() > 0
        and all(
            parameter is None
            or (
                isinstance(parameter, torch.Tensor)
                and parameter.shape == (512,)
                and parameter.device == x.device
                and parameter.dtype == torch.float32
                and parameter.is_contiguous()
            )
            for parameter in (weight, bias)
        )
    )


def post_layer_norm_residual(
    x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    """Fuse the GraphCast FP32 LayerNorm/residual chain on MThreads."""

    if not _can_use_graphcast_fp32_path(x, residual, normalized_shape, weight, bias):
        return _common_post_layer_norm_residual(
            x, residual, normalized_shape, weight, bias, eps
        )

    width = math.prod(normalized_shape)
    rows = x.numel() // width
    output = torch.empty_like(x)
    tile_m = 1
    with torch_device_fn.device(x.device):
        _post_layer_norm_residual_kernel[(triton.cdiv(rows, tile_m),)](
            x,
            residual,
            output,
            weight,
            bias,
            rows,
            width,
            eps,
            TILE_M=tile_m,
        )
    return output


__all__ = ["post_layer_norm_residual"]
