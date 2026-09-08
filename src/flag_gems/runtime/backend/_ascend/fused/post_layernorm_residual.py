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
import math

import torch
import triton
import triton.language as tl

from flag_gems.fused.post_layernorm_residual import (
    post_layer_norm_residual as common_post_layer_norm_residual,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def post_layer_norm_residual_kernel(
    x_ptr,
    residual_ptr,
    output_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    rows = ext.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    cols = tl.arange(0, TILE_N)[None, :]
    mask = (rows[:, None] < M) & (cols < N)
    offsets = rows[:, None] * N + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    centered = x - mean[:, None]
    variance = tl.sum(tl.where(cols < N, centered * centered, 0.0), axis=1) / N
    rstd = tl.math.rsqrt(variance + eps)
    normalized = centered * rstd[:, None]
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        normalized = normalized * weight
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        normalized = normalized + bias
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = normalized + residual
    tl.store(output_ptr + offsets, output, mask=mask)


def _can_use_fast_path(x, residual, normalized_shape, weight=None, bias=None):
    # The fused kernel has no backward implementation. Keep grad-enabled calls
    # on the composition even when the current inputs do not require gradients.
    if torch.is_grad_enabled():
        return False
    if (
        x.device.type != "npu"
        or x.shape != residual.shape
        or x.dtype != residual.dtype
        or x.device != residual.device
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or x.numel() == 0
    ):
        return False
    if not isinstance(normalized_shape, (tuple, list, torch.Size)):
        return False
    normalized_shape = tuple(normalized_shape)
    if (
        not normalized_shape
        or len(normalized_shape) > x.ndim
        or any(not isinstance(size, int) or size <= 0 for size in normalized_shape)
        or tuple(x.shape[-len(normalized_shape) :]) != normalized_shape
        or math.prod(normalized_shape) > 4096
    ):
        return False
    return all(
        tensor is None
        or (
            getattr(tensor, "shape", None) == normalized_shape
            and tensor.dtype == x.dtype
            and tensor.device == x.device
            and tensor.is_contiguous()
        )
        for tensor in (weight, bias)
    )


def post_layer_norm_residual(
    x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    if not _can_use_fast_path(x, residual, normalized_shape, weight, bias):
        return common_post_layer_norm_residual(
            x, residual, normalized_shape, weight, bias, eps
        )

    N = math.prod(normalized_shape)
    M = x.numel() // N
    tile_n = triton.next_power_of_2(N)
    # Bound a tile to 4096 elements, following the Ascend one-pass LayerNorm.
    tile_m = max(1, min(8, 4096 // tile_n)) if M >= 256 else 1
    output = torch.empty_like(x)
    try:
        with torch_device_fn.device(x.device):
            post_layer_norm_residual_kernel[(triton.cdiv(M, tile_m), 1, 1)](
                x,
                residual,
                output,
                weight,
                bias,
                M,
                N,
                eps,
                TILE_M=tile_m,
                TILE_N=tile_n,
            )
    except Exception:
        logger.warning(
            "Ascend post-LayerNorm residual launch failed; using composition",
            exc_info=True,
        )
        return common_post_layer_norm_residual(
            x, residual, normalized_shape, weight, bias, eps
        )
    return output
