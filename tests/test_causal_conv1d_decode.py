# Copyright 2026- Xcoresigma Technology Co., Ltd
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

import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

PAD_SLOT_ID = -1

# Keep the reference below verbatim-consistent with the golden source
# (tutorials/tle/03-causal-conv1d-decode.py, itself in sync with
# vllm-ascend's causal_conv1d_update). When vllm-ascend (>= 0.17) is
# installed, additionally cross-check against its PyTorch golden
# (vllm_ascend._310p.ops.causal_conv1d), the same one its nightly tests use.
# See tutorials/tle/08-add-rms-norm-bias.py for the guard pattern.
enable_vllm = True
try:
    import vllm  # noqa: F401
    if vllm.__version__ < "0.17.0":
        enable_vllm = False
except ImportError:
    enable_vllm = False

if QUICK_MODE:
    BATCH_SIZES = [3]
    DIMS = [2048 + 16]
    WIDTHS = [3]
    SEQLENS = [1]
    WITH_PADDINGS = [False]
else:
    BATCH_SIZES = [3, 64]
    DIMS = [2048 + 16, 4096]
    WIDTHS = [3, 4]
    SEQLENS = [1, 3]
    WITH_PADDINGS = [True, False]


def causal_conv1d_decode_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(-(width - 1), 0, dtype=torch.long,
                                 device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1))
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def vllm_ascend_causal_conv1d_update(
    x, conv_state, weight, bias, activation, conv_state_indices, pad_slot_id
):
    """Run vllm-ascend's PyTorch golden on the same NPU inputs as the kernel."""
    import vllm_ascend  # noqa: F401
    from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_update

    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        pad_slot_id=pad_slot_id,
    )


@pytest.mark.skipif(
    not hasattr(flag_gems, "causal_conv1d_decode"),
    reason="causal_conv1d_decode is only implemented on the ascend backend",
)
@pytest.mark.causal_conv1d_decode
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("with_padding", WITH_PADDINGS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_causal_conv1d_decode(
    batch_size, with_padding, dim, width, seqlen, has_bias, dtype
):
    device = flag_gems.device
    activation = "silu"

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    total_entries = 10 * batch_size

    # x is (batch, dim, seqlen), contiguous along the dim axis
    x = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    padded_state_indices = torch.cat(
        [
            conv_state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )

    # conv_state is (cache_lines, dim, state_len), contiguous along the dim axis
    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    x_ref = utils.to_reference(x, True)
    weight_ref = utils.to_reference(weight, True)
    bias_ref = utils.to_reference(bias, True)
    conv_state_ref = utils.to_reference(
        conv_state[conv_state_indices].detach().clone(), True
    )

    # snapshot the in-place kernel inputs (x and conv_state are overwritten)
    # for the optional vllm-ascend golden run below
    if enable_vllm:
        x_va = x.detach().clone()
        conv_state_va = conv_state.detach().clone()

    out = flag_gems.causal_conv1d_decode(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )

    out_ref = causal_conv1d_decode_ref(
        x_ref[:batch_size], conv_state_ref, weight_ref, bias_ref, activation=activation
    )

    utils.gems_assert_close(out[:batch_size], out_ref, dtype, atol=5e-2)
    utils.gems_assert_close(
        conv_state[conv_state_indices], conv_state_ref, dtype, atol=5e-2
    )

    if enable_vllm:
        out_va = vllm_ascend_causal_conv1d_update(
            x_va,
            conv_state_va,
            weight,
            bias,
            activation,
            padded_state_indices,
            PAD_SLOT_ID,
        )
        utils.gems_assert_close(
            out_va[:batch_size].to(out_ref.device), out_ref, dtype, atol=5e-2
        )
        utils.gems_assert_close(
            conv_state_va[conv_state_indices].to(out_ref.device),
            conv_state_ref,
            dtype,
            atol=5e-2,
        )
