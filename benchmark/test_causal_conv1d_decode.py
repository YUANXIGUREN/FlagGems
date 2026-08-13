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

from . import base

# (batch, dim, width, seqlen); decode step has a small seqlen (1-3).
# Mirrors the accuracy cases in tests/test_causal_conv1d_decode.py.
DECODE_SHAPES = [
    (64, 4096, 4, 1),
    (64, 2048 + 16, 3, 1),
    (64, 4096, 4, 3),
]


def causal_conv1d_decode_ref(x, conv_state, weight, bias=None, activation=None, cache_seqlens=None, **_):
    """Grouped F.conv1d baseline kept verbatim-consistent with the golden source
    (tutorials/tle/03-causal-conv1d-decode.py), running on the same device as
    the kernel.

    x: (batch, dim) or (batch, dim, seqlen); conv_state: (batch, dim, state_len),
    state_len >= width - 1, updated in place.
    Extra keyword arguments (conv_state_indices / pad_slot_id) from the gems op
    are ignored so the same (args, kwargs) feed both the baseline and the kernel.
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


class CausalConv1dDecodeBenchmark(base.GenericBenchmark):
    """causal_conv1d_decode has no generic M/N shape; iterate over its own configs."""

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in DECODE_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def causal_conv1d_decode_kwargs(shape, dtype, device):
    batch, dim, width, seqlen = shape
    # x: (batch, dim, seqlen)
    x = torch.randn(batch, seqlen, dim, device=device, dtype=dtype).transpose(1, 2)
    # dense conv_state cache (no paging): identity indices map slot i -> i.
    conv_state = torch.randn(batch, width - 1, dim, device=device, dtype=dtype).transpose(1, 2)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_state_indices = torch.arange(batch, dtype=torch.int32, device=device)

    kwargs = {
        "activation": "silu",
        "conv_state_indices": conv_state_indices,
        "pad_slot_id": -1,
    }
    yield x, conv_state, weight, bias, kwargs


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="causal_conv1d_decode is only implemented on the ascend backend",
)
@pytest.mark.skipif(
    not hasattr(flag_gems, "causal_conv1d_decode"),
    reason="flag_gems.causal_conv1d_decode is unavailable",
)
@pytest.mark.causal_conv1d_decode
def test_causal_conv1d_decode():
    bench = CausalConv1dDecodeBenchmark(
        op_name="causal_conv1d_decode",
        input_fn=causal_conv1d_decode_kwargs,
        torch_op=causal_conv1d_decode_ref,
        gems_op=flag_gems.causal_conv1d_decode,
        dtypes=[torch.bfloat16],
    )
    bench.run()
