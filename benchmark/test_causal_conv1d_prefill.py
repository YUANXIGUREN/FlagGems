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

# (batch, dim, seqlen, width); seqlen is split into `batch` varlen segments.
# Mirrors the accuracy cases in tests/test_causal_conv1d_prefill.py.
PREFILL_SHAPES = [
    (4, 4096, 1024, 4),
    (8, 4096, 2048, 4),
    (4, 2048 + 16, 2048, 3),
]


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """Grouped F.conv1d baseline kept verbatim-consistent with the golden source
    (tutorials/tle/02-causal-conv1d-prefill.py), running on the same device as
    the kernel."""
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def _varlen_ref(
    x_ref, seqlens, padded_state_indices, weight, bias, activation,
    final_states_ref, has_initial_states,
):
    """Split the varlen tensor back into per-sequence slices and run the reference
    conv1d on each, mirroring how the kernel iterates over sequences."""
    out_ref_b = []
    splits = torch.split(x_ref[0], seqlens[0], dim=-1)
    for i in range(len(seqlens[0])):
        cache_idx = padded_state_indices[i].long().item()
        if cache_idx == -1:
            continue
        x_s = splits[i].unsqueeze(0)
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[cache_idx].unsqueeze(0),
                initial_states=final_states_ref[cache_idx].unsqueeze(0)
                if bool(has_initial_states[i]) else None,
            )
        )
    return torch.cat([t[0] for t in out_ref_b], dim=2)


def causal_conv1d_prefill_ref(
    x_sq,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    activation=None,
    pad_slot_id=-1,
    **_,
):
    """Baseline wrapper sharing the kernel's (args, kwargs).

    x_sq: (dim, seqlen); conv_states: (cache_lines, dim, width - 1), updated in place.
    """
    x_ref = x_sq.unsqueeze(0)
    seqlens = torch.diff(query_start_loc).tolist()
    return _varlen_ref(
        x_ref, [seqlens], cache_indices, weight, bias, activation,
        conv_states, has_initial_state,
    )


class CausalConv1dPrefillBenchmark(base.GenericBenchmark):
    """causal_conv1d_prefill has no generic M/N shape; iterate over its own configs."""

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in PREFILL_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def causal_conv1d_prefill_kwargs(shape, dtype, device):
    batch, dim, seqlen, width = shape
    # split seqlen into `batch` equal varlen segments
    seg = seqlen // batch
    query_start_loc = torch.tensor(
        [seg * i for i in range(batch + 1)], dtype=torch.int32, device=device
    )

    # x: (1, dim, seqlen); the wrapper receives x.squeeze(0) -> (dim, seqlen)
    x = torch.randn(1, dim, seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)

    # conv_states: (batch, dim, width - 1), updated in place; no paging.
    conv_states = torch.randn(batch, width - 1, dim, device=device, dtype=dtype).transpose(1, 2)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(batch, dtype=torch.bool, device=device)

    kwargs = {"activation": "silu", "pad_slot_id": -1}
    yield (
        x.squeeze(0),
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        kwargs,
    )


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="causal_conv1d_prefill is only implemented on the ascend backend",
)
@pytest.mark.skipif(
    not hasattr(flag_gems, "causal_conv1d_prefill"),
    reason="flag_gems.causal_conv1d_prefill is unavailable",
)
@pytest.mark.causal_conv1d_prefill
def test_causal_conv1d_prefill():
    bench = CausalConv1dPrefillBenchmark(
        op_name="causal_conv1d_prefill",
        input_fn=causal_conv1d_prefill_kwargs,
        torch_op=causal_conv1d_prefill_ref,
        gems_op=flag_gems.causal_conv1d_prefill,
        dtypes=[torch.bfloat16],
    )
    bench.run()
