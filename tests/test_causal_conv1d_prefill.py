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
# (tutorials/tle/02-causal-conv1d-prefill.py, itself in sync with
# vllm-ascend's causal_conv1d). When vllm-ascend (>= 0.17) is installed,
# additionally cross-check against its PyTorch golden
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
    BATCH_SIZES = [4]
    DIMS = [64]
    WIDTHS = [4]
    SEQLENS = [16]
    WITH_PADDINGS = [True]
else:
    BATCH_SIZES = [4, 8]
    DIMS = [64, 4096]
    WIDTHS = [4]
    SEQLENS = [16, 249]
    WITH_PADDINGS = [True, False]


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
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


def vllm_ascend_causal_conv1d_fn(
    x, weight, bias, activation, conv_states, query_start_loc, cache_indices,
    has_initial_state, pad_slot_id
):
    """Run vllm-ascend's PyTorch golden on the same NPU inputs as the kernel."""
    import vllm_ascend  # noqa: F401
    from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_fn

    return causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        activation=activation,
        conv_states=conv_states,
        has_initial_state=has_initial_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        pad_slot_id=pad_slot_id,
    )


def _varlen_ref(x_ref, seqlens, padded_state_indices, weight_ref, bias_ref, activation,
                final_states_ref, has_initial_states):
    """Split the varlen tensor back into per-sequence slices and run the reference
    conv1d on each, mirroring how the kernel iterates over sequences."""
    out_ref_b = []
    # x_ref: (1, dim, seqlen) -> split along seqlen
    splits = torch.split(x_ref[0], seqlens[0], dim=-1)
    for i in range(len(seqlens[0])):
        cache_idx = padded_state_indices[i].long().item()
        if cache_idx == PAD_SLOT_ID:
            continue
        x_s = splits[i].unsqueeze(0)  # (1, dim, s_i)
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight_ref,
                bias_ref,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[cache_idx].unsqueeze(0),
                initial_states=final_states_ref[cache_idx].unsqueeze(0)
                if bool(has_initial_states[i]) else None,
            ))
    # concat per-seq outputs along seqlen
    return torch.cat([t[0] for t in out_ref_b], dim=2)  # (1, dim, sum)


@pytest.mark.skipif(
    not hasattr(flag_gems, "causal_conv1d_prefill"),
    reason="causal_conv1d_prefill is only implemented on the ascend backend",
)
@pytest.mark.causal_conv1d_prefill
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("with_padding", WITH_PADDINGS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seqlen", SEQLENS)
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_causal_conv1d_prefill(
    batch_size, with_padding, dim, seqlen, width, has_bias, dtype
):
    device = flag_gems.device
    activation = "silu"

    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    # build random per-sequence lengths on cpu that sum to `seqlen`
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seq = torch.diff(
        torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
    ).tolist()
    assert sum(seq) == seqlen
    assert all(s > 0 for s in seq)

    cumsum = torch.cumsum(torch.tensor(seq), dim=0).to(torch.int32)
    cumsum = torch.cat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)

    total_entries = batch_size * 10

    # x: (1, dim, seqlen); the wrapper receives x.squeeze(0) -> (dim, seqlen)
    x = torch.randn(1, dim, seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    # conv_states: (total_entries, dim, width - 1), updated in place by the kernel
    final_states = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    has_initial_states = torch.randint(
        0, 2, (cumsum.shape[0] - 1,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[:batch_size]
    padded_state_indices = torch.cat(
        [
            state_indices,
            torch.full((padding,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ]
    )

    # CPU references (upcast) -- must clone final_states before the in-place gem update
    x_ref = utils.to_reference(x, True)
    weight_ref = utils.to_reference(weight, True)
    bias_ref = utils.to_reference(bias, True)
    final_states_ref = utils.to_reference(final_states.detach().clone(), True)
    has_initial_states_ref = utils.to_reference(has_initial_states, True)
    padded_state_indices_ref = utils.to_reference(padded_state_indices, True)

    # snapshot the in-place kernel input (conv_states is overwritten) for the
    # optional vllm-ascend golden run below
    if enable_vllm:
        final_states_va = final_states.detach().clone()

    out = flag_gems.causal_conv1d_prefill(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=final_states,
        query_start_loc=cumsum.to(device),
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_states,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )

    out_ref = _varlen_ref(
        x_ref, [seq], padded_state_indices_ref, weight_ref, bias_ref, activation,
        final_states_ref, has_initial_states_ref,
    )

    # conv_states updated in place -> compare the real (non-padded) cache lines.
    # final_states_ref may live on CPU under --ref cpu, so its index must match
    # that device (indexing a CPU tensor with an NPU index raises on torch_npu).
    state_indices_long = state_indices.long()
    utils.gems_assert_close(
        final_states[state_indices_long],
        final_states_ref[state_indices_long.to(final_states_ref.device)],
        dtype,
        atol=5e-2,
    )

    # out: (dim, seqlen); out_ref: (1, dim, sum_real) -> squeeze to (dim, sum_real)
    unpadded_out = out[:, : out_ref.shape[-1]]
    utils.gems_assert_close(unpadded_out, out_ref.squeeze(0), dtype, atol=5e-2)

    if enable_vllm:
        out_va = vllm_ascend_causal_conv1d_fn(
            x.squeeze(0),
            weight,
            bias,
            activation,
            final_states_va,
            cumsum.to(device),
            padded_state_indices,
            has_initial_states,
            PAD_SLOT_ID,
        )
        utils.gems_assert_close(
            final_states_va[state_indices_long],
            final_states_ref[state_indices_long.to(final_states_ref.device)],
            dtype,
            atol=5e-2,
        )
        utils.gems_assert_close(
            out_va[:, : out_ref.shape[-1]], out_ref.squeeze(0), dtype, atol=5e-2
        )
