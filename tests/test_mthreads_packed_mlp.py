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

from pathlib import Path

import pytest
import torch

import flag_gems


ROOT = Path(__file__).parents[1]
SILU_ADDMM = ROOT / "src/flag_gems/fused/silu_addmm.py"
ADD_ADD_SILU_ADDMM = ROOT / "src/flag_gems/fused/add_add_silu_addmm.py"
MTHREADS = ROOT / "src/flag_gems/runtime/backend/_mthreads/fused/packed_mlp.py"
FUSED_INIT = ROOT / "src/flag_gems/fused/__init__.py"
MTHREADS_INIT = ROOT / "src/flag_gems/runtime/backend/_mthreads/fused/__init__.py"


def test_public_packed_mlp_compositions_stay_inside_flaggems():
    assert SILU_ADDMM.is_file()
    assert ADD_ADD_SILU_ADDMM.is_file()
    source = SILU_ADDMM.read_text() + ADD_ADD_SILU_ADDMM.read_text()
    assert "from flag_gems.ops.addmm import addmm" in source
    assert "from flag_gems.ops.add import add" in source
    assert "from flag_gems.ops.silu import silu" in source
    assert "torch." not in source
    assert "silu_addmm" in FUSED_INIT.read_text()
    assert "add_add_silu_addmm" in FUSED_INIT.read_text()


def test_mthreads_packed_mlp_uses_only_owned_triton_kernels():
    source = MTHREADS.read_text()

    assert "def _silu_to_tf32_fp16_kernel(" in source
    assert "def _add_add_silu_to_tf32_fp16_kernel(" in source
    assert "ext.round_to_tf32(activated)" in source
    assert "bits & 0xFFFFE000" not in source
    assert source.count("tl.fdiv(") == 2
    assert "addmm_sqmma(" in source
    assert "addmm as _mthreads_addmm" in source
    assert "from flag_gems.ops.add import add" in source
    assert "from flag_gems.ops.silu import silu" in source
    assert "_common_silu_addmm" not in source
    assert "_common_add_add_silu_addmm" not in source
    assert "from .packed_mlp import" in MTHREADS_INIT.read_text()
    for forbidden in (
        "torch.add",
        "torch.addmm",
        "torch.ops.aten",
        "torch_musa",
        "redispatch",
    ):
        assert forbidden not in source


@pytest.mark.skipif(flag_gems.vendor_name != "mthreads", reason="MThreads only")
def test_mthreads_packed_mlp_matches_existing_fast_fp32_sequence():
    m, n, k = 40962, 512, 512
    hidden = torch.randn((m, k), device=flag_gems.device, dtype=torch.float32)
    residual_a = torch.randn_like(hidden)
    residual_b = torch.randn_like(hidden)
    weight = torch.randn((n, k), device=flag_gems.device, dtype=torch.float32)
    mat2 = weight.t()
    bias = torch.randn((n,), device=flag_gems.device, dtype=torch.float32)

    expected = flag_gems.addmm(bias, flag_gems.silu(hidden), mat2)
    actual = flag_gems.silu_addmm(hidden, bias, mat2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    residual_input = flag_gems.add(flag_gems.add(hidden, residual_a), residual_b)
    expected = flag_gems.addmm(bias, flag_gems.silu(residual_input), mat2)
    actual = flag_gems.add_add_silu_addmm(
        hidden,
        residual_a,
        residual_b,
        bias,
        mat2,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(flag_gems.vendor_name != "mthreads", reason="MThreads only")
def test_mthreads_packed_mlp_ineligible_shape_uses_mthreads_addmm_fallback():
    m, n, k = 512, 83, 512
    hidden = torch.randn((m, k), device=flag_gems.device, dtype=torch.float32)
    weight = torch.randn((n, k), device=flag_gems.device, dtype=torch.float32)
    mat2 = weight.t()
    bias = torch.randn((n,), device=flag_gems.device, dtype=torch.float32)

    expected = flag_gems.addmm(bias, flag_gems.silu(hidden), mat2)
    actual = flag_gems.silu_addmm(hidden, bias, mat2)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
