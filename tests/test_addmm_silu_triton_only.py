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
COMMON = ROOT / "src/flag_gems/fused/addmm_silu.py"
ASCEND = ROOT / "src/flag_gems/runtime/backend/_ascend/fused/addmm_silu.py"
FUSED_INIT = ROOT / "src/flag_gems/fused/__init__.py"
ASCEND_INIT = ROOT / "src/flag_gems/runtime/backend/_ascend/fused/__init__.py"


def test_addmm_silu_public_and_ascend_routes_exist():
    assert COMMON.is_file()
    assert ASCEND.is_file()
    assert "from flag_gems.fused.addmm_silu import addmm_silu" in FUSED_INIT.read_text()
    assert "from .addmm_silu import addmm_silu" in ASCEND_INIT.read_text()


def test_common_fallback_is_a_flaggems_owned_triton_composition():
    source = COMMON.read_text()

    assert "from flag_gems.ops.addmm import addmm" in source
    assert "from flag_gems.ops.silu import silu" in source
    assert "torch.addmm" not in source
    assert "torch.mm" not in source
    assert "torch.matmul" not in source


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend-only fusion")
def test_ascend_addmm_silu_matches_materialized_composition():
    m, n, k = 4096, 512, 184
    mat1 = torch.randn((m, k), device=flag_gems.device, dtype=torch.float32)
    storage = torch.randn((n, 1536), device=flag_gems.device, dtype=torch.float32)
    mat2 = storage[:, :k].t()
    bias = torch.randn((n,), device=flag_gems.device, dtype=torch.float32)

    reference = torch.nn.functional.silu(torch.addmm(bias, mat1, mat2))
    actual = flag_gems.addmm_silu(bias, mat1, mat2)

    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
