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
COMMON = ROOT / "src/flag_gems/fused/add_add_silu.py"
ASCEND = ROOT / "src/flag_gems/runtime/backend/_ascend/fused/add_add_silu.py"
FUSED_INIT = ROOT / "src/flag_gems/fused/__init__.py"
ASCEND_INIT = ROOT / "src/flag_gems/runtime/backend/_ascend/fused/__init__.py"


def test_add_add_silu_public_and_ascend_routes_exist():
    assert COMMON.is_file()
    assert ASCEND.is_file()
    assert (
        "from flag_gems.fused.add_add_silu import add_add_silu"
        in FUSED_INIT.read_text()
    )
    assert "from .add_add_silu import add_add_silu" in ASCEND_INIT.read_text()


def test_common_fallback_stays_inside_flaggems_owned_triton_ops():
    source = COMMON.read_text()

    assert "from flag_gems.ops.add import add" in source
    assert "from flag_gems.ops.silu import silu" in source
    assert "silu(add(addend, residual_a), residual_b)" in source
    assert "torch.add" not in source
    assert "torch.nn.functional" not in source


def test_ascend_fast_path_is_a_single_triton_kernel_without_native_redispatch():
    source = ASCEND.read_text()

    assert "@triton.jit" in source
    assert "def _add_add_silu_kernel(" in source
    assert "tl.exp" in source
    assert "_common_add_add_silu" in source
    for forbidden in (
        "torch.add",
        "torch.ops.aten",
        "torch_npu",
        "redispatch",
        "_ExcludeDispatchKeyGuard",
    ):
        assert forbidden not in source


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend-only fusion")
def test_ascend_add_add_silu_matches_materialized_fp32_composition():
    shape = (2, 131072, 512)
    addend = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    residual_a = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)
    residual_b = torch.randn(shape, device=flag_gems.device, dtype=torch.float32)

    first = addend + residual_a
    reference = torch.nn.functional.silu(first + residual_b)
    actual = flag_gems.add_add_silu(addend, residual_a, residual_b)

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
