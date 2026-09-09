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
COMMON = ROOT / "src/flag_gems/fused/post_layernorm_residual.py"
MTHREADS = (
    ROOT
    / "src/flag_gems/runtime/backend/_mthreads/fused/post_layernorm_residual.py"
)
PUBLIC_INIT = ROOT / "src/flag_gems/fused/__init__.py"
MTHREADS_INIT = ROOT / "src/flag_gems/runtime/backend/_mthreads/fused/__init__.py"


def test_post_layernorm_residual_composition_uses_flaggems_ops_only():
    source = COMMON.read_text()

    assert "from flag_gems.ops.add import add" in source
    assert "from flag_gems.ops.layernorm import layer_norm" in source
    assert "torch." not in source
    assert "post_layer_norm_residual" in PUBLIC_INIT.read_text()


def test_mthreads_post_layernorm_residual_is_one_owned_triton_kernel():
    source = MTHREADS.read_text()

    assert "def _post_layer_norm_residual_kernel(" in source
    assert "tl.math.rsqrt" in source
    assert "normalized + residual" in source
    assert "from .post_layernorm_residual import post_layer_norm_residual" in (
        MTHREADS_INIT.read_text()
    )
    for forbidden in (
        "torch.add",
        "torch.layer_norm",
        "torch.ops.aten",
        "torch_musa",
        "redispatch",
    ):
        assert forbidden not in source


@pytest.mark.skipif(flag_gems.vendor_name != "mthreads", reason="MThreads only")
@pytest.mark.parametrize("rows", [40962, 131072])
def test_mthreads_post_layernorm_residual_matches_materialized_sequence(rows):
    width = 512
    x = torch.randn((rows, width), device=flag_gems.device, dtype=torch.float32)
    residual = torch.randn_like(x)
    weight = torch.randn((width,), device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn((width,), device=flag_gems.device, dtype=torch.float32)

    normalized = flag_gems.layer_norm(x, (width,), weight, bias, 1e-5)[0]
    expected = flag_gems.add(normalized, residual)
    actual = flag_gems.post_layer_norm_residual(
        x, residual, (width,), weight, bias, 1e-5
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
