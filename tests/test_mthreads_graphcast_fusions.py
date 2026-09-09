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
ADDMM = ROOT / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py"
ADD_ADD_SILU = (
    ROOT / "src/flag_gems/runtime/backend/_mthreads/fused/add_add_silu.py"
)
FUSED_INIT = ROOT / "src/flag_gems/runtime/backend/_mthreads/fused/__init__.py"


def test_mthreads_rejects_slow_addmm_epilogue_fusion():
    source = ADDMM.read_text()

    assert "FUSE_SILU: tl.constexpr" not in source
    assert "def addmm_silu(" not in source
    assert "from .addmm_silu import addmm_silu" not in FUSED_INIT.read_text()


def test_mthreads_residual_activation_is_a_single_triton_kernel():
    source = ADD_ADD_SILU.read_text()

    assert "@triton.jit" in source
    assert "def _add_add_silu_kernel(" in source
    assert "tl.exp" in source
    assert "tl.fdiv(" in source
    assert "from .add_add_silu import add_add_silu" in FUSED_INIT.read_text()
    for forbidden in (
        "torch.add",
        "torch.ops.aten",
        "torch_musa",
        "redispatch",
        "_ExcludeDispatchKeyGuard",
    ):
        assert forbidden not in source


@pytest.mark.skipif(flag_gems.vendor_name != "mthreads", reason="MThreads only")
def test_mthreads_fusions_match_materialized_compositions():
    m, n, k = 40962, 512, 512
    mat1 = torch.randn((m, k), device=flag_gems.device, dtype=torch.float32)
    weight = torch.randn((n, k), device=flag_gems.device, dtype=torch.float32)
    mat2 = weight.t()
    bias = torch.randn((n,), device=flag_gems.device, dtype=torch.float32)
    residual_a = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)
    residual_b = torch.randn((m, n), device=flag_gems.device, dtype=torch.float32)

    addmm_actual = flag_gems.addmm(bias, mat1, mat2)
    first = addmm_actual + residual_a
    residual_reference = torch.nn.functional.silu(first + residual_b)
    residual_actual = flag_gems.add_add_silu(addmm_actual, residual_a, residual_b)
    torch.testing.assert_close(residual_actual, residual_reference, rtol=1e-5, atol=1e-6)
