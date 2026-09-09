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

import importlib.util
from pathlib import Path

import pytest
import torch

import flag_gems


def _ascend_add_module():
    module_path = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_ascend/ops/add.py"
    )
    spec = importlib.util.spec_from_file_location("_test_ascend_add", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_add_redispatch_preserves_alpha_semantics():
    add_module = _ascend_add_module()
    lhs = torch.tensor([1.0, -2.0], dtype=torch.float32)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = add_module._native_add(lhs, rhs, alpha=2)

    torch.testing.assert_close(result, torch.tensor([7.0, 6.0]))


def test_native_add_route_rejects_non_npu_inputs():
    add_module = _ascend_add_module()
    lhs = torch.ones((2, 4), dtype=torch.float32)
    rhs = torch.ones((2, 4), dtype=torch.float32)

    assert add_module._can_use_native_fp32_add(lhs, rhs) is False


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only Native add routing"
)
@pytest.mark.parametrize("shape", [(1, 257, 512), (1, 513, 512)])
@pytest.mark.parametrize("alpha", [1, 2])
def test_ascend_public_add_uses_native_route_for_inference(shape, alpha):
    add_module = importlib.import_module(flag_gems.add.__module__)
    lhs = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    rhs = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    expected = torch.add(lhs, rhs, alpha=alpha)

    assert flag_gems.add is add_module.add
    assert add_module._can_use_native_fp32_add(lhs, rhs) is True
    with torch.no_grad(), flag_gems.use_gems():
        actual = torch.add(lhs, rhs, alpha=alpha)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only Native add routing"
)
@pytest.mark.parametrize("case", ["broadcast", "noncontiguous", "float16"])
def test_ascend_public_add_preserves_common_fallbacks(case):
    add_module = importlib.import_module(flag_gems.add.__module__)
    if case == "broadcast":
        lhs = torch.randn((2, 3, 8), dtype=torch.float32, device=flag_gems.device)
        rhs = torch.randn((1, 3, 1), dtype=torch.float32, device=flag_gems.device)
    elif case == "noncontiguous":
        lhs = torch.randn((2, 3, 8), dtype=torch.float32, device=flag_gems.device)
        rhs = torch.randn((2, 8, 3), dtype=torch.float32, device=flag_gems.device)
        rhs = rhs.transpose(1, 2)
    else:
        lhs = torch.randn((2, 3, 8), dtype=torch.float16, device=flag_gems.device)
        rhs = torch.randn_like(lhs)
    expected = torch.add(lhs, rhs)

    assert add_module._can_use_native_fp32_add(lhs, rhs) is False
    with flag_gems.use_gems():
        actual = torch.add(lhs, rhs)

    torch.testing.assert_close(actual, expected)
