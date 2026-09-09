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

import ast
import importlib
from pathlib import Path

import pytest
import torch

import flag_gems


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src/flag_gems/runtime/backend/_ascend/ops/add.py"
)
OPS_INIT_PATH = SOURCE_PATH.parent / "__init__.py"


def _load_pure_functions(*names):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    missing = set(names) - functions.keys()
    assert not missing, f"missing required functions: {sorted(missing)}"
    namespace = {}
    module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


@pytest.mark.parametrize(
    "shape,strides_a,strides_b,expected",
    [
        ((2, 31, 13), (403, 13, 1), (403, 13, 1), "contiguous"),
        ((2, 31, 13), (5456, 176, 1), (403, 13, 1), "suffix_strided"),
        ((2, 31, 1), (5456, 176, 1), (31, 1, 1), "suffix_strided"),
        ((31, 13), (176, 1), (176, 1), "suffix_strided"),
        ((2, 31, 33), (5456, 176, 1), (5456, 176, 1), "unsupported"),
        ((2, 31, 13), (6000, 176, 1), (403, 13, 1), "unsupported"),
        ((2, 31, 13), (5456, 176, 2), (403, 13, 1), "unsupported"),
    ],
)
def test_ascend_add_layout_classifier(
    shape, strides_a, strides_b, expected
):
    _, classify = _load_pure_functions(
        "_collapsed_row_stride", "classify_ascend_add_layout"
    )

    assert classify(shape, strides_a, strides_b) == expected


@pytest.mark.parametrize(
    "tensor_pair,same_shape,is_fp32,grad_enabled,layout,expected",
    [
        (True, True, True, False, "contiguous", "flat"),
        (True, True, True, False, "suffix_strided", "suffix_strided"),
        (True, True, True, True, "suffix_strided", "common"),
        (True, True, False, False, "suffix_strided", "common"),
        (True, False, True, False, "suffix_strided", "common"),
        (False, True, True, False, "contiguous", "common"),
        (True, True, True, False, "unsupported", "common"),
    ],
)
def test_ascend_add_route_is_model_independent(
    tensor_pair, same_shape, is_fp32, grad_enabled, layout, expected
):
    (select,) = _load_pure_functions("select_ascend_add_route")

    assert (
        select(tensor_pair, same_shape, is_fp32, grad_enabled, layout)
        == expected
    )


@pytest.mark.parametrize(
    "suffix,n_rows,expected",
    [
        (1, 31, 8),
        (1, 1_024, 32),
        (1, 16_384, 512),
        (1, 1_038_240, 8),
        (13, 31, 64),
        (13, 1_024, 32),
        (13, 16_384, 32),
        (13, 1_038_240, 256),
        (31, 1_038_240, 2),
    ],
)
def test_ascend_suffix_rows_per_program_uses_generalized_size_classes(
    suffix, n_rows, expected
):
    (select,) = _load_pure_functions("_select_rows_per_program")

    assert select(suffix, n_rows) == expected


@pytest.mark.parametrize(
    "n_elements,expected",
    [(31, 256), (16_384, 1024), (65_536, 8192), (67_108_864, 8192)],
)
def test_ascend_flat_block_size_uses_generalized_size_classes(
    n_elements, expected
):
    (select,) = _load_pure_functions("_select_flat_block_size")

    assert select(n_elements) == expected


def test_ascend_add_source_has_only_triton_or_common_routes():
    source = SOURCE_PATH.read_text()

    for forbidden in (
        "_FALLBACK_KEYSET",
        "_native_add",
        "redispatch",
        "get_kernel",
        "call_boxed",
        "torch.add",
        "torch.ops.aten.add",
    ):
        assert forbidden not in source
    assert "_flat_add_kernel[grid]" in source
    assert "_suffix_add_kernel[grid]" in source
    assert "return _common_add(A, B, alpha=alpha)" in source


def test_ascend_add_emits_the_standard_dispatch_record_marker():
    source = SOURCE_PATH.read_text()

    assert source.count('logger.debug("GEMS ADD")') == 2
    assert "GEMS_ASCEND ADD_" not in source


def test_ascend_ops_exports_functional_add():
    source = OPS_INIT_PATH.read_text()

    assert "from .add import add" in source
    assert '"add"' in source


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only route test"
)
@pytest.mark.parametrize("suffix", [1, 13])
@pytest.mark.parametrize("alpha", [0, 0.5, -2])
def test_ascend_suffix_add_public_path(suffix, alpha):
    backend = importlib.import_module(
        "flag_gems.runtime.backend._ascend.ops.add"
    )
    base_a = torch.randn(
        (1, 31, 176), device=flag_gems.device, dtype=torch.float32
    )
    base_b = torch.randn_like(base_a)
    a = base_a[..., :suffix]
    b = base_b[..., :suffix]
    reference = torch.add(a, b, alpha=alpha)

    assert backend.explain_ascend_add_route(a, b) == "suffix_strided"
    with flag_gems.use_gems():
        actual = torch.add(a, b, alpha=alpha)

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only route test"
)
def test_ascend_grad_enabled_add_uses_common_triton_path():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._ascend.ops.add"
    )
    base = torch.randn(
        (2, 31, 176), device=flag_gems.device, dtype=torch.float32
    )
    a = base[..., :13].requires_grad_()
    b = torch.randn_like(a, requires_grad=True)

    assert backend.explain_ascend_add_route(a, b) == "common"


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend-only route test"
)
def test_ascend_broadcast_add_uses_common_triton_path():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._ascend.ops.add"
    )
    a = torch.randn(
        (2, 31, 13), device=flag_gems.device, dtype=torch.float32
    )
    b = torch.randn((13,), device=flag_gems.device, dtype=torch.float32)
    reference = torch.add(a, b, alpha=0.5)

    assert backend.explain_ascend_add_route(a, b) == "common"
    with flag_gems.use_gems():
        actual = torch.add(a, b, alpha=0.5)

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
