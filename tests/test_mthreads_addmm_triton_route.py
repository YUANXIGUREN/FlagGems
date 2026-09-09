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
    / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py"
)


def _load_pure_function(name):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert name in functions, f"missing required routing function: {name}"
    namespace = {}
    module = ast.Module(body=[functions[name]], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "K,stride_bk,stride_bn,expected",
    [
        (4, 1, 4, "skinny_k"),
        (184, 184, 1, "rhs_row"),
        (512, 1, 512, "rhs_k_compact"),
        (512, 1, 1536, "rhs_k_padded"),
        (1024, 1, 1024, "rhs_k_compact"),
        (384, 7, 11, "general"),
    ],
)
def test_mthreads_layout_class_is_stable(K, stride_bk, stride_bn, expected):
    classify = _load_pure_function("classify_mthreads_addmm_layout")

    assert classify(K, stride_bk, stride_bn) == expected


@pytest.mark.parametrize(
    "K,sqmma_compatible,promotes_to_fp32,expected",
    [
        (4, False, False, "skinny_k"),
        (184, False, False, "pointer"),
        (512, True, False, "sqmma"),
        (512, True, True, "pointer"),
    ],
)
def test_mthreads_route_never_returns_native(
    K, sqmma_compatible, promotes_to_fp32, expected
):
    select = _load_pure_function("select_mthreads_addmm_route")

    route = select(K, sqmma_compatible, promotes_to_fp32)

    assert route == expected
    assert route in {"pointer", "sqmma", "skinny_k"}


@pytest.mark.parametrize(
    (
        "all_fp32,grad_sensitive,m,n,k,a_contiguous,stride_bk,stride_bn,"
        "bias_is_vector,out_contiguous,expected"
    ),
    [
        (True, False, 131072, 512, 4, True, 1, 4, True, True, True),
        (True, False, 40962, 512, 4, True, 1, 4, True, True, True),
        (True, False, 1024, 512, 4, True, 1, 4, True, True, False),
        (True, False, 131072, 256, 4, True, 1, 4, True, True, False),
        (True, False, 131072, 512, 8, True, 1, 8, True, True, False),
        (True, True, 131072, 512, 4, True, 1, 4, True, True, False),
        (False, False, 131072, 512, 4, True, 1, 4, True, True, False),
        (True, False, 131072, 512, 4, False, 1, 4, True, True, False),
        (True, False, 131072, 512, 4, True, 512, 1, True, True, False),
        (True, False, 131072, 512, 4, True, 1, 4, False, True, False),
    ],
)
def test_mthreads_k4_tiled_contract_is_narrow_and_data_independent(
    all_fp32,
    grad_sensitive,
    m,
    n,
    k,
    a_contiguous,
    stride_bk,
    stride_bn,
    bias_is_vector,
    out_contiguous,
    expected,
):
    select = _load_pure_function("can_use_mthreads_k4_tiled_contract")

    assert (
        select(
            all_fp32,
            grad_sensitive,
            m,
            n,
            k,
            a_contiguous,
            stride_bk,
            stride_bn,
            bias_is_vector,
            out_contiguous,
        )
        is expected
    )


@pytest.mark.parametrize(
    "is_fp32,fast_enabled,grad_sensitive,m,n,k,expected",
    [
        (True, True, False, 40962, 512, 512, True),
        (True, True, False, 40962, 512, 184, False),
        (True, True, False, 1038240, 83, 512, False),
        (True, True, False, 40962, 512, 1024, True),
        (True, True, False, 1038240, 512, 1024, True),
        (True, True, False, 131072, 512, 4, False),
        (True, False, False, 40962, 512, 512, False),
        (False, True, False, 40962, 512, 512, False),
        (True, True, True, 40962, 512, 512, False),
    ],
)
def test_mthreads_tf32_sqmma_contract_is_data_independent(
    is_fp32, fast_enabled, grad_sensitive, m, n, k, expected
):
    select = _load_pure_function("can_use_mthreads_tf32_sqmma_contract")

    assert (
        select(is_fp32, fast_enabled, grad_sensitive, m, n, k) is expected
    )


@pytest.mark.parametrize(
    "is_fp32,fast_enabled,route,expected",
    [
        (True, True, "pointer", True),
        (True, True, "skinny_k", False),
        (True, False, "pointer", False),
        (False, True, "pointer", False),
        (True, True, "sqmma", False),
    ],
)
def test_mthreads_inline_rounding_policy_is_limited_to_fp32_pointer_paths(
    is_fp32, fast_enabled, route, expected
):
    select = _load_pure_function("should_inline_round_mthreads_addmm")

    assert select(is_fp32, fast_enabled, route) is expected


@pytest.mark.parametrize(
    "is_fp32,fast_enabled,expected",
    [
        (True, True, "tf32"),
        (True, False, "ieee"),
        (False, True, "ieee"),
    ],
)
def test_mthreads_pointer_precision_matches_native_fast_mode(
    is_fp32, fast_enabled, expected
):
    select = _load_pure_function("select_mthreads_pointer_precision")

    assert select(is_fp32, fast_enabled) == expected


def test_mthreads_addmm_source_has_no_native_dispatch_hooks():
    source = SOURCE_PATH.read_text()

    for forbidden in (
        "_load_native_addmm_kernels",
        "_NATIVE_ADDMM_KERNEL",
        "_NATIVE_ADDMM_MODE",
        "get_kernel",
        "call_boxed",
        "redispatch",
        "torch.addmm",
        "torch.ops.aten.addmm",
    ):
        assert forbidden not in source


def test_mthreads_fp32_pointer_route_uses_rne_tf32_without_materialized_copies():
    source = SOURCE_PATH.read_text()

    assert "round_to_tf32_copy(mat1)" not in source
    assert "mat2 = _get_rounded_tf32_rhs(mat2)" not in source
    assert "ROUND_TF32_INPUTS=inline_round" in source
    assert 'input_precision="tf32"' in source
    assert ".contiguous()" not in source
    assert "BETA_IS_ZERO=beta == 0" in source


def test_mthreads_tf32_sqmma_route_is_owned_by_triton():
    source = SOURCE_PATH.read_text()

    assert "def round_to_tf32_fp16_copy(" in source
    assert "_get_tf32_fp16_rhs(mat2)" in source
    assert "def addmm_tf32_sqmma(" in source
    assert "return addmm_tf32_sqmma(" in source
    assert "TensorDescriptor.from_tensor" in source


def test_mthreads_skinny_route_launches_a_dedicated_kernel():
    source = SOURCE_PATH.read_text()

    assert "def addmm_skinny_k_flat_kernel(" in source
    assert "if route == \"skinny_k\":" in source
    assert "addmm_skinny_k_flat_kernel[skinny_grid]" in source


def test_mthreads_graph_shape_k4_route_uses_2d_data_reuse():
    source = SOURCE_PATH.read_text()

    assert "def addmm_skinny_k4_tiled_kernel(" in source
    assert "def addmm_skinny_k4_tiled(" in source
    assert "BLOCK_SIZE_M=32" in source
    assert "BLOCK_SIZE_N=64" in source
    assert "return addmm_skinny_k4_tiled(" in source


@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="Moore Threads-only test"
)
def test_mthreads_public_fp32_addmm_uses_backend_triton_route():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    assert flag_gems.addmm.__module__ == "_mthreads.ops.addmm"
    mat1 = torch.randn((257, 4), dtype=torch.float32, device=flag_gems.device)
    mat2 = torch.randn((4, 512), dtype=torch.float32, device=flag_gems.device)
    bias = torch.randn((512,), dtype=torch.float32, device=flag_gems.device)

    route = backend.explain_mthreads_addmm_route(
        bias, mat1, mat2, out=None, out_dtype=None
    )
    previous = torch.backends.mudnn.allow_tf32
    torch.backends.mudnn.allow_tf32 = False
    try:
        reference = torch.addmm(bias, mat1, mat2)
        with flag_gems.use_gems():
            actual = torch.addmm(bias, mat1, mat2)
    finally:
        torch.backends.mudnn.allow_tf32 = previous

    assert route == "skinny_k"
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="Moore Threads-only test"
)
def test_mthreads_large_k4_addmm_uses_tiled_triton_route():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    mat1 = torch.randn((40962, 4), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((512, 4), dtype=torch.float32, device=flag_gems.device)
    mat2 = weight.t()
    bias = torch.randn((512,), dtype=torch.float32, device=flag_gems.device)
    previous = torch.backends.mudnn.allow_tf32
    torch.backends.mudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            route = backend.explain_mthreads_addmm_route(
                bias, mat1, mat2, out=None, out_dtype=torch.float32
            )
            reference = torch.addmm(bias, mat1, mat2)
            actual = backend.addmm(bias, mat1, mat2)
    finally:
        torch.backends.mudnn.allow_tf32 = previous

    assert route == "skinny_k4_tiled"
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-4)


@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads", reason="Moore Threads-only test"
)
def test_mthreads_k184_fp32_addmm_preserves_exponent_on_pointer_route():
    backend = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    mat1 = torch.randn((257, 184), dtype=torch.float32, device=flag_gems.device)
    mat2 = torch.randn((184, 512), dtype=torch.float32, device=flag_gems.device)
    bias = torch.randn((512,), dtype=torch.float32, device=flag_gems.device)
    previous = torch.backends.mudnn.allow_tf32
    torch.backends.mudnn.allow_tf32 = True
    try:
        with torch.inference_mode():
            route = backend.explain_mthreads_addmm_route(
                bias, mat1, mat2, out=None, out_dtype=torch.float32
            )
            reference = torch.addmm(bias, mat1, mat2)
            with flag_gems.use_gems():
                actual = torch.addmm(bias, mat1, mat2)
    finally:
        torch.backends.mudnn.allow_tf32 = previous

    assert route == "pointer"
    normalized_rmse = (
        (actual - reference).square().mean().sqrt()
        / reference.square().mean().sqrt().clamp_min(1e-30)
    )
    assert torch.isfinite(actual).all()
    assert normalized_rmse < 1e-3
