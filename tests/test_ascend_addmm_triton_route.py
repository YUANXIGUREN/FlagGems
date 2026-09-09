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
from pathlib import Path

import pytest
import torch
import yaml

import flag_gems


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src/flag_gems/runtime/backend/_ascend/ops/addmm.py"
)
TUNE_PATH = SOURCE_PATH.parents[1] / "tune_configs.yaml"


def _load_pure_function(name):
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


def test_layout_classifier_distinguishes_compact_padded_and_row_rhs():
    classify = _load_pure_function("classify_addmm_layout")

    assert classify(512, 512, 1, 1, 512) == "a_row_b_k_contiguous"
    assert classify(512, 512, 1, 1, 1536) == "a_row_b_k_padded"
    assert classify(512, 512, 1, 512, 1) == "a_row_b_row"
    assert classify(512, 1, 4096, 1, 512) == "a_column_b_k_contiguous"
    assert classify(512, 4096, 2, 1024, 2) == "general"


def test_empty_k_selector_uses_fma_reduction():
    select = _load_pure_function("select_ascend_addmm_kernel")

    assert select(0, "a_row_b_k_padded") == "skinny_k"


@pytest.mark.parametrize("K", [1, 4, 15, 16, 184, 512, 1024])
def test_non_empty_selector_uses_grouped_gemm(K):
    select = _load_pure_function("select_ascend_addmm_kernel")

    assert select(K, "a_row_b_k_padded") == "grouped_gemm"


@pytest.mark.parametrize(
    "M,N,K,fp32_operands,fast_enabled,expected",
    [
        (4096, 512, 184, True, False, "ieee"),
        (4096, 512, 184, True, True, "hf32"),
        (4096, 512, 184, False, True, "ieee"),
        (495, 5333, 71, True, True, "ieee"),
        (257, 512, 184, True, True, "ieee"),
    ],
)
def test_precision_selector_only_maps_fp32_fast_mode_to_hf32(
    M, N, K, fp32_operands, fast_enabled, expected
):
    select = _load_pure_function("select_ascend_input_precision")

    assert select(M, N, K, fp32_operands, fast_enabled) == expected


def test_grouped_dot_consumes_compile_time_input_precision():
    source = SOURCE_PATH.read_text()

    assert "input_precision=INPUT_PRECISION" in source
    assert "INPUT_PRECISION=select_ascend_input_precision(" in source


def test_addmm_silu_reuses_grouped_gemm_with_a_compile_time_epilogue():
    source = SOURCE_PATH.read_text()

    assert "FUSE_SILU: tl.constexpr" in source
    assert "if FUSE_SILU:" in source
    assert "def addmm_silu(" in source
    assert "fuse_silu=True" in source


def test_grouped_kernel_uses_addmm_owned_measured_configurations():
    source = SOURCE_PATH.read_text()
    configs = yaml.safe_load(TUNE_PATH.read_text())["addmm"]
    metas = [config["META"] for config in configs]

    assert 'configs=runtime.get_tuned_config("addmm")' in source
    assert {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "GROUP_M": 8,
        "SPLIT_K": 1,
    } in metas
    assert {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 128,
        "GROUP_M": 8,
        "SPLIT_K": 1,
    } in metas
    assert all(meta["BLOCK_M"] * meta["BLOCK_N"] <= 32768 for meta in metas)
    assert all(meta["BLOCK_K"] <= 128 for meta in metas)


@pytest.mark.parametrize(
    "M,fp32_operands,expected",
    [
        (15, False, False),
        (4096, False, False),
        (257, True, False),
        (4096, True, True),
        (131072, True, True),
    ],
)
def test_large_tile_selector_is_limited_to_fp32_large_m(
    M, fp32_operands, expected
):
    select = _load_pure_function("use_large_ascend_addmm_tiles")

    assert select(M, fp32_operands) is expected


def test_grouped_tuner_prunes_unsafe_large_tiles():
    source = SOURCE_PATH.read_text()

    assert "prune_configs_by={" in source
    assert '"early_config_prune": prune_ascend_addmm_configs' in source


def test_source_has_no_native_linear_or_redispatch_route():
    source = SOURCE_PATH.read_text()

    assert "npu_linear" not in source
    assert "redispatch" not in source
    assert "get_kernel" not in source
    assert "call_boxed" not in source


@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend-only HF32 test")
def test_ascend_hf32_grouped_kernel_executes():
    M, N, K = 4096, 512, 184
    mat1 = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
    storage = torch.randn((N, 1536), dtype=torch.float32, device=flag_gems.device)
    mat2 = storage[:, :K].t()
    bias = torch.randn((N,), dtype=torch.float32, device=flag_gems.device)
    previous = torch.npu.matmul.allow_hf32
    try:
        torch.npu.matmul.allow_hf32 = True
        reference = torch.addmm(bias, mat1, mat2)
        with flag_gems.use_gems():
            result = torch.addmm(bias, mat1, mat2)
    finally:
        torch.npu.matmul.allow_hf32 = previous

    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
