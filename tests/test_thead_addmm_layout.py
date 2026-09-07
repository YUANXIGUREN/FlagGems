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

import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flag_gems


def test_thead_addmm_tuning_keeps_large_m_low_stage_candidates():
    tune_config = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_thead/tune_configs.yaml"
    ).read_text()
    addmm_config = tune_config[tune_config.index("addmm:") :]

    prefix = (
        "BLOCK_SIZE_M: 128\n"
        "      BLOCK_SIZE_N: 64\n"
        "      BLOCK_SIZE_K: 32\n"
        "    num_stages: 1\n"
    )
    assert prefix + "    num_warps: 4" in addmm_config
    assert prefix + "    num_warps: 8" in addmm_config


def _load_thead_op(name):
    module_prefix = (
        "_thead.ops"
        if flag_gems.vendor_name == "thead"
        else "flag_gems.runtime.backend._thead.ops"
    )
    try:
        return importlib.import_module(f"{module_prefix}.{name}")
    except ModuleNotFoundError:
        pytest.fail(f"the THead {name} vendor module is not registered")


def _load_thead_addmm():
    return _load_thead_op("addmm")


@pytest.mark.parametrize(
    "rows,dtype,column_major,grad_enabled,allow_tf32,should_materialize",
    [
        (1024, torch.float32, True, False, False, False),
        (1025, torch.float32, True, False, False, True),
        (1025, torch.float32, True, False, True, False),
        (1025, torch.float32, True, True, False, False),
        (1025, torch.float32, False, False, False, False),
        (1025, torch.float16, True, False, False, False),
        (1025, torch.bfloat16, True, False, False, False),
    ],
)
def test_thead_addmm_materializes_only_large_fp32_transposed_rhs(
    rows, dtype, column_major, grad_enabled, allow_tf32, should_materialize
):
    module = _load_thead_addmm()
    mat1 = torch.empty((rows, 3), dtype=dtype)
    if column_major:
        mat2 = torch.arange(15, dtype=dtype).reshape(5, 3).t()
    else:
        mat2 = torch.arange(15, dtype=dtype).reshape(3, 5)

    grad_context = torch.enable_grad if grad_enabled else torch.no_grad
    with grad_context():
        prepared = module._prepare_mat2(mat1, mat2, allow_tf32=allow_tf32)

    if should_materialize:
        assert prepared is not mat2
        assert prepared.is_contiguous()
        torch.testing.assert_close(prepared, mat2)
    else:
        assert prepared is mat2


@pytest.mark.parametrize(
    "api_name",
    [
        "addmm",
        "addmm_out",
        "addmm_dtype",
        "addmm_dtype_out",
    ],
)
def test_thead_addmm_apis_launch_vendor_kernel(monkeypatch, api_name):
    module = _load_thead_addmm()
    mat1 = torch.zeros((1025, 3), dtype=torch.float32)
    mat2 = torch.arange(15, dtype=torch.float32).reshape(5, 3).t()
    bias = torch.zeros(5, dtype=torch.float32)
    out = torch.empty((1025, 5), dtype=torch.float32)
    observed = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            return launch

    def launch(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs

    monkeypatch.setattr(module, "addmm_kernel", FakeKernel())
    monkeypatch.setattr(
        module,
        "torch_device_fn",
        SimpleNamespace(device=lambda _device: nullcontext()),
    )
    monkeypatch.setattr(
        module,
        "should_use_fast_float32_matmul",
        lambda *_args: False,
    )
    api = getattr(module, api_name)
    with torch.no_grad():
        if api_name == "addmm":
            result = api(bias, mat1, mat2, beta=2, alpha=3)
        elif api_name == "addmm_out":
            result = api(bias, mat1, mat2, beta=2, alpha=3, out=out)
        elif api_name == "addmm_dtype":
            result = api(bias, mat1, mat2, torch.float32, beta=2, alpha=3)
        else:
            result = api(
                bias,
                mat1,
                mat2,
                torch.float32,
                beta=2,
                alpha=3,
                out=out,
            )

    assert result is observed["args"][3]
    prepared_mat2 = observed["args"][1]
    assert prepared_mat2 is not mat2
    assert prepared_mat2.is_contiguous()
    torch.testing.assert_close(prepared_mat2, mat2)
    assert observed["args"][4] == 3
    assert observed["args"][5] == 2
    assert observed["kwargs"]["ALLOW_TF32"] is False
    if "out" in api_name:
        assert result is out


def test_thead_inplace_addmm_uses_the_vendor_kernel():
    vendor_module = _load_thead_addmm()
    inplace_module = _load_thead_op("addmm_")

    assert inplace_module.addmm_kernel is vendor_module.addmm_kernel


@pytest.mark.skipif(
    flag_gems.vendor_name != "thead",
    reason="Public vendor dispatch is observable only on T-Head",
)
def test_thead_public_addmm_dispatch_uses_the_vendor_implementation():
    vendor_module = _load_thead_addmm()

    assert flag_gems.addmm is vendor_module.addmm


def test_thead_inplace_addmm_passes_the_full_vendor_kernel_contract(monkeypatch):
    module = _load_thead_op("addmm_")
    self = torch.zeros((4, 5), dtype=torch.float32)
    mat1 = torch.zeros((4, 3), dtype=torch.float32)
    mat2 = torch.zeros((3, 5), dtype=torch.float32)
    observed = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            return launch

    def launch(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs

    monkeypatch.setattr(module, "addmm_kernel", FakeKernel())
    monkeypatch.setattr(
        module,
        "torch_device_fn",
        SimpleNamespace(device=lambda _device: nullcontext()),
    )
    monkeypatch.setattr(
        module,
        "should_use_fast_float32_matmul",
        lambda *_args: True,
    )

    result = module.addmm_(self, mat1, mat2, beta=2, alpha=3)

    assert result is self
    assert observed["kwargs"]["BIAS_IS_VECTOR"] is False
    assert observed["kwargs"]["BIAS_IS_SCALAR"] is False
    assert observed["kwargs"]["HAS_K"] is True
    assert observed["kwargs"]["ALLOW_TF32"] is True
