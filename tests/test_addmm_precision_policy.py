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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flag_gems
from flag_gems.runtime import matmul_precision


def test_mthreads_fast_fp32_tune_pool_excludes_unsafe_k32_four_warp_tile():
    source = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py"
    ).read_text(encoding="utf-8")
    fma_tuner = source[source.index("@libentry()") : source.index("def addmm_kernel")]

    assert (
        '{"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}'
        not in fma_tuner
    )
    assert (
        '{"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 16}'
        in fma_tuner
    )


def test_mthreads_protective_route_captures_vendor_kernel_before_registration():
    source = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py"
    ).read_text(encoding="utf-8")

    assert 'torch.library.get_kernel("aten::addmm", "PrivateUse1")' in source
    assert 'torch.library.get_kernel("aten::addmm.out", "PrivateUse1")' in source
    assert "_NATIVE_ADDMM_KERNEL.call_boxed(" in source
    assert "_NATIVE_ADDMM_OUT_KERNEL.call_boxed(" in source
    assert "def _can_use_native_fp32_addmm(" in source


def _fake_torch(*, cuda=None, mudnn=None, npu=None, fallback="highest"):
    backends = SimpleNamespace()
    if cuda is not None:
        backends.cuda = SimpleNamespace(matmul=cuda)
    if mudnn is not None:
        backends.mudnn = mudnn
    return SimpleNamespace(
        backends=backends,
        npu=npu,
        get_float32_matmul_precision=lambda: fallback,
    )


@pytest.mark.parametrize(
    "vendor_name,fake_torch,expected",
    [
        ("nvidia", _fake_torch(cuda=SimpleNamespace(allow_tf32=False)), False),
        ("nvidia", _fake_torch(cuda=SimpleNamespace(allow_tf32=True)), True),
        ("hygon", _fake_torch(cuda=SimpleNamespace(allow_tf32=False)), False),
        ("hygon", _fake_torch(cuda=SimpleNamespace(allow_tf32=True)), True),
        (
            "ascend",
            _fake_torch(npu=SimpleNamespace(matmul=SimpleNamespace(allow_hf32=False))),
            False,
        ),
        (
            "ascend",
            _fake_torch(npu=SimpleNamespace(matmul=SimpleNamespace(allow_hf32=True))),
            True,
        ),
        ("mthreads", _fake_torch(mudnn=SimpleNamespace(allow_tf32=False)), False),
        ("mthreads", _fake_torch(mudnn=SimpleNamespace(allow_tf32=True)), True),
        ("nvidia", _fake_torch(fallback="high"), True),
        ("unknown", _fake_torch(fallback="high"), False),
    ],
)
def test_runtime_reports_fast_float32_matmul_mode(
    monkeypatch, vendor_name, fake_torch, expected
):
    monkeypatch.setattr(matmul_precision, "torch", fake_torch)

    assert matmul_precision.is_fast_float32_matmul_enabled(vendor_name) is expected


def test_runtime_precision_query_failure_is_strict(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("backend precision state is unavailable")

    fake_torch = _fake_torch()
    fake_torch.get_float32_matmul_precision = raise_runtime_error
    monkeypatch.setattr(matmul_precision, "torch", fake_torch)

    assert matmul_precision.is_fast_float32_matmul_enabled("nvidia") is False


@pytest.mark.parametrize(
    "dtype,runtime_enabled,expected",
    [
        (torch.float32, False, False),
        (torch.float32, True, True),
        (torch.float16, True, False),
        (torch.bfloat16, True, False),
        (torch.float64, True, False),
    ],
)
def test_fast_float32_is_limited_to_fp32_operands(
    monkeypatch, dtype, runtime_enabled, expected
):
    monkeypatch.setattr(
        matmul_precision,
        "is_fast_float32_matmul_enabled",
        lambda _vendor_name: runtime_enabled,
    )
    lhs = torch.empty((2, 3), dtype=dtype)
    rhs = torch.empty((3, 4), dtype=dtype)

    assert (
        matmul_precision.should_use_fast_float32_matmul("nvidia", lhs, rhs)
        is expected
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_common_addmm_passes_precision_mode_to_kernel(monkeypatch, enabled):
    addmm_module = importlib.import_module("flag_gems.ops.addmm")
    captured = {}

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **kwargs):
                captured.update(kwargs)

            return launch

    monkeypatch.setattr(addmm_module, "addmm_kernel", FakeKernel())
    monkeypatch.setattr(
        addmm_module,
        "torch_device_fn",
        SimpleNamespace(device=lambda _device: nullcontext()),
    )
    monkeypatch.setattr(
        addmm_module,
        "should_use_fast_float32_matmul",
        lambda *_args: enabled,
    )

    bias = torch.zeros(4, dtype=torch.float32)
    mat1 = torch.zeros((2, 3), dtype=torch.float32)
    mat2 = torch.zeros((3, 4), dtype=torch.float32)
    addmm_module._addmm_impl(bias, mat1, mat2, None, beta=1, alpha=1)

    assert captured["ALLOW_TF32"] is enabled


def test_common_addmm_tuning_cache_separates_precision_modes():
    addmm_module = importlib.import_module("flag_gems.ops.addmm")
    tuner = addmm_module.addmm_kernel.fn
    args = {
        "M": 128,
        "N": 256,
        "K": 512,
        "stride_am": 512,
        "stride_bk": 256,
        "ALLOW_TF32": False,
    }

    strict_key = tuner.get_key(args)
    args["ALLOW_TF32"] = True
    fast_key = tuner.get_key(args)

    assert strict_key != fast_key


def test_ascend_addmm_connects_runtime_policy_to_dot_precision():
    source_path = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_ascend/ops/addmm.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "from flag_gems.runtime.matmul_precision import (" in source
    assert 'should_use_fast_float32_matmul("ascend", mat1, mat2)' in source
    assert '"hf32"' in source
    assert '"ieee"' in source
    assert "input_precision=INPUT_PRECISION" in source
    assert "acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)" not in source


def test_ascend_missing_optional_tle_does_not_disable_vendor_package():
    source_path = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_ascend/ops/__init__.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "try:\n    from .cholesky_solve import" in source
    assert 'missing_module == "triton.experimental.tle"' in source
    assert 'missing_module.startswith("triton.experimental.tle.")' in source
    assert '__all__.remove("cholesky_solve")' in source
    assert '__all__.remove("cholesky_solve_out")' in source


@contextmanager
def _float32_matmul_mode(enabled):
    vendor_name = flag_gems.vendor_name
    if vendor_name == "ascend":
        backend = getattr(getattr(torch, "npu", None), "matmul", None)
        attribute = "allow_hf32"
    elif vendor_name == "mthreads":
        backend = getattr(torch.backends, "mudnn", None)
        attribute = "allow_tf32"
    else:
        backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        attribute = "allow_tf32"

    if backend is None or not hasattr(backend, attribute):
        pytest.skip(f"{vendor_name} does not expose a fast-FP32 matmul switch")

    previous = getattr(backend, attribute)
    setattr(backend, attribute, enabled)
    try:
        assert (
            matmul_precision.is_fast_float32_matmul_enabled(vendor_name) is enabled
        )
        yield
    finally:
        setattr(backend, attribute, previous)


@pytest.mark.skipif(
    flag_gems.runtime.device.device_count == 0,
    reason="requires an accelerator device",
)
def test_device_addmm_validates_strict_and_fast_fp32_accuracy():
    torch.manual_seed(20260904)
    M, N, K = 257, 129, 512
    mat1 = torch.randn((M, K), device=flag_gems.device, dtype=torch.float32)
    mat2 = torch.randn((K, N), device=flag_gems.device, dtype=torch.float32)
    bias = torch.randn((N,), device=flag_gems.device, dtype=torch.float32)
    reference = torch.addmm(
        bias.cpu().double(),
        mat1.cpu().double(),
        mat2.cpu().double(),
    ).to(torch.float32)

    with _float32_matmul_mode(False):
        with flag_gems.use_gems():
            strict = torch.addmm(bias, mat1, mat2)

    with _float32_matmul_mode(True):
        with flag_gems.use_gems():
            fast = torch.addmm(bias, mat1, mat2)

    strict_error = strict.cpu() - reference
    fast_error = fast.cpu() - reference
    reference_rms = reference.square().mean().sqrt()
    strict_normalized_rmse = strict_error.square().mean().sqrt() / reference_rms
    strict_normalized_max = strict_error.abs().max() / reference.abs().max()
    fast_normalized_rmse = fast_error.square().mean().sqrt() / reference_rms
    fast_normalized_max = fast_error.abs().max() / reference.abs().max()

    assert strict_normalized_rmse.item() <= 1e-4
    assert strict_normalized_max.item() <= 5e-4
    assert fast_normalized_rmse.item() <= 1e-2
    assert fast_normalized_max.item() <= 2e-2
    assert strict.dtype == fast.dtype == torch.float32
