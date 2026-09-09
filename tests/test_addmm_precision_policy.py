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

from types import SimpleNamespace

import pytest
import torch

from flag_gems.runtime import matmul_precision


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
        ("unknown", _fake_torch(fallback="high"), False),
    ],
)
def test_runtime_reports_backend_fast_float32_mode(
    monkeypatch, vendor_name, fake_torch, expected
):
    monkeypatch.setattr(matmul_precision, "torch", fake_torch)

    assert matmul_precision.is_fast_float32_matmul_enabled(vendor_name) is expected


@pytest.mark.parametrize("vendor_name", ["ascend", "hygon", "mthreads", "unknown"])
def test_missing_backend_precision_attribute_is_strict(monkeypatch, vendor_name):
    monkeypatch.setattr(matmul_precision, "torch", _fake_torch())

    assert matmul_precision.is_fast_float32_matmul_enabled(vendor_name) is False


@pytest.mark.parametrize("vendor_name", ["ascend", "hygon", "mthreads"])
def test_runtime_precision_query_failure_is_strict(monkeypatch, vendor_name):
    class FailingNamespace:
        def __getattr__(self, name):
            raise RuntimeError(f"cannot read {name}")

    fake_torch = SimpleNamespace(
        backends=FailingNamespace(),
        npu=FailingNamespace(),
        get_float32_matmul_precision=FailingNamespace(),
    )
    monkeypatch.setattr(matmul_precision, "torch", fake_torch)

    assert matmul_precision.is_fast_float32_matmul_enabled(vendor_name) is False


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
def test_fast_float32_is_limited_to_fp32_matrix_operands(
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
        matmul_precision.should_use_fast_float32_matmul("ascend", lhs, rhs)
        is expected
    )


def test_fast_float32_requires_every_operand_to_be_fp32(monkeypatch):
    monkeypatch.setattr(
        matmul_precision,
        "is_fast_float32_matmul_enabled",
        lambda _vendor_name: True,
    )
    fp32 = torch.empty((2, 2), dtype=torch.float32)
    bf16 = torch.empty((2, 2), dtype=torch.bfloat16)

    assert not matmul_precision.should_use_fast_float32_matmul(
        "mthreads", fp32, bf16
    )
    assert not matmul_precision.should_use_fast_float32_matmul("mthreads")
