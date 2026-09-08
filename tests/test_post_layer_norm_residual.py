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
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flag_gems


def _make_affine(normalized_shape, dtype, affine):
    if not affine:
        return None, None
    return (
        torch.randn(normalized_shape, dtype=dtype),
        torch.randn(normalized_shape, dtype=dtype),
    )


@pytest.mark.parametrize(
    "shape,normalized_shape,dtype,affine,eps",
    [
        ((2, 3, 5), (5,), torch.float32, True, 1e-5),
        ((2, 3, 5), (5,), torch.float16, False, 1e-6),
        ((2, 3, 4, 5), (4, 5), torch.bfloat16, True, 2e-4),
    ],
)
def test_post_layer_norm_residual_matches_torch_composition(
    shape, normalized_shape, dtype, affine, eps
):
    x = torch.randn(shape, dtype=dtype)
    residual = torch.randn((1,) + shape[1:], dtype=dtype)
    weight, bias = _make_affine(normalized_shape, dtype, affine)

    expected = torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual
    actual = flag_gems.post_layer_norm_residual(
        x, residual, normalized_shape, weight, bias, eps
    )

    torch.testing.assert_close(actual, expected)


def test_post_layer_norm_residual_handles_noncontiguous_inputs():
    x = torch.randn((3, 2, 4), dtype=torch.float32).transpose(0, 1)
    residual = torch.randn((3, 2, 4), dtype=torch.float32).transpose(0, 1)
    assert not x.is_contiguous()
    assert not residual.is_contiguous()

    expected = torch.layer_norm(x, (4,)) + residual
    actual = flag_gems.post_layer_norm_residual(x, residual, (4,))

    torch.testing.assert_close(actual, expected)


def test_post_layer_norm_residual_handles_empty_leading_dimensions():
    x = torch.empty((0, 3, 4), dtype=torch.float32)
    residual = torch.randn((1, 3, 4), dtype=torch.float32)

    expected = torch.layer_norm(x, (4,)) + residual
    actual = flag_gems.post_layer_norm_residual(x, residual, (4,))

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (0, 3, 4)


def test_post_layer_norm_residual_preserves_autograd():
    x = torch.randn((2, 4), dtype=torch.float32, requires_grad=True)
    residual = torch.randn((2, 4), dtype=torch.float32, requires_grad=True)
    weight = torch.randn((4,), dtype=torch.float32, requires_grad=True)
    bias = torch.randn((4,), dtype=torch.float32, requires_grad=True)
    grad_output = torch.tensor(
        [[0.5, -1.0, 2.0, -0.25], [1.5, 0.0, -0.5, 0.75]], dtype=torch.float32
    )

    expected = torch.layer_norm(x, (4,), weight, bias, 3e-5) + residual
    actual = flag_gems.post_layer_norm_residual(x, residual, (4,), weight, bias, 3e-5)
    expected_grads = torch.autograd.grad(
        expected, (x, residual, weight, bias), grad_output, retain_graph=True
    )
    actual_grads = torch.autograd.grad(actual, (x, residual, weight, bias), grad_output)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "normalized_shape,weight,bias,residual",
    [
        ((5,), None, None, torch.randn((2, 3, 4))),
        ((4,), torch.randn((3,)), None, torch.randn((2, 3, 4))),
        ((4,), None, None, torch.randn((2, 2))),
    ],
)
def test_post_layer_norm_residual_preserves_torch_validation(
    normalized_shape, weight, bias, residual
):
    x = torch.randn((2, 3, 4))

    with pytest.raises(Exception) as expected_error:
        torch.layer_norm(x, normalized_shape, weight, bias, 1e-5) + residual

    with pytest.raises(
        type(expected_error.value), match=re.escape(str(expected_error.value))
    ):
        flag_gems.post_layer_norm_residual(x, residual, normalized_shape, weight, bias)


def test_post_layer_norm_residual_normalizes_before_adding_residual():
    x = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    residual = torch.tensor([[10.0, 0.0, -10.0, 0.0]])

    expected = torch.layer_norm(x, (4,)) + residual
    add_then_normalize = torch.layer_norm(x + residual, (4,))
    actual = flag_gems.post_layer_norm_residual(x, residual, (4,))

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, add_then_normalize)


def _load_ascend_post_layernorm():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/flag_gems/runtime/backend/_ascend/fused/post_layernorm_residual.py"
    )
    assert path.is_file(), "Ascend post-LayerNorm residual implementation is missing"
    spec = importlib.util.spec_from_file_location("ascend_post_layernorm_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NpuMetadata:
    """Keep real CPU tensor metadata, replacing only the unavailable device."""

    def __init__(self, tensor, index=0):
        self.tensor = tensor
        self.device = SimpleNamespace(type="npu", index=index)

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 5, 4095, 4096])
@pytest.mark.parametrize("affine", ["none", "weight", "bias", "both"])
def test_ascend_fast_guard_accepts_supported_inference(dtype, width, affine):
    backend = _load_ascend_post_layernorm()
    x = _NpuMetadata(torch.empty((2, 2, width), dtype=dtype))
    residual = _NpuMetadata(torch.empty_like(x.tensor))
    weight = _NpuMetadata(torch.empty((width,), dtype=dtype))
    bias = _NpuMetadata(torch.empty((width,), dtype=dtype))
    with torch.no_grad():
        assert backend._can_use_fast_path(
            x,
            residual,
            (width,),
            weight if "weight" == affine or affine == "both" else None,
            bias if "bias" == affine or affine == "both" else None,
        )


@pytest.mark.parametrize(
    "case",
    [
        "grad",
        "shape",
        "dtype",
        "device",
        "cpu",
        "x_layout",
        "residual_layout",
        "large",
        "empty",
        "zero_width",
        "suffix",
        "empty_suffix",
        "integer_suffix",
        "unsupported_dtype",
        "weight_shape",
        "bias_shape",
        "weight_dtype",
        "bias_dtype",
        "weight_device",
        "bias_device",
        "weight_layout",
        "bias_layout",
        "weight_nontensor",
        "bias_nontensor",
    ],
)
def test_ascend_fast_guard_rejects_unsupported_inputs(case):
    backend = _load_ascend_post_layernorm()
    tensors = {
        name: _NpuMetadata(torch.empty(shape))
        for name, shape in (
            ("x", (2, 4)),
            ("residual", (2, 4)),
            ("weight", (4,)),
            ("bias", (4,)),
        )
    }
    normalized_shape = (4,)
    if case == "shape":
        tensors["residual"] = _NpuMetadata(torch.empty((1, 4)))
    elif case == "dtype":
        tensors["residual"] = _NpuMetadata(torch.empty((2, 4), dtype=torch.float16))
    elif case == "device":
        tensors["residual"].device.index = 1
    elif case == "cpu":
        tensors["x"].device.type = "cpu"
        tensors["residual"].device.type = "cpu"
    elif case in ("x_layout", "residual_layout"):
        tensors[case.removesuffix("_layout")] = _NpuMetadata(torch.empty((4, 2)).t())
    elif case in ("large", "empty", "zero_width", "unsupported_dtype"):
        shape = {"large": (2, 4097), "empty": (0, 4), "zero_width": (2, 0)}.get(
            case, (2, 4)
        )
        dtype = torch.float64 if case == "unsupported_dtype" else torch.float32
        tensors = {
            name: _NpuMetadata(torch.empty(shape, dtype=dtype))
            for name in ("x", "residual")
        }
        tensors.update(weight=None, bias=None)
        normalized_shape = (shape[-1],)
    elif case in ("suffix", "empty_suffix", "integer_suffix"):
        normalized_shape = {"suffix": (3,), "empty_suffix": (), "integer_suffix": 4}[
            case
        ]
    elif case.startswith(("weight_", "bias_")):
        name, change = case.split("_")
        if change == "shape":
            tensors[name] = _NpuMetadata(torch.empty(3))
        elif change == "dtype":
            tensors[name] = _NpuMetadata(torch.empty(4, dtype=torch.float16))
        elif change == "device":
            tensors[name].device.index = 1
        elif change == "nontensor":
            tensors[name] = 1.0
        else:
            tensors[name] = _NpuMetadata(torch.empty(8)[::2])
    with torch.set_grad_enabled(case == "grad"):
        assert not backend._can_use_fast_path(
            normalized_shape=normalized_shape, **tensors
        )


def test_ascend_wrapper_uses_common_fallback_without_recursion(monkeypatch):
    backend = _load_ascend_post_layernorm()
    x = torch.randn((2, 4), requires_grad=True)
    residual = torch.randn((1, 4), requires_grad=True)
    expected = torch.layer_norm(x, (4,)) + residual
    monkeypatch.setattr(
        flag_gems, "post_layer_norm_residual", backend.post_layer_norm_residual
    )
    actual = flag_gems.post_layer_norm_residual(x, residual, (4,))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(residual.grad, torch.full_like(residual, 2))


def test_ascend_launch_failure_returns_composition_not_partial_output(monkeypatch):
    backend = _load_ascend_post_layernorm()

    class FailingKernel:
        def __getitem__(self, grid):
            def launch(x, residual, output, *args, **kwargs):
                output.fill_(12345)
                raise RuntimeError("test compiler failure")

            return launch

    # CPU tests cannot launch Ascend code; replace only the device boundary.
    monkeypatch.setattr(backend, "_can_use_fast_path", lambda *args: True)
    monkeypatch.setattr(backend.torch_device_fn, "device", lambda *args: nullcontext())
    monkeypatch.setattr(backend, "post_layer_norm_residual_kernel", FailingKernel())
    x = torch.randn((2, 4))
    residual = torch.randn_like(x)
    expected = torch.layer_norm(x, (4,)) + residual
    with torch.no_grad():
        actual = backend.post_layer_norm_residual(x, residual, (4,))
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(flag_gems.device != "npu", reason="requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [((3, 5), (5,)), ((2, 3, 4), (3, 4)), ((2, 4096), (4096,))],
)
@pytest.mark.parametrize("affine", ["none", "weight", "bias", "both"])
def test_ascend_post_layernorm_device_semantics(
    monkeypatch, dtype, shape, normalized_shape, affine
):
    import sys

    backend = sys.modules[flag_gems.post_layer_norm_residual.__module__]
    assert "_ascend.fused.post_layernorm_residual" in backend.__name__

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("eligible Ascend input used the common fallback")

    monkeypatch.setattr(backend, "common_post_layer_norm_residual", unexpected_fallback)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    residual = torch.randn_like(x) * 3
    weight = (
        torch.randn(normalized_shape, dtype=dtype, device=x.device)
        if affine in ("weight", "both")
        else None
    )
    bias = (
        torch.randn(normalized_shape, dtype=dtype, device=x.device)
        if affine in ("bias", "both")
        else None
    )
    with torch.no_grad():
        expected = torch.layer_norm(x, normalized_shape, weight, bias, 2e-4) + residual
        actual = flag_gems.post_layer_norm_residual(
            x, residual, normalized_shape, weight, bias, 2e-4
        )
    tolerance = {torch.float32: 1e-5, torch.float16: 8e-3, torch.bfloat16: 6e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual.dtype == dtype and actual.device == x.device
    assert actual.shape == x.shape and actual.is_contiguous()
    assert actual.data_ptr() not in (x.data_ptr(), residual.data_ptr())


@pytest.mark.parametrize(
    "case", ["grad", "broadcast", "layout", "large", "affine_layout"]
)
def test_ascend_fallback_preserves_composition(case):
    backend = _load_ascend_post_layernorm()
    device = flag_gems.device if flag_gems.device == "npu" else "cpu"
    width = 4097 if case == "large" else 5
    x = torch.randn((3, width), device=device, requires_grad=case == "grad")
    residual = torch.randn((1 if case == "broadcast" else 3, width), device=device)
    weight = torch.randn((width,), device=device)
    if case == "layout":
        x = torch.randn((width, 3), device=device).t()
    if case == "affine_layout":
        weight = torch.randn((width * 2,), device=device)[::2]
    with torch.set_grad_enabled(case == "grad"):
        expected = torch.layer_norm(x, (width,), weight) + residual
        actual = backend.post_layer_norm_residual(x, residual, (width,), weight)
        torch.testing.assert_close(actual, expected)
        if case == "grad":
            actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected.sum(), x)[0]
            torch.testing.assert_close(actual_grad, expected_grad)
