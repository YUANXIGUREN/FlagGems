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

import re

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
