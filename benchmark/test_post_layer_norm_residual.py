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

"""Complete post-LayerNorm residual calls; the baseline is Torch composition.

There is no Official fused baseline. Source-isolated component-composition
comparisons and raw repeats are recorded separately by the platform campaign.
The correctness suite consumes this same generator to cover every public case.
"""

import pytest
import torch

import flag_gems


def public_inputs(dtype, device):
    cases = [
        ("rows_small", (1, 257, 512), (512,), True, False, 1e-5),
        ("rows_large", (1, 8193, 512), (512,), True, False, 1e-5),
        ("width_below", (1, 64, 4095), (4095,), True, False, 1e-5),
        ("width_at", (1, 64, 4096), (4096,), True, False, 1e-5),
        ("width_above", (1, 64, 4097), (4097,), True, False, 1e-5),
        ("strided", (1, 257, 512), (512,), True, True, 1e-5),
        ("non_affine", (1, 257, 512), (512,), False, False, 1e-6),
        ("multi_normalized", (2, 17, 16, 32), (16, 32), True, False, 2e-4),
    ]
    for name, shape, normalized_shape, affine, strided, eps in cases:
        allocation_shape = shape[:-1] + (shape[-1] * (2 if strided else 1),)
        x = torch.randn(allocation_shape, device=device, dtype=dtype)
        residual = torch.randn_like(x)
        if strided:
            x, residual = x[..., ::2], residual[..., ::2]
        weight = (
            torch.randn(normalized_shape, device=device, dtype=dtype)
            if affine
            else None
        )
        bias = (
            torch.randn(normalized_shape, device=device, dtype=dtype)
            if affine
            else None
        )
        yield name, (x, residual, normalized_shape, weight, bias, eps)


def torch_op(x, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual


@pytest.mark.post_layer_norm_residual
def test_post_layer_norm_residual():
    from . import base, consts

    class PostLayerNormResidualBenchmark(base.Benchmark):
        def get_input_iter(self, dtype):
            for _, args in public_inputs(dtype, self.device):
                yield args

    bench = PostLayerNormResidualBenchmark(
        op_name="post_layer_norm_residual",
        torch_op=torch_op,
        gems_op=flag_gems.post_layer_norm_residual,
        dtypes=consts.FLOAT_DTYPES,
    )
    with torch.no_grad():
        bench.run()
