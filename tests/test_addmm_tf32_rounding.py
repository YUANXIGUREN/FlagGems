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

from pathlib import Path

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.testing import assert_close
from flag_gems.utils import triton_lang_extension as ext

from .test_addmm_precision_policy import _float32_matmul_mode

pytestmark = pytest.mark.skipif(
    flag_gems.runtime.device.device_count == 0,
    reason="requires an accelerator device",
)

requires_rne_addmm = pytest.mark.skipif(
    flag_gems.vendor_name not in {"mthreads"},
    reason="explicit TF32 RNE addmm is enabled only for Moore Threads",
)


@triton.jit
def _round_bits_kernel(src, dst, N: tl.constexpr):
    offsets = tl.arange(0, 32)
    values = tl.load(src + offsets, offsets < N, 0).to(tl.float32, bitcast=True)
    rounded = ext.round_to_tf32(values)
    tl.store(dst + offsets, rounded.to(tl.uint32, bitcast=True), offsets < N)


def test_round_tf32_special_and_tie_values():
    pairs = [
        (0x00000000, 0x00000000),
        (0x80000000, 0x80000000),
        (0x7F800000, 0x7F800000),
        (0xFF800000, 0xFF800000),
        (0x7FC12345, 0x7FC12345),
        (0xFFC12345, 0xFFC12345),
        (0x3F801000, 0x3F800000),
        (0xBF801000, 0xBF800000),
        (0x3F803000, 0x3F804000),
        (0xBF803000, 0xBF804000),
        (0x3F801001, 0x3F802000),
        (0x3F800FFF, 0x3F800000),
        (0x3FFFFFFF, 0x40000000),
        (0x7F7FFFFF, 0x7F800000),
        (0x00001000, 0x00000000),
        (0x00001001, 0x00002000),
        (0x80001000, 0x80000000),
        (0x007FFFFF, 0x00800000),
    ]
    values = np.array([before for before, _ in pairs], dtype=np.uint32)
    expected = np.array([after for _, after in pairs], dtype=np.uint32)
    src = torch.from_numpy(values).to(flag_gems.device)
    dst = torch.empty_like(src)
    with torch_device_fn.device(src.device):
        _round_bits_kernel[(1,)](src, dst, len(pairs))

    np.testing.assert_array_equal(dst.cpu().numpy(), expected)


def test_mthreads_addmm_rounds_once_before_tiled_dot():
    source = (
        Path(__file__).parents[1]
        / "src/flag_gems/runtime/backend/_mthreads/ops/addmm.py"
    ).read_text(encoding="utf-8")

    assert "def round_to_tf32_copy(" in source
    assert "def _get_rounded_tf32_rhs(" in source
    assert "mat1 = round_to_tf32_copy(mat1)" in source
    assert "mat2 = _get_rounded_tf32_rhs(mat2)" in source
    kernel = source[source.index("def addmm_kernel(") : source.index("def addmm_fma(")]
    assert "ext.round_to_tf32(a)" not in kernel
    assert "ext.round_to_tf32(b)" not in kernel


@requires_rne_addmm
@pytest.mark.parametrize("column_major", [False, True])
def test_addmm_tf32_rounds_inputs(column_major):
    values = [
        1.0007,
        -1.0007,
        1.0002,
        -1.0002,
        1.00048828125,
        -1.00048828125,
        1.00146484375,
        -1.00146484375,
    ]
    expected = [
        1.0009765625,
        -1.0009765625,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.001953125,
        -1.001953125,
    ]
    mat1 = torch.tensor(values, device=flag_gems.device).repeat(64, 8)
    mat2 = torch.eye(64, device=flag_gems.device)
    if column_major:
        mat2 = mat2.T
    bias = torch.zeros(64, device=flag_gems.device)

    with _float32_matmul_mode(True), torch.inference_mode():
        result = flag_gems.addmm(bias, mat1, mat2).cpu()
    torch.testing.assert_close(
        result, torch.tensor(expected).repeat(64, 8), rtol=0, atol=0
    )

    with _float32_matmul_mode(False), torch.inference_mode():
        strict = flag_gems.addmm(bias, mat1, mat2).cpu()
    torch.testing.assert_close(strict, mat1.cpu(), rtol=0, atol=0)


def _oracle_round(values):
    mantissa, exponent = np.frexp(values.astype(np.float64))
    return np.ldexp(np.rint(mantissa * 2048), exponent - 11)


@requires_rne_addmm
@pytest.mark.parametrize("shape", [(64, 256, 184), (128, 512, 512)])
@pytest.mark.parametrize("out_api", [False, True])
def test_addmm_tf32_dense_against_rounded_fp64(shape, out_api):
    m, n, k = shape
    rng = np.random.default_rng(519)
    mat1 = rng.standard_normal((m, k)).astype(np.float32)
    mat2 = rng.standard_normal((k, n)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    expected = (
        0.75 * (_oracle_round(mat1) @ _oracle_round(mat2))
        - 0.5 * bias.astype(np.float64)
    )
    lhs = torch.from_numpy(mat1).to(flag_gems.device)
    rhs = torch.from_numpy(mat2.T.copy()).to(flag_gems.device).T
    inp = torch.from_numpy(bias).to(flag_gems.device)

    with _float32_matmul_mode(True), torch.inference_mode():
        if out_api:
            out = torch.empty((n, m), device=lhs.device).T
            actual = flag_gems.addmm_out(
                inp, lhs, rhs, alpha=0.75, beta=-0.5, out=out
            )
            assert actual.data_ptr() == out.data_ptr()
        else:
            actual = flag_gems.addmm(inp, lhs, rhs, alpha=0.75, beta=-0.5)

    assert_close(
        actual.cpu(), torch.from_numpy(expected), torch.float32, reduce_dim=k
    )
