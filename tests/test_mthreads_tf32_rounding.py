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

from contextlib import contextmanager
import importlib

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext


SPECIAL_AND_TIE_CASES = (
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
)


def tf32_rne_oracle(bits):
    """Round one binary32 encoding to a TF32 encoding using integer RNE."""

    bits = int(bits)
    if bits & 0x7F800000 == 0x7F800000:
        return bits
    rounded = bits + 0x00000FFF + ((bits >> 13) & 1)
    return rounded & 0xFFFFE000


def test_tf32_rne_oracle_handles_special_values_and_halfway_ties():
    actual = [tf32_rne_oracle(before) for before, _ in SPECIAL_AND_TIE_CASES]
    expected = [after for _, after in SPECIAL_AND_TIE_CASES]

    assert actual == expected


requires_s5000 = pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads"
    or flag_gems.runtime.device.device_count == 0,
    reason="requires a Moore Threads accelerator",
)


@contextmanager
def _mthreads_tf32(enabled):
    backend = torch.backends.mudnn
    previous = backend.allow_tf32
    backend.allow_tf32 = enabled
    try:
        yield
    finally:
        backend.allow_tf32 = previous


@triton.jit
def _round_bits_kernel(src, dst, n_elements: tl.constexpr):
    offsets = tl.arange(0, 1024)
    values = tl.load(src + offsets, offsets < n_elements, 0).to(
        tl.float32, bitcast=True
    )
    rounded = ext.round_to_tf32(values)
    tl.store(
        dst + offsets,
        rounded.to(tl.uint32, bitcast=True),
        offsets < n_elements,
    )


@requires_s5000
def test_s5000_triton_rounding_matches_independent_bit_oracle():
    rng = np.random.default_rng(20260909)
    random_bits = rng.integers(0, 2**32, size=512, dtype=np.uint32)
    inputs = np.concatenate(
        [
            np.array([before for before, _ in SPECIAL_AND_TIE_CASES]),
            random_bits,
        ]
    ).astype(np.uint32)
    expected = np.array([tf32_rne_oracle(value) for value in inputs], np.uint32)
    src = torch.from_numpy(inputs).to(flag_gems.device)
    dst = torch.empty_like(src)

    with torch_device_fn.device(src.device):
        _round_bits_kernel[(1,)](src, dst, len(inputs))

    np.testing.assert_array_equal(dst.cpu().numpy(), expected)


@requires_s5000
def test_round_to_tf32_copy_materializes_logical_noncontiguous_values():
    backend_addmm = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    source = torch.tensor(
        [
            1.0007,
            -1.0007,
            1.0002,
            -1.0002,
            1.00048828125,
            -1.00048828125,
            1.00146484375,
            -1.00146484375,
        ],
        device=flag_gems.device,
        dtype=torch.float32,
    ).reshape(2, 4).t()
    logical_bits = source.cpu().contiguous().numpy().view(np.uint32)
    expected_bits = np.vectorize(tf32_rne_oracle, otypes=[np.uint32])(
        logical_bits
    )

    rounded = backend_addmm.round_to_tf32_copy(source)

    assert rounded.is_contiguous()
    np.testing.assert_array_equal(
        rounded.cpu().numpy().view(np.uint32), expected_bits
    )


@requires_s5000
def test_round_to_tf32_fp16_copy_matches_musa_truncation_for_representable_values():
    backend_addmm = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    source = torch.tensor(
        [
            1.0007,
            -1.0007,
            1.0002,
            -1.0002,
            1.00048828125,
            -1.00048828125,
            1.00146484375,
            -1.00146484375,
        ],
        device=flag_gems.device,
        dtype=torch.float32,
    ).reshape(2, 4).t()
    expected = torch.tensor(
        [
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            -1.0,
            1.0009765625,
            -1.0009765625,
        ],
        device=flag_gems.device,
        dtype=torch.float32,
    ).reshape(2, 4).t()

    actual = backend_addmm.round_to_tf32_fp16_copy(source)

    assert actual.dtype == torch.float16
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)


@requires_s5000
def test_fast_fp32_pointer_addmm_stays_close_to_native_tf32():
    backend_addmm = importlib.import_module(
        "flag_gems.runtime.backend._mthreads.ops.addmm"
    )
    values = torch.tensor(
        [
            1.0007,
            -1.0007,
            1.0002,
            -1.0002,
            1.00048828125,
            -1.00048828125,
            1.00146484375,
            -1.00146484375,
        ],
        dtype=torch.float32,
    )
    mat1 = values.repeat(64, 8).to(flag_gems.device)
    mat2 = torch.eye(64, dtype=torch.float32, device=flag_gems.device)
    bias = torch.zeros(64, dtype=torch.float32, device=flag_gems.device)

    with _mthreads_tf32(True), torch.inference_mode():
        native = torch.addmm(bias, mat1, mat2).cpu()
        fast = backend_addmm.addmm_fma(bias, mat1, mat2).cpu()
    with _mthreads_tf32(False), torch.inference_mode():
        strict = backend_addmm.addmm_fma(bias, mat1, mat2).cpu()

    torch.testing.assert_close(fast, native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(strict, mat1.cpu(), rtol=0, atol=0)
