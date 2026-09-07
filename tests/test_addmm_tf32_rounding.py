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

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

pytestmark = pytest.mark.skipif(
    flag_gems.runtime.device.device_count == 0,
    reason="requires an accelerator device",
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
