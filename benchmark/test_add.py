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

import pytest
import torch

import flag_gems

from . import base, consts


class AddSuffixStridedBenchmark(base.BinaryPointwiseBenchmark):
    def set_more_shapes(self):
        return [(2, 31, 1), (2, 31, 13), (2, 31, 31)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) == 3 and shape[-1] in (1, 13, 31):
                storage_shape = (*shape[:-1], 176)
                inp1 = torch.randn(
                    storage_shape, dtype=dtype, device=self.device
                )[..., : shape[-1]]
                inp2 = torch.randn(
                    storage_shape, dtype=dtype, device=self.device
                )[..., : shape[-1]]
                yield inp1, inp2
            else:
                inp1 = base.generate_tensor_input(shape, dtype, self.device)
                inp2 = base.generate_tensor_input(shape, dtype, self.device)
                yield inp1, inp2


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_add():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend suffix-strided benchmark"
)
def test_add_suffix_strided():
    bench = AddSuffixStridedBenchmark(
        op_name="add",
        torch_op=torch.add,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.add_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_add_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="add_",
        torch_op=lambda a, b: a.add_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
