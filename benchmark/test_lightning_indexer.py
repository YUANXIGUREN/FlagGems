# Copyright 2026- Xcoresigma Technology Co., Ltd
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
import torch_npu

import flag_gems

from . import base

# Layout / shape constants fixed by the FlagTree reference (TND query / PA_BSND key).
QUERY_HEAD_NUM = 64
KV_HEAD_NUM = 1
HEAD_DIM = 128
BLOCK_SIZE = 128

# (batch, seq_q, seq_k, sparse_count); constraints: seq_q <= seq_k, seq_k >= sparse_count.
# Mirrors the accuracy cases in tests/test_lightning_indexer.py.
INDEXER_SHAPES = [
    (4, 1024, 8192, 2048),
    (4, 4096, 8192, 2048),
    (8, 1024, 8192, 2048),
]


class LightningIndexerBenchmark(base.GenericBenchmark):
    """lightning_indexer has no generic M/N shape; iterate over its own configs."""

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in INDEXER_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def lightning_indexer_kwargs(shape, dtype, device):
    batch, seq_q, seq_k, sparse_count = shape
    total_q = seq_q * batch
    query = torch.rand(total_q, QUERY_HEAD_NUM, HEAD_DIM, device=device, dtype=dtype) * 20 - 10
    num_kv_blocks = batch * (seq_k // BLOCK_SIZE)
    key = torch.rand(num_kv_blocks, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM, device=device, dtype=dtype) * 20 - 10
    weights = torch.rand(total_q, QUERY_HEAD_NUM, device=device, dtype=dtype) * 2 - 1

    # cumulative query lengths (TND layout) and per-batch equal key lengths.
    actual_seq_lengths_query = torch.tensor(
        [seq_q * i for i in range(1, batch + 1)], dtype=torch.int32, device=device
    )
    actual_seq_lengths_key = torch.full((batch,), seq_k, dtype=torch.int32, device=device)
    # identity block table: block i maps to physical block i.
    block_table = torch.arange(num_kv_blocks, dtype=torch.int32, device=device).reshape(batch, -1)

    kwargs = {
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "block_table": block_table,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": sparse_count,
        "sparse_mode": 3,
    }
    yield query, key, weights, kwargs


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="lightning_indexer is only implemented on the ascend backend",
)
@pytest.mark.skipif(
    not hasattr(flag_gems, "lightning_indexer"),
    reason="flag_gems.lightning_indexer is unavailable",
)
@pytest.mark.skipif(
    not hasattr(torch_npu, "npu_lightning_indexer"),
    reason="golden torch_npu.npu_lightning_indexer is unavailable",
)
@pytest.mark.lightning_indexer
def test_lightning_indexer():
    bench = LightningIndexerBenchmark(
        op_name="lightning_indexer",
        input_fn=lightning_indexer_kwargs,
        torch_op=torch_npu.npu_lightning_indexer,
        gems_op=flag_gems.lightning_indexer,
        dtypes=[torch.bfloat16],
    )
    bench.run()
