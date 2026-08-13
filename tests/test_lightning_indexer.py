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

from .conftest import QUICK_MODE

# Layout / shape constants fixed by the FlagTree reference (TND query / PA_BSND key)
QUERY_HEAD_NUM = 64
KV_HEAD_NUM = 1
HEAD_DIM = 128
BLOCK_SIZE = 128

# intersection-ratio threshold for the per-token top-k index set comparison;
# top-k order is not unique on score ties, so element-wise closeness does not apply.
INTERSECTION_RATIO_THRESHOLD = 0.95

if QUICK_MODE:
    # (batch, seq_q, seq_k, sparse_count); constraints: seq_q <= seq_k, seq_k >= sparse_count
    CASES = [(2, 256, 1024, 512), (2, 1024, 2048, 2048)]
else:
    CASES = [(4, 1024, 8192, 2048), (4, 4096, 8192, 2048), (8, 1024, 8192, 2048)]


def _gen_inputs(batch, seq_q, seq_k, sparse_count, device, dtype):
    total_q = seq_q * batch
    query = (torch.rand(total_q, QUERY_HEAD_NUM, HEAD_DIM, device=device, dtype=dtype) * 20 - 10)
    num_kv_blocks = batch * (seq_k // BLOCK_SIZE)
    key = (torch.rand(num_kv_blocks, BLOCK_SIZE, KV_HEAD_NUM, HEAD_DIM, device=device, dtype=dtype) * 20 - 10)
    weights = (torch.rand(total_q, QUERY_HEAD_NUM, device=device, dtype=dtype) * 2 - 1)

    # cumulative query lengths (TND layout) and per-batch key lengths (all equal here)
    actual_seq_lengths_query = torch.tensor(
        [seq_q * i for i in range(1, batch + 1)], dtype=torch.int32, device=device
    )
    actual_seq_lengths_key = torch.full((batch,), seq_k, dtype=torch.int32, device=device)

    # identity block table: block i maps to physical block i
    block_table = torch.arange(num_kv_blocks, dtype=torch.int32, device=device).reshape(batch, -1)
    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table


def _assert_set_similar(actual, expected):
    """Per-token top-k index set similarity; requires >= threshold intersection."""
    assert actual.shape == expected.shape
    batch_size = actual.shape[0]
    total_intersection = 0
    total_elements = 0
    for i in range(batch_size):
        actual_set = set(actual[i][0].cpu().numpy().tolist())
        expected_set = set(expected[i][0].cpu().numpy().tolist())
        intersection = actual_set & expected_set
        ratio = len(intersection) / len(expected_set)
        total_intersection += len(intersection)
        total_elements += len(expected_set)
        assert ratio >= INTERSECTION_RATIO_THRESHOLD, (
            f"Token {i}: set intersection ratio {ratio:.4f} < "
            f"{INTERSECTION_RATIO_THRESHOLD}"
        )
    overall_ratio = total_intersection / total_elements
    assert overall_ratio >= INTERSECTION_RATIO_THRESHOLD, (
        f"Overall set intersection ratio {overall_ratio:.4f} < "
        f"{INTERSECTION_RATIO_THRESHOLD}"
    )


@pytest.mark.skipif(
    not hasattr(flag_gems, "lightning_indexer"),
    reason="lightning_indexer is only implemented on the ascend backend",
)
@pytest.mark.skipif(
    not hasattr(torch_npu, "npu_lightning_indexer"),
    reason="golden torch_npu.npu_lightning_indexer is unavailable",
)
@pytest.mark.lightning_indexer
@pytest.mark.parametrize("batch,seq_q,seq_k,sparse_count", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_lightning_indexer(batch, seq_q, seq_k, sparse_count, dtype):
    device = flag_gems.device
    torch.manual_seed(3)
    query, key, weights, asl_q, asl_k, block_table = _gen_inputs(
        batch, seq_q, seq_k, sparse_count, device, dtype
    )

    indices, _ = flag_gems.lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query=asl_q,
        actual_seq_lengths_key=asl_k,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=3,
    )

    ref_indices, _ = torch_npu.npu_lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query=asl_q,
        actual_seq_lengths_key=asl_k,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=3,
    )

    _assert_set_similar(indices, ref_indices)
