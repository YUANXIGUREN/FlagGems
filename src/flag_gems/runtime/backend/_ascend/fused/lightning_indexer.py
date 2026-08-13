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

import logging

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice

logger = logging.getLogger(__name__)

pipe = tle.dsa.ascend.PIPE


@triton.autotune(configs=[triton.Config({'disable_auto_inject_block_sync': True, 'unit_flag': True})],
                 key=['K_TILE', 'N_CORE'])
@triton.jit
def lightning_indexer_tnd_pa_stage1_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    wsp_ptr,
    out_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    block_table_ptr,
    stride_qt,
    stride_qn,
    stride_kbn,
    stride_wt,
    stride_out_0,
    stride_block_table_b,
    block_size: tl.constexpr,
    query_head_num: tl.constexpr,
    head_dim: tl.constexpr,
    REQ_NUM: tl.constexpr,
    Q_TILE: tl.constexpr = 4,
    K_TILE: tl.constexpr = 64,
    M_CORE: tl.constexpr = 6,
    N_CORE: tl.constexpr = 4,
    m_coef: tl.constexpr = 2,
    sparse_mode: tl.constexpr = 3,
):
    # output: (total_query_seqs, max_tokens_num) 沿total_query_seqs分核
    core_id = tl.program_id(0)
    # TND在线分核初始化
    k_blk_step = block_size // K_TILE
    b = 0
    t_i = core_id // N_CORE * Q_TILE
    pre_len_q = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    seq_len_q = cur_len_q
    cur_len_k = tl.load(seq_lens_k_ptr)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    db_flag = 0
    while b < REQ_NUM:
        # 跨batch更新TND分核参数
        while t_i >= cur_len_q and b < REQ_NUM - 1:
            q_tail = seq_len_q % Q_TILE
            if q_tail:
                # 单batch不能均匀分核时，修正边界
                t_i -= Q_TILE - q_tail
            b += 1
            pre_len_q = cur_len_q
            cur_len_q = tl.load(seq_lens_q_ptr + b)
            seq_len_q = cur_len_q - pre_len_q
            cur_len_k = tl.load(seq_lens_k_ptr + b)
        if t_i >= cur_len_q:
            b = REQ_NUM
        else:
            # mask边界预计算
            act_len_k = cur_len_k - (cur_len_q - t_i) + 1
            k_blk_cnt = (act_len_k + K_TILE - 1) // K_TILE

            q_offsets = (t_i * stride_qt + tl.arange(0, Q_TILE * query_head_num)[:, None] * stride_qn +
                         tl.arange(0, head_dim)[None, :])
            q_block = tl.load(q_ptr + q_offsets)

            weight_offsets = t_i * stride_wt + tl.arange(0, query_head_num)

            # vector任务划分，各自处理Q_TILE的一半，vector权重预加载
            weight_block_0 = tl.load(weights_ptr + weight_offsets +
                                     (m_coef * tle.dsa.ascend.sub_vec_id() + 0) * stride_wt)[:, None]
            weight_block_1 = tl.load(weights_ptr + weight_offsets +
                                     (m_coef * tle.dsa.ascend.sub_vec_id() + 1) * stride_wt)[:, None]

            for k_i in range(core_id % N_CORE, k_blk_cnt, N_CORE):
                actual_k_block_i = tl.load(block_table_ptr + b * stride_block_table_b + k_i // k_blk_step)

                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + actual_k_block_i * stride_kbn + k_i % k_blk_step * K_TILE * head_dim,
                    shape=(K_TILE, head_dim),
                    strides=(head_dim, 1),
                    offsets=(0, 0),
                    block_shape=(K_TILE, head_dim),
                    order=(1, 0),
                )
                k_block = tl.load(k_block_ptr)
                qk_block = tl.dot(q_block, tl.trans(k_block))
                tle.dsa.ascend.sync_block_wait("vector", "cube", (db_flag % 2), pipe.PIPE_MTE2, pipe.PIPE_FIX)
                qk_block = libdevice.relu(qk_block)
                tl.store(
                    wsp_ptr + tl.num_programs(0) * Q_TILE * query_head_num * K_TILE * (db_flag % 2) +
                    core_id * Q_TILE * query_head_num * K_TILE + tl.arange(0, Q_TILE * query_head_num * K_TILE),
                    tl.reshape(qk_block, (Q_TILE * query_head_num * K_TILE, )),
                )

                tle.dsa.ascend.sync_block_set("cube", "vector", (db_flag % 2), pipe.PIPE_FIX, pipe.PIPE_MTE2)
                tle.dsa.ascend.sync_block_wait("cube", "vector", (db_flag % 2), pipe.PIPE_FIX, pipe.PIPE_MTE2)

                out_offsets = t_i * stride_out_0 + k_i * K_TILE + tl.arange(0, K_TILE)
                # 循环去除，分开拷贝，造vec与mte2流水并行，加快wsp释放，进一步缩减下一步的cube间隙
                qk_slice_0 = tl.load(wsp_ptr + tl.num_programs(0) * Q_TILE * query_head_num * K_TILE * (db_flag % 2) +
                                     core_id * Q_TILE * query_head_num * K_TILE +
                                     (m_coef * tle.dsa.ascend.sub_vec_id() + 0) * K_TILE * query_head_num +
                                     tl.arange(0, K_TILE * query_head_num))
                qk_slice_1 = tl.load(wsp_ptr + tl.num_programs(0) * Q_TILE * query_head_num * K_TILE * (db_flag % 2) +
                                     core_id * Q_TILE * query_head_num * K_TILE +
                                     (m_coef * tle.dsa.ascend.sub_vec_id() + 1) * K_TILE * query_head_num +
                                     tl.arange(0, K_TILE * query_head_num))

                tle.dsa.ascend.sync_block_set('vector', 'cube', (db_flag % 2), pipe.PIPE_MTE2, pipe.PIPE_FIX)

                tmp_reduce_res_block = tl.sum((tl.reshape(qk_slice_0, (query_head_num, K_TILE)) * weight_block_0), 0)
                if t_i + m_coef * tle.dsa.ascend.sub_vec_id() + 0 < cur_len_q:
                    tl.store(
                        out_ptr + out_offsets + (m_coef * tle.dsa.ascend.sub_vec_id() + 0) * stride_out_0,
                        tmp_reduce_res_block,
                        mask=k_i * K_TILE + tl.arange(0, K_TILE) < act_len_k + m_coef * tle.dsa.ascend.sub_vec_id() + 0,
                    )

                tmp_reduce_res_block = tl.sum((tl.reshape(qk_slice_1, (query_head_num, K_TILE)) * weight_block_1), 0)
                if t_i + m_coef * tle.dsa.ascend.sub_vec_id() + 1 < cur_len_q:
                    tl.store(
                        out_ptr + out_offsets + (m_coef * tle.dsa.ascend.sub_vec_id() + 1) * stride_out_0,
                        tmp_reduce_res_block,
                        mask=k_i * K_TILE + tl.arange(0, K_TILE) < act_len_k + m_coef * tle.dsa.ascend.sub_vec_id() + 1,
                    )

                db_flag += 1
            t_i += Q_TILE * M_CORE

    tle.dsa.ascend.sync_block_wait('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_wait('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)


@triton.jit
def lightning_indexer_tnd_pa_stage2_kernel(
    i_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    o_ptr,
    stride_in,
    stride_on,
    REQ_NUM: tl.constexpr,
    N_TILE: tl.constexpr,
    TOP_K: tl.constexpr = 2048,
):
    core_id = tl.program_id(0)
    k_blk_cnt = (TOP_K + N_TILE - 1) // N_TILE
    # TND在线分核初始化
    b = 0
    t_i = core_id
    cur_len_q = tl.load(seq_lens_q_ptr)
    cur_len_k = tl.load(seq_lens_k_ptr)

    while b < REQ_NUM:
        # 跨batch更新TND分核参数
        while t_i >= cur_len_q and b < REQ_NUM - 1:
            b += 1
            cur_len_q = tl.load(seq_lens_q_ptr + b)
            cur_len_k = tl.load(seq_lens_k_ptr + b)
        if t_i >= cur_len_q:
            b = REQ_NUM
        elif cur_len_k - (cur_len_q - t_i) + 1 < TOP_K:
            act_len_k = cur_len_k - (cur_len_q - t_i) + 1
            for k_i in range(k_blk_cnt):
                line_offset = k_i * N_TILE + tl.arange(0, N_TILE)
                i_block = tl.load(i_ptr + t_i * stride_in + line_offset).to(tl.int32)
                inner_block = tl.where(i_block < act_len_k, i_block, -1)
                tl.store(o_ptr + t_i * stride_on + line_offset, inner_block, mask=line_offset < TOP_K)
        else:
            act_len_k = cur_len_k - (cur_len_q - t_i) + 1
            for k_i in range(k_blk_cnt):
                line_offset = k_i * N_TILE + tl.arange(0, N_TILE)
                i_block = tl.load(i_ptr + t_i * stride_in + line_offset).to(tl.int32)
                tl.store(o_ptr + t_i * stride_on + line_offset, i_block, mask=line_offset < TOP_K)
        t_i += tl.num_programs(0)


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor = None,
    actual_seq_lengths_key: torch.Tensor = None,
    block_table: torch.Tensor = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    return_value: bool = False,
):
    total_query_seqs, query_head_num, head_dim = query.shape
    _, block_size, _, head_dim = key.shape
    max_tokens_num = int(actual_seq_lengths_key.max().item())

    out = torch.full(
        (total_query_seqs, max_tokens_num),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    USED_CORES = 24
    Q_TILE = 4
    K_TILE = 128
    wsp = torch.empty(
        (2 * USED_CORES * Q_TILE * query_head_num * K_TILE),
        dtype=torch.float32,
        device=query.device,
    )
    req_num = actual_seq_lengths_key.shape[0]

    N_CORE = 1
    M_CORE = USED_CORES // N_CORE

    lightning_indexer_tnd_pa_stage1_kernel[(USED_CORES, )](
        query,
        key,
        weights,
        wsp,
        out,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        weights.stride(0),
        out.stride(0),
        block_table.stride(0),
        block_size,
        query_head_num,
        head_dim,
        REQ_NUM=req_num,
        Q_TILE=Q_TILE,
        K_TILE=K_TILE,
        M_CORE=M_CORE,
        N_CORE=N_CORE,
        sparse_mode=sparse_mode,
    )

    out = torch.topk(out, sparse_count)
    out_indices = out.indices.reshape((-1, 1, sparse_count))
    ouptut = torch.empty(out_indices.shape, dtype=torch.int32, device=out.indices.device)
    N_TILE = sparse_count
    lightning_indexer_tnd_pa_stage2_kernel[(USED_CORES * 2, )](
        out_indices,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        ouptut,
        out_indices.stride(-2),
        ouptut.stride(-2),
        REQ_NUM=req_num,
        N_TILE=N_TILE,
        TOP_K=sparse_count,
    )
    if return_value:
        out_values = out.values.to(query.dtype).reshape((-1, 1, sparse_count))
        return ouptut, out_values
    else:
        return ouptut, None
