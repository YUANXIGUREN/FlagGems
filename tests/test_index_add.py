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

import gc
import importlib
import random
import time
from contextlib import nullcontext

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
else:
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


CONTIGUOUS_SUFFIX_CASES = [
    ((1024, 4), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
    ((2, 8, 2048, 32), 2),
    ((1024, 64), 0),
]


def _make_repeated_index(index_len):
    index_range = max(index_len // 2, 1)
    return torch.arange(index_len, device=flag_gems.device) % index_range


def _run_torch_index_add(inp, dim, index, src, inplace, alpha=1):
    if inplace:
        result = inp.index_add_(dim, index, src, alpha=alpha)
        assert result is inp
        return result
    return torch.index_add(inp, dim, index, src, alpha=alpha)


def _run_flag_gems_index_add(inp, dim, index, src, inplace, alpha=1):
    if inplace:
        result = flag_gems.index_add_(inp, dim, index, src, alpha=alpha)
        assert result is inp
        return result
    return flag_gems.index_add(inp, dim, index, src, alpha=alpha)


def _get_active_index_add_module():
    module = importlib.import_module(flag_gems.index_add.__module__)
    if module.index_add is not flag_gems.index_add:
        raise AssertionError("resolved a duplicate index_add backend module")
    return module


_INDEX_ADD_FIX_IS_ACTIVE = (
    flag_gems.index_add.__module__ == "flag_gems.ops.index_add"
    or flag_gems.vendor_name in ("metax", "mthreads")
)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_empty_index(inplace, dtype):
    inp = torch.randn((2, 7, 17), dtype=dtype, device=flag_gems.device)
    src = torch.empty((2, 0, 17), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    if inplace:
        assert result is inp
    else:
        assert result.data_ptr() != inp.data_ptr()
        assert result.is_contiguous()
    utils.gems_assert_equal(result, ref_inp)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_empty_index_noncontiguous_input(inplace, dtype):
    inp = torch.randn((7, 2, 17), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    src = torch.empty((2, 0, 17), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    original_stride = inp.stride()

    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    if inplace:
        assert result is inp
        assert result.stride() == original_stride
    else:
        assert result.data_ptr() != inp.data_ptr()
        assert result.is_contiguous()
    utils.gems_assert_equal(result, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_negative_dim_contiguous_suffix(inplace):
    shape = (2, 7, 17)
    dim = -2
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    src = torch.ones((2, 4, 17), dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    ref_result = _run_torch_index_add(ref_inp, dim, ref_index, ref_src, inplace)
    result = _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dim, src_shape, index_values",
    [
        (3, (4, 7, 17), [0, 1, 1, 0]),
        (-4, (2, 7, 4), [0, 2, 2, 6]),
    ],
)
def test_index_add_invalid_dim(inplace, dim, src_shape, index_values):
    inp = torch.zeros((2, 7, 17), dtype=torch.float32, device=flag_gems.device)
    src = torch.ones(src_shape, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor(index_values, device=flag_gems.device)
    original = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    # Some vendor Torch backends surface their native invalid-dimension error
    # as RuntimeError. The FlagGems public contract remains the normalized
    # IndexError asserted below.
    with pytest.raises((IndexError, RuntimeError)):
        _run_torch_index_add(ref_inp, dim, ref_index, ref_src, inplace)
    with pytest.raises(IndexError):
        _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, original)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "inp_noncontiguous, src_noncontiguous",
    [(True, False), (False, True)],
    ids=["noncontiguous-input", "noncontiguous-source"],
)
def test_index_add_noncontiguous_input_and_source(
    inplace, inp_noncontiguous, src_noncontiguous
):
    inp = torch.arange(
        2 * 7 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 7, 17)
    src = torch.arange(
        2 * 4 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 4, 17)
    if inp_noncontiguous:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)
    if src_noncontiguous:
        src = src.transpose(0, 1).contiguous().transpose(0, 1)

    assert inp.is_contiguous() == (not inp_noncontiguous)
    assert src.is_contiguous() == (not src_noncontiguous)
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src.clone())
    ref_index = utils.to_reference(index)
    ref_result = _run_torch_index_add(ref_inp, 1, ref_index, ref_src, inplace)
    original_stride = inp.stride()

    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    if inplace:
        assert result is inp
        assert inp.stride() == original_stride
    else:
        assert result.is_contiguous()
    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add_
def test_index_add_inplace_input_source_alias():
    inp = torch.arange(
        2 * 7 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 7, 17)
    index = torch.tensor([0, 2, 2, 4, 4, 6, 0], device=flag_gems.device)
    original = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp.clone())
    ref_index = utils.to_reference(index)

    with pytest.raises(RuntimeError):
        ref_inp.index_add_(1, ref_index, ref_inp)
    utils.gems_assert_equal(ref_inp, original)

    with pytest.raises(RuntimeError):
        flag_gems.index_add_(inp, 1, index, inp)
    utils.gems_assert_equal(inp, original)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add_
def test_index_add_inplace_empty_source_alias_is_noop():
    inp = torch.arange(
        2 * 7 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 7, 17)
    src = inp[:, :0, :]
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    original = utils.to_reference(inp.clone())

    result = flag_gems.index_add_(inp, 1, index, src)

    assert result is inp
    utils.gems_assert_equal(inp, original)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add_
def test_index_add_inplace_empty_exact_alias_raises():
    inp = torch.empty((2, 0, 17), device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="overlap"):
        flag_gems.index_add_(inp, 1, index, inp)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_zero_work_skips_bounds_check(inplace):
    inp = torch.empty((0, 3), device=flag_gems.device)
    src = torch.empty((0, 1), device=flag_gems.device)
    index = torch.tensor([99], dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    if inplace:
        assert result is inp
    else:
        assert not torch._C._is_alias_of(result, inp)
        assert result.is_contiguous()
    utils.gems_assert_equal(result, ref_inp)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_empty_index_still_validates_shape(inplace):
    inp = torch.empty((2, 7, 17), device=flag_gems.device)
    src = torch.empty((3, 0, 17), device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(AssertionError, match=r"src\.size\(d\).*d != dim"):
        _run_flag_gems_index_add(inp, 1, index, src, inplace)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add_
def test_index_add_inplace_index_alias_raises():
    inp = torch.arange(4, dtype=torch.int32, device=flag_gems.device)
    index = inp[:2]
    src = torch.ones((2,), dtype=inp.dtype, device=flag_gems.device)
    original = utils.to_reference(inp.clone())

    with pytest.raises(RuntimeError, match="overlap"):
        flag_gems.index_add_(inp, 0, index, src)

    utils.gems_assert_equal(inp, original)


@pytest.mark.skipif(
    not _INDEX_ADD_FIX_IS_ACTIVE,
    reason="common/MetaX/MThreads index_add contract regression",
)
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_accepts_scalar_index(inplace):
    inp = torch.zeros((2, 3), device=flag_gems.device)
    src = torch.ones((2, 1), device=flag_gems.device)
    index = torch.tensor(1, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_result = _run_torch_index_add(ref_inp, 1, ref_index, ref_src, inplace)

    result = _run_flag_gems_index_add(inp, 1, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_partial_contiguous_suffix(inplace, dtype):
    shape = (2, 257, 513)
    dim = 1
    index_len = 129
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    src = torch.ones((2, index_len, 513), dtype=dtype, device=flag_gems.device)
    index = torch.arange(index_len, device=flag_gems.device) % 65
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    ref_result = _run_torch_index_add(
        ref_inp, dim, ref_index, ref_src, inplace, alpha=2
    )
    result = _run_flag_gems_index_add(inp, dim, index, src, inplace, alpha=2)

    utils.gems_assert_close(result, ref_result, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_inplace_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(inp, ref_inp)


# Randomized, non-exactly-representable values: zeros/ones/alpha=2 cannot
# reveal rounding differences between fp32-accumulate-then-cast and native
# bf16 accumulation. A deterministic seed keeps the loose-tolerance native
# parity check stable across runs.
CONTIGUOUS_SUFFIX_STRESS_CASES = [
    ((2, 8, 2048, 32), 2),  # flat path, narrow suffix
    ((2, 8, 2048, 72), 2),  # flat path, partially filled second tile
    ((2, 8, 2048, 256), 2),  # tile path, wide suffix
    ((1024, 64), 0),  # dim == 0 always uses the flat path
]


def _make_dup_index(index_len, dup_factor):
    receiver_range = max(index_len // dup_factor, 1)
    return torch.arange(index_len, device=flag_gems.device) % receiver_range


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_STRESS_CASES)
@pytest.mark.parametrize("dup_factor", [2, 32])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_contiguous_suffix_randomized(shape, dim, dup_factor, inplace):
    torch.manual_seed(2024 + dup_factor * 8 + (1 if inplace else 0))
    dtype = torch.bfloat16
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_len = shape[dim] // 2
    index = _make_dup_index(index_len, dup_factor)
    src_shape = list(shape)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 0.7

    # Native bf16-accumulate reference (what the vendor torch does).
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    if inplace:
        ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    else:
        ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)

    # High-precision reference models accumulate then cast once, which is the
    # contract of the optimized bf16 fallback. Per-add bf16 rounding would
    # fail this check at dup_factor=32.
    exact_inp = utils.to_reference(inp, upcast=True)
    exact_src = utils.to_reference(src, upcast=True)
    exact_out = torch.index_add(exact_inp, dim, ref_index, exact_src, alpha=alpha)

    result = _run_flag_gems_index_add(inp, dim, index, src, inplace, alpha=alpha)

    target = ref_inp if inplace else ref_out
    utils.gems_assert_close(result, exact_out, dtype=dtype, reduce_dim=1, atol=0.05)
    utils.gems_assert_close(result, target, dtype=dtype, reduce_dim=1, atol=0.5)


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX-specific routing policy"
)
@pytest.mark.parametrize(
    "dim, suffix_size, expected_route",
    [
        (1, 64, "flat"),
        (1, 65, "flat"),
        (1, 79, "flat"),
        (1, 80, "tile"),
        (1, 512, "tile"),
        (0, 256, "flat"),
    ],
)
def test_index_add_metax_contiguous_suffix_route(
    monkeypatch, dim, suffix_size, expected_route
):
    metax_index_add = _get_active_index_add_module()
    selected = []

    def record_flat(out, dim, index, src, alpha, use_fp16_config=False):
        assert not use_fp16_config
        selected.append("flat")
        return out

    def record_tile(out, dim, index, src, alpha):
        selected.append("tile")
        return out

    monkeypatch.setattr(
        metax_index_add, "_run_contiguous_suffix_flat_path", record_flat
    )
    monkeypatch.setattr(
        metax_index_add, "_run_contiguous_suffix_tile_path", record_tile
    )
    if dim == 0:
        inp = torch.empty(
            (2, suffix_size), dtype=torch.float32, device=flag_gems.device
        )
        src = torch.empty(
            (1, suffix_size), dtype=torch.float32, device=flag_gems.device
        )
    else:
        inp = torch.empty(
            (1, 2, suffix_size), dtype=torch.float32, device=flag_gems.device
        )
        src = torch.empty(
            (1, 1, suffix_size), dtype=torch.float32, device=flag_gems.device
        )
    index = torch.zeros((1,), dtype=torch.int64, device=flag_gems.device)

    flag_gems.index_add(inp, dim, index, src)

    assert selected == [expected_route]


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX-specific FP16 flat config"
)
def test_index_add_metax_fp16_flat_uses_safe_config(monkeypatch):
    metax_index_add = _get_active_index_add_module()
    configs = metax_index_add.runtime.get_tuned_config(
        "index_add_contiguous_suffix_fp16_flat"
    )
    assert [config.kwargs["BLOCK_SIZE"] for config in configs] == [128]

    launches = []

    class KernelSpy:
        def __getitem__(self, grid):
            assert grid({"BLOCK_SIZE": 128}) == (8192,)

            def launch(*args, **kwargs):
                launches.append((args, kwargs))

            return launch

    class ForbiddenKernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                pytest.fail(f"unexpected {self.name} kernel")

            return launch

    monkeypatch.setattr(
        metax_index_add,
        "_index_add_contiguous_suffix_fp16_flat_kernel",
        KernelSpy(),
    )
    monkeypatch.setattr(
        metax_index_add,
        "_index_add_contiguous_suffix_flat_kernel",
        ForbiddenKernel("generic flat"),
    )
    monkeypatch.setattr(
        metax_index_add,
        "_index_add_contiguous_suffix_tile_kernel",
        ForbiddenKernel("tile"),
    )

    inp = torch.zeros((4096, 256), dtype=torch.float16, device=flag_gems.device)
    src = torch.zeros_like(inp)
    index = torch.arange(4096, dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    result = flag_gems.index_add(inp, 0, index, src)

    assert len(launches) == 1
    args, kwargs = launches[0]
    assert args[3:7] == (4096 * 256, 4096, 4096, 256)
    assert "ACCUMULATE_FP32" not in kwargs
    utils.gems_assert_equal(result, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_invalid_index(inplace):
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, shape[dim]], device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_index(inplace, index_dtype):
    shape = (1, 2, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.stack(
        (
            torch.ones((1, 8), device=flag_gems.device),
            torch.full((1, 8), 2.0, device=flag_gems.device),
        ),
        dim=1,
    )
    raw_index = torch.tensor([-1, 0], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    assert index.is_neg()

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = torch.tensor([1, 0], dtype=index_dtype, device=ref_inp.device)
    ref_result = _run_torch_index_add(ref_inp, dim, ref_index, ref_src, inplace)

    result = _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_index_fallback(inplace, index_dtype):
    shape = (1, 2, 8)
    dim = 1
    inp = torch.zeros(
        (shape[0], shape[2], shape[1]), device=flag_gems.device
    ).transpose(1, 2)
    src = torch.stack(
        (
            torch.ones((1, 8), device=flag_gems.device),
            torch.full((1, 8), 2.0, device=flag_gems.device),
        ),
        dim=2,
    ).transpose(1, 2)
    assert not inp.is_contiguous()
    assert not src.is_contiguous()

    raw_index = torch.tensor([-1, 0], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    assert index.is_neg()

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = torch.tensor([1, 0], dtype=index_dtype, device=ref_inp.device)
    ref_result = _run_torch_index_add(ref_inp, dim, ref_index, ref_src, inplace)

    result = _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_oob_does_not_write_input(
    inplace, index_dtype, fallback
):
    shape = (1, 2, 8)
    dim = 1
    if fallback:
        storage_shape = (shape[0], shape[2], shape[1])
        inp = torch.zeros(storage_shape, device=flag_gems.device).transpose(1, 2)
        src = torch.ones(storage_shape, device=flag_gems.device).transpose(1, 2)
        assert not inp.is_contiguous()
        assert not src.is_contiguous()
    else:
        inp = torch.zeros(shape, device=flag_gems.device)
        src = torch.ones(shape, device=flag_gems.device)
    # The physical values are valid, while the lazy-negative logical value -1
    # is invalid. This distinguishes correct materialization from accidentally
    # validating the un-negated storage.
    raw_index = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    original = utils.to_reference(inp.clone())

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, original)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name not in ("metax", "mthreads"),
    reason="vendor fast paths formerly cached index bounds",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_revalidates_index_after_data_mutation(monkeypatch, inplace):
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, 1], device=flag_gems.device)

    _run_flag_gems_index_add(inp, dim, index, src, inplace)
    before_invalid_call = utils.to_reference(inp.clone())

    version = index._version
    index.data[-1] = shape[dim]
    assert index._version == version

    vendor_index_add = _get_active_index_add_module()

    def fail_if_kernel_is_reached(*args, **kwargs):
        pytest.fail("kernel reached before mutated index was revalidated")

    monkeypatch.setattr(
        vendor_index_add, "_run_contiguous_suffix_path", fail_if_kernel_is_reached
    )
    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        _run_flag_gems_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, before_invalid_call)


def _make_mthreads_trusted_root_case(out_dim=8192):
    index_len = 4096
    suffix = 128
    inp = torch.zeros(
        (1, out_dim, suffix),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    index_root = torch.arange(
        index_len * 2,
        dtype=torch.int64,
        device=flag_gems.device,
    ) // 2
    index_views = (
        index_root[:index_len],
        index_root[index_len:],
    )
    src = torch.ones(
        (1, index_len, suffix),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    return inp, index_root, index_views, src


def _record_mthreads_device_bounds_checks(monkeypatch):
    vendor_index_add = _get_active_index_add_module()
    original = vendor_index_add._index_is_in_bounds_on_device
    reads = []

    def record_bounds_read(index, upper_bound):
        reads.append((index.numel(), upper_bound))
        return original(index, upper_bound)

    monkeypatch.setattr(
        vendor_index_add,
        "_index_is_in_bounds_on_device",
        record_bounds_read,
    )
    return vendor_index_add, reads


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add inference contract",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_default_scans_each_view(monkeypatch, inplace):
    inp, _, index_views, src = _make_mthreads_trusted_root_case()
    _, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with flag_gems.use_gems(include=["index_add", "index_add_"]):
            for view_id, index in enumerate(index_views):
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )
                expected = torch.zeros_like(result)
                receiver_start = view_id * 2048
                expected[
                    :, receiver_start : receiver_start + 2048, :
                ] = 2.0
                utils.gems_assert_equal(result, expected)

    assert reads == [(4096, 8192), (4096, 8192)]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add inference contract",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_scope_validates_root_once(
    monkeypatch, inplace
):
    inp, _, index_views, src = _make_mthreads_trusted_root_case()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                for view_id, index in enumerate(index_views):
                    result = _run_torch_index_add(
                        inp.clone(), 1, index, src, inplace
                    )
                    expected = torch.zeros_like(result)
                    receiver_start = view_id * 2048
                    expected[
                        :, receiver_start : receiver_start + 2048, :
                    ] = 2.0
                    utils.gems_assert_equal(result, expected)

    assert reads == [(8192, 8192)]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add complete-root validation",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_scope_rejects_invalid_outside_first_view(
    monkeypatch, inplace
):
    inp, index_root, index_views, src = _make_mthreads_trusted_root_case()
    index_root[-1] = inp.size(1)
    before = inp.clone()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                with pytest.raises(
                    AssertionError,
                    match=r"0 <= index < self\.size\(dim\)",
                ):
                    _run_torch_index_add(
                        inp, 1, index_views[0], src, inplace
                    )

    assert reads == [(8192, 8192)]
    utils.gems_assert_equal(inp, before)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add mutation fallback",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_scope_falls_back_after_tracked_mutation(
    monkeypatch, inplace
):
    inp, index_root, index_views, src = _make_mthreads_trusted_root_case()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                _run_torch_index_add(
                    inp.clone(), 1, index_views[0], src, inplace
                )
                version = index_root._version
                index_root[0] = 1
                assert index_root._version > version
                _run_torch_index_add(
                    inp.clone(), 1, index_views[0], src, inplace
                )
                _run_torch_index_add(
                    inp.clone(), 1, index_views[1], src, inplace
                )

    assert reads == [(8192, 8192), (4096, 8192), (4096, 8192)]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add upper-bound fallback",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_scope_requires_exact_upper_bound(
    monkeypatch, inplace
):
    inp, _, index_views, src = _make_mthreads_trusted_root_case()
    larger_inp = torch.zeros(
        (1, 8193, 128),
        dtype=inp.dtype,
        device=inp.device,
    )
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                _run_torch_index_add(
                    inp.clone(), 1, index_views[0], src, inplace
                )
                _run_torch_index_add(
                    larger_inp, 1, index_views[1], src, inplace
                )

    assert reads == [(8192, 8192), (4096, 8193)]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add scope cleanup",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_trusted_scope_exit_restores_default(
    monkeypatch, inplace
):
    inp, _, index_views, src = _make_mthreads_trusted_root_case()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.inference_mode():
        with flag_gems.use_gems(include=["index_add", "index_add_"]):
            with vendor_index_add.use_trusted_index_add_inference():
                _run_torch_index_add(
                    inp.clone(), 1, index_views[0], src, inplace
                )
            _run_torch_index_add(
                inp.clone(), 1, index_views[1], src, inplace
            )

    assert reads == [(8192, 8192), (4096, 8192)]


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add weak-root cache cleanup",
)
def test_index_add_mthreads_trusted_scope_evicts_dead_roots_and_retries_slot():
    vendor_index_add = _get_active_index_add_module()
    upper_bound = 8192

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            roots = vendor_index_add._TRUSTED_INDEX_ROOTS.get()
            old_root = torch.arange(
                upper_bound,
                dtype=torch.int64,
                device=flag_gems.device,
            ) // 2
            stale_entry = vendor_index_add._trusted_index_entry(
                old_root, upper_bound
            )
            old_root_id = id(old_root)
            del old_root
            gc.collect()

            assert stale_entry.root_ref() is None
            assert old_root_id not in roots

            new_root = torch.arange(
                upper_bound,
                dtype=torch.int64,
                device=flag_gems.device,
            ) // 2
            roots[id(new_root)] = stale_entry
            replacement_entry = vendor_index_add._trusted_index_entry(
                new_root, upper_bound
            )

            assert replacement_entry is not None
            assert replacement_entry.root_ref() is new_root
            assert roots[id(new_root)] is replacement_entry


@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add tracked-OOB safety",
)
def test_index_add_mthreads_trusted_scope_tracked_oob_preserves_inplace_input(
    monkeypatch,
):
    inp, index_root, index_views, src = _make_mthreads_trusted_root_case()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add_"]):
                _run_torch_index_add(inp, 1, index_views[0], src, True)
                before_invalid_call = inp.clone()
                index_root[0] = inp.size(1)
                with pytest.raises(
                    AssertionError,
                    match=r"0 <= index < self\.size\(dim\)",
                ):
                    _run_torch_index_add(inp, 1, index_views[0], src, True)

    assert reads == [(8192, 8192), (4096, 8192)]
    utils.gems_assert_equal(inp, before_invalid_call)


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted index_add root/storage replacement safety",
)
@pytest.mark.parametrize("replacement_kind", ["storage", "root"])
def test_index_add_mthreads_trusted_scope_revalidates_replaced_root(
    monkeypatch, replacement_kind
):
    inp, index_root, index_views, src = _make_mthreads_trusted_root_case()
    vendor_index_add, reads = _record_mthreads_device_bounds_checks(monkeypatch)

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add"]):
                _run_torch_index_add(
                    inp.clone(), 1, index_views[0], src, False
                )
                replacement = torch.arange(
                    index_root.numel(),
                    dtype=index_root.dtype,
                    device=index_root.device,
                ) // 2
                if replacement_kind == "storage":
                    del index_views
                    index_root.set_(replacement)
                else:
                    index_root = replacement
                replacement_view = index_root[: src.size(1)]
                _run_torch_index_add(
                    inp.clone(), 1, replacement_view, src, False
                )

    expected_second_read = 4096 if replacement_kind == "storage" else 8192
    assert reads == [(8192, 8192), (expected_second_read, 8192)]


def _make_mthreads_exact_three_run_case(
    *, index_dtype=torch.int64, prefix=1, view_offset=0, index_len=4096
):
    run_count = max(4100, (view_offset + index_len + 2) // 3)
    out_dim = max(8192, run_count + 1)
    suffix = 128
    index_root = torch.arange(
        run_count * 3,
        dtype=index_dtype,
        device=flag_gems.device,
    ) // 3
    index = index_root[view_offset : view_offset + index_len]
    inp = torch.full(
        (prefix, out_dim, suffix),
        0.25,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    prefix_values = torch.arange(
        1,
        prefix + 1,
        dtype=torch.float32,
        device=flag_gems.device,
    ).reshape(prefix, 1, 1)
    src = torch.ones(
        (prefix, index_len, suffix),
        dtype=torch.float32,
        device=flag_gems.device,
    ) * prefix_values
    return inp, index_root, index, src


def _make_mthreads_exact_three_run_nd_case(prefix_shape, index_len=65538):
    run_count = (index_len + 2) // 3
    out_dim = run_count + 1
    suffix = 128
    index_root = torch.arange(
        run_count * 3,
        dtype=torch.int64,
        device=flag_gems.device,
    ) // 3
    index = index_root[:index_len]
    inp = torch.randn(
        (*prefix_shape, out_dim, suffix),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    src = torch.randn(
        (*prefix_shape, index_len, suffix),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    return inp, index_root, index, src, len(prefix_shape)


def _expected_mthreads_exact_three_run_result(inp, index, src, alpha):
    counts = torch.bincount(
        index.to(torch.int64),
        minlength=inp.size(1),
    ).to(inp.dtype)
    source_values = src[:, 0, 0].reshape(src.size(0), 1, 1)
    return inp + alpha * source_values * counts.reshape(1, -1, 1)


def _record_mthreads_sorted_run_launches(
    monkeypatch, index_min_elements=4096
):
    vendor_index_add = _get_active_index_add_module()
    original = vendor_index_add._run_sorted_run_path
    launches = []

    if index_min_elements is not None:
        monkeypatch.setattr(
            vendor_index_add,
            "_SORTED_RUN_INDEX_MIN_ELEMENTS",
            index_min_elements,
        )

    def record_sorted_run_launch(out, dim, index, src, alpha, entry):
        launches.append(
            (
                index.storage_offset() - entry.start,
                index.numel(),
                src.shape[-1],
            )
        )
        return original(out, dim, index, src, alpha, entry)

    monkeypatch.setattr(
        vendor_index_add,
        "_run_sorted_run_path",
        record_sorted_run_launch,
    )
    return vendor_index_add, launches


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted exact-three sorted runs",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "prefix,view_offset,alpha",
    [
        (1, 0, 1.0),
        (1, 1, -0.5),
        (2, 2, 0.25),
    ],
)
def test_index_add_mthreads_sorted_runs_handles_root_aligned_views(
    monkeypatch,
    inplace,
    index_dtype,
    prefix,
    view_offset,
    alpha,
):
    inp, _, index, src = _make_mthreads_exact_three_run_case(
        index_dtype=index_dtype,
        prefix=prefix,
        view_offset=view_offset,
    )
    expected = _expected_mthreads_exact_three_run_result(
        inp, index, src, alpha
    )
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(),
                    1,
                    index,
                    src,
                    inplace,
                    alpha=alpha,
                )

    utils.gems_assert_close(result, expected, dtype=torch.float32)
    assert launches == [(view_offset, index.numel(), src.shape[-1])]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run production threshold",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "index_len,should_launch",
    [(65535, False), (65538, True)],
)
def test_index_add_mthreads_sorted_runs_production_threshold(
    monkeypatch, inplace, index_len, should_launch
):
    inp, _, index, src = _make_mthreads_exact_three_run_case(
        index_len=index_len
    )
    expected = _run_torch_index_add(
        inp.clone(), 1, index, src, inplace
    )
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch,
        index_min_elements=None,
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )

    utils.gems_assert_close(
        result,
        expected,
        dtype=torch.float32,
        reduce_dim=1,
    )
    expected_launches = (
        [(0, index.numel(), src.shape[-1])] if should_launch else []
    )
    assert launches == expected_launches


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run multiple-prefix dimensions",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("prefix_shape", [(2, 1), (1, 2, 1)])
def test_index_add_mthreads_sorted_runs_support_multiple_prefix_dimensions(
    monkeypatch, inplace, prefix_shape
):
    inp, _, index, src, dim = _make_mthreads_exact_three_run_nd_case(
        prefix_shape
    )
    expected = _run_torch_index_add(
        inp.clone(), dim, index, src, inplace, alpha=-0.5
    )
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch,
        index_min_elements=None,
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(),
                    dim,
                    index,
                    src,
                    inplace,
                    alpha=-0.5,
                )

    utils.gems_assert_close(
        result,
        expected,
        dtype=torch.float32,
        reduce_dim=dim,
    )
    assert launches == [(0, index.numel(), src.shape[-1])]


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run unsupported-case fallback",
)
@pytest.mark.parametrize("fallback_kind", ["layout", "alias"])
def test_index_add_mthreads_sorted_runs_targeted_fallbacks(
    monkeypatch, fallback_kind
):
    shape = (1, 8192, 128)
    index_len = 4096
    dtype = torch.float32
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if fallback_kind == "layout":
        index_storage = torch.zeros(
            index_len * 2,
            dtype=torch.int64,
            device=flag_gems.device,
        )
        index_storage[::2] = torch.arange(
            index_len,
            dtype=torch.int64,
            device=flag_gems.device,
        ) // 3
        index = index_storage[::2]
    else:
        index = torch.arange(
            index_len,
            dtype=torch.int64,
            device=flag_gems.device,
        ) // 3
    if fallback_kind == "alias":
        src = inp[:, :index_len, :]
    else:
        src = torch.randn(
            (1, index_len, shape[-1]),
            dtype=dtype,
            device=flag_gems.device,
        )
    expected = torch.index_add(inp, 1, index, src)
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add"]):
                result = torch.index_add(inp, 1, index, src)

    utils.gems_assert_close(
        result,
        expected,
        dtype=dtype,
        reduce_dim=1,
    )
    assert launches == []


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run shared floating-point implementation",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_index_add_mthreads_sorted_runs_share_float_dtype_algorithm(
    monkeypatch, inplace, dtype
):
    inp, _, index, src = _make_mthreads_exact_three_run_case()
    inp = inp.to(dtype)
    src = src.to(dtype)
    expected = _expected_mthreads_exact_three_run_result(
        inp, index, src, 0.25
    )
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(),
                    1,
                    index,
                    src,
                    inplace,
                    alpha=0.25,
                )

    utils.gems_assert_close(result, expected, dtype=dtype)
    assert launches == [(0, index.numel(), src.shape[-1])]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run random accumulation accuracy",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("view_offset", [0, 1, 2])
def test_index_add_mthreads_sorted_runs_match_native_random_source(
    monkeypatch, inplace, view_offset
):
    inp, _, index, src = _make_mthreads_exact_three_run_case(
        prefix=2,
        view_offset=view_offset,
    )
    inp = torch.randn_like(inp)
    src = torch.randn_like(src)
    alpha = -0.75
    expected = _run_torch_index_add(
        inp.clone(),
        1,
        index,
        src,
        inplace,
        alpha=alpha,
    )
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(),
                    1,
                    index,
                    src,
                    inplace,
                    alpha=alpha,
                )

    utils.gems_assert_close(
        result,
        expected,
        dtype=torch.float32,
        reduce_dim=1,
    )
    assert launches == [(view_offset, index.numel(), src.shape[-1])]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run conservative fallback",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("pattern", ["unordered", "unequal", "overlong"])
def test_index_add_mthreads_sorted_runs_rejects_nonuniform_roots(
    monkeypatch, inplace, pattern
):
    inp, index_root, index, src = _make_mthreads_exact_three_run_case()
    if pattern == "unordered":
        index_root[3:6] = 2
        index_root[6:9] = 1
    elif pattern == "unequal":
        index_root[4] = 2
    else:
        index_root[3] = index_root[2]
    expected = _expected_mthreads_exact_three_run_result(inp, index, src, 1)
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )

    utils.gems_assert_close(result, expected, dtype=torch.float32)
    assert launches == []


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run opt-in isolation",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_sorted_runs_require_trusted_scope(
    monkeypatch, inplace
):
    inp, _, index, src = _make_mthreads_exact_three_run_case()
    expected = _expected_mthreads_exact_three_run_result(inp, index, src, 1)
    _, launches = _record_mthreads_sorted_run_launches(monkeypatch)

    with torch.no_grad():
        with flag_gems.use_gems(include=["index_add", "index_add_"]):
            result = _run_torch_index_add(
                inp.clone(), 1, index, src, inplace
            )

    utils.gems_assert_close(result, expected, dtype=torch.float32)
    assert launches == []


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run measured crossover",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_sorted_runs_preserve_small_atomic_path(
    monkeypatch, inplace
):
    inp, _, index, src = _make_mthreads_exact_three_run_case()
    expected = _expected_mthreads_exact_three_run_result(inp, index, src, 1)
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch,
        index_min_elements=None,
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )

    utils.gems_assert_close(result, expected, dtype=torch.float32)
    assert launches == []


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads sorted-run tracked mutation fallback",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_sorted_runs_reject_stale_metadata(
    monkeypatch, inplace
):
    inp, index_root, index, src = _make_mthreads_exact_three_run_case()
    vendor_index_add, launches = _record_mthreads_sorted_run_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                _run_torch_index_add(inp.clone(), 1, index, src, inplace)
                index_root[0] = 1
                expected = _expected_mthreads_exact_three_run_result(
                    inp, index, src, 1
                )
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )

    utils.gems_assert_close(result, expected, dtype=torch.float32)
    assert launches == [(0, index.numel(), src.shape[-1])]


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads large cases must stay on FlagGems",
)
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mthreads_large_nonuniform_stays_on_flaggems(
    monkeypatch, inplace
):
    shape = (1, 8192, 128)
    index_len = 4096
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    index = torch.arange(
        index_len, dtype=torch.int64, device=flag_gems.device
    ) % 2048
    src = torch.randn(
        (1, index_len, shape[-1]),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    expected = _run_torch_index_add(
        inp.clone(), 1, index, src, inplace
    )
    vendor_index_add = _get_active_index_add_module()
    original = vendor_index_add._run_contiguous_suffix_path
    launches = []

    def record_flaggems_launch(out, dim, call_index, call_src, alpha):
        launches.append((call_index.numel(), call_src.shape[-1]))
        return original(out, dim, call_index, call_src, alpha)

    monkeypatch.setattr(
        vendor_index_add,
        "_run_contiguous_suffix_path",
        record_flaggems_launch,
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                result = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace
                )

    utils.gems_assert_close(
        result,
        expected,
        dtype=torch.float32,
        reduce_dim=1,
    )
    assert launches == [(index_len, shape[-1])]


def _make_mthreads_unsorted_segment_case(
    *, index_len=4096, receiver_count=256, dtype=torch.float32
):
    out_dim = receiver_count * 2
    suffix = 128
    positions = torch.arange(
        index_len,
        dtype=torch.int64,
        device=flag_gems.device,
    )
    index_root = (positions * 37) % receiver_count
    inp = torch.randn(
        (1, out_dim, suffix),
        dtype=dtype,
        device=flag_gems.device,
    )
    src = torch.randn(
        (1, index_len, suffix),
        dtype=dtype,
        device=flag_gems.device,
    )
    return inp, index_root, src


def _record_mthreads_segmented_launches(
    monkeypatch, index_min_elements=1
):
    vendor_index_add = _get_active_index_add_module()
    original_build = vendor_index_add._build_segmented_index_view
    original_run = vendor_index_add._run_segmented_path
    builds = []
    launches = []

    if index_min_elements is not None:
        monkeypatch.setattr(
            vendor_index_add,
            "_SEGMENTED_INDEX_MIN_ELEMENTS",
            index_min_elements,
        )
    monkeypatch.setattr(vendor_index_add, "_SEGMENTED_MIN_MEAN_RUN", 2.0)

    def record_build(index, entry):
        builds.append((index.storage_offset() - entry.start, index.numel()))
        return original_build(index, entry)

    def record_run(out, dim, index, src, alpha, segmented):
        launches.append(
            (index.storage_offset(), index.numel(), segmented.receivers.numel())
        )
        return original_run(out, dim, index, src, alpha, segmented)

    monkeypatch.setattr(vendor_index_add, "_build_segmented_index_view", record_build)
    monkeypatch.setattr(vendor_index_add, "_run_segmented_path", record_run)
    return vendor_index_add, builds, launches


@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads segmented reduction production threshold",
)
@pytest.mark.parametrize(
    "index_len,should_launch",
    [(32767, False), (32768, True)],
)
def test_index_add_mthreads_segmented_production_threshold(
    monkeypatch, index_len, should_launch
):
    inp, index, src = _make_mthreads_unsorted_segment_case(
        index_len=index_len
    )
    expected = inp.clone().index_add_(1, index, src)
    vendor_index_add, _, launches = _record_mthreads_segmented_launches(
        monkeypatch,
        index_min_elements=None,
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add_"]):
                result = inp.clone().index_add_(1, index, src)

    utils.gems_assert_close(
        result, expected, dtype=torch.float32, reduce_dim=1
    )
    expected_launches = (
        [(0, index.numel(), 256)] if should_launch else []
    )
    assert launches == expected_launches


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads trusted unsorted segmented reduction",
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_index_add_mthreads_segmented_unsorted_matches_native_and_caches_view(
    monkeypatch, inplace, dtype
):
    inp, index, src = _make_mthreads_unsorted_segment_case(dtype=dtype)
    expected = _run_torch_index_add(
        inp.clone(), 1, index, src, inplace, alpha=-0.75
    )
    vendor_index_add, builds, launches = _record_mthreads_segmented_launches(
        monkeypatch
    )

    with torch.no_grad():
        with vendor_index_add.use_trusted_index_add_inference():
            with flag_gems.use_gems(include=["index_add", "index_add_"]):
                first = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace, alpha=-0.75
                )
                second = _run_torch_index_add(
                    inp.clone(), 1, index, src, inplace, alpha=-0.75
                )

    tolerance = {
        torch.float16: 0.04,
        torch.bfloat16: 0.3,
        torch.float32: 1e-4,
    }[dtype]
    torch.testing.assert_close(
        first.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        second.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance
    )
    assert builds == [(0, index.numel())]
    assert launches == [
        (0, index.numel(), 256),
        (0, index.numel(), 256),
    ]


@pytest.mark.index_add_
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="MThreads segmented reduction conservative dispatch",
)
@pytest.mark.parametrize("fallback", ["small", "low-duplication", "no-trust"])
def test_index_add_mthreads_segmented_path_has_conservative_fallbacks(
    monkeypatch, fallback
):
    receiver_count = 4096 if fallback == "low-duplication" else 256
    inp, index, src = _make_mthreads_unsorted_segment_case(
        receiver_count=receiver_count,
        dtype=torch.float32,
    )
    expected = inp.clone().index_add_(1, index, src)
    vendor_index_add, _, launches = _record_mthreads_segmented_launches(monkeypatch)
    if fallback == "small":
        monkeypatch.setattr(
            vendor_index_add,
            "_SEGMENTED_INDEX_MIN_ELEMENTS",
            index.numel() + 1,
        )

    context = (
        nullcontext()
        if fallback == "no-trust"
        else vendor_index_add.use_trusted_index_add_inference()
    )
    with torch.no_grad():
        with context:
            with flag_gems.use_gems(include=["index_add_"]):
                result = inp.clone().index_add_(1, index, src)

    utils.gems_assert_close(
        result, expected, dtype=torch.float32, reduce_dim=1
    )
    assert launches == []
