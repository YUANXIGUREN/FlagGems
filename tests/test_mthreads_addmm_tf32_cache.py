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
import importlib.util
from pathlib import Path
import sys
import weakref

import torch


module_path = (
    Path(__file__).parents[1]
    / "src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py"
)
module_spec = importlib.util.spec_from_file_location(
    "_flaggems_test_mthreads_tf32_cache", module_path
)
assert module_spec is not None and module_spec.loader is not None
tf32_cache = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = tf32_cache
module_spec.loader.exec_module(tf32_cache)
TF32RHSCache = tf32_cache.TF32RHSCache


class CountingRounder:
    def __init__(self):
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        return tensor.contiguous().clone()


def test_cache_reuses_recreated_transpose_view_in_inference_mode():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    with torch.inference_mode():
        first = cache.get(weight.T, rounder)
        second = cache.get(weight.T, rounder)

    assert rounder.calls == 1
    assert first is second
    assert tuple(first.shape) == (6, 4)
    assert first.is_contiguous()


def test_cache_misses_after_inplace_weight_mutation():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.ones((4, 6), dtype=torch.float32)

    with torch.inference_mode():
        first = cache.get(weight.T, rounder)
        weight.add_(1)
        second = cache.get(weight.T, rounder)

    assert rounder.calls == 2
    assert first is not second
    torch.testing.assert_close(second, weight.T)


def test_cache_separates_view_metadata():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.arange(48, dtype=torch.float32).reshape(8, 6)

    with torch.inference_mode():
        full = cache.get(weight.T, rounder)
        sliced = cache.get(weight[::2].T, rounder)

    assert rounder.calls == 2
    assert tuple(full.shape) == (6, 8)
    assert tuple(sliced.shape) == (6, 4)


def test_cache_bypasses_grad_enabled_execution():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.ones((4, 6), dtype=torch.float32, requires_grad=True)

    first = cache.get(weight.T, rounder)
    second = cache.get(weight.T, rounder)

    assert rounder.calls == 2
    assert first is not second
    assert len(cache) == 0


def test_cache_evicts_least_recently_used_entry():
    cache = TF32RHSCache(max_entries=2)
    rounder = CountingRounder()
    weights = [torch.full((2, 2), value, dtype=torch.float32) for value in range(3)]

    with torch.inference_mode():
        first = cache.get(weights[0], rounder)
        cache.get(weights[1], rounder)
        assert cache.get(weights[0], rounder) is first
        cache.get(weights[2], rounder)
        cache.get(weights[1], rounder)

    assert rounder.calls == 4
    assert len(cache) == 2


def test_cache_releases_entry_when_base_tensor_dies():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.ones((4, 6), dtype=torch.float32)
    base_ref = weakref.ref(weight)

    with torch.inference_mode():
        cache.get(weight.T, rounder)
    assert len(cache) == 1

    del weight
    gc.collect()

    assert base_ref() is None
    assert len(cache) == 0
