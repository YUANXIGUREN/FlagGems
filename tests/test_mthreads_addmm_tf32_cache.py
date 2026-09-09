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

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
from pathlib import Path
import sys
import threading
import weakref

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py"
)
if MODULE_PATH.exists():
    MODULE_SPEC = importlib.util.spec_from_file_location(
        "_flaggems_test_mthreads_tf32_cache", MODULE_PATH
    )
    assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
    TF32_CACHE = importlib.util.module_from_spec(MODULE_SPEC)
    sys.modules[MODULE_SPEC.name] = TF32_CACHE
    MODULE_SPEC.loader.exec_module(TF32_CACHE)
    TF32RHSCache = TF32_CACHE.TF32RHSCache
else:
    TF32RHSCache = None


requires_cache_module = pytest.mark.skipif(
    TF32RHSCache is None, reason="TF32 RHS cache is not implemented"
)


def test_tf32_rhs_cache_module_exists():
    assert MODULE_PATH.exists()


class CountingRounder:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, tensor):
        with self.lock:
            self.calls += 1
        return tensor.contiguous().clone()


class FakeStorage:
    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self):
        return self.pointer


class FakeTensor:
    def __init__(self):
        self._base = None
        self._version = 0
        self.device = "cpu"
        self.dtype = torch.float32
        self.shape = (2, 3)
        self._stride = (3, 1)
        self._storage_offset = 0
        self._storage_pointer = 1000

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._storage_offset

    def untyped_storage(self):
        return FakeStorage(self._storage_pointer)


@requires_cache_module
def test_cache_reuses_reconstructed_transpose_view_in_inference_mode():
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


@requires_cache_module
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


@requires_cache_module
def test_cache_identity_includes_tensor_metadata_and_storage_pointer():
    cache = TF32RHSCache(max_entries=16)
    tensor = FakeTensor()
    calls = 0

    def rounder(value):
        nonlocal calls
        calls += 1
        return object()

    mutations = (
        ("shape", (3, 2)),
        ("_stride", (1, 2)),
        ("_storage_offset", 1),
        ("dtype", torch.float64),
        ("device", "meta"),
        ("_storage_pointer", 2000),
    )
    with torch.inference_mode():
        cache.get(tensor, rounder)
        for attribute, value in mutations:
            old_value = getattr(tensor, attribute)
            setattr(tensor, attribute, value)
            cache.get(tensor, rounder)
            setattr(tensor, attribute, old_value)

    assert calls == 1 + len(mutations)


@requires_cache_module
def test_cache_bypasses_grad_enabled_execution():
    cache = TF32RHSCache(max_entries=4)
    rounder = CountingRounder()
    weight = torch.ones((4, 6), dtype=torch.float32, requires_grad=True)

    first = cache.get(weight.T, rounder)
    second = cache.get(weight.T, rounder)

    assert rounder.calls == 2
    assert first is not second
    assert len(cache) == 0


@requires_cache_module
def test_cache_evicts_least_recently_used_entry():
    cache = TF32RHSCache(max_entries=2)
    rounder = CountingRounder()
    weights = [
        torch.full((2, 2), value, dtype=torch.float32) for value in range(3)
    ]

    with torch.inference_mode():
        first = cache.get(weights[0], rounder)
        cache.get(weights[1], rounder)
        assert cache.get(weights[0], rounder) is first
        cache.get(weights[2], rounder)
        cache.get(weights[1], rounder)

    assert rounder.calls == 4
    assert len(cache) == 2


@requires_cache_module
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


@requires_cache_module
def test_concurrent_gets_leave_valid_bounded_bookkeeping():
    cache = TF32RHSCache(max_entries=2)
    rounder = CountingRounder()
    weight = torch.ones((32, 32), dtype=torch.float32)

    def get_weight(_):
        with torch.inference_mode():
            return cache.get(weight.T, rounder)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(get_weight, range(32)))

    calls_after_workers = rounder.calls
    with torch.inference_mode():
        cached = cache.get(weight.T, rounder)
        cached_again = cache.get(weight.T, rounder)

    assert results
    assert len(cache) == 1
    assert rounder.calls == calls_after_workers
    assert cached is cached_again


@requires_cache_module
def test_cache_rejects_nonpositive_capacity():
    for capacity in (0, -1):
        try:
            TF32RHSCache(max_entries=capacity)
        except ValueError as error:
            assert "positive" in str(error)
        else:
            raise AssertionError("nonpositive cache capacity was accepted")
