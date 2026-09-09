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

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Callable
import weakref

import torch


@dataclass
class _CacheEntry:
    base_ref: weakref.ReferenceType
    storage_ptr: int
    version: int
    rounded: torch.Tensor


class TF32RHSCache:
    """Bounded, version-aware cache for rounded inference RHS views."""

    def __init__(self, max_entries: int = 512):
        """Create an empty bounded cache."""

        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _base_tensor(tensor: torch.Tensor) -> torch.Tensor:
        base = tensor
        while isinstance(getattr(base, "_base", None), torch.Tensor):
            base = base._base
        return base

    @staticmethod
    def _version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except RuntimeError:
            return None

    @staticmethod
    def _storage_ptr(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor.untyped_storage().data_ptr())
        except RuntimeError:
            return None

    @staticmethod
    def _key(base: torch.Tensor, tensor: torch.Tensor):
        return (
            id(base),
            str(tensor.device),
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
        )

    def _discard_dead(self, key, base_ref):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.base_ref is base_ref:
                self._entries.pop(key, None)

    def get(
        self,
        tensor: torch.Tensor,
        rounder: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Return a valid cached tensor or round and cache the input."""

        if not torch.is_inference_mode_enabled():
            return rounder(tensor)

        base = self._base_tensor(tensor)
        version = self._version(base)
        storage_ptr = self._storage_ptr(tensor)
        if version is None or storage_ptr is None:
            return rounder(tensor)

        key = self._key(base, tensor)
        with self._lock:
            entry = self._entries.get(key)
            if (
                entry is not None
                and entry.base_ref() is base
                and entry.storage_ptr == storage_ptr
                and entry.version == version
            ):
                self._entries.move_to_end(key)
                return entry.rounded

        rounded = rounder(tensor)
        with self._lock:
            base_ref = weakref.ref(
                base,
                lambda ref, cache=self, entry_key=key: cache._discard_dead(
                    entry_key, ref
                ),
            )
            self._entries[key] = _CacheEntry(
                base_ref=base_ref,
                storage_ptr=storage_ptr,
                version=version,
                rounded=rounded,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return rounded

    def clear(self) -> None:
        """Drop every entry."""

        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Return the number of live entries."""

        with self._lock:
            return len(self._entries)
