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

"""Read backend fast-FP32 policy without invoking or mutating backend compute."""

from typing import Any

import torch


__all__ = [
    "is_fast_float32_matmul_enabled",
    "should_use_fast_float32_matmul",
]


def _read_attr(root: Any, *names: str) -> Any | None:
    value = root
    for name in names:
        if value is None:
            return None
        try:
            value = getattr(value, name)
        except (AttributeError, RuntimeError):
            return None
    return value


def is_fast_float32_matmul_enabled(vendor_name: str) -> bool:
    """Read the requested fast-FP32 policy, defaulting to strict mode."""

    if vendor_name == "ascend":
        return bool(_read_attr(torch, "npu", "matmul", "allow_hf32"))
    if vendor_name == "mthreads":
        return bool(_read_attr(torch, "backends", "mudnn", "allow_tf32"))
    if vendor_name == "hygon":
        return bool(_read_attr(torch, "backends", "cuda", "matmul", "allow_tf32"))
    return False


def should_use_fast_float32_matmul(
    vendor_name: str,
    *operands: torch.Tensor,
) -> bool:
    """Return true only for enabled fast mode with all-FP32 operands."""

    return bool(operands) and all(
        operand.dtype == torch.float32 for operand in operands
    ) and is_fast_float32_matmul_enabled(vendor_name)
