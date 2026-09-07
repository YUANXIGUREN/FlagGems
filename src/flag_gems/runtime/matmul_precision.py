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
    """Return whether the selected Torch backend requests fast FP32 matmul."""

    if vendor_name == "ascend":
        return bool(_read_attr(torch, "npu", "matmul", "allow_hf32"))
    if vendor_name == "mthreads":
        return bool(_read_attr(torch, "backends", "mudnn", "allow_tf32"))

    cuda_matmul = _read_attr(torch, "backends", "cuda", "matmul")
    if vendor_name == "hygon":
        return bool(_read_attr(cuda_matmul, "allow_tf32"))
    if vendor_name not in {"nvidia", "metax", "thead"}:
        return False

    fp32_precision = _read_attr(cuda_matmul, "fp32_precision")
    if fp32_precision is not None:
        normalized_precision = str(fp32_precision).lower()
        if normalized_precision in {"tf32", "high", "medium"}:
            return True
        if normalized_precision in {"ieee", "highest"}:
            return False
        if normalized_precision not in {"", "none"}:
            return False

    allow_tf32 = _read_attr(cuda_matmul, "allow_tf32")
    if allow_tf32 is not None:
        return bool(allow_tf32)

    get_precision = _read_attr(torch, "get_float32_matmul_precision")
    if callable(get_precision):
        try:
            precision = get_precision()
        except (AttributeError, RuntimeError):
            return False
        return str(precision).lower() in {"high", "medium"}
    return False


def should_use_fast_float32_matmul(
    vendor_name: str,
    *operands: torch.Tensor,
) -> bool:
    """Limit fast-FP32 lowering to matrix operands that are all FP32."""

    return bool(operands) and all(
        operand.dtype == torch.float32 for operand in operands
    ) and is_fast_float32_matmul_enabled(vendor_name)
