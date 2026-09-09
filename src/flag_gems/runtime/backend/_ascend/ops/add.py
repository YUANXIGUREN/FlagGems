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

import logging

import torch

from flag_gems.ops.add import add as _common_add

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _native_add(A, B, *, alpha=1):
    """Call the backend add implementation without re-entering FlagGems."""
    return torch.ops.aten.add.Tensor.redispatch(
        _FALLBACK_KEYSET, A, B, alpha=alpha
    )


def _can_use_native_fp32_add(A, B):
    """Select the inference subset validated for Ascend Native add."""
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        return False
    return (
        A.device.type == "npu"
        and B.device == A.device
        and A.dtype == torch.float32
        and B.dtype == torch.float32
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
        and not (torch.is_grad_enabled() and (A.requires_grad or B.requires_grad))
    )


def add(A, B, *, alpha=1):
    if _can_use_native_fp32_add(A, B):
        logger.debug("GEMS_ASCEND ADD NATIVE")
        return _native_add(A, B, alpha=alpha)
    return _common_add(A, B, alpha=alpha)
