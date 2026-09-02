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
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.index_add import _validate_index_add_args
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"
_NATIVE_INDEX_MIN_ELEMENTS = 4096
_NATIVE_SUFFIX_MIN_ELEMENTS = 128
_SORTED_RUN_INDEX_MIN_ELEMENTS = 65536
_NATIVE_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_TRUSTED_INDEX_ROOTS = ContextVar(
    "mthreads_trusted_index_add_roots",
    default=None,
)


def _evict_dead_trusted_index_root(roots, root_id, dead_ref):
    entry = roots.get(root_id)
    if entry is not None and entry.root_ref is dead_ref:
        roots.pop(root_id, None)


class _TrustedIndexRoot:
    __slots__ = (
        "root_ref",
        "storage_ptr",
        "start",
        "end",
        "dtype",
        "device",
        "upper_bound",
        "version",
        "uniform_run_length",
    )

    def __init__(
        self,
        root,
        upper_bound,
        version,
        uniform_run_length,
        roots,
        root_id,
    ):
        self.root_ref = weakref.ref(
            root,
            lambda dead_ref: _evict_dead_trusted_index_root(
                roots, root_id, dead_ref
            ),
        )
        self.storage_ptr = root.untyped_storage().data_ptr()
        self.start = root.storage_offset()
        self.end = self.start + root.numel()
        self.dtype = root.dtype
        self.device = root.device
        self.upper_bound = upper_bound
        self.version = version
        self.uniform_run_length = uniform_run_length


@contextmanager
def use_trusted_index_add_inference():
    """Validate immutable index roots once inside an inference scope.

    The caller promises that roots are not changed through ``.data``, storage
    aliases, DMA, or external writers while this context is active. Ordinary
    tracked PyTorch mutations invalidate the entry and restore per-call bounds
    validation.
    """

    token = _TRUSTED_INDEX_ROOTS.set({})
    try:
        yield
    finally:
        _TRUSTED_INDEX_ROOTS.reset(token)


def _read_index_bounds(index):
    return index.min().item(), index.max().item()


@libentry()
@triton.jit
def _index_bounds_kernel(
    index,
    invalid,
    index_len,
    upper_bound,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_len
    values = tl.load(index + offsets * index_stride, mask=mask, other=0)
    out_of_bounds = mask & ((values < 0) | (values >= upper_bound))
    invalid_ptrs = invalid + tl.zeros((BLOCK_SIZE,), tl.int32)
    ones = tl.full((BLOCK_SIZE,), 1, tl.int32)
    tl.atomic_xchg(invalid_ptrs, ones, mask=out_of_bounds)


def _native_index_is_in_bounds(index, upper_bound):
    logical_index = _resolve_index_for_kernel(index)
    invalid = torch.zeros((), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(logical_index.numel(), 1024),)
    with torch_device_fn.device(index.device):
        _index_bounds_kernel[grid](
            logical_index,
            invalid,
            logical_index.numel(),
            upper_bound,
            logical_index.stride(0),
            BLOCK_SIZE=1024,
        )
    return invalid.item() == 0


@libentry()
@triton.jit
def _index_uniform_run_length_three_kernel(
    index,
    invalid,
    index_len,
    index_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < index_len
    values = tl.load(index + offsets * index_stride, mask=mask, other=0)
    previous_offsets = tl.maximum(offsets - 1, 0)
    previous = tl.load(
        index + previous_offsets * index_stride,
        mask=mask,
        other=0,
    )
    position = offsets % 3
    unequal_inside_run = (position != 0) & (values != previous)
    unordered_run_start = (
        (position == 0) & (offsets > 0) & (values <= previous)
    )
    invalid_pattern = mask & (unequal_inside_run | unordered_run_start)
    invalid_ptrs = invalid + tl.zeros((BLOCK_SIZE,), tl.int32)
    ones = tl.full((BLOCK_SIZE,), 1, tl.int32)
    tl.atomic_xchg(invalid_ptrs, ones, mask=invalid_pattern)


def _native_index_uniform_run_length(index):
    if index.numel() == 0 or index.numel() % 3 != 0:
        return None

    invalid = torch.zeros((), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), 1024),)
    with torch_device_fn.device(index.device):
        _index_uniform_run_length_three_kernel[grid](
            index,
            invalid,
            index.numel(),
            index.stride(0),
            BLOCK_SIZE=1024,
        )
    return 3 if invalid.item() == 0 else None


def _resolve_index_for_kernel(index):
    # A contiguous lazy-negative tensor still exposes the un-negated storage
    # to a pointer-based Triton kernel. Materialize only that exceptional case.
    # Calling resolve_neg() from inside use_gems() re-enters FlagGems' Python
    # override and can negate the logical value twice. Toggle the metadata bit
    # off first, then explicitly negate the ordinary physical view.
    if index.is_neg():
        return torch.neg(torch._neg_view(index))
    return index


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() == 0:
        return
    idx_min, idx_max = _read_index_bounds(index)
    if idx_min < 0 or idx_max >= upper_bound:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _assert_native_index_in_bounds(index, upper_bound):
    if not _native_index_is_in_bounds(index, upper_bound):
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _resolve_trusted_index_root(index):
    if (
        index.is_neg()
        or index.ndim != 1
        or index.dtype not in (torch.int32, torch.int64)
        or not index.is_contiguous()
    ):
        return None

    root = index
    visited = set()
    while root._base is not None:
        root_id = id(root)
        if root_id in visited:
            return None
        visited.add(root_id)
        root = root._base

    if (
        root.ndim != 1
        or root.dtype != index.dtype
        or root.device != index.device
        or not root.is_contiguous()
    ):
        return None
    return root


def _trusted_view_matches_entry(index, root, entry, upper_bound, version):
    if (
        entry.root_ref() is not root
        or entry.upper_bound != upper_bound
        or entry.version != version
        or entry.dtype != index.dtype
        or entry.device != index.device
        or entry.storage_ptr != index.untyped_storage().data_ptr()
    ):
        return False

    start = index.storage_offset()
    end = start + index.numel()
    return entry.start <= start and end <= entry.end


def _trusted_index_entry(index, upper_bound):
    roots = _TRUSTED_INDEX_ROOTS.get()
    if roots is None or torch.is_grad_enabled():
        return None

    root = _resolve_trusted_index_root(index)
    if root is None:
        return None

    version = _tensor_version(root)
    if version is None:
        return None

    root_id = id(root)
    entry = roots.get(root_id)
    if entry is not None:
        if entry.root_ref() is root:
            if _trusted_view_matches_entry(
                index, root, entry, upper_bound, version
            ):
                return entry
            return None
        roots.pop(root_id, None)

    if not _native_index_is_in_bounds(root, upper_bound):
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)

    validated_version = _tensor_version(root)
    if validated_version is None or validated_version != version:
        return None

    uniform_run_length = None
    if root.numel() >= _SORTED_RUN_INDEX_MIN_ELEMENTS:
        uniform_run_length = _native_index_uniform_run_length(root)
    metadata_version = _tensor_version(root)
    if metadata_version is None or metadata_version != validated_version:
        return None

    entry = _TrustedIndexRoot(
        root,
        upper_bound,
        metadata_version,
        uniform_run_length,
        roots,
        root_id,
    )
    roots[root_id] = entry
    if not _trusted_view_matches_entry(
        index,
        root,
        entry,
        upper_bound,
        metadata_version,
    ):
        return None
    return entry


def _trusted_index_can_skip_bounds(index, upper_bound):
    return _trusted_index_entry(index, upper_bound) is not None


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    return (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and inp.dtype in (torch.float16, torch.float32)
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
        and _volume(src.shape[dim + 1 :]) > 1
    )


def _should_redispatch_native(inp, dim, index, src):
    # Native MUSA is faster once both the scatter length and contiguous suffix
    # are large; smaller or non-FP32 inputs keep using the custom kernels.
    return (
        _can_use_contiguous_suffix_path(inp, dim, index, src)
        and inp.dtype == torch.float32
        and not (
            torch.is_grad_enabled() and (inp.requires_grad or src.requires_grad)
        )
        and index.numel() >= _NATIVE_INDEX_MIN_ELEMENTS
        and _volume(src.shape[dim + 1 :]) >= _NATIVE_SUFFIX_MIN_ELEMENTS
        and not torch._C._is_alias_of(inp, src)
        and not torch._C._is_alias_of(inp, index)
    )


def _native_index_add(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add.default.redispatch(
        _NATIVE_FALLBACK_KEYSET,
        inp,
        dim,
        index,
        src,
        alpha=alpha,
    )


def _native_index_add_(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add_.default.redispatch(
        _NATIVE_FALLBACK_KEYSET,
        inp,
        dim,
        index,
        src,
        alpha=alpha,
    )


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel for index_add operation with autotune.

    After dim_compress, tensors are reshaped so that:
    - inp has shape (M, inp_len) where inp_len is the size of target dimension
    - src has shape (M, N) where N is the size of index

    For each row m and each index position n:
        out[m, index[n]] += alpha * src[m, n]
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    # Calculate row and column offsets
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks
    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    # Load indices for this block of columns
    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    # Calculate offsets into inp/out (which has shape M x inp_len)
    inp_off = rows_offset * inp_len + cur_indices

    # Calculate offsets into src (which has shape M x N)
    src_off = rows_offset * N + cols_offset

    # Load source values
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    # Use atomic_add to correctly handle repeated indices in index,
    # aligned with the common op (src/flag_gems/ops/index_add.py).
    # When multiple source elements map to the same output position (duplicate
    # indices), plain load-store would cause race conditions or lost updates.
    # atomic_add guarantees all contributions are accumulated correctly.
    tl.atomic_add(out_ptr + inp_off, alpha * cur_src, mask=block_mask)


@libentry()
@triton.jit
def _index_add_contiguous_suffix_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    mask = row_mask & (cols < suffix_size)
    edge = rows % index_len
    prefix = rows // index_len
    receiver = tl.load(index + edge, mask=row_mask, other=0).to(tl.int64)
    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask)


def _contiguous_suffix_config(suffix_size):
    block_n = min(512, triton.next_power_of_2(suffix_size))
    return 4, block_n


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    block_m, block_n = _contiguous_suffix_config(suffix_size)
    grid = (
        triton.cdiv(row_count, block_m),
        triton.cdiv(suffix_size, block_n),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return out


@libentry()
@triton.jit
def _index_add_sorted_run_kernel(
    out,
    index,
    src,
    index_len,
    run_count,
    out_dim,
    suffix_size,
    first_run_size,
    alpha,
    BLOCK_N: tl.constexpr,
):
    linear_run = ext.program_id(0)
    prefix = linear_run // run_count
    run = linear_run % run_count
    cols = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    edge = tl.where(
        run == 0,
        0,
        first_run_size + (run - 1) * 3,
    )
    current_run_size = tl.where(run == 0, first_run_size, 3)
    valid_run = edge < index_len
    receiver = tl.load(index + edge, mask=valid_run, other=0).to(tl.int64)
    col_mask = cols < suffix_size

    values = tl.zeros((BLOCK_N,), tl.float32)
    for run_offset in tl.static_range(0, 3):
        source_edge = edge + run_offset
        source_mask = (
            valid_run
            & (run_offset < current_run_size)
            & (source_edge < index_len)
            & col_mask
        )
        source_offsets = (
            (
                prefix.to(tl.int64) * index_len
                + source_edge.to(tl.int64)
            )
            * suffix_size
            + cols
        )
        values += tl.load(
            src + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)

    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    mask = valid_run & col_mask
    previous = tl.load(out + out_offsets, mask=mask, other=0.0)
    tl.store(out + out_offsets, previous + values * alpha, mask=mask)


def _should_use_sorted_run_path(entry, index):
    return (
        entry is not None
        and entry.uniform_run_length == 3
        and index.numel() >= _SORTED_RUN_INDEX_MIN_ELEMENTS
    )


def _run_sorted_run_path(out, dim, index, src, alpha, entry):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_count = _volume(src.shape[:dim])
    view_alignment = (index.storage_offset() - entry.start) % 3
    first_run_size = 3 - view_alignment if view_alignment else 3
    run_count = triton.cdiv(index.numel() + view_alignment, 3)
    block_n = min(512, triton.next_power_of_2(suffix_size))
    grid = (
        prefix_count * run_count,
        triton.cdiv(suffix_size, block_n),
    )
    with torch_device_fn.device(out.device):
        _index_add_sorted_run_kernel[grid](
            out,
            index,
            src,
            index.numel(),
            run_count,
            out.size(dim),
            suffix_size,
            first_run_size,
            alpha,
            BLOCK_N=block_n,
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    """
    Optimized index_add for mthreads backend.

    self.index_add_(dim, index, source, alpha=1) -> Tensor

    For a 3-D tensor the output is:
        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src.numel() == 0:
        return inp.clone(memory_format=torch.contiguous_format)
    if _should_redispatch_native(inp, dim, index, src):
        trusted_entry = _trusted_index_entry(index, inp.size(dim))
        if trusted_entry is None:
            _assert_native_index_in_bounds(index, inp.size(dim))
        if _should_use_sorted_run_path(trusted_entry, index):
            out = inp.clone(memory_format=torch.contiguous_format)
            return _run_sorted_run_path(
                out, dim, index, src, alpha, trusted_entry
            )
        return _native_index_add(inp, dim, index, src, alpha)

    use_contiguous_suffix_path = _can_use_contiguous_suffix_path(
        inp, dim, index, src
    ) and not torch._C._is_alias_of(inp, src)

    # Make inputs contiguous. resolve_neg() is a no-op for normal indices.
    inp = inp.contiguous()
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    _assert_index_in_bounds(index, inp_len)

    if use_contiguous_suffix_path:
        out = inp.clone()
        return _run_contiguous_suffix_path(out, dim, index, src, alpha)

    # Move target dim to last position for coalesced memory access
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    # Clone input for output
    out = inp.clone()

    # Calculate grid with autotune
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](out, index, src, M, N, alpha, inp_len)

    # Restore original dimension order if needed
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    """
    In-place version of index_add.
    """
    logger.debug("GEMS_MTHREADS INDEX_ADD_")

    dim = _validate_index_add_args(inp, dim, index, src)
    if src is inp or index is inp:
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )
    if src.numel() == 0:
        return inp
    if torch._C._is_alias_of(inp, src) or torch._C._is_alias_of(inp, index):
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )
    if _should_redispatch_native(inp, dim, index, src):
        trusted_entry = _trusted_index_entry(index, inp.size(dim))
        if trusted_entry is None:
            _assert_native_index_in_bounds(index, inp.size(dim))
        if _should_use_sorted_run_path(trusted_entry, index):
            return _run_sorted_run_path(
                inp, dim, index, src, alpha, trusted_entry
            )
        return _native_index_add_(inp, dim, index, src, alpha)

    use_contiguous_suffix_path = _can_use_contiguous_suffix_path(
        inp, dim, index, src
    ) and not torch._C._is_alias_of(inp, src)

    # Make index and src contiguous. resolve_neg() is a no-op normally.
    index = _resolve_index_for_kernel(index).contiguous()
    src = src.contiguous()

    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Reject invalid receivers before a pointer kernel can observe them.
    # Use min/max to avoid allocating full-size boolean tensors.
    _assert_index_in_bounds(index, inp_len)

    if use_contiguous_suffix_path:
        return _run_contiguous_suffix_path(inp, dim, index, src, alpha)

    # Move target dim to last position
    final_dim = inp.ndim - 1

    if dim != final_dim:
        # Need to work on a permuted copy
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_work, index, src_work, M, N, alpha, inp_len)

        # Restore original dimension order and copy back
        order = list(range(inp_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = inp_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        # Can work directly on input if already contiguous
        inp_contig = inp.contiguous()

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_contig, index, src, M, N, alpha, inp_len)

        # Copy back if input wasn't contiguous
        if not inp.is_contiguous():
            inp.copy_(inp_contig)

    return inp
