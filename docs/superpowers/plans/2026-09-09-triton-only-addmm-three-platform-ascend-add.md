# Triton-Only AddMM on Three Platforms and Ascend Add Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every rejected Native-routed AddMM/Add hot path with FlagGems-owned Triton computation, preserve the pinned GraphCast 5% correctness contract, and reach at least 16% MFU in the 40-step seven-operator GraphCast run on Ascend 910C, Hygon BW3000, and Moore Threads S5000.

**Architecture:** Keep the common AddMM and pointwise Add implementations as Triton correctness fallbacks, then add backend-specific direct Triton kernels and model-independent selectors for the measured layout classes. Share only the runtime precision-policy reader; keep tuning, skinny-K strategy, S5000 TF32 rounding/cache, and Ascend suffix-strided Add backend-local. Prove the no-Native rule twice: an AST/source audit in the repository and a target-device profiler audit of the actual GraphCast calls.

**Tech Stack:** Python 3, PyTorch dispatcher APIs for registration only, Triton JIT/FlagGems `libentry` and `libtuner`, pytest, FlagOS remote runners, platform profilers, and the pinned `ZYX223/ai4s_graphcast` evaluation scripts.

**Spec:** `docs/superpowers/specs/2026-09-09-triton-only-addmm-three-platform-ascend-add-design.md`

## Global Constraints

- Work only in `/mdata/data/zhangyuxin/wxt/FlagGems/.worktrees/triton-only-addmm-three-platform-ascend-add-v535` on `codex/triton-only-addmm-three-platform-ascend-add-v535`.
- Do not cherry-pick `404be08bf60eacb8cd80436295285a8e8cd920f1` or `93624d8010530a8341124e55992a5f6d27980695`. Port reviewed Triton-only fragments manually and preserve authorship in commit trailers when applicable.
- Target computation may not call `redispatch`, `get_kernel`, `call_boxed`, `torch.ops.npu.npu_linear`, `torch.addmm`, `torch.mm`, `torch.matmul`, `torch.add`, or a vendor extension. Tensor allocation and metadata-only views are allowed.
- Unsupported specialized cases must enter an existing FlagGems Triton implementation, not Torch/vendor Native.
- Production dispatch may use dtype, rank, shape class, stride class, bias class, precision policy, and gradient state. It may not use `GraphCast`, a call ordinal, or an exact production M value.
- Do not report performance from a run with failed operator correctness, failed GraphCast diagnostics, incomplete 7/7 coverage, wrong module provenance, or Native target computation.
- Make one focused commit after each task's tests pass. Never mix device-measured configuration changes with unrelated semantic refactors.

---

## Task 1: Freeze Provenance and Add an Enforceable No-Native Audit

**Files:**

- Create: `tools/check_triton_only_target_ops.py`
- Create: `tests/test_triton_only_target_ops.py`
- Verify: `docs/superpowers/specs/2026-09-09-triton-only-addmm-three-platform-ascend-add-design.md`

- [ ] **Step 1: Write the failing audit tests**

Add tests that load the audit module by file path and exercise both forbidden and permitted syntax. The test must use temporary source files so it proves the checker rather than merely grepping the current tree.

```python
def test_audit_rejects_native_compute_routes(tmp_path):
    source = tmp_path / "addmm.py"
    source.write_text(
        "import torch\n"
        "def addmm(a, b, c):\n"
        "    k = torch.library.get_kernel('aten::addmm', 'PrivateUse1')\n"
        "    return k.call_boxed(a, b, c)\n"
    )
    violations = audit_paths([source])
    assert {item.symbol for item in violations} == {
        "torch.library.get_kernel",
        "k.call_boxed",
    }


def test_audit_accepts_triton_and_metadata_only_torch(tmp_path):
    source = tmp_path / "addmm.py"
    source.write_text(
        "import torch\nimport triton\n"
        "def addmm(a, m, n):\n"
        "    out = torch.empty((m, n), device=a.device, dtype=a.dtype)\n"
        "    kernel[(m,)](a, out)\n"
        "    return out\n"
    )
    assert audit_paths([source]) == []
```

The forbidden set must cover attribute suffixes and fully qualified names for `redispatch`, `get_kernel`, `call_boxed`, `npu_linear`, and Torch/vendor AddMM/MM/MatMul/Add compute calls. It must not reject `torch.empty`, `torch.Tensor`, `.contiguous()`, `.broadcast_to()`, or normal Triton launches.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
python -m pytest -q tests/test_triton_only_target_ops.py
```

Expected: collection fails because `tools/check_triton_only_target_ops.py` does not exist.

- [ ] **Step 3: Implement the AST audit**

Implement:

```python
@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    symbol: str


def audit_paths(paths: Sequence[Path]) -> list[Violation]:
    """Return every forbidden call found below the supplied Python paths."""
```

Use `ast.parse`, reconstruct dotted `Name`/`Attribute` call targets, and match both fully qualified forbidden calls and forbidden terminal attributes. Expose a CLI accepting one or more files/directories and return code 1 when violations exist, 0 otherwise, and 2 for unreadable/invalid input. Directory traversal is restricted to `*.py`.

- [ ] **Step 4: Run the tests and audit the initial production target files**

Run:

```bash
python -m pytest -q tests/test_triton_only_target_ops.py
python tools/check_triton_only_target_ops.py \
  src/flag_gems/ops/addmm.py \
  src/flag_gems/ops/add.py \
  src/flag_gems/runtime/backend/_ascend/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/ops/addmm.py
```

Expected: tests pass and the v5.3.5 starting paths produce zero violations. The later-created Hygon AddMM and Ascend Add files are added to this command in Task 9.

- [ ] **Step 5: Commit**

```bash
git add tools/check_triton_only_target_ops.py tests/test_triton_only_target_ops.py
git commit -m "test: enforce Triton-only target operator paths"
```

---

## Task 2: Centralize Fast-FP32 Policy Without Performing Compute

**Files:**

- Create: `src/flag_gems/runtime/matmul_precision.py`
- Create: `tests/test_addmm_precision_policy.py`

- [ ] **Step 1: Write precision-policy tests**

Use `types.SimpleNamespace` to replace the module-local `torch` object. Cover:

- Ascend `torch.npu.matmul.allow_hf32` false/true;
- Moore Threads `torch.backends.mudnn.allow_tf32` false/true;
- Hygon `torch.backends.cuda.matmul.allow_tf32` false/true;
- missing attributes and `RuntimeError` reads defaulting to strict mode;
- fast mode applying only when every matrix operand is FP32;
- FP16, BF16, and FP64 never being relabeled as fast FP32.

The public interface is:

```python
def is_fast_float32_matmul_enabled(vendor_name: str) -> bool:
    """Read the requested fast-FP32 policy, defaulting to strict mode."""


def should_use_fast_float32_matmul(
    vendor_name: str,
    *operands: torch.Tensor,
) -> bool:
    """Return true only for enabled fast mode with all-FP32 operands."""
```

- [ ] **Step 2: Run the test and confirm RED**

```bash
python -m pytest -q tests/test_addmm_precision_policy.py
```

Expected: import error because `flag_gems.runtime.matmul_precision` is absent.

- [ ] **Step 3: Implement fail-closed policy reading**

Port the attribute-reading logic from historical commit `13966fb45` only. Do not port any call site that invokes Native compute or mutates a global precision flag. Catch `AttributeError` and `RuntimeError`, return `False` for unknown vendors, and export only the two functions above.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q tests/test_addmm_precision_policy.py
python -m compileall -q src/flag_gems/runtime/matmul_precision.py
git add src/flag_gems/runtime/matmul_precision.py tests/test_addmm_precision_policy.py
git commit -m "feat(addmm): centralize backend fast-fp32 policy"
```

---

## Task 3: Fix Common Triton AddMM Semantics Before Backend Tuning

**Files:**

- Modify: `src/flag_gems/ops/addmm.py`
- Modify: `tests/test_addmm.py`
- Modify: `benchmark/test_addmm.py`

- [ ] **Step 1: Add failing semantic cases**

Extend `tests/test_addmm.py` with model-independent cases:

Add `test_addmm_beta_zero_does_not_read_nan_bias`, parameterized over row- and column-major RHS; `test_addmm_empty_or_skinny_k`, parameterized over `(31, 37, 0)`, `(257, 512, 4)`, and `(513, 127, 184)`; and `test_addmm_padded_k_contiguous_rhs`, which creates a `(512, 1536)` FP32 storage tensor on `flag_gems.device`, slices 512 columns, transposes it, and asserts the resulting RHS stride is `(1, 1536)` before comparing Torch and FlagGems outputs.

For `beta=0`, fill the bias with `NaN` and require a finite result matching `alpha * mat1 @ mat2`. Cover scalar/vector/matrix bias because none may be loaded. Preserve existing dtype-aware tolerances.

- [ ] **Step 2: Run the target suite on the available accelerator and confirm the beta-zero failure**

```bash
python -m pytest -q tests/test_addmm.py -k 'beta_zero or empty_or_skinny_k or padded_k_contiguous_rhs' --maxfail=1
```

Expected on a supported device: the NaN-bias case fails before the kernel change because the existing epilogue unconditionally loads and multiplies the bias. If no local accelerator is present, record the local skip/collection status and perform the RED run first on each remote target before implementation is installed there.

- [ ] **Step 3: Make beta-zero a compile-time epilogue branch**

Add `BETA_IS_ZERO: tl.constexpr` to `addmm_kernel`. The epilogue must be structurally equivalent to:

```python
if BETA_IS_ZERO:
    result = accumulator * alpha
else:
    # Existing vector/scalar/broadcast bias load.
    result = accumulator * alpha + bias.to(accumulator.dtype) * beta
```

Pass `BETA_IS_ZERO=beta == 0` from `_addmm_impl`. Keep `HAS_K` so K=0 returns the correctly scaled bias without entering `tl.dot`.

- [ ] **Step 4: Add generalized benchmark shapes**

Extend only `AddmmVectorBiasBenchmark.set_more_shapes()` with these bounded cases in `(batch, M, N, K)` form:

```python
return [
    (1, 4096, 512, 4),
    (1, 4096, 512, 184),
    (1, 4096, 512, 512),
    (1, 4096, 512, 1024),
]
```

Enhance `record_shapes` with both RHS strides so benchmark evidence distinguishes compact and padded K-contiguous layouts. Exact GraphCast row counts remain outside upstream benchmark code.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q tests/test_addmm.py -k 'beta_zero or empty_or_skinny_k or padded_k_contiguous_rhs' --maxfail=1
python -m compileall -q src/flag_gems/ops/addmm.py benchmark/test_addmm.py
git add src/flag_gems/ops/addmm.py tests/test_addmm.py benchmark/test_addmm.py
git commit -m "fix(addmm): preserve beta-zero and skinny layout semantics"
```

---

## Task 4: Implement Ascend 910C Direct Triton AddMM Paths

**Files:**

- Modify: `src/flag_gems/runtime/backend/_ascend/ops/addmm.py`
- Modify: `src/flag_gems/runtime/backend/_ascend/tune_configs.yaml`
- Modify: `tests/test_addmm.py`
- Create: `tests/test_ascend_addmm_triton_route.py`

- [ ] **Step 1: Write routing/precision tests before changing the backend**

Test pure helpers without an NPU by importing the backend file through its normal package path only on Ascend and otherwise testing a small selector module-level function with lightweight tensor metadata. The required behavior is:

```python
assert classify_addmm_layout(stride_am=4, stride_ak=1,
                             stride_bk=1, stride_bn=1536) == "a_row_b_k_padded"
assert classify_addmm_layout(stride_am=4, stride_ak=1,
                             stride_bk=512, stride_bn=1) == "a_row_b_row"
assert select_ascend_addmm_kernel(K=4, layout="a_row_b_k_padded") == "skinny_k"
assert select_ascend_addmm_kernel(K=184, layout="a_row_b_k_padded") == "grouped_gemm"
```

Also inspect the launched kernel object or monkeypatch the launch boundary so an FP32 call maps disabled HF32 to `INPUT_PRECISION="ieee"` and enabled HF32 to `INPUT_PRECISION="hf32"`. FP16/BF16 must not use the FP32 policy branch.

- [ ] **Step 2: Run the new tests and confirm RED**

```bash
python -m pytest -q tests/test_ascend_addmm_triton_route.py
```

Expected: missing classifier/selector and precision launch metadata.

- [ ] **Step 3: Refactor the existing grouped GEMM into an explicit direct-Triton launch**

Keep the current grouped scheduling and vector-bias epilogue. Add these compile-time fields:

```python
BETA_IS_ZERO: tl.constexpr
INPUT_PRECISION: tl.constexpr
```

Use:

```python
acc += tl.dot(
    a,
    b,
    out_dtype=dot_out_dtype,
    input_precision=INPUT_PRECISION,
)
```

Select `"hf32"` only through `should_use_fast_float32_matmul("ascend", mat1, mat2)`, otherwise `"ieee"`. There must be no `npu_linear` helper or branch.

- [ ] **Step 4: Add a skinny-K Triton kernel**

Add `addmm_skinny_k_kernel` for `K < 16`. Each program owns a rectangular `(BLOCK_M, BLOCK_N)` output tile, loads the entire small K extent with masks, accumulates FP32 products along K, fuses `alpha` and all bias classes, and stores once. Start with these bounded tuning candidates:

```python
SKINNY_K_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 8}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 8}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 8}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 8}, num_warps=8),
]
```

If the Ascend Triton compiler rejects a 3-D broadcast/reduce formulation, retain the same ownership and use a compile-time unrolled K loop. Do not route K=4 through a vendor matmul.

- [ ] **Step 5: Separate tuning keys by layout class and precision**

The grouped path tuning key must include M/N/K, an integer RHS layout class, and the compile-time precision mode. Sweep:

- `BLOCK_M`: 32, 64, 128;
- `BLOCK_N`: 64, 128, 256 where supported;
- `BLOCK_K`: 16, 32, 64;
- `GROUP_M`: 1, 4, 8;
- stages: 1, 2, 3;
- supported Ascend execution widths from the current `mm` heuristics.

Run crossover points with M in 4096, 16384, 65536 and N/K classes 512/184, 512/512, 512/1024 for row and padded-K-contiguous RHS. Record every candidate, compilation failure, median kernel time, and selected config. Encode winners by generalized class in `tune_configs.yaml`; do not add exact production M dispatch.

- [ ] **Step 6: Run Ascend correctness and compile proof**

On 910C, run in a fresh process with HF32 disabled and enabled:

```bash
GEMS_VENDOR=ascend python -m pytest -q tests/test_addmm.py tests/test_ascend_addmm_triton_route.py --maxfail=1
```

Expected: all supported dtype/layout/bias/out cases pass. Save Triton compilation logs proving both `ieee` and `hf32` variants compile.

- [ ] **Step 7: Commit**

```bash
git add src/flag_gems/runtime/backend/_ascend/ops/addmm.py \
  src/flag_gems/runtime/backend/_ascend/tune_configs.yaml \
  tests/test_addmm.py tests/test_ascend_addmm_triton_route.py
git commit -m "perf(ascend): add direct Triton addmm paths"
```

---

## Task 5: Implement Hygon BW3000 Direct Triton AddMM and Safe Fast-FP32 Selection

**Files:**

- Create: `src/flag_gems/runtime/backend/_hygon/ops/addmm.py`
- Modify: `src/flag_gems/runtime/backend/_hygon/ops/__init__.py`
- Modify: `src/flag_gems/runtime/backend/_hygon/tune_configs.yaml`
- Create: `tests/test_hygon_addmm_triton_route.py`
- Modify: `tests/test_addmm.py`

- [ ] **Step 1: Write RED tests for registration and stable class selection**

Require `_hygon.ops` to export `addmm`, `addmm_out`, `addmm_dtype`, and `addmm_dtype_out`. Define and test a stable classification with exactly these diagnostic classes:

```text
skinny_k                 K < 16
k184                     16 <= K < 256
k512_rhs_row             K == 512 and stride_bn == 1
k512_rhs_k_contiguous    K == 512 and stride_bk == 1
k1024_plus               K >= 1024
general                  everything else
```

The initial production fast-FP32 allowlist is empty. A test must reject call-index- or exact-M-based selectors.

- [ ] **Step 2: Run and confirm RED**

```bash
python -m pytest -q tests/test_hygon_addmm_triton_route.py
```

Expected: Hygon does not yet export a backend AddMM implementation.

- [ ] **Step 3: Add a Hygon-owned direct Triton kernel**

Copy the semantic structure of `src/flag_gems/ops/addmm.py`, not the historical captured-kernel wrapper. Preserve vector/scalar/broadcast bias and `beta=0`. Add grouped large-M scheduling and a skinny-K kernel. Fast FP32 is a compile-time `ALLOW_TF32`/supported Hygon Triton dot precision choice; it must never mutate `torch.backends.cuda.matmul.allow_tf32` during a call.

The exported Python entry points allocate outputs, classify layout/precision, and launch only these Hygon Triton kernels. General valid layouts may call `flag_gems.ops.addmm._addmm_impl`, which is also Triton.

- [ ] **Step 4: Tune strict kernels first**

With fast FP32 disabled, sweep the same M/N/K tile ranges as Ascend plus Hygon-supported warps 4/8/16 and stages 1/2/3. Compare direct padded-K-contiguous loads with a separate Triton materialization kernel. Count the pack time in end-to-end AddMM latency. Permit caching only for immutable inference RHS and only after Task 6's cache invariants are shared or duplicated with tests.

- [ ] **Step 5: Run isolated fast-FP32 class experiments**

In fresh GraphCast processes, evaluate these variants:

1. strict for every class;
2. fast for every class;
3. fast for exactly one class at a time;
4. cumulative fast classes ordered by measured weighted replay benefit.

Each variant must generate a complete 40-step trajectory and run both fixed-reference diagnostics. A class is eligible for the production allowlist only when the cumulative variant preserves strict `<5%` overall and Z500 against both references. Store the chosen class names as a tuple constant; never store call ordinals or production M sizes.

- [ ] **Step 6: Verify public semantics and registration on Hygon**

```bash
GEMS_VENDOR=hygon python -m pytest -q \
  tests/test_addmm.py tests/test_addmm_precision_policy.py \
  tests/test_hygon_addmm_triton_route.py --maxfail=1
```

Expected: all applicable FP32/FP16/BF16/layout/out tests pass in strict and chosen fast modes.

- [ ] **Step 7: Commit strict kernel and measured selector separately**

```bash
git add src/flag_gems/runtime/backend/_hygon/ops/addmm.py \
  src/flag_gems/runtime/backend/_hygon/ops/__init__.py \
  src/flag_gems/runtime/backend/_hygon/tune_configs.yaml \
  tests/test_hygon_addmm_triton_route.py tests/test_addmm.py
git commit -m "perf(hygon): add direct Triton addmm backend"
```

After a fast class passes the complete diagnostic, make a second commit containing only the selector/config evidence:

```bash
git commit -am "perf(hygon): enable validated Triton fast-fp32 addmm classes"
```

If no class passes, keep the allowlist empty and do not create the second commit.

---

## Task 6: Add S5000 Triton TF32 Rounding and a Safe RHS Cache

**Files:**

- Create: `src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py`
- Create: `tests/test_mthreads_addmm_tf32_cache.py`
- Create: `tests/test_mthreads_tf32_rounding.py`
- Modify: `src/flag_gems/runtime/backend/_mthreads/ops/addmm.py`

- [ ] **Step 1: Write CPU-only cache tests**

Port and strengthen the historical cache tests to require:

- repeated reconstructed transpose views reuse one rounded RHS in `torch.inference_mode()`;
- in-place mutation changes `_version` and misses;
- storage pointer, shape, stride, storage offset, dtype, and device participate in identity;
- grad-enabled execution always bypasses the cache;
- weakref owner death removes the entry;
- capacity is bounded and least-recently-used;
- concurrent `get` operations do not corrupt bookkeeping.

The cache interface remains:

```python
class TF32RHSCache:
    def __init__(self, max_entries: int = 512):
        """Create an empty bounded cache."""
    def get(self, tensor, rounder):
        """Return a valid cached tensor or call rounder once and cache it."""
    def clear(self) -> None:
        """Drop every entry."""
    def __len__(self) -> int:
        """Return the number of live entries."""
```

- [ ] **Step 2: Run cache tests and confirm RED**

```bash
python -m pytest -q tests/test_mthreads_addmm_tf32_cache.py
```

Expected: the cache module is missing.

- [ ] **Step 3: Implement the bounded, version-aware cache**

Port the `OrderedDict`/weakref/RLock implementation from the historical Triton preprocessing work. Do not port `_load_native_addmm_kernels`, `get_kernel`, `call_boxed`, or any captured dispatch state.

- [ ] **Step 4: Write an independent bit-level TF32 rounding oracle**

The test oracle operates on CPU `uint32` bit patterns and handles finite values, signed zero, infinities, and NaNs. For finite FP32, round-to-nearest-even from 23 fraction bits to 10 by adding `0x00000FFF + ((bits >> 13) & 1)` before clearing `0x1FFF`; preserve non-finite encodings according to the backend's documented behavior. Compare the S5000 Triton copy output bit-for-bit for adversarial halfway values and random inputs.

- [ ] **Step 5: Add the Triton rounding/materialization kernel**

Port only:

```python
@libentry()
@triton.jit
def _round_to_tf32_copy_kernel(src, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, ext.round_to_tf32(values), mask=mask)
```

`round_to_tf32_copy` must materialize the logical tensor contiguously with a Triton kernel, not a Torch/vendor compute conversion. Changing LHS activations are rounded per call; only inference RHS uses `TF32RHSCache`.

- [ ] **Step 6: Run the rounding tests on S5000**

```bash
GEMS_VENDOR=mthreads python -m pytest -q \
  tests/test_mthreads_tf32_rounding.py \
  tests/test_mthreads_addmm_tf32_cache.py --maxfail=1
```

Expected: bit-for-bit match and all invalidation tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py \
  src/flag_gems/runtime/backend/_mthreads/ops/addmm.py \
  tests/test_mthreads_addmm_tf32_cache.py tests/test_mthreads_tf32_rounding.py
git commit -m "feat(mthreads): add Triton TF32 rounding cache"
```

---

## Task 7: Tune S5000 Direct Triton AddMM Instead of Redispatching

**Files:**

- Modify: `src/flag_gems/runtime/backend/_mthreads/ops/addmm.py`
- Modify: `src/flag_gems/runtime/backend/_mthreads/tune_configs.yaml`
- Modify: `src/flag_gems/runtime/backend/_mthreads/addmm_mthreads_expand.yaml`
- Create: `tests/test_mthreads_addmm_triton_route.py`
- Modify: `tests/test_addmm.py`

- [ ] **Step 1: Add RED routing tests**

Require FP32 vector-bias inference calls to select one of `pointer`, `sqmma`, or `skinny_k`, never `native`. Require FP16/BF16 to retain the current eligible SQMMA path and dtype-out promotion to retain FP32 accumulation semantics.

- [ ] **Step 2: Remove every historical Native hook if present**

Ensure the module contains none of `_load_native_addmm_kernels`, `_NATIVE_ADDMM_KERNEL`, `_NATIVE_ADDMM_MODE`, `get_kernel`, `call_boxed`, or `redispatch`. The public entry points must end at `addmm_kernel`, `addmm_sqmma_kernel`, `addmm_skinny_k_kernel`, or the Triton round/copy kernel.

- [ ] **Step 3: Make pointer-based FP32 use the precision policy**

Add `ALLOW_TF32` and `BETA_IS_ZERO` compile-time arguments and include `ALLOW_TF32` plus layout class in the tuner key. In fast mode, pass pre-rounded tensors to `tl.dot(a, b, allow_tf32=True)`; in strict mode, retain original tensors and call `tl.dot(a, b, allow_tf32=False)`.

- [ ] **Step 4: Evaluate descriptor/SQMMA FP32 eligibility**

Extend `is_sqmma_compatible` only after a device compile-and-correctness proof for FP32 pre-rounded inputs. The descriptor path must include the fused vector-bias epilogue and `beta=0` branch. If the installed MUSA Triton rejects FP32 descriptors or lowers them without the intended matrix unit, keep FP32 on the pointer path and record the rejection; do not call MUDNN.

- [ ] **Step 5: Add and tune skinny-K**

Use a direct Triton multiply-reduce/unrolled kernel for K<16. Sweep M tiles 16/32/64, N tiles 64/128, stages 1/2, and supported warps 4/8/16. Count activation rounding time and any RHS first-use rounding separately and in total.

- [ ] **Step 6: Tune generalized large-M layout classes**

For K 184/512/1024 and N 512, compare:

- direct pointer loads from compact row RHS;
- direct pointer loads from K-contiguous and padded-K-contiguous RHS;
- one-time Triton RHS pack plus cache reuse;
- descriptor/SQMMA only where Step 4 proved eligibility.

Record cold first-call, warm cached-call, and weighted one-step latency. Select by layout class, not exact M.

- [ ] **Step 7: Verify on S5000 and commit**

```bash
GEMS_VENDOR=mthreads python -m pytest -q \
  tests/test_addmm.py tests/test_addmm_precision_policy.py \
  tests/test_mthreads_addmm_tf32_cache.py \
  tests/test_mthreads_tf32_rounding.py \
  tests/test_mthreads_addmm_triton_route.py --maxfail=1

git add src/flag_gems/runtime/backend/_mthreads/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/tune_configs.yaml \
  src/flag_gems/runtime/backend/_mthreads/addmm_mthreads_expand.yaml \
  tests/test_mthreads_addmm_triton_route.py tests/test_addmm.py
git commit -m "perf(mthreads): tune direct Triton addmm paths"
```

---

## Task 8: Implement Ascend Suffix-Strided Add as a Direct Triton Kernel

**Files:**

- Create: `src/flag_gems/runtime/backend/_ascend/ops/add.py`
- Modify: `src/flag_gems/runtime/backend/_ascend/ops/__init__.py`
- Modify: `tests/test_add.py`
- Create: `tests/test_ascend_add_triton_route.py`
- Modify: `benchmark/test_add.py`

- [ ] **Step 1: Add model-independent RED tests**

Create views from moderate backing storage rather than allocating the production tensor:

```python
base = torch.randn((2, 31, 176), device=flag_gems.device, dtype=torch.float32)
left13 = base[..., :13]       # shape [2, 31, 13], stride suffix 176/1
left1 = base[..., :1]         # shape [2, 31, 1], stride suffix 176/1
right13 = torch.randn_like(left13)
```

Cover suffix 1/13, flat contiguous tensors, odd row tails, `alpha` values 0, 0.5, -2, broadcasting, FP16/BF16, and grad-enabled tensors. The specialized selector should accept same-shape FP32 suffix-strided or contiguous tensors and return common FlagGems Add for other cases.

- [ ] **Step 2: Confirm RED**

```bash
GEMS_VENDOR=ascend python -m pytest -q \
  tests/test_add.py tests/test_ascend_add_triton_route.py \
  -k 'suffix or triton_route' --maxfail=1
```

Expected: backend Add module/export and specialized route are absent.

- [ ] **Step 3: Implement flat and suffix-strided Triton kernels**

The suffix-strided kernel receives logical row count, suffix length, both outer strides, both suffix strides, and output strides. A program processes `ROWS_PER_PROGRAM` rows and `BLOCK_SUFFIX=triton.next_power_of_2(suffix)` lanes, computes `A + alpha * B`, and stores with masks. The contiguous path uses a 1-D block. Neither kernel may call ATen Add.

Exported behavior:

```python
def add(A, B, *, alpha=1):
    if _can_use_contiguous_fp32(A, B):
        return _launch_flat_add(A, B, alpha)
    if _can_use_suffix_strided_fp32(A, B):
        return _launch_suffix_add(A, B, alpha)
    return _common_add(A, B, alpha=alpha)
```

`_common_add` is the FlagGems pointwise Triton implementation. Do not copy `_FALLBACK_KEYSET`, `_native_add`, or `Tensor.redispatch` from the rejected branch.

- [ ] **Step 4: Tune generalized layouts**

Sweep rows per program 1/2/4/8, suffix blocks 1/16/32, and Ascend-supported execution widths using row counts 31, 1024, 16384 and suffix 1/13/31. Encode crossover by suffix/stride class and size range, not the production row count 1,038,240.

- [ ] **Step 5: Extend benchmark and run full Add correctness**

Add bounded suffix-strided cases to `benchmark/test_add.py`, including stride-176 suffix 1/13. Then run:

```bash
GEMS_VENDOR=ascend python -m pytest -q tests/test_add.py tests/test_ascend_add_triton_route.py --maxfail=1
```

Expected: public Add semantics pass; route tests prove the specialized Triton launcher is used only for its supported region and the common Triton function receives all other cases.

- [ ] **Step 6: Commit**

```bash
git add src/flag_gems/runtime/backend/_ascend/ops/add.py \
  src/flag_gems/runtime/backend/_ascend/ops/__init__.py \
  tests/test_add.py tests/test_ascend_add_triton_route.py benchmark/test_add.py
git commit -m "perf(ascend): add direct Triton suffix-strided add"
```

---

## Task 9: Run Repository-Level Correctness and Static Route Verification

**Files:**

- Modify if required: `tests/test_triton_only_target_ops.py`
- Evidence only: `outputs/triton_only_addmm_add_<stamp>/local/`

- [ ] **Step 1: Expand the production audit to every final target file**

The test and CLI invocation must include:

```text
src/flag_gems/ops/addmm.py
src/flag_gems/ops/add.py
src/flag_gems/runtime/backend/_ascend/ops/addmm.py
src/flag_gems/runtime/backend/_ascend/ops/add.py
src/flag_gems/runtime/backend/_hygon/ops/addmm.py
src/flag_gems/runtime/backend/_mthreads/ops/addmm.py
src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py
```

- [ ] **Step 2: Run local tests that do not require an accelerator**

```bash
python -m pytest -q \
  tests/test_triton_only_target_ops.py \
  tests/test_addmm_precision_policy.py \
  tests/test_mthreads_addmm_tf32_cache.py
python -m compileall -q src/flag_gems tests benchmark tools
```

- [ ] **Step 3: Run the static audit**

```bash
python tools/check_triton_only_target_ops.py \
  src/flag_gems/ops/addmm.py \
  src/flag_gems/ops/add.py \
  src/flag_gems/runtime/backend/_ascend/ops/addmm.py \
  src/flag_gems/runtime/backend/_ascend/ops/add.py \
  src/flag_gems/runtime/backend/_hygon/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py
```

Expected: no violations and exit code 0.

- [ ] **Step 4: Review the diff for accidental model coupling**

```bash
git diff a7620cc191a0b42e040194622c5758b22a7a25dc -- src tests benchmark tools \
  | rg -n 'GraphCast|1038240|327660|131072|100064|45882|40962|call_boxed|get_kernel|redispatch|npu_linear'
```

Expected: no source dispatch contains model names or exact production M values and no prohibited compute route appears. Test/doc occurrence of `GraphCast` is permitted only where it describes evidence, never as a selector.

- [ ] **Step 5: Commit any audit-only corrections**

```bash
git add tests/test_triton_only_target_ops.py
git commit -m "test: audit all Triton-only addmm and add targets"
```

Skip the commit when no file changed.

---

## Task 10: Run Official FlagGems PR Checks on All Three Devices

**Files:**

- Existing harness repository: `/mdata/data/zhangyuxin/wxt/flagos-common-wxt-addmm-sync/scripts/flaggems-op-prcheck/`
- Evidence: one immutable timestamped directory per platform/operator

- [ ] **Step 1: Freeze candidate identity before upload**

```bash
git status --short
git rev-parse HEAD
git diff --binary a7620cc191a0b42e040194622c5758b22a7a25dc > /tmp/flaggems-triton-only.patch
sha256sum /tmp/flaggems-triton-only.patch
```

Expected: clean worktree. Record the commit and patch SHA in every run.

- [ ] **Step 2: Probe all targets and verify actual device identity**

From `/mdata/data/zhangyuxin/wxt/flagos-common-wxt-addmm-sync`:

```bash
bash scripts/platform-access/flagos_remote.sh probe 910c
bash scripts/platform-access/flagos_remote.sh probe hygon
bash scripts/platform-access/flagos_remote.sh probe s5000
```

The Hygon access alias may still contain `bw1000` in its host label; record the runtime device name and reject the result if it is not the user-designated BW3000 test environment.

- [ ] **Step 3: Sync a full clean candidate tree**

Use the existing upload/sync helper to place the exact committed worktree in a new timestamped remote directory. Do not overwrite the installed package in place. For editable environments, use the swap runner with `OVERLAY_FILES` listing every changed source/config file and rely on its `trap` restore.

- [ ] **Step 4: Run AddMM official/candidate pytest and benchmark**

For 910C and Hygon, invoke the platform runner inside the existing device container:

```bash
STAMP=triton_only_addmm_$(date +%Y%m%d_%H%M%S)
PLATFORM=ascend OP_NAME=addmm OP_FUNC=addmm STAMP="$STAMP" \
  DTYPES="float32 float16 bfloat16" WARMUP=100 ITER=200 \
  bash scripts/flaggems-op-prcheck/run_flaggems_op_prcheck_platform.sh

PLATFORM=hygon OP_NAME=addmm OP_FUNC=addmm STAMP="$STAMP" \
  DTYPES="float32 float16 bfloat16" WARMUP=100 ITER=200 \
  bash scripts/flaggems-op-prcheck/run_flaggems_op_prcheck_platform.sh
```

For S5000, use `run_editable_flaggems_op_prcheck.sh` with `PLATFORM_NAME=s5000`, the actual container/repository paths discovered in Step 2, and `OVERLAY_FILES` containing common AddMM, MThreads AddMM, TF32 cache, precision policy, config files, and tests. Never run two source swaps concurrently.

- [ ] **Step 5: Run Ascend Add official/candidate pytest and benchmark**

```bash
PLATFORM=ascend OP_NAME=add OP_FUNC=add STAMP="triton_only_add_${STAMP}" \
  DTYPES="float32 float16 bfloat16" WARMUP=100 ITER=200 \
  bash scripts/flaggems-op-prcheck/run_flaggems_op_prcheck_platform.sh
```

- [ ] **Step 6: Validate evidence completeness**

For every run, require `status=passed`, source SHA records, environment/version records, raw CSV/JSON, stdout/stderr, official and candidate results, and restored installed-source SHA. Any missing variant, unexplained skip, compile failure, or restore mismatch blocks the next performance claim.

---

## Task 11: Replay Every Real AddMM/Add Signature and Audit Runtime Kernels

**Files:**

- Create in evidence repository: `/mdata/data/zhangyuxin/wxt/flagos-common-wxt-addmm-sync/scripts/flaggems-op-prcheck/harnesses/replay_graphcast_addmm_calls.py`
- Create in evidence repository: `/mdata/data/zhangyuxin/wxt/flagos-common-wxt-addmm-sync/scripts/flaggems-op-prcheck/harnesses/replay_graphcast_add_calls.py`
- Create in evidence repository: `/mdata/data/zhangyuxin/wxt/flagos-common-wxt-addmm-sync/scripts/flaggems-op-prcheck/harnesses/summarize_triton_only_replay.py`

- [ ] **Step 1: Build replay from the recorded catalog, not guessed shapes**

The AddMM replay reads the existing captured call catalog and reconstructs all 21 unique shape/layout signatures, including storage padding and logical strides. It runs each signature independently and emits:

```json
{
  "signature": {"M": 0, "N": 0, "K": 0, "strides": {}},
  "calls_per_step": 0,
  "native_ms": 0.0,
  "official_ms": 0.0,
  "candidate_ms": 0.0,
  "max_abs_error": 0.0,
  "route": "triton_kernel_name"
}
```

The summary is `sum(signature_median_ms * calls_per_step)`. Assert the call counts sum to 264; otherwise the replay is invalid.

The Ascend Add replay reconstructs all ten calls and asserts the counts are six suffix-13 plus four suffix-1 calls, with the recorded left stride. Its weighted summary uses the same formula.

- [ ] **Step 2: Add harness unit tests**

Test catalog validation, padded-stride reconstruction, weighted aggregation, missing-signature rejection, and non-finite timing/error rejection on CPU metadata. Run these tests before device execution.

- [ ] **Step 3: Run isolated replay in fresh processes**

For each platform, collect strict and configured fast-mode AddMM results with at least 20 warmups and 100 timed iterations per signature. On 910C, also collect Add replay. Synchronize before and after each timed region. Do not combine concurrent jobs on one device.

- [ ] **Step 4: Profile representative dominant and skinny calls**

Capture profiler traces for K=4 and the dominant K=184/512/1024 layout classes. The accepted trace must contain the exact Triton JIT kernel names introduced in Tasks 4/5/7. Search raw profiler events for vendor Add, AddMM, Linear, GEMM, BLAS, MUDNN, or CANN compute. A vendor event servicing target computation invalidates the run; allocator/memcpy events are allowed and reported.

- [ ] **Step 5: Record the optimization accounting**

Report separately:

- first compile/cold call;
- warm strict call;
- warm fast call;
- S5000 first RHS round/pack;
- S5000 cached RHS call;
- weighted 264-call AddMM total;
- weighted 10-call Ascend Add total.

This prevents a cache warmup or compile time from being silently mixed into steady-state results.

---

## Task 12: Run the Pinned GraphCast 40-Step Correctness and MFU Campaign

**Files:**

- Evaluation source: `ZYX223/ai4s_graphcast@3e435216adb7af56a32e02a4a8490dde3d405eb0`
- Runner: `ai4s/graphcast/scripts/inference/run_operational_40step_inference.py`
- Diagnostic: `ai4s/graphcast/scripts/inference/diagnose_inference.py`
- Evidence: one fresh run directory per process/platform/ablation

- [ ] **Step 1: Create a clean pinned evaluation worktree**

Do not use the dirty local training worktree. From the GraphCast repository, create a detached clean worktree at commit `3e435216adb7af56a32e02a4a8490dde3d405eb0`. If a platform's xarray requires them, apply only compatibility commits `b9cade41078ac87dd6912ec2bd295999e4c4d97e` and `f74b909565bc7f70a0394bc37d7a0a5efc317970` and record the resulting diff hash.

Verify:

```bash
sha256sum ai4s/graphcast/scripts/inference/run_operational_40step_inference.py
sha256sum ai4s/graphcast/scripts/inference/diagnose_inference.py
```

Expected runner SHA256:

```text
2c839ecc83cd85e1cb14715624607ec5318c5aa5a0fbc46eaef469ee1d0d4f60
```

Expected diagnostic SHA256 is `2ddf0938bc59a7b8a8c3f338c067746e4f359d568da23d689562a2036bcad9c7` at the pinned commit or `d04fd5a90ee6eebff477d0c98417ebd94c436b376b187998f8e2f239e38e10e6` with both permitted xarray compatibility commits.

- [ ] **Step 2: Verify immutable case assets**

Check the configured files against the recorded SHA256 values:

```text
config:            7ffbc9cc000dad8576d1c8e4be6a66a51897c8ebc20239e9ceb49448ceeeab5d
inputs:            1a021ee9cd17828ec15d953a94242c27b534f7d6ce89e38177155e0962aca409
rollout forcings:  58101077501da8f1e7cf7d3562ba5c0430f924d3a27697054ae35634353f98b4
target template:   df35e0ede4805110cfcfa5c46ba48d296cbd1182378e477685270c709234ab07
weights:           2bd1786e197b432d0f471ce4118f34b231c73c322dbaba0684e5f71097e010b4
fixed baseline:    10227f90bbd6dc94c032cf890ff6c6c9871e21d0a1678123c6ca183ae996be10
metadata:          f7efe0b7ae751a2d809a555585db7f9070c19a8399eb6d585fd33609bc7e2e28
```

Reject stale or mismatched assets.

- [ ] **Step 3: Run ablations before the final 7/7 run**

For each platform run separate fresh processes for:

- Native/off baseline;
- AddMM-only FlagGems;
- Ascend Add-only FlagGems;
- all seven FlagGems operators.

Use `GRAPHCAST_FLAGOS_MODE=flaggems`, `GRAPHCAST_STRICT_FLAGOS=1`, `GRAPHCAST_FLAGGEMS_SOURCE_ROOT` pointing at the exact candidate checkout, an explicit record path, and `GRAPHCAST_FLAGGEMS_OPS` for the ablation. The final value is exactly:

```text
addmm,cat,native_layer_norm,index_add_,silu,index,add
```

Set `--precision tf32`, `--steps 40`, `--warmup-calls 3`, `--model-execution-profile domestic-v1`, `--device-count 1`, and the platform peak (`199`, `236.94`, or `224.07`). Request `--trajectory-output` so the 40-step candidate can be diagnosed.

- [ ] **Step 4: Run exactly three non-overlapping steady-state 7/7 processes**

Compilation and three warmup forwards remain outside the E2E timer. Do not reuse a Python process between repetitions. The report must say `warmup_excluded_from_e2e=true` and `final_output_d2h_included_in_e2e=true`. Calculate the median from the three `measurement.e2e_seconds` values; do not sum the three runs and do not call a 40-run aggregate one run.

- [ ] **Step 5: Run the fixed two-reference diagnostic on every candidate**

Invoke:

```bash
python ai4s/graphcast/scripts/inference/diagnose_inference.py \
  --run-dir <run-1> --run-dir <run-2> --run-dir <run-3> \
  --baseline <fixed-baseline.npz> \
  --baseline-metadata <fixed-baseline.metadata.json> \
  --output-dir <new-diagnostic-directory>
```

Every run must report verified provenance, qualified reference, 7/7 coverage, finite output, and `acceptance_passed=true`. Both overall mean and Z500 must be strictly below 5% against both JAX FP32 and PyTorch GPU references. Worst layer is recorded but is not an independent gate.

- [ ] **Step 6: Enforce per-platform MFU thresholds**

Use only the pinned report formula. The median E2E limits are:

```text
910C:   <= 36.2526 s  (199 TFLOPS, MFU >= 16%)
BW3000: <= 30.4477 s  (236.94 TFLOPS, MFU >= 16%)
S5000:  <= 32.1965 s  (224.07 TFLOPS, MFU >= 16%)
```

Confirm all target AddMM/Add calls appear in the FlagGems record and the profiler audit identifies their Triton-generated kernels with no prohibited Native target compute.

---

## Task 13: Evidence-Gated Triton Fusion if a Platform Remains Below 16% MFU

**Files:**

- Create only after profiling: `docs/superpowers/specs/<date>-<measured-fusion>-design.md`
- Create only after design approval: `docs/superpowers/plans/<date>-<measured-fusion>.md`
- Candidate source location: `src/flag_gems/runtime/backend/_<vendor>/fused/`

- [ ] **Step 1: Stop standalone tuning at the measured plateau**

Declare a plateau only after two complete tuning rounds change weighted replay by less than 2% and the profiler shows the same residual hotspot. Preserve all correctness/no-Native gates. Do not continue random tile searches merely to claim progress.

- [ ] **Step 2: Select one fusion from actual adjacency evidence**

From the 7/7 profiler, compute inclusive device time and launch counts for adjacent pairs. Consider `addmm+bias+activation`, `addmm+silu`, or `layer_norm+residual_add` only when the trace proves direct producer-consumer adjacency, compatible layouts, and removable intermediate traffic. Choose the pair with the largest measured removable time sufficient to close the platform's exact E2E gap.

- [ ] **Step 3: Write and approve a fusion design before code**

The design must specify public integration, full formula, shape/dtype/stride contract, autograd policy, Triton ownership, intermediate elimination, fallback to separate FlagGems Triton kernels, correctness tests, benchmark cases, and required E2E delta. Native sub-operations remain forbidden.

- [ ] **Step 4: Implement with TDD and rerun Tasks 9-12**

The fused kernel gets composition-level reference tests and its own profiler route proof. Adopt it only if the same pinned GraphCast diagnostic passes and median 40-step E2E reaches the platform threshold. Otherwise retain the fastest correct standalone Triton result and report the measured limitation.

---

## Task 14: Final Review, Delivery, and Supersession Notice

**Files:**

- Create: `docs/triton-only-addmm-add-results-2026-09-09.md`
- Update after acceptance in evidence repo: `patches/flaggems/README.md`
- Create after acceptance in evidence repo: `patches/flaggems/wip-addmm-triton-only/`
- Create after acceptance in evidence repo: `patches/flaggems/wip-add-ascend-triton-only/`

- [ ] **Step 1: Run final source and test verification**

```bash
git status --short
python -m pytest -q \
  tests/test_triton_only_target_ops.py \
  tests/test_addmm_precision_policy.py \
  tests/test_mthreads_addmm_tf32_cache.py
python -m compileall -q src/flag_gems tests benchmark tools
python tools/check_triton_only_target_ops.py \
  src/flag_gems/ops/addmm.py \
  src/flag_gems/ops/add.py \
  src/flag_gems/runtime/backend/_ascend/ops/addmm.py \
  src/flag_gems/runtime/backend/_ascend/ops/add.py \
  src/flag_gems/runtime/backend/_hygon/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/ops/addmm.py \
  src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py
```

- [ ] **Step 2: Review every changed line against the hard constraints**

Check:

- no Native compute route;
- no model name/exact production size in source dispatch;
- public semantics and dtype/out variants preserved;
- caches bounded and invalidated safely;
- device tune winners backed by raw evidence;
- no exact GraphCast data/weights/cases committed to upstream FlagGems.

- [ ] **Step 3: Write the result report**

The report must include a per-platform table with official benchmark, weighted real replay, cold start, each of three 40-step times, median E2E, MFU, both-reference overall/Z500 errors, 7/7 counts, Triton kernel names, profiler audit result, source commits/hashes, and explicit pass/fail. Separate historical rejected Native-routing numbers and label them noncompliant.

- [ ] **Step 4: Run a final code review before claiming completion**

Use `superpowers:requesting-code-review`, resolve correctness/performance-evidence findings, and rerun affected tests. Then use `superpowers:verification-before-completion`; do not rely on an earlier run after code changed.

- [ ] **Step 5: Commit the report and package only accepted source**

```bash
git add docs/triton-only-addmm-add-results-2026-09-09.md
git commit -m "docs: report Triton-only addmm and add validation"
```

Create the evidence packages from the accepted commit. Add a clear superseded warning to the old Native-routed `wip-addmm` and `wip-add-ascend` packages without deleting their historical evidence. Push only after verifying the configured GitHub account and remote target.
