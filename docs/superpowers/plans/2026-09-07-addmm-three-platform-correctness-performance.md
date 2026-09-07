# AddMM Three-Platform Correctness and Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver correct fast-FP32 `addmm` paths and >=16% GraphCast 40-step MFU on 910C, BW3000, and S5000.

**Architecture:** A shared runtime policy resolves strict versus fast FP32, while each vendor kernel maps it to its native matrix-unit mode. S5000 additionally moves TF32 nearest-even conversion out of repeated dot tiles and safely caches immutable inference weights. Controlled operator replay and GraphCast diagnostics gate every performance claim.

**Tech Stack:** Python, PyTorch custom operator registration, Triton, Ascend/Hygon/MUSA vendor backends, pytest, GraphCast diagnostic harness.

**Spec:** `docs/superpowers/specs/2026-09-07-addmm-three-platform-correctness-performance-design.md`

## Global Constraints

- Develop only in `/mdata/data/zhangyuxin/wxt/FlagGems-addmm-three-platform-20260907` on `codex/addmm-three-platform-correctness-performance`.
- Preserve all existing dirty worktrees.
- Do not use GraphCast names or exact production dimensions in FlagGems dispatch.
- Correctness against both fixed references must pass before performance is accepted.
- The steady-state 40-step MFU target is at least 16% on all three platforms.
- Record source paths, Git revisions, hashes, precision readback, warmup, and iteration counts.

---

### Task 1: Add the shared fast-FP32 precision policy

**Files:**
- Create: `src/flag_gems/runtime/matmul_precision.py`
- Create: `tests/test_addmm_precision_policy.py`

**Interfaces:**
- Produces: `should_use_fast_float32_matmul(vendor: str, mat1: Tensor, mat2: Tensor) -> bool`.
- The function returns true only for two FP32 operands and a true vendor runtime flag.

- [ ] **Step 1: Write policy tests before production code.**

  Test NVIDIA/Hygon `torch.backends.cuda.matmul.allow_tf32`, Ascend
  `torch.npu.matmul.allow_hf32`, Moore Threads
  `torch.backends.mudnn.allow_tf32`, strict settings, missing attributes,
  unsupported vendor names, and non-FP32 operands.

- [ ] **Step 2: Run the focused test and verify RED.**

  Run `python3 -m pytest tests/test_addmm_precision_policy.py -q` and require a
  missing-module or missing-function failure.

- [ ] **Step 3: Implement the smallest policy reader.**

  Use guarded attribute traversal. Do not mutate runtime flags, inspect model
  names, or default a missing flag to true.

- [ ] **Step 4: Run the focused test and verify GREEN.**

  Run `python3 -m pytest tests/test_addmm_precision_policy.py -q`.

- [ ] **Step 5: Commit the policy change.**

  Commit message: `feat(addmm): centralize fast fp32 precision policy`.

### Task 2: Connect Ascend addmm to HF32 dynamically

**Files:**
- Modify: `src/flag_gems/runtime/backend/_ascend/ops/addmm.py`
- Modify: `tests/test_addmm_precision_policy.py`

**Interfaces:**
- Consumes: `should_use_fast_float32_matmul("ascend", mat1, mat2)`.
- Produces: compile-time `INPUT_PRECISION` equal to `"hf32"` or `"ieee"`.

- [ ] **Step 1: Add a source/launch test that expects dynamic HF32 and verify RED.**

  The test must prove strict mode selects `"ieee"`, fast mode selects
  `"hf32"`, and the kernel does not retain an unconditional
  `allow_tf32=False` dot.

- [ ] **Step 2: Pass `INPUT_PRECISION` through the Ascend launcher.**

  Replace the hard-coded dot precision with
  `tl.dot(..., input_precision=INPUT_PRECISION)` and derive the value once per
  launch from the shared policy.

- [ ] **Step 3: Run policy tests, syntax compilation, and `git diff --check`.**

  Run `python3 -m pytest tests/test_addmm_precision_policy.py -q`,
  `python3 -m py_compile src/flag_gems/runtime/backend/_ascend/ops/addmm.py`, and
  `git diff --check`.

- [ ] **Step 4: Run 910C public addmm correctness.**

  Cover strict FP32 and HF32, vector/scalar/broadcast bias, `beta=0`, two tail
  shapes, contiguous and transposed RHS, and out variants. Save source/hash and
  precision readback with the result.

- [ ] **Step 5: Commit the Ascend change.**

  Commit message: `perf(ascend): honor hf32 policy in addmm`.

### Task 3: Rebaseline current Hygon vendor addmm

**Files:**
- Inspect: `src/flag_gems/runtime/backend/_hygon/ops/addmm.py`
- Inspect: `src/flag_gems/runtime/backend/_hygon/tune_configs.yaml`
- Create results under the external timestamped experiment directory only.

**Interfaces:**
- Produces: source-audited Official/Candidate correctness and timing evidence.

- [ ] **Step 1: Verify the Hygon vendor registration and source hash.**

  Confirm `flag_gems.addmm` resolves to `_hygon/ops/addmm.py` and that runtime
  TF32 readback is true.

- [ ] **Step 2: Run public addmm pytest before timing.**

  Run FP32, FP16, and BF16 plus the GraphCast-relevant layout/bias cases. Stop
  performance collection if Candidate correctness fails.

- [ ] **Step 3: Run official benchmark and 21-layout weighted replay.**

  Compare Native, upstream Official, and the unchanged latest Hygon candidate.
  Use identical warmup/iteration counts and an isolated Triton cache.

- [ ] **Step 4: Decide whether Hygon source tuning is needed.**

  If weighted replay is at or below 316 ms/step and 7/7 E2E is at or below
  30.4477 s, retain upstream unchanged. Otherwise sweep the existing legal
  `BLOCK_M/BLOCK_N/BLOCK_K`, warp, stage, and group settings for the aligned
  large-M semantic region, validate crossover neighbors, then add only the
  winning general configuration through the backend tuning mechanism.

- [ ] **Step 5: Re-run correctness after any selected tuning change and commit.**

  Commit message when source changes: `perf(hygon): tune aligned fp32 addmm`.

### Task 4: Add bit-exact TF32 nearest-even conversion tests

**Files:**
- Modify: `src/flag_gems/utils/triton_lang_extension.py`
- Create: `tests/test_addmm_tf32_rounding.py`

**Interfaces:**
- Produces: Triton helper `round_to_tf32(x)` whose output remains FP32.

- [ ] **Step 1: Write a host bit oracle and device tests before production code.**

  Include positive/negative ties, values on both sides of a tie, signed zero,
  subnormal, smallest normal, finite extrema, infinity, and NaN. Dense addmm
  tests compare against FP64 accumulation over explicitly rounded operands.

- [ ] **Step 2: Run the focused test on S5000 and verify RED.**

  Require failure because `round_to_tf32` is absent from the selected latest
  upstream source.

- [ ] **Step 3: Implement nearest-even rounding using FP32 bit operations.**

  Preserve sign and exponent, add the round-to-nearest-even bias derived from
  the retained least-significant bit, clear the low 13 mantissa bits, and leave
  special values representable.

- [ ] **Step 4: Run the focused S5000 tests and verify GREEN.**

  Save pytest output, source paths, and hashes.

- [ ] **Step 5: Commit the rounding helper.**

  Commit message: `feat(mthreads): add bit exact tf32 rounding helper`.

### Task 5: Move S5000 rounding out of repeated addmm tiles

**Files:**
- Modify: `src/flag_gems/runtime/backend/_mthreads/ops/addmm.py`
- Create: `src/flag_gems/runtime/backend/_mthreads/ops/tf32_cache.py`
- Create: `tests/test_mthreads_addmm_tf32_cache.py`
- Modify: `tests/test_addmm_tf32_rounding.py`

**Interfaces:**
- Produces: `get_rounded_tf32_rhs(tensor) -> Tensor` for eligible no-grad stable
  views and `round_to_tf32_copy(tensor) -> Tensor` for uncached operands.
- Consumes: `should_use_fast_float32_matmul("mthreads", mat1, mat2)` and
  `round_to_tf32`.

- [ ] **Step 1: Write cache lifecycle tests and verify RED.**

  Test a cache hit for repeated transpose views of one base weight; misses after
  `_version`, data pointer, shape, stride, or storage offset changes; weakref
  expiry; grad bypass; and fixed-size LRU eviction.

- [ ] **Step 2: Implement the bounded identity-checked cache.**

  Key by base object identity and view signature, validate the weakref target on
  every hit, and never cache grad-enabled or `requires_grad` tensors.

- [ ] **Step 3: Add the preprocessing kernel and fast launch path.**

  Round activation once per call, fetch or create the rounded RHS, and launch
  the existing dot kernel with fast precision without per-tile bit conversion.
  Preserve strict, FP16, BF16, FP64, dtype-out, and unsupported-layout fallbacks.

- [ ] **Step 4: Run S5000 public correctness and cache tests.**

  Require all focused tests to pass before timing.

- [ ] **Step 5: Compare preprocessing and in-kernel rounding across a crossover sweep.**

  Measure small/medium/large matrices, row/column/general RHS layouts, first
  call, cache hit, and cache invalidation. Select preprocessing only where its
  total latency wins; encode the boundary using operand bytes and reuse tiles.

- [ ] **Step 6: Commit the S5000 path.**

  Commit message: `perf(mthreads): reuse pre-rounded fp32 addmm weights`.

### Task 6: Run operator PR checks on all three platforms

**Files:**
- Validate: `tests/test_addmm.py`
- Validate: `benchmark/test_addmm.py`
- Store logs outside the PR diff in a timestamped experiment directory.

**Interfaces:**
- Produces: four return codes and source audits per platform for Official and
  Candidate pytest/benchmark.

- [ ] **Step 1: Run local syntax, formatting, and diff checks.**

  Compile every changed Python file and run `git diff --check`.

- [ ] **Step 2: Run 910C and Hygon generic PR-check wrappers.**

  Set `OP_NAME=addmm`, `OP_FUNC=addmm`, and
  `DTYPES="float32 float16 bfloat16"`.

- [ ] **Step 3: Run the S5000 editable-swap PR check in isolation.**

  Swap all candidate-dependent source files, clear the selected Triton cache,
  run pytest before benchmark, and verify trap-based restoration byte-for-byte.

- [ ] **Step 4: Summarize only reportable paired runs.**

  Require Official and Candidate pytest/benchmark return codes to be zero and
  all loaded paths/hashes to match the intended snapshots.

### Task 7: Run GraphCast correctness and performance acceptance

**Files:**
- Use: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-profile-routing-worktree/ai4s/graphcast/scripts/inference/diagnose_inference.py`
- Use the existing 21-layout replay harness and 7/7 GraphCast configuration.
- Store all generated artifacts outside the FlagGems PR diff.

**Interfaces:**
- Produces: fixed-reference correctness, weighted addmm replay, cold-start time,
  steady-state 40-step time, and MFU for each platform.

- [ ] **Step 1: Audit source and precision propagation in each container.**

  Record FlagGems Git SHA, vendor `addmm.py` SHA256, public operator source,
  vendor marker, seven enabled operators, tensor dtypes, and TF32/HF32 readback.

- [ ] **Step 2: Run addmm-only GraphCast diagnostics.**

  Require overall <5% and Z500 <5% against both fixed references.

- [ ] **Step 3: Run 7/7 GraphCast diagnostics.**

  Apply the same two-reference gate; do not substitute same-platform Native
  agreement for the fixed-reference criterion.

- [ ] **Step 4: Run weighted replay and three fresh 40-step processes.**

  Exclude compilation/cold start from the steady-state median, report cold
  start separately, and compute MFU from the fixed useful-work figure and each
  platform peak.

- [ ] **Step 5: Profile residual time on any platform below 16% MFU.**

  Rank the remaining six FlagGems operators plus launch/synchronization costs.
  Continue only with the largest source-audited residual bottleneck while
  preserving the same 5% correctness gate.

### Task 8: Final verification and delivery

**Files:**
- Update: `docs/superpowers/specs/2026-09-07-addmm-three-platform-correctness-performance-design.md` only if verified behavior differs from design.
- Create a local experiment report outside the PR diff with commands, revisions,
  hashes, raw result paths, correctness, and timing tables.

**Interfaces:**
- Produces: reviewable branch plus reproducible three-platform evidence.

- [ ] **Step 1: Run the final changed-test suite and `git diff --check`.**

- [ ] **Step 2: Inspect the complete branch diff for model-specific dispatch,
  native fallback masquerading as optimization, and unrelated files.**

- [ ] **Step 3: Report pass/fail separately for correctness, operator
  performance, and 16% end-to-end MFU on each platform.**

- [ ] **Step 4: Use the finishing-development-branch workflow to present local
  integration options without modifying the user's other worktrees.**
