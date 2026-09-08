# Feature Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax and must be checked off as completed.

**Goal:** Reduce Ascend 910C GraphCast 40-step latency by fusing the proven `layer_norm(x) + residual` node-update expression without changing model mathematics or weakening the dual-reference 5% correctness gate.

**Architecture:** Keep exact GraphCast profiling and integration in an isolated `ai4s_graphcast` campaign worktree. Add a common FlagGems semantic fallback plus an Ascend-specialized post-LayerNorm residual kernel selected by the existing `SpecOpRegistrar`. The fast path handles only measured contiguous inference layouts; every other call executes the unfused composition.

**Tech Stack:** Python 3.11+, PyTorch/Torch-NPU, FlagGems, Triton-Ascend, pytest, existing GraphCast operational 40-step runner and diagnostic scripts.

**Spec:** `docs/superpowers/specs/2026-09-08-ascend-residual-add-layernorm-design.md`

## Global Constraints

- Work only below `/mdata/data/zhangyuxin/wxt`.
- FlagGems worktree: `/mdata/data/zhangyuxin/wxt/FlagGems-ascend-residual-add-layernorm`, branch `yuanxi/ascend_residual_add_layernorm`, base `404be08bf60eacb8cd80436295285a8e8cd920f1`.
- Create the GraphCast campaign worktree at `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c` from clean commit `f74b909565bc7f70a0394bc37d7a0a5efc317970`; do not edit `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-profile-routing-worktree`, which contains unrelated user changes.
- The expression is `torch.layer_norm(x, ...) + residual`. `layer_norm(x + residual, ...)` is a different operation and must never be substituted.
- Use three excluded warmup calls and three independent measured 40-step runs; report the median and all raw repeats.
- Formal GraphCast acceptance requires 7/7 requested operators observed, finite outputs, valid source/hash audits, and both established reference errors below 5%.
- Exact GraphCast shapes, call ordinals and model names may exist only in the project campaign. FlagGems dispatch must use general dtype, rank, layout, normalized-width, affine, grad and alias properties.

---

## Task 1: Isolate the GraphCast campaign and freeze source provenance

**Files:**

- Create: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md`
- Verify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/tests/test_operational_40step_runner.py`

- [ ] Create the clean project worktree and branch.

```bash
git --git-dir=/mdata/data/zhangyuxin/wxt/ai4s-graphcast-profile-routing.git \
  worktree add -b yuanxi/910c_post_layernorm_residual \
  /mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c \
  f74b909565bc7f70a0394bc37d7a0a5efc317970
```

Expected: the new worktree is clean and the original dirty worktree is unchanged.

- [ ] Run the existing project tests that bind platform, precision and the 40-step contract.

```bash
cd /mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast
pytest -q tests/test_operational_40step_runner.py \
  tests/test_fixed_baseline_diagnostics.py \
  tests/test_post_layer_norm_residual_fusion.py
```

Expected: all selected CPU tests pass.

- [ ] Record the GraphCast commit, FlagGems commit, Python executable, device identity, Torch/Torch-NPU versions, precision mode, model execution profile, config and asset SHA-256 values in the experiment document.

- [ ] Commit the campaign skeleton.

```bash
git add ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md
git commit -m "docs(graphcast): start 910c post-layernorm residual campaign"
```

---

## Task 2: Add a non-synchronizing candidate-region profiler

**Files:**

- Create: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/src/graphcast_compat/fusion_candidate_profile.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/src/graphcast_compat/google_small.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/scripts/inference/run_operational_40step_inference.py`
- Create: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/tests/test_fusion_candidate_profile.py`

- [ ] Write failing tests for a disabled recorder, shape/layout aggregation, candidate call counts and deferred CUDA-event resolution.

```python
def test_post_layernorm_recorder_groups_without_item_or_synchronize(monkeypatch):
    recorder = PostLayerNormResidualRecorder(output_path="events.json")
    x = torch.empty((2, 3, 4), dtype=torch.float32)
    residual = torch.empty_like(x)
    recorder.observe(x, residual, (4,), torch.empty(4), torch.empty(4), 1e-5)
    assert recorder.groups[0].count == 1
    assert recorder.groups[0].shape == (2, 3, 4)
    assert recorder.groups[0].x_stride == (12, 4, 1)
```

- [ ] Run the test and confirm it fails because `PostLayerNormResidualRecorder` does not exist.

```bash
pytest -q tests/test_fusion_candidate_profile.py
```

Expected: import or attribute failure naming the missing recorder.

- [ ] Implement `PostLayerNormResidualRecorder` so ordinary `observe()` calls only append metadata and device event pairs; synchronize once in `stop()` before calculating totals. Reject recording if `x` and `residual` are on different devices.

- [ ] Add an optional `post_layer_norm_residual_observer` callable to `GoogleSmallGraphNet`. Invoke it immediately before the existing unfused `layer_norm(x) + residual` expression and only for the already-proven single-consumer node-update path.

```python
if self.post_layer_norm_residual_observer is not None:
    self.post_layer_norm_residual_observer(
        x, residual, layer_norm.normalized_shape,
        layer_norm.weight, layer_norm.bias, layer_norm.eps,
    )
return layer_norm(x) + residual
```

- [ ] Add runner flags `--post-layernorm-residual-profile` and `--post-layernorm-residual-profile-path`. Mark instrumented runs as performance-ineligible in `report.json`; preserve uninstrumented execution by default.

- [ ] Run the focused tests.

```bash
pytest -q tests/test_fusion_candidate_profile.py \
  tests/test_post_layer_norm_residual_fusion.py \
  tests/test_operational_40step_runner.py
```

Expected: all selected tests pass.

- [ ] Commit the profiler.

```bash
git add ai4s/graphcast/src/graphcast_compat/fusion_candidate_profile.py \
  ai4s/graphcast/src/graphcast_compat/google_small.py \
  ai4s/graphcast/scripts/inference/run_operational_40step_inference.py \
  ai4s/graphcast/tests/test_fusion_candidate_profile.py
git commit -m "feat(graphcast): profile post-layernorm residual candidates"
```

---

## Task 3: Run final-source 910C profile and leave-one-operator-out gate

**Files:**

- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md`
- Create: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/tools/post_layernorm_residual/summarize_campaign.py`
- Create: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/tests/test_post_layernorm_residual_summary.py`

- [ ] Write failing summary tests using fixed JSON fixtures for raw repeats, median calculation, dispatch coverage, eligible weighted time and rejection of mixed source hashes.

- [ ] Implement the summarizer with these required modes: `native`, `flaggems-7of7`, `minus-add`, `minus-native-layer-norm`, `minus-both`, and `instrumented-7of7`.

- [ ] Run the six modes on 910C with the accepted 25 km assets, `domestic-v1`, FP32 tensors and the campaign's fast-FP32 control. Each timing mode uses three warmups and three 40-step repeats. The instrumented mode is separate and excluded from E2E comparison.

```text
native
addmm,cat,native_layer_norm,index_add_,silu,index,add
addmm,cat,native_layer_norm,index_add_,silu,index
addmm,cat,index_add_,silu,index,add
addmm,cat,index_add_,silu,index
```

- [ ] Run `diagnose_inference.py` for every measured candidate output and retain both reference percentages, dispatch counts, source paths and hashes.

- [ ] Apply the gate: proceed to the fused operator only when the eligible candidate region is observed, has stable layout classes, and its weighted isolated time plus ablation delta is larger than run-to-run noise. Otherwise commit a no-go report and move the 910C effort to `linear + silu` profiling without adding a FlagGems fused API.

- [ ] Commit the evidence and summarizer.

```bash
git add ai4s/graphcast/tools/post_layernorm_residual \
  ai4s/graphcast/tests/test_post_layernorm_residual_summary.py \
  ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md
git commit -m "perf(graphcast): establish 910c fusion gate"
```

---

## Task 4: Define the common FlagGems semantic API by tests

**Files:**

- Create: `tests/test_post_layer_norm_residual.py`
- Create: `src/flag_gems/fused/post_layernorm_residual.py`
- Modify: `src/flag_gems/fused/__init__.py`

- [ ] Write parameterized CPU-capable semantic tests for affine/non-affine FP32, FP16 and BF16; one and multiple trailing normalized dimensions; non-contiguous inputs; empty leading dimensions; custom epsilon; grad-enabled inputs; and invalid shapes.

```python
expected = torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual
actual = flag_gems.post_layer_norm_residual(
    x, residual, normalized_shape, weight, bias, eps
)
torch.testing.assert_close(actual, expected)
```

- [ ] Add an order-sensitivity regression test whose inputs make `layer_norm(x) + residual` observably different from `layer_norm(x + residual)`.

- [ ] Run the focused tests and confirm failure because the public symbol is missing.

```bash
pytest -q tests/test_post_layer_norm_residual.py
```

Expected: `AttributeError: module 'flag_gems' has no attribute 'post_layer_norm_residual'`.

- [ ] Implement the common fallback exactly as a composition and export it.

```python
def post_layer_norm_residual(
    x, residual, normalized_shape, weight=None, bias=None, eps=1e-5
):
    return torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual
```

- [ ] Run focused tests and the existing skip-LayerNorm suite to prove the two APIs remain distinct.

```bash
pytest -q tests/test_post_layer_norm_residual.py tests/test_skip_layer_norm.py
```

Expected: all selected tests pass or device-only cases skip with their existing reason.

- [ ] Commit the public API.

```bash
git add src/flag_gems/fused/post_layernorm_residual.py \
  src/flag_gems/fused/__init__.py tests/test_post_layer_norm_residual.py
git commit -m "feat(fused): add post-layernorm residual fallback"
```

---

## Task 5: Probe the Ascend vendor primitive before writing a kernel

**Files:**

- Create: `tools/probes/probe_ascend_post_layernorm_residual.py`
- Create: `tests/test_ascend_post_layernorm_residual_probe.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md`

- [ ] Write a source-level test requiring the probe to report operation name, callable schema and a numerical comparison against both operation orders.

- [ ] Implement the probe by enumerating Torch-NPU operator schemas containing both `layer_norm` and `add`/`residual`, then execute each callable candidate on deterministic tensors.

```python
post = torch.layer_norm(x, (x.shape[-1],), weight, bias, eps) + residual
pre = torch.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)
```

The probe accepts a primitive only when it matches `post`, does not merely match `pre`, supports affine tensors and preserves the required output dtype/layout.

- [ ] Run the probe on 910C and append its JSON result to the experiment record.

- [ ] If a matching primitive exists, implement the Ascend wrapper around it in Task 6. If none matches, delete no evidence and implement the Triton candidate in Task 6.

- [ ] Commit the reproducible probe.

```bash
git add tools/probes/probe_ascend_post_layernorm_residual.py \
  tests/test_ascend_post_layernorm_residual_probe.py
git commit -m "test(ascend): probe post-layernorm residual primitives"
```

---

## Task 6: Implement the Ascend fast path with model-independent guards

**Files:**

- Create: `src/flag_gems/runtime/backend/_ascend/fused/post_layernorm_residual.py`
- Modify: `src/flag_gems/runtime/backend/_ascend/fused/__init__.py`
- Modify: `tests/test_post_layer_norm_residual.py`

- [ ] Add failing dispatch tests for the measured fast region and fallbacks: grad enabled, mismatched shape/dtype/device, non-contiguous trailing normalized region, normalized width above 4096, and unsupported affine form.

- [ ] Implement `_can_use_fast_path(...)` using only general properties. Start with the measured region: identical shape/dtype/device, contiguous inputs, trailing normalized dimensions, supported device dtype, `N <= 4096`, matching affine tensors and inference mode.

- [ ] Implement the selected candidate. For the Triton branch, derive the row reduction from the existing Ascend LayerNorm kernels, compute statistics from `x` only in FP32, and add `residual` after normalization and affine transformation immediately before `tl.store`.

```python
normalized = (x_value - mean) * rstd
if HAS_WEIGHT:
    normalized *= weight_value
if HAS_BIAS:
    normalized += bias_value
output = normalized + residual_value
tl.store(output_ptr + offsets, output, mask=mask)
```

- [ ] Route unsupported calls to the common fallback. Do not call Ascend `skip_layer_norm`, because it implements the opposite operation order.

- [ ] Export the Ascend symbol so `SpecOpRegistrar` replaces the common implementation by matching name.

- [ ] Run focused static and device tests.

```bash
pytest -q tests/test_post_layer_norm_residual.py \
  tests/test_skip_layer_norm.py tests/test_layer_norm.py
```

Expected: semantic tests pass; fast-path tests show the Ascend implementation for eligible inputs and common composition for fallback inputs.

- [ ] Commit the hardware implementation.

```bash
git add src/flag_gems/runtime/backend/_ascend/fused/post_layernorm_residual.py \
  src/flag_gems/runtime/backend/_ascend/fused/__init__.py \
  tests/test_post_layer_norm_residual.py
git commit -m "perf(ascend): fuse post-layernorm residual inference"
```

---

## Task 7: Add official-style correctness cases and operator benchmarks

**Files:**

- Modify: `tests/test_post_layer_norm_residual.py`
- Create: `benchmark/test_post_layer_norm_residual.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md`

- [ ] Convert every measured GraphCast layout class into a moderate model-independent pytest case and a matching benchmark case. Include cases immediately below and above the fast-path width boundary.

- [ ] Benchmark complete calls for `Torch composition`, `Official FlagGems composition`, and `Candidate fused`, with warmups excluded and all raw timings retained.

- [ ] Require Candidate to beat the current composition on the weighted real-case total and reject any unexplained public case regression greater than 5%.

- [ ] Run the focused test and benchmark suites on 910C and save machine-readable JSON plus the human-readable table.

```bash
pytest -q tests/test_post_layer_norm_residual.py
pytest -q -s benchmark/test_post_layer_norm_residual.py
```

- [ ] Commit benchmark coverage and evidence references.

```bash
git add benchmark/test_post_layer_norm_residual.py \
  tests/test_post_layer_norm_residual.py
git commit -m "bench(ascend): cover post-layernorm residual cases"
```

---

## Task 8: Enable the fused path only at the proven GraphCast seam

**Files:**

- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/src/graphcast_compat/google_small.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/scripts/inference/run_operational_40step_inference.py`
- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/tests/test_post_layer_norm_residual_fusion.py`

- [ ] Extend the existing tests to prove exactly one fused call for each eligible single-consumer node update, no fused call for edge deltas, numerical parity with the unfused path and a clear error when the selected FlagGems source lacks the API.

- [ ] Wire the already-existing `fuse_post_layer_norm_residual` flag into the operational runner and include the flag and loaded symbol module in `report.json` source audits.

- [ ] Run integration tests.

```bash
pytest -q tests/test_post_layer_norm_residual_fusion.py \
  tests/test_operational_40step_runner.py \
  tests/test_fixed_baseline_diagnostics.py
```

Expected: all tests pass and the default unfused execution remains unchanged.

- [ ] Commit the project integration.

```bash
git add ai4s/graphcast/src/graphcast_compat/google_small.py \
  ai4s/graphcast/scripts/inference/run_operational_40step_inference.py \
  ai4s/graphcast/tests/test_post_layer_norm_residual_fusion.py
git commit -m "feat(graphcast): enable 910c post-layernorm residual fusion"
```

---

## Task 9: Run final 910C acceptance and produce the decision

**Files:**

- Modify: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-next-round-910c/ai4s/graphcast/docs/experiments/2026-09-08-910c-post-layernorm-residual.md`
- Create: `docs/benchmarks/2026-09-08-ascend-post-layernorm-residual.md`

- [ ] Run Native, final 7/7 unfused and final 7/7 fused with identical assets, `domestic-v1`, three excluded warmups and three independent 40-step measurements.

- [ ] For every fused repeat, run the established offline diagnostic and require finite output, 7/7 coverage, both reference errors below 5%, correct loaded source roots/hashes and observed `post_layer_norm_residual` calls.

- [ ] Calculate E2E median, MFU, absolute saved time, relative speedup and distance to the 36.253 s / 16% MFU target. Keep all raw runs in the report.

- [ ] Run final FlagGems regression tests.

```bash
pytest -q tests/test_post_layer_norm_residual.py \
  tests/test_skip_layer_norm.py tests/test_layer_norm.py \
  tests/test_addmm.py tests/test_addmm_precision_policy.py
```

Expected: all selected tests pass; device-only tests may skip only for their declared capability reason.

- [ ] Mark the result `Ready` only if all correctness, source, dispatch and performance gates pass. Otherwise mark `Not Ready`, retain the measured evidence and name the first failed gate.

- [ ] Commit the final report separately.

```bash
git add docs/benchmarks/2026-09-08-ascend-post-layernorm-residual.md
git commit -m "docs(ascend): report post-layernorm residual results"
```

---

## Task 10: Final review and handoff

**Files:**

- Review: all files changed on `yuanxi/ascend_residual_add_layernorm`

- [ ] Inspect `git diff 404be08bf60eacb8cd80436295285a8e8cd920f1...HEAD` and confirm there are no GraphCast names, exact production dimensions, call ordinals, hard-coded asset paths or per-call synchronizations in FlagGems source.

- [ ] Run formatting/lint commands used by the repository on every changed Python file.

- [ ] Verify both worktrees are clean, record exact commits and source hashes, and prepare the FlagGems commit for push only after the user requests publication.
