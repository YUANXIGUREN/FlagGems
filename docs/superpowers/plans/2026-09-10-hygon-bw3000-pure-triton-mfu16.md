# Hygon BW3000 Pure-Triton GraphCast MFU 16% Plan

**Goal:** Starting from the accepted three-platform Triton-only branch, make the
single-device GraphCast 25 km 40-step seven-operator run pass both fixed
reference diagnostics below 5% and reach at least 16% MFU on the physical
BW3000 device.

**Metric:** Use 236.94 TFLOPS peak and 1,154,283,779,031,040 useful FLOP.  The
median of three fresh steady-state runs must therefore be no greater than
30.447681 seconds.  Three warm-up calls are excluded; final device-to-host
publication is included.

**Constraint:** Target compute is exclusively FlagGems-owned Triton.  No
`torch.addmm`, ATen redispatch, captured native kernel, rocBLAS, or vendor
extension is permitted.  Unsupported specializations fall back only to an
existing FlagGems Triton kernel.

## Task 1: Freeze the Current Formal Baseline

- Record FlagGems and GraphCast commits/diff hashes, operator source hashes,
  Torch/Triton/device identity, and precision readback.
- Run public Hygon AddMM correctness and the no-Native audit.
- Run the exact 21-signature AddMM replay three times.
- Run GraphCast 40-step 7/7 three times and both fixed-reference diagnostics.
- Reject any timing with incomplete 7/7 coverage or wrong source provenance.

## Task 2: TDD a Hygon Fast-Input Mode

- RED: require strict mode for non-FP32, disabled fast math, skinny K, general
  layout, and the known sensitive K512 classes; require a distinct fast mode
  only for explicitly eligible generalized classes.
- Implement in-kernel FP16 and BF16 matrix operands with FP32 accumulation as
  independent candidates.  Keep strict IEEE FP32 as the semantic fallback.
- Add public correctness for vector/scalar/matrix bias, alpha/beta, beta zero,
  tails, row RHS, K-contiguous RHS, padded RHS, non-finite inputs, and out/dtype
  overloads.
- Audit generated ISA and source to prove Triton MMAC execution with no native
  call.

## Task 3: Measure and Select, Do Not Guess

- Sweep K512 row/K-contiguous/padded and K1024+ independently; retain K4 on the
  existing skinny kernel.
- Compare strict, FP16-input/FP32-accumulate, and BF16-input/FP32-accumulate on
  all 21 exact shape/layout signatures with actual 40-step call weights.
- Run one-class-at-a-time and cumulative 40-step diagnostics.  Native-TF32
  sensitivity results determine experiment order only; they are not a Triton
  production allowlist.
- Promote only generalized K/N/layout/bias classes whose cumulative candidate
  remains strictly below 5% against both references and improves weighted
  replay.

## Task 4: Re-profile and Close the Remaining Amdahl Gap

- Run a fresh 7/7 profiler on the accepted AddMM candidate.
- If E2E remains above 30.447681 seconds, optimize the largest measured pure
  Triton opportunity in this order unless the new profile disproves it:
  true `addmm + bias + silu` epilogue fusion, contention-aware `index_add_`,
  then `addmm + residual`/`residual_add + layer_norm` fusion.
- Every fusion gets composition correctness, operator and kernel timing,
  dispatch/ISA audit, and a one-change GraphCast ablation before retention.

## Task 5: Formal Acceptance and Delivery

- Run complete AddMM and any changed-operator pytest suites for FP32, FP16, and
  BF16, plus all registered overloads.
- Run official benchmark, exact real-call replay, and three fresh GraphCast
  40-step 7/7 processes.
- Require dual-reference overall and Z500 error `<5%`, median E2E
  `<=30.447681 s`, MFU `>=16%`, finite outputs, and no Native target compute.
- Commit only source/tests/necessary benchmark cases; keep model-specific raw
  evidence outside the upstream source diff.
