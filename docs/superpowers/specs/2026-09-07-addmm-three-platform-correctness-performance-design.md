# AddMM Three-Platform Correctness and Performance Design

## Goal

Make the FlagGems `torch.addmm` path numerically acceptable for GraphCast and
reduce 40-step end-to-end time enough to reach at least 16% MFU on Ascend 910C,
Hygon BW3000, and Moore Threads S5000. Correctness is a hard gate: both fixed
GraphCast references must remain below 5% for the overall-layer metric and the
Z500 metric.

## Scope and contract

The public operator is `torch.addmm(input, mat1, mat2, *, beta=1, alpha=1)` as
registered by FlagGems. The implementation must preserve:

- 2-D matrix multiplication with `mat1.shape[1] == mat2.shape[0]`;
- scalar, row-vector, column-vector, and broadcastable 2-D bias;
- `alpha` and `beta`, including `beta=0` without reading NaN bias data;
- row-major, column-major, padded, and general-stride inputs;
- `addmm`, `addmm.out`, `addmm.dtype`, and `addmm.dtype_out` where registered;
- strict FP32, fast FP32, FP16, BF16, and FP64 behavior supported by a backend.

GraphCast exact shapes and call weights remain in the project replay harness.
Upstream dispatch must be expressed through dtype, layout, precision policy,
gradient state, and stable tensor properties; it must not mention GraphCast or
hard-code a production dimension.

## Measured workload and target

One GraphCast autoregressive step contains 21 distinct `addmm` shape/layout
signatures and 264 calls. The dominant region has FP32 inputs, vector bias,
`N=512`, `K` equal to 512 or 1024, and both contiguous and transposed/split
weight layouts. It covers about 90% of weighted `addmm` time.

For 1,154.283779 TFLOP of useful work per 40-step run, 16% MFU requires:

| Platform | Current FlagGems E2E | Required E2E |
| --- | ---: | ---: |
| 910C | 49.1680 s | <= 36.2526 s |
| BW3000 | 47.0432 s | <= 30.4477 s |
| S5000 | 49.2500 s | <= 32.1965 s |

## Precision policy

`tf32` is a fast-FP32 policy, not a tensor dtype. Model state, weights, inputs,
and outputs remain `torch.float32`. A shared policy reader maps runtime settings
to a backend-specific matrix-multiply mode:

- Ascend: `torch.npu.matmul.allow_hf32` -> `input_precision="hf32"`; otherwise
  `input_precision="ieee"`.
- Hygon: `torch.backends.cuda.matmul.allow_tf32` -> `allow_tf32=True`; otherwise
  `False`.
- Moore Threads: `torch.backends.mudnn.allow_tf32` -> TF32 nearest-even input
  rounding plus a fast `tl.dot`; otherwise use the existing strict path.

Only FP32 x FP32 is eligible for fast FP32. FP16, BF16, FP64, mixed dtypes, and
strict-FP32 requests preserve their existing paths.

## Platform design

### Ascend 910C

The current vendor kernel hard-codes `allow_tf32=False`. Add a compile-time
`INPUT_PRECISION` argument and choose `"hf32"` only when the runtime policy is
enabled for two FP32 operands. This is a vendor change because the precision
spelling and hardware path are Ascend-specific. Existing layout handling,
tiling configuration, and strict fallback remain unchanged for the first
candidate so HF32 is the only experimental variable.

After correctness passes, tune the existing Ascend configuration mechanism for
the model-independent large-M, N/K-multiple-of-16 region. A configuration is
accepted only when a sweep on both sides of its dispatch boundary demonstrates
a stable crossover.

### Hygon BW3000

Current upstream already has a Hygon-specific `addmm` that reads the CUDA TF32
policy. First rebaseline this exact source and prove that the vendor kernel is
registered and loaded. Do not replace it with Native: the recorded Native
trajectory fails the fixed reference gate and Native E2E is itself below 16%
MFU.

If the current kernel remains below target, tune through the existing Hygon
`addmm` configuration pool for the generalized large-M, N/K-aligned region.
Preserve direct strided RHS loads when they win; materialize or prepack only
when the measured reuse amortizes the copy. Bias specialization stays fused in
the epilogue.

### Moore Threads S5000

The strict FMA path is correct but slow. The prior WIP establishes that explicit
TF32 nearest-even rounding can pass the fixed GraphCast gate, but performing
bit manipulation inside every output tile repeats RHS conversion across a very
large number of M tiles.

Use two stages for fast FP32:

1. A layout-preserving preprocessing kernel rounds FP32 values to TF32
   nearest-even once per logical element.
2. The existing fast dot kernel consumes pre-rounded FP32 data without repeating
   bit conversion inside the K loop.

Transient activation operands are rounded per call. Stable no-grad RHS weights
are cached by original base-tensor identity plus storage pointer, storage
offset, version, dtype, device, shape, and stride. Each cache entry retains only
a weak reference to the original base tensor and a strong reference to the
rounded device tensor. Lookup verifies object identity and the full signature;
an expired reference, version change, storage change, or metadata change is a
miss. Training, grad-enabled execution, tensors requiring gradients, and unsafe
views bypass the cache and use an uncached or strict path. The cache has a fixed
entry bound and evicts least-recently-used entries to bound device memory.

If a separate activation preprocessing pass costs more than in-kernel rounding
for a small matrix, retain the existing in-kernel path below a measured semantic
crossover. The dispatch condition uses operand byte size and tile-reuse count,
not an exact model shape.

## Correctness gates

Tests are performed after each candidate change and before accepting timing:

1. CPU policy tests prove backend settings map to the intended compile-time
   mode and strict mode remains strict.
2. Device operator tests compare public `torch.addmm` under
   `flag_gems.use_gems()` with Native Torch for supported dtypes, bias forms,
   alpha/beta values, tails, layouts, special values, and out variants.
3. TF32 rounding tests compare against a bit-exact host nearest-even oracle,
   including ties, subnormals, infinities, and NaNs.
4. Cache tests cover hit, miss, view reuse, in-place weight mutation, storage
   reuse defense, weak-reference expiry, grad bypass, and bounded eviction.
5. GraphCast diagnostics require both fixed references to report overall <5%
   and Z500 <5%.

A candidate with an unexplained correctness failure is not benchmarked or
reported as an optimization result.

## Performance protocol

- Record loaded FlagGems package, public operator path, vendor source path, Git
  revision, and SHA256 before every run.
- Compare Native Torch, unmodified upstream FlagGems, and Candidate FlagGems in
  isolated source trees.
- Run official FlagGems benchmark coverage for FP32, FP16, and BF16.
- Replay all 21 real shape/layout signatures with 264-call weights.
- Run GraphCast with only `addmm` enabled, then with all seven operators.
- Keep TF32/HF32 readback in each log.
- Exclude compilation and cold start from steady-state timing; report cold start
  separately.
- Use at least three fresh processes and report median plus range for 40-step
  end-to-end time.

## Acceptance

For each of 910C, BW3000, and S5000:

- operator pytest and source audit pass;
- both GraphCast fixed-reference metrics pass the 5% gate;
- all seven requested FlagGems operators are confirmed active;
- median steady-state 40-step MFU is at least 16%.

If a platform reaches the measured `addmm` limit while whole-model MFU remains
below 16%, profile the residual six operators and launch/synchronization costs.
Any follow-on change keeps the same correctness-first and source-audited gates.
