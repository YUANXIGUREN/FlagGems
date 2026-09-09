# Triton-Only AddMM and Ascend Add Design

## Goal

Replace the previously delivered Native-routed GraphCast hot paths with
FlagGems-owned Triton kernels on Ascend 910C, Hygon BW3000, and Moore Threads
S5000.  The final GraphCast 25 km, 40-step, seven-operator run must pass the
fixed 5% correctness gate and reach at least 16% MFU on every target platform.

If standalone `addmm` and `add` kernels cannot reach the end-to-end target,
Triton-only fusion with their measured neighboring operations is in scope.
Calling a Torch or vendor Native compute operator is never an acceptable way
to meet the performance target.

## Starting point and delivery branch

Development starts from unmodified FlagGems v5.3.5 commit
`a7620cc191a0b42e040194622c5758b22a7a25dc` in the isolated worktree
`/mdata/data/zhangyuxin/wxt/FlagGems/.worktrees/triton-only-addmm-three-platform-ascend-add-v535`
on branch `codex/triton-only-addmm-three-platform-ascend-add-v535`.

The prior candidate commits remain evidence of rejected Native-routing
experiments, not source to merge blindly:

- three-platform AddMM: `404be08bf60eacb8cd80436295285a8e8cd920f1`;
- Ascend Add: `93624d8010530a8341124e55992a5f6d27980695`.

Only independently reviewed Triton changes may be ported from those commits.
This includes precision-policy parsing, valid `tl.dot` precision arguments,
fused bias epilogues, `beta=0` handling, layout classification, Triton
preprocessing, and measured tuning configurations.  Native routing helpers and
their tests are excluded.

## Why the previous results are rejected

The previously reported GraphCast hot paths entered vendor compute before the
available Triton kernels:

| Platform and operator | Rejected hot path |
| --- | --- |
| Ascend 910C AddMM | `torch.ops.npu.npu_linear.default` |
| Hygon BW3000 AddMM | captured CUDA/HIP kernel through `get_kernel` and `call_boxed` |
| Moore Threads S5000 AddMM | captured PrivateUse1/MUSA kernel through `get_kernel` and `call_boxed` |
| Ascend 910C Add | `aten.add.Tensor.redispatch` |

Consequently the S5000 `31.751 s / 16.22%` and Ascend
`30.938 s / 18.75%` figures are not Triton-only acceptance evidence.  They may
be retained only as historical upper-bound references.  The corresponding
`wip-addmm` and `wip-add-ascend` packages must not be presented as compliant
with this design.

## Operator contract

### AddMM

The public API is
`torch.addmm(input, mat1, mat2, *, beta=1, alpha=1)`.  The implementation must
preserve:

- two-dimensional `mat1 @ mat2` with compatible K dimensions;
- scalar, vector, row, column, and broadcastable matrix bias;
- arbitrary valid `alpha` and `beta`, including not reading bias for `beta=0`;
- row-contiguous, column-contiguous, padded transpose views, tails, and general
  valid strides;
- default, `out`, `dtype`, and `dtype_out` variants where registered;
- the supported FP32, FP16, BF16, and FP64 semantics of each backend;
- autograd behavior for APIs and dtypes supported by upstream FlagGems.

GraphCast uses FP32 tensors with one-dimensional bias and mostly
`alpha=beta=1`.  One step contains 21 distinct shape/layout signatures and 264
AddMM calls.  The dominant region has `N=512`, `K` in `{184, 512, 1024}`, very
large M, and RHS strides `[1, K]` or padded `[1, 1536]`.  Skinny `K=4` calls
are frequent enough to require a separate kernel strategy.  Exact production
dimensions stay in the project replay harness, not source dispatch.

### Ascend Add

The public API in scope is `torch.add.Tensor(A, B, *, alpha=1)`.  The optimized
region is same-device, same-shape FP32 tensor addition, including the real
GraphCast non-contiguous views:

- six calls per step shaped `[1, 1038240, 13]`;
- four calls per step shaped `[1, 1038240, 1]`;
- observed left stride `[182730240, 176, 1]`.

The implementation must preserve `A + alpha * B`, broadcasting, supported
dtypes, non-contiguous inputs, output shape and dtype, and autograd semantics.
Unsupported specialized layouts fall through to the common FlagGems Triton
pointwise implementation, never to Torch Native.

## Hard no-Native rule

The target AddMM and Add source paths must not use a Native compute route,
directly or indirectly.  In particular they must not invoke:

- `Tensor.redispatch` or another ATen redispatch for the target computation;
- `torch.library.get_kernel`, a captured dispatcher kernel, or `call_boxed`;
- `torch.ops.npu.npu_linear`, `torch.addmm`, `torch.mm`, `torch.matmul`, or
  `torch.add` to perform the target computation;
- vendor BLAS, CANN, MUSA, MUDNN, HIP, or CUDA extension calls for the target
  computation.

Tensor allocation, metadata operations, and layout views are allowed.
Materialization or precision conversion is allowed only through a FlagGems
Triton kernel when it contributes to the optimized path.  A fallback to an
existing FlagGems Triton kernel is allowed for semantics outside a specialized
kernel's region.

Static policy checks supplement, but do not replace, runtime proof.  Device
profiles must show the Triton-generated Add/AddMM kernels and must not show a
vendor Add, Linear, GEMM, or AddMM compute kernel servicing the measured call.

## Chosen architecture

The selected approach is backend-specific Triton kernels plus backend-specific
tuning.  A single common kernel with only separate config tables is retained as
a correctness fallback, but it is not expected to close the measured gaps.
Model-level rewriting is not the first step.  Triton fusion is activated only
after standalone-kernel profiling proves the remaining end-to-end gap and
identifies an exact neighboring composition.

Dispatch uses stable, model-independent properties: dtype, rank, matrix
dimensions, stride class, bias form, precision policy, and gradient state.  It
must not contain the GraphCast name or an exact production M dimension.

## Ascend 910C AddMM

The rejected `npu_linear` branch is removed.  All AddMM variants enter a
FlagGems Triton kernel.

For FP32 matrix-unit cases, `tl.dot` uses compile-time
`input_precision="hf32"` only when `torch.npu.matmul.allow_hf32` is enabled;
otherwise it uses `input_precision="ieee"`.  FP16, BF16, and FP64 retain their
documented upstream precision behavior.

Two Triton layouts are evaluated independently:

1. A grouped large-M GEMM kernel for `K >= 16`, with a fused vector-bias
   epilogue.  Its sweep varies M/N/K tile sizes, M grouping, stages, and the
   supported Ascend execution width.  Row-contiguous and K-contiguous/padded
   RHS classes receive independent tune keys.
2. A skinny-K kernel for small K, especially K=4.  It computes several output
   rows per program and avoids allocating a mostly empty matrix-unit K tile.

Configurations are selected from crossover sweeps around general dimension
and stride classes.  Exact production M values are not dispatch boundaries.
General layouts use the original Ascend Triton implementation after any
necessary Triton materialization.

## Hygon BW3000 AddMM

The captured CUDA/HIP kernels and process-wide Native precision toggle are
removed.  The public vendor entry calls only direct Triton kernels.

The large-M path uses a fused bias epilogue and separates row-contiguous RHS
from K-contiguous and padded-K-contiguous RHS.  Direct strided loads are kept
when their measured bandwidth beats materialization; otherwise a Triton copy
or pack kernel is permitted only when reuse amortizes its measured cost.

Fast FP32 remains a Triton precision choice.  Existing group diagnostics show
that full fast-FP32 execution can accumulate unacceptable autoregressive
error.  Therefore the experiment compares:

- strict FP32 `tl.dot` for all calls;
- fast-FP32 `tl.dot` for independently validated semantic/layout groups;
- a cumulative combination of individually safe groups.

The final selector is based on stable K/N/layout/bias properties demonstrated
by the group experiment, not a call ordinal or model identity.  Every selected
combination must pass the full two-reference 40-step diagnostic.  A faster
combination that fails the diagnostic is rejected.

## Moore Threads S5000 AddMM

The captured PrivateUse1/MUSA AddMM path is removed.  FP32 AddMM enters a
Triton `tl.dot` or Triton SQMMA-compatible kernel.

For fast FP32, a Triton preprocessing kernel performs IEEE TF32
round-to-nearest-even once per logical element.  Changing activations are
converted per call.  Stable inference RHS tensors may reuse a bounded,
version-aware cache of Triton-rounded tensors; mutations, storage changes,
metadata changes, gradients, and expired owners invalidate or bypass the
cache.  The subsequent matrix multiplication remains a Triton kernel.

Two implementations are benchmarked:

1. pointer-based tiled `tl.dot` with `allow_tf32=True` over pre-rounded data;
2. the backend-supported descriptor/SQMMA Triton form for eligible contiguous
   or safely packed FP32 matrices.

The accepted implementation is the faster correct choice per generalized
layout class.  Tile size, execution width, stages, grouped scheduling, and
skinny-K handling are swept independently.  No performance number from a run
that entered MUSA Native AddMM is reportable.

## Ascend 910C Add

The `aten.add.Tensor.redispatch` route is removed.  A direct same-shape
strided Triton kernel handles the GraphCast-representative region.

The kernel treats the contiguous suffix and outer rows separately.  Programs
cover multiple rows and a power-of-two suffix tile, using the supplied outer
and suffix strides directly.  This avoids the generic pointwise path's repeated
multi-dimensional div/mod address reconstruction.  Contiguous inputs use a
flat vectorized Triton path; broadcast or unsupported strided cases use the
common FlagGems Triton implementation.

The sweep varies rows per program, suffix block, and execution width across
contiguous, suffix-1, suffix-13, tail, and moderate model-independent cases.
The route is expressed as a stride pattern, not the production tensor shape.

## Optional Triton fusion phase

Fusion begins only if all standalone target kernels are correct and tuned but
a platform remains below 16% MFU.  A fresh profiler run must identify the
neighboring operations responsible for the remaining device time or launch
overhead.

Permitted candidates include `addmm + silu`, `addmm + bias + activation`, and
`layer_norm + residual add` when the actual model trace proves adjacency and
the public integration can preserve semantics.  Each fusion is one
`triton.jit` implementation with no Native sub-operation.  It requires its own
composition correctness tests, operator/kernel benchmark, dispatch audit, and
GraphCast ablation.  A fusion is not adopted merely because it is plausible.

If even validated fusion cannot meet the target, the result is reported as a
measured limitation with the remaining hotspot; Native redispatch is not
reintroduced to manufacture a passing MFU.

## Correctness and precision gates

Testing proceeds before performance measurement:

1. Public AddMM tests cover supported dtypes, all bias forms, alpha/beta,
   `beta=0` special values, tails, row/column/padded/general layouts, empty K,
   output variants, and gradients where supported.
2. Public Add tests cover tensor/tensor, alpha, broadcasting, contiguous and
   non-contiguous layouts, tails, special values, supported dtypes, and
   gradients where supported.
3. FP32 fast-mode tests verify backend policy mapping.  S5000 rounding is
   checked bit-for-bit against an independent nearest-even oracle.
4. Source and runtime tests prove that target hot calls cannot reach the
   prohibited Native interfaces.
5. GraphCast AddMM-only, Add-only, then seven-operator runs compare against
   both fixed references.  Overall, Z500, and the established worst-layer
   metrics must each remain strictly below 5%.

An unexplained operator failure or GraphCast diagnostic failure blocks all
candidate performance reporting.

## Performance protocol

Every platform run records the FlagGems commit, dirty-diff hash, loaded module
paths, target source hashes, Python/Torch/Triton versions, device identity,
precision-policy readback, cache directory, warmup count, iteration count, and
commands.

Three evidence layers are required:

1. Official FlagGems benchmark: Torch, unmodified v5.3.5, and Triton-only
   candidate for all configured FP32, FP16, and BF16 cases.
2. Exact model replay: all 21 AddMM shape/layout signatures weighted by 264
   calls per step, and all ten real strided Add calls per step.
3. GraphCast 25 km E2E: AddMM-only, Add-only where applicable, then all seven
   requested FlagGems operators for 40 autoregressive steps.

Compilation and three warmup model runs are excluded from steady-state E2E and
reported separately as cold-start evidence.  Final E2E is the median of at
least three fresh, non-overlapping processes on an idle device.  Dispatch is
validated from source path, FlagGems record, and profiler kernel names.

## MFU acceptance thresholds

Using 1,154.283779 TFLOP of useful work per 40-step run and the supplied device
peaks, 16% MFU requires:

| Platform | Peak | Maximum median 40-step E2E |
| --- | ---: | ---: |
| Ascend 910C | 199 TFLOPS | 36.2526 s |
| Hygon BW3000 | 236.94 TFLOPS | 30.4477 s |
| Moore Threads S5000 | 224.07 TFLOPS | 32.1965 s |

Acceptance is per platform, not a geometric mean.  A platform passes only when:

- the relevant public operator suites pass;
- the two-reference GraphCast diagnostics pass below 5%;
- all seven requested operators are recorded as active;
- the target AddMM/Add calls are proven to execute Triton-generated kernels;
- no prohibited Native target compute appears in source or profiler evidence;
- the median steady-state 40-step E2E meets the platform threshold.

BW3000 Native E2E itself was previously measured at 39.777 seconds, or 12.25%
MFU under the supplied peak.  This makes 16% an aggressive cross-operator
target, not something AddMM alone is assumed to guarantee.  The optional
Triton fusion phase and renewed residual-hotspot profile are therefore part of
the agreed route to the target.

## Delivery

The final source delivery contains only FlagGems source, public correctness
tests, and model-independent benchmark additions required to cover new
branches.  Exact GraphCast cases, raw timings, source audits, profiler evidence,
and E2E diagnostics are stored in the evidence repository.

The old Native-routing packages remain historical and receive an explicit
superseded warning when the Triton-only replacement package is ready.  They are
not silently overwritten, and no compliant claim is made until all gates above
pass.
