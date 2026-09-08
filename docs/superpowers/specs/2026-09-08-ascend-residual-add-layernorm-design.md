# Ascend 910C Residual Add + LayerNorm Optimization Design

## Status and scope

This design starts from FlagGems commit
`404be08bf60eacb8cd80436295285a8e8cd920f1` on branch
`yuanxi/ascend_residual_add_layernorm`.

The work has two gates:

1. reproduce the final seven-operator GraphCast profile and prove that the
   residual-add/LayerNorm sequence has enough end-to-end impact;
2. only after that proof, add the smallest model-independent fused path and
   validate it against the unfused Torch composition.

The first implementation target is inference on Ascend 910C. Training and
unsupported layouts keep composition semantics through a fallback. The old
Triton `addmm_silu` WIP is not part of this work because it reached only
0.226x of the Torch-NPU split expression on the measured 910C environment.

## Evidence and target

The current correctness-valid 40-step GraphCast run has:

- FlagGems 7/7 E2E: 40.176 s;
- Native E2E: 34.963 s;
- FlagGems MFU: 14.44%;
- 16% MFU time target: 36.253 s;
- required reduction from the current result: 3.923 s.

The final AddMM weighted replay is already effectively at Native:
7.873 s versus 7.858 s. The next profile must therefore measure the remaining
six operators and host/launch overhead from the final source instead of
continuing to tune AddMM.

The 40-step dispatch record contains 6,506 `add` calls and 4,859
`native_layer_norm` calls. Those counts motivate the fusion hypothesis but do
not prove adjacency or savings. The profile and ablation gates below provide
that proof.

## Phase 1: final-source profile and ablation

Use the exact GraphCast 25 km inputs, weights, model execution profile and
seven-operator configuration from the accepted AddMM campaign. Every run must
record the loaded FlagGems root, commit, relevant source hashes, device
identity, precision controls and dispatch counts.

Run these modes with three excluded warmups and three independent 40-step
measurements:

1. Native control;
2. final FlagGems 7/7 control;
3. leave-`add`-native-out, with the other six operators still enabled;
4. leave-`native_layer_norm`-native-out;
5. leave both `add` and `native_layer_norm` native-out;
6. an instrumented final 7/7 run that records adjacent operator events and
   device kernels without synchronizing between ordinary calls.

The instrumented run must answer:

- how many `add` results flow directly into LayerNorm;
- their shapes, dtypes, strides, normalized shape, epsilon and affine use;
- device time and launch count for the pair;
- materialized intermediate bytes;
- whether the calls are inference-only and free of aliases or in-place writes.

Profiling instrumentation belongs to the GraphCast project harness, not to the
upstream FlagGems dispatch path.

## Fused operator contract

The proposed mathematical contract is:

```python
residual_add_layer_norm(
    x,
    residual,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-5,
) == torch.nn.functional.layer_norm(
    x + residual,
    normalized_shape,
    weight,
    bias,
    eps,
)
```

The output has the broadcasted add shape and input dtype. The initial fast
path requires `x` and `residual` to have identical shape, dtype and device;
the normalized dimensions must be a contiguous trailing region; affine
parameters, when present, must match `normalized_shape`.

The public wrapper preserves full composition semantics. Inputs outside the
validated fast region, non-contiguous general layouts, alias-sensitive calls,
or calls requiring an unsupported backward path execute the Torch/FlagGems
composition. The implementation must not mention GraphCast or exact production
dimensions.

Automatic eager fusion is not possible by independently registering `add` and
`native_layer_norm`. Model integration must call the fused API through the
GraphCast adapter, or a separately reviewed graph rewrite must replace the
proven expression.

## Candidate implementations

### A. Ascend vendor fused primitive (preferred when available)

Probe the installed Torch-NPU/CANN runtime for a primitive that computes
residual addition and LayerNorm in one supported operation. This has the best
chance of keeping AiCore/CANN code generation and low launch overhead.

Use it only when its numerical contract, optional affine parameters and layout
semantics match the public wrapper. If no matching primitive exists, this
option is rejected rather than emulated with two vendor calls.

### B. Ascend Triton one-pass row kernel

For a trailing normalized axis, one program or cooperative program group owns
one row: it loads `x` and `residual`, forms the sum in registers, accumulates
mean and variance in FP32, normalizes, applies affine parameters and stores the
output once. Configuration is selected from normalized width and row count.

This removes the intermediate tensor and one launch, but must beat the current
vendor composition on real 910C shapes before it is retained.

### C. Graph compiler fusion

Keep both operators unchanged and ask the Ascend graph compiler to fuse the
expression. This may cover more surrounding operations, but changes the model
execution mode and is harder to attribute or upstream as a FlagGems operator.
It is a later model-level experiment, not the first operator implementation.

## Placement and fallback

The public semantic wrapper and composition fallback live in the common fused
namespace. A hardware-specific fast implementation lives under the Ascend
backend fused namespace. Registration remains shallow and selects the backend
implementation through existing FlagGems mechanisms.

Fast-path dispatch uses only stable properties established by crossover
sweeps: backend, dtype, rank, normalized trailing dimensions, contiguity,
affine parameter form, grad mode and alias safety. Any size boundary must be
supported by measurements on both sides. Exact production sizes and model
names are forbidden.

If runtime capability detection, compilation or validation fails, the wrapper
uses the unfused composition. It must not silently return partially computed
output.

## Test and benchmark design

Project evidence keeps exact GraphCast shapes, layouts and call weights. Public
FlagGems tests use moderate model-independent cases that mirror the changed
semantic region:

- FP32, FP16 and BF16 where the backend supports them;
- affine and non-affine LayerNorm;
- one and multiple trailing normalized dimensions;
- power-of-two and tail normalized widths;
- contiguous fast cases and non-contiguous fallback cases;
- different epsilon values, special values and zero-row inputs;
- grad-enabled calls proving fallback correctness.

Every public benchmark case has a matching public correctness case. The
performance comparison is:

```text
Torch/NPU composition | Official FlagGems composition | Candidate fused
```

Timing uses complete operator calls, because the split baselines launch more
than one kernel. Results must include source path/hash audits and report every
configured dtype/layout case.

## Acceptance gates

The implementation may proceed only if the final-source profile proves an
observable E2E contribution and direct adjacency for a stable generalized
region. It is retained only when:

- focused and full operator correctness pass;
- GraphCast dispatch and source audits are valid;
- 40-step output is finite;
- both established GraphCast comparison errors remain below 5%;
- the fused real-case weighted time beats the current composition;
- no public benchmark regression outside the fast region is unexplained;
- three 40-step E2E repeats improve the median beyond run-to-run noise.

The stretch objective is E2E at or below 36.253 s. Failure to meet that target
does not justify model-specific dispatch; remaining time must be attributed
before another operator is selected.

## Deliverables

- final-source profile and leave-one-out report;
- exact project case catalog and scaled public case set;
- fused API, Ascend implementation and composition fallback if the profile
  gate passes;
- mirrored pytest and benchmark coverage;
- Torch/Official/Candidate isolated evidence;
- GraphCast 40-step 7/7 execution audit and 5% diagnostic;
- an explicit Ready/Not Ready upstream assessment.
