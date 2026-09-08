# Ascend post-LayerNorm residual: final acceptance

Decision on 2026-09-09: **Not Ready**. First failing gate:
`performance_target`. The final fused median is **39.48483373969793 s**
(**14.690202441497288% MFU**), missing 36.253 s / 16%. Correctness, source
identity, dispatch, performance eligibility and the final regression pass.
A measurable gain over current FlagGems composition is not Native parity.

## Scope and measurement contract

The expression is `layer_norm(x, normalized_shape, weight, bias, eps) + residual`,
not LayerNorm of the sum. Task 9 ran exactly three independent fresh processes
per mode on Ascend 910C, container `zyx-graphcast-910c`, device `npu:0`
(`Ascend910_9382`). Each process used three excluded warmup calls and 40 measured
dependent six-hour autoregressive calls. Runs were sequential in repeat order:
Native, FlagGems unfused, FlagGems fused.

All modes used accepted readonly assets/references, `domestic-v1`, FP32 tensors,
no autocast, `--precision tf32`, effective fast-FP32/HF32 with both NPU matmul
and convolution HF32 controls true. Python 3.11.14, Torch 2.8.0+cpu and
Torch-NPU 2.8.0.post2 match the accepted environment. There was no event,
candidate-profile or trajectory instrumentation; all nine reports are
performance-eligible. As in Task 3, timing includes final finite checking,
final D2H and completion synchronization; file publication and offline
diagnostics are outside the timer.

## All raw results and decision arithmetic

| Mode | Repeat 1 seconds | Repeat 2 seconds | Repeat 3 seconds | Median seconds | Full range seconds | MFU percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 34.971331420354545 | 34.955021489877254 | 34.96432165009901 | 34.96432165009901 | 0.016309930477291346 | 16.589488187693256 |
| Final FlagGems 7/7 unfused | 40.130463710054755 | 40.18352977000177 | 40.177274259738624 | 40.177274259738624 | 0.053066059947013855 | 14.437022214478084 |
| Final FlagGems 7/7 fused | 39.48483373969793 | 39.456916869618 | 39.49070909013972 | 39.48483373969793 | 0.033792220521718264 | 14.690202441497288 |

MFU percent is `100 * (28.857 * 40) / (199 * median_seconds)`.
The runner's embedded MFU uses its more precise internal model count,
28.857094475776 TFLOP/step; this table intentionally uses the specified 28.857.

- Fused saves 0.6924405200406909 s versus unfused (1.7234631587105471%),
  a 1.0175368731347731x speedup.
- The saving exceeds the conservative sum of the unfused/fused full ranges,
  0.08685828046873212 s. This is an observed-repeat noise check, not a confidence interval.
- Native/fused speedup is 0.8855126978778686x: fused is 4.520512089598924 s slower.
- Fused is 3.231833739697933 s above the 36.253 s target and
  1.3097975585027122 percentage points below 16% MFU.

## Correctness and dispatch

All nine runner and offline diagnostic processes returned zero. All outputs
were finite; the established diagnostic compared every output against both
qualified JAX and PyTorch GPU references. Every overall and Z500 error is
below the strict 5% threshold; the raw percentages are preserved below.

| Run | JAX overall % | JAX Z500 % | PyTorch GPU overall % | PyTorch GPU Z500 % |
| --- | ---: | ---: | ---: | ---: |
| native-r1 | 1.0517017188776356 | 0.013989157740835336 | 1.0516735424635837 | 0.01398721435450833 |
| native-r2 | 1.0653063593228136 | 0.014542400774691855 | 1.0652624064719902 | 0.014539705043003066 |
| native-r3 | 1.0766585873286254 | 0.014758725719427353 | 1.0766202860781355 | 0.014755987174386797 |
| unfused-r1 | 1.039134365229703 | 0.013838942953220426 | 1.0391052615858658 | 0.013836970496199679 |
| unfused-r2 | 1.0317478673821374 | 0.013856502624998879 | 1.0317010446310795 | 0.013854096092107344 |
| unfused-r3 | 0.9959528553419372 | 0.013495632238765189 | 0.9959481173192014 | 0.013493652464180304 |
| fused-r1 | 1.0802615330469285 | 0.014902648618043513 | 1.080146936966273 | 0.014899834299633038 |
| fused-r2 | 1.0748392818218353 | 0.014564742371373814 | 1.0748135233701548 | 0.01456325338419475 |
| fused-r3 | 1.0800982307922762 | 0.014869503616002587 | 1.0799913547805082 | 0.014866563224617192 |

Every FlagGems run requested and observed exactly
`addmm,cat,native_layer_norm,index_add_,silu,index,add` (7/7).
Native had FlagOS off and 0/7 dispatch. Native's installed FlagGems version/path
metadata is not evidence of FlagGems execution.

Every fused repeat recorded fusion requested/effective true, the selected
candidate revision/root, and exactly 860 public API returns: 43 calls including
warmups times 20 node seams. Stage totals were grid2mesh 86, mesh 688, mesh2grid 86.
Counters were unsynchronized and scoped `warmup_and_measured_rollout`.
Native and unfused had zero fused calls. These are **not kernel launch counts**
and include possible fallback. Task 8's separate untimed smoke establishes
actual kernel dispatch: six cases, 12 launches, zero fallbacks, from the identical
candidate/kernel hash. No smoke instrumentation contaminated E2E timings.

## Frozen source and asset identity

Executed sources (later documentation-only commits are not the execution pins):

| Source | Commit | Complete tracked-file SHA-256 |
| --- | --- | --- |
| GraphCast | `d093327ad3a58d92eb28c8a31bd7df615e8187c8` | `de3a9bc51589b56b1712f2a1276d9c8be94ed16cae5f7c5c3a9b0ebaa77b7736` |
| FlagGems | `406fd35a6575fc4ac1feee17ab54658be1d856e0` | `d9956605906b27718e0df7aa36d4278eef7b9558c1cc37821b6ba3edd002a735` |

Remote campaign root:
`/workspace/FlagOS/campaigns/graphcast-post-layernorm-task9-20260909`.
Complete source checkouts are `graphcast-source` and `flaggems-source`;
the accepted replacement regression uses separate `regression-v2-source`
at the same FlagGems commit and complete tree digest. Local source archives
and complete per-file manifests agree with remote manifests. Sources and all
accepted asset/reference hashes were checked before each run and through its
offline diagnosis, not merely before package import.

The loaded fused module is `_ascend.fused.post_layernorm_residual`, file
`flaggems-source/src/flag_gems/runtime/backend/_ascend/fused/post_layernorm_residual.py`
under the campaign root, SHA-256
`94a22afd923333d2d03e3171eaf13317afd8843c1fbce7d196975ea67145a123`.
Runner SHA-256:
`f586c940cd5ada76249261dfc872985abb2dfcf4c4253c6c8ee2053cfb5edfd3`.
Diagnostic implementation SHA-256:
`d04fd5a90ee6eebff477d0c98417ebd94c436b376b187998f8e2f239e38e10e6`.

Accepted assets remain readonly under
`/workspace/FlagOS/campaigns/graphcast-five-platform-tf32-20260906/case/assets`;
qualified references remain readonly under
`/workspace/FlagOS/campaigns/graphcast-fixed-baseline-20260906/baseline`.
Config SHA-256:
`7ffbc9cc000dad8576d1c8e4be6a66a51897c8ebc20239e9ceb49448ceeeab5d`;
weights SHA-256:
`2bd1786e197b432d0f471ce4118f34b231c73c322dbaba0684e5f71097e010b4`.
The exact hashes of inputs, forcing, target template, all normalization files
and qualified reference data/metadata are in immutable `results/campaign.json`
and each run's source audit. No accepted data was moved and no installed package,
production kernel or inference source was edited.

## Final regression and explicit supersession

The final exact-candidate regression passed **511 tests, zero skips, one pytest
warning in 918.00 s**, pytest and wrapper rc 0. It ran all five required files
with absolute paths, `-q -rs`, from an evidence working directory:

```text
tests/test_post_layer_norm_residual.py
tests/test_skip_layer_norm.py
tests/test_layer_norm.py
tests/test_addmm.py
tests/test_addmm_precision_policy.py
```

`results/regression-v2/source-audit.json` binds the full invocation, cwd,
all test-file hashes, loaded package/public-symbol paths and hashes, source
commit/tree before and after tests, driver hash and exit code.
`accuracy_result.json` retains all 511 passing case records.

The first regression also passed 511 tests (801.62 s), but its wrapper failed
the live-process clean-source audit (rc 1): CANN temporarily created
`kernel_meta/buildPidInfo.json` in its source cwd, then removed it at exit.
This was not accepted as a passing regression. A 28-test RED probe reproduced
the dirty source and failed audit (rc 1); the same 28 tests with an evidence cwd
and absolute test paths passed the live-process audit (GREEN rc 0). The full
five-file family was then restarted with a fresh complete source checkout and
fresh caches. This external working-directory configuration fix did not require
a repository implementation change. The original logs/rc and
`results/acceptance.json` remain immutable and are superseded **only for the
regression family** by `regression-v2` and `acceptance-final.json`.
All nine independently audited E2E comparisons remain valid and were not rerun.

Warnings are retained: three startup warnings about the replay-benchmarker
fallback, unimplemented NPU device-capability query and its fallback; one pytest
warning about base-format tensor allocation. No skips were used to waive tests.
The regression's benchmarker warning is not evidence of event instrumentation
in the separate performance-eligible E2E runs.

## Evidence, commands and limitations

Local evidence:
`/mdata/data/zhangyuxin/wxt/ascend-task9-evidence`.
Remote results and predictions:
`/workspace/FlagOS/campaigns/graphcast-post-layernorm-task9-20260909/results`,
container `zyx-graphcast-910c`.
All remote access used `flagos_remote.sh` targeting `910c`; work ran in tmux:

- `gc910c-task9-20260909`: nine E2E runs and original regression.
- `gc910c-task9-audit-probe`: RED reproduction.
- `gc910c-task9-audit-green`: GREEN check.
- `gc910c-task9-regression-v2`: accepted full regression.

Retained `setup.sh`, `run.sh`, `acceptance.py`, `run-regression-v2.sh`,
`regression-v2.py` and `finalize-v2.py` define the full reproducible commands.
Every run directory retains command, report, source audit, completed-run
binding, runner/diagnostic rc, diagnostic summary/details and dispatch log;
`results/logs` retains process logs and shell rc. The final decision contains
raw artifact hashes. Predictions and compiler caches remain remote and are
excluded from the small-result archive.

| Artifact | SHA-256 |
| --- | --- |
| Final small-result archive | `638fae2f59caa6a9ac6e218edebc94eae33a1ba98212253cba8e4ade082652d5` |
| `results/acceptance-final.json` | `856765df7b11a7a74afb2bedbebda34c8b3fb32468ac83c75f9e3dbea8b1ca51` |
| Accepted `regression-v2/pytest.log` | `24e2e43f8abdc3f341d42777cb843f7057c6497e65c7c819bef1eb346ee78ff7` |
| Complete GraphCast source archive | `fc853e99834635b70f0ef0cf2316f835f911f992df4b9df63421876969a497e5` |
| Complete FlagGems source archive | `b567edd613ad536b6a84cc53ce7b803fd7cc49d89c8fa24658a9f6d07af0e581` |
| Task 8 untimed smoke JSON | `441865a037bfdb9bb89649277dd91abc189aa4435d673d3e12b3ec36af387470` |

Local `verify_download.py` independently passed source-manifest matching,
downloaded artifact/hash bindings, all nine diagnostic/dispatch/precision
checks, regression evidence and arithmetic. Tar extraction noted remote-clock
future timestamps; file hashes matched and monotonic durations are unaffected.

Task 7 found a real-shape weighted 2.139x improvement over current composition,
but 22/24 public cases regressed by over 5% against Native. Small-call host
overhead and some device-time regressions remain explicit limitations.
This final E2E run likewise does not establish Native acceleration or meet the
deployment target. No push, PR publication or installed-package rollout was done.
