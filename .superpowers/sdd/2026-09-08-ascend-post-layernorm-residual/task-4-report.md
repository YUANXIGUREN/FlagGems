# Task 4 report — common post-LayerNorm residual API

## Provenance

- Base: `97bbf5eadc69211eeeb87b7f29efbd568c022a5f`
- API commit: `2131d6a18483b3d85ac29b25e48139a229a0ea59`
- Branch: `yuanxi/ascend_residual_add_layernorm`

## Delivered contract

`flag_gems.post_layer_norm_residual(x, residual, normalized_shape, weight,
bias, eps)` is exported at the package top level and performs exactly:

```python
torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual
```

It is a common Torch composition fallback, not a kernel or an Ascend fast
path. Consequently it keeps PyTorch's validation, dtype promotion and
residual broadcasting behavior. It does not call or reuse `skip_layer_norm`,
whose operation order is `layer_norm(x + residual, ...)`.

## Red → green record

The test file was added before the public function existed. The red command:

```text
pytest -q tests/test_post_layer_norm_residual.py
```

produced 10 expected failures, all due to
`AttributeError: module 'flag_gems' has no attribute
'post_layer_norm_residual'`.

After adding the composition fallback and fused-namespace export, the final
focused command passed:

```text
pytest -q tests/test_post_layer_norm_residual.py tests/test_skip_layer_norm.py
19 passed, 1 warning
```

The warning is the pre-existing CPU Triton replay-benchmarker fallback and is
unrelated to the API semantics. The new CPU-capable tests cover FP32, FP16 and
BF16; affine and non-affine calls; one and multiple trailing normalized
dimensions; residual broadcasting; non-contiguous inputs; empty leading
dimensions; custom epsilon; autograd; invalid normalized/affine/residual
shapes; and an explicit ordering regression.

## Verification and review

- `ruff check` passed for the new source and tests.
- `ruff format --check` reported both files formatted.
- `python -m py_compile` passed for the new source and tests.
- `git diff --check` passed before the API commit.
- Manual contract review confirmed that no GraphCast-specific shape, model
  identifier, device specialization, or hardware-performance claim was added.

## Reuse assessment

I reviewed the verified PR #4930 overlay at
`/mdata/data/zhangyuxin/wxt/flagos-common-zyx-flaggems-op-prcheck/patches/flaggems/pr-4930-post-layer-norm-residual`
and its commit chain ending `d7ebcc3828c3fedae522d07e8fc4c44998468a63`.

Reused: the public function name, top-level fused-namespace export pattern,
and the test-coverage themes. Not reused: its custom metadata validation,
Triton kernels, custom autograd, or backend implementation. Those would alter
the required standard Torch composition behavior here, especially residual
broadcasting and native validation errors, and are outside Task 4.
