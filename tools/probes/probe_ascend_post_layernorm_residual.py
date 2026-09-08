#!/usr/bin/env python3
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

"""Fail-closed probe for a post-LayerNorm residual Torch-NPU primitive.

The probe only executes callables backed by an installed dispatcher schema and
only when every required parameter has an unambiguous binding.  An operation
name is discovery evidence, never semantic evidence: each executable candidate
is compared with both LayerNorm-then-add and add-then-LayerNorm references.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

EPS = 1e-5
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
X_NAMES = {"self", "input", "input_x", "x", "x1"}
RESIDUAL_NAMES = {"residual", "skip", "skip_input", "input_residual", "x2"}
WEIGHT_NAMES = {"weight", "gamma", "scale"}
BIAS_NAMES = {"bias", "beta"}
SHAPE_NAMES = {"normalized_shape", "norm_shape", "normalized_dims"}
EPS_NAMES = {"eps", "epsilon"}


def is_plausible_schema(name: str) -> bool:
    """Return whether a registered operation name merits semantic probing."""
    lowered = name.lower()
    return "layer_norm" in lowered and ("add" in lowered or "residual" in lowered)


def probe_identity() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": _dtype_name(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
    }


def make_cases(device: str | torch.device) -> list[SimpleNamespace]:
    """Build deterministic, order-sensitive FP32/FP16/BF16 probe inputs."""
    device = torch.device(device)
    shape = (2, 3, 4)
    values = torch.arange(24, device=device, dtype=torch.float32).reshape(shape)
    x_base = values.div(7.0).sub(1.25)
    residual_base = values.remainder(5).sub(2.0).mul(0.75)
    weight_base = torch.linspace(0.5, 1.25, 4, device=device, dtype=torch.float32)
    bias_base = torch.linspace(-0.4, 0.3, 4, device=device, dtype=torch.float32)
    cases = []
    for dtype in DTYPES:
        for affine in (False, True):
            weight = weight_base.to(dtype) if affine else None
            bias = bias_base.to(dtype) if affine else None
            cases.append(
                SimpleNamespace(
                    name=f"{_dtype_name(dtype)}-{'affine' if affine else 'nonaffine'}",
                    x=x_base.to(dtype),
                    residual=residual_base.to(dtype),
                    normalized_shape=(shape[-1],),
                    weight=weight,
                    bias=bias,
                    eps=EPS,
                    affine=affine,
                )
            )
    return cases


def _references(case: SimpleNamespace) -> tuple[torch.Tensor, torch.Tensor]:
    post = (
        torch.layer_norm(
            case.x, case.normalized_shape, case.weight, case.bias, case.eps
        )
        + case.residual
    )
    pre = torch.layer_norm(
        case.x + case.residual,
        case.normalized_shape,
        case.weight,
        case.bias,
        case.eps,
    )
    return post, pre


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().to("cpu", dtype=torch.float32)
    expected_cpu = expected.detach().to("cpu", dtype=torch.float32)
    difference = (actual_cpu - expected_cpu).abs()
    if actual.dtype == torch.float32:
        tolerance = 1e-4
    elif actual.dtype == torch.bfloat16:
        tolerance = 2e-2
    else:
        tolerance = 5e-3
    return {
        "matches": bool(
            torch.allclose(actual_cpu, expected_cpu, rtol=tolerance, atol=tolerance)
        ),
        "rtol": tolerance,
        "atol": tolerance,
        "max_abs_error": float(difference.max().item()),
    }


def _select_tensor_output(result: Any) -> tuple[torch.Tensor, int | None]:
    if isinstance(result, torch.Tensor):
        return result, None
    if isinstance(result, (tuple, list)):
        for index, value in enumerate(result):
            if isinstance(value, torch.Tensor):
                return value, index
    raise TypeError(f"candidate returned no Tensor output ({type(result).__name__})")


def _requirements_match(
    actual: torch.Tensor, post: torch.Tensor
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "dtype": actual.dtype == post.dtype,
        "device": actual.device == post.device,
        "layout": actual.layout == post.layout,
        "shape": tuple(actual.shape) == tuple(post.shape),
        "stride": tuple(actual.stride()) == tuple(post.stride()),
        "contiguous": actual.is_contiguous() == post.is_contiguous(),
    }
    return all(checks.values()), checks


def _reference_case(case: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        x=case.x.detach().clone(),
        residual=case.residual.detach().clone(),
        normalized_shape=case.normalized_shape,
        weight=case.weight.detach().clone() if case.weight is not None else None,
        bias=case.bias.detach().clone() if case.bias is not None else None,
        eps=case.eps,
    )


def _input_integrity(
    case: SimpleNamespace, reference: SimpleNamespace
) -> dict[str, bool]:
    def unchanged(actual: torch.Tensor | None, expected: torch.Tensor | None) -> bool:
        if actual is None or expected is None:
            return actual is expected
        try:
            return bool(
                actual.dtype == expected.dtype
                and actual.device == expected.device
                and actual.layout == expected.layout
                and tuple(actual.shape) == tuple(expected.shape)
                and tuple(actual.stride()) == tuple(expected.stride())
                and torch.equal(actual, expected)
            )
        except Exception:
            return False

    return {
        "x": unchanged(case.x, reference.x),
        "residual": unchanged(case.residual, reference.residual),
        "weight": unchanged(case.weight, reference.weight),
        "bias": unchanged(case.bias, reference.bias),
    }


def _aliases_input(output: torch.Tensor, case: SimpleNamespace) -> dict[str, bool]:
    aliases = {}
    for name in ("x", "residual", "weight", "bias"):
        tensor = getattr(case, name)
        aliases[name] = bool(
            tensor is not None and torch._C._is_alias_of(output, tensor)
        )
    return aliases


def probe_callable(
    *,
    operation_name: str,
    callable_schema: str,
    candidate: Callable[..., Any],
    cases: list[SimpleNamespace],
    binder: Callable[[SimpleNamespace], tuple[tuple[Any, ...], dict[str, Any]]],
) -> dict[str, Any]:
    """Execute one safely bound candidate against both operation-order oracles."""
    record: dict[str, Any] = {
        "operation_name": operation_name,
        "callable_schema": callable_schema,
        "status": "not_callable",
        "accepted": False,
        "affine_supported": False,
        "cases": [],
    }
    affine_cases = sum(case.affine for case in cases)
    affine_ok = True
    affine_executed = 0
    saw_affine_signature_mismatch = False
    saw_affine_runtime_failure = False
    saw_affine_output_contract_failure = False
    saw_executed_case = False
    saw_numerical_failure = False
    for case in cases:
        case_record: dict[str, Any] = {
            "name": case.name,
            "affine": case.affine,
            "input": _tensor_description(case.x),
        }
        reference = _reference_case(case)
        post, pre = _references(reference)
        try:
            args, kwargs = binder(case)
        except ValueError as error:
            case_record["status"] = "signature_mismatch"
            case_record["error"] = str(error)
            record["cases"].append(case_record)
            saw_affine_signature_mismatch |= case.affine
            continue
        try:
            output, output_index = _select_tensor_output(candidate(*args, **kwargs))
        except (
            Exception
        ) as error:  # Runtime errors are evidence, not a fallback signal.
            case_record["status"] = "runtime_failure"
            case_record["error_type"] = type(error).__name__
            case_record["error"] = str(error)
            record["cases"].append(case_record)
            saw_affine_runtime_failure |= case.affine
            continue

        try:
            input_integrity = _input_integrity(case, reference)
            aliases_input = _aliases_input(output, case)
            requirements_match, requirements = _requirements_match(output, post)
            post_comparison = _comparison(output, post)
            pre_comparison = _comparison(output, pre)
            semantic_match = (
                post_comparison["matches"]
                and not pre_comparison["matches"]
                and requirements_match
                and all(input_integrity.values())
                and not any(aliases_input.values())
            )
            case_record.update(
                {
                    "status": "executed",
                    "semantic_status": (
                        "accepted" if semantic_match else "numerical_failure"
                    ),
                    "input_integrity": input_integrity,
                    "aliases_input": aliases_input,
                    "output": _tensor_description(output),
                    "selected_output_index": output_index,
                    "post_reference": _tensor_description(post),
                    "comparison": {"post": post_comparison, "pre": pre_comparison},
                    "requirements": requirements,
                }
            )
        except Exception as error:
            case_record.update(
                {
                    "status": "output_contract_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            record["cases"].append(case_record)
            saw_affine_output_contract_failure |= case.affine
            continue

        record["cases"].append(case_record)
        saw_executed_case = True
        saw_numerical_failure |= not semantic_match
        if case.affine:
            affine_executed += 1
            affine_ok &= semantic_match

    record["affine_supported"] = affine_executed == affine_cases and affine_ok
    record["semantic_status"] = (
        "accepted"
        if record["affine_supported"]
        else "numerical_failure" if saw_executed_case else "not_evaluated"
    )
    if record["affine_supported"]:
        record["status"] = "accepted"
        record["accepted"] = True
    elif saw_affine_output_contract_failure:
        record["status"] = "output_contract_failure"
    elif saw_affine_runtime_failure:
        record["status"] = "runtime_failure"
    elif saw_affine_signature_mismatch:
        record["status"] = "signature_mismatch"
    elif saw_numerical_failure:
        record["status"] = "numerical_failure"
    return record


def _argument_has_default(argument: Any) -> bool:
    has_default = getattr(argument, "has_default_value", None)
    return bool(has_default() if callable(has_default) else has_default)


def _argument_type(argument: Any) -> str:
    return str(getattr(argument, "type", ""))


def _bind_schema(
    schema: Any, case: SimpleNamespace
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Bind only explicit, stable Torch-NPU argument conventions.

    Unknown required parameters intentionally make the candidate ineligible.
    This avoids discovering a name and then guessing its calling convention.
    """
    args: list[Any] = []
    for argument in schema.arguments:
        name = argument.name.lower()
        type_name = _argument_type(argument)
        value: Any
        if name in X_NAMES:
            value = case.x
        elif name in RESIDUAL_NAMES:
            value = case.residual
        elif name in WEIGHT_NAMES:
            if case.weight is None and "?" not in type_name:
                raise ValueError(f"non-optional affine argument {argument.name!r}")
            value = case.weight
        elif name in BIAS_NAMES:
            if case.bias is None and "?" not in type_name:
                raise ValueError(f"non-optional affine argument {argument.name!r}")
            value = case.bias
        elif name in SHAPE_NAMES:
            value = list(case.normalized_shape)
        elif name in EPS_NAMES:
            value = case.eps
        elif _argument_has_default(argument):
            continue
        else:
            raise ValueError(
                f"cannot safely bind required argument {argument.name!r} ({type_name})"
            )
        args.append(value)
    return tuple(args), {}


def _schema_operation_name(schema: Any) -> str:
    overload = getattr(schema, "overload_name", "")
    return f"{schema.name}.{overload}" if overload else schema.name


def _resolve_callable(schema: Any) -> Callable[..., Any]:
    namespace, operator = schema.name.split("::", maxsplit=1)
    packet = getattr(getattr(torch.ops, namespace), operator)
    overload = getattr(schema, "overload_name", "")
    if overload:
        return getattr(packet, overload)
    return getattr(packet, "default", packet)


def _schema_records() -> list[Any]:
    schemas = torch._C._jit_get_all_schemas()
    return sorted(
        (
            schema
            for schema in schemas
            if is_plausible_schema(_schema_operation_name(schema))
        ),
        key=lambda schema: (schema.name, getattr(schema, "overload_name", "")),
    )


def _public_api_records(module: Any, module_name: str) -> list[dict[str, Any]]:
    records = []
    for name in sorted(dir(module)):
        if name.startswith("_") or not is_plausible_schema(name):
            continue
        value = getattr(module, name)
        if not callable(value):
            continue
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = None
        documentation = inspect.getdoc(value) or ""
        records.append(
            {
                "name": f"{module_name}.{name}",
                "signature": signature,
                "documentation": (
                    documentation.splitlines()[0] if documentation else None
                ),
            }
        )
    return records


def _runtime_identity(torch_npu: Any | None) -> dict[str, Any]:
    npu = getattr(torch_npu, "npu", None)
    identity: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_npu_version": getattr(torch_npu, "__version__", None),
        "torch_npu_file": getattr(torch_npu, "__file__", None),
        "npu_available": bool(npu and npu.is_available()),
    }
    if identity["npu_available"]:
        identity["device"] = "npu:0"
        identity["device_name"] = npu.get_device_name(0)
        get_soc_version = getattr(npu, "get_soc_version", None)
        identity["soc_version"] = get_soc_version() if get_soc_version else None
    return identity


def _report_identity(
    source_revision: str | None, source_worktree: str | None
) -> dict[str, Any]:
    return {
        "probe": probe_identity(),
        "source": {
            "flag_gems_revision": source_revision,
            "flag_gems_worktree": source_worktree,
        },
    }


def _inconclusive_report(
    *,
    source_revision: str | None,
    source_worktree: str | None,
    reason: str,
    runtime: dict[str, Any],
    error: Exception | None = None,
    public_torch_npu_apis: list[dict[str, Any]] | None = None,
    public_torch_npu_npu_apis: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "probe_version": 1,
        **_report_identity(source_revision, source_worktree),
        "decision": "inconclusive",
        "probe_status": "error",
        "probe_complete": False,
        "triton_authorized": False,
        "reason": reason,
        "runtime": runtime,
        "schema_candidates": [],
        "public_torch_npu_apis": public_torch_npu_apis or [],
        "public_torch_npu_npu_apis": public_torch_npu_npu_apis or [],
    }
    if error is not None:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    return report


def _load_torch_npu() -> Any:
    import torch_npu

    return torch_npu


def run_probe(
    device: str = "npu:0",
    source_revision: str | None = None,
    source_worktree: str | None = None,
    torch_npu_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Inspect installed Torch-NPU schemas and execute only safely bound ones."""
    try:
        torch_npu = (torch_npu_loader or _load_torch_npu)()
    except Exception as error:
        return _inconclusive_report(
            source_revision=source_revision,
            source_worktree=source_worktree,
            reason="torch_npu_unavailable",
            runtime=_runtime_identity(None),
            error=error,
        )

    try:
        runtime = _runtime_identity(torch_npu)
        public_torch_npu_apis = _public_api_records(torch_npu, "torch_npu")
        public_torch_npu_npu_apis = _public_api_records(torch_npu.npu, "torch_npu.npu")
    except Exception as error:
        return _inconclusive_report(
            source_revision=source_revision,
            source_worktree=source_worktree,
            reason="runtime_or_public_api_inspection_failure",
            runtime=_runtime_identity(None),
            error=error,
        )
    if not runtime["npu_available"]:
        return _inconclusive_report(
            source_revision=source_revision,
            source_worktree=source_worktree,
            reason="npu_unavailable",
            runtime=runtime,
            public_torch_npu_apis=public_torch_npu_apis,
            public_torch_npu_npu_apis=public_torch_npu_npu_apis,
        )

    try:
        schemas = _schema_records()
        cases = make_cases(device)
    except Exception as error:
        return _inconclusive_report(
            source_revision=source_revision,
            source_worktree=source_worktree,
            reason="schema_discovery_failure",
            runtime=runtime,
            error=error,
            public_torch_npu_apis=public_torch_npu_apis,
            public_torch_npu_npu_apis=public_torch_npu_npu_apis,
        )
    records = []
    for schema in schemas:
        operation_name = _schema_operation_name(schema)
        callable_schema = str(schema)
        try:
            candidate = _resolve_callable(schema)
        except Exception as error:
            records.append(
                {
                    "operation_name": operation_name,
                    "callable_schema": callable_schema,
                    "status": "not_callable",
                    "accepted": False,
                    "affine_supported": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cases": [],
                }
            )
            continue
        records.append(
            probe_callable(
                operation_name=operation_name,
                callable_schema=callable_schema,
                candidate=candidate,
                cases=cases,
                binder=lambda case, schema=schema: _bind_schema(schema, case),
            )
        )

    accepted = [record for record in records if record["accepted"]]
    return {
        "probe_version": 1,
        **_report_identity(source_revision, source_worktree),
        "decision": "vendor_primitive" if accepted else "triton_task_6",
        "probe_status": "complete",
        "probe_complete": True,
        "triton_authorized": not accepted,
        "reason": (
            "validated_post_semantic_primitive"
            if accepted
            else "no_validated_post_semantic_primitive"
        ),
        "runtime": runtime,
        "schema_candidates": records,
        "public_torch_npu_apis": public_torch_npu_apis,
        "public_torch_npu_npu_apis": public_torch_npu_npu_apis,
    }


def _write_report(report: dict[str, Any], output: Path) -> str:
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-worktree", required=True)
    args = parser.parse_args()
    report = run_probe(args.device, args.source_revision, args.source_worktree)
    artifact_sha256 = _write_report(report, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["decision"],
                "sha256": artifact_sha256,
            }
        )
    )
    return 0 if report["probe_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
