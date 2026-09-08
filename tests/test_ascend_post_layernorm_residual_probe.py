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

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

PROBE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "probes"
    / "probe_ascend_post_layernorm_residual.py"
)


def _load_probe_module():
    spec = importlib.util.spec_from_file_location(
        "ascend_post_layernorm_probe", PROBE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _post_layer_norm_residual(x, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(x, normalized_shape, weight, bias, eps) + residual


def _add_then_layer_norm(x, residual, normalized_shape, weight, bias, eps):
    return torch.layer_norm(x + residual, normalized_shape, weight, bias, eps)


def _mutating_post_layer_norm_residual(
    x, residual, normalized_shape, weight, bias, eps
):
    output = _post_layer_norm_residual(x, residual, normalized_shape, weight, bias, eps)
    residual.add_(1)
    return output


def _sparse_post_layer_norm_residual(x, residual, normalized_shape, weight, bias, eps):
    return _post_layer_norm_residual(
        x, residual, normalized_shape, weight, bias, eps
    ).to_sparse()


def _torch_npu_module(available):
    class FakeNpu:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def get_device_name(index):
            assert index == 0
            return "FakeNPU"

        @staticmethod
        def get_soc_version():
            return 1

    return SimpleNamespace(
        __version__="test", __file__="/tmp/torch_npu.py", npu=FakeNpu()
    )


def test_schema_filter_requires_layer_norm_and_add_or_residual():
    probe = _load_probe_module()

    identity = probe.probe_identity()
    assert identity["path"].endswith("probe_ascend_post_layernorm_residual.py")
    assert len(identity["sha256"]) == 64
    assert probe.is_plausible_schema("torch_npu::npu_add_layer_norm")
    assert probe.is_plausible_schema("torch_npu::residual_layer_norm")
    assert not probe.is_plausible_schema("torch_npu::npu_layer_norm")
    assert not probe.is_plausible_schema("torch_npu::npu_add")


def test_probe_records_schema_and_both_order_comparisons_for_post_semantics():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="test::post_layer_norm_residual",
        callable_schema="test::post_layer_norm_residual(Tensor x, Tensor residual)",
        candidate=_post_layer_norm_residual,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (
                case.x,
                case.residual,
                case.normalized_shape,
                case.weight,
                case.bias,
                case.eps,
            ),
            {},
        ),
    )

    assert result["operation_name"] == "test::post_layer_norm_residual"
    assert result["callable_schema"].startswith("test::post_layer_norm_residual")
    assert result["status"] == "accepted"
    assert result["affine_supported"] is True
    assert len(result["cases"]) == 6
    for case in result["cases"]:
        assert case["comparison"]["post"]["matches"] is True
        assert case["comparison"]["pre"]["matches"] is False
        assert case["output"]["dtype"] == case["input"]["dtype"]
        assert case["output"]["shape"] == case["input"]["shape"]
        assert case["output"]["stride"] == case["post_reference"]["stride"]


def test_probe_rejects_add_then_layer_norm_even_when_the_name_is_plausible():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="torch_npu::npu_add_layer_norm",
        callable_schema="torch_npu::npu_add_layer_norm(Tensor x, Tensor residual)",
        candidate=_add_then_layer_norm,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (
                case.x,
                case.residual,
                case.normalized_shape,
                case.weight,
                case.bias,
                case.eps,
            ),
            {},
        ),
    )

    assert result["status"] == "numerical_failure"
    assert result["accepted"] is False
    assert any(case["comparison"]["pre"]["matches"] for case in result["cases"])
    assert all(
        case["semantic_status"] == "numerical_failure" for case in result["cases"]
    )


def test_probe_continues_to_safely_callable_affine_cases_after_nonaffine_mismatch():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="torch_npu::npu_add_layer_norm",
        callable_schema="torch_npu::npu_add_layer_norm(Tensor x, Tensor residual)",
        candidate=_add_then_layer_norm,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (_ for _ in ()).throw(ValueError("affine parameter required"))
            if not case.affine
            else (
                (
                    case.x,
                    case.residual,
                    case.normalized_shape,
                    case.weight,
                    case.bias,
                    case.eps,
                ),
                {},
            )
        ),
    )

    assert result["status"] == "numerical_failure"
    assert len(result["cases"]) == 6
    assert all(
        case["status"] == "executed" for case in result["cases"] if case["affine"]
    )


def test_probe_accepts_exact_affine_primitive_when_nonaffine_schema_binding_is_unsupported():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="torch_npu::affine_post_layer_norm_residual",
        callable_schema="torch_npu::affine_post_layer_norm_residual(Tensor x, Tensor residual)",
        candidate=_post_layer_norm_residual,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (_ for _ in ()).throw(ValueError("affine parameter required"))
            if not case.affine
            else (
                (
                    case.x,
                    case.residual,
                    case.normalized_shape,
                    case.weight,
                    case.bias,
                    case.eps,
                ),
                {},
            )
        ),
    )

    assert result["status"] == "accepted"
    assert result["accepted"] is True
    assert result["affine_supported"] is True


def test_probe_rejects_candidate_that_mutates_a_reference_input():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="test::mutating_post_layer_norm_residual",
        callable_schema="test::mutating_post_layer_norm_residual(Tensor x, Tensor residual)",
        candidate=_mutating_post_layer_norm_residual,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (
                case.x,
                case.residual,
                case.normalized_shape,
                case.weight,
                case.bias,
                case.eps,
            ),
            {},
        ),
    )

    assert result["status"] == "numerical_failure"
    assert all(case["input_integrity"]["residual"] is False for case in result["cases"])
    assert all(
        case["semantic_status"] == "numerical_failure" for case in result["cases"]
    )


def test_probe_records_fail_closed_output_contract_failure_for_unsupported_layout():
    probe = _load_probe_module()

    result = probe.probe_callable(
        operation_name="test::sparse_post_layer_norm_residual",
        callable_schema="test::sparse_post_layer_norm_residual(Tensor x, Tensor residual)",
        candidate=_sparse_post_layer_norm_residual,
        cases=probe.make_cases(device="cpu"),
        binder=lambda case: (
            (
                case.x,
                case.residual,
                case.normalized_shape,
                case.weight,
                case.bias,
                case.eps,
            ),
            {},
        ),
    )

    assert result["status"] == "output_contract_failure"
    assert all(case["status"] == "output_contract_failure" for case in result["cases"])


def test_torch_npu_import_failure_is_inconclusive_and_never_authorizes_triton():
    probe = _load_probe_module()

    def unavailable_torch_npu():
        raise ImportError("torch_npu absent")

    report = probe.run_probe(torch_npu_loader=unavailable_torch_npu)

    assert report["decision"] == "inconclusive"
    assert report["probe_status"] == "error"
    assert report["probe_complete"] is False
    assert report["triton_authorized"] is False
    assert report["reason"] == "torch_npu_unavailable"


def test_npu_unavailable_is_inconclusive_and_never_authorizes_triton():
    probe = _load_probe_module()

    report = probe.run_probe(torch_npu_loader=lambda: _torch_npu_module(False))

    assert report["decision"] == "inconclusive"
    assert report["probe_status"] == "error"
    assert report["probe_complete"] is False
    assert report["triton_authorized"] is False
    assert report["reason"] == "npu_unavailable"


def test_discovery_failure_is_inconclusive_and_never_authorizes_triton(monkeypatch):
    probe = _load_probe_module()

    def unavailable_schemas():
        raise RuntimeError("dispatcher inaccessible")

    monkeypatch.setattr(probe, "_schema_records", unavailable_schemas)
    report = probe.run_probe(torch_npu_loader=lambda: _torch_npu_module(True))

    assert report["decision"] == "inconclusive"
    assert report["reason"] == "schema_discovery_failure"
    assert report["triton_authorized"] is False


def test_cli_writes_partial_inconclusive_artifact_and_returns_nonzero(
    monkeypatch, tmp_path, capsys
):
    probe = _load_probe_module()
    output = tmp_path / "partial.json"
    partial = {
        "decision": "inconclusive",
        "probe_status": "error",
        "probe_complete": False,
        "triton_authorized": False,
        "reason": "torch_npu_unavailable",
    }
    monkeypatch.setattr(probe, "run_probe", lambda *args: partial)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--output",
            str(output),
            "--source-revision",
            "test",
            "--source-worktree",
            "/tmp/source",
        ],
    )

    assert probe.main() == 2
    assert json.loads(output.read_text()) == partial
    assert json.loads(capsys.readouterr().out)["decision"] == "inconclusive"
