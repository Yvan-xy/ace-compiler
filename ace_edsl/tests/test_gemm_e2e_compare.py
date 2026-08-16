"""Unit tests for the local-fixture GEMM E2E harness."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from ace_edsl.tests.gemm_e2e_compare import (
    FAST_PAIR_IMPLEMENTATIONS,
    FOUR_WAY_IMPLEMENTATIONS,
    IMPLEMENTATIONS,
    MODEL_SLOTS,
    THREE_WAY_IMPLEMENTATIONS,
    _balanced_order,
    _compare_output_set,
    _compare_outputs,
    _parse_arguments,
    _parse_run,
    _path_evidence,
    _summarize_timings,
    _validate_model_files,
)


def _write_values(path: Path, values: list[float]) -> Path:
    path.write_text("".join(f"{value!r}\n" for value in values))
    return path


def test_compare_outputs_reports_maximum_difference(tmp_path):
    native = _write_values(tmp_path / "native.txt", [1.0, -2.0, 3.0])
    dsl = _write_values(tmp_path / "dsl.txt", [1.0, -2.0001, 3.0])

    result = _compare_outputs(native, dsl, tolerance=0.0002)

    assert result == {
        "length": 3,
        "max_abs_difference": pytest.approx(0.0001),
        "max_abs_difference_index": 1,
        "tolerance": 0.0002,
    }


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_compare_outputs_rejects_non_finite_values(tmp_path, bad_value):
    native = _write_values(tmp_path / "native.txt", [1.0, bad_value])
    dsl = _write_values(tmp_path / "dsl.txt", [1.0, 2.0])

    with pytest.raises(RuntimeError, match="non-finite at index 1"):
        _compare_outputs(native, dsl, tolerance=0.0002)


def test_compare_outputs_rejects_empty_files(tmp_path):
    native = _write_values(tmp_path / "native.txt", [])
    dsl = _write_values(tmp_path / "dsl.txt", [])

    with pytest.raises(RuntimeError, match="output files are empty"):
        _compare_outputs(native, dsl, tolerance=0.0002)


def test_compare_outputs_rejects_length_mismatch(tmp_path):
    native = _write_values(tmp_path / "native.txt", [1.0])
    dsl = _write_values(tmp_path / "dsl.txt", [1.0, 2.0])

    with pytest.raises(RuntimeError, match="output length mismatch"):
        _compare_outputs(native, dsl, tolerance=0.0002)


def test_compare_outputs_rejects_difference_above_tolerance(tmp_path):
    native = _write_values(tmp_path / "native.txt", [1.0])
    dsl = _write_values(tmp_path / "dsl.txt", [1.1])

    with pytest.raises(RuntimeError, match="exceeds 0.01"):
        _compare_outputs(native, dsl, tolerance=0.01)


@pytest.mark.parametrize("tolerance", [math.nan, math.inf, -1.0])
def test_compare_outputs_rejects_invalid_tolerance(tmp_path, tolerance):
    native = _write_values(tmp_path / "native.txt", [1.0])
    dsl = _write_values(tmp_path / "dsl.txt", [1.0])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        _compare_outputs(native, dsl, tolerance=tolerance)


def test_parse_run_extracts_main_graph_and_validation():
    output = """
RTLib functions                    Count       Elapse
MAIN_GRAPH                             1     0.123456 sec
GEMM_RESULT=PASS max_abs=1e-05 max_abs_index=7 max_rel=0.0002
"""
    completed = subprocess.CompletedProcess(["gemm"], 0, output, "")

    timing, validation = _parse_run(completed, model="i64_o10", implementation="dsl")

    assert timing == 0.123456
    assert validation == {
        "max_abs_error": 1e-05,
        "max_abs_error_index": 7,
        "max_rel_error": 0.0002,
    }


@pytest.mark.parametrize(
    ("timing", "max_abs", "max_rel", "message"),
    [
        ("0.0", "0", "0", "invalid MAIN_GRAPH timing"),
        ("0.1", "inf", "0", "invalid max_abs_error"),
        ("0.1", "0", "nan", "invalid max_rel_error"),
        ("0.1", "-1", "0", "invalid max_abs_error"),
    ],
)
def test_parse_run_rejects_invalid_numeric_results(timing, max_abs, max_rel, message):
    output = (
        f"MAIN_GRAPH 1 {timing} sec\n"
        "GEMM_RESULT=PASS "
        f"max_abs={max_abs} max_abs_index=0 max_rel={max_rel}\n"
    )
    completed = subprocess.CompletedProcess(["gemm"], 0, output, "")

    with pytest.raises(RuntimeError, match=message):
        _parse_run(completed, model="i64_o10", implementation="dsl")


def test_validate_model_files_reports_missing_fixture_pairs(tmp_path):
    with pytest.raises(
        RuntimeError, match="missing local GEMM model fixtures"
    ) as error:
        _validate_model_files(tmp_path, ["i64_o10"])

    message = str(error.value)
    assert str(tmp_path / "i64_o10.onnx") in message
    assert str(tmp_path / "i64_o10.c") in message
    assert "--model-dir" in message

    (tmp_path / "i64_o10.onnx").write_bytes(b"onnx")
    (tmp_path / "i64_o10.c").write_text("driver")
    _validate_model_files(tmp_path, ["i64_o10"])


def test_compare_output_set_cross_checks_every_pair(tmp_path):
    paths = {
        "dsl-fast": _write_values(tmp_path / "dsl.txt", [1.0, 2.0]),
        "cpp-baseline": _write_values(tmp_path / "baseline.txt", [1.0, 2.0001]),
        "metakernel-fast": _write_values(tmp_path / "fast.txt", [1.0, 2.00005]),
    }

    result = _compare_output_set(paths, tolerance=0.0002)

    assert set(result["pairs"]) == {
        "dsl-fast__cpp-baseline",
        "dsl-fast__metakernel-fast",
        "cpp-baseline__metakernel-fast",
    }
    assert result["max_abs_difference"] == pytest.approx(0.0001)
    assert result["max_abs_difference_pair"] == "dsl-fast__cpp-baseline"


def test_balanced_order_rotates_and_reverses_complete_cycles():
    paths = ("dsl-fast", "cpp-baseline", "metakernel-fast")

    assert _balanced_order(paths, 0) == paths
    assert _balanced_order(paths, 1) == ("cpp-baseline", "metakernel-fast", "dsl-fast")
    assert _balanced_order(paths, 2) == ("metakernel-fast", "dsl-fast", "cpp-baseline")
    assert _balanced_order(paths, 3) == tuple(reversed(paths))
    assert IMPLEMENTATIONS == ("native", "dsl")


def test_three_path_timing_summary_reports_pairwise_gaps():
    summary = _summarize_timings(
        {
            "dsl-fast": [2.0, 2.2, 1.8],
            "cpp-baseline": [4.0, 4.2, 3.8],
            "metakernel-fast": [1.0, 1.2, 0.8],
        }
    )

    dsl_over_baseline = summary["comparisons"]["dsl-fast_over_cpp-baseline"]
    dsl_over_fast = summary["comparisons"]["dsl-fast_over_metakernel-fast"]
    assert dsl_over_baseline["runtime_ratio"] == pytest.approx(0.5)
    assert dsl_over_baseline["gap_percent"] == pytest.approx(-50.0)
    assert dsl_over_baseline["right_speedup_vs_left"] == pytest.approx(0.5)
    assert dsl_over_fast["runtime_ratio"] == pytest.approx(2.0)
    assert dsl_over_fast["gap_percent"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("implementation", "provenance"),
    [("dsl-fast", "cpp"), ("python-dsl-fast", "python")],
)
def test_dsl_fast_path_evidence_requires_actual_fast_plan_and_inlining(
    implementation, provenance
):
    helper_name = "__ace_vkernel_fast_gemm_" + "a" * 64
    tensor_ir = (
        f'FUN[1] "{helper_name}"\n'
        f'  call "{helper_name}"\n'
        "IMRA Metakernel: gs=2\n"
        "gemm result reduce->Ps=1\n"
        "gemm result reduce->(kp/np)=1\n"
        "ATTR[nums=(0,2)]\n"
    )
    prepared = SimpleNamespace(
        kind="fast-gemm",
        provenance=provenance,
        specialization_key="fast-gemm:test",
        helper_name=helper_name,
        constants=(SimpleNamespace(role="weight", content_hash="sha256:x"),),
        rotations=(SimpleNamespace(role="grid", candidates=(0, 2)),),
        result_type=SimpleNamespace(element_type="f32", shape=(8,)),
        slot=SimpleNamespace(policy="logical-output-elements", value=4),
    )

    evidence = _path_evidence(
        implementation,
        {
            "tensor2vector": tensor_ir,
            "vector_kernel_inline": f'STR[1] "{helper_name}"\n',
        },
        [prepared],
    )

    assert evidence["requested_plan_kind"] == "fast-gemm"
    assert evidence["actual_plan_kind"] == "fast-gemm"
    assert evidence["requested_plan_provider"] == provenance
    assert evidence["actual_plan_provider"] == provenance
    assert evidence["plan_provenance"] == provenance
    assert evidence["specialization_key"] == "fast-gemm:test"
    assert evidence["pre_inline_helper_definitions"] == 1
    assert evidence["pre_inline_helper_calls"] == 1
    assert evidence["post_inline_helper_definitions"] == 0
    assert evidence["post_inline_helper_calls"] == 0
    assert evidence["post_inline_helper_name_occurrences"] == 1
    assert evidence["rotation_rnums"] == [[0, 2]]
    assert evidence["inline_success"]
    assert not evidence["fallback_used"]

    blocked = _path_evidence(
        implementation,
        {"tensor2vector": tensor_ir},
        [prepared],
        require_inlined=False,
    )
    assert not blocked["inline_success"]
    assert blocked["post_inline_helper_definitions"] is None
    assert blocked["post_inline_helper_calls"] is None


def test_implementation_cli_preserves_default_and_accepts_exact_three_way(
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["gemm_e2e_compare.py"])
    default = _parse_arguments()
    assert tuple(default.implementations) == IMPLEMENTATIONS

    monkeypatch.setattr(
        sys,
        "argv",
        ["gemm_e2e_compare.py", "--implementations", *THREE_WAY_IMPLEMENTATIONS],
    )
    selected = _parse_arguments()
    assert tuple(selected.implementations) == THREE_WAY_IMPLEMENTATIONS

    monkeypatch.setattr(
        sys,
        "argv",
        ["gemm_e2e_compare.py", "--implementations", *FAST_PAIR_IMPLEMENTATIONS],
    )
    selected = _parse_arguments()
    assert tuple(selected.implementations) == FAST_PAIR_IMPLEMENTATIONS


def test_implementation_cli_accepts_exact_four_way(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemm_e2e_compare.py", "--implementations", *FOUR_WAY_IMPLEMENTATIONS],
    )
    selected = _parse_arguments()
    assert tuple(selected.implementations) == FOUR_WAY_IMPLEMENTATIONS


def test_new_4096_by_10_model_cli_uses_4096_slots(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemm_e2e_compare.py", "--models", "gemmh10w4096"],
    )

    selected = _parse_arguments()

    assert selected.models == ["gemmh10w4096"]
    assert MODEL_SLOTS["gemmh10w4096"] == 4096


def test_context_cli_accepts_resnet_slot_and_ring_settings(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemm_e2e_compare.py",
            "--max-slots",
            "32768",
            "--poly-degree",
            "65536",
        ],
    )

    selected = _parse_arguments()

    assert selected.max_slots == 32768
    assert selected.poly_degree == 65536


@pytest.mark.parametrize(
    "implementations",
    [
        ("dsl-fast", "dsl-fast", "cpp-baseline"),
        ("dsl-fast", "cpp-baseline"),
    ],
)
def test_implementation_cli_rejects_duplicates_and_partial_modes(
    monkeypatch, implementations
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemm_e2e_compare.py", "--implementations", *implementations],
    )
    with pytest.raises(SystemExit):
        _parse_arguments()
