"""Unit tests for the local-fixture baseline-GEMM E2E harness."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

import pytest

from ace_edsl.tests.gemm_e2e_compare import (
    _compare_outputs,
    _parse_run,
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
