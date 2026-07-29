"""Unit tests for the local-fixture baseline-Conv E2E harness."""

from __future__ import annotations

import re

import pytest

from ace_edsl.tests import conv_e2e_compare as conv
from ace_edsl.tests import gemm_e2e_compare as shared


def test_specialization_matches_conv_driver_terminal_block(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "MODEL_SLOTS", {})
    monkeypatch.setattr(shared, "IMPLEMENTATIONS", ())
    monkeypatch.setattr(shared, "DRIVER_END", "not Conv")
    monkeypatch.setattr(shared, "VALIDATING_DRIVER_END", "not Conv")
    monkeypatch.setattr(shared, "RESULT_PATTERN", re.compile("not Conv"))
    conv._specialize_shared_harness()

    assert shared.MODEL_SLOTS == conv.MODEL_SLOTS
    assert shared.IMPLEMENTATIONS == conv.IMPLEMENTATIONS
    assert shared.DRIVER_END == conv._CONV_DRIVER_END
    assert shared.VALIDATING_DRIVER_END == conv._CONV_VALIDATING_DRIVER_END
    assert shared.RESULT_PATTERN is conv._CONV_RESULT_PATTERN

    model = next(iter(conv.MODEL_SLOTS))
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    (model_dir / f"{model}.c").write_text(
        "int main(void) {\n" + conv._CONV_DRIVER_END
    )

    driver = shared._write_validating_driver(model, output_dir, model_dir)

    generated = driver.read_text()
    assert conv._CONV_DRIVER_END not in generated
    assert "CONV_RESULT=%s" in generated
    assert "GEMM_RESULT" not in generated


def test_validate_model_files_reports_missing_conv_fixture_pairs(tmp_path):
    model = next(iter(conv.MODEL_SLOTS))
    with pytest.raises(
        RuntimeError, match="missing local Conv model fixtures"
    ) as error:
        conv._validate_model_files(tmp_path, [model])

    message = str(error.value)
    assert str(tmp_path / f"{model}.onnx") in message
    assert str(tmp_path / f"{model}.c") in message
    assert "--model-dir" in message

    (tmp_path / f"{model}.onnx").write_bytes(b"onnx")
    (tmp_path / f"{model}.c").write_text("driver")
    conv._validate_model_files(tmp_path, [model])
