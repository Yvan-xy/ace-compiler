"""Unit tests for the local-fixture baseline-Conv E2E harness."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

import pytest

from ace_edsl.tests import conv_e2e_compare as conv
from ace_edsl.tests import gemm_e2e_compare as shared


def test_specialization_matches_conv_driver_terminal_block(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "MODEL_SLOTS", {})
    monkeypatch.setattr(shared, "IMPLEMENTATIONS", ())
    monkeypatch.setattr(shared, "DRIVER_END", "not Conv")
    monkeypatch.setattr(shared, "VALIDATING_DRIVER_END", "not Conv")
    monkeypatch.setattr(shared, "RESULT_PATTERN", re.compile("not Conv"))
    monkeypatch.setattr(shared, "_write_validating_driver", lambda: None)
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


def test_validating_driver_accepts_compact_terminal_block(tmp_path):
    model = next(iter(conv.MODEL_SLOTS))
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    (model_dir / f"{model}.c").write_text(
        "int main(void) {\n" + conv._CONV_COMPACT_DRIVER_END
    )

    driver = conv._write_validating_driver(model, output_dir, model_dir)

    generated = driver.read_text()
    assert conv._CONV_COMPACT_DRIVER_END not in generated
    assert "CONV_RESULT=%s" in generated


def test_comparison_cli_requires_complete_distinct_path_set(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conv_e2e_compare.py",
            "--implementations",
            "metakernel-fast",
            "dsl-fast",
            "cpp-baseline",
        ],
    )
    arguments = conv._parse_arguments()
    assert set(arguments.implementations) == set(
        conv.THREE_WAY_IMPLEMENTATIONS
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "conv_e2e_compare.py",
            "--implementations",
            "python-dsl-fast",
            "metakernel-fast",
            "dsl-fast",
            "cpp-baseline",
        ],
    )
    arguments = conv._parse_arguments()
    assert set(arguments.implementations) == set(
        conv.FOUR_WAY_IMPLEMENTATIONS
    )

    invalid_selections = (
        ("dsl-fast", "cpp-baseline"),
        ("dsl-fast", "cpp-baseline", "python-dsl-fast"),
        ("dsl-fast", "dsl-fast", "metakernel-fast"),
        ("native", "dsl-fast", "cpp-baseline", "metakernel-fast"),
        ("dsl", "native"),
    )
    for selection in invalid_selections:
        monkeypatch.setattr(
            sys,
            "argv",
            ["conv_e2e_compare.py", "--implementations", *selection],
        )
        with pytest.raises(SystemExit):
            conv._parse_arguments()


def test_generated_helper_ir_excludes_caller_rotations():
    helper_name = "__ace_vkernel_fast_conv_" + "a" * 64
    module_ir = f'''\
FUN[0] "Main_graph"
  VECTOR.roll ATTR[nums=(1,2)]
  call "{helper_name}" ENT[0x3]
FUN[0x2] "{helper_name}"
  func_entry "{helper_name}" ENT[0x3]
  COMMENT "BlockingRot replicate=1"
  COMMENT "BlockingRot HRot=bs=4"
  VECTOR.roll ATTR[nums=(3,4)]
'''

    helper_ir = conv._generated_helper_ir(module_ir, helper_name)

    assert shared._rotation_numbers(helper_ir) == [[3, 4]]
    assert "ATTR[nums=(1,2)]" not in helper_ir
    with pytest.raises(RuntimeError, match="expected one AIR definition region"):
        conv._generated_helper_ir(module_ir, helper_name + "b")


def test_sequence_offsets_require_one_exact_contiguous_copy():
    assert conv._sequence_offsets([[1], [2], [3]], [[2], [3]]) == [1]
    assert conv._sequence_offsets([[1], [2], [1], [2]], [[1], [2]]) == [0, 2]
    assert conv._sequence_offsets([[1], [3]], [[1], [2]]) == []
    assert conv._sequence_offsets([[1]], []) == []


def test_unsupported_inventory_records_unreadable_onnx(tmp_path, monkeypatch):
    def fail_load(*_args, **_kwargs):
        raise ValueError("not an ONNX model")

    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=fail_load))
    (tmp_path / "broken.onnx").write_bytes(b"not an ONNX model")

    unsupported = conv._unsupported_conv_models(tmp_path)

    assert len(unsupported) == 1
    assert unsupported[0]["model"] == "broken"
    assert unsupported[0]["conv_nodes"] is None
    assert "cannot be inspected" in unsupported[0]["reason"]
