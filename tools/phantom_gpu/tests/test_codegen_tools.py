from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import onnx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"
sys.path.insert(0, str(TOOLS_ROOT))

from check_configuration import verify_context_manifest  # noqa: E402
from generate_native_terminal_probe import build_model  # noqa: E402


CLEAN_SOURCE = """#include "rt_phantom/rt_phantom.h"
void probe() {
  Add_ciph();
  Mul_ciph();
  Rotate_ciph();
}
static const uint32_t phantom_data_q_bit_sizes[] = {60, 56, 56};
static const uint32_t phantom_special_p_bit_sizes[] = {60};
extern "C" const PHANTOM_CONTEXT_MANIFEST* Get_phantom_context_manifest() {
  static const PHANTOM_CONTEXT_MANIFEST context = {
    1, PHANTOM_PACKING_FULL, 16384, 8192, 3,
    phantom_data_q_bit_sizes, 1, phantom_special_p_bit_sizes,
    1, 1, 192, 0, 60, 56, 1
  };
  return &context;
}
extern "C" const PHANTOM_RESOURCE_MANIFEST* Get_phantom_resource_manifest() {
  static const PHANTOM_RESOURCE_MANIFEST resources = {1, 1, 0, 0, nullptr};
  return &resources;
}
"""


CONTEXT = {
    "schema_version": 1,
    "packing": "full",
    "polynomial_degree": 16384,
    "logical_slot_capacity": 8192,
    "data_q_bit_sizes": [60, 56, 56],
    "special_p_bit_sizes": [60],
    "input_level": 1,
    "q_part_count": 1,
    "hamming_weight": 192,
    "security_level": 0,
    "first_modulus_bits": 60,
    "scaling_modulus_bits": 56,
    "resource_schema_version": 1,
}


def run_audit(
    tmp_path: Path, source: str, *arguments: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    source_path = tmp_path / "probe.cu"
    report_path = tmp_path / "report.json"
    source_path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_ROOT / "check_primitive_codegen.py"),
            str(source_path),
            "--report",
            str(report_path),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result, json.loads(report_path.read_text(encoding="utf-8"))


def write_context_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "compiler_context_manifest.json"
    path.write_text(
        json.dumps(CONTEXT, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_compiler_context_manifest_schema_is_checked() -> None:
    verify_context_manifest(CONTEXT)


def test_default_and_manifest_bound_source_audits(tmp_path: Path) -> None:
    context_path = write_context_manifest(tmp_path)
    result, report = run_audit(
        tmp_path, CLEAN_SOURCE, "--context-manifest", str(context_path)
    )
    assert result.returncode == 0, result.stderr
    assert report["status"] == "pass"
    assert report["context_mismatches"] == []
    assert report["emitted_context"] == CONTEXT
    assert context_path.read_bytes() == json.dumps(
        CONTEXT, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def test_custom_required_tokens_support_native_add_probe(tmp_path: Path) -> None:
    result, report = run_audit(
        tmp_path,
        CLEAN_SOURCE,
        "--required-token",
        '#include "rt_phantom/rt_phantom.h"',
        "--required-token",
        "Add_ciph(",
    )
    assert result.returncode == 0, result.stderr
    assert report["status"] == "pass"


@pytest.mark.parametrize(
    ("forbidden_source", "category"),
    [
        ('#include "rt_ant/rt_ant.h"', "ANT runtime header"),
        ('#include "rt_seal/rt_seal.h"', "SEAL runtime header"),
        ("LIB_ANT", "ANT provider"),
        ("LIB_SEAL", "SEAL provider"),
        ("Hw_add();", "hardware-level call"),
        ("Poly_add();", "POLY-level call"),
        ("Bootstrap();", "opaque bootstrap call"),
        ("Need_bts();", "obsolete bootstrap flag"),
        ("Eval_bootstrap_ciph();", "ANT bootstrap evaluator"),
        ("Phantom_bootstrap();", "native Phantom bootstrap call"),
        ("bootstrap_3();", "native Phantom bootstrap implementation"),
        ("Bootstrapper value;", "native Phantom bootstrap implementation"),
        ('#include "phantom.h"', "direct Phantom header"),
        ('#include <boot/Bootstrapper.cuh>', "direct Phantom header"),
        ("phantom::CKKSEvaluator value;", "direct Phantom implementation"),
        ("bootstrap_coeffs_to_slots();", "bootstrap coefficient stage"),
        ("CoeffToSlots();", "bootstrap coefficient stage"),
        ("EvalMod();", "bootstrap evaluation stage"),
        ("bootstrap_slots_to_coeffs();", "bootstrap slot stage"),
        ("SlotToCoeffs();", "bootstrap slot stage"),
    ],
)
def test_source_audit_rejects_native_or_staged_bootstrap(
    tmp_path: Path, forbidden_source: str, category: str
) -> None:
    result, report = run_audit(
        tmp_path, CLEAN_SOURCE + "\n" + forbidden_source + "\n"
    )
    assert result.returncode == 1
    assert category in report["forbidden_matches"]


def test_manifest_mismatch_fails_source_audit(tmp_path: Path) -> None:
    context_path = write_context_manifest(tmp_path)
    mismatched = CLEAN_SOURCE.replace("16384, 8192, 3", "32768, 8192, 3")
    result, report = run_audit(
        tmp_path, mismatched, "--context-manifest", str(context_path)
    )
    assert result.returncode == 1
    assert report["context_mismatches"] == [
        "generated context initializer differs from manifest"
    ]


def test_native_terminal_model_is_deterministic_and_checked() -> None:
    first = build_model()
    second = build_model()
    onnx.checker.check_model(first)
    first_bytes = first.SerializeToString(deterministic=True)
    second_bytes = second.SerializeToString(deterministic=True)
    assert first_bytes == second_bytes
    assert len(first.graph.node) == 1
    node = first.graph.node[0]
    assert node.op_type == "Add"
    assert list(node.input) == ["x", "y"]
    assert list(node.output) == ["z"]
