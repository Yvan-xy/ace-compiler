from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import onnx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"
PROFILE_PATH = (
    REPO_ROOT / "fhe-cmplr/rtlib/phantom/config/fullpacked_bts_v1.json"
)
sys.path.insert(0, str(TOOLS_ROOT))

from check_configuration import (  # noqa: E402
    profile_codegen_parameters,
    render_cpp_profile_header,
)
from generate_native_terminal_probe import build_model  # noqa: E402


CLEAN_SOURCE = """#include "rt_phantom/rt_phantom.h"
void probe() {
  Add_ciph();
  Mul_ciph();
  Rotate_ciph();
}
CKKS_PARAMS* Get_context_params() {
  static CKKS_PARAMS parm = {
    LIB_PHANTOM, 16384, 0, 25, 1, 60, 56, 3, 192, 1, { 3 }
  };
  return &parm;
}
"""


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


def test_profile_mapping_and_native_header_follow_json() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    parameters = profile_codegen_parameters(profile)
    assert parameters == {
        "poly_degree": 16384,
        "mul_level": 26,
        "input_level": 1,
        "security_level": 0,
        "scaling_factor_bits": 56,
        "first_prime_bits": 60,
        "hamming_weight": 192,
        "ct_encode": False,
    }

    digest = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()
    header = render_cpp_profile_header(profile, digest)
    assert digest in header
    assert "kPolynomialDegree = 16384" in header
    assert "kActiveSlotCount = 8192" in header
    assert "std::array<int, 27> kDataQBitSizes" in header
    assert "std::array<int, 9> kSpecialPBitSizes" in header

    mutated = copy.deepcopy(profile)
    mutated["ring"]["polynomial_degree"] = 32768
    mutated["security"]["secret_key_hamming_weight"] = 128
    changed = profile_codegen_parameters(mutated)
    assert changed["poly_degree"] == 32768
    assert changed["hamming_weight"] == 128


def test_default_and_profile_bound_source_audits(tmp_path: Path) -> None:
    result, report = run_audit(
        tmp_path, CLEAN_SOURCE, "--profile", str(PROFILE_PATH)
    )
    assert result.returncode == 0, result.stderr
    assert report["status"] == "pass"
    assert report["profile_mismatches"] == []


def test_custom_required_tokens_support_native_add_probe(tmp_path: Path) -> None:
    source = '#include "rt_phantom/rt_phantom.h"\nLIB_PHANTOM\nAdd_ciph();\n'
    result, report = run_audit(
        tmp_path,
        source,
        "--required-token",
        '#include "rt_phantom/rt_phantom.h"',
        "--required-token",
        "LIB_PHANTOM",
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


def test_profile_mismatch_fails_source_audit(tmp_path: Path) -> None:
    mismatched = CLEAN_SOURCE.replace(
        "LIB_PHANTOM, 16384, 0, 25", "LIB_PHANTOM, 16384, 0, 24"
    )
    result, report = run_audit(
        tmp_path, mismatched, "--profile", str(PROFILE_PATH)
    )
    assert result.returncode == 1
    assert report["profile_mismatches"] == [
        "mul_depth: expected 25, got 24"
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
