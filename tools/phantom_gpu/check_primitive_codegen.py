#!/usr/bin/env python3
"""Audit a generated Phantom primitive CUDA translation unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from check_configuration import (
    profile_codegen_parameters,
    read_json,
    verify_profile,
)


REQUIRED = (
    '#include "rt_phantom/rt_phantom.h"',
    "LIB_PHANTOM",
    "Add_ciph(",
    "Mul_ciph(",
    "Rotate_ciph(",
)

FORBIDDEN = {
    "ANT runtime header": re.compile(r"rt_ant/"),
    "SEAL runtime header": re.compile(r"rt_seal/"),
    "direct Phantom header": re.compile(
        r'^\s*#\s*include\s*[<"]'
        r'(?:phantom\.h|boot/Bootstrapper\.cuh)[>"]',
        re.MULTILINE,
    ),
    "ANT provider": re.compile(r"\bLIB_ANT\b"),
    "SEAL provider": re.compile(r"\bLIB_SEAL\b"),
    "hardware-level call": re.compile(r"\bHw_[A-Za-z0-9_]*\s*\("),
    "POLY-level call": re.compile(r"\bPoly_[A-Za-z0-9_]*\s*\("),
    "opaque bootstrap call": re.compile(r"\bBootstrap\s*\("),
    "ANT bootstrap evaluator": re.compile(r"\bEval_bootstrap[A-Za-z0-9_]*\s*\("),
    "native Phantom bootstrap call": re.compile(r"\bPhantom_bootstrap\s*\("),
    "native Phantom bootstrap implementation": re.compile(
        r"\b(?:Bootstrapper|bootstrap_3)\b"
    ),
    "bootstrap coefficient stage": re.compile(
        r"\b(?:bootstrap_coeffs_to_slots|CoeffToSlots?)\b"
    ),
    "bootstrap evaluation stage": re.compile(
        r"\b(?:bootstrap_eval_mod|EvalMod)\b"
    ),
    "bootstrap slot stage": re.compile(
        r"\b(?:bootstrap_slots_to_coeffs|SlotToCoeffs?)\b"
    ),
    "direct Phantom implementation": re.compile(r"\bphantom::"),
}

CKKS_PARAMS_PATTERN = re.compile(
    r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*LIB_PHANTOM\s*,\s*"
    + r"\s*,\s*".join([r"(\d+)"] * 9),
    re.MULTILINE,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--required-token", action="append")
    parser.add_argument("--profile", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_bytes = arguments.source.read_bytes()
    source = source_bytes.decode("utf-8")
    required = tuple(arguments.required_token or REQUIRED)
    missing = [token for token in required if token not in source]
    forbidden = [
        label for label, pattern in FORBIDDEN.items() if pattern.search(source)
    ]
    profile_mismatches: list[str] = []
    if arguments.profile:
        repo_root = Path(__file__).resolve().parents[2]
        profile = read_json(arguments.profile)
        verify_profile(repo_root, profile)
        expected = profile_codegen_parameters(profile)
        match = CKKS_PARAMS_PATTERN.search(source)
        if not match:
            profile_mismatches.append("CKKS_PARAMS initializer is absent")
        else:
            (
                poly_degree,
                security_level,
                mul_depth,
                input_level,
                first_prime_bits,
                scaling_factor_bits,
                q_parts,
                hamming_weight,
                _rotation_count,
            ) = (int(value) for value in match.groups())
            observed = {
                "poly_degree": poly_degree,
                "security_level": security_level,
                "mul_depth": mul_depth,
                "input_level": input_level,
                "first_prime_bits": first_prime_bits,
                "scaling_factor_bits": scaling_factor_bits,
                "q_parts": q_parts,
                "hamming_weight": hamming_weight,
            }
            expected_source = {
                "poly_degree": expected["poly_degree"],
                "security_level": expected["security_level"],
                "mul_depth": expected["mul_level"] - 1,
                "input_level": expected["input_level"],
                "first_prime_bits": expected["first_prime_bits"],
                "scaling_factor_bits": expected["scaling_factor_bits"],
                "q_parts": profile["special_p"]["derivation"]["q_parts"],
                "hamming_weight": expected["hamming_weight"],
            }
            for key, value in expected_source.items():
                if observed[key] != value:
                    profile_mismatches.append(
                        f"{key}: expected {value}, got {observed[key]}"
                    )
    report = {
        "status": (
            "pass"
            if not missing and not forbidden and not profile_mismatches
            else "fail"
        ),
        "source": str(arguments.source),
        "size_bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "required_tokens": list(required),
        "missing_required_tokens": missing,
        "forbidden_matches": forbidden,
        "profile": str(arguments.profile) if arguments.profile else None,
        "profile_mismatches": profile_mismatches,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
