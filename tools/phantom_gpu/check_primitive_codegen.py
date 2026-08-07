#!/usr/bin/env python3
"""Audit a generated Phantom primitive CUDA translation unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from check_configuration import read_json, verify_context_manifest


REQUIRED = (
    '#include "rt_phantom/rt_phantom.h"',
    "Get_phantom_context_manifest()",
    "Get_phantom_resource_manifest()",
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
    "obsolete bootstrap flag": re.compile(r"\bNeed_bts\s*\("),
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
    "duplicate legacy context": re.compile(
        r"\b(?:CKKS_PARAMS|Get_context_params|Get_phantom_ordinary_features)\b"
    ),
}

ARRAY_PATTERN = re.compile(
    r"static\s+const\s+uint32_t\s+(?P<name>[A-Za-z_]\w*)\[\]\s*=\s*"
    r"\{(?P<values>[^}]*)\}\s*;",
    re.MULTILINE,
)
CONTEXT_PATTERN = re.compile(
    r"static\s+const\s+PHANTOM_CONTEXT_MANIFEST\s+context\s*=\s*"
    r"\{(?P<fields>[^}]*)\}\s*;",
    re.MULTILINE,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--required-token", action="append")
    parser.add_argument("--context-manifest", type=Path)
    return parser.parse_args()


def _comma_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def extract_context(source: str) -> tuple[dict[str, object] | None, list[str]]:
    arrays = {
        match.group("name"): [
            int(value) for value in _comma_values(match.group("values"))
        ]
        for match in ARRAY_PATTERN.finditer(source)
    }
    data_names = [name for name in arrays if name.endswith("phantom_data_q_bit_sizes")]
    special_names = [
        name for name in arrays if name.endswith("phantom_special_p_bit_sizes")
    ]
    if len(data_names) != 1 or len(special_names) != 1:
        return None, [
            "generated source must define exactly one data-Q and special-P array"
        ]
    match = CONTEXT_PATTERN.search(source)
    if match is None:
        return None, ["PHANTOM_CONTEXT_MANIFEST initializer is absent"]
    fields = _comma_values(match.group("fields"))
    if len(fields) != 15:
        return None, ["PHANTOM_CONTEXT_MANIFEST initializer has the wrong arity"]
    data_name = data_names[0]
    special_name = special_names[0]
    if fields[1] != "PHANTOM_PACKING_FULL" or fields[5] != data_name or fields[7] != special_name:
        return None, ["PHANTOM_CONTEXT_MANIFEST initializer has invalid structural fields"]
    try:
        numeric = [
            int(fields[index])
            for index in (0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14)
        ]
    except ValueError:
        return None, ["PHANTOM_CONTEXT_MANIFEST initializer contains a non-integer field"]
    (
        schema_version,
        polynomial_degree,
        logical_slots,
        data_q_count,
        special_p_count,
        input_level,
        q_part_count,
        hamming_weight,
        security_level,
        first_modulus_bits,
        scaling_modulus_bits,
        resource_schema_version,
    ) = numeric
    data_q = arrays[data_name]
    special_p = arrays[special_name]
    errors: list[str] = []
    if data_q_count != len(data_q) or special_p_count != len(special_p):
        errors.append("generated context array lengths disagree with their counts")
    return {
        "schema_version": schema_version,
        "packing": "full",
        "polynomial_degree": polynomial_degree,
        "logical_slot_capacity": logical_slots,
        "data_q_bit_sizes": data_q,
        "special_p_bit_sizes": special_p,
        "input_level": input_level,
        "q_part_count": q_part_count,
        "hamming_weight": hamming_weight,
        "security_level": security_level,
        "first_modulus_bits": first_modulus_bits,
        "scaling_modulus_bits": scaling_modulus_bits,
        "resource_schema_version": resource_schema_version,
    }, errors


def compare_context(source: str, manifest: dict[str, object]) -> list[str]:
    mismatches: list[str] = []
    arrays = {
        match.group("name"): [int(value) for value in _comma_values(match.group("values"))]
        for match in ARRAY_PATTERN.finditer(source)
    }
    data_names = [name for name in arrays if name.endswith("phantom_data_q_bit_sizes")]
    special_names = [
        name for name in arrays if name.endswith("phantom_special_p_bit_sizes")
    ]
    if len(data_names) != 1 or len(special_names) != 1:
        return ["generated source must define exactly one data-Q and special-P array"]
    data_name = data_names[0]
    special_name = special_names[0]
    if arrays[data_name] != manifest["data_q_bit_sizes"]:
        mismatches.append("generated data-Q array differs from context manifest")
    if arrays[special_name] != manifest["special_p_bit_sizes"]:
        mismatches.append("generated special-P array differs from context manifest")

    match = CONTEXT_PATTERN.search(source)
    if match is None:
        mismatches.append("PHANTOM_CONTEXT_MANIFEST initializer is absent")
        return mismatches
    fields = _comma_values(match.group("fields"))
    expected = [
        str(manifest["schema_version"]),
        "PHANTOM_PACKING_FULL",
        str(manifest["polynomial_degree"]),
        str(manifest["logical_slot_capacity"]),
        str(len(manifest["data_q_bit_sizes"])),
        data_name,
        str(len(manifest["special_p_bit_sizes"])),
        special_name,
        str(manifest["input_level"]),
        str(manifest["q_part_count"]),
        str(manifest["hamming_weight"]),
        str(manifest["security_level"]),
        str(manifest["first_modulus_bits"]),
        str(manifest["scaling_modulus_bits"]),
        str(manifest["resource_schema_version"]),
    ]
    if fields != expected:
        mismatches.append("generated context initializer differs from manifest")
    return mismatches


def main() -> int:
    arguments = parse_arguments()
    source_bytes = arguments.source.read_bytes()
    source = source_bytes.decode("utf-8")
    required = tuple(arguments.required_token or REQUIRED)
    missing = [token for token in required if token not in source]
    forbidden = [
        label for label, pattern in FORBIDDEN.items() if pattern.search(source)
    ]
    emitted_context, context_mismatches = extract_context(source)
    context_digest = None
    if arguments.context_manifest:
        manifest = read_json(arguments.context_manifest)
        verify_context_manifest(manifest)
        context_mismatches.extend(compare_context(source, manifest))
        context_digest = hashlib.sha256(arguments.context_manifest.read_bytes()).hexdigest()
    report = {
        "status": (
            "pass"
            if not missing and not forbidden and not context_mismatches
            else "fail"
        ),
        "source": str(arguments.source),
        "size_bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "required_tokens": list(required),
        "missing_required_tokens": missing,
        "forbidden_matches": forbidden,
        "compiler_context_manifest": (
            str(arguments.context_manifest) if arguments.context_manifest else None
        ),
        "compiler_context_manifest_sha256": context_digest,
        "emitted_context": emitted_context,
        "context_mismatches": context_mismatches,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
