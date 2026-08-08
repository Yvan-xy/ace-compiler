#!/usr/bin/env python3
"""Generate ANT/POLY2C and Phantom/CKKS2C retained conformance sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ace_edsl.edsl import AceEDSL, AcePipeline  # noqa: E402
from ckks_retained_ops import (  # noqa: E402
    GENERATED_FUNCTIONS,
    declare_retained_ckks_conformance_module,
)
from retained_air_tools import (  # noqa: E402
    canonicalize_checkout_paths,
    require_clean_tracked_sources,
    sha256_text,
    write_json,
)


RETAINED_CALLS = (
    "Conjugate_ciph",
    "Rotate_batch_ciph",
    "Raise_mod",
    "Mul_mono_ciph",
)


def _load_context(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "polynomial_degree",
        "data_q_bit_sizes",
        "input_level",
        "security_level",
        "scaling_modulus_bits",
        "first_modulus_bits",
        "hamming_weight",
    )
    if not isinstance(value, dict) or any(name not in value for name in required):
        raise SystemExit("context manifest is missing retained generation fields")
    return value


def _compile(
    context_path: Path,
    fixture_path: Path,
    *,
    provider: str,
    codegen_ir: str,
    context_output: Path | None = None,
    resource_output: Path | None = None,
) -> tuple[str, str]:
    context = _load_context(context_path)
    declare_retained_ckks_conformance_module(context_path, fixture_path)
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise SystemExit("retained conformance declaration emitted no AIR")
    pipeline = AcePipeline(module).configure_fhe(
        poly_degree=int(context["polynomial_degree"]),
        mul_level=len(context["data_q_bit_sizes"]),
        input_level=int(context["input_level"]),
        security_level=int(context["security_level"]),
        scaling_factor_bits=int(context["scaling_modulus_bits"]),
        first_prime_bits=int(context["first_modulus_bits"]),
        hamming_weight=int(context["hamming_weight"]),
        data_file="",
        provider=provider,
        codegen_ir=codegen_ir,
        context_manifest_file=str(context_output or ""),
        resource_manifest_file=str(resource_output or ""),
    )
    pipeline.set_ckks_extended_op_rewrite(False)
    result = pipeline.run(start_domain="fhe::ckks", dump_stages=True, verbose=False)
    if not result.success:
        raise SystemExit(f"{provider}/{codegen_ir} generation failed: {result.error}")
    expected_terminal = "poly2c" if codegen_ir == "poly" else "ckks2c"
    if not result.stages_completed or result.stages_completed[-1] != expected_terminal:
        raise SystemExit(
            f"{provider}/{codegen_ir} selected unexpected stages: "
            f"{result.stages_completed}"
        )
    post_air = result.air_dumps.get("ckks_driver", "")
    source = result.c_code or ""
    if not post_air or not source:
        raise SystemExit(f"{provider}/{codegen_ir} returned empty artifacts")
    return canonicalize_checkout_paths(post_air, REPOSITORY), source


def _audit_stable_functions(source: str, provider: str) -> None:
    for function in GENERATED_FUNCTIONS:
        declaration = re.compile(
            rf"\bCIPHERTEXT\s+{re.escape(function)}\s*\(\s*CIPHERTEXT\s+p0"
        )
        if declaration.search(source) is None:
            raise SystemExit(f"{provider} source omits stable function {function}")


def _audit_phantom_source(source: str) -> None:
    _audit_stable_functions(source, "Phantom")
    for call in RETAINED_CALLS:
        if f"{call}(" not in source:
            raise SystemExit(f"Phantom source omits retained call {call}")
    forbidden_calls = (
        r"\b(?:Eval_bootstrap_ciph|Bootstrap|bootstrap_[A-Za-z0-9_]*)\s*\(",
        r"\b(?:Coeff_to_slot|Slot_to_coeff|Eval_mod)\s*\(",
        r"\b(?:Hw_[A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*poly[A-Za-z0-9_]*)\s*\(",
    )
    for pattern in forbidden_calls:
        if re.search(pattern, source, re.IGNORECASE):
            raise SystemExit(f"Phantom source contains forbidden call matching {pattern}")


def generate(arguments: argparse.Namespace) -> None:
    expected_suffixes = {
        arguments.ant_source: ".cxx",
        arguments.phantom_source: ".cu",
        arguments.interface_header: ".h",
        arguments.ant_post_ckks_air: ".air",
        arguments.phantom_post_ckks_air: ".air",
    }
    for path, suffix in expected_suffixes.items():
        if path.suffix != suffix:
            raise SystemExit(f"{path} must use the {suffix} suffix")
    ace_commit = require_clean_tracked_sources(
        REPOSITORY,
        (
            "ace_edsl/examples/ckks_retained_ops.py",
            "ace_edsl/edsl",
            "tools/phantom_gpu/fixtures/retained_ckks_v1.json",
            "tools/phantom_gpu/generate_retained_ckks_sources.py",
            "tools/phantom_gpu/retained_air_tools.py",
        ),
    )
    ant_air, ant_source = _compile(
        arguments.context_manifest,
        arguments.fixture,
        provider="ant",
        codegen_ir="poly",
    )
    phantom_air, phantom_source = _compile(
        arguments.context_manifest,
        arguments.fixture,
        provider="phantom",
        codegen_ir="ckks",
        context_output=arguments.phantom_context_manifest,
        resource_output=arguments.phantom_resource_manifest,
    )
    if ant_air != phantom_air:
        raise SystemExit("ANT and Phantom terminal paths received different post-CKKS AIR")
    emitted_context = _load_context(arguments.phantom_context_manifest)
    if emitted_context != _load_context(arguments.context_manifest):
        raise SystemExit("emitted Phantom context differs from the generator input")
    _audit_stable_functions(ant_source, "ANT")
    _audit_phantom_source(phantom_source)
    artifacts = (
        (arguments.ant_source, ant_source),
        (arguments.phantom_source, phantom_source),
        (arguments.ant_post_ckks_air, ant_air),
        (arguments.phantom_post_ckks_air, phantom_air),
    )
    for path, value in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    interface = """#ifndef ACE_RETAINED_CKKS_GENERATED_INTERFACE_H
#define ACE_RETAINED_CKKS_GENERATED_INTERFACE_H

// Include rt_ant/rt_ant.h or rt_phantom/rt_phantom.h before this file.
// Both generated sources use C++ linkage; retain the .cxx/.cu suffixes.
// The input is passed by value exactly as emitted by POLY2C and CKKS2C.
CIPHERTEXT retained_ckks_conjugate(CIPHERTEXT input);
CIPHERTEXT retained_ckks_rotate_batches(CIPHERTEXT input);
CIPHERTEXT retained_ckks_raise_mod(CIPHERTEXT input);
CIPHERTEXT retained_ckks_mul_monomials(CIPHERTEXT input);
CIPHERTEXT retained_ckks_composite(CIPHERTEXT input);

#endif
"""
    arguments.interface_header.parent.mkdir(parents=True, exist_ok=True)
    arguments.interface_header.write_text(interface, encoding="utf-8")
    canonical_argv = [
        "tools/phantom_gpu/generate_retained_ckks_sources.py",
        "--context-manifest", str(arguments.context_manifest),
        "--fixture", str(arguments.fixture),
        "--ant-source", str(arguments.ant_source),
        "--phantom-source", str(arguments.phantom_source),
        "--ant-post-ckks-air", str(arguments.ant_post_ckks_air),
        "--phantom-post-ckks-air", str(arguments.phantom_post_ckks_air),
        "--phantom-context-manifest", str(arguments.phantom_context_manifest),
        "--phantom-resource-manifest", str(arguments.phantom_resource_manifest),
        "--generation-record", str(arguments.generation_record),
        "--interface-header", str(arguments.interface_header),
    ]
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        arguments.generation_record,
        {
            "schema_version": "ace.retained_ckks.generated-sources/1.0.0",
            "argv": canonical_argv,
            "normalized_argv_sha256": hashlib.sha256(
                json.dumps(
                    canonical_argv, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "canonical_module": "ace_edsl/examples/ckks_retained_ops.py",
            "ace_commit": ace_commit,
            "fixture_sha256": digest(arguments.fixture),
            "input_context_manifest_sha256": digest(arguments.context_manifest),
            "emitted_context_manifest_sha256": digest(
                arguments.phantom_context_manifest
            ),
            "resource_manifest_sha256": digest(
                arguments.phantom_resource_manifest
            ),
            "generated_functions": list(GENERATED_FUNCTIONS),
            "function_abi": "CIPHERTEXT function(CIPHERTEXT input)",
            "linkage": "C++ (.cxx for ANT/POLY2C; .cu for Phantom/CKKS2C)",
            "invocation": (
                "CIPHERTEXT output = retained_ckks_composite(*input_cipher);"
            ),
            "retained_runtime_calls": list(RETAINED_CALLS),
            "post_ckks_air_sha256": sha256_text(ant_air),
            "ant_post_ckks_air_sha256": digest(arguments.ant_post_ckks_air),
            "phantom_post_ckks_air_sha256": digest(
                arguments.phantom_post_ckks_air
            ),
            "ant_source_sha256": hashlib.sha256(ant_source.encode()).hexdigest(),
            "phantom_source_sha256": hashlib.sha256(
                phantom_source.encode()
            ).hexdigest(),
        },
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--ant-source", required=True, type=Path)
    parser.add_argument("--phantom-source", required=True, type=Path)
    parser.add_argument("--ant-post-ckks-air", required=True, type=Path)
    parser.add_argument("--phantom-post-ckks-air", required=True, type=Path)
    parser.add_argument("--phantom-context-manifest", required=True, type=Path)
    parser.add_argument("--phantom-resource-manifest", required=True, type=Path)
    parser.add_argument("--generation-record", required=True, type=Path)
    parser.add_argument("--interface-header", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    generate(parse_arguments())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
