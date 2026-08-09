#!/usr/bin/env python3
"""Generate reproducible full primitive-bootstrap compile-only artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "ace_edsl" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from bootstrap_full import (  # noqa: E402
    G_COEFFICIENTS_UNIFORM_HW_192,
    _bootstrap_trace_config,
    bootstrap_full,
)
from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext  # noqa: E402
from retained_air_tools import canonicalize_checkout_paths  # noqa: E402


SCHEMA = "ace.phantom.bootstrap-generation/2.0.0"
CONTROL_ENVIRONMENT = (
    "ACE_BOOTSTRAP_POLY_DEGREE",
    "ACE_BOOTSTRAP_MUL_LEVEL",
    "ACE_BOOTSTRAP_INPUT_LEVEL",
    "ACE_BOOTSTRAP_FIRST_PRIME_BITS",
    "ACE_BOOTSTRAP_FIRST_MOD_SIZE",
    "ACE_BOOTSTRAP_SCALING_FACTOR_BITS",
    "ACE_BOOTSTRAP_SCALING_MOD_SIZE",
    "ACE_BOOTSTRAP_HAMMING_WEIGHT",
    "ACE_BOOTSTRAP_Q_PARTS",
    "ACE_BOOTSTRAP_TRANSFORM_LEVEL_BUDGET",
    "ACE_BOOTSTRAP_ENC_BUDGET",
    "ACE_BOOTSTRAP_DEC_BUDGET",
    "ACE_BOOTSTRAP_CT_ENCODE",
    "ACE_BOOTSTRAP_RUNTIME_RAISE_LEVEL",
    "ACE_BOOTSTRAP_STAGE_PROBE",
    "ACE_BOOTSTRAP_FUNCTION_NAME_PREFIX",
    "ACE_BOOTSTRAP_CONSTANT_NAME_PREFIX",
    "ACE_BOOTSTRAP_PT_FROM_MSG_NAME",
    "ACE_BOOTSTRAP_RAISE_LEVEL_NAME",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--poly-degree", type=int, default=16384)
    parser.add_argument("--mul-level", type=int, default=26)
    parser.add_argument("--input-level", type=int, default=1)
    parser.add_argument("--security-level", type=int, default=0)
    parser.add_argument("--scaling-factor-bits", type=int, default=56)
    parser.add_argument("--first-prime-bits", type=int, default=60)
    parser.add_argument("--hamming-weight", type=int, default=192)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    arguments = parse_arguments()
    if arguments.output_dir.exists():
        raise SystemExit("output_dir must not already exist")
    unexpected = [name for name in CONTROL_ENVIRONMENT if os.environ.get(name)]
    if unexpected:
        raise SystemExit(
            "ambient bootstrap controls are forbidden: " + ", ".join(unexpected)
        )

    expected = {
        "poly_degree": 16384,
        "mul_level": 26,
        "input_level": 1,
        "security_level": 0,
        "scaling_factor_bits": 56,
        "first_prime_bits": 60,
        "hamming_weight": 192,
    }
    observed = {name: getattr(arguments, name) for name in expected}
    if observed != expected:
        raise SystemExit(
            "bootstrap qualification requires the approved default context: "
            + json.dumps(expected, sort_keys=True)
        )

    arguments.output_dir.mkdir(parents=True)
    source_path = arguments.output_dir / "bootstrap_qualification.cu"
    raw_air_path = arguments.output_dir / "bootstrap_raw.air"
    post_ckks_air_path = arguments.output_dir / "bootstrap_post_ckks.air"
    context_path = arguments.output_dir / "compiler_context_manifest.json"
    resource_path = arguments.output_dir / "compiler_resource_manifest.json"
    constant_path = arguments.output_dir / "compiler_constant_manifest.json"
    record_path = arguments.output_dir / "generation.json"

    os.environ["ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING"] = "1"
    AceEDSL._get_dsl.cache_clear()
    config = _bootstrap_trace_config()
    if (
        config.poly_degree != arguments.poly_degree
        or config.mul_level != arguments.mul_level
        or config.scaling_factor_bits != arguments.scaling_factor_bits
        or config.first_prime_bits != arguments.first_prime_bits
        or config.hamming_weight != arguments.hamming_weight
        or config.q_parts != 3
        or config.ct_encode
    ):
        raise SystemExit(
            "checked-in bootstrap defaults disagree with the qualification context"
        )

    shape = (config.poly_degree,)
    ciphertext = CkksCiphertext(shape=shape, name="input_ct")
    zero = CkksCiphertext(shape=shape, name="zero_ct")
    kernel_arguments = (
        [ciphertext, zero, 1.0]
        + list(G_COEFFICIENTS_UNIFORM_HW_192)
        + list(config.double_angle_scalars)
        + [config.post_scale]
    )
    bootstrap_full(*kernel_arguments)
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise SystemExit("bootstrap tracing returned no AIR module")
    raw_air = canonicalize_checkout_paths(module.dump(), REPO_ROOT)
    if not raw_air:
        raise SystemExit("bootstrap tracing returned empty raw AIR")
    raw_air_path.write_text(raw_air, encoding="utf-8")

    pipeline = AcePipeline(module).configure_fhe(
        **observed,
        data_file="",
        provider="phantom",
        codegen_ir="ckks",
        context_manifest_file=str(context_path),
        resource_manifest_file=str(resource_path),
        constant_manifest_file=str(constant_path),
    )
    pipeline.set_ckks_extended_op_rewrite(False)
    result = pipeline.run(
        start_domain="fhe::ckks", dump_stages=True, verbose=False
    )
    if not result.success:
        raise SystemExit(f"bootstrap CKKS2C generation failed: {result.error}")
    if result.stages_completed != ["ckks_driver", "ckks2c"]:
        raise SystemExit(f"unexpected terminal stages: {result.stages_completed}")
    source = result.c_code or ""
    if not source:
        raise SystemExit("bootstrap CKKS2C generation returned empty source")
    post_ckks_air = result.air_dumps.get("ckks_driver", "")
    if not post_ckks_air:
        raise SystemExit("bootstrap CKKS2C generation returned no post-CKKS AIR")
    post_ckks_air = canonicalize_checkout_paths(post_ckks_air, REPO_ROOT)
    post_ckks_air_path.write_text(post_ckks_air, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")

    manifests = {}
    for name, path in (
        ("context", context_path),
        ("resource", resource_path),
        ("constant", constant_path),
    ):
        if not path.is_file() or path.read_bytes().endswith(b"\n"):
            raise SystemExit(
                f"{name} manifest must use the compiler's canonical "
                "no-newline serialization"
            )
        manifests[name] = {
            "path": path.name,
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }

    constants = json.loads(constant_path.read_text(encoding="utf-8"))["constants"]
    if not constants:
        raise SystemExit("full bootstrap emitted no cached complex constants")
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    required_true = (
        "relinearization_key",
        "conjugation_key",
        "rotate_batch",
        "raise_mod",
        "complex_plaintext",
    )
    if any(resource.get(field) is not True for field in required_true):
        raise SystemExit("full bootstrap resource manifest is incomplete")
    if resource.get("native_bootstrap_precompute") is not False:
        raise SystemExit("native bootstrap precomputation must remain disabled")
    if not resource.get("rotation_steps") or not resource.get("monomial_powers"):
        raise SystemExit("full bootstrap rotations/monomials must be nonempty")

    record = {
        "schema_version": SCHEMA,
        "status": "pass",
        "qualification_scope": "full-generated-bootstrap-compile-only",
        "compiler_parameters": observed,
        "bootstrap_parameters": {
            "q_parts": config.q_parts,
            "enc_budget": config.enc_budget,
            "dec_budget": config.dec_budget,
            "ct_encode": config.ct_encode,
        },
        "stages_completed": result.stages_completed,
        "source": {
            "path": source_path.name,
            "sha256": sha256_path(source_path),
            "bytes": source_path.stat().st_size,
        },
        "air": {
            "raw": {
                "path": raw_air_path.name,
                "sha256": sha256_path(raw_air_path),
                "bytes": raw_air_path.stat().st_size,
            },
            "post_ckks": {
                "path": post_ckks_air_path.name,
                "sha256": sha256_path(post_ckks_air_path),
                "bytes": post_ckks_air_path.stat().st_size,
            },
        },
        "manifests": manifests,
        "constant_count": len(constants),
        "rotation_count": len(resource["rotation_steps"]),
        "rotation_batch_count": len(resource["rotation_batches"]),
        "monomial_count": len(resource["monomial_powers"]),
        "native_bootstrap_precompute": False,
        "generated_program_executed": False,
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"generated {source_path} with {len(constants)} cached constants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
