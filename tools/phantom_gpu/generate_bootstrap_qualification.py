#!/usr/bin/env python3
"""Generate explicit, reproducible full-packed bootstrap artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "ace_edsl" / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from bootstrap_full import (  # noqa: E402
    EVALMOD_COMPONENT_LOWER_BOUND,
    EVALMOD_COMPONENT_UPPER_BOUND,
    G_COEFFICIENTS_UNIFORM_HW_192,
    UNIFORM_COEFFICIENT_HAMMING_WEIGHT_MAX,
    bootstrap_full,
    bootstrap_trace_configuration,
    build_bootstrap_trace_config,
)
from ace_edsl.edsl import (  # noqa: E402
    AIRValue,
    AceEDSL,
    AcePipeline,
    CkksCiphertext,
    ckks_kernel,
)
from ace_edsl.edsl.core.bootstrap_decomposition import (  # noqa: E402
    build_bootstrap_evalmod_scalar_manifest,
    build_bootstrap_transform_payload_manifest,
)
from retained_air_tools import canonicalize_checkout_paths  # noqa: E402
from bootstrap_domain_attestation import (  # noqa: E402
    CLEAR_MAP_BUDGET_FRACTION,
    derive_supported_identity_domain,
)


GENERATION_SCHEMA = "ace.phantom.bootstrap-generation/4.0.0"
INVOCATION_SCHEMA = "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0"
SEMANTICS_SCHEMA = "ace.phantom.generated-bootstrap.semantics/2.0.0"
POST_OPERATIONS_SCHEMA = "ace.phantom.bootstrap-post-operation-semantics/1.0.0"
GENERATOR_TOOL = "tools/phantom_gpu/generate_bootstrap_qualification.py"
RAW_SCALE_COORDINATE_TOLERANCE = 1.0e-4
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
    "ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING",
    "ACE_BOOTSTRAP_FUNCTION_NAME_PREFIX",
    "ACE_BOOTSTRAP_CONSTANT_NAME_PREFIX",
    "ACE_BOOTSTRAP_PT_FROM_MSG_NAME",
    "ACE_BOOTSTRAP_RAISE_LEVEL_NAME",
    "ACE_CKKS_PRIMITIVE_REWRITE",
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--poly-degree", required=True, type=int)
    parser.add_argument("--vector-capacity", required=True, type=int)
    parser.add_argument("--mul-level", required=True, type=int)
    parser.add_argument("--input-level", required=True, type=int)
    parser.add_argument("--security-level", required=True, type=int)
    parser.add_argument("--scaling-factor-bits", required=True, type=int)
    parser.add_argument("--first-prime-bits", required=True, type=int)
    parser.add_argument("--hamming-weight", required=True, type=int)
    parser.add_argument("--q-part-count", required=True, type=int)
    parser.add_argument("--encode-transform-budget", required=True, type=int)
    parser.add_argument("--decode-transform-budget", required=True, type=int)
    parser.add_argument(
        "--ciphertext-constant-encoding",
        required=True,
        choices=("enabled", "disabled"),
    )
    parser.add_argument("--packing", required=True, choices=("full",))
    parser.add_argument("--post-multiply-real", required=True, type=float)
    parser.add_argument("--post-multiply-imag", required=True, type=float)
    parser.add_argument("--post-multiply-scale-degree", required=True, type=int)
    parser.add_argument("--post-rotation-step", required=True, type=int)
    parser.add_argument("--identity-error-threshold", required=True, type=float)
    return parser.parse_args(argv)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_arguments(arguments: argparse.Namespace) -> None:
    positive = (
        "poly_degree",
        "vector_capacity",
        "mul_level",
        "input_level",
        "scaling_factor_bits",
        "first_prime_bits",
        "hamming_weight",
        "q_part_count",
        "encode_transform_budget",
        "decode_transform_budget",
    )
    for name in positive:
        if getattr(arguments, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if arguments.poly_degree % 2 != 0:
        raise SystemExit("--poly-degree must be even")
    if arguments.vector_capacity != arguments.poly_degree // 2:
        raise SystemExit(
            "full packing requires --vector-capacity to equal half --poly-degree"
        )
    if arguments.first_prime_bits < arguments.scaling_factor_bits:
        raise SystemExit(
            "--first-prime-bits must be at least --scaling-factor-bits"
        )
    if arguments.security_level not in (0, 128, 192, 256):
        raise SystemExit("--security-level must be 0, 128, 192, or 256")
    if not math.isfinite(arguments.post_multiply_real) or not math.isfinite(
        arguments.post_multiply_imag
    ):
        raise SystemExit("post-operation multiplier components must be finite")
    if arguments.post_multiply_real == 0.0 and arguments.post_multiply_imag == 0.0:
        raise SystemExit("post-operation multiplier must be nonzero")
    if arguments.post_multiply_scale_degree != 0:
        raise SystemExit(
            "direct multiply at final active-Q 1 requires "
            "--post-multiply-scale-degree 0"
        )
    if arguments.post_rotation_step % arguments.vector_capacity == 0:
        raise SystemExit("post-operation rotation must be nonzero modulo vector capacity")
    if (
        not math.isfinite(arguments.identity_error_threshold)
        or arguments.identity_error_threshold <= 0.0
    ):
        raise SystemExit("--identity-error-threshold must be finite and positive")
    normalized_rotation = normalize_runtime_rotation_step(
        arguments.post_rotation_step,
        arguments.vector_capacity,
    )
    if arguments.post_rotation_step != normalized_rotation:
        raise SystemExit(
            "--post-rotation-step must use the canonical signed logical-slot "
            f"coordinate {normalized_rotation}"
        )


def normalize_runtime_rotation_step(step: int, logical_slots: int) -> int:
    normalized = step % logical_slots
    if normalized > logical_slots // 2:
        normalized -= logical_slots
    return normalized


def validate_post_rotation_resource(
    resource: dict[str, Any], arguments: argparse.Namespace
) -> None:
    runtime_step = normalize_runtime_rotation_step(
        arguments.post_rotation_step,
        arguments.vector_capacity,
    )
    rotation_steps = resource.get("rotation_steps")
    if not isinstance(rotation_steps, list) or runtime_step not in rotation_steps:
        raise SystemExit(
            "post-operation rotation step is absent from the compiler-emitted "
            "resource manifest"
        )


def reject_ambient_controls() -> None:
    unexpected = [name for name in CONTROL_ENVIRONMENT if name in os.environ]
    if unexpected:
        raise SystemExit(
            "ambient bootstrap controls are forbidden: " + ", ".join(unexpected)
        )


def typed_options(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "ciphertext_constant_encoding": arguments.ciphertext_constant_encoding,
        "decode_transform_budget": arguments.decode_transform_budget,
        "encode_transform_budget": arguments.encode_transform_budget,
        "first_prime_bits": arguments.first_prime_bits,
        "hamming_weight": arguments.hamming_weight,
        "input_level": arguments.input_level,
        "mul_level": arguments.mul_level,
        "packing": arguments.packing,
        "post_multiply_imag": arguments.post_multiply_imag,
        "post_multiply_real": arguments.post_multiply_real,
        "post_multiply_scale_degree": arguments.post_multiply_scale_degree,
        "post_rotation_step": arguments.post_rotation_step,
        "poly_degree": arguments.poly_degree,
        "q_part_count": arguments.q_part_count,
        "scaling_factor_bits": arguments.scaling_factor_bits,
        "security_level": arguments.security_level,
        "vector_capacity": arguments.vector_capacity,
    }


def normalized_argv(arguments: argparse.Namespace) -> list[str]:
    return [
        GENERATOR_TOOL,
        "--poly-degree", str(arguments.poly_degree),
        "--vector-capacity", str(arguments.vector_capacity),
        "--mul-level", str(arguments.mul_level),
        "--input-level", str(arguments.input_level),
        "--security-level", str(arguments.security_level),
        "--scaling-factor-bits", str(arguments.scaling_factor_bits),
        "--first-prime-bits", str(arguments.first_prime_bits),
        "--hamming-weight", str(arguments.hamming_weight),
        "--q-part-count", str(arguments.q_part_count),
        "--encode-transform-budget", str(arguments.encode_transform_budget),
        "--decode-transform-budget", str(arguments.decode_transform_budget),
        "--ciphertext-constant-encoding",
        arguments.ciphertext_constant_encoding,
        "--packing", arguments.packing,
        "--post-multiply-real", repr(arguments.post_multiply_real),
        "--post-multiply-imag", repr(arguments.post_multiply_imag),
        "--post-multiply-scale-degree", str(arguments.post_multiply_scale_degree),
        "--post-rotation-step", str(arguments.post_rotation_step),
    ]


def build_invocation_record(arguments: argparse.Namespace) -> dict[str, Any]:
    argv = normalized_argv(arguments)
    return {
        "schema_version": INVOCATION_SCHEMA,
        "status": "pass",
        "tool": GENERATOR_TOOL,
        "normalized_argv": argv,
        "normalized_argv_sha256": sha256_bytes(canonical_json_bytes(argv)),
        "options": typed_options(arguments),
        "output_destination_in_identity": False,
    }


def build_config(arguments: argparse.Namespace):
    try:
        return build_bootstrap_trace_config(
            poly_degree=arguments.poly_degree,
            mul_level=arguments.mul_level,
            first_prime_bits=arguments.first_prime_bits,
            scaling_factor_bits=arguments.scaling_factor_bits,
            hamming_weight=arguments.hamming_weight,
            q_parts=arguments.q_part_count,
            enc_budget=arguments.encode_transform_budget,
            dec_budget=arguments.decode_transform_budget,
            ct_encode=arguments.ciphertext_constant_encoding == "enabled",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def trace_bootstrap(config) -> tuple[Any, str]:
    AceEDSL._get_dsl.cache_clear()
    shape = (config.poly_degree,)
    ciphertext = CkksCiphertext(shape=shape, name="input_ct")
    zero = CkksCiphertext(shape=shape, name="zero_ct")
    kernel_arguments = (
        [ciphertext, zero, 1.0]
        + list(G_COEFFICIENTS_UNIFORM_HW_192)
        + list(config.double_angle_scalars)
        + [config.post_scale]
    )
    with bootstrap_trace_configuration(config):
        bootstrap_full(*kernel_arguments)
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise SystemExit("bootstrap tracing returned no AIR module")
    raw_air = canonicalize_checkout_paths(module.dump(), REPO_ROOT)
    if not raw_air:
        raise SystemExit("bootstrap tracing returned empty raw AIR")
    return module, raw_air


def compile_terminal(
    arguments: argparse.Namespace,
    config,
    *,
    provider: str,
    codegen_ir: str,
    context_path: Path | None = None,
    resource_path: Path | None = None,
    constant_path: Path | None = None,
) -> tuple[str, str, list[str], str]:
    module, raw_air = trace_bootstrap(config)
    pipeline = AcePipeline(module).configure_fhe(
        poly_degree=arguments.poly_degree,
        mul_level=arguments.mul_level,
        input_level=arguments.input_level,
        security_level=arguments.security_level,
        scaling_factor_bits=arguments.scaling_factor_bits,
        first_prime_bits=arguments.first_prime_bits,
        hamming_weight=arguments.hamming_weight,
        data_file="",
        ct_encode=arguments.ciphertext_constant_encoding == "enabled",
        provider=provider,
        codegen_ir=codegen_ir,
        context_manifest_file=str(context_path or ""),
        resource_manifest_file=str(resource_path or ""),
        constant_manifest_file=str(constant_path or ""),
    )
    pipeline.configure_vector_kernel_lowering(max_slots=arguments.vector_capacity)
    pipeline.set_ckks_extended_op_rewrite(False)
    result = pipeline.run(start_domain="fhe::ckks", dump_stages=True, verbose=False)
    if not result.success:
        raise SystemExit(f"{provider}/{codegen_ir} generation failed: {result.error}")
    expected_stages = (
        ["ckks_driver", "ckks2c"]
        if codegen_ir == "ckks"
        else ["ckks_driver", "poly_driver", "poly2c"]
    )
    if result.stages_completed != expected_stages:
        raise SystemExit(
            f"{provider}/{codegen_ir} selected unexpected stages: "
            f"{result.stages_completed}"
        )
    source = result.c_code or ""
    post_air = result.air_dumps.get("ckks_driver", "")
    if not source or not post_air:
        raise SystemExit(f"{provider}/{codegen_ir} returned empty artifacts")
    return (
        source,
        canonicalize_checkout_paths(post_air, REPO_ROOT),
        result.stages_completed,
        raw_air,
    )


def compile_post_operation_air(
    arguments: argparse.Namespace,
    *,
    bootstrap_output_attributes: dict[str, int],
) -> str:
    AceEDSL._get_dsl.cache_clear()
    ciphertext = CkksCiphertext(
        shape=(arguments.poly_degree,),
        name="bootstrap_output",
        level=bootstrap_output_attributes["level"],
    )
    multiplier = complex(
        arguments.post_multiply_real,
        arguments.post_multiply_imag,
    )
    config = build_config(arguments)
    rotation_step = arguments.post_rotation_step
    output_level = bootstrap_output_attributes["level"]
    output_rescale_level = bootstrap_output_attributes["rescale_level"]
    output_scale_degree = bootstrap_output_attributes["scale"]

    @ckks_kernel
    def bootstrap_post_operation_attestation(
        value: CkksCiphertext,
    ) -> CkksCiphertext:
        """Rotate then multiply without an implicit rescale."""
        value.value.set_u32_attr("level", output_level)
        value.value.set_u32_attr("rescale_level", output_rescale_level)
        value.value.set_u32_attr("scale", output_scale_degree)
        value.value.set_u32_attr("explicit_scale_coordinate", 1)
        multiplier_array = value.container.new_array_const(
            [float(multiplier.real), float(multiplier.imag)]
        )
        plain_node = value.container.new_ckks_encode_complex(
            multiplier_array,
            1,
            arguments.post_multiply_scale_degree,
            output_level,
            config.num_p,
            True,
        )
        plain_node.set_u32_attr("raw_scale_one", 1)
        plain = AIRValue(
            plain_node,
            value.container,
            domain=getattr(value, "domain", None),
        )
        rotated = value.rotate(rotation_step)
        multiply_node = rotated.container.new_ckks_mul(rotated.value, plain.value)
        if hasattr(multiply_node, "set_u32_attr"):
            multiply_node.set_u32_attr("skip_auto_rescale", 1)
        return AIRValue(
            multiply_node,
            rotated.container,
            domain=getattr(rotated, "domain", None),
        )

    bootstrap_post_operation_attestation(ciphertext)
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise SystemExit("post-operation tracing returned no AIR module")
    pipeline = AcePipeline(module).configure_fhe(
        poly_degree=arguments.poly_degree,
        mul_level=arguments.mul_level,
        input_level=bootstrap_output_attributes["level"],
        security_level=arguments.security_level,
        scaling_factor_bits=arguments.scaling_factor_bits,
        first_prime_bits=arguments.first_prime_bits,
        hamming_weight=arguments.hamming_weight,
        data_file="",
        ct_encode=arguments.ciphertext_constant_encoding == "enabled",
        provider="phantom",
        codegen_ir="ckks",
    )
    result = pipeline.run_ckks_driver()
    if not result.get("success", False):
        raise SystemExit(
            "post-operation CKKS analysis failed: "
            + str(result.get("message", "unknown error"))
        )
    post_air = canonicalize_checkout_paths(module.dump(), REPO_ROOT)
    if not post_air:
        raise SystemExit("post-operation CKKS analysis returned empty AIR")
    return post_air


def validate_context(context: dict[str, Any], arguments: argparse.Namespace) -> None:
    expected = {
        "packing": arguments.packing,
        "polynomial_degree": arguments.poly_degree,
        "logical_slot_capacity": arguments.vector_capacity,
        "input_level": arguments.input_level,
        "q_part_count": arguments.q_part_count,
        "hamming_weight": arguments.hamming_weight,
        "security_level": arguments.security_level,
        "first_modulus_bits": arguments.first_prime_bits,
        "scaling_modulus_bits": arguments.scaling_factor_bits,
    }
    mismatches = {
        name: {"expected": value, "observed": context.get(name)}
        for name, value in expected.items()
        if context.get(name) != value
    }
    expected_q = [arguments.first_prime_bits] + [
        arguments.scaling_factor_bits
    ] * (arguments.mul_level - 1)
    if context.get("data_q_bit_sizes") != expected_q:
        mismatches["data_q_bit_sizes"] = {
            "expected": expected_q,
            "observed": context.get("data_q_bit_sizes"),
        }
    if mismatches:
        raise SystemExit(
            "compiler context disagrees with explicit invocation: "
            + json.dumps(mismatches, sort_keys=True)
        )


def parse_air_attributes(value: str) -> dict[str, int]:
    attributes = {}
    for name in ("level", "rescale_level", "scale"):
        match = re.search(rf"(?:^|,){name}=(-?[0-9]+)(?:,|$)", value)
        if match is None:
            raise SystemExit(f"terminal post-CKKS AIR omits {name} metadata")
        attributes[name] = int(match.group(1))
    return attributes


def operation_attributes(post_air: str, opcode: str) -> list[dict[str, int]]:
    matches = re.findall(
        rf"CKKS\.{re.escape(opcode)} ATTR\[([^\]]+)\]",
        post_air,
    )
    return [parse_air_attributes(value) for value in matches]


def attest_post_operations(
    post_air: str,
    *,
    bootstrap_output: dict[str, int],
) -> dict[str, Any]:
    rotations = operation_attributes(post_air, "rotate")
    multiplies = operation_attributes(post_air, "mul")
    if len(rotations) != 1 or len(multiplies) != 1:
        raise SystemExit(
            "post-operation AIR must contain one rotation and one multiply"
        )
    rotation = rotations[0]
    multiply = multiplies[0]
    if rotation != bootstrap_output:
        raise SystemExit(
            "post-operation rotation metadata differs from bootstrap output"
        )
    if multiply != rotation:
        raise SystemExit(
            "raw-scale-1 ciphertext/plaintext multiply changed metadata"
        )
    return {
        "input_coordinate": {
            "ace_logical_level": bootstrap_output["level"],
            "rescale_level": bootstrap_output["rescale_level"],
            "scale_degree": rotation["scale"],
        },
        "rotation": {
            "air_attributes": rotation,
            "transition": {
                "ace_logical_level_delta": 0,
                "active_q_count_delta": 0,
                "phantom_chain_index_delta": 0,
                "scale_degree_delta": 0,
                "raw_scale_multiplier": 1.0,
                "logical_slots": "preserved",
                "ciphertext_size": "preserved",
                "ntt_state": "preserved",
            },
        },
        "ciphertext_plaintext_multiply": {
            "air_attributes": multiply,
            "transition": {
                "ace_logical_level_delta": 0,
                "active_q_count_delta": 0,
                "phantom_chain_index_delta": 0,
                "scale_degree_delta": multiply["scale"] - rotation["scale"],
                "raw_scale_multiplier": 1.0,
                "logical_slots": "preserved",
                "ciphertext_size": "preserved",
                "ntt_state": "preserved",
            },
        },
    }


SELF_ADD = re.compile(
    r'^\s*ld "(?P<source>[^"]+)"[^\n]*\n'
    r'\s*ld "(?P=source)"[^\n]*\n'
    r'\s*CKKS\.add ATTR\[(?P<operation_attributes>[^\]]+)\][^\n]*\n'
    r'\s*st "(?P<destination>[^"]+)"[^\n]*ATTR\[(?P<store_attributes>[^\]]+)\]',
    re.MULTILINE,
)
RETURN_STORE = re.compile(
    r'^\s*ld "(?P<source>[^"]+)"[^\n]*\n'
    r'\s*st "(?P<return_name>__ret_tmp_[^"]+)"[^\n]*'
    r'ATTR\[(?P<attributes>[^\]]+)\][^\n]*\n'
    r'\s*ld "(?P=return_name)"[^\n]*\n\s*retv\b',
    re.MULTILINE,
)


def attest_terminal_restoration(post_air: str, config) -> dict[str, Any]:
    return_matches = list(RETURN_STORE.finditer(post_air))
    if len(return_matches) != 1:
        raise SystemExit("post-CKKS AIR must contain exactly one terminal return store")
    returned = return_matches[0]
    output_attributes = parse_air_attributes(returned.group("attributes"))
    self_adds: dict[str, dict[str, Any]] = {}
    for match in SELF_ADD.finditer(post_air):
        self_adds[match.group("destination")] = {
            "source": match.group("source"),
            "destination": match.group("destination"),
            "operation_attributes": match.group("operation_attributes"),
            "store_attributes": match.group("store_attributes"),
        }
    chain_reversed = []
    cursor = returned.group("source")
    while cursor in self_adds:
        link = self_adds[cursor]
        operation_attributes = parse_air_attributes(
            link["operation_attributes"]
        )
        store_attributes = parse_air_attributes(link["store_attributes"])
        if operation_attributes != store_attributes:
            raise SystemExit("post-CKKS AIR self-add metadata changes at its store")
        if operation_attributes != output_attributes:
            raise SystemExit("terminal restoration self-add changes output metadata")
        chain_reversed.append(link)
        cursor = link["source"]
    chain = list(reversed(chain_reversed))
    expected_count = config.post_scale_degree
    if len(chain) != expected_count:
        raise SystemExit(
            "terminal restoration self-add count disagrees with expanded post scale: "
            f"expected {expected_count}, observed {len(chain)}"
        )
    restored_factor = 2 ** len(chain)
    if not math.isclose(restored_factor, config.post_scale, rel_tol=0.0, abs_tol=0.0):
        raise SystemExit("terminal restoration factor disagrees with expanded semantics")
    return {
        "output_attributes": output_attributes,
        "restoration": {
            "kind": "terminal-ciphertext-self-add-chain",
            "self_add_count": len(chain),
            "self_add_chain": [
                {"source": link["source"], "destination": link["destination"]}
                for link in chain
            ],
            "restored_factor": restored_factor,
            "expanded_post_scale_matches": True,
        },
    }


def float_sequence_sha256(values: Sequence[float]) -> str:
    return sha256_bytes(b"".join(struct.pack("<d", float(value)) for value in values))


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_path(path), "bytes": path.stat().st_size}


def build_semantics_record(
    arguments: argparse.Namespace,
    config,
    invocation_path: Path,
    raw_air_path: Path,
    post_air_path: Path,
    post_operations_air_path: Path,
    post_operations_attestation_path: Path,
    context_path: Path,
    resource_path: Path,
    constant_path: Path,
    phantom_source_path: Path,
    ant_source_path: Path,
    context: dict[str, Any],
    air_attestation: dict[str, Any],
    post_operation_attestation: dict[str, Any],
) -> dict[str, Any]:
    attributes = air_attestation["output_attributes"]
    restoration = dict(air_attestation["restoration"])
    restoration["air_sha256"] = sha256_path(post_air_path)
    restored_factor = restoration["restored_factor"]
    scale_degree = attributes["scale"]
    raw_scale = math.ldexp(1.0, scale_degree * arguments.scaling_factor_bits)
    data_q_count = len(context["data_q_bit_sizes"])
    ace_level = attributes["level"]
    coefficient_hash = float_sequence_sha256(config.chebyshev_coefficients)
    post_operation_contracts = json.loads(json.dumps(post_operation_attestation))
    post_operation_contracts["status"] = "pass"
    post_operation_contracts["air_sha256"] = sha256_path(post_operations_air_path)
    bindings = {
        "compiler_invocation_sha256": sha256_path(invocation_path),
        "raw_air_sha256": sha256_path(raw_air_path),
        "post_ckks_air_sha256": sha256_path(post_air_path),
        "post_operations_air_sha256": sha256_path(post_operations_air_path),
        "post_operations_attestation_sha256": sha256_path(
            post_operations_attestation_path
        ),
        "context_manifest_sha256": sha256_path(context_path),
        "resource_manifest_sha256": sha256_path(resource_path),
        "constant_manifest_sha256": sha256_path(constant_path),
        "phantom_source_sha256": sha256_path(phantom_source_path),
        "generated_dsl_ant_source_sha256": sha256_path(ant_source_path),
    }
    domain_bindings = {
        key: value
        for key, value in bindings.items()
        if key
        not in {
            "post_operations_air_sha256",
            "post_operations_attestation_sha256",
        }
    }
    evalmod_scalar_manifest = build_bootstrap_evalmod_scalar_manifest(config)
    supported_identity_domain, identity_domain_attestation = (
        derive_supported_identity_domain(
            coefficients=config.chebyshev_coefficients,
            scalars=config.double_angle_scalars,
            overflow_bound=config.eval_sin_upper_bound_k,
            restoration_factor=float(restored_factor),
            evalmod_lower=EVALMOD_COMPONENT_LOWER_BOUND,
            evalmod_upper=EVALMOD_COMPONENT_UPPER_BOUND,
            provider_clear_threshold=arguments.identity_error_threshold,
            artifact_bindings=domain_bindings,
            polynomial_degree=arguments.poly_degree,
            logical_slots=arguments.vector_capacity,
            transform_payload_manifest=(
                build_bootstrap_transform_payload_manifest(config)
            ),
            constant_manifest=json.loads(
                constant_path.read_text(encoding="utf-8")
            ),
            evalmod_scalar_manifest=evalmod_scalar_manifest,
            raw_air=raw_air_path.read_text(encoding="utf-8"),
        )
    )
    return {
        "schema_version": SEMANTICS_SCHEMA,
        "status": "pass",
        "bindings": bindings,
        "expanded_bootstrap": {
            "packing": arguments.packing,
            "logical_slot_capacity": arguments.vector_capacity,
            "transform_budgets": {
                "encode": config.enc_budget,
                "decode": config.dec_budget,
            },
            "ciphertext_constant_encoding": arguments.ciphertext_constant_encoding,
            "coefficient_family": {
                "kind": "ant-uniform-hamming-weight-at-most-threshold",
                "hamming_weight_max": UNIFORM_COEFFICIENT_HAMMING_WEIGHT_MAX,
                "selected_hamming_weight": config.hamming_weight,
                "coefficient_count": len(config.chebyshev_coefficients),
                "coefficient_payload_sha256": coefficient_hash,
                "evalmod_component_interval": {
                    "lower": EVALMOD_COMPONENT_LOWER_BOUND,
                    "upper": EVALMOD_COMPONENT_UPPER_BOUND,
                    "lower_inclusive": True,
                    "upper_inclusive": True,
                },
            },
            "eval_sin_upper_bound_k": config.eval_sin_upper_bound_k,
            "double_angle": {
                "count": len(config.double_angle_scalars),
                "scalar_payload_sha256": float_sequence_sha256(
                    config.double_angle_scalars
                ),
            },
            "evalmod_scalar_encodings": (
                evalmod_scalar_manifest
            ),
            "post_scale": {
                "degree": config.post_scale_degree,
                "factor": config.post_scale,
            },
            "coeffs_to_slots_factor": config.coeffs_to_slots_factor,
        },
        "output_air_contract": {
            "ace_logical_level": ace_level,
            "rescale_level": attributes["rescale_level"],
            "scale_degree": scale_degree,
            "active_q_count": ace_level,
            "phantom_chain_index": 1 + data_q_count - ace_level,
            "raw_scale_contract": {
                "kind": "ace-log2-scale-coordinate",
                "nominal_raw_scale": raw_scale.hex(),
                "scaling_modulus_bits": arguments.scaling_factor_bits,
                "expected_scale_degree": scale_degree,
                "maximum_absolute_coordinate_error": (
                    RAW_SCALE_COORDINATE_TOLERANCE
                ),
            },
            "logical_slots": arguments.vector_capacity,
            "ciphertext_size": 2,
            "ntt_state": True,
            "derivation": "terminal-post-ckks-air-and-compiler-context",
        },
        "restoration_attestation": restoration,
        "supported_identity_domain": supported_identity_domain,
        "identity_domain_attestation": identity_domain_attestation,
        "post_operation_contracts": post_operation_contracts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    validate_arguments(arguments)
    reject_ambient_controls()
    if arguments.output_dir.exists():
        raise SystemExit("output_dir must not already exist")
    config = build_config(arguments)

    arguments.output_dir.mkdir(parents=True)
    phantom_source_path = arguments.output_dir / "bootstrap_qualification.cu"
    ant_source_path = arguments.output_dir / "bootstrap_qualification_ant.cxx"
    raw_air_path = arguments.output_dir / "bootstrap_raw.air"
    post_air_path = arguments.output_dir / "bootstrap_post_ckks.air"
    post_operations_air_path = arguments.output_dir / "bootstrap_post_operations.air"
    post_operations_attestation_path = (
        arguments.output_dir / "post_operations_attestation.json"
    )
    context_path = arguments.output_dir / "compiler_context_manifest.json"
    resource_path = arguments.output_dir / "compiler_resource_manifest.json"
    constant_path = arguments.output_dir / "compiler_constant_manifest.json"
    invocation_path = arguments.output_dir / "compiler_invocation.json"
    semantics_path = arguments.output_dir / "bootstrap_semantics.json"
    record_path = arguments.output_dir / "generation.json"

    phantom_source, phantom_post_air, phantom_stages, phantom_raw_air = compile_terminal(
        arguments,
        config,
        provider="phantom",
        codegen_ir="ckks",
        context_path=context_path,
        resource_path=resource_path,
        constant_path=constant_path,
    )
    ant_source, ant_post_air, ant_stages, ant_raw_air = compile_terminal(
        arguments,
        config,
        provider="ant",
        codegen_ir="poly",
    )
    if phantom_raw_air != ant_raw_air:
        raise SystemExit("terminal paths were traced from different raw AIR")
    if phantom_post_air != ant_post_air:
        raise SystemExit("terminal paths received different post-CKKS AIR")

    raw_air_path.write_text(phantom_raw_air, encoding="utf-8")
    post_air_path.write_text(phantom_post_air, encoding="utf-8")
    phantom_source_path.write_text(phantom_source, encoding="utf-8")
    ant_source_path.write_text(ant_source, encoding="utf-8")

    manifests = {}
    for name, path in (
        ("context", context_path),
        ("resource", resource_path),
        ("constant", constant_path),
    ):
        if not path.is_file() or path.read_bytes().endswith(b"\n"):
            raise SystemExit(
                f"{name} manifest must use the compiler's canonical no-newline serialization"
            )
        manifests[name] = file_record(path)

    context = json.loads(context_path.read_text(encoding="utf-8"))
    validate_context(context, arguments)
    constant_manifest = json.loads(constant_path.read_text(encoding="utf-8"))
    constants = constant_manifest.get("constants")
    if not isinstance(constants, list) or not constants:
        raise SystemExit("full bootstrap emitted no cached complex constants")
    if constant_manifest.get("context_manifest_sha256") != sha256_path(context_path):
        raise SystemExit("constant manifest is not bound to the emitted context")
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
    validate_post_rotation_resource(resource, arguments)

    invocation = build_invocation_record(arguments)
    write_json(invocation_path, invocation)
    air_attestation = attest_terminal_restoration(phantom_post_air, config)
    post_operations_air = compile_post_operation_air(
        arguments,
        bootstrap_output_attributes=air_attestation["output_attributes"],
    )
    post_operations_air_path.write_text(post_operations_air, encoding="utf-8")
    post_operation_attestation = attest_post_operations(
        post_operations_air,
        bootstrap_output=air_attestation["output_attributes"],
    )
    post_operation_record = {
        "schema_version": POST_OPERATIONS_SCHEMA,
        "status": "attested",
        "bindings": {
            "compiler_invocation_sha256": sha256_path(invocation_path),
            "post_ckks_air_sha256": sha256_path(post_air_path),
            "context_manifest_sha256": sha256_path(context_path),
            "resource_manifest_sha256": sha256_path(resource_path),
            "post_operations_air_sha256": sha256_path(post_operations_air_path),
        },
        "inputs": {
            "multiply_constant": {
                "real": arguments.post_multiply_real,
                "imaginary": arguments.post_multiply_imag,
                "plaintext_scale_degree": arguments.post_multiply_scale_degree,
            },
            "rotation_step": arguments.post_rotation_step,
        },
        "input_coordinate": post_operation_attestation["input_coordinate"],
        "rotation": post_operation_attestation["rotation"],
        "ciphertext_plaintext_multiply": post_operation_attestation[
            "ciphertext_plaintext_multiply"
        ],
    }
    write_json(post_operations_attestation_path, post_operation_record)
    semantics = build_semantics_record(
        arguments,
        config,
        invocation_path,
        raw_air_path,
        post_air_path,
        post_operations_air_path,
        post_operations_attestation_path,
        context_path,
        resource_path,
        constant_path,
        phantom_source_path,
        ant_source_path,
        context,
        air_attestation,
        post_operation_attestation,
    )
    write_json(semantics_path, semantics)

    record = {
        "schema_version": GENERATION_SCHEMA,
        "status": "pass",
        "qualification_scope": "full-generated-bootstrap-dual-terminal-generation",
        "compiler_parameters": typed_options(arguments),
        "bootstrap_parameters": {
            "q_parts": config.q_parts,
            "enc_budget": config.enc_budget,
            "dec_budget": config.dec_budget,
            "ct_encode": config.ct_encode,
        },
        "identity_domain_policy": {
            "provider_clear_maximum_absolute": (
                arguments.identity_error_threshold
            ),
            "clear_map_budget_fraction": CLEAR_MAP_BUDGET_FRACTION,
        },
        "normalized_compiler_invocation": file_record(invocation_path),
        "bootstrap_semantics": file_record(semantics_path),
        "post_operations_air": file_record(post_operations_air_path),
        "post_operations_attestation": file_record(
            post_operations_attestation_path
        ),
        "stages_completed": phantom_stages,
        "terminal_paths": {
            "phantom": {
                "provider": "phantom",
                "codegen_ir": "ckks",
                "stages_completed": phantom_stages,
                "post_ckks_air_sha256": sha256_path(post_air_path),
            },
            "generated_dsl_ant": {
                "provider": "ant",
                "codegen_ir": "poly",
                "stages_completed": ant_stages,
                "post_ckks_air_sha256": sha256_path(post_air_path),
            },
        },
        "source": file_record(phantom_source_path),
        "sources": {
            "phantom": file_record(phantom_source_path),
            "generated_dsl_ant": file_record(ant_source_path),
        },
        "air": {
            "raw": file_record(raw_air_path),
            "post_ckks": file_record(post_air_path),
        },
        "manifests": manifests,
        "constant_count": len(constants),
        "rotation_count": len(resource["rotation_steps"]),
        "rotation_batch_count": len(resource["rotation_batches"]),
        "monomial_count": len(resource["monomial_powers"]),
        "native_bootstrap_precompute": False,
        "generated_program_executed": False,
    }
    write_json(record_path, record)
    print(
        f"generated {phantom_source_path} and {ant_source_path} "
        f"with {len(constants)} cached constants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
