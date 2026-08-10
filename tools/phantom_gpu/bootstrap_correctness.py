#!/usr/bin/env python3
"""Freeze and compare provider-neutral generated-bootstrap correctness data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Sequence

from bootstrap_domain_attestation import (
    DOMAIN_ATTESTATION_SCHEMA,
    DOMAIN_ARTIFACT_BINDING_KEYS,
    validate_supported_identity_domain,
)

FIXTURE_SCHEMA = "ace.phantom.bootstrap-correctness-fixture/2.0.0"
INVOCATION_SCHEMA = "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0"
SEMANTICS_SCHEMA = "ace.phantom.generated-bootstrap.semantics/2.0.0"
POST_OPERATION_SCHEMA = "ace.phantom.bootstrap-post-operation-semantics/1.0.0"
SANITIZER_SCHEMA = "ace.phantom.bootstrap-compute-sanitizer/1.0.0"
COMPARISON_SCHEMA = "ace.phantom.bootstrap-three-way-comparison/1.0.0"
PROVIDER_SCHEMAS = {
    "native-ant": "ace.phantom.bootstrap-native-ant/1.0.0",
    "generated-ant": "ace.phantom.bootstrap-generated-ant/1.0.0",
    "generated-phantom": "ace.phantom.bootstrap-generated-phantom/1.0.0",
}
BINARY_FORMAT = "ace.bootstrap-correctness.complex_float64le/1.0.0"
MAGIC = b"ACEBSC01"
HEADER = struct.Struct("<8sHHIQQ32s32s32s")
PAIR = struct.Struct("<dd")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MASK64 = (1 << 64) - 1
CASE_KINDS = (
    "bounded_random_complex",
    "zero",
    "real_constant",
    "complex_constant",
    "alternating_real_imaginary_signs",
    "positive_boundary_inside",
    "negative_boundary_inside",
)
BINDING_INPUTS = (
    "ace_source_manifest",
    "phantom_source_manifest",
    "compiler_invocation",
    "raw_air",
    "post_ckks_air",
    "context_manifest",
    "resource_manifest",
    "constant_manifest",
    "bootstrap_semantics",
    "post_operations_air",
    "post_operation_attestation",
)
GENERATOR_TOOL = "tools/phantom_gpu/generate_bootstrap_qualification.py"
RAW_SCALE_COORDINATE_TOLERANCE = 1.0e-4
INVOCATION_OPTION_KEYS = {
    "ciphertext_constant_encoding",
    "decode_transform_budget",
    "encode_transform_budget",
    "first_prime_bits",
    "hamming_weight",
    "input_level",
    "mul_level",
    "packing",
    "post_multiply_imag",
    "post_multiply_real",
    "post_multiply_scale_degree",
    "post_rotation_step",
    "poly_degree",
    "q_part_count",
    "scaling_factor_bits",
    "security_level",
    "vector_capacity",
}


class CorrectnessError(ValueError):
    pass


def fail(message: str) -> None:
    raise CorrectnessError(message)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
            parse_constant=lambda value: fail(f"non-finite JSON number: {value}"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"top-level JSON value in {path} must be an object")
    return value


def canonical_identity_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest(path: Path) -> str:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as error:
        fail(f"cannot hash {path}: {error}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_keys(value: Any, keys: Iterable[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    expected = set(keys)
    if set(value) != expected:
        fail(f"{context} keys differ: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}")
    return value


def finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{context} must be a finite number")
    return float(value)


def positive(value: Any, context: str) -> float:
    result = finite(value, context)
    if result <= 0.0:
        fail(f"{context} must be positive")
    return result


def uint(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{context} must be an integer >= {minimum}")
    return value


def sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        fail(f"{context} must be lowercase SHA-256")
    return value


def write_fresh(path: Path, contents: bytes) -> None:
    if path.exists():
        fail(f"refusing to replace existing output {path}")
    try:
        with path.open("xb") as stream:
            stream.write(contents)
            stream.flush()
    except OSError as error:
        fail(f"cannot create {path}: {error}")


def bindings_from_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {f"{name}_sha256": digest(paths[name]) for name in BINDING_INPUTS}


def _find_binding(container: dict[str, Any], key: str) -> Any:
    bindings = container.get("bindings")
    if not isinstance(bindings, dict) or key not in bindings:
        fail(f"bootstrap semantics does not bind {key}")
    return bindings[key]


def validate_authorities(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    invocation = load_json(paths["compiler_invocation"])
    require_keys(
        invocation,
        {
            "schema_version", "status", "tool", "normalized_argv",
            "normalized_argv_sha256", "options", "output_destination_in_identity",
        },
        "compiler invocation",
    )
    if (
        invocation["schema_version"] != INVOCATION_SCHEMA
        or invocation["status"] != "pass"
        or invocation["tool"] != GENERATOR_TOOL
        or invocation["output_destination_in_identity"] is not False
    ):
        fail("unsupported compiler invocation schema")
    options = require_keys(
        invocation["options"], INVOCATION_OPTION_KEYS, "compiler invocation.options"
    )
    integer_options = INVOCATION_OPTION_KEYS - {
        "ciphertext_constant_encoding", "packing", "post_multiply_imag",
        "post_multiply_real",
    }
    for name in integer_options:
        if isinstance(options[name], bool) or not isinstance(options[name], int):
            fail(f"compiler invocation option {name} must be an integer")
    for name in (
        "decode_transform_budget", "encode_transform_budget", "first_prime_bits",
        "hamming_weight", "input_level", "mul_level", "poly_degree",
        "q_part_count", "scaling_factor_bits", "vector_capacity",
    ):
        uint(options[name], f"compiler invocation option {name}", 1)
    for name in ("post_multiply_real", "post_multiply_imag"):
        if isinstance(options[name], bool) or not isinstance(options[name], float):
            fail(f"compiler invocation option {name} must be a float")
        finite(options[name], f"compiler invocation option {name}")
    if not isinstance(options["packing"], str) or not options["packing"]:
        fail("compiler invocation packing must be a nonempty string")
    if options["ciphertext_constant_encoding"] not in {"enabled", "disabled"}:
        fail("compiler invocation ciphertext constant encoding is invalid")
    normalized_argv = [
        GENERATOR_TOOL,
        "--poly-degree", str(options["poly_degree"]),
        "--vector-capacity", str(options["vector_capacity"]),
        "--mul-level", str(options["mul_level"]),
        "--input-level", str(options["input_level"]),
        "--security-level", str(options["security_level"]),
        "--scaling-factor-bits", str(options["scaling_factor_bits"]),
        "--first-prime-bits", str(options["first_prime_bits"]),
        "--hamming-weight", str(options["hamming_weight"]),
        "--q-part-count", str(options["q_part_count"]),
        "--encode-transform-budget", str(options["encode_transform_budget"]),
        "--decode-transform-budget", str(options["decode_transform_budget"]),
        "--ciphertext-constant-encoding", options["ciphertext_constant_encoding"],
        "--packing", options["packing"],
        "--post-multiply-real", repr(options["post_multiply_real"]),
        "--post-multiply-imag", repr(options["post_multiply_imag"]),
        "--post-multiply-scale-degree", str(options["post_multiply_scale_degree"]),
        "--post-rotation-step", str(options["post_rotation_step"]),
    ]
    if invocation["normalized_argv"] != normalized_argv:
        fail("compiler invocation normalized argv differs from typed options")
    if invocation["normalized_argv_sha256"] != sha256_bytes(
        canonical_identity_bytes(normalized_argv)
    ):
        fail("compiler invocation normalized argv hash differs")

    semantics = load_json(paths["bootstrap_semantics"])
    if (
        semantics.get("schema_version") != SEMANTICS_SCHEMA
        or semantics.get("status") != "pass"
    ):
        fail("unsupported bootstrap semantics schema")
    domain = semantics.get("supported_identity_domain")
    require_keys(domain, {"kind", "components", "lower_exclusive", "upper_exclusive", "period", "evidence"}, "supported_identity_domain")
    lower = finite(domain["lower_exclusive"], "supported_identity_domain.lower_exclusive")
    upper = finite(domain["upper_exclusive"], "supported_identity_domain.upper_exclusive")
    if not lower < 0.0 < upper:
        fail("supported identity domain must straddle zero")
    if not isinstance(semantics.get("output_air_contract"), dict):
        fail("bootstrap semantics must contain output_air_contract")
    declared_post_operations = semantics.get("post_operation_contracts")
    if not isinstance(declared_post_operations, dict) or declared_post_operations.get("status") != "pass":
        fail("bootstrap semantics lacks a passing dedicated post-operation contract")
    post_attestation = load_json(paths["post_operation_attestation"])
    if post_attestation.get("schema_version") != POST_OPERATION_SCHEMA or post_attestation.get("status") != "attested":
        fail("unsupported or incomplete post-operation attestation")
    require_keys(post_attestation, {"schema_version", "status", "bindings", "input_coordinate", "rotation", "ciphertext_plaintext_multiply", "inputs"}, "post-operation attestation")

    context = load_json(paths["context_manifest"])
    for key in ("logical_slot_capacity", "input_level"):
        uint(context.get(key), f"context manifest.{key}", 1)
    expected_context = {
        "packing": options["packing"],
        "polynomial_degree": options["poly_degree"],
        "logical_slot_capacity": options["vector_capacity"],
        "input_level": options["input_level"],
        "q_part_count": options["q_part_count"],
        "hamming_weight": options["hamming_weight"],
        "security_level": options["security_level"],
        "first_modulus_bits": options["first_prime_bits"],
        "scaling_modulus_bits": options["scaling_factor_bits"],
    }
    for name, expected in expected_context.items():
        if context.get(name) != expected:
            fail(f"context manifest {name} differs from compiler invocation")
    output_contract = semantics["output_air_contract"]
    raw_scale_contract = require_keys(
        output_contract.get("raw_scale_contract"),
        {
            "kind", "nominal_raw_scale", "scaling_modulus_bits",
            "expected_scale_degree", "maximum_absolute_coordinate_error",
        },
        "output_air_contract.raw_scale_contract",
    )
    try:
        nominal_raw_scale = float.fromhex(raw_scale_contract["nominal_raw_scale"])
    except (TypeError, ValueError) as error:
        raise CorrectnessError("nominal raw scale is not hexadecimal") from error
    expected_scale_degree = uint(
        output_contract.get("scale_degree"), "output AIR scale degree"
    )
    if (
        raw_scale_contract["kind"] != "ace-log2-scale-coordinate"
        or raw_scale_contract["scaling_modulus_bits"]
        != context["scaling_modulus_bits"]
        or raw_scale_contract["expected_scale_degree"] != expected_scale_degree
        or raw_scale_contract["maximum_absolute_coordinate_error"]
        != RAW_SCALE_COORDINATE_TOLERANCE
        or nominal_raw_scale
        != math.ldexp(
            1.0,
            expected_scale_degree * context["scaling_modulus_bits"],
        )
    ):
        fail("raw scale contract differs from the compiler scale coordinate")
    data_q = context.get("data_q_bit_sizes")
    if not isinstance(data_q, list) or len(data_q) != options["mul_level"]:
        fail("context manifest data-Q count differs from compiler invocation")
    resources = load_json(paths["resource_manifest"])
    rotation_steps = resources.get("rotation_steps")
    if (
        not isinstance(rotation_steps, list)
        or options["post_rotation_step"] not in rotation_steps
    ):
        fail("compiler invocation rotation step is absent from resource manifest")
    if options["post_multiply_scale_degree"] != 0:
        fail("compiler invocation post multiply scale degree must be zero")
    if options["post_rotation_step"] % options["vector_capacity"] == 0:
        fail("compiler invocation rotation step must be nonzero modulo capacity")
    normalized_rotation = options["post_rotation_step"] % options["vector_capacity"]
    if normalized_rotation > options["vector_capacity"] // 2:
        normalized_rotation -= options["vector_capacity"]
    if options["post_rotation_step"] != normalized_rotation:
        fail("compiler invocation rotation step is not canonical signed")
    if options["post_multiply_real"] == 0.0 and options["post_multiply_imag"] == 0.0:
        fail("compiler invocation post multiply constant must be nonzero")
    hashes = bindings_from_paths(paths)
    for name in ("compiler_invocation", "raw_air", "post_ckks_air", "context_manifest", "resource_manifest", "constant_manifest"):
        key = f"{name}_sha256"
        if _find_binding(semantics, key) != hashes[key]:
            fail(f"bootstrap semantics {key} disagrees with supplied artifact")
    semantics_bindings = semantics.get("bindings")
    if not isinstance(semantics_bindings, dict):
        fail("bootstrap semantics bindings must be an object")
    try:
        domain_bindings = {
            key: semantics_bindings[key]
            for key in DOMAIN_ARTIFACT_BINDING_KEYS
        }
        validate_supported_identity_domain(
            domain,
            semantics.get("identity_domain_attestation"),
            provider_clear_threshold=1.0e-2,
            artifact_bindings=domain_bindings,
            constant_manifest=load_json(paths["constant_manifest"]),
            evalmod_scalar_manifest=semantics["expanded_bootstrap"][
                "evalmod_scalar_encodings"
            ],
            raw_air=paths["raw_air"].read_text(encoding="utf-8"),
        )
    except (KeyError, OSError, UnicodeError, ValueError) as error:
        fail(f"bootstrap identity-domain attestation is invalid: {error}")
    attestation_bindings = require_keys(post_attestation["bindings"], {"compiler_invocation_sha256", "post_ckks_air_sha256", "context_manifest_sha256", "resource_manifest_sha256", "post_operations_air_sha256"}, "post-operation attestation.bindings")
    for key, value in attestation_bindings.items():
        if value != hashes[key]:
            fail(f"post-operation attestation {key} disagrees with supplied artifact")
    if semantics["bindings"].get("post_operations_attestation_sha256") != hashes["post_operation_attestation_sha256"]:
        fail("bootstrap semantics binds a different post-operation attestation")
    expected_contracts = {key: post_attestation[key] for key in ("input_coordinate", "rotation", "ciphertext_plaintext_multiply")}
    expected_contracts["status"] = "pass"
    expected_contracts["air_sha256"] = hashes["post_operations_air_sha256"]
    if declared_post_operations != expected_contracts:
        fail("bootstrap semantics and post-operation attestation contracts differ")
    post_inputs = require_keys(post_attestation["inputs"], {"multiply_constant", "rotation_step"}, "post-operation attestation.inputs")
    multiply_constant = require_keys(post_inputs["multiply_constant"], {"real", "imaginary", "plaintext_scale_degree"}, "post-operation multiply constant")
    if (
        multiply_constant["real"] != options.get("post_multiply_real")
        or multiply_constant["imaginary"] != options.get("post_multiply_imag")
        or post_inputs["rotation_step"] != options.get("post_rotation_step")
        or multiply_constant["plaintext_scale_degree"] != options.get("post_multiply_scale_degree")
        or multiply_constant["plaintext_scale_degree"] != 0
    ):
        fail("post-operation attestation inputs differ from explicit compiler invocation")
    return invocation, semantics, context, hashes


def freeze_fixture(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = {name: Path(getattr(arguments, name)) for name in BINDING_INPUTS}
    _, semantics, _, bindings = validate_authorities(paths)
    post_attestation = load_json(paths["post_operation_attestation"])
    seed = uint(arguments.seed, "seed")
    if seed > MASK64:
        fail("seed exceeds uint64")
    domain = semantics["supported_identity_domain"]
    lower, upper = float(domain["lower_exclusive"]), float(domain["upper_exclusive"])
    margin = positive(arguments.inside_margin, "inside margin")
    if margin >= min(-lower, upper):
        fail("inside margin does not remain within the supported domain")
    if not (lower < lower + margin < upper and lower < upper - margin < upper):
        fail("inside margin is not representable strictly within the domain")
    thresholds = {
        "provider_clear_maximum_absolute": positive(arguments.provider_clear_threshold, "provider-clear threshold"),
        "gpu_native_maximum_absolute": positive(arguments.gpu_native_threshold, "GPU-native threshold"),
        "gpu_generated_maximum_absolute": positive(arguments.gpu_generated_threshold, "GPU-generated threshold"),
        "repeat_maximum_absolute": positive(arguments.repeat_threshold, "repeat threshold"),
    }
    if thresholds["provider_clear_maximum_absolute"] != 1e-2:
        fail("provider-clear threshold must be the frozen 1e-2 gate")
    scale = min(-lower, upper)
    case_recipes = [
        {"id": "bounded-random-complex", "kind": CASE_KINDS[0], "component_bound": scale * 0.125, "seed_xor": "0x42534352414e444f"},
        {"id": "zero", "kind": CASE_KINDS[1]},
        {"id": "nonzero-real-constant", "kind": CASE_KINDS[2], "value": [scale * 0.125, 0.0]},
        {"id": "nonzero-complex-constant", "kind": CASE_KINDS[3], "value": [scale * 0.0625, -scale * 0.09375]},
        {"id": "alternating-real-imaginary-signs", "kind": CASE_KINDS[4], "magnitude": scale * 0.125},
        {"id": "positive-boundary-inside", "kind": CASE_KINDS[5], "offset_from_attested_boundary": margin},
        {"id": "negative-boundary-inside", "kind": CASE_KINDS[6], "offset_from_attested_boundary": margin},
    ]
    fixture = {
        "schema_version": FIXTURE_SCHEMA,
        "fixture_id": arguments.fixture_id,
        "bindings": bindings,
        "seed": {"algorithm": "splitmix64-float53-v1", "value": str(seed)},
        "supported_identity_domain": {
            "authority": "bootstrap_semantics",
            "bootstrap_semantics_sha256": bindings["bootstrap_semantics_sha256"],
            "attested_domain": domain,
        },
        "case_recipes": case_recipes,
        "tolerances": thresholds,
        "post_operations": {
            "inputs": post_attestation["inputs"],
            "input_coordinate": post_attestation["input_coordinate"],
            "rotation": post_attestation["rotation"],
            "ciphertext_plaintext_multiply": post_attestation["ciphertext_plaintext_multiply"],
        },
        "repeat_contract": {"calls_per_case": 3, "independently_owned_clones": True},
    }
    validate_fixture(fixture)
    write_fresh(Path(arguments.output), canonical_bytes(fixture))
    return fixture


def validate_fixture(fixture: dict[str, Any]) -> None:
    require_keys(fixture, {"schema_version", "fixture_id", "bindings", "seed", "supported_identity_domain", "case_recipes", "tolerances", "post_operations", "repeat_contract"}, "fixture")
    if fixture["schema_version"] != FIXTURE_SCHEMA:
        fail("unsupported correctness fixture schema")
    if not isinstance(fixture["fixture_id"], str) or not fixture["fixture_id"]:
        fail("fixture_id must be nonempty")
    bindings = require_keys(fixture["bindings"], {f"{name}_sha256" for name in BINDING_INPUTS}, "fixture.bindings")
    for key, value in bindings.items():
        sha(value, f"fixture.bindings.{key}")
    seed = require_keys(fixture["seed"], {"algorithm", "value"}, "fixture.seed")
    if seed["algorithm"] != "splitmix64-float53-v1" or not isinstance(seed["value"], str) or not seed["value"].isdigit() or int(seed["value"]) > MASK64:
        fail("fixture seed contract is invalid")
    domain = require_keys(fixture["supported_identity_domain"], {"authority", "bootstrap_semantics_sha256", "attested_domain"}, "fixture.supported_identity_domain")
    if domain["authority"] != "bootstrap_semantics" or domain["bootstrap_semantics_sha256"] != bindings["bootstrap_semantics_sha256"]:
        fail("fixture identity domain is not bound to bootstrap semantics")
    attested = require_keys(domain["attested_domain"], {"kind", "components", "lower_exclusive", "upper_exclusive", "period", "evidence"}, "fixture.supported_identity_domain.attested_domain")
    if not finite(attested["lower_exclusive"], "domain.lower_exclusive") < 0.0 < finite(attested["upper_exclusive"], "domain.upper_exclusive"):
        fail("fixture identity domain is invalid")
    recipes = fixture["case_recipes"]
    if not isinstance(recipes, list) or [item.get("kind") if isinstance(item, dict) else None for item in recipes] != list(CASE_KINDS):
        fail("fixture case recipe order or coverage changed")
    ids = [item.get("id") for item in recipes]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        fail("fixture case identifiers are invalid")
    tolerances = require_keys(fixture["tolerances"], {"provider_clear_maximum_absolute", "gpu_native_maximum_absolute", "gpu_generated_maximum_absolute", "repeat_maximum_absolute"}, "fixture.tolerances")
    for key, value in tolerances.items():
        positive(value, f"fixture.tolerances.{key}")
    if tolerances["provider_clear_maximum_absolute"] != 1e-2:
        fail("fixture provider-clear threshold changed")
    evidence = require_keys(
        attested["evidence"],
        {
            "attestation_schema_version",
            "attestation_sha256",
            "provider_clear_maximum_absolute",
            "maximum_complex_clear_map_error",
            "reserved_provider_numerical_error",
        },
        "fixture identity-domain evidence",
    )
    sha(evidence["attestation_sha256"], "fixture identity-domain attestation")
    if (
        evidence["attestation_schema_version"] != DOMAIN_ATTESTATION_SCHEMA
        or evidence["provider_clear_maximum_absolute"]
        != tolerances["provider_clear_maximum_absolute"]
        or evidence["maximum_complex_clear_map_error"] != 0.005
        or evidence["reserved_provider_numerical_error"] != 0.005
    ):
        fail("fixture identity-domain error budget differs from frozen tolerances")
    if not isinstance(fixture["post_operations"], dict) or not fixture["post_operations"]:
        fail("fixture post_operations must be nonempty")
    if fixture["repeat_contract"] != {"calls_per_case": 3, "independently_owned_clones": True}:
        fail("fixture repeat contract changed")


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, value ^ (value >> 31)


def materialize_cases(fixture: dict[str, Any], slots: int) -> list[tuple[str, list[complex]]]:
    validate_fixture(fixture)
    uint(slots, "logical slots", 1)
    seed = int(fixture["seed"]["value"])
    attested_domain = fixture["supported_identity_domain"]["attested_domain"]
    lower = attested_domain["lower_exclusive"]
    upper = attested_domain["upper_exclusive"]
    result: list[tuple[str, list[complex]]] = []
    for recipe in fixture["case_recipes"]:
        kind = recipe["kind"]
        if kind == "bounded_random_complex":
            bound = positive(recipe["component_bound"], "random component bound")
            state = seed ^ int(recipe["seed_xor"], 16)
            values = []
            for _ in range(slots):
                state, left = splitmix64(state)
                state, right = splitmix64(state)
                real = (2.0 * ((left >> 11) / float(1 << 53)) - 1.0) * bound
                imag = (2.0 * ((right >> 11) / float(1 << 53)) - 1.0) * bound
                values.append(complex(real, imag))
        elif kind == "zero":
            values = [0j] * slots
        elif kind in ("real_constant", "complex_constant"):
            pair = recipe["value"]
            if not isinstance(pair, list) or len(pair) != 2:
                fail(f"{kind} value must be a complex pair")
            values = [complex(finite(pair[0], f"{kind}.real"), finite(pair[1], f"{kind}.imag"))] * slots
        elif kind == "alternating_real_imaginary_signs":
            magnitude = positive(recipe["magnitude"], "alternating magnitude")
            values = [complex(magnitude if index % 2 == 0 else -magnitude, -magnitude if index % 2 == 0 else magnitude) for index in range(slots)]
        elif kind == "positive_boundary_inside":
            values = [complex(upper - positive(recipe["offset_from_attested_boundary"], "positive boundary offset"), 0.0)] * slots
        elif kind == "negative_boundary_inside":
            values = [complex(lower + positive(recipe["offset_from_attested_boundary"], "negative boundary offset"), 0.0)] * slots
        else:
            fail(f"unsupported case recipe {kind}")
        for value in values:
            if not (lower < value.real < upper and lower < value.imag < upper):
                fail(f"case {recipe['id']} leaves the supported identity domain")
        result.append((recipe["id"], values))
    return result


def case_manifest(records: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes([{"case_id": item["case_id"], "oracle": item["oracle"], "value_count": len(item["values"])} for item in records]))


def write_value_file(path: Path, fixture_sha256: str, context_sha256: str, records: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sha(fixture_sha256, "fixture SHA-256")
    sha(context_sha256, "context SHA-256")
    payload = bytearray()
    descriptors = []
    identities: set[tuple[str, str]] = set()
    for position, item in enumerate(records):
        require_keys(item, {"case_id", "oracle", "values", "metadata"}, f"record[{position}]")
        identity = (item["case_id"], item["oracle"])
        if not all(isinstance(part, str) and part for part in identity) or identity in identities:
            fail("invalid or duplicate value record identity")
        identities.add(identity)
        values = item["values"]
        if not isinstance(values, list) or not values:
            fail("value record must be nonempty")
        offset = len(payload)
        for value in values:
            if not isinstance(value, complex) or not math.isfinite(value.real) or not math.isfinite(value.imag):
                fail("value record contains nonfinite or noncomplex data")
            payload.extend(PAIR.pack(value.real, value.imag))
        descriptors.append({"case_id": identity[0], "oracle": identity[1], "offset_bytes": offset, "value_count": len(values), "metadata": item["metadata"]})
    manifest = case_manifest(records)
    contents = HEADER.pack(MAGIC, 1, 0, 0, len(records), len(payload), bytes.fromhex(fixture_sha256), bytes.fromhex(context_sha256), bytes.fromhex(manifest)) + payload
    write_fresh(path, contents)
    return ({"format": BINARY_FORMAT, "sha256": sha256_bytes(contents), "size_bytes": len(contents), "header_size": HEADER.size, "record_count": len(records), "payload_bytes": len(payload), "case_manifest_sha256": manifest}, descriptors)


def read_value_file(path: Path, info: Any, descriptors: Any, fixture_sha256: str, context_sha256: str, oracle: str) -> list[tuple[str, list[complex], dict[str, Any]]]:
    require_keys(info, {"format", "sha256", "size_bytes", "header_size", "record_count", "payload_bytes", "case_manifest_sha256"}, "binary")
    if info["format"] != BINARY_FORMAT or info["header_size"] != HEADER.size:
        fail("unsupported provider binary format")
    try:
        contents = path.read_bytes()
    except OSError as error:
        fail(f"cannot read provider value file {path}: {error}")
    if len(contents) != info["size_bytes"] or sha256_bytes(contents) != info["sha256"]:
        fail("provider binary size or checksum mismatch")
    if len(contents) < HEADER.size:
        fail("provider binary is truncated")
    magic, major, minor, flags, count, payload_bytes, fixture_digest, context_digest, manifest_digest = HEADER.unpack_from(contents)
    if (magic, major, minor, flags) != (MAGIC, 1, 0, 0) or fixture_digest.hex() != fixture_sha256 or context_digest.hex() != context_sha256:
        fail("provider binary header or provenance mismatch")
    if not isinstance(descriptors, list) or count != len(descriptors) or count != info["record_count"] or HEADER.size + payload_bytes != len(contents):
        fail("provider binary record count or length mismatch")
    payload = memoryview(contents)[HEADER.size:]
    expected_offset = 0
    records = []
    manifest_records = []
    for position, item in enumerate(descriptors):
        require_keys(item, {"case_id", "oracle", "offset_bytes", "value_count", "metadata"}, f"records[{position}]")
        if item["oracle"] != oracle or item["offset_bytes"] != expected_offset or not isinstance(item["metadata"], dict):
            fail("provider binary descriptor order, oracle, offset, or metadata mismatch")
        count_values = uint(item["value_count"], "value_count", 1)
        end = expected_offset + count_values * PAIR.size
        if end > len(payload):
            fail("provider value descriptor exceeds payload")
        values = [complex(*PAIR.unpack_from(payload, expected_offset + index * PAIR.size)) for index in range(count_values)]
        if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for value in values):
            fail("provider values contain NaN or infinity")
        records.append((item["case_id"], values, item["metadata"]))
        manifest_records.append({"case_id": item["case_id"], "oracle": oracle, "values": values})
        expected_offset = end
    if expected_offset != len(payload) or case_manifest(manifest_records) != manifest_digest.hex() or manifest_digest.hex() != info["case_manifest_sha256"]:
        fail("provider binary payload or case manifest mismatch")
    return records


def metric(actual: Sequence[complex], reference: Sequence[complex], threshold: float) -> dict[str, Any]:
    if len(actual) != len(reference) or not actual:
        fail("comparison vectors have unequal or zero length")
    errors = [abs(left - right) for left, right in zip(actual, reference)]
    maximum = max(errors)
    index = errors.index(maximum)
    return {"status": "pass" if maximum <= threshold else "fail", "comparison_count": len(errors), "threshold": threshold, "maximum_absolute_error": maximum, "maximum_absolute_error_index": index, "mean_absolute_error": sum(errors) / len(errors), "root_mean_square_error": math.sqrt(sum(value * value for value in errors) / len(errors)), "estimated_precision_bits": None if maximum == 0.0 else -math.log2(maximum), "estimated_precision_is_infinite": maximum == 0.0}


def validate_recorded_metric(recorded: Any, expected: dict[str, Any], context: str) -> None:
    require_keys(recorded, expected.keys(), context)
    for key in ("status", "comparison_count", "threshold", "maximum_absolute_error_index", "estimated_precision_is_infinite"):
        if recorded[key] != expected[key]:
            fail(f"{context}.{key} differs from recomputed full-slot metric")
    for key in ("maximum_absolute_error", "mean_absolute_error", "root_mean_square_error"):
        observed = finite(recorded[key], f"{context}.{key}")
        if not math.isclose(observed, expected[key], rel_tol=1e-15, abs_tol=1e-15):
            fail(f"{context}.{key} differs from recomputed full-slot metric")
    precision = recorded["estimated_precision_bits"]
    if expected["estimated_precision_is_infinite"]:
        if precision is not None:
            fail(f"{context}.estimated_precision_bits must be null at zero error")
    elif not math.isclose(finite(precision, f"{context}.estimated_precision_bits"), expected["estimated_precision_bits"], rel_tol=1e-15, abs_tol=1e-15):
        fail(f"{context}.estimated_precision_bits differs from recomputed metric")


def validate_operation_metric(recorded: Any, threshold: float, slots: int, context: str) -> None:
    keys = {"status", "comparison_count", "threshold", "maximum_absolute_error", "maximum_absolute_error_index", "mean_absolute_error", "root_mean_square_error", "estimated_precision_bits", "estimated_precision_is_infinite"}
    require_keys(recorded, keys, context)
    if recorded["status"] != "pass" or recorded["comparison_count"] != slots or recorded["threshold"] != threshold:
        fail(f"{context} status, count, or threshold differs from its frozen contract")
    maximum = finite(recorded["maximum_absolute_error"], f"{context}.maximum_absolute_error")
    index = uint(recorded["maximum_absolute_error_index"], f"{context}.maximum_absolute_error_index")
    if maximum < 0.0 or maximum > threshold or index >= slots:
        fail(f"{context} maximum error or index is invalid")
    for key in ("mean_absolute_error", "root_mean_square_error"):
        if finite(recorded[key], f"{context}.{key}") < 0.0:
            fail(f"{context}.{key} must be nonnegative")
    if maximum == 0.0:
        if recorded["estimated_precision_bits"] is not None or recorded["estimated_precision_is_infinite"] is not True:
            fail(f"{context} zero-error precision encoding is invalid")
    elif recorded["estimated_precision_is_infinite"] is not False or not math.isclose(finite(recorded["estimated_precision_bits"], f"{context}.estimated_precision_bits"), -math.log2(maximum), rel_tol=1e-15, abs_tol=1e-15):
        fail(f"{context} estimated precision is invalid")


def _bootstrap_metadata(semantics: dict[str, Any]) -> dict[str, Any]:
    value = semantics["output_air_contract"]
    raw_scale = value["raw_scale_contract"]["nominal_raw_scale"]
    return {
        "ace_level": value["ace_logical_level"],
        "active_q_count": value["active_q_count"],
        "phantom_chain_index": value["phantom_chain_index"],
        "raw_scale": float.fromhex(raw_scale),
        "scale_degree": value["scale_degree"],
        "logical_slots": value["logical_slots"],
        "ciphertext_size": value["ciphertext_size"],
        "ntt_state": value["ntt_state"],
    }


def _validate_raw_scale(
    raw_scale: Any,
    contract: dict[str, Any],
    context: str,
) -> float:
    observed = positive(raw_scale, context)
    scaling_bits = uint(
        contract.get("scaling_modulus_bits"),
        f"{context} scaling modulus bits",
        1,
    )
    expected_degree = uint(
        contract.get("expected_scale_degree"),
        f"{context} expected scale degree",
    )
    tolerance = positive(
        contract.get("maximum_absolute_coordinate_error"),
        f"{context} coordinate tolerance",
    )
    coordinate = math.log2(observed) / scaling_bits
    if abs(coordinate - expected_degree) > tolerance:
        fail(f"{context} differs from the AIR scale coordinate")
    return observed


def _transition(before: dict[str, Any], transition: dict[str, Any]) -> dict[str, Any]:
    after = dict(before)
    deltas = {
        "ace_level": "ace_logical_level_delta",
        "active_q_count": "active_q_count_delta",
        "phantom_chain_index": "phantom_chain_index_delta",
        "scale_degree": "scale_degree_delta",
    }
    for output, delta in deltas.items():
        after[output] += transition[delta]
    multiplier = transition["raw_scale_multiplier"]
    after["raw_scale"] *= float.fromhex(multiplier) if isinstance(multiplier, str) else multiplier
    for key in ("logical_slots", "ciphertext_size", "ntt_state"):
        if transition[key] != "preserved":
            fail(f"unsupported post-operation transition for {key}")
    return after


def validate_gpu_metadata(
    records: list[tuple[str, list[complex], dict[str, Any]]],
    fixture: dict[str, Any],
    semantics: dict[str, Any],
    slots: int,
) -> None:
    nominal = _bootstrap_metadata(semantics)
    if nominal["logical_slots"] != slots:
        fail("bootstrap output AIR contract disagrees with context slots")
    fixed_expected = dict(nominal)
    del fixed_expected["raw_scale"]
    raw_scale_contract = semantics["output_air_contract"]["raw_scale_contract"]
    canonical_metadata: dict[str, Any] | None = None
    post = fixture["post_operations"]
    repeat_threshold = fixture["tolerances"]["repeat_maximum_absolute"]
    for case_id, _, metadata in records:
        require_keys(metadata, {"bootstrap", "metrics_vs_clear", "repeatability", "post_operations", "ownership"}, f"GPU {case_id} metadata")
        observed_metadata = require_keys(
            metadata["bootstrap"],
            set(nominal),
            f"GPU {case_id} bootstrap metadata",
        )
        observed_fixed = dict(observed_metadata)
        raw_scale = observed_fixed.pop("raw_scale")
        _validate_raw_scale(
            raw_scale,
            raw_scale_contract,
            f"GPU {case_id} raw scale",
        )
        if observed_fixed != fixed_expected:
            fail(f"GPU {case_id} bootstrap metadata differs from AIR")
        if canonical_metadata is None:
            canonical_metadata = dict(observed_metadata)
        elif observed_metadata != canonical_metadata:
            fail(f"GPU {case_id} bootstrap metadata differs across cases")
        expected = observed_metadata
        expected_multiply = _transition(
            expected, post["ciphertext_plaintext_multiply"]["transition"]
        )
        expected_rotation = _transition(
            expected, post["rotation"]["transition"]
        )
        repeats = require_keys(metadata["repeatability"], {"calls", "independently_owned_clones", "maximum_absolute_difference", "threshold", "call_metrics", "call_metadata"}, f"GPU {case_id} repeatability")
        if repeats["calls"] != 3 or repeats["independently_owned_clones"] is not True or repeats["threshold"] != repeat_threshold or finite(repeats["maximum_absolute_difference"], "repeat difference") > repeat_threshold or repeats["call_metadata"] != [expected, expected, expected] or not isinstance(repeats["call_metrics"], list) or len(repeats["call_metrics"]) != 3:
            fail(f"GPU {case_id} repeatability contract failed")
        for call, call_metric in enumerate(repeats["call_metrics"]):
            validate_operation_metric(
                call_metric,
                fixture["tolerances"]["provider_clear_maximum_absolute"],
                slots,
                f"GPU {case_id} repeat call {call} metric",
            )
        if repeats["call_metrics"][0] != metadata["metrics_vs_clear"]:
            fail(f"GPU {case_id} primary repeat metric differs from output metric")
        operations = require_keys(metadata["post_operations"], {"ciphertext_plaintext_multiply", "rotation"}, f"GPU {case_id} post operations")
        multiply = require_keys(operations["ciphertext_plaintext_multiply"], {"metric", "metadata", "values_sha256"}, f"GPU {case_id} multiply")
        rotation = require_keys(operations["rotation"], {"metric", "metadata", "step", "values_sha256"}, f"GPU {case_id} rotation")
        sha(multiply["values_sha256"], f"GPU {case_id} multiply values SHA-256")
        sha(rotation["values_sha256"], f"GPU {case_id} rotation values SHA-256")
        constant = post["inputs"]["multiply_constant"]
        multiply_threshold = fixture["tolerances"]["provider_clear_maximum_absolute"] * max(1.0, abs(complex(constant["real"], constant["imaginary"])))
        validate_operation_metric(multiply["metric"], multiply_threshold, slots, f"GPU {case_id} multiply metric")
        validate_operation_metric(rotation["metric"], fixture["tolerances"]["provider_clear_maximum_absolute"], slots, f"GPU {case_id} rotation metric")
        if multiply["metadata"] != expected_multiply:
            fail(f"GPU {case_id} multiply result or transition failed")
        if rotation["metadata"] != expected_rotation or rotation["step"] != post["inputs"]["rotation_step"]:
            fail(f"GPU {case_id} rotation result or transition failed")
        if metadata["ownership"] != {
            "input_clones_released": 3,
            "zero_argument_clones_released": 3,
            "bootstrap_results_released": 3,
            "post_operation_ciphertexts_released": 4,
            "post_operation_plaintexts_released": 1,
            "base_encrypted_arguments_released": 2,
            "post_operation_sources_independent": True,
            "all_owned_objects_released": True,
        }:
            fail(f"GPU {case_id} ownership closure failed")


def load_provider(record_path: Path, values_path: Path, provider: str, fixture: dict[str, Any], fixture_sha: str, bindings: dict[str, str], slots: int, semantics: dict[str, Any] | None = None, invocation: dict[str, Any] | None = None, context_manifest: dict[str, Any] | None = None) -> list[tuple[str, list[complex], dict[str, Any]]]:
    value = load_json(record_path)
    require_keys(value, {"schema_version", "provider", "fixture_sha256", "bindings", "case_order", "logical_slots", "binary", "records", "execution"}, f"{provider} record")
    if value["schema_version"] != PROVIDER_SCHEMAS[provider] or value["provider"] != provider or value["fixture_sha256"] != fixture_sha or value["bindings"] != bindings:
        fail(f"{provider} schema or provenance mismatch")
    if value["logical_slots"] != slots:
        fail(f"{provider} logical slot count mismatch")
    expected_order = [item["id"] for item in fixture["case_recipes"]]
    if value["case_order"] != expected_order:
        fail(f"{provider} case order mismatch")
    execution = value["execution"]
    if not isinstance(execution, dict) or execution.get("status") != "pass" or execution.get("skip_count") != 0 or execution.get("timeout_count") != 0 or execution.get("fallback_count") != 0 or execution.get("nonfinite_count") != 0:
        fail(f"{provider} execution is incomplete")
    common_execution = {"status", "skip_count", "timeout_count", "fallback_count", "nonfinite_count"}
    if provider == "native-ant":
        require_keys(execution, common_execution | {"native_bootstrap_invocation_count", "early_identity_copy_count", "context_attestation", "provenance"}, "native ANT execution")
    elif provider == "generated-ant":
        require_keys(execution, common_execution | {"generated_bootstrap_invocation_count", "linked_post_ckks_air_sha256", "linked_generated_source_sha256", "context_attestation", "provenance"}, "generated ANT execution")
    else:
        require_keys(execution, common_execution | {"generated_bootstrap_invocation_count", "executable_sha256", "teardown_completed"}, "generated Phantom execution")
    if provider == "native-ant" and (execution.get("native_bootstrap_invocation_count") != len(expected_order) or execution.get("early_identity_copy_count") != 0):
        fail("native ANT did not prove a real bootstrap for every case")
    if provider == "generated-ant" and execution.get("generated_bootstrap_invocation_count") != len(expected_order):
        fail("generated ANT did not run every case")
    if provider == "generated-phantom":
        if execution.get("generated_bootstrap_invocation_count") != len(expected_order) * 3 or execution.get("teardown_completed") is not True:
            fail("generated Phantom call count is invalid")
        sha(execution.get("executable_sha256"), "generated Phantom executable SHA-256")
    else:
        if invocation is None or context_manifest is None:
            fail(f"{provider} validation requires invocation and context authorities")
        options = invocation["options"]
        expected_context_attestation = {
            "provider": provider,
            "polynomial_degree": context_manifest["polynomial_degree"],
            "logical_slots": slots,
            "data_q_count": len(context_manifest["data_q_bit_sizes"]),
            "special_p_count": len(context_manifest["special_p_bit_sizes"]),
            "data_q_requested_bit_sizes": context_manifest["data_q_bit_sizes"],
            "special_p_requested_bit_sizes": context_manifest["special_p_bit_sizes"],
            "input_level": context_manifest["input_level"],
            "scaling_factor_bits": context_manifest["scaling_modulus_bits"],
            "first_prime_bits": context_manifest["first_modulus_bits"],
            "hamming_weight": context_manifest["hamming_weight"],
            "security_level": context_manifest["security_level"],
            "q_part_count": context_manifest["q_part_count"],
            "transform_budgets": {"encode": options["encode_transform_budget"], "decode": options["decode_transform_budget"]},
            "phantom_context_manifest_sha256": bindings["context_manifest_sha256"],
            "physical_prime_identity_authority": "independent-non-authoritative-for-gpu",
            "schedule_authority": "independent-non-authoritative-for-gpu",
        }
        if execution.get("context_attestation") != expected_context_attestation:
            fail(f"{provider} context attestation differs from explicit authorities")
        provenance = require_keys(execution["provenance"], {"fixture_sha256", "compiler_invocation_sha256", "compiler_context_manifest_sha256", "compiler_resource_manifest_sha256", "compiler_constant_manifest_sha256", "bootstrap_semantics_sha256", "qualification_bindings", "executable_sha256"}, f"{provider} provenance")
        expected_provenance = {
            "fixture_sha256": fixture_sha,
            "compiler_invocation_sha256": bindings["compiler_invocation_sha256"],
            "compiler_context_manifest_sha256": bindings["context_manifest_sha256"],
            "compiler_resource_manifest_sha256": bindings["resource_manifest_sha256"],
            "compiler_constant_manifest_sha256": bindings["constant_manifest_sha256"],
            "bootstrap_semantics_sha256": bindings["bootstrap_semantics_sha256"],
            "qualification_bindings": bindings,
        }
        if any(provenance[key] != expected for key, expected in expected_provenance.items()):
            fail(f"{provider} executable provenance differs from authorities")
        sha(provenance["executable_sha256"], f"{provider} executable SHA-256")
        if provider == "generated-ant":
            if semantics is None or execution["linked_post_ckks_air_sha256"] != bindings["post_ckks_air_sha256"] or execution["linked_generated_source_sha256"] != semantics["bindings"]["generated_dsl_ant_source_sha256"]:
                fail("generated ANT linked source or AIR provenance mismatch")
    records = read_value_file(values_path, value["binary"], value["records"], fixture_sha, bindings["context_manifest_sha256"], provider)
    if [item[0] for item in records] != expected_order or any(len(item[1]) != slots for item in records):
        fail(f"{provider} record order or full-slot length mismatch")
    clear_cases = dict(materialize_cases(fixture, slots))
    recipes = {item["id"]: item["kind"] for item in fixture["case_recipes"]}
    threshold = fixture["tolerances"]["provider_clear_maximum_absolute"]
    attestation_keys = {"invocation_count", "early_identity_copy_count", "full_execution_count", "coeffs_to_slots_entry_count", "coeffs_to_slots_completion_count", "eval_mod_entry_count", "eval_mod_completion_count", "slots_to_coeffs_entry_count", "slots_to_coeffs_completion_count", "full_completion_count"}
    for case_id, provider_values, metadata in records:
        expected_keys = {"recipe", "metrics_vs_clear"}
        if provider == "native-ant":
            expected_keys.add("execution_attestation")
        if provider != "generated-phantom":
            require_keys(metadata, expected_keys, f"{provider} {case_id} metadata")
            if metadata["recipe"] != recipes[case_id]:
                fail(f"{provider} {case_id} recipe mismatch")
        recorded_metric = metadata["metrics_vs_clear"]
        validate_recorded_metric(recorded_metric, metric(provider_values, clear_cases[case_id], threshold), f"{provider} {case_id} metrics_vs_clear")
        if provider == "native-ant":
            attestation = require_keys(metadata["execution_attestation"], attestation_keys, f"native ANT {case_id} execution attestation")
            if any(attestation[key] != (0 if key == "early_identity_copy_count" else 1) for key in attestation_keys):
                fail(f"native ANT {case_id} did not attest one complete bootstrap")
    if provider == "generated-phantom":
        if semantics is None:
            fail("generated Phantom validation requires bootstrap semantics")
        validate_gpu_metadata(records, fixture, semantics, slots)
    return records


def compare(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = {name: Path(getattr(arguments, name)) for name in BINDING_INPUTS}
    _, semantics, context, bindings = validate_authorities(paths)
    fixture_path = Path(arguments.fixture)
    fixture = load_json(fixture_path)
    validate_fixture(fixture)
    if fixture["bindings"] != bindings:
        fail("fixture bindings disagree with supplied qualification artifacts")
    if fixture["supported_identity_domain"]["attested_domain"] != semantics["supported_identity_domain"]:
        fail("fixture supported identity domain differs from bootstrap semantics")
    post_attestation = load_json(paths["post_operation_attestation"])
    expected_post_operations = {
        "inputs": post_attestation["inputs"],
        "input_coordinate": post_attestation["input_coordinate"],
        "rotation": post_attestation["rotation"],
        "ciphertext_plaintext_multiply": post_attestation[
            "ciphertext_plaintext_multiply"
        ],
    }
    if fixture["post_operations"] != expected_post_operations:
        fail("fixture post-operation semantics differ from the AIR attestation")
    fixture_sha = digest(fixture_path)
    slots = uint(context["logical_slot_capacity"], "logical slots", 1)
    clear = materialize_cases(fixture, slots)
    providers = {
        "native-ant": load_provider(Path(arguments.native_record), Path(arguments.native_values), "native-ant", fixture, fixture_sha, bindings, slots, semantics=semantics, invocation=load_json(paths["compiler_invocation"]), context_manifest=context),
        "generated-ant": load_provider(Path(arguments.generated_record), Path(arguments.generated_values), "generated-ant", fixture, fixture_sha, bindings, slots, semantics=semantics, invocation=load_json(paths["compiler_invocation"]), context_manifest=context),
        "generated-phantom": load_provider(Path(arguments.gpu_record), Path(arguments.gpu_values), "generated-phantom", fixture, fixture_sha, bindings, slots, semantics),
    }
    sanitizer = load_json(Path(arguments.sanitizer_record))
    require_keys(sanitizer, {"schema_version", "status", "bindings", "exit_status", "error_summary_occurrences", "error_count", "coverage"}, "sanitizer record")
    if sanitizer["schema_version"] != SANITIZER_SCHEMA or sanitizer["status"] != "pass" or sanitizer["exit_status"] != 0 or sanitizer["error_summary_occurrences"] != 1 or sanitizer["error_count"] != 0:
        fail("Compute Sanitizer did not close cleanly")
    sanitizer_bindings = require_keys(sanitizer["bindings"], {"gpu_executable_sha256", "gpu_record_sha256", "gpu_values_sha256", "log_sha256", "tool_version_sha256"}, "sanitizer.bindings")
    for key, value in sanitizer_bindings.items():
        sha(value, f"sanitizer.bindings.{key}")
    if sanitizer_bindings["gpu_record_sha256"] != digest(Path(arguments.gpu_record)) or sanitizer_bindings["gpu_values_sha256"] != digest(Path(arguments.gpu_values)):
        fail("Compute Sanitizer record binds different GPU output")
    gpu_document = load_json(Path(arguments.gpu_record))
    if (
        sanitizer_bindings["gpu_executable_sha256"] != digest(Path(arguments.gpu_executable))
        or sanitizer_bindings["gpu_executable_sha256"] != gpu_document["execution"].get("executable_sha256")
        or sanitizer_bindings["log_sha256"] != digest(Path(arguments.sanitizer_log))
        or sanitizer_bindings["tool_version_sha256"] != digest(Path(arguments.sanitizer_tool_version))
    ):
        fail("Compute Sanitizer artifact binding mismatch")
    try:
        sanitizer_log = Path(arguments.sanitizer_log).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        fail(f"cannot read Compute Sanitizer log: {error}")
    summary_lines = [
        line
        for line in sanitizer_log.splitlines()
        if "ERROR SUMMARY:" in line
    ]
    if summary_lines != ["========= ERROR SUMMARY: 0 errors"]:
        fail("Compute Sanitizer log lacks exactly one zero-error summary")
    if sanitizer["coverage"] != {"bootstrap": True, "post_operations": True, "repeatability": True, "teardown": True}:
        fail("Compute Sanitizer coverage is incomplete")
    thresholds = fixture["tolerances"]
    comparison_specs = (
        ("native_ant_vs_clear", providers["native-ant"], clear, thresholds["provider_clear_maximum_absolute"]),
        ("generated_ant_vs_clear", providers["generated-ant"], clear, thresholds["provider_clear_maximum_absolute"]),
        ("generated_phantom_vs_clear", providers["generated-phantom"], clear, thresholds["provider_clear_maximum_absolute"]),
        ("generated_phantom_vs_native_ant", providers["generated-phantom"], providers["native-ant"], thresholds["gpu_native_maximum_absolute"]),
        ("generated_phantom_vs_generated_ant", providers["generated-phantom"], providers["generated-ant"], thresholds["gpu_generated_maximum_absolute"]),
    )
    comparisons = []
    overall = "pass"
    for comparison_id, actual, reference, threshold in comparison_specs:
        cases = []
        for position, ((case_id, values, _), reference_item) in enumerate(zip(actual, reference)):
            ref_id, ref_values = reference_item[0], reference_item[1]
            if case_id != ref_id:
                fail(f"{comparison_id} case order mismatch at {position}")
            result = metric(values, ref_values, threshold)
            overall = "fail" if result["status"] == "fail" else overall
            cases.append({"case_id": case_id, **result})
        comparisons.append({"comparison_id": comparison_id, "cases": cases, "status": "pass" if all(item["status"] == "pass" for item in cases) else "fail"})
    output = {"schema_version": COMPARISON_SCHEMA, "status": overall, "fixture_sha256": fixture_sha, "bindings": bindings, "sanitizer_record_sha256": digest(Path(arguments.sanitizer_record)), "logical_slots": slots, "case_order": [item[0] for item in clear], "comparisons": comparisons}
    write_fresh(Path(arguments.output), canonical_bytes(output))
    if overall != "pass":
        fail("three-way correctness comparison failed")
    return output


def verify_host(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = {name: Path(getattr(arguments, name)) for name in BINDING_INPUTS}
    _, semantics, context, bindings = validate_authorities(paths)
    fixture_path = Path(arguments.fixture)
    fixture = load_json(fixture_path)
    validate_fixture(fixture)
    if fixture["bindings"] != bindings or fixture["supported_identity_domain"]["attested_domain"] != semantics["supported_identity_domain"]:
        fail("host replay fixture provenance differs from supplied authorities")
    post_attestation = load_json(paths["post_operation_attestation"])
    expected_post_operations = {
        "inputs": post_attestation["inputs"],
        "input_coordinate": post_attestation["input_coordinate"],
        "rotation": post_attestation["rotation"],
        "ciphertext_plaintext_multiply": post_attestation[
            "ciphertext_plaintext_multiply"
        ],
    }
    if fixture["post_operations"] != expected_post_operations:
        fail("host replay fixture post-operation semantics differ from AIR")
    fixture_sha = digest(fixture_path)
    slots = uint(context["logical_slot_capacity"], "logical slots", 1)
    clear = materialize_cases(fixture, slots)
    invocation = load_json(paths["compiler_invocation"])
    native = load_provider(Path(arguments.native_record), Path(arguments.native_values), "native-ant", fixture, fixture_sha, bindings, slots, semantics=semantics, invocation=invocation, context_manifest=context)
    generated = load_provider(Path(arguments.generated_record), Path(arguments.generated_values), "generated-ant", fixture, fixture_sha, bindings, slots, semantics=semantics, invocation=invocation, context_manifest=context)
    cases = []
    threshold = fixture["tolerances"]["provider_clear_maximum_absolute"]
    for (case_id, native_values, _), (generated_id, generated_values, _), (clear_id, clear_values) in zip(native, generated, clear):
        if case_id != clear_id or generated_id != clear_id:
            fail("host replay case order mismatch")
        native_metric = metric(native_values, clear_values, threshold)
        generated_metric = metric(generated_values, clear_values, threshold)
        cases.append({"case_id": clear_id, "native_ant_vs_clear": native_metric, "generated_ant_vs_clear": generated_metric})
    status = "pass" if all(item["native_ant_vs_clear"]["status"] == "pass" and item["generated_ant_vs_clear"]["status"] == "pass" for item in cases) else "fail"
    output = {"schema_version": "ace.phantom.bootstrap-host-oracle-replay/1.0.0", "status": status, "fixture_sha256": fixture_sha, "bindings": bindings, "logical_slots": slots, "case_order": [item[0] for item in clear], "cases": cases}
    write_fresh(Path(arguments.output), canonical_bytes(output))
    if status != "pass":
        fail("host oracle replay failed")
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-fixture")
    freeze.add_argument("--fixture-id", required=True)
    freeze.add_argument("--seed", required=True, type=int)
    freeze.add_argument("--inside-margin", required=True, type=float)
    freeze.add_argument("--provider-clear-threshold", required=True, type=float)
    freeze.add_argument("--gpu-native-threshold", required=True, type=float)
    freeze.add_argument("--gpu-generated-threshold", required=True, type=float)
    freeze.add_argument("--repeat-threshold", required=True, type=float)
    freeze.add_argument("--output", required=True)
    for name in BINDING_INPUTS:
        freeze.add_argument(f"--{name.replace('_', '-')}", required=True)
    freeze.set_defaults(action=freeze_fixture)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--fixture", required=True)
    compare_parser.add_argument("--native-record", required=True)
    compare_parser.add_argument("--native-values", required=True)
    compare_parser.add_argument("--generated-record", required=True)
    compare_parser.add_argument("--generated-values", required=True)
    compare_parser.add_argument("--gpu-record", required=True)
    compare_parser.add_argument("--gpu-values", required=True)
    compare_parser.add_argument("--sanitizer-record", required=True)
    compare_parser.add_argument("--sanitizer-log", required=True)
    compare_parser.add_argument("--sanitizer-tool-version", required=True)
    compare_parser.add_argument("--gpu-executable", required=True)
    compare_parser.add_argument("--output", required=True)
    for name in BINDING_INPUTS:
        compare_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    compare_parser.set_defaults(action=compare)
    host = commands.add_parser("verify-host")
    host.add_argument("--fixture", required=True)
    host.add_argument("--native-record", required=True)
    host.add_argument("--native-values", required=True)
    host.add_argument("--generated-record", required=True)
    host.add_argument("--generated-values", required=True)
    host.add_argument("--output", required=True)
    for name in BINDING_INPUTS:
        host.add_argument(f"--{name.replace('_', '-')}", required=True)
    host.set_defaults(action=verify_host)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.action(arguments)
        return 0
    except CorrectnessError as error:
        print(f"bootstrap correctness failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
