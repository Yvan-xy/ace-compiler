#!/usr/bin/env python3
"""Strict ordinary-CKKS fixture, value file, oracle ingestion, and comparator.

This module intentionally does not implement an ANT oracle. ``generate-analytic``
creates an incomplete CPU reference, and ``ingest-ant`` accepts only a separately
produced ANT provider result. Comparison is a qualification failure until that
independent result has been ingested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Sequence


FIXTURE_SCHEMA = "ace.phantom.ordinary_ckks.fixture/1.0.0"
REFERENCE_SCHEMA = "ace.phantom.ordinary_ckks.cpu-reference/1.0.0"
PROVIDER_SCHEMA = "ace.phantom.ordinary_ckks.provider-result/1.0.0"
RAW_PROVIDER_SCHEMA = "ace.phantom.ordinary_ckks.raw-provider/1.0.0"
COMPARE_SCHEMA = "ace.phantom.ordinary_ckks.compare/1.0.0"
OWNERSHIP_SCHEMA = "ace.phantom.ordinary_ckks.ownership/1.0.0"
INVOCATION_SCHEMA = "ace.phantom.compiler-invocation/1.0.0"
QUALIFICATION_INVOCATION_SCHEMA = (
    "ace.phantom.qualification-invocation/1.0.0"
)
BINARY_FORMAT = "ace.ordinary_ckks.complex_float64le/1.0.0"
MAGIC = b"ACECKK01"
HEADER = struct.Struct("<8sHHIQQ32s32s32s")
PAIR = struct.Struct("<dd")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PHANTOM_POLY_DEGREE_MAX = 131072
PHANTOM_USER_MODULUS_BITS_MIN = 2
PHANTOM_USER_MODULUS_BITS_MAX = 60
PHANTOM_COEFF_MODULUS_COUNT_MAX = 64


class OrdinaryCkksError(ValueError):
    """A deterministic ordinary-CKKS input or comparison failure."""


def fail(message: str) -> None:
    raise OrdinaryCkksError(message)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    fail(f"non-finite JSON number: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"top-level JSON value in {path} must be an object")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        fail(f"cannot hash {path}: {error}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _expect_keys(
    value: Any,
    required: set[str],
    context: str,
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    optional = optional or set()
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        fail(f"{context} is missing keys: {sorted(missing)}")
    if unknown:
        fail(f"{context} has unknown keys: {sorted(unknown)}")
    return value


def _integer(value: Any, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{context} must be at least {minimum}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{context} must be finite")
    return result


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def _validate_qualification_bindings(value: Any) -> dict[str, Any]:
    bindings = _expect_keys(
        value,
        {
            "status",
            "deterministic_seed",
            "deterministic_generator",
            "normalized_compiler_command_sha256",
            "post_ckks_air_sha256",
        },
        "fixture.qualification_bindings",
    )
    seed = _integer(
        bindings["deterministic_seed"],
        "fixture.qualification_bindings.deterministic_seed",
        0,
    )
    if seed > (1 << 64) - 1:
        fail("fixture.qualification_bindings.deterministic_seed exceeds uint64")
    if bindings["deterministic_generator"] != "splitmix64-float53-complex-v1":
        fail("fixture deterministic input generator is unsupported")
    status = bindings["status"]
    digest_keys = (
        "normalized_compiler_command_sha256",
        "post_ckks_air_sha256",
    )
    if status == "unbound":
        if any(bindings[key] is not None for key in digest_keys):
            fail("unbound fixture qualification hashes must be null")
    elif status == "bound":
        for key in digest_keys:
            _sha256(bindings[key], f"fixture.qualification_bindings.{key}")
    else:
        fail("fixture.qualification_bindings.status must be 'unbound' or 'bound'")
    return bindings


def _seeded_complex_values(seed: int, count: int) -> list[complex]:
    mask = (1 << 64) - 1
    state = seed
    components: list[float] = []
    for _ in range(count * 2):
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        components.append(((value >> 11) / float(1 << 53)) * 2.0 - 1.0)
    return [
        complex(components[index], components[index + 1])
        for index in range(0, len(components), 2)
    ]


def _validate_seeded_fixture_input(fixture: dict[str, Any]) -> None:
    specification = fixture["inputs"].get("complex_y")
    if not isinstance(specification, dict):
        fail("fixture.inputs.complex_y must contain the seeded input")
    overrides = specification.get("overrides")
    if not isinstance(overrides, list) or len(overrides) != 8:
        fail("fixture.inputs.complex_y must contain eight seeded overrides")
    expected = _seeded_complex_values(
        fixture["qualification_bindings"]["deterministic_seed"],
        len(overrides),
    )
    for index, (entry, expected_value) in enumerate(zip(overrides, expected)):
        if (
            entry[0] != index
            or float(entry[1]) != expected_value.real
            or float(entry[2]) != expected_value.imag
        ):
            fail(
                "fixture.inputs.complex_y does not match deterministic seed "
                f"at override {index}"
            )


def _load_normalized_invocation(
    path: Path,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    invocation = _expect_keys(
        load_json(path),
        {"schema_version", "argv", "normalized_argv_sha256"},
        "compiler invocation",
    )
    if invocation["schema_version"] != INVOCATION_SCHEMA:
        fail("compiler invocation schema is unsupported")
    argv = invocation["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(argument, str) or not argument for argument in argv)
    ):
        fail("compiler invocation argv must be a non-empty string array")
    recorded = _sha256(
        invocation["normalized_argv_sha256"],
        "compiler invocation.normalized_argv_sha256",
    )
    observed = sha256_bytes(canonical_bytes(argv))
    if recorded != observed:
        fail(
            "normalized compiler command hash mismatch: "
            f"expected {recorded}, got {observed}"
        )
    if argv[0] != "tools/phantom_gpu/generate_ckks2c_probe.py":
        fail("compiler invocation has an unexpected executable")
    arguments = argv[1:]
    if len(arguments) % 2:
        fail("compiler invocation has an option without a value")
    pairs = dict(zip(arguments[0::2], arguments[1::2]))
    artifact_paths = {
        "--output": "ckks2c/add_mul_rotate.cu",
        "--post-ckks-air": "ckks2c/ordinary_ckks_post_ckks.air",
        "--context-manifest": "ckks2c/compiler_context_manifest.json",
        "--resource-manifest": "ckks2c/compiler_resource_manifest.json",
    }
    context_options = {
        "--poly-degree",
        "--mul-level",
        "--input-level",
        "--security-level",
        "--scaling-factor-bits",
        "--first-prime-bits",
        "--hamming-weight",
    }
    if (
        len(pairs) != len(arguments) // 2
        or set(pairs) != set(artifact_paths) | context_options
    ):
        fail("compiler invocation options are incomplete or duplicated")
    for option, expected in artifact_paths.items():
        if pairs[option] != expected:
            fail(f"compiler invocation {option} has a noncanonical artifact path")
    for option in context_options:
        if re.fullmatch(r"[0-9]+", pairs[option]) is None:
            fail(f"compiler invocation {option} must be a nonnegative integer")
    return invocation, observed, pairs


def verify_invocation_context(
    compiler_pairs: dict[str, str], context_manifest: dict[str, Any]
) -> None:
    validate_context_manifest(context_manifest)
    expected = {
        "--poly-degree": context_manifest["polynomial_degree"],
        "--mul-level": len(context_manifest["data_q_bit_sizes"]),
        "--input-level": context_manifest["input_level"],
        "--security-level": context_manifest["security_level"],
        "--scaling-factor-bits": context_manifest["scaling_modulus_bits"],
        "--first-prime-bits": context_manifest["first_modulus_bits"],
        "--hamming-weight": context_manifest["hamming_weight"],
    }
    for option, context_value in expected.items():
        if int(compiler_pairs[option]) != context_value:
            fail(
                f"compiler invocation {option} disagrees with the emitted "
                "context manifest"
            )


def verify_qualification_bindings(
    fixture: dict[str, Any],
    context_manifest_path: Path,
    compiler_invocation_path: Path,
    post_ckks_air_path: Path,
) -> None:
    bindings = _validate_qualification_bindings(fixture["qualification_bindings"])
    if bindings["status"] != "bound":
        fail("ordinary CKKS fixture is not bound to qualification artifacts")
    _, invocation_sha256, compiler_pairs = _load_normalized_invocation(
        compiler_invocation_path
    )
    verify_invocation_context(
        compiler_pairs, load_json(context_manifest_path)
    )
    if invocation_sha256 != bindings["normalized_compiler_command_sha256"]:
        fail("fixture normalized compiler command binding does not match")
    try:
        if post_ckks_air_path.stat().st_size == 0:
            fail("post-CKKS AIR artifact must be non-empty")
    except OSError as error:
        fail(f"cannot inspect post-CKKS AIR {post_ckks_air_path}: {error}")
    if sha256_path(post_ckks_air_path) != bindings["post_ckks_air_sha256"]:
        fail("fixture post-CKKS AIR binding does not match")


def bind_fixture(
    fixture_path: Path,
    context_manifest_path: Path,
    compiler_invocation_path: Path,
    post_ckks_air_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    fixture = load_json(fixture_path)
    validate_fixture(fixture)
    bindings = fixture["qualification_bindings"]
    if bindings["status"] != "unbound":
        fail("fixture template must be unbound before qualification binding")
    context_manifest = load_json(context_manifest_path)
    validate_context_manifest(context_manifest)
    context_manifest_sha256 = sha256_path(context_manifest_path)
    _, invocation_sha256, compiler_pairs = _load_normalized_invocation(
        compiler_invocation_path
    )
    verify_invocation_context(compiler_pairs, context_manifest)
    try:
        if post_ckks_air_path.stat().st_size == 0:
            fail("post-CKKS AIR artifact must be non-empty")
    except OSError as error:
        fail(f"cannot inspect post-CKKS AIR {post_ckks_air_path}: {error}")
    fixture["qualification_bindings"] = {
        **bindings,
        "status": "bound",
        "normalized_compiler_command_sha256": invocation_sha256,
        "post_ckks_air_sha256": sha256_path(post_ckks_air_path),
    }
    fixture["compiler_context_manifest"] = {
        "sha256": context_manifest_sha256,
    }
    validate_fixture(fixture)
    write_json(output_path, fixture)
    return fixture


def _complex_pair(value: Any, context: str) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{context} must be [real, imaginary]")
    return complex(_number(value[0], f"{context}[0]"), _number(value[1], f"{context}[1]"))


def validate_context_manifest(value: Any) -> dict[str, int]:
    manifest = _expect_keys(
        value,
        {
            "schema_version",
            "packing",
            "polynomial_degree",
            "logical_slot_capacity",
            "data_q_bit_sizes",
            "special_p_bit_sizes",
            "input_level",
            "q_part_count",
            "hamming_weight",
            "security_level",
            "first_modulus_bits",
            "scaling_modulus_bits",
            "resource_schema_version",
        },
        "compiler context manifest",
    )
    if manifest["schema_version"] != 1 or manifest["resource_schema_version"] != 1:
        fail("compiler context manifest schema is unsupported")
    if manifest["packing"] != "full":
        fail("compiler context manifest packing is unsupported")
    degree = _integer(
        manifest["polynomial_degree"],
        "compiler context manifest.polynomial_degree",
        2,
    )
    if degree & (degree - 1):
        fail("compiler context manifest polynomial degree must be a power of two")
    if degree > PHANTOM_POLY_DEGREE_MAX:
        fail("compiler context manifest polynomial degree exceeds provider limits")
    slots = _integer(
        manifest["logical_slot_capacity"],
        "compiler context manifest.logical_slot_capacity",
        1,
    )
    if slots != degree // 2:
        fail("compiler context manifest logical slots disagree with full packing")
    data_q = manifest["data_q_bit_sizes"]
    special_p = manifest["special_p_bit_sizes"]
    if not isinstance(data_q, list) or not data_q:
        fail("compiler context manifest data-Q list must be non-empty")
    if not isinstance(special_p, list) or not special_p:
        fail("compiler context manifest special-P list must be non-empty")
    first_bits = _integer(
        manifest["first_modulus_bits"],
        "compiler context manifest.first_modulus_bits",
        1,
    )
    scale_bits = _integer(
        manifest["scaling_modulus_bits"],
        "compiler context manifest.scaling_modulus_bits",
        1,
    )
    if not (
        PHANTOM_USER_MODULUS_BITS_MIN
        <= first_bits
        <= PHANTOM_USER_MODULUS_BITS_MAX
        and PHANTOM_USER_MODULUS_BITS_MIN
        <= scale_bits
        <= PHANTOM_USER_MODULUS_BITS_MAX
    ):
        fail("compiler context manifest data-Q bit sizes exceed provider limits")
    for index, bits in enumerate(data_q):
        expected = first_bits if index == 0 else scale_bits
        if (
            _integer(
                bits,
                f"compiler context manifest.data_q_bit_sizes[{index}]",
                PHANTOM_USER_MODULUS_BITS_MIN,
            )
            != expected
        ):
            fail("compiler context manifest data-Q list disagrees with its prime policy")
    for index, bits in enumerate(special_p):
        if (
            _integer(
                bits,
                f"compiler context manifest.special_p_bit_sizes[{index}]",
                PHANTOM_USER_MODULUS_BITS_MIN,
            )
            > PHANTOM_USER_MODULUS_BITS_MAX
        ):
            fail("compiler context manifest special-P bit size exceeds provider limits")
    if len(data_q) + len(special_p) > PHANTOM_COEFF_MODULUS_COUNT_MAX:
        fail("compiler context manifest combined Q/P count exceeds provider limits")
    input_level = _integer(
        manifest["input_level"], "compiler context manifest.input_level", 1
    )
    q_parts = _integer(
        manifest["q_part_count"], "compiler context manifest.q_part_count", 1
    )
    if input_level > len(data_q) or q_parts > len(data_q):
        fail("compiler context manifest input level or Q-part count is out of range")
    hamming_weight = _integer(
        manifest["hamming_weight"],
        "compiler context manifest.hamming_weight",
        1,
    )
    if hamming_weight > degree:
        fail("compiler context manifest hamming weight exceeds the degree")
    security = _integer(
        manifest["security_level"], "compiler context manifest.security_level", 0
    )
    if security not in {0, 128, 192, 256}:
        fail("compiler context manifest security setting is unsupported")
    return {
        "polynomial_degree": degree,
        "logical_slots": slots,
        "data_q_count": len(data_q),
        "scaling_modulus_bits": scale_bits,
    }


def resolved_context(fixture: dict[str, Any]) -> dict[str, int]:
    context = fixture.get("_resolved_context")
    if not isinstance(context, dict):
        fail("fixture has not been bound to its compiler context manifest")
    return context


def resolve_active_q_count(value: Any, fixture: dict[str, Any], context: str) -> int:
    data_q_count = resolved_context(fixture)["data_q_count"]
    if value == "bottom":
        return 1
    if value == "middle":
        return (data_q_count + 1) // 2
    if value == "after_one_drop":
        if data_q_count < 2:
            fail(f"{context} requires at least two compiler context data-Q primes")
        return data_q_count - 1
    if value == "full":
        return data_q_count
    fail(f"{context} must use a symbolic compiler-context coordinate")


def _validate_expression(value: Any, inputs: set[str], context: str) -> None:
    if not isinstance(value, dict):
        fail(f"{context} must be an expression object")
    if set(value) == {"input"}:
        if value["input"] not in inputs:
            fail(f"{context} references unknown input {value['input']!r}")
        return
    if set(value) == {"scalar"}:
        _complex_pair(value["scalar"], f"{context}.scalar")
        return
    operation = value.get("op")
    if operation in {"add", "subtract", "multiply"} and set(value) == {"op", "args"}:
        args = value["args"]
        if not isinstance(args, list) or len(args) != 2:
            fail(f"{context}.args must contain two expressions")
        _validate_expression(args[0], inputs, f"{context}.args[0]")
        _validate_expression(args[1], inputs, f"{context}.args[1]")
        return
    if operation == "rotate" and set(value) == {"op", "arg", "step"}:
        _validate_expression(value["arg"], inputs, f"{context}.arg")
        _integer(value["step"], f"{context}.step")
        return
    fail(f"{context} has an unsupported expression shape")


def validate_fixture(fixture: dict[str, Any]) -> None:
    top_keys = {
        "schema_version",
        "fixture_id",
        "description",
        "qualification_bindings",
        "compiler_context_manifest",
        "coordinate_rules",
        "tolerances",
        "binary_format",
        "inputs",
        "metadata_contracts",
        "alias_matrix",
        "case_families",
        "ownership_cases",
        "rejections",
    }
    _expect_keys(fixture, top_keys, "fixture")
    if fixture["schema_version"] != FIXTURE_SCHEMA:
        fail("unsupported fixture schema")
    if fixture["fixture_id"] != "ordinary_ckks_v1":
        fail("unexpected fixture identifier")
    if not isinstance(fixture["description"], str) or not fixture["description"]:
        fail("fixture.description must be non-empty")
    qualification_bindings = _validate_qualification_bindings(
        fixture["qualification_bindings"]
    )

    context_manifest = _expect_keys(
        fixture["compiler_context_manifest"],
        {"sha256"},
        "fixture.compiler_context_manifest",
    )
    context_sha256 = context_manifest["sha256"]
    if qualification_bindings["status"] == "unbound":
        if context_sha256 is not None:
            fail("unbound fixture compiler context manifest hash must be null")
    else:
        _sha256(
            context_sha256,
            "fixture.compiler_context_manifest.sha256",
        )
    coordinate_rules = _expect_keys(
        fixture["coordinate_rules"],
        {
            "input_length",
            "full_level",
            "after_one_drop_level",
            "middle_level",
            "bottom_level",
            "chain_index_formula",
            "rotation_formula",
        },
        "fixture.coordinate_rules",
    )
    if coordinate_rules != {
        "input_length": "compiler_context.logical_slot_capacity",
        "full_level": "compiler_context.data_q_count",
        "after_one_drop_level": "compiler_context.data_q_count - 1",
        "middle_level": "ceil(compiler_context.data_q_count / 2)",
        "bottom_level": "1",
        "chain_index_formula": (
            "adapter_first_data_chain_index + compiler_context.data_q_count "
            "- active_q_count"
        ),
        "rotation_formula": (
            "output[i] = input[(i + normalized_step) % "
            "compiler_context.logical_slot_capacity]"
        ),
    }:
        fail("fixture coordinate rules changed")
    if coordinate_rules["chain_index_formula"] != (
        "adapter_first_data_chain_index + compiler_context.data_q_count "
        "- active_q_count"
    ):
        fail("fixture chain-index formula changed")

    tolerances = _expect_keys(
        fixture["tolerances"],
        {
            "absolute",
            "relative",
            "hard_maximum_absolute",
            "relative_metric_floor",
            "metadata_scale_relative",
        },
        "fixture.tolerances",
    )
    fixed_tolerances = {
        "absolute": 1.0e-4,
        "relative": 1.0e-3,
        "hard_maximum_absolute": 5.0e-3,
        "relative_metric_floor": 1.0e-12,
        "metadata_scale_relative": 1.0e-12,
    }
    for key, expected in fixed_tolerances.items():
        if _number(tolerances[key], f"fixture.tolerances.{key}") != expected:
            fail(f"fixture.tolerances.{key} must be {expected}")

    binary = _expect_keys(
        fixture["binary_format"],
        {"id", "magic_ascii", "endianness", "value_encoding"},
        "fixture.binary_format",
    )
    if binary != {
        "id": BINARY_FORMAT,
        "magic_ascii": "ACECKK01",
        "endianness": "little",
        "value_encoding": "interleaved IEEE-754 binary64 real,imaginary pairs",
    }:
        fail("fixture binary format changed")

    inputs = fixture["inputs"]
    if not isinstance(inputs, dict) or not inputs:
        fail("fixture.inputs must be a non-empty object")
    allowed_types = {
        "complex_float64",
        "float32",
        "float64",
        "float32_mask",
        "float64_mask",
    }
    for name, specification in inputs.items():
        context = f"fixture.inputs.{name}"
        if not isinstance(name, str) or not name:
            fail("input names must be non-empty strings")
        base = _expect_keys(
            specification,
            {"element_type", "length", "storage", "default"},
            context,
            {"overrides", "segments"},
        )
        if base["element_type"] not in allowed_types:
            fail(f"{context}.element_type is unsupported")
        if base["length"] != "compiler_context.logical_slot_capacity":
            fail(f"{context}.length must come from the compiler context")
        default = _complex_pair(base["default"], f"{context}.default")
        if "complex" not in base["element_type"] and default.imag != 0.0:
            fail(f"{context} is real but has an imaginary default")
        storage = base["storage"]
        if storage == "default_overrides":
            if set(base) != {"element_type", "length", "storage", "default", "overrides"}:
                fail(f"{context} must contain only overrides for this storage")
            overrides = base["overrides"]
            if not isinstance(overrides, list):
                fail(f"{context}.overrides must be an array")
            previous = -1
            for position, entry in enumerate(overrides):
                entry_context = f"{context}.overrides[{position}]"
                if not isinstance(entry, list) or len(entry) != 3:
                    fail(f"{entry_context} must be [index, real, imaginary]")
                index = _integer(entry[0], f"{entry_context}[0]", 0)
                if index <= previous:
                    fail(f"{entry_context} indices must be sorted and unique")
                previous = index
                item = complex(
                    _number(entry[1], f"{entry_context}[1]"),
                    _number(entry[2], f"{entry_context}[2]"),
                )
                if "complex" not in base["element_type"] and item.imag != 0.0:
                    fail(f"{entry_context} is real but has an imaginary value")
        elif storage == "segments":
            if set(base) != {"element_type", "length", "storage", "default", "segments"}:
                fail(f"{context} must contain only segments for this storage")
            segments = base["segments"]
            if not isinstance(segments, list):
                fail(f"{context}.segments must be an array")
            previous_end = 0
            for position, entry in enumerate(segments):
                entry_context = f"{context}.segments[{position}]"
                if not isinstance(entry, list) or len(entry) != 4:
                    fail(f"{entry_context} must be [start, count, real, imaginary]")
                start = _integer(entry[0], f"{entry_context}[0]", 0)
                count = _integer(entry[1], f"{entry_context}[1]", 1)
                if start < previous_end:
                    fail(f"{entry_context} segments must be sorted and disjoint")
                previous_end = start + count
                _number(entry[2], f"{entry_context}[2]")
                if _number(entry[3], f"{entry_context}[3]") != 0.0:
                    fail(f"{entry_context} mask values must be real")
        else:
            fail(f"{context}.storage is unsupported")
    _validate_seeded_fixture_input(fixture)

    contracts = fixture["metadata_contracts"]
    if not isinstance(contracts, dict) or not contracts:
        fail("fixture.metadata_contracts must be a non-empty object")
    metadata_keys = {
        "object_kind",
        "active_q_count",
        "scale_degree",
        "raw_scale",
        "ciphertext_size",
        "ntt",
    }
    for name, contract in contracts.items():
        context = f"fixture.metadata_contracts.{name}"
        _expect_keys(contract, metadata_keys, context)
        if contract["object_kind"] not in {"ciphertext", "plaintext"}:
            fail(f"{context}.object_kind is unsupported")
        active_q = contract["active_q_count"]
        if active_q not in {"bottom", "middle", "after_one_drop", "full"}:
            fail(f"{context}.active_q_count must use a symbolic coordinate")
        _integer(contract["scale_degree"], f"{context}.scale_degree", 1)
        if not isinstance(contract["ntt"], bool):
            fail(f"{context} has invalid NTT state")
        size = contract["ciphertext_size"]
        if contract["object_kind"] == "ciphertext":
            _integer(size, f"{context}.ciphertext_size", 2)
        elif size is not None:
            fail(f"{context}.ciphertext_size must be null for plaintext")
        raw_scale = contract["raw_scale"]
        if not isinstance(raw_scale, dict):
            fail(f"{context}.raw_scale must be an object")
        if raw_scale.get("kind") == "exact_scale_degree":
            _expect_keys(raw_scale, {"kind"}, f"{context}.raw_scale")
        elif raw_scale.get("kind") == "divide_by_reported_dropped_modulus":
            _expect_keys(
                raw_scale,
                {"kind", "numerator_scale_degree"},
                f"{context}.raw_scale",
            )
            _integer(
                raw_scale["numerator_scale_degree"],
                f"{context}.raw_scale.numerator_scale_degree",
                1,
            )
        else:
            fail(f"{context}.raw_scale kind is unsupported")

    alias_matrix = fixture["alias_matrix"]
    expected_alias_matrix = {
        "ct_ct": {
            "distinct": "supported",
            "lhs": "supported",
            "rhs": "supported",
        },
        "ct_plain": {
            "distinct": "supported",
            "lhs": "supported",
            "rhs": "not_applicable",
        },
        "ct_scalar": {
            "distinct": "supported",
            "lhs": "supported",
            "rhs": "not_applicable",
        },
        "cipher_unary": {"distinct": "supported", "inplace": "supported"},
        "copy": {"distinct": "supported", "self": "supported"},
        "encode": {"distinct": "supported", "inplace": "not_applicable"},
        "query": {"distinct": "supported", "inplace": "not_applicable"},
        "zero": {"destination_source": "not_applicable"},
        "free": {"destination_source": "not_applicable"},
        "free_array": {"destination_source": "not_applicable"},
    }
    if alias_matrix != expected_alias_matrix:
        fail(
            "fixture.alias_matrix must exactly describe the ordinary CKKS "
            "type-valid and structurally inapplicable relations"
        )

    families = fixture["case_families"]
    if not isinstance(families, list) or not families:
        fail("fixture.case_families must be a non-empty array")
    family_ids: set[str] = set()
    case_ids: set[str] = set()
    allowed_operations = {
        "add",
        "copy",
        "encode_complex",
        "encode_mask_f32",
        "encode_mask_f64",
        "encode_real_f32",
        "encode_real_f64",
        "modswitch",
        "multiply",
        "query_all",
        "relinearize",
        "rescale",
        "rotate",
        "subtract",
    }
    operations_by_form = {
        "ct_ct": {"add", "subtract", "multiply"},
        "ct_plain": {"add", "subtract", "multiply"},
        "ct_scalar": {"add", "subtract", "multiply"},
        "cipher_unary": {"modswitch", "relinearize", "rescale", "rotate"},
        "copy": {"copy"},
        "encode": {
            "encode_complex",
            "encode_mask_f32",
            "encode_mask_f64",
            "encode_real_f32",
            "encode_real_f64",
        },
        "query": {"query_all"},
    }
    for position, family in enumerate(families):
        context = f"fixture.case_families[{position}]"
        _expect_keys(family, {"id", "operation", "form", "aliases", "expected", "metadata"}, context)
        family_id = family["id"]
        if not isinstance(family_id, str) or not family_id or family_id in family_ids:
            fail(f"{context}.id must be non-empty and unique")
        family_ids.add(family_id)
        operation = family["operation"]
        if operation not in allowed_operations:
            fail(f"{context}.operation is unsupported")
        if family_id.endswith("_ct_ct"):
            expected_form = "ct_ct"
        elif family_id.endswith("_ct_plain"):
            expected_form = "ct_plain"
        elif family_id.endswith("_ct_scalar"):
            expected_form = "ct_scalar"
        elif operation.startswith("encode_"):
            expected_form = "encode"
        elif operation == "query_all":
            expected_form = "query"
        elif operation == "copy":
            expected_form = "copy"
        else:
            expected_form = "cipher_unary"
        if family["form"] != expected_form:
            fail(
                f"{context}.form must be {expected_form} for ordinary API "
                f"{family_id}"
            )
        if family["form"] not in alias_matrix:
            fail(f"{context}.form is absent from the alias matrix")
        if operation not in operations_by_form[family["form"]]:
            fail(f"{context}.operation does not belong to form {family['form']}")
        if family["metadata"] not in contracts:
            fail(f"{context}.metadata references an unknown contract")
        aliases = family["aliases"]
        if not isinstance(aliases, list) or not aliases:
            fail(f"{context}.aliases must be a non-empty array")
        seen_aliases: set[str] = set()
        for alias in aliases:
            if alias in seen_aliases:
                fail(f"{context}.aliases contains a duplicate")
            seen_aliases.add(alias)
            disposition = alias_matrix[family["form"]].get(alias)
            if disposition != "supported":
                fail(f"{context} lists alias {alias!r} that is not supported")
            case_id = f"{family_id}.{alias}"
            if case_id in case_ids:
                fail(f"duplicate expanded case identifier: {case_id}")
            case_ids.add(case_id)
        supported_aliases = [
            alias
            for alias, disposition in alias_matrix[family["form"]].items()
            if disposition == "supported"
        ]
        if aliases != supported_aliases:
            fail(
                f"{context}.aliases must exercise every supported relation "
                f"for form {family['form']} in matrix order"
            )
        _validate_expression(family["expected"], set(inputs), f"{context}.expected")

    ownership = fixture["ownership_cases"]
    if not isinstance(ownership, list) or not ownership:
        fail("fixture.ownership_cases must be a non-empty array")
    ownership_ids: set[str] = set()
    expected_ownership = {
        "copy_independence": None,
        "destination_reuse": None,
        "free_array_1": 1,
        "free_array_4": 4,
        "zero_then_free": None,
    }
    for position, item in enumerate(ownership):
        context = f"fixture.ownership_cases[{position}]"
        value = _expect_keys(item, {"id", "iterations"}, context, {"array_length"})
        if not isinstance(value["id"], str) or not value["id"] or value["id"] in ownership_ids:
            fail(f"{context}.id must be non-empty and unique")
        ownership_ids.add(value["id"])
        if value["id"] not in expected_ownership:
            fail(f"{context}.id is not a supported ownership contract")
        if _integer(value["iterations"], f"{context}.iterations", 1) != 100:
            fail(f"{context}.iterations must be 100")
        expected_array_length = expected_ownership[value["id"]]
        if expected_array_length is None:
            if "array_length" in value:
                fail(f"{context}.array_length is only valid for array ownership cases")
        elif _integer(
            value.get("array_length"), f"{context}.array_length", 1
        ) != expected_array_length:
            fail(
                f"{context}.array_length must be {expected_array_length} for "
                f"{value['id']}"
            )
    if ownership_ids != set(expected_ownership):
        fail("fixture.ownership_cases does not contain the complete ownership contract")

    rejections = fixture["rejections"]
    if not isinstance(rejections, list) or not rejections:
        fail("fixture.rejections must be a non-empty array")
    expected_rejections = {
        "encode_invalid_length": "static",
        "encode_invalid_level": "static",
        "encode_invalid_scale": "static",
        "incompatible_level": "static",
        "incompatible_scale": "static",
        "missing_declared_relin_key": "static",
        "missing_declared_rotation_key": "static",
        "reverse_scalar_subtraction": "static",
        "bottom_modswitch": "runtime",
        "bottom_rescale": "runtime",
        "double_free": "runtime",
        "invalid_ciphertext_size": "runtime",
        "missing_loaded_relin_key": "runtime",
        "missing_loaded_rotation_key": "runtime",
        "relinearize_size2": "runtime",
        "runtime_level_mismatch": "runtime",
        "runtime_scale_mismatch": "runtime",
        "use_after_free": "runtime",
    }
    rejection_ids: set[str] = set()
    diagnostics: set[str] = set()
    for position, item in enumerate(rejections):
        context = f"fixture.rejections[{position}]"
        value = _expect_keys(
            item,
            {"id", "phase", "diagnostic_id"},
            context,
            {"provider_diagnostic", "runner_profile"},
        )
        if not isinstance(value["id"], str) or not value["id"] or value["id"] in rejection_ids:
            fail(f"{context}.id must be non-empty and unique")
        rejection_ids.add(value["id"])
        expected_phase = expected_rejections.get(value["id"])
        if expected_phase is None:
            fail(f"{context}.id is not a supported rejection contract")
        if value["phase"] != expected_phase:
            fail(f"{context}.phase must be {expected_phase} for {value['id']}")
        diagnostic = value["diagnostic_id"]
        if (
            not isinstance(diagnostic, str)
            or not diagnostic.startswith("ORDINARY_CKKS_")
            or diagnostic in diagnostics
        ):
            fail(f"{context}.diagnostic_id must be prefixed and unique")
        diagnostics.add(diagnostic)
        if value["phase"] == "runtime":
            provider_diagnostic = value.get("provider_diagnostic")
            if (
                not isinstance(provider_diagnostic, str)
                or not provider_diagnostic
                or any(
                    character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                    for character in provider_diagnostic
                )
            ):
                fail(f"{context}.provider_diagnostic must be an uppercase token")
            if value.get("runner_profile") not in {"ordinary", "keyless"}:
                fail(f"{context}.runner_profile must be ordinary or keyless")
        elif "provider_diagnostic" in value or "runner_profile" in value:
            fail(f"{context} static rejection cannot select a runtime diagnostic or runner")
    if rejection_ids != set(expected_rejections):
        fail("fixture.rejections does not contain the complete rejection contract")


def validate_ownership_result(
    ownership: dict[str, Any], fixture: dict[str, Any]
) -> int:
    """Validate the GPU ownership report against the complete frozen contract."""
    validate_fixture(fixture)
    value = _expect_keys(
        ownership,
        {"schema_version", "status", "total_iterations", "cases"},
        "ownership result",
    )
    if value["schema_version"] != OWNERSHIP_SCHEMA:
        fail("unsupported ownership result schema")
    if value["status"] != "pass":
        fail("ownership result status must be pass")
    total_iterations = _integer(
        value["total_iterations"], "ownership result.total_iterations", 1
    )
    cases = value["cases"]
    expected_cases = fixture["ownership_cases"]
    if not isinstance(cases, list):
        fail("ownership result.cases must be an array")
    if len(cases) != len(expected_cases):
        fail("ownership result must contain every fixture case exactly once")

    observed_total = 0
    for position, (case, expected) in enumerate(zip(cases, expected_cases)):
        context = f"ownership result.cases[{position}]"
        required = {"id", "iterations", "status"}
        if "array_length" in expected:
            required.add("array_length")
        report = _expect_keys(
            case, required, context
        )
        if report["id"] != expected["id"]:
            fail(f"{context}.id does not match the fixture order")
        iterations = _integer(report["iterations"], f"{context}.iterations", 1)
        if iterations != expected["iterations"]:
            fail(f"{context}.iterations does not match the fixture")
        if report["status"] != "pass":
            fail(f"{context}.status must be pass")
        if "array_length" in expected:
            array_length = _integer(
                report["array_length"], f"{context}.array_length", 1
            )
            if array_length != expected["array_length"]:
                fail(f"{context}.array_length does not match the fixture")
        observed_total += iterations

    if total_iterations != observed_total:
        fail("ownership result.total_iterations does not match its cases")
    return total_iterations


def load_fixture(
    path: Path,
    context_manifest_path: Path | None = None,
    require_qualification_bindings: bool = True,
) -> tuple[dict[str, Any], str]:
    fixture_bytes = path.read_bytes()
    fixture = load_json(path)
    validate_fixture(fixture)
    if (
        require_qualification_bindings
        and fixture["qualification_bindings"]["status"] != "bound"
    ):
        fail("ordinary CKKS fixture is not bound to qualification artifacts")
    if context_manifest_path is not None:
        observed = sha256_path(context_manifest_path)
        expected = fixture["compiler_context_manifest"]["sha256"]
        if observed != expected:
            fail(
                "compiler context manifest hash mismatch: "
                f"expected {expected}, got {observed}"
            )
        fixture["_resolved_context"] = validate_context_manifest(
            load_json(context_manifest_path)
        )
        slots = fixture["_resolved_context"]["logical_slots"]
        for name, specification in fixture["inputs"].items():
            if specification["storage"] == "default_overrides":
                for entry in specification["overrides"]:
                    if entry[0] >= slots:
                        fail(
                            f"fixture.inputs.{name} override index {entry[0]} "
                            "exceeds compiler logical slots"
                        )
            else:
                for entry in specification["segments"]:
                    if entry[0] + entry[1] > slots:
                        fail(
                            f"fixture.inputs.{name} segment exceeds compiler "
                            "logical slots"
                        )
        for name, contract in fixture["metadata_contracts"].items():
            resolve_active_q_count(
                contract["active_q_count"],
                fixture,
                f"fixture.metadata_contracts.{name}.active_q_count",
            )
    return fixture, sha256_bytes(fixture_bytes)


def expand_input(
    specification: dict[str, Any], fixture: dict[str, Any]
) -> list[complex]:
    default = _complex_pair(specification["default"], "input.default")
    if specification["length"] != "compiler_context.logical_slot_capacity":
        fail("input length is not bound to the compiler context")
    result = [default] * resolved_context(fixture)["logical_slots"]
    if specification["storage"] == "default_overrides":
        for index, real, imaginary in specification["overrides"]:
            result[index] = complex(float(real), float(imaginary))
    else:
        for start, count, real, imaginary in specification["segments"]:
            item = complex(float(real), float(imaginary))
            result[start : start + count] = [item] * count
    return result


def evaluate_expression(expression: dict[str, Any], fixture: dict[str, Any]) -> list[complex]:
    slots = resolved_context(fixture)["logical_slots"]
    if "input" in expression:
        return expand_input(fixture["inputs"][expression["input"]], fixture)
    if "scalar" in expression:
        return [_complex_pair(expression["scalar"], "expression.scalar")] * slots
    operation = expression["op"]
    if operation == "rotate":
        source = evaluate_expression(expression["arg"], fixture)
        step = expression["step"] % slots
        return [source[(index + step) % slots] for index in range(slots)]
    left = evaluate_expression(expression["args"][0], fixture)
    right = evaluate_expression(expression["args"][1], fixture)
    if operation == "add":
        return [a + b for a, b in zip(left, right)]
    if operation == "subtract":
        return [a - b for a, b in zip(left, right)]
    if operation == "multiply":
        return [a * b for a, b in zip(left, right)]
    fail(f"unsupported analytic operation: {operation}")


def expanded_cases(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in fixture["case_families"]:
        for alias in family["aliases"]:
            result.append(
                {
                    "case_id": f"{family['id']}.{alias}",
                    "family_id": family["id"],
                    "operation": family["operation"],
                    "form": family["form"],
                    "alias": alias,
                    "expected": family["expected"],
                    "metadata_contract": family["metadata"],
                }
            )
    return result


def analytic_records(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "oracle": "analytic",
            "values": evaluate_expression(case["expected"], fixture),
            "metadata": {},
        }
        for case in expanded_cases(fixture)
    ]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    rendered = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, rendered.encode("utf-8"))


def record_qualification_invocation(
    output_path: Path, argv: Sequence[str]
) -> dict[str, Any]:
    arguments = list(argv)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments or arguments[0] != "tools/phantom_gpu/compile_only.sh":
        fail("qualification invocation must name tools/phantom_gpu/compile_only.sh")
    options = arguments[1:]
    if len(options) % 2:
        fail("qualification invocation has an option without a value")
    pairs = dict(zip(options[0::2], options[1::2]))
    expected = {
        "--gate",
        "--poly-degree",
        "--mul-level",
        "--input-level",
        "--security-level",
        "--scaling-factor-bits",
        "--first-prime-bits",
        "--hamming-weight",
    }
    if len(pairs) != len(options) // 2 or set(pairs) != expected:
        fail("qualification invocation options are incomplete or duplicated")
    if pairs["--gate"] != "ordinary":
        fail("qualification invocation gate must be ordinary")
    for option in expected - {"--gate"}:
        if re.fullmatch(r"[0-9]+", pairs[option]) is None:
            fail(
                f"qualification invocation {option} must be a nonnegative integer"
            )
    record = {
        "schema_version": QUALIFICATION_INVOCATION_SCHEMA,
        "argv": arguments,
        "normalized_argv_sha256": sha256_bytes(canonical_bytes(arguments)),
    }
    write_json(output_path, record)
    return record


def _case_manifest(records: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes(
        canonical_bytes(
            [
                {
                    "case_id": record["case_id"],
                    "oracle": record["oracle"],
                    "value_count": len(record["values"]),
                }
                for record in records
            ]
        )
    )


def write_value_file(
    path: Path,
    fixture_sha256: str,
    context_manifest_sha256: str,
    records: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _sha256(fixture_sha256, "fixture SHA-256")
    _sha256(context_manifest_sha256, "compiler context manifest SHA-256")
    payload = bytearray()
    descriptors: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for position, record in enumerate(records):
        _expect_keys(record, {"case_id", "oracle", "values", "metadata"}, f"record[{position}]")
        identity = (record["case_id"], record["oracle"])
        if not all(isinstance(item, str) and item for item in identity) or identity in identities:
            fail(f"record[{position}] has a duplicate or invalid identity")
        identities.add(identity)
        values = record["values"]
        if not isinstance(values, list) or not values:
            fail(f"record[{position}].values must be a non-empty list")
        offset = len(payload)
        for index, value in enumerate(values):
            if not isinstance(value, complex):
                fail(f"record[{position}].values[{index}] must be complex")
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                fail(f"record[{position}].values[{index}] is non-finite")
            payload.extend(PAIR.pack(value.real, value.imag))
        if not isinstance(record["metadata"], dict):
            fail(f"record[{position}].metadata must be an object")
        descriptors.append(
            {
                "case_id": identity[0],
                "oracle": identity[1],
                "offset_bytes": offset,
                "value_count": len(values),
                "metadata": record["metadata"],
            }
        )
    manifest_sha256 = _case_manifest(records)
    header = HEADER.pack(
        MAGIC,
        1,
        0,
        0,
        len(records),
        len(payload),
        bytes.fromhex(fixture_sha256),
        bytes.fromhex(context_manifest_sha256),
        bytes.fromhex(manifest_sha256),
    )
    contents = header + payload
    _atomic_write(path, contents)
    binary = {
        "format": BINARY_FORMAT,
        "sha256": sha256_bytes(contents),
        "size_bytes": len(contents),
        "header_size": HEADER.size,
        "record_count": len(records),
        "payload_bytes": len(payload),
        "case_manifest_sha256": manifest_sha256,
    }
    return binary, descriptors


def _validate_binary_info(value: Any, context: str) -> dict[str, Any]:
    info = _expect_keys(
        value,
        {
            "format",
            "sha256",
            "size_bytes",
            "header_size",
            "record_count",
            "payload_bytes",
            "case_manifest_sha256",
        },
        context,
    )
    if info["format"] != BINARY_FORMAT:
        fail(f"{context}.format is unsupported")
    _sha256(info["sha256"], f"{context}.sha256")
    _sha256(info["case_manifest_sha256"], f"{context}.case_manifest_sha256")
    for key in ("size_bytes", "header_size", "record_count", "payload_bytes"):
        _integer(info[key], f"{context}.{key}", 0)
    if info["header_size"] != HEADER.size:
        fail(f"{context}.header_size is unsupported")
    return info


def _descriptor_records(
    descriptors: Any, context: str, expected_oracle: str | None = None
) -> list[dict[str, Any]]:
    if not isinstance(descriptors, list):
        fail(f"{context} must be an array")
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for position, descriptor in enumerate(descriptors):
        item_context = f"{context}[{position}]"
        item = _expect_keys(
            descriptor,
            {"case_id", "oracle", "offset_bytes", "value_count", "metadata"},
            item_context,
        )
        if not isinstance(item["case_id"], str) or not item["case_id"]:
            fail(f"{item_context}.case_id must be non-empty")
        if not isinstance(item["oracle"], str) or not item["oracle"]:
            fail(f"{item_context}.oracle must be non-empty")
        if expected_oracle is not None and item["oracle"] != expected_oracle:
            fail(f"{item_context}.oracle must be {expected_oracle}")
        identity = (item["case_id"], item["oracle"])
        if identity in identities:
            fail(f"{context} contains duplicate record {identity}")
        identities.add(identity)
        _integer(item["offset_bytes"], f"{item_context}.offset_bytes", 0)
        _integer(item["value_count"], f"{item_context}.value_count", 1)
        if item["offset_bytes"] % PAIR.size:
            fail(f"{item_context}.offset_bytes is not value-aligned")
        if not isinstance(item["metadata"], dict):
            fail(f"{item_context}.metadata must be an object")
        result.append(item)
    return result


def read_value_file(
    path: Path,
    binary_value: Any,
    descriptors_value: Any,
    fixture_sha256: str,
    context_manifest_sha256: str,
) -> list[dict[str, Any]]:
    info = _validate_binary_info(binary_value, "binary")
    descriptors = _descriptor_records(descriptors_value, "records")
    try:
        contents = path.read_bytes()
    except OSError as error:
        fail(f"cannot read value file {path}: {error}")
    if len(contents) != info["size_bytes"] or sha256_bytes(contents) != info["sha256"]:
        fail("value file size or SHA-256 mismatch")
    if len(contents) < HEADER.size:
        fail("value file is shorter than its header")
    (
        magic,
        major,
        minor,
        flags,
        record_count,
        payload_bytes,
        fixture_digest,
        context_manifest_digest,
        manifest_digest,
    ) = HEADER.unpack_from(contents)
    if magic != MAGIC or (major, minor, flags) != (1, 0, 0):
        fail("value file magic, version, or flags are unsupported")
    if record_count != len(descriptors) or record_count != info["record_count"]:
        fail("value file record count mismatch")
    if payload_bytes != info["payload_bytes"] or HEADER.size + payload_bytes != len(contents):
        fail("value file payload length mismatch")
    if (
        fixture_digest.hex() != fixture_sha256
        or context_manifest_digest.hex() != context_manifest_sha256
    ):
        fail("value file fixture/context-manifest hash mismatch")
    manifest_records = [
        {
            "case_id": item["case_id"],
            "oracle": item["oracle"],
            "values": [0j] * item["value_count"],
            "metadata": {},
        }
        for item in descriptors
    ]
    manifest = _case_manifest(manifest_records)
    if manifest != manifest_digest.hex() or manifest != info["case_manifest_sha256"]:
        fail("value file case-manifest hash mismatch")

    expected_offset = 0
    result: list[dict[str, Any]] = []
    payload = memoryview(contents)[HEADER.size:]
    for position, item in enumerate(descriptors):
        offset = item["offset_bytes"]
        byte_count = item["value_count"] * PAIR.size
        if offset != expected_offset:
            kind = "overlap" if offset < expected_offset else "gap"
            fail(f"record[{position}] has a payload {kind}")
        if offset + byte_count > len(payload):
            fail(f"record[{position}] exceeds the payload")
        values: list[complex] = []
        for value_offset in range(offset, offset + byte_count, PAIR.size):
            real, imaginary = PAIR.unpack_from(payload, value_offset)
            if not math.isfinite(real) or not math.isfinite(imaginary):
                fail(f"record[{position}] contains a non-finite value")
            values.append(complex(real, imaginary))
        expected_offset += byte_count
        result.append({**item, "values": values})
    if expected_offset != len(payload):
        fail("unreferenced bytes remain in the value payload")
    return result


def _expected_case_ids(fixture: dict[str, Any]) -> list[str]:
    return [case["case_id"] for case in expanded_cases(fixture)]


def generate_analytic_reference(
    fixture_path: Path,
    context_manifest_path: Path,
    output_json: Path,
    output_bin: Path,
) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(fixture_path, context_manifest_path)
    records = analytic_records(fixture)
    binary, descriptors = write_value_file(
        output_bin,
        fixture_sha256,
        fixture["compiler_context_manifest"]["sha256"],
        records,
    )
    reference = {
        "schema_version": REFERENCE_SCHEMA,
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": fixture[
            "compiler_context_manifest"
        ]["sha256"],
        "binary": binary,
        "analytic": {"status": "complete", "records": descriptors},
        "ant": {
            "status": "missing",
            "required_for_qualification": True,
            "producer": None,
            "independent_encoding": False,
            "input_kind": None,
            "records": [],
        },
    }
    write_json(output_json, reference)
    return reference


def _load_reference(
    path: Path,
    binary_path: Path,
    fixture: dict[str, Any],
    fixture_sha256: str,
    require_ant: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = load_json(path)
    _expect_keys(
        reference,
        {
            "schema_version",
            "fixture_id",
            "fixture_sha256",
            "compiler_context_manifest_sha256",
            "binary",
            "analytic",
            "ant",
        },
        "CPU reference",
    )
    if reference["schema_version"] != REFERENCE_SCHEMA:
        fail("unsupported CPU reference schema")
    if reference["fixture_id"] != fixture["fixture_id"]:
        fail("CPU reference fixture identifier mismatch")
    if reference["fixture_sha256"] != fixture_sha256:
        fail("CPU reference fixture hash mismatch")
    if reference["compiler_context_manifest_sha256"] != fixture[
        "compiler_context_manifest"
    ]["sha256"]:
        fail("CPU reference compiler context manifest hash mismatch")
    analytic = _expect_keys(reference["analytic"], {"status", "records"}, "CPU reference analytic")
    if analytic["status"] != "complete":
        fail("analytic reference is incomplete")
    analytic_descriptors = _descriptor_records(
        analytic["records"], "CPU reference analytic.records", "analytic"
    )
    ant = _expect_keys(
        reference["ant"],
        {
            "status",
            "required_for_qualification",
            "producer",
            "independent_encoding",
            "input_kind",
            "records",
        },
        "CPU reference ANT",
    )
    if ant["required_for_qualification"] is not True:
        fail("CPU reference must require ANT for qualification")
    if ant["status"] == "missing":
        if ant["records"] or ant["producer"] is not None or ant["independent_encoding"] is not False or ant["input_kind"] is not None:
            fail("missing ANT section contains result claims")
        if require_ant:
            fail("independent ANT records are missing")
        ant_descriptors: list[dict[str, Any]] = []
    elif ant["status"] == "complete":
        if not isinstance(ant["producer"], str) or not ant["producer"]:
            fail("complete ANT section has no producer identity")
        if ant["independent_encoding"] is not True or ant["input_kind"] != "clear_fixture_values":
            fail("ANT section does not attest independent clear-input encoding")
        ant_descriptors = _descriptor_records(
            ant["records"], "CPU reference ant.records", "ant"
        )
    else:
        fail("CPU reference ANT status is unsupported")
    records = read_value_file(
        binary_path,
        reference["binary"],
        analytic_descriptors + ant_descriptors,
        fixture_sha256,
        fixture["compiler_context_manifest"]["sha256"],
    )
    expected = _expected_case_ids(fixture)
    observed_analytic = [item["case_id"] for item in records if item["oracle"] == "analytic"]
    if observed_analytic != expected:
        fail("analytic records do not exactly match fixture cases")
    if ant["status"] == "complete":
        observed_ant = [item["case_id"] for item in records if item["oracle"] == "ant"]
        if observed_ant != expected:
            fail("ANT records do not exactly match fixture cases")
    return reference, records


def _load_provider(
    path: Path,
    binary_path: Path,
    fixture: dict[str, Any],
    fixture_sha256: str,
    provider_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    provider = load_json(path)
    _expect_keys(
        provider,
        {
            "schema_version",
            "provider",
            "fixture_id",
            "fixture_sha256",
            "compiler_context_manifest_sha256",
            "producer",
            "independent_encoding",
            "input_kind",
            "binary",
            "records",
        },
        "provider result",
    )
    if provider["schema_version"] != PROVIDER_SCHEMA or provider["provider"] != provider_name:
        fail(f"provider result is not a {provider_name} ordinary-CKKS result")
    if provider["fixture_id"] != fixture["fixture_id"] or provider["fixture_sha256"] != fixture_sha256:
        fail("provider fixture identity/hash mismatch")
    if provider["compiler_context_manifest_sha256"] != fixture[
        "compiler_context_manifest"
    ]["sha256"]:
        fail("provider compiler context manifest hash mismatch")
    if not isinstance(provider["producer"], str) or not provider["producer"]:
        fail("provider result has no producer identity")
    if provider["independent_encoding"] is not True or provider["input_kind"] != "clear_fixture_values":
        fail("provider result must independently encode clear fixture values")
    descriptors = _descriptor_records(provider["records"], "provider records", provider_name)
    records = read_value_file(
        binary_path,
        provider["binary"],
        descriptors,
        fixture_sha256,
        fixture["compiler_context_manifest"]["sha256"],
    )
    if [record["case_id"] for record in records] != _expected_case_ids(fixture):
        fail("provider records do not exactly match fixture cases")
    return provider, records


def ingest_raw_provider(
    fixture_path: Path,
    context_manifest_path: Path,
    provider_name: str,
    raw_json: Path,
    output_json: Path,
    output_bin: Path,
) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(
        fixture_path, context_manifest_path
    )
    raw = load_json(raw_json)
    _expect_keys(
        raw,
        {
            "schema_version",
            "provider",
            "fixture_id",
            "fixture_sha256",
            "compiler_context_manifest_sha256",
            "producer",
            "independent_encoding",
            "input_kind",
            "records",
        },
        "raw provider result",
    )
    if raw["schema_version"] != RAW_PROVIDER_SCHEMA:
        fail("raw provider result schema is unsupported")
    if raw["provider"] != provider_name:
        fail("raw provider identity does not match the requested provider")
    if raw["fixture_id"] != fixture["fixture_id"]:
        fail("raw provider fixture identifier mismatch")
    if raw["fixture_sha256"] != fixture_sha256:
        fail("raw provider fixture hash mismatch")
    if raw["compiler_context_manifest_sha256"] != fixture[
        "compiler_context_manifest"
    ]["sha256"]:
        fail("raw provider compiler context manifest hash mismatch")
    if not isinstance(raw["producer"], str) or not raw["producer"]:
        fail("raw provider producer identity is missing")
    if raw["independent_encoding"] is not True:
        fail("raw provider must attest independent encoding")
    if raw["input_kind"] != "clear_fixture_values":
        fail("raw provider input kind is unsupported")
    raw_records = raw["records"]
    if not isinstance(raw_records, list):
        fail("raw provider records must be an array")
    expected_case_ids = _expected_case_ids(fixture)
    if len(raw_records) != len(expected_case_ids):
        fail("raw provider record count mismatch")
    value_count = resolved_context(fixture)["logical_slots"]
    records: list[dict[str, Any]] = []
    for position, (item, expected_case_id) in enumerate(
        zip(raw_records, expected_case_ids)
    ):
        context = f"raw provider records[{position}]"
        record = _expect_keys(item, {"case_id", "values", "metadata"}, context)
        if record["case_id"] != expected_case_id:
            fail(f"{context}.case_id does not match the frozen case order")
        if not isinstance(record["values"], list) or len(record["values"]) != value_count:
            fail(f"{context}.values length does not match compiler logical slots")
        values = [
            _complex_pair(value, f"{context}.values[{index}]")
            for index, value in enumerate(record["values"])
        ]
        if not isinstance(record["metadata"], dict):
            fail(f"{context}.metadata must be an object")
        records.append(
            {
                "case_id": expected_case_id,
                "oracle": provider_name,
                "values": values,
                "metadata": record["metadata"],
            }
        )
    binary, descriptors = write_value_file(
        output_bin,
        fixture_sha256,
        fixture["compiler_context_manifest"]["sha256"],
        records,
    )
    provider = {
        "schema_version": PROVIDER_SCHEMA,
        "provider": provider_name,
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": fixture[
            "compiler_context_manifest"
        ]["sha256"],
        "producer": raw["producer"],
        "independent_encoding": True,
        "input_kind": "clear_fixture_values",
        "binary": binary,
        "records": descriptors,
    }
    write_json(output_json, provider)
    return provider


def ingest_ant(
    fixture_path: Path,
    context_manifest_path: Path,
    analytic_json: Path,
    analytic_bin: Path,
    ant_json: Path,
    ant_bin: Path,
    output_json: Path,
    output_bin: Path,
) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(
        fixture_path, context_manifest_path
    )
    reference, analytic = _load_reference(
        analytic_json, analytic_bin, fixture, fixture_sha256, require_ant=False
    )
    if reference["ant"]["status"] != "missing":
        fail("ANT ingestion input is already complete")
    provider, ant = _load_provider(
        ant_json, ant_bin, fixture, fixture_sha256, "ant"
    )
    combined = [
        {
            "case_id": item["case_id"],
            "oracle": item["oracle"],
            "values": item["values"],
            "metadata": item["metadata"],
        }
        for item in analytic + ant
    ]
    binary, descriptors = write_value_file(
        output_bin,
        fixture_sha256,
        fixture["compiler_context_manifest"]["sha256"],
        combined,
    )
    split = len(analytic)
    result = {
        "schema_version": REFERENCE_SCHEMA,
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": fixture[
            "compiler_context_manifest"
        ]["sha256"],
        "binary": binary,
        "analytic": {"status": "complete", "records": descriptors[:split]},
        "ant": {
            "status": "complete",
            "required_for_qualification": True,
            "producer": provider["producer"],
            "independent_encoding": True,
            "input_kind": "clear_fixture_values",
            "records": descriptors[split:],
        },
    }
    write_json(output_json, result)
    return result


def _relative_close(observed: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(observed) and abs(observed - expected) <= tolerance * max(abs(expected), 1.0)


def validate_observed_metadata(
    observed_value: Any,
    contract: dict[str, Any],
    tolerances: dict[str, Any],
    fixture: dict[str, Any],
    first_data_chain_index: int,
) -> tuple[bool, list[str]]:
    required = {
        "object_kind",
        "ace_level",
        "active_q_count",
        "phantom_chain_index",
        "scale_degree",
        "raw_scale",
        "slots",
        "ciphertext_size",
        "ntt",
    }
    raw_rule = contract["raw_scale"]
    optional = {"dropped_modulus"} if raw_rule["kind"] == "divide_by_reported_dropped_modulus" else set()
    try:
        observed = _expect_keys(observed_value, required, "observed metadata", optional)
    except OrdinaryCkksError as error:
        return False, [str(error)]
    errors: list[str] = []
    context = resolved_context(fixture)
    try:
        active_q_count = resolve_active_q_count(
            contract["active_q_count"], fixture, "metadata contract.active_q_count"
        )
    except OrdinaryCkksError as error:
        return False, [str(error)]
    expected = {
        "object_kind": contract["object_kind"],
        "ace_level": active_q_count,
        "active_q_count": active_q_count,
        "phantom_chain_index": (
            first_data_chain_index + context["data_q_count"] - active_q_count
        ),
        "scale_degree": contract["scale_degree"],
        "slots": context["logical_slots"],
        "ciphertext_size": contract["ciphertext_size"],
        "ntt": contract["ntt"],
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            errors.append(
                f"{key}: expected {expected_value!r}, got {observed[key]!r}"
            )
    try:
        raw_scale = _number(observed["raw_scale"], "observed metadata.raw_scale")
        scale_bits = context["scaling_modulus_bits"]
        if raw_rule["kind"] == "exact_scale_degree":
            expected_scale = math.ldexp(
                1.0, contract["scale_degree"] * scale_bits
            )
        else:
            dropped = _integer(observed.get("dropped_modulus"), "observed metadata.dropped_modulus", 2)
            if dropped.bit_length() != scale_bits:
                errors.append(
                    "dropped_modulus: expected "
                    f"{scale_bits} bits, got {dropped.bit_length()}"
                )
            expected_scale = math.ldexp(
                1.0, raw_rule["numerator_scale_degree"] * scale_bits
            ) / dropped
        if not _relative_close(
            raw_scale, expected_scale, tolerances["metadata_scale_relative"]
        ):
            errors.append(
                f"raw_scale: expected {expected_scale!r}, got {raw_scale!r}"
            )
    except OrdinaryCkksError as error:
        errors.append(str(error))
    return not errors, errors


def comparison_metrics(
    observed: Sequence[complex], reference: Sequence[complex], tolerances: dict[str, Any]
) -> dict[str, Any]:
    if len(observed) != len(reference) or not observed:
        fail("comparison length mismatch or empty comparison")
    errors = [abs(value - expected) for value, expected in zip(observed, reference)]
    if not all(math.isfinite(error) for error in errors):
        fail("comparison produced a non-finite error")
    max_index = max(range(len(errors)), key=errors.__getitem__)
    maximum = errors[max_index]
    mae = math.fsum(errors) / len(errors)
    rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    reference_rms = math.sqrt(
        math.fsum(abs(value) ** 2 for value in reference) / len(reference)
    )
    floor = tolerances["relative_metric_floor"]
    relative = [
        (error / abs(expected), index)
        for index, (error, expected) in enumerate(zip(errors, reference))
        if abs(expected) >= floor
    ]
    if relative:
        maximum_relative, relative_index = max(relative)
    else:
        maximum_relative, relative_index = None, None
    threshold_failures = [
        index
        for index, (error, expected) in enumerate(zip(errors, reference))
        if error > tolerances["absolute"] + tolerances["relative"] * abs(expected)
    ]
    passed = not threshold_failures and maximum <= tolerances["hard_maximum_absolute"]
    if rmse == 0.0:
        precision_bits = None
        precision_is_infinite = True
    else:
        precision_bits = -math.log2(rmse / max(reference_rms, floor))
        precision_is_infinite = False
    return {
        "pass": passed,
        "compared_count": len(errors),
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_index": max_index,
        "maximum_relative_error": maximum_relative,
        "maximum_relative_error_index": relative_index,
        "relative_metric_count": len(relative),
        "relative_metric_excluded_count": len(errors) - len(relative),
        "mae": mae,
        "rmse": rmse,
        "reference_rms": reference_rms,
        "estimated_precision_bits": precision_bits,
        "estimated_precision_is_infinite": precision_is_infinite,
        "mixed_tolerance_failure_count": len(threshold_failures),
        "first_mixed_tolerance_failure_index": (
            threshold_failures[0] if threshold_failures else None
        ),
        "hard_maximum_absolute": tolerances["hard_maximum_absolute"],
    }


def compare_oracles(
    observed: Sequence[complex],
    analytic: Sequence[complex],
    ant: Sequence[complex],
    tolerances: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare one provider output to two independent references."""
    return {
        "gpu_vs_analytic": comparison_metrics(observed, analytic, tolerances),
        "gpu_vs_ant": comparison_metrics(observed, ant, tolerances),
    }


def verify_ant_reference(
    fixture_path: Path,
    context_manifest_path: Path,
    cpu_json: Path,
    cpu_bin: Path,
    output_json: Path,
) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(
        fixture_path, context_manifest_path
    )
    _, records = _load_reference(
        cpu_json, cpu_bin, fixture, fixture_sha256, require_ant=True
    )
    analytic = {
        record["case_id"]: record["values"]
        for record in records
        if record["oracle"] == "analytic"
    }
    ant = {
        record["case_id"]: record["values"]
        for record in records
        if record["oracle"] == "ant"
    }
    reports = []
    overall = True
    for case_id in _expected_case_ids(fixture):
        metrics = comparison_metrics(
            ant[case_id], analytic[case_id], fixture["tolerances"]
        )
        overall = overall and metrics["pass"]
        reports.append({"case_id": case_id, **metrics})
    result = {
        "schema_version": "ace.phantom.ordinary_ckks.ant-verification/1.0.0",
        "status": "pass" if overall else "fail",
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": fixture[
            "compiler_context_manifest"
        ]["sha256"],
        "case_count": len(reports),
        "passing_case_count": sum(report["pass"] for report in reports),
        "cases": reports,
    }
    write_json(output_json, result)
    return result


def compare_results(
    fixture_path: Path,
    context_manifest_path: Path,
    cpu_json: Path,
    cpu_bin: Path,
    gpu_json: Path,
    gpu_bin: Path,
    output_json: Path,
) -> dict[str, Any]:
    fixture, fixture_sha256 = load_fixture(
        fixture_path, context_manifest_path
    )
    reference, cpu_records = _load_reference(
        cpu_json, cpu_bin, fixture, fixture_sha256, require_ant=True
    )
    gpu_provider, gpu_records = _load_provider(
        gpu_json, gpu_bin, fixture, fixture_sha256, "phantom"
    )
    maps: dict[str, dict[str, dict[str, Any]]] = {"analytic": {}, "ant": {}, "phantom": {}}
    for record in cpu_records + gpu_records:
        maps[record["oracle"]][record["case_id"]] = record
    cases = expanded_cases(fixture)
    tolerances = fixture["tolerances"]
    full_case = maps["phantom"].get("encode_real_f64_full.distinct")
    if full_case is None:
        fail("Phantom result is missing the full-chain coordinate case")
    full_metadata = full_case["metadata"]
    if not isinstance(full_metadata, dict):
        fail("Phantom full-chain coordinate metadata must be an object")
    first_data_chain_index = _integer(
        full_metadata.get("phantom_chain_index"),
        "Phantom full-chain coordinate chain index",
        0,
    )
    reports: list[dict[str, Any]] = []
    overall = True
    for case in cases:
        case_id = case["case_id"]
        analytic = maps["analytic"][case_id]
        ant = maps["ant"][case_id]
        gpu = maps["phantom"][case_id]
        regenerated = evaluate_expression(case["expected"], fixture)
        if analytic["values"] != regenerated:
            fail(f"analytic record {case_id} does not match the frozen expression")
        oracle_metrics = compare_oracles(
            gpu["values"], analytic["values"], ant["values"], tolerances
        )
        analytic_metrics = oracle_metrics["gpu_vs_analytic"]
        ant_metrics = oracle_metrics["gpu_vs_ant"]
        contract = fixture["metadata_contracts"][case["metadata_contract"]]
        metadata_pass, metadata_errors = validate_observed_metadata(
            gpu["metadata"],
            contract,
            tolerances,
            fixture,
            first_data_chain_index,
        )
        case_pass = analytic_metrics["pass"] and ant_metrics["pass"] and metadata_pass
        overall = overall and case_pass
        reports.append(
            {
                "case_id": case_id,
                "operation": case["operation"],
                "form": case["form"],
                "alias": case["alias"],
                "metadata_contract": case["metadata_contract"],
                "metadata_pass": metadata_pass,
                "metadata_errors": metadata_errors,
                "gpu_vs_analytic": analytic_metrics,
                "gpu_vs_ant": ant_metrics,
                "pass": case_pass,
            }
        )
    result = {
        "schema_version": COMPARE_SCHEMA,
        "status": "pass" if overall else "fail",
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": fixture[
            "compiler_context_manifest"
        ]["sha256"],
        "cpu_reference_sha256": sha256_path(cpu_json),
        "cpu_values_sha256": sha256_path(cpu_bin),
        "gpu_result_sha256": sha256_path(gpu_json),
        "gpu_values_sha256": sha256_path(gpu_bin),
        "ant": {
            "status": reference["ant"]["status"],
            "producer": reference["ant"]["producer"],
            "independent_encoding": reference["ant"]["independent_encoding"],
        },
        "phantom": {
            "producer": gpu_provider["producer"],
            "independent_encoding": gpu_provider["independent_encoding"],
        },
        "case_count": len(reports),
        "passing_case_count": sum(report["pass"] for report in reports),
        "cases": reports,
    }
    write_json(output_json, result)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_invocation = subparsers.add_parser(
        "record-qualification-invocation"
    )
    record_invocation.add_argument("--output-json", required=True, type=Path)
    record_invocation.add_argument("argv", nargs=argparse.REMAINDER)

    bind = subparsers.add_parser("bind-fixture")
    bind.add_argument("--fixture", required=True, type=Path)
    bind.add_argument("--context-manifest", required=True, type=Path)
    bind.add_argument("--compiler-invocation", required=True, type=Path)
    bind.add_argument("--post-ckks-air", required=True, type=Path)
    bind.add_argument("--output-json", required=True, type=Path)

    validate = subparsers.add_parser("validate-fixture")
    validate.add_argument("--fixture", required=True, type=Path)
    validate.add_argument("--context-manifest", required=True, type=Path)
    validate.add_argument("--compiler-invocation", required=True, type=Path)
    validate.add_argument("--post-ckks-air", required=True, type=Path)

    analytic = subparsers.add_parser("generate-analytic")
    analytic.add_argument("--fixture", required=True, type=Path)
    analytic.add_argument("--context-manifest", required=True, type=Path)
    analytic.add_argument("--output-json", required=True, type=Path)
    analytic.add_argument("--output-bin", required=True, type=Path)

    provider = subparsers.add_parser("validate-provider")
    provider.add_argument("--fixture", required=True, type=Path)
    provider.add_argument("--context-manifest", required=True, type=Path)
    provider.add_argument("--provider", required=True, choices=("ant", "phantom"))
    provider.add_argument("--provider-json", required=True, type=Path)
    provider.add_argument("--provider-bin", required=True, type=Path)

    raw_provider = subparsers.add_parser("ingest-raw-provider")
    raw_provider.add_argument("--fixture", required=True, type=Path)
    raw_provider.add_argument("--context-manifest", required=True, type=Path)
    raw_provider.add_argument(
        "--provider", required=True, choices=("ant", "phantom")
    )
    raw_provider.add_argument("--raw-json", required=True, type=Path)
    raw_provider.add_argument("--output-json", required=True, type=Path)
    raw_provider.add_argument("--output-bin", required=True, type=Path)

    ingest = subparsers.add_parser("ingest-ant")
    ingest.add_argument("--fixture", required=True, type=Path)
    ingest.add_argument("--context-manifest", required=True, type=Path)
    ingest.add_argument("--analytic-json", required=True, type=Path)
    ingest.add_argument("--analytic-bin", required=True, type=Path)
    ingest.add_argument("--ant-json", required=True, type=Path)
    ingest.add_argument("--ant-bin", required=True, type=Path)
    ingest.add_argument("--output-json", required=True, type=Path)
    ingest.add_argument("--output-bin", required=True, type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--fixture", required=True, type=Path)
    compare.add_argument("--context-manifest", required=True, type=Path)
    compare.add_argument("--cpu-json", required=True, type=Path)
    compare.add_argument("--cpu-bin", required=True, type=Path)
    compare.add_argument("--gpu-json", required=True, type=Path)
    compare.add_argument("--gpu-bin", required=True, type=Path)
    compare.add_argument("--output-json", required=True, type=Path)

    verify_ant = subparsers.add_parser("verify-ant-reference")
    verify_ant.add_argument("--fixture", required=True, type=Path)
    verify_ant.add_argument("--context-manifest", required=True, type=Path)
    verify_ant.add_argument("--cpu-json", required=True, type=Path)
    verify_ant.add_argument("--cpu-bin", required=True, type=Path)
    verify_ant.add_argument("--output-json", required=True, type=Path)

    validate_ownership = subparsers.add_parser("validate-ownership")
    validate_ownership.add_argument("--fixture", required=True, type=Path)
    validate_ownership.add_argument(
        "--ownership-json", required=True, type=Path
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "record-qualification-invocation":
            record = record_qualification_invocation(
                arguments.output_json, arguments.argv
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "normalized_argv_sha256": record[
                            "normalized_argv_sha256"
                        ],
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "bind-fixture":
            fixture = bind_fixture(
                arguments.fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
                arguments.output_json,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "fixture_id": fixture["fixture_id"],
                        "qualification_bindings": fixture[
                            "qualification_bindings"
                        ],
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "validate-fixture":
            fixture, digest = load_fixture(
                arguments.fixture, arguments.context_manifest
            )
            verify_qualification_bindings(
                fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "fixture_id": fixture["fixture_id"],
                        "fixture_sha256": digest,
                        "expanded_case_count": len(expanded_cases(fixture)),
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "generate-analytic":
            result = generate_analytic_reference(
                arguments.fixture,
                arguments.context_manifest,
                arguments.output_json,
                arguments.output_bin,
            )
            print(json.dumps({"status": "incomplete", "ant": result["ant"]}, sort_keys=True))
        elif arguments.command == "validate-provider":
            fixture, digest = load_fixture(
                arguments.fixture, arguments.context_manifest
            )
            provider, records = _load_provider(
                arguments.provider_json,
                arguments.provider_bin,
                fixture,
                digest,
                arguments.provider,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "provider": provider["provider"],
                        "record_count": len(records),
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "ingest-raw-provider":
            result = ingest_raw_provider(
                arguments.fixture,
                arguments.context_manifest,
                arguments.provider,
                arguments.raw_json,
                arguments.output_json,
                arguments.output_bin,
            )
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "provider": result["provider"],
                        "record_count": len(result["records"]),
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "ingest-ant":
            result = ingest_ant(
                arguments.fixture,
                arguments.context_manifest,
                arguments.analytic_json,
                arguments.analytic_bin,
                arguments.ant_json,
                arguments.ant_bin,
                arguments.output_json,
                arguments.output_bin,
            )
            print(json.dumps({"status": "pass", "ant": result["ant"]["status"]}, sort_keys=True))
        elif arguments.command == "verify-ant-reference":
            result = verify_ant_reference(
                arguments.fixture,
                arguments.context_manifest,
                arguments.cpu_json,
                arguments.cpu_bin,
                arguments.output_json,
            )
            print(
                json.dumps(
                    {"status": result["status"], "case_count": result["case_count"]},
                    sort_keys=True,
                )
            )
            return 0 if result["status"] == "pass" else 1
        elif arguments.command == "validate-ownership":
            fixture = load_json(arguments.fixture)
            ownership = load_json(arguments.ownership_json)
            total_iterations = validate_ownership_result(ownership, fixture)
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "total_iterations": total_iterations,
                    },
                    sort_keys=True,
                )
            )
        else:
            result = compare_results(
                arguments.fixture,
                arguments.context_manifest,
                arguments.cpu_json,
                arguments.cpu_bin,
                arguments.gpu_json,
                arguments.gpu_bin,
                arguments.output_json,
            )
            print(json.dumps({"status": result["status"], "case_count": result["case_count"]}, sort_keys=True))
            return 0 if result["status"] == "pass" else 1
    except OrdinaryCkksError as error:
        print(f"ordinary CKKS validation failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
