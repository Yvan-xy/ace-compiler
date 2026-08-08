#!/usr/bin/env python3
"""Bind and materialize deterministic provider-neutral retained CKKS oracles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Iterable, Sequence


FIXTURE_SCHEMA = "ace.phantom.retained_ckks.fixture/1.0.0"
ANALYTIC_SCHEMA = "ace.phantom.retained_ckks.analytic/1.0.0"
EXACT_SCHEMA = "ace.phantom.retained_ckks.exact-rns/1.0.0"
INVOCATION_SCHEMA = "ace.phantom.compiler-invocation/1.0.0"
DECODED_MAGIC = b"ACERCK01"
EXACT_MAGIC = b"ACERNS01"
MASK64 = (1 << 64) - 1
RETAINED_RESOURCE_SCHEMA_VERSION = 2


class RetainedFixtureError(ValueError):
    pass


def fail(message: str) -> None:
    raise RetainedFixtureError(message)


def _reject_constant(value: str) -> None:
    fail(f"non-finite JSON constant is forbidden: {value}")


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read strict JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
        output.flush()
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value) + b"\n")


def _expect_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(value)
    if observed != expected:
        fail(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _integer(value: Any, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{context} must be at least {minimum}")
    return value


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{context} must be a lowercase SHA-256 digest")
    return value


def validate_context_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    required = {
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
    }
    _expect_keys(manifest, required, "compiler context manifest")
    if manifest["schema_version"] != 1 or manifest["packing"] != "full":
        fail("unsupported compiler context manifest schema or packing")
    degree = _integer(manifest["polynomial_degree"], "polynomial_degree", 2)
    if degree & (degree - 1):
        fail("polynomial_degree must be a power of two")
    slots = _integer(manifest["logical_slot_capacity"], "logical slots", 1)
    if slots != degree // 2:
        fail("logical_slot_capacity must equal polynomial_degree / 2")
    data_q = manifest["data_q_bit_sizes"]
    special_p = manifest["special_p_bit_sizes"]
    if not isinstance(data_q, list) or not data_q:
        fail("data_q_bit_sizes must be a nonempty array")
    if not isinstance(special_p, list) or not special_p:
        fail("special_p_bit_sizes must be a nonempty array")
    for name, values in (("data_q_bit_sizes", data_q), ("special_p_bit_sizes", special_p)):
        for index, bits in enumerate(values):
            bits = _integer(bits, f"{name}[{index}]", 2)
            if bits > 60:
                fail(f"{name}[{index}] exceeds 60 bits")
    input_level = _integer(manifest["input_level"], "input_level", 1)
    if input_level > len(data_q):
        fail("input_level exceeds the data-Q count")
    if manifest["first_modulus_bits"] != data_q[0]:
        fail("first_modulus_bits disagrees with the first data-Q entry")
    scaling_bits = _integer(
        manifest["scaling_modulus_bits"], "scaling_modulus_bits", 2
    )
    if any(bits != scaling_bits for bits in data_q[1:]):
        fail("data-Q entries after q0 must use scaling_modulus_bits")
    q_part_count = _integer(manifest["q_part_count"], "q_part_count", 1)
    if q_part_count > len(special_p):
        fail("q_part_count exceeds the special-P count")
    _integer(manifest["hamming_weight"], "hamming_weight", 1)
    _integer(manifest["security_level"], "security_level", 0)
    if manifest["resource_schema_version"] != RETAINED_RESOURCE_SCHEMA_VERSION:
        fail(
            "resource_schema_version must match the retained resource ABI "
            f"version {RETAINED_RESOURCE_SCHEMA_VERSION}"
        )
    return {
        "polynomial_degree": degree,
        "logical_slots": slots,
        "full_data_q_count": len(data_q),
        "input_active_q_count": input_level,
        "data_q_bit_sizes": tuple(data_q),
        "special_p_bit_sizes": tuple(special_p),
        "scaling_modulus_bits": scaling_bits,
    }


def validate_template(fixture: dict[str, Any], *, require_bound: bool) -> None:
    required = {
        "schema_version",
        "fixture_id",
        "description",
        "qualification_bindings",
        "determinism",
        "context_rules",
        "inputs",
        "rotate_batch_steps",
        "production_rotation_batches",
        "production_rotation_source",
        "monomial_powers",
        "decoded_cases",
        "exact_algebraic_cases",
        "metadata_contract",
        "ownership",
        "runtime_rejections",
        "tolerances",
        "decoded_binary_format",
        "exact_binary_format",
    }
    _expect_keys(fixture, required, "retained fixture")
    if fixture["schema_version"] != FIXTURE_SCHEMA:
        fail("unsupported retained fixture schema")
    if fixture["fixture_id"] != "retained_ckks_v1":
        fail("unexpected retained fixture identifier")
    determinism = fixture["determinism"]
    _expect_keys(determinism, {"seed", "generator"}, "determinism")
    _integer(determinism["seed"], "deterministic seed", 0)
    if determinism["generator"] != "splitmix64-float53-complex-v1":
        fail("unsupported deterministic generator")
    if fixture["rotate_batch_steps"] != [5, 0, -7, 5]:
        fail("rotate_batch_steps must preserve [5, 0, -7, 5]")
    production_batches = fixture["production_rotation_batches"]
    if not isinstance(production_batches, list):
        fail("production_rotation_batches must be an array")
    if require_bound and not production_batches:
        fail("a bound fixture must contain production rotation batches")
    for batch_index, batch in enumerate(production_batches):
        if not isinstance(batch, list) or not batch:
            fail(f"production_rotation_batches[{batch_index}] must be nonempty")
        for step_index, step in enumerate(batch):
            _integer(step, f"production_rotation_batches[{batch_index}][{step_index}]")
    production_source = fixture["production_rotation_source"]
    _expect_keys(
        production_source,
        {"kind", "post_ckks_air_sha256"},
        "production_rotation_source",
    )
    if production_source["kind"] != "audited_default_post_ckks_air":
        fail("production rotation source kind changed")
    binding_status = fixture["qualification_bindings"].get("status")
    if binding_status == "unbound" and production_source["post_ckks_air_sha256"] is not None:
        fail(
            "an unbound fixture must not predeclare its production post-CKKS "
            "AIR hash"
        )
    if binding_status == "bound" and production_source["post_ckks_air_sha256"] is None:
        fail("a bound fixture must bind its production post-CKKS AIR hash")
    if production_source["post_ckks_air_sha256"] is not None:
        _sha256(
            production_source["post_ckks_air_sha256"],
            "production_rotation_source.post_ckks_air_sha256",
        )
    if fixture["monomial_powers"] != [
        "0",
        "N/2",
        "N",
        "3N/2",
        "2N-1",
        "2N+1",
    ]:
        fail("the symbolic monomial power matrix changed")
    if fixture["exact_algebraic_cases"] != [
        "exact_algebraic.raise_mod",
        "exact_algebraic.mul_mono.0",
        "exact_algebraic.mul_mono.N_over_2",
        "exact_algebraic.mul_mono.N",
        "exact_algebraic.mul_mono.3N_over_2",
        "exact_algebraic.mul_mono.2N_minus_1",
        "exact_algebraic.mul_mono.2N_plus_1",
        "exact_algebraic.mul_mono.inverse_composition",
    ]:
        fail("the exact algebraic case matrix changed")
    expected_decoded_cases = [
        {"id": "conjugate.bounded_nonperiodic", "oracle": ["analytic", "ant"]},
        {"id": "conjugate_twice.bounded_nonperiodic", "oracle": ["analytic", "ant"]},
        {"id": "rotate_batch.bounded_nonperiodic", "oracle": ["analytic", "ant"]},
        {"id": "raise_mod.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.0.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.N_over_2.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.N.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.3N_over_2.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.2N_minus_1.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.2N_plus_1.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "mul_mono.inverse_composition.bounded_nonperiodic", "oracle": ["ant"]},
        {"id": "composite.bounded_nonperiodic", "oracle": ["ant"]},
    ]
    if fixture["decoded_cases"] != expected_decoded_cases:
        fail("the decoded case matrix changed")
    bindings = fixture["qualification_bindings"]
    binding_status = bindings.get("status")
    if require_bound and binding_status != "bound":
        fail("retained fixture must be bound before oracle generation")
    if binding_status == "bound":
        _expect_keys(
            bindings,
            {
                "status",
                "compiler_context_manifest_sha256",
                "normalized_compiler_command_sha256",
                "post_ckks_air_sha256",
            },
            "qualification_bindings",
        )
        for key in bindings:
            if key != "status":
                _sha256(bindings[key], f"qualification_bindings.{key}")
    elif binding_status == "unbound":
        _expect_keys(bindings, {"status", "required"}, "qualification_bindings")
        if bindings["required"] != [
            "compiler_context_manifest_sha256",
            "normalized_compiler_command_sha256",
            "post_ckks_air_sha256",
        ]:
            fail("unbound qualification requirements changed")
    else:
        fail("qualification_bindings.status is invalid")
    expected_context_rules = {
        "polynomial_degree": "compiler_context.polynomial_degree",
        "logical_slots": "compiler_context.logical_slot_capacity",
        "full_data_q_count": "len(compiler_context.data_q_bit_sizes)",
        "input_active_q_count": "compiler_context.input_level",
        "ordered_data_q_bits": "compiler_context.data_q_bit_sizes",
        "ordered_special_p_bits": "compiler_context.special_p_bit_sizes",
    }
    if fixture["context_rules"] != expected_context_rules:
        fail("context derivation rules changed")
    expected_inputs = [
        {"id": "zero", "recipe": "zero"},
        {"id": "impulse", "recipe": "seeded_impulse"},
        {"id": "ramp", "recipe": "centered_complex_ramp"},
        {"id": "alternating", "recipe": "alternating_sign"},
        {"id": "bounded_nonperiodic", "recipe": "seeded_bounded_nonperiodic"},
    ]
    if fixture["inputs"] != expected_inputs:
        fail("deterministic input recipes changed")
    expected_metadata_contract = {
        "scale_degree": 1,
        "ciphertext_size": 2,
        "logical_slots": "compiler_context.logical_slot_capacity",
        "ntt": True,
        "preserve_raw_scale": True,
        "preserve_source_for_out_of_place": True,
        "ordinary_result_active_q_count": "compiler_context.input_level",
        "raised_result_active_q_count": "len(compiler_context.data_q_bit_sizes)",
    }
    if fixture["metadata_contract"] != expected_metadata_contract:
        fail("metadata contract changed")
    if fixture["ownership"] != {
        "iterations": 100,
        "batch_outputs_are_independent": True,
        "free_order": [3, 1, 0, 2],
    }:
        fail("ownership contract changed")
    expected_runtime_rejections = [
        {
            "id": "conjugate_missing_key",
            "diagnostic": "CONJUGATE_KEY_MISSING",
            "manifest": "keyless-conjugation",
        },
        {
            "id": "rotate_batch_missing_nonzero_key",
            "diagnostic": "RESOURCE_ROTATE_BATCH",
            "manifest": "keyless-rotation",
        },
        {
            "id": "rotate_batch_source_overlap",
            "diagnostic": "ROTATE_BATCH_ALIAS",
            "manifest": "production",
        },
        {
            "id": "raise_alias",
            "diagnostic": "RAISE_MOD_ALIAS",
            "manifest": "production",
        },
        {
            "id": "raise_size",
            "diagnostic": "RAISE_MOD_SIZE",
            "manifest": "production",
        },
        {
            "id": "raise_chain",
            "diagnostic": "RAISE_MOD_SOURCE_LEVEL",
            "manifest": "production",
        },
        {
            "id": "raise_target",
            "diagnostic": "RAISE_MOD_TARGET",
            "manifest": "production",
        },
        {
            "id": "raise_malformed_metadata",
            "diagnostic": "RAISE_MOD_SOURCE",
            "manifest": "production",
        },
        {
            "id": "mul_mono_undeclared",
            "diagnostic": "MUL_MONO_RESOURCE",
            "manifest": "production",
        },
    ]
    if fixture["runtime_rejections"] != expected_runtime_rejections:
        fail("runtime rejection identifiers, diagnostics, or linkage changed")
    _expect_keys(
        fixture["tolerances"],
        {
            "primitive_absolute",
            "primitive_relative",
            "composite_absolute",
            "composite_relative",
            "hard_maximum_absolute",
            "relative_metric_floor",
            "exact_mismatches",
        },
        "tolerances",
    )
    expected_tolerances = {
        "primitive_absolute": 1e-4,
        "primitive_relative": 1e-3,
        "composite_absolute": 5e-4,
        "composite_relative": 2e-3,
        "hard_maximum_absolute": 5e-3,
        "relative_metric_floor": 1e-12,
        "exact_mismatches": 0,
    }
    if fixture["tolerances"] != expected_tolerances:
        fail("reviewed numerical tolerances changed")
    _expect_keys(
        fixture["decoded_binary_format"],
        {"id", "magic_ascii", "endianness", "encoding"},
        "decoded_binary_format",
    )
    _expect_keys(
        fixture["exact_binary_format"],
        {"id", "magic_ascii", "endianness", "encoding"},
        "exact_binary_format",
    )
    if fixture["decoded_binary_format"]["magic_ascii"] != DECODED_MAGIC.decode():
        fail("decoded binary magic changed")
    if (
        fixture["decoded_binary_format"]["id"]
        != "ace.retained_ckks.complex_float64le/1.0.0"
        or fixture["decoded_binary_format"]["endianness"] != "little"
    ):
        fail("decoded binary format changed")
    if fixture["exact_binary_format"]["magic_ascii"] != EXACT_MAGIC.decode():
        fail("exact binary magic changed")
    if (
        fixture["exact_binary_format"]["id"]
        != "ace.retained_ckks.rns_uint64le/1.0.0"
        or fixture["exact_binary_format"]["endianness"] != "little"
    ):
        fail("exact binary format changed")


def _load_invocation(path: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    invocation = load_json(path)
    _expect_keys(
        invocation,
        {"schema_version", "argv", "normalized_argv_sha256"},
        "compiler invocation",
    )
    if invocation["schema_version"] != INVOCATION_SCHEMA:
        fail("unsupported compiler invocation schema")
    if (
        not isinstance(invocation["argv"], list)
        or not invocation["argv"]
        or not all(isinstance(item, str) and item for item in invocation["argv"])
    ):
        fail("compiler invocation argv must be a nonempty string array")
    recorded = _sha256(
        invocation["normalized_argv_sha256"], "normalized_argv_sha256"
    )
    observed = sha256_bytes(canonical_bytes(invocation["argv"]))
    if recorded != observed:
        fail("normalized compiler command hash does not match argv")
    argv = invocation["argv"]
    if argv[0] != "tools/phantom_gpu/generate_ckks2c_probe.py":
        fail("compiler invocation has an unexpected executable")
    arguments = argv[1:]
    if len(arguments) % 2:
        fail("compiler invocation has an option without a value")
    pairs = dict(zip(arguments[0::2], arguments[1::2]))
    artifact_options = {
        "--output",
        "--post-ckks-air",
        "--context-manifest",
        "--resource-manifest",
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
        or set(pairs) != artifact_options | context_options
    ):
        fail("compiler invocation options are incomplete or duplicated")
    expected_suffixes = {
        "--output": ".cu",
        "--post-ckks-air": ".air",
        "--context-manifest": ".json",
        "--resource-manifest": ".json",
    }
    for option, suffix in expected_suffixes.items():
        artifact = Path(pairs[option])
        if artifact.is_absolute() or ".." in artifact.parts or artifact.suffix != suffix:
            fail(f"compiler invocation {option} has an unsafe artifact path")
    for option in context_options:
        if re.fullmatch(r"[0-9]+", pairs[option]) is None:
            fail(f"compiler invocation {option} must be a nonnegative integer")
    return invocation, observed, pairs


def verify_invocation_context(
    compiler_pairs: dict[str, str], context_manifest: dict[str, Any]
) -> None:
    expected = {
        "--poly-degree": context_manifest["polynomial_degree"],
        "--mul-level": len(context_manifest["data_q_bit_sizes"]),
        "--input-level": context_manifest["input_level"],
        "--security-level": context_manifest["security_level"],
        "--scaling-factor-bits": context_manifest["scaling_modulus_bits"],
        "--first-prime-bits": context_manifest["first_modulus_bits"],
        "--hamming-weight": context_manifest["hamming_weight"],
    }
    for option, manifest_value in expected.items():
        if int(compiler_pairs[option]) != manifest_value:
            fail(f"compiler invocation {option} disagrees with context manifest")


_ROTATE_BATCH_PATTERN = re.compile(
    r"CKKS[.:]{1,2}rotate_batch\b[^\n]*?\bnums=\(([^)]*)\)", re.IGNORECASE
)


def extract_production_rotation_batches(post_air_path: Path) -> list[list[int]]:
    try:
        source = post_air_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        fail(f"cannot read post-CKKS AIR {post_air_path}: {error}")
    batches: list[list[int]] = []
    for match in _ROTATE_BATCH_PATTERN.finditer(source):
        fields = [field.strip() for field in match.group(1).split(",")]
        if not fields or any(re.fullmatch(r"-?[0-9]+", field) is None for field in fields):
            fail("post-CKKS AIR contains a malformed rotate_batch RNUM")
        batches.append([int(field) for field in fields])
    if not batches:
        fail("post-CKKS AIR contains no production rotate_batch RNUM")
    return batches


def bind_fixture(
    template_path: Path,
    context_path: Path,
    invocation_path: Path,
    post_air_path: Path,
    production_air_path: Path | None = None,
) -> dict[str, Any]:
    fixture = load_json(template_path)
    validate_template(fixture, require_bound=False)
    if fixture["qualification_bindings"].get("status") != "unbound":
        fail("fixture binding requires an unbound reviewed template")
    context = load_json(context_path)
    validate_context_manifest(context)
    _, invocation_sha256, compiler_pairs = _load_invocation(invocation_path)
    verify_invocation_context(compiler_pairs, context)
    if not post_air_path.is_file() or post_air_path.stat().st_size == 0:
        fail("post-CKKS AIR must be a nonempty file")
    bound = copy.deepcopy(fixture)
    production_air_path = production_air_path or post_air_path
    production_batches = extract_production_rotation_batches(production_air_path)
    observed_production_sha256 = sha256_path(production_air_path)
    if fixture["production_rotation_batches"] and (
        fixture["production_rotation_batches"] != production_batches
    ):
        fail("audited production rotation batches changed")
    bound["production_rotation_batches"] = production_batches
    bound["production_rotation_source"] = {
        "kind": "audited_default_post_ckks_air",
        "post_ckks_air_sha256": observed_production_sha256,
    }
    bound["qualification_bindings"] = {
        "status": "bound",
        "compiler_context_manifest_sha256": sha256_path(context_path),
        "normalized_compiler_command_sha256": invocation_sha256,
        "post_ckks_air_sha256": sha256_path(post_air_path),
    }
    validate_template(bound, require_bound=True)
    return bound


def verify_bindings(
    fixture: dict[str, Any],
    context_path: Path,
    invocation_path: Path,
    post_air_path: Path,
    production_air_path: Path | None = None,
) -> dict[str, Any]:
    validate_template(fixture, require_bound=True)
    context = load_json(context_path)
    resolved = validate_context_manifest(context)
    _, invocation_sha256, compiler_pairs = _load_invocation(invocation_path)
    verify_invocation_context(compiler_pairs, context)
    expected = fixture["qualification_bindings"]
    observed = {
        "compiler_context_manifest_sha256": sha256_path(context_path),
        "normalized_compiler_command_sha256": invocation_sha256,
        "post_ckks_air_sha256": sha256_path(post_air_path),
    }
    for key, value in observed.items():
        if expected[key] != value:
            fail(f"qualification binding mismatch for {key}")
    production_air_path = production_air_path or post_air_path
    if (
        fixture["production_rotation_source"]["post_ckks_air_sha256"]
        != sha256_path(production_air_path)
    ):
        fail("production rotation AIR binding mismatch")
    if fixture["production_rotation_batches"] != extract_production_rotation_batches(
        production_air_path
    ):
        fail("production rotation batches disagree with post-CKKS AIR")
    return resolved


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64
        self.draw_count = 0

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        self.draw_count += 1
        return (value ^ (value >> 31)) & MASK64

    def signed_float53(self) -> float:
        return ((self.next_u64() >> 11) / float(1 << 53)) * 2.0 - 1.0


def deterministic_inputs(fixture: dict[str, Any], slots: int) -> tuple[dict[str, list[complex]], int]:
    generator = SplitMix64(fixture["determinism"]["seed"])
    values: dict[str, list[complex]] = {}
    for descriptor in fixture["inputs"]:
        input_id = descriptor["id"]
        recipe = descriptor["recipe"]
        if recipe == "zero":
            output = [0j] * slots
        elif recipe == "seeded_impulse":
            output = [0j] * slots
            output[0] = complex(
                0.75 * generator.signed_float53(),
                0.75 * generator.signed_float53(),
            )
        elif recipe == "centered_complex_ramp":
            denominator = max(1, slots - 1)
            output = [
                complex((2.0 * index - denominator) / denominator * 0.5,
                        ((index * 3) % max(2, slots) - slots / 2) / max(1, slots))
                for index in range(slots)
            ]
        elif recipe == "alternating_sign":
            output = [
                complex((0.375 + (index % 7) / 32.0) * (-1 if index & 1 else 1),
                        (0.25 + (index % 5) / 40.0) * (-1 if index & 2 else 1))
                for index in range(slots)
            ]
        elif recipe == "seeded_bounded_nonperiodic":
            output = [
                complex(0.875 * generator.signed_float53(),
                        0.875 * generator.signed_float53())
                for _ in range(slots)
            ]
        else:
            fail(f"unknown input recipe: {recipe}")
        if len(output) != slots or any(
            not math.isfinite(item.real) or not math.isfinite(item.imag)
            for item in output
        ):
            fail(f"input recipe {input_id} produced invalid values")
        values[input_id] = output
    return values, generator.draw_count


def normalize_power(power: int, degree: int) -> int:
    return power % (2 * degree)


def resolve_power(symbol: str, degree: int) -> int:
    table = {
        "0": 0,
        "N/2": degree // 2,
        "N": degree,
        "3N/2": 3 * degree // 2,
        "2N-1": 2 * degree - 1,
        "2N+1": 2 * degree + 1,
    }
    if symbol not in table:
        fail(f"unknown monomial power symbol: {symbol}")
    return table[symbol]


def rotate_values(values: Sequence[complex], step: int) -> list[complex]:
    count = len(values)
    normalized = step % count
    return [values[(index + normalized) % count] for index in range(count)]


def _pack_complex(values: Sequence[complex]) -> bytes:
    output = bytearray()
    for value in values:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            fail("decoded values contain NaN or infinity")
        output.extend(struct.pack("<dd", value.real, value.imag))
    return bytes(output)


def _append_blob(buffer: bytearray, values: bytes, count: int) -> dict[str, Any]:
    offset = len(buffer)
    buffer.extend(values)
    return {
        "offset_bytes": offset,
        "count": count,
        "byte_length": len(values),
        "sha256": sha256_bytes(values),
    }


def expected_metadata(resolved: dict[str, Any], *, raised: bool = False) -> dict[str, Any]:
    return {
        "ace_level": (
            resolved["full_data_q_count"]
            if raised
            else resolved["input_active_q_count"]
        ),
        "active_q_count": (
            resolved["full_data_q_count"]
            if raised
            else resolved["input_active_q_count"]
        ),
        "scale_degree": 1,
        "logical_slots": resolved["logical_slots"],
        "ciphertext_size": 2,
        "ntt": True,
    }


def analytic_case_values(
    fixture: dict[str, Any], inputs: dict[str, list[complex]]
) -> list[tuple[str, str, list[complex], int | None]]:
    source = inputs["bounded_nonperiodic"]
    records: list[tuple[str, str, list[complex], int | None]] = [
        (
            "conjugate.bounded_nonperiodic",
            "conjugate",
            [value.conjugate() for value in source],
            None,
        ),
        (
            "conjugate_twice.bounded_nonperiodic",
            "conjugate_twice",
            list(source),
            None,
        ),
    ]
    for position, step in enumerate(fixture["rotate_batch_steps"]):
        records.append(
            (
                f"rotate_batch.bounded_nonperiodic.output_{position}.step_{step}",
                "rotate_batch",
                rotate_values(source, step),
                step,
            )
        )
    for batch_index, batch in enumerate(fixture["production_rotation_batches"]):
        for position, step in enumerate(batch):
            records.append(
                (
                    "rotate_batch.production_"
                    f"{batch_index}.output_{position}.step_{step}",
                    "rotate_batch",
                    rotate_values(source, step),
                    step,
                )
            )
    # Centering each RNS ciphertext component is not additive across a change
    # of modulus.  Consequently it cannot supply an isolated clear decoded
    # identity oracle for raise_mod.  Exact centered RNS evidence and the
    # independently encoded/encrypted ANT result are the two authorities.
    return records


def generate_analytic(
    fixture_path: Path,
    context_path: Path,
    invocation_path: Path,
    post_air_path: Path,
    output_json: Path,
    output_bin: Path,
    production_air_path: Path | None = None,
) -> dict[str, Any]:
    fixture = load_json(fixture_path)
    resolved = verify_bindings(
        fixture,
        context_path,
        invocation_path,
        post_air_path,
        production_air_path,
    )
    inputs, draw_count = deterministic_inputs(fixture, resolved["logical_slots"])
    binary = bytearray(DECODED_MAGIC)
    input_index: list[dict[str, Any]] = []
    for input_id, values in inputs.items():
        descriptor = _append_blob(binary, _pack_complex(values), len(values))
        input_index.append({"input_id": input_id, "values": descriptor})
    records: list[dict[str, Any]] = []
    for case_id, operation, values, step in analytic_case_values(fixture, inputs):
        descriptor = _append_blob(binary, _pack_complex(values), len(values))
        record = {
            "case_id": case_id,
            "operation": operation,
            "metadata": expected_metadata(
                resolved, raised=operation == "raise_mod"
            ),
            "source_input_id": "bounded_nonperiodic",
            "values": descriptor,
        }
        if step is not None:
            record["batch_step"] = step
        records.append(record)
    _atomic_write(output_bin, bytes(binary))
    result = {
        "schema_version": ANALYTIC_SCHEMA,
        "fixture_sha256": sha256_path(fixture_path),
        "qualification_bindings": fixture["qualification_bindings"],
        "deterministic_generator": fixture["determinism"]["generator"],
        "deterministic_seed": fixture["determinism"]["seed"],
        "generator_draw_count": draw_count,
        "binary": {
            "format": fixture["decoded_binary_format"]["id"],
            "size_bytes": len(binary),
            "sha256": sha256_bytes(bytes(binary)),
        },
        "inputs": input_index,
        "records": records,
    }
    write_json(output_json, result)
    return result


def centered_lift(source_residues: Sequence[int], moduli: Sequence[int]) -> list[list[int]]:
    if not moduli:
        fail("centered lift requires at least one modulus")
    q0 = moduli[0]
    result: list[list[int]] = []
    for modulus in moduli:
        lifted: list[int] = []
        for residue in source_residues:
            if residue < 0 or residue >= q0:
                fail("bottom-Q source residue is outside [0, q0)")
            centered = residue if residue <= q0 // 2 else residue - q0
            lifted.append(centered % modulus)
        result.append(lifted)
    return result


def negacyclic_monomial(
    coefficients: Sequence[int], power: int, modulus: int
) -> list[int]:
    degree = len(coefficients)
    normalized = normalize_power(power, degree)
    result = [0] * degree
    for index, coefficient in enumerate(coefficients):
        destination = index + normalized
        wraps, destination = divmod(destination, degree)
        value = coefficient % modulus
        result[destination] = value if wraps % 2 == 0 else (-value) % modulus
    return result


def _pack_u64(values: Iterable[int]) -> bytes:
    output = bytearray()
    for value in values:
        if value < 0 or value > MASK64:
            fail("RNS residue does not fit uint64")
        output.extend(struct.pack("<Q", value))
    return bytes(output)


def _load_moduli(path: Path, resolved: dict[str, Any]) -> list[int]:
    value = load_json(path)
    _expect_keys(value, {"ordered_data_q_moduli"}, "modulus attestation")
    moduli = value["ordered_data_q_moduli"]
    if not isinstance(moduli, list) or len(moduli) != resolved["full_data_q_count"]:
        fail("observed data-Q modulus count disagrees with the context manifest")
    result: list[int] = []
    for index, modulus in enumerate(moduli):
        modulus = _integer(modulus, f"ordered_data_q_moduli[{index}]", 3)
        if modulus.bit_length() != resolved["data_q_bit_sizes"][index]:
            fail(f"observed data-Q modulus {index} has the wrong bit length")
        result.append(modulus)
    return result


def _seeded_coefficients(seed: int, count: int, bound: int) -> list[int]:
    generator = SplitMix64(seed)
    width = 2 * bound + 1
    return [int(generator.next_u64() % width) - bound for _ in range(count)]


def generate_exact(
    fixture_path: Path,
    context_path: Path,
    invocation_path: Path,
    post_air_path: Path,
    moduli_path: Path,
    output_json: Path,
    output_bin: Path,
    production_air_path: Path | None = None,
) -> dict[str, Any]:
    fixture = load_json(fixture_path)
    resolved = verify_bindings(
        fixture,
        context_path,
        invocation_path,
        post_air_path,
        production_air_path,
    )
    moduli = _load_moduli(moduli_path, resolved)
    degree = resolved["polynomial_degree"]
    seed = fixture["determinism"]["seed"]
    coefficient_components = [
        _seeded_coefficients(seed ^ 0x435430, degree, 1 << 20),
        _seeded_coefficients(seed ^ 0x435431, degree, 1 << 20),
    ]
    binary = bytearray(EXACT_MAGIC)
    records: list[dict[str, Any]] = []

    q0 = moduli[0]
    bottom = [[coefficient % q0 for coefficient in component]
              for component in coefficient_components]
    lifted = [centered_lift(component, moduli) for component in bottom]
    source_blob = _pack_u64(value for component in bottom for value in component)
    expected_blob = _pack_u64(
        value
        for component in lifted
        for modulus_values in component
        for value in modulus_values
    )
    records.append(
        {
            "case_id": "exact_algebraic.raise_mod",
            "operation": "raise_mod",
            "source_kind": "deterministic_harness_import",
            "normalized_power": None,
            "source_metadata": {
                "active_q_count": 1,
                "ciphertext_size": 2,
                "ntt": False,
                "chain_position": "bottom_data_q",
            },
            "result_metadata": {
                "active_q_count": len(moduli),
                "ciphertext_size": 2,
                "ntt": False,
                "chain_position": "full_data_q",
            },
            "source": _append_blob(binary, source_blob, len(bottom) * degree),
            "expected": _append_blob(
                binary, expected_blob, len(bottom) * len(moduli) * degree
            ),
            "layout": {
                "component_count": len(bottom),
                "source_modulus_count": 1,
                "result_modulus_count": len(moduli),
                "coefficient_count": degree,
                "ordering": "component,modulus,coefficient",
            },
        }
    )

    source_by_modulus = [
        [[coefficient % modulus for coefficient in component] for modulus in moduli]
        for component in coefficient_components
    ]
    source_blob = _pack_u64(
        value
        for component in source_by_modulus
        for modulus_values in component
        for value in modulus_values
    )
    symbols = list(fixture["monomial_powers"])
    for symbol in symbols:
        power = resolve_power(symbol, degree)
        normalized = normalize_power(power, degree)
        expected = [
            [
                negacyclic_monomial(component[index], normalized, modulus)
                for index, modulus in enumerate(moduli)
            ]
            for component in source_by_modulus
        ]
        expected_blob = _pack_u64(
            value
            for component in expected
            for modulus_values in component
            for value in modulus_values
        )
        label = {
            "0": "0",
            "N/2": "N_over_2",
            "N": "N",
            "3N/2": "3N_over_2",
            "2N-1": "2N_minus_1",
            "2N+1": "2N_plus_1",
        }[symbol]
        records.append(
            {
                "case_id": f"exact_algebraic.mul_mono.{label}",
                "operation": "mul_mono",
                "source_kind": "deterministic_harness_import",
                "normalized_power": normalized,
                "source_metadata": {
                    "active_q_count": len(moduli),
                    "ciphertext_size": 2,
                    "ntt": False,
                    "chain_position": "full_data_q",
                },
                "result_metadata": {
                    "active_q_count": len(moduli),
                    "ciphertext_size": 2,
                    "ntt": False,
                    "chain_position": "full_data_q",
                },
                "source": _append_blob(
                    binary,
                    source_blob,
                    len(coefficient_components) * len(moduli) * degree,
                ),
                "expected": _append_blob(
                    binary,
                    expected_blob,
                    len(coefficient_components) * len(moduli) * degree,
                ),
                "layout": {
                    "component_count": len(coefficient_components),
                    "source_modulus_count": len(moduli),
                    "result_modulus_count": len(moduli),
                    "coefficient_count": degree,
                    "ordering": "component,modulus,coefficient",
                },
            }
        )

    inverse_expected_blob = source_blob
    records.append(
        {
            "case_id": "exact_algebraic.mul_mono.inverse_composition",
            "operation": "mul_mono_inverse_composition",
            "source_kind": "deterministic_harness_import",
            "normalized_power": 0,
            "source_metadata": {
                "active_q_count": len(moduli),
                "ciphertext_size": 2,
                "ntt": False,
                "chain_position": "full_data_q",
            },
            "result_metadata": {
                "active_q_count": len(moduli),
                "ciphertext_size": 2,
                "ntt": False,
                "chain_position": "full_data_q",
            },
            "source": _append_blob(
                binary,
                source_blob,
                len(coefficient_components) * len(moduli) * degree,
            ),
            "expected": _append_blob(
                binary,
                inverse_expected_blob,
                len(coefficient_components) * len(moduli) * degree,
            ),
            "layout": {
                "component_count": len(coefficient_components),
                "source_modulus_count": len(moduli),
                "result_modulus_count": len(moduli),
                "coefficient_count": degree,
                "ordering": "component,modulus,coefficient",
                "composition_powers": [2 * degree - 1, 1],
            },
        }
    )
    _atomic_write(output_bin, bytes(binary))
    result = {
        "schema_version": EXACT_SCHEMA,
        "fixture_sha256": sha256_path(fixture_path),
        "qualification_bindings": fixture["qualification_bindings"],
        "context_manifest_sha256": sha256_path(context_path),
        "modulus_attestation_sha256": sha256_path(moduli_path),
        "ordered_data_q_moduli": moduli,
        "conversion_convention": (
            "coefficient residues; q0 residues centered at q0/2; "
            "X^N=-1 signed negacyclic permutation"
        ),
        "source_contract": (
            "The harness must import these deterministic coefficient residues "
            "as size-2 test ciphertexts without encoding or encryption."
        ),
        "binary": {
            "format": fixture["exact_binary_format"]["id"],
            "size_bytes": len(binary),
            "sha256": sha256_bytes(bytes(binary)),
        },
        "records": records,
    }
    write_json(output_json, result)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind-fixture")
    validate = subparsers.add_parser("validate-fixture")
    analytic = subparsers.add_parser("generate-analytic")
    exact = subparsers.add_parser("generate-exact")
    for command in (bind, validate, analytic, exact):
        command.add_argument("--fixture", required=True, type=Path)
        command.add_argument("--context-manifest", required=True, type=Path)
        command.add_argument("--compiler-invocation", required=True, type=Path)
        command.add_argument("--post-ckks-air", required=True, type=Path)
        command.add_argument("--production-post-ckks-air", type=Path)
    bind.add_argument("--output-json", required=True, type=Path)
    analytic.add_argument("--output-json", required=True, type=Path)
    analytic.add_argument("--output-bin", required=True, type=Path)
    exact.add_argument("--moduli-json", required=True, type=Path)
    exact.add_argument("--output-json", required=True, type=Path)
    exact.add_argument("--output-bin", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "bind-fixture":
            result = bind_fixture(
                arguments.fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
                arguments.production_post_ckks_air,
            )
            write_json(arguments.output_json, result)
        elif arguments.command == "validate-fixture":
            fixture = load_json(arguments.fixture)
            resolved = verify_bindings(
                fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
                arguments.production_post_ckks_air,
            )
            result = {
                "status": "pass",
                "fixture_sha256": sha256_path(arguments.fixture),
                "logical_slots": resolved["logical_slots"],
            }
        elif arguments.command == "generate-analytic":
            result = generate_analytic(
                arguments.fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
                arguments.output_json,
                arguments.output_bin,
                arguments.production_post_ckks_air,
            )
        else:
            result = generate_exact(
                arguments.fixture,
                arguments.context_manifest,
                arguments.compiler_invocation,
                arguments.post_ckks_air,
                arguments.moduli_json,
                arguments.output_json,
                arguments.output_bin,
                arguments.production_post_ckks_air,
            )
        print(json.dumps({"status": "pass", "schema_version": result.get("schema_version")}, sort_keys=True))
        return 0
    except RetainedFixtureError as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
