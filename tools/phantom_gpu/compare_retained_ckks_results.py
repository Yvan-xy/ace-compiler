#!/usr/bin/env python3
"""Strictly compare retained CKKS GPU results with independent references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
from typing import Any, Sequence


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from generate_retained_ckks_fixtures import (  # noqa: E402
    ANALYTIC_SCHEMA,
    DECODED_MAGIC,
    EXACT_SCHEMA,
    EXACT_MAGIC,
    EXACT_SOURCE_MAGIC,
    RetainedFixtureError,
    SplitMix64,
    centered_lift,
    expected_metadata,
    load_json,
    negacyclic_monomial,
    sha256_bytes,
    sha256_path,
    _load_invocation,
    verify_bindings,
    verify_invocation_context,
    write_json,
)


PROVIDER_SCHEMA = "ace.phantom.retained_ckks.provider-result/3.0.0"
EXACT_OBSERVED_SCHEMA = "ace.phantom.retained_ckks.exact-observed/2.0.0"
COMPARISON_SCHEMA = "ace.phantom.retained_ckks.comparison/1.0.0"
EXACT_EVIDENCE_SCHEMA = "ace.phantom.retained_ckks.exact-evidence/2.0.0"
PROVIDER_ATTESTATION_SCHEMA = (
    "ace.phantom.retained_ckks.provider-attestation/1.0.0"
)
BUILD_ATTESTATION_SCHEMA = "ace.phantom.retained_ckks.build-attestation/1.0.0"
RUN_ATTESTATION_SCHEMA = "ace.phantom.retained_ckks.run-attestation/1.0.0"


class ComparisonError(ValueError):
    pass


def fail(message: str) -> None:
    raise ComparisonError(message)


def expect_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    observed = set(value)
    if observed != expected:
        fail(
            f"{context} keys differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{context} must be finite")
    return result


def integer(value: Any, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{context} must be an integer at least {minimum}")
    return value


def expected_provider_operation(case_id: str) -> str:
    if case_id == "conjugate.bounded_nonperiodic":
        return "conjugate"
    if case_id == "conjugate_twice.bounded_nonperiodic":
        return "conjugate_twice"
    if case_id.startswith("rotate_batch."):
        return "rotate_batch"
    if case_id == "raise_mod.bounded_nonperiodic":
        return "raise_mod"
    if case_id == "mul_mono.inverse_composition.bounded_nonperiodic":
        return "mul_mono_inverse_composition"
    if case_id.startswith("mul_mono."):
        return "mul_mono"
    if case_id == "composite.bounded_nonperiodic":
        return "composite"
    fail(f"provider case {case_id!r} has no contracted operation")


def provider_independent_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        field: metadata[field]
        for field in (
            "ace_level",
            "active_q_count",
            "scale_degree",
            "logical_slots",
            "ciphertext_size",
            "ntt",
        )
    }


def validate_decoded_projection(
    value: Any, *, required: bool, context: str
) -> dict[str, Any] | None:
    if not required:
        if value is not None:
            fail(f"{context} must be null for a bottom-Q result")
        return None
    marker = expect_keys(value, {"kind", "active_q_count"}, context)
    if marker["kind"] != "strict_q0_prefix_drop":
        fail(f"{context}.kind is unsupported")
    if marker["active_q_count"] != 1:
        fail(f"{context}.active_q_count must equal 1")
    return marker


def hex_digest(value: Any, length: int, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{context} must be a lowercase hexadecimal digest")
    return value


def load_source_manifest(path: Path, kind: str) -> dict[str, Any]:
    manifest = expect_keys(
        load_json(path),
        {
            "schema_version",
            "kind",
            "source_method",
            "commit",
            "commit_timestamp",
            "tree",
            "archive",
            "archive_size",
            "archive_sha256",
            "allowed_paths",
            "excluded_paths",
            "members",
            "member_count",
            "regular_bytes",
        },
        f"{kind} source manifest",
    )
    if (
        manifest["schema_version"] != "1.0.0"
        or manifest["kind"] != kind
        or manifest["source_method"] != "git-commit-object-archive"
    ):
        fail(f"{kind} source manifest identity is unsupported")
    hex_digest(manifest["commit"], 40, f"{kind} source commit")
    hex_digest(manifest["tree"], 40, f"{kind} source tree")
    hex_digest(manifest["archive_sha256"], 64, f"{kind} source archive")
    integer(manifest["commit_timestamp"], f"{kind} commit timestamp")
    integer(manifest["archive_size"], f"{kind} archive size", 1)
    integer(manifest["member_count"], f"{kind} member count", 1)
    integer(manifest["regular_bytes"], f"{kind} regular bytes", 1)
    if not isinstance(manifest["archive"], str) or not manifest["archive"]:
        fail(f"{kind} source archive name must be nonempty")
    for field in ("allowed_paths", "excluded_paths"):
        values = manifest[field]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            fail(f"{kind} source manifest {field} must be a string array")
        if values != sorted(set(values)):
            fail(f"{kind} source manifest {field} is not unique canonical order")
        for value in values:
            manifest_path = PurePosixPath(value)
            if (
                manifest_path.is_absolute()
                or not manifest_path.parts
                or any(part in {"", ".", ".."} for part in manifest_path.parts)
            ):
                fail(f"{kind} source manifest {field} contains an unsafe path")
    if not manifest["allowed_paths"]:
        fail(f"{kind} source manifest allowed_paths must be nonempty")
    members = manifest["members"]
    if not isinstance(members, list) or len(members) != manifest["member_count"]:
        fail(f"{kind} source manifest member count differs")
    observed_paths: list[str] = []
    observed_regular_bytes = 0
    for index, raw_member in enumerate(members):
        if not isinstance(raw_member, dict):
            fail(f"{kind} source member {index} must be an object")
        member_type = raw_member.get("type")
        required = {"path", "type", "mode", "size"}
        if member_type == "file":
            required.add("sha256")
        elif member_type == "symlink":
            required.add("link_target")
        elif member_type != "directory":
            fail(f"{kind} source member {index} has an invalid type")
        member = expect_keys(raw_member, required, f"{kind} source member {index}")
        if not all(
            isinstance(member[field], str) and member[field]
            for field in ("path", "mode")
        ):
            fail(f"{kind} source member {index} has invalid string fields")
        member_path = PurePosixPath(member["path"])
        if (
            member_path.is_absolute()
            or not member_path.parts
            or member_path.parts[0] != f"{kind}-source"
            or (len(member_path.parts) == 1 and member_type != "directory")
            or any(part in {"", ".", ".."} for part in member_path.parts)
        ):
            fail(f"{kind} source member {index} has an unsafe path")
        if len(member["mode"]) != 4 or any(
            character not in "01234567" for character in member["mode"]
        ):
            fail(f"{kind} source member {index} has an invalid mode")
        integer(member["size"], f"{kind} source member {index} size")
        observed_paths.append(member["path"])
        if member_type == "file":
            hex_digest(member["sha256"], 64, f"{kind} source member {index}")
            observed_regular_bytes += member["size"]
        elif member_type == "symlink":
            if not isinstance(member["link_target"], str) or not member["link_target"]:
                fail(f"{kind} source member {index} link target is invalid")
            target = PurePosixPath(member["link_target"])
            if target.is_absolute():
                fail(f"{kind} source member {index} link target is absolute")
            resolved_parts = list(member_path.parent.parts[1:])
            for part in target.parts:
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not resolved_parts:
                        fail(f"{kind} source member {index} link target escapes")
                    resolved_parts.pop()
                else:
                    resolved_parts.append(part)
        if member_type != "file" and member["size"] != 0:
            fail(f"{kind} source member {index} must have zero size")
    if observed_paths != sorted(observed_paths) or len(set(observed_paths)) != len(
        observed_paths
    ):
        fail(f"{kind} source manifest paths are not unique canonical order")
    if observed_regular_bytes != manifest["regular_bytes"]:
        fail(f"{kind} source manifest regular byte count differs")
    return manifest


def load_build_attestation(
    path: Path,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    value = expect_keys(
        load_json(path),
        {
            "schema_version",
            "status",
            "architecture",
            "ace_commit",
            "phantom_commit",
            "source_mode",
            "ace_source_manifest_sha256",
            "phantom_source_manifest_sha256",
            "compiler_context_manifest_sha256",
            "compiler_resource_manifest_sha256",
            "fixture_sha256",
            "compiler_invocation_sha256",
            "generated_ant_source_sha256",
            "generated_phantom_source_sha256",
            "archives",
            "executables",
            "link_commands_sha256",
            "container",
            "link_mode",
            "archive_inspection",
            "undefined_symbol_inspection",
            "cubin_architecture_inspection",
            "host_tests",
            "host_ant_oracle_was_run",
            "gpu_executables_were_run",
        },
        "build attestation",
    )
    if (
        value["schema_version"] != BUILD_ATTESTATION_SCHEMA
        or value["status"] != "pass"
        or value["architecture"] != "sm_80"
        or value["source_mode"] != "snapshot"
        or value["link_mode"] != "explicit_compile-device-link-host-link"
        or any(
            value[field] != "pass"
            for field in (
                "archive_inspection",
                "undefined_symbol_inspection",
                "cubin_architecture_inspection",
                "host_tests",
            )
        )
        or value["host_ant_oracle_was_run"] is not True
        or value["gpu_executables_were_run"] is not False
    ):
        fail("build attestation did not record the reviewed host build contract")
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            fail(f"build attestation {field} mismatch")
    archives = expect_keys(
        value["archives"],
        {"adapter", "provider", "common", "ant", "ant_encode"},
        "build attestation archives",
    )
    for name, digest in archives.items():
        hex_digest(digest, 64, f"build archive {name}")
    executables = expect_keys(
        value["executables"],
        {"ant_oracle", "phantom_sm80"},
        "build attestation executables",
    )
    for name, digest in executables.items():
        hex_digest(digest, 64, f"build executable {name}")
    hex_digest(value["link_commands_sha256"], 64, "build link commands")
    container = expect_keys(
        value["container"],
        {"image", "config_digest", "bootstrap_sha256"},
        "build attestation container",
    )
    if not isinstance(container["image"], str) or not container["image"]:
        fail("build container image must be nonempty")
    config_digest = container["config_digest"]
    if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
        fail("build container config digest must use the sha256 prefix")
    hex_digest(config_digest.removeprefix("sha256:"), 64, "build container config")
    hex_digest(container["bootstrap_sha256"], 64, "build container bootstrap")
    return value


def load_run_attestation(path: Path, *, expected: dict[str, Any]) -> dict[str, Any]:
    value = expect_keys(
        load_json(path),
        {
            "schema_version",
            "status",
            "ace_commit",
            "phantom_commit",
            "ace_source_manifest_sha256",
            "phantom_source_manifest_sha256",
            "generation_attestation_sha256",
            "build_attestation_sha256",
            "fixture_sha256",
            "compiler_context_manifest_sha256",
            "compiler_resource_manifest_sha256",
            "executables",
            "provider_results",
            "exact_observed",
            "gpu",
        },
        "run attestation",
    )
    if value["schema_version"] != RUN_ATTESTATION_SCHEMA or value["status"] != "pass":
        fail("run attestation schema or status mismatch")
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            fail(f"run attestation {field} mismatch")
    executables = expect_keys(
        value["executables"],
        {"ant_oracle", "phantom_sm80"},
        "run attestation executables",
    )
    for name, digest in executables.items():
        hex_digest(digest, 64, f"run executable {name}")
    providers = expect_keys(
        value["provider_results"], {"ant", "phantom"}, "run provider results"
    )
    for provider_name, raw_result in providers.items():
        result = expect_keys(
            raw_result,
            {"json_sha256", "binary_sha256"},
            f"run {provider_name} result",
        )
        for name, digest in result.items():
            hex_digest(digest, 64, f"run {provider_name} {name}")
    exact = expect_keys(
        value["exact_observed"],
        {"json_sha256", "binary_sha256"},
        "run exact observed",
    )
    for name, digest in exact.items():
        hex_digest(digest, 64, f"run exact observed {name}")
    gpu = expect_keys(value["gpu"], {"device_count", "device_name"}, "run GPU")
    if gpu["device_count"] != 1 or not isinstance(gpu["device_name"], str) or not gpu[
        "device_name"
    ]:
        fail("run GPU identity is invalid")
    return value


def load_binary(
    path: Path,
    descriptor: dict[str, Any],
    magic: bytes,
    expected_format: str,
) -> bytes:
    expect_keys(descriptor, {"format", "size_bytes", "sha256"}, "binary descriptor")
    if descriptor["format"] != expected_format:
        fail(f"binary {path} format does not match {expected_format}")
    try:
        data = path.read_bytes()
    except OSError as error:
        fail(f"cannot read binary {path}: {error}")
    if not data.startswith(magic):
        fail(f"binary {path} has the wrong magic")
    if descriptor["size_bytes"] != len(data):
        fail(f"binary {path} size does not match its descriptor")
    if descriptor["sha256"] != sha256_bytes(data):
        fail(f"binary {path} SHA-256 does not match its descriptor")
    return data


def read_blob(data: bytes, descriptor: Any, *, unit_size: int, context: str) -> bytes:
    descriptor = expect_keys(
        descriptor,
        {"offset_bytes", "count", "byte_length", "sha256"},
        f"{context} descriptor",
    )
    offset = descriptor["offset_bytes"]
    count = descriptor["count"]
    length = descriptor["byte_length"]
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or isinstance(length, bool)
        or not isinstance(length, int)
        or offset < len(DECODED_MAGIC)
        or count < 0
        or length != count * unit_size
        or offset % 8
        or offset + length > len(data)
    ):
        fail(f"{context} has an invalid binary range")
    blob = data[offset : offset + length]
    if descriptor["sha256"] != sha256_bytes(blob):
        fail(f"{context} content hash mismatch")
    return blob


def read_complex(data: bytes, descriptor: Any, context: str) -> list[complex]:
    blob = read_blob(data, descriptor, unit_size=16, context=context)
    values: list[complex] = []
    for index, (real, imaginary) in enumerate(struct.iter_unpack("<dd", blob)):
        if not math.isfinite(real) or not math.isfinite(imaginary):
            fail(f"{context}[{index}] is non-finite")
        values.append(complex(real, imaginary))
    return values


def read_u64(data: bytes, descriptor: Any, context: str) -> list[int]:
    blob = read_blob(data, descriptor, unit_size=8, context=context)
    return [item[0] for item in struct.iter_unpack("<Q", blob)]


def read_i64(data: bytes, descriptor: Any, context: str) -> list[int]:
    blob = read_blob(data, descriptor, unit_size=8, context=context)
    return [item[0] for item in struct.iter_unpack("<q", blob)]


def expected_analytic_contracts(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = [
        {
            "case_id": "conjugate.bounded_nonperiodic",
            "operation": "conjugate",
            "batch_step": None,
        },
        {
            "case_id": "conjugate_twice.bounded_nonperiodic",
            "operation": "conjugate_twice",
            "batch_step": None,
        },
    ]
    batches = [("bounded_nonperiodic", fixture["rotate_batch_steps"])] + [
        (f"production_{index}", batch)
        for index, batch in enumerate(fixture["production_rotation_batches"])
    ]
    for prefix, batch in batches:
        for position, step in enumerate(batch):
            contracts.append(
                {
                    "case_id": (
                        f"rotate_batch.{prefix}.output_{position}.step_{step}"
                    ),
                    "operation": "rotate_batch",
                    "batch_step": step,
                }
            )
    contracts.append(
        {
            "case_id": "raise_mod.bounded_nonperiodic",
            "operation": "raise_mod",
            "batch_step": None,
        }
    )
    case_ids = [contract["case_id"] for contract in contracts]
    if len(case_ids) != len(set(case_ids)):
        fail("fixture expands to duplicate analytic case identifiers")
    return contracts


def expected_deterministic_inputs(
    fixture: dict[str, Any], slots: int
) -> tuple[dict[str, list[complex]], int]:
    generator = SplitMix64(fixture["determinism"]["seed"])
    result: dict[str, list[complex]] = {}
    for descriptor in fixture["inputs"]:
        input_id = descriptor["id"]
        recipe = descriptor["recipe"]
        if recipe == "zero":
            values = [0j] * slots
        elif recipe == "seeded_impulse":
            values = [0j] * slots
            values[0] = complex(
                0.75 * generator.signed_float53(),
                0.75 * generator.signed_float53(),
            )
        elif recipe == "centered_complex_ramp":
            denominator = max(1, slots - 1)
            values = [
                complex(
                    (2.0 * index - denominator) / denominator * 0.5,
                    ((index * 3) % max(2, slots) - slots / 2) / max(1, slots),
                )
                for index in range(slots)
            ]
        elif recipe == "alternating_sign":
            values = [
                complex(
                    (0.375 + (index % 7) / 32.0)
                    * (-1 if index & 1 else 1),
                    (0.25 + (index % 5) / 40.0)
                    * (-1 if index & 2 else 1),
                )
                for index in range(slots)
            ]
        elif recipe == "seeded_bounded_nonperiodic":
            values = [
                complex(
                    0.875 * generator.signed_float53(),
                    0.875 * generator.signed_float53(),
                )
                for _ in range(slots)
            ]
        else:
            fail(f"unsupported deterministic input recipe {recipe!r}")
        result[input_id] = values
    return result, generator.draw_count


def expected_provider_order(fixture: dict[str, Any]) -> list[str]:
    order = [contract["case_id"] for contract in expected_analytic_contracts(fixture)]
    present = set(order)
    decoded_seen: set[str] = set()
    for index, descriptor in enumerate(fixture["decoded_cases"]):
        descriptor = expect_keys(
            descriptor, {"id", "oracle"}, f"fixture decoded case {index}"
        )
        case_id = descriptor["id"]
        if case_id in decoded_seen:
            fail(f"fixture contains duplicate decoded case {case_id}")
        decoded_seen.add(case_id)
        if case_id == "rotate_batch.bounded_nonperiodic" or case_id in present:
            continue
        present.add(case_id)
        order.append(case_id)
    return order


def validate_exact_case_order(
    observed: Sequence[str], expected: Sequence[str], context: str
) -> None:
    if list(observed) != list(expected):
        fail(
            f"{context} case order differs from the fixture expansion: "
            f"observed_count={len(observed)}, expected_count={len(expected)}"
        )


def validate_canonical_offsets(
    descriptors: Sequence[dict[str, Any]], data_size: int, context: str
) -> None:
    expected_offset = len(DECODED_MAGIC)
    for index, descriptor in enumerate(descriptors):
        if descriptor["offset_bytes"] != expected_offset:
            fail(
                f"{context} descriptor {index} starts at "
                f"{descriptor['offset_bytes']}, expected {expected_offset}"
            )
        expected_offset += descriptor["byte_length"]
    if expected_offset != data_size:
        fail(f"{context} binary has unindexed trailing bytes")


def validate_metadata(
    metadata: Any,
    resolved: dict[str, Any],
    *,
    raised: bool,
    first_data_chain_index: int,
    context: str,
) -> dict[str, Any]:
    metadata = expect_keys(
        metadata,
        {
            "ace_level",
            "active_q_count",
            "scale_degree",
            "logical_slots",
            "ciphertext_size",
            "ntt",
            "chain_index",
            "raw_scale",
        },
        context,
    )
    expected = expected_metadata(resolved, raised=raised)
    for field in (
        "ace_level",
        "active_q_count",
        "scale_degree",
        "logical_slots",
        "ciphertext_size",
    ):
        integer(metadata[field], f"{context}.{field}")
    if not isinstance(metadata["ntt"], bool):
        fail(f"{context}.ntt must be a boolean")
    for field, expected_value in expected.items():
        if metadata[field] != expected_value:
            fail(f"{context}.{field} is {metadata[field]!r}, expected {expected_value!r}")
    integer(metadata["chain_index"], f"{context}.chain_index")
    expected_chain_index = (
        first_data_chain_index
        + resolved["full_data_q_count"]
        - expected["active_q_count"]
    )
    if metadata["chain_index"] != expected_chain_index:
        fail(
            f"{context}.chain_index is {metadata['chain_index']}, "
            f"expected {expected_chain_index}"
        )
    if finite_number(metadata["raw_scale"], f"{context}.raw_scale") <= 0:
        fail(f"{context}.raw_scale must be positive")
    return metadata


def load_analytic(
    path: Path,
    binary_path: Path,
    fixture_sha256: str,
    fixture: dict[str, Any],
    resolved: dict[str, Any],
) -> tuple[dict[str, list[complex]], list[dict[str, Any]]]:
    slots = resolved["logical_slots"]
    value = expect_keys(
        load_json(path),
        {
            "schema_version",
            "fixture_sha256",
            "qualification_bindings",
            "deterministic_generator",
            "deterministic_seed",
            "generator_draw_count",
            "binary",
            "inputs",
            "records",
        },
        "analytic reference",
    )
    if value["schema_version"] != ANALYTIC_SCHEMA or value["fixture_sha256"] != fixture_sha256:
        fail("analytic reference schema or fixture binding mismatch")
    if value["qualification_bindings"] != fixture["qualification_bindings"]:
        fail("analytic reference qualification bindings mismatch")
    if value["deterministic_generator"] != fixture["determinism"]["generator"]:
        fail("analytic deterministic generator differs from the fixture")
    if value["deterministic_seed"] != fixture["determinism"]["seed"]:
        fail("analytic deterministic seed differs from the fixture")
    expected_inputs, expected_draw_count = expected_deterministic_inputs(
        fixture, slots
    )
    if value["generator_draw_count"] != expected_draw_count:
        fail(
            "analytic generator draw count is "
            f"{value['generator_draw_count']!r}, expected {expected_draw_count}"
        )
    data = load_binary(
        binary_path,
        value["binary"],
        DECODED_MAGIC,
        "ace.retained_ckks.complex_float64le/1.0.0",
    )
    inputs: dict[str, list[complex]] = {}
    descriptors: list[dict[str, Any]] = []
    expected_input_ids = [descriptor["id"] for descriptor in fixture["inputs"]]
    if not isinstance(value["inputs"], list) or any(
        not isinstance(item, dict) for item in value["inputs"]
    ):
        fail("analytic inputs must be an array of objects")
    if [item.get("input_id") for item in value["inputs"]] != expected_input_ids:
        fail("analytic input order differs from the fixture recipe order")
    for index, item in enumerate(value["inputs"]):
        item = expect_keys(item, {"input_id", "values"}, f"analytic input {index}")
        if item["input_id"] in inputs:
            fail(f"duplicate analytic input {item['input_id']}")
        inputs[item["input_id"]] = read_complex(
            data, item["values"], f"analytic input {item['input_id']}"
        )
        descriptors.append(item["values"])
        if len(inputs[item["input_id"]]) != slots:
            fail(f"analytic input {item['input_id']} has the wrong slot count")
        if inputs[item["input_id"]] != expected_inputs[item["input_id"]]:
            fail(
                f"analytic input {item['input_id']} does not reproduce the "
                "fixture seed and generator"
            )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_contracts = expected_analytic_contracts(fixture)
    if not isinstance(value["records"], list) or any(
        not isinstance(item, dict) for item in value["records"]
    ):
        fail("analytic records must be an array of objects")
    validate_exact_case_order(
        [item.get("case_id") for item in value["records"]],
        [contract["case_id"] for contract in expected_contracts],
        "analytic",
    )
    for index, item in enumerate(value["records"]):
        expected_keys = {"case_id", "operation", "metadata", "source_input_id", "values"}
        if item.get("operation") == "rotate_batch":
            expected_keys.add("batch_step")
        item = expect_keys(item, expected_keys, f"analytic record {index}")
        if item["case_id"] in seen:
            fail(f"duplicate analytic case {item['case_id']}")
        seen.add(item["case_id"])
        contract = expected_contracts[index]
        if item["operation"] != contract["operation"]:
            fail(f"analytic operation differs for {item['case_id']}")
        if item.get("batch_step") != contract["batch_step"]:
            fail(f"analytic batch step differs for {item['case_id']}")
        if item["source_input_id"] != "bounded_nonperiodic":
            fail(f"analytic source input differs for {item['case_id']}")
        metadata = expect_keys(
            item["metadata"],
            {
                "ace_level",
                "active_q_count",
                "scale_degree",
                "logical_slots",
                "ciphertext_size",
                "ntt",
            },
            f"analytic metadata {item['case_id']}",
        )
        for field in (
            "ace_level",
            "active_q_count",
            "scale_degree",
            "logical_slots",
            "ciphertext_size",
        ):
            integer(metadata[field], f"analytic metadata {item['case_id']}.{field}")
        if not isinstance(metadata["ntt"], bool):
            fail(f"analytic metadata {item['case_id']}.ntt must be a boolean")
        if metadata != expected_metadata(
            resolved, raised=item["operation"] == "raise_mod"
        ):
            fail(f"analytic metadata differs for {item['case_id']}")
        item = dict(item)
        item["decoded_values"] = read_complex(data, item["values"], item["case_id"])
        descriptors.append(item["values"])
        if len(item["decoded_values"]) != slots:
            fail(f"analytic case {item['case_id']} has the wrong slot count")
        source_values = expected_inputs["bounded_nonperiodic"]
        if item["operation"] == "conjugate":
            expected_values = [value.conjugate() for value in source_values]
        elif item["operation"] == "conjugate_twice":
            expected_values = list(source_values)
        elif item["operation"] == "rotate_batch":
            step = item["batch_step"]
            expected_values = [
                source_values[(position + step) % slots]
                for position in range(slots)
            ]
        else:
            if item["operation"] != "raise_mod":
                fail(f"unsupported analytic operation {item['operation']}")
            expected_values = list(source_values)
        if item["decoded_values"] != expected_values:
            fail(f"analytic values differ from the fixture oracle for {item['case_id']}")
        records.append(item)
    validate_canonical_offsets(descriptors, len(data), "analytic")
    return inputs, records


def load_provider(
    path: Path,
    binary_path: Path,
    provider_name: str,
    fixture_sha256: str,
    context_sha256: str,
    bindings: dict[str, Any],
    expected_order: Sequence[str],
    resolved: dict[str, Any],
    attested_identifiers: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    value = expect_keys(
        load_json(path),
        {
            "schema_version",
            "provider",
            "fixture_sha256",
            "context_manifest_sha256",
            "qualification_bindings",
            "identifiers",
            "first_data_chain_index",
            "binary",
            "records",
        },
        f"{provider_name} result",
    )
    if value["schema_version"] != PROVIDER_SCHEMA or value["provider"] != provider_name:
        fail(f"{provider_name} result schema or provider mismatch")
    if value["fixture_sha256"] != fixture_sha256 or value["context_manifest_sha256"] != context_sha256:
        fail(f"{provider_name} result artifact binding mismatch")
    if value["qualification_bindings"] != bindings:
        fail(f"{provider_name} qualification bindings mismatch")
    identifiers = expect_keys(value["identifiers"], {"ace_commit", "phantom_commit", "executable_sha256"}, f"{provider_name} identifiers")
    for field, digest in identifiers.items():
        hex_digest(
            digest,
            40 if field != "executable_sha256" else 64,
            f"{provider_name} identifiers.{field}",
        )
    if identifiers != attested_identifiers:
        fail(f"{provider_name} identifiers differ from the provider attestation")
    first_data_chain_index = value["first_data_chain_index"]
    if (
        isinstance(first_data_chain_index, bool)
        or not isinstance(first_data_chain_index, int)
        or first_data_chain_index < 0
    ):
        fail(f"{provider_name} first_data_chain_index must be nonnegative")
    data = load_binary(
        binary_path,
        value["binary"],
        DECODED_MAGIC,
        "ace.retained_ckks.complex_float64le/1.0.0",
    )
    records: list[dict[str, Any]] = []
    observed_order: list[str] = []
    ownership_tokens: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    for index, raw_record in enumerate(value["records"]):
        record = expect_keys(
            raw_record,
            {
                "case_id",
                "operation",
                "metadata",
                "source_metadata_before",
                "source_metadata_after",
                "source_values_sha256_before",
                "source_values_sha256_after",
                "ownership_token",
                "decoded_projection",
                "values",
            },
            f"{provider_name} record {index}",
        )
        case_id = record["case_id"]
        if not isinstance(case_id, str) or not case_id:
            fail(f"{provider_name} record {index}.case_id must be a string")
        observed_order.append(case_id)
        expected_operation = expected_provider_operation(case_id)
        if record["operation"] != expected_operation:
            fail(
                f"{provider_name}.{case_id}.operation is "
                f"{record['operation']!r}, expected {expected_operation!r}"
            )
        raised = expected_operation in ("raise_mod", "composite")
        decoded_projection = validate_decoded_projection(
            record["decoded_projection"],
            required=raised,
            context=f"{provider_name}.{case_id}.decoded_projection",
        )
        metadata = validate_metadata(record["metadata"], resolved, raised=raised, first_data_chain_index=first_data_chain_index, context=f"{provider_name}.{case_id}.metadata")
        source_before = validate_metadata(record["source_metadata_before"], resolved, raised=False, first_data_chain_index=first_data_chain_index, context=f"{provider_name}.{case_id}.source_before")
        source_after = validate_metadata(record["source_metadata_after"], resolved, raised=False, first_data_chain_index=first_data_chain_index, context=f"{provider_name}.{case_id}.source_after")
        if source_before != source_after:
            fail(f"{provider_name}.{case_id} mutated source metadata")
        if metadata["raw_scale"] != source_before["raw_scale"]:
            fail(f"{provider_name}.{case_id} changed the raw scale")
        if record["source_values_sha256_before"] != record["source_values_sha256_after"]:
            fail(f"{provider_name}.{case_id} mutated source values")
        hex_digest(
            record["source_values_sha256_before"],
            64,
            f"{provider_name}.{case_id}.source_values_sha256_before",
        )
        token = record["ownership_token"]
        if expected_operation == "rotate_batch":
            if not isinstance(token, str) or not token or token in ownership_tokens:
                fail(f"{provider_name}.{case_id} lacks independent batch ownership")
            ownership_tokens.add(token)
        elif token is not None:
            fail(f"{provider_name}.{case_id} has an unexpected ownership token")
        materialized = dict(record)
        materialized["metadata"] = metadata
        materialized["decoded_projection"] = decoded_projection
        materialized["decoded_values"] = read_complex(data, record["values"], f"{provider_name}.{case_id}")
        descriptors.append(record["values"])
        if len(materialized["decoded_values"]) != resolved["logical_slots"]:
            fail(f"{provider_name}.{case_id} has the wrong slot count")
        records.append(materialized)
    validate_exact_case_order(observed_order, expected_order, provider_name)
    validate_canonical_offsets(descriptors, len(data), provider_name)
    return records, first_data_chain_index


def load_identity_chain(arguments: argparse.Namespace) -> dict[str, Any]:
    ace_source = load_source_manifest(arguments.ace_source_manifest, "ace")
    phantom_source = load_source_manifest(arguments.phantom_source_manifest, "phantom")
    ace_source_sha256 = sha256_path(arguments.ace_source_manifest)
    phantom_source_sha256 = sha256_path(arguments.phantom_source_manifest)

    generation, _normalized_argv_sha256, _generation_options = _load_invocation(
        arguments.generation_attestation
    )
    verify_invocation_context(
        generation,
        arguments.context_manifest,
        arguments.post_ckks_air,
        arguments.generation_fixture,
    )
    generation_sha256 = sha256_path(arguments.generation_attestation)
    generation_expected = {
        "ace_commit": ace_source["commit"],
        "fixture_sha256": sha256_path(arguments.generation_fixture),
        "input_context_manifest_sha256": sha256_path(arguments.context_manifest),
        "emitted_context_manifest_sha256": sha256_path(
            arguments.emitted_context_manifest
        ),
        "resource_manifest_sha256": sha256_path(arguments.resource_manifest),
        "post_ckks_air_sha256": sha256_path(arguments.post_ckks_air),
        "ant_post_ckks_air_sha256": sha256_path(arguments.ant_post_ckks_air),
        "phantom_post_ckks_air_sha256": sha256_path(arguments.post_ckks_air),
        "ant_source_sha256": sha256_path(arguments.generated_ant_source),
        "phantom_source_sha256": sha256_path(arguments.generated_phantom_source),
    }
    for field, expected in generation_expected.items():
        if generation[field] != expected:
            fail(f"generation attestation {field} mismatch")
    if arguments.ant_post_ckks_air.read_bytes() != arguments.post_ckks_air.read_bytes():
        fail("ANT and Phantom post-CKKS AIR artifacts differ")
    if load_json(arguments.emitted_context_manifest) != load_json(
        arguments.context_manifest
    ):
        fail("emitted and input compiler context manifests differ")

    fixture_sha256 = sha256_path(arguments.fixture)
    context_sha256 = sha256_path(arguments.context_manifest)
    resource_sha256 = sha256_path(arguments.resource_manifest)
    ant_executable_sha256 = sha256_path(arguments.ant_executable)
    phantom_executable_sha256 = sha256_path(arguments.phantom_executable)
    build_expected = {
        "ace_commit": ace_source["commit"],
        "phantom_commit": phantom_source["commit"],
        "ace_source_manifest_sha256": ace_source_sha256,
        "phantom_source_manifest_sha256": phantom_source_sha256,
        "compiler_context_manifest_sha256": context_sha256,
        "compiler_resource_manifest_sha256": resource_sha256,
        "fixture_sha256": fixture_sha256,
        "compiler_invocation_sha256": generation_sha256,
        "generated_ant_source_sha256": sha256_path(arguments.generated_ant_source),
        "generated_phantom_source_sha256": sha256_path(
            arguments.generated_phantom_source
        ),
    }
    build = load_build_attestation(arguments.build_attestation, expected=build_expected)
    if build["executables"] != {
        "ant_oracle": ant_executable_sha256,
        "phantom_sm80": phantom_executable_sha256,
    }:
        fail("build attestation executable hashes mismatch")
    build_sha256 = sha256_path(arguments.build_attestation)

    run_expected = {
        "ace_commit": ace_source["commit"],
        "phantom_commit": phantom_source["commit"],
        "ace_source_manifest_sha256": ace_source_sha256,
        "phantom_source_manifest_sha256": phantom_source_sha256,
        "generation_attestation_sha256": generation_sha256,
        "build_attestation_sha256": build_sha256,
        "fixture_sha256": fixture_sha256,
        "compiler_context_manifest_sha256": context_sha256,
        "compiler_resource_manifest_sha256": resource_sha256,
    }
    run = load_run_attestation(arguments.run_attestation, expected=run_expected)
    if run["executables"] != build["executables"]:
        fail("run and build executable hashes differ")
    expected_provider_results = {
        "ant": {
            "json_sha256": sha256_path(arguments.ant_json),
            "binary_sha256": sha256_path(arguments.ant_bin),
        },
        "phantom": {
            "json_sha256": sha256_path(arguments.gpu_json),
            "binary_sha256": sha256_path(arguments.gpu_bin),
        },
    }
    if run["provider_results"] != expected_provider_results:
        fail("run attestation provider result hashes mismatch")
    if run["exact_observed"] != {
        "json_sha256": sha256_path(arguments.exact_observed_json),
        "binary_sha256": sha256_path(arguments.exact_observed_bin),
    }:
        fail("run attestation exact observed hashes mismatch")
    return {
        "ace_commit": ace_source["commit"],
        "phantom_commit": phantom_source["commit"],
        "ace_source_manifest_sha256": ace_source_sha256,
        "phantom_source_manifest_sha256": phantom_source_sha256,
        "generation_sha256": generation_sha256,
        "build_sha256": build_sha256,
        "run_sha256": sha256_path(arguments.run_attestation),
        "identifiers": {
            "ant": {
                "ace_commit": ace_source["commit"],
                "phantom_commit": phantom_source["commit"],
                "executable_sha256": ant_executable_sha256,
            },
            "phantom": {
                "ace_commit": ace_source["commit"],
                "phantom_commit": phantom_source["commit"],
                "executable_sha256": phantom_executable_sha256,
            },
        },
    }


def load_provider_attestation(
    path: Path,
    *,
    fixture_sha256: str,
    context_sha256: str,
    generation_sha256: str,
    build_sha256: str,
    run_sha256: str,
    expected_identifiers: dict[str, dict[str, Any]],
    ant_json: Path,
    ant_binary: Path,
    gpu_json: Path,
    gpu_binary: Path,
    exact_observed_json: Path,
    exact_observed_binary: Path,
) -> dict[str, dict[str, Any]]:
    value = expect_keys(
        load_json(path),
        {
            "schema_version",
            "fixture_sha256",
            "context_manifest_sha256",
            "compiler_invocation_sha256",
            "build_attestation_sha256",
            "run_attestation_sha256",
            "providers",
            "exact_observed",
        },
        "provider attestation",
    )
    if value["schema_version"] != PROVIDER_ATTESTATION_SCHEMA:
        fail("provider attestation schema mismatch")
    expected_bindings = {
        "fixture_sha256": fixture_sha256,
        "context_manifest_sha256": context_sha256,
        "compiler_invocation_sha256": generation_sha256,
        "build_attestation_sha256": build_sha256,
        "run_attestation_sha256": run_sha256,
    }
    for field, expected in expected_bindings.items():
        if value[field] != expected:
            fail(f"provider attestation {field} mismatch")
    expected_artifacts = {
        "ant": (sha256_path(ant_json), sha256_path(ant_binary)),
        "phantom": (sha256_path(gpu_json), sha256_path(gpu_binary)),
    }
    providers = value["providers"]
    if not isinstance(providers, list) or [
        item.get("provider") if isinstance(item, dict) else None
        for item in providers
    ] != ["ant", "phantom"]:
        fail("provider attestation must contain ordered ANT and Phantom records")
    identifiers_by_provider: dict[str, dict[str, Any]] = {}
    for index, raw_provider in enumerate(providers):
        provider = expect_keys(
            raw_provider,
            {
                "provider",
                "identifiers",
                "result_json_sha256",
                "result_binary_sha256",
            },
            f"provider attestation record {index}",
        )
        provider_name = provider["provider"]
        identifiers = expect_keys(
            provider["identifiers"],
            {"ace_commit", "phantom_commit", "executable_sha256"},
            f"provider attestation {provider_name} identifiers",
        )
        for field, digest in identifiers.items():
            hex_digest(
                digest,
                64 if field == "executable_sha256" else 40,
                f"provider attestation {provider_name}.{field}",
            )
        if identifiers != expected_identifiers[provider_name]:
            fail(
                f"provider attestation {provider_name} identifiers differ "
                "from audited source/build identity"
            )
        expected_json, expected_binary = expected_artifacts[provider_name]
        if (
            provider["result_json_sha256"] != expected_json
            or provider["result_binary_sha256"] != expected_binary
        ):
            fail(f"provider attestation {provider_name} artifact hash mismatch")
        identifiers_by_provider[provider_name] = identifiers
    exact = expect_keys(
        value["exact_observed"],
        {
            "provider",
            "executable_sha256",
            "result_json_sha256",
            "result_binary_sha256",
        },
        "provider attestation exact observed",
    )
    if exact["provider"] != "phantom":
        fail("exact observed attestation must name the Phantom provider")
    if (
        exact["executable_sha256"]
        != identifiers_by_provider["phantom"]["executable_sha256"]
        or exact["result_json_sha256"] != sha256_path(exact_observed_json)
        or exact["result_binary_sha256"] != sha256_path(exact_observed_binary)
    ):
        fail("exact observed artifact differs from the provider attestation")
    return identifiers_by_provider


def metric(
    actual: Sequence[complex],
    reference: Sequence[complex],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    hard_maximum: float,
    relative_floor: float,
) -> dict[str, Any]:
    if len(actual) != len(reference) or not actual:
        fail("decoded vectors have unequal or zero length")
    errors = [abs(a - b) for a, b in zip(actual, reference)]
    maximum = max(errors)
    maximum_index = errors.index(maximum)
    relative = [error / abs(expected) for error, expected in zip(errors, reference) if abs(expected) >= relative_floor]
    failed = [
        index
        for index, (error, expected) in enumerate(zip(errors, reference))
        if error > absolute_tolerance + relative_tolerance * abs(expected)
        or error > hard_maximum
    ]
    return {
        "status": "pass" if not failed else "fail",
        "comparison_count": len(errors),
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_index": maximum_index,
        "maximum_relative_error_above_floor": max(relative, default=0.0),
        "mean_absolute_error": sum(errors) / len(errors),
        "root_mean_square_error": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "estimated_precision_bits": None if maximum == 0.0 else -math.log2(maximum),
        "failed_indices": failed,
    }


def reduce_signed_coefficients(
    components: Sequence[Sequence[int]], moduli: Sequence[int]
) -> list[int]:
    return [
        coefficient % modulus
        for component in components
        for modulus in moduli
        for coefficient in component
    ]


def load_exact_source(
    json_path: Path,
    binary_path: Path,
    fixture: dict[str, Any],
    fixture_sha256: str,
    context_sha256: str,
    resolved: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    source = expect_keys(
        load_json(json_path),
        {
            "schema_version",
            "fixture_sha256",
            "qualification_bindings",
            "context_manifest_sha256",
            "determinism",
            "conversion_convention",
            "layout",
            "source_id",
            "signed_coefficients",
            "binary",
            "records",
        },
        "exact signed source",
    )
    if source["schema_version"] != EXACT_SCHEMA:
        fail("exact signed source schema mismatch")
    if (
        source["fixture_sha256"] != fixture_sha256
        or source["context_manifest_sha256"] != context_sha256
        or source["qualification_bindings"] != fixture["qualification_bindings"]
    ):
        fail("exact signed source artifact binding mismatch")
    recipe = fixture["exact_source_recipe"]
    determinism = expect_keys(
        source["determinism"],
        {
            "generator",
            "seed",
            "component_seed_xors",
            "coefficient_absolute_bound",
            "generator_draw_count",
        },
        "exact signed source determinism",
    )
    expected_determinism = {
        "generator": recipe["generator"],
        "seed": fixture["determinism"]["seed"],
        "component_seed_xors": recipe["component_seed_xors"],
        "coefficient_absolute_bound": recipe["coefficient_absolute_bound"],
        "generator_draw_count": (
            len(recipe["component_seed_xors"]) * resolved["polynomial_degree"]
        ),
    }
    if determinism != expected_determinism:
        fail("exact signed source determinism differs from the fixture recipe")
    layout = expect_keys(
        source["layout"],
        {"component_count", "coefficient_count", "ordering"},
        "exact signed source layout",
    )
    component_count = len(recipe["component_seed_xors"])
    degree = resolved["polynomial_degree"]
    if layout != {
        "component_count": component_count,
        "coefficient_count": degree,
        "ordering": "component,coefficient",
    }:
        fail("exact signed source layout differs from the fixture context")
    if source["source_id"] != "signed_coefficients.default":
        fail("exact signed source identifier changed")
    data = load_binary(
        binary_path,
        source["binary"],
        EXACT_SOURCE_MAGIC,
        "ace.retained_ckks.signed_int64le/1.0.0",
    )
    flat_coefficients = read_i64(
        data, source["signed_coefficients"], "exact signed coefficients"
    )
    validate_canonical_offsets(
        [source["signed_coefficients"]], len(data), "exact signed source"
    )
    if len(flat_coefficients) != component_count * degree:
        fail("exact signed coefficient count differs from its layout")
    components = [
        flat_coefficients[index * degree : (index + 1) * degree]
        for index in range(component_count)
    ]
    bound = recipe["coefficient_absolute_bound"]
    if any(abs(value) > bound for value in flat_coefficients):
        fail("exact signed coefficient exceeds the fixture bound")
    expected_coefficients: list[int] = []
    width = 2 * bound + 1
    for seed_xor in recipe["component_seed_xors"]:
        generator = SplitMix64(fixture["determinism"]["seed"] ^ seed_xor)
        expected_coefficients.extend(
            int(generator.next_u64() % width) - bound for _ in range(degree)
        )
    if flat_coefficients != expected_coefficients:
        fail("exact signed coefficients do not reproduce the fixture recipe")

    label_by_symbol = {
        "0": "0",
        "N/2": "N_over_2",
        "N": "N",
        "3N/2": "3N_over_2",
        "2N-1": "2N_minus_1",
        "2N+1": "2N_plus_1",
    }
    expected_records: list[dict[str, Any]] = [
        {
            "case_id": "exact_algebraic.raise_mod",
            "operation": "raise_mod",
            "normalized_power": None,
            "source_id": source["source_id"],
        }
    ]
    for symbol in fixture["monomial_powers"]:
        expected_records.append(
            {
                "case_id": f"exact_algebraic.mul_mono.{label_by_symbol[symbol]}",
                "operation": "mul_mono",
                "normalized_power": (
                    {
                        "0": 0,
                        "N/2": degree // 2,
                        "N": degree,
                        "3N/2": 3 * degree // 2,
                        "2N-1": 2 * degree - 1,
                        "2N+1": 1,
                    }[symbol]
                ),
                "source_id": source["source_id"],
            }
        )
    expected_records.append(
        {
            "case_id": "exact_algebraic.mul_mono.inverse_composition",
            "operation": "mul_mono_inverse_composition",
            "normalized_power": 0,
            "source_id": source["source_id"],
        }
    )
    if source["records"] != expected_records:
        fail("exact signed source case order or contract differs from the fixture")
    return components, source


def compare_exact(
    exact_source_json: Path,
    exact_source_binary: Path,
    observed_json: Path,
    observed_binary: Path,
    fixture: dict[str, Any],
    fixture_sha256: str,
    context_sha256: str,
    resolved: dict[str, Any],
    expected_first_data_chain_index: int,
) -> dict[str, Any]:
    signed_components, exact_source = load_exact_source(
        exact_source_json,
        exact_source_binary,
        fixture,
        fixture_sha256,
        context_sha256,
        resolved,
    )
    observed = expect_keys(
        load_json(observed_json),
        {
            "schema_version",
            "fixture_sha256",
            "context_manifest_sha256",
            "exact_source_json_sha256",
            "exact_source_binary_sha256",
            "ordered_data_q_moduli",
            "first_data_chain_index",
            "conversion_convention",
            "binary",
            "records",
        },
        "exact observed result",
    )
    if observed["schema_version"] != EXACT_OBSERVED_SCHEMA:
        fail("exact observed schema mismatch")
    if observed["fixture_sha256"] != fixture_sha256 or observed["context_manifest_sha256"] != context_sha256:
        fail("exact observed binding mismatch")
    if (
        observed["exact_source_json_sha256"] != sha256_path(exact_source_json)
        or observed["exact_source_binary_sha256"]
        != sha256_path(exact_source_binary)
    ):
        fail("exact observed result references another signed source")
    if observed["conversion_convention"] != exact_source["conversion_convention"]:
        fail("exact observed conversion convention differs from its signed source")
    moduli = observed["ordered_data_q_moduli"]
    first_data_chain_index = observed["first_data_chain_index"]
    integer(first_data_chain_index, "exact first_data_chain_index")
    if first_data_chain_index != expected_first_data_chain_index:
        fail(
            "exact first_data_chain_index differs from the Phantom provider "
            f"result: observed {first_data_chain_index}, "
            f"expected {expected_first_data_chain_index}"
        )
    if not isinstance(moduli, list) or len(moduli) != resolved["full_data_q_count"]:
        fail("exact runtime modulus count differs from the context manifest")
    for index, modulus in enumerate(moduli):
        if (
            isinstance(modulus, bool)
            or not isinstance(modulus, int)
            or modulus < 3
            or modulus.bit_length() != resolved["data_q_bit_sizes"][index]
        ):
            fail(f"exact runtime modulus {index} disagrees with the context manifest")
    observed_data = load_binary(
        observed_binary,
        observed["binary"],
        EXACT_MAGIC,
        "ace.retained_ckks.rns_uint64le/1.0.0",
    )
    observed_records = observed["records"]
    degree = resolved["polynomial_degree"]
    contracts = [
        ("exact_runtime.raise_mod", "raise_mod", None),
        ("exact_runtime.mul_mono.0", "mul_mono", 0),
        ("exact_runtime.mul_mono.N_over_2", "mul_mono", degree // 2),
        ("exact_runtime.mul_mono.N", "mul_mono", degree),
        ("exact_runtime.mul_mono.3N_over_2", "mul_mono", 3 * degree // 2),
        ("exact_runtime.mul_mono.2N_minus_1", "mul_mono", 2 * degree - 1),
        ("exact_runtime.mul_mono.2N_plus_1", "mul_mono", 1),
        (
            "exact_runtime.mul_mono.inverse_composition",
            "mul_mono_inverse_composition",
            0,
        ),
    ]
    expected_case_ids = [case_id for case_id, _operation, _power in contracts]
    contract_by_id = {
        case_id: (operation, normalized_power)
        for case_id, operation, normalized_power in contracts
    }
    if not isinstance(observed_records, list) or any(
        not isinstance(record, dict) for record in observed_records
    ):
        fail("exact observed records must be an array of objects")
    validate_exact_case_order(
        [record.get("case_id") for record in observed_records],
        expected_case_ids,
        "exact runtime",
    )
    evidence_records: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    total = 0
    all_mismatches: list[dict[str, Any]] = []
    for record_index, actual_record in enumerate(observed_records):
        actual_record = expect_keys(
            actual_record,
            {
                "case_id",
                "operation",
                "normalized_power",
                "source_metadata",
                "source_metadata_after",
                "result_metadata",
                "source",
                "source_after",
                "actual",
                "layout",
            },
            f"exact observed record {record_index}",
        )
        case_id = actual_record["case_id"]
        expected_operation, expected_power = contract_by_id[case_id]
        if actual_record["operation"] != expected_operation:
            fail(
                f"{case_id}.operation is {actual_record['operation']!r}, "
                f"expected {expected_operation!r}"
            )
        if actual_record["normalized_power"] != expected_power or (
            expected_power is not None
            and (
                isinstance(actual_record["normalized_power"], bool)
                or not isinstance(actual_record["normalized_power"], int)
            )
        ):
            fail(
                f"{case_id}.normalized_power is "
                f"{actual_record['normalized_power']!r}, expected {expected_power!r}"
            )
        source_values = read_u64(observed_data, actual_record["source"], f"exact observed source {record_index}")
        source_after_values = read_u64(
            observed_data,
            actual_record["source_after"],
            f"exact observed source after {record_index}",
        )
        if source_after_values != source_values:
            fail(f"exact operation mutated its source for {actual_record['case_id']}")
        metadata_keys = {
            "active_q_count",
            "ciphertext_size",
            "ntt",
            "chain_index",
            "scale_degree",
            "raw_scale",
        }
        source_metadata = expect_keys(
            actual_record["source_metadata"], metadata_keys, f"exact source metadata {record_index}"
        )
        source_metadata_after = expect_keys(
            actual_record["source_metadata_after"], metadata_keys, f"exact source-after metadata {record_index}"
        )
        result_metadata = expect_keys(
            actual_record["result_metadata"], metadata_keys, f"exact result metadata {record_index}"
        )
        if source_metadata_after != source_metadata:
            fail(f"exact operation mutated source metadata for {actual_record['case_id']}")
        for name, metadata in (("source", source_metadata), ("result", result_metadata)):
            integer(
                metadata["active_q_count"],
                f"exact {name} metadata.active_q_count",
                1,
            )
            integer(
                metadata["ciphertext_size"],
                f"exact {name} metadata.ciphertext_size",
                1,
            )
            integer(
                metadata["chain_index"],
                f"exact {name} metadata.chain_index",
            )
            integer(
                metadata["scale_degree"],
                f"exact {name} metadata.scale_degree",
                1,
            )
            if metadata["ciphertext_size"] != 2 or metadata["ntt"] is not False:
                fail(f"exact {name} metadata requires size-2 coefficient form")
            if metadata["scale_degree"] != 1:
                fail(f"exact {name} metadata scale degree must be 1")
            if finite_number(metadata["raw_scale"], f"exact {name} metadata.raw_scale") <= 0:
                fail(f"exact {name} metadata raw scale must be positive")
        if result_metadata["raw_scale"] != source_metadata["raw_scale"]:
            fail(f"exact operation changed raw scale for {case_id}")
        if result_metadata["scale_degree"] != source_metadata["scale_degree"]:
            fail(f"exact operation changed scale degree for {case_id}")
        actual_values = read_u64(observed_data, actual_record["actual"], f"exact observed result {record_index}")
        descriptors.extend(
            (
                actual_record["source"],
                actual_record["source_after"],
                actual_record["actual"],
            )
        )
        layout = expect_keys(
            actual_record["layout"],
            {
                "component_count",
                "source_modulus_count",
                "result_modulus_count",
                "coefficient_count",
                "ordering",
            },
            f"exact observed layout {record_index}",
        )
        component_count = layout["component_count"]
        source_modulus_count = layout["source_modulus_count"]
        result_modulus_count = layout["result_modulus_count"]
        coefficient_count = layout["coefficient_count"]
        for name, value in (
            ("component_count", component_count),
            ("source_modulus_count", source_modulus_count),
            ("result_modulus_count", result_modulus_count),
            ("coefficient_count", coefficient_count),
        ):
            integer(value, f"exact observed layout.{name}", 1)
        if component_count != 2 or coefficient_count != resolved["polynomial_degree"]:
            fail(f"exact observed shape mismatch for {actual_record['case_id']}")
        if layout["ordering"] != "component,modulus,coefficient":
            fail("exact observed layout ordering is unsupported")
        expected_source_values = component_count * source_modulus_count * coefficient_count
        expected_result_values = component_count * result_modulus_count * coefficient_count
        if (
            len(source_values) != expected_source_values
            or len(source_after_values) != expected_source_values
            or actual_record["source"]["count"] != expected_source_values
            or actual_record["source_after"]["count"] != expected_source_values
        ):
            fail(f"exact source blob count disagrees with layout for {case_id}")
        if (
            len(actual_values) != expected_result_values
            or actual_record["actual"]["count"] != expected_result_values
        ):
            fail(f"exact result blob count disagrees with layout for {case_id}")
        if expected_operation == "raise_mod":
            if source_metadata["active_q_count"] != 1 or result_metadata["active_q_count"] != len(moduli):
                fail("exact raise metadata has the wrong active-Q counts")
            if (
                source_metadata["chain_index"]
                != first_data_chain_index + len(moduli) - 1
                or result_metadata["chain_index"] != first_data_chain_index
            ):
                fail("exact raise metadata has the wrong chain coordinates")
            if source_modulus_count != 1 or result_modulus_count != len(moduli):
                fail("exact raise layout has the wrong modulus counts")
            expected_source = reduce_signed_coefficients(
                signed_components, [moduli[0]]
            )
            if source_values != expected_source:
                fail(
                    "exact raise source is not the exact q0 reduction of the "
                    "provider-neutral signed coefficients"
                )
            if any(value >= moduli[0] for value in source_values):
                fail("exact raise source contains a noncanonical q0 residue")
            expected_values = []
            for component in range(component_count):
                begin = component * coefficient_count
                lifted = centered_lift(source_values[begin : begin + coefficient_count], moduli)
                expected_values.extend(value for modulus_values in lifted for value in modulus_values)
        else:
            if source_metadata["active_q_count"] != len(moduli) or result_metadata["active_q_count"] != len(moduli):
                fail("exact monomial metadata has the wrong active-Q counts")
            if (
                source_metadata["chain_index"] != first_data_chain_index
                or result_metadata["chain_index"] != first_data_chain_index
            ):
                fail("exact monomial metadata has the wrong chain coordinates")
            if source_modulus_count != len(moduli) or result_modulus_count != len(moduli):
                fail("exact monomial layout has the wrong modulus counts")
            expected_source = reduce_signed_coefficients(
                signed_components, moduli
            )
            if source_values != expected_source:
                fail(
                    "exact monomial source is not the exact runtime-prime "
                    "reduction of the provider-neutral signed coefficients"
                )
            expected_values = []
            source_offset = 0
            normalized_power = expected_power
            for _component in range(component_count):
                for modulus in moduli:
                    coefficients = source_values[source_offset : source_offset + coefficient_count]
                    source_offset += coefficient_count
                    if any(value >= modulus for value in coefficients):
                        fail(f"exact monomial source contains a noncanonical residue for {case_id}")
                    if expected_operation == "mul_mono":
                        expected_values.extend(
                            negacyclic_monomial(coefficients, normalized_power, modulus)
                        )
                    elif expected_operation == "mul_mono_inverse_composition":
                        first = negacyclic_monomial(coefficients, 2 * coefficient_count - 1, modulus)
                        expected_values.extend(negacyclic_monomial(first, 1, modulus))
                    else:
                        fail(f"unknown exact operation {expected_operation}")
        if len(actual_values) != len(expected_values):
            fail(f"exact comparison length differs for {actual_record['case_id']}")
        modulus_count = layout["result_modulus_count"]
        mismatches: list[dict[str, Any]] = []
        for flat_index, (expected_value, actual_value) in enumerate(zip(expected_values, actual_values)):
            if expected_value == actual_value:
                continue
            component, remainder = divmod(flat_index, modulus_count * coefficient_count)
            modulus, coefficient = divmod(remainder, coefficient_count)
            mismatch = {
                "case_id": actual_record["case_id"],
                "component": component,
                "modulus": modulus,
                "coefficient": coefficient,
                "expected": expected_value,
                "actual": actual_value,
            }
            mismatches.append(mismatch)
            all_mismatches.append(mismatch)
        total += len(expected_values)
        evidence_records.append(
            {
                "case_id": actual_record["case_id"],
                "operation": actual_record["operation"],
                "normalized_power": actual_record["normalized_power"],
                "source_metadata": source_metadata,
                "source_metadata_after": source_metadata_after,
                "result_metadata": result_metadata,
                "source_observed_sha256": actual_record["source"]["sha256"],
                "source_after_sha256": actual_record["source_after"]["sha256"],
                "expected_sha256": sha256_bytes(struct.pack(f"<{len(expected_values)}Q", *expected_values)),
                "actual_sha256": actual_record["actual"]["sha256"],
                "comparison_count": len(expected_values),
                "mismatches": mismatches,
            }
        )
    validate_canonical_offsets(descriptors, len(observed_data), "exact observed")
    return {
        "schema_version": EXACT_EVIDENCE_SCHEMA,
        "status": "pass" if not all_mismatches else "fail",
        "fixture_sha256": fixture_sha256,
        "context_manifest_sha256": context_sha256,
        "exact_source_json_sha256": sha256_path(exact_source_json),
        "exact_source_binary_sha256": sha256_path(exact_source_binary),
        "observed_sha256": sha256_path(observed_json),
        "ordered_data_q_moduli": moduli,
        "first_data_chain_index": first_data_chain_index,
        "conversion_convention": observed["conversion_convention"],
        "comparison_count": total,
        "mismatch_count": len(all_mismatches),
        "mismatches": all_mismatches,
        "records": evidence_records,
    }


def compare(arguments: argparse.Namespace) -> dict[str, Any]:
    fixture = load_json(arguments.fixture)
    resolved = verify_bindings(
        fixture,
        arguments.context_manifest,
        arguments.generation_attestation,
        arguments.post_ckks_air,
        arguments.production_post_ckks_air,
    )
    fixture_sha256 = sha256_path(arguments.fixture)
    context_sha256 = sha256_path(arguments.context_manifest)
    inputs, analytic_records = load_analytic(
        arguments.analytic_json,
        arguments.analytic_bin,
        fixture_sha256,
        fixture,
        resolved,
    )
    analytic_ids = [record["case_id"] for record in analytic_records]
    expected_order = expected_provider_order(fixture)
    if analytic_ids != expected_order[: len(analytic_ids)]:
        fail("analytic cases are not the exact provider-order prefix")
    identity = load_identity_chain(arguments)
    attested_identifiers = load_provider_attestation(
        arguments.provider_attestation,
        fixture_sha256=fixture_sha256,
        context_sha256=context_sha256,
        generation_sha256=identity["generation_sha256"],
        build_sha256=identity["build_sha256"],
        run_sha256=identity["run_sha256"],
        expected_identifiers=identity["identifiers"],
        ant_json=arguments.ant_json,
        ant_binary=arguments.ant_bin,
        gpu_json=arguments.gpu_json,
        gpu_binary=arguments.gpu_bin,
        exact_observed_json=arguments.exact_observed_json,
        exact_observed_binary=arguments.exact_observed_bin,
    )
    ant, _ant_first_data_chain_index = load_provider(
        arguments.ant_json,
        arguments.ant_bin,
        "ant",
        fixture_sha256,
        context_sha256,
        fixture["qualification_bindings"],
        expected_order,
        resolved,
        attested_identifiers["ant"],
    )
    gpu, gpu_first_data_chain_index = load_provider(
        arguments.gpu_json,
        arguments.gpu_bin,
        "phantom",
        fixture_sha256,
        context_sha256,
        fixture["qualification_bindings"],
        expected_order,
        resolved,
        attested_identifiers["phantom"],
    )
    ant_by_id = {record["case_id"]: record for record in ant}
    gpu_by_id = {record["case_id"]: record for record in gpu}
    analytic_by_id = {record["case_id"]: record for record in analytic_records}
    tolerances = fixture["tolerances"]
    analytic_metrics: list[dict[str, Any]] = []
    ant_metrics: list[dict[str, Any]] = []
    failed = False
    for case_id in expected_order:
        if provider_independent_metadata(
            gpu_by_id[case_id]["metadata"]
        ) != provider_independent_metadata(ant_by_id[case_id]["metadata"]):
            fail(f"GPU and ANT provider-independent metadata differ for {case_id}")
        is_composite = case_id.startswith("composite.")
        absolute = tolerances["composite_absolute"] if is_composite else tolerances["primitive_absolute"]
        relative = tolerances["composite_relative"] if is_composite else tolerances["primitive_relative"]
        if case_id in analytic_by_id:
            result = metric(gpu_by_id[case_id]["decoded_values"], analytic_by_id[case_id]["decoded_values"], absolute_tolerance=absolute, relative_tolerance=relative, hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])
            result["case_id"] = case_id
            analytic_metrics.append(result)
            failed |= result["status"] != "pass"
        result = metric(gpu_by_id[case_id]["decoded_values"], ant_by_id[case_id]["decoded_values"], absolute_tolerance=absolute, relative_tolerance=relative, hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])
        result["case_id"] = case_id
        ant_metrics.append(result)
        failed |= result["status"] != "pass"

    source = inputs["bounded_nonperiodic"]
    metamorphic: dict[str, bool] = {}
    metamorphic["conjugate_twice_identity"] = metric(gpu_by_id["conjugate_twice.bounded_nonperiodic"]["decoded_values"], source, absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    metamorphic["raise_mod_identity"] = metric(gpu_by_id["raise_mod.bounded_nonperiodic"]["decoded_values"], source, absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    zero_id = "rotate_batch.bounded_nonperiodic.output_1.step_0"
    duplicate_a = "rotate_batch.bounded_nonperiodic.output_0.step_5"
    duplicate_b = "rotate_batch.bounded_nonperiodic.output_3.step_5"
    metamorphic["zero_step_identity"] = metric(gpu_by_id[zero_id]["decoded_values"], source, absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    metamorphic["duplicate_steps_equal"] = gpu_by_id[duplicate_a]["decoded_values"] == gpu_by_id[duplicate_b]["decoded_values"]
    mono_zero = gpu_by_id["mul_mono.0.bounded_nonperiodic"]["decoded_values"]
    mono_n = gpu_by_id["mul_mono.N.bounded_nonperiodic"]["decoded_values"]
    metamorphic["monomial_zero_identity"] = metric(mono_zero, source, absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    metamorphic["monomial_degree_negation"] = metric(mono_n, [-value for value in source], absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    metamorphic["monomial_inverse_composition"] = metric(gpu_by_id["mul_mono.inverse_composition.bounded_nonperiodic"]["decoded_values"], source, absolute_tolerance=tolerances["primitive_absolute"], relative_tolerance=tolerances["primitive_relative"], hard_maximum=tolerances["hard_maximum_absolute"], relative_floor=tolerances["relative_metric_floor"])["status"] == "pass"
    failed |= not all(metamorphic.values())

    exact = compare_exact(
        arguments.exact_source_json,
        arguments.exact_source_bin,
        arguments.exact_observed_json,
        arguments.exact_observed_bin,
        fixture,
        fixture_sha256,
        context_sha256,
        resolved,
        gpu_first_data_chain_index,
    )
    if exact["mismatch_count"] != tolerances["exact_mismatches"]:
        failed = True
    write_json(arguments.exact_output_json, exact)
    exact_errors = [
        abs(mismatch["actual"] - mismatch["expected"])
        for mismatch in exact["mismatches"]
    ]
    if exact_errors:
        maximum_exact_error = max(exact_errors)
        maximum_exact_index = exact["mismatches"][
            exact_errors.index(maximum_exact_error)
        ]
        exact_relative = [
            error / abs(mismatch["expected"])
            for error, mismatch in zip(exact_errors, exact["mismatches"])
            if abs(mismatch["expected"]) >= tolerances["relative_metric_floor"]
        ]
        exact_mean = sum(exact_errors) / exact["comparison_count"]
        exact_rmse = math.sqrt(
            sum(error * error for error in exact_errors)
            / exact["comparison_count"]
        )
    else:
        maximum_exact_error = 0
        maximum_exact_index = None
        exact_relative = []
        exact_mean = 0
        exact_rmse = 0
    result = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "fail" if failed else "pass",
        "fixture_sha256": fixture_sha256,
        "context_manifest_sha256": context_sha256,
        "gpu_vs_analytic": analytic_metrics,
        "gpu_vs_ant": ant_metrics,
        "gpu_vs_exact": {
            "status": "pass" if exact["mismatch_count"] == 0 else "fail",
            "comparison_count": exact["comparison_count"],
            "mismatch_count": exact["mismatch_count"],
            "maximum_absolute_error": maximum_exact_error,
            "maximum_absolute_error_index": maximum_exact_index,
            "maximum_relative_error_above_floor": max(exact_relative, default=0),
            "mean_absolute_error": exact_mean,
            "root_mean_square_error": exact_rmse,
            "estimated_precision_bits": None,
        },
        "metamorphic_checks": metamorphic,
        "artifact_hashes": {
            "ace_source_manifest_sha256": identity[
                "ace_source_manifest_sha256"
            ],
            "phantom_source_manifest_sha256": identity[
                "phantom_source_manifest_sha256"
            ],
            "generation_attestation_sha256": identity["generation_sha256"],
            "build_attestation_sha256": identity["build_sha256"],
            "run_attestation_sha256": identity["run_sha256"],
            "analytic_json_sha256": sha256_path(arguments.analytic_json),
            "ant_json_sha256": sha256_path(arguments.ant_json),
            "gpu_json_sha256": sha256_path(arguments.gpu_json),
            "provider_attestation_sha256": sha256_path(
                arguments.provider_attestation
            ),
            "exact_source_json_sha256": sha256_path(
                arguments.exact_source_json
            ),
            "exact_source_binary_sha256": sha256_path(
                arguments.exact_source_bin
            ),
            "exact_evidence_sha256": sha256_path(arguments.exact_output_json),
        },
    }
    write_json(arguments.output_json, result)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fixture", "context-manifest", "post-ckks-air", "ant-post-ckks-air",
        "production-post-ckks-air",
        "ace-source-manifest", "phantom-source-manifest",
        "generation-attestation", "generation-fixture",
        "emitted-context-manifest", "resource-manifest",
        "generated-ant-source", "generated-phantom-source",
        "ant-executable", "phantom-executable",
        "build-attestation", "run-attestation",
        "provider-attestation",
        "analytic-json", "analytic-bin", "ant-json", "ant-bin", "gpu-json", "gpu-bin",
        "exact-source-json", "exact-source-bin",
        "exact-observed-json", "exact-observed-bin",
        "exact-output-json", "output-json",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        result = compare(parse_arguments())
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return 0 if result["status"] == "pass" else 1
    except (ComparisonError, RetainedFixtureError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
