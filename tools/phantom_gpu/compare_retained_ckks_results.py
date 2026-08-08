#!/usr/bin/env python3
"""Strictly compare retained CKKS GPU results with independent references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Sequence


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from generate_retained_ckks_fixtures import (  # noqa: E402
    ANALYTIC_SCHEMA,
    DECODED_MAGIC,
    EXACT_MAGIC,
    RetainedFixtureError,
    centered_lift,
    expected_metadata,
    load_json,
    negacyclic_monomial,
    sha256_bytes,
    sha256_path,
    verify_bindings,
    write_json,
)


PROVIDER_SCHEMA = "ace.phantom.retained_ckks.provider-result/1.0.0"
EXACT_OBSERVED_SCHEMA = "ace.phantom.retained_ckks.exact-observed/1.0.0"
COMPARISON_SCHEMA = "ace.phantom.retained_ckks.comparison/1.0.0"
EXACT_EVIDENCE_SCHEMA = "ace.phantom.retained_ckks.exact-evidence/1.0.0"


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


def hex_digest(value: Any, length: int, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail(f"{context} must be a lowercase hexadecimal digest")
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
    for field, expected_value in expected.items():
        if metadata[field] != expected_value:
            fail(f"{context}.{field} is {metadata[field]!r}, expected {expected_value!r}")
    if isinstance(metadata["chain_index"], bool) or not isinstance(
        metadata["chain_index"], int
    ):
        fail(f"{context}.chain_index must be an integer")
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
    bindings: dict[str, Any],
    slots: int,
) -> tuple[dict[str, list[complex]], list[dict[str, Any]]]:
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
    if value["qualification_bindings"] != bindings:
        fail("analytic reference qualification bindings mismatch")
    data = load_binary(
        binary_path,
        value["binary"],
        DECODED_MAGIC,
        "ace.retained_ckks.complex_float64le/1.0.0",
    )
    inputs: dict[str, list[complex]] = {}
    descriptors: list[dict[str, Any]] = []
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
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value["records"]):
        expected_keys = {"case_id", "operation", "metadata", "source_input_id", "values"}
        if item.get("operation") == "rotate_batch":
            expected_keys.add("batch_step")
        item = expect_keys(item, expected_keys, f"analytic record {index}")
        if item["case_id"] in seen:
            fail(f"duplicate analytic case {item['case_id']}")
        seen.add(item["case_id"])
        item = dict(item)
        item["decoded_values"] = read_complex(data, item["values"], item["case_id"])
        descriptors.append(item["values"])
        if len(item["decoded_values"]) != slots:
            fail(f"analytic case {item['case_id']} has the wrong slot count")
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
) -> list[dict[str, Any]]:
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
                "values",
            },
            f"{provider_name} record {index}",
        )
        case_id = record["case_id"]
        observed_order.append(case_id)
        raised = record["operation"] in ("raise_mod", "composite")
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
        if record["operation"] == "rotate_batch":
            if not isinstance(token, str) or not token or token in ownership_tokens:
                fail(f"{provider_name}.{case_id} lacks independent batch ownership")
            ownership_tokens.add(token)
        elif token is not None:
            fail(f"{provider_name}.{case_id} has an unexpected ownership token")
        materialized = dict(record)
        materialized["metadata"] = metadata
        materialized["decoded_values"] = read_complex(data, record["values"], f"{provider_name}.{case_id}")
        descriptors.append(record["values"])
        if len(materialized["decoded_values"]) != resolved["logical_slots"]:
            fail(f"{provider_name}.{case_id} has the wrong slot count")
        records.append(materialized)
    if observed_order != list(expected_order):
        fail(f"{provider_name} case order differs from the frozen order")
    validate_canonical_offsets(descriptors, len(data), provider_name)
    return records


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


def compare_exact(
    observed_json: Path,
    observed_binary: Path,
    fixture_sha256: str,
    context_sha256: str,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    observed = expect_keys(
        load_json(observed_json),
        {
            "schema_version",
            "fixture_sha256",
            "context_manifest_sha256",
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
    moduli = observed["ordered_data_q_moduli"]
    first_data_chain_index = observed["first_data_chain_index"]
    if (
        isinstance(first_data_chain_index, bool)
        or not isinstance(first_data_chain_index, int)
        or first_data_chain_index < 0
    ):
        fail("exact first_data_chain_index must be nonnegative")
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
    expected_case_ids = ["exact_runtime.raise_mod"] + [
        f"exact_runtime.mul_mono.{label}"
        for label in (
            "0", "N_over_2", "N", "3N_over_2", "2N_minus_1", "2N_plus_1"
        )
    ] + ["exact_runtime.mul_mono.inverse_composition"]
    if [record.get("case_id") for record in observed_records] != expected_case_ids:
        fail("exact runtime case order mismatch")
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
        source_values = read_u64(observed_data, actual_record["source"], f"exact observed source {record_index}")
        source_after_values = read_u64(
            observed_data,
            actual_record["source_after"],
            f"exact observed source after {record_index}",
        )
        if source_after_values != source_values:
            fail(f"exact operation mutated its source for {actual_record['case_id']}")
        metadata_keys = {"active_q_count", "ciphertext_size", "ntt", "chain_index"}
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
        for name, metadata in (
            ("source", source_metadata), ("result", result_metadata)
        ):
            if metadata["ciphertext_size"] != 2 or metadata["ntt"] is not False:
                fail(f"exact {name} metadata requires size-2 coefficient form")
            if isinstance(metadata["chain_index"], bool) or not isinstance(metadata["chain_index"], int):
                fail(f"exact {name} chain index must be an integer")
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
        coefficient_count = layout["coefficient_count"]
        if component_count != 2 or coefficient_count != resolved["polynomial_degree"]:
            fail(f"exact observed shape mismatch for {actual_record['case_id']}")
        if layout["ordering"] != "component,modulus,coefficient":
            fail("exact observed layout ordering is unsupported")
        if actual_record["operation"] == "raise_mod":
            if source_metadata["active_q_count"] != 1 or result_metadata["active_q_count"] != len(moduli):
                fail("exact raise metadata has the wrong active-Q counts")
            if (
                source_metadata["chain_index"]
                != first_data_chain_index + len(moduli) - 1
                or result_metadata["chain_index"] != first_data_chain_index
            ):
                fail("exact raise metadata has the wrong chain coordinates")
            if layout["source_modulus_count"] != 1 or layout["result_modulus_count"] != len(moduli):
                fail("exact raise layout has the wrong modulus counts")
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
            if layout["source_modulus_count"] != len(moduli) or layout["result_modulus_count"] != len(moduli):
                fail("exact monomial layout has the wrong modulus counts")
            expected_values = []
            source_offset = 0
            normalized_power = actual_record["normalized_power"]
            if (
                isinstance(normalized_power, bool)
                or not isinstance(normalized_power, int)
                or normalized_power < 0
                or normalized_power >= 2 * coefficient_count
            ):
                fail("exact monomial power is not normalized into [0, 2N)")
            for _component in range(component_count):
                for modulus in moduli:
                    coefficients = source_values[source_offset : source_offset + coefficient_count]
                    source_offset += coefficient_count
                    if actual_record["operation"] == "mul_mono":
                        expected_values.extend(
                            negacyclic_monomial(coefficients, actual_record["normalized_power"], modulus)
                        )
                    elif actual_record["operation"] == "mul_mono_inverse_composition":
                        first = negacyclic_monomial(coefficients, 2 * coefficient_count - 1, modulus)
                        expected_values.extend(negacyclic_monomial(first, 1, modulus))
                    else:
                        fail(f"unknown exact operation {actual_record['operation']}")
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
        "observed_sha256": sha256_path(observed_json),
        "ordered_data_q_moduli": moduli,
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
        arguments.compiler_invocation,
        arguments.post_ckks_air,
        arguments.production_post_ckks_air,
    )
    fixture_sha256 = sha256_path(arguments.fixture)
    context_sha256 = sha256_path(arguments.context_manifest)
    inputs, analytic_records = load_analytic(
        arguments.analytic_json,
        arguments.analytic_bin,
        fixture_sha256,
        fixture["qualification_bindings"],
        resolved["logical_slots"],
    )
    analytic_ids = [record["case_id"] for record in analytic_records]
    expected_order = analytic_ids + [
        record["id"]
        for record in fixture["decoded_cases"]
        if record["id"] not in analytic_ids
        and record["id"] != "rotate_batch.bounded_nonperiodic"
    ]
    ant = load_provider(arguments.ant_json, arguments.ant_bin, "ant", fixture_sha256, context_sha256, fixture["qualification_bindings"], expected_order, resolved)
    gpu = load_provider(arguments.gpu_json, arguments.gpu_bin, "phantom", fixture_sha256, context_sha256, fixture["qualification_bindings"], expected_order, resolved)
    ant_by_id = {record["case_id"]: record for record in ant}
    gpu_by_id = {record["case_id"]: record for record in gpu}
    analytic_by_id = {record["case_id"]: record for record in analytic_records}
    tolerances = fixture["tolerances"]
    analytic_metrics: list[dict[str, Any]] = []
    ant_metrics: list[dict[str, Any]] = []
    failed = False
    for case_id in expected_order:
        if gpu_by_id[case_id]["metadata"] != ant_by_id[case_id]["metadata"]:
            fail(f"GPU and ANT metadata differ for {case_id}")
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
        arguments.exact_observed_json,
        arguments.exact_observed_bin,
        fixture_sha256,
        context_sha256,
        resolved,
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
            "analytic_json_sha256": sha256_path(arguments.analytic_json),
            "ant_json_sha256": sha256_path(arguments.ant_json),
            "gpu_json_sha256": sha256_path(arguments.gpu_json),
            "exact_evidence_sha256": sha256_path(arguments.exact_output_json),
        },
    }
    write_json(arguments.output_json, result)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fixture", "context-manifest", "compiler-invocation", "post-ckks-air",
        "production-post-ckks-air",
        "analytic-json", "analytic-bin", "ant-json", "ant-bin", "gpu-json", "gpu-bin",
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
