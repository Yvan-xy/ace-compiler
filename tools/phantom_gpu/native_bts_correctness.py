#!/usr/bin/env python3
"""Prepare and verify native Phantom bootstrap correctness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Sequence

import bootstrap_correctness as bootstrap


COMPATIBILITY_SCHEMA = "ace.phantom.native-bts-compatibility/1.0.0"
CLEAR_SCHEMA = "ace.phantom.native-bts-clear-inputs/1.0.0"
RAW_SCHEMA = "ace.phantom.native-bts-raw-execution/1.0.0"
PROVIDER_SCHEMA = "ace.phantom.bootstrap-native-phantom/1.0.0"
COMPARISON_SCHEMA = "ace.phantom.native-bts-comparison/1.0.0"
SANITIZER_SCHEMA = "ace.phantom.native-bts-compute-sanitizer/1.0.0"
LINK_AUDIT_SCHEMA = "ace.phantom.native-bts-link-audit/1.0.0"
PAIR = struct.Struct("<dd")


class NativeCorrectnessError(ValueError):
    pass


def fail(message: str) -> None:
    raise NativeCorrectnessError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def digest(path: Path) -> str:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as error:
        fail(f"cannot hash {path}: {error}")


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=bootstrap._object,
            parse_constant=lambda value: fail(f"non-finite JSON number: {value}"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"top-level JSON value in {path} must be an object")
    return value


def write_fresh(path: Path, contents: bytes) -> None:
    if path.exists():
        fail(f"refusing to replace existing output {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(contents)
    except OSError as error:
        fail(f"cannot create {path}: {error}")


def sha(value: Any, context: str) -> str:
    try:
        return bootstrap.sha(value, context)
    except bootstrap.CorrectnessError as error:
        fail(str(error))


def require_keys(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    try:
        return bootstrap.require_keys(value, keys, context)
    except bootstrap.CorrectnessError as error:
        fail(str(error))


def _authorities(arguments: argparse.Namespace) -> dict[str, Path]:
    return {
        name: Path(getattr(arguments, name)) for name in bootstrap.BINDING_INPUTS
    }


def _validate_baseline(
    arguments: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    list[tuple[str, list[complex], dict[str, Any]]],
]:
    paths = _authorities(arguments)
    try:
        invocation, semantics, context, bindings = bootstrap.validate_authorities(
            paths
        )
        fixture_path = Path(arguments.fixture)
        fixture = bootstrap.load_json(fixture_path)
        bootstrap.validate_fixture(fixture)
        if fixture["bindings"] != bindings:
            fail("fixture bindings differ from the closed baseline authorities")
        slots = bootstrap.uint(context["logical_slot_capacity"], "logical slots", 1)
        generated = bootstrap.load_provider(
            Path(arguments.baseline_gpu_record),
            Path(arguments.baseline_gpu_values),
            "generated-phantom",
            fixture,
            digest(fixture_path),
            bindings,
            slots,
            semantics=semantics,
        )
    except bootstrap.CorrectnessError as error:
        fail(str(error))

    comparison = load(Path(arguments.baseline_comparison))
    if (
        comparison.get("schema_version") != bootstrap.COMPARISON_SCHEMA
        or comparison.get("status") != "pass"
        or comparison.get("fixture_sha256") != digest(Path(arguments.fixture))
        or comparison.get("bindings") != bindings
        or len(comparison.get("comparisons", [])) != 5
    ):
        fail("closed generated-bootstrap comparison is not a passing baseline")
    sanitizer = load(Path(arguments.baseline_sanitizer))
    if (
        sanitizer.get("schema_version") != bootstrap.SANITIZER_SCHEMA
        or sanitizer.get("status") != "pass"
        or sanitizer.get("exit_status") != 0
        or sanitizer.get("error_summary_occurrences") != 1
        or sanitizer.get("error_count") != 0
    ):
        fail("closed generated-bootstrap sanitizer record is not passing")
    gpu_document = load(Path(arguments.baseline_gpu_record))
    baseline_executable_sha = gpu_document.get("execution", {}).get(
        "executable_sha256"
    )
    sha(baseline_executable_sha, "baseline generated executable SHA-256")
    if sanitizer.get("bindings", {}).get("gpu_executable_sha256") != baseline_executable_sha:
        fail("baseline record and sanitizer bind different generated executables")
    return fixture, invocation, semantics, context, bindings, generated


def _source_identity(path: Path, kind: str) -> dict[str, Any]:
    value = load(path)
    if value.get("kind") != kind:
        fail(f"{kind} source manifest has the wrong kind")
    commit = value.get("commit")
    tree = value.get("tree")
    if not isinstance(commit, str) or len(commit) != 40:
        fail(f"{kind} source manifest has an invalid commit")
    if not isinstance(tree, str) or len(tree) != 40:
        fail(f"{kind} source manifest has an invalid tree")
    return {"commit": commit, "tree": tree, "manifest_sha256": digest(path)}


def prepare(arguments: argparse.Namespace) -> None:
    fixture, invocation, semantics, context, bindings, generated = _validate_baseline(
        arguments
    )
    fixture_path = Path(arguments.fixture)
    fixture_sha = digest(fixture_path)
    context_sha = bindings["context_manifest_sha256"]
    slots = context["logical_slot_capacity"]
    degree = context["polynomial_degree"]
    if degree != slots * 2 or slots < 2 or slots & (slots - 1):
        fail("native API requires a power-of-two full-packed context")
    if context.get("packing") != "full":
        fail("native API compatibility is restricted to full packing")
    data_q = context.get("data_q_bit_sizes")
    special_p = context.get("special_p_bit_sizes")
    if not isinstance(data_q, list) or not data_q or not isinstance(special_p, list):
        fail("compiler context lacks explicit Q/P modulus lists")
    if context.get("input_level") != 1:
        fail("native bootstrap compatibility requires the emitted bottom input level")

    clear_cases = bootstrap.materialize_cases(fixture, slots)
    raw_payload = bytearray()
    clear_records = []
    for case_id, values in clear_cases:
        offset = len(raw_payload)
        for value in values:
            raw_payload.extend(PAIR.pack(value.real, value.imag))
        clear_records.append(
            {
                "case_id": case_id,
                "offset_bytes": offset,
                "value_count": len(values),
            }
        )
    clear_output = Path(arguments.clear_values_output)
    write_fresh(clear_output, bytes(raw_payload))
    clear_record = {
        "schema_version": CLEAR_SCHEMA,
        "status": "pass",
        "fixture_sha256": fixture_sha,
        "context_manifest_sha256": context_sha,
        "case_order": [item[0] for item in clear_cases],
        "logical_slots": slots,
        "records": clear_records,
        "values": {
            "format": "complex-float64le-pairs-without-header",
            "sha256": digest(clear_output),
            "size_bytes": len(raw_payload),
        },
    }
    clear_record_output = Path(arguments.clear_record_output)
    write_fresh(clear_record_output, canonical_bytes(clear_record))

    current_ace = _source_identity(Path(arguments.current_ace_source_manifest), "ace")
    current_phantom = _source_identity(
        Path(arguments.current_phantom_source_manifest), "phantom"
    )
    baseline_ace = _source_identity(Path(arguments.ace_source_manifest), "ace")
    baseline_phantom = _source_identity(
        Path(arguments.phantom_source_manifest), "phantom"
    )
    if current_phantom["commit"] != baseline_phantom["commit"]:
        fail("native qualification changed the closed Phantom revision")

    generated_record = load(Path(arguments.baseline_gpu_record))
    generated_executable_sha = generated_record["execution"]["executable_sha256"]
    output_contract = semantics["output_air_contract"]
    if (
        output_contract["scale_degree"] != 1
        or output_contract["ciphertext_size"] != 2
        or output_contract["logical_slots"] != slots
        or output_contract["ntt_state"] is not True
    ):
        fail("emitted bootstrap output has incompatible fixed invariants")

    compatibility = {
        "schema_version": COMPATIBILITY_SCHEMA,
        "status": "compatible",
        "scope": "native-phantom-bootstrap-3-correctness",
        "source_identity": {
            "qualification": {"ace": current_ace, "phantom": current_phantom},
            "closed_generated_baseline": {
                "ace": baseline_ace,
                "phantom": baseline_phantom,
            },
        },
        "bindings": {
            **bindings,
            "fixture_sha256": fixture_sha,
            "clear_record_sha256": digest(clear_record_output),
            "clear_values_sha256": digest(clear_output),
            "baseline_gpu_record_sha256": digest(Path(arguments.baseline_gpu_record)),
            "baseline_gpu_values_sha256": digest(Path(arguments.baseline_gpu_values)),
            "baseline_comparison_sha256": digest(Path(arguments.baseline_comparison)),
            "baseline_sanitizer_sha256": digest(Path(arguments.baseline_sanitizer)),
            "baseline_generated_executable_sha256": generated_executable_sha,
        },
        "compiler_context": {
            "polynomial_degree": degree,
            "logical_slots": slots,
            "data_q_bit_sizes": data_q,
            "special_p_bit_sizes": special_p,
            "q_part_count": context["q_part_count"],
            "hamming_weight": context["hamming_weight"],
            "security_level": context["security_level"],
            "input_level": context["input_level"],
            "scaling_modulus_bits": context["scaling_modulus_bits"],
        },
        "native_configuration": {
            "api": "Bootstrapper::bootstrap_3",
            "constructor": {
                "loge": 10,
                "logn": int(math.log2(slots)),
                "logNh": int(math.log2(degree // 2)),
                "total_level": len(data_q) - 1,
                "final_scale": math.ldexp(1.0, context["scaling_modulus_bits"]),
                "boundary_K": 25,
                "sin_cos_degree": 59,
                "scale_factor": 2,
                "inverse_degree": 1,
                "enable_slim_relu": False,
            },
            "precomputation": {
                "prepare_mod_polynomial": True,
                "rotation_planner": "addLeftRotKeys_Linear_to_vector_3",
                "slot_vec": [int(math.log2(slots))],
                "linear_transform": "generate_LT_coefficient_3",
            },
            "input": {
                "active_q_count": 1,
                "phantom_chain_index": 1 + len(data_q) - 1,
                "raw_scale": math.ldexp(1.0, context["scaling_modulus_bits"]),
                "ciphertext_size": 2,
                "logical_slots": slots,
                "ntt_state": True,
            },
            "output_invariants": {
                "metadata_convention": "phantom-active-q-count-minus-one",
                "chain_index_relation": "1+data_q_count-active_q_count",
                "minimum_active_q_count": 1,
                "maximum_active_q_count": len(data_q),
                "raw_scale": math.ldexp(1.0, context["scaling_modulus_bits"]),
                "scale_degree": output_contract["scale_degree"],
                "ciphertext_size": output_contract["ciphertext_size"],
                "logical_slots": output_contract["logical_slots"],
                "ntt_state": output_contract["ntt_state"],
            },
        },
        "correctness_contract": {
            "case_order": [item[0] for item in clear_cases],
            "calls_per_case": fixture["repeat_contract"]["calls_per_case"],
            "provider_clear_maximum_absolute": fixture["tolerances"][
                "provider_clear_maximum_absolute"
            ],
            "baseline_differential_maximum_absolute": 0.02,
            "repeat_maximum_absolute": fixture["tolerances"][
                "repeat_maximum_absolute"
            ],
            "post_multiply": fixture["post_operations"]["inputs"][
                "multiply_constant"
            ],
            "post_rotation_step": fixture["post_operations"]["inputs"][
                "rotation_step"
            ],
            "sanitizer_sentinel": "alternating-real-imaginary-signs",
        },
        "representability": {
            "status": "pass",
            "context_adjusted": False,
            "manual_parameter_profile_used": False,
            "full_packing_supported": True,
            "q_and_p_lists_consumed_verbatim": True,
            "native_output_depth_is_provider_schedule": True,
            "generated_output_metadata_is_not_a_native_oracle": True,
        },
    }
    write_fresh(Path(arguments.output), canonical_bytes(compatibility))


def _read_raw_values(path: Path, records: Sequence[dict[str, Any]]) -> list[list[complex]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        fail(f"cannot read raw native values: {error}")
    result: list[list[complex]] = []
    expected_offset = 0
    for index, record in enumerate(records):
        require_keys(
            record,
            {"case_id", "offset_bytes", "value_count", "metadata"},
            f"raw records[{index}]",
        )
        if record["offset_bytes"] != expected_offset:
            fail("raw native values are not contiguous")
        count = bootstrap.uint(record["value_count"], "native value count", 1)
        end = expected_offset + count * PAIR.size
        if end > len(payload):
            fail("raw native values descriptor exceeds payload")
        values = [
            complex(*PAIR.unpack_from(payload, expected_offset + item * PAIR.size))
            for item in range(count)
        ]
        if any(not math.isfinite(item.real) or not math.isfinite(item.imag) for item in values):
            fail("raw native values contain NaN or infinity")
        result.append(values)
        expected_offset = end
    if expected_offset != len(payload):
        fail("raw native values contain unreferenced bytes")
    return result


def finalize(arguments: argparse.Namespace) -> None:
    compatibility_path = Path(arguments.compatibility)
    compatibility = load(compatibility_path)
    if (
        compatibility.get("schema_version") != COMPATIBILITY_SCHEMA
        or compatibility.get("status") != "compatible"
    ):
        fail("native compatibility record is not passing")
    raw = load(Path(arguments.raw_record))
    require_keys(
        raw,
        {
            "schema_version",
            "status",
            "mode",
            "compatibility_sha256",
            "case_order",
            "logical_slots",
            "records",
            "execution",
        },
        "raw native record",
    )
    if (
        raw["schema_version"] != RAW_SCHEMA
        or raw["status"] != "pass"
        or raw["compatibility_sha256"] != digest(compatibility_path)
    ):
        fail("raw native execution provenance differs from compatibility")
    expected_order = compatibility["correctness_contract"]["case_order"]
    if raw["mode"] == "all":
        if raw["case_order"] != expected_order:
            fail("native execution does not contain all ordered cases")
    elif raw["mode"] == "sentinel":
        if raw["case_order"] != [compatibility["correctness_contract"]["sanitizer_sentinel"]]:
            fail("native sentinel execution selected the wrong case")
    else:
        fail("unknown native execution mode")
    if raw["logical_slots"] != compatibility["compiler_context"]["logical_slots"]:
        fail("native execution slot count differs from compatibility")
    values = _read_raw_values(Path(arguments.raw_values), raw["records"])
    value_records = [
        {
            "case_id": record["case_id"],
            "oracle": "native-phantom",
            "values": case_values,
            "metadata": record["metadata"],
        }
        for record, case_values in zip(raw["records"], values)
    ]
    binary, descriptors = bootstrap.write_value_file(
        Path(arguments.values_output),
        compatibility["bindings"]["fixture_sha256"],
        compatibility["bindings"]["context_manifest_sha256"],
        value_records,
    )
    execution = raw["execution"]
    execution["executable_sha256"] = digest(Path(arguments.executable))
    execution["ordinary_library_sha256"] = digest(Path(arguments.ordinary_library))
    execution["native_library_sha256"] = digest(Path(arguments.native_library))
    output = {
        "schema_version": PROVIDER_SCHEMA,
        "provider": "native-phantom",
        "mode": raw["mode"],
        "fixture_sha256": compatibility["bindings"]["fixture_sha256"],
        "compatibility_sha256": digest(compatibility_path),
        "bindings": compatibility["bindings"],
        "case_order": raw["case_order"],
        "logical_slots": raw["logical_slots"],
        "binary": binary,
        "records": descriptors,
        "execution": execution,
    }
    write_fresh(Path(arguments.output), canonical_bytes(output))


def _validate_native_metadata(
    records: list[tuple[str, list[complex], dict[str, Any]]],
    compatibility: dict[str, Any],
    clear: dict[str, list[complex]],
) -> None:
    contract = compatibility["correctness_contract"]
    output = compatibility["native_configuration"]["output_invariants"]
    input_metadata = compatibility["native_configuration"]["input"]
    common_output: dict[str, Any] | None = None
    for case_id, values, metadata in records:
        require_keys(
            metadata,
            {
                "input",
                "bootstrap",
                "metrics_vs_clear",
                "repeatability",
                "post_operations",
                "ownership",
            },
            f"native {case_id} metadata",
        )
        if metadata["input"] != input_metadata:
            fail(f"native {case_id} input metadata differs from compatibility")
        observed = metadata["bootstrap"]
        if set(observed) != {
            "ace_level",
            "active_q_count",
            "phantom_chain_index",
            "raw_scale",
            "scale_degree",
            "ciphertext_size",
            "logical_slots",
            "ntt_state",
        }:
            fail(f"native {case_id} output metadata shape differs")
        active_q_count = observed["active_q_count"]
        if (
            not isinstance(active_q_count, int)
            or isinstance(active_q_count, bool)
            or not output["minimum_active_q_count"]
            <= active_q_count
            <= output["maximum_active_q_count"]
            or observed["ace_level"] != active_q_count - 1
            or observed["phantom_chain_index"]
            != 1
            + len(compatibility["compiler_context"]["data_q_bit_sizes"])
            - active_q_count
        ):
            fail(f"native {case_id} output depth metadata is inconsistent")
        for key in ("scale_degree", "ciphertext_size", "logical_slots", "ntt_state"):
            if observed[key] != output[key]:
                fail(f"native {case_id} output invariant differs at {key}")
        coordinate = math.log2(observed["raw_scale"]) / compatibility[
            "compiler_context"
        ]["scaling_modulus_bits"]
        if not math.isfinite(coordinate) or abs(coordinate - output["scale_degree"]) > 1.0e-4:
            fail(f"native {case_id} raw scale coordinate differs")
        if common_output is None:
            common_output = observed
        elif observed != common_output:
            fail(f"native {case_id} output metadata differs across cases")
        expected_metric = bootstrap.metric(
            values,
            clear[case_id],
            contract["provider_clear_maximum_absolute"],
        )
        try:
            bootstrap.validate_recorded_metric(
                metadata["metrics_vs_clear"], expected_metric, f"native {case_id}"
            )
        except bootstrap.CorrectnessError as error:
            fail(str(error))
        repeat = metadata["repeatability"]
        if (
            repeat.get("calls") != contract["calls_per_case"]
            or repeat.get("independently_owned_executions") is not True
            or repeat.get("maximum_absolute_difference", math.inf)
            > contract["repeat_maximum_absolute"]
            or repeat.get("threshold") != contract["repeat_maximum_absolute"]
        ):
            fail(f"native {case_id} repeatability contract failed")
        call_metadata = repeat.get("call_metadata")
        call_metrics = repeat.get("call_metrics")
        if (
            not isinstance(call_metadata, list)
            or len(call_metadata) != contract["calls_per_case"]
            or any(item != observed for item in call_metadata)
            or not isinstance(call_metrics, list)
            or len(call_metrics) != contract["calls_per_case"]
            or any(
                not isinstance(item, dict) or item.get("status") != "pass"
                for item in call_metrics
            )
        ):
            fail(f"native {case_id} repeat metadata differs")
        post = metadata["post_operations"]
        for name in ("ciphertext_plaintext_multiply", "rotation"):
            metric_value = post.get(name, {}).get("metric")
            if not isinstance(metric_value, dict) or metric_value.get("status") != "pass":
                fail(f"native {case_id} {name} compatibility failed")
            if post[name].get("metadata") != observed:
                fail(f"native {case_id} {name} metadata differs")
        if post["rotation"].get("step") != contract["post_rotation_step"]:
            fail(f"native {case_id} rotation step differs")
        if metadata["ownership"] != {
            "all_owned_objects_released": True,
            "bootstrap_inputs_released": contract["calls_per_case"],
            "bootstrap_results_released": contract["calls_per_case"],
            "post_operation_objects_released": True,
            "teardown_completed": True,
        }:
            fail(f"native {case_id} ownership closure failed")


def compare(arguments: argparse.Namespace) -> None:
    fixture, _, _, context, _, generated = _validate_baseline(arguments)
    compatibility_path = Path(arguments.compatibility)
    compatibility = load(compatibility_path)
    if (
        compatibility.get("schema_version") != COMPATIBILITY_SCHEMA
        or compatibility.get("status") != "compatible"
        or compatibility["bindings"]["fixture_sha256"] != digest(Path(arguments.fixture))
        or compatibility["bindings"]["baseline_gpu_record_sha256"]
        != digest(Path(arguments.baseline_gpu_record))
        or compatibility["bindings"]["baseline_gpu_values_sha256"]
        != digest(Path(arguments.baseline_gpu_values))
    ):
        fail("native compatibility record is stale")
    native_document = load(Path(arguments.native_record))
    if (
        native_document.get("schema_version") != PROVIDER_SCHEMA
        or native_document.get("provider") != "native-phantom"
        or native_document.get("mode") != "all"
        or native_document.get("fixture_sha256")
        != compatibility["bindings"]["fixture_sha256"]
        or native_document.get("compatibility_sha256") != digest(compatibility_path)
        or native_document.get("bindings") != compatibility["bindings"]
    ):
        fail("native provider record provenance differs")
    try:
        native = bootstrap.read_value_file(
            Path(arguments.native_values),
            native_document["binary"],
            native_document["records"],
            native_document["fixture_sha256"],
            compatibility["bindings"]["context_manifest_sha256"],
            "native-phantom",
        )
    except bootstrap.CorrectnessError as error:
        fail(str(error))
    expected_order = compatibility["correctness_contract"]["case_order"]
    slots = context["logical_slot_capacity"]
    if (
        native_document["case_order"] != expected_order
        or [item[0] for item in native] != expected_order
        or any(len(item[1]) != slots for item in native)
    ):
        fail("native provider case order or full-slot length differs")
    clear_cases = bootstrap.materialize_cases(fixture, slots)
    clear = dict(clear_cases)
    _validate_native_metadata(native, compatibility, clear)
    execution = native_document.get("execution", {})
    expected_calls = len(expected_order) * compatibility["correctness_contract"][
        "calls_per_case"
    ]
    if (
        execution.get("status") != "pass"
        or execution.get("bootstrap_3_invocation_count") != expected_calls
        or execution.get("bootstrap_3_completion_count") != expected_calls
        or execution.get("skip_count") != 0
        or execution.get("fallback_count") != 0
        or execution.get("identity_path_count") != 0
        or execution.get("nonfinite_count") != 0
        or execution.get("teardown_completed") is not True
        or execution.get("direct_bootstrap_3_callsite") is not True
    ):
        fail("native executable did not prove complete bootstrap_3 execution")
    for name in ("executable_sha256", "ordinary_library_sha256", "native_library_sha256"):
        sha(execution.get(name), f"native execution {name}")

    sanitizer = load(Path(arguments.sanitizer_record))
    if (
        sanitizer.get("schema_version") != SANITIZER_SCHEMA
        or sanitizer.get("status") != "pass"
        or sanitizer.get("exit_status") != 0
        or sanitizer.get("error_count") != 0
        or sanitizer.get("error_summary_occurrences") != 1
        or sanitizer.get("coverage")
        != {"bootstrap": True, "decrypt": True, "post_operations": True, "teardown": True}
        or sanitizer.get("sentinel")
        != compatibility["correctness_contract"]["sanitizer_sentinel"]
    ):
        fail("native Compute Sanitizer record is incomplete")
    sanitizer_bindings = sanitizer.get("bindings", {})
    expected_sanitizer_bindings = {
        "executable_sha256": digest(Path(arguments.executable)),
        "sentinel_record_sha256": digest(Path(arguments.sentinel_record)),
        "sentinel_values_sha256": digest(Path(arguments.sentinel_values)),
        "log_sha256": digest(Path(arguments.sanitizer_log)),
        "tool_version_sha256": digest(Path(arguments.sanitizer_tool_version)),
    }
    if sanitizer_bindings != expected_sanitizer_bindings:
        fail("native sanitizer artifact bindings differ")
    try:
        summary = [
            line
            for line in Path(arguments.sanitizer_log).read_text(encoding="utf-8").splitlines()
            if "ERROR SUMMARY:" in line
        ]
    except (OSError, UnicodeError) as error:
        fail(f"cannot read sanitizer log: {error}")
    if summary != ["========= ERROR SUMMARY: 0 errors"]:
        fail("native sanitizer log lacks exactly one zero-error summary")

    link_audit = load(Path(arguments.link_audit))
    if (
        link_audit.get("schema_version") != LINK_AUDIT_SCHEMA
        or link_audit.get("status") != "pass"
        or link_audit.get("native_executable", {}).get("bootstrapper_symbol_count", 0) < 1
        or link_audit.get("native_executable", {}).get("bootstrap_3_symbol_count", 0) < 1
        or link_audit.get("native_library", {}).get("bootstrap_3_symbol_count", 0) < 1
        or link_audit.get("ordinary_library", {}).get("forbidden_native_symbol_count") != 0
        or link_audit.get("closed_generated_production", {}).get("native_bootstrap_symbol_count") != 0
        or link_audit.get("closed_generated_production", {}).get("status") != "pass"
    ):
        fail("native/production link separation audit failed")

    threshold_clear = compatibility["correctness_contract"][
        "provider_clear_maximum_absolute"
    ]
    threshold_baseline = compatibility["correctness_contract"][
        "baseline_differential_maximum_absolute"
    ]
    comparisons = []
    status = "pass"
    for comparison_id, actual, reference, threshold in (
        ("native_phantom_vs_clear", native, clear_cases, threshold_clear),
        ("native_phantom_vs_closed_generated_phantom", native, generated, threshold_baseline),
    ):
        cases = []
        for (case_id, values, _), reference_record in zip(actual, reference):
            reference_id, reference_values = reference_record[0], reference_record[1]
            if case_id != reference_id:
                fail(f"{comparison_id} case order mismatch")
            result = bootstrap.metric(values, reference_values, threshold)
            if result["status"] != "pass":
                status = "fail"
            cases.append({"case_id": case_id, **result})
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "status": "pass" if all(item["status"] == "pass" for item in cases) else "fail",
                "cases": cases,
            }
        )
    output = {
        "schema_version": COMPARISON_SCHEMA,
        "status": status,
        "fixture_sha256": digest(Path(arguments.fixture)),
        "compatibility_sha256": digest(compatibility_path),
        "native_record_sha256": digest(Path(arguments.native_record)),
        "native_values_sha256": digest(Path(arguments.native_values)),
        "baseline_gpu_record_sha256": digest(Path(arguments.baseline_gpu_record)),
        "baseline_gpu_values_sha256": digest(Path(arguments.baseline_gpu_values)),
        "baseline_generated_executable_sha256": compatibility["bindings"][
            "baseline_generated_executable_sha256"
        ],
        "native_executable_sha256": digest(Path(arguments.executable)),
        "sanitizer_record_sha256": digest(Path(arguments.sanitizer_record)),
        "link_audit_sha256": digest(Path(arguments.link_audit)),
        "logical_slots": slots,
        "case_order": expected_order,
        "comparisons": comparisons,
    }
    write_fresh(Path(arguments.output), canonical_bytes(output))
    if status != "pass":
        fail("native bootstrap correctness comparison failed")


def _add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", required=True)
    for name in bootstrap.BINDING_INPUTS:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--baseline-gpu-record", required=True)
    parser.add_argument("--baseline-gpu-values", required=True)
    parser.add_argument("--baseline-comparison", required=True)
    parser.add_argument("--baseline-sanitizer", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _add_baseline_arguments(prepare_parser)
    prepare_parser.add_argument("--current-ace-source-manifest", required=True)
    prepare_parser.add_argument("--current-phantom-source-manifest", required=True)
    prepare_parser.add_argument("--clear-record-output", required=True)
    prepare_parser.add_argument("--clear-values-output", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(action=prepare)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--compatibility", required=True)
    finalize_parser.add_argument("--raw-record", required=True)
    finalize_parser.add_argument("--raw-values", required=True)
    finalize_parser.add_argument("--executable", required=True)
    finalize_parser.add_argument("--ordinary-library", required=True)
    finalize_parser.add_argument("--native-library", required=True)
    finalize_parser.add_argument("--values-output", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.set_defaults(action=finalize)

    compare_parser = commands.add_parser("compare")
    _add_baseline_arguments(compare_parser)
    compare_parser.add_argument("--compatibility", required=True)
    compare_parser.add_argument("--native-record", required=True)
    compare_parser.add_argument("--native-values", required=True)
    compare_parser.add_argument("--sentinel-record", required=True)
    compare_parser.add_argument("--sentinel-values", required=True)
    compare_parser.add_argument("--sanitizer-record", required=True)
    compare_parser.add_argument("--sanitizer-log", required=True)
    compare_parser.add_argument("--sanitizer-tool-version", required=True)
    compare_parser.add_argument("--executable", required=True)
    compare_parser.add_argument("--link-audit", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.set_defaults(action=compare)
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.action(arguments)
        return 0
    except (NativeCorrectnessError, bootstrap.CorrectnessError) as error:
        print(f"native Phantom bootstrap correctness failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
