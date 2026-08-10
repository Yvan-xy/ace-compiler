from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pytest


TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ace_edsl/examples"))
import bootstrap_correctness as correctness  # noqa: E402
from bootstrap_domain_attestation import derive_supported_identity_domain  # noqa: E402
from bootstrap_domain_test_support import transform_authorities  # noqa: E402
from bootstrap_full import build_bootstrap_trace_config  # noqa: E402
from ace_edsl.edsl.core.bootstrap_decomposition import (  # noqa: E402
    build_bootstrap_evalmod_scalar_manifest,
)


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def authorities(tmp_path: Path) -> dict[str, Path]:
    config = build_bootstrap_trace_config(
        poly_degree=32,
        mul_level=2,
        first_prime_bits=41,
        scaling_factor_bits=40,
        hamming_weight=4,
        q_parts=1,
        enc_budget=1,
        dec_budget=1,
        ct_encode=False,
    )
    transform_manifest, constant_manifest, raw_air_text = (
        transform_authorities(config)
    )
    paths = {
        "ace_source_manifest": write_json(tmp_path / "ace-source.json", {"commit": "a" * 40, "tree": "b" * 40}),
        "phantom_source_manifest": write_json(tmp_path / "phantom-source.json", {"commit": "c" * 40, "tree": "d" * 40}),
        "raw_air": tmp_path / "raw.air",
        "post_ckks_air": write_json(tmp_path / "post.air", {"stage": "post-ckks"}),
        "context_manifest": write_json(tmp_path / "context.json", {"logical_slot_capacity": 16, "input_level": 1, "polynomial_degree": 32, "packing": "full", "data_q_bit_sizes": [40, 40], "special_p_bit_sizes": [41], "scaling_modulus_bits": 40, "first_modulus_bits": 41, "hamming_weight": 4, "security_level": 0, "q_part_count": 1}),
        "resource_manifest": write_json(tmp_path / "resources.json", {"schema_version": 3, "rotation_steps": [4]}),
        "constant_manifest": write_json(
            tmp_path / "constants.json", constant_manifest
        ),
        "post_operations_air": write_json(tmp_path / "post-operations.air", {"operations": ["multiply_plain", "rotate"]}),
    }
    paths["raw_air"].write_text(raw_air_text, encoding="utf-8")
    options = {
        "ciphertext_constant_encoding": "disabled",
        "decode_transform_budget": 1,
        "encode_transform_budget": 1,
        "first_prime_bits": 41,
        "hamming_weight": 4,
        "input_level": 1,
        "mul_level": 2,
        "packing": "full",
        "post_multiply_imag": 0.0,
        "post_multiply_real": -1.0,
        "post_multiply_scale_degree": 0,
        "post_rotation_step": 4,
        "poly_degree": 32,
        "q_part_count": 1,
        "scaling_factor_bits": 40,
        "security_level": 0,
        "vector_capacity": 16,
    }
    normalized_argv = [
        correctness.GENERATOR_TOOL,
        "--poly-degree", "32", "--vector-capacity", "16",
        "--mul-level", "2", "--input-level", "1",
        "--security-level", "0", "--scaling-factor-bits", "40",
        "--first-prime-bits", "41", "--hamming-weight", "4",
        "--q-part-count", "1", "--encode-transform-budget", "1",
        "--decode-transform-budget", "1",
        "--ciphertext-constant-encoding", "disabled", "--packing", "full",
        "--post-multiply-real", "-1.0", "--post-multiply-imag", "0.0",
        "--post-multiply-scale-degree", "0", "--post-rotation-step", "4",
    ]
    paths["compiler_invocation"] = write_json(
        tmp_path / "compiler-invocation.json",
        {
            "schema_version": correctness.INVOCATION_SCHEMA,
            "status": "pass",
            "tool": correctness.GENERATOR_TOOL,
            "normalized_argv": normalized_argv,
            "normalized_argv_sha256": correctness.sha256_bytes(
                correctness.canonical_identity_bytes(normalized_argv)
            ),
            "options": options,
            "output_destination_in_identity": False,
        },
    )
    semantic_bindings = {
        f"{name}_sha256": correctness.digest(paths[name])
        for name in ("compiler_invocation", "raw_air", "post_ckks_air", "context_manifest", "resource_manifest", "constant_manifest")
    }
    semantic_bindings["phantom_source_sha256"] = "e" * 64
    semantic_bindings["generated_dsl_ant_source_sha256"] = "f" * 64
    domain_bindings = {
        key: value
        for key, value in semantic_bindings.items()
        if key
        in {
            "compiler_invocation_sha256",
            "raw_air_sha256",
            "post_ckks_air_sha256",
            "context_manifest_sha256",
            "resource_manifest_sha256",
            "constant_manifest_sha256",
            "phantom_source_sha256",
            "generated_dsl_ant_source_sha256",
        }
    }
    evalmod_scalar_manifest = build_bootstrap_evalmod_scalar_manifest(config)
    supported_domain, identity_attestation = derive_supported_identity_domain(
        coefficients=config.chebyshev_coefficients,
        scalars=config.double_angle_scalars,
        overflow_bound=config.eval_sin_upper_bound_k,
        restoration_factor=config.post_scale,
        evalmod_lower=-1.0,
        evalmod_upper=1.0,
        provider_clear_threshold=1.0e-2,
        artifact_bindings=domain_bindings,
        polynomial_degree=32,
        logical_slots=16,
        transform_payload_manifest=transform_manifest,
        constant_manifest=constant_manifest,
        evalmod_scalar_manifest=evalmod_scalar_manifest,
        raw_air=raw_air_text,
    )
    preserved_transition = {
        "ace_logical_level_delta": 0, "active_q_count_delta": 0,
        "phantom_chain_index_delta": 0, "scale_degree_delta": 0,
        "raw_scale_multiplier": 1.0, "logical_slots": "preserved",
        "ciphertext_size": "preserved", "ntt_state": "preserved",
    }
    multiply_transition = dict(preserved_transition)
    operation_contract = {
        "status": "pass",
        "input_coordinate": {"ace_logical_level": 1, "scale_degree": 1},
        "rotation": {"air_attributes": {"level": 1}, "transition": preserved_transition},
        "ciphertext_plaintext_multiply": {"air_attributes": {"level": 1}, "transition": multiply_transition},
        "air_sha256": correctness.digest(paths["post_operations_air"]),
    }
    semantics = {
        "schema_version": correctness.SEMANTICS_SCHEMA,
        "status": "pass",
        "bindings": semantic_bindings,
        "supported_identity_domain": supported_domain,
        "identity_domain_attestation": identity_attestation,
        "expanded_bootstrap": {
            "evalmod_scalar_encodings": evalmod_scalar_manifest,
        },
        "output_air_contract": {"ace_logical_level": 1, "active_q_count": 1, "phantom_chain_index": 1, "raw_scale_contract": {"kind": "ace-log2-scale-coordinate", "nominal_raw_scale": "0x1.0000000000000p+40", "scaling_modulus_bits": 40, "expected_scale_degree": 1, "maximum_absolute_coordinate_error": 1.0e-4}, "scale_degree": 1, "logical_slots": 16, "ciphertext_size": 2, "ntt_state": True},
        "post_operation_contracts": operation_contract,
    }
    paths["bootstrap_semantics"] = write_json(tmp_path / "bootstrap-semantics.json", semantics)
    operation_bindings = {
        f"{name}_sha256": correctness.digest(paths[name])
        for name in ("compiler_invocation", "post_ckks_air", "context_manifest", "resource_manifest", "post_operations_air")
    }
    attestation = {
            "schema_version": correctness.POST_OPERATION_SCHEMA,
            "status": "attested",
            "bindings": operation_bindings,
            "inputs": {"multiply_constant": {"real": -1.0, "imaginary": 0.0, "plaintext_scale_degree": 0}, "rotation_step": 4},
            "input_coordinate": operation_contract["input_coordinate"],
            "rotation": operation_contract["rotation"],
            "ciphertext_plaintext_multiply": operation_contract["ciphertext_plaintext_multiply"],
        }
    paths["post_operation_attestation"] = write_json(
        tmp_path / "post-operation-attestation.json",
        attestation,
    )
    semantics["bindings"]["post_operations_air_sha256"] = correctness.digest(paths["post_operations_air"])
    semantics["bindings"]["post_operations_attestation_sha256"] = correctness.digest(paths["post_operation_attestation"])
    write_json(paths["bootstrap_semantics"], semantics)
    return paths


def freeze(tmp_path: Path) -> tuple[dict[str, Path], Path, dict]:
    paths = authorities(tmp_path)
    output = tmp_path / "fixture.json"
    args = argparse.Namespace(
        **{name: str(path) for name, path in paths.items()},
        fixture_id="generated-bootstrap-correctness-v1",
        seed=1234567,
        inside_margin=0.125,
        provider_clear_threshold=1e-2,
        gpu_native_threshold=2e-2,
        gpu_generated_threshold=2e-2,
        repeat_threshold=1e-6,
        output=str(output),
    )
    fixture = correctness.freeze_fixture(args)
    return paths, output, fixture


def provider_record(
    tmp_path: Path,
    provider: str,
    fixture_path: Path,
    fixture: dict,
    paths: dict[str, Path],
    values: list[tuple[str, list[complex]]],
) -> tuple[Path, Path]:
    binary_path = tmp_path / f"{provider}-values.bin"
    if provider == "generated-phantom":
        semantics = json.loads(paths["bootstrap_semantics"].read_text())
        bootstrap = correctness._bootstrap_metadata(semantics)
        multiply = correctness._transition(bootstrap, fixture["post_operations"]["ciphertext_plaintext_multiply"]["transition"])
        rotation = correctness._transition(bootstrap, fixture["post_operations"]["rotation"]["transition"])
        gpu_metadata = {
            "bootstrap": bootstrap,
            "repeatability": {"calls": 3, "independently_owned_clones": True, "maximum_absolute_difference": 0.0, "threshold": fixture["tolerances"]["repeat_maximum_absolute"], "call_metrics": [], "call_metadata": [bootstrap, bootstrap, bootstrap]},
            "post_operations": {
                "ciphertext_plaintext_multiply": {"metric": {}, "metadata": multiply, "values_sha256": "a" * 64},
                "rotation": {"metric": {}, "metadata": rotation, "values_sha256": "b" * 64, "step": fixture["post_operations"]["inputs"]["rotation_step"]},
            },
            "ownership": {"input_clones_released": 3, "zero_argument_clones_released": 3, "bootstrap_results_released": 3, "post_operation_ciphertexts_released": 4, "post_operation_plaintexts_released": 1, "base_encrypted_arguments_released": 2, "post_operation_sources_independent": True, "all_owned_objects_released": True},
        }
    records = []
    recipes = {item["id"]: item["kind"] for item in fixture["case_recipes"]}
    for case_id, case_values in values:
        recorded_metric = correctness.metric(case_values, case_values, fixture["tolerances"]["provider_clear_maximum_absolute"])
        if provider == "generated-phantom":
            metadata = json.loads(json.dumps(gpu_metadata))
            metadata["metrics_vs_clear"] = recorded_metric
            metadata["repeatability"]["call_metrics"] = [
                recorded_metric,
                recorded_metric,
                recorded_metric,
            ]
            metadata["post_operations"]["ciphertext_plaintext_multiply"]["metric"] = recorded_metric
            metadata["post_operations"]["rotation"]["metric"] = recorded_metric
        else:
            metadata = {"recipe": recipes[case_id], "metrics_vs_clear": recorded_metric}
            if provider == "native-ant":
                metadata["execution_attestation"] = {
                    "invocation_count": 1, "early_identity_copy_count": 0,
                    "full_execution_count": 1, "coeffs_to_slots_entry_count": 1,
                    "coeffs_to_slots_completion_count": 1, "eval_mod_entry_count": 1,
                    "eval_mod_completion_count": 1, "slots_to_coeffs_entry_count": 1,
                    "slots_to_coeffs_completion_count": 1, "full_completion_count": 1,
                }
        records.append({"case_id": case_id, "oracle": provider, "values": case_values, "metadata": metadata})
    binary, descriptors = correctness.write_value_file(binary_path, correctness.digest(fixture_path), correctness.digest(paths["context_manifest"]), records)
    execution = {"status": "pass", "skip_count": 0, "timeout_count": 0, "fallback_count": 0, "nonfinite_count": 0}
    context = json.loads(paths["context_manifest"].read_text())
    invocation = json.loads(paths["compiler_invocation"].read_text())
    if provider != "generated-phantom":
        execution["context_attestation"] = {
            "provider": provider, "polynomial_degree": context["polynomial_degree"],
            "logical_slots": context["logical_slot_capacity"], "data_q_count": len(context["data_q_bit_sizes"]),
            "special_p_count": len(context["special_p_bit_sizes"]), "data_q_requested_bit_sizes": context["data_q_bit_sizes"],
            "special_p_requested_bit_sizes": context["special_p_bit_sizes"], "input_level": context["input_level"],
            "scaling_factor_bits": context["scaling_modulus_bits"], "first_prime_bits": context["first_modulus_bits"],
            "hamming_weight": context["hamming_weight"], "security_level": context["security_level"],
            "q_part_count": context["q_part_count"], "transform_budgets": {"encode": invocation["options"]["encode_transform_budget"], "decode": invocation["options"]["decode_transform_budget"]},
            "phantom_context_manifest_sha256": fixture["bindings"]["context_manifest_sha256"],
            "physical_prime_identity_authority": "independent-non-authoritative-for-gpu", "schedule_authority": "independent-non-authoritative-for-gpu",
        }
        execution["provenance"] = {
            "fixture_sha256": correctness.digest(fixture_path), "compiler_invocation_sha256": fixture["bindings"]["compiler_invocation_sha256"],
            "compiler_context_manifest_sha256": fixture["bindings"]["context_manifest_sha256"], "compiler_resource_manifest_sha256": fixture["bindings"]["resource_manifest_sha256"],
            "compiler_constant_manifest_sha256": fixture["bindings"]["constant_manifest_sha256"], "bootstrap_semantics_sha256": fixture["bindings"]["bootstrap_semantics_sha256"],
            "qualification_bindings": fixture["bindings"], "executable_sha256": "c" * 64,
        }
    if provider == "native-ant":
        execution.update(native_bootstrap_invocation_count=7, early_identity_copy_count=0)
    elif provider == "generated-ant":
        execution.update(generated_bootstrap_invocation_count=7, linked_post_ckks_air_sha256=fixture["bindings"]["post_ckks_air_sha256"], linked_generated_source_sha256="f" * 64)
    else:
        execution.update(generated_bootstrap_invocation_count=21, executable_sha256="e" * 64, teardown_completed=True)
    record_path = tmp_path / f"{provider}.json"
    write_json(
        record_path,
        {
            "schema_version": correctness.PROVIDER_SCHEMAS[provider],
            "provider": provider,
            "fixture_sha256": correctness.digest(fixture_path),
            "bindings": fixture["bindings"],
            "case_order": [case_id for case_id, _ in values],
            "logical_slots": len(values[0][1]),
            "binary": binary,
            "records": descriptors,
            "execution": execution,
        },
    )
    return record_path, binary_path


def test_freeze_is_provider_neutral_and_context_length_is_runtime_derived(tmp_path: Path) -> None:
    paths, fixture_path, fixture = freeze(tmp_path)
    assert [case["kind"] for case in fixture["case_recipes"]] == list(correctness.CASE_KINDS)
    serialized = fixture_path.read_text(encoding="utf-8")
    for forbidden in ("logical_slot_capacity", "input_level", "poly_degree", "q_chain", "scaling_factor_bits"):
        assert forbidden not in serialized
    assert fixture["supported_identity_domain"]["attested_domain"] == json.loads(paths["bootstrap_semantics"].read_text())["supported_identity_domain"]
    assert all(len(values) == 8 for _, values in correctness.materialize_cases(fixture, 8))
    assert all(len(values) == 16 for _, values in correctness.materialize_cases(fixture, 16))


def test_invocation_identity_matches_generator_compact_json(tmp_path: Path) -> None:
    paths = authorities(tmp_path)
    invocation = json.loads(paths["compiler_invocation"].read_text(encoding="utf-8"))
    assert invocation["normalized_argv_sha256"] == hashlib.sha256(
        json.dumps(
            invocation["normalized_argv"],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    invocation["normalized_argv_sha256"] = correctness.sha256_bytes(
        correctness.canonical_bytes(invocation["normalized_argv"])
    )
    write_json(paths["compiler_invocation"], invocation)
    with pytest.raises(correctness.CorrectnessError, match="normalized argv hash"):
        correctness.validate_authorities(paths)


def test_raw_scale_contract_accepts_exact_prime_drift_only() -> None:
    contract = {
        "scaling_modulus_bits": 40,
        "expected_scale_degree": 1,
        "maximum_absolute_coordinate_error": 1.0e-4,
    }
    observed = (2.0**40) * (1.0 - 1.0e-6)
    assert correctness._validate_raw_scale(observed, contract, "test") == observed
    with pytest.raises(
        correctness.CorrectnessError, match="differs from the AIR scale coordinate"
    ):
        correctness._validate_raw_scale(2.0**38, contract, "test")


def test_splitmix_fixture_stream_has_frozen_cross_language_words() -> None:
    state = 7640891576956012809 ^ 0x42534352414E444F
    observed = []
    for _ in range(4):
        state, value = correctness.splitmix64(state)
        observed.append(value)
    assert observed == [
        0x6C5F24FC11A466E8,
        0x559CFF89394149CA,
        0x79DAC66A159692A3,
        0x74F6283DB86357A2,
    ]


def test_freeze_refuses_existing_output_and_changed_attestation(tmp_path: Path) -> None:
    paths, fixture_path, _ = freeze(tmp_path)
    args = argparse.Namespace(
        **{name: str(path) for name, path in paths.items()}, fixture_id="another", seed=1,
        inside_margin=0.125, provider_clear_threshold=1e-2, gpu_native_threshold=1e-2,
        gpu_generated_threshold=1e-2, repeat_threshold=1e-6, output=str(fixture_path),
    )
    with pytest.raises(correctness.CorrectnessError, match="replace existing"):
        correctness.freeze_fixture(args)
    changed = json.loads(paths["post_operation_attestation"].read_text())
    changed["bindings"]["post_operations_air_sha256"] = "0" * 64
    write_json(paths["post_operation_attestation"], changed)
    args.output = str(tmp_path / "new-fixture.json")
    with pytest.raises(correctness.CorrectnessError, match="post_operations_air_sha256"):
        correctness.freeze_fixture(args)


def test_three_way_comparison_requires_full_provenance_and_external_sanitizer(tmp_path: Path) -> None:
    paths, fixture_path, fixture = freeze(tmp_path)
    clear = correctness.materialize_cases(fixture, 16)
    executable = tmp_path / "runner"
    executable.write_bytes(b"runner")
    native = provider_record(tmp_path, "native-ant", fixture_path, fixture, paths, clear)
    generated = provider_record(tmp_path, "generated-ant", fixture_path, fixture, paths, clear)
    host_replay = correctness.verify_host(argparse.Namespace(
        **{name: str(path) for name, path in paths.items()}, fixture=str(fixture_path),
        native_record=str(native[0]), native_values=str(native[1]), generated_record=str(generated[0]),
        generated_values=str(generated[1]), output=str(tmp_path / "host-replay.json"),
    ))
    assert host_replay["status"] == "pass"
    gpu = provider_record(tmp_path, "generated-phantom", fixture_path, fixture, paths, clear)
    gpu_document = json.loads(gpu[0].read_text())
    gpu_document["execution"]["executable_sha256"] = correctness.digest(executable)
    write_json(gpu[0], gpu_document)
    log = tmp_path / "sanitizer.log"
    exact_summary = "========= ERROR SUMMARY: 0 errors\n"
    log.write_text(exact_summary, encoding="utf-8")
    tool = tmp_path / "sanitizer-version.txt"
    tool.write_text("compute-sanitizer test\n", encoding="utf-8")
    sanitizer = tmp_path / "sanitizer.json"
    write_json(sanitizer, {
        "schema_version": correctness.SANITIZER_SCHEMA, "status": "pass", "exit_status": 0,
        "error_summary_occurrences": 1, "error_count": 0,
        "bindings": {"gpu_executable_sha256": correctness.digest(executable), "gpu_record_sha256": correctness.digest(gpu[0]), "gpu_values_sha256": correctness.digest(gpu[1]), "log_sha256": correctness.digest(log), "tool_version_sha256": correctness.digest(tool)},
        "coverage": {"bootstrap": True, "post_operations": True, "repeatability": True, "teardown": True},
    })
    output = tmp_path / "comparison.json"
    arguments = argparse.Namespace(
        **{name: str(path) for name, path in paths.items()}, fixture=str(fixture_path),
        native_record=str(native[0]), native_values=str(native[1]), generated_record=str(generated[0]),
        generated_values=str(generated[1]), gpu_record=str(gpu[0]), gpu_values=str(gpu[1]),
        sanitizer_record=str(sanitizer), sanitizer_log=str(log), sanitizer_tool_version=str(tool),
        gpu_executable=str(executable), output=str(output),
    )
    result = correctness.compare(arguments)
    assert result["status"] == "pass"
    assert len(result["comparisons"]) == 5

    for index, invalid_log in enumerate(
        (
            "========= ERROR SUMMARY: 00 errors\n",
            exact_summary + exact_summary,
        )
    ):
        log.write_text(invalid_log, encoding="utf-8")
        invalid_sanitizer = json.loads(sanitizer.read_text())
        invalid_sanitizer["bindings"]["log_sha256"] = correctness.digest(log)
        invalid_sanitizer_path = tmp_path / f"invalid-log-sanitizer-{index}.json"
        write_json(invalid_sanitizer_path, invalid_sanitizer)
        arguments.sanitizer_record = str(invalid_sanitizer_path)
        arguments.output = str(tmp_path / f"invalid-log-comparison-{index}.json")
        with pytest.raises(
            correctness.CorrectnessError,
            match="exactly one zero-error summary",
        ):
            correctness.compare(arguments)

    log.write_text(exact_summary, encoding="utf-8")
    restored = json.loads(sanitizer.read_text())
    restored["bindings"]["log_sha256"] = correctness.digest(log)
    write_json(sanitizer, restored)

    broken = json.loads(sanitizer.read_text())
    broken["coverage"]["teardown"] = False
    broken_path = tmp_path / "broken-sanitizer.json"
    write_json(broken_path, broken)
    arguments.sanitizer_record = str(broken_path)
    arguments.output = str(tmp_path / "broken-comparison.json")
    with pytest.raises(correctness.CorrectnessError, match="coverage"):
        correctness.compare(arguments)


def test_binary_checksum_and_case_order_are_strict(tmp_path: Path) -> None:
    paths, fixture_path, fixture = freeze(tmp_path)
    clear = correctness.materialize_cases(fixture, 16)
    record, binary = provider_record(tmp_path, "native-ant", fixture_path, fixture, paths, clear)
    data = bytearray(binary.read_bytes())
    data[-1] ^= 1
    binary.write_bytes(data)
    with pytest.raises(correctness.CorrectnessError, match="checksum"):
        correctness.load_provider(record, binary, "native-ant", fixture, correctness.digest(fixture_path), fixture["bindings"], 16, semantics=json.loads(paths["bootstrap_semantics"].read_text()), invocation=json.loads(paths["compiler_invocation"].read_text()), context_manifest=json.loads(paths["context_manifest"].read_text()))


def test_generated_phantom_harness_covers_full_correctness_lifecycle() -> None:
    source = (TOOLS_ROOT / "harness/generated_bootstrap_phantom_correctness.cu").read_text(encoding="utf-8")
    compact = "".join(source.split())
    assert "extern CIPHERTEXT bootstrap_full(CIPHERTEXT, CIPHERTEXT);" in source
    assert "CIPHERTEXTobserved=bootstrap_full(owned_input,owned_zero);Register_ciph_lifetime(&observed);" in compact
    assert "for(intcall=0;call<3;++call)" in compact
    assert "CIPHERTEXTowned_input=encrypted;Register_ciph_lifetime(&owned_input);" in compact
    assert "CIPHERTEXTowned_zero=encrypted_zero;Register_ciph_lifetime(&owned_zero);" in compact
    assert "if(call==0)Copy_ciph(&primary,&observed);" in compact
    assert "PLAINplain=newPLAINTEXT();Register_plain_lifetime(plain);" in compact
    assert "input.context.at(\"logical_slot_capacity\")" in source
    assert "input.context.at(\"input_level\")" in source
    assert "ace.phantom.bootstrap-correctness-fixture/2.0.0" in source
    assert "ace.phantom.generated-bootstrap.semantics/2.0.0" in source
    assert "ace.phantom.bootstrap-clear-evalmod-domain/1.0.0" in source
    assert "bootstrap metadata differs from AIR contract: observed=" in source
    assert 'metadata.dump() + " expected=" + observed_contract.dump()' in source
    assert "canonical_identity_attestation" in source
    assert "identity_attestation_sha256" in source
    assert (
        'domain_evidence.at("attestation_sha256") ==\n'
        "                  identity_attestation_sha256"
    ) in source
    assert "conservative-certified-binary64-marker-used-as-exclusive-endpoint" in source
    assert 'domain_errors.at("target")' in source
    assert 'domain_proof.at("selected_radius")' in source
    for field in (
        "comparison=",
        "maximum_absolute_error=",
        "maximum_absolute_error_index=",
        "actual_real=",
        "actual_imaginary=",
        "expected_real=",
        "expected_imaginary=",
        "mean_absolute_error=",
        "root_mean_square_error=",
        "threshold=",
    ):
        assert field in source
    assert "std::numeric_limits<double>::max_digits10" in source
    for comparison in (
        "/bootstrap-call-",
        "/ciphertext-plaintext-multiply",
        "/rotation",
        "/primary-output",
    ):
        assert comparison in source
    random_real = source.index("const double real = generator.Symmetric(bound)")
    random_imaginary = source.index(
        "const double imaginary = generator.Symmetric(bound)"
    )
    random_complex = source.index("value = {real, imaginary}")
    assert random_real < random_imaginary < random_complex
    for query in ("Level(value)", "Active_q_count(value)", "Chain_index(value)", "Raw_scale(value)", "Sc_degree(value)", "Get_ciph_slots(value)", "Get_ciph_size(value)", "Is_ciph_ntt(value)"):
        assert query in source
    assert "Mul_plain(&multiply_result" in source
    assert 'at("plaintext_scale_degree")' in source
    assert "Rotate_ciph(&rotate_result" in source
    assert "repeat_maximum<=input.repeat_threshold" in compact
    assert "std::log2(observed_raw_scale)" in source
    assert "bootstrapmetadatadiffersacrosscallsorcases" in compact
    assert '"executable_sha256"' in source
    assert "kMagic={'A','C','E','B','S','C','0','1'}" in compact
    assert "sanitizer_invalid_access_count" not in source
    for forbidden in ("16384", "8192", "hamming_weight", "scaling_factor_bits"):
        assert forbidden not in source
