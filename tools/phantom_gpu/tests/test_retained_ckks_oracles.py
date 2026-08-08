from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest


TOOLS = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fixture_tool = _load("retained_fixture_tool", "generate_retained_ckks_fixtures.py")
comparator = _load("retained_comparator", "compare_retained_ckks_results.py")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _context() -> dict:
    return {
        "schema_version": 1,
        "packing": "full",
        "polynomial_degree": 8,
        "logical_slot_capacity": 4,
        "data_q_bit_sizes": [5, 5, 5],
        "special_p_bit_sizes": [5],
        "input_level": 1,
        "q_part_count": 1,
        "hamming_weight": 4,
        "security_level": 0,
        "first_modulus_bits": 5,
        "scaling_modulus_bits": 5,
        "resource_schema_version": 2,
    }


def _invocation(context_path: Path, fixture_path: Path, air_path: Path) -> dict:
    argv = [
        "tools/phantom_gpu/generate_retained_ckks_sources.py",
        "--context-manifest", "inputs/compiler_context_manifest.json",
        "--fixture", "inputs/retained_ckks_fixture.json",
        "--ant-source", "outputs/retained_ckks_ant.cxx",
        "--phantom-source", "outputs/retained_ckks_phantom.cu",
        "--ant-post-ckks-air", "outputs/retained_ckks_ant_post.air",
        "--phantom-post-ckks-air", "outputs/retained_ckks_phantom_post.air",
        "--phantom-context-manifest", "outputs/compiler_context_manifest.json",
        "--phantom-resource-manifest", "outputs/compiler_resource_manifest.json",
        "--generation-record", "outputs/retained_ckks_generation.json",
        "--interface-header", "outputs/retained_ckks_generated_interface.h",
        "--polynomial-degree", "8",
        "--mul-level", "3",
        "--input-level", "1",
        "--security-level", "0",
        "--scaling-modulus-bits", "5",
        "--first-modulus-bits", "5",
        "--hamming-weight", "4",
    ]
    context_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    air_sha = hashlib.sha256(air_path.read_bytes()).hexdigest()
    return {
        "schema_version": fixture_tool.INVOCATION_SCHEMA,
        "argv": argv,
        "normalized_argv_sha256": hashlib.sha256(
            fixture_tool.canonical_bytes(argv)
        ).hexdigest(),
        "canonical_module": "ace_edsl/examples/ckks_retained_ops.py",
        "ace_commit": "1" * 40,
        "fixture_sha256": fixture_sha,
        "input_context_manifest_sha256": context_sha,
        "emitted_context_manifest_sha256": "2" * 64,
        "resource_manifest_sha256": "3" * 64,
        "generated_functions": [
            "retained_ckks_conjugate", "retained_ckks_rotate_batches",
            "retained_ckks_raise_mod", "retained_ckks_mul_monomials",
            "retained_ckks_composite",
        ],
        "function_abi": "CIPHERTEXT function(CIPHERTEXT input)",
        "linkage": "C++ (.cxx for ANT/POLY2C; .cu for Phantom/CKKS2C)",
        "invocation": "CIPHERTEXT output = retained_ckks_composite(*input_cipher);",
        "retained_runtime_calls": [
            "Conjugate_ciph", "Rotate_batch_ciph", "Raise_mod", "Mul_mono_ciph",
        ],
        "post_ckks_air_sha256": air_sha,
        "ant_post_ckks_air_sha256": air_sha,
        "phantom_post_ckks_air_sha256": air_sha,
        "ant_source_sha256": "4" * 64,
        "phantom_source_sha256": "5" * 64,
    }


def _qualification_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    template = fixture_tool.load_json(
        TOOLS / "fixtures" / "retained_ckks_v1.json"
    )
    template["qualification_bindings"] = {
        "status": "unbound",
        "required": [
            "compiler_context_manifest_sha256",
            "normalized_compiler_command_sha256",
            "post_ckks_air_sha256",
        ],
    }
    template["production_rotation_batches"] = []
    template["production_rotation_source"] = {
        "kind": "audited_default_post_ckks_air",
        "post_ckks_air_sha256": None,
    }
    template_path = tmp_path / "template.json"
    context_path = tmp_path / "context.json"
    invocation_path = tmp_path / "invocation.json"
    air_path = tmp_path / "post.air"
    _write_json(template_path, template)
    context = _context()
    _write_json(context_path, context)
    air_path.write_text(
        "CKKS.rotate_batch ATTR[nums=(2,-1,2)] RTYPE[1](cipher_batch_3)\n"
        "CKKS.rotate_batch ATTR[nums=(3,0)] RTYPE[2](cipher_batch_2)\n"
    )
    _write_json(invocation_path, _invocation(context_path, template_path, air_path))
    return template_path, context_path, invocation_path, air_path


def test_binding_derives_context_and_production_batches(tmp_path: Path) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    bound = fixture_tool.bind_fixture(template, context, invocation, air)
    assert bound["production_rotation_batches"] == [[2, -1, 2], [3, 0]]
    resolved = fixture_tool.verify_bindings(bound, context, invocation, air)
    assert resolved["polynomial_degree"] == 8
    assert resolved["logical_slots"] == 4
    assert resolved["full_data_q_count"] == 3


def test_binding_rejects_self_reported_hash_and_context_disagreement(tmp_path: Path) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    record = fixture_tool.load_json(invocation)
    record["normalized_argv_sha256"] = "0" * 64
    _write_json(invocation, record)
    with pytest.raises(fixture_tool.RetainedFixtureError, match="hash"):
        fixture_tool.bind_fixture(template, context, invocation, air)

    _write_json(invocation, _invocation(context, template, air))
    context_record = _context()
    context_record["hamming_weight"] = 5
    _write_json(context, context_record)
    with pytest.raises(
        fixture_tool.RetainedFixtureError, match="input_context_manifest_sha256"
    ):
        fixture_tool.bind_fixture(template, context, invocation, air)


def test_generation_receipt_is_root_independent_and_rejects_absolute_paths(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "freeze-root"
    second_root = tmp_path / "replay-root"
    first_root.mkdir()
    second_root.mkdir()
    first = _qualification_files(first_root)
    second = _qualification_files(second_root)
    first_record = _invocation(first[1], first[0], first[3])
    second_record = _invocation(second[1], second[0], second[3])
    assert first_record["argv"] == second_record["argv"]
    assert (
        first_record["normalized_argv_sha256"]
        == second_record["normalized_argv_sha256"]
    )

    invalid = dict(first_record)
    invalid["argv"] = list(first_record["argv"])
    invalid["argv"][2] = str(first[1].resolve())
    invalid["normalized_argv_sha256"] = hashlib.sha256(
        fixture_tool.canonical_bytes(invalid["argv"])
    ).hexdigest()
    invalid_path = tmp_path / "absolute-invocation.json"
    _write_json(invalid_path, invalid)
    with pytest.raises(fixture_tool.RetainedFixtureError, match="unsafe artifact path"):
        fixture_tool._load_invocation(invalid_path)


def test_binding_rejects_compiler_cli_parameter_manifest_disagreement(
    tmp_path: Path,
) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    record = fixture_tool.load_json(invocation)
    option_index = record["argv"].index("--polynomial-degree")
    record["argv"][option_index + 1] = "16"
    record["normalized_argv_sha256"] = hashlib.sha256(
        fixture_tool.canonical_bytes(record["argv"])
    ).hexdigest()
    _write_json(invocation, record)
    with pytest.raises(
        fixture_tool.RetainedFixtureError,
        match="--polynomial-degree disagrees",
    ):
        fixture_tool.bind_fixture(template, context, invocation, air)


def test_generator_consumes_seed_and_is_binary_deterministic(tmp_path: Path) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    bound = fixture_tool.bind_fixture(template, context, invocation, air)
    bound_path = tmp_path / "bound.json"
    _write_json(bound_path, bound)
    first_json, first_bin = tmp_path / "first.json", tmp_path / "first.bin"
    second_json, second_bin = tmp_path / "second.json", tmp_path / "second.bin"
    first = fixture_tool.generate_analytic(
        bound_path, context, invocation, air, first_json, first_bin
    )
    second = fixture_tool.generate_analytic(
        bound_path, context, invocation, air, second_json, second_bin
    )
    assert first["generator_draw_count"] == 10
    assert first["binary"]["sha256"] == second["binary"]["sha256"]
    assert first_bin.read_bytes() == second_bin.read_bytes()
    assert [record["batch_step"] for record in first["records"] if record["operation"] == "rotate_batch"] == [5, 0, -7, 5, 2, -1, 2, 3, 0]
    expected_order = comparator.expected_provider_order(bound)
    analytic_order = [record["case_id"] for record in first["records"]]
    assert analytic_order == expected_order[: len(analytic_order)]
    assert analytic_order[-1] == "raise_mod.bounded_nonperiodic"
    raise_record = first["records"][-1]
    assert raise_record["operation"] == "raise_mod"
    assert raise_record["metadata"]["active_q_count"] == 3
    production_ids = [
        case_id for case_id in analytic_order if case_id.startswith("rotate_batch.production_")
    ]
    assert production_ids == [
        "rotate_batch.production_0.output_0.step_2",
        "rotate_batch.production_0.output_1.step_-1",
        "rotate_batch.production_0.output_2.step_2",
        "rotate_batch.production_1.output_0.step_3",
        "rotate_batch.production_1.output_1.step_0",
    ]
    for name, invalid in (
        ("truncated", expected_order[:-1]),
        ("reordered", expected_order[1:2] + expected_order[:1] + expected_order[2:]),
        ("duplicated", expected_order + expected_order[-1:]),
    ):
        with pytest.raises(comparator.ComparisonError, match="case order"):
            comparator.validate_exact_case_order(invalid, expected_order, name)

    resolved = fixture_tool.validate_context_manifest(fixture_tool.load_json(context))
    comparator.load_analytic(
        first_json,
        first_bin,
        fixture_tool.sha256_path(bound_path),
        bound,
        resolved,
    )
    invalid_analytic = fixture_tool.load_json(first_json)
    invalid_analytic["records"][0]["metadata"]["unexpected"] = 1
    _write_json(first_json, invalid_analytic)
    with pytest.raises(comparator.ComparisonError, match="analytic metadata"):
        comparator.load_analytic(
            first_json,
            first_bin,
            fixture_tool.sha256_path(bound_path),
            bound,
            resolved,
        )


def test_exact_centered_lift_and_negacyclic_normalization() -> None:
    assert fixture_tool.centered_lift([0, 8, 9, 16], [17, 19]) == [
        [0, 8, 9, 16],
        [0, 8, 11, 18],
    ]
    coefficients = [1, 2, 3, 4]
    assert fixture_tool.negacyclic_monomial(coefficients, 0, 17) == coefficients
    assert fixture_tool.negacyclic_monomial(coefficients, 4, 17) == [16, 15, 14, 13]
    assert fixture_tool.negacyclic_monomial(coefficients, 9, 17) == fixture_tool.negacyclic_monomial(coefficients, 1, 17)
    inverse = fixture_tool.negacyclic_monomial(
        fixture_tool.negacyclic_monomial(coefficients, 7, 17), 1, 17
    )
    assert inverse == coefficients


def test_host_exact_generator_emits_only_provider_neutral_signed_sources(
    tmp_path: Path,
) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    bound = fixture_tool.bind_fixture(template, context, invocation, air)
    bound_path = tmp_path / "bound.json"
    _write_json(bound_path, bound)
    output_json, output_bin = tmp_path / "exact.json", tmp_path / "exact.bin"
    exact = fixture_tool.generate_exact(
        bound_path, context, invocation, air, output_json, output_bin
    )
    assert exact["schema_version"] == fixture_tool.EXACT_SCHEMA
    assert output_bin.read_bytes().startswith(fixture_tool.EXACT_SOURCE_MAGIC)
    assert exact["binary"]["format"] == "ace.retained_ckks.signed_int64le/1.0.0"
    assert exact["determinism"]["generator_draw_count"] == 16
    assert exact["layout"] == {
        "component_count": 2,
        "coefficient_count": 8,
        "ordering": "component,coefficient",
    }
    assert "ordered_data_q_moduli" not in exact
    assert "modulus_attestation_sha256" not in exact
    assert all("expected" not in record for record in exact["records"])


def test_strict_json_and_metric_reject_invalid_data(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n')
    with pytest.raises(fixture_tool.RetainedFixtureError, match="duplicate"):
        fixture_tool.load_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}\n')
    with pytest.raises(fixture_tool.RetainedFixtureError, match="non-finite"):
        fixture_tool.load_json(nonfinite)
    result = comparator.metric(
        [1 + 2j],
        [1 + 2j],
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
        hard_maximum=5e-3,
        relative_floor=1e-12,
    )
    assert result["status"] == "pass"
    assert result["comparison_count"] == 1
    assert result["maximum_absolute_error"] == 0


def _write_source_manifest(path: Path, kind: str, commit: str) -> None:
    prefix = f"{kind}-source"
    member_path = f"{prefix}/README"
    _write_json(
        path,
        {
            "schema_version": "1.0.0",
            "kind": kind,
            "source_method": "git-commit-object-archive",
            "commit": commit,
            "commit_timestamp": 1,
            "tree": "e" * 40,
            "archive": f"{prefix}.tar.gz",
            "archive_size": 1,
            "archive_sha256": "f" * 64,
            "allowed_paths": ["README"],
            "excluded_paths": [],
            "members": [
                {
                    "path": prefix,
                    "type": "directory",
                    "mode": "0755",
                    "size": 0,
                },
                {
                    "path": member_path,
                    "type": "file",
                    "mode": "0644",
                    "size": 1,
                    "sha256": "d" * 64,
                }
            ],
            "member_count": 2,
            "regular_bytes": 1,
        },
    )


def test_identity_chain_hashes_sources_generation_build_run_and_artifacts(
    tmp_path: Path,
) -> None:
    generation_fixture, context, generation_path, air = _qualification_files(tmp_path)
    bound = fixture_tool.bind_fixture(
        generation_fixture, context, generation_path, air
    )
    fixture = tmp_path / "bound-fixture.json"
    _write_json(fixture, bound)
    emitted_context = tmp_path / "emitted-context.json"
    emitted_context.write_bytes(context.read_bytes())
    resource = tmp_path / "resource.json"
    generated_ant = tmp_path / "generated.cxx"
    generated_phantom = tmp_path / "generated.cu"
    ant_executable = tmp_path / "ant-executable"
    phantom_executable = tmp_path / "phantom-executable"
    ant_json, ant_bin = tmp_path / "ant.json", tmp_path / "ant.bin"
    gpu_json, gpu_bin = tmp_path / "gpu.json", tmp_path / "gpu.bin"
    exact_json, exact_bin = tmp_path / "exact.json", tmp_path / "exact.bin"
    for path, contents in (
        (resource, b"{}\n"),
        (generated_ant, b"generated ant\n"),
        (generated_phantom, b"generated phantom\n"),
        (ant_executable, b"ant executable\n"),
        (phantom_executable, b"phantom executable\n"),
        (ant_json, b"ant json\n"),
        (ant_bin, b"ant binary\n"),
        (gpu_json, b"gpu json\n"),
        (gpu_bin, b"gpu binary\n"),
        (exact_json, b"exact json\n"),
        (exact_bin, b"exact binary\n"),
    ):
        path.write_bytes(contents)

    ace_commit, phantom_commit = "a" * 40, "b" * 40
    ace_source = tmp_path / "ace-source.json"
    phantom_source = tmp_path / "phantom-source.json"
    _write_source_manifest(ace_source, "ace", ace_commit)
    _write_source_manifest(phantom_source, "phantom", phantom_commit)

    generation = fixture_tool.load_json(generation_path)
    generation.update(
        {
            "ace_commit": ace_commit,
            "emitted_context_manifest_sha256": fixture_tool.sha256_path(
                emitted_context
            ),
            "resource_manifest_sha256": fixture_tool.sha256_path(resource),
            "ant_source_sha256": fixture_tool.sha256_path(generated_ant),
            "phantom_source_sha256": fixture_tool.sha256_path(generated_phantom),
        }
    )
    _write_json(generation_path, generation)
    build_path = tmp_path / "build.json"
    build = {
        "schema_version": comparator.BUILD_ATTESTATION_SCHEMA,
        "status": "pass",
        "architecture": "sm_80",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "source_mode": "snapshot",
        "ace_source_manifest_sha256": fixture_tool.sha256_path(ace_source),
        "phantom_source_manifest_sha256": fixture_tool.sha256_path(phantom_source),
        "compiler_context_manifest_sha256": fixture_tool.sha256_path(context),
        "compiler_resource_manifest_sha256": fixture_tool.sha256_path(resource),
        "fixture_sha256": fixture_tool.sha256_path(fixture),
        "compiler_invocation_sha256": fixture_tool.sha256_path(generation_path),
        "generated_ant_source_sha256": fixture_tool.sha256_path(generated_ant),
        "generated_phantom_source_sha256": fixture_tool.sha256_path(
            generated_phantom
        ),
        "archives": {
            "adapter": "1" * 64,
            "provider": "2" * 64,
            "common": "3" * 64,
            "ant": "4" * 64,
            "ant_encode": "5" * 64,
        },
        "executables": {
            "ant_oracle": fixture_tool.sha256_path(ant_executable),
            "phantom_sm80": fixture_tool.sha256_path(phantom_executable),
        },
        "link_commands_sha256": "6" * 64,
        "container": {
            "image": "pinned@example",
            "config_digest": "sha256:" + "7" * 64,
            "bootstrap_sha256": "8" * 64,
        },
        "link_mode": "explicit_compile-device-link-host-link",
        "archive_inspection": "pass",
        "undefined_symbol_inspection": "pass",
        "cubin_architecture_inspection": "pass",
        "host_tests": "pass",
        "host_ant_oracle_was_run": True,
        "gpu_executables_were_run": False,
    }
    _write_json(build_path, build)
    run_path = tmp_path / "run.json"
    run = {
        "schema_version": comparator.RUN_ATTESTATION_SCHEMA,
        "status": "pass",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_source_manifest_sha256": fixture_tool.sha256_path(ace_source),
        "phantom_source_manifest_sha256": fixture_tool.sha256_path(phantom_source),
        "generation_attestation_sha256": fixture_tool.sha256_path(generation_path),
        "build_attestation_sha256": fixture_tool.sha256_path(build_path),
        "fixture_sha256": fixture_tool.sha256_path(fixture),
        "compiler_context_manifest_sha256": fixture_tool.sha256_path(context),
        "compiler_resource_manifest_sha256": fixture_tool.sha256_path(resource),
        "executables": build["executables"],
        "provider_results": {
            "ant": {
                "json_sha256": fixture_tool.sha256_path(ant_json),
                "binary_sha256": fixture_tool.sha256_path(ant_bin),
            },
            "phantom": {
                "json_sha256": fixture_tool.sha256_path(gpu_json),
                "binary_sha256": fixture_tool.sha256_path(gpu_bin),
            },
        },
        "exact_observed": {
            "json_sha256": fixture_tool.sha256_path(exact_json),
            "binary_sha256": fixture_tool.sha256_path(exact_bin),
        },
        "gpu": {"device_count": 1, "device_name": "NVIDIA A100-SXM4-80GB"},
    }
    _write_json(run_path, run)
    arguments = argparse.Namespace(
        ace_source_manifest=ace_source,
        phantom_source_manifest=phantom_source,
        generation_attestation=generation_path,
        generation_fixture=generation_fixture,
        context_manifest=context,
        post_ckks_air=air,
        ant_post_ckks_air=air,
        emitted_context_manifest=emitted_context,
        resource_manifest=resource,
        fixture=fixture,
        generated_ant_source=generated_ant,
        generated_phantom_source=generated_phantom,
        ant_executable=ant_executable,
        phantom_executable=phantom_executable,
        build_attestation=build_path,
        run_attestation=run_path,
        ant_json=ant_json,
        ant_bin=ant_bin,
        gpu_json=gpu_json,
        gpu_bin=gpu_bin,
        exact_observed_json=exact_json,
        exact_observed_bin=exact_bin,
    )
    identity = comparator.load_identity_chain(arguments)
    assert identity["identifiers"]["ant"]["ace_commit"] == ace_commit
    assert identity["identifiers"]["phantom"]["phantom_commit"] == phantom_commit

    valid_ace_source = fixture_tool.load_json(ace_source)
    invalid_ace_source = dict(valid_ace_source, unexpected="open schema")
    _write_json(ace_source, invalid_ace_source)
    with pytest.raises(comparator.ComparisonError, match="source manifest keys differ"):
        comparator.load_identity_chain(arguments)
    _write_json(ace_source, valid_ace_source)

    invalid_run = dict(run, unexpected="open schema")
    _write_json(run_path, invalid_run)
    with pytest.raises(comparator.ComparisonError, match="run attestation keys differ"):
        comparator.load_identity_chain(arguments)
    _write_json(run_path, run)

    invalid_build = dict(build, phantom_commit="c" * 40)
    _write_json(build_path, invalid_build)
    with pytest.raises(comparator.ComparisonError, match="phantom_commit mismatch"):
        comparator.load_identity_chain(arguments)
    _write_json(build_path, build)
    run["build_attestation_sha256"] = fixture_tool.sha256_path(build_path)
    _write_json(run_path, run)

    generated_phantom.write_bytes(b"mutated generated phantom\n")
    with pytest.raises(comparator.ComparisonError, match="phantom_source_sha256"):
        comparator.load_identity_chain(arguments)


def test_provider_attestation_binds_identifiers_and_all_result_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = {}
    for name in (
        "compiler-invocation.json",
        "ant.json",
        "ant.bin",
        "phantom.json",
        "phantom.bin",
        "exact.json",
        "exact.bin",
    ):
        path = tmp_path / name
        path.write_bytes(f"artifact:{name}".encode())
        artifacts[name] = path
    fixture_sha = "a" * 64
    context_sha = "b" * 64
    identifiers = {
        "ant": {
            "ace_commit": "1" * 40,
            "phantom_commit": "2" * 40,
            "executable_sha256": "3" * 64,
        },
        "phantom": {
            "ace_commit": "1" * 40,
            "phantom_commit": "2" * 40,
            "executable_sha256": "4" * 64,
        },
    }
    attestation = {
        "schema_version": comparator.PROVIDER_ATTESTATION_SCHEMA,
        "fixture_sha256": fixture_sha,
        "context_manifest_sha256": context_sha,
        "compiler_invocation_sha256": fixture_tool.sha256_path(
            artifacts["compiler-invocation.json"]
        ),
        "build_attestation_sha256": "5" * 64,
        "run_attestation_sha256": "6" * 64,
        "providers": [
            {
                "provider": provider,
                "identifiers": identifiers[provider],
                "result_json_sha256": fixture_tool.sha256_path(
                    artifacts[f"{'ant' if provider == 'ant' else 'phantom'}.json"]
                ),
                "result_binary_sha256": fixture_tool.sha256_path(
                    artifacts[f"{'ant' if provider == 'ant' else 'phantom'}.bin"]
                ),
            }
            for provider in ("ant", "phantom")
        ],
        "exact_observed": {
            "provider": "phantom",
            "executable_sha256": identifiers["phantom"]["executable_sha256"],
            "result_json_sha256": fixture_tool.sha256_path(artifacts["exact.json"]),
            "result_binary_sha256": fixture_tool.sha256_path(artifacts["exact.bin"]),
        },
    }
    attestation_path = tmp_path / "provider-attestation.json"
    _write_json(attestation_path, attestation)

    observed = comparator.load_provider_attestation(
        attestation_path,
        fixture_sha256=fixture_sha,
        context_sha256=context_sha,
        generation_sha256=attestation["compiler_invocation_sha256"],
        build_sha256=attestation["build_attestation_sha256"],
        run_sha256=attestation["run_attestation_sha256"],
        expected_identifiers=identifiers,
        ant_json=artifacts["ant.json"],
        ant_binary=artifacts["ant.bin"],
        gpu_json=artifacts["phantom.json"],
        gpu_binary=artifacts["phantom.bin"],
        exact_observed_json=artifacts["exact.json"],
        exact_observed_binary=artifacts["exact.bin"],
    )
    assert observed == identifiers

    invalid = json.loads(json.dumps(attestation))
    invalid["providers"][1]["identifiers"]["executable_sha256"] = "7" * 64
    invalid_path = tmp_path / "invalid-provider-attestation.json"
    _write_json(invalid_path, invalid)
    with pytest.raises(comparator.ComparisonError, match="audited source/build identity"):
        comparator.load_provider_attestation(
            invalid_path,
            fixture_sha256=fixture_sha,
            context_sha256=context_sha,
            generation_sha256=attestation["compiler_invocation_sha256"],
            build_sha256=attestation["build_attestation_sha256"],
            run_sha256=attestation["run_attestation_sha256"],
            expected_identifiers=identifiers,
            ant_json=artifacts["ant.json"],
            ant_binary=artifacts["ant.bin"],
            gpu_json=artifacts["phantom.json"],
            gpu_binary=artifacts["phantom.bin"],
            exact_observed_json=artifacts["exact.json"],
            exact_observed_binary=artifacts["exact.bin"],
        )


def test_provider_operation_and_metadata_projection_are_case_driven() -> None:
    assert comparator.expected_provider_operation(
        "conjugate_twice.bounded_nonperiodic"
    ) == "conjugate_twice"
    assert comparator.expected_provider_operation(
        "rotate_batch.production_2.output_1.step_0"
    ) == "rotate_batch"
    assert comparator.expected_provider_operation(
        "mul_mono.2N_plus_1.bounded_nonperiodic"
    ) == "mul_mono"
    metadata = {
        "ace_level": 1,
        "active_q_count": 1,
        "scale_degree": 1,
        "logical_slots": 4,
        "ciphertext_size": 2,
        "ntt": True,
        "chain_index": 17,
        "raw_scale": 32.0,
    }
    projected = comparator.provider_independent_metadata(metadata)
    assert projected == {
        "ace_level": 1,
        "active_q_count": 1,
        "scale_degree": 1,
        "logical_slots": 4,
        "ciphertext_size": 2,
        "ntt": True,
    }
    assert "chain_index" not in projected
    assert "raw_scale" not in projected
    marker = comparator.validate_decoded_projection(
        {"kind": "strict_q0_prefix_drop", "active_q_count": 1},
        required=True,
        context="projection",
    )
    assert marker == {
        "kind": "strict_q0_prefix_drop",
        "active_q_count": 1,
    }
    assert (
        comparator.validate_decoded_projection(
            None, required=False, context="projection"
        )
        is None
    )
    with pytest.raises(comparator.ComparisonError, match="keys differ"):
        comparator.validate_decoded_projection(
            {
                "kind": "strict_q0_prefix_drop",
                "active_q_count": 1,
                "scale": 32.0,
            },
            required=True,
            context="projection",
        )

    resolved = fixture_tool.validate_context_manifest(_context())
    provider_local = dict(metadata, chain_index=9)
    assert comparator.validate_metadata(
        provider_local,
        resolved,
        raised=False,
        first_data_chain_index=7,
        context="provider-local",
    )["chain_index"] == 9
    with pytest.raises(comparator.ComparisonError, match="chain_index"):
        comparator.validate_metadata(
            dict(provider_local, chain_index=8),
            resolved,
            raised=False,
            first_data_chain_index=7,
            context="provider-local",
        )


def _write_exact_source(
    tmp_path: Path,
    fixture: dict,
    resolved: dict,
    fixture_sha: str,
    context_sha: str,
) -> tuple[Path, Path, list[list[int]]]:
    degree = resolved["polynomial_degree"]
    components, draw_count = fixture_tool.exact_signed_coefficients(
        fixture, degree
    )
    binary = bytearray(fixture_tool.EXACT_SOURCE_MAGIC)
    descriptor = fixture_tool._append_blob(
        binary,
        fixture_tool._pack_i64(
            coefficient for component in components for coefficient in component
        ),
        2 * degree,
    )
    label_by_symbol = {
        "0": "0",
        "N/2": "N_over_2",
        "N": "N",
        "3N/2": "3N_over_2",
        "2N-1": "2N_minus_1",
        "2N+1": "2N_plus_1",
    }
    records = [
        {
            "case_id": "exact_algebraic.raise_mod",
            "operation": "raise_mod",
            "normalized_power": None,
            "source_id": "signed_coefficients.default",
        }
    ]
    for symbol in fixture["monomial_powers"]:
        records.append(
            {
                "case_id": f"exact_algebraic.mul_mono.{label_by_symbol[symbol]}",
                "operation": "mul_mono",
                "normalized_power": fixture_tool.normalize_power(
                    fixture_tool.resolve_power(symbol, degree), degree
                ),
                "source_id": "signed_coefficients.default",
            }
        )
    records.append(
        {
            "case_id": "exact_algebraic.mul_mono.inverse_composition",
            "operation": "mul_mono_inverse_composition",
            "normalized_power": 0,
            "source_id": "signed_coefficients.default",
        }
    )
    binary_path = tmp_path / "exact-source.bin"
    json_path = tmp_path / "exact-source.json"
    binary_path.write_bytes(binary)
    _write_json(
        json_path,
        {
            "schema_version": fixture_tool.EXACT_SCHEMA,
            "fixture_sha256": fixture_sha,
            "qualification_bindings": fixture["qualification_bindings"],
            "context_manifest_sha256": context_sha,
            "determinism": {
                "generator": fixture["exact_source_recipe"]["generator"],
                "seed": fixture["determinism"]["seed"],
                "component_seed_xors": fixture["exact_source_recipe"][
                    "component_seed_xors"
                ],
                "coefficient_absolute_bound": fixture["exact_source_recipe"][
                    "coefficient_absolute_bound"
                ],
                "generator_draw_count": draw_count,
            },
            "conversion_convention": "provider-neutral signed test convention",
            "layout": {
                "component_count": 2,
                "coefficient_count": degree,
                "ordering": "component,coefficient",
            },
            "source_id": "signed_coefficients.default",
            "signed_coefficients": descriptor,
            "binary": {
                "format": "ace.retained_ckks.signed_int64le/1.0.0",
                "size_bytes": len(binary),
                "sha256": fixture_tool.sha256_bytes(binary),
            },
            "records": records,
        },
    )
    return json_path, binary_path, components


def test_runtime_exact_oracle_recomputes_from_observed_sources(tmp_path: Path) -> None:
    context = _context()
    resolved = fixture_tool.validate_context_manifest(context)
    fixture = fixture_tool.load_json(TOOLS / "fixtures/retained_ckks_v1.json")
    fixture["qualification_bindings"] = {
        "status": "bound",
        "compiler_context_manifest_sha256": "1" * 64,
        "normalized_compiler_command_sha256": "2" * 64,
        "post_ckks_air_sha256": "3" * 64,
    }
    moduli = [17, 19, 23]
    degree = context["polynomial_degree"]
    binary = bytearray(fixture_tool.EXACT_MAGIC)
    records = []

    def append(values):
        return fixture_tool._append_blob(
            binary, struct.pack(f"<{len(values)}Q", *values), len(values)
        )

    fixture_sha = "a" * 64
    context_sha = "b" * 64
    exact_source_json, exact_source_bin, coefficient_components = _write_exact_source(
        tmp_path, fixture, resolved, fixture_sha, context_sha
    )
    raise_source = [value % moduli[0] for component in coefficient_components for value in component]
    raise_actual = []
    for component in coefficient_components:
        raise_actual.extend(
            value
            for modulus_values in fixture_tool.centered_lift(
                [value % moduli[0] for value in component], moduli
            )
            for value in modulus_values
        )
    coefficient_metadata = {
        "active_q_count": 1,
        "ciphertext_size": 2,
        "ntt": False,
        "chain_index": 2,
        "scale_degree": 1,
        "raw_scale": 32.0,
    }
    full_metadata = {
        "active_q_count": len(moduli),
        "ciphertext_size": 2,
        "ntt": False,
        "chain_index": 0,
        "scale_degree": 1,
        "raw_scale": 32.0,
    }
    records.append(
        {
            "case_id": "exact_runtime.raise_mod",
            "operation": "raise_mod",
            "normalized_power": None,
            "source_metadata": coefficient_metadata,
            "source_metadata_after": coefficient_metadata,
            "result_metadata": full_metadata,
            "source": append(raise_source),
            "source_after": append(raise_source),
            "actual": append(raise_actual),
            "layout": {
                "component_count": 2,
                "source_modulus_count": 1,
                "result_modulus_count": len(moduli),
                "coefficient_count": degree,
                "ordering": "component,modulus,coefficient",
            },
        }
    )
    full_source = [
        value % modulus
        for component in coefficient_components
        for modulus in moduli
        for value in component
    ]
    powers = [0, degree // 2, degree, 3 * degree // 2, 2 * degree - 1, 1]
    labels = ("0", "N_over_2", "N", "3N_over_2", "2N_minus_1", "2N_plus_1")
    for label, power in zip(labels, powers):
        actual = []
        offset = 0
        for _component in coefficient_components:
            for modulus in moduli:
                actual.extend(
                    fixture_tool.negacyclic_monomial(
                        full_source[offset : offset + degree], power, modulus
                    )
                )
                offset += degree
        records.append(
            {
                "case_id": f"exact_runtime.mul_mono.{label}",
                "operation": "mul_mono",
                "normalized_power": power,
                "source_metadata": full_metadata,
                "source_metadata_after": full_metadata,
                "result_metadata": full_metadata,
                "source": append(full_source),
                "source_after": append(full_source),
                "actual": append(actual),
                "layout": {
                    "component_count": 2,
                    "source_modulus_count": len(moduli),
                    "result_modulus_count": len(moduli),
                    "coefficient_count": degree,
                    "ordering": "component,modulus,coefficient",
                },
            }
        )
    records.append(
        {
            "case_id": "exact_runtime.mul_mono.inverse_composition",
            "operation": "mul_mono_inverse_composition",
            "normalized_power": 0,
            "source_metadata": full_metadata,
            "source_metadata_after": full_metadata,
            "result_metadata": full_metadata,
            "source": append(full_source),
            "source_after": append(full_source),
            "actual": append(full_source),
            "layout": {
                "component_count": 2,
                "source_modulus_count": len(moduli),
                "result_modulus_count": len(moduli),
                "coefficient_count": degree,
                "ordering": "component,modulus,coefficient",
            },
        }
    )
    binary_path = tmp_path / "observed.bin"
    binary_path.write_bytes(binary)
    observed = {
        "schema_version": comparator.EXACT_OBSERVED_SCHEMA,
        "fixture_sha256": fixture_sha,
        "context_manifest_sha256": context_sha,
        "exact_source_json_sha256": fixture_tool.sha256_path(exact_source_json),
        "exact_source_binary_sha256": fixture_tool.sha256_path(exact_source_bin),
        "ordered_data_q_moduli": moduli,
        "first_data_chain_index": 0,
        "conversion_convention": "provider-neutral signed test convention",
        "binary": {
            "format": "ace.retained_ckks.rns_uint64le/1.0.0",
            "size_bytes": len(binary),
            "sha256": fixture_tool.sha256_bytes(binary),
        },
        "records": records,
    }
    observed_path = tmp_path / "observed.json"
    _write_json(observed_path, observed)
    evidence = comparator.compare_exact(
        exact_source_json,
        exact_source_bin,
        observed_path,
        binary_path,
        fixture,
        fixture_sha,
        context_sha,
        resolved,
        0,
    )
    assert evidence["mismatch_count"] == 0
    assert evidence["comparison_count"] > 0
    assert all(not record["mismatches"] for record in evidence["records"])

    invalid_source = json.loads(json.dumps(observed))
    invalid_binary = bytearray(binary)
    monomial = invalid_source["records"][1]
    for field in ("source", "source_after"):
        descriptor = monomial[field]
        offset = descriptor["offset_bytes"]
        original = struct.unpack_from("<Q", invalid_binary, offset)[0]
        struct.pack_into("<Q", invalid_binary, offset, (original + 1) % moduli[0])
        blob = bytes(
            invalid_binary[offset : offset + descriptor["byte_length"]]
        )
        descriptor["sha256"] = fixture_tool.sha256_bytes(blob)
    invalid_source["binary"]["sha256"] = fixture_tool.sha256_bytes(invalid_binary)
    invalid_source_binary = tmp_path / "invalid-source.bin"
    invalid_source_json = tmp_path / "invalid-source.json"
    invalid_source_binary.write_bytes(invalid_binary)
    _write_json(invalid_source_json, invalid_source)
    with pytest.raises(comparator.ComparisonError, match="exact runtime-prime reduction"):
        comparator.compare_exact(
            exact_source_json,
            exact_source_bin,
            invalid_source_json,
            invalid_source_binary,
            fixture,
            fixture_sha,
            context_sha,
            resolved,
            0,
        )

    mutations = (
        (
            "operation",
            lambda value: value["records"][1].__setitem__("operation", "raise_mod"),
            "operation",
        ),
        (
            "normalized_power",
            lambda value: value["records"][2].__setitem__(
                "normalized_power", degree // 2 + 1
            ),
            "normalized_power",
        ),
        (
            "layout_product",
            lambda value: value["records"][1]["layout"].__setitem__(
                "source_modulus_count", len(moduli) - 1
            ),
            "blob count disagrees with layout",
        ),
        (
            "chain_coordinate",
            lambda value: value["records"][1]["result_metadata"].__setitem__(
                "chain_index", 1
            ),
            "chain coordinates",
        ),
        (
            "raw_scale",
            lambda value: value["records"][1]["result_metadata"].__setitem__(
                "raw_scale", 64.0
            ),
            "changed raw scale",
        ),
    )
    for name, mutate, diagnostic in mutations:
        invalid = json.loads(json.dumps(observed))
        mutate(invalid)
        invalid_path = tmp_path / f"invalid_{name}.json"
        _write_json(invalid_path, invalid)
        with pytest.raises(comparator.ComparisonError, match=diagnostic):
            comparator.compare_exact(
                exact_source_json,
                exact_source_bin,
                invalid_path,
                binary_path,
                fixture,
                fixture_sha,
                context_sha,
                resolved,
                0,
            )


def test_fixture_contains_no_independent_context_numbers() -> None:
    fixture = fixture_tool.load_json(TOOLS / "fixtures" / "retained_ckks_v1.json")
    rules = fixture["context_rules"]
    assert all(isinstance(value, str) for value in rules.values())
    serialized = json.dumps(rules, sort_keys=True)
    for forbidden in ("16384", "8192", "56", "60", "192"):
        assert forbidden not in serialized


def test_fixture_owns_frozen_runtime_rejection_contract() -> None:
    fixture = fixture_tool.load_json(TOOLS / "fixtures" / "retained_ckks_v1.json")
    fixture_tool.validate_template(fixture, require_bound=False)
    ids = [item["id"] for item in fixture["runtime_rejections"]]
    assert ids == [
        "conjugate_missing_key",
        "rotate_batch_missing_nonzero_key",
        "rotate_batch_source_overlap",
        "raise_alias",
        "raise_size",
        "raise_chain",
        "raise_target",
        "raise_malformed_metadata",
        "mul_mono_undeclared",
    ]
    invalid = json.loads(json.dumps(fixture))
    invalid["runtime_rejections"][4]["diagnostic"] = "UNSTABLE_TOKEN"
    with pytest.raises(fixture_tool.RetainedFixtureError, match="runtime rejection"):
        fixture_tool.validate_template(invalid, require_bound=False)
