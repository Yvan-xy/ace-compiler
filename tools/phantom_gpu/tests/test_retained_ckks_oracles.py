from __future__ import annotations

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


def _invocation(context: dict) -> dict:
    argv = [
        "tools/phantom_gpu/generate_ckks2c_probe.py",
        "--output", "ckks2c/add_mul_rotate.cu",
        "--post-ckks-air", "ckks2c/ordinary_ckks_post_ckks.air",
        "--context-manifest", "ckks2c/compiler_context_manifest.json",
        "--resource-manifest", "ckks2c/compiler_resource_manifest.json",
        "--poly-degree", str(context["polynomial_degree"]),
        "--mul-level", str(len(context["data_q_bit_sizes"])),
        "--input-level", str(context["input_level"]),
        "--security-level", str(context["security_level"]),
        "--scaling-factor-bits", str(context["scaling_modulus_bits"]),
        "--first-prime-bits", str(context["first_modulus_bits"]),
        "--hamming-weight", str(context["hamming_weight"]),
    ]
    return {
        "schema_version": fixture_tool.INVOCATION_SCHEMA,
        "argv": argv,
        "normalized_argv_sha256": hashlib.sha256(
            fixture_tool.canonical_bytes(argv)
        ).hexdigest(),
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
    _write_json(invocation_path, _invocation(context))
    air_path.write_text(
        "CKKS.rotate_batch ATTR[nums=(2,-1,2)] RTYPE[1](cipher_batch_3)\n"
        "CKKS.rotate_batch ATTR[nums=(3,0)] RTYPE[2](cipher_batch_2)\n"
    )
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

    _write_json(invocation, _invocation(_context()))
    context_record = _context()
    context_record["hamming_weight"] = 5
    _write_json(context, context_record)
    with pytest.raises(fixture_tool.RetainedFixtureError, match="hamming-weight"):
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


def test_exact_generator_uses_explicit_algebraic_source_kind(tmp_path: Path) -> None:
    template, context, invocation, air = _qualification_files(tmp_path)
    bound = fixture_tool.bind_fixture(template, context, invocation, air)
    bound_path = tmp_path / "bound.json"
    _write_json(bound_path, bound)
    moduli = tmp_path / "moduli.json"
    _write_json(moduli, {"ordered_data_q_moduli": [17, 19, 23]})
    output_json, output_bin = tmp_path / "exact.json", tmp_path / "exact.bin"
    result = fixture_tool.generate_exact(
        bound_path, context, invocation, air, moduli, output_json, output_bin
    )
    assert result["source_contract"].startswith("The harness must import")
    assert all(record["case_id"].startswith("exact_algebraic.") for record in result["records"])
    assert all(record["source_kind"] == "deterministic_harness_import" for record in result["records"])


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


def test_runtime_exact_oracle_recomputes_from_observed_sources(tmp_path: Path) -> None:
    context = _context()
    resolved = fixture_tool.validate_context_manifest(context)
    moduli = [17, 19, 23]
    degree = context["polynomial_degree"]
    binary = bytearray(fixture_tool.EXACT_MAGIC)
    records = []

    def append(values):
        return fixture_tool._append_blob(
            binary, struct.pack(f"<{len(values)}Q", *values), len(values)
        )

    coefficient_components = [list(range(degree)), list(range(degree, 2 * degree))]
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
    }
    full_metadata = {
        "active_q_count": len(moduli),
        "ciphertext_size": 2,
        "ntt": False,
        "chain_index": 0,
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
    fixture_sha = "a" * 64
    context_sha = "b" * 64
    binary_path = tmp_path / "observed.bin"
    binary_path.write_bytes(binary)
    observed = {
        "schema_version": comparator.EXACT_OBSERVED_SCHEMA,
        "fixture_sha256": fixture_sha,
        "context_manifest_sha256": context_sha,
        "ordered_data_q_moduli": moduli,
        "first_data_chain_index": 0,
        "conversion_convention": "coefficient residues in component,modulus,coefficient order",
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
        observed_path, binary_path, fixture_sha, context_sha, resolved
    )
    assert evidence["mismatch_count"] == 0
    assert evidence["comparison_count"] > 0
    assert all(not record["mismatches"] for record in evidence["records"])


def test_fixture_contains_no_independent_context_numbers() -> None:
    fixture = fixture_tool.load_json(TOOLS / "fixtures" / "retained_ckks_v1.json")
    rules = fixture["context_rules"]
    assert all(isinstance(value, str) for value in rules.values())
    serialized = json.dumps(rules, sort_keys=True)
    for forbidden in ("16384", "8192", "56", "60", "192"):
        assert forbidden not in serialized
