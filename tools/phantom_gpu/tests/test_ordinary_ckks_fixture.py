from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"
FIXTURE_PATH = TOOLS_ROOT / "fixtures/ordinary_ckks_v1.json"
sys.path.insert(0, str(TOOLS_ROOT))

import ordinary_ckks_fixture as ordinary  # noqa: E402


CONTEXT_MANIFEST_TEXT = (
    '{"data_q_bit_sizes":[60,56,56,56],"first_modulus_bits":60,'
    '"hamming_weight":192,"input_level":1,"logical_slot_capacity":8192,'
    '"packing":"full","polynomial_degree":16384,"q_part_count":2,'
    '"resource_schema_version":1,"scaling_modulus_bits":56,'
    '"schema_version":1,"security_level":0,'
    '"special_p_bit_sizes":[60,60]}'
)


def write_context_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "compiler_context_manifest.json"
    path.write_text(CONTEXT_MANIFEST_TEXT, encoding="utf-8")
    return path


def generate_analytic(tmp_path: Path, stem: str = "reference") -> tuple[Path, Path, dict]:
    json_path = tmp_path / f"{stem}.json"
    binary_path = tmp_path / f"{stem}.bin"
    context_manifest = write_context_manifest(tmp_path)
    result = ordinary.generate_analytic_reference(
        FIXTURE_PATH, context_manifest, json_path, binary_path
    )
    return json_path, binary_path, result


def test_frozen_fixture_derives_coordinates_and_covers_aliases_and_tolerances(
    tmp_path: Path,
) -> None:
    fixture, digest = ordinary.load_fixture(
        FIXTURE_PATH, write_context_manifest(tmp_path)
    )
    assert len(digest) == 64
    assert fixture["coordinate_rules"] == {
        "input_length": "compiler_context.logical_slot_capacity",
        "first_level": "compiler_context.data_q_count",
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
    }
    assert fixture["_resolved_context"] == {
        "polynomial_degree": 16384,
        "logical_slots": 8192,
        "data_q_count": 4,
        "scaling_modulus_bits": 56,
    }
    assert fixture["tolerances"]["absolute"] == 1.0e-4
    assert fixture["tolerances"]["relative"] == 1.0e-3
    assert fixture["tolerances"]["hard_maximum_absolute"] == 5.0e-3
    assert {
        contract["active_q_count"]
        for contract in fixture["metadata_contracts"].values()
    } >= {"full", "middle", "bottom", 4}
    subtraction = next(
        family for family in fixture["case_families"] if family["id"] == "sub_ct_ct"
    )
    assert subtraction["aliases"] == ["distinct", "lhs", "rhs"]
    assert fixture["alias_matrix"]["ct_ct"]["rhs"] == "supported"
    assert len(ordinary.expanded_cases(fixture)) == 43


def test_analytic_generation_is_deterministic_and_never_claims_ant(
    tmp_path: Path,
) -> None:
    first_json, first_binary, first = generate_analytic(tmp_path, "first")
    second_json, second_binary, second = generate_analytic(tmp_path, "second")
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_binary.read_bytes() == second_binary.read_bytes()
    assert first == second
    assert first["analytic"]["status"] == "complete"
    assert first["ant"] == {
        "status": "missing",
        "required_for_qualification": True,
        "producer": None,
        "independent_encoding": False,
        "input_kind": None,
        "records": [],
    }

    fixture, fixture_digest = ordinary.load_fixture(
        FIXTURE_PATH, write_context_manifest(tmp_path)
    )
    with pytest.raises(ordinary.OrdinaryCkksError, match="ANT records are missing"):
        ordinary._load_reference(
            first_json,
            first_binary,
            fixture,
            fixture_digest,
            require_ant=True,
        )


def test_duplicate_json_keys_and_unknown_fixture_fields_are_rejected(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"one","schema_version":"two"}\n')
    with pytest.raises(ordinary.OrdinaryCkksError, match="duplicate JSON key"):
        ordinary.load_json(duplicate)

    fixture = ordinary.load_json(FIXTURE_PATH)
    fixture["unexpected"] = True
    with pytest.raises(ordinary.OrdinaryCkksError, match="unknown keys"):
        ordinary.validate_fixture(fixture)


def test_context_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    altered_manifest = write_context_manifest(tmp_path)
    altered_manifest.write_bytes(altered_manifest.read_bytes() + b"\n")
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="compiler context manifest hash mismatch",
    ):
        ordinary.load_fixture(FIXTURE_PATH, altered_manifest)


def test_binary_overlap_gap_length_hash_and_nonfinite_are_rejected(
    tmp_path: Path,
) -> None:
    _, binary_path, reference = generate_analytic(tmp_path)
    fixture, fixture_digest = ordinary.load_fixture(
        FIXTURE_PATH, write_context_manifest(tmp_path)
    )
    descriptors = reference["analytic"]["records"]

    overlap = copy.deepcopy(descriptors)
    overlap[1]["offset_bytes"] = 0
    with pytest.raises(ordinary.OrdinaryCkksError, match="payload overlap"):
        ordinary.read_value_file(
            binary_path,
            reference["binary"],
            overlap,
            fixture_digest,
            fixture["compiler_context_manifest"]["sha256"],
        )

    gap = copy.deepcopy(descriptors)
    gap[1]["offset_bytes"] += ordinary.PAIR.size
    with pytest.raises(ordinary.OrdinaryCkksError, match="payload gap"):
        ordinary.read_value_file(
            binary_path,
            reference["binary"],
            gap,
            fixture_digest,
            fixture["compiler_context_manifest"]["sha256"],
        )

    truncated = tmp_path / "truncated.bin"
    truncated_contents = binary_path.read_bytes()[:-1]
    truncated.write_bytes(truncated_contents)
    truncated_info = copy.deepcopy(reference["binary"])
    truncated_info["size_bytes"] = len(truncated_contents)
    truncated_info["sha256"] = hashlib.sha256(truncated_contents).hexdigest()
    with pytest.raises(ordinary.OrdinaryCkksError, match="payload length mismatch"):
        ordinary.read_value_file(
            truncated,
            truncated_info,
            descriptors,
            fixture_digest,
            fixture["compiler_context_manifest"]["sha256"],
        )

    nonfinite = tmp_path / "nonfinite.bin"
    nonfinite_contents = bytearray(binary_path.read_bytes())
    struct.pack_into("<d", nonfinite_contents, ordinary.HEADER.size, float("nan"))
    nonfinite.write_bytes(nonfinite_contents)
    nonfinite_info = copy.deepcopy(reference["binary"])
    nonfinite_info["sha256"] = hashlib.sha256(nonfinite_contents).hexdigest()
    with pytest.raises(ordinary.OrdinaryCkksError, match="non-finite value"):
        ordinary.read_value_file(
            nonfinite,
            nonfinite_info,
            descriptors,
            fixture_digest,
            fixture["compiler_context_manifest"]["sha256"],
        )

    corrupted_info = copy.deepcopy(reference["binary"])
    corrupted_info["sha256"] = "0" * 64
    with pytest.raises(ordinary.OrdinaryCkksError, match="size or SHA-256 mismatch"):
        ordinary.read_value_file(
            binary_path,
            corrupted_info,
            descriptors,
            fixture_digest,
            fixture["compiler_context_manifest"]["sha256"],
        )


def test_metrics_report_separate_relative_and_precision_fields() -> None:
    tolerances = {
        "absolute": 1.0e-4,
        "relative": 1.0e-3,
        "hard_maximum_absolute": 5.0e-3,
        "relative_metric_floor": 1.0e-12,
    }
    passing = ordinary.comparison_metrics(
        [1.0005 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j], tolerances
    )
    assert passing["pass"] is True
    assert passing["maximum_absolute_error_index"] == 0
    assert passing["maximum_relative_error_index"] == 0
    assert passing["relative_metric_count"] == 1
    assert passing["relative_metric_excluded_count"] == 1
    assert passing["mae"] > 0.0
    assert passing["rmse"] > 0.0
    assert passing["estimated_precision_bits"] is not None

    failing = ordinary.comparison_metrics(
        [1.006 + 0.0j], [1.0 + 0.0j], tolerances
    )
    assert failing["pass"] is False
    assert failing["mixed_tolerance_failure_count"] == 1
    assert failing["maximum_absolute_error"] > tolerances["hard_maximum_absolute"]

    separate = ordinary.compare_oracles(
        [1.0 + 0.0j],
        [1.0 + 0.0j],
        [1.006 + 0.0j],
        tolerances,
    )
    assert separate["gpu_vs_analytic"]["pass"] is True
    assert separate["gpu_vs_ant"]["pass"] is False


def test_metadata_requires_exact_coordinates_and_rescale_rule(
    tmp_path: Path,
) -> None:
    fixture, _ = ordinary.load_fixture(
        FIXTURE_PATH, write_context_manifest(tmp_path)
    )
    contract = fixture["metadata_contracts"]["cipher_q3_rescaled"]
    dropped = (1 << 56) - 5
    observed = {
        "object_kind": "ciphertext",
        "ace_level": 3,
        "active_q_count": 3,
        "phantom_chain_index": 2,
        "scale_degree": 1,
        "raw_scale": (2.0**112) / dropped,
        "dropped_modulus": dropped,
        "slots": 8192,
        "ciphertext_size": 2,
        "ntt": True,
    }
    passed, errors = ordinary.validate_observed_metadata(
        observed, contract, fixture["tolerances"], fixture, 1
    )
    assert passed is True
    assert errors == []

    wrong = copy.deepcopy(observed)
    wrong["phantom_chain_index"] = 3
    passed, errors = ordinary.validate_observed_metadata(
        wrong, contract, fixture["tolerances"], fixture, 1
    )
    assert passed is False
    assert any("phantom_chain_index" in error for error in errors)


def test_provider_interface_rejects_claim_without_independent_encoding(
    tmp_path: Path,
) -> None:
    fixture, fixture_digest = ordinary.load_fixture(
        FIXTURE_PATH, write_context_manifest(tmp_path)
    )
    provider_path = tmp_path / "provider.json"
    provider_path.write_text(
        json.dumps(
            {
                "schema_version": ordinary.PROVIDER_SCHEMA,
                "provider": "ant",
                "fixture_id": fixture["fixture_id"],
                "fixture_sha256": fixture_digest,
                "compiler_context_manifest_sha256": fixture[
                    "compiler_context_manifest"
                ]["sha256"],
                "producer": "invalid-test-producer",
                "independent_encoding": False,
                "input_kind": "clear_fixture_values",
                "binary": {},
                "records": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ordinary.OrdinaryCkksError, match="independently encode clear fixture values"
    ):
        ordinary._load_provider(
            provider_path,
            tmp_path / "absent.bin",
            fixture,
            fixture_digest,
            "ant",
        )
