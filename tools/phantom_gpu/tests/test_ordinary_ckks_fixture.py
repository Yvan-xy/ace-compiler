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


def write_unbound_fixture(tmp_path: Path) -> Path:
    fixture = ordinary.load_json(FIXTURE_PATH)
    fixture["qualification_bindings"].update(
        {
            "status": "unbound",
            "normalized_compiler_command_sha256": None,
            "post_ckks_air_sha256": None,
        }
    )
    fixture["compiler_context_manifest"]["sha256"] = None
    path = tmp_path / "ordinary_ckks_unbound_template.json"
    ordinary.write_json(path, fixture)
    ordinary.validate_fixture(fixture)
    return path


def compiler_argv(mul_level: int = 4) -> list[str]:
    return [
        "tools/phantom_gpu/generate_ckks2c_probe.py",
        "--output",
        "ckks2c/add_mul_rotate.cu",
        "--post-ckks-air",
        "ckks2c/ordinary_ckks_post_ckks.air",
        "--context-manifest",
        "ckks2c/compiler_context_manifest.json",
        "--resource-manifest",
        "ckks2c/compiler_resource_manifest.json",
        "--poly-degree",
        "16384",
        "--mul-level",
        str(mul_level),
        "--input-level",
        "1",
        "--security-level",
        "0",
        "--scaling-factor-bits",
        "56",
        "--first-prime-bits",
        "60",
        "--hamming-weight",
        "192",
    ]


def write_bound_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    context_manifest_path = write_context_manifest(tmp_path)
    invocation_path = tmp_path / "compiler_invocation.json"
    argv = compiler_argv()
    invocation_path.write_text(
        json.dumps(
            {
                "schema_version": ordinary.INVOCATION_SCHEMA,
                "argv": argv,
                "normalized_argv_sha256": ordinary.sha256_bytes(
                    ordinary.canonical_bytes(argv)
                ),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    air_path = tmp_path / "ordinary_ckks_post_ckks.air"
    air_path.write_text("fhe::ckks test fixture\n", encoding="utf-8")
    bound_path = tmp_path / "ordinary_ckks_v1.json"
    ordinary.bind_fixture(
        write_unbound_fixture(tmp_path),
        context_manifest_path,
        invocation_path,
        air_path,
        bound_path,
    )
    return bound_path, invocation_path, air_path


def test_qualification_invocation_recorder_has_one_parameter_authority(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification_invocation.json"
    argv = [
        "tools/phantom_gpu/compile_only.sh",
        "--gate",
        "ordinary",
        "--poly-degree",
        "16384",
        "--mul-level",
        "4",
        "--input-level",
        "1",
        "--security-level",
        "0",
        "--scaling-factor-bits",
        "56",
        "--first-prime-bits",
        "60",
        "--hamming-weight",
        "192",
    ]
    record = ordinary.record_qualification_invocation(output, argv)
    assert ordinary.load_json(output) == record
    assert record == {
        "schema_version": ordinary.QUALIFICATION_INVOCATION_SCHEMA,
        "argv": argv,
        "normalized_argv_sha256": ordinary.sha256_bytes(
            ordinary.canonical_bytes(argv)
        ),
    }

    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="incomplete or duplicated",
    ):
        ordinary.record_qualification_invocation(
            tmp_path / "invalid.json", argv[:-2]
        )


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
    assert fixture["qualification_bindings"] == {
        "status": "bound",
        "deterministic_seed": 20260807,
        "deterministic_generator": "splitmix64-float53-complex-v1",
        "normalized_compiler_command_sha256": (
            "e966e89fb34bab0c760909b76940cc4d8a6020320bb676705cde30a190097170"
        ),
        "post_ckks_air_sha256": (
            "176eda2e57195873dd1793bd921ccea679750f8b953db4292ac6cbaacc7b46ce"
        ),
    }
    assert fixture["compiler_context_manifest"]["sha256"] == (
        "2a2927711c260f72257abebffd9f74f3cb9aeaffd5a8f44c4d7efa5764e6569f"
    )
    assert digest == "10114d36488b06a12237fce96bf867eb6f06c42c9d9dd4d05bab771111b3a28e"
    assert fixture["coordinate_rules"] == {
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
    assert fixture["qualification_bindings"]["deterministic_seed"] == 20260807
    assert {
        contract["active_q_count"]
        for contract in fixture["metadata_contracts"].values()
    } == {"full", "after_one_drop", "middle", "bottom"}
    subtraction = next(
        family for family in fixture["case_families"] if family["id"] == "sub_ct_ct"
    )
    assert subtraction["aliases"] == ["distinct", "lhs", "rhs"]
    assert fixture["alias_matrix"]["ct_ct"]["rhs"] == "supported"
    assert fixture["alias_matrix"]["ct_plain"]["rhs"] == "not_applicable"
    assert fixture["alias_matrix"]["ct_scalar"]["rhs"] == "not_applicable"
    assert fixture["alias_matrix"]["encode"] == {
        "distinct": "supported",
        "inplace": "not_applicable",
    }
    assert fixture["alias_matrix"]["query"] == {
        "distinct": "supported",
        "inplace": "not_applicable",
    }
    assert all(
        disposition != "rejected"
        for relations in fixture["alias_matrix"].values()
        for disposition in relations.values()
    )
    for family in fixture["case_families"]:
        assert family["aliases"] == [
            alias
            for alias, disposition in fixture["alias_matrix"][family["form"]].items()
            if disposition == "supported"
        ]
    assert len(ordinary.expanded_cases(fixture)) == sum(
        len(family["aliases"]) for family in fixture["case_families"]
    )


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


def test_ownership_and_runtime_diagnostic_contracts_are_closed() -> None:
    fixture = ordinary.load_json(FIXTURE_PATH)
    ordinary.validate_fixture(fixture)

    unknown_ownership = copy.deepcopy(fixture)
    unknown_ownership["ownership_cases"][0]["id"] = "unimplemented_case"
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="not a supported ownership contract",
    ):
        ordinary.validate_fixture(unknown_ownership)

    wrong_array_length = copy.deepcopy(fixture)
    free_array = next(
        item
        for item in wrong_array_length["ownership_cases"]
        if item["id"] == "free_array_4"
    )
    free_array["array_length"] = 3
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="array_length must be 4",
    ):
        ordinary.validate_fixture(wrong_array_length)

    incomplete_ownership = copy.deepcopy(fixture)
    incomplete_ownership["ownership_cases"].pop()
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="complete ownership contract",
    ):
        ordinary.validate_fixture(incomplete_ownership)

    missing_provider_diagnostic = copy.deepcopy(fixture)
    runtime = next(
        item
        for item in missing_provider_diagnostic["rejections"]
        if item["phase"] == "runtime"
    )
    runtime.pop("provider_diagnostic")
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="provider_diagnostic must be an uppercase token",
    ):
        ordinary.validate_fixture(missing_provider_diagnostic)

    static_with_runner = copy.deepcopy(fixture)
    static = next(
        item
        for item in static_with_runner["rejections"]
        if item["phase"] == "static"
    )
    static["runner_profile"] = "ordinary"
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="static rejection cannot select",
    ):
        ordinary.validate_fixture(static_with_runner)


def test_alias_contract_is_api_specific_and_closed() -> None:
    fixture = ordinary.load_json(FIXTURE_PATH)
    ordinary.validate_fixture(fixture)

    missing_supported_alias = copy.deepcopy(fixture)
    subtraction = next(
        family
        for family in missing_supported_alias["case_families"]
        if family["id"] == "sub_ct_ct"
    )
    subtraction["aliases"].remove("rhs")
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="must exercise every supported relation",
    ):
        ordinary.validate_fixture(missing_supported_alias)

    query_with_cipher_alias_contract = copy.deepcopy(fixture)
    query = next(
        family
        for family in query_with_cipher_alias_contract["case_families"]
        if family["id"] == "query_all"
    )
    query["form"] = "cipher_unary"
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="form must be query",
    ):
        ordinary.validate_fixture(query_with_cipher_alias_contract)

    fabricated_rejection = copy.deepcopy(fixture)
    fabricated_rejection["alias_matrix"]["ct_ct"]["rhs"] = "rejected"
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="must exactly describe",
    ):
        ordinary.validate_fixture(fabricated_rejection)

    for form in ("zero", "free", "free_array"):
        assert fixture["alias_matrix"][form] == {
            "destination_source": "not_applicable"
        }


@pytest.mark.parametrize(
    "duplicate_input",
    ("fixture", "context_manifest", "compiler_invocation"),
)
def test_bind_fixture_rejects_duplicate_keys_in_owned_json_inputs(
    tmp_path: Path, duplicate_input: str
) -> None:
    fixture_path = write_unbound_fixture(tmp_path)
    context_manifest_path = write_context_manifest(tmp_path)
    invocation_path = tmp_path / "compiler_invocation.json"
    argv = compiler_argv()
    ordinary.write_json(
        invocation_path,
        {
            "schema_version": ordinary.INVOCATION_SCHEMA,
            "argv": argv,
            "normalized_argv_sha256": ordinary.sha256_bytes(
                ordinary.canonical_bytes(argv)
            ),
        },
    )
    if duplicate_input == "fixture":
        fixture_path = tmp_path / "duplicate-fixture.json"
        fixture_path.write_text(
            '{"schema_version":"one","schema_version":"two"}\n',
            encoding="utf-8",
        )
    elif duplicate_input == "context_manifest":
        context_manifest_path.write_text(
            '{"schema_version":1,"schema_version":1}\n', encoding="utf-8"
        )
    else:
        invocation_path.write_text(
            '{"schema_version":"one","schema_version":"two"}\n',
            encoding="utf-8",
        )
    air_path = tmp_path / "ordinary_ckks_post_ckks.air"
    air_path.write_text("post-CKKS AIR\n", encoding="utf-8")

    with pytest.raises(ordinary.OrdinaryCkksError, match="duplicate JSON key"):
        ordinary.bind_fixture(
            fixture_path,
            context_manifest_path,
            invocation_path,
            air_path,
            tmp_path / "bound-fixture.json",
        )


def test_qualification_binding_rejects_unbound_and_changed_artifacts(
    tmp_path: Path,
) -> None:
    context_manifest = write_context_manifest(tmp_path)
    template_path = write_unbound_fixture(tmp_path)
    template = ordinary.load_json(template_path)
    assert template["compiler_context_manifest"] == {"sha256": None}
    ordinary.validate_fixture(template)
    invalid_template = copy.deepcopy(template)
    invalid_template["compiler_context_manifest"]["sha256"] = "0" * 64
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="unbound fixture compiler context manifest hash must be null",
    ):
        ordinary.validate_fixture(invalid_template)
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="not bound to qualification artifacts",
    ):
        ordinary.load_fixture(template_path, context_manifest)

    checked, _ = ordinary.load_fixture(FIXTURE_PATH, context_manifest)
    assert checked["qualification_bindings"]["status"] == "bound"

    fixture_path, invocation_path, air_path = write_bound_fixture(tmp_path)
    fixture, _ = ordinary.load_fixture(fixture_path, context_manifest)
    ordinary.verify_qualification_bindings(
        fixture, context_manifest, invocation_path, air_path
    )
    invalid_bound = copy.deepcopy(fixture)
    invalid_bound.pop("_resolved_context")
    invalid_bound["compiler_context_manifest"]["sha256"] = None
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="must be a lowercase SHA-256 digest",
    ):
        ordinary.validate_fixture(invalid_bound)

    air_path.write_text("changed post-CKKS AIR\n", encoding="utf-8")
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="post-CKKS AIR binding does not match",
    ):
        ordinary.verify_qualification_bindings(
            fixture, context_manifest, invocation_path, air_path
        )

    changed_seed = ordinary.load_json(FIXTURE_PATH)
    changed_seed["qualification_bindings"]["deterministic_seed"] += 1
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="does not match deterministic seed",
    ):
        ordinary.validate_fixture(changed_seed)


def test_bind_fixture_strictly_validates_context_before_atomic_output(
    tmp_path: Path,
) -> None:
    _, invocation_path, air_path = write_bound_fixture(tmp_path)
    invalid_context = json.loads(CONTEXT_MANIFEST_TEXT)
    invalid_context["packing"] = "half"
    context_manifest_path = tmp_path / "invalid-context.json"
    ordinary.write_json(context_manifest_path, invalid_context)
    output_path = tmp_path / "bound-fixture.json"
    output_path.write_text("preserve-existing-output\n", encoding="utf-8")

    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="packing is unsupported",
    ):
        ordinary.bind_fixture(
            write_unbound_fixture(tmp_path),
            context_manifest_path,
            invocation_path,
            air_path,
            output_path,
        )
    assert output_path.read_text(encoding="utf-8") == "preserve-existing-output\n"


def test_bind_fixture_rejects_compiler_command_context_disagreement(
    tmp_path: Path,
) -> None:
    context_manifest_path = write_context_manifest(tmp_path)
    invocation_path = tmp_path / "compiler_invocation.json"
    argv = compiler_argv(mul_level=5)
    ordinary.write_json(
        invocation_path,
        {
            "schema_version": ordinary.INVOCATION_SCHEMA,
            "argv": argv,
            "normalized_argv_sha256": ordinary.sha256_bytes(
                ordinary.canonical_bytes(argv)
            ),
        },
    )
    air_path = tmp_path / "ordinary_ckks_post_ckks.air"
    air_path.write_text("post-CKKS AIR\n", encoding="utf-8")
    with pytest.raises(
        ordinary.OrdinaryCkksError,
        match="--mul-level disagrees with the emitted context manifest",
    ):
        ordinary.bind_fixture(
            write_unbound_fixture(tmp_path),
            context_manifest_path,
            invocation_path,
            air_path,
            tmp_path / "bound-fixture.json",
        )


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    (
        ("degree", "polynomial degree exceeds provider limits"),
        ("data_bits", "data-Q bit sizes exceed provider limits"),
        ("special_bits", "special_p_bit_sizes\\[0\\] must be at least 2"),
        ("modulus_count", "combined Q/P count exceeds provider limits"),
    ),
)
def test_context_manifest_rejects_provider_unsupported_parameters(
    case: str, diagnostic: str
) -> None:
    manifest = json.loads(CONTEXT_MANIFEST_TEXT)
    if case == "degree":
        manifest["polynomial_degree"] = 262144
        manifest["logical_slot_capacity"] = 131072
    elif case == "data_bits":
        manifest["first_modulus_bits"] = 1
        manifest["data_q_bit_sizes"][0] = 1
    elif case == "special_bits":
        manifest["special_p_bit_sizes"][0] = 1
    elif case == "modulus_count":
        manifest["data_q_bit_sizes"] = [60] + [56] * 62
        manifest["input_level"] = 1
        manifest["q_part_count"] = 1
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(case)
    with pytest.raises(ordinary.OrdinaryCkksError, match=diagnostic):
        ordinary.validate_context_manifest(manifest)


def test_symbolic_coordinates_follow_a_different_context_chain(
    tmp_path: Path,
) -> None:
    manifest = json.loads(CONTEXT_MANIFEST_TEXT)
    manifest["data_q_bit_sizes"].append(56)
    manifest_path = tmp_path / "compiler_context_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    template_path = write_unbound_fixture(tmp_path)
    invocation_path = tmp_path / "compiler_invocation.json"
    argv = compiler_argv(mul_level=5)
    ordinary.write_json(
        invocation_path,
        {
            "schema_version": ordinary.INVOCATION_SCHEMA,
            "argv": argv,
            "normalized_argv_sha256": ordinary.sha256_bytes(
                ordinary.canonical_bytes(argv)
            ),
        },
    )
    air_path = tmp_path / "ordinary_ckks_post_ckks.air"
    air_path.write_text("different-chain AIR\n", encoding="utf-8")
    fixture_path = tmp_path / "ordinary_ckks_v1.json"
    ordinary.bind_fixture(
        template_path,
        manifest_path,
        invocation_path,
        air_path,
        fixture_path,
    )
    fixture, _ = ordinary.load_fixture(fixture_path, manifest_path)

    assert ordinary.resolve_active_q_count("full", fixture, "test") == 5
    assert ordinary.resolve_active_q_count(
        "after_one_drop", fixture, "test"
    ) == 4
    assert ordinary.resolve_active_q_count("middle", fixture, "test") == 3
    assert ordinary.resolve_active_q_count("bottom", fixture, "test") == 1


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
    contract = fixture["metadata_contracts"]["cipher_after_one_drop_rescaled"]
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
