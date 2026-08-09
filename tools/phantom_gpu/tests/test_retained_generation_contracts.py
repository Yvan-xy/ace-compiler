"""Static contracts for retained source and production-RNUM generation."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import re

import pytest


TOOLS = Path(__file__).resolve().parents[1]
REPOSITORY = TOOLS.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fixture_tool = _load(
    "retained_fixture_generation_contract",
    TOOLS / "generate_retained_ckks_fixtures.py",
)
air_tools = _load("retained_air_generation_contract", TOOLS / "retained_air_tools.py")


def test_unbound_fixture_retains_reviewed_batches_but_has_no_air_hash() -> None:
    fixture = fixture_tool.load_json(TOOLS / "fixtures/retained_ckks_v1.json")
    fixture["qualification_bindings"] = {
        "status": "unbound",
        "required": [
            "compiler_context_manifest_sha256",
            "normalized_compiler_command_sha256",
            "post_ckks_air_sha256",
        ],
    }
    fixture["production_rotation_source"]["post_ckks_air_sha256"] = None
    fixture_tool.validate_template(fixture, require_bound=False)
    assert fixture["qualification_bindings"]["status"] == "unbound"
    assert fixture["production_rotation_batches"]
    assert fixture["production_rotation_source"]["post_ckks_air_sha256"] is None

    invalid = copy.deepcopy(fixture)
    invalid["production_rotation_source"]["post_ckks_air_sha256"] = "0" * 64
    with pytest.raises(
        fixture_tool.RetainedFixtureError,
        match="unbound fixture must not predeclare",
    ):
        fixture_tool.validate_template(invalid, require_bound=False)


def test_checked_fixture_is_bound_to_the_reviewed_candidate() -> None:
    fixture = fixture_tool.load_json(TOOLS / "fixtures/retained_ckks_v1.json")
    assert fixture["qualification_bindings"] == {
        "status": "bound",
        "compiler_context_manifest_sha256": (
            "bfe7c938a76947da37fa3fd4473916a979447f56ec5e8fb3b283ac1de3718891"
        ),
        "normalized_compiler_command_sha256": (
            "ef32ef6f9f075e5e58068867d1628dd90bdec40e5e0f0a6a0971c5f3ea9fae72"
        ),
        "post_ckks_air_sha256": (
            "ea9a7405835d8339af0248da11ec6329be00c00ab1663384ef041af628a8ff68"
        ),
    }
    assert fixture["production_rotation_source"]["post_ckks_air_sha256"] == (
        "8957f4d21c700058b9cca1650bf0f883ca471319f7f21e67c18832601ca867fe"
    )
    fixture_tool.validate_template(fixture, require_bound=True)

    formal_gate = (TOOLS / "retained_runpod_evidence.py").read_text(
        encoding="utf-8"
    )
    assert 'get("status") != "bound"' in formal_gate
    assert "checked fixture differs from the generation fixture" in formal_gate
    assert (
        'fixture_bindings.get("compiler_context_manifest_sha256")'
        in formal_gate
    )
    assert 'fixture.get("compiler_context_manifest")' not in formal_gate


def test_checkout_path_canonicalization_updates_air_string_sizes() -> None:
    physical = (REPOSITORY / "ace_edsl/examples/ckks_retained_ops.py").as_posix()
    logical = "ace_edsl/examples/ckks_retained_ops.py"
    air = (
        f"STRING TABLE ({len(physical)} Bytes)\n"
        f'STR[0x1] "{physical}" length(0x{len(physical):x})\n'
    )
    canonical = air_tools.canonicalize_checkout_paths(air, REPOSITORY)
    assert physical not in canonical
    assert f"STRING TABLE ({len(logical)} Bytes)" in canonical
    assert f'"{logical}" length(0x{len(logical):x})' in canonical


def test_default_production_recipe_stops_after_ckks_driver() -> None:
    source = (TOOLS / "generate_retained_production_air.py").read_text(
        encoding="utf-8"
    )
    assert "pipeline.run_ckks_driver()" in source
    assert '"terminal_driver": None' in source
    for forbidden in (
        "run_poly_driver(",
        "run_poly2c(",
        "run_ckks2c(",
        "Prepare_context(",
        "Encrypt(",
        "Decrypt(",
    ):
        assert forbidden not in source


def test_retained_generator_cli_owns_compiler_context_parameters() -> None:
    source = (TOOLS / "generate_retained_ckks_sources.py").read_text(
        encoding="utf-8"
    )
    for option, parameter in (
        ("--polynomial-degree", "poly_degree=compiler_parameters"),
        ("--mul-level", "mul_level=compiler_parameters"),
        ("--input-level", "input_level=compiler_parameters"),
        ("--security-level", "security_level=compiler_parameters"),
        ("--scaling-modulus-bits", "scaling_factor_bits=compiler_parameters"),
        ("--first-modulus-bits", "first_prime_bits=compiler_parameters"),
        ("--hamming-weight", "hamming_weight=compiler_parameters"),
    ):
        assert f'parser.add_argument("{option}", required=True' in source
        assert parameter in source


def test_host_qualification_has_one_context_parameter_authority() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    assignments = {
        "POLYNOMIAL_DEGREE": "16384",
        "MUL_LEVEL": "26",
        "INPUT_LEVEL": "1",
        "SECURITY_LEVEL": "0",
        "SCALING_MODULUS_BITS": "56",
        "FIRST_MODULUS_BITS": "60",
        "HAMMING_WEIGHT": "192",
    }
    assert (
        "These are compiler CLI arguments, not a second parameter profile" in host
    )
    assert "only downstream parameter authority" in host
    for name, value in assignments.items():
        assert re.findall(rf"(?m)^{name}=([0-9]+)$", host) == [value]

    generator = (TOOLS / "generate_retained_ckks_sources.py").read_text(
        encoding="utf-8"
    )
    assert "if compiler_parameters != expected_parameters:" in generator
    assert '"argv": canonical_argv' in generator
    assert '"normalized_argv_sha256": hashlib.sha256(' in generator
    evidence = (TOOLS / "retained_runpod_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "retained_comparator.verify_invocation_context(" in evidence
    assert 'artifact.get("normalized_compiler_command_sha256")' in evidence

    downstream = (
        "package_runpod_sources.sh",
        "run_build_and_health.sh",
        "run_local_reproduction.sh",
        "runpod_transfer.sh",
        "harness/retained_ckks_ant_oracle.cxx",
        "harness/retained_ckks_phantom_conformance.cu",
    )
    assignment_pattern = re.compile(
        rf"(?m)^({'|'.join(assignments)})\s*=\s*[0-9]+$"
    )
    literal_flag_pattern = re.compile(
        r"--(?:polynomial-degree|poly-degree|mul-level|input-level|"
        r"security-level|scaling-modulus-bits|scaling-factor-bits|"
        r"first-modulus-bits|first-prime-bits|hamming-weight)[ =]+[0-9]+"
    )
    for relative in downstream:
        source = (TOOLS / relative).read_text(encoding="utf-8")
        assert "kArithmeticLevel" not in source
        assert not assignment_pattern.search(source)
        assert not literal_flag_pattern.search(source)


def test_ant_oracle_invokes_the_generated_composite() -> None:
    source = (TOOLS / "harness/retained_ckks_ant_oracle.cxx").read_text(
        encoding="utf-8"
    )
    assert "CIPHERTEXT retained_ckks_composite(CIPHERTEXT input);" in source
    assert "CIPHERTEXT result = retained_ckks_composite(*source);" in source
    assert "DecodeStrictQ0" in source
    assert "Modswitch_ciph(projected)" in source
    assert "ANT decoded projection changed the retained q0 tower" in source
    assert "ANT decoded projection mutated the full-Q result" in source
    assert source.count("full_q_count, slots, nullptr, true") == 2


def test_phantom_harness_invokes_the_generated_composite_interface() -> None:
    source = (
        TOOLS / "harness/retained_ckks_phantom_conformance.cu"
    ).read_text(encoding="utf-8")
    assert '#include "retained_ckks_generated_interface.h"' in source
    assert "CIPHERTEXT result = retained_ckks_composite(*source);" in source
