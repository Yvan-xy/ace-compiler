"""Static contracts for retained source and production-RNUM generation."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

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


def test_ant_oracle_invokes_the_generated_composite() -> None:
    source = (TOOLS / "harness/retained_ckks_ant_oracle.cxx").read_text(
        encoding="utf-8"
    )
    assert "CIPHERTEXT retained_ckks_composite(CIPHERTEXT input);" in source
    assert "CIPHERTEXT result = retained_ckks_composite(*source);" in source
