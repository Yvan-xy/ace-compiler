"""Raw-AIR tests for the provider-neutral retained CKKS micrographs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
FIXTURE = REPOSITORY / "tools" / "phantom_gpu" / "fixtures" / "retained_ckks_v1.json"


def _context(path: Path) -> None:
    value = {
        "schema_version": 1,
        "packing": "full",
        "polynomial_degree": 16384,
        "logical_slot_capacity": 8192,
        "data_q_bit_sizes": [60, 56, 56, 56],
        "special_p_bit_sizes": [60, 60],
        "input_level": 1,
        "q_part_count": 2,
        "hamming_weight": 192,
        "security_level": 0,
        "first_modulus_bits": 60,
        "scaling_modulus_bits": 56,
        "resource_schema_version": 2,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _emit(tmp_path: Path, graph: str) -> str:
    context = tmp_path / "context.json"
    _context(context)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), str(EXAMPLES), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                from pathlib import Path
                from ckks_retained_ops import emit_air
                emit_air(
                    Path({str(context)!r}),
                    Path({str(FIXTURE)!r}),
                    {graph!r},
                    Path({str(tmp_path / (graph + '.air'))!r}),
                )
                """
            ),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return (tmp_path / f"{graph}.air").read_text(encoding="utf-8").lower()


def _assert_no_forbidden_boundary_ops(air: str) -> None:
    for token in (
        "ckks.bootstrap",
        "bootstrap_coeffs_to_slots",
        "bootstrap_eval_mod",
        "bootstrap_slots_to_coeffs",
        "fhe::poly",
    ):
        assert token not in air


def test_four_focused_micrographs_preserve_their_ckks_operations(tmp_path: Path) -> None:
    expected = {
        "conjugate": "ckks.conjugate",
        "rotate_batch": "ckks.rotate_batch",
        "raise_mod": "ckks.raise_mod",
        "mul_mono": "ckks.mul_mono",
    }
    for graph, token in expected.items():
        air = _emit(tmp_path, graph)
        assert token in air
        _assert_no_forbidden_boundary_ops(air)


def test_composite_preserves_ordered_batch_and_all_retained_operations(tmp_path: Path) -> None:
    air = _emit(tmp_path, "composite")
    for token in (
        "ckks.raise_mod",
        "ckks.mul_mono",
        "ckks.conjugate",
        "ckks.rotate_batch",
    ):
        assert token in air
    compact = "".join(air.split())
    assert "nums=(5,0,-7,5)" in compact
    _assert_no_forbidden_boundary_ops(air)


def test_rotation_micrograph_emits_edge_and_every_frozen_production_batch(
    tmp_path: Path,
) -> None:
    air = "".join(_emit(tmp_path, "rotate_batch").split())
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    batches = [[5, 0, -7, 5]] + fixture["production_rotation_batches"]
    assert air.count("ckks.rotate_batch") == len(batches)
    for batch in batches:
        rendered = ",".join(str(step) for step in batch)
        assert f"nums=({rendered})" in air


def test_rotation_batches_survive_driver_and_source_emission(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    _context(context)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), str(EXAMPLES), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext
                from ckks_retained_ops import create_retained_ckks_micrographs

                context_path = Path({str(context)!r})
                fixture_path = Path({str(FIXTURE)!r})
                manifest = json.loads(context_path.read_text(encoding="utf-8"))
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                batches = [[5, 0, -7, 5]] + fixture["production_rotation_batches"]
                AceEDSL._get_dsl.cache_clear()
                graphs = create_retained_ckks_micrographs(context_path, fixture_path)
                graphs.rotate_batch(
                    CkksCiphertext(shape=(graphs.polynomial_degree,), name="input_ct")
                )
                module = AceEDSL._get_dsl().current_air_module
                pipeline = AcePipeline(module).configure_fhe(
                    poly_degree=manifest["polynomial_degree"],
                    mul_level=len(manifest["data_q_bit_sizes"]),
                    input_level=manifest["input_level"],
                    security_level=manifest["security_level"],
                    scaling_factor_bits=manifest["scaling_modulus_bits"],
                    first_prime_bits=manifest["first_modulus_bits"],
                    hamming_weight=manifest["hamming_weight"],
                    data_file="",
                    provider="phantom",
                    codegen_ir="ckks",
                )
                compiled = pipeline.run(
                    start_domain="fhe::ckks", dump_stages=True, verbose=False
                )
                if (
                    not compiled.success
                    and "context_manifest_file" in compiled.error
                    and "incompatible function arguments" in compiled.error
                ):
                    print("RETAINED_BINDINGS_REQUIRE_REBUILD")
                    raise SystemExit(0)
                assert compiled.success, compiled.error
                post_air = compiled.air_dumps["ckks_driver"].lower()
                assert post_air.count("ckks.rotate_batch") == len(batches)
                source = compiled.c_code or ""
                assert source.count("Rotate_batch_ciph(") == len(batches)
                cursor = -1
                for batch in batches:
                    initializer = "{{" + ", ".join(str(step) for step in batch) + "}}"
                    cursor = source.find(initializer, cursor + 1)
                    assert cursor >= 0, initializer
                print("RETAINED_ROTATION_SOURCE_OK")
                """
            ),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if "RETAINED_BINDINGS_REQUIRE_REBUILD" in result.stdout:
        pytest.skip("local Python bindings predate the retained manifest ABI")
    assert "RETAINED_ROTATION_SOURCE_OK" in result.stdout


def test_micrograph_dimensions_are_manifest_derived(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    _context(context)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), str(EXAMPLES), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                from pathlib import Path
                from ckks_retained_ops import create_retained_ckks_micrographs
                graphs = create_retained_ckks_micrographs(
                    Path({str(context)!r}), Path({str(FIXTURE)!r})
                )
                assert graphs.polynomial_degree == 16384
                assert graphs.logical_slots == 8192
                assert graphs.full_data_q_count == 4
                assert len(graphs.production_rotation_batches) == 6
                """
            ),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_phantom_pipeline_rejects_runtime_raise_helper(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    _context(context)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), str(EXAMPLES), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import json
                from pathlib import Path
                from ace_edsl.edsl import (
                    AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
                )

                manifest = json.loads(
                    Path({str(context)!r}).read_text(encoding="utf-8")
                )

                @ckks_kernel
                def invalid_raise(ct: CkksCiphertext) -> CkksCiphertext:
                    return ct.raise_mod(
                        len(manifest["data_q_bit_sizes"]),
                        runtime_raise_level=True,
                    )

                AceEDSL._get_dsl.cache_clear()
                invalid_raise(
                    CkksCiphertext(
                        shape=(manifest["polynomial_degree"],), name="input_ct"
                    )
                )
                module = AceEDSL._get_dsl().current_air_module
                pipeline = AcePipeline(module).configure_fhe(
                    poly_degree=manifest["polynomial_degree"],
                    mul_level=len(manifest["data_q_bit_sizes"]),
                    input_level=manifest["input_level"],
                    security_level=manifest["security_level"],
                    scaling_factor_bits=manifest["scaling_modulus_bits"],
                    first_prime_bits=manifest["first_modulus_bits"],
                    hamming_weight=manifest["hamming_weight"],
                    data_file="",
                    provider="phantom",
                    codegen_ir="ckks",
                )
                compiled = pipeline.run(start_domain="fhe::ckks", verbose=False)
                assert not compiled.success
                assert "forbids runtime target-level helpers" in compiled.error
                print("RETAINED_DYNAMIC_RAISE_REJECTED")
                """
            ),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        result.returncode != 0
        and "forbids runtime target-level helpers" in result.stderr
        and "ValueError" in result.stderr
    ):
        pytest.skip("local Python bindings predate provider-bound rejection")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RETAINED_DYNAMIC_RAISE_REJECTED" in result.stdout
