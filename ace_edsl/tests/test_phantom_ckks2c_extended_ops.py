"""Raw-AIR tests for the provider-neutral retained CKKS micrographs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

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
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for token in (
        "ckks.raise_mod",
        "ckks.mul_mono",
        "ckks.conjugate",
        "ckks.rotate_batch",
    ):
        assert token in air
    compact = "".join(air.split())
    steps = ",".join(str(step) for step in fixture["rotate_batch_steps"])
    assert f"nums=({steps})" in compact
    _assert_no_forbidden_boundary_ops(air)


def test_rotation_micrograph_emits_edge_and_every_frozen_production_batch(
    tmp_path: Path,
) -> None:
    air = "".join(_emit(tmp_path, "rotate_batch").split())
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    batches = [fixture["rotate_batch_steps"]] + fixture[
        "production_rotation_batches"
    ]
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
                batches = [fixture["rotate_batch_steps"]] + fixture["production_rotation_batches"]
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
                import json
                from pathlib import Path
                from ckks_retained_ops import create_retained_ckks_micrographs
                manifest = json.loads(
                    Path({str(context)!r}).read_text(encoding="utf-8")
                )
                graphs = create_retained_ckks_micrographs(
                    Path({str(context)!r}), Path({str(FIXTURE)!r})
                )
                assert graphs.polynomial_degree == manifest["polynomial_degree"]
                assert graphs.logical_slots == manifest["logical_slot_capacity"]
                assert graphs.full_data_q_count == len(manifest["data_q_bit_sizes"])
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
    script = tmp_path / "invalid_dynamic_raise.py"
    script.write_text(
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
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RETAINED_DYNAMIC_RAISE_REJECTED" in result.stdout


def test_one_conformance_module_declares_the_complete_retained_matrix(
    tmp_path: Path,
) -> None:
    air = _emit(tmp_path, "conformance")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert air.count("ckks.conjugate") == 2
    assert air.count("ckks.rotate_batch") == (
        2 + len(fixture["production_rotation_batches"])
    )
    assert air.count("ckks.raise_mod") == 2
    assert air.count("ckks.mul_mono") == len(fixture["monomial_powers"]) + 1
    for function in (
        "retained_ckks_conjugate",
        "retained_ckks_rotate_batches",
        "retained_ckks_raise_mod",
        "retained_ckks_mul_monomials",
        "retained_ckks_composite",
    ):
        assert function in air
    _assert_no_forbidden_boundary_ops(air)


def test_same_conformance_module_generates_both_terminal_sources(
    tmp_path: Path,
) -> None:
    context = tmp_path / "context.json"
    _context(context)
    outputs = {
        "ant_source": tmp_path / "retained_ant.cxx",
        "phantom_source": tmp_path / "retained_phantom.cu",
        "ant_air": tmp_path / "retained_ant.air",
        "phantom_air": tmp_path / "retained_phantom.air",
        "emitted_context": tmp_path / "emitted_context.json",
        "emitted_resources": tmp_path / "emitted_resources.json",
        "record": tmp_path / "generation.json",
        "interface": tmp_path / "retained_generated.h",
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY), str(EXAMPLES), environment.get("PYTHONPATH", ""))
    )
    manifest = json.loads(context.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "tools/phantom_gpu/generate_retained_ckks_sources.py"),
            "--context-manifest", str(context),
            "--fixture", str(FIXTURE),
            "--ant-source", str(outputs["ant_source"]),
            "--phantom-source", str(outputs["phantom_source"]),
            "--ant-post-ckks-air", str(outputs["ant_air"]),
            "--phantom-post-ckks-air", str(outputs["phantom_air"]),
            "--phantom-context-manifest", str(outputs["emitted_context"]),
            "--phantom-resource-manifest", str(outputs["emitted_resources"]),
            "--generation-record", str(outputs["record"]),
            "--interface-header", str(outputs["interface"]),
            "--polynomial-degree", str(manifest["polynomial_degree"]),
            "--mul-level", str(len(manifest["data_q_bit_sizes"])),
            "--input-level", str(manifest["input_level"]),
            "--security-level", str(manifest["security_level"]),
            "--scaling-modulus-bits", str(manifest["scaling_modulus_bits"]),
            "--first-modulus-bits", str(manifest["first_modulus_bits"]),
            "--hamming-weight", str(manifest["hamming_weight"]),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert outputs["ant_air"].read_bytes() == outputs["phantom_air"].read_bytes()
    functions = (
        "retained_ckks_conjugate",
        "retained_ckks_rotate_batches",
        "retained_ckks_raise_mod",
        "retained_ckks_mul_monomials",
        "retained_ckks_composite",
    )
    for source_path in (outputs["ant_source"], outputs["phantom_source"]):
        source = source_path.read_text(encoding="utf-8")
        for function in functions:
            assert f"CIPHERTEXT {function}(CIPHERTEXT p0" in source
    interface = outputs["interface"].read_text(encoding="utf-8")
    assert (
        "CIPHERTEXT retained_ckks_composite(CIPHERTEXT input);" in interface
    )
    record = json.loads(outputs["record"].read_text(encoding="utf-8"))
    assert record["argv"][0] == (
        "tools/phantom_gpu/generate_retained_ckks_sources.py"
    )
    assert record["invocation"] == (
        "CIPHERTEXT output = retained_ckks_composite(*input_cipher);"
    )
    assert record["ant_post_ckks_air_sha256"] == (
        record["phantom_post_ckks_air_sha256"]
        == record["post_ckks_air_sha256"]
    )
    phantom = outputs["phantom_source"].read_text(encoding="utf-8")
    for call in ("Conjugate_ciph", "Rotate_batch_ciph", "Raise_mod", "Mul_mono_ciph"):
        assert f"{call}(" in phantom
    for forbidden in (
        "Eval_bootstrap_ciph(",
        "Bootstrap(",
        "Coeff_to_slot(",
        "Slot_to_coeff(",
        "Eval_mod(",
        "Hw_mod",
        "Poly_",
    ):
        assert forbidden not in phantom
    resources = json.loads(outputs["emitted_resources"].read_text(encoding="utf-8"))
    assert resources["rotation_batches"] == [
        fixture["rotate_batch_steps"],
        *fixture["production_rotation_batches"],
        fixture["rotate_batch_steps"],
    ]
    degree = manifest["polynomial_degree"]
    expected_powers = {
        "0": 0,
        "N/2": degree // 2,
        "N": degree,
        "3N/2": degree + degree // 2,
        "2N-1": 2 * degree - 1,
        "2N+1": 1,
    }
    assert resources["monomial_powers"] == sorted(
        expected_powers[symbol] for symbol in fixture["monomial_powers"]
    )
    assert 0 not in resources["rotation_steps"]
    assert len(resources["rotation_steps"]) == len(set(resources["rotation_steps"]))
