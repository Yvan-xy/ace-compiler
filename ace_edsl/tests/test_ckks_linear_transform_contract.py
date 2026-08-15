"""Contracts for the compiler-only CKKS linear-transform descriptor."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_linear_transform_descriptor_survives_ckks_analysis():
    result = _run_isolated(
        r'''
        from ace_edsl.edsl import (
            AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
        )

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def transform_kernel(a: CkksCiphertext) -> CkksCiphertext:
            coefficients = [complex(index, -index) for index in range(12)]
            return a._linear_transform(
                coefficients,
                rot_in=[0, 1],
                rot_out=[0, -2],
                slots=4,
                term_count=3,
                scale_degree=1,
                plain_level=3,
                num_p=2,
                encode_cache=True,
            )

        value = CkksCiphertext(shape=(4,), name="value")
        transform_kernel(value)
        glob = AceEDSL._get_dsl().current_air_module
        raw = glob.dump().lower()
        assert raw.count("ckks.linear_transform") == 1, raw
        for token in (
            "lt_schema_version=1",
            "lt_slots=4",
            "lt_term_count=3",
            "lt_rot_in=(0,1)",
            "lt_rot_out=(0,-2)",
            "lt_scale_degree=1",
            "lt_plain_level=3",
            "lt_num_p=2",
            "lt_encode_cache=1",
        ):
            assert token in raw.replace(" ", ""), (token, raw)

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16,
            mul_level=4,
            input_level=4,
            security_level=0,
            scaling_factor_bits=20,
            first_prime_bits=60,
            hamming_weight=8,
            provider="phantom",
            codegen_ir="ckks",
        )
        analyzed = pipeline.run_ckks_driver()
        assert analyzed["success"], analyzed
        post = glob.dump().lower().replace(" ", "")
        assert post.count("ckks.linear_transform") == 1, post
        assert "scale=2" in post, post
        assert "level=4" in post, post
        print("CKKS_LINEAR_TRANSFORM_CONTRACT_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS_LINEAR_TRANSFORM_CONTRACT_OK" in result.stdout


def test_linear_transform_descriptor_rejects_inconsistent_shape():
    result = _run_isolated(
        r'''
        from ace_edsl.edsl import AceEDSL, CkksCiphertext, ckks_kernel

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def invalid_kernel(a: CkksCiphertext) -> CkksCiphertext:
            return a._linear_transform(
                [0j] * 7,
                rot_in=[0, 1],
                rot_out=[0, 2],
                slots=4,
                term_count=2,
            )

        try:
            invalid_kernel(CkksCiphertext(shape=(4,), name="value"))
        except RuntimeError as error:
            assert "coefficient count must equal term_count * slots" in str(error)
        else:
            raise AssertionError("invalid descriptor was accepted")
        print("CKKS_LINEAR_TRANSFORM_REJECTION_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS_LINEAR_TRANSFORM_REJECTION_OK" in result.stdout


def test_linear_transform_reaches_lpoly_without_semantic_op_or_batch_call():
    result = _run_isolated(
        r'''
        from ace_edsl.edsl import (
            AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
        )

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def transform_kernel(a: CkksCiphertext) -> CkksCiphertext:
            coefficients = [complex(index + 1, -index) for index in range(16)]
            return a._linear_transform(
                coefficients,
                rot_in=[0, 1],
                rot_out=[0, -2],
                slots=4,
                term_count=4,
                scale_degree=1,
                plain_level=3,
                num_p=2,
                encode_cache=True,
            )

        transform_kernel(CkksCiphertext(shape=(4,), name="value"))
        glob = AceEDSL._get_dsl().current_air_module
        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16,
            # This profile deliberately makes the configured sf_bits=20 use
            # two P primes while the binding default sf_bits=40 would use
            # three.  POLY type preparation must preserve invocation-owned
            # parameters before LINEAR_TRANSFORM validates its descriptor.
            mul_level=10,
            input_level=10,
            security_level=0,
            scaling_factor_bits=20,
            first_prime_bits=60,
            hamming_weight=8,
            provider="ant",
            codegen_ir="poly",
            poly_lowering="linear_transform",
        )
        analyzed = pipeline.run_ckks_driver()
        assert analyzed["success"], analyzed
        lowered = pipeline.run_poly_driver()
        assert lowered["success"], lowered
        post = glob.dump().lower()
        assert "ckks.linear_transform" not in post, post
        assert "ckks.rotate_batch" not in post, post
        # The LT-only CPU path deliberately preserves whole-QP arithmetic to
        # the final C boundary instead of expanding it into per-prime Hw calls.
        assert "poly.add_ext" in post, post
        assert "poly.mul_ext" in post, post
        assert "poly.mac_ext" in post, post
        assert "poly.precomp" in post, post
        assert "poly.dot_prod" in post, post
        assert "poly.mod_down" in post, post
        c_code = pipeline.run_poly2c()
        assert c_code is not None
        emitted = c_code.lower()
        assert "linear_transform" not in emitted, c_code
        assert "rotate_batch_ciph" not in emitted, c_code
        assert "eval_bootstrap" not in emitted, c_code
        assert "add_poly" in emitted, c_code
        assert "mul_poly" in emitted, c_code
        assert "mac_poly" in emitted, c_code
        assert "rotate_poly_with_cached_rotation_idx" in emitted, c_code
        assert "#pragma omp parallel sections" in emitted, c_code
        assert "#pragma omp section" in emitted, c_code
        assert "hw_modadd" not in emitted, c_code
        assert "hw_modmul" not in emitted, c_code
        assert "precomp" in emitted, c_code
        assert "mod_down" in emitted, c_code
        assert emitted.count("alloc_polys(") == 2, c_code
        assert emitted.count("free_polys(") == 2, c_code
        print("CKKS_LINEAR_TRANSFORM_LPOLY_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS_LINEAR_TRANSFORM_LPOLY_OK" in result.stdout


def test_default_spoly_rejects_unlowered_linear_transform_without_mutation():
    result = _run_isolated(
        r'''
        from ace_bindings import air_builder
        from ace_edsl.edsl import (
            AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
        )

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def transform_kernel(a: CkksCiphertext) -> CkksCiphertext:
            return a._linear_transform(
                [0j] * 8,
                rot_in=[0, 1],
                rot_out=[0, 2],
                slots=4,
                term_count=2,
            )

        transform_kernel(CkksCiphertext(shape=(4,), name="value"))
        glob = AceEDSL._get_dsl().current_air_module
        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16,
            mul_level=4,
            input_level=4,
            security_level=0,
            scaling_factor_bits=20,
            first_prime_bits=60,
            hamming_weight=8,
            provider="ant",
            codegen_ir="poly",
        )
        assert pipeline.run_ckks_driver()["success"]
        before = (glob.get_native_ptr(), glob.dump())
        lowered = air_builder.run_poly_driver(glob)
        after = (glob.get_native_ptr(), glob.dump())
        assert not lowered["success"], lowered
        assert "mode 'spoly' cannot lower ckks.linear_transform" in (
            lowered["message"].lower()
        ), lowered
        assert "poly_lowering='linear_transform'" in lowered["message"], lowered
        assert after == before
        print("CKKS_LINEAR_TRANSFORM_SPOLY_GUARD_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS_LINEAR_TRANSFORM_SPOLY_GUARD_OK" in result.stdout
