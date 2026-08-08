"""Regressions for the dedicated post-CKKS AIR source emitter."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "ace_edsl", "examples")


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((REPO_ROOT, EXAMPLES_DIR))
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bootstrap_scalar_constants_request_dynamic_encode_level(monkeypatch):
    from ace_edsl.edsl.core import bootstrap_decomposition

    encoded = []
    encoded_value = object()

    def capture_encode(x, value, *, scale_degree, level):
        encoded.append((value, scale_degree, level))
        return encoded_value

    class Cipher:
        container = object()

        def __add__(self, rhs):
            assert rhs is encoded_value
            return "add-result"

        def __mul__(self, rhs):
            assert rhs is encoded_value
            return "mul-result"

    monkeypatch.setattr(
        bootstrap_decomposition, "_encode_scalar_like", capture_encode
    )
    cipher = Cipher()
    config = object()

    assert (
        bootstrap_decomposition._add_const_like(cipher, -1.0, config)
        == "add-result"
    )
    assert (
        bootstrap_decomposition._mul_const_like(cipher, 0.5, config)
        == "mul-result"
    )
    assert encoded == [(-1.0, 1, 0), (0.5, 1, 0)]


def test_phantom_add_mul_rotate_uses_only_dedicated_ckks2c():
    result = _run_isolated(
        r'''
        from ace_edsl.edsl import (
            AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
        )

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def arithmetic_kernel(
            a: CkksCiphertext, b: CkksCiphertext
        ) -> CkksCiphertext:
            return (a * b) + a.rotate(3)

        a = CkksCiphertext(shape=(16384,), name="a")
        b = CkksCiphertext(shape=(16384,), name="b")
        arithmetic_kernel(a, b)
        glob = AceEDSL._get_dsl().current_air_module
        raw_air = glob.dump().lower()
        assert "fhe::poly" not in raw_air

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16384,
            mul_level=4,
            input_level=1,
            security_level=0,
            scaling_factor_bits=56,
            first_prime_bits=60,
            hamming_weight=192,
            data_file="",
            provider="phantom",
            codegen_ir="ckks",
        )
        compiled = pipeline.run(
            start_domain="fhe::ckks", dump_stages=True, verbose=False
        )
        assert compiled.success, compiled.error
        assert compiled.stages_completed == ["ckks_driver", "ckks2c"]
        post_air = compiled.air_dumps["ckks_driver"].lower()
        assert "fhe::poly" not in post_air

        source = compiled.c_code or ""
        required = (
            '#include "rt_phantom/rt_phantom.h"',
            "Get_phantom_context_manifest()",
            "Get_phantom_resource_manifest()",
            "Add_ciph(",
            "Mul_ciph(",
            "Rotate_ciph(",
        )
        forbidden = (
            "rt_ant/", "rt_seal/", "LIB_ANT", "LIB_SEAL",
            "Hw_", "Poly_", "Bootstrap(", "Eval_bootstrap",
            "bootstrap_coeffs_to_slots", "bootstrap_eval_mod",
            "bootstrap_slots_to_coeffs", "phantom::",
            "Need_bts(", "CKKS_PARAMS",
        )
        for token in required:
            assert token in source, token
        for token in forbidden:
            assert token not in source, token
        print("CKKS2C_ARITHMETIC_SOURCE_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS2C_ARITHMETIC_SOURCE_OK" in result.stdout


def test_full_primitive_bootstrap_preserves_retained_ops_through_ckks2c():
    result = _run_isolated(
        r'''
        import os
        import re
        os.environ["ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING"] = "1"
        from bootstrap_full import (
            G_COEFFICIENTS_UNIFORM_HW_192,
            _bootstrap_trace_config,
            bootstrap_full,
        )
        from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext

        AceEDSL._get_dsl.cache_clear()
        config = _bootstrap_trace_config()
        ct = CkksCiphertext(shape=(config.poly_degree,), name="input_ct")
        zero = CkksCiphertext(shape=(config.poly_degree,), name="zero_ct")
        args = (
            [ct, zero, 1.0]
            + list(G_COEFFICIENTS_UNIFORM_HW_192)
            + list(config.double_angle_scalars)
            + [config.post_scale]
        )
        bootstrap_full(*args)
        glob = AceEDSL._get_dsl().current_air_module
        raw_air = glob.dump().lower()

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=config.poly_degree,
            mul_level=config.mul_level,
            input_level=1,
            security_level=0,
            scaling_factor_bits=config.scaling_factor_bits,
            first_prime_bits=config.first_prime_bits,
            hamming_weight=config.hamming_weight,
            data_file="",
            provider="phantom",
            codegen_ir="ckks",
        )
        compiled = pipeline.run(
            start_domain="fhe::ckks", dump_stages=True, verbose=False
        )
        assert compiled.success, compiled.error
        post_air = compiled.air_dumps["ckks_driver"].lower()

        retained_air = (
            "ckks.conjugate", "ckks.rotate_batch",
            "ckks.raise_mod", "ckks.mul_mono",
        )
        forbidden_air = (
            "ckks.bootstrap", "ckks.bootstrap_coeffs_to_slots",
            "ckks.bootstrap_eval_mod", "ckks.bootstrap_slots_to_coeffs",
            "fhe::poly",
        )
        for air in (raw_air, post_air):
            for token in retained_air:
                assert token in air, token
            for token in forbidden_air:
                assert token not in air, token

        source = compiled.c_code or ""
        retained_calls = (
            "Conjugate_ciph(", "Rotate_batch_ciph(",
            "Raise_mod(", "Mul_mono_ciph(", "Mod_switch(",
        )
        forbidden_calls = (
            "Bootstrap(", "Eval_bootstrap", "Phantom_bootstrap",
            "bootstrap_coeffs_to_slots", "bootstrap_eval_mod",
            "bootstrap_slots_to_coeffs", "Hw_", "Poly_",
            "rt_ant/", "rt_seal/", "LIB_ANT", "LIB_SEAL", "phantom::",
        )
        for token in retained_calls:
            assert token in source, token
        assert re.search(
            r"(?P<level>_preg_[0-9]+)\s*=\s*Level\([^;]+\);\s*"
            r"Encode_(?:float|double)(?:_mask)?\([^;]+,\s*"
            r"(?P=level)\s*\);",
            source,
        )
        for token in forbidden_calls:
            assert token not in source, token
        print("CKKS2C_FULL_PRIMITIVE_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unexpected operator" not in result.stderr
    assert "CKKS2C_FULL_PRIMITIVE_OK" in result.stdout


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("ct.bootstrap()", "bootstrap"),
        ("ct.bootstrap_coeffs_to_slots()", "bootstrap_coeffs_to_slots"),
        ("ct.bootstrap_eval_mod()", "bootstrap_eval_mod"),
        ("ct.bootstrap_slots_to_coeffs()", "bootstrap_slots_to_coeffs"),
    ),
)
def test_phantom_rejects_opaque_bootstrap_and_each_stage(expression, expected):
    result = _run_isolated(
        f'''
        import os
        os.environ["ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING"] = "0"
        from ace_edsl.edsl import (
            AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel,
        )

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def rejected(ct: CkksCiphertext) -> CkksCiphertext:
            return {expression}

        rejected(CkksCiphertext(shape=(16384,), name="ct"))
        pipeline = AcePipeline(AceEDSL._get_dsl().current_air_module)
        pipeline.configure_fhe(
            poly_degree=16384, mul_level=8, input_level=1,
            security_level=0, scaling_factor_bits=56,
            first_prime_bits=60, hamming_weight=192,
            data_file="", provider="phantom", codegen_ir="ckks",
        )
        compiled = pipeline.run(start_domain="fhe::ckks", verbose=False)
        assert not compiled.success
        assert "Phantom primitive CKKS2C rejects" in compiled.error
        assert {expected!r} in compiled.error.lower()
        print("CKKS2C_REJECTION_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS2C_REJECTION_OK" in result.stdout


def test_legacy_enable_poly_false_forwards_to_ckks2c_without_source_patching():
    result = _run_isolated(
        r'''
        from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel

        AceEDSL._get_dsl.cache_clear()

        @ckks_kernel
        def legacy(ct: CkksCiphertext) -> CkksCiphertext:
            return ct.rotate(1)

        legacy(CkksCiphertext(shape=(16384,), name="ct"))
        pipeline = AcePipeline(AceEDSL._get_dsl().current_air_module)
        pipeline.configure_fhe(
            poly_degree=16384, mul_level=2, security_level=0,
            data_file="", enable_poly=False,
        )
        compiled = pipeline.run(start_domain="fhe::ckks", verbose=False)
        assert compiled.success, compiled.error
        assert compiled.stages_completed == ["ckks_driver", "ckks2c"]
        source = compiled.c_code or ""
        assert 'rt_ant/rt_ant.h' in source
        assert "LIB_ANT" in source
        assert "rt_seal/" not in source and "LIB_SEAL" not in source
        print("CKKS2C_FORWARD_OK")
        '''
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CKKS2C_FORWARD_OK" in result.stdout
