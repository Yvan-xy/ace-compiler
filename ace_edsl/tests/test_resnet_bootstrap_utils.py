"""Regression tests for resnet bootstrap integration helpers."""

import argparse
from dataclasses import replace
import os
import sys
import tempfile
import unittest
from unittest import mock


def _setup_sys_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(repo_root, ".."))
    examples_dir = os.path.join(repo_root, "examples")
    for path in (repo_root, parent_root, examples_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_sys_path()

from ace_edsl.edsl.core import bootstrap_decomposition as bootstrap_decomposition
import bootstrap_full
import resnet_bootstrap_utils


def _bootstrap_config(slots: int = 32768, mul_level: int = 30):
    return bootstrap_decomposition.BootstrapConfig(
        poly_degree=slots * 2,
        mul_level=mul_level,
        first_prime_bits=60,
        scaling_factor_bits=56,
        hamming_weight=192,
        q_parts=3,
        enc_budget=3,
        dec_budget=3,
        ct_encode=True,
        eval_sin_upper_bound_k=bootstrap_full.EVAL_SIN_UPPER_BOUND_K,
        chebyshev_coefficients=tuple(bootstrap_full.G_COEFFICIENTS_UNIFORM_HW_192),
        double_angle_scalars=tuple(
            bootstrap_full.get_double_angle_scalars(bootstrap_full.NUM_DOUBLE_ANGLE)
        ),
    )


class TestBootstrapStagePlanning(unittest.TestCase):
    def test_context_mul_level_controls_linear_transform_p_basis(self):
        config = replace(
            _bootstrap_config(slots=8),
            context_mul_level=31,
        )
        self.assertEqual(config.mul_level, 30)
        self.assertEqual(config.num_p, 11)
        with self.assertRaisesRegex(ValueError, "context_mul_level"):
            replace(config, context_mul_level=29)

    def test_linear_transform_emission_is_explicit_and_disabled_by_default(self):
        self.assertFalse(_bootstrap_config(slots=8).emit_linear_transform)
        self.assertFalse(_bootstrap_config(slots=8).parallel_eval_mod)
        with self.assertRaisesRegex(
            ValueError, "emit_linear_transform must be a bool"
        ):
            replace(_bootstrap_config(slots=8), emit_linear_transform=1)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(bootstrap_full._bootstrap_linear_transform_enabled())
        with mock.patch.dict(
            os.environ, {"ACE_BOOTSTRAP_LINEAR_TRANSFORM": "1"}, clear=True
        ):
            self.assertTrue(bootstrap_full._bootstrap_linear_transform_enabled())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(bootstrap_full._bootstrap_parallel_eval_mod_enabled())
        with mock.patch.dict(
            os.environ, {"ACE_BOOTSTRAP_PARALLEL_EVAL_MOD": "1"}, clear=True
        ):
            self.assertTrue(bootstrap_full._bootstrap_parallel_eval_mod_enabled())
        with self.assertRaisesRegex(
            ValueError, "parallel_eval_mod requires emit_linear_transform"
        ):
            replace(_bootstrap_config(slots=8), parallel_eval_mod=True)

    def test_openfhe_sparse_evalmod_profile_is_opt_in(self):
        common = dict(
            poly_degree=16,
            mul_level=10,
            first_prime_bits=60,
            scaling_factor_bits=56,
            hamming_weight=192,
            q_parts=3,
            enc_budget=3,
            dec_budget=3,
            ct_encode=True,
        )
        default = bootstrap_full.build_bootstrap_trace_config(**common)
        sparse = bootstrap_full.build_bootstrap_trace_config(
            **common, evalmod_profile="openfhe_sparse"
        )
        self.assertEqual(default.eval_sin_upper_bound_k, bootstrap_full.EVAL_SIN_UPPER_BOUND_K)
        self.assertEqual(
            len(default.chebyshev_coefficients),
            bootstrap_full.UNIFORM_COEFF_SIZE_HW_192,
        )
        self.assertEqual(sparse.eval_sin_upper_bound_k, bootstrap_full.OPENFHE_SPARSE_UPPER_BOUND_K)
        self.assertEqual(
            len(sparse.chebyshev_coefficients),
            bootstrap_full.OPENFHE_SPARSE_COEFF_SIZE,
        )
        self.assertEqual(
            sparse.chebyshev_coefficients,
            tuple(bootstrap_full.G_COEFFICIENTS_OPENFHE_SPARSE),
        )
        self.assertEqual(default.bootstrap_depth, sparse.bootstrap_depth)
        with self.assertRaisesRegex(ValueError, "unsupported bootstrap EvalMod profile"):
            bootstrap_full.build_bootstrap_trace_config(**common, evalmod_profile="bad")
        with self.assertRaisesRegex(ValueError, "requires hamming_weight=192"):
            bootstrap_full.build_bootstrap_trace_config(
                **{**common, "hamming_weight": 191}, evalmod_profile="openfhe_sparse"
            )

    def test_linear_transform_descriptor_emits_one_op_per_collapsed_stage(self):
        config = replace(
            _bootstrap_config(slots=8),
            emit_linear_transform=True,
        )

        class LinearTransformRecorder:
            def __init__(self):
                self.container = object()
                self.calls = []
                self.rescale_count = 0

            def _linear_transform(self, coefficients, **attrs):
                self.calls.append((tuple(coefficients), attrs))
                return self

            def rescale(self):
                self.rescale_count += 1
                return self

        value = LinearTransformRecorder()
        value = bootstrap_decomposition.coeffs_to_slots_primitive(
            value, config=config
        )
        value = bootstrap_decomposition.slots_to_coeffs_primitive(
            value, config=config
        )

        self.assertEqual(len(value.calls), config.enc_budget + config.dec_budget)
        self.assertEqual(value.rescale_count, len(value.calls))
        for coefficients, attrs in value.calls:
            self.assertEqual(attrs["schema_version"], 1)
            self.assertEqual(attrs["slots"], config.slots)
            self.assertEqual(
                len(coefficients), attrs["term_count"] * attrs["slots"]
            )
            self.assertTrue(
                all(
                    isinstance(coefficient, complex)
                    for coefficient in coefficients
                )
            )
            self.assertTrue(attrs["rot_in"])
            self.assertTrue(attrs["rot_out"])
            self.assertEqual(attrs["rot_out"][0], 0)
            self.assertEqual(attrs["scale_degree"], 1)
            self.assertEqual(attrs["num_p"], config.num_p)
            self.assertEqual(attrs["encode_cache"], config.ct_encode)
            self.assertTrue(
                all(
                    -config.slots // 2 <= rotation <= config.slots // 2
                    for rotation in attrs["rot_in"] + attrs["rot_out"]
                )
            )

    def test_explicit_transform_giant_step_controls_outer_rotation_count(self):
        expected_outer_rotations = {8: 42, 16: 18, 32: 6}
        expected_batch_widths = {
            8: [4, 8, 8, 8, 8, 4],
            16: [8, 16, 16, 16, 16, 8],
            32: [16, 32, 32, 32, 32, 16],
        }
        for giant_step, expected in expected_outer_rotations.items():
            config = replace(
                _bootstrap_config(slots=32768),
                transform_giant_step=giant_step,
            )
            stages = []
            for encoding in (True, False):
                _, direction_stages = (
                    bootstrap_decomposition._collapsed_fft_stage_plan(
                        config, encoding
                    )
                )
                stages.extend(direction_stages)

            self.assertEqual(len(stages), 6)
            self.assertEqual(
                {stage["giant_step"] for stage in stages}, {giant_step}
            )
            self.assertEqual(
                sum(stage["baby_step"] - 1 for stage in stages), expected
            )
            grouped_term_counts = []
            for encoding in (True, False):
                coeff, direction_stages = (
                    bootstrap_decomposition._collapsed_fft_stage_plan(
                        config, encoding
                    )
                )
                grouped_term_counts.extend(
                    len(
                        bootstrap_decomposition._group_collapsed_fft_stage_terms(
                            coeff, stage, config.slots
                        )
                    )
                    for stage in direction_stages
                )
            self.assertEqual(
                [
                    (count + stages[index]["baby_step"] - 1)
                    // stages[index]["baby_step"]
                    for index, count in enumerate(grouped_term_counts)
                ],
                expected_batch_widths[giant_step],
            )

    def test_linear_transform_defaults_to_hoisted_runtime_bsgs_plan(self):
        primitive_config = _bootstrap_config(slots=32768)
        linear_transform_config = replace(
            primitive_config,
            emit_linear_transform=True,
        )

        _, primitive_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
            primitive_config, True
        )
        _, linear_transform_stages = (
            bootstrap_decomposition._collapsed_fft_stage_plan(
                linear_transform_config, True
            )
        )

        self.assertEqual(
            {stage["giant_step"] for stage in primitive_stages}, {8}
        )
        self.assertEqual(
            {stage["giant_step"] for stage in linear_transform_stages}, {16}
        )

    def test_transform_giant_step_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            replace(_bootstrap_config(), transform_giant_step=-1)
        with self.assertRaisesRegex(ValueError, "must not exceed slots"):
            replace(_bootstrap_config(), transform_giant_step=32769)
        with self.assertRaisesRegex(ValueError, "power of two"):
            replace(_bootstrap_config(), transform_giant_step=12)
        with self.assertRaisesRegex(ValueError, "exceeds stage rotation count"):
            bootstrap_decomposition._collapsed_fft_stage_plan(
                replace(_bootstrap_config(), transform_giant_step=64), True
            )

    def test_collapsed_fft_stage_levels_match_rtlib_formulas(self):
        config = _bootstrap_config(slots=32768, mul_level=30)
        _, enc_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
            config, True
        )
        _, dec_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
            config, False
        )

        self.assertEqual(
            [stage["plain_level"] for stage in enc_stages],
            [31, 30, 29],
        )
        self.assertEqual(
            [stage["plain_level"] for stage in dec_stages],
            [19, 18, 17],
        )

    def test_collapsed_fft_grouping_reduces_duplicate_rotations(self):
        config = _bootstrap_config(slots=32768)
        coeff, enc_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
            config, True
        )
        grouped = [
            bootstrap_decomposition._group_collapsed_fft_stage_terms(
                coeff, stage, 32768
            )
            for stage in enc_stages
        ]

        self.assertEqual(enc_stages[0]["num_rot"], 63)
        self.assertEqual(len(grouped[0]), 32)
        self.assertEqual(enc_stages[1]["num_rot"], 63)
        self.assertEqual(len(grouped[1]), 63)
        self.assertEqual(enc_stages[2]["num_rot"], 63)
        self.assertEqual(len(grouped[2]), 63)

    def test_no_cross_stage_duplicate_plain_diags_after_grouping(self):
        for slots in (8192, 32768):
            config = _bootstrap_config(slots=slots)
            for encoding in (True, False):
                coeff, stages = bootstrap_decomposition._collapsed_fft_stage_plan(
                    config, encoding
                )
                keys = []
                for stage in stages:
                    for _, diag in bootstrap_decomposition._group_collapsed_fft_stage_terms(
                        coeff, stage, slots
                    ):
                        keys.append(
                            (
                                stage["plain_level"],
                                tuple((float(v.real), float(v.imag)) for v in diag),
                            )
                        )
                self.assertEqual(len(keys), len(set(keys)))


class TestResnetBootstrapUtils(unittest.TestCase):
    def test_emit_body_copies_direct_codegen_output(self):
        bootstrap_c = """\
CKKS_PARAMS* Get_context_params() {
  return 0;
}
RT_DATA_INFO* Get_rt_data_info() {
  return 0;
}
CIPHERTEXT Rotate(CIPHERTEXT ciph_0, int32_t rot_idx_1) {
  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);
  if (rot_idx_1 == 0) {
    Copy_ciphertext(&_pgen_rot_res_2, &ciph_0);
    RTLIB_TM_END(20, rtm);
    return _pgen_rot_res_2;
  }
  return _pgen_rot_res_2;
}
CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1) {
  _cst_7 = 0;
  Raise_mod(&p0, &p1, 31);
  Rotate(p0, 1);
  Relinearize(&p0, &p1);
  return p0;
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "bootstrap.c")
            dst = os.path.join(tmpdir, "body.c")
            with open(src, "w", encoding="utf-8") as f:
                f.write(bootstrap_c)

            args = argparse.Namespace(
                bootstrap_c=src,
                output=dst,
                ctxparams_name="Get_extra_context_params",
                entry_name="dsl_bootstrap_full",
                rtdata_name="dsl_bootstrap_get_rt_data_info",
                pt_from_msg_name="dsl_bts_Pt_from_msg",
                rotate_name="dsl_bts_Rotate",
                relin_name="dsl_bts_Relinearize",
                raise_level_name="dsl_bts_raise_level",
                const_prefix="dsl_bts",
            )
            self.assertEqual(resnet_bootstrap_utils.emit_body(args), 0)

            body = open(dst, "r", encoding="utf-8").read()
            self.assertEqual(body, bootstrap_c)
            self.assertIn("Get_context_params()", body)
            self.assertIn("Get_rt_data_info()", body)
            self.assertIn("bootstrap_full", body)
            self.assertIn("Raise_mod(&p0, &p1, 31)", body)
            self.assertIn("if (rot_idx_1 == 0)", body)

    def test_emit_body_allows_stage_probe_without_text_injection(self):
        bootstrap_c = """\
CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1) {
  Conjugate_ciph(&p0, &p1);
  return p0;
}
"""

        old = os.environ.get("ACE_BOOTSTRAP_STAGE_PROBE")
        try:
            os.environ["ACE_BOOTSTRAP_STAGE_PROBE"] = "1"
            with tempfile.TemporaryDirectory() as tmpdir:
                src = os.path.join(tmpdir, "bootstrap.c")
                dst = os.path.join(tmpdir, "body.c")
                with open(src, "w", encoding="utf-8") as f:
                    f.write(bootstrap_c)

                args = argparse.Namespace(
                    bootstrap_c=src,
                    output=dst,
                    ctxparams_name="Get_extra_context_params",
                    entry_name="dsl_bootstrap_full",
                    rtdata_name="dsl_bootstrap_get_rt_data_info",
                    pt_from_msg_name="dsl_bts_Pt_from_msg",
                    rotate_name="dsl_bts_Rotate",
                    relin_name="dsl_bts_Relinearize",
                    raise_level_name="dsl_bts_raise_level",
                    const_prefix="dsl_bts",
                )
                self.assertEqual(resnet_bootstrap_utils.emit_body(args), 0)

                body = open(dst, "r", encoding="utf-8").read()
                self.assertEqual(body, bootstrap_c)
                self.assertNotIn("dsl_bts_stage_mark", body)
        finally:
            if old is None:
                os.environ.pop("ACE_BOOTSTRAP_STAGE_PROBE", None)
            else:
                os.environ["ACE_BOOTSTRAP_STAGE_PROBE"] = old

    def test_stage_probe_prologue_redirects_structured_helpers(self):
        old = os.environ.get("ACE_BOOTSTRAP_STAGE_PROBE")
        try:
            os.environ["ACE_BOOTSTRAP_STAGE_PROBE"] = "1"
            c_code = """\
// external header files
#include "rt_ant/rt_ant.h"

CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1) {
  Conjugate_ciph(&p0, &p1);
  Mul_mono_ciph(&p0, &p1, 4);
  Init_ciph_same_scale(&p0, &p1, &p1);
  return p0;
}
"""
            wrapped = bootstrap_full._with_stage_probe_prologue(c_code)
            self.assertIn(
                "#define Conjugate_ciph dsl_bts_probe_Conjugate_ciph",
                wrapped,
            )
            self.assertIn(
                "#define Mul_mono_ciph dsl_bts_probe_Mul_mono_ciph",
                wrapped,
            )
            self.assertIn(
                "#define Init_ciph_same_scale "
                "dsl_bts_probe_Init_ciph_same_scale",
                wrapped,
            )
            self.assertIn(
                "ACE_DSL_BTS_STAGE_PROBE_WEAK CIPHER "
                "dsl_bts_probe_Conjugate_ciph",
                wrapped,
            )
            self.assertIn("CIPHERTEXT bootstrap_full", wrapped)
        finally:
            if old is None:
                os.environ.pop("ACE_BOOTSTRAP_STAGE_PROBE", None)
            else:
                os.environ["ACE_BOOTSTRAP_STAGE_PROBE"] = old

    def test_emit_shim_exposes_extra_context_bridge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "shim.c")
            args = argparse.Namespace(
                output=dst,
                entry_name="dsl_bts_bootstrap_full",
                ctxparams_name="dsl_bts_Get_context_params",
                rtdata_name="dsl_bts_Get_rt_data_info",
                pt_from_msg_name="dsl_bts_Pt_from_msg",
                raise_level_name="dsl_bts_raise_level",
                bootstrap_depth=15,
                bootstrap_call_name="Eval_bootstrap_ciph_dsl",
                log_prefix="dsl_bts",
                dump_label="dsl_bts_round1",
                target_level_setter_name="ace_cpu_bootstrap_set_target_level",
            )
            self.assertEqual(resnet_bootstrap_utils.emit_shim(args), 0)

            shim = open(dst, "r", encoding="utf-8").read()
            self.assertIn("CKKS_PARAMS* Get_extra_context_params(void)", shim)
            self.assertIn("return dsl_bts_Get_context_params();", shim)
            self.assertIn("dsl_bts_raise_level(void)", shim)
            self.assertIn("dsl_bts_bootstrap_full(in_copy, in_copy)", shim)
            self.assertIn(
                "void ace_cpu_bootstrap_set_target_level(uint32_t target_level)",
                shim,
            )
            self.assertIn(
                "g_dsl_bts_target_level_after_bts = target_level;", shim
            )

    def test_emit_shim_exposes_stage_probe_wrappers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dst = os.path.join(tmpdir, "shim.c")
            args = argparse.Namespace(
                output=dst,
                entry_name="dsl_bts_bootstrap_full",
                ctxparams_name="dsl_bts_Get_context_params",
                rtdata_name="dsl_bts_Get_rt_data_info",
                pt_from_msg_name="dsl_bts_Pt_from_msg",
                raise_level_name="dsl_bts_raise_level",
                bootstrap_depth=15,
                bootstrap_call_name="Eval_bootstrap_ciph_dsl",
                log_prefix="dsl_bts",
                dump_label="dsl_bts_round1",
            )
            self.assertEqual(resnet_bootstrap_utils.emit_shim(args), 0)

            shim = open(dst, "r", encoding="utf-8").read()
            self.assertIn("dsl_bts_stage_probe_begin();", shim)
            self.assertIn("dsl_bts_stage_probe_finish();", shim)
            self.assertIn("ace_cpu_bootstrap_stage_probe_begin", shim)
            self.assertIn("ace_cpu_bootstrap_stage_probe_finish", shim)
            self.assertIn("dsl_bts_probe_Conjugate_ciph", shim)
            self.assertIn("dsl_bts_probe_Mul_mono_ciph", shim)
            self.assertIn("dsl_bts_probe_Init_ciph_same_scale", shim)
            self.assertIn('"coeff_to_slots"', shim)
            self.assertIn('"dual_evalmod"', shim)
            self.assertIn('"slots_to_coeffs"', shim)


if __name__ == "__main__":
    unittest.main()
