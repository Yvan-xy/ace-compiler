"""Regression tests for resnet bootstrap integration helpers."""

import argparse
import os
import sys
import tempfile
import unittest


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
            )
            self.assertEqual(resnet_bootstrap_utils.emit_shim(args), 0)

            shim = open(dst, "r", encoding="utf-8").read()
            self.assertIn("CKKS_PARAMS* Get_extra_context_params(void)", shim)
            self.assertIn("return dsl_bts_Get_context_params();", shim)
            self.assertIn("dsl_bts_raise_level(void)", shim)
            self.assertIn("dsl_bts_bootstrap_full(in_copy, in_copy)", shim)

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
            self.assertIn("dsl_bts_probe_Conjugate_ciph", shim)
            self.assertIn("dsl_bts_probe_Mul_mono_ciph", shim)
            self.assertIn("dsl_bts_probe_Init_ciph_same_scale", shim)
            self.assertIn('"coeff_to_slots"', shim)
            self.assertIn('"dual_evalmod"', shim)
            self.assertIn('"slots_to_coeffs"', shim)


if __name__ == "__main__":
    unittest.main()
