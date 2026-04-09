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
import resnet_bootstrap_utils


class TestBootstrapStagePlanning(unittest.TestCase):
    def test_collapsed_fft_stage_levels_match_rtlib_formulas(self):
        old = os.environ.get("ACE_BOOTSTRAP_MUL_LEVEL")
        try:
            os.environ["ACE_BOOTSTRAP_MUL_LEVEL"] = "30"
            _, enc_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
                32768, True
            )
            _, dec_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
                32768, False
            )
        finally:
            if old is None:
                os.environ.pop("ACE_BOOTSTRAP_MUL_LEVEL", None)
            else:
                os.environ["ACE_BOOTSTRAP_MUL_LEVEL"] = old

        self.assertEqual(
            [stage["plain_level"] for stage in enc_stages],
            [31, 30, 29],
        )
        self.assertEqual(
            [stage["plain_level"] for stage in dec_stages],
            [19, 18, 17],
        )

    def test_collapsed_fft_grouping_reduces_duplicate_rotations(self):
        coeff, enc_stages = bootstrap_decomposition._collapsed_fft_stage_plan(
            32768, True
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
            for encoding in (True, False):
                coeff, stages = bootstrap_decomposition._collapsed_fft_stage_plan(
                    slots, encoding
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
    def test_emit_body_strips_wrappers_and_renames_symbols(self):
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
            self.assertNotIn("Get_context_params()", body)
            self.assertNotIn("Get_rt_data_info()", body)
            self.assertIn("Get_extra_context_params()", body)
            self.assertIn("dsl_bootstrap_get_rt_data_info()", body)
            self.assertIn("dsl_bootstrap_full", body)
            self.assertIn("dsl_bts_Rotate", body)
            self.assertIn("dsl_bts_Relinearize", body)
            self.assertIn("dsl_bts_raise_level()", body)
            self.assertIn("dsl_bts_cst_7", body)
            self.assertIn("if (rot_idx_1 == 0)", body)


if __name__ == "__main__":
    unittest.main()
