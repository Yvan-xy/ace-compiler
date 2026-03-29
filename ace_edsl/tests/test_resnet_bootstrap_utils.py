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


class TestResnetBootstrapUtils(unittest.TestCase):
    def test_patch_resnet_context_adds_rotations_and_zero_fast_path(self):
        bootstrap_c = """\
static CKKS_PARAMS parm = {
    LIB_ANT, 65536, 0, 29, 1, 60, 56, 3, 192, 0,
    { }
};
CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1) {
  Rotate(p0, 1024);
  Rotate(p0, 31744);
  Conjugate_ciph(&p0, &p0);
  return p0;
}
"""
        resnet_inc = """\
static CKKS_PARAMS parm = {
    LIB_ANT, 65536, 0, 30, 17, 60, 56, 3, 192, 2,
    { 7, 8 }
};
CIPHERTEXT Rotate(CIPHERTEXT ciph_0, int32_t rot_idx_1) {
  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);
  return _pgen_rot_res_2;
}
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            bootstrap_path = os.path.join(tmpdir, "bootstrap.c")
            resnet_path = os.path.join(tmpdir, "resnet.inc")
            with open(bootstrap_path, "w", encoding="utf-8") as f:
                f.write(bootstrap_c)
            with open(resnet_path, "w", encoding="utf-8") as f:
                f.write(resnet_inc)

            args = argparse.Namespace(bootstrap_c=bootstrap_path, resnet_inc=resnet_path)
            self.assertEqual(resnet_bootstrap_utils.patch_resnet_context(args), 0)

            patched = open(resnet_path, "r", encoding="utf-8").read()
            self.assertIn("1024", patched)
            self.assertIn("31744", patched)
            self.assertIn(str(2 * 65536 - 1), patched)
            self.assertIn("if (rot_idx_1 == 0)", patched)

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
  return _pgen_rot_res_2;
}
CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1) {
  _cst_7 = 0;
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
                entry_name="dsl_bootstrap_full",
                rotate_name="dsl_bts_Rotate",
                relin_name="dsl_bts_Relinearize",
                const_prefix="dsl_bts",
            )
            self.assertEqual(resnet_bootstrap_utils.emit_body(args), 0)

            body = open(dst, "r", encoding="utf-8").read()
            self.assertNotIn("Get_context_params()", body)
            self.assertNotIn("Get_rt_data_info()", body)
            self.assertIn("dsl_bootstrap_full", body)
            self.assertIn("dsl_bts_Rotate", body)
            self.assertIn("dsl_bts_Relinearize", body)
            self.assertIn("dsl_bts_cst_7", body)
            self.assertIn("if (rot_idx_1 == 0)", body)


if __name__ == "__main__":
    unittest.main()
