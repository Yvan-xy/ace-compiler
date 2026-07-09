#!/usr/bin/env python3
"""Test Python list handling and AIR-array conversion in ace_edsl."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
)

from ace_edsl.edsl import (
    AceEDSL,
    AcePipeline,
    CkksCiphertext,
    CkksPlaintext,
    ckks_kernel,
    range_dynamic,
)


@ckks_kernel
def ckks_air_array_literal(
    ct: CkksCiphertext,
    zero: CkksCiphertext,
) -> CkksCiphertext:
    d0 = ct.rotate(1)
    d1 = ct.rotate(2)
    arr = [d0, d1].to_air()
    return arr[0] + arr[1]


def diagonal_transform(x, diagonals):
    acc = x.rotate(0) * diagonals[0]
    for r, plain in enumerate(diagonals[1:], start=1):
        acc = acc + x.rotate(r) * plain
    return acc


@ckks_kernel
def ckks_python_list_diagonal_transform(
    x: CkksCiphertext,
    d0: CkksPlaintext,
    d1: CkksPlaintext,
    d2: CkksPlaintext,
    d3: CkksPlaintext,
) -> CkksCiphertext:
    return diagonal_transform(x, [d0, d1, d2, d3])


@ckks_kernel
def diagonal_transform_dynamic_loop(x, diag, n):
    diag = diag.to_air()
    rotated = [
        x.rotate(0),
        x.rotate(1),
        x.rotate(2),
        x.rotate(3),
    ].to_air()
    acc = rotated[0] * diag[0]
    for i in range_dynamic(1, n):
        acc = acc + rotated[i] * diag[i]
    return acc


@ckks_kernel
def ckks_air_array_dynamic_loop_diagonal_transform(
    x: CkksCiphertext,
    d0: CkksPlaintext,
    d1: CkksPlaintext,
    d2: CkksPlaintext,
    d3: CkksPlaintext,
) -> CkksCiphertext:
    diag = [d0, d1, d2, d3]
    return diagonal_transform_dynamic_loop(x, diag, len(diag))


class TestAirArray(unittest.TestCase):
    def test_list_literal_to_air_array(self):
        AceEDSL._get_dsl.cache_clear()
        dsl = AceEDSL._get_dsl()

        ct = CkksCiphertext(shape=(16384,), name="ct_input")
        zero = CkksCiphertext(shape=(16384,), name="ct_zero")

        ckks_air_array_literal(ct, zero)

        glob = dsl.current_air_module
        self.assertIsNotNone(glob)
        air = glob.dump()

        self.assertIn("array", air)
        self.assertIn("ist", air.lower())
        self.assertIn("ild", air.lower())
        self.assertEqual(air.count("CKKS.rotate"), 2)
        self.assertIn("CKKS.add", air)

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16384,
            mul_level=5,
            data_file="",
        )
        ckks_result = pipeline.run_ckks_driver()
        self.assertTrue(ckks_result.get("success", False), ckks_result)

        poly_result = pipeline.run_poly_driver()
        self.assertTrue(poly_result.get("success", False), poly_result)

        c_code = pipeline.run_poly2c()
        self.assertTrue(c_code)
        self.assertIn("Copy_ciph", c_code)
        self.assertIn("[0]", c_code)
        self.assertIn("[1]", c_code)

    def test_python_list_diagonal_transform(self):
        AceEDSL._get_dsl.cache_clear()
        dsl = AceEDSL._get_dsl()

        ct = CkksCiphertext(shape=(16384,), name="ct_input")
        plain = CkksPlaintext(shape=(16384,), name="diag")

        ckks_python_list_diagonal_transform(ct, plain, plain, plain, plain)

        glob = dsl.current_air_module
        self.assertIsNotNone(glob)
        air = glob.dump()

        self.assertEqual(air.count("CKKS.rotate"), 4)
        self.assertEqual(air.count("CKKS.mul"), 4)
        self.assertEqual(air.count("CKKS.add"), 3)
        self.assertIn("PLAINTEXT", air)

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16384,
            mul_level=5,
            data_file="",
        )
        ckks_result = pipeline.run_ckks_driver()
        self.assertTrue(ckks_result.get("success", False), ckks_result)

        poly_result = pipeline.run_poly_driver()
        self.assertTrue(poly_result.get("success", False), poly_result)

        c_code = pipeline.run_poly2c()
        self.assertTrue(c_code)
        self.assertIn("Rotate(", c_code)
        self.assertIn("Hw_modmul", c_code)
        self.assertIn("Hw_modadd", c_code)

    def test_air_array_dynamic_loop_diagonal_transform(self):
        AceEDSL._get_dsl.cache_clear()
        dsl = AceEDSL._get_dsl()

        ct = CkksCiphertext(shape=(16384,), name="ct_input")
        plain = CkksPlaintext(shape=(16384,), name="diag")

        ckks_air_array_dynamic_loop_diagonal_transform(
            ct, plain, plain, plain, plain
        )

        glob = dsl.current_air_module
        self.assertIsNotNone(glob)
        air = glob.dump()

        self.assertEqual(air.count("CKKS.rotate"), 4)
        self.assertEqual(air.count("CKKS.mul"), 2)
        self.assertEqual(air.count("CKKS.add"), 1)
        self.assertIn("do_loop", air)
        self.assertIn("array", air)
        self.assertIn("ild", air.lower())
        self.assertIn("PLAINTEXT", air)

        pipeline = AcePipeline(glob).configure_fhe(
            poly_degree=16384,
            mul_level=5,
            data_file="",
        )
        ckks_result = pipeline.run_ckks_driver()
        self.assertTrue(ckks_result.get("success", False), ckks_result)

        poly_result = pipeline.run_poly_driver()
        self.assertTrue(poly_result.get("success", False), poly_result)

        c_code = pipeline.run_poly2c()
        self.assertTrue(c_code)
        self.assertIn("for", c_code)
        self.assertIn("[", c_code)


if __name__ == "__main__":
    unittest.main()
