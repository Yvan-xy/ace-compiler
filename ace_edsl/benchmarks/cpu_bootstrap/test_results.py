"""Ensure failed or mismatched runs cannot produce a speedup claim."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from run import summarize


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        state = dict(status="pass", ring_dimension=65536, slots=32768,
                     context_q=31, p_count=11, raised_q=31, input_q=2,
                     input_scale_degree=1, output_q=14, output_scale_degree=1,
                     output_scale=2.0 ** 56, raw_output_q=16,
                     raw_output_scale_degree=2, threads=16, maximum_error=0.001)
        self.rows = {}
        for impl, times in (("dsl_cpu", [100, 2, 4, 3]), ("openfhe", [200, 6, 8, 7])):
            self.rows[impl] = [dict(state, implementation=impl, sample=i,
                                   timed=i > 0, seconds=t) for i, t in enumerate(times)]

    def evaluate(self):
        for impl, rows in self.rows.items():
            (self.work / f"{impl}.log").write_text("\n".join(
                "CPU_BTS_RESULT=" + json.dumps(row) for row in rows))
        with contextlib.redirect_stdout(io.StringIO()):
            summarize(self.work, 3)

    def test_warmup_excluded_and_median_used(self):
        self.evaluate()
        result = json.loads((self.work / "summary.json").read_text())
        self.assertEqual(result["dsl_cpu"]["median_seconds"], 3)
        self.assertEqual(result["openfhe"]["median_seconds"], 7)
        self.assertAlmostEqual(result["openfhe_over_dsl"], 7 / 3)

    def test_output_mismatch_rejected(self):
        self.rows["dsl_cpu"][1]["output_q"] = 13
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_missing_sample_rejected(self):
        self.rows["openfhe"].pop()
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_nonfinite_or_inaccurate_result_rejected(self):
        good = copy.deepcopy(self.rows)
        for field, value in (("seconds", float("nan")), ("maximum_error", 0.03),
                             ("output_scale", 2.0 ** 55)):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(good)
                self.rows["openfhe"][1][field] = value
                with self.assertRaises(ValueError):
                    self.evaluate()


if __name__ == "__main__":
    unittest.main()
