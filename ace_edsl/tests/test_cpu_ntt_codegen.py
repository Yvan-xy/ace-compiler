"""CPU NTT budget validation and actual POLY codegen integration."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from ace_edsl.edsl import AcePipeline, FHEConfig, Pipeline


def test_ntt_budget_validation():
    for value in (0, 1, 16):
        assert FHEConfig(decomp_ntt_threads=value).decomp_ntt_threads == value
    for value in (-1, True, 2.5, 1 << 32):
        with pytest.raises(ValueError, match="uint32"):
            FHEConfig(decomp_ntt_threads=value)
    for provider in ("ant", "phantom"):
        with pytest.raises(ValueError, match="ANT POLY"):
            FHEConfig(provider=provider, codegen_ir="ckks", decomp_ntt_threads=2)


def test_both_pipelines_forward_ntt_budget():
    class Glob:
        def run_poly2c(self, **kwargs):
            self.kwargs = kwargs
            return True
        def get_c_code(self):
            return "code"
    a, b = Glob(), Glob()
    first = AcePipeline(a).configure_fhe(decomp_ntt_threads=3)
    assert first.run_poly2c() == "code"
    second = Pipeline(dump_ir=False, verbose=False).set_glob(b)
    second.configure_fhe(decomp_ntt_threads=3)
    assert second._run_phase("poly2c")
    assert a.kwargs["decomp_ntt_threads"] == b.kwargs["decomp_ntt_threads"] == 3


def test_bootstrap_experiment_disables_outer_sections(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import bootstrap_full
    monkeypatch.setenv("ACE_BOOTSTRAP_PARALLEL_EVAL_MOD", "1")
    monkeypatch.setenv("ACE_BOOTSTRAP_DECOMP_NTT_THREADS", "0")
    assert bootstrap_full._bootstrap_parallel_eval_mod_enabled()
    for value in ("1", "16"):
        monkeypatch.setenv("ACE_BOOTSTRAP_DECOMP_NTT_THREADS", value)
        assert not bootstrap_full._bootstrap_parallel_eval_mod_enabled()


@pytest.mark.parametrize("threads", [0, 1, 3])
def test_small_square_uses_selected_runtime_entry(tmp_path, threads):
    # Independent process avoids persistent tracing state between codegen runs.
    script = textwrap.dedent(f"""
        from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext, ckks_kernel
        @ckks_kernel
        def square(a: CkksCiphertext) -> CkksCiphertext:
            return a * a
        square(CkksCiphertext(shape=(8,), name='a'))
        glob = AceEDSL._get_dsl().current_air_module
        p = AcePipeline(glob).configure_fhe(
            poly_degree=16, mul_level=5, input_level=5, security_level=0,
            first_prime_bits=50, scaling_factor_bits=40, hamming_weight=8,
            data_file='', decomp_ntt_threads={threads})
        result = p.run_ckks_driver()
        assert result['success'], result
        result = p.run_poly_driver()
        assert result['success'], result
        code = p.run_poly2c()
        assert code is not None
        from pathlib import Path
        Path('square.c').write_text(code)
    """)
    source = tmp_path / "square.py"
    source.write_text(script)
    result = subprocess.run([sys.executable, str(source)], cwd=tmp_path,
                            env=os.environ.copy(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    code = (tmp_path / "square.c").read_text()
    if threads == 0:
        assert "Decomp_modup(" in code
        assert "Decomp_modup_with_ntt_threads(" not in code
    else:
        calls = [line for line in code.splitlines() if "Decomp_modup_with_ntt_threads(" in line]
        assert calls, code
        assert all(line.rstrip().endswith(f", {threads});") for line in calls)
        assert "Decomp_modup(" not in code
