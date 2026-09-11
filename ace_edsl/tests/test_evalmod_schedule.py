"""Compiler-owned EvalMod regions and execution policy configuration."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest
from ace_edsl.edsl import FHEConfig


def test_policy_rejects_invalid_and_conflicting_configs():
    for mode in [True, 1.0, -1, 3]:
        with pytest.raises(ValueError):
            FHEConfig(evalmod_schedule=mode, poly_lowering='linear_transform')
    with pytest.raises(ValueError):
        FHEConfig(evalmod_schedule=1, decomp_ntt_threads=2, poly_lowering='linear_transform')
    with pytest.raises(ValueError):
        FHEConfig(evalmod_schedule=1, provider='phantom',codegen_ir='ckks')


@pytest.mark.parametrize('mode',[0,1,2])
def test_only_tagged_region_uses_new_calls(tmp_path,mode):
    source=textwrap.dedent(f'''
        from ace_edsl.edsl import AceEDSL,AcePipeline,CkksCiphertext,ckks_kernel
        @ckks_kernel
        def kernel(a:CkksCiphertext,b:CkksCiphertext)->CkksCiphertext:
            before=a*a
            a.container.new_ckks_parallel_sections_begin(cpu_evalmod=True)
            a.container.new_ckks_parallel_section_begin()
            r=before*before
            a.container.new_ckks_parallel_section_end()
            a.container.new_ckks_parallel_section_begin()
            i=b*b
            a.container.new_ckks_parallel_section_end()
            a.container.new_ckks_parallel_sections_end()
            return (r+i)*a
        kernel(CkksCiphertext(shape=(8,),name='a'),CkksCiphertext(shape=(8,),name='b'))
        glob=AceEDSL._get_dsl().current_air_module
        p=AcePipeline(glob).configure_fhe(poly_degree=16,mul_level=12,input_level=12,
            security_level=0,first_prime_bits=50,scaling_factor_bits=40,hamming_weight=8,
            data_file='',poly_lowering='linear_transform',evalmod_schedule={mode})
        assert p.run_ckks_driver()['success']
        assert p.run_poly_driver()['success']
        from pathlib import Path
        Path('post.air').write_text(glob.dump())
        code=p.run_poly2c(); assert code
        Path('result.c').write_text(code)
    ''')
    script=tmp_path/'test.py';script.write_text(source)
    result=subprocess.run([sys.executable,str(script)],cwd=tmp_path,env=os.environ.copy(),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    code=(tmp_path/'result.c').read_text()
    if not mode:
        assert 'Evalmod_decomp(' not in code
        assert '#pragma omp parallel sections' in code
    else:
        assert 'cpu_evalmod' in (tmp_path/'post.air').read_text()
        begin=code.index('EVALMOD_EXEC __ace_evalmod_exec')
        end=code.index('Evalmod_exec_report(')
        assert 'Evalmod_decomp(' not in code[:begin]+code[end:]
        assert 'Decomp_modup(' in code[:begin] and 'Decomp_modup(' in code[end:]
        middle=code[begin:end]
        assert middle.count('Evalmod_decomp(')==2
        assert 'Evalmod_mod_down(' in middle and 'Evalmod_hw_modmul(' in middle
        assert ('#pragma omp task default(shared)' in middle)==(mode==2)
