#!/usr/bin/env python3
"""Profile copies of the matched CPU benchmark; keep baseline artifacts intact."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import statistics

from run import HERE, ROOT, run, sha, summarize, verify_coefficients


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"profiling anchor changed: {old[:100]!r}")
    return text.replace(old, new)


def mark(stage, state=None, ant=False):
    code = f'ace_profile_mark("{stage}");\n'
    if state:
        if ant:
            code += f'ace_profile_state(Level(&{state}), Sc_degree(&{state}));\n'
        else:
            code += f'ace_profile_state({state}->GetElements()[0].GetNumOfElements(), {state}->GetNoiseScaleDeg());\n'
    return code


def instrument_ant(text):
    text = '#include "profile_hooks.h"\n' + text
    # Boundaries are checked against the actual generated source, not inferred
    # from plaintext-cache misses (which disappear after warm-up).
    raise_call = re.search(r'  Raise_mod\(&([^,]+),[^;]+;', text).group(0)
    raised = re.search(r'Raise_mod\(&([^,]+),', raise_call).group(1)
    text = replace_one(text, raise_call, mark("modraise") + raise_call + '\n' + mark("coeff_to_slots", raised, True))
    conj = re.search(r'  Conjugate_ciph\([^;]+;', text).group(0)
    text = replace_one(text, conj, mark("split") + conj)
    mono = list(re.finditer(r'  Mul_mono_ciph\(&([^,]+),[^;]+;', text))
    if len(mono) != 2:
        raise ValueError("expected full-packed conjugate split/recombine")
    end = mono[0].end()
    # Both sequential and two-section EvalMod schedules are valid. The two
    # releases after the split belong to split, regardless of the next opcode.
    cleanup = re.match(r'\n(?:  Free_poly_data\([^\n]+\);\n)*', text[end:])
    section = end + cleanup.end()
    text = text[:section] + mark("evalmod", mono[0].group(1), True) + text[section:]
    second = mono[1].group(0)
    text = replace_one(text, second, mark("recombine") + second)
    after = text.index(second) + len(second)
    c2 = re.search(r'  (_preg_\d+) = Num_decomp\(&([^.]*)\._c1_poly\);', text[after:])
    start = after + c2.start()
    text = text[:start] + mark("slots_to_coeffs", c2.group(2), True) + text[start:]
    # The four final doublings implement the restoration factor 16.
    matches = list(re.finditer(r'  Init_ciph_same_scale\(&([^,]+), &([^,]+), &\2\);', text))
    final = matches[-4]
    text = text[:final.start()] + mark("postscale", final.group(2), True) + text[final.start():]
    return text


def instrument_openfhe(text):
    begin = text.index('Ciphertext<DCRTPoly> FHECKKSRNS::EvalBootstrap(')
    end = text.index('Ciphertext<DCRTPoly> FHECKKSRNS::EvalBootstrapStCFirst(', begin)
    body = text[begin:end]
    for anchor, addition in [
        ('    auto raised = ciphertext->Clone();', mark("modraise", "ciphertext")),
        ('    uint64_t corFactor =', mark("postscale", "ctxtDec")),
    ]:
        body = replace_one(body, anchor, addition + anchor)
    full_start = body.index('    if (slots == cc->GetCyclotomicOrder() / 4) {')
    full_end = body.index('\n    else {', full_start)
    full = body[full_start:full_end]
    edits = [
        ('        auto ctxtEnc =\n', mark("coeff_to_slots", "raised")),
        ('        auto& evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());', mark("split", "ctxtEnc")),
        ('        ctxtEnc  = algo->EvalChebyshevSeries(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound);', mark("evalmod", "ctxtEnc")),
        ('        algo->MultByMonomialInPlace(ctxtEncI, slots);', mark("recombine", "ctxtEnc")),
        ('        ctxtDec = (isLTBootstrap) ? EvalLinearTransform(p.m_U0Pre, ctxtEnc) : EvalSlotsToCoeffs(p.m_U0PreFFT, ctxtEnc);', mark("slots_to_coeffs", "ctxtEnc")),
    ]
    for anchor, addition in edits:
        full = replace_one(full, anchor, addition + anchor)
    body = body[:full_start] + full + body[full_end:]
    return '#include "profile_hooks.h"\n' + text[:begin] + body + text[end:]


def build(args, env):
    work, baseline = args.work_dir, args.baseline
    # Reuse the matching encoded constants; never regenerate them for profiling.
    original = baseline / "generator/output/bootstrap_full.c"
    (work / "bootstrap.c").write_text(instrument_ant(original.read_text()))
    for name in ("profile_hooks.h", "profile_hooks.c", "common.h"):
        shutil.copy2(HERE / name, work / name)
    for impl in ("ant", "openfhe"):
        text = '#include "profile_hooks.h"\n' + (HERE / f"{impl}_main.cxx").read_text()
        text = replace_one(text, '    double start = bench::Now();',
                           '    ace_profile_begin(sample);\n    double start = bench::Now();')
        text = replace_one(text, '    unsigned raw_q =', '    ace_profile_mark("normalize");\n    unsigned raw_q =')
        text = replace_one(text, '    double seconds = bench::Now() - start;',
                           '    double seconds = bench::Now() - start;\n    ace_profile_end();')
        (work / f"{impl}_main.cxx").write_text(text)
    prefix = args.ace_prefix
    flags = ['-O3', '-g', '-DNDEBUG', '-DRTLIB_SUPPORT_LINUX', '-fopenmp', '-I' + str(work)]
    flags += ['-I' + str(prefix / p) for p in ('include', 'rtlib/include', 'rtlib/include/ant')]
    for source, obj in [(work / 'bootstrap.c', 'bootstrap.o'), (work / 'profile_hooks.c', 'hooks.o')]:
        run(['gcc', '-std=gnu11', *flags, '-c', source, '-o', work / obj], work / (obj + '.log'))
    run(['g++', '-std=gnu++17', *flags, '-Wl,--export-dynamic', work / 'ant_main.cxx',
         work / 'bootstrap.o', baseline / 'shim.o', work / 'hooks.o',
         prefix / 'rtlib/lib/libFHErt_ant.a', prefix / 'rtlib/lib/libFHErt_common.a',
         prefix / 'lib/libAIRutil.a', '-lgmp', '-lm', '-o', work / 'ant_bench'], work / 'ant-build.log')
    ofsrc = work / 'openfhe-src'
    if not ofsrc.exists():
        shutil.copytree(args.openfhe_source, ofsrc, ignore=shutil.ignore_patterns('.git'))
    rel = 'src/pke/lib/scheme/ckksrns/ckksrns-fhe.cpp'
    (ofsrc / rel).write_text(instrument_openfhe((args.openfhe_source / rel).read_text()))
    # The modified DSO calls hooks exported by the profiling executable.
    install = work / 'openfhe-install'
    run(['cmake', '-S', ofsrc, '-B', work / 'openfhe-build', '-DCMAKE_BUILD_TYPE=Release',
         f'-DCMAKE_INSTALL_PREFIX={install}', f'-DCMAKE_CXX_FLAGS=-g -I{work}',
         '-DBUILD_UNITTESTS=OFF', '-DBUILD_EXAMPLES=OFF', '-DBUILD_BENCHMARKS=OFF',
         '-DWITH_OPENMP=ON', '-DWITH_NATIVEOPT=OFF'], work / 'openfhe-configure.log')
    run(['cmake', '--build', work / 'openfhe-build', '-j', '8'], work / 'openfhe-build.log')
    run(['cmake', '--install', work / 'openfhe-build'], work / 'openfhe-install.log')
    cmake = (HERE / 'CMakeLists.txt').read_text().replace('LANGUAGES CXX', 'LANGUAGES C CXX')
    cmake = cmake.replace('openfhe_main.cxx)', 'openfhe_main.cxx profile_hooks.c)')
    cmake += '\ntarget_link_options(openfhe_bench PRIVATE -Wl,--export-dynamic)\n'
    (work / 'CMakeLists.txt').write_text(cmake)
    run(['cmake', '-S', work, '-B', work / 'openfhe-bench-build', '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CXX_FLAGS=-g', f'-DCMAKE_PREFIX_PATH={install}'], work / 'harness-configure.log')
    run(['cmake', '--build', work / 'openfhe-bench-build', '-j', '4'], work / 'harness-build.log')
    (work / 'source-hashes.json').write_text(json.dumps({str(p): sha(p) for p in
        [original, ofsrc / rel, work / 'bootstrap.c', HERE / 'profile.py', HERE / 'profile_hooks.c']}, indent=2))


def report(work, repetitions, perf):
    summarize(work, repetitions)
    summary = {}
    for impl in ('dsl_cpu', 'openfhe'):
        rows = [json.loads(x.split('=', 1)[1]) for x in (work / f'{impl}.log').read_text().splitlines()
                if x.startswith('CPU_BTS_PHASE=')]
        grouped = defaultdict(list)
        expected = ['entry', 'modraise', 'coeff_to_slots', 'split', 'evalmod',
                    'recombine', 'slots_to_coeffs', 'postscale', 'normalize']
        for sample in range(repetitions + 1):
            phases = [r for r in rows if r['sample'] == sample]
            if [r['stage'] for r in phases] != expected:
                raise ValueError(f'{impl}: missing or misordered phase boundaries')
            if any(a['end'] != b['start'] for a, b in zip(phases, phases[1:])):
                raise ValueError(f'{impl}: non-contiguous phase boundaries')
        for row in rows:
            if row['sample']:
                grouped[row['stage']].append(row)
        summary[impl] = {}
        for stage, samples in grouped.items():
            if len(samples) != repetitions:
                raise ValueError(f'{impl}/{stage}: incomplete phase data')
            fields = ('seconds', 'cpu_seconds', 'system_seconds', 'minor_faults', 'major_faults', 'input_q', 'input_scale_degree')
            medians = {key: statistics.median(x[key] for x in samples) for key in fields}
            medians['effective_cpus'] = medians['cpu_seconds'] / medians['seconds']
            summary[impl][stage] = medians
        if perf:
            # Report the first measured invocation only: excludes keygen,
            # warm-up, input cloning, decryption, validation and teardown.
            sample = [r for r in rows if r['sample'] == 1]
            windows = [('all', min(r['start'] for r in sample), max(r['end'] for r in sample))]
            windows += [(r['stage'], r['start'], r['end']) for r in sample
                        if r['stage'] in ('coeff_to_slots', 'evalmod', 'slots_to_coeffs')]
            for stage, start, end in windows:
                run(['perf', 'report', '-i', work / f'{impl}.perf.data', '--stdio', '--no-children',
                     '--call-graph', 'none', '--percent-limit', '0.5', '--time', f'{start:.6f},{end:.6f}',
                     '-s', 'dso,symbol'], work / f'{impl}.{stage}.hotspots.txt')
    (work / 'phases.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ace-prefix', type=Path, required=True)
    parser.add_argument('--openfhe-source', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'tmp_openfhe_cpu_compare/benchmark')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'tmp_openfhe_cpu_compare/profile')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--perf', action='store_true')
    parser.add_argument('--run-name', default='threads16', help='subdirectory for this run and its reports')
    args = parser.parse_args()
    for name in ('ace_prefix', 'openfhe_source', 'baseline', 'work_dir'):
        setattr(args, name, getattr(args, name).resolve())
    if args.work_dir == args.baseline:
        parser.error('profiling must use a separate directory')
    verify_coefficients(args.openfhe_source)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OMP_', 'ACE_BOOTSTRAP_', 'RTLIB_TIMING'))}
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close',
               OMP_PLACES='cores', OMP_MAX_ACTIVE_LEVELS='1')
    if not args.skip_build:
        build(args, env)
    env['LD_LIBRARY_PATH'] = str(work / 'openfhe-install/lib')
    output = work / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run-settings.json').write_text(json.dumps({'threads': args.threads,
        'repetitions': args.repetitions, 'perf': args.perf,
        'openmp': {k: v for k, v in env.items() if k.startswith('OMP_')},
        'allocator': {k: v for k, v in env.items() if k.startswith('MALLOC_') or k == 'GLIBC_TUNABLES'}}, indent=2))
    for impl, binary in [('openfhe', work / 'openfhe-bench-build/openfhe_bench'), ('dsl_cpu', work / 'ant_bench')]:
        command = [binary, args.repetitions]
        if args.perf:
            command = ['perf', 'record', '--clockid', 'mono', '-e', 'cpu-clock:u', '-F', '99',
                       '--call-graph', 'dwarf,4096', '-o', output / f'{impl}.perf.data', '--', *command]
        run(command, output / f'{impl}.log', env, work)
    report(output, args.repetitions, args.perf)


if __name__ == '__main__':
    main()
