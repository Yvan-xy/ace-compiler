#!/usr/bin/env python3
"""Formal comparison of legacy / serial / limb NTT schedules using step-two artifacts."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics

from run import HERE, ROOT, run, sha
from profile import instrument_ant

MODES = {"legacy": 0, "serial": 1, "limb": 16}


def records(path, prefix):
    return [json.loads(line.split("=", 1)[1]) for line in path.read_text().splitlines()
            if line.startswith(prefix + "=")]


def prepare(args):
    work, prefix = args.work_dir, args.ace_prefix
    includes = ["-I" + str(prefix / p) for p in ("include", "rtlib/include", "rtlib/include/ant")]
    flags = ["-O3", "-DNDEBUG", "-DRTLIB_SUPPORT_LINUX", "-fopenmp", *includes]
    libs = [prefix / "rtlib/lib/libFHErt_ant.a", prefix / "rtlib/lib/libFHErt_common.a",
            prefix / "lib/libAIRutil.a"]
    manifest = {"libraries": {str(p): sha(p) for p in libs}, "modes": {},
                "source_hashes": {str(p): sha(p) for p in [HERE / "ant_main.cxx", HERE / "common.h",
                                    HERE / "profile.py", HERE / "profile_hooks.c", HERE / "compare_ntt.py"]}}
    for mode, budget in MODES.items():
        case = work / mode
        case.mkdir(exist_ok=True)
        original = args.artifacts / f"bts{budget}"
        generated = original / "generator/output/bootstrap_full.c"
        metadata = json.loads((original / "build.json").read_text())
        if int(metadata["generator_environment"]["ACE_BOOTSTRAP_DECOMP_NTT_THREADS"]) != budget:
            raise ValueError("step-two artifact budget mismatch")
        for p in [generated, original / "shim.c", *libs[:2]]:
            if metadata["files_sha256"][str(p)] != sha(p):
                raise ValueError(f"step-two artifact/library changed: {p}")
        text = generated.read_text()
        if text.count("Decomp_modup_with_ntt_threads(") != (32 if budget else 0):
            raise ValueError("unexpected decomposition call sites")
        if text.count("#pragma omp parallel sections") != (6 if budget else 7):
            raise ValueError("unexpected outer section schedule")
        # A fresh compile/link ensures no link-time NTT observer enters timing runs.
        for src, obj in [(generated, case / "bootstrap.o"), (original / "shim.c", case / "shim.o")]:
            run(["gcc", "-std=gnu11", *flags, "-c", src, "-o", obj], obj.with_suffix(".log"))
        run(["g++", "-std=gnu++17", *flags, HERE / "ant_main.cxx", case / "bootstrap.o",
             case / "shim.o", *libs, "-lgmp", "-lm", "-o", case / "plain"], case / "plain-build.log")
        # Stage timings are a separate diagnostic run, never the headline timings.
        (case / "phases.c").write_text(instrument_ant(text))
        main = '#include "profile_hooks.h"\n' + (HERE / "ant_main.cxx").read_text()
        main = main.replace("    double start = bench::Now();", "    ace_profile_begin(sample);\n    double start = bench::Now();")
        main = main.replace("    unsigned raw_q =", '    ace_profile_mark("normalize");\n    unsigned raw_q =')
        main = main.replace("    double seconds = bench::Now() - start;", "    double seconds = bench::Now() - start;\n    ace_profile_end();")
        (case / "phase_main.cxx").write_text(main)
        for src, obj in [(case / "phases.c", case / "phases.o"), (HERE / "profile_hooks.c", case / "hooks.o")]:
            run(["gcc", "-std=gnu11", *flags, "-I" + str(HERE), "-c", src, "-o", obj], obj.with_suffix(".log"))
        run(["g++", "-std=gnu++17", *flags, "-I" + str(HERE), case / "phase_main.cxx",
             case / "phases.o", case / "hooks.o", case / "shim.o", *libs, "-lgmp", "-lm",
             "-o", case / "phase"], case / "phase-build.log")
        manifest["modes"][mode] = {"budget": budget, "generated_source": str(generated),
                                  "source_sha256": sha(generated), "plain_sha256": sha(case / "plain"),
                                  "phase_sha256": sha(case / "phase")}
    (work / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate(path, repetitions, threads):
    result = records(path, "CPU_BTS_RESULT")
    usage = records(path, "CPU_BTS_RESOURCE")
    if len(result) != repetitions + 1 or len(usage) != len(result):
        raise ValueError(f"missing result/resource samples: {path}")
    expected = dict(ring_dimension=65536, slots=32768, context_q=31, p_count=11,
                    raised_q=31, input_q=2, input_scale_degree=1,
                    output_q=14, output_scale_degree=1, threads=threads)
    for sample, (r, u) in enumerate(zip(result, usage)):
        if (r["sample"] != sample or u["sample"] != sample or r["timed"] != bool(sample)
                or r["status"] != "pass" or any(r[k] != v for k, v in expected.items())):
            raise ValueError(f"parameter/sample mismatch: {path}")
        if (not math.isfinite(r["seconds"]) or r["seconds"] <= 0 or
                not 0 <= r["maximum_error"] <= 0.02 or
                not math.isfinite(r["output_scale"]) or r["output_scale"] <= 0 or
                abs(math.log2(r["output_scale"]) - 56) >= 0.01):
            raise ValueError(f"invalid time/accuracy/scale: {path}")
    measured = result[1:]
    return dict(samples_seconds=[r["seconds"] for r in measured],
                median_seconds=statistics.median(r["seconds"] for r in measured),
                maximum_error=max(r["maximum_error"] for r in result),
                cpu_seconds=statistics.median(r["cpu_seconds"] for r in usage[1:]),
                system_seconds=statistics.median(r["system_seconds"] for r in usage[1:]),
                minor_faults=statistics.median(r["minor_faults"] for r in usage[1:]),
                major_faults=max(r["major_faults"] for r in usage),
                process_peak_rss_kib=max(r["process_peak_rss_kib"] for r in usage))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ace-prefix", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, default=ROOT / "tmp_cpu_ntt_step2")
    p.add_argument("--work-dir", type=Path, default=ROOT / "tmp_cpu_ntt_step3")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--threads", type=int, nargs="+", default=[16, 1])
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()
    if args.repetitions < 3 or args.repetitions > 100 or any(t < 1 for t in args.threads):
        p.error("3..100 measured repetitions and positive thread counts required")
    for name in ("ace_prefix", "artifacts", "work_dir"):
        setattr(args, name, getattr(args, name).resolve())
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        prepare(args)
    else:
        saved = json.loads((work / "build.json").read_text())
        for path, digest in {**saved["libraries"], **saved["source_hashes"]}.items():
            if sha(Path(path)) != digest: raise ValueError(f"rebuild required: {path}")
        for mode, item in saved["modes"].items():
            for kind in ("plain", "phase"):
                if sha(work / mode / kind) != item[kind + "_sha256"]:
                    raise ValueError("binary changed")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "MALLOC_", "ACE_BOOTSTRAP_", "RTLIB_TIMING", "GOMP_"))
           and k not in ("GLIBC_TUNABLES", "LD_PRELOAD")}
    env.update(OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_MAX_ACTIVE_LEVELS="1")
    run(["lscpu"], work / "cpu.txt")
    (work / "settings.json").write_text(json.dumps({"repetitions": args.repetitions,
        "threads": args.threads, "openmp": {k: v for k, v in env.items() if k.startswith("OMP_")},
        "allocator": "default", "ntt_probe": False, "perf_sampling": False}, indent=2))
    summary = {"plain": {}, "phase": {}}
    (work / "summary.json").unlink(missing_ok=True)
    # Finish every primary measurement before running stage diagnostics.
    for threads in args.threads:
        env["OMP_NUM_THREADS"] = str(threads)
        summary["plain"][str(threads)] = {}
        order = list(MODES) if threads != 1 else list(reversed(MODES))
        for mode in order:
            log = work / f"{mode}-t{threads}.log"
            run([work / mode / "plain", args.repetitions], log, env, work)
            summary["plain"][str(threads)][mode] = validate(log, args.repetitions, threads)
            (work / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
    if 16 in args.threads:
        env["OMP_NUM_THREADS"] = "16"
        for mode in MODES:
            log = work / f"{mode}-phases.log"
            run([work / mode / "phase", args.repetitions], log, env, work)
            validate(log, args.repetitions, 16)
            rows = records(log, "CPU_BTS_PHASE")
            expected = ["entry", "modraise", "coeff_to_slots", "split", "evalmod", "recombine", "slots_to_coeffs", "postscale", "normalize"]
            for sample in range(args.repetitions + 1):
                batch = [r for r in rows if r["sample"] == sample]
                if [r["stage"] for r in batch] != expected:
                    raise ValueError("phase boundary mismatch")
                if any(a["end"] != b["start"] for a, b in zip(batch, batch[1:])):
                    raise ValueError("noncontiguous phase boundaries")
            grouped = defaultdict(list)
            for r in rows:
                if r["sample"]: grouped[r["stage"]].append(r)
            summary["phase"][mode] = {stage: {key: statistics.median(r[key] for r in group)
                for key in ("seconds", "cpu_seconds", "system_seconds", "minor_faults")}
                for stage, group in grouped.items()}
            (work / "progress.json").write_text(json.dumps(summary, indent=2) + "\n")
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
