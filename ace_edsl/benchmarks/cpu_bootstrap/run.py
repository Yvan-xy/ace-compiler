#!/usr/bin/env python3
"""Build and run one matched full-packed DSL/OpenFHE CPU bootstrap profile."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OPENFHE_COMMIT = "1306d14f8c26bb6150d3e6ad54f28dfe1007689e"


def run(command, log, env=None, cwd=None):
    print(f"Running {command[0]} ({log})", flush=True)
    with log.open("w") as out:
        try:
            subprocess.run([str(x) for x in command], check=True, stdout=out,
                           stderr=subprocess.STDOUT, env=env, cwd=cwd)
        except subprocess.CalledProcessError:
            print(f"Failed; see {log}", file=sys.stderr)
            raise


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_coefficients(source):
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENFHE_COMMIT:
        raise ValueError("benchmark requires the pinned OpenFHE v1.5.1 commit")
    if subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--", "src"], text=True):
        raise ValueError("OpenFHE source has local changes")
    spec = importlib.util.spec_from_file_location(
        "bts_constants", ROOT / "ace_edsl/examples/bootstrap_ant_constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    header = (source / "src/pke/include/scheme/ckksrns/ckksrns-fhe.h").read_text()
    body = re.search(r"g_coefficientsSparse\s*\{([^}]+)\}", header).group(1)
    coefficients = tuple(float(x.strip()) for x in body.split(",") if x.strip())
    if (coefficients != constants.G_COEFFICIENTS_OPENFHE_SPARSE or
            len(coefficients) != 45 or constants.OPENFHE_SPARSE_UPPER_BOUND_K != 28 or
            constants.R_UNIFORM_HW_192 != 3):
        raise ValueError("DSL and OpenFHE EvalMod parameters differ")


def build(args, work, env):
    generator = work / "generator"
    generator.mkdir(exist_ok=True)
    for name in ("bootstrap_full.py", "bootstrap_ant_constants.py",
                 "ant_bootstrap_ref.py", "resnet_bootstrap_utils.py"):
        shutil.copy2(ROOT / "ace_edsl/examples" / name, generator / name)
    # Remove inherited experimental settings before selecting this fixed profile.
    gen_env = {k: v for k, v in env.items()
               if not k.startswith(("ACE_BOOTSTRAP_", "ACE_CT_ENCODE"))}
    gen_env.update({
        "ACE_BOOTSTRAP_EVALMOD_PROFILE": "openfhe_sparse",
        "ACE_BOOTSTRAP_LINEAR_TRANSFORM": "1",
        "ACE_BOOTSTRAP_PARALLEL_EVAL_MOD": "1",
        "ACE_BOOTSTRAP_DECOMP_NTT_THREADS": str(args.decomp_ntt_threads),
        "ACE_BOOTSTRAP_EVALMOD_SCHEDULE": str(args.evalmod_schedule),
        "ACE_BOOTSTRAP_CT_ENCODE": "1", "ACE_CT_ENCODE_DEPTH": "30",
        "ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL": "31", "ACE_BOOTSTRAP_INPUT_LEVEL": "2",
        "ACE_BOOTSTRAP_FIRST_PRIME_BITS": "60", "ACE_BOOTSTRAP_SCALING_FACTOR_BITS": "56",
        "ACE_BOOTSTRAP_HAMMING_WEIGHT": "192", "ACE_BOOTSTRAP_Q_PARTS": "3",
        "ACE_BOOTSTRAP_ENC_BUDGET": "3", "ACE_BOOTSTRAP_DEC_BUDGET": "3",
    })
    gen_env["PYTHONPATH"] = os.pathsep.join(
        [str(args.bindings_dir), str(ROOT), str(ROOT / "ace_edsl")])
    run([sys.executable, generator / "resnet_bootstrap_utils.py", "generate-demo",
         "--impl", "primitive", "--poly-degree", "65536", "--mul-level", "30",
         "--bsgs-giant-step", "16"], work / "generate.log", gen_env, generator)
    generated = generator / "output/bootstrap_full.c"
    code = generated.read_text()
    raw = (generator / "output/bootstrap_full_raw.air").read_text()
    if "Eval_bootstrap_ciph(" in code or "CKKS.bootstrap" in raw:
        raise ValueError("DSL artifact unexpectedly delegates bootstrap")
    # No global application weight file is needed here. All generated Free_data
    # operands are owned temporaries (the shim copies pre-encoded plaintexts).
    # ANT's default Free_data is a no-op without that global file, so release
    # these temporaries directly in this standalone executable.
    generated.write_text(code.replace("Free_data(", "Free_poly_data("))
    run([sys.executable, ROOT / "ace_edsl/examples/resnet_bootstrap_utils.py",
         "emit-shim", "--output", work / "shim.c"], work / "shim.log")
    prefix = args.ace_prefix
    includes = ["-I" + str(prefix / p) for p in
                ("include", "rtlib/include", "rtlib/include/ant")]
    flags = ["-O3", "-DNDEBUG", "-DRTLIB_SUPPORT_LINUX", "-fopenmp"] + includes
    for source, obj in [(generated, "bootstrap.o"), (work / "shim.c", "shim.o")]:
        run(["gcc", "-std=gnu11", *flags, "-c", source, "-o", work / obj],
            work / (obj + ".log"))
    run(["g++", "-std=gnu++17", *flags, HERE / "ant_main.cxx",
         *([HERE / "ntt_probe.cxx", "-Wl,--wrap=Decomp_modup_with_ntt_threads",
            "-Wl,--wrap=Ftt_fwd", "-Wl,--wrap=Ftt_inv"] if args.ntt_probe else []),
         work / "bootstrap.o", work / "shim.o",
         prefix / "rtlib/lib/libFHErt_ant.a", prefix / "rtlib/lib/libFHErt_common.a",
         prefix / "lib/libAIRutil.a", "-lgmp", "-lm", "-o", work / "ant_bench"],
        work / "ant-build.log")
    run(["cmake", "-S", HERE, "-B", work / "openfhe-bench-build",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={args.openfhe_prefix}"],
        work / "openfhe-configure.log")
    run(["cmake", "--build", work / "openfhe-bench-build", "-j", "4"],
        work / "openfhe-build.log")
    provenance = {
        "ace_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "openfhe_revision": OPENFHE_COMMIT,
        "ntt_probe": args.ntt_probe,
        "generator_environment": {k: v for k, v in gen_env.items() if k.startswith("ACE_")},
        "files_sha256": {str(p): sha(p) for p in [
            *HERE.glob("*.cxx"), HERE / "common.h", HERE / "run.py", HERE / "CMakeLists.txt",
            *generator.glob("*.py"),
            ROOT / "ace_edsl/edsl/core/bootstrap_decomposition.py",
            ROOT / "ace_edsl/edsl/core/bootstrap_math.py",
            generated, work / "shim.c", work / "ant_bench",
            work / "openfhe-bench-build/openfhe_bench",
            prefix / "rtlib/lib/libFHErt_ant.a", prefix / "rtlib/lib/libFHErt_common.a",
            *args.openfhe_prefix.glob("lib/libOPENFHE*.so.1.5.1"),
            *args.bindings_dir.glob("ace_bindings/*.so")]},
    }
    (work / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")


def summarize(work, repetitions):
    summary = {}
    for impl in ("openfhe", "dsl_cpu"):
        rows = [json.loads(line.split("=", 1)[1])
                for line in (work / f"{impl}.log").read_text().splitlines()
                if line.startswith("CPU_BTS_RESULT=")]
        if len(rows) != repetitions + 1 or [r["sample"] for r in rows] != list(range(repetitions + 1)):
            raise ValueError(f"{impl}: missing benchmark samples")
        for i, row in enumerate(rows):
            if (row["status"] != "pass" or row["timed"] != (i != 0) or
                    row["implementation"] != impl):
                raise ValueError(f"{impl}: failed result or incorrect warm-up")
            if (not math.isfinite(row["seconds"]) or row["seconds"] <= 0 or
                    not math.isfinite(row["maximum_error"]) or
                    not 0 <= row["maximum_error"] <= 0.02 or
                    not math.isfinite(row["output_scale"]) or row["output_scale"] <= 0 or
                    abs(math.log2(row["output_scale"]) - 56) >= 0.01):
                raise ValueError(f"{impl}: invalid timing, accuracy, or scale")
        times = [r["seconds"] for r in rows if r["timed"]]
        summary[impl] = {"median_seconds": statistics.median(times), "samples_seconds": times,
                         "maximum_error": max(r["maximum_error"] for r in rows),
                         "raw_output_q": rows[0]["raw_output_q"],
                         "raw_output_scale_degree": rows[0]["raw_output_scale_degree"]}
        state = {k: rows[0][k] for k in ("ring_dimension", "slots", "context_q", "p_count",
                 "raised_q", "input_q", "input_scale_degree", "output_q", "output_scale_degree", "threads")}
        if "matched_state" in summary and summary["matched_state"] != state:
            raise ValueError("benchmark parameter mismatch")
        if any(any(row[k] != v for k, v in state.items()) for row in rows):
            raise ValueError("benchmark state changed between repetitions")
        summary["matched_state"] = state
    summary["openfhe_over_dsl"] = summary["openfhe"]["median_seconds"] / summary["dsl_cpu"]["median_seconds"]
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-prefix", type=Path, required=True)
    parser.add_argument("--bindings-dir", type=Path, required=True,
                        help="directory containing the ace_bindings package")
    parser.add_argument("--openfhe-prefix", type=Path, required=True)
    parser.add_argument("--openfhe-source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp_openfhe_cpu_compare/benchmark")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--decomp-ntt-threads", type=int, default=0,
                        help="0: legacy branches; 1: new serial; >1: NTT limb budget")
    parser.add_argument("--evalmod-schedule", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--ntt-probe", action="store_true",
                        help="observe actual NTT workers (correctness runs only)")
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100 or args.threads < 1:
        parser.error("positive thread count and 1..100 repetitions required")
    if args.evalmod_schedule and (args.decomp_ntt_threads or args.ntt_probe):
        parser.error("full EvalMod mode conflicts with partial NTT mode/probe")
    if not 0 <= args.decomp_ntt_threads <= 0xFFFFFFFF:
        parser.error("decomp-ntt-threads must be uint32")
    for name in ("ace_prefix", "bindings_dir", "openfhe_prefix", "openfhe_source", "work_dir"):
        setattr(args, name, getattr(args, name).resolve())
    verify_coefficients(args.openfhe_source)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMP_", "ACE_BOOTSTRAP_", "RTLIB_TIMING"))}
    env.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OMP_PROC_BIND="close", OMP_PLACES="cores", OMP_MAX_ACTIVE_LEVELS="1")
    env["LD_LIBRARY_PATH"] = str(args.openfhe_prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    if not args.skip_build:
        build(args, work, env)
    else:
        provenance = json.loads((work / "build.json").read_text())
        if provenance.get("ntt_probe", False) != args.ntt_probe:
            raise ValueError("NTT probe differs from build; rebuild required")
        if int(provenance["generator_environment"].get("ACE_BOOTSTRAP_EVALMOD_SCHEDULE", "0")) != args.evalmod_schedule:
            raise ValueError("EvalMod mode differs from build")
        built_budget = int(provenance["generator_environment"].get("ACE_BOOTSTRAP_DECOMP_NTT_THREADS", "0"))
        if built_budget != args.decomp_ntt_threads:
            raise ValueError("NTT thread budget differs from build; rebuild required")
        for path, digest in provenance["files_sha256"].items():
            if sha(Path(path)) != digest:
                raise ValueError(f"build inputs changed; rebuild required: {path}")
    if args.build_only:
        return
    # A failed new run must not leave an old comparison looking current.
    (work / "summary.json").unlink(missing_ok=True)
    # Sequential runs avoid competition for CPU/memory bandwidth.
    run(["lscpu"], work / "cpu.txt")
    (work / "run.json").write_text(json.dumps({"threads": args.threads,
        "repetitions": args.repetitions, "openmp": {k: v for k, v in env.items() if k.startswith("OMP_")}}, indent=2))
    for impl, binary in [("openfhe", work / "openfhe-bench-build/openfhe_bench"),
                         ("dsl_cpu", work / "ant_bench")]:
        run([binary, args.repetitions], work / f"{impl}.log", env, work)
    summarize(work, args.repetitions)


if __name__ == "__main__":
    main()
