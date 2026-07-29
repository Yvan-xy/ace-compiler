#!/usr/bin/env python3
"""Compile, validate, and benchmark native/DSL baseline GEMM ONNX models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "model_and_main"
MODEL_SLOTS = {
    "i64_o10": 128,
    "i512_o10": 512,
    "i4096_o10": 4096,
    "i1024_o4096": 4096,
}
IMPLEMENTATIONS = ("native", "dsl")
DRIVER_END = """\
  Finalize_context();

  return 0;
}
"""
VALIDATING_DRIVER_END = r"""  const char *abs_error_text = getenv("ABS_ERROR");
  const char *rel_error_text = getenv("REL_ERROR");
  const double abs_tolerance =
      abs_error_text == NULL ? 0.0001 : atof(abs_error_text);
  const double rel_tolerance =
      rel_error_text == NULL ? 0.001 : atof(rel_error_text);
  bool valid = result != NULL;
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  int max_absolute_index = -1;
  for (int i = 0; result != NULL && i < Expected_len; ++i) {
    if (!isfinite(result[i]) || !isfinite(Expected_data[i])) {
      fprintf(
          stderr,
          "GEMM non-finite value at %d: actual=%.17g expected=%.17g\n",
          i, result[i], Expected_data[i]);
      max_absolute_error = INFINITY;
      max_relative_error = INFINITY;
      max_absolute_index = i;
      valid = false;
      continue;
    }
    const double absolute_error = fabs(result[i] - Expected_data[i]);
    const double denominator = fabs(Expected_data[i]);
    const double relative_error =
        denominator == 0.0
            ? (absolute_error == 0.0 ? 0.0 : 1.0)
            : absolute_error / denominator;
    if (absolute_error > max_absolute_error) {
      max_absolute_error = absolute_error;
      max_absolute_index = i;
    }
    if (relative_error > max_relative_error) {
      max_relative_error = relative_error;
    }
    if (absolute_error > abs_tolerance &&
        relative_error > rel_tolerance) {
      fprintf(
          stderr,
          "GEMM mismatch at %d: actual=%.17g expected=%.17g "
          "absolute_error=%.17g relative_error=%.17g\n",
          i, result[i], Expected_data[i], absolute_error, relative_error);
      valid = false;
    }
  }

  const char *output_path = getenv("ACE_OUTPUT_FILE");
  if (result != NULL && output_path != NULL) {
    FILE *output = fopen(output_path, "w");
    if (output == NULL) {
      perror("fopen ACE_OUTPUT_FILE");
      valid = false;
    } else {
      for (int i = 0; i < Expected_len; ++i) {
        fprintf(output, "%.17g\n", result[i]);
      }
      fclose(output);
    }
  }

  free(result);
  Finalize_context();

  printf(
      "GEMM_RESULT=%s max_abs=%.17g max_abs_index=%d max_rel=%.17g\n",
      valid ? "PASS" : "FAIL", max_absolute_error, max_absolute_index,
      max_relative_error);
  return valid ? 0 : 1;
}
"""
MAIN_GRAPH_PATTERN = re.compile(
    r"^\s*MAIN_GRAPH\s+\d+\s+([0-9]+(?:\.[0-9]+)?)\s+sec",
    re.MULTILINE,
)
RESULT_PATTERN = re.compile(
    r"^GEMM_RESULT=(PASS|FAIL)\s+max_abs=([^\s]+)"
    r"\s+max_abs_index=(-?\d+)\s+max_rel=([^\s]+)$",
    re.MULTILINE,
)


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_files(model_dir: Path, models: list[str]) -> None:
    missing = [
        model_dir / f"{model}{suffix}"
        for model in models
        for suffix in (".onnx", ".c")
        if not (model_dir / f"{model}{suffix}").is_file()
    ]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise RuntimeError(
            "missing local GEMM model fixtures:\n"
            f"{formatted}\n"
            "Pass their directory with --model-dir."
        )


def _generate_one(
    model: str,
    implementation: str,
    output_dir: Path,
    model_dir: Path,
) -> None:
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget
    from ace_edsl.edsl.vector_kernel_baseline_gemm import (
        configure_baseline_gemm_dsl,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    pipeline = (
        Pipeline(
            f"{model}_{implementation}",
            output_dir=str(output_dir / "ir"),
            dump_ir=model == "i64_o10",
            verbose=True,
        )
        .load_onnx(str(model_dir / f"{model}.onnx"))
        .configure_fhe(
            scaling_factor_bits=56,
            first_prime_bits=60,
            hamming_weight=192,
            data_file=f"{model}.weight",
            free_poly=True,
        )
    )
    slots = MODEL_SLOTS[model]
    if implementation == "native":
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="baseline-gemm",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
        )
    else:
        configure_baseline_gemm_dsl(
            pipeline,
            mask_fuse=False,
            max_slots=slots,
        )

    started = time.monotonic()
    result = pipeline.run(target=PipelineTarget.C)
    elapsed = time.monotonic() - started
    if not result.success:
        raise RuntimeError(result.error)
    expected_stages = [
        "tensor2vector",
        *(["vector_kernel_inline"] if implementation == "dsl" else []),
        "vector2sihe",
        "sihe2ckks",
        "ckks_driver",
        "poly_driver",
        "poly2c",
    ]
    if result.stages_completed != expected_stages:
        raise RuntimeError(
            f"unexpected stages: {result.stages_completed!r}; "
            f"expected {expected_stages!r}"
        )

    generated_c = output_dir / f"{model}.generated.c"
    generated_c.write_text(result.c_code)
    metadata = {
        "model": model,
        "implementation": implementation,
        "max_slots": slots,
        "stages": result.stages_completed,
        "generation_seconds": elapsed,
        "phase_seconds": pipeline.timings,
        "generated_c_bytes": generated_c.stat().st_size,
        "model_onnx_sha256": _sha256(model_dir / f"{model}.onnx"),
        "driver_sha256": _sha256(model_dir / f"{model}.c"),
    }
    (output_dir / "generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "GENERATION_RESULT=" + json.dumps(metadata, sort_keys=True, allow_nan=False),
        flush=True,
    )


def _generate_subprocess(
    script: Path,
    model: str,
    implementation: str,
    output_dir: Path,
    timeout: int,
    model_dir: Path,
) -> dict[str, object]:
    print(f"[generate] {model} {implementation}", flush=True)
    result = _run(
        [
            sys.executable,
            "-u",
            str(script),
            "--internal-generate",
            model,
            implementation,
            str(output_dir),
            str(model_dir),
        ],
        cwd=REPO_ROOT,
        timeout=timeout,
    )
    (output_dir / "generation.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"generation failed for {model} {implementation}\n{result.stdout}"
        )
    metadata = json.loads((output_dir / "generation.json").read_text())
    print(
        f"[generate] PASS {model} {implementation}: "
        f"{metadata['generation_seconds']:.3f}s, "
        f"{metadata['generated_c_bytes']} C bytes",
        flush=True,
    )
    return metadata


def _write_validating_driver(
    model: str,
    output_dir: Path,
    model_dir: Path,
) -> Path:
    source = (model_dir / f"{model}.c").read_text()
    if source.count(DRIVER_END) != 1:
        raise RuntimeError(f"cannot identify terminal block in {model}.c")
    source = source.replace(DRIVER_END, VALIDATING_DRIVER_END)
    driver = output_dir / f"{model}.main.inc"
    driver.write_text(source)
    return driver


def _compile_runner(
    model: str,
    output_dir: Path,
    timeout: int,
    model_dir: Path,
) -> Path:
    driver = _write_validating_driver(model, output_dir, model_dir)
    generated = output_dir / f"{model}.generated.c"
    runner = output_dir / f"{model}.runner.cxx"
    runner.write_text(f'#include "{driver.name}"\n#include "{generated.name}"\n')
    executable = output_dir / f"{model}.ace"
    command = [
        "c++",
        str(runner),
        "-DRTLIB_SUPPORT_LINUX",
        "-I/usr/local/include",
        "-I/usr/local/rtlib/include",
        "-I/usr/local/rtlib/include/ant",
        "-O3",
        "-DNDEBUG",
        "-std=gnu++17",
        "-fopenmp",
        "/usr/local/rtlib/lib/libFHErt_ant.a",
        "/usr/local/rtlib/lib/libFHErt_common.a",
        "/usr/local/lib/libAIRutil.a",
        "-lgmp",
        "-lm",
        "-lgomp",
        "-o",
        str(executable),
    ]
    print(f"[compile] {model} {output_dir.name}", flush=True)
    result = _run(command, cwd=output_dir, timeout=timeout)
    (output_dir / "compile.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"compile failed for {model} in {output_dir}\n{result.stdout}"
        )
    return executable


def _parse_run(
    result: subprocess.CompletedProcess[str],
    *,
    model: str,
    implementation: str,
) -> tuple[float, dict[str, object]]:
    if result.returncode != 0:
        raise RuntimeError(
            f"runtime failed for {model} {implementation} "
            f"(exit {result.returncode})\n{result.stdout}"
        )
    timing_match = MAIN_GRAPH_PATTERN.search(result.stdout)
    validation_match = RESULT_PATTERN.search(result.stdout)
    if timing_match is None or validation_match is None:
        raise RuntimeError(
            f"missing timing or validation result for {model} "
            f"{implementation}\n{result.stdout}"
        )
    if validation_match.group(1) != "PASS":
        raise RuntimeError(
            f"validation failed for {model} {implementation}\n{result.stdout}"
        )
    timing = float(timing_match.group(1))
    validation: dict[str, float | int] = {
        "max_abs_error": float(validation_match.group(2)),
        "max_abs_error_index": int(validation_match.group(3)),
        "max_rel_error": float(validation_match.group(4)),
    }
    if not math.isfinite(timing) or timing <= 0.0:
        raise RuntimeError(
            f"invalid MAIN_GRAPH timing for {model} {implementation}: "
            f"{timing_match.group(1)}"
        )
    for metric in ("max_abs_error", "max_rel_error"):
        value = validation[metric]
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(
                f"invalid {metric} for {model} {implementation}: {value}"
            )
    return timing, validation


def _run_one(
    model: str,
    implementation: str,
    executable: Path,
    output_path: Path,
    log_path: Path,
    timeout: int,
) -> tuple[float, dict[str, object]]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "RTLIB_DISABLE_BOOTSTRAP_PRECOM": "1",
            "RTLIB_TIMING_OUTPUT": "stdout",
            "ABS_ERROR": "0.0001",
            "REL_ERROR": "0.001",
            "ACE_OUTPUT_FILE": str(output_path),
        }
    )
    result = _run(
        [str(executable)],
        cwd=executable.parent,
        env=env,
        timeout=timeout,
    )
    log_path.write_text(result.stdout)
    return _parse_run(
        result,
        model=model,
        implementation=implementation,
    )


def _read_values(path: Path) -> list[float]:
    return [float(line) for line in path.read_text().splitlines()]


def _compare_outputs(
    native_path: Path,
    dsl_path: Path,
    *,
    tolerance: float,
) -> dict[str, object]:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"output tolerance must be finite and nonnegative: {tolerance}"
        )
    native = _read_values(native_path)
    dsl = _read_values(dsl_path)
    if len(native) != len(dsl):
        raise RuntimeError(
            f"native/DSL output length mismatch: {len(native)} != {len(dsl)}"
        )
    if not native:
        raise RuntimeError("native/DSL output files are empty")
    for implementation, values in (("native", native), ("DSL", dsl)):
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{implementation} output is non-finite at index {index}: {value}"
                )
    differences = [abs(left - right) for left, right in zip(native, dsl)]
    max_difference = max(differences, default=0.0)
    max_index = differences.index(max_difference) if differences else -1
    if not math.isfinite(max_difference) or max_difference > tolerance:
        raise RuntimeError(
            f"native/DSL max absolute difference {max_difference} at "
            f"index {max_index} exceeds {tolerance}"
        )
    return {
        "length": len(native),
        "max_abs_difference": max_difference,
        "max_abs_difference_index": max_index,
        "tolerance": tolerance,
    }


def _summarize_timings(
    timings: dict[str, list[float]],
) -> dict[str, object]:
    native_median = statistics.median(timings["native"])
    dsl_median = statistics.median(timings["dsl"])
    ratio = dsl_median / native_median
    return {
        implementation: {
            "seconds": values,
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        for implementation, values in timings.items()
    } | {
        "dsl_over_native_ratio": ratio,
        "dsl_gap_percent": (ratio - 1.0) * 100.0,
    }


def _benchmark_model(
    script: Path,
    root: Path,
    model: str,
    warmups: int,
    runs: int,
    timeout: int,
    cross_tolerance: float,
    model_dir: Path,
) -> dict[str, object]:
    model_root = root / model
    generation = {}
    executables = {}
    for implementation in IMPLEMENTATIONS:
        output_dir = model_root / implementation
        output_dir.mkdir(parents=True, exist_ok=True)
        generation[implementation] = _generate_subprocess(
            script, model, implementation, output_dir, timeout, model_dir
        )
        executables[implementation] = _compile_runner(
            model, output_dir, timeout, model_dir
        )

    validation_samples: dict[str, list[dict[str, object]]] = {
        implementation: [] for implementation in IMPLEMENTATIONS
    }
    for warmup in range(warmups):
        for implementation in IMPLEMENTATIONS:
            print(
                f"[warmup {warmup + 1}/{warmups}] {model} {implementation}",
                flush=True,
            )
            output = model_root / implementation / f"warmup-{warmup}.txt"
            timing, validation = _run_one(
                model,
                implementation,
                executables[implementation],
                output,
                model_root / implementation / f"warmup-{warmup}.log",
                timeout,
            )
            validation_samples[implementation].append(validation)
            print(
                f"[warmup] PASS {model} {implementation}: MAIN_GRAPH={timing:.6f}s",
                flush=True,
            )

    timings = {implementation: [] for implementation in IMPLEMENTATIONS}
    output_comparisons = []
    for run in range(runs):
        order = IMPLEMENTATIONS if run % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
        for implementation in order:
            print(
                f"[run {run + 1}/{runs}] {model} {implementation}",
                flush=True,
            )
            output = model_root / implementation / f"run-{run}.txt"
            timing, validation = _run_one(
                model,
                implementation,
                executables[implementation],
                output,
                model_root / implementation / f"run-{run}.log",
                timeout,
            )
            validation_samples[implementation].append(validation)
            timings[implementation].append(timing)
            print(
                f"[run] PASS {model} {implementation}: MAIN_GRAPH={timing:.6f}s",
                flush=True,
            )
        output_comparison = _compare_outputs(
            model_root / "native" / f"run-{run}.txt",
            model_root / "dsl" / f"run-{run}.txt",
            tolerance=cross_tolerance,
        )
        output_comparison["sample"] = f"measured-run-{run + 1}"
        output_comparisons.append(output_comparison)

    worst_output = max(
        output_comparisons,
        key=lambda sample: sample["max_abs_difference"],
    )
    output_comparison = {
        "samples": output_comparisons,
        "length": worst_output["length"],
        "max_abs_difference": worst_output["max_abs_difference"],
        "max_abs_difference_index": worst_output["max_abs_difference_index"],
        "max_abs_difference_sample": worst_output["sample"],
        "tolerance": cross_tolerance,
    }
    validation = {}
    for implementation, samples in validation_samples.items():
        max_absolute_sample = max(samples, key=lambda sample: sample["max_abs_error"])
        validation[implementation] = {
            "samples": samples,
            "max_abs_error": max_absolute_sample["max_abs_error"],
            "max_abs_error_index": max_absolute_sample["max_abs_error_index"],
            "max_rel_error": max(sample["max_rel_error"] for sample in samples),
        }
    performance = _summarize_timings(timings)
    summary = {
        "model": model,
        "max_slots": MODEL_SLOTS[model],
        "generation": generation,
        "validation": validation,
        "native_dsl_output": output_comparison,
        "performance": performance,
    }
    (model_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"[model] PASS {model}: output max_abs_difference="
        f"{output_comparison['max_abs_difference']:.6g}; "
        f"native median={performance['native']['median_seconds']:.6f}s; "
        f"DSL median={performance['dsl']['median_seconds']:.6f}s; "
        f"gap={performance['dsl_gap_percent']:+.2f}%",
        flush=True,
    )
    return summary


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SLOTS),
        default=list(MODEL_SLOTS),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="directory containing matching <model>.onnx and <model>.c files",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ace-gemm-e2e"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--cross-tolerance", type=float, default=0.0002)
    parser.add_argument(
        "--internal-generate",
        nargs=4,
        metavar=("MODEL", "IMPLEMENTATION", "OUTPUT_DIR", "MODEL_DIR"),
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments.warmups < 0 or arguments.runs < 1:
        parser.error("--warmups must be nonnegative and --runs must be positive")
    if arguments.timeout < 1:
        parser.error("--timeout must be positive")
    if not math.isfinite(arguments.cross_tolerance) or arguments.cross_tolerance < 0.0:
        parser.error("--cross-tolerance must be finite and nonnegative")
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    if arguments.internal_generate is not None:
        model, implementation, output_dir, model_dir = arguments.internal_generate
        if model not in MODEL_SLOTS or implementation not in IMPLEMENTATIONS:
            raise ValueError((model, implementation))
        resolved_model_dir = Path(model_dir).resolve()
        _validate_model_files(resolved_model_dir, [model])
        _generate_one(
            model,
            implementation,
            Path(output_dir).resolve(),
            resolved_model_dir,
        )
        return 0

    script = Path(__file__).resolve()
    root = arguments.output_dir.resolve()
    model_dir = arguments.model_dir.resolve()
    _validate_model_files(model_dir, arguments.models)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in arguments.models:
        summaries.append(
            _benchmark_model(
                script,
                root,
                model,
                arguments.warmups,
                arguments.runs,
                arguments.timeout,
                arguments.cross_tolerance,
                model_dir,
            )
        )
    aggregate = {
        "settings": {
            "models": arguments.models,
            "model_dir": str(model_dir),
            "warmups": arguments.warmups,
            "runs": arguments.runs,
            "timeout_seconds": arguments.timeout,
            "cross_tolerance": arguments.cross_tolerance,
            "omp_num_threads": 1,
            "rtlib_disable_bootstrap_precom": 1,
        },
        "models": summaries,
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("GEMM_E2E_RESULT=PASS", flush=True)
    print(f"SUMMARY={root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
