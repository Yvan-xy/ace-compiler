#!/usr/bin/env python3
"""Compile, validate, and benchmark selected GEMM lowering paths."""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
    "gemmh10w4096": 4096,
    "i1024_o4096": 4096,
}
IMPLEMENTATIONS = ("native", "dsl")
THREE_WAY_IMPLEMENTATIONS = ("dsl-fast", "cpp-baseline", "metakernel-fast")
ALL_IMPLEMENTATIONS = IMPLEMENTATIONS + THREE_WAY_IMPLEMENTATIONS
SCALING_FACTOR_BITS = 56
FIRST_PRIME_BITS = 60
HAMMING_WEIGHT = 192
FREE_POLY = True
OMP_NUM_THREADS = 1
ABS_ERROR_TOLERANCE = 0.0001
REL_ERROR_TOLERANCE = 0.001
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


def _helper_records(ir: str, helper_prefix: str) -> tuple[int, int]:
    definitions = len(
        re.findall(rf'^FUN\[[^\n]*"{re.escape(helper_prefix)}', ir, re.MULTILINE)
    )
    calls = len(re.findall(rf'^\s+call "{re.escape(helper_prefix)}', ir, re.MULTILINE))
    return definitions, calls


def _rotation_numbers(ir: str) -> list[list[int]]:
    values = []
    for raw in re.findall(r"ATTR\[nums=([^]]+)\]", ir):
        fields = raw.strip("()").split(",")
        values.append([int(field) for field in fields if field])
    return values


def _expected_rotation_numbers(prepared: object) -> list[list[int]]:
    values = []
    for rotation in prepared.rotations:
        candidates = list(rotation.candidates)
        if rotation.role == "input-duplication":
            values.extend([[candidate] for candidate in candidates])
        else:
            values.append(candidates)
    return values


def _path_evidence(
    implementation: str,
    phase_irs: dict[str, str],
    prepared_plans: list[object],
    *,
    require_inlined: bool = True,
) -> dict[str, object]:
    tensor_ir = phase_irs.get("tensor2vector", "")
    if not tensor_ir:
        raise RuntimeError("path proof is missing Tensor-to-Vector IR")
    if re.search(r"\bNN\.gemm\b", tensor_ir, re.IGNORECASE):
        raise RuntimeError("Tensor-to-Vector path left an NN Gemm behind")

    if implementation == "dsl-fast":
        kernel_impl = "dsl"
        requested_plan_kind = "fast-gemm"
    elif implementation == "metakernel-fast":
        kernel_impl = "native"
        requested_plan_kind = "fast-gemm"
    else:
        kernel_impl = "dsl" if implementation == "dsl" else "native"
        requested_plan_kind = "baseline-gemm"

    evidence: dict[str, object] = {
        "requested_implementation": implementation,
        "requested_plan_provider": "cpp",
        "requested_kernel_impl": kernel_impl,
        "requested_plan_kind": requested_plan_kind,
        "requested_fallback": "error",
        "source_gemm_removed": True,
        "fallback_used": False,
        "fast_metakernel_comment": "IMRA Metakernel:" in tensor_ir,
        "packed_partition_comment": "gemm result reduce->Ps=" in tensor_ir,
        "kp_over_np_comment": "gemm result reduce->(kp/np)=" in tensor_ir,
    }
    if implementation == "dsl-fast":
        if len(prepared_plans) != 1:
            raise RuntimeError(
                "DSL-fast path requires exactly one captured prepared plan"
            )
        prepared = prepared_plans[0]
        if prepared.kind != "fast-gemm" or prepared.provenance != "cpp":
            raise RuntimeError(
                "explicit DSL-fast path did not receive the C++ fast-GEMM plan"
            )
        helper_name = prepared.helper_name
        definitions, calls = _helper_records(tensor_ir, helper_name)
        if definitions != 1 or calls != 1:
            raise RuntimeError(
                "DSL-fast path lacks one specialized helper and typed call"
            )
        if not all(
            (
                evidence["fast_metakernel_comment"],
                evidence["packed_partition_comment"],
                evidence["kp_over_np_comment"],
            )
        ):
            raise RuntimeError("DSL-fast Tensor-to-Vector IR lacks fast markers")
        rotation_numbers = _rotation_numbers(tensor_ir)
        expected_rotation_numbers = _expected_rotation_numbers(prepared)
        if rotation_numbers != expected_rotation_numbers:
            raise RuntimeError(
                "DSL-fast rotation attributes do not match the frozen plan"
            )
        inlined_ir = phase_irs.get("vector_kernel_inline", "")
        post_definitions, post_calls = (
            _helper_records(inlined_ir, helper_name) if inlined_ir else (None, None)
        )
        inline_success = bool(inlined_ir) and post_definitions == post_calls == 0
        if require_inlined and not inline_success:
            raise RuntimeError("DSL-fast helper or call survived the tentative inliner")
        evidence.update(
            {
                "actual_plan_kind": prepared.kind,
                "plan_provenance": prepared.provenance,
                "specialization_key": prepared.specialization_key,
                "helper_name": helper_name,
                "prepared_callback_count": len(prepared_plans),
                "pre_inline_helper_definitions": definitions,
                "pre_inline_helper_calls": calls,
                "inline_success": inline_success,
                "post_inline_helper_definitions": post_definitions,
                "post_inline_helper_calls": post_calls,
                "post_inline_helper_name_occurrences": (
                    inlined_ir.count(helper_name) if inlined_ir else None
                ),
                "constant_hashes": {
                    item.role: item.content_hash for item in prepared.constants
                },
                "rotation_candidates": {
                    item.role: list(item.candidates) for item in prepared.rotations
                },
                "rotation_rnums": rotation_numbers,
                "result_type": [
                    prepared.result_type.element_type,
                    list(prepared.result_type.shape),
                ],
                "slot": [prepared.slot.policy, prepared.slot.value],
            }
        )
    elif implementation == "metakernel-fast":
        if not all(
            (
                evidence["fast_metakernel_comment"],
                evidence["packed_partition_comment"],
                evidence["kp_over_np_comment"],
            )
        ):
            raise RuntimeError("forced native-fast IR lacks fast markers")
        evidence["actual_plan_kind"] = "fast-gemm"
    else:
        if evidence["fast_metakernel_comment"]:
            raise RuntimeError("baseline path unexpectedly selected fast Gemm")
        evidence["actual_plan_kind"] = "baseline-gemm"
    return evidence


def _generate_one(
    model: str,
    implementation: str,
    output_dir: Path,
    model_dir: Path,
) -> None:
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget
    from ace_edsl.edsl.kernels.vector.baseline_gemm import (
        configure_baseline_gemm_dsl,
    )
    from ace_edsl.edsl.kernels.vector.fast_gemm import fast_gemm_recipe

    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    phase_irs: dict[str, str] = {}

    def capture_phase(phase: str, ir: str) -> None:
        if phase in ("tensor2vector", "vector_kernel_inline"):
            phase_irs[phase] = ir

    pipeline = (
        Pipeline(
            f"{model}_{implementation}",
            output_dir=str(output_dir / "ir"),
            dump_ir=model == "i64_o10",
            verbose=True,
            on_phase_complete=capture_phase,
        )
        .load_onnx(str(model_dir / f"{model}.onnx"))
        .configure_fhe(
            scaling_factor_bits=SCALING_FACTOR_BITS,
            first_prime_bits=FIRST_PRIME_BITS,
            hamming_weight=HAMMING_WEIGHT,
            data_file=f"{model}.weight",
            free_poly=FREE_POLY,
        )
    )
    slots = MODEL_SLOTS[model]
    prepared_plans: list[object] = []
    if implementation in ("native", "cpp-baseline"):
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="baseline-gemm",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
        )
    elif implementation == "dsl":
        configure_baseline_gemm_dsl(
            pipeline,
            mask_fuse=False,
            max_slots=slots,
        )
    elif implementation == "metakernel-fast":
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="fast-gemm",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
        )
    elif implementation == "dsl-fast":

        def capture_recipe(trace, prepared):
            prepared_plans.append(prepared)
            return fast_gemm_recipe(trace, prepared)

        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="dsl",
            plan_kind="fast-gemm",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
        ).register_vector_kernel_recipe("fast-gemm", capture_recipe)
    else:
        raise ValueError(f"unsupported implementation: {implementation}")

    started = time.monotonic()
    result = pipeline.run(target=PipelineTarget.C)
    elapsed = time.monotonic() - started
    if not result.success:
        for phase, ir in phase_irs.items():
            (output_dir / f"{phase}.air").write_text(ir)
        path_evidence = None
        if implementation == "dsl-fast" and "tensor2vector" in phase_irs:
            path_evidence = _path_evidence(
                implementation,
                phase_irs,
                prepared_plans,
                require_inlined=False,
            )
        failure = {
            "success": False,
            "model": model,
            "implementation": implementation,
            "error": result.error,
            "stages": result.stages_completed,
            "generation_seconds": elapsed,
            "path_evidence": path_evidence,
        }
        (output_dir / "generation-failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        raise RuntimeError(result.error)
    uses_inliner = implementation in ("dsl", "dsl-fast")
    expected_stages = [
        "tensor2vector",
        *(["vector_kernel_inline"] if uses_inliner else []),
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

    path_evidence = _path_evidence(implementation, phase_irs, prepared_plans)
    for phase, ir in phase_irs.items():
        (output_dir / f"{phase}.air").write_text(ir)

    generated_c = output_dir / f"{model}.generated.c"
    generated_c.write_text(result.c_code)
    weight = output_dir / f"{model}.weight"
    if not weight.is_file():
        raise RuntimeError(f"compiler did not emit {weight.name}")
    metadata = {
        "model": model,
        "implementation": implementation,
        "max_slots": slots,
        "stages": result.stages_completed,
        "generation_seconds": elapsed,
        "phase_seconds": pipeline.timings,
        "generated_c_bytes": generated_c.stat().st_size,
        "generated_c_sha256": _sha256(generated_c),
        "weight_sha256": _sha256(weight),
        "model_onnx_sha256": _sha256(model_dir / f"{model}.onnx"),
        "driver_sha256": _sha256(model_dir / f"{model}.c"),
        "path_evidence": path_evidence,
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
) -> tuple[Path, dict[str, object]]:
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
    started = time.monotonic()
    result = _run(command, cwd=output_dir, timeout=timeout)
    elapsed = time.monotonic() - started
    (output_dir / "compile.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"compile failed for {model} in {output_dir}\n{result.stdout}"
        )
    metadata: dict[str, object] = {
        "success": True,
        "seconds": elapsed,
        "command": command,
        "runner_sha256": _sha256(runner),
        "executable_bytes": executable.stat().st_size,
    }
    (output_dir / "compile.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return executable, metadata


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
            "OMP_NUM_THREADS": str(OMP_NUM_THREADS),
            "RTLIB_DISABLE_BOOTSTRAP_PRECOM": "1",
            "RTLIB_TIMING_OUTPUT": "stdout",
            "ABS_ERROR": str(ABS_ERROR_TOLERANCE),
            "REL_ERROR": str(REL_ERROR_TOLERANCE),
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
    left_path: Path,
    right_path: Path,
    *,
    tolerance: float,
    left_name: str = "native",
    right_name: str = "DSL",
) -> dict[str, object]:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"output tolerance must be finite and nonnegative: {tolerance}"
        )
    left_values = _read_values(left_path)
    right_values = _read_values(right_path)
    if len(left_values) != len(right_values):
        raise RuntimeError(
            f"{left_name}/{right_name} output length mismatch: "
            f"{len(left_values)} != {len(right_values)}"
        )
    if not left_values:
        raise RuntimeError(f"{left_name}/{right_name} output files are empty")
    for implementation, values in (
        (left_name, left_values),
        (right_name, right_values),
    ):
        for index, value in enumerate(values):
            if not math.isfinite(value):
                raise RuntimeError(
                    f"{implementation} output is non-finite at index {index}: {value}"
                )
    differences = [abs(left - right) for left, right in zip(left_values, right_values)]
    max_difference = max(differences, default=0.0)
    max_index = differences.index(max_difference) if differences else -1
    if not math.isfinite(max_difference) or max_difference > tolerance:
        raise RuntimeError(
            f"{left_name}/{right_name} max absolute difference "
            f"{max_difference} at index {max_index} exceeds {tolerance}"
        )
    return {
        "length": len(left_values),
        "max_abs_difference": max_difference,
        "max_abs_difference_index": max_index,
        "tolerance": tolerance,
    }


def _compare_output_set(
    output_paths: dict[str, Path],
    *,
    tolerance: float,
) -> dict[str, object]:
    pairwise = {}
    for left, right in itertools.combinations(output_paths, 2):
        pairwise[f"{left}__{right}"] = _compare_outputs(
            output_paths[left],
            output_paths[right],
            tolerance=tolerance,
            left_name=left,
            right_name=right,
        )
    if not pairwise:
        raise RuntimeError("cross-output comparison requires two paths")
    worst_name, worst = max(
        pairwise.items(), key=lambda item: item[1]["max_abs_difference"]
    )
    return {
        "pairs": pairwise,
        "max_abs_difference": worst["max_abs_difference"],
        "max_abs_difference_index": worst["max_abs_difference_index"],
        "max_abs_difference_pair": worst_name,
        "length": worst["length"],
        "tolerance": tolerance,
    }


def _balanced_order(implementations: tuple[str, ...], sample: int) -> tuple[str, ...]:
    cycle, offset = divmod(sample, len(implementations))
    base = implementations if cycle % 2 == 0 else tuple(reversed(implementations))
    return base[offset:] + base[:offset]


def _summarize_timings(
    timings: dict[str, list[float]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        implementation: {
            "seconds": values,
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        for implementation, values in timings.items()
    }
    comparisons = {}
    for left, right in itertools.combinations(timings, 2):
        left_median = summary[left]["median_seconds"]
        right_median = summary[right]["median_seconds"]
        ratio = left_median / right_median
        comparisons[f"{left}_over_{right}"] = {
            "runtime_ratio": ratio,
            "gap_percent": (ratio - 1.0) * 100.0,
            "right_speedup_vs_left": ratio,
            "left_speedup_vs_right": 1.0 / ratio,
        }
    summary["comparisons"] = comparisons
    if "native" in timings and "dsl" in timings:
        ratio = summary["dsl"]["median_seconds"] / summary["native"]["median_seconds"]
        summary["dsl_over_native_ratio"] = ratio
        summary["dsl_gap_percent"] = (ratio - 1.0) * 100.0
    return summary


def _benchmark_model(
    script: Path,
    root: Path,
    model: str,
    warmups: int,
    runs: int,
    timeout: int,
    cross_tolerance: float,
    model_dir: Path,
    implementations: tuple[str, ...] = IMPLEMENTATIONS,
) -> dict[str, object]:
    model_root = root / model
    generation = {}
    compilation = {}
    executables = {}
    for implementation in implementations:
        output_dir = model_root / implementation
        output_dir.mkdir(parents=True, exist_ok=True)
        generation[implementation] = _generate_subprocess(
            script, model, implementation, output_dir, timeout, model_dir
        )
        executable, compile_metadata = _compile_runner(
            model, output_dir, timeout, model_dir
        )
        executables[implementation] = executable
        compilation[implementation] = compile_metadata

    for field in ("model_onnx_sha256", "driver_sha256"):
        hashes = {metadata[field] for metadata in generation.values()}
        if len(hashes) != 1:
            raise RuntimeError(f"{model} implementations disagree on {field}: {hashes}")

    validation_samples: dict[str, list[dict[str, object]]] = {
        implementation: [] for implementation in implementations
    }
    warmup_orders = []
    for warmup in range(warmups):
        order = _balanced_order(implementations, warmup)
        warmup_orders.append(list(order))
        for implementation in order:
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

    timings = {implementation: [] for implementation in implementations}
    output_comparisons = []
    measured_orders = []
    for run in range(runs):
        order = _balanced_order(implementations, run)
        measured_orders.append(list(order))
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
        output_comparison = _compare_output_set(
            {
                implementation: model_root / implementation / f"run-{run}.txt"
                for implementation in implementations
            },
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
        "max_abs_difference_pair": worst_output["max_abs_difference_pair"],
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
        "implementations": list(implementations),
        "generation": generation,
        "compilation": compilation,
        "validation": validation,
        "cross_output": output_comparison,
        "performance": performance,
        "run_order": {
            "warmups": warmup_orders,
            "measured": measured_orders,
        },
        "unsupported": [],
        "fallbacks": [],
    }
    if implementations == IMPLEMENTATIONS:
        summary["native_dsl_output"] = output_comparison
    (model_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    medians = "; ".join(
        f"{implementation}={performance[implementation]['median_seconds']:.6f}s"
        for implementation in implementations
    )
    print(
        f"[model] PASS {model}: output max_abs_difference="
        f"{output_comparison['max_abs_difference']:.6g}; {medians}",
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
        "--implementations",
        nargs="+",
        choices=ALL_IMPLEMENTATIONS,
        default=list(IMPLEMENTATIONS),
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
    if len(set(arguments.implementations)) != len(arguments.implementations):
        parser.error("--implementations must not contain duplicates")
    selected = tuple(arguments.implementations)
    if selected != IMPLEMENTATIONS and set(selected) != set(THREE_WAY_IMPLEMENTATIONS):
        parser.error(
            "--implementations must select the default native/DSL pair or "
            "all three fast-comparison paths"
        )
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
        if model not in MODEL_SLOTS or implementation not in ALL_IMPLEMENTATIONS:
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
                tuple(arguments.implementations),
            )
        )
    aggregate = {
        "settings": {
            "models": arguments.models,
            "implementations": arguments.implementations,
            "model_dir": str(model_dir),
            "warmups": arguments.warmups,
            "runs": arguments.runs,
            "timeout_seconds": arguments.timeout,
            "cross_tolerance": arguments.cross_tolerance,
            "compiler_settings": {
                "scaling_factor_bits": SCALING_FACTOR_BITS,
                "first_prime_bits": FIRST_PRIME_BITS,
                "hamming_weight": HAMMING_WEIGHT,
                "free_poly": FREE_POLY,
                "mask_fuse": False,
                "plan_provider": "cpp",
                "fallback": "error",
                "cxx_optimization": "-O3",
                "cxx_openmp": True,
            },
            "runtime_settings": {
                "omp_num_threads": OMP_NUM_THREADS,
                "rtlib_disable_bootstrap_precom": 1,
                "abs_error_tolerance": ABS_ERROR_TOLERANCE,
                "rel_error_tolerance": REL_ERROR_TOLERANCE,
            },
            "input_identity": (
                "model and driver SHA-256 values are checked equal across "
                "implementations for each model"
            ),
        },
        "models": summaries,
        "unsupported": [],
        "fallbacks": [],
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("GEMM_E2E_RESULT=PASS", flush=True)
    print(f"SUMMARY={root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
