#!/usr/bin/env python3
"""Compile, validate, and benchmark transition native/DSL baseline Conv."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

from ace_edsl.tests import gemm_e2e_compare as _shared


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "model_and_main"
MODEL_SLOTS = {"i16x32x32_ci16_co16_k3p1s1": 32768}
MODEL_POLY_DEGREES = {"i16x32x32_ci16_co16_k3p1s1": 65536}
IMPLEMENTATIONS = ("native", "dsl")
INLINER_IMPLEMENTATION = "tentative-binding-transition"
_CONV_DRIVER_END = """\
  Finalize_context();


  return 0;
}
"""
_CONV_VALIDATING_DRIVER_END = _shared.VALIDATING_DRIVER_END.replace(
    "GEMM", "CONV"
)
_CONV_RESULT_PATTERN = re.compile(
    r"^CONV_RESULT=(PASS|FAIL)\s+max_abs=([^\s]+)"
    r"\s+max_abs_index=(-?\d+)\s+max_rel=([^\s]+)$",
    re.MULTILINE,
)


def _specialize_shared_harness() -> None:
    """Specialize the process-local generic mechanics for Conv output."""
    _shared.MODEL_SLOTS = MODEL_SLOTS
    _shared.IMPLEMENTATIONS = IMPLEMENTATIONS
    _shared.DRIVER_END = _CONV_DRIVER_END
    _shared.VALIDATING_DRIVER_END = _CONV_VALIDATING_DRIVER_END
    _shared.RESULT_PATTERN = _CONV_RESULT_PATTERN


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
            "missing local Conv model fixtures:\n"
            f"{formatted}\n"
            "Pass their directory with --model-dir."
        )


def _generate_one(
    model: str,
    implementation: str,
    output_dir: Path,
    model_dir: Path,
) -> None:
    from ace_edsl.edsl.kernels.vector.baseline_conv import (
        configure_baseline_conv_dsl,
    )
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = (
        Pipeline(
            f"{model}_{implementation}",
            output_dir=str(output_dir / "ir"),
            dump_ir=True,
            verbose=True,
        )
        .load_onnx(str(model_dir / f"{model}.onnx"))
        .configure_fhe(
            poly_degree=MODEL_POLY_DEGREES[model],
            scaling_factor_bits=56,
            first_prime_bits=60,
            hamming_weight=192,
            data_file=str(output_dir / f"{model}.weight"),
            free_poly=True,
        )
    )
    if implementation == "native":
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="baseline-conv",
            fallback="error",
            mask_fuse=False,
            max_slots=MODEL_SLOTS[model],
        )
    else:
        configure_baseline_conv_dsl(
            pipeline,
            max_slots=MODEL_SLOTS[model],
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
        "inliner_impl": (
            INLINER_IMPLEMENTATION if implementation == "dsl" else None
        ),
        "max_slots": MODEL_SLOTS[model],
        "poly_degree": MODEL_POLY_DEGREES[model],
        "stages": result.stages_completed,
        "generation_seconds": elapsed,
        "phase_seconds": pipeline.timings,
        "generated_c_bytes": generated_c.stat().st_size,
        "model_onnx_sha256": _shared._sha256(model_dir / f"{model}.onnx"),
        "driver_sha256": _shared._sha256(model_dir / f"{model}.c"),
    }
    (output_dir / "generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "GENERATION_RESULT="
        + json.dumps(metadata, sort_keys=True, allow_nan=False),
        flush=True,
    )


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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ace-conv-transition-e2e"),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=14400)
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
    if (
        not math.isfinite(arguments.cross_tolerance)
        or arguments.cross_tolerance < 0.0
    ):
        parser.error("--cross-tolerance must be finite and nonnegative")
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    _specialize_shared_harness()
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
            _shared._benchmark_model(
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
            "inliner_impl": INLINER_IMPLEMENTATION,
        },
        "models": summaries,
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("CONV_E2E_RESULT=PASS", flush=True)
    print(f"SUMMARY={root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
