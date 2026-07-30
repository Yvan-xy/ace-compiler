#!/usr/bin/env python3
"""Compile, validate, and benchmark selected Conv lowering paths."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from ace_edsl.tests import gemm_e2e_compare as _shared


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "model_and_main"
_ISOLATED_CONV_MODELS = (
    "i16x32x32_ci16_co16_k3p1s1",
    "i16x32x32_ci16_co32_k1p0s2",
    "i16x32x32_ci16_co32_k3p1s2",
    "i32x16x16_ci32_co32_k3p1s1",
    "i32x16x16_ci32_co64_k1p0s2",
    "i32x16x16_ci32_co64_k3p1s2",
    "i64x16x16_ci64_co128_k3p1s1",
    "i64x8x8_ci64_co64_k3p1s1",
)
MODEL_SLOTS = {model: 32768 for model in _ISOLATED_CONV_MODELS}
MODEL_POLY_DEGREES = {model: 65536 for model in _ISOLATED_CONV_MODELS}
IMPLEMENTATIONS = ("native", "dsl")
THREE_WAY_IMPLEMENTATIONS = (
    "dsl-fast",
    "cpp-baseline",
    "metakernel-fast",
)
ALL_IMPLEMENTATIONS = IMPLEMENTATIONS + THREE_WAY_IMPLEMENTATIONS
INLINER_IMPLEMENTATION = "tentative-binding-transition"
_CONV_DRIVER_END = """\
  Finalize_context();


  return 0;
}
"""
_CONV_COMPACT_DRIVER_END = """\
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
    _shared._write_validating_driver = _write_validating_driver


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


def _unsupported_conv_models(model_dir: Path) -> list[dict[str, object]]:
    import onnx

    unsupported = []
    for model_path in sorted(model_dir.glob("*.onnx")):
        try:
            model = onnx.load(model_path, load_external_data=False)
        except Exception as error:
            unsupported.append(
                {
                    "model": model_path.stem,
                    "conv_nodes": None,
                    "operations": {},
                    "reason": (
                        "ONNX fixture cannot be inspected: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            )
            continue
        operation_counts: dict[str, int] = {}
        for node in model.graph.node:
            operation_counts[node.op_type] = (
                operation_counts.get(node.op_type, 0) + 1
            )
        conv_count = operation_counts.get("Conv", 0)
        if conv_count == 0 or model_path.stem in MODEL_SLOTS:
            continue
        reasons = []
        if conv_count != 1 or set(operation_counts) != {"Conv"}:
            reasons.append(
                "mixed-network fixture is not an isolated Conv runtime case"
            )
        if not model_path.with_suffix(".c").is_file():
            reasons.append("matching correctness driver is missing")
        if not reasons:
            reasons.append("slot/poly-degree configuration is not frozen")
        unsupported.append(
            {
                "model": model_path.stem,
                "conv_nodes": conv_count,
                "operations": operation_counts,
                "reason": "; ".join(reasons),
            }
        )
    return unsupported


def _expected_fast_conv_rnums(prepared: object) -> list[list[int]]:
    values = [
        [-prepared.input_size * copy]
        for copy in range(1, prepared.input_duplications)
    ]
    values.extend(list(rotation.candidates) for rotation in prepared.rotations)
    return values


def _generated_helper_ir(ir: str, helper_name: str) -> str:
    pattern = re.compile(
        rf'^FUN\[[^\n]*"{re.escape(helper_name)}"\n.*?(?=^FUN\[|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    matches = pattern.findall(ir)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one AIR definition region for {helper_name}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _sequence_offsets(values: list[object], expected: list[object]) -> list[int]:
    if not expected:
        return []
    width = len(expected)
    return [
        offset
        for offset in range(len(values) - width + 1)
        if values[offset : offset + width] == expected
    ]


def _prepared_evidence(prepared: object) -> dict[str, object]:
    return {
        "actual_plan_kind": prepared.kind,
        "plan_provenance": prepared.provenance,
        "specialization_key": prepared.specialization_key,
        "helper_name": prepared.helper_name,
        "dimensions": [
            prepared.channel_in,
            prepared.channel_out,
            prepared.output_height,
            prepared.output_width,
            prepared.kernel_hw,
            prepared.group,
            prepared.stride,
        ],
        "blocking": [
            prepared.input_size,
            prepared.output_size,
            prepared.num_slots,
            prepared.num_grid,
            prepared.num_block,
            prepared.width_block,
            prepared.width_block_data,
            prepared.width_block_pad,
            prepared.position_block,
            prepared.capacity_block,
            prepared.input_duplications,
            prepared.blocking_outer_depth,
        ],
        "cyclic_roll": prepared.cyclic_roll,
        "sharding_offset": (
            None
            if prepared.sharding_offset is None
            else {
                "type": prepared.sharding_offset.type,
                "scale": prepared.sharding_offset.scale,
            }
        ),
        "runtime_vector_inputs": [
            [item.element_type, list(item.shape)]
            for item in prepared.runtime_vector_inputs
        ],
        "runtime_scalar_inputs": list(prepared.runtime_scalar_inputs),
        "loops": [
            [item.role, item.lower, item.upper, item.step, item.nesting_depth]
            for item in prepared.loops
        ],
        "slices": [
            [
                item.role,
                list(item.index.iv_coefficients),
                item.index.constant,
                item.index.uses_sharding_offset,
                item.width,
            ]
            for item in prepared.slices
        ],
        "rotation_candidates": {
            item.role: list(item.candidates) for item in prepared.rotations
        },
        "reductions": [
            [
                item.role,
                item.kind,
                item.factor,
                item.block_width,
                item.padding,
            ]
            for item in prepared.reductions
        ],
        "mask": [prepared.mask.policy, prepared.mask.valid_length],
        "constants": [
            [
                item.role,
                item.type.element_type,
                list(item.type.shape),
                item.content_hash,
            ]
            for item in prepared.constants
        ],
        "runtime_preparations": [
            [
                item.role,
                item.source_operand,
                item.kind,
                item.result_type.element_type,
                list(item.result_type.shape),
                item.logical_input_size,
                item.replications,
                item.blocking_width,
                list(item.rotation_candidates),
                item.outer_block_depth,
            ]
            for item in prepared.runtime_preparations
        ],
        "scalar_preparations": [
            [item.role, item.source_operand, item.type, item.scale]
            for item in prepared.scalar_preparations
        ],
        "result_type": [
            prepared.result_type.element_type,
            list(prepared.result_type.shape),
        ],
        "slot": [prepared.slot.policy, prepared.slot.value],
    }


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
    if re.search(r"\bNN\.conv\b", tensor_ir, re.IGNORECASE):
        raise RuntimeError("Tensor-to-Vector path left an NN Conv behind")

    if implementation == "dsl-fast":
        kernel_impl = "dsl"
        requested_plan_kind = "fast-conv"
    elif implementation == "metakernel-fast":
        kernel_impl = "native"
        requested_plan_kind = "fast-conv"
    else:
        kernel_impl = "dsl" if implementation == "dsl" else "native"
        requested_plan_kind = "baseline-conv"

    blocking_markers = {
        "replication": "BlockingRot replicate=" in tensor_ir,
        "alignment": "BlockingRot HRot=bs=" in tensor_ir,
    }
    evidence: dict[str, object] = {
        "requested_implementation": implementation,
        "requested_plan_provider": "cpp",
        "requested_kernel_impl": kernel_impl,
        "requested_plan_kind": requested_plan_kind,
        "requested_fallback": "error",
        "source_conv_removed": True,
        "fallback_used": False,
        "blocking_markers": blocking_markers,
    }
    if implementation == "dsl-fast":
        if len(prepared_plans) != 1:
            raise RuntimeError(
                "DSL-fast path requires exactly one captured prepared plan"
            )
        prepared = prepared_plans[0]
        if prepared.kind != "fast-conv" or prepared.provenance != "cpp":
            raise RuntimeError(
                "explicit DSL-fast path did not receive the C++ fast-Conv plan"
            )
        if not re.fullmatch(
            r"__ace_vkernel_fast_conv_[0-9a-f]{64}", prepared.helper_name
        ):
            raise RuntimeError("DSL-fast helper name is not canonical")
        definitions, calls = _shared._helper_records(
            tensor_ir, prepared.helper_name
        )
        if definitions != 1 or calls != 1:
            raise RuntimeError(
                "DSL-fast path lacks one specialized helper and typed call"
            )
        helper_ir = _generated_helper_ir(tensor_ir, prepared.helper_name)
        blocking_markers = {
            "replication": "BlockingRot replicate=" in helper_ir,
            "alignment": "BlockingRot HRot=bs=" in helper_ir,
        }
        evidence["blocking_markers"] = blocking_markers
        if not all(blocking_markers.values()):
            raise RuntimeError("DSL-fast IR lacks the fast blocking topology")
        rnums = _shared._rotation_numbers(helper_ir)
        expected_rnums = _expected_fast_conv_rnums(prepared)
        if rnums != expected_rnums:
            raise RuntimeError(
                "DSL-fast rotation attributes do not match the frozen plan: "
                f"observed={rnums!r}, expected={expected_rnums!r}"
            )
        inlined_ir = phase_irs.get("vector_kernel_inline", "")
        post_definitions, post_calls = (
            _shared._helper_records(inlined_ir, prepared.helper_name)
            if inlined_ir
            else (None, None)
        )
        post_rnums = (
            _shared._rotation_numbers(inlined_ir) if inlined_ir else []
        )
        post_rnum_offsets = _sequence_offsets(post_rnums, expected_rnums)
        post_blocking_markers = {
            "replication": "BlockingRot replicate=" in inlined_ir,
            "alignment": "BlockingRot HRot=bs=" in inlined_ir,
        }
        inline_success = (
            bool(inlined_ir)
            and post_definitions == post_calls == 0
            and len(post_rnum_offsets) == 1
            and all(post_blocking_markers.values())
        )
        if require_inlined and not inline_success:
            raise RuntimeError(
                "DSL-fast helper removal or frozen topology preservation "
                "failed during inlining"
            )
        evidence.update(_prepared_evidence(prepared))
        evidence.update(
            {
                "prepared_callback_count": len(prepared_plans),
                "pre_inline_helper_definitions": definitions,
                "pre_inline_helper_calls": calls,
                "inline_success": inline_success,
                "post_inline_helper_definitions": post_definitions,
                "post_inline_helper_calls": post_calls,
                "post_inline_helper_name_occurrences": (
                    inlined_ir.count(prepared.helper_name)
                    if inlined_ir
                    else None
                ),
                "post_inline_blocking_markers": post_blocking_markers,
                "post_inline_rotation_rnums": post_rnums,
                "post_inline_plan_rnum_offset": (
                    post_rnum_offsets[0]
                    if len(post_rnum_offsets) == 1
                    else None
                ),
                "rotation_rnums": rnums,
            }
        )
    elif implementation == "metakernel-fast":
        if not all(blocking_markers.values()):
            raise RuntimeError("forced native-fast IR lacks fast Conv markers")
        evidence["actual_plan_kind"] = "fast-conv"
    else:
        if any(blocking_markers.values()):
            raise RuntimeError("baseline path unexpectedly selected fast Conv")
        evidence["actual_plan_kind"] = "baseline-conv"
    return evidence


def _generate_one(
    model: str,
    implementation: str,
    output_dir: Path,
    model_dir: Path,
) -> None:
    from ace_edsl.edsl.kernels.vector.baseline_conv import (
        configure_baseline_conv_dsl,
    )
    from ace_edsl.edsl.kernels.vector.fast_conv import fast_conv_recipe
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

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
            dump_ir=False,
            verbose=True,
            on_phase_complete=capture_phase,
        )
        .load_onnx(str(model_dir / f"{model}.onnx"))
        .configure_fhe(
            poly_degree=MODEL_POLY_DEGREES[model],
            scaling_factor_bits=_shared.SCALING_FACTOR_BITS,
            first_prime_bits=_shared.FIRST_PRIME_BITS,
            hamming_weight=_shared.HAMMING_WEIGHT,
            data_file=f"{model}.weight",
            free_poly=_shared.FREE_POLY,
        )
    )
    slots = MODEL_SLOTS[model]
    prepared_plans: list[object] = []
    if implementation in ("native", "cpp-baseline"):
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="baseline-conv",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
            conv_parallel=False,
            sharding=False,
        )
    elif implementation == "dsl":
        configure_baseline_conv_dsl(pipeline, max_slots=slots)
    elif implementation == "metakernel-fast":
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="native",
            plan_kind="fast-conv",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
            conv_parallel=False,
            sharding=False,
        )
    elif implementation == "dsl-fast":

        def capture_recipe(trace, prepared):
            prepared_plans.append(prepared)
            return fast_conv_recipe(trace, prepared)

        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl="dsl",
            plan_kind="fast-conv",
            fallback="error",
            mask_fuse=False,
            max_slots=slots,
            conv_parallel=False,
            sharding=False,
        ).register_vector_kernel_recipe("fast-conv", capture_recipe)
    else:
        raise ValueError(f"unsupported implementation: {implementation}")

    started = time.monotonic()
    result = pipeline.run(target=PipelineTarget.C)
    elapsed = time.monotonic() - started
    if not result.success:
        for phase, ir in phase_irs.items():
            (output_dir / f"{phase}.air").write_text(ir)
        partial_evidence = None
        evidence_error = None
        if "tensor2vector" in phase_irs:
            try:
                partial_evidence = _path_evidence(
                    implementation,
                    phase_irs,
                    prepared_plans,
                    require_inlined=False,
                )
            except Exception as error:
                evidence_error = str(error)
        failure = {
            "success": False,
            "model": model,
            "implementation": implementation,
            "error": result.error,
            "stages": result.stages_completed,
            "generation_seconds": elapsed,
            "path_evidence": partial_evidence,
            "path_evidence_error": evidence_error,
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

    for phase, ir in phase_irs.items():
        (output_dir / f"{phase}.air").write_text(ir)
    path_evidence = _path_evidence(
        implementation, phase_irs, prepared_plans
    )

    generated_c = output_dir / f"{model}.generated.c"
    generated_c.write_text(result.c_code)
    weight = output_dir / f"{model}.weight"
    if not weight.is_file():
        raise RuntimeError(f"compiler did not emit {weight.name}")
    metadata = {
        "success": True,
        "model": model,
        "implementation": implementation,
        "inliner_impl": (
            INLINER_IMPLEMENTATION if uses_inliner else None
        ),
        "max_slots": slots,
        "poly_degree": MODEL_POLY_DEGREES[model],
        "stages": result.stages_completed,
        "generation_seconds": elapsed,
        "phase_seconds": pipeline.timings,
        "generated_c_bytes": generated_c.stat().st_size,
        "generated_c_sha256": _shared._sha256(generated_c),
        "weight_sha256": _shared._sha256(weight),
        "model_onnx_sha256": _shared._sha256(model_dir / f"{model}.onnx"),
        "driver_sha256": _shared._sha256(model_dir / f"{model}.c"),
        "path_evidence": path_evidence,
    }
    (output_dir / "generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        "GENERATION_RESULT="
        + json.dumps(metadata, sort_keys=True, allow_nan=False),
        flush=True,
    )


def _write_validating_driver(
    model: str,
    output_dir: Path,
    model_dir: Path,
) -> Path:
    source = (model_dir / f"{model}.c").read_text()
    matches = [
        ending
        for ending in (_CONV_DRIVER_END, _CONV_COMPACT_DRIVER_END)
        if source.count(ending) == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot identify terminal block in {model}.c")
    driver = output_dir / f"{model}.main.inc"
    driver.write_text(source.replace(matches[0], _CONV_VALIDATING_DRIVER_END))
    return driver


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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ace-conv-e2e"),
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
    if len(set(arguments.implementations)) != len(arguments.implementations):
        parser.error("--implementations must not contain duplicates")
    selected = tuple(arguments.implementations)
    if selected != IMPLEMENTATIONS and set(selected) != set(
        THREE_WAY_IMPLEMENTATIONS
    ):
        parser.error(
            "--implementations must select the default native/DSL pair or "
            "all three fast-comparison paths"
        )
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
    unsupported = _unsupported_conv_models(model_dir)
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
                "poly_degree": {
                    model: MODEL_POLY_DEGREES[model]
                    for model in arguments.models
                },
                "max_slots": {
                    model: MODEL_SLOTS[model] for model in arguments.models
                },
                "scaling_factor_bits": _shared.SCALING_FACTOR_BITS,
                "first_prime_bits": _shared.FIRST_PRIME_BITS,
                "hamming_weight": _shared.HAMMING_WEIGHT,
                "free_poly": _shared.FREE_POLY,
                "mask_fuse": False,
                "conv_parallel": False,
                "sharding": False,
                "plan_provider": "cpp",
                "fallback": "error",
                "cxx_optimization": "-O3",
                "cxx_openmp": True,
            },
            "runtime_settings": {
                "omp_num_threads": _shared.OMP_NUM_THREADS,
                "rtlib_disable_bootstrap_precom": 1,
                "abs_error_tolerance": _shared.ABS_ERROR_TOLERANCE,
                "rel_error_tolerance": _shared.REL_ERROR_TOLERANCE,
            },
            "input_identity": (
                "model and driver SHA-256 values are checked equal across "
                "implementations for each model"
            ),
            "inliner_impl": INLINER_IMPLEMENTATION,
        },
        "models": summaries,
        "unsupported": unsupported,
        "fallbacks": [],
    }
    (root / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("CONV_E2E_RESULT=PASS", flush=True)
    print(f"SUMMARY={root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
