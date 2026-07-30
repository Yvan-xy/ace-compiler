"""Fast-Gemm destination-recipe integration and AIR parity tests."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


_WORKER_FLAG = "--fast-gemm-vector-kernel-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_SLOTS = 128

_CASES = (
    {
        "name": "single-partition-mask",
        "n": 4,
        "k": 4,
        "mask_fuse": False,
        "dimensions": [4, 4, 4, 4, 4, 4],
        "blocking": [2, 2, 1, 2, 2],
        "result_shape": [8],
        "rotations": [
            ["input-duplication", [-4]],
            ["blocking-alignment", [0, 1]],
            ["grid", [0, 2]],
        ],
        "reductions": [],
        "mask": ["clear-valid-prefix", 4],
        "stats": [3, 3, 4, 2],
    },
    {
        "name": "single-partition-no-mask",
        "n": 4,
        "k": 4,
        "mask_fuse": True,
        "dimensions": [4, 4, 4, 4, 4, 4],
        "blocking": [2, 2, 1, 2, 2],
        "result_shape": [8],
        "rotations": [
            ["input-duplication", [-4]],
            ["blocking-alignment", [0, 1]],
            ["grid", [0, 2]],
        ],
        "reductions": [],
        "mask": ["none", 0],
        "stats": [3, 3, 4, 1],
    },
    {
        "name": "packed-partitions",
        "n": 8,
        "k": 8,
        "mask_fuse": False,
        "dimensions": [8, 8, 8, 8, 8, 8],
        "blocking": [2, 4, 2, 2, 3],
        "result_shape": [24],
        "rotations": [
            ["input-duplication", [-8, -16]],
            ["blocking-alignment", [0, 1]],
            ["grid", [0, 2]],
            ["packed-partitions", [12]],
        ],
        "reductions": [["packed-partitions", "power-of-two", 2, 8, 4]],
        "mask": ["clear-valid-prefix", 8],
        "stats": [4, 5, 6, 2],
    },
    {
        "name": "kp-over-np-stride-one",
        "n": 1,
        "k": 4,
        "mask_fuse": False,
        "dimensions": [1, 4, 1, 4, 1, 4],
        "blocking": [1, 1, 1, 1, 2],
        "result_shape": [5],
        "rotations": [
            ["input-duplication", [-4]],
            ["blocking-alignment", [0]],
            ["grid", [0]],
            ["kp-over-np", [1, 2]],
        ],
        "reductions": [["kp-over-np", "power-of-two", 4, 1, 0]],
        "mask": ["clear-valid-prefix", 1],
        "stats": [4, 4, 5, 2],
    },
    {
        "name": "multiple-input-replications",
        "n": 8,
        "k": 2,
        "mask_fuse": False,
        "dimensions": [8, 2, 8, 2, 2, 8],
        "blocking": [1, 2, 1, 2, 5],
        "result_shape": [10],
        "rotations": [
            ["input-duplication", [-2, -4, -6, -8]],
            ["blocking-alignment", [0]],
            ["grid", [0, 1]],
        ],
        "reductions": [],
        "mask": ["clear-valid-prefix", 8],
        "stats": [3, 6, 7, 2],
    },
)


def _write_gemm_model(path: Path, n: int, k: int, copies: int = 1):
    values = (np.arange(n * k, dtype=np.float32) + 1.0) / 8.0
    weight = numpy_helper.from_array(values.reshape(n, k), "weight")
    bias = numpy_helper.from_array(np.arange(n, dtype=np.float32) / 4.0, "bias")
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, k])
    nodes = []
    outputs = []
    for index in range(copies):
        output = "output" if copies == 1 else f"gemm_output_{index}"
        outputs.append(output)
        nodes.append(
            helper.make_node(
                "Gemm",
                ["input", "weight", "bias"],
                [output],
                name=f"gemm_{index}",
                transB=1,
            )
        )
    if copies == 1:
        output = outputs[0]
    else:
        output = "output"
        nodes.append(helper.make_node("Add", outputs, [output], name="combine"))
    output_info = helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, n])
    graph = helper.make_graph(
        nodes,
        "fast_gemm",
        [input_info],
        [output_info],
        [weight, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run_worker(
    model: Path,
    implementation: str,
    mask_fuse: bool,
    behavior: str = "normal",
    allow_process_failure: bool = False,
):
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(_REPO_ROOT), old_pythonpath))
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
        str(model),
        implementation,
        "1" if mask_fuse else "0",
        behavior,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if allow_process_failure and completed.returncode != 0:
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    assert completed.returncode == 0, (
        f"worker failed ({' '.join(command)}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    return json.loads(lines[-1])


def _expected_rnums(rotations):
    values = []
    for role, candidates in rotations:
        if role == "input-duplication":
            values.extend([[candidate] for candidate in candidates])
        else:
            values.append(candidates)
    return values


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_cpp_plan_dsl_fast_gemm_matches_native_air(tmp_path, case):
    model = tmp_path / f"{case['name']}.onnx"
    _write_gemm_model(model, case["n"], case["k"])

    native = _run_worker(model, "native", case["mask_fuse"])
    dsl = _run_worker(model, "dsl", case["mask_fuse"])

    assert native["success"] and dsl["success"]
    assert native["verify"] and dsl["verify"]
    assert native["stages_completed"] == dsl["stages_completed"] == ["tensor2vector"]
    assert not native["has_nn_gemm"] and not dsl["has_nn_gemm"]
    assert native["helper_count"] == native["helper_calls"] == 0
    assert dsl["helper_count"] == dsl["helper_calls"] == 1
    assert dsl["object_oracle"]["kind"] == "helper"
    assert native["object_oracle"]["kind"] == "native"
    assert dsl["object_oracle"]["bridge_ok"]
    assert dsl["object_oracle"]["prepared_input_ok"]

    from ace_bindings import air_builder

    direct_comparison = air_builder._compare_normalized_vector_kernel_air_for_testing(
        dsl["object_oracle"]["native_normalized"],
        dsl["object_oracle"]["normalized"],
    )
    pipeline_comparison = air_builder._compare_normalized_vector_kernel_air_for_testing(
        native["object_oracle"]["normalized"],
        dsl["object_oracle"]["normalized"],
    )
    assert direct_comparison["equal"], direct_comparison
    assert pipeline_comparison["equal"], pipeline_comparison
    assert native["stats"] == dsl["stats"]
    assert [dsl["stats"][name] for name in ("loops", "rolls", "adds", "muls")] == case[
        "stats"
    ]
    assert dsl["stats"]["zeros"] == 2
    assert dsl["stats"]["slices"] == 1
    assert dsl["stats"]["indexed_stores"] == 1
    assert dsl["stats"]["indexed_loads"] == 2

    plan = dsl["plan"]
    assert plan["kind"] == "fast-gemm"
    assert plan["provenance"] == "cpp"
    assert plan["dimensions"] == case["dimensions"]
    assert plan["blocking"] == case["blocking"]
    assert plan["runtime_vector_inputs"] == [["f32", [1, case["k"]]]]
    assert plan["runtime_scalar_inputs"] == []
    assert plan["scalar_preparations"] == []
    assert plan["result_type"] == ["f32", case["result_shape"]]
    assert plan["loops"] == [
        ["grid", 0, case["blocking"][3], 1, 0],
        ["block", 0, case["blocking"][0], 1, 1],
    ]
    assert plan["slice"] == [
        "weight",
        [case["blocking"][0], 1],
        0,
        False,
        case["result_shape"][0],
    ]
    assert plan["rotations"] == case["rotations"]
    assert plan["reductions"] == case["reductions"]
    assert plan["mask"] == case["mask"]
    assert plan["slot"] == ["logical-output-elements", case["n"]]
    assert plan["runtime_preparations"] == [
        [
            "input",
            0,
            "blocking-rotations",
            "f32",
            [1, case["k"]],
            case["dimensions"][3],
            case["blocking"][4],
            case["blocking"][0],
            case["rotations"][1][1],
            0,
        ]
    ]
    assert plan["immutable"] and plan["owned_constants"]
    assert re.fullmatch(r"__ace_vkernel_fast_gemm_[0-9a-f]{64}", plan["helper_name"])
    assert dsl["helper_names"] == [plan["helper_name"]]

    constants = {record[0]: record for record in plan["constants"]}
    expected_roles = {"weight", "bias", "rotation-table"}
    if case["mask"][0] != "none":
        expected_roles.add("mask")
    assert set(constants) == expected_roles
    assert constants["weight"][1:3] == [
        "f32",
        [case["blocking"][3] * case["blocking"][0], case["result_shape"][0]],
    ]
    assert constants["rotation-table"][1:3] == ["s32", [case["blocking"][0]]]
    assert all(
        record[3] == 4 * int(np.prod(record[2])) for record in constants.values()
    )
    assert all(record[4].startswith("sha256:") for record in constants.values())
    assert sorted(dsl["stats"]["constant_hashes"]) == sorted(
        record[4] for record in constants.values()
    )
    assert constants["weight"][4] in dsl["stats"]["constant_hashes"]

    assert dsl["stats"]["rnums"] == _expected_rnums(case["rotations"])
    assert dsl["stats"]["slot_values"] == [case["n"]]
    assert dsl["stats"]["comments"] == [
        f"BlockingRot replicate={case['blocking'][4]}",
        f"BlockingRot HRot=bs={case['blocking'][0]}",
        f"IMRA Metakernel: gs={case['blocking'][3]}",
        f"gemm result reduce->Ps={case['blocking'][2]}",
        f"gemm result reduce->(kp/np)={case['dimensions'][3] // case['dimensions'][2]}",
        "gemm add bias",
    ]


def test_equal_plans_share_one_helper_and_two_typed_calls(tmp_path):
    model = tmp_path / "repeated.onnx"
    _write_gemm_model(model, 4, 4, copies=2)

    result = _run_worker(model, "dsl", False)

    assert result["success"] and result["verify"]
    assert result["stages_completed"] == ["tensor2vector"]
    assert not result["has_nn_gemm"]
    assert result["callback_count"] == 1
    assert result["helper_count"] == 1
    assert result["helper_calls"] == 2
    assert result["helper_names"] == [result["plan"]["helper_name"]]


@pytest.mark.parametrize(
    "n,k,behavior,diagnostic",
    [
        (4, 4, "wrong-kind", "another plan kind"),
        (4, 4, "wrong-slot", "SLOT does not match"),
        (4, 4, "missing-weight", "weight"),
        (4, 4, "wrong-loop", "loop topology"),
        (4, 4, "wrong-slice", "weight slice"),
        (4, 4, "missing-duplication", "rotation topology"),
        (4, 4, "wrong-grid-rotation", "rotation topology"),
        (4, 4, "wrong-mask", "mask length"),
        (4, 4, "wrong-preparation", "input preparation"),
        (8, 8, "missing-reduction", "reduction topology"),
        (8, 8, "wrong-reduction", "reduction topology"),
        (4, 4, "post-emission", "post-emission failure"),
    ],
)
def test_invalid_or_failed_recipe_never_falls_back_after_callback(
    tmp_path, n, k, behavior, diagnostic
):
    model = tmp_path / f"{behavior}.onnx"
    _write_gemm_model(model, n, k)

    result = _run_worker(model, "dsl", False, behavior)

    assert not result["success"]
    assert diagnostic in result["error"]
    assert result["callback_count"] == 1


def test_missing_fast_recipe_obeys_error_and_explicit_fallback(tmp_path):
    model = tmp_path / "missing.onnx"
    _write_gemm_model(model, 4, 4)

    failed = _run_worker(model, "missing-error", False, allow_process_failure=True)
    fallback = _run_worker(model, "missing-fallback", False)

    assert failed["returncode"] != 0
    assert "no DSL recipe is registered for the validated plan kind" in failed["stderr"]
    assert fallback["success"] and fallback["verify"]
    assert fallback["callback_count"] == 0
    assert fallback["helper_count"] == fallback["helper_calls"] == 0
    assert fallback["object_oracle"]["kind"] == "native"


def test_fast_gemm_exports_and_configuration_are_canonical():
    import ace_edsl.edsl as edsl
    from ace_edsl.edsl.kernels.vector.fast_gemm import (
        configure_fast_gemm_dsl as canonical_configure,
        fast_gemm_recipe as canonical_recipe,
        fast_gemm_vector_kernel as canonical_kernel,
    )
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.vector_kernel_fast_gemm import (
        configure_fast_gemm_dsl as compatibility_configure,
        fast_gemm_recipe as compatibility_recipe,
        fast_gemm_vector_kernel as compatibility_kernel,
    )

    assert edsl.fast_gemm_recipe is canonical_recipe
    assert edsl.fast_gemm_vector_kernel is canonical_kernel
    assert edsl.configure_fast_gemm_dsl is canonical_configure
    assert compatibility_recipe is canonical_recipe
    assert compatibility_kernel is canonical_kernel
    assert compatibility_configure is canonical_configure

    pipeline = Pipeline("fast-gemm-config", dump_ir=False, verbose=False)
    configured = canonical_configure(pipeline, mask_fuse=True, max_slots=_MAX_SLOTS)
    assert configured is pipeline
    assert pipeline.vector_kernel_config.plan_provider == "cpp"
    assert pipeline.vector_kernel_config.kernel_impl == "dsl"
    assert pipeline.vector_kernel_config.plan_kind == "fast-gemm"
    assert pipeline.vector_kernel_config.fallback == "error"
    assert pipeline.vector_kernel_config.mask_fuse
    assert pipeline.vector_kernel_config.max_slots == _MAX_SLOTS
    assert pipeline.vector_kernel_recipes == {"fast-gemm": canonical_recipe}


def _parse_rnums(dump: str):
    values = []
    for raw in re.findall(r"ATTR\[nums=([^]]+)\]", dump):
        text = raw.strip("()")
        values.append([int(value) for value in text.split(",")])
    return values


def _normalized_stats(normalized: str, dump: str):
    return {
        "loops": normalized.count("{CORE.do_loop,"),
        "rolls": normalized.count("{VECTOR.roll,"),
        "adds": normalized.count("{VECTOR.add,"),
        "muls": normalized.count("{VECTOR.mul,"),
        "zeros": normalized.count("{CORE.zero,"),
        "slices": normalized.count("{VECTOR.slice,"),
        "indexed_stores": normalized.count("{CORE.ist,"),
        "indexed_loads": normalized.count("{CORE.ild,"),
        "rnums": _parse_rnums(dump),
        "slot_values": [
            int.from_bytes(bytes.fromhex(value), "little")
            for value in re.findall(r"slot:u32:1:([0-9a-f]{8})", normalized)
        ],
        "comments": re.findall(
            r"\{CORE\.comment,comment=([^,]*),children=\[\]\}",
            normalized,
        ),
        "constant_hashes": re.findall(r"hash=(sha256:[0-9a-f]{64})", normalized),
    }


def _plan_summary(prepared):
    weight_slice = prepared.slice("weight")
    runtime_input = prepared.runtime_preparation("input")
    immutable = False
    try:
        prepared.kind = "changed"
    except dataclasses.FrozenInstanceError:
        immutable = True
    return {
        "kind": prepared.kind,
        "provenance": prepared.provenance,
        "specialization_key": prepared.specialization_key,
        "helper_name": prepared.helper_name,
        "dimensions": [
            prepared.n,
            prepared.k,
            prepared.np,
            prepared.kp,
            prepared.nd,
            prepared.kd,
        ],
        "blocking": [
            prepared.block_size,
            prepared.blocks_per_partition,
            prepared.packed_partitions,
            prepared.grid_size,
            prepared.input_replications,
        ],
        "runtime_vector_inputs": [
            [item.element_type, list(item.shape)]
            for item in prepared.runtime_vector_inputs
        ],
        "runtime_scalar_inputs": list(prepared.runtime_scalar_inputs),
        "result_type": [
            prepared.result_type.element_type,
            list(prepared.result_type.shape),
        ],
        "loops": [
            [
                item.role,
                item.lower,
                item.upper,
                item.step,
                item.nesting_depth,
            ]
            for item in prepared.loops
        ],
        "slice": [
            weight_slice.role,
            list(weight_slice.index.iv_coefficients),
            weight_slice.index.constant,
            weight_slice.index.uses_sharding_offset,
            weight_slice.width,
        ],
        "rotations": [
            [item.role, list(item.candidates)] for item in prepared.rotations
        ],
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
        "slot": [prepared.slot.policy, prepared.slot.value],
        "constants": [
            [
                item.role,
                item.type.element_type,
                list(item.type.shape),
                len(item.bytes),
                item.content_hash,
            ]
            for item in prepared.constants
        ],
        "runtime_preparations": [
            [
                runtime_input.role,
                runtime_input.source_operand,
                runtime_input.kind,
                runtime_input.result_type.element_type,
                list(runtime_input.result_type.shape),
                runtime_input.logical_input_size,
                runtime_input.replications,
                runtime_input.blocking_width,
                list(runtime_input.rotation_candidates),
                runtime_input.outer_block_depth,
            ]
        ],
        "scalar_preparations": list(prepared.scalar_preparations),
        "immutable": immutable,
        "owned_constants": all(
            isinstance(item.bytes, bytes) for item in prepared.constants
        ),
    }


def _worker_main():
    from ace_edsl.edsl.kernels.vector.fast_gemm import fast_gemm_recipe
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    model = Path(sys.argv[2])
    implementation = sys.argv[3]
    mask_fuse = sys.argv[4] == "1"
    behavior = sys.argv[5]
    pipeline = Pipeline(
        "fast-gemm-worker",
        output_dir="/tmp/fast-gemm-worker",
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))

    seen_plans = []

    def recipe(trace, prepared):
        seen_plans.append(prepared)
        adjusted = prepared
        if behavior == "wrong-kind":
            adjusted = dataclasses.replace(prepared, kind="baseline-gemm")
        elif behavior == "wrong-slot":
            adjusted = dataclasses.replace(
                prepared,
                slot=dataclasses.replace(prepared.slot, value=prepared.slot.value + 1),
            )
        elif behavior == "missing-weight":
            adjusted = dataclasses.replace(
                prepared,
                constants=tuple(
                    item for item in prepared.constants if item.role != "weight"
                ),
            )
        elif behavior == "wrong-loop":
            adjusted = dataclasses.replace(
                prepared,
                loops=(
                    dataclasses.replace(
                        prepared.loops[0], upper=prepared.loops[0].upper + 1
                    ),
                    *prepared.loops[1:],
                ),
            )
        elif behavior == "wrong-slice":
            weight_slice = prepared.slices[0]
            adjusted = dataclasses.replace(
                prepared,
                slices=(
                    dataclasses.replace(
                        weight_slice,
                        index=dataclasses.replace(
                            weight_slice.index,
                            iv_coefficients=(1, prepared.block_size),
                        ),
                    ),
                ),
            )
        elif behavior == "missing-duplication":
            adjusted = dataclasses.replace(
                prepared,
                rotations=tuple(
                    item
                    for item in prepared.rotations
                    if item.role != "input-duplication"
                ),
            )
        elif behavior == "wrong-grid-rotation":
            adjusted = dataclasses.replace(
                prepared,
                rotations=tuple(
                    dataclasses.replace(
                        item,
                        candidates=tuple(value + 1 for value in item.candidates),
                    )
                    if item.role == "grid"
                    else item
                    for item in prepared.rotations
                ),
            )
        elif behavior == "wrong-mask":
            adjusted = dataclasses.replace(
                prepared,
                mask=dataclasses.replace(
                    prepared.mask, valid_length=prepared.mask.valid_length + 1
                ),
            )
        elif behavior == "wrong-preparation":
            adjusted = dataclasses.replace(
                prepared,
                runtime_preparations=(
                    dataclasses.replace(
                        prepared.runtime_preparations[0],
                        replications=prepared.runtime_preparations[0].replications + 1,
                    ),
                ),
            )
        elif behavior == "missing-reduction":
            adjusted = dataclasses.replace(prepared, reductions=())
        elif behavior == "wrong-reduction":
            adjusted = dataclasses.replace(
                prepared,
                reductions=(
                    dataclasses.replace(
                        prepared.reductions[0],
                        factor=prepared.reductions[0].factor + 1,
                    ),
                    *prepared.reductions[1:],
                ),
            )
        result = fast_gemm_recipe(trace, adjusted)
        if behavior == "post-emission":
            raise RuntimeError("fast Gemm post-emission failure")
        return result

    if implementation == "native":
        kernel_impl = "native"
        fallback = "error"
        register = False
    elif implementation == "dsl":
        kernel_impl = "dsl"
        fallback = "cpp-native" if behavior != "normal" else "error"
        register = True
    elif implementation == "missing-error":
        kernel_impl = "dsl"
        fallback = "error"
        register = False
    elif implementation == "missing-fallback":
        kernel_impl = "dsl"
        fallback = "cpp-native"
        register = False
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=kernel_impl,
        plan_kind="fast-gemm",
        fallback=fallback,
        mask_fuse=mask_fuse,
        max_slots=_MAX_SLOTS,
    )
    if register:
        pipeline.register_vector_kernel_recipe("fast-gemm", recipe)

    result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    if not result.success:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": result.error,
                    "callback_count": len(seen_plans),
                }
            )
        )
        return

    dump = pipeline.glob.dump()
    helper_names = sorted(
        set(re.findall(r'"(__ace_vkernel_fast_gemm_[0-9a-f]{64})"', dump))
    )
    helper_calls = len(
        re.findall(r'^\s+call "__ace_vkernel_fast_gemm_', dump, re.MULTILINE)
    )
    object_oracle = None
    if helper_calls <= 1:
        object_oracle = pipeline.glob._inspect_fast_gemm_air_for_testing()
    normalized = object_oracle["normalized"] if object_oracle else ""

    print(
        json.dumps(
            {
                "success": True,
                "error": None,
                "verify": pipeline.glob.verify_ir(),
                "stages_completed": result.stages_completed,
                "has_nn_gemm": bool(re.search(r"\bNN\.gemm\b", dump, re.I)),
                "helper_names": helper_names,
                "helper_count": len(helper_names),
                "helper_calls": helper_calls,
                "callback_count": len(seen_plans),
                "object_oracle": object_oracle,
                "stats": _normalized_stats(normalized, dump),
                "plan": _plan_summary(seen_plans[0]) if seen_plans else None,
            }
        )
    )


if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == _WORKER_FLAG:
        _worker_main()
