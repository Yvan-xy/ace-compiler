"""Fast-Conv destination-recipe integration and AIR parity tests."""

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


_WORKER_FLAG = "--fast-conv-vector-kernel-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_SLOTS = 128

_CASES = (
    {
        "name": "collective-mask",
        "channel_in": 2,
        "channel_out": 4,
        "height": 2,
        "kernel": 3,
        "group": 1,
        "mask_fuse": False,
        "conv_parallel": False,
        "branch": "collective-single-block",
    },
    {
        "name": "collective-no-mask",
        "channel_in": 2,
        "channel_out": 4,
        "height": 2,
        "kernel": 3,
        "group": 1,
        "mask_fuse": True,
        "conv_parallel": False,
        "branch": "collective-single-block",
    },
    {
        "name": "cyclic",
        "channel_in": 2,
        "channel_out": 3,
        "height": 5,
        "kernel": 1,
        "group": 1,
        "mask_fuse": False,
        "conv_parallel": False,
        "branch": "cyclic",
    },
    {
        "name": "depthwise",
        "channel_in": 4,
        "channel_out": 4,
        "height": 4,
        "kernel": 3,
        "group": 4,
        "mask_fuse": False,
        "conv_parallel": False,
        "branch": "ordinary",
    },
    {
        "name": "collective-blocks",
        "channel_in": 8,
        "channel_out": 16,
        "height": 1,
        "kernel": 3,
        "group": 1,
        "mask_fuse": False,
        "conv_parallel": True,
        "branch": "collective-blocks",
    },
    {
        "name": "collective-blocks-no-policy-mask",
        "channel_in": 8,
        "channel_out": 16,
        "height": 1,
        "kernel": 3,
        "group": 1,
        "mask_fuse": True,
        "conv_parallel": True,
        "branch": "collective-blocks",
    },
    {
        "name": "full-slot",
        "channel_in": 2,
        "channel_out": 8,
        "height": 4,
        "kernel": 3,
        "group": 1,
        "mask_fuse": False,
        "conv_parallel": False,
        "branch": "full-slot",
    },
    {
        "name": "full-slot-no-mask",
        "channel_in": 2,
        "channel_out": 8,
        "height": 4,
        "kernel": 3,
        "group": 1,
        "mask_fuse": True,
        "conv_parallel": False,
        "branch": "full-slot",
    },
    {
        "name": "one-by-one-capacity",
        "channel_in": 4,
        "channel_out": 8,
        "height": 2,
        "kernel": 1,
        "group": 1,
        "mask_fuse": False,
        "conv_parallel": False,
        "branch": "collective-single-block",
    },
)


def _write_conv_model(
    path: Path,
    channel_in: int,
    channel_out: int,
    height: int,
    kernel: int,
    *,
    group: int = 1,
    copies: int = 1,
):
    weight_shape = (
        channel_out,
        channel_in // group,
        kernel,
        kernel,
    )
    weight_values = (
        np.arange(np.prod(weight_shape), dtype=np.float32) + 1.0
    ) / 64.0
    weight = numpy_helper.from_array(
        weight_values.reshape(weight_shape), "weight"
    )
    bias = numpy_helper.from_array(
        np.arange(channel_out, dtype=np.float32) / 16.0, "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, channel_in, height, height]
    )
    nodes = []
    outputs = []
    pad = kernel // 2
    for index in range(copies):
        output = "output" if copies == 1 else f"conv_output_{index}"
        outputs.append(output)
        nodes.append(
            helper.make_node(
                "Conv",
                ["input", "weight", "bias"],
                [output],
                name=f"conv_{index}",
                kernel_shape=[kernel, kernel],
                pads=[pad, pad, pad, pad],
                strides=[1, 1],
                group=group,
            )
        )
    output = outputs[0]
    if copies == 2:
        output = "output"
        nodes.append(helper.make_node("Add", outputs, [output], name="combine"))
    output_info = helper.make_tensor_value_info(
        output, TensorProto.FLOAT, [1, channel_out, height, height]
    )
    graph = helper.make_graph(
        nodes,
        "fast_conv",
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
    *,
    mask_fuse: bool = False,
    conv_parallel: bool = False,
    sharding: bool = False,
    behavior: str = "normal",
    mutate_actual: int = -1,
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
        "1" if conv_parallel else "0",
        "1" if sharding else "0",
        behavior,
        str(mutate_actual),
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
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


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_cpp_plan_dsl_fast_conv_matches_native_air(tmp_path, case):
    model = tmp_path / f"{case['name']}.onnx"
    _write_conv_model(
        model,
        case["channel_in"],
        case["channel_out"],
        case["height"],
        case["kernel"],
        group=case["group"],
    )
    options = {
        "mask_fuse": case["mask_fuse"],
        "conv_parallel": case["conv_parallel"],
    }
    native = _run_worker(model, "native", **options)
    dsl = _run_worker(model, "dsl", **options)

    assert native["success"] and dsl["success"]
    assert native["verify"] and dsl["verify"]
    assert native["stages_completed"] == dsl["stages_completed"] == [
        "tensor2vector"
    ]
    assert native["helper_count"] == native["helper_calls"] == 0
    assert dsl["helper_count"] == dsl["helper_calls"] == 1
    assert dsl["object_oracle"]["bridge_ok"]
    assert dsl["object_oracle"]["prepared_input_ok"]
    assert dsl["object_oracle"]["native_normalized"] == dsl["normalized"]
    assert {
        key: value
        for key, value in native["stats"].items()
        if key != "preg_loads"
    } == {
        key: value
        for key, value in dsl["stats"].items()
        if key != "preg_loads"
    }

    plan = dsl["plan"]
    assert plan["kind"] == "fast-conv"
    assert plan["provenance"] == "cpp"
    assert plan["group"] == case["group"]
    assert plan["kernel_hw"] == case["kernel"] ** 2
    assert plan["slot"] == ["logical-output-elements", plan["output_size"]]
    assert plan["runtime_scalar_inputs"] == []
    assert plan["scalar_preparations"] == []
    assert plan["runtime_preparations"][0][2] == "blocking-rotations"
    assert plan["runtime_preparations"][0][6] == plan["input_duplications"]
    assert plan["runtime_preparations"][0][7] == plan["capacity_block"]
    assert plan["loops"][:2] == [
        ["grid", 0, plan["num_grid"], 1, 0],
        ["capacity-block", 0, plan["capacity_block"], 1, 1],
    ]
    assert plan["slices"][0] == [
        "weight",
        [plan["capacity_block"], 1],
        0,
        False,
        plan["num_block"] * plan["width_block"],
    ]
    assert plan["constants"][:3][0][0] == "weight"
    assert [item[0] for item in plan["constants"][:3]] == [
        "weight",
        "bias",
        "rotation-table",
    ]
    assert all(item[4].startswith("sha256:") for item in plan["constants"])
    assert sorted(set(dsl["stats"]["constant_hashes"])) == sorted(
        item[4] for item in plan["constants"]
    )
    assert dsl["stats"]["slot_values"] == [plan["output_size"]]
    assert dsl["stats"]["comments"][:2] == [
        f"BlockingRot replicate={plan['input_duplications']}",
        f"BlockingRot HRot=bs={plan['capacity_block']}",
    ]

    roles = [item[0] for item in plan["rotations"]]
    reduction_kinds = [item[1] for item in plan["reductions"]]
    if case["branch"] == "cyclic":
        assert plan["cyclic_roll"]
        assert plan["reductions"] == []
        assert roles == [
            "blocking-alignment",
            "cyclic-left",
            "cyclic-right",
        ]
        assert [item[0] for item in plan["slices"]][-2:] == [
            "cyclic-mask-left",
            "cyclic-mask-right",
        ]
    elif case["branch"] == "ordinary":
        assert not plan["cyclic_roll"]
        assert plan["reductions"] == []
        assert plan["group"] == plan["channel_in"]
        assert roles == ["blocking-alignment", "grid"]
    elif case["branch"] == "collective-blocks":
        assert plan["num_block"] == 2
        assert reduction_kinds == ["collective-blocks"]
        assert [item[0] for item in plan["loops"]][-2:] == [
            "collective-data",
            "collective-gap",
        ]
    elif case["branch"] == "full-slot":
        assert plan["output_size"] == plan["num_slots"]
        assert "collective-tail" not in roles
        assert len(plan["reductions"]) == 1
    else:
        assert reduction_kinds == ["collective-single-block"]
        assert roles[-1] == "collective-tail"
    if case["mask_fuse"]:
        assert plan["mask"] == ["none", 0]
    elif plan["reductions"]:
        assert plan["mask"] == [
            "collective-reduction",
            plan["output_size"],
        ]


def test_equal_fast_conv_plans_share_one_helper(tmp_path):
    model = tmp_path / "repeated.onnx"
    _write_conv_model(model, 2, 4, 2, 3, copies=2)
    result = _run_worker(model, "dsl")

    assert result["success"] and result["verify"]
    assert result["callback_count"] == 1
    assert result["helper_count"] == 1
    assert result["helper_calls"] == 2


@pytest.mark.parametrize(
    "behavior,diagnostic",
    [
        ("wrong-kind", "another plan kind"),
        ("wrong-slot", "SLOT"),
        ("wrong-loop", "loop topology"),
        ("wrong-slice", "weight slice"),
        ("wrong-rotation", "rotation topology"),
        ("wrong-cyclic", "cyclic topology policy"),
        ("missing-reduction", "collective topology policy"),
        ("wrong-reduction-kind", "collective"),
        ("wrong-reduction-padding", "collective padding"),
        ("wrong-mask", "collective topology"),
        ("wrong-preparation", "input preparation"),
        ("missing-weight", "constant topology"),
        ("wrong-bias-shape", "constant descriptors"),
        ("wrong-result-type", "dimensions"),
        ("wrong-runtime-type", "dimensions"),
        ("consistent-wrong-runtime-shape", "frozen identity"),
        ("wrong-rotation-payload", "payload or hash"),
        ("wrong-collective-mask-payload", "payload or hash"),
        ("wrong-constant-hash", "payload or hash"),
        ("post-emission", "post-emission failure"),
    ],
)
def test_invalid_or_failed_recipe_never_falls_back(
    tmp_path, behavior, diagnostic
):
    model = tmp_path / f"{behavior}.onnx"
    _write_conv_model(model, 2, 4, 2, 3)
    result = _run_worker(model, "dsl", behavior=behavior)

    assert not result["success"]
    assert diagnostic in result["error"]
    assert result["callback_count"] == 1


@pytest.mark.parametrize(
    "behavior,diagnostic",
    [
        ("wrong-scalar-type", "sharding scalar"),
        ("wrong-scalar-scale", "sharding scalar"),
        ("missing-offset", "unsharded fast Conv has scalar"),
        ("wrong-outer-depth", "input preparation"),
        ("consistent-invalid-outer-depth", "sharding scalar"),
        ("consistent-wrong-scalar-scale", "frozen identity"),
    ],
)
def test_invalid_sharded_package_never_falls_back(
    tmp_path, behavior, diagnostic
):
    model = tmp_path / f"{behavior}.onnx"
    _write_conv_model(model, 1, 2, 12, 3)
    result = _run_worker(
        model, "dsl", sharding=True, behavior=behavior
    )

    assert not result["success"]
    assert diagnostic in result["error"]
    assert result["callback_count"] == 1


def test_sharded_fast_conv_has_typed_scalar_bridge_and_scaled_slice(tmp_path):
    model = tmp_path / "sharded.onnx"
    _write_conv_model(model, 1, 2, 12, 3)
    result = _run_worker(model, "dsl", sharding=True)

    assert result["success"] and result["verify"]
    assert result["object_oracle"]["bridge_ok"]
    assert result["object_oracle"]["prepared_input_ok"]
    assert (
        result["object_oracle"]["native_normalized"]
        == result["normalized"]
    )
    plan = result["plan"]
    assert plan["runtime_scalar_inputs"] == ["s32"]
    assert plan["scalar_preparations"] == [
        ["weight-offset", 1, "s32", 9]
    ]
    assert plan["sharding_offset"] == ["s32", 9]
    assert plan["blocking_outer_depth"] == 1
    assert plan["slices"][0][3]
    assert result["stats"]["core_mul_by_offset_scale"] == 1

    wrong_vector = _run_worker(
        model, "dsl", sharding=True, mutate_actual=0
    )
    wrong_scalar = _run_worker(
        model, "dsl", sharding=True, mutate_actual=1
    )
    assert not wrong_vector["object_oracle"]["bridge_ok"]
    assert not wrong_vector["object_oracle"]["prepared_input_ok"]
    assert not wrong_scalar["object_oracle"]["bridge_ok"]
    assert wrong_scalar["object_oracle"]["prepared_input_ok"]


@pytest.mark.parametrize(
    "name,options,behavior,diagnostic",
    [
        (
            "collective-no-mask-missing-reduction",
            {"mask_fuse": True},
            "missing-reduction",
            "collective topology policy",
        ),
        (
            "cyclic-mask-payload",
            {},
            "wrong-cyclic-mask-payload",
            "payload or hash",
        ),
        (
            "gap-mask-payload",
            {"conv_parallel": True},
            "wrong-gap-mask-payload",
            "payload or hash",
        ),
    ],
)
def test_branch_specific_frozen_package_mutations_fail_closed(
    tmp_path, name, options, behavior, diagnostic
):
    model = tmp_path / f"{name}.onnx"
    if name == "cyclic-mask-payload":
        _write_conv_model(model, 2, 3, 5, 1)
    elif name == "gap-mask-payload":
        _write_conv_model(model, 8, 16, 1, 3)
    else:
        _write_conv_model(model, 2, 4, 2, 3)

    result = _run_worker(model, "dsl", behavior=behavior, **options)

    assert not result["success"]
    assert diagnostic in result["error"]
    assert result["callback_count"] == 1


def test_missing_fast_conv_recipe_obeys_error_and_explicit_fallback(tmp_path):
    model = tmp_path / "missing.onnx"
    _write_conv_model(model, 2, 4, 2, 3)
    failed = _run_worker(
        model, "missing-error", allow_process_failure=True
    )
    fallback = _run_worker(model, "missing-fallback")

    assert failed["returncode"] != 0
    assert "no DSL recipe is registered" in failed["stderr"]
    assert fallback["success"] and fallback["verify"]
    assert fallback["callback_count"] == 0
    assert fallback["helper_count"] == fallback["helper_calls"] == 0


def test_fast_conv_exports_and_configuration_are_canonical():
    import ace_edsl.edsl as edsl
    from ace_edsl.edsl.kernels.vector.fast_conv import (
        configure_fast_conv_dsl as canonical_configure,
        fast_conv_recipe as canonical_recipe,
        fast_conv_sharded_vector_kernel as canonical_sharded,
        fast_conv_vector_kernel as canonical_kernel,
    )
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.vector_kernel_fast_conv import (
        configure_fast_conv_dsl as compatibility_configure,
        fast_conv_recipe as compatibility_recipe,
        fast_conv_sharded_vector_kernel as compatibility_sharded,
        fast_conv_vector_kernel as compatibility_kernel,
    )

    assert edsl.fast_conv_recipe is canonical_recipe
    assert edsl.fast_conv_vector_kernel is canonical_kernel
    assert edsl.fast_conv_sharded_vector_kernel is canonical_sharded
    assert edsl.configure_fast_conv_dsl is canonical_configure
    assert compatibility_recipe is canonical_recipe
    assert compatibility_kernel is canonical_kernel
    assert compatibility_sharded is canonical_sharded
    assert compatibility_configure is canonical_configure

    pipeline = Pipeline("fast-conv-config", dump_ir=False, verbose=False)
    configured = canonical_configure(
        pipeline,
        mask_fuse=True,
        max_slots=_MAX_SLOTS,
        conv_parallel=True,
        sharding=True,
    )
    assert configured is pipeline
    config = pipeline.vector_kernel_config
    assert config.plan_provider == "cpp"
    assert config.kernel_impl == "dsl"
    assert config.plan_kind == "fast-conv"
    assert config.fallback == "error"
    assert config.mask_fuse
    assert config.max_slots == _MAX_SLOTS
    assert config.conv_parallel
    assert config.sharding
    assert pipeline.vector_kernel_recipes == {"fast-conv": canonical_recipe}


def _parse_rnums(dump: str):
    values = []
    for raw in re.findall(r"ATTR\[nums=([^]]+)\]", dump):
        values.append([int(value) for value in raw.strip("()").split(",")])
    return values


def _normalized_stats(normalized: str, dump: str, offset_scale: int | None):
    offset_mul_pattern = (
        rf"\{{CORE\.mul,rtype=type\(1,domain=0,primitive=s32\),children="
        rf"\[\{{CORE\.ld,[^{{}}]*datum=1:type\(1,domain=0,primitive=s32\),"
        rf"[^{{}}]*\}},\{{CORE\.intconst,[^{{}}]*value={offset_scale},"
        rf"children=\[\]\}}\]\}}"
        if offset_scale is not None
        else r"(?!)"
    )
    return {
        "loops": normalized.count("{CORE.do_loop,"),
        "rolls": normalized.count("{VECTOR.roll,"),
        "adds": normalized.count("{VECTOR.add,"),
        "muls": normalized.count("{VECTOR.mul,"),
        "zeros": normalized.count("{CORE.zero,"),
        "slices": normalized.count("{VECTOR.slice,"),
        "indexed_stores": normalized.count("{CORE.ist,"),
        "indexed_loads": normalized.count("{CORE.ild,"),
        "preg_stores": normalized.count("{CORE.stp,"),
        "preg_loads": normalized.count("{CORE.ldp,"),
        "rnums": _parse_rnums(dump),
        "slot_values": [
            int.from_bytes(bytes.fromhex(value), "little")
            for value in re.findall(
                r"slot:u32:1:([0-9a-f]{8})", normalized
            )
        ],
        "comments": re.findall(
            r"\{CORE\.comment,comment=([^,]*),children=\[\]\}",
            normalized,
        ),
        "constant_hashes": re.findall(
            r"hash=(sha256:[0-9a-f]{64})", normalized
        ),
        "core_mul_by_offset_scale": len(
            re.findall(offset_mul_pattern, normalized)
        ),
    }


def _plan_summary(prepared):
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
        "channel_in": prepared.channel_in,
        "channel_out": prepared.channel_out,
        "output_height": prepared.output_height,
        "output_width": prepared.output_width,
        "kernel_hw": prepared.kernel_hw,
        "group": prepared.group,
        "stride": prepared.stride,
        "input_size": prepared.input_size,
        "output_size": prepared.output_size,
        "num_slots": prepared.num_slots,
        "num_grid": prepared.num_grid,
        "num_block": prepared.num_block,
        "width_block": prepared.width_block,
        "width_block_data": prepared.width_block_data,
        "width_block_pad": prepared.width_block_pad,
        "position_block": prepared.position_block,
        "capacity_block": prepared.capacity_block,
        "input_duplications": prepared.input_duplications,
        "blocking_outer_depth": prepared.blocking_outer_depth,
        "cyclic_roll": prepared.cyclic_roll,
        "sharding_offset": (
            None
            if prepared.sharding_offset is None
            else [
                prepared.sharding_offset.type,
                prepared.sharding_offset.scale,
            ]
        ),
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
        "immutable": immutable,
        "owned_constants": all(
            isinstance(item.bytes, bytes) for item in prepared.constants
        ),
    }


def _adjust_plan(prepared, behavior):
    if behavior == "wrong-kind":
        return dataclasses.replace(prepared, kind="baseline-conv")
    if behavior == "wrong-slot":
        return dataclasses.replace(
            prepared,
            slot=dataclasses.replace(
                prepared.slot, value=prepared.slot.value + 1
            ),
        )
    if behavior == "wrong-loop":
        return dataclasses.replace(
            prepared,
            loops=(
                dataclasses.replace(prepared.loops[0], upper=0),
                *prepared.loops[1:],
            ),
        )
    if behavior == "wrong-slice":
        return dataclasses.replace(
            prepared,
            slices=(
                dataclasses.replace(
                    prepared.slices[0], width=prepared.slices[0].width + 1
                ),
                *prepared.slices[1:],
            ),
        )
    if behavior == "wrong-rotation":
        return dataclasses.replace(
            prepared,
            rotations=(
                prepared.rotations[0],
                dataclasses.replace(
                    prepared.rotations[1],
                    candidates=(prepared.rotations[1].candidates[0] + 1,),
                ),
                *prepared.rotations[2:],
            ),
        )
    if behavior == "wrong-cyclic":
        return dataclasses.replace(
            prepared, cyclic_roll=not prepared.cyclic_roll
        )
    if behavior == "missing-reduction":
        return dataclasses.replace(prepared, reductions=())
    if behavior == "wrong-reduction-kind":
        reduction = prepared.reductions[0]
        return dataclasses.replace(
            prepared,
            reductions=(
                dataclasses.replace(
                    reduction,
                    kind=(
                        "collective-blocks"
                        if reduction.kind == "collective-single-block"
                        else "collective-single-block"
                    ),
                ),
            ),
        )
    if behavior == "wrong-reduction-padding":
        reduction = prepared.reductions[0]
        return dataclasses.replace(
            prepared,
            reductions=(
                dataclasses.replace(
                    reduction, padding=reduction.padding + 1
                ),
            ),
        )
    if behavior == "wrong-mask":
        return dataclasses.replace(
            prepared,
            mask=dataclasses.replace(prepared.mask, valid_length=0),
        )
    if behavior == "wrong-preparation":
        return dataclasses.replace(
            prepared,
            runtime_preparations=(
                dataclasses.replace(
                    prepared.runtime_preparations[0],
                    logical_input_size=prepared.input_size + 1,
                ),
            ),
        )
    if behavior == "missing-weight":
        return dataclasses.replace(
            prepared,
            constants=tuple(
                item for item in prepared.constants if item.role != "weight"
            ),
        )
    if behavior == "wrong-bias-shape":
        return dataclasses.replace(
            prepared,
            constants=tuple(
                dataclasses.replace(
                    item,
                    type=dataclasses.replace(
                        item.type, shape=(prepared.output_size + 1,)
                    ),
                )
                if item.role == "bias"
                else item
                for item in prepared.constants
            ),
        )
    if behavior == "wrong-result-type":
        return dataclasses.replace(
            prepared,
            result_type=dataclasses.replace(
                prepared.result_type, element_type="s32"
            ),
        )
    if behavior == "wrong-runtime-type":
        runtime_type = dataclasses.replace(
            prepared.runtime_vector_inputs[0], element_type="s32"
        )
        runtime_preparation = dataclasses.replace(
            prepared.runtime_preparations[0], result_type=runtime_type
        )
        return dataclasses.replace(
            prepared,
            runtime_vector_inputs=(runtime_type,),
            runtime_preparations=(runtime_preparation,),
        )
    if behavior == "consistent-wrong-runtime-shape":
        runtime_type = dataclasses.replace(
            prepared.runtime_vector_inputs[0],
            shape=(prepared.runtime_vector_inputs[0].shape[0] + 1,),
        )
        runtime_preparation = dataclasses.replace(
            prepared.runtime_preparations[0], result_type=runtime_type
        )
        return dataclasses.replace(
            prepared,
            runtime_vector_inputs=(runtime_type,),
            runtime_preparations=(runtime_preparation,),
        )
    if behavior in (
        "wrong-rotation-payload",
        "wrong-collective-mask-payload",
        "wrong-cyclic-mask-payload",
        "wrong-gap-mask-payload",
    ):
        roles = {
            "wrong-rotation-payload": "rotation-table",
            "wrong-collective-mask-payload": "collective-mask",
            "wrong-cyclic-mask-payload": "cyclic-mask-left",
            "wrong-gap-mask-payload": "collective-gap-mask",
        }
        role = roles[behavior]
        return dataclasses.replace(
            prepared,
            constants=tuple(
                dataclasses.replace(
                    item,
                    bytes=bytes([item.bytes[0] ^ 1]) + item.bytes[1:],
                )
                if item.role == role
                else item
                for item in prepared.constants
            ),
        )
    if behavior == "wrong-constant-hash":
        return dataclasses.replace(
            prepared,
            constants=tuple(
                dataclasses.replace(item, content_hash="sha256:" + "0" * 64)
                if item.role == "bias"
                else item
                for item in prepared.constants
            ),
        )
    if behavior == "wrong-scalar-type":
        return dataclasses.replace(prepared, runtime_scalar_inputs=("s64",))
    if behavior == "wrong-scalar-scale":
        scalar = prepared.scalar_preparations[0]
        return dataclasses.replace(
            prepared,
            scalar_preparations=(
                dataclasses.replace(scalar, scale=scalar.scale + 1),
            ),
        )
    if behavior == "missing-offset":
        return dataclasses.replace(prepared, sharding_offset=None)
    if behavior == "wrong-outer-depth":
        runtime = prepared.runtime_preparations[0]
        return dataclasses.replace(
            prepared,
            runtime_preparations=(
                dataclasses.replace(
                    runtime, outer_block_depth=runtime.outer_block_depth + 1
                ),
            ),
        )
    if behavior == "consistent-invalid-outer-depth":
        runtime = prepared.runtime_preparations[0]
        return dataclasses.replace(
            prepared,
            blocking_outer_depth=2,
            runtime_preparations=(
                dataclasses.replace(runtime, outer_block_depth=2),
            ),
        )
    if behavior == "consistent-wrong-scalar-scale":
        scalar = prepared.scalar_preparations[0]
        offset = prepared.sharding_offset
        return dataclasses.replace(
            prepared,
            sharding_offset=dataclasses.replace(
                offset, scale=offset.scale + 1
            ),
            scalar_preparations=(
                dataclasses.replace(scalar, scale=scalar.scale + 1),
            ),
        )
    return prepared


def _worker_main():
    from ace_edsl.edsl.kernels.vector.fast_conv import fast_conv_recipe
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    model = Path(sys.argv[2])
    implementation = sys.argv[3]
    mask_fuse = sys.argv[4] == "1"
    conv_parallel = sys.argv[5] == "1"
    sharding = sys.argv[6] == "1"
    behavior = sys.argv[7]
    mutate_actual = int(sys.argv[8])
    pipeline = Pipeline(
        "fast-conv-worker",
        output_dir="/tmp/fast-conv-worker",
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))

    seen_plans = []
    callback_count = 0

    def recipe(trace, prepared):
        nonlocal callback_count
        callback_count += 1
        seen_plans.append(prepared)
        adjusted = _adjust_plan(prepared, behavior)
        result = fast_conv_recipe(trace, adjusted)
        if behavior == "post-emission":
            raise RuntimeError("post-emission failure")
        return result

    fallback = "error"
    recipes = implementation == "dsl"
    if implementation == "missing-error":
        implementation = "dsl"
    elif implementation == "missing-fallback":
        implementation = "dsl"
        fallback = "cpp-native"
    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=implementation,
        plan_kind="fast-conv",
        fallback=fallback,
        mask_fuse=mask_fuse,
        max_slots=_MAX_SLOTS,
        conv_parallel=conv_parallel,
        sharding=sharding,
    )
    if recipes:
        pipeline.register_vector_kernel_recipe("fast-conv", recipe)

    result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    payload = {
        "success": result.success,
        "error": result.error,
        "callback_count": callback_count,
        "stages_completed": result.stages_completed,
    }
    if not result.success:
        print(json.dumps(payload))
        return

    if mutate_actual >= 0:
        pipeline.glob._mutate_fast_conv_call_actual_for_testing(
            mutate_actual
        )
    dump = pipeline.glob.dump()
    helper_names = sorted(set(re.findall(
        r'^FUN\[[^]]+\] "(__ace_vkernel_fast_conv_[0-9a-f]{64})"',
        dump,
        re.MULTILINE,
    )))
    helper_calls = len(re.findall(
        r'^\s+call "__ace_vkernel_fast_conv_', dump, re.MULTILINE
    ))
    object_oracle = None
    normalized = ""
    if helper_calls <= 1:
        object_oracle = pipeline.glob._inspect_fast_conv_air_for_testing()
        normalized = object_oracle["normalized"]
    offset_scale = None
    plan = None
    if seen_plans:
        plan = _plan_summary(seen_plans[0])
        if seen_plans[0].sharding_offset is not None:
            offset_scale = seen_plans[0].sharding_offset.scale
    payload.update({
        "verify": pipeline.glob.verify_ir(),
        "helper_count": len(helper_names),
        "helper_calls": helper_calls,
        "helper_names": helper_names,
        "object_oracle": (
            None if object_oracle is None else dict(object_oracle)
        ),
        "normalized": normalized,
        "plan": plan,
        "stats": _normalized_stats(normalized, dump, offset_scale),
    })
    print(json.dumps(payload))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
