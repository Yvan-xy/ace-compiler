"""Baseline-Conv destination-recipe integration and parity tests."""

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


_WORKER_FLAG = "--baseline-conv-vector-kernel-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_SLOTS = 128
_CHANNEL_IN = 4
_CHANNEL_OUT = 2
_HEIGHT = 4
_WIDTH = 4
_KERNEL = 3
_KERNEL_HW = _KERNEL * _KERNEL
_OUTPUT_SIZE = _CHANNEL_OUT * _HEIGHT * _WIDTH
_INPUT_SIZE = _CHANNEL_IN * _HEIGHT * _WIDTH
_KERNEL_ROTATIONS = [-5, -4, -3, -1, 0, 1, 3, 4, 5]
_CONSTANT_HASHES = {
    "sha256:c3cd257c1cfc51649d940b8b16a2bf3a6f4b1a867971f3be12f5dc420fa81b12",
    "sha256:0fe5a5c048a568a39fafae8838c10f0812d4950cc72be684dfd899c279fdca08",
    "sha256:cdc1443f67aa1685a68fa1c5fa72d17d7d20c47c58ae38d5851fccd5a0ddcf23",
}


def _write_conv_model(path: Path):
    weight_values = (
        np.arange(
            _CHANNEL_OUT * _CHANNEL_IN * _KERNEL_HW,
            dtype=np.float32,
        )
        + 1.0
    ) / 32.0
    weight = numpy_helper.from_array(
        weight_values.reshape(
            _CHANNEL_OUT, _CHANNEL_IN, _KERNEL, _KERNEL
        ),
        "weight",
    )
    bias = numpy_helper.from_array(
        np.array([0.25, -0.5], dtype=np.float32), "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, _CHANNEL_IN, _HEIGHT, _WIDTH],
    )
    output_name = "output"
    conv = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        [output_name],
        name="conv",
        kernel_shape=[_KERNEL, _KERNEL],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
    )
    output_info = helper.make_tensor_value_info(
        output_name,
        TensorProto.FLOAT,
        [1, _CHANNEL_OUT, _HEIGHT, _WIDTH],
    )
    graph = helper.make_graph(
        [conv],
        "baseline_conv",
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
    behavior: str = "normal",
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
        behavior,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"worker failed ({' '.join(command)}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    return json.loads(lines[-1])


def _decode_air_int(line: str):
    match = re.search(r"#(0x[0-9a-f]+|-?\d+)", line, re.IGNORECASE)
    assert match, f"missing AIR integer constant in: {line}"
    value = int(match.group(1), 0)
    if value >= 1 << 63:
        value -= 1 << 64
    return value


def _indent(line: str):
    return len(line) - len(line.lstrip())


def _loop_records(dump: str):
    lines = dump.splitlines()
    records = []
    for loop_index, line in enumerate(lines):
        if not line.lstrip().startswith("do_loop ID"):
            continue
        root_indent = _indent(line)
        header_indent = root_indent + 2
        cursor = loop_index - 1
        found = {}
        for name, prefix in (
            ("end", "end_block"),
            ("body", "block ID"),
            ("step_op", "add "),
            ("condition", "lt "),
            ("lower", "intconst "),
        ):
            while cursor >= 0:
                text = lines[cursor].lstrip()
                if (
                    _indent(lines[cursor]) == header_indent
                    and text.startswith(prefix)
                ):
                    found[name] = cursor
                    cursor -= 1
                    break
                cursor -= 1
            else:
                raise AssertionError(f"malformed AIR do_loop near: {line}")
        upper_values = [
            _decode_air_int(lines[index])
            for index in range(found["lower"] + 1, found["condition"])
            if _indent(lines[index]) == header_indent + 2
            and lines[index].lstrip().startswith("intconst ")
        ]
        step_values = [
            _decode_air_int(lines[index])
            for index in range(found["condition"] + 1, found["step_op"])
            if _indent(lines[index]) == header_indent + 2
            and lines[index].lstrip().startswith("intconst ")
        ]
        assert len(upper_values) == 1 and len(step_values) == 1
        records.append(
            [
                _decode_air_int(lines[found["lower"]]),
                upper_values[0],
                step_values[0],
                root_indent,
            ]
        )
    if records:
        nesting = {
            indent: depth
            for depth, indent in enumerate(
                sorted({record[3] for record in records})
            )
        }
        for record in records:
            record[3] = nesting[record[3]]
        records.sort(key=lambda record: record[3])
    return records


def _direct_child_lines(lines, operation_index):
    operation_indent = _indent(lines[operation_index])
    children = []
    cursor = operation_index - 1
    while cursor >= 0 and _indent(lines[cursor]) > operation_indent:
        if _indent(lines[cursor]) == operation_indent + 2:
            children.append(lines[cursor])
        cursor -= 1
    children.reverse()
    return children


def _slice_records(dump: str):
    lines = dump.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("VECTOR.slice "):
            continue
        children = _direct_child_lines(lines, index)
        widths = [
            _decode_air_int(child)
            for child in children
            if child.lstrip().startswith("intconst ")
        ]
        direct_loads = [
            child for child in children if child.lstrip().startswith("ld ")
        ]
        assert len(widths) == 1
        records.append(
            ["iv" if len(direct_loads) == 1 else "expression", widths[0]]
        )
    return records


def _roll_operands(dump: str):
    lines = dump.splitlines()
    operands = []
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("VECTOR.roll "):
            continue
        cursor = index - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        root = lines[cursor].lstrip()
        if root.startswith("intconst "):
            operands.append(["const", _decode_air_int(lines[cursor])])
        elif root.startswith("ild "):
            operands.append(["indexed-load"])
        elif root.startswith("ld "):
            operands.append(["iv"])
        else:
            raise AssertionError(f"unsupported AIR roll operand: {root}")
    return operands


def _kernel_stats(dump: str, normalized: str):
    rotations = [
        [int(value) for value in re.findall(r"-?\d+", encoded)]
        for encoded in re.findall(r"ATTR\[nums=([^]]+)\]", dump)
    ]
    return {
        "loop_records": _loop_records(dump),
        "roll_operands": _roll_operands(dump),
        "slice_records": _slice_records(dump),
        "rotations": rotations,
        "loops": normalized.count("{CORE.do_loop"),
        "rolls": normalized.count("{VECTOR.roll"),
        "slices": normalized.count("{VECTOR.slice"),
        "adds": normalized.count("{VECTOR.add"),
        "muls": normalized.count("{VECTOR.mul"),
        "core_adds": normalized.count("{CORE.add"),
        "core_muls": normalized.count("{CORE.mul"),
        "ilds": normalized.count("{CORE.ild"),
        "ldcas": normalized.count("{CORE.ldca"),
        "slot_attrs": normalized.count("slot:u32"),
        "constant_hashes": sorted(
            set(re.findall(r"constant-hash=(sha256:[0-9a-f]{64})", normalized))
        ),
    }


def test_cpp_plan_dsl_baseline_conv_matches_native_air(tmp_path):
    model = tmp_path / "conv_ci4_co2_k3.onnx"
    _write_conv_model(model)

    native = _run_worker(model, "native")
    dsl = _run_worker(model, "dsl")
    dsl_auto = _run_worker(model, "dsl-auto")

    assert native["success"] and dsl["success"] and dsl_auto["success"]
    assert native["verify"] and dsl["verify"] and dsl_auto["verify"]
    assert not native["has_nn_conv"]
    assert not dsl["has_nn_conv"] and not dsl_auto["has_nn_conv"]
    assert (
        native["kernel_stats"]
        == dsl["kernel_stats"]
        == dsl_auto["kernel_stats"]
    )
    assert dsl["kernel_stats"] == {
        "loop_records": [[0, _CHANNEL_IN, 1, 0], [0, _KERNEL_HW, 1, 1]],
        "roll_operands": [
            ["const", -_INPUT_SIZE],
            ["indexed-load"],
            ["const", _HEIGHT * _WIDTH],
        ],
        "slice_records": [["expression", _OUTPUT_SIZE]],
        "rotations": [
            [-_INPUT_SIZE],
            _KERNEL_ROTATIONS,
            [_HEIGHT * _WIDTH],
        ],
        "loops": 2,
        "rolls": 3,
        "slices": 1,
        "adds": 3,
        "muls": 1,
        "core_adds": 3,
        "core_muls": 1,
        "ilds": 1,
        "ldcas": 1,
        "slot_attrs": 1,
        "constant_hashes": sorted(_CONSTANT_HASHES),
    }

    from ace_bindings import air_builder

    assert native["object_oracle"]["kind"] == "native"
    assert dsl["object_oracle"]["kind"] == "helper"
    assert dsl["object_oracle"]["bridge_ok"]
    assert dsl_auto["object_oracle"]["kind"] == "helper"
    assert dsl_auto["object_oracle"]["bridge_ok"]
    assert dsl["object_oracle"]["prepared_input_ok"]
    comparison = air_builder._compare_normalized_vector_kernel_air_for_testing(
        dsl["object_oracle"]["native_normalized"],
        dsl["object_oracle"]["normalized"],
    )
    assert comparison["equal"], comparison

    assert native["helper_count"] == native["helper_calls"] == 0
    assert dsl["helper_count"] == dsl["helper_calls"] == 1
    assert dsl_auto["helper_count"] == dsl_auto["helper_calls"] == 1
    assert dsl["helper_formals"] == ["packed_input_0"]
    assert dsl_auto["helper_formals"] == ["packed_input_0"]
    assert dsl["callback_count"] == 1
    assert dsl_auto["callback_count"] == 1
    assert "vector2sihe" not in dsl["stages_completed"]
    assert "vector2sihe" not in dsl_auto["stages_completed"]

    plan = dsl["plan"]
    assert plan == {
        "kind": "baseline-conv",
        "provenance": "cpp",
        "channel_in": _CHANNEL_IN,
        "channel_out": _CHANNEL_OUT,
        "output_height": _HEIGHT,
        "output_width": _WIDTH,
        "kernel_hw": _KERNEL_HW,
        "stride": 1,
        "input_duplications": 2,
        "runtime_input_shapes": [[_INPUT_SIZE]],
        "result_shape": [1, _CHANNEL_OUT, _HEIGHT, _WIDTH],
        "loops": [
            ["channel-in", 0, _CHANNEL_IN, 1, 0],
            ["kernel-hw", 0, _KERNEL_HW, 1, 1],
        ],
        "slice": ["weight", [_KERNEL_HW, 1], 0, False, _OUTPUT_SIZE],
        "rotations": [
            ["input-duplication", [-_INPUT_SIZE]],
            ["kernel-alignment", _KERNEL_ROTATIONS],
            ["channel-step", [_HEIGHT * _WIDTH]],
        ],
        "mask": ["none", 0],
        "slot": ["logical-output-elements", _OUTPUT_SIZE],
        "constants": [
            ["weight", "f32", [_CHANNEL_IN * _KERNEL_HW, _OUTPUT_SIZE],
             "sha256:c3cd257c1cfc51649d940b8b16a2bf3a6f4b1a867971f3be12f5dc420fa81b12",
             _CHANNEL_IN * _KERNEL_HW * _OUTPUT_SIZE * 4],
            ["bias", "f32", [_OUTPUT_SIZE],
             "sha256:0fe5a5c048a568a39fafae8838c10f0812d4950cc72be684dfd899c279fdca08",
             _OUTPUT_SIZE * 4],
            ["rotation-table", "s32", [_KERNEL_HW],
             "sha256:cdc1443f67aa1685a68fa1c5fa72d17d7d20c47c58ae38d5851fccd5a0ddcf23",
             _KERNEL_HW * 4],
        ],
        "runtime_preparation": [
            "input", 0, "flatten-packed-vector", [_INPUT_SIZE],
            _INPUT_SIZE, 2, 0, [-_INPUT_SIZE], 0,
        ],
        "immutable": True,
        "owned_constants": True,
    }
    assert dsl_auto["plan"] == plan
    assert dsl["trace_expired"]
    assert dsl["container_expired"]
    assert dsl["node_expired"]
    assert dsl["type_expired"]
    assert dsl["type_factory_same_scope"]
    assert dsl["state_restored"]


def test_conv_bridge_rejects_a_same_typed_wrong_prepared_input(tmp_path):
    model = tmp_path / "wrong_prepared_input.onnx"
    _write_conv_model(model)

    result = _run_worker(model, "dsl", "wrong-actual")

    assert result["success"]
    assert result["verify"]
    assert not result["object_oracle"]["bridge_ok"]
    assert not result["object_oracle"]["prepared_input_ok"]
    assert (
        result["object_oracle"]["bridge_message"]
        == "CALL actual order does not match expected inputs"
    )


def test_unconfigured_conv_pipeline_preserves_default_native_behavior(tmp_path):
    model = tmp_path / "default_native_conv.onnx"
    _write_conv_model(model)

    default = _run_worker(model, "default")
    explicit = _run_worker(model, "native-auto")
    forced_baseline = _run_worker(model, "native")

    assert default["success"] and explicit["success"]
    assert forced_baseline["success"]
    assert default["verify"] and explicit["verify"]
    assert forced_baseline["verify"]
    assert (
        default["kernel_stats"]
        == explicit["kernel_stats"]
        == forced_baseline["kernel_stats"]
    )
    assert default["helper_count"] == explicit["helper_count"] == 0
    assert forced_baseline["helper_count"] == 0
    assert default["state_restored"] and explicit["state_restored"]
    assert forced_baseline["state_restored"]


def test_baseline_conv_exports_use_the_canonical_vector_kernel_module():
    import ace_edsl.edsl as edsl
    from ace_edsl.edsl.kernels.vector.baseline_conv import (
        baseline_conv_recipe as canonical_recipe,
        baseline_conv_vector_kernel as canonical_kernel,
        configure_baseline_conv_dsl as canonical_configure,
    )
    from ace_edsl.edsl.vector_kernel_baseline_conv import (
        baseline_conv_recipe as compatibility_recipe,
        baseline_conv_vector_kernel as compatibility_kernel,
        configure_baseline_conv_dsl as compatibility_configure,
    )

    assert edsl.baseline_conv_recipe is canonical_recipe
    assert edsl.baseline_conv_vector_kernel is canonical_kernel
    assert edsl.configure_baseline_conv_dsl is canonical_configure
    assert compatibility_recipe is canonical_recipe
    assert compatibility_kernel is canonical_kernel
    assert compatibility_configure is canonical_configure


def test_baseline_conv_freezer_rejects_another_plan_kind():
    from ace_edsl.edsl.vector_kernel_lowering import (
        _freeze_prepared_baseline_conv_plan,
    )

    with pytest.raises(ValueError, match="another plan kind"):
        _freeze_prepared_baseline_conv_plan({"kind": "baseline-gemm"})


def _worker_main():
    from ace_bindings import air_builder
    from ace_edsl.edsl.core.air_value import AIRValue
    from ace_edsl.edsl.domain_ast_decorators import snapshot_tracing_state
    from ace_edsl.edsl.kernels.vector.baseline_conv import baseline_conv_recipe
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    model = Path(sys.argv[2])
    implementation = sys.argv[3]
    behavior = sys.argv[4]
    pipeline = Pipeline(
        "baseline-conv-worker",
        output_dir="/tmp/ace-baseline-conv-worker",
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))

    seen_plans = []
    retained_traces = []
    retained_containers = []
    retained_nodes = []
    retained_types = []
    immutable = []
    owned_constants = []
    type_factory_same_scope = []

    def recipe(trace, prepared):
        seen_plans.append(prepared)
        retained_traces.append(trace)
        retained_containers.append(trace.container)
        formal = trace.formals[0]
        formal_type = formal.rtype()
        i32_type = trace.i32_type
        result_type = trace.result_type
        retained_nodes.append(formal)
        retained_types.extend((formal_type, i32_type, result_type))
        type_factory_same_scope.append(
            i32_type.same_scope(air_builder.Type.make_int(32))
        )
        try:
            prepared.channel_in = 0
        except dataclasses.FrozenInstanceError:
            immutable.append(True)
        owned_constants.append(
            isinstance(prepared.constants, tuple)
            and all(
                isinstance(item.bytes, bytes) for item in prepared.constants
            )
        )
        return baseline_conv_recipe(trace, prepared)

    if implementation != "default":
        auto_kind = implementation.endswith("-auto")
        kernel_impl = (
            implementation.removesuffix("-auto")
            if auto_kind
            else implementation
        )
        plan_kind = "auto" if auto_kind else "baseline-conv"
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl=kernel_impl,
            plan_kind=plan_kind,
            fallback="error",
            mask_fuse=False,
            max_slots=0 if implementation == "native-auto" else _MAX_SLOTS,
        )
        if kernel_impl == "dsl":
            pipeline.register_vector_kernel_recipe("baseline-conv", recipe)

    tracing_state_before = snapshot_tracing_state()
    air_state_before = (
        AIRValue.FLAT_IR_MODE,
        AIRValue.SOURCE_LOC_ENABLED,
        AIRValue.FRESH_LOAD_ENABLED,
        AIRValue._temp_counter,
    )
    result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    state_restored = (
        snapshot_tracing_state() == tracing_state_before
        and (
            AIRValue.FLAT_IR_MODE,
            AIRValue.SOURCE_LOC_ENABLED,
            AIRValue.FRESH_LOAD_ENABLED,
            AIRValue._temp_counter,
        )
        == air_state_before
    )

    if result.success and behavior == "wrong-actual":
        pipeline.glob._mutate_baseline_conv_call_actual_for_testing()

    container_expired = True
    for retained_container in retained_containers:
        for operation in (
            lambda: retained_container.new_intconst(7),
            lambda: retained_container.new_zeros([1], "f32"),
        ):
            try:
                operation()
                container_expired = False
            except RuntimeError as error:
                container_expired &= "expired" in str(error)

    trace_expired = True
    for trace in retained_traces:
        if trace.active:
            trace_expired = False
        for operation in (
            lambda trace=trace: trace.container,
            lambda trace=trace: trace.formals,
        ):
            try:
                operation()
                trace_expired = False
            except RuntimeError as error:
                trace_expired &= "expired" in str(error)

    node_expired = True
    for retained_node in retained_nodes:
        try:
            retained_node.rtype()
            node_expired = False
        except RuntimeError as error:
            node_expired &= "expired" in str(error)

    type_expired = True
    for retained_type in retained_types:
        for operation in (
            retained_type.to_string,
            retained_type.is_array,
            retained_type.is_integer,
            retained_type.is_float,
            retained_type.is_scalar,
            retained_type.bit_width,
            retained_type.shape,
            lambda retained_type=retained_type: retained_type.same_scope(
                retained_type
            ),
        ):
            try:
                operation()
                type_expired = False
            except RuntimeError as error:
                type_expired &= "expired" in str(error)

    if not result.success:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": result.error,
                    "callback_count": len(seen_plans),
                    "trace_expired": trace_expired,
                    "container_expired": container_expired,
                    "node_expired": node_expired,
                    "type_expired": type_expired,
                    "type_factory_same_scope": all(type_factory_same_scope),
                    "state_restored": state_restored,
                }
            )
        )
        return

    dump = pipeline.glob.dump()
    helper_names = sorted(
        set(
            re.findall(
                r'"(__ace_vkernel_baseline_conv_[^"]+)"', dump
            )
        )
    )
    helper_calls = len(
        re.findall(
            r'^\s+call "__ace_vkernel_baseline_conv_',
            dump,
            re.MULTILINE,
        )
    )
    helper_formals = []
    if helper_names:
        helper_marker = re.compile(
            r'^FUN\[[^\n]*"' + re.escape(helper_names[0]) + r'"',
            re.MULTILINE,
        )
        matches = list(helper_marker.finditer(dump))
        helper_body = dump[matches[-1].start() :] if matches else ""
        helper_formals = re.findall(
            r'^\s+FML\[[^]]+\] "([^"]+)"',
            helper_body,
            re.MULTILINE,
        )

    object_oracle = None
    if helper_calls <= 1:
        object_oracle = pipeline.glob._inspect_baseline_conv_air_for_testing()
    normalized = object_oracle["normalized"] if object_oracle else ""

    plan_summary = None
    if seen_plans:
        plan = seen_plans[0]
        weight_slice = plan.slice("weight")
        runtime_input = plan.runtime_preparation("input")
        plan_summary = {
            "kind": plan.kind,
            "provenance": plan.provenance,
            "channel_in": plan.channel_in,
            "channel_out": plan.channel_out,
            "output_height": plan.output_height,
            "output_width": plan.output_width,
            "kernel_hw": plan.kernel_hw,
            "stride": plan.stride,
            "input_duplications": plan.input_duplications,
            "runtime_input_shapes": [
                list(item.shape) for item in plan.runtime_vector_inputs
            ],
            "result_shape": list(plan.result_type.shape),
            "loops": [
                [
                    item.role,
                    item.lower,
                    item.upper,
                    item.step,
                    item.nesting_depth,
                ]
                for item in plan.loops
            ],
            "slice": [
                weight_slice.role,
                list(weight_slice.index.iv_coefficients),
                weight_slice.index.constant,
                weight_slice.index.uses_sharding_offset,
                weight_slice.width,
            ],
            "rotations": [
                [item.role, list(item.candidates)] for item in plan.rotations
            ],
            "mask": [plan.mask.policy, plan.mask.valid_length],
            "slot": [plan.slot.policy, plan.slot.value],
            "constants": [
                [
                    item.role,
                    item.type.element_type,
                    list(item.type.shape),
                    item.content_hash,
                    len(item.bytes),
                ]
                for item in plan.constants
            ],
            "runtime_preparation": [
                runtime_input.role,
                runtime_input.source_operand,
                runtime_input.kind,
                list(runtime_input.result_type.shape),
                runtime_input.logical_input_size,
                runtime_input.replications,
                runtime_input.blocking_width,
                list(runtime_input.rotation_candidates),
                runtime_input.outer_block_depth,
            ],
            "immutable": all(immutable),
            "owned_constants": all(owned_constants),
        }

    print(
        json.dumps(
            {
                "success": True,
                "error": None,
                "verify": pipeline.glob.verify_ir(),
                "stages_completed": result.stages_completed,
                "has_nn_conv": bool(
                    re.search(r"\bNN\.conv\b", dump, re.IGNORECASE)
                ),
                "kernel_stats": _kernel_stats(dump, normalized),
                "helper_count": len(helper_names),
                "helper_calls": helper_calls,
                "helper_formals": helper_formals,
                "callback_count": len(seen_plans),
                "trace_expired": trace_expired,
                "container_expired": container_expired,
                "node_expired": node_expired,
                "type_expired": type_expired,
                "type_factory_same_scope": all(type_factory_same_scope),
                "state_restored": state_restored,
                "object_oracle": object_oracle,
                "plan": plan_summary,
            }
        )
    )


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
