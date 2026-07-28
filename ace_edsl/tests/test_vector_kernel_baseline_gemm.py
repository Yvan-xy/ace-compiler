"""M5 baseline-Gemm destination-recipe integration and parity tests."""

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


_WORKER_FLAG = "--m5-vector-kernel-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_gemm_model(path: Path, height: int, width: int, copies: int = 1):
    values = (np.arange(height * width, dtype=np.float32) + 1.0) / 8.0
    weight = numpy_helper.from_array(values.reshape(height, width), "weight")
    bias = numpy_helper.from_array(
        np.arange(height, dtype=np.float32) / 4.0, "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, width]
    )
    gemm_outputs = []
    nodes = []
    for index in range(copies):
        output_name = f"output_{index}"
        gemm_outputs.append(output_name)
        nodes.append(
            helper.make_node(
                "Gemm",
                ["input", "weight", "bias"],
                [output_name],
                name=f"gemm_{index}",
                transB=1,
            )
        )
    if copies == 1:
        output_name = gemm_outputs[0]
    else:
        output_name = "combined_output"
        nodes.append(
            helper.make_node(
                "Add", gemm_outputs, [output_name], name="combine_gemms"
            )
        )
    output_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, height]
    )
    graph = helper.make_graph(
        nodes, "m5_baseline_gemm", [input_info], [output_info], [weight, bias]
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
    )
    assert completed.returncode == 0, (
        f"worker failed ({' '.join(command)}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    return json.loads(lines[-1])


def _expected_case(height: int, width: int, masked: bool):
    reduction = width // height
    reduction_shifts = []
    while reduction > 1:
        reduction_shifts.append(height << len(reduction_shifts))
        reduction //= 2
    rotations = [[-width], list(range(height))]
    if reduction_shifts:
        rotations.append(reduction_shifts)
    return {
        "height": height,
        "width": width,
        "result_shape": [2 * width],
        "loops": 1 + int(bool(reduction_shifts)),
        "rolls": 2 + int(bool(reduction_shifts)),
        "slices": 1,
        "adds": 3 + int(bool(reduction_shifts)),
        "muls": 1 + int(masked),
        "shls": int(bool(reduction_shifts)),
        "rotations": rotations,
        "masked": masked,
    }


@pytest.mark.parametrize(
    "height,width,mask_fuse",
    [(4, 4, True), (2, 8, False)],
)
def test_cpp_plan_dsl_baseline_gemm_matches_native_structure(
    tmp_path, height, width, mask_fuse
):
    model = tmp_path / f"gemm_{height}x{width}.onnx"
    _write_gemm_model(model, height, width)

    native = _run_worker(model, "native", mask_fuse)
    dsl = _run_worker(model, "dsl", mask_fuse)
    expected = _expected_case(height, width, not mask_fuse)

    assert native["success"] and dsl["success"]
    assert native["verify"] and dsl["verify"]
    assert not native["has_nn_gemm"] and not dsl["has_nn_gemm"]
    assert native["stats"] == dsl["stats"]
    for name in ("loops", "rolls", "slices", "adds", "muls", "shls"):
        assert dsl["stats"][name] == expected[name]
    assert dsl["stats"]["rotations"] == expected["rotations"]
    assert dsl["stats"]["slot_attrs"] == 0
    assert dsl["stats"]["bias_ldcs"] == 1
    assert dsl["stats"]["mask_ldcs"] == int(expected["masked"])
    expected_loop_records = [[0, height, 1, 0]]
    if width > height:
        expected_loop_records.append(
            [0, len(expected["rotations"][-1]), 1, 0]
        )
    assert dsl["stats"]["loop_records"] == expected_loop_records
    expected_roll_operands = [["const", -width], ["iv"]]
    if width > height:
        expected_roll_operands.append(
            ["power-of-two-block", 1, height]
        )
    assert dsl["stats"]["roll_operands"] == expected_roll_operands
    assert dsl["stats"]["slice_records"] == [["iv", width]]
    expected_placements = [["bias", "add"]]
    if expected["masked"]:
        expected_placements.append(["mask", "mul"])
    assert dsl["stats"]["constant_placements"] == expected_placements
    expected_epilogue = ["loops", "bias"]
    if expected["masked"]:
        expected_epilogue.append("mask")
    assert dsl["stats"]["epilogue_order"] == expected_epilogue

    from ace_bindings import air_builder

    assert native["object_oracle"]["kind"] == "native"
    assert dsl["object_oracle"]["kind"] == "helper"
    assert dsl["object_oracle"]["bridge_ok"]
    object_comparison = (
        air_builder._compare_normalized_vector_kernel_air_for_testing(
            native["object_oracle"]["normalized"],
            dsl["object_oracle"]["normalized"],
        )
    )
    assert object_comparison["equal"], object_comparison

    assert native["helper_count"] == 0
    assert native["helper_calls"] == 0
    assert dsl["helper_count"] == 1
    assert dsl["helper_calls"] == 1
    assert dsl["helper_formals"] == ["packed_input_0"]
    assert dsl["bridge"] == {
        "terminal_retvs": 1,
        "calls": 1,
        "ldps": 1,
        "call_before_ldp": True,
        "helper_result_type": f"f32_{2 * width}",
        "caller_result_type": f"f32_{2 * width}",
    }

    plan = dsl["plan"]
    assert plan["kind"] == "baseline-gemm"
    assert plan["height"] == expected["height"]
    assert plan["width"] == expected["width"]
    assert plan["input_duplications"] == 2
    assert plan["result_shape"] == expected["result_shape"]
    assert plan["mask"] == (
        "clear-valid-prefix" if expected["masked"] else "none"
    )
    assert plan["slot"] == "absent-native-baseline"
    assert plan["rotations"] == expected["rotations"]
    expected_loops = [["gemm", 0, height, 1, 0]]
    if width > height:
        expected_loops.append(
            ["block-reduction", 0, len(expected["rotations"][-1]), 1, 0]
        )
    assert plan["loops"] == expected_loops
    assert plan["immutable"]
    assert plan["owned_constants"]
    assert dsl["trace_expired"]
    assert dsl["container_expired"]
    assert dsl["node_expired"]
    assert dsl["type_expired"]
    assert dsl["type_factory_same_scope"]
    assert dsl["state_restored"]


def test_equal_gemm_plans_share_one_helper_and_invoke_body_once(tmp_path):
    model = tmp_path / "two_equal_gemms.onnx"
    _write_gemm_model(model, 4, 4, copies=2)

    result = _run_worker(model, "dsl", True)

    assert result["success"] and result["verify"]
    assert result["callback_count"] == 1
    assert result["trace_expired"]
    assert result["container_expired"]
    assert result["node_expired"]
    assert result["type_expired"]
    assert result["foreign_objects_usable"]
    assert result["type_factory_same_scope"]
    assert result["state_restored"]
    assert result["helper_count"] == 1
    assert result["helper_calls"] == 2


@pytest.mark.parametrize(
    "behavior,diagnostic",
    [
        ("raise", "m5 recipe failure"),
        ("wrong-result", "result type does not match its prepared ABI"),
        ("foreign-result", "node from another AIR container"),
        ("unbalanced", "unbalanced control-flow state"),
    ],
)
def test_recipe_failures_propagate_without_cpp_native_fallback(
    tmp_path, behavior, diagnostic
):
    model = tmp_path / f"negative_{behavior}.onnx"
    _write_gemm_model(model, 4, 4)

    result = _run_worker(model, "dsl", True, behavior)

    assert not result["success"]
    assert diagnostic in result["error"]
    assert result["callback_count"] == 1
    assert result["trace_expired"]
    assert result["container_expired"]
    assert result["node_expired"]
    assert result["type_expired"]
    assert result["foreign_objects_usable"]
    assert result["type_factory_same_scope"]
    assert result["state_restored"]


def test_unconfigured_pipeline_preserves_default_native_behavior(tmp_path):
    model = tmp_path / "default_native.onnx"
    _write_gemm_model(model, 4, 4)

    default = _run_worker(model, "default", False)
    explicit = _run_worker(model, "native-auto", False)

    assert default["success"] and explicit["success"]
    assert default["verify"] and explicit["verify"]
    assert default["stats"] == explicit["stats"]
    assert default["helper_count"] == explicit["helper_count"] == 0
    assert default["state_restored"] and explicit["state_restored"]


def test_vector_recipe_configuration_is_per_pipeline():
    from ace_edsl.edsl.pipeline import Pipeline

    first = Pipeline("first", dump_ir=False, verbose=False)
    second = Pipeline("second", dump_ir=False, verbose=False)
    recipe = lambda trace, plan: None

    first.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl="dsl",
        plan_kind="baseline-gemm",
        fallback="error",
        mask_fuse=True,
        max_slots=8192,
    ).register_vector_kernel_recipe("baseline-gemm", recipe)

    assert first.vector_kernel_config.kernel_impl == "dsl"
    assert first.vector_kernel_config.mask_fuse
    assert first.vector_kernel_config.max_slots == 8192
    assert first.vector_kernel_recipes == {"baseline-gemm": recipe}
    assert second.vector_kernel_config is None
    assert second.vector_kernel_recipes == {}
    with pytest.raises(ValueError, match="duplicate"):
        first.register_vector_kernel_recipe("baseline-gemm", recipe)
    with pytest.raises(ValueError, match="baseline-gemm only"):
        second.register_vector_kernel_recipe("baseline-conv", recipe)
    with pytest.raises(TypeError, match="must be callable"):
        second.register_vector_kernel_recipe("baseline-gemm", None)



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
                if _indent(lines[cursor]) == header_indent and text.startswith(prefix):
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
        records.append([
            _decode_air_int(lines[found["lower"]]),
            upper_values[0],
            step_values[0],
            root_indent,
        ])
    if records:
        base_indent = min(record[3] for record in records)
        for record in records:
            record[3] = (record[3] - base_indent) // 2
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
        widths = [_decode_air_int(child) for child in children
                  if child.lstrip().startswith("intconst ")]
        indices = [child for child in children
                   if child.lstrip().startswith("ld ")]
        assert len(widths) == 1
        index_kind = "iv" if len(indices) == 1 else "expression"
        records.append([index_kind, widths[0]])
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
        elif root.startswith("ld "):
            operands.append(["iv"])
        elif root.startswith("ild "):
            operands.append(["indexed-load"])
        elif root.startswith("mul "):
            root_indent = _indent(lines[cursor])
            subtree = []
            cursor -= 1
            while cursor >= 0 and _indent(lines[cursor]) > root_indent:
                subtree.append(lines[cursor])
                cursor -= 1
            subtree.reverse()
            constants = [_decode_air_int(item) for item in subtree
                         if item.lstrip().startswith("intconst ")]
            if any(item.lstrip().startswith("shl ") for item in subtree):
                assert len(constants) == 2
                operands.append(["power-of-two-block", *constants])
            else:
                operands.append(["mul"])
        else:
            raise AssertionError(f"unsupported AIR roll operand: {root}")
    return operands


def _constant_parent(lines, constant_index):
    constant_indent = _indent(lines[constant_index])
    for index in range(constant_index + 1, len(lines)):
        if not lines[index].strip():
            continue
        indent = _indent(lines[index])
        if indent == constant_indent - 2:
            return index
        if indent < constant_indent - 2:
            break
    raise AssertionError(
        f"prepared constant has no direct AIR user: {lines[constant_index]}"
    )


def _constant_placement_records(dump: str):
    lines = dump.splitlines()
    records = []
    for index, line in enumerate(lines):
        role_match = re.search(r"vector_kernel_(bias|mask)_", line)
        if not role_match or not line.lstrip().startswith("ldc "):
            continue
        parent_index = _constant_parent(lines, index)
        operation = lines[parent_index].lstrip().split(maxsplit=1)[0]
        records.append((role_match.group(1), operation, parent_index))
    return records


def _constant_placements(dump: str):
    return [
        [role, operation.removeprefix("VECTOR.")]
        for role, operation, _ in _constant_placement_records(dump)
    ]


def _epilogue_order(dump: str):
    lines = dump.splitlines()
    loop_indices = [
        index for index, line in enumerate(lines)
        if line.lstrip().startswith("do_loop ID")
    ]
    assert loop_indices
    placements = _constant_placement_records(dump)
    bias = [index for role, operation, index in placements
            if role == "bias" and operation == "VECTOR.add"]
    mask = [index for role, operation, index in placements
            if role == "mask" and operation == "VECTOR.mul"]
    assert len(bias) == 1 and bias[0] > max(loop_indices)
    if mask:
        assert len(mask) == 1 and mask[0] > bias[0]
    return ["loops", "bias"] + (["mask"] if mask else [])


def _node_rtype(line: str):
    match = re.search(r"RTYPE\[[^]]+\]\(([^)]+)\)", line)
    return match.group(1) if match else None


def _bridge_summary(dump: str):
    helper_names = sorted(set(re.findall(
        r"\"(__ace_vkernel_baseline_gemm_[^\"]+)\"", dump
    )))
    if not helper_names:
        return None
    helper_name = helper_names[0]
    helper_markers = list(re.finditer(
        r"^FUN\[[^\n]*\"" + re.escape(helper_name) + r"\"", dump,
        re.MULTILINE,
    ))
    assert helper_markers
    helper_body = dump[helper_markers[-1].start():]
    helper_lines = helper_body.splitlines()
    retv_indices = [index for index, line in enumerate(helper_lines)
                    if line.lstrip().startswith("retv ID")]
    assert retv_indices
    return_index = retv_indices[-1]
    value_index = return_index - 1
    while value_index >= 0 and not helper_lines[value_index].strip():
        value_index -= 1

    caller_start = dump.rfind("FUN[0] \"Main_graph\"", 0,
                              helper_markers[-1].start())
    caller_body = dump[caller_start:helper_markers[-1].start()]
    caller_lines = caller_body.splitlines()
    call_indices = [index for index, line in enumerate(caller_lines)
                    if line.lstrip().startswith(f"call \"{helper_name}")]
    ldp_indices = [index for index, line in enumerate(caller_lines)
                   if line.lstrip().startswith("ldp PREG")]
    ldp_type = _node_rtype(caller_lines[ldp_indices[-1]]) if ldp_indices else None
    return {
        "terminal_retvs": len(retv_indices),
        "calls": len(call_indices),
        "ldps": len(ldp_indices),
        "call_before_ldp": bool(call_indices and ldp_indices
                                and call_indices[-1] < ldp_indices[-1]),
        "helper_result_type": _node_rtype(helper_lines[value_index]),
        "caller_result_type": ldp_type,
    }


def _stats(dump: str):
    rotations = []
    for encoded in re.findall(r"ATTR\[nums=([^]]+)\]", dump):
        rotations.append([int(value) for value in re.findall(r"-?\d+", encoded)])
    return {
        "loops": len(re.findall(r"^\s+do_loop ID", dump, re.MULTILINE)),
        "loop_indents": [
            len(indent)
            for indent in re.findall(r"^(\s+)do_loop ID", dump, re.MULTILINE)
        ],
        "loop_records": _loop_records(dump),
        "rolls": len(re.findall(r"^\s+VECTOR\.roll ", dump, re.MULTILINE)),
        "slices": len(re.findall(r"^\s+VECTOR\.slice ", dump, re.MULTILINE)),
        "adds": len(re.findall(r"^\s+VECTOR\.add ", dump, re.MULTILINE)),
        "muls": len(re.findall(r"^\s+VECTOR\.mul ", dump, re.MULTILINE)),
        "shls": len(re.findall(r"^\s+shl RTYPE", dump, re.MULTILINE)),
        "rotations": rotations,
        "roll_operands": _roll_operands(dump),
        "slice_records": _slice_records(dump),
        "constant_placements": _constant_placements(dump),
        "epilogue_order": _epilogue_order(dump),
        "slot_attrs": len(re.findall(
            r"ATTR\[[^\n]*\bslot\b", dump, re.IGNORECASE
        )),
        "bias_ldcs": len(
            re.findall(r"^\s+ldc .*\(vector_kernel_bias_", dump, re.MULTILINE)
        ),
        "mask_ldcs": len(
            re.findall(r"^\s+ldc .*\(vector_kernel_mask_", dump, re.MULTILINE)
        ),
    }


def _worker_main():
    from ace_bindings import air_builder
    from ace_edsl.edsl.core.air_value import AIRValue
    from ace_edsl.edsl.domain_ast_decorators import snapshot_tracing_state
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget
    from ace_edsl.edsl.vector_kernel_baseline_gemm import baseline_gemm_recipe

    model = Path(sys.argv[2])
    implementation = sys.argv[3]
    mask_fuse = sys.argv[4] == "1"
    behavior = sys.argv[5]
    pipeline = Pipeline(
        "m5-worker", output_dir="/tmp/ace-m5-worker", dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))

    seen_plans = []
    retained_traces = []
    retained_containers = []
    retained_nodes = []
    retained_types = []
    foreign_objects = []
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
            prepared.height = 0
        except dataclasses.FrozenInstanceError:
            immutable.append(True)
        owned_constants.append(
            isinstance(prepared.constants, tuple)
            and all(isinstance(item.bytes, bytes) for item in prepared.constants)
        )
        if behavior == "raise":
            raise RuntimeError("m5 recipe failure")
        if behavior == "wrong-result":
            return formal
        if behavior == "foreign-result":
            foreign_glob = air_builder.GlobScope()
            foreign_type = foreign_glob.new_array_type(
                formal_type.shape(), "f32"
            )
            foreign_scope = foreign_glob.new_func_with_param_types(
                "foreign", foreign_type, [foreign_type]
            )
            foreign_node = foreign_scope.new_param(
                "foreign_input", foreign_type
            )
            foreign_objects.extend(
                (foreign_glob, foreign_scope, foreign_type, foreign_node)
            )
            return foreign_node
        result = baseline_gemm_recipe(trace, prepared)
        if behavior == "unbalanced":
            trace.container.new_loop_begin_range(0, 1)
        return result

    if implementation != "default":
        kernel_impl = "native" if implementation == "native-auto" else implementation
        plan_kind = "auto" if implementation == "native-auto" else "baseline-gemm"
        fallback = "cpp-native" if behavior == "raise" else "error"
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl=kernel_impl,
            plan_kind=plan_kind,
            fallback=fallback,
            mask_fuse=mask_fuse,
        )
        if kernel_impl == "dsl":
            pipeline.register_vector_kernel_recipe("baseline-gemm", recipe)

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
        ) == air_state_before
    )

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

    foreign_objects_usable = True
    if foreign_objects:
        foreign_glob, foreign_scope, foreign_type, foreign_node = (
            foreign_objects
        )
        try:
            foreign_objects_usable = (
                foreign_type.is_array()
                and foreign_node.rtype().same_scope(foreign_type)
                and foreign_glob.get_type("f32").is_float()
                and foreign_glob.new_array_type([2], "f32").shape() == [2]
                and isinstance(foreign_scope.dump(), str)
            )
        except RuntimeError:
            foreign_objects_usable = False

    if not result.success:
        print(json.dumps({
            "success": False,
            "error": result.error,
            "callback_count": len(seen_plans),
            "trace_expired": trace_expired,
            "container_expired": container_expired,
            "node_expired": node_expired,
            "type_expired": type_expired,
            "foreign_objects_usable": foreign_objects_usable,
            "type_factory_same_scope": all(type_factory_same_scope),
            "state_restored": state_restored,
        }))
        return

    dump = pipeline.glob.dump()
    helper_names = sorted(set(re.findall(r'"(__ace_vkernel_baseline_gemm_[^"]+)"', dump)))
    helper_calls = len(
        re.findall(r'^\s+call "__ace_vkernel_baseline_gemm_', dump, re.MULTILINE)
    )
    helper_formals = []
    if helper_names:
        helper_marker = re.compile(
            r'^FUN\[[^\n]*"' + re.escape(helper_names[0]) + r'"',
            re.MULTILINE,
        )
        matches = list(helper_marker.finditer(dump))
        helper_body = dump[matches[-1].start():] if matches else ""
        helper_formals = re.findall(r'^\s+FML\[[^]]+\] "([^"]+)"', helper_body, re.MULTILINE)

    object_oracle = None
    if helper_calls <= 1:
        object_oracle = pipeline.glob._inspect_baseline_gemm_air_for_testing()

    plan_summary = None
    if seen_plans:
        plan = seen_plans[0]
        plan_summary = {
            "kind": plan.kind,
            "height": plan.height,
            "width": plan.width,
            "input_duplications": plan.input_duplications,
            "result_shape": list(plan.result_type.shape),
            "loops": [
                [item.role, item.lower, item.upper, item.step, item.nesting_depth]
                for item in plan.loops
            ],
            "rotations": [list(item.candidates) for item in plan.rotations],
            "mask": plan.mask.policy,
            "slot": plan.slot.policy,
            "immutable": all(immutable),
            "owned_constants": all(owned_constants),
        }

    print(json.dumps({
        "success": True,
        "error": None,
        "verify": pipeline.glob.verify_ir(),
        "has_nn_gemm": bool(re.search(r"\bNN\.gemm\b", dump, re.IGNORECASE)),
        "stats": _stats(dump),
        "helper_count": len(helper_names),
        "helper_calls": helper_calls,
        "helper_formals": helper_formals,
        "callback_count": len(seen_plans),
        "trace_expired": trace_expired,
        "container_expired": container_expired,
        "node_expired": node_expired,
        "type_expired": type_expired,
        "foreign_objects_usable": foreign_objects_usable,
        "type_factory_same_scope": all(type_factory_same_scope),
        "state_restored": state_restored,
        "bridge": _bridge_summary(dump),
        "object_oracle": object_oracle,
        "plan": plan_summary,
    }))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
