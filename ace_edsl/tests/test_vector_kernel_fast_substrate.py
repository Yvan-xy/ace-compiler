"""Typed mutable Vector arrays and reusable fast-helper topology."""

from __future__ import annotations

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

from ace_bindings import air_builder
from ace_edsl.edsl import AceEDSL, VectorArray, range_dynamic, vector_kernel
from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core.types import VectorTensor
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.kernels.vector.fast_common import (
    _ranked_constant,
    blocking_rot,
    clear_valid_data,
    collective_reduce,
    reduce_add_intra,
    roll_cyclic,
)
from ace_edsl.edsl.vector_kernel_lowering import (
    ConstantPlan,
    LoopPlan,
    MaskPlan,
    PreparedFastConvPlan,
    RankedTypePlan,
    ReductionPlan,
    RotationPlan,
    SlotPlan,
    vector_kernel_recipe,
)


_WORKER_FLAG = "--fast-vector-substrate-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fresh_dsl():
    AceEDSL._get_dsl.cache_clear()
    return AceEDSL._get_dsl()


def _vector_value(node, container, air_type, temp_name=None):
    return VectorValue(
        node,
        container,
        shape=tuple(air_type.shape()),
        temp_name=temp_name,
        air_type=air_type,
    )


_BLOCK_COLLECTIVE_PLAN = PreparedFastConvPlan(
    kind="fast-conv",
    provenance="test",
    specialization_key="collective-block-helper",
    helper_name="__collective_block_helper",
    channel_in=2,
    channel_out=2,
    output_height=1,
    output_width=2,
    kernel_hw=3,
    group=1,
    stride=1,
    input_size=4,
    output_size=4,
    num_slots=16,
    num_grid=1,
    num_block=2,
    width_block=6,
    width_block_data=4,
    width_block_pad=2,
    position_block=2,
    capacity_block=3,
    input_duplications=2,
    blocking_outer_depth=0,
    cyclic_roll=False,
    sharding_offset=None,
    runtime_vector_inputs=(RankedTypePlan("f32", (16,)),),
    runtime_scalar_inputs=(),
    result_type=RankedTypePlan("f32", (16,)),
    loops=(
        LoopPlan("collective-data", 1, 2, 1, 0),
        LoopPlan("collective-gap", 0, 1, 1, 0),
    ),
    slices=(),
    rotations=(
        RotationPlan("collective-data", (6,)),
        RotationPlan("collective-gap", (2,)),
        RotationPlan("collective-tail", (-4,)),
    ),
    reductions=(
        ReductionPlan("collective", "collective-blocks", 2, 4, 2),
    ),
    mask=MaskPlan("collective-reduction", 4),
    slot=SlotPlan("logical-output-elements", 4),
    constants=(
        ConstantPlan(
            "collective-mask",
            RankedTypePlan("f32", (4,)),
            "sha256:test-data-mask",
            b"",
        ),
        ConstantPlan(
            "collective-gap-mask",
            RankedTypePlan("f32", (4,)),
            "sha256:test-gap-mask",
            b"",
        ),
    ),
    runtime_preparations=(),
    scalar_preparations=(),
)


@vector_kernel
def _collective_blocks_topology(
    value: VectorTensor[float, 16],
) -> VectorTensor[float, 16]:
    container = value.container
    data_node = container.new_array_const([1.0, 1.0, 1.0, 1.0])
    gap_node = container.new_array_const([0.0, 0.0, 1.0, 1.0])
    constants = {
        "collective-mask": _vector_value(
            data_node, container, data_node.rtype()
        ),
        "collective-gap-mask": _vector_value(
            gap_node, container, gap_node.rtype()
        ),
    }
    return collective_reduce(
        value,
        _BLOCK_COLLECTIVE_PLAN,
        constants,
        air_builder.Type.make_int(32),
        "__collective_result",
    )


@vector_kernel
def _both_intra_reduction_kinds(
    value: VectorTensor[float, 8],
) -> VectorTensor[float, 8]:
    power = reduce_add_intra(
        value,
        ReductionPlan("power", "power-of-two", 4, 1, 0),
        RotationPlan("power", (1, 2)),
        air_builder.Type.make_int(32),
        "__power_reduction",
    )
    return reduce_add_intra(
        power,
        ReductionPlan("linear", "linear", 3, 1, 0),
        RotationPlan("linear", (1, 2)),
        air_builder.Type.make_int(32),
        "__linear_reduction",
    )


@vector_kernel
def _fast_substrate_smoke(
    packed_input,
    prepared,
    constants,
    result_type,
    i32_type,
):
    container = packed_input.container
    blocked = blocking_rot(
        packed_input,
        prepared,
        constants,
        i32_type,
        "__blocked_input",
        "__duplicated_input",
    )

    consumed_name = "__consumed_input"
    container.new_local(consumed_name, packed_input.air_type)
    container.new_stid(
        consumed_name, container.new_zero(packed_input.air_type)
    )
    preparation = prepared.runtime_preparation("input")
    for iv in range_dynamic(0, preparation.blocking_width, 1):
        current = _vector_value(
            None, container, packed_input.air_type, consumed_name
        )
        container.new_stid(
            consumed_name, (current + blocked.load(iv)).value
        )

    result = _vector_value(
        container.new_zero(result_type), container, result_type
    )
    if prepared.kind == "fast-gemm":
        for reduction in prepared.reductions:
            result = reduce_add_intra(
                result,
                reduction,
                prepared.rotation(reduction.role),
                i32_type,
                "__intra_" + reduction.role,
            )
        result = clear_valid_data(
            result, prepared, constants, "__valid_result"
        )
    elif prepared.cyclic_roll:
        cyclic_name = "__cyclic_result"
        container.new_local(cyclic_name, result_type)
        container.new_stid(cyclic_name, result.value)
        grid = prepared.loop("grid")
        for iv in range_dynamic(grid.lower, grid.upper, grid.step):
            current = _vector_value(
                None, container, result_type, cyclic_name
            )
            container.new_stid(
                cyclic_name,
                roll_cyclic(
                    current, iv, prepared, constants, i32_type
                ).value,
            )
        result = _vector_value(None, container, result_type, cyclic_name)
    elif prepared.reductions:
        result = collective_reduce(
            result,
            prepared,
            constants,
            i32_type,
            "__collective_result",
        )
    return result


_FAST_SUBSTRATE_RECIPE = vector_kernel_recipe(_fast_substrate_smoke)


def test_rank_one_vector_array_emits_real_producer_and_consumer_loops():
    glob = air_builder.create_glob_scope()
    f32 = air_builder.Type.make_float(32)
    vector8 = air_builder.Type.make_array([8], f32)
    outer = air_builder.Type.make_array([3], vector8)

    assert outer.rank() == 1
    assert outer.element_type() == vector8
    assert vector8.rank() == 1
    assert vector8.element_type() == f32

    function = glob.new_func_with_param_types(
        "vector_array_producer_consumer", vector8, [vector8]
    )
    container = function.container()
    value = function.new_param("value", vector8)
    container.new_local("vectors", outer)

    lda = container.new_lda(container.new_ldid("vectors"))
    assert lda.opcode_name() == "air::core::LDA"
    direct_address = container.new_array(lda, container.new_index_const(0))
    assert direct_address.opcode_name() == "air::core::ARRAY"

    producer = container.new_loop_begin_range(0, 3, 32)
    producer_iv = container.new_loop_index(producer)
    store = container.new_ist(
        value, container.new_ldid("vectors"), producer_iv
    )
    assert store.opcode_name() == "air::core::IST"
    container.new_loop_end()

    container.new_local("seen", vector8)
    consumer = container.new_loop_begin_range(0, 3, 32)
    consumer_iv = container.new_loop_index(consumer)
    address = container.new_array(
        container.new_ldid("vectors"), consumer_iv
    )
    loaded = container.new_ild(address)
    assert loaded.opcode_name() == "air::core::ILD"
    assert loaded.rtype() == vector8
    container.new_stid("seen", loaded)
    container.new_loop_end()
    container.new_retv(container.new_ldid("seen"))

    assert glob.verify_ir()
    dump = glob.dump().lower()
    assert dump.count("do_loop id") == 2
    assert dump.count("\n          ist id") == 1
    assert dump.count("\n            ild rtype") == 1
    assert dump.count("lda \"vectors\"") >= 2
    assert dump.count("array rtype") >= 2
    first_loop = dump.index("do_loop id")
    second_loop = dump.index("do_loop id", first_loop + 1)
    assert dump.index("ist id") < first_loop
    assert first_loop < dump.index("ild rtype") < second_loop


def test_vector_array_store_is_inserted_in_innermost_loop():
    glob = air_builder.create_glob_scope()
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)
    outer_type = air_builder.Type.make_array([2], vector4)
    function = glob.new_func_with_param_types(
        "nested_vector_array_store", vector4, [vector4]
    )
    container = function.container()
    value = function.new_param("value", vector4)
    container.new_local("vectors", outer_type)

    outer_loop = container.new_loop_begin_range(0, 2, 32)
    container.new_loop_index(outer_loop)
    inner_loop = container.new_loop_begin_range(0, 2, 32)
    inner_iv = container.new_loop_index(inner_loop)
    store = container.new_ist(
        value, container.new_ldid("vectors"), inner_iv
    )
    assert store.opcode_name() == "air::core::IST"
    container.new_loop_end()
    container.new_loop_end()
    container.new_retv(value)

    assert glob.verify_ir()
    lines = glob.dump().lower().splitlines()
    store_lines = [line for line in lines if line.lstrip().startswith("ist id")]
    loop_lines = [
        line for line in lines if line.lstrip().startswith("do_loop id")
    ]
    assert len(store_lines) == 1
    assert len(loop_lines) == 2
    store_indent = len(store_lines[0]) - len(store_lines[0].lstrip())
    loop_indents = [
        len(line) - len(line.lstrip()) for line in loop_lines
    ]
    assert store_indent > max(loop_indents)


def test_vector_array_rejects_wrong_rank_index_owner_and_store_type():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    i64 = air_builder.Type.make_int(64)
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)
    vector8 = air_builder.Type.make_array([8], f32)
    scalar_outer = air_builder.Type.make_array([2], f32)
    rank_two_outer = air_builder.Type.make_array([2, 2], vector4)
    vector_outer = air_builder.Type.make_array([2], vector4)

    function = glob.new_func_with_param_types(
        "vector_array_validation",
        vector4,
        [vector4, vector8, i32, i64],
    )
    container = function.container()
    narrow = function.new_param("narrow", vector4)
    wide = function.new_param("wide", vector8)
    index = function.new_param("index", i32)
    wide_index = function.new_param("wide_index", i64)
    container.new_local("scalar_outer", scalar_outer)
    container.new_local("rank_two_outer", rank_two_outer)
    container.new_local("vector_outer", vector_outer)

    before = glob.dump().lower().count("ist id")
    with pytest.raises(RuntimeError, match="mutable array of ranked vectors"):
        container.new_ist(
            narrow, container.new_ldid("scalar_outer"), index
        )
    scalar = container.new_ild(
        container.new_ldid("scalar_outer"), wide_index
    )
    assert scalar.rtype() == f32
    with pytest.raises(RuntimeError, match="rank-one outer array"):
        container.new_ild(container.new_ldid("rank_two_outer"), index)
    with pytest.raises(RuntimeError, match="signed i32"):
        container.new_ild(container.new_ldid("vector_outer"), wide_index)
    with pytest.raises(RuntimeError, match="value type does not match"):
        container.new_ist(wide, container.new_ldid("vector_outer"), index)
    with pytest.raises(RuntimeError, match="signed i32 range"):
        container.new_index_const(1 << 31)

    constant = container.new_array_const([1.0, 2.0, 3.0])
    selected = container.new_ild(constant, index)
    assert selected.rtype() == f32
    with pytest.raises(RuntimeError, match="mutable array of ranked vectors"):
        container.new_ist(narrow, constant, index)

    foreign_function = glob.new_func_with_param_types(
        "foreign_vector_array_index", i32, [i32]
    )
    foreign_index = foreign_function.new_param("index", i32)
    with pytest.raises(RuntimeError, match="container"):
        container.new_ild(
            container.new_ldid("vector_outer"), foreign_index
        )
    foreign_function.container().new_retv(foreign_index)

    container.new_preg("epilogue", vector4)
    stored = container.new_stp("epilogue", narrow)
    assert stored.opcode_name() == "air::core::STP"
    assert container.new_ldp("epilogue").rtype() == vector4
    container.new_preg("epilogue", vector4)
    with pytest.raises(RuntimeError, match="incompatible type"):
        container.new_preg("epilogue", vector8)
    with pytest.raises(RuntimeError, match="pseudo-register type"):
        container.new_stp("epilogue", wide)
    with pytest.raises(RuntimeError, match="Unknown pseudo-register"):
        container.new_ldp("missing")

    assert glob.dump().lower().count("ist id") == before
    container.new_retv(narrow)
    assert glob.verify_ir()


def test_vector_array_facade_is_typed_and_container_owned():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)
    vector8 = air_builder.Type.make_array([8], f32)
    function = glob.new_func_with_param_types(
        "vector_array_facade", vector4, [vector4, vector8, i32]
    )
    container = function.container()
    narrow_node = function.new_param("narrow", vector4)
    wide_node = function.new_param("wide", vector8)
    index_node = function.new_param("index", i32)
    narrow = _vector_value(narrow_node, container, vector4)
    wide = _vector_value(wide_node, container, vector8)
    index = AIRValue(
        index_node,
        container,
        domain="air::core",
        air_type=i32,
    )

    values = VectorArray(container, "values", 2, vector4)
    assert values.extent == 2
    assert values.element_type == vector4
    assert values.air_type.rank() == 1
    assert values.air_type.element_type() == vector4
    assert values.address().opcode_name() == "air::core::LDA"
    values.store(index, narrow)
    loaded = values.load(index)
    assert isinstance(loaded, VectorValue)
    assert loaded.air_type == vector4

    with pytest.raises(TypeError, match="value type"):
        values.store(index, wide)
    with pytest.raises(TypeError, match="signed Core i32"):
        values.load(True)
    with pytest.raises(ValueError, match="positive signed i32"):
        VectorArray(container, "empty", 0, vector4)
    with pytest.raises(TypeError, match="ranked vectors"):
        VectorArray(container, "scalars", 2, f32)

    foreign = glob.new_func_with_param_types(
        "foreign_vector_array_value", vector4, [vector4, i32]
    )
    foreign_value_node = foreign.new_param("value", vector4)
    foreign_index_node = foreign.new_param("index", i32)
    foreign_value = _vector_value(
        foreign_value_node, foreign.container(), vector4
    )
    foreign_index = AIRValue(
        foreign_index_node,
        foreign.container(),
        domain="air::core",
        air_type=i32,
    )
    with pytest.raises(ValueError, match="another container"):
        values.store(index, foreign_value)
    with pytest.raises(ValueError, match="another container"):
        values.load(foreign_index)
    foreign.container().new_retv(foreign_value_node)

    container.new_retv(loaded.value)
    assert glob.verify_ir()


def test_ranked_helper_constants_require_exact_prepared_role_and_type():
    glob = air_builder.create_glob_scope()
    f32 = air_builder.Type.make_float(32)
    vector8 = air_builder.Type.make_array([8], f32)
    function = glob.new_func_with_param_types(
        "ranked_helper_constant_validation", vector8, [vector8]
    )
    container = function.container()
    value = function.new_param("value", vector8)
    wrong_node = container.new_array_const([1.0, 1.0, 1.0])
    wrong = _vector_value(wrong_node, container, wrong_node.rtype())
    wrong_element_node = container.new_array_const(
        [complex(1.0, 0.0), complex(1.0, 0.0)]
    )
    wrong_element = _vector_value(
        wrong_element_node, container, wrong_element_node.rtype()
    )

    with pytest.raises(KeyError, match="missing prepared constant role"):
        _ranked_constant(
            _BLOCK_COLLECTIVE_PLAN, {}, "collective-mask"
        )
    with pytest.raises(TypeError, match="wrong shape"):
        _ranked_constant(
            _BLOCK_COLLECTIVE_PLAN,
            {"collective-mask": wrong},
            "collective-mask",
        )
    with pytest.raises(TypeError, match="wrong element type"):
        _ranked_constant(
            _BLOCK_COLLECTIVE_PLAN,
            {"collective-mask": wrong_element},
            "collective-mask",
        )

    container.new_retv(value)
    assert glob.verify_ir()


def test_intra_reduction_helpers_cover_power_of_two_and_linear_topology():
    dsl = _fresh_dsl()
    result = _both_intra_reduction_kinds(VectorTensor[float, 8])

    assert isinstance(result, VectorValue)
    assert result.air_type.shape() == [8]
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump()
    assert dump.count("do_loop ID") == 2
    assert dump.count("VECTOR.roll ") == 2
    assert dump.count("VECTOR.add ") == 2
    assert "ATTR[nums=(1,2)]" in dump
    assert re.search(r"\n\s+shl RTYPE.*int32", dump)


def test_collective_blocks_helper_matches_native_loop_and_preg_topology():
    dsl = _fresh_dsl()
    result = _collective_blocks_topology(VectorTensor[float, 16])

    assert isinstance(result, VectorValue)
    assert result.air_type.shape() == [16]
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump()
    assert dump.count("do_loop ID") == 2
    assert dump.count("stp PREG") >= 3
    assert dump.count("ldp PREG") >= 3
    assert dump.count("VECTOR.roll ") == 3
    assert dump.count("VECTOR.mul ") >= 3
    assert "ATTR[nums=6]" in dump
    assert "ATTR[nums=2]" in dump
    assert "ATTR[nums=-4]" in dump


def _write_gemm_model(path: Path, height: int = 2, width: int = 4):
    weight_values = np.arange(
        1, height * width + 1, dtype=np.float32
    ).reshape(height, width)
    weight = numpy_helper.from_array(weight_values, "weight")
    bias = numpy_helper.from_array(
        np.array([0.25, -0.5], dtype=np.float32), "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, width]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, height]
    )
    node = helper.make_node(
        "Gemm", ["input", "weight", "bias"], ["output"],
        name="gemm", transB=1,
    )
    graph = helper.make_graph(
        [node], "fast_substrate_gemm", [input_info], [output_info],
        [weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_conv_model(
    path: Path,
    channel_in: int,
    channel_out: int,
    spatial: int,
    kernel: int,
):
    weight_values = (
        np.arange(
            channel_out * channel_in * kernel * kernel,
            dtype=np.float32,
        )
        + 1.0
    ) / 32.0
    weight = numpy_helper.from_array(
        weight_values.reshape(channel_out, channel_in, kernel, kernel),
        "weight",
    )
    bias = numpy_helper.from_array(
        np.arange(channel_out, dtype=np.float32) / 4.0, "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, channel_in, spatial, spatial]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, channel_out, spatial, spatial]
    )
    pad = kernel // 2
    node = helper.make_node(
        "Conv", ["input", "weight", "bias"], ["output"], name="conv",
        kernel_shape=[kernel, kernel], pads=[pad, pad, pad, pad],
        strides=[1, 1], group=1,
    )
    graph = helper.make_graph(
        [node], "fast_substrate_conv", [input_info], [output_info],
        [weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run_worker(model: Path, plan_kind: str, max_slots: int):
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
        plan_kind,
        str(max_slots),
    ]
    completed = subprocess.run(
        command, cwd=_REPO_ROOT, env=env, capture_output=True, text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"worker failed ({' '.join(command)}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    return json.loads(lines[-1])


def _operation_count(dump: str, operation: str) -> int:
    return len(re.findall(
        rf"^\s+{re.escape(operation)}(?:\s|$)", dump, re.MULTILINE
    ))


def _plan_summary(prepared):
    common = {
        "kind": prepared.kind,
        "provenance": prepared.provenance,
        "helper_name": prepared.helper_name,
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
            [item.role, list(item.index.iv_coefficients), item.index.constant,
             item.index.uses_sharding_offset, item.width]
            for item in prepared.slices
        ],
        "rotations": [
            [item.role, list(item.candidates)] for item in prepared.rotations
        ],
        "reductions": [
            [item.role, item.kind, item.factor, item.block_width, item.padding]
            for item in prepared.reductions
        ],
        "mask": [prepared.mask.policy, prepared.mask.valid_length],
        "slot": [prepared.slot.policy, prepared.slot.value],
        "constants": [
            [item.role, item.type.element_type, list(item.type.shape),
             item.content_hash, len(item.bytes)]
            for item in prepared.constants
        ],
        "runtime_preparations": [
            [item.role, item.source_operand, item.kind,
             item.result_type.element_type, list(item.result_type.shape),
             item.logical_input_size, item.replications,
             item.blocking_width, list(item.rotation_candidates),
             item.outer_block_depth]
            for item in prepared.runtime_preparations
        ],
        "scalar_preparations": [
            [item.role, item.source_operand, item.type, item.scale]
            for item in prepared.scalar_preparations
        ],
    }
    if prepared.kind == "fast-gemm":
        common["dimensions"] = [
            prepared.n, prepared.k, prepared.np, prepared.kp,
            prepared.nd, prepared.kd,
        ]
        common["blocking"] = [
            prepared.block_size, prepared.blocks_per_partition,
            prepared.packed_partitions, prepared.shift,
            prepared.shift_buffer, prepared.grid_size,
            prepared.input_replications,
        ]
    else:
        common["dimensions"] = [
            prepared.channel_in, prepared.channel_out,
            prepared.output_height, prepared.output_width,
            prepared.kernel_hw, prepared.group, prepared.stride,
        ]
        common["blocking"] = [
            prepared.input_size, prepared.output_size, prepared.num_slots,
            prepared.num_grid, prepared.num_block, prepared.width_block,
            prepared.width_block_data, prepared.width_block_pad,
            prepared.position_block, prepared.capacity_block,
            prepared.input_duplications, prepared.blocking_outer_depth,
        ]
        common["cyclic_roll"] = prepared.cyclic_roll
        common["sharding_offset"] = (
            None if prepared.sharding_offset is None else [
                prepared.sharding_offset.type, prepared.sharding_offset.scale
            ]
        )
    return common


def _worker_main():
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    model = Path(sys.argv[2])
    plan_kind = sys.argv[3]
    max_slots = int(sys.argv[4])
    pipeline = Pipeline(
        "fast-vector-substrate",
        output_dir="/tmp/ace-fast-vector-substrate",
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))
    seen_plans = []

    def recipe(trace, prepared):
        seen_plans.append(prepared)
        return _FAST_SUBSTRATE_RECIPE(trace, prepared)

    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp", kernel_impl="dsl", plan_kind=plan_kind,
        fallback="error", mask_fuse=False, max_slots=max_slots,
    ).register_vector_kernel_recipe(plan_kind, recipe)
    result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    assert result.success
    assert len(seen_plans) == 1

    prepared = seen_plans[0]
    dump = pipeline.glob.dump()
    helper_definition = bool(re.search(
        rf'^FUN\[[^]]+\] "{re.escape(prepared.helper_name)}"',
        dump, re.MULTILINE,
    ))
    helper_calls = len(re.findall(
        rf'^\s+call "{re.escape(prepared.helper_name)}"',
        dump, re.MULTILINE,
    ))
    print(json.dumps({
        "success": result.success,
        "verify": pipeline.glob.verify_ir(),
        "stages_completed": result.stages_completed,
        "has_source_op": bool(re.search(r"\bNN\.(gemm|conv)\b", dump)),
        "helper_definition": helper_definition,
        "helper_calls": helper_calls,
        "plan": _plan_summary(prepared),
        "stats": {
            operation: _operation_count(dump, operation)
            for operation in (
                "lda", "array", "ild", "ist", "do_loop", "VECTOR.roll",
                "VECTOR.slice", "VECTOR.add", "VECTOR.mul", "stp", "ldp",
            )
        },
        "comments": [
            text for text in (
                "BlockingRot replicate=", "BlockingRot HRot=bs="
            ) if text in dump
        ],
        "rotation_attributes": re.findall(
            r"ATTR\[nums=([^]]+)\]", dump
        ),
        "dump_has_rotation_table": "vector_kernel_rotation-table" in dump,
        "dump_has_mask": any(role in dump for role in (
            "vector_kernel_mask", "vector_kernel_collective-mask",
            "vector_kernel_cyclic-mask-left",
        )),
    }))


def _constant(summary, role):
    matches = [item for item in summary["constants"] if item[0] == role]
    assert len(matches) == 1
    return matches[0]


def test_forced_fast_gemm_smoke_consumes_cpp_plan_and_verifies_pre_inline(
    tmp_path,
):
    model = tmp_path / "gemm.onnx"
    _write_gemm_model(model)
    smoke = _run_worker(model, "fast-gemm", 128)

    assert smoke["success"] and smoke["verify"]
    assert smoke["stages_completed"] == ["tensor2vector"]
    assert not smoke["has_source_op"]
    assert smoke["helper_definition"] and smoke["helper_calls"] == 1
    plan = smoke["plan"]
    assert plan["kind"] == "fast-gemm"
    assert plan["provenance"] == "cpp"
    assert plan["dimensions"] == [2, 4, 2, 4, 2, 4]
    assert plan["blocking"] == [1, 2, 1, 1, 2, 2, 2]
    assert plan["result_type"] == ["f32", [6]]
    assert plan["runtime_scalar_inputs"] == []
    assert plan["scalar_preparations"] == []
    assert plan["runtime_preparations"] == [
        ["input", 0, "blocking-rotations", "f32", [1, 4], 4, 2, 1, [0], 0]
    ]
    assert plan["reductions"] == [
        ["kp-over-np", "power-of-two", 2, 2, 0]
    ]
    assert plan["mask"] == ["clear-valid-prefix", 2]
    assert _constant(plan, "rotation-table")[1:3] == ["s32", [1]]
    assert _constant(plan, "rotation-table")[4] == 4
    assert _constant(plan, "mask")[1:3] == ["f32", [2]]
    assert _constant(plan, "mask")[4] == 8
    assert all(item[3].startswith("sha256:") for item in plan["constants"])

    stats = smoke["stats"]
    assert stats["ist"] == 1
    assert stats["ild"] >= 2
    assert stats["lda"] >= 2
    assert stats["array"] >= 2
    assert stats["do_loop"] >= 3
    assert stats["VECTOR.roll"] >= 3
    assert stats["VECTOR.mul"] >= 1
    assert smoke["comments"] == [
        "BlockingRot replicate=", "BlockingRot HRot=bs="
    ]
    assert smoke["rotation_attributes"] == ["-4", "0", "2"]
    assert smoke["dump_has_rotation_table"]
    assert smoke["dump_has_mask"]


def test_forced_fast_conv_single_collective_consumes_plan_topology(
    tmp_path,
):
    model = tmp_path / "conv_collective.onnx"
    _write_conv_model(model, 2, 4, 2, 3)
    smoke = _run_worker(model, "fast-conv", 128)

    assert smoke["success"] and smoke["verify"]
    assert smoke["stages_completed"] == ["tensor2vector"]
    assert not smoke["has_source_op"]
    assert smoke["helper_definition"] and smoke["helper_calls"] == 1
    plan = smoke["plan"]
    assert plan["kind"] == "fast-conv"
    assert plan["provenance"] == "cpp"
    assert plan["dimensions"] == [2, 4, 2, 2, 9, 1, 1]
    assert plan["blocking"] == [
        8, 16, 128, 2, 1, 16, 16, 0, 4, 9, 2, 0
    ]
    assert not plan["cyclic_roll"]
    assert plan["sharding_offset"] is None
    assert plan["result_type"] == ["f32", [1, 4, 2, 2]]
    assert plan["runtime_preparations"] == [[
        "input", 0, "blocking-rotations", "f32", [8], 8, 2, 9,
        [-3, -2, -1, -1, 0, 1, 1, 2, 3], 0,
    ]]
    assert plan["reductions"] == [
        ["collective", "collective-single-block", 1, 16, 0]
    ]
    assert plan["mask"] == ["collective-reduction", 16]
    assert _constant(plan, "rotation-table")[1:3] == ["s32", [9]]
    assert _constant(plan, "rotation-table")[4] == 36
    assert _constant(plan, "collective-mask")[1:3] == ["f32", [16]]

    stats = smoke["stats"]
    assert stats["ist"] == 1
    assert stats["ild"] >= 2
    assert stats["do_loop"] >= 2
    assert stats["VECTOR.roll"] >= 3
    assert stats["VECTOR.mul"] >= 1
    assert stats["stp"] >= 1
    assert stats["ldp"] >= 1
    assert smoke["rotation_attributes"] == [
        "-8",
        "(-3,-2,-1,-1,0,1,1,2,3)",
        "-16",
    ]
    assert smoke["dump_has_rotation_table"]
    assert smoke["dump_has_mask"]


def test_forced_fast_conv_cyclic_consumes_ranked_masks_and_two_rolls(
    tmp_path,
):
    model = tmp_path / "conv_cyclic.onnx"
    _write_conv_model(model, 2, 3, 5, 1)
    smoke = _run_worker(model, "fast-conv", 128)

    assert smoke["success"] and smoke["verify"]
    assert smoke["helper_definition"] and smoke["helper_calls"] == 1
    plan = smoke["plan"]
    assert plan["dimensions"] == [3, 3, 5, 5, 1, 1, 1]
    assert plan["cyclic_roll"]
    assert plan["reductions"] == []
    assert plan["mask"] == ["none", 0]
    left = _constant(plan, "cyclic-mask-left")
    right = _constant(plan, "cyclic-mask-right")
    assert left[1] == right[1] == "f32"
    assert left[2] == right[2]
    assert len(left[2]) == len(right[2]) == 2
    assert left[2][1] == right[2][1] == 75
    assert left[4] == right[4] == left[2][0] * 75 * 4
    roles = [item[0] for item in plan["rotations"]]
    assert roles == ["blocking-alignment", "cyclic-left", "cyclic-right"]
    slice_roles = [item[0] for item in plan["slices"]]
    assert slice_roles[-2:] == ["cyclic-mask-left", "cyclic-mask-right"]

    stats = smoke["stats"]
    assert stats["ist"] == 1
    assert stats["ild"] >= 2
    assert stats["do_loop"] >= 3
    assert stats["VECTOR.slice"] >= 2
    assert stats["VECTOR.roll"] >= 3
    assert stats["VECTOR.mul"] >= 2
    assert smoke["rotation_attributes"] == [
        "0", "(-75,-50,-25)", "(0,25,50)"
    ]
    assert smoke["dump_has_rotation_table"]
    assert smoke["dump_has_mask"]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
