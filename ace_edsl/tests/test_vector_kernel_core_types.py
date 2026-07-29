"""Vector-kernel authoring substrate: lexical domains and exact Core types."""

from __future__ import annotations

import ast
import re

import pytest

from ace_bindings import air_builder
from ace_edsl.base_dsl.ast_preprocessor import DSLPreprocessor
from ace_edsl.base_dsl.ast_helpers import dynamic_expr
from ace_edsl.edsl import AceEDSL, nn_kernel, range_dynamic, vector_kernel
from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core import vector_ops
from ace_edsl.edsl.core.types import Int, Tensor, VectorTensor
from ace_edsl.edsl.core.vector_value import VectorValue


@vector_kernel
def _vector_view_add(
    lhs: VectorTensor[float, 4], rhs: VectorTensor[float, 4]
) -> VectorTensor[float, 4]:
    return lhs + rhs


@nn_kernel
def _nn_with_nested_vector_view(
    lhs: Tensor[float, 4], rhs: Tensor[float, 4]
) -> Tensor[float, 4]:
    vector_sum = _vector_view_add(lhs=lhs, rhs=rhs)
    nn_sum = lhs + rhs
    return vector_sum + vector_sum


@vector_kernel
def _vector_nested_core_loops(
    value: VectorTensor[float, 4], kernel_hw: Int[32]
) -> VectorTensor[float, 4]:
    accumulator = value.zero_like()
    for cin in range_dynamic(0, 2):
        for khw in range_dynamic(0, kernel_hw):
            linear_index = cin * kernel_hw + khw
            shift = 1 << khw
            accumulator = accumulator + value
    return accumulator


@vector_kernel
def _vector_conditional_merge(
    lhs: VectorTensor[float, 4],
    rhs: VectorTensor[float, 4],
    flag: Int[32],
) -> VectorTensor[float, 4]:
    result = lhs
    if dynamic_expr(flag < 1):
        result = lhs + rhs
    else:
        result = rhs + lhs
    return result + lhs


@vector_kernel
def _heterogeneous_vector_signature(
    narrow: VectorTensor[float, 4],
    wide: VectorTensor[float, 8],
    count: Int[32],
) -> VectorTensor[float, 8]:
    return wide + wide


@vector_kernel
def _genuine_vector_primitives(
    narrow: VectorTensor[float, 4],
    wide: VectorTensor[float, 8],
    rows: VectorTensor[float, 3, 8],
    shift: Int[32],
    start: Int[32],
) -> VectorTensor[float, 8]:
    widened = narrow + wide
    rolled = wide.roll(shift, candidates=[-3, 0, 5])
    sliced = rows.slice(start, 8)
    product = rolled * sliced
    return (widened + product).with_slot(8)


@vector_kernel
def _vector_loop_with_scalar_yield(
    value: VectorTensor[float, 4],
) -> VectorTensor[float, 4]:
    accumulator = value
    for iv in range_dynamic(0, 1):
        accumulator = 1
    return accumulator


@vector_kernel
def _vector_if_with_nn_yield(
    value: VectorTensor[float, 4], flag: Int[32]
) -> VectorTensor[float, 4]:
    result = value
    if dynamic_expr(flag < 1):
        result = value.view("nn::core")
    else:
        result = value
    return result


@vector_kernel
def _raising_vector_kernel():
    raise ValueError("nested vector failure")


def _fresh_dsl():
    AceEDSL._get_dsl.cache_clear()
    return AceEDSL._get_dsl()


def test_vector_methods_are_receiver_functional_only_when_registered():
    method_loop = ast.parse(
        "for iv in range_dynamic(0, 4):\n"
        "    output = value.roll(iv, candidates=[0])\n"
    ).body[0]
    mutating_loop = ast.parse(
        "for iv in range_dynamic(0, 4):\n"
        "    value.append(iv)\n"
    ).body[0]
    reassignment_loop = ast.parse(
        "for iv in range_dynamic(0, 4):\n"
        "    value = value + value.roll(iv, candidates=[0])\n"
    ).body[0]

    default = DSLPreprocessor()
    used, carried, _ = default.analyze_region_variables(
        method_loop, {"value"}
    )
    assert used == []
    assert carried == ["value"]

    functional = DSLPreprocessor()
    functional.register_non_mutating_receiver_methods(
        VectorValue.NON_MUTATING_RECEIVER_METHODS
    )
    used, carried, _ = functional.analyze_region_variables(
        method_loop, {"value"}
    )
    assert used == ["value"]
    assert carried == []
    _, carried, _ = functional.analyze_region_variables(
        mutating_loop, {"value"}
    )
    assert carried == ["value"]
    _, carried, _ = functional.analyze_region_variables(
        reassignment_loop, {"value"}
    )
    assert carried == ["value"]

    functional.processed_functions.add(object())
    with pytest.raises(RuntimeError, match="cannot change"):
        functional.register_non_mutating_receiver_methods({"another_method"})


def _assert_ranked_vector_result(result, shape):
    assert isinstance(result, VectorValue)
    assert isinstance(result, AIRValue)
    assert result.domain == "nn::vector"
    assert tuple(result.shape) == tuple(shape)
    assert result.air_type is not None
    assert result.air_type.is_array()
    assert result.air_type.shape() == list(shape)


def test_nested_vector_kernel_views_nn_operands_and_restores_outer_domain():
    dsl = _fresh_dsl()

    result = _nn_with_nested_vector_view(
        Tensor[float, 4], Tensor[float, 4]
    )

    _assert_ranked_vector_result(result, (4,))
    assert dsl.current_domain == "nn::core"
    assert dsl._domain_stack == []
    assert dsl.current_air_module.verify_ir()

    dump = dsl.current_air_module.dump().upper()
    assert dump.count("VECTOR.ADD") == 2
    assert dump.count("NN.ADD") == 1


def test_nested_vector_domain_restores_after_exception():
    dsl = _fresh_dsl()
    dsl.current_domain = "nn::core"
    dsl._in_air_context = True
    try:
        with pytest.raises(ValueError, match="nested vector failure"):
            _raising_vector_kernel()
        assert dsl.current_domain == "nn::core"
        assert dsl._domain_stack == []
    finally:
        dsl._in_air_context = False


def test_annotated_shape_rejects_conflicting_runtime_descriptor():
    _fresh_dsl()

    with pytest.raises(TypeError, match="does not match annotated shape"):
        _vector_view_add(
            VectorTensor[float, 8], VectorTensor[float, 4]
        )


def test_vector_nested_loops_keep_ranked_values_and_use_core_s32_arithmetic():
    dsl = _fresh_dsl()

    result = _vector_nested_core_loops(VectorTensor[float, 4], 3)

    _assert_ranked_vector_result(result, (4,))
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump().upper()
    assert dump.count("DO_LOOP") == 2
    assert re.search(r"\n\s+MUL RTYPE.*INT32", dump)
    assert re.search(r"\n\s+ADD RTYPE.*INT32", dump)
    assert re.search(r"\n\s+SHL RTYPE.*INT32", dump)
    assert re.search(r"\n\s+LT RTYPE.*INT32", dump)
    assert re.search(r"\n\s+ZERO RTYPE.*ARRAY", dump)
    assert "VECTOR.ADD" in dump
    assert "NN.ADD" not in dump
    assert "NN.MUL" not in dump


def test_vector_conditional_merge_preserves_domain_shape_and_type():
    dsl = _fresh_dsl()

    result = _vector_conditional_merge(
        VectorTensor[float, 4], VectorTensor[float, 4], 1
    )

    _assert_ranked_vector_result(result, (4,))
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump().upper()
    assert re.search(r"\n\s+IF ID", dump)
    assert re.search(r"\n\s+LT RTYPE.*INT32", dump)
    assert dump.count("VECTOR.ADD") == 3


def test_control_flow_rejects_scalar_or_wrong_domain_vector_yields():
    _fresh_dsl()
    with pytest.raises(RuntimeError, match="loop-carried result must be an AIRValue"):
        _vector_loop_with_scalar_yield(VectorTensor[float, 4])

    _fresh_dsl()
    with pytest.raises(RuntimeError, match="conditional result domain"):
        _vector_if_with_nn_yield(VectorTensor[float, 4], 1)


def test_explicit_signature_keeps_heterogeneous_ranked_and_scalar_formals():
    dsl = _fresh_dsl()

    result = _heterogeneous_vector_signature(
        VectorTensor[float, 4], VectorTensor[float, 8], 2
    )

    _assert_ranked_vector_result(result, (8,))
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump()
    assert re.search(r'FML\[.*\] "p0".*\(array,"array"\)', dump)
    assert re.search(r'FML\[.*\] "p1".*\(array,"array"\)', dump)
    assert re.search(r'FML\[.*\] "p2".*\(primitive,"int32_t"\)', dump)


def test_decorated_helper_emits_genuine_vector_primitives_and_terminal_slot():
    dsl = _fresh_dsl()

    result = _genuine_vector_primitives(
        VectorTensor[float, 4],
        VectorTensor[float, 8],
        VectorTensor[float, 3, 8],
        1,
        0,
    )

    _assert_ranked_vector_result(result, (8,))
    assert dsl.current_air_module.verify_ir()
    dump = dsl.current_air_module.dump()
    upper_dump = dump.upper()
    assert upper_dump.count("VECTOR.ADD") == 2
    assert upper_dump.count("VECTOR.MUL") == 1
    assert upper_dump.count("VECTOR.ROLL") == 1
    assert upper_dump.count("VECTOR.SLICE") == 1
    assert "NN.ADD" not in upper_dump
    assert "NN.MUL" not in upper_dump
    assert "nums=(-3,0,5)" in dump
    assert dump.count("ATTR[slot=8]") == 1
    assert re.search(
        r'ld "__ret_tmp_\d+".*ATTR\[slot=8\]', dump
    )


def test_vector_wrappers_reject_invalid_domains_types_owners_and_metadata():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    i64 = air_builder.Type.make_int(64)
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)

    function = glob.new_func_with_param_types(
        "m3_vector_wrapper_validation",
        vector4,
        [vector4, i32, i64],
    )
    container = function.container()
    vector_param = function.new_param("vector", vector4)
    index_param = function.new_param("index", i32)
    index64_param = function.new_param("index64", i64)
    vector = VectorValue(
        vector_param,
        container,
        shape=(4,),
        air_type=vector4,
    )
    index = AIRValue(
        index_param,
        container,
        domain="air::core",
        air_type=i32,
    )
    index64 = AIRValue(
        index64_param,
        container,
        domain="air::core",
        air_type=i64,
    )

    assert isinstance(vector.view("nn::vector"), VectorValue)
    assert isinstance(vector.zero_like(), VectorValue)
    assert isinstance(vector.with_slot(4), VectorValue)
    assert isinstance(vector.cast(vector4), VectorValue)
    with pytest.raises(TypeError, match="ranked Vector AIR type"):
        VectorValue(index_param, container, air_type=i32)
    with pytest.raises(NotImplementedError, match="Vector subtraction"):
        vector - vector
    with pytest.raises(NotImplementedError, match="Vector subtraction"):
        1 - vector
    with pytest.raises(NotImplementedError, match="Vector negation"):
        -vector
    with pytest.raises(NotImplementedError, match="Vector division"):
        vector / vector
    with pytest.raises(NotImplementedError, match="Vector division"):
        1 / vector
    with pytest.raises(NotImplementedError, match="Vector division"):
        vector // vector
    with pytest.raises(NotImplementedError, match="Vector division"):
        1 // vector
    with pytest.raises(TypeError, match="nn::vector"):
        vector + index

    with pytest.raises(TypeError, match="nn::vector"):
        vector_ops.vec_add(index, vector)
    with pytest.raises(TypeError, match="ranked Vector AIR type"):
        vector_ops.vec_mul(
            AIRValue(None, container, domain="nn::vector"), vector
        )
    with pytest.raises(TypeError, match="air::core scalar"):
        vector_ops.vec_roll(vector, vector, [0])
    with pytest.raises(TypeError, match="signed Core i32 AIR type"):
        vector_ops.vec_roll(vector, index64, [0])
    with pytest.raises(TypeError, match="sequence of integers"):
        vector_ops.vec_roll(vector, index, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        vector_ops.vec_roll(vector, index, [])
    with pytest.raises(TypeError, match="must be integers"):
        vector_ops.vec_roll(vector, index, [False])
    with pytest.raises(ValueError, match="signed i32 range"):
        vector_ops.vec_roll(vector, index, [1 << 31])
    with pytest.raises(TypeError, match="signed Core i32 AIR type"):
        vector_ops.vec_slice(vector, index64, 1)
    with pytest.raises(TypeError, match="slice_size must be an integer"):
        vector_ops.vec_slice(vector, index, True)
    with pytest.raises(ValueError, match="positive signed i32"):
        vector_ops.vec_slice(vector, index, 0)
    with pytest.raises(TypeError, match="nn::vector"):
        vector_ops.vec_set_slot(index, 1)
    with pytest.raises(TypeError, match="SLOT must be an integer"):
        vector_ops.vec_set_slot(vector, True)
    with pytest.raises(ValueError, match="positive u32"):
        vector_ops.vec_set_slot(vector, 0)
    with pytest.raises(ValueError, match="positive u32"):
        vector_ops.vec_set_slot(vector, 1 << 32)

    foreign_function = glob.new_func_with_param_types(
        "m3_vector_wrapper_foreign_owner", vector4, [vector4, i32]
    )
    foreign_param = foreign_function.new_param("vector", vector4)
    foreign_index_param = foreign_function.new_param("index", i32)
    foreign_vector = VectorValue(
        foreign_param,
        foreign_function.container(),
        shape=(4,),
        air_type=vector4,
    )
    with pytest.raises(ValueError, match="cannot mix AIR containers"):
        vector_ops.vec_add(vector, foreign_vector)
    with pytest.raises(ValueError, match="cannot mix AIR containers"):
        vector * foreign_vector
    foreign_index = AIRValue(
        foreign_index_param,
        foreign_function.container(),
        domain="air::core",
        air_type=i32,
    )
    with pytest.raises(ValueError, match="cannot mix AIR containers"):
        vector_ops.vec_roll(vector, foreign_index, [0])


def test_native_vector_binding_result_rules_attributes_and_validation():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    i64 = air_builder.Type.make_int(64)
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)
    vector8 = air_builder.Type.make_array([8], f32)
    matrix3x8 = air_builder.Type.make_array([3, 8], f32)
    matrix2x2 = air_builder.Type.make_array([2, 2], f32)
    int_vector4 = air_builder.Type.make_array([4], i32)

    function = glob.new_func_with_param_types(
        "m3_native_vector_primitives",
        vector8,
        [vector4, vector8, matrix3x8, matrix2x2, i32, i64, int_vector4],
    )
    container = function.container()
    narrow = function.new_param("narrow", vector4)
    wide = function.new_param("wide", vector8)
    rows = function.new_param("rows", matrix3x8)
    square = function.new_param("square", matrix2x2)
    index = function.new_param("index", i32)
    index64 = function.new_param("index64", i64)
    ints = function.new_param("ints", int_vector4)

    assert container.new_vec_add(narrow, wide).rtype() == vector8
    assert container.new_vec_add(wide, narrow).rtype() == vector8
    assert container.new_vec_add(square, narrow).rtype() == matrix2x2
    assert container.new_vec_add(narrow, square).rtype() == vector4
    assert container.new_vec_mul(narrow, wide).rtype() == vector4
    assert container.new_vec_mul(wide, narrow).rtype() == vector8

    widened = container.new_vec_add(narrow, wide)
    rolled = container.new_vec_roll(wide, index, [-8, -1, 0, 3])
    sliced = container.new_vec_slice(rows, index, 8)
    assert rolled.opcode_name() == "nn::vector::ROLL"
    assert rolled.rtype() == vector8
    assert sliced.opcode_name() == "nn::vector::SLICE"
    assert sliced.rtype() == vector8

    rolled.set_s32_attr("signed_scalar", -2)
    rolled.set_s32_array_attr("signed_vector", [-3, 0, 5])
    rolled.set_u32_array_attr("unsigned_vector", [1, 8])
    product = container.new_vec_mul(widened, sliced)
    result = container.new_vec_add(product, rolled)
    result.set_vector_slot(8)

    with pytest.raises(RuntimeError, match="ranked array operands"):
        container.new_vec_add(narrow, index)
    with pytest.raises(RuntimeError, match="compatible vector element types"):
        container.new_vec_mul(narrow, ints)
    with pytest.raises(RuntimeError, match="Core signed i32 scalar"):
        container.new_vec_roll(wide, wide, [0])
    with pytest.raises(RuntimeError, match="Core signed i32 scalar"):
        container.new_vec_roll(wide, index64, [0])
    with pytest.raises(RuntimeError, match="non-empty rotation candidates"):
        container.new_vec_roll(wide, index, [])
    with pytest.raises(RuntimeError, match="out of range for signed i32"):
        container.new_vec_roll(wide, index, [1 << 31])
    with pytest.raises(RuntimeError, match="rank-2 source"):
        container.new_vec_slice(wide, index, 8)
    with pytest.raises(RuntimeError, match="Core signed i32 scalar"):
        container.new_vec_slice(rows, index64, 8)
    with pytest.raises(RuntimeError, match="positive signed i32"):
        container.new_vec_slice(rows, index, 0)
    with pytest.raises(RuntimeError, match="trailing dimension"):
        container.new_vec_slice(rows, index, 4)

    with pytest.raises(RuntimeError, match="requires at least one value"):
        rolled.set_s32_array_attr("empty", [])
    with pytest.raises(RuntimeError, match="signed i32"):
        rolled.set_s32_attr("overflow", 1 << 31)
    with pytest.raises(RuntimeError, match="unsigned i32"):
        rolled.set_u32_attr("negative", -1)
    with pytest.raises(RuntimeError, match="ranked Vector value"):
        index.set_vector_slot(1)
    with pytest.raises(RuntimeError, match="positive slot count"):
        result.set_vector_slot(0)
    with pytest.raises(RuntimeError, match="unsigned i32"):
        result.set_vector_slot(1 << 32)

    empty_container = air_builder.Container()
    with pytest.raises(RuntimeError, match="real AIR operands"):
        empty_container.new_vec_add(narrow, wide)

    plain_core_add = container.new_core_add(index, index)
    with pytest.raises(RuntimeError, match="opcode that supports AIR attributes"):
        plain_core_add.set_u32_attr("invalid", 1)
    container.new_retv(result)

    foreign_function = glob.new_func_with_param_types(
        "m3_foreign_vector_owner", vector8, [vector8, i32]
    )
    foreign_wide = foreign_function.new_param("wide", vector8)
    foreign_index = foreign_function.new_param("index", i32)
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_vec_add(wide, foreign_wide)
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_vec_roll(wide, foreign_index, [0])
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_vec_slice(rows, foreign_index, 8)
    foreign_function.container().new_retv(foreign_wide)

    assert glob.verify_ir()
    dump = glob.dump()
    assert "nums=(-8,-1,0,3)" in dump
    assert "signed_scalar=-2" in dump
    assert "signed_vector=(-3,0,5)" in dump
    assert "unsigned_vector=(1,8)" in dump
    assert "slot=8" in dump


def test_structural_types_typed_core_ops_locals_and_zero():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    i64 = air_builder.Type.make_int(64)
    f32 = air_builder.Type.make_float(32)
    vector4 = air_builder.Type.make_array([4], f32)
    same_vector4 = air_builder.Type.make_array([4], f32)
    vector8 = air_builder.Type.make_array([8], f32)

    with pytest.raises(RuntimeError, match="at least one dimension"):
        air_builder.Type.make_array([], f32)
    with pytest.raises(RuntimeError, match="dimensions must be positive"):
        air_builder.Type.make_array([0], f32)
    assert air_builder.Type.make_int(16).bit_width() == 16
    with pytest.raises(RuntimeError, match="integer type width"):
        air_builder.Type.make_int(24)
    with pytest.raises(RuntimeError, match="floating type width"):
        air_builder.Type.make_float(16)

    assert vector4 == same_vector4
    assert vector4.structurally_equal(same_vector4)
    assert vector4 != vector8
    assert i32 != i64

    function = glob.new_func_with_param_types(
        "typed_core_substrate", vector8, [vector4, vector8, i32]
    )
    container = function.container()
    vector_param = function.new_param("vector", vector4)
    wide_param = function.new_param("wide", vector8)
    index_param = function.new_param("index", i32)
    assert vector_param.rtype() == vector4
    assert wide_param.rtype() == vector8
    assert index_param.rtype() == i32

    with pytest.raises(RuntimeError, match="requires scalar operands"):
        container.new_core_add(vector_param, vector_param)
    with pytest.raises(RuntimeError, match="requires scalar operands"):
        container.new_core_mul(vector_param, vector_param)

    container.new_local("accumulator", vector4)
    typed_zero = container.new_zero(vector4)
    assert typed_zero.opcode_name() == "air::core::ZERO"
    assert typed_zero.rtype() == vector4
    container.new_stid("accumulator", typed_zero)

    old_flat_mode = AIRValue.FLAT_IR_MODE
    AIRValue.FLAT_IR_MODE = False
    try:
        index = AIRValue(
            index_param,
            container,
            domain="air::core",
            air_type=i32,
        )
        linear_index = index * 9 + index
        shift = 1 << index
        predicate = index < 3
        assert linear_index.value.opcode_name() == "air::core::ADD"
        assert shift.value.opcode_name() == "air::core::SHL"
        assert predicate.value.opcode_name() == "air::core::LT"
        assert linear_index.air_type == i32
        assert shift.air_type == i32
        assert predicate.air_type == i32
        assert index.cast(i32).air_type == i32
        with pytest.raises(RuntimeError, match="no width-conversion opcode"):
            index.cast(i64)
        with pytest.raises(RuntimeError, match="out of range for signed i32"):
            container.new_intconst_typed(1 << 31, i32)
    finally:
        AIRValue.FLAT_IR_MODE = old_flat_mode

    with pytest.raises(RuntimeError, match="incompatible with local"):
        container.new_stid("accumulator", container.new_zero(vector8))

    other_function = glob.new_func_with_param_types(
        "foreign_bound_owner", i32, [i32]
    )
    foreign_bound = other_function.new_param("bound", i32)
    with pytest.raises(RuntimeError, match="different AIR container"):
        container.new_loop_begin_range_dynamic(0, foreign_bound, 32)
    with pytest.raises(RuntimeError, match="out of range"):
        container.new_loop_begin_range(0, 1 << 31, 32)

    return_stmt = container.new_retv(wide_param)
    with pytest.raises(RuntimeError, match="expression operands with result types"):
        container.new_core_add(return_stmt, return_stmt)
    with pytest.raises(RuntimeError, match="expression value with a result type"):
        container.new_stid("invalid_statement_store", return_stmt)
    with pytest.raises(RuntimeError, match="real container and value"):
        container.new_stid("invalid_null_store", None)
    with pytest.raises(RuntimeError, match="expression value with a result type"):
        container.new_checked_cast(return_stmt, i32)
    with pytest.raises(RuntimeError, match="expression with a result type"):
        container.new_loop_begin_range_dynamic(0, return_stmt, 32)
    with pytest.raises(RuntimeError, match="another AIR container"):
        other_function.container().new_retv(index_param)
    other_function.container().new_retv(foreign_bound)

    strict_function = glob.new_func_with_param_types(
        "strict_signature", vector4, [vector4]
    )
    with pytest.raises(RuntimeError, match="does not match the function signature"):
        strict_function.new_param("value", vector8)
    strict_value = strict_function.new_param("value", vector4)
    strict_function.container().new_retv(strict_value)
    assert glob.verify_ir()

    other_glob = air_builder.create_glob_scope()
    other_vector4 = air_builder.Type.make_array(
        [4], air_builder.Type.make_float(32)
    )
    receiver_vector4 = glob.new_array_type([4], "f32")
    assert vector4 == other_vector4
    assert not vector4.same_scope(other_vector4)
    assert receiver_vector4 == vector4
    assert receiver_vector4.same_scope(vector4)


def test_constant_array_indexing_emits_real_ild_with_core_scalar_result():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    i64 = air_builder.Type.make_int(64)
    f32 = air_builder.Type.make_float(32)

    function = glob.new_func_with_param_types(
        "constant_array_indexing", f32, [i32, i64]
    )
    container = function.container()
    index_param = function.new_param("index", i32)
    wide_index_param = function.new_param("wide_index", i64)
    rotations_node = container.new_array_const([-3.0, 0.0, 5.0])
    rotations_type = rotations_node.rtype()
    rotations = VectorValue(
        rotations_node,
        container,
        shape=(3,),
        air_type=rotations_type,
    )
    index = AIRValue(
        index_param,
        container,
        domain="air::core",
        air_type=i32,
    )

    selected = rotations[index]
    assert isinstance(selected, AIRValue)
    assert not isinstance(selected, VectorValue)
    assert selected.domain == "air::core"
    assert selected.shape is None
    assert selected.air_type == f32

    with pytest.raises(
        RuntimeError,
        match="constant-array index requires a Core signed i32 scalar",
    ):
        container.new_ild(rotations_node, wide_index_param)

    foreign_function = glob.new_func_with_param_types(
        "foreign_constant_array_index", i32, [i32]
    )
    foreign_index = foreign_function.new_param("index", i32)
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_ild(rotations_node, foreign_index)

    container.new_retv(selected.value)
    foreign_function.container().new_retv(foreign_index)
    assert glob.verify_ir()
    dump = glob.dump().upper()
    assert "LDC" in dump
    assert "LDCA" in dump
    assert "ARRAY" in dump
    assert "ILD" in dump


def test_planned_packed_vector_store_is_explicit_and_type_checked():
    glob = air_builder.create_glob_scope()
    i32 = air_builder.Type.make_int(32)
    f32 = air_builder.Type.make_float(32)
    vector8 = air_builder.Type.make_array([8], f32)
    vector16 = air_builder.Type.make_array([16], f32)
    int_vector8 = air_builder.Type.make_array([8], i32)

    function = glob.new_func_with_param_types(
        "packed_vector_retyping", vector8, [vector16, i32]
    )
    container = function.container()
    source = function.new_param("source", vector16)
    scalar = function.new_param("scalar", i32)

    packed = container.new_vector_packed_stid(
        "packed_input", vector8, source
    )
    assert packed.rtype() == vector8

    container.new_local("ordinary_store", vector8)
    with pytest.raises(RuntimeError, match="incompatible with local"):
        container.new_stid("ordinary_store", source)
    with pytest.raises(RuntimeError, match="not a valid packed widening"):
        container.new_vector_widening_stid(
            "invalid_widening", vector8, source
        )
    with pytest.raises(RuntimeError, match="matching ranked element types"):
        container.new_vector_packed_stid(
            "integer_target", int_vector8, source
        )
    with pytest.raises(RuntimeError, match="ranked array operands"):
        container.new_vector_packed_stid("scalar_source", vector8, scalar)
    with pytest.raises(RuntimeError, match="local type mismatch"):
        container.new_vector_packed_stid(
            "packed_input", vector16, source
        )

    foreign_function = glob.new_func_with_param_types(
        "foreign_packed_vector_source", vector16, [vector16]
    )
    foreign_source = foreign_function.new_param("source", vector16)
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_vector_packed_stid(
            "foreign_source", vector8, foreign_source
        )

    container.new_retv(packed)
    foreign_function.container().new_retv(foreign_source)
    assert glob.verify_ir()

    foreign_glob = air_builder.create_glob_scope()
    foreign_f32 = air_builder.Type.make_float(32)
    foreign_vector8 = air_builder.Type.make_array([8], foreign_f32)
    with pytest.raises(RuntimeError, match="different GLOB_SCOPE"):
        container.new_vector_packed_stid(
            "foreign_type", foreign_vector8, source
        )
