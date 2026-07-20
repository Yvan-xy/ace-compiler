"""Vector-kernel authoring substrate: lexical domains and exact Core types."""

from __future__ import annotations

import re

import pytest

from ace_bindings import air_builder
from ace_edsl.base_dsl.ast_helpers import dynamic_expr
from ace_edsl.edsl import AceEDSL, nn_kernel, range_dynamic, vector_kernel
from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core.types import Int, Tensor, VectorTensor


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


def _assert_ranked_vector_result(result, shape):
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
