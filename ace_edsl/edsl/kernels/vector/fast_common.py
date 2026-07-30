"""Reusable Vector AIR helpers shared by the planned fast kernels."""

from ace_edsl.base_dsl.ast_helpers import const_expr, range_dynamic
from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core.vector_array import VectorArray
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.domain_kernels import vector_kernel


_PRIMITIVE_TYPE_NAMES = {
    "s32": "i32",
    "s64": "i64",
    "u32": "u32",
    "u64": "u64",
    "f32": "f32",
    "f64": "f64",
}


def _core_i32(value, container, i32_type):
    return AIRValue(
        container.new_intconst_typed(value, i32_type),
        container,
        domain="air::core",
        air_type=i32_type,
    )


def _vector_value(node, container, air_type):
    return VectorValue(
        node,
        container,
        shape=tuple(air_type.shape()),
        air_type=air_type,
    )


def _local_vector(container, name, air_type):
    return _vector_value(container.new_ldid(name), container, air_type)


def _preg_vector(container, name, air_type):
    return _vector_value(container.new_ldp(name), container, air_type)


def _require_vector(value, operation):
    if not isinstance(value, VectorValue):
        raise TypeError(f"{operation} requires a VectorValue")
    if value.air_type is None or not value.air_type.is_array():
        raise TypeError(f"{operation} requires a ranked Vector type")


def _require_core_i32(value, operation):
    if not isinstance(value, AIRValue) or value.domain != "air::core":
        raise TypeError(f"{operation} requires a Core i32 value")
    value_type = value.air_type
    if (
        value_type is None
        or not value_type.is_integer()
        or not value_type.is_scalar()
        or value_type.bit_width() != 32
        or value_type.to_string() != "i32"
    ):
        raise TypeError(f"{operation} requires a Core i32 value")


def _ranked_constant(prepared, constants, role):
    descriptor = prepared.constant(role)
    if role not in constants:
        raise KeyError(f"missing prepared constant role: {role}")
    value = constants[role]
    _require_vector(value, role)
    if tuple(value.air_type.shape()) != descriptor.type.shape:
        raise TypeError(f"prepared constant {role!r} has the wrong shape")
    actual_element = value.air_type.element_type().to_string()
    expected_element = _PRIMITIVE_TYPE_NAMES.get(
        descriptor.type.element_type, descriptor.type.element_type
    )
    if actual_element != expected_element:
        raise TypeError(
            f"prepared constant {role!r} has the wrong element type"
        )
    return value


def _duplicate_packed_input(
    packed_input, prepared, preparation, i32_type, local_name
):
    duplicated = packed_input
    if preparation.replications > 1:
        duplication_candidates = tuple(
            -preparation.logical_input_size * copy
            for copy in range(1, preparation.replications)
        )
        matching_roles = tuple(
            rotation for rotation in prepared.rotations
            if rotation.role == "input-duplication"
        )
        if matching_roles and (
            len(matching_roles) != 1
            or matching_roles[0].candidates != duplication_candidates
        ):
            raise ValueError("input duplication rotation topology is invalid")
        for shift in duplication_candidates:
            duplicated = duplicated + packed_input.roll(
                _core_i32(shift, packed_input.container, i32_type),
                candidates=(shift,),
            )
    container = packed_input.container
    container.new_local(local_name, packed_input.air_type)
    container.new_stid(local_name, duplicated.value)
    return _local_vector(container, local_name, packed_input.air_type)


def _blocking_inputs(packed_input, prepared, constants):
    _require_vector(packed_input, "blocking_rot")
    if prepared.kind not in ("fast-gemm", "fast-conv"):
        raise ValueError("blocking_rot requires a prepared fast plan")
    preparation = prepared.runtime_preparation("input")
    if preparation.kind != "blocking-rotations":
        raise ValueError("blocking_rot requires blocking-rotations preparation")
    if preparation.blocking_width <= 0:
        raise ValueError("blocking_rot requires a positive blocking width")
    rotation = prepared.rotation("blocking-alignment")
    if tuple(preparation.rotation_candidates) != rotation.candidates:
        raise ValueError("blocking rotation roles disagree")
    if len(rotation.candidates) != preparation.blocking_width:
        raise ValueError("blocking rotation count does not match array extent")
    rotation_table = _ranked_constant(
        prepared, constants, "rotation-table"
    )
    if tuple(rotation_table.shape) != (preparation.blocking_width,):
        raise TypeError("blocking rotation table must be rank one")
    return preparation, rotation, rotation_table


def _validate_intra_reduction(value, reduction, rotation):
    _require_vector(value, "reduce_add_intra")
    if reduction.factor <= 1:
        raise ValueError(
            "intra-block reduction requires factor greater than one"
        )
    if reduction.kind not in ("power-of-two", "linear"):
        raise ValueError("unsupported intra-block reduction kind")
    candidates = tuple(rotation.candidates)
    stride = reduction.block_width + reduction.padding
    if stride <= 0 or not candidates:
        raise ValueError("intra-block reduction topology is invalid")
    return candidates, stride


def _validate_clear(value, prepared):
    _require_vector(value, "clear_valid_data")
    if prepared.mask.policy not in ("none", "clear-valid-prefix"):
        raise ValueError("valid-data clearing requires clear-valid-prefix")
    if prepared.mask.policy != "none":
        mask_descriptor = prepared.constant("mask")
        if mask_descriptor.type.shape != (prepared.mask.valid_length,):
            raise ValueError("valid-data mask descriptor is inconsistent")


def _validate_cyclic(value, grid_iv, prepared):
    _require_vector(value, "roll_cyclic")
    _require_core_i32(grid_iv, "roll_cyclic")
    if prepared.kind != "fast-conv" or not prepared.cyclic_roll:
        raise ValueError("roll_cyclic requires a cyclic fast Conv plan")
    if grid_iv.container is not value.container:
        raise ValueError("roll_cyclic cannot mix AIR containers")
    left_slice = prepared.slice("cyclic-mask-left")
    right_slice = prepared.slice("cyclic-mask-right")
    if (
        left_slice.width != prepared.output_size
        or right_slice.width != prepared.output_size
        or left_slice.index.iv_coefficients != (1,)
        or right_slice.index.iv_coefficients != (1,)
    ):
        raise ValueError("cyclic mask slice topology is invalid")
    return left_slice, right_slice


def _collective_plan(value, prepared):
    _require_vector(value, "collective_reduce")
    if prepared.kind != "fast-conv":
        raise ValueError("collective_reduce requires a fast Conv plan")
    reduction = prepared.reduction("collective")
    if reduction.kind not in (
        "collective-single-block",
        "collective-blocks",
    ):
        raise ValueError("unsupported collective reduction kind")
    if prepared.mask.policy not in ("none", "collective-reduction"):
        raise ValueError("collective reduction mask policy is invalid")
    return reduction, prepared.mask.policy == "collective-reduction"


@vector_kernel
def blocking_rot(
    packed_input,
    prepared,
    constants,
    i32_type,
    array_name,
    input_name,
):
    """Duplicate one packed input and populate its rotated vector array."""
    preparation, rotation, rotation_table = _blocking_inputs(
        packed_input, prepared, constants
    )

    container = packed_input.container
    container.new_comment(
        f"BlockingRot replicate={preparation.replications}"
    )
    duplicated = _duplicate_packed_input(
        packed_input, prepared, preparation, i32_type, input_name
    )
    container.new_comment(
        f"BlockingRot HRot=bs={preparation.blocking_width}"
    )
    blocked = VectorArray(
        container,
        array_name,
        preparation.blocking_width,
        packed_input.air_type,
    )
    for iv in range_dynamic(0, preparation.blocking_width, 1):
        shift = rotation_table[iv]
        blocked.store(
            iv,
            duplicated.roll(shift, candidates=rotation.candidates),
        )
    return blocked


@vector_kernel
def reduce_add_intra(value, reduction, rotation, i32_type, local_name):
    """Reduce planned blocks within one packed Vector value."""
    candidates, stride = _validate_intra_reduction(
        value, reduction, rotation
    )

    container = value.container
    container.new_local(local_name, value.air_type)
    container.new_stid(local_name, value.value)
    if reduction.kind == "power-of-two":
        for iv in range_dynamic(0, len(candidates), 1):
            shift = 1 << iv
            if const_expr(stride != 1):
                shift = shift * stride
            current = _local_vector(container, local_name, value.air_type)
            updated = current + current.roll(shift, candidates=candidates)
            container.new_stid(local_name, updated.value)
    else:
        for iv in range_dynamic(1, reduction.factor, 1):
            shift = iv * stride
            current = _local_vector(container, local_name, value.air_type)
            updated = current + value.roll(shift, candidates=candidates)
            container.new_stid(local_name, updated.value)
    return _local_vector(container, local_name, value.air_type)


@vector_kernel
def clear_valid_data(value, prepared, constants, local_name):
    """Apply the prepared valid-prefix mask to a Vector local."""
    _validate_clear(value, prepared)
    result = value
    if prepared.mask.policy != "none":
        mask = _ranked_constant(prepared, constants, "mask")
        container = value.container
        container.new_local(local_name, value.air_type)
        container.new_stid(local_name, (value * mask).value)
        result = _local_vector(container, local_name, value.air_type)
    return result


@vector_kernel
def roll_cyclic(value, grid_iv, prepared, constants, i32_type):
    """Emit the two-mask cyclic grid roll selected by a fast Conv plan."""
    left_slice, right_slice = _validate_cyclic(
        value, grid_iv, prepared
    )
    left_mask = _ranked_constant(
        prepared, constants, "cyclic-mask-left"
    ).slice(grid_iv, left_slice.width)
    right_mask = _ranked_constant(
        prepared, constants, "cyclic-mask-right"
    ).slice(grid_iv, right_slice.width)
    left_rotation = prepared.rotation("cyclic-left")
    right_rotation = prepared.rotation("cyclic-right")
    left_shift = (
        _core_i32(-prepared.output_size, value.container, i32_type)
        + grid_iv * prepared.position_block
    )
    right_shift = grid_iv * prepared.position_block
    left = (value * left_mask).roll(
        left_shift, candidates=left_rotation.candidates
    )
    right = (value * right_mask).roll(
        right_shift, candidates=right_rotation.candidates
    )
    return left + right


@vector_kernel
def collective_reduce(
    value, prepared, constants, i32_type, preg_name
):
    """Emit the prepared single-block or block-collective epilogue."""
    reduction, need_mask = _collective_plan(value, prepared)

    container = value.container
    container.new_preg(preg_name, value.air_type)
    if const_expr(prepared.output_size == prepared.num_slots):
        initial = value
        if need_mask:
            initial = value * _ranked_constant(
                prepared, constants, "collective-mask"
            )
        container.new_stp(preg_name, initial.value)
        return _preg_vector(container, preg_name, value.air_type)

    tail_rotation = prepared.rotation("collective-tail")
    tail_shift = _core_i32(
        -reduction.block_width, container, i32_type
    )
    if const_expr(reduction.kind == "collective-single-block"):
        reduced = value + value.roll(
            tail_shift, candidates=tail_rotation.candidates
        )
        if need_mask:
            reduced = reduced * _ranked_constant(
                prepared, constants, "collective-mask"
            )
        container.new_stp(preg_name, reduced.value)
        return _preg_vector(container, preg_name, value.air_type)

    data_mask = _ranked_constant(
        prepared, constants, "collective-mask"
    )
    gap_mask = _ranked_constant(
        prepared, constants, "collective-gap-mask"
    )
    initial = value * data_mask if need_mask else value
    container.new_stp(preg_name, initial.value)
    stride = reduction.block_width + reduction.padding
    if reduction.factor > 1:
        data_loop = prepared.loop("collective-data")
        data_rotation = prepared.rotation("collective-data")
        for iv in range_dynamic(
            data_loop.lower, data_loop.upper, data_loop.step
        ):
            shift = iv * stride
            contribution = value.roll(
                shift, candidates=data_rotation.candidates
            ) * data_mask
            updated = _preg_vector(
                container, preg_name, value.air_type
            ) + contribution
            container.new_stp(preg_name, updated.value)

        gap_loop = prepared.loop("collective-gap")
        gap_rotation = prepared.rotation("collective-gap")
        for iv in range_dynamic(
            gap_loop.lower, gap_loop.upper, gap_loop.step
        ):
            shift = iv * stride + reduction.padding
            contribution = value.roll(
                shift, candidates=gap_rotation.candidates
            ) * gap_mask
            updated = _preg_vector(
                container, preg_name, value.air_type
            ) + contribution
            container.new_stp(preg_name, updated.value)

    tail = value.roll(tail_shift, candidates=tail_rotation.candidates)
    if need_mask:
        tail = tail * gap_mask
    final = _preg_vector(container, preg_name, value.air_type) + tail
    container.new_stp(preg_name, final.value)
    return _preg_vector(container, preg_name, value.air_type)


__all__ = [
    "blocking_rot",
    "clear_valid_data",
    "collective_reduce",
    "reduce_add_intra",
    "roll_cyclic",
]
