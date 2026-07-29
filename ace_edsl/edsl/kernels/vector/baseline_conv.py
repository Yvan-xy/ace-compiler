"""Baseline Conv destination recipe backed by a real ``@vector_kernel``."""

from ace_edsl.base_dsl.ast_helpers import range_dynamic
from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.domain_kernels import vector_kernel
from ace_edsl.edsl.vector_kernel_lowering import (
    PreparedBaselineConvPlan,
    vector_kernel_recipe,
)


def _vector_value(node, container, air_type, shape, temp_name=None):
    return VectorValue(
        node,
        container,
        shape=tuple(shape),
        temp_name=temp_name,
        air_type=air_type,
    )


def _core_i32(value, container, i32_type):
    return AIRValue(
        container.new_intconst_typed(value, i32_type),
        container,
        domain="air::core",
        air_type=i32_type,
    )


def _require_baseline_conv(prepared):
    if prepared.kind != "baseline-conv":
        raise ValueError("baseline Conv recipe received another plan kind")


def _duplicate_packed_input(packed_input, prepared, i32_type):
    duplicated = packed_input
    if prepared.input_duplications == 1:
        return duplicated
    rotation = prepared.rotation("input-duplication")
    for shift in rotation.candidates:
        duplicated = duplicated + packed_input.roll(
            _core_i32(shift, packed_input.container, i32_type),
            candidates=(shift,),
        )
    return duplicated


@vector_kernel
def baseline_conv_vector_kernel(
    packed_input,
    prepared: PreparedBaselineConvPlan,
    constants,
    result_type,
    i32_type,
):
    """Emit exactly the frozen baseline-Conv metakernel decisions."""
    _require_baseline_conv(prepared)

    container = packed_input.container
    weight = constants["weight"]
    bias = constants["bias"]
    rotation_table = constants["rotation-table"]
    channel_loop = prepared.loop("channel-in")
    kernel_loop = prepared.loop("kernel-hw")
    weight_slice = prepared.slice("weight")
    kernel_rotations = prepared.rotation("kernel-alignment")
    channel_rotations = prepared.rotation("channel-step")

    result_name = "__conv_result"
    container.new_local(result_name, result_type)
    container.new_stid(result_name, container.new_zero(result_type))
    result = _vector_value(
        None, container, result_type, prepared.result_type.shape, result_name
    )

    duplicated = _duplicate_packed_input(
        packed_input, prepared, i32_type
    )
    input_name = "__conv_input"
    duplicated_node = container.new_vector_packed_stid(
        input_name, result_type, duplicated.value
    )
    duplicated = _vector_value(
        duplicated_node,
        container,
        result_type,
        prepared.result_type.shape,
    )

    for cin in range_dynamic(
        channel_loop.lower, channel_loop.upper, channel_loop.step
    ):
        for khw in range_dynamic(
            kernel_loop.lower, kernel_loop.upper, kernel_loop.step
        ):
            shift = rotation_table[khw]
            rolled_input = duplicated.roll(
                shift, candidates=kernel_rotations.candidates
            )
            slice_index = khw + cin * prepared.kernel_hw
            sliced_weight = weight.slice(
                slice_index, weight_slice.width
            )
            accumulated = result + rolled_input * sliced_weight
            container.new_stid(result_name, accumulated.value)

        channel_shift = _core_i32(
            channel_rotations.candidates[0], container, i32_type
        )
        stepped_input = duplicated.roll(
            channel_shift, candidates=channel_rotations.candidates
        )
        container.new_stid(input_name, stepped_input.value)

    container.new_stid(result_name, (result + bias).value)
    return result.with_slot(prepared.slot.value)


baseline_conv_recipe = vector_kernel_recipe(baseline_conv_vector_kernel)


def configure_baseline_conv_dsl(pipeline, *, max_slots=0):
    """Configure one pipeline for the C++-plan/Python baseline Conv path."""
    return pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl="dsl",
        plan_kind="baseline-conv",
        fallback="error",
        mask_fuse=False,
        max_slots=max_slots,
    ).register_vector_kernel_recipe("baseline-conv", baseline_conv_recipe)
