"""Complete fast Gemm destination recipe backed by ``@vector_kernel``."""

from ace_edsl.base_dsl.ast_helpers import range_dynamic
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.domain_kernels import vector_kernel
from ace_edsl.edsl.vector.kernels.fast_common import (
    _ranked_constant,
    blocking_rot,
    clear_valid_data,
    reduce_add_intra,
)
from ace_edsl.edsl.vector.lowering import (
    PreparedFastGemmPlan,
    vector_kernel_recipe,
)


def _local_vector(container, name, air_type):
    return VectorValue(
        None,
        container,
        shape=tuple(air_type.shape()),
        temp_name=name,
        air_type=air_type,
    )


def _expected_reduction(factor, block_width, padding):
    stride = block_width + padding
    if factor <= 1 or stride <= 0:
        raise ValueError("fast Gemm reduction topology is inconsistent")
    if factor & (factor - 1) == 0:
        candidates = tuple(
            (1 << index) * stride for index in range((factor - 1).bit_length())
        )
        return "power-of-two", candidates
    return "linear", tuple(index * stride for index in range(1, factor))


def _validate_prepared(prepared):
    if prepared.kind != "fast-gemm":
        raise ValueError("fast Gemm recipe received another plan kind")
    if len(prepared.runtime_vector_inputs) != 1:
        raise ValueError("fast Gemm recipe requires one Vector input")
    if prepared.runtime_scalar_inputs or prepared.scalar_preparations:
        raise ValueError("fast Gemm recipe does not support scalar inputs")
    preparation = prepared.runtime_preparation("input")
    if (
        preparation.role != "input"
        or preparation.source_operand != 0
        or preparation.kind != "blocking-rotations"
        or preparation.result_type != prepared.runtime_vector_inputs[0]
        or preparation.logical_input_size != prepared.kp
        or preparation.replications != prepared.input_replications
        or preparation.replications <= 0
        or preparation.blocking_width != prepared.block_size
        or preparation.rotation_candidates != tuple(range(prepared.block_size))
        or preparation.outer_block_depth != 0
    ):
        raise ValueError("fast Gemm input preparation is inconsistent")
    if prepared.slot.policy != "logical-output-elements":
        raise ValueError("fast Gemm recipe requires logical-output SLOT")
    if prepared.slot.value != prepared.n:
        raise ValueError("fast Gemm SLOT does not match the logical output")
    if prepared.mask.policy not in ("none", "clear-valid-prefix"):
        raise ValueError("unsupported fast Gemm mask policy")
    if prepared.mask.valid_length != (
        prepared.n if prepared.mask.policy != "none" else 0
    ):
        raise ValueError("fast Gemm mask length is inconsistent")

    if [loop.role for loop in prepared.loops] != ["grid", "block"]:
        raise ValueError("fast Gemm loop topology is inconsistent")
    grid_loop = prepared.loop("grid")
    block_loop = prepared.loop("block")
    if (
        prepared.grid_size <= 0
        or grid_loop.lower != 0
        or grid_loop.upper != prepared.grid_size
        or grid_loop.step != 1
        or grid_loop.nesting_depth != 0
        or block_loop.lower != 0
        or block_loop.upper != prepared.block_size
        or block_loop.step != 1
        or block_loop.nesting_depth != 1
    ):
        raise ValueError("fast Gemm loop topology is inconsistent")

    if [item.role for item in prepared.slices] != ["weight"]:
        raise ValueError("fast Gemm weight slice is inconsistent")
    weight_slice = prepared.slice("weight")
    if (
        prepared.block_size <= 0
        or weight_slice.index.iv_coefficients != (prepared.block_size, 1)
        or weight_slice.index.constant != 0
        or weight_slice.index.uses_sharding_offset
        or weight_slice.width != prepared.result_type.shape[-1]
    ):
        raise ValueError("fast Gemm weight slice is inconsistent")

    if prepared.kp > prepared.np and prepared.kp % prepared.np != 0:
        raise ValueError("fast Gemm reduction topology is inconsistent")
    expected_reductions = []
    if prepared.packed_partitions > 1:
        expected_reductions.append(
            (
                "packed-partitions",
                prepared.packed_partitions,
                prepared.kd,
                prepared.shift_buffer,
            )
        )
    if prepared.kp > prepared.np:
        expected_reductions.append(
            ("kp-over-np", prepared.kp // prepared.np, prepared.np, 0)
        )
    if [item.role for item in prepared.reductions] != [
        item[0] for item in expected_reductions
    ]:
        raise ValueError("fast Gemm reduction topology is inconsistent")

    expected_rotations = []
    if prepared.input_replications > 1:
        expected_rotations.append(
            (
                "input-duplication",
                tuple(
                    -prepared.kp * copy
                    for copy in range(1, prepared.input_replications)
                ),
            )
        )
    expected_rotations.append(("blocking-alignment", tuple(range(prepared.block_size))))
    expected_rotations.append(
        (
            "grid",
            tuple(index * prepared.block_size for index in range(prepared.grid_size)),
        )
    )
    for reduction, (role, factor, block_width, padding) in zip(
        prepared.reductions, expected_reductions
    ):
        kind, candidates = _expected_reduction(factor, block_width, padding)
        if (
            reduction.kind != kind
            or reduction.factor != factor
            or reduction.block_width != block_width
            or reduction.padding != padding
        ):
            raise ValueError("fast Gemm reduction topology is inconsistent")
        expected_rotations.append((role, candidates))
    if [
        (item.role, item.candidates) for item in prepared.rotations
    ] != expected_rotations:
        raise ValueError("fast Gemm rotation topology is inconsistent")


@vector_kernel
def fast_gemm_vector_kernel(
    packed_input,
    prepared: PreparedFastGemmPlan,
    constants,
    result_type,
    i32_type,
):
    """Emit the complete frozen fast-Gemm lowering and epilogue."""
    _validate_prepared(prepared)

    container = packed_input.container
    weight = _ranked_constant(prepared, constants, "weight")
    bias = _ranked_constant(prepared, constants, "bias")
    blocked = blocking_rot(
        packed_input,
        prepared,
        constants,
        i32_type,
        "__blocked_input",
        "__duplicated_input",
    )

    result_name = "__gemm_result"
    block_name = "__block_result"
    container.new_local(result_name, result_type)
    container.new_stid(result_name, container.new_zero(result_type))
    container.new_local(block_name, result_type)

    grid_loop = prepared.loop("grid")
    block_loop = prepared.loop("block")
    weight_slice = prepared.slice("weight")
    grid_rotation = prepared.rotation("grid")

    container.new_comment(f"IMRA Metakernel: gs={prepared.grid_size}")
    for grid_iv in range_dynamic(grid_loop.lower, grid_loop.upper, grid_loop.step):
        container.new_stid(block_name, container.new_zero(result_type))
        for block_iv in range_dynamic(
            block_loop.lower, block_loop.upper, block_loop.step
        ):
            slice_index = block_iv + grid_iv * prepared.block_size
            contribution = blocked.load(block_iv) * weight.slice(
                slice_index, weight_slice.width
            )
            block_result = _local_vector(container, block_name, result_type)
            container.new_stid(block_name, (block_result + contribution).value)

        block_result = _local_vector(container, block_name, result_type)
        shift = grid_iv * prepared.block_size
        rolled = block_result.roll(shift, candidates=grid_rotation.candidates)
        result = _local_vector(container, result_name, result_type)
        container.new_stid(result_name, (result + rolled).value)

    active_name = result_name
    result = _local_vector(container, active_name, result_type)
    container.new_comment(f"gemm result reduce->Ps={prepared.packed_partitions}")
    if prepared.packed_partitions > 1:
        active_name = "__packed_partition_result"
        reduction = prepared.reduction("packed-partitions")
        result = reduce_add_intra(
            result,
            reduction,
            prepared.rotation(reduction.role),
            i32_type,
            active_name,
        )

    ratio = prepared.kp // prepared.np
    container.new_comment(f"gemm result reduce->(kp/np)={ratio}")
    if prepared.kp > prepared.np:
        active_name = "__kp_over_np_result"
        reduction = prepared.reduction("kp-over-np")
        result = reduce_add_intra(
            result,
            reduction,
            prepared.rotation(reduction.role),
            i32_type,
            active_name,
        )

    container.new_comment("gemm add bias")
    container.new_stid(active_name, (result + bias).value)
    result = _local_vector(container, active_name, result_type)
    if prepared.mask.policy != "none":
        result = clear_valid_data(result, prepared, constants, active_name)
    return result.with_slot(prepared.slot.value)


fast_gemm_recipe = vector_kernel_recipe(fast_gemm_vector_kernel)


def configure_fast_gemm_dsl(pipeline, *, mask_fuse=False, max_slots=0):
    """Configure one pipeline for the C++-plan/Python fast-Gemm path."""
    return pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl="dsl",
        plan_kind="fast-gemm",
        fallback="error",
        mask_fuse=mask_fuse,
        max_slots=max_slots,
    ).register_vector_kernel_recipe("fast-gemm", fast_gemm_recipe)


__all__ = [
    "configure_fast_gemm_dsl",
    "fast_gemm_recipe",
    "fast_gemm_vector_kernel",
]
