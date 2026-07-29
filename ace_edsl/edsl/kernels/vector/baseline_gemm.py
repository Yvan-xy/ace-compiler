"""Phase-A baseline Gemm recipe backed by a real ``@vector_kernel``."""

from ace_edsl.edsl.core.air_value import AIRValue
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.domain_kernels import vector_kernel
from ace_edsl.edsl.vector_kernel_lowering import (
    PreparedBaselineGemmPlan,
    vector_kernel_recipe,
)
from ace_edsl.base_dsl.ast_helpers import range_dynamic


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


def _validate_prepared(prepared):
    if prepared.kind != "baseline-gemm":
        raise ValueError("baseline Gemm recipe received another plan kind")
    if prepared.slot.policy != "absent-native-baseline":
        raise ValueError("baseline Gemm recipe cannot introduce SLOT")
    if prepared.input_duplications not in (1, 2):
        raise ValueError("baseline Gemm v1 duplication topology is invalid")
    if prepared.input_duplications == 2:
        duplicate_rotations = prepared.rotation("input-duplication")
        if len(duplicate_rotations.candidates) != 1:
            raise ValueError("baseline Gemm v1 duplication topology is invalid")
    if prepared.reductions:
        reduction = prepared.reduction("block-reduction")
        if reduction.kind != "power-of-two" or reduction.padding != 0:
            raise ValueError("unsupported baseline Gemm reduction topology")
    if prepared.mask.policy not in ("none", "clear-valid-prefix"):
        raise ValueError("unsupported baseline Gemm mask policy")


@vector_kernel
def baseline_gemm_vector_kernel(
    packed_input,
    prepared: PreparedBaselineGemmPlan,
    constants,
    result_type,
    i32_type,
):
    """Emit exactly the frozen baseline-Gemm metakernel decisions."""
    _validate_prepared(prepared)

    container = packed_input.container
    weight = constants["weight"]
    bias = constants["bias"]
    gemm_loop = prepared.loop("gemm")
    weight_slice = prepared.slice("weight")
    gemm_rotations = prepared.rotation("gemm")

    result_name = "__gemm_result"
    container.new_local(result_name, result_type)
    container.new_stid(result_name, container.new_zero(result_type))
    result = _vector_value(
        None, container, result_type, prepared.result_type.shape, result_name
    )

    input_dup = packed_input
    if prepared.input_duplications > 1:
        duplicate_rotations = prepared.rotation("input-duplication")
        duplicate_shift = duplicate_rotations.candidates[0]
        input_dup = packed_input + packed_input.roll(
            _core_i32(duplicate_shift, container, i32_type),
            candidates=duplicate_rotations.candidates,
        )
    input_dup_node = container.new_vector_widening_stid(
        "input_dup", result_type, input_dup.value
    )
    input_dup = _vector_value(
        input_dup_node, container, result_type, prepared.result_type.shape
    )

    for iv in range_dynamic(
        gemm_loop.lower, gemm_loop.upper, gemm_loop.step
    ):
        rolled_input = input_dup.roll(
            iv, candidates=gemm_rotations.candidates
        )
        sliced_weight = weight.slice(iv, weight_slice.width)
        result = result + rolled_input * sliced_weight

    if prepared.reductions:
        reduction = prepared.reduction("block-reduction")
        reduction_loop = prepared.loop("block-reduction")
        reduction_rotations = prepared.rotation("block-reduction")
        for iv in range_dynamic(
            reduction_loop.lower,
            reduction_loop.upper,
            reduction_loop.step,
        ):
            shift = (1 << iv) * reduction.block_width
            result = result + result.roll(
                shift, candidates=reduction_rotations.candidates
            )

    biased = result + bias
    container.new_stid(result_name, biased.value)
    if prepared.mask.policy != "none":
        masked = result * constants["mask"]
        container.new_stid(result_name, masked.value)
    return result


baseline_gemm_recipe = vector_kernel_recipe(baseline_gemm_vector_kernel)


def configure_baseline_gemm_dsl(
    pipeline, *, mask_fuse=False, max_slots=0
):
    """Configure one pipeline for the Phase-A C++-plan/Python-kernel path."""
    return pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl="dsl",
        plan_kind="baseline-gemm",
        fallback="error",
        mask_fuse=mask_fuse,
        max_slots=max_slots,
    ).register_vector_kernel_recipe("baseline-gemm", baseline_gemm_recipe)
