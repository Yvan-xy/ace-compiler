"""Standalone tests for the AIR-independent vector-kernel planner."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib

import numpy as np
import pytest

import ace_edsl.edsl.vector_kernel_planning as planning


_I64_MAX = (1 << 63) - 1


def _payload(role, values, dtype, shape):
    array = np.array(values, dtype=np.dtype(dtype), order="C", copy=True)
    array = array.reshape(shape)
    return planning.TypedPayload(
        role=role,
        values=array,
        content_hash=planning.canonical_payload_hash(array),
    )


def _attribute(name, values, dtype="<i4"):
    array = np.array(values, dtype=np.dtype(dtype), order="C", copy=True)
    return planning.AttributeRecord(name, array)


def _options(**changes):
    return replace(
        planning.OptionSnapshot(
            conv_parallel=False,
            mask_fuse=False,
            selective_strided_slice=False,
            sharding=False,
            is_last_operation=False,
        ),
        **changes,
    )


def _gemm_request(
    kind=planning.RequestedPlanKind.AUTO,
    *,
    n=2,
    k=4,
    slots=16,
    input_shape=None,
    options=None,
    attributes=(),
    weight_values=None,
    bias_values=None,
):
    if weight_values is None:
        weight_values = np.arange(1, n * k + 1, dtype="<f4")
    if bias_values is None:
        bias_values = (
            np.array([0.25, -0.5], dtype="<f4")
            if n == 2
            else np.arange(n, dtype="<f4") / np.float32(4.0)
        )
    input_type = planning.RankedTypePlan("f32", input_shape or (k,))
    weight_type = planning.RankedTypePlan("f32", (n, k))
    bias_type = planning.RankedTypePlan("f32", (n,))
    return planning.PlanningRequest(
        operation=planning.Operation.GEMM,
        attributes=tuple(attributes),
        operand_types=(input_type, weight_type, bias_type),
        declared_result_type=planning.RankedTypePlan("f32", (n,)),
        options=options or _options(),
        target=planning.TargetSnapshot(slots, 1, 65536),
        source_constants=(
            _payload("weight", weight_values, "<f4", (n, k)),
            _payload("bias", bias_values, "<f4", (n,)),
        ),
        requested_plan_kind=kind,
    )


def _conv_request(
    channel_in,
    channel_out,
    height,
    kernel,
    slots,
    kind=planning.RequestedPlanKind.AUTO,
    *,
    sharded=False,
    group=1,
    options=None,
    attribute_dtype="<i4",
):
    kernel_channel_in = channel_in // group if group > 1 else channel_in
    attributes = [
        _attribute("group", [group], attribute_dtype),
        _attribute("strides", [1, 1], attribute_dtype),
        _attribute("pads", [kernel // 2] * 4, attribute_dtype),
    ]
    shard_count = 1
    source_weight_shape = (
        channel_out,
        kernel_channel_in,
        kernel,
        kernel,
    )
    runtime_scalar_types = ()
    if sharded:
        shard_count = 2
        source_weight_shape = (2,) + source_weight_shape
        attributes.extend(
            (
                _attribute("orig_strides", [1, 1], attribute_dtype),
                _attribute("weight-sharded", [1], attribute_dtype),
                _attribute("sharding", [2, 1, 2, 0], attribute_dtype),
            )
        )
        runtime_scalar_types = ("s32",)
    weight_count = shard_count * channel_out * kernel_channel_in * kernel * kernel
    indices = np.arange(weight_count, dtype="<i4")
    weight = ((indices % 17) - 8).astype("<f4") / np.float32(8.0)
    bias = np.arange(channel_out, dtype="<f4") / np.float32(4.0)
    input_type = planning.RankedTypePlan("f32", (1, channel_in, height, height))
    weight_type = planning.RankedTypePlan(
        "f32", (channel_out, kernel_channel_in, kernel, kernel)
    )
    bias_type = planning.RankedTypePlan("f32", (channel_out,))
    return planning.PlanningRequest(
        operation=planning.Operation.CONV,
        attributes=tuple(attributes),
        operand_types=(input_type, weight_type, bias_type),
        declared_result_type=planning.RankedTypePlan(
            "f32", (1, channel_out, height, height)
        ),
        options=options or _options(sharding=sharded),
        target=planning.TargetSnapshot(slots, 1, 65536),
        source_constants=(
            _payload("weight", weight, "<f4", source_weight_shape),
            _payload("bias", bias, "<f4", (channel_out,)),
        ),
        requested_plan_kind=kind,
        runtime_scalar_types=runtime_scalar_types,
    )


def _with_attribute(request, name, values, dtype="<i4", *, options=None):
    attributes = tuple(item for item in request.attributes if item.name != name) + (
        _attribute(name, values, dtype),
    )
    return replace(
        request,
        attributes=attributes,
        options=request.options if options is None else options,
    )


def _payload_by_role(result, role):
    return next(item for item in result.constants if item.role == role)


def _roles(result):
    return tuple(item.role for item in result.constants)


def _normalized(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes(order="C").hex())
    if is_dataclass(value):
        return (
            type(value).__name__,
            tuple(
                (field.name, _normalized(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_normalized(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _normalized(item)) for key, item in value.items()))
    return value


def _assert_payload_integrity(result):
    assert tuple(result.plan.common.constants) == tuple(
        planning.ConstantPlan(item.role, item.type, item.content_hash)
        for item in result.constants
    )
    for item in result.constants:
        assert item.values.flags.c_contiguous
        assert not item.values.flags.writeable
        assert item.values.dtype.byteorder != ">"
        assert planning.canonical_payload_hash(item.values) == item.content_hash


def test_canonical_payload_hash_is_little_endian_bit_exact(monkeypatch):
    bits = np.array([0x3F800000, 0x80000000, 0x7FC01234], dtype="<u4")
    values = bits.view("<f4")
    header = (
        "vector-kernel-constant:v1\nelement=f32\nrank=1\nshape=3\npayload:\n"
    ).encode("ascii")
    expected = "sha256:" + hashlib.sha256(header + bits.tobytes()).hexdigest()

    assert planning.canonical_payload_hash(values) == expected
    with pytest.raises(planning.PlanningError, match="C-contiguous"):
        planning.canonical_payload_hash(np.arange(8, dtype="<f4")[::2])
    with pytest.raises(planning.PlanningError, match="little-endian"):
        planning.canonical_payload_hash(values.astype(">f4"))
    native_values = values.astype("=f4")
    assert native_values.dtype.byteorder == "="
    monkeypatch.setattr(planning.np, "little_endian", False)
    with pytest.raises(planning.PlanningError, match="little-endian"):
        planning.canonical_payload_hash(native_values)


def test_baseline_gemm_matches_canonical_golden():
    result = planning.plan_vector_kernel(
        _gemm_request(planning.RequestedPlanKind.BASELINE_GEMM)
    )
    plan = result.plan

    assert isinstance(plan, planning.BaselineGemmPlan)
    assert (plan.height, plan.width, plan.input_duplications) == (2, 4, 2)
    assert plan.common.runtime_vector_inputs == (planning.RankedTypePlan("f32", (4,)),)
    assert plan.common.runtime_scalar_inputs == ()
    assert plan.common.result_type == planning.RankedTypePlan("f32", (8,))
    assert plan.common.loops == (
        planning.LoopPlan("gemm", 0, 2, 1, 0),
        planning.LoopPlan("block-reduction", 0, 1, 1, 0),
    )
    assert plan.common.slices == (
        planning.SlicePlan("weight", planning.AffineIndexPlan((1,), 0, False), 4),
    )
    assert plan.common.rotations == (
        planning.RotationPlan("input-duplication", (-4,)),
        planning.RotationPlan("gemm", (0, 1)),
        planning.RotationPlan("block-reduction", (2,)),
    )
    assert plan.common.reductions == (
        planning.ReductionPlan(
            "block-reduction",
            planning.ReductionKind.POWER_OF_TWO,
            2,
            2,
            0,
        ),
    )
    assert plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.CLEAR_VALID_PREFIX, 2
    )
    assert plan.common.slot == planning.SlotPlan(
        planning.SlotPolicy.ABSENT_NATIVE_BASELINE, 0
    )
    assert _roles(result) == ("weight", "bias", "mask")
    np.testing.assert_array_equal(
        _payload_by_role(result, "weight").values,
        np.array(
            [[1.0, 6.0, 3.0, 8.0], [2.0, 7.0, 4.0, 5.0]],
            dtype="<f4",
        ),
    )
    assert _payload_by_role(result, "weight").content_hash == (
        "sha256:60dd6aa351452dd12455c4c9fb985dd1d1206e90433d6edd4af06dfebd045fa9"
    )
    assert _payload_by_role(result, "bias").content_hash == (
        "sha256:a8212821f073ff2f119d17825f2887b52fb51f81c74afa3a5c5dd7944da29847"
    )
    assert _payload_by_role(result, "mask").content_hash == (
        "sha256:11153e26e7bd937cbca18baa3a639fffd7478b1dd575ab3ea0f8c5f9eecc5739"
    )
    assert result.runtime_preparations == (
        planning.RuntimePreparation(
            "input",
            0,
            planning.RuntimePreparationKind.PACKED_VECTOR,
            planning.RankedTypePlan("f32", (4,)),
            4,
            2,
            0,
            (-4,),
            0,
        ),
    )
    assert result.scalar_preparations == ()
    _assert_payload_integrity(result)


def test_baseline_gemm_non_power_factor_preserves_reference_reduction():
    result = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.BASELINE_GEMM,
            n=2,
            k=6,
            slots=16,
        )
    )
    plan = result.plan

    assert isinstance(plan, planning.BaselineGemmPlan)
    assert (plan.height, plan.width, plan.input_duplications) == (2, 6, 2)
    assert plan.common.loops[-1] == planning.LoopPlan("block-reduction", 0, 2, 1, 0)
    assert plan.common.rotations[-1] == planning.RotationPlan("block-reduction", (2, 4))
    assert plan.common.reductions == (
        planning.ReductionPlan(
            "block-reduction", planning.ReductionKind.POWER_OF_TWO, 3, 2, 0
        ),
    )
    _assert_payload_integrity(result)


def test_fast_gemm_matches_canonical_golden():
    result = planning.plan_vector_kernel(
        _gemm_request(planning.RequestedPlanKind.FAST_GEMM)
    )
    plan = result.plan

    assert isinstance(plan, planning.FastGemmPlan)
    assert (
        plan.n,
        plan.k,
        plan.np,
        plan.kp,
        plan.nd,
        plan.kd,
        plan.block_size,
        plan.blocks_per_partition,
        plan.packed_partitions,
        plan.shift,
        plan.shift_buffer,
        plan.grid_size,
        plan.input_replications,
    ) == (2, 4, 2, 4, 2, 4, 1, 2, 1, 1, 2, 2, 2)
    assert plan.common.result_type == planning.RankedTypePlan("f32", (6,))
    assert plan.common.loops == (
        planning.LoopPlan("grid", 0, 2, 1, 0),
        planning.LoopPlan("block", 0, 1, 1, 1),
    )
    assert plan.common.slices == (
        planning.SlicePlan("weight", planning.AffineIndexPlan((1, 1), 0, False), 6),
    )
    assert plan.common.rotations == (
        planning.RotationPlan("input-duplication", (-4,)),
        planning.RotationPlan("blocking-alignment", (0,)),
        planning.RotationPlan("grid", (0, 1)),
        planning.RotationPlan("kp-over-np", (2,)),
    )
    assert plan.common.reductions == (
        planning.ReductionPlan(
            "kp-over-np", planning.ReductionKind.POWER_OF_TWO, 2, 2, 0
        ),
    )
    assert plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.CLEAR_VALID_PREFIX, 2
    )
    assert plan.common.slot == planning.SlotPlan(
        planning.SlotPolicy.LOGICAL_OUTPUT_ELEMENTS, 2
    )
    assert _roles(result) == ("weight", "bias", "rotation-table", "mask")
    np.testing.assert_array_equal(
        _payload_by_role(result, "weight").values,
        np.array(
            [
                [1.0, 6.0, 3.0, 8.0, 1.0, 6.0],
                [5.0, 2.0, 7.0, 4.0, 5.0, 2.0],
            ],
            dtype="<f4",
        ),
    )
    assert _payload_by_role(result, "weight").content_hash == (
        "sha256:3b4cb4b3dcd36db407db3350348dec1fc195d92f3ea1953df420eafd7d67bd42"
    )
    assert _payload_by_role(result, "rotation-table").content_hash == (
        "sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf"
    )
    assert result.runtime_preparations == (
        planning.RuntimePreparation(
            "input",
            0,
            planning.RuntimePreparationKind.BLOCKING_ROTATIONS,
            planning.RankedTypePlan("f32", (4,)),
            4,
            2,
            1,
            (0,),
            0,
        ),
    )
    _assert_payload_integrity(result)


def test_auto_and_forced_selection_and_operation_mismatch():
    automatic_gemm = planning.plan_vector_kernel(_gemm_request())
    forced_fast_gemm = planning.plan_vector_kernel(
        _gemm_request(planning.RequestedPlanKind.FAST_GEMM)
    )
    forced_baseline_gemm = planning.plan_vector_kernel(
        _gemm_request(planning.RequestedPlanKind.BASELINE_GEMM)
    )
    assert isinstance(automatic_gemm.plan, planning.FastGemmPlan)
    assert isinstance(forced_baseline_gemm.plan, planning.BaselineGemmPlan)
    assert _normalized(automatic_gemm) == _normalized(forced_fast_gemm)

    automatic_baseline_conv = planning.plan_vector_kernel(_conv_request(4, 2, 2, 1, 32))
    forced_baseline_conv = planning.plan_vector_kernel(
        _conv_request(4, 2, 2, 1, 32, planning.RequestedPlanKind.BASELINE_CONV)
    )
    automatic_fast_conv = planning.plan_vector_kernel(_conv_request(2, 4, 2, 3, 32))
    forced_fast_conv = planning.plan_vector_kernel(
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV)
    )
    assert isinstance(automatic_baseline_conv.plan, planning.BaselineConvPlan)
    assert isinstance(automatic_fast_conv.plan, planning.FastConvPlan)
    assert _normalized(automatic_baseline_conv) == _normalized(forced_baseline_conv)
    assert _normalized(automatic_fast_conv) == _normalized(forced_fast_conv)

    for request in (
        _gemm_request(planning.RequestedPlanKind.BASELINE_CONV),
        _gemm_request(planning.RequestedPlanKind.FAST_CONV),
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.BASELINE_GEMM),
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_GEMM),
    ):
        with pytest.raises(planning.PlanningError, match="plan kind cannot plan"):
            planning.plan_vector_kernel(request)


def test_fast_gemm_imra_cost_tie_prefers_more_packed_partitions():
    result = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.FAST_GEMM,
            n=8,
            k=8,
            slots=128,
        )
    )
    plan = result.plan

    assert isinstance(plan, planning.FastGemmPlan)
    assert (plan.block_size, plan.blocks_per_partition) == (2, 4)
    assert plan.packed_partitions == 2
    assert (plan.shift_buffer, plan.grid_size, plan.input_replications) == (
        4,
        2,
        3,
    )
    assert _payload_by_role(result, "weight").values.shape == (4, 24)
    packed_reduction = next(
        item for item in plan.common.reductions if item.role == "packed-partitions"
    )
    assert packed_reduction == planning.ReductionPlan(
        "packed-partitions",
        planning.ReductionKind.POWER_OF_TWO,
        2,
        8,
        4,
    )
    _assert_payload_integrity(result)


def test_gemm_mask_fusion_controls_owned_mask_payload():
    fused_baseline = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.BASELINE_GEMM,
            options=_options(mask_fuse=True),
        )
    )
    requested_mask = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.BASELINE_GEMM,
            options=_options(mask_fuse=True),
            attributes=(_attribute("mask", [7], "<u4"),),
        )
    )
    fused_last_fast = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.FAST_GEMM,
            options=_options(mask_fuse=True, is_last_operation=True),
        )
    )
    nonlast_fast = planning.plan_vector_kernel(
        _gemm_request(
            planning.RequestedPlanKind.FAST_GEMM,
            options=_options(mask_fuse=True, is_last_operation=False),
        )
    )

    assert "mask" not in _roles(fused_baseline)
    assert fused_baseline.plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.NONE, 0
    )
    assert "mask" in _roles(requested_mask)
    assert "mask" not in _roles(fused_last_fast)
    assert fused_last_fast.plan.common.mask.policy is planning.MaskPolicy.NONE
    assert "mask" in _roles(nonlast_fast)


def test_baseline_conv_matches_canonical_golden():
    result = planning.plan_vector_kernel(
        _conv_request(4, 2, 2, 1, 32, planning.RequestedPlanKind.BASELINE_CONV)
    )
    plan = result.plan

    assert isinstance(plan, planning.BaselineConvPlan)
    assert (
        plan.channel_in,
        plan.channel_out,
        plan.output_height,
        plan.output_width,
        plan.kernel_hw,
        plan.stride,
        plan.input_duplications,
    ) == (4, 2, 2, 2, 1, 1, 2)
    assert plan.common.loops == (
        planning.LoopPlan("channel-in", 0, 4, 1, 0),
        planning.LoopPlan("kernel-hw", 0, 1, 1, 1),
    )
    assert plan.common.slices == (
        planning.SlicePlan("weight", planning.AffineIndexPlan((1, 1), 0, False), 8),
    )
    assert plan.common.rotations == (
        planning.RotationPlan("input-duplication", (-16,)),
        planning.RotationPlan("kernel-alignment", (0,)),
        planning.RotationPlan("channel-step", (4,)),
    )
    assert plan.common.reductions == ()
    assert plan.common.mask == planning.MaskPlan(planning.MaskPolicy.NONE, 0)
    assert plan.common.slot == planning.SlotPlan(
        planning.SlotPolicy.LOGICAL_OUTPUT_ELEMENTS, 8
    )
    assert _roles(result) == ("weight", "bias", "rotation-table")
    assert _payload_by_role(result, "weight").values.shape == (4, 8)
    assert _payload_by_role(result, "weight").content_hash == (
        "sha256:d84150537daa7139bc0cf4beaaff2423ad6ffffbc4d0ff4b7fb725cff3648d0f"
    )
    assert _payload_by_role(result, "bias").content_hash == (
        "sha256:ff206fd9aa2132b161cee3b6849e42de6edb881891903d4da580cad47b484498"
    )
    rotation = _payload_by_role(result, "rotation-table")
    np.testing.assert_array_equal(rotation.values, np.array([0], dtype="<i4"))
    assert rotation.content_hash == (
        "sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf"
    )
    assert result.runtime_preparations == (
        planning.RuntimePreparation(
            "input",
            0,
            planning.RuntimePreparationKind.FLATTEN_PACKED_VECTOR,
            planning.RankedTypePlan("f32", (16,)),
            16,
            2,
            0,
            (-16,),
            0,
        ),
    )
    _assert_payload_integrity(result)


def test_fast_conv_matches_canonical_golden():
    result = planning.plan_vector_kernel(
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV)
    )
    plan = result.plan

    assert isinstance(plan, planning.FastConvPlan)
    assert (
        plan.channel_in,
        plan.channel_out,
        plan.output_height,
        plan.output_width,
        plan.kernel_hw,
        plan.group,
        plan.stride,
        plan.input_size,
        plan.output_size,
        plan.num_slots,
        plan.num_grid,
        plan.num_block,
        plan.width_block,
        plan.width_block_data,
        plan.width_block_pad,
        plan.position_block,
        plan.capacity_block,
        plan.input_duplications,
        plan.blocking_outer_depth,
        plan.cyclic_roll,
        plan.sharding_offset,
    ) == (2, 4, 2, 2, 9, 1, 1, 8, 16, 32, 2, 1, 16, 16, 0, 4, 9, 2, 0, False, None)
    assert plan.common.loops == (
        planning.LoopPlan("grid", 0, 2, 1, 0),
        planning.LoopPlan("capacity-block", 0, 9, 1, 1),
    )
    assert plan.common.slices == (
        planning.SlicePlan("weight", planning.AffineIndexPlan((9, 1), 0, False), 16),
    )
    assert plan.common.rotations == (
        planning.RotationPlan("blocking-alignment", (-3, -2, -1, -1, 0, 1, 1, 2, 3)),
        planning.RotationPlan("grid", (0, 4)),
        planning.RotationPlan("collective-tail", (-16,)),
    )
    assert plan.common.reductions == (
        planning.ReductionPlan(
            "collective",
            planning.ReductionKind.COLLECTIVE_SINGLE_BLOCK,
            1,
            16,
            0,
        ),
    )
    assert plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.COLLECTIVE_REDUCTION, 16
    )
    assert _roles(result) == (
        "weight",
        "bias",
        "rotation-table",
        "collective-mask",
    )
    assert _payload_by_role(result, "weight").values.shape == (18, 16)
    assert _payload_by_role(result, "weight").content_hash == (
        "sha256:b8609c354e6f0e1c23fb8284fec88e0c30f6a1fdef6e27622bd6d2e2377ab28a"
    )
    assert _payload_by_role(result, "bias").content_hash == (
        "sha256:02ce240b0e375bc06004c664427910ae5b8da03401d7401ad9aeba00ab9b48f6"
    )
    assert _payload_by_role(result, "rotation-table").content_hash == (
        "sha256:d5e848269a15cc35ba1af9f9b4e30e2b303ed2a1c3bf8f2390dbb499653ac5ae"
    )
    assert _payload_by_role(result, "collective-mask").content_hash == (
        "sha256:14f0eda2f8622a12587e1de0185d48afb7ee5d7ea18bc9fd09e8eb3c0c36b77d"
    )
    assert result.runtime_preparations[0].rotation_candidates == (
        -3,
        -2,
        -1,
        -1,
        0,
        1,
        1,
        2,
        3,
    )
    _assert_payload_integrity(result)


def test_conv_spatial_padding_masks_weight_and_expanded_bias():
    padded_request = _conv_request(
        4, 2, 4, 3, 256, planning.RequestedPlanKind.BASELINE_CONV
    )
    unpadded_request = _with_attribute(padded_request, "pads", [0, 0, 0, 0])
    padded = planning.plan_vector_kernel(padded_request)
    unpadded = planning.plan_vector_kernel(unpadded_request)

    padded_bias = _payload_by_role(padded, "bias").values.reshape(2, 4, 4)
    unpadded_bias = _payload_by_role(unpadded, "bias").values.reshape(2, 4, 4)
    np.testing.assert_array_equal(padded_bias[1], np.full((4, 4), 0.25, dtype="<f4"))
    np.testing.assert_array_equal(
        unpadded_bias[1],
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0],
                [0.0, 0.25, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype="<f4",
        ),
    )
    unpadded_weight = _payload_by_role(unpadded, "weight").values
    invalid_columns = np.array([0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15])
    invalid_columns = np.concatenate((invalid_columns, invalid_columns + 16))
    assert np.all(unpadded_weight[:, invalid_columns] == np.float32(0.0))
    assert _payload_by_role(padded, "weight").content_hash != (
        _payload_by_role(unpadded, "weight").content_hash
    )


def test_fast_conv_mask_fusion_omits_collective_mask_when_unrequested():
    unfused = planning.plan_vector_kernel(
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV)
    )
    fused = planning.plan_vector_kernel(
        _conv_request(
            2,
            4,
            2,
            3,
            32,
            planning.RequestedPlanKind.FAST_CONV,
            options=_options(mask_fuse=True),
        )
    )

    assert "collective-mask" in _roles(unfused)
    assert unfused.plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.COLLECTIVE_REDUCTION, 16
    )
    assert "collective-mask" not in _roles(fused)
    assert fused.plan.common.mask == planning.MaskPlan(planning.MaskPolicy.NONE, 0)
    assert fused.plan.common.reductions == unfused.plan.common.reductions


def test_fast_conv_cyclic_plan_owns_masks_and_rotation_topology():
    result = planning.plan_vector_kernel(
        _conv_request(2, 3, 3, 1, 32, planning.RequestedPlanKind.FAST_CONV)
    )
    plan = result.plan

    assert isinstance(plan, planning.FastConvPlan)
    assert plan.channel_in == 3
    assert plan.cyclic_roll
    assert plan.common.reductions == ()
    assert plan.common.mask.policy is planning.MaskPolicy.NONE
    assert _roles(result) == (
        "weight",
        "bias",
        "rotation-table",
        "cyclic-mask-left",
        "cyclic-mask-right",
    )
    assert tuple(item.role for item in plan.common.rotations) == (
        "blocking-alignment",
        "cyclic-left",
        "cyclic-right",
    )
    assert plan.common.rotations[1].candidates == (-27, -18, -9)
    assert plan.common.rotations[2].candidates == (0, 9, 18)
    assert tuple(item.role for item in plan.common.slices) == (
        "weight",
        "cyclic-mask-left",
        "cyclic-mask-right",
    )
    left = _payload_by_role(result, "cyclic-mask-left").values
    right = _payload_by_role(result, "cyclic-mask-right").values
    assert left.shape == right.shape == (3, 27)
    np.testing.assert_array_equal(left + right, np.ones((3, 27), dtype="<f4"))
    assert tuple(np.count_nonzero(row) for row in left) == (0, 9, 18)
    _assert_payload_integrity(result)


def test_fast_conv_parallel_plan_has_collective_block_and_gap_topology():
    result = planning.plan_vector_kernel(
        _conv_request(
            32,
            32,
            2,
            3,
            512,
            planning.RequestedPlanKind.FAST_CONV,
            options=_options(conv_parallel=True),
        )
    )
    plan = result.plan

    assert isinstance(plan, planning.FastConvPlan)
    assert (plan.num_block, plan.num_grid, plan.width_block_pad) == (2, 16, 64)
    assert (plan.width_block, plan.input_duplications) == (192, 3)
    assert _payload_by_role(result, "weight").values.shape == (144, 384)
    assert _roles(result)[-2:] == ("collective-mask", "collective-gap-mask")
    assert plan.common.reductions == (
        planning.ReductionPlan(
            "collective",
            planning.ReductionKind.COLLECTIVE_BLOCKS,
            2,
            128,
            64,
        ),
    )
    assert tuple(item.role for item in plan.common.loops) == (
        "grid",
        "capacity-block",
        "collective-data",
        "collective-gap",
    )
    assert tuple(item.role for item in plan.common.rotations[-3:]) == (
        "collective-data",
        "collective-gap",
        "collective-tail",
    )
    gap = _payload_by_role(result, "collective-gap-mask").values
    assert np.count_nonzero(gap) == 64
    assert np.all(gap[:64] == 0.0)
    assert np.all(gap[-64:] == 1.0)
    _assert_payload_integrity(result)


def test_fast_depthwise_conv_uses_kernel_channel_grid_without_reduction():
    result = planning.plan_vector_kernel(
        _conv_request(
            4,
            4,
            4,
            3,
            128,
            planning.RequestedPlanKind.FAST_CONV,
            group=4,
        )
    )
    plan = result.plan

    assert isinstance(plan, planning.FastConvPlan)
    assert plan.group == plan.channel_in == 4
    assert plan.num_grid == 1
    assert not plan.cyclic_roll
    assert plan.common.reductions == ()
    assert _payload_by_role(result, "weight").values.shape == (9, 64)
    assert _roles(result) == ("weight", "bias", "rotation-table")


def test_fast_sharded_conv_owns_weight_and_s32_offset_preparation():
    result = planning.plan_vector_kernel(
        _conv_request(
            2,
            4,
            2,
            3,
            32,
            planning.RequestedPlanKind.FAST_CONV,
            sharded=True,
        )
    )
    plan = result.plan

    assert isinstance(plan, planning.FastConvPlan)
    assert plan.sharding_offset == planning.ShardingOffsetPlan("s32", 18)
    assert plan.blocking_outer_depth == 1
    assert plan.common.runtime_scalar_inputs == ("s32",)
    assert plan.common.slices[0].index.uses_sharding_offset
    assert _payload_by_role(result, "weight").values.shape == (36, 16)
    assert _roles(result) == (
        "weight",
        "bias",
        "rotation-table",
        "collective-mask",
        "collective-gap-mask",
    )
    assert result.runtime_preparations[0].outer_block_depth == 1
    assert result.scalar_preparations == (
        planning.ScalarPreparation("weight-offset", 1, "s32", 18),
    )
    _assert_payload_integrity(result)


@pytest.mark.parametrize("dtype", ["<i4", "<i8", "<u4"])
def test_conv_accepts_canonical_integer_attributes(dtype):
    result = planning.plan_vector_kernel(
        _conv_request(
            2,
            4,
            2,
            3,
            32,
            planning.RequestedPlanKind.FAST_CONV,
            attribute_dtype=dtype,
        )
    )
    canonical = planning.plan_vector_kernel(
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV)
    )
    assert _normalized(result) == _normalized(canonical)


def test_conv_accepts_u32_explicit_mask_attribute():
    request = _conv_request(
        2,
        4,
        2,
        3,
        32,
        planning.RequestedPlanKind.FAST_CONV,
        options=_options(mask_fuse=True),
    )
    request = _with_attribute(request, "mask", [1], "<u4")
    result = planning.plan_vector_kernel(request)

    assert "collective-mask" in _roles(result)
    assert result.plan.common.mask == planning.MaskPlan(
        planning.MaskPolicy.COLLECTIVE_REDUCTION, 16
    )
    _assert_payload_integrity(result)


@pytest.mark.parametrize("name", ["group", "strides", "pads"])
def test_conv_rejects_unsupported_u64_operational_attributes(name):
    request = _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV)
    original = request.attribute(name)
    request = _with_attribute(request, name, original.values, "<u8")

    with pytest.raises(planning.PlanningError, match="not a canonical integer array"):
        planning.plan_vector_kernel(request)


@pytest.mark.parametrize(
    ("mutate", "diagnostic"),
    [
        (
            lambda request: replace(
                request, source_constants=request.source_constants[:1]
            ),
            "owned f32 weight/bias",
        ),
        (
            lambda request: replace(
                request,
                source_constants=(
                    _payload("weight", np.arange(8), "<f8", (2, 4)),
                    request.source_constants[1],
                ),
            ),
            "owned f32 weight/bias",
        ),
        (
            lambda request: replace(
                request,
                source_constants=(
                    _payload("weight", np.arange(8), "<f4", (8,)),
                    request.source_constants[1],
                ),
            ),
            "owned f32 weight/bias",
        ),
        (
            lambda request: replace(request, target=planning.TargetSnapshot(0, 1, 16)),
            "positive dimensions",
        ),
    ],
    ids=("missing-bias", "wrong-dtype", "wrong-rank", "zero-slots"),
)
def test_fast_gemm_rejects_invalid_dimensions_and_payloads(mutate, diagnostic):
    request = mutate(_gemm_request(planning.RequestedPlanKind.FAST_GEMM))
    with pytest.raises(planning.PlanningError, match=diagnostic):
        planning.plan_vector_kernel(request)


@pytest.mark.parametrize(
    ("case_request", "diagnostic"),
    [
        (
            replace(
                _conv_request(2, 4, 2, 3, 32),
                declared_result_type=planning.RankedTypePlan("f32", (1, 4, 1, 2)),
            ),
            "declared result type",
        ),
        (
            _with_attribute(_conv_request(2, 4, 2, 3, 32), "group", [0]),
            "group, stride, or padding",
        ),
        (
            _with_attribute(_conv_request(2, 4, 2, 3, 32), "strides", [1, 2]),
            "equal spatial strides",
        ),
        (
            replace(
                _conv_request(2, 4, 2, 3, 32),
                operand_types=(
                    planning.RankedTypePlan("f32", (2, 2, 2, 2)),
                    planning.RankedTypePlan("f32", (4, 2, 3, 3)),
                    planning.RankedTypePlan("f32", (4,)),
                ),
            ),
            "batch size one",
        ),
        (
            replace(
                _conv_request(2, 4, 2, 3, 32),
                operand_types=(
                    planning.RankedTypePlan("f32", (1, 2, 2, 2)),
                    planning.RankedTypePlan("f32", (4, 2, 2, 3)),
                    planning.RankedTypePlan("f32", (4,)),
                ),
            ),
            "square kernels",
        ),
        (
            replace(
                _conv_request(2, 4, 2, 3, 32),
                source_constants=(
                    _payload("weight", np.arange(8), "<f4", (8,)),
                    _payload("bias", np.arange(4), "<f4", (4,)),
                ),
            ),
            "does not match its shape",
        ),
    ],
    ids=(
        "result-shape",
        "zero-group",
        "unequal-stride",
        "batch",
        "rectangular-kernel",
        "weight-count",
    ),
)
def test_conv_rejects_invalid_dimensions_shapes_and_attributes(
    case_request, diagnostic
):
    with pytest.raises(planning.PlanningError, match=diagnostic):
        planning.plan_vector_kernel(case_request)


def test_conv_checked_arithmetic_rejects_dimension_stride_and_halo_overflow():
    request = _conv_request(2, 4, 2, 3, 32)
    huge_dimensions = replace(
        request,
        operand_types=(
            planning.RankedTypePlan("f32", (1, 2, 1 << 32, 1 << 32)),
            request.operand_types[1],
            request.operand_types[2],
        ),
        declared_result_type=planning.RankedTypePlan("f32", (1, 4, 1 << 32, 1 << 32)),
    )
    with pytest.raises(planning.PlanningError, match="dimensions overflow"):
        planning.plan_vector_kernel(huge_dimensions)

    huge_stride = _with_attribute(
        request,
        "strides",
        [_I64_MAX, _I64_MAX],
        "<i8",
        options=_options(selective_strided_slice=True),
    )
    with pytest.raises(
        planning.PlanningError, match="stride-scaled rotation overflows"
    ):
        planning.plan_vector_kernel(huge_stride)

    sharded = _conv_request(
        2,
        4,
        2,
        3,
        32,
        planning.RequestedPlanKind.FAST_CONV,
        sharded=True,
    )
    huge_halo = _with_attribute(sharded, "sharding", [2, 1, 2, _I64_MAX], "<i8")
    with pytest.raises(planning.PlanningError, match="halo size overflows"):
        planning.plan_vector_kernel(huge_halo)


def test_all_variants_are_deterministic_order_independent_and_own_results():
    requests = (
        _gemm_request(planning.RequestedPlanKind.BASELINE_GEMM),
        _gemm_request(planning.RequestedPlanKind.FAST_GEMM),
        _conv_request(4, 2, 2, 1, 32, planning.RequestedPlanKind.BASELINE_CONV),
        _conv_request(2, 4, 2, 3, 32, planning.RequestedPlanKind.FAST_CONV),
    )

    for request in requests:
        first = planning.plan_vector_kernel(request)
        second = planning.plan_vector_kernel(request)
        reordered = planning.plan_vector_kernel(
            replace(
                request,
                attributes=tuple(reversed(request.attributes)),
                source_constants=tuple(reversed(request.source_constants)),
            )
        )
        assert _normalized(first) == _normalized(second)
        assert _normalized(first) == _normalized(reordered)
        _assert_payload_integrity(first)

        before = _normalized(first)
        for output in first.constants:
            for source in request.source_constants:
                assert not np.shares_memory(output.values, source.values)
            with pytest.raises(ValueError, match="read-only"):
                output.values.reshape(-1)[0] = np.float32(99.0)
        for source in request.source_constants:
            source.values.reshape(-1)[:] = np.float32(-123.0)
        for attribute in request.attributes:
            attribute.values.reshape(-1)[:] = 0
        assert _normalized(first) == before


def test_ranked_gemm_runtime_input_shape_and_float_bits_are_preserved():
    bias_bits = np.array([0x80000000, 0x7FC01234], dtype="<u4")
    request = _gemm_request(
        planning.RequestedPlanKind.BASELINE_GEMM,
        input_shape=(2, 2),
        bias_values=bias_bits.view("<f4"),
    )
    result = planning.plan_vector_kernel(request)

    assert result.plan.common.runtime_vector_inputs == (
        planning.RankedTypePlan("f32", (2, 2)),
    )
    assert result.runtime_preparations[0].result_type == planning.RankedTypePlan(
        "f32", (2, 2)
    )
    assert _payload_by_role(result, "bias").values.tobytes() == bias_bits.tobytes()
