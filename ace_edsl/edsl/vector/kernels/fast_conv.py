"""Complete fast Conv destination recipe backed by ``@vector_kernel``."""

import hashlib

from ace_edsl.base_dsl.ast_helpers import const_expr, range_dynamic
from ace_edsl.edsl.core.vector_value import VectorValue
from ace_edsl.edsl.domain_kernels import vector_kernel
from ace_edsl.edsl.vector.kernels.fast_common import (
    _ranked_constant,
    _require_core_i32,
    blocking_rot,
    collective_reduce,
    roll_cyclic,
)
from ace_edsl.edsl.vector.lowering import PreparedFastConvPlan


def _local_vector(container, name, air_type):
    return VectorValue(
        None,
        container,
        shape=tuple(air_type.shape()),
        temp_name=name,
        air_type=air_type,
    )


def _validate_loop(loop, role, lower, upper, depth):
    if (
        loop.role != role
        or loop.lower != lower
        or loop.upper != upper
        or loop.step != 1
        or loop.nesting_depth != depth
    ):
        raise ValueError("fast Conv loop topology is inconsistent")


def _validate_frozen_identity(prepared):
    vector_inputs = ";".join(
        f"{item.element_type}[{','.join(str(value) for value in item.shape)}]"
        for item in prepared.runtime_vector_inputs
    )
    scalar_inputs = ";".join(prepared.runtime_scalar_inputs)
    sharding = (
        "none"
        if prepared.sharding_offset is None
        else f"{prepared.sharding_offset.type}*{prepared.sharding_offset.scale}"
    )
    fields = (
        ("channel-in", prepared.channel_in),
        ("channel-out", prepared.channel_out),
        ("output-height", prepared.output_height),
        ("output-width", prepared.output_width),
        ("kernel-hw", prepared.kernel_hw),
        ("group", prepared.group),
        ("stride", prepared.stride),
        ("input-size", prepared.input_size),
        ("output-size", prepared.output_size),
        ("num-slots", prepared.num_slots),
        ("num-grid", prepared.num_grid),
        ("num-block", prepared.num_block),
        ("width-block", prepared.width_block),
        ("width-block-data", prepared.width_block_data),
        ("width-block-pad", prepared.width_block_pad),
        ("position-block", prepared.position_block),
        ("capacity-block", prepared.capacity_block),
        ("input-duplications", prepared.input_duplications),
        ("blocking-outer-depth", prepared.blocking_outer_depth),
        ("cyclic-roll", 1 if prepared.cyclic_roll else 0),
        ("sharding-offset", sharding),
    )
    suffix = "".join(f"|{name}={value}" for name, value in fields)
    record_fragments = [
        f"|vector-inputs={len(prepared.runtime_vector_inputs)}"
        f"{{{vector_inputs}}}",
        f"|scalar-inputs={len(prepared.runtime_scalar_inputs)}"
        f"{{{scalar_inputs}}}",
        *(
            f"{len(item.role)}:{item.role}"
            f"({item.lower},{item.upper},{item.step},{item.nesting_depth})"
            for item in prepared.loops
        ),
        *(
            f"{len(item.role)}:{item.role}"
            f"([{','.join(str(value) for value in item.index.iv_coefficients)}],"
            f"{item.index.constant},{1 if item.index.uses_sharding_offset else 0},"
            f"{item.width})"
            for item in prepared.slices
        ),
        *(
            f"{len(item.role)}:{item.role}"
            f"[{','.join(str(value) for value in item.candidates)}]"
            for item in prepared.rotations
        ),
        *(
            f"{len(item.role)}:{item.role}"
            f"({item.kind},{item.factor},{item.block_width},{item.padding})"
            for item in prepared.reductions
        ),
        f"|mask={prepared.mask.policy}:{prepared.mask.valid_length}",
        f"|slot={prepared.slot.policy}:{prepared.slot.value}",
        "|result="
        f"{prepared.result_type.element_type}"
        f"[{','.join(str(value) for value in prepared.result_type.shape)}]",
    ]
    digest = hashlib.sha256(
        prepared.specialization_key.encode("utf-8")
    ).hexdigest()
    if (
        not prepared.specialization_key.startswith(
            "vector-kernel-plan:v1|kind=fast-conv|"
        )
        or not prepared.specialization_key.endswith(suffix)
        or prepared.helper_name != f"__ace_vkernel_fast_conv_{digest}"
        or any(
            fragment not in prepared.specialization_key
            for fragment in record_fragments
        )
    ):
        raise ValueError("fast Conv frozen identity is inconsistent")


def _validate_payload(descriptor):
    element_width = {"f32": 4, "s32": 4}.get(
        descriptor.type.element_type
    )
    if element_width is None:
        raise ValueError("fast Conv constant payload type is unsupported")
    elements = 1
    for extent in descriptor.type.shape:
        if extent <= 0:
            raise ValueError("fast Conv constant payload shape is invalid")
        elements *= extent
    payload = bytes(descriptor.bytes)
    header = (
        "vector-kernel-constant:v1\n"
        f"element={descriptor.type.element_type}\n"
        f"rank={len(descriptor.type.shape)}\n"
        "shape="
        + ",".join(str(extent) for extent in descriptor.type.shape)
        + "\npayload:\n"
    ).encode("ascii")
    digest = "sha256:" + hashlib.sha256(header + payload).hexdigest()
    if (
        len(payload) != elements * element_width
        or descriptor.content_hash != digest
    ):
        raise ValueError(
            "fast Conv constant payload or hash is inconsistent for "
            f"{descriptor.role}: bytes={len(payload)}, "
            f"expected={elements * element_width}, "
            f"hash_matches={descriptor.content_hash == digest}"
        )
    return payload


def _validate_prepared(prepared):
    if prepared.kind != "fast-conv":
        raise ValueError("fast Conv recipe received another plan kind")
    if len(prepared.runtime_vector_inputs) != 1:
        raise ValueError("fast Conv recipe requires one Vector input")
    if (
        prepared.channel_in <= 0
        or prepared.channel_out <= 0
        or prepared.output_height <= 0
        or prepared.output_width <= 0
        or prepared.kernel_hw <= 0
        or prepared.group <= 0
        or prepared.stride <= 0
        or prepared.num_grid <= 0
        or prepared.num_block <= 0
        or prepared.num_slots <= 0
        or prepared.width_block <= 0
        or prepared.width_block_data <= 0
        or prepared.width_block_pad < 0
        or prepared.position_block <= 0
        or prepared.capacity_block <= 0
        or prepared.input_duplications <= 0
        or prepared.blocking_outer_depth < 0
        or prepared.input_size
        != prepared.channel_in
        * prepared.output_height
        * prepared.output_width
        or prepared.output_size
        != prepared.channel_out
        * prepared.output_height
        * prepared.output_width
        or prepared.output_size > prepared.num_slots
        or prepared.width_block_data != prepared.output_size
        or prepared.width_block
        != prepared.width_block_data + prepared.width_block_pad
        or prepared.result_type.shape
        != (
            1,
            prepared.channel_out,
            prepared.output_height,
            prepared.output_width,
        )
        or prepared.result_type.element_type != "f32"
        or prepared.runtime_vector_inputs[0].element_type != "f32"
        or len(prepared.runtime_vector_inputs[0].shape) != 1
        or prepared.runtime_vector_inputs[0].shape[0] <= 0
        or prepared.position_block
        != (
            prepared.output_height
            * prepared.output_width
            * (prepared.capacity_block if prepared.kernel_hw == 1 else 1)
        )
    ):
        raise ValueError("fast Conv dimensions are inconsistent")
    if (
        prepared.slot.policy != "logical-output-elements"
        or prepared.slot.value != prepared.output_size
    ):
        raise ValueError("fast Conv SLOT does not match the logical output")

    preparation = prepared.runtime_preparation("input")
    blocking_rotation = prepared.rotation("blocking-alignment")
    if (
        len(prepared.runtime_preparations) != 1
        or preparation.source_operand != 0
        or preparation.kind != "blocking-rotations"
        or preparation.result_type != prepared.runtime_vector_inputs[0]
        or preparation.logical_input_size != prepared.input_size
        or preparation.replications != prepared.input_duplications
        or preparation.blocking_width != prepared.capacity_block
        or preparation.rotation_candidates != blocking_rotation.candidates
        or preparation.outer_block_depth != prepared.blocking_outer_depth
        or len(blocking_rotation.candidates) != prepared.capacity_block
    ):
        raise ValueError("fast Conv input preparation is inconsistent")

    sharded = prepared.sharding_offset is not None
    if sharded:
        scalar = prepared.scalar_preparation("weight-offset")
        if (
            prepared.runtime_scalar_inputs != ("s32",)
            or len(prepared.scalar_preparations) != 1
            or prepared.sharding_offset.type != "s32"
            or scalar.source_operand != 1
            or scalar.type != "s32"
            or scalar.scale != prepared.sharding_offset.scale
            or scalar.scale <= 0
            or prepared.blocking_outer_depth not in (0, 1)
        ):
            raise ValueError("fast Conv sharding scalar is inconsistent")
    elif (
        prepared.runtime_scalar_inputs
        or prepared.scalar_preparations
        or prepared.blocking_outer_depth != 0
    ):
        raise ValueError("unsharded fast Conv has scalar or outer-depth state")

    expected_loops = ["grid", "capacity-block"]
    _validate_loop(prepared.loop("grid"), "grid", 0, prepared.num_grid, 0)
    _validate_loop(
        prepared.loop("capacity-block"),
        "capacity-block",
        0,
        prepared.capacity_block,
        1,
    )

    weight_slice = prepared.slice("weight")
    slice_width = prepared.num_block * prepared.width_block
    expected_slices = ["weight"]
    if (
        weight_slice.index.iv_coefficients
        != (prepared.capacity_block, 1)
        or weight_slice.index.constant != 0
        or weight_slice.index.uses_sharding_offset != sharded
        or weight_slice.width != slice_width
    ):
        raise ValueError("fast Conv weight slice is inconsistent")

    expected_cyclic = (
        prepared.num_block == 1
        and prepared.output_size > prepared.num_slots // 2
        and prepared.output_size < prepared.num_slots
        and prepared.group != prepared.channel_in
    )
    expected_collective = (
        prepared.channel_in > 1
        and prepared.group != prepared.channel_in
        and not expected_cyclic
    ) or prepared.output_size == prepared.num_slots
    if prepared.cyclic_roll != expected_cyclic:
        raise ValueError("fast Conv cyclic topology policy is inconsistent")
    if bool(prepared.reductions) != expected_collective:
        raise ValueError("fast Conv collective topology policy is inconsistent")

    expected_rotations = ["blocking-alignment"]
    if prepared.cyclic_roll:
        expected_slices.extend(("cyclic-mask-left", "cyclic-mask-right"))
        expected_rotations.extend(("cyclic-left", "cyclic-right"))
        for role in ("cyclic-mask-left", "cyclic-mask-right"):
            mask_slice = prepared.slice(role)
            if (
                mask_slice.index.iv_coefficients != (1,)
                or mask_slice.index.constant != 0
                or mask_slice.index.uses_sharding_offset
                or mask_slice.width != prepared.output_size
            ):
                raise ValueError("fast Conv cyclic slice topology is inconsistent")
        expected_left = tuple(
            index * prepared.position_block - prepared.output_size
            for index in range(prepared.num_grid)
        )
        expected_right = tuple(
            index * prepared.position_block
            for index in range(prepared.num_grid)
        )
        if (
            prepared.rotation("cyclic-left").candidates != expected_left
            or prepared.rotation("cyclic-right").candidates != expected_right
        ):
            raise ValueError("fast Conv cyclic rotation topology is inconsistent")
    else:
        expected_rotations.append("grid")
        expected_grid = tuple(
            index * prepared.position_block
            for index in range(prepared.num_grid)
        )
        if prepared.rotation("grid").candidates != expected_grid:
            raise ValueError("fast Conv grid rotation topology is inconsistent")

    if [item.role for item in prepared.slices] != expected_slices:
        raise ValueError("fast Conv slice topology is inconsistent")

    if len(prepared.reductions) > 1:
        raise ValueError("fast Conv collective topology is inconsistent")
    expected_constant_roles = ["weight", "bias", "rotation-table"]
    if prepared.cyclic_roll:
        expected_constant_roles.extend(
            ("cyclic-mask-left", "cyclic-mask-right")
        )

    if prepared.reductions:
        reduction = prepared.reduction("collective")
        if (
            reduction.kind
            not in ("collective-single-block", "collective-blocks")
            or reduction.factor != prepared.num_block
            or reduction.block_width != prepared.output_size
            or reduction.padding < 0
            or reduction.padding > prepared.output_size
            or prepared.mask.policy
            not in ("none", "collective-reduction")
            or prepared.mask.valid_length
            != (
                prepared.output_size
                if prepared.mask.policy == "collective-reduction"
                else 0
            )
        ):
            raise ValueError("fast Conv collective topology is inconsistent")
        if reduction.kind == "collective-single-block":
            expected_padding = prepared.width_block_pad
            if sharded or prepared.num_block != 1:
                raise ValueError(
                    "fast Conv collective overload is inconsistent"
                )
        else:
            expected_padding = prepared.width_block_pad
            if (
                prepared.num_block == 1
                and prepared.output_size != prepared.num_slots
            ):
                expected_padding = (
                    (prepared.channel_in - 1)
                    * prepared.output_height
                    * prepared.output_width
                )
        if reduction.padding != expected_padding:
            raise ValueError("fast Conv collective padding is inconsistent")
        if prepared.output_size == prepared.num_slots:
            if prepared.mask.policy == "collective-reduction":
                expected_constant_roles.append("collective-mask")
        elif reduction.kind == "collective-single-block":
            expected_rotations.append("collective-tail")
            if prepared.mask.policy == "collective-reduction":
                expected_constant_roles.append("collective-mask")
        else:
            expected_constant_roles.extend(
                ("collective-mask", "collective-gap-mask")
            )
            if prepared.num_block > 1:
                expected_loops.extend(("collective-data", "collective-gap"))
                _validate_loop(
                    prepared.loop("collective-data"),
                    "collective-data",
                    1,
                    prepared.num_block,
                    0,
                )
                _validate_loop(
                    prepared.loop("collective-gap"),
                    "collective-gap",
                    0,
                    prepared.num_block - 1,
                    0,
                )
                expected_rotations.extend(
                    ("collective-data", "collective-gap")
                )
                expected_data = tuple(
                    index * prepared.width_block
                    for index in range(1, prepared.num_block)
                )
                expected_gap = tuple(
                    index * prepared.width_block + reduction.padding
                    for index in range(prepared.num_block - 1)
                )
                if (
                    prepared.rotation("collective-data").candidates
                    != expected_data
                    or prepared.rotation("collective-gap").candidates
                    != expected_gap
                ):
                    raise ValueError(
                        "fast Conv collective rotation topology is inconsistent"
                    )
            expected_rotations.append("collective-tail")
        if (
            prepared.output_size != prepared.num_slots
            and prepared.rotation("collective-tail").candidates
            != (-prepared.output_size,)
        ):
            raise ValueError("fast Conv collective tail is inconsistent")
    elif (
        prepared.mask.policy != "none"
        or prepared.mask.valid_length != 0
    ):
        raise ValueError("fast Conv mask requires collective reduction")

    if [item.role for item in prepared.loops] != expected_loops:
        raise ValueError("fast Conv loop topology is inconsistent")
    if [item.role for item in prepared.rotations] != expected_rotations:
        raise ValueError("fast Conv rotation topology is inconsistent")

    descriptors = {item.role: item for item in prepared.constants}
    if [item.role for item in prepared.constants] != expected_constant_roles:
        raise ValueError("fast Conv constant topology is inconsistent")
    if (
        descriptors["weight"].type.element_type != "f32"
        or len(descriptors["weight"].type.shape) != 2
        or descriptors["weight"].type.shape[0]
        < prepared.num_grid * prepared.capacity_block
        or descriptors["weight"].type.shape[1] != slice_width
        or descriptors["bias"].type.element_type != "f32"
        or descriptors["bias"].type.shape != (prepared.output_size,)
        or descriptors["rotation-table"].type.element_type != "s32"
        or descriptors["rotation-table"].type.shape
        != (prepared.capacity_block,)
    ):
        raise ValueError("fast Conv constant descriptors are inconsistent")
    for role in expected_constant_roles[3:]:
        descriptor = descriptors[role]
        expected_shape = (
            (prepared.num_grid, prepared.output_size)
            if role.startswith("cyclic-mask-")
            else (prepared.output_size,)
        )
        if (
            descriptor.type.element_type != "f32"
            or descriptor.type.shape != expected_shape
        ):
            raise ValueError("fast Conv mask descriptor is inconsistent")

    payloads = {
        role: _validate_payload(descriptor)
        for role, descriptor in descriptors.items()
    }
    if any(
        descriptor.content_hash not in prepared.specialization_key
        for descriptor in prepared.constants
    ):
        raise ValueError("fast Conv constant identity is not frozen in the key")
    rotation_bytes = b"".join(
        int(value).to_bytes(4, "little", signed=True)
        for value in blocking_rotation.candidates
    )
    if payloads["rotation-table"] != rotation_bytes:
        raise ValueError("fast Conv rotation-table payload is inconsistent")

    f32_zero = bytes.fromhex("00000000")
    f32_one = bytes.fromhex("0000803f")
    if prepared.cyclic_roll:
        expected_left = bytearray()
        expected_right = bytearray()
        for grid in range(prepared.num_grid):
            prefix = grid * prepared.position_block
            if prefix > prepared.output_size:
                raise ValueError("fast Conv cyclic mask prefix is invalid")
            for column in range(prepared.output_size):
                in_prefix = column < prefix
                expected_left.extend(f32_one if in_prefix else f32_zero)
                expected_right.extend(f32_zero if in_prefix else f32_one)
        if (
            payloads["cyclic-mask-left"] != bytes(expected_left)
            or payloads["cyclic-mask-right"] != bytes(expected_right)
        ):
            raise ValueError("fast Conv cyclic mask payload is inconsistent")
    if "collective-mask" in payloads and payloads["collective-mask"] != (
        f32_one * prepared.output_size
    ):
        raise ValueError("fast Conv collective mask payload is inconsistent")
    if "collective-gap-mask" in payloads:
        padding = prepared.reduction("collective").padding
        expected_gap = (
            f32_zero * (prepared.output_size - padding)
            + f32_one * padding
        )
        if payloads["collective-gap-mask"] != expected_gap:
            raise ValueError("fast Conv collective gap mask is inconsistent")
    _validate_frozen_identity(prepared)


def _validate_inputs(packed_input, weight_offset, prepared):
    if prepared.sharding_offset is None:
        if weight_offset is not None:
            raise ValueError("unsharded fast Conv received a scalar input")
        return
    if weight_offset is None:
        raise ValueError("sharded fast Conv requires a weight offset")
    _require_core_i32(weight_offset, "fast Conv weight offset")
    if weight_offset.container is not packed_input.container:
        raise ValueError("fast Conv inputs belong to different containers")


@vector_kernel
def _emit_fast_conv(
    packed_input,
    weight_offset,
    prepared,
    constants,
    result_type,
    i32_type,
):
    _validate_prepared(prepared)
    _validate_inputs(packed_input, weight_offset, prepared)

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

    result_name = "__conv_result"
    grid_name = "__grid_result"
    container.new_local(result_name, result_type)
    container.new_stid(result_name, container.new_zero(result_type))
    container.new_local(grid_name, result_type)

    grid_loop = prepared.loop("grid")
    capacity_loop = prepared.loop("capacity-block")
    weight_slice = prepared.slice("weight")
    grid_rotation = (
        None if prepared.cyclic_roll else prepared.rotation("grid")
    )
    for grid_iv in range_dynamic(
        grid_loop.lower, grid_loop.upper, grid_loop.step
    ):
        container.new_stid(grid_name, container.new_zero(result_type))
        for block_iv in range_dynamic(
            capacity_loop.lower,
            capacity_loop.upper,
            capacity_loop.step,
        ):
            slice_index = block_iv + grid_iv * prepared.capacity_block
            if const_expr(weight_offset is not None):
                scaled_offset = (
                    weight_offset * prepared.sharding_offset.scale
                )
                slice_index = scaled_offset + slice_index
            contribution = blocked.load(block_iv) * weight.slice(
                slice_index, weight_slice.width
            )
            grid_result = _local_vector(
                container, grid_name, result_type
            )
            container.new_stid(
                grid_name, (grid_result + contribution).value
            )

        grid_result = _local_vector(container, grid_name, result_type)
        if const_expr(prepared.cyclic_roll):
            aligned = roll_cyclic(
                grid_result, grid_iv, prepared, constants, i32_type
            )
        else:
            shift = grid_iv * prepared.position_block
            aligned = grid_result.roll(
                shift, candidates=grid_rotation.candidates
            )
        result = _local_vector(container, result_name, result_type)
        container.new_stid(result_name, (result + aligned).value)

    result = _local_vector(container, result_name, result_type)
    if const_expr(bool(prepared.reductions)):
        result = collective_reduce(
            result,
            prepared,
            constants,
            i32_type,
            "__collective_result",
        )
    container.new_stid(result_name, (result + bias).value)
    return _local_vector(
        container, result_name, result_type
    ).with_slot(prepared.slot.value)


@vector_kernel
def fast_conv_vector_kernel(
    packed_input,
    prepared: PreparedFastConvPlan,
    constants,
    result_type,
    i32_type,
):
    """Emit one complete unsharded frozen fast-Conv lowering."""
    return _emit_fast_conv(
        packed_input,
        None,
        prepared,
        constants,
        result_type,
        i32_type,
    )


@vector_kernel
def fast_conv_sharded_vector_kernel(
    packed_input,
    weight_offset,
    prepared: PreparedFastConvPlan,
    constants,
    result_type,
    i32_type,
):
    """Emit one complete sharded frozen fast-Conv lowering."""
    return _emit_fast_conv(
        packed_input,
        weight_offset,
        prepared,
        constants,
        result_type,
        i32_type,
    )


def fast_conv_recipe(trace_context, prepared):
    """Dispatch the frozen one- or two-formal fast-Conv ABI."""
    from ace_edsl.edsl.edsl import AceEDSL

    _validate_prepared(prepared)
    kernel = (
        fast_conv_sharded_vector_kernel
        if prepared.sharding_offset is not None
        else fast_conv_vector_kernel
    )
    return AceEDSL._get_dsl().trace_vector_kernel_into(
        trace_context, kernel, prepared
    )


def configure_fast_conv_dsl(
    pipeline,
    *,
    mask_fuse=False,
    max_slots=0,
    conv_parallel=False,
    sharding=False,
):
    """Configure one pipeline for the C++-plan/Python fast-Conv path."""
    return pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl="dsl",
        plan_kind="fast-conv",
        fallback="error",
        mask_fuse=mask_fuse,
        max_slots=max_slots,
        conv_parallel=conv_parallel,
        sharding=sharding,
    ).register_vector_kernel_recipe("fast-conv", fast_conv_recipe)


__all__ = [
    "configure_fast_conv_dsl",
    "fast_conv_recipe",
    "fast_conv_sharded_vector_kernel",
    "fast_conv_vector_kernel",
]
