"""AIR-independent Python planning for Vector Conv and Gemm kernels.

The records in this module deliberately contain only immutable host data.
They mirror the provider-neutral C++ planning boundary; AIR construction,
canonical validation, specialization keys, and helper names remain owned by
C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Iterable, Mapping, Sequence, Union

import numpy as np


_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class PlanningError(ValueError):
    """An invalid request or checked planning operation."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Operation(_StringEnum):
    GEMM = "gemm"
    CONV = "conv"


class RequestedPlanKind(_StringEnum):
    AUTO = "auto"
    BASELINE_GEMM = "baseline-gemm"
    BASELINE_CONV = "baseline-conv"
    FAST_GEMM = "fast-gemm"
    FAST_CONV = "fast-conv"


class MaskPolicy(_StringEnum):
    NONE = "none"
    CLEAR_VALID_PREFIX = "clear-valid-prefix"
    COLLECTIVE_REDUCTION = "collective-reduction"


class SlotPolicy(_StringEnum):
    ABSENT_NATIVE_BASELINE = "absent-native-baseline"
    LOGICAL_OUTPUT_ELEMENTS = "logical-output-elements"
    EXPLICIT = "explicit"


class ReductionKind(_StringEnum):
    POWER_OF_TWO = "power-of-two"
    LINEAR = "linear"
    COLLECTIVE_SINGLE_BLOCK = "collective-single-block"
    COLLECTIVE_BLOCKS = "collective-blocks"


class RuntimePreparationKind(_StringEnum):
    PACKED_VECTOR = "packed-vector"
    FLATTEN_PACKED_VECTOR = "flatten-packed-vector"
    BLOCKING_ROTATIONS = "blocking-rotations"


@dataclass(frozen=True)
class RankedTypePlan:
    element_type: str
    shape: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class TypedPayload:
    role: str
    values: np.ndarray
    content_hash: str

    @property
    def type(self) -> RankedTypePlan:
        return RankedTypePlan(
            _primitive_for_dtype(self.values.dtype), self.values.shape
        )


@dataclass(frozen=True, eq=False)
class AttributeRecord:
    name: str
    values: np.ndarray

    @property
    def type(self) -> RankedTypePlan:
        return RankedTypePlan(
            _primitive_for_dtype(self.values.dtype), self.values.shape
        )


@dataclass(frozen=True)
class OptionSnapshot:
    conv_parallel: bool
    mask_fuse: bool
    selective_strided_slice: bool
    sharding: bool
    is_last_operation: bool


@dataclass(frozen=True)
class TargetSnapshot:
    num_slots: int
    min_slots: int
    max_slots: int


@dataclass(frozen=True, eq=False)
class PlanningRequest:
    operation: Operation
    attributes: tuple[AttributeRecord, ...]
    operand_types: tuple[RankedTypePlan, ...]
    declared_result_type: RankedTypePlan
    options: OptionSnapshot
    target: TargetSnapshot
    source_constants: tuple[TypedPayload, ...]
    requested_plan_kind: RequestedPlanKind
    runtime_scalar_types: tuple[str, ...] = ()

    def attribute(self, name: str) -> AttributeRecord | None:
        return next((item for item in self.attributes if item.name == name), None)

    def source_constant(self, role: str) -> TypedPayload | None:
        return next((item for item in self.source_constants if item.role == role), None)


class _PlanningRequestView:
    """Callback-scoped request view used by the C++ adapter."""

    __slots__ = ("_request", "_active")

    def __init__(self, request: PlanningRequest):
        self._request = request
        self._active = True

    def __getattribute__(self, name):
        if name in {"_expire", "__class__"}:
            return object.__getattribute__(self, name)
        if name.startswith("_"):
            raise AttributeError(
                "vector-kernel planning request internals are callback-private"
            )
        if not object.__getattribute__(self, "_active"):
            raise RuntimeError("vector-kernel planning request view has expired")
        value = getattr(object.__getattribute__(self, "_request"), name)
        if not callable(value):
            return value

        def callback_scoped_method(*args, **kwargs):
            if not object.__getattribute__(self, "_active"):
                raise RuntimeError("vector-kernel planning request view has expired")
            return value(*args, **kwargs)

        return callback_scoped_method

    def _expire(self) -> None:
        object.__setattr__(self, "_active", False)


@dataclass(frozen=True)
class ConstantPlan:
    role: str
    type: RankedTypePlan
    content_hash: str


@dataclass(frozen=True)
class LoopPlan:
    role: str
    lower: int
    upper: int
    step: int
    nesting_depth: int


@dataclass(frozen=True)
class AffineIndexPlan:
    iv_coefficients: tuple[int, ...]
    constant: int
    uses_sharding_offset: bool


@dataclass(frozen=True)
class SlicePlan:
    role: str
    index: AffineIndexPlan
    width: int


@dataclass(frozen=True)
class RotationPlan:
    role: str
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class ReductionPlan:
    role: str
    kind: ReductionKind
    factor: int
    block_width: int
    padding: int


@dataclass(frozen=True)
class MaskPlan:
    policy: MaskPolicy
    valid_length: int


@dataclass(frozen=True)
class SlotPlan:
    policy: SlotPolicy
    value: int


@dataclass(frozen=True)
class CommonPlan:
    runtime_vector_inputs: tuple[RankedTypePlan, ...]
    runtime_scalar_inputs: tuple[str, ...]
    constants: tuple[ConstantPlan, ...]
    result_type: RankedTypePlan
    loops: tuple[LoopPlan, ...]
    slices: tuple[SlicePlan, ...]
    rotations: tuple[RotationPlan, ...]
    reductions: tuple[ReductionPlan, ...]
    mask: MaskPlan
    slot: SlotPlan


@dataclass(frozen=True)
class BaselineGemmPlan:
    common: CommonPlan
    height: int
    width: int
    input_duplications: int


@dataclass(frozen=True)
class BaselineConvPlan:
    common: CommonPlan
    channel_in: int
    channel_out: int
    output_height: int
    output_width: int
    kernel_hw: int
    stride: int
    input_duplications: int


@dataclass(frozen=True)
class FastGemmPlan:
    common: CommonPlan
    n: int
    k: int
    np: int
    kp: int
    nd: int
    kd: int
    block_size: int
    blocks_per_partition: int
    packed_partitions: int
    shift: int
    shift_buffer: int
    grid_size: int
    input_replications: int


@dataclass(frozen=True)
class ShardingOffsetPlan:
    type: str
    scale: int


@dataclass(frozen=True)
class FastConvPlan:
    common: CommonPlan
    channel_in: int
    channel_out: int
    output_height: int
    output_width: int
    kernel_hw: int
    group: int
    stride: int
    input_size: int
    output_size: int
    num_slots: int
    num_grid: int
    num_block: int
    width_block: int
    width_block_data: int
    width_block_pad: int
    position_block: int
    capacity_block: int
    input_duplications: int
    blocking_outer_depth: int
    cyclic_roll: bool
    sharding_offset: ShardingOffsetPlan | None


VectorKernelPlan = Union[BaselineGemmPlan, BaselineConvPlan, FastGemmPlan, FastConvPlan]


@dataclass(frozen=True)
class RuntimePreparation:
    role: str
    source_operand: int
    kind: RuntimePreparationKind
    result_type: RankedTypePlan
    logical_input_size: int
    replications: int
    blocking_width: int
    rotation_candidates: tuple[int, ...]
    outer_block_depth: int


@dataclass(frozen=True)
class ScalarPreparation:
    role: str
    source_operand: int
    type: str
    scale: int


@dataclass(frozen=True, eq=False)
class ProviderResult:
    plan: VectorKernelPlan
    constants: tuple[TypedPayload, ...]
    runtime_preparations: tuple[RuntimePreparation, ...]
    scalar_preparations: tuple[ScalarPreparation, ...]
    provenance: str = "python"


_DTYPES = {
    "bool": np.dtype("?"),
    "s8": np.dtype("i1"),
    "s16": np.dtype("<i2"),
    "s32": np.dtype("<i4"),
    "s64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "u32": np.dtype("<u4"),
    "u64": np.dtype("<u8"),
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "c32": np.dtype("<c8"),
    "c64": np.dtype("<c16"),
}


def _primitive_for_dtype(dtype: np.dtype) -> str:
    normalized = np.dtype(dtype)
    for primitive, expected in _DTYPES.items():
        if (
            normalized.kind == expected.kind
            and normalized.itemsize == expected.itemsize
        ):
            return primitive
    raise PlanningError(f"unsupported vector-kernel dtype: {normalized}")


def _readonly_array(values, primitive: str | None = None, shape=None) -> np.ndarray:
    dtype = None if primitive is None else _DTYPES[primitive]
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if shape is not None:
        array = array.reshape(tuple(int(item) for item in shape))
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    array.setflags(write=False)
    return array


def _payload(role: str, values, primitive: str, shape=None) -> TypedPayload:
    array = _readonly_array(values, primitive, shape)
    return TypedPayload(role, array, canonical_payload_hash(array))


def _copy_payload(role: str, source: TypedPayload) -> TypedPayload:
    array = _readonly_array(source.values)
    return TypedPayload(role, array, canonical_payload_hash(array))


def canonical_payload_hash(values: np.ndarray) -> str:
    array = np.asarray(values)
    primitive = _primitive_for_dtype(array.dtype)
    shape = tuple(int(item) for item in array.shape)
    header = (
        "vector-kernel-constant:v1\n"
        f"element={primitive}\n"
        f"rank={len(shape)}\n"
        f"shape={','.join(str(item) for item in shape)}\n"
        "payload:\n"
    ).encode("ascii")
    if not array.flags.c_contiguous:
        raise PlanningError("canonical payload must be C-contiguous")
    if array.dtype.itemsize > 1 and (
        array.dtype.byteorder == ">"
        or (array.dtype.byteorder == "=" and not np.little_endian)
    ):
        raise PlanningError("canonical payload must be little-endian")
    return "sha256:" + hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _checked_add(lhs: int, rhs: int, message: str) -> int:
    if lhs < 0 or rhs < 0 or lhs > _I64_MAX - rhs:
        raise PlanningError(message)
    return lhs + rhs


def _checked_mul(lhs: int, rhs: int, message: str) -> int:
    if lhs < 0 or rhs < 0 or (rhs and lhs > _I64_MAX // rhs):
        raise PlanningError(message)
    return lhs * rhs


def _checked_signed_add(lhs: int, rhs: int, message: str) -> int:
    value = lhs + rhs
    if value < _I64_MIN or value > _I64_MAX:
        raise PlanningError(message)
    return value


def _checked_signed_mul(lhs: int, rhs: int, message: str) -> int:
    value = lhs * rhs
    if value < _I64_MIN or value > _I64_MAX:
        raise PlanningError(message)
    return value


def _checked_product(values: Iterable[int], message: str) -> int:
    product = 1
    for value in values:
        if value <= 0:
            raise PlanningError(message)
        product = _checked_mul(product, value, message)
    return product


def _i32(value: int, message: str) -> int:
    if value < _I32_MIN or value > _I32_MAX:
        raise PlanningError(message)
    return value


def _ceil_log2(value: int) -> int:
    return 0 if value <= 1 else (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    result = 1 << (value - 1).bit_length()
    if result > _I64_MAX:
        raise PlanningError("fast Gemm next-power-of-two overflows")
    return result


def _duplication_rotations(replications: int, logical_size: int) -> tuple[int, ...]:
    return tuple(
        _i32(
            -_checked_mul(index, logical_size, "duplication rotation overflows"),
            "duplication rotation exceeds s32",
        )
        for index in range(1, replications)
    )


def _range_rotations(count: int, scale: int, start: int = 0) -> tuple[int, ...]:
    return tuple(
        _i32(
            _checked_mul(index, scale, "rotation candidate overflows"),
            "rotation candidate exceeds s32",
        )
        for index in range(start, count)
    )


def _reduction(
    role: str, factor: int, block_width: int, padding: int
) -> tuple[RotationPlan | None, ReductionPlan | None]:
    if factor <= 1:
        return None, None
    stride = _checked_add(block_width, padding, "reduction stride overflows")
    kind = (
        ReductionKind.POWER_OF_TWO if _is_power_of_two(factor) else ReductionKind.LINEAR
    )
    indices = (
        (1 << index for index in range(_ceil_log2(factor)))
        if kind is ReductionKind.POWER_OF_TWO
        else range(1, factor)
    )
    candidates = tuple(
        _i32(
            _checked_mul(index, stride, "reduction rotation overflows"),
            "reduction rotation exceeds s32",
        )
        for index in indices
    )
    return (
        RotationPlan(role, candidates),
        ReductionPlan(role, kind, factor, block_width, padding),
    )


def _descriptor(payload: TypedPayload) -> ConstantPlan:
    return ConstantPlan(payload.role, payload.type, payload.content_hash)


def _integer_attribute(request: PlanningRequest, name: str) -> tuple[int, ...] | None:
    record = request.attribute(name)
    if record is None:
        return None
    if record.type.element_type not in {"s32", "s64", "u32"}:
        raise PlanningError(f"attribute {name!r} is not a canonical integer array")
    return tuple(int(value) for value in record.values.reshape(-1))


def _required_attribute(
    request: PlanningRequest, name: str, minimum_count: int
) -> tuple[int, ...]:
    values = _integer_attribute(request, name)
    if values is None or len(values) < minimum_count:
        raise PlanningError(f"Conv planning requires integer attribute {name!r}")
    return values


def _validate_gemm_request(request: PlanningRequest):
    if request.operation is not Operation.GEMM:
        raise PlanningError("Gemm provider received a non-Gemm request")
    if len(request.operand_types) != 3:
        raise PlanningError("Gemm planning requires exactly three operands")
    weight = request.source_constant("weight")
    bias = request.source_constant("bias")
    if (
        weight is None
        or bias is None
        or weight.type.element_type != "f32"
        or bias.type.element_type != "f32"
        or len(weight.type.shape) != 2
        or len(bias.type.shape) != 1
    ):
        raise PlanningError("Gemm planning requires owned f32 weight/bias payloads")
    if math.prod(weight.type.shape) != weight.values.size:
        raise PlanningError("Gemm weight payload does not match its shape")
    return weight, bias


def _request_mask_enabled(request: PlanningRequest) -> bool:
    return not request.options.mask_fuse or request.attribute("mask") is not None


def _build_baseline_gemm(request: PlanningRequest) -> ProviderResult:
    source_weight, source_bias = _validate_gemm_request(request)
    height, width = source_weight.type.shape
    slots = request.target.num_slots
    if height <= 0 or width <= 0 or height > _I32_MAX or width > _I32_MAX or slots <= 0:
        raise PlanningError("baseline Gemm dimensions are out of range")

    matrix = np.zeros((height, width), dtype="<f4")
    matrix[:, :] = source_weight.values.reshape(height, width)
    if width in (slots, slots // 2):
        padded_height = height
        while padded_height <= width and width % padded_height:
            padded_height += 1
        if padded_height > width:
            raise PlanningError("baseline Gemm height padding has no divisor")
        padded = np.zeros((padded_height, width), dtype="<f4")
        padded[:height, :] = matrix
        matrix = padded
        height = padded_height

    search_limit = _checked_mul(height, width, "baseline Gemm width search overflows")
    padded_width = width
    while padded_width <= search_limit and padded_width % height:
        padded_width += 1
    if padded_width > search_limit:
        raise PlanningError("baseline Gemm width padding has no solution")

    diagonal_count = _checked_mul(
        height, padded_width, "baseline Gemm diagonal payload is too large"
    )
    if diagonal_count > _I32_MAX:
        raise PlanningError("baseline Gemm diagonal payload is too large")
    diagonal = np.zeros((height, padded_width), dtype="<f4")
    for position in range(height):
        for index in range(padded_width):
            row = index % height
            column = (position + index) % padded_width
            if column < width:
                diagonal[position, index] = matrix[row, column]

    input_duplications = 1 if padded_width == slots else 2
    result_width = _checked_mul(
        input_duplications, padded_width, "baseline Gemm result width overflows"
    )
    payloads = [
        _payload("weight", diagonal, "f32", (height, padded_width)),
        _copy_payload("bias", source_bias),
    ]
    need_mask = _request_mask_enabled(request)
    if need_mask:
        payloads.append(_payload("mask", np.ones(height), "f32", (height,)))

    loops = [LoopPlan("gemm", 0, _i32(height, "baseline Gemm loop exceeds s32"), 1, 0)]
    rotations: list[RotationPlan] = []
    duplicate_rotations = _duplication_rotations(input_duplications, padded_width)
    if duplicate_rotations:
        rotations.append(RotationPlan("input-duplication", duplicate_rotations))
    rotations.append(RotationPlan("gemm", _range_rotations(height, 1)))
    reductions: list[ReductionPlan] = []
    factor = padded_width // height
    if factor > 1:
        loop_count = _ceil_log2(factor)
        loops.append(
            LoopPlan(
                "block-reduction",
                0,
                _i32(loop_count, "baseline Gemm reduction loop exceeds s32"),
                1,
                0,
            )
        )
        candidates = tuple(
            _i32(
                _checked_mul(
                    1 << index,
                    height,
                    "baseline Gemm reduction shift overflows",
                ),
                "baseline Gemm reduction shift exceeds s32",
            )
            for index in range(loop_count)
        )
        rotations.append(RotationPlan("block-reduction", candidates))
        reductions.append(
            ReductionPlan(
                "block-reduction", ReductionKind.POWER_OF_TWO, factor, height, 0
            )
        )

    common = CommonPlan(
        (request.operand_types[0],),
        (),
        tuple(map(_descriptor, payloads)),
        RankedTypePlan(request.operand_types[0].element_type, (result_width,)),
        tuple(loops),
        (SlicePlan("weight", AffineIndexPlan((1,), 0, False), padded_width),),
        tuple(rotations),
        tuple(reductions),
        MaskPlan(
            MaskPolicy.CLEAR_VALID_PREFIX if need_mask else MaskPolicy.NONE,
            height if need_mask else 0,
        ),
        SlotPlan(SlotPolicy.ABSENT_NATIVE_BASELINE, 0),
    )
    runtime = RuntimePreparation(
        "input",
        0,
        RuntimePreparationKind.PACKED_VECTOR,
        request.operand_types[0],
        padded_width,
        input_duplications,
        0,
        duplicate_rotations,
        0,
    )
    return ProviderResult(
        BaselineGemmPlan(common, height, padded_width, input_duplications),
        tuple(payloads),
        (runtime,),
        (),
    )


def _pad_gemm(n: int, k: int, slots: int) -> tuple[int, int]:
    if n <= 0 or k <= 0 or slots <= 0:
        raise PlanningError("fast Gemm padding requires positive dimensions")
    original_n, original_k = n, k
    if n > k:
        n, k = k, n
    if _is_power_of_two(k) and not _is_power_of_two(n):
        limit = _next_power_of_two(n)
        padded = n
        while padded <= limit and k % padded:
            padded += 1
        if padded <= limit:
            n = padded
    if k in (slots, slots // 2):
        limit = _checked_mul(n, k, "fast Gemm divisor search overflows")
        padded = n
        while padded <= limit and k % padded:
            padded += 1
        if padded > limit:
            raise PlanningError("fast Gemm divisor search failed")
        n = padded
    increase_n = (k - n % k) % k
    increase_k = (n - k % n) % n
    if increase_n <= increase_k:
        n = _checked_add(n, increase_n, "fast Gemm padded shape overflows")
    else:
        k = _checked_add(k, increase_k, "fast Gemm padded shape overflows")
    if original_n > original_k:
        n, k = k, n
    if slots // 2 < k < slots:
        k = slots
        while n <= slots and n % k and k % n:
            n += 1
    if not (k <= slots // 2 or k == slots) or not (n % k == 0 or k % n == 0):
        raise PlanningError("fast Gemm padded shape violates divisibility/slot rules")
    return n, k


@dataclass(frozen=True)
class _ImraCost:
    block_size: int = 1
    blocks_per_partition: int = 1
    packed_partitions: int = 1
    cost: int = 0


def _imra_cost(nd: int, kd: int, slots: int) -> _ImraCost:
    minimum = _checked_mul(nd, kd, "IMRA initial cost overflows")
    selected = _ImraCost(blocks_per_partition=nd)
    for block_size in range(1, nd + 1):
        if nd % block_size:
            continue
        blocks = nd // block_size
        for packed in range(1, blocks + 1):
            if blocks % packed:
                continue
            packed_width = _checked_mul(
                kd, packed, "IMRA capacity calculation overflows"
            )
            required = _checked_add(
                packed_width, nd, "IMRA capacity calculation overflows"
            )
            if (required > slots and kd != slots) or (kd == slots and packed > 1):
                continue
            replications = (required + kd - 1) // kd
            cost = (
                _ceil_log2(replications)
                + block_size
                - 1
                + blocks // packed
                - 1
                + packed
                - 1
            )
            if cost < minimum or (
                cost == minimum and packed > selected.packed_partitions
            ):
                minimum = cost
                selected = _ImraCost(block_size, blocks, packed, cost)
    return _ImraCost(
        selected.block_size,
        selected.blocks_per_partition,
        selected.packed_partitions,
        minimum,
    )


def _rotate_left(values: np.ndarray, amount: int) -> np.ndarray:
    if not len(values) or amount % len(values) == 0:
        return values.copy()
    amount %= len(values)
    return np.concatenate((values[amount:], values[:amount]))


def _rotate_right(values: np.ndarray, amount: int) -> np.ndarray:
    if not len(values) or amount % len(values) == 0:
        return values.copy()
    amount %= len(values)
    return np.concatenate((values[-amount:], values[:-amount]))


def _build_fast_gemm(request: PlanningRequest) -> ProviderResult:
    source_weight, source_bias = _validate_gemm_request(request)
    n, k = source_weight.type.shape
    npadded, kpadded = _pad_gemm(n, k, request.target.num_slots)
    nd, kd = min(npadded, kpadded), max(npadded, kpadded)
    if npadded > _I32_MAX or kpadded > _I32_MAX:
        raise PlanningError("fast Gemm padded dimensions exceed s32")
    _checked_mul(npadded, kpadded, "fast Gemm padded weight size overflows")
    padded = np.zeros((npadded, kpadded), dtype="<f4")
    padded[:n, :k] = source_weight.values.reshape(n, k)
    irma = np.zeros((nd, kd), dtype="<f4")
    for height in range(nd):
        row = (npadded - height) % npadded
        column = 0
        for width in range(kd):
            irma[height, width] = padded[row, column]
            row = (row + 1) % npadded
            column = (column + 1) % kpadded

    cost = _imra_cost(nd, kd, request.target.num_slots)
    shift = 1
    shift_buffer = nd // cost.packed_partitions
    actual_shift_buffer = (
        0
        if cost.packed_partitions == 1 and kd == request.target.num_slots
        else shift_buffer
    )
    packed_row_width = _checked_add(
        kd, actual_shift_buffer, "fast Gemm packed width exceeds slots"
    )
    packed_row_width = _checked_mul(
        cost.packed_partitions, packed_row_width, "fast Gemm packed width exceeds slots"
    )
    if packed_row_width > request.target.num_slots:
        raise PlanningError("fast Gemm packed width exceeds slots")
    grid_size = cost.blocks_per_partition // cost.packed_partitions
    packed_rows = []
    for grid in range(grid_size):
        for block in range(cost.block_size):
            base_row = grid * cost.block_size + block
            pieces = []
            for partition in range(cost.packed_partitions):
                row = base_row + partition * (nd // cost.packed_partitions)
                rotate_by = (row - base_row) + block % cost.block_size
                if rotate_by < 0 or rotate_by > kd:
                    raise PlanningError("fast Gemm IRMA rotation is out of range")
                rotated = _rotate_left(irma[row], rotate_by)
                if actual_shift_buffer:
                    rotated = np.concatenate((rotated, rotated[:actual_shift_buffer]))
                pieces.append(rotated)
            packed_rows.append(np.concatenate(pieces))
    packed_height = _checked_mul(
        grid_size, cost.block_size, "fast Gemm packed height overflows"
    )
    packed_values = np.stack(packed_rows).reshape(packed_height, packed_row_width)

    replications = (
        1
        if kpadded == request.target.num_slots
        else (cost.packed_partitions * (kd + actual_shift_buffer) + k - 1) // k
    )
    alignments = _range_rotations(cost.block_size, 1)
    payloads = [
        _payload("weight", packed_values, "f32", (packed_height, packed_row_width)),
        _copy_payload("bias", source_bias),
        _payload("rotation-table", alignments, "s32", (len(alignments),)),
    ]
    clear_mask = not request.options.mask_fuse or (
        not request.options.is_last_operation and n != request.target.num_slots
    )
    if clear_mask:
        payloads.append(_payload("mask", np.ones(n), "f32", (n,)))

    rotations: list[RotationPlan] = []
    duplicates = _duplication_rotations(replications, kpadded)
    if duplicates:
        rotations.append(RotationPlan("input-duplication", duplicates))
    rotations.extend(
        (
            RotationPlan("blocking-alignment", alignments),
            RotationPlan("grid", _range_rotations(grid_size, cost.block_size)),
        )
    )
    reductions: list[ReductionPlan] = []
    for role, factor, width, padding in (
        ("packed-partitions", cost.packed_partitions, kd, actual_shift_buffer),
        ("kp-over-np", kpadded // npadded, npadded, 0),
    ):
        rotation, reduction = _reduction(role, factor, width, padding)
        if rotation is not None:
            rotations.append(rotation)
            reductions.append(reduction)

    common = CommonPlan(
        (request.operand_types[0],),
        (),
        tuple(map(_descriptor, payloads)),
        RankedTypePlan(request.operand_types[0].element_type, (packed_row_width,)),
        (
            LoopPlan("grid", 0, _i32(grid_size, "fast Gemm grid exceeds s32"), 1, 0),
            LoopPlan(
                "block", 0, _i32(cost.block_size, "fast Gemm block exceeds s32"), 1, 1
            ),
        ),
        (
            SlicePlan(
                "weight",
                AffineIndexPlan((cost.block_size, 1), 0, False),
                packed_row_width,
            ),
        ),
        tuple(rotations),
        tuple(reductions),
        MaskPlan(
            MaskPolicy.CLEAR_VALID_PREFIX if clear_mask else MaskPolicy.NONE,
            n if clear_mask else 0,
        ),
        SlotPlan(SlotPolicy.LOGICAL_OUTPUT_ELEMENTS, n),
    )
    plan = FastGemmPlan(
        common,
        n,
        k,
        npadded,
        kpadded,
        nd,
        kd,
        cost.block_size,
        cost.blocks_per_partition,
        cost.packed_partitions,
        shift,
        actual_shift_buffer,
        grid_size,
        replications,
    )
    runtime = RuntimePreparation(
        "input",
        0,
        RuntimePreparationKind.BLOCKING_ROTATIONS,
        request.operand_types[0],
        kpadded,
        replications,
        cost.block_size,
        alignments,
        0,
    )
    return ProviderResult(plan, tuple(payloads), (runtime,), ())


@dataclass
class _ConvProblem:
    channel_in_source: int
    channel_in: int
    channel_in_kernel_source: int
    channel_in_kernel: int
    channel_out: int
    input_height: int
    input_width: int
    kernel_height: int
    kernel_width: int
    kernel_hw: int
    group: int
    stride: int
    original_stride: int
    padding: int
    fhe_padding: int
    input_size: int
    output_size: int
    position_size: int
    num_slots: int
    sharded: bool
    shard_x: int
    shard_y: int
    shard_z: int
    halo_size: int
    halo_diameter: int
    blocking_outer_depth: int
    sharding_offset_scale: int
    need_mask: bool
    runtime_input_type: RankedTypePlan
    result_type: RankedTypePlan
    source_weight: np.ndarray
    source_bias: np.ndarray


def _parse_conv_problem(request: PlanningRequest) -> _ConvProblem:
    if request.operation is not Operation.CONV:
        raise PlanningError("Conv provider received a non-Conv request")
    if len(request.operand_types) != 3:
        raise PlanningError("Conv planning requires exactly three operands")
    input_type, weight_type = request.operand_types[:2]
    if (
        input_type.element_type != "f32"
        or weight_type.element_type != "f32"
        or request.declared_result_type.element_type != "f32"
        or len(input_type.shape) != 4
        or len(weight_type.shape) != 4
    ):
        raise PlanningError(
            "Conv planning requires NCHW f32 input, weight, and result types"
        )
    if any(value <= 0 for value in (*input_type.shape, *weight_type.shape)):
        raise PlanningError(
            "Conv input or weight shape contains a nonpositive dimension"
        )
    if input_type.shape[0] != 1:
        raise PlanningError("Conv planning only supports batch size one")

    channel_in_source = input_type.shape[1]
    channel_in = channel_in_source
    channel_in_kernel_source = weight_type.shape[1]
    channel_in_kernel = channel_in_kernel_source
    channel_out = weight_type.shape[0]
    height, width = input_type.shape[2:]
    kernel_height, kernel_width = weight_type.shape[2:]
    if kernel_height != kernel_width:
        raise PlanningError("Conv planning only supports square kernels")
    kernel_hw = _checked_mul(
        kernel_height, kernel_width, "Conv dimensions overflow signed 64-bit arithmetic"
    )
    position_size = _checked_mul(
        height, width, "Conv dimensions overflow signed 64-bit arithmetic"
    )
    input_size = _checked_mul(
        channel_in, position_size, "Conv dimensions overflow signed 64-bit arithmetic"
    )
    output_size = _checked_mul(
        channel_out, position_size, "Conv dimensions overflow signed 64-bit arithmetic"
    )
    slots = request.target.num_slots
    if output_size >= (1 << 32) - 1 or slots <= 0:
        raise PlanningError("Conv input, output, or slot dimensions are out of range")

    group = _required_attribute(request, "group", 1)[0]
    strides = _required_attribute(request, "strides", 1)
    pads = _required_attribute(request, "pads", 4)
    if group <= 0 or strides[0] <= 0 or pads[0] < 0:
        raise PlanningError("Conv group, stride, or padding is invalid")
    if len(strides) > 1 and strides[0] != strides[1]:
        raise PlanningError("Conv planning requires equal spatial strides")
    stride = strides[0] if request.options.selective_strided_slice else 1
    original_stride = stride
    original_strides = _integer_attribute(request, "orig_strides") or ()
    if original_strides:
        if (
            len(original_strides) != 2
            or original_strides[0] != original_strides[1]
            or original_strides[0] <= 0
        ):
            raise PlanningError(
                "Conv planning requires two equal positive orig_strides values"
            )
        original_stride = original_strides[0]
    fhe_padding = 0
    keep_shape_pad = _integer_attribute(request, "keep_shape_pad") or ()
    if keep_shape_pad:
        if keep_shape_pad[0] < 0:
            raise PlanningError("Conv keep_shape_pad must be nonnegative")
        fhe_padding = keep_shape_pad[0]
    if group > 1:
        if channel_in != group:
            raise PlanningError(
                "depthwise Conv requires input channel count equal to group"
            )
    elif channel_in != channel_in_kernel_source:
        raise PlanningError(
            "non-grouped Conv requires matching input and weight channels"
        )

    sharded_values = _integer_attribute(request, "weight-sharded") or ()
    sharded = bool(sharded_values and sharded_values[0])
    shard_x = shard_y = shard_z = 1
    halo_size = halo_diameter = blocking_outer_depth = sharding_offset_scale = 0
    if sharded:
        shard_x, shard_y, shard_z, halo_size = _required_attribute(
            request, "sharding", 4
        )[:4]
        if shard_x <= 0 or shard_y <= 0 or shard_z <= 0 or halo_size < 0:
            raise PlanningError("Conv sharding dimensions or halo size are invalid")
        halo_diameter = _checked_mul(halo_size, 2, "Conv sharding halo size overflows")
        if halo_diameter > height:
            raise PlanningError("Conv sharding dimensions or halo size are invalid")
        blocking_outer_depth = 1
        if shard_z == 1:
            original_group = _required_attribute(request, "orig_group", 1)[0]
            if original_group > 1:
                shard_y = 1
                blocking_outer_depth = 0
        sharding_offset_scale = _checked_mul(
            channel_in_kernel_source, kernel_hw, "Conv sharding offset scale overflows"
        )

    source_weight = request.source_constant("weight")
    source_bias = request.source_constant("bias")
    if (
        source_weight is None
        or source_bias is None
        or source_weight.type.element_type != "f32"
        or source_bias.type.element_type != "f32"
    ):
        raise PlanningError("Conv planning requires owned f32 weight and bias payloads")
    local_count = _checked_product(
        (channel_out, channel_in_kernel_source, kernel_hw),
        "Conv local weight element count overflows",
    )
    expected_weight = local_count
    if sharded:
        expected_weight = _checked_mul(
            expected_weight, shard_x, "Conv sharded weight element count overflows"
        )
        expected_weight = _checked_mul(
            expected_weight, shard_y, "Conv sharded weight element count overflows"
        )
    if (
        source_weight.values.size != expected_weight
        or source_bias.values.size != channel_out
    ):
        raise PlanningError("Conv weight or bias payload does not match its shape")

    if channel_out >= channel_in and channel_out % channel_in:
        while channel_out % channel_in:
            if channel_in == _I64_MAX:
                raise PlanningError("Conv channel padding overflows")
            channel_in += 1
        channel_in_kernel = channel_in
        input_size = _checked_mul(
            channel_in, position_size, "Conv padded input size overflows"
        )
    runtime_input_size = _checked_mul(
        channel_in_source, position_size, "Conv runtime input size overflows"
    )
    runtime_input_type = RankedTypePlan(input_type.element_type, (runtime_input_size,))
    result_type = RankedTypePlan(
        input_type.element_type, (1, channel_out, height, width)
    )
    if result_type != request.declared_result_type:
        raise PlanningError(
            "Conv declared result type does not match native same-shape lowering"
        )
    need_mask = True
    if request.options.mask_fuse:
        mask = _integer_attribute(request, "mask")
        if mask is None:
            need_mask = False
        elif len(mask) != 1 or mask[0] <= 0:
            raise PlanningError("Conv mask attribute must contain one positive integer")

    return _ConvProblem(
        channel_in_source,
        channel_in,
        channel_in_kernel_source,
        channel_in_kernel,
        channel_out,
        height,
        width,
        kernel_height,
        kernel_width,
        kernel_hw,
        group,
        stride,
        original_stride,
        pads[0],
        fhe_padding,
        input_size,
        output_size,
        position_size,
        slots,
        sharded,
        shard_x,
        shard_y,
        shard_z,
        halo_size,
        halo_diameter,
        blocking_outer_depth,
        sharding_offset_scale,
        need_mask,
        runtime_input_type,
        result_type,
        _readonly_array(source_weight.values),
        _readonly_array(source_bias.values),
    )


def _conv_alignment(problem: _ConvProblem) -> list[int]:
    rotations = [0] * problem.kernel_hw
    center = (problem.kernel_height - 1) // 2
    if problem.input_height != problem.input_width:
        for row in range(problem.kernel_height):
            for column in range(problem.kernel_width):
                value = _checked_signed_mul(
                    row - center,
                    problem.input_width,
                    "Conv rotation alignment arithmetic overflows",
                )
                value = _checked_signed_add(
                    value,
                    column - center,
                    "Conv rotation alignment arithmetic overflows",
                )
                rotations[row * problem.kernel_height + column] = value
        return rotations
    edge = _checked_mul(
        problem.input_height, center, "Conv rotation alignment arithmetic overflows"
    )
    edge = _checked_add(edge, center, "Conv rotation alignment arithmetic overflows")
    rotations[0] = -edge
    rotations[-1] = edge
    rotations[problem.kernel_hw // 2] = 0
    if problem.kernel_height > 1:
        rotations[problem.kernel_hw // 2 + 1] = 1
        rotations[problem.kernel_hw // 2 - 1] = -1
    for index in range(1, problem.kernel_hw // 2 - 1):
        value = _checked_signed_add(
            rotations[index - 1], 1, "Conv rotation alignment arithmetic overflows"
        )
        if problem.kernel_height > 3 and index % problem.kernel_height == 0:
            value = _checked_signed_add(
                value,
                problem.input_height - problem.kernel_height,
                "Conv rotation alignment arithmetic overflows",
            )
        rotations[index] = value
    for index in range(problem.kernel_hw - 1, problem.kernel_hw // 2 + 2, -1):
        value = _checked_signed_add(
            rotations[index], -1, "Conv rotation alignment arithmetic overflows"
        )
        if problem.kernel_height > 3 and index % problem.kernel_height == 0:
            value = _checked_signed_add(
                value,
                problem.kernel_height - problem.input_height,
                "Conv rotation alignment arithmetic overflows",
            )
        rotations[index - 1] = value
    return rotations


def _build_im2col(
    weight: np.ndarray,
    channel_in: int,
    height: int,
    width: int,
    channel_out: int,
    kernel_height: int,
    kernel_width: int,
    stride: int,
) -> tuple[list[int], np.ndarray]:
    kernel_hw = _checked_mul(
        kernel_height, kernel_width, "Conv im2col dimensions overflow"
    )
    plane = _checked_mul(height, width, "Conv im2col dimensions overflow")
    output_size = _checked_mul(channel_out, plane, "Conv im2col dimensions overflow")
    row_count = _checked_mul(channel_in, kernel_hw, "Conv im2col dimensions overflow")
    _checked_mul(row_count, output_size, "Conv im2col dimensions overflow")
    if weight.size < channel_out * row_count:
        raise PlanningError("Conv im2col source weight is too small")
    proxy = _ConvProblem(
        channel_in,
        channel_in,
        channel_in,
        channel_in,
        channel_out,
        height,
        width,
        kernel_height,
        kernel_width,
        kernel_hw,
        1,
        stride,
        stride,
        0,
        0,
        channel_in * plane,
        output_size,
        plane,
        output_size,
        False,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        True,
        RankedTypePlan("f32", (channel_in * plane,)),
        RankedTypePlan("f32", (1, channel_out, height, width)),
        weight,
        np.empty(0, dtype="<f4"),
    )
    rotations = _conv_alignment(proxy)
    scaled = [
        _checked_signed_mul(value, stride, "Conv stride-scaled rotation overflows")
        for value in rotations
    ]
    matrix = np.zeros((row_count, output_size), dtype="<f4")
    pad = (kernel_height - 1) // 2
    grid_h, grid_w = np.indices((height, width))
    position = (grid_h * width + grid_w).reshape(-1)
    masks = []
    for kernel_index in range(kernel_hw):
        kernel_row, kernel_column = divmod(kernel_index, kernel_width)
        valid = (
            (grid_h + kernel_row >= pad)
            & (grid_w + kernel_column >= pad)
            & (grid_h + kernel_row < height + pad)
            & (grid_w + kernel_column < width + pad)
        ).reshape(-1)
        rotation = scaled[kernel_index]
        if stride > 1:
            if rotation > 0:
                if rotation > plane:
                    valid[:] = False
                else:
                    valid &= position < plane - rotation
            elif rotation < 0:
                valid &= position >= -rotation
        masks.append(valid)
    flat_weight = weight.reshape(-1)
    for output_channel in range(channel_out):
        output_slice = slice(output_channel * plane, (output_channel + 1) * plane)
        for row in range(row_count):
            kernel_index = row % kernel_hw
            source_row = (row + output_channel * kernel_hw) % row_count
            scalar = flat_weight[output_channel * row_count + source_row]
            target = matrix[row, output_slice]
            target[masks[kernel_index]] = scalar
    return rotations, matrix


def _spatial_mask(problem: _ConvProblem) -> np.ndarray:
    height, width = problem.input_height, problem.input_width
    rows, columns = np.indices((height, width))
    if (problem.original_stride > 1 and problem.padding != 0) or (
        problem.original_stride > 1
        and problem.padding == 0
        and problem.kernel_width == 1
    ):
        return ((rows % problem.original_stride) == 0) & (
            (columns % problem.original_stride) == 0
        )
    if problem.padding != 0 or problem.kernel_width == 1:
        return np.ones((height, width), dtype=bool)
    pad = problem.fhe_padding or (problem.kernel_height - 1) // 2
    valid = (
        (rows >= pad)
        & (rows < height - pad)
        & (columns >= pad)
        & (columns < width - pad)
    )
    if problem.original_stride > 1:
        stride_valid = np.ones((height, width), dtype=bool)
        eligible = rows >= pad
        stride_valid[eligible] = (
            (rows[eligible] - pad) % problem.original_stride == 0
        ) & (
            (columns[eligible] < pad)
            | ((columns[eligible] - pad) % problem.original_stride == 0)
        )
        valid &= stride_valid
    return valid


def _mask_matrix(problem: _ConvProblem, matrix: np.ndarray) -> None:
    valid = _spatial_mask(problem).reshape(-1)
    tiled = np.tile(valid, problem.channel_out)
    matrix[:, ~tiled] = np.float32(0.0)


def _expanded_bias(problem: _ConvProblem) -> np.ndarray:
    bias = np.repeat(problem.source_bias.reshape(-1), problem.position_size).astype(
        "<f4", copy=False
    )
    valid = _spatial_mask(problem).reshape(-1)
    bias.reshape(problem.channel_out, problem.position_size)[:, ~valid] = np.float32(
        0.0
    )
    return bias


def _local_im2col(problem: _ConvProblem) -> tuple[list[int], np.ndarray]:
    local_count = _checked_product(
        (problem.channel_out, problem.channel_in_kernel_source, problem.kernel_hw),
        "Conv local weight size overflows",
    )
    local = problem.source_weight.reshape(-1)[:local_count].copy()
    if problem.channel_in != problem.channel_in_source:
        padded = np.zeros(
            (problem.channel_out, problem.channel_in, problem.kernel_hw),
            dtype="<f4",
        )
        padded[:, : problem.channel_in_source, :] = local.reshape(
            problem.channel_out, problem.channel_in_source, problem.kernel_hw
        )
        local = padded.reshape(-1)
    rotations, matrix = _build_im2col(
        local,
        problem.channel_in_kernel,
        problem.input_height,
        problem.input_width,
        problem.channel_out,
        problem.kernel_height,
        problem.kernel_width,
        problem.stride,
    )
    _mask_matrix(problem, matrix)
    return rotations, matrix


def _mini_factor(channels: int) -> int:
    if channels < 32 or not _is_power_of_two(channels):
        return 1
    minimum_sum, factor = channels, 1
    candidate = 2
    while candidate <= channels // candidate:
        if (
            channels % candidate == 0
            and candidate + channels // candidate < minimum_sum
        ):
            minimum_sum = candidate + channels // candidate
            factor = candidate
        candidate += 1
    return factor


def _fusion_blocks(problem: _ConvProblem) -> int:
    blocks = 1
    if (
        _is_power_of_two(problem.output_size)
        and problem.channel_out % problem.channel_in == 0
        and problem.kernel_hw > 1
        and problem.group == 1
        and problem.num_slots // problem.output_size >= 4
        and problem.channel_out >= problem.channel_in
    ):
        blocks = min(problem.num_slots // problem.output_size // 2, problem.channel_in)
        if (
            blocks >= 2
            and problem.channel_in // blocks + 2 * blocks >= problem.channel_in
        ):
            blocks //= 2
        blocks = min(blocks, 8)
    return blocks


def _build_baseline_conv(
    problem: _ConvProblem, matrix: np.ndarray, unscaled_rotations: Sequence[int]
) -> ProviderResult:
    if problem.sharded:
        raise PlanningError(
            "baseline Conv does not support a runtime sharded weight offset"
        )
    _i32(problem.channel_in, "baseline Conv loop bounds exceed signed 32-bit range")
    _i32(problem.kernel_hw, "baseline Conv loop bounds exceed signed 32-bit range")
    weight_rows = _checked_mul(
        problem.channel_in_kernel,
        problem.kernel_hw,
        "baseline Conv packed weight size overflows",
    )
    weight_count = _checked_mul(
        weight_rows, problem.output_size, "baseline Conv packed weight size overflows"
    )
    weight = np.ascontiguousarray(matrix.reshape(-1), dtype="<f4")
    if weight.size != weight_count:
        raise PlanningError("baseline Conv packed weight size is inconsistent")
    bias = _expanded_bias(problem)
    numerator = (
        _checked_add(
            problem.channel_out,
            problem.channel_in,
            "baseline Conv duplication count overflows",
        )
        - 1
    )
    duplications = numerator // problem.channel_in
    duplications = _checked_add(
        duplications, 1, "baseline Conv duplication count overflows"
    )
    if _is_power_of_two(problem.input_size):
        duplications = min(duplications, problem.num_slots // problem.input_size)
    if (
        _checked_mul(
            problem.input_size, 2, "baseline Conv input duplication size overflows"
        )
        > problem.num_slots
    ):
        duplications = 1
    packed_input = _checked_mul(
        problem.input_size,
        duplications,
        "baseline Conv duplicated input exceeds slot capacity",
    )
    if duplications <= 0 or packed_input > problem.num_slots:
        raise PlanningError("baseline Conv duplicated input exceeds slot capacity")
    duplicate_rotations = _duplication_rotations(duplications, problem.input_size)
    kernel_rotations = tuple(
        _i32(
            _checked_signed_mul(
                value,
                problem.stride,
                "Conv scaled rotation overflows signed 64-bit arithmetic",
            ),
            "Conv rotation candidate exceeds signed 32-bit range",
        )
        for value in unscaled_rotations
    )
    channel_rotations = (
        _i32(
            problem.position_size, "Conv rotation candidate exceeds signed 32-bit range"
        ),
    )
    payloads = [
        _payload("weight", weight, "f32", (weight_rows, problem.output_size)),
        _payload("bias", bias, "f32", (problem.output_size,)),
        _payload("rotation-table", kernel_rotations, "s32", (problem.kernel_hw,)),
    ]
    rotations = []
    if duplicate_rotations:
        rotations.append(RotationPlan("input-duplication", duplicate_rotations))
    rotations.extend(
        (
            RotationPlan("kernel-alignment", kernel_rotations),
            RotationPlan("channel-step", channel_rotations),
        )
    )
    common = CommonPlan(
        (problem.runtime_input_type,),
        (),
        tuple(map(_descriptor, payloads)),
        problem.result_type,
        (
            LoopPlan("channel-in", 0, problem.channel_in, 1, 0),
            LoopPlan("kernel-hw", 0, problem.kernel_hw, 1, 1),
        ),
        (
            SlicePlan(
                "weight",
                AffineIndexPlan((problem.kernel_hw, 1), 0, False),
                problem.output_size,
            ),
        ),
        tuple(rotations),
        (),
        MaskPlan(MaskPolicy.NONE, 0),
        SlotPlan(SlotPolicy.LOGICAL_OUTPUT_ELEMENTS, problem.output_size),
    )
    runtime = RuntimePreparation(
        "input",
        0,
        RuntimePreparationKind.FLATTEN_PACKED_VECTOR,
        problem.runtime_input_type,
        problem.input_size,
        duplications,
        0,
        duplicate_rotations,
        0,
    )
    plan = BaselineConvPlan(
        common,
        problem.channel_in,
        problem.channel_out,
        problem.input_height,
        problem.input_width,
        problem.kernel_hw,
        problem.stride,
        duplications,
    )
    return ProviderResult(plan, tuple(payloads), (runtime,), ())


def _build_sharded_weight(problem: _ConvProblem) -> tuple[np.ndarray, tuple[int, int]]:
    shard_count = _checked_mul(
        problem.shard_x, problem.shard_y, "Conv sharded global dimensions overflow"
    )
    global_channel_out = _checked_mul(
        problem.channel_out, problem.shard_x, "Conv sharded global dimensions overflow"
    )
    global_channel_in = _checked_mul(
        problem.channel_in_kernel_source,
        problem.shard_y,
        "Conv sharded global dimensions overflow",
    )
    expected = _checked_product(
        (global_channel_out, global_channel_in, problem.kernel_hw),
        "Conv sharded source payload does not match global tensor dimensions",
    )
    if problem.source_weight.size != expected:
        raise PlanningError(
            "Conv sharded source payload does not match global tensor dimensions"
        )
    result_rows = _checked_product(
        (problem.channel_in_kernel_source, problem.kernel_hw, shard_count),
        "Conv sharded prepared weight size overflows",
    )
    row_width = _checked_mul(
        problem.channel_out,
        problem.position_size,
        "Conv sharded prepared weight size overflows",
    )
    result_count = _checked_mul(
        result_rows, row_width, "Conv sharded prepared weight size overflows"
    )
    pieces = []
    flat = problem.source_weight.reshape(-1)
    for shard_x in range(problem.shard_x):
        for shard_y in range(problem.shard_y):
            local = np.empty(
                (
                    problem.channel_out,
                    problem.channel_in_kernel_source,
                    problem.kernel_hw,
                ),
                dtype="<f4",
            )
            for output in range(problem.channel_out):
                global_output = shard_x * problem.channel_out + output
                for input_channel in range(problem.channel_in_kernel_source):
                    global_input = (
                        shard_y * problem.channel_in_kernel_source + input_channel
                    )
                    start = (
                        global_output * global_channel_in + global_input
                    ) * problem.kernel_hw
                    local[output, input_channel, :] = flat[
                        start : start + problem.kernel_hw
                    ]
            _, matrix = _build_im2col(
                local.reshape(-1),
                problem.channel_in_kernel_source,
                problem.input_height,
                problem.input_width,
                problem.channel_out,
                problem.kernel_height,
                problem.kernel_width,
                1,
            )
            overlap = _checked_mul(
                problem.halo_size,
                problem.input_width,
                "Conv sharded overlap size overflows",
            )
            if overlap > problem.position_size:
                raise PlanningError("Conv sharded overlap size overflows")
            for input_channel in range(problem.channel_in_kernel_source):
                for kernel in range(problem.kernel_hw):
                    row = matrix[input_channel * problem.kernel_hw + kernel].copy()
                    for output in range(problem.channel_out):
                        base = output * problem.position_size
                        row[base : base + overlap] = np.float32(0.0)
                        row[
                            base + problem.position_size - overlap : base
                            + problem.position_size
                        ] = np.float32(0.0)
                        for h in range(problem.input_height - problem.halo_diameter):
                            for w in range(problem.input_width):
                                if (
                                    h % problem.original_stride
                                    or w % problem.original_stride
                                ):
                                    row[
                                        base
                                        + (problem.halo_size + h) * problem.input_width
                                        + w
                                    ] = np.float32(0.0)
                    shift = (
                        input_channel % problem.channel_out
                    ) * problem.position_size
                    pieces.append(_rotate_right(row, shift))
    prepared = np.concatenate(pieces) if pieces else np.empty(0, dtype="<f4")
    if prepared.size != result_count:
        raise PlanningError("Conv sharded prepared weight has inconsistent size")
    return prepared.reshape(result_rows, row_width), (result_rows, row_width)


def _build_fast_conv(
    request: PlanningRequest,
    problem: _ConvProblem,
    matrix: np.ndarray,
    unscaled_rotations: list[int],
) -> ProviderResult:
    if problem.channel_out < problem.channel_in:
        raise PlanningError("fast Conv requires channel_out >= effective channel_in")
    num_block = _fusion_blocks(problem) if request.options.conv_parallel else 1
    if num_block <= 0 or problem.channel_in_kernel % num_block:
        raise PlanningError("fast Conv block count does not divide input channels")
    width_block_pad = 0 if num_block == 1 else problem.input_size // num_block
    num_grid = problem.channel_in_kernel
    capacity_block = problem.kernel_hw
    position_block = problem.position_size
    input_duplications = problem.channel_out // problem.channel_in

    if problem.sharded:
        prepared_weight, prepared_shape = _build_sharded_weight(problem)
    elif problem.kernel_hw == 1:
        capacity_block = _mini_factor(problem.channel_in_kernel)
        if capacity_block <= 0 or problem.channel_in_kernel % capacity_block:
            raise PlanningError("fast 1x1 Conv capacity does not divide input channels")
        rows = []
        for input_channel in range(problem.channel_in_kernel):
            shift = _checked_mul(
                capacity_block,
                input_channel // capacity_block,
                "fast 1x1 Conv row alignment is out of range",
            )
            shift = _checked_mul(
                shift,
                problem.position_size,
                "fast 1x1 Conv row alignment is out of range",
            )
            if shift > matrix.shape[1]:
                raise PlanningError("fast 1x1 Conv row alignment is out of range")
            rows.append(_rotate_right(matrix[input_channel], shift))
        for index in range(1, capacity_block):
            unscaled_rotations.append(
                _checked_mul(
                    index, problem.position_size, "fast 1x1 Conv alignment overflows"
                )
            )
        if capacity_block > 1:
            input_duplications = _checked_add(
                input_duplications, 1, "fast 1x1 Conv input duplication count overflows"
            )
        num_grid //= capacity_block
        position_block = _checked_mul(
            position_block, capacity_block, "fast 1x1 Conv position block overflows"
        )
        prepared_weight = np.stack(rows)
        prepared_shape = (problem.channel_in_kernel, problem.output_size)
    else:
        offset = (
            _checked_mul(
                problem.channel_in_kernel,
                problem.kernel_hw,
                "fast Conv block offset overflows",
            )
            // num_block
        )
        block_pad = np.zeros(width_block_pad, dtype="<f4")
        rows = []
        for input_channel in range(problem.channel_in_kernel // num_block):
            for kernel in range(problem.kernel_hw):
                pieces = []
                for block in range(num_block):
                    index = input_channel * problem.kernel_hw + kernel + block * offset
                    if index < 0 or index >= len(matrix):
                        raise PlanningError(
                            "fast Conv packed weight index is out of range"
                        )
                    shift = _checked_mul(
                        input_channel,
                        problem.position_size,
                        "fast Conv row alignment is out of range",
                    )
                    if shift > matrix.shape[1]:
                        raise PlanningError("fast Conv row alignment is out of range")
                    pieces.append(_rotate_right(matrix[index], shift))
                    if num_block > 1:
                        pieces.append(block_pad)
                rows.append(np.concatenate(pieces))
        if num_block > 1:
            input_duplications = _checked_mul(
                input_duplications,
                _checked_add(
                    num_block, 1, "fast Conv input duplication count overflows"
                ),
                "fast Conv input duplication count overflows",
            )
            num_grid //= num_block
        packed_width = _checked_mul(
            num_block,
            _checked_add(
                problem.output_size,
                width_block_pad,
                "fast Conv packed weight width overflows",
            ),
            "fast Conv packed weight width overflows",
        )
        prepared_weight = np.stack(rows)
        prepared_shape = (offset, packed_width)
    prepared_weight = np.asarray(prepared_weight, dtype="<f4").reshape(prepared_shape)

    width_block = _checked_add(
        problem.output_size, width_block_pad, "fast Conv width block overflows"
    )
    if (
        num_grid <= 0
        or capacity_block <= 0
        or input_duplications <= 0
        or num_grid > _I32_MAX
        or capacity_block > _I32_MAX
    ):
        raise PlanningError("fast Conv loop or duplication bounds are out of range")
    slice_width = _checked_mul(
        num_block, width_block, "fast Conv slice width overflows"
    )
    blocking_rotations = tuple(
        _i32(value, "Conv rotation candidate exceeds signed 32-bit range")
        for value in unscaled_rotations
    )
    if len(blocking_rotations) != capacity_block:
        raise PlanningError("fast Conv alignment count does not equal capacity block")
    bias = _expanded_bias(problem)
    cyclic = (
        num_block == 1
        and problem.output_size > problem.num_slots // 2
        and problem.output_size < problem.num_slots
        and problem.group != problem.channel_in
    )
    collective = (
        problem.channel_in > 1 and problem.group != problem.channel_in and not cyclic
    ) or problem.num_slots == problem.output_size
    payloads = [
        _payload("weight", prepared_weight, "f32", prepared_shape),
        _payload("bias", bias, "f32", (problem.output_size,)),
        _payload("rotation-table", blocking_rotations, "s32", (capacity_block,)),
    ]
    loops = [
        LoopPlan("grid", 0, num_grid, 1, 0),
        LoopPlan("capacity-block", 0, capacity_block, 1, 1),
    ]
    slices = [
        SlicePlan(
            "weight",
            AffineIndexPlan((capacity_block, 1), 0, problem.sharded),
            slice_width,
        )
    ]
    rotations = [RotationPlan("blocking-alignment", blocking_rotations)]
    reductions: list[ReductionPlan] = []
    if cyclic:
        left_mask = np.zeros((num_grid, problem.output_size), dtype="<f4")
        right_mask = np.ones((num_grid, problem.output_size), dtype="<f4")
        left_rotations, right_rotations = [], []
        for grid in range(num_grid):
            prefix = _checked_mul(
                grid, position_block, "fast Conv cyclic prefix is out of range"
            )
            if prefix > problem.output_size:
                raise PlanningError("fast Conv cyclic prefix is out of range")
            left_mask[grid, :prefix] = np.float32(1.0)
            right_mask[grid, :prefix] = np.float32(0.0)
            left_rotations.append(
                _i32(
                    prefix - problem.output_size,
                    "Conv rotation candidate exceeds signed 32-bit range",
                )
            )
            right_rotations.append(
                _i32(prefix, "Conv rotation candidate exceeds signed 32-bit range")
            )
        payloads.extend(
            (
                _payload("cyclic-mask-left", left_mask, "f32", left_mask.shape),
                _payload("cyclic-mask-right", right_mask, "f32", right_mask.shape),
            )
        )
        slices.extend(
            (
                SlicePlan(
                    "cyclic-mask-left",
                    AffineIndexPlan((1,), 0, False),
                    problem.output_size,
                ),
                SlicePlan(
                    "cyclic-mask-right",
                    AffineIndexPlan((1,), 0, False),
                    problem.output_size,
                ),
            )
        )
        rotations.extend(
            (
                RotationPlan("cyclic-left", tuple(left_rotations)),
                RotationPlan("cyclic-right", tuple(right_rotations)),
            )
        )
    else:
        rotations.append(
            RotationPlan("grid", _range_rotations(num_grid, position_block))
        )

    mask_policy, mask_valid_length = MaskPolicy.NONE, 0
    if collective:
        single_overload = (
            not request.options.conv_parallel
            and not request.options.sharding
            and num_block == 1
        )
        collective_padding = width_block_pad
        if (
            not single_overload
            and num_block == 1
            and problem.num_slots != problem.output_size
        ):
            collective_padding = _checked_mul(
                problem.channel_in - 1,
                problem.position_size,
                "fast Conv collective padding overflows",
            )
        reductions.append(
            ReductionPlan(
                "collective",
                ReductionKind.COLLECTIVE_SINGLE_BLOCK
                if single_overload
                else ReductionKind.COLLECTIVE_BLOCKS,
                num_block,
                problem.output_size,
                collective_padding,
            )
        )
        if problem.need_mask:
            mask_policy, mask_valid_length = (
                MaskPolicy.COLLECTIVE_REDUCTION,
                problem.output_size,
            )
        mask = np.ones(problem.output_size, dtype="<f4")
        if problem.num_slots == problem.output_size:
            if problem.need_mask:
                payloads.append(
                    _payload("collective-mask", mask, "f32", (problem.output_size,))
                )
        elif single_overload:
            rotations.append(RotationPlan("collective-tail", (-problem.output_size,)))
            if problem.need_mask:
                payloads.append(
                    _payload("collective-mask", mask, "f32", (problem.output_size,))
                )
        else:
            if collective_padding > problem.output_size:
                raise PlanningError("fast Conv collective padding exceeds output size")
            gap = np.zeros(problem.output_size, dtype="<f4")
            if collective_padding:
                gap[-collective_padding:] = np.float32(1.0)
            payloads.extend(
                (
                    _payload("collective-mask", mask, "f32", (problem.output_size,)),
                    _payload("collective-gap-mask", gap, "f32", (problem.output_size,)),
                )
            )
            if num_block > 1:
                loops.extend(
                    (
                        LoopPlan("collective-data", 1, num_block, 1, 0),
                        LoopPlan("collective-gap", 0, num_block - 1, 1, 0),
                    )
                )
                rotations.append(
                    RotationPlan(
                        "collective-data", _range_rotations(num_block, width_block, 1)
                    )
                )
                rotations.append(
                    RotationPlan(
                        "collective-gap",
                        tuple(
                            _i32(
                                _checked_add(
                                    _checked_mul(
                                        block,
                                        width_block,
                                        "fast Conv collective rotation overflows",
                                    ),
                                    collective_padding,
                                    "fast Conv collective rotation overflows",
                                ),
                                "Conv rotation candidate exceeds signed 32-bit range",
                            )
                            for block in range(num_block - 1)
                        ),
                    )
                )
            rotations.append(RotationPlan("collective-tail", (-problem.output_size,)))

    scalar_inputs: tuple[str, ...] = ()
    sharding_offset = None
    scalar_preparations: tuple[ScalarPreparation, ...] = ()
    if problem.sharded:
        scalar_inputs = ("s32",)
        sharding_offset = ShardingOffsetPlan("s32", problem.sharding_offset_scale)
        scalar_preparations = (
            ScalarPreparation("weight-offset", 1, "s32", problem.sharding_offset_scale),
        )
    common = CommonPlan(
        (problem.runtime_input_type,),
        scalar_inputs,
        tuple(map(_descriptor, payloads)),
        problem.result_type,
        tuple(loops),
        tuple(slices),
        tuple(rotations),
        tuple(reductions),
        MaskPlan(mask_policy, mask_valid_length),
        SlotPlan(SlotPolicy.LOGICAL_OUTPUT_ELEMENTS, problem.output_size),
    )
    plan = FastConvPlan(
        common,
        problem.channel_in,
        problem.channel_out,
        problem.input_height,
        problem.input_width,
        problem.kernel_hw,
        problem.group,
        problem.stride,
        problem.input_size,
        problem.output_size,
        problem.num_slots,
        num_grid,
        num_block,
        width_block,
        problem.output_size,
        width_block_pad,
        position_block,
        capacity_block,
        input_duplications,
        problem.blocking_outer_depth,
        cyclic,
        sharding_offset,
    )
    runtime = RuntimePreparation(
        "input",
        0,
        RuntimePreparationKind.BLOCKING_ROTATIONS,
        problem.runtime_input_type,
        problem.input_size,
        input_duplications,
        capacity_block,
        blocking_rotations,
        problem.blocking_outer_depth,
    )
    return ProviderResult(plan, tuple(payloads), (runtime,), scalar_preparations)


def plan_vector_kernel(
    request: PlanningRequest | _PlanningRequestView,
) -> ProviderResult:
    """Plan one request with the Python reference provider."""
    operation = Operation(request.operation)
    requested = RequestedPlanKind(request.requested_plan_kind)
    if operation is Operation.GEMM:
        if requested in (RequestedPlanKind.BASELINE_CONV, RequestedPlanKind.FAST_CONV):
            raise PlanningError("Conv plan kind cannot plan a Gemm operation")
        if requested is RequestedPlanKind.BASELINE_GEMM:
            return _build_baseline_gemm(request)
        return _build_fast_gemm(request)
    if requested in (RequestedPlanKind.BASELINE_GEMM, RequestedPlanKind.FAST_GEMM):
        raise PlanningError("Gemm plan kind cannot plan a Conv operation")
    problem = _parse_conv_problem(request)
    rotations, matrix = _local_im2col(problem)
    use_fast = requested is RequestedPlanKind.FAST_CONV or (
        requested is RequestedPlanKind.AUTO
        and problem.channel_out >= problem.channel_in
    )
    return (
        _build_fast_conv(request, problem, matrix, rotations)
        if use_fast
        else _build_baseline_conv(problem, matrix, rotations)
    )


def _type_from_binding(data: Mapping) -> RankedTypePlan:
    return RankedTypePlan(
        str(data["element_type"]), tuple(int(value) for value in data["shape"])
    )


def _planning_request_from_binding(data: Mapping) -> _PlanningRequestView:
    """Create the callback-scoped immutable request used by the pybind adapter."""
    attributes = tuple(
        AttributeRecord(str(item["name"]), item["values"])
        for item in data["attributes"]
    )
    constants = tuple(
        TypedPayload(str(item["role"]), item["values"], str(item["content_hash"]))
        for item in data["source_constants"]
    )
    request = PlanningRequest(
        Operation(data["operation"]),
        attributes,
        tuple(_type_from_binding(item) for item in data["operand_types"]),
        _type_from_binding(data["declared_result_type"]),
        OptionSnapshot(**dict(data["options"])),
        TargetSnapshot(**dict(data["target"])),
        constants,
        RequestedPlanKind(data["requested_plan_kind"]),
        tuple(str(item) for item in data["runtime_scalar_types"]),
    )
    return _PlanningRequestView(request)


def _type_binding(value: RankedTypePlan) -> dict:
    return {"element_type": value.element_type, "shape": value.shape}


def _common_binding(common: CommonPlan) -> dict:
    return {
        "runtime_vector_inputs": tuple(
            map(_type_binding, common.runtime_vector_inputs)
        ),
        "runtime_scalar_inputs": common.runtime_scalar_inputs,
        "constants": tuple(
            {
                "role": item.role,
                "type": _type_binding(item.type),
                "content_hash": item.content_hash,
            }
            for item in common.constants
        ),
        "result_type": _type_binding(common.result_type),
        "loops": tuple(vars(item) for item in common.loops),
        "slices": tuple(
            {
                "role": item.role,
                "index": {
                    "iv_coefficients": item.index.iv_coefficients,
                    "constant": item.index.constant,
                    "uses_sharding_offset": item.index.uses_sharding_offset,
                },
                "width": item.width,
            }
            for item in common.slices
        ),
        "rotations": tuple(
            {"role": item.role, "candidates": item.candidates}
            for item in common.rotations
        ),
        "reductions": tuple(
            {
                "role": item.role,
                "kind": item.kind.value,
                "factor": item.factor,
                "block_width": item.block_width,
                "padding": item.padding,
            }
            for item in common.reductions
        ),
        "mask": {
            "policy": common.mask.policy.value,
            "valid_length": common.mask.valid_length,
        },
        "slot": {"policy": common.slot.policy.value, "value": common.slot.value},
    }


def _plan_binding(plan: VectorKernelPlan) -> dict:
    result = {"common": _common_binding(plan.common)}
    if isinstance(plan, BaselineGemmPlan):
        result.update(
            kind="baseline-gemm",
            height=plan.height,
            width=plan.width,
            input_duplications=plan.input_duplications,
        )
    elif isinstance(plan, BaselineConvPlan):
        result.update(
            kind="baseline-conv",
            channel_in=plan.channel_in,
            channel_out=plan.channel_out,
            output_height=plan.output_height,
            output_width=plan.output_width,
            kernel_hw=plan.kernel_hw,
            stride=plan.stride,
            input_duplications=plan.input_duplications,
        )
    elif isinstance(plan, FastGemmPlan):
        result.update(
            kind="fast-gemm",
            n=plan.n,
            k=plan.k,
            np=plan.np,
            kp=plan.kp,
            nd=plan.nd,
            kd=plan.kd,
            block_size=plan.block_size,
            blocks_per_partition=plan.blocks_per_partition,
            packed_partitions=plan.packed_partitions,
            shift=plan.shift,
            shift_buffer=plan.shift_buffer,
            grid_size=plan.grid_size,
            input_replications=plan.input_replications,
        )
    else:
        result.update(
            kind="fast-conv",
            channel_in=plan.channel_in,
            channel_out=plan.channel_out,
            output_height=plan.output_height,
            output_width=plan.output_width,
            kernel_hw=plan.kernel_hw,
            group=plan.group,
            stride=plan.stride,
            input_size=plan.input_size,
            output_size=plan.output_size,
            num_slots=plan.num_slots,
            num_grid=plan.num_grid,
            num_block=plan.num_block,
            width_block=plan.width_block,
            width_block_data=plan.width_block_data,
            width_block_pad=plan.width_block_pad,
            position_block=plan.position_block,
            capacity_block=plan.capacity_block,
            input_duplications=plan.input_duplications,
            blocking_outer_depth=plan.blocking_outer_depth,
            cyclic_roll=plan.cyclic_roll,
            sharding_offset=(
                None if plan.sharding_offset is None else vars(plan.sharding_offset)
            ),
        )
    return result


def _provider_result_binding_data(result: ProviderResult) -> dict:
    """Return strict transport data; C++ still validates every semantic field."""
    if not isinstance(result, ProviderResult):
        raise TypeError("Python vector-kernel provider must return ProviderResult")
    return {
        "plan": _plan_binding(result.plan),
        "constants": tuple(
            {
                "role": item.role,
                "values": item.values,
                "content_hash": item.content_hash,
            }
            for item in result.constants
        ),
        "runtime_preparations": tuple(
            {
                "role": item.role,
                "source_operand": item.source_operand,
                "kind": item.kind.value,
                "result_type": _type_binding(item.result_type),
                "logical_input_size": item.logical_input_size,
                "replications": item.replications,
                "blocking_width": item.blocking_width,
                "rotation_candidates": item.rotation_candidates,
                "outer_block_depth": item.outer_block_depth,
            }
            for item in result.runtime_preparations
        ),
        "scalar_preparations": tuple(vars(item) for item in result.scalar_preparations),
        "provenance": result.provenance,
    }


__all__ = [
    "AffineIndexPlan",
    "AttributeRecord",
    "BaselineConvPlan",
    "BaselineGemmPlan",
    "CommonPlan",
    "ConstantPlan",
    "FastConvPlan",
    "FastGemmPlan",
    "LoopPlan",
    "MaskPlan",
    "MaskPolicy",
    "Operation",
    "OptionSnapshot",
    "PlanningError",
    "PlanningRequest",
    "ProviderResult",
    "RankedTypePlan",
    "ReductionKind",
    "ReductionPlan",
    "RequestedPlanKind",
    "RotationPlan",
    "RuntimePreparation",
    "RuntimePreparationKind",
    "ScalarPreparation",
    "ShardingOffsetPlan",
    "SlicePlan",
    "SlotPlan",
    "SlotPolicy",
    "TargetSnapshot",
    "TypedPayload",
    "canonical_payload_hash",
    "plan_vector_kernel",
]
