"""Owned host-plan views and adapters for destination Vector recipes."""

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class RankedTypePlan:
    element_type: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class LoopPlan:
    role: str
    lower: int
    upper: int
    step: int
    nesting_depth: int


@dataclass(frozen=True)
class AffineIndexPlan:
    iv_coefficients: Tuple[int, ...]
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
    candidates: Tuple[int, ...]


@dataclass(frozen=True)
class ReductionPlan:
    role: str
    kind: str
    factor: int
    block_width: int
    padding: int


@dataclass(frozen=True)
class MaskPlan:
    policy: str
    valid_length: int


@dataclass(frozen=True)
class SlotPlan:
    policy: str
    value: int


@dataclass(frozen=True)
class ConstantPlan:
    role: str
    type: RankedTypePlan
    content_hash: str
    bytes: bytes


@dataclass(frozen=True)
class RuntimePreparationPlan:
    role: str
    source_operand: int
    kind: str
    result_type: RankedTypePlan
    logical_input_size: int
    replications: int
    blocking_width: int
    rotation_candidates: Tuple[int, ...]
    outer_block_depth: int


@dataclass(frozen=True)
class PreparedBaselineGemmPlan:
    kind: str
    provenance: str
    specialization_key: str
    helper_name: str
    height: int
    width: int
    input_duplications: int
    runtime_vector_inputs: Tuple[RankedTypePlan, ...]
    result_type: RankedTypePlan
    loops: Tuple[LoopPlan, ...]
    slices: Tuple[SlicePlan, ...]
    rotations: Tuple[RotationPlan, ...]
    reductions: Tuple[ReductionPlan, ...]
    mask: MaskPlan
    slot: SlotPlan
    constants: Tuple[ConstantPlan, ...]
    runtime_preparations: Tuple[RuntimePreparationPlan, ...]

    def loop(self, role: str) -> LoopPlan:
        return _unique_role(self.loops, role, "loop")

    def slice(self, role: str) -> SlicePlan:
        return _unique_role(self.slices, role, "slice")

    def rotation(self, role: str) -> RotationPlan:
        return _unique_role(self.rotations, role, "rotation")

    def reduction(self, role: str) -> ReductionPlan:
        return _unique_role(self.reductions, role, "reduction")

    def constant(self, role: str) -> ConstantPlan:
        return _unique_role(self.constants, role, "constant")


@dataclass(frozen=True)
class PreparedBaselineConvPlan:
    kind: str
    provenance: str
    specialization_key: str
    helper_name: str
    channel_in: int
    channel_out: int
    output_height: int
    output_width: int
    kernel_hw: int
    stride: int
    input_duplications: int
    runtime_vector_inputs: Tuple[RankedTypePlan, ...]
    result_type: RankedTypePlan
    loops: Tuple[LoopPlan, ...]
    slices: Tuple[SlicePlan, ...]
    rotations: Tuple[RotationPlan, ...]
    reductions: Tuple[ReductionPlan, ...]
    mask: MaskPlan
    slot: SlotPlan
    constants: Tuple[ConstantPlan, ...]
    runtime_preparations: Tuple[RuntimePreparationPlan, ...]

    def loop(self, role: str) -> LoopPlan:
        return _unique_role(self.loops, role, "loop")

    def slice(self, role: str) -> SlicePlan:
        return _unique_role(self.slices, role, "slice")

    def rotation(self, role: str) -> RotationPlan:
        return _unique_role(self.rotations, role, "rotation")

    def reduction(self, role: str) -> ReductionPlan:
        return _unique_role(self.reductions, role, "reduction")

    def constant(self, role: str) -> ConstantPlan:
        return _unique_role(self.constants, role, "constant")

    def runtime_preparation(self, role: str) -> RuntimePreparationPlan:
        return _unique_role(
            self.runtime_preparations, role, "runtime preparation"
        )


def _unique_role(records, role: str, record_name: str):
    matches = tuple(record for record in records if record.role == role)
    if len(matches) != 1:
        raise KeyError(
            f"prepared {record_name} role {role!r} has {len(matches)} matches"
        )
    return matches[0]


def _ranked_type(data) -> RankedTypePlan:
    return RankedTypePlan(
        element_type=str(data["element_type"]),
        shape=tuple(int(value) for value in data["shape"]),
    )


def _freeze_common_plan_fields(data):
    return {
        "runtime_vector_inputs": tuple(
            _ranked_type(item) for item in data["runtime_vector_inputs"]
        ),
        "result_type": _ranked_type(data["result_type"]),
        "loops": tuple(LoopPlan(**dict(item)) for item in data["loops"]),
        "slices": tuple(
            SlicePlan(
                role=str(item["role"]),
                index=AffineIndexPlan(
                    iv_coefficients=tuple(
                        int(value)
                        for value in item["index"]["iv_coefficients"]
                    ),
                    constant=int(item["index"]["constant"]),
                    uses_sharding_offset=bool(
                        item["index"]["uses_sharding_offset"]
                    ),
                ),
                width=int(item["width"]),
            )
            for item in data["slices"]
        ),
        "rotations": tuple(
            RotationPlan(
                role=str(item["role"]),
                candidates=tuple(
                    int(value) for value in item["candidates"]
                ),
            )
            for item in data["rotations"]
        ),
        "reductions": tuple(
            ReductionPlan(
                role=str(item["role"]),
                kind=str(item["kind"]),
                factor=int(item["factor"]),
                block_width=int(item["block_width"]),
                padding=int(item["padding"]),
            )
            for item in data["reductions"]
        ),
        "mask": MaskPlan(
            policy=str(data["mask"]["policy"]),
            valid_length=int(data["mask"]["valid_length"]),
        ),
        "slot": SlotPlan(
            policy=str(data["slot"]["policy"]),
            value=int(data["slot"]["value"]),
        ),
        "constants": tuple(
            ConstantPlan(
                role=str(item["role"]),
                type=_ranked_type(item["type"]),
                content_hash=str(item["content_hash"]),
                bytes=bytes(item["bytes"]),
            )
            for item in data["constants"]
        ),
        "runtime_preparations": tuple(
            RuntimePreparationPlan(
                role=str(item["role"]),
                source_operand=int(item["source_operand"]),
                kind=str(item["kind"]),
                result_type=_ranked_type(item["result_type"]),
                logical_input_size=int(item["logical_input_size"]),
                replications=int(item["replications"]),
                blocking_width=int(item["blocking_width"]),
                rotation_candidates=tuple(
                    int(value)
                    for value in item["rotation_candidates"]
                ),
                outer_block_depth=int(item["outer_block_depth"]),
            )
            for item in data["runtime_preparations"]
        ),
    }


def _freeze_prepared_baseline_gemm_plan(data) -> PreparedBaselineGemmPlan:
    """Copy a binding snapshot into an immutable, AIR-independent value."""
    if data["kind"] != "baseline-gemm":
        raise ValueError("baseline Gemm freezer received another plan kind")
    return PreparedBaselineGemmPlan(
        kind=str(data["kind"]),
        provenance=str(data["provenance"]),
        specialization_key=str(data["specialization_key"]),
        helper_name=str(data["helper_name"]),
        height=int(data["height"]),
        width=int(data["width"]),
        input_duplications=int(data["input_duplications"]),
        **_freeze_common_plan_fields(data),
    )


def _freeze_prepared_baseline_conv_plan(data) -> PreparedBaselineConvPlan:
    """Copy a baseline Conv snapshot into an immutable owned value."""
    if data["kind"] != "baseline-conv":
        raise ValueError("baseline Conv freezer received another plan kind")
    return PreparedBaselineConvPlan(
        kind=str(data["kind"]),
        provenance=str(data["provenance"]),
        specialization_key=str(data["specialization_key"]),
        helper_name=str(data["helper_name"]),
        channel_in=int(data["channel_in"]),
        channel_out=int(data["channel_out"]),
        output_height=int(data["output_height"]),
        output_width=int(data["output_width"]),
        kernel_hw=int(data["kernel_hw"]),
        stride=int(data["stride"]),
        input_duplications=int(data["input_duplications"]),
        **_freeze_common_plan_fields(data),
    )


def vector_kernel_recipe(kernel: Callable) -> Callable:
    """Adapt a decorated ``@vector_kernel`` to the low-level pass callback."""
    if getattr(kernel, "_py_domain", None) != "nn::vector":
        raise TypeError("vector-kernel recipe requires a @vector_kernel callable")

    def recipe(trace_context, prepared):
        from .edsl import AceEDSL

        return AceEDSL._get_dsl().trace_vector_kernel_into(
            trace_context, kernel, prepared
        )

    return recipe
