"""One invocation-scoped extension point before Vector-to-SIHE lowering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

from .api import (
    FrozenMap,
    AnalysisRequest,
    PassManagerResult,
    PassStep,
    TransformationPass,
    freeze_metrics,
    pass_identity,
    pass_instance_id,
)
from .diagnostics import Diagnostic, DiagnosticSeverity, error
from .manager import AIRPassManager


class PipelinePoint(str, Enum):
    BEFORE_VECTOR2SIHE = "before-vector2sihe"


@dataclass(frozen=True)
class PassInstrumentation:
    on_begin: Optional[Callable] = None
    on_end: Optional[Callable] = None
    on_ir_dump: Optional[Callable] = None


class PassPipelineConfig:
    """Immutable per-run pass tuples; construction rejects duplicate IDs."""

    __slots__ = (
        "_steps",
        "_pass_identities",
        "_pass_instance_ids",
        "instrumentation",
        "_initialized",
    )

    def __init__(
        self,
        steps: Optional[Mapping[PipelinePoint, Tuple[PassStep, ...]]] = None,
        *,
        instrumentation: Optional[PassInstrumentation] = None,
    ):
        if instrumentation is not None:
            if not isinstance(instrumentation, PassInstrumentation):
                raise TypeError("instrumentation must be PassInstrumentation")
            for name in ("on_begin", "on_end", "on_ir_dump"):
                callback = getattr(instrumentation, name)
                if callback is not None and not callable(callback):
                    raise TypeError(f"instrumentation.{name} must be callable")
        normalized = {}
        pass_identities = {}
        for point, configured in (steps or {}).items():
            point = point if isinstance(point, PipelinePoint) else PipelinePoint(point)
            configured = tuple(configured)
            if not all(
                isinstance(step, (AnalysisRequest, TransformationPass))
                for step in configured
            ):
                raise TypeError(
                    f"{point.value} accepts only AnalysisRequest or TransformationPass values"
                )
            identities = [
                pass_instance_id(
                    step.analysis if isinstance(step, AnalysisRequest) else step
                )
                for step in configured
            ]
            duplicates = tuple(
                dict.fromkeys(
                    identity
                    for identity in identities
                    if identities.count(identity) > 1
                )
            )
            if duplicates:
                raise ValueError(
                    f"duplicate pass-instance IDs at {point.value}: "
                    + ", ".join(duplicates)
                )
            normalized[point] = configured
            pass_identities[point] = tuple(
                pass_identity(
                    step.analysis if isinstance(step, AnalysisRequest) else step
                )
                for step in configured
            )
        object.__setattr__(self, "_steps", MappingProxyType(normalized))
        object.__setattr__(
            self, "_pass_identities", MappingProxyType(pass_identities)
        )
        object.__setattr__(
            self, "_pass_instance_ids", MappingProxyType({
                point: tuple(
                    pass_instance_id(
                        step.analysis if isinstance(step, AnalysisRequest) else step
                    )
                    for step in configured
                )
                for point, configured in normalized.items()
            })
        )
        object.__setattr__(self, "instrumentation", instrumentation)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name, value):
        if getattr(self, "_initialized", False):
            raise AttributeError("PassPipelineConfig is immutable")
        object.__setattr__(self, name, value)

    def __getitem__(self, point: PipelinePoint) -> Tuple[PassStep, ...]:
        return self._steps.get(point, ())

    def copy_for_invocation(self) -> "PassPipelineConfig":
        for point, configured in self._steps.items():
            current = tuple(
                pass_identity(
                    step.analysis if isinstance(step, AnalysisRequest) else step
                )
                for step in configured
            )
            if current != self._pass_identities[point]:
                raise TypeError(
                    f"pass configuration mutated after registration at {point.value}"
                )
            current_ids = tuple(
                pass_instance_id(
                    step.analysis if isinstance(step, AnalysisRequest) else step
                )
                for step in configured
            )
            if current_ids != self._pass_instance_ids[point]:
                raise TypeError(
                    f"pass instance ID mutated after registration at {point.value}"
                )
        return PassPipelineConfig(dict(self._steps), instrumentation=self.instrumentation)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PassPipelineConfig)
            and dict(self._steps) == dict(other._steps)
            and self.instrumentation == other.instrumentation
        )


class NativeBarrierStep:
    """Marker for an opaque native step that cannot enter AIRPassManager."""


@dataclass(frozen=True)
class NativeBarrierOutcome:
    success: bool
    changed: bool
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))


@dataclass(frozen=True)
class NativeBarrierExecutionResult:
    step_id: str
    success: bool
    changed: bool
    before_revision: int
    after_revision: int
    diagnostics: Tuple[Diagnostic, ...] = ()
    metrics: FrozenMap = FrozenMap()

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "metrics", freeze_metrics(self.metrics))


@dataclass(frozen=True)
class ManagerSegmentExecution:
    result: PassManagerResult


HookRecord = ManagerSegmentExecution | NativeBarrierExecutionResult


@dataclass(frozen=True)
class HookExecutionResult:
    success: bool
    changed: bool
    records: Tuple[HookRecord, ...] = ()
    diagnostics: Tuple[Diagnostic, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def manager_segments(self) -> Tuple[PassManagerResult, ...]:
        return tuple(
            record.result
            for record in self.records
            if isinstance(record, ManagerSegmentExecution)
        )

    @property
    def barrier_records(self) -> Tuple[NativeBarrierExecutionResult, ...]:
        return tuple(
            record
            for record in self.records
            if isinstance(record, NativeBarrierExecutionResult)
        )


def _revision(glob_scope) -> int:
    return int(glob_scope.air_pass_revision)


def run_pipeline_hook(
    glob_scope,
    point: PipelinePoint,
    config: Optional[PassPipelineConfig] = None,
    *,
    native_barriers: Tuple[NativeBarrierStep, ...] = (),
) -> HookExecutionResult:
    """Run one Python manager segment followed by opaque native barriers."""

    if config is None:
        config = PassPipelineConfig()
    elif not isinstance(config, PassPipelineConfig):
        raise TypeError("config must be PassPipelineConfig")
    invocation = config.copy_for_invocation()
    native_barriers = tuple(native_barriers)
    if not all(isinstance(step, NativeBarrierStep) for step in native_barriers):
        raise TypeError("native_barriers must contain NativeBarrierStep values")
    records = []
    diagnostics = []
    changed = False
    steps = invocation[point]
    if steps or native_barriers:
        manager_result = AIRPassManager().run(
            glob_scope,
            steps,
            {"instrumentation": invocation.instrumentation},
        )
        records.append(ManagerSegmentExecution(manager_result))
        diagnostics.extend(manager_result.diagnostics)
        changed = manager_result.changed
        if not manager_result.success:
            return HookExecutionResult(
                False, changed, tuple(records), tuple(diagnostics)
            )

    for step in native_barriers:
        step_id = pass_instance_id(step)
        before = _revision(glob_scope)
        before_pointer = glob_scope.get_native_ptr()
        before_generation = int(glob_scope.air_pass_generation)
        barrier_diagnostics = []
        try:
            outcome = step.run(glob_scope)
        except Exception as exc:
            outcome = NativeBarrierOutcome(
                False,
                False,
                (
                    error(
                        "air.native-barrier.exception",
                        f"{step_id} raised {type(exc).__name__}: {exc}",
                    ),
                ),
            )
        if not isinstance(outcome, NativeBarrierOutcome):
            outcome = NativeBarrierOutcome(
                False,
                False,
                (
                    error(
                        "air.native-barrier.invalid-result",
                        f"{step_id} returned {type(outcome).__name__}, expected NativeBarrierOutcome",
                    ),
                ),
            )
        barrier_diagnostics.extend(outcome.diagnostics)
        if not outcome.success and not any(
            diagnostic.severity is DiagnosticSeverity.ERROR
            for diagnostic in outcome.diagnostics
        ):
            barrier_diagnostics.append(
                error(
                    "air.native-barrier.failed",
                    f"{step_id} reported failure without an error diagnostic",
                )
            )

        observed_revision = _revision(glob_scope)
        observed_pointer = glob_scope.get_native_ptr()
        observed_generation = int(glob_scope.air_pass_generation)
        actual_changed = (
            observed_revision != before or observed_pointer != before_pointer
        )
        success = outcome.success
        if outcome.changed != actual_changed:
            success = False
            barrier_diagnostics.append(
                error(
                    "air.native-barrier.invalid-change",
                    f"{step_id} reported changed={outcome.changed} but committed AIR changed={actual_changed}",
                )
            )
        if actual_changed:
            if not outcome.success or observed_revision != before + 1:
                success = False
                barrier_diagnostics.append(
                    error(
                        "air.native-barrier.invalid-revision",
                        f"{step_id} reported a change without one verified revision advance",
                    )
                )
        else:
            if observed_revision != before:
                success = False
                barrier_diagnostics.append(
                    error(
                        "air.native-barrier.nontransactional-failure",
                        f"{step_id} advanced revision without replacing AIR",
                    )
                )

        if observed_generation == before_generation:
            try:
                glob_scope.invalidate_air_pass_views(False)
            except Exception as exc:
                success = False
                barrier_diagnostics.append(
                    error(
                        "air.native-barrier.invalidation-failed",
                        f"{step_id} could not close the barrier generation: {type(exc).__name__}: {exc}",
                    )
                )
        observed_revision = _revision(glob_scope)
        observed_generation = int(glob_scope.air_pass_generation)
        if observed_generation != before_generation + 1:
            success = False
            barrier_diagnostics.append(
                error(
                    "air.native-barrier.invalid-generation",
                    f"{step_id} barrier did not advance generation exactly once",
                )
            )

        barrier = NativeBarrierExecutionResult(
            step_id,
            success,
            actual_changed,
            before,
            observed_revision,
            tuple(barrier_diagnostics),
            outcome.metrics,
        )
        records.append(barrier)
        diagnostics.extend(barrier.diagnostics)
        changed = changed or barrier.changed
        if not barrier.success:
            return HookExecutionResult(
                False, changed, tuple(records), tuple(diagnostics)
            )

    return HookExecutionResult(True, changed, tuple(records), tuple(diagnostics))
