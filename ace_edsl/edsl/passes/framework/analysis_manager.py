"""Invocation-local analysis dependency resolution and revision cache."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .api import (
    AnalysisExecutionResult,
    AnalysisPass,
    AnalysisRequest,
    AnalysisResult,
    AIRObjectId,
    FrozenMap,
    IRUnitRef,
    UnitSelector,
    freeze_metrics,
    pass_identity,
    pass_instance_id,
)
from .diagnostics import Diagnostic, error


@dataclass(frozen=True)
class _CacheKey:
    analysis_identity: Tuple[Any, ...]
    unit: IRUnitRef
    revision: int


@dataclass(frozen=True)
class _CacheEntry:
    analysis: AnalysisPass
    result: AnalysisResult
    dependencies: Tuple[_CacheKey, ...]
    analysis_identity: Tuple[Any, ...]
    pass_id: str
    cross_revision_safe: bool


@dataclass(frozen=True)
class _Resolution:
    result: AnalysisResult
    key: _CacheKey
    cached: bool


def _is_immutable_host_value(value: Any, seen: Optional[set[int]] = None) -> bool:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Enum)):
        return True
    if isinstance(value, IRUnitRef):
        return True
    if isinstance(value, tuple):
        return all(_is_immutable_host_value(item, seen) for item in value)
    if isinstance(value, frozenset):
        return all(_is_immutable_host_value(item, seen) for item in value)
    if isinstance(value, FrozenMap):
        return all(
            _is_immutable_host_value(key, seen)
            and _is_immutable_host_value(item, seen)
            for key, item in value.items()
        )
    if is_dataclass(value):
        params = getattr(type(value), "__dataclass_params__", None)
        if not params or not params.frozen:
            return False
        seen = seen or set()
        if id(value) in seen:
            return True
        seen.add(id(value))
        return all(
            _is_immutable_host_value(getattr(value, field.name), seen)
            for field in fields(value)
        )
    return False


def _contains_air_identifier(value: Any, seen: Optional[set[int]] = None) -> bool:
    if isinstance(value, (IRUnitRef, AIRObjectId)):
        return True
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Enum)):
        return False
    if isinstance(value, (tuple, frozenset)):
        return any(_contains_air_identifier(item, seen) for item in value)
    if isinstance(value, FrozenMap):
        return any(
            _contains_air_identifier(key, seen)
            or _contains_air_identifier(item, seen)
            for key, item in value.items()
        )
    if is_dataclass(value):
        seen = seen or set()
        if id(value) in seen:
            return False
        seen.add(id(value))
        return any(
            _contains_air_identifier(getattr(value, field.name), seen)
            for field in fields(value)
        )
    return False


def _safe_pass_instance_id(instance: object) -> str:
    try:
        return pass_instance_id(instance)
    except Exception:
        return f"{instance.__class__.__module__}.{instance.__class__.__qualname__}"


class AnalysisContext:
    """Read-only access to dependencies already declared by an analysis."""

    __slots__ = ("_results", "_module_unit", "user_context")

    def __init__(
        self,
        results: Mapping[_CacheKey, AnalysisResult],
        module_unit: IRUnitRef,
        user_context,
    ):
        self._results = dict(results)
        self._module_unit = module_unit
        self.user_context = user_context

    def result(self, request: AnalysisRequest, unit: IRUnitRef) -> AnalysisResult:
        if request.unit is UnitSelector.CURRENT_MODULE:
            resolved = self._module_unit
        elif request.unit is UnitSelector.CURRENT_PASS_UNIT:
            resolved = unit
        else:
            resolved = request.unit
        for key, result in self._results.items():
            if key.analysis_identity == pass_identity(request.analysis) and key.unit == resolved:
                return result
        raise RuntimeError(
            f"undeclared analysis requested while running {pass_instance_id(request.analysis)}"
        )


class AnalysisManager:
    """One manager-invocation cache; never shared across native barriers."""

    def __init__(
        self,
        module_id: object,
        module_unit: IRUnitRef,
        instrumentation=None,
        ir_dump=None,
    ):
        self.module_id = module_id
        self.module_unit = module_unit
        self._cache: Dict[_CacheKey, _CacheEntry] = {}
        self._active: list[_CacheKey] = []
        self._active_pass_ids: list[str] = []
        self._dependency_trace: list[AnalysisExecutionResult] = []
        self._instrumentation = instrumentation
        self._ir_dump = ir_dump

    @staticmethod
    def _observe(callback, code, message, diagnostics, *args):
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            diagnostics.append(
                error(code, f"{message} raised {type(exc).__name__}: {exc}")
            )

    @property
    def dependency_trace(self) -> Tuple[AnalysisExecutionResult, ...]:
        return tuple(self._dependency_trace)

    def _instrument_dependency_execution(
        self,
        pass_id: str,
        unit: IRUnitRef,
        revision: int,
        result: AnalysisResult,
        unit_loader,
    ) -> AnalysisExecutionResult:
        diagnostics: list[Diagnostic] = []
        try:
            read_only_unit = unit_loader(unit)
        except Exception as exc:
            diagnostics.append(
                error(
                    "air.instrumentation.begin-failed",
                    f"begin callback for {pass_id} could not load AIR: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            self._observe(
                getattr(self._instrumentation, "on_begin", None),
                "air.instrumentation.begin-failed",
                f"begin callback for {pass_id}",
                diagnostics,
                pass_id,
                read_only_unit,
            )
        if getattr(self._instrumentation, "on_ir_dump", None):
            try:
                rendered = self._ir_dump() if self._ir_dump else ""
            except Exception as exc:
                diagnostics.append(
                    error(
                        "air.instrumentation.dump-failed",
                        f"IR dump for {pass_id} raised {type(exc).__name__}: {exc}",
                    )
                )
            else:
                self._observe(
                    self._instrumentation.on_ir_dump,
                    "air.instrumentation.dump-failed",
                    f"IR dump callback for {pass_id}",
                    diagnostics,
                    pass_id,
                    rendered,
                )
        execution = AnalysisExecutionResult(
            pass_id,
            unit,
            result.success,
            False,
            (*result.diagnostics, *diagnostics),
            result.metrics,
            result,
            revision,
            revision,
        )
        self._observe(
            getattr(self._instrumentation, "on_end", None),
            "air.instrumentation.end-failed",
            f"end callback for {pass_id}",
            diagnostics,
            pass_id,
            execution,
        )
        if diagnostics:
            execution = AnalysisExecutionResult(
                pass_id,
                unit,
                result.success,
                False,
                (*result.diagnostics, *diagnostics),
                result.metrics,
                result,
                revision,
                revision,
            )
        return execution

    def checkpoint(self):
        return dict(self._cache), list(self._dependency_trace)

    def restore(self, checkpoint) -> None:
        self._cache, self._dependency_trace = checkpoint

    def clear(self) -> None:
        self._cache.clear()
        self._active.clear()
        self._active_pass_ids.clear()

    def resolve_unit(
        self,
        selector,
        current_unit: IRUnitRef,
    ) -> IRUnitRef:
        if selector is UnitSelector.CURRENT_MODULE:
            return self.module_unit
        if selector is UnitSelector.CURRENT_PASS_UNIT:
            return current_unit
        if not isinstance(selector, IRUnitRef):
            raise TypeError("analysis unit must be a UnitSelector or IRUnitRef")
        if selector.module_id != self.module_id:
            raise ValueError("IRUnitRef belongs to another manager invocation")
        return selector

    def resolve(
        self,
        request: AnalysisRequest,
        current_unit: IRUnitRef,
        revision: int,
        unit_loader,
        user_context=None,
        *,
        dependency: bool = False,
    ) -> _Resolution:
        try:
            unit = self.resolve_unit(request.unit, current_unit)
        except Exception as exc:
            result = AnalysisResult(
                False,
                diagnostics=(
                    error("air.analysis.invalid-unit", str(exc)),
                ),
            )
            try:
                identity = pass_identity(request.analysis)
            except Exception:
                identity = (
                    request.analysis.__class__.__module__,
                    request.analysis.__class__.__qualname__,
                    ("invalid-configuration",),
                )
            key = _CacheKey(identity, current_unit, revision)
            return _Resolution(result, key, False)

        fallback_pass_id = (
            f"{request.analysis.__class__.__module__}."
            f"{request.analysis.__class__.__qualname__}"
        )
        try:
            configured_pass_id = pass_instance_id(request.analysis)
        except Exception:
            configured_pass_id = fallback_pass_id
        try:
            configured_identity = pass_identity(request.analysis)
            configured_safe = bool(request.analysis.cross_revision_safe)
        except Exception as exc:
            diagnostic = error(
                "air.pass.invalid-configuration",
                f"{configured_pass_id} has invalid configuration: "
                f"{type(exc).__name__}: {exc}",
            )
            result = AnalysisResult(False, diagnostics=(diagnostic,))
            key = _CacheKey(
                (
                    request.analysis.__class__.__module__,
                    request.analysis.__class__.__qualname__,
                    ("invalid-configuration",),
                ),
                unit,
                revision,
            )
            if dependency:
                execution = self._instrument_dependency_execution(
                    configured_pass_id,
                    unit,
                    revision,
                    result,
                    unit_loader,
                )
                self._dependency_trace.append(execution)
            return _Resolution(result, key, False)

        key = _CacheKey(configured_identity, unit, revision)
        cached = self._cache.get(key)
        if cached is not None:
            if (
                cached.analysis_identity != configured_identity
                or cached.pass_id != configured_pass_id
                or cached.cross_revision_safe != configured_safe
            ):
                result = AnalysisResult(
                    False,
                    diagnostics=(
                        error(
                            "air.pass.mutated-configuration",
                            f"{cached.pass_id} mutated its configuration after analysis",
                        ),
                    ),
                )
                return _Resolution(result, key, False)
            return _Resolution(cached.result, key, True)

        if key in self._active:
            cycle_start = self._active.index(key)
            names = " -> ".join(
                (*self._active_pass_ids[cycle_start:], configured_pass_id)
            )
            return _Resolution(
                AnalysisResult(
                    False,
                    diagnostics=(
                        error(
                            "air.analysis.dependency-cycle",
                            f"analysis dependency cycle: {names}",
                        ),
                    ),
                ),
                key,
                False,
            )

        self._active.append(key)
        self._active_pass_ids.append(configured_pass_id)
        instrumentation_diagnostics: list[Diagnostic] = []
        dependency_keys: list[_CacheKey] = []
        dependency_results: Dict[_CacheKey, AnalysisResult] = {}
        try:
            if dependency:
                self._observe(
                    getattr(self._instrumentation, "on_begin", None),
                    "air.instrumentation.begin-failed",
                    f"begin callback for {configured_pass_id}",
                    instrumentation_diagnostics,
                    configured_pass_id,
                    unit_loader(unit),
                )
            raw_dependencies = request.analysis.dependencies(unit)
            if not isinstance(raw_dependencies, tuple) or not all(
                isinstance(item, AnalysisRequest) for item in raw_dependencies
            ):
                result = AnalysisResult(
                    False,
                    diagnostics=(
                        error(
                            "air.analysis.invalid-dependencies",
                            f"{configured_pass_id} dependencies must be a tuple of AnalysisRequest values",
                        ),
                    ),
                )
            else:
                for child in raw_dependencies:
                    resolution = self.resolve(
                        child,
                        unit,
                        revision,
                        unit_loader,
                        user_context,
                        dependency=True,
                    )
                    dependency_keys.append(resolution.key)
                    dependency_results[resolution.key] = resolution.result
                    if not resolution.result.success:
                        result = AnalysisResult(
                            False,
                            diagnostics=(
                                error(
                                    "air.analysis.dependency-failed",
                                    f"dependency {_safe_pass_instance_id(child.analysis)} failed for {configured_pass_id}",
                                ),
                                *resolution.result.diagnostics,
                            ),
                        )
                        break
                else:
                    try:
                        raw_result = request.analysis.run(
                            unit_loader(unit),
                            AnalysisContext(
                                dependency_results, self.module_unit, user_context
                            ),
                        )
                    except Exception as exc:
                        result = AnalysisResult(
                            False,
                            diagnostics=(
                                error(
                                    "air.analysis.exception",
                                    f"{configured_pass_id} raised {type(exc).__name__}: {exc}",
                                ),
                            ),
                        )
                    else:
                        if not isinstance(raw_result, AnalysisResult):
                            result = AnalysisResult(
                                False,
                                diagnostics=(
                                    error(
                                        "air.analysis.invalid-result",
                                        f"{configured_pass_id} returned {type(raw_result).__name__}, expected AnalysisResult",
                                    ),
                                ),
                            )
                        elif raw_result.success and not _is_immutable_host_value(raw_result.value):
                            result = AnalysisResult(
                                False,
                                diagnostics=(
                                    error(
                                        "air.analysis.mutable-result",
                                        f"{configured_pass_id} returned mutable or live AIR data",
                                    ),
                                ),
                            )
                        elif (
                            raw_result.success
                            and configured_safe
                            and (
                                _contains_air_identifier(
                                    (raw_result.value, raw_result.metrics)
                                )
                                or any(
                                    diagnostic.location is not None
                                    for diagnostic in raw_result.diagnostics
                                )
                            )
                        ):
                            result = AnalysisResult(
                                False,
                                diagnostics=(
                                    error(
                                        "air.analysis.unsafe-result-contract",
                                        f"cross-revision-safe analysis {configured_pass_id} returned revision-bound identifiers",
                                    ),
                                ),
                            )
                        else:
                            result = raw_result
        except Exception as exc:
            result = AnalysisResult(
                False,
                diagnostics=(
                    error(
                        "air.analysis.exception",
                        f"{configured_pass_id} raised {type(exc).__name__}: {exc}",
                    ),
                ),
            )
        finally:
            self._active.pop()
            self._active_pass_ids.pop()

        try:
            current_identity = pass_identity(request.analysis)
            current_pass_id = pass_instance_id(request.analysis)
            current_safe = bool(request.analysis.cross_revision_safe)
        except Exception as exc:
            result = AnalysisResult(
                False,
                diagnostics=(
                    error(
                        "air.pass.invalid-configuration",
                        f"{configured_pass_id} has invalid configuration: {exc}",
                    ),
                ),
            )
        else:
            if (
                current_identity != configured_identity
                or current_pass_id != configured_pass_id
                or current_safe != configured_safe
            ):
                result = AnalysisResult(
                    False,
                    diagnostics=(
                        error(
                            "air.pass.mutated-configuration",
                        f"{configured_pass_id} mutated its configuration while running",
                        ),
                    ),
                )

        if dependency and getattr(self._instrumentation, "on_ir_dump", None):
            try:
                rendered = self._ir_dump() if self._ir_dump else ""
            except Exception as exc:
                instrumentation_diagnostics.append(
                    error(
                        "air.instrumentation.dump-failed",
                        f"IR dump for {configured_pass_id} raised {type(exc).__name__}: {exc}",
                    )
                )
            else:
                self._observe(
                    self._instrumentation.on_ir_dump,
                    "air.instrumentation.dump-failed",
                    f"IR dump callback for {configured_pass_id}",
                    instrumentation_diagnostics,
                    configured_pass_id,
                    rendered,
                )

        execution = AnalysisExecutionResult(
            configured_pass_id,
            unit,
            result.success,
            False,
            result.diagnostics,
            result.metrics,
            result,
            revision,
            revision,
        )
        if dependency:
            self._observe(
                getattr(self._instrumentation, "on_end", None),
                "air.instrumentation.end-failed",
                f"end callback for {configured_pass_id}",
                instrumentation_diagnostics,
                configured_pass_id,
                execution,
            )
        if instrumentation_diagnostics:
            execution = AnalysisExecutionResult(
                configured_pass_id,
                unit,
                result.success,
                False,
                (*result.diagnostics, *instrumentation_diagnostics),
                result.metrics,
                result,
                revision,
                revision,
            )
        entry = _CacheEntry(
            request.analysis,
            result,
            tuple(dependency_keys),
            configured_identity,
            configured_pass_id,
            configured_safe,
        )
        self._cache[key] = entry
        if dependency:
            self._dependency_trace.append(execution)
        return _Resolution(result, key, False)

    def validate_preservation(self, preserved, revision: int) -> Tuple[Diagnostic, ...]:
        diagnostics: list[Diagnostic] = []
        preserved_keys = {
            key
            for key in self._cache
            if key.revision == revision
            and key.analysis_identity in preserved.identities
        }
        for key in sorted(
            preserved_keys,
            key=lambda item: (repr(item.analysis_identity), item.unit.kind, item.unit.native_id or -1),
        ):
            entry = self._cache[key]
            try:
                unchanged = (
                    pass_identity(entry.analysis) == entry.analysis_identity
                    and pass_instance_id(entry.analysis) == entry.pass_id
                    and bool(entry.analysis.cross_revision_safe)
                    == entry.cross_revision_safe
                )
            except Exception:
                unchanged = False
            if not unchanged:
                diagnostics.append(
                    error(
                        "air.pass.mutated-configuration",
                        f"{entry.pass_id} mutated its configuration after analysis",
                    )
                )
                continue
            if not entry.cross_revision_safe:
                diagnostics.append(
                    error(
                        "air.analysis.unsafe-preservation",
                        f"cannot preserve revision-bound analysis {entry.pass_id}",
                    )
                )
                continue
            missing = tuple(
                dependency
                for dependency in entry.dependencies
                if dependency not in preserved_keys
                or not self._cache[dependency].cross_revision_safe
            )
            if missing:
                names = ", ".join(
                    self._cache[item].pass_id
                    for item in missing
                    if item in self._cache
                ) or "uncached dependency"
                diagnostics.append(
                    error(
                        "air.analysis.preservation-not-closed",
                        f"preserving {entry.pass_id} also requires safe preservation of: {names}",
                    )
                )
        return tuple(diagnostics)

    def advance_revision(self, old_revision: int, new_revision: int, preserved) -> None:
        rekey: list[tuple[_CacheKey, _CacheKey, _CacheEntry]] = []
        for key, entry in tuple(self._cache.items()):
            if key.revision != old_revision:
                continue
            if (
                key.analysis_identity in preserved.identities
                and entry.cross_revision_safe
            ):
                new_key = _CacheKey(key.analysis_identity, key.unit, new_revision)
                rekey.append((key, new_key, entry))
            del self._cache[key]
        for old_key, new_key, entry in rekey:
            dependencies = tuple(
                _CacheKey(dep.analysis_identity, dep.unit, new_revision)
                if dep.revision == old_revision
                else dep
                for dep in entry.dependencies
            )
            self._cache[new_key] = _CacheEntry(
                entry.analysis,
                entry.result,
                dependencies,
                entry.analysis_identity,
                entry.pass_id,
                entry.cross_revision_safe,
            )
