"""Caller-ordered, fail-fast execution of Python AIR pass steps."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Any, Mapping, Optional, Sequence, Tuple

from .analysis_manager import AnalysisManager
from .api import (
    CURRENT_MODULE,
    AnalysisExecutionResult,
    AnalysisRequest,
    AnalysisResult,
    IRUnitRef,
    PassExecutionResult,
    PassManagerResult,
    TransformResult,
    TransformationPass,
    UnitSelector,
    pass_identity,
    pass_instance_id,
)
from .diagnostics import Diagnostic, error


_INVOCATIONS = count(1)


def _revision(glob_scope) -> int:
    value = getattr(glob_scope, "air_pass_revision", None)
    if value is None:
        value = getattr(glob_scope, "_air_pass_revision", 0)
    return int(value() if callable(value) else value)


def _module_native_id(glob_scope) -> int:
    value = getattr(glob_scope, "air_pass_module_id", None)
    if value is None:
        value = getattr(glob_scope, "_air_pass_module_id", None)
    if value is None:
        value = glob_scope.get_native_ptr()
    return int(value() if callable(value) else value)


def _step_unit(step, module_unit: IRUnitRef, analysis_manager: AnalysisManager) -> IRUnitRef:
    configured = getattr(step, "unit", CURRENT_MODULE)
    return analysis_manager.resolve_unit(configured, module_unit)


def _step_id(step) -> str:
    return pass_instance_id(step.analysis if isinstance(step, AnalysisRequest) else step)


def _safe_step_id(step) -> str:
    instance = step.analysis if isinstance(step, AnalysisRequest) else step
    try:
        return pass_instance_id(instance)
    except Exception:
        return f"{instance.__class__.__module__}.{instance.__class__.__qualname__}"


def _module_view(glob_scope, module_id, *, transaction=None):
    from .api import ModuleView, MutableModuleView

    cls = MutableModuleView if transaction is not None else ModuleView
    return cls(glob_scope, module_id, transaction=transaction)


class TransformationContext:
    __slots__ = (
        "_analysis_manager",
        "_declared",
        "_current_unit",
        "_revision",
        "_source_to_candidate",
        "user_context",
    )

    def __init__(
        self,
        analysis_manager,
        declared,
        current_unit,
        revision,
        source_to_candidate,
        user_context,
    ):
        self._analysis_manager = analysis_manager
        self._declared = dict(declared)
        self._current_unit = current_unit
        self._revision = revision
        self._source_to_candidate = source_to_candidate
        self.user_context = user_context

    @property
    def source_to_candidate(self):
        return self._source_to_candidate

    def result(self, request: AnalysisRequest):
        unit = self._analysis_manager.resolve_unit(request.unit, self._current_unit)
        key = (pass_identity(request.analysis), unit)
        try:
            return self._declared[key]
        except KeyError as exc:
            raise RuntimeError(
                f"undeclared analysis requested after transaction opened: {_step_id(request)}"
            ) from exc

    def remap(self, stable_id):
        try:
            return self.source_to_candidate[stable_id]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unmapped or wrong-kind AIR ID: {stable_id!r}") from exc


def _instrumentation(user_context):
    if isinstance(user_context, Mapping):
        return user_context.get("instrumentation")
    return getattr(user_context, "instrumentation", None)


def _call_instrumentation(callback, code, message, diagnostics, *args):
    if callback is None:
        return
    try:
        callback(*args)
    except Exception as exc:
        diagnostics.append(
            error(code, f"{message} raised {type(exc).__name__}: {exc}")
        )


def _dump_for_instrumentation(callback, pass_id, glob_scope, diagnostics):
    if callback is None:
        return
    try:
        ir = glob_scope.dump()
    except Exception as exc:
        diagnostics.append(
            error(
                "air.instrumentation.dump-failed",
                f"IR dump for {pass_id} raised {type(exc).__name__}: {exc}",
            )
        )
        return
    _call_instrumentation(
        callback,
        "air.instrumentation.dump-failed",
        f"IR dump callback for {pass_id}",
        diagnostics,
        pass_id,
        ir,
    )


class AIRPassManager:
    """Serial pass manager with one cache and one transaction per transform."""

    def run(self, glob_scope, steps: Sequence, context=None) -> PassManagerResult:
        steps = tuple(steps)
        module_id = (_module_native_id(glob_scope), next(_INVOCATIONS))
        module_unit = IRUnitRef("module", module_id)
        instrumentation = _instrumentation(context)
        analyses = AnalysisManager(
            module_id,
            module_unit,
            instrumentation,
            glob_scope.dump,
        )
        executions = []
        diagnostics: list[Diagnostic] = []
        changed = False
        failed_at: Optional[int] = None

        try:
            module_view = _module_view(glob_scope, module_id)
        except Exception as exc:
            if not steps:
                return PassManagerResult(
                    False,
                    False,
                    diagnostics=(
                        error(
                            "air.pass.module-view-failed",
                            f"could not open committed AIR view: {type(exc).__name__}: {exc}",
                        ),
                    ),
                )
            pass_id = _safe_step_id(steps[0])
            diagnostic = error(
                "air.pass.module-view-failed",
                f"{pass_id} could not open committed AIR view: {type(exc).__name__}: {exc}",
            )
            if isinstance(steps[0], AnalysisRequest):
                analysis_result = AnalysisResult(False, diagnostics=(diagnostic,))
                revision = _revision(glob_scope)
                record = AnalysisExecutionResult(
                    pass_id,
                    module_unit,
                    False,
                    False,
                    (diagnostic,),
                    analysis_result=analysis_result,
                    before_revision=revision,
                    after_revision=revision,
                )
            else:
                revision = _revision(glob_scope)
                record = PassExecutionResult(
                    pass_id,
                    module_unit,
                    False,
                    False,
                    revision,
                    revision,
                    (diagnostic,),
                )
            return PassManagerResult(
                False,
                False,
                (record,),
                unexecuted_step_ids=tuple(
                    _safe_step_id(step) for step in steps[1:]
                ),
                diagnostics=(diagnostic,),
            )

        def load_unit(unit):
            if unit.kind == "module":
                return module_view
            return module_view.function(unit.native_id)

        for index, step in enumerate(steps):
            pass_id = _safe_step_id(step)
            dependency_start = len(analyses.dependency_trace)
            current_unit = module_unit
            local_diagnostics: list[Diagnostic] = []
            begin = getattr(instrumentation, "on_begin", None)
            end = getattr(instrumentation, "on_end", None)
            dump = getattr(instrumentation, "on_ir_dump", None)
            _call_instrumentation(
                begin,
                "air.instrumentation.begin-failed",
                f"begin callback for {pass_id}",
                local_diagnostics,
                pass_id,
                module_view,
            )
            record = None
            try:
                if isinstance(step, AnalysisRequest):
                    resolution = analyses.resolve(
                        step,
                        module_unit,
                        _revision(glob_scope),
                        load_unit,
                        context,
                    )
                    record = AnalysisExecutionResult(
                        pass_id,
                        resolution.key.unit,
                        resolution.result.success,
                        resolution.cached,
                        (*resolution.result.diagnostics, *local_diagnostics),
                        resolution.result.metrics,
                        resolution.result,
                        _revision(glob_scope),
                        _revision(glob_scope),
                    )
                elif isinstance(step, TransformationPass):
                    try:
                        current_unit = _step_unit(step, module_unit, analyses)
                    except Exception as exc:
                        before = _revision(glob_scope)
                        record = PassExecutionResult(
                            pass_id,
                            module_unit,
                            False,
                            False,
                            before,
                            before,
                            (
                                error("air.pass.invalid-unit", str(exc)),
                                *local_diagnostics,
                            ),
                        )
                    else:
                        record = self._run_transform(
                            glob_scope,
                            step,
                            current_unit,
                            module_id,
                            module_view,
                            analyses,
                            context,
                            tuple(local_diagnostics),
                        )
                        changed = changed or record.changed
                        if record.changed:
                            module_view = _module_view(glob_scope, module_id)
                else:
                    before = _revision(glob_scope)
                    record = PassExecutionResult(
                        pass_id,
                        module_unit,
                        False,
                        False,
                        before,
                        before,
                        (
                            error(
                                "air.pass.invalid-step",
                                f"unsupported pass step type: {type(step).__name__}",
                            ),
                            *local_diagnostics,
                        ),
                    )

                _dump_for_instrumentation(
                    dump, pass_id, glob_scope, local_diagnostics
                )
            except Exception as exc:
                diagnostic = error(
                    "air.pass.unexpected-exception",
                    f"{pass_id} raised {type(exc).__name__}: {exc}",
                )
                if isinstance(step, AnalysisRequest):
                    analysis_result = AnalysisResult(
                        False, diagnostics=(diagnostic,)
                    )
                    record = AnalysisExecutionResult(
                        pass_id,
                        current_unit,
                        False,
                        False,
                        (diagnostic, *local_diagnostics),
                        analysis_result=analysis_result,
                        before_revision=_revision(glob_scope),
                        after_revision=_revision(glob_scope),
                    )
                else:
                    revision = _revision(glob_scope)
                    record = PassExecutionResult(
                        pass_id,
                        current_unit,
                        False,
                        False,
                        revision,
                        revision,
                        (diagnostic, *local_diagnostics),
                    )
            finally:
                _call_instrumentation(
                    end,
                    "air.instrumentation.end-failed",
                    f"end callback for {pass_id}",
                    local_diagnostics,
                    pass_id,
                    record,
                )

            if local_diagnostics and record is not None:
                record = type(record)(
                    **{
                        **record.__dict__,
                        "diagnostics": tuple(
                            dict.fromkeys((*record.diagnostics, *local_diagnostics))
                        ),
                    }
                )
            executions.append(record)
            for dependency_record in analyses.dependency_trace[dependency_start:]:
                diagnostics.extend(dependency_record.diagnostics)
            diagnostics.extend(record.diagnostics)
            if not record.success:
                failed_at = index
                break

        unexecuted = () if failed_at is None else tuple(
            _safe_step_id(step) for step in steps[failed_at + 1 :]
        )
        return PassManagerResult(
            failed_at is None,
            changed,
            tuple(executions),
            analyses.dependency_trace,
            unexecuted,
            tuple(dict.fromkeys(diagnostics)),
        )

    def _run_transform(
        self,
        glob_scope,
        transform,
        current_unit,
        module_id,
        committed_view,
        analyses,
        user_context,
        instrumentation_diagnostics,
    ):
        fallback_pass_id = (
            f"{transform.__class__.__module__}."
            f"{transform.__class__.__qualname__}"
        )
        before = _revision(glob_scope)
        try:
            pass_id = pass_instance_id(transform)
            configured_identity = pass_identity(transform)
        except Exception as exc:
            return PassExecutionResult(
                fallback_pass_id,
                current_unit,
                False,
                False,
                before,
                before,
                (
                    error(
                        "air.pass.invalid-configuration",
                        f"{fallback_pass_id} has invalid configuration: {exc}",
                    ),
                ),
            )
        try:
            requests = transform.required_analyses(current_unit)
        except Exception as exc:
            return PassExecutionResult(
                pass_id,
                current_unit,
                False,
                False,
                before,
                before,
                (
                    error(
                        "air.pass.required-analyses-exception",
                        f"{pass_id} required_analyses raised {type(exc).__name__}: {exc}",
                    ),
                    *instrumentation_diagnostics,
                ),
            )
        if not isinstance(requests, tuple) or not all(
            isinstance(request, AnalysisRequest) for request in requests
        ):
            return PassExecutionResult(
                pass_id,
                current_unit,
                False,
                False,
                before,
                before,
                (
                    error(
                        "air.pass.invalid-required-analyses",
                        f"{pass_id} required_analyses must return a tuple of AnalysisRequest values",
                    ),
                    *instrumentation_diagnostics,
                ),
            )

        declared = {}
        for request in requests:
            resolution = analyses.resolve(
                request,
                current_unit,
                before,
                lambda unit: committed_view
                if unit.kind == "module"
                else committed_view.function(unit.native_id),
                user_context,
                dependency=True,
            )
            if not resolution.result.success:
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        error(
                            "air.pass.required-analysis-failed",
                            f"required analysis {_step_id(request)} failed for {pass_id}",
                        ),
                        *resolution.result.diagnostics,
                        *instrumentation_diagnostics,
                    ),
                )
            declared[(pass_identity(request.analysis), resolution.key.unit)] = (
                resolution.result
            )

        checkpoint = analyses.checkpoint()
        try:
            transaction = glob_scope.begin_air_pass_transaction()
        except Exception as exc:
            return PassExecutionResult(
                pass_id,
                current_unit,
                False,
                False,
                before,
                before,
                (
                    error(
                        "air.transaction.open-failed",
                        f"{pass_id} could not open transaction: {type(exc).__name__}: {exc}",
                    ),
                    *instrumentation_diagnostics,
                ),
            )

        def rollback_diagnostics():
            try:
                transaction.rollback()
            except Exception as exc:
                return (
                    error(
                        "air.transaction.rollback-failed",
                        f"{pass_id} rollback raised {type(exc).__name__}: {exc}",
                    ),
                )
            return ()

        try:
            try:
                edit_module = _module_view(
                    glob_scope, module_id, transaction=transaction
                )
                edit_unit = (
                    edit_module
                    if current_unit.kind == "module"
                    else edit_module.function(
                        transaction.remap_native_id(
                            "function", None, current_unit.native_id
                        )
                    )
                )
                pass_context = TransformationContext(
                    analyses,
                    declared,
                    current_unit,
                    before,
                    edit_module.source_to_candidate,
                    user_context,
                )
            except Exception as exc:
                analyses.restore(checkpoint)
                rollback_errors = rollback_diagnostics()
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        error(
                            "air.transaction.setup-failed",
                            f"{pass_id} could not prepare candidate views: {type(exc).__name__}: {exc}",
                        ),
                        *rollback_errors,
                        *instrumentation_diagnostics,
                    ),
                )

            try:
                raw_result = transform.run(edit_unit, pass_context)
            except Exception as exc:
                result = TransformResult(
                    False,
                    (
                        error(
                            "air.pass.exception",
                            f"{pass_id} raised {type(exc).__name__}: {exc}",
                        ),
                    ),
                )
            else:
                if isinstance(raw_result, TransformResult):
                    result = raw_result
                else:
                    result = TransformResult(
                        False,
                        (
                            error(
                                "air.pass.invalid-result",
                                f"{pass_id} returned {type(raw_result).__name__}, expected TransformResult",
                            ),
                        ),
                    )

            try:
                current_identity = pass_identity(transform)
                current_pass_id = pass_instance_id(transform)
            except Exception as exc:
                result = TransformResult(
                    False,
                    (
                        error(
                            "air.pass.invalid-configuration",
                            f"{pass_id} has invalid configuration: {exc}",
                        ),
                    ),
                )
            else:
                if (
                    current_identity != configured_identity
                    or current_pass_id != pass_id
                ):
                    result = TransformResult(
                        False,
                        (
                            error(
                                "air.pass.mutated-configuration",
                                f"{pass_id} mutated its configuration while running",
                            ),
                        ),
                    )

            if not result.success:
                rollback_errors = rollback_diagnostics()
                analyses.restore(checkpoint)
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        *result.diagnostics,
                        *rollback_errors,
                        *instrumentation_diagnostics,
                    ),
                    result.metrics,
                )

            dirty = bool(transaction.dirty)
            if not dirty:
                rollback_errors = rollback_diagnostics()
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    not rollback_errors,
                    False,
                    before,
                    before,
                    (
                        *result.diagnostics,
                        *rollback_errors,
                        *instrumentation_diagnostics,
                    ),
                    result.metrics,
                )

            preservation_errors = analyses.validate_preservation(
                result.preserved_analyses, before
            )
            if preservation_errors:
                rollback_errors = rollback_diagnostics()
                analyses.restore(checkpoint)
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        *result.diagnostics,
                        *preservation_errors,
                        *rollback_errors,
                        *instrumentation_diagnostics,
                    ),
                    result.metrics,
                )

            try:
                committed = transaction.commit()
            except Exception as exc:
                analyses.restore(checkpoint)
                rollback_errors = rollback_diagnostics()
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        *result.diagnostics,
                        error(
                            "air.transaction.commit-failed",
                            f"{pass_id} commit failed: {type(exc).__name__}: {exc}",
                        ),
                        *rollback_errors,
                        *instrumentation_diagnostics,
                    ),
                    result.metrics,
                )
            if not committed:
                analyses.restore(checkpoint)
                return PassExecutionResult(
                    pass_id,
                    current_unit,
                    False,
                    False,
                    before,
                    before,
                    (
                        *result.diagnostics,
                        error(
                            "air.transaction.verification-failed",
                            f"{pass_id} candidate failed native AIR verification",
                        ),
                        *instrumentation_diagnostics,
                    ),
                    result.metrics,
                )
            after = _revision(glob_scope)
            if after != before + 1:
                raise RuntimeError(
                    f"transaction revision advanced from {before} to {after}, expected {before + 1}"
                )
            analyses.advance_revision(before, after, result.preserved_analyses)
            return PassExecutionResult(
                pass_id,
                current_unit,
                True,
                True,
                before,
                after,
                (*result.diagnostics, *instrumentation_diagnostics),
                result.metrics,
            )
        finally:
            if transaction.active:
                try:
                    transaction.rollback()
                except Exception:
                    pass
