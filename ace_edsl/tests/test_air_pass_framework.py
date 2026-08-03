"""Pure contracts for the ordered Python AIR pass manager."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass
import json
import os
import subprocess
import sys

import pytest

import ace_edsl.edsl.passes.framework.manager as manager_module

from ace_edsl.edsl.passes.framework import (
    AIRPassManager,
    AIRObjectId,
    AIRLocation,
    AnalysisPass,
    AnalysisRequest,
    AnalysisResult,
    CURRENT_MODULE,
    CURRENT_PASS_UNIT,
    Diagnostic,
    DiagnosticSeverity,
    IRUnitRef,
    ModuleView,
    PassInstrumentation,
    PreservedAnalyses,
    TransformResult,
    TransformationPass,
)
from ace_edsl.edsl.passes.framework.api import (
    analysis_success,
    transform_failure,
    transform_success,
)
from ace_edsl.edsl.passes.framework.diagnostics import error
from ace_edsl.edsl.passes.analyses import (
    CallGraphAnalysis,
    FunctionReachabilityAnalysis,
)


def _key(kind, owner, native_id):
    return kind, owner, native_id


def _snapshot(revision, pointer, locals_=()):
    module_key = _key("module", None, pointer // 1000)
    type_key = _key("type", None, 1)
    function_key = _key("function", None, 7)
    second_function_key = _key("function", None, 8)
    local_keys = tuple(_key("local", 7, native_id) for native_id in locals_)
    objects = [
        {"key": module_key},
        {"key": type_key, "name": "int32_t", "type_kind": "primitive"},
        {
            "key": function_key,
            "name": "main",
            "entries": (),
            "formals": (),
            "locals": local_keys,
            "symbols": (),
            "pregs": (),
            "blocks": (),
            "entry_block": None,
        },
        {
            "key": second_function_key,
            "name": "worker",
            "entries": (),
            "formals": (),
            "locals": (),
            "symbols": (),
            "pregs": (),
            "blocks": (),
            "entry_block": None,
        },
    ]
    objects.extend(
        {
            "key": key,
            "name": f"local_{key[2]}",
            "type": type_key,
        }
        for key in local_keys
    )
    return {
        "revision": revision,
        "native_pointer": pointer,
        "module": module_key,
        "types": (type_key,),
        "constants": (),
        "entries": (),
        "functions": (function_key, second_function_key),
        "objects": tuple(objects),
        "structural_order": tuple(item["key"] for item in objects),
    }


class _FakeTransaction:
    def __init__(self, owner):
        self.owner = owner
        self.active = True
        self.dirty = False
        self.air_pass_generation = owner.air_pass_generation + 1
        self.locals = list(owner.locals)
        self._next_local = max(self.locals, default=99) + 1

    @property
    def source_to_candidate(self):
        order = self.air_pass_snapshot()["structural_order"]
        return tuple((key, key) for key in order if key[2] < 100)

    def air_pass_snapshot(self):
        self._require_active()
        return _snapshot(
            self.owner.air_pass_revision,
            self.owner.get_native_ptr() + 1,
            self.locals,
        )

    def remap_native_id(self, kind, owner, native_id):
        self._require_active()
        key = _key(kind, owner, native_id)
        if key not in dict(self.source_to_candidate):
            raise ValueError("unmapped or wrong-kind AIR ID")
        return native_id

    def create_local(self, function_id, name, type_id):
        self._require_active()
        assert function_id == 7 and type_id == 1
        native_id = self._next_local
        self._next_local += 1
        self.locals.append(native_id)
        self.dirty = True
        return _key("local", function_id, native_id)

    def verify(self):
        self._require_active()
        return self.owner.verify_candidates

    def commit(self):
        self._require_active()
        if not self.owner.verify_candidates:
            self.rollback()
            return False
        self.owner.locals = tuple(self.locals)
        self.owner.air_pass_revision += 1
        self.owner.air_pass_generation += 1
        self.owner.pointer += 1
        self.active = False
        return True

    def rollback(self):
        if self.active:
            self.active = False
            self.air_pass_generation += 1

    def _require_active(self):
        if not self.active:
            raise RuntimeError("AIR pass transaction is inactive")


class _FakeGlob:
    _next_module = 1

    def __init__(self):
        self.air_pass_module_id = _FakeGlob._next_module
        _FakeGlob._next_module += 1
        self.air_pass_revision = 0
        self.air_pass_generation = 0
        self.pointer = self.air_pass_module_id * 1000
        self.locals = ()
        self.verify_candidates = True

    def get_native_ptr(self):
        return self.pointer

    def air_pass_snapshot(self):
        return _snapshot(self.air_pass_revision, self.pointer, self.locals)

    def begin_air_pass_transaction(self):
        return _FakeTransaction(self)

    def dump(self):
        return repr((self.air_pass_revision, self.locals))


class _CountingAnalysis(AnalysisPass):
    def __init__(self, name, counts, *, dependency=None, safe=False, value=None):
        self.name = name
        self._counts = counts
        self._dependency = dependency
        self._safe = safe
        self._value = name if value is None else value

    @property
    def identity(self):
        return self.name

    @property
    def pass_id(self):
        return f"analysis.{self.name}"

    @property
    def cross_revision_safe(self):
        return self._safe

    def dependencies(self, unit):
        return (
            (AnalysisRequest(self._dependency, CURRENT_MODULE),)
            if self._dependency
            else ()
        )

    def run(self, unit, context):
        self._counts[self.name] = self._counts.get(self.name, 0) + 1
        return analysis_success(self._value, metrics={"runs": 1})


class _Change(TransformationPass):
    def __init__(
        self,
        pass_id,
        *,
        required=(),
        preserved=None,
        outcome="success",
        ran=None,
        escaped=None,
    ):
        self.pass_id = pass_id
        self._required = tuple(required)
        self._preserved = preserved or PreservedAnalyses.none()
        self._outcome = outcome
        self._ran = ran
        self._escaped = escaped

    def required_analyses(self, unit):
        return tuple(
            AnalysisRequest(analysis, CURRENT_MODULE)
            for analysis in self._required
        )

    def run(self, unit, context):
        if self._ran is not None:
            self._ran.append(self.pass_id)
        local = unit.functions[0].create_local(
            self.pass_id, unit.types[0]
        )
        if self._escaped is not None:
            self._escaped.append(local)
        if self._outcome == "raise":
            raise RuntimeError("expected failure")
        if self._outcome == "failure":
            return transform_failure(error("test.failure", "expected failure"))
        return transform_success(
            metrics={"edits": 1}, preserved=self._preserved
        )


class _NoOp(TransformationPass):
    def __init__(self, pass_id="noop", ran=None):
        self.pass_id = pass_id
        self._ran = ran

    def run(self, unit, context):
        if self._ran is not None:
            self._ran.append(self.pass_id)
        return transform_success()


def _capture_restored_analysis_managers(monkeypatch):
    restored = []
    base = manager_module.AnalysisManager

    class RecordingAnalysisManager(base):
        def restore(self, checkpoint):
            super().restore(checkpoint)
            restored.append(self)

    monkeypatch.setattr(manager_module, "AnalysisManager", RecordingAnalysisManager)
    return restored


def test_result_contracts_are_immutable_and_do_not_accept_changed_or_revisions():
    analysis = AnalysisResult(True, ("host",), metrics={"nested": [1, 2]})
    assert analysis.metrics["nested"] == (1, 2)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        analysis.success = False
    with pytest.raises(ValueError, match="cannot carry a value"):
        AnalysisResult(False, "value")
    with pytest.raises(TypeError):
        AnalysisResult(True, (), changed=False)
    with pytest.raises(TypeError):
        TransformResult(True, changed=False)
    with pytest.raises(TypeError):
        TransformResult(True, before_revision=0)
    transform = TransformResult(True)
    managed = AIRPassManager().run(_FakeGlob(), (_NoOp(),))
    for result in (transform, managed.executions[0], managed):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.success = False


def test_analysis_result_rejects_live_air_views_without_caching_them():
    class ReturnsLiveView(AnalysisPass):
        pass_id = "analysis.returns-live-view"

        def run(self, unit, context):
            return analysis_success(unit)

    result = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(ReturnsLiveView()),)
    )
    assert not result.success
    assert result.executions[0].analysis_result.value is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "air.analysis.mutable-result"
    ]


def test_dependency_cache_order_explicit_hits_and_silence(capsys):
    counts = {}
    first = _CountingAnalysis("first", counts)
    second = _CountingAnalysis("second", counts, dependency=first)
    result = AIRPassManager().run(
        _FakeGlob(),
        (
            AnalysisRequest(second, CURRENT_MODULE),
            AnalysisRequest(first, CURRENT_MODULE),
            AnalysisRequest(second, CURRENT_MODULE),
        ),
    )
    assert result.success and not result.changed
    assert counts == {"first": 1, "second": 1}
    assert [record.pass_id for record in result.dependency_trace] == [
        "analysis.first"
    ]
    assert [record.cached for record in result.executions] == [False, True, True]
    assert capsys.readouterr() == ("", "")


def test_analysis_configuration_is_part_of_the_cache_identity():
    runs = {}

    @dataclass(frozen=True)
    class ConfiguredAnalysis(AnalysisPass):
        mode: str

        def run(self, unit, context):
            runs[self.mode] = runs.get(self.mode, 0) + 1
            return analysis_success(self.mode)

    first = ConfiguredAnalysis("a")
    second = ConfiguredAnalysis("b")
    result = AIRPassManager().run(
        _FakeGlob(),
        tuple(
            AnalysisRequest(analysis, CURRENT_MODULE)
            for analysis in (first, second, first, second)
        ),
    )
    assert result.success and runs == {"a": 1, "b": 1}
    assert [record.cached for record in result.executions] == [
        False,
        False,
        True,
        True,
    ]
    assert [record.analysis_result.value for record in result.executions] == [
        "a",
        "b",
        "a",
        "b",
    ]


def test_dependency_cycle_is_deterministic_and_fail_fast():
    class Cycle(AnalysisPass):
        def __init__(self, name):
            self.name = name
            self.other = None

        @property
        def identity(self):
            return self.name

        @property
        def pass_id(self):
            return f"cycle.{self.name}"

        def dependencies(self, unit):
            return (AnalysisRequest(self.other, CURRENT_MODULE),)

        def run(self, unit, context):
            raise AssertionError("cycle members must not run")

    first, second = Cycle("first"), Cycle("second")
    first.other, second.other = second, first
    result = AIRPassManager().run(
        _FakeGlob(),
        (AnalysisRequest(first), _NoOp("unreached")),
    )
    assert not result.success
    assert result.unexecuted_step_ids == ("unreached",)
    cycle = next(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "air.analysis.dependency-cycle"
    )
    assert cycle.message == (
        "analysis dependency cycle: "
        "cycle.first -> cycle.second -> cycle.first"
    )


def test_safe_preservation_rekeys_dependency_closed_results_once():
    counts = {}
    first = _CountingAnalysis("first", counts, safe=True)
    second = _CountingAnalysis(
        "second", counts, dependency=first, safe=True
    )
    transform = _Change(
        "change",
        required=(second,),
        preserved=PreservedAnalyses.of(first, second),
    )
    glob = _FakeGlob()
    result = AIRPassManager().run(
        glob,
        (transform, AnalysisRequest(second, CURRENT_MODULE)),
    )
    assert result.success and result.changed
    assert counts == {"first": 1, "second": 1}
    assert result.executions[0].before_revision == 0
    assert result.executions[0].after_revision == 1
    assert result.executions[1].cached
    assert glob.air_pass_revision == 1


@pytest.mark.parametrize("preserve_dependency", [False, True])
def test_unsafe_or_non_closed_preservation_rolls_back(preserve_dependency):
    counts = {}
    dependency = _CountingAnalysis("dependency", counts, safe=True)
    if preserve_dependency:
        requested = _CountingAnalysis("unsafe", counts, safe=False)
        preserved = PreservedAnalyses.of(requested)
    else:
        requested = _CountingAnalysis(
            "dependent", counts, dependency=dependency, safe=True
        )
        preserved = PreservedAnalyses.of(requested)
    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(
        glob,
        (_Change("bad-preserve", required=(requested,), preserved=preserved),),
    )
    assert not result.success and not result.changed
    assert glob.air_pass_revision == 0
    assert glob.get_native_ptr() == pointer
    expected = (
        "air.analysis.unsafe-preservation"
        if preserve_dependency
        else "air.analysis.preservation-not-closed"
    )
    assert any(diagnostic.code == expected for diagnostic in result.diagnostics)


def test_failure_restores_checkpoint_and_candidate_views_are_stale(monkeypatch):
    restored = _capture_restored_analysis_managers(monkeypatch)
    counts = {}
    analysis = _CountingAnalysis("needed", counts)
    escaped = []
    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(
        glob,
        (
            _Change(
                "fails",
                required=(analysis,),
                outcome="raise",
                escaped=escaped,
            ),
            _NoOp("later"),
        ),
    )
    assert not result.success
    assert counts == {"needed": 1}
    assert [record.pass_id for record in result.dependency_trace] == [
        "analysis.needed"
    ]
    assert result.unexecuted_step_ids == ("later",)
    assert glob.locals == () and glob.air_pass_revision == 0
    assert glob.get_native_ptr() == pointer
    cached = restored[-1].resolve(
        AnalysisRequest(analysis),
        restored[-1].module_unit,
        glob.air_pass_revision,
        lambda unit: ModuleView(glob, restored[-1].module_id),
    )
    assert cached.cached and counts == {"needed": 1}
    with pytest.raises(RuntimeError, match="stale AIR transaction view"):
        _ = escaped[0].native_id


def test_three_pass_atomicity_commits_a_rolls_back_b_and_skips_c(monkeypatch):
    restored = _capture_restored_analysis_managers(monkeypatch)
    ran = []
    counts = {}
    retained = _CountingAnalysis("retained", counts, safe=True)
    preservation = PreservedAnalyses.of(retained)
    glob = _FakeGlob()
    result = AIRPassManager().run(
        glob,
        (
            _Change(
                "a",
                required=(retained,),
                preserved=preservation,
                ran=ran,
            ),
            _Change("b", required=(retained,), outcome="failure", ran=ran),
            _NoOp("c", ran=ran),
        ),
    )
    assert not result.success and result.changed
    assert ran == ["a", "b"]
    assert result.unexecuted_step_ids == ("c",)
    assert glob.air_pass_revision == 1 and len(glob.locals) == 1
    cached = restored[-1].resolve(
        AnalysisRequest(retained),
        restored[-1].module_unit,
        glob.air_pass_revision,
        lambda unit: ModuleView(glob, restored[-1].module_id),
    )
    assert cached.cached and counts == {"retained": 1}
    assert [(record.changed, record.before_revision, record.after_revision)
            for record in result.executions] == [
        (True, 0, 1),
        (False, 1, 1),
    ]


def test_noop_preserves_pointer_revision_views_and_cache():
    counts = {}
    analysis = _CountingAnalysis("cached", counts)
    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(
        glob,
        (
            AnalysisRequest(analysis),
            _NoOp(),
            AnalysisRequest(analysis),
        ),
    )
    assert result.success and not result.changed
    assert counts == {"cached": 1}
    assert result.executions[2].cached
    assert glob.get_native_ptr() == pointer and glob.air_pass_revision == 0


def test_read_only_analysis_cannot_edit_and_its_result_remains_cached():
    runs = []

    class AttemptsEdit(AnalysisPass):
        pass_id = "analysis.attempts-edit"

        def run(self, unit, context):
            runs.append(unit.unit_ref)
            with pytest.raises(AttributeError):
                unit.functions[0].create_local("forbidden", unit.types[0])
            return analysis_success(("read-only", unit.revision))

    analysis = AttemptsEdit()
    glob = _FakeGlob()
    before = (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision)
    result = AIRPassManager().run(
        glob,
        (
            AnalysisRequest(analysis, CURRENT_MODULE),
            _NoOp(),
            AnalysisRequest(analysis, CURRENT_MODULE),
        ),
    )
    assert result.success
    assert [result.executions[index].cached for index in (0, 2)] == [
        False,
        True,
    ]
    assert len(runs) == 1
    assert (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision) == before


def test_relative_units_are_invocation_local_and_foreign_refs_are_rejected():
    counts = {}
    analysis = _CountingAnalysis("relative", counts)
    first = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(analysis, CURRENT_MODULE),)
    )
    second = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(analysis, CURRENT_MODULE),)
    )
    assert first.success and second.success
    assert first.executions[0].unit.module_id != second.executions[0].unit.module_id
    foreign = first.executions[0].unit
    rejected = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(analysis, foreign),)
    )
    assert not rejected.success
    assert rejected.diagnostics[0].code == "air.analysis.invalid-unit"


def test_cross_revision_safe_analysis_cannot_return_air_ids():
    counts = {}
    analysis = _CountingAnalysis(
        "bad-safe",
        counts,
        safe=True,
        value=IRUnitRef("module", "host"),
    )
    result = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(analysis),)
    )
    assert not result.success
    assert result.diagnostics[0].code == "air.analysis.unsafe-result-contract"


@pytest.mark.parametrize("identifier_site", ["metrics", "location"])
def test_cross_revision_safe_analysis_rejects_all_revision_bound_metadata(
    identifier_site,
):
    class BadSafeMetadata(AnalysisPass):
        pass_id = f"analysis.bad-safe-{identifier_site}"
        cross_revision_safe = True

        def run(self, unit, context):
            if identifier_site == "metrics":
                return analysis_success(
                    (), metrics={"function": unit.functions[0].id}
                )
            return analysis_success(
                (),
                diagnostics=(
                    error(
                        "test.located",
                        "located",
                        location=AIRLocation(
                            unit.unit_ref.module_id,
                            function_id=unit.functions[0].native_id,
                        ),
                    ),
                ),
            )

    result = AIRPassManager().run(
        _FakeGlob(), (AnalysisRequest(BadSafeMetadata()),)
    )
    assert not result.success
    assert result.diagnostics[0].code == "air.analysis.unsafe-result-contract"


def test_cached_analysis_safety_cannot_be_mutated_to_bypass_preservation():
    class MutableSafety(AnalysisPass):
        pass_id = "analysis.mutable-safety"
        identity = "stable"
        cross_revision_safe = False

        def run(self, unit, context):
            return analysis_success(unit.functions[0].id)

    analysis = MutableSafety()

    class FlipsSafety(TransformationPass):
        pass_id = "transform.flips-analysis-safety"

        def required_analyses(self, current_unit):
            return (AnalysisRequest(analysis, CURRENT_MODULE),)

        def run(self, unit, context):
            analysis.cross_revision_safe = True
            unit.functions[0].create_local("must-rollback", unit.types[0])
            return transform_success(
                preserved=PreservedAnalyses.of(analysis)
            )

    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(glob, (FlipsSafety(),))
    assert not result.success and not result.changed
    assert result.diagnostics[0].code == "air.pass.mutated-configuration"
    assert glob.locals == () and glob.air_pass_revision == 0
    assert glob.get_native_ptr() == pointer


def test_instrumentation_is_balanced_observational_and_has_no_timing_field():
    events = []

    def begin(pass_id, unit):
        events.append(("begin", pass_id))
        raise RuntimeError("begin observer")

    def dump(pass_id, ir):
        events.append(("dump", pass_id))
        raise RuntimeError("dump observer")

    def end(pass_id, record):
        events.append(("end", pass_id))
        raise RuntimeError("end observer")

    instrumentation = PassInstrumentation(begin, end, dump)
    result = AIRPassManager().run(
        _FakeGlob(),
        (_NoOp("observed"),),
        {"instrumentation": instrumentation},
    )
    assert result.success and not result.changed
    assert events == [
        ("begin", "observed"),
        ("dump", "observed"),
        ("end", "observed"),
    ]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "air.instrumentation.begin-failed",
        "air.instrumentation.dump-failed",
        "air.instrumentation.end-failed",
    ]
    assert not hasattr(result.executions[0], "elapsed")


class _FunctionAnalysis(AnalysisPass):
    pass_id = "analysis.function"

    def __init__(self, counts):
        self._counts = counts

    @property
    def identity(self):
        return "function"

    def run(self, unit, context):
        native_id = unit.native_id
        self._counts[native_id] = self._counts.get(native_id, 0) + 1
        return analysis_success(native_id)


class _DependentFunctionAnalysis(AnalysisPass):
    pass_id = "analysis.dependent-function"

    def __init__(self, base, counts):
        self._base = base
        self._counts = counts

    @property
    def identity(self):
        return "dependent-function"

    def dependencies(self, unit):
        return (AnalysisRequest(self._base, CURRENT_PASS_UNIT),)

    def run(self, unit, context):
        value = context.result(
            AnalysisRequest(self._base, CURRENT_PASS_UNIT), unit.unit_ref
        ).value
        self._counts[value] = self._counts.get(value, 0) + 1
        return analysis_success(value)


class _FunctionConsumer(TransformationPass):
    pass_id = "transform.function-consumer"

    def __init__(self, analysis):
        self._analysis = analysis

    def _requests(self, module_id):
        return tuple(
            AnalysisRequest(
                self._analysis, IRUnitRef("function", module_id, native_id)
            )
            for native_id in (7, 8)
        )

    def required_analyses(self, current_unit):
        return self._requests(current_unit.module_id)

    def run(self, unit, context):
        assert tuple(
            context.result(request).value
            for request in self._requests(unit.unit_ref.module_id)
        ) == (7, 8)
        unit.functions[0].create_local("changed", unit.types[0])
        return transform_success()


def test_function_unit_cache_entries_and_dependents_invalidate_transitively():
    base_counts = {}
    dependent_counts = {}
    base = _FunctionAnalysis(base_counts)
    analysis = _DependentFunctionAnalysis(base, dependent_counts)
    glob = _FakeGlob()
    result = AIRPassManager().run(
        glob, (_FunctionConsumer(analysis), _FunctionConsumer(analysis))
    )
    assert result.success and result.changed
    assert base_counts == dependent_counts == {7: 2, 8: 2}
    assert glob.air_pass_revision == 2
    assert [record.unit.native_id for record in result.dependency_trace] == [
        7,
        7,
        8,
        8,
        7,
        7,
        8,
        8,
    ]
    assert [record.pass_id for record in result.dependency_trace] == [
        "analysis.function",
        "analysis.dependent-function",
    ] * 4

    other = AIRPassManager().run(
        _FakeGlob(), (_FunctionConsumer(analysis),)
    )
    assert other.success
    assert other.dependency_trace[0].unit.module_id != (
        result.dependency_trace[0].unit.module_id
    )


class _DescriptorAnalysis(AnalysisPass):
    pass_id = "analysis.descriptor"

    def run(self, unit, context):
        return analysis_success(unit.functions[0].id)


class _UseDeclaredDescriptor(TransformationPass):
    pass_id = "transform.use-declared-descriptor"
    analysis = _DescriptorAnalysis()

    def required_analyses(self, current_unit):
        return (AnalysisRequest(self.analysis, CURRENT_MODULE),)

    def run(self, unit, context):
        source_id = context.result(
            AnalysisRequest(_DescriptorAnalysis(), CURRENT_MODULE)
        ).value
        candidate_id = context.remap(source_id)
        assert candidate_id.kind == "function"
        assert unit.lookup(candidate_id).name == "main"
        return transform_success()


def test_declared_stable_ids_remap_explicitly_to_candidate():
    result = AIRPassManager().run(_FakeGlob(), (_UseDeclaredDescriptor(),))
    assert result.success and not result.changed


class _UndeclaredAfterOpen(TransformationPass):
    pass_id = "transform.undeclared-after-open"

    def run(self, unit, context):
        context.result(AnalysisRequest(_DescriptorAnalysis(), CURRENT_MODULE))
        raise AssertionError("unreachable")


def test_undeclared_analysis_after_transaction_open_fails_before_mutation():
    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(glob, (_UndeclaredAfterOpen(),))
    assert not result.success and not result.changed
    assert glob.locals == () and glob.get_native_ptr() == pointer
    assert "undeclared analysis requested after transaction opened" in (
        result.diagnostics[0].message
    )


class _BadAnalysis(AnalysisPass):
    def __init__(self, behavior):
        self.behavior = behavior
        self.pass_id = f"analysis.bad-{behavior}"

    def run(self, unit, context):
        if self.behavior == "raise":
            raise LookupError("analysis boom")
        if self.behavior == "mutable":
            return AnalysisResult(True, [])
        return {"not": "an AnalysisResult"}


@pytest.mark.parametrize(
    ("behavior", "code"),
    [
        ("raise", "air.analysis.exception"),
        ("mutable", "air.analysis.mutable-result"),
        ("invalid", "air.analysis.invalid-result"),
    ],
)
def test_analysis_exceptions_mutable_values_and_invalid_results_are_attributed(
    behavior, code
):
    result = AIRPassManager().run(
        _FakeGlob(),
        (AnalysisRequest(_BadAnalysis(behavior)), _NoOp("not-run")),
    )
    assert not result.success
    assert result.unexecuted_step_ids == ("not-run",)
    assert any(diagnostic.code == code for diagnostic in result.diagnostics)


class _InvalidTransformResult(TransformationPass):
    pass_id = "transform.invalid-result"

    def run(self, unit, context):
        return object()


def test_invalid_transform_result_and_verifier_failure_are_transactional():
    invalid_glob = _FakeGlob()
    invalid_pointer = invalid_glob.get_native_ptr()
    invalid = AIRPassManager().run(invalid_glob, (_InvalidTransformResult(),))
    assert not invalid.success
    assert invalid_glob.get_native_ptr() == invalid_pointer
    assert invalid.diagnostics[0].code == "air.pass.invalid-result"

    verifier_glob = _FakeGlob()
    verifier_glob.verify_candidates = False
    verifier_pointer = verifier_glob.get_native_ptr()
    verifier = AIRPassManager().run(verifier_glob, (_Change("bad-air"),))
    assert not verifier.success
    assert verifier_glob.get_native_ptr() == verifier_pointer
    assert verifier_glob.air_pass_revision == 0 and verifier_glob.locals == ()
    assert verifier.diagnostics[0].code == "air.transaction.verification-failed"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("preservation", "air.analysis.unsafe-preservation"),
        ("verification", "air.transaction.verification-failed"),
        ("commit", "air.transaction.commit-failed"),
    ],
)
def test_transform_diagnostics_survive_post_run_transaction_failures(
    failure, expected_code
):
    counts = {}
    analysis = _CountingAnalysis("unsafe-diagnostic", counts)

    class DiagnosedChange(TransformationPass):
        pass_id = f"transform.diagnosed-{failure}"

        def required_analyses(self, current_unit):
            return (
                (AnalysisRequest(analysis, CURRENT_MODULE),)
                if failure == "preservation"
                else ()
            )

        def run(self, unit, context):
            unit.functions[0].create_local("rolled-back", unit.types[0])
            return transform_success(
                diagnostics=(
                    Diagnostic(
                        DiagnosticSeverity.INFO,
                        "test.before-transaction-failure",
                        "keep this diagnostic",
                    ),
                ),
                preserved=(
                    PreservedAnalyses.of(analysis)
                    if failure == "preservation"
                    else PreservedAnalyses.none()
                ),
            )

    class CommitFailureTransaction(_FakeTransaction):
        def commit(self):
            self._require_active()
            raise RuntimeError("commit boom")

    class CommitFailureGlob(_FakeGlob):
        def begin_air_pass_transaction(self):
            return CommitFailureTransaction(self)

    glob = CommitFailureGlob() if failure == "commit" else _FakeGlob()
    if failure == "verification":
        glob.verify_candidates = False
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(glob, (DiagnosedChange(),))
    assert not result.success and not result.changed
    assert [diagnostic.code for diagnostic in result.diagnostics[:2]] == [
        "test.before-transaction-failure",
        expected_code,
    ]
    assert glob.locals == () and glob.air_pass_revision == 0
    assert glob.get_native_ptr() == pointer


class _MutatesConfiguration(TransformationPass):
    pass_id = "transform.mutates-configuration"

    def __init__(self):
        self.mode = "before"

    def run(self, unit, context):
        unit.functions[0].create_local("must-rollback", unit.types[0])
        self.mode = "after"
        return transform_success()


def test_transform_configuration_mutation_is_a_precommit_contract_failure():
    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(glob, (_MutatesConfiguration(),))
    assert not result.success and not result.changed
    assert glob.locals == () and glob.get_native_ptr() == pointer
    assert result.diagnostics[0].code == "air.pass.mutated-configuration"


def test_transform_pass_id_mutation_is_a_precommit_contract_failure():
    class MutatesPassId(TransformationPass):
        identity = "stable"
        pass_id = "transform.before"

        def run(self, unit, context):
            unit.functions[0].create_local("must-rollback", unit.types[0])
            self.pass_id = "transform.after"
            return transform_success()

    glob = _FakeGlob()
    pointer = glob.get_native_ptr()
    result = AIRPassManager().run(glob, (MutatesPassId(),))
    assert not result.success and not result.changed
    assert result.executions[0].pass_id == "transform.before"
    assert result.diagnostics[0].code == "air.pass.mutated-configuration"
    assert glob.locals == () and glob.get_native_ptr() == pointer


def test_successful_pass_order_preserves_distinct_diagnostics_and_metrics():
    class Annotated(TransformationPass):
        def __init__(self, name, edit):
            self.name = name
            self.edit = edit

        @property
        def pass_id(self):
            return f"transform.{self.name}"

        def run(self, unit, context):
            if self.edit:
                unit.functions[0].create_local(self.name, unit.types[0])
            return transform_success(
                diagnostics=(
                    Diagnostic(
                        DiagnosticSeverity.INFO,
                        f"test.{self.name}",
                        self.name,
                    ),
                ),
                metrics={"marker": self.name},
            )

    result = AIRPassManager().run(
        _FakeGlob(), (Annotated("first", True), Annotated("second", False))
    )
    assert result.success and result.changed
    assert [record.pass_id for record in result.executions] == [
        "transform.first",
        "transform.second",
    ]
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "test.first",
        "test.second",
    ]
    assert [
        (pass_id, metrics["marker"])
        for pass_id, metrics in result.ordered_metrics
    ] == [
        ("transform.first", "first"),
        ("transform.second", "second"),
    ]


def test_explicit_analysis_records_are_unchanged_and_revision_derived():
    counts = {}
    analysis = _CountingAnalysis("record", counts)
    glob = _FakeGlob()
    result = AIRPassManager().run(
        glob,
        (
            _Change("first"),
            AnalysisRequest(analysis, CURRENT_MODULE),
        ),
    )
    record = result.executions[1]
    assert not record.changed
    assert record.before_revision == record.after_revision == 1


def test_ir_dump_materialization_failure_is_observational():
    class BrokenDump(_FakeGlob):
        def dump(self):
            raise RuntimeError("cannot render")

    events = []
    instrumentation = PassInstrumentation(
        lambda pass_id, unit: events.append(("begin", pass_id)),
        lambda pass_id, record: events.append(("end", pass_id)),
        lambda pass_id, ir: events.append(("dump", pass_id)),
    )
    result = AIRPassManager().run(
        BrokenDump(),
        (_NoOp("observed"),),
        {"instrumentation": instrumentation},
    )
    assert result.success and not result.changed
    assert events == [("begin", "observed"), ("end", "observed")]
    assert result.diagnostics[0].code == "air.instrumentation.dump-failed"


def test_dependency_analysis_instrumentation_is_balanced_and_ordered():
    counts = {}
    analysis = _CountingAnalysis("dependency-observed", counts)

    class NeedsAnalysis(TransformationPass):
        pass_id = "transform.needs-observed-analysis"

        def required_analyses(self, current_unit):
            return (AnalysisRequest(analysis, CURRENT_MODULE),)

        def run(self, unit, context):
            return transform_success()

    events = []
    instrumentation = PassInstrumentation(
        lambda pass_id, unit: events.append(("begin", pass_id)),
        lambda pass_id, record: events.append(("end", pass_id)),
        lambda pass_id, ir: events.append(("dump", pass_id)),
    )
    result = AIRPassManager().run(
        _FakeGlob(),
        (NeedsAnalysis(),),
        {"instrumentation": instrumentation},
    )
    assert result.success
    assert events == [
        ("begin", "transform.needs-observed-analysis"),
        ("begin", "analysis.dependency-observed"),
        ("dump", "analysis.dependency-observed"),
        ("end", "analysis.dependency-observed"),
        ("dump", "transform.needs-observed-analysis"),
        ("end", "transform.needs-observed-analysis"),
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "transform-failure",
        "transform-exception",
        "analysis-exception",
        "required-analysis",
        "verifier",
    ],
)
def test_instrumentation_is_balanced_on_every_pass_failure_path(failure):
    events = []
    active = []

    def begin(pass_id, unit):
        events.append(("begin", pass_id))
        active.append(pass_id)

    def dump(pass_id, ir):
        assert active[-1] == pass_id
        events.append(("dump", pass_id))

    def end(pass_id, record):
        assert active.pop() == pass_id
        events.append(("end", pass_id))

    glob = _FakeGlob()
    if failure == "transform-failure":
        step = _Change("failed", outcome="failure")
    elif failure == "transform-exception":
        step = _Change("raised", outcome="raise")
    elif failure == "analysis-exception":
        step = AnalysisRequest(_BadAnalysis("raise"))
    elif failure == "required-analysis":
        step = _Change("requires", required=(_BadAnalysis("raise"),))
    else:
        step = _Change("unverified")
        glob.verify_candidates = False
    before = (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision)
    result = AIRPassManager().run(
        glob,
        (step, _NoOp("later")),
        {
            "instrumentation": PassInstrumentation(begin, end, dump),
        },
    )
    assert not result.success and result.unexecuted_step_ids == ("later",)
    assert not active
    assert [item for item in events if item[0] == "begin"]
    assert sorted(
        pass_id for kind, pass_id in events if kind == "begin"
    ) == sorted(pass_id for kind, pass_id in events if kind == "dump")
    assert [pass_id for kind, pass_id in events if kind == "begin"] == [
        pass_id for kind, pass_id in reversed(events) if kind == "end"
    ]
    assert (glob.dump(), glob.get_native_ptr(), glob.air_pass_revision) == before
    assert glob.locals == ()


def test_invalid_dependency_identity_is_attributed_and_instrumented_as_child():
    class BadIdentity(AnalysisPass):
        pass_id = "analysis.bad-identity"

        @property
        def identity(self):
            raise LookupError("bad identity")

        def run(self, unit, context):
            raise AssertionError("invalid analysis configuration must not run")

    child = BadIdentity()

    class Parent(AnalysisPass):
        pass_id = "analysis.parent"

        def dependencies(self, unit):
            return (AnalysisRequest(child, CURRENT_PASS_UNIT),)

        def run(self, unit, context):
            raise AssertionError("failed dependency must stop the parent")

    parent = Parent()

    class NeedsParent(TransformationPass):
        pass_id = "transform.needs-parent"

        def required_analyses(self, current_unit):
            return (AnalysisRequest(parent, CURRENT_MODULE),)

        def run(self, unit, context):
            raise AssertionError("failed dependency must stop the transform")

    events = []
    instrumentation = PassInstrumentation(
        lambda pass_id, unit: events.append(("begin", pass_id)),
        lambda pass_id, record: events.append(("end", pass_id)),
        lambda pass_id, ir: events.append(("dump", pass_id)),
    )
    result = AIRPassManager().run(
        _FakeGlob(),
        (NeedsParent(),),
        {"instrumentation": instrumentation},
    )
    assert not result.success
    assert [record.pass_id for record in result.dependency_trace] == [
        "analysis.bad-identity",
        "analysis.parent",
    ]
    assert ("begin", "analysis.bad-identity") in events
    assert ("dump", "analysis.bad-identity") in events
    assert ("end", "analysis.bad-identity") in events
    assert any(
        diagnostic.code == "air.pass.invalid-configuration"
        and "analysis.bad-identity" in diagnostic.message
        for diagnostic in result.diagnostics
    )


def test_dependency_instrumentation_diagnostics_are_not_cached_or_replayed():
    counts = {}
    analysis = _CountingAnalysis("instrumented-cache", counts)

    class NeedsAnalysis(TransformationPass):
        pass_id = "transform.needs-instrumented-cache"

        def required_analyses(self, current_unit):
            return (AnalysisRequest(analysis, CURRENT_MODULE),)

        def run(self, unit, context):
            return transform_success()

    instrumentation = PassInstrumentation(
        lambda pass_id, unit: (
            (_ for _ in ()).throw(RuntimeError("observer"))
            if pass_id == analysis.pass_id
            else None
        )
    )
    result = AIRPassManager().run(
        _FakeGlob(),
        (NeedsAnalysis(), AnalysisRequest(analysis, CURRENT_MODULE)),
        {"instrumentation": instrumentation},
    )
    assert result.success and counts == {"instrumented-cache": 1}
    assert result.executions[1].cached
    assert not result.executions[1].analysis_result.diagnostics
    assert [
        diagnostic.code for diagnostic in result.diagnostics
    ].count("air.instrumentation.begin-failed") == 1


class _GraphGlob:
    def __init__(self, *, indirect=False, reverse_records=False):
        self.air_pass_module_id = 91
        self.air_pass_revision = 0
        self.air_pass_generation = 0
        self._indirect = indirect
        self._reverse_records = reverse_records

    def get_native_ptr(self):
        return 91000

    def dump(self):
        return "graph"

    def air_pass_snapshot(self):
        module = _key("module", None, 91)
        type_key = _key("type", None, 1)
        function_ids = (1, 2, 3, 4, 5, 6, 7)
        function_keys = tuple(_key("function", None, value) for value in function_ids)
        entry_keys = tuple(_key("entry", None, 100 + value) for value in function_ids)
        objects = [
            {"key": module},
            {"key": type_key, "name": "i32", "type_kind": "primitive"},
        ]
        for value, function_key, entry_key in zip(
            function_ids, function_keys, entry_keys
        ):
            block = _key("block", value, 200 + value)
            preg = _key("preg", value, 400 + value)
            calls = {
                1: (2, 3),
                2: (4,),
                3: (4,),
                5: (6,),
                6: (5,),
            }.get(value, ())
            statements = tuple(
                _key("statement", value, 1000 + value * 10 + index)
                for index, _ in enumerate(calls)
            )
            if self._indirect and value == 1:
                statements += (_key("statement", value, 1099),)
            objects.extend(
                (
                    {
                        "key": entry_key,
                        "name": f"entry_{value}",
                        "owning_function": function_key,
                        "program_entry": value in {1, 2},
                        "exported": value == 5,
                    },
                    {
                        "key": function_key,
                        "name": f"f{value}",
                        "entries": (entry_key,),
                        "formals": (),
                        "locals": (),
                        "symbols": (),
                        "pregs": (preg,),
                        "blocks": (block,),
                        "entry_block": block,
                    },
                    {"key": preg, "type": type_key},
                    {"key": block, "statements": statements},
                )
            )
            for index, callee in enumerate(calls):
                statement = statements[index]
                node = _key("node", value, 2000 + value * 10 + index)
                objects.extend(
                    (
                        {"key": statement, "node": node, "parent_block": block},
                        {
                            "key": node,
                            "opcode": "call",
                            "children": (),
                            "arguments": (),
                            "result_preg": preg,
                            "call_target": entry_keys[callee - 1],
                            "indirect_call": False,
                            "parent_statement": statement,
                        },
                    )
                )
            if self._indirect and value == 1:
                statement = statements[-1]
                node = _key("node", value, 2099)
                objects.extend(
                    (
                        {"key": statement, "node": node, "parent_block": block},
                        {
                            "key": node,
                            "opcode": "icall",
                            "children": (),
                            "indirect_call": True,
                            "parent_statement": statement,
                        },
                    )
                )
        constants = (
            _key("constant", None, 301),
            _key("constant", None, 302),
        )
        objects.extend(
            (
                {
                    "key": constants[0],
                    "constant_kind": "entry_ptr",
                    "referenced_entry": entry_keys[2],
                },
                {
                    "key": constants[1],
                    "constant_kind": "entry_func_desc",
                    "referenced_entry": entry_keys[5],
                },
            )
        )
        if self._reverse_records:
            objects.reverse()
        return {
            "revision": 0,
            "native_pointer": self.get_native_ptr(),
            "module": module,
            "types": (type_key,),
            "constants": constants,
            "entries": entry_keys,
            "functions": function_keys,
            "objects": tuple(objects),
            "structural_order": tuple(item["key"] for item in objects),
        }


def test_real_call_graph_and_reachability_clients_cover_roots_sccs_and_order():
    graph = CallGraphAnalysis()
    reachability = FunctionReachabilityAnalysis()
    result = AIRPassManager().run(
        _GraphGlob(),
        (
            AnalysisRequest(graph, CURRENT_MODULE),
            AnalysisRequest(reachability, CURRENT_MODULE),
        ),
    )
    assert result.success
    graph_value = result.executions[0].analysis_result.value
    assert [item.native_id for item in graph_value.functions] == list(range(1, 8))
    assert [item.native_id for item in graph_value.program_roots] == [1, 2]
    assert [item.native_id for item in graph_value.exported_roots] == [5]
    assert [item.native_id for item in graph_value.address_taken_roots] == [3]
    assert [item.native_id for item in graph_value.entry_descriptor_roots] == [6]
    assert [[item.native_id for item in component]
            for component in graph_value.recursive_components] == [[5, 6]]
    reachable = result.executions[1].analysis_result.value
    assert [item.native_id for item in reachable.roots] == [1, 2, 5, 3, 6]
    assert [item.native_id for item in reachable.reachable] == [1, 2, 3, 4, 5, 6]
    assert [item.native_id for item in reachable.unreachable] == [7]
    assert result.executions[1].cached is False
    assert [record.pass_id for record in result.dependency_trace] == []

    module = ModuleView(_GraphGlob(), (91, 77))
    call = module.functions[0].entry_block.statements[0].node
    assert call.call_target.owning_function.native_id == 2
    assert call.arguments == ()
    assert call.result_preg.native_id == module.functions[0].pregs[0].native_id

    reversed_result = AIRPassManager().run(
        _GraphGlob(reverse_records=True),
        (
            AnalysisRequest(CallGraphAnalysis(), CURRENT_MODULE),
            AnalysisRequest(FunctionReachabilityAnalysis(), CURRENT_MODULE),
        ),
    )
    reversed_graph = reversed_result.executions[0].analysis_result.value
    reversed_reachability = reversed_result.executions[1].analysis_result.value
    assert [item.native_id for item in reversed_graph.functions] == list(
        range(1, 8)
    )
    assert [
        (edge.caller.native_id, edge.callee.native_id, edge.indirect)
        for edge in reversed_graph.edges
    ] == [
        (edge.caller.native_id, edge.callee.native_id, edge.indirect)
        for edge in graph_value.edges
    ]
    assert [
        item.native_id for item in reversed_reachability.reachable
    ] == [item.native_id for item in reachable.reachable]


def test_indirect_calls_are_conservative_or_fail_closed():
    conservative = AIRPassManager().run(
        _GraphGlob(indirect=True),
        (AnalysisRequest(CallGraphAnalysis(True)),),
    )
    assert conservative.success
    graph = conservative.executions[0].analysis_result.value
    indirect_targets = {
        edge.callee.native_id for edge in graph.edges if edge.indirect
    }
    assert indirect_targets == set(range(1, 8))

    unsafe = AIRPassManager().run(
        _GraphGlob(indirect=True),
        (AnalysisRequest(CallGraphAnalysis(False)),),
    )
    assert not unsafe.success
    assert unsafe.diagnostics[0].code == "air.call-graph.unsafe-indirect-policy"


def test_framework_projection_is_stable_across_fixed_hash_seeds():
    test_file = os.path.abspath(__file__)
    script = f"""
import json
import runpy

namespace = runpy.run_path({test_file!r})
manager = namespace['AIRPassManager']()
dumps = []
instrumentation = namespace['PassInstrumentation'](
    on_ir_dump=lambda pass_id, ir: dumps.append([pass_id, ir])
)
glob = namespace['_GraphGlob']()
result = manager.run(
    glob,
    (
        namespace['AnalysisRequest'](namespace['CallGraphAnalysis']()),
        namespace['AnalysisRequest'](namespace['FunctionReachabilityAnalysis']()),
    ),
    {{'instrumentation': instrumentation}},
)
graph = result.executions[0].analysis_result.value
reachability = result.executions[1].analysis_result.value
print(json.dumps({{
    'passes': [record.pass_id for record in result.executions],
    'edges': [
        [edge.caller.native_id, edge.callee.native_id, edge.indirect]
        for edge in graph.edges
    ],
    'roots': [item.native_id for item in reachability.roots],
    'reachable': [item.native_id for item in reachability.reachable],
    'executions': [
        {{
            'type': type(record).__name__,
            'pass_id': record.pass_id,
            'unit': [
                record.unit.kind,
                record.unit.module_id,
                record.unit.native_id,
            ],
            'success': record.success,
            'changed': record.changed,
            'cached': record.cached,
            'before': record.before_revision,
            'after': record.after_revision,
            'metrics': dict(record.metrics),
            'diagnostics': [
                [diagnostic.code, diagnostic.message]
                for diagnostic in record.diagnostics
            ],
        }}
        for record in result.executions
    ],
    'diagnostics': [
        [diagnostic.code, diagnostic.message]
        for diagnostic in result.diagnostics
    ],
    'dumps': dumps,
    'normalized_air': glob.air_pass_snapshot(),
}}, sort_keys=True))
"""
    projections = []
    for seed in ("1", "77", "314159"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        projections.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=os.path.dirname(test_file),
                env=environment,
                text=True,
            ).strip()
        )
    assert len(set(projections)) == 1
    projection = json.loads(projections[0])
    assert projection["roots"] == [1, 2, 5, 3, 6]
    assert [item[1] for item in projection["dumps"]] == ["graph", "graph"]
    assert projection["normalized_air"]["revision"] == 0
