"""Shared hook, native barrier, and public runner integration contracts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import pytest

from ace_bindings import air_builder
from ace_edsl.edsl.compiler import (
    CompilerOptions,
    Target,
    ace_compile,
)
from ace_edsl.edsl.pipeline import (
    AcePipeline,
    Pipeline,
    PipelineTarget,
)
from ace_edsl.edsl.passes.framework import (
    ManagerSegmentExecution,
    AnalysisPass,
    AnalysisRequest,
    ModuleView,
    MutableModuleView,
    NativeBarrierOutcome,
    NativeBarrierStep,
    PassPipelineConfig,
    PassInstrumentation,
    PipelinePoint,
    TransformationPass,
)
from ace_edsl.edsl.passes.framework.api import (
    analysis_success,
    transform_failure,
    transform_success,
)
from ace_edsl.edsl.passes.framework.diagnostics import error
from ace_edsl.edsl.passes.framework.pipeline_hooks import run_pipeline_hook
from ace_edsl.edsl.passes.transition import TentativeNativeInlinerAdapter


_RUNS = {}


@dataclass(frozen=True)
class _Observe(TransformationPass):
    key: str
    fail: bool = False

    @property
    def pass_id(self):
        return f"test.observe.{self.key}"

    def run(self, unit, context):
        _RUNS[self.key] = _RUNS.get(self.key, 0) + 1
        if self.fail:
            return transform_failure(error("test.hook-failure", self.key))
        return transform_success(metrics={"runs": 1})


def _config(*steps):
    return PassPipelineConfig(
        {PipelinePoint.BEFORE_VECTOR2SIHE: tuple(steps)}
    )


def _native_module():
    glob = air_builder.GlobScope()
    vector = glob.new_array_type([4], "f32")
    function = glob.new_func_with_param_types("main", vector, [vector])
    value = function.new_param("input", vector)
    function.container().new_retv(value)
    assert glob.verify_ir()
    return glob


class _RunnerGlob:
    def __init__(self, glob=None):
        self.inner = glob or _native_module()
        self.native_phases = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def run_cpp_pass(self, name, skip_ops):
        self.native_phases.append(name)
        return True

    def run_poly2c(self, **kwargs):
        self.native_phases.append("poly2c")
        return True

    def get_c_code(self):
        return "generated"


def _hook_shape(result):
    return (
        result.success,
        result.changed,
        tuple(type(record).__name__ for record in result.records),
        tuple(
            (
                segment.success,
                segment.changed,
                tuple(
                    (
                        execution.pass_id,
                        execution.success,
                        execution.changed,
                        execution.before_revision,
                        execution.after_revision,
                        tuple(execution.metrics.items()),
                        tuple(
                            diagnostic.code
                            for diagnostic in execution.diagnostics
                        ),
                    )
                    for execution in segment.executions
                ),
                tuple(
                    execution.pass_id
                    for execution in segment.dependency_trace
                ),
                segment.unexecuted_step_ids,
            )
            for segment in result.manager_segments
        ),
        tuple(
            (
                record.step_id,
                record.success,
                record.changed,
                record.before_revision,
                record.after_revision,
                tuple(record.metrics.items()),
                tuple(diagnostic.code for diagnostic in record.diagnostics),
            )
            for record in result.barrier_records
        ),
        tuple(diagnostic.code for diagnostic in result.diagnostics),
    )


def test_empty_hook_is_behavior_preserving_and_configs_are_invocation_isolated():
    glob = _native_module()
    before = (glob.get_native_ptr(), glob.air_pass_revision, glob.air_pass_generation)
    empty = run_pipeline_hook(
        glob, PipelinePoint.BEFORE_VECTOR2SIHE, PassPipelineConfig()
    )
    assert empty.success and not empty.changed and empty.records == ()
    assert (glob.get_native_ptr(), glob.air_pass_revision, glob.air_pass_generation) == before

    _RUNS.clear()
    first = run_pipeline_hook(
        _native_module(), PipelinePoint.BEFORE_VECTOR2SIHE, _config(_Observe("a"))
    )
    second = run_pipeline_hook(
        _native_module(), PipelinePoint.BEFORE_VECTOR2SIHE, _config(_Observe("b"))
    )
    assert first.success and second.success
    assert _RUNS == {"a": 1, "b": 1}


def test_duplicate_ids_and_invalid_configs_are_rejected_identically():
    with pytest.raises(ValueError, match="duplicate pass-instance IDs"):
        _config(_Observe("same"), _Observe("same"))
    glob = _native_module()
    with pytest.raises(TypeError, match="PassPipelineConfig"):
        AcePipeline(glob, pass_pipeline_config=object())
    with pytest.raises(TypeError, match="PassPipelineConfig"):
        Pipeline(pass_pipeline_config=object())
    with pytest.raises(TypeError, match="PassPipelineConfig"):
        CompilerOptions(pass_pipeline_config=object())
    with pytest.raises(TypeError, match="PassInstrumentation"):
        PassPipelineConfig(instrumentation=object())
    with pytest.raises(TypeError, match="on_begin must be callable"):
        PassPipelineConfig(
            instrumentation=PassInstrumentation(on_begin="not callable")
        )

    @dataclass(frozen=True)
    class RetainsAIR(TransformationPass):
        retained: object

        def run(self, unit, context):
            return transform_success()

    live_view = ModuleView(glob, (glob.air_pass_module_id, 1))
    with pytest.raises(TypeError, match="live AIR"):
        _config(RetainsAIR(live_view))

    class MutableId(TransformationPass):
        identity = "stable"
        pass_id = "test.mutable-id.before"

        def run(self, unit, context):
            return transform_success()

    mutable = MutableId()
    config = _config(mutable)
    mutable.pass_id = "test.mutable-id.after"
    with pytest.raises(TypeError, match="instance ID mutated"):
        run_pipeline_hook(
            _native_module(), PipelinePoint.BEFORE_VECTOR2SIHE, config
        )


def _run_ace_pipeline(glob, config):
    native = []
    pipeline = AcePipeline(glob, pass_pipeline_config=config)
    pipeline.run_vector2sihe = lambda skip: native.append("vector2sihe") or True
    pipeline.run_sihe2ckks = lambda skip: True
    pipeline.run_ckks_driver = lambda: {"success": True}
    pipeline.run_poly_driver = lambda: {"success": True}
    pipeline.run_poly2c = lambda: "generated"
    result = pipeline.run(start_domain="nn::vector", verbose=False)
    return result, native


def _run_pipeline(glob, config):
    native = []
    pipeline = Pipeline(
        dump_ir=False, verbose=False, pass_pipeline_config=config
    ).set_glob(glob)
    pipeline._run_phase = lambda phase: native.append(phase) or True
    result = pipeline.run(phases=["vector2sihe"])
    return result, native


def _run_compile(glob, config):
    class Kernel:
        _ace_domain = "nn::vector"
        air_module = glob

    result = ace_compile(
        Kernel(),
        CompilerOptions(
            target=Target.SIHE,
            skip_cpp_for_registered_ops=False,
            pass_pipeline_config=config,
        ),
    )
    return result, glob.native_phases


def test_pass_runs_exactly_once_and_results_share_one_schema_in_all_three_runners():
    _RUNS.clear()
    config = _config(_Observe("shared"))
    ace_result, ace_native = _run_ace_pipeline(_RunnerGlob(), config)
    assert ace_result.success and ace_native == ["vector2sihe"]

    pipeline_result, pipeline_native = _run_pipeline(_RunnerGlob(), config)
    assert pipeline_result.success and pipeline_native == ["vector2sihe"]

    compile_result, compile_native = _run_compile(_RunnerGlob(), config)
    assert compile_native == ["vector2sihe"]
    assert _RUNS == {"shared": 3}

    point = PipelinePoint.BEFORE_VECTOR2SIHE
    hooks = (
        ace_result.pass_results[point],
        pipeline_result.pass_results[point],
        compile_result.pass_results[point],
    )
    assert all(len(hook.manager_segments) == 1 for hook in hooks)
    assert all(not hook.barrier_records for hook in hooks)
    assert len({_hook_shape(hook) for hook in hooks}) == 1


@pytest.mark.parametrize("runner", ["ace", "pipeline", "compile"])
def test_hook_failure_prevents_vector_to_sihe_in_every_runner(runner):
    _RUNS.clear()
    config = _config(_Observe(runner, fail=True))
    glob = _RunnerGlob()
    if runner == "ace":
        result, native = _run_ace_pipeline(glob, config)
        assert not result.success and native == []
        hook = result.pass_results[PipelinePoint.BEFORE_VECTOR2SIHE]
    elif runner == "pipeline":
        result, native = _run_pipeline(glob, config)
        assert not result.success and native == []
        hook = result.pass_results[PipelinePoint.BEFORE_VECTOR2SIHE]
    else:
        with pytest.raises(RuntimeError, match="before-vector2sihe"):
            _run_compile(glob, config)
        assert glob.native_phases == []
        return
    assert not hook.success
    assert hook.diagnostics[0].code == "test.hook-failure"


def test_tensor_to_vector_stop_skips_hook_while_vector_start_runs_it():
    _RUNS.clear()
    config = _config(_Observe("stop"))
    glob = _RunnerGlob()
    pipeline = Pipeline(
        dump_ir=False, verbose=False, pass_pipeline_config=config
    ).set_glob(glob)
    pipeline._run_phase = lambda phase: True
    stopped = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    assert stopped.success and not stopped.pass_results and _RUNS == {}

    started = pipeline.run(phases=["vector2sihe"])
    assert started.success and _RUNS == {"stop": 1}
    assert PipelinePoint.BEFORE_VECTOR2SIHE in started.pass_results

    class Kernel:
        _ace_domain = "nn::core"
        air_module = _RunnerGlob()

    compiled = ace_compile(
        Kernel(),
        CompilerOptions(
            target=Target.VECTOR,
            skip_cpp_for_registered_ops=False,
            pass_pipeline_config=config,
        ),
    )
    assert not compiled.pass_results and _RUNS == {"stop": 1}


class _AdapterGlob(_RunnerGlob):
    def __init__(self, outcome):
        super().__init__()
        self.outcome = outcome

    def inline_generated_vector_kernel_helpers(self):
        if self.outcome == "changed":
            transaction = self.inner.begin_air_pass_transaction()
            candidate = MutableModuleView(
                self.inner,
                (self.inner.air_pass_module_id, 1),
                transaction=transaction,
            )
            candidate.functions[0].create_local("barrier", candidate.types[0])
            assert transaction.commit()
            return {
                "success": True,
                "changed": True,
                "calls_inlined": 1,
                "helpers_removed": 1,
                "diagnostic": "",
            }
        if self.outcome == "failure":
            return {
                "success": False,
                "changed": False,
                "calls_inlined": 0,
                "helpers_removed": 0,
                "diagnostic": "expected adapter failure",
            }
        return {
            "success": True,
            "changed": False,
            "calls_inlined": 0,
            "helpers_removed": 0,
            "diagnostic": "",
        }


@pytest.mark.parametrize(
    ("outcome", "success", "changed", "revision"),
    [
        ("changed", True, True, 1),
        ("noop", True, False, 0),
        ("failure", False, False, 0),
    ],
)
def test_tentative_adapter_barrier_closes_manager_and_invalidates_views(
    outcome, success, changed, revision
):
    glob = _AdapterGlob(outcome)
    old_view = ModuleView(
        glob, (glob.air_pass_module_id, 99)
    ).functions[0]
    before_pointer = glob.get_native_ptr()
    result = run_pipeline_hook(
        glob,
        PipelinePoint.BEFORE_VECTOR2SIHE,
        PassPipelineConfig(),
        native_barriers=(TentativeNativeInlinerAdapter(),),
    )
    assert result.success is success and result.changed is changed
    assert len(result.records) == 2
    assert isinstance(result.records[0], ManagerSegmentExecution)
    assert result.records[0].result.executions == ()
    assert len(result.barrier_records) == 1
    assert result.barrier_records[0].step_id == (
        "air.transition.tentative-native-inliner"
    )
    assert glob.air_pass_revision == revision
    assert glob.air_pass_generation == 1
    assert (glob.get_native_ptr() != before_pointer) is changed
    with pytest.raises(RuntimeError, match="stale AIR view"):
        _ = old_view.name
    assert all(
        execution.pass_id != "air.transition.tentative-native-inliner"
        for segment in result.manager_segments
        for execution in segment.executions
    )


@dataclass(frozen=True)
class _RaisingBarrier(NativeBarrierStep):
    pass_id = "test.raising-barrier"

    def run(self, glob_scope):
        raise RuntimeError("native boom")


def test_unexpected_native_barrier_exception_is_structured_and_generation_closed():
    glob = _RunnerGlob()
    result = run_pipeline_hook(
        glob,
        PipelinePoint.BEFORE_VECTOR2SIHE,
        PassPipelineConfig(),
        native_barriers=(_RaisingBarrier(),),
    )
    assert not result.success and not result.changed
    assert result.diagnostics[0].code == "air.native-barrier.exception"
    assert glob.air_pass_revision == 0 and glob.air_pass_generation == 1


@dataclass(frozen=True)
class _MalformedChangedBarrier(NativeBarrierStep):
    pass_id = "test.malformed-changed-barrier"

    def run(self, glob_scope):
        return NativeBarrierOutcome(True, True)


@dataclass(frozen=True)
class _SilentFailureBarrier(NativeBarrierStep):
    pass_id = "test.silent-failure-barrier"

    def run(self, glob_scope):
        return NativeBarrierOutcome(False, False)


@dataclass(frozen=True)
class _ClosedNoOpBarrier(NativeBarrierStep):
    pass_id = "test.closed-noop-barrier"

    def run(self, glob_scope):
        glob_scope.invalidate_air_pass_views(False)
        return NativeBarrierOutcome(True, False)


@pytest.mark.parametrize(
    ("barrier", "diagnostic"),
    [
        (_MalformedChangedBarrier(), "air.native-barrier.invalid-change"),
        (_SilentFailureBarrier(), "air.native-barrier.failed"),
    ],
)
def test_malformed_or_silent_barrier_failure_is_diagnosed_and_closes_once(
    barrier, diagnostic
):
    glob = _RunnerGlob()
    old_view = ModuleView(glob, (glob.air_pass_module_id, 1))
    result = run_pipeline_hook(
        glob,
        PipelinePoint.BEFORE_VECTOR2SIHE,
        PassPipelineConfig(),
        native_barriers=(barrier,),
    )
    assert not result.success and not result.changed
    assert result.barrier_records[0].changed is False
    assert diagnostic in {
        item.code for item in result.barrier_records[0].diagnostics
    }
    assert glob.air_pass_revision == 0 and glob.air_pass_generation == 1
    with pytest.raises(RuntimeError, match="stale AIR view"):
        _ = old_view.id


def test_barrier_that_already_closed_generation_is_not_closed_twice():
    glob = _RunnerGlob()
    result = run_pipeline_hook(
        glob,
        PipelinePoint.BEFORE_VECTOR2SIHE,
        PassPipelineConfig(),
        native_barriers=(_ClosedNoOpBarrier(),),
    )
    assert result.success and not result.changed
    assert glob.air_pass_revision == 0 and glob.air_pass_generation == 1


def test_native_barrier_ends_analysis_cache_lifetime_and_record_order():
    counts = {}

    class CountingAnalysis(AnalysisPass):
        pass_id = "analysis.pipeline-cache"

        def run(self, unit, context):
            counts["runs"] = counts.get("runs", 0) + 1
            return analysis_success(counts["runs"])

    analysis = CountingAnalysis()
    config = _config(AnalysisRequest(analysis))
    glob = _AdapterGlob("noop")
    first = run_pipeline_hook(
        glob,
        PipelinePoint.BEFORE_VECTOR2SIHE,
        config,
        native_barriers=(TentativeNativeInlinerAdapter(),),
    )
    second = run_pipeline_hook(
        glob, PipelinePoint.BEFORE_VECTOR2SIHE, config
    )
    assert first.success and second.success and counts == {"runs": 2}
    assert [type(record).__name__ for record in first.records] == [
        "ManagerSegmentExecution",
        "NativeBarrierExecutionResult",
    ]
    assert second.manager_segments[0].executions[0].analysis_result.value == 2


def test_removed_inliner_facade_and_domain_neutral_framework_surface():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("ace_edsl.edsl.passes.function_inliner")
    passes = importlib.import_module("ace_edsl.edsl.passes")
    edsl = importlib.import_module("ace_edsl.edsl")
    assert not hasattr(passes, "FunctionInlinerPass")
    assert not hasattr(edsl, "FunctionInlinerPass")

    passes_root = Path(__file__).parents[1] / "edsl" / "passes"
    sources = (
        *sorted((passes_root / "framework").glob("*.py")),
        *sorted((passes_root / "analyses").glob("*.py")),
    )
    forbidden = ("gemm", "conv", "vector_kernel", "generated_helper", "inliner")
    for source in sources:
        text = source.read_text().lower()
        assert not any(term in text for term in forbidden), source
