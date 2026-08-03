"""Python vector-kernel provider integration and ownership regressions."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
from functools import lru_cache
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from ace_bindings import air_builder
from ace_edsl.edsl import AceEDSL, Tensor, nn_kernel, tensor_ops
from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget
from ace_edsl.edsl.vector.planning import (
    TypedPayload,
    canonical_payload_hash,
    plan_vector_kernel,
)


_TEST_DIRECTORY = str(Path(__file__).resolve().parent)
if _TEST_DIRECTORY not in sys.path:
    sys.path.insert(0, _TEST_DIRECTORY)

from test_nn_conv_gemm_authoring import _CASES, _CASE_SETTINGS  # noqa: E402


_WORKER_FLAG = "--vector-plan-provider-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@nn_kernel
def _direct_non_power_baseline_gemm(
    value: Tensor[float, 1, 6],
) -> Tensor[float, 1, 2]:
    weight = tensor_ops.constant(
        [float(index + 1) / 16.0 for index in range(12)],
        shape=(2, 6),
    )
    bias = tensor_ops.constant([0.25, -0.5])
    return tensor_ops.gemm(value, weight, bias, 1.0, 1.0, 0, 1)


_PROVIDER_CASES = dict(_CASES)
_PROVIDER_CASES["baseline-gemm-non-power"] = {
    "kernel": _direct_non_power_baseline_gemm,
    "input": Tensor[float, 1, 6],
    "result_shape": [1, 2],
    "operation": "gemm",
    "attrs": "ATTR[transB=1,transA=0,beta=1,alpha=1]",
    "inspector": "_inspect_baseline_gemm_air_for_testing",
    "plan_kind": "baseline-gemm",
}
_PROVIDER_CASE_SETTINGS = dict(_CASE_SETTINGS)
_PROVIDER_CASE_SETTINGS["baseline-gemm-non-power"] = {
    "mask_fuse": False,
    "max_slots": 128,
}


def _recipes():
    from ace_edsl.edsl.vector.kernels.baseline_conv import baseline_conv_recipe
    from ace_edsl.edsl.vector.kernels.baseline_gemm import baseline_gemm_recipe
    from ace_edsl.edsl.vector.kernels.fast_conv import fast_conv_recipe
    from ace_edsl.edsl.vector.kernels.fast_gemm import fast_gemm_recipe

    return {
        "baseline-gemm": baseline_gemm_recipe,
        "baseline-conv": baseline_conv_recipe,
        "fast-gemm": fast_gemm_recipe,
        "fast-conv": fast_conv_recipe,
    }


def _plain(value):
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": value.tobytes(order="C").hex(),
        }
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _without_provenance(prepared):
    semantic = dict(prepared)
    semantic.pop("provenance")
    return semantic


def _trace_case(case_name):
    case = _PROVIDER_CASES[case_name]
    AceEDSL._get_dsl.cache_clear()
    case["kernel"](case["input"])
    glob = AceEDSL._get_dsl().current_air_module
    return glob, glob.dump()


def _execute(
    case_name,
    plan_provider="cpp",
    kernel_impl="native",
    plan_kind=None,
    provider_callback=None,
    configure=True,
    automatic_settings=False,
    settings_override=None,
    fallback="error",
):
    case = _PROVIDER_CASES[case_name]
    glob, initial_dump = _trace_case(case_name)
    initial_verify = glob.verify_ir()
    captured = []
    provider_calls = []
    recipes = _recipes()

    def recipe(trace, prepared):
        captured.append(_plain(prepared))
        return recipes[prepared.kind](trace, prepared)

    def observed_provider(request):
        provider_calls.append(str(request.requested_plan_kind))
        if provider_callback is not None:
            return provider_callback(request)
        return plan_vector_kernel(request)

    pipeline = Pipeline(
        "python-vector-plan-provider",
        output_dir="/tmp/ace-python-vector-plan-provider",
        dump_ir=False,
        verbose=False,
    ).set_glob(glob)
    if configure:
        settings = settings_override
        if settings is None:
            settings = (
                {"mask_fuse": False, "max_slots": 128}
                if automatic_settings
                else _PROVIDER_CASE_SETTINGS[case_name]
            )
        pipeline.configure_vector_kernel_lowering(
            plan_provider=plan_provider,
            kernel_impl=kernel_impl,
            plan_kind=(
                case.get("plan_kind", case_name) if plan_kind is None else plan_kind
            ),
            fallback=fallback,
            mask_fuse=settings["mask_fuse"],
            max_slots=settings["max_slots"],
            conv_parallel=False,
            sharding=False,
        )
        if provider_callback is not None:
            pipeline.register_vector_kernel_plan_provider(observed_provider)
        if kernel_impl == "dsl":
            for kind in recipes:
                pipeline.register_vector_kernel_recipe(kind, recipe)

    run = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    dump = pipeline.glob.dump()
    helper_names = sorted(set(re.findall(r'"(__ace_vkernel_[^"]+)"', dump)))
    helper_calls = len(re.findall(r'^\s+call "__ace_vkernel_', dump, re.MULTILINE))
    normalized = None
    if run.success and kernel_impl == "native":
        normalized = getattr(pipeline.glob, case["inspector"])()
    return {
        "success": run.success,
        "error": run.error,
        "initial_verify": initial_verify,
        "verify": pipeline.glob.verify_ir(),
        "stages_completed": run.stages_completed,
        "dump_unchanged": dump == initial_dump,
        "has_source_nn": bool(
            re.search(rf"\bNN\.{case['operation']}\b", dump, re.IGNORECASE)
        ),
        "helper_names": helper_names,
        "helper_count": len(helper_names),
        "helper_calls": helper_calls,
        "prepared": captured,
        "provider_calls": provider_calls,
        "normalized": normalized,
    }


def _replace_constant(result, index, payload):
    constants = list(result.constants)
    constants[index] = payload
    return replace(result, constants=tuple(constants))


def _invalid_provider(mutation):
    def provider(request):
        if mutation == "callback-exception":
            raise RuntimeError("intentional provider boom")
        if mutation == "non-result":
            return {"plan": "not a ProviderResult"}

        result = plan_vector_kernel(request)
        first = result.constants[0]
        if mutation == "missing-role":
            return replace(result, constants=result.constants[:-1])
        if mutation == "extra-role":
            extra = replace(first, role="unexpected")
            return replace(result, constants=result.constants + (extra,))
        if mutation == "duplicate-role":
            return _replace_constant(result, len(result.constants) - 1, first)
        if mutation == "bad-hash":
            return _replace_constant(
                result, 0, replace(first, content_hash="sha256:" + "0" * 64)
            )
        if mutation == "wrong-shape":
            values = np.array(first.values, copy=True).reshape((-1,))
            return _replace_constant(result, 0, replace(first, values=values))
        if mutation == "unsupported-dtype":
            values = np.array(first.values, dtype="<f2", order="C")
            return _replace_constant(result, 0, replace(first, values=values))
        if mutation == "non-contiguous":
            values = np.array(first.values, copy=True)[:, ::-1]
            return _replace_constant(result, 0, replace(first, values=values))
        if mutation == "big-endian":
            values = np.array(first.values, dtype=">f4", order="C")
            return _replace_constant(result, 0, replace(first, values=values))
        if mutation == "invalid-preparation":
            preparation = result.runtime_preparations[0]
            invalid = replace(preparation, replications=preparation.replications + 1)
            return replace(result, runtime_preparations=(invalid,))
        if mutation == "invalid-preparation-kind":
            preparation = result.runtime_preparations[0]
            invalid = replace(preparation, kind="unknown-preparation")
            return replace(result, runtime_preparations=(invalid,))
        if mutation == "invalid-plan-field":
            return replace(
                result, plan=replace(result.plan, height=result.plan.height + 1)
            )
        raise AssertionError(f"unknown provider mutation: {mutation}")

    return provider


def _matrix_worker(case_name):
    runs = {}
    for provider in ("cpp", "python"):
        for implementation in ("native", "dsl"):
            key = f"{provider}-{implementation}"
            runs[key] = _execute(case_name, provider, implementation)
    runs["cpp-auto-dsl"] = _execute(
        case_name,
        "cpp",
        "dsl",
        plan_kind="auto",
        automatic_settings=True,
    )
    runs["python-auto-dsl"] = _execute(
        case_name,
        "python",
        "dsl",
        plan_kind="auto",
        automatic_settings=True,
    )
    return runs


def _negative_worker(mutation, fallback):
    return _execute(
        "baseline-gemm",
        "python",
        "native",
        provider_callback=_invalid_provider(mutation),
        fallback=fallback,
    )


def _lifetime_worker():
    retained = {}

    def provider(request):
        retained["view"] = request
        retained["source_constant_method"] = request.source_constant
        assert retained["source_constant_method"]("weight") is not None
        arrays = tuple(
            item.values for item in (*request.attributes, *request.source_constants)
        )
        retained["arrays"] = arrays
        retained["before"] = [array.tobytes(order="C").hex() for array in arrays]
        retained["writeable"] = [bool(array.flags.writeable) for array in arrays]
        mutation_errors = []
        for array in arrays:
            try:
                array.flat[0] = array.flat[0]
            except (TypeError, ValueError) as error:
                mutation_errors.append(str(error))
        retained["mutation_errors"] = mutation_errors
        return plan_vector_kernel(request)

    run = _execute("fast-gemm", "python", "dsl", provider_callback=provider)
    try:
        retained["view"].operation
    except RuntimeError as error:
        expired = str(error)
    else:
        expired = ""
    try:
        retained["source_constant_method"]("weight")
    except RuntimeError as error:
        expired_method = str(error)
    else:
        expired_method = ""

    after = [array.tobytes(order="C").hex() for array in retained["arrays"]]
    return {
        "run": run,
        "expired": expired,
        "expired_method": expired_method,
        "writeable": retained["writeable"],
        "mutation_errors": retained["mutation_errors"],
        "retained_bytes_equal": retained["before"] == after,
        "array_count": len(after),
    }


def _returned_copy_worker():
    returned_arrays = []
    original_bytes = {}

    def provider(request):
        result = plan_vector_kernel(request)
        copied = []
        for payload in result.constants:
            values = np.array(payload.values, copy=True, order="C")
            copied.append(
                TypedPayload(payload.role, values, canonical_payload_hash(values))
            )
            returned_arrays.append(values)
            original_bytes[payload.role] = values.tobytes(order="C").hex()
        return replace(result, constants=tuple(copied))

    run = _execute("fast-gemm", "python", "dsl", provider_callback=provider)
    for array in returned_arrays:
        array.flat[0] = array.flat[0] + 1
    mutated = any(
        array.tobytes(order="C").hex() != original
        for array, original in zip(returned_arrays, original_bytes.values())
    )
    returned_arrays.clear()
    gc.collect()
    prepared_bytes = {
        item["role"]: item["bytes"] for item in run["prepared"][0]["constants"]
    }
    return {
        "run": run,
        "mutated": mutated,
        "prepared_bytes": prepared_bytes,
        "original_bytes": original_bytes,
    }


def _sequence_worker(case_names):
    calls = []
    results = {}
    for case_name in case_names:

        def provider(request, label=case_name):
            calls.append(label)
            return plan_vector_kernel(request)

        results[case_name] = _execute(
            case_name, "python", "dsl", provider_callback=provider
        )
    return {"calls": calls, "results": results}


def _defaults_worker():
    poisoned_calls = []

    def poisoned(_request):
        poisoned_calls.append("called")
        raise RuntimeError("cpp must not invoke a registered Python provider")

    default = _execute("fast-gemm", configure=False)
    explicit = _execute(
        "fast-gemm",
        "cpp",
        "native",
        plan_kind="auto",
        provider_callback=poisoned,
        settings_override={"mask_fuse": False, "max_slots": 0},
    )
    return {
        "default": default,
        "explicit": explicit,
        "poisoned_calls": poisoned_calls,
    }


def _provenance_worker():
    runs = {}
    for provenance in ("first-origin", "second-origin"):

        def provider(request, label=provenance):
            return replace(plan_vector_kernel(request), provenance=label)

        runs[provenance] = _execute(
            "fast-gemm", "python", "dsl", provider_callback=provider
        )

    glob = air_builder.GlobScope()
    input_type = glob.new_array_type([1, 4], "f32")
    result_type = glob.new_array_type([1, 4], "f32")
    weight_values = [float(index + 1) / 16.0 for index in range(16)]
    bias_values = [0.0, 0.25, 0.5, 0.75]
    for name in ("first_fast_gemm", "second_fast_gemm"):
        scope = glob.new_func_with_param_types(name, result_type, [input_type])
        container = scope.container()
        value = scope.new_param("input", input_type)
        weight = container.new_ranked_f32_const(weight_values, [4, 4])
        bias = container.new_ranked_f32_const(bias_values, [4])
        result = container.new_nn_gemm(value, weight, bias, 1.0, 1.0, 0, 1)
        container.new_retv(result)
    assert glob.verify_ir()
    provenances = iter(("left-origin", "right-origin"))
    provider_calls = []

    def alternating_provider(request):
        provenance = next(provenances)
        provider_calls.append(provenance)
        return replace(plan_vector_kernel(request), provenance=provenance)

    pipeline = Pipeline(
        "python-provider-provenance-dedup",
        output_dir="/tmp/ace-python-provider-provenance-dedup",
        dump_ir=False,
        verbose=False,
    ).set_glob(glob)
    pipeline.configure_vector_kernel_lowering(
        plan_provider="python",
        kernel_impl="dsl",
        plan_kind="fast-gemm",
        fallback="error",
        mask_fuse=False,
        max_slots=128,
    )
    pipeline.register_vector_kernel_plan_provider(alternating_provider)
    pipeline.register_vector_kernel_recipe("fast-gemm", _recipes()["fast-gemm"])
    result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    dump = pipeline.glob.dump()
    helper_names = sorted(set(re.findall(r'"(__ace_vkernel_fast_gemm_[^"]+)"', dump)))
    return {
        "runs": runs,
        "dedup": {
            "success": result.success,
            "verify": pipeline.glob.verify_ir(),
            "provider_calls": provider_calls,
            "helper_names": helper_names,
            "helper_calls": len(
                re.findall(r'^\s+call "__ace_vkernel_fast_gemm_', dump, re.MULTILINE)
            ),
            "has_source_nn": "NN.gemm" in dump,
        },
    }


def _worker_main():
    mode = sys.argv[2]
    if mode == "matrix":
        payload = _matrix_worker(sys.argv[3])
    elif mode == "negative":
        payload = _negative_worker(sys.argv[3], sys.argv[4])
    elif mode == "lifetime":
        payload = _lifetime_worker()
    elif mode == "returned-copy":
        payload = _returned_copy_worker()
    elif mode == "sequence":
        payload = _sequence_worker(tuple(sys.argv[3].split(",")))
    elif mode == "defaults":
        payload = _defaults_worker()
    elif mode == "provenance":
        payload = _provenance_worker()
    elif mode == "seed":
        payload = _execute("fast-conv", "python", "dsl")
    else:
        raise AssertionError(f"unknown worker mode: {mode}")
    print(json.dumps(payload, sort_keys=True))


@lru_cache(maxsize=None)
def _cached_worker(arguments, hash_seed):
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(_REPO_ROOT), old_pythonpath))
    )
    if hash_seed is not None:
        environment["PYTHONHASHSEED"] = str(hash_seed)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
        *arguments,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    diagnostics = "\n".join((completed.stdout, completed.stderr))
    if completed.returncode != 0:
        return {
            "fatal": True,
            "returncode": completed.returncode,
            "diagnostics": diagnostics,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"provider worker produced no JSON:\n{diagnostics}"
    payload = json.loads(lines[-1])
    payload["fatal"] = False
    payload["diagnostics"] = "\n".join((*lines[:-1], completed.stderr))
    return payload


def _run_worker(*arguments, hash_seed=None):
    return _cached_worker(tuple(arguments), hash_seed)


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_cpp_and_python_providers_cross_both_emitters_with_exact_plans(case_name):
    matrix = _run_worker("matrix", case_name)
    assert not matrix["fatal"], matrix["diagnostics"]
    assert "vector-kernel fallback:" not in matrix["diagnostics"]

    for provider in ("cpp", "python"):
        for implementation in ("native", "dsl"):
            run = matrix[f"{provider}-{implementation}"]
            assert run["success"] and run["initial_verify"] and run["verify"]
            assert run["stages_completed"] == ["tensor2vector"]
            assert not run["has_source_nn"]
            assert run["helper_count"] == (1 if implementation == "dsl" else 0)
            assert run["helper_calls"] == (1 if implementation == "dsl" else 0)
            assert run["provider_calls"] == []

    cpp_prepared = matrix["cpp-dsl"]["prepared"]
    python_prepared = matrix["python-dsl"]["prepared"]
    assert len(cpp_prepared) == len(python_prepared) == 1
    assert cpp_prepared[0]["provenance"] == "cpp"
    assert python_prepared[0]["provenance"] == "python"
    assert _without_provenance(cpp_prepared[0]) == _without_provenance(
        python_prepared[0]
    )
    assert cpp_prepared[0]["kind"] == case_name
    assert cpp_prepared[0]["specialization_key"]
    assert cpp_prepared[0]["helper_name"]
    assert all(item["bytes"] for item in cpp_prepared[0]["constants"])
    assert all(
        item["content_hash"].startswith("sha256:")
        for item in cpp_prepared[0]["constants"]
    )

    assert (
        matrix["cpp-native"]["normalized"]["normalized"]
        == (matrix["python-native"]["normalized"]["normalized"])
    )


def test_non_power_baseline_gemm_matches_across_providers_and_emitters():
    matrix = _run_worker("matrix", "baseline-gemm-non-power")
    assert not matrix["fatal"], matrix["diagnostics"]
    assert "vector-kernel fallback:" not in matrix["diagnostics"]

    for provider in ("cpp", "python"):
        for implementation in ("native", "dsl"):
            run = matrix[f"{provider}-{implementation}"]
            assert run["success"] and run["initial_verify"] and run["verify"]
            assert run["stages_completed"] == ["tensor2vector"]
            assert not run["has_source_nn"]
            assert run["helper_count"] == (1 if implementation == "dsl" else 0)
            assert run["helper_calls"] == (1 if implementation == "dsl" else 0)

    cpp = matrix["cpp-dsl"]["prepared"][0]
    python = matrix["python-dsl"]["prepared"][0]
    assert cpp["kind"] == python["kind"] == "baseline-gemm"
    assert _without_provenance(cpp) == _without_provenance(python)

    common = cpp
    assert common["loops"][-1] == {
        "role": "block-reduction",
        "lower": 0,
        "upper": 2,
        "step": 1,
        "nesting_depth": 0,
    }
    assert common["rotations"][-1] == {
        "role": "block-reduction",
        "candidates": [2, 4],
    }
    assert common["reductions"] == [
        {
            "role": "block-reduction",
            "kind": "power-of-two",
            "factor": 3,
            "block_width": 2,
            "padding": 0,
        }
    ]
    assert (
        matrix["cpp-native"]["normalized"]["normalized"]
        == (matrix["python-native"]["normalized"]["normalized"])
    )


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_auto_python_and_cpp_planning_have_exact_prepared_semantics(case_name):
    matrix = _run_worker("matrix", case_name)
    assert not matrix["fatal"], matrix["diagnostics"]
    cpp = matrix["cpp-auto-dsl"]
    python = matrix["python-auto-dsl"]
    for run in (cpp, python):
        assert run["success"] and run["verify"]
        assert not run["has_source_nn"]
        assert run["helper_count"] == run["helper_calls"] == 1
        assert len(run["prepared"]) == 1
    assert cpp["prepared"][0]["kind"] == python["prepared"][0]["kind"]
    assert _without_provenance(cpp["prepared"][0]) == _without_provenance(
        python["prepared"][0]
    )


_INVALID_RESULTS = {
    "callback-exception": r"intentional provider boom",
    "non-result": r"must return ProviderResult",
    "missing-role": r"constant role count does not match plan",
    "extra-role": r"constant role count does not match plan",
    "duplicate-role": r"duplicate constant role",
    "bad-hash": r"constant hash mismatch",
    "wrong-shape": r"dtype or shape does not match its plan descriptor",
    "unsupported-dtype": r"not an exact supported vector-kernel dtype",
    "non-contiguous": r"must be C-contiguous",
    "big-endian": r"canonical little-endian byte order",
    "invalid-preparation": r"runtime preparation mismatch",
    "invalid-preparation-kind": r"unknown-preparation|has no attribute 'value'",
    "invalid-plan-field": r"invalid baseline Gemm plan invariants",
}


@pytest.mark.parametrize("fallback", ("error", "cpp-native"))
@pytest.mark.parametrize("mutation,diagnostic", tuple(_INVALID_RESULTS.items()))
def test_malformed_provider_results_fail_in_isolated_process(
    mutation, diagnostic, fallback
):
    result = _run_worker("negative", mutation, fallback)
    assert re.search(diagnostic, result["diagnostics"]), result["diagnostics"]
    assert "vector-kernel fallback:" not in result["diagnostics"]
    if not result["fatal"]:
        assert not result["success"]
        assert result["dump_unchanged"]


def test_request_view_expires_while_owned_request_arrays_remain_read_only():
    result = _run_worker("lifetime")
    assert not result["fatal"], result["diagnostics"]
    run = result["run"]
    assert run["success"] and run["verify"]
    assert result["array_count"] > 2
    assert result["writeable"] == [False] * result["array_count"]
    assert len(result["mutation_errors"]) == result["array_count"]
    assert result["retained_bytes_equal"]
    assert "request view has expired" in result["expired"]
    assert "request view has expired" in result["expired_method"]


def test_returned_arrays_are_copied_before_mutation_and_release():
    result = _run_worker("returned-copy")
    assert not result["fatal"], result["diagnostics"]
    assert result["run"]["success"] and result["run"]["verify"]
    assert result["mutated"]
    assert result["prepared_bytes"] == result["original_bytes"]


def test_plan_provider_registration_is_per_driver_and_order_independent():
    forward = _run_worker("sequence", "baseline-gemm,fast-conv")
    reverse = _run_worker("sequence", "fast-conv,baseline-gemm")
    for result in (forward, reverse):
        assert not result["fatal"], result["diagnostics"]
        assert all(
            run["success"] and run["verify"] for run in result["results"].values()
        )
    assert forward["calls"] == ["baseline-gemm", "fast-conv"]
    assert reverse["calls"] == ["fast-conv", "baseline-gemm"]
    for case_name in ("baseline-gemm", "fast-conv"):
        left = forward["results"][case_name]["prepared"][0]
        right = reverse["results"][case_name]["prepared"][0]
        assert left == right


def test_cpp_native_remains_default_and_ignores_registered_python_callback():
    result = _run_worker("defaults")
    assert not result["fatal"], result["diagnostics"]
    assert result["poisoned_calls"] == []
    for run in (result["default"], result["explicit"]):
        assert run["success"] and run["verify"]
        assert not run["has_source_nn"]
        assert run["helper_count"] == run["helper_calls"] == 0
    assert (
        result["default"]["normalized"]["normalized"]
        == (result["explicit"]["normalized"]["normalized"])
    )


def test_provenance_does_not_change_identity_or_helper_deduplication():
    result = _run_worker("provenance")
    assert not result["fatal"], result["diagnostics"]
    first = result["runs"]["first-origin"]["prepared"][0]
    second = result["runs"]["second-origin"]["prepared"][0]
    assert first["provenance"] != second["provenance"]
    assert first["specialization_key"] == second["specialization_key"]
    assert first["helper_name"] == second["helper_name"]
    assert _without_provenance(first) == _without_provenance(second)

    dedup = result["dedup"]
    assert dedup["success"] and dedup["verify"]
    assert dedup["provider_calls"] == ["left-origin", "right-origin"]
    assert len(dedup["helper_names"]) == 1
    assert dedup["helper_calls"] == 2
    assert not dedup["has_source_nn"]


def test_python_provider_is_deterministic_across_hash_seeds():
    runs = [_run_worker("seed", hash_seed=seed) for seed in (1, 77, 314159)]
    for run in runs:
        assert not run["fatal"], run["diagnostics"]
        assert run["success"] and run["verify"]
    prepared = [run["prepared"][0] for run in runs]
    assert prepared[1:] == prepared[:-1]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
