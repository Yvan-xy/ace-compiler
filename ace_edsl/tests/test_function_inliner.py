"""Tests for the temporary binding-side generated-helper inliner."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


_WORKER_FLAG = "--function-inliner-worker"
_WORKER_TIMEOUT_SECONDS = 600
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER_PREFIX = "__ace_vkernel_baseline_gemm_"


def _write_gemm_model(path: Path, copies: int = 1):
    height = 2
    width = 8
    values = (np.arange(height * width, dtype=np.float32) + 1.0) / 8.0
    weight = numpy_helper.from_array(values.reshape(height, width), "weight")
    bias = numpy_helper.from_array(np.arange(height, dtype=np.float32) / 4.0, "bias")
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, width])
    gemm_outputs = []
    nodes = []
    for index in range(copies):
        output_name = f"output_{index}"
        gemm_outputs.append(output_name)
        nodes.append(
            helper.make_node(
                "Gemm",
                ["input", "weight", "bias"],
                [output_name],
                name=f"gemm_{index}",
                transB=1,
            )
        )
    output_name = gemm_outputs[0]
    if copies == 2:
        output_name = "combined_output"
        nodes.append(
            helper.make_node("Add", gemm_outputs, [output_name], name="combine_gemms")
        )
    output_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, height]
    )
    graph = helper.make_graph(
        nodes, "function_inliner", [input_info], [output_info], [weight, bias]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run_worker(model: Path, action: str):
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(_REPO_ROOT), old_pythonpath))
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
        str(model),
        action,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=_WORKER_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"worker failed ({' '.join(command)}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    return json.loads(lines[-1])


def test_always_inlines_nested_helper_and_is_idempotent(tmp_path):
    model = tmp_path / "nested_gemm.onnx"
    _write_gemm_model(model)

    result = _run_worker(model, "always")

    assert result["pipeline_success"]
    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 1
    assert result["before"]["helper_scopes"] == 1
    assert result["before"]["helper_symbols"] == 1
    assert result["selection"] == [
        {
            "attribute": "ace.vector_kernel.generated_call",
            "kind": "same-module-generated-leaf",
        }
    ]
    assert result["pass"] == {
        "success": True,
        "changed": True,
        "calls_inlined": 1,
        "helpers_removed": 1,
        "diagnostics": [],
    }
    assert result["after"]["verify"]
    assert result["after"]["helper_calls"] == 0
    assert result["after"]["helper_scopes"] == 0
    assert result["after"]["helper_symbols"] == 0
    assert result["after"]["arg_stores"] == 1
    assert result["after"]["vector_ops"] == result["before"]["vector_ops"]
    assert result["after"]["rotations"] == result["before"]["rotations"]
    assert not result["native_ptr_stable"]
    assert result["stale_type_readable"]
    assert result["stale_type_is_foreign"]
    assert result["cached_type_usable"]
    assert result["second"] == {
        "success": True,
        "changed": False,
        "calls_inlined": 0,
        "helpers_removed": 0,
        "diagnostics": [],
    }
    assert result["second_dump_unchanged"]


def test_shared_helper_all_calls_inline_before_single_cleanup(tmp_path):
    model = tmp_path / "shared_helper.onnx"
    _write_gemm_model(model, copies=2)

    result = _run_worker(model, "always")

    assert result["before"]["helper_calls"] == 2
    assert result["before"]["helper_scopes"] == 1
    assert result["pass"]["calls_inlined"] == 2
    assert result["pass"]["helpers_removed"] == 1
    assert result["after"]["verify"]
    assert result["after"]["helper_calls"] == 0
    assert result["after"]["helper_scopes"] == 0
    assert result["after"]["helper_symbols"] == 0
    assert result["after"]["arg_stores"] == 2


@pytest.mark.parametrize(
    "action,success,diagnostic",
    [
        ("never", True, ""),
        ("predicate-false", True, ""),
        ("predicate-raise", False, "predicate failed: selection failure"),
        ("invalid-policy", False, "unsupported function-inliner policy"),
        ("extra-entry", False, "additional entry point"),
        ("program-entry", False, "one non-program entry point"),
    ],
)
def test_policy_and_predicate_failures_do_not_mutate(
    tmp_path, action, success, diagnostic
):
    model = tmp_path / f"{action}.onnx"
    _write_gemm_model(model)

    result = _run_worker(model, action)

    assert result["pass"]["success"] is success
    assert not result["pass"]["changed"]
    assert result["pass"]["calls_inlined"] == 0
    assert result["pass"]["helpers_removed"] == 0
    assert result["dump_unchanged"]
    assert result["native_ptr_stable"]
    assert result["after"]["verify"]
    if diagnostic:
        assert diagnostic in "; ".join(result["pass"]["diagnostics"])
    else:
        assert result["pass"]["diagnostics"] == []
    if action == "predicate-false":
        assert result["selection"][0]["kind"] == "same-module-generated-leaf"


def test_python_pass_rejects_missing_scope_or_binding():
    from ace_edsl.edsl.passes.function_inliner import FunctionInlinerPass

    missing_scope = FunctionInlinerPass.run(None)
    assert not missing_scope.success
    assert missing_scope.diagnostics == ("glob_scope is None",)

    missing_binding = FunctionInlinerPass.run(object())
    assert not missing_binding.success
    assert "has no generated-helper inlining binding" in missing_binding.diagnostics[0]


def test_pipeline_boundary_and_native_parity(tmp_path):
    model = tmp_path / "pipeline_gemm.onnx"
    _write_gemm_model(model)

    pre_inline = _run_worker(model, "dsl-t2v")
    dsl_vector = _run_worker(model, "dsl-v2s")
    native_vector = _run_worker(model, "native-v2s")
    dsl_ckks = _run_worker(model, "dsl-s2c")
    dsl_c = _run_worker(model, "dsl-c")

    assert pre_inline["success"]
    assert pre_inline["stages"] == ["tensor2vector"]
    assert pre_inline["summary"]["helper_calls"] == 1
    assert pre_inline["summary"]["helper_scopes"] == 1

    assert dsl_vector["success"] and native_vector["success"]
    assert dsl_vector["stages"] == [
        "tensor2vector",
        "vector_kernel_inline",
        "vector2sihe",
    ]
    assert native_vector["stages"] == ["tensor2vector", "vector2sihe"]
    assert dsl_vector["summary"]["helper_calls"] == 0
    assert dsl_vector["summary"]["helper_scopes"] == 0
    assert dsl_vector["summary"]["helper_symbols"] == 0
    assert dsl_vector["summary"]["sihe_ops"] == native_vector["summary"]["sihe_ops"]
    assert dsl_vector["summary"]["rotations"] == native_vector["summary"]["rotations"]
    assert dsl_vector["summary"]["loops"] == native_vector["summary"]["loops"]

    assert dsl_ckks["success"], dsl_ckks["error"]
    assert dsl_ckks["stages"] == [
        "tensor2vector",
        "vector_kernel_inline",
        "vector2sihe",
        "sihe2ckks",
    ]
    assert dsl_ckks["summary"]["verify"]

    assert dsl_c["success"], dsl_c["error"]
    assert dsl_c["c_len"] > 0
    assert not dsl_c["helper_in_c"]


def _summary(glob):
    dump = glob.dump()
    return {
        "verify": glob.verify_ir(),
        "helper_calls": len(
            re.findall(rf'^\s+call "{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "helper_scopes": len(
            re.findall(rf'^FUN\[[^\n]*"{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "helper_symbols": len(
            re.findall(rf'^  FUN\[[^\n]*"{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "arg_stores": len(
            re.findall(r'^\s+st "__ace_inline_[^"]+_arg_0"', dump, re.MULTILINE)
        ),
        "vector_ops": re.findall(r"^\s+VECTOR\.([a-z_]+)", dump, re.MULTILINE),
        "sihe_ops": re.findall(r"^\s+SIHE\.([a-z_]+)", dump, re.MULTILINE),
        "rotations": re.findall(r"ATTR\[nums=([^\]]+)\]", dump),
        "loops": len(re.findall(r"^\s+do_loop ID", dump, re.MULTILINE)),
    }


def _pass_dict(result):
    value = dataclasses.asdict(result)
    value["diagnostics"] = list(value["diagnostics"])
    return value


def _new_pipeline(model: Path, implementation: str, artifact_tag: str = "pass"):
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.kernels.vector.baseline_gemm import baseline_gemm_recipe

    pipeline = Pipeline(
        "function-inliner-worker",
        output_dir=str(model.parent / f"output-{artifact_tag}"),
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))
    pipeline.configure_fhe(data_file=str(model.parent / f"{artifact_tag}.data.msg"))
    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=implementation,
        plan_kind="baseline-gemm",
        fallback="error",
        mask_fuse=False,
        max_slots=128,
    )
    if implementation == "dsl":
        pipeline.register_vector_kernel_recipe("baseline-gemm", baseline_gemm_recipe)
    return pipeline


def _worker_main():
    from ace_edsl.edsl.passes.function_inliner import FunctionInlinerPass
    from ace_edsl.edsl.pipeline import PipelineTarget

    model = Path(sys.argv[2])
    action = sys.argv[3]
    if action.startswith("dsl-") or action.startswith("native-"):
        implementation, target_name = action.split("-", 1)
        pipeline = _new_pipeline(model, implementation, action)
        targets = {
            "t2v": PipelineTarget.TENSOR2VECTOR,
            "v2s": PipelineTarget.VECTOR2SIHE,
            "s2c": PipelineTarget.SIHE2CKKS,
            "c": PipelineTarget.C,
        }
        result = pipeline.run(target=targets[target_name])
        payload = {
            "success": result.success,
            "error": result.error,
            "stages": result.stages_completed,
        }
        if target_name == "c":
            payload["c_len"] = len(result.c_code or "")
            payload["helper_in_c"] = _HELPER_PREFIX in (result.c_code or "")
        else:
            payload["summary"] = _summary(pipeline.glob)
        print(json.dumps(payload))
        return

    pipeline = _new_pipeline(model, "dsl", action)
    pipeline_result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    if action in ("extra-entry", "program-entry"):
        pipeline.glob._mutate_generated_vector_helper_for_testing(action)
    before_dump = pipeline.glob.dump()
    before_ptr = pipeline.glob.get_native_ptr()
    cached_type = pipeline.glob.get_type("f32")
    selection = []

    if action == "always":
        inline_result = FunctionInlinerPass.run(
            pipeline.glob,
            predicate=lambda descriptor: selection.append(descriptor) or True,
        )
    elif action == "never":
        inline_result = FunctionInlinerPass.run(pipeline.glob, policy="never")
    elif action == "predicate-false":
        inline_result = FunctionInlinerPass.run(
            pipeline.glob,
            predicate=lambda descriptor: selection.append(descriptor) or False,
        )
    elif action == "predicate-raise":

        def fail_predicate(_descriptor):
            raise RuntimeError("selection failure")

        inline_result = FunctionInlinerPass.run(pipeline.glob, predicate=fail_predicate)
    elif action == "invalid-policy":
        inline_result = FunctionInlinerPass.run(pipeline.glob, policy="unsupported")
    elif action in ("extra-entry", "program-entry"):
        inline_result = FunctionInlinerPass.run(pipeline.glob)
    else:
        raise ValueError(action)

    cached_type_usable = None
    if action == "always":
        cached_type_usable = bool(pipeline.glob.new_array_type([2], "f32"))
    stale_type_readable = cached_type.is_float()
    stale_type_is_foreign = not cached_type.same_scope(pipeline.glob.get_type("f32"))
    after_dump = pipeline.glob.dump()
    payload = {
        "pipeline_success": pipeline_result.success,
        "before": _summary_from_dump(before_dump, True),
        "after": _summary(pipeline.glob),
        "pass": _pass_dict(inline_result),
        "selection": selection,
        "dump_unchanged": before_dump == after_dump,
        "native_ptr_stable": before_ptr == pipeline.glob.get_native_ptr(),
    }
    if action == "always":
        payload["cached_type_usable"] = cached_type_usable
        payload["stale_type_readable"] = stale_type_readable
        payload["stale_type_is_foreign"] = stale_type_is_foreign
        second = FunctionInlinerPass.run(pipeline.glob)
        payload["second"] = _pass_dict(second)
        payload["second_dump_unchanged"] = pipeline.glob.dump() == after_dump
    print(json.dumps(payload))


def _summary_from_dump(dump: str, verify: bool):
    return {
        "verify": verify,
        "helper_calls": len(
            re.findall(rf'^\s+call "{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "helper_scopes": len(
            re.findall(rf'^FUN\[[^\n]*"{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "helper_symbols": len(
            re.findall(rf'^  FUN\[[^\n]*"{_HELPER_PREFIX}', dump, re.MULTILINE)
        ),
        "arg_stores": len(
            re.findall(r'^\s+st "__ace_inline_[^"]+_arg_0"', dump, re.MULTILINE)
        ),
        "vector_ops": re.findall(r"^\s+VECTOR\.([a-z_]+)", dump, re.MULTILINE),
        "sihe_ops": re.findall(r"^\s+SIHE\.([a-z_]+)", dump, re.MULTILINE),
        "rotations": re.findall(r"ATTR\[nums=([^\]]+)\]", dump),
        "loops": len(re.findall(r"^\s+do_loop ID", dump, re.MULTILINE)),
    }


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
