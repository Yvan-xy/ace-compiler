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
_CONV_HELPER_PREFIX = "__ace_vkernel_baseline_conv_"
_FAST_GEMM_HELPER_PREFIX = "__ace_vkernel_fast_gemm_"
_FAST_CONV_HELPER_PREFIX = "__ace_vkernel_fast_conv_"


def _write_gemm_model(
    path: Path,
    copies: int = 1,
    *,
    height: int = 2,
    width: int = 8,
):
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


def _write_conv_model(path: Path):
    channel_in = 4
    channel_out = 2
    height = 4
    width = 4
    kernel = 3
    weight_values = (
        np.arange(
            channel_out * channel_in * kernel * kernel,
            dtype=np.float32,
        )
        + 1.0
    ) / 32.0
    weight = numpy_helper.from_array(
        weight_values.reshape(
            channel_out, channel_in, kernel, kernel
        ),
        "weight",
    )
    bias = numpy_helper.from_array(
        np.array([0.25, -0.5], dtype=np.float32), "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, channel_in, height, width]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, channel_out, height, width]
    )
    conv = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        name="conv",
        kernel_shape=[kernel, kernel],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [conv],
        "function_inliner_conv",
        [input_info],
        [output_info],
        [weight, bias],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_fast_conv_model(path: Path, *, sharded: bool = False):
    channel_in = 1 if sharded else 2
    channel_out = 2 if sharded else 4
    height = 12 if sharded else 2
    kernel = 3
    weight_shape = (channel_out, channel_in, kernel, kernel)
    weight_values = (
        np.arange(np.prod(weight_shape), dtype=np.float32) + 1.0
    ) / 32.0
    weight = numpy_helper.from_array(
        weight_values.reshape(weight_shape), "weight"
    )
    bias = numpy_helper.from_array(
        np.arange(channel_out, dtype=np.float32) / 8.0, "bias"
    )
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, channel_in, height, height]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, channel_out, height, height]
    )
    conv = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        name="fast_conv",
        kernel_shape=[kernel, kernel],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [conv],
        "function_inliner_fast_conv",
        [input_info],
        [output_info],
        [weight, bias],
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


def test_fast_gemm_mutable_vector_array_inlines_and_is_idempotent(tmp_path):
    model = tmp_path / "fast_gemm.onnx"
    _write_gemm_model(model, height=4, width=4)

    result = _run_worker(model, "fast-always")

    assert result["pipeline_success"]
    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 1
    assert result["before"]["helper_scopes"] == 1
    assert result["before"]["mutable_array_names"] == ["__blocked_input"]
    assert result["before"]["ldas"] == 2
    assert result["before"]["ists"] == 1
    assert result["before"]["ilds"] == 2
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
    assert len(result["after"]["mutable_array_names"]) == 1
    assert re.fullmatch(
        r"__ace_inline_[0-9]+___blocked_input",
        result["after"]["mutable_array_names"][0],
    )
    for field in (
        "ldas",
        "arrays",
        "ists",
        "ilds",
        "loops",
        "vector_ops",
        "rotations",
        "slots",
        "comments",
    ):
        assert result["after"][field] == result["before"][field]
    assert result["second"] == {
        "success": True,
        "changed": False,
        "calls_inlined": 0,
        "helpers_removed": 0,
        "diagnostics": [],
    }
    assert result["second_dump_unchanged"]


def test_fast_gemm_shared_helper_clones_distinct_array_locals(tmp_path):
    model = tmp_path / "shared_fast_gemm.onnx"
    _write_gemm_model(model, copies=2, height=4, width=4)

    result = _run_worker(model, "fast-always")

    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 2
    assert result["before"]["helper_scopes"] == 1
    assert result["before"]["mutable_array_names"] == ["__blocked_input"]
    assert result["pass"]["success"]
    assert result["pass"]["calls_inlined"] == 2
    assert result["pass"]["helpers_removed"] == 1
    assert result["after"]["verify"]
    assert result["after"]["helper_calls"] == 0
    assert result["after"]["helper_scopes"] == 0
    assert result["after"]["helper_symbols"] == 0
    assert len(result["after"]["mutable_array_names"]) == 2
    assert len(set(result["after"]["mutable_array_names"])) == 2
    assert all(
        re.fullmatch(r"__ace_inline_[0-9]+___blocked_input", name)
        for name in result["after"]["mutable_array_names"]
    )
    for field in ("ldas", "arrays", "ists", "ilds"):
        assert result["after"][field] == 2 * result["before"][field]
    expected_vector_ops = 2 * result["before"]["vector_ops"]
    expected_vector_ops.remove("add")  # The caller-side combine Add is not cloned.
    assert sorted(result["after"]["vector_ops"]) == sorted(expected_vector_ops)


def test_fast_conv_complete_helper_inlines_transactionally(tmp_path):
    model = tmp_path / "fast_conv.onnx"
    _write_fast_conv_model(model)

    result = _run_worker(model, "fast-conv-always")

    assert result["pipeline_success"]
    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 1
    assert result["before"]["helper_scopes"] == 1
    assert result["before"]["mutable_array_names"] == ["__blocked_input"]
    assert result["before"]["loops"] == 3
    assert result["before"]["ists"] == 1
    assert result["before"]["ilds"] == 2
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
    assert len(result["after"]["mutable_array_names"]) == 1
    assert re.fullmatch(
        r"__ace_inline_[0-9]+___blocked_input",
        result["after"]["mutable_array_names"][0],
    )
    for field in (
        "ldas",
        "arrays",
        "ists",
        "ilds",
        "loops",
        "vector_ops",
        "rotations",
        "slots",
        "comments",
        "preg_loads",
    ):
        assert result["after"][field] == result["before"][field]
    assert (
        result["after"]["preg_stores"]
        == result["before"]["preg_stores"] + 1
    )
    assert result["second"] == {
        "success": True,
        "changed": False,
        "calls_inlined": 0,
        "helpers_removed": 0,
        "diagnostics": [],
    }
    assert result["second_dump_unchanged"]


def test_sharded_fast_conv_inliner_fails_closed_without_mutation(tmp_path):
    model = tmp_path / "sharded_fast_conv.onnx"
    _write_fast_conv_model(model, sharded=True)

    result = _run_worker(model, "fast-conv-sharded-always")

    assert result["pipeline_success"]
    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 1
    assert not result["pass"]["success"]
    assert not result["pass"]["changed"]
    assert result["pass"]["calls_inlined"] == 0
    assert result["pass"]["helpers_removed"] == 0
    assert "does not support scalar formals" in "; ".join(
        result["pass"]["diagnostics"]
    )
    assert result["dump_unchanged"]
    assert result["native_ptr_stable"]
    assert result["after"]["verify"]


def test_tentative_inliner_clones_conv_constant_array_access(tmp_path):
    model = tmp_path / "conv_constant_array.onnx"
    _write_conv_model(model)

    result = _run_worker(model, "conv-always")

    assert result["pipeline_success"]
    assert result["before"]["verify"]
    assert result["before"]["helper_calls"] == 1
    assert result["before"]["helper_scopes"] == 1
    assert result["before"]["ldcas"] == 1
    assert result["before"]["arrays"] == 1
    assert result["before"]["ilds"] == 1
    assert result["before"]["loops"] == 2
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
    assert result["after"]["ldcas"] == result["before"]["ldcas"]
    assert result["after"]["arrays"] == result["before"]["arrays"]
    assert result["after"]["ilds"] == result["before"]["ilds"]
    assert result["after"]["loops"] == result["before"]["loops"]
    assert result["after"]["vector_ops"] == result["before"]["vector_ops"]
    assert result["after"]["rotations"] == result["before"]["rotations"]
    assert result["second"] == {
        "success": True,
        "changed": False,
        "calls_inlined": 0,
        "helpers_removed": 0,
        "diagnostics": [],
    }
    assert result["second_dump_unchanged"]


@pytest.mark.parametrize(
    "action,success,diagnostic",
    [
        ("never", True, ""),
        ("predicate-false", True, ""),
        ("predicate-raise", False, "predicate failed: selection failure"),
        ("invalid-policy", False, "unsupported function-inliner policy"),
        ("extra-entry", False, "additional entry point"),
        ("program-entry", False, "one non-program entry point"),
        ("conv-array-non-pointer-rtype", False, "array address"),
        ("conv-escaping-ldca", False, "escaping address"),
        ("fast-mutable-array-escape", False, "escaping mutable array"),
        ("fast-mutable-array-global-base", False, "mutable array address"),
        ("fast-mutable-array-flat64-base", False, "mutable array address"),
        ("fast-mutable-array-index-i64", False, "Core signed i32"),
        (
            "fast-mutable-array-load-non-array-address",
            False,
            "indirect load requires an indexed array address",
        ),
        (
            "fast-mutable-array-store-non-array-address",
            False,
            "indirect store requires an indexed array address",
        ),
        (
            "fast-mutable-array-load-type-mismatch",
            False,
            "array load has incompatible",
        ),
        (
            "fast-mutable-array-store-type-mismatch",
            False,
            "mutable-array store has incompatible",
        ),
        ("fast-mutable-array-non-pointer-rtype", False, "array address"),
        ("fast-constant-array-store", False, "constant array address"),
        ("fast-array-arity", False, "array address"),
        ("fast-pointer-local", False, "pointer type"),
        ("fast-pointer-preg", False, "pointer type"),
        ("fast-pointer-global-load", False, "pointer-valued expression"),
        (
            "fast-conv-preg-load-type-mismatch",
            False,
            "preg load has an incompatible",
        ),
        (
            "fast-conv-preg-store-type-mismatch",
            False,
            "preg store has an incompatible",
        ),
        ("fast-unsupported-if", False, "unsupported block control flow"),
        ("fast-loop-iv-i64", False, "helper-owned Core s32 IV"),
    ],
)
def test_policy_and_predicate_failures_do_not_mutate(
    tmp_path, action, success, diagnostic
):
    model = tmp_path / f"{action}.onnx"
    if action.startswith("fast-conv-"):
        _write_fast_conv_model(model)
    elif action.startswith("conv-"):
        _write_conv_model(model)
    elif action.startswith("fast-"):
        _write_gemm_model(model, height=4, width=4)
    else:
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


def test_tentative_pipeline_gate_is_limited_to_supported_dsl_requests():
    from ace_edsl.edsl.pipeline import (
        VectorKernelLoweringConfig,
        _uses_tentative_vector_kernel_inliner,
    )

    base = VectorKernelLoweringConfig(kernel_impl="dsl")
    assert _uses_tentative_vector_kernel_inliner(base)
    assert _uses_tentative_vector_kernel_inliner(
        dataclasses.replace(base, plan_kind="baseline-gemm")
    )
    assert _uses_tentative_vector_kernel_inliner(
        dataclasses.replace(base, plan_kind="baseline-conv")
    )
    assert not _uses_tentative_vector_kernel_inliner(None)
    assert not _uses_tentative_vector_kernel_inliner(
        dataclasses.replace(base, kernel_impl="native")
    )
    assert _uses_tentative_vector_kernel_inliner(
        dataclasses.replace(base, plan_kind="fast-gemm")
    )
    assert _uses_tentative_vector_kernel_inliner(
        dataclasses.replace(base, plan_kind="fast-conv")
    )


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


def test_fast_gemm_pipeline_boundary_and_native_parity(tmp_path):
    model = tmp_path / "pipeline_fast_gemm.onnx"
    _write_gemm_model(model, height=4, width=4)

    pre_inline = _run_worker(model, "fast-dsl-t2v")
    auto_pre_inline = _run_worker(model, "fast-dsl-auto-t2v")
    dsl_vector = _run_worker(model, "fast-dsl-v2s")
    auto_dsl_vector = _run_worker(model, "fast-dsl-auto-v2s")
    native_vector = _run_worker(model, "fast-native-v2s")
    dsl_ckks = _run_worker(model, "fast-dsl-s2c")
    dsl_c = _run_worker(model, "fast-dsl-c")

    assert pre_inline["success"]
    assert pre_inline["stages"] == ["tensor2vector"]
    assert pre_inline["summary"]["helper_calls"] == 1
    assert pre_inline["summary"]["helper_scopes"] == 1
    assert auto_pre_inline["success"], auto_pre_inline["error"]
    assert auto_pre_inline["stages"] == ["tensor2vector"]
    assert auto_pre_inline["summary"]["helper_calls"] == 1
    assert auto_pre_inline["summary"]["helper_scopes"] == 1

    assert dsl_vector["success"], dsl_vector["error"]
    assert auto_dsl_vector["success"], auto_dsl_vector["error"]
    assert native_vector["success"], native_vector["error"]
    assert dsl_vector["stages"] == [
        "tensor2vector",
        "vector_kernel_inline",
        "vector2sihe",
    ]
    assert native_vector["stages"] == ["tensor2vector", "vector2sihe"]
    assert dsl_vector["summary"]["helper_calls"] == 0
    assert dsl_vector["summary"]["helper_scopes"] == 0
    assert dsl_vector["summary"]["helper_symbols"] == 0
    assert auto_dsl_vector["stages"] == dsl_vector["stages"]
    assert auto_dsl_vector["summary"]["helper_calls"] == 0
    assert auto_dsl_vector["summary"]["helper_scopes"] == 0
    assert auto_dsl_vector["summary"]["helper_symbols"] == 0
    assert auto_dsl_vector["summary"]["sihe_ops"] == dsl_vector["summary"]["sihe_ops"]
    assert auto_dsl_vector["summary"]["rotations"] == dsl_vector["summary"]["rotations"]
    assert auto_dsl_vector["summary"]["loops"] == dsl_vector["summary"]["loops"]
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


def test_baseline_conv_pipeline_boundary_and_native_parity(tmp_path):
    model = tmp_path / "pipeline_conv.onnx"
    _write_conv_model(model)

    pre_inline = _run_worker(model, "conv-dsl-t2v")
    dsl_vector = _run_worker(model, "conv-dsl-v2s")
    dsl_auto_vector = _run_worker(model, "conv-dsl-auto-v2s")
    native_vector = _run_worker(model, "conv-native-v2s")
    dsl_ckks = _run_worker(model, "conv-dsl-s2c")
    dsl_c = _run_worker(model, "conv-dsl-c")

    assert pre_inline["success"]
    assert pre_inline["stages"] == ["tensor2vector"]
    assert pre_inline["summary"]["helper_calls"] == 1
    assert pre_inline["summary"]["helper_scopes"] == 1

    assert dsl_vector["success"], dsl_vector["error"]
    assert dsl_auto_vector["success"], dsl_auto_vector["error"]
    assert native_vector["success"], native_vector["error"]
    assert dsl_vector["stages"] == [
        "tensor2vector",
        "vector_kernel_inline",
        "vector2sihe",
    ]
    assert dsl_auto_vector["stages"] == dsl_vector["stages"]
    assert native_vector["stages"] == ["tensor2vector", "vector2sihe"]
    for candidate in (dsl_vector, dsl_auto_vector):
        assert candidate["summary"]["helper_calls"] == 0
        assert candidate["summary"]["helper_scopes"] == 0
        assert candidate["summary"]["helper_symbols"] == 0
        assert (
            candidate["summary"]["sihe_ops"]
            == native_vector["summary"]["sihe_ops"]
        )
        assert (
            candidate["summary"]["rotations"]
            == native_vector["summary"]["rotations"]
        )
        assert (
            candidate["summary"]["loops"]
            == native_vector["summary"]["loops"]
        )

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


def test_fast_conv_pipeline_boundary_and_native_parity(tmp_path):
    model = tmp_path / "pipeline_fast_conv.onnx"
    _write_fast_conv_model(model)

    pre_inline = _run_worker(model, "fast-conv-dsl-t2v")
    dsl_vector = _run_worker(model, "fast-conv-dsl-v2s")
    native_vector = _run_worker(model, "fast-conv-native-v2s")
    dsl_ckks = _run_worker(model, "fast-conv-dsl-s2c")
    dsl_c = _run_worker(model, "fast-conv-dsl-c")

    assert pre_inline["success"], pre_inline["error"]
    assert pre_inline["stages"] == ["tensor2vector"]
    assert pre_inline["summary"]["helper_calls"] == 1
    assert pre_inline["summary"]["helper_scopes"] == 1

    assert dsl_vector["success"], dsl_vector["error"]
    assert native_vector["success"], native_vector["error"]
    assert dsl_vector["stages"] == [
        "tensor2vector",
        "vector_kernel_inline",
        "vector2sihe",
    ]
    assert native_vector["stages"] == ["tensor2vector", "vector2sihe"]
    assert dsl_vector["summary"]["helper_calls"] == 0
    assert dsl_vector["summary"]["helper_scopes"] == 0
    assert dsl_vector["summary"]["helper_symbols"] == 0
    assert (
        dsl_vector["summary"]["sihe_ops"]
        == native_vector["summary"]["sihe_ops"]
    )
    assert (
        dsl_vector["summary"]["rotations"]
        == native_vector["summary"]["rotations"]
    )
    assert (
        dsl_vector["summary"]["loops"]
        == native_vector["summary"]["loops"]
    )

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


def _summary(glob, helper_prefix=_HELPER_PREFIX):
    dump = glob.dump()
    return {
        "verify": glob.verify_ir(),
        "helper_calls": len(
            re.findall(rf'^\s+call "{helper_prefix}', dump, re.MULTILINE)
        ),
        "helper_scopes": len(
            re.findall(rf'^FUN\[[^\n]*"{helper_prefix}', dump, re.MULTILINE)
        ),
        "helper_symbols": len(
            re.findall(rf'^  FUN\[[^\n]*"{helper_prefix}', dump, re.MULTILINE)
        ),
        "arg_stores": len(
            re.findall(r'^\s+st "__ace_inline_[^"]+_arg_0"', dump, re.MULTILINE)
        ),
        "vector_ops": re.findall(r"^\s+VECTOR\.([a-z_]+)", dump, re.MULTILINE),
        "sihe_ops": re.findall(r"^\s+SIHE\.([a-z_]+)", dump, re.MULTILINE),
        "rotations": re.findall(r"ATTR\[nums=([^\]]+)\]", dump),
        "loops": len(re.findall(r"^\s+do_loop ID", dump, re.MULTILINE)),
        "ldcas": len(re.findall(r"^\s+ldca ", dump, re.MULTILINE)),
        "ldas": len(re.findall(r"^\s+lda ", dump, re.MULTILINE)),
        "arrays": len(re.findall(r"^\s+array ", dump, re.MULTILINE)),
        "ilds": len(re.findall(r"^\s+ild ", dump, re.MULTILINE)),
        "ists": len(re.findall(r"^\s+ist ", dump, re.MULTILINE)),
        "preg_stores": len(re.findall(r"^\s+stp ", dump, re.MULTILINE)),
        "preg_loads": len(re.findall(r"^\s+ldp ", dump, re.MULTILINE)),
        "slots": re.findall(r"ATTR\[slot=([^]\s]+)\]", dump),
        "comments": re.findall(r'^\s+comment "([^"]*)"', dump, re.MULTILINE),
        "mutable_array_names": sorted(
            set(
                re.findall(
                    r'^\s+lda "([^"]*__blocked_input)"',
                    dump,
                    re.MULTILINE,
                )
            )
        ),
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


def _new_fast_gemm_pipeline(
    model: Path,
    implementation: str,
    artifact_tag: str = "fast-pass",
    *,
    auto_plan: bool = False,
):
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.kernels.vector.fast_gemm import fast_gemm_recipe

    pipeline = Pipeline(
        "function-inliner-fast-gemm-worker",
        output_dir=str(model.parent / f"output-{artifact_tag}"),
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))
    pipeline.configure_fhe(data_file=str(model.parent / f"{artifact_tag}.data.msg"))
    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=implementation,
        plan_kind="auto" if auto_plan else "fast-gemm",
        fallback="error",
        mask_fuse=False,
        max_slots=128,
    )
    if implementation == "dsl":
        pipeline.register_vector_kernel_recipe("fast-gemm", fast_gemm_recipe)
    return pipeline


def _new_conv_pipeline(
    model: Path,
    implementation: str,
    artifact_tag: str = "conv-pass",
    *,
    auto_plan: bool = False,
):
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.kernels.vector.baseline_conv import baseline_conv_recipe

    pipeline = Pipeline(
        "function-inliner-conv-worker",
        output_dir=str(model.parent / f"output-{artifact_tag}"),
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))
    pipeline.configure_fhe(data_file=str(model.parent / f"{artifact_tag}.data.msg"))
    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=implementation,
        plan_kind="auto" if auto_plan else "baseline-conv",
        fallback="error",
        mask_fuse=False,
        max_slots=128,
    )
    if implementation == "dsl":
        pipeline.register_vector_kernel_recipe("baseline-conv", baseline_conv_recipe)
    return pipeline


def _new_fast_conv_pipeline(
    model: Path,
    implementation: str,
    artifact_tag: str = "fast-conv-pass",
    *,
    sharding: bool = False,
):
    from ace_edsl.edsl.pipeline import Pipeline
    from ace_edsl.edsl.kernels.vector.fast_conv import fast_conv_recipe

    pipeline = Pipeline(
        "function-inliner-fast-conv-worker",
        output_dir=str(model.parent / f"output-{artifact_tag}"),
        dump_ir=False,
        verbose=False,
    ).load_onnx(str(model))
    pipeline.configure_fhe(
        data_file=str(model.parent / f"{artifact_tag}.data.msg")
    )
    pipeline.configure_vector_kernel_lowering(
        plan_provider="cpp",
        kernel_impl=implementation,
        plan_kind="fast-conv",
        fallback="error",
        mask_fuse=False,
        max_slots=128,
        sharding=sharding,
    )
    if implementation == "dsl":
        pipeline.register_vector_kernel_recipe("fast-conv", fast_conv_recipe)
    return pipeline


def _worker_main():
    from ace_edsl.edsl.passes.function_inliner import FunctionInlinerPass
    from ace_edsl.edsl.pipeline import PipelineTarget

    model = Path(sys.argv[2])
    requested_action = sys.argv[3]
    is_fast_conv = requested_action.startswith("fast-conv-")
    is_conv = requested_action.startswith("conv-")
    is_fast_gemm = requested_action.startswith("fast-") and not is_fast_conv
    if is_fast_conv:
        action = requested_action.removeprefix("fast-conv-")
        helper_prefix = _FAST_CONV_HELPER_PREFIX
    elif is_conv:
        action = requested_action.removeprefix("conv-")
        helper_prefix = _CONV_HELPER_PREFIX
    elif is_fast_gemm:
        action = requested_action.removeprefix("fast-")
        helper_prefix = _FAST_GEMM_HELPER_PREFIX
    else:
        action = requested_action
        helper_prefix = _HELPER_PREFIX
    sharded_fast_conv = is_fast_conv and action.startswith("sharded-")
    if sharded_fast_conv:
        action = action.removeprefix("sharded-")
    if action.startswith("dsl-") or action.startswith("native-"):
        auto_plan = action.startswith("dsl-auto-")
        if auto_plan:
            implementation = "dsl"
            target_name = action.removeprefix("dsl-auto-")
        else:
            implementation, target_name = action.split("-", 1)
        if is_fast_conv:
            pipeline = _new_fast_conv_pipeline(
                model,
                implementation,
                requested_action,
                sharding=sharded_fast_conv,
            )
        elif is_conv:
            pipeline = _new_conv_pipeline(
                model,
                implementation,
                requested_action,
                auto_plan=auto_plan,
            )
        elif is_fast_gemm:
            pipeline = _new_fast_gemm_pipeline(
                model,
                implementation,
                requested_action,
                auto_plan=auto_plan,
            )
        else:
            pipeline = _new_pipeline(model, implementation, requested_action)
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
            payload["helper_in_c"] = helper_prefix in (result.c_code or "")
        else:
            payload["summary"] = _summary(pipeline.glob, helper_prefix)
        print(json.dumps(payload))
        return

    if is_fast_conv:
        pipeline = _new_fast_conv_pipeline(
            model,
            "dsl",
            requested_action,
            sharding=sharded_fast_conv,
        )
    elif is_conv:
        pipeline = _new_conv_pipeline(model, "dsl", requested_action)
    elif is_fast_gemm:
        pipeline = _new_fast_gemm_pipeline(model, "dsl", requested_action)
    else:
        pipeline = _new_pipeline(model, "dsl", requested_action)
    pipeline_result = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    if action in (
        "extra-entry",
        "program-entry",
        "array-non-pointer-rtype",
        "escaping-ldca",
        "mutable-array-escape",
        "mutable-array-global-base",
        "mutable-array-flat64-base",
        "mutable-array-index-i64",
        "mutable-array-load-non-array-address",
        "mutable-array-store-non-array-address",
        "mutable-array-load-type-mismatch",
        "mutable-array-store-type-mismatch",
        "mutable-array-non-pointer-rtype",
        "constant-array-store",
        "array-arity",
        "pointer-local",
        "pointer-preg",
        "pointer-global-load",
        "preg-load-type-mismatch",
        "preg-store-type-mismatch",
        "unsupported-if",
        "loop-iv-i64",
    ):
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
    elif action in (
        "extra-entry",
        "program-entry",
        "array-non-pointer-rtype",
        "escaping-ldca",
        "mutable-array-escape",
        "mutable-array-global-base",
        "mutable-array-flat64-base",
        "mutable-array-index-i64",
        "mutable-array-load-non-array-address",
        "mutable-array-store-non-array-address",
        "mutable-array-load-type-mismatch",
        "mutable-array-store-type-mismatch",
        "mutable-array-non-pointer-rtype",
        "constant-array-store",
        "array-arity",
        "pointer-local",
        "pointer-preg",
        "pointer-global-load",
        "preg-load-type-mismatch",
        "preg-store-type-mismatch",
        "unsupported-if",
        "loop-iv-i64",
    ):
        inline_result = FunctionInlinerPass.run(pipeline.glob)
    else:
        raise ValueError(action)

    committed = action == "always" and inline_result.success
    cached_type_usable = None
    if committed:
        cached_type_usable = bool(pipeline.glob.new_array_type([2], "f32"))
    stale_type_readable = cached_type.is_float()
    stale_type_is_foreign = not cached_type.same_scope(pipeline.glob.get_type("f32"))
    after_dump = pipeline.glob.dump()
    payload = {
        "pipeline_success": pipeline_result.success,
        "before": _summary_from_dump(before_dump, True, helper_prefix),
        "after": _summary(pipeline.glob, helper_prefix),
        "pass": _pass_dict(inline_result),
        "selection": selection,
        "dump_unchanged": before_dump == after_dump,
        "native_ptr_stable": before_ptr == pipeline.glob.get_native_ptr(),
    }
    if committed:
        payload["cached_type_usable"] = cached_type_usable
        payload["stale_type_readable"] = stale_type_readable
        payload["stale_type_is_foreign"] = stale_type_is_foreign
        second = FunctionInlinerPass.run(pipeline.glob)
        payload["second"] = _pass_dict(second)
        payload["second_dump_unchanged"] = pipeline.glob.dump() == after_dump
    print(json.dumps(payload))


def _summary_from_dump(dump: str, verify: bool, helper_prefix=_HELPER_PREFIX):
    return {
        "verify": verify,
        "helper_calls": len(
            re.findall(rf'^\s+call "{helper_prefix}', dump, re.MULTILINE)
        ),
        "helper_scopes": len(
            re.findall(rf'^FUN\[[^\n]*"{helper_prefix}', dump, re.MULTILINE)
        ),
        "helper_symbols": len(
            re.findall(rf'^  FUN\[[^\n]*"{helper_prefix}', dump, re.MULTILINE)
        ),
        "arg_stores": len(
            re.findall(r'^\s+st "__ace_inline_[^"]+_arg_0"', dump, re.MULTILINE)
        ),
        "vector_ops": re.findall(r"^\s+VECTOR\.([a-z_]+)", dump, re.MULTILINE),
        "sihe_ops": re.findall(r"^\s+SIHE\.([a-z_]+)", dump, re.MULTILINE),
        "rotations": re.findall(r"ATTR\[nums=([^\]]+)\]", dump),
        "loops": len(re.findall(r"^\s+do_loop ID", dump, re.MULTILINE)),
        "ldcas": len(re.findall(r"^\s+ldca ", dump, re.MULTILINE)),
        "ldas": len(re.findall(r"^\s+lda ", dump, re.MULTILINE)),
        "arrays": len(re.findall(r"^\s+array ", dump, re.MULTILINE)),
        "ilds": len(re.findall(r"^\s+ild ", dump, re.MULTILINE)),
        "ists": len(re.findall(r"^\s+ist ", dump, re.MULTILINE)),
        "preg_stores": len(re.findall(r"^\s+stp ", dump, re.MULTILINE)),
        "preg_loads": len(re.findall(r"^\s+ldp ", dump, re.MULTILINE)),
        "slots": re.findall(r"ATTR\[slot=([^]\s]+)\]", dump),
        "comments": re.findall(r'^\s+comment "([^"]*)"', dump, re.MULTILINE),
        "mutable_array_names": sorted(
            set(
                re.findall(
                    r'^\s+lda "([^"]*__blocked_input)"',
                    dump,
                    re.MULTILINE,
                )
            )
        ),
    }


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
