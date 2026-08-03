"""Typed direct NN Conv/Gemm authoring and pre-inline Vector integration."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import pytest

from ace_bindings import air_builder
from ace_edsl.edsl import (
    AceEDSL,
    Tensor,
    VectorTensor,
    nn_kernel,
    tensor_ops,
    vector_kernel,
)


_WORKER_FLAG = "--nn-authoring-worker"
_REPO_ROOT = Path(__file__).resolve().parents[2]


@nn_kernel
def _direct_baseline_gemm(
    value: Tensor[float, 1, 4],
) -> Tensor[float, 1, 2]:
    weight = tensor_ops.constant(
        [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        shape=(2, 4),
    )
    bias = tensor_ops.constant([0.25, -0.5])
    return tensor_ops.gemm(value, weight, bias, 1.0, 1.0, 0, 1)


@nn_kernel
def _direct_fast_gemm(
    value: Tensor[float, 1, 4],
) -> Tensor[float, 1, 4]:
    weight = tensor_ops.constant(
        [
            0.0625,
            0.125,
            0.1875,
            0.25,
            0.3125,
            0.375,
            0.4375,
            0.5,
            0.5625,
            0.625,
            0.6875,
            0.75,
            0.8125,
            0.875,
            0.9375,
            1.0,
        ],
        shape=(4, 4),
        dtype="float32",
    ).view("nn::core")
    bias = tensor_ops.constant(
        [0.0, 0.25, 0.5, 0.75], dtype=float
    ).view("nn::core")
    return tensor_ops.gemm(value, weight, bias)


@nn_kernel
def _direct_baseline_conv(
    value: Tensor[float, 1, 4, 4, 4],
) -> Tensor[float, 1, 2, 4, 4]:
    weight = tensor_ops.constant([0.03125] * 72, shape=(2, 4, 3, 3))
    bias = tensor_ops.constant([0.25, -0.5])
    return tensor_ops.conv(
        value,
        weight,
        bias,
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        kernel_shape=(3, 3),
        group=1,
    )


@nn_kernel
def _direct_fast_depthwise_conv(
    value: Tensor[float, 1, 4, 4, 4],
) -> Tensor[float, 1, 4, 4, 4]:
    weight = tensor_ops.constant([0.0625] * 36, shape=(4, 1, 3, 3))
    bias = tensor_ops.constant([0.0, 0.125, 0.25, 0.375])
    return tensor_ops.conv(
        value,
        weight,
        bias,
        stride=1,
        padding=1,
        dilation=1,
        kernel_size=3,
        group=4,
    )


@nn_kernel
def _gemm_with_unsupported_alpha(
    value: Tensor[float, 1, 4],
) -> Tensor[float, 1, 2]:
    weight = tensor_ops.constant([0.125] * 8, shape=(2, 4))
    bias = tensor_ops.constant([0.0, 0.0])
    return tensor_ops.gemm(value, weight, bias, alpha=2.0)


@nn_kernel
def _gemm_without_bias(
    value: Tensor[float, 1, 4],
) -> Tensor[float, 1, 2]:
    weight = tensor_ops.constant([0.125] * 8, shape=(2, 4))
    return tensor_ops.gemm(value, weight)


@nn_kernel
def _conv_with_unsupported_stride(
    value: Tensor[float, 1, 2, 4, 4],
) -> Tensor[float, 1, 2, 4, 4]:
    weight = tensor_ops.constant([0.125] * 36, shape=(2, 2, 3, 3))
    bias = tensor_ops.constant([0.0, 0.0])
    return tensor_ops.conv(
        value,
        weight,
        bias,
        strides=(2, 2),
        pads=(1, 1, 1, 1),
        dilations=(1, 1),
        kernel_shape=(3, 3),
    )


@nn_kernel
def _gemm_with_formal_weight(
    value: Tensor[float, 1, 4],
    weight: Tensor[float, 2, 4],
    bias: Tensor[float, 2],
) -> Tensor[float, 1, 2]:
    return tensor_ops.gemm(value, weight, bias)


@vector_kernel
def _constant_in_wrong_domain(
    value: VectorTensor[float, 4],
) -> VectorTensor[float, 4]:
    tensor_ops.constant([0.0, 0.0, 0.0, 0.0])
    return value


@vector_kernel
def _gemm_in_wrong_domain(
    value: VectorTensor[float, 4],
) -> VectorTensor[float, 4]:
    return tensor_ops.gemm(value, value, value)


@vector_kernel
def _conv_in_wrong_domain(
    value: VectorTensor[float, 4],
) -> VectorTensor[float, 4]:
    return tensor_ops.conv(value, value, value)


_CASES = {
    "baseline-gemm": {
        "kernel": _direct_baseline_gemm,
        "input": Tensor[float, 1, 4],
        "result_shape": [1, 2],
        "operation": "gemm",
        "attrs": "ATTR[transB=1,transA=0,beta=1,alpha=1]",
        "inspector": "_inspect_baseline_gemm_air_for_testing",
    },
    "fast-gemm": {
        "kernel": _direct_fast_gemm,
        "input": Tensor[float, 1, 4],
        "result_shape": [1, 4],
        "operation": "gemm",
        "attrs": "ATTR[transB=1,transA=0,beta=1,alpha=1]",
        "inspector": "_inspect_fast_gemm_air_for_testing",
    },
    "baseline-conv": {
        "kernel": _direct_baseline_conv,
        "input": Tensor[float, 1, 4, 4, 4],
        "result_shape": [1, 2, 4, 4],
        "operation": "conv",
        "attrs": (
            "ATTR[strides=(1,1),pads=(1,1,1,1),"
            "kernel_shape=(3,3),group=1,dilations=(1,1)]"
        ),
        "inspector": "_inspect_baseline_conv_air_for_testing",
    },
    "fast-conv": {
        "kernel": _direct_fast_depthwise_conv,
        "input": Tensor[float, 1, 4, 4, 4],
        "result_shape": [1, 4, 4, 4],
        "operation": "conv",
        "attrs": (
            "ATTR[strides=(1,1),pads=(1,1,1,1),"
            "kernel_shape=(3,3),group=4,dilations=(1,1)]"
        ),
        "inspector": "_inspect_fast_conv_air_for_testing",
    },
}


_CASE_SETTINGS = {
    "baseline-gemm": {"mask_fuse": True, "max_slots": 0},
    "fast-gemm": {"mask_fuse": False, "max_slots": 128},
    "baseline-conv": {"mask_fuse": False, "max_slots": 128},
    "fast-conv": {"mask_fuse": False, "max_slots": 128},
}


def _canonical_constant_hash(constant):
    shape = ",".join(str(dimension) for dimension in constant["shape"])
    header = (
        "vector-kernel-constant:v1\n"
        f"element={constant['element_type']}\n"
        f"rank={len(constant['shape'])}\n"
        f"shape={shape}\npayload:\n"
    ).encode("ascii")
    payload = bytes.fromhex(constant["bytes_hex"])
    return "sha256:" + hashlib.sha256(header + payload).hexdigest()


def _run_worker(case_name: str, implementation: str):
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(_REPO_ROOT)
        if not old_pythonpath
        else os.pathsep.join((str(_REPO_ROOT), old_pythonpath))
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
        case_name,
        implementation,
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        diagnostics = "\n".join((completed.stdout, completed.stderr))
        assert implementation == "missing", (
            f"worker failed ({' '.join(command)}):\n{diagnostics}"
        )
        return {
            "success": False,
            "fatal": True,
            "error": diagnostics,
            "worker_diagnostics": diagnostics,
            "fallback_log": "vector-kernel fallback:" in diagnostics,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, f"worker produced no result; stderr:\n{completed.stderr}"
    payload = json.loads(lines[-1])
    diagnostics = "\n".join((*lines[:-1], completed.stderr))
    payload["worker_diagnostics"] = diagnostics
    payload["fallback_log"] = "vector-kernel fallback:" in diagnostics
    return payload


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_direct_nn_source_has_complete_schema_and_exact_ranked_types(case_name):
    result = _run_worker(case_name, "source")
    case = _CASES[case_name]

    assert result["initial_verify"]
    assert result["source_result_shape"] == case["result_shape"]
    assert result["source_result_element"] == "f32"
    assert result["source_nn_count"] == 1
    assert result["source_ldc_count"] == 2
    assert result["source_constant_count"] == 2
    assert result["source_vector_count"] == 0
    assert case["attrs"] in result["initial_dump"]


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_cpp_native_and_dsl_lower_direct_nn_with_structural_parity(case_name):
    native = _run_worker(case_name, "native")
    dsl = _run_worker(case_name, "dsl")

    assert native["success"] and dsl["success"]
    assert native["verify"] and dsl["verify"]
    assert native["stages_completed"] == dsl["stages_completed"] == [
        "tensor2vector"
    ]
    assert native["fallback"] == dsl["fallback"] == "error"
    settings = _CASE_SETTINGS[case_name]
    for result, implementation in ((native, "native"), (dsl, "dsl")):
        assert result["selection"] == {
            "plan_provider": "cpp",
            "kernel_impl": implementation,
            "plan_kind": case_name,
            "fallback": "error",
            "mask_fuse": settings["mask_fuse"],
            "max_slots": settings["max_slots"],
            "conv_parallel": False,
            "sharding": False,
        }
        assert not result["fallback_log"], result["worker_diagnostics"]
    assert not native["has_source_nn"] and not dsl["has_source_nn"]
    assert native["recipe_calls"] == []
    assert dsl["recipe_calls"] == [case_name]
    assert native["helper_count"] == native["helper_calls"] == 0
    assert dsl["helper_count"] == dsl["helper_calls"] == 1
    assert native["oracle"]["kind"] == "native"
    assert dsl["oracle"]["kind"] == "helper"
    assert dsl["oracle"]["bridge_ok"]
    assert dsl["oracle"]["prepared_input_ok"]
    prepared = dsl["prepared"]
    assert prepared["kind"] == case_name
    assert prepared["provenance"] == "cpp"
    assert prepared["owned_constants"]
    assert {"weight", "bias"} <= set(prepared["constant_roles"])
    assert all(
        item["shape"] and all(dimension > 0 for dimension in item["shape"])
        for item in prepared["constant_types"]
    )
    types_by_role = {item["role"]: item for item in prepared["constant_types"]}
    assert types_by_role["weight"]["element_type"] == "f32"
    assert types_by_role["bias"]["element_type"] == "f32"
    for constant in prepared["constant_types"]:
        assert constant["content_hash"] == _canonical_constant_hash(constant)
        assert len(bytes.fromhex(constant["bytes_hex"])) == (
            4 * math.prod(constant["shape"])
        )

    retained_oracle = dsl["oracle"]["native_normalized"]
    if case_name == "baseline-gemm":
        assert retained_oracle is None
        native_references = [native["oracle"]["normalized"]]
    else:
        assert retained_oracle is not None
        native_references = [retained_oracle]
        if case_name == "fast-gemm":
            native_references.append(native["oracle"]["normalized"])
    for native_reference in native_references:
        comparison = air_builder._compare_normalized_vector_kernel_air_for_testing(
            native_reference, dsl["oracle"]["normalized"]
        )
        assert comparison["equal"], comparison


@pytest.mark.parametrize("case_name", ("fast-gemm", "baseline-conv"))
def test_unconfigured_direct_nn_matches_explicit_cpp_native_auto(case_name):
    default = _run_worker(case_name, "default")
    explicit = _run_worker(case_name, "native-auto")

    assert default["success"] and explicit["success"]
    assert default["verify"] and explicit["verify"]
    assert default["stages_completed"] == explicit["stages_completed"] == [
        "tensor2vector"
    ]
    assert default["selection"] is None
    assert explicit["selection"] == {
        "plan_provider": "cpp",
        "kernel_impl": "native",
        "plan_kind": "auto",
        "fallback": "error",
        "mask_fuse": False,
        "max_slots": 0,
        "conv_parallel": False,
        "sharding": False,
    }
    for result in (default, explicit):
        assert not result["fallback_log"], result["worker_diagnostics"]
        assert result["recipe_calls"] == []
        assert result["oracle"]["kind"] == "native"
    assert default["helper_count"] == explicit["helper_count"] == 0
    assert default["helper_calls"] == explicit["helper_calls"] == 0
    assert not default["has_source_nn"] and not explicit["has_source_nn"]
    comparison = air_builder._compare_normalized_vector_kernel_air_for_testing(
        default["oracle"]["normalized"], explicit["oracle"]["normalized"]
    )
    assert comparison["equal"], comparison


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_missing_dsl_recipe_with_error_fallback_fails_in_isolated_process(case_name):
    result = _run_worker(case_name, "missing")

    assert not result["success"]
    assert result["fatal"]
    assert not result["fallback_log"], result["worker_diagnostics"]
    assert "no DSL recipe is registered for the validated plan kind" in result[
        "error"
    ]


def _new_scope(name, input_shape, result_shape, element="f32"):
    glob = air_builder.GlobScope()
    input_type = glob.new_array_type(input_shape, element)
    result_type = glob.new_array_type(result_shape, "f32")
    scope = glob.new_func_with_param_types(name, result_type, [input_type])
    return glob, scope, scope.container(), scope.new_param("input", input_type)


def test_ranked_f32_constant_preserves_shape_type_and_rejects_bad_payloads():
    glob, _, container, _ = _new_scope("ranked_constant", [1, 4], [1, 4])
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected_bytes = struct.pack("<6f", *values)
    constant = container.new_ranked_f32_const(values, [2, 3])
    constant_type = constant.rtype()
    assert constant_type.shape() == [2, 3]
    assert constant_type.element_type().to_string() == "f32"
    assert constant.inline_array_constant_bytes() == expected_bytes

    values[:] = [9.0] * 6
    assert constant.inline_array_constant_bytes() == expected_bytes

    for values, shape, message in (
        ([1.0], [2], "value count"),
        ([1.0], [0], "positive dimensions"),
        ([True], [1], "not bool"),
        ([1 + 2j], [1], "real numbers"),
        ([], [sys.maxsize, 2], "overflows"),
    ):
        before = container.dump()
        with pytest.raises(RuntimeError, match=message):
            container.new_ranked_f32_const(values, shape)
        assert container.dump() == before
    assert glob.verify_ir()


@pytest.mark.parametrize(
    "call,message",
    (
        (lambda: tensor_ops._flatten_ranked_values([[1.0], [2.0, 3.0]]), "rectangular"),
        (lambda: tensor_ops.constant([1.0], dtype="f64"), "only f32"),
        (lambda: tensor_ops.constant([1.0], shape=(2,)), "value count"),
        (lambda: tensor_ops.constant([1.0], shape=(0,)), "positive"),
    ),
)
def test_python_constant_validation_rejects_invalid_specs_before_tracing(
    call, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        call()


def test_nn_authoring_requires_an_active_nn_lexical_domain():
    AceEDSL._get_dsl.cache_clear()
    with pytest.raises(RuntimeError, match="active nn::core kernel trace"):
        tensor_ops.constant([0.0])

    for kernel in (
        _constant_in_wrong_domain,
        _gemm_in_wrong_domain,
        _conv_in_wrong_domain,
    ):
        AceEDSL._get_dsl.cache_clear()
        with pytest.raises(RuntimeError, match="active nn::core kernel trace"):
            kernel(VectorTensor[float, 4])


@pytest.mark.parametrize(
    "kernel,args,message,_opcode",
    (
        (
            _gemm_with_unsupported_alpha,
            (Tensor[float, 1, 4],),
            "supports alpha=1",
            "gemm",
        ),
        (
            _gemm_without_bias,
            (Tensor[float, 1, 4],),
            "requires an inline ranked bias",
            "gemm",
        ),
        (
            _conv_with_unsupported_stride,
            (Tensor[float, 1, 2, 4, 4],),
            "unit authored strides",
            "conv",
        ),
        (
            _gemm_with_formal_weight,
            (
                Tensor[float, 1, 4],
                Tensor[float, 2, 4],
                Tensor[float, 2],
            ),
            "weight must be an inline",
            "gemm",
        ),
    ),
)
def test_public_nn_authoring_rejects_unsupported_or_unowned_inputs(
    kernel, args, message, _opcode
):
    AceEDSL._get_dsl.cache_clear()
    with pytest.raises(RuntimeError, match=message):
        kernel(*args)
    assert AceEDSL._get_dsl().current_air_module is None


def test_raw_typed_builders_infer_heterogeneous_results_and_complete_attrs():
    gemm_glob, _, gemm_container, gemm_input = _new_scope(
        "typed_gemm", [1, 4], [1, 3]
    )
    gemm_weight = gemm_container.new_ranked_f32_const([0.125] * 12, [3, 4])
    gemm_bias = gemm_container.new_ranked_f32_const([0.0] * 3, [3])
    gemm = gemm_container.new_nn_gemm(
        gemm_input, gemm_weight, gemm_bias, 1.0, 1.0, 0, 1
    )
    assert gemm.rtype().shape() == [1, 3]
    gemm_container.new_retv(gemm)
    assert gemm_glob.verify_ir()
    assert "NN.gemm ATTR[transB=1,transA=0,beta=1,alpha=1]" in gemm_glob.dump()

    conv_glob, _, conv_container, conv_input = _new_scope(
        "typed_conv", [1, 2, 4, 4], [1, 3, 4, 4]
    )
    conv_weight = conv_container.new_ranked_f32_const([0.125] * 54, [3, 2, 3, 3])
    conv_bias = conv_container.new_ranked_f32_const([0.0] * 3, [3])
    conv = conv_container.new_nn_conv(
        conv_input,
        conv_weight,
        conv_bias,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        [3, 3],
        1,
    )
    assert conv.rtype().shape() == [1, 3, 4, 4]
    conv_container.new_retv(conv)
    assert conv_glob.verify_ir()
    assert (
        "NN.conv ATTR[strides=(1,1),pads=(1,1,1,1),"
        "kernel_shape=(3,3),group=1,dilations=(1,1)]"
        in conv_glob.dump()
    )


def test_three_argument_conv_binding_accepts_legacy_formal_operands():
    glob = air_builder.GlobScope()
    input_type = glob.new_array_type([1, 2, 4, 4], "f32")
    weight_type = glob.new_array_type([3, 2, 3, 3], "f32")
    bias_type = glob.new_array_type([3], "f32")
    scope = glob.new_func_with_param_types(
        "legacy_conv", input_type, [input_type, weight_type, bias_type]
    )
    container = scope.container()
    value = scope.new_param("input", input_type)
    weight = scope.new_param("weight", weight_type)
    bias = scope.new_param("bias", bias_type)
    result = container.new_nn_conv(value, weight, bias)

    assert result.rtype().shape() == [1, 2, 4, 4]
    container.new_retv(result)
    assert glob.verify_ir()
    dump = glob.dump()
    assert "NN.conv" in dump
    assert "NN.conv ATTR[" not in dump

    foreign_glob = air_builder.GlobScope()
    foreign_input_type = foreign_glob.new_array_type([1, 2, 4, 4], "f32")
    foreign_weight_type = foreign_glob.new_array_type([3, 2, 3, 3], "f32")
    foreign_bias_type = foreign_glob.new_array_type([3], "f32")
    foreign_scope = foreign_glob.new_func_with_param_types(
        "foreign_legacy_conv",
        foreign_input_type,
        [foreign_input_type, foreign_weight_type, foreign_bias_type],
    )
    foreign_value = foreign_scope.new_param("input", foreign_input_type)
    foreign_weight = foreign_scope.new_param("weight", foreign_weight_type)
    foreign_bias = foreign_scope.new_param("bias", foreign_bias_type)

    for operands in (
        (foreign_value, weight, bias),
        (value, foreign_weight, bias),
        (value, weight, foreign_bias),
    ):
        before = container.dump()
        with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
            container.new_nn_conv(*operands)
        assert container.dump() == before
    assert foreign_glob.verify_ir()


def test_raw_gemm_rejects_semantic_and_shape_errors_without_mutation():
    _, _, container, value = _new_scope("reject_gemm", [1, 4], [1, 3])
    weight = container.new_ranked_f32_const([0.125] * 12, [3, 4])
    bias = container.new_ranked_f32_const([0.0] * 3, [3])
    bad_weight = container.new_ranked_f32_const([0.125] * 15, [3, 5])
    bad_bias = container.new_ranked_f32_const([0.0] * 2, [2])

    invalid_calls = (
        (lambda: container.new_nn_gemm(value, weight, bias, 2.0, 1.0, 0, 1), "alpha=1"),
        (lambda: container.new_nn_gemm(value, weight, bias, 1.0, 0.0, 0, 1), "beta=1"),
        (lambda: container.new_nn_gemm(value, weight, bias, 1.0, 1.0, 1, 1), "transA=0"),
        (lambda: container.new_nn_gemm(value, weight, bias, 1.0, 1.0, 0, 0), "transB=1"),
        (lambda: container.new_nn_gemm(value, bad_weight, bias, 1.0, 1.0, 0, 1), "reduction"),
        (lambda: container.new_nn_gemm(value, weight, bad_bias, 1.0, 1.0, 0, 1), "bias length"),
    )
    for call, message in invalid_calls:
        before = container.dump()
        with pytest.raises(RuntimeError, match=message):
            call()
        assert container.dump() == before


def test_raw_conv_rejects_ignored_semantics_without_mutation():
    _, _, container, value = _new_scope(
        "reject_conv", [1, 2, 4, 4], [1, 3, 4, 4]
    )
    weight = container.new_ranked_f32_const([0.125] * 54, [3, 2, 3, 3])
    bias = container.new_ranked_f32_const([0.0] * 3, [3])
    calls = (
        (lambda: container.new_nn_conv(value, weight, bias, [2, 2], [1] * 4, [1, 1], [3, 3], 1), "unit authored strides"),
        (lambda: container.new_nn_conv(value, weight, bias, [1, 1], [1] * 4, [2, 2], [3, 3], 1), "unit dilation"),
        (lambda: container.new_nn_conv(value, weight, bias, [1, 1], [1, 0, 1, 0], [1, 1], [3, 3], 1), "equal spatial padding"),
        (lambda: container.new_nn_conv(value, weight, bias, [1, 1], [1] * 4, [1, 1], [1, 1], 1), "kernel_shape"),
        (lambda: container.new_nn_conv(value, weight, bias, [1, 1], [1] * 4, [1, 1], [3, 3], 2), "depthwise"),
    )
    for call, message in calls:
        before = container.dump()
        with pytest.raises(RuntimeError, match=message):
            call()
        assert container.dump() == before


def test_raw_conv_rejects_nonconstant_and_foreign_weights_without_mutation():
    glob = air_builder.GlobScope()
    input_type = glob.new_array_type([1, 2, 4, 4], "f32")
    weight_type = glob.new_array_type([3, 2, 3, 3], "f32")
    bias_type = glob.new_array_type([3], "f32")
    result_type = glob.new_array_type([1, 3, 4, 4], "f32")
    scope = glob.new_func_with_param_types(
        "conv_ownership", result_type, [input_type, weight_type, bias_type]
    )
    container = scope.container()
    value = scope.new_param("input", input_type)
    formal_weight = scope.new_param("weight", weight_type)
    formal_bias = scope.new_param("bias", bias_type)
    inline_weight = container.new_ranked_f32_const(
        [0.125] * 54, [3, 2, 3, 3]
    )
    inline_bias = container.new_ranked_f32_const([0.0] * 3, [3])
    attrs = ([1, 1], [1, 1, 1, 1], [1, 1], [3, 3], 1)

    before = container.dump()
    with pytest.raises(RuntimeError, match="weight.*inline ARRAY"):
        container.new_nn_conv(value, formal_weight, inline_bias, *attrs)
    assert container.dump() == before
    with pytest.raises(RuntimeError, match="bias.*inline ARRAY"):
        container.new_nn_conv(value, inline_weight, formal_bias, *attrs)
    assert container.dump() == before

    foreign_glob, _, foreign_container, _ = _new_scope(
        "foreign_conv", [1, 2, 4, 4], [1, 3, 4, 4]
    )
    foreign_weight = foreign_container.new_ranked_f32_const(
        [0.125] * 54, [3, 2, 3, 3]
    )
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_nn_conv(value, foreign_weight, inline_bias, *attrs)
    assert container.dump() == before
    assert foreign_glob.verify_ir()


def test_raw_builders_reject_nonconstants_cross_container_and_non_f32():
    first_glob = air_builder.GlobScope()
    f32_input = first_glob.new_array_type([1, 4], "f32")
    f32_weight = first_glob.new_array_type([3, 4], "f32")
    f32_bias = first_glob.new_array_type([3], "f32")
    f32_result = first_glob.new_array_type([1, 3], "f32")
    scope = first_glob.new_func_with_param_types(
        "ownership", f32_result, [f32_input, f32_weight, f32_bias]
    )
    container = scope.container()
    value = scope.new_param("input", f32_input)
    formal_weight = scope.new_param("weight", f32_weight)
    formal_bias = scope.new_param("bias", f32_bias)
    inline_weight = container.new_ranked_f32_const([0.125] * 12, [3, 4])
    inline_bias = container.new_ranked_f32_const([0.0] * 3, [3])

    before = container.dump()
    with pytest.raises(RuntimeError, match="weight.*inline ARRAY"):
        container.new_nn_gemm(value, formal_weight, inline_bias, 1.0, 1.0, 0, 1)
    assert container.dump() == before
    with pytest.raises(RuntimeError, match="bias.*inline ARRAY"):
        container.new_nn_gemm(value, inline_weight, formal_bias, 1.0, 1.0, 0, 1)
    assert container.dump() == before

    second_glob, _, second_container, _ = _new_scope(
        "foreign", [1, 4], [1, 3]
    )
    foreign_weight = second_container.new_ranked_f32_const([0.125] * 12, [3, 4])
    with pytest.raises(RuntimeError, match="cannot mix AIR containers"):
        container.new_nn_gemm(value, foreign_weight, inline_bias, 1.0, 1.0, 0, 1)
    assert container.dump() == before

    _, _, f64_container, f64_value = _new_scope(
        "wrong_element", [1, 4], [1, 3], element="f64"
    )
    f64_weight = f64_container.new_ranked_f32_const([0.125] * 12, [3, 4])
    f64_bias = f64_container.new_ranked_f32_const([0.0] * 3, [3])
    f64_before = f64_container.dump()
    with pytest.raises(RuntimeError, match="f32 ranked operands"):
        f64_container.new_nn_gemm(
            f64_value, f64_weight, f64_bias, 1.0, 1.0, 0, 1
        )
    assert f64_container.dump() == f64_before
    assert second_glob.verify_ir()


def _worker_main():
    from ace_edsl.edsl.vector.kernels.baseline_conv import baseline_conv_recipe
    from ace_edsl.edsl.vector.kernels.baseline_gemm import baseline_gemm_recipe
    from ace_edsl.edsl.vector.kernels.fast_conv import fast_conv_recipe
    from ace_edsl.edsl.vector.kernels.fast_gemm import fast_gemm_recipe
    from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget

    case_name = sys.argv[2]
    implementation = sys.argv[3]
    case = _CASES[case_name]
    AceEDSL._get_dsl.cache_clear()
    source_result = case["kernel"](case["input"])
    glob = AceEDSL._get_dsl().current_air_module
    initial_dump = glob.dump()
    source_summary = {
        "initial_verify": glob.verify_ir(),
        "initial_dump": initial_dump,
        "source_result_shape": list(source_result.shape),
        "source_result_element": source_result.air_type.element_type().to_string(),
        "source_nn_count": len(
            re.findall(rf"^\s+NN\.{case['operation']} ", initial_dump, re.MULTILINE)
        ),
        "source_ldc_count": len(
            re.findall(r"^\s+ldc ", initial_dump, re.MULTILINE)
        ),
        "source_constant_count": len(
            re.findall(r"^\s+CST\[[^]]+\] array", initial_dump, re.MULTILINE)
        ),
        "source_vector_count": len(
            re.findall(r"^\s+VECTOR\.", initial_dump, re.MULTILINE)
        ),
    }
    if implementation == "source":
        print(json.dumps(source_summary))
        return

    recipes = {
        "baseline-gemm": baseline_gemm_recipe,
        "fast-gemm": fast_gemm_recipe,
        "baseline-conv": baseline_conv_recipe,
        "fast-conv": fast_conv_recipe,
    }
    recipe_calls = []
    prepared_summary = {}

    def recipe(trace, prepared):
        recipe_calls.append(prepared.kind)
        prepared_summary.update(
            {
                "kind": prepared.kind,
                "provenance": prepared.provenance,
                "constant_roles": [item.role for item in prepared.constants],
                "constant_types": [
                    {
                        "role": item.role,
                        "element_type": item.type.element_type,
                        "shape": list(item.type.shape),
                        "content_hash": item.content_hash,
                        "bytes_hex": item.bytes.hex(),
                    }
                    for item in prepared.constants
                ],
                "owned_constants": isinstance(prepared.constants, tuple)
                and all(isinstance(item.bytes, bytes) for item in prepared.constants),
            }
        )
        return recipes[case_name](trace, prepared)

    pipeline = Pipeline(
        "direct-nn-authoring",
        output_dir="/tmp/ace-direct-nn-authoring",
        dump_ir=False,
        verbose=False,
    ).set_glob(glob)
    settings = _CASE_SETTINGS[case_name]
    fallback = None
    if implementation != "default":
        auto_selection = implementation == "native-auto"
        kernel_impl = "native" if auto_selection else implementation
        if implementation == "missing":
            kernel_impl = "dsl"
        plan_kind = "auto" if auto_selection else case_name
        fallback = "error"
        pipeline.configure_vector_kernel_lowering(
            plan_provider="cpp",
            kernel_impl=kernel_impl,
            plan_kind=plan_kind,
            fallback=fallback,
            mask_fuse=False if auto_selection else settings["mask_fuse"],
            max_slots=0 if auto_selection else settings["max_slots"],
            conv_parallel=False,
            sharding=False,
        )
        if kernel_impl == "dsl" and implementation != "missing":
            pipeline.register_vector_kernel_recipe(case_name, recipe)

    config = pipeline.vector_kernel_config
    selection = (
        None
        if config is None
        else {
            "plan_provider": config.plan_provider,
            "kernel_impl": config.kernel_impl,
            "plan_kind": config.plan_kind,
            "fallback": config.fallback,
            "mask_fuse": config.mask_fuse,
            "max_slots": config.max_slots,
            "conv_parallel": config.conv_parallel,
            "sharding": config.sharding,
        }
    )

    run = pipeline.run(target=PipelineTarget.TENSOR2VECTOR)
    dump = pipeline.glob.dump()
    helper_prefix = "__ace_vkernel_" + case_name.replace("-", "_") + "_"
    helper_names = sorted(
        set(re.findall(rf'"({re.escape(helper_prefix)}[^"]+)"', dump))
    )
    helper_calls = len(
        re.findall(rf'^\s+call "{re.escape(helper_prefix)}', dump, re.MULTILINE)
    )
    oracle = None
    if run.success:
        oracle = getattr(pipeline.glob, case["inspector"])()
    payload = {
        **source_summary,
        "success": run.success,
        "error": run.error,
        "verify": pipeline.glob.verify_ir(),
        "stages_completed": run.stages_completed,
        "fallback": fallback,
        "selection": selection,
        "recipe_calls": recipe_calls,
        "prepared": prepared_summary,
        "dump_unchanged": dump == initial_dump,
        "has_source_nn": bool(
            re.search(rf"\bNN\.{case['operation']}\b", dump, re.IGNORECASE)
        ),
        "helper_count": len(helper_names),
        "helper_calls": helper_calls,
        "oracle": oracle,
    }
    print(json.dumps(payload))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == _WORKER_FLAG:
    _worker_main()
