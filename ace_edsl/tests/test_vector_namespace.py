"""Canonical Vector namespace ownership and public exports."""

from importlib import import_module
from importlib.util import find_spec

import pytest


_RECIPE_CASES = (
    (
        "baseline_gemm",
        (
            "baseline_gemm_recipe",
            "baseline_gemm_vector_kernel",
            "configure_baseline_gemm_dsl",
        ),
    ),
    (
        "baseline_conv",
        (
            "baseline_conv_recipe",
            "baseline_conv_vector_kernel",
            "configure_baseline_conv_dsl",
        ),
    ),
    (
        "fast_gemm",
        (
            "fast_gemm_recipe",
            "fast_gemm_vector_kernel",
            "configure_fast_gemm_dsl",
        ),
    ),
    (
        "fast_conv",
        (
            "fast_conv_recipe",
            "fast_conv_sharded_vector_kernel",
            "fast_conv_vector_kernel",
            "configure_fast_conv_dsl",
        ),
    ),
)

_REMOVED_MODULES = (
    "ace_edsl.edsl.kernels",
    "ace_edsl.edsl.vector_kernel_baseline_conv",
    "ace_edsl.edsl.vector_kernel_baseline_gemm",
    "ace_edsl.edsl.vector_kernel_fast_conv",
    "ace_edsl.edsl.vector_kernel_fast_gemm",
    "ace_edsl.edsl.vector_kernel_lowering",
    "ace_edsl.edsl.vector_kernel_planning",
)


@pytest.mark.parametrize("module_name", _REMOVED_MODULES)
def test_legacy_vector_modules_are_removed(module_name):
    assert find_spec(module_name) is None


def test_planning_is_owned_by_canonical_vector_namespace():
    planning = import_module("ace_edsl.edsl.vector.planning")

    assert planning.plan_vector_kernel.__module__ == planning.__name__
    assert planning.PlanningRequest.__module__ == planning.__name__
    assert planning.ProviderResult.__module__ == planning.__name__


def test_lowering_is_owned_by_canonical_vector_namespace():
    lowering = import_module("ace_edsl.edsl.vector.lowering")

    assert lowering.vector_kernel_recipe.__module__ == lowering.__name__
    assert lowering.PreparedBaselineGemmPlan.__module__ == lowering.__name__
    assert lowering.PreparedFastConvPlan.__module__ == lowering.__name__


@pytest.mark.parametrize(("leaf", "exports"), _RECIPE_CASES)
def test_recipe_implementations_have_one_canonical_identity(leaf, exports):
    canonical = import_module(f"ace_edsl.edsl.vector.kernels.{leaf}")
    top_level = import_module("ace_edsl.edsl")

    for name in exports:
        assert getattr(top_level, name) is getattr(canonical, name)

    kernel_names = tuple(name for name in exports if "vector_kernel" in name)
    assert kernel_names
    for name in kernel_names:
        assert getattr(canonical, name).__module__ == canonical.__name__


def test_fast_common_helpers_are_owned_by_canonical_vector_namespace():
    common = import_module("ace_edsl.edsl.vector.kernels.fast_common")

    for name in (
        "_ranked_constant",
        "_require_core_i32",
        "blocking_rot",
        "collective_reduce",
    ):
        assert getattr(common, name).__module__ == common.__name__
