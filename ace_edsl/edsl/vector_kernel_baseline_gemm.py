"""Compatibility exports for the Vector baseline-GEMM kernel recipe."""

from .kernels.vector.baseline_gemm import (
    baseline_gemm_recipe,
    baseline_gemm_vector_kernel,
    configure_baseline_gemm_dsl,
)

__all__ = [
    "baseline_gemm_recipe",
    "baseline_gemm_vector_kernel",
    "configure_baseline_gemm_dsl",
]
