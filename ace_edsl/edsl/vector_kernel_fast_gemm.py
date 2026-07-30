"""Compatibility exports for the Vector fast-GEMM kernel recipe."""

from .kernels.vector.fast_gemm import (
    configure_fast_gemm_dsl,
    fast_gemm_recipe,
    fast_gemm_vector_kernel,
)

__all__ = [
    "configure_fast_gemm_dsl",
    "fast_gemm_recipe",
    "fast_gemm_vector_kernel",
]
