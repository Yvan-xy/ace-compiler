"""Compatibility exports for the Vector baseline-Conv kernel recipe."""

from .kernels.vector.baseline_conv import (
    baseline_conv_recipe,
    baseline_conv_vector_kernel,
    configure_baseline_conv_dsl,
)

__all__ = [
    "baseline_conv_recipe",
    "baseline_conv_vector_kernel",
    "configure_baseline_conv_dsl",
]
