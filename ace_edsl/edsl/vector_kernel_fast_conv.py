"""Compatibility exports for the Vector fast-Conv kernel recipe."""

from .kernels.vector.fast_conv import (
    configure_fast_conv_dsl,
    fast_conv_recipe,
    fast_conv_sharded_vector_kernel,
    fast_conv_vector_kernel,
)

__all__ = [
    "configure_fast_conv_dsl",
    "fast_conv_recipe",
    "fast_conv_sharded_vector_kernel",
    "fast_conv_vector_kernel",
]
