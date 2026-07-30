"""Kernel recipes authored directly in ``nn::vector`` AIR."""

from .fast_common import (
    blocking_rot,
    clear_valid_data,
    collective_reduce,
    reduce_add_intra,
    roll_cyclic,
)
from .fast_gemm import (
    configure_fast_gemm_dsl,
    fast_gemm_recipe,
    fast_gemm_vector_kernel,
)
from .fast_conv import (
    configure_fast_conv_dsl,
    fast_conv_recipe,
    fast_conv_sharded_vector_kernel,
    fast_conv_vector_kernel,
)


__all__ = [
    "blocking_rot",
    "clear_valid_data",
    "collective_reduce",
    "configure_fast_conv_dsl",
    "configure_fast_gemm_dsl",
    "fast_conv_recipe",
    "fast_conv_sharded_vector_kernel",
    "fast_conv_vector_kernel",
    "fast_gemm_recipe",
    "fast_gemm_vector_kernel",
    "reduce_add_intra",
    "roll_cyclic",
]
