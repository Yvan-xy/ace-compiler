"""
Tensor Domain Operations

Domain-specific operations for the tensor domain.
These functions generate AIR operations via operator overloading.
"""

import math
import sys
from numbers import Integral, Real
from typing import Any, List, Optional, Tuple

from .air_value import AIRValue, _wrap_air_value


_I32_MIN = -(1 << 31)
_I32_MAX = (1 << 31) - 1


def _normalize_shape(shape: Any) -> Tuple[int, ...]:
    if isinstance(shape, bool):
        raise TypeError("constant shape dimensions must be integers")
    if isinstance(shape, Integral):
        dimensions = (int(shape),)
    elif isinstance(shape, (list, tuple)):
        dimensions = tuple(shape)
    else:
        raise TypeError("constant shape must be an integer, list, or tuple")
    if not dimensions:
        raise ValueError("constant shape must have at least one dimension")

    normalized = []
    element_count = 1
    for dimension in dimensions:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral):
            raise TypeError("constant shape dimensions must be integers")
        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError("constant shape dimensions must be positive")
        if dimension > sys.maxsize // element_count:
            raise OverflowError("constant shape element count overflows")
        normalized.append(dimension)
        element_count *= dimension
    return tuple(normalized)


def _flatten_ranked_values(
    values: Any,
) -> Tuple[Tuple[int, ...], List[float]]:
    if not isinstance(values, (list, tuple)) and hasattr(values, "tolist"):
        values = values.tolist()

    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("constant dimensions must be non-empty")
        child_shape = None
        flattened = []
        for item in values:
            item_shape, item_values = _flatten_ranked_values(item)
            if child_shape is None:
                child_shape = item_shape
            elif item_shape != child_shape:
                raise ValueError("constant values must form a rectangular tensor")
            flattened.extend(item_values)
        return (len(values),) + child_shape, flattened

    if isinstance(values, bool) or not isinstance(values, Real):
        raise TypeError("constant values must be real numbers, not bool")
    return (), [float(values)]


def _normalize_f32_dtype(dtype: Any) -> None:
    if dtype is float:
        return
    if isinstance(dtype, str) and dtype.lower() in {"f32", "float32"}:
        return
    if getattr(dtype, "__name__", None) == "float32":
        return
    raise TypeError("direct NN constants support only f32 values")


def constant(values: Any, *, shape: Any = None, dtype: Any = "f32") -> AIRValue:
    """Create a destination-owned ranked f32 ``CORE.LDC`` array constant."""
    _normalize_f32_dtype(dtype)
    inferred_shape, flattened = _flatten_ranked_values(values)
    if not inferred_shape:
        raise ValueError("constant values must have at least one rank")

    if shape is None:
        normalized_shape = inferred_shape
    else:
        normalized_shape = _normalize_shape(shape)
        expected_count = math.prod(normalized_shape)
        if len(flattened) != expected_count:
            raise ValueError("constant value count does not match explicit shape")
        if inferred_shape != normalized_shape and len(inferred_shape) != 1:
            raise ValueError("nested constant shape does not match explicit shape")

    from ..domain_ast_decorators import get_current_container, get_current_domain

    if get_current_domain() != "nn::core":
        raise RuntimeError(
            "constant() requires an active nn::core kernel trace"
        )
    container = get_current_container()
    if container is None:
        raise RuntimeError("constant() must be called while tracing a kernel")
    if not hasattr(container, "new_ranked_f32_const"):
        raise NotImplementedError(
            "Container does not support ranked f32 constants"
        )
    node = container.new_ranked_f32_const(flattened, normalized_shape)
    air_type = node.rtype()
    result = _wrap_air_value(
        node,
        container,
        shape=normalized_shape,
        domain="nn::core",
        air_type=air_type,
    )
    return result


def _ranked_f32_shape(
    value: AIRValue, name: str, expected_rank: int
) -> Tuple[int, ...]:
    if not isinstance(value, AIRValue):
        raise TypeError(f"{name} must be an AIRValue")
    air_type = value.air_type
    if air_type is None or not air_type.is_array():
        raise TypeError(f"{name} must have a ranked f32 AIR type")
    shape = tuple(air_type.shape())
    element_type = air_type.element_type()
    if (
        len(shape) != expected_rank
        or not element_type.is_float()
        or element_type.bit_width() != 32
    ):
        raise TypeError(f"{name} must be a rank-{expected_rank} f32 tensor")
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{name} must have positive dimensions")
    return shape


def _require_nn_domain(*operands: AIRValue) -> None:
    from ..domain_ast_decorators import get_current_domain

    if get_current_domain() != "nn::core":
        raise RuntimeError("NN operations require an active nn::core kernel trace")
    if any(operand.domain != "nn::core" for operand in operands):
        raise ValueError("NN operations require nn::core operands")


def _require_same_container(reference: AIRValue, *operands: AIRValue) -> None:
    if reference.container is None:
        raise RuntimeError("NN authoring requires a real AIR container")
    if any(operand.container is not reference.container for operand in operands):
        raise ValueError("NN operands cannot mix AIR containers")


def _require_inline_constant(value: AIRValue, name: str) -> None:
    node = getattr(value, "_node", None)
    if (
        node is None
        or not hasattr(node, "is_inline_array_constant")
        or not node.is_inline_array_constant()
    ):
        raise TypeError(f"{name} must be an inline CORE.LDC ARRAY constant")


def _normalize_s32(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < _I32_MIN or value > _I32_MAX:
        raise ValueError(f"{name} is out of signed i32 range")
    return value


def _normalize_spatial_pair(value: Any, name: str) -> Tuple[int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        item = _normalize_s32(value, name)
        return item, item
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must be an integer or a length-two sequence")
    return tuple(_normalize_s32(item, name) for item in value)


def _normalize_pads(value: Any) -> Tuple[int, int, int, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        item = _normalize_s32(value, "pads")
        return item, item, item, item
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 4):
        raise TypeError("pads must be an integer or a length-two/four sequence")
    normalized = tuple(_normalize_s32(item, "pads") for item in value)
    if len(normalized) == 2:
        return normalized + normalized
    return normalized


def _wrap_nn_result(node: Any, container: Any) -> AIRValue:
    air_type = node.rtype()
    shape = tuple(air_type.shape())
    return _wrap_air_value(
        node,
        container,
        shape=shape,
        domain="nn::core",
        air_type=air_type,
    )


def conv(
    x: AIRValue,
    weight: AIRValue,
    bias: AIRValue,
    *,
    strides: Any = (1, 1),
    pads: Any = (0, 0, 0, 0),
    dilations: Any = (1, 1),
    kernel_shape: Any = None,
    group: Any = 1,
    **kwargs: Any,
) -> AIRValue:
    """Author a typed NCHW ``NN.CONV`` in the supported Vector subset."""
    if "stride" in kwargs:
        if strides != (1, 1):
            raise TypeError("conv received both stride and strides")
        strides = kwargs.pop("stride")
    if "padding" in kwargs:
        if pads != (0, 0, 0, 0):
            raise TypeError("conv received both padding and pads")
        pads = kwargs.pop("padding")
    if "dilation" in kwargs:
        if dilations != (1, 1):
            raise TypeError("conv received both dilation and dilations")
        dilations = kwargs.pop("dilation")
    if "kernel_size" in kwargs:
        if kernel_shape is not None:
            raise TypeError("conv received both kernel_size and kernel_shape")
        kernel_shape = kwargs.pop("kernel_size")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"conv received unsupported keyword arguments: {unknown}")

    _require_nn_domain(x, weight, bias)
    x_shape = _ranked_f32_shape(x, "input", 4)
    weight_shape = _ranked_f32_shape(weight, "weight", 4)
    bias_shape = _ranked_f32_shape(bias, "bias", 1)
    _require_same_container(x, weight, bias)
    _require_inline_constant(weight, "weight")
    _require_inline_constant(bias, "bias")

    normalized_strides = _normalize_spatial_pair(strides, "strides")
    normalized_pads = _normalize_pads(pads)
    normalized_dilations = _normalize_spatial_pair(dilations, "dilations")
    normalized_kernel = (
        tuple(weight_shape[2:])
        if kernel_shape is None
        else _normalize_spatial_pair(kernel_shape, "kernel_shape")
    )
    normalized_group = _normalize_s32(group, "group")

    if x_shape[0] != 1:
        raise ValueError("conv supports only batch-one NCHW input")
    if normalized_kernel != tuple(weight_shape[2:]):
        raise ValueError("kernel_shape must match the weight shape")
    if normalized_kernel[0] != normalized_kernel[1]:
        raise ValueError("conv supports only square spatial kernels")
    if normalized_dilations != (1, 1):
        raise ValueError("conv supports only unit dilation")
    if normalized_strides != (1, 1):
        raise ValueError("conv supports only unit authored strides")
    if len(set(normalized_pads)) != 1:
        raise ValueError("conv supports only equal spatial padding")
    if normalized_group <= 0:
        raise ValueError("group must be positive")
    if normalized_group == 1:
        if weight_shape[1] != x_shape[1]:
            raise ValueError("weight/input channels do not match")
    elif (
        normalized_group != x_shape[1]
        or weight_shape[1] != 1
        or weight_shape[0] != x_shape[1]
    ):
        raise ValueError("grouped input is supported only as depthwise Conv")
    if bias_shape[0] != weight_shape[0]:
        raise ValueError("bias length must match output channels")

    inferred_spatial = []
    for axis in range(2):
        effective_kernel = (
            (normalized_kernel[axis] - 1) * normalized_dilations[axis] + 1
        )
        numerator = (
            x_shape[axis + 2]
            + normalized_pads[axis]
            + normalized_pads[axis + 2]
            - effective_kernel
        )
        if numerator < 0:
            raise ValueError("conv kernel exceeds the padded input")
        inferred_spatial.append(numerator // normalized_strides[axis] + 1)
    if tuple(inferred_spatial) != tuple(x_shape[2:]):
        raise ValueError("conv requires output-preserving spatial attributes")

    container = x.container
    if not hasattr(container, "new_nn_conv"):
        raise NotImplementedError("Container does not support typed NN.CONV")
    result_node = container.new_nn_conv(
        x.value,
        weight.value,
        bias.value,
        normalized_strides,
        normalized_pads,
        normalized_dilations,
        normalized_kernel,
        normalized_group,
    )
    return _wrap_nn_result(result_node, container)


def relu(x: AIRValue) -> AIRValue:
    """
    ReLU activation function for tensor domain.
    
    Args:
        x: Input tensor
        
    Returns:
        AIRValue representing ReLU result
    """
    container = x.container
    
    # Generate nn::core::RELU operation
    if hasattr(container, 'new_nn_relu'):
        result_node = container.new_nn_relu(x.value)
    elif hasattr(container, 'new_relu'):
        result_node = container.new_relu(x.value)
    else:
        # Fallback: max(0, x)
        zero = AIRValue(container.new_intconst(0), container)
        result_node = (x > zero).value  # This returns bool, need proper max
        # For now, just return input
        result_node = x.value
    
    return AIRValue(result_node, container, x.shape)


def softmax(x: AIRValue, dim: Optional[int] = None) -> AIRValue:
    """
    Softmax activation function for tensor domain.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax (None = last dimension)
        
    Returns:
        AIRValue representing softmax result
    """
    container = x.container
    
    # Generate nn::core::SOFTMAX operation
    if hasattr(container, 'new_nn_softmax'):
        result_node = container.new_nn_softmax(x.value)
    elif hasattr(container, 'new_softmax'):
        result_node = container.new_softmax(x.value)
    else:
        # Fallback: exp(x) / sum(exp(x))
        # For now, just return input
        result_node = x.value
    
    return AIRValue(result_node, container, x.shape)


def matmul(a: AIRValue, b: AIRValue) -> AIRValue:
    """
    Matrix multiplication for tensor domain.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        AIRValue representing matrix multiplication result
    """
    # Use AIRValue's __matmul__ operator
    return a @ b


def gemm(
    a: AIRValue,
    b: AIRValue,
    c: Optional[AIRValue] = None,
    alpha: Any = 1.0,
    beta: Any = 1.0,
    transA: Any = 0,
    transB: Any = 1,
) -> AIRValue:
    """Author a typed ``NN.GEMM`` with a transposed ranked weight constant."""
    if c is None:
        raise ValueError("gemm requires an inline ranked bias constant")
    _require_nn_domain(a, b, c)
    a_shape = _ranked_f32_shape(a, "input", 2)
    b_shape = _ranked_f32_shape(b, "weight", 2)
    c_shape = _ranked_f32_shape(c, "bias", 1)
    _require_same_container(a, b, c)
    _require_inline_constant(b, "weight")
    _require_inline_constant(c, "bias")

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be a real number")
    normalized_alpha = float(alpha)
    normalized_beta = float(beta)
    if not math.isfinite(normalized_alpha) or not math.isfinite(normalized_beta):
        raise ValueError("alpha and beta must be finite")
    normalized_trans_a = _normalize_s32(transA, "transA")
    normalized_trans_b = _normalize_s32(transB, "transB")
    if normalized_trans_a not in (0, 1) or normalized_trans_b not in (0, 1):
        raise ValueError("transA and transB must be zero or one")
    if (
        normalized_alpha != 1.0
        or normalized_beta != 1.0
        or normalized_trans_a != 0
        or normalized_trans_b != 1
    ):
        raise ValueError(
            "gemm supports alpha=1, beta=1, transA=0, and transB=1"
        )

    if a_shape[0] != 1:
        raise ValueError("gemm supports only a single input row")
    if a_shape[1] != b_shape[1]:
        raise ValueError("gemm reduction dimensions do not match")
    if c_shape[0] != b_shape[0]:
        raise ValueError("gemm bias length must match output columns")

    container = a.container
    if not hasattr(container, "new_nn_gemm"):
        raise NotImplementedError("Container does not support typed NN.GEMM")
    result_node = container.new_nn_gemm(
        a.value,
        b.value,
        c.value,
        normalized_alpha,
        normalized_beta,
        normalized_trans_a,
        normalized_trans_b,
    )
    return _wrap_nn_result(result_node, container)


def average_pool(x: AIRValue, kernel_size: tuple = (2, 2), 
                 stride: tuple = None, padding: tuple = (0, 0)) -> AIRValue:
    """
    Average pooling for tensor domain.
    
    Args:
        x: Input tensor
        kernel_size: Pooling kernel size
        stride: Stride (defaults to kernel_size)
        padding: Padding
        
    Returns:
        AIRValue representing pooling result
    """
    container = x.container
    
    if hasattr(container, 'new_nn_average_pool'):
        result_node = container.new_nn_average_pool(x.value)
    elif hasattr(container, 'new_average_pool'):
        result_node = container.new_average_pool(x.value)
    else:
        # Fallback: return input
        result_node = x.value
    
    return AIRValue(result_node, container)


def max_pool(x: AIRValue, kernel_size: tuple = (2, 2),
            stride: tuple = None, padding: tuple = (0, 0)) -> AIRValue:
    """
    Max pooling for tensor domain.
    
    Args:
        x: Input tensor
        kernel_size: Pooling kernel size
        stride: Stride (defaults to kernel_size)
        padding: Padding
        
    Returns:
        AIRValue representing pooling result
    """
    container = x.container
    
    if hasattr(container, 'new_nn_max_pool'):
        result_node = container.new_nn_max_pool(x.value)
    elif hasattr(container, 'new_max_pool'):
        result_node = container.new_max_pool(x.value)
    else:
        # Fallback: return input
        result_node = x.value
    
    return AIRValue(result_node, container)


def flatten(x: AIRValue, start_dim: int = 0, end_dim: int = -1) -> AIRValue:
    """
    Flatten tensor for tensor domain.
    
    Args:
        x: Input tensor
        start_dim: Start dimension
        end_dim: End dimension (-1 = last)
        
    Returns:
        AIRValue representing flattened tensor
    """
    container = x.container
    
    if hasattr(container, 'new_nn_flatten'):
        result_node = container.new_nn_flatten(x.value)
    elif hasattr(container, 'new_flatten'):
        result_node = container.new_flatten(x.value)
    else:
        # Fallback: return input
        result_node = x.value
    
    return AIRValue(result_node, container)


def reshape(x: AIRValue, shape: tuple) -> AIRValue:
    """
    Reshape tensor for tensor domain.
    
    Args:
        x: Input tensor
        shape: Target shape
        
    Returns:
        AIRValue representing reshaped tensor
    """
    container = x.container
    
    if hasattr(container, 'new_nn_reshape'):
        result_node = container.new_nn_reshape(x.value)
    elif hasattr(container, 'new_reshape'):
        result_node = container.new_reshape(x.value)
    else:
        # Fallback: return input
        result_node = x.value
    
    return AIRValue(result_node, container, shape)

