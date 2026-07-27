"""
Vector Domain Operations

Domain-specific operations for the vector domain.
These functions generate AIR operations via operator overloading.
"""

from collections.abc import Sequence

from .air_value import AIRValue


def _require_vector_value(value: AIRValue, name: str) -> None:
    if not isinstance(value, AIRValue):
        raise TypeError(f"{name} must be an AIRValue")
    if value.domain != "nn::vector":
        raise TypeError(f"{name} must be an nn::vector AIRValue")
    if value.air_type is None or not value.air_type.is_array():
        raise TypeError(f"{name} must carry a ranked Vector AIR type")


def _require_core_s32(value: AIRValue, name: str) -> None:
    if not isinstance(value, AIRValue):
        raise TypeError(f"{name} must be an AIRValue")
    if value.domain != "air::core":
        raise TypeError(f"{name} must be an air::core scalar")
    if (
        value.air_type is None
        or not value.air_type.is_scalar()
        or not value.air_type.is_integer()
        or value.air_type.bit_width() != 32
        or value.air_type.to_string() != "i32"
    ):
        raise TypeError(f"{name} must carry a signed Core i32 AIR type")


def vec_add(a: AIRValue, b: AIRValue) -> AIRValue:
    """Emit genuine native Vector addition."""
    _require_vector_value(a, "a")
    _require_vector_value(b, "b")
    if a.container is not b.container:
        raise ValueError("vec_add cannot mix AIR containers")
    container = a.container
    if not hasattr(container, "new_vec_add"):
        raise NotImplementedError("Container does not support native Vector add")
    return a._flatten_result(container.new_vec_add(a.value, b.value))


def vec_mul(a: AIRValue, b: AIRValue) -> AIRValue:
    """Emit genuine native Vector multiplication."""
    _require_vector_value(a, "a")
    _require_vector_value(b, "b")
    if a.container is not b.container:
        raise ValueError("vec_mul cannot mix AIR containers")
    container = a.container
    if not hasattr(container, "new_vec_mul"):
        raise NotImplementedError("Container does not support native Vector mul")
    return a._flatten_result(container.new_vec_mul(a.value, b.value))


def vec_roll(
    value: AIRValue,
    shift: AIRValue,
    candidates: Sequence[int],
) -> AIRValue:
    """Emit a dynamic Vector roll with an ordered signed ``RNUM`` set."""
    _require_vector_value(value, "value")
    _require_core_s32(shift, "shift")
    if value.container is not shift.container:
        raise ValueError("vec_roll cannot mix AIR containers")
    if isinstance(candidates, (str, bytes)) or not isinstance(
        candidates, Sequence
    ):
        raise TypeError("roll candidates must be a sequence of integers")
    if not candidates:
        raise ValueError("roll candidates must not be empty")
    checked = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise TypeError("roll candidates must be integers")
        if candidate < -(1 << 31) or candidate >= (1 << 31):
            raise ValueError("roll candidate is out of signed i32 range")
        checked.append(candidate)
    container = value.container
    if not hasattr(container, "new_vec_roll"):
        raise NotImplementedError("Container does not support native Vector roll")
    return value._flatten_result(
        container.new_vec_roll(value.value, shift.value, checked)
    )


def vec_slice(
    value: AIRValue,
    start: AIRValue,
    slice_size: int,
) -> AIRValue:
    """Emit a dynamic Vector slice with the native planned result type."""
    _require_vector_value(value, "value")
    _require_core_s32(start, "start")
    if value.container is not start.container:
        raise ValueError("vec_slice cannot mix AIR containers")
    if isinstance(slice_size, bool) or not isinstance(slice_size, int):
        raise TypeError("slice_size must be an integer")
    if slice_size <= 0 or slice_size >= (1 << 31):
        raise ValueError("slice_size must be a positive signed i32 value")
    container = value.container
    if not hasattr(container, "new_vec_slice"):
        raise NotImplementedError("Container does not support native Vector slice")
    return value._flatten_result(
        container.new_vec_slice(value.value, start.value, slice_size)
    )


def vec_set_slot(value: AIRValue, slot: int) -> AIRValue:
    """Attach a typed positive ``SLOT`` attribute to a Vector value."""
    _require_vector_value(value, "value")
    return value.with_slot(slot)


def vec_sub(a: AIRValue, b: AIRValue) -> AIRValue:
    """
    Vector subtraction for vector domain.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        AIRValue representing vector subtraction result
    """
    container = a.container
    
    # Use vector-specific sub operation
    if hasattr(container, 'new_vec_sub'):
        result_node = container.new_vec_sub(a.value, b.value)
    else:
        # Fallback to regular sub
        return a - b
    
    return AIRValue(result_node, container, a.shape)


def vec_dot(a: AIRValue, b: AIRValue) -> AIRValue:
    """
    Vector dot product for vector domain.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        AIRValue representing dot product result (scalar)
    """
    container = a.container
    
    if hasattr(container, 'new_vec_dot'):
        result_node = container.new_vec_dot(a.value, b.value)
    else:
        # Fallback: element-wise multiply then sum
        result = a * b
        # TODO: Add reduction sum
        return result
    
    return AIRValue(result_node, container)


def vec_conv(x: AIRValue, weight: AIRValue, bias: AIRValue, **kwargs) -> AIRValue:
    """
    Vectorized convolution for vector domain.
    
    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Bias tensor
        **kwargs: Additional arguments
        
    Returns:
        AIRValue representing convolution result
    """
    container = x.container
    
    if hasattr(container, 'new_vec_conv'):
        result_node = container.new_vec_conv(x.value, weight.value, bias.value)
    else:
        # Fallback: use element-wise operations
        temp = vec_mul(x, weight)
        result = vec_add(temp, bias)
        return result
    
    return AIRValue(result_node, container, x.shape)

