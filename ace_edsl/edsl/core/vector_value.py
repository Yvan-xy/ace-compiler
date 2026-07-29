"""Typed value facade for authoring ``nn::vector`` AIR."""

from typing import Any, Optional, Sequence, Tuple

from .air_value import AIRValue


class VectorValue(AIRValue):
    """An ``AIRValue`` specialized with genuine Vector-domain operations.

    The wrapper retains the node, container, type, and temporary-load ownership
    model from ``AIRValue``. It only adds Vector authoring syntax; validation
    and AIR emission remain owned by ``vector_ops``.
    """

    NON_MUTATING_RECEIVER_METHODS = frozenset({"roll", "slice"})

    def __init__(
        self,
        node: Any,
        container: Any,
        shape: Optional[Tuple[int, ...]] = None,
        domain: Optional[str] = "nn::vector",
        temp_name: Optional[str] = None,
        air_type: Any = None,
        vector_slot: Optional[int] = None,
    ):
        if domain not in (None, "nn::vector"):
            raise ValueError("VectorValue domain must be nn::vector")
        super().__init__(
            node,
            container,
            shape=shape,
            domain="nn::vector",
            temp_name=temp_name,
            air_type=air_type,
            vector_slot=vector_slot,
        )
        if self.air_type is not None and not self.air_type.is_array():
            raise TypeError("VectorValue requires a ranked Vector AIR type")

    def __add__(self, other: Any) -> "VectorValue":
        """Emit genuine Vector addition."""
        self._set_loc()
        from .vector_ops import vec_add

        return vec_add(self, other)

    def __radd__(self, other: Any) -> "VectorValue":
        return self.__add__(other)

    def __mul__(self, other: Any) -> "VectorValue":
        """Emit genuine Vector multiplication."""
        self._set_loc()
        from .vector_ops import vec_mul

        return vec_mul(self, other)

    def __rmul__(self, other: Any) -> "VectorValue":
        return self.__mul__(other)

    def __sub__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector subtraction has no genuine nn::vector AIR operation"
        )

    def __rsub__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector subtraction has no genuine nn::vector AIR operation"
        )

    def __neg__(self) -> "VectorValue":
        raise NotImplementedError(
            "Vector negation has no genuine nn::vector AIR operation"
        )

    def __truediv__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector division has no genuine nn::vector AIR operation"
        )

    def __rtruediv__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector division has no genuine nn::vector AIR operation"
        )

    def __floordiv__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector division has no genuine nn::vector AIR operation"
        )

    def __rfloordiv__(self, other: Any) -> "VectorValue":
        raise NotImplementedError(
            "Vector division has no genuine nn::vector AIR operation"
        )

    def roll(
        self,
        shift: AIRValue,
        *,
        candidates: Sequence[int],
    ) -> "VectorValue":
        """Emit a dynamic Vector roll with explicit planned candidates."""
        self._set_loc()
        from .vector_ops import vec_roll

        return vec_roll(self, shift, candidates)

    def slice(self, start: AIRValue, slice_size: int) -> "VectorValue":
        """Emit a dynamic Vector slice with the planned trailing width."""
        self._set_loc()
        from .vector_ops import vec_slice

        return vec_slice(self, start, slice_size)


__all__ = ["VectorValue"]
