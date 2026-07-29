"""Typed mutable arrays whose elements are destination-owned Vector values."""

from typing import Any

from ace_bindings import air_builder

from .air_value import AIRValue
from .vector_value import VectorValue


class VectorArray:
    """A rank-one mutable AIR array with a ranked Vector element type."""

    NON_MUTATING_RECEIVER_METHODS = frozenset({"load", "store", "address"})

    def __init__(
        self,
        container: Any,
        name: str,
        extent: int,
        element_type: Any,
    ):
        if container is None:
            raise TypeError("VectorArray requires an AIR container")
        if not isinstance(name, str) or not name:
            raise ValueError("VectorArray name must be a non-empty string")
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError("VectorArray extent must be an integer")
        if extent <= 0 or extent >= (1 << 31):
            raise ValueError(
                "VectorArray extent must be a positive signed i32 value"
            )
        if element_type is None or not hasattr(element_type, "is_array"):
            raise TypeError("VectorArray requires a concrete AIR element type")
        if not element_type.is_array():
            raise TypeError("VectorArray elements must be ranked vectors")
        if element_type.rank() < 1:
            raise TypeError("VectorArray elements must have positive rank")

        outer_type = air_builder.Type.make_array([extent], element_type)
        if not outer_type.same_scope(element_type):
            raise ValueError("VectorArray types must share one GLOB_SCOPE")
        container.new_local(name, outer_type)

        self._container = container
        self._name = name
        self._extent = extent
        self._element_type = element_type
        self._outer_type = outer_type

    @property
    def container(self):
        return self._container

    @property
    def name(self) -> str:
        return self._name

    @property
    def extent(self) -> int:
        return self._extent

    @property
    def element_type(self):
        return self._element_type

    @property
    def air_type(self):
        return self._outer_type

    def _index_node(self, index: Any):
        if isinstance(index, bool):
            raise TypeError("VectorArray index must be a signed Core i32")
        if isinstance(index, int):
            return self._container.new_index_const(index)
        if not isinstance(index, AIRValue):
            raise TypeError("VectorArray index must be a signed Core i32")
        if index.container is not self._container:
            raise ValueError("VectorArray index belongs to another container")
        if index.domain != "air::core":
            raise TypeError("VectorArray index must be a Core scalar")
        index_type = index.air_type
        if (
            index_type is None
            or not index_type.is_integer()
            or not index_type.is_scalar()
            or index_type.bit_width() != 32
            or index_type.to_string() != "i32"
        ):
            raise TypeError("VectorArray index must be a signed Core i32")
        return index.value

    def address(self):
        """Return a fresh FLAT32 LDA for the outer mutable array."""
        return self._container.new_lda(
            self._container.new_ldid(self._name)
        )

    def load(self, index: Any) -> VectorValue:
        """Dynamically load one ranked Vector element."""
        result = self._container.new_ild(
            self._container.new_ldid(self._name), self._index_node(index)
        )
        result_type = result.rtype()
        if result_type != self._element_type:
            raise RuntimeError("VectorArray load produced the wrong element type")
        return VectorValue(
            result,
            self._container,
            shape=tuple(self._element_type.shape()),
            air_type=self._element_type,
        )

    def store(self, index: Any, value: VectorValue) -> None:
        """Dynamically store one exactly matching ranked Vector element."""
        if not isinstance(value, VectorValue):
            raise TypeError("VectorArray stores require a VectorValue")
        if value.container is not self._container:
            raise ValueError("VectorArray value belongs to another container")
        if value.air_type is None or value.air_type != self._element_type:
            raise TypeError(
                "VectorArray value type does not match its element type"
            )
        self._container.new_ist(
            value.value,
            self._container.new_ldid(self._name),
            self._index_node(index),
        )


__all__ = ["VectorArray"]
