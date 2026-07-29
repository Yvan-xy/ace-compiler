"""
AIRValue - Wrapper class for operator overloading.

This class wraps AIR nodes to enable Python operators (+, *, [], etc.)
to emit AIR nodes directly during function execution.

Source location tracking is automatic - each operation captures the
Python source line where it was called and attaches it to the AIR node.

Example:
    @tensor_kernel
    def add(a: Tensor[64], b: Tensor[64]) -> Tensor[64]:
        return a + b  # AIRValue.__add__ emits tensor::add AIR node with line info
"""

import os
from typing import Any, Optional, Sequence, Tuple, Union

# Import AIR bindings
from ace_bindings import air_builder, nn_addon

# Import source location tracking
try:
    from ace_edsl.base_dsl.loc import set_current_loc
except ImportError:
    # Fallback: no-op if import fails
    def set_current_loc(container, skip=0):
        pass


class AIRValue:
    """
    Python wrapper around an AIR node that overloads operators.
    
    When you write `a + b` in a @kernel function, the + operator
    calls __add__, which directly creates an AIR node via container.new_add().
    
    Flat IR Mode:
        When FLAT_IR_MODE is True, each operation result is stored to a temporary
        variable and loaded back. This creates flat SSA-style IR instead of nested
        tree structures, avoiding exponential blowup in IR dumps.
    
    Attributes:
        _node: The underlying AIR node (from C++ bindings)
        _container: The container holding this node (for creating new operations)
        _shape: Optional shape information for tensor operations
    """
    
    # Class-level settings
    FLAT_IR_MODE = True  # Enable flat IR generation to avoid exponential dump size
    SOURCE_LOC_ENABLED = True
    FRESH_LOAD_ENABLED = False
    _temp_counter = 0    # Counter for generating unique temp names
    
    @classmethod
    def reset_temp_counter(cls):
        """Reset the temp counter (call at start of each kernel)."""
        cls._temp_counter = 0
    
    @classmethod
    def _next_temp_name(cls) -> str:
        """Generate a unique temporary variable name."""
        name = f"_t{cls._temp_counter}"
        cls._temp_counter += 1
        return name

    @staticmethod
    def _bootstrap_stage_primitive_enabled() -> bool:
        """Enable experimental stage-op lowering to explicit CKKS primitives."""
        raw = os.environ.get("ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING")
        if raw is None:
            # Default to primitive lowering; set env to 0/false/off to disable.
            return True
        return raw.strip().lower() in ("1", "true", "yes", "on")
    
    def __init__(
        self,
        node: Any,
        container: Any,
        shape: Optional[Tuple[int, ...]] = None,
        domain: Optional[str] = None,
        temp_name: Optional[str] = None,
        air_type: Any = None,
        vector_slot: Optional[int] = None,
    ):
        self._node = node
        self._container = container
        self._shape = shape
        self._domain = domain
        self._temp_name = temp_name  # For on-demand fresh loads
        self._air_type = air_type or self._infer_air_type(node)
        self._vector_slot = vector_slot

    @staticmethod
    def _infer_air_type(node: Any) -> Any:
        if node is not None and hasattr(node, "rtype"):
            try:
                inferred = node.rtype()
                if inferred is not None and inferred.to_string() != "void":
                    return inferred
            except (AttributeError, RuntimeError):
                pass
        return None

    def _is_core_scalar(self) -> bool:
        if self._air_type is None:
            return self._domain == "air::core"
        try:
            return bool(self._air_type.is_scalar())
        except (AttributeError, RuntimeError):
            return self._domain == "air::core"

    def view(self, target_domain: str) -> 'AIRValue':
        """Return a non-mutating lexical view in ``target_domain``.

        Primitive values remain Core scalars even while a Vector kernel is
        active; ranked operands adopt the target kernel domain.
        """
        domain = "air::core" if self._is_core_scalar() else target_domain
        return _wrap_air_value(
            self._node,
            self._container,
            self._shape,
            domain,
            self._temp_name,
            self._air_type,
            self._vector_slot,
        )

    def cast(self, air_type: Any) -> 'AIRValue':
        """Perform an explicit checked cast.

        AIR Core currently has no integer width-conversion opcode, so only an
        already structurally compatible type is a legal no-op cast.
        """
        if not hasattr(self._container, "new_checked_cast"):
            raise NotImplementedError("Container does not support checked casts")
        cast_node = self._container.new_checked_cast(self.value, air_type)
        return _wrap_air_value(
            cast_node,
            self._container,
            self._shape,
            "air::core" if air_type.is_scalar() else self._domain,
            air_type=air_type,
        )

    def zero_like(self) -> 'AIRValue':
        """Create a real zero carrying this value's exact AIR type."""
        if self._air_type is None:
            raise TypeError("zero_like requires AIR type metadata")
        zero = self._container.new_zero(self._air_type)
        return _wrap_air_value(
            zero,
            self._container,
            self._shape,
            self._domain,
            air_type=self._air_type,
        )

    def _apply_vector_slot(self, node: Any) -> Any:
        """Attach a pending typed Vector SLOT to ``node`` when present."""
        if self._vector_slot is not None:
            if node is None or not hasattr(node, "set_vector_slot"):
                raise NotImplementedError(
                    "AIR node does not support typed Vector SLOT attributes"
                )
            node.set_vector_slot(self._vector_slot)
        return node

    def with_slot(self, slot: int) -> 'AIRValue':
        """Return this Vector value with a positive typed ``SLOT`` attribute."""
        if self._domain != "nn::vector":
            raise TypeError("with_slot requires an nn::vector AIRValue")
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("Vector SLOT must be an integer")
        if slot <= 0 or slot >= (1 << 32):
            raise ValueError("Vector SLOT must be a positive u32 value")
        return _wrap_air_value(
            self._node,
            self._container,
            self._shape,
            self._domain,
            temp_name=self._temp_name,
            air_type=self._air_type,
            vector_slot=slot,
        )

    def _value_without_vector_slot(self) -> Any:
        """Materialize a value without applying its pending Vector SLOT."""
        if self._temp_name is not None and hasattr(self._container, "new_ldid"):
            return self._container.new_ldid(self._temp_name)
        if (AIRValue.FRESH_LOAD_ENABLED and self._node is not None and
                hasattr(self._container, "new_fresh_load")):
            return self._container.new_fresh_load(self._node)
        return self._node
    
    def _flatten_result(self, result_node: Any) -> 'AIRValue':
        """
        Flatten the result by storing to a temp and loading back.
        
        This creates flat SSA-style IR:
            %t0 = ADD(%a, %b)  # Store result
            %t1 = LD(t0)       # Load for next use
        
        Instead of nested tree IR that causes exponential dump blowup.
        
        Note: We store the temp_name, not the load_node. Each access to .value
        creates a fresh load to avoid sharing the same node across multiple uses.
        """
        result_type = self._infer_air_type(result_node)
        result_shape = self._shape
        if result_type is not None and result_type.is_array():
            result_shape = tuple(result_type.shape())
        elif result_type is not None and result_type.is_scalar():
            result_shape = None
        result_domain = (
            "air::core"
            if result_type is not None and result_type.is_scalar()
            else self._domain
        )

        if not AIRValue.FLAT_IR_MODE:
            return _wrap_air_value(
                result_node, self._container, result_shape, result_domain,
                air_type=result_type,
            )
        
        # Store result to a temporary
        temp_name = AIRValue._next_temp_name()
        if hasattr(self._container, 'new_stid'):
            if (result_type is not None and
                    hasattr(self._container, "new_local")):
                self._container.new_local(temp_name, result_type)
            store_node = self._container.new_stid(temp_name, result_node)
            # Return AIRValue with temp_name - loads created on-demand in .value
            if hasattr(self._container, 'new_ldid'):
                return _wrap_air_value(
                    node=None,  # No cached node - will load on demand
                    container=self._container, 
                    shape=result_shape,
                    domain=result_domain,
                    temp_name=temp_name,  # Store the name for fresh loads
                    air_type=result_type,
                )
            else:
                # Fallback: use the store node directly
                return _wrap_air_value(
                    store_node,
                    self._container,
                    result_shape,
                    result_domain,
                    air_type=result_type,
                )
        else:
            # Fallback: no flattening if stid not available
            return _wrap_air_value(
                result_node,
                self._container,
                result_shape,
                result_domain,
                air_type=result_type,
            )
    
    @property
    def value(self) -> Any:
        """Return the underlying AIR node.
        
        This provides API consistency with Numeric.value pattern,
        where .value gives access to the underlying IR representation.
        
        If temp_name is set (flat IR mode), creates a fresh load each time
        to ensure each use gets a distinct node. This avoids issues where
        the same node is used in multiple places.
        """
        return self._apply_vector_slot(self._value_without_vector_slot())
    
    @property
    def container(self) -> Any:
        """Return the container holding this node."""
        return self._container
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return the shape of this value (if a tensor)."""
        return self._shape

    @property
    def domain(self) -> Optional[str]:
        """Return the current domain for this value (if set)."""
        return self._domain

    @property
    def air_type(self) -> Any:
        """Return the exact AIR result type carried by this value."""
        return self._air_type
    
    def _get_other_node(self, other: Any) -> Any:
        """Helper to extract node from other (AIRValue, scalar, or node).
        
        For CKKS domain, scalar values are encoded into plaintext polynomials
        so they can be used in FHE operations with ciphertexts.
        """
        if isinstance(other, AIRValue):
            return other.value
        elif isinstance(other, (int, float)):
            # Convert scalar to constant node
            if isinstance(other, int):
                if (self._is_core_scalar() and self._air_type is not None and
                        hasattr(self._container, "new_intconst_typed") and
                        self._air_type.is_integer()):
                    const_node = self._container.new_intconst_typed(
                        other, self._air_type
                    )
                else:
                    const_node = self._container.new_intconst(other)
            else:
                # For float, create float constant if available, else use int
                if (self._is_core_scalar() and self._air_type is not None and
                        hasattr(self._container, "new_floatconst_typed") and
                        self._air_type.is_float()):
                    const_node = self._container.new_floatconst_typed(
                        other, self._air_type
                    )
                elif hasattr(self._container, 'new_floatconst'):
                    const_node = self._container.new_floatconst(other)
                else:
                    const_node = self._container.new_intconst(int(other))
            
            # For CKKS domain, encode scalar into plaintext polynomial
            if self._domain == "fhe::ckks" and hasattr(self._container, 'new_ckks_encode'):
                return self._container.new_ckks_encode(const_node)
            return const_node
        else:
            # Assume it's already a node
            return other
    
    # =========================================================================
    # Arithmetic Operators - Generate AIR operations directly
    # Each operator captures Python source location for debugging
    # =========================================================================
    
    def _set_loc(self):
        """
        Set source location from the user's code that called the operator.
        
        Walks up the call stack to find the first frame outside the ace_edsl
        library, which should be the user's source code.
        
        Note: All values passed to set_loc() are explicitly wrapped with int()
        to ensure they are proper Python integers for the C++ binding.
        """
        if (not AIRValue.SOURCE_LOC_ENABLED or self._container is None or
                not hasattr(self._container, 'set_loc')):
            return
        
        try:
            from ace_edsl.base_dsl.loc import find_user_frame, register_file
        except ImportError:
            return  # Source location tracking not available
        
        user_frame = find_user_frame()
        if user_frame is not None:
            file_id = register_file(user_frame.f_code.co_filename)
            # Explicitly wrap with int() to ensure proper integer types for C++ binding
            self._container.set_loc(int(file_id), int(user_frame.f_lineno), int(0))
    
    def __add__(self, other: Any) -> 'AIRValue':
        """Emit AIR add operation: a + b → container.new_add()"""
        self._set_loc()  # Capture source location
        other_node = self._get_other_node(other)
        
        # Try domain-specific operations first, then fallback to generic
        # Use self.value to get fresh load if in flat IR mode
        self_node = self.value
        if self._is_core_scalar() and hasattr(self._container, 'new_core_add'):
            result_node = self._container.new_core_add(self_node, other_node)
        elif self._domain == "fhe::ckks" and hasattr(self._container, 'new_ckks_add'):
            result_node = self._container.new_ckks_add(self_node, other_node)
        elif self._domain == "fhe::sihe" and hasattr(self._container, 'new_sihe_add'):
            result_node = self._container.new_sihe_add(self_node, other_node)
        elif self._domain == "nn::core" and hasattr(self._container, 'new_nn_add'):
            result_node = self._container.new_nn_add(self_node, other_node)
        elif self._domain == "nn::vector" and hasattr(self._container, 'new_vec_add'):
            result_node = self._container.new_vec_add(self_node, other_node)
        elif hasattr(self._container, 'new_add'):
            result_node = self._container.new_add(self_node, other_node)
        elif hasattr(self._container, 'new_core_add'):
            result_node = self._container.new_core_add(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support add operation")
        
        return self._flatten_result(result_node)
    
    def __radd__(self, other: Any) -> 'AIRValue':
        """Handle reverse add (e.g., scalar + AIRValue)."""
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> 'AIRValue':
        """Emit AIR subtract operation: a - b → container.new_sub()"""
        self._set_loc()
        other_node = self._get_other_node(other)
        
        self_node = self.value
        if self._domain == "fhe::ckks" and hasattr(self._container, 'new_ckks_sub'):
            result_node = self._container.new_ckks_sub(self_node, other_node)
        elif self._domain == "fhe::sihe" and hasattr(self._container, 'new_sihe_sub'):
            result_node = self._container.new_sihe_sub(self_node, other_node)
        elif self._domain == "nn::core" and hasattr(self._container, 'new_nn_sub'):
            result_node = self._container.new_nn_sub(self_node, other_node)
        elif self._domain == "nn::vector" and hasattr(self._container, 'new_vec_sub'):
            result_node = self._container.new_vec_sub(self_node, other_node)
        elif hasattr(self._container, 'new_sub'):
            result_node = self._container.new_sub(self_node, other_node)
        elif hasattr(self._container, 'new_core_sub'):
            result_node = self._container.new_core_sub(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support sub operation")
        
        return self._flatten_result(result_node)
    
    def __rsub__(self, other: Any) -> 'AIRValue':
        """Handle reverse subtract (e.g., scalar - AIRValue)."""
        self._set_loc()
        # Create constant node for scalar
        other_node = self._get_other_node(other)
        # Reverse: other - self = -(self - other)
        neg_self = self.__neg__()
        return AIRValue(other_node, self._container, self._shape, self._domain).__add__(neg_self)
    
    def __mul__(self, other: Any) -> 'AIRValue':
        """Emit AIR multiply operation: a * b → container.new_mul()

        For CKKS domain, emit only the multiply node. Rescale placement is
        delegated to the CKKS scale manager unless the caller explicitly emits
        `.rescale()`. Inserting `new_ckks_rescale()` here conflicts with the
        primitive bootstrap decomposition, which relies on the pipeline to
        manage scale and level transitions.
        """
        self._set_loc()
        other_node = self._get_other_node(other)
        
        self_node = self.value
        if self._is_core_scalar() and hasattr(self._container, 'new_core_mul'):
            result_node = self._container.new_core_mul(self_node, other_node)
        elif self._domain == "fhe::ckks" and hasattr(self._container, 'new_ckks_mul'):
            result_node = self._container.new_ckks_mul(self_node, other_node)
        elif self._domain == "fhe::sihe" and hasattr(self._container, 'new_sihe_mul'):
            result_node = self._container.new_sihe_mul(self_node, other_node)
        elif self._domain == "nn::core" and hasattr(self._container, 'new_nn_mul'):
            result_node = self._container.new_nn_mul(self_node, other_node)
        elif self._domain == "nn::vector" and hasattr(self._container, 'new_vec_mul'):
            result_node = self._container.new_vec_mul(self_node, other_node)
        elif hasattr(self._container, 'new_mul'):
            result_node = self._container.new_mul(self_node, other_node)
        elif hasattr(self._container, 'new_core_mul'):
            result_node = self._container.new_core_mul(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support mul operation")
        
        return self._flatten_result(result_node)
    
    def __rmul__(self, other: Any) -> 'AIRValue':
        """Handle reverse multiply."""
        return self.__mul__(other)

    def __lshift__(self, other: Any) -> 'AIRValue':
        """Emit typed Core left shift: ``a << b``."""
        self._set_loc()
        if not self._is_core_scalar():
            raise TypeError("left shift requires a Core scalar AIRValue")
        other_node = self._get_other_node(other)
        if hasattr(self._container, "new_core_shl"):
            result_node = self._container.new_core_shl(
                self.value, other_node
            )
        elif hasattr(self._container, "new_shl"):
            result_node = self._container.new_shl(self.value, other_node)
        else:
            raise NotImplementedError("Container does not support Core SHL")
        return self._flatten_result(result_node)

    def __rlshift__(self, other: Any) -> 'AIRValue':
        """Emit typed Core left shift for a Python literal lhs."""
        self._set_loc()
        if not isinstance(other, int):
            raise TypeError("reverse left shift requires an integer lhs")
        if not self._is_core_scalar() or self._air_type is None:
            raise TypeError("reverse left shift requires a typed Core scalar")
        lhs = self._container.new_intconst_typed(other, self._air_type)
        if hasattr(self._container, "new_core_shl"):
            result_node = self._container.new_core_shl(lhs, self.value)
        elif hasattr(self._container, "new_shl"):
            result_node = self._container.new_shl(lhs, self.value)
        else:
            raise NotImplementedError("Container does not support Core SHL")
        return self._flatten_result(result_node)
    
    def __truediv__(self, other: Any) -> 'AIRValue':
        """Emit AIR divide operation: a / b → container.new_div()"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_div'):
            result_node = self._container.new_div(self_node, other_node)
        elif hasattr(self._container, 'new_core_div'):
            result_node = self._container.new_core_div(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support div operation")
        
        return self._flatten_result(result_node)
    
    def __rtruediv__(self, other: Any) -> 'AIRValue':
        """Handle reverse divide (e.g., scalar / AIRValue)."""
        self._set_loc()
        raise NotImplementedError("Reverse division not yet implemented")
    
    def __floordiv__(self, other: Any) -> 'AIRValue':
        """Emit AIR floor divide operation: a // b"""
        self._set_loc()
        return self.__truediv__(other)
    
    def __rfloordiv__(self, other: Any) -> 'AIRValue':
        """Handle reverse floor divide."""
        self._set_loc()
        raise NotImplementedError("Reverse floor division not yet implemented")
    
    def __mod__(self, other: Any) -> 'AIRValue':
        """Emit AIR modulo operation: a % b"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_mod'):
            result_node = self._container.new_mod(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support mod operation")
        
        return self._flatten_result(result_node)
    
    def __rmod__(self, other: Any) -> 'AIRValue':
        """Handle reverse modulo."""
        self._set_loc()
        raise NotImplementedError("Reverse modulo not yet implemented")
    
    def __pow__(self, other: Any, modulo: Optional[Any] = None) -> 'AIRValue':
        """Emit AIR power operation: a ** b"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_pow'):
            result_node = self._container.new_pow(self_node, other_node)
        elif hasattr(self._container, 'new_core_pow'):
            result_node = self._container.new_core_pow(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support pow operation")
        
        return self._flatten_result(result_node)
    
    def __rpow__(self, other: Any) -> 'AIRValue':
        """Handle reverse power."""
        self._set_loc()
        raise NotImplementedError("Reverse power not yet implemented")
    
    def __matmul__(self, other: Any) -> 'AIRValue':
        """Emit AIR matrix multiplication: a @ b → container.new_matmul()"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_matmul'):
            result_node = self._container.new_matmul(self_node, other_node)
        elif hasattr(self._container, 'new_core_matmul'):
            result_node = self._container.new_core_matmul(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support matmul operation")
        
        return self._flatten_result(result_node)
    
    # =========================================================================
    # Unary Operators
    # =========================================================================
    
    def __neg__(self) -> 'AIRValue':
        """Emit AIR negation: -a"""
        self._set_loc()
        self_node = self.value
        if hasattr(self._container, 'new_neg'):
            result_node = self._container.new_neg(self_node)
        elif hasattr(self._container, 'new_core_neg'):
            result_node = self._container.new_core_neg(self_node)
        else:
            # Negation as 0 - self
            if self._air_type is not None and hasattr(self._container, 'new_zero'):
                zero_node = self._container.new_zero(self._air_type)
            else:
                zero_node = self._container.new_intconst(0)
            result_node = self._container.new_sub(zero_node, self_node)
        
        return self._flatten_result(result_node)
    
    def __pos__(self) -> 'AIRValue':
        """Emit AIR positive: +a (no-op, return self)"""
        return self
    
    def __abs__(self) -> 'AIRValue':
        """Emit AIR absolute value: abs(a)"""
        self._set_loc()
        self_node = self.value
        if hasattr(self._container, 'new_abs'):
            result_node = self._container.new_abs(self_node)
        elif hasattr(self._container, 'new_core_abs'):
            result_node = self._container.new_core_abs(self_node)
        else:
            raise NotImplementedError("Container does not support abs operation")
        
        return self._flatten_result(result_node)
    
    # =========================================================================
    # Comparison Operators
    # =========================================================================
    
    def __eq__(self, other: Any) -> 'AIRValue':
        """Emit AIR equality comparison: a == b"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_eq'):
            result_node = self._container.new_eq(self_node, other_node)
        elif hasattr(self._container, 'new_core_eq'):
            result_node = self._container.new_core_eq(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support eq operation")
        
        return AIRValue(result_node, self._container, self._shape, self._domain)
    
    def __ne__(self, other: Any) -> 'AIRValue':
        """Emit AIR inequality comparison: a != b"""
        self._set_loc()
        eq_result = self.__eq__(other)
        if hasattr(self._container, 'new_not'):
            result_node = self._container.new_not(eq_result.value)
        else:
            raise NotImplementedError("Container does not support ne operation")
        
        return AIRValue(result_node, self._container, self._shape, self._domain)
    
    def __lt__(self, other: Any) -> 'AIRValue':
        """Emit AIR less-than comparison: a < b"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if self._is_core_scalar() and hasattr(self._container, 'new_core_lt'):
            result_node = self._container.new_core_lt(self_node, other_node)
        elif hasattr(self._container, 'new_lt'):
            result_node = self._container.new_lt(self_node, other_node)
        elif hasattr(self._container, 'new_core_lt'):
            result_node = self._container.new_core_lt(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support lt operation")
        
        result_type = self._infer_air_type(result_node)
        return AIRValue(
            result_node,
            self._container,
            self._shape,
            "air::core" if self._is_core_scalar() else self._domain,
            air_type=result_type,
        )
    
    def __le__(self, other: Any) -> 'AIRValue':
        """Emit AIR less-than-or-equal comparison: a <= b"""
        self._set_loc()
        gt_result = self.__gt__(other)
        if hasattr(self._container, 'new_not'):
            result_node = self._container.new_not(gt_result.value)
        else:
            raise NotImplementedError("Container does not support le operation")
        
        return AIRValue(result_node, self._container, self._shape, self._domain)
    
    def __gt__(self, other: Any) -> 'AIRValue':
        """Emit AIR greater-than comparison: a > b"""
        self._set_loc()
        other_node = self._get_other_node(other)
        self_node = self.value
        
        if hasattr(self._container, 'new_gt'):
            result_node = self._container.new_gt(self_node, other_node)
        elif hasattr(self._container, 'new_core_gt'):
            result_node = self._container.new_core_gt(self_node, other_node)
        else:
            raise NotImplementedError("Container does not support gt operation")
        
        return AIRValue(result_node, self._container, self._shape, self._domain)
    
    def __ge__(self, other: Any) -> 'AIRValue':
        """Emit AIR greater-than-or-equal comparison: a >= b"""
        self._set_loc()
        lt_result = self.__lt__(other)
        if hasattr(self._container, 'new_not'):
            result_node = self._container.new_not(lt_result.value)
        else:
            raise NotImplementedError("Container does not support ge operation")
        
        return AIRValue(result_node, self._container, self._shape, self._domain)
    
    # =========================================================================
    # FHE/CKKS Operations
    # =========================================================================
    
    def rotate(self, amount: int) -> 'AIRValue':
        """
        Emit CKKS rotation operation: ct.rotate(amount)
        
        Rotates ciphertext slots by the given amount.
        Positive amount = rotate left, negative = rotate right.
        
        Args:
            amount: Number of slots to rotate
            
        Returns:
            AIRValue representing the rotated ciphertext
            
        Example:
            @ckks_kernel
            def dft_layer(ct):
                rotated = ct.rotate(4)
                return ct + rotated
        """
        self._set_loc()
        
        # Handle Index type from range_constexpr (has .value attribute)
        if hasattr(amount, 'value'):
            amount = amount.value
        amount = int(amount)
        
        # The C++ binding expects an int for the rotation amount, not a Node
        self_node = self.value
        if hasattr(self._container, 'new_ckks_rotate'):
            result_node = self._container.new_ckks_rotate(self_node, amount)
        elif hasattr(self._container, 'new_sihe_rotate'):
            result_node = self._container.new_sihe_rotate(self_node, amount)
        else:
            raise NotImplementedError("Container does not support rotate operation")
        
        return self._flatten_result(result_node)

    def rotate_batch(self, amounts: list[int]) -> 'AIRValue':
        """
        Emit a grouped CKKS rotate batch over one source ciphertext.

        Returns an array of ciphertexts that can be indexed with ``[i]`` to
        access the rotated outputs.
        """
        self._set_loc()
        if not amounts:
            raise ValueError("rotate_batch requires at least one rotation")

        norm_amounts = []
        for amount in amounts:
            if hasattr(amount, 'value'):
                amount = amount.value
            norm_amounts.append(int(amount))

        self_node = self.value
        if not hasattr(self._container, 'new_ckks_rotate_batch'):
            raise NotImplementedError("Container does not support rotate_batch")

        result_node = self._container.new_ckks_rotate_batch(self_node, norm_amounts)

        if not AIRValue.FLAT_IR_MODE or not hasattr(self._container, 'new_stid'):
            return AIRValue(
                result_node,
                self._container,
                shape=(len(norm_amounts),),
                domain=self._domain,
            )

        temp_name = AIRValue._next_temp_name()
        self._container.new_stid(temp_name, result_node)
        return AIRValue(
            node=None,
            container=self._container,
            shape=(len(norm_amounts),),
            domain=self._domain,
            temp_name=temp_name,
        )
    
    def rescale(self) -> 'AIRValue':
        """
        Emit CKKS rescale operation.

        Reduces ciphertext scale after multiplication.

        Returns:
            AIRValue representing the rescaled ciphertext
        """
        self._set_loc()
        self_node = self.value

        if hasattr(self._container, 'new_ckks_rescale'):
            result_node = self._container.new_ckks_rescale(self_node)
        else:
            # No-op if rescale not available
            return self

        return self._flatten_result(result_node)

    def mod_switch(self) -> 'AIRValue':
        """
        Emit CKKS modulus switch operation.

        Reduces ciphertext modulus (level) by one without rescaling.

        Returns:
            AIRValue representing ciphertext with reduced level
        """
        self._set_loc()
        self_node = self.value

        if hasattr(self._container, 'new_ckks_mod_switch'):
            result_node = self._container.new_ckks_mod_switch(self_node)
        else:
            # No-op if mod_switch not available
            return self

        return self._flatten_result(result_node)

    def relin(self) -> 'AIRValue':
        """
        Emit CKKS relinearization operation.
        
        After multiplication, ciphertext has 3 polynomials.
        Relinearization reduces back to 2 polynomials.
        
        Returns:
            AIRValue representing the relinearized ciphertext
        """
        self._set_loc()
        self_node = self.value
        
        if hasattr(self._container, 'new_ckks_relin'):
            result_node = self._container.new_ckks_relin(self_node)
        else:
            # No-op if relin not available
            return self
        
        return self._flatten_result(result_node)
    
    def bootstrap(self) -> 'AIRValue':
        """
        Emit CKKS bootstrapping operation.
        
        Refreshes ciphertext noise budget through bootstrapping.
        
        Returns:
            AIRValue representing the bootstrapped ciphertext
        """
        self._set_loc()
        self_node = self.value
        
        if hasattr(self._container, 'new_ckks_bootstrap'):
            result_node = self._container.new_ckks_bootstrap(self_node)
        else:
            raise NotImplementedError("Container does not support bootstrap operation")
        
        return self._flatten_result(result_node)

    def _bootstrap_coeffs_to_slots_primitive(self, num_slots: int = 0) -> 'AIRValue':
        """Full CoeffToSlot decomposition (DFT butterfly rotation pattern).

        Emits raise_mod + baby-step/giant-step rotation structure matching
        the rtlib Coeff_slots_transform encoding path.
        """
        from .bootstrap_decomposition import coeffs_to_slots_primitive
        return coeffs_to_slots_primitive(self, num_slots)

    def _bootstrap_eval_mod_primitive(self) -> 'AIRValue':
        """Full EvalMod decomposition (Paterson-Stockmeyer Chebyshev + double-angle).

        Replaces the identity surrogate with the actual approximate modular
        reduction: degree-54 Chebyshev series evaluated via PS algorithm
        (k=8, m=3) followed by 3 double-angle iterations.
        Mirrors Eval_approx_mod in bootstrap.c / chebyshev_impl.c.
        """
        from .bootstrap_decomposition import eval_mod_primitive
        return eval_mod_primitive(self)

    def _bootstrap_slots_to_coeffs_primitive(self, num_slots: int = 0) -> 'AIRValue':
        """Full SlotToCoeff decomposition (inverse DFT rotation pattern).

        Emits baby-step/giant-step rotation structure matching
        the rtlib Coeff_slots_transform decoding path.
        """
        from .bootstrap_decomposition import slots_to_coeffs_primitive
        return slots_to_coeffs_primitive(self, num_slots)

    def bootstrap_coeffs_to_slots(self, num_slots: int = 0) -> 'AIRValue':
        """
        Emit CKKS bootstrap coeffs-to-slots stage operation.

        Args:
            num_slots: Target slots for precom lookup (0 = use ciphertext slots)

        Returns:
            AIRValue representing transformed ciphertext
        """
        self._set_loc()
        self_node = self.value
        num_slots = int(num_slots)

        if self._bootstrap_stage_primitive_enabled():
            return self._bootstrap_coeffs_to_slots_primitive(num_slots)

        if hasattr(self._container, "new_ckks_bootstrap_coeffs_to_slots"):
            result_node = self._container.new_ckks_bootstrap_coeffs_to_slots(
                self_node, num_slots
            )
        else:
            raise NotImplementedError(
                "Container does not support bootstrap_coeffs_to_slots operation"
            )

        return self._flatten_result(result_node)

    def bootstrap_eval_mod(self) -> 'AIRValue':
        """
        Emit CKKS bootstrap EvalMod stage operation.

        Returns:
            AIRValue representing transformed ciphertext
        """
        self._set_loc()
        self_node = self.value

        if self._bootstrap_stage_primitive_enabled():
            return self._bootstrap_eval_mod_primitive()

        if hasattr(self._container, "new_ckks_bootstrap_eval_mod"):
            result_node = self._container.new_ckks_bootstrap_eval_mod(self_node)
        else:
            raise NotImplementedError(
                "Container does not support bootstrap_eval_mod operation"
            )

        return self._flatten_result(result_node)

    def bootstrap_slots_to_coeffs(self, num_slots: int = 0) -> 'AIRValue':
        """
        Emit CKKS bootstrap slots-to-coeffs stage operation.

        Args:
            num_slots: Target slots for precom lookup (0 = use ciphertext slots)

        Returns:
            AIRValue representing transformed ciphertext
        """
        self._set_loc()
        self_node = self.value
        num_slots = int(num_slots)

        if self._bootstrap_stage_primitive_enabled():
            return self._bootstrap_slots_to_coeffs_primitive(num_slots)

        if hasattr(self._container, "new_ckks_bootstrap_slots_to_coeffs"):
            result_node = self._container.new_ckks_bootstrap_slots_to_coeffs(
                self_node, num_slots
            )
        else:
            raise NotImplementedError(
                "Container does not support bootstrap_slots_to_coeffs operation"
            )

        return self._flatten_result(result_node)

    def conjugate(self) -> 'AIRValue':
        """
        Emit CKKS conjugation operation.

        Returns:
            AIRValue representing the conjugated ciphertext
        """
        self._set_loc()
        self_node = self.value

        if hasattr(self._container, 'new_ckks_conjugate'):
            result_node = self._container.new_ckks_conjugate(self_node)
        else:
            raise NotImplementedError("Container does not support conjugate operation")

        return self._flatten_result(result_node)

    def mul_mono(self, power: int) -> 'AIRValue':
        """
        Emit CKKS multiply-by-monomial operation.

        Args:
            power: Monomial power (X^power)

        Returns:
            AIRValue representing the transformed ciphertext
        """
        self._set_loc()
        self_node = self.value
        power = int(power)

        if hasattr(self._container, 'new_ckks_mul_mono'):
            result_node = self._container.new_ckks_mul_mono(self_node, power)
        else:
            raise NotImplementedError("Container does not support mul_mono operation")

        return self._flatten_result(result_node)

    def raise_mod(
        self, mod_size: int, runtime_raise_level: bool = False
    ) -> 'AIRValue':
        """
        Emit CKKS raise_mod operation.

        Args:
            mod_size: Target modulus size/level parameter used by compiler
                analysis
            runtime_raise_level: Emit the C raise target through the configured
                runtime helper while preserving mod_size in AIR metadata

        Returns:
            AIRValue representing raised ciphertext
        """
        self._set_loc()
        self_node = self.value
        mod_size = int(mod_size)

        if hasattr(self._container, 'new_ckks_raise_mod'):
            result_node = self._container.new_ckks_raise_mod(
                self_node, mod_size, runtime_raise_level
            )
        else:
            raise NotImplementedError("Container does not support raise_mod operation")

        return self._flatten_result(result_node)
    
    # =========================================================================
    # Indexing - Generate AIR load/store operations
    # =========================================================================
    
    def __getitem__(self, idx: Union[int, Tuple[int, ...], slice]) -> 'AIRValue':
        """
        Emit AIR indexed load operation: a[i] → container.new_ild()
        
        Supports:
            x[0]      - Single index
            x[0, 1]   - Multiple indices
            x[0:10]   - Slicing (emits slice operation)
        """
        self._set_loc()
        self_node = self.value
        
        if isinstance(idx, slice):
            # Handle slicing
            start = idx.start or 0
            stop = idx.stop
            step = idx.step or 1
            
            if hasattr(self._container, 'new_slice'):
                result_node = self._container.new_slice(self_node, start, stop, step)
            else:
                raise NotImplementedError("Slicing not supported by container")
            
            return self._flatten_result(result_node)
        
        # Handle tuple of indices
        if isinstance(idx, tuple):
            idx_nodes = []
            for i in idx:
                if isinstance(i, int):
                    idx_nodes.append(self._container.new_intconst(i))
                elif isinstance(i, AIRValue):
                    idx_nodes.append(i.value)
                else:
                    idx_nodes.append(i)
            idx_node = idx_nodes[0] if len(idx_nodes) == 1 else idx_nodes
        elif isinstance(idx, int):
            idx_node = self._container.new_intconst(idx)
        elif isinstance(idx, AIRValue):
            idx_node = idx.value
        else:
            idx_node = idx
        
        # Create indexed load operation
        if hasattr(self._container, 'new_ild'):
            result_node = self._container.new_ild(self_node, idx_node)
        elif hasattr(self._container, 'new_load'):
            result_node = self._container.new_load(self_node, idx_node)
        else:
            raise NotImplementedError("Indexed load not supported by container")
        
        return self._flatten_result(result_node)
    
    def __setitem__(self, idx: Union[int, Tuple[int, ...]], value: Any) -> None:
        """
        Emit AIR indexed store operation: a[i] = value → container.new_ist()
        
        Note: This modifies the container but returns None (like Python assignment).
        """
        self._set_loc()
        self_node = self.value
        
        # Extract value node
        if isinstance(value, AIRValue):
            value_node = value.value
        elif isinstance(value, (int, float)):
            value_node = self._container.new_intconst(int(value))
        else:
            value_node = value
        
        # Extract index node
        if isinstance(idx, tuple):
            idx_nodes = []
            for i in idx:
                if isinstance(i, int):
                    idx_nodes.append(self._container.new_intconst(i))
                elif isinstance(i, AIRValue):
                    idx_nodes.append(i.value)
                else:
                    idx_nodes.append(i)
            idx_node = idx_nodes[0] if len(idx_nodes) == 1 else idx_nodes
        elif isinstance(idx, int):
            idx_node = self._container.new_intconst(idx)
        elif isinstance(idx, AIRValue):
            idx_node = idx.value
        else:
            idx_node = idx
        
        # Create indexed store operation
        if hasattr(self._container, 'new_ist'):
            self._container.new_ist(value_node, self_node, idx_node)
        elif hasattr(self._container, 'new_store'):
            self._container.new_store(value_node, self_node, idx_node)
        else:
            raise NotImplementedError("Indexed store not supported by container")
    
    # =========================================================================
    # String Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if hasattr(self._node, 'to_string'):
            return f"AIRValue({self._node.to_string()})"
        elif hasattr(self._node, 'name'):
            return f"AIRValue({self._node.name()})"
        else:
            return f"AIRValue({self._node})"
    
    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()


def _wrap_air_value(
    node: Any,
    container: Any,
    shape: Optional[Tuple[int, ...]] = None,
    domain: Optional[str] = None,
    temp_name: Optional[str] = None,
    air_type: Any = None,
    vector_slot: Optional[int] = None,
) -> AIRValue:
    """Construct a domain facade without changing AIR object ownership."""
    if domain == "nn::vector":
        from .vector_value import VectorValue

        return VectorValue(
            node,
            container,
            shape=shape,
            temp_name=temp_name,
            air_type=air_type,
            vector_slot=vector_slot,
        )
    return AIRValue(
        node,
        container,
        shape=shape,
        domain=domain,
        temp_name=temp_name,
        air_type=air_type,
        vector_slot=vector_slot,
    )


def to_air(values: Sequence[Any]) -> AIRValue:
    """Convert a Python sequence of AIR values into an AIR array value.

    Python's built-in ``list`` type cannot be extended with new methods at
    runtime. The EDSL preprocessor rewrites ``[a, b].to_air()`` to
    ``to_air([a, b])`` before execution.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError("to_air expects a Python list or tuple of AIRValue objects")
    if not values:
        raise ValueError("to_air requires at least one AIRValue")

    container = None
    domain = None
    for item in values:
        if isinstance(item, AIRValue):
            container = item.container
            domain = item.domain
            break

    if container is None:
        raise TypeError("to_air requires at least one AIRValue to provide a container")
    if not hasattr(container, "new_air_array"):
        raise NotImplementedError("Container does not support AIR array construction")

    nodes = []
    for index, item in enumerate(values):
        if not isinstance(item, AIRValue):
            raise TypeError(
                f"to_air expects AIRValue elements, got {type(item).__name__} "
                f"at index {index}"
            )
        if item.container is not container:
            raise ValueError("to_air cannot mix AIRValue objects from different containers")
        nodes.append(item.value)

    temp_name = AIRValue._next_temp_name()
    container.new_air_array(temp_name, nodes)
    return _wrap_air_value(
        node=None,
        container=container,
        shape=(len(values),),
        domain=domain,
        temp_name=temp_name,
    )
