"""
AceEDSL - ACE Embedded DSL with Multiple Domains

Based on BaseDSL structure but generates AIR instead of MLIR.
Uses operator overloading to generate AIR operations during function execution.

IMPORTANT: We do NOT generate MLIR.
- BaseDSL (which we inherit from) contains MLIR code (_mlir_helpers, etc.)
- We only use BaseDSL for AST preprocessing and infrastructure
- All IR generation happens through AIR via operator overloading
- We ignore all MLIR functionality in BaseDSL
- This is not GPU-specific - generates AIR for any target (not compute-capability specific).
"""

from typing import Any, Optional, Dict
from contextlib import contextmanager
import inspect
import sys
import os

# Import BaseDSL infrastructure
# 
# IMPORTANT: BaseDSL contains MLIR code (_mlir_helpers, MLIR imports, etc.)
# We do NOT use any MLIR functionality - we only use BaseDSL for:
# - AST preprocessing (ast_preprocessor.py)
# - Infrastructure (logging, environment variables, etc.)
# - Base class structure (BaseDSL class)
#
# When we import BaseDSL, it will pull in MLIR dependencies, but we ignore them.
# All IR generation happens through AIR via operator overloading, NOT MLIR.
from ..base_dsl.dsl import BaseDSL
from ..base_dsl.ast_helpers import *
from ..base_dsl.utils.logger import log

# Import AIR bindings
from ace_bindings import air_builder

from .core.air_value import AIRValue, _wrap_air_value, to_air
from .core.vector_array import VectorArray
from .core.vector_value import VectorValue
from .core.domain_registry import DOMAIN_PIPELINES
from .core.type_mapping import python_type_to_air_type, is_plaintext_annotation
from .domain_ast_decorators import (
    _is_dynamic_expression,
    _loop_execute_range_dynamic,
    _if_execute_dynamic,
    _while_execute_dynamic,
)


class StubCompilerProvider:
    """
    Stub compiler provider for BaseDSL.
    
    BaseDSL requires a compiler_provider, but AceEDSL doesn't use MLIR compilation.
    This is a minimal stub that satisfies BaseDSL's constructor requirements.
    """
    pass


class AceEDSL(BaseDSL):
    """
    Single DSL class that handles multiple domains and generates AIR.
    
    Mimics BaseDSL structure but generates AIR instead of MLIR.
    Uses operator overloading to generate AIR operations during function execution.
    
    Domains are selected via decorators (@tensor_kernel, @vector_kernel, etc.)
    which set the _py_domain attribute on functions.
    """
    
    def __init__(self):
        name = "Py_DSL"
        # BaseDSL requires compiler_provider, but we don't use MLIR compilation
        # Use a stub compiler provider to satisfy BaseDSL's requirements
        compiler_provider = StubCompilerProvider()
        # Not GPU-specific - these are required by BaseDSL but not used for AIR
        # Use a placeholder value since BaseDSL checks for non-empty string
        pass_sm_arch_name = "air"  # Placeholder (not GPU-specific, but satisfies BaseDSL requirement)
        device_compilation_only = False  # Not device-specific
        super().__init__(
            name,
            compiler_provider,
            pass_sm_arch_name,
            device_compilation_only,
            preprocess=True  # Enable AST preprocessing (only part of BaseDSL we use)
        )
        self.preprocessor.register_non_mutating_receiver_methods(
            VectorValue.NON_MUTATING_RECEIVER_METHODS
            | VectorArray.NON_MUTATING_RECEIVER_METHODS
        )
        self.no_cache = True
        self.current_domain = None  # Set by decorator
        self._domain_stack = []
        self.current_air_module = None  # Store AIR module after generation
        self._ret_temp_counter = 0
        self._in_air_context = False  # Track if we're inside AIR generation
        self._scalar_encode_map = {}  # arg_name -> scalar value (encode in kernel body)
        self._array_encode_map = {}   # arg_name -> {"values": list, "is_complex": bool, "complex_len": int}
        
        # Register AST decorator functions with executor
        # This enables dynamic execution of loops and conditionals
        from ..base_dsl.ast_helpers import executor
        executor.set_functions(
            is_dynamic_expression=_is_dynamic_expression,
            loop_execute_range_dynamic=_loop_execute_range_dynamic,
            if_dynamic=_if_execute_dynamic,
            while_dynamic=_while_execute_dynamic,
        )

    @contextmanager
    def _domain_scope(self, domain):
        """Push a nested kernel domain and restore it even on failure."""
        previous_domain = self.current_domain
        self._domain_stack.append(domain)
        self.current_domain = domain
        from .domain_ast_decorators import set_current_domain
        set_current_domain(domain)
        try:
            yield
        finally:
            self._domain_stack.pop()
            self.current_domain = previous_domain
            set_current_domain(previous_domain)

    def _view_for_domain(self, value, domain):
        """Recursively create non-mutating lexical views of AIR operands."""
        if isinstance(value, AIRValue):
            return value.view(domain)
        if isinstance(value, list):
            return [self._view_for_domain(item, domain) for item in value]
        if isinstance(value, tuple):
            return tuple(self._view_for_domain(item, domain) for item in value)
        if isinstance(value, dict):
            return {
                key: self._view_for_domain(item, domain)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _domain_for_air_type(air_type, default_domain):
        if air_type is not None:
            try:
                if air_type.is_scalar():
                    return "air::core"
            except (AttributeError, RuntimeError):
                pass
        return default_domain

    def trace_vector_kernel_into(self, trace_context, kernel, prepared):
        """Trace one real ``@vector_kernel`` into a C++-owned helper body.

        The pass owns the function, signature, body, constants, and terminal
        return. This method only establishes a lexical EDSL context around the
        borrowed destination wrappers for the duration of one callback.
        """
        if not getattr(trace_context, "active", False):
            raise RuntimeError("vector-kernel destination trace is not active")
        if getattr(kernel, "_py_domain", None) != "nn::vector":
            raise TypeError("destination recipe requires a @vector_kernel callable")

        from .domain_ast_decorators import (
            restore_tracing_state,
            set_current_container,
            set_destination_trace_mode,
            snapshot_tracing_state,
        )

        trace_state = snapshot_tracing_state()
        previous_in_context = self._in_air_context
        previous_domain = self.current_domain
        previous_stack = list(self._domain_stack)
        previous_module = self.current_air_module
        previous_flat_mode = AIRValue.FLAT_IR_MODE
        previous_source_loc_mode = AIRValue.SOURCE_LOC_ENABLED
        previous_fresh_load_mode = AIRValue.FRESH_LOAD_ENABLED
        previous_temp_counter = AIRValue._temp_counter
        missing = object()
        previous_original = getattr(self, "_original_funcBody", missing)

        container = trace_context.container

        def wrap(node):
            air_type = node.rtype()
            shape = tuple(air_type.shape()) if air_type.is_array() else None
            return _wrap_air_value(
                node,
                container,
                shape=shape,
                domain=self._domain_for_air_type(air_type, "nn::vector"),
                air_type=air_type,
            )

        formals = tuple(wrap(node) for node in trace_context.formals)
        constants = {
            role: wrap(trace_context.constant(role))
            for role in trace_context.constant_roles
        }
        expected_type = trace_context.result_type

        try:
            self._in_air_context = True
            self.current_domain = None
            self._domain_stack = []
            AIRValue.FLAT_IR_MODE = False
            AIRValue.SOURCE_LOC_ENABLED = False
            AIRValue.FRESH_LOAD_ENABLED = True
            set_current_container(container, None, "nn::vector")
            set_destination_trace_mode(True)
            result = kernel(*formals, prepared, constants, expected_type,
                            trace_context.i32_type)
            if not isinstance(result, AIRValue):
                raise TypeError(
                    "vector-kernel destination recipe must return an AIRValue"
                )
            if result.container is not container:
                raise ValueError(
                    "vector-kernel destination recipe returned another container"
                )
            if result.domain != "nn::vector":
                raise TypeError(
                    "vector-kernel destination recipe result is not Vector AIR"
                )
            if result.air_type is None or result.air_type != expected_type:
                raise TypeError(
                    "vector-kernel destination recipe result type does not match "
                    "the prepared ABI"
                )
            return result.value
        finally:
            restore_tracing_state(trace_state)
            self._in_air_context = previous_in_context
            self.current_domain = previous_domain
            self._domain_stack = previous_stack
            self.current_air_module = previous_module
            AIRValue.FLAT_IR_MODE = previous_flat_mode
            AIRValue.SOURCE_LOC_ENABLED = previous_source_loc_mode
            AIRValue.FRESH_LOAD_ENABLED = previous_fresh_load_mode
            AIRValue._temp_counter = previous_temp_counter
            if previous_original is missing:
                if hasattr(self, "_original_funcBody"):
                    del self._original_funcBody
            else:
                self._original_funcBody = previous_original
    
    def _kernel_helper(self, funcBody, *args, **kwargs):
        """
        Domain-aware kernel generation.
        
        Mimics BaseDSL._kernel_helper but generates AIR instead of MLIR.
        """
        # Get domain from function attribute (set by decorator)
        # If funcBody was preprocessed, get domain from original function
        original_funcBody = funcBody
        if hasattr(funcBody, '__wrapped__'):
            original_funcBody = funcBody.__wrapped__
        elif hasattr(funcBody, 'dsl_object') and hasattr(funcBody.dsl_object, 'funcBody'):
            # Preprocessed function - get original from dsl_object
            original_funcBody = funcBody.dsl_object.funcBody
        
        domain = getattr(original_funcBody, '_py_domain', 'tensor')
        previous_original = getattr(self, '_original_funcBody', None)

        if self._in_air_context:
            with self._domain_scope(domain):
                self._original_funcBody = original_funcBody
                try:
                    viewed_args = self._view_for_domain(args, domain)
                    viewed_kwargs = self._view_for_domain(kwargs, domain)
                    result = self._func_air(
                        funcBody, *viewed_args, **viewed_kwargs
                    )
                    return self._view_for_domain(result, domain)
                finally:
                    self._original_funcBody = previous_original

        # Keep the top-level domain observable after generation, as before.
        self.current_domain = domain
        self._original_funcBody = original_funcBody
        try:
            return self._func_air(funcBody, *args, **kwargs)
        finally:
            self._in_air_context = False
            self._original_funcBody = previous_original
    
    def _func_air(self, funcBody, *args, **kwargs):
        """
        Execute function and generate AIR via operator overloading.
        
        Mimics BaseDSL._func() but generates AIR instead of MLIR.
        """
        # Check if we're already in an AIR context (similar to BaseDSL checking ir.InsertionPoint)
        # This enables nested kernel calls - when one @ckks_kernel calls another,
        # the inner kernel's operations are traced into the outer kernel's AIR.
        if self._in_air_context:
            # Already generating AIR - just execute function body directly
            # Operations will be traced into the existing container via operator overloading
            return funcBody(*args, **kwargs)
        
        function_name = funcBody.__name__
        # Store original function body for argument checking
        # If funcBody was preprocessed, we need to use the original function for signature
        original_funcBody = funcBody
        if hasattr(self, '_original_funcBody'):
            # Use original function stored by _kernel_helper
            original_funcBody = self._original_funcBody
        elif hasattr(funcBody, '__wrapped__'):
            # Preprocessed function - use original for signature
            original_funcBody = funcBody.__wrapped__
        elif hasattr(funcBody, 'dsl_object') and hasattr(funcBody.dsl_object, 'funcBody'):
            # Preprocessed function - get original from dsl_object
            original_funcBody = funcBody.dsl_object.funcBody
        
        self.funcBody = original_funcBody  # Use original for _check_arg_count
        
        # Extract kwargs (similar to BaseDSL._func)
        pipeline = kwargs.pop("pipeline", None)
        # Removed gpu_module_attrs as it's not GPU-specific
        no_cache = kwargs.pop("no_cache", False)
        compile_only = kwargs.pop("compile_only", False)
        
        # Check argument count (using original function signature)
        self._check_arg_count(*args, **kwargs)
        
        # Get args_spec - handle preprocessed functions that can't be inspected
        try:
            args_spec = inspect.getfullargspec(original_funcBody)
        except (OSError, TypeError) as e:
            # Preprocessed function created by exec() - try to get signature from funcBody
            # or use a default signature based on the number of args passed
            try:
                args_spec = inspect.getfullargspec(funcBody)
            except (OSError, TypeError):
                # Fallback: create a minimal FullArgSpec from the arguments passed
                # This is a workaround for preprocessed functions
                import inspect as inspect_module
                # Create a minimal args_spec with the function name and args
                # We'll infer from the actual call
                args_spec = inspect_module.FullArgSpec(
                    args=[f'arg{i}' for i in range(len(args))],
                    varargs=None,
                    varkw=None,
                    defaults=None,
                    kwonlyargs=[],
                    kwonlydefaults=None,
                    annotations={}
                )

        # ``getfullargspec`` leaves postponed annotations as strings. Resolve
        # them against the original function's globals before mapping them to
        # AIR types so annotations such as ``Int[32]`` keep their exact width.
        try:
            resolved_annotations = inspect.get_annotations(
                original_funcBody, eval_str=True
            )
        except (NameError, TypeError, ValueError) as exc:
            raise TypeError(
                f"Unable to resolve annotations for {function_name}: {exc}"
            ) from exc
        args_spec = args_spec._replace(annotations=resolved_annotations)
        
        # Use the actual funcBody (which may be preprocessed) for execution
        # But use original_funcBody for signature checking
        
        # Generate AIR (similar to BaseDSL.generate_mlir)
        # Use funcBody (may be preprocessed) for actual execution
        result = self.generate_air(
            funcBody,  # Use funcBody (may be preprocessed) for execution
            kwargs,
            function_name,
            # Removed gpu_module_attrs
            args,
            args_spec,  # Use original_funcBody's signature
            pipeline,
            no_cache,
            compile_only,
        )
        
        return result
    
    def generate_air(
        self,
        funcBody,
        kwargs,
        function_name,
        # Removed gpu_module_attrs
        args,
        args_spec,
        pipeline,
        no_cache,
        compile_only,
    ):
        """
        Generate AIR module from function execution.
        
        Mimics BaseDSL.generate_mlir() but generates AIR.
        """
        if air_builder is None:
            raise ImportError("air_builder bindings not available")
        
        def build_air_module():
            """Build AIR module by executing function (mimics BaseDSL.build_ir_module)"""
            # Mark that we're now in AIR generation context
            # This enables nested kernel calls - inner kernels just execute directly
            self._in_air_context = True
            
            # Reset temp counter for flat IR generation
            AIRValue.reset_temp_counter()
            
            # Create AIR global scope
            glob_scope = air_builder.create_glob_scope()
            
            # Set up source location tracking
            try:
                from ace_edsl.base_dsl.loc import set_glob_scope as set_loc_glob
                set_loc_glob(glob_scope)
            except ImportError:
                pass  # Source location tracking not available
            
            # Count parameters (excluding 'self' if present)
            param_names = [name for name in args_spec.args if name != 'self']
            
            # Create function scope with correct number of parameters
            # Prefer per-parameter types if the binding supports it
            # Scalar/array constants passed for CkksPlaintext-annotated params are
            # NOT formal parameters — they are encoded inside the kernel body
            # (like cipher+scalar), so we exclude them from param_types.
            param_types = []
            self._scalar_encode_map = {}  # arg_name -> scalar value (to encode in body)
            self._array_encode_map = {}   # arg_name -> {"values": list, "is_complex": bool, "complex_len": int}
            actual_args = list(args) if args is not None else []
            actual_idx = 0
            for i, arg_name in enumerate(args_spec.args):
                if arg_name == 'self':
                    continue
                arg_type = args_spec.annotations.get(arg_name, None)
                runtime_arg = actual_args[actual_idx] if actual_idx < len(actual_args) else None
                actual_idx += 1
                # Scalar constant for CkksPlaintext annotation: encode inside kernel
                # body just like cipher+scalar, not a formal parameter
                if isinstance(runtime_arg, (int, float)) and is_plaintext_annotation(arg_type):
                    self._scalar_encode_map[arg_name] = runtime_arg
                    continue  # Not a formal parameter
                # Array constant for CkksPlaintext annotation: encode inside kernel
                # body as CONSTANT_KIND::ARRAY → CKKS.encode (no MASK attribute)
                if isinstance(runtime_arg, (list, tuple)) and is_plaintext_annotation(arg_type):
                    normalized_vals = []
                    has_complex = False
                    for v in runtime_arg:
                        if isinstance(v, complex):
                            has_complex = True
                            normalized_vals.append(v)
                        elif isinstance(v, (list, tuple)) and len(v) == 2:
                            has_complex = True
                            normalized_vals.append(complex(float(v[0]), float(v[1])))
                        else:
                            normalized_vals.append(float(v))
                    self._array_encode_map[arg_name] = {
                        "values": normalized_vals,
                        "is_complex": has_complex,
                        "complex_len": len(runtime_arg),
                    }
                    continue  # Not a formal parameter
                # An explicit annotation is authoritative. Runtime metadata is
                # only a fallback for unannotated parameters.
                air_type = None
                if arg_type is not None:
                    try:
                        air_type = self._get_air_type_for_param(
                            arg_type, self.current_domain
                        )
                    except Exception:
                        air_type = None
                if air_type is None and runtime_arg is not None:
                    try:
                        air_type = python_type_to_air_type(runtime_arg, self.current_domain)
                    except Exception:
                        air_type = None
                if air_type is None:
                    air_type = self._get_air_type_for_param(arg_type, self.current_domain)
                param_types.append(air_type)
            
            num_params = len(param_types)

            ret_type = None
            ret_annot = args_spec.annotations.get('return', None)
            if ret_annot is not None:
                try:
                    ret_type = python_type_to_air_type(ret_annot, self.current_domain)
                except Exception:
                    ret_type = None
            if ret_type is None:
                # Preserve the historical unannotated-kernel convention: use
                # the first formal type rather than silently fabricating a
                # differently ranked [64] return.
                ret_type = (
                    param_types[0]
                    if param_types
                    else air_builder.Type.make_array(
                        [64], air_builder.Type.make_float(32)
                    )
                )

            # Preserve exact return and per-formal types, including ranked
            # shapes and scalar widths. Marker types (cipher/plaintext) are
            # resolved lazily by this API. Keep the established uniform
            # ciphertext path because it also initializes the FHE lowering
            # context expected by the CKKS driver.
            uniform_ciphertext_signature = (
                ret_type.to_string() == "CIPHERTEXT"
                and all(
                    param_type.to_string() == "CIPHERTEXT"
                    for param_type in param_types
                )
            )
            if uniform_ciphertext_signature:
                func_scope = glob_scope.new_func_with_type(
                    function_name, num_params, [64], "CIPHERTEXT"
                )
            elif hasattr(glob_scope, "new_func_with_param_types"):
                func_scope = glob_scope.new_func_with_param_types(
                    function_name, ret_type, param_types
                )
            else:
                # Fallback: uniform array parameters
                func_scope = glob_scope.new_func_with_params(function_name, num_params, [64])
            container = func_scope.container()
            
            # Generate execution arguments (convert Python args to AIRValue objects)
            # Similar to BaseDSL.generate_execution_arguments()
            ir_args = self.generate_execution_arguments_air(
                args, func_scope, function_name, args_spec
            )
            
            # Set the current container for loop operations (used by _loop_execute_range_dynamic)
            from .domain_ast_decorators import set_current_container
            set_current_container(container, func_scope, self.current_domain)
            
            # Execute function body - operator overloading generates AIR
            # Similar to BaseDSL.generate_original_ir() where funcBody(*ir_args, **kwargs) is called
            try:
                # Log execution for debugging
                log().debug(f"Executing function {function_name} with {len(ir_args)} arguments")
                log().debug(f"  Function: {funcBody}")
                log().debug(f"  Function type: {type(funcBody)}")
                log().debug(f"  Arguments: {[type(arg).__name__ for arg in ir_args]}")
                
                # Check if function was preprocessed
                if hasattr(funcBody, '__wrapped__'):
                    log().debug(f"  Function was wrapped (preprocessed)")
                if hasattr(funcBody, 'dsl_object'):
                    log().debug(f"  Function has dsl_object")
                
                # Execute the function body
                # This should generate AIR operations via operator overloading
                result = funcBody(*ir_args, **kwargs)
                
                log().debug(f"Function {function_name} executed, result type: {type(result)}")
                
                # Check how many operations were generated
                # Count operations in container (rough estimate)
                try:
                    ir_dump = glob_scope.dump()
                    op_count = ir_dump.count('new_') + ir_dump.count('call') + ir_dump.count('ret')
                    log().debug(f"  Operations in AIR module: ~{op_count}")
                except:
                    pass
                
                if result is not None:
                    if isinstance(result, AIRValue):
                        # If the result node was created inside a control-flow body
                        # (e.g., loop), return a load from a temporary stored in the
                        # outer scope to avoid container mismatches.
                        ret_container = result.container or container
                        temp_name = f"__ret_tmp_{self._ret_temp_counter}"
                        self._ret_temp_counter += 1
                        if (result.air_type is not None and
                                hasattr(ret_container, "new_local")):
                            ret_container.new_local(temp_name, result.air_type)
                        ret_node = ret_container.new_stid(
                            temp_name, result._value_without_vector_slot()
                        )
                        result._apply_vector_slot(ret_node)
                        ret_container.new_retv(ret_node)
                        log().debug(f"Returning AIRValue via temp '{temp_name}'")
                    else:
                        container.new_ret()
                        log().debug(f"Returning void (non-AIRValue result)")
            except Exception as e:
                log().error(f"Error during AIR generation for {function_name}: {e}")
                import traceback
                log().error(traceback.format_exc())
                raise RuntimeError(f"Error during AIR generation: {e}") from e
            finally:
                # Clear the container reference
                set_current_container(None, None, None)
                # Exit AIR generation context
                self._in_air_context = False
            
            return glob_scope, result
        
        # Build AIR module
        glob_scope, result = build_air_module()
        
        # Store AIR module in instance for access after execution
        self.current_air_module = glob_scope
        
        # TODO: Compile AIR and cache (similar to BaseDSL.compile_and_cache())
        # For now, just return the glob_scope
        
        return result
    
    def generate_execution_arguments_air(
        self, args, func_scope, function_name, args_spec: inspect.FullArgSpec
    ):
        """
        Convert Python arguments to AIRValue objects.
        
        Mimics BaseDSL.generate_execution_arguments() but creates AIRValue instead of MLIR values.
        
        Supports instantiation patterns:
        - None: Use type annotation for shape/dtype
        - CkksCiphertext instance: Use instance's shape/dtype/name
        
        NOTE: We generate AIR, not MLIR. This method creates AIRValue objects that wrap AIR nodes.
        """
        ir_args = []
        container = func_scope.container()
        
        # Create AIR parameters for each function argument.
        # If runtime args are provided, prefer their shape/dtype metadata
        # to derive AIR types; otherwise fall back to annotations.
        #
        # Scalar constants for CkksPlaintext-annotated params are NOT formal
        # parameters.  They are encoded inside the kernel body (intconst →
        # CKKS.encode), just like cipher+scalar operations.
        actual_args = list(args) if args is not None else []
        actual_idx = 0
        for i, arg_name in enumerate(args_spec.args):
            # Skip 'self' if present (for methods)
            if i == 0 and arg_name == 'self':
                continue
            
            # Get type annotation for this parameter
            arg_type = args_spec.annotations.get(arg_name, None)
            
            # Prefer runtime arg instance metadata (shape/dtype) when available
            runtime_arg = actual_args[actual_idx] if actual_idx < len(actual_args) else None
            actual_idx += 1

            # Scalar constant for CkksPlaintext annotation → encode inside body
            # (same path as cipher+scalar: intconst → CKKS.encode)
            if arg_name in getattr(self, '_scalar_encode_map', {}):
                scalar_val = self._scalar_encode_map[arg_name]
                if isinstance(scalar_val, float):
                    if hasattr(container, 'new_floatconst'):
                        const_node = container.new_floatconst(scalar_val)
                    else:
                        const_node = container.new_intconst(int(scalar_val))
                else:
                    const_node = container.new_intconst(int(scalar_val))
                # Encode scalar into plaintext polynomial (like cipher+scalar)
                if hasattr(container, 'new_ckks_encode'):
                    encoded_node = container.new_ckks_encode(const_node)
                else:
                    encoded_node = const_node
                air_value = AIRValue(
                    encoded_node, container, domain=self.current_domain
                )
                ir_args.append(air_value)
                continue

            # Array constant for CkksPlaintext annotation → encode inside body
            # (CONSTANT_KIND::ARRAY → ldc → CKKS.encode, no MASK attribute)
            if arg_name in getattr(self, '_array_encode_map', {}):
                array_info = self._array_encode_map[arg_name]
                if isinstance(array_info, dict):
                    array_val = array_info.get("values", [])
                    is_complex_array = bool(array_info.get("is_complex", False))
                    complex_len = int(array_info.get("complex_len", len(array_val)))
                else:
                    # Backward compatibility for older map format.
                    array_val = array_info
                    is_complex_array = False
                    complex_len = len(array_val)
                # Create an LDC node referencing an ARRAY constant
                array_node = container.new_array_const(array_val)
                # Encode array into plaintext polynomial
                if hasattr(container, 'new_ckks_encode'):
                    if is_complex_array and hasattr(container, 'new_ckks_encode_complex'):
                        encoded_node = container.new_ckks_encode_complex(
                            array_node, complex_len
                        )
                    else:
                        encoded_node = container.new_ckks_encode(array_node)
                else:
                    encoded_node = array_node
                air_value = AIRValue(
                    encoded_node, container, domain=self.current_domain
                )
                ir_args.append(air_value)
                continue

            # An explicit ranked annotation is authoritative for both the AIR
            # type and the wrapper metadata. Reject a conflicting runtime
            # descriptor instead of constructing an internally inconsistent
            # AIRValue.
            annotation_shape = tuple(
                getattr(arg_type, "_shape", ()) or ()
            )
            runtime_shape = None
            instance_name = None
            if runtime_arg is not None:
                # Check if it's a ciphertext/tensor instance with shape
                if hasattr(runtime_arg, 'shape'):
                    candidate_shape = runtime_arg.shape
                    try:
                        runtime_shape = tuple(candidate_shape)
                    except TypeError:
                        runtime_shape = None
                if hasattr(runtime_arg, 'name') and runtime_arg.name:
                    instance_name = runtime_arg.name
                    log().debug(f"Using instance name '{instance_name}' for parameter '{arg_name}'")
            if (annotation_shape and runtime_shape and
                    annotation_shape != runtime_shape):
                raise TypeError(
                    f"Runtime shape {runtime_shape} for '{arg_name}' does not "
                    f"match annotated shape {annotation_shape}"
                )
            shape = annotation_shape or runtime_shape

            # Create AIR parameter node
            # Map to AIR types based on domain, prefer runtime instance if provided
            air_type = None
            if arg_type is not None:
                try:
                    air_type = self._get_air_type_for_param(
                        arg_type, self.current_domain
                    )
                except Exception:
                    air_type = None
            if air_type is None and runtime_arg is not None:
                try:
                    air_type = python_type_to_air_type(runtime_arg, self.current_domain)
                except Exception:
                    air_type = None
            if air_type is None:
                air_type = self._get_air_type_for_param(arg_type, self.current_domain)
            param_node = func_scope.new_param(arg_name, air_type)
            formal_type = (
                param_node.rtype()
                if hasattr(param_node, "rtype")
                else air_type
            )
            if (
                not shape
                and formal_type is not None
                and formal_type.is_array()
            ):
                shape = tuple(formal_type.shape())
            
            # Wrap in AIRValue for operator overloading
            # Pass shape from instance if available
            air_value = _wrap_air_value(
                param_node, 
                container, 
                shape=shape,
                domain=self._domain_for_air_type(
                    formal_type, self.current_domain
                ),
                air_type=formal_type,
            )
            ir_args.append(air_value)
        
        return ir_args
    
    def _get_domain_type_name(self) -> str:
        """
        Get the type name for the current domain.
        
        This is used by new_func_with_type() to properly register domain-specific
        types like CIPHERTEXT for FHE domains.
        """
        domain = self.current_domain
        
        # Map domains to type names (matching acepy's domain_kernels.py)
        domain_type_map = {
            "fhe::sihe": "CIPHERTEXT",
            "fhe::ckks": "CIPHERTEXT",
            "fhe::poly": "polynomial",
            "nn::vector": "vector_tensor",
            "nn::core": "tensor",
            "air::core": "tensor",
        }
        
        return domain_type_map.get(domain, "tensor")
    
    def _get_air_type_for_param(self, param_type, domain):
        """
        Get AIR type for parameter based on domain.
        
        Uses python_type_to_air_type() to map Python abstract types to AIR types.
        This is ace_edsl's type mapping system - no MLIR needed.
        """
        if air_builder is None:
            return None
        
        # Use ace_edsl's type mapping system to convert Python types to AIR types
        if param_type is not None:
            air_type = python_type_to_air_type(param_type, domain=domain)
            if air_type is not None:
                return air_type
        
        # Default fallback if type mapping fails
        return air_builder.Type.make_array([64], air_builder.Type.make_float(32))
    
    def _build_gpu_module(self, attrs):
        """
        Build GPU module (required by BaseDSL abstract method).
        
        NOTE: This method exists because BaseDSL is designed for GPU-specific MLIR.
        We generate AIR for any target, so this method is a no-op.
        """
        # Not applicable for AIR, but required by abstract method
        if attrs is None:
            attrs = {}
        # TODO: Create AIR GPU module if needed (probably not needed)
        return None
    
    def _get_pipeline(self, pipeline: Optional[list] = None) -> list:
        """Return AIR pass pipeline based on current domain"""
        if pipeline is not None:
            return pipeline
        
        # Get domain from current context
        domain = self.current_domain or 'tensor'
        
        # Map legacy domain names to acepy-compatible names
        domain_map = {
            'tensor': 'air::core',  # Map tensor to air::core
        }
        domain = domain_map.get(domain, domain)
        
        # Look up pipeline for domain
        if domain in DOMAIN_PIPELINES:
            return DOMAIN_PIPELINES[domain]
        
        # Default pipeline (empty - no passes)
        return []
    
    def _get_globals(self):
        """Get globals for AST preprocessing (required by BaseDSL)"""
        caller_globals = self.frame.f_globals if self.frame else {}
        caller_locals = self.frame.f_locals if self.frame else {}
        all_globals = globals().copy()
        all_globals.update(caller_globals)
        all_globals.update(caller_locals)
        return all_globals
    
    def _generate_mlir_type_for_tensor_descriptor(self, tensor):
        """Required by BaseDSL - not used for AIR"""
        # Not applicable for AIR, but required by abstract method
        pass
    
    def _generate_executable_arg_for_tensor_descriptor(self, mlir_value=None, ptr_tensor_ty=None, tensor=None):
        """Required by BaseDSL - not used for AIR"""
        # Not applicable for AIR, but required by abstract method
        pass

# Backward compatibility alias
PyDSL = AceEDSL
