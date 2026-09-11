"""
ACE EDSL Pipeline - Lowering passes from AIR to C code.

This module encapsulates the full compilation pipeline:
    AIR (nn::core/nn::vector) → tensor2vector → vector2sihe → sihe2ckks → CKKS driver → Poly driver → poly2c → C code

Usage (Simple - matching acepy):
    from ace_edsl.edsl.pipeline import Pipeline
    
    result = (Pipeline("my_test")
        .load_onnx("model.onnx")
        .configure_fhe(scaling_factor_bits=56)
        .run())
    
    if result.success:
        print(f"Generated {len(result.c_code)} bytes of C code")

Usage (Manual Control):
    from ace_edsl.edsl.pipeline import AcePipeline
    
    pipeline = AcePipeline(glob_scope)
    pipeline.configure_fhe()  # Optional: customize FHE parameters
    c_code = pipeline.run()   # Run full pipeline, returns C code

Noise Budget Analysis:
    The sihe2ckks pass automatically analyzes noise budgets and inserts
    bootstrap operations where needed. Operations registered with Python
    lowerings (via register_lowering) can be skipped by C++ passes and
    handled in Python post-processing.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import time

from .passes.framework.pipeline_hooks import (
    HookExecutionResult,
    PassPipelineConfig,
    PipelinePoint,
    run_pipeline_hook,
)
from .passes.transition import tentative_native_inliner_barriers


def _pass_pipeline_config(config):
    if config is None:
        return PassPipelineConfig()
    if not isinstance(config, PassPipelineConfig):
        raise TypeError("pass_pipeline_config must be PassPipelineConfig")
    return config


class PolyLowering(str, Enum):
    """Compiler lowering strategy used by the CKKS-to-POLY driver."""

    SPOLY = "spoly"
    LINEAR_TRANSFORM = "linear_transform"


@dataclass
class FHEConfig:
    """FHE parameter configuration for CKKS/Poly lowering."""
    poly_degree: int = 0          # 0 = auto-determine
    mul_level: int = 0            # 0 = auto-determine
    input_level: int = 0          # 0 = runtime default/full input level
    security_level: int = 0       # 0 = auto-determine
    scaling_factor_bits: int = 56
    first_prime_bits: int = 60
    hamming_weight: int = 192
    
    # SIHE params
    relu_vr_def: float = 3.0
    relu_vr: str = ""
    
    # Source-codegen options. enable_poly is a deprecated compatibility alias
    # for codegen_ir and is resolved once in __post_init__.
    data_file: str = "data.msg"
    ct_encode: bool = False
    free_poly: bool = True
    provider: str = "ant"
    codegen_ir: Optional[str] = None
    enable_poly: Optional[bool] = None
    poly_lowering: str | PolyLowering = PolyLowering.SPOLY
    function_name_prefix: str = ""
    constant_name_prefix: str = ""
    pt_from_msg_name: str = "Pt_from_msg"
    raise_mod_level_func: str = ""
    context_manifest_file: str = ""
    resource_manifest_file: str = ""
    constant_manifest_file: str = ""
    # CPU POLY codegen experiment: 0 legacy, 1 serial, >1 limb thread budget.
    decomp_ntt_threads: int = 0
    evalmod_schedule: int = 0

    def __post_init__(self):
        self.provider = self.provider.strip().lower()
        if self.provider not in ("ant", "phantom"):
            raise ValueError("provider must be 'ant' or 'phantom'")

        selected = self.codegen_ir
        if selected is not None:
            selected = selected.strip().lower()
        if selected is None:
            selected = "poly" if self.enable_poly is not False else "ckks"
        if selected not in ("ckks", "poly"):
            raise ValueError("codegen_ir must be 'ckks' or 'poly'")
        if self.enable_poly is not None:
            compatibility_selection = "poly" if self.enable_poly else "ckks"
            if selected != compatibility_selection:
                raise ValueError("enable_poly conflicts with codegen_ir")
        if self.provider == "phantom" and selected != "ckks":
            raise ValueError("provider='phantom' requires codegen_ir='ckks'")

        self.codegen_ir = selected
        self.enable_poly = selected == "poly"

        poly_lowering = self.poly_lowering
        if isinstance(poly_lowering, PolyLowering):
            poly_lowering = poly_lowering.value
        elif isinstance(poly_lowering, str):
            poly_lowering = poly_lowering.strip().lower()
        else:
            raise TypeError("poly_lowering must be a PolyLowering or string")
        supported_poly_lowerings = {mode.value for mode in PolyLowering}
        if poly_lowering not in supported_poly_lowerings:
            raise ValueError(
                "poly_lowering must be 'spoly' or 'linear_transform'"
            )
        if (poly_lowering == PolyLowering.LINEAR_TRANSFORM.value and
                selected != "poly"):
            raise ValueError(
                "poly_lowering='linear_transform' requires codegen_ir='poly'"
            )
        self.poly_lowering = poly_lowering
        if (isinstance(self.decomp_ntt_threads, bool) or
                not isinstance(self.decomp_ntt_threads, int) or
                not 0 <= self.decomp_ntt_threads <= 0xFFFFFFFF):
            raise ValueError("decomp_ntt_threads must be a uint32 integer")
        if self.decomp_ntt_threads and (self.provider != "ant" or selected != "poly"):
            raise ValueError("decomp_ntt_threads requires ANT POLY codegen")
        if (isinstance(self.evalmod_schedule, bool) or not isinstance(self.evalmod_schedule, int)
                or self.evalmod_schedule not in (0, 1, 2)):
            raise ValueError("evalmod_schedule must be 0, 1 or 2")
        if self.evalmod_schedule and (self.provider != "ant" or selected != "poly" or
                self.poly_lowering != "linear_transform" or self.decomp_ntt_threads):
            raise ValueError("evalmod_schedule requires ANT linear_transform and decomp_ntt_threads=0")


@dataclass(frozen=True)
class VectorKernelLoweringConfig:
    plan_provider: str = "cpp"
    kernel_impl: str = "native"
    plan_kind: str = "auto"
    fallback: str = "error"
    mask_fuse: bool = False
    max_slots: int = 0
    conv_parallel: bool = False
    sharding: bool = False


_VECTOR_KERNEL_RECIPE_KINDS = frozenset(
    ("baseline-gemm", "baseline-conv", "fast-gemm", "fast-conv")
)
def _selected_vector_kernel_plan_provider(config, registered_provider):
    if config is None or config.plan_provider != "python":
        return None
    if registered_provider is not None:
        return registered_provider
    from .vector.planning import plan_vector_kernel

    return plan_vector_kernel


@dataclass
class PipelineResult:
    """Result of running the pipeline."""
    success: bool
    c_code: Optional[str] = None
    error: Optional[str] = None
    stages_completed: list = field(default_factory=list)
    air_dumps: Dict[str, str] = field(default_factory=dict)
    pass_results: Dict[PipelinePoint, HookExecutionResult] = field(
        default_factory=dict
    )


class AcePipeline:
    """
    ACE EDSL compilation pipeline.
    
    Runs lowering passes to transform AIR from high-level domains (nn::core, nn::vector)
    down to C code via FHE compilation.
    
    Pipeline stages:
        1. tensor2vector (nn::core → nn::vector)
        2. vector2sihe (nn::vector → fhe::sihe)
        3. CKKS driver (fhe::sihe → fhe::ckks)
        4. Poly driver (fhe::ckks → fhe::poly)
        5. poly2c (fhe::poly → C code)
    
    Example:
        @nn_kernel
        def my_kernel(a, b):
            return a + b
        
        my_kernel(x, y)
        
        pipeline = AcePipeline(dsl.current_air_module)
        result = pipeline.run()
        if result.success:
            print(result.c_code)
    """
    
    def __init__(
        self,
        glob_scope,
        fhe_config: Optional[FHEConfig] = None,
        pass_pipeline_config: Optional[PassPipelineConfig] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            glob_scope: The AIR GlobScope from AceEDSL.current_air_module
            fhe_config: Optional FHE configuration (uses defaults if not provided)
        """
        self.glob_scope = glob_scope
        self.fhe_config = fhe_config or FHEConfig()
        self._air_builder = None
        env_rewrite = os.environ.get("ACE_CKKS_PRIMITIVE_REWRITE", "").strip().lower()
        self._rewrite_ckks_extended_ops = env_rewrite in ("1", "true", "yes", "on")
        self.vector_kernel_config: Optional[VectorKernelLoweringConfig] = None
        self.vector_kernel_recipes: Dict[str, Callable] = {}
        self.vector_kernel_plan_provider: Optional[Callable] = None
        self.pass_pipeline_config = _pass_pipeline_config(pass_pipeline_config)
        
    def _get_air_builder(self):
        """Lazy import air_builder."""
        if self._air_builder is None:
            from ace_bindings import air_builder
            self._air_builder = air_builder
        return self._air_builder

    def set_ckks_extended_op_rewrite(self, enabled: bool = True) -> "AcePipeline":
        """
        Enable/disable primitive rewrite of CKKS extended ops.

        Controlled by default via env var `ACE_CKKS_PRIMITIVE_REWRITE`.
        """
        self._rewrite_ckks_extended_ops = bool(enabled)
        return self
    
    def configure_fhe(
        self,
        poly_degree: int = 0,
        mul_level: int = 0,
        input_level: int = 0,
        security_level: int = 0,
        scaling_factor_bits: int = 56,
        first_prime_bits: int = 60,
        hamming_weight: int = 192,
        data_file: str = "data.msg",
        ct_encode: bool = False,
        free_poly: bool = True,
        provider: str = "ant",
        codegen_ir: Optional[str] = None,
        enable_poly: Optional[bool] = None,
        poly_lowering: str | PolyLowering = PolyLowering.SPOLY,
        function_name_prefix: str = "",
        constant_name_prefix: str = "",
        pt_from_msg_name: str = "Pt_from_msg",
        raise_mod_level_func: str = "",
        context_manifest_file: str = "",
        resource_manifest_file: str = "",
        constant_manifest_file: str = "",
        decomp_ntt_threads: int = 0,
        evalmod_schedule: int = 0,
    ) -> "AcePipeline":
        """
        Configure FHE parameters.
        
        Args:
            poly_degree: Polynomial degree (0 = auto)
            mul_level: Multiplication level (0 = auto)
            input_level: CKKS input ciphertext level (0 = runtime default)
            security_level: Security level (0 = auto)
            scaling_factor_bits: CKKS scaling factor bits
            first_prime_bits: First prime bits (q0)
            hamming_weight: Secret key hamming weight
            data_file: Output data file name for poly2c
            ct_encode: Enable ciphertext encoding
            free_poly: Free polynomial after use
            enable_poly: Enable poly lowering (False = CKKS-level C for debugging)
            poly_lowering: CKKS-to-POLY strategy (`spoly` or experimental
                `linear_transform`)
            function_name_prefix: Prefix for generated C function symbols
            constant_name_prefix: Prefix for generated C constant symbols
            pt_from_msg_name: Plaintext data loader function name
            raise_mod_level_func: Runtime helper name for attributed raise_mod
            context_manifest_file: Optional generated context attestation path
            resource_manifest_file: Optional generated resource attestation path
            constant_manifest_file: Optional generated constant attestation path
            
        Returns:
            self (for method chaining)
        """
        self.fhe_config = FHEConfig(
            poly_degree=poly_degree,
            mul_level=mul_level,
            input_level=input_level,
            security_level=security_level,
            scaling_factor_bits=scaling_factor_bits,
            first_prime_bits=first_prime_bits,
            hamming_weight=hamming_weight,
            data_file=data_file,
            ct_encode=ct_encode,
            free_poly=free_poly,
            provider=provider,
            codegen_ir=codegen_ir,
            enable_poly=enable_poly,
            poly_lowering=poly_lowering,
            function_name_prefix=function_name_prefix,
            constant_name_prefix=constant_name_prefix,
            pt_from_msg_name=pt_from_msg_name,
            raise_mod_level_func=raise_mod_level_func,
            context_manifest_file=context_manifest_file,
            resource_manifest_file=resource_manifest_file,
            constant_manifest_file=constant_manifest_file,
            decomp_ntt_threads=decomp_ntt_threads,
            evalmod_schedule=evalmod_schedule,
        )
        return self

    def configure_vector_kernel_lowering(
        self,
        plan_provider: str = "cpp",
        kernel_impl: str = "native",
        plan_kind: str = "auto",
        fallback: str = "error",
        mask_fuse: bool = False,
        max_slots: int = 0,
        conv_parallel: bool = False,
        sharding: bool = False,
    ) -> "AcePipeline":
        self.vector_kernel_config = VectorKernelLoweringConfig(
            plan_provider, kernel_impl, plan_kind, fallback, mask_fuse,
            max_slots, conv_parallel, sharding,
        )
        return self

    def register_vector_kernel_recipe(
        self, plan_kind: str, recipe: Callable
    ) -> "AcePipeline":
        if plan_kind not in _VECTOR_KERNEL_RECIPE_KINDS:
            raise ValueError(
                "vector-kernel recipe registration supports baseline-gemm, "
                "baseline-conv, fast-gemm, and fast-conv"
            )
        if not callable(recipe):
            raise TypeError("vector-kernel recipe must be callable")
        if plan_kind in self.vector_kernel_recipes:
            raise ValueError(f"duplicate vector-kernel recipe: {plan_kind}")
        self.vector_kernel_recipes[plan_kind] = recipe
        return self

    def register_vector_kernel_plan_provider(self, provider: Callable) -> "AcePipeline":
        if not callable(provider):
            raise TypeError("vector-kernel plan provider must be callable")
        if self.vector_kernel_plan_provider is not None:
            raise ValueError("a vector-kernel plan provider is already registered")
        self.vector_kernel_plan_provider = provider
        return self

    def set_pass_pipeline_config(
        self, config: Optional[PassPipelineConfig]
    ) -> "AcePipeline":
        self.pass_pipeline_config = _pass_pipeline_config(config)
        return self
    
    def dump_air(self, stage_name: str) -> str:
        """Dump current AIR state."""
        if self.glob_scope is None:
            return ""
        return self.glob_scope.dump()
    
    def run_tensor2vector(self) -> bool:
        """
        Run tensor2vector pass (nn::core → nn::vector).
        
        Returns:
            True if successful
        """
        if self.glob_scope is None:
            return False
        try:
            if self.vector_kernel_config is None:
                self.glob_scope.run_cpp_pass("tensor2vector", [])
            else:
                config = self.vector_kernel_config
                self.glob_scope.run_tensor2vector_with_recipes(
                    [], config.plan_provider, config.kernel_impl,
                    config.plan_kind, config.fallback,
                    dict(self.vector_kernel_recipes),
                    config.mask_fuse, config.max_slots,
                    config.conv_parallel, config.sharding,
                    _selected_vector_kernel_plan_provider(
                        config, self.vector_kernel_plan_provider
                    ),
                )
            return True
        except Exception as e:
            print(f"tensor2vector failed: {e}")
            return False
    
    def run_vector2sihe(self, skip_ops: Optional[List[str]] = None) -> bool:
        """
        Run vector2sihe pass (nn::vector → fhe::sihe).
        
        Args:
            skip_ops: Operations to skip during lowering
        
        Returns:
            True if successful
        """
        if self.glob_scope is None:
            return False
        try:
            self.glob_scope.run_cpp_pass("vector2sihe", skip_ops or [])
            return True
        except Exception as e:
            print(f"vector2sihe failed: {e}")
            return False
    
    def run_sihe2ckks(self, skip_ops: Optional[List[str]] = None) -> bool:
        """
        Run sihe2ckks pass (fhe::sihe → fhe::ckks).
        
        This pass performs noise budget analysis and inserts bootstrap
        operations where the noise budget is exhausted.
        
        Args:
            skip_ops: Operations to skip (e.g., bootstrap for Python lowering)
        
        Returns:
            True if successful
        """
        if self.glob_scope is None:
            return False
        try:
            self.glob_scope.run_cpp_pass("sihe2ckks", skip_ops or [])
            return True
        except Exception as e:
            print(f"sihe2ckks failed: {e}")
            return False
    
    def run_ckks_driver(self) -> Dict[str, Any]:
        """
        Run CKKS driver (fhe::sihe → fhe::ckks).
        
        Returns:
            Dict with 'success' and 'message' keys
        """
        if self.glob_scope is None:
            return {"success": False, "message": "No AIR module"}
        
        # Apply FHE configuration
        self.glob_scope.configure_fhe_params(
            poly_degree=self.fhe_config.poly_degree,
            mul_level=self.fhe_config.mul_level,
            input_level=self.fhe_config.input_level,
            security_level=self.fhe_config.security_level,
            scaling_factor_bits=self.fhe_config.scaling_factor_bits,
            first_prime_bits=self.fhe_config.first_prime_bits,
            hamming_weight=self.fhe_config.hamming_weight,
        )
        
        air_builder = self._get_air_builder()
        return air_builder.run_ckks_driver(self.glob_scope)
    
    def run_poly_driver(self) -> Dict[str, Any]:
        """
        Run Poly driver (fhe::ckks → fhe::poly).
        
        Returns:
            Dict with 'success' and 'message' keys
        """
        if self.glob_scope is None:
            return {"success": False, "message": "No AIR module"}
        
        air_builder = self._get_air_builder()
        return air_builder.run_poly_driver(
            self.glob_scope, self.fhe_config.poly_lowering
        )
    
    def run_poly2c(self) -> Optional[str]:
        """
        Run poly2c (fhe::poly → C code).
        
        Returns:
            Generated C code string, or None on failure
        """
        if self.glob_scope is None:
            return None

        if self.fhe_config.codegen_ir == "ckks":
            return self.run_ckks2c()
        
        if not hasattr(self.glob_scope, "run_poly2c"):
            return None
        
        # Use configured data_file (empty string keeps constants inline).
        data_file = self.fhe_config.data_file
        
        ok = self.glob_scope.run_poly2c(
            data_file=data_file,
            ct_encode=self.fhe_config.ct_encode,
            free_poly=self.fhe_config.free_poly,
            enable_poly=True,
            function_name_prefix=self.fhe_config.function_name_prefix,
            constant_name_prefix=self.fhe_config.constant_name_prefix,
            pt_from_msg_name=self.fhe_config.pt_from_msg_name,
            raise_mod_level_func=self.fhe_config.raise_mod_level_func,
            **({"decomp_ntt_threads": self.fhe_config.decomp_ntt_threads}
               if self.fhe_config.decomp_ntt_threads else {}),
            **({"evalmod_schedule": self.fhe_config.evalmod_schedule} if self.fhe_config.evalmod_schedule else {}),
        )
        
        if ok and hasattr(self.glob_scope, "get_c_code"):
            return self.glob_scope.get_c_code()
        return None

    def run_ckks2c(self) -> Optional[str]:
        """Run the dedicated post-CKKS AIR source emitter."""
        if self.glob_scope is None or not hasattr(self.glob_scope, "run_ckks2c"):
            return None

        ok = self.glob_scope.run_ckks2c(
            data_file=self.fhe_config.data_file,
            ct_encode=self.fhe_config.ct_encode,
            free_poly=self.fhe_config.free_poly,
            provider=self.fhe_config.provider,
            function_name_prefix=self.fhe_config.function_name_prefix,
            constant_name_prefix=self.fhe_config.constant_name_prefix,
            pt_from_msg_name=self.fhe_config.pt_from_msg_name,
            raise_mod_level_func=self.fhe_config.raise_mod_level_func,
            context_manifest_file=self.fhe_config.context_manifest_file,
            resource_manifest_file=self.fhe_config.resource_manifest_file,
            constant_manifest_file=self.fhe_config.constant_manifest_file,
        )
        if ok and hasattr(self.glob_scope, "get_c_code"):
            return self.glob_scope.get_c_code()
        return None
    
    def run(
        self,
        start_domain: str = "nn::core",
        dump_stages: bool = False,
        verbose: bool = True,
        skip_ops: Optional[List[str]] = None,
        python_lowering_func: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline from AIR to C code.
        
        Args:
            start_domain: Starting domain ("nn::core", "nn::vector", "fhe::sihe", "fhe::ckks")
            dump_stages: If True, collect AIR dumps at each stage
            verbose: If True, print progress messages
            skip_ops: Operations to skip in C++ passes (for Python lowering)
            python_lowering_func: Function to run after sihe2ckks for Python lowerings
            
        Returns:
            PipelineResult with success status, C code, and optional AIR dumps
        """
        result = PipelineResult(success=False)
        
        if self.glob_scope is None:
            result.error = "No AIR module available"
            return result
        
        skip_ops = skip_ops or []
        
        def log(msg: str):
            if verbose:
                print(msg)
        
        try:
            # Stage 1: tensor2vector (skip if starting at nn::vector or later)
            if start_domain == "nn::core":
                log("Running tensor2vector (nn::core → nn::vector)...")
                if not self.run_tensor2vector():
                    result.error = "tensor2vector failed"
                    return result
                result.stages_completed.append("tensor2vector")
                if dump_stages:
                    result.air_dumps["tensor2vector"] = self.dump_air("tensor2vector")
            
            # Stage 2: vector2sihe (skip if starting at fhe::sihe or later)
            if start_domain in ("nn::core", "nn::vector"):
                hook_result = run_pipeline_hook(
                    self.glob_scope,
                    PipelinePoint.BEFORE_VECTOR2SIHE,
                    self.pass_pipeline_config,
                    native_barriers=tentative_native_inliner_barriers(
                        self.vector_kernel_config
                    ),
                )
                result.pass_results[PipelinePoint.BEFORE_VECTOR2SIHE] = hook_result
                if not hook_result.success:
                    result.error = "before-vector2sihe pass hook failed: " + "; ".join(
                        diagnostic.message for diagnostic in hook_result.diagnostics
                    )
                    return result
                if hook_result.barrier_records:
                    result.stages_completed.append("vector_kernel_inline")
                    if dump_stages:
                        result.air_dumps["vector_kernel_inline"] = self.dump_air(
                            "vector_kernel_inline"
                        )

                log("Running vector2sihe (nn::vector → fhe::sihe)...")
                if not self.run_vector2sihe(skip_ops):
                    result.error = "vector2sihe failed"
                    return result
                result.stages_completed.append("vector2sihe")
                if dump_stages:
                    result.air_dumps["vector2sihe"] = self.dump_air("vector2sihe")
            
            # Stage 3: sihe2ckks (performs noise analysis, inserts bootstrap)
            if start_domain in ("nn::core", "nn::vector", "fhe::sihe"):
                log("Running sihe2ckks (fhe::sihe → fhe::ckks, inserts bootstrap)...")
                if not self.run_sihe2ckks(skip_ops):
                    result.error = "sihe2ckks failed"
                    return result
                result.stages_completed.append("sihe2ckks")
                if dump_stages:
                    result.air_dumps["sihe2ckks"] = self.dump_air("sihe2ckks")
                
                # Run Python lowering after sihe2ckks if provided
                if python_lowering_func:
                    log("Running Python lowering pass...")
                    python_lowering_func(self.glob_scope)
                    result.stages_completed.append("python_lower")
                    if dump_stages:
                        result.air_dumps["python_lower"] = self.dump_air("python_lower")
            
            # Stage 4: CKKS driver (scale management)
            if (self.fhe_config.provider == "phantom" and
                    self.fhe_config.codegen_ir == "ckks" and
                    self._rewrite_ckks_extended_ops):
                result.error = (
                    "rewrite_ckks_extended_ops is prohibited for the Phantom "
                    "CKKS2C backend"
                )
                return result
            if self._rewrite_ckks_extended_ops:
                log("Running CKKS extended-op rewrite (primitive lowering)...")
                from .passes.ckks_extended_ops_rewrite import (
                    rewrite_extended_ckks_ops_to_primitives,
                )

                rewrite_result = rewrite_extended_ckks_ops_to_primitives(
                    self.glob_scope, verbose=verbose
                )
                if not rewrite_result.get("success", False):
                    errs = "; ".join(rewrite_result.get("errors", []))
                    result.error = f"CKKS extended-op rewrite failed: {errs}"
                    return result
                result.stages_completed.append("ckks_extended_rewrite")
                if dump_stages:
                    result.air_dumps["ckks_extended_rewrite"] = self.dump_air(
                        "ckks_extended_rewrite"
                    )

            log("Running CKKS driver (scale management)...")
            ckks_result = self.run_ckks_driver()
            if not ckks_result.get("success"):
                result.error = f"CKKS driver failed: {ckks_result.get('message')}"
                return result
            result.stages_completed.append("ckks_driver")
            if dump_stages:
                result.air_dumps["ckks_driver"] = self.dump_air("ckks_driver")
            
            # Exactly one terminal path is selected by codegen_ir.
            if self.fhe_config.codegen_ir == "poly":
                log("Running Poly driver (fhe::ckks → fhe::poly)...")
                poly_result = self.run_poly_driver()
                if not poly_result.get("success"):
                    result.error = f"Poly driver failed: {poly_result.get('message')}"
                    return result
                result.stages_completed.append("poly_driver")
                if dump_stages:
                    result.air_dumps["poly_driver"] = self.dump_air("poly_driver")
                log("Running poly2c (fhe::poly → C code)...")
                c_code = self.run_poly2c()
                terminal = "poly2c"
            else:
                log("Running ckks2c (fhe::ckks → C code)...")
                c_code = self.run_ckks2c()
                terminal = "ckks2c"
            if c_code is None:
                result.error = f"{terminal} failed to generate source"
                return result
            result.stages_completed.append(terminal)
            if dump_stages:
                result.air_dumps[terminal] = self.dump_air(terminal)
            
            result.success = True
            result.c_code = c_code
            log(f"✓ Pipeline completed successfully ({len(c_code)} bytes of C code)")
            
        except Exception as e:
            result.error = str(e)
            
        return result
    
    def run_to_c(self, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Convenience method to run pipeline and optionally write C code to file.
        
        Args:
            output_path: If provided, write C code to this file
            **kwargs: Additional arguments passed to run()
            
        Returns:
            Generated C code string, or None on failure
        """
        result = self.run(**kwargs)
        
        if not result.success:
            print(f"Pipeline failed: {result.error}")
            return None
        
        if output_path and result.c_code:
            import os
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result.c_code)
            print(f"✓ C code written to: {output_path}")
        
        return result.c_code


# Convenience function for quick pipeline execution
def compile_to_c(
    glob_scope,
    output_path: Optional[str] = None,
    start_domain: str = "nn::core",
    fhe_config: Optional[FHEConfig] = None,
    verbose: bool = True,
) -> Optional[str]:
    """
    Convenience function to compile AIR to C code.
    
    Args:
        glob_scope: AIR GlobScope from AceEDSL.current_air_module
        output_path: Optional path to write C code
        start_domain: Starting domain ("nn::core", "nn::vector", "fhe::sihe")
        fhe_config: Optional FHE configuration
        verbose: Print progress messages
        
    Returns:
        Generated C code, or None on failure
        
    Example:
        from ace_edsl.edsl.pipeline import compile_to_c
        
        @nn_kernel
        def my_kernel(a, b):
            return a + b
        
        my_kernel(x, y)
        c_code = compile_to_c(dsl.current_air_module, "output.c")
    """
    pipeline = AcePipeline(glob_scope, fhe_config)
    return pipeline.run_to_c(output_path, start_domain=start_domain, verbose=verbose)


# =============================================================================
# Pipeline Target Enum (matching acepy)
# =============================================================================

class PipelineTarget(Enum):
    """Target level for pipeline execution."""
    INITIAL = "initial"        # Just load, no passes
    TENSOR2VECTOR = "tensor2vector"  # Stop after tensor2vector
    VECTOR2SIHE = "vector2sihe"      # Stop after vector2sihe
    SIHE2CKKS = "sihe2ckks"          # Stop after sihe2ckks
    CKKS_DRIVER = "ckks_driver"      # Stop after CKKS driver
    POLY_DRIVER = "poly_driver"      # Stop after Poly driver
    C = "c"                          # Full pipeline to C code


# =============================================================================
# Pipeline Class (matching acepy's dump_utils.Pipeline)
# =============================================================================

class Pipeline:
    """
    High-level compilation pipeline matching acepy's Pipeline class.
    
    This class provides a fluent API for FHE compilation with:
    - ONNX model loading
    - FHE parameter configuration
    - Skip ops for Python lowerings
    - Python lowering callbacks
    - IR dumping at each phase
    
    Example:
        from ace_edsl.edsl.pipeline import Pipeline, PipelineTarget
        
        result = (Pipeline("resnet20_test")
            .load_onnx("resnet20.onnx")
            .configure_fhe(scaling_factor_bits=56)
            .set_skip_ops(["fhe::ckks::bootstrap"])  # Skip bootstrap in C++
            .set_python_lowering(my_lowering_func)   # Handle in Python
            .run(target=PipelineTarget.C))
        
        if result.success:
            with open("output.c", "w") as f:
                f.write(result.c_code)
    """
    
    # Standard pipeline phases
    PHASES = [
        "tensor2vector",
        "vector2sihe", 
        "sihe2ckks",
        "ckks_driver",
        "poly_driver",
        "poly2c"
    ]
    
    # Map targets to phases needed
    TARGET_PHASES = {
        PipelineTarget.INITIAL: [],
        PipelineTarget.TENSOR2VECTOR: ["tensor2vector"],
        PipelineTarget.VECTOR2SIHE: ["tensor2vector", "vector2sihe"],
        PipelineTarget.SIHE2CKKS: ["tensor2vector", "vector2sihe", "sihe2ckks"],
        PipelineTarget.CKKS_DRIVER: ["tensor2vector", "vector2sihe", "sihe2ckks", "ckks_driver"],
        PipelineTarget.POLY_DRIVER: ["tensor2vector", "vector2sihe", "sihe2ckks", "ckks_driver", "poly_driver"],
        PipelineTarget.C: PHASES,
    }
    
    def __init__(
        self,
        name: str = "pipeline",
        output_dir: Optional[str] = None,
        dump_ir: bool = True,
        verbose: bool = True,
        on_phase_complete: Optional[Callable[[str, str], None]] = None,
        pass_pipeline_config: Optional[PassPipelineConfig] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            name: Name for output files and logging
            output_dir: Directory for output files (default: ./output/{name}/)
            dump_ir: Whether to dump IR at each phase
            verbose: Print progress messages
            on_phase_complete: Callback(phase_name, ir_dump) after each phase
        """
        self.name = name
        self.verbose = verbose
        self.dump_ir = dump_ir
        self.on_phase_complete = on_phase_complete
        
        # Output directory
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "..", "output", name)
        self.output_dir = output_dir
        
        # State
        self.glob = None
        self.config = FHEConfig()
        self.skip_ops: List[str] = []
        self.python_lowering_func: Optional[Callable] = None
        self.vector_kernel_config: Optional[VectorKernelLoweringConfig] = None
        self.vector_kernel_recipes: Dict[str, Callable] = {}
        self.vector_kernel_plan_provider: Optional[Callable] = None
        self.pass_pipeline_config = _pass_pipeline_config(pass_pipeline_config)
        env_rewrite = os.environ.get("ACE_CKKS_PRIMITIVE_REWRITE", "").strip().lower()
        self.rewrite_ckks_extended_ops: bool = env_rewrite in ("1", "true", "yes", "on")
        
        # Results
        self.phase_irs: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}

    def set_pass_pipeline_config(
        self, config: Optional[PassPipelineConfig]
    ) -> "Pipeline":
        self.pass_pipeline_config = _pass_pipeline_config(config)
        return self
        
    def load_onnx(self, model_path: str) -> "Pipeline":
        """
        Load an ONNX model.
        
        Args:
            model_path: Path to .onnx file
            
        Returns:
            self for chaining
        """
        from ace_bindings import air_builder
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if self.verbose:
            print(f"[Pipeline] Loading ONNX model: {model_path}")
        
        t0 = time.time()
        result = air_builder.load_onnx_model(model_path)
        load_time = time.time() - t0
        
        if not result["success"]:
            raise RuntimeError(f"Failed to load ONNX model: {result['message']}")
        
        self.glob = result["glob_scope"]
        
        if self.verbose:
            print(f"  ✓ Loaded ({load_time:.2f}s)")
        
        return self
    
    def set_glob(self, glob) -> "Pipeline":
        """
        Set the GlobScope directly (for programmatic IR).
        
        Args:
            glob: GlobScope object
            
        Returns:
            self for chaining
        """
        self.glob = glob
        return self
    
    def configure_fhe(
        self,
        poly_degree: int = 0,
        mul_level: int = 0,
        input_level: int = 0,
        security_level: int = 0,
        scaling_factor_bits: int = 56,
        first_prime_bits: int = 60,
        hamming_weight: int = 192,
        relu_vr_def: float = 3.0,
        relu_vr: str = "",
        data_file: str = "data.msg",
        ct_encode: bool = False,
        free_poly: bool = True,
        provider: str = "ant",
        codegen_ir: Optional[str] = None,
        enable_poly: Optional[bool] = None,
        poly_lowering: str | PolyLowering = PolyLowering.SPOLY,
        context_manifest_file: str = "",
        resource_manifest_file: str = "",
        constant_manifest_file: str = "",
        decomp_ntt_threads: int = 0,
        evalmod_schedule: int = 0,
    ) -> "Pipeline":
        """
        Configure FHE parameters.
        
        Args:
            poly_degree: Polynomial degree (0 = auto)
            mul_level: Multiplication level (0 = auto)
            input_level: CKKS input ciphertext level (0 = runtime default)
            security_level: Security level (0 = auto)
            scaling_factor_bits: CKKS scaling factor bits
            first_prime_bits: First prime bits
            hamming_weight: Secret key hamming weight
            relu_vr_def: Default ReLU value range (SIHE option)
            relu_vr: Per-layer ReLU value ranges (SIHE option)
            data_file: Output data file for poly2c
            ct_encode: Enable ciphertext encoding
            free_poly: Free polynomials after use
            enable_poly: Enable poly lowering (False = CKKS-level C for debugging)
            poly_lowering: CKKS-to-POLY strategy (`spoly` or experimental
                `linear_transform`)
            
        Returns:
            self for chaining
        """
        self.config = FHEConfig(
            poly_degree=poly_degree,
            mul_level=mul_level,
            input_level=input_level,
            security_level=security_level,
            scaling_factor_bits=scaling_factor_bits,
            first_prime_bits=first_prime_bits,
            hamming_weight=hamming_weight,
            relu_vr_def=relu_vr_def,
            relu_vr=relu_vr,
            data_file=data_file,
            ct_encode=ct_encode,
            free_poly=free_poly,
            provider=provider,
            codegen_ir=codegen_ir,
            enable_poly=enable_poly,
            poly_lowering=poly_lowering,
            context_manifest_file=context_manifest_file,
            resource_manifest_file=resource_manifest_file,
            constant_manifest_file=constant_manifest_file,
            decomp_ntt_threads=decomp_ntt_threads,
            evalmod_schedule=evalmod_schedule,
        )
        return self
    
    def set_skip_ops(self, ops: List[str]) -> "Pipeline":
        """
        Set operations to skip during C++ lowering.
        
        These ops will be preserved by C++ passes and can be lowered
        via Python post-processing (set_python_lowering).
        
        Args:
            ops: List of op names to skip (e.g., ["fhe::ckks::bootstrap"])
            
        Returns:
            self for chaining
        """
        self.skip_ops = ops
        return self
    
    def set_python_lowering(self, func: Callable) -> "Pipeline":
        """
        Set a Python lowering function to run after sihe2ckks.
        
        The function receives the glob_scope and should apply Python
        lowerings for skipped ops.
        
        Args:
            func: Function(glob_scope) that applies Python lowerings
            
        Returns:
            self for chaining
        """
        self.python_lowering_func = func
        return self

    def configure_vector_kernel_lowering(
        self,
        plan_provider: str = "cpp",
        kernel_impl: str = "native",
        plan_kind: str = "auto",
        fallback: str = "error",
        mask_fuse: bool = False,
        max_slots: int = 0,
        conv_parallel: bool = False,
        sharding: bool = False,
    ) -> "Pipeline":
        self.vector_kernel_config = VectorKernelLoweringConfig(
            plan_provider, kernel_impl, plan_kind, fallback, mask_fuse,
            max_slots, conv_parallel, sharding,
        )
        return self

    def register_vector_kernel_recipe(
        self, plan_kind: str, recipe: Callable
    ) -> "Pipeline":
        if plan_kind not in _VECTOR_KERNEL_RECIPE_KINDS:
            raise ValueError(
                "vector-kernel recipe registration supports baseline-gemm, "
                "baseline-conv, fast-gemm, and fast-conv"
            )
        if not callable(recipe):
            raise TypeError("vector-kernel recipe must be callable")
        if plan_kind in self.vector_kernel_recipes:
            raise ValueError(f"duplicate vector-kernel recipe: {plan_kind}")
        self.vector_kernel_recipes[plan_kind] = recipe
        return self

    def register_vector_kernel_plan_provider(self, provider: Callable) -> "Pipeline":
        if not callable(provider):
            raise TypeError("vector-kernel plan provider must be callable")
        if self.vector_kernel_plan_provider is not None:
            raise ValueError("a vector-kernel plan provider is already registered")
        self.vector_kernel_plan_provider = provider
        return self

    def set_ckks_extended_op_rewrite(self, enabled: bool = True) -> "Pipeline":
        """Enable/disable primitive rewrite of CKKS extended ops."""
        self.rewrite_ckks_extended_ops = bool(enabled)
        return self
    
    def get_ir(self) -> str:
        """Get current IR dump."""
        if self.glob:
            return self.glob.dump()
        return ""
    
    def _run_phase(self, phase: str) -> bool:
        """Run a single pipeline phase."""
        from ace_bindings import air_builder
        
        if phase == "tensor2vector":
            # Configure SIHE params before tensor2vector
            if hasattr(self.glob, 'configure_sihe_params'):
                if self.config.relu_vr:
                    self.glob.configure_sihe_params(
                        relu_vr_def=self.config.relu_vr_def,
                        relu_vr=self.config.relu_vr
                    )
                else:
                    self.glob.configure_sihe_params(
                        relu_vr_def=self.config.relu_vr_def
                    )
            
            if self.vector_kernel_config is None:
                return self.glob.run_cpp_pass("tensor2vector", self.skip_ops)
            config = self.vector_kernel_config
            return self.glob.run_tensor2vector_with_recipes(
                self.skip_ops, config.plan_provider, config.kernel_impl,
                config.plan_kind, config.fallback,
                dict(self.vector_kernel_recipes),
                config.mask_fuse, config.max_slots,
                config.conv_parallel, config.sharding,
                _selected_vector_kernel_plan_provider(
                    config, self.vector_kernel_plan_provider
                ),
            )
        
        elif phase == "vector2sihe":
            return self.glob.run_cpp_pass("vector2sihe", self.skip_ops)
        
        elif phase == "sihe2ckks":
            # sihe2ckks handles noise budget analysis and inserts bootstrap
            success = self.glob.run_cpp_pass("sihe2ckks", self.skip_ops)
            
            # Run Python lowering after sihe2ckks if configured
            if success and self.python_lowering_func:
                if self.verbose:
                    print("  [Python Lowering] Running Python lowering pass...")
                self.python_lowering_func(self.glob)
            
            return success
        
        elif phase == "ckks_driver":
            if (self.config.provider == "phantom" and
                    self.config.codegen_ir == "ckks" and
                    self.rewrite_ckks_extended_ops):
                raise RuntimeError(
                    "rewrite_ckks_extended_ops is prohibited for the Phantom "
                    "CKKS2C backend"
                )
            if self.rewrite_ckks_extended_ops:
                from .passes.ckks_extended_ops_rewrite import (
                    rewrite_extended_ckks_ops_to_primitives,
                )
                rewrite_result = rewrite_extended_ckks_ops_to_primitives(
                    self.glob, verbose=self.verbose
                )
                if not rewrite_result.get("success", False):
                    if self.verbose:
                        print(
                            "  [CKKS Rewrite] Failed: "
                            + "; ".join(rewrite_result.get("errors", []))
                        )
                    return False

            # Apply FHE config
            if hasattr(self.glob, 'configure_fhe_params'):
                self.glob.configure_fhe_params(
                    poly_degree=self.config.poly_degree,
                    mul_level=self.config.mul_level,
                    input_level=self.config.input_level,
                    security_level=self.config.security_level,
                    scaling_factor_bits=self.config.scaling_factor_bits,
                    first_prime_bits=self.config.first_prime_bits,
                    hamming_weight=self.config.hamming_weight,
                )
            result = air_builder.run_ckks_driver(self.glob)
            return result.get("success", False)
        
        elif phase == "poly_driver":
            if self.config.codegen_ir != "poly":
                raise RuntimeError("poly_driver is not selected by codegen_ir")
            result = air_builder.run_poly_driver(
                self.glob, self.config.poly_lowering
            )
            return result.get("success", False)
        
        elif phase == "poly2c":
            if hasattr(self.glob, "run_poly2c"):
                return self.glob.run_poly2c(
                    data_file=self.config.data_file,
                    ct_encode=self.config.ct_encode,
                    free_poly=self.config.free_poly,
                    enable_poly=True,
                    **({"decomp_ntt_threads": self.config.decomp_ntt_threads}
                       if self.config.decomp_ntt_threads else {}),
                    **({"evalmod_schedule": self.config.evalmod_schedule} if self.config.evalmod_schedule else {}),
                )
            return False

        elif phase == "ckks2c":
            if hasattr(self.glob, "run_ckks2c"):
                return self.glob.run_ckks2c(
                    data_file=self.config.data_file,
                    ct_encode=self.config.ct_encode,
                    free_poly=self.config.free_poly,
                    provider=self.config.provider,
                    function_name_prefix=self.config.function_name_prefix,
                    constant_name_prefix=self.config.constant_name_prefix,
                    pt_from_msg_name=self.config.pt_from_msg_name,
                    raise_mod_level_func=self.config.raise_mod_level_func,
                    context_manifest_file=self.config.context_manifest_file,
                    resource_manifest_file=self.config.resource_manifest_file,
                    constant_manifest_file=self.config.constant_manifest_file,
                )
            return False
        
        return False
    
    def run(
        self,
        target: PipelineTarget = PipelineTarget.C,
        phases: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Run the compilation pipeline.
        
        Args:
            target: Target level to compile to (default: C)
            phases: Override automatic phase selection
            
        Returns:
            PipelineResult with success status, C code, etc.
        """
        if self.glob is None:
            return PipelineResult(
                success=False,
                error="No model loaded. Call load_onnx() or set_glob() first."
            )
        
        result = PipelineResult(success=True)
        
        # Determine phases to run
        if phases is None:
            phases = list(self.TARGET_PHASES.get(target, self.PHASES))
            if target == PipelineTarget.C and self.config.codegen_ir == "ckks":
                phases = phases[: phases.index("poly_driver")] + ["ckks2c"]
        terminal_phases = [phase for phase in phases
                           if phase in ("ckks2c", "poly2c")]
        if len(terminal_phases) > 1:
            return PipelineResult(
                success=False, error="pipeline selects more than one terminal pass"
            )
        if terminal_phases and terminal_phases[0] != self.config.codegen_ir + "2c":
            return PipelineResult(
                success=False,
                error="terminal phase conflicts with codegen_ir",
            )
        
        if self.verbose:
            print(f"[Pipeline] Running {len(phases)} phases: {phases}")
            if self.skip_ops:
                print(f"[Pipeline] Skip ops: {self.skip_ops}")
        
        # Dump initial state
        if self.dump_ir:
            self.phase_irs["initial"] = self.glob.dump()
            self._save_ir("initial")
        if self.on_phase_complete:
            self.on_phase_complete("initial", self.glob.dump())
        
        # Run each phase
        try:
            for phase in phases:
                if phase == "vector2sihe":
                    hook_result = run_pipeline_hook(
                        self.glob,
                        PipelinePoint.BEFORE_VECTOR2SIHE,
                        self.pass_pipeline_config,
                        native_barriers=tentative_native_inliner_barriers(
                            self.vector_kernel_config
                        ),
                    )
                    result.pass_results[
                        PipelinePoint.BEFORE_VECTOR2SIHE
                    ] = hook_result
                    if not hook_result.success:
                        result.success = False
                        result.error = (
                            "Phase before-vector2sihe pass hook failed: "
                            + "; ".join(
                                diagnostic.message
                                for diagnostic in hook_result.diagnostics
                            )
                        )
                        return result
                    if hook_result.barrier_records:
                        result.stages_completed.append("vector_kernel_inline")
                        if self.dump_ir:
                            self.phase_irs["vector_kernel_inline"] = self.glob.dump()
                            self._save_ir("vector_kernel_inline")
                        if self.on_phase_complete:
                            self.on_phase_complete(
                                "vector_kernel_inline", self.glob.dump()
                            )

                if self.verbose:
                    print(f"[Pipeline] Running {phase}...")
                
                t0 = time.time()
                success = self._run_phase(phase)
                elapsed = time.time() - t0
                
                self.timings[phase] = elapsed
                
                if not success:
                    result.success = False
                    result.error = f"Phase {phase} failed"
                    if self.verbose:
                        print(f"  ✗ {phase} failed ({elapsed:.2f}s)")
                    return result
                
                result.stages_completed.append(phase)
                
                if self.verbose:
                    print(f"  ✓ {phase} ({elapsed:.2f}s)")
                
                # Dump IR
                if self.dump_ir:
                    self.phase_irs[phase] = self.glob.dump()
                    self._save_ir(phase)
                if self.on_phase_complete:
                    self.on_phase_complete(phase, self.glob.dump())
            
            # Get source code if one terminal pass was run.
            if terminal_phases and hasattr(self.glob, "get_c_code"):
                result.c_code = self.glob.get_c_code()
            
            if self.verbose:
                if result.c_code:
                    print(f"[Pipeline] ✓ Generated {len(result.c_code)} bytes of C code")
                else:
                    print(f"[Pipeline] ✓ Completed {len(phases)} phases")
        
        except Exception as e:
            result.success = False
            result.error = str(e)
            if self.verbose:
                print(f"[Pipeline] ✗ Error: {e}")
        
        return result
    
    def _save_ir(self, phase: str):
        """Save IR dump to file."""
        if not self.dump_ir or phase not in self.phase_irs:
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f"{self.name}_air_{phase}.txt")
        
        with open(filepath, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Phase: {phase}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(self.phase_irs[phase])
    
    def print_summary(self):
        """Print summary of pipeline execution."""
        print(f"\n{'=' * 60}")
        print(f"Pipeline Summary: {self.name}")
        print(f"{'=' * 60}")
        
        total_time = sum(self.timings.values())
        print(f"Total time: {total_time:.2f}s")
        
        print("\nPhase timings:")
        for phase, elapsed in self.timings.items():
            print(f"  {phase}: {elapsed:.2f}s")
        
        if self.dump_ir:
            print(f"\nIR dumps saved to: {self.output_dir}/")
        
        print(f"{'=' * 60}\n")
