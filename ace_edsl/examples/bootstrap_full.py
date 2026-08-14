"""
Full CKKS Bootstrap Algorithm Implementation for ACE EDSL
=========================================================

This module emits the primitive decomposition path in EDSL:
`CoeffToSlot -> EvalMod -> SlotToCoeff`.

The primitive path keeps bootstrap visible in CKKS AIR without direct
`Bootstrap(...)` / `Eval_bootstrap_ciph(...)` emission.

**EvalMod:** The kernel mirrors ANT's EvalMod core (bootstrap.c): Chebyshev
series (55 coeffs from G_coefficients_uniform_hw_192), double-angle iterations
(x -> 2*x^2 + scalar_j, j=1,2,3), and post scale 16.

Bootstrap refreshes a ciphertext's noise budget through:
1. ModRaise: Raise the ciphertext modulus (implicit in CKKS)
2. CoeffToSlot: Homomorphic DFT (coefficient → slot representation)
3. EvalMod: Approximate modular reduction using sine polynomial
4. SlotToCoeff: Homomorphic inverse DFT (slot → coefficient representation)

This version uses subtraction instead of negation since ckks2poly
supports sub but not neg. Compiles through the full pipeline to C code.

Key difference from acepy/examples/bootstrap_full.py:
-----------------------------------------------------
In acepy, this would be compiled to AIR using @ckks_kernel.compile() and then
manually inlined by the PythonLoweringPass.

In ace_edsl, when this function is called from another kernel, the DSL
detects the nested call and executes the function body directly, tracing
operations into the caller's AIR (automatic inlining).
No separate inlining pass is needed!

References:
- "Bootstrapping for Approximate Homomorphic Encryption" (Cheon et al., 2018)
- "Improved Bootstrapping for Approximate Homomorphic Encryption" (Chen & Cheon, 2019)
- "Better Bootstrapping for Approximate Homomorphic Encryption" (Lee et al., 2020)
"""

import sys
import os
import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

# Setup path for imports
def _setup_sys_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(repo_root, ".."))
    for path in (repo_root, parent_root):
        if path not in sys.path:
            sys.path.insert(0, path)

_setup_sys_path()

from ace_edsl.edsl import ckks_kernel, CkksCiphertext, CkksPlaintext

# ANT bootstrap constants (shared by kernel/driver only; ant_bootstrap_ref stays stdlib-only)
from bootstrap_ant_constants import (
    G_COEFFICIENTS_UNIFORM_HW_192,
    get_double_angle_scalars,
    BOOTSTRAP_POST_SCALE,
    UNIFORM_COEFF_SIZE_HW_192,
    R_UNIFORM_HW_192,
)
from ace_edsl.edsl.core.bootstrap_decomposition import BootstrapConfig
from ace_edsl.edsl.core.bootstrap_math import EVAL_SIN_UPPER_BOUND_K

# ANT full bootstrap reference (Python port of Eval_bootstrap); see plan ant_full_bootstrap_python_port
try:
    from ant_bootstrap_ref import ant_bootstrap_full_reference  # noqa: F401
except ImportError:
    ant_bootstrap_full_reference = None  # type: ignore[misc, assignment]

# =============================================================================
# Configuration
# =============================================================================

LOG_SLOTS = 3  # 8 slots for demo
NUM_SLOTS = 1 << LOG_SLOTS
NUM_DOUBLE_ANGLE = R_UNIFORM_HW_192  # 3, matches ANT
CHEB_COEFF_COUNT = UNIFORM_COEFF_SIZE_HW_192  # 55
UNIFORM_COEFFICIENT_HAMMING_WEIGHT_MAX = 192
EVALMOD_COMPONENT_LOWER_BOUND = -1.0
EVALMOD_COMPONENT_UPPER_BOUND = 1.0


_EXPLICIT_TRACE_CONFIG: ContextVar[Optional[BootstrapConfig]] = ContextVar(
    "bootstrap_full_explicit_trace_config",
    default=None,
)


def _env_int(names, default: int, min_value: int = 1) -> int:
    """Parse the first valid integer from one or more env var names."""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= min_value:
            return value
    return default


def _bootstrap_poly_degree() -> int:
    """Return the poly degree used when generating bootstrap demo artifacts."""
    return _env_int("ACE_BOOTSTRAP_POLY_DEGREE", 16384)


def _bootstrap_mul_level() -> int:
    """Return CKKS mul level budget for bootstrap demo pipeline."""
    # Demo default matches the full available Q-level count for N=16384.
    return _env_int("ACE_BOOTSTRAP_MUL_LEVEL", 26)


def _bootstrap_input_level() -> int:
    """Return the configured CKKS input ciphertext level for the demo."""
    # Bootstrap should consume a low-level ciphertext by default.
    return _env_int("ACE_BOOTSTRAP_INPUT_LEVEL", 1, min_value=0)


def _bootstrap_first_prime_bits() -> int:
    return _env_int(
        ("ACE_BOOTSTRAP_FIRST_PRIME_BITS", "ACE_BOOTSTRAP_FIRST_MOD_SIZE"),
        60,
    )


def _bootstrap_scaling_factor_bits() -> int:
    return _env_int(
        ("ACE_BOOTSTRAP_SCALING_FACTOR_BITS", "ACE_BOOTSTRAP_SCALING_MOD_SIZE"),
        56,
    )


def _bootstrap_hamming_weight() -> int:
    return _env_int("ACE_BOOTSTRAP_HAMMING_WEIGHT", 192)


def _bootstrap_q_parts() -> int:
    return _env_int("ACE_BOOTSTRAP_Q_PARTS", 3)


def _bootstrap_transform_level_budget() -> int:
    return _env_int("ACE_BOOTSTRAP_TRANSFORM_LEVEL_BUDGET", 3)


def _bootstrap_enc_budget() -> int:
    return _env_int("ACE_BOOTSTRAP_ENC_BUDGET", _bootstrap_transform_level_budget())


def _bootstrap_dec_budget() -> int:
    return _env_int("ACE_BOOTSTRAP_DEC_BUDGET", _bootstrap_transform_level_budget())


def _bootstrap_ct_encode() -> bool:
    """Whether to pre-encode bootstrap plaintext constants into the data file."""
    raw = os.environ.get("ACE_BOOTSTRAP_CT_ENCODE", "").strip().lower()
    if not raw:
        return False
    return raw not in ("0", "false", "off", "no")


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return bool(raw) and raw not in ("0", "false", "off", "no")


def _bootstrap_runtime_raise_level() -> bool:
    return _env_flag("ACE_BOOTSTRAP_RUNTIME_RAISE_LEVEL")


def _bootstrap_stage_probe_enabled() -> bool:
    return _env_flag("ACE_BOOTSTRAP_STAGE_PROBE")


def _bootstrap_function_name_prefix() -> str:
    return os.environ.get("ACE_BOOTSTRAP_FUNCTION_NAME_PREFIX", "")


def _bootstrap_constant_name_prefix() -> str:
    return os.environ.get("ACE_BOOTSTRAP_CONSTANT_NAME_PREFIX", "")


def _bootstrap_pt_from_msg_name() -> str:
    return os.environ.get("ACE_BOOTSTRAP_PT_FROM_MSG_NAME", "Pt_from_msg")


def _bootstrap_raise_level_name() -> str:
    return os.environ.get("ACE_BOOTSTRAP_RAISE_LEVEL_NAME", "")


def _stage_probe_c_prologue() -> str:
    """Return direct-codegen probe redirects emitted before generated code.

    The ResNet shim provides strong timing wrappers for these symbols. Weak
    fallbacks keep standalone generated bootstrap experiments linkable.
    """
    return """
#ifndef ACE_DSL_BTS_STAGE_PROBE_WEAK
#if defined(__GNUC__) || defined(__clang__)
#define ACE_DSL_BTS_STAGE_PROBE_WEAK __attribute__((weak))
#else
#define ACE_DSL_BTS_STAGE_PROBE_WEAK
#endif
#endif

ACE_DSL_BTS_STAGE_PROBE_WEAK CIPHER dsl_bts_probe_Conjugate_ciph(CIPHER res, CIPHER ciph) {
  return Conjugate_ciph(res, ciph);
}

ACE_DSL_BTS_STAGE_PROBE_WEAK CIPHER dsl_bts_probe_Mul_mono_ciph(CIPHER res, CIPHER ciph,
                                                                 uint32_t power) {
  return Mul_mono_ciph(res, ciph, power);
}

ACE_DSL_BTS_STAGE_PROBE_WEAK void dsl_bts_probe_Init_ciph_same_scale(CIPHER res,
                                                                      CIPHER ciph1,
                                                                      CIPHER ciph2) {
  Init_ciph_same_scale(res, ciph1, ciph2);
}

#define Conjugate_ciph dsl_bts_probe_Conjugate_ciph
#define Mul_mono_ciph dsl_bts_probe_Mul_mono_ciph
#define Init_ciph_same_scale dsl_bts_probe_Init_ciph_same_scale

"""


def _with_stage_probe_prologue(c_code: str) -> str:
    """Add structured helper redirects for direct-codegen stage timing."""
    if not _bootstrap_stage_probe_enabled():
        return c_code
    if "dsl_bts_probe_Conjugate_ciph" in c_code:
        return c_code
    include_anchor = '#include "rt_ant/rt_ant.h"\n'
    prologue = _stage_probe_c_prologue()
    if include_anchor in c_code:
        return c_code.replace(include_anchor, include_anchor + prologue, 1)
    return prologue + c_code


def build_bootstrap_trace_config(
    *,
    poly_degree: int,
    mul_level: int,
    first_prime_bits: int,
    scaling_factor_bits: int,
    hamming_weight: int,
    q_parts: int,
    enc_budget: int,
    dec_budget: int,
    ct_encode: bool,
    clear_imag: bool = False,
) -> BootstrapConfig:
    """Build a trace config without reading process-global configuration.

    ``clear_imag`` is only valid when the caller has proved that the semantic
    result is real-valued.  It selects the conjugate projection already used
    by the full-packed primitive; the default preserves complex semantics.
    """
    if hamming_weight > UNIFORM_COEFFICIENT_HAMMING_WEIGHT_MAX:
        raise ValueError(
            "the expanded uniform coefficient family supports hamming weight "
            f"at most {UNIFORM_COEFFICIENT_HAMMING_WEIGHT_MAX}"
        )
    return BootstrapConfig(
        poly_degree=poly_degree,
        mul_level=mul_level,
        first_prime_bits=first_prime_bits,
        scaling_factor_bits=scaling_factor_bits,
        hamming_weight=hamming_weight,
        q_parts=q_parts,
        enc_budget=enc_budget,
        dec_budget=dec_budget,
        ct_encode=ct_encode,
        eval_sin_upper_bound_k=EVAL_SIN_UPPER_BOUND_K,
        chebyshev_coefficients=tuple(G_COEFFICIENTS_UNIFORM_HW_192),
        double_angle_scalars=tuple(get_double_angle_scalars(NUM_DOUBLE_ANGLE)),
        clear_imag=clear_imag,
    )


@contextmanager
def bootstrap_trace_configuration(
    config: BootstrapConfig,
) -> Iterator[None]:
    """Scope one explicit configuration to a kernel trace."""
    token = _EXPLICIT_TRACE_CONFIG.set(config)
    try:
        yield
    finally:
        _EXPLICIT_TRACE_CONFIG.reset(token)


def _bootstrap_trace_config() -> BootstrapConfig:
    """Build trace metadata, retaining environment defaults for the demo only."""
    explicit = _EXPLICIT_TRACE_CONFIG.get()
    if explicit is not None:
        return explicit
    return build_bootstrap_trace_config(
        poly_degree=_bootstrap_poly_degree(),
        mul_level=_bootstrap_mul_level(),
        first_prime_bits=_bootstrap_first_prime_bits(),
        scaling_factor_bits=_bootstrap_scaling_factor_bits(),
        hamming_weight=_bootstrap_hamming_weight(),
        q_parts=_bootstrap_q_parts(),
        enc_budget=_bootstrap_enc_budget(),
        dec_budget=_bootstrap_dec_budget(),
        ct_encode=_bootstrap_ct_encode(),
    )


def bootstrap_full_python_reference(values):
    """Cleartext reference matching primitive bootstrap EvalMod math."""
    if ant_bootstrap_full_reference is not None:
        return ant_bootstrap_full_reference(values)
    return [math.sin(8.0 * float(v)) for v in values]  # fallback


def _coerce_numeric(value):
    """Preserve complex values for cleartext bootstrap simulation."""
    if isinstance(value, complex):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(value), 0.0)


class _ClearSlots:
    """Minimal cleartext vector type to run the kernel body in pure Python."""

    def __init__(self, vals):
        self.vals = [_coerce_numeric(v) for v in vals]

    def _binary(self, other, op):
        if isinstance(other, _ClearSlots):
            assert len(self.vals) == len(other.vals)
            return _ClearSlots([op(a, b) for a, b in zip(self.vals, other.vals)])
        scalar = _coerce_numeric(other)
        return _ClearSlots([op(a, scalar) for a in self.vals])

    def rotate(self, k):
        n = len(self.vals)
        r = int(k) % n
        if r == 0:
            return _ClearSlots(self.vals)
        return _ClearSlots(self.vals[-r:] + self.vals[:-r])

    def raise_mod(self, _mod_size, runtime_raise_level=False):
        del runtime_raise_level
        # Cleartext model: modulus raising is value-preserving.
        return _ClearSlots(self.vals)

    def conjugate(self):
        return _ClearSlots([v.conjugate() for v in self.vals])

    def mul_mono(self, power):
        # Match current primitive rewrite surrogate in this demo.
        return self.rotate(int(power))

    def rescale(self):
        # Cleartext model: rescale is value-preserving.
        return _ClearSlots(self.vals)

    def mod_switch(self):
        # Cleartext model: mod_switch is value-preserving.
        return _ClearSlots(self.vals)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        scalar = _coerce_numeric(other)
        return _ClearSlots([scalar - a for a in self.vals])


def bootstrap_full_python_dsl_reference(values):
    """Run the undecorated @ckks_kernel body in Python and return cleartext slots."""
    kernel_body = getattr(bootstrap_full, "__wrapped__", bootstrap_full)
    ct = _ClearSlots(values)
    zero = _ClearSlots([0.0] * len(values))
    bootstrap_config = _bootstrap_trace_config()
    coeffs = list(G_COEFFICIENTS_UNIFORM_HW_192)
    da_scalars = get_double_angle_scalars(NUM_DOUBLE_ANGLE)
    args = [ct, zero, 1.0] + coeffs + da_scalars + [bootstrap_config.post_scale]
    out = kernel_body(*args)
    if isinstance(out, _ClearSlots):
        normalized = []
        for v in out.vals:
            if isinstance(v, complex) and abs(v.imag) < 1e-9:
                normalized.append(float(v.real))
            else:
                normalized.append(v)
        return normalized
    raise TypeError("bootstrap_full_python_dsl_reference expected _ClearSlots output")


# =============================================================================
# Complete Bootstrap Kernel (ANT EvalMod: Chebyshev 55 + double-angle 3 + post scale)
# =============================================================================

@ckks_kernel
def bootstrap_full(
    ct: CkksCiphertext,
    zero: CkksCiphertext,
    one: CkksPlaintext,
    g0: CkksPlaintext,
    g1: CkksPlaintext,
    g2: CkksPlaintext,
    g3: CkksPlaintext,
    g4: CkksPlaintext,
    g5: CkksPlaintext,
    g6: CkksPlaintext,
    g7: CkksPlaintext,
    g8: CkksPlaintext,
    g9: CkksPlaintext,
    g10: CkksPlaintext,
    g11: CkksPlaintext,
    g12: CkksPlaintext,
    g13: CkksPlaintext,
    g14: CkksPlaintext,
    g15: CkksPlaintext,
    g16: CkksPlaintext,
    g17: CkksPlaintext,
    g18: CkksPlaintext,
    g19: CkksPlaintext,
    g20: CkksPlaintext,
    g21: CkksPlaintext,
    g22: CkksPlaintext,
    g23: CkksPlaintext,
    g24: CkksPlaintext,
    g25: CkksPlaintext,
    g26: CkksPlaintext,
    g27: CkksPlaintext,
    g28: CkksPlaintext,
    g29: CkksPlaintext,
    g30: CkksPlaintext,
    g31: CkksPlaintext,
    g32: CkksPlaintext,
    g33: CkksPlaintext,
    g34: CkksPlaintext,
    g35: CkksPlaintext,
    g36: CkksPlaintext,
    g37: CkksPlaintext,
    g38: CkksPlaintext,
    g39: CkksPlaintext,
    g40: CkksPlaintext,
    g41: CkksPlaintext,
    g42: CkksPlaintext,
    g43: CkksPlaintext,
    g44: CkksPlaintext,
    g45: CkksPlaintext,
    g46: CkksPlaintext,
    g47: CkksPlaintext,
    g48: CkksPlaintext,
    g49: CkksPlaintext,
    g50: CkksPlaintext,
    g51: CkksPlaintext,
    g52: CkksPlaintext,
    g53: CkksPlaintext,
    g54: CkksPlaintext,
    da1: CkksPlaintext,
    da2: CkksPlaintext,
    da3: CkksPlaintext,
    post_scale: CkksPlaintext,
) -> CkksCiphertext:
    """Primitive bootstrap kernel: emit full CKKS-level decomposition."""
    from ace_edsl.edsl.core.bootstrap_decomposition import (
        fullpacked_bootstrap_primitive,
    )

    bootstrap_config = _bootstrap_trace_config()
    # Raise to the full available tower before the staged bootstrap flow.
    x_in = ct.raise_mod(
        bootstrap_config.raise_level,
        runtime_raise_level=_bootstrap_runtime_raise_level(),
    )
    return fullpacked_bootstrap_primitive(
        x_in,
        config=bootstrap_config,
        clear_imag=bootstrap_config.clear_imag,
    )


# =============================================================================
# Standalone Demo
# =============================================================================

def run_demo():
    """Run bootstrap_full as a standalone demo, compiling to C code."""
    bootstrap_config = _bootstrap_trace_config()
    print("=" * 70)
    print("Full CKKS Bootstrap Algorithm - ACE EDSL")
    print("=" * 70)
    print("Implementation mode: primitive")
    
    print(f"""
Bootstrap Algorithm:
┌─────────────────────────────────────────────────────────────────────┐
│  primitive mode (full decomposition):                               │
│    CoeffToSlot                 - U0hat diagonal linear transform     │
│    Full-packed split           - conjugate + add/sub + mul_mono      │
│    Dual EvalMod (PS)           - Chebyshev 55 (k=8,m=3) + {len(bootstrap_config.double_angle_scalars)} DA     │
│    Recombine                   - mul_mono + add                      │
│    SlotToCoeff                 - U0 diagonal linear transform        │
│    Post-scale                  - * {bootstrap_config.post_scale:g} (q0/sf ratio)                  │
└─────────────────────────────────────────────────────────────────────┘

Key Difference from acepy:
  In acepy: bootstrap_full.compile() → AIR → separate inlining pass
  In ace_edsl: Just call the function → automatic tracing via operators!
  
  Nested kernel calls work automatically because the DSL detects when
  _in_air_context is True and executes the function body directly.
""")
    
    # ========================================================================
    # Step 1: Trace to AIR via operator overloading
    # ========================================================================
    print("=" * 70)
    print("Step 1: Trace to CKKS AIR (via operator overloading)")
    print("=" * 70)
    
    from ace_edsl.edsl import AceEDSL
    
    # Clear DSL singleton state
    AceEDSL._get_dsl.cache_clear()
    
    # Execute the kernel - this triggers tracing
    poly_degree = bootstrap_config.poly_degree
    ct = CkksCiphertext(shape=(poly_degree,), name="input_ct")
    zero = CkksCiphertext(shape=(poly_degree,), name="zero_ct")
    coeffs = list(G_COEFFICIENTS_UNIFORM_HW_192)
    da_scalars = list(bootstrap_config.double_angle_scalars)
    kernel_args = [ct, zero, 1.0] + coeffs + da_scalars + [bootstrap_config.post_scale]
    bootstrap_full(*kernel_args)
    
    dsl = AceEDSL._get_dsl()
    glob = dsl.current_air_module
    
    if glob is None:
        print("ERROR: No AIR module generated")
        return False
    
    ir = glob.dump()
    
    # Count operations
    rotate_count = ir.lower().count('ckks.rotate')
    mul_count = ir.lower().count('ckks.mul')
    add_count = ir.lower().count('ckks.add')
    sub_count = ir.lower().count('ckks.sub')
    
    print(f"\nOperation counts:")
    print(f"  CKKS.rotate: {rotate_count:3d}  (CoeffToSlot/SlotToCoeff)")
    print(f"  CKKS.mul:    {mul_count:3d}")
    print(f"  CKKS.add:    {add_count:3d}  (combining terms)")
    print(f"  CKKS.sub:    {sub_count:3d}  (subtraction terms)")
    print(f"  ─────────────────")
    print(f"  Total:       {rotate_count + mul_count + add_count + sub_count}")
    
    print(f"\nCKKS AIR (first 2000 chars):")
    print("-" * 40)
    print(ir[:2000])
    if len(ir) > 2000:
        print(f"... ({len(ir)} total chars)")
    
    # ========================================================================
    # Step 2: Run full pipeline to C code
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Run Pipeline (CKKS AIR → C code)")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    raw_air_file = os.path.join(output_dir, "bootstrap_full_raw.air")
    with open(raw_air_file, "w") as f:
        f.write(ir)
    print(f"  ✓ AIR dump (raw @ckks_kernel): {raw_air_file}")
    data_file_path = os.path.abspath(os.path.join(output_dir, "bootstrap_full_data.msg"))
    
    from ace_edsl.edsl import AcePipeline
    
    pipeline = AcePipeline(glob)
    pipeline.configure_fhe(
        poly_degree=bootstrap_config.poly_degree,
        mul_level=bootstrap_config.mul_level,
        input_level=_bootstrap_input_level(),
        security_level=0,  # 0 = skip validation (mul_depth=23 exceeds 128-bit limit at N=16384/32768)
        scaling_factor_bits=bootstrap_config.scaling_factor_bits,
        first_prime_bits=bootstrap_config.first_prime_bits,
        hamming_weight=bootstrap_config.hamming_weight,
        data_file=data_file_path,
        ct_encode=bootstrap_config.ct_encode,
        enable_poly=True,   # Poly-level C (Hw_modadd, Rotate, etc.) for ANT rtlib; scale handled in pipeline
        function_name_prefix=_bootstrap_function_name_prefix(),
        constant_name_prefix=_bootstrap_constant_name_prefix(),
        pt_from_msg_name=_bootstrap_pt_from_msg_name(),
        raise_mod_level_func=_bootstrap_raise_level_name(),
    )
    # Keep CKKS extended-op semantics intact for staged bootstrap flow.
    # The current generic rewrite maps conjugate -> identity, which breaks
    # full-packed bootstrap split/recombine math.
    pipeline.set_ckks_extended_op_rewrite(False)
    
    result = pipeline.run(start_domain="fhe::ckks", dump_stages=True, verbose=True)
    
    if not result.success:
        print(f"ERROR: Pipeline failed: {result.error}")
        return False
    
    # ========================================================================
    # Step 3: Write output
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Write Output")
    print("=" * 70)
    
    c_file = os.path.join(output_dir, "bootstrap_full.c")
    # Append the wrapper (Main_graph, Get_encode_scheme, etc.) so the output
    # is a single self-contained file matching the native compiler pattern.
    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")
    wrapper_path = os.path.join(tests_dir, "bootstrap_full_wrapper.inc")
    wrapper_code = ""
    if os.path.isfile(wrapper_path):
        with open(wrapper_path) as wf:
            wrapper_code = wf.read()
    c_code = _with_stage_probe_prologue(result.c_code)
    with open(c_file, 'w') as f:
        f.write(c_code)
        if wrapper_code:
            f.write("\n// --- Wrapper (Main_graph, encode/decode schemes) ---\n")
            f.write(wrapper_code)
    total_bytes = len(c_code) + len(wrapper_code)
    print(f"  ✓ C code: {c_file} ({total_bytes:,} bytes, includes wrapper)")
    
    for stage, dump in result.air_dumps.items():
        air_file = os.path.join(output_dir, f"bootstrap_full_{stage}.air")
        with open(air_file, 'w') as f:
            f.write(dump)
        print(f"  ✓ AIR dump ({stage}): {air_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    lines = c_code.split('\n')
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    phase_summary = (
        "Bootstrap Phases (full-packed decomposition):\n"
        "  ├─ CoeffToSlot:  U0hat diagonal linear transform\n"
        "  ├─ Conjugate:    split real/imag + mul_mono\n"
        f"  ├─ Dual EvalMod: PS Chebyshev (k=8, m=3, deg=54) + {len(bootstrap_config.double_angle_scalars)} DA\n"
        "  ├─ Recombine:    mul_mono + add\n"
        "  ├─ SlotToCoeff:  U0 diagonal linear transform\n"
        f"  └─ Post-scale:   * {bootstrap_config.post_scale:g}"
    )
    print(f"""
✓ Full bootstrap compiled to C code!

Pipeline Results:
  ├─ CKKS AIR:    {len(ir):,} chars
  └─ C Code:      {len(lines)} lines

{phase_summary}

Key Advantage of ace_edsl:
  ✓ No separate compile() call needed
  ✓ No InliningEngine pass needed
  ✓ Operators automatically trace to AIR
  ✓ Nested @ckks_kernel calls work automatically
""")
    
    print("=" * 70)
    print("✓ Bootstrap demo complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
