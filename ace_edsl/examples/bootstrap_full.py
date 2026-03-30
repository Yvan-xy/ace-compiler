"""
Full CKKS Bootstrap Algorithm Implementation for ACE EDSL
=========================================================

This module supports three implementations selected by `ACE_BOOTSTRAP_IMPL`:
- `primitive` (default; aliases: `inline`, `ops`, `dsl`, `mimic`): staged
  bootstrap ops in EDSL (`CoeffToSlot -> EvalMod -> SlotToCoeff`).
- `evalmod` (legacy): explicit large EvalMod demo (Chebyshev + double-angle).
- `rtlib`: mimic ANT rtlib bootstrap by emitting CKKS `Bootstrap` op directly.

`rtlib` mode is the closest match to rtlib behavior in generated code because it
lowers to the runtime bootstrap path (`Eval_bootstrap_ciph(...)` on ANT).
`primitive` mode emits first-class bootstrap stage ops in EDSL, providing a
full bootstrap semantic path without direct `Bootstrap(...)` call emission.

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


def _bootstrap_poly_degree() -> int:
    """Return the poly degree used when generating bootstrap demo artifacts."""
    raw = os.environ.get("ACE_BOOTSTRAP_POLY_DEGREE", "").strip()
    if not raw:
        return 16384
    try:
        degree = int(raw)
    except ValueError:
        return 16384
    return degree if degree > 0 else 16384


def _bootstrap_m_by_4() -> int:
    return _bootstrap_poly_degree() // 2


def _bootstrap_three_m_by_4() -> int:
    return (_bootstrap_poly_degree() * 3) // 2


def _skip_preprocessor(func):
    """Mark a kernel to bypass AST preprocessing.

    bootstrap_full contains ordinary Python mode-selection branches. The AST
    preprocessor currently rewrites those branches in a way that collapses the
    primitive path back to `ct.bootstrap()`. Skipping preprocessing for this
    kernel preserves the real decomposition body during AIR tracing.
    """
    func._ace_skip_preprocessor = True
    return func


def _bootstrap_impl_mode() -> str:
    """Return selected bootstrap implementation mode."""
    mode = os.environ.get("ACE_BOOTSTRAP_IMPL", "primitive").strip().lower()
    if mode in ("rtlib", "runtime", "native"):
        return "rtlib"
    if mode in ("evalmod", "cheb", "legacy"):
        return "evalmod"
    if mode in ("primitive", "inline", "ops", "dsl", "mimic"):
        return "primitive"
    return "primitive"


def _use_rtlib_bootstrap() -> bool:
    return _bootstrap_impl_mode() == "rtlib"

def _bootstrap_mul_level() -> int:
    """Return CKKS mul level budget for bootstrap demo pipeline."""
    raw = os.environ.get("ACE_BOOTSTRAP_MUL_LEVEL", "").strip()
    if raw:
        try:
            lvl = int(raw)
            if lvl > 0:
                return lvl
        except ValueError:
            pass
    # Demo default matches the full available Q-level count for N=16384.
    return 26


def _bootstrap_input_level() -> int:
    """Return the configured CKKS input ciphertext level for the demo."""
    raw = os.environ.get("ACE_BOOTSTRAP_INPUT_LEVEL", "").strip()
    if not raw:
        # Bootstrap should consume a low-level ciphertext by default.
        return 1
    try:
        lvl = int(raw)
    except ValueError:
        return 0
    return lvl if lvl >= 0 else 0


def _identity_bootstrap_cleartext_reference(values):
    """Cleartext model for message-preserving bootstrap paths (rtlib mode only)."""
    return [float(v) for v in values]


def bootstrap_full_python_reference(values):
    """Cleartext reference matching selected implementation mode."""
    impl_mode = _bootstrap_impl_mode()
    if impl_mode == "rtlib":
        return _identity_bootstrap_cleartext_reference(values)
    # primitive mode now does real EvalMod math; use ANT reference.
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

    def raise_mod(self, _mod_size):
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

    def bootstrap_coeffs_to_slots(self, _num_slots=0):
        # Cleartext: CoeffToSlot is the DFT which converts coefficients
        # to slots.  In cleartext slot-domain, this is approximately
        # identity (DFT(IDFT(x)) = x).  Use rotation+add butterfly to
        # match the homomorphic structure.
        x = _ClearSlots(self.vals)
        n = len(x.vals)
        log_n = max(1, int(math.log2(n))) if n > 1 else 1
        for i in range(log_n - 1, -1, -1):
            r = x.rotate(1 << i)
            x = x + r
        return x

    def bootstrap_eval_mod(self):
        # Cleartext: apply the actual Chebyshev + double-angle per slot.
        from ant_bootstrap_ref import (
            eval_mod_cleartext as _eval_mod,
        )
        evaled = [_eval_mod(v) for v in self.vals]
        return _ClearSlots(evaled)

    def bootstrap_slots_to_coeffs(self, _num_slots=0):
        # Cleartext: SlotToCoeff is the IDFT.
        x = _ClearSlots(self.vals)
        n = len(x.vals)
        log_n = max(1, int(math.log2(n))) if n > 1 else 1
        for i in range(log_n):
            r = x.rotate(-(1 << i))
            x = x + r
        return x

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
    if _use_rtlib_bootstrap():
        return _identity_bootstrap_cleartext_reference(values)
    kernel_body = getattr(bootstrap_full, "__wrapped__", bootstrap_full)
    ct = _ClearSlots(values)
    zero = _ClearSlots([0.0] * len(values))
    coeffs = list(G_COEFFICIENTS_UNIFORM_HW_192)
    da_scalars = get_double_angle_scalars()
    args = [ct, zero, 1.0] + coeffs + da_scalars + [float(BOOTSTRAP_POST_SCALE)]
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


def _bootstrap_extended_prelude(ct):
    """Materialize bootstrap-adjacent CKKS ops while keeping value stable."""
    x0 = ct.raise_mod(2)
    x1 = x0.conjugate()
    # Keep x0 value while still exercising conjugate in IR.
    x2 = x0 + x1 - x1
    # Mul-by-monomial is rewritten to rotate in primitive mode.
    mono = x2.mul_mono(4)
    return x2 + mono - mono


# =============================================================================
# Complete Bootstrap Kernel (ANT EvalMod: Chebyshev 55 + double-angle 3 + post scale)
# =============================================================================

@ckks_kernel
@_skip_preprocessor
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
    """Bootstrap kernel with selectable implementation mode.

    `primitive`: emit bootstrap-stage ops in EDSL
                 (CoeffToSlot -> EvalMod -> SlotToCoeff).
    `evalmod`: full EDSL flow output (CoeffToSlot + EvalMod + SlotToCoeff).
    `rtlib`: emit CKKS Bootstrap op directly (lowered by runtime bootstrap path).
    """
    out = ct
    impl_mode = _bootstrap_impl_mode()
    if _use_rtlib_bootstrap():
        if hasattr(ct, "bootstrap"):
            out = ct.bootstrap()
        elif isinstance(ct, _ClearSlots):
            # Python cleartext fallback path for reference execution.
            out = _ClearSlots(_identity_bootstrap_cleartext_reference(ct.vals))
        else:
            # Keep kernel preprocess-friendly: no early return branches.
            out = ct
    else:
        if impl_mode == "primitive":
            # Full-packed bootstrap decomposition via bootstrap_decomposition.
            from ace_edsl.edsl.core.bootstrap_decomposition import (
                fullpacked_bootstrap_primitive,
            )
            x_in = ct.raise_mod(_bootstrap_mul_level())
            # Convert post_scale to float for the decomposition.
            # During cleartext execution it's already a float;
            # during tracing, extract the compile-time constant.
            try:
                ps_val = float(post_scale)
            except (TypeError, ValueError):
                ps_val = float(BOOTSTRAP_POST_SCALE)
            out = fullpacked_bootstrap_primitive(
                x_in,
                m_by_4=_bootstrap_m_by_4(),
                three_m_by_4=_bootstrap_three_m_by_4(),
                post_scale=ps_val,
            )
        else:
            # Shared explicit EDSL flow (legacy): prelude + staged transforms.
            x_in = _bootstrap_extended_prelude(ct)
            # Full EvalMod demo path (deep Chebyshev + double-angle).
            rot4 = x_in.rotate(4)
            dft0 = x_in + rot4
            rot2 = dft0.rotate(2)
            dft1 = dft0 + rot2
            rot1 = dft1.rotate(1)
            x = dft1 + rot1
            T_prev2 = zero + one
            T_prev1 = x
            out = T_prev2 * g0 + T_prev1 * g1
            cheb_coeffs = (
                g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15,
                g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27,
                g28, g29, g30, g31, g32, g33, g34, g35, g36, g37, g38, g39,
                g40, g41, g42, g43, g44, g45, g46, g47, g48, g49, g50, g51,
                g52, g53, g54,
            )
            for gk in cheb_coeffs:
                t_curr = 2 * x * T_prev1 - T_prev2
                out = out + t_curr * gk
                T_prev2, T_prev1 = T_prev1, t_curr
            da = out
            da = da * da
            da = da + da
            da = da + da1
            da = da * da
            da = da + da
            da = da + da2
            da = da * da
            da = da + da
            da = da + da3
            evalmod_result = da * post_scale
            rot1b = evalmod_result.rotate(1)
            idft0 = evalmod_result - rot1b
            rot2b = idft0.rotate(2)
            idft1 = idft0 - rot2b
            rot4b = idft1.rotate(4)
            out = idft1 - rot4b
    return out


# =============================================================================
# Standalone Demo
# =============================================================================

def run_demo():
    """Run bootstrap_full as a standalone demo, compiling to C code."""
    impl_mode = _bootstrap_impl_mode()
    print("=" * 70)
    print("Full CKKS Bootstrap Algorithm - ACE EDSL")
    print("=" * 70)
    print(f"Implementation mode: {impl_mode}")
    
    print("""
Bootstrap Algorithm:
┌─────────────────────────────────────────────────────────────────────┐
│  primitive mode (full decomposition):                               │
│    CoeffToSlot                 - U0hat diagonal linear transform     │
│    Full-packed split           - conjugate + add/sub + mul_mono      │
│    Dual EvalMod (PS)           - Chebyshev 55 (k=8,m=3) + 3 DA     │
│    Recombine                   - mul_mono + add                      │
│    SlotToCoeff                 - U0 diagonal linear transform        │
│    Post-scale                  - * 16 (q0/sf ratio)                  │
│  evalmod mode:                                                      │
│    Full-flow output            - CoeffToSlot + EvalMod + SlotToCoeff│
│    EvalMod (ANT)               - Chebyshev 55 + 3 double-angles     │
│                                + post scale 16                       │
│  rtlib mode:                                                        │
│    Direct CKKS Bootstrap op   - lowers to rtlib bootstrap path      │
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
    poly_degree = _bootstrap_poly_degree()
    ct = CkksCiphertext(shape=(poly_degree,), name="input_ct")
    zero = CkksCiphertext(shape=(poly_degree,), name="zero_ct")
    coeffs = list(G_COEFFICIENTS_UNIFORM_HW_192)
    da_scalars = get_double_angle_scalars()
    kernel_args = [ct, zero, 1.0] + coeffs + da_scalars + [float(BOOTSTRAP_POST_SCALE)]
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
        poly_degree=poly_degree,
        mul_level=_bootstrap_mul_level(),
        input_level=_bootstrap_input_level(),
        security_level=0,  # 0 = skip validation (mul_depth=23 exceeds 128-bit limit at N=16384/32768)
        scaling_factor_bits=56,
        first_prime_bits=60,
        hamming_weight=192,
        data_file=data_file_path,
        enable_poly=True,   # Poly-level C (Hw_modadd, Rotate, etc.) for ANT rtlib; scale handled in pipeline
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
    with open(c_file, 'w') as f:
        f.write(result.c_code)
        if wrapper_code:
            f.write("\n// --- Wrapper (Main_graph, encode/decode schemes) ---\n")
            f.write(wrapper_code)
    total_bytes = len(result.c_code) + len(wrapper_code)
    print(f"  ✓ C code: {c_file} ({total_bytes:,} bytes, includes wrapper)")
    
    for stage, dump in result.air_dumps.items():
        air_file = os.path.join(output_dir, f"bootstrap_full_{stage}.air")
        with open(air_file, 'w') as f:
            f.write(dump)
        print(f"  ✓ AIR dump ({stage}): {air_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    lines = result.c_code.split('\n')
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if impl_mode == "rtlib":
        phase_summary = (
            "Bootstrap Phases:\n"
            "  └─ Direct CKKS.bootstrap lowering to rtlib bootstrap path"
        )
    elif impl_mode == "primitive":
        phase_summary = (
            "Bootstrap Phases (full-packed decomposition):\n"
            "  ├─ CoeffToSlot:  U0hat diagonal linear transform\n"
            "  ├─ Conjugate:    split real/imag + mul_mono\n"
            f"  ├─ Dual EvalMod: PS Chebyshev (k=8, m=3, deg=54) + {NUM_DOUBLE_ANGLE} DA\n"
            "  ├─ Recombine:    mul_mono + add\n"
            "  ├─ SlotToCoeff:  U0 diagonal linear transform\n"
            f"  └─ Post-scale:   * {BOOTSTRAP_POST_SCALE}"
        )
    else:
        phase_summary = (
            "Bootstrap Phases:\n"
            f"  ├─ Phase 1:      {LOG_SLOTS} CoeffToSlot butterfly layers\n"
            f"  ├─ EvalMod:      Chebyshev {CHEB_COEFF_COUNT} coeffs + {NUM_DOUBLE_ANGLE} double-angles + post scale {BOOTSTRAP_POST_SCALE}\n"
            f"  └─ Phase 3:      {LOG_SLOTS} SlotToCoeff butterfly layers"
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
