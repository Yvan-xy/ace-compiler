"""
Full mathematical decomposition of CKKS bootstrap stages for the EDSL.

Replaces the identity-surrogate primitives with actual Paterson-Stockmeyer
Chebyshev evaluation, double-angle iterations, and the full-packed bootstrap
branch (conjugate split, dual EvalMod, monomial recombine).

These functions operate on AIRValue objects and emit real CKKS operations.
The generated C code will contain the same algorithmic structure as the
ANT rtlib Eval_bootstrap (bootstrap.c / chebyshev_impl.c).

Note on rescale/level management:
    The rtlib explicitly rescales at several points (Eval_linear_wsum,
    Apply_double_angle, conjugate split, recombine) and mod-switches
    baby-step T_i to level-align with T_k. In the EDSL pipeline, the
    scale manager auto-inserts rescales after multiplications, so this
    module avoids emitting explicit rescale nodes. It does emit the same
    baby-step mod-switch alignment used by the rtlib, because that level
    normalization materially affects the downstream bootstrap levels.

Usage (from AIRValue._bootstrap_*_primitive methods):

    from .bootstrap_decomposition import (
        eval_mod_primitive,
        coeffs_to_slots_primitive,
        slots_to_coeffs_primitive,
    )
"""

import hashlib
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from .bootstrap_math import (
    CHEBYSHEV_COEFFICIENTS,
    DOUBLE_ANGLE_SCALARS,
    NUM_DOUBLE_ANGLE,
    EVAL_SIN_UPPER_BOUND_K,
    compute_degree_ps,
    compute_chebyshev_depths,
    get_degree_from_coeffs,
    is_even_poly,
    long_div_chebyshev,
)


@dataclass(frozen=True)
class BootstrapConfig:
    """Trace-time parameters for primitive full-packed bootstrap generation."""

    poly_degree: int
    mul_level: int
    first_prime_bits: int
    scaling_factor_bits: int
    hamming_weight: int
    q_parts: int
    enc_budget: int
    dec_budget: int
    ct_encode: bool
    eval_sin_upper_bound_k: int
    chebyshev_coefficients: Tuple[float, ...]
    double_angle_scalars: Tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(
            self,
            "chebyshev_coefficients",
            tuple(float(v) for v in self.chebyshev_coefficients),
        )
        object.__setattr__(
            self,
            "double_angle_scalars",
            tuple(float(v) for v in self.double_angle_scalars),
        )
        if self.poly_degree <= 0 or self.poly_degree % 2 != 0:
            raise ValueError("BootstrapConfig.poly_degree must be a positive even value")
        if self.slots <= 0 or (self.slots & (self.slots - 1)) != 0:
            raise ValueError("BootstrapConfig.slots must be a positive power of two")
        if self.mul_level <= 0:
            raise ValueError("BootstrapConfig.mul_level must be positive")
        if self.first_prime_bits <= 0 or self.scaling_factor_bits <= 0:
            raise ValueError("BootstrapConfig prime bit sizes must be positive")
        if self.first_prime_bits < self.scaling_factor_bits:
            raise ValueError(
                "BootstrapConfig.first_prime_bits must be >= scaling_factor_bits"
            )
        if self.q_parts <= 0:
            raise ValueError("BootstrapConfig.q_parts must be positive")
        if self.enc_budget <= 0 or self.dec_budget <= 0:
            raise ValueError("BootstrapConfig transform budgets must be positive")
        if self.eval_sin_upper_bound_k <= 0:
            raise ValueError("BootstrapConfig.eval_sin_upper_bound_k must be positive")
        if not self.chebyshev_coefficients:
            raise ValueError("BootstrapConfig.chebyshev_coefficients must not be empty")

    @property
    def slots(self) -> int:
        return int(self.poly_degree) // 2

    @property
    def raise_level(self) -> int:
        return int(self.mul_level)

    @property
    def post_scale_degree(self) -> int:
        return int(self.first_prime_bits - self.scaling_factor_bits)

    @property
    def post_scale(self) -> float:
        return float(2 ** self.post_scale_degree)

    @property
    def bootstrap_depth(self) -> int:
        chebyshev_degree = get_degree_from_coeffs(list(self.chebyshev_coefficients))
        ps_k, ps_m = compute_degree_ps(chebyshev_degree)
        approx_mod_depth = (
            int(math.ceil(math.log2(ps_k))) + ps_m + len(self.double_angle_scalars)
        )
        return approx_mod_depth + int(self.enc_budget) + int(self.dec_budget)

    @property
    def transform_levels(self) -> tuple[int, int]:
        level_0 = int(self.mul_level) + (1 if self.ct_encode else 0)
        enc_level = max(1, level_0 - self.enc_budget)
        dec_level = max(1, level_0 - self.bootstrap_depth)
        return enc_level, dec_level

    @property
    def const_level(self) -> int:
        enc_level, _ = self.transform_levels
        return enc_level

    @property
    def num_p(self) -> int:
        num_per_part = math.ceil(float(self.mul_level) / float(self.q_parts))
        bit_num = self.first_prime_bits + (num_per_part - 1) * self.scaling_factor_bits
        return int(math.ceil(float(bit_num) / 60.0))

    @property
    def coeffs_to_slots_factor(self) -> float:
        return (
            1.0
            / float(self.poly_degree)
            / float(self.eval_sin_upper_bound_k)
            / float(self.post_scale)
        )


def _require_bootstrap_config(
    config: Optional[BootstrapConfig],
    caller: str,
) -> BootstrapConfig:
    if config is None:
        raise ValueError(f"{caller} requires an explicit BootstrapConfig")
    return config


def _compat_bootstrap_config(x=None, num_slots: int = 0) -> BootstrapConfig:
    """Build a deterministic config for legacy primitive stage-op calls.

    Full bootstrap generation should pass BootstrapConfig explicitly. This
    fallback keeps direct AIRValue stage primitives working when users enable
    ACE_BOOTSTRAP_STAGE_PRIMITIVE_LOWERING without going through bootstrap_full.
    """
    slots = int(num_slots) if int(num_slots) > 0 else 0
    if slots <= 0 and x is not None:
        shape = getattr(x, "shape", None)
        if shape:
            try:
                poly_degree = int(shape[0])
            except (TypeError, ValueError, IndexError):
                poly_degree = 0
            if poly_degree > 0:
                slots = poly_degree // 2
    if slots <= 0:
        slots = 8192

    return BootstrapConfig(
        poly_degree=slots * 2,
        mul_level=26,
        first_prime_bits=60,
        scaling_factor_bits=56,
        hamming_weight=192,
        q_parts=3,
        enc_budget=3,
        dec_budget=3,
        ct_encode=False,
        eval_sin_upper_bound_k=EVAL_SIN_UPPER_BOUND_K,
        chebyshev_coefficients=CHEBYSHEV_COEFFICIENTS,
        double_angle_scalars=DOUBLE_ANGLE_SCALARS,
    )


# =========================================================================
# Paterson-Stockmeyer Chebyshev evaluation
# =========================================================================

def eval_chebyshev_ps(
    x, coeffs: Optional[List[float]] = None, config: Optional[BootstrapConfig] = None
):
    """Evaluate Chebyshev series via Paterson-Stockmeyer on a ciphertext.

    Emits O(k + m + 2^{m-1}) multiplications with depth ceil(log2 k) + m,
    matching chebyshev_impl.c Eval_chebyshev_ps.

    Args:
        x: AIRValue ciphertext (input assumed in [-1, 1]).
        coeffs: Chebyshev series coefficients (default: UNIFORM_HW_192).

    Returns:
        AIRValue — Chebyshev polynomial evaluated at x.
    """
    if coeffs is None:
        coeffs = list(
            config.chebyshev_coefficients
            if config is not None
            else CHEBYSHEV_COEFFICIENTS
        )

    n = get_degree_from_coeffs(coeffs)
    even = is_even_poly(coeffs)

    # Trim trailing zeros
    f2 = list(coeffs[: n + 1])

    k, m = compute_degree_ps(n)
    if even and k % 2 == 1:
        k += 1

    # No linear transform needed for range [-1, 1].
    t0 = x
    y = t0  # kept for T_{odd} = 2*T_a*T_b - y

    # ------------------------------------------------------------------
    # Step 1: baby-step Chebyshev polynomials T_1 .. T_k
    # ------------------------------------------------------------------
    t_list: List = [None] * k
    t_list[0] = t0  # T_1(y) = y

    for i in range(2, k + 1):
        j = i - 1  # 0-indexed slot in t_list
        if (i & (i - 1)) == 0:
            # power of 2: T_i = 2 * T_{i/2}^2 - 1
            half = t_list[i // 2 - 1]
            prod = half * half          # mul
            t_j = prod + prod           # double  (2 * T^2)
            t_j = _add_const_like(t_j, -1.0, config)  # - 1
            t_list[j] = t_j
        elif i % 2 == 1:
            # odd, non-power-of-2
            if even:
                t_list[j] = None        # skip odd terms for even poly
                continue
            # T_i = 2 * T_{floor(i/2)} * T_{ceil(i/2)} - y
            half_lo = t_list[i // 2 - 1]
            half_hi = t_list[i // 2]
            prod = half_lo * half_hi
            t_j = prod + prod           # 2 * product
            t_j = t_j - y              # - y
            t_list[j] = t_j
        else:
            # even, non-power-of-2
            ihalf_1 = i // 2
            if even and ihalf_1 % 2 == 1:
                ihalf_1 += 1
            ihalf_2 = i - ihalf_1
            h1 = t_list[ihalf_1 - 1]
            h2 = t_list[ihalf_2 - 1]
            prod = h1 * h2
            t_j = prod + prod           # 2 * product
            if ihalf_1 == ihalf_2:
                t_j = _add_const_like(t_j, -1.0, config)  # - 1
            else:
                t_j = t_j - t_list[1]  # - T_2
            t_list[j] = t_j

    # Note (P1): Level alignment of baby-step T_i to T_k's level.
    # The rtlib (chebyshev_impl.c:611-630) mod-switches shallower T_i
    # down to match T_k before the giant-step phase. Without that
    # alignment, the compiler path keeps the same values but tends to
    # burn extra levels later when combining branches.
    depths = compute_chebyshev_depths(k, even)
    tk_depth = depths[k - 1]
    for i in range(1, k):
        if t_list[i] is None or depths[i] < 0:
            continue
        depth_diff = tk_depth - depths[i]
        while depth_diff > 0:
            t_list[i] = t_list[i].mod_switch()
            depth_diff -= 1

    # ------------------------------------------------------------------
    # Step 2: doubling — T_{2k}, T_{4k}, …, T_{2^{m-1}·k}
    # ------------------------------------------------------------------
    t2_list: List = [None] * m
    t2_list[0] = t_list[k - 1]         # T_k
    for i in range(1, m):
        prev = t2_list[i - 1]
        prod = prev * prev
        t2_i = prod + prod              # 2 * T^2
        t2_i = _add_const_like(t2_i, -1.0, config)  # - 1
        t2_list[i] = t2_i

    # ------------------------------------------------------------------
    # Step 3: T_{k(2^m - 1)}
    # ------------------------------------------------------------------
    t2km1 = t2_list[0]                  # start with T_k
    for i in range(1, m):
        prod = t2km1 * t2_list[i]
        t2km1 = prod + prod             # 2 * product
        t2km1 = t2km1 - t2_list[0]     # - T_k

    # ------------------------------------------------------------------
    # Step 4: extend f2 with T_{k(2^m-1)} and evaluate via inner PS
    # ------------------------------------------------------------------
    k2m2k = k * (1 << (m - 1)) - k
    target_len = 2 * k2m2k + k + 1
    while len(f2) < target_len:
        f2.append(0.0)
    f2[target_len - 1] = 1.0

    out = _inner_eval_chebyshev_ps(
        f2, k, m, t_list, t2_list, y, False, depths, config
    )
    out = out - t2km1

    return out


def _inner_eval_chebyshev_ps(coeffs, k, m, t_list, t2_list, y, in_recursion,
                             t_depths, config: Optional[BootstrapConfig] = None):
    """Recursive PS inner evaluation (mirrors Inner_eval_chebyshev_ps)."""
    k2m2k = k * (1 << (m - 1)) - k

    # Divide by T^{k·2^{m-1}}
    tkm = [0.0] * (k2m2k + k + 1)
    tkm[-1] = 1.0

    div_q, div_r = long_div_chebyshev(coeffs, tkm)

    # r2 = r - x^{k(2^{m-1}-1)}
    r2 = list(div_r)
    while len(r2) <= k2m2k:
        r2.append(0.0)
    r2[k2m2k] -= 1.0
    deg_r2 = get_degree_from_coeffs(r2)
    if deg_r2 > 0:
        r2 = r2[: deg_r2 + 1]

    # Divide r2 by q
    divr2_q, divr2_r = long_div_chebyshev(r2, div_q)

    # s2 = remainder + x^{k(2^{m-1}-1)}
    s2_len = max(len(divr2_r), k2m2k + 1)
    s2 = list(divr2_r) + [0.0] * (s2_len - len(divr2_r))
    s2[s2_len - 1] = 1.0

    # Evaluate c = divr2_q at u (using T_1..T_k)
    dc = get_degree_from_coeffs(divr2_q)
    flag_c = False
    cu = None
    if dc >= 1:
        if dc == 1:
            q1 = divr2_q[1]
            cu = _mul_const_like(t_list[0], q1, config) if q1 != 1.0 else t_list[0]
        else:
            cu = _eval_linear_wsum(t_list, divr2_q[1: dc + 1], config)
        cu = _add_const_like(cu, divr2_q[0] / 2.0, config)
        flag_c = True

    # Evaluate qu
    if get_degree_from_coeffs(div_q) > k:
        qu = _inner_eval_chebyshev_ps(
            div_q, k, m - 1, t_list, t2_list, y, True, t_depths, config
        )
    else:
        qu = _eval_quot_or_rem(t_list, div_q, k, True, in_recursion, config)

    # Evaluate su
    if get_degree_from_coeffs(s2) > k:
        su = _inner_eval_chebyshev_ps(
            s2, k, m - 1, t_list, t2_list, y, True, t_depths, config
        )
    else:
        su = _eval_quot_or_rem(t_list, s2, k, False, in_recursion, config)

    # Combine: (T_{2^{m-1}·k} + cu) * qu + su
    t2_m_1 = t2_list[m - 1]
    if flag_c:
        target_depth = t_depths[k - 1] + (m - 1)
        if dc == 1:
            cu_depth = t_depths[0] + (1 if divr2_q[1] != 1.0 else 0)
        else:
            used_depths = [
                t_depths[idx]
                for idx, coeff in enumerate(divr2_q[1: dc + 1])
                if coeff != 0.0 and idx < len(t_depths) and t_depths[idx] >= 0
            ]
            cu_depth = (max(used_depths) if used_depths else t_depths[0]) + 1
        depth_diff = target_depth - cu_depth
        while depth_diff > 0:
            cu = cu.mod_switch()
            depth_diff -= 1
    if flag_c:
        combined = t2_m_1 + cu
    else:
        combined = _add_const_like(t2_m_1, divr2_q[0] / 2.0, config)

    out = combined * qu
    out = out + su
    return out


def _eval_linear_wsum(t_list, weights, config: Optional[BootstrapConfig] = None):
    """Weighted sum: sum weights[i] * t_list[i] (mul_const + accumulate).

    Note (P0a): The rtlib rescales the accumulated result here
    (chebyshev_impl.c:309). We leave the rescale placement to the CKKS
    scale manager because forcing an explicit rescale here conflicts with
    the current compiler pass ordering.
    """
    result = None
    for i, w in enumerate(weights):
        if w == 0.0:
            continue
        if i >= len(t_list) or t_list[i] is None:
            continue
        term = _mul_const_like(t_list[i], w, config)
        if result is None:
            result = term
        else:
            result = result + term
    return result


def _eval_quot_or_rem(
    t_list, quot_rem, k, is_quotient, in_recursion,
    config: Optional[BootstrapConfig] = None,
):
    """Evaluate quotient or remainder at u using baby-step T_1..T_k.

    Mirrors Eval_quot_or_rem in chebyshev_impl.c.
    """
    # Truncate to k elements
    qr = list(quot_rem[: k])
    while len(qr) < k:
        qr.append(0.0)

    t_k_1 = t_list[k - 1]  # T_k
    dg = get_degree_from_coeffs(qr)

    if dg > 0:
        out = _eval_linear_wsum(t_list, qr[1: dg + 1], config)

        if is_quotient:
            if in_recursion:
                quot_last = quot_rem[-1] if len(quot_rem) > 0 else 1.0
                num_adds = int(math.log2(abs(quot_last))) if abs(quot_last) > 0 else 0
                sum_val = t_k_1
                for _ in range(num_adds):
                    sum_val = sum_val + sum_val
                out = out + sum_val
            else:
                # quot_last is always 2 after first division
                out = out + t_k_1
                out = out + t_k_1
        else:
            # remainder: leading coeff is 1
            out = out + t_k_1
    else:
        if is_quotient:
            out = t_k_1
            quot_last = quot_rem[-1] if len(quot_rem) > 0 else 1.0
            end = int(math.log2(abs(quot_last))) if (in_recursion and abs(quot_last) > 0) else int(abs(quot_last))
            for _ in range(end):
                out = out + t_k_1
        else:
            out = t_k_1

    # free term (c0/2)
    out = _add_const_like(out, qr[0] / 2.0, config)
    return out


# =========================================================================
# Double-angle iterations
# =========================================================================

def apply_double_angle(
    x,
    num_iter: Optional[int] = None,
    scalars: Optional[List[float]] = None,
    config: Optional[BootstrapConfig] = None,
):
    """Apply double-angle iterations: x -> 2x^2 + scalar_j, j=1..r.

    Mirrors Apply_double_angle_iterations in bootstrap.c.
    Note (P0b): The rtlib rescales after each iteration's add-scalar
    (bootstrap.c:1513).  The EDSL pipeline scale manager auto-inserts
    rescales after the x*x multiply, so no explicit rescale is needed.
    """
    if scalars is None:
        scalars = list(
            config.double_angle_scalars
            if config is not None
            else DOUBLE_ANGLE_SCALARS
        )
    if num_iter is None:
        num_iter = len(scalars) if config is not None else NUM_DOUBLE_ANGLE
    for j in range(num_iter):
        x = x * x          # x^2
        x = x + x          # 2x^2
        x = _add_const_like(x, scalars[j], config)  # + scalar_j
    return x


def build_bootstrap_evalmod_scalar_manifest(config: BootstrapConfig):
    """Trace the exact scalar-encode order of both expanded EvalMod branches."""
    encoded = []

    class ScalarTrace:
        def _binary(self, other):
            if isinstance(other, (int, float)) and not isinstance(other, bool):
                encoded.append(float(other))
            return self

        def __add__(self, other):
            return self._binary(other)

        def __radd__(self, other):
            return self._binary(other)

        def __sub__(self, other):
            if isinstance(other, (int, float)) and not isinstance(other, bool):
                encoded.append(-float(other))
            return self

        def __rsub__(self, other):
            return self._binary(other)

        def __mul__(self, other):
            return self._binary(other)

        def __rmul__(self, other):
            return self._binary(other)

        def mod_switch(self):
            return self

    traced = eval_chebyshev_ps(
        ScalarTrace(),
        list(config.chebyshev_coefficients),
        config,
    )
    apply_double_angle(
        traced,
        num_iter=len(config.double_angle_scalars),
        scalars=list(config.double_angle_scalars),
        config=config,
    )
    branch = list(encoded)
    both_branches = branch + branch

    def record(values):
        payload = b"".join(
            struct.pack("<dQQ", value, 1, 1) for value in values
        )
        return {
            "count": len(values),
            "values_binary64_hex": [value.hex() for value in values],
            "plaintext_length": 1,
            "scale_degree": 1,
            "ordered_payload_sha256": hashlib.sha256(payload).hexdigest(),
        }

    return {
        "schema_version": (
            "ace.phantom.bootstrap-evalmod-scalar-encodings/1.0.0"
        ),
        "status": "pass",
        "branch_count": 2,
        "single_branch": record(branch),
        "full_program": record(both_branches),
    }


# =========================================================================
# EvalMod  (Chebyshev + double-angle)
# =========================================================================

def eval_approx_mod(x, coeffs: Optional[List[float]] = None,
                    num_double_angle: Optional[int] = None,
                    da_scalars: Optional[List[float]] = None,
                    config: Optional[BootstrapConfig] = None):
    """Approximate modular reduction: PS Chebyshev + double-angle.

    Mirrors Eval_approx_mod in bootstrap.c (UNIFORM_HW_UNDER_192 path,
    non-even polynomial, range [-1,1]).
    """
    out = eval_chebyshev_ps(x, coeffs, config)
    if da_scalars is None:
        da_scalars = list(
            config.double_angle_scalars
            if config is not None
            else DOUBLE_ANGLE_SCALARS
        )
    if num_double_angle is None:
        num_double_angle = len(da_scalars) if config is not None else NUM_DOUBLE_ANGLE
    out = apply_double_angle(
        out,
        num_iter=num_double_angle,
        scalars=da_scalars,
        config=config,
    )
    return out


# =========================================================================
# Full-packed bootstrap primitive decomposition
# =========================================================================

def eval_mod_primitive(x, config: Optional[BootstrapConfig] = None):
    """Full EvalMod decomposition for _bootstrap_eval_mod_primitive.

    Replaces the identity surrogate with PS Chebyshev + double-angle.
    """
    cfg = config
    if hasattr(x, "container") and cfg is None:
        cfg = _compat_bootstrap_config(x)
    return eval_approx_mod(x, config=cfg)


def _configured_slots(
    num_slots: int,
    config: Optional[BootstrapConfig],
    caller: str,
) -> int:
    """Return the configured slot count and reject trace-time mismatches."""
    cfg = _require_bootstrap_config(config, caller)
    slots = int(num_slots) if int(num_slots) > 0 else cfg.slots
    if slots != cfg.slots:
        raise ValueError(
            f"{caller} num_slots={slots} does not match config slots={cfg.slots}"
        )
    return slots


def _primitive_transform_levels(
    config: BootstrapConfig,
) -> tuple[int, int]:
    """Return explicit encode levels for CoeffToSlot and SlotToCoeff."""
    return config.transform_levels


def _primitive_bootstrap_depth(
    enc_budget: int,
    dec_budget: int,
    config: BootstrapConfig,
) -> int:
    """Return the rtlib-equivalent bootstrap depth for the primitive path."""
    coeffs = list(config.chebyshev_coefficients)
    num_double_angle = len(config.double_angle_scalars)
    chebyshev_degree = get_degree_from_coeffs(coeffs)
    ps_k, ps_m = compute_degree_ps(chebyshev_degree)
    approx_mod_depth = (
        int(math.ceil(math.log2(ps_k)))
        + ps_m
        + num_double_angle
    )
    return approx_mod_depth + enc_budget + dec_budget


def _bootstrap_ct_encode_enabled(config: BootstrapConfig) -> bool:
    return bool(config.ct_encode)


def _coeffs_to_slots_factor(
    config: BootstrapConfig,
) -> float:
    """Return the full-packed normalization used by rtlib CoeffToSlot.

    rtlib scales the full-packed CoeffToSlot matrices by:
      1 / ring_degree / K / (q0 / sf)

    """
    return config.coeffs_to_slots_factor


def _bootstrap_num_p(
    config: BootstrapConfig,
) -> int:
    """Return the p-prime count derived from the active trace config."""
    return config.num_p


def _reduce_rotation(index: int, slots: int) -> int:
    """Match rtlib Reduce_rotation for power-of-two slot counts."""
    return int(index) % int(slots)


def _select_layers(log_slots: int, budget: int):
    """Python port of rtlib Select_layers."""
    layers = math.ceil(log_slots / budget)
    rows = log_slots // layers
    rem = log_slots % layers
    dim = rows + (1 if rem != 0 else 0)

    if dim < budget:
        layers -= 1
        rows = log_slots // layers
        rem = log_slots - rows * layers
        dim = rows + (1 if rem != 0 else 0)
        while dim != budget:
            rows -= 1
            rem = log_slots - rows * layers
            dim = rows + (1 if rem != 0 else 0)

    return (layers, rows, rem)


def _get_colls_fft_params(slots: int, level_budget: int = 3, dim1: int = 0):
    """Python port of rtlib Get_colls_fft_params."""
    log_slots = int(math.log2(slots))
    layers_coll, _, rem_coll = _select_layers(log_slots, level_budget)
    flag_rem = 0 if rem_coll == 0 else 1

    num_rot = (1 << (layers_coll + 1)) - 1
    num_rot_rem = (1 << (rem_coll + 1)) - 1

    def choose_dsl_g(num_rotations: int, default_g: int) -> int:
        """Choose a power-of-two giant step that minimizes emitted rotates.

        The rtlib helper picks `g` assuming hoisted/internal fast-rotate
        machinery. The DSL path materializes each high-level `Rotate(...)`
        helper call, so the relevant proxy is:

            nonzero fast-rotates + nonzero outer baby-step rotates

        which is `(g - 1) + (b - 1)` for `b = ceil((num_rot + 1) / g)`.
        """
        best_g = default_g
        target = math.sqrt(num_rotations + 1)
        best_score = (
            (default_g - 1) + (((num_rotations + 1 + default_g - 1) // default_g) - 1),
            abs(default_g - target),
            default_g,
        )
        max_g = num_rotations + 1
        g = 1
        while g <= max_g:
            b = (num_rotations + 1 + g - 1) // g
            score = ((g - 1) + (b - 1), abs(g - target), g)
            if score < best_score:
                best_score = score
                best_g = g
            g <<= 1
        return best_g

    if dim1 == 0 or dim1 > num_rot:
        if num_rot > 7:
            g = choose_dsl_g(num_rot, 1 << (layers_coll // 2 + 2))
        else:
            g = choose_dsl_g(num_rot, 1 << (layers_coll // 2 + 1))
    else:
        g = dim1
    b = (num_rot + 1 + g - 1) // g

    b_rem = 0
    g_rem = 0
    if flag_rem:
        if num_rot_rem > 7:
            g_rem = choose_dsl_g(num_rot_rem, 1 << (rem_coll // 2 + 2))
        else:
            g_rem = choose_dsl_g(num_rot_rem, 1 << (rem_coll // 2 + 1))
        b_rem = (num_rot_rem + 1 + g_rem - 1) // g_rem

    return {
        "level_budget": level_budget,
        "layers_coll": layers_coll,
        "rem_coll": rem_coll,
        "flag_rem": flag_rem,
        "num_rot": num_rot,
        "b": b,
        "g": g,
        "num_rot_rem": num_rot_rem,
        "b_rem": b_rem,
        "g_rem": g_rem,
    }


@lru_cache(maxsize=None)
def _bootstrap_rot_group(slots: int):
    slots4 = 4 * slots
    rot_group = []
    five_pow = 1
    for _ in range(slots):
        rot_group.append(five_pow)
        five_pow = (five_pow * 5) % slots4
    return tuple(rot_group)


@lru_cache(maxsize=None)
def _bootstrap_ksi_pows(slots: int):
    slots4 = 4 * slots
    vals = []
    for idx in range(slots4):
        angle = 2.0 * math.pi * idx / slots4
        vals.append(complex(math.cos(angle), math.sin(angle)))
    vals.append(vals[0])
    return tuple(vals)


def _coeff_one_level(slots: int, encoding: bool):
    """Port of rtlib Coeff_enc_one_level / Coeff_dec_one_level for full-packed mode."""
    ksipows = _bootstrap_ksi_pows(slots)
    rot_group = _bootstrap_rot_group(slots)
    dim = len(ksipows) - 1
    log_slots = int(math.log2(slots))
    coeff = [[0j] * slots for _ in range(3 * log_slots)]

    m = slots
    while m > 1:
        s = int(math.log2(m)) - 1
        coeff_s = coeff[s]
        coeff_logslots = coeff[s + log_slots]
        coeff_2logslots = coeff[s + 2 * log_slots]
        for k in range(0, slots, m):
            lenh = m >> 1
            lenq = m << 2
            for j in range(lenh):
                if encoding:
                    j_twiddle = (lenq - rot_group[j] % lenq) * (dim // lenq)
                    w = ksipows[j_twiddle]
                    coeff_logslots[j + k] = 1
                    coeff_2logslots[j + k] = 1
                    coeff_logslots[j + k + lenh] = -w
                    coeff_s[j + k + lenh] = w
                else:
                    j_twiddle = (rot_group[j] % lenq) * (dim // lenq)
                    w = ksipows[j_twiddle]
                    coeff_logslots[j + k] = 1
                    coeff_2logslots[j + k] = w
                    coeff_logslots[j + k + lenh] = -w
                    coeff_s[j + k + lenh] = 1
        m >>= 1

    return coeff


@lru_cache(maxsize=None)
def _coeff_collapse(slots: int, level_budget: int, encoding: bool):
    """Python port of rtlib Coeff_collapse for full-packed mode (flag=False)."""
    params = _get_colls_fft_params(slots, level_budget)
    layers_coll = params["layers_coll"]
    rem_coll = params["rem_coll"]
    flag_rem = params["flag_rem"]
    num_rot = params["num_rot"]
    num_rot_rem = params["num_rot_rem"]
    dim_coll = level_budget
    log_slots = int(math.log2(slots))
    coeff1 = _coeff_one_level(slots, encoding)

    coeff = []
    for s in range(dim_coll):
        if flag_rem:
            after_remainder = (encoding and s >= 1) or ((not encoding) and s < level_budget - 1)
            stage_num_rot = num_rot if after_remainder else num_rot_rem
        else:
            stage_num_rot = num_rot
        coeff.append([[0j] * slots for _ in range(stage_num_rot)])

    for s in range(dim_coll):
        top = (
            log_slots - (dim_coll - 1 - s) * layers_coll - 1
            if encoding
            else s * layers_coll
        )
        is_rem = flag_rem and ((encoding and s == 0) or ((not encoding) and s == dim_coll - 1))
        end_l = rem_coll if is_rem else layers_coll

        for l in range(end_l):
            if l == 0:
                coeff[s][0] = list(coeff1[top])
                coeff[s][1] = list(coeff1[top + log_slots])
                coeff[s][2] = list(coeff1[top + 2 * log_slots])
                continue

            coeff_temp = [[0j] * slots for _ in range(len(coeff[s]))]
            if encoding:
                t = 0
                for u in range((1 << (l + 1)) - 1):
                    temp_u = coeff[s][u]
                    for k in range(slots):
                        rot_idx = _reduce_rotation(k - (1 << (top - l)), slots)
                        rot_idx2 = _reduce_rotation(k + (1 << (top - l)), slots)
                        coeff_temp[u + t][k] += coeff1[top - l][k] * temp_u[rot_idx]
                        coeff_temp[u + t + 1][k] += coeff1[top - l + log_slots][k] * temp_u[k]
                        coeff_temp[u + t + 2][k] += coeff1[top - l + 2 * log_slots][k] * temp_u[rot_idx2]
                    t += 1
            else:
                width = (1 << (l + 1)) - 1
                for u in range(width):
                    temp_u = coeff[s][u]
                    for k in range(slots):
                        coeff_temp[u][k] += coeff1[top + l][k] * temp_u[k]
                        coeff_temp[u + (1 << l)][k] += coeff1[top + l + log_slots][k] * temp_u[k]
                        coeff_temp[u + (1 << (l + 1))][k] += coeff1[top + l + 2 * log_slots][k] * temp_u[k]
            coeff[s] = coeff_temp

    return tuple(tuple(tuple(row) for row in stage) for stage in coeff)


def _collapsed_fft_stage_plan(config: BootstrapConfig, encoding: bool):
    """Return direct stage plans equivalent to rtlib Rotate_precomp for full-packed mode."""
    slots = config.slots
    level_budget = config.enc_budget if encoding else config.dec_budget
    params = _get_colls_fft_params(slots, level_budget)
    coeff = _coeff_collapse(slots, level_budget, encoding)
    enc_level, dec_level = _primitive_transform_levels(config)

    # Distribute the CoeffToSlot factor across the collapsed stages exactly like
    # rtlib Coeffs2slots_precomp.
    encode_stage_factor = _coeffs_to_slots_factor(config) ** (1.0 / level_budget)
    stages = []

    def encoding_plain_level(stage: int) -> int:
        # Rotate_precomp computes enc_level = level + 1.
        return enc_level + 1 + stage

    def decoding_plain_level(stage: int) -> int:
        # Rotate_precomp computes dec_level = level + level_budget.
        return dec_level + level_budget - stage

    if encoding:
        start = 1 if params["flag_rem"] else 0
        end = level_budget
        for s in range(end - 1, start - 1, -1):
            shift = 1 << ((s - params["flag_rem"]) * params["layers_coll"] + params["rem_coll"])
            stages.append({
                "s": s,
                "num_rot": params["num_rot"],
                "baby_step": params["b"],
                "giant_step": params["g"],
                "is_remainder": False,
                "shift": shift,
                "plain_level": encoding_plain_level(s),
                "diag_scale": encode_stage_factor,
            })
        if params["flag_rem"]:
            stages.append({
                "s": 0,
                "num_rot": params["num_rot_rem"],
                "baby_step": params["b_rem"],
                "giant_step": params["g_rem"],
                "is_remainder": True,
                "shift": 1,
                "plain_level": encoding_plain_level(0),
                "diag_scale": encode_stage_factor,
            })
    else:
        for s in range(0, level_budget - params["flag_rem"]):
            shift = 1 << (s * params["layers_coll"])
            stages.append({
                "s": s,
                "num_rot": params["num_rot"],
                "baby_step": params["b"],
                "giant_step": params["g"],
                "is_remainder": False,
                "shift": shift,
                "plain_level": decoding_plain_level(s),
                "diag_scale": 1.0,
            })
        if params["flag_rem"]:
            s = level_budget - 1
            stages.append({
                "s": s,
                "num_rot": params["num_rot_rem"],
                "baby_step": params["b_rem"],
                "giant_step": params["g_rem"],
                "is_remainder": True,
                "shift": 1 << (s * params["layers_coll"]),
                "plain_level": decoding_plain_level(level_budget - 1),
                "diag_scale": 1.0,
            })

    return coeff, stages


def _group_collapsed_fft_stage_terms(coeff, stage, slots: int):
    """Group a collapsed-FFT stage by rotation and sum diagonals per rotation."""
    s = stage["s"]
    num_rot = stage["num_rot"]
    shift = stage["shift"]
    mid = (num_rot + 1) // 2
    grouped = {}
    order = []
    for dim2 in range(num_rot):
        rot = _reduce_rotation((dim2 - mid + 1) * shift, slots)
        diag = coeff[s][dim2]
        accum = grouped.get(rot)
        if accum is None:
            grouped[rot] = list(diag)
            order.append(rot)
        else:
            for idx, val in enumerate(diag):
                accum[idx] += val
    return [(rot, grouped[rot]) for rot in order]


def _compact_grouped_stage_terms(
    coeff,
    stage,
    slots: int,
    config: BootstrapConfig,
):
    """Return grouped stage terms when they form a contiguous rotation run."""
    if not _bootstrap_ct_encode_enabled(config):
        return None

    grouped_terms = _group_collapsed_fft_stage_terms(coeff, stage, slots)
    if len(grouped_terms) >= stage["num_rot"]:
        return None

    grouped_terms = sorted(grouped_terms, key=lambda item: item[0])
    expected = [
        _reduce_rotation(idx * stage["shift"], slots)
        for idx in range(len(grouped_terms))
    ]
    actual = [rot for rot, _ in grouped_terms]
    if actual != expected:
        return None
    return grouped_terms


def _rotate_plain_vector(values, rotation: int):
    """Match rtlib Rotate_vector semantics for plaintext diagonals."""
    length = len(values)
    rot = int(rotation) % length
    if rot == 0:
        return list(values)
    return [values[(idx + rot) % length] for idx in range(length)]


def _complex_vector_sha256(values) -> str:
    payload = bytearray()
    for value in values:
        payload.extend(
            struct.pack("<dd", float(value.real), float(value.imag))
        )
    return hashlib.sha256(payload).hexdigest()


def _collapsed_fft_stage_payload_manifest(
    config: BootstrapConfig,
    encoding: bool,
):
    """Describe the exact ordered plaintext arrays emitted by one transform."""
    slots = config.slots
    coeff, stages = _collapsed_fft_stage_plan(config, encoding)
    result = []
    for stage_order, stage in enumerate(stages):
        s = stage["s"]
        num_rot = stage["num_rot"]
        baby_step = stage["baby_step"]
        giant_step = stage["giant_step"]
        shift = stage["shift"]
        diag_scale = stage["diag_scale"]
        rot_in = [
            _reduce_rotation(
                (j - ((num_rot + 1) // 2) + 1) * shift, slots
            )
            for j in range(giant_step)
        ]
        compact_terms = _compact_grouped_stage_terms(
            coeff, stage, slots, config
        )
        if compact_terms is None:
            term_count = num_rot
            stage_giant_step = giant_step
        else:
            term_count = len(compact_terms)
            stage_giant_step = max(
                1, (term_count + baby_step - 1) // baby_step
            )
            rot_in = [
                _reduce_rotation(j * shift, slots)
                for j in range(min(stage_giant_step, term_count))
            ]

        payload_hashes = []
        payload_roles = []
        for i in range(baby_step):
            giant = stage_giant_step * i
            giant_rot = giant * shift
            diag_rotation = _reduce_rotation(-giant_rot, slots)
            for j in range(len(rot_in)):
                dim2 = giant + j
                if dim2 >= term_count:
                    continue
                if compact_terms is None:
                    diag = coeff[s][dim2]
                else:
                    _, diag = compact_terms[dim2]
                if diag_scale != 1.0:
                    diag = [value * diag_scale for value in diag]
                if diag_rotation != 0:
                    diag = _rotate_plain_vector(diag, diag_rotation)
                payload_sha256 = _complex_vector_sha256(diag)
                payload_roles.append(
                    {
                        "term_order": len(payload_roles),
                        "baby_step_index": i,
                        "rotation_batch_index": j,
                        "collapsed_term_index": dim2,
                        "input_rotation": rot_in[j],
                        "giant_rotation": _reduce_rotation(
                            giant_rot, slots
                        ),
                        "diagonal_rotation": diag_rotation,
                        "payload_sha256": payload_sha256,
                    }
                )
                payload_hashes.append(payload_sha256)
        result.append(
            {
                "stage_order": stage_order,
                "collapsed_stage": s,
                "plaintext_level": stage["plain_level"],
                "diagonal_scale": diag_scale,
                "diagonal_scale_hex": float(diag_scale).hex(),
                "payload_count": len(payload_hashes),
                "ordered_payload_sha256": payload_hashes,
                "payload_roles": payload_roles,
            }
        )
    return result


def build_bootstrap_transform_payload_manifest(config: BootstrapConfig):
    """Return semantic roles for every emitted transform plaintext payload."""
    encoding = _collapsed_fft_stage_payload_manifest(config, True)
    decoding = _collapsed_fft_stage_payload_manifest(config, False)

    def flatten(stages):
        return [
            payload
            for stage in stages
            for payload in stage["ordered_payload_sha256"]
        ]

    encoding_payloads = flatten(encoding)
    decoding_payloads = flatten(decoding)
    all_payloads = encoding_payloads + decoding_payloads
    encoding_scale_product = math.prod(
        stage["diagonal_scale"] for stage in encoding
    )
    transform_gain = 2 * config.slots
    component_input_gain = transform_gain * encoding_scale_product
    return {
        "schema_version": (
            "ace.phantom.bootstrap-transform-payload-semantics/1.0.0"
        ),
        "status": "pass",
        "context": {
            "polynomial_degree": config.poly_degree,
            "logical_slots": config.slots,
            "mul_level": config.mul_level,
            "first_prime_bits": config.first_prime_bits,
            "scaling_factor_bits": config.scaling_factor_bits,
            "q_part_count": config.q_parts,
            "encode_transform_budget": config.enc_budget,
            "decode_transform_budget": config.dec_budget,
            "overflow_bound": config.eval_sin_upper_bound_k,
            "restoration_factor": config.post_scale,
        },
        "normalization": {
            "configured_coefficients_to_slots_factor": (
                config.coeffs_to_slots_factor
            ),
            "configured_coefficients_to_slots_factor_hex": (
                config.coeffs_to_slots_factor.hex()
            ),
            "encoding_stage_scale_product": encoding_scale_product,
            "encoding_stage_scale_product_hex": (
                encoding_scale_product.hex()
            ),
            "nominal_full_packed_transform_gain": transform_gain,
            "nominal_component_input_gain": component_input_gain,
            "nominal_component_input_gain_hex": component_input_gain.hex(),
        },
        "coefficients_to_slots": {
            "stages": encoding,
            "payload_count": len(encoding_payloads),
            "ordered_payload_set_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(value) for value in encoding_payloads)
            ).hexdigest(),
        },
        "slots_to_coefficients": {
            "stages": decoding,
            "payload_count": len(decoding_payloads),
            "ordered_payload_set_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(value) for value in decoding_payloads)
            ).hexdigest(),
        },
        "constant_manifest_order": {
            "payload_count": len(all_payloads),
            "ordered_payload_sha256": all_payloads,
            "ordered_payload_set_sha256": hashlib.sha256(
                b"".join(bytes.fromhex(value) for value in all_payloads)
            ).hexdigest(),
        },
    }


def _apply_collapsed_fft_transform(
    x,
    config: BootstrapConfig,
    encoding: bool,
):
    """Apply the full-packed collapsed-FFT transform with CKKS ops."""
    slots = config.slots
    coeff, stages = _collapsed_fft_stage_plan(config, encoding)

    # Cleartext fallback: bootstrap is message-preserving, so use identity.
    if not hasattr(x, "container"):
        return x.__class__(x.vals)

    result = x
    for stage in stages:
        s = stage["s"]
        num_rot = stage["num_rot"]
        baby_step = stage["baby_step"]
        giant_step = stage["giant_step"]
        plain_level = stage["plain_level"]
        diag_scale = stage["diag_scale"]
        shift = stage["shift"]
        rot_in = []
        for j in range(giant_step):
            idx = (j - ((num_rot + 1) // 2) + 1) * shift
            rot = _reduce_rotation(idx, slots)
            rot_in.append(rot)

        compact_terms = _compact_grouped_stage_terms(coeff, stage, slots, config)
        if compact_terms is None:
            term_count = num_rot
            stage_giant_step = giant_step
        else:
            term_count = len(compact_terms)
            stage_giant_step = max(1, (term_count + baby_step - 1) // baby_step)
            rot_in = [
                _reduce_rotation(j * shift, slots)
                for j in range(min(stage_giant_step, term_count))
            ]

        fast_rot_batch = result.rotate_batch(rot_in)
        fast_rot = [fast_rot_batch[j] for j in range(len(rot_in))]

        stage_acc = None
        for i in range(baby_step):
            giant = stage_giant_step * i
            giant_rot = giant * shift
            diag_rotation = _reduce_rotation(-giant_rot, slots)
            inner = None
            for j in range(len(rot_in)):
                dim2 = giant + j
                if dim2 >= term_count:
                    continue
                if compact_terms is None:
                    diag = coeff[s][dim2]
                else:
                    _, diag = compact_terms[dim2]
                if diag_scale != 1.0:
                    diag = [val * diag_scale for val in diag]
                if diag_rotation != 0:
                    diag = _rotate_plain_vector(diag, diag_rotation)
                plain = _encode_plain_vector_like(
                    x, diag, scale_degree=1, level=plain_level, config=config
                )
                term = _mul_plain_lazy_rescale(fast_rot[j], plain)
                inner = term if inner is None else inner + term
            if inner is None:
                continue
            if i == 0:
                stage_acc = inner
            else:
                rot = _reduce_rotation(giant_rot, slots)
                moved = inner if rot == 0 else inner.rotate(rot)
                stage_acc = moved if stage_acc is None else stage_acc + moved
        # Match the rtlib transform structure more closely: accumulate each
        # stage at the multiplied scale and only rescale when a following stage
        # still needs to consume this result.
        result = stage_acc.rescale()
    return result


def _sample_indices(length: int):
    candidates = [0, 1, 2, length // 2, length - 1]
    seen = []
    for idx in candidates:
        if 0 <= idx < length and idx not in seen:
            seen.append(idx)
    return seen


def _fnv1a64_bytes(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _digest_complex_row(row) -> str:
    payload = bytearray()
    for val in row:
        payload.extend(struct.pack("<dd", float(val.real), float(val.imag)))
    return f"{_fnv1a64_bytes(payload):016x}"


def _complex_pair(val):
    return [float(val.real), float(val.imag)]


def _summarize_coeff_collapse(coeff):
    stage_summaries = []
    for stage_idx, stage in enumerate(coeff):
        rows = []
        for row_idx, row in enumerate(stage):
            sample_idxs = _sample_indices(len(row))
            rows.append(
                {
                    "row": row_idx,
                    "digest": _digest_complex_row(row),
                    "samples": {
                        str(idx): _complex_pair(row[idx]) for idx in sample_idxs
                    },
                }
            )
        stage_summaries.append(
            {
                "stage": stage_idx,
                "rows": rows,
            }
        )
    return stage_summaries


def get_bootstrap_precompute_summary(slots: int, level_budget: int = 3):
    """Return a compact summary of the full-packed collapsed-FFT precompute."""
    enc_params = _get_colls_fft_params(slots, level_budget)
    dec_params = _get_colls_fft_params(slots, level_budget)
    enc_coeff = _coeff_collapse(slots, level_budget, True)
    dec_coeff = _coeff_collapse(slots, level_budget, False)
    return {
        "slots": int(slots),
        "level_budget": int(level_budget),
        "encode_params": enc_params,
        "decode_params": dec_params,
        "encode_coeff_collapse": _summarize_coeff_collapse(enc_coeff),
        "decode_coeff_collapse": _summarize_coeff_collapse(dec_coeff),
    }


def _encode_plain_vector_like(
    x,
    values,
    scale_degree: int = 1,
    level: int = 0,
    config: Optional[BootstrapConfig] = None,
):
    """Encode a complex plaintext vector as a CKKS plaintext AIRValue."""
    if not hasattr(x, "container"):
        return x.__class__(values)
    cfg = _require_bootstrap_config(config, "_encode_plain_vector_like")

    from .air_value import AIRValue

    container = x.container
    array_node = container.new_array_const(list(values))
    num_p = _bootstrap_num_p(cfg)
    if hasattr(container, "new_ckks_encode_complex"):
        plain_node = container.new_ckks_encode_complex(
            array_node,
            len(values),
            scale_degree,
            level,
            num_p,
            True,
        )
    else:
        plain_node = container.new_ckks_encode(
            array_node, len(values), scale_degree, level, True
        )
    return AIRValue(plain_node, container, domain=getattr(x, "domain", None))


def _mul_plain_lazy_rescale(ct, plain):
    """Multiply ct*plain and defer compiler auto-rescale for this node."""
    if not hasattr(ct, "container"):
        return ct * plain

    from .air_value import AIRValue

    container = ct.container
    mul_node = container.new_ckks_mul(ct.value, plain.value)
    if hasattr(mul_node, "set_u32_attr"):
        mul_node.set_u32_attr("skip_auto_rescale", 1)
    return AIRValue(mul_node, container, domain=getattr(ct, "domain", None))


def _encode_scalar_like(x, value, scale_degree: int = 1, level: int = 0):
    """Encode a scalar; level zero lets scale management bind it to the lhs."""
    if not hasattr(x, "container"):
        return value

    from .air_value import AIRValue

    container = x.container
    if isinstance(value, int):
        const_node = container.new_intconst(int(value))
    else:
        const_node = container.new_floatconst(float(value))
    plain_node = container.new_ckks_encode(const_node, 1, scale_degree, level)
    return AIRValue(plain_node, container, domain=getattr(x, "domain", None))


def _add_const_like(x, value, config: Optional[BootstrapConfig] = None):
    if not hasattr(x, "container"):
        return x + value
    _require_bootstrap_config(config, "_add_const_like")
    return x + _encode_scalar_like(
        x, value, scale_degree=1, level=0
    )


def _mul_const_like(x, value, config: Optional[BootstrapConfig] = None):
    if not hasattr(x, "container"):
        return x * value
    _require_bootstrap_config(config, "_mul_const_like")
    return x * _encode_scalar_like(
        x, value, scale_degree=1, level=0
    )


def _mul_by_power_of_two(x, value: float):
    """Multiply by an exact power of two using ciphertext doubling only."""
    ivalue = int(round(float(value)))
    if ivalue <= 0 or ivalue & (ivalue - 1):
        return x * value

    out = x
    doublings = int(math.log2(ivalue))
    for _ in range(doublings):
        out = out + out
    return out


def coeffs_to_slots_primitive(
    x,
    num_slots: int = 0,
    config: Optional[BootstrapConfig] = None,
):
    """CoeffToSlot decomposition using full-packed collapsed FFT semantics."""
    if not hasattr(x, "container") and config is None:
        return x.__class__(x.vals)
    cfg = config if config is not None else _compat_bootstrap_config(x, num_slots)
    _configured_slots(num_slots, cfg, "coeffs_to_slots_primitive")
    return _apply_collapsed_fft_transform(
        x, _require_bootstrap_config(cfg, "coeffs_to_slots_primitive"), encoding=True
    )


def slots_to_coeffs_primitive(
    x,
    num_slots: int = 0,
    config: Optional[BootstrapConfig] = None,
):
    """SlotToCoeff decomposition using full-packed collapsed FFT semantics."""
    if not hasattr(x, "container") and config is None:
        return x.__class__(x.vals)
    cfg = config if config is not None else _compat_bootstrap_config(x, num_slots)
    _configured_slots(num_slots, cfg, "slots_to_coeffs_primitive")
    return _apply_collapsed_fft_transform(
        x, _require_bootstrap_config(cfg, "slots_to_coeffs_primitive"), encoding=False
    )


def fullpacked_bootstrap_primitive(ct, m_by_4: Optional[int] = None,
                                   three_m_by_4: Optional[int] = None,
                                   post_scale: float = None,
                                   clear_imag: bool = False,
                                   config: Optional[BootstrapConfig] = None):
    """Full-packed bootstrap branch decomposition.

    Implements the full-packed path from Eval_bootstrap (slots == m/4):
      1. CoeffToSlot (DFT)
      2. Conjugate split: real = ct + conj, imag = ct - conj
      3. Mul_by_monomial on imag (3m/4)
      4. EvalMod on real and imag independently
      5. Mul_by_monomial on imag result (m/4)
      6. Recombine: real + imag
      7. SlotToCoeff (IDFT)
      8. Post-processing (clear_imag or standard post-scale)

    Note on rescale/level:
        The rtlib rescales at several points inside this flow (conjugate
        split, recombine, final).  The EDSL pipeline's scale manager
        auto-inserts rescales after multiplications.  Explicit rescale
        nodes are not emitted here to avoid conflicting with auto scale
        management.

    Args:
        ct: AIRValue ciphertext to bootstrap.
        m_by_4: Optional consistency check for config.slots.
        three_m_by_4: Optional consistency check for 3 * config.slots.
        post_scale: Optional consistency check for config.post_scale.
        clear_imag: If True, use conjugate-based imag clearing (P2).
        config: Explicit trace-time bootstrap/FHE metadata.

    Returns:
        AIRValue -- bootstrapped ciphertext.
    """
    if not hasattr(ct, "container"):
        # Cleartext fallback: bootstrap is message-preserving.
        return ct.__class__(ct.vals)

    cfg = (
        config
        if config is not None
        else _compat_bootstrap_config(ct, int(m_by_4) if m_by_4 is not None else 0)
    )
    if m_by_4 is None:
        m_by_4 = cfg.slots
    elif int(m_by_4) != cfg.slots:
        raise ValueError(
            f"fullpacked_bootstrap_primitive m_by_4={m_by_4} "
            f"does not match config slots={cfg.slots}"
        )
    if three_m_by_4 is None:
        three_m_by_4 = 3 * cfg.slots
    elif int(three_m_by_4) != 3 * cfg.slots:
        raise ValueError(
            f"fullpacked_bootstrap_primitive three_m_by_4={three_m_by_4} "
            f"does not match 3 * config slots={3 * cfg.slots}"
        )
    if post_scale is None:
        post_scale = cfg.post_scale
    elif float(post_scale) != float(cfg.post_scale):
        raise ValueError(
            f"fullpacked_bootstrap_primitive post_scale={post_scale} "
            f"does not match config post_scale={cfg.post_scale}"
        )
    deg = cfg.post_scale_degree

    # Step 1: CoeffToSlot
    enc = coeffs_to_slots_primitive(ct, num_slots=m_by_4, config=cfg)

    # Step 2: Conjugate split (full-packed)
    conj = enc.conjugate()
    real_part = enc + conj          # 2 * Re(enc)
    imag_part = enc - conj          # 2i * Im(enc)
    imag_part = imag_part.mul_mono(three_m_by_4)

    # Step 3: Dual EvalMod
    real_evmod = eval_approx_mod(real_part, config=cfg)
    imag_evmod = eval_approx_mod(imag_part, config=cfg)

    # Step 4: Recombine
    imag_evmod = imag_evmod.mul_mono(m_by_4)
    combined = real_evmod + imag_evmod

    # Step 5: SlotToCoeff
    out = slots_to_coeffs_primitive(combined, num_slots=m_by_4, config=cfg)

    # Step 6: Post-processing (P2)
    if clear_imag and deg >= 1:
        # Clear imaginary part via conjugate: out = (out + conj(out)) * 2^(deg-1)
        out_conj = out.conjugate()
        out = out + out_conj
        scale_val = float(2 ** (deg - 1))
        out = _mul_by_power_of_two(out, scale_val)
    else:
        # Standard post-scale: out = out * 2^deg
        out = _mul_by_power_of_two(out, post_scale)

    return out
