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

import math
import os
import struct
from functools import lru_cache
from typing import List, Optional

from .bootstrap_math import (
    CHEBYSHEV_COEFFICIENTS,
    DOUBLE_ANGLE_SCALARS,
    NUM_DOUBLE_ANGLE,
    BOOTSTRAP_POST_SCALE,
    BOOTSTRAP_POST_SCALE_DEG,
    EVAL_SIN_UPPER_BOUND_K,
    compute_degree_ps,
    compute_chebyshev_depths,
    get_degree_from_coeffs,
    is_even_poly,
    long_div_chebyshev,
)


# =========================================================================
# Paterson-Stockmeyer Chebyshev evaluation
# =========================================================================

def eval_chebyshev_ps(x, coeffs: Optional[List[float]] = None):
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
        coeffs = list(CHEBYSHEV_COEFFICIENTS)

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
            t_j = _add_const_like(t_j, -1.0)  # - 1
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
                t_j = _add_const_like(t_j, -1.0)  # - 1
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
        t2_i = _add_const_like(t2_i, -1.0)  # - 1
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

    out = _inner_eval_chebyshev_ps(f2, k, m, t_list, t2_list, y, False, depths)
    out = out - t2km1

    return out


def _inner_eval_chebyshev_ps(coeffs, k, m, t_list, t2_list, y, in_recursion,
                             t_depths):
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
            cu = _mul_const_like(t_list[0], q1) if q1 != 1.0 else t_list[0]
        else:
            cu = _eval_linear_wsum(t_list, divr2_q[1: dc + 1])
        cu = _add_const_like(cu, divr2_q[0] / 2.0)
        flag_c = True

    # Evaluate qu
    if get_degree_from_coeffs(div_q) > k:
        qu = _inner_eval_chebyshev_ps(
            div_q, k, m - 1, t_list, t2_list, y, True, t_depths
        )
    else:
        qu = _eval_quot_or_rem(t_list, div_q, k, True, in_recursion)

    # Evaluate su
    if get_degree_from_coeffs(s2) > k:
        su = _inner_eval_chebyshev_ps(
            s2, k, m - 1, t_list, t2_list, y, True, t_depths
        )
    else:
        su = _eval_quot_or_rem(t_list, s2, k, False, in_recursion)

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
        combined = _add_const_like(t2_m_1, divr2_q[0] / 2.0)

    out = combined * qu
    out = out + su
    return out


def _eval_linear_wsum(t_list, weights):
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
        term = _mul_const_like(t_list[i], w)
        if result is None:
            result = term
        else:
            result = result + term
    return result


def _eval_quot_or_rem(t_list, quot_rem, k, is_quotient, in_recursion):
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
        out = _eval_linear_wsum(t_list, qr[1: dg + 1])

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
    out = _add_const_like(out, qr[0] / 2.0)
    return out


# =========================================================================
# Double-angle iterations
# =========================================================================

def apply_double_angle(x, num_iter: int = NUM_DOUBLE_ANGLE,
                       scalars: Optional[List[float]] = None):
    """Apply double-angle iterations: x -> 2x^2 + scalar_j, j=1..r.

    Mirrors Apply_double_angle_iterations in bootstrap.c.
    Note (P0b): The rtlib rescales after each iteration's add-scalar
    (bootstrap.c:1513).  The EDSL pipeline scale manager auto-inserts
    rescales after the x*x multiply, so no explicit rescale is needed.
    """
    if scalars is None:
        scalars = list(DOUBLE_ANGLE_SCALARS)
    for j in range(num_iter):
        x = x * x          # x^2
        x = x + x          # 2x^2
        x = _add_const_like(x, scalars[j])  # + scalar_j
    return x


# =========================================================================
# EvalMod  (Chebyshev + double-angle)
# =========================================================================

def eval_approx_mod(x, coeffs: Optional[List[float]] = None,
                    num_double_angle: int = NUM_DOUBLE_ANGLE,
                    da_scalars: Optional[List[float]] = None):
    """Approximate modular reduction: PS Chebyshev + double-angle.

    Mirrors Eval_approx_mod in bootstrap.c (UNIFORM_HW_UNDER_192 path,
    non-even polynomial, range [-1,1]).
    """
    out = eval_chebyshev_ps(x, coeffs)
    out = apply_double_angle(out, num_iter=num_double_angle, scalars=da_scalars)
    return out


# =========================================================================
# Full-packed bootstrap primitive decomposition
# =========================================================================

def eval_mod_primitive(x):
    """Full EvalMod decomposition for _bootstrap_eval_mod_primitive.

    Replaces the identity surrogate with PS Chebyshev + double-angle.
    """
    return eval_approx_mod(x)


def _default_demo_slots(num_slots: int) -> int:
    """Return the slot count used by the primitive bootstrap demo."""
    if num_slots > 0:
        return int(num_slots)
    raw = os.environ.get("ACE_BOOTSTRAP_POLY_DEGREE", "").strip()
    if raw:
        try:
            degree = int(raw)
            if degree > 0:
                return degree // 2
        except ValueError:
            pass
    # Default to the full-packed slot count for the example's N=16384 context.
    return 8192


def _primitive_transform_levels():
    """Return demo encode levels for CoeffToSlot and SlotToCoeff.

    Mirror the current bootstrap demo configuration:
      mul_level ~= ACE_BOOTSTRAP_MUL_LEVEL (default 26)
      enc_budget = 3
      dec_budget = 3
      approx_mod_depth = 9  (UNIFORM_HW_192 path: PS depth 6 + 3 double-angle)
    """
    raw = os.environ.get("ACE_BOOTSTRAP_MUL_LEVEL", "").strip()
    try:
        mul_level = int(raw) if raw else 26
    except ValueError:
        mul_level = 26

    level_0 = mul_level
    enc_budget = 3
    dec_budget = 3
    approx_mod_depth = 9
    bts_depth = approx_mod_depth + enc_budget + dec_budget

    enc_level = max(1, level_0 - enc_budget)
    dec_level = max(1, level_0 - bts_depth)
    return enc_level, dec_level


def _bootstrap_const_level() -> int:
    """Use the high bootstrap plaintext level for scalar constants too."""
    enc_level, _ = _primitive_transform_levels()
    return enc_level


def _coeffs_to_slots_factor(slots: int) -> float:
    """Return the full-packed normalization used by rtlib CoeffToSlot.

    rtlib scales the full-packed CoeffToSlot matrices by:
      1 / ring_degree / K / (q0 / sf)

    For the full-packed path, slots = ring_degree / 2.
    """
    ring_degree = slots * 2
    return 1.0 / ring_degree / EVAL_SIN_UPPER_BOUND_K / float(BOOTSTRAP_POST_SCALE)


def _bootstrap_num_p(slots: int) -> int:
    """Return the rtlib p-prime count for the current demo/bootstrap context."""
    if slots >= 32768:
        return 11
    return 9


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

    if dim1 == 0 or dim1 > num_rot:
        if num_rot > 7:
            g = 1 << (layers_coll // 2 + 2)
        else:
            g = 1 << (layers_coll // 2 + 1)
    else:
        g = dim1
    b = (num_rot + 1) // g

    b_rem = 0
    g_rem = 0
    if flag_rem:
        if num_rot_rem > 7:
            g_rem = 1 << (rem_coll // 2 + 2)
        else:
            g_rem = 1 << (rem_coll // 2 + 1)
        b_rem = (num_rot_rem + 1) // g_rem

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


def _collapsed_fft_stage_plan(slots: int, encoding: bool, level_budget: int = 3):
    """Return direct stage plans equivalent to rtlib Rotate_precomp for full-packed mode."""
    params = _get_colls_fft_params(slots, level_budget)
    coeff = _coeff_collapse(slots, level_budget, encoding)
    enc_level, dec_level = _primitive_transform_levels()

    # Distribute the CoeffToSlot factor across the collapsed stages exactly like
    # rtlib Coeffs2slots_precomp.
    encode_stage_factor = _coeffs_to_slots_factor(slots) ** (1.0 / level_budget)
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


def _rotate_plain_vector(values, rotation: int):
    """Match rtlib Rotate_vector semantics for plaintext diagonals."""
    length = len(values)
    rot = int(rotation) % length
    if rot == 0:
        return list(values)
    return [values[(idx + rot) % length] for idx in range(length)]


def _apply_collapsed_fft_transform(x, slots: int, encoding: bool):
    """Apply the full-packed collapsed-FFT transform with CKKS ops."""
    coeff, stages = _collapsed_fft_stage_plan(slots, encoding)

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

        fast_rot = [
            result if rot == 0 else result.rotate(rot)
            for rot in rot_in
        ]

        stage_acc = None
        for i in range(baby_step):
            giant = giant_step * i
            giant_rot = giant * shift
            diag_rotation = _reduce_rotation(-giant_rot, slots)
            inner = None
            for j in range(giant_step):
                dim2 = giant + j
                if dim2 == num_rot:
                    continue
                diag = coeff[s][dim2]
                if diag_scale != 1.0:
                    diag = [val * diag_scale for val in diag]
                if diag_rotation != 0:
                    diag = _rotate_plain_vector(diag, diag_rotation)
                plain = _encode_plain_vector_like(
                    x, diag, scale_degree=1, level=plain_level
                )
                term = fast_rot[j] * plain
                inner = term if inner is None else inner + term
            if inner is None:
                continue
            if i == 0:
                stage_acc = inner
            else:
                rot = _reduce_rotation(giant_rot, slots)
                moved = inner if rot == 0 else inner.rotate(rot)
                stage_acc = moved if stage_acc is None else stage_acc + moved
        result = stage_acc
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


def _encode_plain_vector_like(x, values, scale_degree: int = 1, level: int = 0):
    """Encode a complex plaintext vector as a CKKS plaintext AIRValue."""
    if not hasattr(x, "container"):
        return x.__class__(values)

    from .air_value import AIRValue

    container = x.container
    array_node = container.new_array_const(list(values))
    num_p = _bootstrap_num_p(len(values))
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


def _encode_scalar_like(x, value, scale_degree: int = 1, level: int = 0):
    """Encode a scalar as CKKS plaintext with explicit level/scale."""
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


def _add_const_like(x, value):
    if not hasattr(x, "container"):
        return x + value
    return x + _encode_scalar_like(x, value, scale_degree=1, level=_bootstrap_const_level())


def _mul_const_like(x, value):
    if not hasattr(x, "container"):
        return x * value
    return x * _encode_scalar_like(x, value, scale_degree=1, level=_bootstrap_const_level())


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


def coeffs_to_slots_primitive(x, num_slots: int = 0):
    """CoeffToSlot decomposition using full-packed collapsed FFT semantics."""
    slots = _default_demo_slots(num_slots)
    return _apply_collapsed_fft_transform(x, slots, encoding=True)


def slots_to_coeffs_primitive(x, num_slots: int = 0):
    """SlotToCoeff decomposition using full-packed collapsed FFT semantics."""
    slots = _default_demo_slots(num_slots)
    return _apply_collapsed_fft_transform(x, slots, encoding=False)


def fullpacked_bootstrap_primitive(ct, m_by_4: int = 8192,
                                   three_m_by_4: int = 24576,
                                   post_scale: float = None,
                                   clear_imag: bool = False):
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
        m_by_4: m/4 = ring_degree/2 (default 8192 for N=16384).
        three_m_by_4: 3m/4 (default 24576).
        post_scale: Post-scale value (default: BOOTSTRAP_POST_SCALE).
        clear_imag: If True, use conjugate-based imag clearing (P2).

    Returns:
        AIRValue -- bootstrapped ciphertext.
    """
    if not hasattr(ct, "container"):
        # Cleartext fallback: bootstrap is message-preserving.
        return ct.__class__(ct.vals)

    if post_scale is None:
        post_scale = float(BOOTSTRAP_POST_SCALE)
    deg = BOOTSTRAP_POST_SCALE_DEG

    # Step 1: CoeffToSlot
    enc = coeffs_to_slots_primitive(ct, num_slots=m_by_4)

    # Step 2: Conjugate split (full-packed)
    conj = enc.conjugate()
    real_part = enc + conj          # 2 * Re(enc)
    imag_part = enc - conj          # 2i * Im(enc)
    imag_part = imag_part.mul_mono(three_m_by_4)

    # Step 3: Dual EvalMod
    real_evmod = eval_approx_mod(real_part)
    imag_evmod = eval_approx_mod(imag_part)

    # Step 4: Recombine
    imag_evmod = imag_evmod.mul_mono(m_by_4)
    combined = real_evmod + imag_evmod

    # Step 5: SlotToCoeff
    out = slots_to_coeffs_primitive(combined, num_slots=m_by_4)

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
