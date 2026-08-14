#!/usr/bin/env python3
"""Deterministic CKKS EvalMod scale/value diagnostic.

This is a CPU-only model of the expanded degree-54 Chebyshev/Paterson-
Stockmeyer EvalMod graph emitted by ``bootstrap_decomposition.py``.  It tracks
the decoded value and raw CKKS scale at every operation boundary for three
semantics:

* ``phantom_current``: Phantom's below-2^bits modulus chain, exact-prime
  rescale metadata, and scalar plaintexts encoded at the nominal scale.
* ``phantom_corrected``: ANT's balanced modulus chain, exact-prime rescale
  metadata, and scalar plaintexts encoded at the ciphertext's current raw
  scale.
* ``ant_nominal_oracle``: the compiler/ANT logical oracle in which each
  rescale removes one nominal scale degree and scalar addition uses the
  ciphertext scale.  It is a schedule oracle, not a bit-exact CRT simulator.

The zero input is intentional: it isolates scale semantics from ciphertext
noise and makes the first value error the ``-1`` addition in T2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import runpy
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_MATH_PATH = REPO_ROOT / "ace_edsl/edsl/core/bootstrap_math.py"

# Load this pure-Python source directly.  Importing through ace_edsl.edsl would
# initialize the full EDSL and make this diagnostic depend on optional runtime
# packages that are irrelevant to the numerical model.
_BOOTSTRAP_MATH = runpy.run_path(str(BOOTSTRAP_MATH_PATH))
CHEBYSHEV_COEFFICIENTS = tuple(
    float(value) for value in _BOOTSTRAP_MATH["CHEBYSHEV_COEFFICIENTS"]
)
DOUBLE_ANGLE_SCALARS = tuple(
    float(value) for value in _BOOTSTRAP_MATH["DOUBLE_ANGLE_SCALARS"]
)
compute_chebyshev_depths = _BOOTSTRAP_MATH["compute_chebyshev_depths"]
compute_degree_ps = _BOOTSTRAP_MATH["compute_degree_ps"]
get_degree_from_coeffs = _BOOTSTRAP_MATH["get_degree_from_coeffs"]
is_even_poly = _BOOTSTRAP_MATH["is_even_poly"]
long_div_chebyshev = _BOOTSTRAP_MATH["long_div_chebyshev"]


SCHEMA_VERSION = "ace.phantom.evalmod-scale-diagnostic/1.0.0"


class DiagnosticError(ValueError):
    """Raised when the requested model cannot represent the configuration."""


@dataclass(frozen=True)
class DiagnosticConfig:
    poly_degree: int = 65536
    data_q_count: int = 26
    special_p_count: int = 9
    first_modulus_bits: int = 60
    scaling_modulus_bits: int = 56
    coeff_to_slots_rescales: int = 3

    @property
    def nominal_scale(self) -> float:
        return float(1 << self.scaling_modulus_bits)

    def validate(self) -> None:
        if self.poly_degree < 2 or self.poly_degree & (self.poly_degree - 1):
            raise DiagnosticError("polynomial degree must be a power of two")
        if self.data_q_count < 13:
            raise DiagnosticError(
                "EvalMod diagnostic requires at least 13 data-Q primes"
            )
        if self.special_p_count < 1:
            raise DiagnosticError("special-P count must be positive")
        for name, bits in (
            ("first modulus", self.first_modulus_bits),
            ("scaling modulus", self.scaling_modulus_bits),
        ):
            if bits < 2 or bits > 60:
                raise DiagnosticError(f"{name} bits must be in [2, 60]")
        if self.coeff_to_slots_rescales < 0:
            raise DiagnosticError("CoeffToSlots rescale count cannot be negative")
        # The expanded degree-54 graph consumes nine levels.  q[0] must remain.
        if self.coeff_to_slots_rescales + 9 >= self.data_q_count:
            raise DiagnosticError("data-Q chain is too short for EvalMod")


@dataclass(frozen=True)
class Semantics:
    name: str
    modulus_policy: str
    scalar_policy: str
    rescale_policy: str
    description: str


@dataclass(frozen=True)
class State:
    value: float
    raw_scale: float
    dropped_q_count: int

    def record(self, q_count: int) -> dict[str, Any]:
        return {
            "value": self.value,
            "value_hex": self.value.hex(),
            "raw_scale": self.raw_scale,
            "raw_scale_hex": self.raw_scale.hex(),
            "dropped_q_count": self.dropped_q_count,
            "active_q_count": q_count - self.dropped_q_count,
        }


SEMANTICS = (
    Semantics(
        name="phantom_current",
        modulus_policy="phantom_all_below",
        scalar_policy="nominal",
        rescale_policy="exact_prime",
        description=(
            "Current adapter: Phantom CoeffModulus::Create all-below primes, "
            "S/q rescale metadata, and scalar plaintext scale Delta."
        ),
    ),
    Semantics(
        name="phantom_corrected",
        modulus_policy="ant_balanced",
        scalar_policy="ciphertext_raw",
        rescale_policy="exact_prime",
        description=(
            "Candidate repair: ANT-balanced Q, S/q rescale metadata, and "
            "scalar plaintexts encoded at the lhs ciphertext raw scale."
        ),
    ),
    Semantics(
        name="ant_nominal_oracle",
        modulus_policy="ant_balanced",
        scalar_policy="ciphertext_raw",
        rescale_policy="nominal",
        description=(
            "Logical ANT/compiler oracle: each rescale removes one nominal "
            "Delta and scalar addition uses the ciphertext raw scale."
        ),
    ),
)


def _is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin primality test for unsigned 64-bit values."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime

    odd_part = value - 1
    power_of_two = 0
    while odd_part & 1 == 0:
        power_of_two += 1
        odd_part >>= 1

    # This witness set is deterministic for every n < 2^64.
    for witness in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if witness % value == 0:
            continue
        result = pow(witness, odd_part, value)
        if result in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            result = result * result % value
            if result == value - 1:
                break
        else:
            return False
    return True


def _phantom_primes_below(poly_degree: int, bits: int, count: int) -> list[int]:
    """Match Phantom get_primes: descending primes strictly below 2^bits."""
    order = 2 * poly_degree
    candidate = (1 << bits) - order + 1
    result: list[int] = []
    lower_bound = 1 << (bits - 1)
    while candidate > lower_bound and len(result) < count:
        if _is_prime(candidate):
            result.append(candidate)
        candidate -= order
    if len(result) != count:
        raise DiagnosticError("Phantom prime search exhausted its bit range")
    return result


def phantom_all_below_q(config: DiagnosticConfig) -> list[int]:
    """Match current Q assignment, including q0 sharing the 60-bit P pool."""
    first_pool = _phantom_primes_below(
        config.poly_degree,
        config.first_modulus_bits,
        1 + config.special_p_count,
    )
    scaling_pool = _phantom_primes_below(
        config.poly_degree,
        config.scaling_modulus_bits,
        config.data_q_count - 1,
    )
    return [first_pool[-1], *reversed(scaling_pool)]


def _ant_first_prime(poly_degree: int, bits: int) -> int:
    order = 2 * poly_degree
    candidate = (1 << bits) + order + 1
    while not _is_prime(candidate):
        candidate += order
    return candidate


def _ant_previous_prime(value: int, order: int) -> int:
    candidate = value - order
    while not _is_prime(candidate):
        candidate -= order
    return candidate


def _ant_next_prime(value: int, order: int) -> int:
    # Match ANT Gen_next_prime exactly.  It initializes at value+order and then
    # increments once more before its first primality test.
    candidate = value + order
    while True:
        candidate += order
        if _is_prime(candidate):
            return candidate


def ant_balanced_q(config: DiagnosticConfig) -> list[int]:
    """Match ANT Generate_q_primes exactly, including alternating direction."""
    count = config.data_q_count
    order = 2 * config.poly_degree
    result = [0] * count
    pivot = _ant_first_prime(config.poly_degree, config.scaling_modulus_bits)
    result[-1] = pivot
    previous = pivot
    following = pivot
    for offset in range(max(0, count - 2)):
        index = count - 2 - offset
        if offset % 2 == 0:
            previous = _ant_previous_prime(previous, order)
            result[index] = previous
        else:
            following = _ant_next_prime(following, order)
            result[index] = following
    if config.first_modulus_bits == config.scaling_modulus_bits:
        result[0] = _ant_previous_prime(previous, order)
    else:
        first = _ant_first_prime(config.poly_degree, config.first_modulus_bits)
        result[0] = _ant_previous_prime(first, order)
    return result


class TraceModel:
    """Execute the fixed EvalMod graph while recording operation boundaries."""

    def __init__(
        self,
        config: DiagnosticConfig,
        semantics: Semantics,
        q_values: Sequence[int],
    ) -> None:
        self.config = config
        self.semantics = semantics
        self.q_values = tuple(q_values)
        self.events: list[dict[str, Any]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.input_scale_derivation: list[dict[str, Any]] = []

    @property
    def delta(self) -> float:
        return self.config.nominal_scale

    def state_record(self, state: State) -> dict[str, Any]:
        return state.record(len(self.q_values))

    def _event(
        self,
        stage: str,
        label: str,
        operation: str,
        before: State,
        after: State,
        **details: Any,
    ) -> None:
        if not all(math.isfinite(value) for value in (after.value, after.raw_scale)):
            raise DiagnosticError(f"non-finite state after {label}")
        self.events.append(
            {
                "index": len(self.events),
                "stage": stage,
                "label": label,
                "operation": operation,
                "before": self.state_record(before),
                "after": self.state_record(after),
                **details,
            }
        )

    def checkpoint(self, stage: str, label: str, state: State) -> None:
        self.checkpoints.append(
            {
                "index": len(self.checkpoints),
                "stage": stage,
                "label": label,
                "state": self.state_record(state),
            }
        )

    def initial_state(self) -> State:
        scale = self.delta
        dropped = 0
        for iteration in range(1, self.config.coeff_to_slots_rescales + 1):
            q_index = len(self.q_values) - 1 - dropped
            dropped_prime = self.q_values[q_index]
            input_scale = scale * self.delta
            divisor = (
                float(dropped_prime)
                if self.semantics.rescale_policy == "exact_prime"
                else self.delta
            )
            scale = input_scale / divisor
            dropped += 1
            self.input_scale_derivation.append(
                {
                    "iteration": iteration,
                    "q_index": q_index,
                    "dropped_prime": dropped_prime,
                    "rescale_divisor": divisor,
                    "raw_scale": scale,
                    "raw_scale_hex": scale.hex(),
                }
            )
        state = State(0.0, scale, dropped)
        self.checkpoint("input", "evalmod.input", state)
        return state

    @staticmethod
    def _require_same_level(left: State, right: State, label: str) -> None:
        if left.dropped_q_count != right.dropped_q_count:
            raise DiagnosticError(f"level mismatch at {label}")

    def align_pair(
        self, left: State, right: State, stage: str, label: str
    ) -> tuple[State, State]:
        """Mirror CKKS lowering's automatic alignment to the deeper level."""
        left_step = 0
        right_step = 0
        while left.dropped_q_count < right.dropped_q_count:
            left_step += 1
            left = self.mod_switch(left, stage, f"{label}.align_left.{left_step}")
        while right.dropped_q_count < left.dropped_q_count:
            right_step += 1
            right = self.mod_switch(right, stage, f"{label}.align_right.{right_step}")
        return left, right

    def multiply(self, left: State, right: State, stage: str, label: str) -> State:
        left, right = self.align_pair(left, right, stage, label)
        self._require_same_level(left, right, label)
        result = State(
            left.value * right.value,
            left.raw_scale * right.raw_scale,
            left.dropped_q_count,
        )
        self._event(
            stage,
            f"{label}.multiply",
            "multiply_cipher",
            left,
            result,
            right=self.state_record(right),
        )
        return result

    def multiply_plain(
        self, source: State, scalar: float, stage: str, label: str
    ) -> State:
        result = State(
            source.value * scalar,
            source.raw_scale * self.delta,
            source.dropped_q_count,
        )
        self._event(
            stage,
            f"{label}.multiply_plain",
            "multiply_plain",
            source,
            result,
            scalar=scalar,
            scalar_hex=scalar.hex(),
            plaintext_scale=self.delta,
            plaintext_scale_hex=self.delta.hex(),
        )
        return result

    def rescale(self, source: State, stage: str, label: str) -> State:
        q_index = len(self.q_values) - 1 - source.dropped_q_count
        if q_index <= 0:
            raise DiagnosticError(f"cannot rescale bottom-chain state at {label}")
        dropped_prime = self.q_values[q_index]
        divisor = (
            float(dropped_prime)
            if self.semantics.rescale_policy == "exact_prime"
            else self.delta
        )
        result = State(
            # The logical oracle and the Phantom metadata model both regard
            # rescale as value preserving.  Their difference is the divisor
            # recorded in raw scale metadata.
            source.value,
            source.raw_scale / divisor,
            source.dropped_q_count + 1,
        )
        self._event(
            stage,
            f"{label}.rescale",
            "rescale",
            source,
            result,
            q_index=q_index,
            dropped_prime=dropped_prime,
            rescale_divisor=divisor,
            rescale_divisor_hex=divisor.hex(),
        )
        return result

    def multiply_rescale(
        self, left: State, right: State, stage: str, label: str
    ) -> State:
        return self.rescale(self.multiply(left, right, stage, label), stage, label)

    def multiply_plain_rescale(
        self, source: State, scalar: float, stage: str, label: str
    ) -> State:
        return self.rescale(
            self.multiply_plain(source, scalar, stage, label), stage, label
        )

    def add_cipher(
        self,
        left: State,
        right: State,
        stage: str,
        label: str,
        *,
        subtract: bool = False,
    ) -> State:
        left, right = self.align_pair(left, right, stage, label)
        self._require_same_level(left, right, label)
        sign = -1.0 if subtract else 1.0
        result = State(
            (left.value * left.raw_scale + sign * right.value * right.raw_scale)
            / left.raw_scale,
            left.raw_scale,
            left.dropped_q_count,
        )
        self._event(
            stage,
            label,
            "subtract_cipher" if subtract else "add_cipher",
            left,
            result,
            right=self.state_record(right),
            output_scale_policy="lhs_raw_scale",
        )
        return result

    def add_scalar(self, source: State, scalar: float, stage: str, label: str) -> State:
        encoded_scale = (
            self.delta
            if self.semantics.scalar_policy == "nominal"
            else source.raw_scale
        )
        self._event(
            stage,
            f"{label}.encode",
            "scalar_encode",
            source,
            source,
            scalar=scalar,
            scalar_hex=scalar.hex(),
            plaintext_scale=encoded_scale,
            plaintext_scale_hex=encoded_scale.hex(),
            scalar_policy=self.semantics.scalar_policy,
        )
        result = State(
            source.value + scalar * encoded_scale / source.raw_scale,
            source.raw_scale,
            source.dropped_q_count,
        )
        self._event(
            stage,
            f"{label}.add",
            "scalar_add",
            source,
            result,
            scalar=scalar,
            scalar_hex=scalar.hex(),
            plaintext_scale=encoded_scale,
            plaintext_scale_hex=encoded_scale.hex(),
        )
        return result

    def mod_switch(self, source: State, stage: str, label: str) -> State:
        q_index = len(self.q_values) - 1 - source.dropped_q_count
        if q_index <= 0:
            raise DiagnosticError(f"cannot mod-switch bottom-chain state at {label}")
        result = State(
            source.value,
            source.raw_scale,
            source.dropped_q_count + 1,
        )
        self._event(
            stage,
            label,
            "mod_switch",
            source,
            result,
            q_index=q_index,
            dropped_prime=self.q_values[q_index],
        )
        return result


def _linear_weighted_sum(
    model: TraceModel,
    t_list: Sequence[State | None],
    weights: Sequence[float],
    label: str,
) -> State:
    result: State | None = None
    for index, weight in enumerate(weights):
        if weight == 0.0 or index >= len(t_list) or t_list[index] is None:
            continue
        term = model.multiply_plain_rescale(
            t_list[index], weight, "ps", f"{label}.term{index + 1}"
        )
        result = (
            term
            if result is None
            else model.add_cipher(result, term, "ps", f"{label}.accumulate{index + 1}")
        )
    if result is None:
        raise DiagnosticError(f"empty weighted sum at {label}")
    return result


def _eval_quotient_or_remainder(
    model: TraceModel,
    t_list: Sequence[State | None],
    quot_rem: Sequence[float],
    k: int,
    is_quotient: bool,
    in_recursion: bool,
    label: str,
) -> State:
    qr = list(quot_rem[:k])
    qr.extend([0.0] * (k - len(qr)))
    t_k = t_list[k - 1]
    if t_k is None:
        raise DiagnosticError("missing T_k")
    degree = get_degree_from_coeffs(qr)

    if degree > 0:
        result = _linear_weighted_sum(
            model, t_list, qr[1 : degree + 1], f"{label}.weighted"
        )
        if is_quotient:
            if in_recursion:
                last = quot_rem[-1] if quot_rem else 1.0
                additions = int(math.log2(abs(last))) if abs(last) > 0 else 0
                sum_value = t_k
                for index in range(additions):
                    sum_value = model.add_cipher(
                        sum_value,
                        sum_value,
                        "ps",
                        f"{label}.quotient_power.{index + 1}",
                    )
                result = model.add_cipher(
                    result, sum_value, "ps", f"{label}.add_quotient_power"
                )
            else:
                result = model.add_cipher(result, t_k, "ps", f"{label}.add_Tk.1")
                result = model.add_cipher(result, t_k, "ps", f"{label}.add_Tk.2")
        else:
            result = model.add_cipher(result, t_k, "ps", f"{label}.add_Tk")
    else:
        result = t_k
        if is_quotient:
            last = quot_rem[-1] if quot_rem else 1.0
            additions = (
                int(math.log2(abs(last)))
                if in_recursion and abs(last) > 0
                else int(abs(last))
            )
            for index in range(additions):
                result = model.add_cipher(
                    result, t_k, "ps", f"{label}.constant_quotient.{index + 1}"
                )

    return model.add_scalar(result, qr[0] / 2.0, "ps", f"{label}.free_term")


def _inner_eval_chebyshev_ps(
    model: TraceModel,
    coefficients: Sequence[float],
    k: int,
    m: int,
    t_list: Sequence[State | None],
    t2_list: Sequence[State],
    y: State,
    in_recursion: bool,
    t_depths: Sequence[int],
    path: str,
) -> State:
    k2m2k = k * (1 << (m - 1)) - k
    tkm = [0.0] * (k2m2k + k + 1)
    tkm[-1] = 1.0
    div_q, div_r = long_div_chebyshev(list(coefficients), tkm)

    r2 = list(div_r)
    r2.extend([0.0] * max(0, k2m2k + 1 - len(r2)))
    r2[k2m2k] -= 1.0
    degree_r2 = get_degree_from_coeffs(r2)
    if degree_r2 > 0:
        r2 = r2[: degree_r2 + 1]
    divr2_q, divr2_r = long_div_chebyshev(r2, div_q)

    s2_length = max(len(divr2_r), k2m2k + 1)
    s2 = list(divr2_r) + [0.0] * (s2_length - len(divr2_r))
    s2[-1] = 1.0

    degree_c = get_degree_from_coeffs(divr2_q)
    has_c = False
    cu: State | None = None
    if degree_c >= 1:
        if degree_c == 1:
            q1 = divr2_q[1]
            first = t_list[0]
            if first is None:
                raise DiagnosticError("missing T1")
            cu = (
                model.multiply_plain_rescale(first, q1, "ps", f"{path}.c.linear")
                if q1 != 1.0
                else first
            )
        else:
            cu = _linear_weighted_sum(
                model,
                t_list,
                divr2_q[1 : degree_c + 1],
                f"{path}.c",
            )
        cu = model.add_scalar(cu, divr2_q[0] / 2.0, "ps", f"{path}.c.free_term")
        has_c = True

    if get_degree_from_coeffs(div_q) > k:
        qu = _inner_eval_chebyshev_ps(
            model,
            div_q,
            k,
            m - 1,
            t_list,
            t2_list,
            y,
            True,
            t_depths,
            f"{path}.q",
        )
    else:
        qu = _eval_quotient_or_remainder(
            model,
            t_list,
            div_q,
            k,
            True,
            in_recursion,
            f"{path}.q",
        )

    if get_degree_from_coeffs(s2) > k:
        su = _inner_eval_chebyshev_ps(
            model,
            s2,
            k,
            m - 1,
            t_list,
            t2_list,
            y,
            True,
            t_depths,
            f"{path}.s",
        )
    else:
        su = _eval_quotient_or_remainder(
            model,
            t_list,
            s2,
            k,
            False,
            in_recursion,
            f"{path}.s",
        )

    t2_m_1 = t2_list[m - 1]
    if has_c:
        if cu is None:
            raise DiagnosticError("missing c polynomial")
        target_depth = t_depths[k - 1] + (m - 1)
        if degree_c == 1:
            cu_depth = t_depths[0] + (1 if divr2_q[1] != 1.0 else 0)
        else:
            used_depths = [
                t_depths[index]
                for index, coefficient in enumerate(divr2_q[1 : degree_c + 1])
                if coefficient != 0.0 and index < len(t_depths) and t_depths[index] >= 0
            ]
            cu_depth = (max(used_depths) if used_depths else t_depths[0]) + 1
        for step in range(target_depth - cu_depth):
            cu = model.mod_switch(cu, "ps", f"{path}.c.align.{step + 1}")
        combined = model.add_cipher(t2_m_1, cu, "ps", f"{path}.pre_product")
    else:
        combined = model.add_scalar(
            t2_m_1,
            divr2_q[0] / 2.0,
            "ps",
            f"{path}.pre_product.free_term",
        )

    result = model.multiply_rescale(combined, qu, "ps", f"{path}.product")
    result = model.add_cipher(result, su, "ps", f"{path}.combine")
    model.checkpoint("ps", f"ps.{path}.combine", result)
    return result


def evaluate_zero_evalmod(model: TraceModel) -> State:
    coefficients = list(CHEBYSHEV_COEFFICIENTS)
    degree = get_degree_from_coeffs(coefficients)
    even = is_even_poly(coefficients)
    k, m = compute_degree_ps(degree)
    if even and k % 2 == 1:
        k += 1

    x = model.initial_state()
    y = x
    t_list: list[State | None] = [None] * k
    t_list[0] = x
    model.checkpoint("baby_steps", "baby.T1", x)

    for order in range(2, k + 1):
        index = order - 1
        if order & (order - 1) == 0:
            half = t_list[order // 2 - 1]
            if half is None:
                raise DiagnosticError(f"missing T{order // 2}")
            product = model.multiply_rescale(
                half, half, "baby_steps", f"baby.T{order}.square"
            )
            current = model.add_cipher(
                product, product, "baby_steps", f"baby.T{order}.double"
            )
            current = model.add_scalar(
                current, -1.0, "baby_steps", f"baby.T{order}.add_minus_one"
            )
            t_list[index] = current
        elif order % 2 == 1:
            if even:
                continue
            low = t_list[order // 2 - 1]
            high = t_list[order // 2]
            if low is None or high is None:
                raise DiagnosticError(f"missing T inputs for T{order}")
            product = model.multiply_rescale(
                low, high, "baby_steps", f"baby.T{order}.product"
            )
            current = model.add_cipher(
                product, product, "baby_steps", f"baby.T{order}.double"
            )
            t_list[index] = model.add_cipher(
                current,
                y,
                "baby_steps",
                f"baby.T{order}.subtract_T1",
                subtract=True,
            )
        else:
            half_1 = order // 2
            if even and half_1 % 2 == 1:
                half_1 += 1
            half_2 = order - half_1
            left = t_list[half_1 - 1]
            right = t_list[half_2 - 1]
            if left is None or right is None:
                raise DiagnosticError(f"missing T inputs for T{order}")
            product = model.multiply_rescale(
                left, right, "baby_steps", f"baby.T{order}.product"
            )
            current = model.add_cipher(
                product, product, "baby_steps", f"baby.T{order}.double"
            )
            if half_1 == half_2:
                current = model.add_scalar(
                    current,
                    -1.0,
                    "baby_steps",
                    f"baby.T{order}.add_minus_one",
                )
            else:
                t2 = t_list[1]
                if t2 is None:
                    raise DiagnosticError("missing T2")
                current = model.add_cipher(
                    current,
                    t2,
                    "baby_steps",
                    f"baby.T{order}.subtract_T2",
                    subtract=True,
                )
            t_list[index] = current
        if t_list[index] is not None:
            model.checkpoint("baby_steps", f"baby.T{order}", t_list[index])

    depths = compute_chebyshev_depths(k, even)
    tk_depth = depths[k - 1]
    for index in range(1, k):
        current = t_list[index]
        if current is None or depths[index] < 0:
            continue
        for step in range(tk_depth - depths[index]):
            current = model.mod_switch(
                current,
                "baby_alignment",
                f"baby.T{index + 1}.align.{step + 1}",
            )
        t_list[index] = current
        model.checkpoint("baby_alignment", f"baby.T{index + 1}.aligned", current)

    tk = t_list[k - 1]
    if tk is None:
        raise DiagnosticError("missing T_k")
    t2_list = [tk]
    for index in range(1, m):
        previous = t2_list[index - 1]
        product = model.multiply_rescale(
            previous,
            previous,
            "giant_steps",
            f"giant.T{(1 << index) * k}.square",
        )
        current = model.add_cipher(
            product,
            product,
            "giant_steps",
            f"giant.T{(1 << index) * k}.double",
        )
        current = model.add_scalar(
            current,
            -1.0,
            "giant_steps",
            f"giant.T{(1 << index) * k}.add_minus_one",
        )
        t2_list.append(current)
        model.checkpoint("giant_steps", f"giant.T{(1 << index) * k}", current)

    t2km1 = t2_list[0]
    for index in range(1, m):
        product = model.multiply_rescale(
            t2km1,
            t2_list[index],
            "giant_steps",
            f"giant.t2km1.{index}.product",
        )
        doubled = model.add_cipher(
            product,
            product,
            "giant_steps",
            f"giant.t2km1.{index}.double",
        )
        t2km1 = model.add_cipher(
            doubled,
            t2_list[0],
            "giant_steps",
            f"giant.t2km1.{index}.subtract_Tk",
            subtract=True,
        )
        model.checkpoint("giant_steps", f"giant.t2km1.{index}", t2km1)

    k2m2k = k * (1 << (m - 1)) - k
    target_length = 2 * k2m2k + k + 1
    coefficients.extend([0.0] * max(0, target_length - len(coefficients)))
    coefficients[target_length - 1] = 1.0
    result = _inner_eval_chebyshev_ps(
        model,
        coefficients,
        k,
        m,
        t_list,
        t2_list,
        y,
        False,
        depths,
        "root",
    )
    result = model.add_cipher(
        result,
        t2km1,
        "chebyshev",
        "chebyshev.subtract_t2km1",
        subtract=True,
    )
    model.checkpoint("chebyshev", "chebyshev.final", result)

    for iteration, scalar in enumerate(DOUBLE_ANGLE_SCALARS, start=1):
        product = model.multiply_rescale(
            result,
            result,
            "double_angle",
            f"double_angle.{iteration}.square",
        )
        doubled = model.add_cipher(
            product,
            product,
            "double_angle",
            f"double_angle.{iteration}.double",
        )
        result = model.add_scalar(
            doubled,
            scalar,
            "double_angle",
            f"double_angle.{iteration}.add_scalar",
        )
        model.checkpoint("double_angle", f"double_angle.{iteration}", result)
    model.checkpoint("output", "evalmod.output", result)
    return result


def _find_event(events: Iterable[dict[str, Any]], label: str) -> dict[str, Any]:
    matches = [event for event in events if event["label"] == label]
    if len(matches) != 1:
        raise DiagnosticError(f"expected one event named {label}, found {len(matches)}")
    return matches[0]


def _find_checkpoint(
    checkpoints: Iterable[dict[str, Any]], label: str
) -> dict[str, Any]:
    matches = [entry for entry in checkpoints if entry["label"] == label]
    if len(matches) != 1:
        raise DiagnosticError(
            f"expected one checkpoint named {label}, found {len(matches)}"
        )
    return matches[0]


def _model_report(model: TraceModel, final: State) -> dict[str, Any]:
    t2_encode = _find_event(model.events, "baby.T2.add_minus_one.encode")
    t2_add = _find_event(model.events, "baby.T2.add_minus_one.add")
    scalar_encodes = [
        event for event in model.events if event["operation"] == "scalar_encode"
    ]
    rescales = [event for event in model.events if event["operation"] == "rescale"]
    mod_switches = [
        event for event in model.events if event["operation"] == "mod_switch"
    ]
    return {
        "semantics": {
            "name": model.semantics.name,
            "description": model.semantics.description,
            "modulus_policy": model.semantics.modulus_policy,
            "scalar_policy": model.semantics.scalar_policy,
            "rescale_policy": model.semantics.rescale_policy,
        },
        "ordered_data_q_moduli": list(model.q_values),
        "input_scale_derivation": model.input_scale_derivation,
        "checkpoints": model.checkpoints,
        "boundaries": model.events,
        "summary": {
            "event_count": len(model.events),
            "checkpoint_count": len(model.checkpoints),
            "scalar_encode_count": len(scalar_encodes),
            "rescale_count": len(rescales),
            "mod_switch_count": len(mod_switches),
            "t2_first_scalar": {
                "label": "baby.T2.add_minus_one",
                "pre_scalar": t2_encode["before"],
                "scalar": t2_encode["scalar"],
                "encoded_scale": t2_encode["plaintext_scale"],
                "encoded_scale_hex": t2_encode["plaintext_scale_hex"],
                "post_scalar": t2_add["after"],
            },
            "final_chebyshev": _find_checkpoint(model.checkpoints, "chebyshev.final")[
                "state"
            ],
            "double_angle": [
                _find_checkpoint(model.checkpoints, f"double_angle.{index}")["state"]
                for index in range(1, len(DOUBLE_ANGLE_SCALARS) + 1)
            ],
            "final_evalmod": model.state_record(final),
        },
    }


def _different(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) > tolerance


def _compare_models(
    left: dict[str, Any], right: dict[str, Any], value_tolerance: float
) -> dict[str, Any]:
    left_input = _find_checkpoint(left["checkpoints"], "evalmod.input")["state"]
    right_input = _find_checkpoint(right["checkpoints"], "evalmod.input")["state"]
    first_scale: dict[str, Any] | None = None
    if left_input["raw_scale"] != right_input["raw_scale"]:
        first_scale = {
            "label": "evalmod.input",
            "operation": "checkpoint",
            "left": left_input,
            "right": right_input,
        }

    left_events = left["boundaries"]
    right_events = right["boundaries"]
    if len(left_events) != len(right_events):
        raise DiagnosticError("model event counts differ")
    first_value: dict[str, Any] | None = None
    for left_event, right_event in zip(left_events, right_events):
        signature = (left_event["label"], left_event["operation"])
        if signature != (right_event["label"], right_event["operation"]):
            raise DiagnosticError("model event schedules differ")
        if first_scale is None and (
            left_event["after"]["raw_scale"] != right_event["after"]["raw_scale"]
        ):
            first_scale = {
                "label": left_event["label"],
                "operation": left_event["operation"],
                "left": left_event["after"],
                "right": right_event["after"],
            }
        if first_value is None and _different(
            left_event["after"]["value"],
            right_event["after"]["value"],
            value_tolerance,
        ):
            first_value = {
                "label": left_event["label"],
                "operation": left_event["operation"],
                "left": left_event["after"],
                "right": right_event["after"],
                "absolute_value_difference": abs(
                    left_event["after"]["value"] - right_event["after"]["value"]
                ),
            }
    return {
        "left_model": left["semantics"]["name"],
        "right_model": right["semantics"]["name"],
        "value_divergence_tolerance": value_tolerance,
        "first_raw_scale_divergence": first_scale,
        "first_value_divergence": first_value,
        "final_value_difference": (
            left["summary"]["final_evalmod"]["value"]
            - right["summary"]["final_evalmod"]["value"]
        ),
    }


def _checkpoint_comparison(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    model_names = [semantics.name for semantics in SEMANTICS]
    schedules = {name: models[name]["checkpoints"] for name in model_names}
    lengths = {len(schedule) for schedule in schedules.values()}
    if len(lengths) != 1:
        raise DiagnosticError("model checkpoint counts differ")
    result: list[dict[str, Any]] = []
    for index, current in enumerate(schedules[model_names[0]]):
        states: dict[str, dict[str, Any]] = {}
        for name in model_names:
            entry = schedules[name][index]
            if (entry["stage"], entry["label"]) != (
                current["stage"],
                current["label"],
            ):
                raise DiagnosticError("model checkpoint schedules differ")
            states[name] = entry["state"]
        oracle = states["ant_nominal_oracle"]
        result.append(
            {
                "index": index,
                "stage": current["stage"],
                "label": current["label"],
                "models": states,
                "current_minus_oracle": {
                    "value": states["phantom_current"]["value"] - oracle["value"],
                    "raw_scale": states["phantom_current"]["raw_scale"]
                    - oracle["raw_scale"],
                },
                "corrected_minus_oracle": {
                    "value": states["phantom_corrected"]["value"] - oracle["value"],
                    "raw_scale": states["phantom_corrected"]["raw_scale"]
                    - oracle["raw_scale"],
                },
            }
        )
    return result


def _boundary_comparison(models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    relevant = {"scalar_encode", "scalar_add", "rescale", "mod_switch"}
    model_names = [semantics.name for semantics in SEMANTICS]
    schedules = {
        name: [
            event
            for event in models[name]["boundaries"]
            if event["operation"] in relevant
        ]
        for name in model_names
    }
    lengths = {len(schedule) for schedule in schedules.values()}
    if len(lengths) != 1:
        raise DiagnosticError("model boundary counts differ")
    result: list[dict[str, Any]] = []
    for index, current in enumerate(schedules[model_names[0]]):
        entries: dict[str, dict[str, Any]] = {}
        for name in model_names:
            event = schedules[name][index]
            if (event["label"], event["operation"]) != (
                current["label"],
                current["operation"],
            ):
                raise DiagnosticError("model boundary schedules differ")
            entry: dict[str, Any] = {"state": event["after"]}
            for field in (
                "scalar",
                "plaintext_scale",
                "q_index",
                "dropped_prime",
                "rescale_divisor",
            ):
                if field in event:
                    entry[field] = event[field]
            entries[name] = entry
        oracle = entries["ant_nominal_oracle"]["state"]
        result.append(
            {
                "index": index,
                "stage": current["stage"],
                "label": current["label"],
                "operation": current["operation"],
                "models": entries,
                "current_minus_oracle": {
                    "value": entries["phantom_current"]["state"]["value"]
                    - oracle["value"],
                    "raw_scale": entries["phantom_current"]["state"]["raw_scale"]
                    - oracle["raw_scale"],
                },
                "corrected_minus_oracle": {
                    "value": entries["phantom_corrected"]["state"]["value"]
                    - oracle["value"],
                    "raw_scale": entries["phantom_corrected"]["state"]["raw_scale"]
                    - oracle["raw_scale"],
                },
            }
        )
    return result


def build_report(config: DiagnosticConfig | None = None) -> dict[str, Any]:
    config = config or DiagnosticConfig()
    config.validate()
    modulus_sequences = {
        "phantom_all_below": phantom_all_below_q(config),
        "ant_balanced": ant_balanced_q(config),
    }
    models: dict[str, dict[str, Any]] = {}
    for semantics in SEMANTICS:
        trace = TraceModel(
            config,
            semantics,
            modulus_sequences[semantics.modulus_policy],
        )
        final = evaluate_zero_evalmod(trace)
        models[semantics.name] = _model_report(trace, final)

    current_vs_oracle = _compare_models(
        models["phantom_current"],
        models["ant_nominal_oracle"],
        value_tolerance=1.0e-15,
    )
    corrected_vs_oracle = _compare_models(
        models["phantom_corrected"],
        models["ant_nominal_oracle"],
        value_tolerance=1.0e-15,
    )
    first = current_vs_oracle["first_value_divergence"]
    if first is None:
        raise DiagnosticError("current semantics unexpectedly match the oracle")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "configuration": {
            "polynomial_degree": config.poly_degree,
            "logical_slots": config.poly_degree // 2,
            "data_q_count": config.data_q_count,
            "special_p_count": config.special_p_count,
            "first_modulus_bits": config.first_modulus_bits,
            "scaling_modulus_bits": config.scaling_modulus_bits,
            "nominal_scale": config.nominal_scale,
            "nominal_scale_hex": config.nominal_scale.hex(),
            "coeff_to_slots_rescales": config.coeff_to_slots_rescales,
            "evalmod_input_value": 0.0,
            "chebyshev_degree": len(CHEBYSHEV_COEFFICIENTS) - 1,
            "double_angle_iterations": len(DOUBLE_ANGLE_SCALARS),
        },
        "models": models,
        "checkpoint_comparison": _checkpoint_comparison(models),
        "boundary_comparison": _boundary_comparison(models),
        "comparisons": {
            "current_vs_ant_nominal": current_vs_oracle,
            "corrected_vs_ant_nominal": corrected_vs_oracle,
        },
        "diagnosis": {
            "first_material_value_divergence": first,
            "classification": "scalar plaintext scale mismatch after exact-prime rescale",
            "explanation": (
                "The zero-valued T1 square remains zero, but Phantom records "
                "the exact S/q raw scale. Encoding T2's -1 at nominal Delta "
                "therefore adds -Delta/S instead of -1. Dynamic scalar "
                "encoding keeps the T2 value exact while preserving complex "
                "values."
            ),
        },
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poly-degree", type=int, default=65536)
    parser.add_argument("--data-q-count", type=int, default=26)
    parser.add_argument("--special-p-count", type=int, default=9)
    parser.add_argument("--first-modulus-bits", type=int, default=60)
    parser.add_argument("--scaling-modulus-bits", type=int, default=56)
    parser.add_argument("--coeff-to-slots-rescales", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact", action="store_true", help="emit compact canonical JSON"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    config = DiagnosticConfig(
        poly_degree=arguments.poly_degree,
        data_q_count=arguments.data_q_count,
        special_p_count=arguments.special_p_count,
        first_modulus_bits=arguments.first_modulus_bits,
        scaling_modulus_bits=arguments.scaling_modulus_bits,
        coeff_to_slots_rescales=arguments.coeff_to_slots_rescales,
    )
    try:
        report = build_report(config)
    except DiagnosticError as error:
        raise SystemExit(f"EvalMod scale diagnostic failed: {error}") from error
    payload = (
        json.dumps(
            report,
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=(",", ":") if arguments.compact else None,
        )
        + "\n"
    )
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
