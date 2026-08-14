#!/usr/bin/env python3
"""Derive a provider-neutral clear EvalMod identity-domain attestation."""

from __future__ import annotations

from collections import Counter
from decimal import (
    Context,
    Decimal,
    ROUND_CEILING,
    ROUND_FLOOR,
    __libmpdec_version__,
)
from functools import lru_cache
from fractions import Fraction
import hashlib
import json
import math
import re
import struct
from typing import Any, Sequence


DOMAIN_ATTESTATION_SCHEMA = "ace.phantom.bootstrap-clear-evalmod-domain/1.0.0"
TRANSFORM_PAYLOAD_SCHEMA = (
    "ace.phantom.bootstrap-transform-payload-semantics/1.0.0"
)
CLEAR_MAP_BUDGET_FRACTION = 0.5
DIRECTED_DECIMAL_PRECISION = 256
DOMAIN_ARTIFACT_BINDING_KEYS = {
    "compiler_invocation_sha256",
    "constant_manifest_sha256",
    "context_manifest_sha256",
    "generated_dsl_ant_source_sha256",
    "phantom_source_sha256",
    "post_ckks_air_sha256",
    "raw_air_sha256",
    "resource_manifest_sha256",
}

_DOWN = Context(prec=DIRECTED_DECIMAL_PRECISION, rounding=ROUND_FLOOR)
_UP = Context(prec=DIRECTED_DECIMAL_PRECISION, rounding=ROUND_CEILING)
_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_SCALAR_DEFINITION = re.compile(
    r"^\s*float64_t\s+([A-Za-z_]\w*)\s*=\s*([^;]+);\s*$",
    re.MULTILINE,
)
_MONOMIAL_OPERATION = re.compile(
    r"intconst\s+#(0x[0-9a-f]+|[0-9]+)[^\n]*\n\s*"
    r"CKKS\.mul_mono\b"
)

Interval = tuple[Decimal, Decimal]
Polynomial = list[Interval]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _float_sequence_sha256(values: Sequence[float]) -> str:
    return _sha256_bytes(b"".join(struct.pack("<d", value) for value in values))


def _point(value: Decimal) -> Interval:
    return value, value


def _add(left: Interval, right: Interval) -> Interval:
    return _DOWN.add(left[0], right[0]), _UP.add(left[1], right[1])


def _negate(value: Interval) -> Interval:
    return _DOWN.minus(value[1]), _UP.minus(value[0])


def _subtract(left: Interval, right: Interval) -> Interval:
    return _add(left, _negate(right))


def _multiply(left: Interval, right: Interval) -> Interval:
    lower_products = [
        _DOWN.multiply(left_value, right_value)
        for left_value in left
        for right_value in right
    ]
    upper_products = [
        _UP.multiply(left_value, right_value)
        for left_value in left
        for right_value in right
    ]
    return min(lower_products), max(upper_products)


def _divide(left: Interval, right: Interval) -> Interval:
    if right[0] <= 0 <= right[1]:
        raise ValueError("directed interval division contains zero")
    reciprocal = (
        _DOWN.divide(_ONE, right[1]),
        _UP.divide(_ONE, right[0]),
    )
    return _multiply(left, reciprocal)


def _poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [_point(_ZERO) for _ in range(max(len(left), len(right)))]
    for index in range(len(result)):
        left_value = left[index] if index < len(left) else _point(_ZERO)
        right_value = right[index] if index < len(right) else _point(_ZERO)
        result[index] = _add(left_value, right_value)
    while len(result) > 1 and result[-1] == _point(_ZERO):
        result.pop()
    return result


def _poly_negate(value: Polynomial) -> Polynomial:
    return [_negate(item) for item in value]


def _poly_scale(value: Polynomial, scalar: Interval) -> Polynomial:
    return [_multiply(item, scalar) for item in value]


def _poly_shift_x(value: Polynomial) -> Polynomial:
    return [_point(_ZERO), *value]


def _poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result = [_point(_ZERO) for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            degree = left_degree + right_degree
            result[degree] = _add(
                result[degree], _multiply(left_value, right_value)
            )
    return result


def _expanded_power_polynomial(
    coefficients: tuple[str, ...], scalars: tuple[str, ...]
) -> Polynomial:
    coefficient_values = [_point(Decimal.from_float(float.fromhex(value))) for value in coefficients]
    scalar_values = [_point(Decimal.from_float(float.fromhex(value))) for value in scalars]
    half = _point(Decimal("0.5"))
    chebyshev: list[Polynomial] = [[_point(_ONE)]]
    if len(coefficient_values) > 1:
        chebyshev.append([_point(_ZERO), _point(_ONE)])
    for _ in range(2, len(coefficient_values)):
        chebyshev.append(
            _poly_add(
                _poly_scale(_poly_shift_x(chebyshev[-1]), _point(_TWO)),
                _poly_negate(chebyshev[-2]),
            )
        )
    result = _poly_scale([_point(_ONE)], _multiply(coefficient_values[0], half))
    for degree, coefficient in enumerate(coefficient_values[1:], 1):
        result = _poly_add(result, _poly_scale(chebyshev[degree], coefficient))
    for scalar in scalar_values:
        result = _poly_add(
            _poly_scale(_poly_multiply(result, result), _point(_TWO)),
            [scalar],
        )
    return result


def _translate_polynomial(value: Polynomial, point: Interval) -> Polynomial:
    """Return interval coefficients for value(point + z), using Horner."""
    result = [value[-1]]
    for coefficient in reversed(value[:-1]):
        updated = [_point(_ZERO) for _ in range(len(result) + 1)]
        updated[0] = _add(_multiply(result[0], point), coefficient)
        for degree in range(1, len(result)):
            updated[degree] = _add(
                _multiply(result[degree], point), result[degree - 1]
            )
        updated[-1] = result[-1]
        result = updated
    return result


def _phase_error_polynomial(
    polynomial: Polynomial,
    phase: int,
    restoration_factor: float,
    component_input_gain: float,
) -> Polynomial:
    restoration = _point(Decimal.from_float(restoration_factor))
    input_gain = _point(Decimal.from_float(component_input_gain))
    phase_coordinate = _multiply(
        _point(Decimal(phase)), _multiply(restoration, input_gain)
    )
    result = _translate_polynomial(polynomial, phase_coordinate)
    gain_power = _point(_ONE)
    for degree in range(len(result)):
        result[degree] = _multiply(
            result[degree], _multiply(restoration, gain_power)
        )
        gain_power = _multiply(gain_power, input_gain)
    if len(result) < 2:
        result.append(_point(-_ONE))
    else:
        result[1] = _subtract(result[1], _point(_ONE))
    return result


def _interval_absolute_upper(value: Interval) -> Decimal:
    return max(_UP.abs(value[0]), _UP.abs(value[1]))


def _float_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _float_from_bits(value: int) -> float:
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def _component_envelope(radius: float, coefficients: tuple[Decimal, ...]) -> Decimal:
    value = Decimal.from_float(radius)
    power = _ONE
    result = _ZERO
    for coefficient in coefficients:
        result = _UP.add(result, _UP.multiply(coefficient, power))
        power = _UP.multiply(power, value)
    return result


def _complex_squared_envelope(
    radius: float, coefficients: tuple[Decimal, ...]
) -> Decimal:
    component = _component_envelope(radius, coefficients)
    return _UP.multiply(_TWO, _UP.multiply(component, component))


@lru_cache(maxsize=8)
def _derive_polynomial_proof(
    polynomial_rationals: tuple[tuple[int, int], ...],
    overflow_bound: int,
    restoration_factor_hex: str,
    component_input_gain_hex: str,
    maximum_complex_clear_map_error_hex: str,
) -> dict[str, Any]:
    restoration_factor = float.fromhex(restoration_factor_hex)
    component_input_gain = float.fromhex(component_input_gain_hex)
    maximum_error = float.fromhex(maximum_complex_clear_map_error_hex)
    polynomial = [
        _divide(_point(Decimal(numerator)), _point(Decimal(denominator)))
        for numerator, denominator in polynomial_rationals
    ]
    phase_polynomials = [
        _phase_error_polynomial(
            polynomial, phase, restoration_factor, component_input_gain
        )
        for phase in range(-overflow_bound, overflow_bound + 1)
    ]
    coefficient_count = max(len(polynomial) for polynomial in phase_polynomials)
    coefficient_envelope = tuple(
        max(
            _interval_absolute_upper(
                polynomial[degree]
                if degree < len(polynomial)
                else _point(_ZERO)
            )
            for polynomial in phase_polynomials
        )
        for degree in range(coefficient_count)
    )
    envelope_payload = [str(value) for value in coefficient_envelope]
    budget = Decimal.from_float(maximum_error)
    budget_squared_lower = _DOWN.multiply(budget, budget)

    def certified(radius: float) -> bool:
        return (
            _complex_squared_envelope(radius, coefficient_envelope)
            <= budget_squared_lower
        )

    upper = restoration_factor / 2.0
    if certified(upper):
        raise ValueError("clear EvalMod proof search interval never leaves its budget")
    if not certified(0.0):
        raise ValueError("clear-map error budget is exhausted at the origin")
    lower_bits = _float_bits(0.0)
    upper_bits = _float_bits(upper)
    while upper_bits - lower_bits > 1:
        middle_bits = (lower_bits + upper_bits) // 2
        if certified(_float_from_bits(middle_bits)):
            lower_bits = middle_bits
        else:
            upper_bits = middle_bits
    radius = _float_from_bits(lower_bits)
    outward = _float_from_bits(upper_bits)
    selected_squared = _complex_squared_envelope(radius, coefficient_envelope)
    outward_squared = _complex_squared_envelope(outward, coefficient_envelope)
    if not selected_squared <= budget_squared_lower < outward_squared:
        raise ValueError("clear EvalMod binary64 boundary isolation failed")
    input_gain = _point(Decimal.from_float(component_input_gain))
    restoration = _point(Decimal.from_float(restoration_factor))
    maximum_phase = _point(Decimal(overflow_bound))
    radius_interval = _point(Decimal.from_float(radius))
    maximum_coordinate = _multiply(
        input_gain,
        _add(_multiply(maximum_phase, restoration), radius_interval),
    )
    minimum_coordinate = _negate(maximum_coordinate)
    return {
        "algorithm": (
            "directed-decimal-emitted-exact-rational-polynomial-"
            "phase-envelope-v1"
        ),
        "arithmetic": {
            "implementation": "python-decimal-libmpdec",
            "libmpdec_version": __libmpdec_version__,
            "precision_decimal_digits": DIRECTED_DECIMAL_PRECISION,
            "lower_rounding": "ROUND_FLOOR",
            "upper_rounding": "ROUND_CEILING",
            "binary64_inputs_promoted_exactly": True,
            "transcendental_approximations_used": False,
        },
        "composite_polynomial_degree": len(polynomial) - 1,
        "integer_phase_range_inclusive": [-overflow_bound, overflow_bound],
        "phase_count": len(phase_polynomials),
        "coefficient_envelope_count": len(coefficient_envelope),
        "coefficient_envelope_sha256": _sha256_bytes(
            _canonical_bytes(envelope_payload)
        ),
        "selected_radius": radius,
        "selected_radius_hex": radius.hex(),
        "selected_complex_clear_map_error_squared_bound": str(selected_squared),
        "maximum_complex_clear_map_error_squared_lower_bound": str(
            budget_squared_lower
        ),
        "next_outward_radius": outward,
        "next_outward_radius_hex": outward.hex(),
        "next_outward_complex_clear_map_error_squared_envelope": str(
            outward_squared
        ),
        "proof_evaluation_coordinate_envelope": {
            "lower_directed_decimal": str(minimum_coordinate[0]),
            "upper_directed_decimal": str(maximum_coordinate[1]),
            "fit_interval_containment_required": False,
        },
        "binary64_selection": (
            "conservative-certified-binary64-marker-used-as-exclusive-endpoint"
        ),
    }


def _validate_artifact_bindings(artifact_bindings: dict[str, str]) -> None:
    if set(artifact_bindings) != DOMAIN_ARTIFACT_BINDING_KEYS or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in artifact_bindings.values()
    ):
        raise ValueError("clear EvalMod artifact bindings are incomplete or invalid")


def _air_rotation_outputs(
    air: str,
    *,
    logical_slots: int,
    air_label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Recover rotate-batch result provenance from the textual AIR tree."""
    lines = air.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    batches: dict[str, dict[str, Any]] = {}
    batch_pattern = re.compile(
        r"^\s*CKKS\.rotate_batch ATTR\[[^\]]*\bnums=\(([^)]*)\)"
    )
    for index, line in enumerate(lines):
        match = batch_pattern.match(line)
        if match is None:
            continue
        try:
            rotations = [
                int(value.strip(), 0) % logical_slots
                for value in match.group(1).split(",")
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError(
                f"{air_label} rotate-batch attributes are invalid"
            ) from error
        if not rotations or len(rotations) != len(set(rotations)):
            raise ValueError(
                f"{air_label} rotate-batch rotations are empty or duplicate"
            )
        store_index = index + 1
        while store_index < len(lines) and not lines[store_index].strip():
            store_index += 1
        store_match = (
            re.match(r'^\s*st\s+"([^"]+)"\s+', lines[store_index])
            if store_index < len(lines)
            else None
        )
        if store_match is None:
            raise ValueError(
                f"{air_label} rotate-batch result has no direct store"
            )
        symbol = store_match.group(1)
        if symbol in batches:
            raise ValueError(
                f"{air_label} rotate-batch result symbol is duplicated"
            )
        batches[symbol] = {
            "symbol": symbol,
            "rotations": rotations,
            "position": offsets[index],
        }

    outputs: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        load_match = re.match(r'^\s*lda\s+"([^"]+)"\s+', line)
        if load_match is None or load_match.group(1) not in batches:
            continue
        significant = []
        cursor = index + 1
        while cursor < len(lines) and len(significant) < 4:
            if lines[cursor].strip():
                significant.append(cursor)
            cursor += 1
        if len(significant) != 4:
            raise ValueError(f"{air_label} rotate-batch extraction is incomplete")
        index_match = re.match(
            r"^\s*intconst\s+#(0x[0-9a-f]+|[0-9]+)\b",
            lines[significant[0]],
        )
        result_match = re.match(
            r'^\s*st\s+"([^"]+)"\s+', lines[significant[3]]
        )
        if (
            index_match is None
            or re.match(r"^\s*array\b", lines[significant[1]]) is None
            or re.match(r"^\s*ild\b", lines[significant[2]]) is None
            or result_match is None
        ):
            raise ValueError(
                f"{air_label} rotate-batch extraction shape is invalid"
            )
        batch = batches[load_match.group(1)]
        output_index = int(index_match.group(1), 0)
        if output_index >= len(batch["rotations"]):
            raise ValueError(
                f"{air_label} rotate-batch extraction index is out of range"
            )
        result_symbol = result_match.group(1)
        if result_symbol in outputs:
            raise ValueError(f"{air_label} rotated-output symbol is duplicated")
        outputs[result_symbol] = {
            "batch_symbol": batch["symbol"],
            "batch_position": batch["position"],
            "batch_rotations": batch["rotations"],
            "rotation_batch_index": output_index,
            "input_rotation": batch["rotations"][output_index],
        }
    if not batches or not outputs:
        raise ValueError(f"{air_label} omits rotate-batch operand provenance")
    return batches, outputs


def _attest_air_constant_use(
    air: str,
    *,
    constant_id: int,
    logical_slots: int,
    ace_level: int,
    num_p: int,
    rotation_outputs: dict[str, dict[str, Any]],
    air_label: str,
) -> dict[str, Any]:
    """Bind one constant to its unique direct rotated-operand multiply."""
    lines = air.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    load_pattern = re.compile(
        rf"^\s*ldc\s+CST\[{constant_id:#x}\]\s+"
    )
    matching_lines = [
        index for index, line in enumerate(lines) if load_pattern.search(line)
    ]
    if len(matching_lines) != 1:
        raise ValueError(
            f"{air_label} does not uniquely load a transform payload constant"
        )
    load_index = matching_lines[0]
    load_indent = len(lines[load_index]) - len(lines[load_index].lstrip())

    encode_index = None
    for index in range(load_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        indent = len(lines[index]) - len(lines[index].lstrip())
        if indent < load_indent:
            encode_index = index
            break
    if encode_index is None:
        raise ValueError(f"{air_label} constant load has no enclosing encode")
    encode_line = lines[encode_index].strip()
    encode_indent = len(lines[encode_index]) - len(lines[encode_index].lstrip())
    encode_match = re.fullmatch(
        r"CKKS\.encode ATTR\[([^\]]+)\] RTYPE\[[^\]]+\]\(PLAINTEXT\)",
        encode_line,
    )
    argument_values = []
    for line in lines[load_index + 1 : encode_index]:
        match = re.match(r"\s*intconst\s+#(0x[0-9a-f]+|[0-9]+)\b", line)
        if match:
            argument_values.append(int(match.group(1), 0))
    expected_num_p = f"num_p={num_p}"
    if (
        encode_indent != load_indent - 2
        or encode_match is None
        or argument_values != [logical_slots, 1, ace_level]
        or f"level={ace_level}" not in encode_match.group(1).split(",")
        or "scale=1" not in encode_match.group(1).split(",")
        or expected_num_p not in encode_match.group(1).split(",")
    ):
        raise ValueError(
            f"{air_label} transform payload load has the wrong direct encode metadata"
        )

    cipher_index = load_index - 1
    while cipher_index >= 0 and not lines[cipher_index].strip():
        cipher_index -= 1
    if cipher_index < 0:
        raise ValueError(
            f"{air_label} transform payload multiply has no ciphertext operand"
        )
    cipher_indent = len(lines[cipher_index]) - len(lines[cipher_index].lstrip())
    cipher_match = re.match(
        r'^ld\s+"([^"]+)"\s+', lines[cipher_index].strip()
    )
    if (
        cipher_indent != encode_indent
        or cipher_match is None
    ):
        raise ValueError(
            f"{air_label} transform payload encode is not paired with a ciphertext load"
        )

    multiply_index = None
    for index in range(encode_index + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        indent = len(lines[index]) - len(lines[index].lstrip())
        if indent < encode_indent:
            multiply_index = index
            break
    if (
        multiply_index is None
        or len(lines[multiply_index]) - len(lines[multiply_index].lstrip())
        != encode_indent - 2
        or (
            (multiply_match := re.fullmatch(
                r"CKKS\.mul ATTR\[([^\]]+)\] "
                r"RTYPE\[[^\]]+\]\(CIPHERTEXT\)",
                lines[multiply_index].strip(),
            ))
            is None
        )
        or "skip_auto_rescale=1" not in multiply_match.group(1).split(",")
    ):
        raise ValueError(
            f"{air_label} transform payload encode is not directly consumed by multiply"
        )
    cipher_symbol = cipher_match.group(1)
    rotation_binding = rotation_outputs.get(cipher_symbol)
    if rotation_binding is None:
        raise ValueError(
            f"{air_label} transform payload multiply does not consume a proven "
            "rotate-batch output"
        )
    return {
        "position": offsets[load_index],
        "cipher_symbol": cipher_symbol,
        **rotation_binding,
    }


def _bootstrap_function_region(
    air: str,
    *,
    air_label: str,
) -> tuple[int, int, str]:
    function_marker = 'func_entry "bootstrap_full"'
    function_start = air.find(function_marker)
    if function_start < 0:
        raise ValueError(f"{air_label} omits bootstrap_full")
    if air.find(function_marker, function_start + len(function_marker)) >= 0:
        raise ValueError(f"{air_label} duplicates bootstrap_full")
    function_end = air.find("\n    end_block", function_start)
    if function_end < 0:
        raise ValueError(f"{air_label} bootstrap_full has no exact terminator")
    function_end = air.find("\n", function_end + 1)
    if function_end < 0:
        function_end = len(air)
    return function_start, function_end, air[function_start:function_end]


def _bootstrap_formals_and_return_source(
    air: str,
    *,
    air_label: str,
) -> tuple[list[str], str]:
    """Recover the exact bootstrap ABI and returned value inside its block."""
    _, _, function = _bootstrap_function_region(air, air_label=air_label)
    formals = re.findall(
        r'^\s+idname\s+"([^"]+)"\s+FML\b', function, flags=re.MULTILINE
    )
    returns = list(
        re.finditer(
            r'^\s*ld\s+"(?P<source>[^"]+)"[^\n]*\n'
            r'\s*st\s+"(?P<temporary>__ret_tmp_[^"]+)"[^\n]*\n'
            r'\s*ld\s+"(?P=temporary)"[^\n]*\n\s*retv\b',
            function,
            flags=re.MULTILINE,
        )
    )
    if len(formals) != 2 or len(set(formals)) != 2:
        raise ValueError(f"{air_label} bootstrap_full formal ABI is invalid")
    if len(returns) != 1:
        raise ValueError(f"{air_label} bootstrap_full return dataflow is invalid")
    return formals, returns[0].group("source")


def _air_top_level_stores(
    air: str,
    *,
    air_label: str,
) -> list[dict[str, Any]]:
    """Recover bootstrap block stores and their direct expression inputs."""
    function_start, function_end, function = _bootstrap_function_region(
        air, air_label=air_label
    )
    prefix = air[:function_start]
    lines = function.splitlines(keepends=True)
    offsets = []
    offset = len(prefix)
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    stores = []
    store_symbols = set()
    for store_index, line in enumerate(lines):
        store_match = re.match(r'^ {6}st\s+"([^"]+)"\s+', line)
        if store_match is None:
            continue
        if store_match.group(1) in store_symbols:
            raise ValueError(f"{air_label} duplicates a bootstrap_full store")
        store_symbols.add(store_match.group(1))
        expression_start = store_index - 1
        while expression_start >= 0:
            candidate = lines[expression_start]
            if candidate.strip() and len(candidate) - len(candidate.lstrip()) <= 6:
                break
            expression_start -= 1
        expression_start += 1
        expression_lines = lines[expression_start:store_index]
        significant = [value for value in expression_lines if value.strip()]
        if not significant:
            raise ValueError(f"{air_label} has an empty top-level store")
        root = significant[-1].strip()
        operation_match = re.search(r"\bCKKS\.([A-Za-z_][A-Za-z0-9_]*)\b", root)
        operation = (
            f"CKKS.{operation_match.group(1)}"
            if operation_match is not None
            else root.split(None, 1)[0]
        )
        operations = [
            f"CKKS.{match.group(1)}"
            for value in expression_lines
            for match in [
                re.search(r"\bCKKS\.([A-Za-z_][A-Za-z0-9_]*)\b", value)
            ]
            if match is not None
        ]
        numbers_match = re.search(r"\bnums=(?:\()?(-?[0-9]+)", root)
        dependencies = []
        constants = []
        integers = []
        scalar_literals = []
        for expression_line in expression_lines:
            load_match = re.match(r'^\s*ld\s+"([^"]+)"\s+', expression_line)
            if load_match is not None:
                dependencies.append(load_match.group(1))
            constant_match = re.match(
                r"^\s*ldc\s+CST\[(0x[0-9a-f]+|[0-9]+)\]",
                expression_line,
            )
            if constant_match is not None:
                constants.append(int(constant_match.group(1), 0))
            integer_match = re.match(
                r"^\s*intconst\s+#(0x[0-9a-f]+|[0-9]+)\b",
                expression_line,
            )
            if integer_match is not None:
                integers.append(int(integer_match.group(1), 0))
            scalar_match = re.match(
                r"^\s*ldc\s+#([^\s]+)\s+RTYPE", expression_line
            )
            if scalar_match is not None:
                scalar_literals.append(scalar_match.group(1))
        stores.append(
            {
                "symbol": store_match.group(1),
                "operation": operation,
                "operations": operations,
                "dependencies": dependencies,
                "constant_ids": constants,
                "integer_constants": integers,
                "scalar_literals": scalar_literals,
                "scalar_encode_count": sum(
                    "CKKS.encode" in value and "mask=0" in value
                    for value in expression_lines
                ),
                "rotation": (
                    int(numbers_match.group(1), 0)
                    if numbers_match is not None
                    else None
                ),
                "expression_position": offsets[expression_start],
                "store_position": offsets[store_index],
            }
        )
    if not stores:
        raise ValueError(f"{air_label} has no bootstrap_full stores")
    return stores


def _fraction_poly_add(
    left: list[Fraction], right: list[Fraction]
) -> list[Fraction]:
    result = [Fraction(0) for _ in range(max(len(left), len(right)))]
    for index in range(len(result)):
        result[index] = (
            (left[index] if index < len(left) else Fraction(0))
            + (right[index] if index < len(right) else Fraction(0))
        )
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _fraction_poly_negate(value: list[Fraction]) -> list[Fraction]:
    return [-item for item in value]


def _fraction_poly_multiply(
    left: list[Fraction], right: list[Fraction]
) -> list[Fraction]:
    result = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for left_degree, left_value in enumerate(left):
        for right_degree, right_value in enumerate(right):
            result[left_degree + right_degree] += left_value * right_value
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _validate_evalmod_scalar_manifest(manifest: dict[str, Any]) -> list[Fraction]:
    try:
        branch = manifest["single_branch"]
        full = manifest["full_program"]
        values = [
            Fraction.from_float(float.fromhex(value))
            for value in full["values_binary64_hex"]
        ]
        payload = b"".join(
            struct.pack("<dQQ", float(value), 1, 1) for value in values
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("EvalMod scalar encoding manifest is incomplete") from error
    if (
        manifest.get("schema_version")
        != "ace.phantom.bootstrap-evalmod-scalar-encodings/1.0.0"
        or manifest.get("status") != "pass"
        or manifest.get("branch_count") != 2
        or branch.get("plaintext_length") != 1
        or branch.get("scale_degree") != 1
        or full.get("plaintext_length") != 1
        or full.get("scale_degree") != 1
        or full.get("count") != len(values)
        or branch.get("count") * 2 != len(values)
        or full.get("values_binary64_hex")
        != branch.get("values_binary64_hex") * 2
        or full.get("ordered_payload_sha256") != _sha256_bytes(payload)
        or branch.get("ordered_payload_sha256")
        != _sha256_bytes(payload[: len(payload) // 2])
    ):
        raise ValueError("EvalMod scalar encoding manifest is invalid")
    return values


def _attest_evalmod_polynomial(
    *,
    air: str,
    air_label: str,
    scalar_manifest: dict[str, Any],
) -> tuple[list[Fraction], dict[str, Any]]:
    """Recover the exact expanded EvalMod polynomial from emitted AIR."""
    stores = _air_top_level_stores(air, air_label=air_label)
    records = {record["symbol"]: record for record in stores}
    indexes = {record["symbol"]: index for index, record in enumerate(stores)}
    monomials = [
        record for record in stores if record["operation"] == "CKKS.mul_mono"
    ]
    if len(monomials) != 2:
        raise ValueError(f"{air_label} EvalMod routing shape is invalid")
    first_monomial_index = indexes[monomials[0]["symbol"]]
    split_conjugates = [
        record
        for record in stores[:first_monomial_index]
        if record["operation"] == "CKKS.conjugate"
    ]
    if len(split_conjugates) != 1:
        raise ValueError(f"{air_label} EvalMod routing shape is invalid")
    conjugate = split_conjugates[0]
    conjugate_index = indexes[conjugate["symbol"]]
    split_candidates = [
        record
        for record in stores[conjugate_index + 1 :]
        if record["operation"] in {"CKKS.add", "CKKS.sub"}
        and conjugate["symbol"] in record["dependencies"]
    ]
    if (
        len(conjugate["dependencies"]) != 1
        or len(split_candidates) < 2
        or split_candidates[0]["operation"] != "CKKS.add"
        or split_candidates[1]["operation"] != "CKKS.sub"
        or Counter(split_candidates[0]["dependencies"])
        != Counter(
            (
                conjugate["symbol"],
                conjugate["dependencies"][0],
            )
        )
        or Counter(split_candidates[1]["dependencies"])
        != Counter(split_candidates[0]["dependencies"])
    ):
        raise ValueError(f"{air_label} EvalMod split dataflow is invalid")
    split_real, split_imaginary = split_candidates[:2]
    first_monomial, second_monomial = monomials
    if (
        first_monomial["dependencies"] != [split_imaginary["symbol"]]
        or len(second_monomial["dependencies"]) != 1
    ):
        raise ValueError(f"{air_label} EvalMod monomial routing is invalid")
    second_monomial_index = indexes[second_monomial["symbol"]]
    recombinations = [
        record
        for record in stores[second_monomial_index + 1 :]
        if record["operation"] == "CKKS.add"
        and second_monomial["symbol"] in record["dependencies"]
    ]
    if len(recombinations) != 1 or len(recombinations[0]["dependencies"]) != 2:
        raise ValueError(f"{air_label} EvalMod recombination dataflow is invalid")
    recombination = recombinations[0]
    first_outputs = [
        value
        for value in recombination["dependencies"]
        if value != second_monomial["symbol"]
    ]
    if len(first_outputs) != 1:
        raise ValueError(f"{air_label} EvalMod first branch output is ambiguous")
    branch_inputs = (split_real["symbol"], first_monomial["symbol"])
    branch_outputs = (first_outputs[0], second_monomial["dependencies"][0])

    scalar_values = _validate_evalmod_scalar_manifest(scalar_manifest)
    scalar_records = [
        record
        for record in stores[
            indexes[split_real["symbol"]] + 1 : second_monomial_index
        ]
        if record["scalar_encode_count"]
    ]
    if (
        len(scalar_records) != len(scalar_values)
        or any(record["scalar_encode_count"] != 1 for record in scalar_records)
        or any(len(record["scalar_literals"]) != 1 for record in scalar_records)
    ):
        raise ValueError(
            f"{air_label} EvalMod scalar encode count differs from compiler semantics"
        )
    expected_scalar_literals = [format(float(value), ".6g") for value in scalar_values]
    observed_scalar_literals = [
        record["scalar_literals"][0] for record in scalar_records
    ]
    if observed_scalar_literals != expected_scalar_literals:
        raise ValueError(
            f"{air_label} EvalMod scalar literals differ from compiler semantics"
        )
    scalar_by_symbol = {
        record["symbol"]: value
        for record, value in zip(scalar_records, scalar_values, strict=True)
    }

    def evaluate(output: str, branch_input: str) -> list[Fraction]:
        memo: dict[str, list[Fraction]] = {
            branch_input: [Fraction(0), Fraction(1)]
        }
        visiting = set()

        def visit(symbol: str) -> list[Fraction]:
            if symbol in memo:
                return memo[symbol]
            if symbol in visiting or symbol not in records:
                raise ValueError(
                    f"{air_label} EvalMod branch has an external or cyclic operand"
                )
            visiting.add(symbol)
            record = records[symbol]
            dependency_symbols = record["dependencies"]
            if (
                symbol in scalar_by_symbol
                and dependency_symbols
                and len(set(dependency_symbols)) == 1
            ):
                dependency_symbols = dependency_symbols[:1]
            operands = [visit(value) for value in dependency_symbols]
            if symbol in scalar_by_symbol:
                operands.append([scalar_by_symbol[symbol]])
            operation = record["operation"]
            if operation == "CKKS.add" and len(operands) == 2:
                value = _fraction_poly_add(operands[0], operands[1])
            elif operation == "CKKS.sub" and len(operands) == 2:
                value = _fraction_poly_add(
                    operands[0], _fraction_poly_negate(operands[1])
                )
            elif operation == "CKKS.mul" and len(operands) == 2:
                value = _fraction_poly_multiply(operands[0], operands[1])
            elif (
                operation == "CKKS.rescale"
                and "CKKS.mul" in record["operations"]
                and len(operands) == 2
            ):
                value = _fraction_poly_multiply(operands[0], operands[1])
            elif operation in {
                "CKKS.rescale",
                "CKKS.mod_switch",
                "CKKS.modswitch",
            } and len(operands) == 1:
                value = operands[0]
            else:
                raise ValueError(
                    f"{air_label} EvalMod branch has an unexpected operation "
                    f"or arity at {symbol}: {operation}, {len(operands)}"
                )
            visiting.remove(symbol)
            memo[symbol] = value
            return value

        return visit(output)

    branches = [
        evaluate(output, branch_input)
        for output, branch_input in zip(branch_outputs, branch_inputs, strict=True)
    ]
    if branches[0] != branches[1]:
        raise ValueError(f"{air_label} EvalMod branches implement different polynomials")
    polynomial = branches[0]
    rational_payload = [
        [str(value.numerator), str(value.denominator)] for value in polynomial
    ]
    return polynomial, {
        "kind": "exact-rational-emitted-evalmod-polynomial-v1",
        "degree": len(polynomial) - 1,
        "coefficient_count": len(polynomial),
        "coefficient_rational_sha256": _sha256_bytes(
            _canonical_bytes(rational_payload)
        ),
        "branch_count": 2,
        "scalar_encode_count": len(scalar_values),
        "scalar_encode_payload_sha256": scalar_manifest["full_program"][
            "ordered_payload_sha256"
        ],
        "scalar_literal_render_contract": "cxx-defaultfloat-precision-6/.6g-v1",
        "scalar_literal_sequence_sha256": _sha256_bytes(
            _canonical_bytes(observed_scalar_literals)
        ),
    }


def validate_evalmod_air_polynomial(
    *,
    air: str,
    air_label: str,
    scalar_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact emitted EvalMod polynomial attestation for one AIR."""
    _, record = _attest_evalmod_polynomial(
        air=air,
        air_label=air_label,
        scalar_manifest=scalar_manifest,
    )
    return record


def _attest_transform_group_dataflow(
    *,
    air: str,
    air_label: str,
    logical_slots: int,
    restoration_factor: float,
    expected_roles: list[dict[str, Any]],
    actual_constants: list[dict[str, Any]],
    stage_batch_symbols: dict[tuple[str, int], str],
    clear_imag: bool = False,
) -> list[dict[str, Any]]:
    """Bind baby/giant transform groups to the emitted AIR store graph."""
    stores = _air_top_level_stores(air, air_label=air_label)
    records = {record["symbol"]: record for record in stores}
    indexes = {record["symbol"]: index for index, record in enumerate(stores)}
    constant_store = {}
    for record in stores:
        for constant_id in record["constant_ids"]:
            if constant_id in constant_store:
                raise ValueError(
                    f"{air_label} transform constant appears in multiple stores"
                )
            constant_store[constant_id] = record["symbol"]

    @lru_cache(maxsize=None)
    def reachable_constants(symbol: str) -> frozenset[int]:
        record = records.get(symbol)
        if record is None:
            return frozenset()
        result = set(record["constant_ids"])
        for dependency in record["dependencies"]:
            result.update(reachable_constants(dependency))
        return frozenset(result)

    def validate_add_fold(
        symbol: str,
        expected_constant_ids: frozenset[int],
        visited: set[str] | None = None,
    ) -> None:
        """Require an add-only tree over direct rotated-constant products."""
        if visited is None:
            visited = set()
        if symbol in visited or symbol not in records:
            raise ValueError(
                f"{air_label} transform baby group has cyclic or external dataflow"
            )
        visited.add(symbol)
        record = records[symbol]
        direct_constants = record["constant_ids"]
        if (
            len(direct_constants) != len(set(direct_constants))
            or not set(direct_constants) <= expected_constant_ids
        ):
            raise ValueError(
                f"{air_label} transform baby group has unexpected constants"
            )
        fold_dependencies = [
            dependency
            for dependency in record["dependencies"]
            if reachable_constants(dependency)
        ]
        dependency_sets = [
            reachable_constants(dependency) for dependency in fold_dependencies
        ]
        combined_constants = set(direct_constants)
        for dependency_constants in dependency_sets:
            if combined_constants.intersection(dependency_constants):
                raise ValueError(
                    f"{air_label} transform baby group reuses a constant contribution"
                )
            combined_constants.update(dependency_constants)
        if frozenset(combined_constants) != reachable_constants(symbol):
            raise ValueError(
                f"{air_label} transform baby group has ambiguous constant dataflow"
            )
        additive_operand_count = len(direct_constants) + len(fold_dependencies)
        operations = Counter(record["operations"])
        expected_add_count = max(0, additive_operand_count - 1)
        if (
            additive_operand_count <= 0
            or set(operations) - {"CKKS.encode", "CKKS.mul", "CKKS.add"}
            or operations["CKKS.encode"] != len(direct_constants)
            or operations["CKKS.mul"] != len(direct_constants)
            or operations["CKKS.add"] != expected_add_count
            or record["operation"]
            != ("CKKS.add" if expected_add_count else "CKKS.mul")
            or record["scalar_encode_count"]
        ):
            raise ValueError(
                f"{air_label} transform baby group is not an add-only constant fold"
            )
        for dependency in fold_dependencies:
            validate_add_fold(
                dependency,
                expected_constant_ids,
                visited,
            )
        visited.remove(symbol)

    grouped: dict[tuple[str, int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    stage_order = []
    for role, entry in zip(expected_roles, actual_constants, strict=True):
        stage_key = (role["direction"], role["stage_order"])
        if stage_key not in stage_order:
            stage_order.append(stage_key)
        grouped.setdefault((*stage_key, role["baby_step_index"]), []).append(
            (role, entry)
        )

    stage_bindings = []
    for stage_key in stage_order:
        baby_keys = [key for key in grouped if key[:2] == stage_key]
        baby_keys.sort(key=lambda value: value[2])
        if [key[2] for key in baby_keys] != list(range(len(baby_keys))):
            raise ValueError(
                f"{air_label} transform baby-step indexes are not contiguous"
            )
        accumulator = None
        last_index = -1
        giant_rotations = []
        for baby_key in baby_keys:
            pairs = grouped[baby_key]
            roles = [pair[0] for pair in pairs]
            entries = [pair[1] for pair in pairs]
            expected_constant_ids = [entry["constant_id"] for entry in entries]
            if [role["term_order"] for role in roles] != sorted(
                role["term_order"] for role in roles
            ):
                raise ValueError(
                    f"{air_label} transform group term order is invalid"
                )
            if len({role["giant_rotation"] for role in roles}) != 1:
                raise ValueError(
                    f"{air_label} transform group giant rotation is inconsistent"
                )
            direct_symbols = [constant_store.get(value) for value in expected_constant_ids]
            if any(value is None for value in direct_symbols):
                raise ValueError(
                    f"{air_label} transform baby group does not combine its exact constants"
                )
            last_direct_index = max(indexes[value] for value in direct_symbols)
            candidates = [
                record
                for record in stores[last_direct_index:]
                if reachable_constants(record["symbol"])
                == frozenset(expected_constant_ids)
            ]
            if not candidates:
                raise ValueError(
                    f"{air_label} transform baby group does not combine its exact constants"
                )
            final_symbol = candidates[0]["symbol"]
            group_index = indexes[final_symbol]
            validate_add_fold(final_symbol, frozenset(expected_constant_ids))
            if group_index <= last_index:
                raise ValueError(
                    f"{air_label} transform baby groups are not emitted in role order"
                )
            giant = roles[0]["giant_rotation"] % logical_slots
            giant_rotations.append(giant)
            contribution = final_symbol
            contribution_index = group_index
            if giant:
                contribution_index += 1
                if contribution_index >= len(stores):
                    raise ValueError(
                        f"{air_label} transform baby group omits its giant rotation"
                    )
                rotation = stores[contribution_index]
                if (
                    rotation["operation"] != "CKKS.rotate"
                    or rotation["dependencies"] != [final_symbol]
                    or rotation["constant_ids"]
                    or rotation["rotation"] is None
                    or rotation["rotation"] % logical_slots != giant
                ):
                    raise ValueError(
                        f"{air_label} transform baby group has the wrong giant rotation"
                    )
                contribution = rotation["symbol"]
            elif baby_key[2] != 0:
                following = (
                    stores[contribution_index + 1]
                    if contribution_index + 1 < len(stores)
                    else None
                )
                if (
                    following is not None
                    and following["operation"] == "CKKS.rotate"
                    and following["dependencies"] == [final_symbol]
                ):
                    raise ValueError(
                        f"{air_label} transform baby group has an undeclared giant rotation"
                    )
            if accumulator is None:
                if baby_key[2] != 0 or giant:
                    raise ValueError(
                        f"{air_label} transform first baby group is not unrotated"
                    )
                accumulator = contribution
                last_index = contribution_index
                continue
            add_index = contribution_index + 1
            if add_index >= len(stores):
                raise ValueError(
                    f"{air_label} transform baby group omits stage accumulation"
                )
            addition = stores[add_index]
            if (
                addition["operation"] != "CKKS.add"
                or Counter(addition["dependencies"])
                != Counter((accumulator, contribution))
                or addition["constant_ids"]
            ):
                raise ValueError(
                    f"{air_label} transform baby group has the wrong stage accumulation"
                )
            accumulator = addition["symbol"]
            last_index = add_index
        if accumulator is None:
            raise ValueError(f"{air_label} transform stage has no accumulator")
        stage_bindings.append(
            {
                "direction": stage_key[0],
                "stage_order": stage_key[1],
                "baby_step_count": len(baby_keys),
                "giant_rotations": giant_rotations,
                "final_accumulator": accumulator,
            }
        )

    for current, following in zip(stage_bindings, stage_bindings[1:]):
        if current["direction"] != following["direction"]:
            continue
        following_key = (following["direction"], following["stage_order"])
        batch_symbol = stage_batch_symbols[following_key]
        batch = records.get(batch_symbol)
        if batch is None or batch["operation"] != "CKKS.rotate_batch":
            raise ValueError(
                f"{air_label} transform stage omits the following rotate batch"
            )
        if len(batch["dependencies"]) != 1:
            raise ValueError(
                f"{air_label} transform rotate batch has the wrong input arity"
            )
        cursor = batch["dependencies"][0]
        visited = set()
        while cursor != current["final_accumulator"]:
            if cursor in visited or cursor not in records:
                raise ValueError(
                    f"{air_label} transform stages are not dataflow connected"
                )
            visited.add(cursor)
            record = records[cursor]
            if (
                record["operation"] not in {"CKKS.rescale", "CKKS.mod_switch"}
                or len(record["dependencies"]) != 1
                or record["constant_ids"]
            ):
                raise ValueError(
                    f"{air_label} transform stages have an unexpected connector"
                )
            cursor = record["dependencies"][0]

    def require_connector_path(
        source: str,
        target: str,
        *,
        label: str,
    ) -> None:
        visited = set()
        cursor = source
        while cursor != target:
            if cursor in visited or cursor not in records:
                raise ValueError(f"{air_label} {label} is not dataflow connected")
            visited.add(cursor)
            connector = records[cursor]
            if (
                connector["operation"]
                not in {"CKKS.rescale", "CKKS.mod_switch", "CKKS.modswitch"}
                or len(connector["dependencies"]) != 1
                or connector["constant_ids"]
            ):
                raise ValueError(f"{air_label} {label} has an unexpected connector")
            cursor = connector["dependencies"][0]

    final_encoding = next(
        (
            record
            for record in reversed(stage_bindings)
            if record["direction"] == "coefficients-to-slots"
        ),
        None,
    )
    monomials = [
        record for record in stores if record["operation"] == "CKKS.mul_mono"
    ]
    if len(monomials) != 2:
        raise ValueError(f"{air_label} EvalMod-to-decoding boundary is invalid")
    first_monomial_index = indexes[monomials[0]["symbol"]]
    split_conjugates = [
        record
        for record in stores[:first_monomial_index]
        if record["operation"] == "CKKS.conjugate"
    ]
    if (
        final_encoding is None
        or len(split_conjugates) != 1
        or len(split_conjugates[0]["dependencies"]) != 1
    ):
        raise ValueError(f"{air_label} encoding-to-EvalMod boundary is invalid")
    require_connector_path(
        split_conjugates[0]["dependencies"][0],
        final_encoding["final_accumulator"],
        label="encoding-to-EvalMod boundary",
    )

    second_monomial = monomials[1]
    second_monomial_index = indexes[second_monomial["symbol"]]
    recombinations = [
        record
        for record in stores[second_monomial_index + 1 :]
        if record["operation"] == "CKKS.add"
        and second_monomial["symbol"] in record["dependencies"]
    ]
    first_decoding_key = ("slots-to-coefficients", 0)
    first_decoding_batch = records.get(
        stage_batch_symbols.get(first_decoding_key, "")
    )
    if (
        len(recombinations) != 1
        or len(recombinations[0]["dependencies"]) != 2
        or first_decoding_batch is None
        or first_decoding_batch["operation"] != "CKKS.rotate_batch"
        or len(first_decoding_batch["dependencies"]) != 1
    ):
        raise ValueError(f"{air_label} EvalMod-to-decoding boundary is invalid")
    require_connector_path(
        first_decoding_batch["dependencies"][0],
        recombinations[0]["symbol"],
        label="EvalMod-to-decoding boundary",
    )

    formals, return_source = _bootstrap_formals_and_return_source(
        air, air_label=air_label
    )
    first_stage_key = ("coefficients-to-slots", 0)
    first_batch = records.get(stage_batch_symbols.get(first_stage_key, ""))
    if first_batch is None or first_batch["operation"] != "CKKS.rotate_batch":
        raise ValueError(f"{air_label} omits the first encoding rotate batch")
    if len(first_batch["dependencies"]) != 1:
        raise ValueError(f"{air_label} first encoding stage has the wrong input arity")
    raised = records.get(first_batch["dependencies"][0])
    if (
        raised is None
        or raised["operation"] != "CKKS.raise_mod"
        or raised["dependencies"] != [formals[0]]
        or raised["constant_ids"]
    ):
        raise ValueError(
            f"{air_label} first encoding stage is not rooted at the raised primary input"
        )

    if (
        not math.isfinite(restoration_factor)
        or restoration_factor < 1.0
        or not restoration_factor.is_integer()
    ):
        raise ValueError(f"{air_label} restoration factor is invalid")
    restoration_integer = int(restoration_factor)
    if restoration_integer & (restoration_integer - 1):
        raise ValueError(f"{air_label} restoration factor is not a power of two")
    restoration_count = restoration_integer.bit_length() - 1
    if clear_imag:
        if restoration_count < 1:
            raise ValueError(
                f"{air_label} real projection requires a compensating restoration"
            )
        restoration_count -= 1
    cursor = return_source
    restoration_chain = []
    for _ in range(restoration_count):
        link = records.get(cursor)
        if (
            link is None
            or link["operation"] != "CKKS.add"
            or len(link["dependencies"]) != 2
            or link["dependencies"][0] != link["dependencies"][1]
            or link["constant_ids"]
            or link["operations"] != ["CKKS.add"]
        ):
            raise ValueError(
                f"{air_label} final decode result has an invalid restoration chain"
            )
        restoration_chain.append(link["symbol"])
        cursor = link["dependencies"][0]
    real_projection = None
    if clear_imag:
        projection = records.get(cursor)
        if (
            projection is None
            or projection["operation"] != "CKKS.add"
            or len(projection["dependencies"]) != 2
            or projection["dependencies"][0] == projection["dependencies"][1]
            or projection["constant_ids"]
            or projection["operations"] != ["CKKS.add"]
        ):
            raise ValueError(
                f"{air_label} final decode result has an invalid real projection"
            )
        dependency_records = [
            records.get(dependency) for dependency in projection["dependencies"]
        ]
        conjugate_indexes = [
            index
            for index, record in enumerate(dependency_records)
            if record is not None and record["operation"] == "CKKS.conjugate"
        ]
        if len(conjugate_indexes) != 1:
            raise ValueError(
                f"{air_label} final decode result has an invalid real projection"
            )
        conjugate_index = conjugate_indexes[0]
        source_index = 1 - conjugate_index
        conjugate = dependency_records[conjugate_index]
        source = projection["dependencies"][source_index]
        if (
            conjugate is None
            or conjugate["dependencies"] != [source]
            or conjugate["constant_ids"]
            or conjugate["operations"] != ["CKKS.conjugate"]
        ):
            raise ValueError(
                f"{air_label} final decode result has an invalid real projection"
            )
        real_projection = {
            "kind": "terminal-conjugate-real-projection",
            "source": source,
            "conjugate": conjugate["symbol"],
            "destination": projection["symbol"],
            "semantic_divisor": 2,
            "projected_component": "real",
            "caller_proof_required": True,
        }
        cursor = source
    final_decode = stage_bindings[-1]
    if final_decode["direction"] != "slots-to-coefficients":
        raise ValueError(f"{air_label} final transform direction is invalid")
    rescale = records.get(cursor)
    if (
        rescale is None
        or rescale["operation"] != "CKKS.rescale"
        or rescale["dependencies"] != [final_decode["final_accumulator"]]
        or rescale["constant_ids"]
        or rescale["operations"] != ["CKKS.rescale"]
    ):
        raise ValueError(
            f"{air_label} final decode accumulator does not feed restoration"
        )
    stage_bindings[0]["primary_input"] = formals[0]
    final_encoding["evalmod_input"] = split_conjugates[0]["symbol"]
    next(
        record
        for record in stage_bindings
        if record["direction"] == "slots-to-coefficients"
    )["evalmod_recombination"] = recombinations[0]["symbol"]
    terminal_restoration = {
        "rescale": rescale["symbol"],
        "self_add_count": restoration_count,
        "self_add_chain": list(reversed(restoration_chain)),
        "return_source": return_source,
    }
    if real_projection is not None:
        terminal_restoration["real_projection"] = real_projection
    stage_bindings[-1]["terminal_restoration"] = terminal_restoration
    return stage_bindings


def _attest_normalization(
    *,
    polynomial_degree: int,
    logical_slots: int,
    overflow_bound: int,
    restoration_factor: float,
    transform_payload_manifest: dict[str, Any],
    constant_manifest: dict[str, Any],
    raw_air: str,
    air_label: str = "raw AIR",
    clear_imag: bool = False,
) -> dict[str, Any]:
    if not isinstance(clear_imag, bool):
        raise ValueError("clear_imag must be a bool")
    if polynomial_degree <= 0 or logical_slots * 2 != polynomial_degree:
        raise ValueError("clear EvalMod normalization is not full-packed")
    expected_configured_factor = (
        1.0
        / float(polynomial_degree)
        / float(overflow_bound)
        / restoration_factor
    )
    try:
        if (
            transform_payload_manifest["schema_version"]
            != TRANSFORM_PAYLOAD_SCHEMA
            or transform_payload_manifest["status"] != "pass"
            or transform_payload_manifest["context"]
            != {
                "polynomial_degree": polynomial_degree,
                "logical_slots": logical_slots,
                "mul_level": transform_payload_manifest["context"][
                    "mul_level"
                ],
                "first_prime_bits": transform_payload_manifest["context"][
                    "first_prime_bits"
                ],
                "scaling_factor_bits": transform_payload_manifest[
                    "context"
                ]["scaling_factor_bits"],
                "q_part_count": transform_payload_manifest["context"][
                    "q_part_count"
                ],
                "encode_transform_budget": transform_payload_manifest[
                    "context"
                ]["encode_transform_budget"],
                "decode_transform_budget": transform_payload_manifest[
                    "context"
                ]["decode_transform_budget"],
                "overflow_bound": overflow_bound,
                "restoration_factor": restoration_factor,
            }
        ):
            raise ValueError("transform payload context differs")
        transform_normalization = transform_payload_manifest["normalization"]
        configured_factor = transform_normalization[
            "configured_coefficients_to_slots_factor"
        ]
        encoding_stages = transform_payload_manifest[
            "coefficients_to_slots"
        ]["stages"]
        decoding_stages = transform_payload_manifest[
            "slots_to_coefficients"
        ]["stages"]
        encoding_payloads = [
            payload
            for stage in encoding_stages
            for payload in stage["ordered_payload_sha256"]
        ]
        decoding_payloads = [
            payload
            for stage in decoding_stages
            for payload in stage["ordered_payload_sha256"]
        ]
        ordered_payloads = encoding_payloads + decoding_payloads
        actual_constants = constant_manifest["constants"]
        actual_payloads = [entry["payload_sha256"] for entry in actual_constants]
    except (KeyError, TypeError) as error:
        raise ValueError("transform payload semantics are incomplete") from error
    if configured_factor != expected_configured_factor or (
        transform_normalization[
            "configured_coefficients_to_slots_factor_hex"
        ]
        != float(configured_factor).hex()
    ):
        raise ValueError("CoeffToSlots normalization differs from explicit context")
    if not encoding_stages or not decoding_stages:
        raise ValueError("transform payload semantics omit a transform stage")
    for expected_order, stage in enumerate(encoding_stages):
        if (
            stage.get("stage_order") != expected_order
            or stage.get("diagonal_scale_hex")
            != float(stage.get("diagonal_scale")).hex()
            or stage.get("payload_count")
            != len(stage.get("ordered_payload_sha256", []))
            or [
                role.get("payload_sha256")
                for role in stage.get("payload_roles", [])
            ]
            != stage.get("ordered_payload_sha256")
        ):
            raise ValueError("CoeffToSlots stage semantics are inconsistent")
    for expected_order, stage in enumerate(decoding_stages):
        if (
            stage.get("stage_order") != expected_order
            or stage.get("diagonal_scale_hex")
            != float(stage.get("diagonal_scale")).hex()
            or stage.get("payload_count")
            != len(stage.get("ordered_payload_sha256", []))
            or [
                role.get("payload_sha256")
                for role in stage.get("payload_roles", [])
            ]
            != stage.get("ordered_payload_sha256")
        ):
            raise ValueError("SlotsToCoefficients stage semantics are inconsistent")
    scale_product = math.prod(
        float(stage["diagonal_scale"]) for stage in encoding_stages
    )
    transform_gain = 2 * logical_slots
    input_gain = scale_product * transform_gain
    if (
        transform_normalization["encoding_stage_scale_product"]
        != scale_product
        or transform_normalization["encoding_stage_scale_product_hex"]
        != scale_product.hex()
        or transform_normalization["nominal_full_packed_transform_gain"]
        != transform_gain
        or transform_normalization["nominal_component_input_gain"]
        != input_gain
        or transform_normalization["nominal_component_input_gain_hex"]
        != input_gain.hex()
    ):
        raise ValueError("transform payload normalization is inconsistent")
    for name, payloads in (
        ("coefficients_to_slots", encoding_payloads),
        ("slots_to_coefficients", decoding_payloads),
    ):
        record = transform_payload_manifest[name]
        if (
            record["payload_count"] != len(payloads)
            or record["ordered_payload_set_sha256"]
            != _sha256_bytes(
                b"".join(bytes.fromhex(value) for value in payloads)
            )
        ):
            raise ValueError(f"{name} payload semantics are inconsistent")
    all_payload_record = transform_payload_manifest["constant_manifest_order"]
    if (
        constant_manifest.get("schema_version") != 1
        or all_payload_record["payload_count"] != len(ordered_payloads)
        or all_payload_record["ordered_payload_sha256"] != ordered_payloads
        or all_payload_record["ordered_payload_set_sha256"]
        != _sha256_bytes(
            b"".join(bytes.fromhex(value) for value in ordered_payloads)
        )
        or actual_payloads != ordered_payloads
    ):
        raise ValueError(
            "compiler constant manifest differs from transform payload semantics"
        )
    expected_roles = []
    expected_stage_rotations = {}
    for direction, stages in (
        ("coefficients-to-slots", encoding_stages),
        ("slots-to-coefficients", decoding_stages),
    ):
        for stage in stages:
            stage_key = (direction, stage["stage_order"])
            rotations_by_index = {}
            for role in stage["payload_roles"]:
                rotation_index = role["rotation_batch_index"]
                input_rotation = role["input_rotation"]
                previous = rotations_by_index.setdefault(
                    rotation_index, input_rotation
                )
                if previous != input_rotation:
                    raise ValueError(
                        "transform payload stage has inconsistent input rotations"
                    )
                expected_roles.append(
                    {
                        "direction": direction,
                        "stage_order": stage["stage_order"],
                        "collapsed_stage": stage["collapsed_stage"],
                        "plaintext_level": stage["plaintext_level"],
                        "diagonal_scale_hex": stage["diagonal_scale_hex"],
                        **role,
                    }
                )
            if sorted(rotations_by_index) != list(range(len(rotations_by_index))):
                raise ValueError(
                    "transform payload stage rotation indexes are not contiguous"
                )
            expected_stage_rotations[stage_key] = [
                rotations_by_index[index] for index in range(len(rotations_by_index))
            ]
    if len(expected_roles) != len(actual_constants):
        raise ValueError("transform payload role count differs from constants")
    constant_role_bindings = []
    constant_load_positions = []
    raw_air_batches, rotation_outputs = _air_rotation_outputs(
        raw_air,
        logical_slots=logical_slots,
        air_label=air_label,
    )
    stage_batch_symbols = {}
    compiler_context = transform_payload_manifest["context"]
    mul_level = compiler_context["mul_level"]
    first_prime_bits = compiler_context["first_prime_bits"]
    scaling_factor_bits = compiler_context["scaling_factor_bits"]
    q_part_count = compiler_context["q_part_count"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            mul_level,
            first_prime_bits,
            scaling_factor_bits,
            q_part_count,
        )
    ):
        raise ValueError("transform payload prime context is invalid")
    q_count_per_part = (mul_level + q_part_count - 1) // q_part_count
    num_p = (
        first_prime_bits + (q_count_per_part - 1) * scaling_factor_bits + 59
    ) // 60
    expected_raw_scale = math.ldexp(1.0, scaling_factor_bits).hex()
    for index, (role, entry) in enumerate(
        zip(expected_roles, actual_constants, strict=True)
    ):
        if (
            entry.get("entry_id") != index
            or entry.get("payload_sha256") != role["payload_sha256"]
            or entry.get("slot_count") != logical_slots
            or entry.get("ace_level") != role["plaintext_level"]
            or entry.get("chain_index")
            != mul_level - role["plaintext_level"] + 1
            or entry.get("scale_degree") != 1
            or entry.get("raw_scale") != expected_raw_scale
            or entry.get("element_type") != "complex_f64"
            or entry.get("symbol") != f"_cst_{entry.get('constant_id')}"
        ):
            raise ValueError(
                "compiler constant descriptor differs from transform payload role"
            )
        observed_operand = _attest_air_constant_use(
            raw_air,
            constant_id=entry["constant_id"],
            logical_slots=logical_slots,
            ace_level=entry["ace_level"],
            num_p=num_p,
            rotation_outputs=rotation_outputs,
            air_label=air_label,
        )
        stage_key = (role["direction"], role["stage_order"])
        expected_rotations = expected_stage_rotations[stage_key]
        observed_batch = observed_operand["batch_symbol"]
        prior_batch = stage_batch_symbols.setdefault(stage_key, observed_batch)
        if (
            observed_operand["rotation_batch_index"]
            != role["rotation_batch_index"]
            or observed_operand["input_rotation"] != role["input_rotation"]
            or observed_operand["batch_rotations"] != expected_rotations
            or prior_batch != observed_batch
            or observed_operand["batch_position"] >= observed_operand["position"]
        ):
            raise ValueError(
                f"{air_label} transform payload uses the wrong rotated operand for its role"
            )
        constant_load_positions.append(observed_operand["position"])
        constant_role_bindings.append(
            {
                **role,
                "entry_id": entry["entry_id"],
                "constant_id": entry["constant_id"],
                "symbol": entry["symbol"],
                "slot_count": entry["slot_count"],
                "ace_level": entry["ace_level"],
                "chain_index": entry["chain_index"],
                "scale_degree": entry["scale_degree"],
                "raw_scale": entry["raw_scale"],
                "raw_air_rotated_operand": {
                    "batch_symbol": observed_batch,
                    "cipher_symbol": observed_operand["cipher_symbol"],
                    "rotation_batch_index": observed_operand[
                        "rotation_batch_index"
                    ],
                    "input_rotation": observed_operand["input_rotation"],
                },
            }
        )
    if constant_load_positions != sorted(constant_load_positions):
        raise ValueError(f"{air_label} transform payload loads are not role ordered")
    if set(stage_batch_symbols.values()) != set(raw_air_batches):
        raise ValueError(
            f"{air_label} rotate-batch stages differ from transform roles"
        )
    stage_dataflow = _attest_transform_group_dataflow(
        air=raw_air,
        air_label=air_label,
        logical_slots=logical_slots,
        restoration_factor=restoration_factor,
        expected_roles=expected_roles,
        actual_constants=actual_constants,
        stage_batch_symbols=stage_batch_symbols,
        clear_imag=clear_imag,
    )
    conjugates = [match.start() for match in re.finditer(r"\bCKKS\.conjugate\b", raw_air)]
    monomial_matches = list(_MONOMIAL_OPERATION.finditer(raw_air))
    monomial_powers = [int(match.group(1), 0) for match in monomial_matches]
    expected_conjugate_count = 2 if clear_imag else 1
    if (
        len(conjugates) != expected_conjugate_count
        or monomial_powers != [3 * logical_slots, logical_slots]
    ):
        raise ValueError(
            f"{air_label} does not contain the full-packed dual-EvalMod routing"
        )
    first_monomial = monomial_matches[0].start()
    split_conjugates = [value for value in conjugates if value < first_monomial]
    terminal_conjugates = [value for value in conjugates if value > monomial_matches[1].end()]
    if len(split_conjugates) != 1 or len(terminal_conjugates) != int(clear_imag):
        raise ValueError(
            f"{air_label} does not contain the full-packed dual-EvalMod routing"
        )
    conjugate = split_conjugates[0]
    split_region = raw_air[conjugate:first_monomial]
    add_position = split_region.find("CKKS.add")
    subtract_position = split_region.find("CKKS.sub")
    if add_position < 0 or subtract_position < 0 or add_position >= subtract_position:
        raise ValueError(
            f"{air_label} does not attest ordered real/imaginary splitting"
        )
    recombine_region = raw_air[monomial_matches[1].end() :]
    recombine_add = recombine_region.find("CKKS.add")
    if recombine_add < 0:
        raise ValueError(f"{air_label} does not attest dual-EvalMod recombination")
    rotate_positions = [
        match.start() for match in re.finditer(r"\bCKKS\.rotate_batch\b", raw_air)
    ]
    if not any(position < conjugate for position in rotate_positions) or not any(
        position > monomial_matches[1].end() + recombine_add
        for position in rotate_positions
    ):
        raise ValueError(
            f"{air_label} does not bracket EvalMod with both transforms"
        )
    encoding_count = len(encoding_payloads)
    if not (
        all(position < conjugate for position in constant_load_positions[:encoding_count])
        and all(
            position > monomial_matches[1].end()
            for position in constant_load_positions[encoding_count:]
        )
    ):
        raise ValueError(
            f"{air_label} transform payload roles do not bracket dual EvalMod"
        )
    return {
        "kind": "full-packed-coefficients-slots-dual-evalmod-v1",
        "compiler_context": {
            "polynomial_degree": polynomial_degree,
            "logical_slots": logical_slots,
            "full_packing_relation": "polynomial_degree=2*logical_slots",
        },
        "coefficients_to_slots": {
            "configured_factor": configured_factor,
            "configured_factor_hex": configured_factor.hex(),
            "emitted_stage_scale_product": scale_product,
            "emitted_stage_scale_product_hex": scale_product.hex(),
            "factor_derivation": "1/polynomial_degree/overflow_bound/restoration_factor",
        },
        "nominal_transform_gain": {
            "dft_gain": logical_slots,
            "conjugate_split_gain": 2,
            "combined_gain": transform_gain,
        },
        "nominal_component_input_gain": input_gain,
        "nominal_component_input_gain_hex": input_gain.hex(),
        "gain_scope": "nominal-full-packed-transform-stage-scaling",
        "operator_rounding_treatment": (
            "binary64-twiddle-and-per-entry-matrix-rounding-is-reserved-"
            "provider-numerical-error"
        ),
        "decode_payload_scope": (
            "artifact-and-terminal-path-closure-not-clear-map-input-gain"
        ),
        "compiler_transform_payload_semantics": transform_payload_manifest,
        "constant_role_bindings": constant_role_bindings,
        "transform_stage_dataflow": stage_dataflow,
        "raw_air_topology": {
            "conjugate_count": len(conjugates),
            "split_operation_order": ["conjugate", "add", "subtract"],
            "monomial_routing_powers": monomial_powers,
            "transform_placement": "before-split-and-after-recombine",
            **(
                {"output_projection": {
                    "kind": "terminal-conjugate-real-projection",
                    "caller_proof_required": True,
                }}
                if clear_imag
                else {}
            ),
        },
        "matrix_constant_authority": (
            "compiler-emitted-source-and-constant-manifest-artifact-bindings"
        ),
    }


def validate_transform_air_semantics(
    *,
    polynomial_degree: int,
    logical_slots: int,
    overflow_bound: int,
    restoration_factor: float,
    transform_payload_manifest: dict[str, Any],
    constant_manifest: dict[str, Any],
    air: str,
    air_label: str,
    clear_imag: bool = False,
) -> dict[str, Any]:
    """Replay transform-role and stage dataflow against an emitted AIR form."""
    return _attest_normalization(
        polynomial_degree=polynomial_degree,
        logical_slots=logical_slots,
        overflow_bound=overflow_bound,
        restoration_factor=restoration_factor,
        transform_payload_manifest=transform_payload_manifest,
        constant_manifest=constant_manifest,
        raw_air=air,
        air_label=air_label,
        clear_imag=clear_imag,
    )


def derive_supported_identity_domain(
    *,
    coefficients: Sequence[float],
    scalars: Sequence[float],
    overflow_bound: int,
    restoration_factor: float,
    evalmod_lower: float,
    evalmod_upper: float,
    provider_clear_threshold: float,
    artifact_bindings: dict[str, str],
    polynomial_degree: int,
    logical_slots: int,
    transform_payload_manifest: dict[str, Any],
    constant_manifest: dict[str, Any],
    evalmod_scalar_manifest: dict[str, Any],
    raw_air: str,
    clear_imag: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(clear_imag, bool):
        raise ValueError("clear_imag must be a bool")
    if not math.isfinite(provider_clear_threshold) or provider_clear_threshold <= 0:
        raise ValueError("provider clear threshold must be finite and positive")
    if (evalmod_lower, evalmod_upper) != (-1.0, 1.0):
        raise ValueError("unsupported emitted EvalMod polynomial interval")
    if isinstance(overflow_bound, bool) or not isinstance(overflow_bound, int) or overflow_bound <= 0:
        raise ValueError("clear EvalMod overflow bound must be a positive integer")
    if not math.isfinite(restoration_factor) or restoration_factor <= 0:
        raise ValueError("clear EvalMod restoration factor must be positive")
    if not coefficients or any(not math.isfinite(value) for value in coefficients):
        raise ValueError("clear EvalMod coefficients must be finite and nonempty")
    if any(not math.isfinite(value) for value in scalars):
        raise ValueError("clear EvalMod double-angle scalars must be finite")
    _validate_artifact_bindings(artifact_bindings)
    coefficient_hex = tuple(float(value).hex() for value in coefficients)
    scalar_hex = tuple(float(value).hex() for value in scalars)
    clear_map_budget = provider_clear_threshold * CLEAR_MAP_BUDGET_FRACTION
    normalization = _attest_normalization(
        polynomial_degree=polynomial_degree,
        logical_slots=logical_slots,
        overflow_bound=overflow_bound,
        restoration_factor=restoration_factor,
        transform_payload_manifest=transform_payload_manifest,
        constant_manifest=constant_manifest,
        raw_air=raw_air,
        clear_imag=clear_imag,
    )
    executed_polynomial, emitted_polynomial_record = _attest_evalmod_polynomial(
        air=raw_air,
        air_label="raw AIR",
        scalar_manifest=evalmod_scalar_manifest,
    )
    proof = _derive_polynomial_proof(
        tuple(
            (value.numerator, value.denominator)
            for value in executed_polynomial
        ),
        overflow_bound,
        float(restoration_factor).hex(),
        float(normalization["nominal_component_input_gain"]).hex(),
        float(clear_map_budget).hex(),
    )
    expression = {
        "topology": "full-packed-separable-dual-evalmod",
        "component_map": (
            "restoration*P(nominal_component_input_gain*"
            "(integer_phase*restoration+message))"
        ),
        "chebyshev": {
            "basis": "first-kind",
            "constant_term_weight": 0.5,
            "coordinate_interval": [evalmod_lower, evalmod_upper],
            "coefficients_binary64_hex": list(coefficient_hex),
            "coefficient_payload_sha256": _float_sequence_sha256(coefficients),
        },
        "double_angle": {
            "recurrence": "square-double-add-ordered-scalar",
            "scalars_binary64_hex": list(scalar_hex),
            "scalar_payload_sha256": _float_sequence_sha256(scalars),
        },
        "emitted_evalmod_polynomial": emitted_polynomial_record,
        "overflow_bound": overflow_bound,
        "restoration_factor": restoration_factor,
        "restoration_factor_hex": float(restoration_factor).hex(),
        "phase_contract": {
            "integer_phase_range_inclusive": [-overflow_bound, overflow_bound],
            "integer_phase_source": "integer_phase*restoration",
            "nominal_component_input_gain": normalization[
                "nominal_component_input_gain"
            ],
            "nominal_component_input_gain_hex": normalization[
                "nominal_component_input_gain_hex"
            ],
            "polynomial_fit_interval": [evalmod_lower, evalmod_upper],
            "proof_evaluation_coordinate_envelope": proof[
                "proof_evaluation_coordinate_envelope"
            ],
            "proof_envelope_covers_all_declared_phases": True,
        },
    }
    error_contract = {
        "target": "original-clear-complex-identity",
        "aggregate_norm": "complex-absolute-l2",
        "provider_clear_maximum_absolute": provider_clear_threshold,
        "clear_map_budget_fraction": CLEAR_MAP_BUDGET_FRACTION,
        "maximum_complex_clear_map_error": clear_map_budget,
        "reserved_provider_numerical_error": (
            provider_clear_threshold - clear_map_budget
        ),
        "reserved_provider_numerical_error_scope": [
            "binary64-transform-matrix-and-twiddle-rounding",
            "encrypted-provider-arithmetic-and-decoding",
        ],
    }
    radius = proof["selected_radius"]
    attestation = {
        "schema_version": DOMAIN_ATTESTATION_SCHEMA,
        "status": "attested",
        "scope": {
            "purpose": "domain-attestation-only",
            "provider_value_oracle": False,
            "may_supply_expected_case_values": False,
        },
        "artifact_bindings": dict(sorted(artifact_bindings.items())),
        "normalization": normalization,
        "expanded_clear_component_map": expression,
        "expanded_clear_component_map_sha256": _sha256_bytes(
            _canonical_bytes(expression)
        ),
        "error_contract": error_contract,
        "proof": proof,
    }
    output_projection = {
        "kind": "conjugate-real-projection",
        "caller_proof_required": True,
        "semantic_input_requirement": "real-valued",
    }
    if clear_imag:
        attestation["output_projection"] = output_projection
    attestation_sha256 = _sha256_bytes(_canonical_bytes(attestation))
    domain = {
        "kind": (
            "centered-evalmod-real-projection-error-bounded"
            if clear_imag
            else "centered-evalmod-complex-error-bounded"
        ),
        "components": ["real"] if clear_imag else ["real", "imaginary"],
        "lower_exclusive": -radius,
        "upper_exclusive": radius,
        "period": restoration_factor,
        "evidence": {
            "attestation_schema_version": DOMAIN_ATTESTATION_SCHEMA,
            "attestation_sha256": attestation_sha256,
            "provider_clear_maximum_absolute": provider_clear_threshold,
            "maximum_complex_clear_map_error": clear_map_budget,
            "reserved_provider_numerical_error": (
                provider_clear_threshold - clear_map_budget
            ),
        },
    }
    if clear_imag:
        domain["output_projection"] = output_projection
    return domain, attestation


def validate_supported_identity_domain(
    domain: Any,
    attestation: Any,
    *,
    provider_clear_threshold: float,
    artifact_bindings: dict[str, str],
    constant_manifest: dict[str, Any],
    evalmod_scalar_manifest: dict[str, Any],
    raw_air: str,
    clear_imag: bool = False,
) -> None:
    try:
        expression = attestation["expanded_clear_component_map"]
        chebyshev = expression["chebyshev"]
        double_angle = expression["double_angle"]
        coefficient_hex = chebyshev["coefficients_binary64_hex"]
        scalar_hex = double_angle["scalars_binary64_hex"]
        coordinate_interval = chebyshev["coordinate_interval"]
        coefficients = [float.fromhex(value) for value in coefficient_hex]
        scalars = [float.fromhex(value) for value in scalar_hex]
        if [value.hex() for value in coefficients] != coefficient_hex:
            raise ValueError("clear EvalMod coefficient hex values are not canonical")
        if [value.hex() for value in scalars] != scalar_hex:
            raise ValueError("clear EvalMod scalar hex values are not canonical")
        normalization = attestation["normalization"]
        compiler_context = normalization["compiler_context"]
        transform_payload_manifest = normalization[
            "compiler_transform_payload_semantics"
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("clear EvalMod attestation shape is invalid") from error
    expected_domain, expected_attestation = derive_supported_identity_domain(
        coefficients=coefficients,
        scalars=scalars,
        overflow_bound=expression["overflow_bound"],
        restoration_factor=expression["restoration_factor"],
        evalmod_lower=coordinate_interval[0],
        evalmod_upper=coordinate_interval[1],
        provider_clear_threshold=provider_clear_threshold,
        artifact_bindings=artifact_bindings,
        polynomial_degree=compiler_context["polynomial_degree"],
        logical_slots=compiler_context["logical_slots"],
        transform_payload_manifest=transform_payload_manifest,
        constant_manifest=constant_manifest,
        evalmod_scalar_manifest=evalmod_scalar_manifest,
        raw_air=raw_air,
        clear_imag=clear_imag,
    )
    if attestation != expected_attestation or domain != expected_domain:
        raise ValueError("clear EvalMod identity-domain attestation differs on replay")
