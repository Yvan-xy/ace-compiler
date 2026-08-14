from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools" / "phantom_gpu"
ADAPTER = REPO_ROOT / "fhe-cmplr" / "rtlib" / "phantom" / "src" / "phantom_lib.cu"
sys.path.insert(0, str(TOOLS_ROOT))

import compare_retained_ckks_results as retained  # noqa: E402
import ordinary_ckks_fixture as ordinary  # noqa: E402


DATA_Q_REQUESTED_SIZES = (60,) + (56,) * 25
SPECIAL_P_REQUESTED_SIZES = (60,) * 9

FROZEN_Q_16384 = (
    1152921504606748673,
    72057594044153857,
    72057594031144961,
    72057594042646529,
    72057594031276033,
    72057594042449921,
    72057594031865857,
    72057594042286081,
    72057594031964161,
    72057594042089473,
    72057594033012737,
    72057594041368577,
    72057594034913281,
    72057594041106433,
    72057594035306497,
    72057594040680449,
    72057594036256769,
    72057594040320001,
    72057594036551681,
    72057594039992321,
    72057594036879361,
    72057594039205889,
    72057594037338113,
    72057594038747137,
    72057594037370881,
    72057594038321153,
)

FROZEN_P_16384 = (
    1152921504606683137,
    1152921504606584833,
    1152921504605962241,
    1152921504604979201,
    1152921504600260609,
    1152921504599080961,
    1152921504598720513,
    1152921504597114881,
    1152921504597016577,
)

FROZEN_Q_65536 = (
    1152921504606584833,
    72057594069123073,
    72057594004111361,
    72057594065453057,
    72057594006863873,
    72057594063093761,
    72057594011189249,
    72057594062831617,
    72057594020626433,
    72057594062438401,
    72057594021150721,
    72057594061651969,
    72057594023903233,
    72057594058899457,
    72057594027704321,
    72057594058375169,
    72057594029015041,
    72057594057195521,
    72057594030981121,
    72057594047889409,
    72057594034913281,
    72057594042646529,
    72057594035306497,
    72057594040680449,
    72057594036879361,
    72057594038321153,
)

FROZEN_P_65536 = (
    1152921504598720513,
    1152921504597016577,
    1152921504595968001,
    1152921504592822273,
    1152921504592429057,
    1152921504589938689,
    1152921504586530817,
    1152921504583647233,
    1152921504581419009,
)


# These seven witnesses make Miller-Rabin deterministic for every uint64_t.
_UINT64_WITNESSES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _is_prime(candidate: int) -> bool:
    if candidate < 2:
        return False
    for divisor in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if candidate % divisor == 0:
            return candidate == divisor

    odd_factor = candidate - 1
    power_of_two = 0
    while odd_factor % 2 == 0:
        odd_factor //= 2
        power_of_two += 1
    for witness in _UINT64_WITNESSES:
        if witness % candidate == 0:
            continue
        residue = pow(witness, odd_factor, candidate)
        if residue in (1, candidate - 1):
            continue
        for _ in range(power_of_two - 1):
            residue = residue * residue % candidate
            if residue == candidate - 1:
                break
        else:
            return False
    return True


def _ant_first_prime(ring_degree: int, requested_size: int) -> int:
    order = 2 * ring_degree
    candidate = (1 << requested_size) + order + 1
    while not _is_prime(candidate):
        candidate += order
    return candidate


def _ant_previous_prime(prime: int, order: int) -> int:
    candidate = prime - order
    while not _is_prime(candidate):
        candidate -= order
    return candidate


def _ant_next_prime(prime: int, order: int) -> int:
    # ANT's Gen_next_prime skips prime + order and tests prime + 2*order first.
    candidate = prime + order
    while True:
        candidate += order
        if _is_prime(candidate):
            return candidate


def _ant_q_primes(
    ring_degree: int,
    count: int,
    first_requested_size: int,
    scaling_requested_size: int,
) -> tuple[int, ...]:
    order = 2 * ring_degree
    primes = [0] * count
    lower = _ant_first_prime(ring_degree, scaling_requested_size)
    upper = lower
    primes[-1] = lower
    for offset, index in enumerate(range(count - 2, 0, -1)):
        if offset % 2 == 0:
            lower = _ant_previous_prime(lower, order)
            primes[index] = lower
        else:
            upper = _ant_next_prime(upper, order)
            primes[index] = upper
    if first_requested_size == scaling_requested_size:
        primes[0] = _ant_previous_prime(lower, order)
    else:
        first = _ant_first_prime(ring_degree, first_requested_size)
        primes[0] = _ant_previous_prime(first, order)
    return tuple(primes)


def _ant_p_primes(
    ring_degree: int,
    count: int,
    requested_size: int,
    data_q: tuple[int, ...],
) -> tuple[int, ...]:
    order = 2 * ring_degree
    used = set(data_q)
    cursor = _ant_first_prime(ring_degree, requested_size)
    primes = []
    for _ in range(count):
        while True:
            cursor = _ant_previous_prime(cursor, order)
            if cursor not in used:
                break
        primes.append(cursor)
        used.add(cursor)
    return tuple(primes)


def _requested_prime_size(prime: int) -> int:
    floor_log2 = prime.bit_length() - 1
    lower = 1 << floor_log2
    upper = lower << 1
    return floor_log2 if prime - lower <= upper - prime else floor_log2 + 1


@pytest.mark.parametrize(
    ("ring_degree", "frozen_q", "frozen_p"),
    (
        (16384, FROZEN_Q_16384, FROZEN_P_16384),
        (65536, FROZEN_Q_65536, FROZEN_P_65536),
    ),
)
def test_ant_prime_policy_matches_complete_frozen_q_and_p_chains(
    ring_degree: int,
    frozen_q: tuple[int, ...],
    frozen_p: tuple[int, ...],
) -> None:
    data_q = _ant_q_primes(
        ring_degree,
        len(DATA_Q_REQUESTED_SIZES),
        DATA_Q_REQUESTED_SIZES[0],
        DATA_Q_REQUESTED_SIZES[1],
    )
    special_p = _ant_p_primes(
        ring_degree,
        len(SPECIAL_P_REQUESTED_SIZES),
        SPECIAL_P_REQUESTED_SIZES[0],
        data_q,
    )

    assert data_q == frozen_q
    assert special_p == frozen_p

    complete_chain = data_q + special_p
    requested_sizes = DATA_Q_REQUESTED_SIZES + SPECIAL_P_REQUESTED_SIZES
    assert len(set(complete_chain)) == len(complete_chain)
    assert all(_is_prime(prime) for prime in complete_chain)
    assert all(prime % (2 * ring_degree) == 1 for prime in complete_chain)
    assert tuple(map(_requested_prime_size, complete_chain)) == requested_sizes

    above_scaling_power = [prime for prime in data_q[1:] if prime > (1 << 56)]
    assert above_scaling_power
    assert all(prime.bit_length() == 57 for prime in above_scaling_power)
    assert all(_requested_prime_size(prime) == 56 for prime in above_scaling_power)


def test_adapter_constructs_the_context_with_the_ant_prime_chain() -> None:
    source = ADAPTER.read_text(encoding="utf-8")

    builder_start = source.index("BuildAntCompatibleCoefficientModuli(")
    builder_end = source.index("CheckedScaleDegree(", builder_start)
    builder = source[builder_start:builder_end]
    build_call = source.index("BuildAntCompatibleCoefficientModuli(*_manifest)")
    install_call = source.index("parameters.set_coeff_modulus(coefficient_moduli)")
    assert build_call < install_call
    assert source.count("BuildAntCompatibleCoefficientModuli(*_manifest)") == 1
    assert "CoeffModulus::Create" not in source
    assert "AntPreviousPrime(lower, order)" in builder
    assert "AntNextPrime(upper, order)" in builder
    assert "RequestedPrimeSize(prime)" in builder
    assert "RequestedPrimeSize(special_cursor)" in builder
    assert "original - lower <= upper - original" in source
    assert "_ordered_coefficient_moduli.push_back(modulus.value())" in source


@pytest.mark.parametrize(
    ("prime", "expected"),
    (
        (2, 1),
        ((1 << 56) - 1, 56),
        (1 << 56, 56),
        (FROZEN_Q_65536[1], 56),
        (3 << 55, 56),
        ((3 << 55) + 1, 57),
        ((1 << 57) - 1, 57),
    ),
)
@pytest.mark.parametrize(
    "helper",
    (ordinary.requested_prime_size, retained.requested_prime_size),
    ids=("ordinary", "retained"),
)
def test_requested_prime_size_uses_the_nearest_power_convention(
    helper: object,
    prime: int,
    expected: int,
) -> None:
    assert callable(helper)
    assert helper(prime) == expected


@pytest.mark.parametrize(
    ("helper", "error_type"),
    (
        (ordinary.requested_prime_size, ordinary.OrdinaryCkksError),
        (retained.requested_prime_size, retained.ComparisonError),
    ),
    ids=("ordinary", "retained"),
)
@pytest.mark.parametrize("invalid", (True, 1, 0, -1, 2.0))
def test_requested_prime_size_rejects_non_integer_or_too_small_inputs(
    helper: object,
    error_type: type[Exception],
    invalid: object,
) -> None:
    assert callable(helper)
    with pytest.raises(error_type, match=r"integer at least (?:two|2)"):
        helper(invalid)
