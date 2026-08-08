"""Algebraic guard for the retained centered component-wise raise oracle."""

from __future__ import annotations


def _center(residue: int, modulus: int) -> int:
    return residue if residue <= modulus // 2 else residue - modulus


def test_component_wise_centered_raise_is_not_a_decoded_identity() -> None:
    """Centering does not commute with ciphertext-component addition."""

    source_modulus = 17
    target_modulus = 19
    c0 = c1 = 8

    # At q0 the two components decrypt (for the toy secret s=1) to -1.
    decoded_at_source = _center((c0 + c1) % source_modulus, source_modulus)
    assert decoded_at_source == -1

    # The primitive centers and lifts c0/c1 independently.  Adding those
    # lifted components at q1 produces 16, not the lift of decoded -1 (18).
    component_wise_lift = (
        _center(c0, source_modulus) + _center(c1, source_modulus)
    ) % target_modulus
    decoded_value_lift = decoded_at_source % target_modulus
    assert component_wise_lift == 16
    assert decoded_value_lift == 18
    assert component_wise_lift != decoded_value_lift
