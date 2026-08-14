"""Focused tests for the explicit real-output bootstrap trace mode."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
for path in (REPOSITORY, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bootstrap_full as bootstrap_module  # noqa: E402
from ace_edsl.edsl import AceEDSL, CkksCiphertext  # noqa: E402


class _FakeCiphertext:
    def __init__(self) -> None:
        self.raise_calls: list[tuple[int, bool]] = []

    def raise_mod(self, level: int, *, runtime_raise_level: bool):
        self.raise_calls.append((level, runtime_raise_level))
        return self


def _config(*, clear_imag: bool = False):
    return bootstrap_module.build_bootstrap_trace_config(
        poly_degree=32,
        mul_level=7,
        first_prime_bits=45,
        scaling_factor_bits=41,
        hamming_weight=64,
        q_parts=2,
        enc_budget=2,
        dec_budget=3,
        ct_encode=False,
        clear_imag=clear_imag,
    )


@pytest.mark.parametrize("clear_imag", (False, True))
def test_bootstrap_full_forwards_explicit_real_output_mode(clear_imag: bool) -> None:
    config = _config(clear_imag=clear_imag)
    source = _FakeCiphertext()
    expected = object()
    kernel_body = getattr(
        bootstrap_module.bootstrap_full,
        "__wrapped__",
        bootstrap_module.bootstrap_full,
    )

    with bootstrap_module.bootstrap_trace_configuration(config):
        with patch(
            "ace_edsl.edsl.core.bootstrap_decomposition."
            "fullpacked_bootstrap_primitive",
            return_value=expected,
        ) as primitive:
            observed = kernel_body(source, *([None] * 61))

    assert observed is expected
    assert source.raise_calls == [(config.raise_level, False)]
    primitive.assert_called_once_with(
        source,
        config=config,
        clear_imag=clear_imag,
    )


def test_bootstrap_config_rejects_non_boolean_real_output_mode() -> None:
    with pytest.raises(ValueError, match="clear_imag must be a bool"):
        _config(clear_imag=1)  # type: ignore[arg-type]


def test_real_output_mode_requires_a_compensating_post_scale() -> None:
    with pytest.raises(ValueError, match="first_prime_bits > scaling_factor_bits"):
        bootstrap_module.build_bootstrap_trace_config(
            poly_degree=32,
            mul_level=7,
            first_prime_bits=41,
            scaling_factor_bits=41,
            hamming_weight=64,
            q_parts=2,
            enc_budget=2,
            dec_budget=3,
            ct_encode=False,
            clear_imag=True,
        )


def test_real_output_trace_adds_one_final_conjugate() -> None:
    conjugate_counts = []
    for clear_imag in (False, True):
        config = _config(clear_imag=clear_imag)
        AceEDSL._get_dsl.cache_clear()
        ciphertext = CkksCiphertext(shape=(config.poly_degree,), name="input_ct")
        zero = CkksCiphertext(shape=(config.poly_degree,), name="zero_ct")
        arguments = (
            [ciphertext, zero, 1.0]
            + list(bootstrap_module.G_COEFFICIENTS_UNIFORM_HW_192)
            + list(config.double_angle_scalars)
            + [config.post_scale]
        )
        with bootstrap_module.bootstrap_trace_configuration(config):
            bootstrap_module.bootstrap_full(*arguments)
        module = AceEDSL._get_dsl().current_air_module
        assert module is not None
        conjugate_counts.append(module.dump().count("CKKS.conjugate"))

    assert conjugate_counts[1] == conjugate_counts[0] + 1
