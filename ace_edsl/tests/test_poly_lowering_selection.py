"""Selection contracts for CKKS-to-POLY lowering strategies."""

from __future__ import annotations

import os
import sys

import pytest

from ace_bindings import air_builder
from ace_edsl.edsl import AcePipeline, FHEConfig, Pipeline, PolyLowering


def test_fhe_config_normalizes_and_validates_poly_lowering():
    assert FHEConfig().poly_lowering == "spoly"
    assert FHEConfig(poly_lowering=PolyLowering.LINEAR_TRANSFORM).poly_lowering == (
        "linear_transform"
    )
    assert FHEConfig(poly_lowering=" Linear_Transform ").poly_lowering == (
        "linear_transform"
    )

    with pytest.raises(ValueError, match="must be 'spoly' or 'linear_transform'"):
        FHEConfig(poly_lowering="unknown")
    with pytest.raises(TypeError, match="PolyLowering or string"):
        FHEConfig(poly_lowering=object())
    with pytest.raises(ValueError, match="requires codegen_ir='poly'"):
        FHEConfig(
            provider="ant",
            codegen_ir="ckks",
            poly_lowering="linear_transform",
        )


def test_both_pipeline_runners_forward_poly_lowering(monkeypatch):
    calls = []

    def fake_run_poly_driver(glob, poly_lowering="spoly"):
        calls.append((glob, poly_lowering))
        return {"success": True, "message": ""}

    monkeypatch.setattr(air_builder, "run_poly_driver", fake_run_poly_driver)
    first_glob = object()
    ace_pipeline = AcePipeline(
        first_glob,
        FHEConfig(poly_lowering=PolyLowering.LINEAR_TRANSFORM),
    )
    ace_pipeline._air_builder = air_builder
    assert ace_pipeline.run_poly_driver()["success"]

    second_glob = object()
    pipeline = Pipeline(dump_ir=False, verbose=False).set_glob(second_glob)
    pipeline.config = FHEConfig(poly_lowering="linear_transform")
    assert pipeline._run_phase("poly_driver")

    assert calls == [
        (first_glob, "linear_transform"),
        (second_glob, "linear_transform"),
    ]


def test_bootstrap_selects_linear_transform_lowering_iff_it_emits_contract():
    examples = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples")
    )
    if examples not in sys.path:
        sys.path.insert(0, examples)
    from bootstrap_full import (  # pylint: disable=import-outside-toplevel
        _bootstrap_poly_lowering,
        build_bootstrap_trace_config,
    )

    common = dict(
        poly_degree=16,
        mul_level=4,
        first_prime_bits=60,
        scaling_factor_bits=20,
        hamming_weight=8,
        q_parts=1,
        enc_budget=1,
        dec_budget=1,
        ct_encode=False,
    )
    assert _bootstrap_poly_lowering(
        build_bootstrap_trace_config(**common, emit_linear_transform=False)
    ) == "spoly"
    assert _bootstrap_poly_lowering(
        build_bootstrap_trace_config(**common, emit_linear_transform=True)
    ) == "linear_transform"
