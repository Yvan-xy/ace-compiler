from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import math
from pathlib import Path
import re
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[3]
TOOLS = REPOSITORY / "tools" / "phantom_gpu"
TESTS = TOOLS / "tests"
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(EXAMPLES))

import bootstrap_domain_attestation as domain_attestation  # noqa: E402
from bootstrap_domain_test_support import transform_authorities  # noqa: E402
from bootstrap_full import build_bootstrap_trace_config  # noqa: E402
from ace_edsl.edsl.core.bootstrap_decomposition import (  # noqa: E402
    build_bootstrap_evalmod_scalar_manifest,
)


ARTIFACT_BINDINGS = {
    name: "0" * 64
    for name in domain_attestation.DOMAIN_ARTIFACT_BINDING_KEYS
}


def current_profile() -> tuple[dict, dict, object, dict, dict, str]:
    config = build_bootstrap_trace_config(
        poly_degree=16384,
        mul_level=26,
        first_prime_bits=60,
        scaling_factor_bits=56,
        hamming_weight=192,
        q_parts=3,
        enc_budget=3,
        dec_budget=3,
        ct_encode=False,
    )
    transform, constants, raw_air = transform_authorities(config)
    supported, attestation = domain_attestation.derive_supported_identity_domain(
        coefficients=config.chebyshev_coefficients,
        scalars=config.double_angle_scalars,
        overflow_bound=config.eval_sin_upper_bound_k,
        restoration_factor=config.post_scale,
        evalmod_lower=-1.0,
        evalmod_upper=1.0,
        provider_clear_threshold=1.0e-2,
        artifact_bindings=ARTIFACT_BINDINGS,
        polynomial_degree=config.poly_degree,
        logical_slots=config.slots,
        transform_payload_manifest=transform,
        constant_manifest=constants,
        evalmod_scalar_manifest=build_bootstrap_evalmod_scalar_manifest(config),
        raw_air=raw_air,
    )
    return supported, attestation, config, transform, constants, raw_air


def test_current_emitted_map_has_a_directed_conservative_domain() -> None:
    supported, attestation, _, _, _, _ = current_profile()
    proof = attestation["proof"]
    assert supported["kind"] == "centered-evalmod-complex-error-bounded"
    assert supported["upper_exclusive"].hex() == "0x1.081ecb37b0e2dp-1"
    assert supported["lower_exclusive"] == -supported["upper_exclusive"]
    assert supported["upper_exclusive"] < supported["period"] / 2.0
    selected = Decimal(
        proof["selected_complex_clear_map_error_squared_bound"]
    )
    budget = Decimal(
        proof["maximum_complex_clear_map_error_squared_lower_bound"]
    )
    outward = Decimal(
        proof["next_outward_complex_clear_map_error_squared_envelope"]
    )
    assert selected <= budget < outward
    assert proof["arithmetic"]["transcendental_approximations_used"] is False
    assert (
        proof["coefficient_envelope_sha256"]
        == "579280ba196befb96c754c69b92ae00424b415c74f7d1bd75fa8ba111cd15741"
    )
    assert (
        attestation["normalization"]["nominal_component_input_gain_hex"]
        == "0x1.0000000000005p-9"
    )
    coordinate = proof["proof_evaluation_coordinate_envelope"]
    assert Decimal(coordinate["upper_directed_decimal"]) > Decimal(1)
    assert coordinate["fit_interval_containment_required"] is False
    assert attestation["scope"] == {
        "purpose": "domain-attestation-only",
        "provider_value_oracle": False,
        "may_supply_expected_case_values": False,
    }


def test_proof_is_independent_of_the_ambient_decimal_context() -> None:
    supported, attestation, _, _, _, _ = current_profile()
    domain_attestation._derive_polynomial_proof.cache_clear()
    with localcontext() as context:
        context.prec = 9
        context.rounding = "ROUND_HALF_EVEN"
        repeated_supported, repeated_attestation, _, _, _, _ = current_profile()
    assert repeated_supported == supported
    assert repeated_attestation == attestation


def test_attestation_replay_rejects_semantic_or_provenance_tampering() -> None:
    supported, attestation, config, _, constants, raw_air = current_profile()
    scalar_manifest = build_bootstrap_evalmod_scalar_manifest(config)
    domain_attestation.validate_supported_identity_domain(
        supported,
        attestation,
        provider_clear_threshold=1.0e-2,
        artifact_bindings=ARTIFACT_BINDINGS,
        constant_manifest=constants,
        evalmod_scalar_manifest=scalar_manifest,
        raw_air=raw_air,
    )
    changed = deepcopy(attestation)
    changed["scope"]["provider_value_oracle"] = True
    with pytest.raises(ValueError, match="differs on replay"):
        domain_attestation.validate_supported_identity_domain(
            supported,
            changed,
            provider_clear_threshold=1.0e-2,
            artifact_bindings=ARTIFACT_BINDINGS,
            constant_manifest=constants,
            evalmod_scalar_manifest=scalar_manifest,
            raw_air=raw_air,
        )

    monomial_positions = [
        match.start() for match in re.finditer(r"CKKS\.mul_mono\b", raw_air)
    ]
    assert len(monomial_positions) == 2
    evalmod_multiply_matches = list(
        re.finditer(
            r"CKKS\.mul RTYPE",
            raw_air[monomial_positions[0] : monomial_positions[1]],
        )
    )
    assert len(evalmod_multiply_matches) > 1
    first_multiply = monomial_positions[0] + evalmod_multiply_matches[0].start()
    one_branch_wrong = (
        raw_air[:first_multiply]
        + raw_air[first_multiply:].replace("CKKS.mul", "CKKS.add", 1)
    )
    with pytest.raises(ValueError, match="different polynomials"):
        domain_attestation.validate_evalmod_air_polynomial(
            air=one_branch_wrong,
            air_label="one-branch mutated AIR",
            scalar_manifest=scalar_manifest,
        )

    assert len(evalmod_multiply_matches) % 2 == 0
    corresponding = monomial_positions[0] + evalmod_multiply_matches[
        len(evalmod_multiply_matches) // 2
    ].start()
    both_branches_wrong = raw_air
    for position in sorted((first_multiply, corresponding), reverse=True):
        both_branches_wrong = (
            both_branches_wrong[:position]
            + both_branches_wrong[position:].replace(
                "CKKS.mul", "CKKS.add", 1
            )
        )
    with pytest.raises(ValueError, match="differs on replay|budget is exhausted"):
        domain_attestation.validate_supported_identity_domain(
            supported,
            attestation,
            provider_clear_threshold=1.0e-2,
            artifact_bindings=ARTIFACT_BINDINGS,
            constant_manifest=constants,
            evalmod_scalar_manifest=scalar_manifest,
            raw_air=both_branches_wrong,
        )
    changed_bindings = dict(ARTIFACT_BINDINGS)
    changed_bindings["raw_air_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="differs on replay"):
        domain_attestation.validate_supported_identity_domain(
            supported,
            attestation,
            provider_clear_threshold=1.0e-2,
            artifact_bindings=changed_bindings,
            constant_manifest=constants,
            evalmod_scalar_manifest=scalar_manifest,
            raw_air=raw_air,
        )


def test_normalization_and_topology_are_mandatory_emitted_authorities() -> None:
    _, _, config, transform, constants, raw_air = current_profile()
    arguments = {
        "coefficients": config.chebyshev_coefficients,
        "scalars": config.double_angle_scalars,
        "overflow_bound": config.eval_sin_upper_bound_k,
        "restoration_factor": config.post_scale,
        "evalmod_lower": -1.0,
        "evalmod_upper": 1.0,
        "provider_clear_threshold": 1.0e-2,
        "artifact_bindings": ARTIFACT_BINDINGS,
        "polynomial_degree": config.poly_degree,
        "logical_slots": config.slots,
        "constant_manifest": constants,
        "evalmod_scalar_manifest": build_bootstrap_evalmod_scalar_manifest(config),
        "raw_air": raw_air,
    }
    changed_transform = deepcopy(transform)
    changed_factor = math.nextafter(
        changed_transform["normalization"][
            "configured_coefficients_to_slots_factor"
        ],
        math.inf,
    )
    changed_transform["normalization"][
        "configured_coefficients_to_slots_factor"
    ] = changed_factor
    changed_transform["normalization"][
        "configured_coefficients_to_slots_factor_hex"
    ] = changed_factor.hex()
    with pytest.raises(ValueError, match="normalization differs"):
        domain_attestation.derive_supported_identity_domain(
            transform_payload_manifest=changed_transform, **arguments
        )
    with pytest.raises(ValueError, match="dual-EvalMod routing"):
        domain_attestation.derive_supported_identity_domain(
            transform_payload_manifest=transform,
            **{
                **arguments,
                "raw_air": raw_air.replace(
                    f"#{3 * config.slots:#x}",
                    f"#{3 * config.slots - 1:#x}",
                    1,
                ),
            },
        )
    changed_constants = deepcopy(constants)
    changed_constants["constants"][0]["payload_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="constant manifest differs"):
        domain_attestation.derive_supported_identity_domain(
            transform_payload_manifest=transform,
            **{**arguments, "constant_manifest": changed_constants},
        )
    with pytest.raises(ValueError, match="directly consumed"):
        domain_attestation.derive_supported_identity_domain(
            transform_payload_manifest=transform,
            **{
                **arguments,
                "raw_air": raw_air.replace(
                    "CKKS.mul ATTR[skip_auto_rescale=1]",
                    "CKKS.add",
                    1,
                ),
            },
        )

    operand_symbols = re.findall(
        r'^\s*ld\s+"(_rot_[^"]+)"\s+', raw_air, flags=re.MULTILINE
    )
    assert len(operand_symbols) >= 2
    first, second = operand_symbols[:2]
    swapped_air = raw_air.replace(f'ld "{first}"', 'ld "_swap_tmp"', 1)
    swapped_air = swapped_air.replace(f'ld "{second}"', f'ld "{first}"', 1)
    swapped_air = swapped_air.replace('ld "_swap_tmp"', f'ld "{second}"', 1)
    with pytest.raises(ValueError, match="wrong rotated operand"):
        domain_attestation.derive_supported_identity_domain(
            transform_payload_manifest=transform,
            **{**arguments, "raw_air": swapped_air},
        )


def test_budget_exhaustion_and_nonfinite_constants_fail_closed() -> None:
    _, _, config, transform, constants, raw_air = current_profile()
    common = {
        "scalars": [],
        "overflow_bound": config.eval_sin_upper_bound_k,
        "restoration_factor": config.post_scale,
        "evalmod_lower": -1.0,
        "evalmod_upper": 1.0,
        "provider_clear_threshold": 1.0e-2,
        "artifact_bindings": ARTIFACT_BINDINGS,
        "polynomial_degree": config.poly_degree,
        "logical_slots": config.slots,
        "transform_payload_manifest": transform,
        "constant_manifest": constants,
        "evalmod_scalar_manifest": build_bootstrap_evalmod_scalar_manifest(config),
        "raw_air": raw_air,
    }
    normalization = domain_attestation._attest_normalization(
        polynomial_degree=config.poly_degree,
        logical_slots=config.slots,
        overflow_bound=config.eval_sin_upper_bound_k,
        restoration_factor=config.post_scale,
        transform_payload_manifest=transform,
        constant_manifest=constants,
        raw_air=raw_air,
    )
    with pytest.raises(ValueError, match="exhausted at the origin"):
        domain_attestation._derive_polynomial_proof(
            ((2, 1),),
            config.eval_sin_upper_bound_k,
            config.post_scale.hex(),
            normalization["nominal_component_input_gain_hex"],
            (1.0e-2 * domain_attestation.CLEAR_MAP_BUDGET_FRACTION).hex(),
        )
    with pytest.raises(ValueError, match="finite and nonempty"):
        domain_attestation.derive_supported_identity_domain(
            coefficients=[math.nan], **common
        )


def test_plaintext_prime_count_uses_explicit_prime_bits_not_q_count() -> None:
    config = build_bootstrap_trace_config(
        poly_degree=32,
        mul_level=3,
        first_prime_bits=30,
        scaling_factor_bits=20,
        hamming_weight=4,
        q_parts=1,
        enc_budget=1,
        dec_budget=1,
        ct_encode=False,
    )
    assert config.num_p == 2
    assert math.ceil(config.mul_level / config.q_parts) == 3
    transform, constants, raw_air = transform_authorities(config)
    domain_attestation._attest_normalization(
        polynomial_degree=config.poly_degree,
        logical_slots=config.slots,
        overflow_bound=config.eval_sin_upper_bound_k,
        restoration_factor=config.post_scale,
        transform_payload_manifest=transform,
        constant_manifest=constants,
        raw_air=raw_air,
    )
    with pytest.raises(ValueError, match="direct encode metadata"):
        domain_attestation._attest_normalization(
            polynomial_degree=config.poly_degree,
            logical_slots=config.slots,
            overflow_bound=config.eval_sin_upper_bound_k,
            restoration_factor=config.post_scale,
            transform_payload_manifest=transform,
            constant_manifest=constants,
            raw_air=raw_air.replace("num_p=2", "num_p=3", 1),
        )


def test_transform_dataflow_is_closed_from_primary_input_through_return() -> None:
    config = build_bootstrap_trace_config(
        poly_degree=32,
        mul_level=3,
        first_prime_bits=30,
        scaling_factor_bits=20,
        hamming_weight=4,
        q_parts=1,
        enc_budget=1,
        dec_budget=1,
        ct_encode=False,
    )
    transform, constants, raw_air = transform_authorities(config)
    arguments = {
        "polynomial_degree": config.poly_degree,
        "logical_slots": config.slots,
        "overflow_bound": config.eval_sin_upper_bound_k,
        "restoration_factor": config.post_scale,
        "transform_payload_manifest": transform,
        "constant_manifest": constants,
    }

    normalization = domain_attestation._attest_normalization(
        raw_air=raw_air, **arguments
    )
    stages = normalization["transform_stage_dataflow"]
    assert stages[0]["primary_input"] == "p0"
    assert stages[-1]["terminal_restoration"] == {
        "rescale": re.search(
            r'ld "([^"]+)"[^\n]*\n\s*ld "\1"[^\n]*\n'
            r"\s*CKKS\.add[^\n]*\n\s*st \"_restored_0\"",
            raw_air,
        ).group(1),
        "self_add_count": config.post_scale_degree,
        "self_add_chain": [
            f"_restored_{index}" for index in range(config.post_scale_degree)
        ],
        "return_source": f"_restored_{config.post_scale_degree - 1}",
    }

    first_transform_add = raw_air.index("CKKS.add RTYPE[1](CIPHERTEXT)")
    assert first_transform_add < raw_air.index("CKKS.conjugate")
    with pytest.raises(ValueError, match="add-only constant fold"):
        domain_attestation._attest_normalization(
            raw_air=(
                raw_air[:first_transform_add]
                + raw_air[first_transform_add:].replace(
                    "CKKS.add", "CKKS.sub", 1
                )
            ),
            **arguments,
        )

    giant_match = re.search(r"CKKS\.rotate ATTR\[nums=([1-9][0-9]*)\]", raw_air)
    assert giant_match is not None
    wrong_giant = (int(giant_match.group(1)) + 1) % config.slots
    if wrong_giant == 0:
        wrong_giant = 1
    with pytest.raises(ValueError, match="wrong giant rotation"):
        domain_attestation._attest_normalization(
            raw_air=(
                raw_air[: giant_match.start(1)]
                + str(wrong_giant)
                + raw_air[giant_match.end(1) :]
            ),
            **arguments,
        )

    with pytest.raises(ValueError, match="raised primary input"):
        domain_attestation._attest_normalization(
            raw_air=raw_air.replace(
                'ld "p0" FML[1]', 'ld "p1" FML[2]', 1
            ),
            **arguments,
        )

    conjugate_input = re.search(
        r'ld "([^"]+)"[^\n]*\n\s*CKKS\.conjugate', raw_air
    )
    assert conjugate_input is not None
    with pytest.raises(ValueError, match="encoding-to-EvalMod boundary"):
        domain_attestation._attest_normalization(
            raw_air=(
                raw_air[: conjugate_input.start(1)]
                + "p1"
                + raw_air[conjugate_input.end(1) :]
            ),
            **arguments,
        )

    with pytest.raises(ValueError, match="EvalMod-to-decoding boundary"):
        domain_attestation._attest_normalization(
            raw_air=raw_air.replace(
                'ld "_recombined" VAR[1] RTYPE[1](CIPHERTEXT)',
                'ld "_split_real" VAR[1] RTYPE[1](CIPHERTEXT)',
                1,
            ),
            **arguments,
        )

    final_return = f"_restored_{config.post_scale_degree - 1}"
    prior_return = f"_restored_{config.post_scale_degree - 2}"
    return_load = f'        ld "{final_return}" VAR[1] RTYPE[1](CIPHERTEXT)\n'
    with pytest.raises(ValueError, match="invalid restoration chain"):
        domain_attestation._attest_normalization(
            raw_air=raw_air.replace(
                return_load,
                f'        ld "{prior_return}" VAR[1] RTYPE[1](CIPHERTEXT)\n',
                1,
            ),
            **arguments,
        )

    after_end_bypass = raw_air.replace(
        return_load,
        '        ld "_after_end" VAR[1] RTYPE[1](CIPHERTEXT)\n',
        1,
    )
    after_end_bypass += f'''\
          ld "{final_return}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "{final_return}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
      st "_after_end" VAR[1] ID(1)
'''
    with pytest.raises(ValueError, match="invalid restoration chain"):
        domain_attestation._attest_normalization(
            raw_air=after_end_bypass, **arguments
        )

    duplicate_inside = raw_air.replace(
        "    end_block ID(1)",
        '''\
          ld "p0" FML[1] RTYPE[1](CIPHERTEXT)
        CKKS.raise_mod RTYPE[1](CIPHERTEXT)
      st "_raised_primary" VAR[1] ID(1)
    end_block ID(1)''',
        1,
    )
    with pytest.raises(ValueError, match="duplicates a bootstrap_full store"):
        domain_attestation._attest_normalization(
            raw_air=duplicate_inside, **arguments
        )

    scalar_manifest = build_bootstrap_evalmod_scalar_manifest(config)
    split_start = raw_air.index('st "_conjugate"')
    split_input_start = raw_air.index(
        f'ld "{conjugate_input.group(1)}"', split_start
    )
    split_input_end = split_input_start + len(
        f'ld "{conjugate_input.group(1)}"'
    )
    with pytest.raises(ValueError, match="split dataflow"):
        domain_attestation.validate_evalmod_air_polynomial(
            air=(
                raw_air[:split_input_start]
                + 'ld "p1"'
                + raw_air[split_input_end:]
            ),
            air_label="mutated AIR",
            scalar_manifest=scalar_manifest,
        )

    scalar_literals = list(
        re.finditer(r"\bldc\s+#([^\s]+)\s+RTYPE", raw_air)
    )
    assert len(scalar_literals) == scalar_manifest["full_program"]["count"]
    first_literal = scalar_literals[0]
    with pytest.raises(ValueError, match="scalar literals differ"):
        domain_attestation.validate_evalmod_air_polynomial(
            air=(
                raw_air[: first_literal.start(1)]
                + "0"
                + raw_air[first_literal.end(1) :]
            ),
            air_label="mutated AIR",
            scalar_manifest=scalar_manifest,
        )

    distinct_literal = next(
        value
        for value in scalar_literals[1:]
        if value.group(1) != first_literal.group(1)
    )
    swapped = raw_air
    replacements = (
        (first_literal.start(1), first_literal.end(1), distinct_literal.group(1)),
        (
            distinct_literal.start(1),
            distinct_literal.end(1),
            first_literal.group(1),
        ),
    )
    for start, end, value in sorted(replacements, reverse=True):
        swapped = swapped[:start] + value + swapped[end:]
    with pytest.raises(ValueError, match="scalar literals differ"):
        domain_attestation.validate_evalmod_air_polynomial(
            air=swapped,
            air_label="swapped AIR",
            scalar_manifest=scalar_manifest,
        )
