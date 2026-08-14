from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
TOOLS = REPOSITORY / "tools" / "phantom_gpu"
sys.path.insert(0, str(TOOLS))

import generate_bootstrap_qualification as generator  # noqa: E402


OPTION_VALUES = {
    "--poly-degree": "32",
    "--vector-capacity": "16",
    "--mul-level": "7",
    "--input-level": "1",
    "--security-level": "0",
    "--scaling-factor-bits": "41",
    "--first-prime-bits": "45",
    "--hamming-weight": "64",
    "--q-part-count": "2",
    "--encode-transform-budget": "2",
    "--decode-transform-budget": "3",
    "--ciphertext-constant-encoding": "disabled",
    "--packing": "full",
    "--post-multiply-real": "-1",
    "--post-multiply-imag": "0",
    "--post-multiply-scale-degree": "0",
    "--post-rotation-step": "3",
    "--identity-error-threshold": "0.01",
}


def command(output: str = "fresh-output") -> list[str]:
    values = [output]
    for name, value in OPTION_VALUES.items():
        values.extend((name, value))
    return values


def arguments() -> argparse.Namespace:
    value = generator.parse_arguments(command())
    generator.validate_arguments(value)
    return value


@pytest.mark.parametrize("missing", tuple(OPTION_VALUES))
def test_every_semantic_option_is_mandatory(missing: str) -> None:
    argv = command()
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        generator.parse_arguments(argv)


def test_normalized_invocation_is_typed_and_destination_independent() -> None:
    first = generator.build_invocation_record(arguments())
    second_arguments = generator.parse_arguments(command("another-destination"))
    generator.validate_arguments(second_arguments)
    second = generator.build_invocation_record(second_arguments)
    assert first == second
    assert first["schema_version"] == generator.INVOCATION_SCHEMA
    assert first["options"] == generator.typed_options(arguments())
    assert "fresh-output" not in json.dumps(first)
    assert "another-destination" not in json.dumps(first)
    assert "identity_error_threshold" not in first["options"]
    assert "--identity-error-threshold" not in first["normalized_argv"]
    assert first["options"]["clear_imag"] is False
    assert "--clear-imag" not in first["normalized_argv"]


def test_clear_imag_is_explicitly_identity_bound_when_enabled() -> None:
    value = generator.parse_arguments([*command(), "--clear-imag"])
    generator.validate_arguments(value)
    invocation = generator.build_invocation_record(value)
    assert invocation["options"]["clear_imag"] is True
    assert invocation["normalized_argv"][-1] == "--clear-imag"


def test_rotation_normalization_preserves_even_capacity_half_turn() -> None:
    assert generator.normalize_runtime_rotation_step(8, 16) == 8
    assert generator.normalize_runtime_rotation_step(-8, 16) == 8
    assert generator.normalize_runtime_rotation_step(9, 16) == -7


def test_ambient_control_is_rejected_even_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = generator.CONTROL_ENVIRONMENT[0]
    monkeypatch.setenv(name, "")
    with pytest.raises(SystemExit, match=name):
        generator.reject_ambient_controls()


def test_dynamic_context_validation_accepts_explicit_profile() -> None:
    value = arguments()
    context = {
        "packing": value.packing,
        "polynomial_degree": value.poly_degree,
        "logical_slot_capacity": value.vector_capacity,
        "input_level": value.input_level,
        "q_part_count": value.q_part_count,
        "hamming_weight": value.hamming_weight,
        "security_level": value.security_level,
        "first_modulus_bits": value.first_prime_bits,
        "scaling_modulus_bits": value.scaling_factor_bits,
        "data_q_bit_sizes": [value.first_prime_bits]
        + [value.scaling_factor_bits] * (value.mul_level - 1),
    }
    generator.validate_context(context, value)
    context["logical_slot_capacity"] -= 1
    with pytest.raises(SystemExit, match="logical_slot_capacity"):
        generator.validate_context(context, value)


def test_full_packing_rejects_capacity_override() -> None:
    value = generator.parse_arguments(command())
    value.vector_capacity -= 1
    with pytest.raises(SystemExit, match="half --poly-degree"):
        generator.validate_arguments(value)


def test_zero_post_operation_is_rejected() -> None:
    value = generator.parse_arguments(command())
    value.post_multiply_real = 0.0
    value.post_multiply_imag = 0.0
    with pytest.raises(SystemExit, match="multiplier must be nonzero"):
        generator.validate_arguments(value)
    value = generator.parse_arguments(command())
    value.post_rotation_step = value.vector_capacity
    with pytest.raises(SystemExit, match="nonzero modulo"):
        generator.validate_arguments(value)
    value = generator.parse_arguments(command())
    value.post_multiply_scale_degree = 1
    with pytest.raises(SystemExit, match="active-Q 1"):
        generator.validate_arguments(value)


def test_post_rotation_must_be_canonical_and_compiler_provisioned() -> None:
    value = generator.parse_arguments(command())
    value.post_rotation_step += value.vector_capacity
    with pytest.raises(SystemExit, match="canonical signed"):
        generator.validate_arguments(value)

    value = arguments()
    generator.validate_post_rotation_resource(
        {"rotation_steps": [-4, value.post_rotation_step, 4]},
        value,
    )
    with pytest.raises(SystemExit, match="absent from the compiler-emitted"):
        generator.validate_post_rotation_resource(
            {"rotation_steps": [-4, 4]},
            value,
        )


def test_terminal_restoration_is_derived_from_air_chain() -> None:
    attributes = "level=1,rescale_level=3,scale=1"
    air = f"""
      ld "_base" VAR[1] ATTR[{attributes}]
      ld "_base" VAR[1] ATTR[{attributes}]
    CKKS.add ATTR[{attributes}] RTYPE[1](CIPHERTEXT)
  st "_double" VAR[2] ATTR[{attributes}]
      ld "_double" VAR[2] ATTR[{attributes}]
      ld "_double" VAR[2] ATTR[{attributes}]
    CKKS.add ATTR[{attributes}] RTYPE[1](CIPHERTEXT)
  st "_quadruple" VAR[3] ATTR[{attributes}]
    ld "_quadruple" VAR[3] ATTR[{attributes}]
  st "__ret_tmp_0" VAR[4] ATTR[{attributes}]
    ld "__ret_tmp_0" VAR[4] ATTR[{attributes}]
  retv ID(5)
"""
    config = SimpleNamespace(post_scale_degree=2, post_scale=4.0)
    attestation = generator.attest_terminal_restoration(air, config)
    assert attestation["restoration"]["self_add_count"] == 2
    assert attestation["restoration"]["restored_factor"] == 4

    broken = air.replace('ld "_double" VAR[2]', 'ld "_other" VAR[2]', 1)
    with pytest.raises(SystemExit, match="self-add count"):
        generator.attest_terminal_restoration(broken, config)


def test_clear_imag_terminal_projection_and_restoration_are_attested() -> None:
    source_attributes = "level=5,rescale_level=3,scale=1"
    output_attributes = "level=1,rescale_level=3,scale=1"
    air = f"""
      ld "_pre_source" VAR[0] ATTR[{source_attributes}]
  st "_source" VAR[1] ATTR[{source_attributes}]
      ld "_source" VAR[1] ATTR[{source_attributes}]
    CKKS.conjugate ATTR[{source_attributes}] RTYPE[1](CIPHERTEXT)
  st "_conjugate" VAR[2] ATTR[{output_attributes}]
      ld "_source" VAR[1] ATTR[{output_attributes}]
      ld "_conjugate" VAR[2] ATTR[{output_attributes}]
    CKKS.add ATTR[{output_attributes}] RTYPE[1](CIPHERTEXT)
  st "_projection" VAR[3] ATTR[{output_attributes}]
      ld "_projection" VAR[3] ATTR[{output_attributes}]
      ld "_projection" VAR[3] ATTR[{output_attributes}]
    CKKS.add ATTR[{output_attributes}] RTYPE[1](CIPHERTEXT)
  st "_restored" VAR[4] ATTR[{output_attributes}]
    ld "_restored" VAR[4] ATTR[{output_attributes}]
  st "__ret_tmp_0" VAR[5] ATTR[{output_attributes}]
    ld "__ret_tmp_0" VAR[5] ATTR[{output_attributes}]
  retv ID(6)
"""
    config = SimpleNamespace(
        clear_imag=True,
        post_scale_degree=2,
        post_scale=4.0,
    )
    attestation = generator.attest_terminal_restoration(air, config)
    restoration = attestation["restoration"]
    assert restoration["self_add_count"] == 1
    assert restoration["restored_factor"] == 4
    assert restoration["real_projection"] == {
        "kind": "terminal-conjugate-real-projection",
        "source": "_source",
        "conjugate": "_conjugate",
        "destination": "_projection",
        "semantic_divisor": 2,
        "projected_component": "real",
        "caller_proof_required": True,
    }


def test_post_operation_air_transitions_are_checked() -> None:
    air = """
    CKKS.rotate ATTR[level=1,rescale_level=3,scale=1]
    CKKS.mul ATTR[level=1,rescale_level=3,scale=1]
"""
    value = generator.attest_post_operations(
        air,
        bootstrap_output={"level": 1, "rescale_level": 3, "scale": 1},
        data_q_count=7,
    )
    assert value["rotation"]["transition"]["scale_degree_delta"] == 0
    assert (
        value["ciphertext_plaintext_multiply"]["transition"][
            "scale_degree_delta"
        ]
        == 0
    )
    serialized = json.loads(json.dumps(value))
    assert serialized["input_coordinate"] == {
        "ace_logical_level": 5,
        "rescale_level": 3,
        "scale_degree": 1,
    }
    assert (
        serialized["ciphertext_plaintext_multiply"]["transition"][
            "raw_scale_multiplier"
        ]
        == 1.0
    )
    invalid = air.replace(
        "CKKS.mul ATTR[level=1,rescale_level=3,scale=1]",
        "CKKS.mul ATTR[level=0,rescale_level=3,scale=1]",
    )
    with pytest.raises(SystemExit, match="changed metadata"):
        generator.attest_post_operations(
            invalid,
            bootstrap_output={"level": 1, "rescale_level": 3, "scale": 1},
            data_q_count=7,
        )


def test_marked_raw_scale_one_post_operations_preserve_final_coordinate() -> None:
    output = {"level": 1, "rescale_level": 13, "scale": 1}
    air = generator.compile_post_operation_air(
        arguments(),
        bootstrap_output_attributes=output,
    )
    attestation = generator.attest_post_operations(
        air,
        bootstrap_output=output,
        data_q_count=26,
    )
    assert attestation["rotation"]["air_attributes"] == output
    assert attestation["ciphertext_plaintext_multiply"]["air_attributes"] == output


def test_runtime_cipher_coordinate_uses_physical_rescale_position() -> None:
    assert generator.runtime_cipher_coordinate(
        data_q_count=26, rescale_level=16
    ) == {
        "ace_logical_level": 11,
        "active_q_count": 11,
        "phantom_chain_index": 16,
    }
    with pytest.raises(SystemExit, match="outside the data-Q chain"):
        generator.runtime_cipher_coordinate(data_q_count=26, rescale_level=27)


def test_unmarked_zero_scale_encode_keeps_legacy_scale_degree_one() -> None:
    value = arguments()
    generator.AceEDSL._get_dsl.cache_clear()

    @generator.ckks_kernel
    def unmarked_zero_scale(ciphertext: generator.CkksCiphertext):
        constant = ciphertext.container.new_array_const([-1.0, 0.0])
        encoded = ciphertext.container.new_ckks_encode_complex(
            constant,
            1,
            0,
            1,
            2,
            True,
        )
        plain = generator.AIRValue(
            encoded,
            ciphertext.container,
            domain=getattr(ciphertext, "domain", None),
        )
        multiplied = ciphertext.container.new_ckks_mul(
            ciphertext.value,
            plain.value,
        )
        multiplied.set_u32_attr("skip_auto_rescale", 1)
        return generator.AIRValue(
            multiplied,
            ciphertext.container,
            domain=getattr(ciphertext, "domain", None),
        )

    unmarked_zero_scale(
        generator.CkksCiphertext(
            shape=(value.poly_degree,),
            name="bootstrap_output",
            level=2,
        )
    )
    module = generator.AceEDSL._get_dsl().current_air_module
    assert module is not None
    pipeline = generator.AcePipeline(module).configure_fhe(
        poly_degree=value.poly_degree,
        mul_level=value.mul_level,
        input_level=2,
        security_level=value.security_level,
        scaling_factor_bits=value.scaling_factor_bits,
        first_prime_bits=value.first_prime_bits,
        hamming_weight=value.hamming_weight,
        data_file="",
        provider="phantom",
        codegen_ir="ckks",
    )
    result = pipeline.run_ckks_driver()
    assert result["success"], result
    post_air = generator.canonicalize_checkout_paths(module.dump(), REPOSITORY)
    assert generator.operation_attributes(post_air, "mul") == [
        {"level": 2, "rescale_level": value.mul_level - 1, "scale": 2}
    ]
