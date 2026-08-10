"""Synthetic emitted authorities for clear-domain attestation tests."""

from __future__ import annotations

import math
from typing import Any

from ace_edsl.edsl.core.bootstrap_decomposition import (
    _collapsed_fft_stage_plan,
    _compact_grouped_stage_terms,
    _reduce_rotation,
    _rotate_plain_vector,
    build_bootstrap_transform_payload_manifest,
    eval_approx_mod,
)


def transform_authorities(config: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Materialize descriptors and raw AIR matching a compiler transform plan."""
    transform = build_bootstrap_transform_payload_manifest(config)
    roles: list[dict[str, Any]] = []
    for direction in ("coefficients_to_slots", "slots_to_coefficients"):
        for stage in transform[direction]["stages"]:
            for role in stage["payload_roles"]:
                roles.append(
                    {
                        "direction": direction,
                        "plaintext_level": stage["plaintext_level"],
                        **role,
                    }
                )

    raw_scale = math.ldexp(1.0, config.scaling_factor_bits).hex()
    constants = []
    for entry_id, role in enumerate(roles):
        constant_id = 60 + entry_id
        constants.append(
            {
                "entry_id": entry_id,
                "constant_id": constant_id,
                "symbol": f"_cst_{constant_id}",
                "payload_sha256": role["payload_sha256"],
                "ace_level": role["plaintext_level"],
                "chain_index": config.mul_level - role["plaintext_level"] + 1,
                "scale_degree": 1,
                "raw_scale": raw_scale,
                "element_type": "complex_f64",
                "slot_count": config.slots,
            }
        )
    constant_manifest = {"schema_version": 1, "constants": constants}

    value_index = 0

    def new_value(tag: str) -> str:
        nonlocal value_index
        value = f"_{tag}_{value_index}"
        value_index += 1
        return value

    def store(expression: str, symbol: str) -> str:
        return expression + f'      st "{symbol}" VAR[1] ID(1)\n'

    def constant_use(
        entry: dict[str, Any], cipher_symbol: str, output_symbol: str
    ) -> str:
        return f'''\
          ld "{cipher_symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
            ldc CST[{entry["constant_id"]:#x}] RTYPE[1](const_arr)
            intconst #{config.slots:#x} RTYPE[1](uint32_t)
            intconst #0x1 RTYPE[1](uint32_t)
            intconst #{entry["ace_level"]:#x} RTYPE[1](uint32_t)
          CKKS.encode ATTR[num_p={config.num_p},encode_dcmplx=1,encode_cache=1,level={entry["ace_level"]},scale=1] RTYPE[1](PLAINTEXT)
        CKKS.mul ATTR[skip_auto_rescale=1] RTYPE[1](CIPHERTEXT)
      st "{output_symbol}" VAR[1] ID(1)
'''

    entry_index = 0

    def transform_air(direction: str, initial_input: str) -> tuple[str, str]:
        nonlocal entry_index
        result = []
        stage_input = initial_input
        for stage in transform[direction]["stages"]:
            roles = stage["payload_roles"]
            rotations = {}
            for role in roles:
                rotations.setdefault(
                    role["rotation_batch_index"], role["input_rotation"]
                )
            ordered_rotations = [rotations[index] for index in range(len(rotations))]
            tag = f"{direction}_{stage['stage_order']}"
            batch_symbol = f"_batch_{tag}"
            result.append(
                f'''\
          ld "{stage_input}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.rotate_batch ATTR[nums=({",".join(str(value) for value in ordered_rotations)})] RTYPE[1](cipher_batch)
      st "{batch_symbol}" VAR[1] ID(1)
'''
            )
            for rotation_index in range(len(ordered_rotations)):
                result.append(
                    f'''\
            lda "{batch_symbol}" VAR[1] RTYPE[1](cipher_batch)
            intconst #{rotation_index:#x} RTYPE[1](int64_t)
          array RTYPE[1](cipher_batch)
        ild RTYPE[1](CIPHERTEXT)
      st "_rot_{tag}_{rotation_index}" VAR[1] ID(1)
'''
                )
            baby_indexes = sorted({role["baby_step_index"] for role in roles})
            stage_accumulator = None
            for baby_index in baby_indexes:
                baby_roles = [
                    role
                    for role in roles
                    if role["baby_step_index"] == baby_index
                ]
                group_accumulator = None
                for role in baby_roles:
                    entry = constants[entry_index]
                    entry_index += 1
                    term = new_value(f"term_{tag}_{baby_index}")
                    result.append(
                        constant_use(
                            entry,
                            f"_rot_{tag}_{role['rotation_batch_index']}",
                            term,
                        )
                    )
                    if group_accumulator is None:
                        group_accumulator = term
                    else:
                        combined = new_value(f"group_{tag}_{baby_index}")
                        result.append(
                            store(
                                f'''\
          ld "{group_accumulator}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "{term}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
''',
                                combined,
                            )
                        )
                        group_accumulator = combined
                assert group_accumulator is not None
                giant = baby_roles[0]["giant_rotation"] % config.slots
                contribution = group_accumulator
                if giant:
                    rotated = new_value(f"giant_{tag}_{baby_index}")
                    result.append(
                        store(
                            f'''\
          ld "{group_accumulator}" VAR[1] RTYPE[1](CIPHERTEXT)
          intconst #{giant:#x} RTYPE[1](int32_t)
        CKKS.rotate ATTR[nums={giant}] RTYPE[1](CIPHERTEXT)
''',
                            rotated,
                        )
                    )
                    contribution = rotated
                if stage_accumulator is None:
                    stage_accumulator = contribution
                else:
                    accumulated = new_value(f"stage_{tag}_{baby_index}")
                    result.append(
                        store(
                            f'''\
          ld "{stage_accumulator}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "{contribution}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
''',
                            accumulated,
                        )
                    )
                    stage_accumulator = accumulated
            assert stage_accumulator is not None
            stage_output = new_value(f"rescale_{tag}")
            result.append(
                store(
                    f'''\
          ld "{stage_accumulator}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.rescale RTYPE[1](CIPHERTEXT)
''',
                    stage_output,
                )
            )
            stage_input = stage_output
        return "".join(result), stage_input

    raised_input = "_raised_primary"
    encoding, encoding_output = transform_air(
        "coefficients_to_slots", raised_input
    )
    evalmod_split = "".join(
        (
            store(
                f'''\
          ld "{encoding_output}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.conjugate RTYPE[1](CIPHERTEXT)
''',
                "_conjugate",
            ),
            store(
                f'''\
          ld "{encoding_output}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "_conjugate" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
''',
                "_split_real",
            ),
            store(
                f'''\
          ld "{encoding_output}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "_conjugate" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.sub RTYPE[1](CIPHERTEXT)
''',
                "_split_imaginary",
            ),
            store(
                f'''\
          ld "_split_imaginary" VAR[1] RTYPE[1](CIPHERTEXT)
          intconst #{3 * config.slots:#x} RTYPE[1](int64_t)
        CKKS.mul_mono RTYPE[1](CIPHERTEXT)
''',
                "_imaginary_routed",
            ),
        )
    )

    evalmod_operations: list[str] = []

    class EvalValue:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def _binary(self, operation: str, other: Any) -> "EvalValue":
            output = new_value("evalmod")
            if isinstance(other, EvalValue):
                expression = f'''\
          ld "{self.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "{other.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.{operation} RTYPE[1](CIPHERTEXT)
'''
            else:
                scalar = float(other)
                scalar_literal = format(scalar, ".6g")
                expression = f'''\
          ld "{self.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
            ldc #{scalar_literal} RTYPE[1](float64_t)
            intconst #0x1 RTYPE[1](uint32_t)
            intconst #0x1 RTYPE[1](uint32_t)
            intconst #0 RTYPE[1](uint32_t)
          CKKS.encode ATTR[scale=1,mask=0] RTYPE[1](PLAINTEXT)
        CKKS.{operation} RTYPE[1](CIPHERTEXT)
'''
            evalmod_operations.append(store(expression, output))
            return EvalValue(output)

        def __add__(self, other: Any) -> "EvalValue":
            return self._binary("add", other)

        def __sub__(self, other: Any) -> "EvalValue":
            return self._binary("sub", other)

        def __mul__(self, other: Any) -> "EvalValue":
            return self._binary("mul", other)

        def mod_switch(self) -> "EvalValue":
            output = new_value("evalmod_modswitch")
            evalmod_operations.append(
                store(
                    f'''\
          ld "{self.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.mod_switch RTYPE[1](CIPHERTEXT)
''',
                    output,
                )
            )
            return EvalValue(output)

    real_evalmod = eval_approx_mod(
        EvalValue("_split_real"), config=config
    )
    imaginary_evalmod = eval_approx_mod(
        EvalValue("_imaginary_routed"), config=config
    )
    evalmod_recombine = "".join(
        (
            store(
                f'''\
          ld "{imaginary_evalmod.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
          intconst #{config.slots:#x} RTYPE[1](int64_t)
        CKKS.mul_mono RTYPE[1](CIPHERTEXT)
''',
                "_imaginary_returned",
            ),
            store(
                f'''\
          ld "{real_evalmod.symbol}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "_imaginary_returned" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
''',
                "_recombined",
            ),
        )
    )
    decoding, decoding_output = transform_air(
        "slots_to_coefficients", "_recombined"
    )
    restoration = []
    restored = decoding_output
    for index in range(config.post_scale_degree):
        destination = f"_restored_{index}"
        restoration.append(
            store(
                f'''\
          ld "{restored}" VAR[1] RTYPE[1](CIPHERTEXT)
          ld "{restored}" VAR[1] RTYPE[1](CIPHERTEXT)
        CKKS.add RTYPE[1](CIPHERTEXT)
''',
                destination,
            )
        )
        restored = destination
    raw_air = f'''\
  func_entry "bootstrap_full" ENT[1] ID(0)
    idname "p0" FML[1] RTYPE[1](CIPHERTEXT)
    idname "p1" FML[2] RTYPE[1](CIPHERTEXT)
    block ID(1)
          ld "p0" FML[1] RTYPE[1](CIPHERTEXT)
          intconst #{config.mul_level:#x} RTYPE[1](uint32_t)
        CKKS.raise_mod RTYPE[1](CIPHERTEXT)
      st "{raised_input}" VAR[1] ID(1)
{encoding}{evalmod_split}{''.join(evalmod_operations)}{evalmod_recombine}{decoding}{''.join(restoration)}        ld "{restored}" VAR[1] RTYPE[1](CIPHERTEXT)
      st "__ret_tmp_0" VAR[1] ID(1)
        ld "__ret_tmp_0" VAR[1] RTYPE[1](CIPHERTEXT)
      retv ID(1)
    end_block ID(1)
'''
    return transform, constant_manifest, raw_air


def transform_payload_values(config: Any) -> list[list[complex]]:
    """Return emitted transform payload values in constant-manifest order."""
    result: list[list[complex]] = []
    for encoding in (True, False):
        coefficients, stages = _collapsed_fft_stage_plan(config, encoding)
        for stage in stages:
            collapsed_stage = stage["s"]
            baby_step = stage["baby_step"]
            giant_step = stage["giant_step"]
            shift = stage["shift"]
            rotations = [
                _reduce_rotation(
                    (index - ((stage["num_rot"] + 1) // 2) + 1) * shift,
                    config.slots,
                )
                for index in range(giant_step)
            ]
            compact = _compact_grouped_stage_terms(
                coefficients, stage, config.slots, config
            )
            if compact is None:
                term_count = stage["num_rot"]
                stage_giant_step = giant_step
            else:
                term_count = len(compact)
                stage_giant_step = max(
                    1, (term_count + baby_step - 1) // baby_step
                )
                rotations = [
                    _reduce_rotation(index * shift, config.slots)
                    for index in range(min(stage_giant_step, term_count))
                ]
            for baby_index in range(baby_step):
                giant = stage_giant_step * baby_index
                diagonal_rotation = _reduce_rotation(
                    -(giant * shift), config.slots
                )
                for rotation_index in range(len(rotations)):
                    term = giant + rotation_index
                    if term >= term_count:
                        continue
                    diagonal = (
                        coefficients[collapsed_stage][term]
                        if compact is None
                        else compact[term][1]
                    )
                    if stage["diag_scale"] != 1.0:
                        diagonal = [
                            value * stage["diag_scale"] for value in diagonal
                        ]
                    if diagonal_rotation:
                        diagonal = _rotate_plain_vector(
                            diagonal, diagonal_rotation
                        )
                    result.append(list(diagonal))
    return result
