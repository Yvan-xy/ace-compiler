from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"
sys.path.insert(0, str(TOOLS_ROOT))
sys.path.insert(0, str(TOOLS_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "ace_edsl/examples"))

from check_bootstrap_generated_artifacts import audit, cache_key_sha256  # noqa: E402
from bootstrap_domain_attestation import derive_supported_identity_domain  # noqa: E402
from bootstrap_domain_test_support import (  # noqa: E402
    transform_authorities,
    transform_payload_values,
)
from bootstrap_full import build_bootstrap_trace_config  # noqa: E402
from ace_edsl.edsl.core.bootstrap_decomposition import (  # noqa: E402
    build_bootstrap_evalmod_scalar_manifest,
)


CONTEXT: dict[str, Any] = {
    "schema_version": 1,
    "packing": "full",
    "polynomial_degree": 16384,
    "logical_slot_capacity": 8192,
    "data_q_bit_sizes": [60, 56, 56],
    "special_p_bit_sizes": [60],
    "input_level": 1,
    "q_part_count": 1,
    "hamming_weight": 192,
    "security_level": 0,
    "first_modulus_bits": 60,
    "scaling_modulus_bits": 56,
    "resource_schema_version": 3,
}

RESOURCE: dict[str, Any] = {
    "schema_version": 3,
    "context_schema_version": 1,
    "relinearization_key": True,
    "rotation_steps": [-3, 5],
    "conjugation_key": True,
    "rotate_batch": True,
    "rotation_batches": [[5, 0, 8189]],
    "raise_mod": True,
    "monomial_powers": [8192, 24576],
    "complex_plaintext": True,
    "native_bootstrap_precompute": False,
}


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


PAYLOADS = {
    "phantom_constant_17": (1.0, -0.0, 2.5, -3.25),
    "phantom_constant_29": (2.0, 0.5),
}


def payload_sha256(values: tuple[float, ...]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<d", value) for value in values)).hexdigest()


def constant_manifest(context_sha256: str) -> dict[str, Any]:
    constants: list[dict[str, Any]] = []
    for entry_id, constant_id, level, chain, scale, symbol in (
        (0, 17, 3, 1, "0x1.0000000000000p+40", "phantom_constant_17"),
        (1, 29, 2, 2, "0x1.0000000000000p+80", "phantom_constant_29"),
    ):
        payload = PAYLOADS[symbol]
        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "constant_id": constant_id,
            "symbol": symbol,
            "payload_sha256": payload_sha256(payload),
            "ace_level": level,
            "chain_index": chain,
            "scale_degree": entry_id + 1,
            "raw_scale": scale,
            "element_type": "complex_f64",
            "slot_count": len(payload) // 2,
            "cache_key_sha256": "",
        }
        entry["cache_key_sha256"] = cache_key_sha256(context_sha256, entry)
        constants.append(entry)
    return {
        "schema_version": 1,
        "context_schema_version": 1,
        "resource_schema_version": 3,
        "context_manifest_sha256": context_sha256,
        "constants": constants,
    }


SOURCE_TEMPLATE = r'''#include "rt_phantom/rt_phantom.h"
static const uint32_t phantom_data_q_bit_sizes[] = {60, 56, 56};
static const uint32_t phantom_special_p_bit_sizes[] = {60};
extern "C" const PHANTOM_CONTEXT_MANIFEST* Get_phantom_context_manifest() {
  static const PHANTOM_CONTEXT_MANIFEST context = {
    1, PHANTOM_PACKING_FULL, 16384, 8192, 3,
    phantom_data_q_bit_sizes, 1, phantom_special_p_bit_sizes,
    1, 1, 192, 0, 60, 56, 3
  };
  return &context;
}
static const int32_t phantom_rotation_steps[] = {-3, 5};
static const size_t phantom_rotation_batch_offsets[] = {0, 3};
static const int32_t phantom_rotation_batch_steps[] = {5, 0, 8189};
static const uint32_t phantom_monomial_powers[] = {8192, 24576};
extern "C" const PHANTOM_RESOURCE_MANIFEST* Get_phantom_resource_manifest() {
  static const PHANTOM_RESOURCE_MANIFEST resources = {
    3,
    1,
    PHANTOM_RESOURCE_RELIN_KEY | PHANTOM_RESOURCE_ROTATION_KEYS |
      PHANTOM_RESOURCE_CONJUGATION_KEY | PHANTOM_RESOURCE_ROTATE_BATCH |
      PHANTOM_RESOURCE_RAISE_MOD | PHANTOM_RESOURCE_MONOMIALS |
      PHANTOM_RESOURCE_COMPLEX_PLAINTEXT,
    2,
    phantom_rotation_steps,
    1,
    phantom_rotation_batch_offsets,
    phantom_rotation_batch_steps,
    2,
    phantom_monomial_powers
  };
  return &resources;
}
double phantom_constant_17[2][2] = {
  1, -0, 2.5, -3.25
};
double phantom_constant_29[1][2] = {
  2, 0.5
};
static const PHANTOM_CONSTANT_ENTRY phantom_constant_entries[] = {
  // ACE_PHANTOM_CONSTANT_ENTRY entry_id=0 constant_id=17
  {0, 17, PHANTOM_CONSTANT_COMPLEX_F64, 2, 3, 1, 1,
   0x1.0000000000000p+40, "phantom_constant_17", "__PAYLOAD_17__",
   "__CACHE_17__", 4, (const double*)phantom_constant_17},
  // ACE_PHANTOM_CONSTANT_ENTRY entry_id=1 constant_id=29
  {1, 29, PHANTOM_CONSTANT_COMPLEX_F64, 1, 2, 2, 2,
   0x1.0000000000000p+80, "phantom_constant_29", "__PAYLOAD_29__",
   "__CACHE_29__", 2, (const double*)phantom_constant_29},
};
extern "C" const PHANTOM_CONSTANT_MANIFEST* Get_phantom_constant_manifest() {
  static const PHANTOM_CONSTANT_MANIFEST constants = {
    1, 1, 3, "__CONTEXT_SHA256__", 2, phantom_constant_entries
  };
  return &constants;
}
void generated_program() {
  PLAIN first;
  PLAIN second;
  Load_cached_plain(&first, 0);
  Load_cached_plain(&second, 1);
  Conjugate_ciph(nullptr, nullptr);
  Rotate_batch_ciph(nullptr, nullptr, nullptr, 0);
  Raise_mod(nullptr, nullptr, 0);
  Mul_mono_ciph(nullptr, nullptr, 0);
}
'''

RAW_AIR = """\
ckks.conjugate
ckks.rotate_batch
ckks.raise_mod
ckks.mul_mono
"""
POST_CKKS_AIR = RAW_AIR


def render_source(constants: dict[str, Any]) -> str:
    by_id = {entry["constant_id"]: entry for entry in constants["constants"]}
    return (
        SOURCE_TEMPLATE.replace(
            "__CONTEXT_SHA256__", constants["context_manifest_sha256"]
        )
        .replace("__PAYLOAD_17__", by_id[17]["payload_sha256"])
        .replace("__CACHE_17__", by_id[17]["cache_key_sha256"])
        .replace("__PAYLOAD_29__", by_id[29]["payload_sha256"])
        .replace("__CACHE_29__", by_id[29]["cache_key_sha256"])
    )


DEFAULT_CONTEXT_SHA256 = hashlib.sha256(canonical_bytes(CONTEXT)).hexdigest()
SOURCE = render_source(constant_manifest(DEFAULT_CONTEXT_SHA256))


def write_inputs(
    tmp_path: Path,
    *,
    context: dict[str, Any] | None = None,
    resource: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
    raw_air: str | None = None,
    post_ckks_air: str | None = None,
    source: str | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    context_value = json.loads(json.dumps(context if context is not None else CONTEXT))
    context_path = tmp_path / "compiler_context_manifest.json"
    context_path.write_bytes(canonical_bytes(context_value))
    context_sha256 = hashlib.sha256(context_path.read_bytes()).hexdigest()
    constants_value = (
        json.loads(json.dumps(constants))
        if constants is not None
        else constant_manifest(context_sha256)
    )
    resource_value = json.loads(json.dumps(resource if resource is not None else RESOURCE))
    resource_path = tmp_path / "compiler_resource_manifest.json"
    constant_path = tmp_path / "compiler_constant_manifest.json"
    raw_air_path = tmp_path / "bootstrap_raw.air"
    post_ckks_air_path = tmp_path / "bootstrap_post_ckks.air"
    source_path = tmp_path / "generated.cu"
    resource_path.write_bytes(canonical_bytes(resource_value))
    constant_path.write_bytes(canonical_bytes(constants_value))
    raw_air_path.write_text(RAW_AIR if raw_air is None else raw_air, encoding="utf-8")
    post_ckks_air_path.write_text(
        POST_CKKS_AIR if post_ckks_air is None else post_ckks_air,
        encoding="utf-8",
    )
    source_path.write_text(
        render_source(constants_value) if source is None else source,
        encoding="utf-8",
    )
    return (
        context_path,
        resource_path,
        constant_path,
        raw_air_path,
        post_ckks_air_path,
        source_path,
    )


def run_audit(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    return audit(*write_inputs(tmp_path, **changes))


def artifact_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _versioned_terminal_source(
    context: dict[str, Any],
    resource: dict[str, Any],
    constants: dict[str, Any],
    payload_values: list[list[complex]],
    evalmod_values: list[float],
    evalmod_scalar_values: list[float],
    operation_trace: list[str],
    *,
    phantom: bool,
) -> str:
    scalar_definitions = "".join(
        f"float64_t _evalmod_{index} = {value!r};\n"
        for index, value in enumerate(evalmod_values)
    )
    payload_definitions = []
    for entry, values in zip(
        constants["constants"], payload_values, strict=True
    ):
        flattened = [component for value in values for component in (value.real, value.imag)]
        payload_definitions.append(
            f"float64_t {entry['symbol']}[{len(flattened)}] = {{\n  "
            + ", ".join(repr(float(value)) for value in flattened)
            + "\n};\n"
        )
    body = [
        "CIPHERTEXT bootstrap_full(CIPHERTEXT input, CIPHERTEXT zero) {\n",
        "  (void)zero;\n",
        "  static const int32_t body_rotation_steps[] = {",
        str(resource["rotation_batches"][0][0]),
        "};\n",
    ]
    current = "input"
    payload_index = 0
    scalar_index = 0
    for operation_index, operation in enumerate(operation_trace):
        output = f"trace_value_{operation_index}"
        body.append(f"  CIPHERTEXT {output};\n")
        if operation == "mul" and payload_index < len(constants["constants"]):
            entry = constants["constants"][payload_index]
            plain = f"payload_plain_{payload_index}"
            body.append(f"  PLAINTEXT {plain};\n")
            if phantom:
                body.append(
                    f"  Load_cached_plain(&{plain}, {entry['entry_id']});\n"
                )
                body.append(f"  Mul_plain(&{output}, &{current}, &{plain});\n")
            else:
                body.append(
                    f"  Encode_dcmplx_ext(&{plain}, (DCMPLX*){entry['symbol']}, "
                    f"{entry['slot_count']}, {entry['ace_level']}, "
                    f"{len(context['special_p_bit_sizes'])});\n"
                )
                body.append(
                    f"  Init_ciph_up_scale_plain(&{output}, &{current}, &{plain});\n"
                )
                body.append(
                    f"  Hw_modmul(Coeffs(&{output}._c0_poly, 0, degree), 0, 0, 0, degree);\n"
                )
            payload_index += 1
        elif operation == "mul" and scalar_index < len(evalmod_scalar_values):
            plain = f"scalar_plain_{scalar_index}"
            value = evalmod_scalar_values[scalar_index]
            body.extend(
                (
                    f"  PLAINTEXT {plain};\n",
                    f"  Encode_double_mask(&{plain}, {value!r}, 1, 1, 1);\n",
                )
            )
            if phantom:
                body.append(f"  Mul_plain(&{output}, &{current}, &{plain});\n")
            else:
                body.append(
                    f"  Init_ciph_up_scale_plain(&{output}, &{current}, &{plain});\n"
                )
                body.append(
                    f"  Hw_modmul(Coeffs(&{output}._c0_poly, 0, degree), 0, 0, 0, degree);\n"
                )
            scalar_index += 1
        elif operation == "mul":
            if phantom:
                body.append(f"  Mul_ciph(&{output}, &{current}, &{current});\n")
            else:
                product = f"trace_product_{operation_index}"
                aux0 = f"trace_aux0_{operation_index}"
                aux1 = f"trace_aux1_{operation_index}"
                body.append(
                    f"  Init_ciph3_up_scale(&{product}, &{current}, &{current});\n"
                )
                body.extend(
                    f"  Hw_mod{kind}(Coeffs(&{destination}, 0, degree), 0, 0, 0, degree);\n"
                    for kind, destination in (
                        ("mul", aux0),
                        ("mul", aux1),
                        ("mul", f"{product}._c0_poly"),
                        ("add", f"{product}._c1_poly"),
                        ("mul", f"{product}._c2_poly"),
                    )
                )
                body.append(f"  {output} = Relinearize({product});\n")
        elif operation in {"add", "sub"}:
            if phantom:
                runtime = "Add_ciph" if operation == "add" else "Sub_ciph"
                body.append(f"  {runtime}(&{output}, &{current}, &{current});\n")
            else:
                body.append(
                    f"  Init_ciph_same_scale(&{output}, &{current}, &{current});\n"
                )
                body.append(
                    f"  Hw_mod{operation}(Coeffs(&{output}._c0_poly, 0, degree), 0, 0, 0, degree);\n"
                )
        elif operation == "rescale":
            if phantom:
                body.append(f"  Rescale_ciph(&{output}, &{current});\n")
            else:
                body.append(f"  Init_ciph_down_scale(&{output}, &{current});\n")
                body.append(
                    f"  Rescale(&{output}._c0_poly, &{current}._c0_poly);\n"
                )
        elif operation == "modswitch":
            if phantom:
                body.append(f"  Mod_switch(&{output}, &{current});\n")
            else:
                body.append(
                    f"  Init_ciph_same_scale(&{output}, &{current}, &{current});\n"
                )
                body.append(
                    f"  Modswitch(&{output}._c0_poly, &{current}._c0_poly);\n"
                )
        elif operation == "rotate":
            if phantom:
                body.append(f"  Rotate_ciph(&{output}, &{current}, 1);\n")
            else:
                body.append(f"  {output} = Rotate({current}, 1);\n")
        elif operation == "rotate_batch":
            batch = f"trace_batch_{operation_index}"
            body.append(f"  CIPHERTEXT {batch}[1];\n")
            body.append(
                f"  Rotate_batch_ciph({batch}, &{current}, body_rotation_steps, 1);\n"
            )
            if phantom:
                body.append(f"  Copy_ciph(&{output}, &{batch}[0]);\n")
            else:
                body.append(f"  {output} = {batch}[0];\n")
        elif operation == "raise_mod":
            body.append(
                f"  Raise_mod(&{output}, &{current}, {len(context['data_q_bit_sizes'])});\n"
            )
        elif operation == "conjugate":
            body.append(f"  Conjugate_ciph(&{output}, &{current});\n")
        elif operation == "mul_mono":
            body.append(
                f"  Mul_mono_ciph(&{output}, &{current}, {context['logical_slot_capacity']});\n"
            )
        else:
            raise AssertionError(f"unsupported synthetic operation {operation}")
        current = output
    assert payload_index == len(constants["constants"])
    assert scalar_index == len(evalmod_scalar_values)
    body.extend(
        (
            "  CIPHERTEXT returned;\n",
            f"  Copy_ciph(&returned, &{current});\n",
            "  return returned;\n",
            "}\n",
        )
    )
    terminal_body = "".join(body)
    common = scalar_definitions + "".join(payload_definitions) + terminal_body
    if not phantom:
        return common

    data_q = ", ".join(str(value) for value in context["data_q_bit_sizes"])
    special_p = ", ".join(
        str(value) for value in context["special_p_bit_sizes"]
    )
    rotations = ", ".join(str(value) for value in resource["rotation_steps"])
    batch_steps = [step for batch in resource["rotation_batches"] for step in batch]
    batch_offsets = [0]
    for batch in resource["rotation_batches"]:
        batch_offsets.append(batch_offsets[-1] + len(batch))
    monomials = ", ".join(str(value) for value in resource["monomial_powers"])
    entries = []
    for entry in constants["constants"]:
        entries.append(
            f"  // ACE_PHANTOM_CONSTANT_ENTRY entry_id={entry['entry_id']} "
            f"constant_id={entry['constant_id']}\n"
            "  {"
            f"{entry['entry_id']}, {entry['constant_id']}, "
            f"PHANTOM_CONSTANT_COMPLEX_F64, {entry['slot_count']}, "
            f"{entry['ace_level']}, {entry['chain_index']}, "
            f"{entry['scale_degree']}, {entry['raw_scale']}, "
            f"{json.dumps(entry['symbol'])}, "
            f"{json.dumps(entry['payload_sha256'])}, "
            f"{json.dumps(entry['cache_key_sha256'])}, "
            f"{entry['slot_count'] * 2}, "
            f"(const double*){entry['symbol']}"
            "},\n"
        )
    return f'''#include "rt_phantom/rt_phantom.h"
static const uint32_t phantom_data_q_bit_sizes[] = {{{data_q}}};
static const uint32_t phantom_special_p_bit_sizes[] = {{{special_p}}};
extern "C" const PHANTOM_CONTEXT_MANIFEST* Get_phantom_context_manifest() {{
  static const PHANTOM_CONTEXT_MANIFEST context = {{
    1, PHANTOM_PACKING_FULL, {context["polynomial_degree"]},
    {context["logical_slot_capacity"]}, {len(context["data_q_bit_sizes"])},
    phantom_data_q_bit_sizes, {len(context["special_p_bit_sizes"])},
    phantom_special_p_bit_sizes, {context["input_level"]},
    {context["q_part_count"]}, {context["hamming_weight"]},
    {context["security_level"]}, {context["first_modulus_bits"]},
    {context["scaling_modulus_bits"]}, 3
  }};
  return &context;
}}
static const int32_t phantom_rotation_steps[] = {{{rotations}}};
static const size_t phantom_rotation_batch_offsets[] = {{{", ".join(str(value) for value in batch_offsets)}}};
static const int32_t phantom_rotation_batch_steps[] = {{{", ".join(str(value) for value in batch_steps)}}};
static const uint32_t phantom_monomial_powers[] = {{{monomials}}};
extern "C" const PHANTOM_RESOURCE_MANIFEST* Get_phantom_resource_manifest() {{
  static const PHANTOM_RESOURCE_MANIFEST resources = {{
    3, 1,
    PHANTOM_RESOURCE_RELIN_KEY | PHANTOM_RESOURCE_ROTATION_KEYS |
      PHANTOM_RESOURCE_CONJUGATION_KEY | PHANTOM_RESOURCE_ROTATE_BATCH |
      PHANTOM_RESOURCE_RAISE_MOD | PHANTOM_RESOURCE_MONOMIALS |
      PHANTOM_RESOURCE_COMPLEX_PLAINTEXT,
    {len(resource["rotation_steps"])}, phantom_rotation_steps,
    {len(resource["rotation_batches"])}, phantom_rotation_batch_offsets,
    phantom_rotation_batch_steps, {len(resource["monomial_powers"])},
    phantom_monomial_powers
  }};
  return &resources;
}}
{scalar_definitions}{"".join(payload_definitions)}
static const PHANTOM_CONSTANT_ENTRY phantom_constant_entries[] = {{
{"".join(entries)}}};
extern "C" const PHANTOM_CONSTANT_MANIFEST* Get_phantom_constant_manifest() {{
  static const PHANTOM_CONSTANT_MANIFEST constants = {{
    1, 1, 3, {json.dumps(constants["context_manifest_sha256"])},
    {len(constants["constants"])}, phantom_constant_entries
  }};
  return &constants;
}}
{terminal_body}
'''


def write_versioned_closure(tmp_path: Path) -> tuple[Path, ...]:
    config = build_bootstrap_trace_config(
        poly_degree=4,
        mul_level=26,
        first_prime_bits=60,
        scaling_factor_bits=56,
        hamming_weight=4,
        q_parts=3,
        enc_budget=1,
        dec_budget=1,
        ct_encode=False,
    )
    transform, descriptor_manifest, transform_air = transform_authorities(config)
    payload_values = transform_payload_values(config)
    expected_payloads = transform["constant_manifest_order"][
        "ordered_payload_sha256"
    ]
    assert [
        payload_sha256(
            tuple(
                component
                for value in values
                for component in (float(value.real), float(value.imag))
            )
        )
        for values in payload_values
    ] == expected_payloads

    context = {
        "schema_version": 1,
        "packing": "full",
        "polynomial_degree": config.poly_degree,
        "logical_slot_capacity": config.slots,
        "data_q_bit_sizes": [config.first_prime_bits]
        + [config.scaling_factor_bits] * (config.mul_level - 1),
        "special_p_bit_sizes": [60] * config.num_p,
        "input_level": 1,
        "q_part_count": config.q_parts,
        "hamming_weight": config.hamming_weight,
        "security_level": 0,
        "first_modulus_bits": config.first_prime_bits,
        "scaling_modulus_bits": config.scaling_factor_bits,
        "resource_schema_version": 3,
    }
    context_sha256 = hashlib.sha256(canonical_bytes(context)).hexdigest()
    constants = {
        "schema_version": 1,
        "context_schema_version": 1,
        "resource_schema_version": 3,
        "context_manifest_sha256": context_sha256,
        "constants": [],
    }
    for descriptor in descriptor_manifest["constants"]:
        entry = dict(descriptor)
        entry["cache_key_sha256"] = ""
        entry["cache_key_sha256"] = cache_key_sha256(context_sha256, entry)
        constants["constants"].append(entry)
    resource = {
        "schema_version": 3,
        "context_schema_version": 1,
        "relinearization_key": True,
        "rotation_steps": [1],
        "conjugation_key": True,
        "rotate_batch": True,
        "rotation_batches": [[1]],
        "raise_mod": True,
        "monomial_powers": [config.slots, 3 * config.slots],
        "complex_plaintext": True,
        "native_bootstrap_precompute": False,
    }
    evalmod_values = list(config.chebyshev_coefficients) + list(
        config.double_angle_scalars
    ) + [config.post_scale]
    evalmod_scalar_manifest = build_bootstrap_evalmod_scalar_manifest(config)
    evalmod_scalar_values = [
        float.fromhex(value)
        for value in evalmod_scalar_manifest["full_program"][
            "values_binary64_hex"
        ]
    ]
    raw_air = "CKKS.raise_mod\n" + transform_air
    operation_trace = re.findall(
        r"\bCKKS\.(add|sub|mul|rescale|rotate|conjugate|raise_mod|"
        r"rotate_batch|mul_mono|modswitch)\b",
        raw_air,
    )
    phantom_source = _versioned_terminal_source(
        context,
        resource,
        constants,
        payload_values,
        evalmod_values,
        evalmod_scalar_values,
        operation_trace,
        phantom=True,
    )
    ant_source = _versioned_terminal_source(
        context,
        resource,
        constants,
        payload_values,
        evalmod_values,
        evalmod_scalar_values,
        operation_trace,
        phantom=False,
    )
    post_air = raw_air + """
  st "__ret_tmp_0" VAR[4] ATTR[level=1,rescale_level=13,scale=1]
    ld "__ret_tmp_0" VAR[4] ATTR[level=1,rescale_level=13,scale=1]
  retv ID(5)
"""
    base = write_inputs(
        tmp_path,
        context=context,
        resource=resource,
        constants=constants,
        raw_air=raw_air,
        post_ckks_air=post_air,
        source=phantom_source,
    )
    (
        context_path,
        resource_path,
        constant_path,
        raw_air_path,
        post_air_path,
        source_path,
    ) = base
    ant_source_path = tmp_path / "generated_ant.cxx"
    ant_source_path.write_text(ant_source, encoding="utf-8")
    operations_air_path = tmp_path / "bootstrap_post_operations.air"
    operations_air_path.write_text(
        "CKKS.rotate ATTR[level=1,rescale_level=13,scale=1]\n"
        "CKKS.mul ATTR[level=1,rescale_level=13,scale=1]\n",
        encoding="utf-8",
    )
    options = {
        "poly_degree": config.poly_degree,
        "vector_capacity": config.slots,
        "mul_level": config.mul_level,
        "input_level": 1,
        "security_level": 0,
        "scaling_factor_bits": config.scaling_factor_bits,
        "first_prime_bits": config.first_prime_bits,
        "hamming_weight": config.hamming_weight,
        "q_part_count": config.q_parts,
        "packing": "full",
        "encode_transform_budget": config.enc_budget,
        "decode_transform_budget": config.dec_budget,
        "ciphertext_constant_encoding": "disabled",
        "post_multiply_real": -1.0,
        "post_multiply_imag": 0.0,
        "post_multiply_scale_degree": 0,
        "post_rotation_step": 1,
    }
    normalized = [
        "tools/phantom_gpu/generate_bootstrap_qualification.py",
        "--poly-degree", str(options["poly_degree"]),
        "--vector-capacity", str(options["vector_capacity"]),
        "--mul-level", str(options["mul_level"]),
        "--input-level", str(options["input_level"]),
        "--security-level", str(options["security_level"]),
        "--scaling-factor-bits", str(options["scaling_factor_bits"]),
        "--first-prime-bits", str(options["first_prime_bits"]),
        "--hamming-weight", str(options["hamming_weight"]),
        "--q-part-count", str(options["q_part_count"]),
        "--encode-transform-budget", str(options["encode_transform_budget"]),
        "--decode-transform-budget", str(options["decode_transform_budget"]),
        "--ciphertext-constant-encoding", options["ciphertext_constant_encoding"],
        "--packing", options["packing"],
        "--post-multiply-real", repr(options["post_multiply_real"]),
        "--post-multiply-imag", repr(options["post_multiply_imag"]),
        "--post-multiply-scale-degree", str(options["post_multiply_scale_degree"]),
        "--post-rotation-step", str(options["post_rotation_step"]),
    ]
    invocation_path = tmp_path / "compiler_invocation.json"
    invocation_path.write_text(
        json.dumps(
            {
                "schema_version": "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0",
                "status": "pass",
                "tool": normalized[0],
                "normalized_argv": normalized,
                "normalized_argv_sha256": hashlib.sha256(
                    json.dumps(
                        normalized, separators=(",", ":"), sort_keys=True
                    ).encode()
                ).hexdigest(),
                "options": options,
                "output_destination_in_identity": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    common_transition = {
        "ace_logical_level_delta": 0,
        "active_q_count_delta": 0,
        "phantom_chain_index_delta": 0,
        "scale_degree_delta": 0,
        "raw_scale_multiplier": 1.0,
        "logical_slots": "preserved",
        "ciphertext_size": "preserved",
        "ntt_state": "preserved",
    }
    attrs = {"level": 1, "rescale_level": 13, "scale": 1}
    attestation_path = tmp_path / "post_operations_attestation.json"
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": "ace.phantom.bootstrap-post-operation-semantics/1.0.0",
                "status": "attested",
                "bindings": {
                    "compiler_invocation_sha256": hashlib.sha256(invocation_path.read_bytes()).hexdigest(),
                    "post_ckks_air_sha256": hashlib.sha256(post_air_path.read_bytes()).hexdigest(),
                    "context_manifest_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                    "resource_manifest_sha256": hashlib.sha256(resource_path.read_bytes()).hexdigest(),
                    "post_operations_air_sha256": hashlib.sha256(operations_air_path.read_bytes()).hexdigest(),
                },
                "inputs": {
                    "multiply_constant": {
                        "real": -1.0,
                        "imaginary": 0.0,
                        "plaintext_scale_degree": 0,
                    },
                    "rotation_step": 1,
                },
                "input_coordinate": {
                    "ace_logical_level": 1,
                    "rescale_level": 13,
                    "scale_degree": 1,
                },
                "rotation": {"air_attributes": attrs, "transition": common_transition},
                "ciphertext_plaintext_multiply": {
                    "air_attributes": attrs,
                    "transition": common_transition,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    semantics_bindings = {
        "compiler_invocation_sha256": hashlib.sha256(invocation_path.read_bytes()).hexdigest(),
        "raw_air_sha256": hashlib.sha256(raw_air_path.read_bytes()).hexdigest(),
        "post_ckks_air_sha256": hashlib.sha256(post_air_path.read_bytes()).hexdigest(),
        "post_operations_air_sha256": hashlib.sha256(operations_air_path.read_bytes()).hexdigest(),
        "post_operations_attestation_sha256": hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        "context_manifest_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "resource_manifest_sha256": hashlib.sha256(resource_path.read_bytes()).hexdigest(),
        "constant_manifest_sha256": hashlib.sha256(constant_path.read_bytes()).hexdigest(),
        "phantom_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "generated_dsl_ant_source_sha256": hashlib.sha256(ant_source_path.read_bytes()).hexdigest(),
    }
    domain_bindings = {
        key: value
        for key, value in semantics_bindings.items()
        if key not in {
            "post_operations_air_sha256",
            "post_operations_attestation_sha256",
        }
    }
    supported_domain, domain_attestation = derive_supported_identity_domain(
        coefficients=config.chebyshev_coefficients,
        scalars=config.double_angle_scalars,
        overflow_bound=config.eval_sin_upper_bound_k,
        restoration_factor=config.post_scale,
        evalmod_lower=-1.0,
        evalmod_upper=1.0,
        provider_clear_threshold=1.0e-2,
        artifact_bindings=domain_bindings,
        polynomial_degree=config.poly_degree,
        logical_slots=config.slots,
        transform_payload_manifest=transform,
        constant_manifest=constants,
        evalmod_scalar_manifest=evalmod_scalar_manifest,
        raw_air=raw_air,
    )
    operation_contracts = {
        "status": "pass",
        "air_sha256": hashlib.sha256(operations_air_path.read_bytes()).hexdigest(),
        "input_coordinate": {
            "ace_logical_level": 1,
            "rescale_level": 13,
            "scale_degree": 1,
        },
        "rotation": {"air_attributes": attrs, "transition": common_transition},
        "ciphertext_plaintext_multiply": {
            "air_attributes": attrs,
            "transition": common_transition,
        },
    }
    semantics_path = tmp_path / "bootstrap_semantics.json"
    semantics_path.write_text(
        json.dumps(
            {
                "schema_version": "ace.phantom.generated-bootstrap.semantics/2.0.0",
                "status": "pass",
                "bindings": semantics_bindings,
                "expanded_bootstrap": {
                    "packing": "full",
                    "logical_slot_capacity": config.slots,
                    "transform_budgets": {
                        "encode": config.enc_budget,
                        "decode": config.dec_budget,
                    },
                    "ciphertext_constant_encoding": "disabled",
                    "coefficient_family": {
                        "coefficient_count": len(config.chebyshev_coefficients),
                        "coefficient_payload_sha256": hashlib.sha256(
                            b"".join(struct.pack("<d", value) for value in config.chebyshev_coefficients)
                        ).hexdigest(),
                        "evalmod_component_interval": {
                            "lower": -1.0,
                            "upper": 1.0,
                            "lower_inclusive": True,
                            "upper_inclusive": True,
                        },
                    },
                    "double_angle": {
                        "count": len(config.double_angle_scalars),
                        "scalar_payload_sha256": hashlib.sha256(
                            b"".join(struct.pack("<d", value) for value in config.double_angle_scalars)
                        ).hexdigest(),
                    },
                    "evalmod_scalar_encodings": evalmod_scalar_manifest,
                    "eval_sin_upper_bound_k": config.eval_sin_upper_bound_k,
                    "post_scale": {"factor": config.post_scale},
                    "coeffs_to_slots_factor": config.coeffs_to_slots_factor,
                },
                "output_air_contract": {
                    "ace_logical_level": 1,
                    "active_q_count": 1,
                    "rescale_level": 13,
                    "scale_degree": 1,
                    "raw_scale_contract": {
                        "kind": "ace-log2-scale-coordinate",
                        "nominal_raw_scale": math.ldexp(1.0, config.scaling_factor_bits).hex(),
                        "scaling_modulus_bits": config.scaling_factor_bits,
                        "expected_scale_degree": 1,
                        "maximum_absolute_coordinate_error": 1.0e-4,
                    },
                    "logical_slots": config.slots,
                },
                "post_operation_contracts": operation_contracts,
                "supported_identity_domain": supported_domain,
                "identity_domain_attestation": domain_attestation,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generation_path = tmp_path / "generation.json"
    post_hash = hashlib.sha256(post_air_path.read_bytes()).hexdigest()
    generation_path.write_text(
        json.dumps(
            {
                "schema_version": "ace.phantom.bootstrap-generation/4.0.0",
                "status": "pass",
                "identity_domain_policy": {
                    "provider_clear_maximum_absolute": 1.0e-2,
                    "clear_map_budget_fraction": 0.5,
                },
                "source": artifact_record(source_path),
                "sources": {
                    "phantom": artifact_record(source_path),
                    "generated_dsl_ant": artifact_record(ant_source_path),
                },
                "normalized_compiler_invocation": artifact_record(invocation_path),
                "bootstrap_semantics": artifact_record(semantics_path),
                "post_operations_air": artifact_record(operations_air_path),
                "post_operations_attestation": artifact_record(attestation_path),
                "air": {
                    "raw": artifact_record(raw_air_path),
                    "post_ckks": artifact_record(post_air_path),
                },
                "terminal_paths": {
                    "phantom": {
                        "provider": "phantom",
                        "codegen_ir": "ckks",
                        "stages_completed": ["ckks_driver", "ckks2c"],
                        "post_ckks_air_sha256": post_hash,
                    },
                    "generated_dsl_ant": {
                        "provider": "ant",
                        "codegen_ir": "poly",
                        "stages_completed": [
                            "ckks_driver", "poly_driver", "poly2c"
                        ],
                        "post_ckks_air_sha256": post_hash,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return base + (
        ant_source_path,
        generation_path,
        invocation_path,
        semantics_path,
        operations_air_path,
        attestation_path,
    )


def run_versioned_audit(tmp_path: Path) -> dict[str, Any]:
    return audit(*write_versioned_closure(tmp_path))


def rebind_terminal_sources(paths: tuple[Path, ...]) -> None:
    semantics = json.loads(paths[9].read_text(encoding="utf-8"))
    semantics["bindings"]["phantom_source_sha256"] = hashlib.sha256(
        paths[5].read_bytes()
    ).hexdigest()
    semantics["bindings"]["generated_dsl_ant_source_sha256"] = hashlib.sha256(
        paths[6].read_bytes()
    ).hexdigest()
    paths[9].write_text(json.dumps(semantics, sort_keys=True), encoding="utf-8")
    generation = json.loads(paths[7].read_text(encoding="utf-8"))
    generation["source"] = artifact_record(paths[5])
    generation["sources"] = {
        "phantom": artifact_record(paths[5]),
        "generated_dsl_ant": artifact_record(paths[6]),
    }
    generation["bootstrap_semantics"] = artifact_record(paths[9])
    paths[7].write_text(json.dumps(generation, sort_keys=True), encoding="utf-8")


def rebind_post_ckks_air(paths: tuple[Path, ...]) -> None:
    """Rebind a post-CKKS mutation without changing the raw domain authority."""
    post_hash = hashlib.sha256(paths[4].read_bytes()).hexdigest()
    post_attestation = json.loads(paths[11].read_text(encoding="utf-8"))
    post_attestation["bindings"]["post_ckks_air_sha256"] = post_hash
    paths[11].write_text(
        json.dumps(post_attestation, sort_keys=True), encoding="utf-8"
    )

    semantics = json.loads(paths[9].read_text(encoding="utf-8"))
    semantics["bindings"]["post_ckks_air_sha256"] = post_hash
    semantics["bindings"]["post_operations_attestation_sha256"] = (
        hashlib.sha256(paths[11].read_bytes()).hexdigest()
    )
    identity_attestation = semantics["identity_domain_attestation"]
    identity_attestation["artifact_bindings"]["post_ckks_air_sha256"] = post_hash
    semantics["supported_identity_domain"]["evidence"][
        "attestation_sha256"
    ] = hashlib.sha256(canonical_bytes(identity_attestation)).hexdigest()
    paths[9].write_text(json.dumps(semantics, sort_keys=True), encoding="utf-8")

    generation = json.loads(paths[7].read_text(encoding="utf-8"))
    generation["air"]["post_ckks"] = artifact_record(paths[4])
    for terminal in generation["terminal_paths"].values():
        terminal["post_ckks_air_sha256"] = post_hash
    generation["bootstrap_semantics"] = artifact_record(paths[9])
    generation["post_operations_attestation"] = artifact_record(paths[11])
    paths[7].write_text(json.dumps(generation, sort_keys=True), encoding="utf-8")


def test_versioned_generated_artifact_closure_passes(tmp_path: Path) -> None:
    report = run_versioned_audit(tmp_path)
    assert report["status"] == "pass", report["errors"]
    assert report["schema_version"] == "ace.phantom.bootstrap-generated-artifact-audit/4.0.0"
    assert report["qualification_closure"]["post_operations_attested"] is True
    assert report["qualification_closure"]["identity_domain_attested"] is True
    assert (
        report["qualification_closure"][
            "post_ckks_evalmod_polynomial_attested"
        ]
        is True
    )
    body_closure = report["qualification_closure"]["terminal_body_closure"]
    assert body_closure["phantom"]["transform_constant_count"] == 6
    assert body_closure["generated_dsl_ant"]["transform_constant_count"] == 6
    assert body_closure["phantom"]["reachable_required_stage_count"] == 4
    assert body_closure["generated_dsl_ant"][
        "reachable_required_stage_count"
    ] == 4
    assert body_closure["phantom"]["scalar_encode_count"] == body_closure[
        "generated_dsl_ant"
    ]["scalar_encode_count"]
    assert body_closure["phantom"][
        "scalar_encode_payload_sha256"
    ] == body_closure["generated_dsl_ant"]["scalar_encode_payload_sha256"]


def test_versioned_dead_bootstrap_body_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[5].read_text(encoding="utf-8").replace(
        "CIPHERTEXT bootstrap_full(",
        "CIPHERTEXT dead_generated_program(",
        1,
    )
    source += (
        "\nCIPHERTEXT bootstrap_full(CIPHERTEXT input, CIPHERTEXT zero) {\n"
        "  (void)zero;\n"
        "  return input;\n"
        "}\n"
    )
    paths[5].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "does not return an owned result" in report["errors"][0]


def test_versioned_dead_ant_bootstrap_body_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[6].read_text(encoding="utf-8").replace(
        "CIPHERTEXT bootstrap_full(",
        "CIPHERTEXT dead_generated_program(",
        1,
    )
    source += (
        "\nCIPHERTEXT bootstrap_full(CIPHERTEXT input, CIPHERTEXT zero) {\n"
        "  (void)zero;\n"
        "  return input;\n"
        "}\n"
    )
    paths[6].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "does not return an owned result" in report["errors"][0]


def test_versioned_ant_terminal_payload_bypass_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source, replacement_count = re.subn(
        r"Copy_ciph\(&returned, &trace_value_[0-9]+\);",
        "Copy_ciph(&returned, &input);",
        paths[6].read_text(encoding="utf-8"),
        count=1,
    )
    assert replacement_count == 1
    paths[6].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert (
        "return is not copied from a generated value" in report["errors"][0]
        or "does not depend on every transform constant" in report["errors"][0]
    )


@pytest.mark.parametrize("source_index", (5, 6))
def test_versioned_post_return_payload_overwrite_is_rejected(
    tmp_path: Path,
    source_index: int,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source, copy_count = re.subn(
        r"Copy_ciph\(&returned, &trace_value_[0-9]+\);",
        "Copy_ciph(&returned, &input);",
        paths[source_index].read_text(encoding="utf-8"),
        count=1,
    )
    assert copy_count == 1
    source, replacement_count = re.subn(
        r"(\breturn\s+returned\s*;)(\s*\n\s*})",
        r"\1\n  Add_ciph(&returned, &input, &input);\2",
        source,
        count=1,
    )
    assert replacement_count == 1
    paths[source_index].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert (
        "return is not copied from a generated value" in report["errors"][0]
        or "does not depend on every transform constant" in report["errors"][0]
    )


def test_versioned_unreachable_terminal_payload_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source, replacement_count = re.subn(
        r"Mul_plain\(&(trace_value_[0-9]+), &(trace_value_[0-9]+), "
        r"&payload_plain_0\);",
        r"Mul_ciph(&\1, &\2, &\2);",
        paths[5].read_text(encoding="utf-8"),
        count=1,
    )
    assert replacement_count == 1
    paths[5].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "does not depend on every transform constant" in report["errors"][0]


def test_versioned_phantom_primary_input_bypass_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[5].read_text(encoding="utf-8").replace(
        "Raise_mod(&trace_value_0, &input,",
        "Raise_mod(&trace_value_0, &zero,",
        1,
    )
    paths[5].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "does not depend on its primary input" in report["errors"][0]


def test_versioned_terminal_scalar_tamper_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[5].read_text(encoding="utf-8")
    source, replacement_count = re.subn(
        r"^\s*Encode_double_mask\(&scalar_plain_0,[^\n]+\n",
        "",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    assert replacement_count == 1
    paths[5].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "differ in EvalMod scalar encodings" in report["errors"][0]


def test_versioned_matching_terminal_scalar_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    pattern = re.compile(
        r"(Encode_double_mask\(&scalar_plain_0,\s*)[^,]+(,\s*1,\s*1,\s*1\);)"
    )
    for source_index in (5, 6):
        source, replacement_count = pattern.subn(
            r"\g<1>0.25\g<2>",
            paths[source_index].read_text(encoding="utf-8"),
            count=1,
        )
        assert replacement_count == 1
        paths[source_index].write_text(source, encoding="utf-8")
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "attested polynomial" in report["errors"][0]


def test_versioned_post_ckks_evalmod_operator_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    post_air = paths[4].read_text(encoding="utf-8")
    conjugate = post_air.index("CKKS.conjugate")
    suffix, replacement_count = re.subn(
        r"\bCKKS\.mul(\s+ATTR\[[^\]]+\])?\s+RTYPE",
        r"CKKS.add\1 RTYPE",
        post_air[conjugate:],
        count=1,
    )
    assert replacement_count == 1
    paths[4].write_text(post_air[:conjugate] + suffix, encoding="utf-8")
    rebind_post_ckks_air(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert (
        "post-CKKS AIR EvalMod" in report["errors"][0]
        or "terminal operation trace differs" in report["errors"][0]
    )


@pytest.mark.parametrize(
    ("original", "replacement"),
    (("Mul_ciph(", "Add_ciph("), ("Sub_ciph(", "Add_ciph(")),
)
def test_versioned_phantom_terminal_operator_substitution_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[5].read_text(encoding="utf-8")
    assert original in source
    paths[5].write_text(
        source.replace(original, replacement, 1), encoding="utf-8"
    )
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "terminal operation trace differs" in report["errors"][0]


@pytest.mark.parametrize(
    ("original", "replacement"),
    (("Hw_modmul(", "Hw_modadd("), ("Hw_modsub(", "Hw_modadd(")),
)
def test_versioned_ant_terminal_operator_substitution_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    paths = write_versioned_closure(tmp_path)
    source = paths[6].read_text(encoding="utf-8")
    assert original in source
    paths[6].write_text(
        source.replace(original, replacement, 1), encoding="utf-8"
    )
    rebind_terminal_sources(paths)
    report = audit(*paths)
    assert report["status"] == "fail"
    assert (
        "lowering" in report["errors"][0]
        or "terminal operation trace differs" in report["errors"][0]
    )


def test_versioned_ant_abi_tamper_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    paths[6].write_text(
        paths[6].read_text(encoding="utf-8").replace(
            "CIPHERTEXT bootstrap_full(", "CIPHERTEXT wrong(", 1
        ),
        encoding="utf-8",
    )
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "exact bootstrap_full" in report["errors"][0]


def test_versioned_terminal_path_hash_tamper_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    generation = json.loads(paths[7].read_text())
    generation["terminal_paths"]["generated_dsl_ant"]["post_ckks_air_sha256"] = "0" * 64
    paths[7].write_text(json.dumps(generation), encoding="utf-8")
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "terminal path binding differs" in report["errors"][0]


def test_versioned_post_operation_air_tamper_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    paths[10].write_text(
        paths[10].read_text().replace("CKKS.mul ATTR[level=1", "CKKS.mul ATTR[level=2"),
        encoding="utf-8",
    )
    report = audit(*paths)
    assert report["status"] == "fail"


def test_versioned_invocation_option_set_tamper_is_rejected(tmp_path: Path) -> None:
    paths = write_versioned_closure(tmp_path)
    invocation = json.loads(paths[8].read_text())
    invocation["options"]["unexpected"] = 1
    paths[8].write_text(json.dumps(invocation), encoding="utf-8")
    invocation_hash = hashlib.sha256(paths[8].read_bytes()).hexdigest()
    attestation = json.loads(paths[11].read_text())
    attestation["bindings"]["compiler_invocation_sha256"] = invocation_hash
    paths[11].write_text(json.dumps(attestation), encoding="utf-8")
    semantics = json.loads(paths[9].read_text())
    semantics["bindings"]["compiler_invocation_sha256"] = invocation_hash
    semantics["bindings"]["post_operations_attestation_sha256"] = hashlib.sha256(
        paths[11].read_bytes()
    ).hexdigest()
    paths[9].write_text(json.dumps(semantics), encoding="utf-8")
    generation = json.loads(paths[7].read_text())
    generation["normalized_compiler_invocation"] = artifact_record(paths[8])
    generation["bootstrap_semantics"] = artifact_record(paths[9])
    generation["post_operations_attestation"] = artifact_record(paths[11])
    paths[7].write_text(json.dumps(generation), encoding="utf-8")
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "compiler invocation options keys differ" in report["errors"][0]


def test_versioned_post_operation_coordinate_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    paths = write_versioned_closure(tmp_path)
    attestation = json.loads(paths[11].read_text())
    attestation["input_coordinate"]["rescale_level"] = 12
    paths[11].write_text(json.dumps(attestation), encoding="utf-8")
    semantics = json.loads(paths[9].read_text())
    semantics["bindings"]["post_operations_attestation_sha256"] = hashlib.sha256(
        paths[11].read_bytes()
    ).hexdigest()
    semantics["post_operation_contracts"]["input_coordinate"][
        "rescale_level"
    ] = 12
    paths[9].write_text(json.dumps(semantics), encoding="utf-8")
    generation = json.loads(paths[7].read_text())
    generation["bootstrap_semantics"] = artifact_record(paths[9])
    generation["post_operations_attestation"] = artifact_record(paths[11])
    paths[7].write_text(json.dumps(generation), encoding="utf-8")
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "input coordinate differs" in report["errors"][0]


def test_complete_generated_artifact_closure_passes(tmp_path: Path) -> None:
    report = run_audit(tmp_path)
    assert report["status"] == "pass", report["errors"]
    assert (
        report["schema_version"]
        == "ace.phantom.bootstrap-generated-artifact-audit/2.0.0"
    )
    assert report["counts"] == {
        "constants": 2,
        "monomial_powers": 2,
        "rotation_batches": 1,
        "rotation_steps": 2,
    }
    assert report["forbidden_native_bts_matches"] == []
    assert report["forbidden_matches"] == {
        "raw_air": [],
        "post_ckks_air": [],
        "source": [],
    }
    assert report["air"]["raw"]["required_opcode_counts"] == {
        "ckks.conjugate": 1,
        "ckks.rotate_batch": 1,
        "ckks.raise_mod": 1,
        "ckks.mul_mono": 1,
    }
    assert report["air"]["post_ckks"]["required_opcode_counts"] == report[
        "air"
    ]["raw"]["required_opcode_counts"]
    assert report["source"]["required_call_counts"] == {
        "Conjugate_ciph": 1,
        "Rotate_batch_ciph": 1,
        "Raise_mod": 1,
        "Mul_mono_ciph": 1,
    }
    assert set(report["inputs"]) == {
        "constant_manifest",
        "context_manifest",
        "raw_air",
        "post_ckks_air",
        "resource_manifest",
        "source",
    }


def test_report_is_independent_of_absolute_input_root(tmp_path: Path) -> None:
    local_root = tmp_path / "local" / "state"
    remote_root = tmp_path / "remote" / "work"
    local_root.mkdir(parents=True)
    remote_root.mkdir(parents=True)

    local = run_audit(local_root)
    remote = run_audit(remote_root)

    assert local == remote
    assert local["inputs"]["source"]["path"] == "generated.cu"


def test_cli_writes_a_checksum_bound_failure_report(tmp_path: Path) -> None:
    paths = write_inputs(tmp_path, source=SOURCE + "\nBootstrapper forbidden;\n")
    report_path = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_ROOT / "check_bootstrap_generated_artifacts.py"),
            "--context-manifest",
            str(paths[0]),
            "--resource-manifest",
            str(paths[1]),
            "--constant-manifest",
            str(paths[2]),
            "--raw-air",
            str(paths[3]),
            "--post-ckks-air",
            str(paths[4]),
            "--source",
            str(paths[5]),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["status"] == "fail"
    assert report["forbidden_native_bts_matches"] == [
        "native Phantom bootstrap type"
    ]
    assert SHA256_KEYS <= set(report["inputs"]["source"])


SHA256_KEYS = {"path", "sha256", "size_bytes"}


@pytest.mark.parametrize("air_name", ("raw_air", "post_ckks_air"))
@pytest.mark.parametrize(
    "opcode",
    ("ckks.conjugate", "ckks.rotate_batch", "ckks.raise_mod", "ckks.mul_mono"),
)
def test_each_air_requires_every_retained_opcode(
    tmp_path: Path, air_name: str, opcode: str
) -> None:
    report = run_audit(tmp_path, **{air_name: RAW_AIR.replace(opcode + "\n", "")})
    assert report["status"] == "fail"
    assert f"{air_name.removesuffix('_air')} AIR is missing" in report["errors"][0]
    assert opcode in report["errors"][0]
    assert report["air"][air_name.removesuffix("_air")][
        "required_opcode_counts"
    ][opcode] == 0


@pytest.mark.parametrize("air_name", ("raw_air", "post_ckks_air"))
@pytest.mark.parametrize(
    ("forbidden_opcode", "diagnostic"),
    (
        ("ckks.bootstrap", "opaque CKKS bootstrap opcode"),
        ("ckks.bootstrap_coeffs_to_slots", "coefficient-to-slot stage opcode"),
        ("ckks.bootstrap_eval_mod", "evaluation stage opcode"),
        ("ckks.bootstrap_slots_to_coeffs", "slot-to-coefficient stage opcode"),
        ("poly.add", "POLY/HPOLY/LPOLY opcode"),
        ("hpoly.mul", "POLY/HPOLY/LPOLY opcode"),
        ("lpoly.rotate", "POLY/HPOLY/LPOLY opcode"),
    ),
)
def test_each_air_rejects_opaque_stage_and_lower_level_opcodes(
    tmp_path: Path, air_name: str, forbidden_opcode: str, diagnostic: str
) -> None:
    report = run_audit(
        tmp_path, **{air_name: RAW_AIR + forbidden_opcode + "\n"}
    )
    assert report["status"] == "fail"
    assert diagnostic in report["forbidden_matches"][air_name]


@pytest.mark.parametrize(
    "call", ("Conjugate_ciph", "Rotate_batch_ciph", "Raise_mod", "Mul_mono_ciph")
)
def test_generated_source_requires_each_retained_call(
    tmp_path: Path, call: str
) -> None:
    source = re.sub(
        r"^\s*" + re.escape(call) + r"\([^\n]*\);\n",
        "",
        SOURCE,
        flags=re.MULTILINE,
    )
    report = run_audit(tmp_path, source=source)
    assert report["status"] == "fail"
    assert call in report["errors"][0]
    assert report["source"]["required_call_counts"][call] == 0


@pytest.mark.parametrize(
    ("forbidden_call", "diagnostic"),
    (
        ("Bootstrap();", "opaque bootstrap call"),
        ("Eval_bootstrap_ciph();", "opaque ANT bootstrap call"),
        ("Eval_bootstrap_coeffs_to_slots_ciph();", "opaque ANT bootstrap call"),
        ("bootstrap_eval_mod();", "native Phantom evaluation stage"),
        ("bootstrap_slots_to_coeffs();", "native Phantom slot-to-coefficient stage"),
    ),
)
def test_generated_source_rejects_opaque_and_stage_calls(
    tmp_path: Path, forbidden_call: str, diagnostic: str
) -> None:
    report = run_audit(tmp_path, source=SOURCE + "\n" + forbidden_call + "\n")
    assert report["status"] == "fail"
    assert diagnostic in report["forbidden_matches"]["source"]


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    (
        (lambda value: value.__setitem__("native_bootstrap_precompute", True), "must be false"),
        (lambda value: value["rotation_steps"].append(32767), "conjugation sentinel"),
        (lambda value: value["monomial_powers"].append(32768), "canonical in [0, 2N)"),
    ),
)
def test_resource_contract_rejects_non_bootstrap_requirements(
    tmp_path: Path, mutation: Any, diagnostic: str
) -> None:
    resource = json.loads(json.dumps(RESOURCE))
    mutation(resource)
    report = run_audit(tmp_path, resource=resource)
    assert report["status"] == "fail"
    assert diagnostic in report["errors"][0]


def test_generated_resource_flags_are_bound_to_sidecar(tmp_path: Path) -> None:
    source = SOURCE.replace("PHANTOM_RESOURCE_COMPLEX_PLAINTEXT", "0")
    report = run_audit(tmp_path, source=source)
    assert report["status"] == "fail"
    assert "resource flags differ" in report["errors"][0]


def test_raw_batch_steps_must_close_through_canonical_signed_keys(
    tmp_path: Path,
) -> None:
    resource = json.loads(json.dumps(RESOURCE))
    resource["rotation_batches"][0][-1] = 8188
    report = run_audit(tmp_path, resource=resource)
    assert report["status"] == "fail"
    assert "not closed by rotation_steps" in report["errors"][0]


@pytest.mark.parametrize(
    ("source_change", "diagnostic"),
    (
        (
            lambda source: source.replace(
                "ACE_PHANTOM_CONSTANT_ENTRY entry_id=1 constant_id=29",
                "ACE_PHANTOM_CONSTANT_ENTRY entry_id=1 constant_id=30",
            ),
            "entry markers differ",
        ),
        (
            lambda source: source.replace(
                "(const double*)phantom_constant_29",
                "(const double*)unknown_constant",
            ),
            "13-field initializer differs",
        ),
        (
            lambda source: source.replace("Load_cached_plain(&second, 1);", ""),
            "missing=[1]",
        ),
        (
            lambda source: source + "\nLoad_cached_plain(&first, 99);\n",
            "unknown=[99]",
        ),
    ),
)
def test_constant_source_closure_is_exact(
    tmp_path: Path, source_change: Any, diagnostic: str
) -> None:
    report = run_audit(tmp_path, source=source_change(SOURCE))
    assert report["status"] == "fail"
    assert diagnostic in report["errors"][0]


def replace_entry_field(source: str, field_index: int, replacement: str) -> str:
    pattern = re.compile(
        r"(ACE_PHANTOM_CONSTANT_ENTRY\s+entry_id=0\s+constant_id=17[^{}]*\{)"
        r"(?P<body>[^{}]*)(\})",
        re.DOTALL,
    )
    match = pattern.search(source)
    assert match is not None
    fields = [field.strip() for field in match.group("body").split(",")]
    assert len(fields) == 13
    fields[field_index] = replacement
    return source[: match.start("body")] + ", ".join(fields) + source[match.end("body") :]


@pytest.mark.parametrize(
    ("field_index", "replacement"),
    (
        (0, "9"),
        (1, "18"),
        (2, "PHANTOM_CONSTANT_UNKNOWN"),
        (3, "3"),
        (4, "2"),
        (5, "2"),
        (6, "2"),
        (7, "0x1.0000000000000p+41"),
        (8, '"other_symbol"'),
        (9, '"' + "0" * 64 + '"'),
        (10, '"' + "0" * 64 + '"'),
        (11, "5"),
        (12, "(const double*)phantom_constant_29"),
    ),
)
def test_each_constant_entry_initializer_field_is_bound_to_json(
    tmp_path: Path, field_index: int, replacement: str
) -> None:
    report = run_audit(
        tmp_path,
        source=replace_entry_field(SOURCE, field_index, replacement),
    )
    assert report["status"] == "fail"
    assert "13-field initializer differs from manifest" in report["errors"][0]


def replace_top_manifest_field(source: str, field_index: int, replacement: str) -> str:
    pattern = re.compile(
        r"(static\s+const\s+PHANTOM_CONSTANT_MANIFEST\s+constants\s*=\s*\{)"
        r"(?P<body>[^}]*)(\})",
        re.DOTALL,
    )
    match = pattern.search(source)
    assert match is not None
    fields = [field.strip() for field in match.group("body").split(",")]
    assert len(fields) == 6
    fields[field_index] = replacement
    return source[: match.start("body")] + ", ".join(fields) + source[match.end("body") :]


@pytest.mark.parametrize(
    ("field_index", "replacement"),
    (
        (0, "2"),
        (1, "2"),
        (2, "4"),
        (3, '"' + "0" * 64 + '"'),
        (4, "1"),
        (5, "nullptr"),
    ),
)
def test_each_top_level_constant_initializer_field_is_bound_to_json(
    tmp_path: Path, field_index: int, replacement: str
) -> None:
    report = run_audit(
        tmp_path,
        source=replace_top_manifest_field(SOURCE, field_index, replacement),
    )
    assert report["status"] == "fail"
    assert "6-field PHANTOM_CONSTANT_MANIFEST initializer differs" in report["errors"][0]


def test_emitted_payload_bytes_are_rehashed_as_little_endian_float64(
    tmp_path: Path,
) -> None:
    report = run_audit(tmp_path, source=SOURCE.replace("2.5, -3.25", "2.75, -3.25"))
    assert report["status"] == "fail"
    assert "payload SHA-256 differs from emitted float64 bytes" in report["errors"][0]


def test_float64_typedef_payload_definitions_are_accepted(tmp_path: Path) -> None:
    source = SOURCE.replace("double phantom_constant_17", "float64_t phantom_constant_17")
    report = run_audit(tmp_path, source=source)
    assert report["status"] == "pass"


def test_non_binary64_payload_typedef_is_rejected(tmp_path: Path) -> None:
    source = SOURCE.replace("double phantom_constant_17", "float32_t phantom_constant_17")
    report = run_audit(tmp_path, source=source)
    assert report["status"] == "fail"
    assert "exactly one binary64-array definition" in report["errors"][0]


def test_emitted_payload_shape_must_equal_twice_the_slot_count(tmp_path: Path) -> None:
    report = run_audit(tmp_path, source=SOURCE.replace("[2][2]", "[2][3]"))
    assert report["status"] == "fail"
    assert "element count differs from its descriptor" in report["errors"][0]


def test_constant_cache_key_covers_context_chain_scale_type_and_slots(
    tmp_path: Path,
) -> None:
    paths = write_inputs(tmp_path)
    constants = json.loads(paths[2].read_text(encoding="utf-8"))
    constants["constants"][0]["chain_index"] += 1
    paths[2].write_bytes(canonical_bytes(constants))
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "cache key is not canonical" in report["errors"][0]


@pytest.mark.parametrize("raw_scale", ["1099511627776", "0x1p+40", "0X1.0P+40"])
def test_constant_scale_requires_canonical_cpp_hexfloat(
    tmp_path: Path, raw_scale: str
) -> None:
    paths = write_inputs(tmp_path)
    constants = json.loads(paths[2].read_text(encoding="utf-8"))
    entry = constants["constants"][0]
    entry["raw_scale"] = raw_scale
    entry["cache_key_sha256"] = cache_key_sha256(
        constants["context_manifest_sha256"], entry
    )
    paths[2].write_bytes(canonical_bytes(constants))
    report = audit(*paths)
    assert report["status"] == "fail"
    assert "raw_scale is not canonical" in report["errors"][0]
