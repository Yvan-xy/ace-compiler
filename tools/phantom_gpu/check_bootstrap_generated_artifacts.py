#!/usr/bin/env python3
"""Audit bootstrap Phantom manifests and their generated CUDA source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(EXAMPLES))

from check_primitive_codegen import compare_context, extract_context
from bootstrap_full import build_bootstrap_trace_config
from ace_edsl.edsl.core.bootstrap_decomposition import (
    build_bootstrap_evalmod_scalar_manifest,
    build_bootstrap_transform_payload_manifest,
)
from bootstrap_domain_attestation import (
    CLEAR_MAP_BUDGET_FRACTION,
    validate_evalmod_air_polynomial,
    validate_supported_identity_domain,
    validate_transform_air_semantics,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
C_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
HEXFLOAT_PATTERN = re.compile(r"^0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+$")
SCHEMA = "ace.phantom.bootstrap-generated-artifact-audit/2.0.0"
V4_SCHEMA = "ace.phantom.bootstrap-generated-artifact-audit/4.0.0"
RAW_SCALE_COORDINATE_TOLERANCE = 1.0e-4

INVOCATION_OPTION_KEYS = {
    "ciphertext_constant_encoding",
    "decode_transform_budget",
    "encode_transform_budget",
    "first_prime_bits",
    "hamming_weight",
    "input_level",
    "mul_level",
    "packing",
    "post_multiply_imag",
    "post_multiply_real",
    "post_multiply_scale_degree",
    "post_rotation_step",
    "poly_degree",
    "q_part_count",
    "scaling_factor_bits",
    "security_level",
    "vector_capacity",
}

REQUIRED_AIR_OPCODES = (
    "ckks.conjugate",
    "ckks.rotate_batch",
    "ckks.raise_mod",
    "ckks.mul_mono",
)
REQUIRED_SOURCE_CALLS = (
    "Conjugate_ciph",
    "Rotate_batch_ciph",
    "Raise_mod",
    "Mul_mono_ciph",
)

FORBIDDEN_AIR = {
    "opaque CKKS bootstrap opcode": re.compile(
        r"\bckks\.bootstrap(?:\b|[._])", re.IGNORECASE
    ),
    "coefficient-to-slot stage opcode": re.compile(
        r"\b(?:ckks\.)?bootstrap_coeffs_to_slots\b", re.IGNORECASE
    ),
    "evaluation stage opcode": re.compile(
        r"\b(?:ckks\.)?bootstrap_eval_mod\b", re.IGNORECASE
    ),
    "slot-to-coefficient stage opcode": re.compile(
        r"\b(?:ckks\.)?bootstrap_slots_to_coeffs\b", re.IGNORECASE
    ),
    "POLY/HPOLY/LPOLY opcode": re.compile(
        r"(?:\bfhe::(?:poly|hpoly|lpoly)\b|\b(?:poly|hpoly|lpoly)\.[A-Za-z_])",
        re.IGNORECASE,
    ),
}

CONTEXT_KEYS = {
    "data_q_bit_sizes",
    "first_modulus_bits",
    "hamming_weight",
    "input_level",
    "logical_slot_capacity",
    "packing",
    "polynomial_degree",
    "q_part_count",
    "resource_schema_version",
    "scaling_modulus_bits",
    "schema_version",
    "security_level",
    "special_p_bit_sizes",
}
RESOURCE_KEYS = {
    "complex_plaintext",
    "conjugation_key",
    "context_schema_version",
    "monomial_powers",
    "native_bootstrap_precompute",
    "raise_mod",
    "relinearization_key",
    "rotate_batch",
    "rotation_batches",
    "rotation_steps",
    "schema_version",
}
CONSTANT_KEYS = {
    "constants",
    "context_manifest_sha256",
    "context_schema_version",
    "resource_schema_version",
    "schema_version",
}
CONSTANT_ENTRY_KEYS = {
    "ace_level",
    "cache_key_sha256",
    "chain_index",
    "constant_id",
    "element_type",
    "entry_id",
    "payload_sha256",
    "raw_scale",
    "scale_degree",
    "slot_count",
    "symbol",
}

RESOURCE_FLAGS = {
    "relinearization_key": "PHANTOM_RESOURCE_RELIN_KEY",
    "conjugation_key": "PHANTOM_RESOURCE_CONJUGATION_KEY",
    "rotate_batch": "PHANTOM_RESOURCE_ROTATE_BATCH",
    "raise_mod": "PHANTOM_RESOURCE_RAISE_MOD",
    "complex_plaintext": "PHANTOM_RESOURCE_COMPLEX_PLAINTEXT",
}
DERIVED_RESOURCE_FLAGS = {
    "rotation_steps": "PHANTOM_RESOURCE_ROTATION_KEYS",
    "monomial_powers": "PHANTOM_RESOURCE_MONOMIALS",
}
NATIVE_RESOURCE_FLAG = "PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE"

FORBIDDEN_NATIVE_BTS = {
    "native Phantom bootstrap header": re.compile(
        r"^\s*#\s*include\s*[<\"]boot/Bootstrapper\.cuh[>\"]", re.MULTILINE
    ),
    "native Phantom bootstrap type": re.compile(r"\bBootstrapper\b"),
    "native Phantom bootstrap entry point": re.compile(
        r"\b(?:Phantom_bootstrap|bootstrap_3)\b"
    ),
    "opaque bootstrap call": re.compile(r"\bBootstrap\s*\("),
    "opaque ANT bootstrap call": re.compile(
        r"\bEval_bootstrap[A-Za-z0-9_]*\s*\("
    ),
    "native Phantom coefficient-to-slot stage": re.compile(
        r"\b(?:bootstrap_coeffs_to_slots|CoeffToSlots?)\b"
    ),
    "native Phantom evaluation stage": re.compile(
        r"\b(?:bootstrap_eval_mod|EvalMod)\b"
    ),
    "native Phantom slot-to-coefficient stage": re.compile(
        r"\b(?:bootstrap_slots_to_coeffs|SlotToCoeffs?)\b"
    ),
}

ARRAY_PATTERN = re.compile(
    r"static\s+const\s+(?:int32_t|uint32_t|size_t)\s+"
    r"(?P<name>[A-Za-z_]\w*)\[\]\s*=\s*\{(?P<values>[^}]*)\}\s*;",
    re.MULTILINE,
)
RESOURCE_PATTERN = re.compile(
    r"static\s+const\s+PHANTOM_RESOURCE_MANIFEST\s+resources\s*=\s*"
    r"\{(?P<fields>[^}]*)\}\s*;",
    re.MULTILINE,
)
CONSTANT_MARKER_PATTERN = re.compile(
    r"ACE_PHANTOM_CONSTANT_ENTRY\s+entry_id=(?P<entry_id>[0-9]+)\s+"
    r"constant_id=(?P<constant_id>[0-9]+)"
)
CONSTANT_ENTRY_ARRAY_PATTERN = re.compile(
    r"static\s+const\s+PHANTOM_CONSTANT_ENTRY\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\[\s*\]\s*=\s*\{"
    r"(?P<body>.*?)\n\s*\}\s*;",
    re.DOTALL,
)
CONSTANT_INITIALIZER_PATTERN = re.compile(
    CONSTANT_MARKER_PATTERN.pattern + r"[^{}]*\{(?P<body>[^{}]*)\}", re.DOTALL
)
CONSTANT_CAST_PATTERN = re.compile(
    r"\(\s*const\s+double\s*\*\s*\)\s*(?P<symbol>[A-Za-z_]\w*)"
)
CONSTANT_MANIFEST_PATTERN = re.compile(
    r"static\s+const\s+PHANTOM_CONSTANT_MANIFEST\s+constants\s*=\s*"
    r"\{(?P<fields>[^}]*)\}\s*;",
    re.MULTILINE,
)
BINARY64_ARRAY_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:static\s+)?(?:const\s+)?(?:double|float64_t)\s+"
    r"(?P<symbol>[A-Za-z_]\w*)"
    r"(?P<dimensions>(?:\s*\[\s*[0-9]+\s*\])+?)\s*=\s*"
    r"\{(?P<values>[^{}]*)\}\s*;",
    re.MULTILINE,
)
LOAD_CACHE_PATTERN = re.compile(
    r"\bLoad_cached_plain\s*\(\s*[^,]+,\s*(?P<entry_id>[0-9]+)\s*\)"
)


class AuditError(ValueError):
    """A deterministic input-contract violation."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-manifest", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--constant-manifest", required=True, type=Path)
    parser.add_argument("--raw-air", required=True, type=Path)
    parser.add_argument("--post-ckks-air", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--ant-source", type=Path)
    parser.add_argument("--generation-record", type=Path)
    parser.add_argument("--compiler-invocation", type=Path)
    parser.add_argument("--bootstrap-semantics", type=Path)
    parser.add_argument("--post-operations-air", type=Path)
    parser.add_argument("--post-operation-attestation", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_uint(value: object) -> bool:
    return type(value) is int and value >= 0


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise AuditError(f"{label} keys differ from schema: missing={missing}, extra={extra}")


def verify_context_manifest(manifest: dict[str, Any]) -> None:
    _require_exact_keys(manifest, CONTEXT_KEYS, "context manifest")
    if manifest["schema_version"] != 1 or manifest["resource_schema_version"] != 3:
        raise AuditError("context manifest must declare schema versions 1 and 3")
    degree = manifest["polynomial_degree"]
    slots = manifest["logical_slot_capacity"]
    if (
        not _is_uint(degree)
        or degree < 2
        or degree > 131072
        or degree & (degree - 1)
        or manifest["packing"] != "full"
        or slots != degree // 2
    ):
        raise AuditError("context manifest has invalid degree, packing, or slots")
    data_q = manifest["data_q_bit_sizes"]
    special_p = manifest["special_p_bit_sizes"]
    if (
        not isinstance(data_q, list)
        or not isinstance(special_p, list)
        or not data_q
        or not special_p
        or any(not _is_uint(bits) or not 2 <= bits <= 60 for bits in data_q + special_p)
    ):
        raise AuditError("context manifest has invalid Q/P arrays")
    if len(data_q) + len(special_p) > 64:
        raise AuditError("context manifest Q/P count exceeds Phantom limits")
    if data_q[0] != manifest["first_modulus_bits"] or any(
        bits != manifest["scaling_modulus_bits"] for bits in data_q[1:]
    ):
        raise AuditError("context manifest Q array disagrees with prime metadata")
    if not _is_uint(manifest["input_level"]) or not 1 <= manifest["input_level"] <= len(data_q):
        raise AuditError("context manifest input level is invalid")
    if not _is_uint(manifest["q_part_count"]) or not 1 <= manifest["q_part_count"] <= len(data_q):
        raise AuditError("context manifest Q-part count is invalid")
    if not _is_uint(manifest["hamming_weight"]) or not 1 <= manifest["hamming_weight"] <= degree:
        raise AuditError("context manifest hamming weight is invalid")
    if manifest["security_level"] not in (0, 128, 192, 256):
        raise AuditError("context manifest security setting is invalid")


def verify_resource_manifest(
    manifest: dict[str, Any], context: dict[str, Any]
) -> None:
    _require_exact_keys(manifest, RESOURCE_KEYS, "resource manifest")
    if manifest["schema_version"] != 3 or manifest["context_schema_version"] != 1:
        raise AuditError("resource manifest must declare schema versions 3 and 1")
    for field in (
        "complex_plaintext",
        "conjugation_key",
        "native_bootstrap_precompute",
        "raise_mod",
        "relinearization_key",
        "rotate_batch",
    ):
        if type(manifest[field]) is not bool:
            raise AuditError(f"resource manifest {field} must be boolean")
    required_true = (
        "complex_plaintext",
        "conjugation_key",
        "raise_mod",
        "relinearization_key",
        "rotate_batch",
    )
    for field in required_true:
        if not manifest[field]:
            raise AuditError(f"full-bootstrap resource {field} must be true")
    if manifest["native_bootstrap_precompute"]:
        raise AuditError("native bootstrap precomputation must be false")

    slots = context["logical_slot_capacity"]
    rotations = manifest["rotation_steps"]
    conjugation_sentinel = 2 * context["polynomial_degree"] - 1
    if isinstance(rotations, list) and conjugation_sentinel in rotations:
        raise AuditError("rotation_steps contains the conjugation sentinel")
    if (
        not isinstance(rotations, list)
        or not rotations
        or any(type(step) is not int or step == 0 or not -slots < step < slots for step in rotations)
        or rotations != sorted(set(rotations))
    ):
        raise AuditError("rotation_steps must be sorted unique real nonzero rotations")

    batches = manifest["rotation_batches"]
    if not isinstance(batches, list) or not batches:
        raise AuditError("full-bootstrap rotation_batches must be nonempty")
    rotation_set = set(rotations)
    for index, batch in enumerate(batches):
        if not isinstance(batch, list) or not batch:
            raise AuditError(f"rotation batch {index} must be a nonempty list")
        if any(type(step) is not int or not -slots < step < slots for step in batch):
            raise AuditError(f"rotation batch {index} contains a non-real step")
        canonical_steps = []
        for step in batch:
            reduced = step % slots
            canonical = reduced - slots if reduced > slots // 2 else reduced
            if canonical != 0:
                canonical_steps.append(canonical)
        if any(step not in rotation_set for step in canonical_steps):
            raise AuditError(f"rotation batch {index} is not closed by rotation_steps")

    monomials = manifest["monomial_powers"]
    period = 2 * context["polynomial_degree"]
    if (
        not isinstance(monomials, list)
        or not monomials
        or any(not _is_uint(power) or power >= period for power in monomials)
        or monomials != sorted(set(monomials))
    ):
        raise AuditError("monomial_powers must be sorted, unique, and canonical in [0, 2N)")


def cache_key_sha256(context_manifest_sha256: str, entry: dict[str, Any]) -> str:
    """Return the canonical bootstrap qualification immutable-plaintext cache key digest."""
    key = {
        "chain_index": entry["chain_index"],
        "constant_id": entry["constant_id"],
        "context_manifest_sha256": context_manifest_sha256,
        "element_type": entry["element_type"],
        "raw_scale": entry["raw_scale"],
        "slot_count": entry["slot_count"],
    }
    encoded = json.dumps(key, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _sha256_bytes(encoded)


def _verify_hexfloat(value: object) -> bool:
    if not isinstance(value, str) or HEXFLOAT_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed > 0 and parsed.hex() == value


def verify_constant_manifest(
    manifest: dict[str, Any], context: dict[str, Any], context_sha256: str
) -> None:
    _require_exact_keys(manifest, CONSTANT_KEYS, "constant manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["context_schema_version"] != 1
        or manifest["resource_schema_version"] != 3
    ):
        raise AuditError("constant manifest must declare schema versions 1, 1, and 3")
    if manifest["context_manifest_sha256"] != context_sha256:
        raise AuditError("constant manifest context hash does not match its context sidecar")
    constants = manifest["constants"]
    if not isinstance(constants, list) or not constants:
        raise AuditError("full-bootstrap constant manifest must be nonempty")
    cache_keys: set[str] = set()
    symbol_metadata: dict[str, tuple[str, str, int]] = {}
    for expected_entry_id, entry in enumerate(constants):
        if not isinstance(entry, dict):
            raise AuditError(f"constant entry {expected_entry_id} is not an object")
        _require_exact_keys(entry, CONSTANT_ENTRY_KEYS, f"constant entry {expected_entry_id}")
        if entry["entry_id"] != expected_entry_id:
            raise AuditError("constant entry IDs must be contiguous and ordered from zero")
        for field in ("ace_level", "chain_index", "constant_id", "entry_id"):
            if not _is_uint(entry[field]):
                raise AuditError(f"constant entry {expected_entry_id} {field} must be uint")
        if not 1 <= entry["ace_level"] <= len(context["data_q_bit_sizes"]):
            raise AuditError(f"constant entry {expected_entry_id} ACE level is invalid")
        if type(entry["scale_degree"]) is not int:
            raise AuditError(f"constant entry {expected_entry_id} scale_degree must be integer")
        if entry["element_type"] != "complex_f64":
            raise AuditError(f"constant entry {expected_entry_id} element_type is unsupported")
        if not _is_uint(entry["slot_count"]) or not 1 <= entry["slot_count"] <= context["logical_slot_capacity"]:
            raise AuditError(f"constant entry {expected_entry_id} slot_count is invalid")
        if not _verify_hexfloat(entry["raw_scale"]):
            raise AuditError(f"constant entry {expected_entry_id} raw_scale is not canonical C++ hexfloat")
        if not isinstance(entry["symbol"], str) or C_IDENTIFIER_PATTERN.fullmatch(entry["symbol"]) is None:
            raise AuditError(f"constant entry {expected_entry_id} symbol is not a C identifier")
        for field in ("cache_key_sha256", "payload_sha256"):
            if not isinstance(entry[field], str) or SHA256_PATTERN.fullmatch(entry[field]) is None:
                raise AuditError(f"constant entry {expected_entry_id} {field} is not lowercase SHA-256")
        expected_key = cache_key_sha256(context_sha256, entry)
        if entry["cache_key_sha256"] != expected_key:
            raise AuditError(f"constant entry {expected_entry_id} cache key is not canonical")
        if entry["cache_key_sha256"] in cache_keys:
            raise AuditError("constant manifest contains a duplicate cache key")
        cache_keys.add(entry["cache_key_sha256"])
        metadata = (entry["payload_sha256"], entry["element_type"], entry["slot_count"])
        prior = symbol_metadata.setdefault(entry["symbol"], metadata)
        if prior != metadata:
            raise AuditError(f"constant symbol {entry['symbol']} has inconsistent payload metadata")


def _comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _source_arrays(source: str) -> dict[str, list[int]]:
    arrays: dict[str, list[int]] = {}
    for match in ARRAY_PATTERN.finditer(source):
        try:
            arrays[match.group("name")] = [
                int(value) for value in _comma_values(match.group("values"))
            ]
        except ValueError as error:
            raise AuditError(f"generated array {match.group('name')} is not integral") from error
    return arrays


def _one_suffix(arrays: dict[str, list[int]], suffix: str) -> tuple[str, list[int]]:
    matches = [(name, values) for name, values in arrays.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise AuditError(f"generated source must define exactly one {suffix} array")
    return matches[0]


def compare_resource_source(source: str, manifest: dict[str, Any]) -> None:
    arrays = _source_arrays(source)
    rotation_name, rotations = _one_suffix(arrays, "phantom_rotation_steps")
    offsets_name, offsets = _one_suffix(arrays, "phantom_rotation_batch_offsets")
    batch_name, batch_steps = _one_suffix(arrays, "phantom_rotation_batch_steps")
    monomial_name, monomials = _one_suffix(arrays, "phantom_monomial_powers")
    expected_offsets = [0]
    flattened_batches: list[int] = []
    for batch in manifest["rotation_batches"]:
        flattened_batches.extend(batch)
        expected_offsets.append(len(flattened_batches))
    expected_arrays = {
        rotation_name: manifest["rotation_steps"],
        offsets_name: expected_offsets,
        batch_name: flattened_batches,
        monomial_name: manifest["monomial_powers"],
    }
    for name, expected in expected_arrays.items():
        if arrays[name] != expected:
            raise AuditError(f"generated {name} differs from resource manifest")

    match = RESOURCE_PATTERN.search(source)
    if match is None:
        raise AuditError("PHANTOM_RESOURCE_MANIFEST initializer is absent")
    fields = _comma_values(match.group("fields"))
    if len(fields) != 10:
        raise AuditError("PHANTOM_RESOURCE_MANIFEST initializer has the wrong arity")
    expected_flags = {
        flag for field, flag in RESOURCE_FLAGS.items() if manifest[field]
    }
    expected_flags.update(
        flag for field, flag in DERIVED_RESOURCE_FLAGS.items() if manifest[field]
    )
    if manifest["native_bootstrap_precompute"]:
        expected_flags.add(NATIVE_RESOURCE_FLAG)
    observed_flags = {token.strip() for token in fields[2].split("|")}
    if observed_flags != expected_flags:
        raise AuditError("generated resource flags differ from resource manifest")
    expected_fields = [
        "3",
        "1",
        fields[2],
        str(len(manifest["rotation_steps"])),
        rotation_name,
        str(len(manifest["rotation_batches"])),
        offsets_name,
        batch_name,
        str(len(manifest["monomial_powers"])),
        monomial_name,
    ]
    if fields != expected_fields:
        raise AuditError("generated resource initializer differs from resource manifest")


def _quoted_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _payload_definitions(source: str) -> dict[str, list[re.Match[str]]]:
    definitions: dict[str, list[re.Match[str]]] = {}
    for definition in BINARY64_ARRAY_PATTERN.finditer(source):
        definitions.setdefault(definition.group("symbol"), []).append(definition)
    return definitions


def _payload_bytes(
    definitions: dict[str, list[re.Match[str]]], symbol: str, expected_count: int
) -> bytes:
    matches = definitions.get(symbol, [])
    if len(matches) != 1:
        raise AuditError(
            f"constant payload symbol {symbol} must have exactly one binary64-array definition"
        )
    definition = matches[0]
    dimensions = [
        int(value)
        for value in re.findall(r"\[\s*([0-9]+)\s*\]", definition.group("dimensions"))
    ]
    declared_count = math.prod(dimensions)
    tokens = _comma_values(definition.group("values"))
    if declared_count != expected_count or len(tokens) != expected_count:
        raise AuditError(
            f"constant payload symbol {symbol} element count differs from its descriptor"
        )
    result = bytearray()
    for index, token in enumerate(tokens):
        try:
            value = float.fromhex(token) if "0x" in token.lower() else float(token)
        except ValueError as error:
            raise AuditError(
                f"constant payload symbol {symbol} element {index} is not float64"
            ) from error
        if not math.isfinite(value):
            raise AuditError(
                f"constant payload symbol {symbol} element {index} is non-finite"
            )
        result.extend(struct.pack("<d", value))
    return bytes(result)


def compare_constant_payload_source(
    source: str, manifest: dict[str, Any], source_name: str
) -> None:
    """Bind each manifest payload to its exact emitted terminal-source bytes."""
    definitions = _payload_definitions(source)
    for entry in manifest["constants"]:
        payload = _payload_bytes(
            definitions, entry["symbol"], entry["slot_count"] * 2
        )
        if _sha256_bytes(payload) != entry["payload_sha256"]:
            raise AuditError(
                f"constant entry {entry['entry_id']} payload SHA-256 differs "
                f"from {source_name} float64 bytes"
            )


def compare_constant_source(source: str, manifest: dict[str, Any]) -> None:
    entry_arrays = list(CONSTANT_ENTRY_ARRAY_PATTERN.finditer(source))
    if len(entry_arrays) != 1:
        raise AuditError("generated source must define exactly one PHANTOM_CONSTANT_ENTRY array")
    entry_array = entry_arrays[0]
    entry_array_name = entry_array.group("name")
    if "Get_phantom_constant_manifest()" not in source:
        raise AuditError("Get_phantom_constant_manifest accessor is absent")
    expected = {entry["entry_id"]: entry for entry in manifest["constants"]}

    markers = [
        (int(match.group("entry_id")), int(match.group("constant_id")))
        for match in CONSTANT_MARKER_PATTERN.finditer(entry_array.group("body"))
    ]
    if len(markers) != len(set(markers)):
        raise AuditError("generated source contains duplicate constant entry markers")
    expected_markers = {(entry_id, entry["constant_id"]) for entry_id, entry in expected.items()}
    if set(markers) != expected_markers:
        raise AuditError("constant manifest entries and generated entry markers differ")

    initializer_matches = list(CONSTANT_INITIALIZER_PATTERN.finditer(entry_array.group("body")))
    if len(initializer_matches) != len(expected):
        raise AuditError("constant marker/initializer count differs from manifest")
    observed_cast_symbols: list[str] = []
    for match in initializer_matches:
        entry_id = int(match.group("entry_id"))
        entry = expected[entry_id]
        fields = _comma_values(match.group("body"))
        expected_fields = [
            str(entry["entry_id"]),
            str(entry["constant_id"]),
            "PHANTOM_CONSTANT_COMPLEX_F64",
            str(entry["slot_count"]),
            str(entry["ace_level"]),
            str(entry["chain_index"]),
            str(entry["scale_degree"]),
            entry["raw_scale"],
            _quoted_string(entry["symbol"]),
            _quoted_string(entry["payload_sha256"]),
            _quoted_string(entry["cache_key_sha256"]),
            str(entry["slot_count"] * 2),
            f"(const double*){entry['symbol']}",
        ]
        if len(fields) != 13:
            raise AuditError(f"constant entry {entry_id} initializer must have 13 fields")
        if fields != expected_fields:
            raise AuditError(
                f"constant entry {entry_id} 13-field initializer differs from manifest"
            )
        observed_cast_symbols.append(entry["symbol"])
    all_cast_symbols = [match.group("symbol") for match in CONSTANT_CAST_PATTERN.finditer(source)]
    if all_cast_symbols != observed_cast_symbols:
        raise AuditError("generated source contains an unbound constant payload cast")

    payload_definitions = _payload_definitions(source)
    symbol_payloads: dict[str, bytes] = {}
    for entry in expected.values():
        symbol = entry["symbol"]
        payload = symbol_payloads.get(symbol)
        if payload is None:
            payload = _payload_bytes(
                payload_definitions, symbol, entry["slot_count"] * 2
            )
            symbol_payloads[symbol] = payload
        if _sha256_bytes(payload) != entry["payload_sha256"]:
            raise AuditError(
                f"constant entry {entry['entry_id']} payload SHA-256 differs from emitted float64 bytes"
            )
        expected_cache_key = cache_key_sha256(manifest["context_manifest_sha256"], entry)
        if entry["cache_key_sha256"] != expected_cache_key:
            raise AuditError(
                f"constant entry {entry['entry_id']} cache key SHA-256 is not canonical"
            )

    top_matches = list(CONSTANT_MANIFEST_PATTERN.finditer(source))
    if len(top_matches) != 1:
        raise AuditError(
            "generated source must define exactly one PHANTOM_CONSTANT_MANIFEST initializer"
        )
    top_fields = _comma_values(top_matches[0].group("fields"))
    expected_top_fields = [
        str(manifest["schema_version"]),
        str(manifest["context_schema_version"]),
        str(manifest["resource_schema_version"]),
        _quoted_string(manifest["context_manifest_sha256"]),
        str(len(manifest["constants"])),
        entry_array_name,
    ]
    if len(top_fields) != 6:
        raise AuditError("PHANTOM_CONSTANT_MANIFEST initializer must have 6 fields")
    if top_fields != expected_top_fields:
        raise AuditError("6-field PHANTOM_CONSTANT_MANIFEST initializer differs from manifest")

    loads = [int(match.group("entry_id")) for match in LOAD_CACHE_PATTERN.finditer(source)]
    unknown_loads = sorted(set(loads) - set(expected))
    missing_loads = sorted(set(expected) - set(loads))
    if unknown_loads or missing_loads:
        raise AuditError(
            "Load_cached_plain entry closure differs from manifest: "
            f"missing={missing_loads}, unknown={unknown_loads}"
        )


def inspect_air(air: str) -> dict[str, Any]:
    lowered = air.lower()
    return {
        "required_opcode_counts": {
            opcode: lowered.count(opcode) for opcode in REQUIRED_AIR_OPCODES
        },
        "forbidden_matches": [
            label for label, pattern in FORBIDDEN_AIR.items() if pattern.search(air)
        ],
    }


def inspect_source(source: str) -> dict[str, Any]:
    return {
        "required_call_counts": {
            call: len(re.findall(r"\b" + re.escape(call) + r"\s*\(", source))
            for call in REQUIRED_SOURCE_CALLS
        },
        "forbidden_matches": [
            label
            for label, pattern in FORBIDDEN_NATIVE_BTS.items()
            if pattern.search(source)
        ],
    }


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise AuditError(f"{label} must be lowercase SHA-256")
    return value


def _verify_file_record(record: object, path: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise AuditError(f"{label} must be an object")
    _require_exact_keys(record, {"path", "sha256", "bytes"}, label)
    payload = path.read_bytes()
    if record != {
        "path": path.name,
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
    }:
        raise AuditError(f"{label} differs from its artifact")


def _operation_attributes(air: str, opcode: str) -> list[dict[str, int]]:
    result = []
    for value in re.findall(
        rf"CKKS\.{re.escape(opcode)} ATTR\[([^\]]+)\]", air
    ):
        attributes = {}
        for name in ("level", "rescale_level", "scale"):
            match = re.search(rf"(?:^|,){name}=(-?[0-9]+)(?:,|$)", value)
            if match is None:
                raise AuditError(f"{opcode} AIR omits {name}")
            attributes[name] = int(match.group(1))
        result.append(attributes)
    return result


def _terminal_attributes(air: str) -> dict[str, int]:
    matches = re.findall(
        r'st "__ret_tmp_[^"]+"[^\n]*ATTR\[([^\]]+)\][^\n]*\n'
        r'\s*ld "__ret_tmp_[^"]+"[^\n]*\n\s*retv\b',
        air,
    )
    if len(matches) != 1:
        raise AuditError("post-CKKS AIR must have one terminal return metadata record")
    result = _operation_attributes(
        "CKKS.terminal ATTR[" + matches[0] + "]", "terminal"
    )
    return result[0]


BOOTSTRAP_ABI = re.compile(
    r"\bCIPHERTEXT\s+bootstrap_full\s*\(\s*"
    r"CIPHERTEXT\s+(?P<input>[A-Za-z_]\w*)\s*,\s*"
    r"CIPHERTEXT\s+(?P<auxiliary>[A-Za-z_]\w*)\s*\)\s*\{"
)


def _verify_bootstrap_abi(source: str, label: str) -> None:
    if len(BOOTSTRAP_ABI.findall(source)) != 1:
        raise AuditError(
            f"{label} must define exact bootstrap_full(CIPHERTEXT,CIPHERTEXT) ABI"
        )


def _bootstrap_function_body(
    source: str, label: str
) -> tuple[str, tuple[str, str]]:
    matches = list(BOOTSTRAP_ABI.finditer(source))
    if len(matches) != 1:
        raise AuditError(
            f"{label} must define exact bootstrap_full(CIPHERTEXT,CIPHERTEXT) ABI"
        )
    match = matches[0]
    start = match.end() - 1
    depth = 1
    index = start + 1
    mode = "code"
    quote = ""
    while index < len(source) and depth:
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if mode == "line-comment":
            if character == "\n":
                mode = "code"
        elif mode == "block-comment":
            if character == "*" and following == "/":
                mode = "code"
                index += 1
        elif mode == "string":
            if character == "\\":
                index += 1
            elif character == quote:
                mode = "code"
        elif character == "/" and following == "/":
            mode = "line-comment"
            index += 1
        elif character == "/" and following == "*":
            mode = "block-comment"
            index += 1
        elif character in {'"', "'"}:
            mode = "string"
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        index += 1
    if depth != 0:
        raise AuditError(f"{label} bootstrap_full body is not balanced")
    return source[start + 1 : index - 1], (
        match.group("input"),
        match.group("auxiliary"),
    )


def _split_call_arguments(arguments: str) -> list[str]:
    result = []
    start = 0
    depth = 0
    for index, character in enumerate(arguments):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def _variable_reference(value: str) -> str | None:
    match = re.search(r"&?\s*([A-Za-z_]\w*(?:\[[0-9]+\])?)", value)
    return match.group(1) if match is not None else None


def _verify_terminal_copy(
    body: str, parameters: tuple[str, str], label: str
) -> tuple[str, int, int]:
    returns = list(re.finditer(r"\breturn\s+([A-Za-z_]\w*)\s*;", body))
    if len(returns) != 1 or returns[0].group(1) in parameters:
        raise AuditError(f"{label} bootstrap_full does not return an owned result")
    returned = returns[0].group(1)
    copies = list(
        re.finditer(
            rf"\bCopy_ciph\s*\(\s*&{re.escape(returned)}\s*,\s*"
            r"&([A-Za-z_]\w*(?:\[[0-9]+\])?)\s*\)\s*;",
            body,
        )
    )
    if (
        len(copies) != 1
        or copies[0].group(1) in parameters
        or copies[0].start() >= returns[0].start()
    ):
        raise AuditError(
            f"{label} bootstrap_full return is not copied from a generated value"
        )
    return returned, copies[0].start(), returns[0].start()


_PHANTOM_VALUE_PRODUCERS = {
    "Add_ciph",
    "Add_plain",
    "Conjugate_ciph",
    "Copy_ciph",
    "Encode_double_mask",
    "Mod_switch",
    "Mul_ciph",
    "Mul_mono_ciph",
    "Mul_plain",
    "Raise_mod",
    "Relin",
    "Rescale_ciph",
    "Rotate_batch_ciph",
    "Rotate_ciph",
    "Sub_ciph",
}

_AIR_TERMINAL_OPERATION_PATTERN = re.compile(
    r"\bCKKS\.(add|sub|mul|rescale|rotate|conjugate|raise_mod|"
    r"rotate_batch|mul_mono|modswitch)\b"
)
_PHANTOM_TERMINAL_OPERATIONS = {
    "Add_ciph": "add",
    "Add_plain": "add",
    "Sub_ciph": "sub",
    "Sub_plain": "sub",
    "Mul_ciph": "mul",
    "Mul_plain": "mul",
    "Rescale_ciph": "rescale",
    "Rotate_ciph": "rotate",
    "Conjugate_ciph": "conjugate",
    "Raise_mod": "raise_mod",
    "Rotate_batch_ciph": "rotate_batch",
    "Mul_mono_ciph": "mul_mono",
    "Mod_switch": "modswitch",
}


def _post_ckks_operation_trace(post_air: str) -> tuple[str, ...]:
    trace = tuple(_AIR_TERMINAL_OPERATION_PATTERN.findall(post_air))
    if not trace:
        raise AuditError("canonical post-CKKS AIR has no terminal operations")
    return trace


def _phantom_operation_trace(body: str) -> tuple[str, ...]:
    names = "|".join(
        re.escape(name)
        for name in sorted(_PHANTOM_TERMINAL_OPERATIONS, key=len, reverse=True)
    )
    return tuple(
        _PHANTOM_TERMINAL_OPERATIONS[match.group(1)]
        for match in re.finditer(rf"\b({names})\s*\(", body)
    )


def _require_operation_trace(
    observed: tuple[str, ...], expected: tuple[str, ...], label: str
) -> None:
    if observed == expected:
        return
    mismatch = next(
        (
            index
            for index, (observed_op, expected_op) in enumerate(
                zip(observed, expected)
            )
            if observed_op != expected_op
        ),
        min(len(observed), len(expected)),
    )
    observed_op = observed[mismatch] if mismatch < len(observed) else "<end>"
    expected_op = expected[mismatch] if mismatch < len(expected) else "<end>"
    raise AuditError(
        f"{label} terminal operation trace differs from canonical post-CKKS "
        f"AIR at operation {mismatch}: observed={observed_op}, "
        f"expected={expected_op}, observed_count={len(observed)}, "
        f"expected_count={len(expected)}"
    )


def _verify_phantom_bootstrap_dataflow(
    body: str,
    parameters: tuple[str, str],
    manifest: dict[str, Any],
    expected_operation_trace: tuple[str, ...],
) -> dict[str, Any]:
    returned, terminal_copy_position, return_position = _verify_terminal_copy(
        body, parameters, "Phantom source"
    )
    reachable_body = body[:return_position]
    _require_operation_trace(
        _phantom_operation_trace(reachable_body),
        expected_operation_trace,
        "Phantom source",
    )
    graph: dict[str, list[str]] = {}
    producer: dict[str, str] = {}
    constant_leaf: dict[str, int] = {}
    call_pattern = re.compile(
        r"\b([A-Za-z_]\w*)\s*\((.*?)\)\s*;", re.DOTALL
    )
    for call in call_pattern.finditer(reachable_body):
        name = call.group(1)
        arguments = _split_call_arguments(call.group(2))
        if not arguments:
            continue
        output = _variable_reference(arguments[0])
        if output is None:
            continue
        if name == "Load_cached_plain":
            if len(arguments) != 2 or not arguments[1].isdigit():
                raise AuditError("Phantom source has a malformed cached-plain load")
            graph[output] = []
            producer[output] = name
            constant_leaf[output] = int(arguments[1])
        elif name in _PHANTOM_VALUE_PRODUCERS:
            dependencies = []
            for argument in arguments[1:]:
                dependencies.extend(
                    re.findall(
                        r"&\s*([A-Za-z_]\w*(?:\[[0-9]+\])?)",
                        argument,
                    )
                )
            graph[output] = dependencies
            producer[output] = name

    reachable_constants: set[int] = set()
    reachable_producers: set[str] = set()
    visited: set[str] = set()

    def visit(value: str) -> None:
        base = value.split("[", 1)[0]
        node = value if value in graph else base
        if node in visited:
            return
        visited.add(node)
        if node in producer:
            reachable_producers.add(producer[node])
        if node in constant_leaf:
            reachable_constants.add(constant_leaf[node])
        for dependency in graph.get(node, []):
            visit(dependency)

    visit(returned)
    expected_constants = {
        entry["entry_id"] for entry in manifest["constants"]
    }
    if reachable_constants != expected_constants:
        missing = sorted(expected_constants - reachable_constants)
        extra = sorted(reachable_constants - expected_constants)
        raise AuditError(
            "Phantom bootstrap return does not depend on every transform "
            f"constant: missing={missing}, extra={extra}"
        )
    required_producers = set(REQUIRED_SOURCE_CALLS)
    if not required_producers <= reachable_producers:
        raise AuditError(
            "Phantom bootstrap return does not depend on every required stage: "
            + ", ".join(sorted(required_producers - reachable_producers))
        )
    if parameters[0] not in visited:
        raise AuditError(
            "Phantom bootstrap return does not depend on its primary input"
        )
    loads = [
        int(value)
        for value in re.findall(
            r"\bLoad_cached_plain\s*\(\s*[^,]+,\s*([0-9]+)\s*\)",
            body[:terminal_copy_position],
        )
    ]
    if loads != [entry["entry_id"] for entry in manifest["constants"]]:
        raise AuditError(
            "Phantom bootstrap body cached-plain order differs from the manifest"
        )
    scalar_encodings = []
    for call in call_pattern.finditer(body[:terminal_copy_position]):
        if call.group(1) != "Encode_double_mask":
            continue
        arguments = _split_call_arguments(call.group(2))
        output = _variable_reference(arguments[0]) if arguments else None
        if len(arguments) != 5 or output not in visited:
            raise AuditError(
                "Phantom bootstrap return does not depend on every scalar encode"
            )
        try:
            value = float(arguments[1])
            length = int(arguments[2], 0)
            scale_degree = int(arguments[3], 0)
        except ValueError as error:
            raise AuditError(
                "Phantom bootstrap has a malformed scalar encode"
            ) from error
        if not math.isfinite(value) or length <= 0 or scale_degree <= 0:
            raise AuditError("Phantom bootstrap scalar encode is invalid")
        scalar_encodings.append(
            (struct.pack("<d", value), length, scale_degree)
        )
    if not scalar_encodings:
        raise AuditError("Phantom bootstrap body has no reachable EvalMod scalars")
    scalar_payload = b"".join(
        value + struct.pack("<QQ", length, scale_degree)
        for value, length, scale_degree in scalar_encodings
    )
    return {
        "reachable_value_count": len(visited),
        "transform_constant_count": len(reachable_constants),
        "reachable_required_stage_count": len(required_producers),
        "scalar_encode_count": len(scalar_encodings),
        "scalar_encode_payload_sha256": _sha256_bytes(scalar_payload),
    }


_ANT_POINTER_PRODUCERS = _PHANTOM_VALUE_PRODUCERS | {
    "Init_ciph3_up_scale",
    "Init_ciph_down_scale",
    "Init_ciph_same_scale",
    "Init_ciph_same_scale_plain",
    "Init_ciph_up_scale_plain",
}
_ANT_RETURN_PRODUCERS = {"Relinearize", "Rotate"}


def _ant_lowered_operation_trace(body: str) -> tuple[str, ...]:
    events: list[tuple[int, str]] = []
    for name, operation in (
        ("Raise_mod", "raise_mod"),
        ("Rotate_batch_ciph", "rotate_batch"),
        ("Mul_mono_ciph", "mul_mono"),
        ("Conjugate_ciph", "conjugate"),
    ):
        events.extend(
            (match.start(), operation)
            for match in re.finditer(rf"\b{re.escape(name)}\s*\(", body)
        )
    events.extend(
        (match.start(), "rotate")
        for match in re.finditer(
            r"\b[A-Za-z_]\w*\s*=\s*Rotate\s*\(", body
        )
    )

    hardware_writes: dict[str, list[tuple[int, str]]] = {}
    for match in re.finditer(
        r"\bHw_mod(add|sub|mul)\s*\(\s*Coeffs\(\s*&"
        r"([A-Za-z_]\w*)(?:\.|\s*,)",
        body,
    ):
        hardware_writes.setdefault(match.group(2), []).append(
            (match.start(), match.group(1))
        )
    if not hardware_writes:
        raise AuditError(
            "generated DSL/ANT source has no lowered modular-arithmetic writes"
        )

    plaintext_multiply_initializers = list(
        re.finditer(
            r"\bInit_ciph_up_scale_plain\s*\(\s*&([A-Za-z_]\w*)",
            body,
        )
    )
    for initializer in plaintext_multiply_initializers:
        output = initializer.group(1)
        output_operations = {
            operation for _, operation in hardware_writes.get(output, [])
        }
        if output_operations != {"mul"}:
            raise AuditError(
                "generated DSL/ANT plaintext-multiply lowering has an "
                f"unexpected modular operation for {output}: "
                f"{sorted(output_operations)}"
            )
        events.append((initializer.start(), "mul"))

    ciphertext_multiply_initializers = list(
        re.finditer(
            r"\bInit_ciph3_up_scale\s*\(\s*&([A-Za-z_]\w*)",
            body,
        )
    )
    relinearizations = list(
        re.finditer(r"\b[A-Za-z_]\w*\s*=\s*Relinearize\s*\(", body)
    )
    if len(ciphertext_multiply_initializers) != len(relinearizations):
        raise AuditError(
            "generated DSL/ANT ciphertext-multiply/relinearize counts differ"
        )
    for index, (initializer, relinearization) in enumerate(
        zip(ciphertext_multiply_initializers, relinearizations)
    ):
        if initializer.start() >= relinearization.start() or (
            index + 1 < len(ciphertext_multiply_initializers)
            and ciphertext_multiply_initializers[index + 1].start()
            < relinearization.start()
        ):
            raise AuditError(
                "generated DSL/ANT ciphertext-multiply lowering is not "
                "paired with its relinearization"
            )
        lowering_operations = tuple(
            match.group(1)
            for match in re.finditer(
                r"\bHw_mod(add|sub|mul)\s*\(",
                body[initializer.end() : relinearization.start()],
            )
        )
        if lowering_operations != ("mul", "mul", "mul", "add", "mul"):
            raise AuditError(
                "generated DSL/ANT ciphertext-multiply modular operation "
                f"trace is invalid: {lowering_operations}"
            )
        events.append((initializer.start(), "mul"))

    modswitch_writes: dict[str, list[int]] = {}
    for match in re.finditer(
        r"\bModswitch\s*\(\s*&([A-Za-z_]\w*)\.", body
    ):
        modswitch_writes.setdefault(match.group(1), []).append(match.start())
    for initializer in re.finditer(
        r"\bInit_ciph_same_scale(?:_plain)?\s*\(\s*&([A-Za-z_]\w*)",
        body,
    ):
        output = initializer.group(1)
        output_operations = {
            operation for _, operation in hardware_writes.get(output, [])
        }
        has_modswitch = output in modswitch_writes
        if output_operations:
            if has_modswitch or len(output_operations) != 1 or not (
                output_operations <= {"add", "sub"}
            ):
                raise AuditError(
                    "generated DSL/ANT same-scale lowering has mixed modular "
                    f"operations for {output}: {sorted(output_operations)}"
                )
            events.append((initializer.start(), next(iter(output_operations))))
        elif has_modswitch:
            events.append((initializer.start(), "modswitch"))

    scale_writes: dict[str, set[str]] = {}
    for runtime_name, operation in (
        ("Rescale", "rescale"),
        ("Modswitch", "modswitch"),
    ):
        for match in re.finditer(
            rf"\b{runtime_name}\s*\(\s*&([A-Za-z_]\w*)\.", body
        ):
            scale_writes.setdefault(match.group(1), set()).add(operation)
    for initializer in re.finditer(
        r"\bInit_ciph_down_scale\s*\(\s*&([A-Za-z_]\w*)", body
    ):
        output = initializer.group(1)
        operations = scale_writes.get(output, set())
        if len(operations) != 1:
            raise AuditError(
                "generated DSL/ANT down-scale lowering has invalid runtime "
                f"operations for {output}: {sorted(operations)}"
            )
        events.append((initializer.start(), next(iter(operations))))

    events.sort()
    return tuple(operation for _, operation in events)


def _verify_ant_bootstrap_payload_uses(
    body: str,
    parameters: tuple[str, str],
    manifest: dict[str, Any],
    num_p: int,
    expected_operation_trace: tuple[str, ...],
) -> dict[str, Any]:
    returned, terminal_copy_position, return_position = _verify_terminal_copy(
        body, parameters, "generated DSL/ANT source"
    )
    prefix = body[:terminal_copy_position]
    reachable_body = body[:return_position]
    _require_operation_trace(
        _ant_lowered_operation_trace(reachable_body),
        expected_operation_trace,
        "generated DSL/ANT source",
    )
    encode_positions = []
    for entry in manifest["constants"]:
        pattern = re.compile(
            r"\bEncode_dcmplx_ext\s*\([^;]*?\(DCMPLX\s*\*\)\s*"
            + re.escape(entry["symbol"])
            + rf"\s*,\s*{entry['slot_count']}\s*,\s*"
            + rf"{entry['ace_level']}\s*,\s*{num_p}\s*\)\s*;",
            re.DOTALL,
        )
        matches = list(pattern.finditer(prefix))
        if len(matches) != 1:
            raise AuditError(
                "generated DSL/ANT bootstrap body does not use constant "
                f"{entry['entry_id']} with its attested encode metadata"
            )
        encode_positions.append(matches[0].start())
    if encode_positions != sorted(encode_positions):
        raise AuditError(
            "generated DSL/ANT bootstrap constant encodes differ from manifest order"
        )

    producer_names = sorted(
        _ANT_POINTER_PRODUCERS
        | {"Encode_dcmplx_ext", "Encode_double_mask"},
        key=len,
        reverse=True,
    )
    pointer_calls = re.compile(
        r"\b("
        + "|".join(re.escape(name) for name in producer_names)
        + r")\s*\((.*?)\)\s*;",
        re.DOTALL,
    )
    assignments = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*"
        r"([A-Za-z_]\w*(?:\[[0-9]+\])?)\s*;"
    )
    return_calls = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*("
        + "|".join(sorted(_ANT_RETURN_PRODUCERS))
        + r")\s*"
        r"\((.*?)\)\s*;",
        re.DOTALL,
    )
    events = [
        (match.start(), "call", match)
        for match in pointer_calls.finditer(reachable_body)
    ]
    events.extend(
        (match.start(), "assignment", match)
        for match in assignments.finditer(reachable_body)
    )
    events.extend(
        (match.start(), "return-call", match)
        for match in return_calls.finditer(reachable_body)
    )
    events.sort(key=lambda value: value[0])
    symbol_entries = {
        entry["symbol"]: entry["entry_id"] for entry in manifest["constants"]
    }
    dependencies: dict[str, set[tuple[str, object]]] = {
        parameter: {("parameter", parameter)} for parameter in parameters
    }
    scalar_encodings: list[tuple[bytes, int, int]] = []

    def value_dependencies(value: str) -> set[tuple[str, object]]:
        return dependencies.get(
            value,
            dependencies.get(value.split("[", 1)[0], set()),
        )

    for _, event_kind, match in events:
        if event_kind == "assignment":
            dependencies[match.group(1)] = set(
                value_dependencies(match.group(2))
            )
            continue
        if event_kind == "return-call":
            arguments = _split_call_arguments(match.group(3))
            inputs = [
                _variable_reference(argument) for argument in arguments
            ]
            value = {("stage", match.group(2))}
            for input_value in inputs:
                if input_value is not None:
                    value.update(value_dependencies(input_value))
            dependencies[match.group(1)] = value
            continue

        name = match.group(1)
        arguments = _split_call_arguments(match.group(2))
        output = _variable_reference(arguments[0]) if arguments else None
        if output is None:
            continue
        if name == "Encode_dcmplx_ext":
            symbol_match = (
                re.search(
                    r"\(DCMPLX\s*\*\)\s*([A-Za-z_]\w*)",
                    arguments[1],
                )
                if len(arguments) == 5
                else None
            )
            if (
                symbol_match is None
                or symbol_match.group(1) not in symbol_entries
            ):
                continue
            dependencies[output] = {
                ("constant", symbol_entries[symbol_match.group(1)]),
                ("stage", name),
            }
            continue
        if name == "Encode_double_mask":
            if len(arguments) != 5:
                raise AuditError(
                    "generated DSL/ANT bootstrap has a malformed scalar encode"
                )
            try:
                scalar = float(arguments[1])
                length = int(arguments[2], 0)
                scale_degree = int(arguments[3], 0)
            except ValueError as error:
                raise AuditError(
                    "generated DSL/ANT bootstrap has a malformed scalar encode"
                ) from error
            if (
                not math.isfinite(scalar)
                or length <= 0
                or scale_degree <= 0
            ):
                raise AuditError(
                    "generated DSL/ANT bootstrap scalar encode is invalid"
                )
            scalar_index = len(scalar_encodings)
            scalar_encodings.append(
                (struct.pack("<d", scalar), length, scale_degree)
            )
            dependencies[output] = {
                ("scalar", scalar_index),
                ("stage", name),
            }
            continue
        input_values = []
        for argument in arguments[1:]:
            input_values.extend(
                re.findall(
                    r"&\s*([A-Za-z_]\w*(?:\[[0-9]+\])?)",
                    argument,
                )
            )
        value = {("stage", name)}
        for input_value in input_values:
            value.update(value_dependencies(input_value))
        dependencies[output] = value

    if not scalar_encodings:
        raise AuditError(
            "generated DSL/ANT bootstrap body has no EvalMod scalar encodes"
        )
    returned_dependencies = value_dependencies(returned)
    expected_constants = {
        entry["entry_id"] for entry in manifest["constants"]
    }
    reachable_constants = {
        value for kind, value in returned_dependencies if kind == "constant"
    }
    if reachable_constants != expected_constants:
        missing = sorted(expected_constants - reachable_constants)
        extra = sorted(reachable_constants - expected_constants)
        raise AuditError(
            "generated DSL/ANT bootstrap return does not depend on every "
            f"transform constant: missing={missing}, extra={extra}"
        )
    expected_scalars = set(range(len(scalar_encodings)))
    reachable_scalars = {
        value for kind, value in returned_dependencies if kind == "scalar"
    }
    if reachable_scalars != expected_scalars:
        raise AuditError(
            "generated DSL/ANT bootstrap return does not depend on every "
            "EvalMod scalar encode"
        )
    reachable_stages = {
        value for kind, value in returned_dependencies if kind == "stage"
    }
    required_stages = set(REQUIRED_SOURCE_CALLS)
    if not required_stages <= reachable_stages:
        raise AuditError(
            "generated DSL/ANT bootstrap return does not depend on every "
            "required stage: "
            + ", ".join(sorted(required_stages - reachable_stages))
        )
    if ("parameter", parameters[0]) not in returned_dependencies:
        raise AuditError(
            "generated DSL/ANT bootstrap return does not depend on its input"
        )
    scalar_payload = b"".join(
        value + struct.pack("<QQ", length, scale_degree)
        for value, length, scale_degree in scalar_encodings
    )
    return {
        "returned_dependency_count": len(returned_dependencies),
        "transform_constant_count": len(reachable_constants),
        "reachable_required_stage_count": len(required_stages),
        "scalar_encode_count": len(scalar_encodings),
        "scalar_encode_payload_sha256": _sha256_bytes(scalar_payload),
    }


def _air_constant_load_sequence(air: str) -> list[int]:
    return [
        int(value, 16)
        for value in re.findall(r"\bldc\s+CST\[0x([0-9a-f]+)\]", air)
    ]


SCALAR_CONSTANT_DEFINITION = re.compile(
    r"^\s*float64_t\s+([A-Za-z_]\w*)\s*=\s*([^;]+);\s*$",
    re.MULTILINE,
)


def _bind_expanded_constants_to_source(
    source: str, values: list[float], label: str
) -> dict[str, Any]:
    emitted: list[tuple[str, float]] = []
    for name, literal in SCALAR_CONSTANT_DEFINITION.findall(source):
        try:
            value = float(literal)
        except ValueError as error:
            raise AuditError(f"{label} has an invalid scalar constant") from error
        if not math.isfinite(value):
            raise AuditError(f"{label} has a nonfinite scalar constant")
        emitted.append((name, value))
    expected_bits = b"".join(struct.pack("<d", value) for value in values)
    matches = []
    for start in range(len(emitted) - len(values) + 1):
        candidate_bits = b"".join(
            struct.pack("<d", value)
            for _, value in emitted[start : start + len(values)]
        )
        if candidate_bits == expected_bits:
            matches.append(start)
    if len(matches) != 1:
        raise AuditError(
            f"{label} does not contain one exact expanded EvalMod constant sequence"
        )
    start = matches[0]
    symbols = [name for name, _ in emitted[start : start + len(values)]]
    return {
        "constant_count": len(symbols),
        "first_symbol": symbols[0],
        "last_symbol": symbols[-1],
        "payload_sha256": _sha256_bytes(expected_bits),
    }


def verify_v4_closure(
    *,
    context_path: Path,
    resource_path: Path,
    constant_path: Path,
    raw_air_path: Path,
    post_ckks_air_path: Path,
    source_path: Path,
    ant_source_path: Path,
    generation_path: Path,
    invocation_path: Path,
    semantics_path: Path,
    post_operations_air_path: Path,
    post_operation_attestation_path: Path,
) -> dict[str, Any]:
    context = _read_json(context_path)
    resource = _read_json(resource_path)
    generation = _read_json(generation_path)
    invocation = _read_json(invocation_path)
    semantics = _read_json(semantics_path)
    post_record = _read_json(post_operation_attestation_path)
    post_air = post_ckks_air_path.read_text(encoding="utf-8")
    operations_air = post_operations_air_path.read_text(encoding="utf-8")
    phantom_source = source_path.read_text(encoding="utf-8")
    ant_source = ant_source_path.read_text(encoding="utf-8")

    if generation.get("schema_version") != "ace.phantom.bootstrap-generation/4.0.0":
        raise AuditError("generation record is not schema version 4")
    if generation.get("status") != "pass":
        raise AuditError("generation record did not pass")
    _verify_bootstrap_abi(phantom_source, "Phantom source")
    _verify_bootstrap_abi(ant_source, "generated DSL/ANT source")
    ant_inspection = inspect_source(ant_source)
    if ant_inspection["forbidden_matches"]:
        raise AuditError("generated DSL/ANT source contains forbidden bootstrap calls")

    _verify_file_record(generation.get("source"), source_path, "generation source")
    sources = generation.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "phantom", "generated_dsl_ant"
    }:
        raise AuditError("generation sources keys differ from schema")
    _verify_file_record(sources["phantom"], source_path, "Phantom source record")
    _verify_file_record(
        sources["generated_dsl_ant"], ant_source_path, "generated DSL/ANT source record"
    )
    _verify_file_record(
        generation.get("normalized_compiler_invocation"),
        invocation_path,
        "normalized compiler invocation record",
    )
    _verify_file_record(
        generation.get("bootstrap_semantics"), semantics_path, "bootstrap semantics record"
    )
    _verify_file_record(
        generation.get("post_operations_air"),
        post_operations_air_path,
        "post-operations AIR record",
    )
    _verify_file_record(
        generation.get("post_operations_attestation"),
        post_operation_attestation_path,
        "post-operation attestation record",
    )
    air = generation.get("air")
    if not isinstance(air, dict) or set(air) != {"raw", "post_ckks"}:
        raise AuditError("generation AIR keys differ from schema")
    _verify_file_record(air["raw"], raw_air_path, "raw AIR record")
    _verify_file_record(air["post_ckks"], post_ckks_air_path, "post-CKKS AIR record")
    post_hash = _sha256_bytes(post_ckks_air_path.read_bytes())
    paths = generation.get("terminal_paths")
    if not isinstance(paths, dict) or set(paths) != {"phantom", "generated_dsl_ant"}:
        raise AuditError("terminal path keys differ from schema")
    expected_paths = {
        "phantom": ("phantom", "ckks", ["ckks_driver", "ckks2c"]),
        "generated_dsl_ant": (
            "ant", "poly", ["ckks_driver", "poly_driver", "poly2c"]
        ),
    }
    for name, (provider, codegen_ir, stages) in expected_paths.items():
        if paths[name] != {
            "provider": provider,
            "codegen_ir": codegen_ir,
            "stages_completed": stages,
            "post_ckks_air_sha256": post_hash,
        }:
            raise AuditError(f"{name} terminal path binding differs")

    _require_exact_keys(
        invocation,
        {
            "schema_version", "status", "tool", "normalized_argv",
            "normalized_argv_sha256", "options", "output_destination_in_identity",
        },
        "compiler invocation",
    )
    if (
        invocation["schema_version"]
        != "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0"
        or invocation["status"] != "pass"
        or invocation["output_destination_in_identity"] is not False
        or not isinstance(invocation["normalized_argv"], list)
        or invocation["normalized_argv_sha256"]
        != _sha256_bytes(
            json.dumps(
                invocation["normalized_argv"], separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        )
    ):
        raise AuditError("normalized compiler invocation is invalid")
    options = invocation["options"]
    if not isinstance(options, dict):
        raise AuditError("compiler invocation options must be an object")
    _require_exact_keys(options, INVOCATION_OPTION_KEYS, "compiler invocation options")
    expected_argv = [
        invocation["tool"],
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
        "--ciphertext-constant-encoding",
        str(options["ciphertext_constant_encoding"]),
        "--packing", str(options["packing"]),
        "--post-multiply-real", repr(options["post_multiply_real"]),
        "--post-multiply-imag", repr(options["post_multiply_imag"]),
        "--post-multiply-scale-degree",
        str(options["post_multiply_scale_degree"]),
        "--post-rotation-step", str(options["post_rotation_step"]),
    ]
    if invocation["normalized_argv"] != expected_argv:
        raise AuditError("normalized compiler argv differs from typed options")
    expected_context_options = {
        "poly_degree": context["polynomial_degree"],
        "vector_capacity": context["logical_slot_capacity"],
        "mul_level": len(context["data_q_bit_sizes"]),
        "input_level": context["input_level"],
        "security_level": context["security_level"],
        "scaling_factor_bits": context["scaling_modulus_bits"],
        "first_prime_bits": context["first_modulus_bits"],
        "hamming_weight": context["hamming_weight"],
        "q_part_count": context["q_part_count"],
        "packing": context["packing"],
    }
    for name, expected in expected_context_options.items():
        if options.get(name) != expected:
            raise AuditError(f"compiler invocation option {name} differs from context")
    requested_rotation = options["post_rotation_step"]
    logical_slots = context["logical_slot_capacity"]
    normalized_rotation = requested_rotation % logical_slots
    if normalized_rotation > logical_slots // 2:
        normalized_rotation -= logical_slots
    if requested_rotation != normalized_rotation:
        raise AuditError("post-operation rotation step is not canonical")
    if normalized_rotation not in resource.get("rotation_steps", []):
        raise AuditError(
            "post-operation rotation step is absent from resource manifest"
        )

    if semantics.get("schema_version") != "ace.phantom.generated-bootstrap.semantics/2.0.0" or semantics.get("status") != "pass":
        raise AuditError("bootstrap semantics record is invalid")
    bindings = semantics.get("bindings")
    expected_bindings = {
        "compiler_invocation_sha256": _sha256_bytes(invocation_path.read_bytes()),
        "raw_air_sha256": _sha256_bytes(raw_air_path.read_bytes()),
        "post_ckks_air_sha256": post_hash,
        "post_operations_air_sha256": _sha256_bytes(post_operations_air_path.read_bytes()),
        "post_operations_attestation_sha256": _sha256_bytes(post_operation_attestation_path.read_bytes()),
        "context_manifest_sha256": _sha256_bytes(context_path.read_bytes()),
        "resource_manifest_sha256": _sha256_bytes(resource_path.read_bytes()),
        "constant_manifest_sha256": _sha256_bytes(constant_path.read_bytes()),
        "phantom_source_sha256": _sha256_bytes(source_path.read_bytes()),
        "generated_dsl_ant_source_sha256": _sha256_bytes(ant_source_path.read_bytes()),
    }
    if bindings != expected_bindings:
        raise AuditError("bootstrap semantics bindings differ from artifacts")
    identity_policy = generation.get("identity_domain_policy")
    if not isinstance(identity_policy, dict) or set(identity_policy) != {
        "provider_clear_maximum_absolute",
        "clear_map_budget_fraction",
    }:
        raise AuditError("generation identity-domain policy is invalid")
    provider_clear_threshold = identity_policy[
        "provider_clear_maximum_absolute"
    ]
    if (
        isinstance(provider_clear_threshold, bool)
        or not isinstance(provider_clear_threshold, (int, float))
        or not math.isfinite(provider_clear_threshold)
        or provider_clear_threshold <= 0.0
        or identity_policy["clear_map_budget_fraction"]
        != CLEAR_MAP_BUDGET_FRACTION
    ):
        raise AuditError("generation identity-domain policy is unsupported")
    domain_bindings = {
        key: value
        for key, value in expected_bindings.items()
        if key
        not in {
            "post_operations_air_sha256",
            "post_operations_attestation_sha256",
        }
    }
    domain = semantics.get("supported_identity_domain")
    identity_attestation = semantics.get("identity_domain_attestation")
    constant_manifest = _read_json(constant_path)
    try:
        trace_config = build_bootstrap_trace_config(
            poly_degree=options["poly_degree"],
            mul_level=options["mul_level"],
            first_prime_bits=options["first_prime_bits"],
            scaling_factor_bits=options["scaling_factor_bits"],
            hamming_weight=options["hamming_weight"],
            q_parts=options["q_part_count"],
            enc_budget=options["encode_transform_budget"],
            dec_budget=options["decode_transform_budget"],
            ct_encode=(
                options["ciphertext_constant_encoding"] == "enabled"
            ),
        )
        expected_transform_payload_manifest = (
            build_bootstrap_transform_payload_manifest(
                trace_config
            )
        )
        embedded_transform_payload_manifest = identity_attestation[
            "normalization"
        ]["compiler_transform_payload_semantics"]
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(
            "identity-domain transform payload semantics are invalid"
        ) from error
    if embedded_transform_payload_manifest != expected_transform_payload_manifest:
        raise AuditError(
            "identity-domain transform payload semantics differ from the "
            "normalized compiler invocation"
        )
    phantom_body, phantom_parameters = _bootstrap_function_body(
        phantom_source, "Phantom source"
    )
    ant_body, ant_parameters = _bootstrap_function_body(
        ant_source, "generated DSL/ANT source"
    )
    expected_operation_trace = _post_ckks_operation_trace(post_air)
    phantom_body_closure = _verify_phantom_bootstrap_dataflow(
        phantom_body,
        phantom_parameters,
        constant_manifest,
        expected_operation_trace,
    )
    ant_body_closure = _verify_ant_bootstrap_payload_uses(
        ant_body,
        ant_parameters,
        constant_manifest,
        trace_config.num_p,
        expected_operation_trace,
    )
    if (
        phantom_body_closure["scalar_encode_count"]
        != ant_body_closure["scalar_encode_count"]
        or phantom_body_closure["scalar_encode_payload_sha256"]
        != ant_body_closure["scalar_encode_payload_sha256"]
    ):
        raise AuditError(
            "terminal bootstrap bodies differ in EvalMod scalar encodings"
        )
    expression = identity_attestation["expanded_clear_component_map"]
    expected_scalar_manifest = build_bootstrap_evalmod_scalar_manifest(
        trace_config
    )
    embedded_scalar_manifest = semantics.get("expanded_bootstrap", {}).get(
        "evalmod_scalar_encodings"
    )
    if embedded_scalar_manifest != expected_scalar_manifest:
        raise AuditError(
            "EvalMod scalar encoding semantics differ from the normalized "
            "compiler invocation"
        )
    expected_scalar_program = expected_scalar_manifest["full_program"]
    if (
        phantom_body_closure["scalar_encode_count"]
        != expected_scalar_program["count"]
        or phantom_body_closure["scalar_encode_payload_sha256"]
        != expected_scalar_program["ordered_payload_sha256"]
    ):
        raise AuditError(
            "terminal bootstrap EvalMod scalar encodings differ from the "
            "attested polynomial"
        )
    try:
        validate_supported_identity_domain(
            domain,
            identity_attestation,
            provider_clear_threshold=float(provider_clear_threshold),
            artifact_bindings=domain_bindings,
            constant_manifest=constant_manifest,
            evalmod_scalar_manifest=expected_scalar_manifest,
            raw_air=raw_air_path.read_text(encoding="utf-8"),
        )
    except ValueError as error:
        raise AuditError(str(error)) from error
    try:
        post_transform = validate_transform_air_semantics(
            polynomial_degree=context["polynomial_degree"],
            logical_slots=context["logical_slot_capacity"],
            overflow_bound=expression["overflow_bound"],
            restoration_factor=expression["restoration_factor"],
            transform_payload_manifest=expected_transform_payload_manifest,
            constant_manifest=constant_manifest,
            air=post_air,
            air_label="post-CKKS AIR",
        )
        post_evalmod_polynomial = validate_evalmod_air_polynomial(
            air=post_air,
            air_label="post-CKKS AIR",
            scalar_manifest=expected_scalar_manifest,
        )
    except ValueError as error:
        raise AuditError(str(error)) from error
    if post_evalmod_polynomial != expression.get("emitted_evalmod_polynomial"):
        raise AuditError(
            "post-CKKS AIR EvalMod polynomial differs from the raw-AIR "
            "identity-domain attestation"
        )
    expected_constant_ids = [
        entry["constant_id"] for entry in constant_manifest["constants"]
    ]
    if _air_constant_load_sequence(raw_air_path.read_text(encoding="utf-8")) != expected_constant_ids:
        raise AuditError(
            "raw AIR transform constant load order differs from the compiler manifest"
        )
    if _air_constant_load_sequence(post_air) != expected_constant_ids:
        raise AuditError(
            "post-CKKS AIR transform constant load order differs from raw AIR"
        )
    normalization = identity_attestation["normalization"]
    normalization_context = normalization["compiler_context"]
    configured_factor = normalization["coefficients_to_slots"][
        "configured_factor"
    ]
    emitted_factor = normalization["coefficients_to_slots"][
        "emitted_stage_scale_product"
    ]
    expected_configured_factor = (
        1.0
        / float(context["polynomial_degree"])
        / float(expression["overflow_bound"])
        / float(expression["restoration_factor"])
    )
    expected_nominal_gain = (
        float(normalization["nominal_transform_gain"]["combined_gain"])
        * float(emitted_factor)
    )
    if normalization_context != {
        "polynomial_degree": context["polynomial_degree"],
        "logical_slots": context["logical_slot_capacity"],
        "full_packing_relation": "polynomial_degree=2*logical_slots",
    } or (
        configured_factor != expected_configured_factor
        or normalization["coefficients_to_slots"]["configured_factor_hex"]
        != float(configured_factor).hex()
        or normalization["coefficients_to_slots"][
            "emitted_stage_scale_product_hex"
        ]
        != float(emitted_factor).hex()
        or normalization["nominal_component_input_gain"]
        != expected_nominal_gain
        or float.fromhex(normalization["nominal_component_input_gain_hex"])
        != expected_nominal_gain
    ):
        raise AuditError(
            "identity-domain normalization differs from compiler context"
        )
    coefficient_hex = expression["chebyshev"]["coefficients_binary64_hex"]
    scalar_hex = expression["double_angle"]["scalars_binary64_hex"]
    coefficients = [float.fromhex(value) for value in coefficient_hex]
    scalars = [float.fromhex(value) for value in scalar_hex]
    expanded = semantics.get("expanded_bootstrap")
    if not isinstance(expanded, dict) or (
        expanded.get("packing") != context["packing"]
        or expanded.get("logical_slot_capacity")
        != context["logical_slot_capacity"]
        or expanded.get("transform_budgets")
        != {
            "encode": options["encode_transform_budget"],
            "decode": options["decode_transform_budget"],
        }
        or expanded.get("ciphertext_constant_encoding")
        != options["ciphertext_constant_encoding"]
        or expanded.get("coefficient_family", {}).get("coefficient_count")
        != len(coefficients)
        or expanded.get("coefficient_family", {}).get(
            "coefficient_payload_sha256"
        )
        != expression["chebyshev"]["coefficient_payload_sha256"]
        or expanded.get("double_angle", {}).get("count") != len(scalars)
        or expanded.get("double_angle", {}).get("scalar_payload_sha256")
        != expression["double_angle"]["scalar_payload_sha256"]
        or expanded.get("evalmod_scalar_encodings")
        != expected_scalar_manifest
        or expanded.get("eval_sin_upper_bound_k")
        != expression["overflow_bound"]
        or expanded.get("post_scale", {}).get("factor")
        != expression["restoration_factor"]
        or expanded.get("coeffs_to_slots_factor")
        != identity_attestation["normalization"]["coefficients_to_slots"][
            "configured_factor"
        ]
        or expanded.get("coefficient_family", {}).get(
            "evalmod_component_interval"
        )
        != {
            "lower": expression["chebyshev"]["coordinate_interval"][0],
            "upper": expression["chebyshev"]["coordinate_interval"][1],
            "lower_inclusive": True,
            "upper_inclusive": True,
        }
    ):
        raise AuditError("identity-domain constants differ from expanded semantics")
    emitted_values = coefficients + scalars + [expression["restoration_factor"]]
    terminal_constant_bindings = {
        "phantom": _bind_expanded_constants_to_source(
            phantom_source, emitted_values, "Phantom source"
        ),
        "generated_dsl_ant": _bind_expanded_constants_to_source(
            ant_source, emitted_values, "generated DSL/ANT source"
        ),
    }
    terminal = _terminal_attributes(post_air)
    contract = semantics.get("output_air_contract")
    raw_scale_contract = (
        contract.get("raw_scale_contract") if isinstance(contract, dict) else None
    )
    expected_nominal_scale = math.ldexp(
        1,
        terminal["scale"] * context["scaling_modulus_bits"],
    ).hex()
    if not isinstance(contract, dict) or (
        contract.get("ace_logical_level") != terminal["level"]
        or contract.get("active_q_count") != terminal["level"]
        or contract.get("rescale_level") != terminal["rescale_level"]
        or contract.get("scale_degree") != terminal["scale"]
        or contract.get("logical_slots") != context["logical_slot_capacity"]
        or raw_scale_contract
        != {
            "kind": "ace-log2-scale-coordinate",
            "nominal_raw_scale": expected_nominal_scale,
            "scaling_modulus_bits": context["scaling_modulus_bits"],
            "expected_scale_degree": terminal["scale"],
            "maximum_absolute_coordinate_error": (
                RAW_SCALE_COORDINATE_TOLERANCE
            ),
        }
    ):
        raise AuditError("bootstrap output contract differs from final return AIR metadata")

    if (
        post_record.get("schema_version")
        != "ace.phantom.bootstrap-post-operation-semantics/1.0.0"
        or post_record.get("status") != "attested"
    ):
        raise AuditError("post-operation attestation is invalid")
    expected_post_bindings = {
        "compiler_invocation_sha256": _sha256_bytes(invocation_path.read_bytes()),
        "post_ckks_air_sha256": post_hash,
        "context_manifest_sha256": _sha256_bytes(context_path.read_bytes()),
        "resource_manifest_sha256": _sha256_bytes(resource_path.read_bytes()),
        "post_operations_air_sha256": _sha256_bytes(post_operations_air_path.read_bytes()),
    }
    if post_record.get("bindings") != expected_post_bindings:
        raise AuditError("post-operation bindings differ from artifacts")
    expected_input_coordinate = {
        "ace_logical_level": terminal["level"],
        "rescale_level": terminal["rescale_level"],
        "scale_degree": terminal["scale"],
    }
    if post_record.get("input_coordinate") != expected_input_coordinate:
        raise AuditError("post-operation input coordinate differs from bootstrap return")
    expected_post_inputs = {
        "multiply_constant": {
            "real": options["post_multiply_real"],
            "imaginary": options["post_multiply_imag"],
            "plaintext_scale_degree": options["post_multiply_scale_degree"],
        },
        "rotation_step": options["post_rotation_step"],
    }
    if post_record.get("inputs") != expected_post_inputs:
        raise AuditError("post-operation inputs differ from normalized compiler invocation")
    expected_contracts = {
        name: post_record[name]
        for name in (
            "input_coordinate",
            "rotation",
            "ciphertext_plaintext_multiply",
        )
    }
    expected_contracts["status"] = "pass"
    expected_contracts["air_sha256"] = expected_post_bindings[
        "post_operations_air_sha256"
    ]
    if semantics.get("post_operation_contracts") != expected_contracts:
        raise AuditError(
            "bootstrap semantics and post-operation attestation contracts differ"
        )
    rotations = _operation_attributes(operations_air, "rotate")
    multiplies = _operation_attributes(operations_air, "mul")
    if rotations != [terminal] or multiplies != [terminal]:
        raise AuditError("dedicated post-operation AIR does not preserve final metadata")
    if (
        post_record.get("rotation", {}).get("air_attributes") != terminal
        or post_record.get("ciphertext_plaintext_multiply", {}).get("air_attributes")
        != terminal
        or post_record.get("inputs", {}).get("multiply_constant", {}).get(
            "plaintext_scale_degree"
        ) != 0
    ):
        raise AuditError("post-operation transition attestation differs from AIR")
    required_transition = {
        "ace_logical_level_delta": 0,
        "active_q_count_delta": 0,
        "phantom_chain_index_delta": 0,
        "scale_degree_delta": 0,
        "raw_scale_multiplier": 1.0,
        "logical_slots": "preserved",
        "ciphertext_size": "preserved",
        "ntt_state": "preserved",
    }
    for name in ("rotation", "ciphertext_plaintext_multiply"):
        if post_record[name].get("transition") != required_transition:
            raise AuditError(f"{name} transition is not exact metadata preservation")
    return {
        "canonical_post_ckks_air_sha256": post_hash,
        "output_air_contract": terminal,
        "post_operations_attested": True,
        "identity_domain_attested": True,
        "post_ckks_transform_roles_attested": True,
        "post_ckks_evalmod_polynomial_attested": True,
        "post_ckks_transform_stage_count": len(
            post_transform["transform_stage_dataflow"]
        ),
        "terminal_constant_bindings": terminal_constant_bindings,
        "terminal_body_closure": {
            "phantom": phantom_body_closure,
            "generated_dsl_ant": ant_body_closure,
        },
    }


def audit(
    context_path: Path,
    resource_path: Path,
    constant_path: Path,
    raw_air_path: Path,
    post_ckks_air_path: Path,
    source_path: Path,
    ant_source_path: Path | None = None,
    generation_path: Path | None = None,
    invocation_path: Path | None = None,
    semantics_path: Path | None = None,
    post_operations_air_path: Path | None = None,
    post_operation_attestation_path: Path | None = None,
) -> dict[str, Any]:
    input_paths = {
        "context_manifest": context_path,
        "resource_manifest": resource_path,
        "constant_manifest": constant_path,
        "raw_air": raw_air_path,
        "post_ckks_air": post_ckks_air_path,
        "source": source_path,
    }
    v3_paths = {
        "ant_source": ant_source_path,
        "generation_record": generation_path,
        "compiler_invocation": invocation_path,
        "bootstrap_semantics": semantics_path,
        "post_operations_air": post_operations_air_path,
        "post_operation_attestation": post_operation_attestation_path,
    }
    missing_v3 = []
    if any(path is not None for path in v3_paths.values()):
        missing_v3 = sorted(name for name, path in v3_paths.items() if path is None)
        input_paths.update({name: path for name, path in v3_paths.items() if path is not None})
    v3_enabled = all(path is not None for path in v3_paths.values())
    report: dict[str, Any] = {
        "schema_version": V4_SCHEMA if v3_enabled else SCHEMA,
        "status": "fail",
        "inputs": {},
        "errors": [],
        "air": {
            "raw": {"required_opcode_counts": {}, "forbidden_matches": []},
            "post_ckks": {
                "required_opcode_counts": {},
                "forbidden_matches": [],
            },
        },
        "source": {"required_call_counts": {}, "forbidden_matches": []},
        "forbidden_matches": {
            "raw_air": [],
            "post_ckks_air": [],
            "source": [],
        },
        "forbidden_native_bts_matches": [],
    }
    try:
        if missing_v3:
            raise AuditError(
                "version 3 audit inputs are incomplete: " + ", ".join(missing_v3)
            )
        raw = {name: path.read_bytes() for name, path in input_paths.items()}
        report["inputs"] = {
            name: {
                "path": input_paths[name].name,
                "sha256": _sha256_bytes(value),
                "size_bytes": len(value),
            }
            for name, value in raw.items()
        }
        context = _read_json(context_path)
        resource = _read_json(resource_path)
        constants = _read_json(constant_path)
        raw_air = raw["raw_air"].decode("utf-8")
        post_ckks_air = raw["post_ckks_air"].decode("utf-8")
        source = raw["source"].decode("utf-8")
        report["air"]["raw"] = inspect_air(raw_air)
        report["air"]["post_ckks"] = inspect_air(post_ckks_air)
        report["source"] = inspect_source(source)
        report["forbidden_matches"] = {
            "raw_air": report["air"]["raw"]["forbidden_matches"],
            "post_ckks_air": report["air"]["post_ckks"]["forbidden_matches"],
            "source": report["source"]["forbidden_matches"],
        }
        report["forbidden_native_bts_matches"] = report["source"][
            "forbidden_matches"
        ]
        for air_name in ("raw", "post_ckks"):
            missing = [
                opcode
                for opcode, count in report["air"][air_name][
                    "required_opcode_counts"
                ].items()
                if count == 0
            ]
            if missing:
                raise AuditError(
                    f"{air_name} AIR is missing required CKKS opcodes: "
                    + ", ".join(missing)
                )
            forbidden_air = report["air"][air_name]["forbidden_matches"]
            if forbidden_air:
                raise AuditError(
                    f"{air_name} AIR contains forbidden opcodes: "
                    + ", ".join(forbidden_air)
                )
        missing_calls = [
            call
            for call, count in report["source"]["required_call_counts"].items()
            if count == 0
        ]
        if missing_calls:
            raise AuditError(
                "generated source is missing required primitive calls: "
                + ", ".join(missing_calls)
            )
        forbidden = report["source"]["forbidden_matches"]
        if forbidden:
            raise AuditError(
                "generated source contains forbidden bootstrap calls or helpers: "
                + ", ".join(forbidden)
            )
        verify_context_manifest(context)
        verify_resource_manifest(resource, context)
        verify_constant_manifest(
            constants, context, report["inputs"]["context_manifest"]["sha256"]
        )
        emitted_context, context_errors = extract_context(source)
        if context_errors:
            raise AuditError("; ".join(context_errors))
        context_mismatches = compare_context(source, context)
        if context_mismatches:
            raise AuditError("; ".join(context_mismatches))
        if emitted_context != context:
            raise AuditError("generated context differs from context manifest")
        compare_resource_source(source, resource)
        compare_constant_source(source, constants)
        if v3_enabled:
            compare_constant_payload_source(
                ant_source_path.read_text(encoding="utf-8"),
                constants,
                "generated DSL/ANT source",
            )
        for accessor in (
            "Get_phantom_context_manifest()",
            "Get_phantom_resource_manifest()",
            "Get_phantom_constant_manifest()",
        ):
            if accessor not in source:
                raise AuditError(f"required generated accessor is absent: {accessor}")
        if v3_enabled:
            report["qualification_closure"] = verify_v4_closure(
                context_path=context_path,
                resource_path=resource_path,
                constant_path=constant_path,
                raw_air_path=raw_air_path,
                post_ckks_air_path=post_ckks_air_path,
                source_path=source_path,
                ant_source_path=ant_source_path,
                generation_path=generation_path,
                invocation_path=invocation_path,
                semantics_path=semantics_path,
                post_operations_air_path=post_operations_air_path,
                post_operation_attestation_path=post_operation_attestation_path,
            )
        report["counts"] = {
            "constants": len(constants["constants"]),
            "monomial_powers": len(resource["monomial_powers"]),
            "rotation_batches": len(resource["rotation_batches"]),
            "rotation_steps": len(resource["rotation_steps"]),
        }
        report["status"] = "pass"
    except (AuditError, OSError, UnicodeDecodeError) as error:
        report["errors"].append(str(error))
    return report


def main() -> int:
    arguments = parse_arguments()
    report = audit(
        arguments.context_manifest,
        arguments.resource_manifest,
        arguments.constant_manifest,
        arguments.raw_air,
        arguments.post_ckks_air,
        arguments.source,
        arguments.ant_source,
        arguments.generation_record,
        arguments.compiler_invocation,
        arguments.bootstrap_semantics,
        arguments.post_operations_air,
        arguments.post_operation_attestation,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
