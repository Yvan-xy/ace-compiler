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
from typing import Any

from check_primitive_codegen import compare_context, extract_context


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
C_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
HEXFLOAT_PATTERN = re.compile(r"^0x[0-9a-f]+\.[0-9a-f]+p[+-][0-9]+$")

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
    parser.add_argument("--source", required=True, type=Path)
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


def audit(
    context_path: Path,
    resource_path: Path,
    constant_path: Path,
    source_path: Path,
) -> dict[str, Any]:
    input_paths = {
        "context_manifest": context_path,
        "resource_manifest": resource_path,
        "constant_manifest": constant_path,
        "source": source_path,
    }
    report: dict[str, Any] = {
        "status": "fail",
        "inputs": {},
        "errors": [],
        "forbidden_native_bts_matches": [],
    }
    try:
        raw = {name: path.read_bytes() for name, path in input_paths.items()}
        report["inputs"] = {
            name: {
                "path": str(input_paths[name]),
                "sha256": _sha256_bytes(value),
                "size_bytes": len(value),
            }
            for name, value in raw.items()
        }
        context = _read_json(context_path)
        resource = _read_json(resource_path)
        constants = _read_json(constant_path)
        source = raw["source"].decode("utf-8")
        forbidden = [
            label for label, pattern in FORBIDDEN_NATIVE_BTS.items() if pattern.search(source)
        ]
        report["forbidden_native_bts_matches"] = forbidden
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
        for accessor in (
            "Get_phantom_context_manifest()",
            "Get_phantom_resource_manifest()",
            "Get_phantom_constant_manifest()",
        ):
            if accessor not in source:
                raise AuditError(f"required generated accessor is absent: {accessor}")
        if forbidden:
            raise AuditError("generated source contains forbidden native BTS symbols")
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
        arguments.source,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
