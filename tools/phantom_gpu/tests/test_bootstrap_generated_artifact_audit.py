from __future__ import annotations

import hashlib
import json
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

from check_bootstrap_generated_artifacts import audit, cache_key_sha256  # noqa: E402


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
