#!/usr/bin/env python3
"""Compare stable local and RunPod pipeline evidence fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "ace.phantom.environment-comparison/2.0.0"
CORRECTNESS_REPORT_SCHEMA = (
    "ace.phantom.generated-bootstrap-correctness-environment-comparison/1.0.0"
)
RETAINED_REPLAY_SCHEMA = "ace.phantom.retained_ckks.local-replay/1.0.0"
RETAINED_HOST_SCHEMA = "ace.phantom.retained_ckks.host-qualification/1.0.0"
RETAINED_BUILD_SCHEMA = "ace.phantom.retained_ckks.build-attestation/1.0.0"
RETAINED_ARTIFACT_SCHEMA = "ace.phantom.retained_ckks.artifact-manifest/1.0.0"
RETAINED_SUMMARY_SCHEMA = "ace.phantom.retained_ckks.ant-semantic-summary/1.0.0"
SEMANTIC_REPLAY_MODE = "semantic-summary-only-no-decoded-byte-comparison"
BOOTSTRAP_ARTIFACT_SCHEMA = "ace.phantom.bootstrap-artifacts/2.0.0"
BOOTSTRAP_GENERATION_SCHEMA = "ace.phantom.bootstrap-generation/2.0.0"
BOOTSTRAP_GENERATED_ARTIFACT_AUDIT_SCHEMA = (
    "ace.phantom.bootstrap-generated-artifact-audit/2.0.0"
)
BOOTSTRAP_FROZEN_REFERENCE_SCHEMA = (
    "ace.phantom.bootstrap-frozen-reference/2.0.0"
)
BOOTSTRAP_QUALIFICATION_INVOCATION_SCHEMA = (
    "ace.phantom.qualification-invocation/1.0.0"
)
BOOTSTRAP_GENERATION_INVOCATION_SCHEMA = (
    "ace.phantom.bootstrap-qualification-invocation/1.0.0"
)
BOOTSTRAP_SYMBOL_CLOSURE_SCHEMA = (
    "ace.phantom.bootstrap-symbol-closure/2.0.0"
)
BOOTSTRAP_IO_HELPER_CLOSURE_SCHEMA = (
    "ace.phantom.bootstrap-io-helper-closure/1.0.0"
)
ARCHIVE_MEMBER_AUDIT_SCHEMA = (
    "ace.phantom.production-archive-members/1.0.0"
)
BOOTSTRAP_REQUIRED_SYMBOLS = (
    "bootstrap_full",
    "Conjugate_ciph",
    "Rotate_batch_ciph",
    "Raise_mod",
    "Mul_mono_ciph",
    "Phantom_conjugate",
    "Phantom_rotate_batch",
    "Phantom_raise_mod",
    "Phantom_mul_mono",
    "complex_conjugate_inplace",
    "rotate_batch",
    "raise_modulus",
    "multiply_by_monomial",
    "Get_phantom_context_manifest",
    "Get_phantom_resource_manifest",
    "Get_phantom_constant_manifest",
    "Load_cached_plain",
    "main",
)
BOOTSTRAP_IO_HELPERS = (
    "Get_input_count",
    "Get_output_count",
    "Get_encode_scheme",
    "Get_decode_scheme",
)
FORBIDDEN_BOOTSTRAP_SYMBOL_PATTERN = re.compile(
    r"Bootstrapper|\bBootstrap\b|Phantom_bootstrap|Eval_bootstrap|bootstrap_3|"
    r"bootstrap_(?:coeffs_to_slots|eval_mod|slots_to_coeffs)|"
    r"cnn_phantom|conv_eval|FHErt_(?:ant|poly)|fhe::(?:ant|poly)|"
    r"CoeffToSlot|SlotToCoeff|EvalMod|Native.*[Pp]recom|"
    r"[Pp]recom.*Native|Bootstrap.*[Ss]tage|[Ss]tage.*Bootstrap",
    re.IGNORECASE,
)
REPOSITORY = Path(__file__).resolve().parents[2]
CORRECTNESS_COMPARISON_ENTRYPOINT = "tools/phantom_gpu/compare_environments.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> Path:
    with tarfile.open(archive, "r:gz") as source:
        root = destination.resolve()
        for member in source.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"unsafe result member: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"unsupported result member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = source.extractfile(member)
                if payload is None:
                    raise SystemExit(f"cannot read result member: {member.name}")
                with target.open("xb") as output:
                    for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                        output.write(chunk)
    return destination / "results"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"evidence record is not an object: {path}")
    return value


def require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SystemExit(f"{label} is not a lowercase SHA-256 digest")
    return value


def require_selected_commit_entrypoint(
    repository: Path,
    selected_commit: str,
    entrypoint: Path,
    repository_relative_path: str,
) -> None:
    """Require the executing correctness comparator bytes from its source commit."""
    if re.fullmatch(r"[0-9a-f]{40}", selected_commit) is None:
        raise SystemExit("correctness comparison ACE commit is invalid")
    try:
        expected = subprocess.run(
            [
                "git", "-C", str(repository), "show",
                f"{selected_commit}:{repository_relative_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        observed = entrypoint.read_bytes()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            "cannot resolve correctness comparison entrypoint from the selected "
            "ACE commit"
        ) from error
    if observed != expected:
        raise SystemExit(
            "correctness comparison entrypoint differs from the selected ACE commit"
        )


def validate_terminal_body_closure(
    source_audit: dict[str, Any],
    semantics: dict[str, Any],
    post_ckks_air_sha256: str,
) -> None:
    try:
        constant_count = source_audit["counts"]["constants"]
        qualification_closure = source_audit["qualification_closure"]
        closure = qualification_closure["terminal_body_closure"]
        phantom = closure["phantom"]
        ant = closure["generated_dsl_ant"]
        transform_semantics = semantics["identity_domain_attestation"][
            "normalization"
        ]["compiler_transform_payload_semantics"]
        transform_stage_groups = [
            transform_semantics[direction]["stages"]
            for direction in ("coefficients_to_slots", "slots_to_coefficients")
        ]
        expected_transform_stage_count = sum(
            len(stages) for stages in transform_stage_groups
        )
    except (KeyError, TypeError) as error:
        raise SystemExit("correctness terminal-body closure is incomplete") from error
    if (
        any(not isinstance(stages, list) or not stages for stages in transform_stage_groups)
        or expected_transform_stage_count <= 0
        or qualification_closure.get("canonical_post_ckks_air_sha256")
        != post_ckks_air_sha256
        or qualification_closure.get("identity_domain_attested") is not True
        or qualification_closure.get("post_operations_attested") is not True
        or qualification_closure.get("post_ckks_transform_roles_attested") is not True
        or qualification_closure.get("post_ckks_evalmod_polynomial_attested")
        is not True
        or isinstance(
            qualification_closure.get("post_ckks_transform_stage_count"), bool
        )
        or not isinstance(
            qualification_closure.get("post_ckks_transform_stage_count"), int
        )
        or qualification_closure["post_ckks_transform_stage_count"] <= 0
        or qualification_closure["post_ckks_transform_stage_count"]
        != expected_transform_stage_count
        or set(closure) != {"phantom", "generated_dsl_ant"}
        or not isinstance(phantom, dict)
        or set(phantom)
        != {
            "reachable_value_count",
            "transform_constant_count",
            "reachable_required_stage_count",
            "scalar_encode_count",
            "scalar_encode_payload_sha256",
        }
        or not isinstance(ant, dict)
        or set(ant)
        != {
            "returned_dependency_count",
            "transform_constant_count",
            "reachable_required_stage_count",
            "scalar_encode_count",
            "scalar_encode_payload_sha256",
        }
        or isinstance(constant_count, bool)
        or not isinstance(constant_count, int)
        or constant_count <= 0
        or phantom["transform_constant_count"] != constant_count
        or ant["transform_constant_count"] != constant_count
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (
                phantom["reachable_value_count"],
                phantom["reachable_required_stage_count"],
                phantom["scalar_encode_count"],
                ant["returned_dependency_count"],
                ant["reachable_required_stage_count"],
                ant["scalar_encode_count"],
            )
        )
        or phantom["reachable_required_stage_count"] != 4
        or ant["reachable_required_stage_count"] != 4
        or phantom["scalar_encode_count"] != ant["scalar_encode_count"]
        or phantom["scalar_encode_payload_sha256"]
        != ant["scalar_encode_payload_sha256"]
    ):
        raise SystemExit("correctness terminal-body closure is invalid")
    require_digest(
        phantom["scalar_encode_payload_sha256"],
        "correctness terminal scalar payload",
    )


def require_fields(record: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for name, value in expected.items():
        if record.get(name) != value:
            raise SystemExit(f"{label} {name} violates the comparison contract")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_invocation(
    path: Path, schema: str, expected_argv: list[str] | None, label: str
) -> tuple[str, str]:
    record = read_json(path)
    if set(record) != {"schema_version", "argv", "normalized_argv_sha256"}:
        raise SystemExit(f"{label} has an invalid shape")
    if record.get("schema_version") != schema:
        raise SystemExit(f"{label} has an invalid schema")
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise SystemExit(f"{label} argv is invalid")
    if expected_argv is not None and argv != expected_argv:
        raise SystemExit(f"{label} argv violates the comparison contract")
    normalized = require_digest(
        record.get("normalized_argv_sha256"), f"{label} normalized argv"
    )
    if normalized != canonical_json_sha256(argv):
        raise SystemExit(f"{label} normalized argv hash is inconsistent")
    return sha256(path), normalized


def bootstrap_symbol_projection(
    record: dict[str, Any], inventory_path: Path
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "linked_binary_symbols_sha256",
        "required_symbols",
        "missing_symbols",
        "native_bootstrap_symbol_count",
    }
    if set(record) != expected_keys:
        raise SystemExit("bootstrap symbol closure has an invalid shape")
    inventory = inventory_path.read_text(encoding="utf-8")
    require_fields(
        record,
        {
            "schema_version": BOOTSTRAP_SYMBOL_CLOSURE_SCHEMA,
            "status": "pass",
            "linked_binary_symbols_sha256": sha256(inventory_path),
            "required_symbols": list(BOOTSTRAP_REQUIRED_SYMBOLS),
            "missing_symbols": [],
            "native_bootstrap_symbol_count": 0,
        },
        "bootstrap symbol closure",
    )
    missing = [
        symbol
        for symbol in BOOTSTRAP_REQUIRED_SYMBOLS
        if re.search(
            r"(?<![A-Za-z0-9_])"
            + re.escape(symbol)
            + r"(?![A-Za-z0-9_])",
            inventory,
        )
        is None
    ]
    forbidden_count = len(FORBIDDEN_BOOTSTRAP_SYMBOL_PATTERN.findall(inventory))
    if missing or forbidden_count != 0:
        raise SystemExit("bootstrap raw symbol inventory violates its closure record")
    return {
        "schema_version": record["schema_version"],
        "status": record["status"],
        "required_symbols": record["required_symbols"],
        "missing_symbols": record["missing_symbols"],
        "native_bootstrap_symbol_count": record[
            "native_bootstrap_symbol_count"
        ],
    }


def _helper_counts(path: Path, source: bool) -> dict[str, int]:
    value = path.read_text(encoding="utf-8")
    if source:
        return {
            helper: len(re.findall(r"\b" + re.escape(helper) + r"\s*\(", value))
            for helper in BOOTSTRAP_IO_HELPERS
        }
    lines = value.splitlines()
    return {
        helper: sum(
            re.search(r"\b" + re.escape(helper) + r"(?:\(.*\))?$", line)
            is not None
            for line in lines
        )
        for helper in BOOTSTRAP_IO_HELPERS
    }


def bootstrap_io_helper_projection(
    record: dict[str, Any],
    source_path: Path,
    harness_path: Path,
    generated_symbols_path: Path,
    harness_symbols_path: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "helpers",
        "counts",
        "generated_source_sha256",
        "harness_source_sha256",
        "generated_object_symbols_sha256",
        "harness_object_symbols_sha256",
    }
    if set(record) != expected_keys:
        raise SystemExit("bootstrap I/O-helper closure has an invalid shape")
    counts = {
        "generated_source": _helper_counts(source_path, True),
        "harness_source": _helper_counts(harness_path, True),
        "generated_object": _helper_counts(generated_symbols_path, False),
        "harness_object": _helper_counts(harness_symbols_path, False),
    }
    require_fields(
        record,
        {
            "schema_version": BOOTSTRAP_IO_HELPER_CLOSURE_SCHEMA,
            "status": "pass",
            "helpers": list(BOOTSTRAP_IO_HELPERS),
            "counts": counts,
            "generated_source_sha256": sha256(source_path),
            "harness_source_sha256": sha256(harness_path),
            "generated_object_symbols_sha256": sha256(generated_symbols_path),
            "harness_object_symbols_sha256": sha256(harness_symbols_path),
        },
        "bootstrap I/O-helper closure",
    )
    for helper in BOOTSTRAP_IO_HELPERS:
        if (
            counts["generated_source"][helper] != 0
            or counts["generated_object"][helper] != 0
            or counts["harness_source"][helper] != 1
            or counts["harness_object"][helper] != 1
        ):
            raise SystemExit("bootstrap I/O-helper ownership is not exact")
    return {
        "schema_version": record["schema_version"],
        "status": record["status"],
        "helpers": record["helpers"],
        "counts": record["counts"],
        "generated_source_sha256": record["generated_source_sha256"],
        "harness_source_sha256": record["harness_source_sha256"],
    }


def bootstrap_generated_artifact_audit_projection(
    record: dict[str, Any], expected_inputs: dict[str, Path], raw_air_path: Path,
    post_ckks_air_path: Path, source_path: Path
) -> dict[str, Any]:
    required_opcodes = (
        "ckks.conjugate",
        "ckks.rotate_batch",
        "ckks.raise_mod",
        "ckks.mul_mono",
    )
    required_calls = (
        "Conjugate_ciph",
        "Rotate_batch_ciph",
        "Raise_mod",
        "Mul_mono_ciph",
    )
    expected_keys = {
        "schema_version",
        "status",
        "inputs",
        "errors",
        "air",
        "source",
        "forbidden_matches",
        "forbidden_native_bts_matches",
        "counts",
    }
    if set(record) != expected_keys:
        raise SystemExit("bootstrap generated-artifact audit has an invalid shape")
    require_fields(
        record,
        {
            "schema_version": BOOTSTRAP_GENERATED_ARTIFACT_AUDIT_SCHEMA,
            "status": "pass",
            "errors": [],
            "forbidden_native_bts_matches": [],
        },
        "bootstrap generated-artifact audit",
    )
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
        raise SystemExit("bootstrap generated-artifact audit inputs are invalid")
    for name, expected_path in expected_inputs.items():
        entry = inputs.get(name)
        expected_entry = {
            "path": expected_path.name,
            "sha256": sha256(expected_path),
            "size_bytes": expected_path.stat().st_size,
        }
        if not isinstance(entry, dict) or entry != expected_entry:
            raise SystemExit(
                f"bootstrap generated-artifact audit {name} hash is inconsistent"
            )

    air = record.get("air")
    if not isinstance(air, dict) or set(air) != {"raw", "post_ckks"}:
        raise SystemExit("bootstrap generated-artifact AIR audit is invalid")
    air_counts: dict[str, dict[str, int]] = {}
    for stage in ("raw", "post_ckks"):
        stage_record = air.get(stage)
        if not isinstance(stage_record, dict) or set(stage_record) != {
            "required_opcode_counts",
            "forbidden_matches",
        }:
            raise SystemExit(f"bootstrap {stage} AIR audit has an invalid shape")
        counts = stage_record.get("required_opcode_counts")
        if not isinstance(counts, dict) or set(counts) != set(required_opcodes):
            raise SystemExit(f"bootstrap {stage} AIR opcode counts are invalid")
        air_text = (
            raw_air_path if stage == "raw" else post_ckks_air_path
        ).read_text(encoding="utf-8").lower()
        observed_counts = {name: air_text.count(name) for name in required_opcodes}
        if counts != observed_counts or any(
            type(counts[name]) is not int or counts[name] <= 0
            for name in required_opcodes
        ):
            raise SystemExit(f"bootstrap {stage} AIR lacks required CKKS opcodes")
        if stage_record.get("forbidden_matches") != []:
            raise SystemExit(f"bootstrap {stage} AIR contains forbidden operations")
        air_counts[stage] = counts

    source = record.get("source")
    if not isinstance(source, dict) or set(source) != {
        "required_call_counts",
        "forbidden_matches",
    }:
        raise SystemExit("bootstrap generated-source call audit has an invalid shape")
    source_counts = source.get("required_call_counts")
    if not isinstance(source_counts, dict) or set(source_counts) != set(required_calls):
        raise SystemExit("bootstrap generated-source call counts are invalid")
    source_text = source_path.read_text(encoding="utf-8")
    observed_source_counts = {
        name: len(re.findall(r"\b" + re.escape(name) + r"\s*\(", source_text))
        for name in required_calls
    }
    if source_counts != observed_source_counts or any(
        type(source_counts[name]) is not int or source_counts[name] <= 0
        for name in required_calls
    ):
        raise SystemExit("bootstrap generated source lacks required runtime calls")
    if source.get("forbidden_matches") != []:
        raise SystemExit("bootstrap generated source contains forbidden operations")

    forbidden = record.get("forbidden_matches")
    if not isinstance(forbidden, dict) or set(forbidden) != {
        "raw_air",
        "post_ckks_air",
        "source",
    }:
        raise SystemExit("bootstrap forbidden-match audit is invalid")
    if any(forbidden[name] != [] for name in forbidden):
        raise SystemExit("bootstrap generated artifacts contain forbidden operations")

    counts = record.get("counts")
    expected_count_keys = {
        "constants",
        "monomial_powers",
        "rotation_batches",
        "rotation_steps",
    }
    if not isinstance(counts, dict) or set(counts) != expected_count_keys:
        raise SystemExit("bootstrap generated-artifact summary counts are invalid")
    if any(type(counts[name]) is not int or counts[name] <= 0 for name in counts):
        raise SystemExit("bootstrap generated-artifact summary counts are empty")
    return {
        "schema_version": record["schema_version"],
        "status": record["status"],
        "air": air_counts,
        "source": source_counts,
        "forbidden_matches": forbidden,
        "counts": counts,
    }


def validate_bootstrap_generation(
    record: dict[str, Any],
    source_path: Path,
    raw_air_path: Path,
    post_ckks_air_path: Path,
    manifest_paths: dict[str, Path],
    resource: dict[str, Any],
    constant_count: int,
    compiler_parameters: dict[str, Any],
    bootstrap_parameters: dict[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "qualification_scope",
        "compiler_parameters",
        "bootstrap_parameters",
        "stages_completed",
        "source",
        "air",
        "manifests",
        "constant_count",
        "rotation_count",
        "rotation_batch_count",
        "monomial_count",
        "native_bootstrap_precompute",
        "generated_program_executed",
    }
    if set(record) != expected_keys:
        raise SystemExit("bootstrap generation record has an invalid shape")
    require_fields(
        record,
        {
            "schema_version": BOOTSTRAP_GENERATION_SCHEMA,
            "status": "pass",
            "qualification_scope": "full-generated-bootstrap-compile-only",
            "compiler_parameters": compiler_parameters,
            "bootstrap_parameters": bootstrap_parameters,
            "stages_completed": ["ckks_driver", "ckks2c"],
            "constant_count": constant_count,
            "rotation_count": len(resource["rotation_steps"]),
            "rotation_batch_count": len(resource["rotation_batches"]),
            "monomial_count": len(resource["monomial_powers"]),
            "native_bootstrap_precompute": False,
            "generated_program_executed": False,
        },
        "bootstrap generation record",
    )

    expected_files = {
        "source": (source_path, "bootstrap_qualification.cu"),
        "raw": (raw_air_path, "bootstrap_raw.air"),
        "post_ckks": (post_ckks_air_path, "bootstrap_post_ckks.air"),
    }
    source_entry = record.get("source")
    source_file, source_name = expected_files["source"]
    if source_entry != {
        "path": source_name,
        "sha256": sha256(source_file),
        "bytes": source_file.stat().st_size,
    }:
        raise SystemExit("bootstrap generation source binding is inconsistent")
    air = record.get("air")
    if not isinstance(air, dict) or set(air) != {"raw", "post_ckks"}:
        raise SystemExit("bootstrap generation AIR binding has an invalid shape")
    for stage in ("raw", "post_ckks"):
        path, name = expected_files[stage]
        if air.get(stage) != {
            "path": name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }:
            raise SystemExit(f"bootstrap generation {stage} AIR binding is inconsistent")
    manifests = record.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(manifest_paths):
        raise SystemExit("bootstrap generation manifest bindings have an invalid shape")
    for name, path in manifest_paths.items():
        if manifests.get(name) != {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }:
            raise SystemExit(f"bootstrap generation {name} binding is inconsistent")


def bootstrap_fields(root: Path) -> dict[str, Any]:
    evidence = root / "bootstrap-qualification"
    output = evidence / "bootstrap_qualification"
    frozen_reference_path = root / "bootstrap-frozen-reference.json"
    frozen_reference = read_json(frozen_reference_path)
    run = read_json(evidence / "manifest.json")
    artifact_path = evidence / "artifact_manifest.json"
    artifact = read_json(artifact_path)
    generation_path = output / "generation.json"
    generation = read_json(generation_path)
    audit_path = output / "source-audit.json"
    audit = read_json(audit_path)
    qualification_path = output / "qualification.json"
    qualification = read_json(qualification_path)
    symbol_closure_path = output / "symbol-closure.json"
    symbol_closure = read_json(symbol_closure_path)
    linked_symbols_path = output / "linked_binary_symbols.txt"
    io_helper_closure_path = output / "io-helper-closure.json"
    io_helper_closure = read_json(io_helper_closure_path)
    generated_symbols_path = output / "generated_object_symbols.txt"
    harness_symbols_path = output / "harness_object_symbols.txt"
    archive_audit_path = output / "archive-member-audit.json"
    archive_audit = read_json(archive_audit_path)
    context_path = output / "compiler_context_manifest.json"
    context = read_json(context_path)
    resource_path = output / "compiler_resource_manifest.json"
    resource = read_json(resource_path)
    constant_path = output / "compiler_constant_manifest.json"
    constants = read_json(constant_path)
    raw_air_path = output / "bootstrap_raw.air"
    post_ckks_air_path = output / "bootstrap_post_ckks.air"
    source_path = output / "bootstrap_qualification.cu"
    harness_path = output / "bootstrap_phantom_constants.cu"
    binary_path = output / "bootstrap_phantom_constants_sm80"
    ace_source_path = evidence / "ace_source_manifest.json"
    phantom_source_path = evidence / "phantom_source_manifest.json"
    ace_source = read_json(ace_source_path)
    phantom_source = read_json(phantom_source_path)

    ace_commit = run.get("ace_commit")
    phantom_commit = run.get("phantom_commit")
    if not isinstance(ace_commit, str) or re.fullmatch(r"[0-9a-f]{40}", ace_commit) is None:
        raise SystemExit("bootstrap qualification ACE commit is invalid")
    if not isinstance(phantom_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", phantom_commit
    ) is None:
        raise SystemExit("bootstrap qualification Phantom commit is invalid")
    require_fields(
        run,
        {
            "status": "pass",
            "gate": "bootstrap",
            "source_mode": "snapshot",
            "ace_commit": ace_commit,
            "phantom_commit": phantom_commit,
            "ace_worktree_dirty": False,
            "gpu_executables_were_run": False,
        },
        "bootstrap run manifest",
    )
    require_fields(
        artifact,
        {
            "schema_version": BOOTSTRAP_ARTIFACT_SCHEMA,
            "status": "bound",
            "source_mode": "snapshot",
            "ace_commit": ace_commit,
            "phantom_commit": phantom_commit,
        },
        "bootstrap artifact manifest",
    )
    require_fields(
        ace_source,
        {"schema_version": "1.0.0", "kind": "ace", "commit": ace_commit},
        "bootstrap ACE source manifest",
    )
    require_fields(
        phantom_source,
        {
            "schema_version": "1.0.0",
            "kind": "phantom",
            "commit": phantom_commit,
        },
        "bootstrap Phantom source manifest",
    )

    generation_invocation_path = evidence / "bootstrap_generation_invocation.json"
    generation_invocation = read_json(generation_invocation_path)
    generation_argv = generation_invocation.get("argv")
    generation_prefix = [
        "tools/phantom_gpu/generate_bootstrap_qualification.py",
        "bootstrap_qualification",
    ]
    if not isinstance(generation_argv, list) or generation_argv[:2] != generation_prefix:
        raise SystemExit("bootstrap generation invocation executable/output is invalid")
    parameters = generation_argv[2:]
    if len(parameters) % 2:
        raise SystemExit("bootstrap generation invocation has an unpaired option")
    generation_options = dict(zip(parameters[0::2], parameters[1::2]))
    if len(generation_options) != len(parameters) // 2:
        raise SystemExit("bootstrap generation invocation options are not unique")
    required_generation_options = {
        "--poly-degree", "--mul-level", "--input-level", "--security-level",
        "--scaling-factor-bits", "--first-prime-bits", "--hamming-weight",
    }
    if set(generation_options) != required_generation_options:
        raise SystemExit("bootstrap generation invocation omits an explicit option")
    compiler_options = {
        option.removeprefix("--").replace("-", "_"): int(value)
        for option, value in generation_options.items()
    }
    generation_invocation_sha, normalized_generation = validate_invocation(
        generation_invocation_path,
        BOOTSTRAP_GENERATION_INVOCATION_SCHEMA,
        None,
        "bootstrap generation invocation",
    )
    qualification_invocation_path = evidence / "qualification_invocation.json"
    qualification_invocation_sha, normalized_qualification = validate_invocation(
        qualification_invocation_path,
        BOOTSTRAP_QUALIFICATION_INVOCATION_SCHEMA,
        ["tools/phantom_gpu/compile_only.sh", "--gate", "bootstrap"] + parameters,
        "bootstrap qualification invocation",
    )
    required_context_keys = {
        "data_q_bit_sizes", "first_modulus_bits", "hamming_weight",
        "input_level", "logical_slot_capacity", "packing", "polynomial_degree",
        "q_part_count", "resource_schema_version", "scaling_modulus_bits",
        "schema_version", "security_level", "special_p_bit_sizes",
    }
    if set(context) != required_context_keys:
        raise SystemExit("bootstrap context manifest has an invalid shape")
    require_fields(
        context,
        {
            "schema_version": 1,
            "resource_schema_version": 3,
            "polynomial_degree": int(generation_options["--poly-degree"]),
            "logical_slot_capacity": int(generation_options["--poly-degree"]) // 2,
            "packing": context.get("packing"),
            "input_level": int(generation_options["--input-level"]),
            "security_level": int(generation_options["--security-level"]),
            "scaling_modulus_bits": int(generation_options["--scaling-factor-bits"]),
            "first_modulus_bits": int(generation_options["--first-prime-bits"]),
            "hamming_weight": int(generation_options["--hamming-weight"]),
        },
        "bootstrap context manifest",
    )
    if len(context.get("data_q_bit_sizes", [])) != int(
        generation_options["--mul-level"]
    ):
        raise SystemExit("bootstrap context data-Q count is invalid")

    required_resource_keys = {
        "schema_version", "context_schema_version", "relinearization_key",
        "rotation_steps", "conjugation_key", "rotate_batch",
        "rotation_batches", "raise_mod", "monomial_powers",
        "complex_plaintext", "native_bootstrap_precompute",
    }
    if set(resource) != required_resource_keys:
        raise SystemExit("bootstrap resource manifest has an invalid shape")
    require_fields(
        resource,
        {
            "schema_version": 3,
            "context_schema_version": 1,
            "relinearization_key": True,
            "conjugation_key": True,
            "rotate_batch": True,
            "raise_mod": True,
            "complex_plaintext": True,
            "native_bootstrap_precompute": False,
        },
        "bootstrap resource manifest",
    )
    if any(not resource.get(name) for name in (
        "rotation_steps", "rotation_batches", "monomial_powers"
    )):
        raise SystemExit("bootstrap resource manifest lacks primitive resources")

    context_sha = sha256(context_path)
    resource_sha = sha256(resource_path)
    constant_sha = sha256(constant_path)
    if set(constants) != {
        "constants", "context_manifest_sha256", "context_schema_version",
        "resource_schema_version", "schema_version",
    }:
        raise SystemExit("bootstrap constant manifest has an invalid shape")
    require_fields(
        constants,
        {
            "schema_version": 1,
            "context_schema_version": 1,
            "resource_schema_version": 3,
            "context_manifest_sha256": context_sha,
        },
        "bootstrap constant manifest",
    )
    constant_entries = constants.get("constants")
    if not isinstance(constant_entries, list) or not constant_entries:
        raise SystemExit("bootstrap constant manifest is empty")

    raw_air_sha = sha256(raw_air_path)
    post_ckks_air_sha = sha256(post_ckks_air_path)
    source_sha = sha256(source_path)
    harness_sha = sha256(harness_path)
    binary_sha = sha256(binary_path)
    generation_sha = sha256(generation_path)
    audit_sha = sha256(audit_path)
    qualification_sha = sha256(qualification_path)
    symbol_closure_sha = sha256(symbol_closure_path)
    io_helper_closure_sha = sha256(io_helper_closure_path)
    archive_audit_sha = sha256(archive_audit_path)
    ace_source_sha = sha256(ace_source_path)
    phantom_source_sha = sha256(phantom_source_path)

    validate_bootstrap_generation(
        generation,
        source_path,
        raw_air_path,
        post_ckks_air_path,
        {
            "context": context_path,
            "resource": resource_path,
            "constant": constant_path,
        },
        resource,
        len(constant_entries),
        compiler_options,
        generation.get("bootstrap_parameters"),
    )

    audit_inputs = {
        "raw_air": raw_air_path,
        "post_ckks_air": post_ckks_air_path,
        "context_manifest": context_path,
        "resource_manifest": resource_path,
        "constant_manifest": constant_path,
        "source": source_path,
    }
    audit_projection = bootstrap_generated_artifact_audit_projection(
        audit, audit_inputs, raw_air_path, post_ckks_air_path, source_path
    )
    if audit.get("counts", {}).get("constants") != len(constant_entries):
        raise SystemExit("bootstrap source audit constant count is inconsistent")

    symbol_projection = bootstrap_symbol_projection(
        symbol_closure, linked_symbols_path
    )
    io_helper_projection = bootstrap_io_helper_projection(
        io_helper_closure,
        source_path,
        harness_path,
        generated_symbols_path,
        harness_symbols_path,
    )
    require_fields(
        archive_audit,
        {
            "schema_version": ARCHIVE_MEMBER_AUDIT_SCHEMA,
            "status": "pass",
            "forbidden_member_count": 0,
            "forbidden_members": [],
        },
        "bootstrap archive-member audit",
    )

    stable_hashes = {
        "raw_air_sha256": raw_air_sha,
        "post_ckks_air_sha256": post_ckks_air_sha,
        "generated_source_sha256": source_sha,
        "compiler_context_manifest_sha256": context_sha,
        "compiler_resource_manifest_sha256": resource_sha,
        "compiler_constant_manifest_sha256": constant_sha,
        "generation_record_sha256": generation_sha,
        "generated_artifact_audit_sha256": audit_sha,
        "linked_binary_sha256": binary_sha,
        "harness_source_sha256": harness_sha,
        "symbol_closure_sha256": symbol_closure_sha,
        "io_helper_closure_sha256": io_helper_closure_sha,
        "archive_member_audit_sha256": archive_audit_sha,
    }
    require_fields(
        qualification,
        {
            "status": "pass",
            "gate": "bootstrap",
            "architecture": "sm_80",
            "phantom_commit": phantom_commit,
            "generated_source_contains_native_bootstrap": False,
            "production_archive_contains_native_bootstrap": False,
            "primitive_only_provider_archive": True,
            "executable_was_run": False,
            **stable_hashes,
        },
        "bootstrap host qualification",
    )
    archive_hashes = {
        name: require_digest(
            qualification.get(name), f"bootstrap host qualification {name}"
        )
        for name in (
            "adapter_archive_sha256", "provider_archive_sha256",
            "common_archive_sha256",
        )
    }

    artifact_files = artifact.get("files")
    if not isinstance(artifact_files, dict):
        raise SystemExit("bootstrap artifact file inventory is invalid")
    required_artifact_files = {
        "ace_source_manifest.json": ace_source_sha,
        "phantom_source_manifest.json": phantom_source_sha,
        "qualification_invocation.json": qualification_invocation_sha,
        "bootstrap_generation_invocation.json": generation_invocation_sha,
        "bootstrap_qualification/bootstrap_raw.air": raw_air_sha,
        "bootstrap_qualification/bootstrap_post_ckks.air": post_ckks_air_sha,
        "bootstrap_qualification/compiler_context_manifest.json": context_sha,
        "bootstrap_qualification/compiler_resource_manifest.json": resource_sha,
        "bootstrap_qualification/compiler_constant_manifest.json": constant_sha,
        "bootstrap_qualification/generation.json": generation_sha,
        "bootstrap_qualification/source-audit.json": audit_sha,
        "bootstrap_qualification/bootstrap_qualification.cu": source_sha,
        "bootstrap_qualification/bootstrap_phantom_constants.cu": harness_sha,
        "bootstrap_qualification/bootstrap_phantom_constants_sm80": binary_sha,
        "bootstrap_qualification/linked_binary_symbols.txt": sha256(
            linked_symbols_path
        ),
        "bootstrap_qualification/generated_object_symbols.txt": sha256(
            generated_symbols_path
        ),
        "bootstrap_qualification/harness_object_symbols.txt": sha256(
            harness_symbols_path
        ),
        "bootstrap_qualification/symbol-closure.json": symbol_closure_sha,
        "bootstrap_qualification/io-helper-closure.json": io_helper_closure_sha,
        "bootstrap_qualification/archive-member-audit.json": archive_audit_sha,
        "bootstrap_qualification/qualification.json": qualification_sha,
    }
    for name, value in required_artifact_files.items():
        if artifact_files.get(name) != value:
            raise SystemExit(f"bootstrap artifact file hash is inconsistent: {name}")
    require_fields(
        artifact,
        {
            "normalized_qualification_argv_sha256": normalized_qualification,
            "normalized_generation_argv_sha256": normalized_generation,
            "raw_air_sha256": raw_air_sha,
            "post_ckks_air_sha256": post_ckks_air_sha,
            **{
                name: value
                for name, value in stable_hashes.items()
                if name not in {
                    "generated_source_sha256",
                    "symbol_closure_sha256", "io_helper_closure_sha256",
                    "archive_member_audit_sha256",
                }
            },
        },
        "bootstrap artifact manifest",
    )
    require_fields(
        run,
        {
            "ace_tracked_source_manifest_sha256": ace_source_sha,
            "phantom_source_manifest_sha256": phantom_source_sha,
            "qualification_invocation_sha256": qualification_invocation_sha,
            "normalized_qualification_argv_sha256": normalized_qualification,
            "compiler_context_manifest_sha256": context_sha,
            "compiler_resource_manifest_sha256": resource_sha,
            "compiler_constant_manifest_sha256": constant_sha,
            "bootstrap_generation_record_sha256": generation_sha,
            "bootstrap_generated_artifact_audit_sha256": audit_sha,
            "bootstrap_raw_air_sha256": raw_air_sha,
            "bootstrap_post_ckks_air_sha256": post_ckks_air_sha,
            "bootstrap_linked_binary_sha256": binary_sha,
            "bootstrap_harness_source_sha256": harness_sha,
            "bootstrap_host_qualification_sha256": qualification_sha,
            "artifact_manifest_sha256": sha256(artifact_path),
        },
        "bootstrap run manifest",
    )
    frozen_keys = {
        "schema_version",
        "status",
        "comparison",
        "raw_air_matches",
        "post_ckks_air_matches",
        "generated_source_matches",
        "architecture",
        "semantic_symbol_closure",
        "packaged_artifact_manifest_sha256",
        "packaged_host_qualification_sha256",
        "packaged_source_audit_sha256",
        "regenerated_artifact_manifest_sha256",
        "regenerated_host_qualification_sha256",
        "regenerated_source_audit_sha256",
        "packaged_generated_source_sha256",
        "regenerated_generated_source_sha256",
        "packaged_raw_air_sha256",
        "regenerated_raw_air_sha256",
        "packaged_post_ckks_air_sha256",
        "regenerated_post_ckks_air_sha256",
        "packaged_harness_source_sha256",
        "regenerated_harness_source_sha256",
        "packaged_linked_binary_sha256",
        "regenerated_linked_binary_sha256",
        "source_and_setup_match",
        "linked_binary_byte_identity_required",
    }
    if set(frozen_reference) != frozen_keys:
        raise SystemExit("bootstrap frozen reference has an invalid shape")
    for field in frozen_keys:
        if field.endswith("_sha256"):
            require_digest(
                frozen_reference.get(field), f"bootstrap frozen reference {field}"
            )
    require_fields(
        frozen_reference,
        {
            "schema_version": BOOTSTRAP_FROZEN_REFERENCE_SCHEMA,
            "status": "pass",
            "comparison": "exact-source-and-setup-with-per-run-cuda-artifacts",
            "raw_air_matches": True,
            "post_ckks_air_matches": True,
            "generated_source_matches": True,
            "architecture": "sm_80",
            "semantic_symbol_closure": symbol_projection,
            "regenerated_artifact_manifest_sha256": sha256(artifact_path),
            "regenerated_host_qualification_sha256": qualification_sha,
            "regenerated_source_audit_sha256": audit_sha,
            "packaged_generated_source_sha256": source_sha,
            "regenerated_generated_source_sha256": source_sha,
            "packaged_raw_air_sha256": raw_air_sha,
            "regenerated_raw_air_sha256": raw_air_sha,
            "packaged_post_ckks_air_sha256": post_ckks_air_sha,
            "regenerated_post_ckks_air_sha256": post_ckks_air_sha,
            "packaged_harness_source_sha256": harness_sha,
            "regenerated_harness_source_sha256": harness_sha,
            "regenerated_linked_binary_sha256": binary_sha,
            "source_and_setup_match": True,
            "linked_binary_byte_identity_required": False,
        },
        "bootstrap frozen reference",
    )

    return {
        "bootstrap_ace_commit": ace_commit,
        "bootstrap_phantom_commit": phantom_commit,
        "bootstrap_ace_source_manifest_sha256": ace_source_sha,
        "bootstrap_phantom_source_manifest_sha256": phantom_source_sha,
        "bootstrap_qualification_invocation_sha256": qualification_invocation_sha,
        "bootstrap_normalized_qualification_argv_sha256": normalized_qualification,
        "bootstrap_generation_invocation_sha256": generation_invocation_sha,
        "bootstrap_normalized_generation_argv_sha256": normalized_generation,
        "bootstrap_context_manifest_sha256": context_sha,
        "bootstrap_resource_manifest_sha256": resource_sha,
        "bootstrap_constant_manifest_sha256": constant_sha,
        "bootstrap_generation_record_sha256": generation_sha,
        "bootstrap_generated_artifact_audit_sha256": audit_sha,
        "bootstrap_generated_artifact_audit_projection": audit_projection,
        "bootstrap_raw_air_sha256": raw_air_sha,
        "bootstrap_post_ckks_air_sha256": post_ckks_air_sha,
        "bootstrap_generated_source_sha256": source_sha,
        "bootstrap_harness_source_sha256": harness_sha,
        "bootstrap_linked_binary_sha256": binary_sha,
        "bootstrap_archive_member_audit_sha256": archive_audit_sha,
        "bootstrap_symbol_closure_sha256": symbol_closure_sha,
        "bootstrap_symbol_closure_projection": symbol_projection,
        "bootstrap_io_helper_closure_sha256": io_helper_closure_sha,
        "bootstrap_io_helper_closure_projection": io_helper_projection,
        "bootstrap_frozen_reference_sha256": sha256(frozen_reference_path),
        **{f"bootstrap_{name}": value for name, value in archive_hashes.items()},
    }


def retained_fields(root: Path) -> dict[str, Any]:
    replay = read_json(root / "retained-frozen-reference.json")
    host = read_json(root / "retained-host/manifest.json")
    build = read_json(root / "retained-host/build-attestation.json")
    artifact = read_json(root / "retained-host/artifact-manifest.json")

    require_fields(
        replay,
        {
            "schema_version": RETAINED_REPLAY_SCHEMA,
            "status": "pass",
            "provider_neutral_ant_reference_matches": True,
        },
        "retained replay",
    )
    ace_commit = replay.get("ace_commit")
    phantom_commit = replay.get("phantom_commit")
    if not isinstance(ace_commit, str) or re.fullmatch(r"[0-9a-f]{40}", ace_commit) is None:
        raise SystemExit("retained replay ACE commit is invalid")
    if not isinstance(phantom_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", phantom_commit
    ) is None:
        raise SystemExit("retained replay Phantom commit is invalid")

    require_fields(
        host,
        {
            "schema_version": RETAINED_HOST_SCHEMA,
            "status": "pass",
            "gate": "retained_ckks",
            "source_mode": "snapshot",
            "fixture_lifecycle": "checked-bound",
            "ace_commit": ace_commit,
            "phantom_commit": phantom_commit,
            "ace_worktree_dirty": False,
            "host_ant_oracle_was_run": True,
            "gpu_executables_were_run": False,
        },
        "retained host manifest",
    )
    require_fields(
        build,
        {
            "schema_version": RETAINED_BUILD_SCHEMA,
            "status": "pass",
            "architecture": "sm_80",
            "source_mode": "snapshot",
            "ace_commit": ace_commit,
            "phantom_commit": phantom_commit,
            "host_ant_oracle_was_run": True,
            "gpu_executables_were_run": False,
        },
        "retained build attestation",
    )
    require_fields(
        artifact,
        {
            "schema_version": RETAINED_ARTIFACT_SCHEMA,
            "status": "bound",
            "fixture_lifecycle": "checked-bound",
            "ace_commit": ace_commit,
            "phantom_commit": phantom_commit,
        },
        "retained artifact manifest",
    )

    digest_bindings = {
        "ace_source_manifest_sha256": (
            replay,
            host,
            build,
            artifact,
        ),
        "phantom_source_manifest_sha256": (
            replay,
            host,
            build,
            artifact,
        ),
        "context_manifest_sha256": (replay,),
        "resource_manifest_sha256": (replay,),
        "fixture_sha256": (replay, host, build, artifact),
        "generation_attestation_sha256": (replay,),
    }
    resolved: dict[str, str] = {}
    for name, records in digest_bindings.items():
        value = require_digest(records[0].get(name), f"retained {name}")
        for record in records[1:]:
            if record.get(name) != value:
                raise SystemExit(f"retained {name} is internally inconsistent")
        resolved[name] = value

    cross_bindings = {
        "compiler_context_manifest_sha256": resolved["context_manifest_sha256"],
        "compiler_resource_manifest_sha256": resolved["resource_manifest_sha256"],
    }
    for name, value in cross_bindings.items():
        for label, record in (
            ("host manifest", host),
            ("build attestation", build),
            ("artifact manifest", artifact),
        ):
            if record.get(name) != value:
                raise SystemExit(f"retained {label} {name} is internally inconsistent")
    if build.get("compiler_invocation_sha256") != resolved[
        "generation_attestation_sha256"
    ]:
        raise SystemExit("retained build compiler invocation binding is inconsistent")

    ant_replay = replay.get("ant_replay")
    if not isinstance(ant_replay, dict):
        raise SystemExit("retained ANT replay record is absent")
    require_fields(
        ant_replay,
        {"status": "pass", "comparison": SEMANTIC_REPLAY_MODE},
        "retained ANT replay",
    )
    summary = ant_replay.get("summary")
    if not isinstance(summary, dict):
        raise SystemExit("retained ANT semantic summary is absent")
    require_fields(
        summary,
        {"schema_version": RETAINED_SUMMARY_SCHEMA, "status": "pass"},
        "retained ANT semantic summary",
    )
    summary_sha256 = require_digest(
        ant_replay.get("summary_sha256"), "retained ANT semantic summary"
    )
    if canonical_json_sha256(summary) != summary_sha256:
        raise SystemExit("retained ANT semantic summary hash differs")

    exact_artifacts = replay.get("artifacts")
    exact_artifact_count = replay.get("exact_artifact_count")
    if (
        not isinstance(exact_artifacts, dict)
        or not exact_artifacts
        or isinstance(exact_artifact_count, bool)
        or not isinstance(exact_artifact_count, int)
        or exact_artifact_count != len(exact_artifacts)
    ):
        raise SystemExit("retained exact replay artifact inventory is invalid")
    for name, value in exact_artifacts.items():
        if not isinstance(name, str) or not name:
            raise SystemExit("retained exact replay artifact name is invalid")
        require_digest(value, f"retained exact replay artifact {name}")

    generated_ant = require_digest(
        build.get("generated_ant_source_sha256"), "retained generated ANT source"
    )
    generated_phantom = require_digest(
        build.get("generated_phantom_source_sha256"),
        "retained generated Phantom source",
    )
    normalized_command = require_digest(
        artifact.get("normalized_compiler_command_sha256"),
        "retained normalized compiler command",
    )
    post_air = require_digest(
        artifact.get("post_ckks_air_sha256"), "retained post-CKKS AIR"
    )
    production_post_air = require_digest(
        artifact.get("production_post_ckks_air_sha256"),
        "retained production post-CKKS AIR",
    )
    expected_exact_bindings = {
        "context_manifest": resolved["context_manifest_sha256"],
        "resource_manifest": resolved["resource_manifest_sha256"],
        "fixture": resolved["fixture_sha256"],
        "generation_attestation": resolved["generation_attestation_sha256"],
        "phantom_post_ckks_air": post_air,
        "production_post_ckks_air": production_post_air,
    }
    for name, value in expected_exact_bindings.items():
        if exact_artifacts.get(name) != value:
            raise SystemExit(f"retained exact replay artifact {name} is inconsistent")
    if (
        summary.get("fixture_sha256") != resolved["fixture_sha256"]
        or summary.get("context_manifest_sha256")
        != resolved["context_manifest_sha256"]
    ):
        raise SystemExit("retained ANT semantic summary bindings are inconsistent")

    compared = {
        "retained_ace_commit": ace_commit,
        "retained_phantom_commit": phantom_commit,
        **{f"retained_{name}": value for name, value in resolved.items()},
        "retained_exact_artifact_count": exact_artifact_count,
        "retained_exact_artifacts": dict(sorted(exact_artifacts.items())),
        "retained_ant_semantic_summary_sha256": summary_sha256,
        "retained_generated_ant_source_sha256": generated_ant,
        "retained_generated_phantom_source_sha256": generated_phantom,
        "retained_normalized_compiler_command_sha256": normalized_command,
        "retained_post_ckks_air_sha256": post_air,
        "retained_production_post_ckks_air_sha256": production_post_air,
    }
    return compared


def fields(root: Path) -> dict[str, Any]:
    configuration = read_json(root / "qualification/ckks2c/configuration.json")
    ace_audit = read_json(root / "ace-source-audit.json")
    phantom_audit = read_json(root / "phantom-source-audit.json")
    qualification = read_json(root / "qualification/ckks2c/qualification.json")
    frozen_reference = read_json(root / "ordinary-frozen-reference.json")
    compared = {
        "base_image": configuration["environment_identity"]["base_image"],
        "base_config_digest": configuration["environment_identity"]["config_digest"],
        "bootstrap_sha256": configuration["environment_identity"]["bootstrap_sha256"],
        "ace_commit": configuration["ace"]["commit"],
        "phantom_commit": configuration["phantom"]["commit"],
        "source_manifest_sha256": configuration["source_manifest_sha256"],
        "ace_archive_sha256": ace_audit["archive_sha256"],
        "phantom_archive_sha256": phantom_audit["archive_sha256"],
        "cuda_architecture": configuration["cuda_architecture"],
        "tool_versions": configuration["toolchain"]["versions"],
        "compiler_context_manifest_sha256": configuration[
            "compiler_context_manifest"
        ]["sha256"],
        "apt_lock_sha256": sha256(root / "environment/apt-packages.lock"),
        "python_lock_sha256": sha256(
            root / "environment/python-requirements-hashed.lock"
        ),
        "base_files_lock_sha256": sha256(root / "environment/base-files.sha256"),
        "dpkg_manifest_sha256": sha256(root / "environment/dpkg-manifest.txt"),
        "python_manifest_sha256": sha256(root / "environment/python-manifest.txt"),
        "generated_source_sha256": qualification["source_sha256"],
        "terminal_selection_sha256": qualification["terminal_selection_sha256"],
        "generated_binary_sha256": qualification["binary_sha256"],
        "ordinary_context_manifest_sha256": frozen_reference[
            "context_manifest_sha256"
        ],
        "ordinary_resource_manifest_sha256": frozen_reference[
            "resource_manifest_sha256"
        ],
        "ordinary_fixture_sha256": frozen_reference["fixture_sha256"],
        "ordinary_cpu_reference_sha256": frozen_reference[
            "cpu_reference_sha256"
        ],
        "ordinary_cpu_values_sha256": frozen_reference["cpu_values_sha256"],
        "ordinary_ant_verification_sha256": frozen_reference[
            "ant_verification_sha256"
        ],
        **bootstrap_fields(root),
        **retained_fields(root),
    }
    if compared["bootstrap_ace_commit"] != compared["ace_commit"]:
        raise SystemExit("ordinary and bootstrap ACE commits are inconsistent")
    if compared["bootstrap_phantom_commit"] != compared["phantom_commit"]:
        raise SystemExit("ordinary and bootstrap Phantom commits are inconsistent")
    if compared["bootstrap_ace_source_manifest_sha256"] != compared[
        "source_manifest_sha256"
    ]:
        raise SystemExit("ordinary and bootstrap ACE source manifests are inconsistent")
    if compared["retained_ace_commit"] != compared["ace_commit"]:
        raise SystemExit("ordinary and retained ACE commits are inconsistent")
    if compared["retained_phantom_commit"] != compared["phantom_commit"]:
        raise SystemExit("ordinary and retained Phantom commits are inconsistent")
    if (
        compared["retained_ace_source_manifest_sha256"]
        != compared["source_manifest_sha256"]
    ):
        raise SystemExit("ordinary and retained ACE source manifests are inconsistent")
    if compared["bootstrap_ace_source_manifest_sha256"] != compared[
        "retained_ace_source_manifest_sha256"
    ]:
        raise SystemExit("bootstrap and retained ACE source manifests are inconsistent")
    if compared["bootstrap_phantom_source_manifest_sha256"] != compared[
        "retained_phantom_source_manifest_sha256"
    ]:
        raise SystemExit("bootstrap and retained Phantom source manifests are inconsistent")
    return compared


def comparison_report(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    if set(local) != set(remote):
        raise SystemExit("local and remote comparison field inventories differ")
    per_run_fields = {
        "bootstrap_adapter_archive_sha256",
        "bootstrap_common_archive_sha256",
        "bootstrap_frozen_reference_sha256",
        "bootstrap_io_helper_closure_sha256",
        "bootstrap_linked_binary_sha256",
        "bootstrap_provider_archive_sha256",
        "bootstrap_symbol_closure_sha256",
        "generated_binary_sha256",
    }
    stable = sorted(set(local) - per_run_fields)
    matching = [key for key in stable if local[key] == remote[key]]
    mismatches = {
        key: {"local": local[key], "remote": remote[key]}
        for key in stable
        if local[key] != remote[key]
    }
    retained = sorted(key for key in stable if key.startswith("retained_"))
    bootstrap = sorted(key for key in stable if key.startswith("bootstrap_"))
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not mismatches else "fail",
        "comparison_contract": {
            "stable_field_count": len(stable),
            "retained_field_count": len(retained),
            "retained_fields": retained,
            "bootstrap_field_count": len(bootstrap),
            "bootstrap_fields": bootstrap,
        },
        "matching_fields": matching,
        "mismatches": mismatches,
        "allowed_differences": {
            "generated_binary_sha256": {
                "local": local["generated_binary_sha256"],
                "remote": remote["generated_binary_sha256"],
                "reason": (
                    "recorded but not required equal because tool output can "
                    "embed build-host details"
                ),
            },
            "bootstrap_frozen_reference_sha256": {
                "local": local["bootstrap_frozen_reference_sha256"],
                "remote": remote["bootstrap_frozen_reference_sha256"],
                "reason": (
                    "recorded but not required equal because it binds per-run "
                    "artifact inventories"
                ),
            },
            "bootstrap_cuda_toolchain_provenance": {
                field: {
                    "local": local[field],
                    "remote": remote[field],
                    "reason": (
                        "recorded and internally validated per run; raw CUDA 12.4 "
                        "objects, archives, binaries, and symbol inventories may "
                        "contain nondeterministic toolchain bytes"
                    ),
                }
                for field in sorted(
                    per_run_fields
                    - {
                        "bootstrap_frozen_reference_sha256",
                        "generated_binary_sha256",
                    }
                )
            },
            "retained_per_run_receipts": [
                "randomized ANT ciphertext-derived hashes and decoded bytes",
                "host manifest, artifact manifest, build attestation, and checksum receipt hashes",
                "archive, object, executable, ownership-token, log, and timing hashes",
            ],
            "bootstrap_per_run_evidence": [
                "GPU identity, driver, pod, hostname, kernel, and timestamps",
                "constant-cache execution receipt, stdout, stderr, and raw provider output",
                "setup timing, host-memory, device-memory, and cache-usage measurements",
            ],
            "remote_only": [
                "driver",
                "gpu_uuid",
                "pod_id",
                "timestamps",
                "hostname",
                "kernel",
                "retained GPU correctness, exact-RNS, alias, rejection, ownership, and sanitizer results",
            ],
        },
    }


def verify_result_checksum_closure(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise SystemExit("result checksum record has a malformed line")
        digest, name = parts
        relative = name.removeprefix("*").removeprefix("./")
        require_digest(digest, f"result checksum for {relative}")
        if relative in listed:
            raise SystemExit("result checksum record contains a duplicate path")
        listed[relative] = digest
    actual = {
        str(path.relative_to(root)): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if listed != actual:
        raise SystemExit("result checksum closure differs from the result tree")


def correctness_host_executable_digests(
    build_files: dict[str, Any],
    native_record: dict[str, Any],
    generated_record: dict[str, Any],
) -> tuple[str, str]:
    bindings = (
        (
            "native ANT",
            "bootstrap_qualification/generated_bootstrap_native_ant_oracle",
            native_record,
        ),
        (
            "generated DSL/ANT",
            "bootstrap_qualification/generated_bootstrap_dsl_ant_oracle",
            generated_record,
        ),
    )
    observed: list[str] = []
    for label, artifact_name, record in bindings:
        artifact_digest = require_digest(
            build_files.get(artifact_name), f"rebuilt {label} executable"
        )
        record_digest = require_digest(
            record.get("execution", {}).get("provenance", {}).get(
                "executable_sha256"
            ),
            f"executed {label} executable",
        )
        if record_digest != artifact_digest:
            raise SystemExit(
                f"executed {label} binary differs from rebuilt provenance"
            )
        observed.append(record_digest)
    return observed[0], observed[1]


def correctness_fields(root: Path, expected_mode: str) -> dict[str, Any]:
    verify_result_checksum_closure(root)
    pipeline = read_json(root / "pipeline-result.json")
    require_fields(
        pipeline,
        {"schema_version": "1.0.0", "status": "pass", "mode": expected_mode,
         "exit_code": 0},
        "correctness pipeline result",
    )
    lifecycle = read_json(root / "generated-bootstrap-correctness-lifecycle.json")
    expected_execution = "pass" if expected_mode == "runpod" else "skipped-local-no-gpu"
    require_fields(
        lifecycle,
        {
            "schema_version":
                "ace.phantom.generated-bootstrap-correctness-lifecycle/1.0.0",
            "status": "pass",
            "host_oracles_replayed_before_gpu": True,
            "gpu_execution": expected_execution,
        },
        "correctness lifecycle",
    )
    for replay_name in (
        "packaged-correctness-host-replay.json",
        "regenerated-correctness-host-replay.json",
    ):
        replay = read_json(root / replay_name)
        if (
            replay.get("schema_version")
            != "ace.phantom.bootstrap-host-oracle-replay/1.0.0"
            or replay.get("status") != "pass"
            or not replay.get("cases")
        ):
            raise SystemExit(f"host replay is incomplete: {replay_name}")

    packaged = root / "correctness-input"
    payload = read_json(packaged / "payload.json")
    if (
        payload.get("schema_version")
        != "ace.phantom.generated-bootstrap-correctness-payload/1.0.0"
        or payload.get("status") != "pass"
        or payload.get("host_oracles_replayed") is not True
    ):
        raise SystemExit("correctness payload contract is invalid")
    payload_files = payload.get("files")
    observed_files = {
        path.name: sha256(path)
        for path in packaged.iterdir()
        if path.is_file() and path.name not in {"payload.json", "SHA256SUMS"}
    }
    if payload_files != observed_files:
        raise SystemExit("correctness payload file inventory differs")
    compiler_invocation = read_json(
        packaged / "correctness-compiler-invocation.json"
    )
    if (
        compiler_invocation.get("schema_version")
        != "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0"
        or compiler_invocation.get("status") != "pass"
        or not isinstance(compiler_invocation.get("options"), dict)
    ):
        raise SystemExit("correctness compiler invocation is invalid")
    artifact = read_json(packaged / "correctness-artifact-manifest.json")
    source_audit = read_json(packaged / "correctness-source-audit.json")
    host_qualification = read_json(
        packaged / "correctness-host-qualification.json"
    )
    if (
        artifact.get("schema_version") != "ace.phantom.bootstrap-artifacts/3.0.0"
        or artifact.get("status") != "bound"
        or source_audit.get("schema_version")
        != "ace.phantom.bootstrap-generated-artifact-audit/4.0.0"
        or source_audit.get("status") != "pass"
        or host_qualification.get("schema_version")
        != "ace.phantom.bootstrap-host-qualification/2.0.0"
        or host_qualification.get("status") != "pass"
        or host_qualification.get("architecture") != "sm_80"
    ):
        raise SystemExit("correctness artifact, source audit, or architecture is invalid")
    semantics = read_json(packaged / "correctness-bootstrap-semantics.json")
    generation = read_json(packaged / "correctness-generation.json")
    post_ckks_air_sha256 = sha256(packaged / "correctness-post-ckks.air")
    if (
        generation.get("air", {}).get("post_ckks", {}).get("sha256")
        != post_ckks_air_sha256
    ):
        raise SystemExit("correctness generated post-CKKS AIR binding is invalid")
    validate_terminal_body_closure(
        source_audit, semantics, post_ckks_air_sha256
    )
    fixture = read_json(packaged / "correctness-fixture.json")
    identity_attestation = semantics.get("identity_domain_attestation")
    identity_domain = semantics.get("supported_identity_domain")
    if (
        semantics.get("schema_version")
        != "ace.phantom.generated-bootstrap.semantics/2.0.0"
        or semantics.get("status") != "pass"
        or fixture.get("schema_version")
        != "ace.phantom.bootstrap-correctness-fixture/2.0.0"
        or not isinstance(identity_domain, dict)
        or fixture.get("supported_identity_domain", {}).get(
            "attested_domain"
        )
        != identity_domain
        or not isinstance(identity_attestation, dict)
        or identity_attestation.get("schema_version")
        != "ace.phantom.bootstrap-clear-evalmod-domain/1.0.0"
        or identity_attestation.get("status") != "attested"
        or identity_attestation.get("scope")
        != {
            "purpose": "domain-attestation-only",
            "provider_value_oracle": False,
            "may_supply_expected_case_values": False,
        }
        or identity_domain.get("evidence", {}).get("attestation_sha256")
        != hashlib.sha256(
            json.dumps(
                identity_attestation, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise SystemExit("correctness identity-domain authority is invalid")
    for kind in ("ace", "phantom"):
        source = read_json(packaged / f"{kind}-source.manifest.json")
        if source.get("commit") != payload.get(f"{kind}_commit"):
            raise SystemExit(f"correctness {kind} source commit differs")
        if payload.get("source_snapshots", {}).get(f"{kind}_manifest_sha256") != sha256(
            packaged / f"{kind}-source.manifest.json"
        ):
            raise SystemExit(f"correctness {kind} source manifest binding differs")

    build_artifact = read_json(
        root / "correctness-build-artifact-manifest.json"
    )
    build_qualification = read_json(
        root / "correctness-build-host-qualification.json"
    )
    if (
        build_artifact.get("schema_version")
        != "ace.phantom.bootstrap-artifacts/3.0.0"
        or build_artifact.get("status") != "bound"
        or build_qualification.get("schema_version")
        != "ace.phantom.bootstrap-host-qualification/2.0.0"
        or build_qualification.get("status") != "pass"
    ):
        raise SystemExit("rebuilt correctness artifact provenance is invalid")
    build_files = build_artifact.get("files")
    if not isinstance(build_files, dict):
        raise SystemExit("rebuilt correctness artifact file inventory is invalid")
    gpu_executable = require_digest(
        build_files.get(
            "bootstrap_qualification/"
            "generated_bootstrap_phantom_correctness_sm80"
        ),
        "rebuilt generated Phantom executable",
    )
    gpu_elf_report = require_digest(
        build_files.get(
            "bootstrap_qualification/gpu-runner-inspection/cuda_elf.txt"
        ),
        "rebuilt generated Phantom ELF report",
    )
    native_record = read_json(root / "native-ant-reference.json")
    generated_record = read_json(root / "generated-ant-reference.json")
    native_executable, generated_executable = correctness_host_executable_digests(
        build_files, native_record, generated_record
    )
    archive_digests = {
        name: require_digest(
            build_qualification.get(f"{name}_archive_sha256"),
            f"rebuilt {name} archive",
        )
        for name in ("adapter", "provider", "common")
    }
    if expected_mode == "runpod":
        gpu_record = read_json(root / "generated-phantom-correctness.json")
        if (
            gpu_record.get("execution", {}).get("executable_sha256")
            != gpu_executable
        ):
            raise SystemExit(
                "executed generated Phantom binary differs from rebuilt provenance"
            )

    stable_names = (
        "ace-source.manifest.json",
        "phantom-source.manifest.json",
        "correctness-qualification-invocation.json",
        "correctness-generation-invocation.json",
        "correctness-compiler-invocation.json",
        "correctness-raw.air",
        "correctness-post-ckks.air",
        "correctness-post-operations.air",
        "correctness-context-manifest.json",
        "correctness-resource-manifest.json",
        "correctness-constant-manifest.json",
        "correctness-bootstrap-semantics.json",
        "correctness-post-operation-attestation.json",
        "correctness-fixture.json",
        "correctness-generation.json",
        "correctness-source-audit.json",
        "correctness-gpu-harness.cu",
        "correctness-generated-phantom.cu",
        "correctness-generated-ant.cxx",
        "run_build_and_health.sh",
        "bootstrap_environment.sh",
        "bootstrap_domain_attestation.py",
        "bootstrap_correctness.py",
        "phase_helpers.sh",
        "source_archive.py",
        "dependencies.env",
        "toolchain.env",
        "apt-packages.lock",
        "python-requirements-hashed.lock",
        "base-files.sha256",
    )
    result: dict[str, Any] = {
        "ace_commit": payload["ace_commit"],
        "phantom_commit": payload["phantom_commit"],
        **{name: sha256(packaged / name) for name in stable_names},
        "per_run_ace_archive_sha256": read_json(root / "ace-source-audit.json").get(
            "archive_sha256"
        ),
        "per_run_phantom_archive_sha256": read_json(
            root / "phantom-source-audit.json"
        ).get(
            "archive_sha256"
        ),
        "per_run_gpu_executable_sha256": gpu_executable,
        "per_run_gpu_elf_report_sha256": gpu_elf_report,
        "per_run_native_ant_executable_sha256": native_executable,
        "per_run_generated_ant_executable_sha256": generated_executable,
        "per_run_adapter_archive_sha256": archive_digests["adapter"],
        "per_run_provider_archive_sha256": archive_digests["provider"],
        "per_run_common_archive_sha256": archive_digests["common"],
    }
    for name, value in result.items():
        if name.endswith("sha256"):
            require_digest(value, f"correctness comparison {name}")
    if expected_mode == "runpod":
        comparison = read_json(root / "generated-bootstrap-three-way-comparison.json")
        sanitizer = read_json(root / "correctness-sanitizer.json")
        if comparison.get("status") != "pass" or sanitizer.get("status") != "pass":
            raise SystemExit("remote correctness or sanitizer result did not pass")
    return result


def correctness_comparison_report(
    local: dict[str, Any], remote: dict[str, Any]
) -> dict[str, Any]:
    if set(local) != set(remote):
        raise SystemExit("correctness comparison field inventories differ")
    per_run = {key for key in local if key.startswith("per_run_")}
    stable = set(local) - per_run
    mismatches = {
        key: {"local": local[key], "remote": remote[key]}
        for key in sorted(stable)
        if local[key] != remote[key]
    }
    return {
        "schema_version": CORRECTNESS_REPORT_SCHEMA,
        "status": "pass" if not mismatches else "fail",
        "stable_field_count": len(stable),
        "matching_fields": [
            key for key in sorted(stable) if local[key] == remote[key]
        ],
        "mismatches": mismatches,
        "per_run_provenance_policy": {
            "cuda_binary_and_archive_bytes": "checksum-bound-per-run",
            "gpu_execution": "required-remotely-and-skipped-locally",
            "semantic_authorities": "byte-identical-and-replayed-in-each-environment",
            "recorded_digests": {
                key: {"local": local[key], "remote": remote[key]}
                for key in sorted(per_run)
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("full", "generated-bootstrap-correctness"),
        default="full"
    )
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        arguments.mode == "generated-bootstrap-correctness"
        and arguments.output.exists()
    ):
        raise SystemExit(
            f"correctness comparison output already exists: {arguments.output}"
        )
    with tempfile.TemporaryDirectory(prefix="ace-environment-compare-") as temporary:
        base = Path(temporary)
        local_root = safe_extract(arguments.local, base / "local")
        remote_root = safe_extract(arguments.remote, base / "remote")
        if arguments.mode == "generated-bootstrap-correctness":
            local = correctness_fields(local_root, "local")
            remote = correctness_fields(remote_root, "runpod")
            if local["ace_commit"] != remote["ace_commit"]:
                raise SystemExit(
                    "correctness comparison source commits differ"
                )
            require_selected_commit_entrypoint(
                REPOSITORY,
                local["ace_commit"],
                Path(__file__).resolve(),
                CORRECTNESS_COMPARISON_ENTRYPOINT,
            )
        else:
            local = fields(local_root)
            remote = fields(remote_root)
    report = (
        correctness_comparison_report(local, remote)
        if arguments.mode == "generated-bootstrap-correctness"
        else comparison_report(local, remote)
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
