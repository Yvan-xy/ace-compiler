#!/usr/bin/env python3
"""Compare stable local and RunPod pipeline evidence fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile
from typing import Any


REPORT_SCHEMA = "ace.phantom.environment-comparison/2.0.0"
RETAINED_REPLAY_SCHEMA = "ace.phantom.retained_ckks.local-replay/1.0.0"
RETAINED_HOST_SCHEMA = "ace.phantom.retained_ckks.host-qualification/1.0.0"
RETAINED_BUILD_SCHEMA = "ace.phantom.retained_ckks.build-attestation/1.0.0"
RETAINED_ARTIFACT_SCHEMA = "ace.phantom.retained_ckks.artifact-manifest/1.0.0"
RETAINED_SUMMARY_SCHEMA = "ace.phantom.retained_ckks.ant-semantic-summary/1.0.0"
SEMANTIC_REPLAY_MODE = "semantic-summary-only-no-decoded-byte-comparison"
BOOTSTRAP_ARTIFACT_SCHEMA = "ace.phantom.bootstrap-artifacts/1.0.0"
BOOTSTRAP_GENERATION_SCHEMA = "ace.phantom.bootstrap-generation/1.0.0"
BOOTSTRAP_FROZEN_REFERENCE_SCHEMA = (
    "ace.phantom.bootstrap-frozen-reference/1.0.0"
)
BOOTSTRAP_QUALIFICATION_INVOCATION_SCHEMA = (
    "ace.phantom.qualification-invocation/1.0.0"
)
BOOTSTRAP_GENERATION_INVOCATION_SCHEMA = (
    "ace.phantom.bootstrap-qualification-invocation/1.0.0"
)
BOOTSTRAP_SYMBOL_CLOSURE_SCHEMA = (
    "ace.phantom.bootstrap-symbol-closure/1.0.0"
)
BOOTSTRAP_IO_HELPER_CLOSURE_SCHEMA = (
    "ace.phantom.bootstrap-io-helper-closure/1.0.0"
)
ARCHIVE_MEMBER_AUDIT_SCHEMA = (
    "ace.phantom.production-archive-members/1.0.0"
)


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


def require_fields(record: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for name, value in expected.items():
        if record.get(name) != value:
            raise SystemExit(f"{label} {name} violates the comparison contract")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_invocation(
    path: Path, schema: str, expected_argv: list[str], label: str
) -> tuple[str, str]:
    record = read_json(path)
    if set(record) != {"schema_version", "argv", "normalized_argv_sha256"}:
        raise SystemExit(f"{label} has an invalid shape")
    require_fields(
        record,
        {"schema_version": schema, "argv": expected_argv},
        label,
    )
    normalized = require_digest(
        record.get("normalized_argv_sha256"), f"{label} normalized argv"
    )
    if normalized != canonical_json_sha256(expected_argv):
        raise SystemExit(f"{label} normalized argv hash is inconsistent")
    return sha256(path), normalized


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
    io_helper_closure_path = output / "io-helper-closure.json"
    io_helper_closure = read_json(io_helper_closure_path)
    archive_audit_path = output / "archive-member-audit.json"
    archive_audit = read_json(archive_audit_path)
    context_path = output / "compiler_context_manifest.json"
    context = read_json(context_path)
    resource_path = output / "compiler_resource_manifest.json"
    resource = read_json(resource_path)
    constant_path = output / "compiler_constant_manifest.json"
    constants = read_json(constant_path)
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

    parameters = [
        "--poly-degree", "16384", "--mul-level", "26",
        "--input-level", "1", "--security-level", "0",
        "--scaling-factor-bits", "56", "--first-prime-bits", "60",
        "--hamming-weight", "192",
    ]
    qualification_invocation_path = evidence / "qualification_invocation.json"
    qualification_invocation_sha, normalized_qualification = validate_invocation(
        qualification_invocation_path,
        BOOTSTRAP_QUALIFICATION_INVOCATION_SCHEMA,
        ["tools/phantom_gpu/compile_only.sh", "--gate", "bootstrap"] + parameters,
        "bootstrap qualification invocation",
    )
    generation_invocation_path = evidence / "bootstrap_generation_invocation.json"
    generation_invocation_sha, normalized_generation = validate_invocation(
        generation_invocation_path,
        BOOTSTRAP_GENERATION_INVOCATION_SCHEMA,
        [
            "tools/phantom_gpu/generate_bootstrap_qualification.py",
            "bootstrap_qualification",
        ] + parameters,
        "bootstrap generation invocation",
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
            "polynomial_degree": 16384,
            "logical_slot_capacity": 8192,
            "packing": "full",
            "input_level": 1,
            "security_level": 0,
            "scaling_modulus_bits": 56,
            "first_modulus_bits": 60,
            "hamming_weight": 192,
        },
        "bootstrap context manifest",
    )
    if len(context.get("data_q_bit_sizes", [])) != 26:
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

    require_fields(
        generation,
        {
            "schema_version": BOOTSTRAP_GENERATION_SCHEMA,
            "status": "pass",
            "constant_count": len(constant_entries),
        },
        "bootstrap generation record",
    )
    if generation.get("source", {}).get("sha256") != source_sha:
        raise SystemExit("bootstrap generation source hash is inconsistent")
    for name, value in (
        ("context", context_sha), ("resource", resource_sha),
        ("constant", constant_sha),
    ):
        if generation.get("manifests", {}).get(name, {}).get("sha256") != value:
            raise SystemExit(f"bootstrap generation {name} hash is inconsistent")

    require_fields(audit, {"status": "pass"}, "bootstrap source audit")
    for name, value in (
        ("context_manifest", context_sha), ("resource_manifest", resource_sha),
        ("constant_manifest", constant_sha), ("source", source_sha),
    ):
        if audit.get("inputs", {}).get(name, {}).get("sha256") != value:
            raise SystemExit(f"bootstrap source audit {name} hash is inconsistent")
    if audit.get("counts", {}).get("constants") != len(constant_entries):
        raise SystemExit("bootstrap source audit constant count is inconsistent")

    require_fields(
        symbol_closure,
        {
            "schema_version": BOOTSTRAP_SYMBOL_CLOSURE_SCHEMA,
            "status": "pass",
            "missing_symbols": [],
            "native_bootstrap_symbol_count": 0,
        },
        "bootstrap symbol closure",
    )
    require_fields(
        io_helper_closure,
        {"schema_version": BOOTSTRAP_IO_HELPER_CLOSURE_SCHEMA, "status": "pass"},
        "bootstrap I/O-helper closure",
    )
    if (
        io_helper_closure.get("generated_source_sha256") != source_sha
        or io_helper_closure.get("harness_source_sha256") != harness_sha
    ):
        raise SystemExit("bootstrap I/O-helper closure source hashes are inconsistent")
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
        "bootstrap_qualification/compiler_context_manifest.json": context_sha,
        "bootstrap_qualification/compiler_resource_manifest.json": resource_sha,
        "bootstrap_qualification/compiler_constant_manifest.json": constant_sha,
        "bootstrap_qualification/generation.json": generation_sha,
        "bootstrap_qualification/source-audit.json": audit_sha,
        "bootstrap_qualification/bootstrap_qualification.cu": source_sha,
        "bootstrap_qualification/bootstrap_phantom_constants.cu": harness_sha,
        "bootstrap_qualification/bootstrap_phantom_constants_sm80": binary_sha,
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
            "bootstrap_linked_binary_sha256": binary_sha,
            "bootstrap_harness_source_sha256": harness_sha,
            "bootstrap_host_qualification_sha256": qualification_sha,
            "artifact_manifest_sha256": sha256(artifact_path),
        },
        "bootstrap run manifest",
    )
    require_fields(
        frozen_reference,
        {
            "schema_version": BOOTSTRAP_FROZEN_REFERENCE_SCHEMA,
            "status": "pass",
            "comparison": "deterministic-host-artifact-hashes",
            "artifact_manifest_sha256": sha256(artifact_path),
            "host_qualification_sha256": qualification_sha,
            "source_audit_sha256": audit_sha,
            "expected_generated_source_sha256": source_sha,
            "expected_harness_source_sha256": harness_sha,
            "expected_linked_binary_sha256": binary_sha,
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
        "bootstrap_generated_source_sha256": source_sha,
        "bootstrap_harness_source_sha256": harness_sha,
        "bootstrap_linked_binary_sha256": binary_sha,
        "bootstrap_archive_member_audit_sha256": archive_audit_sha,
        "bootstrap_symbol_closure_sha256": symbol_closure_sha,
        "bootstrap_io_helper_closure_sha256": io_helper_closure_sha,
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
    stable = sorted(set(local) - {"generated_binary_sha256"})
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
                "reason": "recorded but not required equal because tool output can embed build-host details",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ace-environment-compare-") as temporary:
        base = Path(temporary)
        local = fields(safe_extract(arguments.local, base / "local"))
        remote = fields(safe_extract(arguments.remote, base / "remote"))
    report = comparison_report(local, remote)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
