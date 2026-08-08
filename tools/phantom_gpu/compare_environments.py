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
            raise SystemExit(f"{label} {name} violates the retained comparison contract")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        **retained_fields(root),
    }
    if compared["retained_ace_commit"] != compared["ace_commit"]:
        raise SystemExit("ordinary and retained ACE commits are inconsistent")
    if compared["retained_phantom_commit"] != compared["phantom_commit"]:
        raise SystemExit("ordinary and retained Phantom commits are inconsistent")
    if (
        compared["retained_ace_source_manifest_sha256"]
        != compared["source_manifest_sha256"]
    ):
        raise SystemExit("ordinary and retained ACE source manifests are inconsistent")
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
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass" if not mismatches else "fail",
        "comparison_contract": {
            "stable_field_count": len(stable),
            "retained_field_count": len(retained),
            "retained_fields": retained,
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
