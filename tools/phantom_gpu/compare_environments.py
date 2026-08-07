#!/usr/bin/env python3
"""Compare stable local and RunPod pipeline evidence fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any


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
    return json.loads(path.read_text(encoding="utf-8"))


def fields(root: Path) -> dict[str, Any]:
    configuration = read_json(root / "qualification/ckks2c/configuration.json")
    ace_audit = read_json(root / "ace-source-audit.json")
    phantom_audit = read_json(root / "phantom-source-audit.json")
    qualification = read_json(root / "qualification/ckks2c/qualification.json")
    frozen_reference = read_json(root / "ordinary-frozen-reference.json")
    return {
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
    stable = sorted(set(local) - {"generated_binary_sha256"})
    matching = [key for key in stable if local[key] == remote[key]]
    mismatches = {
        key: {"local": local[key], "remote": remote[key]}
        for key in stable
        if local[key] != remote[key]
    }
    report = {
        "schema_version": "1.0.0",
        "status": "pass" if not mismatches else "fail",
        "matching_fields": matching,
        "mismatches": mismatches,
        "allowed_differences": {
            "generated_binary_sha256": {
                "local": local["generated_binary_sha256"],
                "remote": remote["generated_binary_sha256"],
                "reason": "recorded but not required equal because tool output can embed build-host details",
            },
            "remote_only": [
                "driver",
                "gpu_uuid",
                "pod_id",
                "timestamps",
                "hostname",
                "kernel",
            ],
        },
    }
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
