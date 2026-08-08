"""Retained-aware local versus RunPod evidence comparison contracts."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "compare_environments.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_environments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(character: str) -> str:
    return character * 64


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def retained_records(receipt: str, executable: str) -> dict[str, dict]:
    ace_commit = "a" * 40
    phantom_commit = "b" * 40
    ace_source = digest("1")
    phantom_source = digest("2")
    context = digest("3")
    resource = digest("4")
    fixture = digest("5")
    generation = digest("6")
    summary = {
        "schema_version": "ace.phantom.retained_ckks.ant-semantic-summary/1.0.0",
        "status": "pass",
        "fixture_sha256": fixture,
        "context_manifest_sha256": context,
        "record_order": ["conjugate.input"],
        "metamorphic_checks": {"conjugate_twice_identity": True},
    }
    summary_sha256 = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    exact_artifacts = {
        "context_manifest": context,
        "resource_manifest": resource,
        "fixture": fixture,
        "generation_attestation": generation,
        "phantom_post_ckks_air": digest("d"),
        "production_post_ckks_air": digest("e"),
    }
    replay = {
        "schema_version": "ace.phantom.retained_ckks.local-replay/1.0.0",
        "status": "pass",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_source_manifest_sha256": ace_source,
        "phantom_source_manifest_sha256": phantom_source,
        "host_manifest_sha256": digest(receipt),
        "artifact_manifest_sha256": digest(receipt),
        "build_attestation_sha256": digest(receipt),
        "sha256_manifest_sha256": digest(receipt),
        "context_manifest_sha256": context,
        "resource_manifest_sha256": resource,
        "fixture_sha256": fixture,
        "generation_attestation_sha256": generation,
        "exact_artifact_count": len(exact_artifacts),
        "provider_neutral_ant_reference_matches": True,
        "ant_replay": {
            "status": "pass",
            "comparison": "semantic-summary-only-no-decoded-byte-comparison",
            "summary": summary,
            "summary_sha256": summary_sha256,
        },
        "artifacts": exact_artifacts,
    }
    host = {
        "schema_version": "ace.phantom.retained_ckks.host-qualification/1.0.0",
        "status": "pass",
        "gate": "retained_ckks",
        "source_mode": "snapshot",
        "fixture_lifecycle": "checked-bound",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_worktree_dirty": False,
        "ace_source_manifest_sha256": ace_source,
        "phantom_source_manifest_sha256": phantom_source,
        "compiler_context_manifest_sha256": context,
        "compiler_resource_manifest_sha256": resource,
        "fixture_sha256": fixture,
        "cpu_reference_sha256": digest(receipt),
        "cpu_values_sha256": digest(receipt),
        "build_attestation_sha256": digest(receipt),
        "artifact_manifest_sha256": digest(receipt),
        "evidence_sha256_manifest_sha256": digest(receipt),
        "host_ant_oracle_was_run": True,
        "gpu_executables_were_run": False,
    }
    build = {
        "schema_version": "ace.phantom.retained_ckks.build-attestation/1.0.0",
        "status": "pass",
        "architecture": "sm_80",
        "source_mode": "snapshot",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_source_manifest_sha256": ace_source,
        "phantom_source_manifest_sha256": phantom_source,
        "compiler_context_manifest_sha256": context,
        "compiler_resource_manifest_sha256": resource,
        "fixture_sha256": fixture,
        "compiler_invocation_sha256": generation,
        "generated_ant_source_sha256": digest("8"),
        "generated_phantom_source_sha256": digest("9"),
        "host_ant_oracle_was_run": True,
        "gpu_executables_were_run": False,
        "archives": {"adapter": digest(executable)},
        "executables": {"phantom_sm80": digest(executable)},
    }
    artifact = {
        "schema_version": "ace.phantom.retained_ckks.artifact-manifest/1.0.0",
        "status": "bound",
        "fixture_lifecycle": "checked-bound",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_source_manifest_sha256": ace_source,
        "phantom_source_manifest_sha256": phantom_source,
        "compiler_context_manifest_sha256": context,
        "compiler_resource_manifest_sha256": resource,
        "normalized_compiler_command_sha256": digest("c"),
        "post_ckks_air_sha256": digest("d"),
        "production_post_ckks_air_sha256": digest("e"),
        "fixture_sha256": fixture,
        "build_attestation_sha256": digest(receipt),
        "files": {"build/retained_ckks_phantom_sm80": digest(executable)},
    }
    return {"replay": replay, "host": host, "build": build, "artifact": artifact}


def result_root(base: Path, records: dict[str, dict]) -> Path:
    root = base / "results"
    write_json(
        root / "qualification/ckks2c/configuration.json",
        {
            "environment_identity": {
                "base_image": "image@sha256:base",
                "config_digest": "sha256:config",
                "bootstrap_sha256": digest("f"),
            },
            "ace": {"commit": "a" * 40},
            "phantom": {"commit": "b" * 40},
            "source_manifest_sha256": digest("1"),
            "cuda_architecture": "80",
            "toolchain": {"versions": {"nvcc": "12.4"}},
            "compiler_context_manifest": {"sha256": digest("3")},
        },
    )
    write_json(root / "ace-source-audit.json", {"archive_sha256": digest("a")})
    write_json(root / "phantom-source-audit.json", {"archive_sha256": digest("b")})
    write_json(
        root / "qualification/ckks2c/qualification.json",
        {
            "source_sha256": digest("8"),
            "terminal_selection_sha256": digest("9"),
            "binary_sha256": digest("0"),
        },
    )
    write_json(
        root / "ordinary-frozen-reference.json",
        {
            "context_manifest_sha256": digest("3"),
            "resource_manifest_sha256": digest("4"),
            "fixture_sha256": digest("5"),
            "cpu_reference_sha256": digest("6"),
            "cpu_values_sha256": digest("7"),
            "ant_verification_sha256": digest("8"),
        },
    )
    environment = root / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    for name in (
        "apt-packages.lock",
        "python-requirements-hashed.lock",
        "base-files.sha256",
        "dpkg-manifest.txt",
        "python-manifest.txt",
    ):
        (environment / name).write_text(name + "\n", encoding="utf-8")
    write_json(root / "retained-frozen-reference.json", records["replay"])
    write_json(root / "retained-host/manifest.json", records["host"])
    write_json(root / "retained-host/build-attestation.json", records["build"])
    write_json(root / "retained-host/artifact-manifest.json", records["artifact"])
    return root


def archive(root: Path, output: Path) -> Path:
    with tarfile.open(output, "w:gz") as target:
        target.add(root, arcname="results")
    return output


def run_comparison(local: Path, remote: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--local",
            str(local),
            "--remote",
            str(remote),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_retained_stable_fields_match_while_per_run_receipts_differ(
    tmp_path: Path,
) -> None:
    module = load_module()
    local = result_root(tmp_path / "local", retained_records("a", "b"))
    remote = result_root(tmp_path / "remote", retained_records("f", "0"))
    report = module.comparison_report(module.fields(local), module.fields(remote))

    assert report["schema_version"] == module.REPORT_SCHEMA
    assert report["status"] == "pass"
    assert report["comparison_contract"]["retained_field_count"] > 10
    assert "retained_exact_artifacts" in report["matching_fields"]
    assert "retained_ant_semantic_summary_sha256" in report["matching_fields"]
    assert report["mismatches"] == {}


@pytest.mark.parametrize(
    "mutation",
    (
        lambda records: records["replay"].__setitem__("status", "fail"),
        lambda records: records["replay"].__setitem__(
            "provider_neutral_ant_reference_matches", False
        ),
        lambda records: records["replay"]["ant_replay"].__setitem__(
            "comparison", "decoded-byte-comparison"
        ),
        lambda records: records["replay"]["ant_replay"].__setitem__(
            "summary_sha256", digest("0")
        ),
        lambda records: records["replay"].update(
            {"artifacts": {}, "exact_artifact_count": 0}
        ),
    ),
)
def test_invalid_retained_replay_is_rejected(tmp_path: Path, mutation) -> None:
    module = load_module()
    records = retained_records("a", "b")
    mutation(records)
    root = result_root(tmp_path, records)
    with pytest.raises(SystemExit):
        module.fields(root)


def test_internally_inconsistent_retained_bindings_are_rejected(tmp_path: Path) -> None:
    module = load_module()
    records = retained_records("a", "b")
    records["build"]["compiler_context_manifest_sha256"] = digest("0")
    root = result_root(tmp_path, records)
    with pytest.raises(SystemExit, match="internally inconsistent"):
        module.fields(root)


def test_archive_cli_reports_a_retained_mismatch(tmp_path: Path) -> None:
    local_records = retained_records("a", "b")
    remote_records = copy.deepcopy(local_records)
    remote_records["artifact"]["post_ckks_air_sha256"] = digest("0")
    remote_records["replay"]["artifacts"]["phantom_post_ckks_air"] = digest("0")
    local_root = result_root(tmp_path / "local", local_records)
    remote_root = result_root(tmp_path / "remote", remote_records)
    local_archive = archive(local_root, tmp_path / "local.tar.gz")
    remote_archive = archive(remote_root, tmp_path / "remote.tar.gz")
    output = tmp_path / "comparison.json"

    result = run_comparison(local_archive, remote_archive, output)

    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "ace.phantom.environment-comparison/2.0.0"
    assert report["status"] == "fail"
    assert set(report["mismatches"]) == {
        "retained_exact_artifacts",
        "retained_post_ckks_air_sha256",
    }
