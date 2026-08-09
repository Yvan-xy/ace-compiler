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


def bootstrap_evidence(root: Path, ace_commit: str, phantom_commit: str) -> tuple[str, str]:
    evidence = root / "bootstrap-qualification"
    output = evidence / "bootstrap_qualification"
    ace_source = {"schema_version": "1.0.0", "kind": "ace", "commit": ace_commit}
    phantom_source = {
        "schema_version": "1.0.0",
        "kind": "phantom",
        "commit": phantom_commit,
    }
    write_json(evidence / "ace_source_manifest.json", ace_source)
    write_json(evidence / "phantom_source_manifest.json", phantom_source)
    ace_source_sha = hashlib.sha256(
        (evidence / "ace_source_manifest.json").read_bytes()
    ).hexdigest()
    phantom_source_sha = hashlib.sha256(
        (evidence / "phantom_source_manifest.json").read_bytes()
    ).hexdigest()

    parameters = [
        "--poly-degree", "16384", "--mul-level", "26",
        "--input-level", "1", "--security-level", "0",
        "--scaling-factor-bits", "56", "--first-prime-bits", "60",
        "--hamming-weight", "192",
    ]
    qualification_argv = [
        "tools/phantom_gpu/compile_only.sh", "--gate", "bootstrap",
    ] + parameters
    generation_argv = [
        "tools/phantom_gpu/generate_bootstrap_qualification.py",
        "bootstrap_qualification",
    ] + parameters
    qualification_invocation = {
        "schema_version": "ace.phantom.qualification-invocation/1.0.0",
        "argv": qualification_argv,
        "normalized_argv_sha256": hashlib.sha256(
            json.dumps(
                qualification_argv, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    generation_invocation = {
        "schema_version": (
            "ace.phantom.bootstrap-qualification-invocation/1.0.0"
        ),
        "argv": generation_argv,
        "normalized_argv_sha256": hashlib.sha256(
            json.dumps(
                generation_argv, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(evidence / "qualification_invocation.json", qualification_invocation)
    write_json(
        evidence / "bootstrap_generation_invocation.json", generation_invocation
    )

    context = {
        "schema_version": 1,
        "resource_schema_version": 3,
        "packing": "full",
        "polynomial_degree": 16384,
        "logical_slot_capacity": 8192,
        "data_q_bit_sizes": [60] + [56] * 25,
        "special_p_bit_sizes": [60, 60],
        "input_level": 1,
        "q_part_count": 3,
        "hamming_weight": 192,
        "security_level": 0,
        "first_modulus_bits": 60,
        "scaling_modulus_bits": 56,
    }
    resources = {
        "schema_version": 3,
        "context_schema_version": 1,
        "relinearization_key": True,
        "rotation_steps": [1],
        "conjugation_key": True,
        "rotate_batch": True,
        "rotation_batches": [[1]],
        "raise_mod": True,
        "monomial_powers": [1],
        "complex_plaintext": True,
        "native_bootstrap_precompute": False,
    }
    write_json(output / "compiler_context_manifest.json", context)
    write_json(output / "compiler_resource_manifest.json", resources)
    context_sha = hashlib.sha256(
        (output / "compiler_context_manifest.json").read_bytes()
    ).hexdigest()
    constants = {
        "schema_version": 1,
        "context_schema_version": 1,
        "resource_schema_version": 3,
        "context_manifest_sha256": context_sha,
        "constants": [{"entry_id": 0}],
    }
    write_json(output / "compiler_constant_manifest.json", constants)
    (output / "bootstrap_qualification.cu").write_text(
        "int generated_bootstrap_source;\n", encoding="utf-8"
    )
    (output / "bootstrap_phantom_constants.cu").write_text(
        "int bootstrap_harness;\n", encoding="utf-8"
    )
    (output / "bootstrap_phantom_constants_sm80").write_bytes(b"binary")
    hashes = {
        name: hashlib.sha256((output / path).read_bytes()).hexdigest()
        for name, path in {
            "context": "compiler_context_manifest.json",
            "resource": "compiler_resource_manifest.json",
            "constant": "compiler_constant_manifest.json",
            "source": "bootstrap_qualification.cu",
            "harness": "bootstrap_phantom_constants.cu",
            "binary": "bootstrap_phantom_constants_sm80",
        }.items()
    }
    generation = {
        "schema_version": "ace.phantom.bootstrap-generation/1.0.0",
        "status": "pass",
        "constant_count": 1,
        "source": {"sha256": hashes["source"]},
        "manifests": {
            name: {"sha256": hashes[name]}
            for name in ("context", "resource", "constant")
        },
    }
    write_json(output / "generation.json", generation)
    audit = {
        "status": "pass",
        "counts": {"constants": 1},
        "inputs": {
            "context_manifest": {"sha256": hashes["context"]},
            "resource_manifest": {"sha256": hashes["resource"]},
            "constant_manifest": {"sha256": hashes["constant"]},
            "source": {"sha256": hashes["source"]},
        },
    }
    write_json(output / "source-audit.json", audit)
    symbol_closure = {
        "schema_version": "ace.phantom.bootstrap-symbol-closure/1.0.0",
        "status": "pass",
        "missing_symbols": [],
        "native_bootstrap_symbol_count": 0,
    }
    io_helper_closure = {
        "schema_version": "ace.phantom.bootstrap-io-helper-closure/1.0.0",
        "status": "pass",
        "generated_source_sha256": hashes["source"],
        "harness_source_sha256": hashes["harness"],
    }
    archive_audit = {
        "schema_version": "ace.phantom.production-archive-members/1.0.0",
        "status": "pass",
        "forbidden_member_count": 0,
        "forbidden_members": [],
    }
    write_json(output / "symbol-closure.json", symbol_closure)
    write_json(output / "io-helper-closure.json", io_helper_closure)
    write_json(output / "archive-member-audit.json", archive_audit)
    hashes.update(
        {
            name: hashlib.sha256((output / path).read_bytes()).hexdigest()
            for name, path in {
                "generation": "generation.json",
                "audit": "source-audit.json",
                "symbol": "symbol-closure.json",
                "io": "io-helper-closure.json",
                "archive_audit": "archive-member-audit.json",
            }.items()
        }
    )
    qualification = {
        "status": "pass",
        "gate": "bootstrap",
        "architecture": "sm_80",
        "phantom_commit": phantom_commit,
        "generated_source_sha256": hashes["source"],
        "compiler_context_manifest_sha256": hashes["context"],
        "compiler_resource_manifest_sha256": hashes["resource"],
        "compiler_constant_manifest_sha256": hashes["constant"],
        "generation_record_sha256": hashes["generation"],
        "generated_artifact_audit_sha256": hashes["audit"],
        "linked_binary_sha256": hashes["binary"],
        "harness_source_sha256": hashes["harness"],
        "symbol_closure_sha256": hashes["symbol"],
        "io_helper_closure_sha256": hashes["io"],
        "archive_member_audit_sha256": hashes["archive_audit"],
        "adapter_archive_sha256": digest("a"),
        "provider_archive_sha256": digest("b"),
        "common_archive_sha256": digest("c"),
        "generated_source_contains_native_bootstrap": False,
        "production_archive_contains_native_bootstrap": False,
        "primitive_only_provider_archive": True,
        "executable_was_run": False,
    }
    write_json(output / "qualification.json", qualification)
    qualification_sha = hashlib.sha256(
        (output / "qualification.json").read_bytes()
    ).hexdigest()
    qualification_invocation_sha = hashlib.sha256(
        (evidence / "qualification_invocation.json").read_bytes()
    ).hexdigest()
    generation_invocation_sha = hashlib.sha256(
        (evidence / "bootstrap_generation_invocation.json").read_bytes()
    ).hexdigest()
    artifact_files = {
        "ace_source_manifest.json": ace_source_sha,
        "phantom_source_manifest.json": phantom_source_sha,
        "qualification_invocation.json": qualification_invocation_sha,
        "bootstrap_generation_invocation.json": generation_invocation_sha,
        **{
            f"bootstrap_qualification/{path}": hashlib.sha256(
                (output / path).read_bytes()
            ).hexdigest()
            for path in (
                "compiler_context_manifest.json",
                "compiler_resource_manifest.json",
                "compiler_constant_manifest.json",
                "generation.json",
                "source-audit.json",
                "bootstrap_qualification.cu",
                "bootstrap_phantom_constants.cu",
                "bootstrap_phantom_constants_sm80",
                "symbol-closure.json",
                "io-helper-closure.json",
                "archive-member-audit.json",
                "qualification.json",
            )
        },
    }
    artifact = {
        "schema_version": "ace.phantom.bootstrap-artifacts/1.0.0",
        "status": "bound",
        "source_mode": "snapshot",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "normalized_qualification_argv_sha256": qualification_invocation[
            "normalized_argv_sha256"
        ],
        "normalized_generation_argv_sha256": generation_invocation[
            "normalized_argv_sha256"
        ],
        "compiler_context_manifest_sha256": hashes["context"],
        "compiler_resource_manifest_sha256": hashes["resource"],
        "compiler_constant_manifest_sha256": hashes["constant"],
        "generation_record_sha256": hashes["generation"],
        "generated_artifact_audit_sha256": hashes["audit"],
        "linked_binary_sha256": hashes["binary"],
        "harness_source_sha256": hashes["harness"],
        "files": artifact_files,
    }
    write_json(evidence / "artifact_manifest.json", artifact)
    run = {
        "status": "pass",
        "gate": "bootstrap",
        "source_mode": "snapshot",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_worktree_dirty": False,
        "gpu_executables_were_run": False,
        "ace_tracked_source_manifest_sha256": ace_source_sha,
        "phantom_source_manifest_sha256": phantom_source_sha,
        "qualification_invocation_sha256": qualification_invocation_sha,
        "normalized_qualification_argv_sha256": qualification_invocation[
            "normalized_argv_sha256"
        ],
        "compiler_context_manifest_sha256": hashes["context"],
        "compiler_resource_manifest_sha256": hashes["resource"],
        "compiler_constant_manifest_sha256": hashes["constant"],
        "bootstrap_generation_record_sha256": hashes["generation"],
        "bootstrap_generated_artifact_audit_sha256": hashes["audit"],
        "bootstrap_linked_binary_sha256": hashes["binary"],
        "bootstrap_harness_source_sha256": hashes["harness"],
        "bootstrap_host_qualification_sha256": qualification_sha,
        "artifact_manifest_sha256": hashlib.sha256(
            (evidence / "artifact_manifest.json").read_bytes()
        ).hexdigest(),
    }
    write_json(evidence / "manifest.json", run)
    write_json(
        root / "bootstrap-frozen-reference.json",
        {
            "schema_version": "ace.phantom.bootstrap-frozen-reference/1.0.0",
            "status": "pass",
            "comparison": "deterministic-host-artifact-hashes",
            "artifact_manifest_sha256": hashlib.sha256(
                (evidence / "artifact_manifest.json").read_bytes()
            ).hexdigest(),
            "host_qualification_sha256": qualification_sha,
            "source_audit_sha256": hashes["audit"],
            "expected_generated_source_sha256": hashes["source"],
            "expected_harness_source_sha256": hashes["harness"],
            "expected_linked_binary_sha256": hashes["binary"],
        },
    )
    return ace_source_sha, phantom_source_sha


def result_root(base: Path, records: dict[str, dict]) -> Path:
    root = base / "results"
    ace_commit = "a" * 40
    phantom_commit = "b" * 40
    ace_source_sha, phantom_source_sha = bootstrap_evidence(
        root, ace_commit, phantom_commit
    )
    for record in records.values():
        if "ace_source_manifest_sha256" in record:
            record["ace_source_manifest_sha256"] = ace_source_sha
        if "phantom_source_manifest_sha256" in record:
            record["phantom_source_manifest_sha256"] = phantom_source_sha
    write_json(
        root / "qualification/ckks2c/configuration.json",
        {
            "environment_identity": {
                "base_image": "image@sha256:base",
                "config_digest": "sha256:config",
                "bootstrap_sha256": digest("f"),
            },
            "ace": {"commit": ace_commit},
            "phantom": {"commit": phantom_commit},
            "source_manifest_sha256": ace_source_sha,
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
    assert report["comparison_contract"]["bootstrap_field_count"] > 15
    assert "retained_exact_artifacts" in report["matching_fields"]
    assert "retained_ant_semantic_summary_sha256" in report["matching_fields"]
    assert "bootstrap_linked_binary_sha256" in report["matching_fields"]
    assert "bootstrap_io_helper_closure_sha256" in report["matching_fields"]
    assert (
        "bootstrap_frozen_reference_sha256"
        in report["allowed_differences"]
    )
    assert "setup timing" in " ".join(
        report["allowed_differences"]["bootstrap_per_run_evidence"]
    )
    assert report["mismatches"] == {}


def test_bootstrap_stable_archive_hash_tamper_is_reported(tmp_path: Path) -> None:
    module = load_module()
    local = result_root(tmp_path / "local", retained_records("a", "b"))
    remote = result_root(tmp_path / "remote", retained_records("a", "b"))
    qualification_path = (
        remote
        / "bootstrap-qualification/bootstrap_qualification/qualification.json"
    )
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["provider_archive_sha256"] = digest("0")
    write_json(qualification_path, qualification)
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    artifact_path = remote / "bootstrap-qualification/artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["files"][
        "bootstrap_qualification/qualification.json"
    ] = qualification_sha
    write_json(artifact_path, artifact)
    run_path = remote / "bootstrap-qualification/manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["bootstrap_host_qualification_sha256"] = qualification_sha
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    run["artifact_manifest_sha256"] = artifact_sha
    write_json(run_path, run)
    frozen_path = remote / "bootstrap-frozen-reference.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["artifact_manifest_sha256"] = artifact_sha
    frozen["host_qualification_sha256"] = qualification_sha
    write_json(frozen_path, frozen)

    report = module.comparison_report(module.fields(local), module.fields(remote))

    assert report["status"] == "fail"
    assert set(report["mismatches"]) == {"bootstrap_provider_archive_sha256"}


def test_bootstrap_per_run_frozen_reference_hash_may_differ(tmp_path: Path) -> None:
    module = load_module()
    local = result_root(tmp_path / "local", retained_records("a", "b"))
    remote = result_root(tmp_path / "remote", retained_records("a", "b"))
    frozen_path = remote / "bootstrap-frozen-reference.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["per_run_note"] = "different extraction root"
    write_json(frozen_path, frozen)

    report = module.comparison_report(module.fields(local), module.fields(remote))

    assert report["status"] == "pass"
    assert report["mismatches"] == {}
    difference = report["allowed_differences"][
        "bootstrap_frozen_reference_sha256"
    ]
    assert difference["local"] != difference["remote"]


def test_bootstrap_cross_record_inconsistency_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    root = result_root(tmp_path, retained_records("a", "b"))
    artifact_path = root / "bootstrap-qualification/artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["compiler_constant_manifest_sha256"] = digest("0")
    write_json(artifact_path, artifact)
    run_path = root / "bootstrap-qualification/manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["artifact_manifest_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    write_json(run_path, run)

    with pytest.raises(SystemExit, match="artifact manifest"):
        module.fields(root)


def test_bootstrap_invocation_tamper_is_rejected(tmp_path: Path) -> None:
    module = load_module()
    root = result_root(tmp_path, retained_records("a", "b"))
    invocation_path = (
        root / "bootstrap-qualification/bootstrap_generation_invocation.json"
    )
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation["argv"][-1] = "191"
    write_json(invocation_path, invocation)

    with pytest.raises(SystemExit, match="generation invocation"):
        module.fields(root)


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
