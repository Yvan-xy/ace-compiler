from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "phantom_gpu"


def text(name: str) -> str:
    return (TOOLS / name).read_text(encoding="utf-8")


def test_shell_entrypoints_are_syntactically_valid() -> None:
    paths = [
        TOOLS / "package_runpod_sources.sh",
        TOOLS / "run_local_reproduction.sh",
        TOOLS / "run_build_and_health.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, paths)], check=True)


def test_package_binds_dependency_sources_and_replays_host_oracles() -> None:
    source = text("package_runpod_sources.sh")
    branch = source.index(
        'if [[ "${PACKAGE_MODE}" == "generated-bootstrap-correctness" ]]'
    )
    assert source.index("SELECTED_DEPENDENCY_LOCK") < branch
    assert source.index("SELECTED_PHANTOM_COMMIT") < branch
    assert "host_oracle_replay.json:correctness-compiler-host-replay.json" in source
    replay = source.index("bootstrap_correctness.py\" verify-host")
    payload = source.index(
        '"ace.phantom.generated-bootstrap-correctness-payload/1.0.0"'
    )
    assert replay < payload < branch
    assert '"source_snapshots"' in source
    assert '"files": files' in source


def test_runner_preserves_public_modes_and_orders_the_correctness_gate() -> None:
    source = text("run_build_and_health.sh")
    assert "--mode <freeze-host|local|runpod>" in source
    assert "generated-bootstrap-correctness-payload/1.0.0" in source
    phases = [
        "correctness_payload_validation",
        "packaged_host_oracle_replay",
        "correctness_build",
        "regenerated_host_oracle_replay",
        "generated_bootstrap_gpu_correctness",
        "correctness_result_completeness",
    ]
    offsets = [source.rindex(f"phase {phase}") for phase in phases]
    assert offsets == sorted(offsets)
    assert "compute-sanitizer --error-exitcode" in source
    assert "bootstrap-compute-sanitizer/1.0.0" in source
    assert "generated-bootstrap-three-way-comparison.json" in source
    assert "correctness-build-host-qualification.json" in source
    assert "generated-bootstrap-gpu-skipped/1.0.0" in source
    assert "host_oracles_replayed_before_gpu:true" in source
    assert 'find "${INPUT}" -mindepth 1 -maxdepth 1 -type f -print0' in source
    completeness = source.index(
        "verify_generated_bootstrap_correctness_completeness()"
    )
    correctness_branch = source.index(
        'if jq -e \'', completeness
    )
    writer = source.index(
        '>"${RESULT_DIR}/result-completeness.json"', completeness
    )
    assert writer < correctness_branch
    assert '"ace.phantom.result-completeness/1.0.0"' in source[
        completeness:correctness_branch
    ]


def test_rebuilt_logs_are_retained_and_required_before_archive_closure() -> None:
    source = text("run_build_and_health.sh")
    build_start = source.index("run_generated_bootstrap_correctness_build()")
    build_end = source.index(
        "replay_regenerated_bootstrap_host_oracles()", build_start
    )
    build = source[build_start:build_end]
    completeness_start = source.index(
        "verify_generated_bootstrap_correctness_completeness()"
    )
    completeness_end = source.index("\nphase payload_verification", completeness_start)
    completeness = source[completeness_start:completeness_end]
    retained = {
        "native-ant.stdout.txt": "correctness-native-ant.stdout.txt",
        "native-ant.stderr.txt": "correctness-native-ant.stderr.txt",
        "generated-ant.stdout.txt": "correctness-generated-ant.stdout.txt",
        "generated-ant.stderr.txt": "correctness-generated-ant.stderr.txt",
        "link-commands.txt": "correctness-link-commands.txt",
        "configuration.json": "correctness-configuration.json",
        "environment.txt": "correctness-environment.txt",
        "gpu-runner-inspection/cuda_elf.txt": (
            "correctness-gpu-runner-cuda-elf.txt"
        ),
        "gpu-runner-inspection/cuda_resources.txt": (
            "correctness-gpu-runner-cuda-resources.txt"
        ),
        "gpu-runner-inspection/file.txt": "correctness-gpu-runner-file.txt",
        "gpu-runner-inspection/readelf.txt": (
            "correctness-gpu-runner-readelf.txt"
        ),
        "gpu-runner-inspection/undefined_symbols.txt": (
            "correctness-gpu-runner-undefined-symbols.txt"
        ),
    }
    for generated, archived in retained.items():
        assert f'"{generated}:{archived}"' in build
        assert archived in completeness
    assert "correctness-build-artifact-manifest.json" in completeness
    assert "correctness-build-host-qualification.json" in completeness


def test_local_replay_keeps_legacy_identity_and_uses_public_local_mode() -> None:
    source = text("run_local_reproduction.sh")
    assert 'TASK_LABEL="retained-ckks-local-reproduction"' in source
    assert 'CONTAINER_NAME="ace-phantom-retained-local-${RUN_ID}"' in source
    assert "bash /retained-qualification/input/run_build_and_health.sh" in source
    assert "--mode local" in source
    assert "generated-bootstrap-correctness" in source


def test_remote_transport_verifies_a_checksum_closed_correctness_archive(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive-root"
    results = root / "results"
    nested = results / "correctness-input"
    nested.mkdir(parents=True)
    records = {
        results / "pipeline-result.json": {
            "schema_version": "1.0.0",
            "status": "pass",
            "mode": "runpod",
            "exit_code": 0,
            "started_utc": "2026-08-09T00:00:00Z",
            "completed_utc": "2026-08-09T00:00:01Z",
        },
        results / "result-completeness.json": {
            "schema_version": "ace.phantom.result-completeness/1.0.0",
            "status": "pass",
            "mode": "runpod",
        },
        results / "generated-bootstrap-correctness-lifecycle.json": {
            "schema_version": (
                "ace.phantom.generated-bootstrap-correctness-lifecycle/1.0.0"
            ),
            "status": "pass",
            "host_oracles_replayed_before_gpu": True,
            "gpu_execution": "pass",
        },
        nested / "payload.json": {
            "schema_version": (
                "ace.phantom.generated-bootstrap-correctness-payload/1.0.0"
            ),
            "status": "pass",
        },
    }
    for path, value in records.items():
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    entries = []
    for path in sorted(records):
        entries.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(results)}\n"
        )
    (results / "SHA256SUMS").write_text("".join(entries), encoding="utf-8")
    archive = tmp_path / "runpod-result.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(results, arcname="results")
    archive_receipt = tmp_path / "runpod-result.tar.gz.sha256"
    archive_receipt.write_text(
        f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
        encoding="utf-8",
    )
    command = (
        'set -euo pipefail; cd "$1"; sha256sum -c "$2"; '
        'source "$3"; verify_result_archive "$4" runpod 0 "$5"'
    )
    completed = subprocess.run(
        [
            "bash", "-c", command, "correctness-transport",
            str(tmp_path), archive_receipt.name,
            str(TOOLS / "transport_helpers.sh"), str(archive),
            str(tmp_path / "verified"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["status"] == "pass"
    assert report["mode"] == "runpod"
    assert report["verified_file_count"] == len(records)
    assert (
        tmp_path / "verified/results/correctness-input/payload.json"
    ).is_file()

    transfer = text("runpod_transfer.sh")
    outer = transfer.index("sha256sum -c runpod-result.tar.gz.sha256")
    inner = transfer.index("verify_result_archive", outer)
    success = transfer.index("RunPod result retrieved and verified", inner)
    assert outer < inner < success


def test_environment_comparison_has_a_strict_correctness_mode() -> None:
    path = TOOLS / "compare_environments.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'choices=("full", "generated-bootstrap-correctness")' in source
    assert "verify_result_checksum_closure" in source
    assert "correctness-gpu-harness.cu" in source
    assert "run_build_and_health.sh" in source
    assert "per_run_gpu_executable_sha256" in source
    assert "per_run_native_ant_executable_sha256" in source
    assert "per_run_generated_ant_executable_sha256" in source
    assert "per_run_adapter_archive_sha256" in source
    assert "per_run_provider_archive_sha256" in source
    assert "per_run_common_archive_sha256" in source
    assert "gpu_execution\": \"required-remotely-and-skipped-locally" in source


def test_environment_comparison_rejects_host_executable_substitution() -> None:
    path = TOOLS / "compare_environments.py"
    spec = importlib.util.spec_from_file_location("compare_environments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native_path = (
        "bootstrap_qualification/generated_bootstrap_native_ant_oracle"
    )
    generated_path = (
        "bootstrap_qualification/generated_bootstrap_dsl_ant_oracle"
    )
    build_files = {native_path: "a" * 64, generated_path: "b" * 64}
    native_record = {
        "execution": {"provenance": {"executable_sha256": "a" * 64}}
    }
    generated_record = {
        "execution": {"provenance": {"executable_sha256": "b" * 64}}
    }
    assert module.correctness_host_executable_digests(
        build_files, native_record, generated_record
    ) == ("a" * 64, "b" * 64)
    generated_record["execution"]["provenance"]["executable_sha256"] = "c" * 64
    with pytest.raises(SystemExit, match="differs from rebuilt provenance"):
        module.correctness_host_executable_digests(
            build_files, native_record, generated_record
        )


def test_compiler_options_are_authorities_not_embedded_profiles() -> None:
    for name in (
        "package_runpod_sources.sh",
        "run_build_and_health.sh",
        "compare_environments.py",
    ):
        source = text(name)
        assert "expected_parameters =" not in source
        assert 'generation_options["--poly-degree"]' in source
        if name != "compare_environments.py":
            assert 'generation_options["--vector-capacity"]' in source
            assert 'generation_options["--packing"]' in source


def test_full_environment_report_remains_callable() -> None:
    path = TOOLS / "compare_environments.py"
    spec = importlib.util.spec_from_file_location("compare_environments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {
        "bootstrap_adapter_archive_sha256": "a",
        "bootstrap_common_archive_sha256": "b",
        "bootstrap_frozen_reference_sha256": "c",
        "bootstrap_io_helper_closure_sha256": "d",
        "bootstrap_linked_binary_sha256": "e",
        "bootstrap_provider_archive_sha256": "f",
        "bootstrap_symbol_closure_sha256": "g",
        "generated_binary_sha256": "h",
        "stable": "same",
    }
    report = module.comparison_report(values, dict(values))
    assert report["status"] == "pass"
    assert report["mismatches"] == {}
