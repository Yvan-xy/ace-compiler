from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest


TOOLS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "source_archive", TOOLS / "source_archive.py"
)
assert SPEC and SPEC.loader
SOURCE_ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE_ARCHIVE)


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Pipeline Test")
    git(repo, "config", "user.email", "pipeline@example.invalid")
    tracked = repo / "tools/phantom_gpu"
    tracked.mkdir(parents=True)
    (tracked / "script.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (repo / "fhe-cmplr/rtlib/ant/dataset").mkdir(parents=True)
    (repo / "fhe-cmplr/rtlib/ant/dataset/private.onnx").write_bytes(b"model")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "test tree")
    return repo, git(repo, "rev-parse", "HEAD")


def create(repo: Path, commit: str, output: Path) -> tuple[Path, Path]:
    archive = output / "ace.tar.gz"
    manifest = output / "ace.json"
    arguments = type(
        "Arguments",
        (),
        {
            "repo": repo,
            "commit": commit,
            "kind": "ace",
            "output": archive,
            "manifest": manifest,
        },
    )()
    SOURCE_ARCHIVE.create_archive(arguments)
    return archive, manifest


def test_commit_archive_is_deterministic_and_excludes_dirty_data(tmp_path: Path) -> None:
    repo, commit = repository(tmp_path)
    dirty = repo / "tools/phantom_gpu/untracked-secret.pem"
    dirty.write_text("not uploaded", encoding="utf-8")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_archive, first_manifest = create(repo, commit, first)
    second_archive, second_manifest = create(repo, commit, second)

    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
    paths = {record["path"] for record in manifest["members"]}
    assert "ace-source/tools/phantom_gpu/script.sh" in paths
    assert not any("dataset" in path for path in paths)
    assert not any("untracked-secret.pem" in path for path in paths)


def test_audit_rejects_corruption(tmp_path: Path) -> None:
    repo, commit = repository(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    archive, manifest_path = create(repo, commit, output)
    damaged = output / "damaged.tar.gz"
    content = bytearray(archive.read_bytes())
    content[len(content) // 2] ^= 0x01
    damaged.write_bytes(content)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(SystemExit, match="outer archive SHA-256"):
        SOURCE_ARCHIVE.audit_archive(damaged, "ace", manifest)


def test_audit_rejects_escaping_symlink(tmp_path: Path) -> None:
    repo, _ = repository(tmp_path)
    link = repo / "tools/phantom_gpu/escape"
    link.symlink_to("../../../outside")
    git(repo, "add", "tools/phantom_gpu/escape")
    git(repo, "commit", "-qm", "escaping link")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(SystemExit, match="escaping symlink"):
        create(repo, git(repo, "rev-parse", "HEAD"), output)


def test_audit_rejects_sensitive_tracked_file(tmp_path: Path) -> None:
    repo, _ = repository(tmp_path)
    key = repo / "tools/phantom_gpu/id_ed25519"
    key.write_text("private", encoding="utf-8")
    git(repo, "add", "tools/phantom_gpu/id_ed25519")
    git(repo, "commit", "-qm", "sensitive path")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(SystemExit, match="sensitive filename"):
        create(repo, git(repo, "rev-parse", "HEAD"), output)


def test_timed_phase_stops_at_first_failure(tmp_path: Path) -> None:
    timings = tmp_path / "timings.tsv"
    marker = tmp_path / "continued"
    script = f"""
set -euo pipefail
source {TOOLS / 'phase_helpers.sh'}
fail_then_continue() {{
  false
  touch "$1"
}}
set +e
run_timed_phase "$1" deliberate_failure fail_then_continue "$2"
phase_result=$?
set -e
[[ ${{phase_result}} -ne 0 ]]
[[ ! -e "$2" ]]
"""
    subprocess.run(
        ["bash", "-c", script, "phase-test", str(timings), str(marker)],
        check=True,
    )
    fields = timings.read_text(encoding="utf-8").strip().split("\t")
    assert fields[0] == "deliberate_failure"
    assert fields[-1] != "0"


def test_remote_command_preserves_space_containing_gpu_name() -> None:
    expected = "NVIDIA A100-SXM4-80GB"
    script = r'''
set -euo pipefail
source "$1"
expected="$2"
remote_args=(
  env
  "ACE_RUNPOD_EXPECTED_GPU_NAME=${expected}"
  bash -c 'printf "%s" "${ACE_RUNPOD_EXPECTED_GPU_NAME}"'
)
shell_join remote_command "${remote_args[@]}"
bash -c "${remote_command}"
'''
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "transport-test",
            str(TOOLS / "transport_helpers.sh"),
            expected,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == expected


def result_archive(
    tmp_path: Path,
    *,
    recorded_data_digest: str | None = None,
    include_nested_checksum: bool = False,
) -> Path:
    root = tmp_path / "archive-root"
    results = root / "results"
    results.mkdir(parents=True)
    data = results / "data.txt"
    data.write_text("qualified\n", encoding="utf-8")
    pipeline = results / "pipeline-result.json"
    pipeline.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "pass",
                "mode": "local",
                "exit_code": 0,
                "started_utc": "2026-08-07T00:00:00Z",
                "completed_utc": "2026-08-07T00:00:01Z",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    covered_paths = [data, pipeline]
    if include_nested_checksum:
        nested_data = results / "qualification" / "inner.txt"
        nested_data.parent.mkdir()
        nested_data.write_bytes(b"inner")
        nested_checksum = results / "qualification" / "SHA256SUMS"
        nested_checksum.write_text(
            f"{hashlib.sha256(b'inner').hexdigest()}  inner.txt\n",
            encoding="utf-8",
        )
        covered_paths.extend((nested_data, nested_checksum))
    entries = []
    for path in covered_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path == data and recorded_data_digest is not None:
            digest = recorded_data_digest
        entries.append(f"{digest}  {path.relative_to(results)}\n")
    (results / "SHA256SUMS").write_text("".join(entries), encoding="utf-8")
    archive = tmp_path / "result.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(results, arcname="results")
    return archive


def run_result_archive_verifier(
    archive: Path, extraction: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; '
            'verify_result_archive "$2" local 0 "$3"',
            "result-verifier",
            str(TOOLS / "transport_helpers.sh"),
            str(archive),
            str(extraction),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_result_archive_verifier_checks_every_internal_entry(
    tmp_path: Path,
) -> None:
    archive = result_archive(tmp_path)
    result = run_result_archive_verifier(archive, tmp_path / "verified")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "pass"
    assert report["verified_file_count"] == 2


def test_result_archive_verifier_rejects_internal_hash_mismatch(
    tmp_path: Path,
) -> None:
    archive = result_archive(tmp_path, recorded_data_digest="0" * 64)
    result = run_result_archive_verifier(archive, tmp_path / "rejected")
    assert result.returncode != 0
    assert "incomplete or a result file hash mismatched" in result.stderr


def test_result_archive_covers_nested_qualification_checksum(
    tmp_path: Path,
) -> None:
    archive = result_archive(tmp_path, include_nested_checksum=True)
    result = run_result_archive_verifier(archive, tmp_path / "verified")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified_file_count"] == 4

    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "find . -type f ! -path ./SHA256SUMS -print0" in source
    assert "find . -type f ! -name SHA256SUMS -print0" not in source


def test_remote_pipeline_uses_the_packaged_frozen_cpu_reference() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert (
        "phase source_audit_and_extraction extract_sources\n"
        "configure_qualification_environment\n"
        "phase qualification"
    ) in source
    assert 'cmp "${generated_context}" "${INPUT}/ordinary-context-manifest.json"' in source
    assert 'cmp "${generated_resources}" "${INPUT}/ordinary-resource-manifest.json"' in source
    assert 'cpu_reference="${INPUT}/ordinary-cpu-reference.json"' in source
    assert 'cpu_values="${INPUT}/ordinary-cpu-values.bin"' in source
    assert 'fixture="${INPUT}/ordinary-fixture.json"' in source
    assert '"${qualification_arguments[@]}"' in source
    assert ".compiler_context_options" not in source
    assert 'cmp "${run_root}/compiler_invocation.json"' in source
    assert '"${INPUT}/compiler-invocation.json"' in source
    assert 'cmp "${ordinary_dir}/ordinary_ckks_cpu_reference.json"' not in source
    assert '"${ordinary_dir}/ordinary_ckks_ant_verification.json"' in source


def test_failed_qualification_captures_partial_run_before_returning() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    capture_start = source.index("capture_failed_qualification() {")
    qualification_start = source.index("run_qualification() {")
    capture = source[capture_start:qualification_start]
    qualification_end = source.index(
        "\nverify_frozen_ordinary_reference()", qualification_start
    )
    qualification = source[qualification_start:qualification_end]

    assert (
        'cp -a -- "${canonical_run_root}" "${RESULT_DIR}/qualification"'
        in capture
    )
    assert "qualification-failure-files.sha256" in capture
    assert "ace.phantom.failed-qualification-evidence/1.0.0" in capture
    assert '"${canonical_runs_root}"/*' in capture
    assert 'current_gate}" != "ordinary"' in capture
    assert 'current_status}" != "failed"' in capture
    assert 'current_exit}" -ne "${qualification_exit}"' in capture

    compile_call = qualification.index(
        'bash "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compile_only.sh"'
    )
    status_capture = qualification.index(
        'qualification_status=("${PIPESTATUS[@]}")', compile_call
    )
    failure_capture = qualification.index(
        'capture_failed_qualification "${current}" "${qualification_exit}"',
        status_capture,
    )
    failure_return = qualification.index(
        'return "${qualification_exit}"', failure_capture
    )
    tee_failure = qualification.index(
        'qualification log capture failed with exit ${qualification_status[1]}',
        failure_return,
    )
    success_copy = qualification.index(
        'cp -a "${run_root}" "${RESULT_DIR}/qualification"', tee_failure
    )
    assert (
        compile_call
        < status_capture
        < failure_capture
        < failure_return
        < tee_failure
        < success_copy
    )


def test_gpu_conformance_keeps_wrapper_addresses_unique() -> None:
    source = (TOOLS / "harness/ordinary_ckks_gpu_runner.cu").read_text(
        encoding="utf-8"
    )
    start = source.index("Json RunConformance(const Json& fixture)")
    end = source.index("double MaximumError", start)
    body = source[start:end]

    assert body.count("ObjectArena arena;") == 1
    assert body.index("ObjectArena arena;") < body.index(
        'for (const auto& family : fixture.at("case_families"))'
    )
    assert "ObjectArena arena;\n      CaseResult result" not in body


def test_source_packaging_requires_local_ordinary_evidence() -> None:
    source = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    assert "--ordinary-run-root" in source
    assert "--poly-degree)" not in source
    assert "--mul-level)" not in source
    assert "compiler-invocation.json" in source
    assert "qualification-invocation.json" in source
    assert "normalized_argv_sha256" in source
    assert 'run_manifest.get("source_mode") != "snapshot"' in source
    assert "frozen SHA256SUMS does not enumerate the complete run root" in source
    assert "frozen evidence ACE producer does not match" in source
    assert "frozen CPU/ANT producer does not match" in source
    assert "ordinary-context-manifest.json" in source
    assert "ordinary-resource-manifest.json" in source
    assert "ordinary-cpu-reference.json" in source
    assert "ordinary-cpu-values.bin" in source
    assert "ordinary-ant-verification.json" in source
    assert "ordinary-post-ckks.air" in source
    assert "ordinary-run-manifest.json" in source
    assert "ordinary-host-qualification.json" in source
    assert "ordinary-artifact-manifest.json" in source


def test_local_reproduction_does_not_restate_compiler_context() -> None:
    source = (TOOLS / "run_local_reproduction.sh").read_text(encoding="utf-8")
    for option in (
        "--poly-degree",
        "--mul-level",
        "--input-level",
        "--security-level",
        "--scaling-factor-bits",
        "--first-prime-bits",
        "--hamming-weight",
    ):
        assert option not in source


def test_local_reproduction_uses_a_distinct_container_work_root() -> None:
    local = (TOOLS / "run_local_reproduction.sh").read_text(encoding="utf-8")
    host_freeze = (TOOLS / "freeze_ordinary_host_evidence.sh").read_text(
        encoding="utf-8"
    )

    assert '"${PAYLOAD}:/ordinary-replay/input:ro"' in local
    assert '"${OUTPUT}:/ordinary-replay/output:rw"' in local
    assert "--input-dir /ordinary-replay/input" in local
    assert "--work-dir /ordinary-replay/output/work" in local
    assert (
        "--result-archive /ordinary-replay/output/local-result.tar.gz" in local
    )
    assert "/ordinary-replay/" not in host_freeze
    assert "--work-dir /workspace/output/work" in host_freeze


def test_native_health_uses_the_exact_current_ordinary_run() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    start = source.index("run_native_health() {")
    end = source.index("\nphase payload_verification", start)
    body = source[start:end]

    assert (
        'current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/'
        'current-ordinary.json"' in body
    )
    assert "ordinary:pass" in body
    assert 'run_root="$(jq -er \'.run_root\' "${current}")"' in body
    assert (
        'health_binary="${run_root}/ckks2c/native_phantom_health_sm80"' in body
    )
    assert (
        'context_manifest="${run_root}/ckks2c/'
        'compiler_context_manifest.json"' in body
    )
    assert "find " not in body


def test_host_freeze_runner_returns_fresh_candidate_without_frozen_checks() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "<freeze-host|local|runpod>" in source
    assert (
        'if [[ "${MODE}" != "freeze-host" ]]; then\n'
        "  phase frozen_ordinary_reference verify_frozen_ordinary_reference"
    ) in source
    assert 'if [[ "${MODE}" == "freeze-host" ]]; then' in source
    assert "ace.phantom.host-freeze-candidate/1.0.0" in source
    assert 'cmp "${run_root}/qualification_invocation.json"' in source
    assert 'phase qualification run_qualification' in source


def test_host_freeze_payload_is_source_only_and_commit_addressed() -> None:
    script = TOOLS / "package_host_freeze_sources.sh"
    source = script.read_text(encoding="utf-8")
    assert os.access(script, os.X_OK)
    assert "--qualification-invocation" in source
    assert "source_archive.py\" create" in source
    assert '--kind ace' in source
    assert '--kind phantom' in source
    assert "ace.phantom.host-freeze-payload/1.0.0" in source
    assert "audited-source-snapshots-and-qualification-invocation-only" in source
    assert "does not implement the host-freeze runner mode" in source
    assert "--ordinary-run-root" not in source
    assert "ordinary-cpu-reference" not in source
    assert "compiler-invocation.json" not in source
    assert "source commits must be full lowercase 40-character object IDs" in source


def test_host_freeze_wrapper_owns_and_cleans_only_its_exact_container() -> None:
    script = TOOLS / "freeze_ordinary_host_evidence.sh"
    source = script.read_text(encoding="utf-8")
    assert os.access(script, os.X_OK)
    assert "docker create --name" in source
    assert 'docker start -a "${CONTAINER_ID}"' in source
    assert 'docker rm -f "${exact_id}"' in source
    assert 'docker inspect --type container "${exact_id}"' in source
    assert "removed-and-absent" in source
    assert '--label "ace.phantom.task=${TASK_LABEL}"' in source
    assert '--mode freeze-host' in source
    assert (
        '"${BASE_IMAGE}" \\\n'
        "  bash /workspace/input/run_build_and_health.sh"
    ) in source
    for forbidden in (
        "docker run",
        "docker image tag",
        "docker image rm",
        "docker ps",
        "docker system prune",
        "ace-compiler-dev",
    ):
        assert forbidden not in source


def test_host_freeze_candidate_comes_only_from_verified_archive() -> None:
    source = (TOOLS / "freeze_ordinary_host_evidence.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'VERIFIED_RESULTS="${OUTPUT}/verified-host-freeze-result/results"'
        in source
    )
    assert 'CANDIDATE="${VERIFIED_RESULTS}/qualification"' in source
    assert '"${VERIFIED_RESULTS}/host-freeze-candidate.json"' in source
    assert '${OUTPUT}/work/results/qualification' not in source
    assert '${OUTPUT}/work/results/host-freeze-candidate.json' not in source
    assert "diff --no-dereference --recursive" not in source
    assert "live host-freeze candidate" not in source
    assert "jq -e" not in source
    assert "object_pairs_hook=reject_duplicates" in source
    assert "hashlib.sha256(" in source


def test_host_freeze_packager_rejects_missing_required_arguments() -> None:
    result = subprocess.run(
        [str(TOOLS / "package_host_freeze_sources.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--qualification-invocation FILE" in result.stderr
