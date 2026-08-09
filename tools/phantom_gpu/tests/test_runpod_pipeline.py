from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
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

PHANTOM_REQUIRED_TEST_PATHS = {
    "tests/CMakeLists.txt",
    "tests/ckks_retained_primitives.cu",
    "tests/compiler_context_manifest.h",
    "tests/native_bts_oracle_link.cu",
}
BOOTSTRAP_REQUIRED_SYMBOLS = [
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
]


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


def phantom_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "phantom"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Pipeline Test")
    git(repo, "config", "user.email", "pipeline@example.invalid")
    tracked = {
        "CMakeLists.txt": "add_subdirectory(src)\nadd_subdirectory(tests)\n",
        "cmake/Dependencies.cmake": "# build dependencies\n",
        "include/phantom.h": "#pragma once\n",
        "src/phantom.cu": "// Phantom implementation\n",
        "tests/CMakeLists.txt": "# retained test targets\n",
        "tests/ckks_retained_primitives.cu": "// retained gate\n",
        "tests/compiler_context_manifest.h": "#pragma once\n",
        "tests/native_bts_oracle_link.cu": "// configure-source closure\n",
        "tests/add.cu": "// unrelated test\n",
        "tests/._ckks_retained_primitives.cu": "AppleDouble\n",
        "outputs/generated.cu": "// generated output\n",
        "models/checkpoint.onnx": "model data\n",
    }
    for relative, content in tracked.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "phantom test tree")
    return repo, git(repo, "rev-parse", "HEAD")


def create_phantom(repo: Path, commit: str, output: Path) -> tuple[Path, Path]:
    archive = output / "phantom.tar.gz"
    manifest = output / "phantom.json"
    arguments = type(
        "Arguments",
        (),
        {
            "repo": repo,
            "commit": commit,
            "kind": "phantom",
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


def test_phantom_archive_includes_only_retained_test_source_closure(
    tmp_path: Path,
) -> None:
    repo, commit = phantom_repository(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    archive, manifest_path = create_phantom(repo, commit, output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member_paths = {record["path"] for record in manifest["members"]}
    required_members = {
        f"phantom-source/{path}" for path in PHANTOM_REQUIRED_TEST_PATHS
    }
    archived_test_files = {
        path
        for path in member_paths
        if path.startswith("phantom-source/tests/")
    }

    assert PHANTOM_REQUIRED_TEST_PATHS <= set(manifest["allowed_paths"])
    assert archived_test_files == required_members
    assert manifest["member_count"] == len(manifest["members"])
    with tarfile.open(archive, "r:gz") as source:
        archive_paths = {member.name.rstrip("/") for member in source.getmembers()}
    assert archive_paths == member_paths
    assert not any("._" in path for path in member_paths)
    assert not any("outputs" in path or "models" in path for path in member_paths)
    SOURCE_ARCHIVE.audit_archive(archive, "phantom", manifest)


@pytest.mark.parametrize(
    ("member", "message"),
    (
        ("phantom-source/tests/add.cu", "outside the phantom allowlist"),
        ("phantom-source/tests/._CMakeLists.txt", "outside the phantom allowlist"),
        ("phantom-source/outputs/generated.cu", "outside the phantom allowlist"),
        ("phantom-source/src/models/checkpoint.cu", "disallowed source path"),
        ("phantom-source/include/._phantom.h", "AppleDouble file"),
        ("phantom-source/src/checkpoint.onnx", "sensitive or data file"),
        ("phantom-source/include/id_ed25519", "sensitive filename"),
    ),
)
def test_phantom_policy_rejects_non_source_members(member: str, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        SOURCE_ARCHIVE.policy_path(PurePosixPath(member), "phantom")


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
    include_completeness: bool = True,
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
    if include_completeness:
        completeness = results / "result-completeness.json"
        completeness.write_text(
            json.dumps(
                {
                    "schema_version": "ace.phantom.result-completeness/1.0.0",
                    "status": "pass",
                    "mode": "local",
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        covered_paths.append(completeness)
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
    assert report["verified_file_count"] == 3


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
    assert json.loads(result.stdout)["verified_file_count"] == 5

    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "find . -type f ! -path ./SHA256SUMS -print0" in source
    assert "find . -type f ! -name SHA256SUMS -print0" not in source


def test_result_archive_verifier_requires_success_completeness(
    tmp_path: Path,
) -> None:
    archive = result_archive(tmp_path, include_completeness=False)
    result = run_result_archive_verifier(archive, tmp_path / "rejected")
    assert result.returncode != 0
    assert "lacks its completeness record" in result.stderr


def test_remote_pipeline_uses_the_packaged_frozen_cpu_reference() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert (
        "phase source_audit_and_extraction extract_sources\n"
        "CURRENT_PHASE=qualification_environment\n"
        "configure_qualification_environment\n"
        'CURRENT_PHASE=""\n'
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


def test_gpu_ownership_uses_explicit_json_and_dedicated_sanitizer_log() -> None:
    runner = (TOOLS / "harness/ordinary_ckks_gpu_runner.cu").read_text(
        encoding="utf-8"
    )
    ownership_start = runner.index("Json RunOwnership(const Json& fixture)")
    ownership_end = runner.index("void RunRejection", ownership_start)
    ownership_body = runner[ownership_start:ownership_end]
    main_start = runner.index("int main(int argc, char** argv)")
    main = runner[main_start:]
    assert "std::cout" not in ownership_body
    assert 'Fail("ownership mode requires an output path")' in main
    assert "WriteJson(argv[4], RunOwnership(fixture));" in main

    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    start = pipeline.index("run_ordinary_gpu_qualification() {")
    end = pipeline.index("\nrun_native_health() {", start)
    body = pipeline[start:end]
    assert 'compute-sanitizer --version >"${sanitizer_version}" 2>&1' in body
    assert '--log-file "${sanitizer_log}"' in body
    assert '"${ownership_json}" >"${sanitizer_stdout}"' in body
    assert '2>"${sanitizer_stderr}"' in body
    assert "tail -n 1" not in body
    assert "ordinary_ckks_sanitizer.stdout.txt" in body
    assert "ordinary_ckks_sanitizer.stderr.txt" in body
    assert "ordinary_ckks_sanitizer.log.txt" in body
    assert "validate-ownership" in body


def test_nonzero_sanitizer_exit_precedes_evidence_acceptance() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    start = source.index("run_ordinary_gpu_qualification() {")
    end = source.index("\nrun_native_health() {", start)
    body = source[start:end]
    exit_capture = body.index("local sanitizer_exit=$?")
    nonzero_gate = body.index("if [[ ${sanitizer_exit} -ne 0 ]]", exit_capture)
    exit_return = body.index('return "${sanitizer_exit}"', nonzero_gate)
    summary_gate = body.index("zero_summary_count=", exit_return)
    ownership_gate = body.index("validate-ownership", summary_gate)
    assert (
        exit_capture
        < nonzero_gate
        < exit_return
        < summary_gate
        < ownership_gate
    )


def test_source_packaging_requires_local_ordinary_evidence() -> None:
    source = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    assert "--ordinary-run-root" in source
    assert "--retained-run-root" in source
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


def test_retained_replay_payload_descriptor_matches_packager_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    if shutil.which("jq") is None:
        pytest.skip("jq is required to exercise the retained payload predicate")

    package = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    package_start = package.index('"contents": (') + len('"contents": (')
    package_end = package.index("\n    ),", package_start)
    package_descriptor = "".join(
        line.strip()[1:-1]
        for line in package[package_start:package_end].splitlines()
        if line.strip().startswith('"')
    )
    expected_descriptor = (
        "audited-source-snapshots-and-frozen-provider-neutral-references-"
        "with-exact-bootstrap-compile-handoff-and-regeneration-attestations"
    )
    assert package_descriptor == expected_descriptor

    function_start = pipeline.index("verify_frozen_retained_reference() {")
    function_end = pipeline.index("\nrun_retained_gpu_qualification()", function_start)
    verifier = pipeline[function_start:function_end]
    assert expected_descriptor in verifier

    input_directory = tmp_path / "input"
    result_directory = tmp_path / "results"
    input_directory.mkdir()
    result_directory.mkdir()
    payload_path = input_directory / "payload.json"
    payload = {
        "contents": expected_descriptor,
        "frozen_retained_reference": {
            "status": "pass",
            "contents": (
                "provider-neutral-references-and-attestations-no-build-output"
            ),
        },
    }
    payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (input_directory / "ace-source.manifest.json").write_text(
        '{"commit":"' + "a" * 40 + '"}\n', encoding="utf-8"
    )
    (input_directory / "phantom-source.manifest.json").write_text(
        '{"commit":"' + "b" * 40 + '"}\n', encoding="utf-8"
    )
    harness = (
        "set -euo pipefail\n"
        'INPUT="$1"\nRESULT_DIR="$2"\n'
        'RETAINED_HOST_ROOT="$3"\nACE_PHANTOM_REPO_ROOT="$4"\n'
        "cmp() { :; }\n"
        "python3() {\n"
        "  printf '%s\\n' "
        "'{\"status\":\"pass\",\"provider_neutral_ant_reference_matches\":true,'"
        "'\"exact_artifact_count\":1}'\n"
        "}\n"
        + verifier
        + "\nverify_frozen_retained_reference\n"
    )
    arguments = [
        "bash",
        "-c",
        harness,
        "retained-descriptor-contract",
        str(input_directory),
        str(result_directory),
        str(tmp_path / "retained"),
        str(tmp_path / "ace"),
    ]
    subprocess.run(arguments, check=True)

    payload["contents"] = (
        "audited-source-snapshots-and-frozen-provider-neutral-references-"
        "without-build-output"
    )
    payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    rejected = subprocess.run(arguments, check=False)
    assert rejected.returncode != 0


def test_bootstrap_compile_handoff_is_complete_and_checksum_bound() -> None:
    package = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    handoff_files = (
        "bootstrap-raw.air",
        "bootstrap-post-ckks.air",
        "bootstrap-generated.cu",
        "bootstrap-host-linked-sm80",
        "bootstrap-link-commands.txt",
        "bootstrap-cuda-elf.txt",
        "bootstrap-cuda-resources.txt",
        "bootstrap-file-report.txt",
        "bootstrap-readelf.txt",
        "bootstrap-linked-symbols.txt",
        "bootstrap-symbol-closure.json",
    )
    for name in handoff_files:
        assert name in package
        assert name in pipeline
    assert '"schema_version": "2.0.0"' in package
    assert '"architecture": "sm_80"' in package
    assert '"packaged_compile_handoff": True' in package
    assert '"linked_binary_byte_identity_required": False' in package
    assert '"${output}/link-commands.txt"' not in pipeline[
        pipeline.index("verify_frozen_bootstrap_qualification() {") :
        pipeline.index("verify_frozen_ordinary_reference() {")
    ]


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


def test_reproduction_uses_one_canonical_container_build_root() -> None:
    local = (TOOLS / "run_local_reproduction.sh").read_text(encoding="utf-8")
    remote = (TOOLS / "runpod_transfer.sh").read_text(encoding="utf-8")
    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    bootstrap_freeze = (
        TOOLS / "freeze_bootstrap_host_evidence.sh"
    ).read_text(encoding="utf-8")
    host_freeze = (TOOLS / "freeze_ordinary_host_evidence.sh").read_text(
        encoding="utf-8"
    )

    assert '"${PAYLOAD}:/retained-qualification/input:ro"' in local
    assert '"${OUTPUT}:/retained-qualification/output:rw"' in local
    assert "--input-dir /retained-qualification/input" in local
    assert "--work-dir /retained-qualification/work" in local
    assert "--work-dir /retained-qualification/work" in remote
    assert "WORK=/retained-qualification/work" in bootstrap_freeze
    assert 'ACE_PHANTOM_STATE_ROOT="${WORK}/state"' in pipeline
    assert 'ACE_PHANTOM_STATE_ROOT="${WORK}/build-state"' not in pipeline
    assert "/retained-qualification/output/work" not in local
    assert "WORK=${OUTPUT}/work" not in bootstrap_freeze
    assert (
        "--result-archive /retained-qualification/output/local-result.tar.gz"
        in local
    )
    assert "/retained-qualification/" not in host_freeze
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


def test_bootstrap_qualification_is_frozen_rebuilt_and_executed_once() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    host_start = source.index("run_bootstrap_host_qualification() {")
    host_end = source.index("\nverify_frozen_bootstrap_qualification()", host_start)
    host = source[host_start:host_end]
    gpu_start = source.index("run_bootstrap_constant_cache_qualification() {")
    gpu_end = source.index("\nrun_native_health() {", gpu_start)
    gpu = source[gpu_start:gpu_end]

    assert "bootstrap-qualification-invocation.json" in host
    assert "jq -er '.argv[1:][]'" in host
    assert 'mapfile -t arguments <<<"${arguments_output}"' in host
    assert "current-bootstrap.json" in host
    assert "bootstrap-generation-invocation.json" in host
    assert host.count('compile_only.sh"') == 1

    invocation = 'timeout 1800 "${binary}" "${raw}"'
    assert gpu.count(invocation) == 1
    assert "while " not in gpu
    assert "for required" not in gpu
    assert '"logical_device_bytes"' in gpu
    assert "expected_logical_bytes" in gpu
    assert '"invocation_count": 1' in gpu
    assert "bootstrap-setup-metrics/1.0.0" in gpu

    phases = source[source.index("phase payload_verification") :]
    assert phases.index("phase bootstrap_payload_validation") < phases.index(
        "phase bootstrap_host_qualification"
    ) < phases.index("phase frozen_bootstrap_qualification")
    assert phases.index("phase native_a100_health") < phases.index(
        "phase bootstrap_constant_cache_qualification"
    ) < phases.index("phase ordinary_gpu_qualification")


def test_bootstrap_freeze_requires_exact_source_but_allows_fresh_cuda_bytes(
    tmp_path: Path,
) -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    function_start = source.index("verify_frozen_bootstrap_qualification() {")
    heredoc_start = source.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    heredoc_end = source.index("\nPY\n", heredoc_start)
    verifier = source[heredoc_start:heredoc_end]

    stable_paths = (
        "bootstrap_qualification/bootstrap_raw.air",
        "bootstrap_qualification/bootstrap_post_ckks.air",
        "bootstrap_qualification/bootstrap_qualification.cu",
        "bootstrap_qualification/bootstrap_phantom_constants.cu",
        "bootstrap_qualification/compiler_context_manifest.json",
        "bootstrap_qualification/compiler_resource_manifest.json",
        "bootstrap_qualification/compiler_constant_manifest.json",
        "bootstrap_qualification/generation.json",
        "bootstrap_qualification/source-audit.json",
    )
    binary_path = "bootstrap_qualification/bootstrap_phantom_constants_sm80"
    stable_hashes = {
        path: f"{index + 1:064x}" for index, path in enumerate(stable_paths)
    }
    packaged_binary = "a" * 64
    regenerated_binary_path = tmp_path / "bootstrap_phantom_constants_sm80"
    regenerated_binary_path.write_bytes(b"fresh CUDA compiler output")
    regenerated_binary = hashlib.sha256(
        regenerated_binary_path.read_bytes()
    ).hexdigest()
    payload = {
        "bootstrap_qualification": {
            "raw_air_sha256": stable_hashes[stable_paths[0]],
            "post_ckks_air_sha256": stable_hashes[stable_paths[1]],
            "generated_source_sha256": stable_hashes[stable_paths[2]],
            "harness_source_sha256": stable_hashes[stable_paths[3]],
            "host_linked_binary_sha256": packaged_binary,
            "host_qualification_sha256": "c" * 64,
            "source_audit_sha256": "d" * 64,
        }
    }
    frozen = {"files": dict(stable_hashes) | {binary_path: packaged_binary}}
    generated = {
        "files": dict(stable_hashes) | {binary_path: regenerated_binary}
    }
    qualification = {
        "architecture": "sm_80",
        "raw_air_sha256": stable_hashes[stable_paths[0]],
        "post_ckks_air_sha256": stable_hashes[stable_paths[1]],
        "generated_source_sha256": stable_hashes[stable_paths[2]],
        "linked_binary_sha256": regenerated_binary,
        "harness_source_sha256": stable_hashes[stable_paths[3]],
        "compiler_context_manifest_sha256": stable_hashes[stable_paths[4]],
        "compiler_resource_manifest_sha256": stable_hashes[stable_paths[5]],
        "compiler_constant_manifest_sha256": stable_hashes[stable_paths[6]],
        "generation_record_sha256": stable_hashes[stable_paths[7]],
        "generated_artifact_audit_sha256": stable_hashes[stable_paths[8]],
    }
    audit = {
        "schema_version": (
            "ace.phantom.bootstrap-generated-artifact-audit/2.0.0"
        ),
        "status": "pass",
        "counts": {"constants": 1},
    }
    symbol_closure = {
        "schema_version": "ace.phantom.bootstrap-symbol-closure/2.0.0",
        "status": "pass",
        "required_symbols": BOOTSTRAP_REQUIRED_SYMBOLS,
        "missing_symbols": [],
        "native_bootstrap_symbol_count": 0,
    }
    input_root = tmp_path / "input"
    regenerated_root = tmp_path / "regenerated"
    input_root.mkdir()
    regenerated_root.mkdir()
    (input_root / "payload.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    (input_root / "bootstrap-symbol-closure.json").write_text(
        json.dumps(symbol_closure) + "\n", encoding="utf-8"
    )
    (input_root / "bootstrap-cuda-elf.txt").write_text(
        "arch = sm_80\n", encoding="utf-8"
    )
    regenerated_symbol_path = regenerated_root / "symbol-closure.json"
    regenerated_symbol_path.write_text(
        json.dumps(symbol_closure) + "\n", encoding="utf-8"
    )
    (regenerated_root / "cuda_elf.txt").write_text(
        "arch = sm_80\n", encoding="utf-8"
    )
    inputs = {
        "frozen.json": frozen,
        "generated.json": generated,
        "qualification.json": qualification,
        "audit.json": audit,
    }
    for name, value in inputs.items():
        (tmp_path / name).write_text(
            json.dumps(value) + "\n", encoding="utf-8"
        )
    output = tmp_path / "reference.json"
    arguments = [
        "python3",
        "-c",
        verifier,
        str(input_root),
        str(regenerated_root),
        *(str(tmp_path / name) for name in inputs),
        str(regenerated_binary_path),
        str(output),
    ]

    subprocess.run(arguments, check=True)
    reference = json.loads(output.read_text(encoding="utf-8"))
    assert reference["source_and_setup_match"] is True
    assert reference["raw_air_matches"] is True
    assert reference["post_ckks_air_matches"] is True
    assert reference["generated_source_matches"] is True
    assert reference["architecture"] == "sm_80"
    assert reference["semantic_symbol_closure"] == symbol_closure
    assert reference["linked_binary_byte_identity_required"] is False
    assert reference["packaged_linked_binary_sha256"] == packaged_binary
    assert reference["regenerated_linked_binary_sha256"] == regenerated_binary

    regenerated_binary_path.write_bytes(b"post-manifest mutation")
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "linked binary bytes differ" in rejected.stderr
    regenerated_binary_path.write_bytes(b"fresh CUDA compiler output")

    frozen["files"][stable_paths[0]] = "e" * 64
    (tmp_path / "frozen.json").write_text(
        json.dumps(frozen) + "\n", encoding="utf-8"
    )
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "regenerated bootstrap artifact differs" in rejected.stderr

    frozen["files"][stable_paths[0]] = stable_hashes[stable_paths[0]]
    (tmp_path / "frozen.json").write_text(
        json.dumps(frozen) + "\n", encoding="utf-8"
    )
    tampered_symbol_closure = dict(symbol_closure)
    tampered_symbol_closure["missing_symbols"] = ["Conjugate_ciph"]
    regenerated_symbol_path.write_text(
        json.dumps(tampered_symbol_closure) + "\n", encoding="utf-8"
    )
    rejected = subprocess.run(arguments, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "semantic symbol closure differs" in rejected.stderr


def test_bootstrap_stable_symbol_inventories_are_root_independent(
    tmp_path: Path,
) -> None:
    compile_only = (TOOLS / "compile_only.sh").read_text(encoding="utf-8")
    bootstrap_start = compile_only.index("build_bootstrap_host_qualification() {")
    bootstrap_end = compile_only.index(
        "\nbuild_ordinary_conformance() {", bootstrap_start
    )
    bootstrap = compile_only[bootstrap_start:bootstrap_end]
    for target in ("source_object", "harness_object", "binary"):
        stable_command = 'nm -C --defined-only "$' + "{" + target + '}"'
        prefixed_command = 'nm -A -C --defined-only "$' + "{" + target + '}"'
        assert stable_command in bootstrap
        assert prefixed_command not in bootstrap

    source = tmp_path / "symbol.cc"
    source.write_text("int stable_symbol() { return 7; }\n", encoding="utf-8")
    local_object = tmp_path / "local" / "symbol.o"
    remote_object = tmp_path / "remote" / "symbol.o"
    local_object.parent.mkdir()
    remote_object.parent.mkdir()
    subprocess.run(
        ["c++", "-c", str(source), "-o", str(local_object)],
        check=True,
    )
    remote_object.write_bytes(local_object.read_bytes())

    def symbols(path: Path) -> str:
        return subprocess.run(
            ["nm", "-C", "--defined-only", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    local_symbols = symbols(local_object)
    remote_symbols = symbols(remote_object)
    assert local_symbols == remote_symbols
    assert str(local_object) not in local_symbols
    assert str(remote_object) not in remote_symbols


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


def test_bootstrap_uses_failure_preserving_timed_phases() -> None:
    source = (TOOLS / "bootstrap_environment.sh").read_text(encoding="utf-8")
    phase_start = source.index("phase() {")
    phase_end = source.index("\n}", phase_start) + 2
    phase_body = source[phase_start:phase_end]

    assert 'source "${SCRIPT_DIR}/phase_helpers.sh"' in source
    assert 'run_timed_phase "${TIMINGS}" "$@"' in phase_body
    assert "set +e" not in phase_body


def test_pipeline_phase_wrapper_stops_at_first_failure(tmp_path: Path) -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    phase_start = source.index("phase() {")
    phase_end = source.index("\n}\n\nverify_outer_payload", phase_start) + 2
    phase_function = source[phase_start:phase_end]
    timings = tmp_path / "timings.tsv"
    marker = tmp_path / "continued"
    script = (
        "set -euo pipefail\n"
        f"source {TOOLS / 'phase_helpers.sh'}\n"
        'TIMINGS="$1"\n'
        "CURRENT_PHASE=initialization\nFAILED_PHASE=\"\"\n"
        + phase_function
        + "\nfail_then_continue() { false; touch \"$1\"; }\n"
        + "set +e\nphase deliberate_failure fail_then_continue \"$2\"\n"
        + "phase_exit=$?\nset -e\n"
        + "[[ ${phase_exit} -ne 0 ]]\n[[ ! -e \"$2\" ]]\n"
        + "[[ ${FAILED_PHASE} == deliberate_failure ]]\n"
    )
    subprocess.run(
        ["bash", "-c", script, "phase-wrapper", str(timings), str(marker)],
        check=True,
    )
    fields = timings.read_text(encoding="utf-8").strip().split("\t")
    assert fields[0] == "deliberate_failure"
    assert fields[-1] != "0"


def test_shell_array_and_dependency_producers_have_checked_status() -> None:
    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    compile_only = (TOOLS / "compile_only.sh").read_text(encoding="utf-8")

    assert "< <(" not in pipeline
    assert "< <(" not in compile_only
    assert 'qualification_arguments_output="$(' in pipeline
    assert '-name .git -print0 |' in pipeline
    assert 'done <"${dependency_git_dirs}"' in pipeline
    assert 'checkout_commit="$(git -C "${checkout}" rev-parse HEAD)"' in pipeline
    assert 'contract_context_output="$(' in compile_only


def test_native_health_uses_an_explicit_json_artifact() -> None:
    pipeline = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    legacy = (TOOLS / "run_a100_health.sh").read_text(encoding="utf-8")
    harness = (TOOLS / "harness/native_phantom_health.cu").read_text(
        encoding="utf-8"
    )

    assert 'timeout 300 "${health_binary}" "${health_raw}"' in pipeline
    assert '"${RESULT_DIR}/native_health.raw.json"' in legacy
    assert "tail -n 1" not in pipeline[pipeline.index("run_native_health() {") :]
    assert "tail -n 1" not in legacy
    assert "usage: native_phantom_health OUTPUT.json" in harness
    assert "std::ofstream result(argv[1]" in harness


def test_runpod_success_completeness_requires_every_terminal_record(
    tmp_path: Path,
) -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    start = source.index("verify_success_evidence() {")
    end = source.index("\nphase payload_verification", start)
    function = source[start:end]
    results = tmp_path / "results"
    results.mkdir()
    retained_context = results / "retained_compiler_context_manifest.json"
    retained_context.write_text('{"schema_version":1}\n', encoding="utf-8")
    native_stdout = results / "retained-native-primitives.stdout.txt"
    native_stderr = results / "retained-native-primitives.stderr.txt"
    native_stdout.write_text("[  PASSED  ] 9 tests.\n", encoding="utf-8")
    native_stderr.write_text("", encoding="utf-8")
    native_binary_sha256 = "a" * 64
    generated_source_sha256 = "b" * 64
    artifact_directory = results / "retained-host"
    artifact_directory.mkdir()
    (artifact_directory / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "build/retained_ckks_native_primitives_sm80": (
                        native_binary_sha256
                    ),
                    "outputs/retained_ckks_phantom.cu": generated_source_sha256,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bootstrap_binary_sha256 = "c" * 64
    bootstrap_artifact_directory = results / "bootstrap-qualification"
    bootstrap_artifact_directory.mkdir()
    (bootstrap_artifact_directory / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "bootstrap_qualification/bootstrap_phantom_constants_sm80": (
                        bootstrap_binary_sha256
                    )
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = {
        "qualification-current.json": {
            "gate": "ordinary",
            "status": "pass",
            "exit_code": 0,
        },
        "native-health.json": {"status": "pass", "device_count": 1},
        "bootstrap-qualification-current.json": {
            "gate": "bootstrap",
            "status": "pass",
            "exit_code": 0,
        },
        "bootstrap-frozen-reference.json": {
            "schema_version": "ace.phantom.bootstrap-frozen-reference/2.0.0",
            "status": "pass",
            "comparison": "exact-source-and-setup-with-per-run-cuda-artifacts",
            "source_and_setup_match": True,
            "linked_binary_byte_identity_required": False,
            "raw_air_matches": True,
            "post_ckks_air_matches": True,
            "generated_source_matches": True,
            "architecture": "sm_80",
            "semantic_symbol_closure": {
                "schema_version": (
                    "ace.phantom.bootstrap-symbol-closure/2.0.0"
                ),
                "status": "pass",
                "required_symbols": BOOTSTRAP_REQUIRED_SYMBOLS,
                "missing_symbols": [],
                "native_bootstrap_symbol_count": 0,
            },
            "packaged_artifact_manifest_sha256": "1" * 64,
            "packaged_host_qualification_sha256": "2" * 64,
            "packaged_source_audit_sha256": "3" * 64,
            "regenerated_artifact_manifest_sha256": "4" * 64,
            "regenerated_host_qualification_sha256": "5" * 64,
            "regenerated_source_audit_sha256": "6" * 64,
            "packaged_raw_air_sha256": "a" * 64,
            "regenerated_raw_air_sha256": "a" * 64,
            "packaged_post_ckks_air_sha256": "d" * 64,
            "regenerated_post_ckks_air_sha256": "d" * 64,
            "packaged_generated_source_sha256": "7" * 64,
            "regenerated_generated_source_sha256": "7" * 64,
            "packaged_harness_source_sha256": "8" * 64,
            "regenerated_harness_source_sha256": "8" * 64,
            "packaged_linked_binary_sha256": "9" * 64,
            "regenerated_linked_binary_sha256": bootstrap_binary_sha256,
        },
        "bootstrap-constant-cache.json": {
            "status": "pass",
            "gpu": "NVIDIA A100-SXM4-80GB",
            "resource_flags": 127,
            "counts": {
                "constants_declared": 2,
                "correctness_encodes": 2,
                "cached_loads": 2,
            },
            "entries": [{}, {}],
        },
        "bootstrap-setup-metrics.json": {
            "schema_version": "ace.phantom.bootstrap-setup-metrics/1.0.0",
            "status": "pass",
            "constant_count": 2,
            "emitted_payload_bytes": 256,
            "expected_logical_device_bytes": 4096,
            "context_and_keys": {"seconds": 0.25, "device_bytes": 1024},
            "plaintext_cache": {
                "seconds": 0.5,
                "device_bytes": 2048,
                "logical_device_bytes": 4096,
                "host_bytes": 512,
                "entries": 2,
            },
        },
        "bootstrap-constant-cache-execution.json": {
            "schema_version": (
                "ace.phantom.bootstrap-constant-cache-execution/1.0.0"
            ),
            "status": "pass",
            "invocation_count": 1,
            "binary_sha256": bootstrap_binary_sha256,
        },
        "ordinary_ckks_compare.json": {
            "status": "pass",
            "case_count": 43,
            "passing_case_count": 43,
        },
        "ordinary_ckks_diagnostics.json": {
            "status": "pass",
            "cases": [{"case_id": "expected-rejection"}],
        },
        "ordinary_ckks_ownership.json": {
            "schema_version": "ace.phantom.ordinary_ckks.ownership/1.0.0",
            "status": "pass",
            "total_iterations": 500,
        },
        "ordinary_ckks_sanitizer.json": {
            "status": "pass",
            "exit_code": 0,
        },
        "retained-frozen-reference.json": {
            "status": "pass",
            "provider_neutral_ant_reference_matches": True,
            "exact_artifact_count": 12,
            "ant_replay": {
                "comparison": (
                    "semantic-summary-only-no-decoded-byte-comparison"
                )
            },
        },
        "retained_ckks_native_primitives.json": {
            "schema_version": (
                "ace.phantom.retained_ckks.native-primitives/1.0.0"
            ),
            "status": "pass",
            "test_count": 9,
            "binary_sha256": native_binary_sha256,
            "emitted_context_manifest_sha256": hashlib.sha256(
                retained_context.read_bytes()
            ).hexdigest(),
            "stdout_sha256": hashlib.sha256(native_stdout.read_bytes()).hexdigest(),
            "stderr_sha256": hashlib.sha256(native_stderr.read_bytes()).hexdigest(),
        },
        "retained_ckks_compare.json": {
            "status": "pass",
            "gpu_vs_exact": {"status": "pass", "mismatch_count": 0},
        },
        "retained_ckks_exact_rns.json": {
            "status": "pass",
            "mismatch_count": 0,
        },
        "retained_ckks_adapter_aliases.json": {
            "schema_version": (
                "ace.phantom.retained_ckks.adapter-aliases/1.0.0"
            ),
            "status": "pass",
            "cases": [
                {
                    "status": "pass",
                    "alias_returned": True,
                    "metadata_matches_out_of_place": True,
                    "decoded_values_match_out_of_place": True,
                    "residues_match_out_of_place": True,
                },
                {
                    "status": "pass",
                    "alias_returned": True,
                    "metadata_matches_out_of_place": True,
                    "decoded_values_match_out_of_place": True,
                    "residues_match_out_of_place": True,
                },
            ],
        },
        "retained_ckks_rejections.json": {
            "schema_version": "ace.phantom.retained_ckks.rejections/1.0.0",
            "status": "pass",
            "cases": [
                {
                    "case_id": "expected-rejection",
                    "diagnostic": "EXPECTED",
                    "manifest": "production",
                    "exit_code": 134,
                    "status": "pass",
                }
            ],
        },
        "retained_ckks_ownership.json": {
            "schema_version": "ace.phantom.retained_ckks.ownership/1.0.0",
            "status": "pass",
            "iterations": 100,
            "ordered_steps": [5, 0, -7, 5],
            "batch_outputs_are_independent": True,
            "free_order": [3, 1, 0, 2],
        },
        "retained_ckks_sanitizer.json": {
            "status": "pass",
            "error_summary": 0,
        },
        "retained_production_archive_audit.json": {
            "schema_version": (
                "ace.phantom.retained_ckks.production-archive-audit/1.0.0"
            ),
            "status": "pass",
            "forbidden_entry_count": 0,
            "inventories": [
                {"forbidden_entries": []} for _ in range(6)
            ],
        },
        "retained_generated_source_audit.json": {
            "schema_version": (
                "ace.phantom.retained_ckks.generated-source-audit/2.0.0"
            ),
            "status": "pass",
            "source_sha256": generated_source_sha256,
            "required_calls": [
                "Conjugate_ciph",
                "Rotate_batch_ciph",
                "Raise_mod",
                "Mul_mono_ciph",
            ],
            "first_distinct_calls": [
                "Conjugate_ciph",
                "Rotate_batch_ciph",
                "Raise_mod",
                "Mul_mono_ciph",
            ],
            "call_counts": {
                "Conjugate_ciph": 1,
                "Rotate_batch_ciph": 1,
                "Raise_mod": 1,
                "Mul_mono_ciph": 1,
            },
            "argument_order": {
                "Conjugate_ciph": True,
                "Rotate_batch_ciph": True,
                "Raise_mod": True,
                "Mul_mono_ciph": True,
            },
            "rotation_array_emission": "ckks-owned-static-int32",
            "expected_rotation_batches": [
                [5, 0, -7, 5],
                [1],
                [-2, 3],
                [5, 0, -7, 5],
            ],
            "observed_rotation_batches": [
                [5, 0, -7, 5],
                [1],
                [-2, 3],
                [5, 0, -7, 5],
            ],
            "cipher_array_copy_count": 7,
            "expected_cipher_array_copy_count": 7,
            "cipher_array_copy_indices": [0, 0, 0, 0, 1, 2, 3],
            "expected_cipher_array_copy_indices": [0, 0, 0, 0, 1, 2, 3],
            "raw_cipher_array_assignments": [],
            "forbidden_matches": [],
        },
        "retained_gtest_source_attestation.json": {
            "schema_version": "ace.phantom.retained_ckks.gtest-source/1.0.0",
            "status": "pass",
            "source_method": "ace-pinned-external-project-source-reuse",
            "fetchcontent_source_override": True,
            "fetchcontent_fully_disconnected": True,
        },
        "retained_ckks_v1.json": {
            "runtime_rejections": [
                {
                    "id": "expected-rejection",
                    "diagnostic": "EXPECTED",
                    "manifest": "production",
                }
            ],
            "rotate_batch_steps": [5, 0, -7, 5],
            "production_rotation_batches": [[1], [-2, 3]],
            "ownership": {
                "iterations": 100,
                "batch_outputs_are_independent": True,
                "free_order": [3, 1, 0, 2],
            },
        },
    }
    for name, record in records.items():
        (results / name).write_text(json.dumps(record) + "\n", encoding="utf-8")

    script = (
        "set -euo pipefail\n"
        + function
        + '\nMODE=runpod\nRESULT_DIR="$1"\nverify_success_evidence\n'
    )
    passed = subprocess.run(
        ["bash", "-c", script, "completeness-test", str(results)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stderr
    assert json.loads(
        (results / "result-completeness.json").read_text(encoding="utf-8")
    ) == {
        "schema_version": "ace.phantom.result-completeness/1.0.0",
        "status": "pass",
        "mode": "runpod",
    }

    bootstrap_execution = results / "bootstrap-constant-cache-execution.json"
    bootstrap_execution.unlink()
    (results / "result-completeness.json").unlink()
    missing_bootstrap = subprocess.run(
        ["bash", "-c", script, "completeness-test", str(results)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_bootstrap.returncode != 0
    assert missing_bootstrap.stderr == (
        "result completeness: missing or empty terminal record: "
        "bootstrap-constant-cache-execution.json\n"
    )
    bootstrap_execution.write_text(
        json.dumps(records["bootstrap-constant-cache-execution.json"]) + "\n",
        encoding="utf-8",
    )

    invalid_metrics = json.loads(
        json.dumps(records["bootstrap-setup-metrics.json"])
    )
    invalid_metrics["plaintext_cache"]["logical_device_bytes"] += 1
    (results / "bootstrap-setup-metrics.json").write_text(
        json.dumps(invalid_metrics) + "\n", encoding="utf-8"
    )
    rejected_metrics = subprocess.run(
        ["bash", "-c", script, "completeness-test", str(results)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_metrics.returncode != 0
    assert rejected_metrics.stderr == (
        "result completeness: terminal record violates its contract: "
        "bootstrap-setup-metrics.json\n"
    )
    (results / "bootstrap-setup-metrics.json").write_text(
        json.dumps(records["bootstrap-setup-metrics.json"]) + "\n",
        encoding="utf-8",
    )

    (results / "ordinary_ckks_sanitizer.json").unlink()
    failed = subprocess.run(
        ["bash", "-c", script, "completeness-test", str(results)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert failed.stderr == (
        "result completeness: missing or empty terminal record: "
        "ordinary_ckks_sanitizer.json\n"
    )
    assert not (results / "result-completeness.json").exists()

    records["retained_ckks_rejections.json"]["schema_version"] = (
        "ace.phantom.retained_ckks.rejections/0.0.0"
    )
    (results / "retained_ckks_rejections.json").write_text(
        json.dumps(records["retained_ckks_rejections.json"]) + "\n",
        encoding="utf-8",
    )
    (results / "ordinary_ckks_sanitizer.json").write_text(
        json.dumps(records["ordinary_ckks_sanitizer.json"]) + "\n",
        encoding="utf-8",
    )
    rejected_schema = subprocess.run(
        ["bash", "-c", script, "completeness-test", str(results)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_schema.returncode != 0
    assert rejected_schema.stderr == (
        "result completeness: terminal record violates its contract: "
        "retained_ckks_rejections.json\n"
    )
    assert not (results / "result-completeness.json").exists()

    records["retained_ckks_rejections.json"]["schema_version"] = (
        "ace.phantom.retained_ckks.rejections/1.0.0"
    )
    (results / "retained_ckks_rejections.json").write_text(
        json.dumps(records["retained_ckks_rejections.json"]) + "\n",
        encoding="utf-8",
    )
    generated_path = results / "retained_generated_source_audit.json"
    valid_generated = records["retained_generated_source_audit.json"]

    def assert_generated_audit_rejected(record: dict[str, object]) -> None:
        generated_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        rejected = subprocess.run(
            ["bash", "-c", script, "completeness-test", str(results)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0
        assert rejected.stderr == (
            "result completeness: terminal record violates its contract: "
            "retained_generated_source_audit.json\n"
        )
        assert not (results / "result-completeness.json").exists()

    wrong_count = json.loads(json.dumps(valid_generated))
    wrong_count["cipher_array_copy_count"] = 6
    wrong_count["expected_cipher_array_copy_count"] = 6
    assert_generated_audit_rejected(wrong_count)

    wrong_indices = json.loads(json.dumps(valid_generated))
    wrong_indices["cipher_array_copy_indices"] = [0] * 7
    wrong_indices["expected_cipher_array_copy_indices"] = [0] * 7
    assert_generated_audit_rejected(wrong_indices)

    wrong_source = json.loads(json.dumps(valid_generated))
    wrong_source["source_sha256"] = "c" * 64
    assert_generated_audit_rejected(wrong_source)

    raw_assignment = json.loads(json.dumps(valid_generated))
    raw_assignment["raw_cipher_array_assignments"] = ["_t0 = _t1[0];"]
    assert_generated_audit_rejected(raw_assignment)

    missing_key = json.loads(json.dumps(valid_generated))
    missing_key.pop("source_sha256")
    assert_generated_audit_rejected(missing_key)

    unexpected_key = json.loads(json.dumps(valid_generated))
    unexpected_key["unbound_claim"] = True
    assert_generated_audit_rejected(unexpected_key)


def test_pipeline_failure_record_names_the_failed_phase() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "FAILED_PHASE=\"\"" in source
    assert 'FAILED_PHASE="${phase_name}"' in source
    assert '"${RESULT_DIR}/pipeline-failure.json"' in source
