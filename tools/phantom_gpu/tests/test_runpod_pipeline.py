from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

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
