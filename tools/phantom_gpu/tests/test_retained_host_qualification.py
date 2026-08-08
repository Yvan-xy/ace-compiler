"""Focused static and snapshot contracts for retained host qualification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS / "run_retained_ckks_correctness.sh"


def _load_air_tools():
    path = TOOLS / "retained_air_tools.py"
    spec = importlib.util.spec_from_file_location(
        "retained_host_qualification_air_tools", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_snapshot_manifest(root: Path, manifest: Path) -> tuple[str, str]:
    commit = "a" * 40
    selected = root / "tools/phantom_gpu/generator.py"
    selected.parent.mkdir(parents=True)
    selected.write_text("print('audited')\n", encoding="utf-8")
    value = {
        "schema_version": "1.0.0",
        "kind": "ace",
        "source_method": "git-commit-object-archive",
        "commit": commit,
        "tree": "b" * 40,
        "members": [
            {
                "path": "ace-source/tools/phantom_gpu/generator.py",
                "type": "file",
                "mode": "0755",
                "size": selected.stat().st_size,
                "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return commit, hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_snapshot_generation_inputs_are_authenticated_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    air_tools = _load_air_tools()
    root = tmp_path / "ace-source"
    manifest = tmp_path / "ace-source.manifest.json"
    commit, manifest_sha = _write_snapshot_manifest(root, manifest)
    monkeypatch.setenv("ACE_PHANTOM_SOURCE_MODE", "snapshot")
    monkeypatch.setenv("ACE_PHANTOM_ACE_COMMIT", commit)
    monkeypatch.setenv("ACE_PHANTOM_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setenv("ACE_PHANTOM_SOURCE_MANIFEST_SHA256", manifest_sha)

    assert air_tools.require_clean_tracked_sources(
        root, ("tools/phantom_gpu",)
    ) == commit
    (root / "tools/phantom_gpu/generator.py").write_text(
        "print('modified')\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differs from its manifest"):
        air_tools.require_clean_tracked_sources(root, ("tools/phantom_gpu",))


def test_host_gate_builds_both_generated_paths_but_runs_only_ant() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "usage: $0 --host-qualify" in source
    assert "ACE_PHANTOM_SOURCE_MODE" in source
    assert "verify_extracted_source_manifest" in source
    assert "generate_retained_ckks_sources.py" in source
    assert "generate_retained_production_air.py" in source
    assert "bind-fixture" in source
    assert "generate-analytic" in source
    assert "generate-exact" in source
    assert "retained_ckks_cpu_reference.json" in source
    assert '"${NVCC}" "${nvcc_common[@]}" -dlink' in source
    assert '"gpu_executables_were_run": False' in source
    assert 'RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 "${ANT_BINARY}"' in source
    assert "nvidia-smi" not in source
    assert "cuda-memcheck" not in source


def test_host_gate_attests_context_build_and_complete_result_tree() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "compiler_context_manifest_sha256",
        "normalized_compiler_command_sha256",
        "ace_source_manifest_sha256",
        "phantom_source_manifest_sha256",
        "generated_ant_source_sha256",
        "generated_phantom_source_sha256",
        "build_attestation_sha256",
        "artifact_manifest.json",
        "SHA256SUMS",
        "phantom-undefined-symbols.txt",
        "phantom-cuda-elf.txt",
        "provider-archive-members.txt",
        "pytest-retained-host.txt",
        "ctest-retained-host.txt",
    ):
        assert token in source
    assert "milestone" not in source.lower()
