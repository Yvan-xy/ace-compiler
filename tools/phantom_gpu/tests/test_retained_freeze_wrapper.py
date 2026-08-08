"""Static safety contracts for the exact-snapshot retained host freeze."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess


TOOLS = Path(__file__).resolve().parents[1]
FREEZE = TOOLS / "freeze_retained_host_evidence.sh"
PACKAGE = TOOLS / "package_retained_host_freeze_sources.sh"
SNAPSHOT = TOOLS / "run_retained_host_freeze_snapshot.sh"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_scripts_are_valid_shell() -> None:
    for script in (FREEZE, PACKAGE, SNAPSHOT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_payload_comes_only_from_exact_commit_objects() -> None:
    package = source(PACKAGE)
    assert 'cat-file -e "${ACE_COMMIT}^{commit}"' in package
    assert 'cat-file -e "${PHANTOM_COMMIT}^{commit}"' in package
    assert '"${ACE_COMMIT}:tools/phantom_gpu/source_archive.py"' in package
    assert '"${ACE_COMMIT}:${relative}"' in package
    assert "source_archive.py\" create" in package
    assert "requested Phantom commit does not match" in package
    assert "git diff" not in package
    assert "git status" not in package


def test_container_boundary_is_exact_and_gpu_free() -> None:
    freeze = source(FREEZE)
    assert 'CREATED_CONTAINER_ID' in freeze
    assert '^[0-9a-f]{64}$' in freeze
    assert 'docker rm -f "${exact_id}"' in freeze
    assert 'docker inspect --type container "${exact_id}"' in freeze
    assert "removed-and-absent" in freeze
    assert "--runtime runc" in freeze
    assert "NVIDIA_VISIBLE_DEVICES=void" in freeze
    assert "DeviceRequests" in freeze
    assert "--gpus" not in freeze
    for forbidden in ("docker ps", "docker container ls", "docker images", "prune"):
        assert forbidden not in freeze
    assert "ace-compiler-dev" not in freeze


def test_resource_identifiers_are_semantic_and_have_no_milestone_labels() -> None:
    freeze = source(FREEZE)
    assert "ace-retained-fixture-freeze-${RUN_ID}" in freeze
    assert "retained-fixture-host-freeze-${RUN_ID}" in freeze
    identifiers = re.findall(r'(?:CONTAINER_NAME|TASK_LABEL)="([^"]+)"', freeze)
    assert identifiers
    assert all(re.search(r"(?:^|[-_])m[0-9]+(?:[-_]|$)", item, re.I) is None for item in identifiers)


def test_binding_modes_are_separate_and_formal_is_validated() -> None:
    snapshot = source(SNAPSHOT)
    freeze = source(FREEZE)
    assert "ACE_RETAINED_CKKS_PROVISIONAL_BINDING=1" in snapshot
    assert "ACE_RETAINED_CKKS_PROVISIONAL_BINDING=0" in snapshot
    assert "EXPECTED_LIFECYCLE=provisional-bind" in snapshot
    assert "EXPECTED_LIFECYCLE=checked-bound" in snapshot
    assert snapshot.count("validate-frozen") == 1
    assert freeze.count("validate-frozen") == 1
    assert "candidate-for-review-not-frozen-evidence" in freeze
    assert "retained_ckks_v1.json" in freeze


def test_snapshot_entry_invokes_only_the_retained_host_gate() -> None:
    snapshot = source(SNAPSHOT)
    assert "run_retained_ckks_correctness.sh" in snapshot
    assert "--host-qualify" in snapshot
    assert "run_build_and_health.sh" not in snapshot
    assert "nvidia-smi" not in snapshot
    assert "compute-sanitizer" not in snapshot
    assert "cuda-memcheck" not in snapshot
    assert "gpu_executables_were_run" in snapshot
