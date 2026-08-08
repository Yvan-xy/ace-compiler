"""Static safety contracts for the exact-snapshot retained host freeze."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


TOOLS = Path(__file__).resolve().parents[1]
FREEZE = TOOLS / "freeze_retained_host_evidence.sh"
PACKAGE = TOOLS / "package_retained_host_freeze_sources.sh"
SNAPSHOT = TOOLS / "run_retained_host_freeze_snapshot.sh"
DEPENDENCIES = TOOLS / "configs/dependencies.env"
FHE_ROOT = TOOLS.parents[1] / "fhe-cmplr"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_scripts_are_valid_shell() -> None:
    for script in (FREEZE, PACKAGE, SNAPSHOT):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_phantom_pin_is_consistent_across_qualification_and_cmake() -> None:
    matches = re.findall(
        r"^PHANTOM_COMMIT=([0-9a-f]{40})$",
        source(DEPENDENCIES),
        re.MULTILINE,
    )
    assert len(matches) == 1
    expected = matches[0]
    for cmake in (
        FHE_ROOT / "CMakeLists.txt",
        FHE_ROOT / "rtlib/cmake/modules/phantom.cmake",
    ):
        assert re.findall(
            r'set\(PHANTOM_GIT_TAG "([0-9a-f]{40})"', source(cmake)
        ) == [expected]


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


def test_host_validator_dependencies_are_commit_materialized() -> None:
    package = source(PACKAGE)
    for dependency in (
        "retained_runpod_evidence.py",
        "compare_retained_ckks_results.py",
        "generate_retained_ckks_fixtures.py",
        "transport_helpers.sh",
    ):
        assert f"tools/phantom_gpu/{dependency}" in package


def test_materialized_validator_dependency_closure_imports() -> None:
    dependencies = (
        "retained_runpod_evidence.py",
        "compare_retained_ckks_results.py",
        "generate_retained_ckks_fixtures.py",
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shutil.copyfile(TOOLS / dependencies[0], root / dependencies[0])
        missing = subprocess.run(
            [sys.executable, str(root / dependencies[0]), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert missing.returncode != 0
        for dependency in dependencies[1:]:
            shutil.copyfile(TOOLS / dependency, root / dependency)
        complete = subprocess.run(
            [sys.executable, str(root / dependencies[0]), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert complete.returncode == 0, complete.stderr


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


def test_only_verified_archive_extraction_is_consumed() -> None:
    snapshot = source(SNAPSHOT)
    freeze = source(FREEZE)
    assert "--result-archive" in snapshot
    assert "pipeline-result.json" in snapshot
    assert "result-completeness.json" in snapshot
    assert "find . -type f ! -path ./SHA256SUMS -print0" in snapshot
    assert 'RESULT_ARCHIVE="${OUTPUT}/retained-host-result.tar.gz"' in freeze
    assert '"${RESULT_ARCHIVE}.sha256"' in freeze
    assert "verify_result_archive" in freeze
    assert 'VERIFIED_RESULT_ROOT="${VERIFIED_EXTRACTION}/results/qualification"' in freeze
    assert '--root "${VERIFIED_RESULT_ROOT}"' in freeze
    assert '"${VERIFIED_RESULT_ROOT}/inputs/retained_ckks_fixture.json"' in freeze
    assert "--result-root" not in freeze
    assert "failure.json" in snapshot


def test_candidate_export_rejects_non_binding_fixture_drift() -> None:
    freeze = source(FREEZE)
    for token in (
        'normalized_candidate["qualification_bindings"]',
        'normalized_candidate["production_rotation_source"]',
        "normalized_candidate != template_value",
        "compiler_context_manifest_sha256",
        "normalized_compiler_command_sha256",
        "post_ckks_air_sha256",
        "retained_ckks_phantom_post.air",
        "retained_ckks_production_post.air",
        "exact-after-binding-field-normalization",
        "retained_ckks_v1.unbound.json",
        "verified fixture template differs from the selected commit",
    ):
        assert token in freeze


def test_candidate_verifier_accepts_binder_shape_and_rejects_drift() -> None:
    freeze = source(FREEZE)
    verifier = freeze.split(
        "# retained-candidate-verifier-start\n", 1
    )[1].split("# retained-candidate-verifier-end", 1)[0]
    names = (
        "compiler_context_manifest_sha256",
        "normalized_compiler_command_sha256",
        "post_ckks_air_sha256",
    )
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        candidate = base / "candidate"
        root = base / "verified" / "results" / "qualification"
        candidate.mkdir()
        (root / "inputs").mkdir(parents=True)
        (root / "outputs").mkdir()
        template = {
            "schema_version": "test",
            "semantic_value": [1, 2, 3],
            "qualification_bindings": {
                "status": "unbound",
                "required": list(names),
            },
            "production_rotation_source": {
                "kind": "audited_default_post_ckks_air",
                "post_ckks_air_sha256": None,
            },
        }
        context = root / "inputs/compiler_context_manifest.json"
        post_air = root / "outputs/retained_ckks_phantom_post.air"
        production_air = root / "outputs/retained_ckks_production_post.air"
        context.write_bytes(b"context\n")
        post_air.write_bytes(b"post air\n")
        production_air.write_bytes(b"production post air\n")
        generation_hash = "a" * 64
        (root / "outputs/retained_ckks_generation.json").write_text(
            json.dumps({"normalized_argv_sha256": generation_hash}),
            encoding="utf-8",
        )
        (root / "manifest.json").write_text("{}\n", encoding="utf-8")
        bound = copy.deepcopy(template)
        bound["qualification_bindings"] = {
            "status": "bound",
            "compiler_context_manifest_sha256": hashlib.sha256(
                context.read_bytes()
            ).hexdigest(),
            "normalized_compiler_command_sha256": generation_hash,
            "post_ckks_air_sha256": hashlib.sha256(
                post_air.read_bytes()
            ).hexdigest(),
        }
        bound["production_rotation_source"]["post_ckks_air_sha256"] = (
            hashlib.sha256(production_air.read_bytes()).hexdigest()
        )
        for path in (
            candidate / "retained_ckks_v1.unbound.json",
            root / "inputs/retained_ckks_fixture_template.json",
        ):
            path.write_text(json.dumps(template), encoding="utf-8")
        bound_path = candidate / "retained_ckks_v1.json"
        bound_path.write_text(json.dumps(bound), encoding="utf-8")
        command = [
            sys.executable,
            "-c",
            verifier,
            str(candidate),
            str(root),
            "b" * 40,
            "c" * 40,
        ]
        accepted = subprocess.run(command, check=False, capture_output=True, text=True)
        assert accepted.returncode == 0, accepted.stderr
        assert json.loads(
            (candidate / "candidate-binding.json").read_text(encoding="utf-8")
        )["qualification_bindings"]["normalized_compiler_command_sha256"] == (
            generation_hash
        )
        bound["semantic_value"].append(4)
        bound_path.write_text(json.dumps(bound), encoding="utf-8")
        rejected = subprocess.run(command, check=False, capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "non-binding semantic drift" in rejected.stderr


def test_snapshot_entry_invokes_only_the_retained_host_gate() -> None:
    snapshot = source(SNAPSHOT)
    assert "run_retained_ckks_correctness.sh" in snapshot
    assert "--host-qualify" in snapshot
    assert "run_build_and_health.sh" not in snapshot
    assert "nvidia-smi" not in snapshot
    assert "compute-sanitizer" not in snapshot
    assert "cuda-memcheck" not in snapshot
    assert "gpu_executables_were_run" in snapshot


def test_snapshot_bootstraps_python_before_deep_payload_validation() -> None:
    snapshot = source(SNAPSHOT)
    checksum = snapshot.index("sha256sum -c SHA256SUMS")
    bootstrap = snapshot.index('bash "${INPUT}/bootstrap_environment.sh"')
    python_check = snapshot.index("command -v python3")
    identity_call = snapshot.index("\nverify_payload_identity\n")

    assert checksum < bootstrap < python_check < identity_call
