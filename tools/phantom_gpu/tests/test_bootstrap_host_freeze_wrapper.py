"""Source contracts for the exact-commit bootstrap host-freeze lifecycle."""

from pathlib import Path
import subprocess


TOOLS = Path(__file__).resolve().parents[1]
WRAPPER = TOOLS / "freeze_bootstrap_host_evidence.sh"


def test_wrapper_is_valid_shell_and_uses_a_fresh_exact_snapshot() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'if [[ -e "${OUTPUT}" ]]' in source
    assert 'git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${lifecycle_file}"' in source
    assert "ACE_PHANTOM_SOURCE_MODE=snapshot" in source
    assert "ACE_PHANTOM_SOURCE_MANIFEST_SHA256=${ACE_MANIFEST_SHA}" in source
    assert (
        "ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256=${PHANTOM_MANIFEST_SHA}"
        in source
    )
    assert 'ace_worktree_dirty": False' in source
    assert '"source_mode": "snapshot"' in source
    assert "tools/phantom_gpu/phase_helpers.sh" in source
    assert '"${PAYLOAD}/phase_helpers.sh"' in source
    assert "export SOURCE_DATE_EPOCH" in source
    assert r'[\"commit_timestamp\"]' in source
    assert "WORK=/retained-qualification/work" in source
    assert "WORK=${OUTPUT}/work" not in source


def test_wrapper_freezes_the_exact_bootstrap_host_contract_without_a_gpu() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert '--gate bootstrap "${QUALIFICATION_ARGS[@]}"' in source
    assert '"arguments": arguments' in source
    assert "ace.phantom.bootstrap-host-freeze-payload/2.0.0" in source
    assert '"parameters": {' not in source
    assert "--runtime runc" in source
    assert "--env NVIDIA_VISIBLE_DEVICES=void" in source
    assert "--env NVIDIA_DRIVER_CAPABILITIES=none" in source
    assert 'host.get("DeviceRequests") not in (None, [])' in source
    assert 'qualification.get("host_oracle_executables_were_run"),' in source
    assert 'qualification.get("gpu_executable_was_run"))' in source


def test_wrapper_checksum_closes_results_and_removes_only_its_container() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'tee "${RESULTS}/bootstrap-host-qualification.log"' in source
    assert "find . -type f ! -path ./SHA256SUMS -print0" in source
    assert '"${RESULT_ARCHIVE}" bootstrap-host-freeze "${PIPELINE_EXIT}"' in source
    assert 'docker rm -f "${exact_id}"' in source
    assert '"container_cleanup": "removed-and-absent"' in source
    assert "verified-bootstrap-host-result/results/qualification" in source
