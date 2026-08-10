"""Source contracts for the exact-commit bootstrap host-freeze lifecycle."""

from pathlib import Path
import subprocess


TOOLS = Path(__file__).resolve().parents[1]
WRAPPER = TOOLS / "freeze_bootstrap_host_evidence.sh"


def test_wrapper_is_valid_shell_and_uses_a_fresh_exact_snapshot() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'if [[ -e "${OUTPUT}" ]]' in source
    assert '"${selected_commit}:${lifecycle_file}"' in source
    assert 'load_selected_transport_helpers "${ACE_COMMIT}"' in source
    loader = source[source.index("load_selected_transport_helpers()") :]
    assert loader.index('verify_selected_lifecycle_files "${selected_commit}"') < (
        loader.index('source "${SCRIPT_DIR}/transport_helpers.sh"')
    )
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
    assert "--network bridge" in source
    assert "--env NVIDIA_VISIBLE_DEVICES=void" in source
    assert "--env NVIDIA_DRIVER_CAPABILITIES=none" in source
    assert 'host.get("DeviceRequests") not in (None, [])' in source
    assert "set(qualification) != exact_keys" in source
    assert "duplicate JSON key in bootstrap qualification" in source
    assert 'qualification.get("host_oracle_executables_were_run") is not True' in source
    assert 'qualification.get("gpu_executable_was_run") is not False' in source


def test_wrapper_checksum_closes_results_and_removes_only_its_container() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'tee "${RESULTS}/bootstrap-host-qualification.log"' in source
    assert "find . -type f ! -path ./SHA256SUMS -print0" in source
    assert '"${RESULT_ARCHIVE}" bootstrap-host-freeze "${PIPELINE_EXIT}"' in source
    assert 'docker rm -f "${exact_id}"' in source
    assert '"container_cleanup": "removed-and-absent"' in source
    assert "verified-bootstrap-host-result/results/qualification" in source


def test_wrapper_checksum_closes_the_exact_outer_inventory_after_cleanup() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    required = (
        "bootstrap-host-freeze.log",
        "bootstrap-host-result-verification.json",
        "bootstrap-host-result.tar.gz",
        "bootstrap-host-result.tar.gz.sha256",
        "docker/base-image.json",
        "docker/cleanup-verification.tsv",
        "docker/cleanup.txt",
        "docker/container-completed.json",
        "docker/container-created.json",
        "docker/pull.txt",
    )
    function_start = source.index("write_outer_bundle_checksum_closure()")
    function_end = source.index("\n}\n\nwrite_lifecycle_receipt()", function_start)
    function = source[function_start:function_end]
    for relative in required:
        assert relative in function
    assert '[[ ! -s "${OUTPUT}/lifecycle.json" ]]' in function
    assert '! -type d ! -type f' in function
    assert "find . -type f ! -path ./BUNDLE_SHA256SUMS -print0" in function
    assert "LC_ALL=C sort -z" in function
    assert 'cmp -- "${inventory_before}" "${inventory_after}"' in function
    assert "sha256sum -c BUNDLE_SHA256SUMS" in function
    assert "! -name BUNDLE_SHA256SUMS" not in function

    synchronous = source.index('docker start -a "${CONTAINER_ID}"')
    cleanup = source.index("\nremove_exact_container\n", synchronous)
    verification = source.index("\nverify_result_archive \\", cleanup)
    failure = source.index("if [[ ${PIPELINE_EXIT} -ne 0 ]]; then", verification)
    failure_lifecycle = source.index(
        "write_lifecycle_receipt failed", failure
    )
    failure_close = source.index(
        "write_outer_bundle_checksum_closure", failure
    )
    failure_exit = source.index('exit "${PIPELINE_EXIT}"', failure)
    success_lifecycle = source.index(
        "write_lifecycle_receipt pass", failure_exit
    )
    success_close = source.index(
        "write_outer_bundle_checksum_closure", success_lifecycle
    )
    success_message = source.index(
        'echo "bootstrap host evidence: ${QUALIFICATION}"', success_close
    )
    assert cleanup < verification < failure < failure_lifecycle
    assert failure_lifecycle < failure_close < failure_exit
    assert failure_exit < success_lifecycle < success_close < success_message


def test_bundle_install_intent_binds_every_pre_manifest_file() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    intent_start = source.index("bundle_install_intent()")
    closure_start = source.index("write_outer_bundle_checksum_closure()")
    intent = source[intent_start:closure_start]
    assert '"bound_files": bound_files' in intent
    assert '"bound_inventory_sha256"' in intent
    assert '"BUNDLE_SHA256SUMS"' in intent
    assert "intent_path.relative_to(output).as_posix()" in intent
    assert "output.rglob(\"*\")" in intent
    assert "stat.S_ISREG(mode)" in intent
    closure = source[closure_start:source.index("verify_existing_outer_bundle()")]
    assert closure.count("bundle_install_intent validate") >= 2


def test_wrapper_has_a_durable_two_phase_detached_lifecycle() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'RUN_MODE="detached-launch"' in source
    assert 'RUN_MODE="detached-finalize"' in source
    assert "ace.phantom.bootstrap-host-detached-state/1.0.0" in source
    assert "ace.phantom.bootstrap-host-detached-launch/1.0.0" in source
    assert '"${OUTPUT}/detached-launch-state.json"' in source
    assert '"${OUTPUT}/detached-launch.json"' in source
    assert 'docker start "${CONTAINER_ID}"' in source
    assert 'docker start -a "${CONTAINER_ID}"' in source
    assert 'docker logs "${CONTAINER_ID}"' in source

    state = source.index("write_detached_launch_state\n")
    start = source.index('docker start "${CONTAINER_ID}"', state)
    started_inspect = source.index(
        'docker inspect --type container "${CONTAINER_ID}"', start
    )
    launch_receipt = source.index("write_detached_launch_receipt", started_inspect)
    disarm_cleanup = source.index('CONTAINER_ID=""', launch_receipt)
    assert state < start < started_inspect < launch_receipt < disarm_cleanup


def test_detached_state_binds_payload_configuration_and_exact_cleanup() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    for binding in (
        '"output_device"',
        '"output_inode"',
        '"ace_commit"',
        '"phantom_commit"',
        '"base_config_digest"',
        '"qualification_arguments"',
        '"payload_descriptor_sha256"',
        '"payload_manifest_sha256"',
        '"created_receipt_sha256"',
        '"image_config_digest"',
        '"command"',
        '"environment"',
        '"mounts"',
    ):
        assert binding in source
    assert 'runtime_state.get("Running") is True' in source
    assert 'runtime_state.get("Status") != "exited"' in source
    assert 'observed != state["container"]' in source
    assert 'docker rm -f "${exact_id}"' in source
    assert 'name-inventory-empty' in source
    assert 'label-inventory-empty' in source


def test_detached_failures_are_checksum_closed_after_cleanup() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    finalizer = source[
        source.index("finalize_detached_run()") : source.index(
            'if [[ "${RUN_MODE}" == "detached-finalize" ]]',
            source.index("finalize_detached_run()"),
        )
    ]
    cleanup = finalizer.index("ensure_detached_container_cleanup")
    missing = finalizer.index("result archive or sidecar is missing")
    missing_lifecycle = finalizer.index(
        "write_lifecycle_receipt failed", missing
    )
    missing_close = finalizer.index(
        "write_outer_bundle_checksum_closure allow-missing-result", missing
    )
    assert cleanup < missing < missing_lifecycle < missing_close
    assert 'write_result_verification_failure "incoming result entries or sidecar were rejected"' in finalizer
    assert 'write_result_verification_failure "result archive structure verification failed"' in finalizer
