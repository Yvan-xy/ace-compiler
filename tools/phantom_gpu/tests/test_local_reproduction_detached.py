from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

import pytest


TOOLS = Path(__file__).resolve().parents[1]
PHANTOM_COMMIT = "b" * 40
BASE_IMAGE = "registry.example.invalid/qualification@sha256:" + "c" * 64
BASE_CONFIG = "sha256:" + "d" * 64


FAKE_DOCKER = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

args = sys.argv[1:]
state_path = Path(os.environ["FAKE_DOCKER_STATE"])
container_id = "a" * 64

def load():
    return json.loads(state_path.read_text()) if state_path.exists() else None

def save(value):
    state_path.write_text(json.dumps(value, sort_keys=True) + "\n")

def wait_at_gate(release, partial=""):
    if partial:
        sys.stdout.write(partial)
        sys.stdout.flush()
    Path(release + ".waiting").write_text("waiting\n")
    deadline = time.monotonic() + 30
    while not Path(release).exists():
        if time.monotonic() >= deadline:
            raise SystemExit("fake Docker gate timed out")
        time.sleep(0.01)

def publish_result(value):
    staging = Path(next(
        mount["Source"] for mount in value["Mounts"]
        if mount["Destination"] == "/retained-qualification/output"
    ))
    results = Path(tempfile.mkdtemp()) / "results"
    results.mkdir()
    exit_code = int(os.environ.get("FAKE_PIPELINE_EXIT", "0"))
    pipeline = {
        "schema_version": "1.0.0",
        "status": "pass" if exit_code == 0 else "failed",
        "mode": "local",
        "exit_code": exit_code,
        "started_utc": "2026-08-10T00:00:00Z",
        "completed_utc": "2026-08-10T00:00:01Z",
    }
    (results / "pipeline-result.json").write_text(
        json.dumps(pipeline, sort_keys=True) + "\n"
    )
    if exit_code == 0:
        (results / "result-completeness.json").write_text(json.dumps({
            "schema_version": "ace.phantom.result-completeness/1.0.0",
            "status": "pass", "mode": "local",
        }, sort_keys=True) + "\n")
    for index in range(int(os.environ.get("FAKE_RESULT_FILE_COUNT", "0"))):
        (results / f"payload-{index:05d}.txt").write_text(f"{index}\n")
    entries = []
    for path in sorted(results.iterdir()):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (results / "SHA256SUMS").write_text("".join(entries))
    archive = staging / "local-result.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(results, arcname="results")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar = staging / "local-result.tar.gz.sha256"
    if os.environ.get("FAKE_RESULT_MODE") == "invalid":
        sidecar.write_text("0" * 64 + "  local-result.tar.gz\n")
        (staging / "unexpected.txt").write_text("unexpected\n")
    else:
        sidecar.write_text(f"{digest}  local-result.tar.gz\n")

if not args:
    raise SystemExit(2)
command = args[0]
if command == "context":
    print("fake-context")
elif command == "info":
    print("fake-daemon")
elif command == "pull":
    print("pulled")
elif command == "image" and args[1] == "ls":
    print("fake-image")
elif command == "image" and args[1] == "inspect":
    if "-f" in args:
        print(os.environ["FAKE_BASE_CONFIG"])
    else:
        print(json.dumps([{
            "Id": os.environ["FAKE_BASE_CONFIG"],
            "Config": {"Env": [], "Labels": {}},
        }]))
elif command == "system":
    print("fake-disk")
elif command == "ps":
    value = load()
    if value is not None and not value.get("Removed", False):
        print(value["Id"])
elif command == "create":
    name = ""
    labels = {}
    environment = []
    volumes = []
    index = 1
    valued = {"--name", "--label", "--platform", "--runtime", "--restart", "--env", "--volume"}
    while index < len(args) and args[index].startswith("--"):
        option = args[index]
        if option not in valued:
            raise SystemExit(f"unexpected create option: {option}")
        item = args[index + 1]
        if option == "--name":
            name = item
        elif option == "--label":
            key, value = item.split("=", 1)
            labels[key] = value
        elif option == "--env":
            environment.append(item)
        elif option == "--volume":
            source, destination, mode = item.rsplit(":", 2)
            volumes.append({
                "Type": "bind", "Source": source,
                "Destination": destination, "Mode": mode, "RW": mode == "rw",
            })
        index += 2
    image = args[index]
    cmd = args[index + 1:]
    value = {
        "Id": container_id, "Name": "/" + name,
        "Image": os.environ["FAKE_BASE_CONFIG"],
        "Config": {"Image": image, "Cmd": cmd, "Env": environment, "Labels": labels},
        "HostConfig": {
            "Runtime": "runc", "Privileged": False, "Devices": [],
            "DeviceRequests": [], "RestartPolicy": {"Name": "no"},
        },
        "Mounts": volumes,
        "State": {
            "Status": "created", "Running": False, "ExitCode": 0,
            "StartedAt": "0001-01-01T00:00:00Z", "FinishedAt": "0001-01-01T00:00:00Z",
        },
        "Removed": False, "RemovedIds": [],
    }
    save(value)
    print(container_id)
elif command == "inspect":
    value = load()
    if value is None or value.get("Removed", False):
        raise SystemExit(1)
    release = os.environ.get("FAKE_INSPECT_RELEASE")
    if release:
        count_path = Path(release + ".count")
        count = int(count_path.read_text()) + 1 if count_path.exists() else 1
        count_path.write_text(str(count))
        if count == int(os.environ.get("FAKE_INSPECT_GATE_INDEX", "1")):
            wait_at_gate(release, "[")
    print(json.dumps([{key: item for key, item in value.items() if key not in {"Removed", "RemovedIds"}}]))
elif command == "start":
    release = os.environ.get("FAKE_START_RELEASE")
    if release:
        Path(release + ".waiting").write_text("waiting\n")
        deadline = time.monotonic() + 30
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise SystemExit("fake Docker start gate timed out")
            time.sleep(0.01)
    value = load()
    value["State"].update({
        "Status": "exited", "Running": False,
        "ExitCode": int(os.environ.get("FAKE_PIPELINE_EXIT", "0")),
        "StartedAt": "2026-08-10T00:00:00Z",
        "FinishedAt": "2026-08-10T00:00:01Z",
    })
    publish_result(value)
    save(value)
    output_release = os.environ.get("FAKE_START_OUTPUT_RELEASE")
    if output_release:
        wait_at_gate(output_release, container_id[:11])
        print(container_id[11:])
        raise SystemExit(0)
    forced_exit = os.environ.get("FAKE_START_EXIT_AFTER_EFFECT")
    if forced_exit:
        raise SystemExit(int(forced_exit))
    print(container_id)
elif command == "wait":
    print(os.environ.get("FAKE_WAIT_OUTPUT", load()["State"]["ExitCode"]))
elif command == "logs":
    text = os.environ.get("FAKE_LOG_TEXT", "")
    if text:
        print(text)
elif command == "rm":
    value = load()
    value["Removed"] = True
    value["RemovedIds"].append(args[-1])
    save(value)
    print(args[-1])
else:
    raise SystemExit(f"unsupported fake Docker command: {args}")
'''


@pytest.fixture()
def local_lifecycle(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repository"
    tools = repository / "tools/phantom_gpu"
    config = tools / "configs"
    config.mkdir(parents=True)
    runner = tools / "run_local_reproduction.sh"
    helper = tools / "transport_helpers.sh"
    shutil.copy2(TOOLS / runner.name, runner)
    shutil.copy2(TOOLS / helper.name, helper)
    package = tools / "package_runpod_sources.sh"
    package.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "for output; do :; done\nmkdir -p \"${output}\"\n"
        "printf '{}\\n' >\"${output}/payload.json\"\n"
        "(cd \"${output}\" && sha256sum payload.json >SHA256SUMS)\n"
    )
    package.chmod(0o755)
    (config / "dependencies.env").write_text(
        f"CUDA_IMAGE={BASE_IMAGE}\nCUDA_IMAGE_CONFIG={BASE_CONFIG}\n"
        f"PHANTOM_COMMIT={PHANTOM_COMMIT}\n"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    state = tmp_path / "docker-state.json"
    environment = dict(os.environ)
    environment.update({
        "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
        "FAKE_DOCKER_STATE": str(state),
        "FAKE_BASE_CONFIG": BASE_CONFIG,
        "ACE_PHANTOM_BUILD_JOBS": "2",
    })
    output = tmp_path / "local-output"
    common = [
        "bash", str(runner), "--qualification-mode", "full",
        "--ace-commit", commit, "--phantom-commit", PHANTOM_COMMIT,
        "--ordinary-run-root", str(tmp_path / "ordinary"),
        "--retained-run-root", str(tmp_path / "retained"),
        "--bootstrap-run-root", str(tmp_path / "bootstrap"),
    ]
    return {
        "runner": runner, "helper": helper,
        "state": state, "environment": environment,
        "output": output, "launch": common + ["--detach", str(output)],
        "finalize": ["bash", str(runner), "--finalize", str(output)],
    }


def run(arguments: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments, env=environment, text=True, capture_output=True,
        check=False, timeout=30,
    )


def wait_for_path(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        return_code = process.poll()
        if return_code is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"process exited {return_code} before {path} appeared\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


def assert_exact_bundle(output: Path) -> None:
    manifest = output / "BUNDLE_SHA256SUMS"
    subprocess.run(["sha256sum", "-c", manifest.name], cwd=output, check=True, capture_output=True)
    listed = {
        line.split("  ", 1)[1].removeprefix("./")
        for line in manifest.read_text().splitlines()
    }
    actual = {
        str(path.relative_to(output)) for path in output.rglob("*")
        if path.is_file() and path != manifest
    }
    assert listed == actual


def test_detached_fast_exit_recovers_receipt_and_checksum_closes_empty_log(
    local_lifecycle: dict[str, object],
) -> None:
    environment = local_lifecycle["environment"]
    output = local_lifecycle["output"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["Removed"] is False
    intent = json.loads((output / "detached-intent.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{32}", intent["run_nonce"])
    assert state["Config"]["Labels"]["ace.phantom.run-nonce"] == intent["run_nonce"]
    intent_path = output / "detached-intent.json"
    assert intent_path.stat().st_mode & 0o777 == 0o600
    staging_stat = (output / "incoming-results").stat()
    assert intent["result_staging_directory"] == str(
        (output / "incoming-results").resolve()
    )
    assert intent["result_staging_device"] == staging_stat.st_dev
    assert intent["result_staging_inode"] == staging_stat.st_ino
    (output / "docker/disposable-container-created.json").unlink()
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["RemovedIds"] == ["a" * 64]
    assert (output / "local-container.log").read_bytes() == b""
    lifecycle = json.loads((output / "lifecycle.json").read_text())
    assert lifecycle["status"] == "pass"
    assert lifecycle["container_cleanup"] == "removed-and-absent"
    verification = json.loads(
        (output / "local-result-verification.json").read_text()
    )
    assert set(verification) == {
        "status", "mode", "pipeline_exit_code", "verified_file_count",
        "sha256_manifest_sha256",
    }
    result_sums = output / "verified-local-result/results/SHA256SUMS"
    assert verification["verified_file_count"] == 2
    assert verification["sha256_manifest_sha256"] == hashlib.sha256(
        result_sums.read_bytes()
    ).hexdigest()
    assert_exact_bundle(output)
    (output / "BUNDLE_SHA256SUMS").unlink()
    (output / "lifecycle.json").unlink()
    (output / "docker/cleanup-verification.tsv").unlink()
    resumed = run(local_lifecycle["finalize"], environment)
    assert resumed.returncode == 0, resumed.stderr
    assert_exact_bundle(output)
    already_closed = run(local_lifecycle["finalize"], environment)
    assert already_closed.returncode == 0, already_closed.stderr


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("Config", "WorkingDir", "/changed-after-create"),
        ("HostConfig", "ReadonlyRootfs", True),
    ],
)
def test_detached_immutable_config_mismatch_refuses_cleanup(
    local_lifecycle: dict[str, object],
    section: str,
    key: str,
    value: object,
) -> None:
    environment = local_lifecycle["environment"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    state = json.loads(local_lifecycle["state"].read_text())
    state[section][key] = value
    local_lifecycle["state"].write_text(json.dumps(state) + "\n")
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode != 0
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["Removed"] is False
    assert state["RemovedIds"] == []


def test_launch_lock_prevents_concurrent_finalize_during_start(
    local_lifecycle: dict[str, object],
    tmp_path: Path,
) -> None:
    environment = dict(local_lifecycle["environment"])
    release = tmp_path / "release-start"
    environment["FAKE_START_RELEASE"] = str(release)
    launch_process = subprocess.Popen(
        local_lifecycle["launch"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        wait_for_path(Path(str(release) + ".waiting"), launch_process)
        concurrent = run(local_lifecycle["finalize"], environment)
        assert concurrent.returncode != 0
        assert "another local finalizer owns this output" in concurrent.stderr
        state = json.loads(local_lifecycle["state"].read_text())
        assert state["Removed"] is False
        assert state["State"]["Status"] == "created"
        release.write_text("release\n")
        stdout, stderr = launch_process.communicate(timeout=10)
        assert launch_process.returncode == 0, f"{stdout}\n{stderr}"
    finally:
        if launch_process.poll() is None:
            release.write_text("release\n")
            os.killpg(launch_process.pid, signal.SIGKILL)
            launch_process.wait(timeout=10)
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    assert_exact_bundle(local_lifecycle["output"])


@pytest.mark.parametrize(
    ("phase", "gate_variable", "gate_index", "work_receipt", "expected_exit"),
    [
        (
            "created-receipt", "FAKE_INSPECT_RELEASE", "1",
            "docker/disposable-container-created.json.work", 1,
        ),
        (
            "start-identity", "FAKE_START_OUTPUT_RELEASE", None,
            "docker/start.txt.work", 0,
        ),
        (
            "started-receipt", "FAKE_INSPECT_RELEASE", "3",
            "docker/container-started.json.work", 0,
        ),
    ],
)
def test_cutoff_during_atomic_start_receipts_is_safely_finalized(
    local_lifecycle: dict[str, object],
    tmp_path: Path,
    phase: str,
    gate_variable: str,
    gate_index: str | None,
    work_receipt: str,
    expected_exit: int,
) -> None:
    environment = dict(local_lifecycle["environment"])
    release = tmp_path / f"release-{phase}"
    environment[gate_variable] = str(release)
    if gate_index is not None:
        environment["FAKE_INSPECT_GATE_INDEX"] = gate_index
    launch_process = subprocess.Popen(
        local_lifecycle["launch"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        wait_for_path(Path(str(release) + ".waiting"), launch_process)
        os.killpg(launch_process.pid, signal.SIGKILL)
        launch_process.wait(timeout=10)
        assert launch_process.returncode != 0
    finally:
        if launch_process.poll() is None:
            os.killpg(launch_process.pid, signal.SIGKILL)
            launch_process.wait(timeout=10)
    output = local_lifecycle["output"]
    assert (output / work_receipt).is_file()
    resumed_environment = dict(local_lifecycle["environment"])
    finalized = run(local_lifecycle["finalize"], resumed_environment)
    assert finalized.returncode == expected_exit, finalized.stderr
    assert not any(
        path.name.endswith(".work") or path.name.endswith(".tmp")
        for path in output.rglob("*")
    )
    if expected_exit == 0:
        assert (output / "docker/start.txt").read_text() == "a" * 64 + "\n"
        assert json.loads((output / "lifecycle.json").read_text())["status"] == "pass"
    else:
        verification = json.loads(
            (output / "local-result-verification.json").read_text()
        )
        assert "preceded the durable start attempt" in verification["reason"]
        assert json.loads((output / "lifecycle.json").read_text())["status"] == "failed"
    assert_exact_bundle(output)


def test_ambiguous_start_client_error_recovers_observed_daemon_effect(
    local_lifecycle: dict[str, object],
) -> None:
    environment = dict(local_lifecycle["environment"])
    environment["FAKE_START_EXIT_AFTER_EFFECT"] = "42"
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    assert "exact container start effect was observed" in launched.stderr
    output = local_lifecycle["output"]
    observation = json.loads(
        (output / "docker/container-start-observation.json").read_text()
    )
    assert observation["status"] == "effect-observed"
    assert observation["client_exit_code"] == 42
    assert observation["client_output_exact_container_id"] is False
    assert (output / "docker/start.txt").read_text() == "a" * 64 + "\n"
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    assert_exact_bundle(output)


def test_dirty_transport_helper_is_rejected_before_it_is_sourced(
    local_lifecycle: dict[str, object],
    tmp_path: Path,
) -> None:
    helper = local_lifecycle["helper"]
    marker = tmp_path / "dirty-helper-was-sourced"
    helper.write_text(
        helper.read_text() + '\ntouch -- "${DIRTY_HELPER_MARKER}"\n'
    )
    environment = dict(local_lifecycle["environment"])
    environment["DIRTY_HELPER_MARKER"] = str(marker)
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode != 0
    assert "transport helper differs from the selected ACE commit" in launched.stderr
    assert not marker.exists()
    assert not local_lifecycle["output"].exists()


def test_split_result_promotion_resumes(
    local_lifecycle: dict[str, object],
) -> None:
    environment = local_lifecycle["environment"]
    output = local_lifecycle["output"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    (output / "incoming-results/local-result.tar.gz").replace(
        output / "local-result.tar.gz"
    )
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    assert not any((output / "incoming-results").iterdir())
    assert_exact_bundle(output)


def test_verified_result_promotion_resumes_after_first_atomic_move(
    local_lifecycle: dict[str, object],
) -> None:
    environment = local_lifecycle["environment"]
    output = local_lifecycle["output"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    for name in ("local-result.tar.gz", "local-result.tar.gz.sha256"):
        (output / "incoming-results" / name).replace(output / name)
    work_extraction = output / "verified-local-result.work"
    work_receipt = output / "local-result-verification.work.json"
    verified = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; verify_result_archive "$2" local 0 "$3" >"$4"',
            "verification", str(local_lifecycle["helper"]),
            str(output / "local-result.tar.gz"), str(work_extraction),
            str(work_receipt),
        ],
        env=environment, text=True, capture_output=True, check=False,
    )
    assert verified.returncode == 0, verified.stderr
    work_extraction.replace(output / "verified-local-result")
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    assert not work_receipt.exists()
    assert_exact_bundle(output)


def test_partial_verification_extraction_is_retried_after_controller_cutoff(
    local_lifecycle: dict[str, object],
) -> None:
    environment = dict(local_lifecycle["environment"])
    environment["FAKE_RESULT_FILE_COUNT"] = "5000"
    output = local_lifecycle["output"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    finalize_process = subprocess.Popen(
        local_lifecycle["finalize"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    work_extraction = output / "verified-local-result.work"
    wait_for_path(work_extraction, finalize_process)
    os.killpg(finalize_process.pid, signal.SIGKILL)
    finalize_process.wait(timeout=10)
    assert finalize_process.returncode != 0
    assert work_extraction.exists()
    resumed = run(local_lifecycle["finalize"], environment)
    assert resumed.returncode == 0, resumed.stderr
    assert not work_extraction.exists()
    assert_exact_bundle(output)


def test_synchronous_mode_uses_the_same_closed_lifecycle(
    local_lifecycle: dict[str, object],
) -> None:
    environment = local_lifecycle["environment"]
    arguments = [
        argument for argument in local_lifecycle["launch"]
        if argument != "--detach"
    ]
    completed = run(arguments, environment)
    assert completed.returncode == 0, completed.stderr
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["RemovedIds"] == ["a" * 64]
    lifecycle = json.loads(
        (local_lifecycle["output"] / "lifecycle.json").read_text()
    )
    assert lifecycle["status"] == "pass"
    assert_exact_bundle(local_lifecycle["output"])


def test_invalid_incoming_inventory_is_cleaned_and_failure_closed(
    local_lifecycle: dict[str, object],
) -> None:
    environment = dict(local_lifecycle["environment"])
    environment["FAKE_RESULT_MODE"] = "invalid"
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode != 0
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["RemovedIds"] == ["a" * 64]
    lifecycle = json.loads((local_lifecycle["output"] / "lifecycle.json").read_text())
    assert lifecycle["status"] == "failed"
    assert lifecycle["result_archive_status"] == "missing-or-rejected"
    assert (local_lifecycle["output"] / "local-container.log").read_bytes() == b""
    assert_exact_bundle(local_lifecycle["output"])


@pytest.mark.parametrize("damage", ["missing", "partial"])
def test_passing_bundle_cannot_reclose_without_exact_started_receipt(
    local_lifecycle: dict[str, object],
    damage: str,
) -> None:
    environment = local_lifecycle["environment"]
    output = local_lifecycle["output"]
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode == 0, finalized.stderr
    (output / "BUNDLE_SHA256SUMS").unlink()
    started = output / "docker/container-started.json"
    if damage == "missing":
        started.unlink()
    else:
        started.write_text("[")
    refused = run(local_lifecycle["finalize"], environment)
    assert refused.returncode != 0
    assert not (output / "BUNDLE_SHA256SUMS").exists()
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["RemovedIds"] == ["a" * 64]


def test_terminal_wait_error_still_exactly_cleans(
    local_lifecycle: dict[str, object],
) -> None:
    environment = dict(local_lifecycle["environment"])
    launched = run(local_lifecycle["launch"], environment)
    assert launched.returncode == 0, launched.stderr
    environment["FAKE_WAIT_OUTPUT"] = "not-an-exit-code"
    finalized = run(local_lifecycle["finalize"], environment)
    assert finalized.returncode != 0
    state = json.loads(local_lifecycle["state"].read_text())
    assert state["RemovedIds"] == ["a" * 64]
    assert not (local_lifecycle["output"] / "BUNDLE_SHA256SUMS").exists()
    environment.pop("FAKE_WAIT_OUTPUT")
    resumed = run(local_lifecycle["finalize"], environment)
    assert resumed.returncode != 0
    lifecycle = json.loads(
        (local_lifecycle["output"] / "lifecycle.json").read_text()
    )
    assert lifecycle["status"] == "failed"
    verification = json.loads(
        (local_lifecycle["output"] / "local-result-verification.json").read_text()
    )
    assert "container start was confirmed" in verification["reason"]
    assert "preceded start" not in verification["reason"]
    assert_exact_bundle(local_lifecycle["output"])
