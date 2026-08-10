"""Executable fake-Docker coverage for detached bootstrap host evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import textwrap
import time

import pytest


TOOLS = Path(__file__).resolve().parents[1]
WRAPPER = TOOLS / "freeze_bootstrap_host_evidence.sh"
CONFIGS = TOOLS / "configs"
IMAGE_CONFIG = "sha256:" + "1" * 64
IMAGE = "fake.invalid/cuda@sha256:" + "2" * 64
CONTAINER_ID = "a" * 64

QUALIFICATION_ARGUMENTS = [
    "--poly-degree", "16384", "--vector-capacity", "8192",
    "--mul-level", "26", "--input-level", "1", "--security-level", "0",
    "--scaling-factor-bits", "56", "--first-prime-bits", "60",
    "--hamming-weight", "192", "--q-part-count", "3",
    "--encode-transform-budget", "3", "--decode-transform-budget", "3",
    "--ciphertext-constant-encoding", "disabled", "--packing", "full",
    "--post-multiply-real", "-1.0", "--post-multiply-imag", "0.0",
    "--post-multiply-scale-degree", "0", "--post-rotation-step", "4",
    "--fixture-id", "generated-bootstrap-full-packed-a100-correctness-test",
    "--fixture-seed", "7640891576956012809", "--inside-margin", "0.125",
    "--provider-clear-threshold", "0.01", "--gpu-native-threshold", "0.02",
    "--gpu-generated-threshold", "0.02", "--repeat-threshold", "0.000001",
    "--host-oracle-timeout-seconds", "1800",
]

FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, pathlib, signal, sys

root = pathlib.Path(os.environ["FAKE_DOCKER_ROOT"])
root.mkdir(parents=True, exist_ok=True)
state_path = root / "state.json"

def load():
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"containers": {}, "calls": []}

def save(value):
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True))
    os.replace(temporary, state_path)

state = load()
args = sys.argv[1:]
state["calls"].append(args)
save(state)

if args[:2] == ["context", "show"]:
    print("fake-context")
elif args and args[0] == "info":
    print("fake-daemon-id")
elif args and args[0] == "version":
    print("99.0.0")
elif args and args[0] == "pull":
    print("fake pull complete")
elif args[:2] == ["image", "inspect"]:
    if "-f" in args:
        print(os.environ["FAKE_IMAGE_CONFIG"])
    else:
        print(json.dumps([{"Id": os.environ["FAKE_IMAGE_CONFIG"]}]))
elif args and args[0] == "create":
    name = None
    labels, environment, volumes = {}, [], []
    runtime, restart, network = "runc", "no", "bridge"
    index = 1
    value_options = {
        "--name", "--label", "--platform", "--runtime", "--restart",
        "--env", "--volume", "--network",
    }
    while index < len(args) and args[index].startswith("--"):
        option, value = args[index], args[index + 1]
        assert option in value_options, option
        if option == "--name": name = value
        elif option == "--label":
            key, item = value.split("=", 1); labels[key] = item
        elif option == "--env": environment.append(value)
        elif option == "--volume": volumes.append(value)
        elif option == "--runtime": runtime = value
        elif option == "--restart": restart = value
        elif option == "--network": network = value
        index += 2
    image, command = args[index], args[index + 1:]
    mounts = []
    for volume in volumes:
        source, destination, mode = volume.rsplit(":", 2)
        mounts.append({
            "Type": "bind", "Source": str(pathlib.Path(source).resolve()),
            "Destination": destination, "Mode": mode, "RW": mode == "rw",
            "Propagation": "rprivate",
        })
    identifier = os.environ["FAKE_CONTAINER_ID"]
    state["containers"][identifier] = {
        "Id": identifier, "Name": "/" + name,
        "Image": os.environ["FAKE_IMAGE_CONFIG"],
        "Config": {
            "Image": image, "Entrypoint": None, "Cmd": command,
            "Env": environment, "Labels": labels, "User": "", "WorkingDir": "",
        },
        "HostConfig": {
            "Runtime": runtime, "Privileged": False, "Devices": [],
            "DeviceRequests": [],
            "RestartPolicy": {"Name": restart, "MaximumRetryCount": 0},
            "NetworkMode": network,
        },
        "Mounts": mounts,
        "State": {"Status": "created", "Running": False, "ExitCode": 0,
                  "OOMKilled": False},
        "FakeLogs": "",
    }
    save(state)
    if os.environ.get("FAKE_DOCKER_CREATE_GATE") == "1":
        (root / "create.entered").touch()
        while not (root / "create.release").exists():
            import time; time.sleep(0.01)
    if os.environ.get("FAKE_DOCKER_CREATE_GAP") == "1":
        os.kill(os.getppid(), signal.SIGKILL)
    print(identifier)
elif args and args[0] == "start":
    identifier = args[-1]
    item = state["containers"][identifier]
    item["State"].update({"Status": "running", "Running": True, "ExitCode": 0})
    save(state)
    if os.environ.get("FAKE_DOCKER_START_GAP") == "1":
        os.kill(os.getppid(), signal.SIGKILL)
    if os.environ.get("FAKE_DOCKER_START_CLIENT_ERROR_AFTER_EFFECT") == "1":
        print("synthetic start client failure after daemon effect", file=sys.stderr)
        raise SystemExit(44)
    print(identifier)
elif args and args[0] == "inspect":
    identifier = args[-1]
    item = state["containers"].get(identifier)
    if item is None: raise SystemExit(1)
    if "-f" in args:
        template = args[args.index("-f") + 1]
        if template == "{{.Name}}": print(item["Name"])
        elif "ace.phantom.task" in template: print(item["Config"]["Labels"].get("ace.phantom.task", ""))
        elif "ace.phantom.intent-sha256" in template: print(item["Config"]["Labels"].get("ace.phantom.intent-sha256", ""))
        elif "ace.phantom.run-nonce" in template: print(item["Config"]["Labels"].get("ace.phantom.run-nonce", ""))
        else: raise SystemExit("unsupported inspect format: " + template)
    else:
        public = {key: value for key, value in item.items() if key != "FakeLogs"}
        if (os.environ.get("FAKE_DOCKER_CREATED_RECEIPT_GAP") == "1"
                and item["State"]["Status"] == "created"):
            sys.stdout.write('[{"partial":')
            sys.stdout.flush()
            os.kill(os.getppid(), signal.SIGKILL)
        if (os.environ.get("FAKE_DOCKER_STARTED_RECEIPT_GAP") == "1"
                and item["State"]["Status"] == "running"):
            sys.stdout.write('[{"partial":')
            sys.stdout.flush()
            os.kill(os.getppid(), signal.SIGKILL)
        print(json.dumps([public]))
elif args and args[0] == "logs":
    item = state["containers"].get(args[-1])
    if item is None: raise SystemExit(1)
    if (root / "fail-logs-once").exists():
        (root / "fail-logs-once").unlink()
        print("synthetic docker logs failure", file=sys.stderr)
        raise SystemExit(44)
    sys.stdout.write(item.get("FakeLogs", ""))
elif args[:2] == ["rm", "-f"]:
    identifier = args[-1]
    state["containers"].pop(identifier, None)
    save(state)
    if (root / "fail-rm-after-effect").exists():
        (root / "fail-rm-after-effect").unlink()
        raise SystemExit(88)
    print(identifier)
elif args and args[0] == "ps":
    filters = [args[index + 1] for index, value in enumerate(args) if value == "--filter"]
    for identifier, item in state["containers"].items():
        include = True
        for value in filters:
            if value.startswith("name="):
                expected = value.removeprefix("name=^/").removesuffix("$")
                include &= item["Name"] == "/" + expected
            elif value.startswith("label="):
                key, expected = value.removeprefix("label=").split("=", 1)
                include &= item["Config"]["Labels"].get(key) == expected
        if include: print(identifier)
else:
    raise SystemExit("unsupported fake Docker call: " + repr(args))
'''


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def commit(repo: Path) -> str:
    run(["git", "init", "-q"], repo)
    run(["git", "config", "user.email", "test@example.invalid"], repo)
    run(["git", "config", "user.name", "Detached Test"], repo)
    run(["git", "add", "."], repo)
    run(["git", "commit", "-qm", "fixture"], repo)
    return run(["git", "rev-parse", "HEAD"], repo)


class Harness:
    def __init__(self, root: Path, ace: Path, phantom: Path, ace_commit: str,
                 phantom_commit: str, fake_root: Path, env: dict[str, str]):
        self.root, self.ace, self.phantom = root, ace, phantom
        self.ace_commit, self.phantom_commit = ace_commit, phantom_commit
        self.fake_root, self.env = fake_root, env
        self.output = root / "evidence"
        self.wrapper = ace / "tools/phantom_gpu/freeze_bootstrap_host_evidence.sh"

    def invoke(self, *arguments: str, extra_env: dict[str, str] | None = None):
        environment = self.env | (extra_env or {})
        return subprocess.run(
            [str(self.wrapper), *arguments], cwd=self.ace, env=environment,
            text=True, capture_output=True, timeout=20,
        )

    def launch(
        self, create_gap: bool = False, start_gap: bool = False,
        created_receipt_gap: bool = False, started_receipt_gap: bool = False,
        start_client_error_after_effect: bool = False,
    ):
        return self.invoke(
            "--detach", "--ace-commit", self.ace_commit,
            "--phantom-commit", self.phantom_commit,
            *QUALIFICATION_ARGUMENTS, str(self.output),
            extra_env={
                "FAKE_DOCKER_CREATE_GAP": "1" if create_gap else "0",
                "FAKE_DOCKER_START_GAP": "1" if start_gap else "0",
                "FAKE_DOCKER_CREATED_RECEIPT_GAP": (
                    "1" if created_receipt_gap else "0"
                ),
                "FAKE_DOCKER_STARTED_RECEIPT_GAP": (
                    "1" if started_receipt_gap else "0"
                ),
                "FAKE_DOCKER_START_CLIENT_ERROR_AFTER_EFFECT": (
                    "1" if start_client_error_after_effect else "0"
                ),
            },
        )

    def launch_with_create_gate(self):
        environment = self.env | {
            "FAKE_DOCKER_CREATE_GAP": "0", "FAKE_DOCKER_START_GAP": "0",
            "FAKE_DOCKER_CREATE_GATE": "1",
        }
        return subprocess.Popen(
            [str(self.wrapper), "--detach", "--ace-commit", self.ace_commit,
             "--phantom-commit", self.phantom_commit,
             *QUALIFICATION_ARGUMENTS, str(self.output)],
            cwd=self.ace, env=environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def finalize(self):
        return self.invoke("--finalize", str(self.output))

    def docker_state(self):
        return json.loads((self.fake_root / "state.json").read_text())

    def set_terminal(self, exit_code: int, logs: str = "") -> None:
        value = self.docker_state()
        item = value["containers"][CONTAINER_ID]
        item["State"].update(
            {"Status": "exited", "Running": False, "ExitCode": exit_code}
        )
        item["FakeLogs"] = logs
        (self.fake_root / "state.json").write_text(json.dumps(value, sort_keys=True))

    def write_result(self, exit_code: int, sidecar: bytes | None = None) -> None:
        build = self.root / "result-build"
        if build.exists():
            shutil.rmtree(build)
        results = build / "results"
        results.mkdir(parents=True)
        status = "pass" if exit_code == 0 else "failed"
        (results / "pipeline-result.json").write_text(json.dumps({
            "schema_version": "1.0.0", "status": status,
            "mode": "bootstrap-host-freeze", "exit_code": exit_code,
            "started_utc": "2026-08-10T00:00:00Z",
            "completed_utc": "2026-08-10T00:00:01Z",
        }) + "\n")
        if exit_code == 0:
            (results / "result-completeness.json").write_text(json.dumps({
                "schema_version": "ace.phantom.result-completeness/1.0.0",
                "status": "pass", "mode": "bootstrap-host-freeze",
            }) + "\n")
            qualification = results / "qualification"
            qualification.mkdir()
            (qualification / "manifest.json").write_text(json.dumps({
                "status": "pass", "gate": "bootstrap", "source_mode": "snapshot",
                "ace_commit": self.ace_commit, "phantom_commit": self.phantom_commit,
                "ace_worktree_dirty": False, "gpu_executables_were_run": False,
            }) + "\n")
            shutil.copy2(self.output / "payload/ace-source.manifest.json",
                         qualification / "ace_source_manifest.json")
            shutil.copy2(self.output / "payload/phantom-source.manifest.json",
                         qualification / "phantom_source_manifest.json")
            semantic = qualification / "bootstrap_qualification"
            semantic.mkdir()
            digest_keys = {
                "generated_source_sha256", "raw_air_sha256", "post_ckks_air_sha256",
                "post_operations_air_sha256", "compiler_context_manifest_sha256",
                "compiler_resource_manifest_sha256", "compiler_constant_manifest_sha256",
                "generation_record_sha256", "generated_artifact_audit_sha256",
                "linked_binary_sha256", "harness_source_sha256", "symbol_closure_sha256",
                "io_helper_closure_sha256", "archive_member_audit_sha256",
                "adapter_archive_sha256", "provider_archive_sha256", "common_archive_sha256",
                "fixture_sha256", "native_ant_reference_sha256", "native_ant_values_sha256",
                "generated_ant_reference_sha256", "generated_ant_values_sha256",
                "gpu_correctness_runner_sha256", "gpu_correctness_runner_source_sha256",
                "host_oracle_replay_sha256",
            }
            qualification_record = {
                "schema_version": "ace.phantom.bootstrap-host-qualification/2.0.0",
                "status": "pass", "gate": "bootstrap",
                "architecture": "sm_80", "phantom_commit": self.phantom_commit,
                "development_image_id": IMAGE_CONFIG,
                "development_definition_sha256": hashlib.sha256(
                    (self.output / "payload/bootstrap_environment.sh").read_bytes()
                ).hexdigest(),
                "context_contract": {
                    "polynomial_degree": 16384, "vector_capacity": 8192,
                    "mul_level": 26, "input_level": 1, "security_level": 0,
                    "scaling_factor_bits": 56, "first_prime_bits": 60,
                    "hamming_weight": 192, "q_part_count": 3,
                },
                "generated_source_contains_native_bootstrap": False,
                "production_archive_contains_native_bootstrap": False,
                "primitive_only_provider_archive": True,
                "link_mode": "manual_static_closure",
                "host_oracle_executables_were_run": True,
                "gpu_executable_was_run": False,
                "executable_was_run": True,
                **{key: "0" * 64 for key in digest_keys},
            }
            (semantic / "qualification.json").write_text(json.dumps(
                qualification_record
            ) + "\n")
            self._write_sums(qualification)
            nested = results / "nested"
            nested.mkdir()
            (nested / "BUNDLE_SHA256SUMS").write_text(
                "nested manifest with the same basename remains included\n"
            )
        self._write_sums(results)
        incoming = self.output / "incoming-results"
        archive = incoming / "bootstrap-host-result.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(results, arcname="results")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (incoming / "bootstrap-host-result.tar.gz.sha256").write_bytes(
            sidecar if sidecar is not None else
            f"{digest}  bootstrap-host-result.tar.gz\n".encode()
        )

    def repack_result(self) -> None:
        results = self.root / "result-build/results"
        qualification = results / "qualification"
        if qualification.is_dir():
            self._write_sums(qualification)
        self._write_sums(results)
        incoming = self.output / "incoming-results"
        archive = incoming / "bootstrap-host-result.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(results, arcname="results")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (incoming / "bootstrap-host-result.tar.gz.sha256").write_text(
            f"{digest}  bootstrap-host-result.tar.gz\n"
        )

    @staticmethod
    def _write_sums(root: Path) -> None:
        lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != root / "SHA256SUMS":
                relative = path.relative_to(root).as_posix()
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
        (root / "SHA256SUMS").write_text("".join(lines))


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    phantom = tmp_path / "phantom"
    phantom.mkdir()
    (phantom / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\n")
    phantom_commit = commit(phantom)

    ace = tmp_path / "ace"
    tool_root = ace / "tools/phantom_gpu"
    config_root = tool_root / "configs"
    config_root.mkdir(parents=True)
    for name in (
        "freeze_bootstrap_host_evidence.sh", "transport_helpers.sh",
        "source_archive.py", "phase_helpers.sh", "bootstrap_environment.sh",
    ):
        shutil.copy2(TOOLS / name, tool_root / name)
    for name in (
        "apt-packages.lock", "python-requirements-hashed.lock", "base-files.sha256",
        "toolchain.env",
    ):
        (config_root / name).write_text("fixture\n")
    (config_root / "dependencies.env").write_text(
        f"CUDA_IMAGE={IMAGE}\nCUDA_IMAGE_CONFIG={IMAGE_CONFIG}\n"
        f"PHANTOM_COMMIT={phantom_commit}\n"
    )
    ace_commit = commit(ace)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    fake_root = tmp_path / "fake-docker"
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}", "ACE_PHANTOM_REPO": str(phantom),
        "FAKE_DOCKER_ROOT": str(fake_root), "FAKE_IMAGE_CONFIG": IMAGE_CONFIG,
        "FAKE_CONTAINER_ID": CONTAINER_ID, "ACE_PHANTOM_BUILD_JOBS": "2",
    })
    return Harness(tmp_path, ace, phantom, ace_commit, phantom_commit, fake_root, env)


def assert_bundle_closed(root: Path) -> None:
    assert not any(path.is_symlink() for path in root.rglob("*"))
    manifest = root / "BUNDLE_SHA256SUMS"
    listed = {}
    lines = manifest.read_text().splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        assert len(digest) == 64 and digest == digest.lower()
        assert name not in listed
        listed[name] = digest
    assert len(lines) == len(listed)
    actual = {
        "./" + path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    assert listed == actual


def test_detached_success_isolated_reentrant_and_checksum_closed(harness: Harness) -> None:
    launched = harness.launch()
    assert launched.returncode == 0, launched.stderr
    created = harness.docker_state()["containers"][CONTAINER_ID]
    mounts = {(item["Source"], item["Destination"], item["RW"])
              for item in created["Mounts"]}
    assert mounts == {
        (str((harness.output / "payload").resolve()), "/bootstrap-freeze/input", False),
        (str((harness.output / "incoming-results").resolve()), "/bootstrap-freeze/output", True),
    }
    assert all(item["Source"] != str(harness.output.resolve()) for item in created["Mounts"])
    harness.set_terminal(0, "finished\n")
    harness.write_result(0)
    finalized = harness.finalize()
    assert finalized.returncode == 0, finalized.stderr
    assert json.loads((harness.output / "lifecycle.json").read_text())["status"] == "pass"
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)
    install_intent = json.loads(
        (harness.output / "docker/bundle-install-intent.json").read_text()
    )
    bound_names = [item["path"] for item in install_intent["bound_files"]]
    assert "BUNDLE_SHA256SUMS" not in bound_names
    assert "docker/bundle-install-intent.json" not in bound_names
    assert (
        "verified-bootstrap-host-result/results/nested/BUNDLE_SHA256SUMS"
        in bound_names
    )
    (harness.output / "BUNDLE_SHA256SUMS").unlink()
    (harness.output / "docker/bundle-install-intent.json").unlink()
    (harness.output / "bootstrap-host-result-verification.json").unlink()
    shutil.rmtree(harness.output / "verified-bootstrap-host-result")
    pending = harness.output / ".verified-bootstrap-host-result.pending"
    pending.mkdir()
    (pending / "interrupted").write_text("partial extraction\n")
    resumed = harness.finalize()
    assert resumed.returncode == 0, resumed.stderr
    commands = [call[0] for call in harness.docker_state()["calls"]]
    assert commands.count("logs") == 1
    assert commands.count("rm") == 1
    assert_bundle_closed(harness.output)
    bundle_path = harness.output / "BUNDLE_SHA256SUMS"
    first_line = bundle_path.read_text().splitlines()[0] + "\n"
    bundle_path.write_text(first_line + first_line)
    recovered_bundle = harness.finalize()
    assert recovered_bundle.returncode == 0, recovered_bundle.stderr
    assert_bundle_closed(harness.output)
    lines = bundle_path.read_text().splitlines()
    lines[0] = "0" * 64 + lines[0][64:]
    bundle_path.write_text("\n".join(lines) + "\n")
    recovered_digest = harness.finalize()
    assert recovered_digest.returncode == 0, recovered_digest.stderr
    assert_bundle_closed(harness.output)
    calls_before = len(harness.docker_state()["calls"])
    repeated = harness.finalize()
    assert repeated.returncode == 0, repeated.stderr
    assert len(harness.docker_state()["calls"]) == calls_before


def test_running_then_terminal_missing_result_with_empty_log(harness: Harness) -> None:
    assert harness.launch().returncode == 0
    running = harness.finalize()
    assert running.returncode == 75
    assert CONTAINER_ID in harness.docker_state()["containers"]
    assert not (harness.output / "docker/container-completed.json").exists()
    harness.set_terminal(17, "")
    failed = harness.finalize()
    assert failed.returncode == 1, failed.stderr
    assert (harness.output / "bootstrap-host-freeze.log").read_bytes() == b""
    assert json.loads((harness.output / "docker/log-retrieval.json").read_text())["log_size"] == 0
    assert json.loads((harness.output / "lifecycle.json").read_text())["status"] == "failed"
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)


@pytest.mark.parametrize("sidecar", [
    b"0" * 64 + b"  wrong-name\n",
    b"0" * 64 + b"  bootstrap-host-result.tar.gz\nextra\n",
    b"A" * 64 + b"  bootstrap-host-result.tar.gz\n",
    b"0" * 64 + b"  bootstrap-host-result.tar.gz",
    b"0" * 64 + b"  bootstrap-host-result.tar.gz\n",
])
def test_malformed_sidecar_is_rejected_after_exact_cleanup(
    harness: Harness, sidecar: bytes
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(17)
    harness.write_result(17, sidecar=sidecar)
    result = harness.finalize()
    assert result.returncode == 1
    promotion = json.loads((harness.output / "docker/result-promotion.json").read_text())
    assert promotion["status"] == "rejected"
    assert list((harness.output / "incoming-results").iterdir()) == []
    assert not (harness.output / "bootstrap-host-result.tar.gz").exists()
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)


def test_untrusted_symlink_and_tampered_host_receipt_are_fail_closed(harness: Harness) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(17)
    incoming = harness.output / "incoming-results"
    (incoming / "bootstrap-host-result.tar.gz").symlink_to("/dev/null")
    (incoming / "bootstrap-host-result.tar.gz.sha256").write_text(
        "0" * 64 + "  bootstrap-host-result.tar.gz\n"
    )
    result = harness.finalize()
    assert result.returncode == 1
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert not any(path.is_symlink() for path in harness.output.rglob("*"))
    assert_bundle_closed(harness.output)

    second = harness.root / "second"
    second_harness = harness
    second_harness.output = second
    assert second_harness.launch().returncode == 0
    second_harness.set_terminal(17)
    state = second / "detached-launch-state.json"
    value = json.loads(state.read_text())
    value["run_nonce"] = "0" * 64
    state.write_text(json.dumps(value))
    refused = second_harness.finalize()
    assert refused.returncode != 0
    assert CONTAINER_ID in second_harness.docker_state()["containers"]


def test_finalize_lock_cleanup_recovery_and_create_gap(harness: Harness) -> None:
    assert harness.launch().returncode == 0
    lock_path = harness.output / "docker/finalize.lock"
    with lock_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = harness.finalize()
        assert locked.returncode == 75
    harness.set_terminal(17)
    harness.write_result(17)
    (harness.fake_root / "fail-rm-after-effect").touch()
    interrupted = harness.finalize()
    assert interrupted.returncode != 0
    resumed = harness.finalize()
    assert resumed.returncode == 17, resumed.stderr
    assert_bundle_closed(harness.output)

    gap = harness.root / "gap"
    harness.output = gap
    launched = harness.launch(create_gap=True)
    assert launched.returncode != 0
    assert (gap / "detached-precreate-intent.json").is_file()
    assert not (gap / "detached-launch-state.json").exists()
    (harness.fake_root / "fail-rm-after-effect").touch()
    interrupted_gap_cleanup = harness.finalize()
    assert interrupted_gap_cleanup.returncode != 0
    recovered = harness.finalize()
    assert recovered.returncode == 1, recovered.stderr
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(gap)


def test_launch_and_finalize_share_one_lock(harness: Harness) -> None:
    launch = harness.launch_with_create_gate()
    entered = harness.fake_root / "create.entered"
    deadline = time.monotonic() + 5
    while not entered.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entered.exists()
    competing = harness.finalize()
    assert competing.returncode == 75
    (harness.fake_root / "create.release").touch()
    stdout, stderr = launch.communicate(timeout=10)
    assert launch.returncode == 0, stderr
    assert "detached bootstrap host container" in stdout


@pytest.mark.parametrize("terminal_before_finalize", [False, True])
def test_post_start_cutoff_reconstructs_running_or_exited_launch(
    harness: Harness, terminal_before_finalize: bool
) -> None:
    cutoff = harness.launch(start_gap=True)
    assert cutoff.returncode != 0
    assert (harness.output / "detached-launch-state.json").is_file()
    assert not (harness.output / "docker/container-started.json").exists()
    if terminal_before_finalize:
        harness.set_terminal(17)
        harness.write_result(17)
        completed = harness.finalize()
        assert completed.returncode == 17, completed.stderr
        assert CONTAINER_ID not in harness.docker_state()["containers"]
        assert_bundle_closed(harness.output)
    else:
        running = harness.finalize()
        assert running.returncode == 75, running.stderr
        launch = json.loads((harness.output / "detached-launch.json").read_text())
        assert launch["status"] == "started" and launch["observed_running"] is True
        assert CONTAINER_ID in harness.docker_state()["containers"]
        harness.set_terminal(17)
        harness.write_result(17)
        completed = harness.finalize()
        assert completed.returncode == 17, completed.stderr
        assert_bundle_closed(harness.output)


def test_partial_created_receipt_is_atomically_recovered(harness: Harness) -> None:
    cutoff = harness.launch(created_receipt_gap=True)
    assert cutoff.returncode != 0
    created = harness.output / "docker/container-created.json"
    assert not created.exists()
    recovered = harness.finalize()
    assert recovered.returncode == 1, recovered.stderr
    assert json.loads(created.read_text())[0]["Id"] == CONTAINER_ID
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)


def test_partial_started_receipt_and_start_streams_are_recovered(
    harness: Harness,
) -> None:
    cutoff = harness.launch(started_receipt_gap=True)
    assert cutoff.returncode != 0
    assert not (harness.output / "docker/container-started.json").exists()
    start_result = json.loads(
        (harness.output / "docker/start-result.json").read_text()
    )
    assert start_result["status"] == "pass"
    running = harness.finalize()
    assert running.returncode == 75, running.stderr
    assert json.loads(
        (harness.output / "docker/container-started.json").read_text()
    )[0]["State"]["Status"] == "running"


def test_ambiguous_start_client_error_preserves_daemon_truth(
    harness: Harness,
) -> None:
    launched = harness.launch(start_client_error_after_effect=True)
    assert launched.returncode == 44, launched.stderr
    assert CONTAINER_ID in harness.docker_state()["containers"]
    start_result = json.loads(
        (harness.output / "docker/start-result.json").read_text()
    )
    launch = json.loads((harness.output / "detached-launch.json").read_text())
    assert start_result["status"] == "failed"
    assert start_result["client_exit_code"] == 44
    assert launch["status"] == "started"
    assert launch["start_client_status"] == "failed"
    running = harness.finalize()
    assert running.returncode == 75
    harness.set_terminal(17)
    harness.write_result(17)
    completed = harness.finalize()
    assert completed.returncode == 17, completed.stderr
    assert_bundle_closed(harness.output)


def test_partial_promotion_rejection_removes_root_and_incoming_artifacts(
    harness: Harness,
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(17)
    harness.write_result(17)
    (harness.fake_root / "fail-rm-after-effect").touch()
    assert harness.finalize().returncode != 0
    incoming = harness.output / "incoming-results"
    archive = incoming / "bootstrap-host-result.tar.gz"
    sidecar = incoming / "bootstrap-host-result.tar.gz.sha256"
    intent = {
        "schema_version": (
            "ace.phantom.bootstrap-host-result-promotion-intent/1.0.0"
        ),
        "status": "ready",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "archive_name": archive.name,
        "sidecar_name": sidecar.name,
    }
    (harness.output / "docker/result-promotion-intent.json").write_text(
        json.dumps(intent) + "\n"
    )
    os.replace(archive, harness.output / archive.name)
    sidecar.write_text("0" * 64 + "  bootstrap-host-result.tar.gz\n")
    rejected = harness.finalize()
    assert rejected.returncode == 1
    receipt = json.loads(
        (harness.output / "docker/result-promotion.json").read_text()
    )
    assert receipt["status"] == "rejected"
    assert list(incoming.iterdir()) == []
    assert not (harness.output / archive.name).exists()
    assert not (harness.output / sidecar.name).exists()
    assert_bundle_closed(harness.output)


def test_root_result_artifacts_without_intent_cannot_be_recorded_missing(
    harness: Harness,
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(17)
    harness.write_result(17)
    (harness.fake_root / "fail-rm-after-effect").touch()
    assert harness.finalize().returncode != 0
    incoming = harness.output / "incoming-results"
    for name in (
        "bootstrap-host-result.tar.gz",
        "bootstrap-host-result.tar.gz.sha256",
    ):
        os.replace(incoming / name, harness.output / name)
    rejected = harness.finalize()
    assert rejected.returncode == 1
    receipt = json.loads(
        (harness.output / "docker/result-promotion.json").read_text()
    )
    assert receipt["status"] == "rejected"
    assert list(incoming.iterdir()) == []
    assert not (harness.output / "bootstrap-host-result.tar.gz").exists()
    assert not (harness.output / "bootstrap-host-result.tar.gz.sha256").exists()
    assert_bundle_closed(harness.output)


@pytest.mark.parametrize("mutation", ["extra", "duplicate", "underspecified"])
def test_semantic_qualification_requires_exact_duplicate_free_schema(
    harness: Harness, mutation: str
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(0)
    harness.write_result(0)
    path = (
        harness.root / "result-build/results/qualification/"
        "bootstrap_qualification/qualification.json"
    )
    value = json.loads(path.read_text())
    if mutation == "extra":
        value["unexpected"] = True
        path.write_text(json.dumps(value) + "\n")
    elif mutation == "duplicate":
        raw = json.dumps(value)
        path.write_text(raw[:-1] + ', "status": "pass"}\n')
    else:
        path.write_text(json.dumps({
            "schema_version": value["schema_version"], "status": "pass",
            "gate": "bootstrap", "host_oracle_executables_were_run": True,
            "gpu_executable_was_run": False,
        }) + "\n")
    harness.repack_result()
    rejected = harness.finalize()
    assert rejected.returncode == 1
    lifecycle = json.loads((harness.output / "lifecycle.json").read_text())
    assert lifecycle["status"] == "failed"
    assert lifecycle["result_archive_status"] == "qualification-rejected"
    assert_bundle_closed(harness.output)


@pytest.mark.parametrize("mutation", ["extra_file", "empty_directory", "count"])
def test_verified_result_reuse_recomputes_exact_inventory_and_count(
    harness: Harness, mutation: str
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(0)
    harness.write_result(0)
    assert harness.finalize().returncode == 0
    (harness.output / "BUNDLE_SHA256SUMS").unlink()
    results = harness.output / "verified-bootstrap-host-result/results"
    if mutation == "extra_file":
        (results / "unlisted-extra.txt").write_text("unexpected\n")
    elif mutation == "empty_directory":
        (results / "unlisted-empty-directory").mkdir()
    else:
        receipt_path = harness.output / "bootstrap-host-result-verification.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["verified_file_count"] += 1
        receipt_path.write_text(json.dumps(receipt) + "\n")
    refused = harness.finalize()
    assert refused.returncode == 1


@pytest.mark.parametrize(
    "mutation", ["rejected_extra_key", "rejected_inventory_shape", "promoted_extra_key"]
)
def test_promotion_receipt_reentry_requires_exact_schema(
    harness: Harness, mutation: str
) -> None:
    assert harness.launch().returncode == 0
    if mutation.startswith("rejected"):
        harness.set_terminal(17)
        harness.write_result(
            17, sidecar=b"0" * 64 + b"  bootstrap-host-result.tar.gz\n"
        )
        assert harness.finalize().returncode == 1
    else:
        harness.set_terminal(0)
        harness.write_result(0)
        assert harness.finalize().returncode == 0
    receipt_path = harness.output / "docker/result-promotion.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "rejected_inventory_shape":
        receipt["incoming_inventory"][0]["unexpected"] = True
    else:
        receipt["unexpected"] = True
    receipt_path.write_text(json.dumps(receipt) + "\n")
    (harness.output / "BUNDLE_SHA256SUMS").unlink()
    refused = harness.finalize()
    assert refused.returncode == 1


@pytest.mark.parametrize("mutation", [
    "verification_intent_duplicate", "promotion_intent_duplicate",
    "bundle_intent_duplicate", "base_image_extra", "cleanup_text",
])
def test_bundle_install_intent_prevents_mutated_evidence_reclosure(
    harness: Harness, mutation: str
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(0)
    harness.write_result(0)
    assert harness.finalize().returncode == 0
    (harness.output / "BUNDLE_SHA256SUMS").unlink()
    if mutation == "verification_intent_duplicate":
        path = harness.output / "docker/result-verification-intent.json"
        raw = path.read_text()
        path.write_text('{"status":"changed",' + raw.lstrip()[1:])
    elif mutation == "promotion_intent_duplicate":
        path = harness.output / "docker/result-promotion-intent.json"
        raw = path.read_text()
        path.write_text('{"status":"changed",' + raw.lstrip()[1:])
    elif mutation == "bundle_intent_duplicate":
        path = harness.output / "docker/bundle-install-intent.json"
        raw = path.read_text()
        path.write_text('{"status":"changed",' + raw.lstrip()[1:])
    elif mutation == "base_image_extra":
        path = harness.output / "docker/base-image.json"
        value = json.loads(path.read_text())
        value[0]["unexpected"] = True
        path.write_text(json.dumps(value) + "\n")
    else:
        with (harness.output / "docker/cleanup.txt").open("a") as stream:
            stream.write("changed after closure\n")
    refused = harness.finalize()
    assert refused.returncode == 1
    assert not (harness.output / "BUNDLE_SHA256SUMS").exists()


def test_bundle_install_intent_temporary_is_recoverable_and_never_inventoried(
    harness: Harness,
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(0)
    harness.write_result(0)
    assert harness.finalize().returncode == 0
    (harness.output / "BUNDLE_SHA256SUMS").unlink()
    intent = harness.output / "docker/bundle-install-intent.json"
    intent.unlink()
    temporary = intent.with_name(intent.name + ".tmp")
    temporary.write_text("interrupted intent publication\n")
    recovered = harness.finalize()
    assert recovered.returncode == 0, recovered.stderr
    assert not temporary.exists()
    value = json.loads(intent.read_text())
    assert all(item["path"] != "docker/bundle-install-intent.json.tmp"
               for item in value["bound_files"])
    assert_bundle_closed(harness.output)


@pytest.mark.parametrize("field", [
    "command", "environment", "user", "working_directory", "devices",
    "device_requests", "restart", "network", "mount_mode", "mount_propagation",
])
def test_create_gap_rejects_any_changed_immutable_configuration(
    harness: Harness, field: str
) -> None:
    cutoff = harness.launch(create_gap=True)
    assert cutoff.returncode != 0
    state = harness.docker_state()
    item = state["containers"][CONTAINER_ID]
    if field == "command":
        item["Config"]["Cmd"] = ["bash", "-lc", "changed"]
    elif field == "environment":
        item["Config"]["Env"].append("UNEXPECTED=1")
    elif field == "user":
        item["Config"]["User"] = "1234"
    elif field == "working_directory":
        item["Config"]["WorkingDir"] = "/changed"
    elif field == "devices":
        item["HostConfig"]["Devices"] = [{"PathOnHost": "/dev/null"}]
    elif field == "device_requests":
        item["HostConfig"]["DeviceRequests"] = [{"Driver": "changed"}]
    elif field == "restart":
        item["HostConfig"]["RestartPolicy"] = {"Name": "always"}
    elif field == "network":
        item["HostConfig"]["NetworkMode"] = "host"
    elif field == "mount_mode":
        item["Mounts"][0]["Mode"] = "rw"
    elif field == "mount_propagation":
        item["Mounts"][0]["Propagation"] = "rshared"
    (harness.fake_root / "state.json").write_text(json.dumps(state, sort_keys=True))
    refused = harness.finalize()
    assert refused.returncode == 1
    assert CONTAINER_ID in harness.docker_state()["containers"]


def test_log_retrieval_failure_is_cleanup_and_bundle_closed(harness: Harness) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(0)
    harness.write_result(0)
    (harness.fake_root / "fail-logs-once").touch()
    failed = harness.finalize()
    assert failed.returncode == 1, failed.stderr
    receipt = json.loads((harness.output / "docker/log-retrieval.json").read_text())
    assert receipt["status"] == "failed" and receipt["log_size"] == 0
    assert (harness.output / "docker/log-retrieval-failure.txt").is_file()
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)
    calls_before = len(harness.docker_state()["calls"])
    repeated = harness.finalize()
    assert repeated.returncode == 1
    assert len(harness.docker_state()["calls"]) == calls_before


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_nonregular_incoming_entries_are_rejected_and_removed(
    harness: Harness, kind: str
) -> None:
    assert harness.launch().returncode == 0
    harness.set_terminal(17)
    incoming = harness.output / "incoming-results"
    archive = incoming / "bootstrap-host-result.tar.gz"
    if kind == "fifo":
        os.mkfifo(archive)
    else:
        archive.mkdir()
    (incoming / "bootstrap-host-result.tar.gz.sha256").write_text(
        "0" * 64 + "  bootstrap-host-result.tar.gz\n"
    )
    rejected = harness.finalize()
    assert rejected.returncode == 1
    assert list(incoming.iterdir()) == []
    assert CONTAINER_ID not in harness.docker_state()["containers"]
    assert_bundle_closed(harness.output)
