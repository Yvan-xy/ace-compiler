#!/usr/bin/env python3
"""Verify pinned sources, compiler context, and immutable mounted inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


PIN_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
PHANTOM_POLY_DEGREE_MAX = 131072
PHANTOM_USER_MODULUS_BITS_MIN = 2
PHANTOM_USER_MODULUS_BITS_MAX = 60
PHANTOM_COEFF_MODULUS_COUNT_MAX = 64


def fail(message: str) -> None:
    raise SystemExit(f"configuration check failed: {message}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")


def read_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(f"invalid dependency lock line {number}")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            fail(f"invalid dependency key on line {number}")
        values[key] = value
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(
            f"git {' '.join(arguments)} failed in {path}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def command_output(*arguments: str) -> str:
    result = subprocess.run(
        list(arguments), check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        fail(
            f"{' '.join(arguments)} failed with exit {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def verify_toolchain(tools_root: Path, repo_root: Path) -> dict[str, Any]:
    expected = read_lock(tools_root / "configs/toolchain.env")
    required = {
        "NVCC_VERSION",
        "CMAKE_VERSION",
        "CXX_VERSION",
        "PYTHON_VERSION",
        "NTL_PACKAGE_VERSION",
        "GMP_PACKAGE_VERSION",
    }
    if set(expected) != required:
        fail("toolchain expectation keys do not match the frozen schema")

    nvcc_output = command_output("/usr/local/cuda/bin/nvcc", "--version")
    nvcc_match = re.search(r"\bV([0-9]+\.[0-9]+\.[0-9]+)\b", nvcc_output)
    actual = {
        "NVCC_VERSION": nvcc_match.group(1) if nvcc_match else "",
        "CMAKE_VERSION": command_output("cmake", "--version")
        .splitlines()[0]
        .split()[-1],
        "CXX_VERSION": command_output("c++", "-dumpfullversion", "-dumpversion"),
        "PYTHON_VERSION": command_output(
            "python3", "-c", "import platform; print(platform.python_version())"
        ),
        "NTL_PACKAGE_VERSION": command_output(
            "dpkg-query", "-W", "-f=${Version}", "libntl-dev"
        ),
        "GMP_PACKAGE_VERSION": command_output(
            "dpkg-query", "-W", "-f=${Version}", "libgmp-dev"
        ),
    }
    for key, value in expected.items():
        if actual[key] != value:
            fail(f"{key} is {actual[key]!r}, expected {value!r}")

    python_lock = repo_root / "docker/phantom-a100/python-requirements.lock"
    expected_python = python_lock.read_text(encoding="utf-8").splitlines()
    installed_python = sorted(
        command_output("python3", "-m", "pip", "freeze", "--all").splitlines()
    )
    if installed_python != expected_python:
        fail("resolved Python environment differs from its lock file")
    command_output("python3", "-m", "pip", "check")
    return {
        "versions": actual,
        "python_lock_path": os.fspath(python_lock),
        "python_lock_sha256": sha256(python_lock),
    }


def require_path_below(directory: Path, relative: str) -> Path:
    root = directory.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        fail(f"input path escapes declared mount: {relative}")
    return candidate


def verify_input(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = require_path_below(directory, record["mount_relative_path"])
    size = path.stat().st_size
    digest = sha256(path)
    if size != record["size_bytes"]:
        fail(f"size mismatch for {path}: expected {record['size_bytes']}, got {size}")
    if digest != record["sha256"]:
        fail(f"SHA-256 mismatch for {path}: expected {record['sha256']}, got {digest}")
    return {"path": os.fspath(path), "size_bytes": size, "sha256": digest}


def verify_context_manifest(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "packing",
        "polynomial_degree",
        "logical_slot_capacity",
        "data_q_bit_sizes",
        "special_p_bit_sizes",
        "input_level",
        "q_part_count",
        "hamming_weight",
        "security_level",
        "first_modulus_bits",
        "scaling_modulus_bits",
        "resource_schema_version",
    }
    if set(manifest) != required:
        fail("compiler context manifest keys do not match schema")
    if manifest["schema_version"] != 1 or manifest["resource_schema_version"] != 3:
        fail("unsupported compiler context manifest schema")
    degree = manifest["polynomial_degree"]
    slots = manifest["logical_slot_capacity"]
    if (
        not isinstance(degree, int)
        or degree < 2
        or degree > PHANTOM_POLY_DEGREE_MAX
        or degree & (degree - 1)
        or manifest["packing"] != "full"
        or slots != degree // 2
    ):
        fail("invalid compiler-declared degree, packing, or slot capacity")
    data_q = manifest["data_q_bit_sizes"]
    special_p = manifest["special_p_bit_sizes"]
    if not data_q or not special_p or not all(
        isinstance(bits, int)
        and PHANTOM_USER_MODULUS_BITS_MIN
        <= bits
        <= PHANTOM_USER_MODULUS_BITS_MAX
        for bits in data_q + special_p
    ):
        fail("compiler context manifest has invalid Q/P arrays")
    if len(data_q) + len(special_p) > PHANTOM_COEFF_MODULUS_COUNT_MAX:
        fail("compiler context manifest Q/P count exceeds provider limits")
    if data_q[0] != manifest["first_modulus_bits"] or any(
        bits != manifest["scaling_modulus_bits"] for bits in data_q[1:]
    ):
        fail("compiler context manifest Q list disagrees with prime metadata")
    if not 1 <= manifest["input_level"] <= len(data_q):
        fail("compiler context manifest input level is invalid")
    if not 1 <= manifest["q_part_count"] <= len(data_q):
        fail("compiler context manifest Q-part count is invalid")
    if not 1 <= manifest["hamming_weight"] <= degree:
        fail("compiler context manifest hamming weight is invalid")
    if manifest["security_level"] not in (0, 128, 192, 256):
        fail("compiler context manifest security setting is invalid")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/app"))
    parser.add_argument("--models-dir", type=Path, default=Path("/inputs/models"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("/inputs/dataset"))
    parser.add_argument("--phantom-dir", type=Path, default=Path("/deps/phantom-ant"))
    parser.add_argument("--source-mode", choices=("git", "snapshot"), default="git")
    parser.add_argument("--ace-commit")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--base-image")
    parser.add_argument("--base-config-digest")
    parser.add_argument("--bootstrap-sha256")
    parser.add_argument("--context-manifest", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    repo_root = arguments.repo_root.resolve(strict=True)
    tools_root = repo_root / "tools/phantom_gpu"
    lock = read_lock(tools_root / "configs/dependencies.env")
    required_lock = {
        "ACE_BRANCH",
        "PHANTOM_BRANCH",
        "PHANTOM_COMMIT",
        "CUDA_ARCHITECTURES",
        "CUDA_IMAGE",
        "CUDA_IMAGE_CONFIG",
        "DEVELOPMENT_IMAGE",
    }
    if set(lock) != required_lock:
        fail("dependency lock keys do not match the frozen schema")
    if not PIN_PATTERN.fullmatch(lock["PHANTOM_COMMIT"]):
        fail("Phantom dependency is not pinned to a full commit")
    if lock["CUDA_ARCHITECTURES"] != "80":
        fail("dependency lock must target CUDA architecture 80")
    if "@sha256:" not in lock["CUDA_IMAGE"]:
        fail("CUDA image is not digest-pinned")
    if not IMAGE_ID_PATTERN.fullmatch(lock["CUDA_IMAGE_CONFIG"]):
        fail("CUDA image config digest is malformed")
    toolchain = verify_toolchain(tools_root, repo_root)

    phantom_root = arguments.phantom_dir.resolve(strict=True)
    if arguments.source_mode == "snapshot":
        if not PIN_PATTERN.fullmatch(arguments.ace_commit or ""):
            fail("snapshot ACE commit is absent or malformed")
        if not SHA256_PATTERN.fullmatch(arguments.source_manifest_sha256 or ""):
            fail("snapshot source manifest SHA-256 is absent or malformed")
        if (repo_root / ".git").exists() or (phantom_root / ".git").exists():
            fail("source snapshots must not contain Git metadata")
        for required in ("CMakeLists.txt", "include", "src"):
            if not (phantom_root / required).exists():
                fail(f"Phantom source snapshot is missing {required}")
        ace_branch = lock["ACE_BRANCH"]
        ace_commit = arguments.ace_commit
        phantom_branch = lock["PHANTOM_BRANCH"]
        phantom_head = lock["PHANTOM_COMMIT"]
        untracked_count = 0
    else:
        ace_branch = git(repo_root, "branch", "--show-current")
        if ace_branch != lock["ACE_BRANCH"]:
            fail(f"ACE branch is {ace_branch}, expected {lock['ACE_BRANCH']}")
        ace_commit = git(repo_root, "rev-parse", "HEAD")
        phantom_head = git(phantom_root, "rev-parse", "HEAD")
        phantom_branch = git(phantom_root, "branch", "--show-current")
        if phantom_head != lock["PHANTOM_COMMIT"]:
            fail(f"Phantom HEAD is {phantom_head}, expected {lock['PHANTOM_COMMIT']}")
        if phantom_branch != lock["PHANTOM_BRANCH"]:
            fail(f"Phantom branch is {phantom_branch}, expected {lock['PHANTOM_BRANCH']}")
        git(phantom_root, "cat-file", "-e", f"{lock['PHANTOM_COMMIT']}^{{commit}}")
        untracked_count = len(
            [
                line
                for line in git(
                    phantom_root, "status", "--porcelain", "--untracked-files=all"
                ).splitlines()
                if line.startswith("?? ")
            ]
        )

    context_manifest_path = arguments.context_manifest.resolve(strict=True)
    context_manifest = read_json(context_manifest_path)
    verify_context_manifest(context_manifest)
    context_manifest_sha256 = sha256(context_manifest_path)

    if arguments.source_mode == "snapshot":
        if arguments.base_image != lock["CUDA_IMAGE"]:
            fail("base image differs from the dependency lock")
        if not IMAGE_ID_PATTERN.fullmatch(arguments.base_config_digest or ""):
            fail("base image config digest is absent or malformed")
        if arguments.base_config_digest != lock["CUDA_IMAGE_CONFIG"]:
            fail("base image config digest differs from the dependency lock")
        if not SHA256_PATTERN.fullmatch(arguments.bootstrap_sha256 or ""):
            fail("bootstrap SHA-256 is absent or malformed")
        image_record = {
            "base_image": arguments.base_image,
            "config_digest": arguments.base_config_digest,
            "bootstrap_sha256": arguments.bootstrap_sha256,
        }
        fixture_record = None
    else:
        image_id = os.environ.get("ACE_PHANTOM_IMAGE_ID", "")
        definition_sha256 = os.environ.get("ACE_PHANTOM_DEFINITION_SHA256", "")
        if not IMAGE_ID_PATTERN.fullmatch(image_id):
            fail("development image ID is absent or malformed")
        if not SHA256_PATTERN.fullmatch(definition_sha256):
            fail("development image definition hash is absent or malformed")
        image_record = {"id": image_id, "definition_sha256": definition_sha256}

        fixture_path = tools_root / "configs/resnet20_cifar10_pre.json"
        fixture = read_json(fixture_path)
        if fixture.get("schema_version") != "1.0.0":
            fail("unsupported ResNet fixture schema")
        model = verify_input(arguments.models_dir, fixture["inputs"]["model"])
        dataset = verify_input(arguments.dataset_dir, fixture["inputs"]["dataset"])
        reference_path = repo_root / fixture["reference"]["log_path_at_capture"]
        reference: dict[str, Any] = {
            "path": os.fspath(reference_path),
            "present": False,
        }
        if reference_path.is_file():
            reference.update(
                {
                    "present": True,
                    "size_bytes": reference_path.stat().st_size,
                    "sha256": sha256(reference_path),
                }
            )
            if reference["size_bytes"] != fixture["reference"]["log_size_bytes"]:
                fail("preserved reference log size differs from its manifest")
            if reference["sha256"] != fixture["reference"]["log_sha256"]:
                fail("preserved reference log hash differs from its manifest")
        fixture_record = {
            "path": os.fspath(fixture_path),
            "sha256": sha256(fixture_path),
            "model": model,
            "dataset": dataset,
            "reference_log": reference,
        }

    report = {
        "status": "pass",
        "source_mode": arguments.source_mode,
        "ace": {
            "branch": ace_branch,
            "commit": ace_commit,
        },
        "phantom": {
            "branch": phantom_branch,
            "commit": phantom_head,
            "untracked_file_count": untracked_count,
            "build_source_policy": (
                "use the checksum-verified source snapshot"
                if arguments.source_mode == "snapshot"
                else "clone the pinned commit; never build the mounted worktree"
            ),
        },
        "cuda_architecture": 80,
        "environment_identity": image_record,
        "source_manifest_sha256": arguments.source_manifest_sha256,
        "toolchain": toolchain,
        "compiler_context_manifest": {
            "path": os.fspath(context_manifest_path),
            "sha256": context_manifest_sha256,
        },
        "fixture": fixture_record,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
