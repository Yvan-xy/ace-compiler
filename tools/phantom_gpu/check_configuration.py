#!/usr/bin/env python3
"""Verify pinned sources, shared profiles, and immutable mounted inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from typing import Any


PIN_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def verify_profile(repo_root: Path, profile: dict[str, Any]) -> None:
    if profile.get("schema_version") != "1.0.0":
        fail("unsupported Phantom profile schema")
    if profile.get("profile_id") != "fullpacked_bts_v1":
        fail("unexpected Phantom profile identifier")
    ring = profile["ring"]
    if ring != {
        "polynomial_degree": 16384,
        "active_slot_count": 8192,
        "packing_mode": "full",
    }:
        fail("ring profile does not match the frozen full-packed contract")

    data_q = profile["data_q"]
    if data_q["count"] != len(data_q["bit_sizes"]):
        fail("data-Q count does not match its bit-size list")
    if data_q["count"] != profile["depth"]["data_multiplication_depth"] + 1:
        fail("data-Q count does not match the declared multiplication depth")
    if data_q["bit_sizes"] != [60] + [56] * 26:
        fail("data-Q bit sizes do not match the frozen chain")

    special_p = profile["special_p"]
    if special_p["count"] != len(special_p["bit_sizes"]):
        fail("special-P count does not match its bit-size list")
    expected_p = math.ceil(
        (
            data_q["first_modulus_bits"]
            + (
                math.ceil(
                    special_p["derivation"]["data_depth"]
                    / special_p["derivation"]["q_parts"]
                )
                - 1
            )
            * data_q["scaling_modulus_bits"]
        )
        / 60
    )
    if special_p["count"] != expected_p or special_p["bit_sizes"] != [60] * expected_p:
        fail("special-P derivation does not match the frozen chain")

    constants = runpy.run_path(
        os.fspath(repo_root / "ace_edsl/examples/bootstrap_ant_constants.py")
    )
    coefficients = list(constants["G_COEFFICIENTS_UNIFORM_HW_192"])
    scalars = list(constants["get_double_angle_scalars"]())
    if profile["eval_mod"]["chebyshev_coefficients"] != coefficients:
        fail("Chebyshev coefficients differ from the shared EDSL constants")
    if profile["eval_mod"]["double_angle_scalars"] != scalars:
        fail("double-angle scalars differ from the shared EDSL constants")
    if profile["scale"]["post_bootstrap_multiplier"] != constants["BOOTSTRAP_POST_SCALE"]:
        fail("post-bootstrap multiplier differs from the shared EDSL constant")
    if profile["phantom"]["cuda_architecture"] != 80:
        fail("Phantom profile must target CUDA architecture 80")


def profile_codegen_parameters(profile: dict[str, Any]) -> dict[str, Any]:
    """Map the shared profile to the EDSL configuration contract."""
    return {
        "poly_degree": profile["ring"]["polynomial_degree"],
        "mul_level": profile["depth"]["data_multiplication_depth"],
        "input_level": profile["levels"]["input"]["logical_level"],
        "security_level": profile["security"]["validation_level_bits"],
        "scaling_factor_bits": profile["data_q"]["scaling_modulus_bits"],
        "first_prime_bits": profile["data_q"]["first_modulus_bits"],
        "hamming_weight": profile["security"]["secret_key_hamming_weight"],
        "ct_encode": profile["transforms"]["ciphertext_encoded_constants"],
    }


def render_cpp_profile_header(
    profile: dict[str, Any], profile_sha256: str
) -> str:
    """Render native Phantom parameters from the verified JSON profile."""
    if not SHA256_PATTERN.fullmatch(profile_sha256):
        fail("profile SHA-256 is malformed")

    def cpp_array(values: list[int]) -> str:
        return ", ".join(str(value) for value in values)

    data_q = profile["data_q"]["bit_sizes"]
    special_p = profile["special_p"]["bit_sizes"]
    return f"""#pragma once

#include <array>
#include <cstddef>

namespace ace::phantom_profile {{
inline constexpr char kProfileSha256[] = "{profile_sha256}";
inline constexpr std::size_t kPolynomialDegree = {profile['ring']['polynomial_degree']};
inline constexpr std::size_t kActiveSlotCount = {profile['ring']['active_slot_count']};
inline constexpr int kCudaArchitecture = {profile['phantom']['cuda_architecture']};
inline constexpr int kScalingModulusBits = {profile['data_q']['scaling_modulus_bits']};
inline constexpr int kSecretKeyHammingWeight = {profile['security']['secret_key_hamming_weight']};
inline constexpr std::array<int, {len(data_q)}> kDataQBitSizes = {{{cpp_array(data_q)}}};
inline constexpr std::array<int, {len(special_p)}> kSpecialPBitSizes = {{{cpp_array(special_p)}}};
}}  // namespace ace::phantom_profile
"""


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
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--cpp-header-output", type=Path)
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

    profile_path = (
        repo_root
        / "fhe-cmplr/rtlib/phantom/config/fullpacked_bts_v1.json"
    )
    profile = read_json(profile_path)
    verify_profile(repo_root, profile)
    profile_sha256 = sha256(profile_path)
    if arguments.cpp_header_output:
        arguments.cpp_header_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.cpp_header_output.write_text(
            render_cpp_profile_header(profile, profile_sha256), encoding="utf-8"
        )

    if arguments.source_mode == "snapshot":
        if arguments.base_image != lock["CUDA_IMAGE"]:
            fail("base image differs from the dependency lock")
        if not IMAGE_ID_PATTERN.fullmatch(arguments.base_config_digest or ""):
            fail("base image config digest is absent or malformed")
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
        "profile": {
            "path": os.fspath(profile_path),
            "sha256": profile_sha256,
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
