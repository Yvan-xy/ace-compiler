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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/app"))
    parser.add_argument("--models-dir", type=Path, default=Path("/inputs/models"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("/inputs/dataset"))
    parser.add_argument("--phantom-dir", type=Path, default=Path("/deps/phantom-ant"))
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

    ace_branch = git(repo_root, "branch", "--show-current")
    if ace_branch != lock["ACE_BRANCH"]:
        fail(f"ACE branch is {ace_branch}, expected {lock['ACE_BRANCH']}")

    phantom_root = arguments.phantom_dir.resolve(strict=True)
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

    fixture_path = tools_root / "configs/resnet20_cifar10_pre.json"
    fixture = read_json(fixture_path)
    if fixture.get("schema_version") != "1.0.0":
        fail("unsupported ResNet fixture schema")
    model = verify_input(arguments.models_dir, fixture["inputs"]["model"])
    dataset = verify_input(arguments.dataset_dir, fixture["inputs"]["dataset"])

    reference_path = repo_root / fixture["reference"]["log_path_at_capture"]
    reference: dict[str, Any] = {"path": os.fspath(reference_path), "present": False}
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

    report = {
        "status": "pass",
        "ace": {
            "branch": ace_branch,
            "commit": git(repo_root, "rev-parse", "HEAD"),
        },
        "phantom": {
            "branch": phantom_branch,
            "commit": phantom_head,
            "untracked_file_count": untracked_count,
            "build_source_policy": "clone the pinned commit; never build the mounted worktree",
        },
        "cuda_architecture": 80,
        "profile": {
            "path": os.fspath(profile_path),
            "sha256": sha256(profile_path),
        },
        "fixture": {
            "path": os.fspath(fixture_path),
            "sha256": sha256(fixture_path),
            "model": model,
            "dataset": dataset,
            "reference_log": reference,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.json_output:
        arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
