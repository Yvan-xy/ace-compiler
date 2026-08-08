#!/usr/bin/env python3
"""Create, audit, and safely extract deterministic Git source archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SENSITIVE_NAMES = {
    ".env",
    "authorized_keys",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "known_hosts",
}
SENSITIVE_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}
DISALLOWED_COMPONENTS = {
    ".aws",
    ".git",
    ".gnupg",
    ".ssh",
    "build",
    "dataset",
    "datasets",
    "model_and_main",
    "models",
    "pretrained_parameters",
}
DISALLOWED_DATA_SUFFIXES = {
    ".csv",
    ".msg",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".tflite",
}
ARCHIVE_POLICIES = {
    "ace": {
        "prefix": "ace-source",
        "allowed": (
            "ace_bindings",
            "ace_edsl",
            "acepy",
            "air-infra",
            "bindings",
            "docker/phantom-a100",
            "fhe-cmplr",
            "nn-addon",
            "tools/phantom_gpu",
        ),
        "excluded": (
            "fhe-cmplr/rtlib/ant/dataset",
            "fhe-cmplr/rtlib/ant/imagenet",
        ),
    },
    "phantom": {
        "prefix": "phantom-source",
        "allowed": (
            "CMakeLists.txt",
            "cmake",
            "include",
            "src",
            "tests/CMakeLists.txt",
            "tests/ckks_retained_primitives.cu",
            "tests/compiler_context_manifest.h",
            "tests/native_bts_oracle_link.cu",
        ),
        "excluded": (),
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"source archive check failed: {message}")


def sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)[0]


def run_git(repo: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *arguments],
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(errors="replace")
        fail(f"git {' '.join(arguments)} failed in {repo}: {stderr}")
    return result.stdout


def normalized_member_path(name: str) -> PurePosixPath:
    if "\\" in name or name.startswith("/"):
        fail(f"unsafe archive path {name!r}")
    path = PurePosixPath(name)
    if not name or any(part in {"", ".", ".."} for part in path.parts):
        fail(f"unsafe archive path {name!r}")
    return path


def policy_path(member: PurePosixPath, kind: str) -> PurePosixPath:
    policy = ARCHIVE_POLICIES[kind]
    prefix = policy["prefix"]
    if not member.parts or member.parts[0] != prefix:
        fail(f"archive member is outside {prefix}/: {member}")
    relative = PurePosixPath(*member.parts[1:])
    if not relative.parts:
        return relative
    if not any(
        relative == PurePosixPath(allowed)
        or PurePosixPath(allowed) in relative.parents
        or relative in PurePosixPath(allowed).parents
        for allowed in policy["allowed"]
    ):
        fail(f"archive member is outside the {kind} allowlist: {relative}")
    for excluded in policy["excluded"]:
        excluded_path = PurePosixPath(excluded)
        if relative == excluded_path or excluded_path in relative.parents:
            fail(f"excluded source path is present: {relative}")
    lowered = [part.lower() for part in relative.parts]
    if DISALLOWED_COMPONENTS.intersection(lowered):
        fail(f"disallowed source path component is present: {relative}")
    if relative.name.lower() in SENSITIVE_NAMES:
        fail(f"sensitive filename is present: {relative}")
    if relative.suffix.lower() in SENSITIVE_SUFFIXES | DISALLOWED_DATA_SUFFIXES:
        fail(f"sensitive or data file is present: {relative}")
    if relative.name.startswith("._"):
        fail(f"AppleDouble file is present: {relative}")
    return relative


def mode_string(member: tarfile.TarInfo) -> str:
    return f"{stat.S_IMODE(member.mode):04o}"


def audit_archive(
    archive: Path,
    kind: str,
    expected_manifest: dict[str, Any] | None = None,
    extract_to: Path | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    regular_bytes = 0
    if expected_manifest is not None:
        if expected_manifest.get("schema_version") != "1.0.0":
            fail("unsupported source manifest schema")
        if expected_manifest.get("kind") != kind:
            fail("source manifest kind does not match archive")
        if expected_manifest.get("archive_sha256") != sha256_path(archive):
            fail("outer archive SHA-256 does not match source manifest")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            path = normalized_member_path(member.name.rstrip("/"))
            relative = policy_path(path, kind)
            canonical_name = path.as_posix()
            if canonical_name in names:
                fail(f"duplicate archive member {canonical_name}")
            names.add(canonical_name)
            record: dict[str, Any] = {
                "path": canonical_name,
                "mode": mode_string(member),
                "size": member.size,
            }
            if member.isdir():
                record["type"] = "directory"
                if member.size != 0:
                    fail(f"directory has nonzero size: {canonical_name}")
            elif member.isfile():
                record["type"] = "file"
                extracted = tar.extractfile(member)
                if extracted is None:
                    fail(f"cannot read archive member {canonical_name}")
                record["sha256"], actual_size = sha256_stream(extracted)
                if actual_size != member.size:
                    fail(f"size mismatch while reading {canonical_name}")
                regular_bytes += actual_size
            elif member.issym():
                record["type"] = "symlink"
                record["link_target"] = member.linkname
                if member.size != 0:
                    fail(f"symlink has nonzero size: {canonical_name}")
                target = PurePosixPath(member.linkname)
                if target.is_absolute():
                    fail(f"absolute symlink target: {canonical_name}")
                resolved_parts: list[str] = list(relative.parent.parts)
                for part in target.parts:
                    if part in {"", "."}:
                        continue
                    if part == "..":
                        if not resolved_parts:
                            fail(f"escaping symlink target: {canonical_name}")
                        resolved_parts.pop()
                    else:
                        resolved_parts.append(part)
            else:
                fail(f"unsupported archive entry type: {canonical_name}")
            records.append(record)

        records.sort(key=lambda item: item["path"])
        if expected_manifest is not None:
            if expected_manifest.get("members") != records:
                fail("archive members do not match source manifest")
        if extract_to is not None:
            extract_to.mkdir(parents=True, exist_ok=False)
            root = extract_to.resolve(strict=True)
            for member in members:
                path = normalized_member_path(member.name.rstrip("/"))
                target = root.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(stat.S_IMODE(member.mode))
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        fail(f"cannot extract {member.name}")
                    with target.open("xb") as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(chunk)
                    target.chmod(stat.S_IMODE(member.mode))
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(member.linkname)
    return {
        "members": records,
        "member_count": len(records),
        "regular_bytes": regular_bytes,
    }


def create_archive(arguments: argparse.Namespace) -> None:
    repo = arguments.repo.resolve(strict=True)
    commit = str(run_git(repo, "rev-parse", f"{arguments.commit}^{{commit}}")).strip()
    if not COMMIT_PATTERN.fullmatch(commit):
        fail("resolved Git revision is not a full commit")
    tree = str(run_git(repo, "rev-parse", f"{commit}^{{tree}}")).strip()
    commit_timestamp = int(str(run_git(repo, "show", "-s", "--format=%ct", commit)).strip())
    policy = ARCHIVE_POLICIES[arguments.kind]
    allowed_paths = []
    for path in policy["allowed"]:
        present = subprocess.run(
            ["git", "-C", os.fspath(repo), "cat-file", "-e", f"{commit}:{path}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if present.returncode == 0:
            allowed_paths.append(path)
    if not allowed_paths:
        fail(f"no {arguments.kind} allowlisted paths exist at {commit}")
    command = [
        "git",
        "-C",
        os.fspath(repo),
        "archive",
        "--format=tar",
        f"--prefix={policy['prefix']}/",
        commit,
        "--",
        *allowed_paths,
        *[f":(exclude){path}" for path in policy["excluded"]],
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        fail(result.stderr.decode(errors="replace").strip())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
            zipped.write(result.stdout)
    audit = audit_archive(arguments.output, arguments.kind)
    manifest = {
        "schema_version": "1.0.0",
        "kind": arguments.kind,
        "source_method": "git-commit-object-archive",
        "commit": commit,
        "commit_timestamp": commit_timestamp,
        "tree": tree,
        "archive": arguments.output.name,
        "archive_size": arguments.output.stat().st_size,
        "archive_sha256": sha256_path(arguments.output),
        "allowed_paths": list(policy["allowed"]),
        "excluded_paths": list(policy["excluded"]),
        **audit,
    }
    arguments.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def audit_command(arguments: argparse.Namespace) -> None:
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    if not SHA256_PATTERN.fullmatch(str(manifest.get("archive_sha256", ""))):
        fail("source manifest archive SHA-256 is malformed")
    report = audit_archive(arguments.archive, arguments.kind, manifest, arguments.extract)
    print(
        json.dumps(
            {
                "status": "pass",
                "kind": arguments.kind,
                "archive_sha256": sha256_path(arguments.archive),
                "member_count": report["member_count"],
                "regular_bytes": report["regular_bytes"],
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--kind", choices=sorted(ARCHIVE_POLICIES), required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.set_defaults(function=create_archive)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--archive", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--kind", choices=sorted(ARCHIVE_POLICIES), required=True)
    audit.add_argument("--extract", type=Path)
    audit.set_defaults(function=audit_command)
    return root


def main() -> int:
    arguments = parser().parse_args()
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
