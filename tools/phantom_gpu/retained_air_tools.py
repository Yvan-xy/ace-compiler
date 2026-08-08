"""Shared canonical AIR and retained-generation helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
from typing import Any


_STRING = re.compile(
    r'^(?P<prefix>\s*STR\[[^]]+\]\s+")(?P<value>[^"]*)'
    r'(?P<suffix>"\s+length\(0x)(?P<size>[0-9a-fA-F]+)(?P<end>\).*)$'
)


def canonicalize_checkout_paths(air: str, repository: Path) -> str:
    """Replace checkout-local AIR string entries with repository paths."""

    root = repository.resolve().as_posix().rstrip("/") + "/"
    delta = 0
    output: list[str] = []
    for line in air.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        match = _STRING.match(body)
        if match is None:
            output.append(line)
            continue
        value = match.group("value")
        if value.startswith(root):
            canonical = value[len(root) :]
            old_size = int(match.group("size"), 16)
            if old_size != len(value.encode("utf-8")):
                raise ValueError("AIR string-table entry has an invalid byte length")
            new_size = len(canonical.encode("utf-8"))
            delta += new_size - old_size
            body = (
                match.group("prefix")
                + canonical
                + match.group("suffix")
                + format(new_size, "x")
                + match.group("end")
            )
        output.append(body + newline)

    canonical_air = "".join(output)
    header = re.match(r"STRING TABLE \(([0-9]+) Bytes\)\n", canonical_air)
    if header is None:
        raise ValueError("AIR is missing its canonical string-table header")
    size = int(header.group(1)) + delta
    if size < 0:
        raise ValueError("AIR string-table size is inconsistent")
    canonical_air = (
        f"STRING TABLE ({size} Bytes)\n" + canonical_air[header.end() :]
    )
    for line in canonical_air.splitlines():
        match = _STRING.match(line)
        if match is not None and match.group("value").startswith("/"):
            raise ValueError("absolute path remains in canonical AIR")
    return canonical_air


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_clean_tracked_sources(repository: Path, paths: tuple[str, ...]) -> str:
    """Return HEAD only when every generation input is tracked and clean."""

    if os.environ.get("ACE_PHANTOM_SOURCE_MODE") == "snapshot":
        return _require_audited_snapshot_sources(repository, paths)

    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *paths],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise ValueError("generation inputs are not a clean HEAD snapshot: " + status)
    return commit


def _require_audited_snapshot_sources(
    repository: Path, paths: tuple[str, ...]
) -> str:
    """Authenticate selected inputs in a source-only ACE commit archive."""

    commit = os.environ.get("ACE_PHANTOM_ACE_COMMIT", "")
    manifest_name = os.environ.get("ACE_PHANTOM_SOURCE_MANIFEST", "")
    expected_manifest_sha256 = os.environ.get(
        "ACE_PHANTOM_SOURCE_MANIFEST_SHA256", ""
    )
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("snapshot generation requires an exact ACE commit")
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError("snapshot generation requires an ACE manifest SHA-256")
    if (repository / ".git").exists():
        raise ValueError("source-only snapshot unexpectedly contains Git metadata")
    manifest_path = Path(manifest_name)
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read ACE snapshot manifest: {error}") from error
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("ACE snapshot manifest hash differs from its audited binding")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate ACE snapshot manifest key: {key}")
            value[key] = item
        return value

    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ACE snapshot manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "1.0.0"
        or manifest.get("kind") != "ace"
        or manifest.get("source_method") != "git-commit-object-archive"
        or manifest.get("commit") != commit
        or re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("tree", ""))) is None
        or not isinstance(manifest.get("members"), list)
    ):
        raise ValueError("ACE snapshot manifest identity is invalid")

    file_members: dict[str, dict[str, Any]] = {}
    for raw in manifest["members"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ValueError("ACE snapshot manifest contains a malformed member")
        member_path = PurePosixPath(raw["path"])
        if (
            member_path.is_absolute()
            or not member_path.parts
            or member_path.parts[0] != "ace-source"
            or any(part in {"", ".", ".."} for part in member_path.parts)
        ):
            raise ValueError("ACE snapshot manifest contains an unsafe member path")
        relative = PurePosixPath(*member_path.parts[1:]).as_posix()
        if relative in file_members:
            raise ValueError("ACE snapshot manifest contains duplicate member paths")
        if raw.get("type") == "file":
            digest = raw.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError("ACE snapshot file member has an invalid SHA-256")
            file_members[relative] = raw

    for requested_name in paths:
        requested = PurePosixPath(requested_name)
        if (
            requested.is_absolute()
            or not requested.parts
            or any(part in {"", ".", ".."} for part in requested.parts)
        ):
            raise ValueError(f"unsafe audited snapshot input: {requested_name}")
        prefix = requested.as_posix()
        selected = {
            name: record
            for name, record in file_members.items()
            if name == prefix or name.startswith(prefix + "/")
        }
        if not selected:
            raise ValueError(
                f"generation input is absent from the ACE snapshot: {requested_name}"
            )
        for relative_name, record in sorted(selected.items()):
            physical = repository.joinpath(*PurePosixPath(relative_name).parts)
            try:
                observed = hashlib.sha256(physical.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(
                    f"cannot read audited snapshot input {relative_name}: {error}"
                ) from error
            if observed != record["sha256"]:
                raise ValueError(
                    f"audited snapshot input differs from its manifest: {relative_name}"
                )
    return commit


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
