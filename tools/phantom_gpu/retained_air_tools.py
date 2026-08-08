"""Shared canonical AIR and retained-generation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
