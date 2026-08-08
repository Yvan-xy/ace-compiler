#!/usr/bin/env bash

shell_join() {
  if [[ $# -lt 2 ]]; then
    echo "shell_join requires an output variable and at least one argument" >&2
    return 2
  fi
  local output_variable="$1"
  local joined
  shift
  printf -v joined '%q ' "$@"
  printf -v "${output_variable}" '%s' "${joined% }"
}

verify_result_archive() {
  if [[ $# -ne 4 ]]; then
    echo "verify_result_archive requires ARCHIVE MODE EXIT_CODE EXTRACT_DIRECTORY" >&2
    return 2
  fi
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

archive_path = Path(sys.argv[1])
expected_mode = sys.argv[2]
try:
    expected_exit = int(sys.argv[3])
except ValueError as error:
    raise SystemExit("expected pipeline exit code is not an integer") from error
extract_root = Path(sys.argv[4])
if extract_root.exists():
    raise SystemExit(f"verified result extraction path already exists: {extract_root}")
extract_root.mkdir(parents=True, mode=0o700)

def safe_result_path(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0] != "results"
        or str(path) != name.rstrip("/")
    ):
        raise SystemExit(f"unsafe result archive member: {name!r}")
    return path

seen = set()
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    for member in members:
        path = safe_result_path(member.name)
        canonical = str(path)
        if canonical in seen:
            raise SystemExit(f"duplicate result archive member: {canonical}")
        seen.add(canonical)
        if not (member.isdir() or member.isfile()):
            raise SystemExit(f"unsupported result archive member type: {canonical}")
        target = extract_root.joinpath(*path.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise SystemExit(f"could not read result archive member: {canonical}")
        with target.open("xb") as output:
            shutil.copyfileobj(source, output)

results = extract_root / "results"
sums_path = results / "SHA256SUMS"
pipeline_path = results / "pipeline-result.json"
if not sums_path.is_file() or not pipeline_path.is_file():
    raise SystemExit("result archive lacks SHA256SUMS or pipeline-result.json")
listed = {}
for line in sums_path.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
    if match is None:
        raise SystemExit("result SHA256SUMS contains a malformed entry")
    name = match.group(3).removeprefix("./")
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or name == "SHA256SUMS"
    ):
        raise SystemExit("result SHA256SUMS contains an unsafe entry")
    if name in listed:
        raise SystemExit("result SHA256SUMS contains a duplicate entry")
    listed[name] = match.group(1)
actual = {
    str(path.relative_to(results)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in results.rglob("*")
    if path.is_file() and path != sums_path
}
if listed != actual:
    raise SystemExit(
        "result SHA256SUMS is incomplete or a result file hash mismatched"
    )

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in pipeline result: {key}")
        value[key] = item
    return value

pipeline = json.loads(
    pipeline_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
if not isinstance(pipeline, dict) or set(pipeline) != {
    "schema_version", "status", "mode", "exit_code",
    "started_utc", "completed_utc",
}:
    raise SystemExit("pipeline result has an invalid shape")
expected_status = "pass" if expected_exit == 0 else "failed"
if (
    pipeline["schema_version"] != "1.0.0"
    or pipeline["mode"] != expected_mode
    or pipeline["exit_code"] != expected_exit
    or pipeline["status"] != expected_status
    or not isinstance(pipeline["started_utc"], str)
    or not isinstance(pipeline["completed_utc"], str)
):
    raise SystemExit("pipeline result does not match the observed invocation")
print(
    json.dumps(
        {
            "status": "pass",
            "mode": expected_mode,
            "pipeline_exit_code": expected_exit,
            "verified_file_count": len(actual),
            "sha256_manifest_sha256": hashlib.sha256(
                sums_path.read_bytes()
            ).hexdigest(),
        },
        sort_keys=True,
    )
)
PY
}
