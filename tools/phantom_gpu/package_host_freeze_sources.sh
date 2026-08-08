#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"

usage() {
  echo "usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --qualification-invocation FILE OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
QUALIFICATION_INVOCATION_ARGUMENT=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ace-commit)
      [[ $# -ge 2 ]] || usage
      ACE_COMMIT="$2"
      shift 2
      ;;
    --phantom-commit)
      [[ $# -ge 2 ]] || usage
      PHANTOM_COMMIT="$2"
      shift 2
      ;;
    --qualification-invocation)
      [[ $# -ge 2 ]] || usage
      QUALIFICATION_INVOCATION_ARGUMENT="$2"
      shift 2
      ;;
    --)
      shift
      [[ $# -eq 1 && -z "${OUTPUT_ARGUMENT}" ]] || usage
      OUTPUT_ARGUMENT="$1"
      shift
      ;;
    -*) usage ;;
    *)
      [[ -z "${OUTPUT_ARGUMENT}" ]] || usage
      OUTPUT_ARGUMENT="$1"
      shift
      ;;
  esac
done
[[ -n "${ACE_COMMIT}" && -n "${PHANTOM_COMMIT}" &&
   -n "${QUALIFICATION_INVOCATION_ARGUMENT}" &&
   -n "${OUTPUT_ARGUMENT}" ]] || usage
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commits must be full lowercase 40-character object IDs" >&2
  exit 1
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"

DEPENDENCY_LOCK="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
)"
lock_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" <<<"${DEPENDENCY_LOCK}")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "selected ACE commit has an invalid ${key} dependency lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
LOCKED_PHANTOM_COMMIT="$(lock_value PHANTOM_COMMIT)"
if [[ "${PHANTOM_COMMIT}" != "${LOCKED_PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit does not match the selected ACE commit lock" >&2
  echo "requested: ${PHANTOM_COMMIT}" >&2
  echo "locked:    ${LOCKED_PHANTOM_COMMIT}" >&2
  exit 1
fi

QUALIFICATION_INVOCATION="$(realpath -- "${QUALIFICATION_INVOCATION_ARGUMENT}")"
python3 - "${QUALIFICATION_INVOCATION}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"qualification invocation contains duplicate key: {key}")
        value[key] = item
    return value

record = json.loads(
    path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
)
if not isinstance(record, dict) or set(record) != {
    "schema_version", "argv", "normalized_argv_sha256"
}:
    raise SystemExit("qualification invocation has an invalid shape")
if record["schema_version"] != "ace.phantom.qualification-invocation/1.0.0":
    raise SystemExit("qualification invocation has an unsupported schema")
argv = record["argv"]
if not isinstance(argv, list) or not all(
    isinstance(value, str) and value for value in argv
):
    raise SystemExit("qualification invocation argv must be a nonempty string array")
normalized = json.dumps(
    argv, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")
if hashlib.sha256(normalized).hexdigest() != record["normalized_argv_sha256"]:
    raise SystemExit("qualification invocation normalized hash mismatch")
if not argv or argv[0] != "tools/phantom_gpu/compile_only.sh":
    raise SystemExit("qualification invocation has an unexpected executable")
arguments = argv[1:]
if len(arguments) % 2:
    raise SystemExit("qualification invocation has an argument without a value")
pairs = dict(zip(arguments[0::2], arguments[1::2]))
expected = {
    "--gate", "--poly-degree", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight",
}
if len(pairs) != len(arguments) // 2 or set(pairs) != expected:
    raise SystemExit("qualification invocation options are incomplete or duplicated")
if pairs["--gate"] != "ordinary":
    raise SystemExit("qualification invocation is not the ordinary qualification")
for option in expected - {"--gate"}:
    if re.fullmatch(r"[0-9]+", pairs[option]) is None:
        raise SystemExit(f"qualification invocation {option} is not an integer")
PY

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"

ARCHIVE_TOOL_DIRECTORY="$(mktemp -d)"
cleanup_archive_tool() {
  rm -f -- "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"
  rmdir -- "${ARCHIVE_TOOL_DIRECTORY}"
}
trap cleanup_archive_tool EXIT
git -C "${REPO_ROOT}" show \
  "${ACE_COMMIT}:tools/phantom_gpu/source_archive.py" \
  >"${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"

ACE_ARCHIVE="ace-source-${ACE_COMMIT}.tar.gz"
PHANTOM_ARCHIVE="phantom-source-${PHANTOM_COMMIT}.tar.gz"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${REPO_ROOT}" \
  --commit "${ACE_COMMIT}" \
  --kind ace \
  --output "${OUTPUT}/${ACE_ARCHIVE}" \
  --manifest "${OUTPUT}/ace-source.manifest.json" \
  >"${OUTPUT}/ace-source.audit.json"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" \
  --commit "${PHANTOM_COMMIT}" \
  --kind phantom \
  --output "${OUTPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${OUTPUT}/phantom-source.manifest.json" \
  >"${OUTPUT}/phantom-source.audit.json"

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" \
    >"${OUTPUT}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/run_build_and_health.sh
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 \
  "${OUTPUT}/source_archive.py" \
  "${OUTPUT}/phase_helpers.sh" \
  "${OUTPUT}/bootstrap_environment.sh" \
  "${OUTPUT}/run_build_and_health.sh"
if ! grep -Fq 'freeze-host' "${OUTPUT}/run_build_and_health.sh"; then
  echo "selected ACE commit does not implement the host-freeze runner mode" >&2
  exit 1
fi
cp "${QUALIFICATION_INVOCATION}" "${OUTPUT}/qualification-invocation.json"

python3 - "${OUTPUT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
invocation_path = output / "qualification-invocation.json"
invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
payload = {
    "schema_version": "ace.phantom.host-freeze-payload/1.0.0",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "source_snapshots": {
        "ace_manifest_sha256": hashlib.sha256(
            (output / "ace-source.manifest.json").read_bytes()
        ).hexdigest(),
        "phantom_manifest_sha256": hashlib.sha256(
            (output / "phantom-source.manifest.json").read_bytes()
        ).hexdigest(),
    },
    "qualification_invocation": {
        "sha256": hashlib.sha256(invocation_path.read_bytes()).hexdigest(),
        "normalized_argv_sha256": invocation["normalized_argv_sha256"],
    },
    "contents": "audited-source-snapshots-and-qualification-invocation-only",
    "files": sorted(path.name for path in output.iterdir() if path.is_file()),
}
(output / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "audited host-freeze source payload: ${OUTPUT}"
