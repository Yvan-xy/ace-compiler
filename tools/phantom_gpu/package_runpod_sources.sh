#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

OUTPUT="$(realpath -m -- "$1")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"

set -a
source "${SCRIPT_DIR}/configs/dependencies.env"
set +a

ACE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
ACE_BRANCH_ACTUAL="$(git -C "${REPO_ROOT}" branch --show-current)"
if [[ "${ACE_BRANCH_ACTUAL}" != "${ACE_BRANCH}" ]]; then
  echo "ACE branch is ${ACE_BRANCH_ACTUAL}, expected ${ACE_BRANCH}" >&2
  exit 1
fi
if [[ "$(git -C "${PHANTOM_REPO}" rev-parse HEAD)" != "${PHANTOM_COMMIT}" ]]; then
  echo "Phantom checkout is not at the pinned commit" >&2
  exit 1
fi

ACE_ARCHIVE="ace-source-${ACE_COMMIT}.tar.gz"
PHANTOM_ARCHIVE="phantom-source-${PHANTOM_COMMIT}.tar.gz"
python3 "${SCRIPT_DIR}/source_archive.py" create \
  --repo "${REPO_ROOT}" \
  --commit "${ACE_COMMIT}" \
  --kind ace \
  --output "${OUTPUT}/${ACE_ARCHIVE}" \
  --manifest "${OUTPUT}/ace-source.manifest.json" \
  >"${OUTPUT}/ace-source.audit.json"
python3 "${SCRIPT_DIR}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" \
  --commit "${PHANTOM_COMMIT}" \
  --kind phantom \
  --output "${OUTPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${OUTPUT}/phantom-source.manifest.json" \
  >"${OUTPUT}/phantom-source.audit.json"

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" >"${OUTPUT}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
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
  "${OUTPUT}/bootstrap_environment.sh" \
  "${OUTPUT}/run_build_and_health.sh"

(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

python3 - "${OUTPUT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "schema_version": "1.0.0",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "files": sorted(path.name for path in output.iterdir() if path.is_file()),
}
(output / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
(
  cd "${OUTPUT}"
  sha256sum payload.json >>SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "audited RunPod source payload: ${OUTPUT}"
