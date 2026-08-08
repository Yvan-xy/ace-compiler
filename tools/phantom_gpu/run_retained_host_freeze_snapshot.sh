#!/usr/bin/env bash
set -euo pipefail
umask 022
export LC_ALL=C
export TZ=UTC

usage() {
  echo "usage: $0 --input-dir DIR --work-dir DIR --result-archive FILE --binding-mode <provisional|formal>" >&2
  exit 2
}

INPUT=""
WORK=""
RESULT_ARCHIVE=""
BINDING_MODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT="$2"; shift 2 ;;
    --work-dir) WORK="$2"; shift 2 ;;
    --result-archive) RESULT_ARCHIVE="$2"; shift 2 ;;
    --binding-mode) BINDING_MODE="$2"; shift 2 ;;
    *) usage ;;
  esac
done
if [[ -z "${INPUT}" || -z "${WORK}" || -z "${RESULT_ARCHIVE}" ]] ||
   [[ "${BINDING_MODE}" != provisional && "${BINDING_MODE}" != formal ]]; then
  usage
fi
INPUT="$(realpath -- "${INPUT}")"
WORK="$(realpath -m -- "${WORK}")"
RESULT_ARCHIVE="$(realpath -m -- "${RESULT_ARCHIVE}")"
if [[ -e "${WORK}" || -e "${RESULT_ARCHIVE}" ||
      -e "${RESULT_ARCHIVE}.sha256" ]]; then
  echo "retained host-freeze work and result archive paths must not already exist" >&2
  exit 1
fi

RESULTS_DIR="${WORK}/results"
RESULT_ROOT="${RESULTS_DIR}/qualification"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PIPELINE_EXIT=1
mkdir -p "${RESULTS_DIR}"

finalize() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ ${PIPELINE_EXIT} -eq 0 ]]; then
    incoming=0
  fi
  local status=failed
  [[ ${incoming} -eq 0 ]] && status=pass
  if [[ "${status}" == pass ]]; then
    printf '{"schema_version":"ace.phantom.result-completeness/1.0.0","status":"pass","mode":"retained-host-%s"}\n' \
      "${BINDING_MODE}" >"${RESULTS_DIR}/result-completeness.json"
  else
    printf '{"schema_version":"ace.phantom.retained-host-freeze-failure/1.0.0","status":"failed","exit_code":%d}\n' \
      "${incoming}" >"${RESULTS_DIR}/failure.json"
  fi
  printf '{"schema_version":"1.0.0","status":"%s","mode":"retained-host-%s","exit_code":%d,"started_utc":"%s","completed_utc":"%s"}\n' \
    "${status}" "${BINDING_MODE}" "${incoming}" "${STARTED_UTC}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RESULTS_DIR}/pipeline-result.json"
  (
    cd "${RESULTS_DIR}"
    find . -type f ! -path ./SHA256SUMS -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum >SHA256SUMS
  )
  mkdir -p "$(dirname -- "${RESULT_ARCHIVE}")"
  local temporary="${RESULT_ARCHIVE}.tmp"
  tar -C "${WORK}" -czf "${temporary}" results
  mv "${temporary}" "${RESULT_ARCHIVE}"
  (
    cd "$(dirname -- "${RESULT_ARCHIVE}")"
    sha256sum "$(basename -- "${RESULT_ARCHIVE}")" \
      >"$(basename -- "${RESULT_ARCHIVE}").sha256"
  )
  exit "${incoming}"
}
trap finalize EXIT
trap 'exit 130' INT TERM

if [[ "${NVIDIA_VISIBLE_DEVICES:-void}" != void ]] ||
   compgen -G '/dev/nvidia*' >/dev/null; then
  echo "retained host freeze refuses an NVIDIA device or visible-device request" >&2
  exit 1
fi
(
  cd "${INPUT}"
  sha256sum -c SHA256SUMS
)
python3 - "${INPUT}/payload.json" \
  "${INPUT}/ace-source.manifest.json" \
  "${INPUT}/phantom-source.manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate retained payload key: {key}")
        value[key] = item
    return value


def load(path):
    value = json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates
    )
    if not isinstance(value, dict):
        raise SystemExit("retained payload record is not an object")
    return value


payload_path, ace_path, phantom_path = map(Path, sys.argv[1:])
payload = load(payload_path)
if (
    payload.get("schema_version")
    != "ace.phantom.retained-host-freeze-payload/1.0.0"
    or payload.get("contents")
    != "audited-exact-source-snapshots-and-pinned-bootstrap-only"
):
    raise SystemExit("retained payload identity is invalid")
bindings = {}
for kind, path in (("ace", ace_path), ("phantom", phantom_path)):
    manifest = load(path)
    commit = manifest.get("commit")
    archive = manifest.get("archive")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("kind") != kind
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or archive != f"{kind}-source-{commit}.tar.gz"
        or PurePosixPath(archive).name != archive
        or payload.get(f"{kind}_commit") != commit
    ):
        raise SystemExit(f"retained payload has an invalid {kind} snapshot")
    bindings[f"{kind}_manifest_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
if payload.get("source_snapshots") != bindings:
    raise SystemExit("retained payload does not bind its source manifests")
PY

ACE_APT_LOCK="${INPUT}/apt-packages.lock" \
ACE_PYTHON_LOCK="${INPUT}/python-requirements-hashed.lock" \
ACE_BASE_FILES_LOCK="${INPUT}/base-files.sha256" \
ACE_DEPENDENCIES_LOCK="${INPUT}/dependencies.env" \
  bash "${INPUT}/bootstrap_environment.sh" "${WORK}/environment"
export PATH="/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

ACE_ARCHIVE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/ace-source.manifest.json")"
PHANTOM_ARCHIVE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/phantom-source.manifest.json")"
python3 "${INPUT}/source_archive.py" audit \
  --kind ace \
  --archive "${INPUT}/${ACE_ARCHIVE}" \
  --manifest "${INPUT}/ace-source.manifest.json" \
  --extract "${WORK}/ace-extract" \
  >"${WORK}/ace-source-audit.json"
python3 "${INPUT}/source_archive.py" audit \
  --kind phantom \
  --archive "${INPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${INPUT}/phantom-source.manifest.json" \
  --extract "${WORK}/phantom-extract" \
  >"${WORK}/phantom-source-audit.json"

ACE_SOURCE="${WORK}/ace-extract/ace-source"
PHANTOM_SOURCE="${WORK}/phantom-extract/phantom-source"
ACE_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${INPUT}/ace-source.manifest.json")"
PHANTOM_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "${INPUT}/phantom-source.manifest.json")"
ACE_MANIFEST_SHA256="$(sha256sum "${INPUT}/ace-source.manifest.json" | awk '{print $1}')"
PHANTOM_MANIFEST_SHA256="$(sha256sum "${INPUT}/phantom-source.manifest.json" | awk '{print $1}')"
BOOTSTRAP_SHA256="$(sha256sum "${INPUT}/bootstrap_environment.sh" | awk '{print $1}')"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH="${ACE_SOURCE}"
export CMAKE_CUDA_ARCHITECTURES=80
export CUDAARCHS=80
export ACE_PHANTOM_TOOLCHAIN=12.4.1-sm80
export ACE_PHANTOM_SOURCE_MODE=snapshot
export ACE_PHANTOM_REPO_ROOT="${ACE_SOURCE}"
export ACE_PHANTOM_SOURCE_DIR="${PHANTOM_SOURCE}"
export ACE_PHANTOM_STATE_ROOT="${WORK}/state"
export ACE_PHANTOM_ACE_COMMIT="${ACE_COMMIT}"
export ACE_PHANTOM_SOURCE_MANIFEST="${INPUT}/ace-source.manifest.json"
export ACE_PHANTOM_SOURCE_MANIFEST_SHA256="${ACE_MANIFEST_SHA256}"
export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST="${INPUT}/phantom-source.manifest.json"
export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256="${PHANTOM_MANIFEST_SHA256}"
export ACE_RUNPOD_BOOTSTRAP_SHA256="${BOOTSTRAP_SHA256}"
export ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
export ACE_RETAINED_CKKS_RESULT_ROOT="${RESULT_ROOT}"
export ACE_RETAINED_CKKS_WORK_ROOT="${WORK}/retained-build"
if [[ "${BINDING_MODE}" == provisional ]]; then
  export ACE_RETAINED_CKKS_PROVISIONAL_BINDING=1
else
  export ACE_RETAINED_CKKS_PROVISIONAL_BINDING=0
fi

bash "${ACE_SOURCE}/tools/phantom_gpu/run_retained_ckks_correctness.sh" \
  --host-qualify
(
  cd "${RESULT_ROOT}"
  sha256sum -c SHA256SUMS
)
if [[ "${BINDING_MODE}" == formal ]]; then
  python3 "${ACE_SOURCE}/tools/phantom_gpu/retained_runpod_evidence.py" \
    validate-frozen \
    --root "${RESULT_ROOT}" \
    --ace-commit "${ACE_COMMIT}" \
    --phantom-commit "${PHANTOM_COMMIT}"
  EXPECTED_LIFECYCLE=checked-bound
else
  EXPECTED_LIFECYCLE=provisional-bind
fi
python3 - "${RESULT_ROOT}" "${EXPECTED_LIFECYCLE}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
artifact = json.loads(
    (root / "artifact_manifest.json").read_text(encoding="utf-8")
)
fixture = json.loads(
    (root / "inputs/retained_ckks_fixture.json").read_text(encoding="utf-8")
)
if (
    manifest.get("status") != "pass"
    or manifest.get("fixture_lifecycle") != expected
    or manifest.get("gpu_executables_were_run") is not False
    or artifact.get("fixture_lifecycle") != expected
    or fixture.get("qualification_bindings", {}).get("status") != "bound"
):
    raise SystemExit("retained host-freeze lifecycle attestation is inconsistent")
PY
if compgen -G '/dev/nvidia*' >/dev/null; then
  echo "an NVIDIA device appeared during retained host freeze" >&2
  exit 1
fi
echo "retained host freeze passed in ${BINDING_MODE} mode: ${RESULT_ROOT}"
PIPELINE_EXIT=0
