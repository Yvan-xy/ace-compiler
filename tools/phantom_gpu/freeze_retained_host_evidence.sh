#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

usage() {
  cat >&2 <<EOF
usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --binding-mode <provisional|formal> OUTPUT_DIRECTORY
Provisional mode publishes only a review candidate under OUTPUT_DIRECTORY/candidate.
Formal mode requires a checked-bound fixture and validates the complete retained result.
EOF
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
BINDING_MODE=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ace-commit) ACE_COMMIT="$2"; shift 2 ;;
    --phantom-commit) PHANTOM_COMMIT="$2"; shift 2 ;;
    --binding-mode) BINDING_MODE="$2"; shift 2 ;;
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
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
   [[ "${BINDING_MODE}" != provisional && "${BINDING_MODE}" != formal ]] ||
   [[ -z "${OUTPUT_ARGUMENT}" ]]; then
  usage
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"

for wrapper in \
  tools/phantom_gpu/freeze_retained_host_evidence.sh \
  tools/phantom_gpu/package_retained_host_freeze_sources.sh; do
  expected="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${wrapper}" | sha256sum | awk '{print $1}')"
  observed="$(sha256sum "${REPO_ROOT}/${wrapper}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "invoked lifecycle wrapper differs from the selected ACE commit: ${wrapper}" >&2
    exit 1
  fi
done

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"
PAYLOAD="${OUTPUT}/payload"
DOCKER_EVIDENCE="${OUTPUT}/docker"
RESULT_ROOT="${OUTPUT}/retained-host-result"
WORK_ROOT="${OUTPUT}/work"
mkdir -p "${DOCKER_EVIDENCE}"
bash "${SCRIPT_DIR}/package_retained_host_freeze_sources.sh" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}" \
  "${PAYLOAD}"

lock_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" "${PAYLOAD}/dependencies.env")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "retained host-freeze payload has an invalid ${key} lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
BASE_IMAGE="$(lock_value CUDA_IMAGE)"
BASE_CONFIG="$(lock_value CUDA_IMAGE_CONFIG)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
CONTAINER_NAME="ace-retained-fixture-freeze-${RUN_ID}"
TASK_LABEL="retained-fixture-host-freeze-${RUN_ID}"
CONTAINER_ID=""

remove_exact_container() {
  local exact_id="${CONTAINER_ID}"
  [[ -n "${exact_id}" ]] || return 0
  if ! docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "task-created container disappeared before exact cleanup: ${exact_id}" >&2
    return 1
  fi
  local observed_label observed_name
  observed_label="$(docker inspect --type container \
    -f '{{index .Config.Labels "ace.phantom.task"}}' "${exact_id}")"
  observed_name="$(docker inspect --type container -f '{{.Name}}' "${exact_id}")"
  if [[ "${observed_label}" != "${TASK_LABEL}" ||
        "${observed_name}" != "/${CONTAINER_NAME}" ]]; then
    echo "exact cleanup refused a container with an unexpected identity" >&2
    return 1
  fi
  docker rm -f "${exact_id}" >>"${DOCKER_EVIDENCE}/cleanup.txt"
  if docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "exact cleanup absence verification failed for ${exact_id}" >&2
    return 1
  fi
  printf '%s\tremoved-and-absent\n' "${exact_id}" \
    >>"${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  CONTAINER_ID=""
}

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ -n "${CONTAINER_ID}" ]]; then
    set +e
    remove_exact_container
    local cleanup_exit="$?"
    set -e
    if [[ ${cleanup_exit} -ne 0 ]]; then
      incoming=1
    fi
  fi
  exit "${incoming}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

docker pull --platform linux/amd64 "${BASE_IMAGE}" |
  tee "${DOCKER_EVIDENCE}/pull.txt"
ACTUAL_BASE_ID="$(docker image inspect -f '{{.Id}}' "${BASE_IMAGE}")"
if [[ "${ACTUAL_BASE_ID}" != "${BASE_CONFIG}" ]]; then
  echo "pulled base config digest does not match the exact commit lock" >&2
  exit 1
fi
docker image inspect "${BASE_IMAGE}" >"${DOCKER_EVIDENCE}/base-image.json"

CREATED_CONTAINER_ID="$(docker create \
  --name "${CONTAINER_NAME}" \
  --label "ace.phantom.task=${TASK_LABEL}" \
  --platform linux/amd64 \
  --runtime runc \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}" \
  --volume "${PAYLOAD}:/retained-freeze/input:ro" \
  --volume "${OUTPUT}:/retained-freeze/output:rw" \
  "${BASE_IMAGE}" \
  bash /retained-freeze/input/run_retained_host_freeze_snapshot.sh \
    --input-dir /retained-freeze/input \
    --work-dir /retained-freeze/output/work \
    --result-root /retained-freeze/output/retained-host-result \
    --binding-mode "${BINDING_MODE}")"
if [[ ! "${CREATED_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "docker create did not return one exact full container ID" >&2
  exit 1
fi
CONTAINER_ID="${CREATED_CONTAINER_ID}"
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-created.json"
python3 - "${DOCKER_EVIDENCE}/container-created.json" \
  "${CONTAINER_ID}" "${CONTAINER_NAME}" "${TASK_LABEL}" \
  "${BASE_CONFIG}" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(record, list) or len(record) != 1:
    raise SystemExit("Docker did not return one created-container record")
item = record[0]
host = item.get("HostConfig", {})
config = item.get("Config", {})
labels = config.get("Labels") or {}
environment = set(config.get("Env") or [])
if (
    item.get("Id") != sys.argv[2]
    or item.get("Name") != "/" + sys.argv[3]
    or labels.get("ace.phantom.task") != sys.argv[4]
    or item.get("Image") != sys.argv[5]
    or host.get("Privileged") is not False
    or host.get("Runtime") != "runc"
    or host.get("Devices") not in (None, [])
    or host.get("DeviceRequests") not in (None, [])
    or "NVIDIA_VISIBLE_DEVICES=void" not in environment
    or "NVIDIA_DRIVER_CAPABILITIES=none" not in environment
):
    raise SystemExit("created container violates the host-freeze device boundary")
PY

set +e
docker start -a "${CONTAINER_ID}" 2>&1 |
  tee "${OUTPUT}/retained-host-freeze.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-completed.json"
remove_exact_container
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  echo "retained host freeze failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi

(
  cd "${RESULT_ROOT}"
  sha256sum -c SHA256SUMS
)
if [[ "${BINDING_MODE}" == formal ]]; then
  python3 "${PAYLOAD}/retained_runpod_evidence.py" validate-frozen \
    --root "${RESULT_ROOT}" \
    --ace-commit "${ACE_COMMIT}" \
    --phantom-commit "${PHANTOM_COMMIT}"
else
  CANDIDATE="${OUTPUT}/candidate"
  mkdir "${CANDIDATE}"
  cp -- "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    "${CANDIDATE}/retained_ckks_v1.json"
  python3 - "${CANDIDATE}" "${RESULT_ROOT}" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

candidate, root = map(Path, sys.argv[1:3])
fixture = candidate / "retained_ckks_v1.json"
record = {
    "schema_version": "ace.phantom.retained-fixture-candidate/1.0.0",
    "status": "candidate-for-review-not-frozen-evidence",
    "ace_commit": sys.argv[3],
    "phantom_commit": sys.argv[4],
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "provisional_manifest_sha256": hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest(),
}
(candidate / "candidate-binding.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  (
    cd "${CANDIDATE}"
    sha256sum candidate-binding.json retained_ckks_v1.json >SHA256SUMS
    sha256sum -c SHA256SUMS
  )
fi

python3 - "${OUTPUT}/lifecycle.json" "${BINDING_MODE}" \
  "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_IMAGE}" \
  "${BASE_CONFIG}" "${CREATED_CONTAINER_ID}" "${TASK_LABEL}" <<'PY'
import json
from pathlib import Path
import sys

value = {
    "schema_version": "ace.phantom.retained-host-freeze-lifecycle/1.0.0",
    "status": "pass",
    "binding_mode": sys.argv[2],
    "ace_commit": sys.argv[3],
    "phantom_commit": sys.argv[4],
    "base_image": sys.argv[5],
    "base_config_digest": sys.argv[6],
    "container_id": sys.argv[7],
    "task_label": sys.argv[8],
    "container_cleanup": "removed-and-absent",
    "gpu_device_requests": [],
    "gpu_executables_were_run": False,
}
Path(sys.argv[1]).write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
echo "retained host freeze passed: ${OUTPUT}"
