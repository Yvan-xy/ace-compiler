#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${SCRIPT_DIR}/transport_helpers.sh"

usage() {
  cat >&2 <<EOF
usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --qualification-invocation FILE OUTPUT_DIRECTORY
The reusable candidate root is
OUTPUT_DIRECTORY/verified-host-freeze-result/results/qualification.
Exact container cleanup evidence is written under OUTPUT_DIRECTORY/docker.
EOF
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
QUALIFICATION_INVOCATION=""
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
      QUALIFICATION_INVOCATION="$2"
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
   -n "${QUALIFICATION_INVOCATION}" && -n "${OUTPUT_ARGUMENT}" ]] || usage
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commits must be full lowercase 40-character object IDs" >&2
  exit 1
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"

PAYLOAD="${OUTPUT}/payload"
DOCKER_EVIDENCE="${OUTPUT}/docker"
mkdir -p "${DOCKER_EVIDENCE}"
bash "${SCRIPT_DIR}/package_host_freeze_sources.sh" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}" \
  --qualification-invocation "${QUALIFICATION_INVOCATION}" \
  "${PAYLOAD}"

lock_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" "${PAYLOAD}/dependencies.env")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "host-freeze payload has an invalid ${key} dependency lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
BASE_IMAGE="$(lock_value CUDA_IMAGE)"
BASE_CONFIG="$(lock_value CUDA_IMAGE_CONFIG)"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
CONTAINER_NAME="ace-phantom-host-freeze-${RUN_ID}"
TASK_LABEL="ordinary-host-freeze-${RUN_ID}"
CONTAINER_ID=""

remove_exact_container() {
  local exact_id="${CONTAINER_ID}"
  [[ -n "${exact_id}" ]] || return 0
  local observed_label observed_name
  if ! docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "task-created container disappeared before exact cleanup: ${exact_id}" >&2
    return 1
  fi
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
    echo "exact cleanup verification failed for ${exact_id}" >&2
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
  echo "pulled base config digest does not match the selected commit lock" >&2
  exit 1
fi
docker image inspect "${BASE_IMAGE}" >"${DOCKER_EVIDENCE}/base-image.json"

CREATED_CONTAINER_ID="$(docker create --name "${CONTAINER_NAME}" \
  --label "ace.phantom.task=${TASK_LABEL}" \
  --platform linux/amd64 \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}" \
  --volume "${PAYLOAD}:/workspace/input:ro" \
  --volume "${OUTPUT}:/workspace/output:rw" \
  "${BASE_IMAGE}" \
  bash /workspace/input/run_build_and_health.sh \
    --mode freeze-host \
    --input-dir /workspace/input \
    --work-dir /workspace/output/work \
    --result-archive /workspace/output/host-freeze-result.tar.gz)"
if [[ ! "${CREATED_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "docker create did not return one exact full container ID" >&2
  exit 1
fi
CONTAINER_ID="${CREATED_CONTAINER_ID}"
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-created.json"

set +e
docker start -a "${CONTAINER_ID}" 2>&1 |
  tee "${OUTPUT}/host-freeze-container.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-completed.json"
remove_exact_container

if [[ ! -s "${OUTPUT}/host-freeze-result.tar.gz" ||
      ! -s "${OUTPUT}/host-freeze-result.tar.gz.sha256" ]]; then
  echo "host-freeze pipeline did not produce a result archive" >&2
  exit 1
fi
(
  cd "${OUTPUT}"
  sha256sum -c host-freeze-result.tar.gz.sha256
)
verify_result_archive \
  "${OUTPUT}/host-freeze-result.tar.gz" freeze-host "${PIPELINE_EXIT}" \
  "${OUTPUT}/verified-host-freeze-result" \
  >"${OUTPUT}/host-freeze-result-verification.json"
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  echo "host-freeze qualification failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi
# Only the checksum-verified archive extraction is reusable evidence. The live
# container work tree is deliberately outside the host-side trust boundary.
VERIFIED_RESULTS="${OUTPUT}/verified-host-freeze-result/results"
CANDIDATE="${VERIFIED_RESULTS}/qualification"
python3 - \
  "${CANDIDATE}/manifest.json" \
  "${CANDIDATE}/SHA256SUMS" \
  "${VERIFIED_RESULTS}/host-freeze-candidate.json" \
  "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"host-freeze evidence contains duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(file_name):
    return json.loads(
        Path(file_name).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


manifest_file = Path(sys.argv[1])
sums_file = Path(sys.argv[2])
candidate_file = Path(sys.argv[3])
ace_commit = sys.argv[4]
phantom_commit = sys.argv[5]
manifest = load_json(manifest_file)
expected_manifest = {
    "status": "pass",
    "gate": "ordinary",
    "source_mode": "snapshot",
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "ace_worktree_dirty": False,
    "gpu_executables_were_run": False,
}
if not isinstance(manifest, dict) or any(
    manifest.get(key) != value for key, value in expected_manifest.items()
):
    raise SystemExit("verified host-freeze qualification manifest is inconsistent")

candidate = load_json(candidate_file)
expected_candidate = {
    "status": "candidate",
    "evidence_path": "qualification",
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "qualification_manifest_sha256": hashlib.sha256(
        manifest_file.read_bytes()
    ).hexdigest(),
    "qualification_sha256_manifest_sha256": hashlib.sha256(
        sums_file.read_bytes()
    ).hexdigest(),
}
if not isinstance(candidate, dict) or any(
    candidate.get(key) != value for key, value in expected_candidate.items()
):
    raise SystemExit("verified host-freeze candidate binding is inconsistent")
PY
(
  cd "${CANDIDATE}"
  sha256sum -c SHA256SUMS
)
echo "candidate ordinary host evidence: ${CANDIDATE}"
echo "checksummed host-freeze result: ${OUTPUT}/host-freeze-result.tar.gz"
echo "exact container cleanup evidence: ${DOCKER_EVIDENCE}/cleanup-verification.tsv"
