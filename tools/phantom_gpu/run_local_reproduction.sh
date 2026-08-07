#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BASE_IMAGE="docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
BASE_CONFIG="sha256:0131784115794405cb36a8068a82d7aea0937196d1d6e844b9dd021252ccf7e4"
PROTECTED_NAME="ace-compiler-dev"

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
chmod 0755 "${OUTPUT}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CONTAINER_NAME="ace-phantom-runpod-local-${RUN_ID}"
IMAGE_NAME="ace-phantom-runpod-local-base:${RUN_ID}"
PAYLOAD="${OUTPUT}/payload"
DOCKER_EVIDENCE="${OUTPUT}/docker"
mkdir -p "${DOCKER_EVIDENCE}"

docker inspect --type container "${PROTECTED_NAME}" >"${DOCKER_EVIDENCE}/protected-before.json"
PROTECTED_CONTAINER_ID="$(docker inspect --type container -f '{{.Id}}' "${PROTECTED_NAME}")"
PROTECTED_IMAGE_ID="$(docker inspect --type container -f '{{.Image}}' "${PROTECTED_NAME}")"
PROTECTED_STATE="$(docker inspect --type container -f '{{.State.Status}}' "${PROTECTED_NAME}")"
if [[ "${PROTECTED_STATE}" != running ]]; then
  echo "protected container is not running; refusing local reproduction" >&2
  exit 1
fi
docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-before.txt"
docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-before.txt"
docker system df >"${DOCKER_EVIDENCE}/disk-before.txt"

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  local candidate_id=""
  candidate_id="$(docker inspect --type container -f '{{.Id}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
  if [[ -n "${candidate_id}" ]]; then
    if [[ "${candidate_id}" == "${PROTECTED_CONTAINER_ID}" ]]; then
      echo "cleanup guard rejected the protected container" >&2
      exit 1
    fi
    docker rm -f "${candidate_id}" >>"${DOCKER_EVIDENCE}/cleanup.txt"
  fi
  local tagged_id=""
  tagged_id="$(docker image inspect -f '{{.Id}}' "${IMAGE_NAME}" 2>/dev/null || true)"
  if [[ -n "${tagged_id}" ]]; then
    if [[ "${tagged_id}" == "${PROTECTED_IMAGE_ID}" ]]; then
      echo "cleanup guard rejected the protected image" >&2
      exit 1
    fi
    docker image rm "${IMAGE_NAME}" >>"${DOCKER_EVIDENCE}/cleanup.txt"
  fi
  docker inspect --type container "${PROTECTED_NAME}" >"${DOCKER_EVIDENCE}/protected-after.json"
  if [[ "$(docker inspect --type container -f '{{.Id}}' "${PROTECTED_NAME}")" != \
        "${PROTECTED_CONTAINER_ID}" ]] ||
     [[ "$(docker inspect --type container -f '{{.Image}}' "${PROTECTED_NAME}")" != \
        "${PROTECTED_IMAGE_ID}" ]] ||
     [[ "$(docker inspect --type container -f '{{.State.Status}}' "${PROTECTED_NAME}")" != \
        "${PROTECTED_STATE}" ]]; then
    echo "protected Docker identity changed" >&2
    exit 1
  fi
  docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-after.txt"
  docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-after.txt"
  docker system df >"${DOCKER_EVIDENCE}/disk-after.txt"
  exit "${incoming}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

bash "${SCRIPT_DIR}/package_runpod_sources.sh" "${PAYLOAD}"
docker pull --platform linux/amd64 "${BASE_IMAGE}" | tee "${DOCKER_EVIDENCE}/pull.txt"
ACTUAL_BASE_ID="$(docker image inspect -f '{{.Id}}' "${BASE_IMAGE}")"
if [[ "${ACTUAL_BASE_ID}" != "${BASE_CONFIG}" ]]; then
  echo "pulled base config digest does not match the lock" >&2
  exit 1
fi
if [[ "${ACTUAL_BASE_ID}" == "${PROTECTED_IMAGE_ID}" ]]; then
  echo "pinned base unexpectedly aliases the protected image" >&2
  exit 1
fi
docker image tag "${BASE_IMAGE}" "${IMAGE_NAME}"

set +e
docker run --name "${CONTAINER_NAME}" --platform linux/amd64 \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}" \
  --volume "${PAYLOAD}:/workspace/input:ro" \
  --volume "${OUTPUT}:/workspace/output:rw" \
  "${IMAGE_NAME}" \
  bash /workspace/input/run_build_and_health.sh \
    --mode local \
    --input-dir /workspace/input \
    --work-dir /workspace/output/work \
    --result-archive /workspace/output/local-result.tar.gz \
  2>&1 | tee "${OUTPUT}/local-container.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_NAME}" >"${DOCKER_EVIDENCE}/disposable-container.json"
docker image inspect "${IMAGE_NAME}" >"${DOCKER_EVIDENCE}/disposable-image.json"

if [[ ! -s "${OUTPUT}/local-result.tar.gz" ||
      ! -s "${OUTPUT}/local-result.tar.gz.sha256" ]]; then
  echo "local pipeline did not produce a result archive" >&2
  exit 1
fi
(
  cd "${OUTPUT}"
  sha256sum -c local-result.tar.gz.sha256
)
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  echo "local reproduction failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi
echo "local reproduction passed: ${OUTPUT}"
