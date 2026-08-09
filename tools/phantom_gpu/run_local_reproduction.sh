#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
source "${SCRIPT_DIR}/transport_helpers.sh"

usage() {
  echo "usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --ordinary-run-root DIR --retained-run-root DIR --bootstrap-run-root DIR OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
ORDINARY_RUN_ROOT=""
RETAINED_RUN_ROOT=""
BOOTSTRAP_RUN_ROOT=""
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
    --ordinary-run-root)
      [[ $# -ge 2 ]] || usage
      ORDINARY_RUN_ROOT="$2"
      shift 2
      ;;
    --retained-run-root)
      [[ $# -ge 2 ]] || usage
      RETAINED_RUN_ROOT="$2"
      shift 2
      ;;
    --bootstrap-run-root)
      [[ $# -ge 2 ]] || usage
      BOOTSTRAP_RUN_ROOT="$2"
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
   -n "${ORDINARY_RUN_ROOT}" && -n "${RETAINED_RUN_ROOT}" &&
   -n "${BOOTSTRAP_RUN_ROOT}" &&
   -n "${OUTPUT_ARGUMENT}" ]] || usage
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commits must be full lowercase 40-character object IDs" >&2
  exit 1
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"

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
BASE_IMAGE="$(lock_value CUDA_IMAGE)"
BASE_CONFIG="$(lock_value CUDA_IMAGE_CONFIG)"
LOCKED_PHANTOM_COMMIT="$(lock_value PHANTOM_COMMIT)"
if [[ "${PHANTOM_COMMIT}" != "${LOCKED_PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit does not match the selected ACE commit lock" >&2
  exit 1
fi

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0755 "${OUTPUT}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CONTAINER_NAME="ace-phantom-retained-local-${RUN_ID}"
PAYLOAD="${OUTPUT}/payload"
DOCKER_EVIDENCE="${OUTPUT}/docker"
mkdir -p "${DOCKER_EVIDENCE}"

CONTAINER_ID=""
docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-before.txt"
docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-before.txt"
docker system df >"${DOCKER_EVIDENCE}/disk-before.txt"

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  local cleanup_exit=0
  if [[ -n "${CONTAINER_ID}" ]]; then
    if ! docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1; then
      echo "task-created container disappeared before exact cleanup: ${CONTAINER_ID}" >&2
      cleanup_exit=1
    elif [[ "$(docker inspect -f '{{index .Config.Labels "ace.phantom.task"}}' "${CONTAINER_ID}")" != \
          "retained-ckks-local-reproduction" ]]; then
      echo "cleanup refused an unlabeled container" >&2
      cleanup_exit=1
    else
      set +e
      docker rm -f "${CONTAINER_ID}" >>"${DOCKER_EVIDENCE}/cleanup.txt"
      cleanup_exit=$?
      set -e
      if docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1; then
        echo "exact cleanup verification failed for ${CONTAINER_ID}" >&2
        cleanup_exit=1
      elif [[ ${cleanup_exit} -eq 0 ]]; then
        printf '%s\tremoved-and-absent\n' "${CONTAINER_ID}" \
          >>"${DOCKER_EVIDENCE}/cleanup-verification.tsv"
      fi
    fi
  fi
  docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-after.txt"
  docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-after.txt"
  docker system df >"${DOCKER_EVIDENCE}/disk-after.txt"
  if [[ ${cleanup_exit} -ne 0 ]]; then
    incoming=1
  fi
  exit "${incoming}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

bash "${SCRIPT_DIR}/package_runpod_sources.sh" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}" \
  --ordinary-run-root "${ORDINARY_RUN_ROOT}" \
  --retained-run-root "${RETAINED_RUN_ROOT}" \
  --bootstrap-run-root "${BOOTSTRAP_RUN_ROOT}" \
  "${PAYLOAD}"
docker pull --platform linux/amd64 "${BASE_IMAGE}" | tee "${DOCKER_EVIDENCE}/pull.txt"
ACTUAL_BASE_ID="$(docker image inspect -f '{{.Id}}' "${BASE_IMAGE}")"
if [[ "${ACTUAL_BASE_ID}" != "${BASE_CONFIG}" ]]; then
  echo "pulled base config digest does not match the lock" >&2
  exit 1
fi
CONTAINER_ID="$(docker create --name "${CONTAINER_NAME}" \
  --label ace.phantom.task=retained-ckks-local-reproduction \
  --platform linux/amd64 \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}" \
  --volume "${PAYLOAD}:/retained-qualification/input:ro" \
  --volume "${OUTPUT}:/retained-qualification/output:rw" \
  "${BASE_IMAGE}" \
  bash /retained-qualification/input/run_build_and_health.sh \
    --mode local \
    --input-dir /retained-qualification/input \
    --work-dir /retained-qualification/work \
    --result-archive /retained-qualification/output/local-result.tar.gz)"
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/disposable-container-created.json"
set +e
docker start -a "${CONTAINER_ID}" 2>&1 | tee "${OUTPUT}/local-container.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_ID}" >"${DOCKER_EVIDENCE}/disposable-container.json"
docker image inspect "${BASE_IMAGE}" >"${DOCKER_EVIDENCE}/base-image.json"

if [[ ! -s "${OUTPUT}/local-result.tar.gz" ||
      ! -s "${OUTPUT}/local-result.tar.gz.sha256" ]]; then
  echo "local pipeline did not produce a result archive" >&2
  exit 1
fi
(
  cd "${OUTPUT}"
  sha256sum -c local-result.tar.gz.sha256
)
verify_result_archive \
  "${OUTPUT}/local-result.tar.gz" local "${PIPELINE_EXIT}" \
  "${OUTPUT}/verified-local-result" \
  >"${OUTPUT}/local-result-verification.json"
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  echo "local reproduction failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi
echo "local reproduction passed: ${OUTPUT}"
