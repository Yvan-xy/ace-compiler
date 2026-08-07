#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PROTECTED_NAME="ace-compiler-dev"

usage() {
  echo "usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --ordinary-run-root DIR --poly-degree N --mul-level Q --input-level L --security-level B --scaling-factor-bits B --first-prime-bits B --hamming-weight W OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
POLY_DEGREE=""
MUL_LEVEL=""
INPUT_LEVEL=""
SECURITY_LEVEL=""
SCALING_BITS=""
FIRST_PRIME_BITS=""
HAMMING_WEIGHT=""
ORDINARY_RUN_ROOT=""
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
    --poly-degree) [[ $# -ge 2 ]] || usage; POLY_DEGREE="$2"; shift 2 ;;
    --mul-level) [[ $# -ge 2 ]] || usage; MUL_LEVEL="$2"; shift 2 ;;
    --input-level) [[ $# -ge 2 ]] || usage; INPUT_LEVEL="$2"; shift 2 ;;
    --security-level) [[ $# -ge 2 ]] || usage; SECURITY_LEVEL="$2"; shift 2 ;;
    --scaling-factor-bits) [[ $# -ge 2 ]] || usage; SCALING_BITS="$2"; shift 2 ;;
    --first-prime-bits) [[ $# -ge 2 ]] || usage; FIRST_PRIME_BITS="$2"; shift 2 ;;
    --hamming-weight) [[ $# -ge 2 ]] || usage; HAMMING_WEIGHT="$2"; shift 2 ;;
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
   -n "${ORDINARY_RUN_ROOT}" &&
   -n "${OUTPUT_ARGUMENT}" ]] || usage
for value in "${POLY_DEGREE}" "${MUL_LEVEL}" "${INPUT_LEVEL}" \
  "${SECURITY_LEVEL}" "${SCALING_BITS}" "${FIRST_PRIME_BITS}" \
  "${HAMMING_WEIGHT}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || usage
done
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

bash "${SCRIPT_DIR}/package_runpod_sources.sh" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}" \
  --ordinary-run-root "${ORDINARY_RUN_ROOT}" \
  --poly-degree "${POLY_DEGREE}" \
  --mul-level "${MUL_LEVEL}" \
  --input-level "${INPUT_LEVEL}" \
  --security-level "${SECURITY_LEVEL}" \
  --scaling-factor-bits "${SCALING_BITS}" \
  --first-prime-bits "${FIRST_PRIME_BITS}" \
  --hamming-weight "${HAMMING_WEIGHT}" \
  "${PAYLOAD}"
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
