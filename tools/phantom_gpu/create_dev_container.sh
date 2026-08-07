#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
LOCK_FILE="${SCRIPT_DIR}/configs/dependencies.env"
CONTAINER_NAME="ace-compiler-dev"
MODELS_DIR=""
DATASET_DIR=""
PHANTOM_DIR=""
BUILD_IMAGE=1
PRESERVE_EXISTING=0
PRESERVE_RUNNING=0
EXPECTED_IMAGE_ID=""

usage() {
  echo "usage: $0 --models-dir DIR --dataset-dir DIR --phantom-dir DIR [options]"
  echo "  --skip-image-build       use the already-built pinned development image"
  echo "  --expected-image-id ID   require this immutable image ID when skipping a build"
  echo "  --preserve-existing      rename an existing stopped container before creation"
  echo "  --preserve-running       allow preserving an existing running container by rename"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --phantom-dir)
      PHANTOM_DIR="$2"
      shift 2
      ;;
    --skip-image-build)
      BUILD_IMAGE=0
      shift
      ;;
    --expected-image-id)
      EXPECTED_IMAGE_ID="$2"
      shift 2
      ;;
    --preserve-existing)
      PRESERVE_EXISTING=1
      shift
      ;;
    --preserve-running)
      PRESERVE_EXISTING=1
      PRESERVE_RUNNING=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODELS_DIR}" || -z "${DATASET_DIR}" || -z "${PHANTOM_DIR}" ]]; then
  usage >&2
  exit 2
fi
if [[ ${BUILD_IMAGE} -eq 0 && -z "${EXPECTED_IMAGE_ID}" ]]; then
  echo "--skip-image-build requires --expected-image-id" >&2
  exit 2
fi

MODELS_DIR="$(cd -- "${MODELS_DIR}" && pwd -P)"
DATASET_DIR="$(cd -- "${DATASET_DIR}" && pwd -P)"
PHANTOM_DIR="$(cd -- "${PHANTOM_DIR}" && pwd -P)"

if [[ ! -f "${MODELS_DIR}/resnet20_cifar10_pre.onnx" ]]; then
  echo "model directory does not contain resnet20_cifar10_pre.onnx" >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/test_batch.bin" ]]; then
  echo "dataset directory does not contain test_batch.bin" >&2
  exit 1
fi
if [[ ! -d "${PHANTOM_DIR}/.git" ]]; then
  echo "Phantom directory is not a Git working tree" >&2
  exit 1
fi

set -a
source "${LOCK_FILE}"
set +a

if [[ "$(git -C "${REPO_DIR}" branch --show-current)" != "${ACE_BRANCH}" ]]; then
  echo "ACE must be on the pinned development branch ${ACE_BRANCH}" >&2
  exit 1
fi
if [[ "$(git -C "${PHANTOM_DIR}" branch --show-current)" != "${PHANTOM_BRANCH}" ]]; then
  echo "Phantom must be on the pinned branch ${PHANTOM_BRANCH}" >&2
  exit 1
fi
if [[ "$(git -C "${PHANTOM_DIR}" rev-parse HEAD)" != "${PHANTOM_COMMIT}" ]]; then
  echo "Phantom HEAD does not match the pinned commit ${PHANTOM_COMMIT}" >&2
  exit 1
fi

DEFINITION_FILES=(
  "${REPO_DIR}/docker/phantom-a100/Dockerfile"
  "${REPO_DIR}/docker/phantom-a100/.dockerignore"
  "${REPO_DIR}/docker/phantom-a100/python-requirements.lock"
  "${SCRIPT_DIR}/configs/dependencies.env"
  "${SCRIPT_DIR}/configs/toolchain.env"
)
DEFINITION_SHA256="$(
  for FILE_PATH in "${DEFINITION_FILES[@]}"; do
    RELATIVE_PATH="${FILE_PATH#"${REPO_DIR}/"}"
    printf '%s  %s\n' "$(sha256sum "${FILE_PATH}" | awk '{print $1}')" \
      "${RELATIVE_PATH}"
  done | sha256sum | awk '{print $1}'
)"

if [[ ${BUILD_IMAGE} -eq 1 ]]; then
  BUILD_ARGUMENTS=(
    --platform linux/amd64
    --pull
    --build-arg "ACE_PHANTOM_DEFINITION_SHA256=${DEFINITION_SHA256}"
    --file "${REPO_DIR}/docker/phantom-a100/Dockerfile"
    --tag "${DEVELOPMENT_IMAGE}"
    "${REPO_DIR}/docker/phantom-a100"
  )
  docker build "${BUILD_ARGUMENTS[@]}"
elif ! docker image inspect "${DEVELOPMENT_IMAGE}" >/dev/null 2>&1; then
  echo "development image is absent: ${DEVELOPMENT_IMAGE}" >&2
  exit 1
fi

IMAGE_ID="$(
  docker image inspect --format '{{.Id}}' "${DEVELOPMENT_IMAGE}"
)"
IMAGE_DEFINITION_SHA256="$(
  docker image inspect --format \
    '{{index .Config.Labels "org.ace.phantom.definition-sha256"}}' \
    "${DEVELOPMENT_IMAGE}"
)"
if [[ "${IMAGE_DEFINITION_SHA256}" != "${DEFINITION_SHA256}" ]]; then
  echo "development image definition label does not match the repository" >&2
  exit 1
fi
if [[ -n "${EXPECTED_IMAGE_ID}" && "${IMAGE_ID}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "development image ID does not match --expected-image-id" >&2
  exit 1
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ ${PRESERVE_EXISTING} -ne 1 ]]; then
    echo "container ${CONTAINER_NAME} already exists; preservation is required" >&2
    exit 1
  fi
  EXISTING_RUNNING="$(
    docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}"
  )"
  if [[ "${EXISTING_RUNNING}" == "true" && ${PRESERVE_RUNNING} -ne 1 ]]; then
    echo "container ${CONTAINER_NAME} is running; use --preserve-running explicitly" >&2
    exit 1
  fi
  EXISTING_ID="$(
    docker inspect --format '{{.Id}}' "${CONTAINER_NAME}"
  )"
  PRESERVED_NAME="${CONTAINER_NAME}-preserved-${EXISTING_ID:0:12}"
  if docker container inspect "${PRESERVED_NAME}" >/dev/null 2>&1; then
    echo "preservation target already exists: ${PRESERVED_NAME}" >&2
    exit 1
  fi
  docker rename "${CONTAINER_NAME}" "${PRESERVED_NAME}"
  echo "preserved existing container as ${PRESERVED_NAME}"
fi

RUN_ARGUMENTS=(
  --detach
  --name "${CONTAINER_NAME}"
  --workdir /app
  --mount "type=bind,src=${REPO_DIR},dst=/app"
  --mount "type=bind,src=${PHANTOM_DIR},dst=/deps/phantom-ant,readonly"
  --mount "type=bind,src=${MODELS_DIR},dst=/inputs/models,readonly"
  --mount "type=bind,src=${DATASET_DIR},dst=/inputs/dataset,readonly"
  --env ACE_DATASET_DIR=/inputs/dataset
  --env CIFAR10_DIR=/inputs/dataset
  --env "ACE_PHANTOM_IMAGE_ID=${IMAGE_ID}"
  --env "ACE_PHANTOM_DEFINITION_SHA256=${DEFINITION_SHA256}"
  "${DEVELOPMENT_IMAGE}"
)
docker run "${RUN_ARGUMENTS[@]}"

CONTAINER_IMAGE_ID="$(
  docker inspect --format '{{.Image}}' "${CONTAINER_NAME}"
)"
if [[ "${CONTAINER_IMAGE_ID}" != "${IMAGE_ID}" ]]; then
  echo "created container does not use the inspected image ID" >&2
  exit 1
fi

for TARGET in /deps/phantom-ant /inputs/models /inputs/dataset; do
  RW_FLAG="$(
    docker inspect --format "{{range .Mounts}}{{if eq .Destination \"${TARGET}\"}}{{.RW}}{{end}}{{end}}" "${CONTAINER_NAME}"
  )"
  if [[ "${RW_FLAG}" != "false" ]]; then
    echo "mount is not read-only in Docker metadata: ${TARGET}" >&2
    exit 1
  fi
done

docker exec "${CONTAINER_NAME}" bash -lc '
  set -euo pipefail
  test "${ACE_PHANTOM_TOOLCHAIN}" = "12.4.1-sm80"
  test "${CMAKE_CUDA_ARCHITECTURES}" = "80"
  test -n "${ACE_PHANTOM_IMAGE_ID}"
  test -n "${ACE_PHANTOM_DEFINITION_SHA256}"
  for target in /deps/phantom-ant /inputs/models /inputs/dataset; do
    findmnt -T "${target}" -n -o OPTIONS |
      tr "," "\n" |
      grep -Fxq ro
  done
  python3 /app/tools/phantom_gpu/check_configuration.py
'

echo "container ${CONTAINER_NAME} is ready with verified read-only inputs"
