#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR=""
RESULT_DIR=""
ARCHIVE_PATH=""

usage() {
  echo "usage: $0 --bundle-dir DIR --result-dir DIR --archive FILE"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --archive)
      ARCHIVE_PATH="$2"
      shift 2
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

if [[ -z "${BUNDLE_DIR}" || -z "${RESULT_DIR}" || -z "${ARCHIVE_PATH}" ]]; then
  usage >&2
  exit 2
fi

BUNDLE_DIR="$(realpath "${BUNDLE_DIR}")"
if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory must not already exist: ${RESULT_DIR}" >&2
  exit 1
fi
if [[ -e "${ARCHIVE_PATH}" ]]; then
  echo "result archive must not already exist: ${ARCHIVE_PATH}" >&2
  exit 1
fi
mkdir -p "$(dirname -- "${RESULT_DIR}")" "$(dirname -- "${ARCHIVE_PATH}")"
RESULT_DIR="$(realpath -m "${RESULT_DIR}")"
ARCHIVE_PATH="$(realpath -m "${ARCHIVE_PATH}")"
case "${ARCHIVE_PATH}" in
  "${RESULT_DIR}"/*)
    echo "result archive must be outside the result directory" >&2
    exit 1
    ;;
esac
mkdir "${RESULT_DIR}"
RESULT_ACTIVE=1

atomic_json() {
  local target="$1"
  shift
  local temporary
  temporary="$(mktemp "${target}.tmp.XXXXXX")"
  jq "$@" >"${temporary}"
  chmod 0644 "${temporary}"
  mv "${temporary}" "${target}"
}

write_state() {
  local status="$1"
  local exit_code="$2"
  atomic_json "${RESULT_DIR}/state.json" -n \
    --arg status "${status}" \
    --argjson exit_code "${exit_code}" \
    '{status: $status, exit_code: $exit_code}'
}
write_state running 0

archive_on_exit() {
  local exit_code="$?"
  trap - EXIT
  set +e
  if [[ ${RESULT_ACTIVE} -eq 1 ]]; then
    write_state failed "${exit_code}"
  fi
  if ! (
    umask 022
    tar -C "$(dirname -- "${RESULT_DIR}")" -czf "${ARCHIVE_PATH}" \
      "$(basename -- "${RESULT_DIR}")"
  ); then
    if [[ ${exit_code} -eq 0 ]]; then
      exit_code=1
      write_state failed "${exit_code}"
    fi
  fi
  exit "${exit_code}"
}
trap archive_on_exit EXIT

(
  cd "${BUNDLE_DIR}"
  sha256sum -c SHA256SUMS
) >"${RESULT_DIR}/bundle_verification.txt"

BUNDLE_MANIFEST="${BUNDLE_DIR}/bundle_manifest.json"
EXPECTED_IMAGE_ID="$(jq -er '.development_image_id' "${BUNDLE_MANIFEST}")"
EXPECTED_DEFINITION_SHA256="$(
  jq -er '.development_definition_sha256' "${BUNDLE_MANIFEST}"
)"
EXPECTED_CONTEXT_MANIFEST_SHA256="$(
  jq -er '.compiler_context_manifest_sha256' "${BUNDLE_MANIFEST}"
)"
EXPECTED_BINARY_SHA256="$(
  jq -er '.health_binary_sha256' "${BUNDLE_MANIFEST}"
)"
EXPECTED_REGISTRY_IMAGE="$(jq -er '.registry_image' "${BUNDLE_MANIFEST}")"
HEALTH_TIMEOUT_SECONDS="$(
  jq -er '.health_timeout_seconds' "${BUNDLE_MANIFEST}"
)"
test "${HEALTH_TIMEOUT_SECONDS}" = "300"
test "${ACE_PHANTOM_IMAGE_ID:-}" = "${EXPECTED_IMAGE_ID}"
test "${ACE_PHANTOM_DEFINITION_SHA256:-}" = "${EXPECTED_DEFINITION_SHA256}"
test "${ACE_PHANTOM_REGISTRY_IMAGE:-}" = "${EXPECTED_REGISTRY_IMAGE}"

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi -L >"${RESULT_DIR}/nvidia_smi_list.txt"
nvidia-smi \
  --query-gpu=name,uuid,memory.total,driver_version,pstate,power.limit \
  --format=csv,noheader \
  >"${RESULT_DIR}/nvidia_smi_query.csv"
GPU_COUNT="$(wc -l <"${RESULT_DIR}/nvidia_smi_query.csv" | tr -d ' ')"
GPU_NAME="$(cut -d, -f1 "${RESULT_DIR}/nvidia_smi_query.csv" | sed 's/[[:space:]]*$//')"
if [[ "${GPU_COUNT}" != "1" ]]; then
  echo "expected exactly one visible GPU, found ${GPU_COUNT}" >&2
  exit 1
fi
if [[ "${GPU_NAME}" != "NVIDIA A100 80GB PCIe" ]]; then
  echo "expected NVIDIA A100 80GB PCIe, found ${GPU_NAME}" >&2
  exit 1
fi

HEALTH_BINARY="${BUNDLE_DIR}/native_phantom_health_sm80"
test "$(sha256sum "${HEALTH_BINARY}" | awk '{print $1}')" = \
  "${EXPECTED_BINARY_SHA256}"

set +e
timeout "${HEALTH_TIMEOUT_SECONDS}" "${HEALTH_BINARY}" \
  >"${RESULT_DIR}/native_health.stdout.txt" \
  2>"${RESULT_DIR}/native_health.stderr.txt"
HEALTH_EXIT_CODE="$?"
set -e
if [[ ${HEALTH_EXIT_CODE} -ne 0 ]]; then
  echo "native Phantom health exited ${HEALTH_EXIT_CODE}" >&2
  exit "${HEALTH_EXIT_CODE}"
fi

HEALTH_JSON="$(tail -n 1 "${RESULT_DIR}/native_health.stdout.txt")"
echo "${HEALTH_JSON}" | jq -e \
  --arg context_manifest_sha256 "${EXPECTED_CONTEXT_MANIFEST_SHA256}" \
  '.status == "pass"
   and .device_count == 1
   and .gpu == "NVIDIA A100 80GB PCIe"
   and .max_error <= 0.0001
   and .context_manifest_sha256 == $context_manifest_sha256' >/dev/null
echo "${HEALTH_JSON}" >"${RESULT_DIR}/native_health.json"
COMPLETED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

atomic_json "${RESULT_DIR}/result.json" -n \
  --arg status pass \
  --arg started_utc "${STARTED_UTC}" \
  --arg completed_utc "${COMPLETED_UTC}" \
  --arg gpu_name "${GPU_NAME}" \
  --arg image_id "${EXPECTED_IMAGE_ID}" \
  --arg definition_sha256 "${EXPECTED_DEFINITION_SHA256}" \
  --arg registry_image "${EXPECTED_REGISTRY_IMAGE}" \
  --arg context_manifest_sha256 "${EXPECTED_CONTEXT_MANIFEST_SHA256}" \
  --arg health_binary_sha256 "${EXPECTED_BINARY_SHA256}" \
  --argjson health_exit_code "${HEALTH_EXIT_CODE}" \
  --argjson max_error "$(echo "${HEALTH_JSON}" | jq '.max_error')" \
  '{
    status: $status,
    started_utc: $started_utc,
    completed_utc: $completed_utc,
    gpu_count: 1,
    gpu_name: $gpu_name,
    development_image_id: $image_id,
    development_definition_sha256: $definition_sha256,
    registry_image: $registry_image,
    compiler_context_manifest_sha256: $context_manifest_sha256,
    health_binary_sha256: $health_binary_sha256,
    health_exit_code: $health_exit_code,
    max_error: $max_error
  }'
write_state pass 0
RESULT_ACTIVE=0

cat "${RESULT_DIR}/result.json"
