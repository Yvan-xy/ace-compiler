#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PROFILE_PATH="${REPO_ROOT}/fhe-cmplr/rtlib/phantom/config/fullpacked_bts_v1.json"
QUALIFICATION_RECORD=""
REGISTRY_IMAGE=""
OUTPUT_ROOT="${REPO_ROOT}/build/phantom_gpu/a100_health_bundles"

usage() {
  echo "usage: $0 --qualification-record FILE --registry-image IMAGE@DIGEST [--output-root DIR]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qualification-record)
      QUALIFICATION_RECORD="$2"
      shift 2
      ;;
    --registry-image)
      REGISTRY_IMAGE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
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

if [[ -z "${QUALIFICATION_RECORD}" || -z "${REGISTRY_IMAGE}" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "${REGISTRY_IMAGE}" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]; then
  echo "registry image must be pinned by a SHA-256 digest" >&2
  exit 2
fi
if [[ ! -f /.dockerenv ]]; then
  echo "bundle packaging must run inside ace-compiler-dev" >&2
  exit 1
fi

QUALIFICATION_RECORD="$(realpath "${QUALIFICATION_RECORD}")"
EXPECTED_RECORD="$(realpath "${REPO_ROOT}/build/phantom_gpu/compile_only_results/current-ckks2c.json")"
if [[ "${QUALIFICATION_RECORD}" != "${EXPECTED_RECORD}" ]]; then
  echo "qualification record must be the current CKKS2C record" >&2
  exit 1
fi
if ! jq -e '.status == "pass" and .gate == "ckks2c" and .exit_code == 0' \
  "${QUALIFICATION_RECORD}" >/dev/null; then
  echo "qualification record is not a current CKKS2C pass" >&2
  exit 1
fi
RUN_ROOT="$(jq -er '.run_root' "${QUALIFICATION_RECORD}")"
RUN_ID="$(jq -er '.run_id' "${QUALIFICATION_RECORD}")"
if [[ ! "${RUN_ID}" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9]+\.[[:alnum:]]{6}$ ]]; then
  echo "qualification run identifier is malformed" >&2
  exit 1
fi
EXPECTED_RUN_ROOT="${REPO_ROOT}/build/phantom_gpu/compile_only_results/runs/${RUN_ID}"
if [[ "$(realpath "${RUN_ROOT}")" != "${EXPECTED_RUN_ROOT}" ]]; then
  echo "qualification run root is outside the expected results directory" >&2
  exit 1
fi
MANIFEST_PATH="$(jq -er '.manifest_path' "${QUALIFICATION_RECORD}")"
MANIFEST_SHA256="$(jq -er '.manifest_sha256' "${QUALIFICATION_RECORD}")"
if [[ "${MANIFEST_PATH}" != "${RUN_ROOT}/manifest.json" ]]; then
  echo "qualification manifest is not inside its immutable run" >&2
  exit 1
fi
if [[ "$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')" != "${MANIFEST_SHA256}" ]]; then
  echo "qualification manifest hash does not match its current record" >&2
  exit 1
fi
if ! jq -e \
  --arg run_id "${RUN_ID}" \
  --arg image_id "$(jq -er '.development_image_id' "${QUALIFICATION_RECORD}")" \
  --arg definition_sha256 "$(jq -er '.development_definition_sha256' "${QUALIFICATION_RECORD}")" \
  '.status == "pass" and .gate == "ckks2c" and .run_id == $run_id
   and .development_image_id == $image_id
   and .development_definition_sha256 == $definition_sha256' \
  "${MANIFEST_PATH}" >/dev/null; then
  echo "qualification record and run manifest disagree" >&2
  exit 1
fi
HOST_SUMS="${RUN_ROOT}/SHA256SUMS"
HOST_SUMS_SHA256="$(jq -er '.evidence_sha256_manifest_sha256' "${MANIFEST_PATH}")"
if [[ "$(sha256sum "${HOST_SUMS}" | awk '{print $1}')" != "${HOST_SUMS_SHA256}" ]]; then
  echo "host evidence checksum manifest does not match the run manifest" >&2
  exit 1
fi
(
  cd "${RUN_ROOT}"
  sha256sum -c SHA256SUMS
) >/dev/null

HEALTH_BINARY="${RUN_ROOT}/toolchain/native_phantom_health_sm80"
TOOLCHAIN_QUALIFICATION="${RUN_ROOT}/toolchain/qualification.json"
test -x "${HEALTH_BINARY}"
jq -e '.status == "pass" and .executable_was_run == false' \
  "${TOOLCHAIN_QUALIFICATION}" >/dev/null

BUNDLE_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -e "${BUNDLE_DIR}" ]]; then
  echo "bundle directory already exists: ${BUNDLE_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
STAGING_ROOT="$(mktemp -d "${OUTPUT_ROOT}/.bundle.XXXXXX")"
STAGING_DIR="${STAGING_ROOT}/${RUN_ID}"
mkdir "${STAGING_DIR}"
cleanup_staging() {
  rm -rf -- "${STAGING_ROOT}"
}
trap cleanup_staging EXIT
PACKAGE_DIR="${STAGING_DIR}"
cp "${HEALTH_BINARY}" "${PACKAGE_DIR}/native_phantom_health_sm80"
cp "${SCRIPT_DIR}/run_a100_health.sh" "${PACKAGE_DIR}/run_a100_health.sh"
cp "${PROFILE_PATH}" "${PACKAGE_DIR}/fullpacked_bts_v1.json"
cp "${TOOLCHAIN_QUALIFICATION}" "${PACKAGE_DIR}/host_qualification.json"
cp "${MANIFEST_PATH}" "${PACKAGE_DIR}/host_manifest.json"
cp "${HOST_SUMS}" "${PACKAGE_DIR}/host_SHA256SUMS"

IMAGE_ID="$(jq -er '.development_image_id' "${QUALIFICATION_RECORD}")"
DEFINITION_SHA256="$(
  jq -er '.development_definition_sha256' "${QUALIFICATION_RECORD}"
)"
PROFILE_SHA256="$(sha256sum "${PROFILE_PATH}" | awk '{print $1}')"
HEALTH_BINARY_SHA256="$(sha256sum "${HEALTH_BINARY}" | awk '{print $1}')"
if [[ "$(jq -er '.health_binary_sha256' "${TOOLCHAIN_QUALIFICATION}")" != \
      "${HEALTH_BINARY_SHA256}" ]]; then
  echo "health binary does not match the host qualification" >&2
  exit 1
fi
if [[ "$(jq -er '.profile_sha256' "${TOOLCHAIN_QUALIFICATION}")" != \
      "${PROFILE_SHA256}" ]]; then
  echo "profile does not match the host qualification" >&2
  exit 1
fi
if [[ "$(jq -er '.profile_sha256' "${MANIFEST_PATH}")" != \
      "${PROFILE_SHA256}" ]]; then
  echo "profile does not match the host manifest" >&2
  exit 1
fi
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg image_id "${IMAGE_ID}" \
  --arg definition_sha256 "${DEFINITION_SHA256}" \
  --arg profile_sha256 "${PROFILE_SHA256}" \
  --arg health_binary_sha256 "${HEALTH_BINARY_SHA256}" \
  --arg host_manifest_sha256 "${MANIFEST_SHA256}" \
  --arg host_sums_sha256 "${HOST_SUMS_SHA256}" \
  --arg registry_image "${REGISTRY_IMAGE}" \
  '{
    schema_version: "1.0.0",
    run_id: $run_id,
    development_image_id: $image_id,
    development_definition_sha256: $definition_sha256,
    profile_sha256: $profile_sha256,
    health_binary_sha256: $health_binary_sha256,
    host_manifest_sha256: $host_manifest_sha256,
    host_sums_sha256: $host_sums_sha256,
    registry_image: $registry_image,
    expected_gpu_count: 1,
    expected_gpu_name: "NVIDIA A100 80GB PCIe",
    health_timeout_seconds: 300
  }' >"${PACKAGE_DIR}/bundle_manifest.json"

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS -print0 |
    LC_ALL=C sort -z |
    xargs -0 sha256sum
) >"${PACKAGE_DIR}/SHA256SUMS"

ARCHIVE_PATH="${OUTPUT_ROOT}/${RUN_ID}.tar.gz"
ARCHIVE_TEMP="$(mktemp "${OUTPUT_ROOT}/.${RUN_ID}.tar.gz.XXXXXX")"
tar -C "${STAGING_ROOT}" -czf "${ARCHIVE_TEMP}" "${RUN_ID}"
mv "${PACKAGE_DIR}" "${BUNDLE_DIR}"
mv "${ARCHIVE_TEMP}" "${ARCHIVE_PATH}"
ARCHIVE_SHA256="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
printf '%s  %s\n' "${ARCHIVE_SHA256}" "$(basename -- "${ARCHIVE_PATH}")" \
  >"${ARCHIVE_PATH}.sha256"
trap - EXIT
cleanup_staging
echo "bundle_dir=${BUNDLE_DIR}"
echo "archive_path=${ARCHIVE_PATH}"
echo "archive_sha256=${ARCHIVE_SHA256}"
