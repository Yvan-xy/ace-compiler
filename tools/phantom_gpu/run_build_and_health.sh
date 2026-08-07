#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/phase_helpers.sh"

MODE=""
INPUT=""
WORK=""
RESULT_ARCHIVE=""

usage() {
  echo "usage: $0 --mode <local|runpod> --input-dir DIR --work-dir DIR --result-archive FILE" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --input-dir) INPUT="$2"; shift 2 ;;
    --work-dir) WORK="$2"; shift 2 ;;
    --result-archive) RESULT_ARCHIVE="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
if [[ "${MODE}" != "local" && "${MODE}" != "runpod" ]] ||
   [[ -z "${INPUT}" || -z "${WORK}" || -z "${RESULT_ARCHIVE}" ]]; then
  usage
  exit 2
fi
if [[ -e "${WORK}" ]]; then
  echo "work directory already exists: ${WORK}" >&2
  exit 1
fi

INPUT="$(realpath -- "${INPUT}")"
WORK="$(realpath -m -- "${WORK}")"
RESULT_ARCHIVE="$(realpath -m -- "${RESULT_ARCHIVE}")"
RESULT_DIR="${WORK}/results"
mkdir -p "${RESULT_DIR}"
chmod 0755 "${RESULT_DIR}"
TIMINGS="${RESULT_DIR}/phase-timings.tsv"
: >"${TIMINGS}"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PIPELINE_EXIT=1

finalize() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ ${PIPELINE_EXIT} -eq 0 ]]; then
    incoming=0
  fi
  local status=failed
  [[ ${incoming} -eq 0 ]] && status=pass
  printf '{"schema_version":"1.0.0","status":"%s","mode":"%s","exit_code":%d,"started_utc":"%s","completed_utc":"%s"}\n' \
    "${status}" "${MODE}" "${incoming}" "${STARTED_UTC}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RESULT_DIR}/pipeline-result.json"
  (
    cd "${RESULT_DIR}"
    find . -type f ! -name SHA256SUMS -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum >SHA256SUMS
  )
  mkdir -p "$(dirname -- "${RESULT_ARCHIVE}")"
  local temporary="${RESULT_ARCHIVE}.tmp"
  rm -f "${temporary}"
  tar -C "${WORK}" -czf "${temporary}" results
  mv "${temporary}" "${RESULT_ARCHIVE}"
  local archive_directory archive_name
  archive_directory="$(dirname -- "${RESULT_ARCHIVE}")"
  archive_name="$(basename -- "${RESULT_ARCHIVE}")"
  (
    cd "${archive_directory}"
    sha256sum "${archive_name}" >"${archive_name}.sha256"
  )
  exit "${incoming}"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 124' TERM

phase() {
  run_timed_phase "${TIMINGS}" "$@"
}

verify_outer_payload() {
  (
    cd "${INPUT}"
    sha256sum -c SHA256SUMS
  ) | tee "${RESULT_DIR}/payload-verification.txt"
}

bootstrap() {
  ACE_APT_LOCK="${INPUT}/apt-packages.lock" \
  ACE_PYTHON_LOCK="${INPUT}/python-requirements-hashed.lock" \
  ACE_BASE_FILES_LOCK="${INPUT}/base-files.sha256" \
  ACE_DEPENDENCIES_LOCK="${INPUT}/dependencies.env" \
    bash "${INPUT}/bootstrap_environment.sh" "${RESULT_DIR}/environment"
}

extract_sources() {
  local ace_archive phantom_archive
  ace_archive="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/ace-source.manifest.json")"
  phantom_archive="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/phantom-source.manifest.json")"
  python3 "${INPUT}/source_archive.py" audit \
    --kind ace \
    --archive "${INPUT}/${ace_archive}" \
    --manifest "${INPUT}/ace-source.manifest.json" \
    --extract "${WORK}/ace-extract" \
    >"${RESULT_DIR}/ace-source-audit.json"
  python3 "${INPUT}/source_archive.py" audit \
    --kind phantom \
    --archive "${INPUT}/${phantom_archive}" \
    --manifest "${INPUT}/phantom-source.manifest.json" \
    --extract "${WORK}/phantom-extract" \
    >"${RESULT_DIR}/phantom-source-audit.json"
}

run_qualification() {
  local ace_commit source_manifest_sha bootstrap_sha
  ace_commit="$(jq -er .commit "${INPUT}/ace-source.manifest.json")"
  export SOURCE_DATE_EPOCH
  SOURCE_DATE_EPOCH="$(jq -er .commit_timestamp "${INPUT}/ace-source.manifest.json")"
  source_manifest_sha="$(sha256sum "${INPUT}/ace-source.manifest.json" | awk '{print $1}')"
  bootstrap_sha="$(sha256sum "${INPUT}/bootstrap_environment.sh" | awk '{print $1}')"
  export PATH="/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
  export PYTHONPATH="${WORK}/ace-extract/ace-source"
  export CMAKE_CUDA_ARCHITECTURES=80
  export CUDAARCHS=80
  export ACE_PHANTOM_TOOLCHAIN=12.4.1-sm80
  export ACE_PHANTOM_SOURCE_MODE=snapshot
  export ACE_PHANTOM_REPO_ROOT="${WORK}/ace-extract/ace-source"
  export ACE_PHANTOM_SOURCE_DIR="${WORK}/phantom-extract/phantom-source"
  export ACE_PHANTOM_STATE_ROOT="${WORK}/build-state"
  export ACE_PHANTOM_ACE_COMMIT="${ace_commit}"
  export ACE_PHANTOM_SOURCE_MANIFEST="${INPUT}/ace-source.manifest.json"
  export ACE_PHANTOM_SOURCE_MANIFEST_SHA256="${source_manifest_sha}"
  export ACE_RUNPOD_BOOTSTRAP_SHA256="${bootstrap_sha}"
  export ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
  bash "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compile_only.sh" --gate ckks2c \
    2>&1 | tee "${RESULT_DIR}/qualification.log"
  local current run_root
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ckks2c.json"
  run_root="$(jq -er '.run_root' "${current}")"
  [[ "$(jq -er '.status' "${current}")" == pass ]]
  cp -a "${run_root}" "${RESULT_DIR}/qualification"
  cp "${current}" "${RESULT_DIR}/qualification-current.json"
  for cache in \
    "${ACE_PHANTOM_STATE_ROOT}/phantom-"*"-sm80/CMakeCache.txt" \
    "${ACE_PHANTOM_STATE_ROOT}/ace-cuda-sm80/CMakeCache.txt" \
    "${ACE_PHANTOM_STATE_ROOT}/bindings-cuda-sm80/CMakeCache.txt" \
    "${ACE_PHANTOM_STATE_ROOT}/ace-cuda-sm80/rtlib/build/CMakeCache.txt" \
    "${ACE_PHANTOM_STATE_ROOT}/ace-cuda-sm80/rtlib/build/external/src/phantom_external-build/CMakeCache.txt"; do
    [[ -f "${cache}" ]] || continue
    cp "${cache}" "${RESULT_DIR}/$(echo "${cache#${ACE_PHANTOM_STATE_ROOT}/}" | tr / _)"
  done
  : >"${RESULT_DIR}/public-dependency-commits.txt"
  while IFS= read -r git_dir; do
    local checkout
    checkout="${git_dir%/.git}"
    printf '%s\t%s\n' "${checkout#${WORK}/}" "$(git -C "${checkout}" rev-parse HEAD)" \
      >>"${RESULT_DIR}/public-dependency-commits.txt"
  done < <(find "${ACE_PHANTOM_STATE_ROOT}" -type d -name .git -print | LC_ALL=C sort)
}

run_native_health() {
  local expected_gpu query gpu_count gpu_name health_binary profile_sha health_json health_exit
  expected_gpu="${ACE_RUNPOD_EXPECTED_GPU_NAME:-}"
  case "${expected_gpu}" in
    "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB") ;;
    *)
      echo "an exact supported A100 GPU identity is required" >&2
      return 1
      ;;
  esac
  nvidia-smi -L >"${RESULT_DIR}/nvidia-smi-list.txt"
  nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,pstate,power.limit \
    --format=csv,noheader >"${RESULT_DIR}/nvidia-smi-query.csv"
  query="${RESULT_DIR}/nvidia-smi-query.csv"
  gpu_count="$(wc -l <"${query}" | tr -d ' ')"
  gpu_name="$(cut -d, -f1 "${query}" | sed 's/[[:space:]]*$//')"
  [[ "${gpu_count}" == 1 ]]
  [[ "${gpu_name}" == "${expected_gpu}" ]]
  health_binary="$(find "${WORK}/build-state/compile_only_results/runs" \
    -path '*/toolchain/native_phantom_health_sm80' -type f -print -quit)"
  [[ -n "${health_binary}" && -x "${health_binary}" ]]
  profile_sha="$(sha256sum "${WORK}/ace-extract/ace-source/fhe-cmplr/rtlib/phantom/config/fullpacked_bts_v1.json" | awk '{print $1}')"
  set +e
  timeout 300 "${health_binary}" \
    >"${RESULT_DIR}/native-health.stdout.txt" \
    2>"${RESULT_DIR}/native-health.stderr.txt"
  health_exit=$?
  set -e
  [[ ${health_exit} -eq 0 ]]
  health_json="$(tail -n 1 "${RESULT_DIR}/native-health.stdout.txt")"
  jq -e --arg profile_sha "${profile_sha}" --arg expected_gpu "${expected_gpu}" \
    'select(.status == "pass" and .device_count == 1
     and .gpu == $expected_gpu
     and (.max_error | type == "number") and .max_error <= 0.0001
     and .profile_sha256 == $profile_sha)' <<<"${health_json}" \
    >"${RESULT_DIR}/native-health.json"
}

phase payload_verification verify_outer_payload
phase environment_bootstrap bootstrap
export PATH="/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
phase source_audit_and_extraction extract_sources
phase qualification run_qualification
if [[ "${MODE}" == "runpod" ]]; then
  phase native_a100_health run_native_health
else
  printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
    >"${RESULT_DIR}/native-health.json"
fi
PIPELINE_EXIT=0
