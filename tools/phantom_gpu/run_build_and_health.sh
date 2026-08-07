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
  local poly_degree mul_level input_level security_level scaling_bits first_prime_bits hamming_weight
  poly_degree="$(jq -er '.compiler_context_options.poly_degree' "${INPUT}/payload.json")"
  mul_level="$(jq -er '.compiler_context_options.mul_level' "${INPUT}/payload.json")"
  input_level="$(jq -er '.compiler_context_options.input_level' "${INPUT}/payload.json")"
  security_level="$(jq -er '.compiler_context_options.security_level' "${INPUT}/payload.json")"
  scaling_bits="$(jq -er '.compiler_context_options.scaling_factor_bits' "${INPUT}/payload.json")"
  first_prime_bits="$(jq -er '.compiler_context_options.first_prime_bits' "${INPUT}/payload.json")"
  hamming_weight="$(jq -er '.compiler_context_options.hamming_weight' "${INPUT}/payload.json")"
  bash "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compile_only.sh" --gate ordinary \
    --poly-degree "${poly_degree}" \
    --mul-level "${mul_level}" \
    --input-level "${input_level}" \
    --security-level "${security_level}" \
    --scaling-factor-bits "${scaling_bits}" \
    --first-prime-bits "${first_prime_bits}" \
    --hamming-weight "${hamming_weight}" \
    2>&1 | tee "${RESULT_DIR}/qualification.log"
  local current run_root
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json"
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

verify_frozen_ordinary_reference() {
  local run_root ordinary_dir generated_context generated_resources
  local context_sha resources_sha fixture_sha cpu_reference_sha cpu_values_sha
  local ant_verification_sha
  run_root="$(jq -er '.run_root' \
    "${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json")"
  ordinary_dir="${run_root}/ordinary_ckks"
  generated_context="${run_root}/ckks2c/compiler_context_manifest.json"
  generated_resources="${run_root}/ckks2c/compiler_resource_manifest.json"
  cmp "${generated_context}" "${INPUT}/ordinary-context-manifest.json"
  cmp "${generated_resources}" "${INPUT}/ordinary-resource-manifest.json"
  cmp "${ordinary_dir}/ordinary_ckks_v1.json" "${INPUT}/ordinary-fixture.json"

  context_sha="$(sha256sum "${INPUT}/ordinary-context-manifest.json" | awk '{print $1}')"
  resources_sha="$(sha256sum "${INPUT}/ordinary-resource-manifest.json" | awk '{print $1}')"
  fixture_sha="$(sha256sum "${INPUT}/ordinary-fixture.json" | awk '{print $1}')"
  cpu_reference_sha="$(sha256sum "${INPUT}/ordinary-cpu-reference.json" | awk '{print $1}')"
  cpu_values_sha="$(sha256sum "${INPUT}/ordinary-cpu-values.bin" | awk '{print $1}')"
  ant_verification_sha="$(sha256sum "${INPUT}/ordinary-ant-verification.json" | awk '{print $1}')"
  jq -e \
    --arg context_sha "${context_sha}" \
    --arg resources_sha "${resources_sha}" \
    --arg fixture_sha "${fixture_sha}" \
    --arg cpu_reference_sha "${cpu_reference_sha}" \
    --arg cpu_values_sha "${cpu_values_sha}" \
    --arg ant_verification_sha "${ant_verification_sha}" \
    '.frozen_ordinary_reference.context_manifest_sha256 == $context_sha
     and .frozen_ordinary_reference.resource_manifest_sha256 == $resources_sha
     and .frozen_ordinary_reference.fixture_sha256 == $fixture_sha
     and .frozen_ordinary_reference.cpu_reference_sha256 == $cpu_reference_sha
     and .frozen_ordinary_reference.cpu_values_sha256 == $cpu_values_sha
     and .frozen_ordinary_reference.ant_verification_sha256 == $ant_verification_sha' \
    "${INPUT}/payload.json" >/dev/null
  jq -e '.status == "pass" and .case_count == 43' \
    "${INPUT}/ordinary-ant-verification.json" >/dev/null
  python3 "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/ordinary_ckks_fixture.py" \
    verify-ant-reference \
    --fixture "${INPUT}/ordinary-fixture.json" \
    --context-manifest "${INPUT}/ordinary-context-manifest.json" \
    --cpu-json "${INPUT}/ordinary-cpu-reference.json" \
    --cpu-bin "${INPUT}/ordinary-cpu-values.bin" \
    --output-json "${RESULT_DIR}/ordinary-frozen-ant-verification.json"
  jq -e '.status == "pass" and .case_count == 43' \
    "${RESULT_DIR}/ordinary-frozen-ant-verification.json" >/dev/null
  jq -n \
    --arg status pass \
    --arg context_sha256 "${context_sha}" \
    --arg resource_sha256 "${resources_sha}" \
    --arg fixture_sha256 "${fixture_sha}" \
    --arg cpu_reference_sha256 "${cpu_reference_sha}" \
    --arg cpu_values_sha256 "${cpu_values_sha}" \
    --arg ant_verification_sha256 "${ant_verification_sha}" \
    '{schema_version:"1.0.0", status:$status,
      generated_context_matches_frozen:true,
      generated_resources_match_frozen:true,
      generated_fixture_matches_frozen:true,
      context_manifest_sha256:$context_sha256,
      resource_manifest_sha256:$resource_sha256,
      fixture_sha256:$fixture_sha256,
      cpu_reference_sha256:$cpu_reference_sha256,
      cpu_values_sha256:$cpu_values_sha256,
      ant_verification_sha256:$ant_verification_sha256,
      analytic_and_ant_verification:"pass"}' \
    >"${RESULT_DIR}/ordinary-frozen-reference.json"
}

run_ordinary_gpu_qualification() {
  local expected_gpu run_root ordinary_dir context_manifest fixture
  local cpu_reference cpu_values runner keyless_runner raw_results
  local gpu_results gpu_values comparison
  expected_gpu="${ACE_RUNPOD_EXPECTED_GPU_NAME:-}"
  case "${expected_gpu}" in
    "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB") ;;
    *)
      echo "an exact supported A100 GPU identity is required" >&2
      return 1
      ;;
  esac
  command -v compute-sanitizer >/dev/null
  run_root="$(jq -er '.run_root' \
    "${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json")"
  ordinary_dir="${run_root}/ordinary_ckks"
  context_manifest="${run_root}/ckks2c/compiler_context_manifest.json"
  fixture="${INPUT}/ordinary-fixture.json"
  cpu_reference="${INPUT}/ordinary-cpu-reference.json"
  cpu_values="${INPUT}/ordinary-cpu-values.bin"
  runner="${ordinary_dir}/ordinary_ckks_gpu_runner_sm80"
  keyless_runner="${ordinary_dir}/ordinary_ckks_keyless_runner_sm80"
  for required in \
    "${context_manifest}" "${fixture}" "${cpu_reference}" "${cpu_values}" \
    "${runner}" "${keyless_runner}"; do
    [[ -s "${required}" ]]
  done
  raw_results="${RESULT_DIR}/ordinary_ckks_gpu_raw.json"
  gpu_results="${RESULT_DIR}/ordinary_ckks_gpu_results.json"
  gpu_values="${RESULT_DIR}/ordinary_ckks_gpu_values.bin"
  comparison="${RESULT_DIR}/ordinary_ckks_compare.json"
  cp "${context_manifest}" "${RESULT_DIR}/compiler_context_manifest.json"
  cp "${INPUT}/ordinary-resource-manifest.json" \
    "${RESULT_DIR}/compiler_resource_manifest.json"
  cp "${fixture}" "${RESULT_DIR}/ordinary_ckks_v1.json"
  cp "${cpu_reference}" "${RESULT_DIR}/ordinary_ckks_cpu_reference.json"
  cp "${cpu_values}" "${RESULT_DIR}/ordinary_ckks_cpu_values.bin"

  timeout 600 "${runner}" conformance "${fixture}" "${expected_gpu}" \
    "${raw_results}" \
    >"${RESULT_DIR}/ordinary-conformance.stdout.txt" \
    2>"${RESULT_DIR}/ordinary-conformance.stderr.txt"
  python3 "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/ordinary_ckks_fixture.py" \
    ingest-raw-provider \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --provider phantom --raw-json "${raw_results}" \
    --output-json "${gpu_results}" --output-bin "${gpu_values}"
  python3 "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/ordinary_ckks_fixture.py" \
    compare \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --cpu-json "${cpu_reference}" --cpu-bin "${cpu_values}" \
    --gpu-json "${gpu_results}" --gpu-bin "${gpu_values}" \
    --output-json "${comparison}"
  jq -e '.status == "pass" and .case_count == 43 and .passing_case_count == 43' \
    "${comparison}" >/dev/null

  local diagnostics="${RESULT_DIR}/ordinary_ckks_diagnostics.jsonl"
  : >"${diagnostics}"
  local case_id expected_token selected_runner exit_code stdout_file stderr_file
  while IFS=$'\t' read -r case_id expected_token selected_runner; do
    stdout_file="${RESULT_DIR}/reject-${case_id}.stdout.txt"
    stderr_file="${RESULT_DIR}/reject-${case_id}.stderr.txt"
    ulimit -c 0
    set +e
    timeout 120 "${selected_runner}" reject "${fixture}" "${expected_gpu}" \
      "${case_id}" >"${stdout_file}" 2>"${stderr_file}"
    exit_code=$?
    set -e
    if [[ ${exit_code} -eq 0 || ${exit_code} -eq 124 ]]; then
      echo "runtime rejection ${case_id} returned ${exit_code}" >&2
      return 1
    fi
    if ! rg -Fq "ACE_PHANTOM_ORDINARY_ERROR[${expected_token}]" \
        "${stderr_file}"; then
      echo "runtime rejection ${case_id} missed ${expected_token}" >&2
      return 1
    fi
    jq -cn --arg case_id "${case_id}" --arg diagnostic "${expected_token}" \
      --argjson exit_code "${exit_code}" \
      '{case_id:$case_id, diagnostic:$diagnostic, exit_code:$exit_code, status:"pass"}' \
      >>"${diagnostics}"
  done <<EOF
bottom_modswitch	MODSWITCH_BOTTOM_CHAIN	${runner}
bottom_rescale	RESCALE_BOTTOM_CHAIN	${runner}
double_free	DOUBLE_FREE_CIPHER	${runner}
invalid_ciphertext_size	QUERY_SIZE	${runner}
missing_loaded_relin_key	RELIN_KEY_MISSING	${keyless_runner}
missing_loaded_rotation_key	ROTATE_KEY_MISSING	${keyless_runner}
relinearize_size2	RELIN_SIZE	${runner}
runtime_level_mismatch	ADD_COMPAT	${runner}
runtime_scale_mismatch	ADD_COMPAT	${runner}
use_after_free	USE_AFTER_FREE_CIPHER	${runner}
EOF
  jq -s '{schema_version:"1.0.0", status:"pass", cases:.}' \
    "${diagnostics}" >"${RESULT_DIR}/ordinary_ckks_diagnostics.json"
  rm "${diagnostics}"

  set +e
  timeout 900 compute-sanitizer --tool memcheck --error-exitcode=99 \
    "${runner}" ownership "${fixture}" "${expected_gpu}" \
    >"${RESULT_DIR}/ordinary_ckks_sanitizer.stdout.txt" \
    2>"${RESULT_DIR}/ordinary_ckks_sanitizer.stderr.txt"
  local sanitizer_exit=$?
  set -e
  if [[ ${sanitizer_exit} -ne 0 ]]; then
    echo "Compute Sanitizer exited ${sanitizer_exit}" >&2
    return "${sanitizer_exit}"
  fi
  rg -Fq 'ERROR SUMMARY: 0 errors' \
    "${RESULT_DIR}/ordinary_ckks_sanitizer.stderr.txt"
  jq -n --arg status pass --arg tool memcheck --argjson exit_code 0 \
    '{schema_version:"1.0.0", status:$status, tool:$tool,
      error_summary:0, exit_code:$exit_code, ownership_iterations:100}' \
    >"${RESULT_DIR}/ordinary_ckks_sanitizer.json"
  rm "${raw_results}"
}

run_native_health() {
  local expected_gpu query gpu_count gpu_name health_binary context_manifest context_sha health_json health_exit
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
    -path '*/ckks2c/native_phantom_health_sm80' -type f -print -quit)"
  [[ -n "${health_binary}" && -x "${health_binary}" ]]
  context_manifest="$(find "${WORK}/build-state/compile_only_results/runs" \
    -path '*/ckks2c/compiler_context_manifest.json' -type f -print -quit)"
  [[ -n "${context_manifest}" && -s "${context_manifest}" ]]
  context_sha="$(sha256sum "${context_manifest}" | awk '{print $1}')"
  set +e
  timeout 300 "${health_binary}" \
    >"${RESULT_DIR}/native-health.stdout.txt" \
    2>"${RESULT_DIR}/native-health.stderr.txt"
  health_exit=$?
  set -e
  [[ ${health_exit} -eq 0 ]]
  health_json="$(tail -n 1 "${RESULT_DIR}/native-health.stdout.txt")"
  jq -e --arg context_sha "${context_sha}" --arg expected_gpu "${expected_gpu}" \
    'select(.status == "pass" and .device_count == 1
     and .gpu == $expected_gpu
     and (.max_error | type == "number") and .max_error <= 0.0001
     and .context_manifest_sha256 == $context_sha)' <<<"${health_json}" \
    >"${RESULT_DIR}/native-health.json"
}

phase payload_verification verify_outer_payload
phase environment_bootstrap bootstrap
export PATH="/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
phase source_audit_and_extraction extract_sources
phase qualification run_qualification
phase frozen_ordinary_reference verify_frozen_ordinary_reference
if [[ "${MODE}" == "runpod" ]]; then
  phase native_a100_health run_native_health
  phase ordinary_gpu_qualification run_ordinary_gpu_qualification
else
  printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
    >"${RESULT_DIR}/native-health.json"
fi
PIPELINE_EXIT=0
