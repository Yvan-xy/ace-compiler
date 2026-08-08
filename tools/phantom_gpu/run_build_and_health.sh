#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/phase_helpers.sh"

MODE=""
INPUT=""
WORK=""
RESULT_ARCHIVE=""

usage() {
  echo "usage: $0 --mode <freeze-host|local|runpod> --input-dir DIR --work-dir DIR --result-archive FILE" >&2
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
if [[ "${MODE}" != "freeze-host" && "${MODE}" != "local" &&
      "${MODE}" != "runpod" ]] ||
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
CURRENT_PHASE=initialization
FAILED_PHASE=""

finalize() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ ${PIPELINE_EXIT} -eq 0 ]]; then
    incoming=0
  fi
  local status=failed
  [[ ${incoming} -eq 0 ]] && status=pass
  if [[ "${status}" == failed ]]; then
    local failure_phase="${FAILED_PHASE:-${CURRENT_PHASE:-pipeline}}"
    printf '{"schema_version":"1.0.0","status":"failed","phase":"%s","exit_code":%d}\n' \
      "${failure_phase}" "${incoming}" >"${RESULT_DIR}/pipeline-failure.json"
  fi
  printf '{"schema_version":"1.0.0","status":"%s","mode":"%s","exit_code":%d,"started_utc":"%s","completed_utc":"%s"}\n' \
    "${status}" "${MODE}" "${incoming}" "${STARTED_UTC}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RESULT_DIR}/pipeline-result.json"
  (
    cd "${RESULT_DIR}"
    find . -type f ! -path ./SHA256SUMS -print0 |
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
  local phase_name="$1"
  local phase_exit
  local caller_errexit=false
  [[ $- == *e* ]] && caller_errexit=true
  CURRENT_PHASE="${phase_name}"
  set +e
  run_timed_phase "${TIMINGS}" "$@"
  phase_exit=$?
  if [[ "${caller_errexit}" == true ]]; then
    set -e
  else
    set +e
  fi
  if [[ ${phase_exit} -eq 0 ]]; then
    CURRENT_PHASE=""
    return 0
  else
    FAILED_PHASE="${phase_name}"
    CURRENT_PHASE=""
    return "${phase_exit}"
  fi
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
  local ace_archive phantom_archive ace_manifest_sha phantom_manifest_sha
  python3 - "${INPUT}/payload.json" \
    "${INPUT}/ace-source.manifest.json" \
    "${INPUT}/phantom-source.manifest.json" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

payload_path, ace_path, phantom_path = map(Path, sys.argv[1:4])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in source payload: {key}")
        value[key] = item
    return value

def load(path):
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise SystemExit(f"source payload record is not an object: {path.name}")
    return value

payload = load(payload_path)
manifests = {"ace": (ace_path, load(ace_path)),
             "phantom": (phantom_path, load(phantom_path))}
expected_source_bindings = {}
for kind, (path, manifest) in manifests.items():
    commit = manifest.get("commit")
    archive = manifest.get("archive")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("kind") != kind
        or not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or archive != f"{kind}-source-{commit}.tar.gz"
        or PurePosixPath(archive).name != archive
    ):
        raise SystemExit(f"invalid {kind} source manifest identity")
    if payload.get(f"{kind}_commit") != commit:
        raise SystemExit(f"payload does not bind the {kind} source commit")
    expected_source_bindings[f"{kind}_manifest_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
if payload.get("source_snapshots") != expected_source_bindings:
    raise SystemExit("payload does not bind the exact source manifests")
PY
  ace_archive="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/ace-source.manifest.json")"
  phantom_archive="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["archive"])' "${INPUT}/phantom-source.manifest.json")"
  ace_manifest_sha="$(sha256sum "${INPUT}/ace-source.manifest.json" | awk '{print $1}')"
  phantom_manifest_sha="$(sha256sum "${INPUT}/phantom-source.manifest.json" | awk '{print $1}')"
  jq -e \
    --arg ace_commit "$(jq -er .commit "${INPUT}/ace-source.manifest.json")" \
    --arg phantom_commit "$(jq -er .commit "${INPUT}/phantom-source.manifest.json")" \
    --arg ace_manifest_sha "${ace_manifest_sha}" \
    --arg phantom_manifest_sha "${phantom_manifest_sha}" \
    '.ace_commit == $ace_commit and .phantom_commit == $phantom_commit
     and .source_snapshots.ace_manifest_sha256 == $ace_manifest_sha
     and .source_snapshots.phantom_manifest_sha256 == $phantom_manifest_sha' \
    "${INPUT}/payload.json" >/dev/null
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

configure_qualification_environment() {
  local ace_commit source_manifest_sha provider_manifest_sha bootstrap_sha
  ace_commit="$(jq -er .commit "${INPUT}/ace-source.manifest.json")"
  export SOURCE_DATE_EPOCH
  SOURCE_DATE_EPOCH="$(jq -er .commit_timestamp "${INPUT}/ace-source.manifest.json")"
  source_manifest_sha="$(sha256sum "${INPUT}/ace-source.manifest.json" | awk '{print $1}')"
  provider_manifest_sha="$(sha256sum "${INPUT}/phantom-source.manifest.json" | awk '{print $1}')"
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
  export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST="${INPUT}/phantom-source.manifest.json"
  export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256="${provider_manifest_sha}"
  export ACE_RUNPOD_BOOTSTRAP_SHA256="${bootstrap_sha}"
  export ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
}

capture_failed_qualification() {
  local current="$1"
  local qualification_exit="$2"
  local runs_root run_root canonical_runs_root canonical_run_root
  local current_gate current_status current_exit run_id file_count sums_sha

  if [[ ! -s "${current}" ]]; then
    echo "failed qualification did not publish its current-run record" >&2
    return 1
  fi
  cp -- "${current}" "${RESULT_DIR}/qualification-current.json"
  current_gate="$(jq -er '.gate' "${current}")"
  current_status="$(jq -er '.status' "${current}")"
  current_exit="$(jq -er '.exit_code' "${current}")"
  run_id="$(jq -er '.run_id' "${current}")"
  run_root="$(jq -er '.run_root' "${current}")"
  if [[ "${current_gate}" != "ordinary" ||
        "${current_status}" != "failed" ||
        "${current_exit}" -ne "${qualification_exit}" ]]; then
    echo "failed qualification current-run record disagrees with its process exit" >&2
    return 1
  fi

  runs_root="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/runs"
  canonical_runs_root="$(realpath -- "${runs_root}")"
  canonical_run_root="$(realpath -- "${run_root}")"
  case "${canonical_run_root}" in
    "${canonical_runs_root}"/*) ;;
    *)
      echo "failed qualification run escaped its state-root runs directory" >&2
      return 1
      ;;
  esac
  if [[ "${canonical_run_root##*/}" != "${run_id}" ||
        ! -d "${canonical_run_root}" ]]; then
    echo "failed qualification run identity is invalid" >&2
    return 1
  fi

  cp -a -- "${canonical_run_root}" "${RESULT_DIR}/qualification"
  (
    cd "${RESULT_DIR}"
    find qualification -type f -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum >qualification-failure-files.sha256
  )
  file_count="$(wc -l <"${RESULT_DIR}/qualification-failure-files.sha256" | tr -d ' ')"
  sums_sha="$(sha256sum "${RESULT_DIR}/qualification-failure-files.sha256" | awk '{print $1}')"
  jq -n \
    --arg status captured \
    --arg run_id "${run_id}" \
    --arg evidence_path qualification \
    --arg sha256_manifest qualification-failure-files.sha256 \
    --arg sha256_manifest_sha256 "${sums_sha}" \
    --argjson qualification_exit_code "${qualification_exit}" \
    --argjson captured_file_count "${file_count}" \
    '{schema_version:"ace.phantom.failed-qualification-evidence/1.0.0",
      status:$status, run_id:$run_id,
      qualification_exit_code:$qualification_exit_code,
      evidence_path:$evidence_path,
      captured_file_count:$captured_file_count,
      sha256_manifest:$sha256_manifest,
      sha256_manifest_sha256:$sha256_manifest_sha256}' \
    >"${RESULT_DIR}/qualification-failure-evidence.json"
}

run_qualification() {
  local -a invocation_validation_arguments=(
    "${INPUT}/qualification-invocation.json"
    "${INPUT}/payload.json"
  )
  if [[ "${MODE}" != "freeze-host" ]]; then
    invocation_validation_arguments+=("${INPUT}/compiler-invocation.json")
  fi
  python3 - "${invocation_validation_arguments[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

invocation_path = Path(sys.argv[1])
payload_path = Path(sys.argv[2])
compiler_invocation_path = Path(sys.argv[3]) if len(sys.argv) == 4 else None
def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in packaged evidence: {key}")
        value[key] = item
    return value

invocation = json.loads(
    invocation_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
payload = json.loads(
    payload_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
if not isinstance(payload, dict):
    raise SystemExit("packaged payload must be a JSON object")
if compiler_invocation_path is None:
    if set(payload) != {
        "schema_version", "ace_commit", "phantom_commit", "source_snapshots",
        "qualification_invocation", "contents", "files",
    }:
        raise SystemExit("host-freeze payload has an invalid shape")
    if (
        payload["schema_version"]
        != "ace.phantom.host-freeze-payload/1.0.0"
        or payload["contents"]
        != "audited-source-snapshots-and-qualification-invocation-only"
    ):
        raise SystemExit("host-freeze payload has an invalid schema or contents")
if not isinstance(invocation, dict) or set(invocation) != {
    "schema_version", "argv", "normalized_argv_sha256"
}:
    raise SystemExit("packaged qualification invocation has an invalid shape")
if invocation["schema_version"] != "ace.phantom.qualification-invocation/1.0.0":
    raise SystemExit("packaged qualification invocation has an unsupported schema")
argv = invocation["argv"]
if not isinstance(argv, list) or not all(
    isinstance(value, str) and value for value in argv
):
    raise SystemExit("packaged qualification invocation argv must be a nonempty string array")
normalized = json.dumps(
    argv, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")
normalized_sha = hashlib.sha256(normalized).hexdigest()
record_sha = hashlib.sha256(invocation_path.read_bytes()).hexdigest()
if normalized_sha != invocation["normalized_argv_sha256"]:
    raise SystemExit("packaged qualification invocation normalized hash mismatch")
payload_invocation = payload.get("qualification_invocation", {})
if payload_invocation != {
    "sha256": record_sha,
    "normalized_argv_sha256": normalized_sha,
}:
    raise SystemExit("payload does not bind the packaged qualification invocation")
if not argv or argv[0] != "tools/phantom_gpu/compile_only.sh":
    raise SystemExit("packaged compiler invocation has an unexpected executable")
arguments = argv[1:]
if len(arguments) % 2:
    raise SystemExit("packaged compiler invocation has an argument without a value")
pairs = dict(zip(arguments[0::2], arguments[1::2]))
expected_options = {
    "--gate", "--poly-degree", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight",
}
if len(pairs) != len(arguments) // 2 or set(pairs) != expected_options:
    raise SystemExit("packaged compiler invocation options are incomplete or duplicated")
if pairs["--gate"] != "ordinary":
    raise SystemExit("packaged compiler invocation is not the ordinary qualification")
for option in expected_options - {"--gate"}:
    if re.fullmatch(r"[0-9]+", pairs[option]) is None:
        raise SystemExit(f"packaged compiler invocation {option} is not an integer")

if compiler_invocation_path is not None:
    compiler_invocation = json.loads(
        compiler_invocation_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(compiler_invocation, dict):
        raise SystemExit("packaged compiler invocation must be a JSON object")
    compiler_argv = compiler_invocation.get("argv")
    compiler_normalized = json.dumps(
        compiler_argv, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    compiler_normalized_sha = hashlib.sha256(compiler_normalized).hexdigest()
    if (
        set(compiler_invocation)
        != {"schema_version", "argv", "normalized_argv_sha256"}
        or compiler_invocation.get("schema_version")
        != "ace.phantom.compiler-invocation/1.0.0"
        or compiler_invocation.get("normalized_argv_sha256")
        != compiler_normalized_sha
        or not isinstance(compiler_argv, list)
        or not compiler_argv
        or compiler_argv[0] != "tools/phantom_gpu/generate_ckks2c_probe.py"
    ):
        raise SystemExit("packaged compiler invocation is invalid")
    if payload.get("compiler_invocation") != {
        "sha256": hashlib.sha256(compiler_invocation_path.read_bytes()).hexdigest(),
        "normalized_argv_sha256": compiler_normalized_sha,
    }:
        raise SystemExit("payload does not bind the packaged compiler invocation")
PY
  local qualification_arguments_output
  qualification_arguments_output="$(
    jq -er '.argv[1:][]' "${INPUT}/qualification-invocation.json"
  )"
  local -a qualification_arguments
  mapfile -t qualification_arguments <<<"${qualification_arguments_output}"
  local -a qualification_status
  local qualification_exit
  set +e
  bash "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compile_only.sh" \
    "${qualification_arguments[@]}" \
    2>&1 | tee "${RESULT_DIR}/qualification.log"
  qualification_status=("${PIPESTATUS[@]}")
  set -e
  qualification_exit="${qualification_status[0]}"
  local current run_root
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json"
  if [[ ${qualification_exit} -ne 0 ]]; then
    if ! capture_failed_qualification "${current}" "${qualification_exit}"; then
      echo "failed qualification evidence capture was incomplete" >&2
    fi
    return "${qualification_exit}"
  fi
  if [[ "${qualification_status[1]}" -ne 0 ]]; then
    echo "qualification log capture failed with exit ${qualification_status[1]}" >&2
    return "${qualification_status[1]}"
  fi
  run_root="$(jq -er '.run_root' "${current}")"
  [[ "$(jq -er '.status' "${current}")" == pass ]]
  cmp "${run_root}/qualification_invocation.json" \
    "${INPUT}/qualification-invocation.json"
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
  local dependency_git_dirs="${WORK}/public-dependency-git-dirs.list"
  find "${ACE_PHANTOM_STATE_ROOT}" -type d -name .git -print0 |
    LC_ALL=C sort -z >"${dependency_git_dirs}"
  while IFS= read -r -d '' git_dir; do
    local checkout checkout_commit
    checkout="${git_dir%/.git}"
    checkout_commit="$(git -C "${checkout}" rev-parse HEAD)"
    printf '%s\t%s\n' "${checkout#${WORK}/}" "${checkout_commit}" \
      >>"${RESULT_DIR}/public-dependency-commits.txt"
  done <"${dependency_git_dirs}"
  rm "${dependency_git_dirs}"
  if [[ "${MODE}" == "freeze-host" ]]; then
    local qualification_manifest_sha qualification_sums_sha
    qualification_manifest_sha="$(sha256sum "${run_root}/manifest.json" | awk '{print $1}')"
    qualification_sums_sha="$(sha256sum "${run_root}/SHA256SUMS" | awk '{print $1}')"
    jq -n \
      --arg status candidate \
      --arg ace_commit "$(jq -er .commit "${INPUT}/ace-source.manifest.json")" \
      --arg phantom_commit "$(jq -er .commit "${INPUT}/phantom-source.manifest.json")" \
      --arg qualification_manifest_sha256 "${qualification_manifest_sha}" \
      --arg qualification_sha256_manifest_sha256 "${qualification_sums_sha}" \
      '{schema_version:"ace.phantom.host-freeze-candidate/1.0.0",
        status:$status, evidence_path:"qualification",
        ace_commit:$ace_commit, phantom_commit:$phantom_commit,
        qualification_manifest_sha256:$qualification_manifest_sha256,
        qualification_sha256_manifest_sha256:$qualification_sha256_manifest_sha256}' \
      >"${RESULT_DIR}/host-freeze-candidate.json"
  fi
}

verify_frozen_ordinary_reference() {
  local run_root ordinary_dir generated_context generated_resources
  local context_sha resources_sha fixture_sha cpu_reference_sha cpu_values_sha
  local ant_verification_sha expected_case_count
  run_root="$(jq -er '.run_root' \
    "${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json")"
  ordinary_dir="${run_root}/ordinary_ckks"
  generated_context="${run_root}/ckks2c/compiler_context_manifest.json"
  generated_resources="${run_root}/ckks2c/compiler_resource_manifest.json"
  cmp "${run_root}/qualification_invocation.json" \
    "${INPUT}/qualification-invocation.json"
  cmp "${run_root}/compiler_invocation.json" \
    "${INPUT}/compiler-invocation.json"
  cmp "${generated_context}" "${INPUT}/ordinary-context-manifest.json"
  cmp "${generated_resources}" "${INPUT}/ordinary-resource-manifest.json"
  cmp "${run_root}/ckks2c/ordinary_ckks_post_ckks.air" \
    "${INPUT}/ordinary-post-ckks.air"
  cmp "${ordinary_dir}/ordinary_ckks_v1.json" "${INPUT}/ordinary-fixture.json"
  local relative expected_sha actual_sha
  for relative in \
    ckks2c/add_mul_rotate.cu \
    ckks2c/ordinary_runtime_symbols.cu \
    ordinary_ckks/ordinary_ckks_keyless_probe.cu \
    ordinary_ckks/ordinary_ckks_analytic_reference.json \
    ordinary_ckks/ordinary_ckks_analytic_values.bin; do
    expected_sha="$(jq -er --arg path "${relative}" \
      '.files[$path] | select(type == "string")' \
      "${INPUT}/ordinary-artifact-manifest.json")"
    actual_sha="$(sha256sum "${run_root}/${relative}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      echo "regenerated production source differs from frozen evidence: ${relative}" >&2
      return 1
    fi
  done

  context_sha="$(sha256sum "${INPUT}/ordinary-context-manifest.json" | awk '{print $1}')"
  resources_sha="$(sha256sum "${INPUT}/ordinary-resource-manifest.json" | awk '{print $1}')"
  fixture_sha="$(sha256sum "${INPUT}/ordinary-fixture.json" | awk '{print $1}')"
  cpu_reference_sha="$(sha256sum "${INPUT}/ordinary-cpu-reference.json" | awk '{print $1}')"
  cpu_values_sha="$(sha256sum "${INPUT}/ordinary-cpu-values.bin" | awk '{print $1}')"
  ant_verification_sha="$(sha256sum "${INPUT}/ordinary-ant-verification.json" | awk '{print $1}')"
  local post_ckks_air_sha artifact_manifest_sha run_manifest_sha
  local host_qualification_sha
  post_ckks_air_sha="$(sha256sum "${INPUT}/ordinary-post-ckks.air" | awk '{print $1}')"
  artifact_manifest_sha="$(sha256sum "${INPUT}/ordinary-artifact-manifest.json" | awk '{print $1}')"
  run_manifest_sha="$(sha256sum "${INPUT}/ordinary-run-manifest.json" | awk '{print $1}')"
  host_qualification_sha="$(sha256sum "${INPUT}/ordinary-host-qualification.json" | awk '{print $1}')"
  expected_case_count="$(jq -er \
    '[.case_families[].aliases | length] | add | select(. > 0)' \
    "${INPUT}/ordinary-fixture.json")"
  jq -e \
    --arg context_sha "${context_sha}" \
    --arg resources_sha "${resources_sha}" \
    --arg fixture_sha "${fixture_sha}" \
    --arg cpu_reference_sha "${cpu_reference_sha}" \
    --arg cpu_values_sha "${cpu_values_sha}" \
    --arg ant_verification_sha "${ant_verification_sha}" \
    --arg post_ckks_air_sha "${post_ckks_air_sha}" \
    --arg artifact_manifest_sha "${artifact_manifest_sha}" \
    --arg run_manifest_sha "${run_manifest_sha}" \
    --arg host_qualification_sha "${host_qualification_sha}" \
    '.frozen_ordinary_reference.context_manifest_sha256 == $context_sha
     and .frozen_ordinary_reference.resource_manifest_sha256 == $resources_sha
     and .frozen_ordinary_reference.fixture_sha256 == $fixture_sha
     and .frozen_ordinary_reference.cpu_reference_sha256 == $cpu_reference_sha
     and .frozen_ordinary_reference.cpu_values_sha256 == $cpu_values_sha
     and .frozen_ordinary_reference.ant_verification_sha256 == $ant_verification_sha
     and .frozen_ordinary_reference.post_ckks_air_sha256 == $post_ckks_air_sha
     and .frozen_ordinary_reference.artifact_manifest_sha256 == $artifact_manifest_sha
     and .frozen_ordinary_reference.run_manifest_sha256 == $run_manifest_sha
     and .frozen_ordinary_reference.host_qualification_sha256 == $host_qualification_sha' \
    "${INPUT}/payload.json" >/dev/null
  jq -e --argjson expected_case_count "${expected_case_count}" \
    '.status == "pass" and .case_count == $expected_case_count' \
    "${INPUT}/ordinary-ant-verification.json" >/dev/null
  jq -e --argjson expected_case_count "${expected_case_count}" \
    '.status == "pass" and .case_count == $expected_case_count' \
    "${ordinary_dir}/ordinary_ckks_ant_verification.json" >/dev/null
  python3 "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/ordinary_ckks_fixture.py" \
    verify-ant-reference \
    --fixture "${INPUT}/ordinary-fixture.json" \
    --context-manifest "${INPUT}/ordinary-context-manifest.json" \
    --cpu-json "${INPUT}/ordinary-cpu-reference.json" \
    --cpu-bin "${INPUT}/ordinary-cpu-values.bin" \
    --output-json "${RESULT_DIR}/ordinary-frozen-ant-verification.json"
  jq -e --argjson expected_case_count "${expected_case_count}" \
    '.status == "pass" and .case_count == $expected_case_count' \
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
  local gpu_results gpu_values comparison expected_case_count
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
  expected_case_count="$(jq -er \
    '[.case_families[].aliases | length] | add | select(. > 0)' \
    "${fixture}")"
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
  jq -e --argjson expected_case_count "${expected_case_count}" \
    '.status == "pass"
     and .case_count == $expected_case_count
     and .passing_case_count == $expected_case_count' \
    "${comparison}" >/dev/null

  local diagnostics="${RESULT_DIR}/ordinary_ckks_diagnostics.jsonl"
  local runtime_rejections="${RESULT_DIR}/ordinary_runtime_rejections.tsv"
  : >"${diagnostics}"
  jq -er '
    .rejections[]
    | select(.phase == "runtime")
    | [.id, .diagnostic_id, .provider_diagnostic, .runner_profile]
    | @tsv
  ' "${fixture}" >"${runtime_rejections}"
  local case_id diagnostic_id expected_token runner_profile selected_runner
  local exit_code stdout_file stderr_file
  while IFS=$'\t' read -r case_id diagnostic_id expected_token runner_profile; do
    case "${runner_profile}" in
      ordinary) selected_runner="${runner}" ;;
      keyless) selected_runner="${keyless_runner}" ;;
      *)
        echo "unsupported runtime rejection runner profile ${runner_profile}" >&2
        return 1
        ;;
    esac
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
    jq -cn --arg case_id "${case_id}" --arg diagnostic_id "${diagnostic_id}" \
      --arg provider_diagnostic "${expected_token}" \
      --argjson exit_code "${exit_code}" \
      '{case_id:$case_id, diagnostic_id:$diagnostic_id,
        provider_diagnostic:$provider_diagnostic,
        exit_code:$exit_code, status:"pass"}' \
      >>"${diagnostics}"
  done <"${runtime_rejections}"
  jq -s '{schema_version:"1.0.0", status:"pass", cases:.}' \
    "${diagnostics}" >"${RESULT_DIR}/ordinary_ckks_diagnostics.json"
  jq -e --slurpfile fixture "${fixture}" '
    .status == "pass"
    and ([.cases[]
          | {case_id, diagnostic_id, provider_diagnostic}]
         == [$fixture[0].rejections[]
             | select(.phase == "runtime")
             | {case_id:.id, diagnostic_id, provider_diagnostic}])
  ' "${RESULT_DIR}/ordinary_ckks_diagnostics.json" >/dev/null
  rm "${diagnostics}"

  local sanitizer_stdout="${RESULT_DIR}/ordinary_ckks_sanitizer.stdout.txt"
  local sanitizer_stderr="${RESULT_DIR}/ordinary_ckks_sanitizer.stderr.txt"
  local sanitizer_log="${RESULT_DIR}/ordinary_ckks_sanitizer.log.txt"
  local sanitizer_version="${RESULT_DIR}/ordinary_ckks_sanitizer.version.txt"
  local ownership_json="${RESULT_DIR}/ordinary_ckks_ownership.json"
  compute-sanitizer --version >"${sanitizer_version}" 2>&1
  [[ -s "${sanitizer_version}" ]]
  rm -f "${sanitizer_log}" "${ownership_json}"
  set +e
  timeout 900 compute-sanitizer --tool memcheck --error-exitcode=99 \
    --log-file "${sanitizer_log}" \
    "${runner}" ownership "${fixture}" "${expected_gpu}" \
    "${ownership_json}" >"${sanitizer_stdout}" 2>"${sanitizer_stderr}"
  local sanitizer_exit=$?
  set -e
  if [[ ${sanitizer_exit} -ne 0 ]]; then
    echo "Compute Sanitizer exited ${sanitizer_exit}" >&2
    return "${sanitizer_exit}"
  fi
  local zero_summary_count total_summary_count
  zero_summary_count="$(
    rg -c '^========= ERROR SUMMARY: 0 errors$' "${sanitizer_log}" || true
  )"
  total_summary_count="$(
    rg -c '^========= ERROR SUMMARY:' "${sanitizer_log}" || true
  )"
  if [[ "${zero_summary_count:-0}" != 1 ||
        "${total_summary_count:-0}" != 1 ]]; then
    echo "Compute Sanitizer log does not contain one exact zero-error summary" >&2
    return 1
  fi
  local ownership_validation ownership_iterations
  ownership_validation="$(
    python3 \
      "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/ordinary_ckks_fixture.py" \
      validate-ownership --fixture "${fixture}" \
      --ownership-json "${ownership_json}"
  )"
  printf '%s\n' "${ownership_validation}" \
    >"${RESULT_DIR}/ordinary_ckks_ownership.validation.json"
  ownership_iterations="$(
    jq -er '.total_iterations | select(. > 0)' <<<"${ownership_validation}"
  )"
  local sanitizer_log_sha256 sanitizer_version_sha256
  sanitizer_log_sha256="$(sha256sum "${sanitizer_log}" | awk '{print $1}')"
  sanitizer_version_sha256="$(
    sha256sum "${sanitizer_version}" | awk '{print $1}'
  )"
  jq -n --arg status pass --arg tool memcheck --argjson exit_code 0 \
    --argjson ownership_iterations "${ownership_iterations}" \
    --arg sanitizer_log_sha256 "${sanitizer_log_sha256}" \
    --arg sanitizer_version_sha256 "${sanitizer_version_sha256}" \
    '{schema_version:"1.0.0", status:$status, tool:$tool,
      error_summary:0, exit_code:$exit_code,
      ownership_iterations:$ownership_iterations,
      sanitizer_log_sha256:$sanitizer_log_sha256,
      sanitizer_version_sha256:$sanitizer_version_sha256}' \
    >"${RESULT_DIR}/ordinary_ckks_sanitizer.json"
  rm "${raw_results}"
}

run_native_health() {
  local expected_gpu query gpu_count gpu_name current run_root health_binary
  local context_manifest context_sha health_raw health_exit
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
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-ordinary.json"
  [[ "$(jq -er '.gate + ":" + .status' "${current}")" == "ordinary:pass" ]]
  run_root="$(jq -er '.run_root' "${current}")"
  health_binary="${run_root}/ckks2c/native_phantom_health_sm80"
  context_manifest="${run_root}/ckks2c/compiler_context_manifest.json"
  [[ -x "${health_binary}" ]]
  [[ -s "${context_manifest}" ]]
  context_sha="$(sha256sum "${context_manifest}" | awk '{print $1}')"
  health_raw="${RESULT_DIR}/native-health.raw.json"
  set +e
  timeout 300 "${health_binary}" "${health_raw}" \
    >"${RESULT_DIR}/native-health.stdout.txt" \
    2>"${RESULT_DIR}/native-health.stderr.txt"
  health_exit=$?
  set -e
  [[ ${health_exit} -eq 0 ]]
  jq -e --arg context_sha "${context_sha}" --arg expected_gpu "${expected_gpu}" \
    'select(.status == "pass" and .device_count == 1
     and .gpu == $expected_gpu
     and (.max_error | type == "number") and .max_error <= 0.0001
     and .context_manifest_sha256 == $context_sha)' "${health_raw}" \
    >"${RESULT_DIR}/native-health.json"
}

verify_success_evidence() {
  jq -e '
    .gate == "ordinary" and .status == "pass" and .exit_code == 0
  ' "${RESULT_DIR}/qualification-current.json" >/dev/null
  case "${MODE}" in
    freeze-host)
      jq -e '
        .schema_version == "ace.phantom.host-freeze-candidate/1.0.0"
        and .status == "candidate"
      ' "${RESULT_DIR}/host-freeze-candidate.json" >/dev/null
      ;;
    local)
      jq -e '.status == "pass"' \
        "${RESULT_DIR}/ordinary-frozen-reference.json" >/dev/null
      jq -e '.status == "skipped"' \
        "${RESULT_DIR}/native-health.json" >/dev/null
      ;;
    runpod)
      jq -e '.status == "pass" and .device_count == 1' \
        "${RESULT_DIR}/native-health.json" >/dev/null
      jq -e '
        .status == "pass" and .case_count > 0
        and .passing_case_count == .case_count
      ' "${RESULT_DIR}/ordinary_ckks_compare.json" >/dev/null
      jq -e '.status == "pass" and (.cases | length) > 0' \
        "${RESULT_DIR}/ordinary_ckks_diagnostics.json" >/dev/null
      jq -e '
        .schema_version == "ace.phantom.ordinary_ckks.ownership/1.0.0"
        and .status == "pass" and .total_iterations > 0
      ' "${RESULT_DIR}/ordinary_ckks_ownership.json" >/dev/null
      jq -e '.status == "pass" and .exit_code == 0' \
        "${RESULT_DIR}/ordinary_ckks_sanitizer.json" >/dev/null
      ;;
  esac
  jq -n --arg mode "${MODE}" \
    '{schema_version:"ace.phantom.result-completeness/1.0.0",
      status:"pass", mode:$mode}' \
    >"${RESULT_DIR}/result-completeness.json"
}

phase payload_verification verify_outer_payload
phase environment_bootstrap bootstrap
export PATH="/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
phase source_audit_and_extraction extract_sources
CURRENT_PHASE=qualification_environment
configure_qualification_environment
CURRENT_PHASE=""
phase qualification run_qualification
if [[ "${MODE}" != "freeze-host" ]]; then
  phase frozen_ordinary_reference verify_frozen_ordinary_reference
  if [[ "${MODE}" == "runpod" ]]; then
    phase native_a100_health run_native_health
    phase ordinary_gpu_qualification run_ordinary_gpu_qualification
  else
    printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
      >"${RESULT_DIR}/native-health.json"
  fi
fi
phase result_completeness verify_success_evidence
PIPELINE_EXIT=0
