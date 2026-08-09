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
RETAINED_HOST_ROOT="${WORK}/retained-host-qualification"
RETAINED_HOST_WORK="${WORK}/retained-host-build"
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
  export ACE_PHANTOM_STATE_ROOT="${WORK}/state"
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
validate_packaged_bootstrap_qualification() {
  python3 - "${INPUT}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in bootstrap package: {key}")
        value[key] = item
    return value

def load(name):
    path = root / name
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    if not isinstance(value, dict):
        raise SystemExit(f"bootstrap package record is not an object: {name}")
    return value

def digest(name):
    return hashlib.sha256((root / name).read_bytes()).hexdigest()

payload = load("payload.json")
binding = payload.get("bootstrap_qualification")
if not isinstance(binding, dict):
    raise SystemExit("payload lacks its bootstrap qualification binding")
bound_files = {
    "qualification_invocation_sha256": "bootstrap-qualification-invocation.json",
    "generation_invocation_sha256": "bootstrap-generation-invocation.json",
    "artifact_manifest_sha256": "bootstrap-artifact-manifest.json",
    "run_manifest_sha256": "bootstrap-run-manifest.json",
    "context_manifest_sha256": "bootstrap-context-manifest.json",
    "resource_manifest_sha256": "bootstrap-resource-manifest.json",
    "constant_manifest_sha256": "bootstrap-constant-manifest.json",
    "generation_record_sha256": "bootstrap-generation.json",
    "source_audit_sha256": "bootstrap-source-audit.json",
    "host_qualification_sha256": "bootstrap-host-qualification.json",
}
expected_binding_keys = set(bound_files) | {
    "normalized_qualification_argv_sha256",
    "normalized_generation_argv_sha256",
    "expected_generated_source_sha256",
    "expected_linked_binary_sha256",
    "expected_harness_source_sha256",
    "packaged_build_output",
}
if set(binding) != expected_binding_keys:
    raise SystemExit("payload bootstrap qualification binding has an invalid shape")
for field, name in bound_files.items():
    if binding.get(field) != digest(name):
        raise SystemExit(f"payload bootstrap binding is stale: {field}")
if binding.get("packaged_build_output") is not False:
    raise SystemExit("bootstrap payload must not contain packaged build output")

expected_parameters = [
    "--poly-degree", "16384", "--mul-level", "26",
    "--input-level", "1", "--security-level", "0",
    "--scaling-factor-bits", "56", "--first-prime-bits", "60",
    "--hamming-weight", "192",
]

def validate_invocation(name, schema, expected):
    invocation = load(name)
    if set(invocation) != {"schema_version", "argv", "normalized_argv_sha256"}:
        raise SystemExit(f"bootstrap invocation has an invalid shape: {name}")
    encoded = json.dumps(
        invocation["argv"], ensure_ascii=False, separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if (
        invocation["schema_version"] != schema
        or invocation["argv"] != expected
        or invocation["normalized_argv_sha256"]
        != hashlib.sha256(encoded).hexdigest()
    ):
        raise SystemExit(f"bootstrap invocation is not the exact frozen contract: {name}")
    return invocation

qualification_invocation = validate_invocation(
    "bootstrap-qualification-invocation.json",
    "ace.phantom.qualification-invocation/1.0.0",
    ["tools/phantom_gpu/compile_only.sh", "--gate", "bootstrap"]
    + expected_parameters,
)
generation_invocation = validate_invocation(
    "bootstrap-generation-invocation.json",
    "ace.phantom.bootstrap-qualification-invocation/1.0.0",
    ["tools/phantom_gpu/generate_bootstrap_qualification.py",
     "bootstrap_qualification"] + expected_parameters,
)
if (
    binding.get("normalized_qualification_argv_sha256")
    != qualification_invocation["normalized_argv_sha256"]
    or binding.get("normalized_generation_argv_sha256")
    != generation_invocation["normalized_argv_sha256"]
):
    raise SystemExit("payload normalized bootstrap invocation binding is stale")

context = load("bootstrap-context-manifest.json")
context_keys = {
    "data_q_bit_sizes", "first_modulus_bits", "hamming_weight", "input_level",
    "logical_slot_capacity", "packing", "polynomial_degree", "q_part_count",
    "resource_schema_version", "scaling_modulus_bits", "schema_version",
    "security_level", "special_p_bit_sizes",
}
if (
    set(context) != context_keys
    or context.get("schema_version") != 1
    or context.get("resource_schema_version") != 3
    or context.get("polynomial_degree") != 16384
    or context.get("logical_slot_capacity") != 8192
    or context.get("packing") != "full"
    or not isinstance(context.get("data_q_bit_sizes"), list)
    or len(context["data_q_bit_sizes"]) != 26
    or context.get("input_level") != 1
    or context.get("security_level") != 0
    or context.get("scaling_modulus_bits") != 56
    or context.get("first_modulus_bits") != 60
    or context.get("hamming_weight") != 192
):
    raise SystemExit("packaged bootstrap context is not the exact schema-v3 contract")
resources = load("bootstrap-resource-manifest.json")
resource_keys = {
    "schema_version", "context_schema_version", "relinearization_key",
    "rotation_steps", "conjugation_key", "rotate_batch", "rotation_batches",
    "raise_mod", "monomial_powers", "complex_plaintext",
    "native_bootstrap_precompute",
}
if (
    set(resources) != resource_keys
    or resources.get("schema_version") != 3
    or resources.get("context_schema_version") != 1
    or any(resources.get(field) is not True for field in (
        "relinearization_key", "conjugation_key", "rotate_batch", "raise_mod",
        "complex_plaintext",
    ))
    or resources.get("native_bootstrap_precompute") is not False
    or not resources.get("rotation_steps")
    or not resources.get("rotation_batches")
    or not resources.get("monomial_powers")
):
    raise SystemExit("packaged bootstrap resources are not the full primitive set")
constants = load("bootstrap-constant-manifest.json")
if (
    set(constants) != {"constants", "context_manifest_sha256",
                       "context_schema_version", "resource_schema_version",
                       "schema_version"}
    or constants.get("schema_version") != 1
    or constants.get("context_schema_version") != 1
    or constants.get("resource_schema_version") != 3
    or constants.get("context_manifest_sha256")
       != digest("bootstrap-context-manifest.json")
    or not isinstance(constants.get("constants"), list)
    or not constants["constants"]
):
    raise SystemExit("packaged bootstrap constants are empty or not context-bound")
for index, entry in enumerate(constants["constants"]):
    entry_keys = {
        "ace_level", "cache_key_sha256", "chain_index", "constant_id",
        "element_type", "entry_id", "payload_sha256", "raw_scale",
        "scale_degree", "slot_count", "symbol",
    }
    if (
        not isinstance(entry, dict)
        or set(entry) != entry_keys
        or entry.get("entry_id") != index
        or entry.get("element_type") != "complex_f64"
        or not isinstance(entry.get("slot_count"), int)
        or isinstance(entry.get("slot_count"), bool)
        or entry["slot_count"] <= 0
        or not isinstance(entry.get("ace_level"), int)
        or isinstance(entry.get("ace_level"), bool)
        or entry["ace_level"] <= 0
        or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("payload_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("cache_key_sha256"))) is None
    ):
        raise SystemExit(f"packaged bootstrap constant entry is invalid: {index}")

generation = load("bootstrap-generation.json")
audit = load("bootstrap-source-audit.json")
artifact = load("bootstrap-artifact-manifest.json")
qualification = load("bootstrap-host-qualification.json")
generated_source_sha = artifact.get("files", {}).get(
    "bootstrap_qualification/bootstrap_qualification.cu"
)
if (
    generation.get("schema_version") != "ace.phantom.bootstrap-generation/1.0.0"
    or generation.get("status") != "pass"
    or generation.get("constant_count") != len(constants["constants"])
    or audit.get("status") != "pass"
    or audit.get("counts", {}).get("constants") != len(constants["constants"])
    or artifact.get("schema_version") != "ace.phantom.bootstrap-artifacts/1.0.0"
    or artifact.get("status") != "bound"
    or qualification.get("status") != "pass"
    or qualification.get("gate") != "bootstrap"
    or qualification.get("executable_was_run") is not False
):
    raise SystemExit("packaged bootstrap generation/audit/host records are invalid")
for label, name in (
    ("context", "bootstrap-context-manifest.json"),
    ("resource", "bootstrap-resource-manifest.json"),
    ("constant", "bootstrap-constant-manifest.json"),
):
    manifest_record = generation.get("manifests", {}).get(label, {})
    if manifest_record.get("sha256") != digest(name):
        raise SystemExit(f"packaged bootstrap generation lacks {label} binding")
audit_inputs = audit.get("inputs", {})
for label, name in (
    ("context_manifest", "bootstrap-context-manifest.json"),
    ("resource_manifest", "bootstrap-resource-manifest.json"),
    ("constant_manifest", "bootstrap-constant-manifest.json"),
):
    if audit_inputs.get(label, {}).get("sha256") != digest(name):
        raise SystemExit(f"packaged bootstrap audit lacks {label} binding")
if (
    generation.get("source", {}).get("sha256") != generated_source_sha
    or audit_inputs.get("source", {}).get("sha256") != generated_source_sha
):
    raise SystemExit("packaged bootstrap generation/audit source binding is stale")
qualification_hashes = {
    "generated_source_sha256": generated_source_sha,
    "linked_binary_sha256": artifact.get("linked_binary_sha256"),
    "harness_source_sha256": artifact.get("harness_source_sha256"),
    "compiler_context_manifest_sha256": digest("bootstrap-context-manifest.json"),
    "compiler_resource_manifest_sha256": digest("bootstrap-resource-manifest.json"),
    "compiler_constant_manifest_sha256": digest("bootstrap-constant-manifest.json"),
    "generation_record_sha256": digest("bootstrap-generation.json"),
    "generated_artifact_audit_sha256": digest("bootstrap-source-audit.json"),
}
if any(qualification.get(field) != value for field, value in qualification_hashes.items()):
    raise SystemExit("packaged bootstrap host qualification hashes are stale")
if (
    binding.get("expected_generated_source_sha256") != generated_source_sha
    or binding.get("expected_linked_binary_sha256")
       != artifact.get("linked_binary_sha256")
    or binding.get("expected_harness_source_sha256")
       != artifact.get("harness_source_sha256")
    or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in (
        binding.get("expected_generated_source_sha256"),
        binding.get("expected_linked_binary_sha256"),
        binding.get("expected_harness_source_sha256"),
    ))
):
    raise SystemExit("packaged bootstrap artifact hashes are incomplete")
PY
}

capture_failed_bootstrap_qualification() {
  local current="$1"
  local qualification_exit="$2"
  local run_root run_id current_exit canonical_runs_root canonical_run_root
  [[ -s "${current}" ]]
  cp -- "${current}" "${RESULT_DIR}/bootstrap-qualification-current.json"
  [[ "$(jq -er '.gate + ":" + .status' "${current}")" == "bootstrap:failed" ]]
  current_exit="$(jq -er .exit_code "${current}")"
  [[ "${current_exit}" -eq "${qualification_exit}" ]]
  run_id="$(jq -er .run_id "${current}")"
  run_root="$(jq -er .run_root "${current}")"
  canonical_runs_root="$(realpath -- \
    "${ACE_PHANTOM_STATE_ROOT}/compile_only_results/runs")"
  canonical_run_root="$(realpath -- "${run_root}")"
  case "${canonical_run_root}" in
    "${canonical_runs_root}"/*) ;;
    *) return 1 ;;
  esac
  [[ "${canonical_run_root##*/}" == "${run_id}" ]]
  cp -a -- "${canonical_run_root}" \
    "${RESULT_DIR}/bootstrap-qualification"
  (
    cd "${RESULT_DIR}"
    find bootstrap-qualification -type f -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum >bootstrap-qualification-failure-files.sha256
  )
  jq -n --arg status captured --arg run_id "${run_id}" \
    --arg evidence_path bootstrap-qualification \
    --arg sha256_manifest bootstrap-qualification-failure-files.sha256 \
    --argjson qualification_exit_code "${qualification_exit}" \
    '{schema_version:"ace.phantom.failed-bootstrap-qualification-evidence/1.0.0",
      status:$status, run_id:$run_id,
      qualification_exit_code:$qualification_exit_code,
      evidence_path:$evidence_path, sha256_manifest:$sha256_manifest}' \
    >"${RESULT_DIR}/bootstrap-qualification-failure-evidence.json"
}

run_bootstrap_host_qualification() {
  local arguments_output
  arguments_output="$(
    jq -er '.argv[1:][]' "${INPUT}/bootstrap-qualification-invocation.json"
  )"
  local -a arguments
  mapfile -t arguments <<<"${arguments_output}"
  local -a status
  local qualification_exit current run_root
  set +e
  bash "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compile_only.sh" \
    "${arguments[@]}" 2>&1 | tee "${RESULT_DIR}/bootstrap-qualification.log"
  status=("${PIPESTATUS[@]}")
  set -e
  qualification_exit="${status[0]}"
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-bootstrap.json"
  if [[ ${qualification_exit} -ne 0 ]]; then
    capture_failed_bootstrap_qualification "${current}" "${qualification_exit}" ||
      echo "failed bootstrap qualification evidence capture was incomplete" >&2
    return "${qualification_exit}"
  fi
  [[ "${status[1]}" -eq 0 ]]
  [[ "$(jq -er '.gate + ":" + .status' "${current}")" == "bootstrap:pass" ]]
  run_root="$(jq -er .run_root "${current}")"
  cmp "${run_root}/qualification_invocation.json" \
    "${INPUT}/bootstrap-qualification-invocation.json"
  cmp "${run_root}/bootstrap_generation_invocation.json" \
    "${INPUT}/bootstrap-generation-invocation.json"
  cp -a -- "${run_root}" "${RESULT_DIR}/bootstrap-qualification"
  cp -- "${current}" "${RESULT_DIR}/bootstrap-qualification-current.json"
}

verify_frozen_bootstrap_qualification() {
  local current run_root output frozen_artifact generated_artifact
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-bootstrap.json"
  run_root="$(jq -er .run_root "${current}")"
  output="${run_root}/bootstrap_qualification"
  frozen_artifact="${INPUT}/bootstrap-artifact-manifest.json"
  generated_artifact="${run_root}/artifact_manifest.json"
  cmp "${output}/compiler_context_manifest.json" \
    "${INPUT}/bootstrap-context-manifest.json"
  cmp "${output}/compiler_resource_manifest.json" \
    "${INPUT}/bootstrap-resource-manifest.json"
  cmp "${output}/compiler_constant_manifest.json" \
    "${INPUT}/bootstrap-constant-manifest.json"
  cmp "${output}/generation.json" "${INPUT}/bootstrap-generation.json"
  python3 - "${INPUT}/payload.json" "${frozen_artifact}" \
    "${generated_artifact}" "${output}/qualification.json" \
    "${output}/source-audit.json" \
    "${output}/bootstrap_phantom_constants_sm80" \
    "${RESULT_DIR}/bootstrap-frozen-reference.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    payload_path,
    frozen_path,
    generated_path,
    qualification_path,
    audit_path,
    binary_file_path,
    output_path,
) = map(Path, sys.argv[1:])
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
binding = load(payload_path)["bootstrap_qualification"]
frozen = load(frozen_path)
generated = load(generated_path)
qualification = load(qualification_path)
audit = load(audit_path)
stable_paths = (
    "bootstrap_qualification/bootstrap_qualification.cu",
    "bootstrap_qualification/bootstrap_phantom_constants.cu",
    "bootstrap_qualification/compiler_context_manifest.json",
    "bootstrap_qualification/compiler_resource_manifest.json",
    "bootstrap_qualification/compiler_constant_manifest.json",
    "bootstrap_qualification/generation.json",
)
binary_path = "bootstrap_qualification/bootstrap_phantom_constants_sm80"
for relative in stable_paths:
    if generated["files"].get(relative) != frozen["files"].get(relative):
        raise SystemExit(f"regenerated bootstrap artifact differs: {relative}")
packaged_source_sha = binding["expected_generated_source_sha256"]
packaged_harness_sha = binding["expected_harness_source_sha256"]
packaged_binary_sha = binding["expected_linked_binary_sha256"]
regenerated_source_sha = generated["files"][stable_paths[0]]
regenerated_harness_sha = generated["files"][stable_paths[1]]
regenerated_binary_sha = generated["files"][binary_path]
if (
    packaged_source_sha != regenerated_source_sha
    or packaged_harness_sha != regenerated_harness_sha
):
    raise SystemExit("regenerated bootstrap source differs from the packaged host freeze")
for label, value in (
    ("packaged linked binary", packaged_binary_sha),
    ("regenerated linked binary", regenerated_binary_sha),
):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SystemExit(f"{label} hash is invalid")
if digest(binary_file_path) != regenerated_binary_sha:
    raise SystemExit("regenerated linked binary bytes differ from its artifact manifest")
qualification_expected = {
    "generated_source_sha256": regenerated_source_sha,
    "linked_binary_sha256": regenerated_binary_sha,
    "harness_source_sha256": regenerated_harness_sha,
    "compiler_context_manifest_sha256": generated["files"][stable_paths[2]],
    "compiler_resource_manifest_sha256": generated["files"][stable_paths[3]],
    "compiler_constant_manifest_sha256": generated["files"][stable_paths[4]],
    "generation_record_sha256": generated["files"][stable_paths[5]],
}
if any(qualification.get(field) != value for field, value in qualification_expected.items()):
    raise SystemExit("regenerated bootstrap qualification hash binding is stale")
if audit.get("status") != "pass" or not audit.get("counts", {}).get("constants"):
    raise SystemExit("regenerated bootstrap source audit did not pass")
record = {
    "schema_version": "ace.phantom.bootstrap-frozen-reference/1.1.0",
    "status": "pass",
    "comparison": "exact-source-and-setup-with-per-run-cuda-artifacts",
    "source_and_setup_match": True,
    "linked_binary_byte_identity_required": False,
    "packaged_artifact_manifest_sha256": digest(frozen_path),
    "packaged_host_qualification_sha256": binding["host_qualification_sha256"],
    "packaged_source_audit_sha256": binding["source_audit_sha256"],
    "regenerated_artifact_manifest_sha256": digest(generated_path),
    "regenerated_host_qualification_sha256": digest(qualification_path),
    "regenerated_source_audit_sha256": digest(audit_path),
    "packaged_generated_source_sha256": packaged_source_sha,
    "regenerated_generated_source_sha256": regenerated_source_sha,
    "packaged_harness_source_sha256": packaged_harness_sha,
    "regenerated_harness_source_sha256": regenerated_harness_sha,
    "packaged_linked_binary_sha256": packaged_binary_sha,
    "regenerated_linked_binary_sha256": regenerated_binary_sha,
}
output_path.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
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

capture_retained_host_failure() {
  local retained_root="$1"
  local qualification_exit="$2"
  local destination="${RESULT_DIR}/retained-host-failure"
  mkdir -p "${destination}"
  if [[ -d "${retained_root}" ]]; then
    local file_list="${WORK}/retained-host-failure-files.list"
    find "${retained_root}" -type f \
      \( -name '*.json' -o -name '*.log' -o -name '*.txt' \) \
      -size -16M -print0 | LC_ALL=C sort -z >"${file_list}"
    while IFS= read -r -d '' source_file; do
      local relative destination_file
      relative="${source_file#${retained_root}/}"
      destination_file="${destination}/${relative}"
      mkdir -p "$(dirname -- "${destination_file}")"
      cp -- "${source_file}" "${destination_file}"
    done <"${file_list}"
    rm "${file_list}"
  fi
  (
    cd "${RESULT_DIR}"
    find retained-host-failure -type f -print0 | LC_ALL=C sort -z |
      xargs -0 -r sha256sum >retained-host-failure-files.sha256
  )
  jq -n --arg status captured \
    --arg evidence_path retained-host-failure \
    --arg sha256_manifest retained-host-failure-files.sha256 \
    --argjson qualification_exit_code "${qualification_exit}" \
    '{schema_version:"ace.phantom.retained_ckks.failed-host-evidence/1.0.0",
      status:$status, qualification_exit_code:$qualification_exit_code,
      evidence_path:$evidence_path, sha256_manifest:$sha256_manifest}' \
    >"${RESULT_DIR}/retained-host-failure-evidence.json"
}

run_retained_host_qualification() {
  local retained_log="${RESULT_DIR}/retained-host-qualification.log"
  local -a retained_status
  local retained_exit
  set +e
  ACE_RETAINED_CKKS_RESULT_ROOT="${RETAINED_HOST_ROOT}" \
  ACE_RETAINED_CKKS_WORK_ROOT="${RETAINED_HOST_WORK}" \
  ACE_RETAINED_CKKS_PROVISIONAL_BINDING=0 \
    bash \
      "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/run_retained_ckks_correctness.sh" \
      --host-qualify 2>&1 | tee "${retained_log}"
  retained_status=("${PIPESTATUS[@]}")
  set -e
  retained_exit="${retained_status[0]}"
  if [[ ${retained_exit} -ne 0 ]]; then
    capture_retained_host_failure "${RETAINED_HOST_ROOT}" "${retained_exit}" || true
    return "${retained_exit}"
  fi
  if [[ "${retained_status[1]}" -ne 0 ]]; then
    echo "retained host log capture failed with exit ${retained_status[1]}" >&2
    return "${retained_status[1]}"
  fi
  jq -e '
    .schema_version == "ace.phantom.retained_ckks.host-qualification/1.0.0"
    and .status == "pass"
    and .source_mode == "snapshot"
    and .fixture_lifecycle == "checked-bound"
    and .host_ant_oracle_was_run == true
    and .gpu_executables_were_run == false
  ' "${RETAINED_HOST_ROOT}/manifest.json" >/dev/null
  mkdir -p "${RESULT_DIR}/retained-host"
  cp -- "${RETAINED_HOST_ROOT}/manifest.json" \
    "${RESULT_DIR}/retained-host/manifest.json"
  cp -- "${RETAINED_HOST_ROOT}/build_attestation.json" \
    "${RESULT_DIR}/retained-host/build-attestation.json"
  cp -- "${RETAINED_HOST_ROOT}/artifact_manifest.json" \
    "${RESULT_DIR}/retained-host/artifact-manifest.json"
  cp -- "${RETAINED_HOST_ROOT}/tests/pytest-retained-host.txt" \
    "${RESULT_DIR}/retained-host/pytest.txt"
  cp -- "${RETAINED_HOST_ROOT}/tests/ctest-retained-host.txt" \
    "${RESULT_DIR}/retained-host/ctest.txt"
}

verify_frozen_retained_reference() {
  jq -e '
    .contents ==
      "audited-source-snapshots-and-frozen-provider-neutral-references-without-build-output"
    and .frozen_retained_reference.status == "pass"
    and .frozen_retained_reference.contents ==
      "provider-neutral-references-and-attestations-no-build-output"
  ' "${INPUT}/payload.json" >/dev/null
  cmp "${RETAINED_HOST_ROOT}/source/ace_source_manifest.json" \
    "${INPUT}/ace-source.manifest.json"
  cmp "${RETAINED_HOST_ROOT}/source/phantom_source_manifest.json" \
    "${INPUT}/phantom-source.manifest.json"
  python3 \
    "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/retained_runpod_evidence.py" \
    verify-replay \
    --frozen "${INPUT}" \
    --regenerated-root "${RETAINED_HOST_ROOT}" \
    --ace-commit "$(jq -er .commit "${INPUT}/ace-source.manifest.json")" \
    --phantom-commit \
      "$(jq -er .commit "${INPUT}/phantom-source.manifest.json")" \
    >"${RESULT_DIR}/retained-frozen-reference.json"
  jq -e '
    .status == "pass"
    and .provider_neutral_ant_reference_matches == true
    and .exact_artifact_count > 0
  ' "${RESULT_DIR}/retained-frozen-reference.json" >/dev/null
}

run_retained_gpu_qualification() {
  local expected_gpu="${ACE_RUNPOD_EXPECTED_GPU_NAME:-}"
  case "${expected_gpu}" in
    "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB") ;;
    *)
      echo "an exact supported A100 GPU identity is required" >&2
      return 1
      ;;
  esac
  command -v compute-sanitizer >/dev/null
  local fixture="${RETAINED_HOST_ROOT}/inputs/retained_ckks_fixture.json"
  local context="${RETAINED_HOST_ROOT}/inputs/compiler_context_manifest.json"
  local emitted_context="${RETAINED_HOST_ROOT}/outputs/compiler_context_manifest.json"
  local analytic_json="${RETAINED_HOST_ROOT}/outputs/retained_ckks_analytic_reference.json"
  local analytic_bin="${RETAINED_HOST_ROOT}/outputs/retained_ckks_analytic_values.bin"
  local exact_source_json="${RETAINED_HOST_ROOT}/outputs/retained_ckks_exact_source.json"
  local exact_source_bin="${RETAINED_HOST_ROOT}/outputs/retained_ckks_exact_source.bin"
  local ant_json="${INPUT}/retained-ant-reference.json"
  local ant_bin="${INPUT}/retained-ant-values.bin"
  local frozen_build="${INPUT}/retained-host-build-attestation.json"
  local runner="${RETAINED_HOST_ROOT}/build/retained_ckks_phantom_sm80"
  local native_test="${RETAINED_HOST_ROOT}/build/retained_ckks_native_primitives_sm80"
  local conjugation_keyless="${RETAINED_HOST_ROOT}/build/retained_ckks_conjugation_keyless_sm80"
  local rotation_keyless="${RETAINED_HOST_ROOT}/build/retained_ckks_rotation_keyless_sm80"
  local gpu_json="${RESULT_DIR}/retained_ckks_gpu_results.json"
  local gpu_bin="${RESULT_DIR}/retained_ckks_gpu_values.bin"
  local exact_observed_json="${RESULT_DIR}/retained_ckks_exact_observed.json"
  local exact_observed_bin="${RESULT_DIR}/retained_ckks_exact_observed.bin"
  local exact_evidence="${RESULT_DIR}/retained_ckks_exact_rns.json"
  local comparison="${RESULT_DIR}/retained_ckks_compare.json"
  local aliases="${RESULT_DIR}/retained_ckks_adapter_aliases.json"
  local required
  for required in "${fixture}" "${context}" "${analytic_json}" \
    "${analytic_bin}" "${exact_source_json}" "${exact_source_bin}" \
    "${ant_json}" "${ant_bin}" "${frozen_build}" "${emitted_context}" \
    "${runner}" "${native_test}" \
    "${conjugation_keyless}" \
    "${rotation_keyless}"; do
    [[ -s "${required}" ]]
  done

  local native_test_stdout="${RESULT_DIR}/retained-native-primitives.stdout.txt"
  local native_test_stderr="${RESULT_DIR}/retained-native-primitives.stderr.txt"
  timeout 900 "${native_test}" --context-manifest "${emitted_context}" \
    >"${native_test_stdout}" 2>"${native_test_stderr}"
  python3 - "${RESULT_DIR}/retained_ckks_native_primitives.json" \
    "${native_test}" "${emitted_context}" "${native_test_stdout}" \
    "${native_test_stderr}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

output, binary, context, stdout, stderr = map(Path, sys.argv[1:])
lines = stdout.read_text(encoding="utf-8").splitlines()
matches = [
    re.fullmatch(r"\[  PASSED  \]\s+([1-9][0-9]*) tests?\.", line)
    for line in lines
]
counts = [int(match.group(1)) for match in matches if match is not None]
if len(counts) != 1:
    raise SystemExit("manifest-driven Phantom test lacks one pass summary")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "schema_version": "ace.phantom.retained_ckks.native-primitives/1.0.0",
    "status": "pass",
    "test_count": counts[0],
    "binary_sha256": digest(binary),
    "emitted_context_manifest_sha256": digest(context),
    "stdout_sha256": digest(stdout),
    "stderr_sha256": digest(stderr),
}
output.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  jq -e \
    --arg binary_sha256 "$(sha256sum "${native_test}" | awk '{print $1}')" \
    --arg context_sha256 "$(sha256sum "${emitted_context}" | awk '{print $1}')" \
    --arg stdout_sha256 "$(sha256sum "${native_test_stdout}" | awk '{print $1}')" \
    --arg stderr_sha256 "$(sha256sum "${native_test_stderr}" | awk '{print $1}')" '
    (keys | sort) ==
      (["binary_sha256", "emitted_context_manifest_sha256", "schema_version",
        "status", "stderr_sha256", "stdout_sha256", "test_count"] | sort)
    and .schema_version ==
      "ace.phantom.retained_ckks.native-primitives/1.0.0"
    and .status == "pass" and .test_count > 0
    and .binary_sha256 == $binary_sha256
    and .emitted_context_manifest_sha256 == $context_sha256
    and .stdout_sha256 == $stdout_sha256
    and .stderr_sha256 == $stderr_sha256
  ' "${RESULT_DIR}/retained_ckks_native_primitives.json" >/dev/null

  timeout 900 "${runner}" aliases "${fixture}" "${context}" \
    "${analytic_json}" "${analytic_bin}" "${expected_gpu}" "${aliases}" \
    >"${RESULT_DIR}/retained-adapter-aliases.stdout.txt" \
    2>"${RESULT_DIR}/retained-adapter-aliases.stderr.txt"
  jq -e --slurpfile fixture "${fixture}" --slurpfile context "${context}" \
    --arg fixture_sha256 "$(sha256sum "${fixture}" | awk '{print $1}')" \
    --arg context_sha256 "$(sha256sum "${context}" | awk '{print $1}')" '
    def normalized_power($symbol; $degree):
      if $symbol == "0" then 0
      elif $symbol == "N/2" then ($degree / 2)
      elif $symbol == "N" then $degree
      elif $symbol == "3N/2" then ($degree + ($degree / 2))
      elif $symbol == "2N-1" then (2 * $degree - 1)
      elif $symbol == "2N+1" then 1
      else error("unsupported fixture monomial symbol")
      end;
    def power_label($symbol):
      if $symbol == "0" then "0"
      elif $symbol == "N/2" then "N_over_2"
      elif $symbol == "N" then "N"
      elif $symbol == "3N/2" then "3N_over_2"
      elif $symbol == "2N-1" then "2N_minus_1"
      elif $symbol == "2N+1" then "2N_plus_1"
      else error("unsupported fixture monomial symbol")
      end;
    (keys | sort) ==
      (["cases", "context_manifest_sha256", "fixture_sha256",
        "qualification_bindings", "schema_version", "status"] | sort)
    and .schema_version ==
      "ace.phantom.retained_ckks.adapter-aliases/1.0.0"
    and .status == "pass"
    and .fixture_sha256 == $fixture_sha256
    and .context_manifest_sha256 == $context_sha256
    and .qualification_bindings == $fixture[0].qualification_bindings
    and (.cases | length) == (1 + ($fixture[0].monomial_powers | length))
    and (.cases[0].case_id == "conjugate.in_place"
         and .cases[0].operation == "conjugate"
         and .cases[0].symbol == null
         and .cases[0].normalized_power == null)
    and ([.cases[1:][] | .symbol] == $fixture[0].monomial_powers)
    and ([.cases[1:][] | .case_id] ==
         [$fixture[0].monomial_powers[] |
          "mul_mono." + power_label(.) + ".in_place"])
    and ([.cases[1:][] | .operation] | all(. == "mul_mono"))
    and ([.cases[1:][] | .normalized_power] ==
         [$fixture[0].monomial_powers[] |
          normalized_power(.; $context[0].polynomial_degree)])
    and ([.cases[] | keys | sort] | all(
      . == (["alias_returned", "case_id", "decoded_values_match_out_of_place",
             "identity_matches_source", "in_place_decoded_sha256",
             "in_place_metadata", "in_place_residues_sha256",
             "input_decoded_sha256", "input_metadata",
             "input_residues_sha256", "metadata_matches_out_of_place",
             "normalized_power", "operation", "out_of_place_decoded_sha256",
             "out_of_place_metadata", "out_of_place_residues_sha256",
             "residues_match_out_of_place", "source_preserved", "status",
             "symbol"] | sort)
    ))
    and ([.cases[] |
      (.status == "pass" and .alias_returned == true and .source_preserved == true
       and .metadata_matches_out_of_place == true
       and .decoded_values_match_out_of_place == true
       and .residues_match_out_of_place == true
       and .out_of_place_metadata == .in_place_metadata
       and .out_of_place_decoded_sha256 == .in_place_decoded_sha256
       and .out_of_place_residues_sha256 == .in_place_residues_sha256)] | all)
    and ([.cases[1:][] | select(.symbol == "0") |
          (.identity_matches_source == true
           and .input_metadata == .in_place_metadata
           and .input_decoded_sha256 == .in_place_decoded_sha256
           and .input_residues_sha256 == .in_place_residues_sha256)] == [true])
    and ([.cases[] | select(.symbol != "0") |
          .identity_matches_source] | all(. == null))
  ' "${aliases}" >/dev/null

  timeout 900 "${runner}" conformance "${fixture}" "${context}" \
    "${analytic_json}" "${analytic_bin}" "${exact_source_json}" \
    "${exact_source_bin}" "${expected_gpu}" "${gpu_json}" "${gpu_bin}" \
    "${exact_observed_json}" "${exact_observed_bin}" \
    >"${RESULT_DIR}/retained-conformance.stdout.txt" \
    2>"${RESULT_DIR}/retained-conformance.stderr.txt"

  python3 \
    "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/retained_runpod_evidence.py" \
    write-run-attestations \
    --frozen "${INPUT}" \
    --retained-root "${RETAINED_HOST_ROOT}" \
    --result-dir "${RESULT_DIR}" \
    --expected-gpu "${expected_gpu}" \
    >"${RESULT_DIR}/retained-run-evidence.json"

  python3 \
    "${ACE_PHANTOM_REPO_ROOT}/tools/phantom_gpu/compare_retained_ckks_results.py" \
    --fixture "${fixture}" \
    --context-manifest "${context}" \
    --generation-fixture \
      "${RETAINED_HOST_ROOT}/inputs/retained_ckks_fixture_template.json" \
    --generation-attestation \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_generation.json" \
    --emitted-context-manifest \
      "${RETAINED_HOST_ROOT}/outputs/compiler_context_manifest.json" \
    --resource-manifest \
      "${RETAINED_HOST_ROOT}/outputs/compiler_resource_manifest.json" \
    --constant-manifest \
      "${RETAINED_HOST_ROOT}/outputs/compiler_constant_manifest.json" \
    --ant-post-ckks-air \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_ant_post.air" \
    --post-ckks-air \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_phantom_post.air" \
    --production-post-ckks-air \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_production_post.air" \
    --ace-source-manifest \
      "${RETAINED_HOST_ROOT}/source/ace_source_manifest.json" \
    --phantom-source-manifest \
      "${RETAINED_HOST_ROOT}/source/phantom_source_manifest.json" \
    --generated-ant-source \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_ant.cxx" \
    --generated-phantom-source \
      "${RETAINED_HOST_ROOT}/outputs/retained_ckks_phantom.cu" \
    --phantom-executable "${runner}" \
    --frozen-build-attestation "${frozen_build}" \
    --remote-build-attestation \
      "${RETAINED_HOST_ROOT}/build_attestation.json" \
    --run-attestation "${RESULT_DIR}/retained_ckks_run_attestation.json" \
    --provider-attestation \
      "${RESULT_DIR}/retained_ckks_provider_attestation.json" \
    --analytic-json "${analytic_json}" --analytic-bin "${analytic_bin}" \
    --ant-json "${ant_json}" --ant-bin "${ant_bin}" \
    --gpu-json "${gpu_json}" --gpu-bin "${gpu_bin}" \
    --exact-source-json "${exact_source_json}" \
    --exact-source-bin "${exact_source_bin}" \
    --exact-observed-json "${exact_observed_json}" \
    --exact-observed-bin "${exact_observed_bin}" \
    --exact-output-json "${exact_evidence}" \
    --output-json "${comparison}"
  jq -e '
    .status == "pass"
    and .gpu_vs_exact.status == "pass"
    and .gpu_vs_exact.mismatch_count == 0
    and (.gpu_vs_analytic | length) > 0
    and (.gpu_vs_ant | length) > 0
    and ([.metamorphic_checks[]] | all)
  ' "${comparison}" >/dev/null
  jq -e '.status == "pass" and .mismatch_count == 0' \
    "${exact_evidence}" >/dev/null

  local rejection_table="${WORK}/retained-runtime-rejections.tsv"
  jq -er '.runtime_rejections[] | [.id, .diagnostic, .manifest] | @tsv' \
    "${fixture}" >"${rejection_table}"
  local rejection_jsonl="${WORK}/retained-runtime-rejections.jsonl"
  : >"${rejection_jsonl}"
  local case_id diagnostic manifest selected_runner stdout_file stderr_file
  local rejection_exit
  while IFS=$'\t' read -r case_id diagnostic manifest; do
    case "${manifest}" in
      production) selected_runner="${runner}" ;;
      keyless-conjugation) selected_runner="${conjugation_keyless}" ;;
      keyless-rotation) selected_runner="${rotation_keyless}" ;;
      *)
        echo "unsupported retained rejection manifest ${manifest}" >&2
        return 1
        ;;
    esac
    stdout_file="${RESULT_DIR}/retained-reject-${case_id}.stdout.txt"
    stderr_file="${RESULT_DIR}/retained-reject-${case_id}.stderr.txt"
    ulimit -c 0
    set +e
    timeout 180 "${selected_runner}" reject "${fixture}" "${context}" \
      "${analytic_json}" "${analytic_bin}" "${expected_gpu}" "${case_id}" \
      >"${stdout_file}" 2>"${stderr_file}"
    rejection_exit=$?
    set -e
    if [[ ${rejection_exit} -eq 0 || ${rejection_exit} -eq 124 ]]; then
      echo "retained runtime rejection ${case_id} returned ${rejection_exit}" >&2
      return 1
    fi
    rg -Fq "ACE_RETAINED_EXPECT_DIAGNOSTIC[${diagnostic}]" \
      "${stderr_file}"
    rg -Fq "ACE_PHANTOM_ORDINARY_ERROR[${diagnostic}]" "${stderr_file}"
    jq -cn --arg case_id "${case_id}" --arg diagnostic "${diagnostic}" \
      --arg manifest "${manifest}" --argjson exit_code "${rejection_exit}" \
      '{case_id:$case_id, diagnostic:$diagnostic, manifest:$manifest,
        exit_code:$exit_code, status:"pass"}' >>"${rejection_jsonl}"
  done <"${rejection_table}"
  jq -s \
    '{schema_version:"ace.phantom.retained_ckks.rejections/1.0.0",
      status:"pass", cases:.}' "${rejection_jsonl}" \
    >"${RESULT_DIR}/retained_ckks_rejections.json"
  jq -e --slurpfile fixture "${fixture}" '
    (keys | sort) == (["cases", "schema_version", "status"] | sort)
    and .schema_version ==
      "ace.phantom.retained_ckks.rejections/1.0.0"
    and .status == "pass"
    and ([.cases[] | {id:.case_id, diagnostic, manifest}]
         == [$fixture[0].runtime_rejections[] | {id, diagnostic, manifest}])
    and ([.cases[] | keys | sort] | all(
      . == (["case_id", "diagnostic", "exit_code", "manifest", "status"] | sort)
    ))
    and ([.cases[] |
      (.status == "pass" and (.exit_code | type) == "number"
       and .exit_code != 0 and .exit_code != 124)] | all)
  ' "${RESULT_DIR}/retained_ckks_rejections.json" >/dev/null
  rm "${rejection_table}" "${rejection_jsonl}"

  local ownership="${RESULT_DIR}/retained_ckks_ownership.json"
  local sanitizer_log="${RESULT_DIR}/retained_ckks_sanitizer.log.txt"
  local sanitizer_stdout="${RESULT_DIR}/retained_ckks_sanitizer.stdout.txt"
  local sanitizer_stderr="${RESULT_DIR}/retained_ckks_sanitizer.stderr.txt"
  local sanitizer_version="${RESULT_DIR}/retained_ckks_sanitizer.version.txt"
  compute-sanitizer --version >"${sanitizer_version}" 2>&1
  set +e
  timeout 900 compute-sanitizer --tool memcheck --error-exitcode=99 \
    --log-file "${sanitizer_log}" "${runner}" ownership "${fixture}" \
    "${context}" "${analytic_json}" "${analytic_bin}" "${expected_gpu}" \
    "${ownership}" >"${sanitizer_stdout}" 2>"${sanitizer_stderr}"
  local sanitizer_exit=$?
  set -e
  if [[ ${sanitizer_exit} -ne 0 ]]; then
    echo "retained Compute Sanitizer exited ${sanitizer_exit}" >&2
    return "${sanitizer_exit}"
  fi
  [[ "$(rg -c '^========= ERROR SUMMARY: 0 errors$' "${sanitizer_log}" || true)" == 1 ]]
  [[ "$(rg -c '^========= ERROR SUMMARY:' "${sanitizer_log}" || true)" == 1 ]]
  jq -e --slurpfile fixture "${fixture}" '
    (keys | sort) ==
      (["batch_outputs_are_independent", "free_order", "iterations",
        "ordered_steps", "schema_version", "status"] | sort)
    and .schema_version == "ace.phantom.retained_ckks.ownership/1.0.0"
    and .status == "pass"
    and .iterations == $fixture[0].ownership.iterations
    and .ordered_steps == $fixture[0].rotate_batch_steps
    and .batch_outputs_are_independent ==
      $fixture[0].ownership.batch_outputs_are_independent
    and .free_order == $fixture[0].ownership.free_order
  ' "${ownership}" >/dev/null
  local ownership_iterations
  ownership_iterations="$(jq -er '.iterations' "${ownership}")"
  jq -n --arg status pass --arg tool memcheck \
    --argjson ownership_iterations "${ownership_iterations}" \
    --arg log_sha256 "$(sha256sum "${sanitizer_log}" | awk '{print $1}')" \
    --arg version_sha256 \
      "$(sha256sum "${sanitizer_version}" | awk '{print $1}')" \
    '{schema_version:"ace.phantom.retained_ckks.sanitizer/1.0.0",
      status:$status, tool:$tool, error_summary:0,
      ownership_iterations:$ownership_iterations,
      log_sha256:$log_sha256, version_sha256:$version_sha256}' \
    >"${RESULT_DIR}/retained_ckks_sanitizer.json"

  cp -- "${context}" "${RESULT_DIR}/retained_compiler_context_manifest.json"
  cp -- "${RETAINED_HOST_ROOT}/outputs/compiler_resource_manifest.json" \
    "${RESULT_DIR}/retained_compiler_resource_manifest.json"
  cp -- "${RETAINED_HOST_ROOT}/outputs/compiler_constant_manifest.json" \
    "${RESULT_DIR}/retained_compiler_constant_manifest.json"
  cp -- "${fixture}" "${RESULT_DIR}/retained_ckks_v1.json"
  cp -- "${ant_json}" "${RESULT_DIR}/retained_ckks_cpu_reference.json"
  cp -- "${ant_bin}" "${RESULT_DIR}/retained_ckks_cpu_values.bin"
  cp -- "${RETAINED_HOST_ROOT}/outputs/retained_ckks_generation.json" \
    "${RESULT_DIR}/retained_ckks_generation.json"
  cp -- "${RETAINED_HOST_ROOT}/source/ace_source_manifest.json" \
    "${RESULT_DIR}/retained_ace_source_manifest.json"
  cp -- "${RETAINED_HOST_ROOT}/source/phantom_source_manifest.json" \
    "${RESULT_DIR}/retained_phantom_source_manifest.json"
  cp -- \
    "${RETAINED_HOST_ROOT}/build/inspection/production-archive-audit.json" \
    "${RESULT_DIR}/retained_production_archive_audit.json"
  cp -- "${RETAINED_HOST_ROOT}/build/inspection/generated-source-audit.json" \
    "${RESULT_DIR}/retained_generated_source_audit.json"
  cp -- "${RETAINED_HOST_ROOT}/build/gtest-source-attestation.json" \
    "${RESULT_DIR}/retained_gtest_source_attestation.json"
}

run_bootstrap_constant_cache_qualification() {
  local expected_gpu current run_root bootstrap_dir binary context constants
  local raw stdout_file stderr_file execution metrics execution_exit
  expected_gpu="${ACE_RUNPOD_EXPECTED_GPU_NAME:-}"
  case "${expected_gpu}" in
    "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB") ;;
    *)
      echo "an exact supported A100 GPU identity is required" >&2
      return 1
      ;;
  esac
  current="${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-bootstrap.json"
  [[ "$(jq -er '.gate + ":" + .status' "${current}")" == "bootstrap:pass" ]]
  run_root="$(jq -er .run_root "${current}")"
  bootstrap_dir="${run_root}/bootstrap_qualification"
  binary="${bootstrap_dir}/bootstrap_phantom_constants_sm80"
  context="${bootstrap_dir}/compiler_context_manifest.json"
  constants="${bootstrap_dir}/compiler_constant_manifest.json"
  [[ -x "${binary}" && -s "${context}" && -s "${constants}" ]]
  raw="${RESULT_DIR}/bootstrap-constant-cache.raw.json"
  stdout_file="${RESULT_DIR}/bootstrap-constant-cache.stdout.txt"
  stderr_file="${RESULT_DIR}/bootstrap-constant-cache.stderr.txt"
  execution="${RESULT_DIR}/bootstrap-constant-cache-execution.json"
  metrics="${RESULT_DIR}/bootstrap-setup-metrics.json"
  set +e
  timeout 1800 "${binary}" "${raw}" >"${stdout_file}" 2>"${stderr_file}"
  execution_exit=$?
  set -e
  [[ ${execution_exit} -eq 0 ]]
  python3 - "${raw}" "${context}" "${constants}" "${binary}" \
    "${stdout_file}" "${stderr_file}" "${expected_gpu}" \
    "${RESULT_DIR}/bootstrap-constant-cache.json" "${metrics}" \
    "${execution}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

(raw_path, context_path, constants_path, binary_path, stdout_path, stderr_path,
 expected_gpu, output_path, metrics_path, execution_path) = sys.argv[1:]
raw_path, context_path, constants_path, binary_path = map(
    Path, (raw_path, context_path, constants_path, binary_path)
)
stdout_path, stderr_path, output_path, metrics_path, execution_path = map(
    Path, (stdout_path, stderr_path, output_path, metrics_path, execution_path)
)
def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in bootstrap runtime result: {key}")
        value[key] = item
    return value

load = lambda path: json.loads(
    path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
result = load(raw_path)
context = load(context_path)
constants = load(constants_path)
top_keys = {
    "status", "gpu", "context_manifest_sha256", "resource_flags", "counts",
    "tolerance", "maximum_correctness_error", "maximum_cached_error",
    "maximum_mode_error", "setup_metrics", "entries",
}
if set(result) != top_keys or result.get("status") != "pass":
    raise SystemExit("bootstrap constant-cache result has an invalid shape/status")
context_sha = digest(context_path)
if (
    result.get("gpu") != expected_gpu
    or "A100" not in result["gpu"]
    or result.get("context_manifest_sha256") != context_sha
    or result.get("resource_flags") != 127
):
    raise SystemExit("bootstrap result GPU/context/resource binding is invalid")
count = len(constants["constants"])
counts = result.get("counts")
if counts != {
    "constants_declared": count,
    "correctness_encodes": count,
    "cached_loads": count,
} or count <= 0:
    raise SystemExit("bootstrap constant-cache operation counts are invalid")
if result.get("tolerance") != {"absolute": 1.0e-7, "relative": 1.0e-10}:
    raise SystemExit("bootstrap constant-cache tolerance is invalid")
for field in (
    "maximum_correctness_error", "maximum_cached_error", "maximum_mode_error"
):
    value = result.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0:
        raise SystemExit(f"bootstrap result has an invalid error metric: {field}")

setup = result.get("setup_metrics")
if not isinstance(setup, dict) or set(setup) != {"context_and_keys", "plaintext_cache"}:
    raise SystemExit("bootstrap setup metrics have an invalid shape")
context_setup = setup["context_and_keys"]
cache_setup = setup["plaintext_cache"]
if set(context_setup) != {"seconds", "device_bytes"} or set(cache_setup) != {
    "seconds", "device_bytes", "logical_device_bytes", "host_bytes", "entries"
}:
    raise SystemExit("bootstrap setup/cache metric fields are incomplete")
for value in (context_setup["seconds"], cache_setup["seconds"]):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise SystemExit("bootstrap setup duration is not positive and finite")
for field, value in (
    ("context device bytes", context_setup["device_bytes"]),
    ("cache device bytes", cache_setup["device_bytes"]),
    ("cache logical device bytes", cache_setup["logical_device_bytes"]),
    ("cache host bytes", cache_setup["host_bytes"]),
    ("cache entries", cache_setup["entries"]),
):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"bootstrap {field} is not a nonnegative integer")
expected_logical_bytes = sum(
    context["polynomial_degree"] * entry["ace_level"] * 8
    for entry in constants["constants"]
)
emitted_payload_bytes = sum(
    entry["slot_count"] * 2 * 8 for entry in constants["constants"]
)
if (
    expected_logical_bytes <= 0
    or cache_setup["logical_device_bytes"] != expected_logical_bytes
    or cache_setup["host_bytes"] < emitted_payload_bytes
    or cache_setup["entries"] != count
):
    raise SystemExit("bootstrap logical cache-byte/accounting metrics are invalid")

entries = result.get("entries")
if not isinstance(entries, list) or len(entries) != count:
    raise SystemExit("bootstrap entry result count is invalid")
entry_keys = {
    "entry_id", "constant_id", "symbol", "slot_count", "ace_level",
    "chain_index", "raw_scale", "parameter_fingerprint",
    "maximum_correctness_error", "maximum_cached_error", "maximum_mode_error",
}
for index, (entry, manifest) in enumerate(zip(entries, constants["constants"])):
    if set(entry) != entry_keys:
        raise SystemExit(f"bootstrap entry result shape is invalid: {index}")
    for field in (
        "entry_id", "constant_id", "slot_count", "ace_level", "chain_index"
    ):
        if isinstance(entry[field], bool) or not isinstance(entry[field], int) \
                or entry[field] < 0:
            raise SystemExit(f"bootstrap entry integer field is invalid: {index}/{field}")
    if (
        entry["entry_id"] != index
        or entry["constant_id"] != manifest["constant_id"]
        or entry["symbol"] != manifest["symbol"]
        or entry["slot_count"] != manifest["slot_count"]
        or entry["ace_level"] != manifest["ace_level"]
        or entry["chain_index"] != manifest["chain_index"]
        or entry["raw_scale"] != float.fromhex(manifest["raw_scale"])
        or re.fullmatch(r"[0-9a-f]{64}", entry["parameter_fingerprint"]) is None
    ):
        raise SystemExit(f"bootstrap entry differs from its manifest: {index}")
    for field in (
        "maximum_correctness_error", "maximum_cached_error", "maximum_mode_error"
    ):
        value = entry[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise SystemExit(f"bootstrap entry error metric is invalid: {index}/{field}")

output_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
metrics_record = {
    "schema_version": "ace.phantom.bootstrap-setup-metrics/1.0.0",
    "status": "pass",
    "context_manifest_sha256": context_sha,
    "constant_count": count,
    "emitted_payload_bytes": emitted_payload_bytes,
    "expected_logical_device_bytes": expected_logical_bytes,
    "context_and_keys": context_setup,
    "plaintext_cache": cache_setup,
}
metrics_path.write_text(
    json.dumps(metrics_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
execution_record = {
    "schema_version": "ace.phantom.bootstrap-constant-cache-execution/1.0.0",
    "status": "pass",
    "invocation_count": 1,
    "binary_sha256": digest(binary_path),
    "raw_result_sha256": digest(raw_path),
    "stdout_sha256": digest(stdout_path),
    "stderr_sha256": digest(stderr_path),
}
execution_path.write_text(
    json.dumps(execution_record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
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
  require_terminal_record() {
    local record_name="$1"
    shift
    local record_path="${RESULT_DIR}/${record_name}"
    if [[ ! -s "${record_path}" ]]; then
      printf 'result completeness: missing or empty terminal record: %s\n' \
        "${record_name}" >&2
      return 1
    fi
    if ! jq -e "$@" "${record_path}" >/dev/null; then
      printf 'result completeness: terminal record violates its contract: %s\n' \
        "${record_name}" >&2
      return 1
    fi
  }

  require_terminal_record qualification-current.json '
    .gate == "ordinary" and .status == "pass" and .exit_code == 0
  '
  case "${MODE}" in
    freeze-host)
      require_terminal_record host-freeze-candidate.json '
        .schema_version == "ace.phantom.host-freeze-candidate/1.0.0"
        and .status == "candidate"
      '
      ;;
    local)
      require_terminal_record ordinary-frozen-reference.json \
        '.status == "pass"'
      require_terminal_record native-health.json '.status == "skipped"'
      require_terminal_record bootstrap-qualification-current.json '
        .gate == "bootstrap" and .status == "pass" and .exit_code == 0
      '
      require_terminal_record bootstrap-frozen-reference.json '
        .schema_version == "ace.phantom.bootstrap-frozen-reference/1.1.0"
        and .status == "pass"
        and .comparison ==
          "exact-source-and-setup-with-per-run-cuda-artifacts"
        and .source_and_setup_match == true
        and .linked_binary_byte_identity_required == false
        and .packaged_generated_source_sha256 ==
          .regenerated_generated_source_sha256
        and .packaged_harness_source_sha256 ==
          .regenerated_harness_source_sha256
        and ([.packaged_artifact_manifest_sha256,
              .packaged_host_qualification_sha256,
              .packaged_source_audit_sha256,
              .regenerated_artifact_manifest_sha256,
              .regenerated_host_qualification_sha256,
              .regenerated_source_audit_sha256,
              .packaged_generated_source_sha256,
              .regenerated_generated_source_sha256,
              .packaged_harness_source_sha256,
              .regenerated_harness_source_sha256,
              .packaged_linked_binary_sha256,
              .regenerated_linked_binary_sha256] |
             all(test("^[0-9a-f]{64}$")))
      '
      require_terminal_record bootstrap-constant-cache.json \
        '.status == "skipped"'
      require_terminal_record bootstrap-setup-metrics.json \
        '.status == "skipped"'
      require_terminal_record bootstrap-constant-cache-execution.json '
        .status == "skipped" and .invocation_count == 0
      '
      require_terminal_record retained-frozen-reference.json '
        .status == "pass"
        and .provider_neutral_ant_reference_matches == true
        and .exact_artifact_count > 0
      '
      ;;
    runpod)
      require_terminal_record native-health.json \
        '.status == "pass" and .device_count == 1'
      require_terminal_record bootstrap-qualification-current.json '
        .gate == "bootstrap" and .status == "pass" and .exit_code == 0
      '
      require_terminal_record bootstrap-frozen-reference.json '
        .schema_version == "ace.phantom.bootstrap-frozen-reference/1.1.0"
        and .status == "pass"
        and .comparison ==
          "exact-source-and-setup-with-per-run-cuda-artifacts"
        and .source_and_setup_match == true
        and .linked_binary_byte_identity_required == false
        and .packaged_generated_source_sha256 ==
          .regenerated_generated_source_sha256
        and .packaged_harness_source_sha256 ==
          .regenerated_harness_source_sha256
        and ([.packaged_artifact_manifest_sha256,
              .packaged_host_qualification_sha256,
              .packaged_source_audit_sha256,
              .regenerated_artifact_manifest_sha256,
              .regenerated_host_qualification_sha256,
              .regenerated_source_audit_sha256,
              .packaged_generated_source_sha256,
              .regenerated_generated_source_sha256,
              .packaged_harness_source_sha256,
              .regenerated_harness_source_sha256,
              .packaged_linked_binary_sha256,
              .regenerated_linked_binary_sha256] |
             all(test("^[0-9a-f]{64}$")))
      '
      require_terminal_record bootstrap-constant-cache.json '
        .status == "pass" and (.gpu | contains("A100"))
        and .resource_flags == 127
        and .counts.constants_declared > 0
        and .counts.correctness_encodes == .counts.constants_declared
        and .counts.cached_loads == .counts.constants_declared
        and (.entries | length) == .counts.constants_declared
      '
      require_terminal_record bootstrap-setup-metrics.json '
        .schema_version == "ace.phantom.bootstrap-setup-metrics/1.0.0"
        and .status == "pass" and .constant_count > 0
        and .plaintext_cache.entries == .constant_count
        and .plaintext_cache.logical_device_bytes ==
          .expected_logical_device_bytes
        and .expected_logical_device_bytes > 0
        and .plaintext_cache.host_bytes >= .emitted_payload_bytes
        and .context_and_keys.seconds > 0
        and .plaintext_cache.seconds > 0
      '
      require_terminal_record bootstrap-constant-cache-execution.json \
        --slurpfile artifact \
          "${RESULT_DIR}/bootstrap-qualification/artifact_manifest.json" '
        .schema_version ==
          "ace.phantom.bootstrap-constant-cache-execution/1.0.0"
        and .status == "pass" and .invocation_count == 1
        and .binary_sha256 ==
          $artifact[0].files[
            "bootstrap_qualification/bootstrap_phantom_constants_sm80"]
      '
      require_terminal_record ordinary_ckks_compare.json '
        .status == "pass" and .case_count > 0
        and .passing_case_count == .case_count
      '
      require_terminal_record ordinary_ckks_diagnostics.json \
        '.status == "pass" and (.cases | length) > 0'
      require_terminal_record ordinary_ckks_ownership.json '
        .schema_version == "ace.phantom.ordinary_ckks.ownership/1.0.0"
        and .status == "pass" and .total_iterations > 0
      '
      require_terminal_record ordinary_ckks_sanitizer.json \
        '.status == "pass" and .exit_code == 0'
      require_terminal_record retained-frozen-reference.json '
        .status == "pass"
        and .provider_neutral_ant_reference_matches == true
        and .ant_replay.comparison ==
          "semantic-summary-only-no-decoded-byte-comparison"
        and .exact_artifact_count > 0
      '
      require_terminal_record retained_ckks_compare.json '
        .status == "pass"
        and .gpu_vs_exact.status == "pass"
        and .gpu_vs_exact.mismatch_count == 0
      '
      require_terminal_record retained_ckks_exact_rns.json \
        '.status == "pass" and .mismatch_count == 0'
      require_terminal_record retained_ckks_adapter_aliases.json '
        .schema_version ==
          "ace.phantom.retained_ckks.adapter-aliases/1.0.0"
        and .status == "pass" and (.cases | length) > 1
        and ([.cases[].status] | all(. == "pass"))
        and ([.cases[].alias_returned] | all)
        and ([.cases[].metadata_matches_out_of_place] | all)
        and ([.cases[].decoded_values_match_out_of_place] | all)
        and ([.cases[].residues_match_out_of_place] | all)
      '
      require_terminal_record retained_ckks_rejections.json \
        --slurpfile fixture "${RESULT_DIR}/retained_ckks_v1.json" '
        (keys | sort) == (["cases", "schema_version", "status"] | sort)
        and .schema_version ==
          "ace.phantom.retained_ckks.rejections/1.0.0"
        and .status == "pass"
        and ([.cases[] | {id:.case_id, diagnostic, manifest}]
             == [$fixture[0].runtime_rejections[] | {id, diagnostic, manifest}])
        and ([.cases[] | keys | sort] | all(
          . == (["case_id", "diagnostic", "exit_code", "manifest", "status"] | sort)
        ))
        and ([.cases[] |
          (.status == "pass" and (.exit_code | type) == "number"
           and .exit_code != 0 and .exit_code != 124)] | all)
      '
      require_terminal_record retained_ckks_ownership.json \
        --slurpfile fixture "${RESULT_DIR}/retained_ckks_v1.json" '
        (keys | sort) ==
          (["batch_outputs_are_independent", "free_order", "iterations",
            "ordered_steps", "schema_version", "status"] | sort)
        and .schema_version ==
          "ace.phantom.retained_ckks.ownership/1.0.0"
        and .status == "pass"
        and .iterations == $fixture[0].ownership.iterations
        and .ordered_steps == $fixture[0].rotate_batch_steps
        and .batch_outputs_are_independent ==
          $fixture[0].ownership.batch_outputs_are_independent
        and .free_order == $fixture[0].ownership.free_order
      '
      require_terminal_record retained_ckks_sanitizer.json \
        '.status == "pass" and .error_summary == 0'
      require_terminal_record retained_ckks_native_primitives.json \
        --slurpfile artifact \
        "${RESULT_DIR}/retained-host/artifact-manifest.json" \
        --arg context_sha256 \
          "$(sha256sum "${RESULT_DIR}/retained_compiler_context_manifest.json" | awk '{print $1}')" \
        --arg stdout_sha256 \
          "$(sha256sum "${RESULT_DIR}/retained-native-primitives.stdout.txt" | awk '{print $1}')" \
        --arg stderr_sha256 \
          "$(sha256sum "${RESULT_DIR}/retained-native-primitives.stderr.txt" | awk '{print $1}')" '
        (keys | sort) ==
          (["binary_sha256", "emitted_context_manifest_sha256",
            "schema_version", "status", "stderr_sha256", "stdout_sha256",
            "test_count"] | sort)
        and .schema_version ==
          "ace.phantom.retained_ckks.native-primitives/1.0.0"
        and .status == "pass" and .test_count > 0
        and .binary_sha256 ==
          $artifact[0].files["build/retained_ckks_native_primitives_sm80"]
        and .emitted_context_manifest_sha256 == $context_sha256
        and .stdout_sha256 == $stdout_sha256
        and .stderr_sha256 == $stderr_sha256
      '
      require_terminal_record retained_production_archive_audit.json '
        .schema_version ==
          "ace.phantom.retained_ckks.production-archive-audit/1.0.0"
        and .status == "pass" and .forbidden_entry_count == 0
        and (.inventories | length) == 6
        and ([.inventories[].forbidden_entries] | all(. == []))
      '
      require_terminal_record retained_generated_source_audit.json \
        --slurpfile fixture "${RESULT_DIR}/retained_ckks_v1.json" \
        --slurpfile artifact \
          "${RESULT_DIR}/retained-host/artifact-manifest.json" '
        (keys | sort) ==
          (["argument_order", "call_counts", "cipher_array_copy_count",
            "cipher_array_copy_indices", "expected_cipher_array_copy_count",
            "expected_cipher_array_copy_indices",
            "expected_rotation_batches", "first_distinct_calls",
            "forbidden_matches", "observed_rotation_batches",
            "raw_cipher_array_assignments", "required_calls",
            "rotation_array_emission", "schema_version", "source_sha256",
            "status"] | sort)
        and
        .schema_version ==
          "ace.phantom.retained_ckks.generated-source-audit/2.0.0"
        and .status == "pass"
        and .source_sha256 ==
          $artifact[0].files["outputs/retained_ckks_phantom.cu"]
        and .required_calls ==
          ["Conjugate_ciph", "Rotate_batch_ciph", "Raise_mod", "Mul_mono_ciph"]
        and .first_distinct_calls == .required_calls
        and (.call_counts | keys | sort) == (.required_calls | sort)
        and ([.call_counts[] |
          type == "number" and . > 0 and floor == .] | all)
        and .argument_order == {
          "Conjugate_ciph": true, "Rotate_batch_ciph": true,
          "Raise_mod": true, "Mul_mono_ciph": true
        }
        and .rotation_array_emission == "ckks-owned-static-int32"
        and .expected_rotation_batches ==
          ([$fixture[0].rotate_batch_steps]
           + $fixture[0].production_rotation_batches
           + [$fixture[0].rotate_batch_steps])
        and .observed_rotation_batches == .expected_rotation_batches
        and .cipher_array_copy_count == .expected_cipher_array_copy_count
        and .expected_cipher_array_copy_count ==
          (1 + ($fixture[0].production_rotation_batches | length)
           + ($fixture[0].rotate_batch_steps | length))
        and .cipher_array_copy_indices == .expected_cipher_array_copy_indices
        and .expected_cipher_array_copy_indices ==
          ([range(0;
              1 + ($fixture[0].production_rotation_batches | length)) | 0]
           + [range(0; ($fixture[0].rotate_batch_steps | length))])
        and .raw_cipher_array_assignments == []
        and .forbidden_matches == []
      '
      require_terminal_record retained_gtest_source_attestation.json '
        .schema_version == "ace.phantom.retained_ckks.gtest-source/1.0.0"
        and .status == "pass"
        and .source_method == "ace-pinned-external-project-source-reuse"
        and .fetchcontent_source_override == true
        and .fetchcontent_fully_disconnected == true
      '
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
  phase bootstrap_payload_validation validate_packaged_bootstrap_qualification
  phase bootstrap_host_qualification run_bootstrap_host_qualification
  phase frozen_bootstrap_qualification verify_frozen_bootstrap_qualification
  if [[ "${MODE}" == "runpod" ]]; then
    phase native_a100_health run_native_health
    phase bootstrap_constant_cache_qualification \
      run_bootstrap_constant_cache_qualification
    phase ordinary_gpu_qualification run_ordinary_gpu_qualification
    # The retained build and all retained execution deliberately begin only
    # after the complete ordinary GPU prerequisite has passed.
    phase retained_host_qualification run_retained_host_qualification
    phase frozen_retained_reference verify_frozen_retained_reference
    phase retained_gpu_qualification run_retained_gpu_qualification
  else
    printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
      >"${RESULT_DIR}/native-health.json"
    printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
      >"${RESULT_DIR}/bootstrap-constant-cache.json"
    printf '{"status":"skipped","reason":"local host has no GPU"}\n' \
      >"${RESULT_DIR}/bootstrap-setup-metrics.json"
    printf '{"status":"skipped","reason":"local host has no GPU","invocation_count":0}\n' \
      >"${RESULT_DIR}/bootstrap-constant-cache-execution.json"
    phase retained_host_qualification run_retained_host_qualification
    phase frozen_retained_reference verify_frozen_retained_reference
  fi
fi
phase result_completeness verify_success_evidence
PIPELINE_EXIT=0
