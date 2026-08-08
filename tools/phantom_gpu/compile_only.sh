#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/phase_helpers.sh"
REPO_ROOT="${ACE_PHANTOM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)}"
LOCK_FILE="${SCRIPT_DIR}/configs/dependencies.env"
STATE_ROOT="${ACE_PHANTOM_STATE_ROOT:-${REPO_ROOT}/build/phantom_gpu}"
DEPENDENCY_ROOT="${STATE_ROOT}/dependencies"
INSTALL_ROOT="${STATE_ROOT}/install-cuda-sm80"
ACE_BUILD="${STATE_ROOT}/ace-cuda-sm80"
BINDINGS_BUILD="${STATE_ROOT}/bindings-cuda-sm80"
PHANTOM_MOUNT="${ACE_PHANTOM_SOURCE_DIR:-/deps/phantom-ant}"
MODELS_MOUNT="/inputs/models"
DATASET_MOUNT="/inputs/dataset"
CUDA_ROOT="/usr/local/cuda"
NVCC="${CUDA_ROOT}/bin/nvcc"
CUOBJDUMP="${CUDA_ROOT}/bin/cuobjdump"
BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-2}"
SOURCE_MODE="${ACE_PHANTOM_SOURCE_MODE:-git}"
ORIGINAL_ARGUMENTS=("$@")
GATE=""
COMPILER_POLY_DEGREE=""
COMPILER_MUL_LEVEL=""
COMPILER_INPUT_LEVEL=""
COMPILER_SECURITY_LEVEL=""
COMPILER_SCALING_BITS=""
COMPILER_FIRST_PRIME_BITS=""
COMPILER_HAMMING_WEIGHT=""
ORDINARY_RESULTS=""

usage() {
  echo "usage: $0 --gate <toolchain|ckks2c|ordinary|all> [compiler context options]"
  echo "CKKS2C/ordinary/all requires --poly-degree N --mul-level Q --input-level L"
  echo "  --security-level B --scaling-factor-bits B --first-prime-bits B"
  echo "  --hamming-weight W"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate)
      GATE="$2"
      shift 2
      ;;
    --poly-degree)
      COMPILER_POLY_DEGREE="$2"
      shift 2
      ;;
    --mul-level)
      COMPILER_MUL_LEVEL="$2"
      shift 2
      ;;
    --input-level)
      COMPILER_INPUT_LEVEL="$2"
      shift 2
      ;;
    --security-level)
      COMPILER_SECURITY_LEVEL="$2"
      shift 2
      ;;
    --scaling-factor-bits)
      COMPILER_SCALING_BITS="$2"
      shift 2
      ;;
    --first-prime-bits)
      COMPILER_FIRST_PRIME_BITS="$2"
      shift 2
      ;;
    --hamming-weight)
      COMPILER_HAMMING_WEIGHT="$2"
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

case "${GATE}" in
  toolchain|ckks2c|ordinary|all)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ "${GATE}" != "toolchain" ]]; then
  for value in \
    "${COMPILER_POLY_DEGREE}" "${COMPILER_MUL_LEVEL}" \
    "${COMPILER_INPUT_LEVEL}" "${COMPILER_SECURITY_LEVEL}" \
    "${COMPILER_SCALING_BITS}" "${COMPILER_FIRST_PRIME_BITS}" \
    "${COMPILER_HAMMING_WEIGHT}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
      echo "all compiler context options are required nonnegative integers" >&2
      exit 2
    fi
  done
fi

python3 - "${GATE}" "${ORIGINAL_ARGUMENTS[@]}" <<'PY'
import re
import sys

gate = sys.argv[1]
arguments = sys.argv[2:]
if len(arguments) % 2:
    raise SystemExit("qualification arguments must be option/value pairs")
pairs = dict(zip(arguments[0::2], arguments[1::2]))
if len(pairs) != len(arguments) // 2:
    raise SystemExit("qualification arguments must not be duplicated")
if gate == "toolchain":
    if pairs != {"--gate": "toolchain"}:
        raise SystemExit("toolchain qualification accepts only --gate toolchain")
else:
    expected = {
        "--gate", "--poly-degree", "--mul-level", "--input-level",
        "--security-level", "--scaling-factor-bits", "--first-prime-bits",
        "--hamming-weight",
    }
    if set(pairs) != expected or pairs["--gate"] != gate:
        raise SystemExit("qualification compiler context arguments are incomplete")
    for option in expected - {"--gate"}:
        if re.fullmatch(r"[0-9]+", pairs[option]) is None:
            raise SystemExit(f"qualification argument {option} is not an integer")
PY

set -a
source "${LOCK_FILE}"
set +a

case "${SOURCE_MODE}" in
  git|snapshot) ;;
  *)
    echo "ACE_PHANTOM_SOURCE_MODE must be git or snapshot" >&2
    exit 2
    ;;
esac

if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
  PINNED_SOURCE="${PHANTOM_MOUNT}"
  ACE_PHANTOM_IMAGE_ID="${ACE_RUNPOD_BASE_CONFIG_DIGEST:-}"
  ACE_PHANTOM_DEFINITION_SHA256="${ACE_RUNPOD_BOOTSTRAP_SHA256:-}"
else
  PINNED_SOURCE="${DEPENDENCY_ROOT}/phantom-ant-${PHANTOM_COMMIT}"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ace_bindings${PYTHONPATH:+:${PYTHONPATH}}"
PHANTOM_BUILD="${STATE_ROOT}/phantom-${PHANTOM_COMMIT}-sm80"
PHANTOM_ARCHIVE="${PHANTOM_BUILD}/lib/libphantom_ordinary.a"
RESULTS_ROOT="${STATE_ROOT}/compile_only_results"
mkdir -p "${RESULTS_ROOT}/runs"
RUN_ROOT="$(mktemp -d "${RESULTS_ROOT}/runs/$(date -u +%Y%m%dT%H%M%SZ)-$$.XXXXXX")"
chmod 0755 "${RUN_ROOT}"
RUN_ID="${RUN_ROOT##*/}"
TOOLCHAIN_RESULTS="${RUN_ROOT}/toolchain"
CKKS2C_RESULTS="${RUN_ROOT}/ckks2c"
ORDINARY_RESULTS="${RUN_ROOT}/ordinary_ckks"
CURRENT_RECORD="${RESULTS_ROOT}/current-${GATE}.json"
LATEST_SUCCESS_RECORD="${RESULTS_ROOT}/latest-success-${GATE}.json"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ACTIVE=0
TIMINGS_FILE="${RUN_ROOT}/phase-timings.tsv"
: >"${TIMINGS_FILE}"

timed_phase() {
  run_timed_phase "${TIMINGS_FILE}" "$@"
}

atomic_json() {
  local target="$1"
  shift
  local temporary
  temporary="$(mktemp "${target}.tmp.XXXXXX")"
  jq "$@" >"${temporary}"
  chmod 0644 "${temporary}"
  mv "${temporary}" "${target}"
}

write_invocation() {
  local target="$1"
  local schema="$2"
  shift 2
  python3 - "${target}" "${schema}" "$@" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

target = Path(sys.argv[1])
schema = sys.argv[2]
argv = sys.argv[3:]
normalized = json.dumps(
    argv, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")
record = {
    "schema_version": schema,
    "argv": argv,
    "normalized_argv_sha256": hashlib.sha256(normalized).hexdigest(),
}
target.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

write_qualification_invocation() {
  write_invocation "${RUN_ROOT}/qualification_invocation.json" \
    "ace.phantom.qualification-invocation/1.0.0" \
    "tools/phantom_gpu/compile_only.sh" "${ORIGINAL_ARGUMENTS[@]}"
}

record_command() {
  local target="$1"
  shift
  printf '%q ' "$@" >>"${target}"
  printf '\n' >>"${target}"
}

write_run_state() {
  local status="$1"
  local exit_code="$2"
  local completed_utc="$3"
  local manifest_path="${4:-}"
  local manifest_sha256="${5:-}"
  atomic_json "${CURRENT_RECORD}" \
    -n \
    --arg run_id "${RUN_ID}" \
    --arg gate "${GATE}" \
    --arg status "${status}" \
    --arg run_root "${RUN_ROOT}" \
    --arg started_utc "${STARTED_UTC}" \
    --arg completed_utc "${completed_utc}" \
    --arg image_id "${ACE_PHANTOM_IMAGE_ID:-}" \
    --arg definition_sha256 "${ACE_PHANTOM_DEFINITION_SHA256:-}" \
    --arg manifest_path "${manifest_path}" \
    --arg manifest_sha256 "${manifest_sha256}" \
    --argjson exit_code "${exit_code}" \
    '{
      run_id: $run_id,
      gate: $gate,
      status: $status,
      exit_code: $exit_code,
      run_root: $run_root,
      started_utc: $started_utc,
      completed_utc: (if $completed_utc == "" then null else $completed_utc end),
      development_image_id: $image_id,
      development_definition_sha256: $definition_sha256,
      manifest_path: (if $manifest_path == "" then null else $manifest_path end),
      manifest_sha256:
        (if $manifest_sha256 == "" then null else $manifest_sha256 end)
    }'
}

write_artifact_manifest() {
  [[ "${GATE}" == "ordinary" ]] || return 0
  local ace_commit
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    ace_commit="${ACE_PHANTOM_ACE_COMMIT}"
  else
    ace_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  fi
  python3 - "${RUN_ROOT}" "${SCRIPT_DIR}" "${ace_commit}" \
    "${PHANTOM_COMMIT}" "${SOURCE_MODE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
script_dir = Path(sys.argv[2])
ace_commit, phantom_commit, source_mode = sys.argv[3:6]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

excluded = {"artifact_manifest.json", "manifest.json", "SHA256SUMS"}
files = {
    str(path.relative_to(run_root)): digest(path)
    for path in sorted(run_root.rglob("*"))
    if path.is_file() and str(path.relative_to(run_root)) not in excluded
}
required_paths = {
    "qualification_invocation.json",
    "compiler_invocation.json",
    "ckks2c/add_mul_rotate.cu",
    "ckks2c/compiler_context_manifest.json",
    "ckks2c/compiler_resource_manifest.json",
    "ckks2c/ordinary_ckks_post_ckks.air",
    "ckks2c/source_audit.json",
    "ckks2c/configuration.json",
    "ckks2c/environment.txt",
    "ckks2c/link-commands.txt",
    "ckks2c/qualification.json",
    "ckks2c/cuda_elf.txt",
    "ckks2c/cuda_resources.txt",
    "ckks2c/undefined_symbols.txt",
    "ckks2c/adapter_archive_members.txt",
    "ckks2c/common_archive_members.txt",
    "ckks2c/provider_archive_members.txt",
    "ckks2c/adapter_archive_symbols.txt",
    "ckks2c/common_archive_symbols.txt",
    "ckks2c/provider_archive_symbols.txt",
    "ckks2c/linked_binary_symbols.txt",
    "ckks2c/native_bootstrap_symbols.txt",
    "ckks2c/archive-member-audit.json",
    "ordinary_ckks/ordinary_ckks_v1.json",
    "ordinary_ckks/fixture-validation.json",
    "ordinary_ckks/static-rejection-coverage.json",
    "ordinary_ckks/keyless-source-audit.json",
    "ordinary_ckks/link-commands.txt",
    "ordinary_ckks/ordinary_ckks_analytic_reference.json",
    "ordinary_ckks/ordinary_ckks_analytic_values.bin",
    "ordinary_ckks/ordinary_ckks_ant_raw.json",
    "ordinary_ckks/ordinary_ckks_ant_results.json",
    "ordinary_ckks/ordinary_ckks_ant_values.bin",
    "ordinary_ckks/ordinary_ckks_cpu_reference.json",
    "ordinary_ckks/ordinary_ckks_cpu_values.bin",
    "ordinary_ckks/ordinary_ckks_ant_verification.json",
    "ordinary_ckks/runner-symbols.txt",
    "ordinary_ckks/forbidden-runner-symbols.txt",
    "ordinary_ckks/host-qualification.json",
}
required_paths.add(
    "ace_source_manifest.json"
    if source_mode == "snapshot"
    else "ace_tracked_sources.sha256"
)
if source_mode == "snapshot":
    required_paths.add("phantom_source_manifest.json")
missing = sorted(required_paths - files.keys())
if missing:
    raise SystemExit(
        "artifact relationship inputs are missing: " + ", ".join(missing)
    )
qualification_invocation = json.loads(
    (run_root / "qualification_invocation.json").read_text(encoding="utf-8")
)
compiler_invocation = json.loads(
    (run_root / "compiler_invocation.json").read_text(encoding="utf-8")
)
host = json.loads(
    (run_root / "ordinary_ckks/host-qualification.json").read_text(
        encoding="utf-8"
    )
)
record = {
    "schema_version": "ace.phantom.ordinary-artifacts/1.1.0",
    "status": "bound",
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "source_mode": source_mode,
    "source_manifests": (
        {
            "ace_sha256": files["ace_source_manifest.json"],
            "phantom_sha256": files["phantom_source_manifest.json"],
        }
        if source_mode == "snapshot"
        else {
            "ace_sha256": files["ace_tracked_sources.sha256"],
            "phantom_sha256": None,
        }
    ),
    "normalized_qualification_argv_sha256": qualification_invocation[
        "normalized_argv_sha256"
    ],
    "normalized_compiler_command_sha256": compiler_invocation[
        "normalized_argv_sha256"
    ],
    "post_ckks_air_sha256": files[
        "ckks2c/ordinary_ckks_post_ckks.air"
    ],
    "production_sources": {
        path: files[path]
        for path in (
            "ckks2c/add_mul_rotate.cu",
            "ckks2c/ordinary_runtime_symbols.cu",
            "ordinary_ckks/ordinary_ckks_keyless_probe.cu",
        )
    },
    "host_binaries": {
        path: files[path]
        for path in (
            "ckks2c/add_mul_rotate_sm80",
            "ckks2c/native_phantom_health_sm80",
            "ordinary_ckks/ordinary_ckks_gpu_runner_sm80",
            "ordinary_ckks/ordinary_ckks_keyless_runner_sm80",
            "ordinary_ckks/ordinary_ckks_ant_oracle",
        )
    },
    "harness_sources": {
        path: digest(script_dir / path.removeprefix("tools/phantom_gpu/"))
        for path in (
            "tools/phantom_gpu/harness/native_phantom_health.cu",
            "tools/phantom_gpu/harness/ordinary_ckks_ant_oracle.cxx",
            "tools/phantom_gpu/harness/ordinary_ckks_gpu_runner.cu",
        )
    },
    "phantom_producer": host["phantom_producer"],
    "ant_producer": host["ant_producer"],
    "files": files,
}
(run_root / "artifact_manifest.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

handle_exit() {
  local exit_code="$?"
  trap - EXIT
  if [[ ${RUN_ACTIVE} -eq 1 ]]; then
    set +e
    write_run_state failed "${exit_code}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "" ""
  fi
  exit "${exit_code}"
}

write_manifest() {
  local completed_utc
  completed_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    cp "${ACE_PHANTOM_SOURCE_MANIFEST}" "${RUN_ROOT}/ace_source_manifest.json"
    cp "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST}" \
      "${RUN_ROOT}/phantom_source_manifest.json"
  else
    (
      cd "${REPO_ROOT}"
      git ls-files -z |
        LC_ALL=C sort -z |
        xargs -0 sha256sum
    ) >"${RUN_ROOT}/ace_tracked_sources.sha256"
  fi
  write_artifact_manifest
  (
    cd "${RUN_ROOT}"
    find . -type f ! -name manifest.json ! -name SHA256SUMS -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum
  ) >"${RUN_ROOT}/SHA256SUMS"

  local sums_sha256
  local context_manifest_sha256=""
  local resource_manifest_sha256=""
  local qualification_invocation_sha256=""
  local qualification_argv_sha256=""
  local compiler_invocation_sha256=""
  local compiler_argv_sha256=""
  local post_ckks_air_sha256=""
  local host_qualification_sha256=""
  local artifact_manifest_sha256=""
  local ordinary_fixture_sha256=""
  local ordinary_cpu_reference_sha256=""
  local ordinary_cpu_values_sha256=""
  local source_manifest_sha256
  local provider_source_manifest_sha256=""
  local tracked_diff_sha256
  local worktree_status_sha256
  local worktree_dirty=false
  local ace_commit
  local source_manifest_name
  sums_sha256="$(sha256sum "${RUN_ROOT}/SHA256SUMS" | awk '{print $1}')"
  if [[ -s "${CKKS2C_RESULTS}/compiler_context_manifest.json" ]]; then
    context_manifest_sha256="$(
      sha256sum "${CKKS2C_RESULTS}/compiler_context_manifest.json" | awk '{print $1}'
    )"
  fi
  if [[ -s "${CKKS2C_RESULTS}/compiler_resource_manifest.json" ]]; then
    resource_manifest_sha256="$(sha256sum "${CKKS2C_RESULTS}/compiler_resource_manifest.json" | awk '{print $1}')"
  fi
  if [[ -s "${RUN_ROOT}/qualification_invocation.json" ]]; then
    qualification_invocation_sha256="$(sha256sum "${RUN_ROOT}/qualification_invocation.json" | awk '{print $1}')"
    qualification_argv_sha256="$(jq -er .normalized_argv_sha256 "${RUN_ROOT}/qualification_invocation.json")"
  fi
  if [[ -s "${RUN_ROOT}/compiler_invocation.json" ]]; then
    compiler_invocation_sha256="$(sha256sum "${RUN_ROOT}/compiler_invocation.json" | awk '{print $1}')"
    compiler_argv_sha256="$(jq -er .normalized_argv_sha256 "${RUN_ROOT}/compiler_invocation.json")"
  fi
  if [[ -s "${CKKS2C_RESULTS}/ordinary_ckks_post_ckks.air" ]]; then
    post_ckks_air_sha256="$(sha256sum "${CKKS2C_RESULTS}/ordinary_ckks_post_ckks.air" | awk '{print $1}')"
  fi
  if [[ -s "${ORDINARY_RESULTS}/host-qualification.json" ]]; then
    host_qualification_sha256="$(sha256sum "${ORDINARY_RESULTS}/host-qualification.json" | awk '{print $1}')"
  fi
  if [[ -s "${RUN_ROOT}/artifact_manifest.json" ]]; then
    artifact_manifest_sha256="$(
      sha256sum "${RUN_ROOT}/artifact_manifest.json" | awk '{print $1}'
    )"
  fi
  if [[ -s "${ORDINARY_RESULTS}/ordinary_ckks_v1.json" ]]; then
    ordinary_fixture_sha256="$(
      sha256sum "${ORDINARY_RESULTS}/ordinary_ckks_v1.json" | awk '{print $1}'
    )"
    ordinary_cpu_reference_sha256="$(
      sha256sum "${ORDINARY_RESULTS}/ordinary_ckks_cpu_reference.json" | awk '{print $1}'
    )"
    ordinary_cpu_values_sha256="$(
      sha256sum "${ORDINARY_RESULTS}/ordinary_ckks_cpu_values.bin" | awk '{print $1}'
    )"
  fi
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    source_manifest_sha256="$(sha256sum "${RUN_ROOT}/ace_source_manifest.json" | awk '{print $1}')"
    tracked_diff_sha256="$(printf '' | sha256sum | awk '{print $1}')"
    worktree_status_sha256="${tracked_diff_sha256}"
    ace_commit="${ACE_PHANTOM_ACE_COMMIT}"
    source_manifest_name="ace_source_manifest.json"
    provider_source_manifest_sha256="$(sha256sum "${RUN_ROOT}/phantom_source_manifest.json" | awk '{print $1}')"
  else
    source_manifest_sha256="$(
      sha256sum "${RUN_ROOT}/ace_tracked_sources.sha256" | awk '{print $1}'
    )"
    tracked_diff_sha256="$(
      git -C "${REPO_ROOT}" diff --binary HEAD | sha256sum | awk '{print $1}'
    )"
    worktree_status_sha256="$(
      git -C "${REPO_ROOT}" status --porcelain=v1 -z --untracked-files=all |
        sha256sum |
        awk '{print $1}'
    )"
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
      worktree_dirty=true
    fi
    ace_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    source_manifest_name="ace_tracked_sources.sha256"
  fi
  atomic_json "${RUN_ROOT}/manifest.json" \
    -n \
    --arg run_id "${RUN_ID}" \
    --arg gate "${GATE}" \
    --arg started_utc "${STARTED_UTC}" \
    --arg completed_utc "${completed_utc}" \
    --arg ace_commit "${ace_commit}" \
    --arg phantom_commit "${PHANTOM_COMMIT}" \
    --arg source_mode "${SOURCE_MODE}" \
    --arg image_id "${ACE_PHANTOM_IMAGE_ID}" \
    --arg definition_sha256 "${ACE_PHANTOM_DEFINITION_SHA256}" \
    --arg context_manifest_sha256 "${context_manifest_sha256}" \
    --arg resource_manifest_sha256 "${resource_manifest_sha256}" \
    --arg qualification_invocation_sha256 "${qualification_invocation_sha256}" \
    --arg qualification_argv_sha256 "${qualification_argv_sha256}" \
    --arg compiler_invocation_sha256 "${compiler_invocation_sha256}" \
    --arg compiler_argv_sha256 "${compiler_argv_sha256}" \
    --arg post_ckks_air_sha256 "${post_ckks_air_sha256}" \
    --arg host_qualification_sha256 "${host_qualification_sha256}" \
    --arg artifact_manifest_sha256 "${artifact_manifest_sha256}" \
    --arg ordinary_fixture_sha256 "${ordinary_fixture_sha256}" \
    --arg ordinary_cpu_reference_sha256 "${ordinary_cpu_reference_sha256}" \
    --arg ordinary_cpu_values_sha256 "${ordinary_cpu_values_sha256}" \
    --arg sums_sha256 "${sums_sha256}" \
    --arg source_manifest_sha256 "${source_manifest_sha256}" \
    --arg source_manifest_name "${source_manifest_name}" \
    --arg provider_source_manifest_sha256 "${provider_source_manifest_sha256}" \
    --arg tracked_diff_sha256 "${tracked_diff_sha256}" \
    --arg worktree_status_sha256 "${worktree_status_sha256}" \
    --argjson worktree_dirty "${worktree_dirty}" \
    '{
      status: "pass",
      run_id: $run_id,
      gate: $gate,
      started_utc: $started_utc,
      completed_utc: $completed_utc,
      ace_commit: $ace_commit,
      phantom_commit: $phantom_commit,
      source_mode: $source_mode,
      development_image_id: $image_id,
      development_definition_sha256: $definition_sha256,
      compiler_context_manifest_sha256:
        (if $context_manifest_sha256 == "" then null else $context_manifest_sha256 end),
      compiler_resource_manifest_sha256:
        (if $resource_manifest_sha256 == "" then null else $resource_manifest_sha256 end),
      qualification_invocation_sha256:
        (if $qualification_invocation_sha256 == "" then null else $qualification_invocation_sha256 end),
      normalized_qualification_argv_sha256:
        (if $qualification_argv_sha256 == "" then null else $qualification_argv_sha256 end),
      compiler_invocation_sha256:
        (if $compiler_invocation_sha256 == "" then null else $compiler_invocation_sha256 end),
      normalized_compiler_command_sha256:
        (if $compiler_argv_sha256 == "" then null else $compiler_argv_sha256 end),
      post_ckks_air_sha256:
        (if $post_ckks_air_sha256 == "" then null else $post_ckks_air_sha256 end),
      host_qualification_sha256:
        (if $host_qualification_sha256 == "" then null else $host_qualification_sha256 end),
      artifact_manifest_sha256:
        (if $artifact_manifest_sha256 == "" then null else $artifact_manifest_sha256 end),
      ordinary_fixture_sha256:
        (if $ordinary_fixture_sha256 == "" then null else $ordinary_fixture_sha256 end),
      ordinary_cpu_reference_sha256:
        (if $ordinary_cpu_reference_sha256 == "" then null else $ordinary_cpu_reference_sha256 end),
      ordinary_cpu_values_sha256:
        (if $ordinary_cpu_values_sha256 == "" then null else $ordinary_cpu_values_sha256 end),
      ace_worktree_dirty: $worktree_dirty,
      ace_tracked_diff_sha256: $tracked_diff_sha256,
      ace_worktree_status_sha256: $worktree_status_sha256,
      ace_tracked_source_manifest: $source_manifest_name,
      ace_tracked_source_manifest_sha256: $source_manifest_sha256,
      phantom_source_manifest:
        (if $provider_source_manifest_sha256 == "" then null else "phantom_source_manifest.json" end),
      phantom_source_manifest_sha256:
        (if $provider_source_manifest_sha256 == "" then null else $provider_source_manifest_sha256 end),
      evidence_sha256_manifest: "SHA256SUMS",
      evidence_sha256_manifest_sha256: $sums_sha256,
      link_mode: "manual_static_closure",
      installed_cmake_target_qualified: false,
      host_ant_oracle_was_run: ($ordinary_cpu_reference_sha256 != ""),
      gpu_executables_were_run: false
    }'
  local manifest_sha256
  manifest_sha256="$(sha256sum "${RUN_ROOT}/manifest.json" | awk '{print $1}')"
  write_run_state pass 0 "${completed_utc}" \
    "${RUN_ROOT}/manifest.json" "${manifest_sha256}"
  local latest_temporary
  latest_temporary="$(mktemp "${LATEST_SUCCESS_RECORD}.tmp.XXXXXX")"
  cp "${CURRENT_RECORD}" "${latest_temporary}"
  chmod 0644 "${latest_temporary}"
  mv "${latest_temporary}" "${LATEST_SUCCESS_RECORD}"
  RUN_ACTIVE=0
}

mkdir -p "${TOOLCHAIN_RESULTS}" "${CKKS2C_RESULTS}"
exec 9>"${RESULTS_ROOT}/compile-only.lock"
if ! flock -n 9; then
  echo "another compile-only qualification is already running" >&2
  exit 1
fi
RUN_ACTIVE=1
write_run_state running 0 "" "" ""
trap handle_exit EXIT
write_qualification_invocation

require_readonly_mount() {
  local target="$1"
  if ! findmnt -T "${target}" -n -o OPTIONS |
      tr "," "\n" |
      grep -Fxq ro; then
    echo "required read-only mount is absent: ${target}" >&2
    exit 1
  fi
}

require_environment() {
  if [[ ! -f /.dockerenv ]]; then
    echo "qualification must run inside the development container" >&2
    exit 1
  fi
  if [[ "${ACE_PHANTOM_TOOLCHAIN:-}" != "12.4.1-sm80" ]]; then
    echo "unexpected development image toolchain identity" >&2
    exit 1
  fi
  if [[ "${CMAKE_CUDA_ARCHITECTURES:-}" != "${CUDA_ARCHITECTURES}" ]]; then
    echo "CUDA architecture environment does not match the dependency lock" >&2
    exit 1
  fi
  if [[ ! "${ACE_PHANTOM_IMAGE_ID:-}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "development image ID is absent or malformed" >&2
    exit 1
  fi
  if [[ ! "${ACE_PHANTOM_DEFINITION_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "development image definition hash is absent or malformed" >&2
    exit 1
  fi
  for command in ar cmake c++ git ninja python3 rg file readelf nm jq flock; do
    command -v "${command}" >/dev/null
  done
  test -x "${NVCC}"
  test -x "${CUOBJDUMP}"
  test -r "${CUDA_ROOT}/include/cuda_runtime.h"
  test -r "${CUDA_ROOT}/lib64/libcudadevrt.a"
  test -r "${CUDA_ROOT}/lib64/libcudart.so"
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    [[ "${ACE_RUNPOD_BASE_IMAGE:-}" == "${CUDA_IMAGE}" ]]
    [[ "${ACE_PHANTOM_ACE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]]
    [[ "${ACE_PHANTOM_SOURCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]
    [[ -f "${ACE_PHANTOM_SOURCE_MANIFEST:-}" ]]
    [[ "$(sha256sum "${ACE_PHANTOM_SOURCE_MANIFEST}" | awk '{print $1}')" == \
       "${ACE_PHANTOM_SOURCE_MANIFEST_SHA256}" ]]
    [[ "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]
    [[ -f "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST:-}" ]]
    [[ "$(sha256sum "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST}" | awk '{print $1}')" == \
       "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256}" ]]
  else
    require_readonly_mount "${PHANTOM_MOUNT}"
    require_readonly_mount "${MODELS_MOUNT}"
    require_readonly_mount "${DATASET_MOUNT}"
  fi
}

prepare_pinned_source() {
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    if [[ -e "${PINNED_SOURCE}/.git" ]]; then
      echo "Phantom source snapshot unexpectedly contains Git metadata" >&2
      exit 1
    fi
    for required in CMakeLists.txt include src; do
      test -e "${PINNED_SOURCE}/${required}"
    done
    if find "${PINNED_SOURCE}" -type f -name '._*' -print -quit | grep -q .; then
      echo "Phantom source snapshot contains AppleDouble files" >&2
      exit 1
    fi
    return
  fi
  mkdir -p "${DEPENDENCY_ROOT}"
  if [[ ! -e "${PINNED_SOURCE}" ]]; then
    local clone_arguments=(
      clone
      --no-local
      "${PHANTOM_MOUNT}"
      "${PINNED_SOURCE}"
    )
    git "${clone_arguments[@]}"
    git -C "${PINNED_SOURCE}" checkout --detach "${PHANTOM_COMMIT}"
  fi
  if [[ ! -d "${PINNED_SOURCE}/.git" ]]; then
    echo "pinned Phantom source path exists but is not a Git clone" >&2
    exit 1
  fi
  if [[ "$(git -C "${PINNED_SOURCE}" rev-parse HEAD)" != "${PHANTOM_COMMIT}" ]]; then
    echo "pinned Phantom source clone has the wrong commit" >&2
    exit 1
  fi
  if [[ -n "$(git -C "${PINNED_SOURCE}" status --porcelain --untracked-files=all)" ]]; then
    echo "pinned Phantom source clone is not clean" >&2
    exit 1
  fi
  if find "${PINNED_SOURCE}" -type f -name '._*' -print -quit | grep -q .; then
    echo "pinned Phantom source clone contains AppleDouble files" >&2
    exit 1
  fi
}

record_common_configuration() {
  local result_dir="$1"
  local context_manifest="$2"
  mkdir -p "${result_dir}"
  local arguments=(
    --repo-root "${REPO_ROOT}"
    --models-dir "${MODELS_MOUNT}"
    --dataset-dir "${DATASET_MOUNT}"
    --phantom-dir "${PHANTOM_MOUNT}"
    --context-manifest "${context_manifest}"
    --json-output "${result_dir}/configuration.json"
  )
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    arguments+=(
      --source-mode snapshot
      --ace-commit "${ACE_PHANTOM_ACE_COMMIT}"
      --source-manifest-sha256 "${ACE_PHANTOM_SOURCE_MANIFEST_SHA256}"
      --base-image "${ACE_RUNPOD_BASE_IMAGE}"
      --base-config-digest "${ACE_RUNPOD_BASE_CONFIG_DIGEST}"
      --bootstrap-sha256 "${ACE_RUNPOD_BOOTSTRAP_SHA256}"
    )
  fi
  python3 "${SCRIPT_DIR}/check_configuration.py" "${arguments[@]}"
  bash "${SCRIPT_DIR}/collect_environment.sh" "${result_dir}/environment.txt"
}

configure_phantom() {
  local arguments=(
    -S "${PINNED_SOURCE}"
    -B "${PHANTOM_BUILD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES=80
    -DCMAKE_CUDA_STANDARD=17
    -DCMAKE_CUDA_STANDARD_REQUIRED=ON
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DPHANTOM_BUILD_EXAMPLES=OFF
    -DPHANTOM_BUILD_TESTS=OFF
  )
  cmake "${arguments[@]}"
  cmake --build "${PHANTOM_BUILD}" --target phantom_ordinary --parallel "${BUILD_JOBS}"
  test -s "${PHANTOM_ARCHIVE}"
  rg '^CMAKE_CUDA_ARCHITECTURES:[A-Z_]+=80$' "${PHANTOM_BUILD}/CMakeCache.txt"
}

inspect_binary() {
  local binary="$1"
  local result_dir="$2"
  "${CUOBJDUMP}" --list-elf "${binary}" >"${result_dir}/cuda_elf.txt"
  rg 'sm_80' "${result_dir}/cuda_elf.txt"
  "${CUOBJDUMP}" --dump-resource-usage "${binary}" >"${result_dir}/cuda_resources.txt"
  file "${binary}" >"${result_dir}/file.txt"
  readelf -h -S -Ws -d "${binary}" >"${result_dir}/readelf.txt"
  nm -A -C --undefined-only "${binary}" >"${result_dir}/undefined_symbols.txt"
}

audit_production_archive_members() {
  local report="$1"
  shift
  python3 - "${report}" "$@" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

report = Path(sys.argv[1])
inventories = [Path(value) for value in sys.argv[2:]]
forbidden = re.compile(
    r"(?:bootstrap(?:per|_3|_precom(?:pute)?)?|phantom_bootstrap|"
    r"coeff(?:icient)?[_-]?to[_-]?slot|slot[_-]?to[_-]?coeff|"
    r"eval[_-]?mod|modular[_-]?reducer|cnn_phantom|conv_eval|"
    r"fhert_(?:ant|poly)|(?:^|[/_.-])(?:ant|poly|poly2c|ckks2poly|stage)(?:[/_.-]|$)|"
    r"native.*precom|precom.*native|"
    r"(?:bootstrap|eval[_-]?mod|coeff.*slot|slot.*coeff).*stage|"
    r"stage.*(?:bootstrap|eval[_-]?mod|coeff.*slot|slot.*coeff))",
    re.IGNORECASE,
)
records = []
matches = []
for inventory in inventories:
    if not inventory.is_file():
        raise SystemExit(f"archive member inventory is missing: {inventory}")
    members = inventory.read_text(encoding="utf-8").splitlines()
    if not members or any(not member for member in members):
        raise SystemExit(f"archive member inventory is empty or malformed: {inventory}")
    for member in members:
        if forbidden.search(member):
            matches.append({"inventory": inventory.name, "member": member})
    records.append(
        {
            "path": inventory.name,
            "sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
            "member_count": len(members),
        }
    )
value = {
    "schema_version": "ace.phantom.production-archive-members/1.0.0",
    "status": "fail" if matches else "pass",
    "inventories": records,
    "forbidden_member_count": len(matches),
    "forbidden_members": matches,
}
report.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
if matches:
    raise SystemExit("production archive member inventory contains forbidden objects")
PY
}

run_toolchain_gate() {
  require_environment
  prepare_pinned_source
  timed_phase phantom_build configure_phantom

  nm -A -C --defined-only "${PHANTOM_ARCHIVE}" >"${TOOLCHAIN_RESULTS}/phantom_archive_symbols.txt"
  ar t "${PHANTOM_ARCHIVE}" >"${TOOLCHAIN_RESULTS}/phantom_archive_members.txt"
  audit_production_archive_members \
    "${TOOLCHAIN_RESULTS}/archive-member-audit.json" \
    "${TOOLCHAIN_RESULTS}/phantom_archive_members.txt"
  rg 'phantom::arith::CoeffModulus::Create' "${TOOLCHAIN_RESULTS}/phantom_archive_symbols.txt"
  if rg 'Bootstrapper|Phantom_bootstrap|Eval_bootstrap|bootstrap_3|cnn_phantom|conv_eval|FHErt_(ant|poly)|fhe::(ant|poly)|CoeffToSlot|SlotToCoeff|EvalMod|Native.*[Pp]recom|[Pp]recom.*Native|Bootstrap.*[Ss]tage|[Ss]tage.*Bootstrap' \
      "${TOOLCHAIN_RESULTS}/phantom_archive_symbols.txt" \
      >"${TOOLCHAIN_RESULTS}/forbidden_provider_symbols.txt"; then
    echo "primitive-only Phantom archive contains excluded symbols" >&2
    exit 1
  fi

  local archive_sha
  archive_sha="$(sha256sum "${PHANTOM_ARCHIVE}" | awk '{print $1}')"
  local report_arguments=(
    -n
    --arg status pass
    --arg gate toolchain
    --arg architecture sm_80
    --arg phantom_commit "${PHANTOM_COMMIT}"
    --arg phantom_archive_sha256 "${archive_sha}"
    --arg image_id "${ACE_PHANTOM_IMAGE_ID}"
    --arg definition_sha256 "${ACE_PHANTOM_DEFINITION_SHA256}"
  )
  atomic_json "${TOOLCHAIN_RESULTS}/qualification.json" \
    "${report_arguments[@]}" '{
      status: $status,
      gate: $gate,
      architecture: $architecture,
      phantom_commit: $phantom_commit,
      phantom_archive_sha256: $phantom_archive_sha256,
      archive_member_inventory_audit: "pass",
      development_image_id: $image_id,
      development_definition_sha256: $definition_sha256,
      provider_archive_build: "pass",
      executable_was_run: false
    }'
  echo "toolchain and Phantom archive build inspection passed"
}

configure_ace() {
  local arguments=(
    -S "${REPO_ROOT}/fhe-cmplr"
    -B "${ACE_BUILD}"
    -G Ninja
    "-DFHE_WITH_SRC=air-infra;nn-addon"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_ROOT}"
    -DCMAKE_CUDA_ARCHITECTURES=80
    -DCMAKE_CUDA_STANDARD=17
    -DCMAKE_CUDA_STANDARD_REQUIRED=ON
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DFHE_ENABLE_CUDA=ON
    -DFHE_ENABLE_PHANTOM=ON
    -DPHANTOM_SOURCE_DIR="${PINNED_SOURCE}"
    -DPHANTOM_GIT_TAG="${PHANTOM_COMMIT}"
    -DFHE_ENABLE_SEAL=OFF
    -DFHE_ENABLE_SEAL_BTS=OFF
    -DFHE_ENABLE_OPENFHE=OFF
    -DBUILD_UNITTEST=ON
    -DBUILD_BENCH=OFF
    -DFHE_BUILD_TEST=ON
    -DFHE_BUILD_EXAMPLE=OFF
    -DFHE_CODE_CHECK=OFF
    -DAIR_CODE_CHECK=OFF
    -DNN_CODE_CHECK=OFF
  )
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    arguments+=(
      -DACE_SOURCE_COMMIT="${ACE_PHANTOM_ACE_COMMIT}"
      -DPHANTOM_SOURCE_SNAPSHOT=ON
    )
  fi
  cmake "${arguments[@]}"
  cmake --build "${ACE_BUILD}" --target install --parallel "${BUILD_JOBS}"
}

configure_bindings() {
  local onnx_proto_dir="${ACE_BUILD}/nn-addon/onnx2air"
  if [[ ! -f "${onnx_proto_dir}/onnx.pb.h" &&
        -f "${onnx_proto_dir}/onnx/onnx.pb.h" ]]; then
    onnx_proto_dir="${onnx_proto_dir}/onnx"
  fi
  local arguments=(
    -S "${REPO_ROOT}/bindings"
    -B "${BINDINGS_BUILD}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DACE_COMPILER_DIR="${REPO_ROOT}"
    -DACE_INSTALL_DIR="${INSTALL_ROOT}"
    -DONNX_PROTO_DIR="${onnx_proto_dir}"
    -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  )
  cmake "${arguments[@]}"
  cmake --build "${BINDINGS_BUILD}" \
    --target air_builder nn_addon fhe_cmplr passmanager \
    --parallel "${BUILD_JOBS}"
}

run_compiler_tests() {
  local codegen_tests=(
    ace_edsl/tests/test_air_pass_pipeline.py
    ace_edsl/tests/test_ckks2c_codegen.py
    tools/phantom_gpu/tests/test_a100_evidence_archives.py
    tools/phantom_gpu/tests/test_codegen_tools.py
    tools/phantom_gpu/tests/test_ordinary_ckks_fixture.py
    tools/phantom_gpu/tests/test_ordinary_runtime_source.py
    tools/phantom_gpu/tests/test_runpod_pipeline.py
  )
  (
    cd "${REPO_ROOT}"
    python3 -m pytest -q "${codegen_tests[@]}"
  ) 2>&1 | tee "${CKKS2C_RESULTS}/pytest_codegen.txt"
  local regression_tests=(
    ace_edsl/tests/test_air_array.py
    ace_edsl/tests/test_ckks_extended_rewrite.py
    ace_edsl/tests/test_ckks_complex_encode.py
    ace_edsl/tests/test_bootstrap_stage_ops.py
  )
  (
    cd "${REPO_ROOT}"
    python3 -m pytest -q "${regression_tests[@]}"
  ) 2>&1 | tee "${CKKS2C_RESULTS}/pytest_regressions.txt"
  local bootstrap_source_tests=(
    ace_edsl/tests/test_bootstrap_full.py::TestBootstrapFull::test_primitive_air_invariants_before_and_after_ckks_driver
    ace_edsl/tests/test_bootstrap_full.py::TestBootstrapFull::test_generated_c_contains_bootstrap_ops
  )
  (
    cd "${REPO_ROOT}"
    python3 -m pytest -q "${bootstrap_source_tests[@]}"
  ) 2>&1 | tee "${CKKS2C_RESULTS}/pytest_bootstrap_source.txt"

  local ctest_pattern
  ctest_pattern='^(test_fheckks_01|ut_fheckks|test_fhepoly_add_float|test_fhepoly_mul_float|test_fhepoly_relin_01|test_fhepoly_relin_02|test_fhepoly_rotate_01|test_fhepoly_rotate_02|test_fhepoly_rotate_03|test_fhepoly_rotate_04|ut_fhepoly|test_fhedriver_01|test_fherot_01|ut_fhedriver|test_fhe_01)$'
  ctest --test-dir "${ACE_BUILD}" --output-on-failure -R "${ctest_pattern}" 2>&1 |
    tee "${CKKS2C_RESULTS}/ctest.txt"

}

audit_removed_profile_path() {
  local report="${CKKS2C_RESULTS}/removed-profile-path-audit.txt"
  : >"${report}"
  if rg -n \
      'fullpacked_bts_v1\.json|phantom_ordinary_profile\.h|profile_sha256|check_ordinary_profile\.py|check_phantom_profile' \
      "${REPO_ROOT}/fhe-cmplr" "${REPO_ROOT}/tools/phantom_gpu" \
      --glob '!compile_only.sh' >"${report}"; then
    echo "obsolete Phantom profile path remains active" >&2
    return 1
  fi
}

write_static_rejection_coverage() {
  local fixture="$1"
  local output="$2"
  python3 - "${fixture}" \
    "${REPO_ROOT}/fhe-cmplr/ckks/unittest/test_ckks2c_ordinary_contract.cxx" \
    "${CKKS2C_RESULTS}/ctest.txt" "${output}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

fixture_path, source_path, ctest_path, output_path = map(Path, sys.argv[1:5])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate fixture key in static coverage: {key}")
        value[key] = item
    return value

fixture = json.loads(
    fixture_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
source = source_path.read_text(encoding="utf-8")
ctest = ctest_path.read_text(encoding="utf-8")
if "100% tests passed" not in ctest:
    raise SystemExit("focused CTest log does not report a complete pass")
mapping = {
    "encode_invalid_length": (
        "CKKS2COrdinaryAirVerifier", "RejectsEncodeLengthBeyondContextSlots"
    ),
    "encode_invalid_level": (
        "CKKS2COrdinaryAirVerifier", "RejectsEncodeLevelBeyondContextQCount"
    ),
    "encode_invalid_scale": (
        "CKKS2COrdinaryAirVerifier", "RejectsUnrepresentableEncodeScale"
    ),
    "incompatible_level": (
        "CKKS2COrdinaryAirVerifier", "RejectsVisibleAddLevelMismatch"
    ),
    "incompatible_scale": (
        "CKKS2COrdinaryAirVerifier", "RejectsVisibleSubScaleMismatch"
    ),
    "missing_declared_relin_key": (
        "CKKS2COrdinaryContract", "RejectsRelinWithoutKeyRequirement"
    ),
    "missing_declared_rotation_key": (
        "CKKS2COrdinaryContract", "RejectsRotateWithoutKeyRequirement"
    ),
    "reverse_scalar_subtraction": (
        "CKKS2COrdinaryAirVerifier", "RejectsReverseScalarCipherForm"
    ),
}
static_cases = [
    item for item in fixture["rejections"] if item["phase"] == "static"
]
if {item["id"] for item in static_cases} != set(mapping):
    raise SystemExit("fixture static rejections do not match executable coverage")
records = []
for item in static_cases:
    suite, test = mapping[item["id"]]
    token = f"TEST_F({suite}, {test})" if suite.endswith("Verifier") else f"TEST({suite}, {test})"
    if token not in source:
        raise SystemExit(f"static rejection test is absent: {suite}.{test}")
    records.append({
        "case_id": item["id"],
        "diagnostic_id": item["diagnostic_id"],
        "gtest": f"{suite}.{test}",
        "status": "pass",
    })
record = {
    "schema_version": "ace.phantom.ordinary-static-rejections/1.0.0",
    "status": "pass",
    "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    "test_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "ctest_log_sha256": hashlib.sha256(ctest_path.read_bytes()).hexdigest(),
    "cases": records,
}
output_path.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

run_native_terminal_probes() {
  local model="${CKKS2C_RESULTS}/native_terminal_probe.onnx"
  local positive_source="${CKKS2C_RESULTS}/native_ckks_phantom.cu"
  local negative_source="${CKKS2C_RESULTS}/native_poly_phantom.cu"
  local parameter_source="${CKKS2C_RESULTS}/native_parameter_flow.cu"
  local positive_stdout="${CKKS2C_RESULTS}/native_ckks_phantom.stdout.txt"
  local positive_stderr="${CKKS2C_RESULTS}/native_ckks_phantom.stderr.txt"
  local negative_stdout="${CKKS2C_RESULTS}/native_poly_phantom.stdout.txt"
  local negative_stderr="${CKKS2C_RESULTS}/native_poly_phantom.stderr.txt"
  local parameter_stdout="${CKKS2C_RESULTS}/native_parameter_flow.stdout.txt"
  local parameter_stderr="${CKKS2C_RESULTS}/native_parameter_flow.stderr.txt"
  local commands_file="${CKKS2C_RESULTS}/terminal_selection_commands.txt"
  local ckks_context_option
  local security_context_option
  local parameter_context_option

  ckks_context_option="-CKKS:hw=${COMPILER_HAMMING_WEIGHT}:q0=${COMPILER_FIRST_PRIME_BITS}:sf=${COMPILER_SCALING_BITS}:N=${COMPILER_POLY_DEGREE}:icl=${COMPILER_INPUT_LEVEL}:mcl=${COMPILER_MUL_LEVEL}"
  security_context_option="-FHE_SCHEME:sec_lev=${COMPILER_SECURITY_LEVEL}"
  parameter_context_option="-CKKS:hw=192:q0=60:sf=56:N=65536:icl=1:mcl=4"

  python3 "${SCRIPT_DIR}/generate_native_terminal_probe.py" --output "${model}"
  local positive_command=(
    "${INSTALL_ROOT}/bin/fhe_cmplr"
    -FHE:codegen_ir=ckks
    -K2C:lib=phantom
    "${ckks_context_option}"
    "${security_context_option}"
    -o "${positive_source}"
    "${model}"
  )
  local negative_command=(
    "${INSTALL_ROOT}/bin/fhe_cmplr"
    -FHE:codegen_ir=poly
    -P2C:lib=phantom
    "${ckks_context_option}"
    "${security_context_option}"
    -o "${negative_source}"
    "${model}"
  )
  local parameter_command=(
    "${INSTALL_ROOT}/bin/fhe_cmplr"
    -FHE:codegen_ir=ckks
    -K2C:lib=phantom
    -VEC:ms=32768
    "${parameter_context_option}"
    -FHE_SCHEME:sec_lev=0
    -o "${parameter_source}"
    "${model}"
  )
  {
    printf '%q ' "${positive_command[@]}"
    printf '\n'
    printf '%q ' "${negative_command[@]}"
    printf '\n'
    printf '%q ' "${parameter_command[@]}"
    printf '\n'
  } >"${commands_file}"

  local positive_exit_code
  if "${positive_command[@]}" >"${positive_stdout}" 2>"${positive_stderr}"; then
    positive_exit_code=0
  else
    positive_exit_code=$?
  fi
  if [[ ${positive_exit_code} -ne 0 || ! -s "${positive_source}" ]]; then
    echo "native CKKS-to-Phantom terminal probe failed" >&2
    exit 1
  fi
  python3 "${SCRIPT_DIR}/check_primitive_codegen.py" \
    "${positive_source}" \
    --report "${CKKS2C_RESULTS}/native_ckks_phantom_audit.json" \
    --required-token '#include "rt_phantom/rt_phantom.h"' \
    --required-token 'Get_phantom_context_manifest()' \
    --required-token 'Get_phantom_resource_manifest()' \
    --required-token 'Add_ciph('

  if ! "${parameter_command[@]}" \
      >"${parameter_stdout}" 2>"${parameter_stderr}"; then
    echo "compiler-option parameter-flow probe failed" >&2
    exit 1
  fi
  python3 "${SCRIPT_DIR}/check_primitive_codegen.py" \
    "${parameter_source}" \
    --report "${CKKS2C_RESULTS}/native_parameter_flow_audit.json" \
    --required-token '#include "rt_phantom/rt_phantom.h"' \
    --required-token 'Get_phantom_context_manifest()' \
    --required-token 'Get_phantom_resource_manifest()' \
    --required-token 'Add_ciph('
  jq -e '
    .emitted_context.polynomial_degree == 65536 and
    .emitted_context.logical_slot_capacity == 32768 and
    .emitted_context.data_q_bit_sizes == [60, 56, 56, 56] and
    (.emitted_context.special_p_bit_sizes | length) > 0 and
    (.emitted_context.special_p_bit_sizes | all(. == 60)) and
    .emitted_context.input_level == 1 and
    .emitted_context.q_part_count > 0 and
    .emitted_context.hamming_weight == 192 and
    .emitted_context.security_level == 0 and
    .emitted_context.first_modulus_bits == 60 and
    .emitted_context.scaling_modulus_bits == 56
  ' "${CKKS2C_RESULTS}/native_parameter_flow_audit.json" >/dev/null

  if [[ -e "${negative_source}" ]]; then
    echo "negative terminal probe output unexpectedly exists before the run" >&2
    exit 1
  fi
  local negative_exit_code
  if "${negative_command[@]}" >"${negative_stdout}" 2>"${negative_stderr}"; then
    negative_exit_code=0
  else
    negative_exit_code=$?
  fi
  if [[ ${negative_exit_code} -ne 1 ]]; then
    echo "POLY-to-Phantom terminal probe returned ${negative_exit_code}, expected 1" >&2
    exit 1
  fi
  if [[ -e "${negative_source}" ]]; then
    echo "rejected POLY-to-Phantom route created source output" >&2
    exit 1
  fi
  if ! rg -Fq \
    'Phantom source generation requires -FHE:codegen_ir=ckks' \
    "${negative_stdout}" "${negative_stderr}"; then
    echo "POLY-to-Phantom rejection diagnostic was not emitted" >&2
    exit 1
  fi

  local model_sha
  local source_sha
  model_sha="$(sha256sum "${model}" | awk '{print $1}')"
  source_sha="$(sha256sum "${positive_source}" | awk '{print $1}')"
  atomic_json "${CKKS2C_RESULTS}/terminal_selection.json" \
    -n \
    --arg status pass \
    --arg model_sha256 "${model_sha}" \
    --arg source_sha256 "${source_sha}" \
    --argjson positive_exit_code "${positive_exit_code}" \
    --argjson negative_exit_code "${negative_exit_code}" \
    '{
      status: $status,
      model_sha256: $model_sha256,
      positive_exit_code: $positive_exit_code,
      positive_source_sha256: $source_sha256,
      positive_source_audit: "pass",
      negative_exit_code: $negative_exit_code,
      negative_output_created: false,
      negative_diagnostic_matched: true
    }'
}

link_generated_probe() {
  local source="${CKKS2C_RESULTS}/add_mul_rotate.cu"
  local post_ckks_air="${CKKS2C_RESULTS}/ordinary_ckks_post_ckks.air"
  local context_manifest="${CKKS2C_RESULTS}/compiler_context_manifest.json"
  local resource_manifest="${CKKS2C_RESULTS}/compiler_resource_manifest.json"
  local object="${CKKS2C_RESULTS}/add_mul_rotate.o"
  local symbol_source="${CKKS2C_RESULTS}/ordinary_runtime_symbols.cu"
  local symbol_object="${CKKS2C_RESULTS}/ordinary_runtime_symbols.o"
  local main_object="${CKKS2C_RESULTS}/generated_link_main.o"
  local device_link="${CKKS2C_RESULTS}/add_mul_rotate.dlink.o"
  local binary="${CKKS2C_RESULTS}/add_mul_rotate_sm80"
  local health_object="${CKKS2C_RESULTS}/native_phantom_health.o"
  local health_device_link="${CKKS2C_RESULTS}/native_phantom_health.dlink.o"
  local health_binary="${CKKS2C_RESULTS}/native_phantom_health_sm80"
  local runtime_build="${ACE_BUILD}/rtlib/build"
  local adapter_archive="${runtime_build}/phantom/libFHErt_phantom.a"
  local common_archive="${runtime_build}/common/libFHErt_common.a"
  local external_source="${runtime_build}/external/src/phantom_external"
  local external_archive="${runtime_build}/external/src/phantom_external-build/lib/libphantom_ordinary.a"
  local nvcc_common=(-std=c++17 -arch=sm_80 -rdc=true)

  test -s "${adapter_archive}"
  test -s "${common_archive}"
  test -s "${external_archive}"
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    external_source="${PINNED_SOURCE}"
    test ! -e "${external_source}/.git"
  else
    test "$(git -C "${external_source}" rev-parse HEAD)" = "${PHANTOM_COMMIT}"
  fi

  local compiler_arguments=(
    --output ckks2c/add_mul_rotate.cu
    --post-ckks-air ckks2c/ordinary_ckks_post_ckks.air
    --context-manifest ckks2c/compiler_context_manifest.json
    --resource-manifest ckks2c/compiler_resource_manifest.json
    --poly-degree "${COMPILER_POLY_DEGREE}" \
    --mul-level "${COMPILER_MUL_LEVEL}" \
    --input-level "${COMPILER_INPUT_LEVEL}" \
    --security-level "${COMPILER_SECURITY_LEVEL}" \
    --scaling-factor-bits "${COMPILER_SCALING_BITS}" \
    --first-prime-bits "${COMPILER_FIRST_PRIME_BITS}" \
    --hamming-weight "${COMPILER_HAMMING_WEIGHT}"
  )
  write_invocation "${RUN_ROOT}/compiler_invocation.json" \
    "ace.phantom.compiler-invocation/1.0.0" \
    tools/phantom_gpu/generate_ckks2c_probe.py "${compiler_arguments[@]}"
  (
    cd "${RUN_ROOT}"
    python3 "${SCRIPT_DIR}/generate_ckks2c_probe.py" \
      "${compiler_arguments[@]}"
  )
  python3 "${SCRIPT_DIR}/check_primitive_codegen.py" \
    "${source}" \
    --context-manifest "${context_manifest}" \
    --report "${CKKS2C_RESULTS}/source_audit.json"
  record_common_configuration "${CKKS2C_RESULTS}" "${context_manifest}"
  local context_manifest_sha
  context_manifest_sha="$(sha256sum "${context_manifest}" | awk '{print $1}')"

  local contract_binary="${CKKS2C_RESULTS}/ordinary_contract_test"
  local -a contract_context_arguments
  local contract_context_output
  contract_context_output="$(
    jq -er '
      [.polynomial_degree, .logical_slot_capacity,
       (.data_q_bit_sizes | length), .scaling_modulus_bits]
      | .[] | tostring
    ' "${context_manifest}"
  )"
  mapfile -t contract_context_arguments <<<"${contract_context_output}"
  [[ ${#contract_context_arguments[@]} -eq 4 ]]
  c++ -std=c++17 \
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/phantom/include" \
    "${SCRIPT_DIR}/tests/ordinary_contract_test.cc" \
    -o "${contract_binary}"
  "${contract_binary}" "${contract_context_arguments[@]}"
  atomic_json "${CKKS2C_RESULTS}/ordinary-contract-test.json" -n \
    --arg status pass \
    --arg context_manifest_sha256 "${context_manifest_sha}" \
    --argjson arguments "$(printf '%s\n' "${contract_context_arguments[@]}" | jq -Rsc 'split("\n")[:-1] | map(tonumber)')" \
    '{status:$status, executable:"ordinary_contract_test",
      context_manifest_sha256:$context_manifest_sha256,
      context_derived_arguments:$arguments}'

  local compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "${source}"
    -o "${object}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    "${NVCC}" "${compile_arguments[@]}"
  "${NVCC}" "${compile_arguments[@]}"
  python3 "${SCRIPT_DIR}/generate_ordinary_runtime_symbols.py" \
    --output "${symbol_source}"
  python3 "${SCRIPT_DIR}/tests/test_ordinary_runtime_source.py" \
    --source "${symbol_source}" \
    --gpu-runner-source "${SCRIPT_DIR}/harness/ordinary_ckks_gpu_runner.cu"
  local symbol_compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "${symbol_source}"
    -o "${symbol_object}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    "${NVCC}" "${symbol_compile_arguments[@]}"
  "${NVCC}" "${symbol_compile_arguments[@]}"
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    c++ -std=c++17 \
      -I"${REPO_ROOT}/fhe-cmplr/rtlib/include" \
      -c "${SCRIPT_DIR}/harness/generated_link_main.cc" \
      -o "${main_object}"
  c++ -std=c++17 \
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include" \
    -c "${SCRIPT_DIR}/harness/generated_link_main.cc" \
    -o "${main_object}"

  local device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${object}"
    "${symbol_object}"
    "${adapter_archive}"
    "${external_archive}"
    "${common_archive}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${device_link}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    "${NVCC}" "${device_link_arguments[@]}"
  "${NVCC}" "${device_link_arguments[@]}"

  local host_link_arguments=(
    -std=c++17
    "${main_object}"
    "${object}"
    "${symbol_object}"
    "${device_link}"
    -Wl,--start-group
    "${adapter_archive}"
    "${external_archive}"
    "${common_archive}"
    -lntl
    -lgmpxx
    -lgmp
    -Wl,--end-group
    -L"${CUDA_ROOT}/lib64"
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -lcudart
    -pthread
    -fopenmp
    -ldl
    -lrt
    -lm
    -o "${binary}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    c++ "${host_link_arguments[@]}"
  c++ "${host_link_arguments[@]}"

  inspect_binary "${binary}" "${CKKS2C_RESULTS}"

  local health_compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "-DACE_CONTEXT_MANIFEST_SHA256=\"${context_manifest_sha}\""
    "${SCRIPT_DIR}/harness/native_phantom_health.cu"
    -o "${health_object}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    "${NVCC}" "${health_compile_arguments[@]}"
  "${NVCC}" "${health_compile_arguments[@]}"
  local health_device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${object}"
    "${health_object}"
    "${adapter_archive}"
    "${external_archive}"
    "${common_archive}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${health_device_link}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    "${NVCC}" "${health_device_link_arguments[@]}"
  "${NVCC}" "${health_device_link_arguments[@]}"
  local health_host_link_arguments=(
    -std=c++17
    "${health_object}"
    "${object}"
    "${health_device_link}"
    -Wl,--start-group
    "${adapter_archive}"
    "${external_archive}"
    "${common_archive}"
    -lntl
    -lgmpxx
    -lgmp
    -Wl,--end-group
    -L"${CUDA_ROOT}/lib64"
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -lcudart
    -pthread
    -fopenmp
    -ldl
    -lrt
    -lm
    -o "${health_binary}"
  )
  record_command "${CKKS2C_RESULTS}/link-commands.txt" \
    c++ "${health_host_link_arguments[@]}"
  c++ "${health_host_link_arguments[@]}"
  mkdir -p "${CKKS2C_RESULTS}/native_health_inspection"
  inspect_binary "${health_binary}" "${CKKS2C_RESULTS}/native_health_inspection"
  nm -A -C --defined-only "${adapter_archive}" \
    >"${CKKS2C_RESULTS}/adapter_archive_symbols.txt"
  nm -A -C --defined-only "${external_archive}" \
    >"${CKKS2C_RESULTS}/provider_archive_symbols.txt"
  nm -A -C --defined-only "${common_archive}" \
    >"${CKKS2C_RESULTS}/common_archive_symbols.txt"
  ar t "${adapter_archive}" >"${CKKS2C_RESULTS}/adapter_archive_members.txt"
  ar t "${external_archive}" >"${CKKS2C_RESULTS}/provider_archive_members.txt"
  ar t "${common_archive}" >"${CKKS2C_RESULTS}/common_archive_members.txt"
  audit_production_archive_members \
    "${CKKS2C_RESULTS}/archive-member-audit.json" \
    "${CKKS2C_RESULTS}/adapter_archive_members.txt" \
    "${CKKS2C_RESULTS}/provider_archive_members.txt" \
    "${CKKS2C_RESULTS}/common_archive_members.txt"
  nm -A -C --defined-only "${binary}" \
    >"${CKKS2C_RESULTS}/linked_binary_symbols.txt"
  if rg 'Bootstrapper|Phantom_bootstrap|Eval_bootstrap|bootstrap_3|cnn_phantom|conv_eval|FHErt_(ant|poly)|fhe::(ant|poly)|CoeffToSlot|SlotToCoeff|EvalMod|Native.*[Pp]recom|[Pp]recom.*Native|Bootstrap.*[Ss]tage|[Ss]tage.*Bootstrap' \
      "${CKKS2C_RESULTS}/adapter_archive_symbols.txt" \
      "${CKKS2C_RESULTS}/provider_archive_symbols.txt" \
      "${CKKS2C_RESULTS}/common_archive_symbols.txt" \
      "${CKKS2C_RESULTS}/linked_binary_symbols.txt" \
      >"${CKKS2C_RESULTS}/native_bootstrap_symbols.txt"; then
    echo "ordinary production closure contains native-bootstrap symbols" >&2
    exit 1
  fi

  local source_sha
  local binary_sha
  local resource_manifest_sha
  local health_binary_sha
  local terminal_selection_sha
  local adapter_archive_sha provider_archive_sha common_archive_sha
  source_sha="$(sha256sum "${source}" | awk '{print $1}')"
  binary_sha="$(sha256sum "${binary}" | awk '{print $1}')"
  resource_manifest_sha="$(sha256sum "${resource_manifest}" | awk '{print $1}')"
  health_binary_sha="$(sha256sum "${health_binary}" | awk '{print $1}')"
  terminal_selection_sha="$(
    sha256sum "${CKKS2C_RESULTS}/terminal_selection.json" | awk '{print $1}'
  )"
  adapter_archive_sha="$(sha256sum "${adapter_archive}" | awk '{print $1}')"
  provider_archive_sha="$(sha256sum "${external_archive}" | awk '{print $1}')"
  common_archive_sha="$(sha256sum "${common_archive}" | awk '{print $1}')"
  local report_arguments=(
    -n
    --arg status pass
    --arg gate ckks2c
    --arg architecture sm_80
    --arg phantom_commit "${PHANTOM_COMMIT}"
    --arg source_sha256 "${source_sha}"
    --arg binary_sha256 "${binary_sha}"
    --arg health_binary_sha256 "${health_binary_sha}"
    --arg context_manifest_sha256 "${context_manifest_sha}"
    --arg resource_manifest_sha256 "${resource_manifest_sha}"
    --arg terminal_selection_sha256 "${terminal_selection_sha}"
    --arg adapter_archive_sha256 "${adapter_archive_sha}"
    --arg provider_archive_sha256 "${provider_archive_sha}"
    --arg common_archive_sha256 "${common_archive_sha}"
    --arg image_id "${ACE_PHANTOM_IMAGE_ID}"
    --arg definition_sha256 "${ACE_PHANTOM_DEFINITION_SHA256}"
    --argjson production_archive_contains_native_bootstrap false
  )
  atomic_json "${CKKS2C_RESULTS}/qualification.json" \
    "${report_arguments[@]}" '{
      status: $status,
      gate: $gate,
      architecture: $architecture,
      phantom_commit: $phantom_commit,
      source_sha256: $source_sha256,
      binary_sha256: $binary_sha256,
      health_binary_sha256: $health_binary_sha256,
      compiler_context_manifest_sha256: $context_manifest_sha256,
      compiler_resource_manifest_sha256: $resource_manifest_sha256,
      terminal_selection_sha256: $terminal_selection_sha256,
      adapter_archive_sha256: $adapter_archive_sha256,
      provider_archive_sha256: $provider_archive_sha256,
      common_archive_sha256: $common_archive_sha256,
      development_image_id: $image_id,
      development_definition_sha256: $definition_sha256,
      production_archive_contains_native_bootstrap:
        $production_archive_contains_native_bootstrap,
      archive_member_inventory_audit: "pass",
      primitive_only_provider_archive: true,
      generated_source_contains_native_bootstrap: false,
      link_mode: "manual_static_closure",
      installed_cmake_target_qualified: false,
      executable_was_run: false
    }'
  echo "generated CKKS2C source compile, device-link, host-link, and inspection passed"
}

build_ordinary_conformance() {
  mkdir -p "${ORDINARY_RESULTS}"
  local runtime_build="${ACE_BUILD}/rtlib/build"
  local adapter_archive="${runtime_build}/phantom/libFHErt_phantom.a"
  local common_archive="${runtime_build}/common/libFHErt_common.a"
  local ant_archive="${runtime_build}/ant/libFHErt_ant.a"
  local ant_encode_archive="${runtime_build}/ant/libFHErt_ant_encode.a"
  local external_source="${runtime_build}/external/src/phantom_external"
  local provider_archive="${runtime_build}/external/src/phantom_external-build/lib/libphantom_ordinary.a"
  local source="${CKKS2C_RESULTS}/add_mul_rotate.cu"
  local source_object="${CKKS2C_RESULTS}/add_mul_rotate.o"
  local context_manifest="${CKKS2C_RESULTS}/compiler_context_manifest.json"
  local resource_manifest="${CKKS2C_RESULTS}/compiler_resource_manifest.json"
  local checked_fixture="${SCRIPT_DIR}/fixtures/ordinary_ckks_v1.json"
  local fixture="${ORDINARY_RESULTS}/ordinary_ckks_v1.json"
  local post_ckks_air="${CKKS2C_RESULTS}/ordinary_ckks_post_ckks.air"
  local runner_object="${ORDINARY_RESULTS}/ordinary_ckks_gpu_runner.o"
  local runner_device_link="${ORDINARY_RESULTS}/ordinary_ckks_gpu_runner.dlink.o"
  local runner_binary="${ORDINARY_RESULTS}/ordinary_ckks_gpu_runner_sm80"
  local keyless_source="${ORDINARY_RESULTS}/ordinary_ckks_keyless_probe.cu"
  local keyless_context="${ORDINARY_RESULTS}/ordinary_ckks_keyless_context.json"
  local keyless_resources="${ORDINARY_RESULTS}/ordinary_ckks_keyless_resources.json"
  local keyless_object="${ORDINARY_RESULTS}/ordinary_ckks_keyless_probe.o"
  local keyless_device_link="${ORDINARY_RESULTS}/ordinary_ckks_keyless_runner.dlink.o"
  local keyless_binary="${ORDINARY_RESULTS}/ordinary_ckks_keyless_runner_sm80"
  local ant_object="${ORDINARY_RESULTS}/ordinary_ckks_ant_oracle.o"
  local ant_binary="${ORDINARY_RESULTS}/ordinary_ckks_ant_oracle"
  local context_sha fixture_sha producer_id ace_commit
  local nvcc_common=(-std=c++17 -arch=sm_80 -rdc=true)

  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    external_source="${PINNED_SOURCE}"
    ace_commit="${ACE_PHANTOM_ACE_COMMIT}"
  else
    ace_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  fi
  producer_id="ace-${ace_commit}-phantom-${PHANTOM_COMMIT}"
  cp -- "${checked_fixture}" "${fixture}"
  context_sha="$(sha256sum "${context_manifest}" | awk '{print $1}')"
  fixture_sha="$(sha256sum "${fixture}" | awk '{print $1}')"

  python3 "${SCRIPT_DIR}/ordinary_ckks_fixture.py" validate-fixture \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --compiler-invocation "${RUN_ROOT}/compiler_invocation.json" \
    --post-ckks-air "${post_ckks_air}" \
    >"${ORDINARY_RESULTS}/fixture-validation.json"
  write_static_rejection_coverage \
    "${fixture}" "${ORDINARY_RESULTS}/static-rejection-coverage.json"

  local runner_compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "-DACE_CONTEXT_MANIFEST_SHA256=\"${context_sha}\""
    "-DACE_FIXTURE_SHA256=\"${fixture_sha}\""
    "-DACE_PRODUCER_ID=\"${producer_id}\""
    "${SCRIPT_DIR}/harness/ordinary_ckks_gpu_runner.cu"
    -o "${runner_object}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    "${NVCC}" "${runner_compile_arguments[@]}"
  "${NVCC}" "${runner_compile_arguments[@]}"

  local runner_device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${source_object}"
    "${runner_object}"
    "${adapter_archive}"
    "${provider_archive}"
    "${common_archive}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${runner_device_link}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    "${NVCC}" "${runner_device_link_arguments[@]}"
  "${NVCC}" "${runner_device_link_arguments[@]}"

  local runner_host_link_arguments=(
    -std=c++17
    "${runner_object}"
    "${source_object}"
    "${runner_device_link}"
    -Wl,--start-group
    "${adapter_archive}"
    "${provider_archive}"
    "${common_archive}"
    -lntl -lgmpxx -lgmp
    -Wl,--end-group
    -L"${CUDA_ROOT}/lib64"
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
    -lcudadevrt -lcudart -pthread -fopenmp -ldl -lrt -lm
    -o "${runner_binary}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    c++ "${runner_host_link_arguments[@]}"
  c++ "${runner_host_link_arguments[@]}"
  mkdir -p "${ORDINARY_RESULTS}/runner-inspection"
  inspect_binary "${runner_binary}" "${ORDINARY_RESULTS}/runner-inspection"

  python3 "${SCRIPT_DIR}/generate_ckks2c_probe.py" \
    --output "${keyless_source}" \
    --context-manifest "${keyless_context}" \
    --resource-manifest "${keyless_resources}" \
    --poly-degree "${COMPILER_POLY_DEGREE}" \
    --mul-level "${COMPILER_MUL_LEVEL}" \
    --input-level "${COMPILER_INPUT_LEVEL}" \
    --security-level "${COMPILER_SECURITY_LEVEL}" \
    --scaling-factor-bits "${COMPILER_SCALING_BITS}" \
    --first-prime-bits "${COMPILER_FIRST_PRIME_BITS}" \
    --hamming-weight "${COMPILER_HAMMING_WEIGHT}" \
    --resource-mode keyless
  cmp "${context_manifest}" "${keyless_context}"
  jq -e '
    .schema_version == 2 and
    .context_schema_version == 1 and
    .relinearization_key == false and
    .rotation_steps == [] and
    .conjugation_key == false and
    .rotate_batch == false and
    .rotation_batches == [] and
    .raise_mod == false and
    .monomial_powers == []
  ' \
    "${keyless_resources}" >/dev/null
  python3 "${SCRIPT_DIR}/check_primitive_codegen.py" \
    "${keyless_source}" \
    --context-manifest "${keyless_context}" \
    --report "${ORDINARY_RESULTS}/keyless-source-audit.json" \
    --required-token '#include "rt_phantom/rt_phantom.h"' \
    --required-token 'Get_phantom_context_manifest()' \
    --required-token 'Get_phantom_resource_manifest()' \
    --required-token 'Rotate_ciph(' \
    --required-token 'Copy_ciph('
  local keyless_compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "${keyless_source}"
    -o "${keyless_object}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    "${NVCC}" "${keyless_compile_arguments[@]}"
  "${NVCC}" "${keyless_compile_arguments[@]}"
  local keyless_device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${keyless_object}"
    "${runner_object}"
    "${adapter_archive}"
    "${provider_archive}"
    "${common_archive}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${keyless_device_link}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    "${NVCC}" "${keyless_device_link_arguments[@]}"
  "${NVCC}" "${keyless_device_link_arguments[@]}"
  local keyless_host_link_arguments=(
    -std=c++17
    "${runner_object}"
    "${keyless_object}"
    "${keyless_device_link}"
    -Wl,--start-group
    "${adapter_archive}"
    "${provider_archive}"
    "${common_archive}"
    -lntl -lgmpxx -lgmp
    -Wl,--end-group
    -L"${CUDA_ROOT}/lib64"
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
    -lcudadevrt -lcudart -pthread -fopenmp -ldl -lrt -lm
    -o "${keyless_binary}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    c++ "${keyless_host_link_arguments[@]}"
  c++ "${keyless_host_link_arguments[@]}"
  mkdir -p "${ORDINARY_RESULTS}/keyless-runner-inspection"
  inspect_binary "${keyless_binary}" \
    "${ORDINARY_RESULTS}/keyless-runner-inspection"

  local ant_compile_arguments=(
    -std=c++17
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/ant/include"
    "-DACE_CONTEXT_MANIFEST_SHA256=\"${context_sha}\""
    "-DACE_FIXTURE_SHA256=\"${fixture_sha}\""
    "-DACE_PRODUCER_ID=\"${producer_id}-ant\""
    -c "${SCRIPT_DIR}/harness/ordinary_ckks_ant_oracle.cxx"
    -o "${ant_object}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    c++ "${ant_compile_arguments[@]}"
  c++ "${ant_compile_arguments[@]}"
  local ant_link_arguments=(
    -std=c++17
    "${ant_object}"
    -Wl,--start-group
    "${ant_archive}"
    "${ant_encode_archive}"
    "${common_archive}"
    -lntl -lgmpxx -lgmp
    -Wl,--end-group
    -pthread -fopenmp -ldl -lrt -lm
    -o "${ant_binary}"
  )
  record_command "${ORDINARY_RESULTS}/link-commands.txt" \
    c++ "${ant_link_arguments[@]}"
  c++ "${ant_link_arguments[@]}"

  RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 "${ant_binary}" \
    "${fixture}" "${context_manifest}" "${resource_manifest}" \
    "${ORDINARY_RESULTS}/ordinary_ckks_ant_raw.json" \
    disable-native-bootstrap-precompute \
    >"${ORDINARY_RESULTS}/ordinary_ckks_ant.stdout.txt" \
    2>"${ORDINARY_RESULTS}/ordinary_ckks_ant.stderr.txt"
  python3 "${SCRIPT_DIR}/ordinary_ckks_fixture.py" ingest-raw-provider \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --provider ant \
    --raw-json "${ORDINARY_RESULTS}/ordinary_ckks_ant_raw.json" \
    --output-json "${ORDINARY_RESULTS}/ordinary_ckks_ant_results.json" \
    --output-bin "${ORDINARY_RESULTS}/ordinary_ckks_ant_values.bin"
  python3 "${SCRIPT_DIR}/ordinary_ckks_fixture.py" generate-analytic \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --output-json "${ORDINARY_RESULTS}/ordinary_ckks_analytic_reference.json" \
    --output-bin "${ORDINARY_RESULTS}/ordinary_ckks_analytic_values.bin"
  python3 "${SCRIPT_DIR}/ordinary_ckks_fixture.py" ingest-ant \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --analytic-json "${ORDINARY_RESULTS}/ordinary_ckks_analytic_reference.json" \
    --analytic-bin "${ORDINARY_RESULTS}/ordinary_ckks_analytic_values.bin" \
    --ant-json "${ORDINARY_RESULTS}/ordinary_ckks_ant_results.json" \
    --ant-bin "${ORDINARY_RESULTS}/ordinary_ckks_ant_values.bin" \
    --output-json "${ORDINARY_RESULTS}/ordinary_ckks_cpu_reference.json" \
    --output-bin "${ORDINARY_RESULTS}/ordinary_ckks_cpu_values.bin"
  python3 "${SCRIPT_DIR}/ordinary_ckks_fixture.py" verify-ant-reference \
    --fixture "${fixture}" --context-manifest "${context_manifest}" \
    --cpu-json "${ORDINARY_RESULTS}/ordinary_ckks_cpu_reference.json" \
    --cpu-bin "${ORDINARY_RESULTS}/ordinary_ckks_cpu_values.bin" \
    --output-json "${ORDINARY_RESULTS}/ordinary_ckks_ant_verification.json"

  nm -A -C --defined-only "${runner_binary}" \
    >"${ORDINARY_RESULTS}/runner-symbols.txt"
  if rg 'Bootstrapper|Phantom_bootstrap|Eval_bootstrap|bootstrap_3|cnn_phantom|conv_eval|FHErt_(ant|poly)|fhe::(ant|poly)|CoeffToSlot|SlotToCoeff|EvalMod|Native.*[Pp]recom|[Pp]recom.*Native|Bootstrap.*[Ss]tage|[Ss]tage.*Bootstrap' \
      "${ORDINARY_RESULTS}/runner-symbols.txt" \
      >"${ORDINARY_RESULTS}/forbidden-runner-symbols.txt"; then
    echo "ordinary conformance runner contains forbidden provider symbols" >&2
    return 1
  fi
  local ace_source_manifest_sha256=""
  local phantom_source_manifest_sha256=""
  if [[ "${SOURCE_MODE}" == "snapshot" ]]; then
    ace_source_manifest_sha256="${ACE_PHANTOM_SOURCE_MANIFEST_SHA256}"
    phantom_source_manifest_sha256="${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256}"
  fi
  atomic_json "${ORDINARY_RESULTS}/host-qualification.json" -n \
    --arg status pass \
    --arg context_sha256 "${context_sha}" \
    --arg fixture_sha256 "${fixture_sha}" \
    --arg resource_sha256 "$(sha256sum "${resource_manifest}" | awk '{print $1}')" \
    --arg runner_sha256 "$(sha256sum "${runner_binary}" | awk '{print $1}')" \
    --arg keyless_runner_sha256 "$(sha256sum "${keyless_binary}" | awk '{print $1}')" \
    --arg generated_source_sha256 "$(sha256sum "${source}" | awk '{print $1}')" \
    --arg runner_source_sha256 "$(sha256sum "${SCRIPT_DIR}/harness/ordinary_ckks_gpu_runner.cu" | awk '{print $1}')" \
    --arg keyless_source_sha256 "$(sha256sum "${keyless_source}" | awk '{print $1}')" \
    --arg native_health_source_sha256 "$(sha256sum "${SCRIPT_DIR}/harness/native_phantom_health.cu" | awk '{print $1}')" \
    --arg ant_source_sha256 "$(sha256sum "${SCRIPT_DIR}/harness/ordinary_ckks_ant_oracle.cxx" | awk '{print $1}')" \
    --arg ant_binary_sha256 "$(sha256sum "${ant_binary}" | awk '{print $1}')" \
    --arg producer_id "${producer_id}" \
    --arg ace_commit "${ace_commit}" \
    --arg phantom_commit "${PHANTOM_COMMIT}" \
    --arg qualification_invocation_sha256 "$(sha256sum "${RUN_ROOT}/qualification_invocation.json" | awk '{print $1}')" \
    --arg qualification_argv_sha256 "$(jq -er .normalized_argv_sha256 "${RUN_ROOT}/qualification_invocation.json")" \
    --arg compiler_invocation_sha256 "$(sha256sum "${RUN_ROOT}/compiler_invocation.json" | awk '{print $1}')" \
    --arg compiler_argv_sha256 "$(jq -er .normalized_argv_sha256 "${RUN_ROOT}/compiler_invocation.json")" \
    --arg post_ckks_air_sha256 "$(sha256sum "${post_ckks_air}" | awk '{print $1}')" \
    --arg ace_source_manifest_sha256 "${ace_source_manifest_sha256}" \
    --arg phantom_source_manifest_sha256 "${phantom_source_manifest_sha256}" \
    --arg cpu_reference_sha256 "$(sha256sum "${ORDINARY_RESULTS}/ordinary_ckks_cpu_reference.json" | awk '{print $1}')" \
    --arg cpu_values_sha256 "$(sha256sum "${ORDINARY_RESULTS}/ordinary_ckks_cpu_values.bin" | awk '{print $1}')" \
    '{
      status: $status,
      compiler_context_manifest_sha256: $context_sha256,
      compiler_resource_manifest_sha256: $resource_sha256,
      fixture_sha256: $fixture_sha256,
      runner_sha256: $runner_sha256,
      keyless_runner_sha256: $keyless_runner_sha256,
      generated_source_sha256: $generated_source_sha256,
      runner_source_sha256: $runner_source_sha256,
      keyless_source_sha256: $keyless_source_sha256,
      native_health_source_sha256: $native_health_source_sha256,
      ant_source_sha256: $ant_source_sha256,
      ant_binary_sha256: $ant_binary_sha256,
      ace_commit: $ace_commit,
      phantom_commit: $phantom_commit,
      qualification_invocation_sha256: $qualification_invocation_sha256,
      normalized_qualification_argv_sha256: $qualification_argv_sha256,
      compiler_invocation_sha256: $compiler_invocation_sha256,
      normalized_compiler_command_sha256: $compiler_argv_sha256,
      post_ckks_air_sha256: $post_ckks_air_sha256,
      ace_source_manifest_sha256:
        (if $ace_source_manifest_sha256 == "" then null else $ace_source_manifest_sha256 end),
      phantom_source_manifest_sha256:
        (if $phantom_source_manifest_sha256 == "" then null else $phantom_source_manifest_sha256 end),
      phantom_producer: $producer_id,
      ant_producer: ($producer_id + "-ant"),
      cpu_reference_sha256: $cpu_reference_sha256,
      cpu_values_sha256: $cpu_values_sha256,
      ant_vs_analytic: "pass",
      ant_executable_was_run: true,
      gpu_executables_were_run: false,
      compute_sanitizer_was_run: false,
      primitive_only_provider_archive: true
    }'
}

run_ckks2c_gate() {
  timed_phase toolchain_gate run_toolchain_gate
  mkdir -p "${CKKS2C_RESULTS}"
  timed_phase ace_build configure_ace
  timed_phase bindings_build configure_bindings
  timed_phase tests run_compiler_tests
  timed_phase removed_profile_path_audit audit_removed_profile_path
  timed_phase terminal_selection run_native_terminal_probes
  timed_phase generated_compile_links link_generated_probe
}

run_ordinary_gate() {
  run_ckks2c_gate
  timed_phase ordinary_host_qualification build_ordinary_conformance
}

case "${GATE}" in
  toolchain)
    run_toolchain_gate
    ;;
  ckks2c)
    run_ckks2c_gate
    ;;
  ordinary|all)
    run_ordinary_gate
    ;;
esac

write_manifest
echo "qualification evidence: ${RUN_ROOT}"
