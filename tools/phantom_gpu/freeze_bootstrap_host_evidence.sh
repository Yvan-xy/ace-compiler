#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"
TRANSPORT_HELPERS_LOADED=false
TRANSPORT_HELPERS_COMMIT=""

usage() {
  cat >&2 <<EOF
usage: $0 --ace-commit COMMIT --phantom-commit COMMIT [explicit qualification options] OUTPUT_DIRECTORY
       $0 --finalize OUTPUT_DIRECTORY
The reusable qualification root is
OUTPUT_DIRECTORY/verified-bootstrap-host-result/results/qualification.
Exact container cleanup evidence is written under OUTPUT_DIRECTORY/docker.
Pass --detach with the first form to launch the container without attaching.
After it exits, use the second form to retrieve, verify, clean up, and close evidence.
EOF
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
OUTPUT_ARGUMENT=""
RUN_MODE="synchronous"
QUALIFICATION_ARGUMENTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      [[ "${RUN_MODE}" == "synchronous" ]] || usage
      RUN_MODE="detached-launch"
      shift
      ;;
    --finalize)
      [[ "${RUN_MODE}" == "synchronous" ]] || usage
      RUN_MODE="detached-finalize"
      shift
      ;;
    --ace-commit) ACE_COMMIT="$2"; shift 2 ;;
    --phantom-commit) PHANTOM_COMMIT="$2"; shift 2 ;;
    --poly-degree|--vector-capacity|--mul-level|--input-level|--security-level|\
    --scaling-factor-bits|--first-prime-bits|--hamming-weight|--q-part-count|\
    --encode-transform-budget|--decode-transform-budget|\
    --ciphertext-constant-encoding|--packing|--post-multiply-real|\
    --post-multiply-imag|--post-multiply-scale-degree|--post-rotation-step|\
    --fixture-id|--fixture-seed|--inside-margin|--provider-clear-threshold|\
    --gpu-native-threshold|--gpu-generated-threshold|--repeat-threshold|\
    --host-oracle-timeout-seconds)
      QUALIFICATION_ARGUMENTS+=("$1" "$2")
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
if [[ "${RUN_MODE}" == "detached-finalize" ]]; then
  if [[ -n "${ACE_COMMIT}" || -n "${PHANTOM_COMMIT}" ||
        -z "${OUTPUT_ARGUMENT}" || ${#QUALIFICATION_ARGUMENTS[@]} -ne 0 ]]; then
    usage
  fi
else
  if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
        ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ||
        -z "${OUTPUT_ARGUMENT}" || ${#QUALIFICATION_ARGUMENTS[@]} -eq 0 ]]; then
    usage
  fi
fi

verify_selected_lifecycle_files() {
  local selected_commit="$1"
  local lifecycle_file expected observed
  git -C "${REPO_ROOT}" cat-file -e "${selected_commit}^{commit}"
  for lifecycle_file in \
    tools/phantom_gpu/freeze_bootstrap_host_evidence.sh \
    tools/phantom_gpu/transport_helpers.sh; do
    expected="$(git -C "${REPO_ROOT}" show \
      "${selected_commit}:${lifecycle_file}" | sha256sum | awk '{print $1}')"
    observed="$(sha256sum "${REPO_ROOT}/${lifecycle_file}" | awk '{print $1}')"
    if [[ "${observed}" != "${expected}" ]]; then
      echo "bootstrap host-freeze lifecycle file differs from the selected ACE commit: ${lifecycle_file}" >&2
      return 1
    fi
  done
}

load_selected_transport_helpers() {
  local selected_commit="$1"
  verify_selected_lifecycle_files "${selected_commit}" || return 1
  if [[ "${TRANSPORT_HELPERS_LOADED}" == true ]]; then
    [[ "${TRANSPORT_HELPERS_COMMIT}" == "${selected_commit}" ]] || {
      echo "transport helpers are already bound to a different ACE commit" >&2
      return 1
    }
    return 0
  fi
  source "${SCRIPT_DIR}/transport_helpers.sh"
  TRANSPORT_HELPERS_LOADED=true
  TRANSPORT_HELPERS_COMMIT="${selected_commit}"
}

CONTAINER_ID=""
CONTAINER_NAME=""
TASK_LABEL=""
BASE_IMAGE=""
BASE_CONFIG=""
DETACHED_TERMINAL_OWNERSHIP_VALIDATED=false
DETACHED_CLEANUP_COMPLETE=false
DETACHED_START_ATTEMPTED=false

if [[ "${RUN_MODE}" == "detached-finalize" ]]; then
  OUTPUT="$(realpath -e -- "${OUTPUT_ARGUMENT}")"
  if [[ ! -d "${OUTPUT}" ]]; then
    echo "detached bootstrap host output is not a directory: ${OUTPUT}" >&2
    exit 1
  fi
  PAYLOAD="${OUTPUT}/payload"
  DOCKER_EVIDENCE="${OUTPUT}/docker"
  INCOMING_RESULTS="${OUTPUT}/incoming-results"
else
load_selected_transport_helpers "${ACE_COMMIT}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"

DEPENDENCY_LOCK="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
)"
lock_value_from_text() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" <<<"${DEPENDENCY_LOCK}")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "selected ACE commit has an invalid ${key} dependency lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
BASE_IMAGE="$(lock_value_from_text CUDA_IMAGE)"
BASE_CONFIG="$(lock_value_from_text CUDA_IMAGE_CONFIG)"
LOCKED_PHANTOM_COMMIT="$(lock_value_from_text PHANTOM_COMMIT)"
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
chmod 0700 "${OUTPUT}"
PAYLOAD="${OUTPUT}/payload"
DOCKER_EVIDENCE="${OUTPUT}/docker"
INCOMING_RESULTS="${OUTPUT}/incoming-results"
mkdir -p "${PAYLOAD}" "${DOCKER_EVIDENCE}" "${INCOMING_RESULTS}"
chmod 0700 "${INCOMING_RESULTS}"

ARCHIVE_TOOL_DIRECTORY="$(mktemp -d)"
cleanup_archive_tool() {
  rm -f -- "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"
  rmdir -- "${ARCHIVE_TOOL_DIRECTORY}"
}
trap cleanup_archive_tool EXIT
git -C "${REPO_ROOT}" show \
  "${ACE_COMMIT}:tools/phantom_gpu/source_archive.py" \
  >"${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"
ACE_ARCHIVE="ace-source-${ACE_COMMIT}.tar.gz"
PHANTOM_ARCHIVE="phantom-source-${PHANTOM_COMMIT}.tar.gz"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${REPO_ROOT}" --commit "${ACE_COMMIT}" --kind ace \
  --output "${PAYLOAD}/${ACE_ARCHIVE}" \
  --manifest "${PAYLOAD}/ace-source.manifest.json" \
  >"${PAYLOAD}/ace-source.audit.json"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" --commit "${PHANTOM_COMMIT}" --kind phantom \
  --output "${PAYLOAD}/${PHANTOM_ARCHIVE}" \
  --manifest "${PAYLOAD}/phantom-source.manifest.json" \
  >"${PAYLOAD}/phantom-source.audit.json"
cleanup_archive_tool
trap - EXIT

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" \
    >"${PAYLOAD}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 "${PAYLOAD}/source_archive.py" \
  "${PAYLOAD}/phase_helpers.sh" \
  "${PAYLOAD}/bootstrap_environment.sh"

python3 - "${PAYLOAD}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" \
  "${QUALIFICATION_ARGUMENTS[@]}" <<'PY'
import hashlib
import hashlib
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
arguments = sys.argv[4:]
if len(arguments) % 2:
    raise SystemExit("qualification arguments are not option/value pairs")
qualification_arguments = dict(zip(arguments[0::2], arguments[1::2]))
if len(qualification_arguments) != len(arguments) // 2:
    raise SystemExit("qualification arguments are duplicated")
expected = {
    "--poly-degree", "--vector-capacity", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight", "--q-part-count", "--encode-transform-budget",
    "--decode-transform-budget", "--ciphertext-constant-encoding",
    "--packing", "--post-multiply-real", "--post-multiply-imag",
    "--post-multiply-scale-degree", "--post-rotation-step", "--fixture-id",
    "--fixture-seed", "--inside-margin", "--provider-clear-threshold",
    "--gpu-native-threshold", "--gpu-generated-threshold",
    "--repeat-threshold",
    "--host-oracle-timeout-seconds",
}
if set(qualification_arguments) != expected:
    raise SystemExit("qualification arguments are incomplete or unexpected")
integer_options = {
    "--poly-degree", "--vector-capacity", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight", "--q-part-count", "--encode-transform-budget",
    "--decode-transform-budget", "--fixture-seed",
    "--host-oracle-timeout-seconds",
}
for option in integer_options:
    if re.fullmatch(r"[0-9]+", qualification_arguments[option]) is None:
        raise SystemExit(f"qualification argument {option} is not unsigned")
for option in {"--post-multiply-scale-degree", "--post-rotation-step"}:
    if re.fullmatch(r"-?[0-9]+", qualification_arguments[option]) is None:
        raise SystemExit(f"qualification argument {option} is not signed integer")
if qualification_arguments["--packing"] != "full" or qualification_arguments[
    "--ciphertext-constant-encoding"
] not in {"enabled", "disabled"}:
    raise SystemExit("qualification packing or constant encoding mode is invalid")
for option in {
    "--post-multiply-real", "--post-multiply-imag", "--inside-margin",
    "--provider-clear-threshold", "--gpu-native-threshold",
    "--gpu-generated-threshold", "--repeat-threshold",
}:
    try:
        number = float(qualification_arguments[option])
    except ValueError as error:
        raise SystemExit(f"qualification argument {option} is not numeric") from error
    if not math.isfinite(number):
        raise SystemExit(f"qualification argument {option} is nonfinite")
if not qualification_arguments["--fixture-id"]:
    raise SystemExit("qualification fixture identifier is empty")
numbers = {option: int(qualification_arguments[option]) for option in integer_options}
for option in integer_options - {"--security-level", "--fixture-seed"}:
    if numbers[option] <= 0:
        raise SystemExit(f"qualification argument {option} is not positive")
degree, slots = numbers["--poly-degree"], numbers["--vector-capacity"]
if degree % 2 or slots != degree // 2:
    raise SystemExit("qualification is not full-capacity packed")
if numbers["--security-level"] not in {0, 128, 192, 256}:
    raise SystemExit("qualification security level is unsupported")
if numbers["--first-prime-bits"] < numbers["--scaling-factor-bits"]:
    raise SystemExit("qualification first prime is smaller than scaling bits")
if int(qualification_arguments["--post-multiply-scale-degree"]) != 0:
    raise SystemExit("qualification multiply scale degree must be explicit zero")
rotation = int(qualification_arguments["--post-rotation-step"])
if rotation % slots == 0:
    raise SystemExit("qualification rotation is zero modulo slot capacity")
normalized_rotation = rotation % slots
if normalized_rotation > slots // 2:
    normalized_rotation -= slots
if rotation != normalized_rotation:
    raise SystemExit("qualification rotation is not canonical signed modulo capacity")
real = float(qualification_arguments["--post-multiply-real"])
imaginary = float(qualification_arguments["--post-multiply-imag"])
if real == 0.0 and imaginary == 0.0:
    raise SystemExit("qualification multiply constant is zero")
if float(qualification_arguments["--provider-clear-threshold"]) != 1e-2:
    raise SystemExit("qualification provider-clear threshold changed")
for option in {
    "--inside-margin", "--gpu-native-threshold",
    "--gpu-generated-threshold", "--repeat-threshold",
}:
    if float(qualification_arguments[option]) <= 0.0:
        raise SystemExit(f"qualification argument {option} is not positive")
payload = {
    "schema_version": "ace.phantom.bootstrap-host-freeze-payload/2.0.0",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "contents": "audited-exact-source-snapshots-and-pinned-bootstrap-only",
    "source_snapshots": {
        "ace_manifest_sha256": hashlib.sha256(
            (root / "ace-source.manifest.json").read_bytes()
        ).hexdigest(),
        "phantom_manifest_sha256": hashlib.sha256(
            (root / "phantom-source.manifest.json").read_bytes()
        ).hexdigest(),
    },
    "qualification": {
        "gate": "bootstrap",
        "arguments": arguments,
        "options": qualification_arguments,
    },
    "files": sorted(path.name for path in root.iterdir() if path.is_file()),
}
(root / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
(
  cd "${PAYLOAD}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS
)

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
CONTAINER_NAME="ace-bootstrap-host-freeze-${RUN_ID}"
TASK_LABEL="bootstrap-host-freeze-${RUN_ID}"
fi

remove_exact_container() {
  local exact_id="${CONTAINER_ID}"
  [[ -n "${exact_id}" ]] || return 0
  if ! docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "task-created container disappeared before exact cleanup: ${exact_id}" >&2
    return 1
  fi
  local observed_label observed_name
  observed_label="$(docker inspect --type container \
    -f '{{index .Config.Labels "ace.phantom.task"}}' "${exact_id}")"
  observed_name="$(docker inspect --type container -f '{{.Name}}' "${exact_id}")"
  if [[ "${observed_label}" != "${TASK_LABEL}" ||
        "${observed_name}" != "/${CONTAINER_NAME}" ]]; then
    echo "exact cleanup refused a container with an unexpected identity" >&2
    return 1
  fi
  docker rm -f "${exact_id}" >>"${DOCKER_EVIDENCE}/cleanup.txt"
  if docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "exact cleanup verification failed for ${exact_id}" >&2
    return 1
  fi
  local name_inventory label_inventory
  name_inventory="$(docker ps -aq --no-trunc \
    --filter "name=^/${CONTAINER_NAME}$")"
  label_inventory="$(docker ps -aq --no-trunc \
    --filter "label=ace.phantom.task=${TASK_LABEL}")"
  if [[ -n "${name_inventory}" || -n "${label_inventory}" ]]; then
    echo "exact cleanup left a matching name or task-label inventory" >&2
    return 1
  fi
  {
    printf '%s\tremoved-and-absent\n' "${exact_id}"
    printf '%s\tname-inventory-empty\n' "${CONTAINER_NAME}"
    printf '%s\tlabel-inventory-empty\n' "${TASK_LABEL}"
  } >>"${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  CONTAINER_ID=""
}

ensure_detached_container_cleanup() {
  local cleanup_intent="${DOCKER_EVIDENCE}/cleanup-intent.json"
  local cleanup_result="${DOCKER_EVIDENCE}/cleanup-result.json"
  local state_path="${OUTPUT}/detached-launch-state.json"
  if [[ -e "${cleanup_result}" ]]; then
    if [[ ! -f "${cleanup_result}" || -L "${cleanup_result}" ]]; then
      echo "detached cleanup result receipt is nonregular" >&2
      return 1
    fi
    python3 - "${cleanup_result}" "${CONTAINER_ID}" "${CONTAINER_NAME}" \
      "${TASK_LABEL}" <<'PY'
import json
import os
import pathlib
import stat
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": "ace.phantom.bootstrap-host-cleanup-result/1.0.0",
    "status": "pass",
    "container_id": sys.argv[2],
    "container_name": sys.argv[3],
    "task_label": sys.argv[4],
    "container_inventory": "absent",
    "name_inventory": "empty",
    "task_label_inventory": "empty",
}
if value != expected:
    raise SystemExit("detached cleanup result receipt changed")
PY
    if docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1 ||
       [[ -n "$(docker ps -aq --no-trunc --filter "name=^/${CONTAINER_NAME}$")" ]] ||
       [[ -n "$(docker ps -aq --no-trunc \
          --filter "label=ace.phantom.task=${TASK_LABEL}")" ]]; then
      echo "detached cleanup result no longer proves empty inventories" >&2
      return 1
    fi
    DETACHED_CLEANUP_COMPLETE=true
    return 0
  fi
  if [[ -e "${cleanup_intent}" &&
        ( ! -f "${cleanup_intent}" || -L "${cleanup_intent}" ) ]]; then
    echo "detached cleanup intent is nonregular" >&2
    return 1
  fi
  local existed_before_intent=false
  if docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1; then
    existed_before_intent=true
    local observed_name observed_task observed_intent observed_nonce expected_intent expected_nonce
    observed_name="$(docker inspect --type container -f '{{.Name}}' "${CONTAINER_ID}")"
    observed_task="$(docker inspect --type container \
      -f '{{index .Config.Labels "ace.phantom.task"}}' "${CONTAINER_ID}")"
    observed_intent="$(docker inspect --type container \
      -f '{{index .Config.Labels "ace.phantom.intent-sha256"}}' "${CONTAINER_ID}")"
    observed_nonce="$(docker inspect --type container \
      -f '{{index .Config.Labels "ace.phantom.run-nonce"}}' "${CONTAINER_ID}")"
    read -r expected_intent expected_nonce < <(python3 - "${state_path}" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value["intent_sha256"], value["run_nonce"])
PY
)
    if [[ "${observed_name}" != "/${CONTAINER_NAME}" ||
          "${observed_task}" != "${TASK_LABEL}" ||
          "${observed_intent}" != "${expected_intent}" ||
          "${observed_nonce}" != "${expected_nonce}" ]]; then
      echo "detached cleanup refused a container with changed ownership" >&2
      return 1
    fi
  elif [[ ! -f "${cleanup_intent}" ]]; then
    echo "owned detached container disappeared before cleanup intent" >&2
    return 1
  fi
  if [[ ! -f "${cleanup_intent}" ]]; then
    python3 - "${cleanup_intent}" "${CONTAINER_ID}" "${CONTAINER_NAME}" \
      "${TASK_LABEL}" "${state_path}" \
      "${DOCKER_EVIDENCE}/container-completed.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
state_path = pathlib.Path(sys.argv[5])
completed_path = pathlib.Path(sys.argv[6])
value = {
    "schema_version": "ace.phantom.bootstrap-host-cleanup-intent/1.0.0",
    "status": "ready",
    "container_id": sys.argv[2],
    "container_name": sys.argv[3],
    "task_label": sys.argv[4],
    "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    "completed_receipt_sha256": hashlib.sha256(completed_path.read_bytes()).hexdigest(),
}
temporary = path.with_name(path.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
  else
    python3 - "${cleanup_intent}" "${CONTAINER_ID}" "${CONTAINER_NAME}" \
      "${TASK_LABEL}" "${state_path}" \
      "${DOCKER_EVIDENCE}/container-completed.json" <<'PY'
import hashlib
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": "ace.phantom.bootstrap-host-cleanup-intent/1.0.0",
    "status": "ready",
    "container_id": sys.argv[2],
    "container_name": sys.argv[3],
    "task_label": sys.argv[4],
    "state_sha256": hashlib.sha256(pathlib.Path(sys.argv[5]).read_bytes()).hexdigest(),
    "completed_receipt_sha256": hashlib.sha256(
        pathlib.Path(sys.argv[6]).read_bytes()
    ).hexdigest(),
}
if value != expected:
    raise SystemExit("detached cleanup intent receipt changed")
PY
  fi
  if [[ "${existed_before_intent}" == true ]]; then
    if ! docker rm -f "${CONTAINER_ID}" >"${DOCKER_EVIDENCE}/cleanup.txt"; then
      return 1
    fi
  else
    printf '%s\trecovered-absent-after-cleanup-intent\n' "${CONTAINER_ID}" \
      >"${DOCKER_EVIDENCE}/cleanup.txt"
  fi
  if docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1 ||
     [[ -n "$(docker ps -aq --no-trunc --filter "name=^/${CONTAINER_NAME}$")" ]] ||
     [[ -n "$(docker ps -aq --no-trunc \
        --filter "label=ace.phantom.task=${TASK_LABEL}")" ]]; then
    echo "detached cleanup left an owned inventory" >&2
    return 1
  fi
  {
    printf '%s\tremoved-and-absent\n' "${CONTAINER_ID}"
    printf '%s\tname-inventory-empty\n' "${CONTAINER_NAME}"
    printf '%s\tlabel-inventory-empty\n' "${TASK_LABEL}"
  } >"${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  python3 - "${cleanup_result}" "${CONTAINER_ID}" "${CONTAINER_NAME}" \
    "${TASK_LABEL}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = {
    "schema_version": "ace.phantom.bootstrap-host-cleanup-result/1.0.0",
    "status": "pass",
    "container_id": sys.argv[2],
    "container_name": sys.argv[3],
    "task_label": sys.argv[4],
    "container_inventory": "absent",
    "name_inventory": "empty",
    "task_label_inventory": "empty",
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
  DETACHED_CLEANUP_COMPLETE=true
}

verify_outer_receipt_closure() {
  local result_requirement="$1"
  python3 - "${OUTPUT}" "${result_requirement}" <<'PY'
import hashlib
import json
import pathlib
import re
import stat
import sys

root = pathlib.Path(sys.argv[1])
result_requirement = sys.argv[2]

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in outer lifecycle receipt: {key}")
        value[key] = item
    return value

def regular(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False

def read_json(relative):
    path = root / relative
    if not regular(path):
        raise SystemExit(f"outer lifecycle receipt is missing or nonregular: {relative}")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

def digest(relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()

lifecycle = read_json("lifecycle.json")
lifecycle_keys = {
    "schema_version", "status", "execution_mode", "ace_commit",
    "phantom_commit", "base_image", "base_config_digest", "container_id",
    "task_label", "pipeline_exit_code", "result_archive_status",
    "container_cleanup", "container_name_inventory", "task_label_inventory",
    "gpu_device_requests", "gpu_executables_were_run",
}
if (
    set(lifecycle) != lifecycle_keys
    or lifecycle.get("schema_version")
    != "ace.phantom.bootstrap-host-freeze-lifecycle/2.0.0"
    or lifecycle.get("status") not in {"pass", "failed"}
    or lifecycle.get("execution_mode")
    not in {"synchronous", "detached-finalize"}
    or isinstance(lifecycle.get("pipeline_exit_code"), bool)
    or not isinstance(lifecycle.get("pipeline_exit_code"), int)
    or lifecycle.get("container_cleanup") != "removed-and-absent"
    or lifecycle.get("container_name_inventory") != "empty"
    or lifecycle.get("task_label_inventory") != "empty"
    or lifecycle.get("gpu_device_requests") != []
    or lifecycle.get("gpu_executables_were_run") is not False
):
    raise SystemExit("outer lifecycle receipt has an invalid exact schema")
container_id = lifecycle["container_id"]
created = read_json("docker/container-created.json")
completed = read_json("docker/container-completed.json")
if (
    not isinstance(created, list) or len(created) != 1
    or not isinstance(completed, list) or len(completed) != 1
    or created[0].get("Id") != container_id
    or completed[0].get("Id") != container_id
):
    raise SystemExit("outer Docker container receipts do not bind one exact ID")
verification = read_json("bootstrap-host-result-verification.json")
if verification.get("status") == "pass":
    if set(verification) != {
        "status", "mode", "pipeline_exit_code", "verified_file_count",
        "sha256_manifest_sha256",
    } or verification.get("mode") != "bootstrap-host-freeze" \
            or isinstance(verification.get("verified_file_count"), bool) \
            or not isinstance(verification.get("verified_file_count"), int) \
            or verification["verified_file_count"] < 1 \
            or not isinstance(verification.get("sha256_manifest_sha256"), str) \
            or len(verification["sha256_manifest_sha256"]) != 64:
        raise SystemExit("passing result verification receipt has an invalid shape")
elif verification.get("status") == "failed":
    if set(verification) != {
        "schema_version", "status", "mode", "pipeline_exit_code", "reason",
    } or verification.get("schema_version") != \
            "ace.phantom.bootstrap-host-result-verification/1.0.0" \
            or verification.get("mode") != "bootstrap-host-freeze" \
            or not isinstance(verification.get("reason"), str) \
            or not verification["reason"]:
        raise SystemExit("failed result verification receipt has an invalid shape")
else:
    raise SystemExit("result verification receipt has an invalid status")
if verification.get("pipeline_exit_code") != lifecycle["pipeline_exit_code"]:
    raise SystemExit("result verification and lifecycle exit codes differ")
if lifecycle["status"] == "pass" and (
    lifecycle["pipeline_exit_code"] != 0
    or lifecycle["result_archive_status"] != "verified"
    or verification["status"] != "pass"
    or result_requirement != "required"
):
    raise SystemExit("passing lifecycle lacks verified required result evidence")

if lifecycle["execution_mode"] == "detached-finalize":
    intent = read_json("detached-precreate-intent.json")
    state = read_json("detached-launch-state.json")
    launch = read_json("detached-launch.json")
    daemon = read_json("docker/daemon.json")
    cleanup_intent = read_json("docker/cleanup-intent.json")
    cleanup_result = read_json("docker/cleanup-result.json")
    log_result = read_json("docker/log-retrieval.json")
    verification_intent_path = root / "docker/result-verification-intent.json"
    if verification_intent_path.exists():
        verification_intent = read_json("docker/result-verification-intent.json")
        archive_path = root / "bootstrap-host-result.tar.gz"
        if (
            set(verification_intent) != {
                "schema_version", "status", "archive_sha256",
                "pipeline_exit_code", "temporary_extraction",
                "temporary_receipt", "final_extraction", "final_receipt",
            }
            or verification_intent.get("schema_version")
            != "ace.phantom.bootstrap-host-result-verification-intent/1.0.0"
            or verification_intent.get("status") != "ready"
            or not regular(archive_path)
            or verification_intent.get("archive_sha256")
            != hashlib.sha256(archive_path.read_bytes()).hexdigest()
            or verification_intent.get("pipeline_exit_code")
            != lifecycle["pipeline_exit_code"]
            or verification_intent.get("temporary_extraction")
            != ".verified-bootstrap-host-result.pending"
            or verification_intent.get("temporary_receipt")
            != ".result-verification-receipt.pending"
            or verification_intent.get("final_extraction")
            != "verified-bootstrap-host-result"
            or verification_intent.get("final_receipt")
            != "bootstrap-host-result-verification.json"
        ):
            raise SystemExit("result verification intent has an invalid exact binding")
    elif verification.get("status") == "pass":
        raise SystemExit("passing detached result verification lacks its intent")
    promotion = read_json("docker/result-promotion.json")
    promotion_intent_path = root / "docker/result-promotion-intent.json"
    promotion_intent = None
    if promotion_intent_path.exists():
        promotion_intent = read_json("docker/result-promotion-intent.json")
        if (
            set(promotion_intent) != {
                "schema_version", "status", "archive_sha256", "sidecar_sha256",
                "archive_name", "sidecar_name",
            }
            or promotion_intent.get("schema_version")
            != "ace.phantom.bootstrap-host-result-promotion-intent/1.0.0"
            or promotion_intent.get("status") != "ready"
            or promotion_intent.get("archive_name") != "bootstrap-host-result.tar.gz"
            or promotion_intent.get("sidecar_name")
            != "bootstrap-host-result.tar.gz.sha256"
            or re.fullmatch(
                r"[0-9a-f]{64}", promotion_intent.get("archive_sha256", "")
            ) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", promotion_intent.get("sidecar_sha256", "")
            ) is None
        ):
            raise SystemExit("result promotion intent has an invalid exact schema")
    root_archive = root / "bootstrap-host-result.tar.gz"
    root_sidecar = root / "bootstrap-host-result.tar.gz.sha256"
    if promotion.get("status") == "promoted":
        if (
            set(promotion) != {
                "schema_version", "status", "archive_sha256", "sidecar_sha256",
                "incoming_inventory",
            }
            or promotion.get("schema_version")
            != "ace.phantom.bootstrap-host-result-promotion/1.0.0"
            or promotion.get("incoming_inventory") != []
            or promotion_intent is None
            or not regular(root_archive)
            or not regular(root_sidecar)
            or promotion.get("archive_sha256")
            != hashlib.sha256(root_archive.read_bytes()).hexdigest()
            or promotion.get("sidecar_sha256")
            != hashlib.sha256(root_sidecar.read_bytes()).hexdigest()
            or promotion.get("archive_sha256")
            != promotion_intent.get("archive_sha256")
            or promotion.get("sidecar_sha256")
            != promotion_intent.get("sidecar_sha256")
            or root_sidecar.read_bytes()
            != (promotion["archive_sha256"] + "  bootstrap-host-result.tar.gz\n").encode()
        ):
            raise SystemExit("promoted result evidence has an invalid exact binding")
    elif promotion.get("status") == "missing":
        if promotion != {
            "schema_version": "ace.phantom.bootstrap-host-result-promotion/1.0.0",
            "status": "missing", "reason": "incoming-results is empty",
            "incoming_inventory": [],
        } or promotion_intent is not None or root_archive.exists() or root_sidecar.exists():
            raise SystemExit("missing result evidence has an invalid exact binding")
    elif promotion.get("status") == "rejected":
        inventory = promotion.get("incoming_inventory")
        if (
            set(promotion) != {
                "schema_version", "status", "reason", "incoming_inventory",
                "incoming_cleanup",
            }
            or promotion.get("schema_version")
            != "ace.phantom.bootstrap-host-result-promotion/1.0.0"
            or not isinstance(promotion.get("reason"), str)
            or not promotion["reason"]
            or promotion.get("incoming_cleanup") != "empty"
            or not isinstance(inventory, list)
            or any(
                not isinstance(item, dict)
                or set(item) != {"name", "kind"}
                or not isinstance(item.get("name"), str)
                or pathlib.PurePosixPath(item["name"]).name != item["name"]
                or item.get("kind")
                not in {"regular", "symlink", "directory", "nonregular"}
                for item in inventory
            )
            or root_archive.exists() or root_sidecar.exists()
        ):
            raise SystemExit("rejected result evidence has an invalid exact binding")
    else:
        raise SystemExit("result promotion receipt has an invalid status")
    if verification.get("status") == "pass" and promotion.get("status") != "promoted":
        raise SystemExit("passing result verification lacks promoted result evidence")
    intent_keys = {
        "schema_version", "status", "run_nonce", "output_directory",
        "output_device", "output_inode", "incoming_results_directory",
        "incoming_results_device", "incoming_results_inode", "ace_commit",
        "phantom_commit", "base_image", "base_config_digest", "container_name",
        "task_label", "build_jobs", "qualification_arguments",
        "payload_descriptor_sha256", "payload_manifest_sha256",
        "daemon_receipt_sha256", "base_image_receipt_sha256", "docker_daemon",
        "lifecycle_sha256", "container_command_sha256", "mounts",
    }
    state_keys = {
        "schema_version", "status", "output_directory", "output_device",
        "output_inode", "incoming_results_directory", "incoming_results_device",
        "incoming_results_inode", "ace_commit", "phantom_commit", "base_image",
        "base_config_digest", "qualification_arguments",
        "payload_descriptor_sha256", "payload_manifest_sha256",
        "created_receipt_sha256", "intent_sha256", "run_nonce",
        "daemon_receipt_sha256", "container",
    }
    if (
        set(intent) != intent_keys
        or intent.get("schema_version")
        != "ace.phantom.bootstrap-host-precreate-intent/1.0.0"
        or intent.get("status") != "ready-to-create"
    ):
        raise SystemExit("detached pre-create intent has an invalid exact schema")
    if (
        set(state) != state_keys
        or state.get("schema_version")
        != "ace.phantom.bootstrap-host-detached-state/1.0.0"
        or state.get("status") != "created"
    ):
        raise SystemExit("detached launch state has an invalid exact schema")
    base_image = read_json("docker/base-image.json")
    if (
        not isinstance(base_image, list)
        or len(base_image) != 1
        or not isinstance(base_image[0], dict)
        or base_image[0].get("Id") != intent.get("base_config_digest")
        or digest("docker/base-image.json")
        != intent.get("base_image_receipt_sha256")
    ):
        raise SystemExit("base-image receipt differs from its durable intent")
    if (
        state.get("container", {}).get("id") != container_id
        or state.get("created_receipt_sha256") != digest("docker/container-created.json")
        or state.get("intent_sha256") != digest("detached-precreate-intent.json")
        or state.get("daemon_receipt_sha256") != digest("docker/daemon.json")
    ):
        raise SystemExit("detached launch state hashes differ from its receipts")
    if set(daemon) != {
        "schema_version", "context", "daemon_id", "server_version",
    } or daemon.get("schema_version") != "ace.phantom.docker-daemon-identity/1.0.0":
        raise SystemExit("Docker daemon receipt has an invalid exact schema")
    cleanup_intent_keys = {
        "schema_version", "status", "container_id", "container_name",
        "task_label", "state_sha256", "completed_receipt_sha256",
    }
    cleanup_result_keys = {
        "schema_version", "status", "container_id", "container_name",
        "task_label", "container_inventory", "name_inventory",
        "task_label_inventory",
    }
    if (
        set(cleanup_intent) != cleanup_intent_keys
        or cleanup_intent.get("schema_version")
        != "ace.phantom.bootstrap-host-cleanup-intent/1.0.0"
        or cleanup_intent.get("status") != "ready"
        or cleanup_intent.get("container_id") != container_id
        or cleanup_intent.get("state_sha256") != digest("detached-launch-state.json")
        or cleanup_intent.get("completed_receipt_sha256")
        != digest("docker/container-completed.json")
    ):
        raise SystemExit("cleanup intent has an invalid exact binding")
    if (
        set(cleanup_result) != cleanup_result_keys
        or cleanup_result.get("schema_version")
        != "ace.phantom.bootstrap-host-cleanup-result/1.0.0"
        or cleanup_result.get("status") != "pass"
        or cleanup_result.get("container_id") != container_id
        or cleanup_result.get("container_inventory") != "absent"
        or cleanup_result.get("name_inventory") != "empty"
        or cleanup_result.get("task_label_inventory") != "empty"
    ):
        raise SystemExit("cleanup result has an invalid exact binding")
    log_path = root / "bootstrap-host-freeze.log"
    if not regular(log_path):
        raise SystemExit("detached log is nonregular")
    expected_log = {
        "schema_version": "ace.phantom.bootstrap-host-log-retrieval/1.0.0",
        "status": log_result.get("status"),
        "container_id": container_id,
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "log_size": log_path.stat().st_size,
    }
    if log_result.get("status") == "failed":
        diagnostic = root / "docker/log-retrieval-failure.txt"
        if not regular(diagnostic):
            raise SystemExit("log retrieval failure diagnostic is nonregular")
        expected_log["diagnostic_sha256"] = hashlib.sha256(
            diagnostic.read_bytes()
        ).hexdigest()
    if log_result != expected_log:
        raise SystemExit("log retrieval receipt has an invalid exact binding")

    def validate_start_result():
        start_result = read_json("docker/start-result.json")
        stdout_path = root / "docker/start.txt"
        stderr_path = root / "docker/start-stderr.txt"
        if not regular(stdout_path) or not regular(stderr_path):
            raise SystemExit("Docker start streams are missing or nonregular")
        if set(start_result) != {
            "schema_version", "status", "container_id", "client_exit_code",
            "stdout_sha256", "stdout_size", "stderr_sha256", "stderr_size",
        }:
            raise SystemExit("Docker start result has an invalid exact schema")
        exit_code = start_result.get("client_exit_code")
        if (
            start_result.get("schema_version")
            != "ace.phantom.bootstrap-host-start-result/1.0.0"
            or start_result.get("status")
            not in {"pass", "failed", "outcome-unavailable"}
            or start_result.get("container_id") != container_id
            or start_result.get("stdout_sha256")
            != hashlib.sha256(stdout_path.read_bytes()).hexdigest()
            or start_result.get("stdout_size") != stdout_path.stat().st_size
            or start_result.get("stderr_sha256")
            != hashlib.sha256(stderr_path.read_bytes()).hexdigest()
            or start_result.get("stderr_size") != stderr_path.stat().st_size
            or isinstance(exit_code, bool)
            or (start_result["status"] == "outcome-unavailable" and exit_code is not None)
            or (start_result["status"] != "outcome-unavailable" and not isinstance(exit_code, int))
            or (
                start_result["status"] == "pass"
                and (
                    exit_code != 0
                    or stdout_path.read_bytes() != (container_id + "\n").encode()
                )
            )
        ):
            raise SystemExit("Docker start result has an invalid exact binding")
        return start_result

    if launch.get("status") == "started":
        start_result = validate_start_result()
        started = read_json("docker/container-started.json")
        launch_keys = {
            "schema_version", "status", "container_id", "observed_container_status",
            "observed_running", "state_sha256", "started_receipt_sha256",
            "start_result_sha256", "start_client_status",
        }
        if (
            set(launch) != launch_keys
            or launch.get("schema_version")
            != "ace.phantom.bootstrap-host-detached-launch/1.0.0"
            or launch.get("container_id") != container_id
            or launch.get("state_sha256") != digest("detached-launch-state.json")
            or launch.get("started_receipt_sha256")
            != digest("docker/container-started.json")
            or launch.get("start_result_sha256") != digest("docker/start-result.json")
            or launch.get("start_client_status") != start_result.get("status")
            or not isinstance(started, list) or len(started) != 1
            or started[0].get("Id") != container_id
        ):
            raise SystemExit("started launch receipt has an invalid exact binding")
        if completed[0].get("State", {}).get("Status") != "exited":
            raise SystemExit("started lifecycle lacks an exited terminal receipt")
    elif launch.get("status") == "interrupted-before-start":
        expected_launch = {
            "schema_version": "ace.phantom.bootstrap-host-detached-launch/1.0.0",
            "status": "interrupted-before-start", "container_id": container_id,
            "state_sha256": digest("detached-launch-state.json"),
            "created_receipt_sha256": digest("docker/container-created.json"),
        }
        if launch != expected_launch or completed[0].get("State", {}).get("Status") != "created":
            raise SystemExit("interrupted launch receipt has an invalid exact binding")
        start_artifacts = [
            root / "docker/start-result.json", root / "docker/start.txt",
            root / "docker/start-stderr.txt",
        ]
        if any(path.exists() for path in start_artifacts):
            if not all(path.exists() for path in start_artifacts):
                raise SystemExit("interrupted launch has partial Docker start evidence")
            validate_start_result()
        if lifecycle["status"] == "pass":
            raise SystemExit("interrupted-before-start lifecycle cannot pass")
    else:
        raise SystemExit("detached launch receipt has an invalid status")
else:
    if completed[0].get("State", {}).get("Status") != "exited":
        raise SystemExit("synchronous lifecycle lacks an exited terminal receipt")
PY
}

bundle_install_intent() {
  local operation="$1"
  python3 - "${DOCKER_EVIDENCE}/bundle-install-intent.json" "${OUTPUT}" \
    "${operation}" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

intent_path, output, operation = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
output = output.resolve()
output_stat = output.stat()
temporary_manifest = output.parent / f".{output.name}.BUNDLE_SHA256SUMS.pending"
intent_temporary = intent_path.with_name(intent_path.name + ".tmp")

def reject_duplicates(pairs):
    result = {}
    for key, item in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key in bundle install intent: {key}")
        result[key] = item
    return result

if intent_temporary.exists():
    if (
        intent_path.exists()
        or operation != "create"
        or not stat.S_ISREG(intent_temporary.lstat().st_mode)
    ):
        raise SystemExit("bundle install intent temporary is unexpected or nonregular")
    intent_temporary.unlink()

excluded = {
    "BUNDLE_SHA256SUMS",
    intent_path.relative_to(output).as_posix(),
}
bound_files = []
for path in sorted(output.rglob("*"), key=lambda item: item.relative_to(output).as_posix()):
    relative = path.relative_to(output).as_posix()
    pure = pathlib.PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise SystemExit("bundle install inventory contains an unsafe path")
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        continue
    if relative in excluded:
        continue
    if not stat.S_ISREG(mode):
        raise SystemExit(f"bundle install inventory contains a nonregular entry: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    bound_files.append({"path": relative, "sha256": digest.hexdigest()})
names = [item["path"] for item in bound_files]
if names != sorted(names) or len(names) != len(set(names)):
    raise SystemExit("bundle install inventory is unsorted or duplicated")
inventory_bytes = json.dumps(
    bound_files, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("utf-8")

value = {
    "schema_version": "ace.phantom.bootstrap-host-bundle-install-intent/2.0.0",
    "status": "ready",
    "output_directory": str(output),
    "output_device": output_stat.st_dev,
    "output_inode": output_stat.st_ino,
    "temporary_manifest": str(temporary_manifest),
    "final_manifest": "BUNDLE_SHA256SUMS",
    "bound_files": bound_files,
    "bound_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
}
if intent_path.exists():
    if not stat.S_ISREG(intent_path.lstat().st_mode) or json.loads(
        intent_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    ) != value:
        raise SystemExit("bundle install intent changed")
elif operation == "create":
    with intent_temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(intent_temporary, 0o600)
    os.replace(intent_temporary, intent_path)
    directory_fd = os.open(intent_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
else:
    raise SystemExit("bundle install intent is missing")
print(temporary_manifest)
PY
}

write_outer_bundle_checksum_closure() {
  local result_requirement="${1:-required}"
  local relative
  local -a required_files=(
    bootstrap-host-result-verification.json
    docker/base-image.json
    docker/cleanup-verification.tsv
    docker/cleanup.txt
    docker/container-completed.json
    docker/container-created.json
    docker/pull.txt
  )
  for relative in "${required_files[@]}"; do
    if [[ ! -s "${OUTPUT}/${relative}" ]]; then
      echo "outer host-freeze evidence is missing or empty: ${relative}" >&2
      return 1
    fi
  done
  if [[ ! -f "${OUTPUT}/bootstrap-host-freeze.log" ||
        -L "${OUTPUT}/bootstrap-host-freeze.log" ]]; then
    echo "outer host-freeze log is missing or nonregular" >&2
    return 1
  fi
  if [[ -e "${OUTPUT}/detached-launch-state.json" &&
        ! -s "${DOCKER_EVIDENCE}/log-retrieval.json" ]]; then
    echo "detached host-freeze log retrieval receipt is missing or empty" >&2
    return 1
  fi
  if [[ ! -s "${OUTPUT}/lifecycle.json" ]]; then
    echo "outer host-freeze lifecycle receipt is missing or empty" >&2
    return 1
  fi
  if [[ "${result_requirement}" == "required" ]]; then
    for relative in bootstrap-host-result.tar.gz \
      bootstrap-host-result.tar.gz.sha256; do
      if [[ ! -s "${OUTPUT}/${relative}" ]]; then
        echo "outer host-freeze result evidence is missing or empty: ${relative}" >&2
        return 1
      fi
    done
  elif [[ "${result_requirement}" != "allow-missing-result" ]]; then
    echo "invalid outer bundle result requirement: ${result_requirement}" >&2
    return 1
  fi
  if [[ -e "${DOCKER_EVIDENCE}/bundle-install-intent.json" ]]; then
    bundle_install_intent validate >/dev/null || return 1
  fi
  verify_outer_receipt_closure "${result_requirement}" || return 1
  local temporary_manifest
  temporary_manifest="$(bundle_install_intent create)"
  local unexpected
  unexpected="$(find "${OUTPUT}" -mindepth 1 ! -type d ! -type f \
    -print -quit)"
  if [[ -n "${unexpected}" ]]; then
    echo "outer host-freeze evidence contains a nonregular entry: ${unexpected}" >&2
    return 1
  fi
  if [[ -e "${OUTPUT}/BUNDLE_SHA256SUMS" ]]; then
    local existing_exit
    set +e
    verify_existing_outer_bundle >/dev/null
    existing_exit="$?"
    set -e
    if [[ ${existing_exit} -eq 0 ]]; then
      return 0
    fi
    bundle_install_intent validate >/dev/null
    rm -f -- "${OUTPUT}/BUNDLE_SHA256SUMS"
  fi
  if [[ -e "${temporary_manifest}" &&
        ( ! -f "${temporary_manifest}" || -L "${temporary_manifest}" ) ]]; then
    echo "outer host-freeze temporary bundle manifest is nonregular" >&2
    return 1
  fi
  (
    cd "${OUTPUT}"
    local inventory_before inventory_after
    inventory_before="$(mktemp)"
    inventory_after="$(mktemp)"
    trap 'rm -f -- "${inventory_before}" "${inventory_after}"' EXIT
    find . -type f ! -path ./BUNDLE_SHA256SUMS -print0 |
      LC_ALL=C sort -z >"${inventory_before}"
    xargs -0 -r sha256sum <"${inventory_before}" >"${temporary_manifest}"
    find . -type f ! -path ./BUNDLE_SHA256SUMS -print0 |
      LC_ALL=C sort -z >"${inventory_after}"
    cmp -- "${inventory_before}" "${inventory_after}"
    sha256sum -c "${temporary_manifest}"
    sync -f "${temporary_manifest}"
    bundle_install_intent validate >/dev/null
    mv -- "${temporary_manifest}" BUNDLE_SHA256SUMS
    sync -f .
  )
}

verify_existing_outer_bundle() {
  local manifest="${OUTPUT}/BUNDLE_SHA256SUMS"
  if [[ ! -f "${manifest}" || -L "${manifest}" ]]; then
    echo "existing outer bundle manifest is nonregular" >&2
    return 1
  fi
  local unexpected
  unexpected="$(find "${OUTPUT}" -mindepth 1 ! -type d ! -type f \
    -print -quit)"
  if [[ -n "${unexpected}" ]]; then
    echo "existing outer bundle contains a nonregular entry: ${unexpected}" >&2
    return 1
  fi
  if ! (cd "${OUTPUT}" && sha256sum -c BUNDLE_SHA256SUMS); then
    return 1
  fi
  if ! python3 - "${OUTPUT}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
listed = []
for line in (root / "BUNDLE_SHA256SUMS").read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"[0-9a-f]{64}  (\./.+)", line)
    if match is None:
        raise SystemExit("existing outer bundle manifest is malformed")
    listed.append(match.group(1))
actual = sorted(
    "./" + path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != root / "BUNDLE_SHA256SUMS"
)
if sorted(listed) != actual or len(listed) != len(set(listed)):
    raise SystemExit("existing outer bundle inventory differs from its manifest")
PY
  then
    return 1
  fi
  local existing_result_requirement=allow-missing-result
  if [[ -e "${OUTPUT}/bootstrap-host-result.tar.gz" ||
        -e "${OUTPUT}/bootstrap-host-result.tar.gz.sha256" ]]; then
    if [[ ! -s "${OUTPUT}/bootstrap-host-result.tar.gz" ||
          ! -s "${OUTPUT}/bootstrap-host-result.tar.gz.sha256" ]]; then
      echo "existing outer bundle has a partial result archive pair" >&2
      return 1
    fi
    existing_result_requirement=required
  fi
  bundle_install_intent validate >/dev/null || return 1
  verify_outer_receipt_closure "${existing_result_requirement}" || return 1
  local result
  result="$(python3 - "${OUTPUT}/lifecycle.json" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("status") == "pass":
    print(0)
elif value.get("status") == "failed" and isinstance(value.get("pipeline_exit_code"), int):
    print(value["pipeline_exit_code"] or 1)
else:
    raise SystemExit("existing outer bundle lifecycle is invalid")
PY
)"
  echo "detached bootstrap host evidence is already checksum-closed"
  EXISTING_BUNDLE_PIPELINE_EXIT="${result}"
  return 0
}

write_lifecycle_receipt() {
  local status="$1"
  local pipeline_exit="$2"
  local result_archive_status="$3"
  python3 - "${OUTPUT}/lifecycle.json" "${status}" "${RUN_MODE}" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
    "${CREATED_CONTAINER_ID:-${CONTAINER_ID}}" "${TASK_LABEL}" \
  "${pipeline_exit}" "${result_archive_status}" <<'PY'
import json
import os
import pathlib
import stat
import sys

(
    path, status, execution_mode, ace_commit, phantom_commit, base_image,
    base_config, container_id, task_label, pipeline_exit,
    result_archive_status,
) = sys.argv[1:]
if status not in {"pass", "failed"}:
    raise SystemExit("invalid bootstrap host lifecycle status")
value = {
    "schema_version": "ace.phantom.bootstrap-host-freeze-lifecycle/2.0.0",
    "status": status,
    "execution_mode": execution_mode,
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "base_image": base_image,
    "base_config_digest": base_config,
    "container_id": container_id,
    "task_label": task_label,
    "pipeline_exit_code": int(pipeline_exit),
    "result_archive_status": result_archive_status,
    "container_cleanup": "removed-and-absent",
    "container_name_inventory": "empty",
    "task_label_inventory": "empty",
    "gpu_device_requests": [],
    "gpu_executables_were_run": False,
}
target = pathlib.Path(path)
if target.exists():
    if not stat.S_ISREG(target.lstat().st_mode):
        raise SystemExit("bootstrap host lifecycle receipt is nonregular")
    if json.loads(target.read_text(encoding="utf-8")) != value:
        raise SystemExit("bootstrap host lifecycle receipt changed")
else:
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
PY
}

write_result_verification_failure() {
  local reason="$1"
  python3 - "${OUTPUT}/bootstrap-host-result-verification.json" \
    "${PIPELINE_EXIT}" "${reason}" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "ace.phantom.bootstrap-host-result-verification/1.0.0",
            "status": "failed",
            "mode": "bootstrap-host-freeze",
            "pipeline_exit_code": int(sys.argv[2]),
            "reason": sys.argv[3],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

write_docker_daemon_receipt() {
  local context daemon_id server_version
  context="$(docker context show)"
  daemon_id="$(docker info --format '{{.ID}}')"
  server_version="$(docker version --format '{{.Server.Version}}')"
  python3 - "${DOCKER_EVIDENCE}/daemon.json" "${context}" \
    "${daemon_id}" "${server_version}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
fields = sys.argv[2:]
if any(not value or "\n" in value or "\r" in value for value in fields):
    raise SystemExit("Docker daemon identity contains an invalid field")
value = {
    "schema_version": "ace.phantom.docker-daemon-identity/1.0.0",
    "context": fields[0],
    "daemon_id": fields[1],
    "server_version": fields[2],
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
}

verify_docker_daemon_receipt() {
  local receipt="${DOCKER_EVIDENCE}/daemon.json"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] || {
    echo "Docker daemon identity receipt is missing or nonregular" >&2
    return 1
  }
  local context daemon_id server_version
  context="$(docker context show)"
  daemon_id="$(docker info --format '{{.ID}}')"
  server_version="$(docker version --format '{{.Server.Version}}')"
  python3 - "${receipt}" "${context}" "${daemon_id}" "${server_version}" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "schema_version": "ace.phantom.docker-daemon-identity/1.0.0",
    "context": sys.argv[2],
    "daemon_id": sys.argv[3],
    "server_version": sys.argv[4],
}
if value != expected:
    raise SystemExit("Docker context or daemon identity changed")
PY
}

write_detached_precreate_intent() {
  RUN_NONCE="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  python3 - "${OUTPUT}/detached-precreate-intent.json" "${OUTPUT}" \
    "${PAYLOAD}" "${INCOMING_RESULTS}" "${DOCKER_EVIDENCE}/daemon.json" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
    "${CONTAINER_NAME}" "${TASK_LABEL}" "${RUN_NONCE}" "${BUILD_JOBS}" \
    "${REPO_ROOT}/tools/phantom_gpu/freeze_bootstrap_host_evidence.sh" \
    "${DOCKER_EVIDENCE}/base-image.json" \
    "${BOOTSTRAP_HOST_CONTAINER_COMMAND}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    intent_path, output_path, payload_path, incoming_path, daemon_path,
    ace_commit, phantom_commit, base_image, base_config, container_name,
    task_label, nonce, build_jobs, lifecycle_path, base_image_path,
    container_command,
) = sys.argv[1:]
output = pathlib.Path(output_path).resolve()
payload = pathlib.Path(payload_path).resolve()
incoming = pathlib.Path(incoming_path).resolve()
descriptor_path = payload / "payload.json"
manifest_path = payload / "SHA256SUMS"
descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
daemon = json.loads(pathlib.Path(daemon_path).read_text(encoding="utf-8"))
output_stat, incoming_stat = output.stat(), incoming.stat()
value = {
    "schema_version": "ace.phantom.bootstrap-host-precreate-intent/1.0.0",
    "status": "ready-to-create",
    "run_nonce": nonce,
    "output_directory": str(output),
    "output_device": output_stat.st_dev,
    "output_inode": output_stat.st_ino,
    "incoming_results_directory": str(incoming),
    "incoming_results_device": incoming_stat.st_dev,
    "incoming_results_inode": incoming_stat.st_ino,
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "base_image": base_image,
    "base_config_digest": base_config,
    "container_name": container_name,
    "task_label": task_label,
    "build_jobs": int(build_jobs),
    "qualification_arguments": descriptor["qualification"]["arguments"],
    "payload_descriptor_sha256": hashlib.sha256(
        descriptor_path.read_bytes()
    ).hexdigest(),
    "payload_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "daemon_receipt_sha256": hashlib.sha256(
        pathlib.Path(daemon_path).read_bytes()
    ).hexdigest(),
    "base_image_receipt_sha256": hashlib.sha256(
        pathlib.Path(base_image_path).read_bytes()
    ).hexdigest(),
    "docker_daemon": daemon,
    "lifecycle_sha256": hashlib.sha256(
        pathlib.Path(lifecycle_path).read_bytes()
    ).hexdigest(),
    "container_command_sha256": hashlib.sha256(
        container_command.encode("utf-8")
    ).hexdigest(),
    "mounts": [
        {"source": str(payload), "destination": "/bootstrap-freeze/input", "read_write": False},
        {"source": str(incoming), "destination": "/bootstrap-freeze/output", "read_write": True},
    ],
}
target = pathlib.Path(intent_path)
temporary = target.with_name(target.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, target)
directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
  INTENT_SHA256="$(sha256sum "${OUTPUT}/detached-precreate-intent.json" |
    awk '{print $1}')"
}

atomic_install_regular_file() {
  local source="$1"
  local target="$2"
  python3 - "${source}" "${target}" <<'PY'
import os
import pathlib
import stat
import sys

source, target = map(pathlib.Path, sys.argv[1:])
if not stat.S_ISREG(source.lstat().st_mode):
    raise SystemExit("atomic publication source is nonregular")
if target.exists() and not stat.S_ISREG(target.lstat().st_mode):
    raise SystemExit("atomic publication target is nonregular")
temporary = target.with_name(target.name + ".tmp")
if temporary.exists() and not stat.S_ISREG(temporary.lstat().st_mode):
    raise SystemExit("atomic publication temporary is nonregular")
with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
    while chunk := input_stream.read(1024 * 1024):
        output_stream.write(chunk)
    output_stream.flush()
    os.fsync(output_stream.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, target)
directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

validate_intent_owned_container() {
  local receipt="$1"
  local allowed_statuses="$2"
  python3 - "${receipt}" "${OUTPUT}/detached-precreate-intent.json" \
    "${DOCKER_EVIDENCE}/base-image.json" "${OUTPUT}" "${PAYLOAD}" \
    "${INCOMING_RESULTS}" "${allowed_statuses}" \
    "${REPO_ROOT}/tools/phantom_gpu/freeze_bootstrap_host_evidence.sh" \
    "${CONTAINER_ID}" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

(
    receipt_path, intent_path, image_path, output_path, payload_path,
    incoming_path, allowed_text, lifecycle_path, expected_container_id,
) = map(pathlib.Path, sys.argv[1:])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key: {key}")
        value[key] = item
    return value

def read_json(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit(f"nonregular Docker binding receipt: {path}")
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

record = read_json(receipt_path)
intent = read_json(intent_path)
images = read_json(image_path)
if not isinstance(record, list) or len(record) != 1:
    raise SystemExit("Docker container receipt is not singular")
if not isinstance(images, list) or len(images) != 1:
    raise SystemExit("Docker base-image receipt is not singular")
item, image = record[0], images[0]
if not isinstance(item, dict) or not isinstance(image, dict):
    raise SystemExit("Docker binding receipt has an invalid shape")
expected_intent_keys = {
    "schema_version", "status", "run_nonce", "output_directory",
    "output_device", "output_inode", "incoming_results_directory",
    "incoming_results_device", "incoming_results_inode", "ace_commit",
    "phantom_commit", "base_image", "base_config_digest", "container_name",
    "task_label", "build_jobs", "qualification_arguments",
    "payload_descriptor_sha256", "payload_manifest_sha256",
    "daemon_receipt_sha256", "base_image_receipt_sha256", "docker_daemon",
    "lifecycle_sha256", "container_command_sha256", "mounts",
}
if (
    set(intent) != expected_intent_keys
    or intent.get("schema_version")
    != "ace.phantom.bootstrap-host-precreate-intent/1.0.0"
    or intent.get("status") != "ready-to-create"
):
    raise SystemExit("pre-create intent has an unexpected schema")
if hashlib.sha256(image_path.read_bytes()).hexdigest() != intent[
    "base_image_receipt_sha256"
]:
    raise SystemExit("Docker base-image receipt changed after intent")
if hashlib.sha256(lifecycle_path.read_bytes()).hexdigest() != intent[
    "lifecycle_sha256"
]:
    raise SystemExit("host lifecycle bytes changed after intent")
output, payload, incoming = output_path.resolve(), payload_path.resolve(), incoming_path.resolve()
output_stat, incoming_stat = output.stat(), incoming.stat()
if (
    intent["output_directory"] != str(output)
    or intent["output_device"] != output_stat.st_dev
    or intent["output_inode"] != output_stat.st_ino
    or intent["incoming_results_directory"] != str(incoming)
    or intent["incoming_results_device"] != incoming_stat.st_dev
    or intent["incoming_results_inode"] != incoming_stat.st_ino
):
    raise SystemExit("pre-create intent path identity changed")
if image.get("Id") != intent["base_config_digest"]:
    raise SystemExit("base-image configuration digest differs from intent")
config = item.get("Config")
host = item.get("HostConfig")
runtime = item.get("State")
if not all(isinstance(value, dict) for value in (config, host, runtime)):
    raise SystemExit("Docker container configuration is incomplete")
allowed = set(str(allowed_text).split(","))
status_value = runtime.get("Status")
if status_value not in allowed:
    raise SystemExit("Docker container state is outside the permitted recovery state")
if (
    not isinstance(runtime.get("Running"), bool)
    or (status_value == "running") != runtime["Running"]
    or (status_value == "exited" and isinstance(runtime.get("ExitCode"), bool))
    or (status_value == "exited" and not isinstance(runtime.get("ExitCode"), int))
):
    raise SystemExit("Docker container runtime state is inconsistent")

base_config = image.get("Config") or {}
if not isinstance(base_config, dict):
    raise SystemExit("Docker base-image configuration is invalid")

def environment_map(values):
    result = {}
    for entry in values or []:
        if not isinstance(entry, str) or "=" not in entry:
            raise SystemExit("Docker environment entry is invalid")
        key = entry.split("=", 1)[0]
        if not key:
            raise SystemExit("Docker environment key is empty")
        result[key] = entry
    return result

expected_environment = environment_map(base_config.get("Env"))
overrides = [
    "NVIDIA_VISIBLE_DEVICES=void",
    "NVIDIA_DRIVER_CAPABILITIES=none",
    f"ACE_RUNPOD_BASE_IMAGE={intent['base_image']}",
    f"ACE_RUNPOD_BASE_CONFIG_DIGEST={intent['base_config_digest']}",
    f"ACE_PHANTOM_BUILD_JOBS={intent['build_jobs']}",
    "ACE_PHANTOM_INTENT_SHA256=" + hashlib.sha256(intent_path.read_bytes()).hexdigest(),
    f"ACE_PHANTOM_RUN_NONCE={intent['run_nonce']}",
]
expected_environment.update(environment_map(overrides))
observed_environment = config.get("Env") or []
if sorted(observed_environment) != sorted(expected_environment.values()):
    raise SystemExit("Docker container environment differs from intent")

expected_labels = dict(base_config.get("Labels") or {})
expected_labels.update({
    "ace.phantom.task": intent["task_label"],
    "ace.phantom.intent-sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
    "ace.phantom.run-nonce": intent["run_nonce"],
})
command = config.get("Cmd")
if (
    not isinstance(command, list)
    or len(command) != 3
    or command[:2] != ["bash", "-lc"]
    or not isinstance(command[2], str)
    or hashlib.sha256(command[2].encode("utf-8")).hexdigest()
    != intent["container_command_sha256"]
):
    raise SystemExit("Docker container command differs from intent")
if (
    item.get("Id") != str(expected_container_id)
    or item.get("Name") != "/" + intent["container_name"]
    or item.get("Image") != intent["base_config_digest"]
    or config.get("Image") != intent["base_image"]
    or config.get("Entrypoint") != base_config.get("Entrypoint")
    or config.get("Labels") != expected_labels
    or (config.get("User") or "") != (base_config.get("User") or "")
    or (config.get("WorkingDir") or "") != (base_config.get("WorkingDir") or "")
):
    raise SystemExit("Docker container identity or Config differs from intent")
restart = host.get("RestartPolicy") or {}
if (
    host.get("Runtime") != "runc"
    or host.get("Privileged") is not False
    or (host.get("Devices") or []) != []
    or (host.get("DeviceRequests") or []) != []
    or restart.get("Name") != "no"
    or restart.get("MaximumRetryCount", 0) != 0
    or host.get("NetworkMode") != "bridge"
):
    raise SystemExit("Docker HostConfig differs from the immutable launch request")
observed_mounts = sorted(
    [
        {
            "type": mount.get("Type"),
            "source": mount.get("Source"),
            "destination": mount.get("Destination"),
            "mode": mount.get("Mode"),
            "read_write": mount.get("RW"),
            "propagation": mount.get("Propagation"),
        }
        for mount in item.get("Mounts", [])
    ],
    key=lambda value: value["destination"] or "",
)
expected_mounts = sorted(
    [
        {
            "type": "bind", "source": str(payload),
            "destination": "/bootstrap-freeze/input", "mode": "ro",
            "read_write": False, "propagation": "rprivate",
        },
        {
            "type": "bind", "source": str(incoming),
            "destination": "/bootstrap-freeze/output", "mode": "rw",
            "read_write": True, "propagation": "rprivate",
        },
    ],
    key=lambda value: value["destination"],
)
if observed_mounts != expected_mounts:
    raise SystemExit("Docker bind-mount configuration differs from intent")
print(status_value)
PY
}

capture_container_receipt() {
  local target="$1"
  local allowed_statuses="$2"
  local temporary="${DOCKER_EVIDENCE}/.${target##*/}.capture.pending"
  if [[ -e "${temporary}" &&
        ( ! -f "${temporary}" || -L "${temporary}" ) ]]; then
    echo "container receipt capture path is nonregular: ${temporary}" >&2
    return 1
  fi
  rm -f -- "${temporary}"
  if ! docker inspect --type container "${CONTAINER_ID}" >"${temporary}"; then
    rm -f -- "${temporary}"
    return 1
  fi
  local observed_status
  observed_status="$(validate_intent_owned_container \
    "${temporary}" "${allowed_statuses}")"
  atomic_install_regular_file "${temporary}" "${target}"
  rm -f -- "${temporary}"
  printf '%s\n' "${observed_status}"
}

publish_detached_start_result() {
  local stdout_source="$1"
  local stderr_source="$2"
  local client_status="$3"
  local client_exit_code="$4"
  local stdout_path="${DOCKER_EVIDENCE}/start.txt"
  local stderr_path="${DOCKER_EVIDENCE}/start-stderr.txt"
  atomic_install_regular_file "${stdout_source}" "${stdout_path}"
  atomic_install_regular_file "${stderr_source}" "${stderr_path}"
  python3 - "${DOCKER_EVIDENCE}/start-result.json" "${stdout_path}" \
    "${stderr_path}" "${CONTAINER_ID}" "${client_status}" \
    "${client_exit_code}" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

receipt, stdout_path, stderr_path = map(pathlib.Path, sys.argv[1:4])
container_id, status_value, exit_text = sys.argv[4:]
if status_value not in {"pass", "failed", "outcome-unavailable"}:
    raise SystemExit("invalid Docker start client status")
exit_code = None if exit_text == "unavailable" else int(exit_text)
if status_value == "pass" and (
    exit_code != 0 or stdout_path.read_bytes() != (container_id + "\n").encode()
):
    raise SystemExit("passing Docker start output is not the exact container ID")
if status_value == "failed" and exit_code is None:
    raise SystemExit("failed Docker start result lacks a client exit code")
if status_value == "outcome-unavailable" and exit_code is not None:
    raise SystemExit("unavailable Docker start result claims a client exit code")
value = {
    "schema_version": "ace.phantom.bootstrap-host-start-result/1.0.0",
    "status": status_value,
    "container_id": container_id,
    "client_exit_code": exit_code,
    "stdout_sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
    "stdout_size": stdout_path.stat().st_size,
    "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
    "stderr_size": stderr_path.stat().st_size,
}
if receipt.exists():
    if not stat.S_ISREG(receipt.lstat().st_mode) or json.loads(
        receipt.read_text(encoding="utf-8")
    ) != value:
        raise SystemExit("Docker start result receipt changed")
else:
    temporary = receipt.with_name(receipt.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, receipt)
    directory_fd = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
PY
}

verify_detached_start_result() {
  python3 - "${DOCKER_EVIDENCE}/start-result.json" \
    "${DOCKER_EVIDENCE}/start.txt" "${DOCKER_EVIDENCE}/start-stderr.txt" \
    "${CONTAINER_ID}" <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

receipt, stdout_path, stderr_path = map(pathlib.Path, sys.argv[1:4])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in Docker start result: {key}")
        value[key] = item
    return value

for path in (receipt, stdout_path, stderr_path):
    if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise SystemExit("Docker start evidence is missing or nonregular")
value = json.loads(
    receipt.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
)
if set(value) != {
    "schema_version", "status", "container_id", "client_exit_code",
    "stdout_sha256", "stdout_size", "stderr_sha256", "stderr_size",
}:
    raise SystemExit("Docker start result has an unexpected schema")
if (
    value["schema_version"] != "ace.phantom.bootstrap-host-start-result/1.0.0"
    or value["status"] not in {"pass", "failed", "outcome-unavailable"}
    or value["container_id"] != sys.argv[4]
    or value["stdout_sha256"] != hashlib.sha256(stdout_path.read_bytes()).hexdigest()
    or value["stdout_size"] != stdout_path.stat().st_size
    or value["stderr_sha256"] != hashlib.sha256(stderr_path.read_bytes()).hexdigest()
    or value["stderr_size"] != stderr_path.stat().st_size
):
    raise SystemExit("Docker start result no longer matches its evidence")
exit_code = value["client_exit_code"]
if isinstance(exit_code, bool) or (
    value["status"] == "outcome-unavailable" and exit_code is not None
) or (
    value["status"] != "outcome-unavailable" and not isinstance(exit_code, int)
):
    raise SystemExit("Docker start client exit code is inconsistent")
if value["status"] == "pass" and (
    exit_code != 0 or stdout_path.read_bytes() != (sys.argv[4] + "\n").encode()
):
    raise SystemExit("passing Docker start output is not the exact container ID")
print(value["status"])
PY
}

recover_detached_start_result() {
  if [[ -e "${DOCKER_EVIDENCE}/start-result.json" ]]; then
    verify_detached_start_result
    return "$?"
  fi
  local stdout_source="${DOCKER_EVIDENCE}/start.txt"
  local stderr_source="${DOCKER_EVIDENCE}/start-stderr.txt"
  local stdout_pending="${DOCKER_EVIDENCE}/.start.stdout.pending"
  local stderr_pending="${DOCKER_EVIDENCE}/.start.stderr.pending"
  local empty_stdout="" empty_stderr=""
  if [[ ! -e "${stdout_source}" ]]; then
    if [[ -e "${stdout_pending}" ]]; then
      stdout_source="${stdout_pending}"
    else
      empty_stdout="$(mktemp)"
      stdout_source="${empty_stdout}"
    fi
  fi
  if [[ ! -e "${stderr_source}" ]]; then
    if [[ -e "${stderr_pending}" ]]; then
      stderr_source="${stderr_pending}"
    else
      empty_stderr="$(mktemp)"
      stderr_source="${empty_stderr}"
    fi
  fi
  if [[ ! -f "${stdout_source}" || -L "${stdout_source}" ||
        ! -f "${stderr_source}" || -L "${stderr_source}" ]]; then
    [[ -z "${empty_stdout}" ]] || rm -f -- "${empty_stdout}"
    [[ -z "${empty_stderr}" ]] || rm -f -- "${empty_stderr}"
    echo "interrupted Docker start streams are nonregular" >&2
    return 1
  fi
  publish_detached_start_result "${stdout_source}" "${stderr_source}" \
    outcome-unavailable unavailable
  rm -f -- "${stdout_pending}" "${stderr_pending}"
  [[ -z "${empty_stdout}" ]] || rm -f -- "${empty_stdout}"
  [[ -z "${empty_stderr}" ]] || rm -f -- "${empty_stderr}"
  printf '%s\n' outcome-unavailable
}

write_detached_launch_state() {
  python3 - "${DOCKER_EVIDENCE}/container-created.json" \
    "${OUTPUT}/detached-launch-state.json" "${OUTPUT}" "${PAYLOAD}" \
    "${INCOMING_RESULTS}" "${OUTPUT}/detached-precreate-intent.json" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
    "${CONTAINER_ID}" "${CONTAINER_NAME}" "${TASK_LABEL}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    inspect_path, state_path, output_path, payload_path, incoming_path, intent_path, ace_commit,
    phantom_commit, base_image, base_config, container_id, container_name,
    task_label,
) = sys.argv[1:]
output = pathlib.Path(output_path).resolve()
payload = pathlib.Path(payload_path).resolve()
incoming = pathlib.Path(incoming_path).resolve()
created_path = pathlib.Path(inspect_path)
intent_path = pathlib.Path(intent_path)
intent = json.loads(intent_path.read_text(encoding="utf-8"))
created = json.loads(created_path.read_text(encoding="utf-8"))
if not isinstance(created, list) or len(created) != 1:
    raise SystemExit("Docker did not return one created-container record")

def normalized_container(item):
    config = item.get("Config", {})
    host = item.get("HostConfig", {})
    mounts = sorted(
        [
            {
                "type": mount.get("Type"),
                "source": mount.get("Source"),
                "destination": mount.get("Destination"),
                "mode": mount.get("Mode"),
                "read_write": mount.get("RW"),
                "propagation": mount.get("Propagation"),
            }
            for mount in item.get("Mounts", [])
        ],
        key=lambda value: value["destination"] or "",
    )
    return {
        "id": item.get("Id"),
        "name": item.get("Name"),
        "task_label": (config.get("Labels") or {}).get("ace.phantom.task"),
        "image_config_digest": item.get("Image"),
        "configuration": {
            "image": config.get("Image"),
            "entrypoint": config.get("Entrypoint"),
            "command": config.get("Cmd"),
            "environment": sorted(config.get("Env") or []),
            "labels": config.get("Labels") or {},
            "user": config.get("User") or "",
            "working_directory": config.get("WorkingDir") or "",
            "runtime": host.get("Runtime"),
            "privileged": host.get("Privileged"),
            "devices": host.get("Devices") or [],
            "device_requests": host.get("DeviceRequests") or [],
            "restart_policy": host.get("RestartPolicy"),
            "network_mode": host.get("NetworkMode"),
            "mounts": mounts,
        },
    }

container = normalized_container(created[0])
expected_mounts = {
    (str(payload), "/bootstrap-freeze/input", False),
    (str(incoming), "/bootstrap-freeze/output", True),
}
observed_mounts = {
    (item["source"], item["destination"], item["read_write"])
    for item in container["configuration"]["mounts"]
}
if (
    container["id"] != container_id
    or container["name"] != "/" + container_name
    or container["task_label"] != task_label
    or container["configuration"]["labels"].get("ace.phantom.intent-sha256")
    != hashlib.sha256(intent_path.read_bytes()).hexdigest()
    or container["configuration"]["labels"].get("ace.phantom.run-nonce")
    != intent.get("run_nonce")
    or container["image_config_digest"] != base_config
    or container["configuration"]["image"] != base_image
    or (container["configuration"]["restart_policy"] or {}).get("Name") != "no"
    or expected_mounts != observed_mounts
):
    raise SystemExit("created container does not match the detached launch binding")
payload_descriptor = payload / "payload.json"
payload_manifest = payload / "SHA256SUMS"
descriptor = json.loads(payload_descriptor.read_text(encoding="utf-8"))
stat = output.stat()
value = {
    "schema_version": "ace.phantom.bootstrap-host-detached-state/1.0.0",
    "status": "created",
    "output_directory": str(output),
    "output_device": stat.st_dev,
    "output_inode": stat.st_ino,
    "incoming_results_directory": str(incoming),
    "incoming_results_device": incoming.stat().st_dev,
    "incoming_results_inode": incoming.stat().st_ino,
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "base_image": base_image,
    "base_config_digest": base_config,
    "qualification_arguments": descriptor["qualification"]["arguments"],
    "payload_descriptor_sha256": hashlib.sha256(
        payload_descriptor.read_bytes()
    ).hexdigest(),
    "payload_manifest_sha256": hashlib.sha256(
        payload_manifest.read_bytes()
    ).hexdigest(),
    "created_receipt_sha256": hashlib.sha256(created_path.read_bytes()).hexdigest(),
    "intent_sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
    "run_nonce": intent["run_nonce"],
    "daemon_receipt_sha256": intent["daemon_receipt_sha256"],
    "container": container,
}
target = pathlib.Path(state_path)
if target.exists():
    if target.is_symlink() or json.loads(target.read_text(encoding="utf-8")) != value:
        raise SystemExit("detached launch state changed")
else:
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
PY
}

write_detached_launch_receipt() {
  python3 - "${OUTPUT}/detached-launch-state.json" \
    "${DOCKER_EVIDENCE}/container-started.json" \
    "${DOCKER_EVIDENCE}/start-result.json" \
    "${OUTPUT}/detached-launch.json" "${CONTAINER_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

state_path, started_path, start_result_path, receipt_path = map(
    pathlib.Path, sys.argv[1:5]
)
container_id = sys.argv[5]
started = json.loads(started_path.read_text(encoding="utf-8"))
start_result = json.loads(start_result_path.read_text(encoding="utf-8"))
if not isinstance(started, list) or len(started) != 1:
    raise SystemExit("Docker did not return one started-container record")
item = started[0]
runtime_state = item.get("State", {})
if (
    item.get("Id") != str(container_id)
    or runtime_state.get("Status") not in {"running", "exited"}
    or not isinstance(runtime_state.get("Running"), bool)
    or start_result.get("container_id") != container_id
    or start_result.get("status") not in {"pass", "failed", "outcome-unavailable"}
):
    raise SystemExit("detached bootstrap host container did not enter a started state")
value = {
    "schema_version": "ace.phantom.bootstrap-host-detached-launch/1.0.0",
    "status": "started",
    "container_id": str(container_id),
    "observed_container_status": runtime_state["Status"],
    "observed_running": runtime_state["Running"],
    "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    "started_receipt_sha256": hashlib.sha256(started_path.read_bytes()).hexdigest(),
    "start_result_sha256": hashlib.sha256(start_result_path.read_bytes()).hexdigest(),
    "start_client_status": start_result["status"],
}
temporary = receipt_path.with_name(receipt_path.name + ".tmp")
if receipt_path.exists():
    if receipt_path.is_symlink() or json.loads(receipt_path.read_text(encoding="utf-8")) != value:
        raise SystemExit("detached launch receipt changed")
else:
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, receipt_path)
PY
}

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ -n "${CONTAINER_ID}" &&
        ( "${RUN_MODE}" != "detached-launch" ||
          "${DETACHED_START_ATTEMPTED}" != true ) ]]; then
    set +e
    remove_exact_container
    local cleanup_exit="$?"
    set -e
    [[ ${cleanup_exit} -eq 0 ]] || incoming=1
  fi
  exit "${incoming}"
}

detached_recovery_cleanup_trap() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ "${DETACHED_TERMINAL_OWNERSHIP_VALIDATED}" == true &&
        "${DETACHED_CLEANUP_COMPLETE}" != true ]]; then
    set +e
    ensure_detached_container_cleanup
    local cleanup_exit="$?"
    set -e
    if [[ ${cleanup_exit} -ne 0 ]]; then
      incoming=1
    fi
  fi
  exit "${incoming}"
}

verify_bootstrap_qualification() {
  local qualification="${OUTPUT}/verified-bootstrap-host-result/results/qualification"
  python3 - "${qualification}" "${PAYLOAD}" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_CONFIG}" <<'PY' || return 1
import hashlib
import json
import pathlib
import re
import sys

root, payload = map(pathlib.Path, sys.argv[1:3])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in bootstrap qualification: {key}")
        value[key] = item
    return value

def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

manifest = read_json(root / "manifest.json")
expected = {
    "status": "pass",
    "gate": "bootstrap",
    "source_mode": "snapshot",
    "ace_commit": sys.argv[3],
    "phantom_commit": sys.argv[4],
    "ace_worktree_dirty": False,
    "gpu_executables_were_run": False,
}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("verified bootstrap qualification manifest is inconsistent")
if (
    (root / "ace_source_manifest.json").read_bytes()
    != (payload / "ace-source.manifest.json").read_bytes()
    or (root / "phantom_source_manifest.json").read_bytes()
    != (payload / "phantom-source.manifest.json").read_bytes()
):
    raise SystemExit("verified bootstrap source manifests differ from the payload")
qualification = json.loads(
    (root / "bootstrap_qualification/qualification.json").read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
digest_keys = {
    "generated_source_sha256", "raw_air_sha256", "post_ckks_air_sha256",
    "post_operations_air_sha256", "compiler_context_manifest_sha256",
    "compiler_resource_manifest_sha256", "compiler_constant_manifest_sha256",
    "generation_record_sha256", "generated_artifact_audit_sha256",
    "linked_binary_sha256", "harness_source_sha256", "symbol_closure_sha256",
    "io_helper_closure_sha256", "archive_member_audit_sha256",
    "adapter_archive_sha256", "provider_archive_sha256", "common_archive_sha256",
    "fixture_sha256", "native_ant_reference_sha256", "native_ant_values_sha256",
    "generated_ant_reference_sha256", "generated_ant_values_sha256",
    "gpu_correctness_runner_sha256", "gpu_correctness_runner_source_sha256",
    "host_oracle_replay_sha256",
}
exact_keys = digest_keys | {
    "schema_version", "status", "gate", "architecture", "phantom_commit",
    "development_image_id", "development_definition_sha256", "context_contract",
    "generated_source_contains_native_bootstrap",
    "production_archive_contains_native_bootstrap",
    "primitive_only_provider_archive", "link_mode",
    "host_oracle_executables_were_run", "gpu_executable_was_run",
    "executable_was_run",
}
payload_value = read_json(payload / "payload.json")
options = payload_value.get("qualification", {}).get("options", {})
context_options = {
    "polynomial_degree": "--poly-degree",
    "vector_capacity": "--vector-capacity",
    "mul_level": "--mul-level",
    "input_level": "--input-level",
    "security_level": "--security-level",
    "scaling_factor_bits": "--scaling-factor-bits",
    "first_prime_bits": "--first-prime-bits",
    "hamming_weight": "--hamming-weight",
    "q_part_count": "--q-part-count",
}
try:
    expected_context = {
        key: int(options[option]) for key, option in context_options.items()
    }
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit("bootstrap qualification payload context is incomplete") from error
if (
    not isinstance(qualification, dict)
    or set(qualification) != exact_keys
    or qualification.get("schema_version")
    != "ace.phantom.bootstrap-host-qualification/2.0.0"
    or qualification.get("status") != "pass"
    or qualification.get("gate") != "bootstrap"
    or qualification.get("architecture") != "sm_80"
    or qualification.get("phantom_commit") != sys.argv[4]
    or qualification.get("development_image_id") != sys.argv[5]
    or qualification.get("development_definition_sha256")
    != hashlib.sha256((payload / "bootstrap_environment.sh").read_bytes()).hexdigest()
    or qualification.get("context_contract") != expected_context
    or any(
        re.fullmatch(r"[0-9a-f]{64}", qualification.get(key, "")) is None
        for key in digest_keys
    )
    or qualification.get("generated_source_contains_native_bootstrap") is not False
    or qualification.get("production_archive_contains_native_bootstrap") is not False
    or qualification.get("primitive_only_provider_archive") is not True
    or qualification.get("link_mode") != "manual_static_closure"
    or qualification.get("host_oracle_executables_were_run") is not True
    or qualification.get("gpu_executable_was_run") is not False
    or qualification.get("executable_was_run") is not True
):
    raise SystemExit("verified bootstrap semantic qualification is inconsistent")
PY
  (cd "${qualification}" && sha256sum -c SHA256SUMS) || return 1
}

verify_or_reuse_detached_result_archive() {
  local result_archive="$1"
  local verification_path="${OUTPUT}/bootstrap-host-result-verification.json"
  local extracted="${OUTPUT}/verified-bootstrap-host-result"
  if [[ -e "${verification_path}" ]]; then
    if [[ ! -f "${verification_path}" || -L "${verification_path}" ||
          ! -d "${extracted}/results" ]]; then
      echo "detached result verification phase receipt is nonregular or incomplete" >&2
      return 1
    fi
    python3 - "${verification_path}" "${PIPELINE_EXIT}" \
      "${extracted}/results" <<'PY' || return 1
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in result verification receipt: {key}")
        value[key] = item
    return value

receipt = json.loads(
    pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
)
results = pathlib.Path(sys.argv[3])
sums = results / "SHA256SUMS"
if not results.is_dir() or results.is_symlink() or not sums.is_file() or sums.is_symlink():
    raise SystemExit("verified result extraction root or manifest is nonregular")
listed = {}
for line in sums.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
    if match is None:
        raise SystemExit("verified result manifest contains a malformed entry")
    name = match.group(3).removeprefix("./")
    path = pathlib.PurePosixPath(name)
    if (
        not name or path.is_absolute() or ".." in path.parts
        or str(path) != name or name == "SHA256SUMS" or name in listed
    ):
        raise SystemExit("verified result manifest contains an unsafe or duplicate entry")
    listed[name] = match.group(1)
actual = {}
actual_directories = set()
for directory, directory_names, file_names in os.walk(results, followlinks=False):
    root = pathlib.Path(directory)
    for name in directory_names:
        path = root / name
        if not stat.S_ISDIR(path.lstat().st_mode) or path.is_symlink():
            raise SystemExit("verified result extraction contains a nonregular directory")
        actual_directories.add(path.relative_to(results).as_posix())
    for name in file_names:
        path = root / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise SystemExit("verified result extraction contains a nonregular file")
        if path != sums:
            actual[path.relative_to(results).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
expected_directories = {
    parent.as_posix()
    for name in listed
    for parent in pathlib.PurePosixPath(name).parents
    if parent.as_posix() != "."
}
if (
    set(receipt) != {
        "status", "mode", "pipeline_exit_code", "verified_file_count",
        "sha256_manifest_sha256",
    }
    or receipt.get("status") != "pass"
    or receipt.get("mode") != "bootstrap-host-freeze"
    or receipt.get("pipeline_exit_code") != int(sys.argv[2])
    or isinstance(receipt.get("verified_file_count"), bool)
    or receipt.get("verified_file_count") != len(actual)
    or receipt.get("sha256_manifest_sha256")
    != hashlib.sha256(sums.read_bytes()).hexdigest()
    or actual != listed
    or actual_directories != expected_directories
):
    raise SystemExit("detached result verification phase receipt changed")
PY
    return 0
  fi
  local verification_intent="${DOCKER_EVIDENCE}/result-verification-intent.json"
  local prepared temporary_extraction temporary_receipt
  prepared="$(python3 - "${verification_intent}" "${result_archive}" \
    "${PIPELINE_EXIT}" "${OUTPUT}" "${DOCKER_EVIDENCE}" "${extracted}" \
    "${verification_path}" <<'PY'
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys

intent_path, archive_path, exit_code, output, docker_root, extracted, receipt = (
    pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]),
    pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5]), pathlib.Path(sys.argv[6]),
    pathlib.Path(sys.argv[7]),
)
temporary_extraction = output / ".verified-bootstrap-host-result.pending"
temporary_receipt = docker_root / ".result-verification-receipt.pending"

def reject_duplicates(pairs):
    result = {}
    for key, item in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key in result verification intent: {key}")
        result[key] = item
    return result

value = {
    "schema_version": "ace.phantom.bootstrap-host-result-verification-intent/1.0.0",
    "status": "ready",
    "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    "pipeline_exit_code": exit_code,
    "temporary_extraction": temporary_extraction.name,
    "temporary_receipt": temporary_receipt.name,
    "final_extraction": extracted.name,
    "final_receipt": receipt.name,
}
if intent_path.exists():
    if not stat.S_ISREG(intent_path.lstat().st_mode) or json.loads(
        intent_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    ) != value:
        raise SystemExit("detached result verification intent changed")
else:
    temporary = intent_path.with_name(intent_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, intent_path)
if extracted.exists():
    if receipt.exists():
        print("reuse")
        raise SystemExit(0)
    if temporary_receipt.is_file() and not temporary_receipt.is_symlink():
        os.replace(temporary_receipt, receipt)
        print("promoted")
        raise SystemExit(0)
    raise SystemExit("final result extraction exists without a promotable receipt")
if temporary_extraction.exists():
    if temporary_extraction.is_symlink() or not temporary_extraction.is_dir():
        raise SystemExit("temporary result extraction is nonregular")
    shutil.rmtree(temporary_extraction)
if temporary_receipt.exists():
    if temporary_receipt.is_symlink() or not temporary_receipt.is_file():
        raise SystemExit("temporary result verification receipt is nonregular")
    temporary_receipt.unlink()
print(f"verify\t{temporary_extraction}\t{temporary_receipt}")
PY
)"
  if [[ "${prepared}" == "reuse" || "${prepared}" == "promoted" ]]; then
    verify_or_reuse_detached_result_archive "${result_archive}"
    return "$?"
  fi
  IFS=$'\t' read -r _ temporary_extraction temporary_receipt <<<"${prepared}"
  if [[ -z "${temporary_extraction}" || -z "${temporary_receipt}" ]]; then
    echo "detached result verification preparation returned invalid paths" >&2
    return 1
  fi
  local verification_exit
  set +e
  verify_result_archive "${result_archive}" bootstrap-host-freeze \
    "${PIPELINE_EXIT}" "${temporary_extraction}" >"${temporary_receipt}" \
    2>"${DOCKER_EVIDENCE}/result-archive-verification.txt"
  verification_exit="$?"
  set -e
  if [[ ${verification_exit} -ne 0 ]]; then
    rm -f -- "${temporary_receipt}"
    return "${verification_exit}"
  fi
  sync -f "${temporary_extraction}" || return 1
  sync -f "${temporary_receipt}" || return 1
  mv -- "${temporary_extraction}" "${extracted}" || return 1
  mv -- "${temporary_receipt}" "${verification_path}" || return 1
  sync -f "${verification_path}" || return 1
}

promote_incoming_result() {
  python3 - "${INCOMING_RESULTS}" "${OUTPUT}" \
    "${DOCKER_EVIDENCE}/result-promotion-intent.json" \
    "${DOCKER_EVIDENCE}/result-promotion.json" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import sys

incoming, output, intent_path, receipt_path = map(pathlib.Path, sys.argv[1:])
archive_name = "bootstrap-host-result.tar.gz"
sidecar_name = archive_name + ".sha256"
root_archive, root_sidecar = output / archive_name, output / sidecar_name

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def regular(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in result promotion evidence: {key}")
        value[key] = item
    return value

def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

def durable_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def fsync_directory(path):
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

def strict_sidecar(path):
    if not regular(path):
        raise ValueError("sidecar is not one regular file")
    raw = path.read_bytes()
    match = re.fullmatch(
        rb"([0-9a-f]{64})  bootstrap-host-result\.tar\.gz\n", raw
    )
    if match is None:
        raise ValueError("sidecar is not one exact basename-bound line")
    return match.group(1).decode("ascii")

def inventory():
    values = []
    for path in sorted(incoming.iterdir(), key=lambda item: item.name):
        mode = path.lstat().st_mode
        kind = (
            "regular" if stat.S_ISREG(mode) else "symlink" if stat.S_ISLNK(mode)
            else "directory" if stat.S_ISDIR(mode) else "nonregular"
        )
        values.append({"name": path.name, "kind": kind})
    return values

def purge_incoming():
    for path in list(incoming.iterdir()):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            shutil.rmtree(path)
        else:
            path.unlink()

def purge_path(path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(path)
    else:
        path.unlink()

def valid_inventory(value):
    if not isinstance(value, list):
        return False
    names = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "kind"}
            or not isinstance(item["name"], str)
            or not item["name"]
            or pathlib.PurePath(item["name"]).name != item["name"]
            or item["kind"] not in {"regular", "symlink", "directory", "nonregular"}
        ):
            return False
        names.append(item["name"])
    return names == sorted(names) and len(names) == len(set(names))

def reject(reason):
    observed = inventory()
    purge_incoming()
    purge_path(root_archive)
    purge_path(root_sidecar)
    durable_json(
        receipt_path,
        {
            "schema_version": "ace.phantom.bootstrap-host-result-promotion/1.0.0",
            "status": "rejected",
            "reason": reason,
            "incoming_inventory": observed,
            "incoming_cleanup": "empty",
        },
    )
    raise SystemExit(2)

if receipt_path.exists():
    if not regular(receipt_path):
        raise SystemExit("result promotion receipt is nonregular")
    receipt = load_json(receipt_path)
    status_value = receipt.get("status")
    if status_value == "rejected":
        if (
            set(receipt) != {
                "schema_version", "status", "reason", "incoming_inventory",
                "incoming_cleanup",
            }
            or receipt.get("schema_version")
            != "ace.phantom.bootstrap-host-result-promotion/1.0.0"
            or receipt.get("incoming_cleanup") != "empty"
            or not isinstance(receipt.get("reason"), str)
            or not receipt["reason"]
            or not valid_inventory(receipt.get("incoming_inventory"))
            or any(incoming.iterdir())
            or (output / archive_name).exists()
            or (output / sidecar_name).exists()
        ):
            raise SystemExit("rejected result staging is not empty")
        raise SystemExit(2)
    if status_value == "missing":
        if (
            set(receipt) != {
                "schema_version", "status", "reason", "incoming_inventory",
            }
            or receipt.get("schema_version")
            != "ace.phantom.bootstrap-host-result-promotion/1.0.0"
            or receipt.get("reason") != "incoming-results is empty"
            or receipt.get("incoming_inventory") != []
            or any(incoming.iterdir())
            or (output / archive_name).exists()
            or (output / sidecar_name).exists()
        ):
            raise SystemExit("missing result staging is not empty")
        raise SystemExit(3)
    if status_value != "promoted":
        raise SystemExit("result promotion receipt has an invalid status")
    archive, sidecar = output / archive_name, output / sidecar_name
    if (
        set(receipt) != {
            "schema_version", "status", "archive_sha256", "sidecar_sha256",
            "incoming_inventory",
        }
        or receipt.get("schema_version")
        != "ace.phantom.bootstrap-host-result-promotion/1.0.0"
        or receipt.get("incoming_inventory") != []
        or re.fullmatch(r"[0-9a-f]{64}", receipt.get("archive_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", receipt.get("sidecar_sha256", "")) is None
        or not regular(archive)
        or strict_sidecar(sidecar) != receipt.get("archive_sha256")
        or digest(archive) != receipt.get("archive_sha256")
        or digest(sidecar) != receipt.get("sidecar_sha256")
        or any(incoming.iterdir())
    ):
        raise SystemExit("promoted result receipt no longer matches its files")
    raise SystemExit(0)

if intent_path.exists():
    if not regular(intent_path):
        raise SystemExit("result promotion intent is nonregular")
    intent = load_json(intent_path)
    if set(intent) != {
        "schema_version", "status", "archive_sha256", "sidecar_sha256",
        "archive_name", "sidecar_name",
    } or intent.get("schema_version") != "ace.phantom.bootstrap-host-result-promotion-intent/1.0.0" \
            or intent.get("status") != "ready" \
            or intent.get("archive_name") != archive_name \
            or intent.get("sidecar_name") != sidecar_name \
            or re.fullmatch(r"[0-9a-f]{64}", intent.get("archive_sha256", "")) is None \
            or re.fullmatch(r"[0-9a-f]{64}", intent.get("sidecar_sha256", "")) is None:
        raise SystemExit("result promotion intent is invalid")
    allowed = {archive_name, sidecar_name}
    if any(path.name not in allowed for path in incoming.iterdir()):
        reject("unexpected entry appeared during promotion recovery")
    for name, expected in (
        (archive_name, intent["archive_sha256"]),
        (sidecar_name, intent["sidecar_sha256"]),
    ):
        source, target = incoming / name, output / name
        if source.exists() and target.exists():
            reject("promotion recovery found duplicate source and target files")
        candidate = target if target.exists() else source
        if not regular(candidate) or digest(candidate) != expected:
            reject("promotion recovery file differs from its durable intent")
        if candidate == source:
            os.replace(source, target)
            fsync_directory(incoming)
            fsync_directory(output)
    if strict_sidecar(root_sidecar) != intent["archive_sha256"]:
        reject("promoted sidecar no longer binds the archive")
    if any(incoming.iterdir()):
        reject("promotion recovery left incoming entries")
    durable_json(
        receipt_path,
        {
            "schema_version": "ace.phantom.bootstrap-host-result-promotion/1.0.0",
            "status": "promoted",
            "archive_sha256": intent["archive_sha256"],
            "sidecar_sha256": intent["sidecar_sha256"],
            "incoming_inventory": [],
        },
    )
    raise SystemExit(0)

observed = inventory()
if not observed:
    if root_archive.exists() or root_sidecar.exists():
        reject("root result artifacts exist without a promotion intent")
    durable_json(
        receipt_path,
        {
            "schema_version": "ace.phantom.bootstrap-host-result-promotion/1.0.0",
            "status": "missing",
            "reason": "incoming-results is empty",
            "incoming_inventory": [],
        },
    )
    raise SystemExit(3)
if observed != [
    {"name": archive_name, "kind": "regular"},
    {"name": sidecar_name, "kind": "regular"},
]:
    reject("incoming-results does not contain exactly two regular entries")
try:
    claimed = strict_sidecar(incoming / sidecar_name)
except ValueError as error:
    reject(str(error))
actual = digest(incoming / archive_name)
if claimed != actual:
    reject("sidecar digest does not match the result archive")
intent = {
    "schema_version": "ace.phantom.bootstrap-host-result-promotion-intent/1.0.0",
    "status": "ready",
    "archive_sha256": actual,
    "sidecar_sha256": digest(incoming / sidecar_name),
    "archive_name": archive_name,
    "sidecar_name": sidecar_name,
}
durable_json(intent_path, intent)
os.replace(incoming / archive_name, root_archive)
fsync_directory(incoming)
fsync_directory(output)
os.replace(incoming / sidecar_name, root_sidecar)
fsync_directory(incoming)
fsync_directory(output)
durable_json(
    receipt_path,
    {
        "schema_version": "ace.phantom.bootstrap-host-result-promotion/1.0.0",
        "status": "promoted",
        "archive_sha256": actual,
        "sidecar_sha256": intent["sidecar_sha256"],
        "incoming_inventory": [],
    },
)
PY
}

retrieve_detached_logs() {
  local verification_exit
  set +e
  python3 - "${OUTPUT}/bootstrap-host-freeze.log" \
    "${DOCKER_EVIDENCE}/log-retrieval.json" "${CONTAINER_ID}" verify <<'PY'
import hashlib
import json
import pathlib
import stat
import sys

log_path, receipt_path = map(pathlib.Path, sys.argv[1:3])
if not receipt_path.exists():
    raise SystemExit(10)
if not stat.S_ISREG(receipt_path.lstat().st_mode) or not log_path.exists() \
        or not stat.S_ISREG(log_path.lstat().st_mode):
    raise SystemExit("detached log or retrieval receipt is nonregular")
value = json.loads(receipt_path.read_text(encoding="utf-8"))
if value.get("status") == "failed":
    diagnostic = receipt_path.parent / "log-retrieval-failure.txt"
    expected = {
        "schema_version": "ace.phantom.bootstrap-host-log-retrieval/1.0.0",
        "status": "failed",
        "container_id": sys.argv[3],
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "log_size": log_path.stat().st_size,
        "diagnostic_sha256": hashlib.sha256(diagnostic.read_bytes()).hexdigest()
        if diagnostic.is_file() else None,
    }
    if value != expected or value["log_size"] != 0:
        raise SystemExit("detached log retrieval failure receipt changed")
    raise SystemExit(20)
expected = {
    "schema_version": "ace.phantom.bootstrap-host-log-retrieval/1.0.0",
    "status": "pass",
    "container_id": sys.argv[3],
    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    "log_size": log_path.stat().st_size,
}
if value != expected:
    raise SystemExit("detached log retrieval receipt changed")
PY
  verification_exit="$?"
  set -e
  if [[ ${verification_exit} -eq 0 ]]; then
    return 0
  fi
  if [[ ${verification_exit} -ne 10 ]]; then
    return "${verification_exit}"
  fi
  if [[ -e "${OUTPUT}/bootstrap-host-freeze.log" &&
        ( ! -f "${OUTPUT}/bootstrap-host-freeze.log" ||
          -L "${OUTPUT}/bootstrap-host-freeze.log" ) ]]; then
    echo "detached bootstrap host log path is nonregular" >&2
    return 1
  fi
  local temporary
  temporary="$(mktemp)"
  if ! docker logs "${CONTAINER_ID}" >"${temporary}" 2>&1; then
    if [[ ! -s "${temporary}" ]]; then
      printf 'docker logs returned nonzero without diagnostic output\n' >"${temporary}"
    fi
    mv -- "${temporary}" "${DOCKER_EVIDENCE}/log-retrieval-failure.txt"
    : >"${OUTPUT}/bootstrap-host-freeze.log"
    python3 - "${OUTPUT}/bootstrap-host-freeze.log" \
      "${DOCKER_EVIDENCE}/log-retrieval.json" "${CONTAINER_ID}" \
      "${DOCKER_EVIDENCE}/log-retrieval-failure.txt" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

log_path, receipt_path, diagnostic = map(pathlib.Path, (sys.argv[1], sys.argv[2], sys.argv[4]))
value = {
    "schema_version": "ace.phantom.bootstrap-host-log-retrieval/1.0.0",
    "status": "failed",
    "container_id": sys.argv[3],
    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    "log_size": 0,
    "diagnostic_sha256": hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
}
temporary = receipt_path.with_name(receipt_path.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, receipt_path)
PY
    echo "failed to retrieve detached bootstrap host container logs" >&2
    return 20
  fi
  mv -- "${temporary}" "${OUTPUT}/bootstrap-host-freeze.log"
  python3 - "${OUTPUT}/bootstrap-host-freeze.log" \
    "${DOCKER_EVIDENCE}/log-retrieval.json" "${CONTAINER_ID}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

log_path, receipt_path = map(pathlib.Path, sys.argv[1:3])
value = {
    "schema_version": "ace.phantom.bootstrap-host-log-retrieval/1.0.0",
    "status": "pass",
    "container_id": sys.argv[3],
    "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    "log_size": log_path.stat().st_size,
}
temporary = receipt_path.with_name(receipt_path.name + ".tmp")
with temporary.open("w", encoding="utf-8") as stream:
    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, receipt_path)
PY
}

close_interrupted_detached_launch() {
  PIPELINE_EXIT=1
  DETACHED_TERMINAL_OWNERSHIP_VALIDATED=true
  trap detached_recovery_cleanup_trap EXIT
  trap 'exit 130' INT TERM
  local log_retrieval_exit
  if retrieve_detached_logs; then
    log_retrieval_exit=0
  else
    log_retrieval_exit="$?"
  fi
  if [[ ${log_retrieval_exit} -eq 20 ]]; then
    close_detached_log_retrieval_failure
    return "$?"
  elif [[ ${log_retrieval_exit} -ne 0 ]]; then
    return "${log_retrieval_exit}"
  fi
  ensure_detached_container_cleanup || return 1
  set +e
  promote_incoming_result
  local promotion_exit="$?"
  set -e
  if [[ ${promotion_exit} -eq 0 ]]; then
    echo "interrupted launch unexpectedly found a result archive" >&2
    return 1
  fi
  write_result_verification_failure \
    "launch interrupted after create and before start" || return 1
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" launch-interrupted || return 1
  write_outer_bundle_checksum_closure allow-missing-result || return 1
  echo "detached bootstrap host launch was recovered and closed before start" >&2
  return 1
}

close_detached_log_retrieval_failure() {
  ensure_detached_container_cleanup || return 1
  set +e
  promote_incoming_result
  local promotion_exit="$?"
  set -e
  if [[ ${promotion_exit} -ne 0 && ${promotion_exit} -ne 2 &&
        ${promotion_exit} -ne 3 ]]; then
    echo "log-retrieval failure could not safely classify incoming results" >&2
    return 1
  fi
  write_result_verification_failure "container log retrieval failed" || return 1
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" log-retrieval-failed || return 1
  write_outer_bundle_checksum_closure allow-missing-result || return 1
  echo "detached bootstrap host log retrieval failed and evidence was closed" >&2
  return 1
}

recover_detached_create_gap() {
  local intent_path="${OUTPUT}/detached-precreate-intent.json"
  local daemon_path="${DOCKER_EVIDENCE}/daemon.json"
  for required in "${intent_path}" "${daemon_path}" "${PAYLOAD}/payload.json" \
    "${PAYLOAD}/SHA256SUMS" "${DOCKER_EVIDENCE}/base-image.json" \
    "${DOCKER_EVIDENCE}/pull.txt"; do
    if [[ ! -f "${required}" || -L "${required}" || ! -s "${required}" ]]; then
      echo "create-gap recovery receipt is missing or nonregular: ${required}" >&2
      return 1
    fi
  done
  local interrupted_capture
  for interrupted_capture in \
    "${DOCKER_EVIDENCE}/.container-created.json.capture.pending" \
    "${DOCKER_EVIDENCE}/.container-after-start.pending"; do
    if [[ -e "${interrupted_capture}" ]]; then
      if [[ ! -f "${interrupted_capture}" || -L "${interrupted_capture}" ]]; then
        echo "interrupted container capture is nonregular: ${interrupted_capture}" >&2
        return 1
      fi
      rm -f -- "${interrupted_capture}"
    fi
  done
  verify_docker_daemon_receipt || return 1
  local binding
  binding="$(python3 - "${intent_path}" "${OUTPUT}" "${PAYLOAD}" \
    "${INCOMING_RESULTS}" "${daemon_path}" \
    "${DOCKER_EVIDENCE}/base-image.json" \
    "${REPO_ROOT}/tools/phantom_gpu/freeze_bootstrap_host_evidence.sh" <<'PY'
import hashlib
import json
import pathlib
import re
import stat
import sys

intent_path, output_path, payload_path, incoming_path, daemon_path, image_path, lifecycle_path = map(
    pathlib.Path, sys.argv[1:]
)
if stat.S_IMODE(intent_path.lstat().st_mode) != 0o600:
    raise SystemExit("pre-create intent mode changed")
value = json.loads(intent_path.read_text(encoding="utf-8"))
expected_keys = {
    "schema_version", "status", "run_nonce", "output_directory",
    "output_device", "output_inode", "incoming_results_directory",
    "incoming_results_device", "incoming_results_inode", "ace_commit",
    "phantom_commit", "base_image", "base_config_digest", "container_name",
    "task_label", "build_jobs", "qualification_arguments",
    "payload_descriptor_sha256", "payload_manifest_sha256",
    "daemon_receipt_sha256", "base_image_receipt_sha256", "docker_daemon",
    "lifecycle_sha256", "container_command_sha256", "mounts",
}
if set(value) != expected_keys or value.get("schema_version") != \
        "ace.phantom.bootstrap-host-precreate-intent/1.0.0" or value.get("status") != "ready-to-create":
    raise SystemExit("pre-create intent schema changed")
output, payload, incoming = output_path.resolve(), payload_path.resolve(), incoming_path.resolve()
output_stat, incoming_stat = output.stat(), incoming.stat()
if (
    value["output_directory"] != str(output)
    or value["output_device"] != output_stat.st_dev
    or value["output_inode"] != output_stat.st_ino
    or value["incoming_results_directory"] != str(incoming)
    or value["incoming_results_device"] != incoming_stat.st_dev
    or value["incoming_results_inode"] != incoming_stat.st_ino
):
    raise SystemExit("pre-create intent path identity changed")
descriptor, manifest = payload / "payload.json", payload / "SHA256SUMS"
if (
    hashlib.sha256(descriptor.read_bytes()).hexdigest() != value["payload_descriptor_sha256"]
    or hashlib.sha256(manifest.read_bytes()).hexdigest() != value["payload_manifest_sha256"]
    or hashlib.sha256(daemon_path.read_bytes()).hexdigest() != value["daemon_receipt_sha256"]
    or hashlib.sha256(image_path.read_bytes()).hexdigest()
    != value["base_image_receipt_sha256"]
    or hashlib.sha256(lifecycle_path.read_bytes()).hexdigest()
    != value["lifecycle_sha256"]
):
    raise SystemExit("pre-create intent payload, image, daemon, or lifecycle binding changed")
image = json.loads(image_path.read_text(encoding="utf-8"))
if not isinstance(image, list) or len(image) != 1 or image[0].get("Id") != value["base_config_digest"]:
    raise SystemExit("pre-create intent base-image binding changed")
if re.fullmatch(r"[0-9a-f]{64}", value.get("run_nonce", "")) is None:
    raise SystemExit("pre-create intent nonce is invalid")
fields = [
    value["ace_commit"], value["phantom_commit"], value["base_image"],
    value["base_config_digest"], value["container_name"], value["task_label"],
    value["run_nonce"], hashlib.sha256(intent_path.read_bytes()).hexdigest(),
]
if any(not isinstance(item, str) or not item or "\t" in item or "\n" in item for item in fields):
    raise SystemExit("pre-create intent contains an invalid binding")
print("\t".join(fields))
PY
)" || return 1
  local run_nonce intent_sha
  IFS=$'\t' read -r ACE_COMMIT PHANTOM_COMMIT BASE_IMAGE BASE_CONFIG \
    CONTAINER_NAME TASK_LABEL run_nonce intent_sha <<<"${binding}"
  load_selected_transport_helpers "${ACE_COMMIT}" || return 1
  (cd "${PAYLOAD}" && sha256sum -c SHA256SUMS) || return 1
  local interrupted_state="${OUTPUT}/detached-launch-state.json"
  local interrupted_launch="${OUTPUT}/detached-launch.json"
  local interrupted_completed="${DOCKER_EVIDENCE}/container-completed.json"
  if [[ -f "${interrupted_state}" && ! -L "${interrupted_state}" &&
        -f "${interrupted_launch}" && ! -L "${interrupted_launch}" &&
        -f "${interrupted_completed}" && ! -L "${interrupted_completed}" &&
        -f "${DOCKER_EVIDENCE}/cleanup-intent.json" &&
        ! -L "${DOCKER_EVIDENCE}/cleanup-intent.json" ]]; then
    CONTAINER_ID="$(python3 - "${interrupted_state}" "${interrupted_launch}" \
      "${interrupted_completed}" "${intent_sha}" "${run_nonce}" <<'PY'
import hashlib
import json
import pathlib
import sys

state_path, launch_path, completed_path = map(pathlib.Path, sys.argv[1:4])
state = json.loads(state_path.read_text(encoding="utf-8"))
launch = json.loads(launch_path.read_text(encoding="utf-8"))
completed = json.loads(completed_path.read_text(encoding="utf-8"))
container = state.get("container", {})
if (
    launch != {
        "schema_version": "ace.phantom.bootstrap-host-detached-launch/1.0.0",
        "status": "interrupted-before-start",
        "container_id": container.get("id"),
        "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "created_receipt_sha256": state.get("created_receipt_sha256"),
    }
    or state.get("intent_sha256") != sys.argv[4]
    or state.get("run_nonce") != sys.argv[5]
    or not isinstance(completed, list)
    or len(completed) != 1
    or completed[0].get("Id") != container.get("id")
    or completed[0].get("State", {}).get("Status") != "created"
):
    raise SystemExit("interrupted create-gap recovery receipts changed")
print(container["id"])
PY
)" || return 1
    CREATED_CONTAINER_ID="${CONTAINER_ID}"
    close_interrupted_detached_launch
    return "$?"
  fi
  local inventory
  inventory="$(docker ps -aq --no-trunc \
    --filter "label=ace.phantom.intent-sha256=${intent_sha}" \
    --filter "label=ace.phantom.run-nonce=${run_nonce}")" || return 1
  if [[ ! "${inventory}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "create-gap recovery did not resolve exactly one intent-owned container" >&2
    return 1
  fi
  CONTAINER_ID="${inventory}"
  CREATED_CONTAINER_ID="${CONTAINER_ID}"
  local current_temporary="$(mktemp)"
  if ! docker inspect --type container "${CONTAINER_ID}" >"${current_temporary}"; then
    rm -f -- "${current_temporary}"
    return 1
  fi
  local recovered_status
  if ! recovered_status="$(validate_intent_owned_container \
    "${current_temporary}" "created,running,exited")"; then
    rm -f -- "${current_temporary}"
    return 1
  fi
  local created_path="${DOCKER_EVIDENCE}/container-created.json"
  if [[ -e "${interrupted_state}" ]]; then
    [[ -f "${created_path}" && ! -L "${created_path}" ]] || {
      rm -f -- "${current_temporary}"
      echo "create-gap created receipt is missing or nonregular" >&2
      return 1
    }
    validate_intent_owned_container \
      "${created_path}" "created,running,exited" >/dev/null || return 1
  else
    atomic_install_regular_file "${current_temporary}" "${created_path}" || return 1
  fi
  write_detached_launch_state || return 1
  if [[ "${recovered_status}" == "running" ||
        "${recovered_status}" == "exited" ]]; then
    local started_path="${DOCKER_EVIDENCE}/container-started.json"
    recover_detached_start_result >/dev/null || return 1
    atomic_install_regular_file "${current_temporary}" "${started_path}" || return 1
    rm -f -- "${current_temporary}"
    write_detached_launch_receipt || return 1
    if [[ "${recovered_status}" == "running" ]]; then
      echo "detached bootstrap host launch state was recovered; container is still running" >&2
      return 75
    fi
    return 0
  fi
  atomic_install_regular_file \
    "${current_temporary}" "${DOCKER_EVIDENCE}/container-completed.json" || return 1
  rm -f -- "${current_temporary}"
  python3 - "${OUTPUT}/detached-launch.json" "${CONTAINER_ID}" \
    "${OUTPUT}/detached-launch-state.json" "${created_path}" <<'PY' || return 1
import hashlib
import json
import os
import pathlib
import sys

path, state_path, created_path = map(pathlib.Path, (sys.argv[1], sys.argv[3], sys.argv[4]))
value = {
    "schema_version": "ace.phantom.bootstrap-host-detached-launch/1.0.0",
    "status": "interrupted-before-start",
    "container_id": sys.argv[2],
    "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    "created_receipt_sha256": hashlib.sha256(created_path.read_bytes()).hexdigest(),
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
  close_interrupted_detached_launch
  return "$?"
}

finalize_detached_run() {
  local lock_path="${DOCKER_EVIDENCE}/finalize.lock"
  if [[ ! -f "${lock_path}" || -L "${lock_path}" ]]; then
    echo "detached bootstrap host finalize lock is missing or nonregular" >&2
    return 1
  fi
  exec {FINALIZE_LOCK_FD}>>"${lock_path}"
  if ! flock -n "${FINALIZE_LOCK_FD}"; then
    echo "detached bootstrap host finalization is already active" >&2
    return 75
  fi
  if [[ -e "${OUTPUT}/BUNDLE_SHA256SUMS" ]]; then
    local existing_bundle_exit
    set +e
    verify_existing_outer_bundle
    existing_bundle_exit="$?"
    set -e
    if [[ ${existing_bundle_exit} -eq 0 ]]; then
      return "${EXISTING_BUNDLE_PIPELINE_EXIT}"
    fi
    if ! bundle_install_intent validate >/dev/null; then
      echo "invalid outer bundle lacks its exact install intent" >&2
      return 1
    fi
    rm -f -- "${OUTPUT}/BUNDLE_SHA256SUMS"
  fi
  local state_path="${OUTPUT}/detached-launch-state.json"
  local launch_path="${OUTPUT}/detached-launch.json"
  local started_path="${DOCKER_EVIDENCE}/container-started.json"
  local start_result_path="${DOCKER_EVIDENCE}/start-result.json"
  local created_path="${DOCKER_EVIDENCE}/container-created.json"
  local intent_path="${OUTPUT}/detached-precreate-intent.json"
  local daemon_path="${DOCKER_EVIDENCE}/daemon.json"
  if [[ ! -e "${state_path}" || ! -e "${launch_path}" ||
        ! -e "${started_path}" || ! -e "${start_result_path}" ]]; then
    local recovery_exit
    set +e
    recover_detached_create_gap
    recovery_exit="$?"
    set -e
    if [[ ${recovery_exit} -ne 0 ]]; then
      return "${recovery_exit}"
    fi
  fi
  for required in "${state_path}" "${launch_path}" "${started_path}" \
    "${start_result_path}" \
    "${created_path}" "${PAYLOAD}/payload.json" "${PAYLOAD}/SHA256SUMS" \
    "${DOCKER_EVIDENCE}/base-image.json" "${DOCKER_EVIDENCE}/pull.txt" \
    "${intent_path}" "${daemon_path}"; do
    if [[ ! -f "${required}" || -L "${required}" || ! -s "${required}" ]]; then
      echo "detached bootstrap host state is missing or empty: ${required}" >&2
      return 1
    fi
  done
  for required in "${DOCKER_EVIDENCE}/start.txt" \
    "${DOCKER_EVIDENCE}/start-stderr.txt"; do
    if [[ ! -f "${required}" || -L "${required}" ]]; then
      echo "detached Docker start stream is missing or nonregular: ${required}" >&2
      return 1
    fi
  done
  python3 - "${PAYLOAD}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
entries = list(root.iterdir())
if any(not entry.is_file() or entry.is_symlink() for entry in entries):
    raise SystemExit("detached bootstrap host payload contains a nonregular entry")
lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
listed = []
for line in lines:
    match = re.fullmatch(r"[0-9a-f]{64}  ([^/\n]+)", line)
    if match is None:
        raise SystemExit("detached bootstrap host payload manifest is malformed")
    listed.append(match.group(1))
if len(listed) != len(set(listed)):
    raise SystemExit("detached bootstrap host payload manifest contains duplicates")
if sorted(entry.name for entry in entries) != sorted(listed + ["SHA256SUMS"]):
    raise SystemExit("detached bootstrap host payload inventory changed")
PY
  (cd "${PAYLOAD}" && sha256sum -c SHA256SUMS)
  verify_docker_daemon_receipt
  local binding
  binding="$(python3 - "${state_path}" "${launch_path}" "${created_path}" \
    "${started_path}" "${start_result_path}" "${OUTPUT}" "${PAYLOAD}" \
    "${INCOMING_RESULTS}" \
    "${DOCKER_EVIDENCE}/base-image.json" "${intent_path}" "${daemon_path}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

state_path, launch_path, created_path, started_path, start_result_path, output_path, payload_path, incoming_path, image_path, intent_path, daemon_path = map(
    pathlib.Path, sys.argv[1:]
)
output = output_path.resolve()
payload = payload_path.resolve()
incoming = incoming_path.resolve()
state = json.loads(state_path.read_text(encoding="utf-8"))
expected_state_keys = {
    "schema_version", "status", "output_directory", "output_device",
    "output_inode", "incoming_results_directory", "incoming_results_device",
    "incoming_results_inode", "ace_commit", "phantom_commit", "base_image",
    "base_config_digest", "qualification_arguments",
    "payload_descriptor_sha256", "payload_manifest_sha256",
    "created_receipt_sha256", "intent_sha256", "run_nonce",
    "daemon_receipt_sha256", "container",
}
if set(state) != expected_state_keys:
    raise SystemExit("detached launch state has an unexpected schema")
if (
    state["schema_version"] != "ace.phantom.bootstrap-host-detached-state/1.0.0"
    or state["status"] != "created"
    or re.fullmatch(r"[0-9a-f]{40}", state["ace_commit"] or "") is None
    or re.fullmatch(r"[0-9a-f]{40}", state["phantom_commit"] or "") is None
):
    raise SystemExit("detached launch state identity is invalid")
stat = output.stat()
if (
    state["output_directory"] != str(output)
    or state["output_device"] != stat.st_dev
    or state["output_inode"] != stat.st_ino
):
    raise SystemExit("detached launch output directory identity changed")
incoming_stat = incoming.stat()
if (
    state["incoming_results_directory"] != str(incoming)
    or state["incoming_results_device"] != incoming_stat.st_dev
    or state["incoming_results_inode"] != incoming_stat.st_ino
):
    raise SystemExit("detached launch incoming-results identity changed")
descriptor_path = payload / "payload.json"
manifest_path = payload / "SHA256SUMS"
if (
    hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    != state["payload_descriptor_sha256"]
    or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    != state["payload_manifest_sha256"]
    or hashlib.sha256(created_path.read_bytes()).hexdigest()
    != state["created_receipt_sha256"]
    or hashlib.sha256(intent_path.read_bytes()).hexdigest() != state["intent_sha256"]
    or hashlib.sha256(daemon_path.read_bytes()).hexdigest()
    != state["daemon_receipt_sha256"]
):
    raise SystemExit("detached launch payload or created receipt changed")
descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
intent = json.loads(intent_path.read_text(encoding="utf-8"))
if (
    descriptor.get("ace_commit") != state["ace_commit"]
    or descriptor.get("phantom_commit") != state["phantom_commit"]
    or descriptor.get("qualification", {}).get("arguments")
    != state["qualification_arguments"]
    or intent.get("run_nonce") != state["run_nonce"]
    or intent.get("ace_commit") != state["ace_commit"]
    or intent.get("phantom_commit") != state["phantom_commit"]
):
    raise SystemExit("detached launch qualification binding changed")
image = json.loads(image_path.read_text(encoding="utf-8"))
if (
    not isinstance(image, list)
    or len(image) != 1
    or image[0].get("Id") != state["base_config_digest"]
):
    raise SystemExit("detached launch base-image receipt changed")

def normalized_container(item):
    config = item.get("Config", {})
    host = item.get("HostConfig", {})
    mounts = sorted(
        [
            {
                "type": mount.get("Type"),
                "source": mount.get("Source"),
                "destination": mount.get("Destination"),
                "mode": mount.get("Mode"),
                "read_write": mount.get("RW"),
                "propagation": mount.get("Propagation"),
            }
            for mount in item.get("Mounts", [])
        ],
        key=lambda value: value["destination"] or "",
    )
    return {
        "id": item.get("Id"),
        "name": item.get("Name"),
        "task_label": (config.get("Labels") or {}).get("ace.phantom.task"),
        "image_config_digest": item.get("Image"),
        "configuration": {
            "image": config.get("Image"),
            "entrypoint": config.get("Entrypoint"),
            "command": config.get("Cmd"),
            "environment": sorted(config.get("Env") or []),
            "labels": config.get("Labels") or {},
            "user": config.get("User") or "",
            "working_directory": config.get("WorkingDir") or "",
            "runtime": host.get("Runtime"),
            "privileged": host.get("Privileged"),
            "devices": host.get("Devices") or [],
            "device_requests": host.get("DeviceRequests") or [],
            "restart_policy": host.get("RestartPolicy"),
            "network_mode": host.get("NetworkMode"),
            "mounts": mounts,
        },
    }

container = state["container"]
if set(container) != {"id", "name", "task_label", "image_config_digest", "configuration"}:
    raise SystemExit("detached launch container binding has an unexpected schema")
configuration_keys = {
    "image", "entrypoint", "command", "environment", "labels", "user",
    "working_directory", "runtime", "privileged", "devices",
    "device_requests", "restart_policy", "network_mode", "mounts",
}
if set(container.get("configuration", {})) != configuration_keys:
    raise SystemExit("detached launch container configuration has an unexpected schema")
created = json.loads(created_path.read_text(encoding="utf-8"))
started = json.loads(started_path.read_text(encoding="utf-8"))
if not isinstance(created, list) or len(created) != 1 or not isinstance(started, list) or len(started) != 1:
    raise SystemExit("detached launch Docker receipts are not singular")
if normalized_container(created[0]) != container or normalized_container(started[0]) != container:
    raise SystemExit("detached launch Docker configuration changed")
started_state = started[0].get("State", {})
if (
    started_state.get("Status") not in {"running", "exited"}
    or not isinstance(started_state.get("Running"), bool)
):
    raise SystemExit("detached launch receipt did not observe a started state")
expected_mounts = {
    (str(payload), "/bootstrap-freeze/input", False),
    (str(incoming), "/bootstrap-freeze/output", True),
}
observed_mounts = {
    (item["source"], item["destination"], item["read_write"])
    for item in container["configuration"]["mounts"]
}
if expected_mounts != observed_mounts:
    raise SystemExit("detached launch mount topology changed")
launch = json.loads(launch_path.read_text(encoding="utf-8"))
if set(launch) != {
    "schema_version", "status", "container_id", "state_sha256",
    "started_receipt_sha256", "start_result_sha256", "start_client_status",
    "observed_container_status", "observed_running",
}:
    raise SystemExit("detached launch receipt has an unexpected schema")
if (
    launch["schema_version"] != "ace.phantom.bootstrap-host-detached-launch/1.0.0"
    or launch["status"] != "started"
    or launch["container_id"] != container["id"]
    or launch["observed_container_status"] != started_state.get("Status")
    or launch["observed_running"] != started_state.get("Running")
    or launch["state_sha256"] != hashlib.sha256(state_path.read_bytes()).hexdigest()
    or launch["started_receipt_sha256"]
    != hashlib.sha256(started_path.read_bytes()).hexdigest()
    or launch["start_result_sha256"]
    != hashlib.sha256(start_result_path.read_bytes()).hexdigest()
    or launch["start_client_status"]
    != json.loads(start_result_path.read_text(encoding="utf-8")).get("status")
):
    raise SystemExit("detached launch receipt binding changed")
fields = [
    state["ace_commit"], state["phantom_commit"], state["base_image"],
    state["base_config_digest"], container["id"], container["name"][1:],
    container["task_label"],
]
if any(not isinstance(item, str) or not item or "\t" in item or "\n" in item for item in fields):
    raise SystemExit("detached launch binding contains an invalid field")
print("\t".join(fields))
PY
)"
  IFS=$'\t' read -r ACE_COMMIT PHANTOM_COMMIT BASE_IMAGE BASE_CONFIG \
    CONTAINER_ID CONTAINER_NAME TASK_LABEL <<<"${binding}"
  CREATED_CONTAINER_ID="${CONTAINER_ID}"
  verify_detached_start_result >/dev/null
  load_selected_transport_helpers "${ACE_COMMIT}"

  local completed_path="${DOCKER_EVIDENCE}/container-completed.json"
  local completed_temporary="" completed_source pipeline_value
  if [[ -e "${completed_path}" ]]; then
    if [[ ! -f "${completed_path}" || -L "${completed_path}" ]]; then
      echo "detached completed-container receipt is nonregular" >&2
      return 1
    fi
    completed_source="${completed_path}"
  else
    completed_temporary="$(mktemp)"
    if ! docker inspect --type container "${CONTAINER_ID}" >"${completed_temporary}"; then
      rm -f -- "${completed_temporary}"
      echo "detached bootstrap host container is absent before terminal receipt" >&2
      return 1
    fi
    completed_source="${completed_temporary}"
  fi
  if ! pipeline_value="$(python3 - "${state_path}" "${completed_source}" <<'PY'
import json
import pathlib
import sys

state = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
record = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(record, list) or len(record) != 1:
    raise SystemExit("Docker did not return one completed-container record")
item = record[0]
config = item.get("Config", {})
host = item.get("HostConfig", {})
mounts = sorted(
    [
        {
            "type": mount.get("Type"), "source": mount.get("Source"),
            "destination": mount.get("Destination"), "mode": mount.get("Mode"),
            "read_write": mount.get("RW"), "propagation": mount.get("Propagation"),
        }
        for mount in item.get("Mounts", [])
    ],
    key=lambda value: value["destination"] or "",
)
observed = {
    "id": item.get("Id"), "name": item.get("Name"),
    "task_label": (config.get("Labels") or {}).get("ace.phantom.task"),
    "image_config_digest": item.get("Image"),
    "configuration": {
        "image": config.get("Image"), "entrypoint": config.get("Entrypoint"),
        "command": config.get("Cmd"), "environment": sorted(config.get("Env") or []),
        "labels": config.get("Labels") or {}, "user": config.get("User") or "",
        "working_directory": config.get("WorkingDir") or "",
        "runtime": host.get("Runtime"), "privileged": host.get("Privileged"),
        "devices": host.get("Devices") or [],
        "device_requests": host.get("DeviceRequests") or [],
        "restart_policy": host.get("RestartPolicy"),
        "network_mode": host.get("NetworkMode"), "mounts": mounts,
    },
}
if observed != state["container"]:
    raise SystemExit("detached bootstrap host container configuration or identity changed")
runtime_state = item.get("State", {})
if runtime_state.get("Running") is True:
    print("running")
    raise SystemExit(0)
if runtime_state.get("Status") != "exited" or not isinstance(runtime_state.get("ExitCode"), int):
    raise SystemExit("detached bootstrap host container did not reach a clean terminal state")
print(runtime_state["ExitCode"])
PY
)"; then
    [[ -z "${completed_temporary}" ]] || rm -f -- "${completed_temporary}"
    return 1
  fi
  if [[ "${pipeline_value}" == "running" ]]; then
    [[ -z "${completed_temporary}" ]] || rm -f -- "${completed_temporary}"
    echo "detached bootstrap host container is still running; retry finalization later" >&2
    return 75
  fi
  PIPELINE_EXIT="${pipeline_value}"
  if [[ -n "${completed_temporary}" ]]; then
    mv -- "${completed_temporary}" "${completed_path}"
  fi
  DETACHED_TERMINAL_OWNERSHIP_VALIDATED=true
  trap detached_recovery_cleanup_trap EXIT
  trap 'exit 130' INT TERM
  local log_retrieval_exit
  if retrieve_detached_logs; then
    log_retrieval_exit=0
  else
    log_retrieval_exit="$?"
  fi
  if [[ ${log_retrieval_exit} -eq 20 ]]; then
    close_detached_log_retrieval_failure
    return "$?"
  elif [[ ${log_retrieval_exit} -ne 0 ]]; then
    return "${log_retrieval_exit}"
  fi
  ensure_detached_container_cleanup

  local result_archive="${OUTPUT}/bootstrap-host-result.tar.gz"
  local promotion_exit
  set +e
  promote_incoming_result
  promotion_exit="$?"
  set -e
  if [[ ${promotion_exit} -eq 3 ]]; then
    write_result_verification_failure "result archive or sidecar is missing"
    write_lifecycle_receipt failed "${PIPELINE_EXIT}" missing
    write_outer_bundle_checksum_closure allow-missing-result
    echo "detached bootstrap host result archive is missing" >&2
    return 1
  fi
  if [[ ${promotion_exit} -eq 2 ]]; then
    write_result_verification_failure "incoming result entries or sidecar were rejected"
    write_lifecycle_receipt failed "${PIPELINE_EXIT}" rejected
    write_outer_bundle_checksum_closure allow-missing-result
    echo "detached bootstrap host incoming result was rejected" >&2
    return 1
  fi
  if [[ ${promotion_exit} -ne 0 ]]; then
    echo "detached bootstrap host result promotion could not be recovered" >&2
    return 1
  fi
  printf 'strict one-line basename-bound sidecar and archive digest: pass\n' \
    >"${DOCKER_EVIDENCE}/result-sidecar-check.txt"
  if ! verify_or_reuse_detached_result_archive "${result_archive}"; then
    write_result_verification_failure "result archive structure verification failed"
    write_lifecycle_receipt failed "${PIPELINE_EXIT}" rejected
    write_outer_bundle_checksum_closure
    echo "detached bootstrap host result archive is invalid" >&2
    return 1
  fi
  if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
    write_lifecycle_receipt failed "${PIPELINE_EXIT}" verified
    write_outer_bundle_checksum_closure
    echo "detached bootstrap host failed with exit ${PIPELINE_EXIT}" >&2
    return "${PIPELINE_EXIT}"
  fi
  if ! verify_bootstrap_qualification \
      >"${DOCKER_EVIDENCE}/qualification-verification.txt" 2>&1; then
    write_lifecycle_receipt failed "${PIPELINE_EXIT}" qualification-rejected
    write_outer_bundle_checksum_closure
    echo "detached bootstrap host qualification verification failed" >&2
    return 1
  fi
  write_lifecycle_receipt pass "${PIPELINE_EXIT}" verified
  write_outer_bundle_checksum_closure
  echo "bootstrap host evidence: ${OUTPUT}/verified-bootstrap-host-result/results/qualification"
}

if [[ "${RUN_MODE}" == "detached-finalize" ]]; then
  finalize_detached_run
  exit "$?"
fi

trap cleanup EXIT
trap 'exit 130' INT TERM

docker pull --platform linux/amd64 "${BASE_IMAGE}" |
  tee "${DOCKER_EVIDENCE}/pull.txt"
ACTUAL_BASE_ID="$(docker image inspect -f '{{.Id}}' "${BASE_IMAGE}")"
if [[ "${ACTUAL_BASE_ID}" != "${BASE_CONFIG}" ]]; then
  echo "pulled base config digest does not match the selected commit lock" >&2
  exit 1
fi
docker image inspect "${BASE_IMAGE}" >"${DOCKER_EVIDENCE}/base-image.json"

read -r -d '' BOOTSTRAP_HOST_CONTAINER_COMMAND <<'BOOTSTRAP_HOST_CONTAINER_COMMAND_EOF' || true
set -euo pipefail
export LC_ALL=C TZ=UTC
INPUT=/bootstrap-freeze/input
OUTPUT=/bootstrap-freeze/output
WORK=/retained-qualification/work
RESULTS=${WORK}/results
RESULT_ARCHIVE=${OUTPUT}/bootstrap-host-result.tar.gz
STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PIPELINE_EXIT=1
mkdir -p "${RESULTS}"
finalize() {
  local incoming=$?
  trap - EXIT INT TERM
  if [[ ${PIPELINE_EXIT} -eq 0 ]]; then incoming=0; fi
  local status=failed
  [[ ${incoming} -eq 0 ]] && status=pass
  printf "{\"schema_version\":\"1.0.0\",\"status\":\"%s\",\"mode\":\"bootstrap-host-freeze\",\"exit_code\":%d,\"started_utc\":\"%s\",\"completed_utc\":\"%s\"}\n" \
    "${status}" "${incoming}" "${STARTED_UTC}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RESULTS}/pipeline-result.json"
  (
    cd "${RESULTS}"
    find . -type f ! -path ./SHA256SUMS -print0 |
      LC_ALL=C sort -z | xargs -0 -r sha256sum >SHA256SUMS
  )
  local temporary=${RESULT_ARCHIVE}.tmp
  rm -f -- "${temporary}"
  tar -C "${WORK}" -czf "${temporary}" results
  mv "${temporary}" "${RESULT_ARCHIVE}"
  (cd "$(dirname -- "${RESULT_ARCHIVE}")" &&
    sha256sum "$(basename -- "${RESULT_ARCHIVE}")" \
      >"$(basename -- "${RESULT_ARCHIVE}").sha256")
  exit "${incoming}"
}
trap finalize EXIT
trap "exit 130" INT TERM
if [[ "${NVIDIA_VISIBLE_DEVICES:-void}" != void ]] ||
   compgen -G "/dev/nvidia*" >/dev/null; then
  echo "bootstrap host freeze refuses an NVIDIA device" >&2
  exit 1
fi
(cd "${INPUT}" && sha256sum -c SHA256SUMS)
ACE_APT_LOCK=${INPUT}/apt-packages.lock \
ACE_PYTHON_LOCK=${INPUT}/python-requirements-hashed.lock \
ACE_BASE_FILES_LOCK=${INPUT}/base-files.sha256 \
ACE_DEPENDENCIES_LOCK=${INPUT}/dependencies.env \
  bash "${INPUT}/bootstrap_environment.sh" "${RESULTS}/environment"
export PATH=/opt/ace-runpod-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
python3 - "${INPUT}/payload.json" "${INPUT}/ace-source.manifest.json" \
  "${INPUT}/phantom-source.manifest.json" <<"PY"
import hashlib, json, re, sys
from pathlib import Path, PurePosixPath
payload_path, ace_path, phantom_path = map(Path, sys.argv[1:])
payload = json.loads(payload_path.read_text(encoding="utf-8"))
if payload.get("schema_version") != "ace.phantom.bootstrap-host-freeze-payload/2.0.0":
    raise SystemExit("bootstrap host-freeze payload schema mismatch")
bindings = {}
for kind, path in (("ace", ace_path), ("phantom", phantom_path)):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    commit, archive = manifest.get("commit"), manifest.get("archive")
    if (manifest.get("kind") != kind or not isinstance(commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or archive != f"{kind}-source-{commit}.tar.gz"
            or PurePosixPath(archive).name != archive
            or payload.get(f"{kind}_commit") != commit):
        raise SystemExit(f"invalid {kind} source snapshot identity")
    bindings[f"{kind}_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
if payload.get("source_snapshots") != bindings:
    raise SystemExit("payload source-manifest binding mismatch")
PY
ACE_ARCHIVE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"archive\"])" "${INPUT}/ace-source.manifest.json")
PHANTOM_ARCHIVE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"archive\"])" "${INPUT}/phantom-source.manifest.json")
python3 "${INPUT}/source_archive.py" audit --kind ace \
  --archive "${INPUT}/${ACE_ARCHIVE}" \
  --manifest "${INPUT}/ace-source.manifest.json" \
  --extract "${WORK}/ace-extract" >"${RESULTS}/ace-source-audit.json"
python3 "${INPUT}/source_archive.py" audit --kind phantom \
  --archive "${INPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${INPUT}/phantom-source.manifest.json" \
  --extract "${WORK}/phantom-extract" >"${RESULTS}/phantom-source-audit.json"
ACE_SOURCE=${WORK}/ace-extract/ace-source
PHANTOM_SOURCE=${WORK}/phantom-extract/phantom-source
ACE_COMMIT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"commit\"])" "${INPUT}/ace-source.manifest.json")
export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"commit_timestamp\"])" "${INPUT}/ace-source.manifest.json")
ACE_MANIFEST_SHA=$(sha256sum "${INPUT}/ace-source.manifest.json" | awk "{print \$1}")
PHANTOM_MANIFEST_SHA=$(sha256sum "${INPUT}/phantom-source.manifest.json" | awk "{print \$1}")
BOOTSTRAP_SHA=$(sha256sum "${INPUT}/bootstrap_environment.sh" | awk "{print \$1}")
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH=${ACE_SOURCE}
export CMAKE_CUDA_ARCHITECTURES=80 CUDAARCHS=80
export ACE_PHANTOM_TOOLCHAIN=12.4.1-sm80
export ACE_PHANTOM_SOURCE_MODE=snapshot
export ACE_PHANTOM_REPO_ROOT=${ACE_SOURCE}
export ACE_PHANTOM_SOURCE_DIR=${PHANTOM_SOURCE}
export ACE_PHANTOM_STATE_ROOT=${WORK}/state
export ACE_PHANTOM_ACE_COMMIT=${ACE_COMMIT}
export ACE_PHANTOM_SOURCE_MANIFEST=${INPUT}/ace-source.manifest.json
export ACE_PHANTOM_SOURCE_MANIFEST_SHA256=${ACE_MANIFEST_SHA}
export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST=${INPUT}/phantom-source.manifest.json
export ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256=${PHANTOM_MANIFEST_SHA}
export ACE_RUNPOD_BOOTSTRAP_SHA256=${BOOTSTRAP_SHA}
mapfile -d "" -t QUALIFICATION_ARGS < <(python3 - "${INPUT}/payload.json" <<"PY"
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
arguments = value.get("qualification", {}).get("arguments")
if not isinstance(arguments, list) or not arguments or not all(
    isinstance(item, str) and item for item in arguments
):
    raise SystemExit("payload qualification arguments are invalid")
sys.stdout.buffer.write(b"\0".join(item.encode() for item in arguments) + b"\0")
PY
)
bash "${ACE_SOURCE}/tools/phantom_gpu/compile_only.sh" \
  --gate bootstrap "${QUALIFICATION_ARGS[@]}" 2>&1 |
  tee "${RESULTS}/bootstrap-host-qualification.log"
CURRENT=${ACE_PHANTOM_STATE_ROOT}/compile_only_results/current-bootstrap.json
RUN_ROOT=$(python3 - "${CURRENT}" "${ACE_PHANTOM_STATE_ROOT}" <<"PY"
import json, pathlib, sys
current = json.load(open(sys.argv[1], encoding="utf-8"))
state = pathlib.Path(sys.argv[2]).resolve()
run = pathlib.Path(current.get("run_root", "")).resolve()
expected = (state / "compile_only_results/runs").resolve()
if current.get("status") != "pass" or current.get("gate") != "bootstrap":
    raise SystemExit("bootstrap qualification did not publish a passing current record")
if run.parent != expected or not run.is_dir():
    raise SystemExit("bootstrap qualification run root escaped its state directory")
print(run)
PY
)
cp -a -- "${RUN_ROOT}" "${RESULTS}/qualification"
(cd "${RESULTS}/qualification" && sha256sum -c SHA256SUMS)
python3 - "${RESULTS}/qualification" "${INPUT}" \
  "${ACE_RUNPOD_BASE_CONFIG_DIGEST}" "${BOOTSTRAP_SHA}" <<"PY"
import hashlib, json, pathlib, re, sys
root, source = map(pathlib.Path, sys.argv[1:3])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in bootstrap qualification: {key}")
        value[key] = item
    return value

def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

manifest = read_json(root / "manifest.json")
ace = read_json(source / "ace-source.manifest.json")
phantom = read_json(source / "phantom-source.manifest.json")
expected = {"status": "pass", "gate": "bootstrap", "source_mode": "snapshot",
            "ace_commit": ace["commit"], "phantom_commit": phantom["commit"],
            "ace_worktree_dirty": False, "gpu_executables_were_run": False}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("bootstrap snapshot qualification manifest is inconsistent")
if ((root / "ace_source_manifest.json").read_bytes()
        != (source / "ace-source.manifest.json").read_bytes()
        or (root / "phantom_source_manifest.json").read_bytes()
        != (source / "phantom-source.manifest.json").read_bytes()):
    raise SystemExit("bootstrap qualification source manifests are stale")
qualification = read_json(root / "bootstrap_qualification/qualification.json")
digest_keys = {
    "generated_source_sha256", "raw_air_sha256", "post_ckks_air_sha256",
    "post_operations_air_sha256", "compiler_context_manifest_sha256",
    "compiler_resource_manifest_sha256", "compiler_constant_manifest_sha256",
    "generation_record_sha256", "generated_artifact_audit_sha256",
    "linked_binary_sha256", "harness_source_sha256", "symbol_closure_sha256",
    "io_helper_closure_sha256", "archive_member_audit_sha256",
    "adapter_archive_sha256", "provider_archive_sha256", "common_archive_sha256",
    "fixture_sha256", "native_ant_reference_sha256", "native_ant_values_sha256",
    "generated_ant_reference_sha256", "generated_ant_values_sha256",
    "gpu_correctness_runner_sha256", "gpu_correctness_runner_source_sha256",
    "host_oracle_replay_sha256",
}
exact_keys = digest_keys | {
    "schema_version", "status", "gate", "architecture", "phantom_commit",
    "development_image_id", "development_definition_sha256", "context_contract",
    "generated_source_contains_native_bootstrap",
    "production_archive_contains_native_bootstrap",
    "primitive_only_provider_archive", "link_mode",
    "host_oracle_executables_were_run", "gpu_executable_was_run",
    "executable_was_run",
}
payload_value = read_json(source / "payload.json")
options = payload_value.get("qualification", {}).get("options", {})
context_options = {
    "polynomial_degree": "--poly-degree", "vector_capacity": "--vector-capacity",
    "mul_level": "--mul-level", "input_level": "--input-level",
    "security_level": "--security-level",
    "scaling_factor_bits": "--scaling-factor-bits",
    "first_prime_bits": "--first-prime-bits", "hamming_weight": "--hamming-weight",
    "q_part_count": "--q-part-count",
}
try:
    expected_context = {
        key: int(options[option]) for key, option in context_options.items()
    }
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit("bootstrap qualification payload context is incomplete") from error
if (
        not isinstance(qualification, dict)
        or set(qualification) != exact_keys
        or qualification.get("schema_version")
        != "ace.phantom.bootstrap-host-qualification/2.0.0"
        or qualification.get("status") != "pass"
        or qualification.get("gate") != "bootstrap"
        or qualification.get("architecture") != "sm_80"
        or qualification.get("phantom_commit") != phantom["commit"]
        or qualification.get("development_image_id") != sys.argv[3]
        or qualification.get("development_definition_sha256") != sys.argv[4]
        or qualification.get("context_contract") != expected_context
        or any(
            re.fullmatch(r"[0-9a-f]{64}", qualification.get(key, "")) is None
            for key in digest_keys
        )
        or qualification.get("generated_source_contains_native_bootstrap") is not False
        or qualification.get("production_archive_contains_native_bootstrap") is not False
        or qualification.get("primitive_only_provider_archive") is not True
        or qualification.get("link_mode") != "manual_static_closure"
        or qualification.get("host_oracle_executables_were_run") is not True
        or qualification.get("gpu_executable_was_run") is not False
        or qualification.get("executable_was_run") is not True
):
    raise SystemExit("bootstrap host qualification claims are inconsistent")
PY
printf "{\"schema_version\":\"ace.phantom.result-completeness/1.0.0\",\"status\":\"pass\",\"mode\":\"bootstrap-host-freeze\"}\n" \
  >"${RESULTS}/result-completeness.json"
PIPELINE_EXIT=0
BOOTSTRAP_HOST_CONTAINER_COMMAND_EOF

BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
CREATE_IDENTITY_ARGUMENTS=(--label "ace.phantom.task=${TASK_LABEL}")
CREATE_INTENT_ENVIRONMENT=()
if [[ "${RUN_MODE}" == "detached-launch" ]]; then
  write_docker_daemon_receipt
  : >"${DOCKER_EVIDENCE}/finalize.lock"
  chmod 0600 "${DOCKER_EVIDENCE}/finalize.lock"
  exec {DETACHED_LAUNCH_LOCK_FD}>>"${DOCKER_EVIDENCE}/finalize.lock"
  if ! flock -n "${DETACHED_LAUNCH_LOCK_FD}"; then
    echo "detached bootstrap host launch lock is already held" >&2
    exit 75
  fi
  write_detached_precreate_intent
  CREATE_IDENTITY_ARGUMENTS+=(
    --label "ace.phantom.intent-sha256=${INTENT_SHA256}"
    --label "ace.phantom.run-nonce=${RUN_NONCE}"
  )
  CREATE_INTENT_ENVIRONMENT=(
    --env "ACE_PHANTOM_INTENT_SHA256=${INTENT_SHA256}"
    --env "ACE_PHANTOM_RUN_NONCE=${RUN_NONCE}"
  )
fi

CREATED_CONTAINER_ID="$(docker create \
  --name "${CONTAINER_NAME}" \
  "${CREATE_IDENTITY_ARGUMENTS[@]}" \
  --platform linux/amd64 \
  --runtime runc \
  --restart no \
  --network bridge \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${BUILD_JOBS}" \
  "${CREATE_INTENT_ENVIRONMENT[@]}" \
  --volume "${PAYLOAD}:/bootstrap-freeze/input:ro" \
  --volume "${INCOMING_RESULTS}:/bootstrap-freeze/output:rw" \
  "${BASE_IMAGE}" bash -lc "${BOOTSTRAP_HOST_CONTAINER_COMMAND}")"
if [[ ! "${CREATED_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "docker create did not return one exact full container ID" >&2
  exit 1
fi
CONTAINER_ID="${CREATED_CONTAINER_ID}"
if [[ "${RUN_MODE}" == "detached-launch" ]]; then
  capture_container_receipt \
    "${DOCKER_EVIDENCE}/container-created.json" created >/dev/null
else
  CREATED_RECEIPT_CAPTURE="$(mktemp)"
  docker inspect --type container "${CONTAINER_ID}" >"${CREATED_RECEIPT_CAPTURE}"
  atomic_install_regular_file \
    "${CREATED_RECEIPT_CAPTURE}" "${DOCKER_EVIDENCE}/container-created.json"
  rm -f -- "${CREATED_RECEIPT_CAPTURE}"
fi
python3 - "${DOCKER_EVIDENCE}/container-created.json" \
  "${CONTAINER_ID}" "${CONTAINER_NAME}" "${TASK_LABEL}" \
  "${BASE_CONFIG}" <<'PY'
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(record, list) or len(record) != 1:
    raise SystemExit("Docker did not return one created-container record")
item = record[0]
host, config = item.get("HostConfig", {}), item.get("Config", {})
environment = set(config.get("Env") or [])
if (item.get("Id") != sys.argv[2] or item.get("Name") != "/" + sys.argv[3]
        or (config.get("Labels") or {}).get("ace.phantom.task") != sys.argv[4]
        or item.get("Image") != sys.argv[5] or host.get("Privileged") is not False
        or host.get("Runtime") != "runc" or host.get("Devices") not in (None, [])
        or host.get("DeviceRequests") not in (None, [])
        or (host.get("RestartPolicy") or {}).get("Name") != "no"
        or "NVIDIA_VISIBLE_DEVICES=void" not in environment
        or "NVIDIA_DRIVER_CAPABILITIES=none" not in environment):
    raise SystemExit("created container violates the bootstrap host-freeze device boundary")
PY

if [[ "${RUN_MODE}" == "detached-launch" ]]; then
  write_detached_launch_state
  START_STDOUT_PENDING="${DOCKER_EVIDENCE}/.start.stdout.pending"
  START_STDERR_PENDING="${DOCKER_EVIDENCE}/.start.stderr.pending"
  for START_PATH in "${START_STDOUT_PENDING}" "${START_STDERR_PENDING}" \
    "${DOCKER_EVIDENCE}/start.txt" "${DOCKER_EVIDENCE}/start-stderr.txt" \
    "${DOCKER_EVIDENCE}/start-result.json"; do
    if [[ -e "${START_PATH}" ]]; then
      echo "Docker start evidence path already exists: ${START_PATH}" >&2
      exit 1
    fi
  done
  : >"${START_STDOUT_PENDING}"
  : >"${START_STDERR_PENDING}"
  chmod 0600 "${START_STDOUT_PENDING}" "${START_STDERR_PENDING}"
  DETACHED_START_ATTEMPTED=true
  set +e
  docker start "${CONTAINER_ID}" \
    >"${START_STDOUT_PENDING}" 2>"${START_STDERR_PENDING}"
  START_CLIENT_EXIT="$?"
  set -e
  START_CLIENT_STATUS=failed
  if [[ ${START_CLIENT_EXIT} -eq 0 ]] &&
     [[ "$(cat -- "${START_STDOUT_PENDING}")" == "${CONTAINER_ID}" ]] &&
     [[ "$(wc -l <"${START_STDOUT_PENDING}")" -eq 1 ]]; then
    START_CLIENT_STATUS=pass
  fi
  publish_detached_start_result \
    "${START_STDOUT_PENDING}" "${START_STDERR_PENDING}" \
    "${START_CLIENT_STATUS}" "${START_CLIENT_EXIT}"
  rm -f -- "${START_STDOUT_PENDING}" "${START_STDERR_PENDING}"
  START_INSPECT_CAPTURE="${DOCKER_EVIDENCE}/.container-after-start.pending"
  if [[ -e "${START_INSPECT_CAPTURE}" ]]; then
    echo "container-after-start capture path already exists" >&2
    exit 1
  fi
  if docker inspect --type container "${CONTAINER_ID}" >"${START_INSPECT_CAPTURE}"; then
    OBSERVED_START_STATUS="$(validate_intent_owned_container \
      "${START_INSPECT_CAPTURE}" "created,running,exited")"
    if [[ "${OBSERVED_START_STATUS}" == "running" ||
          "${OBSERVED_START_STATUS}" == "exited" ]]; then
      atomic_install_regular_file \
        "${START_INSPECT_CAPTURE}" "${DOCKER_EVIDENCE}/container-started.json"
      write_detached_launch_receipt
    else
      atomic_install_regular_file "${START_INSPECT_CAPTURE}" \
        "${DOCKER_EVIDENCE}/container-after-start-error.json"
    fi
  else
    printf 'Docker start returned, but immediate container-state observation was unavailable.\n' \
      >"${DOCKER_EVIDENCE}/container-after-start-error.txt"
    OBSERVED_START_STATUS=unavailable
  fi
  rm -f -- "${START_INSPECT_CAPTURE}"
  DETACHED_CONTAINER_ID="${CONTAINER_ID}"
  CONTAINER_ID=""
  flock -u "${DETACHED_LAUNCH_LOCK_FD}"
  exec {DETACHED_LAUNCH_LOCK_FD}>&-
  echo "detached bootstrap host container: ${DETACHED_CONTAINER_ID}"
  echo "finalize with: ${BASH_SOURCE[0]} --finalize ${OUTPUT}"
  if [[ "${START_CLIENT_STATUS}" != pass ||
        "${OBSERVED_START_STATUS}" == created ||
        "${OBSERVED_START_STATUS}" == unavailable ]]; then
    echo "Docker start client/daemon outcome requires detached finalization" >&2
    if [[ ${START_CLIENT_EXIT} -ne 0 ]]; then
      exit "${START_CLIENT_EXIT}"
    fi
    exit 1
  fi
  exit 0
fi

set +e
docker start -a "${CONTAINER_ID}" 2>&1 |
  tee "${OUTPUT}/bootstrap-host-freeze.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-completed.json"
remove_exact_container

RESULT_ARCHIVE="${OUTPUT}/bootstrap-host-result.tar.gz"
set +e
promote_incoming_result
PROMOTION_EXIT="$?"
set -e
if [[ ${PROMOTION_EXIT} -eq 3 ]]; then
  write_result_verification_failure "result archive or sidecar is missing"
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" missing
  write_outer_bundle_checksum_closure allow-missing-result
  echo "bootstrap host freeze did not publish its result archive and sidecar" >&2
  exit 1
fi
if [[ ${PROMOTION_EXIT} -eq 2 ]]; then
  write_result_verification_failure "incoming result entries or sidecar were rejected"
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" rejected
  write_outer_bundle_checksum_closure allow-missing-result
  echo "bootstrap host freeze result staging was rejected" >&2
  exit 1
fi
if [[ ${PROMOTION_EXIT} -ne 0 ]]; then
  echo "bootstrap host freeze result promotion could not be recovered" >&2
  exit 1
fi
if ! (
  cd "${OUTPUT}"
  sha256sum -c "$(basename -- "${RESULT_ARCHIVE}.sha256")"
) >"${DOCKER_EVIDENCE}/result-sidecar-check.txt" 2>&1; then
  write_result_verification_failure "result archive sidecar verification failed"
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" rejected
  write_outer_bundle_checksum_closure
  echo "bootstrap host freeze result archive sidecar is invalid" >&2
  exit 1
fi
VERIFICATION_TEMPORARY="$(mktemp)"
set +e
verify_result_archive \
  "${RESULT_ARCHIVE}" bootstrap-host-freeze "${PIPELINE_EXIT}" \
  "${OUTPUT}/verified-bootstrap-host-result" \
  >"${VERIFICATION_TEMPORARY}" \
  2>"${DOCKER_EVIDENCE}/result-archive-verification.txt"
VERIFICATION_EXIT="$?"
set -e
if [[ ${VERIFICATION_EXIT} -ne 0 ]]; then
  rm -f -- "${VERIFICATION_TEMPORARY}"
  write_result_verification_failure "result archive structure verification failed"
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" rejected
  write_outer_bundle_checksum_closure
  echo "bootstrap host freeze result archive is invalid" >&2
  exit 1
fi
mv -- "${VERIFICATION_TEMPORARY}" \
  "${OUTPUT}/bootstrap-host-result-verification.json"
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" verified
  write_outer_bundle_checksum_closure
  echo "bootstrap host freeze failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi

QUALIFICATION="${OUTPUT}/verified-bootstrap-host-result/results/qualification"
if ! verify_bootstrap_qualification \
    >"${DOCKER_EVIDENCE}/qualification-verification.txt" 2>&1; then
  write_lifecycle_receipt failed "${PIPELINE_EXIT}" qualification-rejected
  write_outer_bundle_checksum_closure
  echo "bootstrap host freeze qualification verification failed" >&2
  exit 1
fi
write_lifecycle_receipt pass "${PIPELINE_EXIT}" verified
write_outer_bundle_checksum_closure
echo "bootstrap host evidence: ${QUALIFICATION}"
