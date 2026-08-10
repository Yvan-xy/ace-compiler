#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

TRANSPORT_HELPERS_LOADED=false
load_transport_helpers() {
  if [[ "${TRANSPORT_HELPERS_LOADED}" == false ]]; then
    # The caller verifies these bytes against the selected source commit before
    # allowing the sourced implementation to execute.
    source "${SCRIPT_DIR}/transport_helpers.sh"
    TRANSPORT_HELPERS_LOADED=true
  fi
}

usage() {
  cat >&2 <<EOF
usage: $0 [--detach] [--qualification-mode full|generated-bootstrap-correctness] --ace-commit COMMIT --phantom-commit COMMIT [--ordinary-run-root DIR --retained-run-root DIR] --bootstrap-run-root DIR OUTPUT_DIRECTORY
       $0 --finalize OUTPUT_DIRECTORY
EOF
  exit 2
}

RUN_MODE=synchronous
LAUNCH_OPTION_SEEN=false
ACE_COMMIT=""
PHANTOM_COMMIT=""
QUALIFICATION_MODE=full
ORDINARY_RUN_ROOT=""
RETAINED_RUN_ROOT=""
BOOTSTRAP_RUN_ROOT=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach)
      [[ "${RUN_MODE}" == synchronous ]] || usage
      RUN_MODE=detached-launch
      shift
      ;;
    --finalize)
      [[ "${RUN_MODE}" == synchronous ]] || usage
      RUN_MODE=detached-finalize
      shift
      ;;
    --qualification-mode) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; QUALIFICATION_MODE="$2"; shift 2 ;;
    --ace-commit) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; ACE_COMMIT="$2"; shift 2 ;;
    --phantom-commit) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; PHANTOM_COMMIT="$2"; shift 2 ;;
    --ordinary-run-root) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; ORDINARY_RUN_ROOT="$2"; shift 2 ;;
    --retained-run-root) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; RETAINED_RUN_ROOT="$2"; shift 2 ;;
    --bootstrap-run-root) [[ $# -ge 2 ]] || usage; LAUNCH_OPTION_SEEN=true; BOOTSTRAP_RUN_ROOT="$2"; shift 2 ;;
    --)
      shift
      [[ $# -eq 1 && -z "${OUTPUT_ARGUMENT}" ]] || usage
      OUTPUT_ARGUMENT="$1"
      shift
      ;;
    -*) usage ;;
    *) [[ -z "${OUTPUT_ARGUMENT}" ]] || usage; OUTPUT_ARGUMENT="$1"; shift ;;
  esac
done

if [[ "${RUN_MODE}" == detached-finalize ]]; then
  [[ -z "${ACE_COMMIT}" && -z "${PHANTOM_COMMIT}" &&
     -z "${ORDINARY_RUN_ROOT}" && -z "${RETAINED_RUN_ROOT}" &&
     -z "${BOOTSTRAP_RUN_ROOT}" && -n "${OUTPUT_ARGUMENT}" &&
     "${QUALIFICATION_MODE}" == full && "${LAUNCH_OPTION_SEEN}" == false ]] || usage
else
  [[ "${QUALIFICATION_MODE}" == full ||
     "${QUALIFICATION_MODE}" == generated-bootstrap-correctness ]] || usage
  [[ -n "${ACE_COMMIT}" && -n "${PHANTOM_COMMIT}" &&
     -n "${BOOTSTRAP_RUN_ROOT}" && -n "${OUTPUT_ARGUMENT}" ]] || usage
  if [[ "${QUALIFICATION_MODE}" == full ]]; then
    [[ -n "${ORDINARY_RUN_ROOT}" && -n "${RETAINED_RUN_ROOT}" ]] || usage
  fi
  if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
        ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "source commits must be full lowercase 40-character object IDs" >&2
    exit 1
  fi
fi

OUTPUT=""
PAYLOAD=""
RESULT_STAGING=""
DOCKER_EVIDENCE=""
INTENT=""
INTENT_SHA256=""
CONTAINER_ID=""
CREATED_CONTAINER_ID=""
CONTAINER_NAME=""
TASK_LABEL=""
RUN_LABEL=""
RUN_NONCE=""
BASE_IMAGE=""
BASE_CONFIG=""
BASE_IMAGE_INSPECT_SHA256=""
PIPELINE_EXIT=1
CLEANUP_AUTHORIZED=false
DETACHED_HANDOFF=false
START_ATTEMPT_PRESENT=false

clear_regular_work_file() {
  local path="$1"
  if [[ -e "${path}" || -L "${path}" ]]; then
    [[ -f "${path}" && ! -L "${path}" ]] || {
      echo "refusing an unsafe lifecycle work file: ${path}" >&2
      return 1
    }
    rm -f -- "${path}"
  fi
}

atomic_install_regular() {
  local work="$1"
  local destination="$2"
  python3 - "${work}" "${destination}" <<'PY'
import os, pathlib, stat, sys

work = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
if work.parent.resolve() != destination.parent.resolve():
    raise SystemExit("lifecycle receipt work file is not on the destination filesystem")
metadata = work.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("lifecycle receipt work file is not regular")
with work.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(work, destination)
directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

load_intent() {
  local -a fields=()
  mapfile -t fields < <(python3 - "${INTENT}" <<'PY'
import hashlib, json, pathlib, re, stat, sys

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("duplicate detached-intent key")
        value[key] = item
    return value

path = pathlib.Path(sys.argv[1])
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit("detached local intent is not a regular mode-0600 file")
value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
keys = (
    "schema_version", "status", "requested_execution_mode",
    "qualification_mode", "output_directory", "output_device",
    "output_inode", "result_staging_directory", "result_staging_device",
    "result_staging_inode", "ace_commit", "phantom_commit", "base_image",
    "base_config_digest", "base_image_inspect_sha256", "container_name",
    "task_label", "run_label", "run_nonce", "docker_context",
    "docker_daemon_id", "payload_json_sha256", "payload_sums_sha256",
    "entrypoint_sha256", "helper_sha256", "build_jobs",
)
if not isinstance(value, dict) or set(value) != set(keys):
    raise SystemExit("detached local intent has an invalid shape")
if value["schema_version"] != "ace.phantom.local-reproduction-intent/1.0.0" or value["status"] != "prepared":
    raise SystemExit("detached local intent schema mismatch")
if value["requested_execution_mode"] not in {"synchronous", "detached-launch"} or value["qualification_mode"] not in {"full", "generated-bootstrap-correctness"}:
    raise SystemExit("detached local intent mode mismatch")
for key in ("ace_commit", "phantom_commit"):
    if re.fullmatch(r"[0-9a-f]{40}", str(value[key])) is None:
        raise SystemExit("detached local intent commit is invalid")
for key in (
    "base_image_inspect_sha256", "payload_json_sha256", "payload_sums_sha256",
    "entrypoint_sha256", "helper_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", str(value[key])) is None:
        raise SystemExit("detached local intent checksum is invalid")
if re.fullmatch(r"[0-9a-f]{32}", str(value["run_nonce"])) is None:
    raise SystemExit("detached local intent nonce is invalid")
for key in keys[2:]:
    print(value[key])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
  )
  [[ ${#fields[@]} -eq 25 ]] || {
    echo "could not load detached local intent" >&2
    return 1
  }
  REQUESTED_EXECUTION_MODE="${fields[0]}"
  QUALIFICATION_MODE="${fields[1]}"
  local recorded_output="${fields[2]}"
  local recorded_device="${fields[3]}"
  local recorded_inode="${fields[4]}"
  local recorded_staging="${fields[5]}"
  local recorded_staging_device="${fields[6]}"
  local recorded_staging_inode="${fields[7]}"
  ACE_COMMIT="${fields[8]}"
  PHANTOM_COMMIT="${fields[9]}"
  BASE_IMAGE="${fields[10]}"
  BASE_CONFIG="${fields[11]}"
  BASE_IMAGE_INSPECT_SHA256="${fields[12]}"
  CONTAINER_NAME="${fields[13]}"
  TASK_LABEL="${fields[14]}"
  RUN_LABEL="${fields[15]}"
  RUN_NONCE="${fields[16]}"
  DOCKER_CONTEXT_ID="${fields[17]}"
  DOCKER_DAEMON_ID="${fields[18]}"
  PAYLOAD_JSON_SHA256="${fields[19]}"
  PAYLOAD_SUMS_SHA256="${fields[20]}"
  ENTRYPOINT_SHA256="${fields[21]}"
  HELPER_SHA256="${fields[22]}"
  BUILD_JOBS="${fields[23]}"
  INTENT_SHA256="${fields[24]}"
  if [[ "${recorded_output}" != "${OUTPUT}" ||
        "${recorded_device}" != "$(stat -Lc '%d' -- "${OUTPUT}")" ||
        "${recorded_inode}" != "$(stat -Lc '%i' -- "${OUTPUT}")" ||
        ! -d "${RESULT_STAGING}" || -L "${RESULT_STAGING}" ||
        "${recorded_staging}" != "$(realpath -e -- "${RESULT_STAGING}")" ||
        "${recorded_staging_device}" != "$(stat -Lc '%d' -- "${RESULT_STAGING}")" ||
        "${recorded_staging_inode}" != "$(stat -Lc '%i' -- "${RESULT_STAGING}")" ]]; then
    echo "detached local intent no longer names the same output directory" >&2
    return 1
  fi
}

validate_container() {
  local inspect_file="$1"
  local expected_id="$2"
  local expected_state="${3:-any}"
  local expected_exit="${4:-}"
  python3 - "${INTENT}" "${inspect_file}" "${INTENT_SHA256}" \
    "${expected_id}" "${expected_state}" "${expected_exit}" \
    "${PAYLOAD}" "${RESULT_STAGING}" \
    "${DOCKER_EVIDENCE}/base-image.json" \
    "${DOCKER_EVIDENCE}/disposable-container-created.json" <<'PY'
import json, pathlib, re, sys
intent = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
record = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
(
    intent_sha, expected_id, expected_state, expected_exit, payload, staging,
    base_image_path, baseline_path,
) = sys.argv[3:]
if not isinstance(record, list) or len(record) != 1:
    raise SystemExit("Docker did not return one container record")
item = record[0]
config, host, state = item.get("Config", {}), item.get("HostConfig", {}), item.get("State", {})
container_id = item.get("Id", "")
labels = config.get("Labels") or {}
command = [
    "bash", "/retained-qualification/input/run_build_and_health.sh",
    "--mode", "local", "--input-dir", "/retained-qualification/input",
    "--work-dir", "/retained-qualification/work", "--result-archive",
    "/retained-qualification/output/local-result.tar.gz",
]
owned_environment = [
    "NVIDIA_VISIBLE_DEVICES=void", "NVIDIA_DRIVER_CAPABILITIES=none",
    f"ACE_RUNPOD_BASE_IMAGE={intent['base_image']}",
    f"ACE_RUNPOD_BASE_CONFIG_DIGEST={intent['base_config_digest']}",
    f"ACE_PHANTOM_BUILD_JOBS={intent['build_jobs']}",
]
base_records = json.loads(pathlib.Path(base_image_path).read_text(encoding="utf-8"))
if not isinstance(base_records, list) or len(base_records) != 1:
    raise SystemExit("locked base image inspect evidence is invalid")
base_config = base_records[0].get("Config") or {}
expected_environment = list(owned_environment)
environment_keys = {entry.split("=", 1)[0] for entry in expected_environment}
for entry in base_config.get("Env") or []:
    key = entry.split("=", 1)[0]
    if key not in environment_keys:
        expected_environment.append(entry)
        environment_keys.add(key)
expected_labels = dict(base_config.get("Labels") or {})
expected_labels.update({
    "ace.phantom.task": intent["task_label"],
    "ace.phantom.run": intent["run_label"],
    "ace.phantom.run-nonce": intent["run_nonce"],
    "ace.phantom.intent-sha256": intent_sha,
})
mounts = {(m.get("Source"), m.get("Destination"), bool(m.get("RW"))) for m in item.get("Mounts", [])}
if (
    re.fullmatch(r"[0-9a-f]{64}", container_id) is None
    or (expected_id and container_id != expected_id)
    or item.get("Name") != "/" + intent["container_name"]
    or labels != expected_labels
    or item.get("Image") != intent["base_config_digest"]
    or config.get("Image") != intent["base_image"]
    or config.get("Cmd") != command
    or config.get("Env") != expected_environment
    or host.get("Runtime") != "runc" or host.get("Privileged") is not False
    or (host.get("Devices") or []) != [] or (host.get("DeviceRequests") or []) != []
    or (host.get("RestartPolicy") or {}).get("Name") not in {"", "no"}
    or len(item.get("Mounts", [])) != 2
    or mounts != {(payload, "/retained-qualification/input", False), (staging, "/retained-qualification/output", True)}
):
    raise SystemExit("Docker container identity differs from detached local intent")
baseline = pathlib.Path(baseline_path)
if baseline.is_file():
    baseline_record = json.loads(baseline.read_text(encoding="utf-8"))
    if not isinstance(baseline_record, list) or len(baseline_record) != 1:
        raise SystemExit("created-container host comparison evidence is invalid")
    original = baseline_record[0]

    def exact_json_equal(left, right):
        options = {
            "allow_nan": False,
            "ensure_ascii": False,
            "separators": (",", ":"),
            "sort_keys": True,
        }
        return json.dumps(left, **options) == json.dumps(right, **options)

    def normalized_host_configs(current_record, original_record):
        current = current_record.get("HostConfig")
        original = original_record.get("HostConfig")
        if not isinstance(current, dict) or not isinstance(original, dict):
            return current, original
        current = dict(current)
        original = dict(original)
        key = "OomKillDisable"
        if key in current and key in original:
            current_value = current[key]
            original_value = original[key]
            current_is_normalized = current_value is False or current_value is None
            original_is_normalized = original_value is False or original_value is None
            if current_is_normalized and original_is_normalized:
                current[key] = False
                original[key] = False
        return current, original

    current_host_config, original_host_config = normalized_host_configs(
        item, original
    )

    if (
        not exact_json_equal(item.get("Config"), original.get("Config"))
        or not exact_json_equal(current_host_config, original_host_config)
        or not exact_json_equal(item.get("Mounts"), original.get("Mounts"))
    ):
        raise SystemExit("Docker immutable configuration changed after creation")
if expected_state == "created" and (state.get("Status"), state.get("Running")) != ("created", False):
    raise SystemExit("Docker container was not captured before start")
if expected_state == "started" and (not state.get("StartedAt") or state.get("StartedAt", "").startswith("0001-") or state.get("Status") not in {"running", "exited", "dead"}):
    raise SystemExit("Docker container has no valid start record")
if expected_state == "completed" and (state.get("Running") is not False or state.get("Status") not in {"exited", "dead"} or state.get("ExitCode") != int(expected_exit)):
    raise SystemExit("Docker completion differs from docker wait")
print(container_id)
PY
}

container_status() {
  python3 - "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value[0]["State"]["Status"])
PY
}

capture_container_receipt() {
  local destination="$1"
  local expected_id="$2"
  local expected_state="${3:-any}"
  local expected_exit="${4:-}"
  local allow_replace="${5:-false}"
  local work="${destination}.work"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ "${allow_replace}" == true ]] || {
      [[ -f "${destination}" && ! -L "${destination}" ]] || {
        echo "container receipt is not a regular file: ${destination}" >&2
        return 1
      }
      validate_container "${destination}" "${expected_id}" \
        "${expected_state}" "${expected_exit}" >/dev/null
      return
    }
  fi
  clear_regular_work_file "${work}"
  if ! docker inspect --type container "${expected_id}" >"${work}"; then
    echo "could not inspect the exact lifecycle container" >&2
    return 1
  fi
  validate_container "${work}" "${expected_id}" \
    "${expected_state}" "${expected_exit}" >/dev/null
  atomic_install_regular "${work}" "${destination}"
  validate_container "${destination}" "${expected_id}" \
    "${expected_state}" "${expected_exit}" >/dev/null
}

validate_container_id_receipt() {
  local path="$1"
  local expected_id="$2"
  python3 - "${path}" "${expected_id}" <<'PY'
import pathlib, re, stat, sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("container start identity receipt is not regular")
value = path.read_text(encoding="utf-8")
if re.fullmatch(r"[0-9a-f]{64}\n", value) is None or value != expected + "\n":
    raise SystemExit("container start identity receipt is not one exact container ID")
PY
}

ensure_container_id_receipt() {
  local destination="${DOCKER_EVIDENCE}/start.txt"
  local work="${destination}.work"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] || return 1
    validate_container_id_receipt "${destination}" "${CREATED_CONTAINER_ID}"
    clear_regular_work_file "${work}"
    return
  fi
  if [[ -f "${work}" && ! -L "${work}" ]] &&
     validate_container_id_receipt "${work}" "${CREATED_CONTAINER_ID}"; then
    :
  else
    clear_regular_work_file "${work}"
    printf '%s\n' "${CREATED_CONTAINER_ID}" >"${work}"
    validate_container_id_receipt "${work}" "${CREATED_CONTAINER_ID}"
  fi
  atomic_install_regular "${work}" "${destination}"
  validate_container_id_receipt "${destination}" "${CREATED_CONTAINER_ID}"
}

validate_start_attempt() {
  local path="$1"
  python3 - "${path}" "${INTENT_SHA256}" "${CREATED_CONTAINER_ID}" \
    "${CONTAINER_NAME}" "${RUN_LABEL}" <<'PY'
import json, pathlib, stat, sys

path = pathlib.Path(sys.argv[1])
intent_sha, container_id, container_name, run_label = sys.argv[2:]
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit("container start-attempt receipt is not a regular mode-0600 file")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("container start-attempt receipt has a duplicate key")
        value[key] = item
    return value

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
expected = {
    "schema_version": "ace.phantom.local-container-start-attempt/1.0.0",
    "status": "prepared",
    "intent_sha256": intent_sha,
    "container_id": container_id,
    "container_name": container_name,
    "run_label": run_label,
    "command": ["docker", "start", container_id],
}
if value != expected:
    raise SystemExit("container start-attempt receipt differs from the exact lifecycle identity")
PY
}

publish_start_attempt() {
  local destination="${DOCKER_EVIDENCE}/container-start-attempt.json"
  local work="${destination}.work"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    validate_start_attempt "${destination}"
    START_ATTEMPT_PRESENT=true
    return
  fi
  clear_regular_work_file "${work}"
  python3 - "${work}" "${INTENT_SHA256}" "${CREATED_CONTAINER_ID}" \
    "${CONTAINER_NAME}" "${RUN_LABEL}" <<'PY'
import json, os, pathlib, sys

path = pathlib.Path(sys.argv[1])
intent_sha, container_id, container_name, run_label = sys.argv[2:]
value = {
    "schema_version": "ace.phantom.local-container-start-attempt/1.0.0",
    "status": "prepared",
    "intent_sha256": intent_sha,
    "container_id": container_id,
    "container_name": container_name,
    "run_label": run_label,
    "command": ["docker", "start", container_id],
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(path, 0o600)
PY
  validate_start_attempt "${work}"
  atomic_install_regular "${work}" "${destination}"
  validate_start_attempt "${destination}"
  START_ATTEMPT_PRESENT=true
}

load_start_attempt_state() {
  local destination="${DOCKER_EVIDENCE}/container-start-attempt.json"
  local work="${destination}.work"
  START_ATTEMPT_PRESENT=false
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    validate_start_attempt "${destination}" || return 1
    START_ATTEMPT_PRESENT=true
  else
    clear_regular_work_file "${work}" || return 1
  fi
}

validate_start_observation() {
  local path="$1"
  python3 - "${path}" "${INTENT_SHA256}" "${CREATED_CONTAINER_ID}" <<'PY'
import json, pathlib, stat, sys

path = pathlib.Path(sys.argv[1])
intent_sha, container_id = sys.argv[2:]
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("container start-observation receipt is not regular")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("container start-observation receipt has a duplicate key")
        value[key] = item
    return value

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
keys = {
    "schema_version", "status", "intent_sha256", "container_id",
    "client_exit_code", "client_output_exact_container_id",
    "observed_container_state",
}
if not isinstance(value, dict) or set(value) != keys:
    raise SystemExit("container start-observation receipt has an invalid shape")
if (
    value["schema_version"] != "ace.phantom.local-container-start-observation/1.0.0"
    or value["status"] not in {"effect-observed", "no-effect-observed"}
    or value["intent_sha256"] != intent_sha
    or value["container_id"] != container_id
    or (value["client_exit_code"] is not None and (
        type(value["client_exit_code"]) is not int
        or not 0 <= value["client_exit_code"] <= 255
    ))
    or type(value["client_output_exact_container_id"]) is not bool
    or value["observed_container_state"] not in {
        "created", "running", "exited", "dead",
    }
):
    raise SystemExit("container start-observation receipt is invalid")
if (value["status"] == "no-effect-observed") != (
    value["observed_container_state"] == "created"
):
    raise SystemExit("container start-observation status contradicts the observed state")
PY
}

publish_start_observation() {
  local client_exit="$1"
  local output_exact="$2"
  local observed_state="$3"
  local destination="${DOCKER_EVIDENCE}/container-start-observation.json"
  local work="${destination}.work"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    validate_start_observation "${destination}"
    return
  fi
  clear_regular_work_file "${work}"
  python3 - "${work}" "${INTENT_SHA256}" "${CREATED_CONTAINER_ID}" \
    "${client_exit}" "${output_exact}" "${observed_state}" <<'PY'
import json, pathlib, sys

path = pathlib.Path(sys.argv[1])
intent_sha, container_id, client_exit, output_exact, observed_state = sys.argv[2:]
value = {
    "schema_version": "ace.phantom.local-container-start-observation/1.0.0",
    "status": "no-effect-observed" if observed_state == "created" else "effect-observed",
    "intent_sha256": intent_sha,
    "container_id": container_id,
    "client_exit_code": None if client_exit == "unknown" else int(client_exit),
    "client_output_exact_container_id": output_exact == "true",
    "observed_container_state": observed_state,
}
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  validate_start_observation "${work}"
  atomic_install_regular "${work}" "${destination}"
  validate_start_observation "${destination}"
}

remove_exact_container() {
  local exact_id="${CONTAINER_ID}"
  [[ -n "${exact_id}" ]] || return 0
  [[ "${CLEANUP_AUTHORIZED}" == true ]] || {
    echo "cleanup refused an unverified container" >&2
    return 1
  }
  local inspect_file
  inspect_file="$(mktemp)"
  if ! docker inspect --type container "${exact_id}" >"${inspect_file}" ||
     ! validate_container "${inspect_file}" "${exact_id}" any >/dev/null; then
    rm -f -- "${inspect_file}"
    echo "exact cleanup refused a container with changed identity" >&2
    return 1
  fi
  rm -f -- "${inspect_file}"
  local cleanup_intent="${DOCKER_EVIDENCE}/cleanup-intent.tsv"
  local cleanup_intent_temp="${cleanup_intent}.tmp"
  printf '%s\t%s\t%s\n' "${exact_id}" "${CONTAINER_NAME}" "${RUN_LABEL}" \
    >"${cleanup_intent_temp}"
  mv -- "${cleanup_intent_temp}" "${cleanup_intent}"
  docker rm -f "${exact_id}" >"${DOCKER_EVIDENCE}/cleanup.txt"
  if docker inspect --type container "${exact_id}" >/dev/null 2>&1; then
    echo "exact cleanup verification failed for ${exact_id}" >&2
    return 1
  fi
  local name_inventory run_inventory
  name_inventory="$(docker ps -aq --no-trunc --filter "name=^/${CONTAINER_NAME}$")"
  run_inventory="$(docker ps -aq --no-trunc --filter "label=ace.phantom.run=${RUN_LABEL}")"
  [[ -z "${name_inventory}" && -z "${run_inventory}" ]] || {
    echo "exact cleanup left a matching name or run-label inventory" >&2
    return 1
  }
  local verification="${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  local verification_temp="${verification}.tmp"
  {
    printf '%s\tremoved-and-absent\n' "${exact_id}"
    printf '%s\tname-inventory-empty\n' "${CONTAINER_NAME}"
    printf '%s\trun-label-inventory-empty\n' "${RUN_LABEL}"
  } >"${verification_temp}"
  mv -- "${verification_temp}" "${verification}"
  CONTAINER_ID=""
}

capture_after_inventory() {
  docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-after.txt"
  docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-after.txt"
  docker system df >"${DOCKER_EVIDENCE}/disk-after.txt"
}

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ -n "${CONTAINER_ID}" && "${DETACHED_HANDOFF}" != true ]]; then
    if [[ "${CLEANUP_AUTHORIZED}" == true ]]; then
      set +e
      remove_exact_container
      local cleanup_exit="$?"
      capture_after_inventory
      set -e
      [[ ${cleanup_exit} -eq 0 ]] || incoming=1
    else
      echo "leaving an identity-mismatched container untouched: ${CONTAINER_ID}" >&2
      incoming=1
    fi
  fi
  exit "${incoming}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

write_result_failure() {
  local reason="$1"
  local destination="${OUTPUT}/local-result-verification.json"
  local work="${destination}.work"
  clear_regular_work_file "${work}"
  python3 - "${work}" "${PIPELINE_EXIT}" "${reason}" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "ace.phantom.local-result-verification/1.0.0",
    "status": "failed", "mode": "local",
    "pipeline_exit_code": int(sys.argv[2]), "reason": sys.argv[3],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  atomic_install_regular "${work}" "${destination}"
}

write_lifecycle() {
  local status="$1"
  local archive_status="$2"
  local destination="${OUTPUT}/lifecycle.json"
  local work="${destination}.work"
  clear_regular_work_file "${work}"
  python3 - "${work}" "${status}" "${archive_status}" \
    "${REQUESTED_EXECUTION_MODE}" "${QUALIFICATION_MODE}" "${ACE_COMMIT}" \
    "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
    "${CREATED_CONTAINER_ID}" "${TASK_LABEL}" "${RUN_LABEL}" \
    "${RUN_NONCE}" "${PIPELINE_EXIT}" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "ace.phantom.local-reproduction-lifecycle/1.0.0",
    "status": sys.argv[2], "result_archive_status": sys.argv[3],
    "requested_execution_mode": sys.argv[4], "qualification_mode": sys.argv[5],
    "ace_commit": sys.argv[6], "phantom_commit": sys.argv[7],
    "base_image": sys.argv[8], "base_config_digest": sys.argv[9],
    "container_id": sys.argv[10], "task_label": sys.argv[11],
    "run_label": sys.argv[12], "run_nonce": sys.argv[13],
    "pipeline_exit_code": int(sys.argv[14]),
    "container_cleanup": "removed-and-absent",
    "container_name_inventory": "empty", "run_label_inventory": "empty",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  validate_lifecycle_receipt "${work}" "${status}" "${archive_status}"
  atomic_install_regular "${work}" "${destination}"
  validate_lifecycle_receipt "${destination}" "${status}" "${archive_status}"
}

validate_lifecycle_receipt() {
  local path="$1"
  local expected_status="$2"
  local expected_archive_status="$3"
  python3 - "${path}" "${expected_status}" "${expected_archive_status}" \
    "${REQUESTED_EXECUTION_MODE}" "${QUALIFICATION_MODE}" "${ACE_COMMIT}" \
    "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
    "${CREATED_CONTAINER_ID}" "${TASK_LABEL}" "${RUN_LABEL}" \
    "${RUN_NONCE}" "${PIPELINE_EXIT}" <<'PY'
import json, pathlib, stat, sys

path = pathlib.Path(sys.argv[1])
(
    status, archive_status, execution_mode, qualification_mode, ace_commit,
    phantom_commit, base_image, base_config, container_id, task_label,
    run_label, run_nonce, pipeline_exit,
) = sys.argv[2:]
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("local lifecycle receipt is not regular")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("local lifecycle receipt has a duplicate key")
        value[key] = item
    return value

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
expected = {
    "schema_version": "ace.phantom.local-reproduction-lifecycle/1.0.0",
    "status": status,
    "result_archive_status": archive_status,
    "requested_execution_mode": execution_mode,
    "qualification_mode": qualification_mode,
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "base_image": base_image,
    "base_config_digest": base_config,
    "container_id": container_id,
    "task_label": task_label,
    "run_label": run_label,
    "run_nonce": run_nonce,
    "pipeline_exit_code": int(pipeline_exit),
    "container_cleanup": "removed-and-absent",
    "container_name_inventory": "empty",
    "run_label_inventory": "empty",
}
if value != expected:
    raise SystemExit("local lifecycle receipt differs from the exact observed lifecycle")
PY
}

reject_lifecycle_work_files() {
  local relative
  for relative in \
    docker/disposable-container-created.json.work \
    docker/container-start-attempt.json.work \
    docker/start.txt.work \
    docker/container-started.json.work \
    docker/container-start-observation.json.work \
    docker/container-current.json.work \
    docker/disposable-container.json.work \
    docker/disposable-container.json.tmp \
    local-container.log.work \
    local-container.log.tmp \
    local-result-verification.json.work \
    lifecycle.json.work; do
    [[ ! -e "${OUTPUT}/${relative}" && ! -L "${OUTPUT}/${relative}" ]] || {
      echo "local lifecycle has an uncommitted work receipt: ${relative}" >&2
      return 1
    }
  done
}

checksum_close() {
  validate_closure_receipts
  local unexpected
  unexpected="$(find "${OUTPUT}" -mindepth 1 ! -type d ! -type f -print -quit)"
  [[ -z "${unexpected}" ]] || {
    echo "local evidence contains a nonregular entry: ${unexpected}" >&2
    return 1
  }
  [[ ! -e "${OUTPUT}/BUNDLE_SHA256SUMS" ]] || {
    echo "local bundle checksum manifest already exists" >&2
    return 1
  }
  (
    cd "${OUTPUT}"
    local before after manifest
    before="$(mktemp)"; after="$(mktemp)"
    manifest="$(mktemp "$(dirname -- "${OUTPUT}")/.local-bundle-manifest.XXXXXX")"
    trap 'rm -f -- "${before}" "${after}" "${manifest}"' EXIT
    find . -type f ! -path ./BUNDLE_SHA256SUMS -print0 | LC_ALL=C sort -z >"${before}"
    xargs -0 -r sha256sum <"${before}" >"${manifest}"
    find . -type f ! -path ./BUNDLE_SHA256SUMS -print0 | LC_ALL=C sort -z >"${after}"
    cmp -- "${before}" "${after}"
    mv -- "${manifest}" BUNDLE_SHA256SUMS
    python3 - <<'PY'
import pathlib, re

root = pathlib.Path(".")
manifest = root / "BUNDLE_SHA256SUMS"
entries = list(root.rglob("*"))
if any(
    item.is_symlink() or not (item.is_dir() or item.is_file())
    for item in entries
):
    raise SystemExit("local evidence changed to include a nonregular entry")
listed = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"[0-9a-f]{64}  (.+)", line)
    if match is None:
        raise SystemExit("local bundle manifest is malformed")
    listed.append(match.group(1).removeprefix("./"))
actual = sorted(
    str(item.relative_to(root)) for item in entries
    if item.is_file() and item != manifest
)
if sorted(listed) != actual or len(listed) != len(set(listed)):
    raise SystemExit("final local bundle inventory differs from its manifest")
PY
    sha256sum -c BUNDLE_SHA256SUMS
  )
}

verify_closed_bundle() {
  [[ -f "${OUTPUT}/BUNDLE_SHA256SUMS" ]] || return 1
  local unexpected
  unexpected="$(find "${OUTPUT}" -mindepth 1 ! -type d ! -type f -print -quit)"
  [[ -z "${unexpected}" ]] || return 1
  (
    cd "${OUTPUT}"
    sha256sum -c BUNDLE_SHA256SUMS >/dev/null
    python3 - <<'PY'
import pathlib, re
root = pathlib.Path(".")
manifest = root / "BUNDLE_SHA256SUMS"
listed = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"[0-9a-f]{64}  (.+)", line)
    if match is None:
        raise SystemExit("local bundle manifest is malformed")
    listed.append(match.group(1).removeprefix("./"))
actual = sorted(
    str(path).removeprefix("./") for path in root.rglob("*")
    if path.is_file() and path != manifest
)
if sorted(listed) != actual or len(listed) != len(set(listed)):
    raise SystemExit("local bundle inventory differs from its manifest")
PY
  )
}

verify_or_move_result() {
  local archive_name=local-result.tar.gz
  local sidecar_name=local-result.tar.gz.sha256
  python3 - "${OUTPUT}" "${RESULT_STAGING}" <<'PY'
import hashlib, pathlib, re, sys
output, staging = map(pathlib.Path, sys.argv[1:])
names = ("local-result.tar.gz", "local-result.tar.gz.sha256")
unexpected = [item for item in staging.iterdir() if item.name not in names]
if unexpected or any(item.is_symlink() or not item.is_file() for item in unexpected):
    raise SystemExit("incoming local result inventory is invalid")
locations = {}
for name in names:
    candidates = [root / name for root in (output, staging) if (root / name).exists()]
    if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file():
        raise SystemExit("local result promotion is incomplete or ambiguous")
    locations[name] = candidates[0]
line = locations[names[1]].read_text(encoding="utf-8")
match = re.fullmatch(r"([0-9a-f]{64})  local-result\.tar\.gz\n", line)
if match is None or match.group(1) != hashlib.sha256(locations[names[0]].read_bytes()).hexdigest():
    raise SystemExit("local result checksum is invalid")
PY
  [[ -e "${OUTPUT}/${archive_name}" ]] ||
    mv -- "${RESULT_STAGING}/${archive_name}" "${OUTPUT}/${archive_name}"
  [[ -e "${OUTPUT}/${sidecar_name}" ]] ||
    mv -- "${RESULT_STAGING}/${sidecar_name}" "${OUTPUT}/${sidecar_name}"
  [[ -z "$(find "${RESULT_STAGING}" -mindepth 1 -print -quit)" ]]
  (
    cd "${OUTPUT}"
    [[ "$(wc -l <"${sidecar_name}")" -eq 1 ]]
    grep -Eq '^[0-9a-f]{64}  local-result\.tar\.gz$' "${sidecar_name}"
    sha256sum -c "${sidecar_name}"
  )
}

verify_lifecycle_identity() {
  [[ "$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/run_local_reproduction.sh" | awk '{print $1}')" == "${ENTRYPOINT_SHA256}" &&
     "$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/transport_helpers.sh" | awk '{print $1}')" == "${HELPER_SHA256}" &&
     -f "${DOCKER_EVIDENCE}/base-image.json" &&
     ! -L "${DOCKER_EVIDENCE}/base-image.json" &&
     "$(sha256sum "${DOCKER_EVIDENCE}/base-image.json" | awk '{print $1}')" == "${BASE_IMAGE_INSPECT_SHA256}" ]] || {
    echo "local reproduction lifecycle bytes changed after launch" >&2
    return 1
  }
  if [[ "${QUALIFICATION_MODE}" == generated-bootstrap-correctness ]]; then
    [[ "$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:tools/phantom_gpu/run_local_reproduction.sh" | sha256sum | awk '{print $1}')" == "${ENTRYPOINT_SHA256}" ]] || {
      echo "detached correctness lifecycle differs from the selected ACE commit" >&2
      return 1
    }
  fi
  [[ "$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:tools/phantom_gpu/transport_helpers.sh" | sha256sum | awk '{print $1}')" == "${HELPER_SHA256}" ]] || {
    echo "detached transport helper differs from the selected ACE commit" >&2
    return 1
  }
  [[ "$(sha256sum "${PAYLOAD}/payload.json" | awk '{print $1}')" == "${PAYLOAD_JSON_SHA256}" &&
     "$(sha256sum "${PAYLOAD}/SHA256SUMS" | awk '{print $1}')" == "${PAYLOAD_SUMS_SHA256}" ]]
  (cd "${PAYLOAD}" && sha256sum -c SHA256SUMS >/dev/null)
}

load_created_container() {
  local receipt="${DOCKER_EVIDENCE}/disposable-container-created.json"
  if [[ -e "${receipt}" || -L "${receipt}" ]]; then
    [[ -f "${receipt}" && ! -L "${receipt}" ]] || {
      echo "created-container receipt is not a regular file" >&2
      return 1
    }
    clear_regular_work_file "${receipt}.work"
  else
    clear_regular_work_file "${receipt}.work"
    local -a candidates=()
    mapfile -t candidates < <(docker ps -aq --no-trunc \
      --filter "name=^/${CONTAINER_NAME}$" \
      --filter "label=ace.phantom.intent-sha256=${INTENT_SHA256}")
    [[ ${#candidates[@]} -eq 1 && "${candidates[0]}" =~ ^[0-9a-f]{64}$ ]] || {
      echo "could not recover exactly one intent-bound local container" >&2
      return 1
    }
    capture_container_receipt "${receipt}" "${candidates[0]}" any
  fi
  CONTAINER_ID="$(validate_container "${receipt}" "" any)"
  CREATED_CONTAINER_ID="${CONTAINER_ID}"
  CLEANUP_AUTHORIZED=true
}

prior_cleanup_is_valid() {
  local receipt="${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] || return 1
  cmp -s "${receipt}" <(printf '%s\tremoved-and-absent\n%s\tname-inventory-empty\n%s\trun-label-inventory-empty\n' \
    "${CREATED_CONTAINER_ID}" "${CONTAINER_NAME}" "${RUN_LABEL}") || return 1
  ! docker inspect --type container "${CREATED_CONTAINER_ID}" >/dev/null 2>&1 || return 1
  [[ -z "$(docker ps -aq --no-trunc --filter "name=^/${CONTAINER_NAME}$")" &&
     -z "$(docker ps -aq --no-trunc --filter "label=ace.phantom.run=${RUN_LABEL}")" ]]
}

recover_interrupted_cleanup() {
  prior_cleanup_is_valid && return 0
  local expected_intent
  expected_intent="$(printf '%s\t%s\t%s\n' \
    "${CREATED_CONTAINER_ID}" "${CONTAINER_NAME}" "${RUN_LABEL}")"
  [[ -f "${DOCKER_EVIDENCE}/cleanup-intent.tsv" &&
     ! -L "${DOCKER_EVIDENCE}/cleanup-intent.tsv" &&
     -f "${DOCKER_EVIDENCE}/cleanup.txt" &&
     ! -L "${DOCKER_EVIDENCE}/cleanup.txt" &&
     "$(cat "${DOCKER_EVIDENCE}/cleanup-intent.tsv")" == "${expected_intent}" &&
     -z "$(docker ps -aq --no-trunc --filter "name=^/${CONTAINER_NAME}$")" &&
     -z "$(docker ps -aq --no-trunc --filter "label=ace.phantom.run=${RUN_LABEL}")" ]] || {
    echo "container disappeared without a complete exact-cleanup intent" >&2
    return 1
  }
  local verification="${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  {
    printf '%s\tremoved-and-absent\n' "${CREATED_CONTAINER_ID}"
    printf '%s\tname-inventory-empty\n' "${CONTAINER_NAME}"
    printf '%s\trun-label-inventory-empty\n' "${RUN_LABEL}"
  } >"${verification}.tmp"
  mv -- "${verification}.tmp" "${verification}"
  prior_cleanup_is_valid
}

validate_local_verification_pair() {
  local extraction_root="$1"
  local receipt="$2"
  python3 - "${extraction_root}" "${receipt}" "${PIPELINE_EXIT}" <<'PY'
import hashlib, json, pathlib, re, stat, sys

extraction = pathlib.Path(sys.argv[1])
receipt_path = pathlib.Path(sys.argv[2])
expected_exit = int(sys.argv[3])
if extraction.is_symlink() or not extraction.is_dir():
    raise SystemExit("verified local extraction is not a regular directory")
receipt_stat = receipt_path.lstat()
if not stat.S_ISREG(receipt_stat.st_mode):
    raise SystemExit("local verification receipt is not regular")
if {item.name for item in extraction.iterdir()} != {"results"}:
    raise SystemExit("verified local extraction has an unexpected root inventory")
results = extraction / "results"
if results.is_symlink() or not results.is_dir():
    raise SystemExit("verified local results root is invalid")
for item in extraction.rglob("*"):
    if item.is_symlink() or not (item.is_dir() or item.is_file()):
        raise SystemExit("verified local extraction contains a nonregular entry")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("local verification receipt has a duplicate key")
        value[key] = item
    return value

receipt = json.loads(receipt_path.read_text(encoding="utf-8"), object_pairs_hook=unique)
expected_keys = {
    "status", "mode", "pipeline_exit_code", "verified_file_count",
    "sha256_manifest_sha256",
}
if not isinstance(receipt, dict) or set(receipt) != expected_keys:
    raise SystemExit("local verification receipt has an invalid shape")
if (
    receipt["status"] != "pass"
    or receipt["mode"] != "local"
    or type(receipt["pipeline_exit_code"]) is not int
    or receipt["pipeline_exit_code"] != expected_exit
    or type(receipt["verified_file_count"]) is not int
    or receipt["verified_file_count"] < 0
    or re.fullmatch(r"[0-9a-f]{64}", str(receipt["sha256_manifest_sha256"])) is None
):
    raise SystemExit("local verification receipt differs from the observed invocation")
sums = results / "SHA256SUMS"
if sums.is_symlink() or not sums.is_file():
    raise SystemExit("verified local result lacks a regular SHA256SUMS")
listed = {}
for line in sums.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
    if match is None:
        raise SystemExit("verified local result manifest is malformed")
    name = match.group(3).removeprefix("./")
    if name in listed:
        raise SystemExit("verified local result manifest contains a duplicate")
    listed[name] = match.group(1)
actual = {
    str(item.relative_to(results)): hashlib.sha256(item.read_bytes()).hexdigest()
    for item in results.rglob("*") if item.is_file() and item != sums
}
if listed != actual:
    raise SystemExit("verified local result inventory changed during finalization")
if (
    receipt["verified_file_count"] != len(actual)
    or receipt["sha256_manifest_sha256"] != hashlib.sha256(sums.read_bytes()).hexdigest()
):
    raise SystemExit("local verification receipt does not bind the exact result inventory")
PY
}

clear_local_verification_work() {
  local path
  for path in "${OUTPUT}/verified-local-result" \
    "${OUTPUT}/verified-local-result.work"; do
    if [[ -L "${path}" || ( -e "${path}" && ! -d "${path}" ) ]]; then
      echo "refusing to clear an unsafe local verification directory: ${path}" >&2
      return 1
    fi
  done
  for path in "${OUTPUT}/local-result-verification.json" \
    "${OUTPUT}/local-result-verification.work.json"; do
    if [[ -L "${path}" || ( -e "${path}" && ! -f "${path}" ) ]]; then
      echo "refusing to clear an unsafe local verification receipt: ${path}" >&2
      return 1
    fi
  done
  rm -rf -- "${OUTPUT}/verified-local-result" \
    "${OUTPUT}/verified-local-result.work"
  rm -f -- "${OUTPUT}/local-result-verification.json" \
    "${OUTPUT}/local-result-verification.work.json"
}

promote_local_verification() {
  local final_extraction="${OUTPUT}/verified-local-result"
  local work_extraction="${OUTPUT}/verified-local-result.work"
  local final_receipt="${OUTPUT}/local-result-verification.json"
  local work_receipt="${OUTPUT}/local-result-verification.work.json"
  local extraction_source="" receipt_source="" count=0 candidate
  for candidate in "${final_extraction}" "${work_extraction}"; do
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
      extraction_source="${candidate}"
      count=$((count + 1))
    fi
  done
  [[ ${count} -eq 1 ]] || return 1
  count=0
  for candidate in "${final_receipt}" "${work_receipt}"; do
    if [[ -e "${candidate}" || -L "${candidate}" ]]; then
      receipt_source="${candidate}"
      count=$((count + 1))
    fi
  done
  [[ ${count} -eq 1 ]] || return 1
  validate_local_verification_pair "${extraction_source}" "${receipt_source}" || return 1
  if [[ "${extraction_source}" != "${final_extraction}" ]]; then
    [[ ! -e "${final_extraction}" && ! -L "${final_extraction}" ]] || return 1
    mv -- "${extraction_source}" "${final_extraction}"
  fi
  if [[ "${receipt_source}" != "${final_receipt}" ]]; then
    [[ ! -e "${final_receipt}" && ! -L "${final_receipt}" ]] || return 1
    mv -- "${receipt_source}" "${final_receipt}"
  fi
  validate_local_verification_pair "${final_extraction}" "${final_receipt}"
}

verify_local_result_archive() {
  promote_local_verification && return 0
  clear_local_verification_work || return 2
  set +e
  verify_result_archive "${OUTPUT}/local-result.tar.gz" local \
    "${PIPELINE_EXIT}" "${OUTPUT}/verified-local-result.work" \
    >"${OUTPUT}/local-result-verification.work.json"
  local verification_exit="$?"
  set -e
  if [[ ${verification_exit} -ne 0 ]]; then
    clear_local_verification_work || return 2
    return 1
  fi
  if ! promote_local_verification; then
    echo "could not atomically promote verified local result evidence" >&2
    return 2
  fi
}

validate_passing_closure() {
  [[ ${PIPELINE_EXIT} -eq 0 ]] || {
    echo "refusing a passing local closure for a nonzero pipeline exit" >&2
    return 1
  }
  load_intent
  verify_lifecycle_identity
  load_start_attempt_state
  [[ "${START_ATTEMPT_PRESENT}" == true ]] || {
    echo "passing local lifecycle lacks a durable container start attempt" >&2
    return 1
  }
  validate_container_id_receipt "${DOCKER_EVIDENCE}/start.txt" \
    "${CREATED_CONTAINER_ID}"
  validate_start_observation \
    "${DOCKER_EVIDENCE}/container-start-observation.json"
  [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
       "${DOCKER_EVIDENCE}/container-start-observation.json")" == effect-observed ]] || {
    echo "passing local lifecycle lacks an observed container start effect" >&2
    return 1
  }
  validate_container "${DOCKER_EVIDENCE}/disposable-container-created.json" \
    "${CREATED_CONTAINER_ID}" any >/dev/null
  validate_container "${DOCKER_EVIDENCE}/container-started.json" \
    "${CREATED_CONTAINER_ID}" started >/dev/null
  validate_container "${DOCKER_EVIDENCE}/disposable-container.json" \
    "${CREATED_CONTAINER_ID}" completed 0 >/dev/null
  prior_cleanup_is_valid
  [[ -f "${OUTPUT}/local-container.log" && ! -L "${OUTPUT}/local-container.log" ]]
  [[ -f "${OUTPUT}/local-result.tar.gz" &&
     ! -L "${OUTPUT}/local-result.tar.gz" &&
     -f "${OUTPUT}/local-result.tar.gz.sha256" &&
     ! -L "${OUTPUT}/local-result.tar.gz.sha256" &&
     -z "$(find "${RESULT_STAGING}" -mindepth 1 -print -quit)" ]]
  (
    cd "${OUTPUT}"
    [[ "$(wc -l <local-result.tar.gz.sha256)" -eq 1 ]]
    grep -Eq '^[0-9a-f]{64}  local-result\.tar\.gz$' \
      local-result.tar.gz.sha256
    sha256sum -c local-result.tar.gz.sha256 >/dev/null
  )
  validate_local_verification_pair "${OUTPUT}/verified-local-result" \
    "${OUTPUT}/local-result-verification.json"
  validate_lifecycle_receipt "${OUTPUT}/lifecycle.json" pass verified
  reject_lifecycle_work_files
}

lifecycle_summary() {
  python3 - "${OUTPUT}/lifecycle.json" <<'PY'
import json, pathlib, stat, sys

path = pathlib.Path(sys.argv[1])
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("local lifecycle receipt is not regular")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("local lifecycle receipt has a duplicate key")
        value[key] = item
    return value

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
keys = {
    "schema_version", "status", "result_archive_status",
    "requested_execution_mode", "qualification_mode", "ace_commit",
    "phantom_commit", "base_image", "base_config_digest", "container_id",
    "task_label", "run_label", "run_nonce", "pipeline_exit_code",
    "container_cleanup", "container_name_inventory", "run_label_inventory",
}
if not isinstance(value, dict) or set(value) != keys:
    raise SystemExit("local lifecycle receipt has an invalid shape")
if (
    value["schema_version"] != "ace.phantom.local-reproduction-lifecycle/1.0.0"
    or value["status"] not in {"pass", "failed"}
    or value["result_archive_status"] not in {
        "verified", "missing", "missing-or-rejected", "rejected",
    }
    or type(value["pipeline_exit_code"]) is not int
    or not 0 <= value["pipeline_exit_code"] <= 255
    or (value["status"] == "pass" and (
        value["result_archive_status"] != "verified"
        or value["pipeline_exit_code"] != 0
    ))
):
    raise SystemExit("local lifecycle receipt has an invalid outcome")
print(value["status"])
print(value["result_archive_status"])
print(value["pipeline_exit_code"])
PY
}

validate_failed_result_receipt() {
  python3 - "${OUTPUT}/local-result-verification.json" "${PIPELINE_EXIT}" <<'PY'
import json, pathlib, stat, sys

path = pathlib.Path(sys.argv[1])
expected_exit = int(sys.argv[2])
metadata = path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("failed local result receipt is not regular")

def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit("failed local result receipt has a duplicate key")
        value[key] = item
    return value

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
if (
    not isinstance(value, dict)
    or set(value) != {
        "schema_version", "status", "mode", "pipeline_exit_code", "reason",
    }
    or value["schema_version"] != "ace.phantom.local-result-verification/1.0.0"
    or value["status"] != "failed"
    or value["mode"] != "local"
    or type(value["pipeline_exit_code"]) is not int
    or value["pipeline_exit_code"] != expected_exit
    or not isinstance(value["reason"], str)
    or not value["reason"].strip()
):
    raise SystemExit("failed local result receipt differs from the observed lifecycle")
PY
}

validate_failed_closure() {
  local archive_status="$1"
  validate_container "${DOCKER_EVIDENCE}/disposable-container-created.json" \
    "${CREATED_CONTAINER_ID}" any >/dev/null
  prior_cleanup_is_valid
  load_start_attempt_state
  local observation_status=""
  if [[ -e "${DOCKER_EVIDENCE}/container-start-observation.json" ||
        -L "${DOCKER_EVIDENCE}/container-start-observation.json" ]]; then
    [[ "${START_ATTEMPT_PRESENT}" == true ]] || return 1
    validate_start_observation \
      "${DOCKER_EVIDENCE}/container-start-observation.json"
    observation_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "${DOCKER_EVIDENCE}/container-start-observation.json")"
  fi
  if [[ -e "${DOCKER_EVIDENCE}/container-started.json" ||
        -L "${DOCKER_EVIDENCE}/container-started.json" ]]; then
    [[ "${START_ATTEMPT_PRESENT}" == true &&
       "${observation_status}" == effect-observed ]] || return 1
    validate_container_id_receipt "${DOCKER_EVIDENCE}/start.txt" \
      "${CREATED_CONTAINER_ID}"
    validate_container "${DOCKER_EVIDENCE}/container-started.json" \
      "${CREATED_CONTAINER_ID}" started >/dev/null
  fi
  if [[ -e "${DOCKER_EVIDENCE}/disposable-container.json" ||
        -L "${DOCKER_EVIDENCE}/disposable-container.json" ]]; then
    [[ "${observation_status}" == effect-observed ]] || return 1
    validate_container "${DOCKER_EVIDENCE}/disposable-container.json" \
      "${CREATED_CONTAINER_ID}" completed "${PIPELINE_EXIT}" >/dev/null
  fi
  if [[ "${archive_status}" == verified ]]; then
    [[ -f "${DOCKER_EVIDENCE}/disposable-container.json" &&
       ! -L "${DOCKER_EVIDENCE}/disposable-container.json" ]]
    validate_local_verification_pair "${OUTPUT}/verified-local-result" \
      "${OUTPUT}/local-result-verification.json"
  else
    validate_failed_result_receipt
  fi
  reject_lifecycle_work_files
}

validate_closure_receipts() {
  local -a summary=()
  mapfile -t summary < <(lifecycle_summary)
  [[ ${#summary[@]} -eq 3 ]] || {
    echo "could not load the exact local lifecycle outcome" >&2
    return 1
  }
  local status="${summary[0]}"
  local archive_status="${summary[1]}"
  [[ "${summary[2]}" == "${PIPELINE_EXIT}" ]] || {
    echo "local lifecycle exit differs from controller evidence" >&2
    return 1
  }
  load_intent
  verify_lifecycle_identity
  validate_lifecycle_receipt "${OUTPUT}/lifecycle.json" \
    "${status}" "${archive_status}"
  if [[ "${status}" == pass ]]; then
    validate_passing_closure
  else
    validate_failed_closure "${archive_status}"
  fi
}

close_after_cleanup() {
  if ! verify_or_move_result; then
    write_result_failure "local pipeline did not publish a valid result archive and sidecar"
    write_lifecycle failed missing-or-rejected
    checksum_close
    return 1
  fi
  local verification_error
  verification_error="$(mktemp)"
  set +e
  verify_local_result_archive 2>"${verification_error}"
  local verification_exit="$?"
  set -e
  if [[ ${verification_exit} -eq 1 ]]; then
    write_result_failure "$(tr '\n' ' ' <"${verification_error}")"
    rm -f -- "${verification_error}"
    write_lifecycle failed rejected
    checksum_close
    return 1
  elif [[ ${verification_exit} -ne 0 ]]; then
    cat "${verification_error}" >&2
    rm -f -- "${verification_error}"
    return 1
  fi
  rm -f -- "${verification_error}"
  local lifecycle_status=pass
  [[ ${PIPELINE_EXIT} -eq 0 ]] || lifecycle_status=failed
  write_lifecycle "${lifecycle_status}" verified
  checksum_close
  if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
    echo "local reproduction failed with exit ${PIPELINE_EXIT}" >&2
    return "${PIPELINE_EXIT}"
  fi
  echo "local reproduction passed: ${OUTPUT}"
}

finalize_run() {
  load_intent
  verify_lifecycle_identity
  load_transport_helpers
  [[ "$(docker context show)" == "${DOCKER_CONTEXT_ID}" &&
     "$(docker info --format '{{.ID}}')" == "${DOCKER_DAEMON_ID}" ]] || {
    echo "Docker daemon identity changed after local launch" >&2
    return 1
  }
  load_created_container
  load_start_attempt_state
  if ! docker inspect --type container "${CONTAINER_ID}" >/dev/null 2>&1; then
    recover_interrupted_cleanup || return 1
  fi
  if prior_cleanup_is_valid; then
    CLEANUP_AUTHORIZED=false
    CONTAINER_ID=""
    clear_regular_work_file "${DOCKER_EVIDENCE}/start.txt.work"
    clear_regular_work_file "${DOCKER_EVIDENCE}/container-started.json.work"
    clear_regular_work_file "${DOCKER_EVIDENCE}/container-start-observation.json.work"
    clear_regular_work_file "${DOCKER_EVIDENCE}/container-current.json.work"
    clear_regular_work_file "${DOCKER_EVIDENCE}/disposable-container.json.work"
    clear_regular_work_file "${DOCKER_EVIDENCE}/disposable-container.json.tmp"
    clear_regular_work_file "${OUTPUT}/local-container.log.work"
    clear_regular_work_file "${OUTPUT}/local-container.log.tmp"
    capture_after_inventory
    if [[ -f "${DOCKER_EVIDENCE}/disposable-container.json" ]]; then
      PIPELINE_EXIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["State"]["ExitCode"])' "${DOCKER_EVIDENCE}/disposable-container.json")"
      validate_container "${DOCKER_EVIDENCE}/disposable-container.json" \
        "${CREATED_CONTAINER_ID}" completed "${PIPELINE_EXIT}" >/dev/null
      [[ -f "${OUTPUT}/local-container.log" && ! -L "${OUTPUT}/local-container.log" ]] || return 1
      close_after_cleanup
      return $?
    fi
    if [[ -f "${DOCKER_EVIDENCE}/container-started.json" &&
          ! -L "${DOCKER_EVIDENCE}/container-started.json" ]]; then
      validate_container "${DOCKER_EVIDENCE}/container-started.json" \
        "${CREATED_CONTAINER_ID}" started >/dev/null
      [[ -f "${OUTPUT}/local-container.log" &&
         ! -L "${OUTPUT}/local-container.log" ]] || {
        : >"${OUTPUT}/local-container.log.work"
        atomic_install_regular "${OUTPUT}/local-container.log.work" \
          "${OUTPUT}/local-container.log"
      }
      write_result_failure \
        "container start was confirmed by an observed effect, but terminal controller evidence was unavailable after exact cleanup"
      write_lifecycle failed missing
      checksum_close
      return 1
    fi
    [[ -f "${OUTPUT}/local-container.log" &&
       ! -L "${OUTPUT}/local-container.log" ]] || {
      : >"${OUTPUT}/local-container.log.work"
      atomic_install_regular "${OUTPUT}/local-container.log.work" \
        "${OUTPUT}/local-container.log"
    }
    if [[ "${START_ATTEMPT_PRESENT}" == true ]]; then
      write_result_failure \
        "a durable container start attempt was prepared, but whether it reached Docker and had an effect was unknown before exact cleanup"
    else
      write_result_failure \
        "container was cleaned after controller cutoff preceded the durable start attempt"
    fi
    write_lifecycle failed missing
    checksum_close
    return 1
  fi
  local current="${DOCKER_EVIDENCE}/container-current.json"
  capture_container_receipt "${current}" "${CONTAINER_ID}" any "" true
  local status
  status="$(container_status "${current}")"
  if [[ "${status}" == created ]]; then
    clear_regular_work_file "${DOCKER_EVIDENCE}/start.txt.work"
    if [[ "${START_ATTEMPT_PRESENT}" == true ]]; then
      publish_start_observation unknown false created
    fi
    : >"${OUTPUT}/local-container.log.work"
    atomic_install_regular "${OUTPUT}/local-container.log.work" \
      "${OUTPUT}/local-container.log"
    remove_exact_container
    capture_after_inventory
    if [[ "${START_ATTEMPT_PRESENT}" == true ]]; then
      write_result_failure \
        "a durable container start attempt was prepared, but no effect was observed and whether the request reached Docker is unknown"
    else
      write_result_failure \
        "container remained unstarted after controller cutoff preceded the durable start attempt"
    fi
    write_lifecycle failed missing
    checksum_close
    return 1
  fi
  [[ "${START_ATTEMPT_PRESENT}" == true ]] || {
    echo "an exact container start effect exists without a durable start attempt" >&2
    return 1
  }
  ensure_container_id_receipt
  capture_container_receipt "${DOCKER_EVIDENCE}/container-started.json" \
    "${CONTAINER_ID}" started
  publish_start_observation unknown false "${status}"
  local wait_output
  wait_output="$(docker wait "${CONTAINER_ID}")"
  [[ "${wait_output}" =~ ^[0-9]+$ && ${wait_output} -le 255 ]] || {
    echo "docker wait returned an invalid pipeline exit code" >&2
    return 1
  }
  PIPELINE_EXIT="${wait_output}"
  clear_regular_work_file "${OUTPUT}/local-container.log.work"
  docker logs "${CONTAINER_ID}" >"${OUTPUT}/local-container.log.work" 2>&1
  atomic_install_regular "${OUTPUT}/local-container.log.work" \
    "${OUTPUT}/local-container.log"
  capture_container_receipt "${DOCKER_EVIDENCE}/disposable-container.json" \
    "${CONTAINER_ID}" completed "${PIPELINE_EXIT}"
  remove_exact_container
  capture_after_inventory
  close_after_cleanup
}

if [[ "${RUN_MODE}" == detached-finalize ]]; then
  OUTPUT="$(realpath -e -- "${OUTPUT_ARGUMENT}")"
  [[ -d "${OUTPUT}" ]] || { echo "detached local output is not a directory" >&2; exit 1; }
  exec 9<"${OUTPUT}"
  flock -n 9 || { echo "another local finalizer owns this output" >&2; exit 1; }
  PAYLOAD="${OUTPUT}/payload"
  RESULT_STAGING="${OUTPUT}/incoming-results"
  DOCKER_EVIDENCE="${OUTPUT}/docker"
  INTENT="${OUTPUT}/detached-intent.json"
  if verify_closed_bundle; then
    mapfile -t closed_summary < <(lifecycle_summary)
    [[ ${#closed_summary[@]} -eq 3 ]] || {
      echo "checksum-closed local lifecycle has an invalid outcome receipt" >&2
      exit 1
    }
    PIPELINE_EXIT="${closed_summary[2]}"
    load_intent
    verify_lifecycle_identity
    load_transport_helpers
    load_created_container
    validate_closure_receipts
    CLEANUP_AUTHORIZED=false
    CONTAINER_ID=""
    echo "local reproduction was already checksum-closed: ${OUTPUT}"
    exit "${PIPELINE_EXIT}"
  fi
  finalize_run
  exit $?
fi

git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
DEPENDENCY_LOCK="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env")"
lock_value() {
  local value
  value="$(sed -n "s/^$1=//p" <<<"${DEPENDENCY_LOCK}")"
  [[ -n "${value}" && "${value}" != *$'\n'* ]] || {
    echo "selected ACE commit has an invalid $1 dependency lock" >&2
    exit 1
  }
  printf '%s\n' "${value}"
}
BASE_IMAGE="$(lock_value CUDA_IMAGE)"
BASE_CONFIG="$(lock_value CUDA_IMAGE_CONFIG)"
LOCKED_PHANTOM_COMMIT="$(lock_value PHANTOM_COMMIT)"
[[ "${PHANTOM_COMMIT}" == "${LOCKED_PHANTOM_COMMIT}" ]] || {
  echo "requested Phantom commit does not match the selected ACE commit lock" >&2
  exit 1
}
if [[ "${QUALIFICATION_MODE}" == "generated-bootstrap-correctness" ]]; then
  expected_entrypoint_sha256="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:tools/phantom_gpu/run_local_reproduction.sh" | sha256sum | awk '{print $1}')"
  observed_entrypoint_sha256="$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/run_local_reproduction.sh" | awk '{print $1}')"
  [[ "${observed_entrypoint_sha256}" == "${expected_entrypoint_sha256}" ]] || {
    echo "correctness local-replay entrypoint differs from the selected ACE commit" >&2
    exit 1
  }
fi
expected_helper_sha256="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:tools/phantom_gpu/transport_helpers.sh" | sha256sum | awk '{print $1}')"
observed_helper_sha256="$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/transport_helpers.sh" | awk '{print $1}')"
[[ "${observed_helper_sha256}" == "${expected_helper_sha256}" ]] || {
  echo "local-replay transport helper differs from the selected ACE commit" >&2
  exit 1
}
load_transport_helpers

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
[[ ! -e "${OUTPUT}" ]] || { echo "output already exists: ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"
exec 9<"${OUTPUT}"
flock 9
PAYLOAD="${OUTPUT}/payload"
RESULT_STAGING="${OUTPUT}/incoming-results"
DOCKER_EVIDENCE="${OUTPUT}/docker"
INTENT="${OUTPUT}/detached-intent.json"
mkdir -p "${RESULT_STAGING}" "${DOCKER_EVIDENCE}"
chmod 0700 "${RESULT_STAGING}" "${DOCKER_EVIDENCE}"

RUN_NONCE="$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RUN_NONCE}"
if [[ "${QUALIFICATION_MODE}" == full ]]; then
  CONTAINER_NAME="ace-phantom-retained-local-${RUN_ID}"
  TASK_LABEL="retained-ckks-local-reproduction"
else
  CONTAINER_NAME="ace-phantom-${QUALIFICATION_MODE}-local-${RUN_ID}"
  TASK_LABEL="${QUALIFICATION_MODE}-local-reproduction"
fi
RUN_LABEL="local-reproduction-${RUN_NONCE}"
docker ps -a --no-trunc >"${DOCKER_EVIDENCE}/containers-before.txt"
docker image ls --no-trunc >"${DOCKER_EVIDENCE}/images-before.txt"
docker system df >"${DOCKER_EVIDENCE}/disk-before.txt"

package_arguments=(--mode "${QUALIFICATION_MODE}" --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}" --bootstrap-run-root "${BOOTSTRAP_RUN_ROOT}")
if [[ "${QUALIFICATION_MODE}" == full ]]; then
  package_arguments+=(--ordinary-run-root "${ORDINARY_RUN_ROOT}" \
    --retained-run-root "${RETAINED_RUN_ROOT}")
fi
bash "${SCRIPT_DIR}/package_runpod_sources.sh" "${package_arguments[@]}" "${PAYLOAD}"
(cd "${PAYLOAD}" && sha256sum -c SHA256SUMS >/dev/null)
docker pull --platform linux/amd64 "${BASE_IMAGE}" | tee "${DOCKER_EVIDENCE}/pull.txt"
[[ "$(docker image inspect -f '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_CONFIG}" ]] || {
  echo "pulled base config digest does not match the lock" >&2
  exit 1
}
docker image inspect "${BASE_IMAGE}" >"${DOCKER_EVIDENCE}/base-image.json"
BASE_IMAGE_INSPECT_SHA256="$(sha256sum "${DOCKER_EVIDENCE}/base-image.json" | awk '{print $1}')"

BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ACE_PHANTOM_BUILD_JOBS must be a positive integer" >&2
  exit 1
}
DOCKER_CONTEXT_ID="$(docker context show)"
DOCKER_DAEMON_ID="$(docker info --format '{{.ID}}')"
ENTRYPOINT_SHA256="$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/run_local_reproduction.sh" | awk '{print $1}')"
HELPER_SHA256="$(sha256sum "${REPO_ROOT}/tools/phantom_gpu/transport_helpers.sh" | awk '{print $1}')"
python3 - "${INTENT}" "${OUTPUT}" "${RUN_MODE}" "${QUALIFICATION_MODE}" \
  "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${BASE_IMAGE}" "${BASE_CONFIG}" \
  "${CONTAINER_NAME}" "${TASK_LABEL}" "${RUN_LABEL}" "${DOCKER_CONTEXT_ID}" \
  "${RUN_NONCE}" "${DOCKER_DAEMON_ID}" "${PAYLOAD}" "${ENTRYPOINT_SHA256}" \
  "${HELPER_SHA256}" "${BUILD_JOBS}" "${RESULT_STAGING}" \
  "${BASE_IMAGE_INSPECT_SHA256}" <<'PY'
import hashlib, json, os, pathlib, sys
(
    intent_path, output_path, execution_mode, qualification_mode, ace_commit,
    phantom_commit, base_image, base_config, container_name, task_label,
    run_label, docker_context, run_nonce, docker_daemon, payload_path, entrypoint_sha,
    helper_sha, build_jobs, staging_path, base_image_inspect_sha,
) = sys.argv[1:]
output = pathlib.Path(output_path).resolve()
payload = pathlib.Path(payload_path).resolve()
staging = pathlib.Path(staging_path).resolve()
output_stat = output.stat()
staging_stat = staging.stat()
value = {
    "schema_version": "ace.phantom.local-reproduction-intent/1.0.0", "status": "prepared",
    "requested_execution_mode": execution_mode, "qualification_mode": qualification_mode,
    "output_directory": str(output), "output_device": output_stat.st_dev,
    "output_inode": output_stat.st_ino, "result_staging_directory": str(staging),
    "result_staging_device": staging_stat.st_dev,
    "result_staging_inode": staging_stat.st_ino,
    "ace_commit": ace_commit, "phantom_commit": phantom_commit,
    "base_image": base_image, "base_config_digest": base_config,
    "base_image_inspect_sha256": base_image_inspect_sha,
    "container_name": container_name, "task_label": task_label, "run_label": run_label,
    "run_nonce": run_nonce,
    "docker_context": docker_context, "docker_daemon_id": docker_daemon,
    "payload_json_sha256": hashlib.sha256((payload / "payload.json").read_bytes()).hexdigest(),
    "payload_sums_sha256": hashlib.sha256((payload / "SHA256SUMS").read_bytes()).hexdigest(),
    "entrypoint_sha256": entrypoint_sha, "helper_sha256": helper_sha,
    "build_jobs": int(build_jobs),
}
target = pathlib.Path(intent_path)
temporary = target.with_name(target.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(temporary, target)
directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
load_intent

CREATED_CONTAINER_ID="$(docker create --name "${CONTAINER_NAME}" \
  --label "ace.phantom.task=${TASK_LABEL}" \
  --label "ace.phantom.run=${RUN_LABEL}" \
  --label "ace.phantom.run-nonce=${RUN_NONCE}" \
  --label "ace.phantom.intent-sha256=${INTENT_SHA256}" \
  --platform linux/amd64 --runtime runc --restart no \
  --env NVIDIA_VISIBLE_DEVICES=void --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${BUILD_JOBS}" \
  --volume "${PAYLOAD}:/retained-qualification/input:ro" \
  --volume "${RESULT_STAGING}:/retained-qualification/output:rw" \
  "${BASE_IMAGE}" bash /retained-qualification/input/run_build_and_health.sh \
    --mode local --input-dir /retained-qualification/input \
    --work-dir /retained-qualification/work \
    --result-archive /retained-qualification/output/local-result.tar.gz)"
[[ "${CREATED_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "docker create did not return one exact full container ID" >&2
  exit 1
}
CONTAINER_ID="${CREATED_CONTAINER_ID}"
capture_container_receipt \
  "${DOCKER_EVIDENCE}/disposable-container-created.json" \
  "${CONTAINER_ID}" created
CLEANUP_AUTHORIZED=true

publish_start_attempt
DETACHED_HANDOFF=true
trap '' INT TERM
start_work="${DOCKER_EVIDENCE}/start.txt.work"
clear_regular_work_file "${start_work}"
set +e
docker start "${CONTAINER_ID}" >"${start_work}"
start_client_exit="$?"
set -e
current="${DOCKER_EVIDENCE}/container-current.json"
if ! capture_container_receipt "${current}" "${CONTAINER_ID}" any "" true; then
  trap 'exit 130' INT TERM
  echo "container start result is unknown; exact owned container remains recoverable" >&2
  exit 1
fi
start_status="$(container_status "${current}")"
start_output_exact=false
if validate_container_id_receipt "${start_work}" \
     "${CREATED_CONTAINER_ID}" >/dev/null 2>&1; then
  start_output_exact=true
fi
if [[ "${start_status}" == created ]]; then
  publish_start_observation "${start_client_exit}" \
    "${start_output_exact}" "${start_status}"
  clear_regular_work_file "${start_work}"
  trap 'exit 130' INT TERM
  DETACHED_HANDOFF=false
  : >"${OUTPUT}/local-container.log.work"
  atomic_install_regular "${OUTPUT}/local-container.log.work" \
    "${OUTPUT}/local-container.log"
  remove_exact_container
  capture_after_inventory
  write_result_failure \
    "container start was attempted, but no start effect was observed on the exact container"
  write_lifecycle failed missing
  checksum_close
  exit 1
fi
[[ "${start_status}" == running || "${start_status}" == exited ||
   "${start_status}" == dead ]] || {
  trap 'exit 130' INT TERM
  echo "container start result remains unknown in state ${start_status}; exact owned container remains recoverable" >&2
  exit 1
}
capture_container_receipt "${DOCKER_EVIDENCE}/container-started.json" \
  "${CONTAINER_ID}" started
ensure_container_id_receipt
publish_start_observation "${start_client_exit}" \
  "${start_output_exact}" "${start_status}"
trap 'exit 130' INT TERM
if [[ ${start_client_exit} -ne 0 ]]; then
  echo "docker start reported exit ${start_client_exit}, but the exact container start effect was observed" >&2
fi
if [[ "${RUN_MODE}" == detached-launch ]]; then
  echo "detached local reproduction started: ${OUTPUT}"
  exit 0
fi
finalize_run
