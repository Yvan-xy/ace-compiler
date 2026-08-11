#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

HOST=""
PORT=""
KEY=""
PAYLOAD=""
OUTPUT=""
REMOTE_TIMEOUT=""
EXPECTED_GPU=""

usage() {
  echo "usage: $0 --host HOST --port PORT --key KEY --payload DIR --output DIR --remote-timeout SECONDS --expected-gpu-name NAME" >&2
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --payload) PAYLOAD="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --remote-timeout) REMOTE_TIMEOUT="$2"; shift 2 ;;
    --expected-gpu-name) EXPECTED_GPU="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
if [[ -z "${HOST}" || ! "${PORT}" =~ ^[0-9]+$ || ! -f "${KEY}" ||
      ! -d "${PAYLOAD}" || -z "${OUTPUT}" ||
      ! "${REMOTE_TIMEOUT}" =~ ^[0-9]+$ ]]; then
  usage
  exit 2
fi
case "${EXPECTED_GPU}" in
  "NVIDIA A100 80GB PCIe"|"NVIDIA A100-SXM4-80GB") ;;
  *)
    echo "expected GPU must be an exact supported A100 identity" >&2
    exit 2
    ;;
esac
if (( REMOTE_TIMEOUT < 60 || REMOTE_TIMEOUT > 2100 )); then
  echo "remote timeout must reserve cleanup time within the first-run ceiling" >&2
  exit 2
fi

PAYLOAD="$(realpath -- "${PAYLOAD}")"
CORRECTNESS_ACE_COMMIT="$(python3 - "${PAYLOAD}" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

root = Path(sys.argv[1])
payload_path = root / "payload.json"
correctness_schemas = {
    "ace.phantom.generated-bootstrap-correctness-payload/1.0.0",
    "ace.phantom.native-bts-correctness-payload/1.0.0",
}
if not payload_path.is_file():
    print("")
    raise SystemExit(0)


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate payload key: {key}")
        result[key] = value
    return result


try:
    payload = json.loads(
        payload_path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
except (OSError, UnicodeError, ValueError) as error:
    try:
        text = payload_path.read_text(encoding="utf-8")
        mentions_correctness = any(schema in text for schema in correctness_schemas)
    except (OSError, UnicodeError):
        mentions_correctness = False
    if mentions_correctness:
        raise SystemExit("correctness payload JSON is invalid") from error
    print("")
    raise SystemExit(0)
if not isinstance(payload, dict) or payload.get("schema_version") not in correctness_schemas:
    print("")
    raise SystemExit(0)

sums_path = root / "SHA256SUMS"
if not sums_path.is_file():
    raise SystemExit("correctness payload lacks SHA256SUMS")
listed = {}
for line in sums_path.read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
    if match is None:
        raise SystemExit("correctness payload SHA256SUMS has a malformed entry")
    name = match.group(3).removeprefix("./")
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or str(path) != name
        or name == "SHA256SUMS"
        or name in listed
    ):
        raise SystemExit("correctness payload SHA256SUMS has an unsafe entry")
    listed[name] = match.group(1)
actual = {}
for path in root.iterdir():
    if path.name == "SHA256SUMS":
        continue
    if path.is_symlink() or not path.is_file():
        raise SystemExit("correctness payload contains a non-regular entry")
    actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
if listed != actual:
    raise SystemExit("correctness payload checksum closure is invalid")
generated_schema = "ace.phantom.generated-bootstrap-correctness-payload/1.0.0"
expected_keys = {
    "schema_version", "status", "ace_commit", "phantom_commit",
    "source_snapshots", "files",
}
if payload["schema_version"] == generated_schema:
    expected_keys.add("host_oracles_replayed")
else:
    expected_keys.update({"closed_m6", "compatibility_sha256"})
if set(payload) != expected_keys:
    raise SystemExit("correctness payload schema is invalid")
source_snapshots = payload["source_snapshots"]
files = payload["files"]
expected_files = {
    name: digest
    for name, digest in actual.items()
    if name != "payload.json"
}
if (
    payload["status"] != "pass"
    or (payload["schema_version"] == generated_schema
        and payload["host_oracles_replayed"] is not True)
    or not isinstance(payload["ace_commit"], str)
    or re.fullmatch(r"[0-9a-f]{40}", payload["ace_commit"]) is None
    or not isinstance(payload["phantom_commit"], str)
    or re.fullmatch(r"[0-9a-f]{40}", payload["phantom_commit"]) is None
    or not isinstance(source_snapshots, dict)
    or set(source_snapshots) != {
        "ace_manifest_sha256", "phantom_manifest_sha256"
    }
    or any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in source_snapshots.values()
    )
    or not isinstance(files, dict)
    or files != expected_files
):
    raise SystemExit("correctness payload schema is invalid")
print(payload["ace_commit"])
PY
)"
if [[ -n "${CORRECTNESS_ACE_COMMIT}" ]]; then
  expected_entrypoint_sha256="$(
    git -C "${REPO_ROOT}" show \
      "${CORRECTNESS_ACE_COMMIT}:tools/phantom_gpu/runpod_transfer.sh" |
      sha256sum | awk '{print $1}'
  )"
  observed_entrypoint_sha256="$(
    sha256sum "${REPO_ROOT}/tools/phantom_gpu/runpod_transfer.sh" |
      awk '{print $1}'
  )"
  if [[ "${observed_entrypoint_sha256}" != "${expected_entrypoint_sha256}" ]]; then
    echo "correctness RunPod transfer entrypoint differs from the selected ACE commit" >&2
    exit 1
  fi
  expected_transport_helper_sha256="$(
    git -C "${REPO_ROOT}" show \
      "${CORRECTNESS_ACE_COMMIT}:tools/phantom_gpu/transport_helpers.sh" |
      sha256sum | awk '{print $1}'
  )"
  observed_transport_helper_sha256="$(
    sha256sum "${REPO_ROOT}/tools/phantom_gpu/transport_helpers.sh" |
      awk '{print $1}'
  )"
  if [[ "${observed_transport_helper_sha256}" != \
        "${expected_transport_helper_sha256}" ]]; then
    echo "correctness RunPod transport helper differs from the selected ACE commit" >&2
    exit 1
  fi
fi
source "${SCRIPT_DIR}/transport_helpers.sh"
OUTPUT="$(realpath -m -- "${OUTPUT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"
KNOWN_HOSTS="${OUTPUT}/known_hosts"
TIMINGS="${OUTPUT}/transport-timings.tsv"
: >"${TIMINGS}"

SSH_OPTIONS=(
  -i "${KEY}"
  -p "${PORT}"
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o ForwardAgent=no
  -o ClearAllForwardings=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="${KNOWN_HOSTS}"
  -o ConnectTimeout=10
)
SCP_OPTIONS=(
  -i "${KEY}"
  -P "${PORT}"
  -o BatchMode=yes
  -o IdentitiesOnly=yes
  -o ForwardAgent=no
  -o ClearAllForwardings=yes
  -o StrictHostKeyChecking=yes
  -o UserKnownHostsFile="${KNOWN_HOSTS}"
  -o ConnectTimeout=10
)

record_timing() {
  local phase="$1" started="$2" ended="$3" status="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${phase}" "${started}" "${ended}" "$((ended - started))" "${status}" >>"${TIMINGS}"
}

started="$(date +%s)"
for attempt in $(seq 1 60); do
  if ssh-keyscan -p "${PORT}" -t ed25519 "${HOST}" >"${KNOWN_HOSTS}.tmp" 2>/dev/null &&
     [[ -s "${KNOWN_HOSTS}.tmp" ]]; then
    mv "${KNOWN_HOSTS}.tmp" "${KNOWN_HOSTS}"
    break
  fi
  if [[ ${attempt} -eq 60 ]]; then
    echo "SSH host key did not become available" >&2
    exit 1
  fi
  sleep 2
done
chmod 0600 "${KNOWN_HOSTS}"
ssh-keygen -lf "${KNOWN_HOSTS}" >"${OUTPUT}/host-key-fingerprint.txt"
ended="$(date +%s)"
record_timing ssh_ready "${started}" "${ended}" 0

started="$(date +%s)"
set +e
scp "${SCP_OPTIONS[@]}" -r "${PAYLOAD}" "root@${HOST}:/root/input" \
  >"${OUTPUT}/source-transfer.stdout.txt" \
  2>"${OUTPUT}/source-transfer.stderr.txt"
transfer_exit=$?
set -e
ended="$(date +%s)"
record_timing source_transfer "${started}" "${ended}" "${transfer_exit}"
[[ ${transfer_exit} -eq 0 ]]

REMOTE_ARGS=(
  env
  "ACE_RUNPOD_BASE_IMAGE=${ACE_RUNPOD_BASE_IMAGE}"
  "ACE_RUNPOD_BASE_CONFIG_DIGEST=${ACE_RUNPOD_BASE_CONFIG_DIGEST}"
  "ACE_RUNPOD_EXPECTED_GPU_NAME=${EXPECTED_GPU}"
  "ACE_PHANTOM_BUILD_JOBS=${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
  timeout --signal=TERM --kill-after=30 "${REMOTE_TIMEOUT}"
  bash /root/input/run_build_and_health.sh
  --mode runpod
  --input-dir /root/input
  --work-dir /retained-qualification/work
  --result-archive /root/runpod-result.tar.gz
)
shell_join REMOTE_COMMAND "${REMOTE_ARGS[@]}"
printf '%q ' ssh "${SSH_OPTIONS[@]}" "root@${HOST}" >"${OUTPUT}/remote-command.txt"
printf '%s\n' "${REMOTE_COMMAND}" >>"${OUTPUT}/remote-command.txt"

started="$(date +%s)"
set +e
ssh "${SSH_OPTIONS[@]}" "root@${HOST}" "${REMOTE_COMMAND}" \
  >"${OUTPUT}/remote.stdout.txt" \
  2>"${OUTPUT}/remote.stderr.txt"
remote_exit=$?
set -e
ended="$(date +%s)"
record_timing remote_pipeline "${started}" "${ended}" "${remote_exit}"

started="$(date +%s)"
set +e
scp "${SCP_OPTIONS[@]}" \
  "root@${HOST}:/root/runpod-result.tar.gz" \
  "root@${HOST}:/root/runpod-result.tar.gz.sha256" \
  "${OUTPUT}/" >"${OUTPUT}/result-transfer.stdout.txt" \
  2>"${OUTPUT}/result-transfer.stderr.txt"
retrieve_exit=$?
set -e
ended="$(date +%s)"
record_timing result_retrieval "${started}" "${ended}" "${retrieve_exit}"
[[ ${retrieve_exit} -eq 0 ]]
(
  cd "${OUTPUT}"
  sha256sum -c runpod-result.tar.gz.sha256
)
verify_result_archive \
  "${OUTPUT}/runpod-result.tar.gz" runpod "${remote_exit}" \
  "${OUTPUT}/verified-runpod-result" \
  >"${OUTPUT}/runpod-result-verification.json"

if [[ ${remote_exit} -ne 0 ]]; then
  echo "remote pipeline failed with exit ${remote_exit}" >&2
  exit "${remote_exit}"
fi
echo "RunPod result retrieved and verified: ${OUTPUT}/runpod-result.tar.gz"
