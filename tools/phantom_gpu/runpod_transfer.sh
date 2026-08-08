#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/transport_helpers.sh"

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
  --work-dir /root/runpod-work
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
