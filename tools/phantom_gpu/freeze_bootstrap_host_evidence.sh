#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"
source "${SCRIPT_DIR}/transport_helpers.sh"

usage() {
  cat >&2 <<EOF
usage: $0 --ace-commit COMMIT --phantom-commit COMMIT OUTPUT_DIRECTORY
The reusable qualification root is
OUTPUT_DIRECTORY/verified-bootstrap-host-result/results/qualification.
Exact container cleanup evidence is written under OUTPUT_DIRECTORY/docker.
EOF
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ace-commit) ACE_COMMIT="$2"; shift 2 ;;
    --phantom-commit) PHANTOM_COMMIT="$2"; shift 2 ;;
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
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      -z "${OUTPUT_ARGUMENT}" ]]; then
  usage
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"

for lifecycle_file in \
  tools/phantom_gpu/freeze_bootstrap_host_evidence.sh \
  tools/phantom_gpu/transport_helpers.sh; do
  expected="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${lifecycle_file}" |
    sha256sum | awk '{print $1}')"
  observed="$(sha256sum "${REPO_ROOT}/${lifecycle_file}" | awk '{print $1}')"
  if [[ "${observed}" != "${expected}" ]]; then
    echo "bootstrap host-freeze lifecycle file differs from the selected ACE commit: ${lifecycle_file}" >&2
    exit 1
  fi
done

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
mkdir -p "${PAYLOAD}" "${DOCKER_EVIDENCE}"

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
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 "${PAYLOAD}/source_archive.py" \
  "${PAYLOAD}/bootstrap_environment.sh"

python3 - "${PAYLOAD}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
payload = {
    "schema_version": "ace.phantom.bootstrap-host-freeze-payload/1.0.0",
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
        "parameters": {
            "poly_degree": 16384,
            "mul_level": 26,
            "input_level": 1,
            "security_level": 0,
            "scaling_factor_bits": 56,
            "first_prime_bits": 60,
            "hamming_weight": 192,
        },
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
CONTAINER_ID=""

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
  printf '%s\tremoved-and-absent\n' "${exact_id}" \
    >>"${DOCKER_EVIDENCE}/cleanup-verification.tsv"
  CONTAINER_ID=""
}

cleanup() {
  local incoming="$?"
  trap - EXIT INT TERM
  if [[ -n "${CONTAINER_ID}" ]]; then
    set +e
    remove_exact_container
    local cleanup_exit="$?"
    set -e
    [[ ${cleanup_exit} -eq 0 ]] || incoming=1
  fi
  exit "${incoming}"
}
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

CREATED_CONTAINER_ID="$(docker create \
  --name "${CONTAINER_NAME}" \
  --label "ace.phantom.task=${TASK_LABEL}" \
  --platform linux/amd64 \
  --runtime runc \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env NVIDIA_DRIVER_CAPABILITIES=none \
  --env ACE_RUNPOD_BASE_IMAGE="${BASE_IMAGE}" \
  --env ACE_RUNPOD_BASE_CONFIG_DIGEST="${BASE_CONFIG}" \
  --env ACE_PHANTOM_BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}" \
  --volume "${PAYLOAD}:/bootstrap-freeze/input:ro" \
  --volume "${OUTPUT}:/bootstrap-freeze/output:rw" \
  "${BASE_IMAGE}" bash -lc '
set -euo pipefail
export LC_ALL=C TZ=UTC
INPUT=/bootstrap-freeze/input
OUTPUT=/bootstrap-freeze/output
WORK=${OUTPUT}/work
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
if payload.get("schema_version") != "ace.phantom.bootstrap-host-freeze-payload/1.0.0":
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
bash "${ACE_SOURCE}/tools/phantom_gpu/compile_only.sh" \
  --gate bootstrap --poly-degree 16384 --mul-level 26 --input-level 1 \
  --security-level 0 --scaling-factor-bits 56 --first-prime-bits 60 \
  --hamming-weight 192 2>&1 | tee "${RESULTS}/bootstrap-host-qualification.log"
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
python3 - "${RESULTS}/qualification" "${INPUT}" <<"PY"
import json, pathlib, sys
root, source = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
ace = json.loads((source / "ace-source.manifest.json").read_text(encoding="utf-8"))
phantom = json.loads((source / "phantom-source.manifest.json").read_text(encoding="utf-8"))
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
qualification = json.loads(
    (root / "bootstrap_qualification/qualification.json").read_text(encoding="utf-8")
)
if (qualification.get("status"), qualification.get("gate"),
        qualification.get("executable_was_run")) != ("pass", "bootstrap", False):
    raise SystemExit("bootstrap host qualification claims are inconsistent")
PY
printf "{\"schema_version\":\"ace.phantom.result-completeness/1.0.0\",\"status\":\"pass\",\"mode\":\"bootstrap-host-freeze\"}\n" \
  >"${RESULTS}/result-completeness.json"
PIPELINE_EXIT=0
')"
if [[ ! "${CREATED_CONTAINER_ID}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "docker create did not return one exact full container ID" >&2
  exit 1
fi
CONTAINER_ID="${CREATED_CONTAINER_ID}"
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-created.json"
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
        or "NVIDIA_VISIBLE_DEVICES=void" not in environment
        or "NVIDIA_DRIVER_CAPABILITIES=none" not in environment):
    raise SystemExit("created container violates the bootstrap host-freeze device boundary")
PY

set +e
docker start -a "${CONTAINER_ID}" 2>&1 |
  tee "${OUTPUT}/bootstrap-host-freeze.log"
PIPELINE_EXIT="${PIPESTATUS[0]}"
set -e
docker inspect --type container "${CONTAINER_ID}" \
  >"${DOCKER_EVIDENCE}/container-completed.json"
remove_exact_container

RESULT_ARCHIVE="${OUTPUT}/bootstrap-host-result.tar.gz"
if [[ ! -s "${RESULT_ARCHIVE}" || ! -s "${RESULT_ARCHIVE}.sha256" ]]; then
  echo "bootstrap host freeze did not publish its result archive and sidecar" >&2
  exit 1
fi
(
  cd "${OUTPUT}"
  sha256sum -c "$(basename -- "${RESULT_ARCHIVE}.sha256")"
)
verify_result_archive \
  "${RESULT_ARCHIVE}" bootstrap-host-freeze "${PIPELINE_EXIT}" \
  "${OUTPUT}/verified-bootstrap-host-result" \
  >"${OUTPUT}/bootstrap-host-result-verification.json"
if [[ ${PIPELINE_EXIT} -ne 0 ]]; then
  echo "bootstrap host freeze failed with exit ${PIPELINE_EXIT}" >&2
  exit "${PIPELINE_EXIT}"
fi

QUALIFICATION="${OUTPUT}/verified-bootstrap-host-result/results/qualification"
python3 - "${QUALIFICATION}" "${PAYLOAD}" \
  "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import json, pathlib, sys
root, payload = map(pathlib.Path, sys.argv[1:3])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
expected = {"status": "pass", "gate": "bootstrap", "source_mode": "snapshot",
            "ace_commit": sys.argv[3], "phantom_commit": sys.argv[4],
            "ace_worktree_dirty": False, "gpu_executables_were_run": False}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("verified bootstrap qualification manifest is inconsistent")
if ((root / "ace_source_manifest.json").read_bytes()
        != (payload / "ace-source.manifest.json").read_bytes()
        or (root / "phantom_source_manifest.json").read_bytes()
        != (payload / "phantom-source.manifest.json").read_bytes()):
    raise SystemExit("verified bootstrap source manifests differ from the payload")
PY
(
  cd "${QUALIFICATION}"
  sha256sum -c SHA256SUMS
)
python3 - "${OUTPUT}/lifecycle.json" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" \
  "${BASE_IMAGE}" "${BASE_CONFIG}" "${CREATED_CONTAINER_ID}" \
  "${TASK_LABEL}" <<'PY'
import json, pathlib, sys
value = {
    "schema_version": "ace.phantom.bootstrap-host-freeze-lifecycle/1.0.0",
    "status": "pass",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "base_image": sys.argv[4],
    "base_config_digest": sys.argv[5],
    "container_id": sys.argv[6],
    "task_label": sys.argv[7],
    "container_cleanup": "removed-and-absent",
    "gpu_device_requests": [],
    "gpu_executables_were_run": False,
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
echo "bootstrap host evidence: ${QUALIFICATION}"
