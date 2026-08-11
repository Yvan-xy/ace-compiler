#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <local|runpod> INPUT WORK RESULT_DIR" >&2
  exit 2
fi
MODE="$1"
INPUT="$2"
WORK="$3"
RESULT_DIR="$4"
[[ "${MODE}" == local || "${MODE}" == runpod ]] || exit 2

ACE_SOURCE="${WORK}/ace-extract/ace-source"
PHANTOM_SOURCE="${WORK}/phantom-extract/phantom-source"
BUILD="${WORK}/native-bts-build"
COMMANDS="${RESULT_DIR}/native-bts-commands.txt"
BUILD_LOG="${RESULT_DIR}/native-bts-build.log"
TIMINGS="${RESULT_DIR}/native-bts-phase-timings.tsv"
: >"${COMMANDS}"
: >"${BUILD_LOG}"
: >"${TIMINGS}"

record_command() {
  printf '%q ' "$@" >>"${COMMANDS}"
  printf '\n' >>"${COMMANDS}"
}

run_logged() {
  record_command "$@"
  "$@" >>"${BUILD_LOG}" 2>&1
}

timed() {
  local label="$1"
  shift
  local started ended status
  started="$(date +%s)"
  set +e
  "$@"
  status=$?
  set -e
  ended="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${started}" "${ended}" "$((ended - started))" \
    "${status}" >>"${TIMINGS}"
  return "${status}"
}

verify_payload() {
  python3 - "${INPUT}" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

root = Path(sys.argv[1])
payload = json.loads((root / "payload.json").read_text(encoding="utf-8"))
if payload.get("schema_version") != "ace.phantom.native-bts-correctness-payload/1.0.0" or payload.get("status") != "pass":
    raise SystemExit("unsupported native BTS payload")
listed = {}
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None:
        raise SystemExit("malformed native BTS payload checksum line")
    name = match.group(2).removeprefix("./")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or len(path.parts) != 1 or name in listed:
        raise SystemExit("unsafe native BTS payload checksum path")
    listed[name] = match.group(1)
actual = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in root.iterdir()
    if path.is_file() and path.name != "SHA256SUMS"
}
if listed != actual:
    raise SystemExit("native BTS payload checksum closure differs")
files = {name: digest for name, digest in actual.items() if name != "payload.json"}
if payload.get("files") != files:
    raise SystemExit("native BTS payload inventory differs")
if payload.get("compatibility_sha256") != actual.get("native-compatibility.json"):
    raise SystemExit("native compatibility binding differs")
identity = json.loads((root / "production-source-identity.json").read_text(encoding="utf-8"))
if identity.get("status") != "pass" or identity.get("production_runtime_source_changed") is not False or identity.get("generated_dsl_source_changed") is not False:
    raise SystemExit("production source identity is not closed")
closed = payload.get("closed_m6", {})
expected_closed = {
    "generated_record_sha256": actual["baseline-generated-phantom.json"],
    "generated_values_sha256": actual["baseline-generated-phantom.bin"],
    "comparison_sha256": actual["baseline-comparison.json"],
    "sanitizer_sha256": actual["baseline-sanitizer.json"],
}
if closed != expected_closed:
    raise SystemExit("closed M6 bindings differ")
PY
  cmp -- "${INPUT}/native_phantom_bts_correctness.cu" \
    "${ACE_SOURCE}/tools/phantom_gpu/harness/native_phantom_bts_correctness.cu"
  cmp -- "${INPUT}/native_bts_correctness.py" \
    "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py"
}

archive_qualification_inputs() {
  local name
  for name in \
    payload.json SHA256SUMS native-compatibility.json \
    native-clear-inputs.json native-clear-inputs.bin \
    ace-source.manifest.json phantom-source.manifest.json \
    baseline-ace-source.manifest.json baseline-phantom-source.manifest.json \
    baseline-fixture.json baseline-compiler-invocation.json \
    baseline-context-manifest.json baseline-resource-manifest.json \
    baseline-constant-manifest.json baseline-bootstrap-semantics.json \
    baseline-post-operation-attestation.json \
    baseline-generated-phantom.json baseline-generated-phantom.bin \
    baseline-comparison.json baseline-sanitizer.json \
    baseline-qualification.json baseline-symbol-closure.json \
    baseline-archive-member-audit.json production-source-identity.json \
    native_phantom_bts_correctness.cu; do
    cp -- "${INPUT}/${name}" "${RESULT_DIR}/qualification-input-${name}"
  done
}

build_native_oracle() {
  mkdir "${BUILD}"
  run_logged cmake -S "${PHANTOM_SOURCE}" -B "${BUILD}/phantom" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DPHANTOM_BUILD_EXAMPLES=OFF -DPHANTOM_BUILD_TESTS=OFF
  run_logged cmake --build "${BUILD}/phantom" \
    --target phantom_ordinary phantom_native_bts_oracle \
    --parallel "${ACE_PHANTOM_BUILD_JOBS:-$(nproc)}"
  local include="${PHANTOM_SOURCE}/include"
  local ordinary="${BUILD}/phantom/lib/libphantom_ordinary.a"
  local native="${BUILD}/phantom/lib/libphantom_native_bts_oracle.a"
  local source="${ACE_SOURCE}/tools/phantom_gpu/harness/native_phantom_bts_correctness.cu"
  local object="${BUILD}/native_phantom_bts_correctness.o"
  local dlink="${BUILD}/native_phantom_bts_correctness.dlink.o"
  local executable="${BUILD}/native_phantom_bts_correctness_sm80"
  run_logged /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_80 \
    -rdc=true -dc -I"${include}" "${source}" -o "${object}"
  run_logged /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_80 \
    -rdc=true -dlink "${object}" "${native}" "${ordinary}" \
    -L/usr/local/cuda/lib64 -lcudadevrt -o "${dlink}"
  run_logged c++ -std=c++17 "${object}" "${dlink}" \
    -Wl,--start-group "${native}" "${ordinary}" \
    -lntl -lgmpxx -lgmp -Wl,--end-group \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
    -lcudadevrt -lcudart -pthread -ldl -lrt -lm -o "${executable}"
  file "${executable}" >"${RESULT_DIR}/native-bts-executable-file.txt"
  cuobjdump --list-elf "${executable}" \
    >"${RESULT_DIR}/native-bts-executable-cuda-elf.txt"
  cuobjdump --dump-resource-usage "${executable}" \
    >"${RESULT_DIR}/native-bts-executable-cuda-resources.txt"
  readelf -d -s -W "${executable}" \
    >"${RESULT_DIR}/native-bts-executable-readelf.txt"
}

audit_links() {
  local ordinary="${BUILD}/phantom/lib/libphantom_ordinary.a"
  local native="${BUILD}/phantom/lib/libphantom_native_bts_oracle.a"
  local executable="${BUILD}/native_phantom_bts_correctness_sm80"
  nm -C --defined-only "${ordinary}" \
    >"${RESULT_DIR}/native-bts-ordinary-library-symbols.txt"
  nm -C --defined-only "${native}" \
    >"${RESULT_DIR}/native-bts-oracle-library-symbols.txt"
  nm -C "${executable}" >"${RESULT_DIR}/native-bts-executable-symbols.txt"
  ar t "${ordinary}" >"${RESULT_DIR}/native-bts-ordinary-library-members.txt"
  ar t "${native}" >"${RESULT_DIR}/native-bts-oracle-library-members.txt"
  python3 - "${INPUT}" "${RESULT_DIR}" "${ordinary}" "${native}" \
    "${executable}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

input_root, result_root = map(Path, sys.argv[1:3])
ordinary, native, executable = map(Path, sys.argv[3:6])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
load = lambda name: json.loads((input_root / name).read_text(encoding="utf-8"))
qualification = load("baseline-qualification.json")
closure = load("baseline-symbol-closure.json")
members = load("baseline-archive-member-audit.json")
if (
    qualification.get("status") != "pass"
    or qualification.get("generated_source_contains_native_bootstrap") is not False
    or qualification.get("production_archive_contains_native_bootstrap") is not False
    or qualification.get("primitive_only_provider_archive") is not True
    or qualification.get("symbol_closure_sha256") != digest(input_root / "baseline-symbol-closure.json")
    or qualification.get("archive_member_audit_sha256") != digest(input_root / "baseline-archive-member-audit.json")
    or closure.get("status") != "pass"
    or closure.get("native_bootstrap_symbol_count") != 0
    or members.get("status") != "pass"
    or members.get("forbidden_member_count") != 0
    or (input_root / "baseline-native_bootstrap_symbols.txt").read_bytes()
):
    raise SystemExit("closed generated/production link audit is not passing")
for inventory in members["inventories"]:
    packaged = input_root / ("baseline-" + inventory["path"])
    if digest(packaged) != inventory["sha256"]:
        raise SystemExit("closed production archive-member inventory differs")
ordinary_text = (result_root / "native-bts-ordinary-library-symbols.txt").read_text(
    encoding="utf-8", errors="replace")
native_text = (result_root / "native-bts-oracle-library-symbols.txt").read_text(
    encoding="utf-8", errors="replace")
executable_text = (result_root / "native-bts-executable-symbols.txt").read_text(
    encoding="utf-8", errors="replace")
forbidden = re.compile(r"Bootstrapper::|ModularReducer::|(?:^|[^A-Za-z0-9_])bootstrap_3\(", re.M)
ordinary_forbidden = len(forbidden.findall(ordinary_text))
native_bootstrap = native_text.count("Bootstrapper::bootstrap_3(")
executable_bootstrap = executable_text.count("Bootstrapper::bootstrap_3(")
executable_bootstrapper = executable_text.count("Bootstrapper::")
if ordinary_forbidden != 0 or native_bootstrap < 1 or executable_bootstrap < 1 or executable_bootstrapper < 1:
    raise SystemExit("native/ordinary archive partition audit failed")
record = {
    "schema_version": "ace.phantom.native-bts-link-audit/1.0.0",
    "status": "pass",
    "production_source_identity_sha256": digest(input_root / "production-source-identity.json"),
    "native_executable": {
        "sha256": digest(executable),
        "bootstrapper_symbol_count": executable_bootstrapper,
        "bootstrap_3_symbol_count": executable_bootstrap,
        "symbols_sha256": digest(result_root / "native-bts-executable-symbols.txt"),
    },
    "native_library": {
        "sha256": digest(native),
        "bootstrap_3_symbol_count": native_bootstrap,
        "symbols_sha256": digest(result_root / "native-bts-oracle-library-symbols.txt"),
        "members_sha256": digest(result_root / "native-bts-oracle-library-members.txt"),
    },
    "ordinary_library": {
        "sha256": digest(ordinary),
        "forbidden_native_symbol_count": ordinary_forbidden,
        "symbols_sha256": digest(result_root / "native-bts-ordinary-library-symbols.txt"),
        "members_sha256": digest(result_root / "native-bts-ordinary-library-members.txt"),
    },
    "closed_generated_production": {
        "status": "pass",
        "native_bootstrap_symbol_count": 0,
        "qualification_sha256": digest(input_root / "baseline-qualification.json"),
        "symbol_closure_sha256": digest(input_root / "baseline-symbol-closure.json"),
        "archive_member_audit_sha256": digest(input_root / "baseline-archive-member-audit.json"),
        "linked_generated_executable_sha256": qualification["linked_binary_sha256"],
        "generated_gpu_executable_sha256": load("baseline-generated-phantom.json")["execution"]["executable_sha256"],
        "production_adapter_archive_sha256": qualification["adapter_archive_sha256"],
        "production_provider_archive_sha256": qualification["provider_archive_sha256"],
    },
}
(result_root / "native-bts-link-audit.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

baseline_arguments() {
  BASELINE_ARGUMENTS=(
    --fixture "${INPUT}/baseline-fixture.json"
    --ace-source-manifest "${INPUT}/baseline-ace-source.manifest.json"
    --phantom-source-manifest "${INPUT}/baseline-phantom-source.manifest.json"
    --compiler-invocation "${INPUT}/baseline-compiler-invocation.json"
    --raw-air "${INPUT}/baseline-raw.air"
    --post-ckks-air "${INPUT}/baseline-post-ckks.air"
    --context-manifest "${INPUT}/baseline-context-manifest.json"
    --resource-manifest "${INPUT}/baseline-resource-manifest.json"
    --constant-manifest "${INPUT}/baseline-constant-manifest.json"
    --bootstrap-semantics "${INPUT}/baseline-bootstrap-semantics.json"
    --post-operations-air "${INPUT}/baseline-post-operations.air"
    --post-operation-attestation \
      "${INPUT}/baseline-post-operation-attestation.json"
    --baseline-gpu-record "${INPUT}/baseline-generated-phantom.json"
    --baseline-gpu-values "${INPUT}/baseline-generated-phantom.bin"
    --baseline-comparison "${INPUT}/baseline-comparison.json"
    --baseline-sanitizer "${INPUT}/baseline-sanitizer.json"
  )
}

capture_provenance() {
  {
    echo "base_image=${ACE_RUNPOD_BASE_IMAGE:-unset}"
    echo "base_config_digest=${ACE_RUNPOD_BASE_CONFIG_DIGEST:-unset}"
    /usr/local/cuda/bin/nvcc --version
    gcc --version | head -1
    g++ --version | head -1
    cmake --version | head -1
    ninja --version
    python3 --version
  } >"${RESULT_DIR}/native-bts-toolchain.txt"
  if [[ "${MODE}" == runpod ]]; then
    nvidia-smi -L >"${RESULT_DIR}/native-bts-gpu.txt"
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total \
      --format=csv,noheader >>"${RESULT_DIR}/native-bts-gpu.txt"
  else
    printf 'status=skipped-local-no-qualified-gpu\n' \
      >"${RESULT_DIR}/native-bts-gpu.txt"
  fi
}

run_native_gpu() {
  local executable="${BUILD}/native_phantom_bts_correctness_sm80"
  local ordinary="${BUILD}/phantom/lib/libphantom_ordinary.a"
  local native="${BUILD}/phantom/lib/libphantom_native_bts_oracle.a"
  local compatibility_sha expected_gpu
  compatibility_sha="$(sha256sum "${INPUT}/native-compatibility.json" | awk '{print $1}')"
  expected_gpu="${ACE_RUNPOD_EXPECTED_GPU_NAME:?expected A100 name is required}"
  record_command "${executable}" "${INPUT}/baseline-context-manifest.json" \
    "${INPUT}/native-compatibility.json" "${INPUT}/native-clear-inputs.json" \
    "${INPUT}/native-clear-inputs.bin" "${compatibility_sha}" \
    "${expected_gpu}" all "${RESULT_DIR}/native-bts-raw.json" \
    "${RESULT_DIR}/native-bts-raw.bin" direct-bootstrap-3-no-fallback
  "${executable}" "${INPUT}/baseline-context-manifest.json" \
    "${INPUT}/native-compatibility.json" "${INPUT}/native-clear-inputs.json" \
    "${INPUT}/native-clear-inputs.bin" "${compatibility_sha}" \
    "${expected_gpu}" all "${RESULT_DIR}/native-bts-raw.json" \
    "${RESULT_DIR}/native-bts-raw.bin" direct-bootstrap-3-no-fallback \
    >"${RESULT_DIR}/native-bts-gpu.stdout.txt" \
    2>"${RESULT_DIR}/native-bts-gpu.stderr.txt"
  record_command python3 -B \
    "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    finalize --compatibility "${INPUT}/native-compatibility.json" \
    --raw-record "${RESULT_DIR}/native-bts-raw.json" \
    --raw-values "${RESULT_DIR}/native-bts-raw.bin" \
    --executable "${executable}" --ordinary-library "${ordinary}" \
    --native-library "${native}" \
    --values-output "${RESULT_DIR}/native-bts-values.bin" \
    --output "${RESULT_DIR}/native-bts-correctness.json"
  python3 -B "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    finalize --compatibility "${INPUT}/native-compatibility.json" \
    --raw-record "${RESULT_DIR}/native-bts-raw.json" \
    --raw-values "${RESULT_DIR}/native-bts-raw.bin" \
    --executable "${executable}" --ordinary-library "${ordinary}" \
    --native-library "${native}" \
    --values-output "${RESULT_DIR}/native-bts-values.bin" \
    --output "${RESULT_DIR}/native-bts-correctness.json"

  compute-sanitizer --version \
    >"${RESULT_DIR}/native-bts-sanitizer-version.txt" 2>&1
  record_command compute-sanitizer --tool memcheck --error-exitcode=97 \
    --log-file "${RESULT_DIR}/native-bts-sanitizer.log" \
    "${executable}" "${INPUT}/baseline-context-manifest.json" \
    "${INPUT}/native-compatibility.json" "${INPUT}/native-clear-inputs.json" \
    "${INPUT}/native-clear-inputs.bin" "${compatibility_sha}" \
    "${expected_gpu}" sentinel "${RESULT_DIR}/native-bts-sentinel-raw.json" \
    "${RESULT_DIR}/native-bts-sentinel-raw.bin" \
    direct-bootstrap-3-no-fallback
  set +e
  compute-sanitizer --tool memcheck --error-exitcode=97 \
    --log-file "${RESULT_DIR}/native-bts-sanitizer.log" \
    "${executable}" "${INPUT}/baseline-context-manifest.json" \
    "${INPUT}/native-compatibility.json" "${INPUT}/native-clear-inputs.json" \
    "${INPUT}/native-clear-inputs.bin" "${compatibility_sha}" \
    "${expected_gpu}" sentinel "${RESULT_DIR}/native-bts-sentinel-raw.json" \
    "${RESULT_DIR}/native-bts-sentinel-raw.bin" \
    direct-bootstrap-3-no-fallback \
    >"${RESULT_DIR}/native-bts-sanitizer.stdout.txt" \
    2>"${RESULT_DIR}/native-bts-sanitizer.stderr.txt"
  sanitizer_exit=$?
  set -e
  [[ "${sanitizer_exit}" -eq 0 ]]
  record_command python3 -B \
    "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    finalize --compatibility "${INPUT}/native-compatibility.json" \
    --raw-record "${RESULT_DIR}/native-bts-sentinel-raw.json" \
    --raw-values "${RESULT_DIR}/native-bts-sentinel-raw.bin" \
    --executable "${executable}" --ordinary-library "${ordinary}" \
    --native-library "${native}" \
    --values-output "${RESULT_DIR}/native-bts-sentinel-values.bin" \
    --output "${RESULT_DIR}/native-bts-sentinel.json"
  python3 -B "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    finalize --compatibility "${INPUT}/native-compatibility.json" \
    --raw-record "${RESULT_DIR}/native-bts-sentinel-raw.json" \
    --raw-values "${RESULT_DIR}/native-bts-sentinel-raw.bin" \
    --executable "${executable}" --ordinary-library "${ordinary}" \
    --native-library "${native}" \
    --values-output "${RESULT_DIR}/native-bts-sentinel-values.bin" \
    --output "${RESULT_DIR}/native-bts-sentinel.json"
  python3 - "${executable}" "${RESULT_DIR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

executable, root = Path(sys.argv[1]), Path(sys.argv[2])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
log = root / "native-bts-sanitizer.log"
summary = [line for line in log.read_text(encoding="utf-8").splitlines()
           if "ERROR SUMMARY:" in line]
if summary != ["========= ERROR SUMMARY: 0 errors"]:
    raise SystemExit("Compute Sanitizer lacks exactly one zero-error summary")
record = {
    "schema_version": "ace.phantom.native-bts-compute-sanitizer/1.0.0",
    "status": "pass",
    "exit_status": 0,
    "error_summary_occurrences": 1,
    "error_count": 0,
    "sentinel": "alternating-real-imaginary-signs",
    "coverage": {
        "bootstrap": True, "decrypt": True,
        "post_operations": True, "teardown": True,
    },
    "bindings": {
        "executable_sha256": digest(executable),
        "sentinel_record_sha256": digest(root / "native-bts-sentinel.json"),
        "sentinel_values_sha256": digest(root / "native-bts-sentinel-values.bin"),
        "log_sha256": digest(log),
        "tool_version_sha256": digest(root / "native-bts-sanitizer-version.txt"),
    },
}
(root / "native-bts-sanitizer.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  baseline_arguments
  record_command python3 -B \
    "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    compare "${BASELINE_ARGUMENTS[@]}" \
    --compatibility "${INPUT}/native-compatibility.json" \
    --native-record "${RESULT_DIR}/native-bts-correctness.json" \
    --native-values "${RESULT_DIR}/native-bts-values.bin" \
    --sentinel-record "${RESULT_DIR}/native-bts-sentinel.json" \
    --sentinel-values "${RESULT_DIR}/native-bts-sentinel-values.bin" \
    --sanitizer-record "${RESULT_DIR}/native-bts-sanitizer.json" \
    --sanitizer-log "${RESULT_DIR}/native-bts-sanitizer.log" \
    --sanitizer-tool-version \
      "${RESULT_DIR}/native-bts-sanitizer-version.txt" \
    --executable "${executable}" \
    --link-audit "${RESULT_DIR}/native-bts-link-audit.json" \
    --output "${RESULT_DIR}/native-bts-comparison.json"
  python3 -B "${ACE_SOURCE}/tools/phantom_gpu/native_bts_correctness.py" \
    compare "${BASELINE_ARGUMENTS[@]}" \
    --compatibility "${INPUT}/native-compatibility.json" \
    --native-record "${RESULT_DIR}/native-bts-correctness.json" \
    --native-values "${RESULT_DIR}/native-bts-values.bin" \
    --sentinel-record "${RESULT_DIR}/native-bts-sentinel.json" \
    --sentinel-values "${RESULT_DIR}/native-bts-sentinel-values.bin" \
    --sanitizer-record "${RESULT_DIR}/native-bts-sanitizer.json" \
    --sanitizer-log "${RESULT_DIR}/native-bts-sanitizer.log" \
    --sanitizer-tool-version \
      "${RESULT_DIR}/native-bts-sanitizer-version.txt" \
    --executable "${executable}" \
    --link-audit "${RESULT_DIR}/native-bts-link-audit.json" \
    --output "${RESULT_DIR}/native-bts-comparison.json"
}

write_lifecycle() {
  local gpu_status="$1"
  python3 - "${MODE}" "${gpu_status}" "${INPUT}" "${RESULT_DIR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

mode, gpu_status = sys.argv[1:3]
input_root, result_root = map(Path, sys.argv[3:5])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
compatibility = json.loads((input_root / "native-compatibility.json").read_text())
record = {
    "schema_version": "ace.phantom.native-bts-correctness-lifecycle/1.0.0",
    "status": "pass",
    "mode": mode,
    "gpu_execution": gpu_status,
    "scope": "native-bootstrap-correctness-only",
    "profiling_performed": False,
    "optimization_performed": False,
    "resnet_work_performed": False,
    "compatibility_sha256": digest(input_root / "native-compatibility.json"),
    "link_audit_sha256": digest(result_root / "native-bts-link-audit.json"),
    "closed_m6_generated_executable_sha256": compatibility["bindings"]["baseline_generated_executable_sha256"],
}
(result_root / "native-bts-correctness-lifecycle.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

verify_complete() {
  local gpu_status="$1"
  local common=(
    native-bts-build.log native-bts-commands.txt native-bts-phase-timings.tsv
    native-bts-toolchain.txt native-bts-gpu.txt native-bts-executable-file.txt
    native-bts-executable-cuda-elf.txt native-bts-executable-cuda-resources.txt
    native-bts-executable-readelf.txt native-bts-ordinary-library-symbols.txt
    native-bts-oracle-library-symbols.txt native-bts-executable-symbols.txt
    native-bts-ordinary-library-members.txt native-bts-oracle-library-members.txt
    native-bts-link-audit.json native-bts-correctness-lifecycle.json
    qualification-input-payload.json qualification-input-SHA256SUMS
    qualification-input-native-compatibility.json
    qualification-input-native-clear-inputs.json
    qualification-input-native-clear-inputs.bin
    qualification-input-ace-source.manifest.json
    qualification-input-phantom-source.manifest.json
    qualification-input-baseline-fixture.json
    qualification-input-baseline-compiler-invocation.json
    qualification-input-baseline-context-manifest.json
    qualification-input-baseline-generated-phantom.json
    qualification-input-baseline-generated-phantom.bin
    qualification-input-production-source-identity.json
    qualification-input-native_phantom_bts_correctness.cu
  )
  local name
  for name in "${common[@]}"; do [[ -s "${RESULT_DIR}/${name}" ]]; done
  if [[ "${gpu_status}" == pass ]]; then
    local gpu=(
      native-bts-raw.json native-bts-raw.bin native-bts-correctness.json
      native-bts-values.bin native-bts-gpu.stdout.txt native-bts-gpu.stderr.txt
      native-bts-sentinel-raw.json native-bts-sentinel-raw.bin
      native-bts-sentinel.json native-bts-sentinel-values.bin
      native-bts-sanitizer.log native-bts-sanitizer-version.txt
      native-bts-sanitizer.stdout.txt native-bts-sanitizer.stderr.txt
      native-bts-sanitizer.json native-bts-comparison.json
    )
    for name in "${gpu[@]}"; do [[ -e "${RESULT_DIR}/${name}" ]]; done
  else
    printf '{"schema_version":"ace.phantom.native-bts-gpu-skipped/1.0.0","status":"skipped","reason":"local-environment-has-no-qualified-gpu"}\n' \
      >"${RESULT_DIR}/native-bts-gpu-skipped.json"
  fi
  python3 - "${MODE}" "${gpu_status}" "${RESULT_DIR}/result-completeness.json" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[3]).write_text(json.dumps({
    "schema_version": "ace.phantom.native-bts-result-completeness/1.0.0",
    "status": "pass", "mode": sys.argv[1], "gpu_execution": sys.argv[2],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

timed native_payload_validation verify_payload
timed native_input_archival archive_qualification_inputs
timed native_oracle_build build_native_oracle
timed native_link_audit audit_links
timed native_provenance capture_provenance
if [[ "${MODE}" == runpod ]]; then
  timed native_gpu_correctness run_native_gpu
  write_lifecycle pass
  verify_complete pass
else
  write_lifecycle skipped-local-no-gpu
  verify_complete skipped-local-no-gpu
fi
