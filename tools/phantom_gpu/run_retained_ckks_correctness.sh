#!/usr/bin/env bash
set -euo pipefail
umask 022
export LC_ALL=C
export TZ=UTC

# Deterministic host qualification for the retained CKKS provider surface.
# The Phantom executable built here is evidence for a later GPU run.  This
# script must never execute it (or any other CUDA executable).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${ACE_PHANTOM_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)}"
LOCK_FILE="${SCRIPT_DIR}/configs/dependencies.env"
CUDA_ROOT="/usr/local/cuda"
NVCC="${CUDA_ROOT}/bin/nvcc"
CUOBJDUMP="${CUDA_ROOT}/bin/cuobjdump"
SOURCE_MODE="${ACE_PHANTOM_SOURCE_MODE:-snapshot}"
STATE_ROOT="${ACE_PHANTOM_STATE_ROOT:-${REPO_ROOT}/build/phantom_gpu}"
RESULT_ROOT="${ACE_RETAINED_CKKS_RESULT_ROOT:-${STATE_ROOT}/retained_ckks_host_qualification}"
WORK_ROOT="${ACE_RETAINED_CKKS_WORK_ROOT:-${STATE_ROOT}/retained_ckks_host_work}"
PHANTOM_SOURCE="${ACE_PHANTOM_SOURCE_DIR:-/deps/phantom-ant}"
BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-2}"

# These are compiler CLI arguments, not a second parameter profile.  The
# compiler-emitted context manifest is checked against them by the retained
# source generator and is the only downstream parameter authority.
POLYNOMIAL_DEGREE=16384
MUL_LEVEL=26
INPUT_LEVEL=1
SECURITY_LEVEL=0
SCALING_MODULUS_BITS=56
FIRST_MODULUS_BITS=60
HAMMING_WEIGHT=192

usage() {
  echo "usage: $0 --host-qualify" >&2
}

if [[ $# -ne 1 || "$1" != "--host-qualify" ]]; then
  usage
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${LOCK_FILE}"
set +a

fail() {
  echo "retained CKKS host qualification: $*" >&2
  exit 1
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

record_command() {
  printf '%q ' "$@" >>"${RESULT_ROOT}/build/link-commands.txt"
  printf '\n' >>"${RESULT_ROOT}/build/link-commands.txt"
}

run_recorded() {
  record_command "$@"
  "$@"
}

write_json() {
  local output="$1"
  shift
  python3 - "${output}" "$@" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
items = sys.argv[2:]
if len(items) % 2:
    raise SystemExit("write_json received an odd key/value list")
value = dict(zip(items[0::2], items[1::2]))
output.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

require_environment() {
  [[ -f /.dockerenv ]] || fail "--host-qualify must run in the pinned container"
  [[ "${ACE_PHANTOM_TOOLCHAIN:-}" == "12.4.1-sm80" ]] ||
    fail "ACE_PHANTOM_TOOLCHAIN must be 12.4.1-sm80"
  [[ "${CMAKE_CUDA_ARCHITECTURES:-}" == "${CUDA_ARCHITECTURES}" &&
     "${CUDA_ARCHITECTURES}" == 80 ]] ||
    fail "the pinned build architecture must be exactly sm_80"
  [[ "${ACE_RUNPOD_BASE_IMAGE:-}" == "${CUDA_IMAGE}" ]] ||
    fail "the running container image reference differs from the lock"
  [[ "${ACE_RUNPOD_BASE_CONFIG_DIGEST:-}" == "${CUDA_IMAGE_CONFIG}" ]] ||
    fail "the running container config digest differs from the lock"
  [[ "${ACE_RUNPOD_BOOTSTRAP_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "the pinned bootstrap hash is absent"

  local command
  for command in ar awk c++ cmake ctest file flock nm ninja python3 \
    readelf rg sha256sum; do
    command -v "${command}" >/dev/null || fail "required command is absent: ${command}"
  done
  [[ -x "${NVCC}" && -x "${CUOBJDUMP}" ]] ||
    fail "the pinned CUDA compiler or inspection tool is absent"
  [[ -r "${CUDA_ROOT}/lib64/libcudadevrt.a" &&
     -r "${CUDA_ROOT}/lib64/libcudart.so" ]] ||
    fail "the pinned CUDA link libraries are absent"
  [[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || fail "build jobs must be positive"
  [[ "${SOURCE_MODE}" == snapshot ]] ||
    fail "retained host qualification requires audited source-only snapshots"
}

verify_json_manifest_identity() {
  python3 - "$1" "$2" "$3" <<'PY'
import json
from pathlib import Path
import re
import sys

path, kind, expected = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate source-manifest key: {key}")
        result[key] = value
    return result

value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
if not isinstance(value, dict) or value.get("kind") != kind:
    raise SystemExit(f"source manifest does not identify {kind}")
commit = value.get("commit")
if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit(f"{kind} source manifest has an invalid commit")
if commit != expected:
    raise SystemExit(f"{kind} source manifest commit differs from the selected commit")
PY
}

verify_extracted_source_manifest() {
  python3 - "$1" "$2" "$3" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

root, manifest_path, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
prefix = f"{kind}-source"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for member in manifest["members"]:
    logical = PurePosixPath(member["path"])
    if not logical.parts or logical.parts[0] != prefix:
        raise SystemExit(f"{kind} manifest member escaped its archive prefix")
    relative = PurePosixPath(*logical.parts[1:])
    physical = root.joinpath(*relative.parts)
    member_type = member.get("type")
    if member_type == "file":
        if not physical.is_file() or physical.is_symlink():
            raise SystemExit(f"{kind} snapshot file is absent: {relative}")
        observed = hashlib.sha256(physical.read_bytes()).hexdigest()
        if observed != member.get("sha256"):
            raise SystemExit(f"{kind} snapshot file hash differs: {relative}")
    elif member_type == "directory":
        if not physical.is_dir() or physical.is_symlink():
            raise SystemExit(f"{kind} snapshot directory is absent: {relative}")
    elif member_type == "symlink":
        if not physical.is_symlink() or str(physical.readlink()) != member.get("link_target"):
            raise SystemExit(f"{kind} snapshot symlink differs: {relative}")
    else:
        raise SystemExit(f"{kind} manifest has an unsupported member type")
PY
}

require_clean_sources() {
  [[ "${ACE_PHANTOM_ACE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] ||
    fail "snapshot mode requires ACE_PHANTOM_ACE_COMMIT"
  ACE_COMMIT="${ACE_PHANTOM_ACE_COMMIT}"
  ACE_SOURCE_MANIFEST="${ACE_PHANTOM_SOURCE_MANIFEST:-}"
  PHANTOM_SOURCE_MANIFEST="${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST:-}"
  [[ -f "${ACE_SOURCE_MANIFEST}" && -f "${PHANTOM_SOURCE_MANIFEST}" ]] ||
    fail "snapshot source manifests are absent"
  [[ "$(sha256_of "${ACE_SOURCE_MANIFEST}")" == "${ACE_PHANTOM_SOURCE_MANIFEST_SHA256:-}" ]] ||
    fail "ACE snapshot manifest hash differs from its audited binding"
  [[ "$(sha256_of "${PHANTOM_SOURCE_MANIFEST}")" == "${ACE_PHANTOM_PROVIDER_SOURCE_MANIFEST_SHA256:-}" ]] ||
    fail "Phantom snapshot manifest hash differs from its audited binding"
  verify_json_manifest_identity "${ACE_SOURCE_MANIFEST}" ace "${ACE_COMMIT}"
  verify_json_manifest_identity \
    "${PHANTOM_SOURCE_MANIFEST}" phantom "${PHANTOM_COMMIT}"
  verify_extracted_source_manifest \
    "${REPO_ROOT}" "${ACE_SOURCE_MANIFEST}" ace
  verify_extracted_source_manifest \
    "${PHANTOM_SOURCE}" "${PHANTOM_SOURCE_MANIFEST}" phantom
  SOURCE_DATE_EPOCH="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["commit_timestamp"])' \
    "${ACE_SOURCE_MANIFEST}")"
  [[ ! -e "${REPO_ROOT}/.git" && ! -e "${PHANTOM_SOURCE}/.git" ]] ||
    fail "source-only snapshot mode forbids embedded Git metadata"
  [[ -f "${PHANTOM_SOURCE}/CMakeLists.txt" &&
     -d "${PHANTOM_SOURCE}/include" && -d "${PHANTOM_SOURCE}/src" ]] ||
    fail "the exact Phantom source snapshot is incomplete"
  [[ "${SOURCE_DATE_EPOCH}" =~ ^[1-9][0-9]*$ ]] ||
    fail "the ACE source timestamp is invalid"
  export SOURCE_DATE_EPOCH
}

prepare_roots() {
  [[ ! -e "${RESULT_ROOT}" ]] || fail "result root already exists: ${RESULT_ROOT}"
  [[ ! -e "${WORK_ROOT}" ]] || fail "work root already exists: ${WORK_ROOT}"
  mkdir -p "${RESULT_ROOT}"/{inputs,outputs,build/inspection,tests,source} \
    "${WORK_ROOT}"
  chmod 0755 "${RESULT_ROOT}" "${WORK_ROOT}"
  : >"${RESULT_ROOT}/build/link-commands.txt"
  cp -- "${ACE_SOURCE_MANIFEST}" "${RESULT_ROOT}/source/ace_source_manifest.json"
  cp -- "${PHANTOM_SOURCE_MANIFEST}" \
    "${RESULT_ROOT}/source/phantom_source_manifest.json"
}

configure_and_build() {
  ACE_BUILD="${WORK_ROOT}/ace-cuda-sm80"
  INSTALL_ROOT="${WORK_ROOT}/install-cuda-sm80"
  BINDINGS_BUILD="${WORK_ROOT}/bindings-cuda-sm80"
  local -a configure=(
    -S "${REPO_ROOT}/fhe-cmplr" -B "${ACE_BUILD}" -G Ninja
    "-DFHE_WITH_SRC=air-infra;nn-addon"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}"
    -DCMAKE_CUDA_ARCHITECTURES=80
    -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_STANDARD_REQUIRED=ON
    -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DFHE_ENABLE_CUDA=ON -DFHE_ENABLE_PHANTOM=ON
    "-DPHANTOM_SOURCE_DIR=${PHANTOM_SOURCE}"
    "-DPHANTOM_GIT_TAG=${PHANTOM_COMMIT}"
    -DFHE_ENABLE_SEAL=OFF -DFHE_ENABLE_SEAL_BTS=OFF
    -DFHE_ENABLE_OPENFHE=OFF -DBUILD_UNITTEST=ON -DBUILD_BENCH=OFF
    -DFHE_BUILD_TEST=ON -DFHE_BUILD_EXAMPLE=OFF
    -DFHE_CODE_CHECK=OFF -DAIR_CODE_CHECK=OFF -DNN_CODE_CHECK=OFF
  )
  if [[ "${SOURCE_MODE}" == snapshot ]]; then
    configure+=("-DACE_SOURCE_COMMIT=${ACE_COMMIT}" -DPHANTOM_SOURCE_SNAPSHOT=ON)
  fi
  record_command cmake "${configure[@]}"
  cmake "${configure[@]}" >"${RESULT_ROOT}/build/ace-configure.log" 2>&1
  record_command cmake --build "${ACE_BUILD}" --target install --parallel "${BUILD_JOBS}"
  cmake --build "${ACE_BUILD}" --target install --parallel "${BUILD_JOBS}" \
    >"${RESULT_ROOT}/build/ace-build.log" 2>&1

  local onnx_proto_dir="${ACE_BUILD}/nn-addon/onnx2air"
  if [[ ! -f "${onnx_proto_dir}/onnx.pb.h" &&
        -f "${onnx_proto_dir}/onnx/onnx.pb.h" ]]; then
    onnx_proto_dir="${onnx_proto_dir}/onnx"
  fi
  local -a binding_configure=(
    -S "${REPO_ROOT}/bindings" -B "${BINDINGS_BUILD}" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    "-DACE_COMPILER_DIR=${REPO_ROOT}"
    "-DACE_INSTALL_DIR=${INSTALL_ROOT}"
    "-DONNX_PROTO_DIR=${onnx_proto_dir}"
    "-Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
  )
  record_command cmake "${binding_configure[@]}"
  cmake "${binding_configure[@]}" \
    >"${RESULT_ROOT}/build/bindings-configure.log" 2>&1
  record_command cmake --build "${BINDINGS_BUILD}" --target \
    air_builder nn_addon fhe_cmplr passmanager --parallel "${BUILD_JOBS}"
  cmake --build "${BINDINGS_BUILD}" --target \
    air_builder nn_addon fhe_cmplr passmanager --parallel "${BUILD_JOBS}" \
    >"${RESULT_ROOT}/build/bindings-build.log" 2>&1

  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ace_bindings${PYTHONPATH:+:${PYTHONPATH}}"
}

run_host_tests() {
  local -a python_tests=(
    ace_edsl/tests/test_phantom_ckks2c_extended_ops.py
    tools/phantom_gpu/tests/test_retained_ckks_oracles.py
    tools/phantom_gpu/tests/test_retained_generation_contracts.py
    tools/phantom_gpu/tests/test_retained_host_qualification.py
    tools/phantom_gpu/tests/test_retained_phantom_conformance_source.py
    tools/phantom_gpu/tests/test_retained_raise_oracle_algebra.py
  )
  (
    cd "${REPO_ROOT}"
    python3 -m pytest -q "${python_tests[@]}"
  ) >"${RESULT_ROOT}/tests/pytest-retained-host.txt" 2>&1
  ctest --test-dir "${ACE_BUILD}" --output-on-failure \
    -R '^(ut_fhert_ant_ckks_extended_ops|ut_fheckks)$' \
    >"${RESULT_ROOT}/tests/ctest-retained-host.txt" 2>&1
  rg -F '100% tests passed' "${RESULT_ROOT}/tests/ctest-retained-host.txt" >/dev/null
}

generate_context_authority() {
  local -a arguments=(
    --output "${RESULT_ROOT}/source/context_authority_probe.cu"
    --context-manifest "${RESULT_ROOT}/inputs/compiler_context_manifest.json"
    --resource-manifest "${RESULT_ROOT}/source/context_authority_resources.json"
    --post-ckks-air "${RESULT_ROOT}/source/context_authority_post.air"
    --poly-degree "${POLYNOMIAL_DEGREE}"
    --mul-level "${MUL_LEVEL}"
    --input-level "${INPUT_LEVEL}"
    --security-level "${SECURITY_LEVEL}"
    --scaling-factor-bits "${SCALING_MODULUS_BITS}"
    --first-prime-bits "${FIRST_MODULUS_BITS}"
    --hamming-weight "${HAMMING_WEIGHT}"
  )
  python3 "${SCRIPT_DIR}/generate_ckks2c_probe.py" "${arguments[@]}" \
    >"${RESULT_ROOT}/build/context-authority-generation.log" 2>&1
}

generate_retained_artifacts() {
  cp -- "${SCRIPT_DIR}/fixtures/retained_ckks_v1.json" \
    "${RESULT_ROOT}/inputs/retained_ckks_fixture_template.json"

  python3 "${SCRIPT_DIR}/generate_retained_ckks_sources.py" \
    --context-manifest "${RESULT_ROOT}/inputs/compiler_context_manifest.json" \
    --fixture "${RESULT_ROOT}/inputs/retained_ckks_fixture_template.json" \
    --ant-source "${RESULT_ROOT}/outputs/retained_ckks_ant.cxx" \
    --phantom-source "${RESULT_ROOT}/outputs/retained_ckks_phantom.cu" \
    --ant-post-ckks-air "${RESULT_ROOT}/outputs/retained_ckks_ant_post.air" \
    --phantom-post-ckks-air "${RESULT_ROOT}/outputs/retained_ckks_phantom_post.air" \
    --phantom-context-manifest "${RESULT_ROOT}/outputs/compiler_context_manifest.json" \
    --phantom-resource-manifest "${RESULT_ROOT}/outputs/compiler_resource_manifest.json" \
    --generation-record "${RESULT_ROOT}/outputs/retained_ckks_generation.json" \
    --interface-header "${RESULT_ROOT}/outputs/retained_ckks_generated_interface.h" \
    --polynomial-degree "${POLYNOMIAL_DEGREE}" \
    --mul-level "${MUL_LEVEL}" \
    --input-level "${INPUT_LEVEL}" \
    --security-level "${SECURITY_LEVEL}" \
    --scaling-modulus-bits "${SCALING_MODULUS_BITS}" \
    --first-modulus-bits "${FIRST_MODULUS_BITS}" \
    --hamming-weight "${HAMMING_WEIGHT}" \
    >"${RESULT_ROOT}/build/retained-source-generation.log" 2>&1

  cmp "${RESULT_ROOT}/inputs/compiler_context_manifest.json" \
    "${RESULT_ROOT}/outputs/compiler_context_manifest.json"
  cmp "${RESULT_ROOT}/outputs/retained_ckks_ant_post.air" \
    "${RESULT_ROOT}/outputs/retained_ckks_phantom_post.air"

  python3 "${SCRIPT_DIR}/generate_retained_production_air.py" \
    --output-air "${RESULT_ROOT}/outputs/retained_ckks_production_post.air" \
    --output-record "${RESULT_ROOT}/outputs/retained_ckks_production_rnums.json" \
    >"${RESULT_ROOT}/build/production-rnum-generation.log" 2>&1

  local fixture_tool="${SCRIPT_DIR}/generate_retained_ckks_fixtures.py"
  local common=(
    --context-manifest "${RESULT_ROOT}/inputs/compiler_context_manifest.json"
    --compiler-invocation "${RESULT_ROOT}/outputs/retained_ckks_generation.json"
    --post-ckks-air "${RESULT_ROOT}/outputs/retained_ckks_phantom_post.air"
    --production-post-ckks-air "${RESULT_ROOT}/outputs/retained_ckks_production_post.air"
  )
  python3 "${fixture_tool}" bind-fixture \
    --fixture "${RESULT_ROOT}/inputs/retained_ckks_fixture_template.json" \
    "${common[@]}" \
    --output-json "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    >"${RESULT_ROOT}/build/fixture-binding.log"
  python3 "${fixture_tool}" validate-fixture \
    --fixture "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    "${common[@]}" >"${RESULT_ROOT}/build/fixture-validation.log"
  python3 "${fixture_tool}" generate-analytic \
    --fixture "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    "${common[@]}" \
    --output-json "${RESULT_ROOT}/outputs/retained_ckks_analytic_reference.json" \
    --output-bin "${RESULT_ROOT}/outputs/retained_ckks_analytic_values.bin" \
    >"${RESULT_ROOT}/build/analytic-generation.log"
  python3 "${fixture_tool}" generate-exact \
    --fixture "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    "${common[@]}" \
    --output-json "${RESULT_ROOT}/outputs/retained_ckks_exact_source.json" \
    --output-bin "${RESULT_ROOT}/outputs/retained_ckks_exact_source.bin" \
    >"${RESULT_ROOT}/build/exact-source-generation.log"
}

locate_archives() {
  local runtime_build="${ACE_BUILD}/rtlib/build"
  ADAPTER_ARCHIVE="${runtime_build}/phantom/libFHErt_phantom.a"
  COMMON_ARCHIVE="${runtime_build}/common/libFHErt_common.a"
  ANT_ARCHIVE="${runtime_build}/ant/libFHErt_ant.a"
  ANT_ENCODE_ARCHIVE="${runtime_build}/ant/libFHErt_ant_encode.a"
  PROVIDER_ARCHIVE="${runtime_build}/external/src/phantom_external-build/lib/libphantom_ordinary.a"
  local archive
  for archive in "${ADAPTER_ARCHIVE}" "${COMMON_ARCHIVE}" "${ANT_ARCHIVE}" \
    "${ANT_ENCODE_ARCHIVE}" "${PROVIDER_ARCHIVE}"; do
    [[ -s "${archive}" ]] || fail "required runtime archive is absent: ${archive}"
  done
}

build_and_run_ant_oracle() {
  local ant_generated_object="${RESULT_ROOT}/build/retained_ckks_ant.generated.o"
  local ant_harness_object="${RESULT_ROOT}/build/retained_ckks_ant.harness.o"
  ANT_BINARY="${RESULT_ROOT}/build/retained_ckks_ant_oracle"
  local -a includes=(
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/include"
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/ant/include"
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/ant/util/include"
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/ant/hal/include"
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/ant/poly/include"
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/ant/ckks/include"
    "-I${RESULT_ROOT}/outputs"
  )
  run_recorded c++ -std=c++17 "${includes[@]}" \
    -c "${RESULT_ROOT}/outputs/retained_ckks_ant.cxx" \
    -o "${ant_generated_object}"
  run_recorded c++ -std=c++17 "${includes[@]}" \
    "-DACE_COMMIT_ID=\"${ACE_COMMIT}\"" \
    "-DPHANTOM_COMMIT_ID=\"${PHANTOM_COMMIT}\"" \
    -c "${SCRIPT_DIR}/harness/retained_ckks_ant_oracle.cxx" \
    -o "${ant_harness_object}"
  run_recorded c++ -std=c++17 "${ant_generated_object}" \
    "${ant_harness_object}" -Wl,--start-group "${ANT_ARCHIVE}" \
    "${ANT_ENCODE_ARCHIVE}" "${COMMON_ARCHIVE}" -lntl -lgmpxx -lgmp \
    -Wl,--end-group -pthread -fopenmp -ldl -lrt -lm -o "${ANT_BINARY}"

  # This is the only generated conformance executable run by host qualification.
  RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 "${ANT_BINARY}" \
    "${RESULT_ROOT}/inputs/retained_ckks_fixture.json" \
    "${RESULT_ROOT}/inputs/compiler_context_manifest.json" \
    "${RESULT_ROOT}/outputs/compiler_resource_manifest.json" \
    "${RESULT_ROOT}/outputs/retained_ckks_analytic_reference.json" \
    "${RESULT_ROOT}/outputs/retained_ckks_analytic_values.bin" \
    "${RESULT_ROOT}/outputs/retained_ckks_cpu_reference.json" \
    "${RESULT_ROOT}/outputs/retained_ckks_cpu_values.bin" \
    disable-native-bootstrap-precompute \
    >"${RESULT_ROOT}/build/retained-ant.stdout.txt" \
    2>"${RESULT_ROOT}/build/retained-ant.stderr.txt"
}

build_phantom_without_running() {
  local generated_object="${RESULT_ROOT}/build/retained_ckks_phantom.generated.o"
  local harness_object="${RESULT_ROOT}/build/retained_ckks_phantom.harness.o"
  local device_link="${RESULT_ROOT}/build/retained_ckks_phantom.dlink.o"
  PHANTOM_BINARY="${RESULT_ROOT}/build/retained_ckks_phantom_sm80"
  local -a nvcc_common=(-std=c++17 -arch=sm_80 -rdc=true)
  local -a includes=(
    "-I${REPO_ROOT}/fhe-cmplr/rtlib/include"
    "-I${PHANTOM_SOURCE}/include"
    "-I${RESULT_ROOT}/outputs"
  )
  run_recorded "${NVCC}" "${nvcc_common[@]}" -dc "${includes[@]}" \
    "${RESULT_ROOT}/outputs/retained_ckks_phantom.cu" -o "${generated_object}"
  run_recorded "${NVCC}" "${nvcc_common[@]}" -dc "${includes[@]}" \
    "-DACE_COMMIT_ID=\"${ACE_COMMIT}\"" \
    "-DPHANTOM_COMMIT_ID=\"${PHANTOM_COMMIT}\"" \
    "${SCRIPT_DIR}/harness/retained_ckks_phantom_conformance.cu" \
    -o "${harness_object}"
  run_recorded "${NVCC}" "${nvcc_common[@]}" -dlink \
    "${generated_object}" "${harness_object}" "${ADAPTER_ARCHIVE}" \
    "${PROVIDER_ARCHIVE}" "${COMMON_ARCHIVE}" \
    "-L${CUDA_ROOT}/lib64" -lcudadevrt -o "${device_link}"
  run_recorded c++ -std=c++17 "${generated_object}" "${harness_object}" \
    "${device_link}" -Wl,--start-group "${ADAPTER_ARCHIVE}" \
    "${PROVIDER_ARCHIVE}" "${COMMON_ARCHIVE}" -lntl -lgmpxx -lgmp \
    -Wl,--end-group "-L${CUDA_ROOT}/lib64" \
    "-Wl,-rpath,${CUDA_ROOT}/lib64" -lcudadevrt -lcudart \
    -pthread -fopenmp -ldl -lrt -lm -o "${PHANTOM_BINARY}"
}

inspect_build() {
  local inspection="${RESULT_ROOT}/build/inspection"
  local name archive
  for entry in \
    "adapter:${ADAPTER_ARCHIVE}" "provider:${PROVIDER_ARCHIVE}" \
    "common:${COMMON_ARCHIVE}" "ant:${ANT_ARCHIVE}" \
    "ant_encode:${ANT_ENCODE_ARCHIVE}"; do
    name="${entry%%:*}"
    archive="${entry#*:}"
    ar t "${archive}" >"${inspection}/${name}-archive-members.txt"
    nm -A -C --defined-only "${archive}" >"${inspection}/${name}-archive-symbols.txt"
  done
  nm -A -C --undefined-only "${PHANTOM_BINARY}" \
    >"${inspection}/phantom-undefined-symbols.txt"
  nm -A -C --defined-only "${PHANTOM_BINARY}" \
    >"${inspection}/phantom-defined-symbols.txt"
  nm -A -C --undefined-only "${ANT_BINARY}" \
    >"${inspection}/ant-undefined-symbols.txt"
  "${CUOBJDUMP}" --list-elf "${PHANTOM_BINARY}" \
    >"${inspection}/phantom-cuda-elf.txt"
  "${CUOBJDUMP}" --dump-resource-usage "${PHANTOM_BINARY}" \
    >"${inspection}/phantom-cuda-resources.txt"
  file "${PHANTOM_BINARY}" >"${inspection}/phantom-file.txt"
  readelf -h -S -Ws -d "${PHANTOM_BINARY}" >"${inspection}/phantom-readelf.txt"
  rg -F 'sm_80' "${inspection}/phantom-cuda-elf.txt" >/dev/null
  rg -F 'retained_ckks_composite' "${inspection}/phantom-defined-symbols.txt" >/dev/null
  if rg 'Conjugate_ciph|Rotate_batch_ciph|Raise_mod|Mul_mono_ciph|retained_ckks_' \
      "${inspection}/phantom-undefined-symbols.txt" \
      >"${inspection}/forbidden-retained-undefined-symbols.txt"; then
    fail "the retained Phantom binary has unresolved retained symbols"
  fi
  if rg 'FHErt_(ant|poly)|fhe::(ant|poly)|Eval_bootstrap|Phantom_bootstrap|Bootstrapper' \
      "${inspection}/phantom-defined-symbols.txt" \
      >"${inspection}/forbidden-phantom-binary-symbols.txt"; then
    fail "the retained Phantom binary contains excluded provider code"
  fi
}

write_attestations() {
  python3 - "${RESULT_ROOT}" "${REPO_ROOT}" "${PHANTOM_SOURCE}" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" "${SOURCE_MODE}" \
    "${ADAPTER_ARCHIVE}" "${PROVIDER_ARCHIVE}" "${COMMON_ARCHIVE}" \
    "${ANT_ARCHIVE}" "${ANT_ENCODE_ARCHIVE}" "${ANT_BINARY}" \
    "${PHANTOM_BINARY}" "${CUDA_IMAGE}" "${CUDA_IMAGE_CONFIG}" \
    "${ACE_RUNPOD_BOOTSTRAP_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(root, repo, phantom, ace_commit, phantom_commit, source_mode,
 adapter, provider, common, ant, ant_encode, ant_binary, phantom_binary,
 image, image_config, bootstrap_sha) = sys.argv[1:]
root = Path(root)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

context = root / "inputs/compiler_context_manifest.json"
resource = root / "outputs/compiler_resource_manifest.json"
fixture = root / "inputs/retained_ckks_fixture.json"
generation = root / "outputs/retained_ckks_generation.json"
build = {
    "schema_version": "ace.phantom.retained_ckks.build-attestation/1.0.0",
    "status": "pass",
    "architecture": "sm_80",
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "source_mode": source_mode,
    "ace_source_manifest_sha256": digest(
        root / "source/ace_source_manifest.json"
    ),
    "phantom_source_manifest_sha256": digest(
        root / "source/phantom_source_manifest.json"
    ),
    "compiler_context_manifest_sha256": digest(context),
    "compiler_resource_manifest_sha256": digest(resource),
    "fixture_sha256": digest(fixture),
    "compiler_invocation_sha256": digest(generation),
    "generated_ant_source_sha256": digest(
        root / "outputs/retained_ckks_ant.cxx"
    ),
    "generated_phantom_source_sha256": digest(
        root / "outputs/retained_ckks_phantom.cu"
    ),
    "archives": {
        "adapter": digest(adapter), "provider": digest(provider),
        "common": digest(common), "ant": digest(ant),
        "ant_encode": digest(ant_encode),
    },
    "executables": {
        "ant_oracle": digest(ant_binary),
        "phantom_sm80": digest(phantom_binary),
    },
    "link_commands_sha256": digest(root / "build/link-commands.txt"),
    "container": {
        "image": image, "config_digest": image_config,
        "bootstrap_sha256": bootstrap_sha,
    },
    "link_mode": "explicit_compile-device-link-host-link",
    "archive_inspection": "pass",
    "undefined_symbol_inspection": "pass",
    "cubin_architecture_inspection": "pass",
    "host_tests": "pass",
    "host_ant_oracle_was_run": True,
    "gpu_executables_were_run": False,
}
build_path = root / "build_attestation.json"
write(build_path, build)

excluded = {"artifact_manifest.json", "manifest.json", "SHA256SUMS"}
files = {
    path.relative_to(root).as_posix(): digest(path)
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.relative_to(root).as_posix() not in excluded
}
required = {
    "inputs/compiler_context_manifest.json",
    "inputs/retained_ckks_fixture_template.json",
    "inputs/retained_ckks_fixture.json",
    "outputs/compiler_context_manifest.json",
    "outputs/compiler_resource_manifest.json",
    "outputs/retained_ckks_ant.cxx",
    "outputs/retained_ckks_phantom.cu",
    "outputs/retained_ckks_ant_post.air",
    "outputs/retained_ckks_phantom_post.air",
    "outputs/retained_ckks_production_post.air",
    "outputs/retained_ckks_production_rnums.json",
    "outputs/retained_ckks_generation.json",
    "outputs/retained_ckks_analytic_reference.json",
    "outputs/retained_ckks_analytic_values.bin",
    "outputs/retained_ckks_exact_source.json",
    "outputs/retained_ckks_exact_source.bin",
    "outputs/retained_ckks_cpu_reference.json",
    "outputs/retained_ckks_cpu_values.bin",
    "build/retained_ckks_ant_oracle",
    "build/retained_ckks_phantom_sm80",
    "build_attestation.json",
}
missing = sorted(required - files.keys())
if missing:
    raise SystemExit("retained artifact manifest inputs missing: " + ", ".join(missing))
artifact = {
    "schema_version": "ace.phantom.retained_ckks.artifact-manifest/1.0.0",
    "status": "bound",
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "ace_source_manifest_sha256": digest(
        root / "source/ace_source_manifest.json"
    ),
    "phantom_source_manifest_sha256": digest(
        root / "source/phantom_source_manifest.json"
    ),
    "compiler_context_manifest_sha256": digest(context),
    "compiler_resource_manifest_sha256": digest(resource),
    "normalized_compiler_command_sha256": json.loads(
        generation.read_text(encoding="utf-8")
    )["normalized_argv_sha256"],
    "post_ckks_air_sha256": digest(
        root / "outputs/retained_ckks_phantom_post.air"
    ),
    "production_post_ckks_air_sha256": digest(
        root / "outputs/retained_ckks_production_post.air"
    ),
    "fixture_sha256": digest(fixture),
    "build_attestation_sha256": digest(build_path),
    "files": files,
}
write(root / "artifact_manifest.json", artifact)
PY

  (
    cd "${RESULT_ROOT}"
    find . -type f ! -name manifest.json ! -name SHA256SUMS -print0 |
      LC_ALL=C sort -z | xargs -0 -r sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  )

  local source_manifest_name source_manifest_sha
  source_manifest_name="source/ace_source_manifest.json"
  source_manifest_sha="$(sha256_of "${RESULT_ROOT}/${source_manifest_name}")"
  python3 - "${RESULT_ROOT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" \
    "${SOURCE_MODE}" "${source_manifest_name}" "${source_manifest_sha}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
ace_commit, phantom_commit, source_mode, source_manifest, source_sha = sys.argv[2:]
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
value = {
    "schema_version": "ace.phantom.retained_ckks.host-qualification/1.0.0",
    "status": "pass",
    "gate": "retained_ckks",
    "source_mode": source_mode,
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "ace_worktree_dirty": False,
    "ace_source_manifest": source_manifest,
    "ace_source_manifest_sha256": source_sha,
    "phantom_source_manifest": "source/phantom_source_manifest.json",
    "phantom_source_manifest_sha256": digest(
        root / "source/phantom_source_manifest.json"
    ),
    "compiler_context_manifest_sha256": digest(
        root / "inputs/compiler_context_manifest.json"
    ),
    "compiler_resource_manifest_sha256": digest(
        root / "outputs/compiler_resource_manifest.json"
    ),
    "fixture_sha256": digest(root / "inputs/retained_ckks_fixture.json"),
    "cpu_reference_sha256": digest(
        root / "outputs/retained_ckks_cpu_reference.json"
    ),
    "cpu_values_sha256": digest(root / "outputs/retained_ckks_cpu_values.bin"),
    "build_attestation_sha256": digest(root / "build_attestation.json"),
    "artifact_manifest_sha256": digest(root / "artifact_manifest.json"),
    "evidence_sha256_manifest": "SHA256SUMS",
    "evidence_sha256_manifest_sha256": digest(root / "SHA256SUMS"),
    "host_ant_oracle_was_run": True,
    "gpu_executables_were_run": False,
}
(root / "manifest.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  sha256sum "${RESULT_ROOT}/manifest.json" >"${RESULT_ROOT}.manifest.sha256"
}

require_environment
require_clean_sources
prepare_roots
exec 9>"${STATE_ROOT}/retained_ckks_host_qualification.lock"
flock -n 9 || fail "another retained CKKS host qualification is running"
configure_and_build
run_host_tests
generate_context_authority
generate_retained_artifacts
locate_archives
build_and_run_ant_oracle
build_phantom_without_running
inspect_build
write_attestations

echo "retained CKKS host qualification passed: ${RESULT_ROOT}"
echo "GPU executables were compiled and inspected but not run"
