#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
LOCK_FILE="${SCRIPT_DIR}/configs/dependencies.env"
STATE_ROOT="${REPO_ROOT}/build/phantom_gpu"
DEPENDENCY_ROOT="${STATE_ROOT}/dependencies"
INSTALL_ROOT="${STATE_ROOT}/install-cuda-sm80"
ACE_BUILD="${STATE_ROOT}/ace-cuda-sm80"
BINDINGS_BUILD="${STATE_ROOT}/bindings-cuda-sm80"
PHANTOM_MOUNT="/deps/phantom-ant"
MODELS_MOUNT="/inputs/models"
DATASET_MOUNT="/inputs/dataset"
CUDA_ROOT="/usr/local/cuda"
NVCC="${CUDA_ROOT}/bin/nvcc"
CUOBJDUMP="${CUDA_ROOT}/bin/cuobjdump"
BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-2}"
GATE=""

usage() {
  echo "usage: $0 --gate <toolchain|ckks2c|all>"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate)
      GATE="$2"
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
  toolchain|ckks2c|all)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

set -a
source "${LOCK_FILE}"
set +a

PINNED_SOURCE="${DEPENDENCY_ROOT}/phantom-ant-${PHANTOM_COMMIT}"
PHANTOM_BUILD="${STATE_ROOT}/phantom-${PHANTOM_COMMIT}-sm80"
PHANTOM_ARCHIVE="${PHANTOM_BUILD}/lib/libphantom.a"
TOOLCHAIN_RESULTS="${STATE_ROOT}/compile_only_results/toolchain"
CKKS2C_RESULTS="${STATE_ROOT}/compile_only_results/ckks2c"

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
  for command in cmake c++ git ninja python3 rg file readelf nm jq; do
    command -v "${command}" >/dev/null
  done
  test -x "${NVCC}"
  test -x "${CUOBJDUMP}"
  test -r "${CUDA_ROOT}/include/cuda_runtime.h"
  test -r "${CUDA_ROOT}/lib64/libcudadevrt.a"
  test -r "${CUDA_ROOT}/lib64/libcudart.so"
  require_readonly_mount "${PHANTOM_MOUNT}"
  require_readonly_mount "${MODELS_MOUNT}"
  require_readonly_mount "${DATASET_MOUNT}"
}

prepare_pinned_source() {
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
  mkdir -p "${result_dir}"
  local arguments=(
    --repo-root "${REPO_ROOT}"
    --models-dir "${MODELS_MOUNT}"
    --dataset-dir "${DATASET_MOUNT}"
    --phantom-dir "${PHANTOM_MOUNT}"
    --json-output "${result_dir}/configuration.json"
  )
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
  cmake --build "${PHANTOM_BUILD}" --target phantom --parallel "${BUILD_JOBS}"
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

run_toolchain_gate() {
  require_environment
  record_common_configuration "${TOOLCHAIN_RESULTS}"
  prepare_pinned_source
  configure_phantom

  local source="${SCRIPT_DIR}/harness/minimal_phantom.cu"
  local object="${TOOLCHAIN_RESULTS}/minimal_phantom.o"
  local device_link="${TOOLCHAIN_RESULTS}/minimal_phantom.dlink.o"
  local binary="${TOOLCHAIN_RESULTS}/minimal_phantom_sm80"
  local nvcc_common=(-std=c++17 -arch=sm_80 -rdc=true)

  local compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${PINNED_SOURCE}/include"
    "${source}"
    -o "${object}"
  )
  "${NVCC}" "${compile_arguments[@]}"
  nm -C "${object}" >"${TOOLCHAIN_RESULTS}/object_symbols.txt"
  rg ' U phantom::arith::CoeffModulus::Create' "${TOOLCHAIN_RESULTS}/object_symbols.txt"

  local device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${object}"
    "${PHANTOM_ARCHIVE}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${device_link}"
  )
  "${NVCC}" "${device_link_arguments[@]}"

  local host_link_arguments=(
    -std=c++17
    "${object}"
    "${device_link}"
    -Wl,--start-group
    "${PHANTOM_ARCHIVE}"
    -lntl
    -lgmpxx
    -lgmp
    -Wl,--end-group
    -L"${CUDA_ROOT}/lib64"
    -Wl,-rpath,"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -lcudart
    -pthread
    -ldl
    -lrt
    -lm
    -o "${binary}"
  )
  c++ "${host_link_arguments[@]}"

  inspect_binary "${binary}" "${TOOLCHAIN_RESULTS}"
  nm -A -C --defined-only "${PHANTOM_ARCHIVE}" >"${TOOLCHAIN_RESULTS}/phantom_archive_symbols.txt"
  rg 'phantom::arith::CoeffModulus::Create' "${TOOLCHAIN_RESULTS}/phantom_archive_symbols.txt"

  local binary_sha
  local archive_sha
  binary_sha="$(sha256sum "${binary}" | awk '{print $1}')"
  archive_sha="$(sha256sum "${PHANTOM_ARCHIVE}" | awk '{print $1}')"
  local report_arguments=(
    -n
    --arg status pass
    --arg gate toolchain
    --arg architecture sm_80
    --arg phantom_commit "${PHANTOM_COMMIT}"
    --arg phantom_archive_sha256 "${archive_sha}"
    --arg binary_sha256 "${binary_sha}"
  )
  jq "${report_arguments[@]}" '{
      status: $status,
      gate: $gate,
      architecture: $architecture,
      phantom_commit: $phantom_commit,
      phantom_archive_sha256: $phantom_archive_sha256,
      binary_sha256: $binary_sha256,
      executable_was_run: false
    }' >"${TOOLCHAIN_RESULTS}/qualification.json"
  echo "toolchain compile, device-link, host-link, and static inspection passed"
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
  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ace_bindings"
  local codegen_tests=(
    ace_edsl/tests/test_air_pass_pipeline.py
    ace_edsl/tests/test_ckks2c_codegen.py
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

link_generated_probe() {
  local source="${CKKS2C_RESULTS}/add_mul_rotate.cu"
  local object="${CKKS2C_RESULTS}/add_mul_rotate.o"
  local main_object="${CKKS2C_RESULTS}/generated_link_main.o"
  local device_link="${CKKS2C_RESULTS}/add_mul_rotate.dlink.o"
  local binary="${CKKS2C_RESULTS}/add_mul_rotate_sm80"
  local runtime_build="${ACE_BUILD}/rtlib/build"
  local adapter_archive="${runtime_build}/phantom/libFHErt_phantom.a"
  local common_archive="${runtime_build}/common/libFHErt_common.a"
  local external_source="${runtime_build}/external/src/phantom_external"
  local external_archive="${runtime_build}/external/src/phantom_external-build/lib/libphantom.a"
  local nvcc_common=(-std=c++17 -arch=sm_80 -rdc=true)

  test -s "${adapter_archive}"
  test -s "${common_archive}"
  test -s "${external_archive}"
  test "$(git -C "${external_source}" rev-parse HEAD)" = "${PHANTOM_COMMIT}"

  python3 "${SCRIPT_DIR}/generate_ckks2c_probe.py" --output "${source}"
  python3 "${SCRIPT_DIR}/check_primitive_codegen.py" "${source}" --report "${CKKS2C_RESULTS}/source_audit.json"

  local compile_arguments=(
    "${nvcc_common[@]}"
    -dc
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
    -I"${external_source}/include"
    "${source}"
    -o "${object}"
  )
  "${NVCC}" "${compile_arguments[@]}"
  c++ -std=c++17 \
    -I"${REPO_ROOT}/fhe-cmplr/rtlib/include" \
    -c "${SCRIPT_DIR}/harness/generated_link_main.cc" \
    -o "${main_object}"

  local device_link_arguments=(
    "${nvcc_common[@]}"
    -dlink
    "${object}"
    "${adapter_archive}"
    "${external_archive}"
    "${common_archive}"
    -L"${CUDA_ROOT}/lib64"
    -lcudadevrt
    -o "${device_link}"
  )
  "${NVCC}" "${device_link_arguments[@]}"

  local host_link_arguments=(
    -std=c++17
    "${main_object}"
    "${object}"
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
  c++ "${host_link_arguments[@]}"

  inspect_binary "${binary}" "${CKKS2C_RESULTS}"
  nm -A -C --defined-only "${adapter_archive}" >"${CKKS2C_RESULTS}/adapter_archive_symbols.txt"
  local archive_contains_native_bootstrap=false
  if rg 'Bootstrapper|Phantom_bootstrap|Eval_bootstrap|bootstrap_3' "${CKKS2C_RESULTS}/adapter_archive_symbols.txt" >"${CKKS2C_RESULTS}/native_bootstrap_symbols.txt"; then
    archive_contains_native_bootstrap=true
  fi

  local source_sha
  local binary_sha
  source_sha="$(sha256sum "${source}" | awk '{print $1}')"
  binary_sha="$(sha256sum "${binary}" | awk '{print $1}')"
  local report_arguments=(
    -n
    --arg status pass
    --arg gate ckks2c
    --arg architecture sm_80
    --arg phantom_commit "${PHANTOM_COMMIT}"
    --arg source_sha256 "${source_sha}"
    --arg binary_sha256 "${binary_sha}"
    --argjson production_archive_contains_native_bootstrap "${archive_contains_native_bootstrap}"
  )
  jq "${report_arguments[@]}" '{
      status: $status,
      gate: $gate,
      architecture: $architecture,
      phantom_commit: $phantom_commit,
      source_sha256: $source_sha256,
      binary_sha256: $binary_sha256,
      production_archive_contains_native_bootstrap:
        $production_archive_contains_native_bootstrap,
      production_archive_split_pending: true,
      executable_was_run: false
    }' >"${CKKS2C_RESULTS}/qualification.json"
  echo "generated CKKS2C source compile, device-link, host-link, and inspection passed"
}

run_ckks2c_gate() {
  run_toolchain_gate
  mkdir -p "${CKKS2C_RESULTS}"
  record_common_configuration "${CKKS2C_RESULTS}"
  configure_ace
  configure_bindings
  run_compiler_tests
  link_generated_probe
}

case "${GATE}" in
  toolchain)
    run_toolchain_gate
    ;;
  ckks2c|all)
    run_ckks2c_gate
    ;;
esac
