#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT" >&2
  exit 2
fi

OUTPUT="$1"
mkdir -p "$(dirname -- "${OUTPUT}")"

{
  echo "captured_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "architecture=$(uname -m)"
  echo "kernel=$(uname -sr)"
  echo "cuda_architectures=${CMAKE_CUDA_ARCHITECTURES:-unset}"
  echo "toolchain_identity=${ACE_PHANTOM_TOOLCHAIN:-unset}"
  echo "development_image_id=${ACE_PHANTOM_IMAGE_ID:-unset}"
  echo "development_definition_sha256=${ACE_PHANTOM_DEFINITION_SHA256:-unset}"
  echo
  echo "[os-release]"
  sed -n '1,40p' /etc/os-release
  echo
  echo "[nvcc]"
  /usr/local/cuda/bin/nvcc --version
  echo
  echo "[cmake]"
  cmake --version
  echo
  echo "[c++]"
  c++ --version
  echo
  echo "[ninja]"
  ninja --version
  echo
  echo "[python]"
  python3 --version
  echo
  echo "[packages]"
  PACKAGES=(
    cmake
    g++
    libgmp-dev
    libntl-dev
    libstdc++-dev
    ninja-build
  )
  dpkg-query -W "${PACKAGES[@]}"
  echo
  echo "[all-packages]"
  dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort
  echo
  echo "[python-packages]"
  python3 -m pip freeze --all | LC_ALL=C sort
  echo
  echo "[python-check]"
  python3 -m pip check
  echo
  echo "[cuda-files]"
  CUDA_FILES=(
    /usr/local/cuda/include/cuda_runtime.h
    /usr/local/cuda/lib64/libcudadevrt.a
  )
  sha256sum "${CUDA_FILES[@]}"
  readlink -f /usr/local/cuda/lib64/libcudart.so
  echo
  echo "[cuda-libraries]"
  ldconfig -p | rg 'libcudart|libcuda'
} >"${OUTPUT}"

echo "recorded host toolchain at ${OUTPUT}"
