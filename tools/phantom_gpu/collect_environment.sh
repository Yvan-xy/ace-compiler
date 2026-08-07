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
  echo "base_image=${ACE_RUNPOD_BASE_IMAGE:-unset}"
  echo "base_config_digest=${ACE_RUNPOD_BASE_CONFIG_DIGEST:-unset}"
  echo "bootstrap_sha256=${ACE_RUNPOD_BOOTSTRAP_SHA256:-unset}"
  echo "source_mode=${ACE_PHANTOM_SOURCE_MODE:-git}"
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
  echo "[gcc]"
  gcc --version
  echo
  echo "[g++]"
  g++ --version
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
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "[driver]"
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
      --format=csv,noheader
  fi
} >"${OUTPUT}"

echo "recorded host toolchain at ${OUTPUT}"
