#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/phase_helpers.sh"
APT_LOCK="${ACE_APT_LOCK:-${SCRIPT_DIR}/configs/apt-packages.lock}"
PYTHON_LOCK="${ACE_PYTHON_LOCK:-${SCRIPT_DIR}/configs/python-requirements-hashed.lock}"
BASE_FILES_LOCK="${ACE_BASE_FILES_LOCK:-${SCRIPT_DIR}/configs/base-files.sha256}"
DEPENDENCIES_LOCK="${ACE_DEPENDENCIES_LOCK:-${SCRIPT_DIR}/configs/dependencies.env}"
VENV="${ACE_RUNPOD_VENV:-/opt/ace-runpod-venv}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EVIDENCE_DIRECTORY" >&2
  exit 2
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "environment bootstrap must run as root" >&2
  exit 1
fi

EVIDENCE="$(realpath -m -- "$1")"
mkdir -p "${EVIDENCE}"
chmod 0755 "${EVIDENCE}"
TIMINGS="${EVIDENCE}/bootstrap-timings.tsv"
: >"${TIMINGS}"

phase() {
  run_timed_phase "${TIMINGS}" "$@"
}

verify_base() {
  local expected_image
  expected_image="$(sed -n 's/^CUDA_IMAGE=//p' "${DEPENDENCIES_LOCK}")"
  if [[ "${ACE_RUNPOD_BASE_IMAGE:-}" != "${expected_image}" ]]; then
    echo "requested base image identity is absent or does not match the lock" >&2
    return 1
  fi
  if [[ "${ACE_RUNPOD_BASE_CONFIG_DIGEST:-}" != \
        "sha256:0131784115794405cb36a8068a82d7aea0937196d1d6e844b9dd021252ccf7e4" ]]; then
    echo "base image config digest is absent or does not match the lock" >&2
    return 1
  fi
  sha256sum -c "${BASE_FILES_LOCK}"
}

install_apt() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  local specifications=()
  while IFS='=' read -r package version; do
    [[ -n "${package}" && "${package}" != \#* ]] || continue
    local candidate
    candidate="$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2}')"
    if [[ "${candidate}" != "${version}" ]]; then
      echo "APT candidate for ${package} is ${candidate}, expected ${version}" >&2
      return 1
    fi
    specifications+=("${package}=${version}")
  done <"${APT_LOCK}"
  apt-get install -y --no-install-recommends "${specifications[@]}"
}

install_python() {
  if [[ ! -x "${VENV}/bin/python3" ]]; then
    python3 -m venv "${VENV}"
  fi
  "${VENV}/bin/python3" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --only-binary=:all: \
    --require-hashes \
    --requirement "${PYTHON_LOCK}"
  "${VENV}/bin/python3" -m pip check
}

verify_toolchain() {
  export PATH="${VENV}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  [[ "$(nvcc --version | sed -n 's/.* V\([0-9.]\+\)$/\1/p')" == "12.4.131" ]]
  [[ "$(cmake --version | awk 'NR == 1 {print $3}')" == "3.22.1" ]]
  [[ "$(g++ -dumpfullversion -dumpversion)" == "11.4.0" ]]
  [[ "$(gcc -dumpfullversion -dumpversion)" == "11.4.0" ]]
  [[ "$(python3 -c 'import platform; print(platform.python_version())')" == "3.10.12" ]]
  [[ "$(dpkg-query -W -f='${Version}' libntl-dev)" == "11.5.1-1" ]]
  [[ "$(dpkg-query -W -f='${Version}' libgmp-dev)" == "2:6.2.1+dfsg-3ubuntu1" ]]
}

collect_environment() {
  export PATH="${VENV}/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  cp "${APT_LOCK}" "${EVIDENCE}/apt-packages.lock"
  cp "${PYTHON_LOCK}" "${EVIDENCE}/python-requirements-hashed.lock"
  cp "${BASE_FILES_LOCK}" "${EVIDENCE}/base-files.sha256"
  sha256sum "${APT_LOCK}" "${PYTHON_LOCK}" "${BASE_FILES_LOCK}" >"${EVIDENCE}/lock-files.sha256"
  dpkg-query -W -f='${binary:Package}=${Version}\n' | LC_ALL=C sort >"${EVIDENCE}/dpkg-manifest.txt"
  python3 -m pip freeze --all | LC_ALL=C sort >"${EVIDENCE}/python-manifest.txt"
  python3 -m pip check >"${EVIDENCE}/python-check.txt"
  {
    find /var/lib/apt/lists -maxdepth 1 -type f \
      \( -name '*InRelease' -o -name '*Release' \) -print0 |
      LC_ALL=C sort -z |
      xargs -0 -r sha256sum
  } >"${EVIDENCE}/apt-release-files.sha256"
  {
    for path in /etc/apt/sources.list /etc/apt/sources.list.d/*; do
      [[ -f "${path}" ]] || continue
      echo "[${path}]"
      sed -n '1,240p' "${path}"
    done
  } >"${EVIDENCE}/apt-sources.txt"
  {
    echo "base_image=${ACE_RUNPOD_BASE_IMAGE}"
    echo "base_config_digest=${ACE_RUNPOD_BASE_CONFIG_DIGEST}"
    echo "apt_reproducibility_limit=ordinary signed Ubuntu and NVIDIA repositories may stop retaining these exact versions"
    cat /etc/os-release
    nvcc --version
    gcc --version
    g++ --version
    cmake --version
    ninja --version
    python3 --version
    dpkg-query -W libntl-dev libgmp-dev
    sha256sum /usr/local/cuda/include/cuda_runtime.h /usr/local/cuda/lib64/libcudadevrt.a
  } >"${EVIDENCE}/toolchain.txt"
}

phase verify_base verify_base
phase apt_install install_apt
phase python_install install_python
phase verify_toolchain verify_toolchain
phase collect_environment collect_environment

echo "environment bootstrap passed"
