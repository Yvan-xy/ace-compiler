#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"

usage() {
  echo "usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --ordinary-run-root DIR --poly-degree N --mul-level Q --input-level L --security-level B --scaling-factor-bits B --first-prime-bits B --hamming-weight W OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
POLY_DEGREE=""
MUL_LEVEL=""
INPUT_LEVEL=""
SECURITY_LEVEL=""
SCALING_BITS=""
FIRST_PRIME_BITS=""
HAMMING_WEIGHT=""
ORDINARY_RUN_ROOT=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ace-commit)
      [[ $# -ge 2 ]] || usage
      ACE_COMMIT="$2"
      shift 2
      ;;
    --phantom-commit)
      [[ $# -ge 2 ]] || usage
      PHANTOM_COMMIT="$2"
      shift 2
      ;;
    --ordinary-run-root)
      [[ $# -ge 2 ]] || usage
      ORDINARY_RUN_ROOT="$2"
      shift 2
      ;;
    --poly-degree) [[ $# -ge 2 ]] || usage; POLY_DEGREE="$2"; shift 2 ;;
    --mul-level) [[ $# -ge 2 ]] || usage; MUL_LEVEL="$2"; shift 2 ;;
    --input-level) [[ $# -ge 2 ]] || usage; INPUT_LEVEL="$2"; shift 2 ;;
    --security-level) [[ $# -ge 2 ]] || usage; SECURITY_LEVEL="$2"; shift 2 ;;
    --scaling-factor-bits) [[ $# -ge 2 ]] || usage; SCALING_BITS="$2"; shift 2 ;;
    --first-prime-bits) [[ $# -ge 2 ]] || usage; FIRST_PRIME_BITS="$2"; shift 2 ;;
    --hamming-weight) [[ $# -ge 2 ]] || usage; HAMMING_WEIGHT="$2"; shift 2 ;;
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
[[ -n "${ACE_COMMIT}" && -n "${PHANTOM_COMMIT}" &&
   -n "${ORDINARY_RUN_ROOT}" &&
   -n "${OUTPUT_ARGUMENT}" ]] || usage
for value in "${POLY_DEGREE}" "${MUL_LEVEL}" "${INPUT_LEVEL}" \
  "${SECURITY_LEVEL}" "${SCALING_BITS}" "${FIRST_PRIME_BITS}" \
  "${HAMMING_WEIGHT}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || usage
done
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commits must be full lowercase 40-character object IDs" >&2
  exit 1
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"

ORDINARY_RUN_ROOT="$(realpath -- "${ORDINARY_RUN_ROOT}")"
FROZEN_CONTEXT="${ORDINARY_RUN_ROOT}/ckks2c/compiler_context_manifest.json"
FROZEN_RESOURCES="${ORDINARY_RUN_ROOT}/ckks2c/compiler_resource_manifest.json"
FROZEN_FIXTURE="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_v1.json"
FROZEN_CPU_REFERENCE="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_cpu_reference.json"
FROZEN_CPU_VALUES="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_cpu_values.bin"
FROZEN_ANT_VERIFICATION="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_ant_verification.json"
for required in \
  "${FROZEN_CONTEXT}" "${FROZEN_RESOURCES}" "${FROZEN_FIXTURE}" \
  "${FROZEN_CPU_REFERENCE}" "${FROZEN_CPU_VALUES}" \
  "${FROZEN_ANT_VERIFICATION}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing frozen ordinary-CKKS evidence: ${required}" >&2
    exit 1
  fi
done
EXPECTED_DATA_Q_COUNT="$((MUL_LEVEL + 1))"
python3 - "${FROZEN_CONTEXT}" "${POLY_DEGREE}" \
  "${EXPECTED_DATA_Q_COUNT}" "${INPUT_LEVEL}" "${SECURITY_LEVEL}" \
  "${SCALING_BITS}" "${FIRST_PRIME_BITS}" "${HAMMING_WEIGHT}" <<'PY'
import json
from pathlib import Path
import sys

context = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "polynomial_degree": int(sys.argv[2]),
    "data_q_count": int(sys.argv[3]),
    "input_level": int(sys.argv[4]),
    "security_level": int(sys.argv[5]),
    "scaling_modulus_bits": int(sys.argv[6]),
    "first_modulus_bits": int(sys.argv[7]),
    "hamming_weight": int(sys.argv[8]),
}
observed = {
    "polynomial_degree": context.get("polynomial_degree"),
    "data_q_count": len(context.get("data_q_bit_sizes", [])),
    "input_level": context.get("input_level"),
    "security_level": context.get("security_level"),
    "scaling_modulus_bits": context.get("scaling_modulus_bits"),
    "first_modulus_bits": context.get("first_modulus_bits"),
    "hamming_weight": context.get("hamming_weight"),
}
if observed != expected:
    raise SystemExit(
        f"frozen compiler context does not match requested options: "
        f"expected={expected}, observed={observed}"
    )
PY

DEPENDENCY_LOCK="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
)"
lock_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" <<<"${DEPENDENCY_LOCK}")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "selected ACE commit has an invalid ${key} dependency lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
LOCKED_PHANTOM_COMMIT="$(lock_value PHANTOM_COMMIT)"
if [[ "${PHANTOM_COMMIT}" != "${LOCKED_PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit does not match the selected ACE commit lock" >&2
  echo "requested: ${PHANTOM_COMMIT}" >&2
  echo "locked:    ${LOCKED_PHANTOM_COMMIT}" >&2
  exit 1
fi

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"

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
  --repo "${REPO_ROOT}" \
  --commit "${ACE_COMMIT}" \
  --kind ace \
  --output "${OUTPUT}/${ACE_ARCHIVE}" \
  --manifest "${OUTPUT}/ace-source.manifest.json" \
  >"${OUTPUT}/ace-source.audit.json"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" \
  --commit "${PHANTOM_COMMIT}" \
  --kind phantom \
  --output "${OUTPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${OUTPUT}/phantom-source.manifest.json" \
  >"${OUTPUT}/phantom-source.audit.json"

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" >"${OUTPUT}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/run_build_and_health.sh
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 \
  "${OUTPUT}/source_archive.py" \
  "${OUTPUT}/phase_helpers.sh" \
  "${OUTPUT}/bootstrap_environment.sh" \
  "${OUTPUT}/run_build_and_health.sh"

cp "${FROZEN_CONTEXT}" "${OUTPUT}/ordinary-context-manifest.json"
cp "${FROZEN_RESOURCES}" "${OUTPUT}/ordinary-resource-manifest.json"
cp "${FROZEN_FIXTURE}" "${OUTPUT}/ordinary-fixture.json"
cp "${FROZEN_CPU_REFERENCE}" "${OUTPUT}/ordinary-cpu-reference.json"
cp "${FROZEN_CPU_VALUES}" "${OUTPUT}/ordinary-cpu-values.bin"
cp "${FROZEN_ANT_VERIFICATION}" "${OUTPUT}/ordinary-ant-verification.json"

(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

python3 - "${OUTPUT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" \
  "${POLY_DEGREE}" "${MUL_LEVEL}" "${INPUT_LEVEL}" \
  "${SECURITY_LEVEL}" "${SCALING_BITS}" "${FIRST_PRIME_BITS}" \
  "${HAMMING_WEIGHT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "schema_version": "1.0.0",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "compiler_context_options": {
        "poly_degree": int(sys.argv[4]),
        "mul_level": int(sys.argv[5]),
        "input_level": int(sys.argv[6]),
        "security_level": int(sys.argv[7]),
        "scaling_factor_bits": int(sys.argv[8]),
        "first_prime_bits": int(sys.argv[9]),
        "hamming_weight": int(sys.argv[10]),
    },
    "frozen_ordinary_reference": {
        "context_manifest_sha256": hashlib.sha256(
            (output / "ordinary-context-manifest.json").read_bytes()
        ).hexdigest(),
        "resource_manifest_sha256": hashlib.sha256(
            (output / "ordinary-resource-manifest.json").read_bytes()
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            (output / "ordinary-fixture.json").read_bytes()
        ).hexdigest(),
        "cpu_reference_sha256": hashlib.sha256(
            (output / "ordinary-cpu-reference.json").read_bytes()
        ).hexdigest(),
        "cpu_values_sha256": hashlib.sha256(
            (output / "ordinary-cpu-values.bin").read_bytes()
        ).hexdigest(),
        "ant_verification_sha256": hashlib.sha256(
            (output / "ordinary-ant-verification.json").read_bytes()
        ).hexdigest(),
    },
    "files": sorted(path.name for path in output.iterdir() if path.is_file()),
}
(output / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
(
  cd "${OUTPUT}"
  sha256sum payload.json >>SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "audited RunPod source payload: ${OUTPUT}"
