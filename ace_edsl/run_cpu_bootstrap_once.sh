#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
APP_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
BOOTSTRAP_C="${SCRIPT_DIR}/examples/output/bootstrap_full.c"
RESNET_C="${APP_ROOT}/resnet20_cifar10_pre.c"
GENERATE=false
BUILD_DIR="${APP_ROOT}/tmp_cpu_bootstrap_once"
TARGET_LEVEL=15
MAX_ERROR=0.02
REBUILD=false
BSGS_GIANT_STEP=0
NATIVE_RTL=false

usage() {
  cat <<'EOF'
Usage: ace_edsl/run_cpu_bootstrap_once.sh [options]

Measure exactly one CPU invocation of the generated primitive DSL bootstrap.
The primitive is generated at N=65536/requested depth 30. The runtime base
context is read from the generated ResNet C so its Q chain matches the E2E run.

Options:
  --generated-c FILE  Reuse this generated bootstrap_full.c
  --resnet-c FILE     Read the base runtime profile from this generated ResNet C
  --generate          Generate a fresh ResNet-profile source in --build-dir
  --build-dir DIR     Scratch directory (default: /app/tmp_cpu_bootstrap_once)
  --target-level N    ResNet level after bootstrap (default: 15, first call)
  --bsgs-giant-step N Experimental transform giant step; requires --generate
  --max-error VALUE   All-slot identity tolerance (default: 0.02)
  --native-rtl        Time ANT's native RTL bootstrap with the same harness
  --rebuild           Recompile the generated C instead of using the object cache
  -h, --help          Show this help

Run inside ace-compiler-dev. The harness uses /usr/local/rtlib ANT only; it
does not build or execute ResNet20 and does not link Phantom/GPU libraries.
Setup, key generation, input encryption, decoding, and teardown are excluded
from elapsed_seconds. RTLIB_TIMING_OUTPUT may be set for operation counters.
EOF
}

die() {
  echo "run_cpu_bootstrap_once: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generated-c)
      [[ $# -ge 2 ]] || die "--generated-c requires a value"
      BOOTSTRAP_C="$2"
      shift 2
      ;;
    --resnet-c)
      [[ $# -ge 2 ]] || die "--resnet-c requires a value"
      RESNET_C="$2"
      shift 2
      ;;
    --generate)
      GENERATE=true
      shift
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir requires a value"
      BUILD_DIR="$2"
      shift 2
      ;;
    --target-level)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] ||
        die "--target-level requires a positive integer"
      TARGET_LEVEL="$2"
      shift 2
      ;;
    --bsgs-giant-step)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] ||
        die "--bsgs-giant-step requires a non-negative integer"
      BSGS_GIANT_STEP="$2"
      shift 2
      ;;
    --max-error)
      [[ $# -ge 2 ]] || die "--max-error requires a value"
      MAX_ERROR="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD=true
      shift
      ;;
    --native-rtl)
      NATIVE_RTL=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "${GENERATE}" == false && "${BSGS_GIANT_STEP}" != 0 ]]; then
  die "--bsgs-giant-step only applies while generating a new artifact"
fi

[[ -f /.dockerenv ]] || die "this benchmark must run inside ace-compiler-dev"
[[ "${APP_ROOT}" == /app ]] ||
  die "expected the ace-compiler workspace at /app inside ace-compiler-dev"

readonly ANT_ARCHIVE=/usr/local/rtlib/lib/libFHErt_ant.a
readonly COMMON_ARCHIVE=/usr/local/rtlib/lib/libFHErt_common.a
readonly AIRUTIL_ARCHIVE=/usr/local/lib/libAIRutil.a
for required in "${ANT_ARCHIVE}" "${COMMON_ARCHIVE}" "${AIRUTIL_ARCHIVE}"; do
  [[ -f "${required}" ]] || die "missing installed runtime dependency: ${required}"
done

if [[ "${GENERATE}" == true ]]; then
  [[ "${BOOTSTRAP_C}" == "${SCRIPT_DIR}/examples/output/bootstrap_full.c" ]] ||
    die "--generate cannot be combined with a non-standard --generated-c path"
  GENERATOR_DIR="${BUILD_DIR}/generator"
  mkdir -p "${GENERATOR_DIR}"
  cp "${SCRIPT_DIR}/examples/bootstrap_full.py" \
    "${SCRIPT_DIR}/examples/bootstrap_ant_constants.py" \
    "${SCRIPT_DIR}/examples/ant_bootstrap_ref.py" \
    "${SCRIPT_DIR}/examples/resnet_bootstrap_utils.py" \
    "${GENERATOR_DIR}/"
  GENERATOR_CONTEXT_MUL_LEVEL="${ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL:-30}"
  case "${ACE_BOOTSTRAP_LINEAR_TRANSFORM:-0}" in
    1 | true | TRUE | on | ON | yes | YES)
      GENERATOR_CONTEXT_MUL_LEVEL="${ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL:-31}"
      ;;
  esac
  (
    cd "${GENERATOR_DIR}"
    PYTHONPATH="${SCRIPT_DIR}:${APP_ROOT}" \
      ACE_BOOTSTRAP_CT_ENCODE="${ACE_BOOTSTRAP_CT_ENCODE:-1}" \
      ACE_CT_ENCODE_DEPTH="${ACE_CT_ENCODE_DEPTH:-30}" \
      ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL="${GENERATOR_CONTEXT_MUL_LEVEL}" \
      python3 resnet_bootstrap_utils.py generate-demo \
        --impl primitive \
        --poly-degree 65536 \
        --mul-level 30 \
        --bsgs-giant-step "${BSGS_GIANT_STEP}"
  )
  BOOTSTRAP_C="${GENERATOR_DIR}/output/bootstrap_full.c"
fi

[[ -f "${BOOTSTRAP_C}" ]] || die "missing generated source: ${BOOTSTRAP_C}"
[[ -f "${RESNET_C}" ]] || die "missing generated ResNet source: ${RESNET_C}"

SYMBOLS="$(python3 - "${BOOTSTRAP_C}" <<'PY'
from pathlib import Path
import re
import sys

source_path = Path(sys.argv[1])
source = source_path.read_text(encoding="utf-8")
profile = re.search(
    r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*"
    r"LIB_ANT\s*,\s*(\d+)\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)",
    source,
    re.S,
)
if profile is None:
    raise SystemExit("generated source has no readable ANT CKKS context")
degree, mul_depth, input_level = map(int, profile.groups())
if degree != 65536 or mul_depth not in (29, 30) or input_level != 1:
    raise SystemExit(
        "generated source is not the CPU ResNet profile: "
        f"observed N={degree}, mul_depth={mul_depth}, input_level={input_level}; "
        "expected N=65536, mul_depth=29 or 30, input_level=1. "
        "Pass --generate to refresh it."
    )
if re.search(r"\bEval_bootstrap_ciph\s*\(", source):
    raise SystemExit("generated primitive source calls Eval_bootstrap_ciph")
entry = re.search(r"\bCIPHERTEXT\s+([A-Za-z_][A-Za-z0-9_]*bootstrap_full)\s*\(", source)
context = re.search(r"\bCKKS_PARAMS\s*\*\s*([A-Za-z_][A-Za-z0-9_]*Get_context_params)\s*\(", source)
rtdata = re.search(r"\bRT_DATA_INFO\s*\*\s*([A-Za-z_][A-Za-z0-9_]*Get_rt_data_info)\s*\(", source)
if entry is None or context is None or rtdata is None:
    raise SystemExit("generated source is missing a bootstrap/context/data symbol")
print(f"{entry.group(1)} {context.group(1)} {rtdata.group(1)} {mul_depth}")
print(
    f"validated generated CPU ResNet profile: N={degree}, "
    f"data-Q={mul_depth + 1}, input-level={input_level}",
    file=sys.stderr,
)
PY
)"
read -r ENTRY_NAME CONTEXT_NAME RTDATA_NAME GENERATED_MUL_DEPTH <<<"${SYMBOLS}"
[[ "${CONTEXT_NAME}" != Get_context_params ]] ||
  die "generated bootstrap context must use an auxiliary symbol prefix"

BASE_PROFILE="$(python3 - "${RESNET_C}" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
profile = re.search(
    r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*LIB_ANT\s*,\s*"
    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*"
    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
    source,
    re.S,
)
if profile is None:
    raise SystemExit("generated ResNet source has no readable ANT CKKS context")
print(" ".join(profile.groups()))
PY
)"
read -r BASE_DEGREE BASE_SEC_LEVEL BASE_MUL_DEPTH BASE_INPUT_LEVEL \
  BASE_FIRST_MOD_SIZE BASE_SCALING_MOD_SIZE BASE_NUM_Q_PARTS \
  BASE_HAMMING_WEIGHT <<<"${BASE_PROFILE}"
[[ "${BASE_DEGREE}" == 65536 ]] ||
  die "generated ResNet source is not N=65536: ${RESNET_C}"
[[ "${BASE_MUL_DEPTH}" -ge 30 ]] ||
  die "generated ResNet source needs at least 31 data-Q primes: ${RESNET_C}"

RAW_AIR="$(dirname -- "${BOOTSTRAP_C}")/bootstrap_full_raw.air"
if [[ -f "${RAW_AIR}" ]] && rg -q 'CKKS\.bootstrap' "${RAW_AIR}"; then
  die "primitive raw AIR contains CKKS.bootstrap: ${RAW_AIR}"
fi

mkdir -p "${BUILD_DIR}"
readonly BOOTSTRAP_OBJ="${BUILD_DIR}/bootstrap_full.o"
readonly BOOTSTRAP_STAMP="${BUILD_DIR}/bootstrap_full.o.sha256"
readonly HARNESS_OBJ="${BUILD_DIR}/cpu_bootstrap_once_main.o"
readonly SHIM_C="${BUILD_DIR}/cpu_bootstrap_once_shim.c"
readonly SHIM_OBJ="${BUILD_DIR}/cpu_bootstrap_once_shim.o"
readonly RTL_SHIM_OBJ="${BUILD_DIR}/cpu_bootstrap_once_rtl_shim.o"
readonly BINARY="${BUILD_DIR}/cpu_bootstrap_once"
readonly LOG="${BUILD_DIR}/cpu_bootstrap_once.log"

COMMON_INCLUDES=(
  -I /usr/local/include
  -I /usr/local/rtlib/include
  -I /usr/local/rtlib/include/ant
)

# Canonicalize only the generated entry point. Its context and data symbols
# remain auxiliary so Prepare_context can merge them with the ResNet profile.
GENERATED_DEFINES=()
[[ "${NATIVE_RTL}" == true || "${ENTRY_NAME}" == bootstrap_full ]] ||
  GENERATED_DEFINES+=("-D${ENTRY_NAME}=bootstrap_full")

SHIM_OBJECTS=()
SYMBOL_PREFIX="${ENTRY_NAME%bootstrap_full}"
PT_FROM_MSG_NAME="${SYMBOL_PREFIX}Pt_from_msg"
RAISE_LEVEL_NAME="${SYMBOL_PREFIX}raise_level"
if rg -q "\\b(${PT_FROM_MSG_NAME}|${RAISE_LEVEL_NAME})\\s*\\(" \
    "${BOOTSTRAP_C}"; then
  SHIM_ARGS=(
    --output "${SHIM_C}"
    --entry-name "${ENTRY_NAME}"
    --ctxparams-name "${CONTEXT_NAME}"
    --rtdata-name "${RTDATA_NAME}"
    --pt-from-msg-name "${PT_FROM_MSG_NAME}"
    --raise-level-name "${RAISE_LEVEL_NAME}"
  )
  if [[ "${NATIVE_RTL}" == false ]]; then
    SHIM_ARGS+=(
      --target-level-setter-name ace_cpu_bootstrap_set_target_level
    )
  fi
  python3 "${SCRIPT_DIR}/examples/resnet_bootstrap_utils.py" emit-shim \
    "${SHIM_ARGS[@]}"
  gcc -c "${SHIM_C}" \
    -DRTLIB_SUPPORT_LINUX "${GENERATED_DEFINES[@]}" "${COMMON_INCLUDES[@]}" \
    -O3 -DNDEBUG -std=gnu11 -fopenmp -o "${SHIM_OBJ}"
  SHIM_OBJECTS+=("${SHIM_OBJ}")
  if [[ "${NATIVE_RTL}" == true ]]; then
    gcc -c "${SCRIPT_DIR}/tests/cpu_bootstrap_once_rtl_shim.c" \
      -DRTLIB_SUPPORT_LINUX "${COMMON_INCLUDES[@]}" \
      -O3 -DNDEBUG -std=gnu11 -fopenmp -o "${RTL_SHIM_OBJ}"
    SHIM_OBJECTS+=("${RTL_SHIM_OBJ}")
  fi
else
  die "generated source is missing the ResNet plaintext/level helper calls"
fi

BOOTSTRAP_FINGERPRINT="$({
  sha256sum "${BOOTSTRAP_C}"
  printf '%s\n' \
    'gcc -DRTLIB_SUPPORT_LINUX -O3 -DNDEBUG -std=gnu11 -fopenmp' \
    "${GENERATED_DEFINES[@]}" "${COMMON_INCLUDES[@]}"
} | sha256sum | awk '{print $1}')"
if [[ "${REBUILD}" == false && -f "${BOOTSTRAP_OBJ}" &&
      -f "${BOOTSTRAP_STAMP}" &&
      "$(<"${BOOTSTRAP_STAMP}")" == "${BOOTSTRAP_FINGERPRINT}" ]]; then
  echo "reusing generated object: ${BOOTSTRAP_OBJ}"
else
  gcc -c "${BOOTSTRAP_C}" \
    -DRTLIB_SUPPORT_LINUX "${GENERATED_DEFINES[@]}" "${COMMON_INCLUDES[@]}" \
    -O3 -DNDEBUG -std=gnu11 -fopenmp -o "${BOOTSTRAP_OBJ}"
  printf '%s\n' "${BOOTSTRAP_FINGERPRINT}" >"${BOOTSTRAP_STAMP}"
fi
c++ -c "${SCRIPT_DIR}/tests/cpu_bootstrap_once_main.cxx" \
  -DRTLIB_SUPPORT_LINUX \
  -DACE_CPU_NATIVE_RTL="$([[ "${NATIVE_RTL}" == true ]] && echo 1 || echo 0)" \
  -DACE_CPU_BASE_DEGREE="${BASE_DEGREE}" \
  -DACE_CPU_BASE_SEC_LEVEL="${BASE_SEC_LEVEL}" \
  -DACE_CPU_BASE_MUL_DEPTH="${BASE_MUL_DEPTH}" \
  -DACE_CPU_BASE_INPUT_LEVEL="${BASE_INPUT_LEVEL}" \
  -DACE_CPU_BASE_FIRST_MOD_SIZE="${BASE_FIRST_MOD_SIZE}" \
  -DACE_CPU_BASE_SCALING_MOD_SIZE="${BASE_SCALING_MOD_SIZE}" \
  -DACE_CPU_BASE_NUM_Q_PARTS="${BASE_NUM_Q_PARTS}" \
  -DACE_CPU_BASE_HAMMING_WEIGHT="${BASE_HAMMING_WEIGHT}" \
  "${COMMON_INCLUDES[@]}" -O3 -DNDEBUG -std=gnu++17 -fopenmp \
  -o "${HARNESS_OBJ}"
c++ "${BOOTSTRAP_OBJ}" "${HARNESS_OBJ}" "${SHIM_OBJECTS[@]}" \
  "${ANT_ARCHIVE}" "${COMMON_ARCHIVE}" "${AIRUTIL_ARCHIVE}" \
  -lgmp -lm -lgomp -o "${BINARY}"

ACE_CPU_BOOTSTRAP_TARGET_LEVEL="${TARGET_LEVEL}" \
ACE_CPU_BOOTSTRAP_MAX_ERROR="${MAX_ERROR}" \
RTLIB_TIMING_OUTPUT="${RTLIB_TIMING_OUTPUT:-${BUILD_DIR}/rtlib-timing.txt}" \
  "${BINARY}" | tee "${LOG}"

python3 - "${LOG}" "${TARGET_LEVEL}" "${BASE_MUL_DEPTH}" \
  "${BASE_INPUT_LEVEL}" "${GENERATED_MUL_DEPTH}" "${NATIVE_RTL}" <<'PY'
import json
from pathlib import Path
import sys

prefix = "CPU_BOOTSTRAP_RESULT="
records = [
    json.loads(line[len(prefix):])
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.startswith(prefix)
]
if len(records) != 1:
    raise SystemExit(f"expected one benchmark record, observed {len(records)}")
record = records[0]
expected = {
    "status": "pass",
    "provider": "ant",
    "implementation": "native_rtl" if sys.argv[6] == "true" else "dsl_linear_transform",
    "generated_dsl_invocations": 0 if sys.argv[6] == "true" else 1,
    "polynomial_degree": 65536,
    "logical_slots": 32768,
    "generated_mul_depth": int(sys.argv[5]),
    "runtime_mul_depth": int(sys.argv[3]),
    "data_q_count": int(sys.argv[3]) + 1,
    "runtime_input_level": int(sys.argv[4]),
    "input_level": 1,
    "target_level": int(sys.argv[2]),
    "validation_slot_count": 32768,
    "timing_scope": "bootstrap_full_only",
}
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(f"benchmark {key}={record.get(key)!r}; expected {value!r}")
print(f"one CPU bootstrap: {record['elapsed_seconds']:.6f} seconds")
PY
