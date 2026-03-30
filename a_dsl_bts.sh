#!/bin/bash
set -euo pipefail
set -x

APP_ROOT=/app
ACE_EDSL_DIR="${APP_ROOT}/ace_edsl"
DATASET_DIR="${APP_ROOT}/fhe-cmplr/rtlib/ant/dataset"
LOG_FILE="${APP_ROOT}/tmp_dsl_bts_resnet20/a_dsl_bts.log"

RESNET_GEN_C="${APP_ROOT}/resnet20_cifar10_pre.c"
RESNET_DATASET_INC="${DATASET_DIR}/resnet20_cifar10_pre.onnx.inc"
RESNET_DATASET_SRC="${DATASET_DIR}/resnet20_cifar10.cxx"

BOOTSTRAP_GEN_C="${ACE_EDSL_DIR}/examples/output/bootstrap_full.c"
BOOTSTRAP_RAW_AIR="${ACE_EDSL_DIR}/examples/output/bootstrap_full_raw.air"
BOOTSTRAP_UTILS_PY="${ACE_EDSL_DIR}/examples/resnet_bootstrap_utils.py"
WORK_DIR="${APP_ROOT}/tmp_dsl_bts_resnet20"
BOOTSTRAP_BODY_C="${WORK_DIR}/bootstrap_full_body.c"
BOOTSTRAP_SHIM_C="${WORK_DIR}/dsl_bootstrap_shim.c"
OUTPUT_BIN="${APP_ROOT}/resnet20_cifar10_pre.dsl_bts.ace"
BACKUP_RESNET_INC="${WORK_DIR}/resnet20_cifar10_pre.onnx.inc.orig"

mkdir -p "${WORK_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -f "${RESNET_GEN_C}" ]]; then
  echo "missing generated resnet source: ${RESNET_GEN_C}" >&2
  exit 1
fi

(
  cd "${ACE_EDSL_DIR}/examples"
PYTHONPATH="${ACE_EDSL_DIR}:${APP_ROOT}" \
ACE_CT_ENCODE_DEPTH=30 \
python3 "${BOOTSTRAP_UTILS_PY}" generate-demo \
  --impl primitive \
  --poly-degree 65536 \
  --mul-level 30
)

if [[ ! -f "${BOOTSTRAP_GEN_C}" ]]; then
  echo "failed to generate bootstrap source: ${BOOTSTRAP_GEN_C}" >&2
  exit 1
fi

if ! grep -q 'LIB_ANT, 65536,' "${BOOTSTRAP_GEN_C}"; then
  echo "bootstrap_full.c was not generated for poly_degree=65536" >&2
  exit 1
fi

if grep -q 'Eval_bootstrap_ciph(' "${BOOTSTRAP_GEN_C}"; then
  echo "bootstrap_full.c still lowers to Eval_bootstrap_ciph(...); expected primitive decomposition" >&2
  exit 1
fi

if [[ -f "${BOOTSTRAP_RAW_AIR}" ]] && grep -q 'CKKS.bootstrap' "${BOOTSTRAP_RAW_AIR}"; then
  echo "bootstrap_full_raw.air still contains CKKS.bootstrap; expected primitive decomposition" >&2
  exit 1
fi

if [[ -f "${RESNET_DATASET_INC}" ]]; then
  cp "${RESNET_DATASET_INC}" "${BACKUP_RESNET_INC}"
fi

cleanup() {
  if [[ -f "${BACKUP_RESNET_INC}" ]]; then
    cp "${BACKUP_RESNET_INC}" "${RESNET_DATASET_INC}"
  fi
}
trap cleanup EXIT

cp "${RESNET_GEN_C}" "${RESNET_DATASET_INC}"

python3 "${BOOTSTRAP_UTILS_PY}" patch-resnet-context \
  --bootstrap-c "${BOOTSTRAP_GEN_C}" \
  --resnet-inc "${RESNET_DATASET_INC}"

python3 "${BOOTSTRAP_UTILS_PY}" emit-body \
  --bootstrap-c "${BOOTSTRAP_GEN_C}" \
  --output "${BOOTSTRAP_BODY_C}"

python3 "${BOOTSTRAP_UTILS_PY}" emit-shim \
  --output "${BOOTSTRAP_SHIM_C}"

COMMON_FLAGS=(
  -DRTLIB_SUPPORT_LINUX
  -I /usr/local/include
  -I /usr/local/rtlib/include/
  -I /usr/local/rtlib/include/ant/
  -O3 -DNDEBUG -std=gnu++17 -fopenmp
)

C_FLAGS=(
  -DRTLIB_SUPPORT_LINUX
  -I /usr/local/include
  -I /usr/local/rtlib/include/
  -I /usr/local/rtlib/include/ant/
  -O3 -DNDEBUG -std=gnu11 -fopenmp
)

RESNET_OBJ="${WORK_DIR}/resnet20_cifar10.o"
BOOTSTRAP_OBJ="${WORK_DIR}/bootstrap_full_body.o"
SHIM_OBJ="${WORK_DIR}/dsl_bootstrap_shim.o"

c++ -c \
  -DEval_bootstrap_ciph=Eval_bootstrap_ciph_dsl \
  "${RESNET_DATASET_SRC}" \
  "${COMMON_FLAGS[@]}" \
  -o "${RESNET_OBJ}"

gcc -c \
  "${BOOTSTRAP_BODY_C}" \
  "${C_FLAGS[@]}" \
  -o "${BOOTSTRAP_OBJ}"

gcc -c \
  "${BOOTSTRAP_SHIM_C}" \
  "${C_FLAGS[@]}" \
  -o "${SHIM_OBJ}"

c++ \
  "${RESNET_OBJ}" \
  "${BOOTSTRAP_OBJ}" \
  "${SHIM_OBJ}" \
  /usr/local/rtlib/lib/libFHErt_ant.a \
  /usr/local/rtlib/lib/libFHErt_common.a \
  /usr/local/lib/libAIRutil.a \
  -lgmp -lm -o "${OUTPUT_BIN}" -lgomp

time env RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 \
  "${OUTPUT_BIN}" /app/dataset/test_batch.bin 0 0
