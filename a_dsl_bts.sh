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

cat > "${BOOTSTRAP_SHIM_C}" <<'EOF'
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "ckks/cipher.h"
#include "ckks/ciphertext.h"

#ifdef __cplusplus
extern "C" {
#endif

CIPHERTEXT dsl_bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1);

static unsigned long g_dsl_bts_call_counter = 0;

static double dsl_bts_now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void dsl_bts_dump_first_output(CIPHER ciph) {
  uint32_t dump_len = Get_ciph_slots(ciph);
  if (dump_len > 8) {
    dump_len = 8;
  }
  Print_cipher_msg_with_imag(stderr, "dsl_bts_round1", ciph, dump_len);
  fflush(stderr);
}

static void dsl_bts_maybe_stop_after_first_dump(void) {
  const char* flag = getenv("ACE_STOP_AFTER_FIRST_BTS");
  if (flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0) {
    fprintf(stderr, "[dsl_bts] stopping after first bootstrap dump\n");
    fflush(stderr);
    _Exit(0);
  }
}

CIPHER Eval_bootstrap_ciph_dsl(CIPHER res, CIPHER ciph,
                               uint32_t level_after_bts,
                               uint32_t num_slots) {
  unsigned long call_id = __sync_add_and_fetch(&g_dsl_bts_call_counter, 1);
  size_t in_level = Level(ciph);
  uint32_t in_sfdeg = Sc_degree(ciph);
  uint32_t in_slots = Get_ciph_slots(ciph);
  double t0 = dsl_bts_now_sec();
  fprintf(stderr,
          "[dsl_bts] begin call=%lu target_level=%u num_slots=%u "
          "in_level=%zu in_sfdeg=%u in_slots=%u\n",
          call_id, level_after_bts, num_slots, in_level, in_sfdeg, in_slots);

  CIPHERTEXT in_copy;
  memset(&in_copy, 0, sizeof(in_copy));
  Copy_ciphertext(&in_copy, ciph);

  CIPHERTEXT out = dsl_bootstrap_full(in_copy, in_copy);
  if (level_after_bts != 0) {
    while (Level(&out) > level_after_bts) {
      Modswitch_ciph(&out);
    }
  }
  double elapsed = dsl_bts_now_sec() - t0;
  fprintf(stderr,
          "[dsl_bts] end   call=%lu out_level=%zu out_sfdeg=%u out_slots=%u "
          "elapsed=%.3fs\n",
          call_id, Level(&out), Sc_degree(&out), Get_ciph_slots(&out), elapsed);
  if (call_id == 1) {
    dsl_bts_dump_first_output(&out);
    dsl_bts_maybe_stop_after_first_dump();
  }

  Free_poly_data(Get_c0(&in_copy));
  Free_poly_data(Get_c1(&in_copy));

  if (res == ciph) {
    Free_poly_data(Get_c0(res));
    Free_poly_data(Get_c1(res));
  }

  *res = out;
  return res;
}

#ifdef __cplusplus
}
#endif
EOF

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

time "${OUTPUT_BIN}" /app/dataset/test_batch.bin 0 0
