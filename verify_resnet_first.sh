#!/bin/bash
set -euo pipefail
set -x

APP_ROOT=/app
ACE_EDSL_DIR="${APP_ROOT}/ace_edsl"
DATASET_DIR="${APP_ROOT}/fhe-cmplr/rtlib/ant/dataset"

RESNET_GEN_C="${APP_ROOT}/resnet20_cifar10_pre.c"
RESNET_DATASET_INC="${DATASET_DIR}/resnet20_cifar10_pre.onnx.inc"
BOOTSTRAP_GEN_C="${ACE_EDSL_DIR}/examples/output/bootstrap_full.c"
BOOTSTRAP_RAW_AIR="${ACE_EDSL_DIR}/examples/output/bootstrap_full_raw.air"
BOOTSTRAP_UTILS_PY="${ACE_EDSL_DIR}/examples/resnet_bootstrap_utils.py"

WORK_DIR="${APP_ROOT}/tmp_verify_resnet_first"
BACKUP_RESNET_INC="${WORK_DIR}/resnet20_cifar10_pre.onnx.inc.orig"
BASE_DRIVER_CXX="${WORK_DIR}/verify_resnet_first_driver.cxx"
BASE_BIN="${WORK_DIR}/verify_resnet_first.base.ace"
BASE_BOOTSTRAP_SHIM_C="${WORK_DIR}/baseline_bootstrap_shim.c"
DSL_DRIVER_CXX="${WORK_DIR}/verify_resnet_first_driver.dsl.cxx"
DSL_BOOTSTRAP_BODY_C="${WORK_DIR}/bootstrap_full_body.c"
DSL_BOOTSTRAP_SHIM_C="${WORK_DIR}/dsl_bootstrap_shim.c"
DSL_BIN="${WORK_DIR}/verify_resnet_first.dsl.ace"

mkdir -p "${WORK_DIR}"

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

cat > "${BASE_DRIVER_CXX}" <<'EOF'
#include <cstdio>
#include <cstdlib>

#include "common/rtlib.h"
#include "nn/util/cifar_reader.h"

#define CIFAR_CLASS_COUNT 10
#include "/app/fhe-cmplr/rtlib/ant/dataset/resnet20_cifar10_pre.onnx.inc"

int main(int argc, char* argv[]) {
  const char* dataset = (argc > 1) ? argv[1] : "/app/dataset/test_batch.bin";
  int image_idx = (argc > 2) ? atoi(argv[2]) : 0;

  double mean[]  = {0.485, 0.456, 0.406};
  double stdev[] = {0.229, 0.224, 0.225};
  nn::util::CIFAR_READER<CIFAR_CLASS_COUNT> cifar_reader(dataset, mean, stdev);
  if (!cifar_reader.Initialize()) {
    printf("VERIFY_ERROR=init_failed\n");
    return 2;
  }

  TENSOR* input_data =
      Alloc_tensor(1, cifar_reader.Channel(), cifar_reader.Height(),
                   cifar_reader.Width(), NULL);
  int label = cifar_reader.Load(image_idx, input_data->_vals);
  if (label < 0) {
    printf("VERIFY_ERROR=load_failed\n");
    return 3;
  }

  Prepare_context();
  Prepare_input(input_data, "input");
  Free_tensor(input_data);

  Run_main_graph();

  double* result = Handle_output("output");
  printf("VERIFY_LABEL=%d\n", label);
  printf("VERIFY_RESULT=[");
  for (int i = 0; i < CIFAR_CLASS_COUNT; ++i) {
    if (i) printf(", ");
    printf("%.10f", result[i]);
  }
  printf("]\n");
  fflush(stdout);
  free(result);
  _Exit(0);
}
EOF

cp "${BASE_DRIVER_CXX}" "${DSL_DRIVER_CXX}"

cat > "${BASE_BOOTSTRAP_SHIM_C}" <<'EOF'
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ckks/cipher.h"
#include "ckks/ciphertext.h"

#ifdef __cplusplus
extern "C" {
#endif

static unsigned long g_base_bts_call_counter = 0;

static double base_bts_now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void base_bts_dump_first_output(CIPHER ciph) {
  uint32_t dump_len = Get_ciph_slots(ciph);
  if (dump_len > 8) {
    dump_len = 8;
  }
  Print_cipher_msg_with_imag(stderr, "base_bts_round1", ciph, dump_len);
  fflush(stderr);
}

static void base_bts_maybe_stop_after_first_dump(void) {
  const char* flag = getenv("ACE_STOP_AFTER_FIRST_BTS");
  if (flag != NULL && flag[0] != '\0' && strcmp(flag, "0") != 0) {
    fprintf(stderr, "[base_bts] stopping after first bootstrap dump\n");
    fflush(stderr);
    _Exit(0);
  }
}

CIPHER Eval_bootstrap_ciph_baseline_probe(CIPHER res, CIPHER ciph,
                                          uint32_t level_after_bts,
                                          uint32_t num_slots) {
  unsigned long call_id = __sync_add_and_fetch(&g_base_bts_call_counter, 1);
  double t0 = base_bts_now_sec();
  fprintf(stderr,
          "[base_bts] begin call=%lu target_level=%u num_slots=%u "
          "in_level=%zu in_sfdeg=%u in_slots=%u\n",
          call_id, level_after_bts, num_slots, Level(ciph), Sc_degree(ciph),
          Get_ciph_slots(ciph));

  CIPHER out = Eval_bootstrap_ciph(res, ciph, level_after_bts, num_slots);

  fprintf(stderr,
          "[base_bts] end   call=%lu out_level=%zu out_sfdeg=%u out_slots=%u "
          "elapsed=%.3fs\n",
          call_id, Level(out), Sc_degree(out), Get_ciph_slots(out),
          base_bts_now_sec() - t0);
  if (call_id == 1) {
    base_bts_dump_first_output(out);
    base_bts_maybe_stop_after_first_dump();
  }
  return out;
}

#ifdef __cplusplus
}
#endif
EOF

python3 "${BOOTSTRAP_UTILS_PY}" emit-body \
  --bootstrap-c "${BOOTSTRAP_GEN_C}" \
  --output "${DSL_BOOTSTRAP_BODY_C}"

python3 "${BOOTSTRAP_UTILS_PY}" emit-shim \
  --output "${DSL_BOOTSTRAP_SHIM_C}"

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

BASE_OBJ="${WORK_DIR}/verify_resnet_first.base.o"
BASE_SHIM_OBJ="${WORK_DIR}/baseline_bootstrap_shim.o"
DSL_DRIVER_OBJ="${WORK_DIR}/verify_resnet_first.dsl.o"
DSL_BOOTSTRAP_OBJ="${WORK_DIR}/bootstrap_full_body.o"
DSL_SHIM_OBJ="${WORK_DIR}/dsl_bootstrap_shim.o"

c++ -c -DEval_bootstrap_ciph=Eval_bootstrap_ciph_baseline_probe \
  "${BASE_DRIVER_CXX}" "${COMMON_FLAGS[@]}" -o "${BASE_OBJ}"
gcc -c "${BASE_BOOTSTRAP_SHIM_C}" "${C_FLAGS[@]}" -o "${BASE_SHIM_OBJ}"
c++ "${BASE_OBJ}" "${BASE_SHIM_OBJ}" \
  /usr/local/rtlib/lib/libFHErt_ant.a \
  /usr/local/rtlib/lib/libFHErt_common.a \
  /usr/local/lib/libAIRutil.a \
  -lgmp -lm -o "${BASE_BIN}" -lgomp

c++ -c -DEval_bootstrap_ciph=Eval_bootstrap_ciph_dsl \
  "${DSL_DRIVER_CXX}" \
  "${COMMON_FLAGS[@]}" \
  -o "${DSL_DRIVER_OBJ}"
gcc -c "${DSL_BOOTSTRAP_BODY_C}" "${C_FLAGS[@]}" -o "${DSL_BOOTSTRAP_OBJ}"
gcc -c "${DSL_BOOTSTRAP_SHIM_C}" "${C_FLAGS[@]}" -o "${DSL_SHIM_OBJ}"
c++ "${DSL_DRIVER_OBJ}" "${DSL_BOOTSTRAP_OBJ}" "${DSL_SHIM_OBJ}" \
  /usr/local/rtlib/lib/libFHErt_ant.a \
  /usr/local/rtlib/lib/libFHErt_common.a \
  /usr/local/lib/libAIRutil.a \
  -lgmp -lm -o "${DSL_BIN}" -lgomp

BASE_LOG="${WORK_DIR}/baseline.log"
DSL_LOG="${WORK_DIR}/dsl.log"

"${BASE_BIN}" /app/dataset/test_batch.bin 0 > "${BASE_LOG}" 2>&1
env RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 \
  "${DSL_BIN}" /app/dataset/test_batch.bin 0 > "${DSL_LOG}" 2>&1

echo "=== BASELINE FIRST BOOTSTRAP ==="
grep -E '^\[base_bts\]|^\[base_bts_round1\]' "${BASE_LOG}" || true
echo "=== DSL FIRST BOOTSTRAP ==="
grep -E '^\[dsl_bts\]|^\[dsl_bts_round1\]' "${DSL_LOG}" || true

python3 - "${BASE_LOG}" "${DSL_LOG}" <<'PY'
import ast
import sys

base_log, dsl_log = sys.argv[1:3]

def parse(path):
    label = None
    result = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("VERIFY_LABEL="):
                label = int(line.split("=", 1)[1])
            elif line.startswith("VERIFY_RESULT="):
                result = ast.literal_eval(line.split("=", 1)[1])
    if label is None or result is None:
        raise SystemExit(f"missing verification output in {path}")
    pred = max(range(len(result)), key=lambda i: result[i])
    return label, result, pred

base_label, base_result, base_pred = parse(base_log)
dsl_label, dsl_result, dsl_pred = parse(dsl_log)

print(f"BASE_LABEL={base_label}")
print(f"BASE_PRED={base_pred}")
print(f"BASE_RESULT={base_result}")
print(f"DSL_LABEL={dsl_label}")
print(f"DSL_PRED={dsl_pred}")
print(f"DSL_RESULT={dsl_result}")

if len(base_result) != len(dsl_result):
    print("MATCH=0")
    raise SystemExit(1)

abs_err = [abs(a - b) for a, b in zip(base_result, dsl_result)]
max_abs_err = max(abs_err) if abs_err else 0.0
print(f"MAX_ABS_ERR={max_abs_err:.10f}")
print("MATCH=1" if base_pred == dsl_pred and base_label == dsl_label else "MATCH=0")
PY
