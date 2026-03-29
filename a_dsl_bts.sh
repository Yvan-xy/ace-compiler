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
  ACE_BOOTSTRAP_IMPL=primitive \
  ACE_BOOTSTRAP_POLY_DEGREE=65536 \
  ACE_BOOTSTRAP_MUL_LEVEL=30 \
  PYTHONPATH="${ACE_EDSL_DIR}:${APP_ROOT}" \
  python3 - <<'PY'
import importlib
import bootstrap_full

importlib.reload(bootstrap_full)
ok = bootstrap_full.run_demo()
raise SystemExit(0 if ok else 1)
PY
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

# Extend the resnet runtime context with any additional rotation keys needed by
# the generated DSL bootstrap body.
python3 - "${BOOTSTRAP_GEN_C}" "${RESNET_DATASET_INC}" <<'PY'
import re
import sys
from pathlib import Path

bootstrap_path = Path(sys.argv[1])
resnet_inc_path = Path(sys.argv[2])

bootstrap_c = bootstrap_path.read_text(encoding="utf-8")
resnet_inc = resnet_inc_path.read_text(encoding="utf-8")

rot_idxs = set()
for pattern in (
    r"\bRotate\s*\([^,]+,\s*(-?\d+)\s*\)",
    r"\bRotate_ciph\s*\([^,]+,\s*[^,]+,\s*(-?\d+)\s*\)",
):
    for m in re.finditer(pattern, bootstrap_c):
        rot_idxs.add(int(m.group(1)))

deg_match = re.search(
    r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*LIB_ANT\s*,\s*(\d+)",
    bootstrap_c,
    re.S,
)
if deg_match:
    ring_degree = int(deg_match.group(1))
    if "Conjugate_ciph(" in bootstrap_c:
        rot_idxs.add(2 * ring_degree - 1)

ctx_pat = re.compile(
    r"(static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*"
    r"LIB_ANT\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*"
    r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*)"
    r"(\d+)\s*,\s*\n\s*\{\s*([^}]*)\s*\}",
    re.S,
)
m = ctx_pat.search(resnet_inc)
if not m:
    raise SystemExit("failed to locate CKKS_PARAMS in resnet include")

existing = []
for tok in re.split(r"[,\s]+", m.group(3).strip()):
    if tok:
        existing.append(int(tok))

all_rot_idxs = sorted({v for v in existing if v != 0} | {v for v in rot_idxs if v != 0})
rot_list = ", ".join(str(v) for v in all_rot_idxs)
patched = ctx_pat.sub(
    lambda mm: f"{mm.group(1)}{len(all_rot_idxs)}, \n    {{ {rot_list} }}",
    resnet_inc,
    count=1,
)

rotate_init = "  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);\n"
rotate_fast_path = (
    rotate_init
    + "  if (rot_idx_1 == 0) {\n"
    + "    Copy_ciphertext(&_pgen_rot_res_2, &ciph_0);\n"
    + "    RTLIB_TM_END(20, rtm);\n"
    + "    return _pgen_rot_res_2;\n"
    + "  }\n"
)
patched = patched.replace(rotate_init, rotate_fast_path, 1)

resnet_inc_path.write_text(patched, encoding="utf-8")
PY

# Strip the standalone wrapper globals from bootstrap_full.c so it can be
# linked next to the resnet-generated translation unit, which already provides
# Get_context_params()/Get_rt_data_info().
python3 - "${BOOTSTRAP_GEN_C}" "${BOOTSTRAP_BODY_C}" <<'PY'
import sys
import re

src, dst = sys.argv[1:3]
skip = False
brace_depth = 0
out = []

def starts_wrapper(line: str) -> bool:
    return (
        line.startswith("CKKS_PARAMS* Get_context_params()")
        or line.startswith("RT_DATA_INFO* Get_rt_data_info()")
    )

with open(src, "r", encoding="utf-8") as f:
    for line in f:
        if not skip and starts_wrapper(line):
            skip = True
            brace_depth = line.count("{") - line.count("}")
            continue
        if skip:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                skip = False
            continue
        line = re.sub(r"\bbootstrap_full\b", "dsl_bootstrap_full", line)
        line = re.sub(r"\bRotate\b", "dsl_bts_Rotate", line)
        line = re.sub(r"\bRelinearize\b", "dsl_bts_Relinearize", line)
        line = re.sub(r"\b(_cst_\d+)\b", r"dsl_bts_\1", line)
        out.append(line)

body = "".join(out)
rotate_init = "  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);\n"
rotate_fast_path = (
    rotate_init
    + "  if (rot_idx_1 == 0) {\n"
    + "    Copy_ciphertext(&_pgen_rot_res_2, &ciph_0);\n"
    + "    RTLIB_TM_END(20, rtm);\n"
    + "    return _pgen_rot_res_2;\n"
    + "  }\n"
)
body = body.replace(rotate_init, rotate_fast_path, 1)

with open(dst, "w", encoding="utf-8") as f:
    f.write(body)
PY

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
