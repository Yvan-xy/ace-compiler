//-*-c-*-
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ckks/bootstrap.h"
#include "common/common.h"
#include "util/type.h"

extern VL_I32* Get_colls_fft_params(uint32_t slots, uint32_t level_budget,
                                    uint32_t dim1);
extern VL_VL_VL_DCMPLX* Coeff_collapse(VL_DCMPLX* ksipows, VL_UI32* rot_group,
                                       uint32_t level_budget, bool flag,
                                       bool encoding);

CKKS_PARAMS* Get_context_params(void) { return NULL; }
RT_DATA_INFO* Get_rt_data_info(void) { return NULL; }
int Get_input_count(void) { return 0; }
int Get_output_count(void) { return 0; }
DATA_SCHEME* Get_encode_scheme(int idx) {
  (void)idx;
  return NULL;
}
DATA_SCHEME* Get_decode_scheme(int idx) {
  (void)idx;
  return NULL;
}

static uint64_t Fnv1a64_update(uint64_t hash, const unsigned char* data,
                               size_t len) {
  const uint64_t prime = 0x100000001b3ULL;
  for (size_t i = 0; i < len; ++i) {
    hash ^= (uint64_t)data[i];
    hash *= prime;
  }
  return hash;
}

static uint64_t Digest_row(VL_DCMPLX* row) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  FOR_ALL_ELEM(row, idx) {
    double real = creal(Get_dcmplx_value_at(row, idx));
    double imag = cimag(Get_dcmplx_value_at(row, idx));
    hash = Fnv1a64_update(hash, (const unsigned char*)&real, sizeof(real));
    hash = Fnv1a64_update(hash, (const unsigned char*)&imag, sizeof(imag));
  }
  return hash;
}

static void Print_complex_pair(FILE* fp, DCMPLX val) {
  fprintf(fp, "[%.17g, %.17g]", creal(val), cimag(val));
}

static size_t Collect_sample_indices(size_t len, size_t out[5]) {
  size_t candidates[5] = {0, 1, 2, len / 2, len - 1};
  size_t count         = 0;
  for (size_t i = 0; i < 5; ++i) {
    size_t idx = candidates[i];
    if (idx >= len) continue;
    bool seen = false;
    for (size_t j = 0; j < count; ++j) {
      if (out[j] == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) out[count++] = idx;
  }
  return count;
}

static void Print_params_json(FILE* fp, VL_I32* params) {
  fprintf(fp, "{");
  fprintf(fp, "\"level_budget\": %d, ", Get_i32_value_at(params, LEVEL_BUDGET));
  fprintf(fp, "\"layers_coll\": %d, ", Get_i32_value_at(params, LAYERS_COLL));
  fprintf(fp, "\"rem_coll\": %d, ", Get_i32_value_at(params, LAYERS_REM));
  fprintf(fp, "\"flag_rem\": %d, ",
          Get_i32_value_at(params, LAYERS_REM) == 0 ? 0 : 1);
  fprintf(fp, "\"num_rot\": %d, ", Get_i32_value_at(params, NUM_ROTATIONS));
  fprintf(fp, "\"b\": %d, ", Get_i32_value_at(params, BABY_STEP));
  fprintf(fp, "\"g\": %d, ", Get_i32_value_at(params, GIANT_STEP));
  fprintf(fp, "\"num_rot_rem\": %d, ",
          Get_i32_value_at(params, NUM_ROTATIONS_REM));
  fprintf(fp, "\"b_rem\": %d, ", Get_i32_value_at(params, BABY_STEP_REM));
  fprintf(fp, "\"g_rem\": %d", Get_i32_value_at(params, GIANT_STEP_REM));
  fprintf(fp, "}");
}

static void Print_coeff_summary(FILE* fp, VL_VL_VL_DCMPLX* coeff) {
  fprintf(fp, "[");
  FOR_ALL_ELEM(coeff, stage_idx) {
    if (stage_idx) fprintf(fp, ",");
    fprintf(fp, "{\"stage\": %zu, \"rows\": [", stage_idx);
    VL_VL_DCMPLX* stage = Get_vl_value_at(coeff, stage_idx);
    FOR_ALL_ELEM(stage, row_idx) {
      if (row_idx) fprintf(fp, ",");
      VL_DCMPLX* row = Get_vl_value_at(stage, row_idx);
      uint64_t digest = Digest_row(row);
      size_t sample_idx[5];
      size_t sample_cnt = Collect_sample_indices(LIST_LEN(row), sample_idx);
      fprintf(fp, "{\"row\": %zu, \"digest\": \"%016llx\", \"samples\": {",
              row_idx, (unsigned long long)digest);
      for (size_t i = 0; i < sample_cnt; ++i) {
        if (i) fprintf(fp, ",");
        size_t idx = sample_idx[i];
        fprintf(fp, "\"%zu\": ", idx);
        Print_complex_pair(fp, Get_dcmplx_value_at(row, idx));
      }
      fprintf(fp, "}}");
    }
    fprintf(fp, "]}");
  }
  fprintf(fp, "]");
}

static VL_UI32* Build_rot_group(uint32_t slots) {
  uint32_t slots4 = 4 * slots;
  VL_UI32* rot_group = Alloc_value_list(UI32_TYPE, slots);
  uint32_t five_pows = 1;
  FOR_ALL_ELEM(rot_group, idx) {
    UI32_VALUE_AT(rot_group, idx) = five_pows;
    five_pows *= 5;
    five_pows %= slots4;
  }
  return rot_group;
}

static VL_DCMPLX* Build_ksi_pows(uint32_t slots) {
  uint32_t slots4 = 4 * slots;
  VL_DCMPLX* ksi_pows = Alloc_value_list(DCMPLX_TYPE, slots4 + 1);
  for (uint32_t idx = 0; idx < slots4; ++idx) {
    double angle = 2.0 * M_PI * idx / slots4;
    DCMPLX_VALUE_AT(ksi_pows, idx) = cos(angle) + sin(angle) * I;
  }
  DCMPLX_VALUE_AT(ksi_pows, slots4) = DCMPLX_VALUE_AT(ksi_pows, 0);
  return ksi_pows;
}

int main(int argc, char** argv) {
  uint32_t slots = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : 8192;
  uint32_t level_budget =
      (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 3;

  VL_UI32* rot_group = Build_rot_group(slots);
  VL_DCMPLX* ksi_pows = Build_ksi_pows(slots);
  VL_I32* enc_params = Get_colls_fft_params(slots, level_budget, 0);
  VL_I32* dec_params = Get_colls_fft_params(slots, level_budget, 0);
  VL_VL_VL_DCMPLX* enc_coeff =
      Coeff_collapse(ksi_pows, rot_group, level_budget, false, true);
  VL_VL_VL_DCMPLX* dec_coeff =
      Coeff_collapse(ksi_pows, rot_group, level_budget, false, false);

  FILE* fp = stdout;
  fprintf(fp, "{");
  fprintf(fp, "\"slots\": %u, ", slots);
  fprintf(fp, "\"level_budget\": %u, ", level_budget);
  fprintf(fp, "\"encode_params\": ");
  Print_params_json(fp, enc_params);
  fprintf(fp, ", \"decode_params\": ");
  Print_params_json(fp, dec_params);
  fprintf(fp, ", \"encode_coeff_collapse\": ");
  Print_coeff_summary(fp, enc_coeff);
  fprintf(fp, ", \"decode_coeff_collapse\": ");
  Print_coeff_summary(fp, dec_coeff);
  fprintf(fp, "}\n");

  Free_value_list(enc_coeff);
  Free_value_list(dec_coeff);
  Free_value_list(enc_params);
  Free_value_list(dec_params);
  Free_value_list(rot_group);
  Free_value_list(ksi_pows);
  return 0;
}
