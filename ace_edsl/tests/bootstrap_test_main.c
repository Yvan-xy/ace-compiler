//-*-c-*-
//=============================================================================
// Test harness for generated bootstrap_full kernel.
// Compile with examples/output/bootstrap_full.c (which includes the wrapper
// providing Main_graph, encode/decode schemes) and link with ANT runtime.
// Run with CWD = examples/output so bootstrap_full_data.msg is found.
//=============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rt_ant/rt_ant.h"
#include "rt_ant/rt_api.h"
#include "ckks/ciphertext.h"
#include "common/rt_api.h"

#define NUM_SLOTS 8

#ifdef __cplusplus
extern "C" {
#endif
extern CIPHERTEXT bootstrap_full(CIPHERTEXT p0, CIPHERTEXT p1);
#ifdef __cplusplus
}
#endif

static double Input_p0[] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8};
static double Input_p1[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static double Expected_identity[] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8};

static TENSOR* Generate_input_data(size_t n, size_t c, size_t h, size_t w,
                                  double* data) {
  return Alloc_tensor(n, c, h, w, data);
}

static bool Validate_output_data(double* result, double* expect, int len) {
  double error = 1e-2;
  for (int i = 0; i < len; i++) {
    if (fabs(result[i] - expect[i]) > error) {
      printf("index: %d, value: %f != %f\n", i, result[i], expect[i]);
      return false;
    }
  }
  return true;
}

static void Print_output_data(double* result, int len) {
  printf("RESULT:");
  for (int i = 0; i < len; i++) {
    printf(i == 0 ? "%.9f" : ",%.9f", result[i]);
  }
  printf("\n");
}

static int Env_int(const char* name, int default_value) {
  const char* raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return default_value;
  }
  char* end = NULL;
  long parsed = strtol(raw, &end, 10);
  if (end == raw || (end != NULL && *end != '\0') || parsed < 0) {
    return default_value;
  }
  return (int)parsed;
}

static int Env_flag(const char* name, int default_value) {
  const char* raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') {
    return default_value;
  }
  return strcmp(raw, "0") != 0;
}

static double Now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char* argv[]) {
  int warmup_runs = Env_int("ACE_BOOTSTRAP_WARMUP_RUNS", 0);
  int measure_runs = Env_int("ACE_BOOTSTRAP_MEASURE_RUNS", 1);
  int total_runs = warmup_runs + measure_runs;
  int skip_validate = Env_flag("ACE_BOOTSTRAP_SKIP_VALIDATE", total_runs > 1);
  if (total_runs <= 0) {
    fprintf(stderr, "ACE_BOOTSTRAP_WARMUP_RUNS + ACE_BOOTSTRAP_MEASURE_RUNS must be > 0\n");
    return 2;
  }

  Prepare_context();

  TENSOR* tensor_p0 = Generate_input_data(1, 1, 1, NUM_SLOTS, Input_p0);
  TENSOR* tensor_p1 = Generate_input_data(1, 1, 1, NUM_SLOTS, Input_p1);
  Prepare_input(tensor_p0, "p0");
  Prepare_input(tensor_p1, "p1");
  Free_tensor(tensor_p0);
  Free_tensor(tensor_p1);

  CIPHERTEXT p0 = Get_input_data("p0", 0);
  CIPHERTEXT p1 = Get_input_data("p1", 0);
  CIPHERTEXT result_ct;
  memset(&result_ct, 0, sizeof(result_ct));

  for (int run = 0; run < total_runs; ++run) {
    double t0 = Now_sec();
    result_ct = bootstrap_full(p0, p1);
    printf("BOOTSTRAP_RUN[%d] phase=%s elapsed=%.6f\n",
           run + 1,
           run < warmup_runs ? "warmup" : "measure",
           Now_sec() - t0);
    fflush(stdout);
  }

  Set_output_data("output", 0, &result_ct);
  double* result = Handle_output("output");
  Print_output_data(result, NUM_SLOTS);
  Finalize_context();

  bool res = skip_validate ? true : Validate_output_data(result, Expected_identity, NUM_SLOTS);
  free(result);
  if (res) {
    printf("SUCCESS\n");
    return 0;
  }
  printf("FAILED\n");
  return 1;
}

// The wrapper (Main_graph, Get_encode_scheme, etc.) is now appended to
// bootstrap_full.c by the demo, so no separate #include is needed here.
