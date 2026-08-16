// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The model_and_main DNN drivers call Handle_output(), but do not print or
// validate the returned logits.  The model runners compile those drivers with
// -DHandle_output=Capture_dnn_output so the test can inspect the result without
// modifying the generated driver sources.

#include "common/rtlib.h"

double *Handle_output(const char *name);

void Suppress_dnn_tensor(FILE *stream, TENSOR *tensor) {
  (void)tensor;
  fputs("(input tensor omitted; logits are reported below)\n", stream);
}

double *Capture_dnn_output(const char *name) {
  double *output = Handle_output(name);
  if (output == NULL) {
    fprintf(stderr, "ACE_DNN_OUTPUT_ERROR=null\n");
    return output;
  }

  for (int index = 0; index < 10; ++index) {
    printf("ACE_DNN_OUTPUT index=%d value=%.17g\n", index, output[index]);
  }
  fflush(stdout);
  return output;
}
