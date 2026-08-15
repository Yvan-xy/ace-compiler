// Native ANT bootstrap entry point for the one-round CPU A/B harness.

#include <stdint.h>
#include <string.h>

#include "ckks/cipher.h"

static uint32_t Bootstrap_target_level = 0;

void ace_cpu_bootstrap_set_target_level(uint32_t target_level) {
  Bootstrap_target_level = target_level;
}

CIPHERTEXT bootstrap_full(CIPHERTEXT input, CIPHERTEXT encrypted_zero) {
  (void)encrypted_zero;
  FMT_ASSERT(Bootstrap_target_level > 0, "bootstrap target level was not set");
  CIPHERTEXT result;
  memset(&result, 0, sizeof(result));
  Eval_bootstrap_ciph(&result, &input, Bootstrap_target_level,
                      Get_ciph_slots(&input));
  return result;
}
