//-*-c++-*-
// Measure exactly one invocation of a generated primitive CKKS bootstrap.
// Runtime setup, input encryption, output decoding, and teardown are outside
// the measured interval.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <time.h>

#include "ckks/cipher.h"
#include "ckks/ciphertext.h"
#include "ckks/key_gen.h"
#include "common/rt_api.h"
#include "rt_ant/rt_ant.h"

#ifndef ACE_CPU_NATIVE_RTL
#define ACE_CPU_NATIVE_RTL 0
#endif

extern "C" CIPHERTEXT bootstrap_full(CIPHERTEXT input,
                                       CIPHERTEXT encrypted_zero);
extern "C" void ace_cpu_bootstrap_set_target_level(std::uint32_t target_level);
#if !ACE_CPU_NATIVE_RTL
extern "C" void ace_cpu_bootstrap_stage_probe_begin();
extern "C" void ace_cpu_bootstrap_stage_probe_finish();
#endif

#if !defined(ACE_CPU_BASE_DEGREE) || !defined(ACE_CPU_BASE_SEC_LEVEL) ||      \
    !defined(ACE_CPU_BASE_MUL_DEPTH) || !defined(ACE_CPU_BASE_INPUT_LEVEL) || \
    !defined(ACE_CPU_BASE_FIRST_MOD_SIZE) ||                                  \
    !defined(ACE_CPU_BASE_SCALING_MOD_SIZE) ||                                \
    !defined(ACE_CPU_BASE_NUM_Q_PARTS) ||                                     \
    !defined(ACE_CPU_BASE_HAMMING_WEIGHT)
#error "the generated ResNet base context must be supplied by the runner"
#endif

extern "C" {
CKKS_PARAMS* Get_context_params() {
  static CKKS_PARAMS* parameters = nullptr;
  if (parameters == nullptr) {
    parameters = static_cast<CKKS_PARAMS*>(std::calloc(1, sizeof(CKKS_PARAMS)));
    if (parameters == nullptr) return nullptr;
    parameters->_provider = LIB_ANT;
    parameters->_poly_degree = ACE_CPU_BASE_DEGREE;
    parameters->_sec_level = ACE_CPU_BASE_SEC_LEVEL;
    parameters->_mul_depth = ACE_CPU_BASE_MUL_DEPTH;
    parameters->_input_level = ACE_CPU_BASE_INPUT_LEVEL;
    parameters->_first_mod_size = ACE_CPU_BASE_FIRST_MOD_SIZE;
    parameters->_scaling_mod_size = ACE_CPU_BASE_SCALING_MOD_SIZE;
    parameters->_num_q_parts = ACE_CPU_BASE_NUM_Q_PARTS;
    parameters->_hamming_weight = ACE_CPU_BASE_HAMMING_WEIGHT;
    // Prepare_context merges the generated bootstrap's rotation list through
    // Get_extra_context_params(). ResNet-only keys are unnecessary here.
    parameters->_num_rot_idx = 0;
  }
  return parameters;
}

int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
RT_DATA_INFO* Get_rt_data_info() { return nullptr; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

namespace {

double NowSeconds() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    std::perror("clock_gettime");
    std::exit(2);
  }
  return static_cast<double>(value.tv_sec) +
         static_cast<double>(value.tv_nsec) * 1.0e-9;
}

std::uint32_t TargetLevel() {
  const char* raw = std::getenv("ACE_CPU_BOOTSTRAP_TARGET_LEVEL");
  if (raw == nullptr || raw[0] == '\0') {
    std::fprintf(stderr, "ACE_CPU_BOOTSTRAP_TARGET_LEVEL is required\n");
    std::exit(2);
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    std::fprintf(stderr, "invalid ACE_CPU_BOOTSTRAP_TARGET_LEVEL: %s\n", raw);
    std::exit(2);
  }
  return static_cast<std::uint32_t>(parsed);
}

double MaximumError() {
  const char* raw = std::getenv("ACE_CPU_BOOTSTRAP_MAX_ERROR");
  if (raw == nullptr || raw[0] == '\0') {
    std::fprintf(stderr, "ACE_CPU_BOOTSTRAP_MAX_ERROR is required\n");
    std::exit(2);
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) {
    std::fprintf(stderr, "invalid ACE_CPU_BOOTSTRAP_MAX_ERROR: %s\n", raw);
    std::exit(2);
  }
  return parsed;
}

CIPHER EncryptValues(std::vector<DCMPLX>& values, std::size_t level) {
  PLAIN plain = Alloc_plaintext();
  CIPHER cipher = Alloc_ciphertext();
  if (plain == nullptr || cipher == nullptr) {
    std::fprintf(stderr, "failed to allocate bootstrap input\n");
    std::exit(2);
  }
  Encode_dcmplx(plain, values.data(), values.size(), 1, level);
  Encrypt(cipher, plain);
  Free_plaintext(plain);
  return cipher;
}

void ProvisionConjugationKey(std::uint32_t degree) {
  const std::uint64_t auto_index = 2ULL * degree - 1ULL;
  if (degree == 0 ||
      auto_index > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int32_t>::max())) {
    std::fprintf(stderr, "invalid generated polynomial degree: %u\n", degree);
    std::exit(2);
  }
  auto* generator = reinterpret_cast<CKKS_KEY_GENERATOR*>(Keygen());
  if (generator == nullptr ||
      Insert_rot_map(generator, static_cast<std::int32_t>(auto_index)) ==
          nullptr) {
    std::fprintf(stderr, "failed to provision the conjugation key\n");
    std::exit(2);
  }
}

}  // namespace

int main() {
  // Keep runtime bootstrap precomputation out of this primitive DSL path.
  if (!ACE_CPU_NATIVE_RTL) {
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
  }
  Prepare_context();

  CKKS_PARAMS* parameters = Get_context_params();
  CKKS_PARAMS* generated_parameters = Get_extra_context_params();
  if (parameters == nullptr || generated_parameters == nullptr ||
      parameters->_poly_degree != generated_parameters->_poly_degree ||
      parameters->_poly_degree < 2 || (parameters->_poly_degree & 1U) != 0U) {
    std::fprintf(stderr, "generated bootstrap has invalid context metadata\n");
    return 2;
  }
  const std::uint32_t degree = parameters->_poly_degree;
  const std::size_t slots = degree / 2U;
  const std::size_t input_level = generated_parameters->_input_level;
  ProvisionConjugationKey(degree);

  // Use every logical slot, matching the full-packed ResNet ciphertext shape.
  // Values stay well inside the attested EvalMod approximation interval.
  std::vector<DCMPLX> input_values(slots);
  for (std::size_t index = 0; index < slots; ++index) {
    const double value =
        0.0625 * static_cast<double>(static_cast<int>(index % 17U) - 8);
    input_values[index] = DCMPLX(value, 0.0);
  }
  CIPHER input = EncryptValues(input_values, input_level);
  CIPHERTEXT input_copy{};
  Copy_ciphertext(&input_copy, input);
  const std::uint32_t target_level = TargetLevel();
  ace_cpu_bootstrap_set_target_level(target_level);

  // This is the complete timing boundary: exactly one generated DSL call.
#if !ACE_CPU_NATIVE_RTL
  ace_cpu_bootstrap_stage_probe_begin();
#endif
  const double begin = NowSeconds();
  CIPHERTEXT result = bootstrap_full(input_copy, input_copy);
  const double elapsed_seconds = NowSeconds() - begin;
#if !ACE_CPU_NATIVE_RTL
  ace_cpu_bootstrap_stage_probe_finish();
#endif

  DCMPLX* decoded = Get_msg_with_imag(&result);
  if (decoded == nullptr || Get_slots(&result) != slots) {
    std::fprintf(stderr, "failed to decode the generated bootstrap result\n");
    return 1;
  }
  const std::size_t validation_count = slots;
  double maximum_error = 0.0;
  for (std::size_t index = 0; index < validation_count; ++index) {
    if (!std::isfinite(decoded[index].real()) ||
        !std::isfinite(decoded[index].imag())) {
      std::fprintf(stderr, "non-finite bootstrap result at slot %zu\n", index);
      std::free(decoded);
      return 1;
    }
    const double error = std::abs(decoded[index] - input_values[index]);
    if (error > maximum_error) maximum_error = error;
  }
  std::free(decoded);
  const double maximum_allowed_error = MaximumError();
  if (maximum_error > maximum_allowed_error) {
    std::fprintf(stderr,
                 "bootstrap maximum error %.9g exceeds tolerance %.9g\n",
                 maximum_error, maximum_allowed_error);
    return 1;
  }

  CRT_CONTEXT* crt = Get_crt_context();
  const std::size_t data_q_count =
      crt == nullptr ? 0U : Get_primes_cnt(Get_q(crt));
  const std::size_t special_p_count =
      crt == nullptr ? 0U : Get_primes_cnt(Get_p(crt));
  std::printf(
      "CPU_BOOTSTRAP_RESULT={\"status\":\"pass\","
      "\"provider\":\"ant\",\"implementation\":\"%s\","
      "\"generated_dsl_invocations\":%u,"
      "\"polynomial_degree\":%u,\"logical_slots\":%zu,"
      "\"generated_mul_depth\":%zu,\"runtime_mul_depth\":%zu,"
      "\"data_q_count\":%zu,\"special_p_count\":%zu,"
      "\"input_level\":%zu,\"runtime_input_level\":%zu,"
      "\"target_level\":%u,"
      "\"elapsed_seconds\":%.9f,"
      "\"validation_slot_count\":%zu,\"maximum_error\":%.9g,"
      "\"timing_scope\":\"bootstrap_full_only\"}\n",
      ACE_CPU_NATIVE_RTL ? "native_rtl" : "dsl_linear_transform",
      ACE_CPU_NATIVE_RTL ? 0U : 1U, degree, slots,
      generated_parameters->_mul_depth, parameters->_mul_depth,
      data_q_count, special_p_count, input_level, parameters->_input_level,
      target_level, elapsed_seconds, validation_count, maximum_error);
  std::fflush(stdout);

  Zero_ciph(&result);
  Free_poly_data(Get_c0(&input_copy));
  Free_poly_data(Get_c1(&input_copy));
  Free_ciphertext(input);
  Finalize_context();
  return 0;
}
