#include "boot/ckks_evaluator.cuh"
#include "evaluate.cuh"
#include "rt_phantom/rt_phantom.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// This is a C++ symbol emitted by the ACE-generated bootstrap translation
// unit.  Only the narrow native-CNN bridge below has C linkage.
extern CIPHERTEXT bootstrap_full(CIPHERTEXT, CIPHERTEXT);

namespace {

constexpr std::size_t kPolynomialDegree = 65536;
constexpr std::size_t kLogicalSlots = 32768;
constexpr std::size_t kDataQCount = 32;
constexpr std::size_t kSpecialPCount = 11;
constexpr std::size_t kInputQCount = 1;
constexpr std::size_t kOutputQCount = 17;
constexpr std::size_t kQPartCount = 3;
constexpr std::size_t kHammingWeight = 192;
constexpr std::uint32_t kFirstModulusBits = 60;
constexpr std::uint32_t kScalingModulusBits = 56;
constexpr double kScaleCoordinateTolerance = 1.0e-4;
constexpr double kCanonicalScale = 72057594037927936.0; // 2^56

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

bool HasDegreeOneScale(double scale) {
  if (!std::isfinite(scale) || scale <= 0.0)
    return false;
  const double coordinate = std::log2(scale) / kScalingModulusBits;
  return std::isfinite(coordinate) &&
         std::abs(coordinate - 1.0) <= kScaleCoordinateTolerance;
}

#if defined(ACE_PHANTOM_RUN_CNN_BOOTSTRAP_DIAGNOSTIC)
std::vector<std::complex<double>>
DecodeDiagnosticSlots(PhantomCiphertext &value,
                      phantom::CKKSEvaluator &evaluator) {
  PhantomPlaintext plain;
  std::vector<std::complex<double>> decoded;
  evaluator.decryptor.decrypt(value, plain);
  evaluator.encoder.decode(plain, decoded);
  Require(decoded.size() >= kLogicalSlots,
          "diagnostic decode returned fewer than 32768 slots");
  decoded.resize(kLogicalSlots);
  return decoded;
}

void PrintDiagnosticSlots(const char *label, std::size_t stage,
                          const PhantomCiphertext &value,
                          const std::vector<std::complex<double>> &decoded) {
  long double squared_sum = 0.0L;
  double maximum = 0.0;
  std::size_t maximum_index = 0;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < decoded.size(); ++index) {
    const double magnitude = std::abs(decoded[index]);
    if (!std::isfinite(decoded[index].real()) ||
        !std::isfinite(decoded[index].imag()) || !std::isfinite(magnitude)) {
      ++nonfinite;
      continue;
    }
    squared_sum += static_cast<long double>(magnitude) * magnitude;
    if (magnitude > maximum) {
      maximum = magnitude;
      maximum_index = index;
    }
  }
  const double rms =
      std::sqrt(static_cast<double>(squared_sum / decoded.size()));
  std::fprintf(stderr,
               "ACE_CNN_BTS_DIAGNOSTIC stage=%zu label=%s active_q=%zu "
               "chain=%zu log2_scale=%.17g slots=%zu nonfinite=%zu "
               "max_abs=%.17g max_index=%zu rms=%.17g\n",
               stage, label, value.coeff_modulus_size(), value.chain_index(),
               std::log2(value.scale()), decoded.size(), nonfinite, maximum,
               maximum_index, rms);
  const std::size_t shown = decoded.size() < 8 ? decoded.size() : 8;
  for (std::size_t index = 0; index < shown; ++index) {
    std::fprintf(stderr,
                 "ACE_CNN_BTS_DIAGNOSTIC_SLOT stage=%zu label=%s index=%zu "
                 "real=%.17g imag=%.17g\n",
                 stage, label, index, decoded[index].real(),
                 decoded[index].imag());
  }
  std::fflush(stderr);
}

void PrintDiagnosticComparison(
    std::size_t stage, const std::vector<std::complex<double>> &input,
    const std::vector<std::complex<double>> &output) {
  Require(input.size() == output.size(),
          "diagnostic input/output slot counts differ");
  long double squared_sum = 0.0L;
  double maximum = 0.0;
  std::size_t maximum_index = 0;
  std::size_t nonfinite = 0;
  for (std::size_t index = 0; index < input.size(); ++index) {
    const double error = std::abs(output[index] - input[index]);
    if (!std::isfinite(error)) {
      ++nonfinite;
      continue;
    }
    squared_sum += static_cast<long double>(error) * error;
    if (error > maximum) {
      maximum = error;
      maximum_index = index;
    }
  }
  const double rms = std::sqrt(static_cast<double>(squared_sum / input.size()));
  std::fprintf(stderr,
               "ACE_CNN_BTS_DIAGNOSTIC_COMPARISON stage=%zu slots=%zu "
               "nonfinite=%zu max_abs_error=%.17g max_index=%zu "
               "rms_error=%.17g\n",
               stage, input.size(), nonfinite, maximum, maximum_index, rms);
  std::fflush(stderr);
}
#endif

// A provider object's C++ destructor releases its device allocation, but it
// cannot close ACE's diagnostic lifetime state.  Arm this guard immediately
// after registration so both normal returns and caught exceptions transition
// every bridge-local object through zero to freed.  Zero_ciph deliberately
// accepts an otherwise malformed live object, which also makes cleanup safe
// after a failed output-contract check.
class RegisteredCipherCleanup {
public:
  explicit RegisteredCipherCleanup(PhantomCiphertext *value) : _value(value) {}

  RegisteredCipherCleanup(const RegisteredCipherCleanup &) = delete;
  RegisteredCipherCleanup &operator=(const RegisteredCipherCleanup &) = delete;

  ~RegisteredCipherCleanup() { Cleanup(); }

  void Cleanup() {
    if (!_armed)
      return;
    Zero_ciph(_value);
    Free_ciph(_value);
    _armed = false;
  }

private:
  PhantomCiphertext *_value;
  bool _armed = true;
};

void ValidateCompilerManifest(const PHANTOM_CONTEXT_MANIFEST *manifest,
                              std::size_t logical_slots) {
  Require(manifest != nullptr, "compiler context manifest is null");
  Require(manifest->_schema_version == 1,
          "compiler context schema is not version 1");
  Require(manifest->_resource_schema_version == 3,
          "compiler resource schema is not version 3");
  Require(manifest->_packing == PHANTOM_PACKING_FULL,
          "compiler packing is not full CKKS packing");
  Require(manifest->_poly_degree == kPolynomialDegree,
          "compiler polynomial degree is not 65536");
  Require(manifest->_logical_slots == kLogicalSlots &&
              logical_slots == kLogicalSlots,
          "compiler/caller logical slot count is not 32768");
  Require(manifest->_data_q_count == kDataQCount &&
              manifest->_data_q_bit_sizes != nullptr,
          "compiler data-Q profile is not Q32");
  Require(manifest->_special_p_count == kSpecialPCount &&
              manifest->_special_p_bit_sizes != nullptr,
          "compiler special-P profile is not P11");
  Require(manifest->_input_level == kInputQCount,
          "compiler input level is not 1");
  Require(manifest->_q_part_count == kQPartCount,
          "compiler Q-part count is not 3");
  Require(manifest->_hamming_weight == kHammingWeight,
          "compiler secret-key Hamming weight is not 192");
  Require(manifest->_first_modulus_bits == kFirstModulusBits,
          "compiler first-modulus size is not 60");
  Require(manifest->_scaling_modulus_bits == kScalingModulusBits,
          "compiler scaling-modulus size is not 56");

  Require(manifest->_data_q_bit_sizes[0] == kFirstModulusBits,
          "compiler Q0 entry is not 60");
  for (std::size_t index = 1; index < kDataQCount; ++index) {
    Require(manifest->_data_q_bit_sizes[index] == kScalingModulusBits,
            "compiler data-Q tail is not uniformly sf=56");
  }
  for (std::size_t index = 0; index < kSpecialPCount; ++index) {
    Require(manifest->_special_p_bit_sizes[index] == kFirstModulusBits,
            "compiler special-P profile is not uniformly 60");
  }
}

void ValidateBorrowedRuntime(const PHANTOM_BORROWED_RUNTIME &runtime,
                             const phantom::CKKSEvaluator &evaluator) {
  Require(runtime._context != nullptr && runtime._encoder != nullptr &&
              runtime._secret_key != nullptr &&
              runtime._public_key != nullptr && runtime._relin_key != nullptr &&
              runtime._galois_key != nullptr,
          "ACE singleton did not provide its complete bootstrap keyset");
  Require(runtime._secret_key->is_generated(),
          "ACE singleton secret key is not generated");
  Require(runtime._public_key->is_generated(),
          "ACE singleton public key is not generated");
  Require(runtime._relin_key->is_generated(),
          "ACE singleton relinearization key is not generated");
  Require(runtime._galois_key->is_generated(),
          "ACE singleton bootstrap Galois key is not generated");

  const auto &context = *runtime._context;
  Require(context.total_parm_size() == kDataQCount + 1,
          "runtime data-Q chain is not Q32 plus its key context");
  Require(context.get_first_index() == 1,
          "runtime first data-Q chain index is not 1");
  Require(context.key_context_data().parms().poly_modulus_degree() ==
              kPolynomialDegree,
          "runtime polynomial degree is not 65536");
  Require(context.first_context_data().parms().coeff_modulus().size() ==
              kDataQCount,
          "runtime first data context is not Q32");
  Require(context.key_context_data().parms().coeff_modulus().size() ==
              kDataQCount + kSpecialPCount,
          "runtime key context is not Q32/P11");
  Require(runtime._encoder->logical_slot_count() == kLogicalSlots &&
              runtime._encoder->physical_slot_count() == kLogicalSlots,
          "runtime encoder is not full-packed at 32768 slots");

  const auto &key_parms_id = context.key_context_data().parms().parms_id();
  Require(runtime._relin_key->parms_id() == key_parms_id,
          "ACE relinearization key belongs to another context");
  Require(runtime._galois_key->parms_id() == key_parms_id,
          "ACE bootstrap Galois key belongs to another context");

  // Public CKKSEvaluator state exposes the two provider identities that its
  // native arithmetic actually consumes.  Its CNN rotations may deliberately
  // use a separately generated Galois-key subset, but that subset must still
  // be bound to the exact same parameter identity.
  Require(evaluator.context == runtime._context,
          "CNN evaluator does not borrow the ACE singleton context");
  Require(evaluator.relin_keys == runtime._relin_key,
          "CNN evaluator does not borrow the ACE singleton relin key");
  Require(evaluator.galois_keys != nullptr &&
              evaluator.galois_keys->is_generated(),
          "CNN evaluator Galois key is missing");
  Require(evaluator.galois_keys->parms_id() == key_parms_id,
          "CNN evaluator Galois key belongs to another context");
  Require(evaluator.degree == kPolynomialDegree &&
              evaluator.slot_count == kLogicalSlots &&
              HasDegreeOneScale(evaluator.scale),
          "CNN evaluator metadata does not match the ACE CKKS profile");
}

std::size_t ValidateProviderCiphertext(const PhantomContext &context,
                                       const PhantomCiphertext &value,
                                       const char *label) {
  Require(value.size() == 2, std::string(label) + " ciphertext size is not 2");
  Require(value.is_ntt_form(),
          std::string(label) + " ciphertext is not in NTT form");
  Require(value.poly_modulus_degree() == kPolynomialDegree,
          std::string(label) + " polynomial degree is not 65536");
  Require(HasDegreeOneScale(value.scale()),
          std::string(label) + " scale is not degree 1 for sf=56");
  Require(value.GetNoiseScaleDeg() == 1,
          std::string(label) + " noise-scale degree is not 1");

  const std::size_t chain = value.chain_index();
  Require(chain >= context.get_first_index() &&
              chain < context.total_parm_size(),
          std::string(label) + " chain index is outside the data-Q chain");
  const auto &context_data = context.get_context_data(chain);
  const std::size_t active_q = context_data.parms().coeff_modulus().size();
  Require(active_q >= 1 && active_q <= kDataQCount,
          std::string(label) + " active-Q count is outside Q32");
  Require(value.coeff_modulus_size() == active_q,
          std::string(label) + " coefficient count disagrees with its chain");
  Require(value.parms_id() == context_data.parms().parms_id(),
          std::string(label) + " parameter identity disagrees with its chain");
  return active_q;
}

void ValidateBootstrapOutput(PhantomCiphertext *output,
                             const PhantomContext &context) {
  const std::size_t active_q =
      ValidateProviderCiphertext(context, *output, "bootstrap output");
  const std::size_t expected_chain =
      context.get_first_index() + kDataQCount - kOutputQCount;
  Require(active_q == kOutputQCount,
          "bootstrap output active-Q count is not 17");
  Require(output->chain_index() == expected_chain,
          "bootstrap output chain index is not the Q17 index");

  // Exercise the public ACE metadata contract only after provider-level shape
  // and parameter checks make all adapter queries safe.
  Require(Active_q_count(output) == static_cast<LEVEL_T>(kOutputQCount),
          "ACE bootstrap output active-Q metadata is not 17");
  Require(Level(output) == static_cast<LEVEL_T>(kOutputQCount),
          "ACE bootstrap output level metadata is not 17");
  Require(Chain_index(output) == static_cast<LEVEL_T>(expected_chain),
          "ACE bootstrap output chain metadata is inconsistent");
  Require(Get_ciph_slots(output) == kLogicalSlots,
          "ACE bootstrap output slot metadata is not 32768");
  Require(Get_ciph_size(output) == 2 && Is_ciph_ntt(output),
          "ACE bootstrap output shape metadata is invalid");
  Require(Sc_degree(output) == 1.0,
          "ACE bootstrap output scale degree is not 1");

  const double raw_scale = Raw_scale(output);
  Require(HasDegreeOneScale(raw_scale),
          "ACE bootstrap output raw scale violates the sf=56 coordinate");
}

} // namespace

extern "C" int Ace_phantom_run_cnn_dsl_bootstrap(
    PhantomCiphertext *output, PhantomCiphertext *input,
    phantom::CKKSEvaluator *evaluator, std::size_t logical_slots,
    std::size_t stage) {
  try {
    Require(output != nullptr && input != nullptr && evaluator != nullptr,
            "output, input, and evaluator must be non-null");
    Require(output != input, "in-place native/DSL bridge calls are forbidden");

    // The linked generated artifact is the only compiler-profile authority.
    // Reject the wrong artifact before reading or copying either ciphertext.
    const PHANTOM_CONTEXT_MANIFEST *manifest = Get_phantom_context_manifest();
    ValidateCompilerManifest(manifest, logical_slots);

    const PHANTOM_BORROWED_RUNTIME runtime = Phantom_borrow_runtime();
    ValidateBorrowedRuntime(runtime, *evaluator);
    ValidateProviderCiphertext(*runtime._context, *input, "CNN input");

#if defined(ACE_PHANTOM_RUN_CNN_BOOTSTRAP_DIAGNOSTIC)
    std::vector<std::complex<double>> diagnostic_input;
    if (stage == 1) {
      diagnostic_input = DecodeDiagnosticSlots(*input, *evaluator);
      PrintDiagnosticSlots("input-q1", stage, *input, diagnostic_input);
    }
#endif

    PhantomCiphertext owned_input;
    copy_ciphertext(*runtime._context, *input, owned_input);
    Register_ciph_lifetime(&owned_input);
    RegisteredCipherCleanup owned_input_cleanup(&owned_input);
    while (Active_q_count(&owned_input) > static_cast<LEVEL_T>(kInputQCount)) {
      Mod_switch(&owned_input, &owned_input);
    }
    Require(Active_q_count(&owned_input) ==
                    static_cast<LEVEL_T>(kInputQCount) &&
                Level(&owned_input) == static_cast<LEVEL_T>(kInputQCount),
            "CNN input could not be normalized to active Q1");

    // bootstrap_full's second by-value parameter is an ABI placeholder in the
    // generated program.  Pass a real, independently owned and registered Q1
    // ciphertext rather than an uninitialized sentinel.
    PhantomCiphertext auxiliary;
    copy_ciphertext(*runtime._context, owned_input, auxiliary);
    Register_ciph_lifetime(&auxiliary);
    RegisteredCipherCleanup auxiliary_cleanup(&auxiliary);

    CheckCuda(cudaDeviceSynchronize(),
              "pre-DSL-bootstrap synchronization failed");
    PhantomCiphertext result = bootstrap_full(owned_input, auxiliary);
    Register_ciph_lifetime(&result);
    RegisteredCipherCleanup result_cleanup(&result);
    CheckCuda(cudaDeviceSynchronize(),
              "post-DSL-bootstrap synchronization failed");

    ValidateBootstrapOutput(&result, *runtime._context);

#if defined(ACE_PHANTOM_RUN_CNN_BOOTSTRAP_DIAGNOSTIC)
    std::vector<std::complex<double>> diagnostic_raw_output;
    if (stage == 1) {
      diagnostic_raw_output = DecodeDiagnosticSlots(result, *evaluator);
      PrintDiagnosticSlots("output-q17-raw-scale", stage, result,
                           diagnostic_raw_output);
      PrintDiagnosticComparison(stage, diagnostic_input, diagnostic_raw_output);
    }
#endif

    // The generated ordinary-CKKS bootstrap preserves Phantom's exact
    // post-rescale S/q metadata, while the handwritten CNN polynomial
    // evaluator treats its configured 2^sf scale as an exact recurrence
    // invariant.  A degree-one drift that is harmless to ordinary generated
    // code compounds through the degree-{15,15,27} ReLU and eventually makes
    // its next plaintext encode exceed the modulus.  Validate the generated
    // coordinate above, then establish the native evaluator's documented
    // boundary invariant without consuming a level.
    result.scale() = kCanonicalScale;
    result.SetNoiseScaleDeg(1);
    Require(result.scale() == std::ldexp(1.0, kScalingModulusBits),
            "bootstrap output scale could not be canonicalized to 2^56");

#if defined(ACE_PHANTOM_RUN_CNN_BOOTSTRAP_DIAGNOSTIC)
    if (stage == 1) {
      const auto diagnostic_canonical_output =
          DecodeDiagnosticSlots(result, *evaluator);
      PrintDiagnosticSlots("output-q17-canonical-scale", stage, result,
                           diagnostic_canonical_output);
    }
#endif

    *output = std::move(result);
    Register_ciph_lifetime(output);
    result_cleanup.Cleanup();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr,
                 "ACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP_ERROR stage=%zu: %s\n",
                 stage, error.what());
    std::fflush(stderr);
    return 1;
  } catch (...) {
    std::fprintf(stderr,
                 "ACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP_ERROR stage=%zu: "
                 "unknown exception\n",
                 stage);
    std::fflush(stderr);
    return 2;
  }
}
