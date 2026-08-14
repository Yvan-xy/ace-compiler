//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "rt_phantom/phantom_api.h"
#include "rt_phantom/rt_phantom.h"

#include "common/common.h"
#include "common/io_api.h"
#include "common/rt_api.h"
#include "common/rtlib_timing.h"
#include "phantom_ordinary_contract.h"

#include "ckks.h"
#include "context.cuh"
#include "evaluate.cuh"
#include "secretkey.h"
#include "util/encryptionparams.h"
#include "util/modulus.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using ace::phantom::ordinary::AceLevelToChainIndex;
using ace::phantom::ordinary::ChainIndexToAceLevel;
using ace::phantom::ordinary::ChainIndexToActiveQ;
using ace::phantom::ordinary::NormalizeRotation;
using ace::phantom::ordinary::ScaleDegree;
using ace::phantom::ordinary::ScaleForDegree;

#ifdef ENABLE_PERFORMANCE_STATS
#include <unordered_map>
std::unordered_map<std::string, OperationStats> operation_stats;
#endif

namespace {

enum class ObjectState : uint8_t { kUninitialized, kLive, kZero, kFreed };

struct ConstantCacheKey {
  uint64_t               _constant_id = 0;
  phantom::parms_id_type _parameter_fingerprint{};
  size_t                 _chain_index = 0;
  double                 _raw_scale = 0.0;
  uint32_t               _element_type = 0;
  size_t                 _slot_count = 0;

  [[nodiscard]] bool operator<(const ConstantCacheKey& other) const noexcept {
    return std::tie(_constant_id, _parameter_fingerprint, _chain_index,
                    _raw_scale, _element_type, _slot_count) <
           std::tie(other._constant_id, other._parameter_fingerprint,
                    other._chain_index, other._raw_scale,
                    other._element_type, other._slot_count);
  }
};

[[noreturn]] void Fail(const char* diagnostic, const char* format, ...) {
  std::fprintf(stderr, "ACE_PHANTOM_ORDINARY_ERROR[%s]: ", diagnostic);
  va_list args;
  va_start(args, format);
  std::vfprintf(stderr, format, args);
  va_end(args);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  std::abort();
}

template <typename FN>
void ProviderCall(const char* diagnostic, FN&& fn) {
  try {
    fn();
  } catch (const std::exception& error) {
    Fail(diagnostic, "Phantom rejected the operation: %s", error.what());
  } catch (...) {
    Fail(diagnostic, "Phantom rejected the operation with an unknown error");
  }
}

// ACE's modulus-size manifest uses the ANT convention: the requested size is
// the exponent of the nearest power of two.  In particular, an NTT prime just
// above 2^56 is still a 56-bit scaling prime for compiler purposes even though
// its ordinary binary bit count is 57.
uint32_t RequestedPrimeSize(uint64_t prime) {
  if (prime < 2) Fail("CONTEXT_MODULI", "modulus must be at least two");
  const uint64_t original = prime;
  uint32_t floor_log2 = 0;
  while (prime > 1) {
    ++floor_log2;
    prime >>= 1;
  }
  if (floor_log2 >= 63) {
    Fail("CONTEXT_MODULI", "modulus exceeds the supported size");
  }
  const uint64_t lower = uint64_t{1} << floor_log2;
  const uint64_t upper = lower << 1;
  return original - lower <= upper - original ? floor_log2 : floor_log2 + 1;
}

bool IsPrime(uint64_t candidate) {
  return phantom::arith::Modulus(candidate).is_prime();
}

uint64_t AntFirstPrime(uint32_t ring_degree, uint32_t requested_size) {
  const uint64_t order = uint64_t{2} * ring_degree;
  uint64_t candidate = (uint64_t{1} << requested_size) + order + 1;
  while (!IsPrime(candidate)) {
    if (candidate > std::numeric_limits<uint64_t>::max() - order) {
      Fail("CONTEXT_MODULI", "first-prime search overflowed");
    }
    candidate += order;
  }
  return candidate;
}

uint64_t AntPreviousPrime(uint64_t prime, uint64_t order) {
  uint64_t candidate = prime;
  do {
    if (candidate <= order) {
      Fail("CONTEXT_MODULI", "previous-prime search underflowed");
    }
    candidate -= order;
  } while (!IsPrime(candidate));
  return candidate;
}

uint64_t AntNextPrime(uint64_t prime, uint64_t order) {
  // Match ANT's Gen_next_prime exactly.  It initializes to prime + order and
  // increments once more before testing, so prime + 2*order is the first
  // candidate considered.
  if (prime > std::numeric_limits<uint64_t>::max() - order) {
    Fail("CONTEXT_MODULI", "next-prime search overflowed");
  }
  uint64_t candidate = prime + order;
  do {
    if (candidate > std::numeric_limits<uint64_t>::max() - order) {
      Fail("CONTEXT_MODULI", "next-prime search overflowed");
    }
    candidate += order;
  } while (!IsPrime(candidate));
  return candidate;
}

std::vector<phantom::arith::Modulus> BuildAntCompatibleCoefficientModuli(
    const PHANTOM_CONTEXT_MANIFEST& manifest) {
  const uint32_t ring_degree = manifest._poly_degree;
  const uint64_t order = uint64_t{2} * ring_degree;
  std::vector<uint64_t> data_q(manifest._data_q_count);

  uint64_t lower =
      AntFirstPrime(ring_degree, manifest._scaling_modulus_bits);
  uint64_t upper = lower;
  data_q.back() = lower;
  for (size_t offset = 0; offset + 2 < data_q.size(); ++offset) {
    const size_t index = data_q.size() - 2 - offset;
    data_q[index] = (offset & 1U) == 0
                        ? (lower = AntPreviousPrime(lower, order))
                        : (upper = AntNextPrime(upper, order));
  }
  data_q.front() =
      manifest._first_modulus_bits == manifest._scaling_modulus_bits
          ? AntPreviousPrime(lower, order)
          : AntPreviousPrime(
                AntFirstPrime(ring_degree, manifest._first_modulus_bits),
                order);

  std::set<uint64_t> used;
  std::vector<phantom::arith::Modulus> moduli;
  moduli.reserve(manifest._data_q_count + manifest._special_p_count);
  for (size_t index = 0; index < data_q.size(); ++index) {
    const uint64_t prime = data_q[index];
    if (phantom::arith::Modulus(prime).bit_count() >
            USER_MOD_BIT_COUNT_MAX ||
        !used.insert(prime).second || prime % order != 1 || !IsPrime(prime) ||
        RequestedPrimeSize(prime) != manifest._data_q_bit_sizes[index]) {
      Fail("CONTEXT_Q_PRIME",
           "generated data-Q modulus %zu violates the compiler prime policy",
           index);
    }
    moduli.emplace_back(prime);
  }

  uint64_t special_cursor =
      AntFirstPrime(ring_degree, manifest._special_p_bit_sizes[0]);
  for (size_t index = 0; index < manifest._special_p_count; ++index) {
    do {
      special_cursor = AntPreviousPrime(special_cursor, order);
    } while (used.count(special_cursor) != 0);
    if (phantom::arith::Modulus(special_cursor).bit_count() >
            USER_MOD_BIT_COUNT_MAX ||
        !used.insert(special_cursor).second || special_cursor % order != 1 ||
        !IsPrime(special_cursor) ||
        RequestedPrimeSize(special_cursor) !=
            manifest._special_p_bit_sizes[index]) {
      Fail("CONTEXT_P_PRIME",
           "generated special-P modulus %zu violates the compiler prime policy",
           index);
    }
    moduli.emplace_back(special_cursor);
  }
  return moduli;
}

std::int64_t CheckedScaleDegree(SCALE_T degree, const char* diagnostic) {
  if (!std::isfinite(degree) || degree < 0.0 ||
      degree != std::nearbyint(degree)) {
    Fail(diagnostic, "scale degree %.17g must be a nonnegative integer", degree);
  }
  if (degree > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    Fail(diagnostic, "scale degree %.17g is out of range", degree);
  }
  return static_cast<std::int64_t>(degree);
}

class PHANTOM_CONTEXT final {
public:
  using Ciphertext = PhantomCiphertext;
  using Plaintext = PhantomPlaintext;

  static PHANTOM_CONTEXT* Context() {
    if (_instance == nullptr) {
      Fail("CTX_NOT_INITIALIZED", "context is not initialized");
    }
    return _instance;
  }

  static void Initialize() {
    if (_instance != nullptr) {
      Fail("CTX_ALREADY_INITIALIZED", "context is already initialized");
    }
    try {
      _instance = new PHANTOM_CONTEXT();
    } catch (const std::exception& error) {
      Fail("CTX_INIT", "%s", error.what());
    }
  }

  static void Finalize() {
    if (_instance == nullptr) {
      Fail("CTX_NOT_INITIALIZED", "context is not initialized");
    }
    delete _instance;
    _instance = nullptr;
#ifdef ENABLE_PERFORMANCE_STATS
    Print_performance_stats();
#endif
  }

  PHANTOM_CONTEXT(const PHANTOM_CONTEXT&) = delete;
  PHANTOM_CONTEXT& operator=(const PHANTOM_CONTEXT&) = delete;

  ~PHANTOM_CONTEXT() {
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "ACE_PHANTOM_ORDINARY_ERROR[CTX_SYNC]: CUDA synchronization "
                   "failed during teardown: %s\n",
                   cudaGetErrorString(status));
    }
    // Cached plaintexts own device allocations and must retire while the
    // provider context and its streams are still alive.
    _constant_cache.clear();
    _constant_entries.clear();
    _owned_outputs.clear();
    _owned_inputs.clear();
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      std::fprintf(stderr,
                   "ACE_PHANTOM_ORDINARY_ERROR[CTX_CACHE_SYNC]: CUDA "
                   "synchronization failed after cache teardown: %s\n",
                   cudaGetErrorString(status));
    }
  }

  void PrepareInput(TENSOR* input, const char* name) {
    if (input == nullptr || name == nullptr) {
      Fail("PREPARE_INPUT_NULL", "input tensor and name must be non-null");
    }
    const size_t len = TENSOR_SIZE(input);
    if (len == 0 || len > _logical_slots) {
      Fail("PREPARE_INPUT_LENGTH", "input length %zu is outside [1, %zu]",
           len, _logical_slots);
    }
    std::vector<double> values(input->_vals, input->_vals + len);
    Plaintext plain;
    EncodeVector(&plain, values, 1, _manifest->_input_level,
                 "PREPARE_INPUT_ENCODE");
    auto cipher = std::make_unique<Ciphertext>();
    ProviderCall("PREPARE_INPUT_ENCRYPT", [&] {
      _public_key->encrypt_asymmetric(*_context, plain, *cipher);
    });
    cipher->SetNoiseScaleDeg(1);
    MarkCipher(cipher.get(), ObjectState::kLive);
    Io_set_input(name, 0, cipher.get());
    _owned_inputs.push_back(std::move(cipher));
  }

  void SetOutput(const char* name, size_t index, Ciphertext* cipher) {
    ValidateCipher(cipher, "SET_OUTPUT_SOURCE");
    if (name == nullptr) Fail("SET_OUTPUT_NAME", "output name is null");
    auto copy = std::make_unique<Ciphertext>(*cipher);
    MarkCipher(copy.get(), ObjectState::kLive);
    Io_set_output(name, index, copy.get());
    _owned_outputs.push_back(std::move(copy));
  }

  Ciphertext GetInput(const char* name, size_t index) {
    auto* cipher = static_cast<Ciphertext*>(Io_get_input(name, index));
    ValidateCipher(cipher, "GET_INPUT");
    return *cipher;
  }

  double* HandleOutput(const char* name, size_t index) {
    auto* cipher = static_cast<Ciphertext*>(Io_get_output(name, index));
    ValidateCipher(cipher, "HANDLE_OUTPUT");
    std::vector<double> values;
    Decrypt(cipher, values);
    auto* result = static_cast<double*>(std::malloc(values.size() * sizeof(double)));
    if (result == nullptr) Fail("HANDLE_OUTPUT_ALLOC", "host allocation failed");
    std::memcpy(result, values.data(), values.size() * sizeof(double));
    return result;
  }

  void EncodeFloat(Plaintext* plain, const float* input, size_t len,
                   SCALE_T degree, LEVEL_T level) {
    ValidateEncodeDestination(plain, input, len, "ENCODE_FLOAT");
    std::vector<double> values(input, input + len);
    EncodeVector(plain, values, degree, level, "ENCODE_FLOAT");
  }

  void EncodeDouble(Plaintext* plain, const double* input, size_t len,
                    SCALE_T degree, LEVEL_T level) {
    ValidateEncodeDestination(plain, input, len, "ENCODE_DOUBLE");
    std::vector<double> values(input, input + len);
    EncodeVector(plain, values, degree, level, "ENCODE_DOUBLE");
  }

  void EncodeComplex(Plaintext* plain, const DCMPLX* input, size_t len,
                     SCALE_T degree, LEVEL_T level) {
    ValidateEncodeDestination(plain, input, len, "ENCODE_COMPLEX");
    std::vector<std::complex<double>> values(input, input + len);
    EncodeVector(plain, values, degree, level, "ENCODE_COMPLEX");
  }

  void EncodeManifestConstant(Plaintext* plain, uint32_t entry_id) {
    RequireWritablePlain(plain, "ENCODE_MANIFEST_CONSTANT");
    ForgetBroadcastScalar(plain);
    const PHANTOM_CONSTANT_ENTRY& entry = FindConstantEntry(entry_id);
    const ConstantCacheKey key = ConstantKey(entry, "ENCODE_MANIFEST_CONSTANT");
    EncodeConstantPayload(plain, entry, "ENCODE_MANIFEST_CONSTANT");
    MarkPlain(plain, ObjectState::kLive);
    ValidateConstantPlain(*plain, entry, key, "ENCODE_MANIFEST_CONSTANT");
  }

  void LoadCachedConstant(Plaintext* plain, uint32_t entry_id) {
    RequireWritablePlain(plain, "LOAD_CACHED_CONSTANT");
    ForgetBroadcastScalar(plain);
    const PHANTOM_CONSTANT_ENTRY& entry = FindConstantEntry(entry_id);
    const ConstantCacheKey key = ConstantKey(entry, "LOAD_CACHED_CONSTANT");
    const auto cached = _constant_cache.find(key);
    if (cached == _constant_cache.end()) {
      Fail("CONSTANT_CACHE_MISS",
           "constant entry %u has no context-owned cached plaintext", entry_id);
    }
    ProviderCall("LOAD_CACHED_CONSTANT", [&] { *plain = cached->second; });
    MarkPlain(plain, ObjectState::kLive);
    ValidateConstantPlain(*plain, entry, key, "LOAD_CACHED_CONSTANT");
  }

  PHANTOM_SETUP_METRICS SetupMetrics() const { return _setup_metrics; }

  PHANTOM_BORROWED_RUNTIME BorrowRuntime() const {
    return {_context.get(), _encoder.get(), _secret_key.get(),
            _public_key.get(), _relin_key.get(), _galois_key.get()};
  }

  template <typename T>
  void EncodeMask(Plaintext* plain, T value, size_t len, SCALE_T degree,
                  LEVEL_T level, const char* diagnostic) {
    ValidateEncodeDestination(plain, &value, len, diagnostic);
    // ANT routes a one-element real mask through its scalar encoder, which
    // broadcasts the value across every logical CKKS slot.  Phantom's vector
    // encoder instead treats a one-element vector as a sparse one-slot value,
    // so preserve the runtime API's scalar semantics explicitly here.
    const size_t encoded_len = len == 1 ? _logical_slots : len;
    std::vector<double> values(encoded_len, static_cast<double>(value));
    EncodeVector(plain, values, degree, level, diagnostic);
    if (len == 1) {
      RememberBroadcastScalar(plain, static_cast<double>(value));
    }
  }

  void AddCipher(Ciphertext* result, Ciphertext* left, Ciphertext* right) {
    if (IsZero(left, "ADD_LEFT")) {
      ValidateCipher(right, "ADD_RIGHT");
      CopyCipher(result, right);
      return;
    }
    ValidateAddSub(left, right, "ADD_COMPAT");
    RequireWritableCipher(result, "ADD_DESTINATION");
    const double output_scale = left->scale();
    const size_t output_scale_degree = left->GetNoiseScaleDeg();
    ProviderCall("ADD_PROVIDER", [&] {
      if (result == left) {
        add_inplace(*_context, *result, *right);
      } else if (result == right) {
        add_inplace(*_context, *result, *left);
      } else {
        *result = *left;
        add_inplace(*_context, *result, *right);
      }
    });
    result->scale() = output_scale;
    result->SetNoiseScaleDeg(output_scale_degree);
    MarkCipher(result, ObjectState::kLive);
  }

  void SubCipher(Ciphertext* result, Ciphertext* left, Ciphertext* right) {
    ValidateAddSub(left, right, "SUB_COMPAT");
    RequireWritableCipher(result, "SUB_DESTINATION");
    const double output_scale = left->scale();
    const size_t output_scale_degree = left->GetNoiseScaleDeg();
    ProviderCall("SUB_PROVIDER", [&] {
      if (result == left) {
        result->scale() = right->scale();
        sub_inplace(*_context, *result, *right, false);
      } else if (result == right) {
        // Phantom's negate form computes right_argument - destination, which
        // preserves ACE's left-minus-right order without a repair copy.
        result->scale() = left->scale();
        sub_inplace(*_context, *result, *left, true);
      } else {
        *result = *left;
        result->scale() = right->scale();
        sub_inplace(*_context, *result, *right, false);
      }
    });
    result->scale() = output_scale;
    result->SetNoiseScaleDeg(output_scale_degree);
    MarkCipher(result, ObjectState::kLive);
  }

  void AddPlain(Ciphertext* result, Ciphertext* left, Plaintext* right) {
    ValidateCipherPlainAddSub(left, right, "ADD_PLAIN_COMPAT");
    RequireWritableCipher(result, "ADD_PLAIN_DESTINATION");
    const double output_scale = left->scale();
    const size_t output_scale_degree = left->GetNoiseScaleDeg();
    Plaintext aligned_plain;
    Plaintext* provider_plain = right;
    double broadcast_scalar = 0.0;
    if (FindBroadcastScalar(right, &broadcast_scalar)) {
      // A broadcast scalar is an additive value, not a fixed-point payload.
      // Re-encode it at the ciphertext's exact post-rescale scale, matching
      // Phantom's native add_const semantics and preventing coefficient-domain
      // addition from silently changing the scalar's decoded magnitude.
      EncodeBroadcast(&aligned_plain, broadcast_scalar, left->chain_index(),
                      left->scale(), "ADD_PLAIN_ALIGN_SCALAR");
      provider_plain = &aligned_plain;
    }
    ProviderCall("ADD_PLAIN_PROVIDER", [&] {
      if (result != left) *result = *left;
      add_plain_inplace(*_context, *result, *provider_plain);
    });
    result->scale() = output_scale;
    result->SetNoiseScaleDeg(output_scale_degree);
    MarkCipher(result, ObjectState::kLive);
  }

  void SubPlain(Ciphertext* result, Ciphertext* left, Plaintext* right) {
    ValidateCipherPlainAddSub(left, right, "SUB_PLAIN_COMPAT");
    RequireWritableCipher(result, "SUB_PLAIN_DESTINATION");
    const double output_scale = left->scale();
    const size_t output_scale_degree = left->GetNoiseScaleDeg();
    Plaintext aligned_plain;
    Plaintext* provider_plain = right;
    double broadcast_scalar = 0.0;
    if (FindBroadcastScalar(right, &broadcast_scalar)) {
      EncodeBroadcast(&aligned_plain, broadcast_scalar, left->chain_index(),
                      left->scale(), "SUB_PLAIN_ALIGN_SCALAR");
      provider_plain = &aligned_plain;
    }
    ProviderCall("SUB_PLAIN_PROVIDER", [&] {
      if (result != left) *result = *left;
      result->scale() = provider_plain->scale();
      sub_plain_inplace(*_context, *result, *provider_plain);
    });
    result->scale() = output_scale;
    result->SetNoiseScaleDeg(output_scale_degree);
    MarkCipher(result, ObjectState::kLive);
  }

  void AddScalar(Ciphertext* result, Ciphertext* left, double scalar,
                 bool subtract) {
    ValidateCipher(left, subtract ? "SUB_SCALAR_SOURCE" : "ADD_SCALAR_SOURCE");
    if (!std::isfinite(scalar)) {
      Fail(subtract ? "SUB_SCALAR_VALUE" : "ADD_SCALAR_VALUE",
           "scalar must be finite");
    }
    Plaintext plain;
    EncodeBroadcast(&plain, scalar, left->chain_index(), left->scale(),
                    subtract ? "SUB_SCALAR_ENCODE" : "ADD_SCALAR_ENCODE");
    if (subtract) {
      SubPlain(result, left, &plain);
    } else {
      AddPlain(result, left, &plain);
    }
  }

  void MultiplyCipher(Ciphertext* result, Ciphertext* left,
                      Ciphertext* right) {
    ValidateMultiply(left, right, "MUL_COMPAT");
    RequireWritableCipher(result, "MUL_DESTINATION");
    const std::int64_t output_degree = QueryScaleDegree(left, "MUL_LEFT_SCALE") +
                                       QueryScaleDegree(right, "MUL_RIGHT_SCALE");
    ProviderCall("MUL_PROVIDER", [&] {
      if (result == left) {
        multiply_inplace(*_context, *result, *right);
      } else if (result == right) {
        // Multiplication is commutative; mutating the original right operand
        // avoids replacing it before Phantom has consumed it.
        multiply_inplace(*_context, *result, *left);
      } else {
        *result = *left;
        multiply_inplace(*_context, *result, *right);
      }
    });
    if (result->size() != 3) {
      Fail("MUL_SIZE_TRANSITION", "size-2 x size-2 produced size %zu",
           result->size());
    }
    result->SetNoiseScaleDeg(static_cast<size_t>(output_degree));
    MarkCipher(result, ObjectState::kLive);
  }

  void MultiplyPlain(Ciphertext* result, Ciphertext* left, Plaintext* right) {
    ValidateCipher(left, "MUL_PLAIN_CIPHER");
    ValidatePlain(right, "MUL_PLAIN_PLAIN");
    if (left->chain_index() != right->chain_index()) {
      Fail("MUL_PLAIN_LEVEL", "cipher chain %zu differs from plain chain %zu",
           left->chain_index(), right->chain_index());
    }
    RequireWritableCipher(result, "MUL_PLAIN_DESTINATION");
    const size_t input_size = left->size();
    const std::int64_t output_degree =
        QueryScaleDegree(left, "MUL_PLAIN_CIPHER_SCALE") +
        QueryPlainScaleDegree(right, "MUL_PLAIN_PLAIN_SCALE");
    ProviderCall("MUL_PLAIN_PROVIDER", [&] {
      if (result != left) *result = *left;
      multiply_plain_inplace(*_context, *result, *right);
    });
    if (result->size() != input_size) {
      Fail("MUL_PLAIN_SIZE_TRANSITION", "ciphertext size changed from %zu to %zu",
           input_size, result->size());
    }
    result->SetNoiseScaleDeg(static_cast<size_t>(output_degree));
    MarkCipher(result, ObjectState::kLive);
  }

  void MultiplyScalar(Ciphertext* result, Ciphertext* left, double scalar) {
    ValidateCipher(left, "MUL_SCALAR_SOURCE");
    if (!std::isfinite(scalar)) Fail("MUL_SCALAR_VALUE", "scalar must be finite");
    Plaintext plain;
    EncodeBroadcast(&plain, scalar, left->chain_index(),
                    ScaleForDegree(1, _manifest->_scaling_modulus_bits),
                    "MUL_SCALAR_ENCODE");
    MultiplyPlain(result, left, &plain);
  }

  void Relinearize(Ciphertext* result, Ciphertext* source) {
    ValidateCipher(source, "RELIN_SOURCE");
    RequireWritableCipher(result, "RELIN_DESTINATION");
    if (source->size() != 3) {
      Fail("RELIN_SIZE", "relinearization requires size 3, observed %zu",
           source->size());
    }
    if ((_resource_flags & PHANTOM_RESOURCE_RELIN_KEY) == 0 ||
        !_relin_key || !_relin_key->is_generated()) {
      Fail("RELIN_KEY_MISSING", "the declared relinearization key is absent");
    }
    const size_t chain = source->chain_index();
    const double scale = source->scale();
    ProviderCall("RELIN_PROVIDER", [&] {
      if (result != source) *result = *source;
      relinearize_inplace(*_context, *result, *_relin_key);
    });
    if (result->size() != 2 || result->chain_index() != chain ||
        result->scale() != scale || !result->is_ntt_form()) {
      Fail("RELIN_METADATA", "relinearization violated its metadata contract");
    }
    MarkCipher(result, ObjectState::kLive);
  }

  void Rescale(Ciphertext* result, Ciphertext* source) {
    ValidateCipher(source, "RESCALE_SOURCE");
    RequireWritableCipher(result, "RESCALE_DESTINATION");
    const size_t source_q = ActiveQ(source, "RESCALE_SOURCE");
    if (source_q == 1) {
      Fail("RESCALE_BOTTOM_CHAIN", "cannot rescale a bottom-chain ciphertext");
    }
    const size_t source_size = source->size();
    ProviderCall("RESCALE_PROVIDER", [&] {
      if (result == source) {
        rescale_to_next_inplace(*_context, *result);
      } else {
        *result = rescale_to_next(*_context, *source);
      }
    });
    MarkCipher(result, ObjectState::kLive);
    if (ActiveQ(result, "RESCALE_RESULT") + 1 != source_q ||
        result->size() != source_size || !result->is_ntt_form()) {
      Fail("RESCALE_METADATA", "rescale did not drop exactly one data-Q modulus");
    }
    result->SetNoiseScaleDeg(
        static_cast<size_t>(QueryScaleDegree(result, "RESCALE_SCALE")));
  }

  void ModSwitch(Ciphertext* result, Ciphertext* source) {
    ValidateCipher(source, "MODSWITCH_SOURCE");
    RequireWritableCipher(result, "MODSWITCH_DESTINATION");
    const size_t source_q = ActiveQ(source, "MODSWITCH_SOURCE");
    if (source_q == 1) {
      Fail("MODSWITCH_BOTTOM_CHAIN",
           "cannot modulus-switch a bottom-chain ciphertext");
    }
    const size_t source_size = source->size();
    const double source_scale = source->scale();
    ProviderCall("MODSWITCH_PROVIDER", [&] {
      if (result != source) {
        // This order is deliberate: an out-of-place switch first establishes
        // independent source ownership and then converts the copy.
        *result = *source;
      }
      mod_switch_to_next_inplace(*_context, *result);
    });
    MarkCipher(result, ObjectState::kLive);
    if (ActiveQ(result, "MODSWITCH_RESULT") + 1 != source_q ||
        result->size() != source_size || result->scale() != source_scale ||
        !result->is_ntt_form()) {
      Fail("MODSWITCH_METADATA",
           "modulus switch did not drop exactly one data-Q modulus");
    }
  }

  void Rotate(Ciphertext* result, Ciphertext* source, std::int64_t step) {
    ValidateCipher(source, "ROTATE_SOURCE");
    RequireWritableCipher(result, "ROTATE_DESTINATION");
    if (source->size() != 2) {
      Fail("ROTATE_SIZE", "rotation requires ciphertext size 2, observed %zu",
           source->size());
    }
    const int normalized = NormalizeRotation(step, _logical_slots);
    if (normalized == 0) {
      CopyCipher(result, source);
      return;
    }
    if (_rotation_steps.count(normalized) == 0 || !_galois_key ||
        !_galois_key->is_generated()) {
      Fail("ROTATE_KEY_MISSING",
           "normalized rotation %d has no declared key", normalized);
    }
    ProviderCall("ROTATE_PROVIDER", [&] {
      if (result != source) *result = *source;
      rotate_inplace(*_context, *result, normalized, *_galois_key);
    });
    MarkCipher(result, ObjectState::kLive);
  }

  void Conjugate(Ciphertext* result, Ciphertext* source) {
    ValidateCipher(source, "CONJUGATE_SOURCE");
    RequireWritableCipher(result, "CONJUGATE_DESTINATION");
    if (source->size() != 2) {
      Fail("CONJUGATE_SIZE", "observed ciphertext size %zu, expected 2",
           source->size());
    }
    if ((_resource_flags & PHANTOM_RESOURCE_CONJUGATION_KEY) == 0 ||
        !_galois_key || !_galois_key->is_generated()) {
      Fail("CONJUGATE_KEY_MISSING",
           "observed resource flag=%d and generated key=%d, expected both 1",
           (_resource_flags & PHANTOM_RESOURCE_CONJUGATION_KEY) != 0,
           _galois_key != nullptr && _galois_key->is_generated());
    }
    const size_t source_q = ActiveQ(source, "CONJUGATE_SOURCE");
    const size_t source_chain = source->chain_index();
    const size_t source_size = source->size();
    const size_t source_scale_degree = source->GetNoiseScaleDeg();
    const double source_scale = source->scale();
    const bool source_ntt = source->is_ntt_form();
    ProviderCall("CONJUGATE_PROVIDER", [&] {
      if (result != source) *result = *source;
      complex_conjugate_inplace(*_context, *result, *_galois_key);
    });
    MarkCipher(result, ObjectState::kLive);
    const size_t result_q = ActiveQ(result, "CONJUGATE_RESULT");
    if (result_q != source_q || result->chain_index() != source_chain ||
        result->size() != source_size || result->scale() != source_scale ||
        result->GetNoiseScaleDeg() != source_scale_degree ||
        result->is_ntt_form() != source_ntt) {
      Fail("CONJUGATE_METADATA",
           "observed q=%zu chain=%zu size=%zu scale=%.17g scale_degree=%zu "
           "ntt=%d; expected q=%zu chain=%zu size=%zu scale=%.17g "
           "scale_degree=%zu ntt=%d",
           result_q, result->chain_index(), result->size(), result->scale(),
           result->GetNoiseScaleDeg(), result->is_ntt_form(), source_q,
           source_chain, source_size, source_scale, source_scale_degree,
           source_ntt);
    }
  }

  void RotateBatch(Ciphertext* outputs, Ciphertext* source,
                   const int32_t* steps, size_t count) {
    ValidateCipher(source, "ROTATE_BATCH_SOURCE");
    if (outputs == nullptr || steps == nullptr || count == 0) {
      Fail("ROTATE_BATCH_ARGUMENT",
           "outputs, ordered steps, and count must be non-empty");
    }
    for (size_t index = 0; index < count; ++index) {
      if (&outputs[index] == source) {
        Fail("ROTATE_BATCH_ALIAS",
             "source cannot overlap the rotate_batch output array");
      }
      RequireWritableCipher(&outputs[index], "ROTATE_BATCH_DESTINATION");
    }
    bool declared_batch = false;
    for (size_t batch = 0; batch < _resources->_rotation_batch_count; ++batch) {
      const size_t begin = _resources->_rotation_batch_offsets[batch];
      const size_t end = _resources->_rotation_batch_offsets[batch + 1];
      if (end - begin != count) continue;
      declared_batch = std::equal(steps, steps + count,
                                  _resources->_rotation_batch_steps + begin);
      if (declared_batch) break;
    }
    if (!declared_batch) {
      Fail("ROTATE_BATCH_RESOURCE",
           "observed ordered batch count %zu, expected one of %zu "
           "compiler-declared batches",
           count, _resources->_rotation_batch_count);
    }
    std::vector<int> provider_steps;
    provider_steps.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      provider_steps.push_back(
          NormalizeRotation(steps[index], _logical_slots));
    }
    std::vector<Ciphertext> provider_outputs;
    PhantomGaloisKey empty_key;
    const PhantomGaloisKey& key = _galois_key ? *_galois_key : empty_key;
    ProviderCall("ROTATE_BATCH_PROVIDER", [&] {
      rotate_batch(*_context, *source, provider_steps, key,
                   provider_outputs);
    });
    if (provider_outputs.size() != count) {
      Fail("ROTATE_BATCH_COUNT", "provider returned %zu outputs, expected %zu",
           provider_outputs.size(), count);
    }
    for (size_t index = 0; index < count; ++index) {
      outputs[index] = std::move(provider_outputs[index]);
      MarkCipher(&outputs[index], ObjectState::kLive);
      ValidateCipher(&outputs[index], "ROTATE_BATCH_RESULT");
      const size_t output_q = ActiveQ(&outputs[index], "ROTATE_BATCH_RESULT");
      const size_t source_q = ActiveQ(source, "ROTATE_BATCH_SOURCE");
      if (output_q != source_q ||
          outputs[index].size() != source->size() ||
          outputs[index].scale() != source->scale() ||
          outputs[index].GetNoiseScaleDeg() != source->GetNoiseScaleDeg() ||
          outputs[index].chain_index() != source->chain_index() ||
          outputs[index].is_ntt_form() != source->is_ntt_form()) {
        Fail("ROTATE_BATCH_METADATA",
             "output %zu observed q=%zu chain=%zu size=%zu scale=%.17g "
             "scale_degree=%zu ntt=%d; expected q=%zu chain=%zu size=%zu "
             "scale=%.17g scale_degree=%zu ntt=%d",
             index, output_q, outputs[index].chain_index(),
             outputs[index].size(), outputs[index].scale(),
             outputs[index].GetNoiseScaleDeg(), outputs[index].is_ntt_form(),
             source_q, source->chain_index(), source->size(), source->scale(),
             source->GetNoiseScaleDeg(), source->is_ntt_form());
      }
    }
  }

  void RaiseModulus(Ciphertext* result, Ciphertext* source,
                    size_t target_q_count) {
    RequireWritableCipher(result, "RAISE_MOD_DESTINATION");
    if (result == source) {
      Fail("RAISE_MOD_ALIAS",
           "observed identical source/destination, expected distinct objects");
    }
    CipherState(source, "RAISE_MOD_SOURCE", false);
    if (source->size() != 2) {
      Fail("RAISE_MOD_SIZE", "observed ciphertext size %zu, expected 2",
           source->size());
    }
    ValidateCipher(source, "RAISE_MOD_SOURCE");
    if ((_resource_flags & PHANTOM_RESOURCE_RAISE_MOD) == 0) {
      Fail("RAISE_MOD_RESOURCE",
           "observed raise_mod resource flag 0, expected 1");
    }
    const size_t observed_q = ActiveQ(source, "RAISE_MOD_SOURCE");
    const size_t expected_chain = AceLevelToChainIndex(
        1, _manifest->_data_q_count, _first_data_chain_index);
    if (observed_q != 1 || source->chain_index() != expected_chain) {
      Fail("RAISE_MOD_SOURCE_LEVEL",
           "observed chain %zu with %zu active Qs, expected bottom chain %zu "
           "with 1 active Q",
           source->chain_index(), observed_q, expected_chain);
    }
    if (target_q_count != _manifest->_data_q_count) {
      Fail("RAISE_MOD_TARGET",
           "observed target_q_count %zu, expected full data-Q count %zu",
           target_q_count, _manifest->_data_q_count);
    }
    ProviderCall("RAISE_MOD_PROVIDER", [&] {
      raise_modulus(*_context, *source, target_q_count, *result);
    });
    MarkCipher(result, ObjectState::kLive);
    ValidateCipher(result, "RAISE_MOD_RESULT");
    const size_t result_q = ActiveQ(result, "RAISE_MOD_RESULT");
    const size_t expected_result_chain = AceLevelToChainIndex(
        target_q_count, _manifest->_data_q_count, _first_data_chain_index);
    if (result_q != target_q_count ||
        result->chain_index() != expected_result_chain ||
        result->size() != source->size() ||
        result->scale() != source->scale() ||
        result->GetNoiseScaleDeg() != source->GetNoiseScaleDeg() ||
        result->is_ntt_form() != source->is_ntt_form()) {
      Fail("RAISE_MOD_METADATA",
           "observed q=%zu chain=%zu size=%zu scale=%.17g scale_degree=%zu "
           "ntt=%d; expected q=%zu chain=%zu size=%zu scale=%.17g "
           "scale_degree=%zu ntt=%d",
           result_q, result->chain_index(), result->size(), result->scale(),
           result->GetNoiseScaleDeg(), result->is_ntt_form(), target_q_count,
           expected_result_chain, source->size(), source->scale(),
           source->GetNoiseScaleDeg(), source->is_ntt_form());
    }
  }

  void MultiplyMonomial(Ciphertext* result, Ciphertext* source,
                        uint32_t power) {
    ValidateCipher(source, "MUL_MONO_SOURCE");
    RequireWritableCipher(result, "MUL_MONO_DESTINATION");
    if ((_resource_flags & PHANTOM_RESOURCE_MONOMIALS) == 0 ||
        !std::binary_search(_resources->_monomial_powers,
                            _resources->_monomial_powers +
                                _resources->_monomial_count,
                            power)) {
      Fail("MUL_MONO_RESOURCE",
           "observed normalized monomial power %u absent, expected one of %zu "
           "compiler-declared powers",
           power, _resources->_monomial_count);
    }
    const size_t source_q = ActiveQ(source, "MUL_MONO_SOURCE");
    const size_t source_chain = source->chain_index();
    const size_t source_size = source->size();
    const size_t source_scale_degree = source->GetNoiseScaleDeg();
    const double source_scale = source->scale();
    const bool source_ntt = source->is_ntt_form();
    ProviderCall("MUL_MONO_PROVIDER", [&] {
      if (result == source) {
        multiply_by_monomial_inplace(*_context, *result, power);
      } else {
        multiply_by_monomial(*_context, *source, power, *result);
      }
    });
    MarkCipher(result, ObjectState::kLive);
    const size_t result_q = ActiveQ(result, "MUL_MONO_RESULT");
    if (result_q != source_q || result->chain_index() != source_chain ||
        result->size() != source_size ||
        result->scale() != source_scale ||
        result->GetNoiseScaleDeg() != source_scale_degree ||
        result->is_ntt_form() != source_ntt) {
      Fail("MUL_MONO_METADATA",
           "observed q=%zu chain=%zu size=%zu scale=%.17g scale_degree=%zu "
           "ntt=%d; expected q=%zu chain=%zu size=%zu scale=%.17g "
           "scale_degree=%zu ntt=%d",
           result_q, result->chain_index(), result->size(), result->scale(),
           result->GetNoiseScaleDeg(), result->is_ntt_form(), source_q,
           source_chain, source_size, source_scale, source_scale_degree,
           source_ntt);
    }
  }

  void CopyCipher(Ciphertext* result, Ciphertext* source) {
    RequireWritableCipher(result, "COPY_DESTINATION");
    const ObjectState state = CipherState(source, "COPY_SOURCE", true);
    if (result == source) return;
    if (state == ObjectState::kZero) {
      result->release();
      result->zero_ciph();
      MarkCipher(result, ObjectState::kZero);
      return;
    }
    ValidateCipher(source, "COPY_SOURCE");
    ProviderCall("COPY_PROVIDER", [&] { *result = *source; });
    MarkCipher(result, ObjectState::kLive);
  }

  void ZeroCipher(Ciphertext* result) {
    RequireWritableCipher(result, "ZERO_DESTINATION");
    result->release();
    result->zero_ciph();
    MarkCipher(result, ObjectState::kZero);
  }

  void FreeCipher(Ciphertext* cipher) {
    const ObjectState state = CipherState(cipher, "FREE_CIPHER", true, true);
    if (state == ObjectState::kFreed) {
      Fail("DOUBLE_FREE_CIPHER", "ciphertext was already freed");
    }
    if (state != ObjectState::kZero) ValidateCipher(cipher, "FREE_CIPHER");
    cipher->release();
    MarkCipher(cipher, ObjectState::kFreed);
  }

  void FreePlain(Plaintext* plain) {
    const ObjectState state = PlainState(plain, "FREE_PLAIN", true, true);
    if (state == ObjectState::kFreed) {
      Fail("DOUBLE_FREE_PLAIN", "plaintext was already freed");
    }
    ValidatePlain(plain, "FREE_PLAIN");
    plain->release();
    ForgetBroadcastScalar(plain);
    MarkPlain(plain, ObjectState::kFreed);
  }

  void FreeCipherArray(Ciphertext* array, size_t count) {
    if (array == nullptr || count == 0) {
      Fail("FREE_ARRAY_ARGUMENT", "array must be non-null and non-empty");
    }
    // Preflight every element before releasing any of them.
    for (size_t index = 0; index < count; ++index) {
      const ObjectState state =
          CipherState(&array[index], "FREE_ARRAY", true, true);
      if (state == ObjectState::kFreed) {
        Fail("DOUBLE_FREE_ARRAY", "array element %zu was already freed", index);
      }
      if (state != ObjectState::kZero) ValidateCipher(&array[index], "FREE_ARRAY");
    }
    for (size_t index = 0; index < count; ++index) {
      array[index].release();
      MarkCipher(&array[index], ObjectState::kFreed);
    }
  }

  void RegisterCipherLifetime(Ciphertext* cipher) {
    if (cipher == nullptr) {
      Fail("REGISTER_CIPHER_NULL", "ciphertext pointer is null");
    }
    std::lock_guard<std::mutex> lock(_state_mutex);
    _cipher_states[cipher] =
        cipher->size() == 0 ? ObjectState::kUninitialized
                            : ObjectState::kLive;
  }

  void RegisterPlainLifetime(Plaintext* plain) {
    if (plain == nullptr) {
      Fail("REGISTER_PLAIN_NULL", "plaintext pointer is null");
    }
    std::lock_guard<std::mutex> lock(_state_mutex);
    _broadcast_scalars.erase(plain);
    _plain_states[plain] =
        plain->poly_modulus_degree() == 0 ? ObjectState::kUninitialized
                                          : ObjectState::kLive;
  }

  void RegisterCipherArrayLifetime(Ciphertext* array, size_t count) {
    if (array == nullptr || count == 0) {
      Fail("REGISTER_CIPHER_ARRAY_ARGUMENT",
           "array must be non-null and non-empty");
    }
    for (size_t index = 0; index < count; ++index) {
      RegisterCipherLifetime(&array[index]);
    }
  }

  void RegisterPlainArrayLifetime(Plaintext* array, size_t count) {
    if (array == nullptr || count == 0) {
      Fail("REGISTER_PLAIN_ARRAY_ARGUMENT",
           "array must be non-null and non-empty");
    }
    for (size_t index = 0; index < count; ++index) {
      RegisterPlainLifetime(&array[index]);
    }
  }

  double QueryRawScale(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return cipher->scale();
  }

  std::int64_t QueryScaleDegree(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    try {
      return ScaleDegree(cipher->scale(), _manifest->_scaling_modulus_bits);
    } catch (const std::exception& error) {
      Fail(diagnostic, "invalid ciphertext scale %.17g: %s", cipher->scale(),
           error.what());
    }
  }

  size_t QueryAceLevel(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return ChainIndexToAceLevel(cipher->chain_index(), _manifest->_data_q_count,
                                _first_data_chain_index);
  }

  size_t QueryActiveQ(Ciphertext* cipher, const char* diagnostic) {
    return ActiveQ(cipher, diagnostic);
  }

  size_t QueryChain(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return cipher->chain_index();
  }

  size_t QuerySlots(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return _logical_slots;
  }

  size_t QuerySize(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return cipher->size();
  }

  bool QueryNtt(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return cipher->is_ntt_form();
  }

  double QueryPlainRawScale(Plaintext* plain, const char* diagnostic) {
    ValidatePlain(plain, diagnostic);
    return plain->scale();
  }

  std::int64_t QueryPlainScaleDegree(Plaintext* plain,
                                     const char* diagnostic) {
    ValidatePlain(plain, diagnostic);
    try {
      return ScaleDegree(plain->scale(), _manifest->_scaling_modulus_bits);
    } catch (const std::exception& error) {
      Fail(diagnostic, "invalid plaintext scale %.17g: %s", plain->scale(),
           error.what());
    }
  }

  size_t QueryPlainLevel(Plaintext* plain, const char* diagnostic) {
    ValidatePlain(plain, diagnostic);
    return ChainIndexToAceLevel(plain->chain_index(),
                                _manifest->_data_q_count,
                                _first_data_chain_index);
  }

  size_t QueryPlainChain(Plaintext* plain, const char* diagnostic) {
    ValidatePlain(plain, diagnostic);
    return plain->chain_index();
  }

  size_t QueryPlainSlots(Plaintext* plain, const char* diagnostic) {
    ValidatePlain(plain, diagnostic);
    return _logical_slots;
  }

  void Decrypt(Ciphertext* cipher, std::vector<double>& values) {
    ValidateCipher(cipher, "DECRYPT_SOURCE");
    Plaintext plain;
    ProviderCall("DECRYPT_PROVIDER", [&] {
      _secret_key->decrypt(*_context, *cipher, plain);
      _encoder->decode(*_context, plain, values);
    });
  }

  void DecryptComplex(Ciphertext* cipher,
                      std::vector<std::complex<double>>& values) {
    ValidateCipher(cipher, "DECRYPT_COMPLEX_SOURCE");
    Plaintext plain;
    ProviderCall("DECRYPT_COMPLEX_PROVIDER", [&] {
      _secret_key->decrypt(*_context, *cipher, plain);
      _encoder->decode(*_context, plain, values);
    });
  }

  void Decode(Plaintext* plain, std::vector<double>& values) {
    ValidatePlain(plain, "DECODE_SOURCE");
    ProviderCall("DECODE_PROVIDER",
                 [&] { _encoder->decode(*_context, *plain, values); });
  }

  void DecodeComplex(Plaintext* plain,
                     std::vector<std::complex<double>>& values) {
    ValidatePlain(plain, "DECODE_COMPLEX_SOURCE");
    ProviderCall("DECODE_COMPLEX_PROVIDER",
                 [&] { _encoder->decode(*_context, *plain, values); });
  }

  void EncryptPlain(Ciphertext* cipher, Plaintext* plain) {
    RequireWritableCipher(cipher, "ENCRYPT_DESTINATION");
    ValidatePlain(plain, "ENCRYPT_PLAIN");
    ProviderCall("ENCRYPT_PROVIDER", [&] {
      _public_key->encrypt_asymmetric(*_context, *plain, *cipher);
    });
    cipher->SetNoiseScaleDeg(
        static_cast<size_t>(QueryPlainScaleDegree(plain, "ENCRYPT_SCALE")));
    MarkCipher(cipher, ObjectState::kLive);
    ValidateCipher(cipher, "ENCRYPT_RESULT");
  }

private:
  PHANTOM_CONTEXT()
      : _manifest(Get_phantom_context_manifest()),
        _resources(Get_phantom_resource_manifest()),
        _constants(Get_phantom_constant_manifest()),
        _logical_slots(_manifest == nullptr ? 0 : _manifest->_logical_slots) {
    const auto setup_start = std::chrono::steady_clock::now();
    const size_t setup_free_before = FreeDeviceBytes("CONTEXT_SETUP_MEMORY");
    ValidateProgramManifest();
    _resource_flags = _resources->_flags;

    phantom::EncryptionParameters parameters(phantom::scheme_type::ckks);
    parameters.set_poly_modulus_degree(_manifest->_poly_degree);
    // The AIR parameter sizes have ANT semantics, so reproduce ANT's balanced
    // prime chain explicitly.  Phantom's generic Create() searches only below
    // 2^bits; using it here makes every exact rescale drift in the same
    // direction and materially changes deep generated EvalMod arithmetic.
    const std::vector<phantom::arith::Modulus> coefficient_moduli =
        BuildAntCompatibleCoefficientModuli(*_manifest);
    _ordered_coefficient_moduli.reserve(coefficient_moduli.size());
    for (const auto& modulus : coefficient_moduli) {
      _ordered_coefficient_moduli.push_back(modulus.value());
    }
    parameters.set_coeff_modulus(coefficient_moduli);
    parameters.set_special_modulus_size(_manifest->_special_p_count);
    parameters.set_secret_key_hamming_weight(_manifest->_hamming_weight);
    parameters.set_sparse_slots(_logical_slots);

    _context = std::make_unique<PhantomContext>(parameters);
    ValidateConstructedContext();
    _encoder = std::make_unique<PhantomCKKSEncoder>(*_context);
    if (_encoder->physical_slot_count() != _logical_slots ||
        _encoder->logical_slot_count() != _logical_slots) {
      Fail("CONTEXT_SLOTS",
           "encoder slots do not match the compiler context manifest");
    }
    _secret_key = std::make_unique<PhantomSecretKey>(*_context);
    _public_key =
        std::make_unique<PhantomPublicKey>(_secret_key->gen_publickey(*_context));
    if ((_resource_flags & PHANTOM_RESOURCE_RELIN_KEY) != 0) {
      _relin_key = std::make_unique<PhantomRelinKey>(
          _secret_key->gen_relinkey(*_context));
    }

    std::vector<int> rotation_steps;
    for (size_t index = 0; index < _resources->_rotation_count; ++index) {
      const int normalized =
          NormalizeRotation(_resources->_rotation_steps[index], _logical_slots);
      if (normalized != 0) _rotation_steps.insert(normalized);
    }
    rotation_steps.assign(_rotation_steps.begin(), _rotation_steps.end());
    const bool needs_conjugation =
        (_resource_flags & PHANTOM_RESOURCE_CONJUGATION_KEY) != 0;
    if (!rotation_steps.empty() || needs_conjugation) {
      _galois_key = std::make_unique<PhantomGaloisKey>(
          _secret_key->create_galois_keys_from_steps(*_context,
                                                     rotation_steps,
                                                     needs_conjugation));
    }
    SynchronizeDevice("CONTEXT_SETUP_SYNC");
    const size_t setup_free_after = FreeDeviceBytes("CONTEXT_SETUP_MEMORY");
    _setup_metrics._context_and_key_setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      setup_start)
            .count();
    _setup_metrics._context_and_key_device_bytes =
        DeviceBytesUsed(setup_free_before, setup_free_after);

    // Constant payload encoding and upload are context setup, never graph work.
    BuildConstantCache();
  }

  static void SynchronizeDevice(const char* diagnostic) {
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      Fail(diagnostic, "CUDA synchronization failed: %s",
           cudaGetErrorString(status));
    }
  }

  static size_t FreeDeviceBytes(const char* diagnostic) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
      Fail(diagnostic, "CUDA memory query failed: %s",
           cudaGetErrorString(status));
    }
    return free_bytes;
  }

  static uint64_t DeviceBytesUsed(size_t before, size_t after) {
    return before > after ? static_cast<uint64_t>(before - after) : 0;
  }

  static bool IsSha256(const char* value) {
    if (value == nullptr || std::strlen(value) != 64) return false;
    for (size_t index = 0; index < 64; ++index) {
      const unsigned char byte = static_cast<unsigned char>(value[index]);
      if (!std::isdigit(byte) && (byte < 'a' || byte > 'f')) return false;
    }
    return true;
  }

  void ValidateConstantManifest() const {
    constexpr uint32_t constant_schema_version = 1;
    constexpr uint32_t context_schema_version = 1;
    constexpr uint32_t resource_schema_version = 3;
    if (_constants == nullptr) {
      Fail("CONSTANT_MANIFEST_NULL", "constant manifest is null");
    }
    if (_constants->_schema_version != constant_schema_version ||
        _constants->_context_schema_version != context_schema_version ||
        _constants->_resource_schema_version != resource_schema_version) {
      Fail("CONSTANT_MANIFEST_SCHEMA",
           "unsupported constant/context/resource manifest schema");
    }
    if (!IsSha256(_constants->_context_manifest_sha256)) {
      Fail("CONSTANT_CONTEXT_SHA",
           "context manifest SHA-256 must be 64 lowercase hexadecimal digits");
    }
    if ((_constants->_entry_count == 0) != (_constants->_entries == nullptr)) {
      Fail("CONSTANT_MANIFEST_ENTRIES",
           "constant entry count and array must agree exactly");
    }
    if (_constants->_entry_count != 0 &&
        (_resource_flags & PHANTOM_RESOURCE_COMPLEX_PLAINTEXT) == 0) {
      Fail("CONSTANT_RESOURCE",
           "complex plaintext constants require the declared resource bit");
    }
  }

  ConstantCacheKey ConstantKey(const PHANTOM_CONSTANT_ENTRY& entry,
                               const char* diagnostic) const {
    size_t expected_chain = 0;
    double expected_scale = 0.0;
    try {
      expected_chain = AceLevelToChainIndex(
          entry._ace_level, _manifest->_data_q_count,
          _first_data_chain_index);
      expected_scale = ScaleForDegree(entry._scale_degree,
                                      _manifest->_scaling_modulus_bits);
    } catch (const std::exception& error) {
      Fail(diagnostic, "constant entry %u: %s", entry._entry_id,
           error.what());
    }
    if (entry._chain_index != expected_chain ||
        entry._raw_scale != expected_scale) {
      Fail(diagnostic,
           "constant entry %u level/chain/scale tuple is inconsistent",
           entry._entry_id);
    }
    const auto& context_data = _context->get_context_data(expected_chain);
    if (std::log2(expected_scale) >=
        context_data.total_coeff_modulus_bit_count()) {
      Fail(diagnostic,
           "constant entry %u scale does not fit its declared chain",
           entry._entry_id);
    }
    return ConstantCacheKey{
        entry._constant_id, context_data.parms().parms_id(), expected_chain,
        expected_scale, entry._element_type, entry._slot_count};
  }

  void ValidateConstantEntry(const PHANTOM_CONSTANT_ENTRY& entry) const {
    if (entry._element_type != PHANTOM_CONSTANT_COMPLEX_F64) {
      Fail("CONSTANT_ELEMENT_TYPE",
           "constant entry %u has unsupported element type %u",
           entry._entry_id, entry._element_type);
    }
    if (entry._scale_degree <= 0) {
      Fail("CONSTANT_SCALE_DEGREE",
           "constant entry %u scale degree must be positive",
           entry._entry_id);
    }
    if (entry._slot_count == 0 || entry._slot_count > _logical_slots ||
        entry._slot_count > std::numeric_limits<size_t>::max() / 2 ||
        entry._interleaved_value_count != entry._slot_count * 2 ||
        entry._interleaved_values == nullptr) {
      Fail("CONSTANT_PAYLOAD_SHAPE",
           "constant entry %u has an invalid interleaved complex payload",
           entry._entry_id);
    }
    if (entry._symbol == nullptr || entry._symbol[0] == '\0' ||
        !IsSha256(entry._payload_sha256) ||
        !IsSha256(entry._cache_key_sha256)) {
      Fail("CONSTANT_IDENTITY",
           "constant entry %u has an invalid symbol or SHA-256",
           entry._entry_id);
    }
    for (size_t index = 0; index < entry._interleaved_value_count; ++index) {
      if (!std::isfinite(entry._interleaved_values[index])) {
        Fail("CONSTANT_PAYLOAD_VALUE",
             "constant entry %u payload value %zu is not finite",
             entry._entry_id, index);
      }
    }
    (void)ConstantKey(entry, "CONSTANT_METADATA");
  }

  std::vector<std::complex<double>> ConstantValues(
      const PHANTOM_CONSTANT_ENTRY& entry) const {
    std::vector<std::complex<double>> values;
    values.reserve(entry._slot_count);
    for (size_t index = 0; index < entry._slot_count; ++index) {
      values.emplace_back(entry._interleaved_values[index * 2],
                          entry._interleaved_values[index * 2 + 1]);
    }
    return values;
  }

  void EncodeConstantPayload(Plaintext* plain,
                             const PHANTOM_CONSTANT_ENTRY& entry,
                             const char* diagnostic) {
    const std::vector<std::complex<double>> values = ConstantValues(entry);
    ProviderCall(diagnostic, [&] {
      _encoder->encode(*_context, values, entry._raw_scale, *plain,
                       entry._chain_index);
    });
  }

  void ValidateConstantPlain(const Plaintext& plain,
                             const PHANTOM_CONSTANT_ENTRY& entry,
                             const ConstantCacheKey& key,
                             const char* diagnostic) const {
    ValidatePlainMetadata(plain, diagnostic);
    if (plain.parms_id() != key._parameter_fingerprint ||
        plain.chain_index() != key._chain_index ||
        plain.scale() != key._raw_scale ||
        key._constant_id != entry._constant_id ||
        key._element_type != entry._element_type ||
        key._slot_count != entry._slot_count) {
      Fail(diagnostic,
           "constant entry %u does not match its full cache-key tuple",
           entry._entry_id);
    }
  }

  const PHANTOM_CONSTANT_ENTRY& FindConstantEntry(uint32_t entry_id) const {
    const auto found = _constant_entries.find(entry_id);
    if (found == _constant_entries.end()) {
      Fail("CONSTANT_ENTRY_ID", "constant entry id %u is not declared",
           entry_id);
    }
    return *found->second;
  }

  void BuildConstantCache() {
    const auto cache_start = std::chrono::steady_clock::now();
    const size_t cache_free_before = FreeDeviceBytes("CONSTANT_CACHE_MEMORY");
    ValidateConstantManifest();
    uint64_t attributed_payload_host_bytes = 0;
    uint64_t logical_device_bytes = 0;

    for (size_t index = 0; index < _constants->_entry_count; ++index) {
      const PHANTOM_CONSTANT_ENTRY& entry = _constants->_entries[index];
      if (index > std::numeric_limits<uint32_t>::max() ||
          entry._entry_id != static_cast<uint32_t>(index)) {
        Fail("CONSTANT_ENTRY_ID",
             "constant entry id %u must equal its manifest index %zu",
             entry._entry_id, index);
      }
      ValidateConstantEntry(entry);
      if (entry._interleaved_value_count >
          std::numeric_limits<uint64_t>::max() / sizeof(double)) {
        Fail("CONSTANT_CACHE_MEMORY",
             "constant entry %u payload byte count overflowed",
             entry._entry_id);
      }
      const uint64_t payload_bytes =
          static_cast<uint64_t>(entry._interleaved_value_count) *
          sizeof(double);
      if (payload_bytes >
          std::numeric_limits<uint64_t>::max() -
              attributed_payload_host_bytes) {
        Fail("CONSTANT_CACHE_MEMORY",
             "constant payload host-memory attribution overflowed");
      }
      attributed_payload_host_bytes += payload_bytes;
      if (!_constant_entries.emplace(entry._entry_id, &entry).second) {
        Fail("CONSTANT_ENTRY_ID", "duplicate constant entry id %u",
             entry._entry_id);
      }

      const ConstantCacheKey key = ConstantKey(entry, "CONSTANT_CACHE_KEY");
      Plaintext cached;
      EncodeConstantPayload(&cached, entry, "CONSTANT_CACHE_ENCODE");
      ValidateConstantPlain(cached, entry, key, "CONSTANT_CACHE_ENCODE");
      const uint64_t poly_degree = cached.poly_modulus_degree();
      const uint64_t coefficient_moduli = cached.coeff_modulus_size();
      if (coefficient_moduli == 0 ||
          poly_degree > std::numeric_limits<uint64_t>::max() /
                            coefficient_moduli ||
          poly_degree * coefficient_moduli >
              std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)) {
        Fail("CONSTANT_CACHE_MEMORY",
             "constant entry %u logical device byte count overflowed",
             entry._entry_id);
      }
      const uint64_t entry_device_bytes =
          poly_degree * coefficient_moduli * sizeof(uint64_t);
      if (entry_device_bytes >
          std::numeric_limits<uint64_t>::max() - logical_device_bytes) {
        Fail("CONSTANT_CACHE_MEMORY",
             "plaintext-cache logical device byte count overflowed");
      }
      if (!_constant_cache.emplace(key, std::move(cached)).second) {
        Fail("CONSTANT_CACHE_KEY",
             "constant entry %u duplicates a full cache-key tuple",
             entry._entry_id);
      }
      logical_device_bytes += entry_device_bytes;
    }

    SynchronizeDevice("CONSTANT_CACHE_SYNC");
    const size_t cache_free_after = FreeDeviceBytes("CONSTANT_CACHE_MEMORY");
    _setup_metrics._plaintext_cache_setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      cache_start)
            .count();
    _setup_metrics._plaintext_cache_device_bytes =
        DeviceBytesUsed(cache_free_before, cache_free_after);
    _setup_metrics._plaintext_cache_logical_device_bytes = logical_device_bytes;
    _setup_metrics._plaintext_cache_entries = _constant_cache.size();
    // Attribute immutable compiler-owned payload inputs to setup alongside the
    // context-owned cache/map metadata that consumes them.
    constexpr uint64_t metadata_bytes_per_entry =
        sizeof(std::pair<const ConstantCacheKey, Plaintext>) +
        sizeof(std::pair<const uint32_t, const PHANTOM_CONSTANT_ENTRY*>);
    if (_constant_cache.size() >
        std::numeric_limits<uint64_t>::max() / metadata_bytes_per_entry) {
      Fail("CONSTANT_CACHE_MEMORY",
           "plaintext-cache host metadata byte count overflowed");
    }
    const uint64_t metadata_bytes =
        static_cast<uint64_t>(_constant_cache.size()) *
        metadata_bytes_per_entry;
    if (metadata_bytes > std::numeric_limits<uint64_t>::max() -
                             attributed_payload_host_bytes) {
      Fail("CONSTANT_CACHE_MEMORY",
           "plaintext-cache total host byte count overflowed");
    }
    _setup_metrics._plaintext_cache_host_bytes =
        attributed_payload_host_bytes + metadata_bytes;
  }

  void ValidateProgramManifest() const {
    constexpr uint32_t context_schema_version = 1;
    constexpr uint32_t resource_schema_version = 3;
    constexpr uint64_t known_resource_flags =
        PHANTOM_RESOURCE_RELIN_KEY | PHANTOM_RESOURCE_ROTATION_KEYS |
        PHANTOM_RESOURCE_CONJUGATION_KEY | PHANTOM_RESOURCE_ROTATE_BATCH |
        PHANTOM_RESOURCE_RAISE_MOD | PHANTOM_RESOURCE_MONOMIALS |
        PHANTOM_RESOURCE_COMPLEX_PLAINTEXT |
        PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE;
    if (_manifest == nullptr) {
      Fail("CONTEXT_MANIFEST_NULL", "context manifest is null");
    }
    if (_resources == nullptr) {
      Fail("RESOURCE_MANIFEST_NULL", "resource manifest is null");
    }
    if (_manifest->_schema_version != context_schema_version ||
        _manifest->_resource_schema_version != resource_schema_version ||
        _resources->_schema_version != resource_schema_version ||
        _resources->_context_schema_version != context_schema_version) {
      Fail("MANIFEST_SCHEMA", "unsupported context/resource manifest schema");
    }
    if (_manifest->_packing != PHANTOM_PACKING_FULL) {
      Fail("CONTEXT_PACKING", "only compiler-declared full packing is supported");
    }
    try {
      ace::phantom::ordinary::ValidateLogicalSlots(
          _manifest->_poly_degree,
          ace::phantom::ordinary::PackingConvention::kFull,
          _manifest->_logical_slots);
    } catch (const std::exception& error) {
      Fail("CONTEXT_SLOTS", "%s", error.what());
    }
    if (_manifest->_data_q_count == 0 ||
        _manifest->_data_q_bit_sizes == nullptr ||
        _manifest->_special_p_count == 0 ||
        _manifest->_special_p_bit_sizes == nullptr) {
      Fail("CONTEXT_MODULI", "Q and P arrays must both be non-empty");
    }
    if (_manifest->_first_modulus_bits == 0 ||
        _manifest->_first_modulus_bits > 60 ||
        _manifest->_scaling_modulus_bits == 0 ||
        _manifest->_scaling_modulus_bits > 60 ||
        _manifest->_data_q_bit_sizes[0] !=
            _manifest->_first_modulus_bits) {
      Fail("CONTEXT_Q_BITS", "invalid first/scaling data-Q metadata");
    }
    for (size_t index = 0; index < _manifest->_data_q_count; ++index) {
      const uint32_t expected =
          index == 0 ? _manifest->_first_modulus_bits
                     : _manifest->_scaling_modulus_bits;
      if (_manifest->_data_q_bit_sizes[index] != expected) {
        Fail("CONTEXT_Q_BITS",
             "data-Q bit size %zu disagrees with compiler prime policy",
             index);
      }
    }
    for (size_t index = 0; index < _manifest->_special_p_count; ++index) {
      const uint32_t bits = _manifest->_special_p_bit_sizes[index];
      if (bits == 0 || bits > 60) {
        Fail("CONTEXT_P_BITS", "special-P bit size %zu is invalid", index);
      }
      if (bits != _manifest->_special_p_bit_sizes[0]) {
        Fail("CONTEXT_P_BITS",
             "ANT-compatible special-P sizes must be uniform");
      }
    }
    if (_manifest->_input_level == 0 ||
        _manifest->_input_level > _manifest->_data_q_count ||
        _manifest->_q_part_count == 0 ||
        _manifest->_q_part_count > _manifest->_data_q_count ||
        _manifest->_hamming_weight == 0 ||
        _manifest->_hamming_weight > _manifest->_poly_degree) {
      Fail("CONTEXT_METADATA", "invalid input-level, Q-part, or key metadata");
    }
    if (_manifest->_security_level != 0 &&
        _manifest->_security_level != 128 &&
        _manifest->_security_level != 192 &&
        _manifest->_security_level != 256) {
      Fail("CONTEXT_SECURITY", "unsupported security setting %u",
           _manifest->_security_level);
    }
    phantom::arith::sec_level_type provider_security =
        phantom::arith::sec_level_type::none;
    if (_manifest->_security_level == 128) {
      provider_security = phantom::arith::sec_level_type::tc128;
    } else if (_manifest->_security_level == 192) {
      provider_security = phantom::arith::sec_level_type::tc192;
    } else if (_manifest->_security_level == 256) {
      provider_security = phantom::arith::sec_level_type::tc256;
    }
    size_t total_modulus_bits = 0;
    for (size_t index = 0; index < _manifest->_data_q_count; ++index) {
      total_modulus_bits += _manifest->_data_q_bit_sizes[index];
    }
    for (size_t index = 0; index < _manifest->_special_p_count; ++index) {
      total_modulus_bits += _manifest->_special_p_bit_sizes[index];
    }
    const int maximum_modulus_bits = phantom::arith::CoeffModulus::MaxBitCount(
        _manifest->_poly_degree, provider_security);
    if (maximum_modulus_bits == 0 ||
        total_modulus_bits > static_cast<size_t>(maximum_modulus_bits)) {
      Fail("CONTEXT_SECURITY",
           "Q+P modulus budget %zu exceeds the provider limit %d for "
           "security setting %u",
           total_modulus_bits, maximum_modulus_bits,
           _manifest->_security_level);
    }
    if ((_resources->_flags & ~known_resource_flags) != 0) {
      Fail("RESOURCE_FLAGS", "unknown resource bits 0x%llx",
           static_cast<unsigned long long>(_resources->_flags &
                                           ~known_resource_flags));
    }
    if ((_resources->_flags &
         PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE) != 0) {
      Fail("NATIVE_BOOTSTRAP_FORBIDDEN",
           "generated primitive bootstrap cannot request native bootstrap "
           "precomputation");
    }
    const bool has_rotation_flag =
        (_resources->_flags & PHANTOM_RESOURCE_ROTATION_KEYS) != 0;
    if (has_rotation_flag != (_resources->_rotation_count != 0) ||
        (_resources->_rotation_count != 0 &&
         _resources->_rotation_steps == nullptr)) {
      Fail("RESOURCE_ROTATIONS",
           "rotation flag, count, and array must agree exactly");
    }
    std::set<int> declared_rotations;
    for (size_t index = 0; index < _resources->_rotation_count; ++index) {
      const int declared = _resources->_rotation_steps[index];
      int normalized = 0;
      try {
        normalized = NormalizeRotation(declared, _logical_slots);
      } catch (const std::exception& error) {
        Fail("RESOURCE_ROTATIONS", "%s", error.what());
      }
      if (declared == 0 || normalized != declared ||
          !declared_rotations.insert(declared).second) {
        Fail("RESOURCE_ROTATIONS",
             "rotation step %d is zero, non-canonical, or duplicated",
             declared);
      }
    }
    const bool has_batch_flag =
        (_resources->_flags & PHANTOM_RESOURCE_ROTATE_BATCH) != 0;
    if (has_batch_flag != (_resources->_rotation_batch_count != 0) ||
        (_resources->_rotation_batch_count != 0 &&
         (_resources->_rotation_batch_offsets == nullptr ||
          _resources->_rotation_batch_steps == nullptr))) {
      Fail("RESOURCE_ROTATE_BATCH",
           "rotate_batch flag, count, offsets, and steps must agree exactly");
    }
    if (_resources->_rotation_batch_count != 0 &&
        _resources->_rotation_batch_offsets[0] != 0) {
      Fail("RESOURCE_ROTATE_BATCH", "first rotate_batch offset must be zero");
    }
    for (size_t batch = 0; batch < _resources->_rotation_batch_count; ++batch) {
      const size_t begin = _resources->_rotation_batch_offsets[batch];
      const size_t end = _resources->_rotation_batch_offsets[batch + 1];
      if (end <= begin) {
        Fail("RESOURCE_ROTATE_BATCH",
             "rotation batch %zu must be non-empty", batch);
      }
      for (size_t index = begin; index < end; ++index) {
        const int normalized =
            NormalizeRotation(_resources->_rotation_batch_steps[index],
                              _logical_slots);
        if (normalized != 0 && declared_rotations.count(normalized) == 0) {
          Fail("RESOURCE_ROTATE_BATCH",
               "rotation batch step %d has no canonical ordinary key",
               _resources->_rotation_batch_steps[index]);
        }
      }
    }
    const bool has_monomial_flag =
        (_resources->_flags & PHANTOM_RESOURCE_MONOMIALS) != 0;
    if (has_monomial_flag != (_resources->_monomial_count != 0) ||
        (_resources->_monomial_count != 0 &&
         _resources->_monomial_powers == nullptr)) {
      Fail("RESOURCE_MONOMIALS",
           "monomial flag, count, and powers must agree exactly");
    }
    uint32_t previous_power = 0;
    const uint64_t monomial_period =
        static_cast<uint64_t>(_manifest->_poly_degree) * 2;
    for (size_t index = 0; index < _resources->_monomial_count; ++index) {
      const uint32_t power = _resources->_monomial_powers[index];
      if (power >= monomial_period || (index != 0 && power <= previous_power)) {
        Fail("RESOURCE_MONOMIALS",
             "monomial powers must be canonical, sorted, and unique");
      }
      previous_power = power;
    }
  }

  void ValidateConstructedContext() {
    if (_context->context_data_.size() != _manifest->_data_q_count + 1) {
      Fail("CONTEXT_CHAIN_LENGTH", "observed %zu entries, expected %zu",
           _context->context_data_.size(), _manifest->_data_q_count + 1);
    }
    const auto& key_moduli =
        _context->get_context_data(0).parms().coeff_modulus();
    if (key_moduli.size() !=
        _manifest->_data_q_count + _manifest->_special_p_count) {
      Fail("CONTEXT_KEY_MODULI", "observed %zu Q+P moduli, expected %zu",
           key_moduli.size(),
           _manifest->_data_q_count + _manifest->_special_p_count);
    }
    if (key_moduli.size() != _ordered_coefficient_moduli.size()) {
      Fail("CONTEXT_MODULI", "constructed modulus identity count changed");
    }
    for (size_t index = 0; index < key_moduli.size(); ++index) {
      if (key_moduli[index].value() != _ordered_coefficient_moduli[index]) {
        Fail("CONTEXT_MODULI",
             "constructed modulus %zu changed value or Q/P order", index);
      }
    }
    for (size_t index = 0; index < _manifest->_data_q_count; ++index) {
      if (RequestedPrimeSize(key_moduli[index].value()) !=
          _manifest->_data_q_bit_sizes[index]) {
        Fail("CONTEXT_Q_BITS",
             "data-Q modulus %zu has requested size %u, expected %u", index,
             RequestedPrimeSize(key_moduli[index].value()),
             _manifest->_data_q_bit_sizes[index]);
      }
    }
    for (size_t index = 0; index < _manifest->_special_p_count; ++index) {
      const size_t key_index = _manifest->_data_q_count + index;
      if (RequestedPrimeSize(key_moduli[key_index].value()) !=
          _manifest->_special_p_bit_sizes[index]) {
        Fail("CONTEXT_P_BITS",
             "special-P modulus %zu has requested size %u, expected %u", index,
             RequestedPrimeSize(key_moduli[key_index].value()),
             _manifest->_special_p_bit_sizes[index]);
      }
    }

    _first_data_chain_index = std::numeric_limits<size_t>::max();
    for (size_t chain = 0; chain < _context->context_data_.size(); ++chain) {
      const auto& moduli =
          _context->get_context_data(chain).parms().coeff_modulus();
      if (moduli.size() != _manifest->_data_q_count) continue;
      bool matches = true;
      for (size_t index = 0; index < moduli.size(); ++index) {
        matches = matches &&
                  moduli[index].value() == _ordered_coefficient_moduli[index];
      }
      if (matches) {
        if (_first_data_chain_index != std::numeric_limits<size_t>::max()) {
          Fail("CONTEXT_LEVEL_MAP", "data-Q tower has multiple first entries");
        }
        _first_data_chain_index = chain;
      }
    }
    if (_first_data_chain_index == std::numeric_limits<size_t>::max()) {
      Fail("CONTEXT_LEVEL_MAP", "full data-Q entry was not found");
    }
    for (size_t level = 1; level <= _manifest->_data_q_count; ++level) {
      const size_t chain = AceLevelToChainIndex(
          level, _manifest->_data_q_count, _first_data_chain_index);
      const auto& moduli =
          _context->get_context_data(chain).parms().coeff_modulus();
      if (moduli.size() != level) {
        Fail("CONTEXT_LEVEL_MAP",
             "ACE level %zu maps to chain %zu with %zu active Q moduli",
             level, chain, moduli.size());
      }
      for (size_t index = 0; index < moduli.size(); ++index) {
        if (moduli[index].value() != _ordered_coefficient_moduli[index]) {
          Fail("CONTEXT_LEVEL_MAP",
               "chain %zu data-Q modulus %zu disagrees with the manifest",
               chain, index);
        }
      }
    }
  }

  template <typename T>
  void ValidateEncodeDestination(Plaintext* plain, const T* input, size_t len,
                                 const char* diagnostic) {
    RequireWritablePlain(plain, diagnostic);
    if (input == nullptr) Fail(diagnostic, "input pointer is null");
    if (len == 0 || len > _logical_slots) {
      Fail(diagnostic, "encoding length %zu is outside [1, %zu]", len,
           _logical_slots);
    }
  }

  template <typename T>
  void EncodeVector(Plaintext* plain, const std::vector<T>& values,
                    SCALE_T degree, LEVEL_T level,
                    const char* diagnostic) {
    ForgetBroadcastScalar(plain);
    const std::int64_t integral_degree = CheckedScaleDegree(degree, diagnostic);
    size_t chain = 0;
    double scale = 0.0;
    try {
      chain = AceLevelToChainIndex(static_cast<size_t>(level),
                                   _manifest->_data_q_count,
                                   _first_data_chain_index);
      scale = ScaleForDegree(integral_degree,
                             _manifest->_scaling_modulus_bits);
    } catch (const std::exception& error) {
      Fail(diagnostic, "%s", error.what());
    }
    const auto& context_data = _context->get_context_data(chain);
    if (std::log2(scale) >= context_data.total_coeff_modulus_bit_count()) {
      Fail(diagnostic,
           "scale degree %lld does not fit ACE level %d (chain %zu)",
           static_cast<long long>(integral_degree), level, chain);
    }
    ProviderCall(diagnostic, [&] {
      _encoder->encode(*_context, values, scale, *plain, chain);
    });
    MarkPlain(plain, ObjectState::kLive);
    ValidatePlain(plain, diagnostic);
  }

  void EncodeBroadcast(Plaintext* plain, double value, size_t chain,
                       double scale, const char* diagnostic) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
      Fail(diagnostic, "broadcast value and scale must be finite");
    }
    ChainIndexToActiveQ(chain, _manifest->_data_q_count,
                        _first_data_chain_index);
    std::vector<double> values(_logical_slots, value);
    ProviderCall(diagnostic, [&] {
      _encoder->encode(*_context, values, scale, *plain, chain);
    });
  }

  ObjectState CipherState(Ciphertext* cipher, const char* diagnostic,
                          bool allow_zero, bool allow_freed = false) {
    if (cipher == nullptr) Fail(diagnostic, "ciphertext pointer is null");
    std::lock_guard<std::mutex> lock(_state_mutex);
    auto found = _cipher_states.find(cipher);
    if (found != _cipher_states.end()) {
      if (found->second == ObjectState::kFreed && !allow_freed) {
        Fail("USE_AFTER_FREE_CIPHER", "%s used a freed ciphertext", diagnostic);
      }
      if (!allow_zero && found->second == ObjectState::kZero) {
        Fail(diagnostic, "zero sentinel is not a valid ciphertext operand");
      }
      if (found->second == ObjectState::kUninitialized) {
        Fail(diagnostic, "ciphertext is uninitialized");
      }
      return found->second;
    }
    if (cipher->size() == 0) Fail(diagnostic, "ciphertext is uninitialized");
    _cipher_states.emplace(cipher, ObjectState::kLive);
    return ObjectState::kLive;
  }

  ObjectState PlainState(Plaintext* plain, const char* diagnostic,
                         bool allow_uninitialized, bool allow_freed = false) {
    if (plain == nullptr) Fail(diagnostic, "plaintext pointer is null");
    std::lock_guard<std::mutex> lock(_state_mutex);
    auto found = _plain_states.find(plain);
    if (found != _plain_states.end()) {
      if (found->second == ObjectState::kFreed && !allow_freed) {
        Fail("USE_AFTER_FREE_PLAIN", "%s used a freed plaintext", diagnostic);
      }
      if (found->second == ObjectState::kUninitialized) {
        Fail(diagnostic, "plaintext is uninitialized");
      }
      return found->second;
    }
    if (plain->poly_modulus_degree() == 0)
      Fail(diagnostic, "plaintext is uninitialized");
    _plain_states.emplace(plain, ObjectState::kLive);
    return ObjectState::kLive;
  }

  void MarkCipher(Ciphertext* cipher, ObjectState state) {
    std::lock_guard<std::mutex> lock(_state_mutex);
    _cipher_states[cipher] = state;
  }

  void MarkPlain(Plaintext* plain, ObjectState state) {
    std::lock_guard<std::mutex> lock(_state_mutex);
    _plain_states[plain] = state;
  }

  void RememberBroadcastScalar(Plaintext* plain, double value) {
    std::lock_guard<std::mutex> lock(_state_mutex);
    _broadcast_scalars[plain] = value;
  }

  void ForgetBroadcastScalar(Plaintext* plain) {
    std::lock_guard<std::mutex> lock(_state_mutex);
    _broadcast_scalars.erase(plain);
  }

  bool FindBroadcastScalar(Plaintext* plain, double* value) {
    std::lock_guard<std::mutex> lock(_state_mutex);
    const auto found = _broadcast_scalars.find(plain);
    if (found == _broadcast_scalars.end()) return false;
    *value = found->second;
    return true;
  }

  void RequireWritableCipher(Ciphertext* cipher, const char* diagnostic) {
    if (cipher == nullptr) Fail(diagnostic, "destination is null");
    std::lock_guard<std::mutex> lock(_state_mutex);
    auto found = _cipher_states.find(cipher);
    if (found != _cipher_states.end() &&
        found->second == ObjectState::kFreed) {
      Fail("USE_AFTER_FREE_DESTINATION", "%s targets a freed ciphertext",
           diagnostic);
    }
  }

  void RequireWritablePlain(Plaintext* plain, const char* diagnostic) {
    if (plain == nullptr) Fail(diagnostic, "destination is null");
    std::lock_guard<std::mutex> lock(_state_mutex);
    auto found = _plain_states.find(plain);
    if (found != _plain_states.end() && found->second == ObjectState::kFreed) {
      Fail("USE_AFTER_FREE_PLAIN_DESTINATION", "%s targets a freed plaintext",
           diagnostic);
    }
  }

  bool IsZero(Ciphertext* cipher, const char* diagnostic) {
    return CipherState(cipher, diagnostic, true) == ObjectState::kZero;
  }

  void ValidateCipher(Ciphertext* cipher, const char* diagnostic) {
    CipherState(cipher, diagnostic, false);
    if (cipher->size() != 2 && cipher->size() != 3) {
      Fail(diagnostic, "ciphertext size %zu is outside {2, 3}", cipher->size());
    }
    if (!cipher->is_ntt_form()) {
      Fail(diagnostic, "observed NTT form 0, expected ordinary CKKS NTT form 1");
    }
    const size_t q_count = ChainIndexToActiveQ(
        cipher->chain_index(), _manifest->_data_q_count,
        _first_data_chain_index);
    const auto& context_data =
        _context->get_context_data(cipher->chain_index());
    if (context_data.parms().coeff_modulus().size() != q_count ||
        cipher->parms_id() != context_data.parms().parms_id() ||
        cipher->coeff_modulus_size() != q_count ||
        cipher->poly_modulus_degree() != _manifest->_poly_degree ||
        cipher->data() == nullptr || !std::isfinite(cipher->scale()) ||
        cipher->scale() <= 0.0) {
      Fail(diagnostic,
           "invalid ciphertext metadata: level=%zu chain=%zu q=%zu size=%zu "
           "ntt=%d scale=%.17g",
           ChainIndexToAceLevel(cipher->chain_index(),
                                _manifest->_data_q_count,
                                _first_data_chain_index),
           cipher->chain_index(),
           q_count, cipher->size(), cipher->is_ntt_form(), cipher->scale());
    }
  }

  void ValidatePlain(Plaintext* plain, const char* diagnostic) {
    PlainState(plain, diagnostic, false);
    ValidatePlainMetadata(*plain, diagnostic);
  }

  void ValidatePlainMetadata(const Plaintext& plain,
                             const char* diagnostic) const {
    const size_t q_count = ChainIndexToActiveQ(
        plain.chain_index(), _manifest->_data_q_count,
        _first_data_chain_index);
    const auto& context_data = _context->get_context_data(plain.chain_index());
    if (context_data.parms().coeff_modulus().size() != q_count ||
        plain.parms_id() != context_data.parms().parms_id() ||
        plain.coeff_modulus_size() != q_count ||
        plain.poly_modulus_degree() != _manifest->_poly_degree ||
        plain.data() == nullptr || !std::isfinite(plain.scale()) ||
        plain.scale() <= 0.0) {
      Fail(diagnostic,
           "invalid ordinary Q plaintext metadata: chain=%zu q=%zu scale=%.17g",
           plain.chain_index(), q_count, plain.scale());
    }
  }

  size_t ActiveQ(Ciphertext* cipher, const char* diagnostic) {
    ValidateCipher(cipher, diagnostic);
    return ChainIndexToActiveQ(cipher->chain_index(),
                               _manifest->_data_q_count,
                               _first_data_chain_index);
  }

  void ValidateAddSub(Ciphertext* left, Ciphertext* right,
                      const char* diagnostic) {
    ValidateCipher(left, diagnostic);
    ValidateCipher(right, diagnostic);
    const std::int64_t left_scale_degree =
        QueryScaleDegree(left, diagnostic);
    const std::int64_t right_scale_degree =
        QueryScaleDegree(right, diagnostic);
    if (left->chain_index() != right->chain_index() ||
        left_scale_degree != right_scale_degree ||
        left->size() != right->size() ||
        left->is_ntt_form() != right->is_ntt_form()) {
      Fail(diagnostic,
           "logical compatibility failed: lhs(level=%zu chain=%zu "
           "scale=%.17g scale_degree=%lld size=%zu ntt=%d) "
           "rhs(level=%zu chain=%zu scale=%.17g scale_degree=%lld size=%zu "
           "ntt=%d)",
           ChainIndexToAceLevel(left->chain_index(),
                                _manifest->_data_q_count,
                                _first_data_chain_index),
           left->chain_index(),
           left->scale(), static_cast<long long>(left_scale_degree),
           left->size(), left->is_ntt_form(),
           ChainIndexToAceLevel(right->chain_index(),
                                _manifest->_data_q_count,
                                _first_data_chain_index),
           right->chain_index(),
           right->scale(), static_cast<long long>(right_scale_degree),
           right->size(), right->is_ntt_form());
    }
  }

  void ValidateCipherPlainAddSub(Ciphertext* left, Plaintext* right,
                                 const char* diagnostic) {
    ValidateCipher(left, diagnostic);
    ValidatePlain(right, diagnostic);
    const std::int64_t left_scale_degree =
        QueryScaleDegree(left, diagnostic);
    const std::int64_t right_scale_degree =
        QueryPlainScaleDegree(right, diagnostic);
    if (left->chain_index() != right->chain_index() ||
        left_scale_degree != right_scale_degree) {
      Fail(diagnostic,
           "logical cipher/plain compatibility failed: cipher(chain=%zu "
           "scale=%.17g scale_degree=%lld) plain(chain=%zu scale=%.17g "
           "scale_degree=%lld)",
           left->chain_index(), left->scale(),
           static_cast<long long>(left_scale_degree), right->chain_index(),
           right->scale(), static_cast<long long>(right_scale_degree));
    }
  }

  void ValidateMultiply(Ciphertext* left, Ciphertext* right,
                        const char* diagnostic) {
    ValidateCipher(left, diagnostic);
    ValidateCipher(right, diagnostic);
    if (left->chain_index() != right->chain_index() || left->size() != 2 ||
        right->size() != 2 || !left->is_ntt_form() ||
        !right->is_ntt_form()) {
      Fail(diagnostic,
           "multiply requires same chain and two size-2 NTT ciphertexts; "
           "lhs(chain=%zu size=%zu ntt=%d) rhs(chain=%zu size=%zu ntt=%d)",
           left->chain_index(), left->size(), left->is_ntt_form(),
           right->chain_index(), right->size(), right->is_ntt_form());
    }
  }

  const PHANTOM_CONTEXT_MANIFEST* _manifest = nullptr;
  const PHANTOM_RESOURCE_MANIFEST* _resources = nullptr;
  const PHANTOM_CONSTANT_MANIFEST* _constants = nullptr;
  size_t _logical_slots = 0;
  size_t _first_data_chain_index = 0;
  uint64_t _resource_flags = 0;
  std::unique_ptr<PhantomContext> _context;
  std::unique_ptr<PhantomCKKSEncoder> _encoder;
  std::unique_ptr<PhantomSecretKey> _secret_key;
  std::unique_ptr<PhantomPublicKey> _public_key;
  std::unique_ptr<PhantomRelinKey> _relin_key;
  std::unique_ptr<PhantomGaloisKey> _galois_key;
  std::set<int> _rotation_steps;
  std::vector<uint64_t> _ordered_coefficient_moduli;
  std::map<ConstantCacheKey, Plaintext> _constant_cache;
  std::unordered_map<uint32_t, const PHANTOM_CONSTANT_ENTRY*>
      _constant_entries;
  PHANTOM_SETUP_METRICS _setup_metrics{};
  std::vector<std::unique_ptr<Ciphertext>> _owned_inputs;
  std::vector<std::unique_ptr<Ciphertext>> _owned_outputs;
  std::mutex _state_mutex;
  std::unordered_map<const void*, ObjectState> _cipher_states;
  std::unordered_map<const void*, ObjectState> _plain_states;
  std::unordered_map<const void*, double> _broadcast_scalars;

  static PHANTOM_CONTEXT* _instance;
};

PHANTOM_CONTEXT* PHANTOM_CONTEXT::_instance = nullptr;

}  // namespace

void Prepare_context() {
  Init_rtlib_timing();
  Io_init();
  PHANTOM_CONTEXT::Initialize();
}

void Finalize_context() {
  PHANTOM_CONTEXT::Finalize();
  Io_fini();
}

void Prepare_input(TENSOR* input, const char* name) {
  PHANTOM_CONTEXT::Context()->PrepareInput(input, name);
}

void Prepare_input_dup(TENSOR* input, const char* name) {
  PHANTOM_CONTEXT::Context()->PrepareInput(input, name);
}

double* Handle_output(const char* name) {
  return PHANTOM_CONTEXT::Context()->HandleOutput(name, 0);
}

void Phantom_set_output_data(const char* name, size_t index, CIPHER data) {
  PHANTOM_CONTEXT::Context()->SetOutput(name, index, data);
}

CIPHERTEXT Phantom_get_input_data(const char* name, size_t index) {
  return PHANTOM_CONTEXT::Context()->GetInput(name, index);
}

void Phantom_encode_float(PLAIN plain, float* input, size_t len, SCALE_T scale,
                          LEVEL_T level) {
  PHANTOM_CONTEXT::Context()->EncodeFloat(plain, input, len, scale, level);
}

void Phantom_encode_double(PLAIN plain, const double* input, size_t len,
                           SCALE_T scale, LEVEL_T level) {
  PHANTOM_CONTEXT::Context()->EncodeDouble(plain, input, len, scale, level);
}

void Phantom_encode_dcmplx(PLAIN plain, const DCMPLX* input, size_t len,
                           SCALE_T scale, LEVEL_T level) {
  PHANTOM_CONTEXT::Context()->EncodeComplex(plain, input, len, scale, level);
}

void Phantom_encode_manifest_constant(PLAIN plain, uint32_t entry_id) {
  PHANTOM_CONTEXT::Context()->EncodeManifestConstant(plain, entry_id);
}

void Phantom_load_cached_constant(PLAIN plain, uint32_t entry_id) {
  PHANTOM_CONTEXT::Context()->LoadCachedConstant(plain, entry_id);
}

PHANTOM_SETUP_METRICS Phantom_get_setup_metrics() {
  return PHANTOM_CONTEXT::Context()->SetupMetrics();
}

PHANTOM_BORROWED_RUNTIME Phantom_borrow_runtime() {
  return PHANTOM_CONTEXT::Context()->BorrowRuntime();
}

void Phantom_encode_float_cst_lvl(PLAIN plain, float* input, size_t len,
                                  SCALE_T scale, int level) {
  Phantom_encode_float(plain, input, len, scale, level);
}

void Phantom_encode_float_mask(PLAIN plain, float input, size_t len,
                               SCALE_T scale, LEVEL_T level) {
  PHANTOM_CONTEXT::Context()->EncodeMask(plain, input, len, scale, level,
                                         "ENCODE_FLOAT_MASK");
}

void Phantom_encode_double_mask(PLAIN plain, double input, size_t len,
                                SCALE_T scale, LEVEL_T level) {
  PHANTOM_CONTEXT::Context()->EncodeMask(plain, input, len, scale, level,
                                         "ENCODE_DOUBLE_MASK");
}

void Phantom_encode_float_mask_cst_lvl(PLAIN plain, float input, size_t len,
                                       SCALE_T scale, int level) {
  Phantom_encode_float_mask(plain, input, len, scale, level);
}

void Phantom_add_ciph(CIPHER result, CIPHER left, CIPHER right) {
  PHANTOM_CONTEXT::Context()->AddCipher(result, left, right);
}

void Phantom_sub_ciph(CIPHER result, CIPHER left, CIPHER right) {
  PHANTOM_CONTEXT::Context()->SubCipher(result, left, right);
}

void Phantom_add_plain(CIPHER result, CIPHER left, PLAIN right) {
  PHANTOM_CONTEXT::Context()->AddPlain(result, left, right);
}

void Phantom_sub_plain(CIPHER result, CIPHER left, PLAIN right) {
  PHANTOM_CONTEXT::Context()->SubPlain(result, left, right);
}

void Phantom_add_const(CIPHER result, CIPHER left, double right) {
  PHANTOM_CONTEXT::Context()->AddScalar(result, left, right, false);
}

void Phantom_sub_const(CIPHER result, CIPHER left, double right) {
  PHANTOM_CONTEXT::Context()->AddScalar(result, left, right, true);
}

void Phantom_mul_ciph(CIPHER result, CIPHER left, CIPHER right) {
  PHANTOM_CONTEXT::Context()->MultiplyCipher(result, left, right);
}

void Phantom_mul_plain(CIPHER result, CIPHER left, PLAIN right) {
  PHANTOM_CONTEXT::Context()->MultiplyPlain(result, left, right);
}

void Phantom_mul_ciph_const(CIPHER result, CIPHER left, double right) {
  PHANTOM_CONTEXT::Context()->MultiplyScalar(result, left, right);
}

void Phantom_rotate(CIPHER result, CIPHER source, int step) {
  PHANTOM_CONTEXT::Context()->Rotate(result, source, step);
}

void Phantom_conjugate(CIPHER result, CIPHER source) {
  PHANTOM_CONTEXT::Context()->Conjugate(result, source);
}

void Phantom_rotate_batch(CIPHER outputs, CIPHER source,
                          const int32_t* steps, size_t count) {
  PHANTOM_CONTEXT::Context()->RotateBatch(outputs, source, steps, count);
}

void Phantom_raise_mod(CIPHER result, CIPHER source,
                       uint32_t target_q_count) {
  PHANTOM_CONTEXT::Context()->RaiseModulus(result, source, target_q_count);
}

void Phantom_mul_mono(CIPHER result, CIPHER source, uint32_t power) {
  PHANTOM_CONTEXT::Context()->MultiplyMonomial(result, source, power);
}

void Phantom_rescale(CIPHER result, CIPHER source) {
  PHANTOM_CONTEXT::Context()->Rescale(result, source);
}

void Phantom_mod_switch(CIPHER result, CIPHER source) {
  PHANTOM_CONTEXT::Context()->ModSwitch(result, source);
}

void Phantom_relin(CIPHER result, CIPHER3 source) {
  PHANTOM_CONTEXT::Context()->Relinearize(result, source);
}

void Phantom_copy(CIPHER result, CIPHER source) {
  PHANTOM_CONTEXT::Context()->CopyCipher(result, source);
}

void Phantom_zero(CIPHER result) {
  PHANTOM_CONTEXT::Context()->ZeroCipher(result);
}

void Phantom_register_ciph_lifetime(CIPHER cipher) {
  PHANTOM_CONTEXT::Context()->RegisterCipherLifetime(cipher);
}

void Phantom_register_plain_lifetime(PLAIN plain) {
  PHANTOM_CONTEXT::Context()->RegisterPlainLifetime(plain);
}

void Phantom_register_ciph_array_lifetime(CIPHER array, size_t count) {
  PHANTOM_CONTEXT::Context()->RegisterCipherArrayLifetime(array, count);
}

void Phantom_register_plain_array_lifetime(PLAIN array, size_t count) {
  PHANTOM_CONTEXT::Context()->RegisterPlainArrayLifetime(array, count);
}

void Phantom_free_ciph(CIPHER cipher) {
  PHANTOM_CONTEXT::Context()->FreeCipher(cipher);
}

void Phantom_free_plain(PLAIN plain) {
  PHANTOM_CONTEXT::Context()->FreePlain(plain);
}

void Phantom_free_ciph_array(CIPHER array, size_t count) {
  PHANTOM_CONTEXT::Context()->FreeCipherArray(array, count);
}

SCALE_T Phantom_raw_scale(CIPHER value) {
  return PHANTOM_CONTEXT::Context()->QueryRawScale(value, "QUERY_RAW_SCALE");
}

SCALE_T Phantom_scale_degree(CIPHER value) {
  return static_cast<SCALE_T>(
      PHANTOM_CONTEXT::Context()->QueryScaleDegree(value, "QUERY_SCALE_DEGREE"));
}

LEVEL_T Phantom_ace_level(CIPHER value) {
  return static_cast<LEVEL_T>(
      PHANTOM_CONTEXT::Context()->QueryAceLevel(value, "QUERY_ACE_LEVEL"));
}

LEVEL_T Phantom_active_q_count(CIPHER value) {
  return static_cast<LEVEL_T>(
      PHANTOM_CONTEXT::Context()->QueryActiveQ(value, "QUERY_ACTIVE_Q"));
}

LEVEL_T Phantom_chain_index(CIPHER value) {
  return static_cast<LEVEL_T>(
      PHANTOM_CONTEXT::Context()->QueryChain(value, "QUERY_CHAIN_INDEX"));
}

size_t Phantom_slots(CIPHER value) {
  return PHANTOM_CONTEXT::Context()->QuerySlots(value, "QUERY_SLOTS");
}

size_t Phantom_ciphertext_size(CIPHER value) {
  return PHANTOM_CONTEXT::Context()->QuerySize(value, "QUERY_SIZE");
}

bool Phantom_is_ntt(CIPHER value) {
  return PHANTOM_CONTEXT::Context()->QueryNtt(value, "QUERY_NTT");
}

SCALE_T Phantom_plain_raw_scale(PLAIN value) {
  return PHANTOM_CONTEXT::Context()->QueryPlainRawScale(value,
                                                        "QUERY_PLAIN_SCALE");
}

SCALE_T Phantom_plain_scale_degree(PLAIN value) {
  return static_cast<SCALE_T>(PHANTOM_CONTEXT::Context()->QueryPlainScaleDegree(
      value, "QUERY_PLAIN_SCALE_DEGREE"));
}

LEVEL_T Phantom_plain_ace_level(PLAIN value) {
  return static_cast<LEVEL_T>(PHANTOM_CONTEXT::Context()->QueryPlainLevel(
      value, "QUERY_PLAIN_LEVEL"));
}

LEVEL_T Phantom_plain_active_q_count(PLAIN value) {
  return Phantom_plain_ace_level(value);
}

LEVEL_T Phantom_plain_chain_index(PLAIN value) {
  return static_cast<LEVEL_T>(PHANTOM_CONTEXT::Context()->QueryPlainChain(
      value, "QUERY_PLAIN_CHAIN"));
}

size_t Phantom_plain_slots(PLAIN value) {
  return PHANTOM_CONTEXT::Context()->QueryPlainSlots(value,
                                                      "QUERY_PLAIN_SLOTS");
}

bool Phantom_plain_is_ntt(PLAIN value) {
  PHANTOM_CONTEXT::Context()->QueryPlainLevel(value, "QUERY_PLAIN_NTT");
  return true;
}

void Phantom_decode_float(PLAIN plain, std::vector<double>& output) {
  PHANTOM_CONTEXT::Context()->Decode(plain, output);
}

void Phantom_decode_dcmplx(PLAIN plain,
                           std::vector<std::complex<double>>& output) {
  PHANTOM_CONTEXT::Context()->DecodeComplex(plain, output);
}

void Phantom_encrypt_plain(CIPHER cipher, PLAIN plain) {
  PHANTOM_CONTEXT::Context()->EncryptPlain(cipher, plain);
}

void Phantom_decrypt_dcmplx(
    CIPHER cipher, std::vector<std::complex<double>>& output) {
  PHANTOM_CONTEXT::Context()->DecryptComplex(cipher, output);
}

void Dump_ciph(CIPHER cipher, size_t start, size_t len) {
  std::vector<double> values;
  PHANTOM_CONTEXT::Context()->Decrypt(cipher, values);
  const size_t end = std::min(values.size(), start + len);
  for (size_t index = start; index < end; ++index) {
    std::cout << values[index] << ' ';
  }
  std::cout << std::endl;
}

void Dump_plain(PLAIN plain, size_t start, size_t len) {
  std::vector<double> values;
  PHANTOM_CONTEXT::Context()->Decode(plain, values);
  const size_t end = std::min(values.size(), start + len);
  for (size_t index = start; index < end; ++index) {
    std::cout << values[index] << ' ';
  }
  std::cout << std::endl;
}

void Dump_cipher_msg(const char* name, CIPHER cipher, uint32_t len) {
  std::cout << '[' << name << "]: ";
  Dump_ciph(cipher, 0, len);
}

void Dump_plain_msg(const char* name, PLAIN plain, uint32_t len) {
  std::cout << '[' << name << "]: ";
  Dump_plain(plain, 0, len);
}

double* Get_msg(CIPHER cipher) {
  std::vector<double> values;
  PHANTOM_CONTEXT::Context()->Decrypt(cipher, values);
  auto* result = static_cast<double*>(std::malloc(values.size() * sizeof(double)));
  if (result == nullptr) Fail("GET_MSG_ALLOC", "host allocation failed");
  std::memcpy(result, values.data(), values.size() * sizeof(double));
  return result;
}

double* Get_msg_from_plain(PLAIN plain) {
  std::vector<double> values;
  PHANTOM_CONTEXT::Context()->Decode(plain, values);
  auto* result = static_cast<double*>(std::malloc(values.size() * sizeof(double)));
  if (result == nullptr) Fail("GET_PLAIN_MSG_ALLOC", "host allocation failed");
  std::memcpy(result, values.data(), values.size() * sizeof(double));
  return result;
}
