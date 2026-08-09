#include "common/rt_api.h"
#include "rt_phantom/rt_phantom.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ACE_CONTEXT_MANIFEST_SHA256
#error "ACE_CONTEXT_MANIFEST_SHA256 must identify the exact generated context"
#endif

// The traced bootstrap is not a program entry, so CKKS2C intentionally omits
// the runtime I/O metadata helpers. This harness owns their sole definitions;
// the host gate must reject generated source/object definitions before linking.
extern "C" int Get_input_count() { return 0; }
extern "C" int Get_output_count() { return 0; }
extern "C" DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
extern "C" DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }

namespace {

using Complex = std::complex<double>;
using Json = nlohmann::json;

// This is an encoder-only round trip: no encryption, evaluation, or bootstrap
// noise is present. The mixed tolerance remains well below the full-bootstrap
// 1e-2 threshold while allowing normal double-precision FFT and CKKS rounding.
constexpr double kEncoderAbsoluteTolerance = 1.0e-7;
constexpr double kEncoderRelativeTolerance = 1.0e-10;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

void RequireCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    Fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void WriteJson(const std::string& path, const Json& value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) Fail("cannot create JSON output " + path);
  output << value.dump(2) << '\n';
  output.flush();
  if (!output) Fail("cannot write JSON output " + path);
}

class ContextGuard final {
public:
  ContextGuard() = default;
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;

  ~ContextGuard() {
    if (_prepared) Finalize_context();
  }

  void Prepare() {
    Require(!_prepared, "Phantom context was prepared twice");
    Prepare_context();
    _prepared = true;
  }

  void Finalize() {
    Require(_prepared, "Phantom context was not prepared");
    Finalize_context();
    _prepared = false;
  }

private:
  bool _prepared = false;
};

class RuntimePlain final {
public:
  RuntimePlain() = default;
  RuntimePlain(const RuntimePlain&) = delete;
  RuntimePlain& operator=(const RuntimePlain&) = delete;

  ~RuntimePlain() {
    if (_live) Free_plain(&_value);
  }

  PLAIN Get() { return &_value; }
  const PLAINTEXT& Value() const { return _value; }
  void MarkLive() { _live = true; }
  void Release() {
    if (_live) {
      Free_plain(&_value);
      _live = false;
    }
  }

private:
  PLAINTEXT _value;
  bool _live = false;
};

std::string ParameterFingerprint(const phantom::parms_id_type& value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (uint64_t word : value) output << std::setw(16) << word;
  return output.str();
}

std::string CheckGpu() {
  int count = 0;
  RequireCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  Require(count == 1, "expected exactly one visible CUDA device, observed " +
                          std::to_string(count));
  cudaDeviceProp properties{};
  RequireCuda(cudaGetDeviceProperties(&properties, 0),
              "cudaGetDeviceProperties");
  const std::string name(properties.name);
  Require(name.find("A100") != std::string::npos,
          "bootstrap constant qualification requires an A100, observed " +
              name);
  return name;
}

struct GeneratedManifests {
  const PHANTOM_CONTEXT_MANIFEST* _context;
  const PHANTOM_RESOURCE_MANIFEST* _resources;
  const PHANTOM_CONSTANT_MANIFEST* _constants;
};

GeneratedManifests ValidateGeneratedManifests() {
  const auto* context = Get_phantom_context_manifest();
  const auto* resources = Get_phantom_resource_manifest();
  const auto* constants = Get_phantom_constant_manifest();
  Require(context != nullptr && resources != nullptr && constants != nullptr,
          "all three exact generated manifest objects must be linked");
  Require(context->_schema_version == 1 &&
              context->_resource_schema_version == 3,
          "generated context must use context/resource schemas 1/3");
  Require(resources->_schema_version == 3 &&
              resources->_context_schema_version == 1,
          "generated resource manifest must use schemas 3/1");
  Require(constants->_schema_version == 1 &&
              constants->_context_schema_version == 1 &&
              constants->_resource_schema_version == 3,
          "generated constant manifest must use schemas 1/1/3");
  Require(constants->_context_manifest_sha256 != nullptr &&
              std::string(constants->_context_manifest_sha256) ==
                  ACE_CONTEXT_MANIFEST_SHA256,
          "constant manifest is not bound to the exact context sidecar SHA-256");
  Require(constants->_entry_count != 0 && constants->_entries != nullptr,
          "bootstrap constant manifest must be nonempty");
  Require(constants->_entry_count <=
              std::numeric_limits<size_t>::max() / 2,
          "constant manifest entry count is out of range");

  constexpr uint64_t required_flags =
      PHANTOM_RESOURCE_RELIN_KEY | PHANTOM_RESOURCE_ROTATION_KEYS |
      PHANTOM_RESOURCE_CONJUGATION_KEY | PHANTOM_RESOURCE_ROTATE_BATCH |
      PHANTOM_RESOURCE_RAISE_MOD | PHANTOM_RESOURCE_MONOMIALS |
      PHANTOM_RESOURCE_COMPLEX_PLAINTEXT;
  Require((resources->_flags &
           PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE) == 0,
          "native bootstrap precomputation must remain false");
  Require(resources->_flags == required_flags,
          "generated resource flags do not equal the bootstrap primitive set");
  Require(resources->_rotation_count != 0 &&
              resources->_rotation_steps != nullptr,
          "bootstrap rotation-key requirements are empty");
  Require(resources->_rotation_batch_count != 0 &&
              resources->_rotation_batch_offsets != nullptr &&
              resources->_rotation_batch_steps != nullptr,
          "bootstrap batch-rotation requirements are empty");
  Require(resources->_monomial_count != 0 &&
              resources->_monomial_powers != nullptr,
          "bootstrap monomial requirements are empty");
  return {context, resources, constants};
}

void ValidateEntry(const PHANTOM_CONSTANT_ENTRY& entry, size_t index,
                   size_t logical_slots) {
  Require(index <= std::numeric_limits<uint32_t>::max() &&
              entry._entry_id == static_cast<uint32_t>(index),
          "constant entry IDs must be contiguous manifest indices");
  Require(entry._element_type == PHANTOM_CONSTANT_COMPLEX_F64,
          "constant entry is not complex_f64");
  Require(entry._symbol != nullptr && entry._symbol[0] != '\0',
          "constant entry symbol is empty");
  Require(entry._slot_count != 0 && entry._slot_count <= logical_slots &&
              entry._slot_count <= std::numeric_limits<size_t>::max() / 2 &&
              entry._interleaved_value_count == entry._slot_count * 2 &&
              entry._interleaved_values != nullptr,
          "constant entry has invalid declared slot/payload shape");
  Require(entry._ace_level != 0 && std::isfinite(entry._raw_scale) &&
              entry._raw_scale > 0.0 && entry._scale_degree > 0,
          "constant entry has invalid level or scale metadata");
  for (size_t value = 0; value < entry._interleaved_value_count; ++value) {
    Require(std::isfinite(entry._interleaved_values[value]),
            "constant payload contains a nonfinite component");
  }
}

void ValidatePlainMetadata(PLAIN plain, const PLAINTEXT& direct,
                           const PHANTOM_CONSTANT_ENTRY& entry,
                           const PHANTOM_CONTEXT_MANIFEST& context,
                           const char* mode) {
  const std::string prefix = std::string(mode) + " entry " +
                             std::to_string(entry._entry_id) + ": ";
  Require(Get_plain_chain_index(plain) ==
              static_cast<LEVEL_T>(entry._chain_index),
          prefix + "chain index mismatch");
  Require(Get_plain_level(plain) == static_cast<LEVEL_T>(entry._ace_level) &&
              Get_plain_active_q_count(plain) ==
                  static_cast<LEVEL_T>(entry._ace_level),
          prefix + "ACE level/active-Q mismatch");
  Require(Get_plain_raw_scale(plain) == entry._raw_scale &&
              Get_plain_scale_degree(plain) ==
                  static_cast<SCALE_T>(entry._scale_degree),
          prefix + "raw scale/scale-degree mismatch");
  Require(Get_plain_slots(plain) == context._logical_slots &&
              entry._slot_count <= Get_plain_slots(plain),
          prefix + "declared slots exceed plaintext capacity");
  Require(Is_plain_ntt(plain), prefix + "plaintext is not ordinary-Q NTT");
  Require(direct.parms_id() != phantom::parms_id_zero,
          prefix + "plaintext parameter fingerprint is zero");
  Require(direct.chain_index() == entry._chain_index &&
              direct.scale() == entry._raw_scale &&
              direct.coeff_modulus_size() == entry._ace_level &&
              direct.poly_modulus_degree() == context._poly_degree &&
              direct.data() != nullptr,
          prefix + "direct Phantom plaintext metadata mismatch");
}

struct ErrorSummary {
  double _maximum_correctness_error = 0.0;
  double _maximum_cached_error = 0.0;
  double _maximum_mode_error = 0.0;
};

ErrorSummary ValidateDecoded(const std::vector<Complex>& correctness,
                             const std::vector<Complex>& cached,
                             const PHANTOM_CONSTANT_ENTRY& entry,
                             size_t logical_slots) {
  Require(correctness.size() == logical_slots && cached.size() == logical_slots,
          "decoded plaintext size does not equal the context slot capacity");
  ErrorSummary result;
  for (size_t slot = 0; slot < logical_slots; ++slot) {
    const Complex expected =
        slot < entry._slot_count
            ? Complex(entry._interleaved_values[slot * 2],
                      entry._interleaved_values[slot * 2 + 1])
            : Complex{};
    const Complex observed[] = {correctness[slot], cached[slot]};
    for (size_t mode = 0; mode < 2; ++mode) {
      const Complex& value = observed[mode];
      Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
              "decoded constant contains a nonfinite component");
      const double error = std::abs(value - expected);
      const double expected_magnitude = std::abs(expected);
      const double tolerance =
          kEncoderAbsoluteTolerance +
          kEncoderRelativeTolerance * expected_magnitude;
      Require(std::isfinite(error) && std::isfinite(expected_magnitude) &&
                  std::isfinite(tolerance) && error <= tolerance,
              "decoded constant exceeds the encoder-only tolerance");
      if (mode == 0) {
        result._maximum_correctness_error =
            std::max(result._maximum_correctness_error, error);
      } else {
        result._maximum_cached_error =
            std::max(result._maximum_cached_error, error);
      }
    }
    const double mode_error = std::abs(correctness[slot] - cached[slot]);
    Require(std::isfinite(mode_error),
            "correctness/cache comparison is nonfinite");
    result._maximum_mode_error =
        std::max(result._maximum_mode_error, mode_error);
  }
  return result;
}

void ValidateSetupMetrics(const PHANTOM_SETUP_METRICS& metrics,
                          const PHANTOM_CONSTANT_MANIFEST& constants,
                          const PHANTOM_CONTEXT_MANIFEST& context) {
  Require(std::isfinite(metrics._context_and_key_setup_seconds) &&
              metrics._context_and_key_setup_seconds > 0.0,
          "context/key setup time is not positive and finite");
  Require(std::isfinite(metrics._plaintext_cache_setup_seconds) &&
              metrics._plaintext_cache_setup_seconds > 0.0,
          "plaintext-cache setup time is not positive and finite");
  Require(metrics._plaintext_cache_entries == constants._entry_count,
          "plaintext-cache entry metric disagrees with the manifest");
  uint64_t payload_bytes = 0;
  uint64_t logical_device_bytes = 0;
  for (size_t index = 0; index < constants._entry_count; ++index) {
    const PHANTOM_CONSTANT_ENTRY& entry = constants._entries[index];
    const auto count = entry._interleaved_value_count;
    Require(count <= std::numeric_limits<uint64_t>::max() / sizeof(double) &&
                payload_bytes <= std::numeric_limits<uint64_t>::max() -
                                     count * sizeof(double),
            "constant payload byte count overflowed");
    payload_bytes += count * sizeof(double);
    Require(entry._ace_level != 0 &&
                context._poly_degree <=
                    std::numeric_limits<uint64_t>::max() / entry._ace_level &&
                static_cast<uint64_t>(context._poly_degree) *
                        entry._ace_level <=
                    std::numeric_limits<uint64_t>::max() / sizeof(uint64_t),
            "constant logical device byte count overflowed");
    const uint64_t entry_device_bytes =
        static_cast<uint64_t>(context._poly_degree) * entry._ace_level *
        sizeof(uint64_t);
    Require(logical_device_bytes <=
                std::numeric_limits<uint64_t>::max() - entry_device_bytes,
            "plaintext-cache logical device byte count overflowed");
    logical_device_bytes += entry_device_bytes;
  }
  Require(metrics._plaintext_cache_host_bytes >= payload_bytes,
          "plaintext-cache host bytes omit emitted payload inputs");
  Require(logical_device_bytes > 0 &&
              metrics._plaintext_cache_logical_device_bytes ==
                  logical_device_bytes,
          "plaintext-cache logical device bytes disagree with retained "
          "entries");
}

Json MetricsJson(const PHANTOM_SETUP_METRICS& metrics) {
  return {
      {"context_and_keys",
       {{"seconds", metrics._context_and_key_setup_seconds},
        {"device_bytes", metrics._context_and_key_device_bytes}}},
      {"plaintext_cache",
       {{"seconds", metrics._plaintext_cache_setup_seconds},
        {"device_bytes", metrics._plaintext_cache_device_bytes},
        {"logical_device_bytes",
         metrics._plaintext_cache_logical_device_bytes},
        {"host_bytes", metrics._plaintext_cache_host_bytes},
        {"entries", metrics._plaintext_cache_entries}}},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: bootstrap_phantom_constants OUTPUT.json\n";
    return 2;
  }

  try {
    const std::string gpu = CheckGpu();
    const GeneratedManifests manifests = ValidateGeneratedManifests();
    ContextGuard context_guard;
    context_guard.Prepare();

    const PHANTOM_SETUP_METRICS metrics = Get_phantom_setup_metrics();
    ValidateSetupMetrics(metrics, *manifests._constants,
                         *manifests._context);
    double maximum_correctness_error = 0.0;
    double maximum_cached_error = 0.0;
    double maximum_mode_error = 0.0;
    Json entry_results = Json::array();
    // Retain unique host object addresses for the runtime lifetime registry,
    // but release each pair's device allocation before advancing.
    std::vector<std::unique_ptr<RuntimePlain>> plaintext_objects;
    plaintext_objects.reserve(manifests._constants->_entry_count * 2);

    for (size_t index = 0; index < manifests._constants->_entry_count;
         ++index) {
      const PHANTOM_CONSTANT_ENTRY& entry =
          manifests._constants->_entries[index];
      ValidateEntry(entry, index, manifests._context->_logical_slots);

      plaintext_objects.push_back(std::make_unique<RuntimePlain>());
      RuntimePlain& correctness = *plaintext_objects.back();
      Encode_manifest_constant(correctness.Get(), entry._entry_id);
      correctness.MarkLive();
      plaintext_objects.push_back(std::make_unique<RuntimePlain>());
      RuntimePlain& cached = *plaintext_objects.back();
      Load_cached_plain(cached.Get(), entry._entry_id);
      cached.MarkLive();
      ValidatePlainMetadata(correctness.Get(), correctness.Value(), entry,
                            *manifests._context, "correctness");
      ValidatePlainMetadata(cached.Get(), cached.Value(), entry,
                            *manifests._context, "cached");
      Require(correctness.Value().parms_id() == cached.Value().parms_id(),
              "correctness and cached parameter fingerprints differ");

      std::vector<Complex> correctness_values;
      std::vector<Complex> cached_values;
      Phantom_decode_dcmplx(correctness.Get(), correctness_values);
      Phantom_decode_dcmplx(cached.Get(), cached_values);
      RequireCuda(cudaDeviceSynchronize(), "constant decode synchronize");
      const ErrorSummary errors =
          ValidateDecoded(correctness_values, cached_values, entry,
                          manifests._context->_logical_slots);
      maximum_correctness_error =
          std::max(maximum_correctness_error,
                   errors._maximum_correctness_error);
      maximum_cached_error =
          std::max(maximum_cached_error, errors._maximum_cached_error);
      maximum_mode_error =
          std::max(maximum_mode_error, errors._maximum_mode_error);
      entry_results.push_back(
          {{"entry_id", entry._entry_id},
           {"constant_id", entry._constant_id},
           {"symbol", entry._symbol},
           {"slot_count", entry._slot_count},
           {"ace_level", entry._ace_level},
           {"chain_index", entry._chain_index},
           {"raw_scale", entry._raw_scale},
           {"parameter_fingerprint",
            ParameterFingerprint(correctness.Value().parms_id())},
           {"maximum_correctness_error",
            errors._maximum_correctness_error},
           {"maximum_cached_error", errors._maximum_cached_error},
           {"maximum_mode_error", errors._maximum_mode_error}});
      correctness.Release();
      cached.Release();
    }

    context_guard.Finalize();
    WriteJson(
        argv[1],
        {{"status", "pass"},
         {"gpu", gpu},
         {"context_manifest_sha256", ACE_CONTEXT_MANIFEST_SHA256},
         {"resource_flags", manifests._resources->_flags},
         {"counts",
          {{"constants_declared", manifests._constants->_entry_count},
           {"correctness_encodes", manifests._constants->_entry_count},
           {"cached_loads", manifests._constants->_entry_count}}},
         {"tolerance",
          {{"absolute", kEncoderAbsoluteTolerance},
           {"relative", kEncoderRelativeTolerance}}},
         {"maximum_correctness_error", maximum_correctness_error},
         {"maximum_cached_error", maximum_cached_error},
         {"maximum_mode_error", maximum_mode_error},
         {"setup_metrics", MetricsJson(metrics)},
         {"entries", std::move(entry_results)}});
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Phantom bootstrap constant qualification failed: "
              << error.what() << '\n';
    return 1;
  }
}
