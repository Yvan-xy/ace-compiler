#include "common/rt_api.h"
#include "rt_phantom/rt_phantom.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ACE_CONTEXT_MANIFEST_SHA256
#error "ACE_CONTEXT_MANIFEST_SHA256 must identify the compiler context"
#endif
#ifndef ACE_FIXTURE_SHA256
#error "ACE_FIXTURE_SHA256 must identify the frozen ordinary CKKS fixture"
#endif
#ifndef ACE_PRODUCER_ID
#error "ACE_PRODUCER_ID must identify the tested source and binary"
#endif

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

namespace {

using Json = nlohmann::json;
using Complex = std::complex<double>;

constexpr int kArithmeticLevel = 4;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void RequireCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    Fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void CheckGpu(const std::string& expected_name) {
  int count = 0;
  RequireCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count != 1) {
    Fail("expected exactly one CUDA device, observed " +
         std::to_string(count));
  }
  cudaDeviceProp properties{};
  RequireCuda(cudaGetDeviceProperties(&properties, 0),
              "cudaGetDeviceProperties");
  if (expected_name != properties.name) {
    Fail("expected GPU '" + expected_name + "', observed '" +
         std::string(properties.name) + "'");
  }
}

Json LoadJson(const std::string& path) {
  std::ifstream input(path);
  if (!input) Fail("cannot open JSON input " + path);
  Json value;
  input >> value;
  if (!input.eof() && input.fail()) Fail("cannot parse JSON input " + path);
  return value;
}

void WriteJson(const std::string& path, const Json& value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) Fail("cannot create JSON output " + path);
  output << value.dump(2) << '\n';
  output.flush();
  if (!output) Fail("cannot write JSON output " + path);
}

class ObjectArena final {
public:
  CIPHER NewCipher() {
    _ciphers.push_back(std::make_unique<CIPHERTEXT>());
    return _ciphers.back().get();
  }

  PLAIN NewPlain() {
    _plains.push_back(std::make_unique<PLAINTEXT>());
    return _plains.back().get();
  }

  void FreeCipher(CIPHER value) {
    if (value != nullptr && _freed_ciphers.insert(value).second) {
      Free_ciph(value);
    }
  }

  void FreePlain(PLAIN value) {
    if (value != nullptr && _freed_plains.insert(value).second) {
      Free_plain(value);
    }
  }

  void FreeLiveObjects() {
    for (const auto& value : _ciphers) FreeCipher(value.get());
    for (const auto& value : _plains) FreePlain(value.get());
  }

private:
  std::vector<std::unique_ptr<CIPHERTEXT>> _ciphers;
  std::vector<std::unique_ptr<PLAINTEXT>> _plains;
  std::set<CIPHER> _freed_ciphers;
  std::set<PLAIN> _freed_plains;
};

std::vector<Complex> ExpandInput(const Json& specification,
                                 std::size_t slots) {
  if (specification.at("length") !=
      "compiler_context.logical_slot_capacity") {
    Fail("fixture input length is not compiler-context-derived");
  }
  const auto& default_value = specification.at("default");
  std::vector<Complex> result(
      slots, Complex(default_value.at(0).get<double>(),
                     default_value.at(1).get<double>()));
  const std::string storage = specification.at("storage").get<std::string>();
  if (storage == "default_overrides") {
    for (const auto& entry : specification.at("overrides")) {
      const std::size_t index = entry.at(0).get<std::size_t>();
      if (index >= slots) Fail("fixture override exceeds logical slots");
      result[index] =
          Complex(entry.at(1).get<double>(), entry.at(2).get<double>());
    }
  } else if (storage == "segments") {
    for (const auto& entry : specification.at("segments")) {
      const std::size_t start = entry.at(0).get<std::size_t>();
      const std::size_t count = entry.at(1).get<std::size_t>();
      if (start + count > slots) Fail("fixture segment exceeds logical slots");
      std::fill(result.begin() + start, result.begin() + start + count,
                Complex(entry.at(2).get<double>(),
                        entry.at(3).get<double>()));
    }
  } else {
    Fail("unsupported fixture input storage");
  }
  return result;
}

std::size_t MaskLength(const Json& specification) {
  const auto& segments = specification.at("segments");
  if (segments.size() != 1 || segments.at(0).at(0).get<std::size_t>() != 0) {
    Fail("mask fixture must contain one leading segment");
  }
  return segments.at(0).at(1).get<std::size_t>();
}

PLAIN EncodeComplex(ObjectArena& arena, const std::vector<Complex>& values,
                    int level, double scale_degree = 1.0) {
  PLAIN plain = arena.NewPlain();
  std::vector<DCMPLX> copy(values.begin(), values.end());
  Encode_dcmplx(plain, copy.data(), copy.size(), scale_degree, level);
  return plain;
}

CIPHER EncryptComplex(ObjectArena& arena,
                      const std::vector<Complex>& values, int level,
                      double scale_degree = 1.0) {
  PLAIN plain = EncodeComplex(arena, values, level, scale_degree);
  CIPHER cipher = arena.NewCipher();
  Phantom_encrypt_plain(cipher, plain);
  arena.FreePlain(plain);
  return cipher;
}

std::vector<Complex> DecodeCipher(CIPHER cipher) {
  std::vector<Complex> values;
  Phantom_decrypt_dcmplx(cipher, values);
  return values;
}

std::vector<Complex> DecodePlain(PLAIN plain) {
  std::vector<Complex> values;
  Phantom_decode_dcmplx(plain, values);
  return values;
}

Json ValuesJson(const std::vector<Complex>& values) {
  Json result = Json::array();
  for (const Complex& value : values) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      Fail("provider result contains NaN or Inf");
    }
    result.push_back(Json::array({value.real(), value.imag()}));
  }
  return result;
}

Json CipherMetadata(CIPHER value, std::uint64_t dropped_modulus = 0) {
  Json result = {
      {"object_kind", "ciphertext"},
      {"ace_level", Level(value)},
      {"active_q_count", Active_q_count(value)},
      {"phantom_chain_index", Chain_index(value)},
      {"scale_degree", static_cast<std::int64_t>(Sc_degree(value))},
      {"raw_scale", Raw_scale(value)},
      {"slots", Get_ciph_slots(value)},
      {"ciphertext_size", Get_ciph_size(value)},
      {"ntt", Is_ciph_ntt(value)},
  };
  if (dropped_modulus != 0) result["dropped_modulus"] = dropped_modulus;
  return result;
}

Json PlainMetadata(PLAIN value) {
  return {
      {"object_kind", "plaintext"},
      {"ace_level", Get_plain_level(value)},
      {"active_q_count", Get_plain_active_q_count(value)},
      {"phantom_chain_index", Get_plain_chain_index(value)},
      {"scale_degree",
       static_cast<std::int64_t>(Get_plain_scale_degree(value))},
      {"raw_scale", Get_plain_raw_scale(value)},
      {"slots", Get_plain_slots(value)},
      {"ciphertext_size", nullptr},
      {"ntt", Is_plain_ntt(value)},
  };
}

struct CaseResult {
  std::vector<Complex> _values;
  Json _metadata;
};

CIPHER Destination(ObjectArena& arena, const std::string& alias, CIPHER left,
                   CIPHER right = nullptr) {
  if (alias == "distinct") return arena.NewCipher();
  if (alias == "lhs" || alias == "inplace" || alias == "self") return left;
  if (alias == "rhs" && right != nullptr) return right;
  Fail("unsupported alias " + alias);
}

CaseResult RunCipherCase(const std::string& family_id,
                         const std::string& alias, const Json& fixture,
                         ObjectArena& arena) {
  const auto* manifest = Get_phantom_context_manifest();
  const std::size_t slots = manifest->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const auto y = ExpandInput(fixture.at("inputs").at("complex_y"), slots);
  CIPHER result = nullptr;
  std::uint64_t dropped_modulus = 0;

  if (family_id == "add_ct_ct" || family_id == "sub_ct_ct" ||
      family_id == "mul_ct_ct") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel);
    result = Destination(arena, alias, left, right);
    if (family_id == "add_ct_ct") Add_ciph(result, left, right);
    if (family_id == "sub_ct_ct") Sub_ciph(result, left, right);
    if (family_id == "mul_ct_ct") Mul_ciph(result, left, right);
  } else if (family_id == "add_ct_plain" ||
             family_id == "sub_ct_plain" ||
             family_id == "mul_ct_plain") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    PLAIN right = EncodeComplex(arena, y, kArithmeticLevel);
    result = Destination(arena, alias, left);
    if (family_id == "add_ct_plain") Add_plain(result, left, right);
    if (family_id == "sub_ct_plain") Sub_plain(result, left, right);
    if (family_id == "mul_ct_plain") Mul_plain(result, left, right);
  } else if (family_id == "add_ct_scalar" ||
             family_id == "sub_ct_scalar" ||
             family_id == "mul_ct_scalar") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    result = Destination(arena, alias, left);
    if (family_id == "add_ct_scalar") Add_scalar(result, left, 0.5);
    if (family_id == "sub_ct_scalar") Sub_scalar(result, left, 0.5);
    if (family_id == "mul_ct_scalar") Mul_scalar(result, left, -0.75);
  } else if (family_id == "copy") {
    CIPHER source = EncryptComplex(arena, x, kArithmeticLevel);
    result = Destination(arena, alias, source);
    Copy_ciph(result, source);
  } else if (family_id == "query_all") {
    result = EncryptComplex(arena, x, kArithmeticLevel);
  } else if (family_id == "modswitch") {
    CIPHER source = EncryptComplex(arena, x, kArithmeticLevel);
    result = Destination(arena, alias, source);
    Mod_switch(result, source);
  } else if (family_id == "relinearize") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel);
    CIPHER product = arena.NewCipher();
    Mul_ciph(product, left, right);
    result = Destination(arena, alias, product);
    Relin(result, product);
  } else if (family_id == "rescale") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel);
    CIPHER product = arena.NewCipher();
    CIPHER relinearized = arena.NewCipher();
    Mul_ciph(product, left, right);
    Relin(relinearized, product);
    result = Destination(arena, alias, relinearized);
    Rescale_ciph(result, relinearized);
    const long double numerator = std::ldexp(
        static_cast<long double>(1.0),
        2 * static_cast<int>(manifest->_scaling_modulus_bits));
    dropped_modulus = static_cast<std::uint64_t>(
        std::llround(numerator / static_cast<long double>(Raw_scale(result))));
  } else if (family_id == "rotate_negative" ||
             family_id == "rotate_positive" ||
             family_id == "rotate_zero") {
    CIPHER source = EncryptComplex(arena, x, kArithmeticLevel);
    result = Destination(arena, alias, source);
    const int step = family_id == "rotate_negative"
                         ? -3
                         : (family_id == "rotate_positive" ? 3 : 0);
    Rotate_ciph(result, source, step);
  } else {
    Fail("unsupported ciphertext case family " + family_id);
  }

  CaseResult output{DecodeCipher(result),
                    CipherMetadata(result, dropped_modulus)};
  arena.FreeLiveObjects();
  return output;
}

CaseResult RunEncodeCase(const std::string& family_id, const Json& fixture,
                         ObjectArena& arena) {
  const auto* manifest = Get_phantom_context_manifest();
  const std::size_t slots = manifest->_logical_slots;
  PLAIN plain = arena.NewPlain();
  if (family_id == "encode_complex_q4") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("complex_x"), slots);
    std::vector<DCMPLX> input(values.begin(), values.end());
    Encode_dcmplx(plain, input.data(), input.size(), 1.0, kArithmeticLevel);
  } else if (family_id == "encode_mask_f32_q4") {
    const auto& specification = fixture.at("inputs").at("mask_f32");
    const float value =
        specification.at("segments").at(0).at(2).get<float>();
    Encode_float_mask(plain, value, MaskLength(specification), 1.0,
                      kArithmeticLevel);
  } else if (family_id == "encode_mask_f64_q4") {
    const auto& specification = fixture.at("inputs").at("mask_f64");
    const double value =
        specification.at("segments").at(0).at(2).get<double>();
    Encode_double_mask(plain, value, MaskLength(specification), 1.0,
                       kArithmeticLevel);
  } else if (family_id == "encode_real_f32_q4") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f32"), slots);
    std::vector<float> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    Encode_float(plain, input.data(), input.size(), 1.0, kArithmeticLevel);
  } else if (family_id == "encode_real_f64_bottom" ||
             family_id == "encode_real_f64_middle" ||
             family_id == "encode_real_f64_full") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f64"), slots);
    std::vector<double> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    const int level = family_id == "encode_real_f64_bottom"
                          ? 1
                          : (family_id == "encode_real_f64_middle"
                                 ? static_cast<int>(
                                       (manifest->_data_q_count + 1) / 2)
                                 : static_cast<int>(manifest->_data_q_count));
    Encode_double(plain, input.data(), input.size(), 1.0, level);
  } else {
    Fail("unsupported encode case family " + family_id);
  }
  CaseResult output{DecodePlain(plain), PlainMetadata(plain)};
  arena.FreeLiveObjects();
  return output;
}

Json RunConformance(const Json& fixture) {
  Json records = Json::array();
  // Keep every wrapper object alive for the duration of the prepared runtime
  // context.  The runtime deliberately remembers freed wrapper addresses to
  // diagnose double-free and use-after-free.  Destroying an arena after each
  // case allowed the host allocator to reuse an address for a new wrapper,
  // which is indistinguishable from reusing the freed object itself.
  ObjectArena arena;
  for (const auto& family : fixture.at("case_families")) {
    const std::string family_id = family.at("id").get<std::string>();
    for (const auto& alias_value : family.at("aliases")) {
      const std::string alias = alias_value.get<std::string>();
      CaseResult result = family_id.rfind("encode_", 0) == 0
                              ? RunEncodeCase(family_id, fixture, arena)
                              : RunCipherCase(family_id, alias, fixture, arena);
      if (result._values.size() !=
          Get_phantom_context_manifest()->_logical_slots) {
        Fail("decoded result length disagrees with compiler logical slots");
      }
      records.push_back({
          {"case_id", family_id + "." + alias},
          {"values", ValuesJson(result._values)},
          {"metadata", result._metadata},
      });
    }
  }
  return {
      {"schema_version", "ace.phantom.ordinary_ckks.raw-provider/1.0.0"},
      {"provider", "phantom"},
      {"fixture_id", fixture.at("fixture_id")},
      {"fixture_sha256", ACE_FIXTURE_SHA256},
      {"compiler_context_manifest_sha256", ACE_CONTEXT_MANIFEST_SHA256},
      {"producer", ACE_PRODUCER_ID},
      {"independent_encoding", true},
      {"input_kind", "clear_fixture_values"},
      {"records", std::move(records)},
  };
}

double MaximumError(const std::vector<Complex>& left,
                    const std::vector<Complex>& right) {
  if (left.size() != right.size()) Fail("ownership comparison length mismatch");
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

void RunOwnership(const Json& fixture) {
  const std::size_t slots = Get_phantom_context_manifest()->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  std::vector<std::unique_ptr<CIPHERTEXT[]>> retained_arrays;
  retained_arrays.reserve(200);
  ObjectArena arena;

  for (int iteration = 0; iteration < 100; ++iteration) {
    CIPHER source = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER copy = arena.NewCipher();
    Copy_ciph(copy, source);
    Add_scalar(source, source, 0.25);
    if (MaximumError(DecodeCipher(copy), x) > 5.0e-3) {
      Fail("copy independence failed");
    }
    arena.FreeCipher(source);
    arena.FreeCipher(copy);
  }

  CIPHER reusable = arena.NewCipher();
  for (int iteration = 0; iteration < 100; ++iteration) {
    CIPHER source = EncryptComplex(arena, x, kArithmeticLevel);
    Copy_ciph(reusable, source);
    Add_scalar(reusable, reusable, 0.125);
    arena.FreeCipher(source);
  }
  arena.FreeCipher(reusable);

  for (std::size_t length : {std::size_t{1}, std::size_t{4}}) {
    for (int iteration = 0; iteration < 100; ++iteration) {
      auto array = std::make_unique<CIPHERTEXT[]>(length);
      PLAIN plain = EncodeComplex(arena, x, kArithmeticLevel);
      for (std::size_t index = 0; index < length; ++index) {
        Phantom_encrypt_plain(&array[index], plain);
      }
      Free_ciph_array(array.get(), length);
      arena.FreePlain(plain);
      retained_arrays.push_back(std::move(array));
    }
  }

  for (int iteration = 0; iteration < 100; ++iteration) {
    CIPHER value = arena.NewCipher();
    Zero_ciph(value);
    arena.FreeCipher(value);
  }
  arena.FreeLiveObjects();
  RequireCuda(cudaDeviceSynchronize(), "ownership cudaDeviceSynchronize");
  std::cout << "{\"status\":\"pass\",\"iterations\":100}" << std::endl;
}

void RunRejection(const std::string& rejection_id, const Json& fixture) {
  const std::size_t slots = Get_phantom_context_manifest()->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const auto y = ExpandInput(fixture.at("inputs").at("complex_y"), slots);
  ObjectArena arena;

  if (rejection_id == "bottom_modswitch" ||
      rejection_id == "bottom_rescale") {
    CIPHER source = EncryptComplex(arena, x, 1);
    CIPHER result = arena.NewCipher();
    if (rejection_id == "bottom_modswitch") Mod_switch(result, source);
    Rescale_ciph(result, source);
  } else if (rejection_id == "double_free") {
    CIPHER value = EncryptComplex(arena, x, kArithmeticLevel);
    Free_ciph(value);
    Free_ciph(value);
  } else if (rejection_id == "use_after_free") {
    CIPHER value = EncryptComplex(arena, x, kArithmeticLevel);
    Free_ciph(value);
    (void)Level(value);
  } else if (rejection_id == "relinearize_size2") {
    CIPHER value = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER result = arena.NewCipher();
    Relin(result, value);
  } else if (rejection_id == "missing_loaded_relin_key") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel);
    CIPHER product = arena.NewCipher();
    CIPHER result = arena.NewCipher();
    Mul_ciph(product, left, right);
    Relin(result, product);
  } else if (rejection_id == "missing_loaded_rotation_key") {
    CIPHER value = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER result = arena.NewCipher();
    Rotate_ciph(result, value, 3);
  } else if (rejection_id == "invalid_ciphertext_size") {
    CIPHER value = EncryptComplex(arena, x, kArithmeticLevel);
    value->resize(4, value->coeff_modulus_size(),
                  value->poly_modulus_degree(), nullptr);
    (void)Get_ciph_size(value);
  } else if (rejection_id == "runtime_level_mismatch") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel - 1);
    CIPHER result = arena.NewCipher();
    Add_ciph(result, left, right);
  } else if (rejection_id == "runtime_scale_mismatch") {
    CIPHER left = EncryptComplex(arena, x, kArithmeticLevel, 1.0);
    CIPHER right = EncryptComplex(arena, y, kArithmeticLevel, 2.0);
    CIPHER result = arena.NewCipher();
    Add_ciph(result, left, right);
  } else {
    Fail("unknown runtime rejection case " + rejection_id);
  }
  Fail("runtime rejection case returned successfully: " + rejection_id);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: ordinary_ckks_gpu_runner "
                   "<conformance|ownership|reject> <fixture.json> "
                   "<expected-gpu-name> [output-or-rejection]\n";
      return 2;
    }
    const std::string mode = argv[1];
    const Json fixture = LoadJson(argv[2]);
    CheckGpu(argv[3]);
    Prepare_context();
    if (mode == "conformance") {
      if (argc != 5) Fail("conformance mode requires an output path");
      WriteJson(argv[4], RunConformance(fixture));
    } else if (mode == "ownership") {
      if (argc != 4) Fail("ownership mode takes no fourth argument");
      RunOwnership(fixture);
    } else if (mode == "reject") {
      if (argc != 5) Fail("reject mode requires a rejection identifier");
      RunRejection(argv[4], fixture);
    } else {
      Fail("unsupported runner mode " + mode);
    }
    Finalize_context();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ordinary CKKS GPU runner failed: " << error.what() << '\n';
    return 1;
  }
}
