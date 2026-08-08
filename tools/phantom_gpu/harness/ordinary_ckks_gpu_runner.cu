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
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
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

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

struct LevelCoordinates {
  int _full;
  int _after_one_drop;
  int _middle;
  int _bottom;
};

LevelCoordinates ContextLevelCoordinates() {
  const auto* manifest = Get_phantom_context_manifest();
  if (manifest == nullptr || manifest->_data_q_count < 2) {
    Fail("ordinary CKKS requires at least two compiler-emitted data-Q primes");
  }
  if (manifest->_data_q_count >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    Fail("compiler-emitted data-Q count exceeds the runtime level type");
  }
  const int full = static_cast<int>(manifest->_data_q_count);
  return {full, full - 1, (full + 1) / 2, 1};
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

// EXACT_SOURCE_PRESERVATION_BEGIN
std::size_t CheckedElementProduct(std::size_t left, std::size_t right,
                                  const std::string& context) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    Fail(context + " coefficient element count overflows size_t");
  }
  return left * right;
}

std::uint64_t ExactDoubleBits(double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<std::uint64_t> CopyDeviceCoefficients(
    const std::uint64_t* device_data, std::size_t element_count,
    const std::string& context) {
  if (element_count == 0 || device_data == nullptr) {
    Fail(context + " has no device coefficient buffer");
  }
  const std::size_t byte_count = CheckedElementProduct(
      element_count, sizeof(std::uint64_t), context + " byte count");
  std::vector<std::uint64_t> coefficients(element_count);
  RequireCuda(cudaDeviceSynchronize(),
              "source preservation cudaDeviceSynchronize");
  RequireCuda(cudaMemcpy(coefficients.data(), device_data, byte_count,
                         cudaMemcpyDeviceToHost),
              "source preservation cudaMemcpy");
  return coefficients;
}

struct CipherSnapshot {
  CIPHER _identity;
  phantom::parms_id_type _parameters;
  std::size_t _chain_index;
  std::size_t _ciphertext_size;
  std::size_t _polynomial_degree;
  std::size_t _coefficient_modulus_size;
  std::size_t _noise_scale_degree;
  std::uint64_t _scale_bits;
  bool _ntt;
  bool _asymmetric;
  Json _logical_metadata;
  std::vector<std::uint64_t> _coefficients;
};

struct PlainSnapshot {
  PLAIN _identity;
  phantom::parms_id_type _parameters;
  std::size_t _chain_index;
  std::size_t _polynomial_degree;
  std::size_t _coefficient_modulus_size;
  std::uint64_t _scale_bits;
  Json _logical_metadata;
  std::vector<std::uint64_t> _coefficients;
};

CipherSnapshot CaptureCipher(CIPHER value, const std::string& context) {
  if (value == nullptr) Fail(context + " ciphertext is null");
  const std::size_t polynomial_count = CheckedElementProduct(
      value->size(), value->coeff_modulus_size(), context);
  const std::size_t element_count = CheckedElementProduct(
      polynomial_count, value->poly_modulus_degree(), context);
  return {
      value,
      value->parms_id(),
      value->chain_index(),
      value->size(),
      value->poly_modulus_degree(),
      value->coeff_modulus_size(),
      value->GetNoiseScaleDeg(),
      ExactDoubleBits(value->scale()),
      value->is_ntt_form(),
      value->is_asymmetric(),
      CipherMetadata(value),
      CopyDeviceCoefficients(value->data(), element_count, context),
  };
}

PlainSnapshot CapturePlain(PLAIN value, const std::string& context) {
  if (value == nullptr) Fail(context + " plaintext is null");
  const std::size_t element_count = CheckedElementProduct(
      value->coeff_modulus_size(), value->poly_modulus_degree(), context);
  return {
      value,
      value->parms_id(),
      value->chain_index(),
      value->poly_modulus_degree(),
      value->coeff_modulus_size(),
      ExactDoubleBits(value->scale()),
      PlainMetadata(value),
      CopyDeviceCoefficients(value->data(), element_count, context),
  };
}

void RequireCipherPreserved(const CipherSnapshot& before, CIPHER value,
                            const std::string& context) {
  if (value != before._identity || value->parms_id() != before._parameters ||
      value->chain_index() != before._chain_index ||
      value->size() != before._ciphertext_size ||
      value->poly_modulus_degree() != before._polynomial_degree ||
      value->coeff_modulus_size() != before._coefficient_modulus_size ||
      value->GetNoiseScaleDeg() != before._noise_scale_degree ||
      ExactDoubleBits(value->scale()) != before._scale_bits ||
      value->is_ntt_form() != before._ntt ||
      value->is_asymmetric() != before._asymmetric ||
      CipherMetadata(value) != before._logical_metadata) {
    Fail(context + " changed non-destination ciphertext metadata");
  }
  const std::size_t polynomial_count = CheckedElementProduct(
      value->size(), value->coeff_modulus_size(), context);
  const std::size_t element_count = CheckedElementProduct(
      polynomial_count, value->poly_modulus_degree(), context);
  if (CopyDeviceCoefficients(value->data(), element_count, context) !=
      before._coefficients) {
    Fail(context + " changed non-destination ciphertext coefficients");
  }
}

void RequirePlainPreserved(const PlainSnapshot& before, PLAIN value,
                           const std::string& context) {
  if (value != before._identity || value->parms_id() != before._parameters ||
      value->chain_index() != before._chain_index ||
      value->poly_modulus_degree() != before._polynomial_degree ||
      value->coeff_modulus_size() != before._coefficient_modulus_size ||
      ExactDoubleBits(value->scale()) != before._scale_bits ||
      PlainMetadata(value) != before._logical_metadata) {
    Fail(context + " changed non-destination plaintext metadata");
  }
  const std::size_t element_count = CheckedElementProduct(
      value->coeff_modulus_size(), value->poly_modulus_degree(), context);
  if (CopyDeviceCoefficients(value->data(), element_count, context) !=
      before._coefficients) {
    Fail(context + " changed non-destination plaintext coefficients");
  }
}

template <typename Operation>
void InvokeCipherBinaryPreservingOperands(CIPHER destination, CIPHER left,
                                          CIPHER right,
                                          const std::string& context,
                                          Operation&& operation) {
  const CipherSnapshot left_before = CaptureCipher(left, context + " lhs");
  const CipherSnapshot right_before = CaptureCipher(right, context + " rhs");
  std::forward<Operation>(operation)();
  if (destination != left) {
    RequireCipherPreserved(left_before, left, context + " lhs");
  }
  if (destination != right) {
    RequireCipherPreserved(right_before, right, context + " rhs");
  }
}

template <typename Operation>
void InvokeCipherPlainPreservingOperands(CIPHER destination, CIPHER left,
                                         PLAIN right,
                                         const std::string& context,
                                         Operation&& operation) {
  const CipherSnapshot left_before = CaptureCipher(left, context + " lhs");
  const PlainSnapshot right_before = CapturePlain(right, context + " rhs");
  std::forward<Operation>(operation)();
  if (destination != left) {
    RequireCipherPreserved(left_before, left, context + " lhs");
  }
  RequirePlainPreserved(right_before, right, context + " rhs");
}

template <typename Operation>
void InvokeCipherUnaryPreservingSource(CIPHER destination, CIPHER source,
                                       const std::string& context,
                                       Operation&& operation) {
  const CipherSnapshot source_before =
      CaptureCipher(source, context + " source");
  std::forward<Operation>(operation)();
  if (destination != source) {
    RequireCipherPreserved(source_before, source, context + " source");
  }
}
// EXACT_SOURCE_PRESERVATION_END

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

void RequireInputExpression(const Json& expression,
                            const std::string& input_name) {
  if (!expression.is_object() || expression.size() != 1 ||
      !expression.contains("input") ||
      expression.at("input").get<std::string>() != input_name) {
    Fail("fixture expression does not name expected input " + input_name);
  }
}

const Json& RequireBinaryExpression(const Json& family,
                                    const std::string& operation) {
  const Json& expected = family.at("expected");
  if (!expected.is_object() || expected.size() != 2 ||
      expected.at("op").get<std::string>() != operation ||
      !expected.at("args").is_array() || expected.at("args").size() != 2) {
    Fail("fixture binary expression disagrees with " + operation);
  }
  RequireInputExpression(expected.at("args").at(0), "complex_x");
  return expected;
}

double RequireRealScalar(const Json& expression) {
  const Json& scalar = expression.at("scalar");
  if (!scalar.is_array() || scalar.size() != 2 ||
      scalar.at(1).get<double>() != 0.0) {
    Fail("ordinary scalar fixture must contain a real scalar");
  }
  return scalar.at(0).get<double>();
}

CaseResult RunCipherCase(const Json& family, const std::string& alias,
                         const Json& fixture,
                         ObjectArena& arena) {
  const std::string family_id = family.at("id").get<std::string>();
  const auto* manifest = Get_phantom_context_manifest();
  const LevelCoordinates levels = ContextLevelCoordinates();
  const std::size_t slots = manifest->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const auto y = ExpandInput(fixture.at("inputs").at("complex_y"), slots);
  CIPHER result = nullptr;
  std::uint64_t dropped_modulus = 0;
  const std::string case_label = family_id + "." + alias;

  if (family_id == "add_ct_ct" || family_id == "sub_ct_ct" ||
      family_id == "mul_ct_ct") {
    const std::string operation = family_id == "add_ct_ct"
                                      ? "add"
                                      : (family_id == "sub_ct_ct"
                                             ? "subtract"
                                             : "multiply");
    const Json& expected = RequireBinaryExpression(family, operation);
    RequireInputExpression(expected.at("args").at(1), "complex_y");
    CIPHER left = EncryptComplex(arena, x, levels._full);
    CIPHER right = EncryptComplex(arena, y, levels._full);
    result = Destination(arena, alias, left, right);
    if (family_id == "add_ct_ct") {
      InvokeCipherBinaryPreservingOperands(
          result, left, right, case_label,
          [&] { Add_ciph(result, left, right); });
    }
    if (family_id == "sub_ct_ct") {
      InvokeCipherBinaryPreservingOperands(
          result, left, right, case_label,
          [&] { Sub_ciph(result, left, right); });
    }
    if (family_id == "mul_ct_ct") {
      InvokeCipherBinaryPreservingOperands(
          result, left, right, case_label,
          [&] { Mul_ciph(result, left, right); });
    }
  } else if (family_id == "add_ct_plain" ||
             family_id == "sub_ct_plain" ||
             family_id == "mul_ct_plain") {
    const std::string operation = family_id == "add_ct_plain"
                                      ? "add"
                                      : (family_id == "sub_ct_plain"
                                             ? "subtract"
                                             : "multiply");
    const Json& expected = RequireBinaryExpression(family, operation);
    RequireInputExpression(expected.at("args").at(1), "complex_y");
    CIPHER left = EncryptComplex(arena, x, levels._full);
    PLAIN right = EncodeComplex(arena, y, levels._full);
    result = Destination(arena, alias, left);
    if (family_id == "add_ct_plain") {
      InvokeCipherPlainPreservingOperands(
          result, left, right, case_label,
          [&] { Add_plain(result, left, right); });
    }
    if (family_id == "sub_ct_plain") {
      InvokeCipherPlainPreservingOperands(
          result, left, right, case_label,
          [&] { Sub_plain(result, left, right); });
    }
    if (family_id == "mul_ct_plain") {
      InvokeCipherPlainPreservingOperands(
          result, left, right, case_label,
          [&] { Mul_plain(result, left, right); });
    }
  } else if (family_id == "add_ct_scalar" ||
             family_id == "sub_ct_scalar" ||
             family_id == "mul_ct_scalar") {
    const std::string operation = family_id == "add_ct_scalar"
                                      ? "add"
                                      : (family_id == "sub_ct_scalar"
                                             ? "subtract"
                                             : "multiply");
    const Json& expected = RequireBinaryExpression(family, operation);
    const double scalar = RequireRealScalar(expected.at("args").at(1));
    CIPHER left = EncryptComplex(arena, x, levels._full);
    result = Destination(arena, alias, left);
    if (family_id == "add_ct_scalar") {
      InvokeCipherUnaryPreservingSource(
          result, left, case_label,
          [&] { Add_scalar(result, left, scalar); });
    }
    if (family_id == "sub_ct_scalar") {
      InvokeCipherUnaryPreservingSource(
          result, left, case_label,
          [&] { Sub_scalar(result, left, scalar); });
    }
    if (family_id == "mul_ct_scalar") {
      InvokeCipherUnaryPreservingSource(
          result, left, case_label,
          [&] { Mul_scalar(result, left, scalar); });
    }
  } else if (family_id == "copy") {
    RequireInputExpression(family.at("expected"), "complex_x");
    CIPHER source = EncryptComplex(arena, x, levels._full);
    result = Destination(arena, alias, source);
    InvokeCipherUnaryPreservingSource(
        result, source, case_label, [&] { Copy_ciph(result, source); });
  } else if (family_id == "query_all") {
    RequireInputExpression(family.at("expected"), "complex_x");
    result = EncryptComplex(arena, x, levels._full);
  } else if (family_id == "modswitch") {
    RequireInputExpression(family.at("expected"), "complex_x");
    CIPHER source = EncryptComplex(arena, x, levels._full);
    result = Destination(arena, alias, source);
    InvokeCipherUnaryPreservingSource(
        result, source, case_label, [&] { Mod_switch(result, source); });
  } else if (family_id == "relinearize") {
    const Json& expected = RequireBinaryExpression(family, "multiply");
    RequireInputExpression(expected.at("args").at(1), "complex_y");
    CIPHER left = EncryptComplex(arena, x, levels._full);
    CIPHER right = EncryptComplex(arena, y, levels._full);
    CIPHER product = arena.NewCipher();
    InvokeCipherBinaryPreservingOperands(
        product, left, right, case_label + " setup multiply",
        [&] { Mul_ciph(product, left, right); });
    result = Destination(arena, alias, product);
    InvokeCipherUnaryPreservingSource(
        result, product, case_label, [&] { Relin(result, product); });
  } else if (family_id == "rescale") {
    const Json& expected = RequireBinaryExpression(family, "multiply");
    RequireInputExpression(expected.at("args").at(1), "complex_y");
    CIPHER left = EncryptComplex(arena, x, levels._full);
    CIPHER right = EncryptComplex(arena, y, levels._full);
    CIPHER product = arena.NewCipher();
    CIPHER relinearized = arena.NewCipher();
    InvokeCipherBinaryPreservingOperands(
        product, left, right, case_label + " setup multiply",
        [&] { Mul_ciph(product, left, right); });
    InvokeCipherUnaryPreservingSource(
        relinearized, product, case_label + " setup relinearize",
        [&] { Relin(relinearized, product); });
    result = Destination(arena, alias, relinearized);
    InvokeCipherUnaryPreservingSource(
        result, relinearized, case_label,
        [&] { Rescale_ciph(result, relinearized); });
    const long double numerator = std::ldexp(
        static_cast<long double>(1.0),
        2 * static_cast<int>(manifest->_scaling_modulus_bits));
    dropped_modulus = static_cast<std::uint64_t>(
        std::llround(numerator / static_cast<long double>(Raw_scale(result))));
  } else if (family_id == "rotate_negative" ||
             family_id == "rotate_positive" ||
             family_id == "rotate_zero") {
    const Json& expected = family.at("expected");
    if (expected.at("op").get<std::string>() != "rotate") {
      Fail("fixture rotate expression has the wrong operation");
    }
    RequireInputExpression(expected.at("arg"), "complex_x");
    const std::int64_t step64 = expected.at("step").get<std::int64_t>();
    if (step64 < std::numeric_limits<int>::min() ||
        step64 > std::numeric_limits<int>::max()) {
      Fail("fixture rotation step exceeds runtime integer range");
    }
    const int step = static_cast<int>(step64);
    CIPHER source = EncryptComplex(arena, x, levels._full);
    result = Destination(arena, alias, source);
    InvokeCipherUnaryPreservingSource(
        result, source, case_label,
        [&] { Rotate_ciph(result, source, step); });
  } else {
    Fail("unsupported ciphertext case family " + family_id);
  }

  CaseResult output{DecodeCipher(result),
                    CipherMetadata(result, dropped_modulus)};
  arena.FreeLiveObjects();
  return output;
}

CaseResult RunEncodeCase(const Json& family, const Json& fixture,
                         ObjectArena& arena) {
  const std::string family_id = family.at("id").get<std::string>();
  const auto* manifest = Get_phantom_context_manifest();
  const LevelCoordinates levels = ContextLevelCoordinates();
  const std::size_t slots = manifest->_logical_slots;
  PLAIN plain = arena.NewPlain();
  if (family_id == "encode_complex_full") {
    RequireInputExpression(family.at("expected"), "complex_x");
    const auto values =
        ExpandInput(fixture.at("inputs").at("complex_x"), slots);
    std::vector<DCMPLX> input(values.begin(), values.end());
    Encode_dcmplx(plain, input.data(), input.size(), 1.0, levels._full);
  } else if (family_id == "encode_mask_f32_full") {
    RequireInputExpression(family.at("expected"), "mask_f32");
    const auto& specification = fixture.at("inputs").at("mask_f32");
    const float value =
        specification.at("segments").at(0).at(2).get<float>();
    Encode_float_mask(plain, value, MaskLength(specification), 1.0,
                      levels._full);
  } else if (family_id == "encode_mask_f64_full") {
    RequireInputExpression(family.at("expected"), "mask_f64");
    const auto& specification = fixture.at("inputs").at("mask_f64");
    const double value =
        specification.at("segments").at(0).at(2).get<double>();
    Encode_double_mask(plain, value, MaskLength(specification), 1.0,
                       levels._full);
  } else if (family_id == "encode_real_f32_full") {
    RequireInputExpression(family.at("expected"), "real_f32");
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f32"), slots);
    std::vector<float> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    Encode_float(plain, input.data(), input.size(), 1.0, levels._full);
  } else if (family_id == "encode_real_f64_bottom" ||
             family_id == "encode_real_f64_middle" ||
             family_id == "encode_real_f64_full") {
    RequireInputExpression(family.at("expected"), "real_f64");
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f64"), slots);
    std::vector<double> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    const int level = family_id == "encode_real_f64_bottom"
                          ? levels._bottom
                          : (family_id == "encode_real_f64_middle"
                                 ? levels._middle
                                 : levels._full);
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
                              ? RunEncodeCase(family, fixture, arena)
                              : RunCipherCase(family, alias, fixture, arena);
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

Json RunOwnership(const Json& fixture) {
  const LevelCoordinates levels = ContextLevelCoordinates();
  const std::size_t slots = Get_phantom_context_manifest()->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const double hard_maximum_absolute =
      fixture.at("tolerances").at("hard_maximum_absolute").get<double>();
  std::vector<std::unique_ptr<CIPHERTEXT[]>> retained_arrays;
  std::size_t retained_array_count = 0;
  for (const auto& ownership_case : fixture.at("ownership_cases")) {
    const std::string case_id = ownership_case.at("id").get<std::string>();
    if (case_id == "free_array_1" || case_id == "free_array_4") {
      const int iterations = ownership_case.at("iterations").get<int>();
      if (iterations < 1) Fail("ownership iterations must be positive");
      retained_array_count += static_cast<std::size_t>(iterations);
    }
  }
  retained_arrays.reserve(retained_array_count);
  ObjectArena arena;
  std::set<std::string> seen_cases;
  Json reports = Json::array();
  std::size_t total_iterations = 0;

  for (const auto& ownership_case : fixture.at("ownership_cases")) {
    const std::string case_id = ownership_case.at("id").get<std::string>();
    const int iterations = ownership_case.at("iterations").get<int>();
    if (!seen_cases.insert(case_id).second) {
      Fail("duplicate ownership case " + case_id);
    }
    if (iterations < 1) Fail("ownership iterations must be positive");
    total_iterations += static_cast<std::size_t>(iterations);

    Json report = {
        {"id", case_id},
        {"iterations", iterations},
        {"status", "pass"},
    };
    if (case_id == "copy_independence") {
      if (ownership_case.contains("array_length")) {
        Fail("copy_independence cannot specify array_length");
      }
      for (int iteration = 0; iteration < iterations; ++iteration) {
        CIPHER source = EncryptComplex(arena, x, levels._full);
        CIPHER copy = arena.NewCipher();
        Copy_ciph(copy, source);
        Add_scalar(source, source, 0.25);
        if (MaximumError(DecodeCipher(copy), x) > hard_maximum_absolute) {
          Fail("copy independence failed");
        }
        arena.FreeCipher(source);
        arena.FreeCipher(copy);
      }
    } else if (case_id == "destination_reuse") {
      if (ownership_case.contains("array_length")) {
        Fail("destination_reuse cannot specify array_length");
      }
      std::vector<Complex> expected_reuse = x;
      for (Complex& value : expected_reuse) value += Complex(0.125, 0.0);
      CIPHER reusable = arena.NewCipher();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        CIPHER source = EncryptComplex(arena, x, levels._full);
        Copy_ciph(reusable, source);
        Add_scalar(reusable, reusable, 0.125);
        if (MaximumError(DecodeCipher(reusable), expected_reuse) >
            hard_maximum_absolute) {
          Fail("destination reuse result failed");
        }
        arena.FreeCipher(source);
      }
      arena.FreeCipher(reusable);
    } else if (case_id == "free_array_1" || case_id == "free_array_4") {
      const std::size_t length =
          ownership_case.at("array_length").get<std::size_t>();
      const std::size_t required_length = case_id == "free_array_1" ? 1 : 4;
      if (length != required_length) {
        Fail(case_id + " has an unexpected array_length");
      }
      report["array_length"] = length;
      for (int iteration = 0; iteration < iterations; ++iteration) {
        auto array = std::make_unique<CIPHERTEXT[]>(length);
        PLAIN plain = EncodeComplex(arena, x, levels._full);
        for (std::size_t index = 0; index < length; ++index) {
          Phantom_encrypt_plain(&array[index], plain);
        }
        Free_ciph_array(array.get(), length);
        arena.FreePlain(plain);
        retained_arrays.push_back(std::move(array));
      }
    } else if (case_id == "zero_then_free") {
      if (ownership_case.contains("array_length")) {
        Fail("zero_then_free cannot specify array_length");
      }
      for (int iteration = 0; iteration < iterations; ++iteration) {
        CIPHER value = arena.NewCipher();
        Zero_ciph(value);
        arena.FreeCipher(value);
      }
    } else {
      Fail("unknown ownership case " + case_id);
    }
    reports.push_back(std::move(report));
  }
  arena.FreeLiveObjects();
  RequireCuda(cudaDeviceSynchronize(), "ownership cudaDeviceSynchronize");
  return {
      {"schema_version", "ace.phantom.ordinary_ckks.ownership/1.0.0"},
      {"status", "pass"},
      {"total_iterations", total_iterations},
      {"cases", std::move(reports)},
  };
}

void RunRejection(const std::string& rejection_id, const Json& fixture) {
  bool declared_runtime_rejection = false;
  for (const auto& rejection : fixture.at("rejections")) {
    if (rejection.at("id").get<std::string>() == rejection_id) {
      if (rejection.at("phase").get<std::string>() != "runtime") {
        Fail("rejection case is not a runtime contract: " + rejection_id);
      }
      declared_runtime_rejection = true;
      break;
    }
  }
  if (!declared_runtime_rejection) {
    Fail("runtime rejection is absent from the fixture: " + rejection_id);
  }
  const LevelCoordinates levels = ContextLevelCoordinates();
  const std::size_t slots = Get_phantom_context_manifest()->_logical_slots;
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const auto y = ExpandInput(fixture.at("inputs").at("complex_y"), slots);
  ObjectArena arena;

  if (rejection_id == "bottom_modswitch" ||
      rejection_id == "bottom_rescale") {
    CIPHER source = EncryptComplex(arena, x, levels._bottom);
    CIPHER result = arena.NewCipher();
    if (rejection_id == "bottom_modswitch") Mod_switch(result, source);
    Rescale_ciph(result, source);
  } else if (rejection_id == "double_free") {
    CIPHER value = EncryptComplex(arena, x, levels._full);
    Free_ciph(value);
    Free_ciph(value);
  } else if (rejection_id == "use_after_free") {
    CIPHER value = EncryptComplex(arena, x, levels._full);
    Free_ciph(value);
    (void)Level(value);
  } else if (rejection_id == "relinearize_size2") {
    CIPHER value = EncryptComplex(arena, x, levels._full);
    CIPHER result = arena.NewCipher();
    Relin(result, value);
  } else if (rejection_id == "missing_loaded_relin_key") {
    CIPHER left = EncryptComplex(arena, x, levels._full);
    CIPHER right = EncryptComplex(arena, y, levels._full);
    CIPHER product = arena.NewCipher();
    CIPHER result = arena.NewCipher();
    Mul_ciph(product, left, right);
    Relin(result, product);
  } else if (rejection_id == "missing_loaded_rotation_key") {
    CIPHER value = EncryptComplex(arena, x, levels._full);
    CIPHER result = arena.NewCipher();
    Rotate_ciph(result, value, 3);
  } else if (rejection_id == "invalid_ciphertext_size") {
    CIPHER value = EncryptComplex(arena, x, levels._full);
    value->resize(4, value->coeff_modulus_size(),
                  value->poly_modulus_degree(), nullptr);
    (void)Get_ciph_size(value);
  } else if (rejection_id == "runtime_level_mismatch") {
    CIPHER left = EncryptComplex(arena, x, levels._full);
    CIPHER right = EncryptComplex(arena, y, levels._after_one_drop);
    CIPHER result = arena.NewCipher();
    Add_ciph(result, left, right);
  } else if (rejection_id == "runtime_scale_mismatch") {
    CIPHER left = EncryptComplex(arena, x, levels._full, 1.0);
    CIPHER right = EncryptComplex(arena, y, levels._full, 2.0);
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
      if (argc != 5) Fail("ownership mode requires an output path");
      WriteJson(argv[4], RunOwnership(fixture));
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
