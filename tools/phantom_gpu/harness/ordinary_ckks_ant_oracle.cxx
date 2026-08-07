#include "ckks/cipher.h"
#include "ckks/ciphertext.h"
#include "ckks/plain.h"
#include "ckks/plaintext.h"
#include "common/rt_api.h"
#include "context/ckks_context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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
#error "ACE_PRODUCER_ID must identify the tested ANT oracle"
#endif

namespace {

using Json = nlohmann::json;
using Complex = std::complex<double>;

constexpr std::size_t kArithmeticLevel = 4;
CKKS_PARAMS* context_parameters = nullptr;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
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

void ConfigureContext(const Json& manifest, const Json& resources) {
  if (manifest.at("schema_version") != 1 || manifest.at("packing") != "full" ||
      resources.at("schema_version") != 1 ||
      resources.at("context_schema_version") != 1) {
    Fail("compiler manifest schema or packing is unsupported");
  }
  const std::size_t data_q_count = manifest.at("data_q_bit_sizes").size();
  if (data_q_count == 0) Fail("compiler manifest data-Q list is empty");
  const auto& rotations = resources.at("rotation_steps");
  const std::size_t byte_count =
      sizeof(CKKS_PARAMS) + rotations.size() * sizeof(std::int32_t);
  context_parameters = static_cast<CKKS_PARAMS*>(std::calloc(1, byte_count));
  if (context_parameters == nullptr) Fail("cannot allocate ANT context view");
  context_parameters->_provider = LIB_ANT;
  context_parameters->_poly_degree =
      manifest.at("polynomial_degree").get<std::uint32_t>();
  context_parameters->_sec_level =
      manifest.at("security_level").get<std::size_t>();
  context_parameters->_mul_depth = data_q_count - 1;
  context_parameters->_input_level =
      manifest.at("input_level").get<std::size_t>();
  context_parameters->_first_mod_size =
      manifest.at("first_modulus_bits").get<std::size_t>();
  context_parameters->_scaling_mod_size =
      manifest.at("scaling_modulus_bits").get<std::size_t>();
  context_parameters->_num_q_parts =
      manifest.at("q_part_count").get<std::size_t>();
  context_parameters->_hamming_weight =
      manifest.at("hamming_weight").get<std::size_t>();
  context_parameters->_num_rot_idx = rotations.size();
  for (std::size_t index = 0; index < rotations.size(); ++index) {
    context_parameters->_rot_idxs[index] =
        rotations.at(index).get<std::int32_t>();
  }
}

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

PLAIN EncodeComplex(const std::vector<Complex>& values, std::size_t level,
                    std::uint32_t scale_degree = 1) {
  PLAIN plain = Alloc_plaintext();
  std::vector<DCMPLX> input(values.begin(), values.end());
  Encode_dcmplx(plain, input.data(), input.size(), scale_degree, level);
  return plain;
}

CIPHER EncryptComplex(const std::vector<Complex>& values, std::size_t level,
                      std::uint32_t scale_degree = 1) {
  PLAIN plain = EncodeComplex(values, level, scale_degree);
  CIPHER cipher = Alloc_ciphertext();
  Encrypt(cipher, plain);
  Free_plaintext(plain);
  return cipher;
}

std::vector<Complex> DecodeCipher(CIPHER cipher) {
  DCMPLX* raw = Get_msg_with_imag(cipher);
  const std::size_t slots = Get_slots(cipher);
  std::vector<Complex> result(raw, raw + slots);
  std::free(raw);
  return result;
}

std::vector<Complex> DecodePlain(PLAIN plain) {
  DCMPLX* raw = Get_dcmplx_msg_from_plain(plain);
  const std::size_t slots = Get_plain_slots(plain);
  std::vector<Complex> result(raw, raw + slots);
  std::free(raw);
  return result;
}

Json ValuesJson(const std::vector<Complex>& values) {
  Json result = Json::array();
  for (const Complex& value : values) {
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      Fail("ANT result contains NaN or Inf");
    }
    result.push_back(Json::array({value.real(), value.imag()}));
  }
  return result;
}

std::vector<Complex> Negated(std::vector<Complex> values) {
  for (Complex& value : values) value = -value;
  return values;
}

std::vector<Complex> Broadcast(std::size_t slots, double value) {
  return std::vector<Complex>(slots, Complex(value, 0.0));
}

std::vector<Complex> CipherCase(const std::string& family_id,
                                const Json& fixture, std::size_t slots) {
  const auto x = ExpandInput(fixture.at("inputs").at("complex_x"), slots);
  const auto y = ExpandInput(fixture.at("inputs").at("complex_y"), slots);
  CIPHER left = nullptr;
  CIPHER right = nullptr;
  CIPHER result = nullptr;
  PLAIN plain = nullptr;
  CIPHER3 product3 = nullptr;

  if (family_id == "add_ct_ct" || family_id == "sub_ct_ct" ||
      family_id == "mul_ct_ct") {
    left = EncryptComplex(x, kArithmeticLevel);
    right = EncryptComplex(y, kArithmeticLevel);
    result = Alloc_ciphertext();
    if (family_id == "add_ct_ct") Add_ciph(result, left, right);
    if (family_id == "sub_ct_ct") Sub_ciph(result, left, right);
    if (family_id == "mul_ct_ct") Mul_ciph(result, left, right);
  } else if (family_id == "add_ct_plain" ||
             family_id == "sub_ct_plain" ||
             family_id == "mul_ct_plain") {
    left = EncryptComplex(x, kArithmeticLevel);
    plain = EncodeComplex(family_id == "sub_ct_plain" ? Negated(y) : y,
                          kArithmeticLevel);
    result = Alloc_ciphertext();
    if (family_id == "mul_ct_plain") {
      Mul_plain(result, left, plain);
    } else {
      Add_plain(result, left, plain);
    }
  } else if (family_id == "add_ct_scalar" ||
             family_id == "sub_ct_scalar" ||
             family_id == "mul_ct_scalar") {
    left = EncryptComplex(x, kArithmeticLevel);
    double scalar = family_id == "mul_ct_scalar" ? -0.75 : 0.5;
    if (family_id == "sub_ct_scalar") scalar = -scalar;
    plain = EncodeComplex(Broadcast(slots, scalar), kArithmeticLevel);
    result = Alloc_ciphertext();
    if (family_id == "mul_ct_scalar") {
      Mul_plain(result, left, plain);
    } else {
      Add_plain(result, left, plain);
    }
  } else if (family_id == "copy" || family_id == "query_all") {
    left = EncryptComplex(x, kArithmeticLevel);
    result = Alloc_ciphertext();
    Copy_ciph(result, left);
  } else if (family_id == "modswitch") {
    left = EncryptComplex(x, kArithmeticLevel);
    result = Alloc_ciphertext();
    Copy_ciph(result, left);
    Modswitch_ciph(result);
  } else if (family_id == "relinearize") {
    left = EncryptComplex(x, kArithmeticLevel);
    right = EncryptComplex(y, kArithmeticLevel);
    product3 = Alloc_ciphertext3();
    result = Alloc_ciphertext();
    Mul_ciph3(product3, left, right);
    Relin(result, product3);
  } else if (family_id == "rescale") {
    left = EncryptComplex(x, kArithmeticLevel);
    right = EncryptComplex(y, kArithmeticLevel);
    CIPHER product = Alloc_ciphertext();
    result = Alloc_ciphertext();
    Mul_ciph(product, left, right);
    Rescale_ciph(result, product);
    Free_ciphertext(product);
  } else if (family_id == "rotate_negative" ||
             family_id == "rotate_positive" ||
             family_id == "rotate_zero") {
    left = EncryptComplex(x, kArithmeticLevel);
    result = Alloc_ciphertext();
    const int step = family_id == "rotate_negative"
                         ? -3
                         : (family_id == "rotate_positive" ? 3 : 0);
    if (step == 0) {
      Copy_ciph(result, left);
    } else {
      Rotate_ciph(result, left, step);
    }
  } else {
    Fail("unsupported ANT ciphertext case family " + family_id);
  }

  std::vector<Complex> values = DecodeCipher(result);
  if (left != nullptr) Free_ciphertext(left);
  if (right != nullptr) Free_ciphertext(right);
  if (result != nullptr) Free_ciphertext(result);
  if (plain != nullptr) Free_plaintext(plain);
  if (product3 != nullptr) Free_ciphertext3(product3);
  return values;
}

std::vector<Complex> EncodeCase(const std::string& family_id,
                                const Json& fixture, std::size_t slots,
                                std::size_t data_q_count) {
  PLAIN plain = Alloc_plaintext();
  if (family_id == "encode_complex_q4") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("complex_x"), slots);
    std::vector<DCMPLX> input(values.begin(), values.end());
    Encode_dcmplx(plain, input.data(), input.size(), 1, kArithmeticLevel);
  } else if (family_id == "encode_mask_f32_q4") {
    const auto& specification = fixture.at("inputs").at("mask_f32");
    Encode_float_mask(
        plain, specification.at("segments").at(0).at(2).get<float>(),
        MaskLength(specification), 1, kArithmeticLevel);
  } else if (family_id == "encode_mask_f64_q4") {
    const auto& specification = fixture.at("inputs").at("mask_f64");
    Encode_double_mask(
        plain, specification.at("segments").at(0).at(2).get<double>(),
        MaskLength(specification), 1, kArithmeticLevel);
  } else if (family_id == "encode_real_f32_q4") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f32"), slots);
    std::vector<float> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    Encode_float(plain, input.data(), input.size(), 1, kArithmeticLevel);
  } else if (family_id == "encode_real_f64_bottom" ||
             family_id == "encode_real_f64_middle" ||
             family_id == "encode_real_f64_full") {
    const auto values =
        ExpandInput(fixture.at("inputs").at("real_f64"), slots);
    std::vector<double> input;
    input.reserve(values.size());
    for (const auto& value : values) input.push_back(value.real());
    const std::size_t level = family_id == "encode_real_f64_bottom"
                                  ? 1
                                  : (family_id == "encode_real_f64_middle"
                                         ? (data_q_count + 1) / 2
                                         : data_q_count);
    Encode_double(plain, input.data(), input.size(), 1, level);
  } else {
    Fail("unsupported ANT encode case family " + family_id);
  }
  std::vector<Complex> values = DecodePlain(plain);
  Free_plaintext(plain);
  return values;
}

Json RunOracle(const Json& fixture, std::size_t slots,
               std::size_t data_q_count) {
  Json records = Json::array();
  for (const auto& family : fixture.at("case_families")) {
    const std::string family_id = family.at("id").get<std::string>();
    const auto values = family_id.rfind("encode_", 0) == 0
                            ? EncodeCase(family_id, fixture, slots,
                                         data_q_count)
                            : CipherCase(family_id, fixture, slots);
    if (values.size() != slots) {
      Fail("ANT decoded length disagrees with compiler logical slots");
    }
    for (const auto& alias : family.at("aliases")) {
      records.push_back({
          {"case_id", family_id + "." + alias.get<std::string>()},
          {"values", ValuesJson(values)},
          {"metadata", Json::object()},
      });
    }
  }
  return {
      {"schema_version", "ace.phantom.ordinary_ckks.raw-provider/1.0.0"},
      {"provider", "ant"},
      {"fixture_id", fixture.at("fixture_id")},
      {"fixture_sha256", ACE_FIXTURE_SHA256},
      {"compiler_context_manifest_sha256", ACE_CONTEXT_MANIFEST_SHA256},
      {"producer", ACE_PRODUCER_ID},
      {"independent_encoding", true},
      {"input_kind", "clear_fixture_values"},
      {"records", std::move(records)},
  };
}

}  // namespace

extern "C" {
CKKS_PARAMS* Get_context_params() {
  if (context_parameters == nullptr) std::abort();
  return context_parameters;
}
RT_DATA_INFO* Get_rt_data_info() { return nullptr; }
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: ordinary_ckks_ant_oracle <fixture.json> "
                   "<context-manifest.json> <resource-manifest.json> "
                   "<output.json> <disable-bootstrap-precompute-token>\n";
      return 2;
    }
    const Json fixture = LoadJson(argv[1]);
    const Json manifest = LoadJson(argv[2]);
    const Json resources = LoadJson(argv[3]);
    if (std::string(argv[5]) != "disable-native-bootstrap-precompute") {
      Fail("explicit native-bootstrap precompute disable token is required");
    }
    ConfigureContext(manifest, resources);
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
    Prepare_context();
    if (Get_q_cnt() != manifest.at("data_q_bit_sizes").size() ||
        Get_p_cnt() != manifest.at("special_p_bit_sizes").size()) {
      Fail("ANT Q/P counts disagree with the compiler context manifest");
    }
    const std::size_t slots =
        manifest.at("logical_slot_capacity").get<std::size_t>();
    WriteJson(argv[4],
              RunOracle(fixture, slots,
                        manifest.at("data_q_bit_sizes").size()));
    Finalize_context();
    std::free(context_parameters);
    context_parameters = nullptr;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ordinary CKKS ANT oracle failed: " << error.what() << '\n';
    return 1;
  }
}
