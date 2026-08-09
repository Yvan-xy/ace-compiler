#ifndef ACE_GENERATED_BOOTSTRAP_ANT_COMMON_H
#define ACE_GENERATED_BOOTSTRAP_ANT_COMMON_H

#include "ckks/cipher.h"
#include "ckks/plain.h"
#include "common/rt_api.h"
#include "context/ckks_context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace generated_bootstrap_ant {

using Complex = std::complex<double>;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kValueMagic = {'A', 'C', 'E', 'B',
                                                      'S', 'C', '0', '1'};
constexpr char kValueFormat[] =
    "ace.bootstrap-correctness.complex_float64le/1.0.0";
constexpr char kFixtureSchema[] =
    "ace.phantom.bootstrap-correctness-fixture/1.0.0";
constexpr char kInvocationSchema[] =
    "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0";
constexpr char kSemanticsSchema[] =
    "ace.phantom.generated-bootstrap.semantics/1.0.0";
constexpr char kPostOperationSchema[] =
    "ace.phantom.bootstrap-post-operation-semantics/1.0.0";
constexpr double kRequiredMaximumError = 1.0e-2;

[[noreturn]] inline void Fail(const std::string& message) {
  throw std::runtime_error("GENERATED_BOOTSTRAP_ANT: " + message);
}

inline void Require(bool condition, const std::string& message) {
  if (!condition) Fail(message);
}

inline std::uint32_t RotateRight(std::uint32_t value, std::uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

inline std::string Sha256(const std::uint8_t* bytes, std::size_t length) {
  static constexpr std::array<std::uint32_t, 64> constants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
  std::array<std::uint32_t, 8> state = {0x6a09e667U, 0xbb67ae85U,
                                        0x3c6ef372U, 0xa54ff53aU,
                                        0x510e527fU, 0x9b05688cU,
                                        0x1f83d9abU, 0x5be0cd19U};
  const std::size_t padded = ((length + 9U + 63U) / 64U) * 64U;
  std::vector<std::uint8_t> message(padded, 0);
  if (length != 0) std::copy(bytes, bytes + length, message.begin());
  message[length] = 0x80U;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8U;
  for (std::size_t index = 0; index < 8; ++index) {
    message[padded - 1U - index] =
        static_cast<std::uint8_t>(bit_length >> (index * 8U));
  }
  for (std::size_t offset = 0; offset < padded; offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t position = offset + index * 4U;
      words[index] =
          (static_cast<std::uint32_t>(message[position]) << 24U) |
          (static_cast<std::uint32_t>(message[position + 1]) << 16U) |
          (static_cast<std::uint32_t>(message[position + 2]) << 8U) |
          static_cast<std::uint32_t>(message[position + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
      const std::uint32_t a = words[index - 15];
      const std::uint32_t b = words[index - 2];
      const std::uint32_t s0 =
          RotateRight(a, 7) ^ RotateRight(a, 18) ^ (a >> 3);
      const std::uint32_t s1 =
          RotateRight(b, 17) ^ RotateRight(b, 19) ^ (b >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (std::size_t index = 0; index < 64; ++index) {
      const std::uint32_t s1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choose = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 =
          h + s1 + choose + constants[index] + words[index];
      const std::uint32_t s0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    const std::array<std::uint32_t, 8> work = {a, b, c, d, e, f, g, h};
    for (std::size_t index = 0; index < state.size(); ++index)
      state[index] += work[index];
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint32_t value : state) output << std::setw(8) << value;
  return output.str();
}

inline std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  return Sha256(bytes.data(), bytes.size());
}

inline std::vector<std::uint8_t> ReadBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  Require(static_cast<bool>(input), "cannot open " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff length = input.tellg();
  Require(length >= 0, "cannot size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  if (!bytes.empty())
    input.read(reinterpret_cast<char*>(bytes.data()), length);
  Require(static_cast<bool>(input), "cannot read " + path);
  return bytes;
}

inline std::string Sha256File(const std::string& path) {
  return Sha256(ReadBytes(path));
}

inline Json LoadJson(const std::string& path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "cannot open JSON " + path);
  Json value;
  input >> value;
  Require(static_cast<bool>(input), "cannot parse JSON " + path);
  Require(value.is_object(), "JSON root is not an object: " + path);
  return value;
}

inline void WriteBytes(const std::string& path,
                       const std::vector<std::uint8_t>& bytes) {
  std::ifstream existing(path, std::ios::binary);
  Require(!existing.good(), "refusing to replace existing output " + path);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "cannot create " + path);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  output.flush();
  Require(static_cast<bool>(output), "cannot write " + path);
}

inline void WriteJson(const std::string& path, const Json& value) {
  std::ifstream existing(path, std::ios::binary);
  Require(!existing.good(), "refusing to replace existing output " + path);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "cannot create " + path);
  output << value.dump(2) << '\n';
  output.flush();
  Require(static_cast<bool>(output), "cannot write " + path);
}

inline std::uint64_t RequiredUnsigned(const Json& value,
                                      const std::string& name) {
  const Json& item = value.at(name);
  Require(item.is_number_unsigned() || item.is_number_integer(),
          name + " is not an integer");
  const std::int64_t signed_value = item.get<std::int64_t>();
  Require(signed_value >= 0, name + " is negative");
  return static_cast<std::uint64_t>(signed_value);
}

struct QualificationInputs {
  Json fixture;
  Json invocation;
  Json context;
  Json resources;
  Json constants;
  Json semantics;
  Json post_operation_attestation;
  Json options;
  Json bindings;
  std::string fixture_sha256;
  std::string ace_source_manifest_sha256;
  std::string phantom_source_manifest_sha256;
  std::string invocation_sha256;
  std::string raw_air_sha256;
  std::string post_ckks_air_sha256;
  std::string context_sha256;
  std::string resource_sha256;
  std::string constant_sha256;
  std::string semantics_sha256;
  std::string post_operations_air_sha256;
  std::string post_operation_attestation_sha256;
  std::size_t slots;
  std::size_t input_level;
  std::uint32_t encode_budget;
  std::uint32_t decode_budget;
  double domain_lower;
  double domain_upper;
  double maximum_error;
};

inline QualificationInputs LoadQualificationInputs(
    const std::array<std::string, 12>& paths) {
  QualificationInputs result;
  result.fixture = LoadJson(paths[0]);
  result.invocation = LoadJson(paths[3]);
  result.context = LoadJson(paths[6]);
  result.resources = LoadJson(paths[7]);
  result.constants = LoadJson(paths[8]);
  result.semantics = LoadJson(paths[9]);
  result.post_operation_attestation = LoadJson(paths[11]);
  result.fixture_sha256 = Sha256File(paths[0]);
  result.ace_source_manifest_sha256 = Sha256File(paths[1]);
  result.phantom_source_manifest_sha256 = Sha256File(paths[2]);
  result.invocation_sha256 = Sha256File(paths[3]);
  result.raw_air_sha256 = Sha256File(paths[4]);
  result.post_ckks_air_sha256 = Sha256File(paths[5]);
  result.context_sha256 = Sha256File(paths[6]);
  result.resource_sha256 = Sha256File(paths[7]);
  result.constant_sha256 = Sha256File(paths[8]);
  result.semantics_sha256 = Sha256File(paths[9]);
  result.post_operations_air_sha256 = Sha256File(paths[10]);
  result.post_operation_attestation_sha256 = Sha256File(paths[11]);

  Require(result.fixture.at("schema_version") == kFixtureSchema,
          "unsupported fixture schema");
  Require(result.invocation.at("schema_version") == kInvocationSchema,
          "unsupported compiler invocation schema");
  Require(result.semantics.at("schema_version") == kSemanticsSchema,
          "unsupported bootstrap semantics schema");
  Require(result.semantics.at("status") == "pass" &&
              result.post_operation_attestation.at("schema_version") ==
                  kPostOperationSchema &&
              result.post_operation_attestation.at("status") == "attested",
          "bootstrap or post-operation semantic attestation is incomplete");
  Require(result.context.at("schema_version") == 1 &&
              result.context.at("packing") == "full" &&
              result.resources.at("schema_version") == 3 &&
              result.constants.at("schema_version") == 1,
          "unsupported compiler manifest schema or packing");

  result.options = result.invocation.at("options");
  Require(result.options.is_object(), "compiler options are not an object");
  const std::array<const char*, 16> required_options = {
      "poly_degree", "vector_capacity", "mul_level", "input_level",
      "security_level", "scaling_factor_bits", "first_prime_bits",
      "hamming_weight", "q_part_count", "encode_transform_budget",
      "decode_transform_budget", "ciphertext_constant_encoding",
      "post_multiply_real", "post_multiply_imag",
      "post_multiply_scale_degree", "post_rotation_step"};
  for (const char* name : required_options)
    Require(result.options.contains(name),
            std::string("compiler invocation omits ") + name);
  Require(result.options.contains("packing"),
          "compiler invocation omits packing");
  Require(result.options.size() == required_options.size() + 1U,
          "compiler invocation contains unexpected typed options");

  result.slots = result.context.at("logical_slot_capacity").get<std::size_t>();
  result.input_level = result.context.at("input_level").get<std::size_t>();
  result.encode_budget = static_cast<std::uint32_t>(
      RequiredUnsigned(result.options, "encode_transform_budget"));
  result.decode_budget = static_cast<std::uint32_t>(
      RequiredUnsigned(result.options, "decode_transform_budget"));
  Require(result.encode_budget > 0 && result.decode_budget > 0,
          "transform budgets must be positive");
  Require(RequiredUnsigned(result.options, "poly_degree") ==
                  result.context.at("polynomial_degree").get<std::uint64_t>() &&
              RequiredUnsigned(result.options, "vector_capacity") == result.slots &&
              RequiredUnsigned(result.options, "mul_level") ==
                  result.context.at("data_q_bit_sizes").size() &&
              RequiredUnsigned(result.options, "input_level") == result.input_level &&
              RequiredUnsigned(result.options, "security_level") ==
                  result.context.at("security_level").get<std::uint64_t>() &&
              RequiredUnsigned(result.options, "scaling_factor_bits") ==
                  result.context.at("scaling_modulus_bits").get<std::uint64_t>() &&
              RequiredUnsigned(result.options, "first_prime_bits") ==
                  result.context.at("first_modulus_bits").get<std::uint64_t>() &&
              RequiredUnsigned(result.options, "hamming_weight") ==
                  result.context.at("hamming_weight").get<std::uint64_t>() &&
              RequiredUnsigned(result.options, "q_part_count") ==
                  result.context.at("q_part_count").get<std::uint64_t>() &&
              result.options.at("packing") == result.context.at("packing"),
          "explicit compiler options disagree with the context manifest");
  Require(result.options.at("ciphertext_constant_encoding").is_string() &&
              (result.options.at("ciphertext_constant_encoding") == "enabled" ||
               result.options.at("ciphertext_constant_encoding") == "disabled"),
          "ciphertext_constant_encoding is not an explicit mode");
  Require(result.options.at("post_multiply_real").is_number() &&
              result.options.at("post_multiply_imag").is_number() &&
              result.options.at("post_multiply_scale_degree").is_number_integer() &&
              result.options.at("post_rotation_step").is_number_integer(),
          "post-operation compiler options are not typed scalars");
  const std::int64_t rotation =
      result.options.at("post_rotation_step").get<std::int64_t>();
  const std::int64_t signed_slots = static_cast<std::int64_t>(result.slots);
  std::int64_t normalized_rotation = rotation % signed_slots;
  if (normalized_rotation < 0) normalized_rotation += signed_slots;
  if (normalized_rotation > signed_slots / 2)
    normalized_rotation -= signed_slots;
  Require(rotation == normalized_rotation && rotation != 0,
          "post-operation rotation is not canonical and nonzero");
  const Json& rotations = result.resources.at("rotation_steps");
  Require(rotations.is_array() &&
              std::find(rotations.begin(), rotations.end(), rotation) !=
                  rotations.end(),
          "post-operation rotation is absent from compiler resources");
  Require(result.context.at("polynomial_degree").get<std::size_t>() ==
              2U * result.slots,
          "logical slot capacity is not full-packed");
  Require(result.input_level > 0 &&
              result.input_level <= result.context.at("data_q_bit_sizes").size(),
          "compiler input level is invalid");

  result.bindings = result.fixture.at("bindings");
  Require(result.bindings.is_object(), "fixture bindings are not an object");
  const std::array<const char*, 11> required_bindings = {
      "ace_source_manifest_sha256",
      "phantom_source_manifest_sha256",
      "compiler_invocation_sha256",
      "raw_air_sha256",
      "post_ckks_air_sha256",
      "context_manifest_sha256",
      "resource_manifest_sha256",
      "constant_manifest_sha256",
      "bootstrap_semantics_sha256",
      "post_operations_air_sha256",
      "post_operation_attestation_sha256"};
  for (const char* name : required_bindings) {
    Require(result.bindings.contains(name) &&
                result.bindings.at(name).is_string() &&
                result.bindings.at(name).get<std::string>().size() == 64U &&
                std::all_of(
                    result.bindings.at(name).get_ref<const std::string&>().begin(),
                    result.bindings.at(name).get_ref<const std::string&>().end(),
                    [](unsigned char value) {
                      return (value >= '0' && value <= '9') ||
                             (value >= 'a' && value <= 'f');
                    }),
            std::string("fixture binding is missing or invalid: ") + name);
  }
  Require(result.bindings.size() == required_bindings.size(),
          "fixture has unexpected artifact bindings");
  const std::array<std::pair<const char*, const std::string*>, 11> hashes = {{
      {"ace_source_manifest_sha256", &result.ace_source_manifest_sha256},
      {"phantom_source_manifest_sha256",
       &result.phantom_source_manifest_sha256},
      {"compiler_invocation_sha256", &result.invocation_sha256},
      {"raw_air_sha256", &result.raw_air_sha256},
      {"post_ckks_air_sha256", &result.post_ckks_air_sha256},
      {"context_manifest_sha256", &result.context_sha256},
      {"resource_manifest_sha256", &result.resource_sha256},
      {"constant_manifest_sha256", &result.constant_sha256},
      {"bootstrap_semantics_sha256", &result.semantics_sha256},
      {"post_operations_air_sha256", &result.post_operations_air_sha256},
      {"post_operation_attestation_sha256",
       &result.post_operation_attestation_sha256},
  }};
  for (const auto& [name, digest] : hashes) {
    Require(result.bindings.at(name) == *digest,
            std::string("artifact binding mismatch: ") + name);
  }
  const Json& semantic_bindings = result.semantics.at("bindings");
  for (const char* name : {"compiler_invocation_sha256", "raw_air_sha256",
                           "post_ckks_air_sha256", "context_manifest_sha256",
                           "resource_manifest_sha256",
                           "constant_manifest_sha256"}) {
    Require(semantic_bindings.at(name) == result.bindings.at(name),
            std::string("bootstrap semantics binding mismatch: ") + name);
  }
  Require(semantic_bindings.at("post_operations_attestation_sha256") ==
              result.post_operation_attestation_sha256,
          "bootstrap semantics post-operation attestation binding mismatch");
  const Json& post_bindings = result.post_operation_attestation.at("bindings");
  for (const char* name : {"compiler_invocation_sha256",
                           "post_ckks_air_sha256", "context_manifest_sha256",
                           "resource_manifest_sha256",
                           "post_operations_air_sha256"}) {
    Require(post_bindings.at(name) == result.bindings.at(name),
            std::string("post-operation binding mismatch: ") + name);
  }

  const Json& domain = result.semantics.at("supported_identity_domain");
  Require(domain.at("kind") == "centered-evalmod-half-period",
          "unsupported identity-domain convention");
  result.domain_lower = domain.at("lower_exclusive").get<double>();
  result.domain_upper = domain.at("upper_exclusive").get<double>();
  Require(std::isfinite(result.domain_lower) &&
              std::isfinite(result.domain_upper) && result.domain_lower < 0.0 &&
              result.domain_upper > 0.0,
          "invalid supported identity domain");

  Require(result.fixture.at("supported_identity_domain")
                  .at("attested_domain") == domain &&
              result.fixture.at("supported_identity_domain")
                      .at("bootstrap_semantics_sha256") ==
                  result.semantics_sha256,
          "fixture identity domain differs from bootstrap semantics");
  result.maximum_error = result.fixture.at("tolerances")
                             .at("provider_clear_maximum_absolute")
                             .get<double>();
  Require(result.maximum_error == kRequiredMaximumError,
          "provider-versus-clear threshold is not the frozen value");
  Require(result.fixture.at("case_recipes").is_array() &&
              result.fixture.at("case_recipes").size() == 7U,
          "fixture case list does not contain the required coverage");
  Require(result.fixture.at("seed").at("algorithm") ==
              "splitmix64-float53-v1" &&
              result.fixture.at("seed").at("value").is_string(),
          "fixture deterministic seed contract is invalid");
  Require(result.fixture.at("repeat_contract") ==
              Json{{"calls_per_case", 3},
                   {"independently_owned_clones", true}} &&
              result.fixture.at("post_operations").is_object() &&
              !result.fixture.at("post_operations").empty(),
          "fixture post-operation or repeat contract is incomplete");
  return result;
}

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t state) : state_(state) {}
  std::uint64_t Next() {
    std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }
  double Symmetric(double bound) {
    const double unit = static_cast<double>(Next() >> 11U) * 0x1.0p-53;
    return (2.0 * unit - 1.0) * bound;
  }

 private:
  std::uint64_t state_;
};

struct FixtureCase {
  std::string id;
  std::string recipe;
  std::vector<Complex> clear;
};

inline void ValidateInsideDomain(const std::vector<Complex>& values,
                                 const QualificationInputs& inputs,
                                 const std::string& case_id) {
  Require(values.size() == inputs.slots,
          case_id + " does not contain every logical slot");
  for (std::size_t index = 0; index < values.size(); ++index) {
    const Complex value = values[index];
    Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
            case_id + " contains a non-finite clear value");
    Require(value.real() > inputs.domain_lower &&
                value.real() < inputs.domain_upper &&
                value.imag() > inputs.domain_lower &&
                value.imag() < inputs.domain_upper,
            case_id + " contains a value outside the supported identity domain");
  }
}

inline std::vector<FixtureCase> MaterializeCases(
    const QualificationInputs& inputs) {
  const std::uint64_t seed =
      std::stoull(inputs.fixture.at("seed").at("value").get<std::string>());
  std::vector<FixtureCase> cases;
  std::vector<std::string> ids;
  const std::array<const char*, 7> expected_ids = {
      "bounded-random-complex", "zero", "nonzero-real-constant",
      "nonzero-complex-constant", "alternating-real-imaginary-signs",
      "positive-boundary-inside", "negative-boundary-inside"};
  const std::array<const char*, 7> expected_kinds = {
      "bounded_random_complex", "zero", "real_constant",
      "complex_constant", "alternating_real_imaginary_signs",
      "positive_boundary_inside", "negative_boundary_inside"};
  std::size_t position = 0U;
  for (const Json& descriptor : inputs.fixture.at("case_recipes")) {
    const std::string id = descriptor.at("id").get<std::string>();
    const std::string recipe = descriptor.at("kind").get<std::string>();
    Require(id == expected_ids.at(position) &&
                recipe == expected_kinds.at(position),
            "fixture case order or coverage differs from the frozen contract");
    ++position;
    Require(!id.empty() && std::find(ids.begin(), ids.end(), id) == ids.end(),
            "fixture case identifiers are empty or duplicated");
    ids.push_back(id);
    std::vector<Complex> values(inputs.slots);
    if (recipe == "bounded_random_complex") {
      const double bound = descriptor.at("component_bound").get<double>();
      Require(std::isfinite(bound) && bound > 0.0,
              id + " has an invalid random bound");
      SplitMix64 generator(
          seed ^ std::stoull(descriptor.at("seed_xor").get<std::string>(),
                            nullptr, 16));
      for (Complex& value : values)
        value = Complex(generator.Symmetric(bound), generator.Symmetric(bound));
    } else if (recipe == "zero") {
      std::fill(values.begin(), values.end(), Complex{});
    } else if (recipe == "real_constant" || recipe == "complex_constant") {
      const Json& pair = descriptor.at("value");
      Require(pair.is_array() && pair.size() == 2U,
              id + " has an invalid constant value");
      const double real = pair.at(0).get<double>();
      const double imaginary = pair.at(1).get<double>();
      Require(real != 0.0 || imaginary != 0.0,
              id + " declares a zero constant");
      std::fill(values.begin(), values.end(), Complex(real, imaginary));
    } else if (recipe == "alternating_real_imaginary_signs") {
      const double magnitude = descriptor.at("magnitude").get<double>();
      Require(std::isfinite(magnitude) && magnitude > 0.0,
              id + " has invalid alternating magnitudes");
      for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = Complex(index % 2U == 0 ? magnitude : -magnitude,
                                index % 2U == 0 ? -magnitude : magnitude);
      }
    } else if (recipe == "positive_boundary_inside" ||
               recipe == "negative_boundary_inside") {
      const double offset =
          descriptor.at("offset_from_attested_boundary").get<double>();
      Require(std::isfinite(offset) && offset > 0.0,
              id + " has an invalid boundary offset");
      std::fill(values.begin(), values.end(),
                Complex(recipe == "positive_boundary_inside"
                            ? inputs.domain_upper - offset
                            : inputs.domain_lower + offset,
                        0.0));
    } else {
      Fail("unsupported fixture recipe " + recipe);
    }
    ValidateInsideDomain(values, inputs, id);
    cases.push_back({id, recipe, std::move(values)});
  }
  return cases;
}

inline PLAIN EncodeComplex(const std::vector<Complex>& values,
                           std::size_t level) {
  PLAIN plain = Alloc_plaintext();
  std::vector<DCMPLX> input(values.begin(), values.end());
  Encode_dcmplx(plain, input.data(), input.size(), 1, level);
  return plain;
}

inline CIPHER EncryptComplex(const std::vector<Complex>& values,
                             std::size_t level) {
  PLAIN plain = EncodeComplex(values, level);
  CIPHER cipher = Alloc_ciphertext();
  Encrypt(cipher, plain);
  Free_plaintext(plain);
  return cipher;
}

inline std::vector<Complex> Decode(CIPHER cipher, std::size_t slots) {
  DCMPLX* raw = Get_msg_with_imag(cipher);
  Require(raw != nullptr && Get_slots(cipher) == slots,
          "decoded slot count differs from the compiler context");
  std::vector<Complex> values(raw, raw + slots);
  std::free(raw);
  for (std::size_t index = 0; index < values.size(); ++index) {
    Require(std::isfinite(values[index].real()) &&
                std::isfinite(values[index].imag()),
            "decoded result contains a non-finite value");
  }
  return values;
}

inline void AppendU64Le(std::vector<std::uint8_t>& output,
                        std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
}

inline void AppendU32Le(std::vector<std::uint8_t>& output,
                        std::uint32_t value) {
  for (std::size_t byte = 0; byte < 4; ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
}

inline void AppendU16Le(std::vector<std::uint8_t>& output,
                        std::uint16_t value) {
  for (std::size_t byte = 0; byte < 2; ++byte)
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
}

inline void AppendDigest(std::vector<std::uint8_t>& output,
                         const std::string& digest) {
  Require(digest.size() == 64U, "invalid SHA-256 length");
  for (std::size_t index = 0; index < digest.size(); index += 2U) {
    const std::string octet = digest.substr(index, 2U);
    char* end = nullptr;
    const unsigned long value = std::strtoul(octet.c_str(), &end, 16);
    Require(end != nullptr && *end == '\0' && value <= 0xffU,
            "invalid SHA-256 character");
    output.push_back(static_cast<std::uint8_t>(value));
  }
}

inline void AppendDoubleLe(std::vector<std::uint8_t>& output, double value) {
  Require(std::isfinite(value), "cannot serialize a non-finite value");
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU64Le(output, bits);
}

inline Json AppendValues(std::vector<std::uint8_t>& payload,
                         const std::string& case_id,
                         const std::string& oracle,
                         const std::vector<Complex>& values,
                         Json metadata) {
  const std::size_t offset = payload.size();
  for (const Complex& value : values) {
    AppendDoubleLe(payload, value.real());
    AppendDoubleLe(payload, value.imag());
  }
  return {{"case_id", case_id},
          {"oracle", oracle},
          {"offset_bytes", offset},
          {"value_count", values.size()},
          {"metadata", std::move(metadata)}};
}

inline Json Metric(const std::vector<Complex>& actual,
                   const std::vector<Complex>& expected, double threshold) {
  Require(actual.size() == expected.size() && !actual.empty(),
          "metric vectors have unequal or zero length");
  double maximum = -1.0;
  std::size_t maximum_index = 0;
  double absolute_sum = 0.0;
  double squared_sum = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(actual[index] - expected[index]);
    Require(std::isfinite(error), "metric contains a non-finite error");
    if (error > maximum) {
      maximum = error;
      maximum_index = index;
    }
    absolute_sum += error;
    squared_sum += error * error;
  }
  const bool precision_infinite = maximum == 0.0;
  Json metric = {
      {"status", maximum <= threshold ? "pass" : "fail"},
      {"comparison_count", actual.size()},
      {"maximum_absolute_error", maximum},
      {"maximum_absolute_error_index", maximum_index},
      {"mean_absolute_error", absolute_sum / actual.size()},
      {"root_mean_square_error", std::sqrt(squared_sum / actual.size())},
      {"estimated_precision_bits",
       precision_infinite ? Json(nullptr) : Json(-std::log2(maximum))},
      {"estimated_precision_is_infinite", precision_infinite},
      {"threshold", threshold},
  };
  Require(maximum <= threshold,
          "provider result exceeds the frozen clear-identity threshold");
  return metric;
}

inline std::string CaseManifestSha256(const Json& descriptors) {
  Json manifest = Json::array();
  for (const Json& descriptor : descriptors) {
    manifest.push_back({{"case_id", descriptor.at("case_id")},
                        {"oracle", descriptor.at("oracle")},
                        {"value_count", descriptor.at("value_count")}});
  }
  const std::string canonical = manifest.dump() + "\n";
  return Sha256(reinterpret_cast<const std::uint8_t*>(canonical.data()),
                canonical.size());
}

inline std::vector<std::uint8_t> FinalizeValueFile(
    const std::vector<std::uint8_t>& payload, const Json& descriptors,
    const QualificationInputs& inputs) {
  const std::string manifest = CaseManifestSha256(descriptors);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(128U + payload.size());
  bytes.insert(bytes.end(), kValueMagic.begin(), kValueMagic.end());
  AppendU16Le(bytes, 1U);
  AppendU16Le(bytes, 0U);
  AppendU32Le(bytes, 0U);
  AppendU64Le(bytes, descriptors.size());
  AppendU64Le(bytes, payload.size());
  AppendDigest(bytes, inputs.fixture_sha256);
  AppendDigest(bytes, inputs.context_sha256);
  AppendDigest(bytes, manifest);
  Require(bytes.size() == 128U, "provider binary header size changed");
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

inline Json BinaryDescriptor(const std::vector<std::uint8_t>& bytes,
                             const std::vector<std::uint8_t>& payload,
                             const Json& descriptors) {
  return {{"format", kValueFormat},
          {"size_bytes", bytes.size()},
          {"sha256", Sha256(bytes)},
          {"header_size", 128},
          {"record_count", descriptors.size()},
          {"payload_bytes", payload.size()},
          {"case_manifest_sha256", CaseManifestSha256(descriptors)}};
}

inline Json Provenance(const QualificationInputs& inputs,
                       const std::string& executable_path) {
  Json value = {
      {"fixture_sha256", inputs.fixture_sha256},
      {"compiler_invocation_sha256", inputs.invocation_sha256},
      {"compiler_context_manifest_sha256", inputs.context_sha256},
      {"compiler_resource_manifest_sha256", inputs.resource_sha256},
      {"compiler_constant_manifest_sha256", inputs.constant_sha256},
      {"bootstrap_semantics_sha256", inputs.semantics_sha256},
      {"qualification_bindings", inputs.bindings},
      {"executable_sha256", Sha256File(executable_path)},
  };
  return value;
}

inline std::uint32_t AntPrimeSize(std::uint64_t prime) {
  Require(prime >= 2, "ANT context returned an invalid prime");
  const std::uint64_t original = prime;
  std::uint32_t floor_log2 = 0;
  while (prime > 1) {
    ++floor_log2;
    prime >>= 1;
  }
  Require(floor_log2 < 63, "ANT prime exceeds the supported size");
  const std::uint64_t lower = std::uint64_t{1} << floor_log2;
  const std::uint64_t upper = lower << 1;
  return original - lower <= upper - original ? floor_log2 : floor_log2 + 1;
}

inline void VerifyPrimeChain(const Json& manifest) {
  CRT_CONTEXT* crt = Get_crt_context();
  Require(crt != nullptr, "ANT CRT context is null");
  CRT_PRIMES* q_primes = Get_q(crt);
  CRT_PRIMES* p_primes = Get_p(crt);
  const Json& expected_q = manifest.at("data_q_bit_sizes");
  const Json& expected_p = manifest.at("special_p_bit_sizes");
  Require(q_primes != nullptr && p_primes != nullptr &&
              Get_primes_cnt(q_primes) == expected_q.size() &&
              Get_primes_cnt(p_primes) == expected_p.size(),
          "ANT Q/P counts disagree with the compiler context");
  for (std::size_t index = 0; index < expected_q.size(); ++index) {
    Require(AntPrimeSize(static_cast<std::uint64_t>(
                Get_modulus_val(Get_prime_at(q_primes, index)))) ==
                expected_q.at(index).get<std::uint32_t>(),
            "ANT data-Q bit size disagrees with the compiler context");
  }
  for (std::size_t index = 0; index < expected_p.size(); ++index) {
    Require(AntPrimeSize(static_cast<std::uint64_t>(
                Get_modulus_val(Get_prime_at(p_primes, index)))) ==
                expected_p.at(index).get<std::uint32_t>(),
            "ANT special-P bit size disagrees with the compiler context");
  }
}

inline void VerifyRuntimeContext(const QualificationInputs& inputs) {
  CKKS_PARAMS* parameters = Get_context_params();
  Require(parameters != nullptr, "ANT runtime context parameters are null");
  const Json& rotations = inputs.resources.at("rotation_steps");
  Require(parameters->_provider == LIB_ANT &&
              parameters->_poly_degree ==
                  inputs.context.at("polynomial_degree").get<std::uint32_t>() &&
              parameters->_sec_level ==
                  inputs.context.at("security_level").get<std::size_t>() &&
              parameters->_mul_depth + 1U ==
                  inputs.context.at("data_q_bit_sizes").size() &&
              parameters->_input_level == inputs.input_level &&
              parameters->_first_mod_size ==
                  inputs.context.at("first_modulus_bits").get<std::size_t>() &&
              parameters->_scaling_mod_size ==
                  inputs.context.at("scaling_modulus_bits").get<std::size_t>() &&
              parameters->_num_q_parts ==
                  inputs.context.at("q_part_count").get<std::size_t>() &&
              parameters->_hamming_weight ==
                  inputs.context.at("hamming_weight").get<std::size_t>() &&
              parameters->_num_rot_idx == rotations.size(),
          "linked ANT runtime context differs from compiler authority");
  for (std::size_t index = 0; index < rotations.size(); ++index) {
    Require(parameters->_rot_idxs[index] ==
                rotations.at(index).get<std::int32_t>(),
            "linked ANT rotation resource differs from compiler authority");
  }
}

inline Json RuntimeContextAttestation(const QualificationInputs& inputs,
                                      const std::string& provider) {
  CKKS_PARAMS* parameters = Get_context_params();
  CRT_CONTEXT* crt = Get_crt_context();
  Require(parameters != nullptr && crt != nullptr,
          "cannot attest an incomplete ANT runtime context");
  CRT_PRIMES* q_primes = Get_q(crt);
  CRT_PRIMES* p_primes = Get_p(crt);
  Require(q_primes != nullptr && p_primes != nullptr,
          "cannot attest an incomplete ANT prime chain");
  Json q_sizes = Json::array();
  Json p_sizes = Json::array();
  for (std::size_t index = 0; index < Get_primes_cnt(q_primes); ++index) {
    q_sizes.push_back(AntPrimeSize(static_cast<std::uint64_t>(
        Get_modulus_val(Get_prime_at(q_primes, index)))));
  }
  for (std::size_t index = 0; index < Get_primes_cnt(p_primes); ++index) {
    p_sizes.push_back(AntPrimeSize(static_cast<std::uint64_t>(
        Get_modulus_val(Get_prime_at(p_primes, index)))));
  }
  return {
      {"provider", provider},
      {"polynomial_degree", parameters->_poly_degree},
      {"logical_slots", inputs.slots},
      {"data_q_count", Get_primes_cnt(q_primes)},
      {"special_p_count", Get_primes_cnt(p_primes)},
      {"data_q_requested_bit_sizes", q_sizes},
      {"special_p_requested_bit_sizes", p_sizes},
      {"input_level", parameters->_input_level},
      {"scaling_factor_bits", parameters->_scaling_mod_size},
      {"first_prime_bits", parameters->_first_mod_size},
      {"hamming_weight", parameters->_hamming_weight},
      {"security_level", parameters->_sec_level},
      {"q_part_count", parameters->_num_q_parts},
      {"transform_budgets",
       {{"encode", inputs.encode_budget}, {"decode", inputs.decode_budget}}},
      {"phantom_context_manifest_sha256", inputs.context_sha256},
      {"physical_prime_identity_authority",
       "independent-non-authoritative-for-gpu"},
      {"schedule_authority", "independent-non-authoritative-for-gpu"},
  };
}

}  // namespace generated_bootstrap_ant

#endif  // ACE_GENERATED_BOOTSTRAP_ANT_COMMON_H
