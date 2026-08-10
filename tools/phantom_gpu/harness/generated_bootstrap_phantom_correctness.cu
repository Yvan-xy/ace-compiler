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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern CIPHERTEXT bootstrap_full(CIPHERTEXT, CIPHERTEXT);

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME *Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME *Get_decode_scheme(int) { return nullptr; }
}

namespace {
using Complex = std::complex<double>;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kMagic = {'A', 'C', 'E', 'B',
                                                'S', 'C', '0', '1'};
constexpr char kSchema[] = "ace.phantom.bootstrap-generated-phantom/1.0.0";
constexpr char kFormat[] = "ace.bootstrap-correctness.complex_float64le/1.0.0";
constexpr double kRawScaleCoordinateTolerance = 1.0e-4;

[[noreturn]] void Fail(const std::string &value) {
  throw std::runtime_error(value);
}
void Require(bool condition, const std::string &value) {
  if (!condition)
    Fail(value);
}

std::uint32_t RotateRight(std::uint32_t value, std::uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

std::string Sha256(const std::uint8_t *bytes, std::size_t length) {
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
  std::array<std::uint32_t, 8> state = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                        0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                        0x1f83d9abU, 0x5be0cd19U};
  const std::size_t padded = ((length + 9U + 63U) / 64U) * 64U;
  std::vector<std::uint8_t> message(padded, 0);
  if (length)
    std::copy(bytes, bytes + length, message.begin());
  message[length] = 0x80U;
  const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8U;
  for (std::size_t index = 0; index < 8; ++index)
    message[padded - 1U - index] =
        static_cast<std::uint8_t>(bit_length >> (8U * index));
  for (std::size_t offset = 0; offset < padded; offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t position = offset + 4U * index;
      words[index] = (std::uint32_t(message[position]) << 24U) |
                     (std::uint32_t(message[position + 1]) << 16U) |
                     (std::uint32_t(message[position + 2]) << 8U) |
                     message[position + 3];
    }
    for (std::size_t index = 16; index < 64; ++index) {
      const auto left = words[index - 15], right = words[index - 2];
      words[index] =
          words[index - 16] +
          (RotateRight(left, 7) ^ RotateRight(left, 18) ^ (left >> 3)) +
          words[index - 7] +
          (RotateRight(right, 17) ^ RotateRight(right, 19) ^ (right >> 10));
    }
    auto a = state[0], b = state[1], c = state[2], d = state[3], e = state[4],
         f = state[5], g = state[6], h = state[7];
    for (std::size_t index = 0; index < 64; ++index) {
      const auto t1 =
          h + (RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25)) +
          ((e & f) ^ (~e & g)) + constants[index] + words[index];
      const auto t2 =
          (RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22)) +
          ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    const std::array<std::uint32_t, 8> work = {a, b, c, d, e, f, g, h};
    for (std::size_t index = 0; index < 8; ++index)
      state[index] += work[index];
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (auto value : state)
    output << std::setw(8) << value;
  return output.str();
}

std::vector<std::uint8_t> ReadBytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Require(bool(input), "cannot open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "cannot size " + path);
  input.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty())
    input.read(reinterpret_cast<char *>(bytes.data()), size);
  Require(bool(input), "cannot read " + path);
  return bytes;
}
std::string Sha256File(const std::string &path) {
  const auto bytes = ReadBytes(path);
  return Sha256(bytes.data(), bytes.size());
}
Json LoadJson(const std::string &path) {
  std::ifstream input(path);
  Require(bool(input), "cannot open JSON " + path);
  Json value;
  input >> value;
  Require(bool(input) && value.is_object(), "cannot parse JSON " + path);
  return value;
}
void RequireFresh(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Require(!input.good(), "refusing to replace output " + path);
}
void WriteBytes(const std::string &path,
                const std::vector<std::uint8_t> &bytes) {
  RequireFresh(path);
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create " + path);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.flush();
  Require(bool(output), "cannot write " + path);
}
void WriteJson(const std::string &path, const Json &value) {
  RequireFresh(path);
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create " + path);
  output << value.dump(2) << '\n';
  output.flush();
  Require(bool(output), "cannot write " + path);
}

void CheckGpu(const std::string &expected) {
  Require(expected.find('/') == std::string::npos &&
              expected.find('\\') == std::string::npos,
          "expected GPU must be a basename");
  int count = 0;
  Require(cudaGetDeviceCount(&count) == cudaSuccess && count == 1,
          "expected exactly one CUDA device");
  cudaDeviceProp properties{};
  Require(cudaGetDeviceProperties(&properties, 0) == cudaSuccess,
          "cannot inspect CUDA device");
  Require(expected == properties.name, "CUDA device name mismatch");
}

class SplitMix64 {
public:
  explicit SplitMix64(std::uint64_t value) : state(value) {}
  double Symmetric(double bound) {
    return (2.0 * double(Next() >> 11U) * 0x1.0p-53 - 1.0) * bound;
  }

private:
  std::uint64_t state;
  std::uint64_t Next() {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }
};

struct Inputs {
  Json fixture, invocation, context, semantics, attestation, bindings;
  std::string fixture_sha, context_sha;
  std::size_t slots;
  int input_level;
  double threshold, repeat_threshold;
};

Inputs Authenticate(char **argv) {
  Inputs input{LoadJson(argv[1]),
               LoadJson(argv[4]),
               LoadJson(argv[7]),
               LoadJson(argv[10]),
               LoadJson(argv[12]),
               {},
               Sha256File(argv[1]),
               Sha256File(argv[7]),
               0,
               0,
               0,
               0};
  Require(input.fixture.at("schema_version") ==
              "ace.phantom.bootstrap-correctness-fixture/2.0.0",
          "fixture schema mismatch");
  Require(input.invocation.at("schema_version") ==
              "ace.phantom.generated-bootstrap.compiler-invocation/1.0.0",
          "invocation schema mismatch");
  Require(input.semantics.at("schema_version") ==
              "ace.phantom.generated-bootstrap.semantics/2.0.0",
          "semantics schema mismatch");
  Require(input.attestation.at("schema_version") ==
                  "ace.phantom.bootstrap-post-operation-semantics/1.0.0" &&
              input.attestation.at("status") == "attested",
          "post-operation attestation mismatch");
  input.bindings = input.fixture.at("bindings");
  const std::array<std::pair<const char *, int>, 11> artifacts = {
      {{"ace_source_manifest_sha256", 2},
       {"phantom_source_manifest_sha256", 3},
       {"compiler_invocation_sha256", 4},
       {"raw_air_sha256", 5},
       {"post_ckks_air_sha256", 6},
       {"context_manifest_sha256", 7},
       {"resource_manifest_sha256", 8},
       {"constant_manifest_sha256", 9},
       {"bootstrap_semantics_sha256", 10},
       {"post_operations_air_sha256", 11},
       {"post_operation_attestation_sha256", 12}}};
  Require(input.bindings.size() == artifacts.size(),
          "fixture binding count mismatch");
  for (const auto &item : artifacts)
    Require(input.bindings.at(item.first) == Sha256File(argv[item.second]),
            std::string("artifact hash mismatch: ") + item.first);
  Require(
      input.semantics.at("bindings").at("post_operations_attestation_sha256") ==
          Sha256File(argv[12]),
      "semantics attestation hash mismatch");
  Require(input.attestation.at("bindings").at("post_operations_air_sha256") ==
              Sha256File(argv[11]),
          "attested AIR hash mismatch");
  Require(input.fixture.at("supported_identity_domain").at("attested_domain") ==
              input.semantics.at("supported_identity_domain"),
          "identity domain mismatch");
  const auto &domain = input.semantics.at("supported_identity_domain");
  const auto &domain_evidence = domain.at("evidence");
  const auto &identity_attestation =
      input.semantics.at("identity_domain_attestation");
  const std::string canonical_identity_attestation =
      identity_attestation.dump();
  const std::string identity_attestation_sha256 = Sha256(
      reinterpret_cast<const std::uint8_t *>(
          canonical_identity_attestation.data()),
      canonical_identity_attestation.size());
  const double domain_lower = domain.at("lower_exclusive").get<double>();
  const double domain_upper = domain.at("upper_exclusive").get<double>();
  Require(domain.at("kind") ==
                  "centered-evalmod-complex-error-bounded" &&
              domain.at("components") == Json::array({"real", "imaginary"}) &&
              std::isfinite(domain_lower) && std::isfinite(domain_upper) &&
              domain_lower < 0.0 && domain_upper > 0.0 &&
              domain_lower == -domain_upper &&
              domain_evidence.at("attestation_schema_version") ==
                  "ace.phantom.bootstrap-clear-evalmod-domain/1.0.0" &&
              identity_attestation.at("schema_version") ==
                  "ace.phantom.bootstrap-clear-evalmod-domain/1.0.0" &&
              identity_attestation.at("status") == "attested" &&
              identity_attestation.at("scope") ==
                  Json{{"purpose", "domain-attestation-only"},
                       {"provider_value_oracle", false},
                       {"may_supply_expected_case_values", false}} &&
              domain_evidence.at("attestation_sha256").is_string() &&
              domain_evidence.at("attestation_sha256") ==
                  identity_attestation_sha256,
          "identity-domain attestation is invalid");
  const auto &semantic_bindings = input.semantics.at("bindings");
  const auto &domain_bindings =
      identity_attestation.at("artifact_bindings");
  const std::array<const char *, 8> required_domain_bindings = {
      "compiler_invocation_sha256", "constant_manifest_sha256",
      "context_manifest_sha256", "generated_dsl_ant_source_sha256",
      "phantom_source_sha256", "post_ckks_air_sha256", "raw_air_sha256",
      "resource_manifest_sha256"};
  Require(domain_bindings.size() == required_domain_bindings.size(),
          "identity-domain artifact binding count changed");
  for (const char *name : required_domain_bindings)
    Require(domain_bindings.at(name) == semantic_bindings.at(name),
            std::string("identity-domain artifact binding mismatch: ") + name);
  const auto &domain_errors = identity_attestation.at("error_contract");
  Require(domain_errors.at("target") ==
                  "original-clear-complex-identity" &&
              domain_errors.at("aggregate_norm") == "complex-absolute-l2" &&
              domain_errors.at("provider_clear_maximum_absolute") == 1e-2 &&
              domain_errors.at("clear_map_budget_fraction") == 0.5 &&
              domain_errors.at("maximum_complex_clear_map_error") == 0.005 &&
              domain_errors.at("reserved_provider_numerical_error") == 0.005 &&
              domain_evidence.at("provider_clear_maximum_absolute") == 1e-2 &&
              domain_evidence.at("maximum_complex_clear_map_error") == 0.005 &&
              domain_evidence.at("reserved_provider_numerical_error") == 0.005,
          "identity-domain error budget differs from frozen tolerances");
  const auto &domain_proof = identity_attestation.at("proof");
  Require(domain_proof.at("selected_radius") == domain_upper &&
              domain_proof.at("next_outward_radius").get<double>() >
                  domain_upper &&
              domain_proof.at("binary64_selection") ==
                  "conservative-certified-binary64-marker-used-as-exclusive-endpoint" &&
              domain_proof.at("arithmetic")
                      .at("transcendental_approximations_used") == false,
          "identity-domain boundary proof is invalid");
  const auto &domain_context =
      identity_attestation.at("normalization").at("compiler_context");
  Require(domain_context.at("polynomial_degree") ==
                  input.context.at("polynomial_degree") &&
              domain_context.at("logical_slots") ==
                  input.context.at("logical_slot_capacity"),
          "identity-domain normalization differs from compiler context");
  input.slots = input.context.at("logical_slot_capacity").get<std::size_t>();
  input.input_level = input.context.at("input_level").get<int>();
  Require(input.slots > 0 && input.input_level > 0,
          "invalid runtime context dimensions");
  input.threshold = input.fixture.at("tolerances")
                        .at("provider_clear_maximum_absolute")
                        .get<double>();
  input.repeat_threshold = input.fixture.at("tolerances")
                               .at("repeat_maximum_absolute")
                               .get<double>();
  Require(input.threshold == 1e-2 && input.repeat_threshold > 0,
          "invalid frozen tolerances");
  return input;
}

std::vector<std::pair<std::string, std::vector<Complex>>>
Materialize(const Inputs &input) {
  const auto &domain = input.semantics.at("supported_identity_domain");
  const double lower = domain.at("lower_exclusive"),
               upper = domain.at("upper_exclusive");
  const std::uint64_t seed =
      std::stoull(input.fixture.at("seed").at("value").get<std::string>());
  std::vector<std::pair<std::string, std::vector<Complex>>> result;
  for (const auto &recipe : input.fixture.at("case_recipes")) {
    const std::string id = recipe.at("id"), kind = recipe.at("kind");
    std::vector<Complex> values(input.slots);
    if (kind == "bounded_random_complex") {
      SplitMix64 generator(
          seed ^
          std::stoull(recipe.at("seed_xor").get<std::string>(), nullptr, 16));
      const double bound = recipe.at("component_bound");
      for (auto &value : values) {
        const double real = generator.Symmetric(bound);
        const double imaginary = generator.Symmetric(bound);
        value = {real, imaginary};
      }
    } else if (kind == "zero")
      std::fill(values.begin(), values.end(), Complex{});
    else if (kind == "real_constant" || kind == "complex_constant") {
      const auto &pair = recipe.at("value");
      std::fill(values.begin(), values.end(), Complex(pair.at(0), pair.at(1)));
    } else if (kind == "alternating_real_imaginary_signs") {
      const double magnitude = recipe.at("magnitude");
      for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = {index % 2 ? -magnitude : magnitude,
                         index % 2 ? magnitude : -magnitude};
    } else if (kind == "positive_boundary_inside" ||
               kind == "negative_boundary_inside") {
      const double offset = recipe.at("offset_from_attested_boundary");
      std::fill(values.begin(), values.end(),
                Complex(kind[0] == 'p' ? upper - offset : lower + offset, 0));
    } else
      Fail("unknown fixture case kind");
    for (const auto &value : values)
      Require(std::isfinite(value.real()) && std::isfinite(value.imag()) &&
                  value.real() > lower && value.real() < upper &&
                  value.imag() > lower && value.imag() < upper,
              "fixture value outside supported domain");
    result.push_back({id, std::move(values)});
  }
  return result;
}

PLAIN Encode(const std::vector<Complex> &values, int level,
             double scale_degree = 1.0) {
  PLAIN plain = new PLAINTEXT();
  Register_plain_lifetime(plain);
  std::vector<DCMPLX> copy(values.begin(), values.end());
  Encode_dcmplx(plain, copy.data(), copy.size(), scale_degree, level);
  return plain;
}
void Encrypt(CIPHER result, const std::vector<Complex> &values, int level) {
  PLAIN plain = Encode(values, level);
  Phantom_encrypt_plain(result, plain);
  Free_plain(plain);
  delete plain;
}
std::vector<Complex> Decode(CIPHER value, std::size_t slots) {
  std::vector<Complex> result;
  Phantom_decrypt_dcmplx(value, result);
  Require(result.size() == slots, "decoded length is not full capacity");
  for (auto item : result)
    Require(std::isfinite(item.real()) && std::isfinite(item.imag()),
            "nonfinite decoded value");
  return result;
}
Json Metadata(CIPHER value) {
  return {{"ace_level", Level(value)},
          {"active_q_count", Active_q_count(value)},
          {"phantom_chain_index", Chain_index(value)},
          {"raw_scale", Raw_scale(value)},
          {"scale_degree", static_cast<std::int64_t>(Sc_degree(value))},
          {"logical_slots", Get_ciph_slots(value)},
          {"ciphertext_size", Get_ciph_size(value)},
          {"ntt_state", Is_ciph_ntt(value)}};
}

double HexDouble(const Json &value) {
  Require(value.is_string(), "raw scale contract is not hexadecimal");
  return std::stod(value.get<std::string>());
}
Json ExpectedBootstrapMetadata(const Inputs &input) {
  const auto &value = input.semantics.at("output_air_contract");
  const auto &contract = value.at("raw_scale_contract");
  Require(contract.at("kind") == "ace-log2-scale-coordinate",
          "unsupported raw scale contract");
  const auto scaling_bits =
      contract.at("scaling_modulus_bits").get<std::int64_t>();
  const auto scale_degree = value.at("scale_degree").get<std::int64_t>();
  const double tolerance =
      contract.at("maximum_absolute_coordinate_error").get<double>();
  const double nominal = HexDouble(contract.at("nominal_raw_scale"));
  Require(scaling_bits > 0 && scale_degree >= 0 &&
              contract.at("expected_scale_degree") == scale_degree &&
              tolerance == kRawScaleCoordinateTolerance &&
              nominal == std::ldexp(1.0, scaling_bits * scale_degree),
          "invalid raw scale contract");
  return {{"ace_level", value.at("ace_logical_level")},
          {"active_q_count", value.at("active_q_count")},
          {"phantom_chain_index", value.at("phantom_chain_index")},
          {"raw_scale", nominal},
          {"scale_degree", value.at("scale_degree")},
          {"logical_slots", value.at("logical_slots")},
          {"ciphertext_size", value.at("ciphertext_size")},
          {"ntt_state", value.at("ntt_state")}};
}
Json ExpectedBootstrapMetadata(const Inputs &input, double observed_raw_scale) {
  Json expected = ExpectedBootstrapMetadata(input);
  const auto &contract =
      input.semantics.at("output_air_contract").at("raw_scale_contract");
  const double coordinate =
      std::log2(observed_raw_scale) /
      contract.at("scaling_modulus_bits").get<double>();
  const double expected_degree =
      contract.at("expected_scale_degree").get<double>();
  const double tolerance =
      contract.at("maximum_absolute_coordinate_error").get<double>();
  Require(std::isfinite(observed_raw_scale) && observed_raw_scale > 0.0 &&
              std::isfinite(coordinate) &&
              std::abs(coordinate - expected_degree) <= tolerance,
          "bootstrap raw scale differs from AIR scale coordinate");
  expected["raw_scale"] = observed_raw_scale;
  return expected;
}
Json Transition(const Json &before, const Json &transition) {
  Json after = before;
  for (const char *key :
       {"ace_level", "active_q_count", "phantom_chain_index", "scale_degree"}) {
    const std::string delta =
        std::string(key == std::string("ace_level") ? "ace_logical_level"
                                                    : key) +
        "_delta";
    after[key] = before.at(key).get<std::int64_t>() +
                 transition.at(delta).get<std::int64_t>();
  }
  const auto multiplier = transition.at("raw_scale_multiplier");
  after["raw_scale"] =
      before.at("raw_scale").get<double>() *
      (multiplier.is_string() ? std::stod(multiplier.get<std::string>())
                              : multiplier.get<double>());
  for (const char *key : {"logical_slots", "ciphertext_size", "ntt_state"})
    Require(transition.at(key) == "preserved",
            "unsupported metadata transition");
  return after;
}
double Maximum(const std::vector<Complex> &actual,
               const std::vector<Complex> &expected) {
  Require(actual.size() == expected.size(), "comparison length mismatch");
  double maximum = 0;
  for (std::size_t index = 0; index < actual.size(); ++index)
    maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
  return maximum;
}
Json Metric(const std::vector<Complex> &actual,
            const std::vector<Complex> &expected, double threshold,
            const std::string &comparison) {
  Require(actual.size() == expected.size() && !actual.empty(),
          "metric vectors have unequal or zero length");
  double maximum = -1, sum = 0, squares = 0;
  std::size_t max_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(actual[index] - expected[index]);
    Require(std::isfinite(error), "metric contains a non-finite error");
    if (error > maximum) {
      maximum = error;
      max_index = index;
    }
    sum += error;
    squares += error * error;
  }
  if (maximum > threshold) {
    std::ostringstream message;
    message << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "generated Phantom result exceeds the frozen numerical threshold: "
            << "comparison=" << comparison
            << ", maximum_absolute_error=" << maximum
            << ", maximum_absolute_error_index=" << max_index
            << ", actual_real=" << actual[max_index].real()
            << ", actual_imaginary=" << actual[max_index].imag()
            << ", expected_real=" << expected[max_index].real()
            << ", expected_imaginary=" << expected[max_index].imag()
            << ", mean_absolute_error=" << sum / actual.size()
            << ", root_mean_square_error="
            << std::sqrt(squares / actual.size())
            << ", threshold=" << threshold;
    Fail(message.str());
  }
  return {{"status", "pass"},
          {"comparison_count", actual.size()},
          {"threshold", threshold},
          {"maximum_absolute_error", maximum},
          {"maximum_absolute_error_index", max_index},
          {"mean_absolute_error", sum / actual.size()},
          {"root_mean_square_error", std::sqrt(squares / actual.size())},
          {"estimated_precision_bits",
           maximum == 0 ? Json(nullptr) : Json(-std::log2(maximum))},
          {"estimated_precision_is_infinite", maximum == 0}};
}

void AppendU64(std::vector<std::uint8_t> &out, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8)
    out.push_back(value >> shift);
}
void AppendU32(std::vector<std::uint8_t> &out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8)
    out.push_back(value >> shift);
}
void AppendU16(std::vector<std::uint8_t> &out, std::uint16_t value) {
  for (int shift = 0; shift < 16; shift += 8)
    out.push_back(value >> shift);
}
void AppendDigest(std::vector<std::uint8_t> &out, const std::string &value) {
  Require(value.size() == 64, "bad digest");
  for (std::size_t index = 0; index < 64; index += 2)
    out.push_back(std::stoul(value.substr(index, 2), nullptr, 16));
}
void AppendDouble(std::vector<std::uint8_t> &out, double value) {
  std::uint64_t bits;
  std::memcpy(&bits, &value, 8);
  AppendU64(out, bits);
}
std::string ValuesSha256(const std::vector<Complex> &values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * 16U);
  for (const auto &value : values) {
    AppendDouble(bytes, value.real());
    AppendDouble(bytes, value.imag());
  }
  return Sha256(bytes.data(), bytes.size());
}
std::string CaseManifest(const Json &records) {
  Json value = Json::array();
  for (const auto &record : records)
    value.push_back({{"case_id", record.at("case_id")},
                     {"oracle", record.at("oracle")},
                     {"value_count", record.at("value_count")}});
  const std::string bytes = value.dump() + "\n";
  return Sha256(reinterpret_cast<const std::uint8_t *>(bytes.data()),
                bytes.size());
}

Json Run(const Inputs &input,
         const std::vector<std::pair<std::string, std::vector<Complex>>> &cases,
  std::vector<std::uint8_t> &payload) {
  Json records = Json::array(), order = Json::array();
  Json expected_metadata;
  const auto &operation = input.fixture.at("post_operations");
  const auto &operation_inputs = operation.at("inputs");
  const Complex multiplier(
      operation_inputs.at("multiply_constant").at("real"),
      operation_inputs.at("multiply_constant").at("imaginary"));
  const int step = operation_inputs.at("rotation_step");
  Require(step != 0 && step % static_cast<int>(input.slots) != 0,
          "rotation step is zero modulo slots");
  for (const auto &item : cases) {
    order.push_back(item.first);
    CIPHERTEXT encrypted;
    Register_ciph_lifetime(&encrypted);
    Encrypt(&encrypted, item.second, input.input_level);
    CIPHERTEXT encrypted_zero;
    Register_ciph_lifetime(&encrypted_zero);
    Encrypt(&encrypted_zero, std::vector<Complex>(input.slots),
            input.input_level);
    std::vector<std::vector<Complex>> calls;
    Json call_metadata = Json::array();
    Json call_metrics = Json::array();
    CIPHERTEXT primary;
    Register_ciph_lifetime(&primary);
    for (int call = 0; call < 3; ++call) {
      CIPHERTEXT owned_input = encrypted;
      Register_ciph_lifetime(&owned_input);
      CIPHERTEXT owned_zero = encrypted_zero;
      Register_ciph_lifetime(&owned_zero);
      CIPHERTEXT observed = bootstrap_full(owned_input, owned_zero);
      Register_ciph_lifetime(&observed);
      auto decoded = Decode(&observed, input.slots);
      Json metadata = Metadata(&observed);
      Json observed_contract = ExpectedBootstrapMetadata(
          input, metadata.at("raw_scale").get<double>());
      Require(metadata == observed_contract,
              "bootstrap metadata differs from AIR contract: observed=" +
                  metadata.dump() + " expected=" + observed_contract.dump());
      if (expected_metadata.is_null())
        expected_metadata = metadata;
      else
        Require(metadata == expected_metadata,
                "bootstrap metadata differs across calls or cases");
      call_metrics.push_back(Metric(
          decoded, item.second, input.threshold,
          "generated-phantom/" + item.first + "/bootstrap-call-" +
              std::to_string(call)));
      calls.push_back(decoded);
      call_metadata.push_back(metadata);
      if (call == 0)
        Copy_ciph(&primary, &observed);
      Free_ciph(&observed);
      Free_ciph(&owned_input);
      Free_ciph(&owned_zero);
    }
    double repeat_maximum = 0;
    for (std::size_t call = 1; call < calls.size(); ++call)
      repeat_maximum = std::max(repeat_maximum, Maximum(calls[0], calls[call]));
    Require(repeat_maximum <= input.repeat_threshold,
            "repeat tolerance exceeded");
    CIPHERTEXT multiply_source = primary;
    Register_ciph_lifetime(&multiply_source);
    CIPHERTEXT multiply_result;
    Register_ciph_lifetime(&multiply_result);
    const auto multiply_transition =
        operation.at("ciphertext_plaintext_multiply").at("transition");
    const double plain_scale_degree =
        operation_inputs.at("multiply_constant").at("plaintext_scale_degree");
    Require(plain_scale_degree == 0.0,
            "post-operation plaintext scale degree must be explicitly zero");
    PLAIN plain = Encode(std::vector<Complex>(input.slots, multiplier),
                         Level(&multiply_source), plain_scale_degree);
    Mul_plain(&multiply_result, &multiply_source, plain);
    std::vector<Complex> multiply_expected = item.second;
    for (auto &value : multiply_expected)
      value *= multiplier;
    auto multiply_values = Decode(&multiply_result, input.slots);
    const double post_threshold =
        input.threshold * std::max(1.0, std::abs(multiplier));
    Json multiply_metric =
        Metric(multiply_values, multiply_expected, post_threshold,
               "generated-phantom/" + item.first +
                   "/ciphertext-plaintext-multiply");
    Json multiply_metadata = Metadata(&multiply_result);
    Require(multiply_metadata ==
                Transition(expected_metadata, multiply_transition),
            "multiply metadata transition mismatch");
    CIPHERTEXT rotate_source = primary;
    Register_ciph_lifetime(&rotate_source);
    CIPHERTEXT rotate_result;
    Register_ciph_lifetime(&rotate_result);
    Rotate_ciph(&rotate_result, &rotate_source, step);
    std::vector<Complex> rotate_expected(input.slots);
    for (std::size_t index = 0; index < input.slots; ++index) {
      const auto rotated = (static_cast<std::int64_t>(index) + step +
                            static_cast<std::int64_t>(input.slots)) %
                           static_cast<std::int64_t>(input.slots);
      rotate_expected[index] = item.second[static_cast<std::size_t>(rotated)];
    }
    auto rotate_values = Decode(&rotate_result, input.slots);
    Json rotate_metric =
        Metric(rotate_values, rotate_expected, input.threshold,
               "generated-phantom/" + item.first + "/rotation");
    Json rotate_metadata = Metadata(&rotate_result);
    Require(rotate_metadata ==
                Transition(expected_metadata,
                           operation.at("rotation").at("transition")),
            "rotation metadata transition mismatch");
    const std::size_t offset = payload.size();
    for (const auto &value : calls[0]) {
      AppendDouble(payload, value.real());
      AppendDouble(payload, value.imag());
    }
    Json case_record = {
        {"case_id", item.first},
        {"oracle", "generated-phantom"},
        {"offset_bytes", offset},
        {"value_count", input.slots},
        {"metadata",
         {{"bootstrap", expected_metadata},
          {"metrics_vs_clear",
           Metric(calls[0], item.second, input.threshold,
                  "generated-phantom/" + item.first + "/primary-output")},
          {"repeatability",
           {{"calls", 3},
            {"independently_owned_clones", true},
            {"maximum_absolute_difference", repeat_maximum},
            {"threshold", input.repeat_threshold},
            {"call_metrics", call_metrics},
            {"call_metadata", call_metadata}}},
          {"post_operations",
           {{"ciphertext_plaintext_multiply",
             {{"metric", multiply_metric},
              {"metadata", multiply_metadata},
              {"values_sha256", ValuesSha256(multiply_values)}}},
            {"rotation",
             {{"metric", rotate_metric},
              {"metadata", rotate_metadata},
              {"values_sha256", ValuesSha256(rotate_values)},
              {"step", step}}}}},
          {"ownership", nullptr}}}};
    Free_plain(plain);
    delete plain;
    for (CIPHER value : {&multiply_source, &multiply_result, &rotate_source,
                         &rotate_result, &primary, &encrypted, &encrypted_zero})
      Free_ciph(value);
    case_record["metadata"]["ownership"] = {
        {"input_clones_released", 3},
        {"zero_argument_clones_released", 3},
        {"bootstrap_results_released", 3},
        {"post_operation_ciphertexts_released", 4},
        {"post_operation_plaintexts_released", 1},
        {"base_encrypted_arguments_released", 2},
        {"post_operation_sources_independent", true},
        {"all_owned_objects_released", true},
    };
    records.push_back(std::move(case_record));
  }
  const std::string manifest = CaseManifest(records);
  std::vector<std::uint8_t> binary(kMagic.begin(), kMagic.end());
  AppendU16(binary, 1);
  AppendU16(binary, 0);
  AppendU32(binary, 0);
  AppendU64(binary, records.size());
  AppendU64(binary, payload.size());
  AppendDigest(binary, input.fixture_sha);
  AppendDigest(binary, input.context_sha);
  AppendDigest(binary, manifest);
  Require(binary.size() == 128, "binary header size mismatch");
  binary.insert(binary.end(), payload.begin(), payload.end());
  payload = std::move(binary);
  return {{"schema_version", kSchema},
          {"provider", "generated-phantom"},
          {"fixture_sha256", input.fixture_sha},
          {"bindings", input.bindings},
          {"case_order", order},
          {"logical_slots", input.slots},
          {"binary",
           {{"format", kFormat},
            {"sha256", Sha256(payload.data(), payload.size())},
            {"size_bytes", payload.size()},
            {"header_size", 128},
            {"record_count", records.size()},
            {"payload_bytes", payload.size() - 128},
            {"case_manifest_sha256", manifest}}},
          {"records", records},
          {"execution",
           {{"status", "pass"},
            {"skip_count", 0},
            {"timeout_count", 0},
            {"fallback_count", 0},
            {"nonfinite_count", 0},
            {"generated_bootstrap_invocation_count", cases.size() * 3},
            {"executable_sha256", "filled-after-run"},
            {"teardown_completed", true}}}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    Require(
        argc == 16,
        "usage: generated_bootstrap_phantom_correctness <fixture> "
        "<ace-source-manifest> <phantom-source-manifest> <compiler-invocation> "
        "<raw-air> <post-ckks-air> <context-manifest> <resource-manifest> "
        "<constant-manifest> <bootstrap-semantics> <post-operations-air> "
        "<post-operation-attestation> <expected-gpu-basename> <record-output> "
        "<values-output>");
    RequireFresh(argv[14]);
    RequireFresh(argv[15]);
    CheckGpu(argv[13]);
    Inputs input = Authenticate(argv);
    Prepare_context();
    auto cases = Materialize(input);
    std::vector<std::uint8_t> values;
    Json record = Run(input, cases, values);
    Finalize_context();
    record["execution"]["executable_sha256"] = Sha256File(argv[0]);
    WriteBytes(argv[15], values);
    WriteJson(argv[14], record);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "generated bootstrap Phantom correctness failed: "
              << error.what() << '\n';
    return 1;
  }
}
