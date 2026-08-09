// Focused Phantom conformance runner for the retained CKKS provider surface.
// Context and resource authority always comes from compiler-generated symbols.

#include "common/rt_api.h"
#include "rt_phantom/rt_phantom.h"
#include "retained_ckks_generated_interface.h"

#include "context.cuh"
#include "evaluate.cuh"
#include "util/encryptionparams.h"
#include "util/modulus.h"

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
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ACE_COMMIT_ID
#error "ACE_COMMIT_ID must be the tested ACE commit"
#endif
#ifndef PHANTOM_COMMIT_ID
#error "PHANTOM_COMMIT_ID must be the tested Phantom commit"
#endif
#ifndef ACE_REJECTION_MANIFEST_ID
#define ACE_REJECTION_MANIFEST_ID "production"
#endif

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME *Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME *Get_decode_scheme(int) { return nullptr; }
}

namespace {

using Complex = std::complex<double>;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kDecodedMagic = {'A', 'C', 'E', 'R',
                                                       'C', 'K', '0', '1'};
constexpr std::array<std::uint8_t, 8> kExactMagic = {'A', 'C', 'E', 'R',
                                                     'N', 'S', '0', '1'};
constexpr std::array<std::uint8_t, 8> kExactSourceMagic = {
    'A', 'C', 'E', 'S', 'R', 'C', '0', '1'};
constexpr char kProviderSchema[] =
    "ace.phantom.retained_ckks.provider-result/3.0.0";
constexpr char kExactObservedSchema[] =
    "ace.phantom.retained_ckks.exact-observed/2.0.0";

[[noreturn]] void Fail(const std::string &diagnostic,
                       const std::string &detail) {
  throw std::runtime_error("ACE_RETAINED_CONFORMANCE[" + diagnostic +
                           "]: " + detail);
}

void Require(bool condition, const std::string &diagnostic,
             const std::string &detail) {
  if (!condition)
    Fail(diagnostic, detail);
}

void RequireCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    Fail("CUDA", std::string(operation) + ": " + cudaGetErrorString(status));
  }
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
  std::copy(bytes, bytes + length, message.begin());
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
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint32_t value : state)
    output << std::setw(8) << value;
  return output.str();
}

std::string Sha256(const std::vector<std::uint8_t> &bytes) {
  return Sha256(bytes.data(), bytes.size());
}

std::vector<std::uint8_t> ReadBytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    Fail("INPUT_OPEN", "cannot open " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff length = input.tellg();
  Require(length >= 0, "INPUT_SIZE", "cannot size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char *>(bytes.data()), length);
  }
  Require(static_cast<bool>(input), "INPUT_READ", "cannot read " + path);
  return bytes;
}

Json ParseJson(const std::vector<std::uint8_t> &bytes,
               const std::string &path) {
  std::vector<std::set<std::string>> object_keys;
  std::string duplicate_key;
  Json::parser_callback_t callback =
      [&](int, Json::parse_event_t event, Json &parsed) {
        if (event == Json::parse_event_t::object_start) {
          object_keys.emplace_back();
        } else if (event == Json::parse_event_t::key) {
          Require(!object_keys.empty(), "JSON_PARSE",
                  "JSON key appeared outside an object in " + path);
          const std::string key = parsed.get<std::string>();
          if (!object_keys.back().insert(key).second && duplicate_key.empty()) {
            duplicate_key = key;
          }
        } else if (event == Json::parse_event_t::object_end) {
          Require(!object_keys.empty(), "JSON_PARSE",
                  "JSON object nesting is invalid in " + path);
          object_keys.pop_back();
        }
        return true;
      };
  try {
    Json value = Json::parse(bytes.begin(), bytes.end(), callback, true, false);
    Require(!value.is_discarded(), "JSON_PARSE", "cannot parse " + path);
    Require(duplicate_key.empty(), "JSON_DUPLICATE_KEY",
            "duplicate key '" + duplicate_key + "' in " + path);
    Require(object_keys.empty(), "JSON_PARSE",
            "unterminated JSON object in " + path);
    return value;
  } catch (const Json::exception &error) {
    Fail("JSON_PARSE", "cannot parse " + path + ": " + error.what());
  }
}

Json LoadJson(const std::string &path) {
  return ParseJson(ReadBytes(path), path);
}

void WriteBytes(const std::string &path,
                const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    Fail("OUTPUT_OPEN", "cannot create " + path);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.flush();
  Require(static_cast<bool>(output), "OUTPUT_WRITE", "cannot write " + path);
}

void WriteJson(const std::string &path, const Json &value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    Fail("OUTPUT_OPEN", "cannot create " + path);
  output << value.dump(2) << '\n';
  output.flush();
  Require(static_cast<bool>(output), "OUTPUT_WRITE", "cannot write " + path);
}

void AppendU64Le(std::vector<std::uint8_t> &output, std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    output.push_back(static_cast<std::uint8_t>(value >> (byte * 8U)));
  }
}

std::uint64_t ReadU64Le(const std::uint8_t *input) {
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < 8; ++byte) {
    value |= static_cast<std::uint64_t>(input[byte]) << (byte * 8U);
  }
  return value;
}

void AppendDoubleLe(std::vector<std::uint8_t> &output, double value) {
  Require(std::isfinite(value), "NONFINITE", "decoded value is not finite");
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU64Le(output, bits);
}

double ReadDoubleLe(const std::uint8_t *input) {
  const std::uint64_t bits = ReadU64Le(input);
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  Require(std::isfinite(value), "NONFINITE", "input value is not finite");
  return value;
}

Json AppendBlob(std::vector<std::uint8_t> &output,
                const std::vector<std::uint8_t> &blob,
                std::size_t element_count) {
  const std::size_t offset = output.size();
  output.insert(output.end(), blob.begin(), blob.end());
  return {{"offset_bytes", offset},
          {"count", element_count},
          {"byte_length", blob.size()},
          {"sha256", Sha256(blob)}};
}

std::vector<std::uint8_t> PackComplex(const std::vector<Complex> &values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * 16U);
  for (const Complex &value : values) {
    AppendDoubleLe(bytes, value.real());
    AppendDoubleLe(bytes, value.imag());
  }
  return bytes;
}

std::vector<std::uint8_t> PackU64(const std::vector<std::uint64_t> &values) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(values.size() * 8U);
  for (std::uint64_t value : values)
    AppendU64Le(bytes, value);
  return bytes;
}

void ValidateBlob(const std::vector<std::uint8_t> &file, const Json &descriptor,
                  const std::string &diagnostic) {
  const std::size_t offset = descriptor.at("offset_bytes").get<std::size_t>();
  const std::size_t length = descriptor.at("byte_length").get<std::size_t>();
  Require(offset <= file.size() && length <= file.size() - offset, diagnostic,
          "blob is outside its binary artifact");
  std::vector<std::uint8_t> bytes(file.begin() + offset,
                                  file.begin() + offset + length);
  Require(descriptor.at("sha256").get<std::string>() == Sha256(bytes),
          diagnostic, "blob SHA-256 mismatch");
}

void ValidateBinary(const std::vector<std::uint8_t> &file,
                    const Json &descriptor,
                    const std::array<std::uint8_t, 8> &magic,
                    const std::string &diagnostic) {
  Require(file.size() >= magic.size() &&
              std::equal(magic.begin(), magic.end(), file.begin()),
          diagnostic, "binary magic mismatch");
  Require(descriptor.at("size_bytes").get<std::size_t>() == file.size(),
          diagnostic, "binary size mismatch");
  Require(descriptor.at("sha256").get<std::string>() == Sha256(file),
          diagnostic, "binary SHA-256 mismatch");
}

std::vector<Complex> ReadComplexBlob(const std::vector<std::uint8_t> &file,
                                     const Json &descriptor,
                                     const std::string &diagnostic) {
  ValidateBlob(file, descriptor, diagnostic);
  const std::size_t count = descriptor.at("count").get<std::size_t>();
  Require(descriptor.at("byte_length").get<std::size_t>() == count * 16U,
          diagnostic, "complex blob length mismatch");
  const std::size_t offset = descriptor.at("offset_bytes").get<std::size_t>();
  std::vector<Complex> values;
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    values.emplace_back(ReadDoubleLe(file.data() + offset + index * 16U),
                        ReadDoubleLe(file.data() + offset + index * 16U + 8U));
  }
  return values;
}

std::vector<std::int64_t> ReadI64Blob(const std::vector<std::uint8_t> &file,
                                      const Json &descriptor,
                                      const std::string &diagnostic) {
  ValidateBlob(file, descriptor, diagnostic);
  const std::size_t count = descriptor.at("count").get<std::size_t>();
  Require(descriptor.at("byte_length").get<std::size_t>() == count * 8U,
          diagnostic, "signed coefficient blob length mismatch");
  const std::size_t offset = descriptor.at("offset_bytes").get<std::size_t>();
  std::vector<std::int64_t> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint64_t bits = ReadU64Le(file.data() + offset + index * 8U);
    std::memcpy(&values[index], &bits, sizeof(bits));
  }
  return values;
}

void CheckGpu(const std::string &expected_name) {
  int count = 0;
  RequireCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  Require(count == 1, "GPU_COUNT", "expected exactly one CUDA device");
  cudaDeviceProp properties{};
  RequireCuda(cudaGetDeviceProperties(&properties, 0),
              "cudaGetDeviceProperties");
  Require(expected_name == properties.name, "GPU_NAME",
          "expected '" + expected_name + "', observed '" + properties.name +
              "'");
}

const PHANTOM_CONTEXT_MANIFEST &ContextManifest() {
  const auto *manifest = Get_phantom_context_manifest();
  Require(manifest != nullptr, "CONTEXT_MANIFEST", "manifest is null");
  Require(manifest->_schema_version == 1 &&
              manifest->_packing == PHANTOM_PACKING_FULL,
          "CONTEXT_MANIFEST", "unsupported manifest schema or packing");
  Require(manifest->_poly_degree >= 2 &&
              (manifest->_poly_degree & (manifest->_poly_degree - 1U)) == 0 &&
              manifest->_logical_slots == manifest->_poly_degree / 2U,
          "CONTEXT_MANIFEST", "invalid compiler-derived dimensions");
  Require(manifest->_data_q_count > 0 &&
              manifest->_data_q_bit_sizes != nullptr &&
              manifest->_special_p_count > 0 &&
              manifest->_special_p_bit_sizes != nullptr,
          "CONTEXT_MANIFEST", "compiler-derived modulus arrays are empty");
  return *manifest;
}

std::uint64_t JsonUnsigned(const Json &value, const std::string &field) {
  if (value.is_number_unsigned())
    return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    Require(signed_value >= 0, "CONTEXT_FILE_FIELD",
            field + " must be nonnegative");
    return static_cast<std::uint64_t>(signed_value);
  }
  Fail("CONTEXT_FILE_FIELD", field + " must be an unsigned integer");
}

void RequireContextKeys(const Json &value) {
  static const std::set<std::string> expected = {
      "data_q_bit_sizes",       "first_modulus_bits",
      "hamming_weight",         "input_level",
      "logical_slot_capacity",  "packing",
      "polynomial_degree",      "q_part_count",
      "resource_schema_version", "scaling_modulus_bits",
      "schema_version",         "security_level",
      "special_p_bit_sizes"};
  Require(value.is_object(), "CONTEXT_FILE_SCHEMA",
          "context manifest JSON must be an object");
  std::set<std::string> observed;
  for (auto iterator = value.begin(); iterator != value.end(); ++iterator)
    observed.insert(iterator.key());
  Require(observed == expected, "CONTEXT_FILE_SCHEMA",
          "context manifest JSON fields differ from the compiler schema");
}

struct AuthenticatedContext {
  std::string _sha256;
};

AuthenticatedContext AuthenticateContext(const std::string &path) {
  const auto bytes = ReadBytes(path);
  const Json value = ParseJson(bytes, path);
  RequireContextKeys(value);
  const auto &linked = ContextManifest();
  auto require_equal = [&](const char *field, std::uint64_t expected) {
    const std::uint64_t observed = JsonUnsigned(value.at(field), field);
    Require(observed == expected, "CONTEXT_FILE_MISMATCH",
            std::string(field) + " observed " + std::to_string(observed) +
                ", expected " + std::to_string(expected));
  };
  require_equal("schema_version", linked._schema_version);
  Require(value.at("packing").is_string() &&
              value.at("packing").get<std::string>() == "full" &&
              linked._packing == PHANTOM_PACKING_FULL,
          "CONTEXT_FILE_MISMATCH",
          "packing must equal the linked full-packing manifest");
  require_equal("polynomial_degree", linked._poly_degree);
  require_equal("logical_slot_capacity", linked._logical_slots);
  require_equal("input_level", linked._input_level);
  require_equal("q_part_count", linked._q_part_count);
  require_equal("hamming_weight", linked._hamming_weight);
  require_equal("security_level", linked._security_level);
  require_equal("first_modulus_bits", linked._first_modulus_bits);
  require_equal("scaling_modulus_bits", linked._scaling_modulus_bits);
  require_equal("resource_schema_version", linked._resource_schema_version);

  auto require_ordered_bits = [&](const char *field, const std::uint32_t *bits,
                                  std::size_t count) {
    const Json &array = value.at(field);
    Require(array.is_array(), "CONTEXT_FILE_FIELD",
            std::string(field) + " must be an array");
    Require(array.size() == count, "CONTEXT_FILE_MISMATCH",
            std::string(field) + " count observed " +
                std::to_string(array.size()) + ", expected " +
                std::to_string(count));
    for (std::size_t index = 0; index < count; ++index) {
      const std::uint64_t observed =
          JsonUnsigned(array.at(index), std::string(field) + "[" +
                                            std::to_string(index) + "]");
      Require(observed == bits[index], "CONTEXT_FILE_MISMATCH",
              std::string(field) + "[" + std::to_string(index) +
                  "] observed " + std::to_string(observed) + ", expected " +
                  std::to_string(bits[index]));
    }
  };
  require_ordered_bits("data_q_bit_sizes", linked._data_q_bit_sizes,
                       linked._data_q_count);
  require_ordered_bits("special_p_bit_sizes", linked._special_p_bit_sizes,
                       linked._special_p_count);
  return {Sha256(bytes)};
}

void RequireContextBinding(const Json &fixture,
                           const AuthenticatedContext &context) {
  const Json &bindings = fixture.at("qualification_bindings");
  Require(bindings.at("compiler_context_manifest_sha256").is_string(),
          "CONTEXT_BINDING",
          "fixture context-manifest binding must be a SHA-256 string");
  const std::string expected =
      bindings.at("compiler_context_manifest_sha256").get<std::string>();
  Require(expected.size() == 64U &&
              std::all_of(expected.begin(), expected.end(), [](char value) {
                return (value >= '0' && value <= '9') ||
                       (value >= 'a' && value <= 'f');
              }),
          "CONTEXT_BINDING",
          "fixture context-manifest binding must be lowercase SHA-256");
  Require(expected == context._sha256, "CONTEXT_BINDING",
          "context manifest observed SHA-256 " + context._sha256 +
              ", expected " + expected);
}

const PHANTOM_RESOURCE_MANIFEST &ResourceManifest() {
  const auto *manifest = Get_phantom_resource_manifest();
  Require(manifest != nullptr, "RESOURCE_MANIFEST", "manifest is null");
  Require(manifest->_schema_version ==
              ContextManifest()._resource_schema_version,
          "RESOURCE_MANIFEST", "resource schema mismatch");
  return *manifest;
}

std::vector<Complex> Decode(CIPHER cipher) {
  std::vector<Complex> values;
  Phantom_decrypt_dcmplx(cipher, values);
  Require(values.size() == ContextManifest()._logical_slots, "DECODE_SIZE",
          "decoded slot count differs from the context manifest");
  for (const Complex &value : values) {
    Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
            "NONFINITE", "decoded result contains NaN or infinity");
  }
  return values;
}

Json Metadata(CIPHER cipher) {
  return {{"ace_level", Level(cipher)},
          {"active_q_count", Active_q_count(cipher)},
          {"scale_degree", static_cast<std::int64_t>(Sc_degree(cipher))},
          {"logical_slots", Get_ciph_slots(cipher)},
          {"ciphertext_size", Get_ciph_size(cipher)},
          {"ntt", Is_ciph_ntt(cipher)},
          {"chain_index", Chain_index(cipher)},
          {"raw_scale", Raw_scale(cipher)}};
}

struct RuntimeSnapshot {
  Json _metadata;
  std::vector<Complex> _values;
  std::string _residues_sha256;
  phantom::parms_id_type _parms_id;
  const std::uint64_t *_buffer;
};

std::string CipherResiduesSha256(CIPHER cipher) {
  Require(cipher != nullptr && cipher->data() != nullptr, "RESIDUE_SNAPSHOT",
          "ciphertext has no device residue buffer");
  const std::size_t component_count = cipher->size();
  const std::size_t tower_count = cipher->coeff_modulus_size();
  const std::size_t coefficient_count = cipher->poly_modulus_degree();
  Require(component_count != 0 && tower_count != 0 && coefficient_count != 0,
          "RESIDUE_SNAPSHOT", "ciphertext residue dimensions are empty");
  Require(component_count <=
                  std::numeric_limits<std::size_t>::max() / tower_count &&
              component_count * tower_count <=
                  std::numeric_limits<std::size_t>::max() / coefficient_count,
          "RESIDUE_SNAPSHOT", "ciphertext residue count overflows size_t");
  const std::size_t count = component_count * tower_count * coefficient_count;
  std::vector<std::uint64_t> residues(count);
  RequireCuda(cudaDeviceSynchronize(),
              "residue snapshot cudaDeviceSynchronize");
  RequireCuda(cudaMemcpy(residues.data(), cipher->data(),
                         count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
              "residue snapshot cudaMemcpy");
  return Sha256(PackU64(residues));
}

RuntimeSnapshot Snapshot(CIPHER cipher) {
  const auto values = Decode(cipher);
  return {Metadata(cipher), values, CipherResiduesSha256(cipher),
          cipher->parms_id(), cipher->data()};
}

void RequirePreserved(const RuntimeSnapshot &before, CIPHER source,
                      const std::string &diagnostic) {
  const RuntimeSnapshot after = Snapshot(source);
  Require(before._metadata == after._metadata &&
              before._residues_sha256 == after._residues_sha256 &&
              before._parms_id == after._parms_id &&
              before._buffer == after._buffer,
          diagnostic, "out-of-place operation mutated its source");
}

std::vector<std::uint64_t> Q0Tower(CIPHER cipher) {
  Require(cipher != nullptr && cipher->data() != nullptr, "DECODE_PROJECTION",
          "ciphertext has no device residue buffer");
  const std::size_t components = cipher->size();
  const std::size_t towers = cipher->coeff_modulus_size();
  const std::size_t degree = cipher->poly_modulus_degree();
  Require(components != 0 && towers != 0 && degree != 0,
          "DECODE_PROJECTION", "ciphertext residue dimensions are empty");
  std::vector<std::uint64_t> result(components * degree);
  for (std::size_t component = 0; component < components; ++component) {
    RequireCuda(
        cudaMemcpy(result.data() + component * degree,
                   cipher->data() + component * towers * degree,
                   degree * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
        "decoded projection q0 cudaMemcpy");
  }
  return result;
}

std::vector<Complex> DecodeStrictQ0(CIPHER result) {
  const Json full_metadata = Metadata(result);
  const std::string full_residues = CipherResiduesSha256(result);
  const phantom::parms_id_type full_parms_id = result->parms_id();
  const std::uint64_t *const full_buffer = result->data();
  const std::vector<std::uint64_t> full_q0 = Q0Tower(result);

  CIPHERTEXT projected;
  Register_ciph_lifetime(&projected);
  Copy_ciph(&projected, result);
  while (Active_q_count(&projected) > 1)
    Mod_switch(&projected, &projected);
  Require(Active_q_count(&projected) == 1 &&
              Get_ciph_size(&projected) == Get_ciph_size(result) &&
              Get_ciph_slots(&projected) == Get_ciph_slots(result) &&
              Sc_degree(&projected) == Sc_degree(result) &&
              Raw_scale(&projected) == Raw_scale(result) &&
              Is_ciph_ntt(&projected) == Is_ciph_ntt(result),
          "DECODE_PROJECTION",
          "strict q0 projection changed non-chain metadata");
  Require(Q0Tower(&projected) == full_q0, "DECODE_PROJECTION",
          "strict q0 projection changed the retained tower");
  std::vector<Complex> values = Decode(&projected);
  Free_ciph(&projected);

  Require(Metadata(result) == full_metadata &&
              CipherResiduesSha256(result) == full_residues &&
              result->parms_id() == full_parms_id &&
              result->data() == full_buffer,
          "DECODE_PROJECTION",
          "strict q0 projection mutated the full-Q result");
  return values;
}

class RuntimeArena final {
public:
  CIPHER NewCipher() {
    _ciphers.push_back(std::make_unique<CIPHERTEXT>());
    return _ciphers.back().get();
  }

  PLAIN NewPlain() {
    _plains.push_back(std::make_unique<PLAINTEXT>());
    return _plains.back().get();
  }

  void Free(CIPHER cipher) {
    if (cipher != nullptr && _freed_ciphers.insert(cipher).second) {
      Free_ciph(cipher);
    }
  }

  void Free(PLAIN plain) {
    if (plain != nullptr && _freed_plains.insert(plain).second) {
      Free_plain(plain);
    }
  }

  void FreeAll() {
    for (const auto &cipher : _ciphers)
      Free(cipher.get());
    for (const auto &plain : _plains)
      Free(plain.get());
  }

private:
  std::vector<std::unique_ptr<CIPHERTEXT>> _ciphers;
  std::vector<std::unique_ptr<PLAINTEXT>> _plains;
  std::set<CIPHER> _freed_ciphers;
  std::set<PLAIN> _freed_plains;
};

CIPHER Encrypt(RuntimeArena &arena, const std::vector<Complex> &values,
               int active_q_count) {
  PLAIN plain = arena.NewPlain();
  std::vector<DCMPLX> input(values.begin(), values.end());
  Encode_dcmplx(plain, input.data(), input.size(), 1.0, active_q_count);
  CIPHER cipher = arena.NewCipher();
  Phantom_encrypt_plain(cipher, plain);
  arena.Free(plain);
  return cipher;
}

CIPHER CopyCipher(RuntimeArena &arena, CIPHER source) {
  CIPHER result = arena.NewCipher();
  Register_ciph_lifetime(result);
  Copy_ciph(result, source);
  return result;
}

std::vector<Complex> LoadSource(const Json &analytic,
                                const std::vector<std::uint8_t> &binary) {
  ValidateBinary(binary, analytic.at("binary"), kDecodedMagic, "ANALYTIC");
  for (const auto &input : analytic.at("inputs")) {
    if (input.at("input_id") == "bounded_nonperiodic") {
      auto values =
          ReadComplexBlob(binary, input.at("values"), "ANALYTIC_INPUT");
      Require(values.size() == ContextManifest()._logical_slots,
              "ANALYTIC_INPUT", "source slot count mismatch");
      return values;
    }
  }
  Fail("ANALYTIC_INPUT", "bounded_nonperiodic input is absent");
}

std::uint32_t ResolvePower(const std::string &symbol, std::uint32_t degree) {
  if (symbol == "0")
    return 0;
  if (symbol == "N/2")
    return degree / 2U;
  if (symbol == "N")
    return degree;
  if (symbol == "3N/2")
    return degree + degree / 2U;
  if (symbol == "2N-1")
    return 2U * degree - 1U;
  if (symbol == "2N+1")
    return 1U;
  Fail("POWER", "unsupported symbolic monomial power " + symbol);
}

std::string PowerLabel(const std::string &symbol) {
  if (symbol == "0")
    return "0";
  if (symbol == "N/2")
    return "N_over_2";
  if (symbol == "N")
    return "N";
  if (symbol == "3N/2")
    return "3N_over_2";
  if (symbol == "2N-1")
    return "2N_minus_1";
  if (symbol == "2N+1")
    return "2N_plus_1";
  Fail("POWER", "unsupported symbolic monomial power " + symbol);
}

Json MakeDecodedRecord(const std::string &case_id, const std::string &operation,
                       CIPHER result, const RuntimeSnapshot &before,
                       CIPHER source, const Json &values_descriptor,
                       const Json &ownership_token = nullptr) {
  const RuntimeSnapshot after = Snapshot(source);
  Require(before._metadata == after._metadata &&
              before._residues_sha256 == after._residues_sha256 &&
              before._parms_id == after._parms_id &&
              before._buffer == after._buffer,
          "SOURCE_PRESERVATION", case_id + " mutated its source");
  return {{"case_id", case_id},
          {"operation", operation},
          {"metadata", Metadata(result)},
          {"source_metadata_before", before._metadata},
          {"source_metadata_after", after._metadata},
          {"source_values_sha256_before", before._residues_sha256},
          {"source_values_sha256_after", after._residues_sha256},
          {"ownership_token", ownership_token},
          {"values", values_descriptor}};
}

Json AppendDecodedRecord(std::vector<std::uint8_t> &binary,
                         const std::string &case_id,
                         const std::string &operation, CIPHER result,
                         const RuntimeSnapshot &before, CIPHER source,
                         const Json &ownership_token = nullptr,
                         bool project_to_q0 = false) {
  const auto values = project_to_q0 ? DecodeStrictQ0(result) : Decode(result);
  const Json descriptor =
      AppendBlob(binary, PackComplex(values), values.size());
  Json record = MakeDecodedRecord(case_id, operation, result, before, source,
                                  descriptor, ownership_token);
  record["decoded_projection"] =
      project_to_q0
          ? Json{{"kind", "strict_q0_prefix_drop"},
                 {"active_q_count", 1}}
          : Json(nullptr);
  return record;
}

std::string OwnershipToken(CIPHER output, std::size_t serial) {
  std::ostringstream token;
  token << "cipher-" << serial << '-' << std::hex
        << reinterpret_cast<std::uintptr_t>(output) << '-'
        << reinterpret_cast<std::uintptr_t>(output->data());
  return token.str();
}

void RequireIndependentBatch(CIPHER source, CIPHER outputs, std::size_t count,
                             const std::string &diagnostic) {
  std::set<const std::uint64_t *> buffers;
  buffers.insert(source->data());
  for (std::size_t index = 0; index < count; ++index) {
    Require(&outputs[index] != source, diagnostic,
            "source overlaps the output array");
    Require(outputs[index].data() != nullptr &&
                buffers.insert(outputs[index].data()).second,
            diagnostic, "batch outputs do not own distinct device buffers");
  }
}

void FreeBatchInOrder(CIPHER outputs, std::size_t count,
                      const Json &requested_order) {
  std::vector<bool> freed(count, false);
  for (const auto &raw_index : requested_order) {
    const std::size_t index = raw_index.get<std::size_t>();
    Require(index < count && !freed[index], "OWNERSHIP_FREE_ORDER",
            "free order is invalid");
    Free_ciph(&outputs[index]);
    freed[index] = true;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!freed[index])
      Free_ciph(&outputs[index]);
  }
}

Json RunDecoded(const Json &fixture, const std::vector<Complex> &source_values,
                std::vector<std::uint8_t> &binary) {
  const auto &context = ContextManifest();
  const auto &resources = ResourceManifest();
  Require((resources._flags & PHANTOM_RESOURCE_CONJUGATION_KEY) != 0 &&
              (resources._flags & PHANTOM_RESOURCE_ROTATE_BATCH) != 0 &&
              (resources._flags & PHANTOM_RESOURCE_RAISE_MOD) != 0 &&
              (resources._flags & PHANTOM_RESOURCE_MONOMIALS) != 0,
          "RESOURCE_MANIFEST", "retained operation flags are incomplete");
  Require(context._input_level == 1, "RAISE_MOD_SOURCE_LEVEL",
          "observed compiler input active-Q count " +
              std::to_string(context._input_level) +
              ", expected bottom active-Q count 1");
  Json records = Json::array();
  RuntimeArena arena;
  std::size_t ownership_serial = 0;
  std::vector<std::unique_ptr<CIPHERTEXT[]>> retained_batches;

#ifndef ACE_REJECTION_ONLY
  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHER result = arena.NewCipher();
    Conjugate_ciph(result, source);
    records.push_back(AppendDecodedRecord(binary,
                                          "conjugate.bounded_nonperiodic",
                                          "conjugate", result, before, source));
  }
  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHER result = arena.NewCipher();
    Conjugate_ciph(result, source);
    Conjugate_ciph(result, result);
    records.push_back(
        AppendDecodedRecord(binary, "conjugate_twice.bounded_nonperiodic",
                            "conjugate_twice", result, before, source));
  }

  auto run_batch = [&](const std::vector<std::int32_t> &steps,
                       const std::string &prefix,
                       bool verify_fixture_ownership) {
    Require(!steps.empty(), "ROTATE_BATCH_FIXTURE",
            "rotate_batch_steps must not be empty");
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    auto outputs = std::make_unique<CIPHERTEXT[]>(steps.size());
    Rotate_batch_ciph(outputs.get(), source, steps.data(), steps.size());
    RequireIndependentBatch(source, outputs.get(), outputs ? steps.size() : 0,
                            "ROTATE_BATCH_OWNERSHIP");
    std::vector<std::vector<std::uint8_t>> observed_values;
    observed_values.reserve(steps.size());
    for (std::size_t position = 0; position < steps.size(); ++position) {
      observed_values.push_back(PackComplex(Decode(&outputs[position])));
      if (steps[position] == 0) {
        Require(observed_values.back() == PackComplex(before._values),
                "ROTATE_BATCH_ZERO", "zero rotation is not an identity copy");
      }
      for (std::size_t previous = 0; previous < position; ++previous) {
        if (steps[previous] == steps[position]) {
          Require(observed_values[previous] == observed_values[position],
                  "ROTATE_BATCH_DUPLICATE",
                  "duplicate steps produced different values");
        }
      }
      const std::string case_id = prefix + ".output_" +
                                  std::to_string(position) + ".step_" +
                                  std::to_string(steps[position]);
      records.push_back(AppendDecodedRecord(
          binary, case_id, "rotate_batch", &outputs[position], before, source,
          OwnershipToken(&outputs[position], ownership_serial++)));
    }
    RequirePreserved(before, source, "ROTATE_BATCH_SOURCE");
    if (verify_fixture_ownership) {
      Require(steps.size() > 1U, "ROTATE_BATCH_OWNERSHIP",
              "ownership verification requires at least two outputs");
      const auto mutation = std::find_if(
          steps.begin(), steps.end(), [](std::int32_t step) { return step != 0; });
      Require(mutation != steps.end(), "ROTATE_BATCH_OWNERSHIP",
              "ownership verification requires a nonzero step");
      const std::size_t mutation_index =
          static_cast<std::size_t>(std::distance(steps.begin(), mutation));
      std::vector<std::vector<std::uint8_t>> sibling_values(steps.size());
      for (std::size_t index = 0; index < steps.size(); ++index) {
        if (index != mutation_index)
          sibling_values[index] = PackComplex(Decode(&outputs[index]));
      }
      Add_scalar(&outputs[mutation_index], &outputs[mutation_index], 0.125);
      for (std::size_t index = 0; index < steps.size(); ++index) {
        if (index != mutation_index) {
          Require(PackComplex(Decode(&outputs[index])) == sibling_values[index],
                  "ROTATE_BATCH_OWNERSHIP",
                  "mutating one output changed a sibling output");
        }
      }
    }
    if (steps.size() == fixture.at("ownership").at("free_order").size()) {
      FreeBatchInOrder(outputs.get(), steps.size(),
                       fixture.at("ownership").at("free_order"));
    } else {
      Json reverse_order = Json::array();
      for (std::size_t index = steps.size(); index > 0; --index) {
        reverse_order.push_back(index - 1U);
      }
      FreeBatchInOrder(outputs.get(), steps.size(), reverse_order);
    }
    retained_batches.push_back(std::move(outputs));
  };

  std::vector<std::int32_t> edge_steps;
  for (const auto &step : fixture.at("rotate_batch_steps")) {
    edge_steps.push_back(step.get<std::int32_t>());
  }
  run_batch(edge_steps, "rotate_batch.bounded_nonperiodic", true);
  std::size_t batch_index = 0;
  for (const auto &batch : fixture.at("production_rotation_batches")) {
    std::vector<std::int32_t> steps;
    for (const auto &step : batch)
      steps.push_back(step.get<std::int32_t>());
    run_batch(steps,
              "rotate_batch.production_" + std::to_string(batch_index++),
              false);
  }

  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHER result = arena.NewCipher();
    Raise_mod(result, source,
              static_cast<std::uint32_t>(context._data_q_count));
    records.push_back(AppendDecodedRecord(binary,
                                          "raise_mod.bounded_nonperiodic",
                                          "raise_mod", result, before, source,
                                          nullptr, true));
  }

  for (const auto &raw_symbol : fixture.at("monomial_powers")) {
    const std::string symbol = raw_symbol.get<std::string>();
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHER result = arena.NewCipher();
    Mul_mono_ciph(result, source, ResolvePower(symbol, context._poly_degree));
    records.push_back(AppendDecodedRecord(
        binary, "mul_mono." + PowerLabel(symbol) + ".bounded_nonperiodic",
        "mul_mono", result, before, source));
  }
  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHER first = arena.NewCipher();
    CIPHER result = arena.NewCipher();
    Mul_mono_ciph(first, source, 2U * context._poly_degree - 1U);
    Mul_mono_ciph(result, first, 1U);
    records.push_back(AppendDecodedRecord(
        binary, "mul_mono.inverse_composition.bounded_nonperiodic",
        "mul_mono_inverse_composition", result, before, source));
  }
  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    CIPHERTEXT result = retained_ckks_composite(*source);
    records.push_back(AppendDecodedRecord(binary,
                                          "composite.bounded_nonperiodic",
                                          "composite", &result, before, source,
                                          nullptr, true));
    Zero_ciph(&result);
  }
#endif
  arena.FreeAll();
  return records;
}

std::unique_ptr<PhantomContext> MakeExactContext() {
  const auto &manifest = ContextManifest();
  phantom::EncryptionParameters parameters(phantom::scheme_type::ckks);
  parameters.set_poly_modulus_degree(manifest._poly_degree);
  std::vector<int> bit_sizes;
  bit_sizes.reserve(manifest._data_q_count + manifest._special_p_count);
  for (std::size_t index = 0; index < manifest._data_q_count; ++index) {
    bit_sizes.push_back(static_cast<int>(manifest._data_q_bit_sizes[index]));
  }
  for (std::size_t index = 0; index < manifest._special_p_count; ++index) {
    bit_sizes.push_back(static_cast<int>(manifest._special_p_bit_sizes[index]));
  }
  parameters.set_coeff_modulus(
      phantom::arith::CoeffModulus::Create(manifest._poly_degree, bit_sizes));
  parameters.set_special_modulus_size(manifest._special_p_count);
  parameters.set_secret_key_hamming_weight(manifest._hamming_weight);
  parameters.set_sparse_slots(manifest._logical_slots);
  return std::make_unique<PhantomContext>(parameters);
}

std::vector<std::uint64_t> ExactModuli(const PhantomContext &context) {
  const auto &moduli = context.first_context_data().parms().coeff_modulus();
  std::vector<std::uint64_t> result;
  result.reserve(moduli.size());
  for (const auto &modulus : moduli)
    result.push_back(modulus.value());
  return result;
}

std::uint64_t ReduceSignedCoefficient(std::int64_t value,
                                      std::uint64_t modulus) {
  if (value >= 0)
    return static_cast<std::uint64_t>(value) % modulus;
  const std::uint64_t magnitude =
      static_cast<std::uint64_t>(-(value + 1)) + 1U;
  const std::uint64_t residue = magnitude % modulus;
  return residue == 0 ? 0 : modulus - residue;
}

std::vector<std::uint64_t>
ReduceSignedSource(const std::vector<std::int64_t> &coefficients,
                   const std::vector<std::uint64_t> &moduli) {
  const std::size_t degree = ContextManifest()._poly_degree;
  Require(coefficients.size() == 2U * degree, "EXACT_SOURCE",
          "signed source must contain two polynomial components");
  std::vector<std::uint64_t> result;
  result.reserve(coefficients.size() * moduli.size());
  for (std::size_t component = 0; component < 2U; ++component) {
    const std::size_t begin = component * degree;
    for (std::uint64_t modulus : moduli) {
      for (std::size_t coefficient = 0; coefficient < degree; ++coefficient) {
        result.push_back(
            ReduceSignedCoefficient(coefficients[begin + coefficient], modulus));
      }
    }
  }
  return result;
}

PhantomCiphertext
ImportCoefficientCipher(const PhantomContext &context,
                        const std::vector<std::uint64_t> &values,
                        std::size_t active_q_count) {
  const auto &manifest = ContextManifest();
  Require(active_q_count >= 1 && active_q_count <= manifest._data_q_count,
          "EXACT_IMPORT", "active-Q count is invalid");
  const std::size_t chain_index =
      context.get_first_index() + manifest._data_q_count - active_q_count;
  const std::size_t expected =
      2U * active_q_count * static_cast<std::size_t>(manifest._poly_degree);
  Require(values.size() == expected, "EXACT_IMPORT",
          "coefficient source shape mismatch");
  PhantomCiphertext cipher;
  const auto &stream = phantom::util::global_variables::current_stream();
  cipher.resize(context, chain_index, 2, stream.get_stream());
  cipher.set_ntt_form(false);
  cipher.SetNoiseScaleDeg(1);
  cipher.set_scale(std::ldexp(1.0, manifest._scaling_modulus_bits));
  {
    auto destination_access = cipher.write_access(stream.get_stream());
    RequireCuda(cudaMemcpyAsync(cipher.data(), values.data(),
                                values.size() * sizeof(std::uint64_t),
                                cudaMemcpyHostToDevice, stream.get_stream()),
                "exact import cudaMemcpyAsync");
  }
  RequireCuda(cudaStreamSynchronize(stream.get_stream()),
              "exact import cudaStreamSynchronize");
  return cipher;
}

Json ExactMetadata(const PhantomCiphertext &cipher) {
  return {{"active_q_count", cipher.coeff_modulus_size()},
          {"ciphertext_size", cipher.size()},
          {"ntt", cipher.is_ntt_form()},
          {"chain_index", cipher.chain_index()},
          {"scale_degree", cipher.GetNoiseScaleDeg()},
          {"raw_scale", cipher.scale()}};
}

Json ExactLayout(std::size_t source_q_count, std::size_t result_q_count) {
  return {{"component_count", 2},
          {"source_modulus_count", source_q_count},
          {"result_modulus_count", result_q_count},
          {"coefficient_count", ContextManifest()._poly_degree},
          {"ordering", "component,modulus,coefficient"}};
}

Json AppendExactRecord(std::vector<std::uint8_t> &output,
                       const std::string &case_id, const std::string &operation,
                       const Json &power, const Json &source_metadata,
                       const Json &source_after_metadata,
                       const Json &result_metadata,
                       const std::vector<std::uint64_t> &source,
                       const std::vector<std::uint64_t> &source_after,
                       const std::vector<std::uint64_t> &actual,
                       std::size_t source_q_count, std::size_t result_q_count) {
  return {{"case_id", case_id},
          {"operation", operation},
          {"normalized_power", power},
          {"source_metadata", source_metadata},
          {"source_metadata_after", source_after_metadata},
          {"result_metadata", result_metadata},
          {"source", AppendBlob(output, PackU64(source), source.size())},
          {"source_after",
           AppendBlob(output, PackU64(source_after), source_after.size())},
          {"actual", AppendBlob(output, PackU64(actual), actual.size())},
          {"layout", ExactLayout(source_q_count, result_q_count)}};
}

Json RunExact(const Json &exact_reference,
              const std::vector<std::uint8_t> &exact_input,
              std::vector<std::uint8_t> &output,
              std::vector<std::uint64_t> &ordered_moduli,
              std::size_t &first_data_chain_index) {
  ValidateBinary(exact_input, exact_reference.at("binary"), kExactSourceMagic,
                 "EXACT_REFERENCE");
  Require(exact_reference.at("schema_version") ==
              "ace.phantom.retained_ckks.exact-source/2.0.0",
          "EXACT_REFERENCE", "signed exact source schema mismatch");
  auto context = MakeExactContext();
  first_data_chain_index = context->get_first_index();
  ordered_moduli = ExactModuli(*context);
  const auto signed_coefficients =
      ReadI64Blob(exact_input, exact_reference.at("signed_coefficients"),
                  "EXACT_SIGNED_SOURCE");
  Require(exact_reference.at("source_id") == "signed_coefficients.default" &&
              signed_coefficients.size() ==
                  2U * ContextManifest()._poly_degree,
          "EXACT_SIGNED_SOURCE", "signed exact source shape changed");
  const std::vector<std::uint64_t> bottom_source =
      ReduceSignedSource(signed_coefficients, {ordered_moduli.front()});
  const std::vector<std::uint64_t> full_source =
      ReduceSignedSource(signed_coefficients, ordered_moduli);
  const std::array<std::string, 8> expected_ids = {
      "exact_algebraic.raise_mod",
      "exact_algebraic.mul_mono.0",
      "exact_algebraic.mul_mono.N_over_2",
      "exact_algebraic.mul_mono.N",
      "exact_algebraic.mul_mono.3N_over_2",
      "exact_algebraic.mul_mono.2N_minus_1",
      "exact_algebraic.mul_mono.2N_plus_1",
      "exact_algebraic.mul_mono.inverse_composition"};
  Require(exact_reference.at("records").size() == expected_ids.size(),
          "EXACT_CASES", "exact case count mismatch");
  Json records = Json::array();
  for (std::size_t index = 0; index < expected_ids.size(); ++index) {
    const Json &specification = exact_reference.at("records").at(index);
    Require(specification.at("case_id") == expected_ids[index], "EXACT_CASES",
            "exact case order mismatch");
    Require(specification.at("source_id") == exact_reference.at("source_id"),
            "EXACT_CASES", "exact case references another signed source");
    const auto &source_values = index == 0 ? bottom_source : full_source;
    const std::string runtime_id =
        "exact_runtime." +
        expected_ids[index].substr(std::string("exact_algebraic.").size());
    if (index == 0) {
      PhantomCiphertext source =
          ImportCoefficientCipher(*context, source_values, 1);
      const Json before = ExactMetadata(source);
      const auto source_before =
          export_ciphertext_coefficients(*context, source);
      PhantomCiphertext result;
      raise_modulus(*context, source, ordered_moduli.size(), result);
      const auto source_after =
          export_ciphertext_coefficients(*context, source);
      const auto actual = export_ciphertext_coefficients(*context, result);

      PhantomCiphertext ntt_source =
          copy_ciphertext_to_ntt_form(*context, source);
      const Json ntt_source_metadata = ExactMetadata(ntt_source);
      const auto ntt_source_before =
          export_ciphertext_coefficients(*context, ntt_source);
      const auto *ntt_source_storage = ntt_source.data();
      PhantomCiphertext ntt_result;
      raise_modulus(*context, ntt_source, ordered_moduli.size(), ntt_result);
      const auto ntt_source_after =
          export_ciphertext_coefficients(*context, ntt_source);
      const auto ntt_actual =
          export_ciphertext_coefficients(*context, ntt_result);
      Require(ntt_source.is_ntt_form() && ntt_result.is_ntt_form(),
              "EXACT_NTT_RAISE",
              runtime_id + " did not preserve valid NTT form");
      Require(ExactMetadata(ntt_source) == ntt_source_metadata &&
                  ntt_source.data() == ntt_source_storage &&
                  ntt_source_before == ntt_source_after,
              "EXACT_NTT_RAISE", runtime_id + " mutated its NTT source");
      Require(ntt_source_before == source_before && ntt_actual == actual,
              "EXACT_NTT_RAISE",
              runtime_id +
                  " NTT and coefficient-form exact residues disagree");
      records.push_back(AppendExactRecord(
          output, runtime_id, "raise_mod", nullptr, before,
          ExactMetadata(source), ExactMetadata(result), source_before,
          source_after, actual, 1, ordered_moduli.size()));
      continue;
    }
    PhantomCiphertext source =
        ImportCoefficientCipher(*context, source_values, ordered_moduli.size());
    const Json before = ExactMetadata(source);
    const auto source_before = export_ciphertext_coefficients(*context, source);
    PhantomCiphertext result;
    Json normalized_power = specification.at("normalized_power");
    std::string operation = specification.at("operation");
    if (operation == "mul_mono") {
      const std::uint64_t power = normalized_power.get<std::uint64_t>();
      multiply_by_monomial(*context, source, power, result);
      PhantomCiphertext ntt_source =
          copy_ciphertext_to_ntt_form(*context, source);
      PhantomCiphertext ntt_result;
      multiply_by_monomial(*context, ntt_source, power, ntt_result);
      const auto ntt_coefficients =
          export_ciphertext_coefficients(*context, ntt_result);
      const auto coefficient_result =
          export_ciphertext_coefficients(*context, result);
      Require(ntt_coefficients == coefficient_result, "EXACT_NTT_MONOMIAL",
              runtime_id + " NTT and coefficient paths disagree");
    } else {
      Require(operation == "mul_mono_inverse_composition", "EXACT_OPERATION",
              "unsupported exact operation");
      PhantomCiphertext first;
      multiply_by_monomial(*context, source,
                           2U * ContextManifest()._poly_degree - 1U, first);
      multiply_by_monomial(*context, first, 1U, result);
    }
    const auto source_after = export_ciphertext_coefficients(*context, source);
    const auto actual = export_ciphertext_coefficients(*context, result);
    records.push_back(AppendExactRecord(
        output, runtime_id, operation, normalized_power, before,
        ExactMetadata(source), ExactMetadata(result), source_before,
        source_after, actual, ordered_moduli.size(), ordered_moduli.size()));
  }
  return records;
}

Json BinaryDescriptor(const std::string &format,
                      const std::vector<std::uint8_t> &bytes) {
  return {{"format", format},
          {"size_bytes", bytes.size()},
          {"sha256", Sha256(bytes)}};
}

void RequireDecodedOrder(const Json &fixture, const Json &analytic,
                         const Json &records) {
  std::vector<std::string> expected;
  std::set<std::string> present;
  for (const auto &record : analytic.at("records")) {
    const std::string case_id = record.at("case_id").get<std::string>();
    Require(present.insert(case_id).second, "DECODED_ORDER",
            "analytic reference contains duplicate case " + case_id);
    expected.push_back(case_id);
  }
  for (const auto &record : fixture.at("decoded_cases")) {
    const std::string case_id = record.at("id").get<std::string>();
    if (case_id == "rotate_batch.bounded_nonperiodic" ||
        present.count(case_id) != 0) {
      continue;
    }
    Require(present.insert(case_id).second, "DECODED_ORDER",
            "fixture contains duplicate decoded case " + case_id);
    expected.push_back(case_id);
  }
  Require(records.size() == expected.size(), "DECODED_ORDER",
          "observed decoded record count " + std::to_string(records.size()) +
              ", expected " + std::to_string(expected.size()));
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const std::string observed =
        records.at(index).at("case_id").get<std::string>();
    Require(observed == expected[index], "DECODED_ORDER",
            "observed case " + observed + " at position " +
                std::to_string(index) + ", expected " + expected[index]);
  }
}

void RunConformance(int argc, char **argv) {
  Require(argc == 13, "ARGUMENTS",
          "conformance requires fixture, context manifest, analytic json/bin, "
          "exact json/bin, GPU, decoded json/bin, and exact json/bin outputs");
  const Json fixture = LoadJson(argv[2]);
  const AuthenticatedContext authenticated_context =
      AuthenticateContext(argv[3]);
  RequireContextBinding(fixture, authenticated_context);
  const Json analytic = LoadJson(argv[4]);
  const auto analytic_binary = ReadBytes(argv[5]);
  Require(fixture.at("fixture_id") == "retained_ckks_v1", "FIXTURE_ID",
          "conformance fixture identifier is unsupported");
  const Json exact_reference = LoadJson(argv[6]);
  const auto exact_input = ReadBytes(argv[7]);
  CheckGpu(argv[8]);
  const std::string fixture_sha256 = Sha256(ReadBytes(argv[2]));
  Require(analytic.at("qualification_bindings") ==
                  fixture.at("qualification_bindings") &&
              exact_reference.at("qualification_bindings") ==
                  fixture.at("qualification_bindings"),
          "QUALIFICATION_BINDING",
          "generated reference qualification bindings differ from fixture");
  Require(analytic.at("fixture_sha256") == fixture_sha256 &&
              exact_reference.at("fixture_sha256") == fixture_sha256,
          "FIXTURE_BINDING", "generated references use another fixture");
  Require(exact_reference.at("context_manifest_sha256") ==
              authenticated_context._sha256,
          "CONTEXT_BINDING",
          "exact reference context SHA-256 differs from the authenticated "
          "manifest");

  const auto source_values = LoadSource(analytic, analytic_binary);
  Prepare_context();
  std::vector<std::uint8_t> decoded_binary(kDecodedMagic.begin(),
                                           kDecodedMagic.end());
  Json decoded_records = RunDecoded(fixture, source_values, decoded_binary);
  RequireDecodedOrder(fixture, analytic, decoded_records);
  std::vector<std::uint8_t> exact_binary(kExactMagic.begin(),
                                         kExactMagic.end());
  std::vector<std::uint64_t> ordered_moduli;
  std::size_t first_data_chain_index = 0;
  Json exact_records = RunExact(exact_reference, exact_input, exact_binary,
                                ordered_moduli, first_data_chain_index);
  Finalize_context();

  WriteBytes(argv[10], decoded_binary);
  WriteBytes(argv[12], exact_binary);
  const std::string &context_sha256 = authenticated_context._sha256;
  WriteJson(
      argv[9],
      {{"schema_version", kProviderSchema},
       {"provider", "phantom"},
       {"fixture_sha256", fixture_sha256},
       {"context_manifest_sha256", context_sha256},
       {"qualification_bindings", fixture.at("qualification_bindings")},
       {"first_data_chain_index", first_data_chain_index},
       {"identifiers",
        {{"ace_commit", ACE_COMMIT_ID},
         {"phantom_commit", PHANTOM_COMMIT_ID},
         {"executable_sha256", Sha256(ReadBytes(argv[0]))}}},
       {"binary", BinaryDescriptor(fixture.at("decoded_binary_format").at("id"),
                                   decoded_binary)},
       {"records", std::move(decoded_records)}});
  WriteJson(
      argv[11],
      {{"schema_version", kExactObservedSchema},
       {"fixture_sha256", fixture_sha256},
       {"context_manifest_sha256", context_sha256},
       {"exact_source_json_sha256", Sha256(ReadBytes(argv[6]))},
       {"exact_source_binary_sha256", Sha256(exact_input)},
       {"ordered_data_q_moduli", ordered_moduli},
       {"first_data_chain_index", first_data_chain_index},
       {"conversion_convention", exact_reference.at("conversion_convention")},
       {"binary",
        BinaryDescriptor(
            fixture.at("exact_observed_binary_format").at("id"), exact_binary)},
       {"records", std::move(exact_records)}});
}

void RunOwnership(int argc, char **argv) {
  Require(argc == 8, "ARGUMENTS",
          "ownership requires fixture, context manifest, analytic json/bin, "
          "GPU, and output");
  const Json fixture = LoadJson(argv[2]);
  const AuthenticatedContext authenticated_context =
      AuthenticateContext(argv[3]);
  RequireContextBinding(fixture, authenticated_context);
  const Json analytic = LoadJson(argv[4]);
  const auto analytic_binary = ReadBytes(argv[5]);
  CheckGpu(argv[6]);
  const auto source_values = LoadSource(analytic, analytic_binary);
  const std::size_t iterations =
      fixture.at("ownership").at("iterations").get<std::size_t>();
  Require(iterations > 0, "OWNERSHIP_ITERATIONS",
          "ownership fixture must request at least one iteration");
  Prepare_context();
  RuntimeArena arena;
  const auto &context = ContextManifest();
  std::vector<std::unique_ptr<CIPHERTEXT[]>> retained_batches;
  std::vector<std::int32_t> steps;
  for (const auto &step : fixture.at("rotate_batch_steps")) {
    steps.push_back(step.get<std::int32_t>());
  }
  Require(steps.size() > 1U, "OWNERSHIP_STEPS",
          "ownership fixture requires at least two batch steps");
  const auto mutation = std::find_if(
      steps.begin(), steps.end(), [](std::int32_t step) { return step != 0; });
  Require(mutation != steps.end(), "OWNERSHIP_STEPS",
          "ownership fixture requires a nonzero batch step");
  const std::size_t mutation_index =
      static_cast<std::size_t>(std::distance(steps.begin(), mutation));
  const bool batch_outputs_are_independent =
      fixture.at("ownership").at("batch_outputs_are_independent").get<bool>();
  Require(batch_outputs_are_independent, "OWNERSHIP_CONTRACT",
          "ownership fixture must require independent batch outputs");
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot before = Snapshot(source);
    auto outputs = std::make_unique<CIPHERTEXT[]>(steps.size());
    Rotate_batch_ciph(outputs.get(), source, steps.data(), steps.size());
    RequireIndependentBatch(source, outputs.get(), steps.size(),
                            "OWNERSHIP_BUFFERS");
    std::vector<std::vector<std::uint8_t>> sibling_values(steps.size());
    for (std::size_t index = 0; index < steps.size(); ++index) {
      if (index != mutation_index)
        sibling_values[index] = PackComplex(Decode(&outputs[index]));
    }
    Add_scalar(&outputs[mutation_index], &outputs[mutation_index], 0.25);
    for (std::size_t index = 0; index < steps.size(); ++index) {
      if (index != mutation_index) {
        Require(PackComplex(Decode(&outputs[index])) == sibling_values[index],
                "OWNERSHIP_MUTATION", "batch output storage aliases");
      }
    }
    RequirePreserved(before, source, "OWNERSHIP_SOURCE");
    FreeBatchInOrder(outputs.get(), steps.size(),
                     fixture.at("ownership").at("free_order"));
    retained_batches.push_back(std::move(outputs));
    arena.Free(source);
  }
  arena.FreeAll();
  Finalize_context();
  RequireCuda(cudaDeviceSynchronize(), "ownership cudaDeviceSynchronize");
  WriteJson(argv[7],
            {{"schema_version", "ace.phantom.retained_ckks.ownership/1.0.0"},
             {"status", "pass"},
             {"iterations", iterations},
             {"ordered_steps", steps},
             {"batch_outputs_are_independent", batch_outputs_are_independent},
             {"free_order", fixture.at("ownership").at("free_order")}});
}

void RunAliases(int argc, char **argv) {
  Require(argc == 8, "ARGUMENTS",
          "aliases requires fixture, context manifest, analytic json/bin, "
          "GPU, and output");
  const Json fixture = LoadJson(argv[2]);
  const AuthenticatedContext authenticated_context =
      AuthenticateContext(argv[3]);
  RequireContextBinding(fixture, authenticated_context);
  const Json analytic = LoadJson(argv[4]);
  const auto analytic_binary = ReadBytes(argv[5]);
  CheckGpu(argv[6]);
  const std::string fixture_sha256 = Sha256(ReadBytes(argv[2]));
  Require(fixture.at("fixture_id") == "retained_ckks_v1", "FIXTURE_ID",
          "alias fixture identifier is unsupported");
  Require(analytic.at("fixture_sha256") == fixture_sha256 &&
              analytic.at("qualification_bindings")
                      .at("compiler_context_manifest_sha256") ==
                  authenticated_context._sha256 &&
              analytic.at("qualification_bindings") ==
                  fixture.at("qualification_bindings"),
          "ALIAS_FIXTURE_BINDING",
          "analytic alias input uses another fixture");
  const auto source_values = LoadSource(analytic, analytic_binary);

  Prepare_context();
  RuntimeArena arena;
  const auto &context = ContextManifest();
  Json cases = Json::array();
  auto append_case = [&](const std::string &case_id,
                         const std::string &operation, const Json &symbol,
                         const Json &normalized_power, CIPHER expected,
                         CIPHER alias, const RuntimeSnapshot &alias_before,
                         bool alias_returned) {
    const RuntimeSnapshot expected_after = Snapshot(expected);
    const RuntimeSnapshot alias_after = Snapshot(alias);
    const auto input_values = PackComplex(alias_before._values);
    const auto expected_values = PackComplex(expected_after._values);
    const auto alias_values = PackComplex(alias_after._values);
    const bool metadata_matches =
        expected_after._metadata == alias_after._metadata;
    const bool decoded_values_match = expected_values == alias_values;
    const bool residues_match =
        expected_after._residues_sha256 == alias_after._residues_sha256;
    const bool is_identity = operation == "mul_mono" &&
                             symbol.is_string() && symbol == Json("0");
    const bool identity_matches_source =
        !is_identity ||
        (alias_before._metadata == alias_after._metadata &&
         alias_before._residues_sha256 == alias_after._residues_sha256 &&
         input_values == alias_values);
    Require(alias_returned && metadata_matches && decoded_values_match &&
                residues_match && identity_matches_source,
            "ADAPTER_ALIAS", case_id + " differs from out-of-place dispatch");
    cases.push_back(
        {{"case_id", case_id},
         {"operation", operation},
         {"symbol", symbol},
         {"normalized_power", normalized_power},
         {"status", "pass"},
         {"alias_returned", alias_returned},
         {"source_preserved", true},
         {"metadata_matches_out_of_place", metadata_matches},
         {"decoded_values_match_out_of_place", decoded_values_match},
         {"residues_match_out_of_place", residues_match},
         {"identity_matches_source",
          is_identity ? Json(identity_matches_source) : Json(nullptr)},
         {"input_metadata", alias_before._metadata},
         {"out_of_place_metadata", expected_after._metadata},
         {"in_place_metadata", alias_after._metadata},
         {"input_decoded_sha256", Sha256(input_values)},
         {"out_of_place_decoded_sha256", Sha256(expected_values)},
         {"in_place_decoded_sha256", Sha256(alias_values)},
         {"input_residues_sha256", alias_before._residues_sha256},
         {"out_of_place_residues_sha256", expected_after._residues_sha256},
         {"in_place_residues_sha256", alias_after._residues_sha256}});
  };

  {
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot source_before = Snapshot(source);
    CIPHER alias = CopyCipher(arena, source);
    const RuntimeSnapshot alias_before = Snapshot(alias);
    CIPHER expected = arena.NewCipher();
    Conjugate_ciph(expected, source);
    RequirePreserved(source_before, source, "CONJUGATE_ALIAS_SOURCE");
    const bool alias_returned = Conjugate_ciph(alias, alias) == alias;
    append_case("conjugate.in_place", "conjugate", nullptr, nullptr,
                expected, alias, alias_before, alias_returned);
  }
  for (const auto &raw_symbol : fixture.at("monomial_powers")) {
    const std::string symbol = raw_symbol.get<std::string>();
    const std::uint32_t power = ResolvePower(symbol, context._poly_degree);
    CIPHER source = Encrypt(arena, source_values, context._input_level);
    const RuntimeSnapshot source_before = Snapshot(source);
    CIPHER alias = CopyCipher(arena, source);
    const RuntimeSnapshot alias_before = Snapshot(alias);
    CIPHER expected = arena.NewCipher();
    Mul_mono_ciph(expected, source, power);
    RequirePreserved(source_before, source, "MUL_MONO_ALIAS_SOURCE");
    const bool alias_returned = Mul_mono_ciph(alias, alias, power) == alias;
    append_case("mul_mono." + PowerLabel(symbol) + ".in_place", "mul_mono",
                symbol, power, expected, alias, alias_before, alias_returned);
  }
  arena.FreeAll();
  Finalize_context();
  WriteJson(
      argv[7],
      {{"schema_version", "ace.phantom.retained_ckks.adapter-aliases/1.0.0"},
       {"status", "pass"},
       {"fixture_sha256", fixture_sha256},
       {"context_manifest_sha256", authenticated_context._sha256},
       {"qualification_bindings", fixture.at("qualification_bindings")},
       {"cases", std::move(cases)}});
}

void RunRejection(int argc, char **argv) {
  Require(argc == 8, "ARGUMENTS",
          "reject requires fixture, context manifest, analytic json/bin, GPU, "
          "and rejection ID");
  const Json fixture = LoadJson(argv[2]);
  const AuthenticatedContext authenticated_context =
      AuthenticateContext(argv[3]);
  RequireContextBinding(fixture, authenticated_context);
  const Json analytic = LoadJson(argv[4]);
  const auto analytic_binary = ReadBytes(argv[5]);
  Require(fixture.at("fixture_id") == "retained_ckks_v1", "FIXTURE_ID",
          "rejection fixture identifier is unsupported");
  CheckGpu(argv[6]);
  const std::string id = argv[7];
  const auto source_values = LoadSource(analytic, analytic_binary);
  std::vector<std::int32_t> fixture_steps;
  for (const auto &step : fixture.at("rotate_batch_steps")) {
    Require(step.is_number_integer(), "REJECTION_FIXTURE",
            "rotate_batch_steps must contain integers");
    const std::int64_t value = step.get<std::int64_t>();
    Require(value >= std::numeric_limits<std::int32_t>::min() &&
                value <= std::numeric_limits<std::int32_t>::max(),
            "REJECTION_FIXTURE", "rotate_batch_steps value is out of range");
    fixture_steps.push_back(static_cast<std::int32_t>(value));
  }
  Require(!fixture_steps.empty(), "REJECTION_FIXTURE",
          "rotate_batch_steps must not be empty");
  Require(std::any_of(fixture_steps.begin(), fixture_steps.end(),
                      [](std::int32_t step) { return step != 0; }),
          "REJECTION_FIXTURE",
          "rotate_batch_steps must contain a nonzero step");
  const auto zero_step =
      std::find(fixture_steps.begin(), fixture_steps.end(), 0);
  Require(zero_step != fixture_steps.end(), "REJECTION_FIXTURE",
          "rotate_batch_steps must contain the zero-step ownership case");
  const Json *rejection = nullptr;
  std::set<std::string> rejection_ids;
  for (const auto &candidate : fixture.at("runtime_rejections")) {
    Require(candidate.is_object() && candidate.size() == 3U &&
                candidate.contains("id") && candidate.contains("diagnostic") &&
                candidate.contains("manifest") && candidate.at("id").is_string() &&
                candidate.at("diagnostic").is_string() &&
                candidate.at("manifest").is_string(),
            "REJECTION_FIXTURE",
            "runtime rejection entries require id, diagnostic, and manifest "
            "strings");
    const std::string candidate_id = candidate.at("id").get<std::string>();
    Require(rejection_ids.insert(candidate_id).second, "REJECTION_FIXTURE",
            "duplicate runtime rejection ID " + candidate_id);
    if (candidate_id == id)
      rejection = &candidate;
  }
  Require(rejection != nullptr, "REJECTION_ID", "unknown rejection ID " + id);
  const std::string diagnostic =
      rejection->at("diagnostic").get<std::string>();
  const std::string expected_manifest =
      rejection->at("manifest").get<std::string>();
  Require(expected_manifest == ACE_REJECTION_MANIFEST_ID,
          "REJECTION_LINKAGE",
          id + " requires manifest " + expected_manifest + ", linked " +
              ACE_REJECTION_MANIFEST_ID);
  if (id == "conjugate_missing_key") {
    Require((ResourceManifest()._flags & PHANTOM_RESOURCE_CONJUGATION_KEY) == 0,
            "REJECTION_LINKAGE",
            "keyless rejection binary unexpectedly declares conjugation");
  } else if (id == "rotate_batch_missing_nonzero_key") {
    const auto &resources = ResourceManifest();
#ifdef ACE_REJECTION_ONLY
    Require(resources._rotation_count == 0 &&
                resources._rotation_batch_count == 0,
            "REJECTION_LINKAGE",
            "compiler-emitted keyless manifest unexpectedly declares a "
            "rotation resource");
#else
    Require(resources._rotation_batch_count != 0 &&
                resources._rotation_batch_offsets != nullptr &&
                resources._rotation_batch_steps != nullptr,
            "REJECTION_LINKAGE",
            "keyless rotation manifest has no auditable batch");
    bool missing_nonzero_key = false;
    for (std::size_t batch = 0;
         batch < resources._rotation_batch_count && !missing_nonzero_key;
         ++batch) {
      const std::size_t begin = resources._rotation_batch_offsets[batch];
      const std::size_t end = resources._rotation_batch_offsets[batch + 1U];
      for (std::size_t index = begin; index < end; ++index) {
        const std::int32_t step = resources._rotation_batch_steps[index];
        if (step == 0)
          continue;
        const bool declared =
            resources._rotation_count != 0 &&
            std::find(resources._rotation_steps,
                      resources._rotation_steps + resources._rotation_count,
                      step) !=
                resources._rotation_steps + resources._rotation_count;
        missing_nonzero_key = !declared;
        if (missing_nonzero_key)
          break;
      }
    }
    Require(missing_nonzero_key, "REJECTION_LINKAGE",
            "keyless rotation manifest has no unauthorised nonzero batch "
            "step");
#endif
  }
  std::cerr << "ACE_RETAINED_EXPECT_DIAGNOSTIC[" << diagnostic
            << "] rejection=" << id << " manifest=" << ACE_REJECTION_MANIFEST_ID
            << '\n';
  Prepare_context();
  RuntimeArena arena;
  const auto &context = ContextManifest();
  CIPHER source = Encrypt(arena, source_values, context._input_level);
  CIPHER result = arena.NewCipher();
  if (id == "conjugate_missing_key") {
    Conjugate_ciph(result, source);
  } else if (id == "rotate_batch_missing_nonzero_key") {
    std::vector<CIPHERTEXT> outputs(fixture_steps.size());
    Rotate_batch_ciph(outputs.data(), source, fixture_steps.data(),
                      fixture_steps.size());
  } else if (id == "rotate_batch_source_overlap") {
    const std::int32_t step = *zero_step;
    Rotate_batch_ciph(source, source, &step, 1U);
  } else if (id == "raise_alias") {
    Raise_mod(source, source, context._data_q_count);
  } else if (id == "raise_size") {
    source->resize(source->size() + 1U, source->coeff_modulus_size(),
                   source->poly_modulus_degree(), nullptr);
    Raise_mod(result, source, context._data_q_count);
  } else if (id == "raise_chain") {
    Require(context._data_q_count > 1, "REJECTION_SETUP",
            "raise_chain needs more than one data-Q tower");
    CIPHER upper = Encrypt(arena, source_values, context._input_level + 1U);
    Raise_mod(result, upper, context._data_q_count);
  } else if (id == "raise_target") {
    Require(context._data_q_count > 1, "REJECTION_SETUP",
            "raise_target needs more than one data-Q tower");
    Raise_mod(result, source, context._data_q_count - 1);
  } else if (id == "raise_malformed_metadata") {
    Require(source->is_ntt_form(), "REJECTION_SETUP",
            "malformed-state rejection requires a valid NTT source first");
    source->set_parms_id(phantom::parms_id_zero);
    Raise_mod(result, source, context._data_q_count);
  } else if (id == "mul_mono_undeclared") {
    const auto &resources = ResourceManifest();
    std::uint32_t undeclared = 0;
    const std::uint64_t period = 2ULL * context._poly_degree;
    while (undeclared < period &&
           std::binary_search(resources._monomial_powers,
                              resources._monomial_powers +
                                  resources._monomial_count,
                              undeclared)) {
      ++undeclared;
    }
    Require(undeclared < period, "REJECTION_SETUP",
            "resource manifest declares every normalized monomial power");
    Mul_mono_ciph(result, source, undeclared);
  }
  Fail("REJECTION_RETURNED", id + " unexpectedly succeeded");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: retained_ckks_phantom_conformance "
                   "<conformance|aliases|ownership|reject> ...\n";
      return 2;
    }
    const std::string mode = argv[1];
    if (mode == "conformance") {
      RunConformance(argc, argv);
    } else if (mode == "aliases") {
      RunAliases(argc, argv);
    } else if (mode == "ownership") {
      RunOwnership(argc, argv);
    } else if (mode == "reject") {
      RunRejection(argc, argv);
    } else {
      Fail("MODE", "unsupported mode " + mode);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
