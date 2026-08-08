// Independent ANT decoded-value oracle for the retained CKKS operations.

#include "ckks/cipher.h"
#include "ckks/ciphertext.h"
#include "ckks/plain.h"
#include "ckks/plaintext.h"
#include "common/rt_api.h"
#include "context/ckks_context.h"
#include "poly/rns_poly.h"

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
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ACE_COMMIT_ID
#error "ACE_COMMIT_ID must identify the tested ACE commit"
#endif
#ifndef PHANTOM_COMMIT_ID
#error "PHANTOM_COMMIT_ID must identify the paired Phantom commit"
#endif

CIPHERTEXT retained_ckks_composite(CIPHERTEXT input);

namespace {

using Complex = std::complex<double>;
using Json = nlohmann::json;

constexpr std::array<std::uint8_t, 8> kDecodedMagic = {'A', 'C', 'E', 'R',
                                                       'C', 'K', '0', '1'};
constexpr char kProviderSchema[] =
    "ace.phantom.retained_ckks.provider-result/3.0.0";
constexpr char kBinaryFormat[] = "ace.retained_ckks.complex_float64le/1.0.0";
constexpr std::uint32_t kResourceSchemaVersion = 2;

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error("ACE_RETAINED_ANT: " + message);
}

void Require(bool condition, const std::string &message) {
  if (!condition)
    Fail(message);
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
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4U,  0x5b9cca4fU,
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
    for (std::size_t index = 0; index < state.size(); ++index) {
      state[index] +=
          std::array<std::uint32_t, 8>{a, b, c, d, e, f, g, h}[index];
    }
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
  Require(static_cast<bool>(input), "cannot open " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff length = input.tellg();
  Require(length >= 0, "cannot size " + path);
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  if (!bytes.empty())
    input.read(reinterpret_cast<char *>(bytes.data()), length);
  Require(static_cast<bool>(input), "cannot read " + path);
  return bytes;
}

Json LoadJson(const std::string &path) {
  std::ifstream input(path);
  Require(static_cast<bool>(input), "cannot open JSON " + path);
  Json value;
  input >> value;
  Require(static_cast<bool>(input), "cannot parse JSON " + path);
  return value;
}

void WriteBytes(const std::string &path,
                const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "cannot create " + path);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.flush();
  Require(static_cast<bool>(output), "cannot write " + path);
}

void WriteJson(const std::string &path, const Json &value) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "cannot create " + path);
  output << value.dump(2) << '\n';
  output.flush();
  Require(static_cast<bool>(output), "cannot write " + path);
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
  Require(std::isfinite(value), "decoded value is non-finite");
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU64Le(output, bits);
}

double ReadDoubleLe(const std::uint8_t *input) {
  const std::uint64_t bits = ReadU64Le(input);
  double value = 0;
  std::memcpy(&value, &bits, sizeof(value));
  Require(std::isfinite(value), "analytic input is non-finite");
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

void ValidateBinary(const std::vector<std::uint8_t> &bytes,
                    const Json &descriptor) {
  Require(
      bytes.size() >= kDecodedMagic.size() &&
          std::equal(kDecodedMagic.begin(), kDecodedMagic.end(), bytes.begin()),
      "analytic binary magic mismatch");
  Require(descriptor.at("format") == kBinaryFormat,
          "analytic binary format mismatch");
  Require(descriptor.at("size_bytes").get<std::size_t>() == bytes.size() &&
              descriptor.at("sha256").get<std::string>() == Sha256(bytes),
          "analytic binary descriptor mismatch");
}

std::vector<Complex> ReadComplexBlob(const std::vector<std::uint8_t> &bytes,
                                     const Json &descriptor) {
  const std::size_t offset = descriptor.at("offset_bytes").get<std::size_t>();
  const std::size_t count = descriptor.at("count").get<std::size_t>();
  const std::size_t length = descriptor.at("byte_length").get<std::size_t>();
  Require(length == count * 16U && offset <= bytes.size() &&
              length <= bytes.size() - offset,
          "analytic input range is invalid");
  const std::vector<std::uint8_t> blob(bytes.begin() + offset,
                                       bytes.begin() + offset + length);
  Require(descriptor.at("sha256").get<std::string>() == Sha256(blob),
          "analytic input hash mismatch");
  std::vector<Complex> values;
  values.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    values.emplace_back(ReadDoubleLe(bytes.data() + offset + index * 16U),
                        ReadDoubleLe(bytes.data() + offset + index * 16U + 8U));
  }
  return values;
}

std::uint32_t AntPrimeSize(std::uint64_t prime) {
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

std::uint32_t ResolvePower(const std::string &symbol, std::uint32_t degree);

void ValidateManifest(const Json &context, const Json &resources,
                      const Json &fixture) {
  Require(context.at("schema_version") == 1 &&
              context.at("packing") == "full" &&
              context.at("resource_schema_version") == kResourceSchemaVersion &&
              resources.at("schema_version") == kResourceSchemaVersion &&
              resources.at("context_schema_version") == 1,
          "unsupported compiler manifest schema or packing");
  const std::size_t degree = context.at("polynomial_degree").get<std::size_t>();
  const std::size_t slots =
      context.at("logical_slot_capacity").get<std::size_t>();
  Require(degree >= 2 && (degree & (degree - 1U)) == 0 && slots == degree / 2U,
          "compiler context dimensions are invalid");
  Require(context.at("data_q_bit_sizes").is_array() &&
              !context.at("data_q_bit_sizes").empty() &&
              context.at("special_p_bit_sizes").is_array() &&
              !context.at("special_p_bit_sizes").empty(),
          "compiler context Q/P chains are empty");
  Require(context.at("input_level").get<std::size_t>() == 1,
          "retained modulus raise requires a compiler bottom-Q input level");
  Require(resources.at("conjugation_key").get<bool>() &&
              resources.at("rotate_batch").get<bool>() &&
              resources.at("raise_mod").get<bool>(),
          "retained resource requirements are incomplete");

  Json required_batches = Json::array({fixture.at("rotate_batch_steps")});
  for (const auto &batch : fixture.at("production_rotation_batches")) {
    required_batches.push_back(batch);
  }
  required_batches.push_back(fixture.at("rotate_batch_steps"));
  Require(resources.at("rotation_batches") == required_batches,
          "resource manifest rotation batches differ in content or order");

  const std::uint64_t monomial_period = 2U * degree;
  std::set<std::uint32_t> required_monomial_powers;
  for (const auto &raw_symbol : fixture.at("monomial_powers")) {
    required_monomial_powers.insert(
        ResolvePower(raw_symbol.get<std::string>(),
                     static_cast<std::uint32_t>(degree)) % monomial_period);
  }
  const std::set<std::uint32_t> manifest_monomial_powers =
      resources.at("monomial_powers").get<std::set<std::uint32_t>>();
  Require(manifest_monomial_powers == required_monomial_powers,
          "resource manifest does not contain every normalized retained "
          "monomial power");

  std::set<std::int32_t> required_rotation_keys;
  for (const Json &batch : required_batches) {
    for (std::int32_t step : batch.get<std::vector<std::int32_t>>()) {
      const std::int64_t reduced =
          (static_cast<std::int64_t>(step) % static_cast<std::int64_t>(slots) +
           static_cast<std::int64_t>(slots)) %
          static_cast<std::int64_t>(slots);
      const std::int32_t normalized = static_cast<std::int32_t>(
          reduced > static_cast<std::int64_t>(slots / 2U)
              ? reduced - static_cast<std::int64_t>(slots)
              : reduced);
      if (normalized != 0)
        required_rotation_keys.insert(normalized);
    }
  }
  const std::vector<std::int32_t> rotation_steps =
      resources.at("rotation_steps").get<std::vector<std::int32_t>>();
  Require(std::none_of(rotation_steps.begin(), rotation_steps.end(),
                       [](std::int32_t step) { return step == 0; }),
          "resource rotation key list contains the zero identity step");
  Require(std::set<std::int32_t>(rotation_steps.begin(), rotation_steps.end())
                  .size() == rotation_steps.size(),
          "resource rotation key list contains duplicates");
  Require(std::set<std::int32_t>(rotation_steps.begin(), rotation_steps.end()) ==
              required_rotation_keys,
          "resource rotation key list differs from normalized nonzero batch "
          "steps");
}

void VerifyPrimeChain(const Json &manifest) {
  CRT_CONTEXT *crt = Get_crt_context();
  Require(crt != nullptr, "ANT CRT context is null");
  CRT_PRIMES *q_primes = Get_q(crt);
  CRT_PRIMES *p_primes = Get_p(crt);
  const Json &expected_q = manifest.at("data_q_bit_sizes");
  const Json &expected_p = manifest.at("special_p_bit_sizes");
  Require(q_primes != nullptr && p_primes != nullptr &&
              Get_primes_cnt(q_primes) == expected_q.size() &&
              Get_primes_cnt(p_primes) == expected_p.size(),
          "ANT Q/P counts disagree with compiler context");
  for (std::size_t index = 0; index < expected_q.size(); ++index) {
    Require(AntPrimeSize(static_cast<std::uint64_t>(
                Get_modulus_val(Get_prime_at(q_primes, index)))) ==
                expected_q.at(index).get<std::uint32_t>(),
            "ANT data-Q bit size disagrees with compiler context");
  }
  for (std::size_t index = 0; index < expected_p.size(); ++index) {
    Require(AntPrimeSize(static_cast<std::uint64_t>(
                Get_modulus_val(Get_prime_at(p_primes, index)))) ==
                expected_p.at(index).get<std::uint32_t>(),
            "ANT special-P bit size disagrees with compiler context");
  }
}

std::vector<Complex> LoadSource(const Json &analytic,
                                const std::vector<std::uint8_t> &binary,
                                std::size_t slots) {
  ValidateBinary(binary, analytic.at("binary"));
  for (const auto &input : analytic.at("inputs")) {
    if (input.at("input_id") == "bounded_nonperiodic") {
      std::vector<Complex> values = ReadComplexBlob(binary, input.at("values"));
      Require(values.size() == slots,
              "analytic source differs from compiler logical slots");
      return values;
    }
  }
  Fail("analytic reference omits bounded_nonperiodic input");
}

PLAIN EncodeComplex(const std::vector<Complex> &values, std::size_t level) {
  PLAIN plain = Alloc_plaintext();
  std::vector<DCMPLX> input(values.begin(), values.end());
  Encode_dcmplx(plain, input.data(), input.size(), 1, level);
  return plain;
}

CIPHER EncryptComplex(const std::vector<Complex> &values, std::size_t level) {
  PLAIN plain = EncodeComplex(values, level);
  CIPHER cipher = Alloc_ciphertext();
  Encrypt(cipher, plain);
  Free_plaintext(plain);
  return cipher;
}

std::vector<Complex> Decode(CIPHER cipher, std::size_t expected_slots) {
  DCMPLX *raw = Get_msg_with_imag(cipher);
  Require(raw != nullptr && Get_slots(cipher) == expected_slots,
          "ANT decoded slot count mismatch");
  std::vector<Complex> values(raw, raw + expected_slots);
  std::free(raw);
  for (const Complex &value : values) {
    Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
            "ANT decoded value is non-finite");
  }
  return values;
}

std::string ResidueHash(CIPHER cipher) {
  const std::size_t degree = Get_ciph_degree(cipher);
  const std::size_t q_count = Get_ciph_prime_cnt(cipher);
  Require(Get_ciph_prime_p_cnt(cipher) == 0 &&
              Is_ntt(Get_c0(cipher)) == Is_ntt(Get_c1(cipher)),
          "ANT ciphertext has inconsistent residue metadata");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(2U * degree * q_count * sizeof(std::uint64_t));
  for (POLYNOMIAL *polynomial : {Get_c0(cipher), Get_c1(cipher)}) {
    const int64_t *coefficients = Get_poly_coeffs(polynomial);
    Require(coefficients != nullptr, "ANT ciphertext residue buffer is null");
    for (std::size_t index = 0; index < degree * q_count; ++index) {
      AppendU64Le(bytes, static_cast<std::uint64_t>(coefficients[index]));
    }
  }
  return Sha256(bytes);
}

Json Metadata(CIPHER cipher, std::size_t full_q_count) {
  const std::size_t active_q_count = Get_ciph_prime_cnt(cipher);
  Require(Get_ciph_level(cipher) == active_q_count &&
              Get_ciph_prime_p_cnt(cipher) == 0 &&
              Is_ntt(Get_c0(cipher)) == Is_ntt(Get_c1(cipher)),
          "ANT ciphertext metadata is inconsistent");
  return {{"ace_level", active_q_count},
          {"active_q_count", active_q_count},
          {"scale_degree", Get_ciph_sf_degree(cipher)},
          {"logical_slots", Get_ciph_slots(cipher)},
          {"ciphertext_size", 2},
          {"ntt", Is_ntt(Get_c0(cipher))},
          {"chain_index", full_q_count - active_q_count},
          {"raw_scale", Get_ciph_sfactor(cipher)}};
}

struct Snapshot {
  Json _metadata;
  std::string _residue_hash;
};

Snapshot TakeSnapshot(CIPHER cipher, std::size_t full_q_count) {
  return {Metadata(cipher, full_q_count), ResidueHash(cipher)};
}

void RequirePreserved(const Snapshot &before, CIPHER source,
                      std::size_t full_q_count) {
  const Snapshot after = TakeSnapshot(source, full_q_count);
  Require(before._metadata == after._metadata &&
              before._residue_hash == after._residue_hash,
          "out-of-place ANT operation mutated its source");
}

void RequireStrictQ0Tower(CIPHER full, CIPHER projected,
                          std::size_t full_q_count) {
  Require(Get_ciph_prime_cnt(full) == full_q_count &&
              Get_ciph_prime_cnt(projected) == 1 &&
              Get_ciph_degree(projected) == Get_ciph_degree(full) &&
              Get_ciph_slots(projected) == Get_ciph_slots(full) &&
              Get_ciph_sf_degree(projected) == Get_ciph_sf_degree(full) &&
              Get_ciph_sfactor(projected) == Get_ciph_sfactor(full) &&
              Is_ntt(Get_c0(projected)) == Is_ntt(Get_c0(full)) &&
              Is_ntt(Get_c1(projected)) == Is_ntt(Get_c1(full)),
          "ANT decoded projection changed non-chain metadata");
  const std::size_t bytes =
      static_cast<std::size_t>(Get_ciph_degree(full)) * sizeof(int64_t);
  for (const auto &components :
       {std::pair<POLYNOMIAL *, POLYNOMIAL *>{Get_c0(full),
                                              Get_c0(projected)},
        std::pair<POLYNOMIAL *, POLYNOMIAL *>{Get_c1(full),
                                              Get_c1(projected)}}) {
    Require(std::memcmp(Get_poly_coeffs(components.first),
                        Get_poly_coeffs(components.second), bytes) == 0,
            "ANT decoded projection changed the retained q0 tower");
  }
}

std::vector<Complex> DecodeStrictQ0(CIPHER result, std::size_t full_q_count,
                                    std::size_t slots) {
  const Snapshot before = TakeSnapshot(result, full_q_count);
  CIPHER projected = Alloc_ciphertext();
  Copy_ciph(projected, result);
  while (Get_ciph_prime_cnt(projected) > 1)
    Modswitch_ciph(projected);
  RequireStrictQ0Tower(result, projected, full_q_count);
  std::vector<Complex> values = Decode(projected, slots);
  Free_ciphertext(projected);
  const Snapshot after = TakeSnapshot(result, full_q_count);
  Require(before._metadata == after._metadata &&
              before._residue_hash == after._residue_hash,
          "ANT decoded projection mutated the full-Q result");
  return values;
}

Json AppendRecord(std::vector<std::uint8_t> &binary, const std::string &case_id,
                  const std::string &operation, CIPHER result,
                  const Snapshot &before, CIPHER source,
                  std::size_t full_q_count, std::size_t slots,
                  const Json &ownership_token = nullptr,
                  bool project_to_q0 = false) {
  const Snapshot after = TakeSnapshot(source, full_q_count);
  RequirePreserved(before, source, full_q_count);
  const std::vector<Complex> values =
      project_to_q0 ? DecodeStrictQ0(result, full_q_count, slots)
                    : Decode(result, slots);
  const Json decoded_projection =
      project_to_q0
          ? Json{{"kind", "strict_q0_prefix_drop"},
                 {"active_q_count", 1}}
          : Json(nullptr);
  return {{"case_id", case_id},
          {"operation", operation},
          {"metadata", Metadata(result, full_q_count)},
          {"source_metadata_before", before._metadata},
          {"source_metadata_after", after._metadata},
          {"source_values_sha256_before", before._residue_hash},
          {"source_values_sha256_after", after._residue_hash},
          {"ownership_token", ownership_token},
          {"decoded_projection", decoded_projection},
          {"values", AppendBlob(binary, PackComplex(values), values.size())}};
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
  Fail("unsupported symbolic monomial power " + symbol);
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
  Fail("unsupported symbolic monomial power " + symbol);
}

void FreeBatch(CIPHER outputs, std::size_t count) {
  Free_ciph_poly(outputs, static_cast<std::uint32_t>(count));
  std::free(outputs);
}

Json RunDecoded(const Json &fixture, const Json &context,
                const std::vector<Complex> &source_values,
                std::vector<std::uint8_t> &binary) {
  const std::size_t slots =
      context.at("logical_slot_capacity").get<std::size_t>();
  const std::size_t full_q_count = context.at("data_q_bit_sizes").size();
  const std::size_t input_level = context.at("input_level").get<std::size_t>();
  const std::uint32_t degree =
      context.at("polynomial_degree").get<std::uint32_t>();
  Json records = Json::array();

  {
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER result = Alloc_ciphertext();
    Conjugate_ciph(result, source);
    records.push_back(AppendRecord(binary, "conjugate.bounded_nonperiodic",
                                   "conjugate", result, before, source,
                                   full_q_count, slots));
    Free_ciphertext(result);
    Free_ciphertext(source);
  }
  {
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER first = Alloc_ciphertext();
    CIPHER result = Alloc_ciphertext();
    Conjugate_ciph(first, source);
    Conjugate_ciph(result, first);
    records.push_back(AppendRecord(
        binary, "conjugate_twice.bounded_nonperiodic", "conjugate_twice",
        result, before, source, full_q_count, slots));
    Free_ciphertext(result);
    Free_ciphertext(first);
    Free_ciphertext(source);
  }

  auto run_batch = [&](const Json &specification, const std::string &prefix) {
    std::vector<std::int32_t> steps =
        specification.get<std::vector<std::int32_t>>();
    Require(!steps.empty(), "fixture rotation batch is empty");
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER outputs =
        static_cast<CIPHER>(std::calloc(steps.size(), sizeof(CIPHERTEXT)));
    Require(outputs != nullptr, "cannot allocate ANT rotation batch");
    Rotate_batch_ciph(outputs, source, steps.data(), steps.size());
    std::set<const int64_t *> owned_buffers;
    for (std::size_t index = 0; index < steps.size(); ++index) {
      Require(
          owned_buffers.insert(Get_poly_coeffs(Get_c0(&outputs[index]))).second,
          "ANT rotation batch outputs alias storage");
      const std::string case_id = prefix + ".output_" + std::to_string(index) +
                                  ".step_" + std::to_string(steps[index]);
      records.push_back(AppendRecord(binary, case_id, "rotate_batch",
                                     &outputs[index], before, source,
                                     full_q_count, slots, "ant." + case_id));
    }
    RequirePreserved(before, source, full_q_count);
    FreeBatch(outputs, steps.size());
    Free_ciphertext(source);
  };

  run_batch(fixture.at("rotate_batch_steps"),
            "rotate_batch.bounded_nonperiodic");
  std::size_t batch_index = 0;
  for (const auto &batch : fixture.at("production_rotation_batches")) {
    run_batch(batch,
              "rotate_batch.production_" + std::to_string(batch_index++));
  }
  {
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER result = Alloc_ciphertext();
    Raise_mod(result, source, static_cast<std::uint32_t>(full_q_count));
    records.push_back(AppendRecord(binary, "raise_mod.bounded_nonperiodic",
                                   "raise_mod", result, before, source,
                                   full_q_count, slots, nullptr, true));
    Free_ciphertext(result);
    Free_ciphertext(source);
  }
  for (const auto &raw_symbol : fixture.at("monomial_powers")) {
    const std::string symbol = raw_symbol.get<std::string>();
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER result = Alloc_ciphertext();
    Mul_mono_ciph(result, source, ResolvePower(symbol, degree));
    records.push_back(AppendRecord(
        binary, "mul_mono." + PowerLabel(symbol) + ".bounded_nonperiodic",
        "mul_mono", result, before, source, full_q_count, slots));
    Free_ciphertext(result);
    Free_ciphertext(source);
  }
  {
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHER first = Alloc_ciphertext();
    CIPHER result = Alloc_ciphertext();
    Mul_mono_ciph(first, source, 2U * degree - 1U);
    Mul_mono_ciph(result, first, 1U);
    records.push_back(
        AppendRecord(binary, "mul_mono.inverse_composition.bounded_nonperiodic",
                     "mul_mono_inverse_composition", result, before, source,
                     full_q_count, slots));
    Free_ciphertext(result);
    Free_ciphertext(first);
    Free_ciphertext(source);
  }
  {
    CIPHER source = EncryptComplex(source_values, input_level);
    const Snapshot before = TakeSnapshot(source, full_q_count);
    CIPHERTEXT result = retained_ckks_composite(*source);
    records.push_back(AppendRecord(binary, "composite.bounded_nonperiodic",
                                   "composite", &result, before, source,
                                   full_q_count, slots, nullptr, true));
    Zero_ciph(&result);
    Free_ciphertext(source);
  }
  return records;
}

Json BinaryDescriptor(const std::vector<std::uint8_t> &bytes) {
  return {{"format", kBinaryFormat},
          {"size_bytes", bytes.size()},
          {"sha256", Sha256(bytes)}};
}

} // namespace

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME *Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME *Get_decode_scheme(int) { return nullptr; }
}

int main(int argc, char **argv) {
  try {
    if (argc != 9) {
      std::cerr << "usage: retained_ckks_ant_oracle <fixture.json> "
                   "<context-manifest.json> <resource-manifest.json> "
                   "<analytic.json> <analytic.bin> <output.json> <output.bin> "
                   "<disable-bootstrap-precompute-token>\n";
      return 2;
    }
    Require(std::string(argv[8]) == "disable-native-bootstrap-precompute",
            "explicit native bootstrap precompute disable token is required");
    const Json fixture = LoadJson(argv[1]);
    const Json context = LoadJson(argv[2]);
    const Json resources = LoadJson(argv[3]);
    const Json analytic = LoadJson(argv[4]);
    const auto analytic_binary = ReadBytes(argv[5]);
    const auto fixture_bytes = ReadBytes(argv[1]);
    const auto context_bytes = ReadBytes(argv[2]);
    const std::string fixture_sha256 = Sha256(fixture_bytes);
    const std::string context_sha256 = Sha256(context_bytes);
    Require(fixture.at("schema_version") ==
                    "ace.phantom.retained_ckks.fixture/2.0.0" &&
                fixture.at("qualification_bindings").at("status") == "bound",
            "fixture is not a bound retained CKKS fixture");
    Require(fixture.at("qualification_bindings")
                        .at("compiler_context_manifest_sha256") ==
                    context_sha256 &&
                analytic.at("fixture_sha256") == fixture_sha256 &&
                analytic.at("qualification_bindings") ==
                    fixture.at("qualification_bindings"),
            "fixture, context, or analytic artifact binding mismatch");
    ValidateManifest(context, resources, fixture);
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
    Prepare_context();
    VerifyPrimeChain(context);
    const auto source_values =
        LoadSource(analytic, analytic_binary,
                   context.at("logical_slot_capacity").get<std::size_t>());
    std::vector<std::uint8_t> output_binary(kDecodedMagic.begin(),
                                            kDecodedMagic.end());
    Json records = RunDecoded(fixture, context, source_values, output_binary);
    Finalize_context();
    WriteBytes(argv[7], output_binary);
    WriteJson(argv[6],
              {{"schema_version", kProviderSchema},
               {"provider", "ant"},
               {"fixture_sha256", fixture_sha256},
               {"context_manifest_sha256", context_sha256},
               {"qualification_bindings", fixture.at("qualification_bindings")},
               {"identifiers",
                {{"ace_commit", ACE_COMMIT_ID},
                 {"phantom_commit", PHANTOM_COMMIT_ID},
                 {"executable_sha256", Sha256(ReadBytes(argv[0]))}}},
               {"first_data_chain_index", 0},
               {"binary", BinaryDescriptor(output_binary)},
               {"records", std::move(records)}});
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "retained CKKS ANT oracle failed: " << error.what() << '\n';
    return 1;
  }
}
