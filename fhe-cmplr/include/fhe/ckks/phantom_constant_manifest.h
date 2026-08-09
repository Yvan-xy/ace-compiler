//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_PHANTOM_CONSTANT_MANIFEST_H
#define FHE_CKKS_PHANTOM_CONSTANT_MANIFEST_H

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhe {
namespace ckks {

inline constexpr uint32_t PHANTOM_CONSTANT_SCHEMA_VERSION = 1;

struct PHANTOM_CONSTANT_DESCRIPTOR {
  uint32_t    _entry_id = 0;
  uint64_t    _constant_id = 0;
  std::string _symbol;
  std::string _element_type = "complex_f64";
  size_t      _slot_count = 0;
  uint32_t    _ace_level = 0;
  uint32_t    _chain_index = 0;
  int32_t     _scale_degree = 0;
  double      _raw_scale = 0.0;
  std::string _raw_scale_text;
  std::string _payload_sha256;
  std::string _cache_key_sha256;
};

namespace phantom_manifest_detail {

inline uint32_t Rotate_right(uint32_t value, uint32_t count) {
  return (value >> count) | (value << (32 - count));
}

// Small dependency-free SHA-256 implementation used only for compiler
// attestations.  It keeps the generated context, payload, and cache identities
// independent of host tools and external crypto libraries.
inline std::array<uint8_t, 32> Sha256(const void* input, size_t length) {
  static constexpr uint32_t round_constants[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  std::array<uint32_t, 8> state = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  if (length > (std::numeric_limits<uint64_t>::max() / 8) ||
      (input == nullptr && length != 0)) {
    throw std::invalid_argument("invalid SHA-256 input");
  }
  const uint8_t* bytes = static_cast<const uint8_t*>(input);
  if (length > std::numeric_limits<size_t>::max() - 72) {
    throw std::overflow_error("SHA-256 input is too large");
  }
  const size_t padded_length = ((length + 9 + 63) / 64) * 64;
  std::vector<uint8_t> padded(padded_length, 0);
  if (length != 0) std::memcpy(padded.data(), bytes, length);
  padded[length] = 0x80;
  const uint64_t bit_length = static_cast<uint64_t>(length) * 8;
  for (size_t index = 0; index < 8; ++index) {
    padded[padded_length - 1 - index] =
        static_cast<uint8_t>(bit_length >> (index * 8));
  }

  for (size_t offset = 0; offset < padded_length; offset += 64) {
    uint32_t words[64] = {};
    for (size_t index = 0; index < 16; ++index) {
      const size_t byte = offset + index * 4;
      words[index] = (static_cast<uint32_t>(padded[byte]) << 24) |
                     (static_cast<uint32_t>(padded[byte + 1]) << 16) |
                     (static_cast<uint32_t>(padded[byte + 2]) << 8) |
                     static_cast<uint32_t>(padded[byte + 3]);
    }
    for (size_t index = 16; index < 64; ++index) {
      const uint32_t s0 = Rotate_right(words[index - 15], 7) ^
                          Rotate_right(words[index - 15], 18) ^
                          (words[index - 15] >> 3);
      const uint32_t s1 = Rotate_right(words[index - 2], 17) ^
                          Rotate_right(words[index - 2], 19) ^
                          (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (size_t index = 0; index < 64; ++index) {
      const uint32_t sum1 = Rotate_right(e, 6) ^ Rotate_right(e, 11) ^
                            Rotate_right(e, 25);
      const uint32_t choose = (e & f) ^ ((~e) & g);
      const uint32_t temp1 =
          h + sum1 + choose + round_constants[index] + words[index];
      const uint32_t sum0 = Rotate_right(a, 2) ^ Rotate_right(a, 13) ^
                            Rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
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

  std::array<uint8_t, 32> digest = {};
  for (size_t index = 0; index < state.size(); ++index) {
    digest[index * 4] = static_cast<uint8_t>(state[index] >> 24);
    digest[index * 4 + 1] = static_cast<uint8_t>(state[index] >> 16);
    digest[index * 4 + 2] = static_cast<uint8_t>(state[index] >> 8);
    digest[index * 4 + 3] = static_cast<uint8_t>(state[index]);
  }
  return digest;
}

inline std::string Hex(const std::array<uint8_t, 32>& digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    result[index * 2] = digits[digest[index] >> 4];
    result[index * 2 + 1] = digits[digest[index] & 0x0f];
  }
  return result;
}

}  // namespace phantom_manifest_detail

inline std::string Phantom_sha256(const void* input, size_t length) {
  return phantom_manifest_detail::Hex(
      phantom_manifest_detail::Sha256(input, length));
}

inline std::string Phantom_sha256(const std::string& input) {
  return Phantom_sha256(input.data(), input.size());
}

inline std::string Phantom_raw_scale_text(double raw_scale) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53 &&
                std::numeric_limits<double>::max_exponent == 1024,
                "Phantom constant manifests require IEEE-754 binary64");
  if (!std::isfinite(raw_scale) || raw_scale <= 0.0 ||
      std::fpclassify(raw_scale) != FP_NORMAL) {
    throw std::invalid_argument(
        "Phantom constant manifest requires a finite positive normal scale");
  }
  uint64_t representation = 0;
  static_assert(sizeof(representation) == sizeof(raw_scale));
  std::memcpy(&representation, &raw_scale, sizeof(representation));
  const uint64_t fraction = representation & ((uint64_t{1} << 52) - 1);
  const int32_t exponent =
      static_cast<int32_t>((representation >> 52) & 0x7ff) - 1023;
  std::ostringstream output;
  output << "0x1." << std::hex << std::nouppercase << std::setfill('0')
         << std::setw(13) << fraction << "p" << std::showpos << std::dec
         << exponent;
  return output.str();
}

inline std::string Build_phantom_cache_key_json(
    uint64_t constant_id, const std::string& context_manifest_sha256,
    uint32_t chain_index, const std::string& raw_scale,
    const std::string& element_type, size_t slot_count) {
  std::ostringstream output;
  output << "{\"chain_index\":" << chain_index;
  output << ",\"constant_id\":" << constant_id;
  output << ",\"context_manifest_sha256\":\""
         << context_manifest_sha256 << "\"";
  output << ",\"element_type\":\"" << element_type << "\"";
  output << ",\"raw_scale\":\"" << raw_scale << "\"";
  output << ",\"slot_count\":" << slot_count << '}';
  return output.str();
}

inline std::string Serialize_phantom_constant_manifest(
    const std::string& context_manifest_sha256,
    uint32_t context_schema_version, uint32_t resource_schema_version,
    const std::vector<PHANTOM_CONSTANT_DESCRIPTOR>& constants) {
  std::ostringstream output;
  output << "{\"constants\":[";
  for (size_t index = 0; index < constants.size(); ++index) {
    const PHANTOM_CONSTANT_DESCRIPTOR& constant = constants[index];
    if (index != 0) output << ',';
    output << "{\"ace_level\":" << constant._ace_level;
    output << ",\"cache_key_sha256\":\""
           << constant._cache_key_sha256 << "\"";
    output << ",\"chain_index\":" << constant._chain_index;
    output << ",\"constant_id\":" << constant._constant_id;
    output << ",\"element_type\":\"" << constant._element_type << "\"";
    output << ",\"entry_id\":" << constant._entry_id;
    output << ",\"payload_sha256\":\"" << constant._payload_sha256
           << "\"";
    output << ",\"raw_scale\":\"" << constant._raw_scale_text << "\"";
    output << ",\"scale_degree\":" << constant._scale_degree;
    output << ",\"slot_count\":" << constant._slot_count;
    output << ",\"symbol\":\"" << constant._symbol << "\"}";
  }
  output << "],\"context_manifest_sha256\":\""
         << context_manifest_sha256 << "\"";
  output << ",\"context_schema_version\":" << context_schema_version;
  output << ",\"resource_schema_version\":" << resource_schema_version;
  output << ",\"schema_version\":" << PHANTOM_CONSTANT_SCHEMA_VERSION
         << '}';
  return output.str();
}

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_PHANTOM_CONSTANT_MANIFEST_H
