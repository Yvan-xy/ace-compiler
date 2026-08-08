//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_PHANTOM_CONTEXT_MANIFEST_H
#define FHE_CKKS_PHANTOM_CONTEXT_MANIFEST_H

#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fhe/core/scheme_info.h"

namespace fhe {
namespace ckks {

inline constexpr uint32_t PHANTOM_CONTEXT_SCHEMA_VERSION  = 1;
inline constexpr uint32_t PHANTOM_RESOURCE_SCHEMA_VERSION = 1;
inline constexpr uint32_t PHANTOM_PACKING_FULL             = 1;
inline constexpr uint32_t PHANTOM_POLY_DEGREE_MAX          = 131072;
inline constexpr uint32_t PHANTOM_USER_MODULUS_BITS_MIN    = 2;
inline constexpr uint32_t PHANTOM_USER_MODULUS_BITS_MAX    = 60;
inline constexpr uint32_t PHANTOM_COEFF_MODULUS_COUNT_MAX  = 64;

inline constexpr uint64_t PHANTOM_RESOURCE_RELIN_KEY = uint64_t{1} << 0;
inline constexpr uint64_t PHANTOM_RESOURCE_ROTATION_KEYS = uint64_t{1} << 1;

struct PHANTOM_CONTEXT_DESCRIPTOR {
  uint32_t              _schema_version = PHANTOM_CONTEXT_SCHEMA_VERSION;
  uint32_t              _packing        = PHANTOM_PACKING_FULL;
  uint32_t              _poly_degree    = 0;
  uint32_t              _logical_slots  = 0;
  std::vector<uint32_t> _data_q_bit_sizes;
  std::vector<uint32_t> _special_p_bit_sizes;
  uint32_t              _input_level       = 0;
  uint32_t              _q_part_count      = 0;
  uint32_t              _hamming_weight    = 0;
  uint32_t              _security_level    = 0;
  uint32_t              _first_modulus_bits = 0;
  uint32_t              _scaling_modulus_bits = 0;
  uint32_t _resource_schema_version = PHANTOM_RESOURCE_SCHEMA_VERSION;
};

struct PHANTOM_RESOURCE_DESCRIPTOR {
  uint32_t             _schema_version = PHANTOM_RESOURCE_SCHEMA_VERSION;
  uint32_t             _context_schema_version = PHANTOM_CONTEXT_SCHEMA_VERSION;
  uint64_t             _flags = 0;
  std::vector<int32_t> _rotation_steps;
};

inline bool Is_power_of_two(uint32_t value) {
  return value >= 2 && (value & (value - 1)) == 0;
}

inline int32_t Canonical_signed_rotation(int64_t step,
                                         uint32_t logical_slots) {
  if (logical_slots == 0) {
    throw std::invalid_argument(
        "Phantom resource manifest requires nonzero logical slots");
  }
  int64_t normalized = step % static_cast<int64_t>(logical_slots);
  if (normalized < 0) normalized += logical_slots;
  const int64_t half_slots = static_cast<int64_t>(logical_slots / 2);
  if (normalized > half_slots) normalized -= logical_slots;
  return static_cast<int32_t>(normalized);
}

inline PHANTOM_CONTEXT_DESCRIPTOR Build_phantom_context_descriptor(
    const core::CTX_PARAM& parameters) {
  PHANTOM_CONTEXT_DESCRIPTOR result;
  result._poly_degree = parameters.Get_poly_degree();
  if (!Is_power_of_two(result._poly_degree) ||
      result._poly_degree > PHANTOM_POLY_DEGREE_MAX) {
    throw std::invalid_argument(
        "Phantom context manifest requires a provider-supported power-of-two "
        "polynomial degree");
  }
  result._logical_slots = result._poly_degree / 2;

  const uint32_t data_q_count = parameters.Get_mul_level();
  result._first_modulus_bits = parameters.Get_first_prime_bit_num();
  result._scaling_modulus_bits =
      parameters.Get_scaling_factor_bit_num();
  if (data_q_count == 0 ||
      result._first_modulus_bits < PHANTOM_USER_MODULUS_BITS_MIN ||
      result._first_modulus_bits > PHANTOM_USER_MODULUS_BITS_MAX ||
      result._scaling_modulus_bits < PHANTOM_USER_MODULUS_BITS_MIN ||
      result._scaling_modulus_bits > PHANTOM_USER_MODULUS_BITS_MAX) {
    throw std::invalid_argument(
        "Phantom context manifest requires valid data-Q metadata");
  }
  result._data_q_bit_sizes.reserve(data_q_count);
  result._data_q_bit_sizes.push_back(result._first_modulus_bits);
  result._data_q_bit_sizes.insert(result._data_q_bit_sizes.end(),
                                  data_q_count - 1,
                                  result._scaling_modulus_bits);

  result._q_part_count = parameters.Get_q_part_num();
  if (result._q_part_count == 0 || result._q_part_count > data_q_count) {
    throw std::invalid_argument(
        "Phantom context manifest requires a valid Q-part count");
  }
  const uint32_t special_p_count = parameters.Get_p_prime_num();
  const uint32_t special_p_bits = parameters.Get_p_prime_bit_num();
  if (special_p_count == 0 ||
      special_p_bits < PHANTOM_USER_MODULUS_BITS_MIN ||
      special_p_bits > PHANTOM_USER_MODULUS_BITS_MAX) {
    throw std::invalid_argument(
        "Phantom context manifest requires valid special-P metadata");
  }
  if (static_cast<uint64_t>(data_q_count) + special_p_count >
      PHANTOM_COEFF_MODULUS_COUNT_MAX) {
    throw std::invalid_argument(
        "Phantom context manifest combined Q/P chain exceeds the provider "
        "modulus-count limit");
  }
  result._special_p_bit_sizes.assign(special_p_count, special_p_bits);

  result._input_level = parameters.Get_input_level();
  if (result._input_level == 0 || result._input_level > data_q_count) {
    throw std::invalid_argument(
        "Phantom context manifest input level is outside the data-Q chain");
  }
  result._hamming_weight = parameters.Get_hamming_weight();
  if (result._hamming_weight == 0 ||
      result._hamming_weight > result._poly_degree) {
    throw std::invalid_argument(
        "Phantom context manifest requires a valid secret-key hamming weight");
  }
  result._security_level = parameters.Get_security_level();
  if (result._security_level != 0 && result._security_level != 128 &&
      result._security_level != 192 && result._security_level != 256) {
    throw std::invalid_argument(
        "Phantom context manifest has an unsupported security level");
  }
  return result;
}

inline PHANTOM_RESOURCE_DESCRIPTOR Build_phantom_resource_descriptor(
    const core::CTX_PARAM& parameters,
    const PHANTOM_CONTEXT_DESCRIPTOR& context) {
  PHANTOM_RESOURCE_DESCRIPTOR result;
  if (parameters.Relin_key_required()) {
    result._flags |= PHANTOM_RESOURCE_RELIN_KEY;
  }

  std::set<int32_t> unique_steps;
  for (int32_t step : parameters.Get_rotate_index()) {
    const int32_t normalized =
        Canonical_signed_rotation(step, context._logical_slots);
    if (normalized != 0) unique_steps.insert(normalized);
  }
  result._rotation_steps.assign(unique_steps.begin(), unique_steps.end());
  if (!result._rotation_steps.empty()) {
    result._flags |= PHANTOM_RESOURCE_ROTATION_KEYS;
  }
  return result;
}

template <typename VALUE>
inline void Emit_json_array(std::ostringstream& output,
                            const std::vector<VALUE>& values) {
  output << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

inline std::string Serialize_phantom_context_descriptor(
    const PHANTOM_CONTEXT_DESCRIPTOR& context) {
  std::ostringstream output;
  // Keep this byte-for-byte identical to the evidence tools' canonical JSON:
  // UTF-8, lexicographically sorted keys, no insignificant whitespace, and no
  // trailing newline.  The file digest is therefore the context identity; no
  // second re-serialization or profile hash is needed.
  output << "{\"data_q_bit_sizes\":";
  Emit_json_array(output, context._data_q_bit_sizes);
  output << ",\"first_modulus_bits\":" << context._first_modulus_bits;
  output << ",\"hamming_weight\":" << context._hamming_weight;
  output << ",\"input_level\":" << context._input_level;
  output << ",\"logical_slot_capacity\":" << context._logical_slots;
  output << ",\"packing\":\"full\"";
  output << ",\"polynomial_degree\":" << context._poly_degree;
  output << ",\"q_part_count\":" << context._q_part_count;
  output << ",\"resource_schema_version\":"
         << context._resource_schema_version;
  output << ",\"scaling_modulus_bits\":"
         << context._scaling_modulus_bits;
  output << ",\"schema_version\":" << context._schema_version;
  output << ",\"security_level\":" << context._security_level;
  output << ",\"special_p_bit_sizes\":";
  Emit_json_array(output, context._special_p_bit_sizes);
  output << '}';
  return output.str();
}

inline std::string Serialize_phantom_resource_descriptor(
    const PHANTOM_RESOURCE_DESCRIPTOR& resources) {
  std::ostringstream output;
  output << "{\"context_schema_version\":"
         << resources._context_schema_version;
  output << ",\"relinearization_key\":"
         << ((resources._flags & PHANTOM_RESOURCE_RELIN_KEY) != 0 ? "true"
                                                                  : "false");
  output << ",\"rotation_steps\":";
  Emit_json_array(output, resources._rotation_steps);
  output << ",\"schema_version\":" << resources._schema_version << '}';
  return output.str();
}

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_PHANTOM_CONTEXT_MANIFEST_H
