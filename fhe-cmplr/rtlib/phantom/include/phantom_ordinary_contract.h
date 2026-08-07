//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef RTLIB_PHANTOM_ORDINARY_CONTRACT_H
#define RTLIB_PHANTOM_ORDINARY_CONTRACT_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ace::phantom::ordinary {

enum class ContractError {
  kInvalidAceLevel,
  kInvalidTargetQCount,
  kInvalidChainIndex,
  kInvalidChainLayout,
  kInvalidScaleDegree,
  kInvalidScale,
  kInvalidScalingModulusBits,
  kInvalidPolynomialDegree,
  kInvalidPackingConvention,
  kInvalidSlots,
};

enum class PackingConvention : std::uint8_t {
  kFull,
};

class ContractException final : public std::invalid_argument {
public:
  ContractException(ContractError error, const char* message)
      : std::invalid_argument(message), _error(error) {}

  ContractError Error() const noexcept { return _error; }

private:
  ContractError _error;
};

namespace detail {

inline std::size_t LastDataChainIndex(std::size_t full_data_q_count,
                                      std::size_t first_data_chain_index) {
  if (full_data_q_count == 0 ||
      first_data_chain_index >
          std::numeric_limits<std::size_t>::max() - (full_data_q_count - 1)) {
    throw ContractException(ContractError::kInvalidChainLayout,
                            "data-Q chain layout is invalid");
  }
  return first_data_chain_index + full_data_q_count - 1;
}

inline void ValidateScalingModulusBits(std::size_t scaling_modulus_bits) {
  constexpr std::size_t max_finite_power_of_two_exponent =
      static_cast<std::size_t>(std::numeric_limits<double>::max_exponent - 1);
  if (scaling_modulus_bits == 0 ||
      scaling_modulus_bits > max_finite_power_of_two_exponent) {
    throw ContractException(ContractError::kInvalidScalingModulusBits,
                            "scaling-modulus bit count is invalid");
  }
}

}  // namespace detail

inline std::size_t AceLevelToChainIndex(
    std::size_t ace_level, std::size_t full_data_q_count,
    std::size_t first_data_chain_index) {
  detail::LastDataChainIndex(full_data_q_count, first_data_chain_index);
  if (ace_level == 0 || ace_level > full_data_q_count) {
    throw ContractException(ContractError::kInvalidAceLevel,
                            "ACE logical level is outside the data-Q chain");
  }
  return first_data_chain_index + full_data_q_count - ace_level;
}

inline std::size_t TargetQCountToChainIndex(
    std::size_t target_q_count, std::size_t full_data_q_count,
    std::size_t first_data_chain_index) {
  detail::LastDataChainIndex(full_data_q_count, first_data_chain_index);
  if (target_q_count == 0 || target_q_count > full_data_q_count) {
    throw ContractException(ContractError::kInvalidTargetQCount,
                            "target data-Q count is outside the data-Q chain");
  }
  return first_data_chain_index + full_data_q_count - target_q_count;
}

inline std::size_t ChainIndexToActiveQ(
    std::size_t chain_index, std::size_t full_data_q_count,
    std::size_t first_data_chain_index) {
  const std::size_t last =
      detail::LastDataChainIndex(full_data_q_count, first_data_chain_index);
  if (chain_index < first_data_chain_index || chain_index > last) {
    throw ContractException(ContractError::kInvalidChainIndex,
                            "Phantom chain index is outside the data-Q chain");
  }
  return first_data_chain_index + full_data_q_count - chain_index;
}

inline std::size_t ChainIndexToAceLevel(
    std::size_t chain_index, std::size_t full_data_q_count,
    std::size_t first_data_chain_index) {
  return ChainIndexToActiveQ(chain_index, full_data_q_count,
                             first_data_chain_index);
}

inline std::size_t DeriveLogicalSlots(
    std::size_t polynomial_degree, PackingConvention packing_convention) {
  if (packing_convention != PackingConvention::kFull) {
    throw ContractException(ContractError::kInvalidPackingConvention,
                            "packing convention is unsupported");
  }
  if (polynomial_degree < 2 ||
      (polynomial_degree & (polynomial_degree - 1)) != 0) {
    throw ContractException(ContractError::kInvalidPolynomialDegree,
                            "polynomial degree must be a power of two");
  }
  return polynomial_degree / 2;
}

inline std::size_t ValidateLogicalSlots(
    std::size_t polynomial_degree, PackingConvention packing_convention,
    std::size_t supplied_logical_slots) {
  const std::size_t derived_slots =
      DeriveLogicalSlots(polynomial_degree, packing_convention);
  if (supplied_logical_slots != derived_slots) {
    throw ContractException(ContractError::kInvalidSlots,
                            "logical slots disagree with degree and packing");
  }
  return derived_slots;
}

inline int NormalizeRotation(std::int64_t step, std::size_t logical_slots) {
  if (logical_slots < 2 ||
      logical_slots > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw ContractException(ContractError::kInvalidSlots,
                            "logical slot count is invalid");
  }
  const std::int64_t modulus = static_cast<std::int64_t>(logical_slots);
  std::int64_t normalized = step % modulus;
  if (normalized < 0) normalized += modulus;
  const std::int64_t half_slots = modulus / 2;
  if (normalized > half_slots) normalized -= modulus;
  return static_cast<int>(normalized);
}

inline double ScaleForDegree(std::int64_t scale_degree,
                             std::size_t scaling_modulus_bits) {
  detail::ValidateScalingModulusBits(scaling_modulus_bits);
  constexpr std::int64_t max_finite_power_of_two_exponent =
      std::numeric_limits<double>::max_exponent - 1;
  if (scale_degree < 0 ||
      scale_degree > max_finite_power_of_two_exponent /
                         static_cast<std::int64_t>(scaling_modulus_bits)) {
    throw ContractException(ContractError::kInvalidScaleDegree,
                            "scale degree is outside the finite double range");
  }
  const int exponent = static_cast<int>(
      scale_degree * static_cast<std::int64_t>(scaling_modulus_bits));
  const double scale = std::ldexp(1.0, exponent);
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw ContractException(ContractError::kInvalidScale,
                            "scale is not finite and positive");
  }
  return scale;
}

// Phantom rescale divides by the exact dropped prime. Its value is close to,
// but need not equal, the configured scaling power of two. This tolerance
// accepts that deterministic drift and rejects a different ACE scale degree.
inline std::int64_t ScaleDegree(double scale,
                                std::size_t scaling_modulus_bits,
                                double degree_tolerance = 1.0e-4) {
  detail::ValidateScalingModulusBits(scaling_modulus_bits);
  if (!std::isfinite(scale) || scale <= 0.0) {
    throw ContractException(ContractError::kInvalidScale,
                            "scale is not finite and positive");
  }
  if (!std::isfinite(degree_tolerance) || degree_tolerance < 0.0) {
    throw ContractException(ContractError::kInvalidScaleDegree,
                            "scale-degree tolerance is invalid");
  }
  const double coordinate = std::log2(scale) / scaling_modulus_bits;
  const double rounded = std::nearbyint(coordinate);
  if (!std::isfinite(coordinate) || rounded < 0.0 ||
      std::fabs(coordinate - rounded) > degree_tolerance) {
    throw ContractException(ContractError::kInvalidScaleDegree,
                            "scale is not within tolerance of an ACE degree");
  }
  return static_cast<std::int64_t>(rounded);
}

}  // namespace ace::phantom::ordinary

#endif  // RTLIB_PHANTOM_ORDINARY_CONTRACT_H
