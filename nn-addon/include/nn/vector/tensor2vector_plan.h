//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_PLAN_H
#define NN_VECTOR_TENSOR2VECTOR_PLAN_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "air/base/st_decl.h"
#include "air/base/st_enum.h"

namespace nn {
namespace vector {

constexpr uint32_t VECTOR_KERNEL_PLAN_SCHEMA_VERSION = 1;

enum class VECTOR_KERNEL_PLAN_KIND : uint8_t {
  BASELINE_GEMM,
  BASELINE_CONV,
  FAST_GEMM,
  FAST_CONV,
};

enum class VECTOR_KERNEL_MASK_POLICY : uint8_t {
  NONE,
  CLEAR_VALID_PREFIX,
  COLLECTIVE_REDUCTION,
};

// ABSENT_NATIVE_BASELINE records existing native AIR faithfully. In
// particular, baseline Gemm currently has no SLOT attribute. The later kernel
// port must resolve that native behavior explicitly instead of silently
// changing the M0 oracle.
enum class VECTOR_KERNEL_SLOT_POLICY : uint8_t {
  ABSENT_NATIVE_BASELINE,
  LOGICAL_OUTPUT_ELEMENTS,
  EXPLICIT,
};

enum class VECTOR_KERNEL_REDUCTION_KIND : uint8_t {
  POWER_OF_TWO,
  LINEAR,
  COLLECTIVE_SINGLE_BLOCK,
  COLLECTIVE_BLOCKS,
};

// These records are aggregates with const data members by design. A planner
// computes mutable working data, then freezes it into one of the four plan
// variants. Production emitters must consume a const plan rather than consult
// mutable pass configuration for AIR-affecting decisions.
struct VECTOR_KERNEL_RANKED_TYPE_PLAN {
  const air::base::PRIMITIVE_TYPE _element_type;
  const std::vector<int64_t>      _shape;
};

struct VECTOR_KERNEL_CONSTANT_PLAN {
  // Stable semantic role such as "weight", "bias", "rotation-table", or
  // "cyclic-mask-left". It is independent of the generated AIR symbol name.
  const std::string                    _role;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN _type;
  // Algorithm-qualified digest of canonical element bytes, for example
  // "sha256:<lowercase hex>". AIR IDs and pointers are forbidden.
  const std::string _content_hash;
};

struct VECTOR_KERNEL_LOOP_PLAN {
  const std::string _role;
  const int32_t     _lower;
  const int32_t     _upper;
  const int32_t     _step;
  // Zero for a top-level loop in the helper, one for its direct child, etc.
  const uint32_t _nesting_depth;
};

struct VECTOR_KERNEL_AFFINE_INDEX_PLAN {
  // Coefficients follow enclosing-IV order. For example, baseline Conv uses
  // [kernel_hw, 1] for cin * kernel_hw + khw.
  const std::vector<int64_t> _iv_coefficients;
  const int64_t              _constant;
  const bool                 _uses_sharding_offset;
};

struct VECTOR_KERNEL_SLICE_PLAN {
  const std::string                     _role;
  const VECTOR_KERNEL_AFFINE_INDEX_PLAN _index;
  const int64_t                         _width;
};

struct VECTOR_KERNEL_ROTATION_PLAN {
  const std::string          _role;
  const std::vector<int32_t> _candidates;
};

struct VECTOR_KERNEL_REDUCTION_PLAN {
  const std::string                  _role;
  const VECTOR_KERNEL_REDUCTION_KIND _kind;
  const int64_t                      _factor;
  const int64_t                      _block_width;
  const int64_t                      _padding;
};

struct VECTOR_KERNEL_MASK_PLAN {
  const VECTOR_KERNEL_MASK_POLICY _policy;
  const int64_t                   _valid_length;
};

struct VECTOR_KERNEL_SLOT_PLAN {
  const VECTOR_KERNEL_SLOT_POLICY _policy;
  const uint32_t                  _value;
};

struct VECTOR_KERNEL_COMMON_PLAN {
  // ABI order is all ranked-vector inputs first, followed by Core scalar
  // inputs. Constants are same-GLOB_SCOPE LDC operands, never formals.
  const std::vector<VECTOR_KERNEL_RANKED_TYPE_PLAN> _runtime_vector_inputs;
  const std::vector<air::base::PRIMITIVE_TYPE>      _runtime_scalar_inputs;
  const std::vector<VECTOR_KERNEL_CONSTANT_PLAN>    _constants;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN              _result_type;
  const std::vector<VECTOR_KERNEL_LOOP_PLAN>        _loops;
  const std::vector<VECTOR_KERNEL_SLICE_PLAN>       _slices;
  const std::vector<VECTOR_KERNEL_ROTATION_PLAN>    _rotations;
  const std::vector<VECTOR_KERNEL_REDUCTION_PLAN>   _reductions;
  const VECTOR_KERNEL_MASK_PLAN                     _mask;
  const VECTOR_KERNEL_SLOT_PLAN                     _slot;
};

struct VECTOR_KERNEL_SHARDING_OFFSET_PLAN {
  // The source value is a runtime Core s32 expression derived from sharding
  // loop IVs. _scale is baked into the helper's slice-index expression.
  const air::base::PRIMITIVE_TYPE _type;
  const int64_t                   _scale;
};

struct BASELINE_GEMM_PLAN {
  const VECTOR_KERNEL_COMMON_PLAN _common;
  const int64_t                   _height;
  const int64_t                   _width;
  const int64_t                   _input_duplications;
};

struct BASELINE_CONV_PLAN {
  const VECTOR_KERNEL_COMMON_PLAN _common;
  const int64_t                   _channel_in;
  const int64_t                   _channel_out;
  const int64_t                   _output_height;
  const int64_t                   _output_width;
  const int64_t                   _kernel_hw;
  const int64_t                   _stride;
  const int64_t                   _input_duplications;
};

struct FAST_GEMM_PLAN {
  const VECTOR_KERNEL_COMMON_PLAN _common;
  const int64_t                   _n;
  const int64_t                   _k;
  const int64_t                   _np;
  const int64_t                   _kp;
  const int64_t                   _nd;
  const int64_t                   _kd;
  const int64_t                   _block_size;
  const int64_t                   _blocks_per_partition;
  const int64_t                   _packed_partitions;
  const int64_t                   _shift;
  const int64_t                   _shift_buffer;
  const int64_t                   _grid_size;
  const int64_t                   _input_replications;
};

struct FAST_CONV_PLAN {
  const VECTOR_KERNEL_COMMON_PLAN                  _common;
  const int64_t                                    _channel_in;
  const int64_t                                    _channel_out;
  const int64_t                                    _output_height;
  const int64_t                                    _output_width;
  const int64_t                                    _kernel_hw;
  const int64_t                                    _group;
  const int64_t                                    _stride;
  const int64_t                                    _input_size;
  const int64_t                                    _output_size;
  const int64_t                                    _num_slots;
  const int64_t                                    _num_grid;
  const int64_t                                    _num_block;
  const int64_t                                    _width_block;
  const int64_t                                    _width_block_data;
  const int64_t                                    _width_block_pad;
  const int64_t                                    _position_block;
  const int64_t                                    _capacity_block;
  const int64_t                                    _input_duplications;
  const int64_t                                    _blocking_outer_depth;
  const bool                                       _cyclic_roll;
  const std::optional<VECTOR_KERNEL_SHARDING_OFFSET_PLAN> _sharding_offset;
};

using VECTOR_KERNEL_PLAN =
    std::variant<BASELINE_GEMM_PLAN, BASELINE_CONV_PLAN, FAST_GEMM_PLAN,
                 FAST_CONV_PLAN>;

// Frozen Phase-A helper ABI. A sharded fast Conv needs the optional Core s32
// formal; it follows all packed-vector formals. Every other plan has no scalar
// formals. The helper owns its complete epilogue and has one terminal RETV.
struct VECTOR_KERNEL_HELPER_ABI {
  static constexpr bool     LEAF                       = true;
  static constexpr bool     NONRECURSIVE               = true;
  static constexpr bool     SAME_GLOB_SCOPE_CONSTANTS  = true;
  static constexpr bool     VECTOR_FORMALS_FIRST       = true;
  static constexpr bool     OPTIONAL_S32_OFFSET_LAST   = true;
  static constexpr uint32_t RESULT_COUNT               = 1;
  static constexpr uint32_t TERMINAL_RETV_COUNT        = 1;
};

VECTOR_KERNEL_PLAN_KIND Get_vector_kernel_plan_kind(
    const VECTOR_KERNEL_PLAN& plan);

const char* Vector_kernel_plan_kind_name(VECTOR_KERNEL_PLAN_KIND kind);

// Stable spelling used by plan keys and constant-hash preimages.
const char* Vector_kernel_primitive_type_name(
    air::base::PRIMITIVE_TYPE type);

// Return the lowercase hexadecimal SHA-256 digest of the supplied bytes.
std::string Vector_kernel_sha256(std::string_view bytes);

// Hash a ranked primitive constant using the canonical M0 preimage. The raw
// payload is supplied in host byte order and is converted element-by-element
// to little endian before hashing.
std::string Build_vector_kernel_constant_hash(
    air::base::PRIMITIVE_TYPE element_type,
    const std::vector<int64_t>& shape, const void* payload,
    size_t payload_byte_count);

// Convenience overload for an AIR ARRAY constant with a primitive element
// type.
std::string Build_vector_kernel_constant_hash(
    air::base::CONST_CONSTANT_PTR constant);

// Return the canonical, versioned cache key. It includes every AIR-affecting
// plan field and deliberately excludes generated IDs, source positions,
// pointer values, and symbol spelling.
std::string Build_vector_kernel_specialization_key(
    const VECTOR_KERNEL_PLAN& plan);

// Return the frozen Phase-A helper name derived solely from the canonical
// specialization key.
std::string Build_vector_kernel_helper_name(const VECTOR_KERNEL_PLAN& plan);

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_TENSOR2VECTOR_PLAN_H
