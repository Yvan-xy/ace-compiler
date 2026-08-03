//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_planning.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <type_traits>

namespace nn {
namespace vector {

namespace {

using air::base::PRIMITIVE_TYPE;

constexpr size_t Provider_index(VECTOR_KERNEL_PLAN_PROVIDER_KIND kind) {
  return kind == VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP ? 0U : 1U;
}

bool Valid_operation(VECTOR_KERNEL_OPERATION operation) {
  return operation == VECTOR_KERNEL_OPERATION::GEMM ||
         operation == VECTOR_KERNEL_OPERATION::CONV;
}

bool Valid_requested_kind(VECTOR_KERNEL_REQUESTED_PLAN_KIND kind) {
  return kind == VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO ||
         kind == VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM ||
         kind == VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV ||
         kind == VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM ||
         kind == VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV;
}

bool Valid_provider_kind(VECTOR_KERNEL_PLAN_PROVIDER_KIND kind) {
  return kind == VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP ||
         kind == VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON;
}

bool Valid_implementation(VECTOR_KERNEL_IMPLEMENTATION implementation) {
  return implementation == VECTOR_KERNEL_IMPLEMENTATION::NATIVE ||
         implementation == VECTOR_KERNEL_IMPLEMENTATION::DSL;
}

bool Valid_fallback(VECTOR_KERNEL_FALLBACK_POLICY fallback) {
  return fallback == VECTOR_KERNEL_FALLBACK_POLICY::ERROR ||
         fallback == VECTOR_KERNEL_FALLBACK_POLICY::CPP_NATIVE;
}

const VECTOR_KERNEL_COMMON_PLAN& Common_plan(
    const VECTOR_KERNEL_PLAN& plan) {
  return std::visit(
      [](const auto& typed_plan) -> const VECTOR_KERNEL_COMMON_PLAN& {
        return typed_plan._common;
      },
      plan);
}

bool Equal_type(const VECTOR_KERNEL_RANKED_TYPE_PLAN& lhs,
                const VECTOR_KERNEL_RANKED_TYPE_PLAN& rhs) {
  return lhs._element_type == rhs._element_type && lhs._shape == rhs._shape;
}

std::optional<size_t> Primitive_width(PRIMITIVE_TYPE type) {
  switch (type) {
    case PRIMITIVE_TYPE::BOOL:
    case PRIMITIVE_TYPE::INT_S8:
    case PRIMITIVE_TYPE::INT_U8:
      return 1;
    case PRIMITIVE_TYPE::INT_S16:
    case PRIMITIVE_TYPE::INT_U16:
      return 2;
    case PRIMITIVE_TYPE::INT_S32:
    case PRIMITIVE_TYPE::INT_U32:
    case PRIMITIVE_TYPE::FLOAT_32:
      return 4;
    case PRIMITIVE_TYPE::INT_S64:
    case PRIMITIVE_TYPE::INT_U64:
    case PRIMITIVE_TYPE::FLOAT_64:
    case PRIMITIVE_TYPE::COMPLEX_32:
      return 8;
    case PRIMITIVE_TYPE::COMPLEX_64:
      return 16;
    case PRIMITIVE_TYPE::FLOAT_80:
    case PRIMITIVE_TYPE::FLOAT_128:
    case PRIMITIVE_TYPE::COMPLEX_80:
    case PRIMITIVE_TYPE::COMPLEX_128:
    case PRIMITIVE_TYPE::VOID:
    case PRIMITIVE_TYPE::END:
      return std::nullopt;
  }
  return std::nullopt;
}

bool Checked_mul_size(size_t lhs, size_t rhs, size_t* value) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    return false;
  }
  *value = lhs * rhs;
  return true;
}

bool Checked_mul_i64(int64_t lhs, int64_t rhs, int64_t* value) {
  if (lhs < 0 || rhs < 0) return false;
  if (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return false;
  }
  *value = lhs * rhs;
  return true;
}

bool Checked_add_i64(int64_t lhs, int64_t rhs, int64_t* value) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  *value = lhs + rhs;
  return true;
}

bool Checked_product_i64(const std::vector<int64_t>& factors,
                         int64_t* value) {
  int64_t product = 1;
  for (int64_t factor : factors) {
    if (factor <= 0 || !Checked_mul_i64(product, factor, &product)) {
      return false;
    }
  }
  *value = product;
  return true;
}

bool Fits_i32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

bool Append_i32(std::vector<int32_t>* values, int64_t value,
                std::string* diagnostic) {
  if (!Fits_i32(value)) {
    *diagnostic = "vector-kernel rotation candidate does not fit s32";
    return false;
  }
  values->push_back(static_cast<int32_t>(value));
  return true;
}

bool Equal_loop(const VECTOR_KERNEL_LOOP_PLAN& lhs,
                const VECTOR_KERNEL_LOOP_PLAN& rhs) {
  return lhs._role == rhs._role && lhs._lower == rhs._lower &&
         lhs._upper == rhs._upper && lhs._step == rhs._step &&
         lhs._nesting_depth == rhs._nesting_depth;
}

bool Equal_slice(const VECTOR_KERNEL_SLICE_PLAN& lhs,
                 const VECTOR_KERNEL_SLICE_PLAN& rhs) {
  return lhs._role == rhs._role &&
         lhs._index._iv_coefficients == rhs._index._iv_coefficients &&
         lhs._index._constant == rhs._index._constant &&
         lhs._index._uses_sharding_offset ==
             rhs._index._uses_sharding_offset &&
         lhs._width == rhs._width;
}

bool Equal_rotation(const VECTOR_KERNEL_ROTATION_PLAN& lhs,
                    const VECTOR_KERNEL_ROTATION_PLAN& rhs) {
  return lhs._role == rhs._role && lhs._candidates == rhs._candidates;
}

bool Equal_reduction(const VECTOR_KERNEL_REDUCTION_PLAN& lhs,
                     const VECTOR_KERNEL_REDUCTION_PLAN& rhs) {
  return lhs._role == rhs._role && lhs._kind == rhs._kind &&
         lhs._factor == rhs._factor &&
         lhs._block_width == rhs._block_width &&
         lhs._padding == rhs._padding;
}

template <typename RECORD, typename EQUAL>
bool Validate_exact_records(const std::vector<RECORD>& actual,
                            const std::vector<RECORD>& expected,
                            const char* label, EQUAL equal,
                            std::string* diagnostic) {
  if (actual.size() != expected.size()) {
    *diagnostic = std::string("plan ") + label +
                  " count does not match the canonical v1 schema";
    return false;
  }
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    if (!equal(actual[idx], expected[idx])) {
      *diagnostic = std::string("plan ") + label +
                    " do not match the canonical v1 schema";
      return false;
    }
  }
  return true;
}

template <typename RECORD>
bool Validate_unique_roles(const std::vector<RECORD>& records,
                           const char* label, std::string* diagnostic) {
  std::set<std::string> roles;
  for (const RECORD& record : records) {
    if (record._role.empty() || !roles.insert(record._role).second) {
      *diagnostic = std::string("plan contains an empty or duplicate ") +
                    label + " role";
      return false;
    }
  }
  return true;
}

const VECTOR_KERNEL_CONSTANT_PLAN* Find_constant_descriptor(
    const VECTOR_KERNEL_COMMON_PLAN& common, const std::string& role) {
  for (const VECTOR_KERNEL_CONSTANT_PLAN& descriptor : common._constants) {
    if (descriptor._role == role) return &descriptor;
  }
  return nullptr;
}

bool Validate_constant_schema(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    const std::vector<std::pair<std::string,
                                VECTOR_KERNEL_RANKED_TYPE_PLAN>>& expected,
    std::string* diagnostic) {
  if (common._constants.size() != expected.size()) {
    *diagnostic =
        "plan constant count does not match the canonical v1 schema";
    return false;
  }
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    const VECTOR_KERNEL_CONSTANT_PLAN& actual = common._constants[idx];
    if (actual._role != expected[idx].first ||
        !Equal_type(actual._type, expected[idx].second)) {
      *diagnostic =
          "plan constant roles/types do not match the canonical v1 schema";
      return false;
    }
  }
  return true;
}

bool Build_duplication_rotations(int64_t replications, int64_t logical_size,
                                 std::vector<int32_t>* rotations,
                                 std::string* diagnostic) {
  if (replications <= 0 || logical_size <= 0) return false;
  for (int64_t idx = 1; idx < replications; ++idx) {
    int64_t shift = 0;
    if (!Checked_mul_i64(idx, logical_size, &shift) ||
        !Append_i32(rotations, -shift, diagnostic)) {
      return false;
    }
  }
  return true;
}

bool Build_range_rotations(int64_t count, int64_t scale, int64_t start,
                           std::vector<int32_t>* rotations,
                           std::string* diagnostic) {
  if (count <= 0 || scale < 0 || start < 0 || start > count) return false;
  for (int64_t idx = start; idx < count; ++idx) {
    int64_t value = 0;
    if (!Checked_mul_i64(idx, scale, &value) ||
        !Append_i32(rotations, value, diagnostic)) {
      return false;
    }
  }
  return true;
}

bool Is_power_of_two(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

uint32_t Ceil_log2(int64_t value) {
  uint32_t result = 0;
  int64_t current = 1;
  while (current < value) {
    current *= 2;
    ++result;
  }
  return result;
}

bool Append_reduction_schema(
    const std::string& role, int64_t factor, int64_t block_width,
    int64_t padding, std::vector<VECTOR_KERNEL_ROTATION_PLAN>* rotations,
    std::vector<VECTOR_KERNEL_REDUCTION_PLAN>* reductions,
    std::string* diagnostic) {
  if (factor <= 1) return true;
  int64_t stride = 0;
  if (!Checked_add_i64(block_width, padding, &stride)) {
    *diagnostic = "vector-kernel reduction stride overflows";
    return false;
  }
  std::vector<int32_t> candidates;
  if (Is_power_of_two(factor)) {
    for (uint32_t idx = 0; idx < Ceil_log2(factor); ++idx) {
      int64_t multiplier = int64_t{1} << idx;
      int64_t value = 0;
      if (!Checked_mul_i64(multiplier, stride, &value) ||
          !Append_i32(&candidates, value, diagnostic)) {
        return false;
      }
    }
  } else {
    if (!Build_range_rotations(factor, stride, 1, &candidates, diagnostic)) {
      return false;
    }
  }
  rotations->push_back(
      VECTOR_KERNEL_ROTATION_PLAN{role, std::move(candidates)});
  reductions->push_back(VECTOR_KERNEL_REDUCTION_PLAN{
      role,
      Is_power_of_two(factor) ? VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO
                              : VECTOR_KERNEL_REDUCTION_KIND::LINEAR,
      factor, block_width, padding});
  return true;
}

bool Validate_type(const VECTOR_KERNEL_RANKED_TYPE_PLAN& type,
                   size_t* byte_count, std::string* diagnostic) {
  const std::optional<size_t> width = Primitive_width(type._element_type);
  if (!width.has_value()) {
    *diagnostic = "unsupported vector-kernel primitive type";
    return false;
  }
  if (type._shape.empty()) {
    *diagnostic = "vector-kernel ranked type must have positive rank";
    return false;
  }
  size_t count = 1;
  for (int64_t dimension : type._shape) {
    if (dimension <= 0) {
      *diagnostic = "vector-kernel ranked type has a non-positive dimension";
      return false;
    }
    if (static_cast<uint64_t>(dimension) >
        std::numeric_limits<size_t>::max()) {
      *diagnostic = "vector-kernel ranked dimension does not fit size_t";
      return false;
    }
    if (!Checked_mul_size(count, static_cast<size_t>(dimension), &count)) {
      *diagnostic = "vector-kernel ranked element count overflows";
      return false;
    }
  }
  if (!Checked_mul_size(count, *width, byte_count)) {
    *diagnostic = "vector-kernel ranked payload byte count overflows";
    return false;
  }
  return true;
}

bool Is_lower_hex(char value) {
  return (value >= '0' && value <= '9') ||
         (value >= 'a' && value <= 'f');
}

bool Valid_hash_spelling(const std::string& hash) {
  if (hash.size() != 71 || hash.compare(0, 7, "sha256:") != 0) return false;
  return std::all_of(hash.begin() + 7, hash.end(), Is_lower_hex);
}

std::string Canonical_payload_hash(
    const VECTOR_KERNEL_RANKED_TYPE_PLAN& type,
    const std::vector<uint8_t>& bytes) {
  std::ostringstream header;
  header.imbue(std::locale::classic());
  header << "vector-kernel-constant:v1\n"
         << "element=" << Vector_kernel_primitive_type_name(type._element_type)
         << '\n'
         << "rank=" << type._shape.size() << '\n'
         << "shape=";
  for (size_t idx = 0; idx < type._shape.size(); ++idx) {
    if (idx != 0) header << ',';
    header << type._shape[idx];
  }
  header << "\npayload:\n";
  std::string preimage = header.str();
  if (!bytes.empty()) {
    preimage.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  return "sha256:" + Vector_kernel_sha256(preimage);
}

const VECTOR_KERNEL_TYPED_PAYLOAD* Find_request_constant(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload :
       request._source_constants) {
    if (payload._role == role) return &payload;
  }
  return nullptr;
}

bool Request_has_attribute(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                           const std::string& name) {
  return std::any_of(
      request._attributes.begin(), request._attributes.end(),
      [&](const VECTOR_KERNEL_ATTRIBUTE_RECORD& attribute) {
        return attribute._name == name;
      });
}

const VECTOR_KERNEL_ATTRIBUTE_RECORD* Find_request_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& name) {
  for (const VECTOR_KERNEL_ATTRIBUTE_RECORD& attribute :
       request._attributes) {
    if (attribute._name == name) return &attribute;
  }
  return nullptr;
}

uint64_t Load_unsigned_little_endian(const uint8_t* bytes, size_t width) {
  uint64_t value = 0;
  for (size_t idx = 0; idx < width; ++idx) {
    value |= static_cast<uint64_t>(bytes[idx]) << (idx * 8U);
  }
  return value;
}

bool Signed_from_bits(uint64_t bits, size_t width, int64_t* value) {
  AIR_ASSERT(width == sizeof(int32_t) || width == sizeof(int64_t));
  const uint64_t sign = uint64_t{1} << (width * 8U - 1U);
  const uint64_t mask = width == sizeof(int64_t)
                            ? std::numeric_limits<uint64_t>::max()
                            : (uint64_t{1} << (width * 8U)) - 1U;
  bits &= mask;
  if ((bits & sign) == 0) {
    *value = static_cast<int64_t>(bits);
    return true;
  }
  const uint64_t magnitude = ((~bits) & mask) + 1U;
  if (magnitude == (uint64_t{1} << 63U)) {
    *value = std::numeric_limits<int64_t>::min();
  } else {
    *value = -static_cast<int64_t>(magnitude);
  }
  return true;
}

bool Decode_integer_attribute(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                              const std::string& name,
                              std::vector<int64_t>* values) {
  const VECTOR_KERNEL_ATTRIBUTE_RECORD* attribute =
      Find_request_attribute(request, name);
  if (attribute == nullptr) return false;
  size_t width = 0;
  bool is_signed = false;
  switch (attribute->_type._element_type) {
    case PRIMITIVE_TYPE::INT_S32:
      width = sizeof(int32_t);
      is_signed = true;
      break;
    case PRIMITIVE_TYPE::INT_S64:
      width = sizeof(int64_t);
      is_signed = true;
      break;
    case PRIMITIVE_TYPE::INT_U32:
      width = sizeof(uint32_t);
      break;
    default:
      return false;
  }
  if (attribute->_bytes.size() % width != 0) return false;
  values->clear();
  values->reserve(attribute->_bytes.size() / width);
  for (size_t offset = 0; offset < attribute->_bytes.size();
       offset += width) {
    const uint64_t bits =
        Load_unsigned_little_endian(attribute->_bytes.data() + offset, width);
    int64_t value = 0;
    if (is_signed) {
      if (!Signed_from_bits(bits, width, &value)) return false;
    } else {
      value = static_cast<int64_t>(bits);
    }
    values->push_back(value);
  }
  return true;
}

bool Validate_positive_u32_mask_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const char* operation_name, std::string* diagnostic) {
  if (!request._options._mask_fuse) return true;
  const VECTOR_KERNEL_ATTRIBUTE_RECORD* mask =
      Find_request_attribute(request, "mask");
  if (mask == nullptr) return true;
  if (mask->_type._element_type != PRIMITIVE_TYPE::INT_U32 ||
      mask->_type._shape != std::vector<int64_t>{1} ||
      mask->_bytes.size() != sizeof(uint32_t) ||
      Load_unsigned_little_endian(mask->_bytes.data(), sizeof(uint32_t)) ==
          0) {
    *diagnostic = std::string(operation_name) +
                  " request mask attribute must contain exactly one "
                  "positive u32";
    return false;
  }
  return true;
}

bool Required_integer_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& name,
    size_t minimum_count, std::vector<int64_t>* values,
    std::string* diagnostic) {
  if (!Decode_integer_attribute(request, name, values) ||
      values->size() < minimum_count) {
    *diagnostic = "Conv request requires integer attribute '" + name + "'";
    return false;
  }
  return true;
}

bool Optional_integer_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& name,
    std::vector<int64_t>* values, std::string* diagnostic) {
  if (Find_request_attribute(request, name) == nullptr) {
    values->clear();
    return true;
  }
  if (!Decode_integer_attribute(request, name, values)) {
    *diagnostic = "Conv request attribute '" + name +
                  "' is not a canonical integer array";
    return false;
  }
  return true;
}

struct CONV_REQUEST_FACTS {
  int64_t _source_channel_in;
  int64_t _source_kernel_channel_in;
  int64_t _channel_in;
  int64_t _kernel_channel_in;
  int64_t _channel_out;
  int64_t _height;
  int64_t _width;
  int64_t _kernel_height;
  int64_t _kernel_width;
  int64_t _kernel_hw;
  int64_t _group;
  int64_t _stride;
  int64_t _position_size;
  int64_t _input_size;
  int64_t _output_size;
  bool _sharded;
  int64_t _shard_x;
  int64_t _shard_y;
  int64_t _blocking_outer_depth;
  int64_t _sharding_offset_scale;
};

bool Build_conv_request_facts(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                              CONV_REQUEST_FACTS* facts,
                              std::string* diagnostic) {
  const VECTOR_KERNEL_RANKED_TYPE_PLAN& input = request._operand_types[0];
  const VECTOR_KERNEL_RANKED_TYPE_PLAN& weight = request._operand_types[1];
  facts->_source_channel_in = input._shape[1];
  facts->_source_kernel_channel_in = weight._shape[1];
  facts->_channel_in = facts->_source_channel_in;
  facts->_kernel_channel_in = facts->_source_kernel_channel_in;
  facts->_channel_out = weight._shape[0];
  facts->_height = input._shape[2];
  facts->_width = input._shape[3];
  facts->_kernel_height = weight._shape[2];
  facts->_kernel_width = weight._shape[3];
  if (facts->_kernel_height != facts->_kernel_width ||
      !Checked_mul_i64(facts->_kernel_height, facts->_kernel_width,
                       &facts->_kernel_hw) ||
      !Checked_mul_i64(facts->_height, facts->_width,
                       &facts->_position_size) ||
      !Checked_mul_i64(facts->_channel_out, facts->_position_size,
                       &facts->_output_size)) {
    *diagnostic = "Conv request dimensions are unsupported or overflow";
    return false;
  }

  std::vector<int64_t> group;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  if (!Required_integer_attribute(request, "group", 1, &group, diagnostic) ||
      !Required_integer_attribute(request, "strides", 1, &strides,
                                  diagnostic) ||
      !Required_integer_attribute(request, "pads", 4, &pads, diagnostic)) {
    return false;
  }
  facts->_group = group[0];
  if (facts->_group <= 0 || strides[0] <= 0 || pads[0] < 0 ||
      (strides.size() > 1 && strides[0] != strides[1])) {
    *diagnostic = "Conv request group, stride, or padding is invalid";
    return false;
  }
  facts->_stride = request._options._selective_strided_slice ? strides[0] : 1;
  std::vector<int64_t> original_strides;
  if (!Optional_integer_attribute(request, "orig_strides",
                                  &original_strides, diagnostic)) {
    return false;
  }
  if (!original_strides.empty() &&
      (original_strides.size() != 2 ||
       original_strides[0] != original_strides[1] ||
       original_strides[0] <= 0)) {
    *diagnostic = "Conv request orig_strides are invalid";
    return false;
  }
  std::vector<int64_t> keep_shape_pad;
  if (!Optional_integer_attribute(request, "keep_shape_pad",
                                  &keep_shape_pad, diagnostic)) {
    return false;
  }
  if (!keep_shape_pad.empty() && keep_shape_pad[0] < 0) {
    *diagnostic = "Conv request keep_shape_pad is invalid";
    return false;
  }
  if (facts->_group > 1) {
    if (facts->_source_channel_in != facts->_group) {
      *diagnostic =
          "depthwise Conv request requires input channels equal to group";
      return false;
    }
  } else if (facts->_source_channel_in !=
             facts->_source_kernel_channel_in) {
    *diagnostic =
        "non-grouped Conv request has mismatched input/weight channels";
    return false;
  }

  std::vector<int64_t> sharded;
  if (!Optional_integer_attribute(request, "weight-sharded", &sharded,
                                  diagnostic)) {
    return false;
  }
  facts->_sharded = !sharded.empty() && sharded[0] != 0;
  facts->_shard_x = 1;
  facts->_shard_y = 1;
  facts->_blocking_outer_depth = 0;
  facts->_sharding_offset_scale = 0;
  if (facts->_sharded) {
    std::vector<int64_t> shard;
    if (!Required_integer_attribute(request, "sharding", 4, &shard,
                                    diagnostic)) {
      return false;
    }
    const int64_t shard_z = shard[2];
    const int64_t halo = shard[3];
    int64_t doubled_halo = 0;
    if (shard[0] <= 0 || shard[1] <= 0 || shard_z <= 0 || halo < 0 ||
        !Checked_mul_i64(halo, 2, &doubled_halo) ||
        doubled_halo > facts->_height) {
      *diagnostic = "Conv request sharding dimensions are invalid";
      return false;
    }
    facts->_shard_x = shard[0];
    facts->_shard_y = shard[1];
    facts->_blocking_outer_depth = 1;
    if (shard_z == 1) {
      std::vector<int64_t> original_group;
      if (!Required_integer_attribute(request, "orig_group", 1,
                                      &original_group, diagnostic)) {
        return false;
      }
      if (original_group[0] > 1) {
        facts->_shard_y = 1;
        facts->_blocking_outer_depth = 0;
      }
    }
    if (!Checked_mul_i64(facts->_source_kernel_channel_in,
                         facts->_kernel_hw,
                         &facts->_sharding_offset_scale)) {
      *diagnostic = "Conv request sharding offset scale overflows";
      return false;
    }
  }

  if (facts->_channel_out >= facts->_channel_in &&
      facts->_channel_out % facts->_channel_in != 0) {
    while (facts->_channel_out % facts->_channel_in != 0) {
      if (facts->_channel_in == std::numeric_limits<int64_t>::max()) {
        *diagnostic = "Conv effective channel count overflows";
        return false;
      }
      ++facts->_channel_in;
    }
    facts->_kernel_channel_in = facts->_channel_in;
  }
  if (!Checked_mul_i64(facts->_channel_in, facts->_position_size,
                       &facts->_input_size)) {
    *diagnostic = "Conv request effective input size overflows";
    return false;
  }

  if (facts->_sharded) {
    if (request._runtime_scalar_types !=
        std::vector<PRIMITIVE_TYPE>{PRIMITIVE_TYPE::INT_S32}) {
      *diagnostic = "sharded Conv request requires one s32 weight offset";
      return false;
    }
  } else if (!request._runtime_scalar_types.empty()) {
    *diagnostic = "unsharded Conv request cannot contain runtime scalars";
    return false;
  }
  return true;
}

bool Checked_add_signed_i64(int64_t lhs, int64_t rhs, int64_t* value) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return false;
  }
  *value = lhs + rhs;
  return true;
}

bool Checked_mul_signed_i64(int64_t lhs, int64_t rhs, int64_t* value) {
  if (lhs == 0 || rhs == 0) {
    *value = 0;
    return true;
  }
  if ((lhs == -1 && rhs == std::numeric_limits<int64_t>::min()) ||
      (rhs == -1 && lhs == std::numeric_limits<int64_t>::min())) {
    return false;
  }
  if (lhs > 0) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
        (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs)) {
      return false;
    }
  } else if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
             (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs)) {
    return false;
  }
  *value = lhs * rhs;
  return true;
}

bool Build_conv_alignment_rotations(const CONV_REQUEST_FACTS& facts,
                                    int64_t scale,
                                    std::vector<int32_t>* result,
                                    std::string* diagnostic) {
  if (scale <= 0 ||
      static_cast<uint64_t>(facts._kernel_hw) >
          std::numeric_limits<size_t>::max()) {
    *diagnostic = "Conv rotation table dimensions are invalid";
    return false;
  }
  std::vector<int64_t> rotations(static_cast<size_t>(facts._kernel_hw), 0);
  const int64_t pad = (facts._kernel_height - 1) / 2;
  int64_t edge = 0;
  if (!Checked_mul_signed_i64(facts._height, pad, &edge) ||
      !Checked_add_signed_i64(edge, pad, &edge)) {
    *diagnostic = "Conv rotation alignment overflows";
    return false;
  }
  rotations[0] = -edge;
  rotations[static_cast<size_t>(facts._kernel_hw - 1)] = edge;
  rotations[static_cast<size_t>(facts._kernel_hw / 2)] = 0;
  if (facts._kernel_height > 1) {
    rotations[static_cast<size_t>(facts._kernel_hw / 2 + 1)] = 1;
    rotations[static_cast<size_t>(facts._kernel_hw / 2 - 1)] = -1;
  }
  for (int64_t idx = 1; idx < facts._kernel_hw / 2 - 1; ++idx) {
    int64_t value = 0;
    if (!Checked_add_signed_i64(rotations[static_cast<size_t>(idx - 1)], 1,
                                &value) ||
        (facts._kernel_height > 3 && idx % facts._kernel_height == 0 &&
         !Checked_add_signed_i64(
             value, facts._height - facts._kernel_height, &value))) {
      *diagnostic = "Conv rotation alignment overflows";
      return false;
    }
    rotations[static_cast<size_t>(idx)] = value;
  }
  for (int64_t idx = facts._kernel_hw - 1;
       idx > facts._kernel_hw / 2 + 2; --idx) {
    int64_t value = 0;
    if (!Checked_add_signed_i64(rotations[static_cast<size_t>(idx)], -1,
                                &value) ||
        (facts._kernel_height > 3 && idx % facts._kernel_height == 0 &&
         !Checked_add_signed_i64(
             value, facts._kernel_height - facts._height, &value))) {
      *diagnostic = "Conv rotation alignment overflows";
      return false;
    }
    rotations[static_cast<size_t>(idx - 1)] = value;
  }
  if (facts._height != facts._width) {
    for (int64_t row = 0; row < facts._kernel_height; ++row) {
      for (int64_t column = 0; column < facts._kernel_width; ++column) {
        int64_t value = 0;
        if (!Checked_mul_signed_i64(
                facts._width, row - (facts._kernel_height - 1) / 2,
                &value) ||
            !Checked_add_signed_i64(
                value, column - (facts._kernel_height - 1) / 2, &value)) {
          *diagnostic = "Conv rotation alignment overflows";
          return false;
        }
        rotations[static_cast<size_t>(row * facts._kernel_height + column)] =
            value;
      }
    }
  }
  result->clear();
  result->reserve(rotations.size());
  for (int64_t rotation : rotations) {
    int64_t scaled = 0;
    if (!Checked_mul_signed_i64(rotation, scale, &scaled) ||
        !Append_i32(result, scaled, diagnostic)) {
      return false;
    }
  }
  return true;
}

int64_t Conv_mini_factor(int64_t channels) {
  if (channels < 32 || !Is_power_of_two(channels)) return 1;
  int64_t minimum_sum = channels;
  int64_t factor = 1;
  for (int64_t candidate = 2; candidate <= channels / candidate;
       ++candidate) {
    if (channels % candidate == 0 &&
        candidate + channels / candidate < minimum_sum) {
      minimum_sum = candidate + channels / candidate;
      factor = candidate;
    }
  }
  return factor;
}

int64_t Conv_fusion_blocks(const CONV_REQUEST_FACTS& facts,
                           int64_t num_slots) {
  int64_t blocks = 1;
  if (Is_power_of_two(facts._output_size) &&
      facts._channel_out % facts._channel_in == 0 &&
      facts._kernel_hw > 1 && facts._group == 1 &&
      num_slots / facts._output_size >= 4 &&
      facts._channel_out >= facts._channel_in) {
    blocks = num_slots / facts._output_size / 2;
    blocks = std::min(blocks, facts._channel_in);
    if (blocks >= 2 &&
        facts._channel_in / blocks + 2 * blocks >= facts._channel_in) {
      blocks /= 2;
    }
    blocks = std::min<int64_t>(blocks, 8);
  }
  return blocks;
}

struct FAST_CONV_EXPECTED_TOPOLOGY {
  int64_t _num_grid;
  int64_t _num_block;
  int64_t _width_block;
  int64_t _width_block_pad;
  int64_t _position_block;
  int64_t _capacity_block;
  int64_t _input_duplications;
  int64_t _weight_rows;
  int64_t _weight_columns;
  std::vector<int32_t> _blocking_rotations;
};

bool Build_fast_conv_expected_topology(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const CONV_REQUEST_FACTS& facts, FAST_CONV_EXPECTED_TOPOLOGY* expected,
    std::string* diagnostic) {
  if (facts._channel_out < facts._channel_in) {
    *diagnostic = "fast Conv requires output channels >= input channels";
    return false;
  }
  expected->_num_block = request._options._conv_parallel
                             ? Conv_fusion_blocks(facts,
                                                  request._target._num_slots)
                             : 1;
  if (expected->_num_block <= 0 ||
      facts._kernel_channel_in % expected->_num_block != 0) {
    *diagnostic = "fast Conv block count does not divide request channels";
    return false;
  }
  expected->_width_block_pad =
      expected->_num_block == 1
          ? 0
          : facts._input_size / expected->_num_block;
  expected->_num_grid = facts._kernel_channel_in;
  expected->_capacity_block = facts._kernel_hw;
  expected->_position_block = facts._position_size;
  expected->_input_duplications = facts._channel_out / facts._channel_in;
  if (!Build_conv_alignment_rotations(facts, 1,
                                      &expected->_blocking_rotations,
                                      diagnostic)) {
    return false;
  }

  if (facts._sharded) {
    int64_t shard_count = 0;
    if (!Checked_mul_i64(facts._shard_x, facts._shard_y, &shard_count) ||
        !Checked_product_i64({facts._source_kernel_channel_in,
                              facts._kernel_hw, shard_count},
                             &expected->_weight_rows)) {
      *diagnostic = "fast Conv sharded weight dimensions overflow";
      return false;
    }
    expected->_weight_columns = facts._output_size;
  } else if (facts._kernel_hw == 1) {
    expected->_capacity_block =
        Conv_mini_factor(facts._kernel_channel_in);
    if (expected->_capacity_block <= 0 ||
        facts._kernel_channel_in % expected->_capacity_block != 0) {
      *diagnostic = "fast Conv 1x1 capacity is inconsistent";
      return false;
    }
    for (int64_t idx = 1; idx < expected->_capacity_block; ++idx) {
      int64_t rotation = 0;
      if (!Checked_mul_i64(idx, facts._position_size, &rotation) ||
          !Append_i32(&expected->_blocking_rotations, rotation,
                      diagnostic)) {
        return false;
      }
    }
    if (expected->_capacity_block > 1) {
      if (!Checked_add_i64(expected->_input_duplications, 1,
                           &expected->_input_duplications)) {
        *diagnostic = "fast Conv input duplication count overflows";
        return false;
      }
    }
    expected->_num_grid /= expected->_capacity_block;
    if (!Checked_mul_i64(expected->_position_block,
                         expected->_capacity_block,
                         &expected->_position_block)) {
      *diagnostic = "fast Conv position block overflows";
      return false;
    }
    expected->_weight_rows = facts._kernel_channel_in;
    expected->_weight_columns = facts._output_size;
  } else {
    if (expected->_num_block > 1 &&
        (!Checked_add_i64(expected->_num_block, 1,
                          &expected->_input_duplications) ||
         !Checked_mul_i64(facts._channel_out / facts._channel_in,
                          expected->_input_duplications,
                          &expected->_input_duplications))) {
      *diagnostic = "fast Conv input duplication count overflows";
      return false;
    }
    expected->_num_grid /= expected->_num_block;
    if (!Checked_mul_i64(facts._kernel_channel_in, facts._kernel_hw,
                         &expected->_weight_rows)) {
      *diagnostic = "fast Conv weight row count overflows";
      return false;
    }
    expected->_weight_rows /= expected->_num_block;
  }
  if (!Checked_add_i64(facts._output_size, expected->_width_block_pad,
                       &expected->_width_block)) {
    *diagnostic = "fast Conv width block overflows";
    return false;
  }
  int64_t slice_width = 0;
  if (!Checked_mul_i64(expected->_num_block, expected->_width_block,
                       &slice_width)) {
    *diagnostic = "fast Conv slice width overflows";
    return false;
  }
  if (!facts._sharded && facts._kernel_hw != 1) {
    expected->_weight_columns = slice_width;
  }
  if (expected->_blocking_rotations.size() !=
      static_cast<size_t>(expected->_capacity_block)) {
    *diagnostic = "fast Conv rotation count is inconsistent";
    return false;
  }
  return true;
}

bool Validate_request_payload(const VECTOR_KERNEL_TYPED_PAYLOAD& payload,
                              std::string* diagnostic) {
  if (payload._role.empty()) {
    *diagnostic = "vector-kernel request contains an empty constant role";
    return false;
  }
  size_t expected_bytes = 0;
  if (!Validate_type(payload._type, &expected_bytes, diagnostic)) {
    return false;
  }
  if (payload._bytes.size() != expected_bytes) {
    *diagnostic = "vector-kernel request constant byte count mismatch";
    return false;
  }
  if (!Valid_hash_spelling(payload._content_hash) ||
      payload._content_hash !=
          Canonical_payload_hash(payload._type, payload._bytes)) {
    *diagnostic = "vector-kernel request constant hash mismatch";
    return false;
  }
  return true;
}

bool Validate_planning_request(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                               std::string* diagnostic) {
  if (!Valid_operation(request._operation) ||
      !Valid_requested_kind(request._requested_plan_kind)) {
    *diagnostic = "invalid vector-kernel request operation or plan kind";
    return false;
  }
  if (request._target._min_slots <= 0 ||
      request._target._max_slots < request._target._min_slots ||
      !Fits_i32(request._target._num_slots) ||
      request._target._num_slots < request._target._min_slots ||
      request._target._num_slots > request._target._max_slots) {
    *diagnostic = "invalid vector-kernel target snapshot";
    return false;
  }
  if (request._operand_types.size() != 3) {
    *diagnostic = "vector-kernel request requires exactly three operands";
    return false;
  }
  size_t ignored = 0;
  for (const VECTOR_KERNEL_RANKED_TYPE_PLAN& type : request._operand_types) {
    if (!Validate_type(type, &ignored, diagnostic)) return false;
  }
  if (!Validate_type(request._declared_result_type, &ignored, diagnostic)) {
    return false;
  }

  std::set<std::string> attribute_names;
  for (const VECTOR_KERNEL_ATTRIBUTE_RECORD& attribute :
       request._attributes) {
    if (attribute._name.empty() ||
        !attribute_names.insert(attribute._name).second) {
      *diagnostic =
          "vector-kernel request contains an empty or duplicate attribute";
      return false;
    }
    size_t expected_bytes = 0;
    if (!Validate_type(attribute._type, &expected_bytes, diagnostic)) {
      return false;
    }
    if (attribute._bytes.size() != expected_bytes) {
      *diagnostic = "vector-kernel request attribute byte count mismatch";
      return false;
    }
  }

  if (request._source_constants.size() != 2 ||
      request._source_constants[0]._role != "weight" ||
      request._source_constants[1]._role != "bias") {
    *diagnostic =
        "vector-kernel request must own canonical weight and bias roles";
    return false;
  }
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload :
       request._source_constants) {
    if (!Validate_request_payload(payload, diagnostic)) return false;
  }
  const VECTOR_KERNEL_TYPED_PAYLOAD* weight =
      Find_request_constant(request, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD* bias =
      Find_request_constant(request, "bias");
  if (weight == nullptr || bias == nullptr) {
    *diagnostic = "vector-kernel request is missing weight or bias";
    return false;
  }

  const auto all_f32 = [](const VECTOR_KERNEL_RANKED_TYPE_PLAN& type) {
    return type._element_type == PRIMITIVE_TYPE::FLOAT_32;
  };
  if (!std::all_of(request._operand_types.begin(),
                   request._operand_types.end(), all_f32) ||
      !all_f32(request._declared_result_type) || !all_f32(weight->_type) ||
      !all_f32(bias->_type)) {
    *diagnostic = "v1 Conv/Gemm planning requires f32 ranked values";
    return false;
  }

  if (request._operation == VECTOR_KERNEL_OPERATION::GEMM) {
    if (!Validate_positive_u32_mask_attribute(request, "Gemm", diagnostic)) {
      return false;
    }
    if (!request._runtime_scalar_types.empty()) {
      *diagnostic = "Gemm request cannot contain runtime scalar inputs";
      return false;
    }
    if (weight->_type._shape.size() != 2 ||
        bias->_type._shape.size() != 1 ||
        !Equal_type(weight->_type, request._operand_types[1]) ||
        !Equal_type(bias->_type, request._operand_types[2])) {
      *diagnostic = "Gemm request has incompatible weight or bias types";
      return false;
    }
    int64_t input_count = 0;
    int64_t result_count = 0;
    if (!Checked_product_i64(request._operand_types[0]._shape, &input_count) ||
        !Checked_product_i64(request._declared_result_type._shape,
                             &result_count) ||
        input_count != weight->_type._shape[1] ||
        result_count != weight->_type._shape[0] ||
        bias->_type._shape[0] != weight->_type._shape[0]) {
      *diagnostic = "Gemm request dimensions are inconsistent";
      return false;
    }
  } else {
    if (!Validate_positive_u32_mask_attribute(request, "Conv", diagnostic)) {
      return false;
    }
    const VECTOR_KERNEL_RANKED_TYPE_PLAN& input = request._operand_types[0];
    const VECTOR_KERNEL_RANKED_TYPE_PLAN& weight_operand =
        request._operand_types[1];
    if (input._shape.size() != 4 || weight_operand._shape.size() != 4 ||
        request._operand_types[2]._shape.size() != 1 ||
        request._declared_result_type._shape.size() != 4 ||
        input._shape[0] != 1 || request._declared_result_type._shape[0] != 1 ||
        !Equal_type(bias->_type, request._operand_types[2]) ||
        (weight->_type._shape.size() != 4 &&
         weight->_type._shape.size() != 5)) {
      *diagnostic = "Conv request requires compatible NCHW operand types";
      return false;
    }
    const size_t weight_offset = weight->_type._shape.size() - 4;
    if (!std::equal(weight_operand._shape.begin(), weight_operand._shape.end(),
                    weight->_type._shape.begin() + weight_offset) ||
        request._declared_result_type._shape[1] != weight_operand._shape[0] ||
        request._declared_result_type._shape[2] != input._shape[2] ||
        request._declared_result_type._shape[3] != input._shape[3] ||
        bias->_type._shape[0] != weight_operand._shape[0]) {
      *diagnostic = "Conv request dimensions are inconsistent";
      return false;
    }
    CONV_REQUEST_FACTS facts{};
    if (!Build_conv_request_facts(request, &facts, diagnostic)) return false;
    const size_t expected_weight_rank = facts._sharded ? 5U : 4U;
    if (weight->_type._shape.size() != expected_weight_rank) {
      *diagnostic = facts._sharded
                        ? "sharded Conv request requires a ranked-5 weight"
                        : "unsharded Conv request requires a ranked-4 weight";
      return false;
    }
    const bool uses_fast =
        request._requested_plan_kind ==
            VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV ||
        (request._requested_plan_kind ==
             VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO &&
         facts._channel_out >= facts._channel_in);
    if (uses_fast && facts._sharded && request._options._conv_parallel &&
        Conv_fusion_blocks(facts, request._target._num_slots) > 1) {
      *diagnostic =
          "sharded fast Conv does not support multiple parallel fusion "
          "blocks";
      return false;
    }
  }
  return true;
}

VECTOR_KERNEL_PLAN_KIND Requested_to_plan_kind(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind) {
  switch (kind) {
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM:
      return VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV:
      return VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV;
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM:
      return VECTOR_KERNEL_PLAN_KIND::FAST_GEMM;
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV:
      return VECTOR_KERNEL_PLAN_KIND::FAST_CONV;
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO:
      break;
  }
  return VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
}

bool Kind_matches_operation(VECTOR_KERNEL_OPERATION operation,
                            VECTOR_KERNEL_PLAN_KIND kind) {
  if (operation == VECTOR_KERNEL_OPERATION::GEMM) {
    return kind == VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM ||
           kind == VECTOR_KERNEL_PLAN_KIND::FAST_GEMM;
  }
  return kind == VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV ||
         kind == VECTOR_KERNEL_PLAN_KIND::FAST_CONV;
}

const VECTOR_KERNEL_ROTATION_PLAN* Find_rotation(
    const VECTOR_KERNEL_COMMON_PLAN& common, const std::string& role) {
  for (const VECTOR_KERNEL_ROTATION_PLAN& rotation : common._rotations) {
    if (rotation._role == role) return &rotation;
  }
  return nullptr;
}

bool Equal_runtime_preparation(
    const VECTOR_KERNEL_RUNTIME_PREPARATION& lhs,
    const VECTOR_KERNEL_RUNTIME_PREPARATION& rhs) {
  return lhs._role == rhs._role &&
         lhs._source_operand == rhs._source_operand &&
         lhs._kind == rhs._kind &&
         Equal_type(lhs._result_type, rhs._result_type) &&
         lhs._logical_input_size == rhs._logical_input_size &&
         lhs._replications == rhs._replications &&
         lhs._blocking_width == rhs._blocking_width &&
         lhs._rotation_candidates == rhs._rotation_candidates &&
         lhs._outer_block_depth == rhs._outer_block_depth;
}

bool Equal_scalar_preparation(
    const VECTOR_KERNEL_SCALAR_PREPARATION& lhs,
    const VECTOR_KERNEL_SCALAR_PREPARATION& rhs) {
  return lhs._role == rhs._role &&
         lhs._source_operand == rhs._source_operand &&
         lhs._type == rhs._type && lhs._scale == rhs._scale;
}

std::vector<int32_t> Rotation_candidates(
    const VECTOR_KERNEL_COMMON_PLAN& common, const std::string& role) {
  const VECTOR_KERNEL_ROTATION_PLAN* rotation = Find_rotation(common, role);
  return rotation == nullptr ? std::vector<int32_t>{}
                             : rotation->_candidates;
}

bool Build_expected_preparations(
    const VECTOR_KERNEL_PLAN& plan,
    std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION>* runtime,
    std::vector<VECTOR_KERNEL_SCALAR_PREPARATION>* scalars,
    std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = Common_plan(plan);
  if (common._runtime_vector_inputs.size() != 1) {
    *diagnostic = "v1 vector-kernel plan requires exactly one vector input";
    return false;
  }

  return std::visit(
      [&](const auto& typed_plan) {
        using PLAN = std::decay_t<decltype(typed_plan)>;
        VECTOR_KERNEL_RUNTIME_PREPARATION_KIND prep_kind =
            VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::PACKED_VECTOR;
        int64_t logical_size = 0;
        int64_t replications = 0;
        int64_t blocking_width = 0;
        uint32_t outer_depth = 0;
        std::vector<int32_t> rotations;

        if constexpr (std::is_same_v<PLAN, BASELINE_GEMM_PLAN>) {
          logical_size = typed_plan._width;
          replications = typed_plan._input_duplications;
          rotations = Rotation_candidates(common, "input-duplication");
        } else if constexpr (std::is_same_v<PLAN, BASELINE_CONV_PLAN>) {
          prep_kind = VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::
              FLATTEN_PACKED_VECTOR;
          if (!Checked_mul_i64(typed_plan._channel_in,
                               typed_plan._output_height, &logical_size) ||
              !Checked_mul_i64(logical_size, typed_plan._output_width,
                               &logical_size)) {
            *diagnostic = "baseline Conv input size overflows";
            return false;
          }
          replications = typed_plan._input_duplications;
          rotations = Rotation_candidates(common, "input-duplication");
        } else if constexpr (std::is_same_v<PLAN, FAST_GEMM_PLAN>) {
          prep_kind = VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::
              BLOCKING_ROTATIONS;
          logical_size = typed_plan._kp;
          replications = typed_plan._input_replications;
          blocking_width = typed_plan._block_size;
          rotations = Rotation_candidates(common, "blocking-alignment");
        } else if constexpr (std::is_same_v<PLAN, FAST_CONV_PLAN>) {
          prep_kind = VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::
              BLOCKING_ROTATIONS;
          logical_size = typed_plan._input_size;
          replications = typed_plan._input_duplications;
          blocking_width = typed_plan._capacity_block;
          outer_depth = static_cast<uint32_t>(typed_plan._blocking_outer_depth);
          rotations = Rotation_candidates(common, "blocking-alignment");
          if (typed_plan._sharding_offset.has_value()) {
            scalars->push_back(VECTOR_KERNEL_SCALAR_PREPARATION{
                "weight-offset", 1,
                typed_plan._sharding_offset->_type,
                typed_plan._sharding_offset->_scale});
          }
        }

        runtime->push_back(VECTOR_KERNEL_RUNTIME_PREPARATION{
            "input", 0, prep_kind, common._runtime_vector_inputs[0],
            logical_size, replications, blocking_width, std::move(rotations),
            outer_depth});
        return true;
      },
      plan);
}

bool Validate_common_plan(const VECTOR_KERNEL_COMMON_PLAN& common,
                          std::string* diagnostic) {
  if (common._runtime_vector_inputs.size() != 1) {
    *diagnostic = "v1 plan must contain exactly one runtime vector input";
    return false;
  }
  size_t ignored = 0;
  if (!Validate_type(common._runtime_vector_inputs[0], &ignored,
                     diagnostic) ||
      !Validate_type(common._result_type, &ignored, diagnostic)) {
    return false;
  }
  if (!common._runtime_scalar_inputs.empty() &&
      common._runtime_scalar_inputs !=
          std::vector<PRIMITIVE_TYPE>{PRIMITIVE_TYPE::INT_S32}) {
    *diagnostic = "plan contains an unsupported scalar ABI";
    return false;
  }
  if (!Validate_unique_roles(common._constants, "constant", diagnostic) ||
      !Validate_unique_roles(common._loops, "loop", diagnostic) ||
      !Validate_unique_roles(common._slices, "slice", diagnostic) ||
      !Validate_unique_roles(common._rotations, "rotation", diagnostic) ||
      !Validate_unique_roles(common._reductions, "reduction", diagnostic)) {
    return false;
  }
  for (const VECTOR_KERNEL_CONSTANT_PLAN& constant : common._constants) {
    if (!Validate_type(constant._type, &ignored, diagnostic)) return false;
    if (!Valid_hash_spelling(constant._content_hash)) {
      *diagnostic = "plan constant hash has invalid canonical spelling";
      return false;
    }
  }
  uint32_t previous_depth = 0;
  for (size_t idx = 0; idx < common._loops.size(); ++idx) {
    const VECTOR_KERNEL_LOOP_PLAN& loop = common._loops[idx];
    if (loop._step != 1 || loop._lower < 0 ||
        loop._upper <= loop._lower || loop._nesting_depth > 1) {
      *diagnostic = "plan contains an invalid loop descriptor";
      return false;
    }
    if (idx == 0 && loop._nesting_depth != 0) {
      *diagnostic = "plan's first loop is not top-level";
      return false;
    }
    if (idx != 0 && loop._nesting_depth > previous_depth + 1) {
      *diagnostic = "plan loop nesting skips a depth";
      return false;
    }
    previous_depth = loop._nesting_depth;
  }
  for (const VECTOR_KERNEL_SLICE_PLAN& slice : common._slices) {
    if (slice._width <= 0 ||
        slice._index._iv_coefficients.size() > common._loops.size()) {
      *diagnostic = "plan contains an invalid slice descriptor";
      return false;
    }
  }
  for (const VECTOR_KERNEL_ROTATION_PLAN& rotation : common._rotations) {
    if (rotation._candidates.empty()) {
      *diagnostic = "plan contains an invalid rotation descriptor";
      return false;
    }
  }
  for (const VECTOR_KERNEL_REDUCTION_PLAN& reduction : common._reductions) {
    if (reduction._factor <= 0 || reduction._block_width <= 0 ||
        reduction._padding < 0 ||
        (reduction._kind != VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO &&
         reduction._kind != VECTOR_KERNEL_REDUCTION_KIND::LINEAR &&
         reduction._kind !=
             VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK &&
         reduction._kind != VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS)) {
      *diagnostic = "plan contains an invalid reduction descriptor";
      return false;
    }
  }
  if (common._mask._policy != VECTOR_KERNEL_MASK_POLICY::NONE &&
      common._mask._policy != VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX &&
      common._mask._policy !=
          VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION) {
    *diagnostic = "plan contains an invalid mask policy";
    return false;
  } else if (common._mask._policy == VECTOR_KERNEL_MASK_POLICY::NONE) {
    if (common._mask._valid_length != 0) {
      *diagnostic = "none mask policy must have zero valid length";
      return false;
    }
  } else if (common._mask._valid_length <= 0) {
    *diagnostic = "active mask policy must have positive valid length";
    return false;
  }
  if (common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE &&
      common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS &&
      common._slot._policy != VECTOR_KERNEL_SLOT_POLICY::EXPLICIT) {
    *diagnostic = "plan contains an invalid slot policy";
    return false;
  }
  if (common._slot._policy ==
          VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE &&
      common._slot._value != 0) {
    *diagnostic = "absent slot policy must have value zero";
    return false;
  }
  if (common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE &&
      common._slot._value == 0) {
    *diagnostic = "present slot policy must have a nonzero value";
    return false;
  }
  return true;
}

bool Validate_runtime_input(const VECTOR_KERNEL_COMMON_PLAN& common,
                            const VECTOR_KERNEL_PLANNING_REQUEST& request,
                            std::string* diagnostic) {
  if (request._operation == VECTOR_KERNEL_OPERATION::GEMM) {
    if (Equal_type(common._runtime_vector_inputs[0],
                   request._operand_types[0])) {
      return true;
    }
    *diagnostic = "plan runtime input type does not match the request";
    return false;
  }
  int64_t request_input_count = 0;
  if (!Checked_product_i64(request._operand_types[0]._shape,
                           &request_input_count)) {
    *diagnostic = "vector-kernel request input size overflows";
    return false;
  }
  const VECTOR_KERNEL_RANKED_TYPE_PLAN expected{
      request._operand_types[0]._element_type, {request_input_count}};
  if (!Equal_type(common._runtime_vector_inputs[0], expected)) {
    *diagnostic = "plan runtime input type does not match the request";
    return false;
  }
  return true;
}

bool Validate_baseline_gemm_plan(
    const BASELINE_GEMM_PLAN& plan,
    const VECTOR_KERNEL_PLANNING_REQUEST& request, std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_weight =
      Find_request_constant(request, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_bias =
      Find_request_constant(request, "bias");
  int64_t result_width = 0;
  if (source_weight == nullptr || source_bias == nullptr ||
      plan._height <= 0 || plan._width <= 0 ||
      !Fits_i32(plan._height) || !Fits_i32(plan._width) ||
      plan._input_duplications <= 0 || plan._width % plan._height != 0 ||
      plan._height < source_weight->_type._shape[0] ||
      plan._width < source_weight->_type._shape[1] ||
      plan._input_duplications !=
          (plan._width == request._target._num_slots ? 1 : 2) ||
      !Checked_mul_i64(plan._width, plan._input_duplications,
                       &result_width) ||
      result_width > request._target._num_slots ||
      !common._runtime_scalar_inputs.empty() ||
      common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE ||
      !Equal_type(common._result_type,
                  VECTOR_KERNEL_RANKED_TYPE_PLAN{
                      request._operand_types[0]._element_type,
                      {result_width}})) {
    *diagnostic = "invalid baseline Gemm plan invariants";
    return false;
  }

  const bool need_mask = !request._options._mask_fuse ||
                         Request_has_attribute(request, "mask");
  if (common._mask._policy !=
          (need_mask ? VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX
                     : VECTOR_KERNEL_MASK_POLICY::NONE) ||
      common._mask._valid_length != (need_mask ? plan._height : 0)) {
    *diagnostic = "baseline Gemm mask policy is inconsistent";
    return false;
  }
  std::vector<std::pair<std::string, VECTOR_KERNEL_RANKED_TYPE_PLAN>>
      constants{
          {"weight", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                         PRIMITIVE_TYPE::FLOAT_32,
                         {plan._height, plan._width}}},
          {"bias", source_bias->_type}};
  if (need_mask) {
    constants.push_back(
        {"mask", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                     PRIMITIVE_TYPE::FLOAT_32, {plan._height}}});
  }
  if (!Validate_constant_schema(common, constants, diagnostic)) return false;

  std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{
          "gemm", 0, static_cast<int32_t>(plan._height), 1, 0}};
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  std::vector<int32_t> duplication;
  if (!Build_duplication_rotations(plan._input_duplications, plan._width,
                                   &duplication, diagnostic)) {
    return false;
  }
  if (!duplication.empty()) {
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
        "input-duplication", std::move(duplication)});
  }
  std::vector<int32_t> gemm;
  if (!Build_range_rotations(plan._height, 1, 0, &gemm, diagnostic)) {
    return false;
  }
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"gemm", std::move(gemm)});

  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;
  const int64_t reduction_factor = plan._width / plan._height;
  if (reduction_factor > 1) {
    const uint32_t loop_count = Ceil_log2(reduction_factor);
    if (loop_count > static_cast<uint32_t>(
                         std::numeric_limits<int32_t>::max())) {
      *diagnostic = "baseline Gemm reduction loop count overflows";
      return false;
    }
    loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
        "block-reduction", 0, static_cast<int32_t>(loop_count), 1, 0});
    std::vector<int32_t> shifts;
    for (uint32_t idx = 0; idx < loop_count; ++idx) {
      const int64_t multiplier = int64_t{1} << idx;
      int64_t shift = 0;
      if (!Checked_mul_i64(multiplier, plan._height, &shift) ||
          !Append_i32(&shifts, shift, diagnostic)) {
        return false;
      }
    }
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
        "block-reduction", std::move(shifts)});
    reductions.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        "block-reduction", VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO,
        reduction_factor, plan._height, 0});
  }
  const std::vector<VECTOR_KERNEL_SLICE_PLAN> slices{
      VECTOR_KERNEL_SLICE_PLAN{
          "weight", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
          plan._width}};
  return Validate_exact_records(common._loops, loops, "loops", Equal_loop,
                                diagnostic) &&
         Validate_exact_records(common._slices, slices, "slices", Equal_slice,
                                diagnostic) &&
         Validate_exact_records(common._rotations, rotations, "rotations",
                                Equal_rotation, diagnostic) &&
         Validate_exact_records(common._reductions, reductions, "reductions",
                                Equal_reduction, diagnostic);
}

bool Validate_baseline_conv_plan(
    const BASELINE_CONV_PLAN& plan,
    const VECTOR_KERNEL_PLANNING_REQUEST& request, std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  CONV_REQUEST_FACTS facts{};
  if (!Build_conv_request_facts(request, &facts, diagnostic)) return false;
  int64_t input_size = 0;
  int64_t output_size = 0;
  int64_t kernel_hw = 0;
  int64_t expected_weight_rows = 0;
  if (plan._channel_in <= 0 || plan._channel_out <= 0 ||
      plan._output_height <= 0 || plan._output_width <= 0 ||
      plan._kernel_hw <= 0 || plan._stride <= 0 ||
      plan._input_duplications <= 0 ||
      !Checked_product_i64({plan._channel_in, plan._output_height,
                            plan._output_width},
                           &input_size) ||
      !Checked_product_i64({plan._channel_out, plan._output_height,
                            plan._output_width},
                           &output_size) ||
      !Checked_product_i64({request._operand_types[1]._shape[2],
                            request._operand_types[1]._shape[3]},
                           &kernel_hw) ||
      kernel_hw != plan._kernel_hw ||
      plan._channel_in != facts._channel_in ||
      plan._channel_out != facts._channel_out ||
      plan._output_height != facts._height ||
      plan._output_width != facts._width ||
      plan._kernel_hw != facts._kernel_hw || plan._stride != facts._stride ||
      input_size != facts._input_size || output_size != facts._output_size ||
      output_size > request._target._num_slots ||
      !Checked_mul_i64(plan._channel_in, plan._kernel_hw,
                       &expected_weight_rows) ||
      plan._channel_out != request._declared_result_type._shape[1] ||
      plan._output_height != request._declared_result_type._shape[2] ||
      plan._output_width != request._declared_result_type._shape[3] ||
      !Fits_i32(plan._channel_in) || !Fits_i32(plan._channel_out) ||
      !Fits_i32(plan._output_height) || !Fits_i32(plan._output_width) ||
      !Fits_i32(plan._kernel_hw) || !Fits_i32(plan._stride) ||
      !common._runtime_scalar_inputs.empty() ||
      common._mask._policy != VECTOR_KERNEL_MASK_POLICY::NONE ||
      common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS ||
      output_size > std::numeric_limits<uint32_t>::max() ||
      common._slot._value != static_cast<uint32_t>(output_size) ||
      !Equal_type(common._result_type, request._declared_result_type)) {
    *diagnostic = "invalid baseline Conv plan invariants";
    return false;
  }

  int64_t duplication_numerator = 0;
  int64_t expected_duplications = 0;
  int64_t doubled_input = 0;
  int64_t packed_input = 0;
  if (!Checked_add_i64(plan._channel_out, plan._channel_in - 1,
                       &duplication_numerator)) {
    *diagnostic = "baseline Conv duplication calculation overflows";
    return false;
  }
  expected_duplications = duplication_numerator / plan._channel_in;
  if (!Checked_add_i64(expected_duplications, 1,
                       &expected_duplications)) {
    *diagnostic = "baseline Conv duplication calculation overflows";
    return false;
  }
  if (Is_power_of_two(input_size)) {
    expected_duplications =
        std::min(expected_duplications,
                 request._target._num_slots / input_size);
  }
  if (!Checked_mul_i64(input_size, 2, &doubled_input)) {
    *diagnostic = "baseline Conv duplicated input size overflows";
    return false;
  }
  if (doubled_input > request._target._num_slots) expected_duplications = 1;
  if (expected_duplications <= 0 ||
      plan._input_duplications != expected_duplications ||
      !Checked_mul_i64(input_size, expected_duplications, &packed_input) ||
      packed_input > request._target._num_slots) {
    *diagnostic = "baseline Conv duplication policy is inconsistent";
    return false;
  }

  const VECTOR_KERNEL_CONSTANT_PLAN* weight =
      Find_constant_descriptor(common, "weight");
  if (weight == nullptr ||
      weight->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      weight->_type._shape.size() != 2 ||
      weight->_type._shape[0] != expected_weight_rows ||
      weight->_type._shape[1] != output_size) {
    *diagnostic = "baseline Conv weight descriptor is inconsistent";
    return false;
  }
  const std::vector<
      std::pair<std::string, VECTOR_KERNEL_RANKED_TYPE_PLAN>> constants{
      {"weight", weight->_type},
      {"bias", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                   PRIMITIVE_TYPE::FLOAT_32, {output_size}}},
      {"rotation-table", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                             PRIMITIVE_TYPE::INT_S32, {plan._kernel_hw}}}};
  if (!Validate_constant_schema(common, constants, diagnostic)) return false;

  std::vector<int32_t> duplication;
  if (!Build_duplication_rotations(plan._input_duplications, input_size,
                                   &duplication, diagnostic)) {
    return false;
  }
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  if (!duplication.empty()) {
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
        "input-duplication", std::move(duplication)});
  }
  const VECTOR_KERNEL_ROTATION_PLAN* kernel_alignment =
      Find_rotation(common, "kernel-alignment");
  std::vector<int32_t> expected_kernel_alignment;
  if (!Build_conv_alignment_rotations(facts, facts._stride,
                                      &expected_kernel_alignment,
                                      diagnostic)) {
    return false;
  }
  if (kernel_alignment == nullptr ||
      kernel_alignment->_candidates != expected_kernel_alignment) {
    *diagnostic = "baseline Conv kernel rotations are inconsistent";
    return false;
  }
  rotations.push_back(*kernel_alignment);
  int64_t channel_step = 0;
  if (!Checked_mul_i64(plan._output_height, plan._output_width,
                       &channel_step) ||
      !Fits_i32(channel_step)) {
    *diagnostic = "baseline Conv channel rotation overflows";
    return false;
  }
  rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
      "channel-step", {static_cast<int32_t>(channel_step)}});

  const std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{
          "channel-in", 0, static_cast<int32_t>(plan._channel_in), 1, 0},
      VECTOR_KERNEL_LOOP_PLAN{
          "kernel-hw", 0, static_cast<int32_t>(plan._kernel_hw), 1, 1}};
  const std::vector<VECTOR_KERNEL_SLICE_PLAN> slices{
      VECTOR_KERNEL_SLICE_PLAN{
          "weight",
          VECTOR_KERNEL_AFFINE_INDEX_PLAN{{plan._kernel_hw, 1}, 0, false},
          output_size}};
  if (!common._reductions.empty()) {
    *diagnostic = "baseline Conv cannot contain reductions";
    return false;
  }
  return Validate_exact_records(common._loops, loops, "loops", Equal_loop,
                                diagnostic) &&
         Validate_exact_records(common._slices, slices, "slices", Equal_slice,
                                diagnostic) &&
         Validate_exact_records(common._rotations, rotations, "rotations",
                                Equal_rotation, diagnostic);
}

bool Validate_fast_gemm_plan(
    const FAST_GEMM_PLAN& plan,
    const VECTOR_KERNEL_PLANNING_REQUEST& request, std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_weight =
      Find_request_constant(request, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_bias =
      Find_request_constant(request, "bias");
  int64_t block_product = 0;
  int64_t partition_product = 0;
  int64_t packed_stride = 0;
  int64_t result_width = 0;
  int64_t packed_height = 0;
  if (source_weight == nullptr || source_bias == nullptr || plan._n <= 0 ||
      plan._k <= 0 || plan._np <= 0 || plan._kp <= 0 ||
      plan._n != source_weight->_type._shape[0] ||
      plan._k != source_weight->_type._shape[1] || plan._np < plan._n ||
      plan._kp < plan._k || plan._nd != std::min(plan._np, plan._kp) ||
      plan._kd != std::max(plan._np, plan._kp) ||
      plan._block_size <= 0 || plan._blocks_per_partition <= 0 ||
      plan._packed_partitions <= 0 || plan._shift != 1 ||
      plan._shift_buffer < 0 || plan._grid_size <= 0 ||
      plan._input_replications <= 0 ||
      !Checked_mul_i64(plan._block_size, plan._blocks_per_partition,
                       &block_product) ||
      block_product != plan._nd ||
      !Checked_mul_i64(plan._grid_size, plan._packed_partitions,
                       &partition_product) ||
      partition_product != plan._blocks_per_partition ||
      plan._nd % plan._packed_partitions != 0 ||
      plan._shift_buffer !=
          ((plan._packed_partitions == 1 &&
            plan._kd == request._target._num_slots)
               ? 0
               : plan._nd / plan._packed_partitions) ||
      !Checked_add_i64(plan._kd, plan._shift_buffer, &packed_stride) ||
      !Checked_mul_i64(plan._packed_partitions, packed_stride,
                       &result_width) ||
      result_width > request._target._num_slots ||
      !Checked_mul_i64(plan._grid_size, plan._block_size, &packed_height) ||
      !common._runtime_scalar_inputs.empty() || !Fits_i32(plan._np) ||
      !Fits_i32(plan._kp) || !Fits_i32(plan._block_size) ||
      !Fits_i32(plan._grid_size) || plan._n > UINT32_MAX ||
      common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS ||
      common._slot._value != static_cast<uint32_t>(plan._n) ||
      !Equal_type(common._result_type,
                  VECTOR_KERNEL_RANKED_TYPE_PLAN{
                      request._operand_types[0]._element_type,
                      {result_width}})) {
    *diagnostic = "invalid fast Gemm plan invariants";
    return false;
  }
  int64_t expected_replications = 1;
  if (plan._kp != request._target._num_slots) {
    int64_t numerator = 0;
    if (!Checked_add_i64(result_width, plan._k - 1, &numerator)) {
      *diagnostic = "fast Gemm replication calculation overflows";
      return false;
    }
    expected_replications = numerator / plan._k;
  }
  if (plan._input_replications != expected_replications) {
    *diagnostic = "fast Gemm input replications are inconsistent";
    return false;
  }

  const bool clear_mask =
      !request._options._mask_fuse ||
      (!request._options._is_last_operation &&
       plan._n != request._target._num_slots);
  if (common._mask._policy !=
          (clear_mask ? VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX
                      : VECTOR_KERNEL_MASK_POLICY::NONE) ||
      common._mask._valid_length != (clear_mask ? plan._n : 0)) {
    *diagnostic = "fast Gemm mask policy is inconsistent";
    return false;
  }
  std::vector<std::pair<std::string, VECTOR_KERNEL_RANKED_TYPE_PLAN>>
      constants{
          {"weight", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                         PRIMITIVE_TYPE::FLOAT_32,
                         {packed_height, result_width}}},
          {"bias", source_bias->_type},
          {"rotation-table", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                                 PRIMITIVE_TYPE::INT_S32,
                                 {plan._block_size}}}};
  if (clear_mask) {
    constants.push_back(
        {"mask", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                     PRIMITIVE_TYPE::FLOAT_32, {plan._n}}});
  }
  if (!Validate_constant_schema(common, constants, diagnostic)) return false;

  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  std::vector<int32_t> duplication;
  if (!Build_duplication_rotations(plan._input_replications, plan._kp,
                                   &duplication, diagnostic)) {
    return false;
  }
  if (!duplication.empty()) {
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
        "input-duplication", std::move(duplication)});
  }
  std::vector<int32_t> blocking;
  std::vector<int32_t> grid;
  if (!Build_range_rotations(plan._block_size, 1, 0, &blocking, diagnostic) ||
      !Build_range_rotations(plan._grid_size, plan._block_size, 0, &grid,
                             diagnostic)) {
    return false;
  }
  rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
      "blocking-alignment", std::move(blocking)});
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"grid", std::move(grid)});
  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;
  if (!Append_reduction_schema("packed-partitions", plan._packed_partitions,
                               plan._kd, plan._shift_buffer, &rotations,
                               &reductions, diagnostic) ||
      !Append_reduction_schema("kp-over-np", plan._kp / plan._np, plan._np,
                               0, &rotations, &reductions, diagnostic)) {
    return false;
  }
  const std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{
          "grid", 0, static_cast<int32_t>(plan._grid_size), 1, 0},
      VECTOR_KERNEL_LOOP_PLAN{
          "block", 0, static_cast<int32_t>(plan._block_size), 1, 1}};
  const std::vector<VECTOR_KERNEL_SLICE_PLAN> slices{
      VECTOR_KERNEL_SLICE_PLAN{
          "weight",
          VECTOR_KERNEL_AFFINE_INDEX_PLAN{{plan._block_size, 1}, 0, false},
          result_width}};
  return Validate_exact_records(common._loops, loops, "loops", Equal_loop,
                                diagnostic) &&
         Validate_exact_records(common._slices, slices, "slices", Equal_slice,
                                diagnostic) &&
         Validate_exact_records(common._rotations, rotations, "rotations",
                                Equal_rotation, diagnostic) &&
         Validate_exact_records(common._reductions, reductions, "reductions",
                                Equal_reduction, diagnostic);
}

const VECTOR_KERNEL_TYPED_PAYLOAD* Find_typed_payload(
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& payloads,
    const std::string& role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload : payloads) {
    if (payload._role == role) return &payload;
  }
  return nullptr;
}

bool Decode_s32_payload(const VECTOR_KERNEL_TYPED_PAYLOAD& payload,
                        std::vector<int32_t>* values) {
  if (payload._type._element_type != PRIMITIVE_TYPE::INT_S32 ||
      payload._bytes.size() % sizeof(int32_t) != 0) {
    return false;
  }
  values->clear();
  values->reserve(payload._bytes.size() / sizeof(int32_t));
  for (size_t offset = 0; offset < payload._bytes.size();
       offset += sizeof(int32_t)) {
    uint32_t bits = 0;
    for (size_t byte = 0; byte < sizeof(int32_t); ++byte) {
      bits |= static_cast<uint32_t>(payload._bytes[offset + byte])
              << (byte * 8U);
    }
    const int64_t signed_value =
        bits <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
            ? static_cast<int64_t>(bits)
            : static_cast<int64_t>(bits) - (int64_t{1} << 32U);
    values->push_back(static_cast<int32_t>(signed_value));
  }
  return true;
}

bool F32_payload_element_is(const VECTOR_KERNEL_TYPED_PAYLOAD& payload,
                            size_t index, uint32_t expected_bits) {
  if (payload._type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      index >= payload._bytes.size() / sizeof(uint32_t)) {
    return false;
  }
  const size_t offset = index * sizeof(uint32_t);
  uint32_t bits = 0;
  for (size_t byte = 0; byte < sizeof(uint32_t); ++byte) {
    bits |= static_cast<uint32_t>(payload._bytes[offset + byte])
            << (byte * 8U);
  }
  return bits == expected_bits;
}

bool Validate_f32_fill(const VECTOR_KERNEL_TYPED_PAYLOAD& payload,
                       uint32_t expected_bits, std::string* diagnostic) {
  const size_t count = payload._bytes.size() / sizeof(uint32_t);
  for (size_t idx = 0; idx < count; ++idx) {
    if (!F32_payload_element_is(payload, idx, expected_bits)) {
      *diagnostic = "prepared mask payload is not its canonical v1 value";
      return false;
    }
  }
  return true;
}

bool Validate_rotation_payload(
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& payloads,
    const VECTOR_KERNEL_COMMON_PLAN& common, const std::string& rotation_role,
    std::string* diagnostic) {
  const VECTOR_KERNEL_TYPED_PAYLOAD* payload =
      Find_typed_payload(payloads, "rotation-table");
  const VECTOR_KERNEL_ROTATION_PLAN* rotation =
      Find_rotation(common, rotation_role);
  std::vector<int32_t> values;
  if (payload == nullptr || rotation == nullptr ||
      !Decode_s32_payload(*payload, &values) ||
      values != rotation->_candidates) {
    *diagnostic =
        "rotation-table payload does not match the canonical plan topology";
    return false;
  }
  return true;
}

bool Validate_cyclic_masks(
    const FAST_CONV_PLAN& plan,
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& payloads,
    std::string* diagnostic) {
  const VECTOR_KERNEL_TYPED_PAYLOAD* left =
      Find_typed_payload(payloads, "cyclic-mask-left");
  const VECTOR_KERNEL_TYPED_PAYLOAD* right =
      Find_typed_payload(payloads, "cyclic-mask-right");
  if (left == nullptr || right == nullptr) {
    *diagnostic = "fast Conv cyclic mask payloads are missing";
    return false;
  }
  constexpr uint32_t ZERO = 0x00000000U;
  constexpr uint32_t ONE = 0x3f800000U;
  for (int64_t grid = 0; grid < plan._num_grid; ++grid) {
    int64_t prefix = 0;
    if (!Checked_mul_i64(grid, plan._position_block, &prefix) ||
        prefix > plan._output_size) {
      *diagnostic = "fast Conv cyclic mask prefix is invalid";
      return false;
    }
    for (int64_t column = 0; column < plan._output_size; ++column) {
      int64_t linear = 0;
      if (!Checked_mul_i64(grid, plan._output_size, &linear) ||
          !Checked_add_i64(linear, column, &linear) ||
          static_cast<uint64_t>(linear) >
              std::numeric_limits<size_t>::max()) {
        *diagnostic = "fast Conv cyclic mask index overflows";
        return false;
      }
      const bool in_prefix = column < prefix;
      if (!F32_payload_element_is(*left, static_cast<size_t>(linear),
                                  in_prefix ? ONE : ZERO) ||
          !F32_payload_element_is(*right, static_cast<size_t>(linear),
                                  in_prefix ? ZERO : ONE)) {
        *diagnostic = "fast Conv cyclic mask payload is noncanonical";
        return false;
      }
    }
  }
  return true;
}

bool Validate_fast_conv_plan(
    const FAST_CONV_PLAN& plan,
    const VECTOR_KERNEL_PLANNING_REQUEST& request, std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  CONV_REQUEST_FACTS facts{};
  if (!Build_conv_request_facts(request, &facts, diagnostic)) return false;
  FAST_CONV_EXPECTED_TOPOLOGY expected_topology{};
  if (!Build_fast_conv_expected_topology(request, facts, &expected_topology,
                                         diagnostic)) {
    return false;
  }
  int64_t expected_input = 0;
  int64_t expected_output = 0;
  int64_t expected_width_block = 0;
  int64_t slice_width = 0;
  int64_t position_size = 0;
  if (plan._channel_in <= 0 || plan._channel_out <= 0 ||
      plan._output_height <= 0 || plan._output_width <= 0 ||
      plan._kernel_hw <= 0 || plan._group <= 0 || plan._stride <= 0 ||
      !Checked_mul_i64(plan._channel_in, plan._output_height,
                       &expected_input) ||
      !Checked_mul_i64(expected_input, plan._output_width,
                       &expected_input) ||
      !Checked_mul_i64(plan._channel_out, plan._output_height,
                       &expected_output) ||
      !Checked_mul_i64(expected_output, plan._output_width,
                       &expected_output) ||
      !Checked_mul_i64(plan._output_height, plan._output_width,
                       &position_size) ||
      expected_input != plan._input_size ||
      expected_output != plan._output_size ||
      plan._channel_in != facts._channel_in ||
      plan._channel_out != facts._channel_out ||
      plan._output_height != facts._height ||
      plan._output_width != facts._width ||
      plan._kernel_hw != facts._kernel_hw || plan._group != facts._group ||
      plan._stride != facts._stride || plan._input_size != facts._input_size ||
      plan._output_size != facts._output_size ||
      plan._output_size > request._target._num_slots ||
      plan._num_slots != request._target._num_slots || plan._num_grid <= 0 ||
      plan._num_block <= 0 || plan._width_block <= 0 ||
      plan._width_block_data <= 0 || plan._width_block_pad < 0 ||
      !Checked_add_i64(plan._width_block_data, plan._width_block_pad,
                       &expected_width_block) ||
      expected_width_block != plan._width_block ||
      plan._width_block_data != plan._output_size ||
      plan._num_grid != expected_topology._num_grid ||
      plan._num_block != expected_topology._num_block ||
      plan._width_block != expected_topology._width_block ||
      plan._width_block_pad != expected_topology._width_block_pad ||
      plan._position_block != expected_topology._position_block ||
      plan._capacity_block != expected_topology._capacity_block ||
      plan._input_duplications != expected_topology._input_duplications ||
      plan._blocking_outer_depth != facts._blocking_outer_depth ||
      !Checked_mul_i64(plan._num_block, plan._width_block, &slice_width) ||
      !Fits_i32(slice_width) || plan._position_block <= 0 ||
      plan._capacity_block <= 0 || plan._input_duplications <= 0 ||
      plan._blocking_outer_depth < 0 || !Fits_i32(plan._channel_in) ||
      !Fits_i32(plan._channel_out) || !Fits_i32(plan._output_height) ||
      !Fits_i32(plan._output_width) || !Fits_i32(plan._kernel_hw) ||
      !Fits_i32(plan._group) || !Fits_i32(plan._stride) ||
      !Fits_i32(plan._input_size) || !Fits_i32(plan._output_size) ||
      !Fits_i32(plan._num_grid) || !Fits_i32(plan._num_block) ||
      !Fits_i32(plan._width_block) || !Fits_i32(plan._width_block_data) ||
      !Fits_i32(plan._width_block_pad) || !Fits_i32(plan._position_block) ||
      !Fits_i32(plan._capacity_block) ||
      !Fits_i32(plan._input_duplications) ||
      !Fits_i32(plan._blocking_outer_depth) ||
      plan._channel_out != request._declared_result_type._shape[1] ||
      plan._output_height != request._declared_result_type._shape[2] ||
      plan._output_width != request._declared_result_type._shape[3] ||
      common._slot._policy !=
          VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS ||
      common._slot._value != static_cast<uint32_t>(expected_output) ||
      !Equal_type(common._result_type, request._declared_result_type)) {
    *diagnostic = "invalid fast Conv plan invariants";
    return false;
  }

  int64_t expected_position_block = position_size;
  if (plan._kernel_hw == 1 &&
      !Checked_mul_i64(position_size, plan._capacity_block,
                       &expected_position_block)) {
    *diagnostic = "fast Conv position block overflows";
    return false;
  }
  if (plan._position_block != expected_position_block) {
    *diagnostic = "fast Conv position block is inconsistent";
    return false;
  }

  const bool sharded = plan._sharding_offset.has_value();
  if (sharded) {
    if (plan._sharding_offset->_type != PRIMITIVE_TYPE::INT_S32 ||
        plan._sharding_offset->_scale != facts._sharding_offset_scale ||
        !Fits_i32(plan._sharding_offset->_scale) ||
        common._runtime_scalar_inputs !=
            std::vector<PRIMITIVE_TYPE>{PRIMITIVE_TYPE::INT_S32}) {
      *diagnostic = "invalid fast Conv sharding scalar invariants";
      return false;
    }
  } else if (!common._runtime_scalar_inputs.empty() ||
             plan._blocking_outer_depth != 0) {
    *diagnostic = "unsharded fast Conv has scalar or outer-depth state";
    return false;
  }
  if (sharded != facts._sharded ||
      common._runtime_scalar_inputs != request._runtime_scalar_types) {
    *diagnostic = "fast Conv sharding plan disagrees with the request";
    return false;
  }

  const bool expected_cyclic =
      plan._num_block == 1 && plan._output_size > plan._num_slots / 2 &&
      plan._output_size < plan._num_slots &&
      plan._group != plan._channel_in;
  const bool expected_collective =
      ((plan._channel_in > 1 && plan._group != plan._channel_in &&
        !expected_cyclic) ||
       plan._num_slots == plan._output_size);
  const bool need_mask = !request._options._mask_fuse ||
                         Request_has_attribute(request, "mask");
  if (plan._cyclic_roll != expected_cyclic ||
      common._mask._policy !=
          (expected_collective && need_mask
               ? VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION
               : VECTOR_KERNEL_MASK_POLICY::NONE) ||
      common._mask._valid_length !=
          (expected_collective && need_mask ? plan._output_size : 0)) {
    *diagnostic = "fast Conv mask/topology policy is inconsistent";
    return false;
  }

  const VECTOR_KERNEL_CONSTANT_PLAN* weight =
      Find_constant_descriptor(common, "weight");
  if (weight == nullptr ||
      weight->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      weight->_type._shape.size() != 2 ||
      weight->_type._shape[0] != expected_topology._weight_rows ||
      weight->_type._shape[1] != expected_topology._weight_columns ||
      weight->_type._shape[1] != slice_width) {
    *diagnostic = "fast Conv weight descriptor is inconsistent";
    return false;
  }
  std::vector<std::pair<std::string, VECTOR_KERNEL_RANKED_TYPE_PLAN>>
      constants{
          {"weight", weight->_type},
          {"bias", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                       PRIMITIVE_TYPE::FLOAT_32, {plan._output_size}}},
          {"rotation-table", VECTOR_KERNEL_RANKED_TYPE_PLAN{
                                 PRIMITIVE_TYPE::INT_S32,
                                 {plan._capacity_block}}}};
  if (expected_cyclic) {
    const VECTOR_KERNEL_RANKED_TYPE_PLAN mask_type{
        PRIMITIVE_TYPE::FLOAT_32,
        {plan._num_grid, plan._output_size}};
    constants.push_back({"cyclic-mask-left", mask_type});
    constants.push_back({"cyclic-mask-right", mask_type});
  }

  std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{
          "grid", 0, static_cast<int32_t>(plan._num_grid), 1, 0},
      VECTOR_KERNEL_LOOP_PLAN{
          "capacity-block", 0,
          static_cast<int32_t>(plan._capacity_block), 1, 1}};
  std::vector<VECTOR_KERNEL_SLICE_PLAN> slices{
      VECTOR_KERNEL_SLICE_PLAN{
          "weight",
          VECTOR_KERNEL_AFFINE_INDEX_PLAN{
              {plan._capacity_block, 1}, 0, sharded},
          slice_width}};
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  const VECTOR_KERNEL_ROTATION_PLAN* blocking =
      Find_rotation(common, "blocking-alignment");
  if (blocking == nullptr ||
      blocking->_candidates != expected_topology._blocking_rotations) {
    *diagnostic = "fast Conv blocking rotations are inconsistent";
    return false;
  }
  rotations.push_back(*blocking);
  if (expected_cyclic) {
    slices.push_back(VECTOR_KERNEL_SLICE_PLAN{
        "cyclic-mask-left", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
        plan._output_size});
    slices.push_back(VECTOR_KERNEL_SLICE_PLAN{
        "cyclic-mask-right", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
        plan._output_size});
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    for (int64_t grid_idx = 0; grid_idx < plan._num_grid; ++grid_idx) {
      int64_t shift = 0;
      if (!Checked_mul_i64(grid_idx, plan._position_block, &shift) ||
          shift > plan._output_size ||
          !Append_i32(&left, shift - plan._output_size, diagnostic) ||
          !Append_i32(&right, shift, diagnostic)) {
        return false;
      }
    }
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"cyclic-left", std::move(left)});
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"cyclic-right", std::move(right)});
  } else {
    std::vector<int32_t> grid;
    if (!Build_range_rotations(plan._num_grid, plan._position_block, 0, &grid,
                               diagnostic)) {
      return false;
    }
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"grid", std::move(grid)});
  }

  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;
  if (expected_collective) {
    const bool single_overload = !request._options._conv_parallel &&
                                 !request._options._sharding &&
                                 plan._num_block == 1;
    int64_t collective_padding = plan._width_block_pad;
    if (!single_overload && plan._num_block == 1 &&
        plan._num_slots != plan._output_size &&
        (!Checked_mul_i64(plan._channel_in - 1, plan._output_height,
                          &collective_padding) ||
         !Checked_mul_i64(collective_padding, plan._output_width,
                          &collective_padding))) {
      *diagnostic = "fast Conv collective padding overflows";
      return false;
    }
    if (collective_padding > plan._output_size) {
      *diagnostic = "fast Conv collective padding is out of range";
      return false;
    }
    reductions.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        "collective",
        single_overload
            ? VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK
            : VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS,
        plan._num_block, plan._output_size, collective_padding});

    const VECTOR_KERNEL_RANKED_TYPE_PLAN mask_type{
        PRIMITIVE_TYPE::FLOAT_32, {plan._output_size}};
    if (plan._num_slots == plan._output_size) {
      if (need_mask) constants.push_back({"collective-mask", mask_type});
    } else if (single_overload) {
      if (need_mask) constants.push_back({"collective-mask", mask_type});
      rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
          "collective-tail", {static_cast<int32_t>(-plan._output_size)}});
    } else {
      constants.push_back({"collective-mask", mask_type});
      constants.push_back({"collective-gap-mask", mask_type});
      if (plan._num_block > 1) {
        loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
            "collective-data", 1, static_cast<int32_t>(plan._num_block), 1,
            0});
        loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
            "collective-gap", 0,
            static_cast<int32_t>(plan._num_block - 1), 1, 0});
        std::vector<int32_t> data;
        if (!Build_range_rotations(plan._num_block, plan._width_block, 1,
                                   &data, diagnostic)) {
          return false;
        }
        std::vector<int32_t> gap;
        for (int64_t block = 0; block < plan._num_block - 1; ++block) {
          int64_t value = 0;
          if (!Checked_mul_i64(block, plan._width_block, &value) ||
              !Checked_add_i64(value, collective_padding, &value) ||
              !Append_i32(&gap, value, diagnostic)) {
            return false;
          }
        }
        rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
            "collective-data", std::move(data)});
        rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
            "collective-gap", std::move(gap)});
      }
      rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
          "collective-tail", {static_cast<int32_t>(-plan._output_size)}});
    }
  }
  return Validate_constant_schema(common, constants, diagnostic) &&
         Validate_exact_records(common._loops, loops, "loops", Equal_loop,
                                diagnostic) &&
         Validate_exact_records(common._slices, slices, "slices", Equal_slice,
                                diagnostic) &&
         Validate_exact_records(common._rotations, rotations, "rotations",
                                Equal_rotation, diagnostic) &&
         Validate_exact_records(common._reductions, reductions, "reductions",
                                Equal_reduction, diagnostic);
}

bool Validate_prepared_constant_semantics(
    const VECTOR_KERNEL_PLAN& plan,
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& payloads,
    std::string* diagnostic) {
  constexpr uint32_t ZERO = 0x00000000U;
  constexpr uint32_t ONE = 0x3f800000U;
  return std::visit(
      [&](const auto& typed_plan) {
        using PLAN = std::decay_t<decltype(typed_plan)>;
        if constexpr (std::is_same_v<PLAN, BASELINE_GEMM_PLAN>) {
          const VECTOR_KERNEL_TYPED_PAYLOAD* mask =
              Find_typed_payload(payloads, "mask");
          return mask == nullptr || Validate_f32_fill(*mask, ONE, diagnostic);
        } else if constexpr (std::is_same_v<PLAN, BASELINE_CONV_PLAN>) {
          return Validate_rotation_payload(payloads, typed_plan._common,
                                           "kernel-alignment", diagnostic);
        } else if constexpr (std::is_same_v<PLAN, FAST_GEMM_PLAN>) {
          if (!Validate_rotation_payload(payloads, typed_plan._common,
                                         "blocking-alignment", diagnostic)) {
            return false;
          }
          const VECTOR_KERNEL_TYPED_PAYLOAD* mask =
              Find_typed_payload(payloads, "mask");
          return mask == nullptr || Validate_f32_fill(*mask, ONE, diagnostic);
        } else {
          if (!Validate_rotation_payload(payloads, typed_plan._common,
                                         "blocking-alignment", diagnostic)) {
            return false;
          }
          if (typed_plan._cyclic_roll &&
              !Validate_cyclic_masks(typed_plan, payloads, diagnostic)) {
            return false;
          }
          const VECTOR_KERNEL_TYPED_PAYLOAD* collective =
              Find_typed_payload(payloads, "collective-mask");
          if (collective != nullptr &&
              !Validate_f32_fill(*collective, ONE, diagnostic)) {
            return false;
          }
          const VECTOR_KERNEL_TYPED_PAYLOAD* gap =
              Find_typed_payload(payloads, "collective-gap-mask");
          if (gap == nullptr) return true;
          if (typed_plan._common._reductions.size() != 1) {
            *diagnostic = "collective gap mask has no reduction descriptor";
            return false;
          }
          const int64_t padding =
              typed_plan._common._reductions.front()._padding;
          if (padding < 0 || padding > typed_plan._output_size) {
            *diagnostic = "collective gap mask padding is invalid";
            return false;
          }
          for (int64_t idx = 0; idx < typed_plan._output_size; ++idx) {
            const bool in_gap = idx >= typed_plan._output_size - padding;
            if (!F32_payload_element_is(*gap, static_cast<size_t>(idx),
                                        in_gap ? ONE : ZERO)) {
              *diagnostic = "collective gap mask payload is noncanonical";
              return false;
            }
          }
          return true;
        }
      },
      plan);
}

bool Validate_variant_against_request(
    const VECTOR_KERNEL_PLAN& plan,
    const VECTOR_KERNEL_PLANNING_REQUEST& request, std::string* diagnostic) {
  const VECTOR_KERNEL_COMMON_PLAN& common = Common_plan(plan);
  if (!Validate_common_plan(common, diagnostic) ||
      !Validate_runtime_input(common, request, diagnostic)) {
    return false;
  }
  return std::visit(
      [&](const auto& typed_plan) {
        using PLAN = std::decay_t<decltype(typed_plan)>;
        if constexpr (std::is_same_v<PLAN, BASELINE_GEMM_PLAN>) {
          return Validate_baseline_gemm_plan(typed_plan, request, diagnostic);
        } else if constexpr (std::is_same_v<PLAN, BASELINE_CONV_PLAN>) {
          return Validate_baseline_conv_plan(typed_plan, request, diagnostic);
        } else if constexpr (std::is_same_v<PLAN, FAST_GEMM_PLAN>) {
          return Validate_fast_gemm_plan(typed_plan, request, diagnostic);
        } else {
          return Validate_fast_conv_plan(typed_plan, request, diagnostic);
        }
      },
      plan);
}

VECTOR_KERNEL_PREPARE_RESULT Prepare_failure(std::string diagnostic) {
  return VECTOR_KERNEL_PREPARE_RESULT{
      nullptr, VECTOR_KERNEL_PLANNING_ERROR::INVALID_PROVIDER_RESULT,
      std::move(diagnostic)};
}

VECTOR_KERNEL_PREPARE_RESULT Request_failure(std::string diagnostic) {
  return VECTOR_KERNEL_PREPARE_RESULT{
      nullptr, VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST,
      std::move(diagnostic)};
}

VECTOR_KERNEL_RESOLUTION_RESULT Resolution_failure(
    VECTOR_KERNEL_PLAN_PROVIDER_KIND provider,
    VECTOR_KERNEL_IMPLEMENTATION implementation,
    VECTOR_KERNEL_PLANNING_ERROR error, std::string diagnostic) {
  return VECTOR_KERNEL_RESOLUTION_RESULT{
      nullptr, provider, implementation, error, false, std::move(diagnostic)};
}

const VECTOR_KERNEL_PLAN_PROVIDER* Selected_provider(
    VECTOR_KERNEL_PLAN_PROVIDER_KIND kind,
    const VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY* registry,
    const CPP_VECTOR_KERNEL_PLAN_PROVIDER* cpp_provider) {
  if (registry != nullptr) {
    const VECTOR_KERNEL_PLAN_PROVIDER* injected = registry->Lookup(kind);
    if (injected != nullptr) return injected;
  }
  return kind == VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP ? cpp_provider
                                                       : nullptr;
}

VECTOR_KERNEL_RESOLUTION_RESULT Invoke_and_validate(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    VECTOR_KERNEL_PLAN_PROVIDER_KIND provider_kind,
    VECTOR_KERNEL_IMPLEMENTATION implementation,
    const VECTOR_KERNEL_PLAN_PROVIDER& provider) {
  std::optional<VECTOR_KERNEL_PROVIDER_CALL_RESULT> provider_call;
  try {
    provider_call.emplace(provider.Plan(request));
  } catch (const std::exception& error) {
    return Resolution_failure(
        provider_kind, implementation,
        VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE,
        std::string("vector-kernel provider '") + provider.Name() +
            "' threw: " + error.what());
  } catch (...) {
    return Resolution_failure(
        provider_kind, implementation,
        VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE,
        std::string("vector-kernel provider '") + provider.Name() +
            "' threw an unknown exception");
  }
  if (!provider_call->_result.has_value()) {
    return Resolution_failure(
        provider_kind, implementation,
        VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE,
        std::string("vector-kernel provider '") + provider.Name() +
            "' failed: " + provider_call->_diagnostic);
  }
  VECTOR_KERNEL_PREPARE_RESULT prepared =
      Validate_and_prepare_vector_kernel_plan(request,
                                              *provider_call->_result);
  if (!prepared.Ok()) {
    return Resolution_failure(
        provider_kind, implementation, prepared._error,
        std::string("invalid result from vector-kernel provider '") +
            provider.Name() + "': " + prepared._diagnostic);
  }
  return VECTOR_KERNEL_RESOLUTION_RESULT{
      std::move(prepared._prepared), provider_kind, implementation,
      VECTOR_KERNEL_PLANNING_ERROR::NONE, false, {}};
}

}  // namespace

const char* Vector_kernel_operation_name(VECTOR_KERNEL_OPERATION operation) {
  switch (operation) {
    case VECTOR_KERNEL_OPERATION::GEMM:
      return "gemm";
    case VECTOR_KERNEL_OPERATION::CONV:
      return "conv";
  }
  return "invalid";
}

const char* Vector_kernel_requested_plan_kind_name(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind) {
  switch (kind) {
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO:
      return "auto";
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM:
      return "baseline-gemm";
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV:
      return "baseline-conv";
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM:
      return "fast-gemm";
    case VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV:
      return "fast-conv";
  }
  return "invalid";
}

VECTOR_KERNEL_SELECTION_RESULT Parse_vector_kernel_selection(
    const std::string& plan_provider, const std::string& kernel_impl,
    const std::string& plan_kind, const std::string& fallback) {
  VECTOR_KERNEL_SELECTION selection{};
  if (plan_provider == "cpp") {
    selection._plan_provider = VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP;
  } else if (plan_provider == "python") {
    selection._plan_provider = VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON;
  } else {
    return {std::nullopt, "invalid vector-kernel plan_provider: " +
                              plan_provider};
  }
  if (kernel_impl == "native") {
    selection._kernel_implementation = VECTOR_KERNEL_IMPLEMENTATION::NATIVE;
  } else if (kernel_impl == "dsl") {
    selection._kernel_implementation = VECTOR_KERNEL_IMPLEMENTATION::DSL;
  } else {
    return {std::nullopt,
            "invalid vector-kernel kernel_impl: " + kernel_impl};
  }
  if (plan_kind == "auto") {
    selection._plan_kind = VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO;
  } else if (plan_kind == "baseline-gemm") {
    selection._plan_kind = VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM;
  } else if (plan_kind == "baseline-conv") {
    selection._plan_kind = VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV;
  } else if (plan_kind == "fast-gemm") {
    selection._plan_kind = VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM;
  } else if (plan_kind == "fast-conv") {
    selection._plan_kind = VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV;
  } else {
    return {std::nullopt, "invalid vector-kernel plan_kind: " + plan_kind};
  }
  if (fallback == "error") {
    selection._fallback = VECTOR_KERNEL_FALLBACK_POLICY::ERROR;
  } else if (fallback == "cpp-native") {
    selection._fallback = VECTOR_KERNEL_FALLBACK_POLICY::CPP_NATIVE;
  } else {
    return {std::nullopt, "invalid vector-kernel fallback: " + fallback};
  }
  return {selection, {}};
}

bool VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY::Register(
    VECTOR_KERNEL_PLAN_PROVIDER_KIND kind,
    const VECTOR_KERNEL_PLAN_PROVIDER* provider) {
  if (!Valid_provider_kind(kind) || provider == nullptr) return false;
  const size_t index = Provider_index(kind);
  if (_providers[index] != nullptr) return false;
  _providers[index] = provider;
  return true;
}

bool VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY::Unregister(
    VECTOR_KERNEL_PLAN_PROVIDER_KIND kind) {
  if (!Valid_provider_kind(kind)) return false;
  const size_t index = Provider_index(kind);
  if (_providers[index] == nullptr) return false;
  _providers[index] = nullptr;
  return true;
}

const VECTOR_KERNEL_PLAN_PROVIDER*
VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY::Lookup(
    VECTOR_KERNEL_PLAN_PROVIDER_KIND kind) const {
  if (!Valid_provider_kind(kind)) return nullptr;
  return _providers[Provider_index(kind)];
}

VECTOR_KERNEL_PREPARE_RESULT Validate_and_prepare_vector_kernel_plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const VECTOR_KERNEL_PROVIDER_RESULT& provider_result) {
  std::string diagnostic;
  if (!Validate_planning_request(request, &diagnostic)) {
    return Request_failure(std::move(diagnostic));
  }
  const VECTOR_KERNEL_PLAN_KIND kind =
      Get_vector_kernel_plan_kind(provider_result._plan);
  if (!Kind_matches_operation(request._operation, kind)) {
    return Prepare_failure("provider returned a plan for the wrong operation");
  }
  if (request._requested_plan_kind !=
          VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO &&
      Requested_to_plan_kind(request._requested_plan_kind) != kind) {
    return Prepare_failure("provider ignored the forced plan kind");
  }

  if (!Validate_variant_against_request(provider_result._plan, request,
                                        &diagnostic)) {
    return Prepare_failure(std::move(diagnostic));
  }

  const VECTOR_KERNEL_COMMON_PLAN& common =
      Common_plan(provider_result._plan);
  if (provider_result._constants.size() != common._constants.size()) {
    return Prepare_failure("provider constant role count does not match plan");
  }
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> canonical_constants;
  canonical_constants.reserve(common._constants.size());
  std::set<std::string> seen_roles;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload :
       provider_result._constants) {
    if (!seen_roles.insert(payload._role).second) {
      return Prepare_failure("provider returned a duplicate constant role");
    }
  }
  for (const VECTOR_KERNEL_CONSTANT_PLAN& descriptor : common._constants) {
    auto found = std::find_if(
        provider_result._constants.begin(), provider_result._constants.end(),
        [&](const VECTOR_KERNEL_TYPED_PAYLOAD& payload) {
          return payload._role == descriptor._role;
        });
    if (found == provider_result._constants.end()) {
      return Prepare_failure("provider omitted constant role '" +
                             descriptor._role + "'");
    }
    if (!Equal_type(found->_type, descriptor._type)) {
      return Prepare_failure("provider constant type mismatch for role '" +
                             descriptor._role + "'");
    }
    size_t expected_bytes = 0;
    if (!Validate_type(found->_type, &expected_bytes, &diagnostic)) {
      return Prepare_failure(std::move(diagnostic));
    }
    if (found->_bytes.size() != expected_bytes) {
      return Prepare_failure("provider constant byte count mismatch for role '" +
                             descriptor._role + "'");
    }
    if (!Valid_hash_spelling(found->_content_hash)) {
      return Prepare_failure("provider constant hash has invalid spelling for role '" +
                             descriptor._role + "'");
    }
    const std::string recomputed =
        Canonical_payload_hash(found->_type, found->_bytes);
    if (found->_content_hash != recomputed ||
        descriptor._content_hash != recomputed) {
      return Prepare_failure("provider constant hash mismatch for role '" +
                             descriptor._role + "'");
    }
    canonical_constants.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
        found->_role, found->_type, found->_bytes, recomputed});
  }

  if (!Validate_prepared_constant_semantics(
          provider_result._plan, canonical_constants, &diagnostic)) {
    return Prepare_failure(std::move(diagnostic));
  }

  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> expected_runtime;
  std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> expected_scalars;
  if (!Build_expected_preparations(provider_result._plan, &expected_runtime,
                                   &expected_scalars, &diagnostic)) {
    return Prepare_failure(std::move(diagnostic));
  }
  if (provider_result._runtime_preparations.size() !=
      expected_runtime.size()) {
    return Prepare_failure("provider runtime preparation count mismatch");
  }
  for (size_t idx = 0; idx < expected_runtime.size(); ++idx) {
    if (!Equal_runtime_preparation(
            provider_result._runtime_preparations[idx],
            expected_runtime[idx])) {
      return Prepare_failure("provider runtime preparation mismatch");
    }
  }
  if (provider_result._scalar_preparations.size() !=
      expected_scalars.size()) {
    return Prepare_failure("provider scalar preparation count mismatch");
  }
  for (size_t idx = 0; idx < expected_scalars.size(); ++idx) {
    if (!Equal_scalar_preparation(provider_result._scalar_preparations[idx],
                                  expected_scalars[idx])) {
      return Prepare_failure("provider scalar preparation mismatch");
    }
  }

  const std::string key =
      Build_vector_kernel_specialization_key(provider_result._plan);
  const std::string name =
      Build_vector_kernel_helper_name(provider_result._plan);
  auto prepared = std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN>(
      new PREPARED_VECTOR_KERNEL_PLAN(
          provider_result._plan, std::move(canonical_constants),
          expected_runtime, expected_scalars, provider_result._provenance,
          key, name));
  return VECTOR_KERNEL_PREPARE_RESULT{
      std::move(prepared), VECTOR_KERNEL_PLANNING_ERROR::NONE, {}};
}

VECTOR_KERNEL_RESOLUTION_RESULT Resolve_vector_kernel_plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const VECTOR_KERNEL_SELECTION& selection,
    const VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY* provider_registry,
    VECTOR_KERNEL_DSL_RECIPE_QUERY has_dsl_recipe) {
  if (!Valid_provider_kind(selection._plan_provider) ||
      !Valid_implementation(selection._kernel_implementation) ||
      !Valid_requested_kind(selection._plan_kind) ||
      !Valid_fallback(selection._fallback)) {
    return Resolution_failure(
        selection._plan_provider, selection._kernel_implementation,
        VECTOR_KERNEL_PLANNING_ERROR::INVALID_SELECTION,
        "invalid vector-kernel selection enum value");
  }
  if (selection._plan_kind != request._requested_plan_kind) {
    return Resolution_failure(
        selection._plan_provider, selection._kernel_implementation,
        VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST,
        "selection and request plan kinds disagree");
  }
  if (selection._plan_kind != VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO &&
      !Kind_matches_operation(request._operation,
                              Requested_to_plan_kind(selection._plan_kind))) {
    return Resolution_failure(
        selection._plan_provider, selection._kernel_implementation,
        VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST,
        "forced vector-kernel plan kind is invalid for the operation");
  }

  std::string request_diagnostic;
  if (!Validate_planning_request(request, &request_diagnostic)) {
    return Resolution_failure(
        selection._plan_provider, selection._kernel_implementation,
        VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST,
        std::move(request_diagnostic));
  }

  CPP_VECTOR_KERNEL_PLAN_PROVIDER cpp_provider;
  const VECTOR_KERNEL_PLAN_PROVIDER* provider = Selected_provider(
      selection._plan_provider, provider_registry, &cpp_provider);
  if (provider == nullptr) {
    if (selection._fallback == VECTOR_KERNEL_FALLBACK_POLICY::ERROR) {
      return Resolution_failure(
          selection._plan_provider, selection._kernel_implementation,
          VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_UNAVAILABLE,
          "requested vector-kernel plan provider is unavailable");
    }
    VECTOR_KERNEL_RESOLUTION_RESULT fallback = Invoke_and_validate(
        request, VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
        VECTOR_KERNEL_IMPLEMENTATION::NATIVE, cpp_provider);
    if (fallback.Ok()) {
      fallback._used_fallback = true;
      fallback._diagnostic =
          "vector-kernel fallback: requested provider unavailable; using "
          "cpp/native";
    }
    return fallback;
  }

  VECTOR_KERNEL_RESOLUTION_RESULT selected = Invoke_and_validate(
      request, selection._plan_provider, selection._kernel_implementation,
      *provider);
  if (!selected.Ok()) return selected;

  if (selection._kernel_implementation ==
      VECTOR_KERNEL_IMPLEMENTATION::DSL) {
    const VECTOR_KERNEL_PLAN_KIND kind =
        Get_vector_kernel_plan_kind(selected._prepared->Plan());
    const bool recipe_available =
        has_dsl_recipe && has_dsl_recipe(kind);
    if (!recipe_available) {
      if (selection._fallback == VECTOR_KERNEL_FALLBACK_POLICY::ERROR) {
        return Resolution_failure(
            selection._plan_provider, selection._kernel_implementation,
            VECTOR_KERNEL_PLANNING_ERROR::UNSUPPORTED_DSL_RECIPE,
            "no DSL recipe is registered for the validated plan kind");
      }
      VECTOR_KERNEL_RESOLUTION_RESULT fallback = Invoke_and_validate(
          request, VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
          VECTOR_KERNEL_IMPLEMENTATION::NATIVE, cpp_provider);
      if (fallback.Ok()) {
        fallback._used_fallback = true;
        fallback._diagnostic =
            "vector-kernel fallback: DSL recipe unavailable; using "
            "cpp/native";
      }
      return fallback;
    }
  }
  return selected;
}

bool Equal_vector_kernel_prepared_semantics(
    const PREPARED_VECTOR_KERNEL_PLAN& lhs,
    const PREPARED_VECTOR_KERNEL_PLAN& rhs) {
  if (lhs.Specialization_key() != rhs.Specialization_key() ||
      lhs.Helper_name() != rhs.Helper_name() ||
      lhs.Constants().size() != rhs.Constants().size() ||
      lhs.Runtime_preparations().size() !=
          rhs.Runtime_preparations().size() ||
      lhs.Scalar_preparations().size() != rhs.Scalar_preparations().size()) {
    return false;
  }
  for (size_t idx = 0; idx < lhs.Constants().size(); ++idx) {
    const VECTOR_KERNEL_TYPED_PAYLOAD& left = lhs.Constants()[idx];
    const VECTOR_KERNEL_TYPED_PAYLOAD& right = rhs.Constants()[idx];
    if (left._role != right._role || !Equal_type(left._type, right._type) ||
        left._bytes != right._bytes ||
        left._content_hash != right._content_hash) {
      return false;
    }
  }
  for (size_t idx = 0; idx < lhs.Runtime_preparations().size(); ++idx) {
    if (!Equal_runtime_preparation(lhs.Runtime_preparations()[idx],
                                   rhs.Runtime_preparations()[idx])) {
      return false;
    }
  }
  for (size_t idx = 0; idx < lhs.Scalar_preparations().size(); ++idx) {
    if (!Equal_scalar_preparation(lhs.Scalar_preparations()[idx],
                                  rhs.Scalar_preparations()[idx])) {
      return false;
    }
  }
  return true;
}

}  // namespace vector
}  // namespace nn
