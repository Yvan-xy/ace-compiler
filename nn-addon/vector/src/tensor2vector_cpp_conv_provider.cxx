//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_cpp_conv_provider.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace nn {
namespace vector {

namespace {

using air::base::PRIMITIVE_TYPE;

struct CONV_PROBLEM {
  int64_t _channel_in_source = 0;
  int64_t _channel_in = 0;
  int64_t _channel_in_kernel_source = 0;
  int64_t _channel_in_kernel = 0;
  int64_t _channel_out = 0;
  int64_t _input_height = 0;
  int64_t _input_width = 0;
  int64_t _kernel_height = 0;
  int64_t _kernel_width = 0;
  int64_t _kernel_hw = 0;
  int64_t _group = 0;
  int64_t _stride = 0;
  int64_t _original_stride = 0;
  int64_t _padding = 0;
  int64_t _fhe_padding = 0;
  int64_t _input_size = 0;
  int64_t _output_size = 0;
  int64_t _position_size = 0;
  int64_t _num_slots = 0;
  bool _sharded = false;
  int64_t _shard_x = 1;
  int64_t _shard_y = 1;
  int64_t _shard_z = 1;
  int64_t _halo_size = 0;
  int64_t _halo_diameter = 0;
  int64_t _blocking_outer_depth = 0;
  int64_t _sharding_offset_scale = 0;
  bool _need_mask = true;
  std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN> _runtime_input_type;
  std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN> _result_type;
  std::vector<float> _source_weight;
  std::vector<float> _source_bias;
};

bool Checked_add(int64_t lhs, int64_t rhs, int64_t *result) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool Checked_mul(int64_t lhs, int64_t rhs, int64_t *result) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool Checked_signed_add(int64_t lhs, int64_t rhs, int64_t *result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs)) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool Checked_signed_scale(int64_t value, int64_t scale, int64_t *result) {
  if (scale < 0)
    return false;
  if (value >= 0)
    return Checked_mul(value, scale, result);
  if (scale == 0) {
    *result = 0;
    return true;
  }
  if (value == std::numeric_limits<int64_t>::min()) {
    if (scale != 1)
      return false;
    *result = value;
    return true;
  }
  int64_t magnitude = 0;
  if (!Checked_mul(-value, scale, &magnitude))
    return false;
  *result = -magnitude;
  return true;
}

bool Checked_product(std::initializer_list<int64_t> factors, int64_t *result) {
  int64_t product = 1;
  for (int64_t factor : factors) {
    if (factor <= 0 || !Checked_mul(product, factor, &product))
      return false;
  }
  *result = product;
  return true;
}

bool Fits_size_t(int64_t value) {
  return value >= 0 &&
         static_cast<uint64_t>(value) <=
             static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

bool Fits_i32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

bool Is_power_of_two(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

bool Host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t *>(&value) == 1;
}

template <typename T>
bool From_little_endian_bytes(const std::vector<uint8_t> &bytes,
                              std::vector<T> *values) {
  if (bytes.size() % sizeof(T) != 0)
    return false;
  values->resize(bytes.size() / sizeof(T));
  for (size_t idx = 0; idx < values->size(); ++idx) {
    std::array<uint8_t, sizeof(T)> element{};
    std::copy(bytes.begin() + idx * sizeof(T),
              bytes.begin() + (idx + 1) * sizeof(T), element.begin());
    if (!Host_is_little_endian())
      std::reverse(element.begin(), element.end());
    std::memcpy(&(*values)[idx], element.data(), sizeof(T));
  }
  return true;
}

template <typename T>
std::vector<uint8_t> To_little_endian_bytes(const std::vector<T> &values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  for (size_t idx = 0; idx < values.size(); ++idx) {
    std::array<uint8_t, sizeof(T)> element{};
    std::memcpy(element.data(), &values[idx], sizeof(T));
    if (!Host_is_little_endian())
      std::reverse(element.begin(), element.end());
    std::copy(element.begin(), element.end(), bytes.begin() + idx * sizeof(T));
  }
  return bytes;
}

std::string Payload_hash(const VECTOR_KERNEL_RANKED_TYPE_PLAN &type,
                         const std::vector<uint8_t> &bytes) {
  std::ostringstream header;
  header.imbue(std::locale::classic());
  header << "vector-kernel-constant:v1\n"
         << "element=" << Vector_kernel_primitive_type_name(type._element_type)
         << '\n'
         << "rank=" << type._shape.size() << '\n'
         << "shape=";
  for (size_t idx = 0; idx < type._shape.size(); ++idx) {
    if (idx != 0)
      header << ',';
    header << type._shape[idx];
  }
  header << "\npayload:\n";
  std::string preimage = header.str();
  if (!bytes.empty()) {
    preimage.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  return "sha256:" + Vector_kernel_sha256(preimage);
}

VECTOR_KERNEL_TYPED_PAYLOAD Make_payload(std::string role,
                                         PRIMITIVE_TYPE element_type,
                                         std::vector<int64_t> shape,
                                         std::vector<uint8_t> bytes) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{element_type, std::move(shape)};
  std::string hash = Payload_hash(type, bytes);
  return VECTOR_KERNEL_TYPED_PAYLOAD{std::move(role), std::move(type),
                                     std::move(bytes), std::move(hash)};
}

VECTOR_KERNEL_TYPED_PAYLOAD Make_f32_payload(std::string role,
                                             std::vector<int64_t> shape,
                                             const std::vector<float> &values) {
  return Make_payload(std::move(role), PRIMITIVE_TYPE::FLOAT_32,
                      std::move(shape), To_little_endian_bytes(values));
}

VECTOR_KERNEL_TYPED_PAYLOAD
Make_s32_payload(std::string role, std::vector<int64_t> shape,
                 const std::vector<int32_t> &values) {
  return Make_payload(std::move(role), PRIMITIVE_TYPE::INT_S32,
                      std::move(shape), To_little_endian_bytes(values));
}

VECTOR_KERNEL_CONSTANT_PLAN
Descriptor(const VECTOR_KERNEL_TYPED_PAYLOAD &payload) {
  return VECTOR_KERNEL_CONSTANT_PLAN{payload._role, payload._type,
                                     payload._content_hash};
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT Failure(std::string diagnostic) {
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(std::move(diagnostic));
}

const VECTOR_KERNEL_TYPED_PAYLOAD *
Find_source(const VECTOR_KERNEL_PLANNING_REQUEST &request,
            const std::string &role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : request._source_constants) {
    if (payload._role == role)
      return &payload;
  }
  return nullptr;
}

const VECTOR_KERNEL_ATTRIBUTE_RECORD *
Find_attribute(const VECTOR_KERNEL_PLANNING_REQUEST &request,
               const std::string &name) {
  for (const VECTOR_KERNEL_ATTRIBUTE_RECORD &attribute : request._attributes) {
    if (attribute._name == name)
      return &attribute;
  }
  return nullptr;
}

bool Attribute_i64(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                   const std::string &name, std::vector<int64_t> *values) {
  const VECTOR_KERNEL_ATTRIBUTE_RECORD *attribute =
      Find_attribute(request, name);
  if (attribute == nullptr)
    return false;
  if (attribute->_type._element_type == PRIMITIVE_TYPE::INT_S64) {
    return From_little_endian_bytes(attribute->_bytes, values);
  }
  if (attribute->_type._element_type == PRIMITIVE_TYPE::INT_S32) {
    std::vector<int32_t> narrow;
    if (!From_little_endian_bytes(attribute->_bytes, &narrow))
      return false;
    values->assign(narrow.begin(), narrow.end());
    return true;
  }
  if (attribute->_type._element_type == PRIMITIVE_TYPE::INT_U32) {
    std::vector<uint32_t> narrow;
    if (!From_little_endian_bytes(attribute->_bytes, &narrow))
      return false;
    values->assign(narrow.begin(), narrow.end());
    return true;
  }
  return false;
}

bool Required_attribute(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                        const std::string &name, size_t minimum_count,
                        std::vector<int64_t> *values, std::string *diagnostic) {
  if (!Attribute_i64(request, name, values) || values->size() < minimum_count) {
    *diagnostic = "Conv planning requires integer attribute '" + name + "'";
    return false;
  }
  return true;
}

bool Optional_attribute(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                        const std::string &name, std::vector<int64_t> *values,
                        std::string *diagnostic) {
  if (Find_attribute(request, name) == nullptr) {
    values->clear();
    return true;
  }
  if (!Attribute_i64(request, name, values)) {
    *diagnostic = "Conv attribute '" + name +
                  "' is not a canonical integer "
                  "array";
    return false;
  }
  return true;
}

bool Same_type(const VECTOR_KERNEL_RANKED_TYPE_PLAN &lhs,
               const VECTOR_KERNEL_RANKED_TYPE_PLAN &rhs) {
  return lhs._element_type == rhs._element_type && lhs._shape == rhs._shape;
}

bool Parse_conv_problem(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                        CONV_PROBLEM *problem, std::string *diagnostic) {
  if (request._operation != VECTOR_KERNEL_OPERATION::CONV) {
    *diagnostic = "Conv provider received a non-Conv request";
    return false;
  }
  if (request._operand_types.size() != 3) {
    *diagnostic = "Conv planning requires exactly three operands";
    return false;
  }
  const VECTOR_KERNEL_RANKED_TYPE_PLAN &input = request._operand_types[0];
  const VECTOR_KERNEL_RANKED_TYPE_PLAN &weight_type = request._operand_types[1];
  if (input._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      weight_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      request._declared_result_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      input._shape.size() != 4 || weight_type._shape.size() != 4) {
    *diagnostic =
        "Conv planning requires NCHW f32 input, weight, and result types";
    return false;
  }
  for (int64_t dimension : input._shape) {
    if (dimension <= 0) {
      *diagnostic = "Conv input shape contains a nonpositive dimension";
      return false;
    }
  }
  for (int64_t dimension : weight_type._shape) {
    if (dimension <= 0) {
      *diagnostic = "Conv weight shape contains a nonpositive dimension";
      return false;
    }
  }
  if (input._shape[0] != 1) {
    *diagnostic = "Conv planning only supports batch size one";
    return false;
  }

  problem->_channel_in_source = input._shape[1];
  problem->_channel_in = problem->_channel_in_source;
  problem->_input_height = input._shape[2];
  problem->_input_width = input._shape[3];
  problem->_channel_out = weight_type._shape[0];
  problem->_channel_in_kernel_source = weight_type._shape[1];
  problem->_channel_in_kernel = problem->_channel_in_kernel_source;
  problem->_kernel_height = weight_type._shape[2];
  problem->_kernel_width = weight_type._shape[3];
  if (problem->_kernel_height != problem->_kernel_width) {
    *diagnostic = "Conv planning only supports square kernels";
    return false;
  }
  if (!Checked_mul(problem->_kernel_height, problem->_kernel_width,
                   &problem->_kernel_hw) ||
      !Checked_mul(problem->_input_height, problem->_input_width,
                   &problem->_position_size) ||
      !Checked_mul(problem->_channel_in, problem->_position_size,
                   &problem->_input_size) ||
      !Checked_mul(problem->_channel_out, problem->_position_size,
                   &problem->_output_size)) {
    *diagnostic = "Conv dimensions overflow signed 64-bit arithmetic";
    return false;
  }
  if (!Fits_size_t(problem->_input_size) ||
      !Fits_size_t(problem->_output_size) ||
      problem->_output_size >=
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      request._target._num_slots <= 0) {
    *diagnostic = "Conv input, output, or slot dimensions are out of range";
    return false;
  }
  problem->_num_slots = request._target._num_slots;

  std::vector<int64_t> group;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  if (!Required_attribute(request, "group", 1, &group, diagnostic) ||
      !Required_attribute(request, "strides", 1, &strides, diagnostic) ||
      !Required_attribute(request, "pads", 4, &pads, diagnostic)) {
    return false;
  }
  problem->_group = group[0];
  if (problem->_group <= 0 || strides[0] <= 0 || pads[0] < 0) {
    *diagnostic = "Conv group, stride, or padding is invalid";
    return false;
  }
  if (strides.size() > 1 && strides[0] != strides[1]) {
    *diagnostic = "Conv planning requires equal spatial strides";
    return false;
  }
  problem->_stride = request._options._selective_strided_slice ? strides[0] : 1;
  problem->_padding = pads[0];
  problem->_fhe_padding = 0;

  std::vector<int64_t> original_strides;
  if (!Optional_attribute(request, "orig_strides", &original_strides,
                          diagnostic)) {
    return false;
  }
  problem->_original_stride = problem->_stride;
  if (!original_strides.empty()) {
    if (original_strides.size() != 2 ||
        original_strides[0] != original_strides[1] ||
        original_strides[0] <= 0) {
      *diagnostic =
          "Conv planning requires two equal positive orig_strides values";
      return false;
    }
    problem->_original_stride = original_strides[0];
  }

  std::vector<int64_t> keep_shape_pad;
  if (!Optional_attribute(request, "keep_shape_pad", &keep_shape_pad,
                          diagnostic)) {
    return false;
  }
  if (!keep_shape_pad.empty()) {
    if (keep_shape_pad[0] < 0) {
      *diagnostic = "Conv keep_shape_pad must be nonnegative";
      return false;
    }
    problem->_fhe_padding = keep_shape_pad[0];
  }

  if (problem->_group > 1) {
    if (problem->_channel_in != problem->_group) {
      *diagnostic =
          "depthwise Conv requires input channel count equal to group";
      return false;
    }
  } else if (problem->_channel_in != problem->_channel_in_kernel_source) {
    *diagnostic =
        "non-grouped Conv requires matching input and weight channels";
    return false;
  }

  std::vector<int64_t> sharded;
  if (!Optional_attribute(request, "weight-sharded", &sharded, diagnostic)) {
    return false;
  }
  problem->_sharded = !sharded.empty() && sharded[0] != 0;
  problem->_shard_x = 1;
  problem->_shard_y = 1;
  problem->_shard_z = 1;
  problem->_halo_size = 0;
  problem->_halo_diameter = 0;
  problem->_blocking_outer_depth = 0;
  problem->_sharding_offset_scale = 0;
  if (problem->_sharded) {
    std::vector<int64_t> shard;
    if (!Required_attribute(request, "sharding", 4, &shard, diagnostic)) {
      return false;
    }
    problem->_shard_x = shard[0];
    problem->_shard_y = shard[1];
    problem->_shard_z = shard[2];
    problem->_halo_size = shard[3];
    if (problem->_shard_x <= 0 || problem->_shard_y <= 0 ||
        problem->_shard_z <= 0 || problem->_halo_size < 0) {
      *diagnostic = "Conv sharding dimensions or halo size are invalid";
      return false;
    }
    if (!Checked_mul(problem->_halo_size, 2, &problem->_halo_diameter)) {
      *diagnostic = "Conv sharding halo size overflows";
      return false;
    }
    if (problem->_halo_diameter > problem->_input_height) {
      *diagnostic = "Conv sharding dimensions or halo size are invalid";
      return false;
    }
    problem->_blocking_outer_depth = 1;
    if (problem->_shard_z == 1) {
      std::vector<int64_t> original_group;
      if (!Required_attribute(request, "orig_group", 1, &original_group,
                              diagnostic)) {
        return false;
      }
      if (original_group[0] > 1) {
        problem->_shard_y = 1;
        problem->_blocking_outer_depth = 0;
      }
    }
    if (!Checked_mul(problem->_channel_in_kernel_source, problem->_kernel_hw,
                     &problem->_sharding_offset_scale)) {
      *diagnostic = "Conv sharding offset scale overflows";
      return false;
    }
  }

  const VECTOR_KERNEL_TYPED_PAYLOAD *source_weight =
      Find_source(request, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD *source_bias = Find_source(request, "bias");
  if (source_weight == nullptr || source_bias == nullptr ||
      source_weight->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      source_bias->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      !From_little_endian_bytes(source_weight->_bytes,
                                &problem->_source_weight) ||
      !From_little_endian_bytes(source_bias->_bytes, &problem->_source_bias)) {
    *diagnostic = "Conv planning requires owned f32 weight and bias payloads";
    return false;
  }
  int64_t local_weight_count = 0;
  if (!Checked_product({problem->_channel_out,
                        problem->_channel_in_kernel_source,
                        problem->_kernel_hw},
                       &local_weight_count)) {
    *diagnostic = "Conv local weight element count overflows";
    return false;
  }
  int64_t expected_weight_count = local_weight_count;
  if (problem->_sharded) {
    if (!Checked_mul(expected_weight_count, problem->_shard_x,
                     &expected_weight_count) ||
        !Checked_mul(expected_weight_count, problem->_shard_y,
                     &expected_weight_count)) {
      *diagnostic = "Conv sharded weight element count overflows";
      return false;
    }
  }
  if (static_cast<uint64_t>(expected_weight_count) !=
          problem->_source_weight.size() ||
      static_cast<uint64_t>(problem->_channel_out) !=
          problem->_source_bias.size()) {
    *diagnostic = "Conv weight or bias payload does not match its shape";
    return false;
  }

  if (problem->_channel_out >= problem->_channel_in &&
      problem->_channel_out % problem->_channel_in != 0) {
    int64_t padded_channel = problem->_channel_in;
    while (problem->_channel_out % padded_channel != 0) {
      if (padded_channel == std::numeric_limits<int64_t>::max()) {
        *diagnostic = "Conv channel padding overflows";
        return false;
      }
      ++padded_channel;
    }
    problem->_channel_in = padded_channel;
    problem->_channel_in_kernel = padded_channel;
    if (!Checked_mul(problem->_channel_in, problem->_position_size,
                     &problem->_input_size)) {
      *diagnostic = "Conv padded input size overflows";
      return false;
    }
  }

  int64_t runtime_input_size = 0;
  if (!Checked_mul(problem->_channel_in_source, problem->_position_size,
                   &runtime_input_size)) {
    *diagnostic = "Conv runtime input size overflows";
    return false;
  }
  problem->_runtime_input_type.emplace(VECTOR_KERNEL_RANKED_TYPE_PLAN{
      input._element_type, {runtime_input_size}});
  problem->_result_type.emplace(VECTOR_KERNEL_RANKED_TYPE_PLAN{
      input._element_type,
      {1, problem->_channel_out, problem->_input_height,
       problem->_input_width}});
  if (!Same_type(*problem->_result_type, request._declared_result_type)) {
    *diagnostic =
        "Conv declared result type does not match native same-shape lowering";
    return false;
  }

  problem->_need_mask = true;
  if (request._options._mask_fuse) {
    const VECTOR_KERNEL_ATTRIBUTE_RECORD *mask =
        Find_attribute(request, "mask");
    if (mask == nullptr) {
      problem->_need_mask = false;
    } else {
      std::vector<int64_t> values;
      if (!Attribute_i64(request, "mask", &values) || values.size() != 1 ||
          values[0] <= 0) {
        *diagnostic = "Conv mask attribute must contain one positive integer";
        return false;
      }
    }
  }
  return true;
}

bool Build_rotation_alignment(int64_t height, int64_t width,
                              int64_t kernel_height, int64_t kernel_width,
                              std::vector<int64_t> *rotations,
                              std::string *diagnostic) {
  int64_t kernel_hw = 0;
  if (!Checked_mul(kernel_height, kernel_width, &kernel_hw)) {
    *diagnostic = "Conv rotation table size overflows";
    return false;
  }
  if (!Fits_size_t(kernel_hw)) {
    *diagnostic = "Conv rotation table size is out of range";
    return false;
  }
  rotations->assign(static_cast<size_t>(kernel_hw), 0);
  const int64_t center = (kernel_height - 1) / 2;
  if (height != width) {
    for (int64_t row = 0; row < kernel_height; ++row) {
      for (int64_t column = 0; column < kernel_width; ++column) {
        const int64_t row_offset = row - center;
        const int64_t column_offset = column - center;
        int64_t row_rotation = 0;
        int64_t rotation = 0;
        int64_t row_index = 0;
        int64_t index = 0;
        if (!Checked_signed_scale(row_offset, width, &row_rotation) ||
            !Checked_signed_add(row_rotation, column_offset, &rotation) ||
            !Checked_mul(row, kernel_height, &row_index) ||
            !Checked_add(row_index, column, &index)) {
          *diagnostic = "Conv rotation alignment arithmetic overflows";
          return false;
        }
        (*rotations)[static_cast<size_t>(index)] = rotation;
      }
    }
    return true;
  }
  const int64_t pad = (kernel_height - 1) / 2;
  int64_t edge_rotation = 0;
  if (!Checked_mul(height, pad, &edge_rotation) ||
      !Checked_add(edge_rotation, pad, &edge_rotation)) {
    *diagnostic = "Conv rotation alignment arithmetic overflows";
    return false;
  }
  (*rotations)[0] = -edge_rotation;
  (*rotations)[static_cast<size_t>(kernel_hw - 1)] = edge_rotation;
  (*rotations)[static_cast<size_t>(kernel_hw / 2)] = 0;
  if (kernel_height > 1) {
    (*rotations)[static_cast<size_t>(kernel_hw / 2 + 1)] = 1;
    (*rotations)[static_cast<size_t>(kernel_hw / 2 - 1)] = -1;
  }
  for (int64_t idx = 1; idx < kernel_hw / 2 - 1; ++idx) {
    int64_t rotation = 0;
    if (!Checked_signed_add((*rotations)[static_cast<size_t>(idx - 1)], 1,
                            &rotation)) {
      *diagnostic = "Conv rotation alignment arithmetic overflows";
      return false;
    }
    if (kernel_height > 3 && idx % kernel_height == 0 &&
        !Checked_signed_add(rotation, height - kernel_height, &rotation)) {
      *diagnostic = "Conv rotation alignment arithmetic overflows";
      return false;
    }
    (*rotations)[static_cast<size_t>(idx)] = rotation;
  }
  for (int64_t idx = kernel_hw - 1; idx > kernel_hw / 2 + 2; --idx) {
    int64_t rotation = 0;
    if (!Checked_signed_add((*rotations)[static_cast<size_t>(idx)], -1,
                            &rotation)) {
      *diagnostic = "Conv rotation alignment arithmetic overflows";
      return false;
    }
    if (kernel_height > 3 && idx % kernel_height == 0 &&
        !Checked_signed_add(rotation, kernel_height - height, &rotation)) {
      *diagnostic = "Conv rotation alignment arithmetic overflows";
      return false;
    }
    (*rotations)[static_cast<size_t>(idx - 1)] = rotation;
  }
  return true;
}

bool Build_im2col(const std::vector<float> &weight, int64_t channel_in,
                  int64_t height, int64_t width, int64_t channel_out,
                  int64_t kernel_height, int64_t kernel_width, int64_t stride,
                  std::vector<int64_t> *rotations,
                  std::vector<std::vector<float>> *matrix,
                  std::string *diagnostic) {
  int64_t kernel_hw = 0;
  int64_t plane = 0;
  int64_t output_size = 0;
  int64_t row_count = 0;
  int64_t matrix_count = 0;
  if (!Checked_mul(kernel_height, kernel_width, &kernel_hw) ||
      !Checked_mul(height, width, &plane) ||
      !Checked_mul(channel_out, plane, &output_size) ||
      !Checked_mul(channel_in, kernel_hw, &row_count) ||
      !Checked_mul(row_count, output_size, &matrix_count) ||
      !Fits_size_t(matrix_count)) {
    *diagnostic = "Conv im2col dimensions overflow";
    return false;
  }
  int64_t expected_weight = 0;
  if (!Checked_mul(channel_out, row_count, &expected_weight) ||
      static_cast<uint64_t>(expected_weight) > weight.size()) {
    *diagnostic = "Conv im2col source weight is too small";
    return false;
  }

  if (!Build_rotation_alignment(height, width, kernel_height, kernel_width,
                                rotations, diagnostic)) {
    return false;
  }
  std::vector<int64_t> scaled_rotations(static_cast<size_t>(kernel_hw));
  for (int64_t idx = 0; idx < kernel_hw; ++idx) {
    int64_t scaled_rotation = 0;
    if (!Checked_signed_scale((*rotations)[static_cast<size_t>(idx)], stride,
                              &scaled_rotation) ||
        scaled_rotation == std::numeric_limits<int64_t>::min()) {
      *diagnostic = "Conv stride-scaled rotation overflows";
      return false;
    }
    scaled_rotations[static_cast<size_t>(idx)] = scaled_rotation;
  }
  matrix->assign(static_cast<size_t>(row_count),
                 std::vector<float>(static_cast<size_t>(output_size), 0.0F));
  const int64_t pad = (kernel_height - 1) / 2;
  for (int64_t output_channel = 0; output_channel < channel_out;
       ++output_channel) {
    for (int64_t row = 0; row < row_count; ++row) {
      const int64_t kernel_index = row % kernel_hw;
      const int64_t source_row = (row + output_channel * kernel_hw) % row_count;
      const float weight_value =
          weight[static_cast<size_t>(output_channel * row_count + source_row)];
      const int64_t kernel_row = kernel_index / kernel_width;
      const int64_t kernel_column = kernel_index % kernel_width;
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const bool in_bounds =
              h + kernel_row >= pad && w + kernel_column >= pad &&
              h + kernel_row < height + pad && w + kernel_column < width + pad;
          const int64_t position = h * width + w;
          float value = in_bounds ? weight_value : 0.0F;
          const int64_t scaled_rotation =
              scaled_rotations[static_cast<size_t>(kernel_index)];
          if (stride > 1 &&
              ((scaled_rotation > 0 &&
                (scaled_rotation > plane ||
                 position >= plane - scaled_rotation)) ||
               (scaled_rotation < 0 && position < -scaled_rotation))) {
            value = 0.0F;
          }
          (*matrix)[static_cast<size_t>(row)]
                   [static_cast<size_t>(output_channel * plane + position)] =
                       value;
        }
      }
    }
  }
  return true;
}

void Mask_stride_vector(int64_t height, int64_t width, int64_t channels,
                        int64_t stride, std::vector<float> *values) {
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t row = 0; row < height; ++row) {
      for (int64_t column = 0; column < width; ++column) {
        if (row % stride != 0 || column % stride != 0) {
          (*values)[static_cast<size_t>(channel * height * width + row * width +
                                        column)] = 0.0F;
        }
      }
    }
  }
}

void Mask_no_padding_vector(int64_t height, int64_t width, int64_t channels,
                            int64_t stride, int64_t kernel_height,
                            int64_t kernel_width, int64_t fhe_padding,
                            std::vector<float> *values) {
  int64_t pad = (kernel_height - 1) / 2;
  if (fhe_padding > 0)
    pad = fhe_padding;
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t row = 0; row < height; ++row) {
      for (int64_t column = 0; column < width; ++column) {
        if (row < pad || row >= height - pad || column < pad ||
            column >= width - pad) {
          (*values)[static_cast<size_t>(channel * height * width + row * width +
                                        column)] = 0.0F;
        }
      }
    }
  }
  if (stride > 1) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      for (int64_t row = pad; row < height; ++row) {
        for (int64_t column = 0; column < width; ++column) {
          if ((row - pad) % stride != 0 ||
              (column >= pad && (column - pad) % stride != 0)) {
            (*values)[static_cast<size_t>(channel * height * width +
                                          row * width + column)] = 0.0F;
          }
        }
      }
    }
  }
}

void Mask_matrix_rows(int64_t height, int64_t width, int64_t channels,
                      int64_t padding, int64_t stride, int64_t kernel_height,
                      int64_t kernel_width, int64_t fhe_padding,
                      std::vector<std::vector<float>> *matrix) {
  for (std::vector<float> &row : *matrix) {
    if ((stride > 1 && padding != 0) ||
        (stride > 1 && padding == 0 && kernel_width == 1)) {
      Mask_stride_vector(height, width, channels, stride, &row);
    } else if (padding == 0 && kernel_width != 1) {
      Mask_no_padding_vector(height, width, channels, stride, kernel_height,
                             kernel_width, fhe_padding, &row);
    }
  }
}

void Mask_bias(const CONV_PROBLEM &problem, std::vector<float> *bias) {
  if ((problem._original_stride > 1 && problem._padding != 0) ||
      (problem._original_stride > 1 && problem._padding == 0 &&
       problem._kernel_width == 1)) {
    Mask_stride_vector(problem._input_height, problem._input_width,
                       problem._channel_out, problem._original_stride, bias);
  } else if (problem._padding == 0) {
    Mask_no_padding_vector(problem._input_height, problem._input_width,
                           problem._channel_out, problem._original_stride,
                           problem._kernel_height, problem._kernel_width,
                           problem._fhe_padding, bias);
  }
}

bool Pad_weight_channels(CONV_PROBLEM *problem,
                         std::vector<float> *local_weight,
                         std::string *diagnostic) {
  if (problem->_channel_in == problem->_channel_in_source)
    return true;
  int64_t padded_count = 0;
  if (!Checked_product(
          {problem->_channel_out, problem->_channel_in, problem->_kernel_hw},
          &padded_count) ||
      !Fits_size_t(padded_count)) {
    *diagnostic = "Conv padded weight size overflows";
    return false;
  }
  std::vector<float> padded(static_cast<size_t>(padded_count), 0.0F);
  for (int64_t output = 0; output < problem->_channel_out; ++output) {
    for (int64_t input = 0; input < problem->_channel_in_source; ++input) {
      for (int64_t kernel = 0; kernel < problem->_kernel_hw; ++kernel) {
        padded[static_cast<size_t>(output * problem->_channel_in *
                                       problem->_kernel_hw +
                                   input * problem->_kernel_hw + kernel)] =
            (*local_weight)[static_cast<size_t>(
                output * problem->_channel_in_source * problem->_kernel_hw +
                input * problem->_kernel_hw + kernel)];
      }
    }
  }
  *local_weight = std::move(padded);
  return true;
}

bool Build_expanded_bias(const CONV_PROBLEM &problem,
                         std::vector<float> *expanded,
                         std::string *diagnostic) {
  if (!Fits_size_t(problem._output_size)) {
    *diagnostic = "Conv expanded bias size is out of range";
    return false;
  }
  expanded->resize(static_cast<size_t>(problem._output_size));
  for (int64_t channel = 0; channel < problem._channel_out; ++channel) {
    std::fill(expanded->begin() + channel * problem._position_size,
              expanded->begin() + (channel + 1) * problem._position_size,
              problem._source_bias[static_cast<size_t>(channel)]);
  }
  Mask_bias(problem, expanded);
  return true;
}

bool Build_local_im2col(const CONV_PROBLEM &problem,
                        std::vector<int64_t> *rotations,
                        std::vector<std::vector<float>> *matrix,
                        std::string *diagnostic) {
  int64_t local_count = 0;
  if (!Checked_product({problem._channel_out, problem._channel_in_kernel_source,
                        problem._kernel_hw},
                       &local_count)) {
    *diagnostic = "Conv local weight size overflows";
    return false;
  }
  std::vector<float> local_weight(problem._source_weight.begin(),
                                  problem._source_weight.begin() +
                                      static_cast<size_t>(local_count));
  CONV_PROBLEM padded = problem;
  if (!Pad_weight_channels(&padded, &local_weight, diagnostic))
    return false;
  if (!Build_im2col(local_weight, problem._channel_in_kernel,
                    problem._input_height, problem._input_width,
                    problem._channel_out, problem._kernel_height,
                    problem._kernel_width, problem._stride, rotations, matrix,
                    diagnostic)) {
    return false;
  }
  Mask_matrix_rows(problem._input_height, problem._input_width,
                   problem._channel_out, problem._padding,
                   problem._original_stride, problem._kernel_height,
                   problem._kernel_width, problem._fhe_padding, matrix);
  return true;
}

int64_t Mini_factor(int64_t channels) {
  if (channels < 32 || !Is_power_of_two(channels))
    return 1;
  int64_t minimum_sum = channels;
  int64_t factor = 1;
  for (int64_t candidate = 2; candidate <= channels / candidate; ++candidate) {
    if (channels % candidate == 0 &&
        candidate + channels / candidate < minimum_sum) {
      minimum_sum = candidate + channels / candidate;
      factor = candidate;
    }
  }
  return factor;
}

int64_t Fusion_blocks(const CONV_PROBLEM &problem) {
  int64_t blocks = 1;
  if (Is_power_of_two(problem._output_size) &&
      problem._channel_out % problem._channel_in == 0 &&
      problem._kernel_hw > 1 && problem._group == 1 &&
      problem._num_slots / problem._output_size >= 4 &&
      problem._channel_out >= problem._channel_in) {
    blocks = problem._num_slots / problem._output_size / 2;
    blocks = std::min(blocks, problem._channel_in);
    if (blocks >= 2 &&
        problem._channel_in / blocks + 2 * blocks >= problem._channel_in) {
      blocks /= 2;
    }
    blocks = std::min<int64_t>(blocks, 8);
  }
  return blocks;
}

bool Append_i32(std::vector<int32_t> *values, int64_t value,
                std::string *diagnostic) {
  if (!Fits_i32(value)) {
    *diagnostic = "Conv rotation candidate exceeds signed 32-bit range";
    return false;
  }
  values->push_back(static_cast<int32_t>(value));
  return true;
}

bool Duplication_rotations(int64_t replications, int64_t input_size,
                           std::vector<int32_t> *rotations,
                           std::string *diagnostic) {
  for (int64_t idx = 1; idx < replications; ++idx) {
    int64_t offset = 0;
    if (!Checked_mul(idx, input_size, &offset) ||
        !Append_i32(rotations, -offset, diagnostic)) {
      return false;
    }
  }
  return true;
}

bool I32_rotations(const std::vector<int64_t> &source, int64_t scale,
                   std::vector<int32_t> *result, std::string *diagnostic) {
  result->reserve(source.size());
  for (int64_t rotation : source) {
    int64_t scaled_rotation = 0;
    if (!Checked_signed_scale(rotation, scale, &scaled_rotation)) {
      *diagnostic = "Conv scaled rotation overflows signed 64-bit arithmetic";
      return false;
    }
    if (!Append_i32(result, scaled_rotation, diagnostic))
      return false;
  }
  return true;
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT
Build_baseline_conv(const CONV_PROBLEM &problem,
                    std::vector<std::vector<float>> matrix,
                    const std::vector<int64_t> &unscaled_rotations) {
  std::string diagnostic;
  if (problem._sharded) {
    return Failure("baseline Conv does not support a runtime sharded weight "
                   "offset");
  }
  if (!Fits_i32(problem._channel_in) || !Fits_i32(problem._kernel_hw)) {
    return Failure("baseline Conv loop bounds exceed signed 32-bit range");
  }

  std::vector<float> weight;
  int64_t weight_rows = 0;
  int64_t weight_count = 0;
  if (!Checked_mul(problem._channel_in_kernel, problem._kernel_hw,
                   &weight_rows) ||
      !Checked_mul(weight_rows, problem._output_size, &weight_count) ||
      !Fits_size_t(weight_count)) {
    return Failure("baseline Conv packed weight size overflows");
  }
  weight.reserve(static_cast<size_t>(weight_count));
  for (const std::vector<float> &row : matrix) {
    weight.insert(weight.end(), row.begin(), row.end());
  }
  if (weight.size() != static_cast<size_t>(weight_count)) {
    return Failure("baseline Conv packed weight size is inconsistent");
  }

  std::vector<float> bias;
  if (!Build_expanded_bias(problem, &bias, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  int64_t duplication_numerator = 0;
  if (!Checked_add(problem._channel_out, problem._channel_in,
                   &duplication_numerator)) {
    return Failure("baseline Conv duplication count overflows");
  }
  --duplication_numerator;
  int64_t duplications = duplication_numerator / problem._channel_in;
  if (!Checked_add(duplications, 1, &duplications)) {
    return Failure("baseline Conv duplication count overflows");
  }
  if (Is_power_of_two(problem._input_size)) {
    duplications =
        std::min(duplications, problem._num_slots / problem._input_size);
  }
  int64_t doubled_input = 0;
  if (!Checked_mul(problem._input_size, 2, &doubled_input)) {
    return Failure("baseline Conv input duplication size overflows");
  }
  if (doubled_input > problem._num_slots)
    duplications = 1;
  int64_t packed_input_size = 0;
  if (duplications <= 0 ||
      !Checked_mul(problem._input_size, duplications, &packed_input_size) ||
      packed_input_size > problem._num_slots) {
    return Failure("baseline Conv duplicated input exceeds slot capacity");
  }

  std::vector<int32_t> duplication_candidates;
  if (!Duplication_rotations(duplications, problem._input_size,
                             &duplication_candidates, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  std::vector<int32_t> kernel_rotations;
  if (!I32_rotations(unscaled_rotations, problem._stride, &kernel_rotations,
                     &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  std::vector<int32_t> channel_rotations;
  if (!Append_i32(&channel_rotations, problem._position_size, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }

  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> payloads;
  payloads.push_back(Make_f32_payload(
      "weight", {weight_rows, problem._output_size},
      weight));
  payloads.push_back(Make_f32_payload("bias", {problem._output_size}, bias));
  payloads.push_back(Make_s32_payload("rotation-table", {problem._kernel_hw},
                                      kernel_rotations));

  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> constants;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : payloads) {
    constants.push_back(Descriptor(payload));
  }
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  if (!duplication_candidates.empty()) {
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{"input-duplication",
                                                    duplication_candidates});
  }
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"kernel-alignment", kernel_rotations});
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"channel-step", channel_rotations});

  VECTOR_KERNEL_PLAN plan = BASELINE_CONV_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {*problem._runtime_input_type},
          {},
          std::move(constants),
          *problem._result_type,
          {VECTOR_KERNEL_LOOP_PLAN{"channel-in", 0,
                                   static_cast<int32_t>(problem._channel_in), 1,
                                   0},
           VECTOR_KERNEL_LOOP_PLAN{
               "kernel-hw", 0, static_cast<int32_t>(problem._kernel_hw), 1, 1}},
          {VECTOR_KERNEL_SLICE_PLAN{"weight",
                                    VECTOR_KERNEL_AFFINE_INDEX_PLAN{
                                        {problem._kernel_hw, 1}, 0, false},
                                    problem._output_size}},
          std::move(rotations),
          {},
          VECTOR_KERNEL_MASK_PLAN{VECTOR_KERNEL_MASK_POLICY::NONE, 0},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS,
              static_cast<uint32_t>(problem._output_size)}},
      problem._channel_in,
      problem._channel_out,
      problem._input_height,
      problem._input_width,
      problem._kernel_hw,
      problem._stride,
      duplications};
  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime{
      VECTOR_KERNEL_RUNTIME_PREPARATION{
          "input", 0,
          VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::FLATTEN_PACKED_VECTOR,
          *problem._runtime_input_type, problem._input_size, duplications, 0,
          duplication_candidates, 0}};
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(
      VECTOR_KERNEL_PROVIDER_RESULT{
          std::move(plan), std::move(payloads), std::move(runtime), {}, "cpp"});
}

bool Build_sharded_weight(const CONV_PROBLEM &problem,
                          std::vector<float> *prepared,
                          std::vector<int64_t> *shape,
                          std::string *diagnostic) {
  int64_t shard_count = 0;
  int64_t global_channel_out = 0;
  int64_t global_channel_in = 0;
  if (!Checked_mul(problem._shard_x, problem._shard_y, &shard_count) ||
      !Checked_mul(problem._channel_out, problem._shard_x,
                   &global_channel_out) ||
      !Checked_mul(problem._channel_in_kernel_source, problem._shard_y,
                   &global_channel_in)) {
    *diagnostic = "Conv sharded global dimensions overflow";
    return false;
  }
  int64_t expected = 0;
  if (!Checked_product(
          {global_channel_out, global_channel_in, problem._kernel_hw},
          &expected) ||
      static_cast<uint64_t>(expected) != problem._source_weight.size()) {
    *diagnostic =
        "Conv sharded source payload does not match global tensor dimensions";
    return false;
  }
  int64_t result_rows = 0;
  int64_t row_width = 0;
  int64_t result_count = 0;
  if (!Checked_product(
          {problem._channel_in_kernel_source, problem._kernel_hw, shard_count},
          &result_rows) ||
      !Checked_mul(problem._channel_out, problem._position_size, &row_width) ||
      !Checked_mul(result_rows, row_width, &result_count) ||
      !Fits_size_t(result_count)) {
    *diagnostic = "Conv sharded prepared weight size overflows";
    return false;
  }
  prepared->clear();
  prepared->reserve(static_cast<size_t>(result_count));

  int64_t local_weight_count = 0;
  if (!Checked_product({problem._channel_out,
                        problem._channel_in_kernel_source, problem._kernel_hw},
                       &local_weight_count) ||
      !Fits_size_t(local_weight_count)) {
    *diagnostic = "Conv sharded local weight size overflows";
    return false;
  }
  for (int64_t shard_x = 0; shard_x < problem._shard_x; ++shard_x) {
    for (int64_t shard_y = 0; shard_y < problem._shard_y; ++shard_y) {
      std::vector<float> local;
      local.reserve(static_cast<size_t>(local_weight_count));
      for (int64_t output = 0; output < problem._channel_out; ++output) {
        const int64_t global_output = shard_x * problem._channel_out + output;
        for (int64_t input = 0; input < problem._channel_in_kernel_source;
             ++input) {
          const int64_t global_input =
              shard_y * problem._channel_in_kernel_source + input;
          const int64_t start =
              (global_output * global_channel_in + global_input) *
              problem._kernel_hw;
          local.insert(local.end(), problem._source_weight.begin() + start,
                       problem._source_weight.begin() + start +
                           problem._kernel_hw);
        }
      }
      std::vector<int64_t> rotations;
      std::vector<std::vector<float>> matrix;
      if (!Build_im2col(local, problem._channel_in_kernel_source,
                        problem._input_height, problem._input_width,
                        problem._channel_out, problem._kernel_height,
                        problem._kernel_width, 1, &rotations, &matrix,
                        diagnostic)) {
        return false;
      }
      int64_t overlap = 0;
      if (!Checked_mul(problem._halo_size, problem._input_width, &overlap) ||
          overlap > problem._position_size) {
        *diagnostic = "Conv sharded overlap size overflows";
        return false;
      }
      for (int64_t input = 0; input < problem._channel_in_kernel_source;
           ++input) {
        for (int64_t kernel = 0; kernel < problem._kernel_hw; ++kernel) {
          std::vector<float> &row =
              matrix[static_cast<size_t>(input * problem._kernel_hw + kernel)];
          for (int64_t output = 0; output < problem._channel_out; ++output) {
            const int64_t base = output * problem._position_size;
            std::fill(row.begin() + base, row.begin() + base + overlap, 0.0F);
            std::fill(row.begin() + base + problem._position_size - overlap,
                      row.begin() + base + problem._position_size, 0.0F);
            for (int64_t h = 0;
                 h < problem._input_height - problem._halo_diameter; ++h) {
              for (int64_t w = 0; w < problem._input_width; ++w) {
                if (h % problem._original_stride != 0 ||
                    w % problem._original_stride != 0) {
                  row[static_cast<size_t>(
                      base + (problem._halo_size + h) * problem._input_width +
                      w)] = 0.0F;
                }
              }
            }
          }
          const int64_t shift =
              (input % problem._channel_out) * problem._position_size;
          if (shift != 0) {
            std::rotate(row.begin(), row.end() - shift, row.end());
          }
          prepared->insert(prepared->end(), row.begin(), row.end());
        }
      }
    }
  }
  if (prepared->size() != static_cast<size_t>(result_count)) {
    *diagnostic = "Conv sharded prepared weight has inconsistent size";
    return false;
  }
  *shape = {result_rows, row_width};
  return true;
}

void Rotate_right(std::vector<float> *values, int64_t shift) {
  if (shift == 0 || values->empty())
    return;
  shift %= static_cast<int64_t>(values->size());
  std::rotate(values->begin(), values->end() - shift, values->end());
}

bool Make_range_rotations(int64_t count, int64_t scale, int64_t start,
                          std::vector<int32_t> *rotations,
                          std::string *diagnostic) {
  for (int64_t idx = start; idx < count; ++idx) {
    int64_t value = 0;
    if (!Checked_mul(idx, scale, &value) ||
        !Append_i32(rotations, value, diagnostic)) {
      return false;
    }
  }
  return true;
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT
Build_fast_conv(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                const CONV_PROBLEM &problem,
                std::vector<std::vector<float>> matrix,
                std::vector<int64_t> unscaled_rotations) {
  std::string diagnostic;
  if (problem._channel_out < problem._channel_in) {
    return Failure("fast Conv requires channel_out >= effective channel_in");
  }

  int64_t num_block =
      request._options._conv_parallel ? Fusion_blocks(problem) : 1;
  if (num_block <= 0 || problem._channel_in_kernel % num_block != 0) {
    return Failure("fast Conv block count does not divide input channels");
  }
  int64_t width_block_pad =
      num_block == 1 ? 0 : problem._input_size / num_block;
  int64_t num_grid = problem._channel_in_kernel;
  int64_t capacity_block = problem._kernel_hw;
  int64_t position_block = problem._position_size;
  int64_t input_duplications = problem._channel_out / problem._channel_in;

  std::vector<float> prepared_weight;
  std::vector<int64_t> prepared_weight_shape;
  if (problem._sharded) {
    if (!Build_sharded_weight(problem, &prepared_weight, &prepared_weight_shape,
                              &diagnostic)) {
      return Failure(std::move(diagnostic));
    }
  } else if (problem._kernel_hw == 1) {
    capacity_block = Mini_factor(problem._channel_in_kernel);
    if (capacity_block <= 0 ||
        problem._channel_in_kernel % capacity_block != 0) {
      return Failure("fast 1x1 Conv capacity does not divide input channels");
    }
    for (int64_t input = 0; input < problem._channel_in_kernel; ++input) {
      std::vector<float> &row = matrix[static_cast<size_t>(input)];
      int64_t shift = 0;
      if (!Checked_mul(capacity_block, input / capacity_block, &shift) ||
          !Checked_mul(shift, problem._position_size, &shift) ||
          shift > static_cast<int64_t>(row.size())) {
        return Failure("fast 1x1 Conv row alignment is out of range");
      }
      Rotate_right(&row, shift);
      prepared_weight.insert(prepared_weight.end(), row.begin(), row.end());
    }
    for (int64_t idx = 1; idx < capacity_block; ++idx) {
      int64_t rotation = 0;
      if (!Checked_mul(idx, problem._position_size, &rotation)) {
        return Failure("fast 1x1 Conv alignment overflows");
      }
      unscaled_rotations.push_back(rotation);
    }
    if (capacity_block > 1 &&
        !Checked_add(input_duplications, 1, &input_duplications)) {
      return Failure("fast 1x1 Conv input duplication count overflows");
    }
    num_grid /= capacity_block;
    if (!Checked_mul(position_block, capacity_block, &position_block)) {
      return Failure("fast 1x1 Conv position block overflows");
    }
    prepared_weight_shape = {problem._channel_in_kernel, problem._output_size};
  } else {
    int64_t offset = 0;
    if (!Checked_mul(problem._channel_in, problem._kernel_hw, &offset)) {
      return Failure("fast Conv block offset overflows");
    }
    offset /= num_block;
    std::vector<float> block_pad(static_cast<size_t>(width_block_pad), 0.0F);
    for (int64_t input = 0; input < problem._channel_in_kernel / num_block;
         ++input) {
      for (int64_t kernel = 0; kernel < problem._kernel_hw; ++kernel) {
        for (int64_t block = 0; block < num_block; ++block) {
          int64_t input_offset = 0;
          int64_t block_offset = 0;
          int64_t index = 0;
          if (!Checked_mul(input, problem._kernel_hw, &input_offset) ||
              !Checked_add(input_offset, kernel, &index) ||
              !Checked_mul(block, offset, &block_offset) ||
              !Checked_add(index, block_offset, &index)) {
            return Failure("fast Conv packed weight index overflows");
          }
          if (index < 0 || index >= static_cast<int64_t>(matrix.size())) {
            return Failure("fast Conv packed weight index is out of range");
          }
          std::vector<float> &row = matrix[static_cast<size_t>(index)];
          int64_t shift = 0;
          if (!Checked_mul(input, problem._position_size, &shift) ||
              shift > static_cast<int64_t>(row.size())) {
            return Failure("fast Conv row alignment is out of range");
          }
          Rotate_right(&row, shift);
          prepared_weight.insert(prepared_weight.end(), row.begin(), row.end());
          if (num_block > 1) {
            prepared_weight.insert(prepared_weight.end(), block_pad.begin(),
                                   block_pad.end());
          }
        }
      }
    }
    if (num_block > 1) {
      int64_t duplication_factor = 0;
      if (!Checked_add(num_block, 1, &duplication_factor) ||
          !Checked_mul(input_duplications, duplication_factor,
                       &input_duplications)) {
        return Failure("fast Conv input duplication count overflows");
      }
      num_grid /= num_block;
    }
    int64_t packed_width = 0;
    if (!Checked_add(problem._output_size, width_block_pad, &packed_width) ||
        !Checked_mul(num_block, packed_width, &packed_width)) {
      return Failure("fast Conv packed weight width overflows");
    }
    prepared_weight_shape = {offset, packed_width};
  }

  int64_t expected_weight_count = 0;
  if (!Checked_mul(prepared_weight_shape[0], prepared_weight_shape[1],
                   &expected_weight_count) ||
      static_cast<uint64_t>(expected_weight_count) != prepared_weight.size()) {
    return Failure("fast Conv packed weight has inconsistent dimensions");
  }
  int64_t width_block = 0;
  if (!Checked_add(problem._output_size, width_block_pad, &width_block)) {
    return Failure("fast Conv width block overflows");
  }
  if (num_grid <= 0 || capacity_block <= 0 || input_duplications <= 0 ||
      num_grid > std::numeric_limits<int32_t>::max() ||
      capacity_block > std::numeric_limits<int32_t>::max()) {
    return Failure("fast Conv loop or duplication bounds are out of range");
  }
  int64_t slice_width = 0;
  if (!Checked_mul(num_block, width_block, &slice_width)) {
    return Failure("fast Conv slice width overflows");
  }

  std::vector<int32_t> blocking_rotations;
  if (!I32_rotations(unscaled_rotations, 1, &blocking_rotations, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  if (blocking_rotations.size() != static_cast<size_t>(capacity_block)) {
    return Failure("fast Conv alignment count does not equal capacity block");
  }

  std::vector<float> bias;
  if (!Build_expanded_bias(problem, &bias, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  const bool cyclic = num_block == 1 &&
                      problem._output_size > problem._num_slots / 2 &&
                      problem._output_size < problem._num_slots &&
                      problem._group != problem._channel_in;
  const bool collective = ((problem._channel_in > 1 &&
                            problem._group != problem._channel_in && !cyclic) ||
                           problem._num_slots == problem._output_size);

  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> payloads;
  payloads.push_back(
      Make_f32_payload("weight", prepared_weight_shape, prepared_weight));
  payloads.push_back(Make_f32_payload("bias", {problem._output_size}, bias));
  payloads.push_back(
      Make_s32_payload("rotation-table", {capacity_block}, blocking_rotations));

  std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{"grid", 0, static_cast<int32_t>(num_grid), 1, 0},
      VECTOR_KERNEL_LOOP_PLAN{"capacity-block", 0,
                              static_cast<int32_t>(capacity_block), 1, 1}};
  std::vector<VECTOR_KERNEL_SLICE_PLAN> slices{VECTOR_KERNEL_SLICE_PLAN{
      "weight",
      VECTOR_KERNEL_AFFINE_INDEX_PLAN{{capacity_block, 1}, 0, problem._sharded},
      slice_width}};
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations{
      VECTOR_KERNEL_ROTATION_PLAN{"blocking-alignment", blocking_rotations}};
  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;

  if (cyclic) {
    int64_t mask_count = 0;
    if (!Checked_mul(num_grid, problem._output_size, &mask_count) ||
        !Fits_size_t(mask_count)) {
      return Failure("fast Conv cyclic mask size overflows");
    }
    std::vector<float> mask_left(static_cast<size_t>(mask_count), 0.0F);
    std::vector<float> mask_right(static_cast<size_t>(mask_count), 1.0F);
    for (int64_t grid = 0; grid < num_grid; ++grid) {
      int64_t prefix = 0;
      if (!Checked_mul(grid, position_block, &prefix) ||
          prefix > problem._output_size) {
        return Failure("fast Conv cyclic prefix is out of range");
      }
      for (int64_t idx = 0; idx < prefix; ++idx) {
        mask_left[static_cast<size_t>(grid * problem._output_size + idx)] =
            1.0F;
        mask_right[static_cast<size_t>(grid * problem._output_size + idx)] =
            0.0F;
      }
    }
    payloads.push_back(Make_f32_payload(
        "cyclic-mask-left", {num_grid, problem._output_size}, mask_left));
    payloads.push_back(Make_f32_payload(
        "cyclic-mask-right", {num_grid, problem._output_size}, mask_right));
    slices.push_back(VECTOR_KERNEL_SLICE_PLAN{
        "cyclic-mask-left", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
        problem._output_size});
    slices.push_back(VECTOR_KERNEL_SLICE_PLAN{
        "cyclic-mask-right", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
        problem._output_size});
    std::vector<int32_t> left;
    std::vector<int32_t> right;
    for (int64_t grid = 0; grid < num_grid; ++grid) {
      int64_t shift = 0;
      if (!Checked_mul(grid, position_block, &shift) ||
          !Append_i32(&left, shift - problem._output_size, &diagnostic) ||
          !Append_i32(&right, shift, &diagnostic)) {
        return Failure(std::move(diagnostic));
      }
    }
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"cyclic-left", std::move(left)});
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"cyclic-right", std::move(right)});
  } else {
    std::vector<int32_t> grid_rotations;
    if (!Make_range_rotations(num_grid, position_block, 0, &grid_rotations,
                              &diagnostic)) {
      return Failure(std::move(diagnostic));
    }
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"grid", std::move(grid_rotations)});
  }

  VECTOR_KERNEL_MASK_POLICY mask_policy = VECTOR_KERNEL_MASK_POLICY::NONE;
  int64_t mask_valid_length = 0;
  if (collective) {
    const bool single_overload = !request._options._conv_parallel &&
                                 !request._options._sharding && num_block == 1;
    int64_t collective_padding = width_block_pad;
    if (!single_overload && num_block == 1 &&
        problem._num_slots != problem._output_size &&
        !Checked_mul(problem._channel_in - 1, problem._position_size,
                     &collective_padding)) {
      return Failure("fast Conv collective padding overflows");
    }
    reductions.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        "collective",
        single_overload
            ? VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK
            : VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS,
        num_block, problem._output_size, collective_padding});
    if (problem._need_mask) {
      mask_policy = VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION;
      mask_valid_length = problem._output_size;
    }

    if (problem._num_slots == problem._output_size) {
      if (problem._need_mask) {
        payloads.push_back(Make_f32_payload(
            "collective-mask", {problem._output_size},
            std::vector<float>(static_cast<size_t>(problem._output_size),
                               1.0F)));
      }
    } else if (single_overload) {
      std::vector<int32_t> tail;
      if (!Append_i32(&tail, -problem._output_size, &diagnostic)) {
        return Failure(std::move(diagnostic));
      }
      rotations.push_back(
          VECTOR_KERNEL_ROTATION_PLAN{"collective-tail", std::move(tail)});
      if (problem._need_mask) {
        payloads.push_back(Make_f32_payload(
            "collective-mask", {problem._output_size},
            std::vector<float>(static_cast<size_t>(problem._output_size),
                               1.0F)));
      }
    } else {
      if (collective_padding > problem._output_size) {
        return Failure("fast Conv collective padding exceeds output size");
      }
      std::vector<float> data_mask(static_cast<size_t>(problem._output_size),
                                   1.0F);
      std::vector<float> gap_mask(static_cast<size_t>(problem._output_size),
                                  0.0F);
      std::fill(gap_mask.end() - collective_padding, gap_mask.end(), 1.0F);
      payloads.push_back(Make_f32_payload("collective-mask",
                                          {problem._output_size}, data_mask));
      payloads.push_back(Make_f32_payload("collective-gap-mask",
                                          {problem._output_size}, gap_mask));
      if (num_block > 1) {
        loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
            "collective-data", 1, static_cast<int32_t>(num_block), 1, 0});
        loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
            "collective-gap", 0, static_cast<int32_t>(num_block - 1), 1, 0});
        std::vector<int32_t> data_rotations;
        std::vector<int32_t> gap_rotations;
        if (!Make_range_rotations(num_block, width_block, 1, &data_rotations,
                                  &diagnostic)) {
          return Failure(std::move(diagnostic));
        }
        for (int64_t block = 0; block < num_block - 1; ++block) {
          int64_t rotation = 0;
          if (!Checked_mul(block, width_block, &rotation) ||
              !Checked_add(rotation, collective_padding, &rotation) ||
              !Append_i32(&gap_rotations, rotation, &diagnostic)) {
            return Failure(std::move(diagnostic));
          }
        }
        rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
            "collective-data", std::move(data_rotations)});
        rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{
            "collective-gap", std::move(gap_rotations)});
      }
      std::vector<int32_t> tail;
      if (!Append_i32(&tail, -problem._output_size, &diagnostic)) {
        return Failure(std::move(diagnostic));
      }
      rotations.push_back(
          VECTOR_KERNEL_ROTATION_PLAN{"collective-tail", std::move(tail)});
    }
  }

  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> constants;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : payloads) {
    constants.push_back(Descriptor(payload));
  }
  std::vector<PRIMITIVE_TYPE> scalar_inputs;
  std::optional<VECTOR_KERNEL_SHARDING_OFFSET_PLAN> sharding_offset;
  std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> scalar_preparations;
  if (problem._sharded) {
    scalar_inputs.push_back(PRIMITIVE_TYPE::INT_S32);
    sharding_offset.emplace(VECTOR_KERNEL_SHARDING_OFFSET_PLAN{
        PRIMITIVE_TYPE::INT_S32, problem._sharding_offset_scale});
    scalar_preparations.push_back(VECTOR_KERNEL_SCALAR_PREPARATION{
        "weight-offset", 1, PRIMITIVE_TYPE::INT_S32,
        problem._sharding_offset_scale});
  }

  VECTOR_KERNEL_PLAN plan = FAST_CONV_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {*problem._runtime_input_type},
          std::move(scalar_inputs),
          std::move(constants),
          *problem._result_type,
          std::move(loops),
          std::move(slices),
          std::move(rotations),
          std::move(reductions),
          VECTOR_KERNEL_MASK_PLAN{mask_policy, mask_valid_length},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS,
              static_cast<uint32_t>(problem._output_size)}},
      problem._channel_in,
      problem._channel_out,
      problem._input_height,
      problem._input_width,
      problem._kernel_hw,
      problem._group,
      problem._stride,
      problem._input_size,
      problem._output_size,
      problem._num_slots,
      num_grid,
      num_block,
      width_block,
      problem._output_size,
      width_block_pad,
      position_block,
      capacity_block,
      input_duplications,
      problem._blocking_outer_depth,
      cyclic,
      std::move(sharding_offset)};
  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime{
      VECTOR_KERNEL_RUNTIME_PREPARATION{
          "input", 0,
          VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS,
          *problem._runtime_input_type, problem._input_size, input_duplications,
          capacity_block, blocking_rotations,
          static_cast<uint32_t>(problem._blocking_outer_depth)}};
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(
      VECTOR_KERNEL_PROVIDER_RESULT{std::move(plan), std::move(payloads),
                                    std::move(runtime),
                                    std::move(scalar_preparations), "cpp"});
}

} // namespace

VECTOR_KERNEL_PROVIDER_CALL_RESULT
Plan_cpp_vector_kernel_conv(const VECTOR_KERNEL_PLANNING_REQUEST &request) {
  if (request._requested_plan_kind ==
          VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM ||
      request._requested_plan_kind ==
          VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM) {
    return Failure("Gemm plan kind cannot plan a Conv operation");
  }
  CONV_PROBLEM problem;
  std::string diagnostic;
  if (!Parse_conv_problem(request, &problem, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }

  std::vector<int64_t> rotations;
  std::vector<std::vector<float>> matrix;
  if (!Build_local_im2col(problem, &rotations, &matrix, &diagnostic)) {
    return Failure(std::move(diagnostic));
  }
  const bool use_fast = request._requested_plan_kind ==
                            VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV ||
                        (request._requested_plan_kind ==
                             VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO &&
                         problem._channel_out >= problem._channel_in);
  if (use_fast) {
    return Build_fast_conv(request, problem, std::move(matrix),
                           std::move(rotations));
  }
  return Build_baseline_conv(problem, std::move(matrix), rotations);
}

} // namespace vector
} // namespace nn
