//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "nn/vector/tensor2vector_planning.h"

using air::base::PRIMITIVE_TYPE;
using namespace nn::vector;

namespace {

std::vector<uint8_t> Float_bytes(const std::vector<float> &values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (float value : values) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

std::vector<uint8_t> Int32_bytes(const std::vector<int32_t> &values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(int32_t));
  for (int32_t value : values) {
    const uint32_t bits = static_cast<uint32_t>(value);
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

std::vector<uint8_t> Uint32_bytes(const std::vector<uint32_t> &values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(uint32_t));
  for (uint32_t value : values) {
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
    }
  }
  return bytes;
}

std::vector<uint8_t> Int64_bytes(const std::vector<int64_t> &values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(int64_t));
  for (int64_t value : values) {
    const uint64_t bits = static_cast<uint64_t>(value);
    for (uint32_t shift = 0; shift != 64; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

VECTOR_KERNEL_TYPED_PAYLOAD Float_payload(std::string role,
                                          std::vector<int64_t> shape,
                                          const std::vector<float> &values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::FLOAT_32,
                                      std::move(shape)};
  std::vector<uint8_t> bytes = Float_bytes(values);
  const std::string hash = Build_vector_kernel_constant_hash(
      type._element_type, type._shape, values.data(), bytes.size());
  return VECTOR_KERNEL_TYPED_PAYLOAD{std::move(role), std::move(type),
                                     std::move(bytes), hash};
}

VECTOR_KERNEL_ATTRIBUTE_RECORD Int32_attribute(std::string name,
                                               std::vector<int32_t> values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::INT_S32,
                                      {static_cast<int64_t>(values.size())}};
  return VECTOR_KERNEL_ATTRIBUTE_RECORD{std::move(name), std::move(type),
                                        Int32_bytes(values)};
}

VECTOR_KERNEL_ATTRIBUTE_RECORD Uint32_attribute(std::string name,
                                                std::vector<uint32_t> values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::INT_U32,
                                      {static_cast<int64_t>(values.size())}};
  return VECTOR_KERNEL_ATTRIBUTE_RECORD{std::move(name), std::move(type),
                                        Uint32_bytes(values)};
}

VECTOR_KERNEL_ATTRIBUTE_RECORD Int64_attribute(std::string name,
                                               std::vector<int64_t> values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::INT_S64,
                                      {static_cast<int64_t>(values.size())}};
  return VECTOR_KERNEL_ATTRIBUTE_RECORD{std::move(name), std::move(type),
                                        Int64_bytes(values)};
}

VECTOR_KERNEL_PLANNING_REQUEST
Conv_request(int64_t channel_in, int64_t channel_out, int64_t height,
             int64_t kernel, int64_t slots,
             VECTOR_KERNEL_REQUESTED_PLAN_KIND kind, bool sharded = false) {
  const VECTOR_KERNEL_RANKED_TYPE_PLAN input_type{
      PRIMITIVE_TYPE::FLOAT_32, {1, channel_in, height, height}};
  const VECTOR_KERNEL_RANKED_TYPE_PLAN weight_operand_type{
      PRIMITIVE_TYPE::FLOAT_32, {channel_out, channel_in, kernel, kernel}};
  const VECTOR_KERNEL_RANKED_TYPE_PLAN bias_type{PRIMITIVE_TYPE::FLOAT_32,
                                                 {channel_out}};

  std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> attributes;
  attributes.push_back(Int32_attribute("group", {1}));
  attributes.push_back(Int32_attribute("strides", {1, 1}));
  attributes.push_back(Int32_attribute(
      "pads",
      {static_cast<int32_t>(kernel / 2), static_cast<int32_t>(kernel / 2),
       static_cast<int32_t>(kernel / 2), static_cast<int32_t>(kernel / 2)}));
  int64_t shard_count = 1;
  if (sharded) {
    shard_count = 2;
    attributes.push_back(Int32_attribute("orig_strides", {1, 1}));
    attributes.push_back(Int32_attribute("weight-sharded", {1}));
    attributes.push_back(Int32_attribute("sharding", {2, 1, 2, 0}));
  }

  const int64_t weight_count =
      shard_count * channel_out * channel_in * kernel * kernel;
  std::vector<float> weight(static_cast<size_t>(weight_count));
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    const int32_t signed_value = static_cast<int32_t>(idx % 17) - 8;
    weight[idx] = static_cast<float>(signed_value) / 8.0F;
  }
  std::vector<float> bias(static_cast<size_t>(channel_out));
  for (size_t idx = 0; idx < bias.size(); ++idx) {
    bias[idx] = static_cast<float>(idx) / 4.0F;
  }
  std::vector<int64_t> source_weight_shape{channel_out, channel_in, kernel,
                                           kernel};
  if (sharded)
    source_weight_shape.insert(source_weight_shape.begin(), 2);

  return VECTOR_KERNEL_PLANNING_REQUEST{
      VECTOR_KERNEL_OPERATION::CONV,
      std::move(attributes),
      {input_type, weight_operand_type, bias_type},
      VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                     {1, channel_out, height, height}},
      VECTOR_KERNEL_OPTION_SNAPSHOT{false, false, false, sharded, false},
      VECTOR_KERNEL_TARGET_SNAPSHOT{slots, 1, 65536},
      {Float_payload("weight", std::move(source_weight_shape), weight),
       Float_payload("bias", {channel_out}, bias)},
      kind,
      sharded ? std::vector<PRIMITIVE_TYPE>{PRIMITIVE_TYPE::INT_S32}
              : std::vector<PRIMITIVE_TYPE>{}};
}

VECTOR_KERNEL_SELECTION Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND kind) {
  return VECTOR_KERNEL_SELECTION{VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                 VECTOR_KERNEL_IMPLEMENTATION::NATIVE, kind,
                                 VECTOR_KERNEL_FALLBACK_POLICY::ERROR};
}

std::vector<std::string>
Constant_roles(const PREPARED_VECTOR_KERNEL_PLAN &prepared) {
  std::vector<std::string> roles;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : prepared.Constants()) {
    roles.push_back(payload._role);
  }
  return roles;
}

VECTOR_KERNEL_RESOLUTION_RESULT
Resolve(const VECTOR_KERNEL_PLANNING_REQUEST &request) {
  return Resolve_vector_kernel_plan(
      request, Selection(request._requested_plan_kind), nullptr);
}

VECTOR_KERNEL_PLANNING_REQUEST With_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST &request,
    VECTOR_KERNEL_ATTRIBUTE_RECORD replacement,
    VECTOR_KERNEL_OPTION_SNAPSHOT options) {
  std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> attributes;
  attributes.reserve(request._attributes.size() + 1);
  bool replaced = false;
  for (const VECTOR_KERNEL_ATTRIBUTE_RECORD &attribute :
       request._attributes) {
    if (attribute._name == replacement._name) {
      attributes.push_back(replacement);
      replaced = true;
    } else {
      attributes.push_back(attribute);
    }
  }
  if (!replaced) attributes.push_back(std::move(replacement));
  return VECTOR_KERNEL_PLANNING_REQUEST{
      request._operation, std::move(attributes), request._operand_types,
      request._declared_result_type, options, request._target,
      request._source_constants, request._requested_plan_kind,
      request._runtime_scalar_types};
}

const VECTOR_KERNEL_TYPED_PAYLOAD *Find_payload(
    const PREPARED_VECTOR_KERNEL_PLAN &prepared, const std::string &role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : prepared.Constants()) {
    if (payload._role == role) return &payload;
  }
  return nullptr;
}

void Expect_loop(const VECTOR_KERNEL_LOOP_PLAN &loop, const std::string &role,
                 int32_t upper, uint32_t depth) {
  EXPECT_EQ(loop._role, role);
  EXPECT_EQ(loop._lower, 0);
  EXPECT_EQ(loop._upper, upper);
  EXPECT_EQ(loop._step, 1);
  EXPECT_EQ(loop._nesting_depth, depth);
}

void Expect_rotation(const VECTOR_KERNEL_ROTATION_PLAN &rotation,
                     const std::string &role,
                     const std::vector<int32_t> &candidates) {
  EXPECT_EQ(rotation._role, role);
  EXPECT_EQ(rotation._candidates, candidates);
}

void Expect_deterministic(const VECTOR_KERNEL_PLANNING_REQUEST &request,
                          VECTOR_KERNEL_PLAN_KIND expected_kind,
                          const std::vector<std::string> &expected_roles) {
  const VECTOR_KERNEL_RESOLUTION_RESULT first = Resolve(request);
  const VECTOR_KERNEL_RESOLUTION_RESULT second = Resolve(request);
  ASSERT_TRUE(first.Ok()) << first._diagnostic;
  ASSERT_TRUE(second.Ok()) << second._diagnostic;
  EXPECT_EQ(Get_vector_kernel_plan_kind(first._prepared->Plan()),
            expected_kind);
  EXPECT_EQ(Constant_roles(*first._prepared), expected_roles);
  EXPECT_EQ(first._prepared->Specialization_key(),
            second._prepared->Specialization_key());
  EXPECT_EQ(first._prepared->Helper_name(), second._prepared->Helper_name());
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(*first._prepared,
                                                     *second._prepared));
  ASSERT_EQ(first._prepared->Constants().size(),
            second._prepared->Constants().size());
  for (size_t idx = 0; idx < first._prepared->Constants().size(); ++idx) {
    EXPECT_EQ(first._prepared->Constants()[idx]._type._shape,
              second._prepared->Constants()[idx]._type._shape);
    EXPECT_EQ(first._prepared->Constants()[idx]._content_hash,
              second._prepared->Constants()[idx]._content_hash);
    EXPECT_EQ(first._prepared->Constants()[idx]._bytes,
              second._prepared->Constants()[idx]._bytes);
  }
}

VECTOR_KERNEL_PROVIDER_RESULT
Cpp_result(const VECTOR_KERNEL_PLANNING_REQUEST &request) {
  CPP_VECTOR_KERNEL_PLAN_PROVIDER provider;
  VECTOR_KERNEL_PROVIDER_CALL_RESULT call = provider.Plan(request);
  if (!call._result.has_value()) {
    throw std::runtime_error("C++ provider failed in test fixture: " +
                             call._diagnostic);
  }
  return std::move(*call._result);
}

class ALTERNATE_PROVENANCE_PROVIDER final
    : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char *Name() const override { return "alternate-conv-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT
  Plan(const VECTOR_KERNEL_PLANNING_REQUEST &request) const override {
    VECTOR_KERNEL_PROVIDER_RESULT result = Cpp_result(request);
    result._provenance = Name();
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(std::move(result));
  }
};

class FAIL_IF_CALLED_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char *Name() const override { return "fail-if-called"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT
  Plan(const VECTOR_KERNEL_PLANNING_REQUEST &) const override {
    ++_calls;
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(
        "provider should not be called");
  }

  uint32_t Calls() const { return _calls; }

private:
  mutable uint32_t _calls = 0;
};

VECTOR_KERNEL_COMMON_PLAN With_reductions(
    const VECTOR_KERNEL_COMMON_PLAN &common,
    std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops, common._slices,
      common._rotations, std::move(reductions), common._mask, common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_constants(
    const VECTOR_KERNEL_COMMON_PLAN &common,
    std::vector<VECTOR_KERNEL_CONSTANT_PLAN> constants) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      std::move(constants), common._result_type, common._loops,
      common._slices, common._rotations, common._reductions, common._mask,
      common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_rotations(
    const VECTOR_KERNEL_COMMON_PLAN &common,
    std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops, common._slices,
      std::move(rotations), common._reductions, common._mask, common._slot};
}

VECTOR_KERNEL_PROVIDER_RESULT Baseline_conv_result_with_common(
    const VECTOR_KERNEL_PROVIDER_RESULT &result,
    VECTOR_KERNEL_COMMON_PLAN common,
    std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> constants) {
  const BASELINE_CONV_PLAN &plan =
      std::get<BASELINE_CONV_PLAN>(result._plan);
  return VECTOR_KERNEL_PROVIDER_RESULT{
      VECTOR_KERNEL_PLAN{BASELINE_CONV_PLAN{
          std::move(common), plan._channel_in, plan._channel_out,
          plan._output_height, plan._output_width, plan._kernel_hw,
          plan._stride, plan._input_duplications}},
      std::move(constants), result._runtime_preparations,
      result._scalar_preparations, result._provenance};
}

VECTOR_KERNEL_PROVIDER_RESULT Fast_conv_result_with_common(
    const VECTOR_KERNEL_PROVIDER_RESULT &result,
    VECTOR_KERNEL_COMMON_PLAN common,
    std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> constants) {
  const FAST_CONV_PLAN &plan = std::get<FAST_CONV_PLAN>(result._plan);
  return VECTOR_KERNEL_PROVIDER_RESULT{
      VECTOR_KERNEL_PLAN{FAST_CONV_PLAN{
          std::move(common), plan._channel_in, plan._channel_out,
          plan._output_height, plan._output_width, plan._kernel_hw,
          plan._group, plan._stride, plan._input_size, plan._output_size,
          plan._num_slots, plan._num_grid, plan._num_block,
          plan._width_block, plan._width_block_data, plan._width_block_pad,
          plan._position_block, plan._capacity_block,
          plan._input_duplications, plan._blocking_outer_depth,
          plan._cyclic_roll, plan._sharding_offset}},
      std::move(constants), result._runtime_preparations,
      result._scalar_preparations, result._provenance};
}

VECTOR_KERNEL_PROVIDER_RESULT Fast_conv_result_with_common(
    const VECTOR_KERNEL_PROVIDER_RESULT &result,
    VECTOR_KERNEL_COMMON_PLAN common) {
  return Fast_conv_result_with_common(result, std::move(common),
                                      result._constants);
}

VECTOR_KERNEL_PROVIDER_RESULT Fast_conv_result_with_request_fields(
    const VECTOR_KERNEL_PROVIDER_RESULT &result, int64_t channel_in,
    int64_t group, int64_t stride) {
  const FAST_CONV_PLAN &plan = std::get<FAST_CONV_PLAN>(result._plan);
  return VECTOR_KERNEL_PROVIDER_RESULT{
      VECTOR_KERNEL_PLAN{FAST_CONV_PLAN{
          plan._common, channel_in, plan._channel_out, plan._output_height,
          plan._output_width, plan._kernel_hw, group, stride,
          plan._input_size, plan._output_size, plan._num_slots,
          plan._num_grid, plan._num_block, plan._width_block,
          plan._width_block_data, plan._width_block_pad,
          plan._position_block, plan._capacity_block,
          plan._input_duplications, plan._blocking_outer_depth,
          plan._cyclic_roll, plan._sharding_offset}},
      result._constants, result._runtime_preparations,
      result._scalar_preparations, result._provenance};
}

std::string Payload_hash(const VECTOR_KERNEL_RANKED_TYPE_PLAN &type,
                         const std::vector<uint8_t> &bytes) {
  return Build_vector_kernel_constant_hash(type._element_type, type._shape,
                                           bytes.data(), bytes.size());
}

void Expect_invalid_provider_result(
    const VECTOR_KERNEL_PLANNING_REQUEST &request,
    const VECTOR_KERNEL_PROVIDER_RESULT &provider_result,
    const std::string &diagnostic_fragment) {
  const VECTOR_KERNEL_PREPARE_RESULT result =
      Validate_and_prepare_vector_kernel_plan(request, provider_result);
  EXPECT_FALSE(result.Ok());
  EXPECT_EQ(result._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_PROVIDER_RESULT);
  EXPECT_NE(result._diagnostic.find(diagnostic_fragment), std::string::npos)
      << result._diagnostic;
}

} // namespace

TEST(Tensor2VectorConvPlanning,
     CppAutoBaselineAndFastValidateWithDeterministicOwnedConstants) {
  const VECTOR_KERNEL_PLANNING_REQUEST baseline =
      Conv_request(4, 2, 2, 1, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO);
  Expect_deterministic(baseline, VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV,
                       {"weight", "bias", "rotation-table"});
  const VECTOR_KERNEL_RESOLUTION_RESULT baseline_result = Resolve(baseline);
  ASSERT_TRUE(baseline_result.Ok()) << baseline_result._diagnostic;
  const BASELINE_CONV_PLAN &baseline_plan =
      std::get<BASELINE_CONV_PLAN>(baseline_result._prepared->Plan());
  EXPECT_EQ(baseline_plan._channel_in, 4);
  EXPECT_EQ(baseline_plan._channel_out, 2);
  EXPECT_EQ(baseline_plan._output_height, 2);
  EXPECT_EQ(baseline_plan._output_width, 2);
  EXPECT_EQ(baseline_plan._kernel_hw, 1);
  EXPECT_EQ(baseline_plan._stride, 1);
  EXPECT_EQ(baseline_plan._input_duplications, 2);
  ASSERT_EQ(baseline_plan._common._loops.size(), 2U);
  Expect_loop(baseline_plan._common._loops[0], "channel-in", 4, 0);
  Expect_loop(baseline_plan._common._loops[1], "kernel-hw", 1, 1);
  ASSERT_EQ(baseline_plan._common._slices.size(), 1U);
  EXPECT_EQ(baseline_plan._common._slices[0]._role, "weight");
  EXPECT_EQ(baseline_plan._common._slices[0]._index._iv_coefficients,
            (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(baseline_plan._common._slices[0]._width, 8);
  ASSERT_EQ(baseline_plan._common._rotations.size(), 3U);
  Expect_rotation(baseline_plan._common._rotations[0],
                  "input-duplication", {-16});
  Expect_rotation(baseline_plan._common._rotations[1],
                  "kernel-alignment", {0});
  Expect_rotation(baseline_plan._common._rotations[2], "channel-step", {4});
  EXPECT_TRUE(baseline_plan._common._reductions.empty());
  EXPECT_EQ(baseline_plan._common._mask._policy,
            VECTOR_KERNEL_MASK_POLICY::NONE);
  EXPECT_EQ(baseline_plan._common._slot._policy,
            VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS);
  EXPECT_EQ(baseline_plan._common._slot._value, 8U);
  ASSERT_EQ(baseline_result._prepared->Runtime_preparations().size(), 1U);
  const VECTOR_KERNEL_RUNTIME_PREPARATION &baseline_input =
      baseline_result._prepared->Runtime_preparations()[0];
  EXPECT_EQ(baseline_input._kind,
            VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::FLATTEN_PACKED_VECTOR);
  EXPECT_EQ(baseline_input._result_type._shape,
            (std::vector<int64_t>{16}));
  EXPECT_EQ(baseline_input._logical_input_size, 16);
  EXPECT_EQ(baseline_input._replications, 2);
  EXPECT_EQ(baseline_input._rotation_candidates,
            (std::vector<int32_t>{-16}));
  const VECTOR_KERNEL_TYPED_PAYLOAD *baseline_weight =
      Find_payload(*baseline_result._prepared, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD *baseline_bias =
      Find_payload(*baseline_result._prepared, "bias");
  const VECTOR_KERNEL_TYPED_PAYLOAD *baseline_rotation =
      Find_payload(*baseline_result._prepared, "rotation-table");
  ASSERT_NE(baseline_weight, nullptr);
  ASSERT_NE(baseline_bias, nullptr);
  ASSERT_NE(baseline_rotation, nullptr);
  EXPECT_EQ(baseline_weight->_type._shape,
            (std::vector<int64_t>{4, 8}));
  EXPECT_EQ(baseline_weight->_content_hash,
            "sha256:d84150537daa7139bc0cf4beaaff2423ad6ffffbc4d0ff4b7fb725cff3648d0f");
  EXPECT_EQ(baseline_bias->_content_hash,
            "sha256:ff206fd9aa2132b161cee3b6849e42de6edb881891903d4da580cad47b484498");
  EXPECT_EQ(baseline_rotation->_bytes, Int32_bytes({0}));
  EXPECT_EQ(baseline_rotation->_content_hash,
            "sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf");
  const std::string baseline_key =
      "vector-kernel-plan:v1|kind=baseline-conv|vector-inputs=1{f32[16]}|scalar-inputs=0{}|constants=3{6:weight=f32[4,8]#71:sha256:d84150537daa7139bc0cf4beaaff2423ad6ffffbc4d0ff4b7fb725cff3648d0f;4:bias=f32[8]#71:sha256:ff206fd9aa2132b161cee3b6849e42de6edb881891903d4da580cad47b484498;14:rotation-table=s32[1]#71:sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf}|result=f32[1,2,2,2]|loops=2{10:channel-in(0,4,1,0);9:kernel-hw(0,1,1,1)}|slices=1{6:weight([1,1],0,0,8)}|rotations=3{17:input-duplication[-16];16:kernel-alignment[0];12:channel-step[4]}|reductions=0{}|mask=none:0|slot=logical-output-elements:8|channel-in=4|channel-out=2|output-height=2|output-width=2|kernel-hw=1|stride=1|input-duplications=2";
  EXPECT_EQ(baseline_result._prepared->Specialization_key(), baseline_key);
  EXPECT_EQ(baseline_result._prepared->Helper_name(),
            "__ace_vkernel_baseline_conv_473c5b9903ee8edfcfb2582a045ee8c065ed5c1ae44d2ba0c0eae69970aef18b");
  VECTOR_KERNEL_PLANNING_REQUEST forced_baseline = Conv_request(
      4, 2, 2, 1, 32,
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV);
  const VECTOR_KERNEL_RESOLUTION_RESULT forced_baseline_result =
      Resolve(forced_baseline);
  ASSERT_TRUE(forced_baseline_result.Ok())
      << forced_baseline_result._diagnostic;
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
      *baseline_result._prepared, *forced_baseline_result._prepared));

  const VECTOR_KERNEL_PLANNING_REQUEST fast =
      Conv_request(2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO);
  Expect_deterministic(fast, VECTOR_KERNEL_PLAN_KIND::FAST_CONV,
                       {"weight", "bias", "rotation-table", "collective-mask"});
  const VECTOR_KERNEL_RESOLUTION_RESULT fast_result = Resolve(fast);
  ASSERT_TRUE(fast_result.Ok()) << fast_result._diagnostic;
  const FAST_CONV_PLAN &fast_plan =
      std::get<FAST_CONV_PLAN>(fast_result._prepared->Plan());
  EXPECT_FALSE(fast_plan._cyclic_roll);
  EXPECT_EQ(fast_plan._channel_in, 2);
  EXPECT_EQ(fast_plan._channel_out, 4);
  EXPECT_EQ(fast_plan._output_height, 2);
  EXPECT_EQ(fast_plan._output_width, 2);
  EXPECT_EQ(fast_plan._kernel_hw, 9);
  EXPECT_EQ(fast_plan._group, 1);
  EXPECT_EQ(fast_plan._stride, 1);
  EXPECT_EQ(fast_plan._input_size, 8);
  EXPECT_EQ(fast_plan._output_size, 16);
  EXPECT_EQ(fast_plan._num_slots, 32);
  EXPECT_EQ(fast_plan._num_grid, 2);
  EXPECT_EQ(fast_plan._num_block, 1);
  EXPECT_EQ(fast_plan._width_block, 16);
  EXPECT_EQ(fast_plan._width_block_data, 16);
  EXPECT_EQ(fast_plan._width_block_pad, 0);
  EXPECT_EQ(fast_plan._position_block, 4);
  EXPECT_EQ(fast_plan._capacity_block, 9);
  EXPECT_EQ(fast_plan._input_duplications, 2);
  EXPECT_EQ(fast_plan._blocking_outer_depth, 0);
  EXPECT_FALSE(fast_plan._sharding_offset.has_value());
  ASSERT_EQ(fast_plan._common._loops.size(), 2U);
  Expect_loop(fast_plan._common._loops[0], "grid", 2, 0);
  Expect_loop(fast_plan._common._loops[1], "capacity-block", 9, 1);
  ASSERT_EQ(fast_plan._common._slices.size(), 1U);
  EXPECT_EQ(fast_plan._common._slices[0]._index._iv_coefficients,
            (std::vector<int64_t>{9, 1}));
  EXPECT_EQ(fast_plan._common._slices[0]._width, 16);
  ASSERT_EQ(fast_plan._common._rotations.size(), 3U);
  Expect_rotation(fast_plan._common._rotations[0], "blocking-alignment",
                  {-3, -2, -1, -1, 0, 1, 1, 2, 3});
  Expect_rotation(fast_plan._common._rotations[1], "grid", {0, 4});
  Expect_rotation(fast_plan._common._rotations[2], "collective-tail",
                  {-16});
  ASSERT_EQ(fast_plan._common._reductions.size(), 1U);
  EXPECT_EQ(fast_plan._common._reductions[0]._kind,
            VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK);
  EXPECT_EQ(fast_plan._common._reductions[0]._factor, 1);
  EXPECT_EQ(fast_plan._common._reductions[0]._block_width, 16);
  EXPECT_EQ(fast_plan._common._mask._policy,
            VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION);
  EXPECT_EQ(fast_plan._common._mask._valid_length, 16);
  ASSERT_EQ(fast_result._prepared->Runtime_preparations().size(), 1U);
  const VECTOR_KERNEL_RUNTIME_PREPARATION &fast_input =
      fast_result._prepared->Runtime_preparations()[0];
  EXPECT_EQ(fast_input._kind,
            VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS);
  EXPECT_EQ(fast_input._result_type._shape, (std::vector<int64_t>{8}));
  EXPECT_EQ(fast_input._logical_input_size, 8);
  EXPECT_EQ(fast_input._replications, 2);
  EXPECT_EQ(fast_input._blocking_width, 9);
  EXPECT_EQ(fast_input._rotation_candidates,
            (std::vector<int32_t>{-3, -2, -1, -1, 0, 1, 1, 2, 3}));
  const VECTOR_KERNEL_TYPED_PAYLOAD *fast_weight =
      Find_payload(*fast_result._prepared, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD *fast_bias =
      Find_payload(*fast_result._prepared, "bias");
  const VECTOR_KERNEL_TYPED_PAYLOAD *fast_rotation =
      Find_payload(*fast_result._prepared, "rotation-table");
  const VECTOR_KERNEL_TYPED_PAYLOAD *fast_mask =
      Find_payload(*fast_result._prepared, "collective-mask");
  ASSERT_NE(fast_weight, nullptr);
  ASSERT_NE(fast_bias, nullptr);
  ASSERT_NE(fast_rotation, nullptr);
  ASSERT_NE(fast_mask, nullptr);
  EXPECT_EQ(fast_weight->_type._shape,
            (std::vector<int64_t>{18, 16}));
  EXPECT_EQ(fast_weight->_content_hash,
            "sha256:b8609c354e6f0e1c23fb8284fec88e0c30f6a1fdef6e27622bd6d2e2377ab28a");
  EXPECT_EQ(fast_bias->_content_hash,
            "sha256:02ce240b0e375bc06004c664427910ae5b8da03401d7401ad9aeba00ab9b48f6");
  EXPECT_EQ(fast_rotation->_bytes,
            Int32_bytes({-3, -2, -1, -1, 0, 1, 1, 2, 3}));
  EXPECT_EQ(fast_rotation->_content_hash,
            "sha256:d5e848269a15cc35ba1af9f9b4e30e2b303ed2a1c3bf8f2390dbb499653ac5ae");
  EXPECT_EQ(fast_mask->_content_hash,
            "sha256:14f0eda2f8622a12587e1de0185d48afb7ee5d7ea18bc9fd09e8eb3c0c36b77d");
  const std::string expected_fast_key =
      "vector-kernel-plan:v1|kind=fast-conv|vector-inputs=1{f32[8]}|scalar-inputs=0{}|constants=4{6:weight=f32[18,16]#71:sha256:b8609c354e6f0e1c23fb8284fec88e0c30f6a1fdef6e27622bd6d2e2377ab28a;4:bias=f32[16]#71:sha256:02ce240b0e375bc06004c664427910ae5b8da03401d7401ad9aeba00ab9b48f6;14:rotation-table=s32[9]#71:sha256:d5e848269a15cc35ba1af9f9b4e30e2b303ed2a1c3bf8f2390dbb499653ac5ae;15:collective-mask=f32[16]#71:sha256:14f0eda2f8622a12587e1de0185d48afb7ee5d7ea18bc9fd09e8eb3c0c36b77d}|result=f32[1,4,2,2]|loops=2{4:grid(0,2,1,0);14:capacity-block(0,9,1,1)}|slices=1{6:weight([9,1],0,0,16)}|rotations=3{18:blocking-alignment[-3,-2,-1,-1,0,1,1,2,3];4:grid[0,4];15:collective-tail[-16]}|reductions=1{10:collective(collective-single-block,1,16,0)}|mask=collective-reduction:16|slot=logical-output-elements:16|channel-in=2|channel-out=4|output-height=2|output-width=2|kernel-hw=9|group=1|stride=1|input-size=8|output-size=16|num-slots=32|num-grid=2|num-block=1|width-block=16|width-block-data=16|width-block-pad=0|position-block=4|capacity-block=9|input-duplications=2|blocking-outer-depth=0|cyclic-roll=0|sharding-offset=none";
  EXPECT_EQ(fast_result._prepared->Specialization_key(), expected_fast_key);
  EXPECT_EQ(fast_result._prepared->Helper_name(),
            "__ace_vkernel_fast_conv_074ae83c0a347d08b927303ee5a02e12c129798e6d672ea68fd3a2f96bda119b");
  VECTOR_KERNEL_PLANNING_REQUEST forced_fast = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV);
  const VECTOR_KERNEL_RESOLUTION_RESULT forced_fast_result =
      Resolve(forced_fast);
  ASSERT_TRUE(forced_fast_result.Ok()) << forced_fast_result._diagnostic;
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
      *fast_result._prepared, *forced_fast_result._prepared));
}

TEST(Tensor2VectorConvPlanning,
     BaselineIdentityIgnoresProviderAndProvenance) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      4, 2, 2, 1, 32,
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV);
  const VECTOR_KERNEL_RESOLUTION_RESULT cpp = Resolve(request);
  ASSERT_TRUE(cpp.Ok()) << cpp._diagnostic;

  ALTERNATE_PROVENANCE_PROVIDER alternate_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY registry;
  ASSERT_TRUE(registry.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                                &alternate_provider));
  const VECTOR_KERNEL_SELECTION selection{
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
      VECTOR_KERNEL_IMPLEMENTATION::DSL,
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV,
      VECTOR_KERNEL_FALLBACK_POLICY::ERROR};
  const VECTOR_KERNEL_RESOLUTION_RESULT alternate =
      Resolve_vector_kernel_plan(
          request, selection, &registry,
          [](VECTOR_KERNEL_PLAN_KIND kind) {
            return kind == VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV;
          });
  ASSERT_TRUE(alternate.Ok()) << alternate._diagnostic;

  EXPECT_EQ(cpp._prepared->Provenance(), "cpp");
  EXPECT_EQ(alternate._prepared->Provenance(), "alternate-conv-test");
  EXPECT_EQ(alternate._resolved_provider,
            VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON);
  EXPECT_EQ(alternate._resolved_implementation,
            VECTOR_KERNEL_IMPLEMENTATION::DSL);
  EXPECT_EQ(cpp._prepared->Specialization_key(),
            alternate._prepared->Specialization_key());
  EXPECT_EQ(cpp._prepared->Helper_name(),
            alternate._prepared->Helper_name());
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
      *cpp._prepared, *alternate._prepared));
}

TEST(Tensor2VectorConvPlanning,
     ValidatorRejectsNoncanonicalBaselineDuplicationPolicy) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      4, 2, 2, 1, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const BASELINE_CONV_PLAN &plan =
      std::get<BASELINE_CONV_PLAN>(valid._plan);
  const VECTOR_KERNEL_PROVIDER_RESULT changed{
      VECTOR_KERNEL_PLAN{BASELINE_CONV_PLAN{
          plan._common, plan._channel_in, plan._channel_out,
          plan._output_height, plan._output_width, plan._kernel_hw,
          plan._stride, plan._input_duplications + 1}},
      valid._constants,
      valid._runtime_preparations,
      valid._scalar_preparations,
      valid._provenance};

  Expect_invalid_provider_result(request, changed,
                                 "duplication policy is inconsistent");
}

TEST(Tensor2VectorConvPlanning,
     ValidatorBindsBaselineChannelsStrideWeightRowsAndRotationsToRequest) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      4, 2, 2, 1, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const BASELINE_CONV_PLAN &plan =
      std::get<BASELINE_CONV_PLAN>(valid._plan);

  const VECTOR_KERNEL_PROVIDER_RESULT changed_channel{
      VECTOR_KERNEL_PLAN{BASELINE_CONV_PLAN{
          plan._common, plan._channel_in + 1, plan._channel_out,
          plan._output_height, plan._output_width, plan._kernel_hw,
          plan._stride, plan._input_duplications}},
      valid._constants, valid._runtime_preparations,
      valid._scalar_preparations, valid._provenance};
  Expect_invalid_provider_result(request, changed_channel,
                                 "invalid baseline Conv plan invariants");

  const VECTOR_KERNEL_PROVIDER_RESULT changed_stride{
      VECTOR_KERNEL_PLAN{BASELINE_CONV_PLAN{
          plan._common, plan._channel_in, plan._channel_out,
          plan._output_height, plan._output_width, plan._kernel_hw,
          plan._stride + 1, plan._input_duplications}},
      valid._constants, valid._runtime_preparations,
      valid._scalar_preparations, valid._provenance};
  Expect_invalid_provider_result(request, changed_stride,
                                 "invalid baseline Conv plan invariants");

  const VECTOR_KERNEL_TYPED_PAYLOAD *weight_payload = nullptr;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    if (payload._role == "weight") weight_payload = &payload;
  }
  ASSERT_NE(weight_payload, nullptr);
  ASSERT_GT(weight_payload->_type._shape[0], 1);
  VECTOR_KERNEL_RANKED_TYPE_PLAN smaller_weight_type{
      weight_payload->_type._element_type,
      {weight_payload->_type._shape[0] - 1,
       weight_payload->_type._shape[1]}};
  std::vector<uint8_t> smaller_weight_bytes = weight_payload->_bytes;
  smaller_weight_bytes.resize(
      static_cast<size_t>(smaller_weight_type._shape[0] *
                          smaller_weight_type._shape[1] * sizeof(float)));
  const std::string smaller_weight_hash =
      Payload_hash(smaller_weight_type, smaller_weight_bytes);
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> smaller_payloads;
  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> smaller_descriptors;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    smaller_payloads.push_back(
        payload._role == "weight"
            ? VECTOR_KERNEL_TYPED_PAYLOAD{
                  payload._role, smaller_weight_type, smaller_weight_bytes,
                  smaller_weight_hash}
            : payload);
  }
  for (const VECTOR_KERNEL_CONSTANT_PLAN &descriptor :
       plan._common._constants) {
    smaller_descriptors.push_back(
        descriptor._role == "weight"
            ? VECTOR_KERNEL_CONSTANT_PLAN{
                  descriptor._role, smaller_weight_type,
                  smaller_weight_hash}
            : descriptor);
  }
  Expect_invalid_provider_result(
      request,
      Baseline_conv_result_with_common(
          valid, With_constants(plan._common, smaller_descriptors),
          smaller_payloads),
      "weight descriptor is inconsistent");

  auto rotation = std::find_if(
      plan._common._rotations.begin(), plan._common._rotations.end(),
      [](const VECTOR_KERNEL_ROTATION_PLAN &record) {
        return record._role == "kernel-alignment";
      });
  ASSERT_NE(rotation, plan._common._rotations.end());
  ASSERT_FALSE(rotation->_candidates.empty());
  std::vector<int32_t> changed_candidates = rotation->_candidates;
  ++changed_candidates[0];
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> changed_rotations;
  for (const VECTOR_KERNEL_ROTATION_PLAN &record :
       plan._common._rotations) {
    changed_rotations.push_back(
        record._role == "kernel-alignment"
            ? VECTOR_KERNEL_ROTATION_PLAN{record._role, changed_candidates}
            : record);
  }
  VECTOR_KERNEL_RANKED_TYPE_PLAN rotation_type{
      PRIMITIVE_TYPE::INT_S32,
      {static_cast<int64_t>(changed_candidates.size())}};
  const std::vector<uint8_t> rotation_bytes = Int32_bytes(changed_candidates);
  const std::string rotation_hash = Payload_hash(rotation_type, rotation_bytes);
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> rotation_payloads;
  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> rotation_descriptors;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    rotation_payloads.push_back(
        payload._role == "rotation-table"
            ? VECTOR_KERNEL_TYPED_PAYLOAD{payload._role, rotation_type,
                                          rotation_bytes, rotation_hash}
            : payload);
  }
  for (const VECTOR_KERNEL_CONSTANT_PLAN &descriptor :
       plan._common._constants) {
    rotation_descriptors.push_back(
        descriptor._role == "rotation-table"
            ? VECTOR_KERNEL_CONSTANT_PLAN{descriptor._role, rotation_type,
                                          rotation_hash}
            : descriptor);
  }
  VECTOR_KERNEL_COMMON_PLAN changed_common = With_rotations(
      With_constants(plan._common, std::move(rotation_descriptors)),
      std::move(changed_rotations));
  Expect_invalid_provider_result(
      request,
      Baseline_conv_result_with_common(valid, std::move(changed_common),
                                       std::move(rotation_payloads)),
      "kernel rotations are inconsistent");
}

TEST(Tensor2VectorConvPlanning,
     ValidatorBindsFastChannelsGroupStrideAndWeightRowsToRequest) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const FAST_CONV_PLAN &plan = std::get<FAST_CONV_PLAN>(valid._plan);

  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_request_fields(
          valid, plan._channel_in + 1, plan._group, plan._stride),
      "invalid fast Conv plan invariants");
  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_request_fields(
          valid, plan._channel_in, plan._group + 1, plan._stride),
      "invalid fast Conv plan invariants");
  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_request_fields(
          valid, plan._channel_in, plan._group, plan._stride + 1),
      "invalid fast Conv plan invariants");

  const VECTOR_KERNEL_TYPED_PAYLOAD *weight_payload = nullptr;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    if (payload._role == "weight") weight_payload = &payload;
  }
  ASSERT_NE(weight_payload, nullptr);
  ASSERT_GT(weight_payload->_type._shape[0], 1);
  VECTOR_KERNEL_RANKED_TYPE_PLAN smaller_weight_type{
      weight_payload->_type._element_type,
      {weight_payload->_type._shape[0] - 1,
       weight_payload->_type._shape[1]}};
  std::vector<uint8_t> smaller_weight_bytes = weight_payload->_bytes;
  smaller_weight_bytes.resize(
      static_cast<size_t>(smaller_weight_type._shape[0] *
                          smaller_weight_type._shape[1] * sizeof(float)));
  const std::string smaller_weight_hash =
      Payload_hash(smaller_weight_type, smaller_weight_bytes);
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> smaller_payloads;
  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> smaller_descriptors;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    smaller_payloads.push_back(
        payload._role == "weight"
            ? VECTOR_KERNEL_TYPED_PAYLOAD{
                  payload._role, smaller_weight_type, smaller_weight_bytes,
                  smaller_weight_hash}
            : payload);
  }
  for (const VECTOR_KERNEL_CONSTANT_PLAN &descriptor :
       plan._common._constants) {
    smaller_descriptors.push_back(
        descriptor._role == "weight"
            ? VECTOR_KERNEL_CONSTANT_PLAN{
                  descriptor._role, smaller_weight_type,
                  smaller_weight_hash}
            : descriptor);
  }
  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_common(
          valid, With_constants(plan._common, smaller_descriptors),
          smaller_payloads),
      "weight descriptor is inconsistent");
}

TEST(Tensor2VectorConvPlanning,
     ValidatorRejectsWrongShardedSourceScalarTypeBeforeProviderOutput) {
  const VECTOR_KERNEL_PLANNING_REQUEST valid_request = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true);
  const VECTOR_KERNEL_PROVIDER_RESULT valid_result = Cpp_result(valid_request);
  const VECTOR_KERNEL_PLANNING_REQUEST wrong_scalar_request{
      valid_request._operation, valid_request._attributes,
      valid_request._operand_types, valid_request._declared_result_type,
      valid_request._options, valid_request._target,
      valid_request._source_constants, valid_request._requested_plan_kind,
      {PRIMITIVE_TYPE::INT_S64}};
  const VECTOR_KERNEL_PREPARE_RESULT rejected =
      Validate_and_prepare_vector_kernel_plan(wrong_scalar_request,
                                              valid_result);
  EXPECT_FALSE(rejected.Ok());
  EXPECT_EQ(rejected._error, VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_NE(rejected._diagnostic.find("one s32 weight offset"),
            std::string::npos)
      << rejected._diagnostic;
}

TEST(Tensor2VectorConvPlanning,
     MaskRequiresExactlyOnePositiveU32WhenFusionIsEnabled) {
  for (VECTOR_KERNEL_REQUESTED_PLAN_KIND kind :
       {VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV,
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV}) {
    SCOPED_TRACE(Vector_kernel_requested_plan_kind_name(kind));
    const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
        2, 4, 2, 3, 32, kind);
    const VECTOR_KERNEL_OPTION_SNAPSHOT options{false, true, false, false,
                                                false};
    const VECTOR_KERNEL_RESOLUTION_RESULT positive = Resolve(With_attribute(
        request, Uint32_attribute("mask", {7U}), options));
    ASSERT_TRUE(positive.Ok()) << positive._diagnostic;

    for (VECTOR_KERNEL_ATTRIBUTE_RECORD invalid :
         {Uint32_attribute("mask", {0U}),
          Uint32_attribute("mask", {1U, 2U}),
          Int32_attribute("mask", {1})}) {
      const VECTOR_KERNEL_RESOLUTION_RESULT rejected =
          Resolve(With_attribute(request, std::move(invalid), options));
      EXPECT_FALSE(rejected.Ok());
      EXPECT_EQ(rejected._error,
                VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
      EXPECT_NE(rejected._diagnostic.find("exactly one positive u32"),
                std::string::npos)
          << rejected._diagnostic;
    }
  }
}

TEST(Tensor2VectorConvPlanning,
     RejectsUnsupportedShardedParallelFusionBeforeProviderLookup) {
  const VECTOR_KERNEL_OPTION_SNAPSHOT options{true, false, false, true,
                                              false};
  const VECTOR_KERNEL_PLANNING_REQUEST supported = With_attribute(
      Conv_request(2, 4, 2, 3, 32,
                   VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true),
      Int32_attribute("group", {1}), options);
  const VECTOR_KERNEL_RESOLUTION_RESULT supported_result = Resolve(supported);
  ASSERT_TRUE(supported_result.Ok()) << supported_result._diagnostic;

  const VECTOR_KERNEL_PLANNING_REQUEST unsupported = With_attribute(
      Conv_request(32, 32, 2, 3, 512,
                   VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true),
      Int32_attribute("group", {1}), options);
  FAIL_IF_CALLED_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY registry;
  ASSERT_TRUE(registry.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                &provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT rejected = Resolve_vector_kernel_plan(
      unsupported,
      Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV), &registry);
  EXPECT_FALSE(rejected.Ok());
  EXPECT_EQ(rejected._error, VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_NE(rejected._diagnostic.find(
                "does not support multiple parallel fusion blocks"),
            std::string::npos)
      << rejected._diagnostic;
  EXPECT_EQ(provider.Calls(), 0U);
}

TEST(Tensor2VectorConvPlanning,
     CppProviderAndCentralValidationRejectConvArithmeticOverflow) {
  const VECTOR_KERNEL_PLANNING_REQUEST sharded = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true);
  const VECTOR_KERNEL_PLANNING_REQUEST huge_halo = With_attribute(
      sharded,
      Int64_attribute("sharding",
                      {2, 1, 2, std::numeric_limits<int64_t>::max()}),
      sharded._options);
  CPP_VECTOR_KERNEL_PLAN_PROVIDER provider;
  const VECTOR_KERNEL_PROVIDER_CALL_RESULT provider_halo =
      provider.Plan(huge_halo);
  EXPECT_FALSE(provider_halo._result.has_value());
  EXPECT_NE(provider_halo._diagnostic.find("halo size overflows"),
            std::string::npos)
      << provider_halo._diagnostic;
  const VECTOR_KERNEL_RESOLUTION_RESULT central_halo = Resolve(huge_halo);
  EXPECT_FALSE(central_halo.Ok());
  EXPECT_EQ(central_halo._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);

  const VECTOR_KERNEL_PLANNING_REQUEST baseline = Conv_request(
      2, 4, 2, 3, 32,
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV);
  const VECTOR_KERNEL_PLANNING_REQUEST huge_stride = With_attribute(
      baseline,
      Int64_attribute("strides", {std::numeric_limits<int64_t>::max(),
                                   std::numeric_limits<int64_t>::max()}),
      VECTOR_KERNEL_OPTION_SNAPSHOT{false, false, true, false, false});
  const VECTOR_KERNEL_PROVIDER_CALL_RESULT provider_stride =
      provider.Plan(huge_stride);
  EXPECT_FALSE(provider_stride._result.has_value());
  EXPECT_NE(provider_stride._diagnostic.find(
                "stride-scaled rotation overflows"),
            std::string::npos)
      << provider_stride._diagnostic;
  const VECTOR_KERNEL_RESOLUTION_RESULT central_stride =
      Resolve(huge_stride);
  EXPECT_FALSE(central_stride.Ok());
  EXPECT_EQ(central_stride._error,
            VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE);
  EXPECT_NE(central_stride._diagnostic.find(
                "stride-scaled rotation overflows"),
            std::string::npos)
      << central_stride._diagnostic;
}

TEST(Tensor2VectorConvPlanning,
     CppForcedFastCyclicOwnsMasksAndCompleteRotationTopology) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      2, 3, 3, 1, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV);
  Expect_deterministic(request, VECTOR_KERNEL_PLAN_KIND::FAST_CONV,
                       {"weight", "bias", "rotation-table", "cyclic-mask-left",
                        "cyclic-mask-right"});
  const VECTOR_KERNEL_RESOLUTION_RESULT result = Resolve(request);
  ASSERT_TRUE(result.Ok()) << result._diagnostic;
  const FAST_CONV_PLAN &plan =
      std::get<FAST_CONV_PLAN>(result._prepared->Plan());
  EXPECT_TRUE(plan._cyclic_roll);
  EXPECT_TRUE(plan._common._reductions.empty());
  EXPECT_EQ(plan._common._mask._policy, VECTOR_KERNEL_MASK_POLICY::NONE);
  ASSERT_EQ(plan._common._rotations.size(), 3U);
  EXPECT_EQ(plan._common._rotations[0]._role, "blocking-alignment");
  EXPECT_EQ(plan._common._rotations[1]._role, "cyclic-left");
  EXPECT_EQ(plan._common._rotations[2]._role, "cyclic-right");
  ASSERT_EQ(plan._common._slices.size(), 3U);
  EXPECT_EQ(plan._common._slices[1]._role, "cyclic-mask-left");
  EXPECT_EQ(plan._common._slices[2]._role, "cyclic-mask-right");
}

TEST(Tensor2VectorConvPlanning,
     CppForcedFastShardedOwnsTransposedWeightAndScalarDescriptor) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true);
  Expect_deterministic(request, VECTOR_KERNEL_PLAN_KIND::FAST_CONV,
                       {"weight", "bias", "rotation-table", "collective-mask",
                        "collective-gap-mask"});
  const VECTOR_KERNEL_RESOLUTION_RESULT result = Resolve(request);
  ASSERT_TRUE(result.Ok()) << result._diagnostic;
  const FAST_CONV_PLAN &plan =
      std::get<FAST_CONV_PLAN>(result._prepared->Plan());
  ASSERT_TRUE(plan._sharding_offset.has_value());
  EXPECT_EQ(plan._sharding_offset->_type, PRIMITIVE_TYPE::INT_S32);
  EXPECT_EQ(plan._sharding_offset->_scale, 18);
  EXPECT_EQ(plan._blocking_outer_depth, 1);
  ASSERT_EQ(result._prepared->Scalar_preparations().size(), 1U);
  EXPECT_EQ(result._prepared->Scalar_preparations()[0]._role, "weight-offset");
  EXPECT_EQ(result._prepared->Scalar_preparations()[0]._source_operand, 1U);
  EXPECT_EQ(result._prepared->Scalar_preparations()[0]._scale, 18);
  ASSERT_FALSE(result._prepared->Constants().empty());
  EXPECT_EQ(result._prepared->Constants()[0]._type._shape,
            (std::vector<int64_t>{36, 16}));
}

TEST(Tensor2VectorConvPlanning,
     ValidatorRejectsFastConvCollectiveTopologyAndPaddingChanges) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const VECTOR_KERNEL_PREPARE_RESULT valid_result =
      Validate_and_prepare_vector_kernel_plan(request, valid);
  ASSERT_TRUE(valid_result.Ok()) << valid_result._diagnostic;
  const FAST_CONV_PLAN &plan = std::get<FAST_CONV_PLAN>(valid._plan);
  ASSERT_EQ(plan._common._reductions.size(), 1U);
  const VECTOR_KERNEL_REDUCTION_PLAN &reduction =
      plan._common._reductions.front();

  const VECTOR_KERNEL_REDUCTION_KIND changed_kind =
      reduction._kind == VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS
          ? VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK
          : VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS;
  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> changed_topology{
      VECTOR_KERNEL_REDUCTION_PLAN{
          reduction._role, changed_kind, reduction._factor,
          reduction._block_width, reduction._padding}};
  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_common(
          valid,
          With_reductions(plan._common, std::move(changed_topology))),
      "reductions");

  const int64_t changed_padding =
      reduction._padding == plan._output_size ? reduction._padding - 1
                                              : reduction._padding + 1;
  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> changed_padding_records{
      VECTOR_KERNEL_REDUCTION_PLAN{
          reduction._role, reduction._kind, reduction._factor,
          reduction._block_width, changed_padding}};
  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_common(
          valid, With_reductions(plan._common,
                                 std::move(changed_padding_records))),
      "reductions");
}

TEST(Tensor2VectorConvPlanning,
     ValidatorRejectsNoncanonicalFastConvAuxiliaryMaskBytes) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Conv_request(
      2, 4, 2, 3, 32, VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV, true);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const FAST_CONV_PLAN &plan = std::get<FAST_CONV_PLAN>(valid._plan);
  ASSERT_EQ(plan._common._reductions.size(), 1U);

  const VECTOR_KERNEL_TYPED_PAYLOAD *gap_payload = nullptr;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    if (payload._role == "collective-gap-mask") {
      gap_payload = &payload;
      break;
    }
  }
  ASSERT_NE(gap_payload, nullptr);
  ASSERT_GE(gap_payload->_bytes.size(), sizeof(float));
  const int64_t padding = plan._common._reductions.front()._padding;
  ASSERT_GE(padding, 0);
  ASSERT_LE(padding, plan._output_size);

  std::vector<uint8_t> changed_bytes = gap_payload->_bytes;
  const bool first_element_is_one = padding == plan._output_size;
  const std::vector<uint8_t> replacement =
      Float_bytes({first_element_is_one ? 0.0F : 1.0F});
  for (size_t idx = 0; idx < sizeof(float); ++idx) {
    changed_bytes[idx] = replacement[idx];
  }
  const std::string changed_hash =
      Payload_hash(gap_payload->_type, changed_bytes);

  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> changed_payloads;
  changed_payloads.reserve(valid._constants.size());
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : valid._constants) {
    if (payload._role == gap_payload->_role) {
      changed_payloads.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
          payload._role, payload._type, changed_bytes, changed_hash});
    } else {
      changed_payloads.push_back(payload);
    }
  }
  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> changed_descriptors;
  changed_descriptors.reserve(plan._common._constants.size());
  for (const VECTOR_KERNEL_CONSTANT_PLAN &descriptor :
       plan._common._constants) {
    changed_descriptors.push_back(VECTOR_KERNEL_CONSTANT_PLAN{
        descriptor._role, descriptor._type,
        descriptor._role == gap_payload->_role ? changed_hash
                                                : descriptor._content_hash});
  }

  Expect_invalid_provider_result(
      request,
      Fast_conv_result_with_common(
          valid,
          With_constants(plan._common, std::move(changed_descriptors)),
          std::move(changed_payloads)),
      "gap mask payload is noncanonical");
}
