//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "nn/vector/tensor2vector_planning.h"

using air::base::PRIMITIVE_TYPE;
using namespace nn::vector;

namespace {

std::vector<uint8_t> Float_bytes(const std::vector<float>& values) {
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

template <typename T>
std::vector<uint8_t> Integer_bytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(T));
  for (T value : values) {
    using UNSIGNED = std::make_unsigned_t<T>;
    const UNSIGNED bits = static_cast<UNSIGNED>(value);
    for (uint32_t shift = 0; shift != sizeof(T) * 8; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

template <typename T>
VECTOR_KERNEL_ATTRIBUTE_RECORD Integer_attribute(
    std::string name, PRIMITIVE_TYPE type, const std::vector<T>& values) {
  return VECTOR_KERNEL_ATTRIBUTE_RECORD{
      std::move(name),
      VECTOR_KERNEL_RANKED_TYPE_PLAN{
          type, {static_cast<int64_t>(values.size())}},
      Integer_bytes(values)};
}

VECTOR_KERNEL_TYPED_PAYLOAD Float_payload(
    std::string role, std::vector<int64_t> shape,
    const std::vector<float>& values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::FLOAT_32,
                                      std::move(shape)};
  std::vector<uint8_t> bytes = Float_bytes(values);
  const std::string hash = Build_vector_kernel_constant_hash(
      type._element_type, type._shape, values.data(), bytes.size());
  return VECTOR_KERNEL_TYPED_PAYLOAD{
      std::move(role), std::move(type), std::move(bytes), hash};
}

VECTOR_KERNEL_PLANNING_REQUEST Gemm_request(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind =
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO,
    std::vector<int64_t> input_shape = {4}, bool mask_fuse = false,
    std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> attributes = {}) {
  const VECTOR_KERNEL_RANKED_TYPE_PLAN input_type{
      PRIMITIVE_TYPE::FLOAT_32, std::move(input_shape)};
  const VECTOR_KERNEL_RANKED_TYPE_PLAN weight_type{
      PRIMITIVE_TYPE::FLOAT_32, {2, 4}};
  const VECTOR_KERNEL_RANKED_TYPE_PLAN bias_type{
      PRIMITIVE_TYPE::FLOAT_32, {2}};
  return VECTOR_KERNEL_PLANNING_REQUEST{
      VECTOR_KERNEL_OPERATION::GEMM,
      std::move(attributes),
      {input_type, weight_type, bias_type},
      VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}},
      VECTOR_KERNEL_OPTION_SNAPSHOT{false, mask_fuse, false, false, false},
      VECTOR_KERNEL_TARGET_SNAPSHOT{16, 1, 65536},
      {Float_payload("weight", {2, 4},
                     {1.0F, 2.0F, 3.0F, 4.0F,
                      5.0F, 6.0F, 7.0F, 8.0F}),
       Float_payload("bias", {2}, {0.25F, -0.5F})},
      kind};
}

VECTOR_KERNEL_SELECTION Selection(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind,
    VECTOR_KERNEL_PLAN_PROVIDER_KIND provider =
        VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
    VECTOR_KERNEL_IMPLEMENTATION implementation =
        VECTOR_KERNEL_IMPLEMENTATION::NATIVE,
    VECTOR_KERNEL_FALLBACK_POLICY fallback =
        VECTOR_KERNEL_FALLBACK_POLICY::ERROR) {
  return VECTOR_KERNEL_SELECTION{provider, implementation, kind, fallback};
}

VECTOR_KERNEL_PROVIDER_RESULT Cpp_result(
    const VECTOR_KERNEL_PLANNING_REQUEST& request) {
  CPP_VECTOR_KERNEL_PLAN_PROVIDER provider;
  VECTOR_KERNEL_PROVIDER_CALL_RESULT call = provider.Plan(request);
  if (!call._result.has_value()) {
    throw std::runtime_error("C++ provider failed in test fixture: " +
                             call._diagnostic);
  }
  return std::move(*call._result);
}

void Expect_ranked_type(const VECTOR_KERNEL_RANKED_TYPE_PLAN& actual,
                        PRIMITIVE_TYPE expected_element_type,
                        const std::vector<int64_t>& expected_shape) {
  EXPECT_EQ(actual._element_type, expected_element_type);
  EXPECT_EQ(actual._shape, expected_shape);
}

void Expect_constant_descriptor(
    const VECTOR_KERNEL_CONSTANT_PLAN& actual, const std::string& expected_role,
    PRIMITIVE_TYPE expected_element_type,
    const std::vector<int64_t>& expected_shape,
    const std::string& expected_hash) {
  EXPECT_EQ(actual._role, expected_role);
  Expect_ranked_type(actual._type, expected_element_type, expected_shape);
  EXPECT_EQ(actual._content_hash, expected_hash);
}

void Expect_payload(const VECTOR_KERNEL_TYPED_PAYLOAD& actual,
                    const std::string& expected_role,
                    PRIMITIVE_TYPE expected_element_type,
                    const std::vector<int64_t>& expected_shape,
                    const std::vector<uint8_t>& expected_bytes,
                    const std::string& expected_hash) {
  EXPECT_EQ(actual._role, expected_role);
  Expect_ranked_type(actual._type, expected_element_type, expected_shape);
  EXPECT_EQ(actual._bytes, expected_bytes);
  EXPECT_EQ(actual._content_hash, expected_hash);
}

void Expect_loop(const VECTOR_KERNEL_LOOP_PLAN& actual,
                 const std::string& expected_role, int32_t expected_lower,
                 int32_t expected_upper, int32_t expected_step,
                 uint32_t expected_nesting_depth) {
  EXPECT_EQ(actual._role, expected_role);
  EXPECT_EQ(actual._lower, expected_lower);
  EXPECT_EQ(actual._upper, expected_upper);
  EXPECT_EQ(actual._step, expected_step);
  EXPECT_EQ(actual._nesting_depth, expected_nesting_depth);
}

void Expect_slice(const VECTOR_KERNEL_SLICE_PLAN& actual,
                  const std::vector<int64_t>& expected_coefficients,
                  int64_t expected_width) {
  EXPECT_EQ(actual._role, "weight");
  EXPECT_EQ(actual._index._iv_coefficients, expected_coefficients);
  EXPECT_EQ(actual._index._constant, 0);
  EXPECT_FALSE(actual._index._uses_sharding_offset);
  EXPECT_EQ(actual._width, expected_width);
}

void Expect_rotation(const VECTOR_KERNEL_ROTATION_PLAN& actual,
                     const std::string& expected_role,
                     const std::vector<int32_t>& expected_candidates) {
  EXPECT_EQ(actual._role, expected_role);
  EXPECT_EQ(actual._candidates, expected_candidates);
}

void Expect_reduction(const VECTOR_KERNEL_REDUCTION_PLAN& actual,
                      const std::string& expected_role,
                      int64_t expected_factor, int64_t expected_block_width) {
  EXPECT_EQ(actual._role, expected_role);
  EXPECT_EQ(actual._kind, VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO);
  EXPECT_EQ(actual._factor, expected_factor);
  EXPECT_EQ(actual._block_width, expected_block_width);
  EXPECT_EQ(actual._padding, 0);
}

void Expect_runtime_preparation(
    const VECTOR_KERNEL_RUNTIME_PREPARATION& actual,
    VECTOR_KERNEL_RUNTIME_PREPARATION_KIND expected_kind,
    int64_t expected_replications, int64_t expected_blocking_width,
    const std::vector<int32_t>& expected_rotations) {
  EXPECT_EQ(actual._role, "input");
  EXPECT_EQ(actual._source_operand, 0U);
  EXPECT_EQ(actual._kind, expected_kind);
  Expect_ranked_type(actual._result_type, PRIMITIVE_TYPE::FLOAT_32, {4});
  EXPECT_EQ(actual._logical_input_size, 4);
  EXPECT_EQ(actual._replications, expected_replications);
  EXPECT_EQ(actual._blocking_width, expected_blocking_width);
  EXPECT_EQ(actual._rotation_candidates, expected_rotations);
  EXPECT_EQ(actual._outer_block_depth, 0U);
}

void Expect_baseline_gemm_golden(
    const PREPARED_VECTOR_KERNEL_PLAN& prepared) {
  constexpr char WEIGHT_HASH[] =
      "sha256:60dd6aa351452dd12455c4c9fb985dd1d1206e90433d6edd4af06dfebd045fa9";
  constexpr char BIAS_HASH[] =
      "sha256:a8212821f073ff2f119d17825f2887b52fb51f81c74afa3a5c5dd7944da29847";
  constexpr char MASK_HASH[] =
      "sha256:11153e26e7bd937cbca18baa3a639fffd7478b1dd575ab3ea0f8c5f9eecc5739";

  ASSERT_TRUE(std::holds_alternative<BASELINE_GEMM_PLAN>(prepared.Plan()));
  const BASELINE_GEMM_PLAN& plan =
      std::get<BASELINE_GEMM_PLAN>(prepared.Plan());
  EXPECT_EQ(plan._height, 2);
  EXPECT_EQ(plan._width, 4);
  EXPECT_EQ(plan._input_duplications, 2);

  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  ASSERT_EQ(common._runtime_vector_inputs.size(), 1U);
  Expect_ranked_type(common._runtime_vector_inputs.front(),
                     PRIMITIVE_TYPE::FLOAT_32, {4});
  EXPECT_TRUE(common._runtime_scalar_inputs.empty());
  ASSERT_EQ(common._constants.size(), 3U);
  Expect_constant_descriptor(common._constants[0], "weight",
                             PRIMITIVE_TYPE::FLOAT_32, {2, 4}, WEIGHT_HASH);
  Expect_constant_descriptor(common._constants[1], "bias",
                             PRIMITIVE_TYPE::FLOAT_32, {2}, BIAS_HASH);
  Expect_constant_descriptor(common._constants[2], "mask",
                             PRIMITIVE_TYPE::FLOAT_32, {2}, MASK_HASH);
  Expect_ranked_type(common._result_type, PRIMITIVE_TYPE::FLOAT_32, {8});
  ASSERT_EQ(common._loops.size(), 2U);
  Expect_loop(common._loops[0], "gemm", 0, 2, 1, 0);
  Expect_loop(common._loops[1], "block-reduction", 0, 1, 1, 0);
  ASSERT_EQ(common._slices.size(), 1U);
  Expect_slice(common._slices.front(), {1}, 4);
  ASSERT_EQ(common._rotations.size(), 3U);
  Expect_rotation(common._rotations[0], "input-duplication", {-4});
  Expect_rotation(common._rotations[1], "gemm", {0, 1});
  Expect_rotation(common._rotations[2], "block-reduction", {2});
  ASSERT_EQ(common._reductions.size(), 1U);
  Expect_reduction(common._reductions.front(), "block-reduction", 2, 2);
  EXPECT_EQ(common._mask._policy,
            VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX);
  EXPECT_EQ(common._mask._valid_length, 2);
  EXPECT_EQ(common._slot._policy,
            VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE);
  EXPECT_EQ(common._slot._value, 0U);

  ASSERT_EQ(prepared.Constants().size(), 3U);
  Expect_payload(prepared.Constants()[0], "weight",
                 PRIMITIVE_TYPE::FLOAT_32, {2, 4},
                 Float_bytes({1.0F, 6.0F, 3.0F, 8.0F,
                              2.0F, 7.0F, 4.0F, 5.0F}),
                 WEIGHT_HASH);
  Expect_payload(prepared.Constants()[1], "bias",
                 PRIMITIVE_TYPE::FLOAT_32, {2},
                 Float_bytes({0.25F, -0.5F}), BIAS_HASH);
  Expect_payload(prepared.Constants()[2], "mask",
                 PRIMITIVE_TYPE::FLOAT_32, {2},
                 Float_bytes({1.0F, 1.0F}), MASK_HASH);
  ASSERT_EQ(prepared.Runtime_preparations().size(), 1U);
  Expect_runtime_preparation(
      prepared.Runtime_preparations().front(),
      VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::PACKED_VECTOR, 2, 0, {-4});
  EXPECT_TRUE(prepared.Scalar_preparations().empty());
  EXPECT_EQ(prepared.Provenance(), "cpp");

  EXPECT_EQ(
      prepared.Specialization_key(),
      "vector-kernel-plan:v1|kind=baseline-gemm"
      "|vector-inputs=1{f32[4]}|scalar-inputs=0{}"
      "|constants=3{6:weight=f32[2,4]#71:"
      "sha256:60dd6aa351452dd12455c4c9fb985dd1d1206e90433d6edd4af06dfebd045fa9"
      ";4:bias=f32[2]#71:"
      "sha256:a8212821f073ff2f119d17825f2887b52fb51f81c74afa3a5c5dd7944da29847"
      ";4:mask=f32[2]#71:"
      "sha256:11153e26e7bd937cbca18baa3a639fffd7478b1dd575ab3ea0f8c5f9eecc5739}"
      "|result=f32[8]"
      "|loops=2{4:gemm(0,2,1,0);15:block-reduction(0,1,1,0)}"
      "|slices=1{6:weight([1],0,0,4)}"
      "|rotations=3{17:input-duplication[-4];4:gemm[0,1];"
      "15:block-reduction[2]}"
      "|reductions=1{15:block-reduction(power-of-two,2,2,0)}"
      "|mask=clear-valid-prefix:2|slot=absent-native-baseline:0"
      "|height=2|width=4|input-duplications=2");
  EXPECT_EQ(
      prepared.Helper_name(),
      "__ace_vkernel_baseline_gemm_"
      "ae1e9fd62380ffbbc155c58b42ea58f1fb5941b137c64c90083172706f40b93b");
}

void Expect_fast_gemm_golden(const PREPARED_VECTOR_KERNEL_PLAN& prepared) {
  constexpr char WEIGHT_HASH[] =
      "sha256:3b4cb4b3dcd36db407db3350348dec1fc195d92f3ea1953df420eafd7d67bd42";
  constexpr char BIAS_HASH[] =
      "sha256:a8212821f073ff2f119d17825f2887b52fb51f81c74afa3a5c5dd7944da29847";
  constexpr char ROTATION_HASH[] =
      "sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf";
  constexpr char MASK_HASH[] =
      "sha256:11153e26e7bd937cbca18baa3a639fffd7478b1dd575ab3ea0f8c5f9eecc5739";

  ASSERT_TRUE(std::holds_alternative<FAST_GEMM_PLAN>(prepared.Plan()));
  const FAST_GEMM_PLAN& plan = std::get<FAST_GEMM_PLAN>(prepared.Plan());
  EXPECT_EQ(plan._n, 2);
  EXPECT_EQ(plan._k, 4);
  EXPECT_EQ(plan._np, 2);
  EXPECT_EQ(plan._kp, 4);
  EXPECT_EQ(plan._nd, 2);
  EXPECT_EQ(plan._kd, 4);
  EXPECT_EQ(plan._block_size, 1);
  EXPECT_EQ(plan._blocks_per_partition, 2);
  EXPECT_EQ(plan._packed_partitions, 1);
  EXPECT_EQ(plan._shift, 1);
  EXPECT_EQ(plan._shift_buffer, 2);
  EXPECT_EQ(plan._grid_size, 2);
  EXPECT_EQ(plan._input_replications, 2);

  const VECTOR_KERNEL_COMMON_PLAN& common = plan._common;
  ASSERT_EQ(common._runtime_vector_inputs.size(), 1U);
  Expect_ranked_type(common._runtime_vector_inputs.front(),
                     PRIMITIVE_TYPE::FLOAT_32, {4});
  EXPECT_TRUE(common._runtime_scalar_inputs.empty());
  ASSERT_EQ(common._constants.size(), 4U);
  Expect_constant_descriptor(common._constants[0], "weight",
                             PRIMITIVE_TYPE::FLOAT_32, {2, 6}, WEIGHT_HASH);
  Expect_constant_descriptor(common._constants[1], "bias",
                             PRIMITIVE_TYPE::FLOAT_32, {2}, BIAS_HASH);
  Expect_constant_descriptor(common._constants[2], "rotation-table",
                             PRIMITIVE_TYPE::INT_S32, {1}, ROTATION_HASH);
  Expect_constant_descriptor(common._constants[3], "mask",
                             PRIMITIVE_TYPE::FLOAT_32, {2}, MASK_HASH);
  Expect_ranked_type(common._result_type, PRIMITIVE_TYPE::FLOAT_32, {6});
  ASSERT_EQ(common._loops.size(), 2U);
  Expect_loop(common._loops[0], "grid", 0, 2, 1, 0);
  Expect_loop(common._loops[1], "block", 0, 1, 1, 1);
  ASSERT_EQ(common._slices.size(), 1U);
  Expect_slice(common._slices.front(), {1, 1}, 6);
  ASSERT_EQ(common._rotations.size(), 4U);
  Expect_rotation(common._rotations[0], "input-duplication", {-4});
  Expect_rotation(common._rotations[1], "blocking-alignment", {0});
  Expect_rotation(common._rotations[2], "grid", {0, 1});
  Expect_rotation(common._rotations[3], "kp-over-np", {2});
  ASSERT_EQ(common._reductions.size(), 1U);
  Expect_reduction(common._reductions.front(), "kp-over-np", 2, 2);
  EXPECT_EQ(common._mask._policy,
            VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX);
  EXPECT_EQ(common._mask._valid_length, 2);
  EXPECT_EQ(common._slot._policy,
            VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS);
  EXPECT_EQ(common._slot._value, 2U);

  ASSERT_EQ(prepared.Constants().size(), 4U);
  Expect_payload(prepared.Constants()[0], "weight",
                 PRIMITIVE_TYPE::FLOAT_32, {2, 6},
                 Float_bytes({1.0F, 6.0F, 3.0F, 8.0F, 1.0F, 6.0F,
                              5.0F, 2.0F, 7.0F, 4.0F, 5.0F, 2.0F}),
                 WEIGHT_HASH);
  Expect_payload(prepared.Constants()[1], "bias",
                 PRIMITIVE_TYPE::FLOAT_32, {2},
                 Float_bytes({0.25F, -0.5F}), BIAS_HASH);
  Expect_payload(prepared.Constants()[2], "rotation-table",
                 PRIMITIVE_TYPE::INT_S32, {1}, {0, 0, 0, 0},
                 ROTATION_HASH);
  Expect_payload(prepared.Constants()[3], "mask",
                 PRIMITIVE_TYPE::FLOAT_32, {2},
                 Float_bytes({1.0F, 1.0F}), MASK_HASH);
  ASSERT_EQ(prepared.Runtime_preparations().size(), 1U);
  Expect_runtime_preparation(
      prepared.Runtime_preparations().front(),
      VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS, 2, 1, {0});
  EXPECT_TRUE(prepared.Scalar_preparations().empty());
  EXPECT_EQ(prepared.Provenance(), "cpp");

  EXPECT_EQ(
      prepared.Specialization_key(),
      "vector-kernel-plan:v1|kind=fast-gemm"
      "|vector-inputs=1{f32[4]}|scalar-inputs=0{}"
      "|constants=4{6:weight=f32[2,6]#71:"
      "sha256:3b4cb4b3dcd36db407db3350348dec1fc195d92f3ea1953df420eafd7d67bd42"
      ";4:bias=f32[2]#71:"
      "sha256:a8212821f073ff2f119d17825f2887b52fb51f81c74afa3a5c5dd7944da29847"
      ";14:rotation-table=s32[1]#71:"
      "sha256:c32620e947bec80857725342c2431271fe0ff6a51f914d936a860d4dd935d2cf"
      ";4:mask=f32[2]#71:"
      "sha256:11153e26e7bd937cbca18baa3a639fffd7478b1dd575ab3ea0f8c5f9eecc5739}"
      "|result=f32[6]"
      "|loops=2{4:grid(0,2,1,0);5:block(0,1,1,1)}"
      "|slices=1{6:weight([1,1],0,0,6)}"
      "|rotations=4{17:input-duplication[-4];"
      "18:blocking-alignment[0];4:grid[0,1];10:kp-over-np[2]}"
      "|reductions=1{10:kp-over-np(power-of-two,2,2,0)}"
      "|mask=clear-valid-prefix:2|slot=logical-output-elements:2"
      "|n=2|k=4|np=2|kp=4|nd=2|kd=4|block-size=1"
      "|blocks-per-partition=2|packed-partitions=1|shift=1"
      "|shift-buffer=2|grid-size=2|input-replications=2");
  EXPECT_EQ(
      prepared.Helper_name(),
      "__ace_vkernel_fast_gemm_"
      "df23015fef25b397cb7fb8737804b0cef112d8882b616c1fff07009d436a3d4c");
}

std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> Replace_first_hash(
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& constants,
    std::string hash) {
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> changed;
  changed.reserve(constants.size());
  changed.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
      constants.front()._role, constants.front()._type,
      constants.front()._bytes, std::move(hash)});
  for (auto iter = constants.begin() + 1; iter != constants.end(); ++iter) {
    changed.push_back(*iter);
  }
  return changed;
}

std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> Replace_first_type(
    const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& constants,
    VECTOR_KERNEL_RANKED_TYPE_PLAN type) {
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> changed;
  changed.reserve(constants.size());
  changed.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
      constants.front()._role, std::move(type), constants.front()._bytes,
      constants.front()._content_hash});
  for (auto iter = constants.begin() + 1; iter != constants.end(); ++iter) {
    changed.push_back(*iter);
  }
  return changed;
}

VECTOR_KERNEL_COMMON_PLAN With_loops(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    std::vector<VECTOR_KERNEL_LOOP_PLAN> loops) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, std::move(loops),
      common._slices, common._rotations, common._reductions, common._mask,
      common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_runtime_vector_inputs(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    std::vector<VECTOR_KERNEL_RANKED_TYPE_PLAN> runtime_vector_inputs) {
  return VECTOR_KERNEL_COMMON_PLAN{
      std::move(runtime_vector_inputs), common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops, common._slices,
      common._rotations, common._reductions, common._mask, common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_slices(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    std::vector<VECTOR_KERNEL_SLICE_PLAN> slices) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops,
      std::move(slices), common._rotations, common._reductions, common._mask,
      common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_rotations(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops, common._slices,
      std::move(rotations), common._reductions, common._mask, common._slot};
}

VECTOR_KERNEL_COMMON_PLAN With_reductions(
    const VECTOR_KERNEL_COMMON_PLAN& common,
    std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions) {
  return VECTOR_KERNEL_COMMON_PLAN{
      common._runtime_vector_inputs, common._runtime_scalar_inputs,
      common._constants, common._result_type, common._loops, common._slices,
      common._rotations, std::move(reductions), common._mask, common._slot};
}

VECTOR_KERNEL_PROVIDER_RESULT Baseline_gemm_result_with_common(
    const VECTOR_KERNEL_PROVIDER_RESULT& result,
    VECTOR_KERNEL_COMMON_PLAN common) {
  const BASELINE_GEMM_PLAN& plan =
      std::get<BASELINE_GEMM_PLAN>(result._plan);
  return VECTOR_KERNEL_PROVIDER_RESULT{
      VECTOR_KERNEL_PLAN{BASELINE_GEMM_PLAN{
          std::move(common), plan._height, plan._width,
          plan._input_duplications}},
      result._constants, result._runtime_preparations,
      result._scalar_preparations, result._provenance};
}

void Expect_invalid_provider_result(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const VECTOR_KERNEL_PROVIDER_RESULT& provider_result,
    const std::string& diagnostic_fragment) {
  const VECTOR_KERNEL_PREPARE_RESULT result =
      Validate_and_prepare_vector_kernel_plan(request, provider_result);
  EXPECT_FALSE(result.Ok());
  EXPECT_EQ(result._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_PROVIDER_RESULT);
  EXPECT_NE(result._diagnostic.find(diagnostic_fragment), std::string::npos)
      << result._diagnostic;
}

std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> Change_replications(
    const std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION>& preparations) {
  const VECTOR_KERNEL_RUNTIME_PREPARATION& first = preparations.front();
  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> changed;
  changed.push_back(VECTOR_KERNEL_RUNTIME_PREPARATION{
      first._role, first._source_operand, first._kind, first._result_type,
      first._logical_input_size, first._replications + 1,
      first._blocking_width, first._rotation_candidates,
      first._outer_block_depth});
  for (auto iter = preparations.begin() + 1; iter != preparations.end();
       ++iter) {
    changed.push_back(*iter);
  }
  return changed;
}

class THROWING_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "throwing-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST&) const override {
    throw std::runtime_error("deliberate provider exception");
  }
};

class DECLARED_FAILURE_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "declared-failure-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST&) const override {
    ++_calls;
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(
        "deliberate declared provider failure");
  }

  uint32_t Calls() const { return _calls; }

private:
  mutable uint32_t _calls = 0;
};

class INVALID_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "invalid-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const override {
    VECTOR_KERNEL_PROVIDER_RESULT result = Cpp_result(request);
    result._constants.pop_back();
    result._provenance = "invalid-test";
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(std::move(result));
  }
};

class CPP_SEMANTICS_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "alternate-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const override {
    VECTOR_KERNEL_PROVIDER_RESULT result = Cpp_result(request);
    result._provenance = "alternate-provider";
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(std::move(result));
  }
};

class COUNTING_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "counting-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST&) const override {
    ++_calls;
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(
        "counting provider should not be called");
  }

  uint32_t Calls() const { return _calls; }

private:
  mutable uint32_t _calls = 0;
};

}  // namespace

TEST(Tensor2VectorPlanning, DefaultSelectorsParseAndResolveAsCppNativeAuto) {
  const VECTOR_KERNEL_SELECTION_RESULT parsed =
      Parse_vector_kernel_selection("cpp", "native", "auto", "error");
  ASSERT_TRUE(parsed._selection.has_value()) << parsed._diagnostic;
  EXPECT_EQ(parsed._selection->_plan_provider,
            VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP);
  EXPECT_EQ(parsed._selection->_kernel_implementation,
            VECTOR_KERNEL_IMPLEMENTATION::NATIVE);
  EXPECT_EQ(parsed._selection->_plan_kind,
            VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO);
  EXPECT_EQ(parsed._selection->_fallback,
            VECTOR_KERNEL_FALLBACK_POLICY::ERROR);

  const VECTOR_KERNEL_PLANNING_REQUEST request = Gemm_request();
  const VECTOR_KERNEL_RESOLUTION_RESULT parsed_resolution =
      Resolve_vector_kernel_plan(request, *parsed._selection, nullptr);
  const VECTOR_KERNEL_RESOLUTION_RESULT explicit_resolution =
      Resolve_vector_kernel_plan(
          request, Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO),
          nullptr);
  ASSERT_TRUE(parsed_resolution.Ok()) << parsed_resolution._diagnostic;
  ASSERT_TRUE(explicit_resolution.Ok()) << explicit_resolution._diagnostic;
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
      *parsed_resolution._prepared, *explicit_resolution._prepared));
  EXPECT_FALSE(parsed_resolution._used_fallback);
}

TEST(Tensor2VectorPlanning, RejectsUnknownSelectorsAndInvalidForcedPair) {
  EXPECT_FALSE(
      Parse_vector_kernel_selection("native-cpp", "native", "auto", "error")
          ._selection.has_value());
  EXPECT_FALSE(
      Parse_vector_kernel_selection("cpp", "air", "auto", "error")
          ._selection.has_value());
  EXPECT_FALSE(
      Parse_vector_kernel_selection("cpp", "native", "quick", "error")
          ._selection.has_value());
  EXPECT_FALSE(
      Parse_vector_kernel_selection("cpp", "native", "auto", "silent")
          ._selection.has_value());

  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV);
  const VECTOR_KERNEL_RESOLUTION_RESULT resolution =
      Resolve_vector_kernel_plan(
          request, Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV),
          nullptr);
  EXPECT_FALSE(resolution.Ok());
  EXPECT_EQ(resolution._error, VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_FALSE(resolution._used_fallback);
  EXPECT_NE(resolution._diagnostic.find("invalid for the operation"),
            std::string::npos);
}

TEST(Tensor2VectorPlanning, RejectsInvalidEnumValuesBeforeProviderLookup) {
  const auto invalid_provider =
      static_cast<VECTOR_KERNEL_PLAN_PROVIDER_KIND>(0xffU);
  CPP_VECTOR_KERNEL_PLAN_PROVIDER cpp_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY invalid_registry;
  EXPECT_FALSE(invalid_registry.Register(invalid_provider, &cpp_provider));
  EXPECT_EQ(invalid_registry.Lookup(invalid_provider), nullptr);
  EXPECT_FALSE(invalid_registry.Unregister(invalid_provider));

  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  VECTOR_KERNEL_SELECTION invalid_selection =
      Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  invalid_selection._plan_provider = invalid_provider;
  const VECTOR_KERNEL_RESOLUTION_RESULT selection_result =
      Resolve_vector_kernel_plan(request, invalid_selection, nullptr);
  EXPECT_FALSE(selection_result.Ok());
  EXPECT_EQ(selection_result._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_SELECTION);

  const VECTOR_KERNEL_PLANNING_REQUEST invalid_operation{
      static_cast<VECTOR_KERNEL_OPERATION>(0xffU),
      request._attributes,
      request._operand_types,
      request._declared_result_type,
      request._options,
      request._target,
      request._source_constants,
      request._requested_plan_kind};
  COUNTING_PROVIDER counting_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY counting_registry;
  ASSERT_TRUE(counting_registry.Register(
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON, &counting_provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT request_result =
      Resolve_vector_kernel_plan(
          invalid_operation,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON),
          &counting_registry);
  EXPECT_FALSE(request_result.Ok());
  EXPECT_EQ(request_result._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_EQ(counting_provider.Calls(), 0U);
}

TEST(Tensor2VectorPlanning, CppGemmAutoAndForcedPlansAreDeterministic) {
  struct CASE {
    VECTOR_KERNEL_REQUESTED_PLAN_KIND _requested;
    VECTOR_KERNEL_PLAN_KIND _expected;
  };
  const CASE cases[] = {
      {VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO,
       VECTOR_KERNEL_PLAN_KIND::FAST_GEMM},
      {VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
       VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM},
      {VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM,
       VECTOR_KERNEL_PLAN_KIND::FAST_GEMM},
  };

  std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> automatic;
  std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> forced_fast;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(Vector_kernel_requested_plan_kind_name(test_case._requested));
    const VECTOR_KERNEL_PLANNING_REQUEST request =
        Gemm_request(test_case._requested);
    const VECTOR_KERNEL_SELECTION selection = Selection(test_case._requested);
    const VECTOR_KERNEL_RESOLUTION_RESULT first =
        Resolve_vector_kernel_plan(request, selection, nullptr);
    const VECTOR_KERNEL_RESOLUTION_RESULT second =
        Resolve_vector_kernel_plan(request, selection, nullptr);
    ASSERT_TRUE(first.Ok()) << first._diagnostic;
    ASSERT_TRUE(second.Ok()) << second._diagnostic;
    EXPECT_EQ(Get_vector_kernel_plan_kind(first._prepared->Plan()),
              test_case._expected);
    EXPECT_EQ(first._prepared->Specialization_key(),
              second._prepared->Specialization_key());
    EXPECT_EQ(first._prepared->Helper_name(),
              second._prepared->Helper_name());
    EXPECT_EQ(first._prepared->Constants().front()._bytes,
              second._prepared->Constants().front()._bytes);
    EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
        *first._prepared, *second._prepared));
    if (test_case._expected == VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM) {
      Expect_baseline_gemm_golden(*first._prepared);
    } else {
      Expect_fast_gemm_golden(*first._prepared);
    }
    if (test_case._requested == VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO) {
      automatic = first._prepared;
    } else if (test_case._requested ==
               VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM) {
      forced_fast = first._prepared;
    }
  }
  ASSERT_NE(automatic, nullptr);
  ASSERT_NE(forced_fast, nullptr);
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(*automatic,
                                                     *forced_fast));
  EXPECT_EQ(automatic->Specialization_key(),
            forced_fast->Specialization_key());
  EXPECT_EQ(automatic->Helper_name(), forced_fast->Helper_name());
}

TEST(Tensor2VectorPlanning,
     RankedGemmInputIsPreservedByBothReferencePlans) {
  for (VECTOR_KERNEL_REQUESTED_PLAN_KIND kind :
       {VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM}) {
    SCOPED_TRACE(Vector_kernel_requested_plan_kind_name(kind));
    const VECTOR_KERNEL_PLANNING_REQUEST request =
        Gemm_request(kind, {2, 2});
    const VECTOR_KERNEL_RESOLUTION_RESULT resolved =
        Resolve_vector_kernel_plan(request, Selection(kind), nullptr);
    ASSERT_TRUE(resolved.Ok()) << resolved._diagnostic;

    const VECTOR_KERNEL_COMMON_PLAN& common = std::visit(
        [](const auto& plan) -> const VECTOR_KERNEL_COMMON_PLAN& {
          return plan._common;
        },
        resolved._prepared->Plan());
    ASSERT_EQ(common._runtime_vector_inputs.size(), 1U);
    Expect_ranked_type(common._runtime_vector_inputs.front(),
                       PRIMITIVE_TYPE::FLOAT_32, {2, 2});
    ASSERT_EQ(resolved._prepared->Runtime_preparations().size(), 1U);
    const VECTOR_KERNEL_RUNTIME_PREPARATION& preparation =
        resolved._prepared->Runtime_preparations().front();
    Expect_ranked_type(preparation._result_type, PRIMITIVE_TYPE::FLOAT_32,
                       {2, 2});
    EXPECT_EQ(preparation._logical_input_size, 4);
  }
}

TEST(Tensor2VectorPlanning,
     ValidatorRejectsFlattenedRuntimeTypeForRankedGemmRequest) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Gemm_request(
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM, {2, 2});
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const BASELINE_GEMM_PLAN& plan =
      std::get<BASELINE_GEMM_PLAN>(valid._plan);
  VECTOR_KERNEL_PROVIDER_RESULT flattened = Baseline_gemm_result_with_common(
      valid,
      With_runtime_vector_inputs(
          plan._common,
          {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}}}));
  Expect_invalid_provider_result(request, flattened,
                                 "runtime input type does not match");
}

TEST(Tensor2VectorPlanning,
     GemmMaskRequiresExactlyOnePositiveU32BeforeProviderLookup) {
  for (VECTOR_KERNEL_REQUESTED_PLAN_KIND kind :
       {VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM}) {
    SCOPED_TRACE(Vector_kernel_requested_plan_kind_name(kind));
    const VECTOR_KERNEL_RESOLUTION_RESULT absent = Resolve_vector_kernel_plan(
        Gemm_request(kind, {4}, true), Selection(kind), nullptr);
    ASSERT_TRUE(absent.Ok()) << absent._diagnostic;
    const VECTOR_KERNEL_RESOLUTION_RESULT positive =
        Resolve_vector_kernel_plan(
            Gemm_request(
                kind, {4}, true,
                {Integer_attribute<uint32_t>(
                    "mask", PRIMITIVE_TYPE::INT_U32, {7U})}),
            Selection(kind), nullptr);
    ASSERT_TRUE(positive.Ok()) << positive._diagnostic;
  }

  COUNTING_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY registry;
  ASSERT_TRUE(registry.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                                &provider));
  const VECTOR_KERNEL_SELECTION selection = Selection(
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON);
  const std::vector<std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD>> invalid = {
      {Integer_attribute<uint32_t>("mask", PRIMITIVE_TYPE::INT_U32, {0U})},
      {Integer_attribute<uint32_t>("mask", PRIMITIVE_TYPE::INT_U32,
                                   {1U, 2U})},
      {Integer_attribute<int32_t>("mask", PRIMITIVE_TYPE::INT_S32, {1})}};
  for (const auto& attributes : invalid) {
    const VECTOR_KERNEL_PLANNING_REQUEST request = Gemm_request(
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM, {4}, true,
        attributes);
    const VECTOR_KERNEL_RESOLUTION_RESULT resolved =
        Resolve_vector_kernel_plan(request, selection, &registry);
    EXPECT_FALSE(resolved.Ok());
    EXPECT_EQ(resolved._error, VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
    EXPECT_NE(resolved._diagnostic.find("exactly one positive u32"),
              std::string::npos)
        << resolved._diagnostic;
  }
  EXPECT_EQ(provider.Calls(), 0U);
}

TEST(Tensor2VectorPlanning, MissingPythonProviderHasExplicitErrorOrFallback) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);

  const VECTOR_KERNEL_RESOLUTION_RESULT error =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON),
          nullptr);
  EXPECT_FALSE(error.Ok());
  EXPECT_EQ(error._error,
            VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_UNAVAILABLE);
  EXPECT_FALSE(error._used_fallback);

  const VECTOR_KERNEL_RESOLUTION_RESULT fallback =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                    VECTOR_KERNEL_IMPLEMENTATION::DSL,
                    VECTOR_KERNEL_FALLBACK_POLICY::CPP_NATIVE),
          nullptr);
  ASSERT_TRUE(fallback.Ok()) << fallback._diagnostic;
  EXPECT_TRUE(fallback._used_fallback);
  EXPECT_EQ(fallback._resolved_provider,
            VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP);
  EXPECT_EQ(fallback._resolved_implementation,
            VECTOR_KERNEL_IMPLEMENTATION::NATIVE);
  EXPECT_NE(fallback._diagnostic.find("provider unavailable"),
            std::string::npos);
  EXPECT_EQ(Get_vector_kernel_plan_kind(fallback._prepared->Plan()),
            VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM);
}

TEST(Tensor2VectorPlanning, MissingDslRecipeHasExplicitErrorOrFallback) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_DSL_RECIPE_QUERY no_recipe =
      [](VECTOR_KERNEL_PLAN_KIND) { return false; };

  const VECTOR_KERNEL_RESOLUTION_RESULT error =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                    VECTOR_KERNEL_IMPLEMENTATION::DSL),
          nullptr, no_recipe);
  EXPECT_FALSE(error.Ok());
  EXPECT_EQ(error._error,
            VECTOR_KERNEL_PLANNING_ERROR::UNSUPPORTED_DSL_RECIPE);
  EXPECT_FALSE(error._used_fallback);

  const VECTOR_KERNEL_RESOLUTION_RESULT fallback =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                    VECTOR_KERNEL_IMPLEMENTATION::DSL,
                    VECTOR_KERNEL_FALLBACK_POLICY::CPP_NATIVE),
          nullptr, no_recipe);
  ASSERT_TRUE(fallback.Ok()) << fallback._diagnostic;
  EXPECT_TRUE(fallback._used_fallback);
  EXPECT_EQ(fallback._resolved_provider,
            VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP);
  EXPECT_EQ(fallback._resolved_implementation,
            VECTOR_KERNEL_IMPLEMENTATION::NATIVE);
  EXPECT_NE(fallback._diagnostic.find("DSL recipe unavailable"),
            std::string::npos);
}

TEST(Tensor2VectorPlanning, ProviderExceptionsAndInvalidResultsNeverFallback) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_SELECTION selection =
      Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                VECTOR_KERNEL_IMPLEMENTATION::NATIVE,
                VECTOR_KERNEL_FALLBACK_POLICY::CPP_NATIVE);

  THROWING_PROVIDER throwing_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY throwing_registry;
  ASSERT_TRUE(throwing_registry.Register(
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON, &throwing_provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT thrown =
      Resolve_vector_kernel_plan(request, selection, &throwing_registry);
  EXPECT_FALSE(thrown.Ok());
  EXPECT_EQ(thrown._error, VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE);
  EXPECT_FALSE(thrown._used_fallback);
  EXPECT_NE(thrown._diagnostic.find("deliberate provider exception"),
            std::string::npos);

  DECLARED_FAILURE_PROVIDER declared_failure_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY declared_failure_registry;
  ASSERT_TRUE(declared_failure_registry.Register(
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
      &declared_failure_provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT declared_failure =
      Resolve_vector_kernel_plan(request, selection,
                                 &declared_failure_registry);
  EXPECT_EQ(declared_failure_provider.Calls(), 1U);
  EXPECT_FALSE(declared_failure.Ok());
  EXPECT_EQ(declared_failure._prepared, nullptr);
  EXPECT_EQ(declared_failure._error,
            VECTOR_KERNEL_PLANNING_ERROR::PROVIDER_FAILURE);
  EXPECT_FALSE(declared_failure._used_fallback);
  EXPECT_EQ(declared_failure._resolved_provider,
            VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON);
  EXPECT_EQ(declared_failure._resolved_implementation,
            VECTOR_KERNEL_IMPLEMENTATION::NATIVE);
  EXPECT_EQ(
      declared_failure._diagnostic,
      "vector-kernel provider 'declared-failure-test' failed: "
      "deliberate declared provider failure");

  INVALID_PROVIDER invalid_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY invalid_registry;
  ASSERT_TRUE(invalid_registry.Register(
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON, &invalid_provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT invalid =
      Resolve_vector_kernel_plan(request, selection, &invalid_registry);
  EXPECT_FALSE(invalid.Ok());
  EXPECT_EQ(invalid._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_PROVIDER_RESULT);
  EXPECT_FALSE(invalid._used_fallback);
  EXPECT_NE(invalid._diagnostic.find("invalid result"), std::string::npos);
}

TEST(Tensor2VectorPlanning, RequestAndPreparedPackageOwnTheirPayloadBytes) {
  std::vector<float> source_values{1.0F, 2.0F, 3.0F, 4.0F,
                                   5.0F, 6.0F, 7.0F, 8.0F};
  std::vector<uint8_t> source_bytes = Float_bytes(source_values);
  const std::vector<uint8_t> expected_source_bytes = source_bytes;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN source_type{
      PRIMITIVE_TYPE::FLOAT_32, {2, 4}};
  const std::string source_hash = Build_vector_kernel_constant_hash(
      source_type._element_type, source_type._shape, source_values.data(),
      source_bytes.size());
  const VECTOR_KERNEL_TYPED_PAYLOAD source_payload{
      "weight", source_type, source_bytes, source_hash};
  source_bytes.assign(source_bytes.size(), 0xffU);
  EXPECT_EQ(source_payload._bytes, expected_source_bytes);

  const VECTOR_KERNEL_PLANNING_REQUEST request{
      VECTOR_KERNEL_OPERATION::GEMM,
      {},
      {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
       source_type,
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}}},
      VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}},
      VECTOR_KERNEL_OPTION_SNAPSHOT{false, false, false, false, false},
      VECTOR_KERNEL_TARGET_SNAPSHOT{16, 1, 65536},
      {source_payload,
       Float_payload("bias", {2}, {0.25F, -0.5F})},
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM};
  ASSERT_EQ(request._source_constants.size(), 2U);
  EXPECT_EQ(request._source_constants[0]._bytes, expected_source_bytes);

  std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> prepared;
  std::vector<uint8_t> expected_prepared_bytes;
  {
    VECTOR_KERNEL_PROVIDER_RESULT provider_result = Cpp_result(request);
    expected_prepared_bytes = provider_result._constants.front()._bytes;
    const VECTOR_KERNEL_PREPARE_RESULT validation =
        Validate_and_prepare_vector_kernel_plan(request, provider_result);
    ASSERT_TRUE(validation.Ok()) << validation._diagnostic;
    prepared = validation._prepared;

    provider_result._constants.clear();
    provider_result._runtime_preparations.clear();
    provider_result._provenance = "mutated-after-validation";
  }
  ASSERT_NE(prepared, nullptr);
  ASSERT_FALSE(prepared->Constants().empty());
  ASSERT_FALSE(prepared->Runtime_preparations().empty());
  EXPECT_EQ(prepared->Constants().front()._bytes, expected_prepared_bytes);
  EXPECT_EQ(prepared->Provenance(), "cpp");
}

TEST(Tensor2VectorPlanning, ValidatorRejectsHashRoleAndPreparationChanges) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);

  {
    const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
    VECTOR_KERNEL_PROVIDER_RESULT invalid{
        valid._plan,
        Replace_first_hash(valid._constants,
                           "sha256:" + std::string(64, 'f')),
        valid._runtime_preparations,
        valid._scalar_preparations,
        valid._provenance};
    const VECTOR_KERNEL_PREPARE_RESULT result =
        Validate_and_prepare_vector_kernel_plan(request, invalid);
    EXPECT_FALSE(result.Ok());
    EXPECT_EQ(result._error,
              VECTOR_KERNEL_PLANNING_ERROR::INVALID_PROVIDER_RESULT);
    EXPECT_NE(result._diagnostic.find("hash mismatch"), std::string::npos);
  }

  {
    const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
    std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> duplicate_roles;
    duplicate_roles.push_back(valid._constants[0]);
    duplicate_roles.push_back(valid._constants[0]);
    for (auto iter = valid._constants.begin() + 2;
         iter != valid._constants.end(); ++iter) {
      duplicate_roles.push_back(*iter);
    }
    VECTOR_KERNEL_PROVIDER_RESULT invalid{
        valid._plan,
        std::move(duplicate_roles),
        valid._runtime_preparations,
        valid._scalar_preparations,
        valid._provenance};
    const VECTOR_KERNEL_PREPARE_RESULT result =
        Validate_and_prepare_vector_kernel_plan(request, invalid);
    EXPECT_FALSE(result.Ok());
    EXPECT_NE(result._diagnostic.find("duplicate constant role"),
              std::string::npos);
  }

  {
    const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
    std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> missing_role =
        valid._constants;
    missing_role.pop_back();
    VECTOR_KERNEL_PROVIDER_RESULT invalid{
        valid._plan,
        std::move(missing_role),
        valid._runtime_preparations,
        valid._scalar_preparations,
        valid._provenance};
    const VECTOR_KERNEL_PREPARE_RESULT result =
        Validate_and_prepare_vector_kernel_plan(request, invalid);
    EXPECT_FALSE(result.Ok());
    EXPECT_NE(result._diagnostic.find("role count"), std::string::npos);
  }

  {
    const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
    VECTOR_KERNEL_PROVIDER_RESULT invalid{
        valid._plan,
        valid._constants,
        Change_replications(valid._runtime_preparations),
        valid._scalar_preparations,
        valid._provenance};
    const VECTOR_KERNEL_PREPARE_RESULT result =
        Validate_and_prepare_vector_kernel_plan(request, invalid);
    EXPECT_FALSE(result.Ok());
    EXPECT_NE(result._diagnostic.find("runtime preparation mismatch"),
              std::string::npos);
  }
}

TEST(Tensor2VectorPlanning, InvalidOwnedRequestsNeverReachProvider) {
  const VECTOR_KERNEL_PLANNING_REQUEST valid =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  COUNTING_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY registry;
  ASSERT_TRUE(registry.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                                &provider));
  const VECTOR_KERNEL_SELECTION selection =
      Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON);

  const VECTOR_KERNEL_PLANNING_REQUEST below_minimum{
      valid._operation,
      valid._attributes,
      valid._operand_types,
      valid._declared_result_type,
      valid._options,
      VECTOR_KERNEL_TARGET_SNAPSHOT{8, 16, valid._target._max_slots},
      valid._source_constants,
      valid._requested_plan_kind};
  const VECTOR_KERNEL_RESOLUTION_RESULT bad_target =
      Resolve_vector_kernel_plan(below_minimum, selection, &registry);
  EXPECT_FALSE(bad_target.Ok());
  EXPECT_EQ(bad_target._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_NE(bad_target._diagnostic.find("target snapshot"),
            std::string::npos)
      << bad_target._diagnostic;
  EXPECT_EQ(provider.Calls(), 0U);

  const VECTOR_KERNEL_PLANNING_REQUEST bad_hash{
      valid._operation,
      valid._attributes,
      valid._operand_types,
      valid._declared_result_type,
      valid._options,
      valid._target,
      Replace_first_hash(valid._source_constants,
                         "sha256:" + std::string(64, 'f')),
      valid._requested_plan_kind};
  const VECTOR_KERNEL_RESOLUTION_RESULT invalid_payload =
      Resolve_vector_kernel_plan(bad_hash, selection, &registry);
  EXPECT_FALSE(invalid_payload.Ok());
  EXPECT_EQ(invalid_payload._error,
            VECTOR_KERNEL_PLANNING_ERROR::INVALID_REQUEST);
  EXPECT_NE(invalid_payload._diagnostic.find("request constant hash mismatch"),
            std::string::npos)
      << invalid_payload._diagnostic;
  EXPECT_EQ(provider.Calls(), 0U);
}

TEST(Tensor2VectorPlanning, ValidatorRejectsWrongOperationAndForcedKind) {
  const VECTOR_KERNEL_PLANNING_REQUEST baseline_request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_PROVIDER_RESULT baseline = Cpp_result(baseline_request);
  const BASELINE_GEMM_PLAN& baseline_plan =
      std::get<BASELINE_GEMM_PLAN>(baseline._plan);
  const VECTOR_KERNEL_PROVIDER_RESULT wrong_operation{
      VECTOR_KERNEL_PLAN{BASELINE_CONV_PLAN{
          baseline_plan._common, 1, 1, 1, 1, 1, 1, 1}},
      baseline._constants,
      baseline._runtime_preparations,
      baseline._scalar_preparations,
      baseline._provenance};
  Expect_invalid_provider_result(baseline_request, wrong_operation,
                                 "wrong operation");

  const VECTOR_KERNEL_PLANNING_REQUEST auto_request = Gemm_request();
  const VECTOR_KERNEL_PROVIDER_RESULT automatic = Cpp_result(auto_request);
  ASSERT_TRUE(std::holds_alternative<FAST_GEMM_PLAN>(automatic._plan));
  Expect_invalid_provider_result(baseline_request, automatic,
                                 "forced plan kind");
}

TEST(Tensor2VectorPlanning,
     ValidatorRejectsExtraAndDuplicatePlanRecordRoles) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  const VECTOR_KERNEL_COMMON_PLAN& common =
      std::get<BASELINE_GEMM_PLAN>(valid._plan)._common;
  ASSERT_FALSE(common._loops.empty());
  ASSERT_FALSE(common._slices.empty());
  ASSERT_FALSE(common._rotations.empty());
  ASSERT_FALSE(common._reductions.empty());

  {
    std::vector<VECTOR_KERNEL_LOOP_PLAN> records = common._loops;
    records.push_back(records.front());
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_loops(common, std::move(records))),
        "duplicate loop role");
  }
  {
    std::vector<VECTOR_KERNEL_LOOP_PLAN> records = common._loops;
    const VECTOR_KERNEL_LOOP_PLAN& first = records.front();
    records.push_back(VECTOR_KERNEL_LOOP_PLAN{
        "unexpected", first._lower, first._upper, first._step,
        first._nesting_depth});
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_loops(common, std::move(records))),
        "loops count");
  }
  {
    std::vector<VECTOR_KERNEL_SLICE_PLAN> records = common._slices;
    records.push_back(records.front());
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_slices(common, std::move(records))),
        "duplicate slice role");
  }
  {
    std::vector<VECTOR_KERNEL_SLICE_PLAN> records = common._slices;
    const VECTOR_KERNEL_SLICE_PLAN& first = records.front();
    records.push_back(VECTOR_KERNEL_SLICE_PLAN{
        "unexpected", first._index, first._width});
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_slices(common, std::move(records))),
        "slices count");
  }
  {
    std::vector<VECTOR_KERNEL_ROTATION_PLAN> records = common._rotations;
    records.push_back(records.front());
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_rotations(common, std::move(records))),
        "duplicate rotation role");
  }
  {
    std::vector<VECTOR_KERNEL_ROTATION_PLAN> records = common._rotations;
    const VECTOR_KERNEL_ROTATION_PLAN& first = records.front();
    records.push_back(VECTOR_KERNEL_ROTATION_PLAN{
        "unexpected", first._candidates});
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_rotations(common, std::move(records))),
        "rotations count");
  }
  {
    std::vector<VECTOR_KERNEL_REDUCTION_PLAN> records = common._reductions;
    records.push_back(records.front());
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_reductions(common, std::move(records))),
        "duplicate reduction role");
  }
  {
    std::vector<VECTOR_KERNEL_REDUCTION_PLAN> records = common._reductions;
    const VECTOR_KERNEL_REDUCTION_PLAN& first = records.front();
    records.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        "unexpected", first._kind, first._factor, first._block_width,
        first._padding});
    Expect_invalid_provider_result(
        request,
        Baseline_gemm_result_with_common(
            valid, With_reductions(common, std::move(records))),
        "reductions count");
  }
}

TEST(Tensor2VectorPlanning, ValidatorRejectsWrongConstantTypeAndShape) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  ASSERT_FALSE(valid._constants.empty());
  const VECTOR_KERNEL_TYPED_PAYLOAD& weight = valid._constants.front();

  const VECTOR_KERNEL_PROVIDER_RESULT wrong_type{
      valid._plan,
      Replace_first_type(
          valid._constants,
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::INT_S32,
                                         weight._type._shape}),
      valid._runtime_preparations,
      valid._scalar_preparations,
      valid._provenance};
  Expect_invalid_provider_result(request, wrong_type,
                                 "constant type mismatch");

  const int64_t element_count =
      static_cast<int64_t>(weight._bytes.size() / sizeof(float));
  const VECTOR_KERNEL_PROVIDER_RESULT wrong_shape{
      valid._plan,
      Replace_first_type(
          valid._constants,
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                         {element_count}}),
      valid._runtime_preparations,
      valid._scalar_preparations,
      valid._provenance};
  Expect_invalid_provider_result(request, wrong_shape,
                                 "constant type mismatch");
}

TEST(Tensor2VectorPlanning, ValidatorRejectsFastGemmProductOverflow) {
  const VECTOR_KERNEL_PLANNING_REQUEST request = Gemm_request();
  const VECTOR_KERNEL_PROVIDER_RESULT valid = Cpp_result(request);
  ASSERT_TRUE(std::holds_alternative<FAST_GEMM_PLAN>(valid._plan));
  const FAST_GEMM_PLAN& plan = std::get<FAST_GEMM_PLAN>(valid._plan);
  const FAST_GEMM_PLAN overflowing{
      plan._common,
      plan._n,
      plan._k,
      plan._np,
      plan._kp,
      plan._nd,
      plan._kd,
      std::numeric_limits<int64_t>::max(),
      2,
      plan._packed_partitions,
      plan._shift,
      plan._shift_buffer,
      plan._grid_size,
      plan._input_replications};
  const VECTOR_KERNEL_PROVIDER_RESULT invalid{
      VECTOR_KERNEL_PLAN{overflowing},
      valid._constants,
      valid._runtime_preparations,
      valid._scalar_preparations,
      valid._provenance};
  Expect_invalid_provider_result(request, invalid,
                                 "fast Gemm plan invariants");
}

TEST(Tensor2VectorPlanning, SemanticIdentityIgnoresProviderAndProvenance) {
  const VECTOR_KERNEL_PLANNING_REQUEST request =
      Gemm_request(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);
  const VECTOR_KERNEL_RESOLUTION_RESULT cpp =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM),
          nullptr);
  ASSERT_TRUE(cpp.Ok()) << cpp._diagnostic;

  CPP_SEMANTICS_PROVIDER alternate_provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY registry;
  ASSERT_TRUE(registry.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                                &alternate_provider));
  const VECTOR_KERNEL_RESOLUTION_RESULT alternate =
      Resolve_vector_kernel_plan(
          request,
          Selection(VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
                    VECTOR_KERNEL_PLAN_PROVIDER_KIND::PYTHON,
                    VECTOR_KERNEL_IMPLEMENTATION::DSL),
          &registry, [](VECTOR_KERNEL_PLAN_KIND) { return true; });
  ASSERT_TRUE(alternate.Ok()) << alternate._diagnostic;

  EXPECT_EQ(cpp._prepared->Provenance(), "cpp");
  EXPECT_EQ(alternate._prepared->Provenance(), "alternate-provider");
  EXPECT_EQ(cpp._prepared->Specialization_key(),
            alternate._prepared->Specialization_key());
  EXPECT_EQ(cpp._prepared->Helper_name(),
            alternate._prepared->Helper_name());
  EXPECT_TRUE(Equal_vector_kernel_prepared_semantics(
      *cpp._prepared, *alternate._prepared));
}
