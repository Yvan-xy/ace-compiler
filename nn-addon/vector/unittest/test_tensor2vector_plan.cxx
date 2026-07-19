//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nn/vector/tensor2vector_plan.h"

using air::base::PRIMITIVE_TYPE;
using namespace nn::vector;

namespace {

static_assert(!std::is_default_constructible_v<BASELINE_GEMM_PLAN>);
static_assert(!std::is_copy_assignable_v<BASELINE_GEMM_PLAN>);
static_assert(!std::is_copy_assignable_v<BASELINE_CONV_PLAN>);
static_assert(!std::is_copy_assignable_v<FAST_GEMM_PLAN>);
static_assert(!std::is_copy_assignable_v<FAST_CONV_PLAN>);

std::string Digest(char value) {
  return "sha256:" + std::string(64, value);
}

BASELINE_GEMM_PLAN Baseline_gemm_fixture(
    std::vector<int32_t> reduction_rotations = {2, 4}) {
  return BASELINE_GEMM_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}}},
          {},
          {VECTOR_KERNEL_CONSTANT_PLAN{
               "weight",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {2, 8}},
               Digest('0')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "bias",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}},
               Digest('1')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "mask",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}},
               Digest('2')}},
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {16}},
          {VECTOR_KERNEL_LOOP_PLAN{"gemm", 0, 2, 1, 0},
           VECTOR_KERNEL_LOOP_PLAN{"block-reduction", 0, 2, 1, 0}},
          {VECTOR_KERNEL_SLICE_PLAN{
              "weight", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false}, 8}},
          {VECTOR_KERNEL_ROTATION_PLAN{"input-duplication", {-8}},
           VECTOR_KERNEL_ROTATION_PLAN{"gemm", {0, 1}},
           VECTOR_KERNEL_ROTATION_PLAN{"block-reduction",
                                       std::move(reduction_rotations)}},
          {VECTOR_KERNEL_REDUCTION_PLAN{
              "block-reduction", VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO,
              4, 2, 0}},
          VECTOR_KERNEL_MASK_PLAN{
              VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX, 2},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE, 0}},
      2,
      8,
      2};
}

BASELINE_CONV_PLAN Baseline_conv_fixture() {
  return BASELINE_CONV_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}}},
          {},
          {VECTOR_KERNEL_CONSTANT_PLAN{
               "weight",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {6, 8}},
               Digest('3')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "bias",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}},
               Digest('4')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "rotation-table",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::INT_S32, {3}},
               Digest('5')}},
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                         {1, 2, 2, 2}},
          {VECTOR_KERNEL_LOOP_PLAN{"channel-in", 0, 2, 1, 0},
           VECTOR_KERNEL_LOOP_PLAN{"kernel-hw", 0, 3, 1, 1}},
          {VECTOR_KERNEL_SLICE_PLAN{
              "weight", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{3, 1}, 0, false},
              8}},
          {VECTOR_KERNEL_ROTATION_PLAN{"input-duplication", {-8}},
           VECTOR_KERNEL_ROTATION_PLAN{"kernel-alignment", {-1, 0, 1}},
           VECTOR_KERNEL_ROTATION_PLAN{"channel-step", {4}}},
          {},
          VECTOR_KERNEL_MASK_PLAN{VECTOR_KERNEL_MASK_POLICY::NONE, 0},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS, 8}},
      2,
      2,
      2,
      2,
      3,
      1,
      2};
}

FAST_GEMM_PLAN Fast_gemm_fixture() {
  return FAST_GEMM_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}}},
          {},
          {VECTOR_KERNEL_CONSTANT_PLAN{
               "weight",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {4, 12}},
               Digest('6')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "bias",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
               Digest('7')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "rotation-table",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::INT_S32, {2}},
               Digest('8')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "mask",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
               Digest('9')}},
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {12}},
          {VECTOR_KERNEL_LOOP_PLAN{"grid", 0, 2, 1, 0},
           VECTOR_KERNEL_LOOP_PLAN{"block", 0, 2, 1, 1}},
          {VECTOR_KERNEL_SLICE_PLAN{
              "weight", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{2, 1}, 0, false},
              12}},
          {VECTOR_KERNEL_ROTATION_PLAN{"input-duplication", {-8}},
           VECTOR_KERNEL_ROTATION_PLAN{"blocking-alignment", {0, 1}},
           VECTOR_KERNEL_ROTATION_PLAN{"grid", {0, 2}},
           VECTOR_KERNEL_ROTATION_PLAN{"kp-over-np", {4}}},
          {VECTOR_KERNEL_REDUCTION_PLAN{
              "kp-over-np", VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO, 2,
              4, 0}},
          VECTOR_KERNEL_MASK_PLAN{
              VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX, 4},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS, 4}},
      4,
      8,
      4,
      8,
      4,
      8,
      2,
      2,
      1,
      1,
      4,
      2,
      2};
}

FAST_CONV_PLAN Fast_conv_fixture(bool sharded = true) {
  std::vector<PRIMITIVE_TYPE> scalar_inputs;
  std::optional<VECTOR_KERNEL_SHARDING_OFFSET_PLAN> offset;
  if (sharded) {
    scalar_inputs.push_back(PRIMITIVE_TYPE::INT_S32);
    offset.emplace(VECTOR_KERNEL_SHARDING_OFFSET_PLAN{
        PRIMITIVE_TYPE::INT_S32, 18});
  }

  return FAST_CONV_PLAN{
      VECTOR_KERNEL_COMMON_PLAN{
          {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}}},
          std::move(scalar_inputs),
          {VECTOR_KERNEL_CONSTANT_PLAN{
               "weight",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {18, 16}},
               Digest('a')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "bias",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {16}},
               Digest('b')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "rotation-table",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::INT_S32, {9}},
               Digest('c')},
           VECTOR_KERNEL_CONSTANT_PLAN{
               "collective-mask",
               VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                              {16}},
               Digest('d')}},
          VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32,
                                         {1, 4, 2, 2}},
          {VECTOR_KERNEL_LOOP_PLAN{"grid", 0, 2, 1, 0},
           VECTOR_KERNEL_LOOP_PLAN{"capacity-block", 0, 9, 1, 1}},
          {VECTOR_KERNEL_SLICE_PLAN{
              "weight",
              VECTOR_KERNEL_AFFINE_INDEX_PLAN{{9, 1}, 0, sharded}, 16}},
          {VECTOR_KERNEL_ROTATION_PLAN{
               "blocking-alignment", {0, 1, 2, 3, 4, 5, 6, 7, 8}},
           VECTOR_KERNEL_ROTATION_PLAN{"grid", {0, 4}}},
          {VECTOR_KERNEL_REDUCTION_PLAN{
              "collective",
              VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK, 1, 16,
              0}},
          VECTOR_KERNEL_MASK_PLAN{
              VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION, 16},
          VECTOR_KERNEL_SLOT_PLAN{
              VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS, 16}},
      2,
      4,
      2,
      2,
      9,
      1,
      1,
      8,
      16,
      32,
      2,
      1,
      16,
      16,
      0,
      4,
      9,
      2,
      0,
      false,
      std::move(offset)};
}

std::string Expected_baseline_gemm_key() {
  return std::string("vector-kernel-plan:v1|kind=baseline-gemm") +
         "|vector-inputs=1{f32[8]}|scalar-inputs=0{}" +
         "|constants=3{6:weight=f32[2,8]#71:" + Digest('0') +
         ";4:bias=f32[2]#71:" + Digest('1') +
         ";4:mask=f32[2]#71:" + Digest('2') +
         "}|result=f32[16]" +
         "|loops=2{4:gemm(0,2,1,0);15:block-reduction(0,2,1,0)}" +
         "|slices=1{6:weight([1],0,0,8)}" +
         "|rotations=3{17:input-duplication[-8];4:gemm[0,1];"
         "15:block-reduction[2,4]}" +
         "|reductions=1{15:block-reduction(power-of-two,4,2,0)}" +
         "|mask=clear-valid-prefix:2|slot=absent-native-baseline:0" +
         "|height=2|width=8|input-duplications=2";
}

std::string Expected_baseline_conv_key() {
  return std::string("vector-kernel-plan:v1|kind=baseline-conv") +
         "|vector-inputs=1{f32[8]}|scalar-inputs=0{}" +
         "|constants=3{6:weight=f32[6,8]#71:" + Digest('3') +
         ";4:bias=f32[8]#71:" + Digest('4') +
         ";14:rotation-table=s32[3]#71:" + Digest('5') +
         "}|result=f32[1,2,2,2]" +
         "|loops=2{10:channel-in(0,2,1,0);9:kernel-hw(0,3,1,1)}" +
         "|slices=1{6:weight([3,1],0,0,8)}" +
         "|rotations=3{17:input-duplication[-8];"
         "16:kernel-alignment[-1,0,1];12:channel-step[4]}" +
         "|reductions=0{}|mask=none:0|slot=logical-output-elements:8" +
         "|channel-in=2|channel-out=2|output-height=2|output-width=2" +
         "|kernel-hw=3|stride=1|input-duplications=2";
}

std::string Expected_fast_gemm_key() {
  return std::string("vector-kernel-plan:v1|kind=fast-gemm") +
         "|vector-inputs=1{f32[8]}|scalar-inputs=0{}" +
         "|constants=4{6:weight=f32[4,12]#71:" + Digest('6') +
         ";4:bias=f32[4]#71:" + Digest('7') +
         ";14:rotation-table=s32[2]#71:" + Digest('8') +
         ";4:mask=f32[4]#71:" + Digest('9') +
         "}|result=f32[12]" +
         "|loops=2{4:grid(0,2,1,0);5:block(0,2,1,1)}" +
         "|slices=1{6:weight([2,1],0,0,12)}" +
         "|rotations=4{17:input-duplication[-8];"
         "18:blocking-alignment[0,1];4:grid[0,2];10:kp-over-np[4]}" +
         "|reductions=1{10:kp-over-np(power-of-two,2,4,0)}" +
         "|mask=clear-valid-prefix:4|slot=logical-output-elements:4" +
         "|n=4|k=8|np=4|kp=8|nd=4|kd=8|block-size=2" +
         "|blocks-per-partition=2|packed-partitions=1|shift=1" +
         "|shift-buffer=4|grid-size=2|input-replications=2";
}

std::string Expected_fast_conv_key() {
  return std::string("vector-kernel-plan:v1|kind=fast-conv") +
         "|vector-inputs=1{f32[8]}|scalar-inputs=1{s32}" +
         "|constants=4{6:weight=f32[18,16]#71:" + Digest('a') +
         ";4:bias=f32[16]#71:" + Digest('b') +
         ";14:rotation-table=s32[9]#71:" + Digest('c') +
         ";15:collective-mask=f32[16]#71:" + Digest('d') +
         "}|result=f32[1,4,2,2]" +
         "|loops=2{4:grid(0,2,1,0);14:capacity-block(0,9,1,1)}" +
         "|slices=1{6:weight([9,1],0,1,16)}" +
         "|rotations=2{18:blocking-alignment[0,1,2,3,4,5,6,7,8];"
         "4:grid[0,4]}" +
         "|reductions=1{10:collective(collective-single-block,1,16,0)}" +
         "|mask=collective-reduction:16|slot=logical-output-elements:16" +
         "|channel-in=2|channel-out=4|output-height=2|output-width=2" +
         "|kernel-hw=9|group=1|stride=1|input-size=8|output-size=16" +
         "|num-slots=32|num-grid=2|num-block=1|width-block=16" +
         "|width-block-data=16|width-block-pad=0|position-block=4" +
         "|capacity-block=9|input-duplications=2|blocking-outer-depth=0" +
         "|cyclic-roll=0|sharding-offset=s32*18";
}

}  // namespace

TEST(Tensor2VectorPlan, Sha256AndCanonicalConstantHashesAreFrozen) {
  EXPECT_EQ(Vector_kernel_sha256(""),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(Vector_kernel_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(Vector_kernel_sha256(
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a"
            "33ce45964ff2167f6ecedd419db06c1");

  // Exact f32 bit patterns for +0, -0, and a quiet NaN with a non-default
  // payload. Hashing bits rather than values preserves all three distinctions.
  const std::array<uint32_t, 3> payload = {0x00000000U, 0x80000000U,
                                            0x7fc01234U};
  const std::string hash = Build_vector_kernel_constant_hash(
      PRIMITIVE_TYPE::FLOAT_32, {3}, payload.data(), sizeof(payload));
  EXPECT_EQ(hash,
            "sha256:a00da14ae384f2fe747a307da87ec9714"
            "7fee8a129d94f45db8f4d77d6314e2d");

  EXPECT_NE(hash, Build_vector_kernel_constant_hash(
                      PRIMITIVE_TYPE::INT_U32, {3}, payload.data(),
                      sizeof(payload)));
  EXPECT_NE(hash, Build_vector_kernel_constant_hash(
                      PRIMITIVE_TYPE::FLOAT_32, {1, 3}, payload.data(),
                      sizeof(payload)));

  const std::array<uint32_t, 1> positive_zero = {0x00000000U};
  const std::array<uint32_t, 1> negative_zero = {0x80000000U};
  EXPECT_NE(Build_vector_kernel_constant_hash(
                PRIMITIVE_TYPE::FLOAT_32, {1}, positive_zero.data(),
                sizeof(positive_zero)),
            Build_vector_kernel_constant_hash(
                PRIMITIVE_TYPE::FLOAT_32, {1}, negative_zero.data(),
                sizeof(negative_zero)));

  const std::array<uint32_t, 1> nan_payload_a = {0x7fc01234U};
  const std::array<uint32_t, 1> nan_payload_b = {0x7fc01235U};
  EXPECT_NE(Build_vector_kernel_constant_hash(
                PRIMITIVE_TYPE::FLOAT_32, {1}, nan_payload_a.data(),
                sizeof(nan_payload_a)),
            Build_vector_kernel_constant_hash(
                PRIMITIVE_TYPE::FLOAT_32, {1}, nan_payload_b.data(),
                sizeof(nan_payload_b)));
}

TEST(Tensor2VectorPlan, HelperSpecializationNameIsFrozen) {
  const VECTOR_KERNEL_PLAN plan = Baseline_gemm_fixture();
  EXPECT_EQ(Build_vector_kernel_helper_name(plan),
            "__ace_vkernel_baseline_gemm_"
            "989bcc883b0d7979c61de25146568196"
            "e27506300dd0a04895e0b03d3f45b794");
}

TEST(Tensor2VectorPlan, HelperAbiIsFrozen) {
  EXPECT_TRUE(VECTOR_KERNEL_HELPER_ABI::LEAF);
  EXPECT_TRUE(VECTOR_KERNEL_HELPER_ABI::NONRECURSIVE);
  EXPECT_TRUE(VECTOR_KERNEL_HELPER_ABI::SAME_GLOB_SCOPE_CONSTANTS);
  EXPECT_TRUE(VECTOR_KERNEL_HELPER_ABI::VECTOR_FORMALS_FIRST);
  EXPECT_TRUE(VECTOR_KERNEL_HELPER_ABI::OPTIONAL_S32_OFFSET_LAST);
  EXPECT_EQ(VECTOR_KERNEL_HELPER_ABI::RESULT_COUNT, 1U);
  EXPECT_EQ(VECTOR_KERNEL_HELPER_ABI::TERMINAL_RETV_COUNT, 1U);
}

TEST(Tensor2VectorPlan, FourVariantSpecializationKeysAreFrozen) {
  const VECTOR_KERNEL_PLAN baseline_gemm = Baseline_gemm_fixture();
  const VECTOR_KERNEL_PLAN baseline_conv = Baseline_conv_fixture();
  const VECTOR_KERNEL_PLAN fast_gemm     = Fast_gemm_fixture();
  const VECTOR_KERNEL_PLAN fast_conv     = Fast_conv_fixture();

  EXPECT_EQ(Get_vector_kernel_plan_kind(baseline_gemm),
            VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM);
  EXPECT_EQ(Get_vector_kernel_plan_kind(baseline_conv),
            VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV);
  EXPECT_EQ(Get_vector_kernel_plan_kind(fast_gemm),
            VECTOR_KERNEL_PLAN_KIND::FAST_GEMM);
  EXPECT_EQ(Get_vector_kernel_plan_kind(fast_conv),
            VECTOR_KERNEL_PLAN_KIND::FAST_CONV);

  EXPECT_EQ(Build_vector_kernel_specialization_key(baseline_gemm),
            Expected_baseline_gemm_key());
  EXPECT_EQ(Build_vector_kernel_specialization_key(baseline_conv),
            Expected_baseline_conv_key());
  EXPECT_EQ(Build_vector_kernel_specialization_key(fast_gemm),
            Expected_fast_gemm_key());
  EXPECT_EQ(Build_vector_kernel_specialization_key(fast_conv),
            Expected_fast_conv_key());
}

TEST(Tensor2VectorPlan, KeyPreservesRotationOrderAndOffsetPolicy) {
  const VECTOR_KERNEL_PLAN ordered = Baseline_gemm_fixture({2, 4});
  const VECTOR_KERNEL_PLAN swapped = Baseline_gemm_fixture({4, 2});
  EXPECT_NE(Build_vector_kernel_specialization_key(ordered),
            Build_vector_kernel_specialization_key(swapped));

  const VECTOR_KERNEL_PLAN sharded   = Fast_conv_fixture(true);
  const VECTOR_KERNEL_PLAN unsharded = Fast_conv_fixture(false);
  EXPECT_NE(Build_vector_kernel_specialization_key(sharded),
            Build_vector_kernel_specialization_key(unsharded));
}

TEST(Tensor2VectorPlan, BaselineGemmPlanCoversReductionAndMaskCase) {
  const BASELINE_GEMM_PLAN fixture = Baseline_gemm_fixture();
  EXPECT_EQ(fixture._height, 2);
  EXPECT_EQ(fixture._width, 8);
  ASSERT_EQ(fixture._common._loops.size(), 2U);
  EXPECT_EQ(fixture._common._loops[0]._upper, 2);
  EXPECT_EQ(fixture._common._loops[1]._upper, 2);
  EXPECT_EQ(fixture._common._slot._policy,
            VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE);
}
