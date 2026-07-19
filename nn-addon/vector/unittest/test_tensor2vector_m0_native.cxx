//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/node.h"
#include "air/base/st.h"
#include "air/base/st_attr.h"
#include "air/base/st_iter.h"
#include "air/base/st_type.h"
#include "air/core/opcode.h"
#include "nn/core/attr.h"
#include "nn/core/opcode.h"
#include "nn/vector/config.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "nn/vector/tensor2vector_plan.h"
#include "nn/vector/tensor2vector_util.h"
#include "nn/vector/vector_ctx.h"
#include "nn/vector/vector_opcode.h"
#include "nn/vector/vector_utils.h"
#include "tensor2vector_air_normalizer.h"

using namespace air::base;
using namespace nn::vector;

namespace {

struct NATIVE_GEMM_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_SCOPE*                 _func_scope = nullptr;
  NODE_PTR                    _result;
};

NATIVE_GEMM_IR Build_native_gemm(const char* function_name, uint32_t line,
                                 int64_t height, int64_t width,
                                 bool need_mask) {
  NATIVE_GEMM_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE* glob = fixture._glob.get();
  const SPOS  spos(0, line, 1, 0);

  TYPE_PTR f32 = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  TYPE_PTR input_type =
      New_array_type(glob, std::string(function_name) + "_input", f32,
                     {width}, spos);
  TYPE_PTR result_type =
      New_array_type(glob, std::string(function_name) + "_result", f32,
                     {2 * width}, spos);

  STR_PTR  name = glob->New_str(function_name);
  FUNC_PTR func = glob->New_func(name, spos);
  func->Set_parent(glob->Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob->New_sig_type();
  glob->New_ret_param(result_type, signature);
  glob->New_param(glob->New_str("packed_input"), input_type, signature, spos);
  signature->Set_complete();
  glob->New_entry_point(signature, func, name, spos);

  fixture._func_scope = &glob->New_func_scope(func);
  CONTAINER* container = &fixture._func_scope->Container();
  STMT_PTR   entry     = container->New_func_entry(spos);
  NODE_PTR   body      = entry->Node()->Last_child();

  std::vector<float> weight(height * width);
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    weight[idx] = static_cast<float>(idx + 1) / 8.0F;
  }
  std::vector<float> bias(height);
  for (size_t idx = 0; idx < bias.size(); ++idx) {
    bias[idx] = static_cast<float>(idx + 1) / 16.0F;
  }
  CONSTANT_PTR weight_constant = New_array_const(
      glob, std::string(function_name) + "_weight", height * width, f32,
      {height, width}, weight.data(), spos);
  CONSTANT_PTR bias_constant = New_array_const(
      glob, std::string(function_name) + "_bias", height, f32, {height},
      bias.data(), spos);

  VECTOR_CTX    vector_ctx;
  VECTOR_CONFIG config;
  vector_ctx.Update_slot(128);
  TENSOR2VECTOR_CTX lowering_ctx(container, vector_ctx, nullptr, config);
  lowering_ctx.Set_cur_func_scope(fixture._func_scope);
  TENSOR2VECTOR_UTIL util(lowering_ctx);
  lowering_ctx.Push(body, body);
  fixture._result = util.New_gemm_metakernel(
      container->New_ld(fixture._func_scope->Formal(0), spos),
      container->New_ldc(weight_constant, spos),
      container->New_ldc(bias_constant, spos), need_mask, spos);
  container->Stmt_list().Append(container->New_retv(fixture._result, spos));
  lowering_ctx.Pop(body, body);
  return fixture;
}

struct NATIVE_GEMM_STATS {
  uint32_t                      _loops       = 0;
  uint32_t                      _vector_muls = 0;
  uint32_t                      _slices      = 0;
  uint32_t                      _slot_attrs  = 0;
  std::vector<std::vector<int>> _rotations;
};

void Collect_stats(NODE_PTR node, NATIVE_GEMM_STATS& stats) {
  if (node->Is_do_loop()) ++stats._loops;
  if (node->Opcode() ==
      OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::MUL)) {
    ++stats._vector_muls;
  }
  if (node->Opcode() ==
      OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::SLICE)) {
    ++stats._slices;
  }
  if (META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) {
    uint32_t   count = 0;
    const int* slot  = node->Attr<int>(nn::core::ATTR::SLOT, &count);
    if (slot != nullptr) ++stats._slot_attrs;
    const int* rotations = node->Attr<int>(nn::core::ATTR::RNUM, &count);
    if (rotations != nullptr) {
      stats._rotations.emplace_back(rotations, rotations + count);
    }
  }

  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Collect_stats(stmt->Node(), stats);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Collect_stats(node->Child(idx), stats);
    }
  }
}

NATIVE_GEMM_STATS Get_stats(const NATIVE_GEMM_IR& fixture) {
  NATIVE_GEMM_STATS stats;
  const CONTAINER& container = fixture._func_scope->Container();
  Collect_stats(container.Stmt(fixture._func_scope->Entry_stmt_id())->Node(),
                stats);
  return stats;
}

std::string Snapshot(const NATIVE_GEMM_IR& fixture) {
  return nn::vector::test::Normalize_vector_kernel_function(
      *fixture._func_scope);
}

class Tensor2VectorM0Native : public ::testing::Test {
protected:
  void SetUp() override {
    _had_core   = META_INFO::Valid_domain(air::core::CORE);
    _had_nn     = META_INFO::Valid_domain(nn::core::NN);
    _had_vector = META_INFO::Valid_domain(VECTOR_DOMAIN::ID);
    if (!_had_core) ASSERT_TRUE(air::core::Register_core());
    if (!_had_nn) ASSERT_TRUE(nn::core::Register_nn());
    if (!_had_vector) ASSERT_TRUE(Register_vector_domain());
  }

  void TearDown() override {
    // META_INFO has no per-domain unregister API. Rebuild the three domains
    // used by ut_nnvector so this fixture leaves the process state unchanged.
    META_INFO::Remove_all();
    if (_had_core) EXPECT_TRUE(air::core::Register_core());
    if (_had_nn) EXPECT_TRUE(nn::core::Register_nn());
    if (_had_vector) EXPECT_TRUE(Register_vector_domain());
  }

private:
  bool _had_core   = false;
  bool _had_nn     = false;
  bool _had_vector = false;
};

}  // namespace

TEST_F(Tensor2VectorM0Native, BaselineGemmFourByFourDirectNativeOracle) {
  VECTOR_CONFIG default_config;
  EXPECT_FALSE(default_config.Python_dsl());

  NATIVE_GEMM_IR fixture =
      Build_native_gemm("baseline_gemm_4x4", 11, 4, 4, false);
  ASSERT_TRUE(fixture._glob->Verify_ir());
  ASSERT_EQ(fixture._result->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({8}));

  const NATIVE_GEMM_STATS stats = Get_stats(fixture);
  EXPECT_EQ(stats._loops, 1U);
  EXPECT_EQ(stats._vector_muls, 1U);
  EXPECT_EQ(stats._slices, 1U);
  EXPECT_EQ(stats._slot_attrs, 0U);
  EXPECT_EQ(stats._rotations,
            (std::vector<std::vector<int>>{{-4}, {0, 1, 2, 3}}));

  const std::string normalized = Snapshot(fixture);
  EXPECT_EQ(Vector_kernel_sha256(normalized),
            "7ce2561a9967a2465146865c27b99636"
            "8afb362aa8221ef3e824f0231210fb4a");

  NATIVE_GEMM_IR renamed =
      Build_native_gemm("renamed_native_oracle", 91, 4, 4, false);
  ASSERT_TRUE(renamed._glob->Verify_ir());
  EXPECT_EQ(normalized, Snapshot(renamed));
}

TEST_F(Tensor2VectorM0Native, BaselineGemmTwoByEightReductionAndMaskOracle) {
  NATIVE_GEMM_IR fixture =
      Build_native_gemm("baseline_gemm_2x8", 17, 2, 8, true);
  ASSERT_TRUE(fixture._glob->Verify_ir());
  ASSERT_EQ(fixture._result->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({16}));

  const NATIVE_GEMM_STATS stats = Get_stats(fixture);
  EXPECT_EQ(stats._loops, 2U);
  EXPECT_EQ(stats._vector_muls, 2U);
  EXPECT_EQ(stats._slices, 1U);
  EXPECT_EQ(stats._slot_attrs, 0U);
  EXPECT_EQ(stats._rotations,
            (std::vector<std::vector<int>>{{-8}, {0, 1}, {2, 4}}));

  EXPECT_EQ(Vector_kernel_sha256(Snapshot(fixture)),
            "fe7aa356576094f22cc23ca600fea792"
            "b8545340bae40312051d7650c8123e34");
}
