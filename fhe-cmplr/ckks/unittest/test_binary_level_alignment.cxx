//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "air/base/container.h"
#include "air/base/meta_info.h"
#include "air/core/opcode.h"
#include "air/driver/driver_ctx.h"
#include "fhe/ckks/ckks_gen.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/ckks/config.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/sihe/sihe_gen.h"
#include "fhe/sihe/sihe_opcode.h"
#include "gtest/gtest.h"
#include "scale_manager.h"

namespace {

using namespace air::base;
using namespace fhe::ckks;
using namespace fhe::core;

enum class SCALE_POLICY : uint32_t {
  ACE_ENTRY,
  EVA_ENTRY,
  PARS_ENTRY,
  REGION_ENTRY,
  EVA_CALLEE,
};

class BinaryLevelAlignmentTest : public testing::Test {
protected:
  void SetUp() override {
    META_INFO::Remove_all();
    ASSERT_TRUE(air::core::Register_core());
    ASSERT_TRUE(fhe::sihe::Register_sihe_domain());
    ASSERT_TRUE(Register_ckks_domain());

    _glob = new GLOB_SCOPE(0, true);
    _spos = _glob->Unknown_simple_spos();
    fhe::sihe::SIHE_GEN(_glob, &_lower_ctx).Register_sihe_types();
    CKKS_GEN(_glob, &_lower_ctx).Register_ckks_types();
    _cipher = _lower_ctx.Get_cipher_type(_glob);

    CTX_PARAM& parameters = _lower_ctx.Get_ctx_param();
    parameters.Set_poly_degree(32, false);
    parameters.Set_mul_level(6, false);
    parameters.Set_input_level(0);
    parameters.Set_first_prime_bit_num(50);
    parameters.Set_scaling_factor_bit_num(40);
    parameters.Set_hamming_weight(8);
  }

  void TearDown() override { delete _glob; }

  OPCODE Binary_opcode(uint32_t selector) const {
    switch (selector) {
      case 0:
        return OPC_ADD;
      case 1:
        return OPC_SUB;
      case 2:
        return OPC_MUL;
      default:
        AIR_ASSERT(false);
        return OPC_ADD;
    }
  }

  NODE_PTR Cipher_load(CONTAINER* container, ADDR_DATUM_PTR formal,
                       uint32_t scale_degree,
                       uint32_t rescale_level) const {
    NODE_PTR load = container->New_ld(formal, _spos);
    const uint32_t explicit_coordinate = 1;
    load->Set_attr(FHE_ATTR_KIND::EXPLICIT_SCALE_COORDINATE,
                   &explicit_coordinate, 1);
    load->Set_attr(FHE_ATTR_KIND::SCALE, &scale_degree, 1);
    load->Set_attr(FHE_ATTR_KIND::RESCALE_LEVEL, &rescale_level, 1);
    return load;
  }

  uint32_t Attr(NODE_PTR node, const char* name) const {
    const uint32_t* value = node->Attr<uint32_t>(name);
    EXPECT_NE(value, nullptr);
    return value == nullptr ? UINT32_MAX : *value;
  }

  uint32_t Count_opcode(NODE_PTR node, OPCODE opcode) const {
    if (node == Null_ptr) return 0;
    uint32_t count = node->Opcode() == opcode ? 1 : 0;
    for (uint32_t child = 0; child < node->Num_child(); ++child) {
      count += Count_opcode(node->Child(child), opcode);
    }
    return count;
  }

  void Configure(CKKS_CONFIG& config, SCALE_POLICY policy) const {
    switch (policy) {
      case SCALE_POLICY::ACE_ENTRY:
      case SCALE_POLICY::EVA_CALLEE:
        break;
      case SCALE_POLICY::EVA_ENTRY:
        config._eva_waterline = true;
        break;
      case SCALE_POLICY::PARS_ENTRY:
        config._pars_rsc = true;
        break;
      case SCALE_POLICY::REGION_ENTRY:
        config._rgn_scl_bts_mng = true;
        break;
    }
  }

  struct FUNCTION_IR {
    FUNC_SCOPE*    Scope;
    ADDR_DATUM_PTR Formal;
    CONTAINER*     Container;
  };

  FUNCTION_IR New_function(SCALE_POLICY policy) {
    FUNC_PTR function = _glob->New_func("binary_level_alignment", _spos);
    FUNC_SCOPE* function_scope = &_glob->New_func_scope(function);
    SIGNATURE_TYPE_PTR signature = _glob->New_sig_type();
    _glob->New_param("input", _cipher, signature, _spos);
    _glob->New_ret_param(_cipher->Id(), signature->Id());
    signature->Set_complete();
    ENTRY_PTR entry_point =
        _glob->New_entry_point(signature, function, "binary_level_alignment",
                               _spos);
    if (policy != SCALE_POLICY::EVA_CALLEE) entry_point->Set_program_entry();

    CONTAINER* container = &function_scope->Container();
    STMT_PTR entry = container->New_func_entry(_spos, 1);
    ADDR_DATUM_PTR formal =
        function_scope->New_formal(_cipher->Id(), "input", _spos);
    entry->Node()->Set_child(0, container->New_idname(formal, _spos));
    return FUNCTION_IR{function_scope, formal, container};
  }

  GLOB_SCOPE* _glob = nullptr;
  LOWER_CTX   _lower_ctx;
  SPOS        _spos;
  TYPE_PTR    _cipher;
};

class BinaryLevelPolicyTest
    : public BinaryLevelAlignmentTest,
      public testing::WithParamInterface<std::tuple<uint32_t, SCALE_POLICY>> {};

TEST_P(BinaryLevelPolicyTest, PropagatesAlignedCoordinateThroughStoreAndLoad) {
  const auto [selector, policy] = GetParam();
  FUNCTION_IR ir                = New_function(policy);
  OPCODE     opcode             = Binary_opcode(selector);

  NODE_PTR lhs = Cipher_load(ir.Container, ir.Formal, 1, 0);
  NODE_PTR rhs = Cipher_load(ir.Container, ir.Formal, 1, 2);
  NODE_PTR binary =
      ir.Container->New_bin_arith(opcode, _cipher, lhs, rhs, _spos);
  ADDR_DATUM_PTR result = ir.Scope->New_var(_cipher, "result", _spos);
  STMT_PTR first_store = ir.Container->New_st(binary, result, _spos);
  ir.Container->Stmt_list().Append(first_store);

  NODE_PTR equal_lhs = ir.Container->New_ld(result, _spos);
  NODE_PTR equal_rhs = ir.Container->New_ld(result, _spos);
  NODE_PTR equal_add = ir.Container->New_bin_arith(
      OPC_ADD, _cipher, equal_lhs, equal_rhs, _spos);
  ADDR_DATUM_PTR final_result =
      ir.Scope->New_var(_cipher, "final_result", _spos);
  STMT_PTR second_store =
      ir.Container->New_st(equal_add, final_result, _spos);
  ir.Container->Stmt_list().Append(second_store);
  NODE_PTR return_load = ir.Container->New_ld(final_result, _spos);
  ir.Container->Stmt_list().Append(
      ir.Container->New_retv(return_load, _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG             config;
  Configure(config, policy);
  SCALE_MANAGER manager(&driver_context, &config, ir.Scope, &_lower_ctx);
  manager.Run();

  constexpr uint32_t target_level = 2;
  uint32_t binary_scale = selector == 2 ? 2 : 1;
  EXPECT_EQ(Attr(binary, FHE_ATTR_KIND::SCALE), binary_scale);
  EXPECT_EQ(Attr(binary, FHE_ATTR_KIND::RESCALE_LEVEL), target_level);
  ASSERT_EQ(binary->Child(0)->Opcode(), OPC_MODSWITCH);
  EXPECT_EQ(Count_opcode(binary->Child(0), OPC_MODSWITCH), 2U);
  EXPECT_EQ(Attr(binary->Child(0), FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(binary->Child(0), FHE_ATTR_KIND::RESCALE_LEVEL),
            target_level);
  EXPECT_EQ(Attr(binary->Child(1), FHE_ATTR_KIND::RESCALE_LEVEL),
            target_level);

  bool result_rescaled = selector == 2 &&
                         policy != SCALE_POLICY::PARS_ENTRY &&
                         policy != SCALE_POLICY::REGION_ENTRY;
  uint32_t downstream_level = target_level + (result_rescaled ? 1U : 0U);
  uint32_t downstream_scale = result_rescaled ? 1U : binary_scale;
  NODE_PTR first_value      = first_store->Node()->Child(0);
  EXPECT_EQ(first_value->Opcode(), result_rescaled ? OPC_RESCALE : opcode);
  EXPECT_EQ(Attr(first_store->Node(), FHE_ATTR_KIND::SCALE), downstream_scale);
  EXPECT_EQ(Attr(first_store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL),
            downstream_level);
  EXPECT_EQ(Attr(equal_lhs, FHE_ATTR_KIND::RESCALE_LEVEL), downstream_level);
  EXPECT_EQ(Attr(equal_rhs, FHE_ATTR_KIND::RESCALE_LEVEL), downstream_level);
  EXPECT_EQ(Count_opcode(equal_add, OPC_MODSWITCH), 0U);
  EXPECT_EQ(Attr(equal_add, FHE_ATTR_KIND::RESCALE_LEVEL), downstream_level);
  EXPECT_EQ(Attr(second_store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL),
            downstream_level);
  EXPECT_EQ(Attr(return_load, FHE_ATTR_KIND::RESCALE_LEVEL), downstream_level);
  EXPECT_EQ(Attr(return_load, FHE_ATTR_KIND::SCALE), downstream_scale);
  EXPECT_TRUE(_glob->Verify_ir());
}

INSTANTIATE_TEST_SUITE_P(
    AllScalePolicies, BinaryLevelPolicyTest,
    testing::Combine(
        testing::Values(0U, 1U, 2U),
        testing::Values(SCALE_POLICY::ACE_ENTRY, SCALE_POLICY::EVA_ENTRY,
                        SCALE_POLICY::PARS_ENTRY, SCALE_POLICY::REGION_ENTRY,
                        SCALE_POLICY::EVA_CALLEE)));

class ProjectedOperandAlignmentTest
    : public BinaryLevelAlignmentTest,
      public testing::WithParamInterface<SCALE_POLICY> {};

TEST_P(ProjectedOperandAlignmentTest,
       ModswitchWrapsScheduledOperandRescale) {
  SCALE_POLICY policy = GetParam();
  FUNCTION_IR ir      = New_function(policy);

  NODE_PTR lhs = Cipher_load(ir.Container, ir.Formal, 2, 0);
  NODE_PTR rhs = Cipher_load(ir.Container, ir.Formal, 1, 2);
  NODE_PTR add =
      ir.Container->New_bin_arith(OPC_ADD, _cipher, lhs, rhs, _spos);
  ADDR_DATUM_PTR result = ir.Scope->New_var(_cipher, "result", _spos);
  STMT_PTR store = ir.Container->New_st(add, result, _spos);
  ir.Container->Stmt_list().Append(store);
  NODE_PTR return_load = ir.Container->New_ld(result, _spos);
  ir.Container->Stmt_list().Append(
      ir.Container->New_retv(return_load, _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG             config;
  Configure(config, policy);
  SCALE_MANAGER manager(&driver_context, &config, ir.Scope, &_lower_ctx);
  manager.Run();

  NODE_PTR modswitch = add->Child(0);
  ASSERT_EQ(modswitch->Opcode(), OPC_MODSWITCH);
  EXPECT_EQ(Attr(modswitch, FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(modswitch, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  NODE_PTR rescale = modswitch->Child(0);
  ASSERT_EQ(rescale->Opcode(), OPC_RESCALE);
  EXPECT_EQ(Attr(rescale, FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(rescale, FHE_ATTR_KIND::RESCALE_LEVEL), 1U);
  EXPECT_EQ(rescale->Child(0), lhs);
  EXPECT_EQ(Attr(rhs, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(add, FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(add, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(return_load, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Count_opcode(add, OPC_RESCALE), 1U);
  EXPECT_EQ(Count_opcode(add, OPC_MODSWITCH), 1U);
  EXPECT_TRUE(_glob->Verify_ir());
}

INSTANTIATE_TEST_SUITE_P(AceAndPars, ProjectedOperandAlignmentTest,
                         testing::Values(SCALE_POLICY::ACE_ENTRY,
                                         SCALE_POLICY::PARS_ENTRY));

TEST_F(BinaryLevelAlignmentTest, AlignsCipherProducerWithoutSourceMetadata) {
  FUNCTION_IR ir = New_function(SCALE_POLICY::ACE_ENTRY);

  NODE_PTR input = Cipher_load(ir.Container, ir.Formal, 1, 0);
  NODE_PTR bootstrap =
      CKKS_GEN(ir.Container, &_lower_ctx).Gen_bootstrap(input, _spos);
  NODE_PTR rhs = Cipher_load(ir.Container, ir.Formal, 1, 2);
  NODE_PTR add =
      ir.Container->New_bin_arith(OPC_ADD, _cipher, bootstrap, rhs, _spos);
  ADDR_DATUM_PTR result = ir.Scope->New_var(_cipher, "result", _spos);
  STMT_PTR store = ir.Container->New_st(add, result, _spos);
  ir.Container->Stmt_list().Append(store);
  NODE_PTR return_load = ir.Container->New_ld(result, _spos);
  ir.Container->Stmt_list().Append(
      ir.Container->New_retv(return_load, _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG             config;
  SCALE_MANAGER manager(&driver_context, &config, ir.Scope, &_lower_ctx);
  manager.Run();

  EXPECT_EQ(bootstrap->Attr<uint32_t>(FHE_ATTR_KIND::SCALE), nullptr);
  EXPECT_EQ(bootstrap->Attr<uint32_t>(FHE_ATTR_KIND::RESCALE_LEVEL), nullptr);
  NODE_PTR outer = add->Child(0);
  ASSERT_EQ(outer->Opcode(), OPC_MODSWITCH);
  EXPECT_EQ(Count_opcode(outer, OPC_MODSWITCH), 2U);
  EXPECT_EQ(Attr(outer, FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(outer, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(outer->Child(0)->Child(0), bootstrap);
  EXPECT_EQ(Attr(add, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(return_load, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_TRUE(_glob->Verify_ir());
}

TEST_F(BinaryLevelAlignmentTest, LeavesSymbolicZeroUnwrapped) {
  FUNCTION_IR ir = New_function(SCALE_POLICY::ACE_ENTRY);

  NODE_PTR zero = ir.Container->New_zero(_cipher, _spos);
  NODE_PTR rhs = Cipher_load(ir.Container, ir.Formal, 1, 2);
  NODE_PTR add =
      ir.Container->New_bin_arith(OPC_ADD, _cipher, zero, rhs, _spos);
  ADDR_DATUM_PTR result = ir.Scope->New_var(_cipher, "result", _spos);
  STMT_PTR store = ir.Container->New_st(add, result, _spos);
  ir.Container->Stmt_list().Append(store);
  NODE_PTR return_load = ir.Container->New_ld(result, _spos);
  ir.Container->Stmt_list().Append(
      ir.Container->New_retv(return_load, _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG             config;
  SCALE_MANAGER manager(&driver_context, &config, ir.Scope, &_lower_ctx);
  manager.Run();

  EXPECT_EQ(add->Child(0), zero);
  EXPECT_EQ(Count_opcode(add, OPC_MODSWITCH), 0U);
  EXPECT_EQ(Attr(add, FHE_ATTR_KIND::SCALE), 1U);
  EXPECT_EQ(Attr(add, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_EQ(Attr(return_load, FHE_ATTR_KIND::RESCALE_LEVEL), 2U);
  EXPECT_TRUE(_glob->Verify_ir());
}

}  // namespace
