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

class ParsLevelMatchTest
    : public testing::TestWithParam<std::tuple<uint32_t, uint32_t, uint32_t>> {
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
    parameters.Set_mul_level(4, false);
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
                       uint32_t rescale_level) const {
    NODE_PTR load = container->New_ld(formal, _spos);
    const uint32_t explicit_coordinate = 1;
    const uint32_t scale_degree        = 1;
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

  void Expect_modswitch_chain(NODE_PTR outer, NODE_PTR original,
                              uint32_t source_level,
                              uint32_t target_level) const {
    NODE_PTR current = outer;
    for (uint32_t level = target_level; level > source_level; --level) {
      ASSERT_EQ(current->Opcode(), OPC_MODSWITCH);
      EXPECT_EQ(Attr(current, FHE_ATTR_KIND::SCALE), 1U);
      EXPECT_EQ(Attr(current, FHE_ATTR_KIND::RESCALE_LEVEL), level);
      ASSERT_EQ(current->Num_child(), 1U);
      current = current->Child(0);
    }
    EXPECT_EQ(current, original);
    EXPECT_EQ(Attr(original, FHE_ATTR_KIND::RESCALE_LEVEL), source_level);
  }

  GLOB_SCOPE* _glob = nullptr;
  LOWER_CTX   _lower_ctx;
  SPOS        _spos;
  TYPE_PTR    _cipher;
};

TEST_P(ParsLevelMatchTest, InsertsExactExplicitModswitchChain) {
  const auto [selector, lhs_level, rhs_level] = GetParam();
  OPCODE opcode = Binary_opcode(selector);

  FUNC_PTR function = _glob->New_func("pars_level_match", _spos);
  FUNC_SCOPE* function_scope = &_glob->New_func_scope(function);
  SIGNATURE_TYPE_PTR signature = _glob->New_sig_type();
  _glob->New_param("input", _cipher, signature, _spos);
  _glob->New_ret_param(_cipher->Id(), signature->Id());
  signature->Set_complete();
  _glob->New_entry_point(signature, function, "pars_level_match", _spos)
      ->Set_program_entry();

  CONTAINER* container = &function_scope->Container();
  STMT_PTR entry = container->New_func_entry(_spos, 1);
  ADDR_DATUM_PTR formal =
      function_scope->New_formal(_cipher->Id(), "input", _spos);
  entry->Node()->Set_child(0, container->New_idname(formal, _spos));

  NODE_PTR lhs = Cipher_load(container, formal, lhs_level);
  NODE_PTR rhs = Cipher_load(container, formal, rhs_level);
  NODE_PTR binary =
      container->New_bin_arith(opcode, _cipher, lhs, rhs, _spos);
  ADDR_DATUM_PTR result =
      function_scope->New_var(_cipher, "result", _spos);
  STMT_PTR store = container->New_st(binary, result, _spos);
  container->Stmt_list().Append(store);
  NODE_PTR reload = container->New_ld(result, _spos);
  container->Stmt_list().Append(container->New_retv(reload, _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG config;
  config._pars_rsc = true;
  SCALE_MANAGER manager(&driver_context, &config, function_scope, &_lower_ctx);
  manager.Run();

  uint32_t target_level = std::max(lhs_level, rhs_level);
  EXPECT_EQ(Attr(binary, FHE_ATTR_KIND::RESCALE_LEVEL), target_level);
  EXPECT_EQ(Attr(store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), target_level);
  EXPECT_EQ(Attr(reload, FHE_ATTR_KIND::RESCALE_LEVEL), target_level);
  EXPECT_EQ(Attr(binary, FHE_ATTR_KIND::SCALE), selector == 2 ? 2U : 1U);
  EXPECT_EQ(Attr(reload, FHE_ATTR_KIND::SCALE), selector == 2 ? 2U : 1U);
  EXPECT_EQ(Count_opcode(binary, OPC_RESCALE), 0U);
  EXPECT_EQ(Count_opcode(binary, OPC_MODSWITCH),
            target_level - std::min(lhs_level, rhs_level));

  if (lhs_level < rhs_level) {
    Expect_modswitch_chain(binary->Child(0), lhs, lhs_level, target_level);
    EXPECT_EQ(binary->Child(1), rhs);
  } else if (rhs_level < lhs_level) {
    EXPECT_EQ(binary->Child(0), lhs);
    Expect_modswitch_chain(binary->Child(1), rhs, rhs_level, target_level);
  } else {
    EXPECT_EQ(binary->Child(0), lhs);
    EXPECT_EQ(binary->Child(1), rhs);
  }
}

TEST_F(ParsLevelMatchTest, KeepsSharedExpressionAlignmentOccurrenceLocal) {
  FUNC_PTR function = _glob->New_func("pars_shared_level_match", _spos);
  FUNC_SCOPE* function_scope = &_glob->New_func_scope(function);
  SIGNATURE_TYPE_PTR signature = _glob->New_sig_type();
  _glob->New_param("input", _cipher, signature, _spos);
  _glob->New_ret_param(_cipher->Id(), signature->Id());
  signature->Set_complete();
  _glob->New_entry_point(signature, function, "pars_shared_level_match", _spos)
      ->Set_program_entry();

  CONTAINER* container = &function_scope->Container();
  STMT_PTR entry = container->New_func_entry(_spos, 1);
  ADDR_DATUM_PTR formal =
      function_scope->New_formal(_cipher->Id(), "input", _spos);
  entry->Node()->Set_child(0, container->New_idname(formal, _spos));

  NODE_PTR shared = Cipher_load(container, formal, 0);
  NODE_PTR high_peer = Cipher_load(container, formal, 3);
  NODE_PTR low_peer = Cipher_load(container, formal, 1);
  NODE_PTR high =
      container->New_bin_arith(OPC_ADD, _cipher, shared, high_peer, _spos);
  NODE_PTR low =
      container->New_bin_arith(OPC_ADD, _cipher, shared, low_peer, _spos);

  ADDR_DATUM_PTR high_result =
      function_scope->New_var(_cipher, "high_result", _spos);
  ADDR_DATUM_PTR low_result =
      function_scope->New_var(_cipher, "low_result", _spos);
  STMT_PTR high_store = container->New_st(high, high_result, _spos);
  STMT_PTR low_store = container->New_st(low, low_result, _spos);
  container->Stmt_list().Append(high_store);
  container->Stmt_list().Append(low_store);

  NODE_PTR high_reload = container->New_ld(high_result, _spos);
  NODE_PTR low_reload = container->New_ld(low_result, _spos);
  ADDR_DATUM_PTR high_copy =
      function_scope->New_var(_cipher, "high_copy", _spos);
  ADDR_DATUM_PTR low_copy =
      function_scope->New_var(_cipher, "low_copy", _spos);
  container->Stmt_list().Append(
      container->New_st(high_reload, high_copy, _spos));
  container->Stmt_list().Append(
      container->New_st(low_reload, low_copy, _spos));
  container->Stmt_list().Append(
      container->New_retv(container->New_ld(low_copy, _spos), _spos));

  air::driver::DRIVER_CTX driver_context;
  CKKS_CONFIG config;
  config._pars_rsc = true;
  SCALE_MANAGER manager(&driver_context, &config, function_scope, &_lower_ctx);
  manager.Run();

  Expect_modswitch_chain(high->Child(0), shared, 0, 3);
  Expect_modswitch_chain(low->Child(0), shared, 0, 1);
  EXPECT_EQ(high->Child(1), high_peer);
  EXPECT_EQ(low->Child(1), low_peer);
  EXPECT_EQ(Attr(shared, FHE_ATTR_KIND::RESCALE_LEVEL), 0U);
  EXPECT_EQ(Attr(high_store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), 3U);
  EXPECT_EQ(Attr(low_store->Node(), FHE_ATTR_KIND::RESCALE_LEVEL), 1U);
  EXPECT_EQ(Attr(high_reload, FHE_ATTR_KIND::RESCALE_LEVEL), 3U);
  EXPECT_EQ(Attr(low_reload, FHE_ATTR_KIND::RESCALE_LEVEL), 1U);
  EXPECT_EQ(Count_opcode(high, OPC_MODSWITCH) +
                Count_opcode(low, OPC_MODSWITCH),
            4U);
  EXPECT_TRUE(_glob->Verify_ir());
}

INSTANTIATE_TEST_SUITE_P(
    AddSubMulBothDirections, ParsLevelMatchTest,
    testing::Values(std::make_tuple(0U, 0U, 2U),
                    std::make_tuple(0U, 2U, 0U),
                    std::make_tuple(0U, 2U, 2U),
                    std::make_tuple(1U, 0U, 2U),
                    std::make_tuple(1U, 2U, 0U),
                    std::make_tuple(1U, 2U, 2U),
                    std::make_tuple(2U, 0U, 2U),
                    std::make_tuple(2U, 2U, 0U),
                    std::make_tuple(2U, 2U, 2U)));

}  // namespace
