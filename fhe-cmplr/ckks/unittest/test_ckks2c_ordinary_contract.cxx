//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <algorithm>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "air/base/container.h"
#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "ckks2c_verifier.h"
#include "fhe/ckks/ckks_gen.h"
#include "fhe/ckks/ckks2c_config.h"
#include "fhe/ckks/ckks2c_driver.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/ckks/config.h"
#include "fhe/ckks/phantom_context_manifest.h"
#include "fhe/core/ctx_param_ana.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/core/scheme_info.h"
#include "fhe/sihe/sihe_gen.h"
#include "gtest/gtest.h"
#include "nn/core/attr.h"

namespace {

using fhe::ckks::CKKS2C_VERIFIER;
using fhe::core::PROVIDER;
using namespace air::base;

constexpr const char* kPhantomSourcePreamble = R"(
#include "rt_phantom/rt_phantom.h"
)";

constexpr const char* kPhantomManifestSource = R"(
extern "C" const PHANTOM_CONTEXT_MANIFEST*
Get_phantom_context_manifest() { return nullptr; }
extern "C" const PHANTOM_RESOURCE_MANIFEST*
Get_phantom_resource_manifest() { return nullptr; }
)";

void SetValidContextParameters(fhe::core::CTX_PARAM& parameters) {
  parameters.Set_poly_degree(32, false);
  parameters.Set_mul_level(4, false);
  parameters.Set_first_prime_bit_num(50);
  parameters.Set_scaling_factor_bit_num(40);
  parameters.Set_q_part_num(2);
  parameters.Set_input_level(3);
  parameters.Set_hamming_weight(8);
  parameters.Set_security_level(0);
}

NODE_PTR FindOpcode(NODE_PTR node, OPCODE opcode) {
  if (node == Null_ptr) return Null_ptr;
  if (node->Opcode() == opcode) return node;
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      NODE_PTR found = FindOpcode(stmt->Node(), opcode);
      if (found != Null_ptr) return found;
    }
    return Null_ptr;
  }
  for (uint32_t child = 0; child < node->Num_child(); ++child) {
    NODE_PTR found = FindOpcode(node->Child(child), opcode);
    if (found != Null_ptr) return found;
  }
  return Null_ptr;
}

TEST(CKKS2COrdinaryContract, NormalizesScalarRotationKeys) {
  EXPECT_EQ(fhe::core::Normalize_scalar_rotation_index(-1, 8192), -1);
  EXPECT_EQ(fhe::core::Normalize_scalar_rotation_index(8192, 8192), 0);
  EXPECT_EQ(fhe::core::Normalize_scalar_rotation_index(8193, 8192), 1);
}

TEST(CKKS2COrdinaryContract, TracksRelinearizationRequirementSeparately) {
  fhe::core::CTX_PARAM param;
  EXPECT_FALSE(param.Relin_key_required());
  param.Require_relin_key();
  EXPECT_TRUE(param.Relin_key_required());
}

TEST(CKKS2COrdinaryContract, SeparatesRetainedResourceRequirements) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);
  parameters.Add_rotate_index(17);
  parameters.Add_rotate_batch({0, -1, -1, 17});
  parameters.Add_rotate_batch({3, 0});
  parameters.Require_conjugation_key();
  parameters.Require_raise_mod();
  parameters.Add_monomial_power(0);
  parameters.Add_monomial_power(63);

  const auto context =
      fhe::ckks::Build_phantom_context_descriptor(parameters);
  const auto resources =
      fhe::ckks::Build_phantom_resource_descriptor(parameters, context);

  EXPECT_EQ(context._resource_schema_version, 2u);
  EXPECT_EQ(resources._schema_version, 2u);
  EXPECT_EQ(resources._rotation_steps,
            (std::vector<int32_t>{-1, 1, 3}));
  EXPECT_EQ(resources._rotation_batches,
            (std::vector<std::vector<int32_t>>{{0, -1, -1, 17}, {3, 0}}));
  EXPECT_EQ(resources._monomial_powers,
            (std::vector<uint32_t>{0, 63}));
  EXPECT_NE(resources._flags & fhe::ckks::PHANTOM_RESOURCE_CONJUGATION_KEY,
            0u);
  EXPECT_NE(resources._flags & fhe::ckks::PHANTOM_RESOURCE_ROTATE_BATCH, 0u);
  EXPECT_NE(resources._flags & fhe::ckks::PHANTOM_RESOURCE_RAISE_MOD, 0u);
  EXPECT_NE(resources._flags & fhe::ckks::PHANTOM_RESOURCE_MONOMIALS, 0u);
  EXPECT_EQ(std::find(resources._rotation_steps.begin(),
                      resources._rotation_steps.end(), 63),
            resources._rotation_steps.end());

  const std::string json =
      fhe::ckks::Serialize_phantom_resource_descriptor(resources);
  EXPECT_NE(json.find("\"conjugation_key\":true"), std::string::npos);
  EXPECT_NE(json.find("\"rotate_batch\":true"), std::string::npos);
  EXPECT_NE(json.find("\"rotation_batches\":[[0,-1,-1,17],[3,0]]"),
            std::string::npos);
  EXPECT_NE(json.find("\"raise_mod\":true"), std::string::npos);
  EXPECT_NE(json.find("\"monomial_powers\":[0,63]"),
            std::string::npos);
}

TEST(CKKS2COrdinaryContract, NormalizesSignedMonomialPowers) {
  EXPECT_EQ(fhe::core::Normalize_monomial_power(-1, 32), 63u);
  EXPECT_EQ(fhe::core::Normalize_monomial_power(64, 32), 0u);
  EXPECT_EQ(fhe::core::Normalize_monomial_power(65, 32), 1u);
}

TEST(CKKS2COrdinaryContract, EmitsClosedEmptyResourceSchema) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);
  const auto context =
      fhe::ckks::Build_phantom_context_descriptor(parameters);
  const auto resources =
      fhe::ckks::Build_phantom_resource_descriptor(parameters, context);
  EXPECT_EQ(fhe::ckks::Serialize_phantom_resource_descriptor(resources),
            "{\"context_schema_version\":1,\"conjugation_key\":false,"
            "\"monomial_powers\":[],\"raise_mod\":false,"
            "\"relinearization_key\":false,\"rotate_batch\":false,"
            "\"rotation_batches\":[],\"rotation_steps\":[],"
            "\"schema_version\":2}");
}

TEST(CKKS2COrdinaryContract, RejectsNoncanonicalManifestMonomialPower) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);
  parameters.Add_monomial_power(64);
  const auto context =
      fhe::ckks::Build_phantom_context_descriptor(parameters);
  EXPECT_THROW(
      fhe::ckks::Build_phantom_resource_descriptor(parameters, context),
      std::invalid_argument);
}

TEST(CKKS2COrdinaryContract, DerivesContextFromCompilerParameters) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);

  const auto context =
      fhe::ckks::Build_phantom_context_descriptor(parameters);
  EXPECT_EQ(context._poly_degree, 32u);
  EXPECT_EQ(context._logical_slots, 16u);
  EXPECT_EQ(context._data_q_bit_sizes,
            (std::vector<uint32_t>{50, 40, 40, 40}));
  EXPECT_EQ(context._input_level, 3u);
  EXPECT_EQ(context._q_part_count, 2u);
  EXPECT_EQ(context._hamming_weight, 8u);
  ASSERT_FALSE(context._special_p_bit_sizes.empty());
  for (uint32_t bit_size : context._special_p_bit_sizes) {
    EXPECT_EQ(bit_size, parameters.Get_p_prime_bit_num());
  }

  const std::string first =
      fhe::ckks::Serialize_phantom_context_descriptor(context);
  const std::string second = fhe::ckks::Serialize_phantom_context_descriptor(
      fhe::ckks::Build_phantom_context_descriptor(parameters));
  EXPECT_EQ(first, second);
}

TEST(CKKS2COrdinaryContract, DistinguishesDifferentCompilerParameters) {
  fhe::core::CTX_PARAM first_parameters;
  first_parameters.Set_poly_degree(32, false);
  first_parameters.Set_mul_level(4, false);
  first_parameters.Set_first_prime_bit_num(50);
  first_parameters.Set_scaling_factor_bit_num(40);
  first_parameters.Set_q_part_num(2);
  first_parameters.Set_input_level(3);
  first_parameters.Set_hamming_weight(8);

  fhe::core::CTX_PARAM second_parameters;
  second_parameters.Set_poly_degree(64, false);
  second_parameters.Set_mul_level(5, false);
  second_parameters.Set_first_prime_bit_num(55);
  second_parameters.Set_scaling_factor_bit_num(45);
  second_parameters.Set_q_part_num(3);
  second_parameters.Set_input_level(4);
  second_parameters.Set_hamming_weight(12);

  const std::string first = fhe::ckks::Serialize_phantom_context_descriptor(
      fhe::ckks::Build_phantom_context_descriptor(first_parameters));
  const std::string second = fhe::ckks::Serialize_phantom_context_descriptor(
      fhe::ckks::Build_phantom_context_descriptor(second_parameters));
  EXPECT_NE(first, second);
}

class CKKSFullQPipelineTest : public testing::Test {
protected:
  void SetUp() override {
    META_INFO::Remove_all();
    ASSERT_TRUE(air::core::Register_core());
    ASSERT_TRUE(fhe::sihe::Register_sihe_domain());
    ASSERT_TRUE(fhe::ckks::Register_ckks_domain());

    _glob = new GLOB_SCOPE(0, true);
    _spos = _glob->Unknown_simple_spos();
    fhe::sihe::SIHE_GEN(_glob, &_lower_ctx).Register_sihe_types();
    fhe::ckks::CKKS_GEN(_glob, &_lower_ctx).Register_ckks_types();
    TYPE_PTR cipher = _lower_ctx.Get_cipher_type(_glob);

    fhe::core::CTX_PARAM& parameters = _lower_ctx.Get_ctx_param();
    parameters.Set_poly_degree(32, false);
    parameters.Set_mul_level(1, false);
    parameters.Set_first_prime_bit_num(60);
    parameters.Set_scaling_factor_bit_num(56);
    parameters.Set_q_part_num(1);
    parameters.Set_input_level(1);
    parameters.Set_hamming_weight(8);
    parameters.Set_security_level(0);

    FUNC_PTR           func       = _glob->New_func("raise_pipeline", _spos);
    FUNC_SCOPE*        func_scope = &_glob->New_func_scope(func);
    SIGNATURE_TYPE_PTR signature  = _glob->New_sig_type();
    _glob->New_param("input", cipher, signature, _spos);
    _glob->New_ret_param(cipher->Id(), signature->Id());
    signature->Set_complete();
    _glob->New_entry_point(signature, func, "raise_pipeline", _spos)
        ->Set_program_entry();

    CONTAINER*     container = &func_scope->Container();
    ADDR_DATUM_PTR input = func_scope->New_formal(cipher->Id(), "input", _spos);
    STMT_PTR       entry = container->New_func_entry(_spos, 1);
    entry->Node()->Set_child(0, container->New_idname(input, _spos));

    TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
    NODE_PTR raise =
        container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, cipher, _spos);
    raise->Set_child(0, container->New_ld(input, _spos));
    raise->Set_child(1, container->New_intconst(u32, 4, _spos));
    container->Stmt_list().Append(container->New_retv(raise, _spos));
  }

  void TearDown() override { delete _glob; }

  NODE_PTR Find_raise(NODE_PTR node) const {
    if (node == Null_ptr) return Null_ptr;
    if (node->Opcode() == fhe::ckks::OPC_RAISE_MOD) return node;
    if (node->Is_block()) {
      for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
           stmt          = stmt->Next()) {
        NODE_PTR found = Find_raise(stmt->Node());
        if (found != Null_ptr) return found;
      }
      return Null_ptr;
    }
    for (uint32_t index = 0; index < node->Num_child(); ++index) {
      NODE_PTR found = Find_raise(node->Child(index));
      if (found != Null_ptr) return found;
    }
    return Null_ptr;
  }

  fhe::ckks::CKKS_CONFIG Configured(uint32_t full_q_count) const {
    fhe::ckks::CKKS_CONFIG config;
    config._max_cipher_lvl   = full_q_count;
    config._input_cipher_lvl = 1;
    config._poly_deg         = 32;
    config._hamming_weight   = 8;
    return config;
  }

  GLOB_SCOPE*          _glob = nullptr;
  fhe::core::LOWER_CTX _lower_ctx;
  SPOS                 _spos;
};

TEST_F(CKKSFullQPipelineTest, ConfiguredFullQPrecedesScaleMetadata) {
  fhe::ckks::CKKS_CONFIG  config = Configured(4);
  air::driver::DRIVER_CTX driver_context;
  R_CODE                  status = R_CODE::INTERNAL;
  GLOB_SCOPE*             output = fhe::ckks::Ckks_driver(
      _glob, &_lower_ctx, &driver_context, &config, &status);
  ASSERT_EQ(status, R_CODE::NORMAL);
  ASSERT_NE(output, nullptr);
  _glob = output;

  NODE_PTR raise =
      Find_raise((*_glob->Begin_func_scope()).Container().Entry_node());
  ASSERT_NE(raise, Null_ptr);
  const uint32_t* input_rescale =
      raise->Child(0)->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL);
  const uint32_t* result_rescale =
      raise->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL);
  ASSERT_NE(input_rescale, nullptr);
  ASSERT_NE(result_rescale, nullptr);
  EXPECT_EQ(*input_rescale, 4u);
  EXPECT_EQ(*result_rescale, 1u);
  EXPECT_EQ(_lower_ctx.Get_ctx_param().Get_mul_level(), 4u);
}

TEST_F(CKKSFullQPipelineTest, MissingMclInfersRaiseTargetBeforeScaleMetadata) {
  fhe::ckks::CKKS_CONFIG  config = Configured(0);
  air::driver::DRIVER_CTX driver_context;
  R_CODE                  status = R_CODE::INTERNAL;
  GLOB_SCOPE*             output = fhe::ckks::Ckks_driver(
      _glob, &_lower_ctx, &driver_context, &config, &status);
  ASSERT_EQ(status, R_CODE::NORMAL);
  ASSERT_NE(output, nullptr);
  _glob = output;

  NODE_PTR raise =
      Find_raise((*_glob->Begin_func_scope()).Container().Entry_node());
  ASSERT_NE(raise, Null_ptr);
  const uint32_t* input_rescale =
      raise->Child(0)->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL);
  const uint32_t* result_rescale =
      raise->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL);
  ASSERT_NE(input_rescale, nullptr);
  ASSERT_NE(result_rescale, nullptr);
  EXPECT_EQ(*input_rescale, 4u);
  EXPECT_EQ(*result_rescale, 1u);
  EXPECT_EQ(_lower_ctx.Get_ctx_param().Get_mul_level(), 4u);
}

TEST_F(CKKSFullQPipelineTest, RaiseBeyondConfiguredFullQIsUserError) {
  fhe::ckks::CKKS_CONFIG  config = Configured(3);
  air::driver::DRIVER_CTX driver_context;
  R_CODE                  status = R_CODE::NORMAL;

  testing::internal::CaptureStderr();
  GLOB_SCOPE* output = fhe::ckks::Ckks_driver(
      _glob, &_lower_ctx, &driver_context, &config, &status);
  const std::string stderr_text = testing::internal::GetCapturedStderr();

  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(status, R_CODE::USER);
  EXPECT_EQ(_lower_ctx.Get_ctx_param().Get_mul_level(), 1u);
  NODE_PTR original_raise =
      Find_raise((*_glob->Begin_func_scope()).Container().Entry_node());
  ASSERT_NE(original_raise, Null_ptr);
  EXPECT_EQ(
      original_raise->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL),
      nullptr);
  EXPECT_NE(
      stderr_text.find(
          "raise_mod target_q_count exceeds configured full data-Q count"),
      std::string::npos);
}

TEST(CKKS2COrdinaryContract, DriverEstablishesRetainedLevelMetadata) {
  META_INFO::Remove_all();
  ASSERT_TRUE(air::core::Register_core());
  ASSERT_TRUE(fhe::sihe::Register_sihe_domain());
  ASSERT_TRUE(fhe::ckks::Register_ckks_domain());

  fhe::core::LOWER_CTX lower_ctx;
  GLOB_SCOPE*          input = new GLOB_SCOPE(0, true);
  const SPOS           spos  = input->Unknown_simple_spos();
  fhe::sihe::SIHE_GEN(input, &lower_ctx).Register_sihe_types();
  fhe::ckks::CKKS_GEN(input, &lower_ctx).Register_ckks_types();
  TYPE_PTR cipher = lower_ctx.Get_cipher_type(input);

  fhe::core::CTX_PARAM& parameters = lower_ctx.Get_ctx_param();
  parameters.Set_poly_degree(32, false);
  parameters.Set_mul_level(1, false);
  parameters.Set_first_prime_bit_num(60);
  parameters.Set_scaling_factor_bit_num(56);
  parameters.Set_q_part_num(1);
  parameters.Set_input_level(1);
  parameters.Set_hamming_weight(8);
  parameters.Set_security_level(0);

  FUNC_PTR    function = input->New_func("retained_level_pipeline", spos);
  FUNC_SCOPE* scope    = &input->New_func_scope(function);
  SIGNATURE_TYPE_PTR signature = input->New_sig_type();
  input->New_param("input", cipher, signature, spos);
  input->New_ret_param(cipher->Id(), signature->Id());
  signature->Set_complete();
  input->New_entry_point(signature, function, "retained_level_pipeline", spos)
      ->Set_program_entry();

  CONTAINER*     container = &scope->Container();
  ADDR_DATUM_PTR formal = scope->New_formal(cipher->Id(), "input", spos);
  STMT_PTR       entry  = container->New_func_entry(spos, 1);
  entry->Node()->Set_child(0, container->New_idname(formal, spos));

  TYPE_PTR u32 = input->Prim_type(PRIMITIVE_TYPE::INT_U32);
  TYPE_PTR i64 = input->Prim_type(PRIMITIVE_TYPE::INT_S64);
  TYPE_PTR i32 = input->Prim_type(PRIMITIVE_TYPE::INT_S32);

  NODE_PTR raise =
      container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, cipher, spos);
  raise->Set_child(0, container->New_ld(formal, spos));
  raise->Set_child(1, container->New_intconst(u32, 4, spos));
  ADDR_DATUM_PTR raised = scope->New_var(cipher, "raised", spos);
  container->Stmt_list().Append(container->New_st(raise, raised, spos));

  NODE_PTR mono =
      container->New_cust_node(fhe::ckks::OPC_MUL_MONO, cipher, spos);
  mono->Set_child(0, container->New_ld(raised, spos));
  mono->Set_child(1, container->New_intconst(i64, 16, spos));
  ADDR_DATUM_PTR multiplied = scope->New_var(cipher, "multiplied", spos);
  container->Stmt_list().Append(
      container->New_st(mono, multiplied, spos));

  NODE_PTR conjugate =
      container->New_cust_node(fhe::ckks::OPC_CONJUGATE, cipher, spos);
  conjugate->Set_child(0, container->New_ld(multiplied, spos));
  ADDR_DATUM_PTR conjugated = scope->New_var(cipher, "conjugated", spos);
  container->Stmt_list().Append(
      container->New_st(conjugate, conjugated, spos));

  TYPE_PTR batch_type = input->New_arr_type(
      input->New_str("retained_level_batch"), cipher, {4}, spos);
  NODE_PTR batch = container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, batch_type, spos);
  batch->Set_child(0, container->New_ld(conjugated, spos));
  const int32_t steps[] = {5, 0, -7, 5};
  batch->Set_attr(nn::core::ATTR::RNUM, steps, 4);
  ADDR_DATUM_PTR rotations = scope->New_var(batch_type, "rotations", spos);
  container->Stmt_list().Append(
      container->New_st(batch, rotations, spos));

  NODE_PTR address = container->New_array(
      container->New_lda(rotations, POINTER_KIND::FLAT64, spos), 1, spos);
  container->Set_array_idx(address, 0,
                           container->New_intconst(i32, 0, spos));
  NODE_PTR first = container->New_ild(address, spos);
  ADDR_DATUM_PTR output = scope->New_var(cipher, "output", spos);
  container->Stmt_list().Append(container->New_st(first, output, spos));
  container->Stmt_list().Append(
      container->New_retv(container->New_ld(output, spos), spos));

  fhe::ckks::CKKS_CONFIG config;
  config._poly_deg         = 32;
  config._max_cipher_lvl   = 4;
  config._input_cipher_lvl = 1;
  config._q0_bit_num       = 60;
  config._scale_factor_bit_num = 56;
  config._hamming_weight       = 8;
  air::driver::DRIVER_CTX driver_context;
  R_CODE                  status = R_CODE::INTERNAL;
  GLOB_SCOPE* result = fhe::ckks::Ckks_driver(
      input, &lower_ctx, &driver_context, &config, &status);
  ASSERT_EQ(status, R_CODE::NORMAL);
  ASSERT_NE(result, nullptr);

  NODE_PTR root = (*result->Begin_func_scope()).Container().Entry_node();
  raise         = FindOpcode(root, fhe::ckks::OPC_RAISE_MOD);
  mono          = FindOpcode(root, fhe::ckks::OPC_MUL_MONO);
  conjugate     = FindOpcode(root, fhe::ckks::OPC_CONJUGATE);
  batch         = FindOpcode(root, fhe::ckks::OPC_ROTATE_BATCH);
  ASSERT_NE(raise, Null_ptr);
  ASSERT_NE(mono, Null_ptr);
  ASSERT_NE(conjugate, Null_ptr);
  ASSERT_NE(batch, Null_ptr);

  auto expect_metadata = [](NODE_PTR node, uint32_t level, uint32_t scale,
                            uint32_t rescale) {
    const uint32_t* observed_level =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    const uint32_t* observed_scale =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::SCALE);
    const uint32_t* observed_rescale =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL);
    ASSERT_NE(observed_level, nullptr);
    ASSERT_NE(observed_scale, nullptr);
    ASSERT_NE(observed_rescale, nullptr);
    EXPECT_EQ(*observed_level, level);
    EXPECT_EQ(*observed_scale, scale);
    EXPECT_EQ(*observed_rescale, rescale);
  };

  expect_metadata(raise->Child(0), 1, 1, 4);
  expect_metadata(raise, 4, 1, 1);
  for (NODE_PTR preserved : {mono, conjugate, batch}) {
    expect_metadata(preserved->Child(0), 4, 1, 1);
    expect_metadata(preserved, 4, 1, 1);
  }
  EXPECT_EQ(lower_ctx.Get_ctx_param().Get_mul_level(), 4u);

  std::string diagnostic;
  EXPECT_TRUE(CKKS2C_VERIFIER::Verify(
      result, lower_ctx.Get_ctx_param(), PROVIDER::PHANTOM, &diagnostic))
      << diagnostic;
  delete result;
}

TEST(CKKS2COrdinaryContract, RejectsProviderUnsupportedPolynomialDegree) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);
  parameters.Set_poly_degree(262144, false);
  EXPECT_THROW(fhe::ckks::Build_phantom_context_descriptor(parameters),
               std::invalid_argument);
}

TEST(CKKS2COrdinaryContract, RejectsProviderUnsupportedDataModulusBits) {
  fhe::core::CTX_PARAM first_parameters;
  SetValidContextParameters(first_parameters);
  first_parameters.Set_first_prime_bit_num(1);
  EXPECT_THROW(fhe::ckks::Build_phantom_context_descriptor(first_parameters),
               std::invalid_argument);

  fhe::core::CTX_PARAM scaling_parameters;
  SetValidContextParameters(scaling_parameters);
  scaling_parameters.Set_scaling_factor_bit_num(1);
  EXPECT_THROW(fhe::ckks::Build_phantom_context_descriptor(scaling_parameters),
               std::invalid_argument);
}

TEST(CKKS2COrdinaryContract, RejectsProviderUnsupportedModulusCount) {
  fhe::core::CTX_PARAM parameters;
  SetValidContextParameters(parameters);
  parameters.Set_mul_level(64, false);
  parameters.Set_q_part_num(64);
  parameters.Set_input_level(1);
  EXPECT_THROW(fhe::ckks::Build_phantom_context_descriptor(parameters),
               std::invalid_argument);
}

TEST(CKKS2COrdinaryContract, RejectsMissingContextManifest) {
  std::string diagnostic;
  EXPECT_FALSE(CKKS2C_VERIFIER::Verify_source(
      kPhantomSourcePreamble, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C source is missing the context manifest");
}

TEST(CKKS2COrdinaryContract, RejectsRelinWithoutKeyRequirement) {
  std::string source = kPhantomSourcePreamble;
  source += kPhantomManifestSource;
  source += R"(
void ordinary_relin_call() { Relin(nullptr, nullptr); }
)";
  std::string diagnostic;
  EXPECT_FALSE(CKKS2C_VERIFIER::Verify_source(
      source, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C source with Relin requires "
            "PHANTOM_RESOURCE_RELIN_KEY");
}

TEST(CKKS2COrdinaryContract, AcceptsMatchingRelinKeyRequirement) {
  std::string source = kPhantomSourcePreamble;
  source += kPhantomManifestSource;
  source += R"(
constexpr uint64_t ordinary_resource_flags = PHANTOM_RESOURCE_RELIN_KEY;
void ordinary_relin_call() { Relin(nullptr, nullptr); }
)";
  std::string diagnostic;
  EXPECT_TRUE(CKKS2C_VERIFIER::Verify_source(
      source, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_TRUE(diagnostic.empty());
}

TEST(CKKS2COrdinaryContract, RejectsRotateWithoutKeyRequirement) {
  std::string source = kPhantomSourcePreamble;
  source += kPhantomManifestSource;
  source += R"(
void ordinary_rotate_call() { Rotate_ciph(nullptr, nullptr, -3); }
)";
  std::string diagnostic;
  EXPECT_FALSE(CKKS2C_VERIFIER::Verify_source(
      source, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C source with nonzero Rotate_ciph requires "
            "PHANTOM_RESOURCE_ROTATION_KEYS");
}

TEST(CKKS2COrdinaryContract, AcceptsZeroRotateWithoutKeyRequirement) {
  std::string source = kPhantomSourcePreamble;
  source += kPhantomManifestSource;
  source += R"(
void ordinary_rotate_call() { Rotate_ciph(nullptr, nullptr, 0); }
)";
  std::string diagnostic;
  EXPECT_TRUE(CKKS2C_VERIFIER::Verify_source(
      source, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_TRUE(diagnostic.empty());
}

TEST(CKKS2COrdinaryContract, AcceptsMatchingRotationKeyRequirement) {
  std::string source = kPhantomSourcePreamble;
  source += kPhantomManifestSource;
  source += R"(
constexpr uint64_t ordinary_resource_flags =
    PHANTOM_RESOURCE_ROTATION_KEYS;
void ordinary_rotate_call() { Rotate_ciph(nullptr, nullptr, 3); }
)";
  std::string diagnostic;
  EXPECT_TRUE(CKKS2C_VERIFIER::Verify_source(
      source, PROVIDER::PHANTOM, &diagnostic));
  EXPECT_TRUE(diagnostic.empty());
}

class CKKS2COrdinaryAirVerifier : public testing::Test {
protected:
  void SetUp() override {
    META_INFO::Remove_all();
    ASSERT_TRUE(air::core::Register_core());
    ASSERT_TRUE(fhe::sihe::Register_sihe_domain());
    ASSERT_TRUE(fhe::ckks::Register_ckks_domain());

    _glob = new GLOB_SCOPE(0, true);
    _spos = _glob->Unknown_simple_spos();
    fhe::sihe::SIHE_GEN(_glob, &_lower_ctx).Register_sihe_types();
    fhe::ckks::CKKS_GEN(_glob, &_lower_ctx).Register_ckks_types();
    _cipher = _lower_ctx.Get_cipher_type(_glob);
    _plain  = _lower_ctx.Get_plain_type(_glob);

    // Deliberately small context: verifier limits must follow compiler state,
    // not a backend-literal ring size or Q count.
    fhe::core::CTX_PARAM& param = _lower_ctx.Get_ctx_param();
    param.Set_poly_degree(32, false);  // 16 logical slots
    param.Set_mul_level(4, false);     // four active data-Q moduli
    param.Set_first_prime_bit_num(60);
    param.Set_scaling_factor_bit_num(20);
    param.Set_q_part_num(2);
    param.Set_input_level(4);
    param.Set_hamming_weight(8);

    FUNC_PTR func = _glob->New_func("ordinary_air", _spos);
    _func_scope   = &_glob->New_func_scope(func);
    SIGNATURE_TYPE_PTR sig = _glob->New_sig_type();
    _glob->New_ret_param(_cipher->Id(), sig->Id());
    sig->Set_complete();
    _glob->New_entry_point(sig, func, "ordinary_air", _spos)
        ->Set_program_entry();
    _container = &_func_scope->Container();
    _container->New_func_entry(_spos, 0);
    _cipher_input = _func_scope->New_var(_cipher, "input", _spos);
  }

  void TearDown() override { delete _glob; }

  NODE_PTR Float_constant(double value = 1.0) {
    TYPE_PTR f64 = _glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
    CONSTANT_PTR constant =
        _glob->New_const(CONSTANT_KIND::FLOAT, f64,
                         static_cast<long double>(value));
    return _container->New_ldc(constant, _spos);
  }

  NODE_PTR Encode(NODE_PTR data, NODE_PTR length, NODE_PTR scale_degree,
                  NODE_PTR logical_level) {
    NODE_PTR encode =
        _container->New_cust_node(fhe::ckks::OPC_ENCODE, _plain, _spos);
    encode->Set_child(0, data);
    encode->Set_child(1, length);
    encode->Set_child(2, scale_degree);
    encode->Set_child(3, logical_level);
    return encode;
  }

  NODE_PTR Encode(NODE_PTR data, int64_t length, int64_t scale_degree,
                  int64_t logical_level) {
    TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
    return Encode(data, _container->New_intconst(u32, length, _spos),
                  _container->New_intconst(u32, scale_degree, _spos),
                  _container->New_intconst(u32, logical_level, _spos));
  }

  NODE_PTR Use_plain(NODE_PTR plain) {
    return _container->New_bin_arith(
        fhe::ckks::OPC_MUL, _cipher,
        _container->New_ld(_cipher_input, _spos), plain, _spos);
  }

  NODE_PTR Cipher_load(uint32_t level = 2, uint32_t scale = 1,
                       uint32_t rescale_level = 0) {
    NODE_PTR load = _container->New_ld(_cipher_input, _spos);
    load->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &level, 1);
    load->Set_attr(fhe::core::FHE_ATTR_KIND::SCALE, &scale, 1);
    load->Set_attr(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL, &rescale_level, 1);
    return load;
  }

  void Set_metadata(NODE_PTR node, uint32_t level = 2,
                    uint32_t scale = 1, uint32_t rescale_level = 0) {
    node->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &level, 1);
    node->Set_attr(fhe::core::FHE_ATTR_KIND::SCALE, &scale, 1);
    node->Set_attr(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL, &rescale_level, 1);
  }

  TYPE_PTR Cipher_array(const std::vector<int64_t>& dimensions) {
    return _glob->New_arr_type(_glob->New_str("cipher_array"), _cipher,
                               dimensions, _spos);
  }

  bool Contains_opcode(NODE_PTR node, OPCODE opcode) const {
    if (node == Null_ptr) return false;
    if (node->Opcode() == opcode) return true;
    if (node->Is_block()) {
      for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
           stmt = stmt->Next()) {
        if (Contains_opcode(stmt->Node(), opcode)) return true;
      }
      return false;
    }
    for (uint32_t index = 0; index < node->Num_child(); ++index) {
      if (Contains_opcode(node->Child(index), opcode)) return true;
    }
    return false;
  }

  bool Verify(NODE_PTR expression, std::string* diagnostic) {
    ADDR_DATUM_PTR result =
        _func_scope->New_var(expression->Rtype(), "result", _spos);
    _container->Stmt_list().Append(
        _container->New_st(expression, result, _spos));
    NODE_PTR returned = expression->Rtype()->Is_array()
                            ? _container->New_ld(_cipher_input, _spos)
                            : _container->New_ld(result, _spos);
    _container->Stmt_list().Append(_container->New_retv(returned, _spos));
    return CKKS2C_VERIFIER::Verify(_glob, _lower_ctx.Get_ctx_param(),
                                   PROVIDER::PHANTOM, diagnostic);
  }

  GLOB_SCOPE*          _glob = nullptr;
  fhe::core::LOWER_CTX _lower_ctx;
  SPOS                 _spos;
  FUNC_SCOPE*          _func_scope = nullptr;
  CONTAINER*           _container  = nullptr;
  TYPE_PTR             _cipher;
  TYPE_PTR             _plain;
  ADDR_DATUM_PTR       _cipher_input;
};

TEST_F(CKKS2COrdinaryAirVerifier, AcceptsContextDerivedEncodeLimits) {
  std::string diagnostic;
  NODE_PTR encode = Encode(Float_constant(), 16, 1, 4);
  EXPECT_TRUE(Verify(Use_plain(encode), &diagnostic)) << diagnostic;
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsEncodeLengthBeyondContextSlots) {
  std::string diagnostic;
  NODE_PTR encode = Encode(Float_constant(), 17, 1, 4);
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode length must be in [1, 16], got 17");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsEncodeLevelBeyondContextQCount) {
  std::string diagnostic;
  NODE_PTR encode = Encode(Float_constant(), 1, 1, 5);
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(
      diagnostic,
      "Phantom CKKS2C encode logical_level must be in [1, 4], got 5");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsUnrepresentableEncodeScale) {
  std::string diagnostic;
  NODE_PTR encode = Encode(Float_constant(), 1, 3, 1);
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode scale_degree 3 is not representable at "
            "logical_level 1");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicEncodeLength) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  ADDR_DATUM_PTR dynamic_length =
      _func_scope->New_var(u32, "dynamic_length", _spos);
  NODE_PTR encode = Encode(
      Float_constant(), _container->New_ld(dynamic_length, _spos),
      _container->New_intconst(u32, 1, _spos),
      _container->New_intconst(u32, 1, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode requires constant length");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicEncodeScale) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  ADDR_DATUM_PTR dynamic_scale =
      _func_scope->New_var(u32, "dynamic_scale", _spos);
  NODE_PTR encode = Encode(
      Float_constant(), _container->New_intconst(u32, 1, _spos),
      _container->New_ld(dynamic_scale, _spos),
      _container->New_intconst(u32, 1, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode requires constant scale_degree");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicEncodeLevel) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  ADDR_DATUM_PTR dynamic_level =
      _func_scope->New_var(u32, "dynamic_level", _spos);
  NODE_PTR encode = Encode(
      Float_constant(), _container->New_intconst(u32, 1, _spos),
      _container->New_intconst(u32, 1, _spos),
      _container->New_ld(dynamic_level, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode requires constant logical_level");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsIntegerEncodeInput) {
  TYPE_PTR i32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
  NODE_PTR encode = Encode(_container->New_intconst(i32, 7, _spos), 1, 1, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(Use_plain(encode), &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C encode does not support integer input; use "
            "floating-point data");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsReverseScalarCipherForm) {
  NODE_PTR reverse = _container->New_bin_arith(
      fhe::ckks::OPC_SUB, _cipher, Float_constant(),
      _container->New_ld(_cipher_input, _spos), _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(reverse, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C sub does not support reverse "
            "plaintext/scalar-cipher operands");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsIntegerScalarArithmetic) {
  TYPE_PTR i32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
  NODE_PTR add = _container->New_bin_arith(
      fhe::ckks::OPC_ADD, _cipher,
      _container->New_ld(_cipher_input, _spos),
      _container->New_intconst(i32, 7, _spos), _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(add, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C add does not support integer scalar operands");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsVisibleAddLevelMismatch) {
  NODE_PTR lhs = _container->New_ld(_cipher_input, _spos);
  NODE_PTR rhs = _container->New_ld(_cipher_input, _spos);
  uint32_t lhs_level = 1;
  uint32_t rhs_level = 2;
  lhs->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &lhs_level, 1);
  rhs->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &rhs_level, 1);
  NODE_PTR add = _container->New_bin_arith(fhe::ckks::OPC_ADD, _cipher, lhs,
                                            rhs, _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(add, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C add requires matching logical levels when "
            "statically known: lhs=1, rhs=2");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsVisibleSubScaleMismatch) {
  NODE_PTR lhs = _container->New_ld(_cipher_input, _spos);
  NODE_PTR rhs = _container->New_ld(_cipher_input, _spos);
  uint32_t lhs_scale = 1;
  uint32_t rhs_scale = 2;
  lhs->Set_attr(fhe::core::FHE_ATTR_KIND::SCALE, &lhs_scale, 1);
  rhs->Set_attr(fhe::core::FHE_ATTR_KIND::SCALE, &rhs_scale, 1);
  NODE_PTR sub = _container->New_bin_arith(fhe::ckks::OPC_SUB, _cipher, lhs,
                                            rhs, _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(sub, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C sub requires matching scale degrees when "
            "statically known: lhs=1, rhs=2");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicScalarRotation) {
  TYPE_PTR i32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
  ADDR_DATUM_PTR dynamic_step =
      _func_scope->New_var(i32, "dynamic_step", _spos);
  NODE_PTR rotate = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE, _cipher, _spos);
  rotate->Set_child(0, _container->New_ld(_cipher_input, _spos));
  rotate->Set_child(1, _container->New_ld(dynamic_step, _spos));
  int32_t attr_step = 1;
  rotate->Set_attr(nn::core::ATTR::RNUM, &attr_step, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(rotate, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C rotate requires a constant signed 32-bit step");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsMissingRotationAttribute) {
  TYPE_PTR i32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
  NODE_PTR rotate = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE, _cipher, _spos);
  rotate->Set_child(0, _container->New_ld(_cipher_input, _spos));
  rotate->Set_child(1, _container->New_intconst(i32, 1, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(rotate, &diagnostic));
  EXPECT_EQ(
      diagnostic,
      "Phantom CKKS2C rotate requires one RNUM entry matching its constant "
      "step");
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsMismatchedRotationAttribute) {
  TYPE_PTR i32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
  NODE_PTR rotate = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE, _cipher, _spos);
  rotate->Set_child(0, _container->New_ld(_cipher_input, _spos));
  rotate->Set_child(1, _container->New_intconst(i32, 1, _spos));
  int32_t attr_step = -1;
  rotate->Set_attr(nn::core::ATTR::RNUM, &attr_step, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(rotate, &diagnostic));
  EXPECT_EQ(
      diagnostic,
      "Phantom CKKS2C rotate requires one RNUM entry matching its constant "
      "step");
}

TEST_F(CKKS2COrdinaryAirVerifier, CodegenDoesNotRepairMissingKeyResources) {
  fhe::ckks::CKKS2C_CONFIG config;
  config.Set_provider("phantom");
  std::ostringstream output;
  fhe::ckks::CKKS2C_DRIVER driver(output, _lower_ctx, config);

  EXPECT_EQ(driver.Ctx().Phantom_resource_descriptor()._flags, 0u);
  EXPECT_TRUE(
      driver.Ctx().Phantom_resource_descriptor()._rotation_steps.empty());
  EXPECT_THROW(driver.Ctx().Require_phantom_relinearization_key(),
               std::runtime_error);
  EXPECT_THROW(driver.Ctx().Require_phantom_rotation_key(1),
               std::runtime_error);
  EXPECT_EQ(driver.Ctx().Phantom_resource_descriptor()._flags, 0u);
  EXPECT_TRUE(
      driver.Ctx().Phantom_resource_descriptor()._rotation_steps.empty());
}

TEST_F(CKKS2COrdinaryAirVerifier, AcceptsRetainedConjugateMetadata) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, Cipher_load(2, 3, 1));
  Set_metadata(conjugate, 2, 3, 1);
  std::string diagnostic;
  EXPECT_TRUE(Verify(conjugate, &diagnostic)) << diagnostic;
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRetainedMetadataMismatchWithNode) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, Cipher_load(2, 3, 1));
  Set_metadata(conjugate, 2, 4, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(conjugate, &diagnostic));
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
  EXPECT_NE(diagnostic.find("conjugate"), std::string::npos);
  EXPECT_NE(diagnostic.find("AIR="), std::string::npos);
  EXPECT_NE(diagnostic.find("SCALE"), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsMissingRetainedResultMetadata) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, Cipher_load(2, 3, 1));
  std::string diagnostic;
  EXPECT_FALSE(Verify(conjugate, &diagnostic));
  EXPECT_NE(diagnostic.find("must preserve LEVEL metadata"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, DiagnosesMissingRetainedOperands) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(conjugate, &diagnostic));
  EXPECT_NE(diagnostic.find("matching CIPHERTEXT input/result types"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, DiagnosesRotateBatchMissingOperand) {
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, Cipher_array({1}), _spos);
  const int32_t step = 0;
  batch->Set_attr(nn::core::ATTR::RNUM, &step, 1);
  Set_metadata(batch);
  std::string diagnostic;
  EXPECT_FALSE(Verify(batch, &diagnostic));
  EXPECT_NE(diagnostic.find("requires a CIPHERTEXT operand"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, DiagnosesRaiseMissingOperand) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  Set_metadata(raise, 4, 1, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("matching CIPHERTEXT input/result types"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, DiagnosesMulMonoMissingOperand) {
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(1, _container->New_intconst(i64, 1, _spos));
  Set_metadata(mono);
  std::string diagnostic;
  EXPECT_FALSE(Verify(mono, &diagnostic));
  EXPECT_NE(diagnostic.find("matching CIPHERTEXT input/result types"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsMissingRetainedMetadataOnBothNodes) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, _container->New_ld(_cipher_input, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(conjugate, &diagnostic));
  EXPECT_NE(diagnostic.find("observed input=missing, result=missing"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRetainedResultTypeChange) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _plain, _spos);
  conjugate->Set_child(0, Cipher_load());
  std::string diagnostic;
  EXPECT_FALSE(Verify(Use_plain(conjugate), &diagnostic));
  EXPECT_NE(diagnostic.find("matching CIPHERTEXT input/result types"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RetainedOpDoesNotBypassOrdinaryMetadata) {
  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, Cipher_load(1, 1, 0));
  Set_metadata(conjugate, 1, 1, 0);
  NODE_PTR add = _container->New_bin_arith(
      fhe::ckks::OPC_ADD, _cipher, conjugate, Cipher_load(2, 1, 0), _spos);
  std::string diagnostic;
  EXPECT_FALSE(Verify(add, &diagnostic));
  EXPECT_EQ(diagnostic,
            "Phantom CKKS2C add requires matching logical levels when "
            "statically known: lhs=1, rhs=2");
}

TEST_F(CKKS2COrdinaryAirVerifier, AcceptsOrderedRotateBatchWithDuplicatesAndZero) {
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, Cipher_array({4}), _spos);
  batch->Set_child(0, Cipher_load(2, 1, 0));
  const int32_t steps[] = {0, -1, -1, 17};
  batch->Set_attr(nn::core::ATTR::RNUM, steps, 4);
  Set_metadata(batch, 2, 1, 0);
  std::string diagnostic;
  EXPECT_TRUE(Verify(batch, &diagnostic)) << diagnostic;
}

TEST_F(CKKS2COrdinaryAirVerifier,
       RecordsRotateBatchesInForwardStatementOrder) {
  auto append_batch = [this](const char* name,
                             const std::vector<int32_t>& steps) {
    TYPE_PTR batch_type = Cipher_array(
        {static_cast<int64_t>(steps.size())});
    NODE_PTR batch = _container->New_cust_node(
        fhe::ckks::OPC_ROTATE_BATCH, batch_type, _spos);
    batch->Set_child(0, Cipher_load(2, 1, 3));
    batch->Set_attr(nn::core::ATTR::RNUM, steps.data(),
                    static_cast<uint32_t>(steps.size()));
    Set_metadata(batch, 2, 1, 3);
    ADDR_DATUM_PTR result = _func_scope->New_var(batch_type, name, _spos);
    _container->Stmt_list().Append(
        _container->New_st(batch, result, _spos));
  };

  const std::vector<int32_t> first{5, 0, -7, 5};
  const std::vector<int32_t> second{256, 512};
  append_batch("first_batch", first);
  append_batch("second_batch", second);
  _container->Stmt_list().Append(
      _container->New_retv(Cipher_load(2, 1, 3), _spos));

  fhe::ckks::CKKS_CONFIG config;
  config._poly_deg         = 32;
  config._max_cipher_lvl   = 4;
  config._input_cipher_lvl = 4;
  air::driver::DRIVER_CTX driver_context;
  fhe::core::CTX_PARAM_ANA analysis(_func_scope, &_lower_ctx,
                                    &driver_context, &config);
  ASSERT_EQ(analysis.Run(), R_CODE::NORMAL);
  EXPECT_EQ(_lower_ctx.Get_ctx_param().Get_rotate_batches(),
            (std::vector<std::vector<int32_t>>{first, second}));
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRotateBatchMissingRnum) {
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, Cipher_array({1}), _spos);
  batch->Set_child(0, Cipher_load());
  Set_metadata(batch);
  std::string diagnostic;
  EXPECT_FALSE(Verify(batch, &diagnostic));
  EXPECT_NE(diagnostic.find("constant, non-empty ordered RNUM"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("opcode="), std::string::npos);
  EXPECT_NE(diagnostic.find("rotate_batch"), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRotateBatchWrongRank) {
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, Cipher_array({2, 2}), _spos);
  batch->Set_child(0, Cipher_load());
  const int32_t steps[] = {1, 2, 3, 4};
  batch->Set_attr(nn::core::ATTR::RNUM, steps, 4);
  Set_metadata(batch);
  std::string diagnostic;
  EXPECT_FALSE(Verify(batch, &diagnostic));
  EXPECT_NE(diagnostic.find("one-dimensional CIPHERTEXT array"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRotateBatchLengthMismatch) {
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, Cipher_array({2}), _spos);
  batch->Set_child(0, Cipher_load());
  const int32_t steps[] = {0, 1, 2};
  batch->Set_attr(nn::core::ATTR::RNUM, steps, 3);
  Set_metadata(batch);
  std::string diagnostic;
  EXPECT_FALSE(Verify(batch, &diagnostic));
  EXPECT_NE(diagnostic.find("exact length matches RNUM"), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, AcceptsFullQRaiseFromBottom) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1, 2, 4));
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  Set_metadata(raise, 4, 2, 1);
  std::string diagnostic;
  EXPECT_TRUE(Verify(raise, &diagnostic)) << diagnostic;
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsPartialRaiseTarget) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1));
  raise->Set_child(1, _container->New_intconst(u32, 3, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("v1 requires the full data-Q count 4"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRaiseTargetBeyondContext) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1));
  raise->Set_child(1, _container->New_intconst(u32, 5, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("target_q_count must be in [1, 4], got 5"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRaiseFromNonBottomInput) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(2, 1, 2));
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  Set_metadata(raise, 4, 1, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("bottom one-Q input with LEVEL=1"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRaiseRescaleMetadataChange) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1, 2, 4));
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  Set_metadata(raise, 4, 2, 3);
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("level coordinates disagree"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRaiseWithMissingScaleMetadata) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR input = _container->New_ld(_cipher_input, _spos);
  uint32_t input_level = 1;
  uint32_t input_rescale = 4;
  input->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &input_level, 1);
  input->Set_attr(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL, &input_rescale, 1);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, input);
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  uint32_t result_level = 4;
  uint32_t result_rescale = 1;
  raise->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &result_level, 1);
  raise->Set_attr(fhe::core::FHE_ATTR_KIND::RESCALE_LEVEL, &result_rescale, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("preserve SCALE metadata"), std::string::npos);
  EXPECT_NE(diagnostic.find("input=missing, result=missing"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsRuntimeRaiseHelperAttribute) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1));
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  uint32_t runtime = 1;
  raise->Set_attr(fhe::core::FHE_ATTR_KIND::RUNTIME_RAISE_LEVEL, &runtime, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("forbids runtime target-level helpers"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicRaiseTarget) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  ADDR_DATUM_PTR target = _func_scope->New_var(u32, "target", _spos);
  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1));
  raise->Set_child(1, _container->New_ld(target, _spos));
  std::string diagnostic;
  EXPECT_FALSE(Verify(raise, &diagnostic));
  EXPECT_NE(diagnostic.find("constant target_q_count"), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsUnnormalizedSignedMulMonoPower) {
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(0, Cipher_load(2, 3, 1));
  mono->Set_child(1, _container->New_intconst(i64, -1, _spos));
  Set_metadata(mono, 2, 3, 1);
  std::string diagnostic;
  EXPECT_FALSE(Verify(mono, &diagnostic));
  EXPECT_NE(diagnostic.find("power must be normalized into [0, 64), got -1"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, AcceptsNormalizedMulMonoPower) {
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(0, Cipher_load(2, 3, 1));
  mono->Set_child(1, _container->New_intconst(i64, 63, _spos));
  Set_metadata(mono, 2, 3, 1);
  std::string diagnostic;
  EXPECT_TRUE(Verify(mono, &diagnostic)) << diagnostic;
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsMulMonoRotationMetadata) {
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(0, Cipher_load());
  mono->Set_child(1, _container->New_intconst(i64, -1, _spos));
  int32_t invalid_rotation = -1;
  mono->Set_attr(nn::core::ATTR::RNUM, &invalid_rotation, 1);
  Set_metadata(mono);
  std::string diagnostic;
  EXPECT_FALSE(Verify(mono, &diagnostic));
  EXPECT_NE(diagnostic.find("forbids rotation RNUM metadata"),
            std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, RejectsDynamicMulMonoPower) {
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
  ADDR_DATUM_PTR power = _func_scope->New_var(i64, "power", _spos);
  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(0, Cipher_load());
  mono->Set_child(1, _container->New_ld(power, _spos));
  Set_metadata(mono);
  std::string diagnostic;
  EXPECT_FALSE(Verify(mono, &diagnostic));
  EXPECT_NE(diagnostic.find("constant monomial power"), std::string::npos);
}

TEST_F(CKKS2COrdinaryAirVerifier, EmitsExactRetainedCallsAndResources) {
  TYPE_PTR u32 = _glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  TYPE_PTR i64 = _glob->Prim_type(PRIMITIVE_TYPE::INT_S64);

  NODE_PTR conjugate =
      _container->New_cust_node(fhe::ckks::OPC_CONJUGATE, _cipher, _spos);
  conjugate->Set_child(0, Cipher_load(2, 1, 3));
  Set_metadata(conjugate, 2, 1, 3);
  ADDR_DATUM_PTR conjugated =
      _func_scope->New_var(_cipher, "conjugated", _spos);
  _container->Stmt_list().Append(
      _container->New_st(conjugate, conjugated, _spos));

  TYPE_PTR batch_type = Cipher_array({4});
  NODE_PTR batch = _container->New_cust_node(
      fhe::ckks::OPC_ROTATE_BATCH, batch_type, _spos);
  batch->Set_child(0, Cipher_load(2, 1, 3));
  const int32_t batch_steps[] = {0, -1, -1, 17};
  batch->Set_attr(nn::core::ATTR::RNUM, batch_steps, 4);
  Set_metadata(batch, 2, 1, 3);
  ADDR_DATUM_PTR rotated =
      _func_scope->New_var(batch_type, "rotated", _spos);
  _container->Stmt_list().Append(_container->New_st(batch, rotated, _spos));

  NODE_PTR raise =
      _container->New_cust_node(fhe::ckks::OPC_RAISE_MOD, _cipher, _spos);
  raise->Set_child(0, Cipher_load(1, 1, 4));
  raise->Set_child(1, _container->New_intconst(u32, 4, _spos));
  Set_metadata(raise, 4, 1, 1);
  ADDR_DATUM_PTR raised = _func_scope->New_var(_cipher, "raised", _spos);
  _container->Stmt_list().Append(_container->New_st(raise, raised, _spos));

  NODE_PTR mono =
      _container->New_cust_node(fhe::ckks::OPC_MUL_MONO, _cipher, _spos);
  mono->Set_child(0, Cipher_load(2, 1, 3));
  mono->Set_child(1, _container->New_intconst(i64, -1, _spos));
  Set_metadata(mono, 2, 1, 3);
  ADDR_DATUM_PTR monomial =
      _func_scope->New_var(_cipher, "monomial", _spos);
  _container->Stmt_list().Append(_container->New_st(mono, monomial, _spos));
  _container->Stmt_list().Append(
      _container->New_retv(_container->New_ld(monomial, _spos), _spos));

  EXPECT_EQ(conjugate->Opcode(), fhe::ckks::OPC_CONJUGATE);
  EXPECT_EQ(batch->Opcode(), fhe::ckks::OPC_ROTATE_BATCH);
  EXPECT_EQ(raise->Opcode(), fhe::ckks::OPC_RAISE_MOD);
  EXPECT_EQ(mono->Opcode(), fhe::ckks::OPC_MUL_MONO);

  air::driver::DRIVER_CTX driver_context;
  fhe::ckks::CKKS_CONFIG analysis_config;
  {
    fhe::core::CTX_PARAM_ANA analysis(_func_scope, &_lower_ctx,
                                      &driver_context, &analysis_config);
    ASSERT_EQ(analysis.Run(), R_CODE::NORMAL);
  }

  const uint32_t* raise_result_level =
      raise->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
  const uint32_t* raise_source_level =
      raise->Child(0)->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
  ASSERT_NE(raise_result_level, nullptr);
  ASSERT_NE(raise_source_level, nullptr);
  EXPECT_EQ(*raise_result_level, 4u);
  EXPECT_EQ(*raise_source_level, 1u);
  EXPECT_EQ(mono->Child(1)->Intconst(), 63u);
  EXPECT_EQ(conjugate->Opcode(), fhe::ckks::OPC_CONJUGATE);
  EXPECT_EQ(batch->Opcode(), fhe::ckks::OPC_ROTATE_BATCH);
  EXPECT_EQ(raise->Opcode(), fhe::ckks::OPC_RAISE_MOD);
  EXPECT_EQ(mono->Opcode(), fhe::ckks::OPC_MUL_MONO);

  const fhe::core::CTX_PARAM& parameters = _lower_ctx.Get_ctx_param();
  EXPECT_TRUE(parameters.Conjugation_key_required());
  EXPECT_TRUE(parameters.Rotate_batch_required());
  EXPECT_EQ(parameters.Get_rotate_batches(),
            (std::vector<std::vector<int32_t>>{{0, -1, -1, 17}}));
  EXPECT_EQ(parameters.Get_rotate_index(), (std::set<int32_t>{-1, 1}));
  EXPECT_TRUE(parameters.Raise_mod_required());
  EXPECT_EQ(parameters.Get_monomial_powers(), (std::set<uint32_t>{63}));

  fhe::ckks::CKKS2C_CONFIG config;
  config.Set_provider("phantom");
  std::ostringstream output;
  fhe::ckks::CKKS2C_DRIVER driver(output, _lower_ctx, config);
  driver.Verify_or_throw(_glob);
  _glob = driver.Flatten(_glob);
  auto flattened_function = _glob->Begin_func_scope();
  ASSERT_NE(flattened_function, _glob->End_func_scope());
  NODE_PTR flattened_entry = (*flattened_function).Container().Entry_node();
  EXPECT_TRUE(Contains_opcode(flattened_entry, fhe::ckks::OPC_CONJUGATE));
  EXPECT_TRUE(Contains_opcode(flattened_entry, fhe::ckks::OPC_ROTATE_BATCH));
  EXPECT_TRUE(Contains_opcode(flattened_entry, fhe::ckks::OPC_RAISE_MOD));
  EXPECT_TRUE(Contains_opcode(flattened_entry, fhe::ckks::OPC_MUL_MONO));
  fhe::ckks::CKKS2C_VISITOR visitor(driver.Ctx());
  driver.Run(_glob, visitor);
  const std::string source = output.str();
  fhe::ckks::CKKS2C_DRIVER::Verify_source_or_throw(source,
                                                   PROVIDER::PHANTOM);

  auto generated_cipher_symbol = [&source](const std::string& stem,
                                            bool is_array) {
    const std::regex declaration(
        "(^|\\n)  CIPHERTEXT (" + stem +
        "_[0-9]+)" + (is_array ? "\\[[0-9]+\\]" : "") +
        ";(\\n|$)");
    std::smatch match;
    const bool  found = std::regex_search(source, match, declaration);
    EXPECT_TRUE(found) << "missing generated declaration for " << stem;
    return found ? match[2].str() : std::string();
  };
  const std::string input_symbol = generated_cipher_symbol("input", false);
  const std::string conjugated_symbol =
      generated_cipher_symbol("conjugated", false);
  const std::string rotated_symbol =
      generated_cipher_symbol("rotated", true);
  const std::string raised_symbol = generated_cipher_symbol("raised", false);
  const std::string monomial_symbol =
      generated_cipher_symbol("monomial", false);

  const std::size_t input_registration =
      source.find("Register_ciph_lifetime(&" + input_symbol + ")");
  const std::size_t first_retained_call = source.find("Conjugate_ciph(");
  ASSERT_NE(input_registration, std::string::npos);
  ASSERT_NE(first_retained_call, std::string::npos);
  EXPECT_LT(input_registration, first_retained_call);
  EXPECT_NE(source.find("Register_ciph_lifetime(&" + conjugated_symbol + ")"),
            std::string::npos);
  EXPECT_NE(source.find("Register_ciph_lifetime(&" + raised_symbol + ")"),
            std::string::npos);
  EXPECT_NE(source.find("Register_ciph_lifetime(&" + monomial_symbol + ")"),
            std::string::npos);
  EXPECT_NE(source.find("Register_ciph_array_lifetime(" + rotated_symbol +
                        ", sizeof(" + rotated_symbol + ") / sizeof(" +
                        rotated_symbol + "[0]))"),
            std::string::npos);
  EXPECT_NE(source.find("Conjugate_ciph(&" + conjugated_symbol + ", &" +
                        input_symbol + ")"),
            std::string::npos);
  EXPECT_NE(source.find(
                "static const int32_t _rot_batch_"),
            std::string::npos);
  EXPECT_NE(source.find("[] = {0, -1, -1, 17}; Rotate_batch_ciph("),
            std::string::npos);
  EXPECT_TRUE(std::regex_search(
      source,
      std::regex("Rotate_batch_ciph\\(" + rotated_symbol + ", &" +
                 input_symbol + ", _rot_batch_[0-9]+, 4\\)")));
  EXPECT_NE(source.find("Raise_mod(&" + raised_symbol + ", &" + input_symbol +
                        ", 4)"),
            std::string::npos);
  EXPECT_NE(source.find("Mul_mono_ciph(&" + monomial_symbol + ", &" +
                        input_symbol + ", 63)"),
            std::string::npos);
  EXPECT_NE(source.find("phantom_rotation_batch_offsets[] = {0, 4}"),
            std::string::npos);
  EXPECT_NE(source.find(
                "phantom_rotation_batch_steps[] = {0, -1, -1, 17}"),
            std::string::npos);
  EXPECT_NE(source.find("phantom_monomial_powers[] = {63}"),
            std::string::npos);
  EXPECT_EQ(source.find("Rotate_ciph("), std::string::npos);
  EXPECT_EQ(source.find("Eval_bootstrap"), std::string::npos);
  EXPECT_EQ(source.find("Bootstrap("), std::string::npos);
  EXPECT_EQ(source.find("bootstrap_coeffs_to_slots"), std::string::npos);
  EXPECT_EQ(source.find("bootstrap_eval_mod"), std::string::npos);
  EXPECT_EQ(source.find("bootstrap_slots_to_coeffs"), std::string::npos);
  EXPECT_EQ(source.find("rt_ant"), std::string::npos);
  EXPECT_EQ(source.find("LIB_ANT"), std::string::npos);
  EXPECT_EQ(source.find("Hw_"), std::string::npos);
  EXPECT_EQ(source.find("Poly_"), std::string::npos);
  EXPECT_EQ(source.find("rt_poly"), std::string::npos);
  EXPECT_EQ(source.find("fhe/poly"), std::string::npos);
}

}  // namespace
