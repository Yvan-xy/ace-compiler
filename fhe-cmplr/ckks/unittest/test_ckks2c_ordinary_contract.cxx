//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <string>
#include <vector>

#include "air/base/container.h"
#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "ckks2c_verifier.h"
#include "fhe/ckks/ckks_gen.h"
#include "fhe/ckks/ckks_opcode.h"
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

TEST(CKKS2COrdinaryContract, DerivesContextFromCompilerParameters) {
  fhe::core::CTX_PARAM parameters;
  parameters.Set_poly_degree(32, false);
  parameters.Set_mul_level(4, false);
  parameters.Set_first_prime_bit_num(50);
  parameters.Set_scaling_factor_bit_num(40);
  parameters.Set_q_part_num(2);
  parameters.Set_input_level(3);
  parameters.Set_hamming_weight(8);
  parameters.Set_security_level(0);

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

  bool Verify(NODE_PTR expression, std::string* diagnostic) {
    ADDR_DATUM_PTR result =
        _func_scope->New_var(_cipher, "result", _spos);
    _container->Stmt_list().Append(
        _container->New_st(expression, result, _spos));
    _container->Stmt_list().Append(
        _container->New_retv(_container->New_ld(result, _spos), _spos));
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

}  // namespace
