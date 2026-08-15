//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/test/lower_ckks.h"

#include <algorithm>
#include <set>
#include <string_view>
#include <vector>

using namespace air::base;
using namespace fhe::core;
using namespace fhe::poly;

namespace fhe {
namespace poly {
namespace test {

namespace {

uint32_t Count_opcode(NODE_PTR node, air::base::OPCODE opcode) {
  if (node == Null_ptr) return 0;
  uint32_t count = node->Opcode() == opcode ? 1 : 0;
  if (node->Is_block()) {
    STMT_LIST statements(node);
    for (STMT_PTR statement = statements.Begin_stmt();
         statement != statements.End_stmt(); statement = statement->Next()) {
      count += Count_opcode(statement->Node(), opcode);
    }
    return count;
  }
  for (uint32_t child = 0; child < node->Num_child(); ++child) {
    count += Count_opcode(node->Child(child), opcode);
  }
  return count;
}

void Collect_opcode(NODE_PTR node, air::base::OPCODE opcode,
                    std::vector<NODE_PTR>& result) {
  if (node == Null_ptr) return;
  if (node->Opcode() == opcode) result.push_back(node);
  if (node->Is_block()) {
    STMT_LIST statements(node);
    for (STMT_PTR statement = statements.Begin_stmt();
         statement != statements.End_stmt(); statement = statement->Next()) {
      Collect_opcode(statement->Node(), opcode, result);
    }
    return;
  }
  for (uint32_t child = 0; child < node->Num_child(); ++child) {
    Collect_opcode(node->Child(child), opcode, result);
  }
}

void Collect_pgen_symbols(NODE_PTR node, std::set<uint32_t>& result) {
  if (node == Null_ptr) return;
  if (node->Has_sym()) {
    SYM_PTR          symbol = node->Addr_datum()->Base_sym();
    std::string_view name(symbol->Name()->Char_str());
    if (name.size() >= 6 && name.substr(0, 6) == "_pgen_") {
      result.insert(symbol->Id().Value());
    }
  }
  if (node->Is_block()) {
    STMT_LIST statements(node);
    for (STMT_PTR statement = statements.Begin_stmt();
         statement != statements.End_stmt(); statement = statement->Next()) {
      Collect_pgen_symbols(statement->Node(), result);
    }
    return;
  }
  for (uint32_t child = 0; child < node->Num_child(); ++child) {
    Collect_pgen_symbols(node->Child(child), result);
  }
}

}  // namespace

class TEST_CKKS2HPOLY : public TEST_LOWER_CKKS<TEST_CONFIG> {};

TEST_P(TEST_CKKS2HPOLY, Handle_add_ciph) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_add_plain) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_p());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_mul_plain) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_mul(Container(), Var_z(), Var_x(), Var_p());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_mul_ciph) {
  STMT_PTR stmt =
      Ckks_ir_gen().Gen_mul(Container(), Var_ciph3(), Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_mul_ciph_preg) {
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph3_ty());
  STMT_PTR stmt = Ckks_ir_gen().Gen_mul(Container(), preg_z, Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_mul_float) {
  STMT_PTR stmt =
      Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_x(), 3.0);
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_relin_inline) {
  Config()._inline_relin = true;
  STMT_PTR stmt = Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Ckks_ir_gen().Append_output();
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_relin_func) {
  GTEST_SKIP() << "relin call is not supported yet";
  Config()._inline_relin = false;
  STMT_PTR stmt = Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_relin_func_preg) {
  GTEST_SKIP() << "relin call is not supported yet";
  Config()._inline_relin = false;
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  STMT_PTR stmt   = Ckks_ir_gen().Gen_relin(Container(), preg_z, Var_ciph3());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_rotate_inline) {
  Config()._inline_rotate = true;
  STMT_PTR stmt = Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Set_rot_idx_attr(stmt->Node()->Child(0), {5});
  Ckks_ir_gen().Append_output();
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_rotate_func) {
  GTEST_SKIP() << "rotate call is not supported yet";
  Config()._inline_rotate = false;
  STMT_PTR stmt = Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_rotate_func_preg) {
  GTEST_SKIP() << "rotate call is not supported yet";
  Config()._inline_rotate = false;
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  PREG_PTR preg_x = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  STMT_PTR stmt   = Ckks_ir_gen().Gen_rotate(Container(), preg_z, preg_x, 5);
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_rescale) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_rescale(Container(), Var_z(), Var_x());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_bts) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_bootstrap(Container(), Var_x());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_encode) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_encode(Container(), Var_p());
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_raise_mod) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_raise_mod(
      Container(), Ckks_ir_gen().Output_var(), Ckks_ir_gen().Input_var(), 0);
  Set_level_attr(stmt->Node()->Child(0)->Child(0), 1);
  Set_sf_degree(stmt->Node()->Child(0)->Child(0), 1);
  Lower();
}

TEST_P(TEST_CKKS2HPOLY, Handle_linear_transform_2x2_exact_graph) {
  Config()._prop_attr = true;
  constexpr uint32_t slots      = 4;
  constexpr uint32_t term_count = 4;
  constexpr uint32_t schema     = 1;
  constexpr uint32_t scale      = 1;
  constexpr uint32_t cache      = 1;
  const uint32_t     level      = this->GetParam().Input_level();
  // Keep the synthetic descriptor aligned with the runtime-backed prime
  // metadata installed when POLY types are materialized by the driver.
  Fhe_ctx().Get_ctx_param().Set_scaling_factor_bit_num(56);
  const uint32_t num_p = Fhe_ctx().Get_ctx_param().Get_p_prime_num();
  const int32_t  rot_in[]  = {0, 1};
  const int32_t  rot_out[] = {0, -2};

  std::vector<double> coefficients;
  coefficients.reserve(term_count * slots * 2);
  for (uint32_t term = 0; term < term_count; ++term) {
    for (uint32_t slot = 0; slot < slots; ++slot) {
      coefficients.push_back(term * 100.0 + slot + 0.25);
      coefficients.push_back(-(term * 100.0 + slot + 0.5));
    }
  }
  GLOB_SCOPE* glob = Container()->Glob_scope();
  TYPE_PTR f64 = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
  TYPE_PTR coefficient_type = glob->New_arr_type(
      "linear_transform_test_coefficients", f64,
      {static_cast<int64_t>(coefficients.size())}, Spos());
  CONSTANT_PTR coefficient = glob->New_const(
      CONSTANT_KIND::ARRAY, coefficient_type, coefficients.data(),
      coefficients.size() * sizeof(double));

  NODE_PTR transform = Container()->New_cust_node(
      fhe::ckks::OPC_LINEAR_TRANSFORM, Ciph_ty(), Spos());
  transform->Set_child(
      0, Container()->New_ld(Ckks_ir_gen().Input_var(), Spos()));
  transform->Set_child(1, Container()->New_ldc(coefficient, Spos()));
  transform->Set_attr(core::FHE_ATTR_KIND::LT_SCHEMA_VERSION, &schema, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_SLOTS, &slots, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_TERM_COUNT, &term_count, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_ROT_IN, rot_in, 2);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_ROT_OUT, rot_out, 2);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_SCALE_DEGREE, &scale, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_PLAIN_LEVEL, &level, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_NUM_P, &num_p, 1);
  transform->Set_attr(core::FHE_ATTR_KIND::LT_ENCODE_CACHE, &cache, 1);
  Set_level_attr(transform, level);
  Set_sf_degree(transform, 2);
  Container()->Stmt_list().Append(
      Container()->New_st(transform, Var_z(), Spos()));
  Ckks_ir_gen().Append_output();

  NODE_PTR lowered = Lower()->Entry_node();
  EXPECT_EQ(Count_opcode(lowered, fhe::ckks::OPC_LINEAR_TRANSFORM), 0U);
  EXPECT_EQ(Count_opcode(lowered, fhe::ckks::OPC_ENCODE), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_PRECOMP), 2U);
  // Two PRECOMP arrays plus all materialized Q/QP intermediates except the
  // final outer pair, which feeds the parent ciphertext store.
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_FREE), 20U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_DOT_PROD), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_EXTEND), 3U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_MUL_EXT), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_MAC_EXT), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_ADD_EXT), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_ROTATE), 4U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_MOD_DOWN), 3U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_SWK_C0), 2U);
  EXPECT_EQ(Count_opcode(lowered, fhe::poly::OPC_SWK_C1), 2U);
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTIONS_BEGIN),
            1U);
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTION_BEGIN),
            2U);
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTION_END),
            2U);
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTIONS_END),
            1U);
  EXPECT_EQ(Count_opcode(lowered, fhe::ckks::OPC_ROTATE_BATCH), 0U);

  std::vector<NODE_PTR> dot_products;
  Collect_opcode(lowered, fhe::poly::OPC_DOT_PROD, dot_products);
  ASSERT_EQ(dot_products.size(), 4U);
  const uint32_t expected_num_decomp =
      Fhe_ctx().Get_ctx_param().Get_num_decomp(level);
  ASSERT_GT(expected_num_decomp, 0U);
  for (NODE_PTR dot_product : dot_products) {
    ASSERT_EQ(dot_product->Child(2)->Opcode(), air::core::OPC_INTCONST);
    EXPECT_EQ(dot_product->Child(2)->Intconst(), expected_num_decomp);
  }

  std::vector<NODE_PTR> encodes;
  Collect_opcode(lowered, fhe::ckks::OPC_ENCODE, encodes);
  ASSERT_EQ(encodes.size(), term_count);
  std::sort(encodes.begin(), encodes.end(), [](NODE_PTR lhs, NODE_PTR rhs) {
    return lhs->Child(0)->Const()->Array_elem<double>(0) <
           rhs->Child(0)->Const()->Array_elem<double>(0);
  });
  for (uint32_t term = 0; term < term_count; ++term) {
    NODE_PTR encode = encodes[term];
    ASSERT_EQ(encode->Child(0)->Opcode(), air::core::OPC_LDC);
    CONSTANT_PTR row = encode->Child(0)->Const();
    ASSERT_NE(row, Null_ptr);
    EXPECT_EQ(row->Kind(), CONSTANT_KIND::ARRAY);
    EXPECT_EQ(row->Array_byte_len(), slots * 2 * sizeof(double));
    EXPECT_DOUBLE_EQ(row->Array_elem<double>(0), term * 100.0 + 0.25);
    EXPECT_DOUBLE_EQ(row->Array_elem<double>(1), -(term * 100.0 + 0.5));
    const uint32_t* complex =
        encode->Attr<uint32_t>(core::FHE_ATTR_KIND::ENCODE_DCMPLX);
    const uint32_t* encoded_scale =
        encode->Attr<uint32_t>(core::FHE_ATTR_KIND::SCALE);
    const uint32_t* encoded_level =
        encode->Attr<uint32_t>(core::FHE_ATTR_KIND::LEVEL);
    const uint32_t* encoded_num_p =
        encode->Attr<uint32_t>(core::FHE_ATTR_KIND::NUM_P);
    const uint32_t* encoded_cache =
        encode->Attr<uint32_t>(core::FHE_ATTR_KIND::ENCODE_CACHE);
    ASSERT_NE(complex, nullptr);
    ASSERT_NE(encoded_scale, nullptr);
    ASSERT_NE(encoded_level, nullptr);
    ASSERT_NE(encoded_num_p, nullptr);
    ASSERT_NE(encoded_cache, nullptr);
    EXPECT_EQ(*complex, 1U);
    EXPECT_EQ(*encoded_scale, scale);
    EXPECT_EQ(*encoded_level, level);
    EXPECT_EQ(*encoded_num_p, num_p);
    EXPECT_EQ(*encoded_cache, cache);
  }
}

TEST_P(TEST_CKKS2HPOLY, Parallel_sections_use_disjoint_spoly_scratch) {
  Config()._linear_transform_only = true;
  Config()._prop_attr             = true;

  STMT_LIST statements = Container()->Stmt_list();
  auto append_marker = [&](air::base::OPCODE opcode) {
    statements.Append(Container()->New_cust_stmt(opcode, Spos()));
  };
  append_marker(fhe::ckks::OPC_PARALLEL_SECTIONS_BEGIN);
  append_marker(fhe::ckks::OPC_PARALLEL_SECTION_BEGIN);
  Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_x(), 2.0);
  append_marker(fhe::ckks::OPC_PARALLEL_SECTION_END);
  append_marker(fhe::ckks::OPC_PARALLEL_SECTION_BEGIN);
  Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_y(), 3.0);
  append_marker(fhe::ckks::OPC_PARALLEL_SECTION_END);
  append_marker(fhe::ckks::OPC_PARALLEL_SECTIONS_END);
  Ckks_ir_gen().Append_output();

  NODE_PTR lowered = Lower()->Entry_node();
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTIONS_BEGIN),
            1U);
  EXPECT_EQ(Count_opcode(lowered,
                         fhe::poly::OPC_PARALLEL_SECTION_BEGIN),
            2U);

  std::set<uint32_t> section_symbols[2];
  int32_t            section      = -1;
  uint32_t           next_section = 0;
  ASSERT_GT(lowered->Num_child(), 0U);
  NODE_PTR lowered_body = lowered->Child(lowered->Num_child() - 1);
  ASSERT_TRUE(lowered_body->Is_block());
  STMT_LIST lowered_statements(lowered_body);
  for (STMT_PTR statement = lowered_statements.Begin_stmt();
       statement != lowered_statements.End_stmt();
       statement = statement->Next()) {
    NODE_PTR node = statement->Node();
    if (node->Opcode() == fhe::poly::OPC_PARALLEL_SECTION_BEGIN) {
      ASSERT_LT(next_section, 2U);
      section = static_cast<int32_t>(next_section++);
      continue;
    }
    if (node->Opcode() == fhe::poly::OPC_PARALLEL_SECTION_END) {
      section = -1;
      continue;
    }
    if (section >= 0) {
      Collect_pgen_symbols(node, section_symbols[section]);
    }
  }

  ASSERT_FALSE(section_symbols[0].empty());
  ASSERT_FALSE(section_symbols[1].empty());
  std::vector<uint32_t> shared_symbols;
  std::set_intersection(
      section_symbols[0].begin(), section_symbols[0].end(),
      section_symbols[1].begin(), section_symbols[1].end(),
      std::back_inserter(shared_symbols));
  EXPECT_TRUE(shared_symbols.empty());
}

TEST_P(TEST_CKKS2HPOLY, Handle_all) {
  Config()._inline_relin  = true;
  Config()._inline_rotate = true;
  Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_y());
  Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_p());
  Ckks_ir_gen().Gen_mul(Container(), Var_z(), Var_x(), Var_p());
  Ckks_ir_gen().Gen_mul(Container(), Var_ciph3(), Var_x(), Var_y());
  Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_x(), 3.0);
  STMT_PTR stmt = Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Set_level_attr(stmt->Node()->Child(0), this->GetParam().Input_level());
  stmt = Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Set_level_attr(stmt->Node()->Child(0), this->GetParam().Input_level());
  Ckks_ir_gen().Gen_rescale(Container(), Var_z(), Var_x());
  Ckks_ir_gen().Gen_bootstrap(Container(), Var_x());
  Ckks_ir_gen().Gen_encode(Container(), Var_p());
  Ckks_ir_gen().Gen_ret(Container(), Var_z());

  Lower();
}

// testing same tests with different config
INSTANTIATE_TEST_SUITE_P(CKKS2POLY_TEST, TEST_CKKS2HPOLY,
                         ::testing::Values(TEST_CONFIG(".hpoly.t", HPOLY)));

}  // namespace test
}  // namespace poly
}  // namespace fhe
