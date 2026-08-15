//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/test/lower_ckks.h"
#include "poly_ir_gen.h"

using namespace air::base;
using namespace fhe::core;
using namespace fhe::poly;

namespace fhe {
namespace poly {
namespace test {
class TEST_H2LPOLY : public TEST_LOWER_CKKS<TEST_CONFIG> {
public:
  std::string Lower_extended_add_mul(bool inline_rns) {
    Config()._inline_rns = inline_rns;
    Config()._prop_attr  = true;

    POLY_MEM_POOL pool;
    pool.Push();
    POLY_IR_GEN irgen(Container(), &Fhe_ctx(), &pool);
    FUNC_SCOPE* func_scope = Container()->Parent_func_scope();
    NODE_PAIR x = irgen.New_ciph_poly_load(
        VAR(func_scope, Ckks_ir_gen().Input_var()), false, Spos());
    NODE_PAIR y = irgen.New_ciph_poly_load(
        VAR(func_scope, Ckks_ir_gen().Input_var()), false, Spos());
    TYPE_PTR rns_poly_type =
        Fhe_ctx().Get_rns_poly_type(Container()->Glob_scope());
    ADDR_DATUM_PTR add_result =
        func_scope->New_var(rns_poly_type, "ext_add_result", Spos());
    ADDR_DATUM_PTR sub_result =
        func_scope->New_var(rns_poly_type, "ext_sub_result", Spos());
    ADDR_DATUM_PTR mul_result =
        func_scope->New_var(rns_poly_type, "ext_mul_result", Spos());

    NODE_PTR add = irgen.New_poly_add_ext(irgen.New_extend(x.first, Spos()),
                                          irgen.New_extend(y.first, Spos()),
                                          Spos());
    NODE_PTR sub = irgen.New_poly_sub_ext(
        irgen.New_extend(Container()->Clone_node_tree(x.first), Spos()),
        irgen.New_extend(Container()->Clone_node_tree(y.first), Spos()),
        Spos());
    NODE_PTR mul = irgen.New_poly_mul_ext(irgen.New_extend(x.second, Spos()),
                                          irgen.New_extend(y.second, Spos()),
                                          Spos());
    Container()->Stmt_list().Append(
        Container()->New_st(add, add_result, Spos()));
    Container()->Stmt_list().Append(
        Container()->New_st(sub, sub_result, Spos()));
    Container()->Stmt_list().Append(
        Container()->New_st(mul, mul_result, Spos()));
    pool.Pop();

    air::base::CONTAINER* lowered = Lower();
    std::ostringstream    rendered;
    lowered->Glob_scope()->Print_ir(rendered);
    return rendered.str();
  }

  std::string Lower_extended_rotate(bool inline_rns) {
    Config()._inline_rns = inline_rns;
    Config()._prop_attr  = true;

    POLY_MEM_POOL pool;
    pool.Push();
    POLY_IR_GEN irgen(Container(), &Fhe_ctx(), &pool);
    FUNC_SCOPE* func_scope = Container()->Parent_func_scope();
    NODE_PAIR input = irgen.New_ciph_poly_load(
        VAR(func_scope, Ckks_ir_gen().Input_var()), false, Spos());
    TYPE_PTR rns_poly_type =
        Fhe_ctx().Get_rns_poly_type(Container()->Glob_scope());
    ADDR_DATUM_PTR result =
        func_scope->New_var(rns_poly_type, "ext_rotate_result", Spos());
    TYPE_PTR i32_type =
        Container()->Glob_scope()->Prim_type(PRIMITIVE_TYPE::INT_S32);
    NODE_PTR rotation = Container()->New_intconst(i32_type, 5, Spos());
    NODE_PTR rotate = irgen.New_poly_rotate(
        irgen.New_extend(input.first, Spos()), rotation, Spos());
    Container()->Stmt_list().Append(
        Container()->New_st(rotate, result, Spos()));
    pool.Pop();

    air::base::CONTAINER* lowered = Lower();
    std::ostringstream    rendered;
    lowered->Glob_scope()->Print_ir(rendered);
    return rendered.str();
  }

  std::string Lower_extended_runtime_boundary() {
    Config()._linear_transform_only = true;
    Config()._prop_attr             = true;

    POLY_MEM_POOL pool;
    pool.Push();
    POLY_IR_GEN irgen(Container(), &Fhe_ctx(), &pool);
    FUNC_SCOPE* func_scope = Container()->Parent_func_scope();
    NODE_PAIR input = irgen.New_ciph_poly_load(
        VAR(func_scope, Ckks_ir_gen().Input_var()), false, Spos());
    TYPE_PTR rns_poly_type =
        Fhe_ctx().Get_rns_poly_type(Container()->Glob_scope());
    ADDR_DATUM_PTR add_result =
        func_scope->New_var(rns_poly_type, "runtime_ext_add", Spos());
    ADDR_DATUM_PTR rotate_result =
        func_scope->New_var(rns_poly_type, "runtime_ext_rotate", Spos());
    NODE_PTR add = irgen.New_poly_add_ext(
        irgen.New_extend(input.first, Spos()),
        irgen.New_extend(Container()->Clone_node_tree(input.first), Spos()),
        Spos());
    TYPE_PTR i32_type =
        Container()->Glob_scope()->Prim_type(PRIMITIVE_TYPE::INT_S32);
    NODE_PTR rotate = irgen.New_poly_rotate(
        Container()->Clone_node_tree(add),
        Container()->New_intconst(i32_type, 5, Spos()), Spos());
    Container()->Stmt_list().Append(
        Container()->New_st(add, add_result, Spos()));
    Container()->Stmt_list().Append(
        Container()->New_st(rotate, rotate_result, Spos()));
    pool.Pop();

    air::base::CONTAINER* lowered = Lower();
    std::ostringstream    rendered;
    lowered->Glob_scope()->Print_ir(rendered);
    return rendered.str();
  }
};

TEST_P(TEST_H2LPOLY, Handle_add_ciph) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_add_plain) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_p());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_mul_plain) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_mul(Container(), Var_z(), Var_x(), Var_p());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_mul_ciph) {
  STMT_PTR stmt =
      Ckks_ir_gen().Gen_mul(Container(), Var_ciph3(), Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_mul_ciph_preg) {
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph3_ty());
  STMT_PTR stmt = Ckks_ir_gen().Gen_mul(Container(), preg_z, Var_x(), Var_y());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_mul_float) {
  STMT_PTR stmt =
      Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_x(), 3.0);
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_extended_add_mul_inline) {
  const std::string ir = Lower_extended_add_mul(true);

  EXPECT_EQ(ir.find("POLY.add_ext"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.sub_ext"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.mul_ext"), std::string::npos);
  EXPECT_NE(ir.find("POLY.hw_modadd"), std::string::npos);
  EXPECT_NE(ir.find("POLY.hw_modsub"), std::string::npos);
  EXPECT_NE(ir.find("POLY.hw_modmul"), std::string::npos);
  EXPECT_NE(ir.find("POLY.extend"), std::string::npos);
  EXPECT_NE(ir.find("POLY.p_modulus"), std::string::npos);
}

TEST_P(TEST_H2LPOLY, Handle_extended_add_mul_function) {
  const std::string ir = Lower_extended_add_mul(false);

  EXPECT_EQ(ir.find("POLY.add_ext"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.sub_ext"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.mul_ext"), std::string::npos);
  EXPECT_NE(ir.find("Rns_add_ext"), std::string::npos);
  EXPECT_NE(ir.find("Rns_sub_ext"), std::string::npos);
  EXPECT_NE(ir.find("Rns_mul_ext"), std::string::npos);
  EXPECT_NE(ir.find("POLY.extend"), std::string::npos);
  EXPECT_NE(ir.find("POLY.p_modulus"), std::string::npos);
}

TEST_P(TEST_H2LPOLY, Handle_extended_rotate_inline) {
  const std::string ir = Lower_extended_rotate(true);

  EXPECT_EQ(ir.find("POLY.rotate"), std::string::npos);
  EXPECT_NE(ir.find("POLY.hw_rotate"), std::string::npos);
  EXPECT_NE(ir.find("POLY.q_modulus"), std::string::npos);
  EXPECT_NE(ir.find("POLY.p_modulus"), std::string::npos);
}

TEST_P(TEST_H2LPOLY, Handle_extended_rotate_function) {
  const std::string ir = Lower_extended_rotate(false);

  EXPECT_EQ(ir.find("POLY.rotate"), std::string::npos);
  EXPECT_NE(ir.find("Rns_rotate_ext"), std::string::npos);
  EXPECT_NE(ir.find("POLY.p_modulus"), std::string::npos);
}

TEST_P(TEST_H2LPOLY, Handle_extended_runtime_boundary) {
  const std::string ir = Lower_extended_runtime_boundary();

  EXPECT_NE(ir.find("POLY.add_ext"), std::string::npos);
  EXPECT_NE(ir.find("POLY.rotate"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.hw_modadd"), std::string::npos);
  EXPECT_EQ(ir.find("POLY.hw_rotate"), std::string::npos);
}

TEST_P(TEST_H2LPOLY, Handle_relin_inline) {
  Config()._inline_relin = true;
  STMT_PTR stmt = Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_relin_func) {
  GTEST_SKIP() << "relin call is not supported yet";
  Config()._inline_relin = false;
  STMT_PTR stmt = Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_relin_func_preg) {
  GTEST_SKIP() << "relin call is not supported yet";
  Config()._inline_relin = false;
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  STMT_PTR stmt   = Ckks_ir_gen().Gen_relin(Container(), preg_z, Var_ciph3());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_rotate_inline) {
  Config()._inline_rotate = true;
  STMT_PTR stmt = Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_rotate_func) {
  GTEST_SKIP() << "rotate call is not supported yet";
  Config()._inline_rotate = false;
  STMT_PTR stmt = Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_rotate_func_preg) {
  GTEST_SKIP() << "rotate call is not supported yet";
  Config()._inline_rotate = false;
  PREG_PTR preg_z = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  PREG_PTR preg_x = Container()->Parent_func_scope()->New_preg(Ciph_ty());
  STMT_PTR stmt   = Ckks_ir_gen().Gen_rotate(Container(), preg_z, preg_x, 5);
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_rescale) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_rescale(Container(), Var_z(), Var_x());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_bts) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_bootstrap(Container(), Var_x());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_encode) {
  STMT_PTR stmt = Ckks_ir_gen().Gen_encode(Container(), Var_p());
  Lower();
}

TEST_P(TEST_H2LPOLY, Handle_all) {
  Config()._inline_relin  = true;
  Config()._inline_rotate = true;
  Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_y());
  Ckks_ir_gen().Gen_add(Container(), Var_z(), Var_x(), Var_p());
  Ckks_ir_gen().Gen_mul(Container(), Var_z(), Var_x(), Var_p());
  Ckks_ir_gen().Gen_mul(Container(), Var_ciph3(), Var_x(), Var_y());
  Ckks_ir_gen().Gen_mul_float(Container(), Var_z(), Var_x(), 3.0);
  Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Ckks_ir_gen().Gen_relin(Container(), Var_z(), Var_ciph3());
  Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Ckks_ir_gen().Gen_rotate(Container(), Var_z(), Var_x(), 5);
  Ckks_ir_gen().Gen_rescale(Container(), Var_z(), Var_x());
  Ckks_ir_gen().Gen_bootstrap(Container(), Var_x());
  Ckks_ir_gen().Gen_encode(Container(), Var_p());
  Ckks_ir_gen().Gen_ret(Container(), Var_z());

  Lower();
}

// testing same tests with different config
INSTANTIATE_TEST_SUITE_P(H2LPOLY_TEST, TEST_H2LPOLY,
                         ::testing::Values(TEST_CONFIG(".lpoly.t", LPOLY)));

}  // namespace test
}  // namespace poly
}  // namespace fhe
