//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "nn/core/attr.h"
#include "nn/core/opcode.h"
#include "nn/vector/config.h"
#include "nn/vector/skip_lowering.h"
#include "nn/vector/tensor2vector_dsl.h"
#include "nn/vector/tensor2vector_plan.h"
#include "nn/vector/vector_gen.h"
#include "nn/vector/vector_opcode.h"
#include "nn/vector/vector_utils.h"
#include "tensor2vector_air_normalizer.h"

using namespace air::base;
using namespace nn::vector;
using namespace nn::vector::test;

namespace {

constexpr int64_t SYNTHETIC_WIDTH = 4;
constexpr int64_t M3_WIDE_WIDTH   = 8;
constexpr int64_t M3_ROW_COUNT    = 3;

struct SYNTHETIC_BODY_RESULT {
  NODE_PTR              _result;
  std::vector<NODE_PTR> _semantic_inputs;
};

SYNTHETIC_BODY_RESULT Emit_synthetic_kernel_body(
    FUNC_SCOPE& scope, NODE_PTR body, const SPOS& spos,
    float constant_scale = 1.0F, bool reverse_loop_add = false) {
  CONTAINER& cntr = scope.Container();
  GLOB_SCOPE& glob = scope.Glob_scope();
  TYPE_PTR    type = scope.Formal(0)->Type();
  AIR_ASSERT(type->Is_array());
  AIR_ASSERT(type->Is_compatible_type(scope.Formal(1)->Type()));

  TYPE_PTR           element_type = type->Cast_to_arr()->Elem_type();
  std::vector<float> values(SYNTHETIC_WIDTH);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    values[idx] = constant_scale * static_cast<float>(idx + 1) / 8.0F;
  }
  CONSTANT_PTR constant = New_array_const(
      &glob, "synthetic_weight", values.size(), element_type,
      {SYNTHETIC_WIDTH}, values.data(), spos);

  VECTOR_GEN vector_gen(&cntr);
  NODE_PTR   input0 = cntr.New_ld(scope.Formal(0), spos);
  NODE_PTR   input1 = cntr.New_ld(scope.Formal(1), spos);
  NODE_PTR initial =
      vector_gen.New_add(input0, cntr.New_ldc(constant, spos), spos);
  ADDR_DATUM_PTR accumulator =
      scope.New_var(type, "synthetic_accumulator", spos);
  STMT_LIST(body).Append(cntr.New_st(initial, accumulator, spos));

  TYPE_PTR       s32 = glob.Prim_type(PRIMITIVE_TYPE::INT_S32);
  ADDR_DATUM_PTR iv  = scope.New_var(s32, "synthetic_iv", spos);
  NODE_PTR       iv_load = cntr.New_ld(iv, spos);
  NODE_PTR       init     = cntr.New_intconst(s32, 0, spos);
  NODE_PTR upper = cntr.New_intconst(s32, 2, spos);
  NODE_PTR condition =
      cntr.New_bin_arith(air::core::OPC_LT, s32, iv_load, upper, spos);
  NODE_PTR increment = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, cntr.New_ld(iv, spos),
      cntr.New_intconst(s32, 1, spos), spos);
  NODE_PTR loop_body = cntr.New_stmt_block(spos);
  STMT_PTR loop =
      cntr.New_do_loop(iv, init, condition, increment, loop_body, spos);

  NODE_PTR lhs = cntr.New_ld(accumulator, spos);
  NODE_PTR rhs = input1;
  NODE_PTR update = reverse_loop_add
                        ? vector_gen.New_add(rhs, lhs, spos)
                        : vector_gen.New_add(lhs, rhs, spos);
  STMT_LIST(loop_body).Append(cntr.New_st(update, accumulator, spos));
  STMT_LIST(body).Append(loop);

  return SYNTHETIC_BODY_RESULT{
      cntr.New_ld(accumulator, spos), {input0, input1}};
}

NODE_PTR Emit_typed_nested_loop_kernel_body(FUNC_SCOPE& scope, NODE_PTR body,
                                            const SPOS& spos) {
  CONTAINER& cntr = scope.Container();
  GLOB_SCOPE& glob = scope.Glob_scope();
  TYPE_PTR    vector_type = scope.Formal(0)->Type();
  TYPE_PTR    s32 = glob.Prim_type(PRIMITIVE_TYPE::INT_S32);
  AIR_ASSERT(vector_type->Is_array());
  AIR_ASSERT(vector_type->Is_compatible_type(scope.Formal(1)->Type()));

  ADDR_DATUM_PTR accumulator =
      scope.New_var(vector_type, "typed_vector_accumulator", spos);
  STMT_LIST(body).Append(
      cntr.New_st(cntr.New_zero(vector_type, spos), accumulator, spos));

  ADDR_DATUM_PTR linear_index =
      scope.New_var(s32, "typed_linear_index", spos);
  ADDR_DATUM_PTR shift_amount =
      scope.New_var(s32, "typed_shift_amount", spos);

  ADDR_DATUM_PTR cin = scope.New_var(s32, "typed_cin", spos);
  NODE_PTR outer_body = cntr.New_stmt_block(spos);
  NODE_PTR outer_condition = cntr.New_bin_arith(
      air::core::OPC_LT, s32, cntr.New_ld(cin, spos),
      cntr.New_intconst(s32, 2, spos), spos);
  NODE_PTR outer_increment = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, cntr.New_ld(cin, spos),
      cntr.New_intconst(s32, 1, spos), spos);

  ADDR_DATUM_PTR khw = scope.New_var(s32, "typed_khw", spos);
  NODE_PTR inner_body = cntr.New_stmt_block(spos);
  NODE_PTR inner_condition = cntr.New_bin_arith(
      air::core::OPC_LT, s32, cntr.New_ld(khw, spos),
      cntr.New_intconst(s32, 3, spos), spos);
  NODE_PTR inner_increment = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, cntr.New_ld(khw, spos),
      cntr.New_intconst(s32, 1, spos), spos);

  NODE_PTR scaled_cin = cntr.New_bin_arith(
      air::core::OPC_MUL, s32, cntr.New_ld(cin, spos),
      cntr.New_intconst(s32, 3, spos), spos);
  NODE_PTR affine_index = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, scaled_cin, cntr.New_ld(khw, spos), spos);
  STMT_LIST(inner_body).Append(
      cntr.New_st(affine_index, linear_index, spos));

  NODE_PTR shifted_one = cntr.New_bin_arith(
      air::core::OPC_SHL, s32, cntr.New_intconst(s32, 1, spos),
      cntr.New_ld(khw, spos), spos);
  STMT_LIST(inner_body).Append(
      cntr.New_st(shifted_one, shift_amount, spos));

  VECTOR_GEN vector_gen(&cntr);
  NODE_PTR operands = vector_gen.New_add(
      cntr.New_ld(scope.Formal(0), spos),
      cntr.New_ld(scope.Formal(1), spos), spos);
  NODE_PTR update = vector_gen.New_add(
      cntr.New_ld(accumulator, spos), operands, spos);
  STMT_LIST(inner_body).Append(cntr.New_st(update, accumulator, spos));

  STMT_LIST(outer_body).Append(cntr.New_do_loop(
      khw, cntr.New_intconst(s32, 0, spos), inner_condition,
      inner_increment, inner_body, spos));
  STMT_LIST(body).Append(cntr.New_do_loop(
      cin, cntr.New_intconst(s32, 0, spos), outer_condition,
      outer_increment, outer_body, spos));
  return cntr.New_ld(accumulator, spos);
}

NODE_PTR Emit_vector_primitive_kernel_body(FUNC_SCOPE& scope, NODE_PTR body,
                                           const SPOS& spos) {
  CONTAINER& cntr = scope.Container();
  GLOB_SCOPE& glob = scope.Glob_scope();
  AIR_ASSERT(scope.Formal_cnt() == 2U);
  TYPE_PTR narrow_type = scope.Formal(0)->Type();
  TYPE_PTR wide_type   = scope.Formal(1)->Type();
  AIR_ASSERT(narrow_type->Is_array() && wide_type->Is_array());
  AIR_ASSERT(narrow_type->Cast_to_arr()->Elem_count() == SYNTHETIC_WIDTH);
  AIR_ASSERT(wide_type->Cast_to_arr()->Elem_count() == M3_WIDE_WIDTH);
  AIR_ASSERT(narrow_type->Cast_to_arr()->Elem_type()->Is_compatible_type(
      wide_type->Cast_to_arr()->Elem_type()));

  ADDR_DATUM_PTR accumulator =
      scope.New_var(wide_type, "m3_vector_accumulator", spos);
  STMT_LIST(body).Append(
      cntr.New_st(cntr.New_zero(wide_type, spos), accumulator, spos));

  TYPE_PTR           element_type = wide_type->Cast_to_arr()->Elem_type();
  std::vector<float> values(M3_ROW_COUNT * M3_WIDE_WIDTH);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    values[idx] = static_cast<float>(idx + 1) / 32.0F;
  }
  CONSTANT_PTR rows = New_array_const(
      &glob, "m3_slice_rows", values.size(), element_type,
      {M3_ROW_COUNT, M3_WIDE_WIDTH}, values.data(), spos);

  TYPE_PTR       s32 = glob.Prim_type(PRIMITIVE_TYPE::INT_S32);
  ADDR_DATUM_PTR iv  = scope.New_var(s32, "m3_vector_iv", spos);
  NODE_PTR loop_body = cntr.New_stmt_block(spos);
  NODE_PTR condition = cntr.New_bin_arith(
      air::core::OPC_LT, s32, cntr.New_ld(iv, spos),
      cntr.New_intconst(s32, M3_ROW_COUNT, spos), spos);
  NODE_PTR increment = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, cntr.New_ld(iv, spos),
      cntr.New_intconst(s32, 1, spos), spos);

  NODE_PTR shift = cntr.New_bin_arith(
      air::core::OPC_ADD, s32, cntr.New_ld(iv, spos),
      cntr.New_intconst(s32, -1, spos), spos);
  VECTOR_GEN vector_gen(&cntr);
  NODE_PTR roll = vector_gen.New_roll(
      cntr.New_ld(scope.Formal(1), spos), shift, {-1, 0, 1}, spos);
  NODE_PTR slice = vector_gen.New_slice(
      cntr.New_ldc(rows, spos), cntr.New_ld(iv, spos),
      cntr.New_intconst(s32, M3_WIDE_WIDTH, spos), spos);
  NODE_PTR widened = vector_gen.New_add(
      cntr.New_ld(scope.Formal(0), spos), roll, spos);
  NODE_PTR product = vector_gen.New_mul(widened, slice, spos);
  NODE_PTR update = vector_gen.New_add(
      cntr.New_ld(accumulator, spos), product, spos);
  STMT_LIST(loop_body).Append(cntr.New_st(update, accumulator, spos));

  STMT_LIST(body).Append(cntr.New_do_loop(
      iv, cntr.New_intconst(s32, 0, spos), condition, increment, loop_body,
      spos));

  NODE_PTR result = cntr.New_ld(accumulator, spos);
  uint32_t slot   = M3_WIDE_WIDTH;
  result->Set_attr(nn::core::ATTR::SLOT, &slot, 1);
  return result;
}

struct SOURCE_ADD_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_SCOPE*                 _scope;
};

SOURCE_ADD_IR Build_source_add(const char* function_name, uint32_t line) {
  SOURCE_ADD_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE& glob = *fixture._glob;
  const SPOS  spos(0, line, 1, 0);

  TYPE_PTR f32 = glob.Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  TYPE_PTR array_type = New_array_type(
      &glob, std::string(function_name) + "_type", f32, {SYNTHETIC_WIDTH},
      spos);
  STR_PTR  name = glob.New_str(function_name);
  FUNC_PTR func = glob.New_func(name, spos);
  func->Set_parent(glob.Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(array_type, signature);
  glob.New_param("lhs", array_type, signature, spos);
  glob.New_param("rhs", array_type, signature, spos);
  signature->Set_complete();
  glob.New_entry_point(signature, func, name, spos);

  fixture._scope = &glob.New_func_scope(func);
  CONTAINER& cntr = fixture._scope->Container();
  cntr.New_func_entry(spos);
  NODE_PTR add = cntr.New_bin_arith(
      OPCODE(nn::core::NN, nn::core::OPCODE::ADD), array_type,
      cntr.New_ld(fixture._scope->Formal(0), spos),
      cntr.New_ld(fixture._scope->Formal(1), spos), spos);
  cntr.Stmt_list().Append(cntr.New_retv(add, spos));
  return fixture;
}

SOURCE_ADD_IR Build_heterogeneous_source_add(const char* function_name,
                                             uint32_t line) {
  SOURCE_ADD_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE& glob = *fixture._glob;
  const SPOS  spos(0, line, 1, 0);

  TYPE_PTR f32 = glob.Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  TYPE_PTR narrow_type = New_array_type(
      &glob, std::string(function_name) + "_narrow", f32,
      {SYNTHETIC_WIDTH}, spos);
  TYPE_PTR wide_type = New_array_type(
      &glob, std::string(function_name) + "_wide", f32, {M3_WIDE_WIDTH},
      spos);
  STR_PTR  name = glob.New_str(function_name);
  FUNC_PTR func = glob.New_func(name, spos);
  func->Set_parent(glob.Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(wide_type, signature);
  glob.New_param("narrow", narrow_type, signature, spos);
  glob.New_param("wide", wide_type, signature, spos);
  signature->Set_complete();
  glob.New_entry_point(signature, func, name, spos);

  fixture._scope = &glob.New_func_scope(func);
  CONTAINER& cntr = fixture._scope->Container();
  cntr.New_func_entry(spos);
  NODE_PTR add = cntr.New_bin_arith(
      OPCODE(nn::core::NN, nn::core::OPCODE::ADD), wide_type,
      cntr.New_ld(fixture._scope->Formal(0), spos),
      cntr.New_ld(fixture._scope->Formal(1), spos), spos);
  cntr.Stmt_list().Append(cntr.New_retv(add, spos));
  return fixture;
}

struct REFERENCE_KERNEL_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_SCOPE*                 _scope;
  std::vector<NODE_PTR>       _inputs;
  NODE_PTR                    _result;
};

REFERENCE_KERNEL_IR Build_reference_kernel(const char* function_name,
                                           uint32_t line,
                                           float constant_scale = 1.0F,
                                           bool reverse_loop_add = false) {
  REFERENCE_KERNEL_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE& glob = *fixture._glob;
  const SPOS  spos(0, line, 1, 0);

  TYPE_PTR f32 = glob.Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  TYPE_PTR type = New_array_type(
      &glob, std::string(function_name) + "_type", f32, {SYNTHETIC_WIDTH},
      spos);
  STR_PTR  name = glob.New_str(function_name);
  FUNC_PTR func = glob.New_func(name, spos);
  func->Set_parent(glob.Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(type, signature);
  glob.New_param("native_a", type, signature, spos);
  glob.New_param("native_b", type, signature, spos);
  signature->Set_complete();
  glob.New_entry_point(signature, func, name, spos);

  fixture._scope = &glob.New_func_scope(func);
  CONTAINER& cntr = fixture._scope->Container();
  STMT_PTR   entry = cntr.New_func_entry(spos);
  NODE_PTR   body  = entry->Node()->Last_child();
  SYNTHETIC_BODY_RESULT emitted = Emit_synthetic_kernel_body(
      *fixture._scope, body, spos, constant_scale, reverse_loop_add);
  fixture._inputs = emitted._semantic_inputs;
  fixture._result = emitted._result;
  STMT_LIST(body).Append(cntr.New_retv(emitted._result, spos));
  return fixture;
}

VECTOR_KERNEL_NATIVE_AIR_VIEW Native_view(REFERENCE_KERNEL_IR& fixture) {
  NODE_PTR body = fixture._scope->Container().Entry_node()->Last_child();
  STMT_PTR terminal = Null_ptr;
  for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
       stmt          = stmt->Next()) {
    terminal = stmt;
  }
  AIR_ASSERT(terminal != Null_ptr &&
             terminal->Node()->Opcode() == air::core::OPC_RETV);
  return VECTOR_KERNEL_NATIVE_AIR_VIEW{
      fixture._scope, fixture._inputs, body->Begin_stmt(), terminal,
      fixture._result};
}

struct AIR_STATS {
  uint32_t                      _calls         = 0;
  uint32_t                      _nn_adds       = 0;
  uint32_t                      _vector_adds   = 0;
  uint32_t                      _vector_muls   = 0;
  uint32_t                      _vector_rolls  = 0;
  uint32_t                      _vector_slices = 0;
  uint32_t                      _loops         = 0;
  uint32_t                      _retvs         = 0;
  uint32_t                      _ldcs          = 0;
  uint32_t                      _core_adds     = 0;
  uint32_t                      _core_muls     = 0;
  uint32_t                      _core_lts      = 0;
  uint32_t                      _core_shls     = 0;
  uint32_t                      _zeros         = 0;
  std::vector<STMT_PTR>         _call_stmts;
  std::vector<NODE_PTR>         _vector_add_nodes;
  std::vector<NODE_PTR>         _vector_mul_nodes;
  std::vector<NODE_PTR>         _vector_roll_nodes;
  std::vector<NODE_PTR>         _vector_slice_nodes;
  std::vector<std::vector<int>> _rnums;
  std::vector<uint32_t>         _slots;
};

void Collect_stats(NODE_PTR node, AIR_STATS& stats) {
  if (node->Is_call()) {
    ++stats._calls;
    stats._call_stmts.push_back(node->Stmt());
  }
  if (node->Opcode() == OPCODE(nn::core::NN, nn::core::OPCODE::ADD)) {
    ++stats._nn_adds;
  }
  if (node->Opcode() == OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::ADD)) {
    ++stats._vector_adds;
    stats._vector_add_nodes.push_back(node);
  }
  if (node->Opcode() == OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::MUL)) {
    ++stats._vector_muls;
    stats._vector_mul_nodes.push_back(node);
  }
  if (node->Opcode() == OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::ROLL)) {
    ++stats._vector_rolls;
    stats._vector_roll_nodes.push_back(node);
  }
  if (node->Opcode() == OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::SLICE)) {
    ++stats._vector_slices;
    stats._vector_slice_nodes.push_back(node);
  }
  if (META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) {
    uint32_t count = 0;
    const int* rnum = node->Attr<int>(nn::core::ATTR::RNUM, &count);
    if (rnum != nullptr) {
      stats._rnums.emplace_back(rnum, rnum + count);
    }
    count = 0;
    const uint32_t* slot =
        node->Attr<uint32_t>(nn::core::ATTR::SLOT, &count);
    if (slot != nullptr) {
      stats._slots.insert(stats._slots.end(), slot, slot + count);
    }
  }
  if (node->Is_do_loop()) ++stats._loops;
  if (node->Opcode() == air::core::OPC_RETV) ++stats._retvs;
  if (node->Opcode() == air::core::OPC_LDC) ++stats._ldcs;
  if (node->Opcode() == air::core::OPC_ADD) ++stats._core_adds;
  if (node->Opcode() == air::core::OPC_MUL) ++stats._core_muls;
  if (node->Opcode() == air::core::OPC_LT) ++stats._core_lts;
  if (node->Opcode() == air::core::OPC_SHL) ++stats._core_shls;
  if (node->Opcode() == air::core::OPC_ZERO) ++stats._zeros;

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

AIR_STATS Get_stats(const FUNC_SCOPE& scope) {
  AIR_STATS stats;
  Collect_stats(scope.Container().Entry_node(), stats);
  return stats;
}

FUNC_SCOPE* Find_helper(GLOB_SCOPE& glob, FUNC_ID caller_id) {
  FUNC_SCOPE* helper = nullptr;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    if ((*iter).Id() != caller_id) {
      AIR_ASSERT(helper == nullptr);
      helper = &(*iter);
    }
  }
  return helper;
}

uint32_t Function_count(GLOB_SCOPE& glob) {
  uint32_t count = 0;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    ++count;
  }
  return count;
}

void Expect_helper_ownership(NODE_PTR node, FUNC_SCOPE& helper) {
  EXPECT_EQ(node->Container(), &helper.Container());
  if (node->Has_rtype()) {
    EXPECT_EQ(&node->Rtype()->Glob_scope(), &helper.Glob_scope());
  }
  if (node->Has_sym()) {
    EXPECT_EQ(node->Addr_datum()->Defining_func_scope(), &helper);
  }
  if (node->Has_preg()) {
    EXPECT_EQ(node->Preg()->Defining_func_scope(), &helper);
  }
  if (node->Has_ret_var()) {
    EXPECT_EQ(node->Ret_preg()->Defining_func_scope(), &helper);
  }
  if (node->Is_do_loop()) {
    EXPECT_EQ(node->Iv()->Defining_func_scope(), &helper);
  }
  if (node->Has_const_id()) {
    EXPECT_EQ(&node->Const()->Glob_scope(), &helper.Glob_scope());
  }

  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Expect_helper_ownership(stmt->Node(), helper);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Expect_helper_ownership(node->Child(idx), helper);
    }
  }
}

void Expect_typed_nested_loop_air(NODE_PTR node, FUNC_SCOPE& helper,
                                  TYPE_PTR vector_type, TYPE_PTR s32) {
  if (node->Is_do_loop()) {
    EXPECT_TRUE(node->Iv()->Type()->Is_compatible_type(s32));
  }
  const OPCODE opcode = node->Opcode();
  if (opcode == air::core::OPC_ADD || opcode == air::core::OPC_MUL ||
      opcode == air::core::OPC_LT || opcode == air::core::OPC_SHL) {
    EXPECT_TRUE(node->Rtype()->Is_compatible_type(s32));
  }
  if (opcode == air::core::OPC_ZERO) {
    EXPECT_TRUE(node->Rtype()->Is_compatible_type(vector_type));
  }
  if (node->Is_st() && node->Has_sym()) {
    EXPECT_TRUE(node->Addr_datum()->Type()->Is_compatible_type(
        node->Child(0)->Rtype()));
  }

  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Expect_typed_nested_loop_air(stmt->Node(), helper, vector_type, s32);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Expect_typed_nested_loop_air(node->Child(idx), helper, vector_type, s32);
    }
  }
}

class Tensor2VectorDslMaterialization : public ::testing::Test {
protected:
  void SetUp() override {
    _had_core   = META_INFO::Valid_domain(air::core::CORE);
    _had_nn     = META_INFO::Valid_domain(nn::core::NN);
    _had_vector = META_INFO::Valid_domain(VECTOR_DOMAIN::ID);
    if (!_had_core) ASSERT_TRUE(air::core::Register_core());
    if (!_had_nn) ASSERT_TRUE(nn::core::Register_nn());
    if (!_had_vector) ASSERT_TRUE(Register_vector_domain());

    const std::set<std::string>& skip_ops =
        SKIP_LOWERING_REGISTRY::Instance().Get_skip_ops();
    _saved_skip_ops.assign(skip_ops.begin(), skip_ops.end());
    Clear_skip_lowering_ops();
  }

  void TearDown() override {
    Clear_skip_lowering_ops();
    Set_skip_lowering_ops(_saved_skip_ops);

    META_INFO::Remove_all();
    if (_had_core) EXPECT_TRUE(air::core::Register_core());
    if (_had_nn) EXPECT_TRUE(nn::core::Register_nn());
    if (_had_vector) EXPECT_TRUE(Register_vector_domain());
  }

private:
  bool                     _had_core   = false;
  bool                     _had_nn     = false;
  bool                     _had_vector = false;
  std::vector<std::string> _saved_skip_ops;
};

}  // namespace

TEST_F(Tensor2VectorDslMaterialization,
       MaterializesDestinationHelperAndTypedCallBridge) {
  SOURCE_ADD_IR source = Build_source_add("m1_source_add", 13);
  ASSERT_TRUE(source._glob->Verify_ir());

  const std::string specialization_key = "vector-kernel-m1-synthetic:v1";
  const std::string helper_name =
      "__ace_vkernel_m1_synthetic_" +
      Vector_kernel_sha256(specialization_key);
  uint32_t callback_count = 0;

  VECTOR_KERNEL_LOWERING_REGISTRY registry;
  ASSERT_TRUE(registry.Register(
      OPCODE(nn::core::NN, nn::core::OPCODE::ADD),
      [&](NODE_PTR source_node, const std::vector<NODE_PTR>& actuals,
          GLOB_SCOPE& destination) {
        ++callback_count;
        EXPECT_EQ(source_node->Opcode(),
                  OPCODE(nn::core::NN, nn::core::OPCODE::ADD));
        EXPECT_EQ(actuals.size(), 2U);
        EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);
        EXPECT_EQ(actuals[1]->Container()->Glob_scope(), &destination);

        VECTOR_KERNEL_HELPER_SPEC spec;
        spec._specialization_key = specialization_key;
        spec._helper_name        = helper_name;
        spec._formal_types       = {actuals[0]->Rtype(), actuals[1]->Rtype()};
        spec._result_type        = actuals[0]->Rtype();
        spec._build_body = [](FUNC_SCOPE& helper, NODE_PTR body,
                              const SPOS& spos) {
          return Emit_synthetic_kernel_body(helper, body, spos)._result;
        };
        return spec;
      }));

  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_lowering_registry(&registry);
  VECTOR_CONFIG config;
  config._decompose_mid_op = true;
  std::unique_ptr<GLOB_SCOPE> lowered(
      Vector_driver(source._glob.get(), vector_ctx, nullptr, config));

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(callback_count, 1U);
  EXPECT_EQ(Function_count(*lowered), 2U);

  FUNC_SCOPE& caller = lowered->Open_func_scope(source._scope->Id());
  FUNC_SCOPE* helper = Find_helper(*lowered, caller.Id());
  ASSERT_NE(helper, nullptr);
  EXPECT_STREQ(helper->Owning_func()->Name()->Char_str(), helper_name.c_str());
  EXPECT_EQ(&helper->Glob_scope(), lowered.get());
  EXPECT_EQ(&helper->Owning_func()->Entry_point()->Glob_scope(), lowered.get());
  ASSERT_EQ(helper->Formal_cnt(), 2U);
  EXPECT_EQ(helper->Formal(0)->Defining_func_scope(), helper);
  EXPECT_EQ(helper->Formal(1)->Defining_func_scope(), helper);

  const AIR_STATS caller_stats = Get_stats(caller);
  EXPECT_EQ(caller_stats._calls, 1U);
  EXPECT_EQ(caller_stats._nn_adds, 0U);
  ASSERT_EQ(caller_stats._call_stmts.size(), 1U);
  const AIR_STATS helper_stats = Get_stats(*helper);
  EXPECT_EQ(helper_stats._calls, 0U);
  EXPECT_EQ(helper_stats._nn_adds, 0U);
  EXPECT_EQ(helper_stats._vector_adds, 2U);
  EXPECT_EQ(helper_stats._loops, 1U);
  EXPECT_EQ(helper_stats._retvs, 1U);
  EXPECT_EQ(helper_stats._ldcs, 1U);
  Expect_helper_ownership(helper->Container().Entry_node(), *helper);

  NODE_PTR caller_body = caller.Container().Entry_node()->Last_child();
  STMT_PTR terminal = Null_ptr;
  for (STMT_PTR stmt = caller_body->Begin_stmt();
       stmt != caller_body->End_stmt(); stmt = stmt->Next()) {
    terminal = stmt;
  }
  ASSERT_NE(terminal, Null_ptr);
  ASSERT_EQ(terminal->Node()->Opcode(), air::core::OPC_RETV);
  NODE_PTR replacement = terminal->Node()->Child(0);
  NODE_PTR real_call   = caller_stats._call_stmts[0]->Node();
  std::vector<NODE_PTR> expected_actuals;
  for (uint32_t idx = 0; idx < real_call->Num_arg(); ++idx) {
    expected_actuals.push_back(real_call->Child(idx));
  }
  VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
      nn::vector::test::Check_vector_kernel_call_bridge(
          caller, caller_stats._call_stmts[0], expected_actuals, replacement,
          *helper);
  EXPECT_TRUE(bridge._equal) << bridge._message;

  ASSERT_EQ(expected_actuals.size(), 2U);
  std::vector<NODE_PTR> swapped_actuals{expected_actuals[1],
                                        expected_actuals[0]};
  VECTOR_KERNEL_AIR_COMPARE_RESULT swapped_bridge =
      nn::vector::test::Check_vector_kernel_call_bridge(
          caller, caller_stats._call_stmts[0], swapped_actuals, replacement,
          *helper);
  EXPECT_FALSE(swapped_bridge._equal);
  EXPECT_EQ(swapped_bridge._message,
            "CALL actual order does not match expected inputs");

  REFERENCE_KERNEL_IR reference =
      Build_reference_kernel("renamed_native_reference", 97);
  ASSERT_TRUE(reference._glob->Verify_ir());
  VECTOR_KERNEL_AIR_COMPARE_RESULT equal =
      nn::vector::test::Compare_native_and_helper_vector_kernel_air(
          Native_view(reference), *helper);
  EXPECT_TRUE(equal._equal) << equal._message;

  REFERENCE_KERNEL_IR changed_constant =
      Build_reference_kernel("changed_constant", 101, 2.0F);
  ASSERT_TRUE(changed_constant._glob->Verify_ir());
  VECTOR_KERNEL_AIR_COMPARE_RESULT constant_mismatch =
      nn::vector::test::Compare_native_and_helper_vector_kernel_air(
          Native_view(changed_constant), *helper);
  EXPECT_FALSE(constant_mismatch._equal);

  REFERENCE_KERNEL_IR changed_order =
      Build_reference_kernel("changed_order", 103, 1.0F, true);
  ASSERT_TRUE(changed_order._glob->Verify_ir());
  VECTOR_KERNEL_AIR_COMPARE_RESULT order_mismatch =
      nn::vector::test::Compare_native_and_helper_vector_kernel_air(
          Native_view(changed_order), *helper);
  EXPECT_FALSE(order_mismatch._equal);

  PREG_PTR wrong_preg = caller.New_preg(replacement->Rtype());
  NODE_PTR wrong_ldp  =
      caller.Container().New_ldp(wrong_preg, replacement->Spos());
  VECTOR_KERNEL_AIR_COMPARE_RESULT bad_bridge =
      nn::vector::test::Check_vector_kernel_call_bridge(
          caller, caller_stats._call_stmts[0], expected_actuals, wrong_ldp,
          *helper);
  EXPECT_FALSE(bad_bridge._equal);
  EXPECT_EQ(bad_bridge._message,
            "replacement is not caller LDP of the CALL result preg");

  NODE_PTR detached_ldp =
      caller.Container().New_ldp(real_call->Ret_preg(), replacement->Spos());
  VECTOR_KERNEL_AIR_COMPARE_RESULT detached_ldp_bridge =
      nn::vector::test::Check_vector_kernel_call_bridge(
          caller, caller_stats._call_stmts[0], expected_actuals, detached_ldp,
          *helper);
  EXPECT_FALSE(detached_ldp_bridge._equal);
  EXPECT_EQ(detached_ldp_bridge._message,
            "replacement LDP is not present exactly once in caller");

  PREG_PTR detached_preg = caller.New_preg(real_call->Ret_preg()->Type());
  STMT_PTR detached_call = caller.Container().New_call(
      helper->Owning_func()->Entry_point(), detached_preg,
      real_call->Num_arg(), real_call->Spos());
  for (uint32_t idx = 0; idx < real_call->Num_arg(); ++idx) {
    caller.Container().New_arg(detached_call, idx, real_call->Child(idx));
  }
  VECTOR_KERNEL_AIR_COMPARE_RESULT detached_bridge =
      nn::vector::test::Check_vector_kernel_call_bridge(
          caller, detached_call, expected_actuals, replacement, *helper);
  EXPECT_FALSE(detached_bridge._equal);
  EXPECT_EQ(detached_bridge._message,
            "the supplied CALL is not present exactly once in caller");
}

TEST_F(Tensor2VectorDslMaterialization,
       MaterializesDestinationOwnedTypedNestedLoopHelper) {
  SOURCE_ADD_IR source = Build_source_add("typed_nested_loop_source", 211);
  ASSERT_TRUE(source._glob->Verify_ir());

  const std::string specialization_key =
      "vector-kernel-typed-nested-loop:v1";
  const std::string helper_name =
      "__ace_vkernel_typed_nested_loop_" +
      Vector_kernel_sha256(specialization_key);
  uint32_t callback_count = 0;

  VECTOR_KERNEL_LOWERING_REGISTRY registry;
  ASSERT_TRUE(registry.Register(
      OPCODE(nn::core::NN, nn::core::OPCODE::ADD),
      [&](NODE_PTR source_node, const std::vector<NODE_PTR>& actuals,
          GLOB_SCOPE& destination) {
        ++callback_count;
        EXPECT_EQ(source_node->Opcode(),
                  OPCODE(nn::core::NN, nn::core::OPCODE::ADD));
        EXPECT_EQ(actuals.size(), 2U);
        EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);
        EXPECT_EQ(actuals[1]->Container()->Glob_scope(), &destination);

        VECTOR_KERNEL_HELPER_SPEC spec;
        spec._specialization_key = specialization_key;
        spec._helper_name        = helper_name;
        spec._formal_types       = {actuals[0]->Rtype(), actuals[1]->Rtype()};
        spec._result_type        = actuals[0]->Rtype();
        spec._build_body = [](FUNC_SCOPE& helper, NODE_PTR body,
                              const SPOS& spos) {
          return Emit_typed_nested_loop_kernel_body(helper, body, spos);
        };
        return spec;
      }));

  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_lowering_registry(&registry);
  VECTOR_CONFIG config;
  config._decompose_mid_op = true;
  std::unique_ptr<GLOB_SCOPE> lowered(
      Vector_driver(source._glob.get(), vector_ctx, nullptr, config));

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(callback_count, 1U);
  EXPECT_EQ(Function_count(*lowered), 2U);

  FUNC_SCOPE& caller = lowered->Open_func_scope(source._scope->Id());
  FUNC_SCOPE* helper = Find_helper(*lowered, caller.Id());
  ASSERT_NE(helper, nullptr);
  EXPECT_STREQ(helper->Owning_func()->Name()->Char_str(), helper_name.c_str());
  ASSERT_EQ(helper->Formal_cnt(), 2U);

  const AIR_STATS caller_stats = Get_stats(caller);
  EXPECT_EQ(caller_stats._calls, 1U);
  EXPECT_EQ(caller_stats._nn_adds, 0U);
  ASSERT_EQ(caller_stats._call_stmts.size(), 1U);

  const AIR_STATS helper_stats = Get_stats(*helper);
  EXPECT_EQ(helper_stats._calls, 0U);
  EXPECT_EQ(helper_stats._nn_adds, 0U);
  EXPECT_EQ(helper_stats._vector_adds, 2U);
  EXPECT_EQ(helper_stats._loops, 2U);
  EXPECT_EQ(helper_stats._retvs, 1U);
  EXPECT_EQ(helper_stats._zeros, 1U);
  EXPECT_EQ(helper_stats._core_muls, 1U);
  EXPECT_EQ(helper_stats._core_lts, 2U);
  EXPECT_EQ(helper_stats._core_shls, 1U);
  EXPECT_EQ(helper_stats._core_adds, 3U);
  Expect_helper_ownership(helper->Container().Entry_node(), *helper);

  TYPE_PTR vector_type = helper->Formal(0)->Type();
  TYPE_PTR s32 = lowered->Prim_type(PRIMITIVE_TYPE::INT_S32);
  Expect_typed_nested_loop_air(helper->Container().Entry_node(), *helper,
                               vector_type, s32);

  NODE_PTR caller_body = caller.Container().Entry_node()->Last_child();
  STMT_PTR terminal    = Null_ptr;
  for (STMT_PTR stmt = caller_body->Begin_stmt();
       stmt != caller_body->End_stmt(); stmt = stmt->Next()) {
    terminal = stmt;
  }
  ASSERT_NE(terminal, Null_ptr);
  ASSERT_EQ(terminal->Node()->Opcode(), air::core::OPC_RETV);
  NODE_PTR replacement = terminal->Node()->Child(0);
  STMT_PTR call = caller_stats._call_stmts[0];
  ASSERT_EQ(call->Node()->Num_arg(), 2U);
  ASSERT_EQ(call->Node()->Child(0)->Opcode(), air::core::OPC_LD);
  ASSERT_EQ(call->Node()->Child(1)->Opcode(), air::core::OPC_LD);
  EXPECT_EQ(call->Node()->Child(0)->Addr_datum(), caller.Formal(0));
  EXPECT_EQ(call->Node()->Child(1)->Addr_datum(), caller.Formal(1));
  std::vector<NODE_PTR> expected_actuals;
  for (uint32_t idx = 0; idx < call->Node()->Num_arg(); ++idx) {
    expected_actuals.push_back(call->Node()->Child(idx));
  }
  VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
      Check_vector_kernel_call_bridge(caller, call, expected_actuals,
                                      replacement, *helper);
  EXPECT_TRUE(bridge._equal) << bridge._message;

  // This is deliberately the pre-inline Phase-A handoff: the typed CALL/LDP
  // remains present and no Python inliner/pass runs in this fixture.
  EXPECT_EQ(Get_stats(caller)._calls, 1U);
  ASSERT_TRUE(lowered->Verify_ir());
}

TEST_F(Tensor2VectorDslMaterialization,
       MaterializesGenuineVectorPrimitivesAfterMv2vOpt) {
  SOURCE_ADD_IR source =
      Build_heterogeneous_source_add("m3_vector_primitive_source", 307);
  ASSERT_TRUE(source._glob->Verify_ir());

  const std::string specialization_key =
      "vector-kernel-genuine-primitives:v1";
  const std::string helper_name =
      "__ace_vkernel_genuine_primitives_" +
      Vector_kernel_sha256(specialization_key);
  uint32_t callback_count = 0;

  VECTOR_KERNEL_LOWERING_REGISTRY registry;
  ASSERT_TRUE(registry.Register(
      OPCODE(nn::core::NN, nn::core::OPCODE::ADD),
      [&](NODE_PTR source_node, const std::vector<NODE_PTR>& actuals,
          GLOB_SCOPE& destination) {
        ++callback_count;
        EXPECT_EQ(source_node->Opcode(),
                  OPCODE(nn::core::NN, nn::core::OPCODE::ADD));
        EXPECT_EQ(actuals.size(), 2U);
        EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);
        EXPECT_EQ(actuals[1]->Container()->Glob_scope(), &destination);

        VECTOR_KERNEL_HELPER_SPEC spec;
        spec._specialization_key = specialization_key;
        spec._helper_name        = helper_name;
        spec._formal_types       = {actuals[0]->Rtype(), actuals[1]->Rtype()};
        spec._result_type        = actuals[1]->Rtype();
        spec._build_body = [](FUNC_SCOPE& helper, NODE_PTR body,
                              const SPOS& spos) {
          return Emit_vector_primitive_kernel_body(helper, body, spos);
        };
        return spec;
      }));

  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_lowering_registry(&registry);
  VECTOR_CONFIG config;
  config._decompose_mid_op = true;
  std::unique_ptr<GLOB_SCOPE> lowered(
      Vector_driver(source._glob.get(), vector_ctx, nullptr, config));

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(callback_count, 1U);
  EXPECT_EQ(Function_count(*lowered), 2U);

  FUNC_SCOPE& caller = lowered->Open_func_scope(source._scope->Id());
  FUNC_SCOPE* helper = Find_helper(*lowered, caller.Id());
  ASSERT_NE(helper, nullptr);
  EXPECT_STREQ(helper->Owning_func()->Name()->Char_str(), helper_name.c_str());
  ASSERT_EQ(helper->Formal_cnt(), 2U);
  EXPECT_EQ(helper->Formal(0)->Type()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({SYNTHETIC_WIDTH}));
  EXPECT_EQ(helper->Formal(1)->Type()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({M3_WIDE_WIDTH}));

  const AIR_STATS caller_stats = Get_stats(caller);
  EXPECT_EQ(caller_stats._calls, 1U);
  EXPECT_EQ(caller_stats._nn_adds, 0U);
  ASSERT_EQ(caller_stats._call_stmts.size(), 1U);

  const AIR_STATS helper_stats = Get_stats(*helper);
  EXPECT_EQ(helper_stats._calls, 0U);
  EXPECT_EQ(helper_stats._nn_adds, 0U);
  EXPECT_EQ(helper_stats._vector_adds, 2U);
  EXPECT_EQ(helper_stats._vector_muls, 1U);
  EXPECT_EQ(helper_stats._vector_rolls, 1U);
  EXPECT_EQ(helper_stats._vector_slices, 1U);
  EXPECT_EQ(helper_stats._loops, 1U);
  EXPECT_EQ(helper_stats._retvs, 1U);
  EXPECT_EQ(helper_stats._zeros, 1U);
  EXPECT_EQ(helper_stats._ldcs, 1U);
  Expect_helper_ownership(helper->Container().Entry_node(), *helper);

  ASSERT_EQ(helper_stats._vector_roll_nodes.size(), 1U);
  NODE_PTR roll = helper_stats._vector_roll_nodes[0];
  ASSERT_EQ(roll->Num_child(), 2U);
  ASSERT_TRUE(roll->Has_rtype());
  EXPECT_EQ(roll->Rtype(), roll->Child(0)->Rtype());
  EXPECT_EQ(roll->Child(1)->Domain(), air::core::CORE);
  ASSERT_TRUE(roll->Child(1)->Rtype()->Is_prim());
  EXPECT_EQ(roll->Child(1)->Rtype()->Cast_to_prim()->Encoding(),
            PRIMITIVE_TYPE::INT_S32);
  ASSERT_EQ(helper_stats._rnums.size(), 1U);
  EXPECT_EQ(helper_stats._rnums[0], std::vector<int>({-1, 0, 1}));

  ASSERT_EQ(helper_stats._vector_slice_nodes.size(), 1U);
  NODE_PTR slice = helper_stats._vector_slice_nodes[0];
  ASSERT_EQ(slice->Num_child(), 3U);
  EXPECT_EQ(slice->Child(0)->Opcode(), air::core::OPC_LDC);
  EXPECT_EQ(slice->Child(0)->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({M3_ROW_COUNT, M3_WIDE_WIDTH}));
  EXPECT_EQ(slice->Child(1)->Domain(), air::core::CORE);
  EXPECT_EQ(slice->Child(1)->Rtype()->Cast_to_prim()->Encoding(),
            PRIMITIVE_TYPE::INT_S32);
  EXPECT_EQ(slice->Child(2)->Opcode(), air::core::OPC_INTCONST);
  EXPECT_EQ(slice->Child(2)->Rtype()->Cast_to_prim()->Encoding(),
            PRIMITIVE_TYPE::INT_S32);
  EXPECT_EQ(slice->Child(2)->Intconst(), M3_WIDE_WIDTH);
  EXPECT_EQ(slice->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({M3_WIDE_WIDTH}));

  ASSERT_EQ(helper_stats._vector_mul_nodes.size(), 1U);
  NODE_PTR multiply = helper_stats._vector_mul_nodes[0];
  EXPECT_EQ(multiply->Rtype(), multiply->Child(0)->Rtype());

  NODE_PTR widening_add = Null_ptr;
  for (NODE_PTR add : helper_stats._vector_add_nodes) {
    if (add->Child(0)->Rtype()->Cast_to_arr()->Elem_count() ==
        SYNTHETIC_WIDTH) {
      widening_add = add;
    }
  }
  ASSERT_NE(widening_add, Null_ptr);
  EXPECT_EQ(widening_add->Rtype(), widening_add->Child(1)->Rtype());
  EXPECT_EQ(widening_add->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({M3_WIDE_WIDTH}));

  EXPECT_EQ(helper_stats._slots,
            std::vector<uint32_t>({static_cast<uint32_t>(M3_WIDE_WIDTH)}));
  const std::string normalized = Normalize_vector_kernel_helper(*helper);
  EXPECT_NE(normalized.find("nums:s32:3:"), std::string::npos);
  EXPECT_NE(normalized.find("slot:u32:1:"), std::string::npos);

  NODE_PTR caller_body = caller.Container().Entry_node()->Last_child();
  STMT_PTR terminal    = Null_ptr;
  for (STMT_PTR stmt = caller_body->Begin_stmt();
       stmt != caller_body->End_stmt(); stmt = stmt->Next()) {
    terminal = stmt;
  }
  ASSERT_NE(terminal, Null_ptr);
  ASSERT_EQ(terminal->Node()->Opcode(), air::core::OPC_RETV);
  NODE_PTR replacement = terminal->Node()->Child(0);
  STMT_PTR call        = caller_stats._call_stmts[0];
  std::vector<NODE_PTR> expected_actuals;
  for (uint32_t idx = 0; idx < call->Node()->Num_arg(); ++idx) {
    expected_actuals.push_back(call->Node()->Child(idx));
  }
  VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
      Check_vector_kernel_call_bridge(caller, call, expected_actuals,
                                      replacement, *helper);
  EXPECT_TRUE(bridge._equal) << bridge._message;

  // M3 intentionally stops at valid post-Mv2v pre-Vector2SIHE AIR.
  EXPECT_EQ(Get_stats(caller)._calls, 1U);
  ASSERT_TRUE(lowered->Verify_ir());
}

TEST_F(Tensor2VectorDslMaterialization,
       DisabledRegistryPreservesNativeLowering) {
  SOURCE_ADD_IR source = Build_source_add("m1_disabled_add", 29);
  ASSERT_TRUE(source._glob->Verify_ir());

  VECTOR_CTX    vector_ctx;
  VECTOR_CONFIG config;
  config._decompose_mid_op = true;
  EXPECT_EQ(vector_ctx.Vector_kernel_lowering_registry(), nullptr);
  EXPECT_FALSE(config.Python_dsl());

  std::unique_ptr<GLOB_SCOPE> lowered(
      Vector_driver(source._glob.get(), vector_ctx, nullptr, config));
  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(Function_count(*lowered), 1U);

  FUNC_SCOPE& caller = lowered->Open_func_scope(source._scope->Id());
  const AIR_STATS stats = Get_stats(caller);
  EXPECT_EQ(stats._calls, 0U);
  EXPECT_EQ(stats._nn_adds, 0U);
  EXPECT_EQ(stats._vector_adds, 1U);
  EXPECT_EQ(stats._loops, 0U);
}

TEST_F(Tensor2VectorDslMaterialization,
       RegistryRejectsDuplicateOpcodeAndSupportsRemoval) {
  VECTOR_KERNEL_LOWERING_REGISTRY registry;
  const OPCODE add_opcode(nn::core::NN, nn::core::OPCODE::ADD);
  auto selector = [](NODE_PTR, const std::vector<NODE_PTR>&,
                     GLOB_SCOPE&) { return VECTOR_KERNEL_HELPER_SPEC{}; };

  EXPECT_TRUE(registry.Register(add_opcode, selector));
  EXPECT_TRUE(registry.Has(add_opcode));
  EXPECT_FALSE(registry.Register(add_opcode, selector));
  EXPECT_TRUE(registry.Unregister(add_opcode));
  EXPECT_FALSE(registry.Has(add_opcode));
  EXPECT_FALSE(registry.Unregister(add_opcode));
}
