//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks2hpoly.h"

#include <limits>
#include <vector>

namespace fhe {
namespace poly {

using namespace air::base;

namespace {

const uint32_t* Linear_transform_scalar_attr(NODE_PTR node, const char* name) {
  uint32_t        count = 0;
  const uint32_t* value = node->Attr<uint32_t>(name, &count);
  CMPLR_ASSERT(value != nullptr && count == 1,
               "linear_transform requires a scalar descriptor attribute");
  return value;
}

NODE_PAIR Materialize_poly_pair(POLY_LOWER_CTX& ctx, NODE_PTR c0, NODE_PTR c1,
                                const SPOS& spos) {
  POLY_IR_GEN& pgen = ctx.Poly_gen();
  VAR          v_c0(ctx.Func_scope(), pgen.New_poly_var(spos));
  VAR          v_c1(ctx.Func_scope(), pgen.New_poly_var(spos));
  ctx.Prepend(pgen.New_var_store(c0, v_c0, spos));
  ctx.Prepend(pgen.New_var_store(c1, v_c1, spos));
  return {pgen.New_var_load(v_c0, spos), pgen.New_var_load(v_c1, spos)};
}

NODE_PTR Materialize_poly(POLY_LOWER_CTX& ctx, NODE_PTR value,
                          const SPOS& spos) {
  POLY_IR_GEN& pgen = ctx.Poly_gen();
  VAR          var(ctx.Func_scope(), pgen.New_poly_var(spos));
  ctx.Prepend(pgen.New_var_store(value, var, spos));
  return pgen.New_var_load(var, spos);
}

void Free_materialized_poly(POLY_LOWER_CTX& ctx, NODE_PTR value,
                            const SPOS& spos) {
  CMPLR_ASSERT(value->Opcode() == air::core::OPC_LD,
               "owned polynomial must be a materialized symbol load");
  VAR var(ctx.Func_scope(), value->Addr_datum());
  ctx.Prepend(ctx.Poly_gen().New_free_poly(var, spos));
}

void Free_materialized_poly_pair(POLY_LOWER_CTX& ctx, const NODE_PAIR& value,
                                 const SPOS& spos) {
  Free_materialized_poly(ctx, value.first, spos);
  Free_materialized_poly(ctx, value.second, spos);
}

}  // namespace

POLY_LOWER_RETV CKKS2HPOLY::Handle_bin_arith_cc(POLY_LOWER_CTX&     ctx,
                                                air::base::NODE_PTR node,
                                                POLY_LOWER_RETV     opnd0_pair,
                                                POLY_LOWER_RETV     opnd1_pair,
                                                OPCODE              opc) {
  air::base::CONTAINER* cntr = ctx.Poly_gen().Container();
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null()), "null node");

  air::base::NODE_PTR opnd0 = ctx.Poly_gen().New_bin_arith(
      opc, opnd0_pair.Node1(), opnd1_pair.Node1(), node->Spos());
  air::base::NODE_PTR opnd1 = ctx.Poly_gen().New_bin_arith(
      opc, opnd0_pair.Node2(), opnd1_pair.Node2(), node->Spos());

  return POLY_LOWER_RETV(opnd0_pair.Kind(), opnd0, opnd1);
}

POLY_LOWER_RETV CKKS2HPOLY::Handle_bin_arith_cp(POLY_LOWER_CTX&     ctx,
                                                air::base::NODE_PTR node,
                                                POLY_LOWER_RETV     opnd0_pair,
                                                POLY_LOWER_RETV     opnd1_pair,
                                                OPCODE              opc) {
  air::base::CONTAINER* cntr = ctx.Poly_gen().Container();
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null()), "null node");

  air::base::NODE_PTR opnd0 = ctx.Poly_gen().New_bin_arith(
      opc, opnd0_pair.Node1(), opnd1_pair.Node1(), node->Spos());

  air::base::NODE_PTR opnd1 = opnd0_pair.Node2();

  return POLY_LOWER_RETV(opnd0_pair.Kind(), opnd0, opnd1);
}

POLY_LOWER_RETV CKKS2HPOLY::Handle_bin_arith_cf(POLY_LOWER_CTX&     ctx,
                                                air::base::NODE_PTR node,
                                                POLY_LOWER_RETV     opnd0_pair,
                                                POLY_LOWER_RETV     opnd1_pair,
                                                OPCODE              opc) {
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null() &&
                opnd0_pair.Kind() == RETV_KIND::RK_CIPH_POLY &&
                opnd1_pair.Node()->Rtype()->Is_float()),
               "invalid node");

  air::base::CONTAINER* cntr = ctx.Poly_gen().Container();
  air::base::SPOS       spos = node->Spos();

  air::base::NODE_PTR n_child0 = node->Child(0);
  CONST_VAR&          v_child0 = ctx.Poly_gen().Node_var(n_child0);

  // 1. encode float data
  air::base::NODE_PTR n_encode = Gen_encode_float_from_ciph(
      ctx, node, v_child0, opnd1_pair.Node1(), false);
  air::base::ADDR_DATUM_PTR sym = ctx.Poly_gen().New_plain_var(spos);
  CONST_VAR& v_encode = ctx.Poly_gen().Add_node_var(node->Child(1), sym);
  air::base::STMT_PTR s_encode =
      ctx.Poly_gen().New_var_store(n_encode, v_encode, spos);
  ctx.Prepend(s_encode);

  // 2. add ciph with encoded float
  air::base::NODE_PTR n_plain =
      ctx.Poly_gen().New_plain_poly_load(v_encode, false, spos);

  air::base::NODE_PTR opnd0 = ctx.Poly_gen().New_bin_arith(
      opc, opnd0_pair.Node1(), n_plain, node->Spos());
  air::base::NODE_PTR opnd1 = cntr->Clone_node_tree(opnd0_pair.Node2());

  return POLY_LOWER_RETV(opnd0_pair.Kind(), opnd0, opnd1);
}

POLY_LOWER_RETV CKKS2HPOLY::Handle_mul_ciph(POLY_LOWER_CTX& ctx, NODE_PTR node,
                                            POLY_LOWER_RETV opnd0_pair,
                                            POLY_LOWER_RETV opnd1_pair) {
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null() &&
                opnd0_pair.Kind() == RETV_KIND::RK_CIPH_POLY &&
                opnd1_pair.Kind() == RETV_KIND::RK_CIPH_POLY &&
                ctx.Lower_ctx()->Is_cipher3_type(node->Rtype_id())),
               "invalid mul_ciph");
  POLY_IR_GEN& pgen = ctx.Poly_gen();
  CONTAINER*   cntr = pgen.Container();
  GLOB_SCOPE*  glob = cntr->Glob_scope();
  SPOS         spos = node->Spos();

  // 1. v_mul_0 = opnd0_c0 * opnd1_c0
  NODE_PTR n_mul_0 =
      pgen.New_poly_mul(opnd0_pair.Node1(), opnd1_pair.Node1(), node->Spos());

  // 2. n_mul_1 = n_mul_1_0 + n_mul_1_1
  // n_mul_1_0 = opnd0_c1 * opnd1_c0
  // n_mul_1_1 = opnd0_c0 * opnd1_c0
  NODE_PTR n_mul_1_0 = pgen.New_poly_mul(
      opnd0_pair.Node2(), cntr->Clone_node_tree(opnd1_pair.Node1()),
      node->Spos());
  NODE_PTR n_mul_1_1 = pgen.New_poly_mul(
      cntr->Clone_node_tree(opnd0_pair.Node1()),
      cntr->Clone_node_tree(opnd1_pair.Node2()), node->Spos());
  NODE_PTR n_mul_1 = pgen.New_poly_add(n_mul_1_0, n_mul_1_1, node->Spos());

  // 3. n_mul_2 = opnd0_c1 * opnd1_c1
  NODE_PTR n_mul_2 = pgen.New_poly_mul(
      cntr->Clone_node_tree(opnd0_pair.Node2()),
      cntr->Clone_node_tree(opnd1_pair.Node2()), node->Spos());

  pgen.Set_mul_ciph(n_mul_0);
  pgen.Set_mul_ciph(n_mul_1_0);
  pgen.Set_mul_ciph(n_mul_1_1);
  pgen.Set_mul_ciph(n_mul_2);

  return POLY_LOWER_RETV(RETV_KIND::RK_CIPH3_POLY, n_mul_0, n_mul_1, n_mul_2);
}

POLY_LOWER_RETV CKKS2HPOLY::Handle_mul_plain(POLY_LOWER_CTX& ctx, NODE_PTR node,
                                             POLY_LOWER_RETV opnd0_pair,
                                             POLY_LOWER_RETV opnd1_pair) {
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null() &&
                opnd0_pair.Kind() == RETV_KIND::RK_CIPH_POLY &&
                opnd1_pair.Kind() == RETV_KIND::RK_PLAIN_POLY),
               "null node");

  CONTAINER* cntr = ctx.Poly_gen().Container();

  // TODO: shall we share the nodes in different op?
  NODE_PTR mul_0 = ctx.Poly_gen().New_poly_mul(
      opnd0_pair.Node1(), opnd1_pair.Node1(), node->Spos());
  NODE_PTR mul_1 = ctx.Poly_gen().New_poly_mul(
      opnd0_pair.Node2(), cntr->Clone_node_tree(opnd1_pair.Node1()),
      node->Spos());

  return POLY_LOWER_RETV(RETV_KIND::RK_CIPH_POLY, mul_0, mul_1);
}

POLY_LOWER_RETV CKKS2HPOLY::Handle_mul_float(POLY_LOWER_CTX& ctx, NODE_PTR node,
                                             POLY_LOWER_RETV opnd0_pair,
                                             POLY_LOWER_RETV opnd1_pair) {
  CMPLR_ASSERT((!opnd0_pair.Is_null() && !opnd1_pair.Is_null() &&
                opnd0_pair.Kind() == RETV_KIND::RK_CIPH_POLY &&
                opnd1_pair.Node()->Rtype()->Is_float()),
               "invalid node");

  CONTAINER* cntr = ctx.Poly_gen().Container();
  SPOS       spos = node->Spos();

  // encode float data
  NODE_PTR   n_child0 = node->Child(0);
  CONST_VAR& v_child0 = ctx.Poly_gen().Node_var(n_child0);
  NODE_PTR   n_encode =
      Gen_encode_float_from_ciph(ctx, node, v_child0, opnd1_pair.Node1(), true);
  air::base::ADDR_DATUM_PTR sym = ctx.Poly_gen().New_plain_var(spos);
  CONST_VAR& v_encode = ctx.Poly_gen().Add_node_var(node->Child(1), sym);
  STMT_PTR   s_encode = ctx.Poly_gen().New_var_store(n_encode, v_encode, spos);
  ctx.Prepend(s_encode);

  NODE_PTR n_plain = ctx.Poly_gen().New_plain_poly_load(v_encode, false, spos);

  NODE_PTR mul_0 =
      ctx.Poly_gen().New_poly_mul(opnd0_pair.Node1(), n_plain, spos);
  NODE_PTR mul_1 = ctx.Poly_gen().New_poly_mul(
      opnd0_pair.Node2(), cntr->Clone_node_tree(n_plain), spos);
  return POLY_LOWER_RETV(RETV_KIND::RK_CIPH_POLY, mul_0, mul_1);
}

POLY_LOWER_RETV CKKS2HPOLY::Expand_relin(POLY_LOWER_CTX& ctx, NODE_PTR node,
                                         NODE_PTR n_c0, NODE_PTR n_c1,
                                         NODE_PTR n_c2) {
  CONTAINER*  cntr = ctx.Poly_gen().Container();
  GLOB_SCOPE* glob = ctx.Poly_gen().Glob_scope();
  SPOS        spos = node->Spos();

  // generate alloc for RNS_POLY array specific to the CPU target
  VAR v_precomp(ctx.Func_scope(), ctx.Poly_gen().New_polys_var(spos));
  if (ctx.Config().Lower_to_lpoly()) {
    NODE_PTR n_alloc = ctx.Poly_gen().New_alloc_for_precomp(n_c2, spos);
    STMT_PTR s_alloc = ctx.Poly_gen().New_var_store(n_alloc, v_precomp, spos);
    ctx.Prepend(s_alloc);
  }

  // generate precompute: v_precompute = KSW_PRECOMP(n_c2)
  NODE_PTR n_precomp = ctx.Poly_gen().New_precomp(n_c2, spos);
  STMT_PTR s_precomp = ctx.Poly_gen().New_var_store(n_precomp, v_precomp, spos);
  ctx.Prepend(s_precomp);

  // generate get relin key: v_swk = SWK()
  VAR      v_swk_c0(ctx.Func_scope(), ctx.Poly_gen().New_swk_var(spos));
  VAR      v_swk_c1(ctx.Func_scope(), ctx.Poly_gen().New_swk_var(spos));
  NODE_PTR n_swk_c0 = ctx.Poly_gen().New_swk_c0(false, spos);
  NODE_PTR n_swk_c1 = ctx.Poly_gen().New_swk_c1(false, spos);
  STMT_PTR s_swk_c0 = ctx.Poly_gen().New_var_store(n_swk_c0, v_swk_c0, spos);
  STMT_PTR s_swk_c1 = ctx.Poly_gen().New_var_store(n_swk_c1, v_swk_c1, spos);
  ctx.Prepend(s_swk_c0);
  ctx.Prepend(s_swk_c1);

  // dot prod size
  NODE_PTR                  n_num_part;
  air::base::CONST_TYPE_PTR ui32_type =
      ctx.Poly_gen().Get_type(VAR_TYPE_KIND::UINT32);
  if (ctx.Config().Lower_to_lpoly()) {
    // for targeting cpu, the value is calculate through rtlib call
    n_num_part = ctx.Poly_gen().New_poly_node(NUM_DECOMP, ui32_type, spos);
    n_num_part->Set_child(0, cntr->Clone_node_tree(n_c1));
  } else {
    // for targeting hpu, the value should be calculate through attribute
    // the logic canbe moved to constant folding
    n_num_part = cntr->New_intconst(ui32_type,
                                    ctx.Poly_gen().Get_num_decomp(node), spos);
  }

  // generate keyswitch dot product n_prod = DOT_PROD()
  NODE_PTR n_prod_c0 = ctx.Poly_gen().New_dot_prod(
      ctx.Poly_gen().New_var_load(v_precomp, spos),
      ctx.Poly_gen().New_var_load(v_swk_c0, spos), n_num_part, spos);
  NODE_PTR n_prod_c1 =
      ctx.Poly_gen().New_dot_prod(ctx.Poly_gen().New_var_load(v_precomp, spos),
                                  ctx.Poly_gen().New_var_load(v_swk_c1, spos),
                                  cntr->Clone_node_tree(n_num_part), spos);

  // generate moddown: MOD_DOWN(n_prod_c0)
  //                   MOD_DOWN(n_prod_c1)
  NODE_PTR n_md_c0 = ctx.Poly_gen().New_mod_down(n_prod_c0, spos);
  NODE_PTR n_md_c1 = ctx.Poly_gen().New_mod_down(n_prod_c1, spos);

  // post add
  NODE_PTR n_add_c0 = ctx.Poly_gen().New_poly_add(n_c0, n_md_c0, spos);
  NODE_PTR n_add_c1 = ctx.Poly_gen().New_poly_add(n_c1, n_md_c1, spos);
  return POLY_LOWER_RETV(RETV_KIND::RK_CIPH_POLY, n_add_c0, n_add_c1);
}

POLY_LOWER_RETV CKKS2HPOLY::Expand_rotate(POLY_LOWER_CTX& ctx, NODE_PTR node,
                                          NODE_PTR n_c0, NODE_PTR n_c1,
                                          NODE_PTR    n_rot_idx,
                                          const SPOS& spos) {
  CONTAINER*  cntr = ctx.Poly_gen().Container();
  GLOB_SCOPE* glob = ctx.Poly_gen().Glob_scope();

  // generate alloc for RNS_POLY array for cpu target
  VAR v_precomp(ctx.Func_scope(), ctx.Poly_gen().New_polys_var(spos));
  if (ctx.Config().Lower_to_lpoly()) {
    NODE_PTR n_alloc = ctx.Poly_gen().New_alloc_for_precomp(n_c1, spos);
    STMT_PTR s_alloc = ctx.Poly_gen().New_var_store(n_alloc, v_precomp, spos);
    ctx.Prepend(s_alloc);
  }

  // generate precompute: v_precompute = KSW_PRECOMP(n_c1)
  NODE_PTR n_precomp = ctx.Poly_gen().New_precomp(n_c1, spos);
  STMT_PTR s_precomp = ctx.Poly_gen().New_var_store(n_precomp, v_precomp, spos);
  ctx.Prepend(s_precomp);

  // get switch key with given index
  VAR      v_swk_c0(ctx.Func_scope(), ctx.Poly_gen().New_swk_var(spos));
  VAR      v_swk_c1(ctx.Func_scope(), ctx.Poly_gen().New_swk_var(spos));
  NODE_PTR n_swk_c0 = ctx.Poly_gen().New_swk_c0(true, spos, n_rot_idx);
  NODE_PTR n_swk_c1 =
      ctx.Poly_gen().New_swk_c1(true, spos, cntr->Clone_node_tree(n_rot_idx));
  STMT_PTR s_swk_c0 = ctx.Poly_gen().New_var_store(n_swk_c0, v_swk_c0, spos);
  STMT_PTR s_swk_c1 = ctx.Poly_gen().New_var_store(n_swk_c1, v_swk_c1, spos);
  ctx.Prepend(s_swk_c0);
  ctx.Prepend(s_swk_c1);

  // dot prod size
  NODE_PTR                  n_num_part;
  air::base::CONST_TYPE_PTR ui32_type =
      ctx.Poly_gen().Get_type(VAR_TYPE_KIND::UINT32);
  if (ctx.Config().Lower_to_lpoly()) {
    // for targeting cpu, the value is calculate through rtlib call
    n_num_part = ctx.Poly_gen().New_poly_node(NUM_DECOMP, ui32_type, spos);
    n_num_part->Set_child(0, cntr->Clone_node_tree(n_c1));
  } else {
    // for targeting hpu, the value should be calculate through attribute
    // the logic canbe moved to constant folding
    n_num_part = cntr->New_intconst(ui32_type,
                                    ctx.Poly_gen().Get_num_decomp(node), spos);
  }

  // generate keyswitch dot product v_prod = DOT_PROD(n_precomp, n_swk)
  NODE_PTR n_prod_c0 = ctx.Poly_gen().New_dot_prod(
      ctx.Poly_gen().New_var_load(v_precomp, spos),
      ctx.Poly_gen().New_var_load(v_swk_c0, spos), n_num_part, spos);
  NODE_PTR n_prod_c1 =
      ctx.Poly_gen().New_dot_prod(ctx.Poly_gen().New_var_load(v_precomp, spos),
                                  ctx.Poly_gen().New_var_load(v_swk_c1, spos),
                                  cntr->Clone_node_tree(n_num_part), spos);

  // generate moddown: MOD_DOWN(n_prod_c0)
  //                   MOD_DOWN(n_prod_c1)
  NODE_PTR n_md_c0 = ctx.Poly_gen().New_mod_down(n_prod_c0, spos);
  NODE_PTR n_md_c1 = ctx.Poly_gen().New_mod_down(n_prod_c1, spos);

  // add_c0
  NODE_PTR n_add_c0 = ctx.Poly_gen().New_poly_add(n_c0, n_md_c0, spos);

  // rotate poly
  NODE_PTR n_rot_c0 = ctx.Poly_gen().New_poly_rotate(
      n_add_c0, cntr->Clone_node_tree(n_rot_idx), spos);
  NODE_PTR n_rot_c1 = ctx.Poly_gen().New_poly_rotate(
      n_md_c1, cntr->Clone_node_tree(n_rot_idx), spos);
  n_rot_c0->Copy_attr(node);
  n_rot_c1->Copy_attr(node);

  return POLY_LOWER_RETV(RETV_KIND::RK_CIPH_POLY, n_rot_c0, n_rot_c1);
}

POLY_LOWER_RETV CKKS2HPOLY::Expand_linear_transform(
    POLY_LOWER_CTX& ctx, NODE_PTR node, POLY_LOWER_RETV input_pair) {
  POLY_IR_GEN& pgen = ctx.Poly_gen();
  CONTAINER*   cntr = pgen.Container();
  GLOB_SCOPE*  glob = pgen.Glob_scope();
  const SPOS&  spos = node->Spos();

  CMPLR_ASSERT(node->Num_child() == 2 &&
                   node->Child(1)->Opcode() == air::core::OPC_LDC,
               "linear_transform coefficients must be one LDC array");
  CONSTANT_PTR coefficient = node->Child(1)->Const();
  CMPLR_ASSERT(coefficient != Null_ptr &&
                   coefficient->Kind() == CONSTANT_KIND::ARRAY &&
                   coefficient->Type()->Is_array(),
               "linear_transform coefficient descriptor is not an array");
  TYPE_PTR coefficient_element =
      coefficient->Type()->Cast_to_arr()->Elem_type();
  CMPLR_ASSERT(coefficient_element->Is_prim() &&
                   coefficient_element->Cast_to_prim()->Encoding() ==
                       PRIMITIVE_TYPE::FLOAT_64,
               "linear_transform coefficients must be interleaved f64");

  const uint32_t schema = *Linear_transform_scalar_attr(
      node, core::FHE_ATTR_KIND::LT_SCHEMA_VERSION);
  const uint32_t slots =
      *Linear_transform_scalar_attr(node, core::FHE_ATTR_KIND::LT_SLOTS);
  const uint32_t term_count = *Linear_transform_scalar_attr(
      node, core::FHE_ATTR_KIND::LT_TERM_COUNT);
  const uint32_t scale_degree = *Linear_transform_scalar_attr(
      node, core::FHE_ATTR_KIND::LT_SCALE_DEGREE);
  const uint32_t plain_level = *Linear_transform_scalar_attr(
      node, core::FHE_ATTR_KIND::LT_PLAIN_LEVEL);
  const uint32_t num_p =
      *Linear_transform_scalar_attr(node, core::FHE_ATTR_KIND::LT_NUM_P);
  const uint32_t encode_cache = *Linear_transform_scalar_attr(
      node, core::FHE_ATTR_KIND::LT_ENCODE_CACHE);
  uint32_t   rot_in_count  = 0;
  uint32_t   rot_out_count = 0;
  const int* rot_in = node->Attr<int>(core::FHE_ATTR_KIND::LT_ROT_IN,
                                      &rot_in_count);
  const int* rot_out = node->Attr<int>(core::FHE_ATTR_KIND::LT_ROT_OUT,
                                       &rot_out_count);

  CMPLR_ASSERT(schema == 1 && slots != 0 && term_count != 0 &&
                   scale_degree == 1 && encode_cache <= 1,
               "unsupported linear_transform v1 descriptor");
  CMPLR_ASSERT(rot_in != nullptr && rot_out != nullptr && rot_in_count != 0 &&
                   rot_out_count != 0 && rot_out[0] == 0,
               "linear_transform giant-step rotations must start with zero");
  const uint64_t schedule_capacity =
      static_cast<uint64_t>(rot_in_count) * rot_out_count;
  CMPLR_ASSERT(term_count <= schedule_capacity,
               "linear_transform term count exceeds BSGS schedule capacity");
  CMPLR_ASSERT(slots <=
                       ctx.Lower_ctx()->Get_ctx_param().Get_poly_degree() / 2,
               "linear_transform slot count exceeds ciphertext capacity");
  CMPLR_ASSERT(plain_level <=
                       ctx.Lower_ctx()->Get_ctx_param().Get_mul_level(),
               "linear_transform plaintext level is out of range");
  const uint32_t context_num_p =
      ctx.Lower_ctx()->Get_ctx_param().Get_p_prime_num();
  CMPLR_ASSERT(num_p == context_num_p,
               "linear_transform num_p %u does not match FHE context %u "
               "(mul_level=%u, q_parts=%u, q0_bits=%u, sf_bits=%u)",
               num_p, context_num_p,
               ctx.Lower_ctx()->Get_ctx_param().Get_mul_level(),
               ctx.Lower_ctx()->Get_ctx_param().Get_q_part_num(),
               ctx.Lower_ctx()->Get_ctx_param().Get_first_prime_bit_num(),
               ctx.Lower_ctx()->Get_ctx_param().Get_scaling_factor_bit_num());

  const uint64_t row_element_count = static_cast<uint64_t>(slots) * 2;
  CMPLR_ASSERT(term_count <=
                       std::numeric_limits<uint64_t>::max() /
                           row_element_count,
               "linear_transform coefficient element count overflows");
  const uint64_t coefficient_count =
      static_cast<uint64_t>(term_count) * row_element_count;
  CMPLR_ASSERT(coefficient_count <=
                       std::numeric_limits<size_t>::max() / sizeof(double) &&
                   coefficient->Array_byte_len() ==
                       coefficient_count * sizeof(double),
               "linear_transform coefficient payload has the wrong size");
  const double* coefficient_data = coefficient->Array_ptr<double>();

  // The RTL BSGS path hoists this precomputation across all baby rotations.
  auto materialize_precomp = [&](NODE_PTR source) {
    VAR v_precomp(ctx.Func_scope(), pgen.New_polys_var(spos));
    if (ctx.Config().Lower_to_lpoly()) {
      ctx.Prepend(pgen.New_var_store(pgen.New_alloc_for_precomp(source, spos),
                                    v_precomp, spos));
      source = cntr->Clone_node_tree(source);
    }
    ctx.Prepend(
        pgen.New_var_store(pgen.New_precomp(source, spos), v_precomp, spos));
    return v_precomp;
  };

  auto extended_key_switch = [&](CONST_VAR v_precomp, NODE_PTR source_c0,
                                 NODE_PTR source_c1, int rotation,
                                 bool add_first) {
    TYPE_PTR rot_type = glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
    NODE_PTR n_rotation = cntr->New_intconst(rot_type, rotation, spos);
    VAR      v_swk_c0(ctx.Func_scope(), pgen.New_swk_var(spos));
    VAR      v_swk_c1(ctx.Func_scope(), pgen.New_swk_var(spos));
    ctx.Prepend(pgen.New_var_store(
        pgen.New_swk_c0(true, spos, n_rotation), v_swk_c0, spos));
    ctx.Prepend(pgen.New_var_store(
        pgen.New_swk_c1(true, spos, cntr->Clone_node_tree(n_rotation)),
        v_swk_c1, spos));

    NODE_PTR n_num_part = Null_ptr;
    TYPE_PTR num_part_type = pgen.Get_type(VAR_TYPE_KIND::UINT32);
    if (ctx.Config().Lower_to_lpoly()) {
      n_num_part = pgen.New_poly_node(NUM_DECOMP, num_part_type, spos);
      n_num_part->Set_child(0, cntr->Clone_node_tree(source_c1));
    } else {
      const uint32_t num_decomp =
          ctx.Lower_ctx()->Get_ctx_param().Get_num_decomp(plain_level);
      n_num_part = cntr->New_intconst(num_part_type, num_decomp, spos);
    }
    NODE_PAIR dots = Materialize_poly_pair(
        ctx,
        pgen.New_dot_prod(pgen.New_var_load(v_precomp, spos),
                          pgen.New_var_load(v_swk_c0, spos), n_num_part, spos),
        pgen.New_dot_prod(pgen.New_var_load(v_precomp, spos),
                          pgen.New_var_load(v_swk_c1, spos),
                          cntr->Clone_node_tree(n_num_part), spos),
        spos);
    NODE_PTR switched_c0 = dots.first;
    if (add_first) {
      switched_c0 = Materialize_poly(
          ctx,
          pgen.New_poly_add_ext(source_c0,
                                cntr->Clone_node_tree(dots.first), spos),
          spos);
    }
    NODE_PAIR rotated = Materialize_poly_pair(
        ctx,
        pgen.New_poly_rotate(cntr->Clone_node_tree(switched_c0),
                             cntr->Clone_node_tree(n_rotation), spos),
        pgen.New_poly_rotate(cntr->Clone_node_tree(dots.second),
                             cntr->Clone_node_tree(n_rotation), spos),
        spos);
    if (add_first) {
      Free_materialized_poly(ctx, switched_c0, spos);
    }
    Free_materialized_poly_pair(ctx, dots, spos);
    return rotated;
  };

  VAR shared_precomp = materialize_precomp(
      cntr->Clone_node_tree(input_pair.Node2()));
  std::vector<NODE_PAIR> baby_rotations;
  baby_rotations.reserve(rot_in_count);
  const bool parallel_baby_rotations = rot_in_count > 1;
  auto prepend_marker = [&](air::base::OPCODE opcode) {
    ctx.Prepend(cntr->New_cust_stmt(opcode, spos));
  };
  if (parallel_baby_rotations) {
    prepend_marker(OPC_PARALLEL_SECTIONS_BEGIN);
  }
  for (uint32_t baby = 0; baby < rot_in_count; ++baby) {
    if (parallel_baby_rotations) {
      prepend_marker(OPC_PARALLEL_SECTION_BEGIN);
    }
    NODE_PAIR rotated;
    if (rot_in[baby] == 0) {
      rotated = Materialize_poly_pair(
          ctx,
          pgen.New_extend(cntr->Clone_node_tree(input_pair.Node1()), spos),
          pgen.New_extend(cntr->Clone_node_tree(input_pair.Node2()), spos),
          spos);
    } else {
      NODE_PTR extended_c0 = Materialize_poly(
          ctx,
          pgen.New_extend(cntr->Clone_node_tree(input_pair.Node1()), spos),
          spos);
      rotated = extended_key_switch(
          shared_precomp, extended_c0,
          cntr->Clone_node_tree(input_pair.Node2()), rot_in[baby], true);
      Free_materialized_poly(ctx, extended_c0, spos);
    }
    baby_rotations.push_back(rotated);
    if (parallel_baby_rotations) {
      prepend_marker(OPC_PARALLEL_SECTION_END);
    }
  }
  if (parallel_baby_rotations) {
    prepend_marker(OPC_PARALLEL_SECTIONS_END);
  }
  // All baby rotations are materialized before this point.  The shared
  // decomposition is large (Q parts times QP polynomials), so retaining it
  // until function exit—or leaking it across bootstrap calls—is prohibitive.
  ctx.Prepend(pgen.New_free_polys(shared_precomp, spos));

  TYPE_PTR row_type = glob->New_arr_type(
      coefficient_element, {static_cast<int64_t>(slots) * 2}, spos);
  TYPE_PTR u32_type = glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
  auto encode_term = [&](uint32_t term) {
    const size_t row_size = static_cast<size_t>(row_element_count);
    CONSTANT_PTR row = glob->New_const(
        CONSTANT_KIND::ARRAY, row_type,
        const_cast<double*>(coefficient_data + term * row_size),
        row_size * sizeof(double));
    NODE_PTR n_encode = pgen.New_encode(
        cntr->New_ldc(row, spos), cntr->New_intconst(u32_type, slots, spos),
        cntr->New_intconst(u32_type, scale_degree, spos),
        cntr->New_intconst(u32_type, plain_level, spos), spos);
    const uint32_t one = 1;
    n_encode->Set_attr(core::FHE_ATTR_KIND::ENCODE_DCMPLX, &one, 1);
    n_encode->Set_attr(core::FHE_ATTR_KIND::SCALE, &scale_degree, 1);
    n_encode->Set_attr(core::FHE_ATTR_KIND::LEVEL, &plain_level, 1);
    n_encode->Set_attr(core::FHE_ATTR_KIND::NUM_P, &num_p, 1);
    n_encode->Set_attr(core::FHE_ATTR_KIND::ENCODE_CACHE, &encode_cache, 1);
    VAR v_plain(ctx.Func_scope(), pgen.New_plain_var(spos));
    ctx.Prepend(pgen.New_var_store(n_encode, v_plain, spos));
    return v_plain;
  };

  std::vector<NODE_PAIR> rows;
  rows.reserve(rot_out_count);
  for (uint32_t giant = 0; giant < rot_out_count; ++giant) {
    VAR  v_row_c0(ctx.Func_scope(), pgen.New_poly_var(spos));
    VAR  v_row_c1(ctx.Func_scope(), pgen.New_poly_var(spos));
    bool row_initialized = false;
    for (uint32_t baby = 0; baby < rot_in_count; ++baby) {
      const uint64_t term = static_cast<uint64_t>(giant) * rot_in_count + baby;
      if (term >= term_count) break;
      VAR v_plain = encode_term(static_cast<uint32_t>(term));
      NODE_PTR next_c0;
      NODE_PTR next_c1;
      if (!row_initialized) {
        next_c0 = pgen.New_poly_mul_ext(
            cntr->Clone_node_tree(baby_rotations[baby].first),
            pgen.New_plain_poly_load(v_plain, false, spos), spos);
        next_c1 = pgen.New_poly_mul_ext(
            cntr->Clone_node_tree(baby_rotations[baby].second),
            pgen.New_plain_poly_load(v_plain, false, spos), spos);
      } else {
        // Keep multiply-accumulate semantic through LPOLY so the CPU runtime
        // can consume each Q/P limb once.  Reusing the row variables also
        // avoids allocating one full QP temporary for every diagonal.
        next_c0 = pgen.New_poly_mac(
            pgen.New_var_load(v_row_c0, spos),
            cntr->Clone_node_tree(baby_rotations[baby].first),
            pgen.New_plain_poly_load(v_plain, false, spos), true, spos);
        next_c1 = pgen.New_poly_mac(
            pgen.New_var_load(v_row_c1, spos),
            cntr->Clone_node_tree(baby_rotations[baby].second),
            pgen.New_plain_poly_load(v_plain, false, spos), true, spos);
      }
      ctx.Prepend(pgen.New_var_store(next_c0, v_row_c0, spos));
      ctx.Prepend(pgen.New_var_store(next_c1, v_row_c1, spos));
      row_initialized = true;
    }
    if (row_initialized) {
      rows.push_back({pgen.New_var_load(v_row_c0, spos),
                      pgen.New_var_load(v_row_c1, spos)});
    }
  }
  CMPLR_ASSERT(!rows.empty(), "linear_transform has no active BSGS rows");
  for (const NODE_PAIR& baby_rotation : baby_rotations) {
    Free_materialized_poly_pair(ctx, baby_rotation, spos);
  }

  NODE_PAIR outer = rows.front();
  for (uint32_t giant = 1; giant < rows.size(); ++giant) {
    NODE_PAIR rotated;
    NODE_PAIR previous_outer = outer;
    if (rot_out[giant] == 0) {
      rotated = rows[giant];
      outer = Materialize_poly_pair(
          ctx,
          pgen.New_poly_add_ext(cntr->Clone_node_tree(outer.first),
                                cntr->Clone_node_tree(rotated.first), spos),
          pgen.New_poly_add_ext(cntr->Clone_node_tree(outer.second),
                                cntr->Clone_node_tree(rotated.second), spos),
          spos);
    } else {
      VAR v_reduced_c1(ctx.Func_scope(), pgen.New_poly_var(spos));
      ctx.Prepend(pgen.New_var_store(
          pgen.New_mod_down(cntr->Clone_node_tree(rows[giant].second), spos),
          v_reduced_c1, spos));
      NODE_PTR reduced_c1 = pgen.New_var_load(v_reduced_c1, spos);
      VAR outer_precomp = materialize_precomp(
          cntr->Clone_node_tree(reduced_c1));
      rotated = extended_key_switch(
          outer_precomp, cntr->Clone_node_tree(rows[giant].first), reduced_c1,
          rot_out[giant], true);
      Free_materialized_poly(ctx, reduced_c1, spos);
      outer = Materialize_poly_pair(
          ctx,
          pgen.New_poly_add_ext(cntr->Clone_node_tree(outer.first),
                                cntr->Clone_node_tree(rotated.first), spos),
          pgen.New_poly_add_ext(cntr->Clone_node_tree(outer.second),
                                cntr->Clone_node_tree(rotated.second), spos),
          spos);
      ctx.Prepend(pgen.New_free_polys(outer_precomp, spos));
      Free_materialized_poly_pair(ctx, rotated, spos);
    }
    Free_materialized_poly_pair(ctx, previous_outer, spos);
    Free_materialized_poly_pair(ctx, rows[giant], spos);
  }

  return POLY_LOWER_RETV(
      RETV_KIND::RK_CIPH_POLY,
      pgen.New_mod_down(cntr->Clone_node_tree(outer.first), spos),
      pgen.New_mod_down(cntr->Clone_node_tree(outer.second), spos));
}

NODE_PTR CKKS2HPOLY::Gen_encode_float_from_ciph(POLY_LOWER_CTX& ctx,
                                                NODE_PTR node, CONST_VAR v_ciph,
                                                NODE_PTR n_cst, bool is_mul) {
  CONTAINER* cntr = ctx.Poly_gen().Container();
  SPOS       spos = n_cst->Spos();
  TYPE_PTR   t_ui32 =
      cntr->Glob_scope()->Prim_type(air::base::PRIMITIVE_TYPE::INT_U32);

  AIR_ASSERT(ctx.Lower_ctx()->Is_cipher_type(v_ciph.Type_id()));
  AIR_ASSERT(n_cst->Opcode() ==
                 air::base::OPCODE(air::core::CORE, air::core::OPCODE::LD) ||
             n_cst->Opcode() ==
                 air::base::OPCODE(air::core::CORE, air::core::OPCODE::LDC));

  NODE_PTR n_ciph = ctx.Poly_gen().New_var_load(v_ciph, spos);

  // child 1: const data node
  if (n_cst->Opcode() ==
      air::base::OPCODE(air::core::CORE, air::core::OPCODE::LD)) {
    n_cst = cntr->New_lda(n_cst->Addr_datum(), air::base::POINTER_KIND::FLAT32,
                          spos);
  } else {
    n_cst =
        cntr->New_ldca(n_cst->Const(), air::base::POINTER_KIND::FLAT32, spos);
  }
  // child 2: data len = 1
  NODE_PTR n_len = cntr->New_intconst(t_ui32, 1, spos);

  // child 3: encode scale degree
  NODE_PTR n_scale;
  NODE_PTR n_level;
  // for ciph.mul_const, encode const to degree 1
  // for ciph.add_const, encode const to v_ciph's degree
  if (is_mul) {
    n_scale = cntr->New_intconst(t_ui32, 1, spos);
  } else {
    n_scale = cntr->New_cust_node(fhe::ckks::OPC_SCALE, t_ui32, spos);
    n_scale->Set_child(0, n_ciph);
  }
  // child 4: encode level get from v_ciph
  n_level = cntr->New_cust_node(fhe::ckks::OPC_LEVEL, t_ui32, spos);
  n_level->Set_child(0, n_ciph);

  NODE_PTR n_enc =
      ctx.Poly_gen().New_encode(n_cst, n_len, n_scale, n_level, spos);
  n_enc->Copy_attr(node);
  return n_enc;
}

}  // namespace poly

}  // namespace fhe
