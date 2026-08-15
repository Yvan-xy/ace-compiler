//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================
#ifndef FHE_POLY_H2LPOLY_H
#define FHE_POLY_H2LPOLY_H

#include "air/base/transform_ctx.h"
#include "air/base/visitor.h"
#include "air/core/default_handler.h"
#include "air/core/handler.h"
#include "fhe/poly/default_handler.h"
#include "fhe/poly/handler.h"
#include "poly_ir_gen.h"
#include "poly_lower_ctx.h"

namespace fhe {
namespace poly {
class CORE2LPOLY;
class H2LPOLY;
using H2LPOLY_VISITOR =
    air::base::VISITOR<POLY_LOWER_CTX, air::core::HANDLER<CORE2LPOLY>,
                       fhe::poly::HANDLER<H2LPOLY>>;

class H2LPOLY : public fhe::poly::DEFAULT_HANDLER {
public:
  //! @brief Construct a new H2LPOLY object
  H2LPOLY() {}

  //! @brief Handle HPOLY_OPERATOR::ADD
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_add(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::ADD_EXT
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_add_ext(VISITOR* visitor,
                                 air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::SUB
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_sub(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::SUB_EXT
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_sub_ext(VISITOR* visitor,
                                 air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::MUL
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_mul(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::MUL_EXT
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_mul_ext(VISITOR* visitor,
                                 air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::MAC
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_mac(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::MAC_EXT
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_mac_ext(VISITOR* visitor,
                                 air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::ROTATE
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_rotate(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle HPOLY_OPERATOR::EXTEND
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_extend(VISITOR* visitor, air::base::NODE_PTR node);

  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_parallel_section_begin(
      VISITOR* visitor, air::base::NODE_PTR node) {
    visitor->Context().Poly_gen().Enter_parallel_section();
    return visitor->Context().template Handle_node<POLY_LOWER_RETV>(visitor,
                                                                    node);
  }

  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_parallel_section_end(
      VISITOR* visitor, air::base::NODE_PTR node) {
    POLY_LOWER_RETV retv =
        visitor->Context().template Handle_node<POLY_LOWER_RETV>(visitor,
                                                                 node);
    visitor->Context().Poly_gen().Leave_parallel_section();
    return retv;
  }

private:
  // Genernal function that process binary operations
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_binary_op(VISITOR* visitor, air::base::NODE_PTR node);

  // Preserve a three-input multiply-accumulate at the whole-QP runtime
  // boundary used by semantic LINEAR_TRANSFORM.
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_mac_op(VISITOR* visitor, air::base::NODE_PTR node);

  // Generate binary op for rns expanded polynomial
  air::base::STMT_PTR Gen_rns_binary_op(
      air::base::OPCODE op, POLY_LOWER_CTX& ctx, CONST_VAR v_res,
      CONST_VAR v_rns_idx, air::base::NODE_PTR opnd1, air::base::NODE_PTR opnd2,
      air::base::STMT_LIST& sl_blk, const air::base::SPOS& spos);

  // Expand hpoly operations to rns loops
  void Expand_op_to_rns(POLY_LOWER_CTX& ctx, air::base::NODE_PTR node,
                        CONST_VAR v_res, air::base::NODE_PTR opnd1,
                        air::base::NODE_PTR opnd2, bool is_ext,
                        air::base::STMT_LIST& sl);

  // Generate call to RNS-expanded functions and create a new function
  // if it does not already exist
  void Call_rns_func(POLY_LOWER_CTX& ctx, air::base::NODE_PTR node,
                     air::base::NODE_PTR n_opnd1, air::base::NODE_PTR n_opnd2,
                     bool is_ext);

  // Check if node contains "extended" attribute
  bool Has_ext_attr(POLY_LOWER_CTX& ctx, air::base::NODE_PTR node);
};

class CORE2LPOLY : public air::core::DEFAULT_HANDLER {
public:
  //! @brief Construct a new CKKS2HPOLY object
  CORE2LPOLY() {}

  //! @brief Handle CORE::STORE
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_st(VISITOR* visitor, air::base::NODE_PTR node);

  //! @brief Handle CORE::STP
  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_stp(VISITOR* visitor, air::base::NODE_PTR node);

  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_stf(VISITOR* visitor, air::base::NODE_PTR node);

  template <typename RETV, typename VISITOR>
  POLY_LOWER_RETV Handle_stpf(VISITOR* visitor, air::base::NODE_PTR node);

  template <typename VISITOR>
  POLY_LOWER_RETV Handle_st_var(VISITOR* visitor, air::base::NODE_PTR node,
                                CONST_VAR var);
};

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_add(VISITOR*            visitor,
                                    air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_add_ext(VISITOR*            visitor,
                                        air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_sub(VISITOR*            visitor,
                                    air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_sub_ext(VISITOR*            visitor,
                                        air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_mul(VISITOR*            visitor,
                                    air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_mul_ext(VISITOR*            visitor,
                                        air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_mac(VISITOR*            visitor,
                                    air::base::NODE_PTR node) {
  return Handle_mac_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_mac_ext(VISITOR*            visitor,
                                        air::base::NODE_PTR node) {
  return Handle_mac_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_mac_op(VISITOR*            visitor,
                                       air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx = visitor->Context();
  if (!ctx.Config().Linear_transform_only()) {
    if (node->Opcode() == OPC_MAC) {
      return fhe::poly::DEFAULT_HANDLER::template Handle_mac<RETV, VISITOR>(
          visitor, node);
    }
    return fhe::poly::DEFAULT_HANDLER::template Handle_mac_ext<RETV, VISITOR>(
        visitor, node);
  }

  CMPLR_ASSERT(node->Num_child() == 3, "invalid mac op node");
  POLY_IR_GEN&          pgen = ctx.Poly_gen();
  air::base::CONTAINER* cntr = pgen.Container();
  POLY_LOWER_RETV       children[3];
  for (uint32_t index = 0; index < 3; ++index) {
    children[index] =
        visitor->template Visit<RETV>(node->Child(index));
    CMPLR_ASSERT(!children[index].Is_null(), "null mac operand");
  }

  CONST_VAR& v_node = pgen.Node_var(node);
  ctx.Prepend(pgen.New_init_poly_by_opnd(
      v_node, node, node->Opcode() == OPC_MAC_EXT || Has_ext_attr(ctx, node),
      node->Spos()));
  air::base::NODE_PTR preserved = cntr->Clone_node(node);
  for (uint32_t index = 0; index < 3; ++index) {
    preserved->Set_child(index, children[index].Node());
  }
  return POLY_LOWER_RETV(preserved);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_rotate(VISITOR*            visitor,
                                       air::base::NODE_PTR node) {
  return Handle_binary_op<RETV>(visitor, node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_extend(VISITOR*            visitor,
                                       air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx  = visitor->Context();
  POLY_IR_GEN&    pgen = ctx.Poly_gen();
  // add init node to allocate memory
  CONST_VAR&          v_res = pgen.Node_var(node);
  air::base::STMT_PTR s_init =
      pgen.New_init_poly_by_opnd(v_res, node, true, node->Spos());
  ctx.Prepend(s_init);
  return ctx.Poly_gen().Container()->Clone_node_tree(node);
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV H2LPOLY::Handle_binary_op(VISITOR*            visitor,
                                          air::base::NODE_PTR node) {
  POLY_LOWER_CTX&       ctx  = visitor->Context();
  POLY_IR_GEN&          pgen = ctx.Poly_gen();
  air::base::CONTAINER* cntr = pgen.Container();
  air::base::SPOS       spos = node->Spos();
  CMPLR_ASSERT(node->Num_child() == 2, "invalid binary op node");

  // visit two child node
  air::base::NODE_PTR n0      = node->Child(0);
  air::base::NODE_PTR n1      = node->Child(1);
  POLY_LOWER_RETV     n0_retv = visitor->template Visit<RETV>(n0);
  POLY_LOWER_RETV     n1_retv = visitor->template Visit<RETV>(n1);
  CMPLR_ASSERT((!n0_retv.Is_null() && !n1_retv.Is_null()), "null node");
  CONST_VAR& v_node = pgen.Node_var(node);

  // Add init poly
  const bool is_ext_opcode =
      node->Opcode() == OPC_ADD_EXT || node->Opcode() == OPC_SUB_EXT ||
      node->Opcode() == OPC_MUL_EXT;
  bool is_ext = is_ext_opcode || Has_ext_attr(ctx, node);
  air::base::STMT_PTR s_init =
      pgen.New_init_poly_by_opnd(v_node, node, is_ext, node->Spos());
  ctx.Prepend(s_init);

  // LINEAR_TRANSFORM deliberately keeps QP scheduling visible at POLY, but
  // the final CPU boundary should use ANT's contiguous whole-polynomial
  // kernels.  Expanding these operations into one Hw_* call per RNS limb
  // creates thousands of tiny calls and is substantially slower than RTL.
  // Run_flatten has already materialized each operation as a store RHS, so it
  // is safe to preserve this one node for POLY IR2C.
  if (ctx.Config().Linear_transform_only() &&
      (is_ext || node->Opcode() == OPC_ROTATE)) {
    air::base::NODE_PTR preserved = cntr->Clone_node(node);
    preserved->Set_child(0, n0_retv.Node());
    preserved->Set_child(1, n1_retv.Node());
    return POLY_LOWER_RETV(preserved);
  }

  if (ctx.Config().Inline_rns()) {
    air::base::STMT_LIST sl = air::base::STMT_LIST::Enclosing_list(s_init);
    Expand_op_to_rns(ctx, node, v_node, n0_retv.Node(), n1_retv.Node(), is_ext,
                     sl);
  } else {
    Call_rns_func(ctx, node, n0_retv.Node(), n1_retv.Node(), is_ext);
  }
  return POLY_LOWER_RETV();
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV CORE2LPOLY::Handle_st(VISITOR*            visitor,
                                      air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx = visitor->Context();
  if (!(node->Child(0)->Is_ld())) {
    ctx.Poly_gen().Add_node_var(node->Child(0), node->Addr_datum());
  }
  return Handle_st_var(visitor, node,
                       VAR(ctx.Func_scope(), node->Addr_datum()));
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV CORE2LPOLY::Handle_stp(VISITOR*            visitor,
                                       air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx = visitor->Context();
  if (!(node->Child(0)->Is_ld())) {
    ctx.Poly_gen().Add_node_var(node->Child(0), node->Preg());
  }
  return Handle_st_var(visitor, node, VAR(ctx.Func_scope(), node->Preg()));
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV CORE2LPOLY::Handle_stf(VISITOR*            visitor,
                                       air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx = visitor->Context();
  if (!(node->Child(0)->Is_ld())) {
    ctx.Poly_gen().Add_node_var(node->Child(0), node->Addr_datum(),
                                node->Field());
  }
  return Handle_st_var(
      visitor, node, VAR(ctx.Func_scope(), node->Addr_datum(), node->Field()));
}

template <typename RETV, typename VISITOR>
POLY_LOWER_RETV CORE2LPOLY::Handle_stpf(VISITOR*            visitor,
                                        air::base::NODE_PTR node) {
  POLY_LOWER_CTX& ctx = visitor->Context();
  if (!(node->Child(0)->Is_ld())) {
    ctx.Poly_gen().Add_node_var(node->Child(0), node->Preg(), node->Field());
  }
  return Handle_st_var(visitor, node,
                       VAR(ctx.Func_scope(), node->Preg(), node->Field()));
}

template <typename VISITOR>
POLY_LOWER_RETV CORE2LPOLY::Handle_st_var(VISITOR*            visitor,
                                          air::base::NODE_PTR node,
                                          CONST_VAR           var) {
  air::base::TYPE_ID    tid       = var.Type_id();
  POLY_LOWER_CTX&       ctx       = visitor->Context();
  fhe::core::LOWER_CTX* lower_ctx = ctx.Lower_ctx();
  if (!ctx.Lower_to_lpoly(node->Child(0))) {
    // clone the tree for
    // 1) store var type is not ciph/plain related
    // 2) load/ldp for ciph/plain, keep store(res, ciph)
    //    do not lower to store polys for now
    air::base::STMT_PTR s_new =
        ctx.Poly_gen().Container()->Clone_stmt_tree(node->Stmt());
    return POLY_LOWER_RETV(s_new->Node());
  } else {
    POLY_LOWER_RETV       retv;
    air::base::CONTAINER* cntr = ctx.Container();
    POLY_LOWER_RETV       rhs =
        visitor->template Visit<POLY_LOWER_RETV>(node->Child(0));
    if (!rhs.Is_null()) {
      air::base::STMT_PTR s_new =
          ctx.Poly_gen().Container()->Clone_stmt(node->Stmt());
      s_new->Node()->Set_child(0, rhs.Node());
      return POLY_LOWER_RETV(s_new->Node());
    } else {
      return POLY_LOWER_RETV();
    }
  }
}

}  // namespace poly
}  // namespace fhe
#endif
