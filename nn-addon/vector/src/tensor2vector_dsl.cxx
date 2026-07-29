//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_dsl.h"

#include <utility>

#include "air/base/container.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "nn/vector/tensor2vector_ctx.h"

namespace nn {
namespace vector {

using namespace air::base;

namespace {

struct HELPER_BODY_STATS {
  uint32_t _calls = 0;
  uint32_t _retvs = 0;
};

void Validate_helper_spec(GLOB_SCOPE& glob,
                          const VECTOR_KERNEL_HELPER_SPEC& spec) {
  AIR_ASSERT_MSG(!spec._specialization_key.empty(),
                 "vector-kernel specialization key is empty");
  AIR_ASSERT_MSG(!spec._helper_name.empty(),
                 "vector-kernel helper name is empty");
  AIR_ASSERT(spec._result_type != Null_ptr);
  AIR_ASSERT(&spec._result_type->Glob_scope() == &glob);
  AIR_ASSERT(static_cast<bool>(spec._build_body));
  for (TYPE_PTR type : spec._formal_types) {
    AIR_ASSERT(type != Null_ptr);
    AIR_ASSERT(&type->Glob_scope() == &glob);
  }
}

void Validate_helper_signature(FUNC_SCOPE& helper,
                               const VECTOR_KERNEL_HELPER_SPEC& spec) {
  FUNC_PTR helper_func = helper.Owning_func();
  AIR_ASSERT(helper_func != Null_ptr);
  AIR_ASSERT_MSG(
      spec._helper_name == helper_func->Name()->Char_str(),
      "vector-kernel specialization key reused with a different helper name");
  AIR_ASSERT_MSG(
      helper.Formal_cnt() == spec._formal_types.size(),
      "vector-kernel specialization key reused with a different formal count");

  ENTRY_PTR entry = helper_func->Entry_point();
  AIR_ASSERT(entry != Null_ptr);
  SIGNATURE_TYPE_PTR signature = entry->Type()->Cast_to_sig();
  PARAM_PTR return_param = signature->Ret_param();
  AIR_ASSERT(return_param != Null_ptr);
  AIR_ASSERT_MSG(
      return_param->Type()->Is_compatible_type(spec._result_type),
      "vector-kernel specialization key reused with an incompatible result "
      "type");
  for (size_t idx = 0; idx < spec._formal_types.size(); ++idx) {
    AIR_ASSERT_MSG(
        helper.Formal(idx)->Type()->Is_compatible_type(
            spec._formal_types[idx]),
        "vector-kernel specialization key reused with an incompatible "
        "ordered formal type");
  }
}

void Validate_helper_name_is_available(
    GLOB_SCOPE& glob, const VECTOR_KERNEL_HELPER_SPEC& spec) {
  for (FUNC_ITER iter = glob.Begin_func(); iter != glob.End_func(); ++iter) {
    FUNC_PTR existing = *iter;
    AIR_ASSERT(existing != Null_ptr);
    if (spec._helper_name == existing->Name()->Char_str()) {
      AIR_ASSERT_MSG(false, "vector-kernel helper name collision");
    }
  }
}

const VECTOR_KERNEL_COMMON_PLAN& Common_plan(
    const PREPARED_VECTOR_KERNEL_PLAN& prepared) {
  return std::visit(
      [](const auto& typed_plan) -> const VECTOR_KERNEL_COMMON_PLAN& {
        return typed_plan._common;
      },
      prepared.Plan());
}

bool Matches_ranked_type(TYPE_PTR air_type,
                         const VECTOR_KERNEL_RANKED_TYPE_PLAN& plan_type) {
  if (air_type == Null_ptr || !air_type->Is_array()) return false;
  ARRAY_TYPE_PTR array = air_type->Cast_to_arr();
  TYPE_PTR element = array->Elem_type();
  return element->Is_prim() &&
         element->Cast_to_prim()->Encoding() == plan_type._element_type &&
         array->Shape() == plan_type._shape;
}

TYPE_PTR Find_or_create_destination_ranked_type(
    GLOB_SCOPE& destination, const VECTOR_KERNEL_RANKED_TYPE_PLAN& plan_type,
    const SPOS& spos) {
  for (TYPE_ITER iter = destination.Begin_type();
       iter != destination.End_type(); ++iter) {
    TYPE_PTR type = *iter;
    if (Matches_ranked_type(type, plan_type)) return type;
  }
  CONST_TYPE_PTR element = destination.Prim_type(plan_type._element_type);
  return destination.New_arr_type(element, plan_type._shape, spos);
}

VECTOR_KERNEL_DESTINATION_ABI Build_destination_abi(
    const PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const std::vector<NODE_PTR>& actuals, GLOB_SCOPE& destination,
    const SPOS& spos) {
  const VECTOR_KERNEL_COMMON_PLAN& common = Common_plan(prepared);
  const size_t vector_count = common._runtime_vector_inputs.size();
  const size_t scalar_count = common._runtime_scalar_inputs.size();
  AIR_ASSERT_MSG(actuals.size() == vector_count + scalar_count,
                 "prepared vector-kernel actual count does not match its ABI");

  VECTOR_KERNEL_DESTINATION_ABI abi;
  abi._formal_types.reserve(actuals.size());
  for (size_t idx = 0; idx < vector_count; ++idx) {
    NODE_PTR actual = actuals[idx];
    AIR_ASSERT(actual != Null_ptr && actual->Has_rtype());
    AIR_ASSERT(actual->Container()->Glob_scope() == &destination);
    AIR_ASSERT_MSG(Matches_ranked_type(actual->Rtype(),
                                       common._runtime_vector_inputs[idx]),
                   "prepared vector actual does not match its ABI");
    abi._formal_types.push_back(actual->Rtype());
  }
  for (size_t idx = 0; idx < scalar_count; ++idx) {
    NODE_PTR actual = actuals[vector_count + idx];
    AIR_ASSERT(actual != Null_ptr && actual->Has_rtype());
    TYPE_PTR formal = actual->Rtype();
    AIR_ASSERT_MSG(
        actual->Container()->Glob_scope() == &destination &&
            formal->Is_prim() &&
            formal->Cast_to_prim()->Encoding() ==
                common._runtime_scalar_inputs[idx],
        "prepared scalar actual does not match its ABI");
    abi._formal_types.push_back(formal);
  }
  abi._result_type = Find_or_create_destination_ranked_type(
      destination, common._result_type, spos);
  return abi;
}

void Validate_prepared_helper_signature(
    const PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const VECTOR_KERNEL_HELPER_SPEC& spec) {
  const VECTOR_KERNEL_COMMON_PLAN& common = Common_plan(prepared);
  const size_t vector_count = common._runtime_vector_inputs.size();
  const size_t scalar_count = common._runtime_scalar_inputs.size();
  AIR_ASSERT_MSG(spec._formal_types.size() == vector_count + scalar_count,
                 "DSL recipe does not implement the prepared helper ABI");
  for (size_t idx = 0; idx < vector_count; ++idx) {
    AIR_ASSERT_MSG(
        Matches_ranked_type(spec._formal_types[idx],
                            common._runtime_vector_inputs[idx]),
        "DSL recipe vector formal does not match the prepared helper ABI");
  }
  for (size_t idx = 0; idx < scalar_count; ++idx) {
    TYPE_PTR formal = spec._formal_types[vector_count + idx];
    AIR_ASSERT_MSG(
        formal != Null_ptr && formal->Is_prim() &&
            formal->Cast_to_prim()->Encoding() ==
                common._runtime_scalar_inputs[idx],
        "DSL recipe scalar formal does not match the prepared helper ABI");
  }
  AIR_ASSERT_MSG(Matches_ranked_type(spec._result_type, common._result_type),
                 "DSL recipe result does not match the prepared helper ABI");
}

void Validate_helper_node(NODE_PTR node, FUNC_SCOPE& helper,
                          HELPER_BODY_STATS& stats) {
  AIR_ASSERT(node != Null_ptr);
  AIR_ASSERT(node->Container() == &helper.Container());

  if (node->Has_rtype()) {
    AIR_ASSERT(&node->Rtype()->Glob_scope() == &helper.Glob_scope());
  }
  if (node->Has_access_type()) {
    AIR_ASSERT(&node->Access_type()->Glob_scope() == &helper.Glob_scope());
  }
  if (node->Has_sym()) {
    AIR_ASSERT(node->Addr_datum()->Defining_func_scope() == &helper);
  }
  if (node->Has_preg()) {
    AIR_ASSERT(node->Preg()->Defining_func_scope() == &helper);
  }
  if (node->Has_ret_var()) {
    AIR_ASSERT(node->Ret_preg()->Defining_func_scope() == &helper);
  }
  if (node->Is_do_loop()) {
    AIR_ASSERT(node->Iv()->Defining_func_scope() == &helper);
  }
  if (node->Has_const_id()) {
    AIR_ASSERT(&node->Const()->Glob_scope() == &helper.Glob_scope());
  }
  if (node->Has_entry()) {
    AIR_ASSERT(&node->Entry()->Glob_scope() == &helper.Glob_scope());
  }

  if (node->Is_call()) ++stats._calls;
  if (node->Opcode() == air::core::OPC_RETV) ++stats._retvs;

  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Validate_helper_node(stmt->Node(), helper, stats);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Validate_helper_node(node->Child(idx), helper, stats);
    }
  }
}

FUNC_SCOPE* Materialize_helper(GLOB_SCOPE& glob,
                               const VECTOR_KERNEL_HELPER_SPEC& spec,
                               const SPOS& spos) {
  Validate_helper_spec(glob, spec);

  STR_PTR  helper_name = glob.New_str(spec._helper_name.c_str());
  FUNC_PTR helper_func = glob.New_func(helper_name, spos);
  helper_func->Set_parent(glob.Comp_env_id());

  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(spec._result_type, signature);
  for (size_t idx = 0; idx < spec._formal_types.size(); ++idx) {
    const std::string formal_name = "packed_input_" + std::to_string(idx);
    glob.New_param(formal_name.c_str(), spec._formal_types[idx], signature,
                   spos);
  }
  signature->Set_complete();
  glob.New_entry_point(signature, helper_func, helper_name, spos);

  FUNC_SCOPE* helper_scope = &glob.New_func_scope(helper_func);
  CONTAINER&  helper_cntr  = helper_scope->Container();
  STMT_PTR    entry_stmt   = helper_cntr.New_func_entry(spos);
  NODE_PTR    body         = entry_stmt->Node()->Last_child();
  AIR_ASSERT(body != Null_ptr && body->Is_block());

  NODE_PTR result = spec._build_body(*helper_scope, body, spos);
  AIR_ASSERT_MSG(result != Null_ptr,
                 "vector-kernel body builder returned no result");
  AIR_ASSERT(result->Container() == &helper_cntr);
  AIR_ASSERT(result->Rtype()->Is_compatible_type(spec._result_type));

  HELPER_BODY_STATS before_return;
  Validate_helper_node(body, *helper_scope, before_return);
  AIR_ASSERT_MSG(before_return._calls == 0,
                 "Phase-A vector-kernel helper must be leaf");
  AIR_ASSERT_MSG(before_return._retvs == 0,
                 "body builder must leave the terminal RETV to M1");

  STMT_PTR retv = helper_cntr.New_retv(result, spos);
  const uint32_t generated_helper = 1;
  retv->Node()->Set_attr(
      VECTOR_KERNEL_GENERATED_HELPER_ATTR, &generated_helper, 1);
  STMT_LIST(body).Append(retv);

  HELPER_BODY_STATS completed;
  Validate_helper_node(body, *helper_scope, completed);
  AIR_ASSERT_MSG(completed._calls == 0,
                 "Phase-A vector-kernel helper must be leaf");
  AIR_ASSERT_MSG(completed._retvs == 1,
                 "vector-kernel helper must have exactly one RETV");

  STMT_PTR last_stmt = Null_ptr;
  for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
       stmt          = stmt->Next()) {
    last_stmt = stmt;
  }
  AIR_ASSERT(last_stmt != Null_ptr);
  AIR_ASSERT_MSG(last_stmt->Node()->Opcode() == air::core::OPC_RETV,
                 "vector-kernel helper RETV must be terminal");
  return helper_scope;
}

std::optional<VECTOR_KERNEL_LOWERING_RESULT> Materialize_helper_and_call(
    TENSOR2VECTOR_CTX& ctx, VECTOR_KERNEL_LOWERING_REGISTRY& registry,
    VECTOR_KERNEL_HELPER_SPEC spec, const std::vector<NODE_PTR>& actuals,
    const SPOS& spos) {
  CONTAINER*  caller_cntr  = ctx.Container();
  FUNC_SCOPE* caller_scope = ctx.Cur_func_scope();
  AIR_ASSERT(caller_cntr != nullptr && caller_scope != nullptr);
  AIR_ASSERT(&caller_scope->Container() == caller_cntr);
  GLOB_SCOPE* destination = caller_cntr->Glob_scope();
  AIR_ASSERT(destination != nullptr);

  Validate_helper_spec(*destination, spec);
  AIR_ASSERT_MSG(spec._formal_types.size() == actuals.size(),
                 "vector-kernel actual/formal count mismatch");
  for (size_t idx = 0; idx < actuals.size(); ++idx) {
    NODE_PTR actual = actuals[idx];
    TYPE_PTR formal = spec._formal_types[idx];
    AIR_ASSERT(actual != Null_ptr && formal != Null_ptr);
    AIR_ASSERT(actual->Container() == caller_cntr);
    AIR_ASSERT(&formal->Glob_scope() == destination);
    AIR_ASSERT_MSG(actual->Rtype()->Is_compatible_type(formal),
                   "vector-kernel actual/formal type mismatch");
  }

  FUNC_SCOPE* helper_scope =
      registry.Lookup_materialized_helper(*destination, spec);
  if (helper_scope == nullptr) {
    Validate_helper_name_is_available(*destination, spec);
    helper_scope = Materialize_helper(*destination, spec, spos);
    registry.Remember_materialized_helper(*destination, spec, *helper_scope);
  }
  AIR_ASSERT(&helper_scope->Glob_scope() == destination);
  AIR_ASSERT(helper_scope->Formal_cnt() == actuals.size());

  TYPE_PTR helper_result = helper_scope->Owning_func()
                               ->Entry_point()
                               ->Type()
                               ->Cast_to_sig()
                               ->Ret_param()
                               ->Type();
  AIR_ASSERT(helper_result->Is_compatible_type(spec._result_type));
  PREG_PTR result_preg = caller_scope->New_preg(helper_result);
  AIR_ASSERT(result_preg->Defining_func_scope() == caller_scope);

  STMT_PTR call = caller_cntr->New_call(
      helper_scope->Owning_func()->Entry_point(), result_preg, actuals.size(),
      spos);
  for (size_t idx = 0; idx < actuals.size(); ++idx) {
    AIR_ASSERT(actuals[idx]->Rtype()->Is_compatible_type(
        helper_scope->Formal(idx)->Type()));
    caller_cntr->New_arg(call, idx, actuals[idx]);
  }
  const uint32_t generated_call = 1;
  call->Node()->Set_attr(VECTOR_KERNEL_GENERATED_CALL_ATTR, &generated_call, 1);
  ctx.Prepend(call);

  NODE_PTR replacement = caller_cntr->New_ldp(result_preg, spos);
  AIR_ASSERT(replacement->Container() == caller_cntr);
  AIR_ASSERT(replacement->Preg()->Defining_func_scope() == caller_scope);
  return VECTOR_KERNEL_LOWERING_RESULT{helper_scope, call, replacement};
}

}  // namespace

bool VECTOR_KERNEL_LOWERING_REGISTRY::Register(
    OPCODE opcode, VECTOR_KERNEL_LOWERING_SELECTOR selector) {
  if (!selector) return false;
  return _selectors.emplace(static_cast<uint32_t>(opcode),
                            std::move(selector))
      .second;
}

bool VECTOR_KERNEL_LOWERING_REGISTRY::Unregister(OPCODE opcode) {
  return _selectors.erase(static_cast<uint32_t>(opcode)) != 0;
}

void VECTOR_KERNEL_LOWERING_REGISTRY::Clear() {
  _selectors.clear();
  _plan_recipes.clear();
  _materialized_helpers.clear();
}

bool VECTOR_KERNEL_LOWERING_REGISTRY::Has(OPCODE opcode) const {
  return _selectors.find(static_cast<uint32_t>(opcode)) != _selectors.end();
}

const VECTOR_KERNEL_LOWERING_SELECTOR*
VECTOR_KERNEL_LOWERING_REGISTRY::Lookup(OPCODE opcode) const {
  auto iter = _selectors.find(static_cast<uint32_t>(opcode));
  return iter == _selectors.end() ? nullptr : &iter->second;
}

bool VECTOR_KERNEL_LOWERING_REGISTRY::Register(
    VECTOR_KERNEL_PLAN_KIND kind, VECTOR_KERNEL_PLAN_RECIPE recipe) {
  if (!recipe) return false;
  return _plan_recipes
      .emplace(static_cast<uint8_t>(kind), std::move(recipe))
      .second;
}

bool VECTOR_KERNEL_LOWERING_REGISTRY::Unregister(
    VECTOR_KERNEL_PLAN_KIND kind) {
  return _plan_recipes.erase(static_cast<uint8_t>(kind)) != 0;
}

bool VECTOR_KERNEL_LOWERING_REGISTRY::Has(
    VECTOR_KERNEL_PLAN_KIND kind) const {
  return _plan_recipes.find(static_cast<uint8_t>(kind)) !=
         _plan_recipes.end();
}

const VECTOR_KERNEL_PLAN_RECIPE* VECTOR_KERNEL_LOWERING_REGISTRY::Lookup(
    VECTOR_KERNEL_PLAN_KIND kind) const {
  auto iter = _plan_recipes.find(static_cast<uint8_t>(kind));
  return iter == _plan_recipes.end() ? nullptr : &iter->second;
}

FUNC_SCOPE*
VECTOR_KERNEL_LOWERING_REGISTRY::Lookup_materialized_helper(
    GLOB_SCOPE& destination, const VECTOR_KERNEL_HELPER_SPEC& spec) const {
  auto destination_iter = _materialized_helpers.find(&destination);
  if (destination_iter == _materialized_helpers.end()) return nullptr;

  const DESTINATION_HELPER_CACHE& destination_cache =
      destination_iter->second;
  auto helper_iter = destination_cache.find(spec._specialization_key);
  if (helper_iter == destination_cache.end()) return nullptr;

  const MATERIALIZED_HELPER& cached = helper_iter->second;
  AIR_ASSERT_MSG(
      cached._helper_name == spec._helper_name,
      "vector-kernel specialization key reused with a different helper name");
  FUNC_SCOPE& helper =
      destination.Open_func_scope(cached._helper_func_id);
  Validate_helper_signature(helper, spec);
  return &helper;
}

void VECTOR_KERNEL_LOWERING_REGISTRY::Remember_materialized_helper(
    GLOB_SCOPE& destination, const VECTOR_KERNEL_HELPER_SPEC& spec,
    FUNC_SCOPE& helper) {
  AIR_ASSERT(&helper.Glob_scope() == &destination);
  Validate_helper_signature(helper, spec);
  DESTINATION_HELPER_CACHE& destination_cache =
      _materialized_helpers[&destination];
  auto inserted = destination_cache.emplace(
      spec._specialization_key,
      MATERIALIZED_HELPER{spec._helper_name, helper.Id()});
  AIR_ASSERT_MSG(inserted.second,
                 "vector-kernel specialization key cached more than once");
}

void VECTOR_KERNEL_LOWERING_REGISTRY::Clear_materialized_helpers() {
  _materialized_helpers.clear();
}

std::optional<VECTOR_KERNEL_LOWERING_RESULT>
Try_materialize_registered_vector_kernel(
    TENSOR2VECTOR_CTX& ctx, NODE_PTR source_node,
    const std::vector<NODE_PTR>& actuals) {
  AIR_ASSERT(source_node != Null_ptr);
  VECTOR_KERNEL_LOWERING_REGISTRY* registry =
      ctx.Vector_kernel_lowering_registry();
  if (registry == nullptr) return std::nullopt;

  const VECTOR_KERNEL_LOWERING_SELECTOR* selector =
      registry->Lookup(source_node->Opcode());
  if (selector == nullptr) return std::nullopt;

  CONTAINER* caller_cntr = ctx.Container();
  AIR_ASSERT(caller_cntr != nullptr);
  GLOB_SCOPE* destination = caller_cntr->Glob_scope();
  AIR_ASSERT(destination != nullptr);

  VECTOR_KERNEL_HELPER_SPEC spec =
      (*selector)(source_node, actuals, *destination);
  return Materialize_helper_and_call(ctx, *registry, std::move(spec), actuals,
                                     source_node->Spos());
}

std::optional<VECTOR_KERNEL_LOWERING_RESULT>
Try_materialize_prepared_vector_kernel(
    TENSOR2VECTOR_CTX& ctx, const PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const std::vector<NODE_PTR>& actuals, const SPOS& spos) {
  VECTOR_KERNEL_LOWERING_REGISTRY* registry =
      ctx.Vector_kernel_lowering_registry();
  if (registry == nullptr) return std::nullopt;

  const VECTOR_KERNEL_PLAN_KIND kind =
      Get_vector_kernel_plan_kind(prepared.Plan());
  const VECTOR_KERNEL_PLAN_RECIPE* recipe = registry->Lookup(kind);
  if (recipe == nullptr) return std::nullopt;

  CONTAINER* caller_cntr = ctx.Container();
  AIR_ASSERT(caller_cntr != nullptr);
  GLOB_SCOPE* destination = caller_cntr->Glob_scope();
  AIR_ASSERT(destination != nullptr);
  VECTOR_KERNEL_DESTINATION_ABI abi =
      Build_destination_abi(prepared, actuals, *destination, spos);
  VECTOR_KERNEL_HELPER_SPEC spec =
      (*recipe)(prepared, actuals, *destination, abi);
  AIR_ASSERT_MSG(spec._specialization_key.empty() ||
                     spec._specialization_key == prepared.Specialization_key(),
                 "DSL recipe changed the validated specialization key");
  AIR_ASSERT_MSG(spec._helper_name.empty() ||
                     spec._helper_name == prepared.Helper_name(),
                 "DSL recipe changed the validated helper name");
  spec._specialization_key = prepared.Specialization_key();
  spec._helper_name        = prepared.Helper_name();
  Validate_prepared_helper_signature(prepared, spec);
  return Materialize_helper_and_call(ctx, *registry, std::move(spec), actuals,
                                     spos);
}

}  // namespace vector
}  // namespace nn
