//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "air_function_inliner.h"

#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "air/base/container.h"
#include "air/base/meta_info.h"
#include "air/base/node.h"
#include "air/base/st_attr.h"
#include "air/base/st_const.h"
#include "air/base/st_iter.h"
#include "air/core/opcode.h"

namespace ace::bindings {
namespace {

// This file intentionally implements only the temporary binding-side path
// described in air_function_inliner.h; it does not close M13.

using namespace air::base;

struct CALL_SITE {
  FUNC_SCOPE* _caller = nullptr;
  FUNC_SCOPE* _helper = nullptr;
  STMT_PTR    _call   = Null_ptr;
};

using FUNC_SCOPE_MAP = std::unordered_map<uint64_t, FUNC_SCOPE*>;

ATTR_PTR Find_attribute(NODE_PTR node, const char* key) {
  if (!META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) return Null_ptr;
  for (ATTR_ITER iter = node->Begin_attr(); iter != node->End_attr(); ++iter) {
    ATTR_PTR attr = *iter;
    if (std::strcmp(attr->Key(), key) == 0) return attr;
  }
  return Null_ptr;
}

bool Is_generated_call(NODE_PTR node, const char* attribute,
                       std::string* diagnostic) {
  ATTR_PTR attr = Find_attribute(node, attribute);
  if (attr == Null_ptr) return false;
  if (node->Opcode() != air::core::OPC_CALL ||
      attr->Type() != PRIMITIVE_TYPE::INT_U32 ||
      attr->Count() != 1 || attr->Value().size() != sizeof(uint32_t)) {
    *diagnostic = "generated-helper marker is malformed or is not on a "
                  "direct CALL";
    return false;
  }
  uint32_t value = 0;
  std::memcpy(&value, attr->Value().data(), sizeof(value));
  if (value != 1) {
    *diagnostic = "generated-helper marker must have value one";
    return false;
  }
  return true;
}

void Collect_calls_in_block(FUNC_SCOPE& caller, NODE_PTR block,
                            const FUNC_SCOPE_MAP& functions,
                            const char* attribute,
                            std::vector<CALL_SITE>* sites,
                            std::string* diagnostic) {
  for (STMT_PTR stmt = block->Begin_stmt();
       stmt != block->End_stmt() && diagnostic->empty(); stmt = stmt->Next()) {
    NODE_PTR node = stmt->Node();
    ATTR_PTR marker = Find_attribute(node, attribute);
    if (marker != Null_ptr) {
      if (!Is_generated_call(node, attribute, diagnostic)) return;
      auto found = functions.find(node->Entry()->Owning_func_id().Value());
      if (found == functions.end()) {
        *diagnostic = "generated-helper CALL targets no defined same-module "
                      "function";
        return;
      }
      sites->push_back(CALL_SITE{&caller, found->second, stmt});
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      NODE_PTR child = node->Child(idx);
      if (child->Is_block()) {
        Collect_calls_in_block(caller, child, functions, attribute, sites,
                               diagnostic);
        if (!diagnostic->empty()) return;
      }
    }
  }
}

size_t Attribute_element_size(PRIMITIVE_TYPE type) {
  switch (type) {
    case PRIMITIVE_TYPE::INT_S8: return sizeof(char);
    case PRIMITIVE_TYPE::INT_S16: return sizeof(int16_t);
    case PRIMITIVE_TYPE::INT_S32: return sizeof(int32_t);
    case PRIMITIVE_TYPE::INT_S64: return sizeof(int64_t);
    case PRIMITIVE_TYPE::INT_U8: return sizeof(unsigned char);
    case PRIMITIVE_TYPE::INT_U16: return sizeof(uint16_t);
    case PRIMITIVE_TYPE::INT_U32: return sizeof(uint32_t);
    case PRIMITIVE_TYPE::INT_U64: return sizeof(uint64_t);
    case PRIMITIVE_TYPE::FLOAT_32: return sizeof(float);
    case PRIMITIVE_TYPE::FLOAT_64: return sizeof(double);
    default: return 0;
  }
}

bool Attribute_is_cloneable(ATTR_PTR attr) {
  if (attr->Count() == 0) {
    return attr->Type() == PRIMITIVE_TYPE::INT_S8;
  }
  const size_t element_size = Attribute_element_size(attr->Type());
  return element_size != 0 &&
         attr->Value().size() == element_size * attr->Count();
}

enum class CLONE_USE {
  VALUE,
  INDIRECT_LOAD_ADDRESS,
  CONSTANT_ARRAY_BASE,
};

bool Validate_cloneable_node(NODE_PTR node, FUNC_SCOPE& helper,
                             bool allow_block, CLONE_USE use,
                             std::string* diagnostic) {
  if (node == Null_ptr || node->Container() != &helper.Container()) {
    *diagnostic = "helper contains a foreign or null AIR node";
    return false;
  }
  if (node->Is_block()) {
    if (!allow_block) {
      *diagnostic = "helper contains an unexpected statement block";
      return false;
    }
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt = stmt->Next()) {
      if (stmt->Node()->Is_ret()) {
        *diagnostic = "helper contains a nested or nonterminal return";
        return false;
      }
      if (!Validate_cloneable_node(stmt->Node(), helper, true,
                                   CLONE_USE::VALUE, diagnostic))
        return false;
    }
    return true;
  }

  const OPCODE opcode = node->Opcode();
  if (node->Is_entry() || node->Is_call() || node->Is_intrn_call() ||
      node->Is_intrn_op() ||
      (node->Has_added_chld() && opcode != air::core::OPC_ARRAY) ||
      opcode == air::core::OPC_LDA ||
      (opcode == air::core::OPC_LDCA &&
       use != CLONE_USE::CONSTANT_ARRAY_BASE)) {
    *diagnostic = "helper is not a supported leaf or has an escaping address";
    return false;
  }
  if (opcode == air::core::OPC_ARRAY) {
    NODE_PTR base = node->Array_base();
    if (use != CLONE_USE::INDIRECT_LOAD_ADDRESS || node->Array_dim() == 0 ||
        base->Opcode() != air::core::OPC_LDCA ||
        !base->Const()->Type()->Is_array() ||
        base->Const()->Type()->Cast_to_arr()->Dim() != node->Array_dim() ||
        !node->Rtype()->Is_ptr() ||
        node->Rtype()->Cast_to_ptr()->Domain_type_id().Is_null() ||
        node->Rtype()->Cast_to_ptr()->Ptr_kind() != POINTER_KIND::FLAT32 ||
        !node->Rtype()->Cast_to_ptr()->Domain_type()->Base_type()->
            Is_compatible_type(
                base->Const()->Type()->Cast_to_arr()->Elem_type()->Base_type())) {
      *diagnostic =
          "helper contains an unsupported or escaping array address";
      return false;
    }
    for (uint32_t dim = 0; dim < node->Array_dim(); ++dim) {
      if (!node->Array_idx(dim)->Rtype()->Is_signed_int()) {
        *diagnostic = "helper constant-array index must be a signed integer";
        return false;
      }
    }
  }
  if (opcode == air::core::OPC_ILD && node->Num_child() == 1 &&
      node->Child(0)->Opcode() == air::core::OPC_ARRAY) {
    if (!node->Child(0)->Rtype()->Is_ptr() ||
        node->Child(0)->Rtype()->Cast_to_ptr()->Domain_type_id().Is_null()) {
      *diagnostic =
          "helper contains an unsupported or escaping array address";
      return false;
    }
    TYPE_PTR element =
        node->Child(0)->Rtype()->Cast_to_ptr()->Domain_type()->Base_type();
    if (!node->Has_access_type() ||
        !node->Access_type()->Is_compatible_type(element) ||
        !node->Rtype()->Is_compatible_type(element)) {
      *diagnostic =
          "helper constant-array load has incompatible element types";
      return false;
    }
  }
  if (META_INFO::Has_prop<OPR_PROP::ENTRY>(opcode) ||
      META_INFO::Has_prop<OPR_PROP::RET_VAR>(opcode) ||
      META_INFO::Has_prop<OPR_PROP::LABEL>(opcode) ||
      META_INFO::Has_prop<OPR_PROP::BARRIER>(opcode) ||
      META_INFO::Has_prop<OPR_PROP::FLAGS>(opcode)) {
    *diagnostic = "helper uses unsupported control-flow or call metadata";
    return false;
  }
  if (META_INFO::Has_prop<OPR_PROP::VALUE>(opcode) &&
      opcode != air::core::OPC_INTCONST) {
    *diagnostic = "helper uses an unsupported literal node";
    return false;
  }
  if (node->Has_rtype() && &node->Rtype()->Glob_scope() != &helper.Glob_scope()) {
    *diagnostic = "helper result type belongs to another GLOB_SCOPE";
    return false;
  }
  if (node->Has_sym()) {
    ADDR_DATUM_PTR datum = node->Addr_datum();
    if (datum->Scope_level() != 0 &&
        datum->Defining_func_scope() != &helper) {
      *diagnostic = "helper references another function's local symbol";
      return false;
    }
  }
  if (node->Has_preg() && node->Preg()->Defining_func_scope() != &helper) {
    *diagnostic = "helper references another function's preg";
    return false;
  }
  if (node->Is_do_loop()) {
    if (node->Iv()->Defining_func_scope() != &helper) {
      *diagnostic = "helper loop IV belongs to another function";
      return false;
    }
  }
  if (node->Has_const_id() &&
      &node->Const()->Glob_scope() != &helper.Glob_scope()) {
    *diagnostic = "helper references a constant from another GLOB_SCOPE";
    return false;
  }
  if (node->Has_access_type() &&
      &node->Access_type()->Glob_scope() != &helper.Glob_scope()) {
    *diagnostic = "helper access type belongs to another GLOB_SCOPE";
    return false;
  }
  if (node->Has_fld() && &node->Field()->Glob_scope() != &helper.Glob_scope()) {
    *diagnostic = "helper field belongs to another GLOB_SCOPE";
    return false;
  }
  if (META_INFO::Has_prop<OPR_PROP::ATTR>(opcode)) {
    for (ATTR_ITER iter = node->Begin_attr(); iter != node->End_attr(); ++iter) {
      if (!Attribute_is_cloneable(*iter)) {
        *diagnostic = "helper carries an unsupported attribute type";
        return false;
      }
    }
  }
  for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
    CLONE_USE child_use = CLONE_USE::VALUE;
    if (opcode == air::core::OPC_ILD && idx == 0) {
      child_use = CLONE_USE::INDIRECT_LOAD_ADDRESS;
    } else if (opcode == air::core::OPC_ARRAY && idx == 0) {
      child_use = CLONE_USE::CONSTANT_ARRAY_BASE;
    }
    if (!Validate_cloneable_node(node->Child(idx), helper, true, child_use,
                                 diagnostic))
      return false;
  }
  return true;
}

bool Validate_call_site(const CALL_SITE& site, const char* helper_attribute,
                        std::string* diagnostic) {
  NODE_PTR call = site._call->Node();
  if (site._caller == site._helper) {
    *diagnostic = "generated helper must be nonrecursive";
    return false;
  }
  if (call->Num_arg() != site._helper->Formal_cnt()) {
    *diagnostic = "generated-helper actual/formal count mismatch";
    return false;
  }
  if (call->Ret_preg_id().Is_null()) {
    *diagnostic = "generated-helper CALL has no result preg";
    return false;
  }
  FUNC_PTR helper_func = site._helper->Owning_func();
  ENTRY_PTR helper_entry = helper_func->Entry_point();
  if (call->Entry()->Id() != helper_entry->Id() ||
      !helper_entry->Type()->Is_signature()) {
    *diagnostic =
        "generated helper must have a matching signature entry point";
    return false;
  }
  if (helper_func->Is_nested_func()) {
    *diagnostic = "generated helper must not be a nested function";
    return false;
  }
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = site._helper->Glob_scope().Begin_func_scope();
       iter != site._helper->Glob_scope().End_func_scope(); ++iter) {
    FUNC_PTR function = (*iter).Owning_func();
    if (function->Is_nested_func() &&
        function->Parent_func_id() == helper_func->Id()) {
      *diagnostic = "generated helper must not own nested functions";
      return false;
    }
  }

  uint32_t helper_entry_count = 0;
  for (ENTRY_ITER iter = site._helper->Glob_scope().Begin_entry();
       iter != site._helper->Glob_scope().End_entry(); ++iter) {
    ENTRY_PTR entry = *iter;
    if (entry->Owning_func_id() != helper_func->Id()) continue;
    ++helper_entry_count;
    if (entry->Id() != helper_entry->Id()) {
      *diagnostic = "generated helper has an additional entry point";
      return false;
    }
  }
  if (helper_entry_count != 1 || helper_entry->Is_program_entry()) {
    *diagnostic =
        "generated helper must have one non-program entry point";
    return false;
  }
  for (CONSTANT_ITER iter = site._helper->Glob_scope().Begin_const();
       iter != site._helper->Glob_scope().End_const(); ++iter) {
    CONSTANT_PTR constant = *iter;
    ENTRY_PTR referenced_entry = Null_ptr;
    if (constant->Kind() == CONSTANT_KIND::ENTRY_PTR) {
      referenced_entry = constant->Entry();
    } else if (constant->Kind() == CONSTANT_KIND::ENTRY_FUNC_DESC) {
      referenced_entry = constant->Func_desc_entry();
    }
    if (referenced_entry != Null_ptr &&
        referenced_entry->Owning_func_id() == helper_func->Id()) {
      *diagnostic = "generated helper entry address escapes through a constant";
      return false;
    }
  }
  for (uint32_t idx = 0; idx < call->Num_arg(); ++idx) {
    NODE_PTR actual = call->Child(idx);
    if (actual->Container() != &site._caller->Container() ||
        !actual->Has_rtype() ||
        !actual->Rtype()->Is_compatible_type(site._helper->Formal(idx)->Type())) {
      *diagnostic = "generated-helper actual/formal type or ownership mismatch";
      return false;
    }
  }

  NODE_PTR entry = site._helper->Container().Entry_node();
  if (entry == Null_ptr || !entry->Is_entry() || !entry->Body_blk()->Is_block()) {
    *diagnostic = "generated helper has no valid function body";
    return false;
  }
  NODE_PTR body = entry->Body_blk();
  STMT_PTR terminal = Null_ptr;
  uint32_t returns = 0;
  for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
       stmt = stmt->Next()) {
    NODE_PTR node = stmt->Node();
    if (node->Is_ret()) {
      ++returns;
      terminal = stmt;
      if (node->Opcode() != air::core::OPC_RETV || node->Num_child() != 1) {
        *diagnostic = "generated helper must return exactly one value";
        return false;
      }
      if (!Validate_cloneable_node(node->Child(0), *site._helper, false,
                                   CLONE_USE::VALUE, diagnostic))
        return false;
      continue;
    }
    if (!Validate_cloneable_node(node, *site._helper, true,
                                 CLONE_USE::VALUE, diagnostic))
      return false;
  }
  if (returns != 1 || terminal == Null_ptr ||
      terminal->Next() != body->End_stmt()) {
    *diagnostic = "generated helper must have one terminal RETV";
    return false;
  }
  ATTR_PTR helper_marker =
      Find_attribute(terminal->Node(), helper_attribute);
  if (helper_marker == Null_ptr ||
      helper_marker->Type() != PRIMITIVE_TYPE::INT_U32 ||
      helper_marker->Count() != 1 ||
      helper_marker->Value().size() != sizeof(uint32_t)) {
    *diagnostic = "generated helper RETV marker is missing or malformed";
    return false;
  }
  uint32_t helper_marker_value = 0;
  std::memcpy(&helper_marker_value, helper_marker->Value().data(),
              sizeof(helper_marker_value));
  if (helper_marker_value != 1) {
    *diagnostic = "generated helper RETV marker must have value one";
    return false;
  }
  if (!call->Ret_preg()->Type()->Is_compatible_type(
          terminal->Node()->Child(0)->Rtype())) {
    *diagnostic = "generated-helper return type does not match its call preg";
    return false;
  }
  return true;
}

template <typename T>
void Copy_typed_attribute(NODE_PTR destination, ATTR_PTR source) {
  destination->Set_attr<T>(
      source->Key(), reinterpret_cast<const T*>(source->Value().data()),
      source->Count());
}

void Copy_attributes(NODE_PTR destination, NODE_PTR source) {
  if (!META_INFO::Has_prop<OPR_PROP::ATTR>(source->Opcode())) return;
  for (ATTR_ITER iter = source->Begin_attr(); iter != source->End_attr(); ++iter) {
    ATTR_PTR attr = *iter;
    if (attr->Count() == 0) {
      std::string value(attr->Value());
      destination->Set_attr(attr->Key(), value.c_str());
      continue;
    }
    switch (attr->Type()) {
      case PRIMITIVE_TYPE::INT_S8:
        Copy_typed_attribute<char>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_S16:
        Copy_typed_attribute<int16_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_S32:
        Copy_typed_attribute<int32_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_S64:
        Copy_typed_attribute<int64_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_U8:
        Copy_typed_attribute<unsigned char>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_U16:
        Copy_typed_attribute<uint16_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_U32:
        Copy_typed_attribute<uint32_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::INT_U64:
        Copy_typed_attribute<uint64_t>(destination, attr);
        break;
      case PRIMITIVE_TYPE::FLOAT_32:
        Copy_typed_attribute<float>(destination, attr);
        break;
      case PRIMITIVE_TYPE::FLOAT_64:
        Copy_typed_attribute<double>(destination, attr);
        break;
      default:
        AIR_ASSERT_MSG(false, "preflight accepted unsupported attribute");
    }
  }
}

class SITE_CLONER {
public:
  explicit SITE_CLONER(const CALL_SITE& site)
      : _site(site),
        _caller(*site._caller),
        _helper(*site._helper),
        _container(site._caller->Container()) {}

  void Inline() {
    NODE_PTR call = _site._call->Node();
    const std::string prefix =
        "__ace_inline_" + std::to_string(_site._call->Id().Value()) + "_";

    NODE_PTR inline_block = _container.New_stmt_block(call->Spos());
    STMT_LIST inline_list(inline_block);

    for (uint32_t idx = 0; idx < call->Num_arg(); ++idx) {
      ADDR_DATUM_PTR formal = _helper.Formal(idx);
      std::string name = prefix + "arg_" + std::to_string(idx);
      ADDR_DATUM_PTR local =
          _caller.New_var(formal->Type(), name.c_str(), call->Spos());
      _datums.emplace(formal->Id().Value(), local);
      NODE_PTR actual = _container.Clone_node_tree(call->Child(idx));
      inline_list.Append(_container.New_st(actual, local, call->Spos()));
    }

    for (VAR_ITER iter = _helper.Begin_var();
         iter != _helper.End_var(); ++iter) {
      ADDR_DATUM_PTR source = *iter;
      std::string name = prefix + source->Name()->Char_str();
      ADDR_DATUM_PTR local =
          _caller.New_var(source->Type(), name.c_str(), source->Spos());
      _datums.emplace(source->Id().Value(), local);
    }
    for (PREG_ITER iter = _helper.Begin_preg();
         iter != _helper.End_preg(); ++iter) {
      PREG_PTR source = *iter;
      _pregs.emplace(source->Id().Value(), _caller.New_preg(source->Type()));
    }

    NODE_PTR helper_body = _helper.Container().Entry_node()->Body_blk();
    for (STMT_PTR stmt = helper_body->Begin_stmt();
         stmt != helper_body->End_stmt(); stmt = stmt->Next()) {
      if (stmt->Node()->Opcode() == air::core::OPC_RETV) {
        NODE_PTR result = Clone_expression(stmt->Node()->Child(0));
        inline_list.Append(
            _container.New_stp(result, call->Ret_preg(), stmt->Node()->Spos()));
      } else {
        inline_list.Append(Clone_statement(stmt));
      }
    }

    STMT_LIST caller_list = STMT_LIST::Enclosing_list(_site._call);
    while (!inline_list.Is_empty()) {
      STMT_PTR next = inline_list.Begin_stmt();
      inline_list.Remove(next);
      caller_list.Prepend(_site._call, next);
    }
    caller_list.Remove(_site._call);
  }

private:
  ADDR_DATUM_PTR Remap_datum(ADDR_DATUM_PTR source) const {
    if (source->Scope_level() == 0) return source;
    auto found = _datums.find(source->Id().Value());
    AIR_ASSERT(found != _datums.end());
    return found->second;
  }

  PREG_PTR Remap_preg(PREG_PTR source) const {
    auto found = _pregs.find(source->Id().Value());
    AIR_ASSERT(found != _pregs.end());
    return found->second;
  }

  void Copy_fields(NODE_PTR destination, NODE_PTR source) {
    if (source->Has_sym())
      destination->Set_addr_datum(Remap_datum(source->Addr_datum()));
    if (source->Has_preg())
      destination->Set_preg(Remap_preg(source->Preg()));
    if (source->Has_const_id()) destination->Set_const(source->Const());
    if (source->Has_access_type())
      destination->Set_access_type(source->Access_type());
    if (source->Has_fld()) destination->Set_field(source->Field());
    if (source->Has_ofst()) destination->Set_ofst(source->Ofst());
    if (source->Opcode() == air::core::OPC_INTCONST)
      destination->Set_intconst(source->Intconst());
    if (source->Is_do_loop()) destination->Set_iv(Remap_datum(source->Iv()));
    if (source->Is_comment()) {
      destination->Set_comment(
          _caller.Glob_scope().New_str(source->Comment())->Id());
    }
    if (source->Is_pragma()) {
      destination->Set_pragma(source->Pragma_id(), source->Pragma_arg0(),
                              source->Pragma_arg1());
    }
    Copy_attributes(destination, source);
  }

  NODE_PTR Clone_block(NODE_PTR source) {
    NODE_PTR destination = _container.New_stmt_block(source->Spos());
    STMT_LIST list(destination);
    for (STMT_PTR stmt = source->Begin_stmt(); stmt != source->End_stmt();
         stmt = stmt->Next()) {
      list.Append(Clone_statement(stmt));
    }
    return destination;
  }

  NODE_PTR Clone_expression(NODE_PTR source) {
    AIR_ASSERT(!source->Is_root() && !source->Is_block());
    const uint32_t added_children =
        source->Opcode() == air::core::OPC_ARRAY ? source->Array_dim() : 0;
    AIR_ASSERT(!source->Has_added_chld() ||
               source->Opcode() == air::core::OPC_ARRAY);
    NODE_PTR destination = _container.New_cust_node(
        source->Opcode(), source->Rtype(), source->Spos(), added_children);
    if (added_children != 0) destination->Set_num_arg(added_children);
    Copy_fields(destination, source);
    for (uint32_t idx = 0; idx < source->Num_child(); ++idx) {
      NODE_PTR child = source->Child(idx)->Is_block()
                           ? Clone_block(source->Child(idx))
                           : Clone_expression(source->Child(idx));
      destination->Set_child(idx, child);
    }
    return destination;
  }

  STMT_PTR Clone_statement(STMT_PTR source_stmt) {
    NODE_PTR source = source_stmt->Node();
    AIR_ASSERT(source->Is_root() && !source->Is_ret());
    STMT_PTR destination_stmt =
        _container.New_cust_stmt(source->Opcode(), source->Spos());
    NODE_PTR destination = destination_stmt->Node();
    Copy_fields(destination, source);
    for (uint32_t idx = 0; idx < source->Num_child(); ++idx) {
      NODE_PTR child = source->Child(idx)->Is_block()
                           ? Clone_block(source->Child(idx))
                           : Clone_expression(source->Child(idx));
      destination->Set_child(idx, child);
      if (child->Is_block()) child->Set_parent_stmt(destination_stmt);
    }
    return destination_stmt;
  }

  const CALL_SITE& _site;
  FUNC_SCOPE&      _caller;
  FUNC_SCOPE&      _helper;
  CONTAINER&       _container;
  std::unordered_map<uint64_t, ADDR_DATUM_PTR> _datums;
  std::unordered_map<uint64_t, PREG_PTR>       _pregs;
};

void Collect_referenced_helpers(const std::unordered_set<uint64_t>& ids,
                                NODE_PTR node,
                                std::unordered_set<uint64_t>* referenced) {
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt = stmt->Next()) {
      Collect_referenced_helpers(ids, stmt->Node(), referenced);
    }
    return;
  }
  if (META_INFO::Has_prop<OPR_PROP::ENTRY>(node->Opcode())) {
    uint64_t id = node->Entry()->Owning_func_id().Value();
    if (ids.count(id) != 0) referenced->insert(id);
  }
  for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
    Collect_referenced_helpers(ids, node->Child(idx), referenced);
  }
}

}  // namespace

GLOB_SCOPE* Clone_glob_with_code(GLOB_SCOPE& source) {
  std::unique_ptr<GLOB_SCOPE> clone(new GLOB_SCOPE(source.Id(), true));
  clone->Clone(source);
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = source.Begin_func_scope();
       iter != source.End_func_scope(); ++iter) {
    FUNC_SCOPE& old_function = *iter;
    FUNC_SCOPE& new_function =
        clone->New_func_scope(old_function.Owning_func_id());
    new_function.Clone(old_function);
    STMT_PTR entry = new_function.Container().Clone_stmt_tree(
        old_function.Container().Entry_stmt());
    new_function.Set_entry_stmt(entry);
  }
  return clone.release();
}

AIR_FUNCTION_INLINE_RESULT Inline_tagged_leaf_helpers(
    GLOB_SCOPE& glob, const char* call_attribute,
    const char* helper_attribute) {
  AIR_FUNCTION_INLINE_RESULT result;
  if (call_attribute == nullptr || call_attribute[0] == '\0') {
    result._diagnostic = "generated-helper call attribute is empty";
    return result;
  }
  if (helper_attribute == nullptr || helper_attribute[0] == '\0') {
    result._diagnostic = "generated-helper RETV attribute is empty";
    return result;
  }
  if (!glob.Verify_ir()) {
    result._diagnostic = "input GLOB_SCOPE does not verify";
    return result;
  }

  FUNC_SCOPE_MAP functions;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    FUNC_SCOPE& function = *iter;
    functions.emplace(function.Id().Value(), &function);
  }

  std::vector<CALL_SITE> sites;
  for (const auto& item : functions) {
    FUNC_SCOPE& function = *item.second;
    Collect_calls_in_block(function, function.Container().Entry_node()->Body_blk(),
                           functions, call_attribute, &sites,
                           &result._diagnostic);
    if (!result._diagnostic.empty()) return result;
  }
  if (sites.empty()) {
    result._success = true;
    return result;
  }

  for (const CALL_SITE& site : sites) {
    if (!Validate_call_site(site, helper_attribute, &result._diagnostic))
      return result;
  }

  std::unordered_set<uint64_t> candidate_helpers;
  for (const CALL_SITE& site : sites) {
    candidate_helpers.insert(site._helper->Id().Value());
    SITE_CLONER(site).Inline();
    ++result._calls_inlined;
  }

  std::unordered_set<uint64_t> referenced_helpers;
  for (const auto& item : functions) {
    FUNC_SCOPE& function = *item.second;
    Collect_referenced_helpers(candidate_helpers,
                               function.Container().Entry_node()->Body_blk(),
                               &referenced_helpers);
  }

  std::vector<FUNC_SCOPE*> dead_helpers;
  for (uint64_t id : candidate_helpers) {
    if (referenced_helpers.count(id) == 0) dead_helpers.push_back(functions.at(id));
  }
  for (FUNC_SCOPE* helper : dead_helpers) {
    FUNC_PTR  helper_func  = helper->Owning_func();
    ENTRY_PTR helper_entry = helper_func->Entry_point();
    helper_func->Set_undefined();
    glob.Delete_func_scope(helper);
    glob.Delete_sym(helper_entry);
    glob.Delete_sym(helper_func);
    ++result._helpers_removed;
  }

  if (!glob.Verify_ir()) {
    result._diagnostic = "inlined GLOB_SCOPE does not verify";
    result._calls_inlined = 0;
    result._helpers_removed = 0;
    return result;
  }
  result._success = true;
  result._changed = result._calls_inlined != 0;
  return result;
}

}  // namespace ace::bindings
