//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "tensor2vector_air_normalizer.h"

#include <algorithm>
#include <iomanip>
#include <locale>
#include <map>
#include <sstream>
#include <string_view>

#include "air/base/container.h"
#include "air/base/meta_info.h"
#include "air/base/node.h"
#include "air/base/st.h"
#include "air/base/st_attr.h"
#include "air/base/st_type.h"
#include "air/core/opcode.h"
#include "nn/vector/tensor2vector_plan.h"

namespace nn {
namespace vector {
namespace test {

using namespace air::base;

namespace {

class AIR_NORMALIZER {
public:
  std::string Normalize_function(const FUNC_SCOPE& scope) {
    Reset();
    AIR_ASSERT(scope.Glob_scope().Verify_ir());
    for (uint32_t idx = 0; idx < scope.Formal_cnt(); ++idx) {
      Datum_ordinal(scope.Formal(idx));
    }

    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << "signature=";
    Append_type(os, scope.Owning_func()->Entry_point()->Type());
    os << "|body=";
    const CONTAINER& container = scope.Container();
    Append_node(os, container.Stmt(scope.Entry_stmt_id())->Node());
    return os.str();
  }

  std::string Normalize_native(const VECTOR_KERNEL_NATIVE_AIR_VIEW& view) {
    AIR_ASSERT(view._scope != nullptr);
    AIR_ASSERT(view._result != Null_ptr);
    AIR_ASSERT(view._scope->Glob_scope().Verify_ir());
    Reset();
    std::vector<TYPE_PTR> input_types;
    for (NODE_PTR input : view._inputs) {
      Seed_input(input);
      input_types.push_back(input->Rtype());
    }

    std::ostringstream os;
    os.imbue(std::locale::classic());
    Append_logical_signature(os, input_types, view._result->Rtype());
    Append_logical_body(os, view._first_stmt, view._end_stmt, view._result);
    return os.str();
  }

  std::string Normalize_helper(const FUNC_SCOPE& scope) {
    AIR_ASSERT(scope.Glob_scope().Verify_ir());
    Reset();

    std::vector<TYPE_PTR> input_types;
    CONTAINER&            container = scope.Container();
    for (uint32_t idx = 0; idx < scope.Formal_cnt(); ++idx) {
      ADDR_DATUM_PTR formal = scope.Formal(idx);
      Datum_ordinal(formal);
      input_types.push_back(formal->Type());
    }

    NODE_PTR entry = container.Stmt(scope.Entry_stmt_id())->Node();
    AIR_ASSERT(entry != Null_ptr && entry->Is_entry());
    NODE_PTR body = entry->Last_child();
    AIR_ASSERT(body != Null_ptr && body->Is_block());

    STMT_PTR terminal = Null_ptr;
    for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
         stmt          = stmt->Next()) {
      terminal = stmt;
    }
    AIR_ASSERT(terminal != Null_ptr);
    AIR_ASSERT(terminal->Node()->Opcode() == air::core::OPC_RETV);
    AIR_ASSERT(Count_opcode(body, air::core::OPC_RETV) == 1);

    NODE_PTR result = terminal->Node()->Child(0);
    std::ostringstream os;
    os.imbue(std::locale::classic());
    Append_logical_signature(os, input_types, result->Rtype());
    Append_logical_body(os, body->Begin_stmt(), terminal, result);
    return os.str();
  }

private:
  void Reset() {
    _datum_ordinals.clear();
    _preg_ordinals.clear();
  }

  void Seed_input(NODE_PTR input) {
    AIR_ASSERT(input != Null_ptr);
    if (input->Has_sym()) {
      Datum_ordinal(input->Addr_datum());
      return;
    }
    if (input->Has_preg()) {
      Preg_ordinal(input->Preg());
      return;
    }
    AIR_ASSERT_MSG(false,
                   "native vector-kernel semantic input must be LD or LDP");
  }

  uint32_t Datum_ordinal(ADDR_DATUM_PTR datum) {
    auto iter =
        _datum_ordinals.emplace(datum->Id().Value(), _datum_ordinals.size())
            .first;
    return iter->second;
  }

  uint32_t Preg_ordinal(PREG_PTR preg) {
    auto iter =
        _preg_ordinals.emplace(preg->Id().Value(), _preg_ordinals.size())
            .first;
    return iter->second;
  }

  void Append_type(std::ostream& os, TYPE_PTR type) {
    if (type == Null_ptr) {
      os << "none";
      return;
    }
    os << "type(" << static_cast<uint32_t>(type->Kind()) << ",domain="
       << type->Domain_tag();
    if (type->Is_prim()) {
      os << ",primitive="
         << Vector_kernel_primitive_type_name(
                type->Cast_to_prim()->Encoding());
    } else if (type->Is_array()) {
      os << ",shape=[";
      const std::vector<int64_t> shape = type->Cast_to_arr()->Shape();
      for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (idx != 0) os << ',';
        os << shape[idx];
      }
      os << "],element=";
      Append_type(os, type->Cast_to_arr()->Elem_type());
    } else if (type->Is_ptr()) {
      os << ",pointer-kind="
         << static_cast<uint32_t>(type->Cast_to_ptr()->Ptr_kind())
         << ",pointee=";
      Append_type(os, type->Cast_to_ptr()->Domain_type());
    } else if (type->Is_signature()) {
      os << ",params=[";
      bool first = true;
      for (PARAM_ITER iter = type->Cast_to_sig()->Begin_param();
           iter != type->Cast_to_sig()->End_param(); ++iter) {
        PARAM_PTR param = *iter;
        if (!first) os << ',';
        first = false;
        os << (param->Is_ret() ? "ret:" : "arg:");
        Append_type(os, param->Type());
      }
      os << ']';
    } else {
      os << ",bits=" << type->Bit_size();
    }
    os << ')';
  }

  void Append_logical_signature(std::ostream& os,
                                const std::vector<TYPE_PTR>& input_types,
                                TYPE_PTR result_type) {
    os << "signature=type(ret=";
    Append_type(os, result_type);
    os << ",args=[";
    for (size_t idx = 0; idx < input_types.size(); ++idx) {
      if (idx != 0) os << ',';
      Append_type(os, input_types[idx]);
    }
    os << "])";
  }

  void Append_attributes(std::ostream& os, NODE_PTR node) {
    if (!META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) return;
    os << ",attrs=[";
    bool first = true;
    for (ATTR_ITER iter = node->Begin_attr(); iter != node->End_attr();
         ++iter) {
      ATTR_PTR attr = *iter;
      if (!first) os << ',';
      first = false;
      os << attr->Key() << ':'
         << Vector_kernel_primitive_type_name(attr->Type()) << ':'
         << attr->Count() << ':';
      const std::string_view value = attr->Value();
      os << std::hex << std::setfill('0');
      for (unsigned char byte : value) {
        os << std::setw(2) << static_cast<uint32_t>(byte);
      }
      os << std::dec;
    }
    os << ']';
  }

  void Append_node(std::ostream& os, NODE_PTR node) {
    os << '{' << META_INFO::Domain_name(node->Domain()) << '.'
       << META_INFO::Op_name(node->Opcode());
    if (node->Has_rtype()) {
      os << ",rtype=";
      Append_type(os, node->Rtype());
    }
    if (node->Has_access_type()) {
      os << ",access=";
      Append_type(os, node->Access_type());
    }
    if (node->Has_sym()) {
      ADDR_DATUM_PTR datum = node->Addr_datum();
      os << ",datum=" << Datum_ordinal(datum) << ':';
      Append_type(os, datum->Type());
    }
    if (node->Has_preg()) {
      PREG_PTR preg = node->Preg();
      os << ",preg=" << Preg_ordinal(preg) << ':';
      Append_type(os, preg->Type());
    }
    if (node->Has_ret_var()) {
      PREG_PTR preg = node->Ret_preg();
      os << ",ret-preg=" << Preg_ordinal(preg) << ':';
      Append_type(os, preg->Type());
    }
    if (node->Is_do_loop()) {
      ADDR_DATUM_PTR iv = node->Iv();
      os << ",iv=" << Datum_ordinal(iv) << ':';
      Append_type(os, iv->Type());
    }
    if (node->Has_const_id()) {
      CONSTANT_PTR constant = node->Const();
      os << ",constant-kind=" << static_cast<uint32_t>(constant->Kind())
         << ",constant-type=";
      Append_type(os, constant->Type());
      os << ",constant-hash="
         << Build_vector_kernel_constant_hash(constant);
    }
    if (node->Opcode() == air::core::OPC_INTCONST) {
      os << ",value=" << node->Intconst();
    }
    if (node->Is_comment()) os << ",comment=" << node->Comment();
    if (node->Has_ofst()) os << ",offset=" << node->Ofst();
    if (node->Has_fld()) {
      os << ",field-type=";
      Append_type(os, node->Field()->Type());
      os << ",field-offset=" << node->Field()->Ofst();
    }
    if (node->Has_entry()) {
      os << ",entry-type=";
      Append_type(os, node->Entry()->Type());
    }
    Append_attributes(os, node);

    os << ",children=[";
    if (node->Is_block()) {
      bool first = true;
      for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
           stmt          = stmt->Next()) {
        if (!first) os << ',';
        first = false;
        Append_node(os, stmt->Node());
      }
    } else {
      for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        if (idx != 0) os << ',';
        Append_node(os, node->Child(idx));
      }
    }
    os << "]}";
  }

  void Append_logical_body(std::ostream& os, STMT_PTR first_stmt,
                           STMT_PTR end_stmt, NODE_PTR result) {
    os << "|body={statements=[";
    bool first = true;
    for (STMT_PTR stmt = first_stmt; stmt != end_stmt; stmt = stmt->Next()) {
      AIR_ASSERT(stmt != Null_ptr);
      if (!first) os << ',';
      first = false;
      Append_node(os, stmt->Node());
    }
    os << "],result=";
    Append_node(os, result);
    os << '}';
  }

  uint32_t Count_opcode(NODE_PTR node, OPCODE opcode) const {
    uint32_t count = node->Opcode() == opcode ? 1U : 0U;
    if (node->Is_block()) {
      for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
           stmt          = stmt->Next()) {
        count += Count_opcode(stmt->Node(), opcode);
      }
    } else {
      for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        count += Count_opcode(node->Child(idx), opcode);
      }
    }
    return count;
  }

  std::map<uint32_t, uint32_t> _datum_ordinals;
  std::map<uint32_t, uint32_t> _preg_ordinals;
};

bool Same_entry(ENTRY_PTR lhs, ENTRY_PTR rhs) {
  return lhs != Null_ptr && rhs != Null_ptr &&
         &lhs->Glob_scope() == &rhs->Glob_scope() && lhs->Id() == rhs->Id();
}

bool Same_preg(PREG_PTR lhs, PREG_PTR rhs) {
  return lhs != Null_ptr && rhs != Null_ptr &&
         lhs->Defining_func_scope() == rhs->Defining_func_scope() &&
         lhs->Id() == rhs->Id();
}

bool Same_stmt(STMT_PTR lhs, STMT_PTR rhs) {
  return lhs != Null_ptr && rhs != Null_ptr &&
         lhs->Node()->Container() == rhs->Node()->Container() &&
         lhs->Id() == rhs->Id();
}

bool Same_node(NODE_PTR lhs, NODE_PTR rhs) {
  return lhs != Null_ptr && rhs != Null_ptr &&
         lhs->Container() == rhs->Container() && lhs->Id() == rhs->Id();
}

uint32_t Count_statement(NODE_PTR node, STMT_PTR target) {
  uint32_t count =
      node->Is_root() && Same_stmt(node->Stmt(), target) ? 1U : 0U;
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      count += Count_statement(stmt->Node(), target);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      count += Count_statement(node->Child(idx), target);
    }
  }
  return count;
}

uint32_t Count_node(NODE_PTR node, NODE_PTR target) {
  uint32_t count = Same_node(node, target) ? 1U : 0U;
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      count += Count_node(stmt->Node(), target);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      count += Count_node(node->Child(idx), target);
    }
  }
  return count;
}

uint32_t Count_target_calls(NODE_PTR node, ENTRY_PTR target) {
  uint32_t count =
      node->Is_call() && Same_entry(node->Entry(), target) ? 1U : 0U;
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      count += Count_target_calls(stmt->Node(), target);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      count += Count_target_calls(node->Child(idx), target);
    }
  }
  return count;
}

uint32_t Count_opcode(NODE_PTR node, OPCODE opcode) {
  uint32_t count = node->Opcode() == opcode ? 1U : 0U;
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      count += Count_opcode(stmt->Node(), opcode);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      count += Count_opcode(node->Child(idx), opcode);
    }
  }
  return count;
}

VECTOR_KERNEL_AIR_COMPARE_RESULT Failure(const std::string& message) {
  return VECTOR_KERNEL_AIR_COMPARE_RESULT{false, 0, message, {}, {}};
}

}  // namespace

std::string Normalize_vector_kernel_function(const FUNC_SCOPE& scope) {
  AIR_NORMALIZER normalizer;
  return normalizer.Normalize_function(scope);
}

std::string Normalize_native_vector_kernel(
    const VECTOR_KERNEL_NATIVE_AIR_VIEW& view) {
  AIR_NORMALIZER normalizer;
  return normalizer.Normalize_native(view);
}

std::string Normalize_vector_kernel_helper(const FUNC_SCOPE& helper_scope) {
  AIR_NORMALIZER normalizer;
  return normalizer.Normalize_helper(helper_scope);
}

VECTOR_KERNEL_AIR_COMPARE_RESULT Compare_normalized_vector_kernel_air(
    const std::string& lhs, const std::string& rhs) {
  if (lhs == rhs) {
    return VECTOR_KERNEL_AIR_COMPARE_RESULT{true, std::string::npos, {}, lhs,
                                            rhs};
  }

  const size_t common = std::min(lhs.size(), rhs.size());
  size_t       offset = 0;
  while (offset < common && lhs[offset] == rhs[offset]) ++offset;
  std::ostringstream message;
  message << "normalized AIR differs at byte " << offset;
  return VECTOR_KERNEL_AIR_COMPARE_RESULT{false, offset, message.str(), lhs,
                                          rhs};
}

VECTOR_KERNEL_AIR_COMPARE_RESULT Compare_native_and_helper_vector_kernel_air(
    const VECTOR_KERNEL_NATIVE_AIR_VIEW& native_view,
    const FUNC_SCOPE&                    helper_scope) {
  return Compare_normalized_vector_kernel_air(
      Normalize_native_vector_kernel(native_view),
      Normalize_vector_kernel_helper(helper_scope));
}

VECTOR_KERNEL_AIR_COMPARE_RESULT Check_vector_kernel_call_bridge(
    const FUNC_SCOPE& caller_scope, STMT_PTR call_stmt,
    const std::vector<NODE_PTR>& expected_actuals, NODE_PTR replacement,
    const FUNC_SCOPE& helper_scope) {
  if (&caller_scope.Glob_scope() != &helper_scope.Glob_scope()) {
    return Failure("caller and helper do not share a destination GLOB_SCOPE");
  }
  if (!caller_scope.Glob_scope().Verify_ir()) {
    return Failure("destination GLOB_SCOPE failed Verify_ir");
  }
  if (call_stmt == Null_ptr || !call_stmt->Node()->Is_call()) {
    return Failure("bridge statement is not CORE.CALL");
  }

  NODE_PTR  call         = call_stmt->Node();
  ENTRY_PTR helper_entry = helper_scope.Owning_func()->Entry_point();
  if (call->Container() != &caller_scope.Container()) {
    return Failure("CALL is not caller-owned");
  }
  if (!Same_entry(call->Entry(), helper_entry)) {
    return Failure("CALL does not target the exact helper entry");
  }
  if (Count_statement(caller_scope.Container().Entry_node(), call_stmt) != 1) {
    return Failure("the supplied CALL is not present exactly once in caller");
  }
  if (Count_target_calls(caller_scope.Container().Entry_node(), helper_entry) !=
      1) {
    return Failure("caller does not contain exactly one call to the helper");
  }
  if (call->Num_arg() != helper_scope.Formal_cnt()) {
    return Failure("CALL argument count does not match helper formals");
  }
  if (expected_actuals.size() != call->Num_arg()) {
    return Failure("expected actual count does not match CALL arguments");
  }
  for (uint32_t idx = 0; idx < call->Num_arg(); ++idx) {
    NODE_PTR actual   = call->Child(idx);
    NODE_PTR expected = expected_actuals[idx];
    if (actual->Container() != &caller_scope.Container()) {
      return Failure("CALL actual is not caller-owned");
    }
    if (expected == Null_ptr ||
        expected->Container() != &caller_scope.Container()) {
      return Failure("expected CALL actual is not caller-owned");
    }
    if (!Same_node(actual, expected)) {
      return Failure("CALL actual order does not match expected inputs");
    }
    if (!actual->Rtype()->Is_compatible_type(
            helper_scope.Formal(idx)->Type())) {
      return Failure("CALL actual type does not match helper formal");
    }
  }

  PREG_PTR result_preg = call->Ret_preg();
  if (result_preg == Null_ptr ||
      result_preg->Defining_func_scope() != &caller_scope) {
    return Failure("CALL result preg is not caller-owned");
  }
  TYPE_PTR helper_result = helper_entry->Type()
                               ->Cast_to_sig()
                               ->Ret_param()
                               ->Type();
  if (!result_preg->Type()->Is_compatible_type(helper_result)) {
    return Failure("CALL result preg type does not match helper return");
  }
  if (replacement == Null_ptr ||
      replacement->Opcode() != air::core::OPC_LDP ||
      replacement->Container() != &caller_scope.Container() ||
      !Same_preg(replacement->Preg(), result_preg)) {
    return Failure("replacement is not caller LDP of the CALL result preg");
  }
  if (Count_node(caller_scope.Container().Entry_node(), replacement) != 1) {
    return Failure("replacement LDP is not present exactly once in caller");
  }

  NODE_PTR helper_body = helper_scope.Container().Entry_node()->Last_child();
  if (Count_opcode(helper_body, air::core::OPC_RETV) != 1) {
    return Failure("helper does not contain exactly one RETV");
  }
  STMT_PTR terminal = Null_ptr;
  for (STMT_PTR stmt = helper_body->Begin_stmt();
       stmt != helper_body->End_stmt(); stmt = stmt->Next()) {
    terminal = stmt;
  }
  if (terminal == Null_ptr ||
      terminal->Node()->Opcode() != air::core::OPC_RETV) {
    return Failure("helper RETV is not terminal");
  }

  return VECTOR_KERNEL_AIR_COMPARE_RESULT{true, std::string::npos, {}, {}, {}};
}

}  // namespace test
}  // namespace vector
}  // namespace nn
