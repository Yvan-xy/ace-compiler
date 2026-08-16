//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks2c_verifier.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <regex>
#include <sstream>
#include <utility>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/base/st_type.h"
#include "air/core/opcode.h"
#include "fhe/ckks/phantom_context_manifest.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/poly/opcode.h"
#include "nn/core/attr.h"

namespace fhe {
namespace ckks {
namespace {

using air::base::NODE_PTR;
using air::base::TYPE_PTR;

constexpr uint64_t kMaxFiniteScaleLog2 =
    std::numeric_limits<double>::max_exponent - 1;

enum class OPERAND_KIND {
  CIPHER,
  CIPHER3,
  PLAIN,
  FLOAT_SCALAR,
  INTEGER_SCALAR,
  OTHER,
};

bool Fail(const std::string& message, std::string* diagnostic) {
  if (diagnostic != nullptr) {
    *diagnostic = message;
  }
  return false;
}

bool Fail_node(NODE_PTR node, const std::string& message,
               std::string* diagnostic) {
  std::ostringstream os;
  os << message << " [opcode="
     << air::base::META_INFO::Op_name(node->Opcode())
     << ", AIR=node#" << node->Id().Value() << ']';
  return Fail(os.str(), diagnostic);
}

NODE_PTR Child_or_null(NODE_PTR node, uint32_t index) {
  if (node == air::base::Null_ptr || index >= node->Num_child() ||
      node->Child_id(index) == air::base::Null_id) {
    return air::base::Null_ptr;
  }
  return node->Child(index);
}

bool Has_record_name(TYPE_PTR type, const char* expected) {
  if (type == air::base::Null_ptr || !type->Is_record() ||
      type->Name() == air::base::Null_ptr) {
    return false;
  }
  return std::strcmp(type->Name()->Char_str(), expected) == 0;
}

OPERAND_KIND Operand_kind(TYPE_PTR type) {
  if (Has_record_name(type, "CIPHERTEXT")) {
    return OPERAND_KIND::CIPHER;
  }
  if (Has_record_name(type, "CIPHERTEXT3")) {
    return OPERAND_KIND::CIPHER3;
  }
  if (Has_record_name(type, "PLAINTEXT")) {
    return OPERAND_KIND::PLAIN;
  }
  if (type != air::base::Null_ptr && type->Is_prim()) {
    air::base::PRIMITIVE_TYPE encoding = type->Cast_to_prim()->Encoding();
    if (encoding == air::base::PRIMITIVE_TYPE::FLOAT_32 ||
        encoding == air::base::PRIMITIVE_TYPE::FLOAT_64) {
      return OPERAND_KIND::FLOAT_SCALAR;
    }
    if (type->Is_int()) {
      return OPERAND_KIND::INTEGER_SCALAR;
    }
  }
  return OPERAND_KIND::OTHER;
}

const char* Operator_name(CKKS_OPERATOR op) {
  switch (op) {
    case CKKS_OPERATOR::ADD:
      return "add";
    case CKKS_OPERATOR::SUB:
      return "sub";
    case CKKS_OPERATOR::MUL:
      return "mul";
    default:
      return "operation";
  }
}

bool Is_cipher_family(OPERAND_KIND kind) {
  return kind == OPERAND_KIND::CIPHER || kind == OPERAND_KIND::CIPHER3;
}

bool Cipher_derived_encode_level_source(NODE_PTR argument,
                                        NODE_PTR* source) {
  if (argument == air::base::Null_ptr || argument->Opcode() != OPC_LEVEL ||
      argument->Num_child() != 1 ||
      argument->Child(0) == air::base::Null_ptr ||
      Operand_kind(argument->Child(0)->Rtype()) != OPERAND_KIND::CIPHER) {
    return false;
  }
  *source = argument->Child(0);
  return true;
}

bool Effective_encode_argument(NODE_PTR node, uint32_t child_index,
                               const char* attr_name, int64_t* value) {
  if (attr_name != nullptr) {
    const uint32_t* attr = node->Attr<uint32_t>(attr_name);
    if (attr != nullptr) {
      *value = *attr;
      return true;
    }
  }
  if (node->Num_child() <= child_index ||
      node->Child(child_index) == air::base::Null_ptr) {
    return false;
  }
  NODE_PTR argument = node->Child(child_index);
  if (argument->Opcode() == air::core::OPC_INTCONST) {
    *value = argument->Intconst();
    return true;
  }

  // Scale management binds an unspecified encode level to the exact cipher
  // operand with CKKS.level(cipher). Internal/callable functions must retain
  // that runtime query because their concrete input chain is not part of the
  // entry-point ABI. Use the compiler's nonzero source metadata only for the
  // verifier's range and scale-representability checks; IR2C still emits the
  // runtime level query. Arbitrary dynamic expressions remain rejected.
  if (attr_name == nullptr ||
      std::strcmp(attr_name, core::FHE_ATTR_KIND::LEVEL) != 0) {
    return false;
  }
  NODE_PTR source = air::base::Null_ptr;
  if (!Cipher_derived_encode_level_source(argument, &source)) {
    return false;
  }
  const uint32_t* source_level =
      source->Attr<uint32_t>(core::FHE_ATTR_KIND::LEVEL);
  // Some internal SSA loads remain deliberately runtime-level-polymorphic.
  // In that case use the minimum valid Q count as a conservative verifier
  // budget. The emitted Level(cipher) and Phantom runtime retain authority.
  *value = source_level == nullptr || *source_level == 0 ? 1 : *source_level;
  return true;
}

bool Known_metadata(NODE_PTR node, const char* attr_name,
                    uint32_t encode_child_index, uint32_t* value) {
  const uint32_t* attr = node->Attr<uint32_t>(attr_name);
  if (attr != nullptr) {
    if (*attr == 0) {
      return false;
    }
    *value = *attr;
    return true;
  }
  if (node->Domain() != CKKS_DOMAIN::ID ||
      static_cast<CKKS_OPERATOR>(node->Operator()) != CKKS_OPERATOR::ENCODE ||
      node->Num_child() <= encode_child_index ||
      node->Child(encode_child_index) == air::base::Null_ptr ||
      node->Child(encode_child_index)->Opcode() != air::core::OPC_INTCONST) {
    return false;
  }
  int64_t constant = node->Child(encode_child_index)->Intconst();
  if (constant <= 0 ||
      static_cast<uint64_t>(constant) >
          std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(constant);
  return true;
}

TYPE_PTR Encode_element_type(TYPE_PTR type) {
  if (type == air::base::Null_ptr) {
    return type;
  }
  if (type->Is_ptr()) {
    return type->Cast_to_ptr()->Domain_type();
  }
  if (type->Is_array()) {
    return type->Cast_to_arr()->Elem_type();
  }
  return type;
}

bool Verify_cipher_derived_encode_parent(NODE_PTR node, NODE_PTR parent,
                                         std::string* diagnostic) {
  if (node->Attr<uint32_t>(core::FHE_ATTR_KIND::LEVEL) != nullptr ||
      node->Num_child() <= 3 || node->Child(3) == air::base::Null_ptr) {
    return true;
  }
  NODE_PTR source = air::base::Null_ptr;
  if (!Cipher_derived_encode_level_source(node->Child(3), &source)) {
    return true;
  }
  if (parent != air::base::Null_ptr && parent->Domain() == CKKS_DOMAIN::ID) {
    CKKS_OPERATOR parent_op =
        static_cast<CKKS_OPERATOR>(parent->Operator());
    if ((parent_op == CKKS_OPERATOR::ADD || parent_op == CKKS_OPERATOR::SUB ||
         parent_op == CKKS_OPERATOR::MUL) &&
        parent->Num_child() == 2 && parent->Child(1) == node &&
        parent->Child(0) == source) {
      return true;
    }
  }
  return Fail(
      "Phantom CKKS2C cipher-derived encode level requires the encode to be "
      "the right operand of add, sub, or mul and to query that operation's "
      "left operand",
      diagnostic);
}

bool Verify_phantom_encode(NODE_PTR node, NODE_PTR parent,
                           const PHANTOM_CONTEXT_DESCRIPTOR& context,
                           std::string* diagnostic) {
  if (node->Num_child() != 4) {
    return Fail("Phantom CKKS2C encode requires exactly four operands",
                diagnostic);
  }

  TYPE_PTR element_type = Encode_element_type(node->Child(0)->Rtype());
  if (element_type != air::base::Null_ptr && element_type->Is_int()) {
    return Fail(
        "Phantom CKKS2C encode does not support integer input; use "
        "floating-point data",
        diagnostic);
  }
  if (element_type == air::base::Null_ptr || !element_type->Is_prim()) {
    return Fail("Phantom CKKS2C encode requires floating-point input data",
                diagnostic);
  }
  air::base::PRIMITIVE_TYPE encoding =
      element_type->Cast_to_prim()->Encoding();
  if (encoding != air::base::PRIMITIVE_TYPE::FLOAT_32 &&
      encoding != air::base::PRIMITIVE_TYPE::FLOAT_64) {
    return Fail("Phantom CKKS2C encode requires float32 or float64 input",
                diagnostic);
  }

  int64_t length = 0;
  if (!Effective_encode_argument(node, 1, nullptr, &length)) {
    return Fail("Phantom CKKS2C encode requires constant length", diagnostic);
  }
  if (length < 1 ||
      static_cast<uint64_t>(length) > context._logical_slots) {
    std::ostringstream os;
    os << "Phantom CKKS2C encode length must be in [1, "
       << context._logical_slots << "], got " << length;
    return Fail(os.str(), diagnostic);
  }

  int64_t scale_degree = 0;
  if (!Effective_encode_argument(node, 2, core::FHE_ATTR_KIND::SCALE,
                                 &scale_degree)) {
    return Fail("Phantom CKKS2C encode requires constant scale_degree",
                diagnostic);
  }
  if (scale_degree <= 0) {
    std::ostringstream os;
    os << "Phantom CKKS2C encode scale_degree must be positive, got "
       << scale_degree;
    return Fail(os.str(), diagnostic);
  }

  int64_t logical_level = 0;
  if (!Effective_encode_argument(node, 3, core::FHE_ATTR_KIND::LEVEL,
                                 &logical_level)) {
    return Fail("Phantom CKKS2C encode requires constant logical_level",
                diagnostic);
  }
  if (!Verify_cipher_derived_encode_parent(node, parent, diagnostic)) {
    return false;
  }
  if (logical_level < 1 ||
      static_cast<uint64_t>(logical_level) >
          context._data_q_bit_sizes.size()) {
    std::ostringstream os;
    os << "Phantom CKKS2C encode logical_level must be in [1, "
       << context._data_q_bit_sizes.size() << "], got " << logical_level;
    return Fail(os.str(), diagnostic);
  }

  uint64_t scale_degree_u64 = static_cast<uint64_t>(scale_degree);
  uint64_t logical_level_u64 = static_cast<uint64_t>(logical_level);
  bool exponent_overflows =
      scale_degree_u64 >
      std::numeric_limits<uint64_t>::max() /
          context._scaling_modulus_bits;
  uint64_t scale_log2 =
      exponent_overflows ? std::numeric_limits<uint64_t>::max()
                         : scale_degree_u64 *
                               context._scaling_modulus_bits;
  uint64_t modulus_bit_budget =
      context._first_modulus_bits +
      (logical_level_u64 - 1) * context._scaling_modulus_bits;
  if (exponent_overflows || scale_log2 > kMaxFiniteScaleLog2 ||
      scale_log2 >= modulus_bit_budget) {
    std::ostringstream os;
    os << "Phantom CKKS2C encode scale_degree " << scale_degree
       << " is not representable at logical_level " << logical_level;
    return Fail(os.str(), diagnostic);
  }
  return true;
}

bool Verify_static_add_sub_metadata(NODE_PTR node, CKKS_OPERATOR op,
                                    std::string* diagnostic) {
  NODE_PTR lhs = node->Child(0);
  NODE_PTR rhs = node->Child(1);
  uint32_t lhs_level = 0;
  uint32_t rhs_level = 0;
  if (Known_metadata(lhs, core::FHE_ATTR_KIND::LEVEL, 3, &lhs_level) &&
      Known_metadata(rhs, core::FHE_ATTR_KIND::LEVEL, 3, &rhs_level) &&
      lhs_level != rhs_level) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " requires matching logical levels when statically known: lhs="
       << lhs_level << ", rhs=" << rhs_level;
    return Fail(os.str(), diagnostic);
  }

  uint32_t lhs_scale = 0;
  uint32_t rhs_scale = 0;
  if (Known_metadata(lhs, core::FHE_ATTR_KIND::SCALE, 2, &lhs_scale) &&
      Known_metadata(rhs, core::FHE_ATTR_KIND::SCALE, 2, &rhs_scale) &&
      lhs_scale != rhs_scale) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " requires matching scale degrees when statically known: lhs="
       << lhs_scale << ", rhs=" << rhs_scale;
    return Fail(os.str(), diagnostic);
  }
  return true;
}

bool Verify_static_cipher_rescale_metadata(NODE_PTR node, CKKS_OPERATOR op,
                                           std::string* diagnostic) {
  const uint32_t* lhs_rescale_level =
      node->Child(0)->Attr<uint32_t>(core::FHE_ATTR_KIND::RESCALE_LEVEL);
  const uint32_t* rhs_rescale_level =
      node->Child(1)->Attr<uint32_t>(core::FHE_ATTR_KIND::RESCALE_LEVEL);
  if (lhs_rescale_level != nullptr && rhs_rescale_level != nullptr &&
      *lhs_rescale_level != *rhs_rescale_level) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " requires matching physical rescale levels when statically known: "
          "lhs="
       << *lhs_rescale_level << ", rhs=" << *rhs_rescale_level;
    return Fail(os.str(), diagnostic);
  }
  return true;
}

bool Verify_static_cipher_plain_level_metadata(NODE_PTR node,
                                               CKKS_OPERATOR op,
                                               std::string* diagnostic) {
  uint32_t cipher_level = 0;
  uint32_t plain_level  = 0;
  if (!Known_metadata(node->Child(0), core::FHE_ATTR_KIND::LEVEL, 3,
                      &cipher_level) ||
      !Known_metadata(node->Child(1), core::FHE_ATTR_KIND::LEVEL, 3,
                      &plain_level) ||
      cipher_level == plain_level) {
    return true;
  }

  std::ostringstream os;
  os << "Phantom CKKS2C " << Operator_name(op)
     << " requires matching ciphertext and plaintext logical levels when "
        "statically known: ciphertext="
     << cipher_level << ", plaintext=" << plain_level;
  return Fail(os.str(), diagnostic);
}

bool Verify_phantom_binary(NODE_PTR node, CKKS_OPERATOR op,
                           std::string* diagnostic) {
  if (node->Num_child() != 2) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " requires exactly two operands";
    return Fail(os.str(), diagnostic);
  }
  OPERAND_KIND lhs_kind = Operand_kind(node->Child(0)->Rtype());
  OPERAND_KIND rhs_kind = Operand_kind(node->Child(1)->Rtype());
  if (lhs_kind == OPERAND_KIND::INTEGER_SCALAR ||
      rhs_kind == OPERAND_KIND::INTEGER_SCALAR) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " does not support integer scalar operands";
    return Fail(os.str(), diagnostic);
  }
  // AIR producers currently retain CIPHERTEXT3 on some values after RELIN,
  // so the record name cannot prove the dynamic component count here.
  if (!Is_cipher_family(lhs_kind)) {
    std::ostringstream os;
    if (Is_cipher_family(rhs_kind) &&
        (lhs_kind == OPERAND_KIND::PLAIN ||
         lhs_kind == OPERAND_KIND::FLOAT_SCALAR)) {
      os << "Phantom CKKS2C " << Operator_name(op)
         << " does not support reverse plaintext/scalar-cipher operands";
    } else {
      os << "Phantom CKKS2C " << Operator_name(op)
         << " requires a ciphertext-family left operand";
    }
    return Fail(os.str(), diagnostic);
  }
  if (!Is_cipher_family(rhs_kind) && rhs_kind != OPERAND_KIND::PLAIN &&
      rhs_kind != OPERAND_KIND::FLOAT_SCALAR) {
    std::ostringstream os;
    os << "Phantom CKKS2C " << Operator_name(op)
       << " requires a ciphertext-family, PLAINTEXT, float32, or float64 "
          "right "
          "operand";
    return Fail(os.str(), diagnostic);
  }
  if (Is_cipher_family(rhs_kind) &&
      !Verify_static_cipher_rescale_metadata(node, op, diagnostic)) {
    return false;
  }
  if (op == CKKS_OPERATOR::MUL && rhs_kind == OPERAND_KIND::PLAIN &&
      !Verify_static_cipher_plain_level_metadata(node, op, diagnostic)) {
    return false;
  }
  if ((op == CKKS_OPERATOR::ADD || op == CKKS_OPERATOR::SUB) &&
      rhs_kind != OPERAND_KIND::FLOAT_SCALAR) {
    return Verify_static_add_sub_metadata(node, op, diagnostic);
  }
  return true;
}

bool Verify_preserved_metadata(NODE_PTR node, NODE_PTR input,
                               std::string* diagnostic) {
  const char* metadata[] = {core::FHE_ATTR_KIND::LEVEL,
                            core::FHE_ATTR_KIND::SCALE,
                            core::FHE_ATTR_KIND::RESCALE_LEVEL};
  for (const char* attr_name : metadata) {
    const uint32_t* input_value = input->Attr<uint32_t>(attr_name);
    const uint32_t* result_value = node->Attr<uint32_t>(attr_name);
    if (input_value == nullptr || result_value == nullptr ||
        *input_value != *result_value) {
      std::ostringstream os;
      os << "Phantom CKKS2C retained operation must preserve " << attr_name
         << " metadata: observed input="
         << (input_value == nullptr ? "missing" : std::to_string(*input_value))
         << ", result="
         << (result_value == nullptr ? "missing"
                                     : std::to_string(*result_value));
      return Fail_node(node, os.str(), diagnostic);
    }
  }
  return true;
}

bool Verify_raise_metadata(NODE_PTR node, NODE_PTR input, uint32_t full_q,
                           uint32_t target_q_count,
                           std::string* diagnostic) {
  const uint32_t* input_scale =
      input->Attr<uint32_t>(core::FHE_ATTR_KIND::SCALE);
  const uint32_t* result_scale =
      node->Attr<uint32_t>(core::FHE_ATTR_KIND::SCALE);
  if (input_scale == nullptr || result_scale == nullptr ||
      *input_scale != *result_scale) {
    std::ostringstream os;
    os << "Phantom CKKS2C raise_mod must preserve SCALE metadata: observed "
       << "input="
       << (input_scale == nullptr ? "missing" : std::to_string(*input_scale))
       << ", result="
       << (result_scale == nullptr ? "missing"
                                   : std::to_string(*result_scale));
    return Fail_node(node, os.str(), diagnostic);
  }

  const uint32_t* input_rescale =
      input->Attr<uint32_t>(core::FHE_ATTR_KIND::RESCALE_LEVEL);
  const uint32_t* result_rescale =
      node->Attr<uint32_t>(core::FHE_ATTR_KIND::RESCALE_LEVEL);
  // RESCALE_LEVEL uses ACE's Q-coordinate count, which has the full data-Q
  // count plus the level-zero coordinate. Thus bottom-Q is full_q and a
  // target with k active Q moduli is (full_q + 1 - k).
  const uint32_t q_coordinate_count = full_q + 1;
  const uint32_t expected_input_rescale = full_q;
  const uint32_t expected_result_rescale =
      q_coordinate_count - target_q_count;
  if (input_rescale == nullptr || result_rescale == nullptr ||
      *input_rescale != expected_input_rescale ||
      *result_rescale != expected_result_rescale) {
    std::ostringstream os;
    os << "Phantom CKKS2C raise_mod level coordinates disagree: observed "
       << "input RESCALE_LEVEL="
       << (input_rescale == nullptr ? "missing"
                                    : std::to_string(*input_rescale))
       << ", result RESCALE_LEVEL="
       << (result_rescale == nullptr ? "missing"
                                     : std::to_string(*result_rescale))
       << "; expected input=" << expected_input_rescale
       << ", result=" << expected_result_rescale;
    return Fail_node(node, os.str(), diagnostic);
  }
  return true;
}

bool Verify_retained_cipher_result(NODE_PTR node, NODE_PTR input,
                                   std::string* diagnostic) {
  if (input == air::base::Null_ptr ||
      Operand_kind(input->Rtype()) != OPERAND_KIND::CIPHER ||
      Operand_kind(node->Rtype()) != OPERAND_KIND::CIPHER ||
      input->Rtype_id() != node->Rtype_id()) {
    return Fail_node(
        node,
        "Phantom CKKS2C retained operation requires matching CIPHERTEXT "
        "input/result types",
        diagnostic);
  }
  return Verify_preserved_metadata(node, input, diagnostic);
}

bool Verify_phantom_conjugate(NODE_PTR node, std::string* diagnostic) {
  if (node->Num_child() != 1) {
    return Fail_node(node,
                     "Phantom CKKS2C conjugate requires exactly one operand",
                     diagnostic);
  }
  return Verify_retained_cipher_result(node, Child_or_null(node, 0),
                                       diagnostic);
}

bool Verify_phantom_rotate_batch(NODE_PTR node, std::string* diagnostic) {
  if (node->Num_child() != 1) {
    return Fail_node(
        node, "Phantom CKKS2C rotate_batch requires exactly one operand",
        diagnostic);
  }
  NODE_PTR input = Child_or_null(node, 0);
  if (input == air::base::Null_ptr ||
      Operand_kind(input->Rtype()) != OPERAND_KIND::CIPHER) {
    return Fail_node(node,
                     "Phantom CKKS2C rotate_batch requires a CIPHERTEXT "
                     "operand",
                     diagnostic);
  }
  TYPE_PTR result_type = node->Rtype();
  uint32_t count = 0;
  const int* rotations = node->Attr<int>(nn::core::ATTR::RNUM, &count);
  if (rotations == nullptr || count == 0) {
    return Fail_node(node,
                     "Phantom CKKS2C rotate_batch requires a constant, "
                     "non-empty ordered RNUM attribute",
                     diagnostic);
  }
  if (result_type == air::base::Null_ptr || !result_type->Is_array() ||
      result_type->Cast_to_arr()->Dim() != 1 ||
      result_type->Cast_to_arr()->Elem_count() != count ||
      Operand_kind(result_type->Cast_to_arr()->Elem_type()) !=
          OPERAND_KIND::CIPHER ||
      result_type->Cast_to_arr()->Elem_type_id() !=
          input->Rtype_id()) {
    return Fail_node(
        node,
        "Phantom CKKS2C rotate_batch requires a one-dimensional "
        "CIPHERTEXT array whose exact length matches RNUM",
        diagnostic);
  }
  return Verify_preserved_metadata(node, input, diagnostic);
}

bool Verify_phantom_raise_mod(NODE_PTR node,
                              const PHANTOM_CONTEXT_DESCRIPTOR& context,
                              std::string* diagnostic) {
  if (node->Attr<uint32_t>(core::FHE_ATTR_KIND::RUNTIME_RAISE_LEVEL) !=
      nullptr) {
    return Fail_node(
        node,
        "Phantom CKKS2C raise_mod forbids runtime target-level helpers",
        diagnostic);
  }
  NODE_PTR target_node = Child_or_null(node, 1);
  if (node->Num_child() != 2 || target_node == air::base::Null_ptr ||
      target_node->Opcode() != air::core::OPC_INTCONST) {
    return Fail_node(
        node,
        "Phantom CKKS2C raise_mod requires a constant target_q_count",
        diagnostic);
  }
  const int64_t target = target_node->Intconst();
  const int64_t full_q = static_cast<int64_t>(context._data_q_bit_sizes.size());
  if (target < 1 || target > full_q) {
    std::ostringstream os;
    os << "Phantom CKKS2C raise_mod target_q_count must be in [1, "
       << full_q << "], got " << target;
    return Fail_node(node, os.str(), diagnostic);
  }
  if (target != full_q) {
    std::ostringstream os;
    os << "Phantom CKKS2C raise_mod v1 requires the full data-Q count "
       << full_q << ", got " << target;
    return Fail_node(node, os.str(), diagnostic);
  }
  NODE_PTR input = Child_or_null(node, 0);
  if (input == air::base::Null_ptr ||
      Operand_kind(input->Rtype()) != OPERAND_KIND::CIPHER ||
      Operand_kind(node->Rtype()) != OPERAND_KIND::CIPHER ||
      input->Rtype_id() != node->Rtype_id()) {
    return Fail_node(
        node,
        "Phantom CKKS2C raise_mod requires matching CIPHERTEXT input/result "
        "types",
        diagnostic);
  }
  const uint32_t* input_level =
      input->Attr<uint32_t>(core::FHE_ATTR_KIND::LEVEL);
  const uint32_t* result_level =
      node->Attr<uint32_t>(core::FHE_ATTR_KIND::LEVEL);
  if (input_level == nullptr || *input_level != 1) {
    return Fail_node(node,
                     "Phantom CKKS2C raise_mod requires a bottom one-Q input "
                     "with LEVEL=1",
                     diagnostic);
  }
  if (result_level == nullptr || *result_level != target) {
    std::ostringstream os;
    os << "Phantom CKKS2C raise_mod result LEVEL must equal target_q_count "
       << target;
    return Fail_node(node, os.str(), diagnostic);
  }
  return Verify_raise_metadata(node, input, static_cast<uint32_t>(full_q),
                               static_cast<uint32_t>(target), diagnostic);
}

bool Verify_phantom_mul_mono(NODE_PTR node,
                             const PHANTOM_CONTEXT_DESCRIPTOR& context,
                             std::string* diagnostic) {
  NODE_PTR power_node = Child_or_null(node, 1);
  if (node->Num_child() != 2 || power_node == air::base::Null_ptr ||
      power_node->Opcode() != air::core::OPC_INTCONST) {
    return Fail_node(node,
                     "Phantom CKKS2C mul_mono requires a constant monomial "
                     "power",
                     diagnostic);
  }
  uint32_t rnum_count = 0;
  if (node->Attr<int>(nn::core::ATTR::RNUM, &rnum_count) != nullptr) {
    return Fail_node(node,
                     "Phantom CKKS2C mul_mono forbids rotation RNUM metadata",
                     diagnostic);
  }
  if (context._poly_degree == 0) {
    return Fail_node(node,
                     "Phantom CKKS2C mul_mono requires a nonzero polynomial "
                     "degree",
                     diagnostic);
  }
  const int64_t power = power_node->Intconst();
  const int64_t period = static_cast<int64_t>(context._poly_degree) * 2;
  if (power < 0 || power >= period) {
    std::ostringstream os;
    os << "Phantom CKKS2C mul_mono power must be normalized into [0, "
       << period << "), got " << power;
    return Fail_node(node, os.str(), diagnostic);
  }
  return Verify_retained_cipher_result(node, Child_or_null(node, 0),
                                       diagnostic);
}

bool Verify_phantom_rotate(NODE_PTR node, std::string* diagnostic) {
  if (node->Num_child() != 2 ||
      node->Child(1)->Opcode() != air::core::OPC_INTCONST) {
    return Fail(
        "Phantom CKKS2C rotate requires a constant signed 32-bit step",
        diagnostic);
  }
  int64_t step = node->Child(1)->Intconst();
  if (step < std::numeric_limits<int32_t>::min() ||
      step > std::numeric_limits<int32_t>::max()) {
    return Fail(
        "Phantom CKKS2C rotate requires a constant signed 32-bit step",
        diagnostic);
  }
  uint32_t   count = 0;
  const int* attr = node->Attr<int>(nn::core::ATTR::RNUM, &count);
  if (attr == nullptr || count != 1 || attr[0] != step) {
    return Fail(
        "Phantom CKKS2C rotate requires one RNUM entry matching its constant "
        "step",
        diagnostic);
  }
  if (!Is_cipher_family(Operand_kind(node->Child(0)->Rtype()))) {
    return Fail("Phantom CKKS2C rotate requires a ciphertext-family operand",
                diagnostic);
  }
  return true;
}

bool Verify_phantom_relin(NODE_PTR node, std::string* diagnostic) {
  OPERAND_KIND input_kind =
      node->Num_child() == 1 ? Operand_kind(node->Child(0)->Rtype())
                             : OPERAND_KIND::OTHER;
  // Current AIR producers do not consistently distinguish a dynamic size-3
  // value from the CIPHERTEXT record at this boundary.  Keep the statically
  // provable family/arity check here; the adapter's checked precondition owns
  // the exact component-count rejection.
  if (node->Num_child() != 1 ||
      (input_kind != OPERAND_KIND::CIPHER &&
       input_kind != OPERAND_KIND::CIPHER3)) {
    return Fail(
        "Phantom CKKS2C relin requires a ciphertext-family operand",
        diagnostic);
  }
  return true;
}

bool Verify_phantom_modulus_drop(NODE_PTR node, CKKS_OPERATOR op,
                                 std::string* diagnostic) {
  OPERAND_KIND input_kind =
      node->Num_child() == 1 ? Operand_kind(node->Child(0)->Rtype())
                             : OPERAND_KIND::OTHER;
  if (node->Num_child() != 1 ||
      (input_kind != OPERAND_KIND::CIPHER &&
       input_kind != OPERAND_KIND::CIPHER3) ||
      Operand_kind(node->Rtype()) != input_kind) {
    std::ostringstream os;
    os << "Phantom CKKS2C "
       << (op == CKKS_OPERATOR::RESCALE ? "rescale" : "modswitch")
       << " requires matching CIPHERTEXT or CIPHERTEXT3 input/result types";
    return Fail(os.str(), diagnostic);
  }
  return true;
}

bool Verify_ckks_node(NODE_PTR node, NODE_PTR parent,
                      const PHANTOM_CONTEXT_DESCRIPTOR& context,
                      core::PROVIDER provider,
                      std::string* diagnostic) {
  CKKS_OPERATOR op = static_cast<CKKS_OPERATOR>(node->Operator());
  if (op == CKKS_OPERATOR::LINEAR_TRANSFORM) {
    return Fail(
        "CKKS2C rejects unlowered compiler-only CKKS.linear_transform",
        diagnostic);
  }
  if (provider == core::PROVIDER::PHANTOM) {
    switch (op) {
      case CKKS_OPERATOR::BOOTSTRAP:
      case CKKS_OPERATOR::BOOTSTRAP_COEFFS_TO_SLOTS:
      case CKKS_OPERATOR::BOOTSTRAP_EVAL_MOD:
      case CKKS_OPERATOR::BOOTSTRAP_SLOTS_TO_COEFFS: {
        std::ostringstream os;
        os << "Phantom primitive CKKS2C rejects forbidden "
           << air::base::META_INFO::Op_name(node->Opcode());
        return Fail(os.str(), diagnostic);
      }
      default:
        break;
    }

    switch (op) {
      case CKKS_OPERATOR::ENCODE:
        return Verify_phantom_encode(node, parent, context, diagnostic);
      case CKKS_OPERATOR::ADD:
      case CKKS_OPERATOR::SUB:
      case CKKS_OPERATOR::MUL:
        return Verify_phantom_binary(node, op, diagnostic);
      case CKKS_OPERATOR::ROTATE:
        return Verify_phantom_rotate(node, diagnostic);
      case CKKS_OPERATOR::RELIN:
        return Verify_phantom_relin(node, diagnostic);
      case CKKS_OPERATOR::RESCALE:
      case CKKS_OPERATOR::MODSWITCH:
        return Verify_phantom_modulus_drop(node, op, diagnostic);
      case CKKS_OPERATOR::CONJUGATE:
        return Verify_phantom_conjugate(node, diagnostic);
      case CKKS_OPERATOR::ROTATE_BATCH:
        return Verify_phantom_rotate_batch(node, diagnostic);
      case CKKS_OPERATOR::RAISE_MOD:
        return Verify_phantom_raise_mod(node, context, diagnostic);
      case CKKS_OPERATOR::MUL_MONO:
        return Verify_phantom_mul_mono(node, context, diagnostic);
      default:
        break;
    }
  }

  switch (op) {
    case CKKS_OPERATOR::ROTATE:
    case CKKS_OPERATOR::ADD:
    case CKKS_OPERATOR::SUB:
    case CKKS_OPERATOR::MUL:
    case CKKS_OPERATOR::ENCODE:
    case CKKS_OPERATOR::RESCALE:
    case CKKS_OPERATOR::MODSWITCH:
    case CKKS_OPERATOR::RELIN:
    case CKKS_OPERATOR::SCALE:
    case CKKS_OPERATOR::LEVEL:
    case CKKS_OPERATOR::FREE:
      return true;

    case CKKS_OPERATOR::CONJUGATE:
      if (node->Num_child() != 1 ||
          Child_or_null(node, 0) == air::base::Null_ptr) {
        return Fail("CKKS2C conjugate requires exactly one operand",
                    diagnostic);
      }
      return true;

    case CKKS_OPERATOR::ROTATE_BATCH: {
      uint32_t count = 0;
      const int* rotations =
          node->Attr<int>(nn::core::ATTR::RNUM, &count);
      if (node->Num_child() != 1 ||
          Child_or_null(node, 0) == air::base::Null_ptr ||
          rotations == nullptr || count == 0 ||
          node->Rtype() == air::base::Null_ptr ||
          !node->Rtype()->Is_array() ||
          node->Rtype()->Cast_to_arr()->Elem_count() != count) {
        return Fail(
            "CKKS2C rotate_batch requires a non-empty ordered RNUM attribute "
            "matching the result array length",
            diagnostic);
      }
      return true;
    }

    case CKKS_OPERATOR::RAISE_MOD: {
      NODE_PTR target_node = Child_or_null(node, 1);
      if (node->Num_child() != 2 ||
          Child_or_null(node, 0) == air::base::Null_ptr ||
          target_node == air::base::Null_ptr ||
          target_node->Opcode() != air::core::OPC_INTCONST) {
        return Fail(
            "CKKS2C raise_mod requires an input and a constant "
            "target_q_count operand",
            diagnostic);
      }
      return true;
    }

    case CKKS_OPERATOR::MUL_MONO: {
      NODE_PTR power_node = Child_or_null(node, 1);
      if (node->Num_child() != 2 ||
          Child_or_null(node, 0) == air::base::Null_ptr ||
          power_node == air::base::Null_ptr ||
          power_node->Opcode() != air::core::OPC_INTCONST) {
        return Fail("CKKS2C mul_mono requires an input and a constant "
                    "monomial power",
                    diagnostic);
      }
      return true;
    }

    case CKKS_OPERATOR::BOOTSTRAP:
    case CKKS_OPERATOR::BOOTSTRAP_COEFFS_TO_SLOTS:
    case CKKS_OPERATOR::BOOTSTRAP_EVAL_MOD:
    case CKKS_OPERATOR::BOOTSTRAP_SLOTS_TO_COEFFS:
      // Retained only for the temporary ANT compatibility route.
      return true;

    case CKKS_OPERATOR::NEG:
    case CKKS_OPERATOR::UPSCALE:
    case CKKS_OPERATOR::BATCH_SIZE:
    default: {
      std::ostringstream os;
      os << "CKKS2C has no capability for "
         << air::base::META_INFO::Op_name(node->Opcode());
      return Fail(os.str(), diagnostic);
    }
  }
}

bool Verify_node(NODE_PTR node, NODE_PTR parent,
                 const PHANTOM_CONTEXT_DESCRIPTOR& context,
                 core::PROVIDER provider,
                 std::string* diagnostic) {
  if (node == air::base::Null_ptr) {
    return true;
  }
  if (node->Domain() == fhe::poly::POLYNOMIAL_DID) {
    return Fail("CKKS2C input contains forbidden POLY AIR", diagnostic);
  }
  if (node->Domain() == CKKS_DOMAIN::ID &&
      !Verify_ckks_node(node, parent, context, provider, diagnostic)) {
    return false;
  }
  if (node->Is_block()) {
    for (air::base::STMT_PTR stmt = node->Begin_stmt();
         stmt != node->End_stmt(); stmt = stmt->Next()) {
      if (!Verify_node(stmt->Node(), node, context, provider, diagnostic)) {
        return false;
      }
    }
    return true;
  }
  for (uint32_t i = 0; i < node->Num_child(); ++i) {
    if (!Verify_node(node->Child(i), node, context, provider, diagnostic)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool CKKS2C_VERIFIER::Verify(air::base::GLOB_SCOPE* glob,
                             const core::CTX_PARAM& ctx_param,
                             core::PROVIDER provider,
                             std::string* diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  if (glob == nullptr) {
    return Fail("CKKS2C received no AIR module", diagnostic);
  }
  PHANTOM_CONTEXT_DESCRIPTOR context;
  if (provider == core::PROVIDER::PHANTOM) {
    try {
      context = Build_phantom_context_descriptor(ctx_param);
    } catch (const std::invalid_argument& error) {
      return Fail(std::string("Phantom CKKS2C context is invalid: ") +
                      error.what(),
                  diagnostic);
    }
  }
  for (air::base::GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
       it != glob->End_func_scope(); ++it) {
    NODE_PTR entry = (*it).Container().Entry_node();
    if (!Verify_node(entry, air::base::Null_ptr, context, provider,
                     diagnostic)) {
      return false;
    }
  }
  return true;
}

bool CKKS2C_VERIFIER::Verify_source(const std::string& source,
                                    core::PROVIDER provider,
                                    std::string* diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  if (provider != core::PROVIDER::PHANTOM) {
    return true;
  }
  if (source.find("#include \"rt_phantom/rt_phantom.h\"") ==
      std::string::npos) {
    return Fail("Phantom CKKS2C source is missing rt_phantom/rt_phantom.h",
                diagnostic);
  }
  if (source.find("Get_phantom_context_manifest()") == std::string::npos) {
    return Fail(
        "Phantom CKKS2C source is missing the context manifest",
        diagnostic);
  }
  if (source.find("Get_phantom_resource_manifest()") == std::string::npos) {
    return Fail(
        "Phantom CKKS2C source is missing the resource manifest",
        diagnostic);
  }
  if (source.find("Get_phantom_constant_manifest()") == std::string::npos) {
    return Fail(
        "Phantom CKKS2C source is missing the constant manifest",
        diagnostic);
  }
  const bool has_complex_plaintext =
      source.find("Encode_dcmplx(") != std::string::npos ||
      source.find("Load_cached_plain(") != std::string::npos;
  const bool requests_complex_plaintext =
      source.find("PHANTOM_RESOURCE_COMPLEX_PLAINTEXT") !=
      std::string::npos;
  if (has_complex_plaintext != requests_complex_plaintext) {
    return Fail(
        "Phantom CKKS2C complex plaintext calls and resource flag disagree",
        diagnostic);
  }
  if (source.find("PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE") !=
      std::string::npos) {
    return Fail(
        "Phantom CKKS2C source requests forbidden native bootstrap "
        "precomputation",
        diagnostic);
  }
  bool has_relin_call = source.find("Relin(") != std::string::npos;
  bool requests_relin_key =
      source.find("PHANTOM_RESOURCE_RELIN_KEY") != std::string::npos;
  if (has_relin_call && !requests_relin_key) {
    return Fail(
        "Phantom CKKS2C source with Relin requires "
        "PHANTOM_RESOURCE_RELIN_KEY",
        diagnostic);
  }
  if (!has_relin_call && requests_relin_key) {
    return Fail(
        "Phantom CKKS2C source requests "
        "PHANTOM_RESOURCE_RELIN_KEY without a Relin call",
        diagnostic);
  }

  const std::regex rotate_call(
      R"(Rotate_ciph\s*\([^;]*,\s*(-?[0-9]+)\s*\))");
  bool has_nonzero_rotate_call = false;
  for (std::sregex_iterator it(source.begin(), source.end(), rotate_call), end;
       it != end; ++it) {
    if (std::stoll((*it)[1].str()) != 0) {
      has_nonzero_rotate_call = true;
      break;
    }
  }
  const std::regex batch_steps(
      R"(phantom_rotation_batch_steps\s*\[\s*\]\s*=\s*\{([^}]*)\})");
  std::smatch batch_match;
  if (std::regex_search(source, batch_match, batch_steps)) {
    const std::regex integer(R"((-?[0-9]+))");
    const std::string values = batch_match[1].str();
    for (std::sregex_iterator it(values.begin(), values.end(), integer), end;
         it != end; ++it) {
      if (std::stoll((*it)[1].str()) != 0) {
        has_nonzero_rotate_call = true;
        break;
      }
    }
  }
  bool requests_rotation_keys =
      source.find("PHANTOM_RESOURCE_ROTATION_KEYS") != std::string::npos;
  if (has_nonzero_rotate_call && !requests_rotation_keys) {
    return Fail(
        "Phantom CKKS2C source with nonzero Rotate_ciph requires "
        "PHANTOM_RESOURCE_ROTATION_KEYS",
        diagnostic);
  }
  if (!has_nonzero_rotate_call && requests_rotation_keys) {
    return Fail(
        "Phantom CKKS2C source requests PHANTOM_RESOURCE_ROTATION_KEYS "
        "without a nonzero Rotate_ciph call",
        diagnostic);
  }

  const struct {
    const char* call;
    const char* flag;
    const char* description;
  } retained_resources[] = {
      {"Conjugate_ciph(", "PHANTOM_RESOURCE_CONJUGATION_KEY",
       "conjugation"},
      {"Rotate_batch_ciph(", "PHANTOM_RESOURCE_ROTATE_BATCH",
       "rotate_batch"},
      {"Raise_mod(", "PHANTOM_RESOURCE_RAISE_MOD", "raise_mod"},
      {"Mul_mono_ciph(", "PHANTOM_RESOURCE_MONOMIALS", "mul_mono"},
  };
  for (const auto& resource : retained_resources) {
    const bool has_call = source.find(resource.call) != std::string::npos;
    const bool has_flag = source.find(resource.flag) != std::string::npos;
    if (has_call != has_flag) {
      std::ostringstream os;
      os << "Phantom CKKS2C source " << resource.description
         << (has_call ? " call has no matching " : " flag has no matching ")
         << (has_call ? resource.flag : resource.call);
      return Fail(os.str(), diagnostic);
    }
  }

  const std::vector<std::pair<const char*, const char*>> forbidden = {
      {"rt_ant/", "ANT runtime header"},
      {"rt_seal/", "SEAL runtime header"},
      {"LIB_ANT", "ANT provider macro"},
      {"LIB_SEAL", "SEAL provider macro"},
      {"Eval_bootstrap", "opaque ANT bootstrap call"},
      {"Phantom_bootstrap", "native Phantom bootstrap call"},
      {"Bootstrap(", "opaque bootstrap call"},
      {"Bootstrapper", "native Phantom bootstrap class"},
      {"bootstrap_coeffs_to_slots", "bootstrap stage call"},
      {"bootstrap_eval_mod", "bootstrap stage call"},
      {"bootstrap_slots_to_coeffs", "bootstrap stage call"},
      {"Hw_", "POLY/HW runtime call"},
      {"Poly_", "POLY runtime call"},
      {"phantom::", "direct Phantom C++ call"},
      {"Get_phantom_ordinary_features", "legacy feature callback"},
      {"PHANTOM_ORDINARY_", "legacy feature flag"},
      {"_pre_plain_", "function-static plaintext cache"},
      {"CKKS_PARAMS* Get_context_params", "duplicate context callback"},
  };
  for (const auto& item : forbidden) {
    if (source.find(item.first) != std::string::npos) {
      return Fail(std::string("Phantom CKKS2C source contains forbidden ") +
                      item.second + ": " + item.first,
                  diagnostic);
    }
  }
  return true;
}

}  // namespace ckks
}  // namespace fhe
