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
      node->Child(child_index) == air::base::Null_ptr ||
      node->Child(child_index)->Opcode() != air::core::OPC_INTCONST) {
    return false;
  }
  *value = node->Child(child_index)->Intconst();
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

bool Verify_phantom_encode(NODE_PTR node,
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

bool Verify_phantom_binary(NODE_PTR node, CKKS_OPERATOR op,
                           bool strict_ordinary_metadata,
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
  if (strict_ordinary_metadata &&
      (op == CKKS_OPERATOR::ADD || op == CKKS_OPERATOR::SUB) &&
      rhs_kind != OPERAND_KIND::FLOAT_SCALAR) {
    return Verify_static_add_sub_metadata(node, op, diagnostic);
  }
  return true;
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

bool Verify_ckks_node(NODE_PTR node,
                      const PHANTOM_CONTEXT_DESCRIPTOR& context,
                      core::PROVIDER provider,
                      bool strict_ordinary_metadata,
                      std::string* diagnostic) {
  CKKS_OPERATOR op = static_cast<CKKS_OPERATOR>(node->Operator());
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
        return Verify_phantom_encode(node, context, diagnostic);
      case CKKS_OPERATOR::ADD:
      case CKKS_OPERATOR::SUB:
      case CKKS_OPERATOR::MUL:
        return Verify_phantom_binary(node, op, strict_ordinary_metadata,
                                     diagnostic);
      case CKKS_OPERATOR::ROTATE:
        return Verify_phantom_rotate(node, diagnostic);
      case CKKS_OPERATOR::RELIN:
        return Verify_phantom_relin(node, diagnostic);
      case CKKS_OPERATOR::RESCALE:
      case CKKS_OPERATOR::MODSWITCH:
        return Verify_phantom_modulus_drop(node, op, diagnostic);
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
    case CKKS_OPERATOR::CONJUGATE:
    case CKKS_OPERATOR::FREE:
      return true;

    case CKKS_OPERATOR::ROTATE_BATCH: {
      uint32_t count = 0;
      const int* rotations =
          node->Attr<int>(nn::core::ATTR::RNUM, &count);
      if (rotations == nullptr || count == 0 || !node->Rtype()->Is_array() ||
          node->Rtype()->Cast_to_arr()->Elem_count() != count) {
        return Fail(
            "CKKS2C rotate_batch requires a non-empty ordered RNUM attribute "
            "matching the result array length",
            diagnostic);
      }
      return true;
    }

    case CKKS_OPERATOR::RAISE_MOD:
      if (node->Num_child() != 2 ||
          node->Child(1)->Opcode() != air::core::OPC_INTCONST) {
        return Fail(
            "CKKS2C raise_mod requires a constant target_q_count operand",
            diagnostic);
      }
      return true;

    case CKKS_OPERATOR::MUL_MONO:
      if (node->Num_child() != 2 ||
          node->Child(1)->Opcode() != air::core::OPC_INTCONST) {
        return Fail("CKKS2C mul_mono requires a constant monomial power",
                    diagnostic);
      }
      return true;

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

bool Verify_node(NODE_PTR node,
                 const PHANTOM_CONTEXT_DESCRIPTOR& context,
                 core::PROVIDER provider,
                 bool strict_ordinary_metadata,
                 std::string* diagnostic) {
  if (node == air::base::Null_ptr) {
    return true;
  }
  if (node->Domain() == fhe::poly::POLYNOMIAL_DID) {
    return Fail("CKKS2C input contains forbidden POLY AIR", diagnostic);
  }
  if (node->Domain() == CKKS_DOMAIN::ID &&
      !Verify_ckks_node(node, context, provider, strict_ordinary_metadata,
                        diagnostic)) {
    return false;
  }
  if (node->Is_block()) {
    for (air::base::STMT_PTR stmt = node->Begin_stmt();
         stmt != node->End_stmt(); stmt = stmt->Next()) {
      if (!Verify_node(stmt->Node(), context, provider,
                       strict_ordinary_metadata, diagnostic)) {
        return false;
      }
    }
    return true;
  }
  for (uint32_t i = 0; i < node->Num_child(); ++i) {
    if (!Verify_node(node->Child(i), context, provider,
                     strict_ordinary_metadata, diagnostic)) {
      return false;
    }
  }
  return true;
}

bool Contains_retained_operator(NODE_PTR node) {
  if (node == air::base::Null_ptr) {
    return false;
  }
  if (node->Domain() == CKKS_DOMAIN::ID) {
    CKKS_OPERATOR op = static_cast<CKKS_OPERATOR>(node->Operator());
    if (op == CKKS_OPERATOR::CONJUGATE ||
        op == CKKS_OPERATOR::ROTATE_BATCH ||
        op == CKKS_OPERATOR::RAISE_MOD ||
        op == CKKS_OPERATOR::MUL_MONO) {
      return true;
    }
  }
  if (node->Is_block()) {
    for (air::base::STMT_PTR stmt = node->Begin_stmt();
         stmt != node->End_stmt(); stmt = stmt->Next()) {
      if (Contains_retained_operator(stmt->Node())) {
        return true;
      }
    }
    return false;
  }
  for (uint32_t i = 0; i < node->Num_child(); ++i) {
    if (Contains_retained_operator(node->Child(i))) {
      return true;
    }
  }
  return false;
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
    // Retained operators are preserved only for the earlier source-generation
    // gate. Their scale/level flow is intentionally not qualified until the
    // later retained-operator work; ordinary-only functions remain strict.
    bool strict_ordinary_metadata = !Contains_retained_operator(entry);
    if (!Verify_node(entry, context, provider, strict_ordinary_metadata,
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
