//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "ckks2c_verifier.h"

#include <sstream>
#include <utility>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/base/st_type.h"
#include "air/core/opcode.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/poly/opcode.h"
#include "nn/core/attr.h"

namespace fhe {
namespace ckks {
namespace {

using air::base::NODE_PTR;

bool Fail(const std::string& message, std::string* diagnostic) {
  if (diagnostic != nullptr) {
    *diagnostic = message;
  }
  return false;
}

bool Verify_ckks_node(NODE_PTR node, core::PROVIDER provider,
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

bool Verify_node(NODE_PTR node, core::PROVIDER provider,
                 std::string* diagnostic) {
  if (node == air::base::Null_ptr) {
    return true;
  }
  if (node->Domain() == fhe::poly::POLYNOMIAL_DID) {
    return Fail("CKKS2C input contains forbidden POLY AIR", diagnostic);
  }
  if (node->Domain() == CKKS_DOMAIN::ID &&
      !Verify_ckks_node(node, provider, diagnostic)) {
    return false;
  }
  if (node->Is_block()) {
    for (air::base::STMT_PTR stmt = node->Begin_stmt();
         stmt != node->End_stmt(); stmt = stmt->Next()) {
      if (!Verify_node(stmt->Node(), provider, diagnostic)) {
        return false;
      }
    }
    return true;
  }
  for (uint32_t i = 0; i < node->Num_child(); ++i) {
    if (!Verify_node(node->Child(i), provider, diagnostic)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool CKKS2C_VERIFIER::Verify(air::base::GLOB_SCOPE* glob,
                             core::PROVIDER provider,
                             std::string* diagnostic) {
  if (diagnostic != nullptr) {
    diagnostic->clear();
  }
  if (glob == nullptr) {
    return Fail("CKKS2C received no AIR module", diagnostic);
  }
  for (air::base::GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
       it != glob->End_func_scope(); ++it) {
    if (!Verify_node((*it).Container().Entry_node(), provider, diagnostic)) {
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
  if (source.find("LIB_PHANTOM") == std::string::npos) {
    return Fail("Phantom CKKS2C source is missing LIB_PHANTOM", diagnostic);
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
