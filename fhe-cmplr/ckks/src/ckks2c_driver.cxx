//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/ckks/ckks2c_driver.h"

#include "air/core/opcode.h"
#include "ckks2c_verifier.h"
#include "nn/vector/vector_opcode.h"

namespace fhe {
namespace ckks {

air::base::GLOB_SCOPE* CKKS2C_DRIVER::Flatten(
    air::base::GLOB_SCOPE* glob) {
  auto should_flatten = [](air::base::NODE_PTR node) {
    if (node->Domain() == air::core::CORE ||
        node->Opcode() == nn::vector::OPC_SLICE) {
      return false;
    }
    return true;
  };
  return BASE::Flatten(glob, should_flatten);
}

void CKKS2C_DRIVER::Verify_or_throw(air::base::GLOB_SCOPE* glob) const {
  std::string diagnostic;
  if (!CKKS2C_VERIFIER::Verify(glob, _provider, &diagnostic)) {
    throw std::runtime_error(diagnostic);
  }
}

void CKKS2C_DRIVER::Verify_source_or_throw(const std::string& source,
                                           core::PROVIDER provider) {
  std::string diagnostic;
  if (!CKKS2C_VERIFIER::Verify_source(source, provider, &diagnostic)) {
    throw std::runtime_error(diagnostic);
  }
}

}  // namespace ckks
}  // namespace fhe
