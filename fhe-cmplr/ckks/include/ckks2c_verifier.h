//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_CKKS2C_VERIFIER_H
#define FHE_CKKS_CKKS2C_VERIFIER_H

#include <string>

#include "air/base/container.h"
#include "fhe/core/lib_provider.h"

namespace fhe {
namespace ckks {

//! @brief Post-CKKS AIR and generated-source legality checks for CKKS2C.
class CKKS2C_VERIFIER {
public:
  static bool Verify(air::base::GLOB_SCOPE* glob, core::PROVIDER provider,
                     std::string* diagnostic);

  static bool Verify_source(const std::string& source,
                            core::PROVIDER provider,
                            std::string* diagnostic);
};

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_CKKS2C_VERIFIER_H
