//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_CKKS2C_DRIVER_H
#define FHE_CKKS_CKKS2C_DRIVER_H

#include <stdexcept>
#include <string>

#include "air/base/visitor.h"
#include "air/core/handler.h"
#include "fhe/cg/ir2c_driver.h"
#include "fhe/ckks/ckks2c_config.h"
#include "fhe/ckks/ckks2c_mfree.h"
#include "fhe/ckks/ckks_handler.h"
#include "fhe/ckks/ir2c_core.h"
#include "fhe/ckks/ir2c_handler.h"
#include "fhe/sihe/ir2c_handler.h"
#include "fhe/sihe/sihe_handler.h"
#include "nn/vector/handler.h"
#include "nn/vector/ir2c_vector.h"

namespace fhe {
namespace ckks {

//! @brief CKKS-owned visitor with no POLY domain handler or context.
using CKKS2C_VISITOR =
    air::base::VISITOR<fhe::ckks::IR2C_CTX,
                       air::core::HANDLER<fhe::ckks::IR2C_CORE>,
                       nn::vector::HANDLER<nn::vector::IR2C_VECTOR>,
                       fhe::sihe::HANDLER<fhe::sihe::IR2C_HANDLER>,
                       fhe::ckks::HANDLER<fhe::ckks::IR2C_HANDLER>>;

//! @brief Direct post-CKKS AIR to runtime-adapter source driver.
class CKKS2C_DRIVER
    : public fhe::cg::IR2C_DRIVER<fhe::ckks::IR2C_CTX> {
  using BASE = fhe::cg::IR2C_DRIVER<fhe::ckks::IR2C_CTX>;

public:
  CKKS2C_DRIVER(std::ostream& os, core::LOWER_CTX& lower_ctx,
                const CKKS2C_CONFIG& cfg)
      : BASE(os, lower_ctx, cfg), _provider(cfg.Provider()) {}

  air::base::GLOB_SCOPE* Flatten(air::base::GLOB_SCOPE* in_scope);

  template <typename VISITOR>
  void Run(air::base::GLOB_SCOPE* glob, VISITOR& visitor) {
    auto prepare_func = [this](air::base::FUNC_SCOPE*,
                               air::base::NODE_PTR body) {
      fhe::ckks::MFREE_PASS mfree(this->Ctx().Lower_ctx());
      mfree.Perform(body);
    };
    auto pre_body_func = [](air::base::FUNC_SCOPE*) {};
    BASE::template Run<VISITOR>(glob, visitor, prepare_func, pre_body_func);
  }

  void Verify_or_throw(air::base::GLOB_SCOPE* glob) const;

  static void Verify_source_or_throw(const std::string& source,
                                     core::PROVIDER provider);

private:
  core::PROVIDER _provider;
};

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_CKKS2C_DRIVER_H
