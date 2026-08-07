//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_CKKS2C_CONFIG_H
#define FHE_CKKS_CKKS2C_CONFIG_H

#include "air/driver/driver_ctx.h"
#include "fhe/cg/ir2c_config.h"

namespace fhe {
namespace ckks {

//! @brief Configuration for direct CKKS AIR source generation.
struct CKKS2C_CONFIG : public fhe::cg::IR2C_CONFIG {
public:
  CKKS2C_CONFIG() : fhe::cg::IR2C_CONFIG("phantom") {}

  void Register_options(air::driver::DRIVER_CTX* ctx);
  void Update_options();
  void Print(std::ostream& os) const;
};

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_CKKS2C_CONFIG_H
