//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_CKKS2C_PASS_H
#define FHE_CKKS_CKKS2C_PASS_H

#include "air/driver/pass.h"
#include "fhe/ckks/ckks2c_config.h"

namespace fhe {

namespace driver {
class FHE_COMPILER;
}  // namespace driver

namespace ckks {

//! @brief Terminal pass that emits source directly from post-driver CKKS AIR.
class CKKS2C_PASS : public air::driver::PASS<CKKS2C_CONFIG> {
public:
  CKKS2C_PASS();

  R_CODE Init(driver::FHE_COMPILER* driver);
  R_CODE Pre_run();
  R_CODE Run();
  void   Post_run();
  void   Fini();

  const char* Name() const { return "CKKS2C"; }

private:
  bool Check_provider_supported(core::PROVIDER provider) const;

  driver::FHE_COMPILER* _driver;
};

}  // namespace ckks
}  // namespace fhe

#endif  // FHE_CKKS_CKKS2C_PASS_H
