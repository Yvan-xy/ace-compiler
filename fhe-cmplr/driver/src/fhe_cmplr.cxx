//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/driver/fhe_cmplr.h"

#include <iostream>
#include <string>

using namespace std;

namespace fhe {

namespace driver {

FHE_COMPILER::FHE_COMPILER(bool standalone) : air::driver::DRIVER(standalone) {}

R_CODE FHE_COMPILER::Pre_run() {
  std::string diagnostic;
  if (!_codegen_config.Update_options(&diagnostic)) {
    CMPLR_USR_MSG(U_CODE::Incorrect_Option, diagnostic.c_str());
    return R_CODE::USER;
  }

  R_CODE rc = Get_pass<sihe::SIHE_PASS, PASS_ID::SIHE>().Pre_run();
  if (rc != R_CODE::NORMAL) return rc;
  rc = Get_pass<ckks::CKKS_PASS, PASS_ID::CKKS>().Pre_run();
  if (rc != R_CODE::NORMAL) return rc;

  if (Codegen_ir() == CODEGEN_IR::CKKS) {
    _pass_mgr.Set_pass_enable<PASS_ID::POLY>(false);
    _pass_mgr.Set_pass_enable<PASS_ID::POLY2C>(false);
    rc = Get_pass<ckks::CKKS2C_PASS, PASS_ID::CKKS2C>().Pre_run();
    _pass_mgr.Set_pass_enable<PASS_ID::CKKS2C>(true);
    return rc;
  }

  _pass_mgr.Set_pass_enable<PASS_ID::CKKS2C>(false);
  _pass_mgr.Set_pass_enable<PASS_ID::POLY>(true);
  rc = Get_pass<poly::POLY_PASS, PASS_ID::POLY>().Pre_run();
  if (rc != R_CODE::NORMAL) return rc;
  auto& poly2c = Get_pass<poly::POLY2C_PASS, PASS_ID::POLY2C>();
  rc = poly2c.Pre_run();
  const auto& poly2c_config =
      static_cast<const poly::POLY2C_PASS&>(poly2c).Config();
  if (rc == R_CODE::NORMAL &&
      poly2c_config.Provider() == core::PROVIDER::PHANTOM) {
    CMPLR_USR_MSG(
        U_CODE::Incorrect_Option,
        "Phantom source generation requires -FHE:codegen_ir=ckks and "
        "the dedicated -K2C:lib=phantom terminal");
    return R_CODE::USER;
  }
  _pass_mgr.Set_pass_enable<PASS_ID::POLY2C>(true);
  return rc;
}

R_CODE FHE_COMPILER::Run() { return _pass_mgr.Run(this); }

void FHE_COMPILER::Post_run() {
  if (Codegen_ir() == CODEGEN_IR::CKKS) {
    Get_pass<ckks::CKKS2C_PASS, PASS_ID::CKKS2C>().Post_run();
  } else {
    Get_pass<poly::POLY2C_PASS, PASS_ID::POLY2C>().Post_run();
    if (!Poly_pass_disabled()) {
      Get_pass<poly::POLY_PASS, PASS_ID::POLY>().Post_run();
    }
  }
  Get_pass<ckks::CKKS_PASS, PASS_ID::CKKS>().Post_run();
  Get_pass<sihe::SIHE_PASS, PASS_ID::SIHE>().Post_run();
}

void FHE_COMPILER::Fini() {
  if (Codegen_ir() == CODEGEN_IR::CKKS) {
    Get_pass<ckks::CKKS2C_PASS, PASS_ID::CKKS2C>().Fini();
  } else {
    Get_pass<poly::POLY2C_PASS, PASS_ID::POLY2C>().Fini();
    if (!Poly_pass_disabled()) {
      Get_pass<poly::POLY_PASS, PASS_ID::POLY>().Fini();
    }
  }
  Get_pass<ckks::CKKS_PASS, PASS_ID::CKKS>().Fini();
  Get_pass<sihe::SIHE_PASS, PASS_ID::SIHE>().Fini();
}

}  // namespace driver

}  // namespace fhe
