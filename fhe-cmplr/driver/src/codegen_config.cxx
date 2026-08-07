//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/driver/codegen_config.h"

#include "air/util/option.h"

namespace fhe {
namespace driver {

using namespace air::util;

static CODEGEN_CONFIG Codegen_config;

static OPTION_DESC Codegen_option[] = {
    {"codegen_ir", "",
     "Terminal FHE source IR: ckks (direct CKKS2C) or poly (POLY2C)",
     &Codegen_config._codegen_ir_str, K_STR, 0, V_EQUAL},
};

static OPTION_DESC_HANDLE Codegen_option_handle = {
    sizeof(Codegen_option) / sizeof(Codegen_option[0]), Codegen_option};

static OPTION_GRP Codegen_option_grp = {
    "FHE", "FHE pipeline configuration", ':', V_EQUAL,
    &Codegen_option_handle};

void CODEGEN_CONFIG::Register_options(air::driver::DRIVER_CTX* ctx) {
  ctx->Register_option_group(&Codegen_option_grp);
}

bool CODEGEN_CONFIG::Update_options(std::string* diagnostic) {
  _codegen_ir_str = Codegen_config._codegen_ir_str;
  if (_codegen_ir_str == "ckks") {
    _codegen_ir = CODEGEN_IR::CKKS;
    return true;
  }
  if (_codegen_ir_str == "poly") {
    _codegen_ir = CODEGEN_IR::POLY;
    return true;
  }
  _codegen_ir = CODEGEN_IR::INVALID;
  if (diagnostic != nullptr) {
    *diagnostic = "-FHE:codegen_ir=" + _codegen_ir_str +
                  " is invalid; expected ckks or poly";
  }
  return false;
}

}  // namespace driver
}  // namespace fhe
