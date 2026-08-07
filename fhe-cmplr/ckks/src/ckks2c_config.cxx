//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/ckks/ckks2c_config.h"

using namespace air::util;

namespace fhe {
namespace ckks {

static CKKS2C_CONFIG Ckks2c_config;

static OPTION_DESC Ckks2c_option[] = {
    DECLARE_COMMON_CONFIG(Ckks2c_config),
    {"lib", "", "Runtime adapter used by direct CKKS codegen: phantom or ant",
     &Ckks2c_config._prov_str, K_STR, 0, V_EQUAL},
    {"df", "data_file", "Store weight data in a separate file",
     &Ckks2c_config._data_file, K_STR, 0, V_EQUAL},
};

static OPTION_DESC_HANDLE Ckks2c_option_handle = {
    sizeof(Ckks2c_option) / sizeof(Ckks2c_option[0]), Ckks2c_option};

static OPTION_GRP Ckks2c_option_grp = {
    "K2C", "Translate CKKS AIR directly to runtime-adapter source", ':',
    V_EQUAL, &Ckks2c_option_handle};

void CKKS2C_CONFIG::Register_options(air::driver::DRIVER_CTX* ctx) {
  ctx->Register_option_group(&Ckks2c_option_grp);
}

void CKKS2C_CONFIG::Update_options() {
  *this = Ckks2c_config;
  _provider = core::Provider_id(_prov_str.c_str());
}

void CKKS2C_CONFIG::Print(std::ostream& os) const { COMMON_CONFIG::Print(os); }

}  // namespace ckks
}  // namespace fhe
