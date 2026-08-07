//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "fhe/ckks/ckks2c_pass.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "fhe/ckks/ckks2c_driver.h"
#include "fhe/driver/fhe_cmplr.h"

namespace fhe {
namespace ckks {
namespace {

std::string Cuda_output_name(driver::FHE_COMPILER* driver) {
  std::string output(driver->Context()->Ofile());
  if (output.empty()) {
    output = driver->Context()->Def_cfile();
  }
  if (output.size() >= 2 && output.compare(output.size() - 2, 2, ".c") == 0) {
    output.replace(output.size() - 2, 2, ".cu");
  } else if (output.size() < 3 ||
             output.compare(output.size() - 3, 3, ".cu") != 0) {
    output.append(".cu");
  }
  return output;
}

}  // namespace

CKKS2C_PASS::CKKS2C_PASS() : _driver(nullptr) {}

R_CODE CKKS2C_PASS::Init(driver::FHE_COMPILER* driver) {
  _driver = driver;
  _config.Register_options(driver->Context());
  return R_CODE::NORMAL;
}

R_CODE CKKS2C_PASS::Pre_run() {
  _config.Update_options();
  if (!Check_provider_supported(_config.Provider())) {
    return R_CODE::USER;
  }
  if (_config.Provider() == core::PROVIDER::PHANTOM && _config.Ct_encode()) {
    CMPLR_USR_MSG(U_CODE::Incorrect_Option,
                  "Phantom CKKS2C does not support compile-time encoding");
    return R_CODE::USER;
  }
  _config.Set_ifile(_driver->Context()->Ifile());
  return R_CODE::NORMAL;
}

R_CODE CKKS2C_PASS::Run() {
  try {
    air::base::GLOB_SCOPE* glob = _driver->Glob_scope();
    std::ostringstream     output;
    CKKS2C_DRIVER ckks2c(output, _driver->Lower_ctx(), _config);

    ckks2c.Verify_or_throw(glob);
    glob = ckks2c.Flatten(glob);
    _driver->Update_glob_scope(glob);
    CKKS2C_VISITOR visitor(ckks2c.Ctx());
    ckks2c.Run(glob, visitor);

    std::string source = output.str();
    CKKS2C_DRIVER::Verify_source_or_throw(source, _config.Provider());
    std::string output_name = Cuda_output_name(_driver);
    std::ofstream file(output_name);
    if (!file.is_open()) {
      CMPLR_USR_MSG(U_CODE::Output_File_Open_Err, output_name.c_str());
      return R_CODE::USER;
    }
    file << source;
    if (!file.good()) {
      throw std::runtime_error("failed while writing CKKS2C output " +
                               output_name);
    }
    return R_CODE::NORMAL;
  } catch (const std::exception& error) {
    CMPLR_USR_MSG(U_CODE::Incorrect_Option, error.what());
    return R_CODE::USER;
  }
}

void CKKS2C_PASS::Post_run() {}

void CKKS2C_PASS::Fini() {}

bool CKKS2C_PASS::Check_provider_supported(core::PROVIDER provider) const {
  if (provider == core::PROVIDER::PHANTOM ||
      provider == core::PROVIDER::ANT) {
    return true;
  }
  CMPLR_USR_MSG(U_CODE::Incorrect_Option,
                "CKKS2C supports only -K2C:lib=phantom|ant");
  return false;
}

}  // namespace ckks
}  // namespace fhe
