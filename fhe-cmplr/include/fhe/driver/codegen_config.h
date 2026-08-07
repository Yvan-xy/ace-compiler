//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_DRIVER_CODEGEN_CONFIG_H
#define FHE_DRIVER_CODEGEN_CONFIG_H

#include <string>

#include "air/driver/driver_ctx.h"

namespace fhe {
namespace driver {

enum class CODEGEN_IR : uint32_t { CKKS, POLY, INVALID };

//! @brief Provider-independent selection of the one terminal codegen path.
struct CODEGEN_CONFIG {
  CODEGEN_CONFIG() : _codegen_ir_str("poly"), _codegen_ir(CODEGEN_IR::POLY) {}

  void Register_options(air::driver::DRIVER_CTX* ctx);
  bool Update_options(std::string* diagnostic);

  CODEGEN_IR  Codegen_ir() const { return _codegen_ir; }
  const char* Codegen_ir_str() const { return _codegen_ir_str.c_str(); }

  std::string _codegen_ir_str;

private:
  CODEGEN_IR _codegen_ir;
};

}  // namespace driver
}  // namespace fhe

#endif  // FHE_DRIVER_CODEGEN_CONFIG_H
