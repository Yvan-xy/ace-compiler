//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_POLY_POLY2C_CONFIG_H
#define FHE_POLY_POLY2C_CONFIG_H

#include "air/driver/driver_ctx.h"
#include "fhe/cg/ir2c_config.h"

namespace fhe {
namespace poly {

struct POLY2C_CONFIG : public fhe::cg::IR2C_CONFIG {
public:
  POLY2C_CONFIG(void) : fhe::cg::IR2C_CONFIG("ant"), _decomp_ntt_threads(0), _evalmod_schedule(0) {}

  // 0: legacy entry; 1: new serial entry; >1: bounded limb parallelism.
  uint64_t Decomp_ntt_threads() const { return _decomp_ntt_threads; }
  uint64_t _decomp_ntt_threads;
  uint64_t _evalmod_schedule; // 0 legacy, 1 operator tasks, 2 branch+operator tasks
  uint64_t Evalmod_schedule() const { return _evalmod_schedule; }

  void Register_options(air::driver::DRIVER_CTX* ctx);
  void Update_options();
  void Print(std::ostream& os) const;
};

//! @brief Macro to define API to access POLY2C config
#define DECLARE_POLY2C_CONFIG_ACCESS_API(cfg) \
  DECLARE_IR2C_CONFIG_ACCESS_API(cfg)

}  // namespace poly
}  // namespace fhe

#endif  // FHE_POLY_POLY2C_CONFIG_H
