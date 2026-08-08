//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include <algorithm>
#include <limits>

#include "air/base/container.h"
#include "air/base/st.h"
#include "air/base/visitor.h"
#include "air/core/handler.h"
#include "air/opt/ssa_build.h"
#include "air/opt/ssa_container.h"
#include "air/util/error.h"
#include "ckks_stats.h"
#include "fhe/ckks/ckks_gen.h"
#include "fhe/ckks/sihe2ckks_lower.h"
#include "fhe/core/ctx_param_ana.h"
#include "fhe/sihe/sihe_handler.h"
#include "resbm.h"
#include "rt_validate_scale_level.h"
#include "scale_manager.h"

using namespace air::base;

namespace fhe {
namespace ckks {

namespace {

void Collect_raise_targets(NODE_PTR node, uint32_t* maximum) {
  if (node == Null_ptr) return;
  if (node->Opcode() == OPC_RAISE_MOD && node->Num_child() == 2 &&
      node->Child(1) != Null_ptr &&
      node->Child(1)->Opcode() == air::core::OPC_INTCONST) {
    const int64_t target = node->Child(1)->Intconst();
    if (target > 0 &&
        static_cast<uint64_t>(target) <= std::numeric_limits<uint32_t>::max()) {
      *maximum = std::max(*maximum, static_cast<uint32_t>(target));
    }
  }
  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Collect_raise_targets(stmt->Node(), maximum);
    }
    return;
  }
  for (uint32_t index = 0; index < node->Num_child(); ++index) {
    Collect_raise_targets(node->Child(index), maximum);
  }
}

R_CODE Establish_full_q_count(GLOB_SCOPE* glob, core::LOWER_CTX* lower_ctx,
                              const air::driver::DRIVER_CTX* driver_ctx,
                              const CKKS_CONFIG*             config) {
  uint32_t maximum_raise_target = 0;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
       it != glob->End_func_scope(); ++it) {
    Collect_raise_targets((*it).Container().Entry_node(),
                          &maximum_raise_target);
  }

  core::CTX_PARAM& parameters = lower_ctx->Get_ctx_param();
  const int64_t    configured = config->Max_cipher_lvl();
  if (configured < 0) {
    CMPLR_ERR_MSG(driver_ctx->Tfile(),
                  "configured maximum ciphertext level must be nonnegative\n");
    return R_CODE::USER;
  }
  if (configured > 0) {
    if (static_cast<uint64_t>(configured) >
        std::numeric_limits<uint32_t>::max()) {
      CMPLR_ERR_MSG(driver_ctx->Tfile(),
                    "configured maximum ciphertext level is out of range\n");
      return R_CODE::USER;
    }
    const uint32_t configured_full_q = static_cast<uint32_t>(configured);
    if (parameters.Get_mul_level() > configured_full_q) {
      CMPLR_ERR_MSG(
          driver_ctx->Tfile(),
          "configured maximum ciphertext level is less than the inferred "
          "full data-Q count: %u\n",
          parameters.Get_mul_level());
      return R_CODE::USER;
    }
    if (maximum_raise_target > configured_full_q) {
      CMPLR_ERR_MSG(
          driver_ctx->Tfile(),
          "raise_mod target_q_count exceeds configured full data-Q count: "
          "%u > %u\n",
          maximum_raise_target, configured_full_q);
      return R_CODE::USER;
    }
    parameters.Set_mul_level(configured_full_q, true);
    return R_CODE::NORMAL;
  }

  // With no configured mcl, infer the full-Q count from the largest valid
  // constant raise target before scale metadata is assigned.
  parameters.Set_mul_level(maximum_raise_target, true);
  return R_CODE::NORMAL;
}

}  // namespace

GLOB_SCOPE* Ckks_driver(GLOB_SCOPE* glob, core::LOWER_CTX* lower_ctx,
                        const air::driver::DRIVER_CTX* driver_ctx,
                        const CKKS_CONFIG* config, R_CODE* result) {
  if (result != nullptr) *result = R_CODE::NORMAL;
  GLOB_SCOPE* new_glob = new GLOB_SCOPE(glob->Id(), true);
  AIR_ASSERT(new_glob != nullptr);
  new_glob->Clone(*glob, true);

  // update hamming_weight of CTX_PARAMS with option
  lower_ctx->Get_ctx_param().Set_hamming_weight(config->Hamming_weight());

  SIHE2CKKS_LOWER sihe2ckks_lower(new_glob, lower_ctx, config);
  if (config->Rgn_scl_bts_mng() || config->Rgn_bts_mng()) {
    // 1. lower SIHE to CKKS domain
    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
         it != glob->End_func_scope(); ++it) {
      FUNC_SCOPE* func = &(*it);
      sihe2ckks_lower.Lower_server_func(func);
    }
    R_CODE establish_result =
        Establish_full_q_count(new_glob, lower_ctx, driver_ctx, config);
    if (establish_result != R_CODE::NORMAL) {
      if (result != nullptr) *result = establish_result;
      delete new_glob;
      return nullptr;
    }
    // 2. perform modular level pass RESBM to insert required scale/level
    // management operations, like bootstrap/rescale/modswitch
    RESBM resbm(driver_ctx, config, new_glob, lower_ctx);
    resbm.Perform();
    // 3. perform other function level pass
    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = new_glob->Begin_func_scope();
         it != new_glob->End_func_scope(); ++it) {
      FUNC_SCOPE*   ckks_func = &(*it);
      SCALE_MANAGER scale_mngr(driver_ctx, config, ckks_func, lower_ctx);
      scale_mngr.Run();
      core::CTX_PARAM_ANA ctx_param_ana(ckks_func, lower_ctx, driver_ctx,
                                        config);
      R_CODE              analysis_result = ctx_param_ana.Run();
      if (analysis_result != R_CODE::NORMAL) {
        if (result != nullptr) *result = analysis_result;
        delete new_glob;
        return nullptr;
      }
    }
  } else {
    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
         it != glob->End_func_scope(); ++it) {
      FUNC_SCOPE* func = &(*it);
      sihe2ckks_lower.Lower_server_func(func);
    }

    R_CODE establish_result =
        Establish_full_q_count(new_glob, lower_ctx, driver_ctx, config);
    if (establish_result != R_CODE::NORMAL) {
      if (result != nullptr) *result = establish_result;
      delete new_glob;
      return nullptr;
    }

    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = new_glob->Begin_func_scope();
         it != new_glob->End_func_scope(); ++it) {
      FUNC_SCOPE*   ckks_func = &(*it);
      SCALE_MANAGER scale_mngr(driver_ctx, config, ckks_func, lower_ctx);
      scale_mngr.Run();

      core::CTX_PARAM_ANA ctx_param_ana(ckks_func, lower_ctx, driver_ctx,
                                        config);
      R_CODE              analysis_result = ctx_param_ana.Run();
      if (analysis_result != R_CODE::NORMAL) {
        if (result != nullptr) *result = analysis_result;
        delete new_glob;
        return nullptr;
      }
    }
  }
  delete glob;

  if (config->Trace_stat() || config->Per_op_trace_stat()) {
    CKKS_STATS stats(*new_glob, *lower_ctx, config->Per_op_trace_stat());
    typedef air::base::OP_STATS_CTX<CKKS_STATS> STATS_CTX;
    STATS_CTX                                   stats_ctx(stats);
    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = new_glob->Begin_func_scope();
         it != new_glob->End_func_scope(); ++it) {
      FUNC_SCOPE*                   func = &(*it);
      air::base::VISITOR<STATS_CTX> trav(stats_ctx);
      trav.Visit<void>(func->Container().Entry_node());
    }
    stats.Fixup_call_stats();
    stats.Print(driver_ctx->Tstream());
  }

  if (config->Rt_val_scl_lvl()) {
    RT_VAL_SCL_LVL val_bts_smo(driver_ctx, config, new_glob, lower_ctx);
    val_bts_smo.Run();
  }
  return new_glob;
}  // Ckks_driver

}  // namespace ckks
}  // namespace fhe
