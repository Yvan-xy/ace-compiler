//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CG_IR2C_DRIVER_H
#define FHE_CG_IR2C_DRIVER_H

#include <cstdint>
#include <functional>
#include <set>
#include <utility>

#include "air/base/container.h"
#include "air/base/flatten_ctx.h"
#include "air/base/st.h"
#include "air/base/visitor.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/core/lib_provider.h"
#include "fhe/core/rt_data_def.h"
#include "fhe/core/rt_timing.h"
#include "nn/core/data_scheme.h"

namespace fhe {
namespace cg {

//! @brief Provider-neutral module traversal and runtime metadata emission.
template <typename CTX>
class IR2C_DRIVER {
public:
  template <typename CONFIG>
  IR2C_DRIVER(std::ostream& os, core::LOWER_CTX& lower_ctx,
              const CONFIG& cfg)
      : _ctx(os, lower_ctx, cfg) {}

  template <typename FLATTEN_PREDICATE>
  air::base::GLOB_SCOPE* Flatten(air::base::GLOB_SCOPE* glob,
                                 FLATTEN_PREDICATE&& should_flatten) {
    using namespace air::base;
    GLOB_SCOPE* new_glob = new GLOB_SCOPE(glob->Id(), true);
    AIR_ASSERT(new_glob != nullptr);
    new_glob->Clone(*glob);

    for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
         it != glob->End_func_scope(); ++it) {
      FUNC_SCOPE* func     = &(*it);
      FUNC_SCOPE* new_func = &new_glob->New_func_scope(func->Id());
      new_func->Clone(*func);
      CONTAINER& cntr = new_func->Container();

      std::function<bool(NODE_PTR)> flatten_func = should_flatten;
      FLATTEN_CTX<TRANSFORM_UTIL> trav_ctx(&cntr, std::move(flatten_func));
      VISITOR<FLATTEN_CTX<TRANSFORM_UTIL>> trav(trav_ctx);
      NODE_PTR entry = func->Container().Entry_node();
      NODE_PTR retv  = trav.template Visit<NODE_PTR>(entry);
      AIR_ASSERT(retv->Is_entry());
      new_func->Set_entry_stmt(retv->Stmt());
    }

    delete glob;
    return new_glob;
  }

  template <typename VISITOR, typename PREPARE_FUNC, typename PRE_BODY_FUNC>
  void Run(air::base::GLOB_SCOPE* glob, VISITOR& visitor,
           PREPARE_FUNC&& prepare_func, PRE_BODY_FUNC&& pre_body_func) {
    _ctx.Emit_global_include();
    _ctx.Emit_global_constants(glob, true);

    for (air::base::FUNC_ITER it = glob->Begin_func(); it != glob->End_func();
         ++it) {
      if ((*it)->Entry_point()->Is_program_entry()) {
        continue;
      }
      _ctx.Emit_func_sig((*it));
      _ctx << ";\n";
    }
    _ctx << "\n";

    for (air::base::GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
         it != glob->End_func_scope(); ++it) {
      air::base::FUNC_SCOPE* func = &(*it);
      air::base::NODE_PTR body =
          func->Container().Stmt_list().Block_node();

      prepare_func(func, body);

      _ctx.Emit_func_def(func);
      _ctx.Begin_func_body(body);
      const core::LOWER_CTX& lower_ctx = _ctx.Lower_ctx();
      if (func->Id() ==
          lower_ctx.Get_func_info(core::FHE_FUNC::ROTATE).Get_func_id()) {
        _ctx << "  RTLIB_TM_START(" << (uint32_t)(core::RTM_FHE_ROTATE)
             << ", rtm);\n";
      } else if (func->Id() ==
                 lower_ctx.Get_func_info(core::FHE_FUNC::RELIN).Get_func_id()) {
        _ctx << "  RTLIB_TM_START(" << (uint32_t)(core::RTM_FHE_RELIN)
             << ", rtm);\n";
      }
      _ctx.Emit_local_var(func);
      pre_body_func(func);

      visitor.template Visit<void>(body);

      _ctx.End_func_body(body);
      _ctx << "\n";

      if (func->Owning_func()->Entry_point()->Is_program_entry()) {
        Emit_helper_function(func);
      }
    }

    Emit_get_context_params();
    _ctx.Emit_need_bts();
    _ctx.Emit_global_constants(glob, false);
  }

  CTX& Ctx() { return _ctx; }

private:
  void Emit_get_context_params() {
    const core::CTX_PARAM& param = _ctx.Lower_ctx().Get_ctx_param();
    const std::set<int32_t>& rot_keys = param.Get_rotate_index();
    _ctx << "CKKS_PARAMS* ";
    _ctx.Emit_identifier(_ctx.Function_name_prefix());
    _ctx << "Get_context_params() {\n";
    _ctx << "  static CKKS_PARAMS parm = {\n";
    _ctx << "    " << fhe::core::Provider_name(_ctx.Provider()) << ", ";
    _ctx << param.Get_poly_degree() << ", ";
    _ctx << param.Get_security_level() << ", ";
    uint32_t mul_level = param.Get_mul_level();
    AIR_ASSERT_MSG(mul_level >= 1, "mul_level must be at least 1.");
    _ctx << (mul_level - 1) << ", ";
    _ctx << param.Get_input_level() << ", ";
    _ctx << param.Get_first_prime_bit_num() << ", ";
    _ctx << param.Get_scaling_factor_bit_num() << ", ";
    _ctx << param.Get_q_part_num() << ", ";
    _ctx << param.Get_hamming_weight() << ", ";
    _ctx << rot_keys.size() << ", \n";
    _ctx << "    { ";
    int i = 0;
    for (auto it = rot_keys.begin(); it != rot_keys.end(); ++it) {
      if (i > 0) {
        _ctx << (((i % 8) == 0) ? ",\n      " : ", ");
      }
      _ctx << (*it);
      ++i;
    }
    _ctx << " }\n";
    _ctx << "  };\n";
    _ctx << "  return &parm;\n";
    _ctx << "}\n\n";

    _ctx << "RT_DATA_INFO* ";
    _ctx.Emit_identifier(_ctx.Function_name_prefix());
    _ctx << "Get_rt_data_info() {\n";
    if (_ctx.Emit_data_file()) {
      _ctx << "  static RT_DATA_INFO info = {\n";
      _ctx << "    \"" << _ctx.Data_file() << "\",\n";
      _ctx << "    \"" << _ctx.Data_file_uuid() << "\",\n";
      _ctx << "    " << core::Data_entry_name(_ctx.Data_entry_type()) << "\n";
      _ctx << "  };\n";
      _ctx << "  return &info;\n";
    } else {
      _ctx << "  return NULL;\n";
    }
    _ctx << "}\n\n";
  }

  uint32_t Emit_chunk_info(air::base::NODE_PTR node, uint32_t idx) {
    uint32_t num_chunk = 1;
    const nn::core::DATA_CHUNK* chunk =
        nn::core::Data_scheme_attr(node, &num_chunk);
    _ctx << "  static MAP_DESC desc_" << idx << "[] = {\n";
    if (chunk != nullptr) {
      for (uint32_t i = 0; i < num_chunk; ++i) {
        _ctx << "    " << chunk[i].To_str() << ",\n";
      }
    } else {
      _ctx << "    {NORMAL, 0, 0, 0, 0}\n";
    }
    _ctx << "  };\n";
    return num_chunk;
  }

  void Emit_data_shape(air::base::NODE_PTR node) {
    uint32_t dim = 0;
    const int64_t* shape = nn::core::Data_shape_attr(node, &dim);
    if (shape == nullptr) {
      _ctx << "{0, 0, 0, 0}, ";
    } else if (dim == 4) {
      _ctx << "{" << shape[0] << ", " << shape[1] << ", " << shape[2]
           << ", " << shape[3] << "}, ";
    } else if (dim == 2) {
      _ctx << "{" << shape[0] << ", " << shape[1] << ", 0, 0}, ";
    } else {
      AIR_ASSERT(false);
    }
  }

  void Emit_helper_function(air::base::FUNC_SCOPE* func_scope) {
    using namespace air::base;
    NODE_PTR entry = func_scope->Container().Entry_stmt()->Node();
    uint32_t parm_count = entry->Num_child() - 1;
    _ctx << "int Get_input_count() {\n";
    _ctx << "  return " << parm_count << ";\n";
    _ctx << "}\n\n";

    _ctx << "DATA_SCHEME* Get_encode_scheme(int idx) {\n";
    for (uint32_t i = 0; i < parm_count; ++i) {
      NODE_PTR formal = entry->Child(i);
      uint32_t num_chunk = Emit_chunk_info(formal, i);
      _ctx << "  static DATA_SCHEME scheme_" << i << " = {\n";
      ADDR_DATUM_PTR parm = func_scope->Formal(i);
      _ctx << "    \"" << parm->Name()->Char_str() << "\", ";
      Emit_data_shape(formal);
      _ctx << num_chunk << ", desc_" << i << "\n";
      _ctx << "  };\n";
    }
    _ctx << "  static DATA_SCHEME* scheme[] = { ";
    for (uint32_t i = 0; i < parm_count; ++i) {
      if (i > 0) {
        _ctx << ", ";
      }
      _ctx << "&scheme_" << i;
    }
    _ctx << " };\n";
    _ctx << "  return scheme[idx];\n";
    _ctx << "}\n\n";

    _ctx << "int Get_output_count() {\n";
    _ctx << "  return 1;\n";
    _ctx << "}\n\n";

    _ctx << "DATA_SCHEME* Get_decode_scheme(int idx) {\n";
    STMT_LIST sl(entry->Last_child());
    NODE_PTR retv = sl.Last_stmt()->Node();
    AIR_ASSERT(retv->Opcode() == air::core::OPC_RETV);
    uint32_t num_chunk = Emit_chunk_info(retv, 0);
    _ctx << "  static DATA_SCHEME scheme = {\n";
    _ctx << "    \"" << _ctx.Output_name() << "\", ";
    Emit_data_shape(retv);
    _ctx << num_chunk << ", desc_0\n";
    _ctx << "  };\n";
    _ctx << "  return &scheme;\n";
    _ctx << "}\n\n";
  }

  CTX _ctx;
};

}  // namespace cg
}  // namespace fhe

#endif  // FHE_CG_IR2C_DRIVER_H
