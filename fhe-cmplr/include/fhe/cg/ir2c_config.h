//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CG_IR2C_CONFIG_H
#define FHE_CG_IR2C_CONFIG_H

#include <string>

#include "air/driver/common_config.h"
#include "fhe/core/lib_provider.h"

namespace fhe {
namespace cg {

//! @brief Provider-neutral configuration shared by FHE source emitters.
struct IR2C_CONFIG : public air::util::COMMON_CONFIG {
public:
  explicit IR2C_CONFIG(const char* provider = "ant")
      : _prov_str(provider),
        _ct_encode(false),
        _free_poly(false),
        _pt_from_msg_name("Pt_from_msg"),
        _provider(core::Provider_id(provider)),
        _ifile(nullptr) {}

  void Set_ifile(const char* ifile) { _ifile = ifile; }
  void Set_provider(const char* provider) {
    _prov_str = provider;
    _provider = core::Provider_id(_prov_str.c_str());
  }

  const char*    Prov_str() const { return _prov_str.c_str(); }
  core::PROVIDER Provider() const { return _provider; }
  const char*    Data_file() const { return _data_file.c_str(); }
  const char*    Ifile() const { return _ifile; }
  bool           Emit_data_file() const { return !_data_file.empty(); }
  bool           Ct_encode() const { return _ct_encode; }
  bool           Free_poly() const { return _free_poly; }
  const char*    Function_name_prefix() const {
    return _function_name_prefix.c_str();
  }
  const char* Constant_name_prefix() const {
    return _constant_name_prefix.c_str();
  }
  const char* Pt_from_msg_name() const { return _pt_from_msg_name.c_str(); }
  const char* Raise_mod_level_func() const {
    return _raise_mod_level_func.c_str();
  }

  // Public for AIR OPTION_DESC and the Python binding configuration adapters.
  std::string _prov_str;
  std::string _data_file;
  bool        _ct_encode;
  bool        _free_poly;
  std::string _function_name_prefix;
  std::string _constant_name_prefix;
  std::string _pt_from_msg_name;
  std::string _raise_mod_level_func;

  core::PROVIDER _provider;
  const char*    _ifile;
};

//! @brief Define APIs for contexts that consume source-codegen configuration.
#define DECLARE_IR2C_CONFIG_ACCESS_API(cfg)                               \
  core::PROVIDER Provider() const { return cfg.Provider(); }              \
  const char*    Data_file() const { return cfg.Data_file(); }            \
  bool           Emit_data_file() const { return cfg.Emit_data_file(); }  \
  bool           Ct_encode() const { return cfg.Ct_encode(); }            \
  bool           Free_poly() const { return cfg.Free_poly(); }            \
  const char*    Function_name_prefix() const {                           \
    return cfg.Function_name_prefix();                                    \
  }                                                                       \
  const char* Constant_name_prefix() const {                              \
    return cfg.Constant_name_prefix();                                    \
  }                                                                       \
  const char* Pt_from_msg_name() const { return cfg.Pt_from_msg_name(); } \
  const char* Raise_mod_level_func() const {                              \
    return cfg.Raise_mod_level_func();                                    \
  }                                                                       \
  DECLARE_COMMON_CONFIG_ACCESS_API(cfg)

}  // namespace cg
}  // namespace fhe

#endif  // FHE_CG_IR2C_CONFIG_H
