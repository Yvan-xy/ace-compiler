//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_POLY_POLY2C_CONFIG_H
#define FHE_POLY_POLY2C_CONFIG_H

#include "air/driver/common_config.h"
#include "air/driver/driver_ctx.h"
#include "fhe/core/lib_provider.h"

namespace fhe {
namespace poly {

struct POLY2C_CONFIG : public air::util::COMMON_CONFIG {
public:
  POLY2C_CONFIG(void)
      : _prov_str("ant"),
        _ct_encode(false),
        _free_poly(false),
        _pt_from_msg_name("Pt_from_msg"),
        _provider(fhe::core::PROVIDER::ANT),
        _ifile(nullptr) {}

  void Register_options(air::driver::DRIVER_CTX* ctx);
  void Update_options();
  void Set_ifile(const char* ifile) { _ifile = ifile; }

  void Print(std::ostream& os) const;

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

  // leave this member public so that OPTION_DESC can access it
  std::string _prov_str;
  std::string _data_file;  // place data in a seperated file
  bool        _ct_encode;  // encode constants to plaintext at compile time
  bool        _free_poly;  // insert free_poly
  std::string _function_name_prefix;
  std::string _constant_name_prefix;
  std::string _pt_from_msg_name;
  std::string _raise_mod_level_func;

  fhe::core::PROVIDER _provider;  // parsed from _prov_str
  const char*         _ifile;     // set ifile if data_file is set
};

//! @brief Macro to define API to access POLY2C config
#define DECLARE_POLY2C_CONFIG_ACCESS_API(cfg)                            \
  core::PROVIDER Provider() const { return cfg.Provider(); }             \
  const char*    Data_file() const { return cfg.Data_file(); }           \
  bool           Emit_data_file() const { return cfg.Emit_data_file(); } \
  bool           Ct_encode() const { return cfg.Ct_encode(); }           \
  bool           Free_poly() const { return cfg.Free_poly(); }           \
  const char*    Function_name_prefix() const {                          \
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

}  // namespace poly
}  // namespace fhe

#endif  // FHE_POLY_POLY2C_CONFIG_H
