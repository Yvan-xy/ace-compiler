//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_CKKS_IR2C_CTX_H
#define FHE_CKKS_IR2C_CTX_H

#include "air/base/container_decl.h"
#include "air/base/st_decl.h"
#include "air/util/debug.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "fhe/core/ir2c_ctx.h"
#include "fhe/ckks/phantom_constant_manifest.h"
#include "fhe/ckks/phantom_context_manifest.h"
#include "fhe/core/rt_context.h"
#include "fhe/core/rt_data_writer.h"
#include "fhe/core/rt_encode_api.h"
#include "nn/core/attr.h"
#include "nn/vector/vector_opcode.h"

namespace fhe {

namespace ckks {

//! @brief Context for CKKS IR to C in fhe-cmplr
class IR2C_CTX : public fhe::core::IR2C_CTX {
  struct PHANTOM_EMITTED_CONSTANT {
    PHANTOM_CONSTANT_DESCRIPTOR _descriptor;
    air::base::CONSTANT_PTR      _constant = air::base::Null_ptr;
  };

public:
  //! @brief Construct a new ir2c ctx object
  IR2C_CTX(std::ostream& os, const fhe::core::LOWER_CTX& lower_ctx,
           const fhe::cg::IR2C_CONFIG& cfg)
      : fhe::core::IR2C_CTX(os, lower_ctx, cfg),
        _rt_data_writer(nullptr),
        _ct_encode(cfg.Ct_encode()) {
    if (cfg.Provider() == fhe::core::PROVIDER::PHANTOM) {
      _phantom_context =
          Build_phantom_context_descriptor(lower_ctx.Get_ctx_param());
      _phantom_resources = Build_phantom_resource_descriptor(
          lower_ctx.Get_ctx_param(), _phantom_context);
      _phantom_context_sha256 =
          Phantom_sha256(Serialize_phantom_context_descriptor(
              _phantom_context));
      _has_phantom_manifests = true;
    }
    if (cfg.Emit_data_file()) {
      // create rt_data_writer
      _data_file_uuid  = "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX";
      _data_entry_type =
          _ct_encode ? fhe::core::DE_PLAINTEXT : fhe::core::DE_MSG_F32;
      _rt_data_writer =
          new fhe::core::RT_DATA_WRITER(cfg.Data_file(), _data_entry_type,
                                        cfg.Ifile(), _data_file_uuid.c_str());
      if (_ct_encode && cfg.Provider() == fhe::core::PROVIDER::ANT) {
        const auto& ctx_param = lower_ctx.Get_ctx_param();
        uint32_t encode_depth =
            ctx_param.Get_mul_level() > 0 ? ctx_param.Get_mul_level() - 1 : 0;
        if (const char* raw = std::getenv("ACE_CT_ENCODE_DEPTH")) {
          char* end = nullptr;
          unsigned long parsed = std::strtoul(raw, &end, 10);
          if (end != raw && *end == '\0') {
            encode_depth = static_cast<uint32_t>(parsed);
          }
        }
        Prepare_encode_context(
            ctx_param.Get_poly_degree(), ctx_param.Get_security_level(),
            encode_depth,
            ctx_param.Get_input_level(), ctx_param.Get_first_prime_bit_num(),
            ctx_param.Get_scaling_factor_bit_num(), ctx_param.Get_q_part_num(),
            ctx_param.Get_hamming_weight());
      }
    }
  }

  bool Emit_provider_context_manifest() {
    if (!_has_phantom_manifests) return false;

    _ir2c_util << "static const uint32_t ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "phantom_data_q_bit_sizes[] = {";
    for (size_t index = 0; index < _phantom_context._data_q_bit_sizes.size();
         ++index) {
      if (index != 0) _ir2c_util << ", ";
      _ir2c_util << _phantom_context._data_q_bit_sizes[index];
    }
    _ir2c_util << "};\n";

    _ir2c_util << "static const uint32_t ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "phantom_special_p_bit_sizes[] = {";
    for (size_t index = 0;
         index < _phantom_context._special_p_bit_sizes.size(); ++index) {
      if (index != 0) _ir2c_util << ", ";
      _ir2c_util << _phantom_context._special_p_bit_sizes[index];
    }
    _ir2c_util << "};\n\n";

    _ir2c_util << "extern \"C\" const PHANTOM_CONTEXT_MANIFEST* ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "Get_phantom_context_manifest() {\n";
    _ir2c_util << "  static const PHANTOM_CONTEXT_MANIFEST context = {\n";
    _ir2c_util << "    " << _phantom_context._schema_version << ",\n";
    _ir2c_util << "    PHANTOM_PACKING_FULL,\n";
    _ir2c_util << "    " << _phantom_context._poly_degree << ",\n";
    _ir2c_util << "    " << _phantom_context._logical_slots << ",\n";
    _ir2c_util << "    " << _phantom_context._data_q_bit_sizes.size()
                 << ",\n";
    _ir2c_util << "    ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "phantom_data_q_bit_sizes,\n";
    _ir2c_util << "    " << _phantom_context._special_p_bit_sizes.size()
                 << ",\n";
    _ir2c_util << "    ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "phantom_special_p_bit_sizes,\n";
    _ir2c_util << "    " << _phantom_context._input_level << ",\n";
    _ir2c_util << "    " << _phantom_context._q_part_count << ",\n";
    _ir2c_util << "    " << _phantom_context._hamming_weight << ",\n";
    _ir2c_util << "    " << _phantom_context._security_level << ",\n";
    _ir2c_util << "    " << _phantom_context._first_modulus_bits << ",\n";
    _ir2c_util << "    " << _phantom_context._scaling_modulus_bits << ",\n";
    _ir2c_util << "    " << _phantom_context._resource_schema_version
                 << "\n";
    _ir2c_util << "  };\n";
    _ir2c_util << "  return &context;\n";
    _ir2c_util << "}\n\n";
    return true;
  }

  //! @brief Destruct ir2c ctx object
  ~IR2C_CTX() {
    if (_ct_encode) {
      Finalize_encode_context();
    }
    if (_rt_data_writer != nullptr) {
      delete _rt_data_writer;
    }
  }

  //! @brief Emit the selected runtime adapter and source-level aliases.
  void Emit_global_include() {
    _ir2c_util << "// external header files" << std::endl;
    _ir2c_util << "#include \"";
    _ir2c_util << fhe::core::Provider_header(Provider());
    _ir2c_util << "\"" << std::endl << std::endl;
    _ir2c_util << "typedef double float64_t;" << std::endl;
    _ir2c_util << "typedef float float32_t;" << std::endl;
    // ANT does not own these two aliases. C++ providers, including Phantom,
    // define their ABI-specific forms in the selected runtime header.
    if (Provider() == fhe::core::PROVIDER::ANT) {
      _ir2c_util << "typedef size_t LEVEL_T;" << std::endl;
      _ir2c_util << "typedef double SCALE_T;" << std::endl;
    }
    _ir2c_util << std::endl;
    if (std::string(Pt_from_msg_name()) != "Pt_from_msg") {
      _ir2c_util << "void* ";
      _ir2c_util.Emit_identifier(Pt_from_msg_name());
      _ir2c_util << "(void* pt, uint32_t index, size_t len, "
                    "uint32_t scale, uint32_t level);"
                 << std::endl;
    }
    if (Raise_mod_level_func()[0] != '\0') {
      _ir2c_util << "uint32_t ";
      _ir2c_util.Emit_identifier(Raise_mod_level_func());
      _ir2c_util << "(void);" << std::endl;
    }
    if (std::string(Pt_from_msg_name()) != "Pt_from_msg" ||
        Raise_mod_level_func()[0] != '\0') {
      _ir2c_util << std::endl;
    }
  }

  //! @brief Emit an FHE server function definition.
  void Emit_func_def(air::base::FUNC_SCOPE* func) {
    air::base::FUNC_PTR decl = func->Owning_func();
    if (decl->Entry_point()->Is_program_entry()) {
      _ir2c_util << "bool " << decl->Name()->Char_str() << "()";
    } else {
      air::base::IR2C_CTX::Emit_func_def(func);
    }
  }

  //! @brief Emit local variables and provider-specific input initialization.
  void Emit_local_var(air::base::FUNC_SCOPE* func) {
    air::base::IR2C_CTX::Emit_local_var(func);
    air::base::ENTRY_PTR entry        = func->Owning_func()->Entry_point();
    bool                 is_prg_entry = entry->Is_program_entry();
    if (Provider() != fhe::core::PROVIDER::ANT) {
      if (is_prg_entry) {
        uint32_t num_args = entry->Type()->Cast_to_sig()->Num_param();
        for (uint32_t i = 0; i < num_args; ++i) {
          air::base::ADDR_DATUM_PTR parm = func->Formal(i);
          AIR_ASSERT(parm->Is_formal());
          AIR_ASSERT(
              Is_cipher_type(parm->Type_id()) ||
              (parm->Type()->Is_array() &&
               Is_cipher_type(parm->Type()->Cast_to_arr()->Elem_type_id())));
          Emit_get_input_data(parm);
        }
      }
      if (Provider() == fhe::core::PROVIDER::PHANTOM) {
        for (auto it = func->Begin_addr_datum(); it != func->End_addr_datum();
             ++it) {
          air::base::TYPE_PTR type = (*it)->Type();
          const bool         is_array = type->Is_array();
          if (is_array && (*it)->Is_formal() && !is_prg_entry) {
            // Array formals alias caller-owned elements; their declaring
            // generated frame registers the element lifetimes.
            continue;
          }
          air::base::TYPE_ID type_id =
              is_array ? type->Cast_to_arr()->Elem_type_id() : type->Id();
          const bool is_cipher =
              Is_cipher_type(type_id) || Is_cipher3_type(type_id);
          if (!is_cipher && !Is_plain_type(type_id)) continue;
          _ir2c_util << "  Register_";
          _ir2c_util << (is_cipher ? "ciph" : "plain");
          if (is_array) {
            _ir2c_util << "_array_lifetime(";
            Emit_var(*it);
            _ir2c_util << ", sizeof(";
            Emit_var(*it);
            _ir2c_util << ") / sizeof(";
            Emit_var(*it);
            _ir2c_util << "[0]));" << std::endl;
          } else {
            _ir2c_util << "_lifetime(&";
            Emit_var(*it);
            _ir2c_util << ");" << std::endl;
          }
        }
        for (auto it = func->Begin_preg(); it != func->End_preg(); ++it) {
          air::base::TYPE_ID type_id = (*it)->Type_id();
          const bool is_cipher =
              Is_cipher_type(type_id) || Is_cipher3_type(type_id);
          if (!is_cipher && !Is_plain_type(type_id)) continue;
          _ir2c_util << "  Register_";
          _ir2c_util << (is_cipher ? "ciph" : "plain");
          _ir2c_util << "_lifetime(&";
          Emit_preg_id((*it)->Id());
          _ir2c_util << ");" << std::endl;
        }
      }
      return;
    }

    _ir2c_util << "  uint32_t  degree = Degree();" << std::endl;
    for (auto it = func->Begin_addr_datum(); it != func->End_addr_datum();
         ++it) {
      air::base::TYPE_PTR type = (*it)->Type();
      air::base::TYPE_ID type_id =
          type->Is_array() ? type->Cast_to_arr()->Elem_type_id() : type->Id();
      if (Is_cipher_type(type_id) || Is_cipher3_type(type_id) ||
          Is_plain_type(type_id) || Is_rns_poly_type(type_id)) {
        if (!(*it)->Is_formal()) {
          _ir2c_util << "  memset(&";
          Emit_var(*it);
          _ir2c_util << ", 0, sizeof(";
          Emit_var(*it);
          _ir2c_util << "));" << std::endl;
        } else if (is_prg_entry) {
          Emit_get_input_data(*it);
        }
      }
    }
    for (auto it = func->Begin_preg(); it != func->End_preg(); ++it) {
      air::base::TYPE_ID type = (*it)->Type_id();
      if (Is_cipher_type(type) || Is_cipher3_type(type) ||
          Is_plain_type(type) || Is_rns_poly_type(type)) {
        _ir2c_util << "  memset(&";
        Emit_preg_id((*it)->Id());
        _ir2c_util << ", 0, sizeof(";
        Emit_preg_id((*it)->Id());
        _ir2c_util << "));" << std::endl;
      } else if (Is_poly_type(type)) {
        _ir2c_util << "  Alloc_lpoly_data(&";
        Emit_preg_id((*it)->Id());
        _ir2c_util << ", degree);" << std::endl;
      }
    }
  }

  //! @brief Emit the runtime feature query without introducing BTS calls.
  void Emit_phantom_constant_manifest() {
    if (!_has_phantom_manifests) return;
    if (!_phantom_constants.empty()) {
      _ir2c_util << "static const PHANTOM_CONSTANT_ENTRY ";
      _ir2c_util.Emit_identifier(Function_name_prefix());
      _ir2c_util << "phantom_constant_entries[] = {\n";
      for (const auto& emitted : _phantom_constants) {
        const PHANTOM_CONSTANT_DESCRIPTOR& constant = emitted._descriptor;
        _ir2c_util << "  // ACE_PHANTOM_CONSTANT_ENTRY entry_id="
                   << constant._entry_id << " constant_id="
                   << constant._constant_id << "\n";
        _ir2c_util << "  {" << constant._entry_id << ", "
                   << constant._constant_id
                   << ", PHANTOM_CONSTANT_COMPLEX_F64, "
                   << constant._slot_count << ", " << constant._ace_level
                   << ", " << constant._chain_index << ", "
                   << constant._scale_degree << ", "
                   << constant._raw_scale_text << ", \"" << constant._symbol
                   << "\", \"" << constant._payload_sha256 << "\", \""
                   << constant._cache_key_sha256 << "\", "
                   << constant._slot_count * 2 << ", (const double*)";
        Emit_constant_name(emitted._constant->Id());
        _ir2c_util << "},\n";
      }
      _ir2c_util << "};\n\n";
    }
    _ir2c_util << "extern \"C\" const PHANTOM_CONSTANT_MANIFEST* ";
    _ir2c_util.Emit_identifier(Function_name_prefix());
    _ir2c_util << "Get_phantom_constant_manifest() {\n";
    _ir2c_util << "  static const PHANTOM_CONSTANT_MANIFEST constants = {\n";
    _ir2c_util << "    " << PHANTOM_CONSTANT_SCHEMA_VERSION << ",\n";
    _ir2c_util << "    " << _phantom_context._schema_version << ",\n";
    _ir2c_util << "    " << _phantom_resources._schema_version << ",\n";
    _ir2c_util << "    \"" << _phantom_context_sha256 << "\",\n";
    _ir2c_util << "    " << _phantom_constants.size() << ",\n";
    if (_phantom_constants.empty()) {
      _ir2c_util << "    nullptr\n";
    } else {
      _ir2c_util << "    ";
      _ir2c_util.Emit_identifier(Function_name_prefix());
      _ir2c_util << "phantom_constant_entries\n";
    }
    _ir2c_util << "  };\n";
    _ir2c_util << "  return &constants;\n";
    _ir2c_util << "}\n\n";
  }

  void Emit_need_bts() {
    if (Provider() == core::PROVIDER::PHANTOM) {
      if (_observed_rotation_batches !=
          _phantom_resources._rotation_batches.size()) {
        throw std::runtime_error(
            "Phantom rotate_batch analysis/codegen requirements disagree");
      }
      if (!_phantom_resources._rotation_steps.empty()) {
        _ir2c_util << "static const int32_t ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_steps[] = {";
        for (size_t index = 0;
             index < _phantom_resources._rotation_steps.size(); ++index) {
          if (index != 0) _ir2c_util << ", ";
          _ir2c_util << _phantom_resources._rotation_steps[index];
        }
        _ir2c_util << "};\n\n";
      }
      if (!_phantom_resources._rotation_batches.empty()) {
        _ir2c_util << "static const size_t ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_batch_offsets[] = {0";
        size_t offset = 0;
        for (const auto& batch : _phantom_resources._rotation_batches) {
          offset += batch.size();
          _ir2c_util << ", " << offset;
        }
        _ir2c_util << "};\n";
        _ir2c_util << "static const int32_t ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_batch_steps[] = {";
        bool has_prior_step = false;
        for (const auto& batch : _phantom_resources._rotation_batches) {
          for (int32_t step : batch) {
            if (has_prior_step) _ir2c_util << ", ";
            _ir2c_util << step;
            has_prior_step = true;
          }
        }
        _ir2c_util << "};\n\n";
      }
      if (!_phantom_resources._monomial_powers.empty()) {
        _ir2c_util << "static const uint32_t ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_monomial_powers[] = {";
        for (size_t index = 0;
             index < _phantom_resources._monomial_powers.size(); ++index) {
          if (index != 0) _ir2c_util << ", ";
          _ir2c_util << _phantom_resources._monomial_powers[index];
        }
        _ir2c_util << "};\n\n";
      }
      _ir2c_util << "extern \"C\" const PHANTOM_RESOURCE_MANIFEST* ";
      _ir2c_util.Emit_identifier(Function_name_prefix());
      _ir2c_util << "Get_phantom_resource_manifest() {\n";
      _ir2c_util << "  static const PHANTOM_RESOURCE_MANIFEST resources = {\n";
      _ir2c_util << "    " << _phantom_resources._schema_version << ",\n";
      _ir2c_util << "    " << _phantom_resources._context_schema_version
                   << ",\n";
      _ir2c_util << "    ";
      if (_phantom_resources._flags == 0) {
        _ir2c_util << "0";
      } else {
        bool separator = false;
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_RELIN_KEY) != 0) {
          _ir2c_util << "PHANTOM_RESOURCE_RELIN_KEY";
          separator = true;
        }
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_ROTATION_KEYS) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_ROTATION_KEYS";
          separator = true;
        }
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_CONJUGATION_KEY) !=
            0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_CONJUGATION_KEY";
          separator = true;
        }
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_ROTATE_BATCH) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_ROTATE_BATCH";
          separator = true;
        }
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_RAISE_MOD) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_RAISE_MOD";
          separator = true;
        }
        if ((_phantom_resources._flags & PHANTOM_RESOURCE_MONOMIALS) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_MONOMIALS";
          separator = true;
        }
        if ((_phantom_resources._flags &
             PHANTOM_RESOURCE_COMPLEX_PLAINTEXT) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_COMPLEX_PLAINTEXT";
          separator = true;
        }
        if ((_phantom_resources._flags &
             PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE) != 0) {
          if (separator) _ir2c_util << " | ";
          _ir2c_util << "PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE";
        }
      }
      _ir2c_util << ",\n";
      _ir2c_util << "    " << _phantom_resources._rotation_steps.size()
                   << ",\n";
      if (_phantom_resources._rotation_steps.empty()) {
        _ir2c_util << "    nullptr,\n";
      } else {
        _ir2c_util << "    ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_steps,\n";
      }
      _ir2c_util << "    " << _phantom_resources._rotation_batches.size()
                   << ",\n";
      if (_phantom_resources._rotation_batches.empty()) {
        _ir2c_util << "    nullptr,\n    nullptr,\n";
      } else {
        _ir2c_util << "    ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_batch_offsets,\n    ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_rotation_batch_steps,\n";
      }
      _ir2c_util << "    " << _phantom_resources._monomial_powers.size()
                   << ",\n";
      if (_phantom_resources._monomial_powers.empty()) {
        _ir2c_util << "    nullptr\n";
      } else {
        _ir2c_util << "    ";
        _ir2c_util.Emit_identifier(Function_name_prefix());
        _ir2c_util << "phantom_monomial_powers\n";
      }
      _ir2c_util << "  };\n";
      _ir2c_util << "  return &resources;\n";
      _ir2c_util << "}\n\n";
      Emit_phantom_constant_manifest();
    }
    if (Provider() == core::PROVIDER::SEAL) {
      _ir2c_util << "bool Need_bts() {\n";
      _ir2c_util << (_need_bts ? "  return true;\n" : "  return false;\n");
      _ir2c_util << "}\n\n";
    }
  }

  bool Has_phantom_manifests() const { return _has_phantom_manifests; }

  const PHANTOM_CONTEXT_DESCRIPTOR& Phantom_context_descriptor() const {
    AIR_ASSERT(_has_phantom_manifests);
    return _phantom_context;
  }

  const PHANTOM_RESOURCE_DESCRIPTOR& Phantom_resource_descriptor() const {
    AIR_ASSERT(_has_phantom_manifests);
    return _phantom_resources;
  }

  std::string Phantom_context_json() const {
    AIR_ASSERT(_has_phantom_manifests);
    return Serialize_phantom_context_descriptor(_phantom_context);
  }

  std::string Phantom_resource_json() const {
    AIR_ASSERT(_has_phantom_manifests);
    return Serialize_phantom_resource_descriptor(_phantom_resources);
  }

  std::string Phantom_constant_json() const {
    AIR_ASSERT(_has_phantom_manifests);
    std::vector<PHANTOM_CONSTANT_DESCRIPTOR> descriptors;
    descriptors.reserve(_phantom_constants.size());
    for (const auto& constant : _phantom_constants) {
      descriptors.push_back(constant._descriptor);
    }
    return Serialize_phantom_constant_manifest(
        _phantom_context_sha256, _phantom_context._schema_version,
        _phantom_resources._schema_version, descriptors);
  }

  void Require_phantom_relinearization_key() {
    if (!_has_phantom_manifests) return;
    if ((_phantom_resources._flags & PHANTOM_RESOURCE_RELIN_KEY) == 0) {
      throw std::runtime_error(
          "Phantom relinearization analysis/codegen requirements disagree: "
          "observed Relin call, expected a predeclared relinearization key");
    }
  }

  void Require_phantom_rotation_key(int64_t step) {
    if (!_has_phantom_manifests) return;
    const int32_t normalized =
        Canonical_signed_rotation(step, _phantom_context._logical_slots);
    if (normalized == 0) return;
    if (!std::binary_search(_phantom_resources._rotation_steps.begin(),
                            _phantom_resources._rotation_steps.end(),
                            normalized)) {
      throw std::runtime_error(
          "Phantom rotation analysis/codegen requirements disagree: observed "
          "normalized step " +
          std::to_string(normalized) +
          ", expected a predeclared ordinary rotation key");
    }
  }

  void Require_phantom_conjugation_key() {
    if (!_has_phantom_manifests) return;
    if ((_phantom_resources._flags & PHANTOM_RESOURCE_CONJUGATION_KEY) == 0) {
      throw std::runtime_error(
          "Phantom conjugation analysis/codegen requirements disagree");
    }
  }

  void Observe_phantom_rotation_batch(const int* steps, uint32_t count) {
    if (!_has_phantom_manifests) return;
    std::vector<int32_t> observed(steps, steps + count);
    if (_observed_rotation_batches <
        _phantom_resources._rotation_batches.size()) {
      if (_phantom_resources
              ._rotation_batches[_observed_rotation_batches] != observed) {
        throw std::runtime_error(
            "Phantom rotate_batch analysis/codegen order disagrees");
      }
    } else {
      throw std::runtime_error(
          "Phantom rotate_batch was not recorded by context analysis");
    }
    ++_observed_rotation_batches;
    if ((_phantom_resources._flags & PHANTOM_RESOURCE_ROTATE_BATCH) == 0) {
      throw std::runtime_error(
          "Phantom rotate_batch analysis/codegen flag disagrees");
    }
    for (int32_t step : observed) {
      const int32_t normalized =
          Canonical_signed_rotation(step, _phantom_context._logical_slots);
      if (normalized != 0 &&
          !std::binary_search(_phantom_resources._rotation_steps.begin(),
                              _phantom_resources._rotation_steps.end(),
                              normalized)) {
        throw std::runtime_error(
            "Phantom rotate_batch key analysis/codegen requirements disagree");
      }
    }
  }

  void Require_phantom_raise_mod() {
    if (!_has_phantom_manifests) return;
    if ((_phantom_resources._flags & PHANTOM_RESOURCE_RAISE_MOD) == 0) {
      throw std::runtime_error(
          "Phantom raise_mod analysis/codegen requirements disagree");
    }
  }

  void Require_phantom_complex_plaintext() {
    if (!_has_phantom_manifests) return;
    if ((_phantom_resources._flags &
         PHANTOM_RESOURCE_COMPLEX_PLAINTEXT) == 0) {
      throw std::runtime_error(
          "Phantom complex-plaintext analysis/codegen requirements "
          "disagree");
    }
  }

  uint32_t Require_phantom_monomial_power(int64_t power) {
    const int64_t period =
        static_cast<int64_t>(_phantom_context._poly_degree) * 2;
    int64_t normalized = power % period;
    if (normalized < 0) normalized += period;
    const uint32_t result = static_cast<uint32_t>(normalized);
    if (!_has_phantom_manifests) return result;
    if ((_phantom_resources._flags & PHANTOM_RESOURCE_MONOMIALS) == 0 ||
        !std::binary_search(_phantom_resources._monomial_powers.begin(),
                            _phantom_resources._monomial_powers.end(),
                            result)) {
      throw std::runtime_error(
          "Phantom mul_mono analysis/codegen requirements disagree");
    }
    return result;
  }

  void Emit_get_input_data(air::base::ADDR_DATUM_PTR var) {
    if (var->Type()->Is_array()) {
      uint64_t elem_count = var->Type()->Cast_to_arr()->Elem_count();
      _ir2c_util << "  for (int input_idx = 0; input_idx < " << elem_count
                 << "; ++input_idx) {" << std::endl;
      _ir2c_util << "    ";
      Emit_var(var);
      _ir2c_util << "[input_idx] = Get_input_data(\""
                 << var->Name()->Char_str() << "\", input_idx);" << std::endl;
      _ir2c_util << "  }" << std::endl;
    } else {
      _ir2c_util << "  ";
      Emit_var(var);
      _ir2c_util << " = Get_input_data(\"" << var->Name()->Char_str()
                 << "\", 0);" << std::endl;
    }
  }

  template <typename RETV, typename VISITOR>
  void Emit_encode(VISITOR* visitor, air::base::NODE_PTR dest,
                   air::base::NODE_PTR node) {
    const uint32_t* complex_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::ENCODE_DCMPLX);
    const uint32_t* scale_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::SCALE);
    const uint32_t* level_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    const uint32_t* num_p_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::NUM_P);
    const uint32_t* cache_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::ENCODE_CACHE);
    bool encoding_dcmplx = (complex_attr != nullptr) && (*complex_attr != 0);
    bool encode_cache     = (cache_attr != nullptr) && (*cache_attr != 0);
    if (encoding_dcmplx && Provider() == core::PROVIDER::PHANTOM) {
      Require_phantom_complex_plaintext();
    }
    // NUM_P describes ANT's extended Q+P plaintext encoding.  Phantom
    // constants always use the ordinary Q-level complex encoder and therefore
    // retain the full (length, scale_degree, logical_level) argument list.
    bool use_extended_dcmplx =
        encoding_dcmplx && Provider() != core::PROVIDER::PHANTOM &&
        num_p_attr != nullptr && *num_p_attr != 0;
    if (Provider() == core::PROVIDER::ANT && !encoding_dcmplx &&
        _rt_data_writer != nullptr &&
        node->Child(0)->Opcode() == air::core::OPC_LDC &&
        node->Child(1)->Opcode() == air::core::OPC_INTCONST &&
        node->Child(2)->Opcode() == air::core::OPC_INTCONST &&
        node->Child(3)->Opcode() == air::core::OPC_INTCONST &&
        node->Child(0)->Const()->Kind() == air::base::CONSTANT_KIND::ARRAY) {
      // Offline encoding path: ldc of an ARRAY constant (not scalar FLOAT
      // from mask encoding — scalar constants fall through to runtime encode)
      air::base::CONSTANT_PTR cst = node->Child(0)->Const();
      AIR_ASSERT(cst->Type()->Is_array());
      AIR_ASSERT(cst->Type()->Cast_to_arr()->Elem_type()->Is_prim());
      AIR_ASSERT(
          cst->Type()->Cast_to_arr()->Elem_type()->Cast_to_prim()->Encoding() ==
          air::base::PRIMITIVE_TYPE::FLOAT_32);
      char name[32];
      snprintf(name, 32, "cst_%d", cst->Id().Value());
      const float* data  = (const float*)cst->Array_buffer();
      uint64_t     count = cst->Array_byte_len() / sizeof(float);
      AIR_ASSERT(count >= node->Child(1)->Intconst());
      // get level & scale from node
      uint32_t sc  = (scale_attr != nullptr) ? *scale_attr : node->Child(2)->Intconst();
      uint32_t lv  = (level_attr != nullptr) ? *level_attr : node->Child(3)->Intconst();
      uint64_t idx = (!_ct_encode)
                         ? _rt_data_writer->Append(name, data, count, sc, lv)
                         : Append_plain_buffer(name, data, count, sc, lv);
      // Pt_from_msg_validate(&dest, cst, index, len, scale, level)
      // Pt_from_msg(&dest, index, len, scale, level)
      // TODO: offline encoding support validate?
      if (Rt_validate()) {
        _ir2c_util << "Pt_from_msg_validate(&";
        Emit_st_var<RETV, VISITOR>(visitor, dest);
        _ir2c_util << ", ";
        Emit_buffer_address<RETV, VISITOR>(visitor, node->Child(0));
      } else {
        Emit_st_var<RETV, VISITOR>(visitor, dest);
        _ir2c_util << " = *(PLAIN)" << Pt_from_msg_name() << "(&";
        Emit_st_var<RETV, VISITOR>(visitor, dest);
      }
      _ir2c_util << ", " << idx << " /* " << name << " */";
    } else if (Provider() == core::PROVIDER::ANT &&
               _rt_data_writer != nullptr &&
               node->Child(0)->Opcode() == nn::vector::OPC_SLICE &&
               node->Child(1)->Opcode() == air::core::OPC_INTCONST) {
#if 0
      air::base::NODE_PTR slice = node->Child(0);
      AIR_ASSERT(slice->Child(0)->Opcode() == air::core::OPC_LDC);
      air::base::CONSTANT_PTR cst = slice->Child(0)->Const();
      AIR_ASSERT(cst->Kind() == air::base::CONSTANT_KIND::ARRAY);
      AIR_ASSERT(cst->Type()->Is_array());
      AIR_ASSERT(cst->Type()->Cast_to_arr()->Elem_type()->Is_prim());
      AIR_ASSERT(
          cst->Type()->Cast_to_arr()->Elem_type()->Cast_to_prim()->Encoding() ==
          air::base::PRIMITIVE_TYPE::FLOAT_32);
      AIR_ASSERT(slice->Child(2)->Opcode() == air::core::OPC_INTCONST);
      AIR_ASSERT(node->Child(1)->Opcode() == air::core::OPC_INTCONST);
      AIR_ASSERT(slice->Child(2)->Intconst() == node->Child(1)->Intconst());

      char name[32];
      snprintf(name, 32, "cst_%d", cst->Id().Value());
      const float*    data  = (const float*)cst->Array_buffer();
      uint64_t        count = cst->Array_byte_len() / sizeof(float);
      const uint32_t* sc_attr =
          node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::SCALE);
      AIR_ASSERT_MSG(sc_attr, "missing scale attribute");
      const uint32_t* lv_attr =
          node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
      AIR_ASSERT_MSG(lv_attr, "missing level attribute");
      uint64_t idx =
          _rt_data_writer->Append(name, data, count, *sc_attr, *lv_attr);
      // Pt_from_msg_ofst_validate(&dest, idx, start, len, scale, level)
      // Pt_from_msg_ofst(&dest, idx, start, len, scale, level)
      if (Rt_validate()) {
        _ir2c_util << "Pt_from_msg_ofst_validate(&";
        Emit_st_var<RETV, VISITOR>(visitor, dest);
        _ir2c_util << ", ";
        Emit_buffer_address<RETV, VISITOR>(visitor, node->Child(0));
      } else {
        Emit_st_var<RETV, VISITOR>(visitor, dest);
        _ir2c_util << " = *(PLAIN)Pt_from_msg_ofst(&";
        Emit_st_var<RETV, VISITOR>(visitor, dest);
      }
      _ir2c_util << ", " << idx << "/* " << name << " */, (";
      visitor->template Visit<RETV>(slice->Child(1));  // row_idx
      _ir2c_util << ") * (";
      visitor->template Visit<RETV>(slice->Child(2));  // col
      _ir2c_util << ")";
#endif
      // the code below doesn't work for complex array subscript because it's
      // very difficult to evaluate complex subscript expression with irregular
      // IV order.
      // disable it for future reference
#if 1
      air::base::NODE_PTR slice = node->Child(0);
      AIR_ASSERT(slice->Child(0)->Opcode() == air::core::OPC_LDC);
      air::base::CONSTANT_PTR cst = slice->Child(0)->Const();
      AIR_ASSERT(cst->Kind() == air::base::CONSTANT_KIND::ARRAY);
      AIR_ASSERT(cst->Type()->Is_array());
      AIR_ASSERT(cst->Type()->Cast_to_arr()->Elem_type()->Is_prim());
      AIR_ASSERT(
          cst->Type()->Cast_to_arr()->Elem_type()->Cast_to_prim()->Encoding() ==
          air::base::PRIMITIVE_TYPE::FLOAT_32);
      AIR_ASSERT(slice->Child(2)->Opcode() == air::core::OPC_INTCONST);
      air::base::NODE_PTR                       start = slice->Child(1);
      std::vector<std::pair<int64_t, int64_t> > subscript;
      bool ret = Parse_subscript_expr(start, subscript);
      AIR_ASSERT(ret == true && subscript.size() > 0);
      uint64_t loop_cnt = 1;
      for (uint64_t i = 0; i < subscript.size(); ++i) {
        // Find the matching DO_LOOP by IV id in the parent stack.
        air::base::NODE_PTR loop = air::base::Null_ptr;
        for (size_t depth = 1;; ++depth) {
          air::base::NODE_PTR cand = visitor->Parent(depth);
          if (cand == air::base::Null_ptr) break;
          if (cand->Opcode() == air::core::DO_LOOP &&
              cand->Iv_id().Value() == subscript[i].first) {
            loop = cand;
            break;
          }
        }
        AIR_ASSERT(loop != air::base::Null_ptr);
        int64_t lb, ub, stride;
        ret = Parse_do_loop(loop, lb, ub, stride);
        AIR_ASSERT(ret == true && lb == 0 && stride == 1);
        AIR_ASSERT(subscript[i].first == loop->Iv_id().Value());
        AIR_ASSERT(subscript[i].second == -1 || subscript[i].second == ub);
        loop_cnt *= ub;
      }
      const float* data        = (const float*)cst->Array_buffer();
      uint64_t     total_count = cst->Array_byte_len() / sizeof(float);
      uint64_t     span        = slice->Child(2)->Intconst();
      uint64_t     count       = node->Child(1)->Intconst();
      for (uint64_t i = 0; i < loop_cnt; ++i) {
        AIR_ASSERT(total_count >= i * span + count);
        char name[32];
        snprintf(name, 32, "cst_%d_%d", cst->Id().Value(), (int)i);
        // get level & scale from node
        uint32_t sc = node->Child(2)->Intconst();
        uint32_t lv = node->Child(3)->Intconst();
        uint64_t idx =
            _rt_data_writer->Append(name, data + i * span, count, sc, lv);
        if (i == 0) {
          // Pt_from_msg(&dest, index, len, scale, level)
          // Pt_from_msg_validate(&dest, cst, index, len, scale, level)
          // TODO: offline encoding support validate?
          if (Rt_validate()) {
            _ir2c_util << "Pt_from_msg_validate(&";
            Emit_st_var<RETV, VISITOR>(visitor, dest);
            _ir2c_util << ", ";
            Emit_buffer_address<RETV, VISITOR>(visitor, node->Child(0));
          } else {
            Emit_st_var<RETV, VISITOR>(visitor, dest);
            _ir2c_util << " = *(PLAIN)" << Pt_from_msg_name() << "(&";
            Emit_st_var<RETV, VISITOR>(visitor, dest);
          }
          _ir2c_util << ", ";
          visitor->template Visit<RETV>(start);
          _ir2c_util << " + " << idx << " /* " << name << " */";
        }
      }
#endif
    } else {
      // runtime encoding with internal data embedded in C code
      // Encode_float(&dest, cst, len, scale, level);
      if (_ct_encode && use_extended_dcmplx && encode_cache &&
          node->Child(0)->Opcode() == air::core::OPC_LDC) {
        Emit_offline_dcmplx_encode<RETV, VISITOR>(visitor, dest, node);
        return;
      } else if (encoding_dcmplx && encode_cache &&
          node->Child(0)->Opcode() == air::core::OPC_LDC) {
        Emit_cached_dcmplx_encode<RETV, VISITOR>(visitor, dest, node);
        return;
      } else {
        Emit_runtime_encode<RETV, VISITOR>(visitor, dest, node);
      }
    }
    _ir2c_util << ", ";
    visitor->template Visit<RETV>(node->Child(1));  // element count
    if (use_extended_dcmplx) {
      // Encode_dcmplx_ext(plain, input, len, level, p_cnt)
      _ir2c_util << ", ";
      if (level_attr != nullptr) {
        _ir2c_util << *level_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(3));  // level
      }
      _ir2c_util << ", " << *num_p_attr;
    } else {
      _ir2c_util << ", ";
      if (scale_attr != nullptr) {
        _ir2c_util << *scale_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(2));  // scale
      }
      _ir2c_util << ", ";
      if (level_attr != nullptr) {
        _ir2c_util << *level_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(3));  // level
      }
    }
    _ir2c_util << ")";
  }

  uint64_t Append_plain_buffer(const char* name, const float* data,
                               uint32_t count, uint32_t sc, uint32_t lv) {
    struct PLAINTEXT_BUFFER* buf =
        Encode_plain_buffer(data, count, sc, lv);
    uint64_t idx =
        _rt_data_writer->Append_pt(name, (const char*)buf, Plain_buffer_length(buf),
                                   sc, lv);
    Free_plain_buffer(buf);
    return idx;
  }

  template <typename RETV, typename VISITOR>
  void Emit_offline_dcmplx_encode(VISITOR* visitor, air::base::NODE_PTR dest,
                                  air::base::NODE_PTR node) {
    air::base::NODE_PTR cst = node->Child(0);
    AIR_ASSERT(cst->Opcode() == air::core::OPC_LDC);
    air::base::CONSTANT_PTR cst_val = cst->Const();
    AIR_ASSERT(cst_val != air::base::Null_ptr);
    AIR_ASSERT(cst_val->Kind() == air::base::CONSTANT_KIND::ARRAY);

    const uint32_t* num_p_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::NUM_P);
    const uint32_t* level_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    AIR_ASSERT(num_p_attr != nullptr && *num_p_attr != 0);
    uint32_t num_p = *num_p_attr;
    if (_ct_encode) {
      AIR_ASSERT_MSG(
          num_p <= Get_p_cnt(),
          "offline Encode_dcmplx_ext requires num_p within encode context");
    }

    uint32_t lv = (level_attr != nullptr) ? *level_attr : node->Child(3)->Intconst();
    uint64_t count = cst_val->Array_byte_len() / sizeof(double);
    AIR_ASSERT((count % 2) == 0);
    uint32_t complex_len = node->Child(1)->Intconst();
    AIR_ASSERT(count >= (uint64_t)complex_len * 2);

    char name[32];
    snprintf(name, 32, "cst_%d", cst_val->Id().Value());
    struct PLAINTEXT_BUFFER* buf = Encode_dcmplx_ext_buffer(
        cst_val->Array_buffer(), complex_len, lv, num_p);
    uint64_t idx = _rt_data_writer->Append_pt(
        name, (const char*)buf, Plain_buffer_length(buf), 1, lv);
    Free_plain_buffer(buf);

    uint32_t node_id = node->Id().Value();
    _ir2c_util << "{ static PLAINTEXT _pre_plain_" << node_id
               << "; static uint32_t _pre_plain_" << node_id
               << "_init = 0; if (!_pre_plain_" << node_id << "_init) {\n";
    _ir2c_util << "#pragma omp critical(_pre_plain_" << node_id << "_lock)\n";
    _ir2c_util << "{ if (!_pre_plain_" << node_id
               << "_init) { Copy_plain(&_pre_plain_" << node_id << ", (PLAIN)"
               << Pt_from_msg_name() << "(&_pre_plain_" << node_id;
    _ir2c_util << ", " << idx << " /* " << name << " */";
    _ir2c_util << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    _ir2c_util << ", 1, " << lv << ")); _pre_plain_" << node_id
               << "_init = 1; } } } ";
    Emit_st_var<RETV, VISITOR>(visitor, dest);
    _ir2c_util << " = _pre_plain_" << node_id << "; }";
  }

  static std::string Sanitize_phantom_identifier(const char* input) {
    std::string result;
    if (input == nullptr) return result;
    for (const unsigned char ch : std::string(input)) {
      result.push_back((std::isalnum(ch) != 0 || ch == '_')
                           ? static_cast<char>(ch)
                           : '_');
    }
    return result;
  }

  static uint32_t Static_positive_u32(air::base::NODE_PTR node,
                                      const char* diagnostic) {
    if (node->Opcode() != air::core::OPC_INTCONST || node->Intconst() <= 0 ||
        static_cast<uint64_t>(node->Intconst()) >
            std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(diagnostic);
    }
    return static_cast<uint32_t>(node->Intconst());
  }

  uint32_t Register_phantom_constant(air::base::NODE_PTR node,
                                     air::base::CONSTANT_PTR constant) {
    AIR_ASSERT(Provider() == core::PROVIDER::PHANTOM);
    if (constant->Kind() != air::base::CONSTANT_KIND::ARRAY ||
        !constant->Type()->Is_array()) {
      throw std::runtime_error(
          "Phantom cached complex plaintext requires an array constant");
    }
    air::base::TYPE_PTR element_type = constant->Type();
    while (element_type->Is_array()) {
      element_type = element_type->Cast_to_arr()->Elem_type();
    }
    if (!element_type->Is_prim() ||
        element_type->Cast_to_prim()->Encoding() !=
            air::base::PRIMITIVE_TYPE::FLOAT_64) {
      throw std::runtime_error(
          "Phantom cached complex plaintext requires interleaved float64 "
          "values");
    }

    const uint32_t slot_count = Static_positive_u32(
        node->Child(1),
        "Phantom cached complex plaintext requires a static slot count");
    if (slot_count > _phantom_context._logical_slots ||
        slot_count > std::numeric_limits<size_t>::max() / 2 ||
        static_cast<size_t>(slot_count) * 2 >
            std::numeric_limits<size_t>::max() / sizeof(double)) {
      throw std::runtime_error(
          "Phantom cached complex plaintext slot count is invalid");
    }
    const size_t interleaved_count = static_cast<size_t>(slot_count) * 2;
    const size_t payload_bytes = interleaved_count * sizeof(double);
    if (constant->Array_byte_len() < payload_bytes) {
      throw std::runtime_error(
          "Phantom cached complex plaintext payload is truncated");
    }

    const uint32_t* scale_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::SCALE);
    const uint32_t* level_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    const uint32_t scale_degree =
        scale_attr != nullptr
            ? *scale_attr
            : Static_positive_u32(
                  node->Child(2),
                  "Phantom cached complex plaintext requires a static scale");
    const uint32_t ace_level =
        level_attr != nullptr
            ? *level_attr
            : Static_positive_u32(
                  node->Child(3),
                  "Phantom cached complex plaintext requires a static level");
    if (scale_degree == 0 ||
        scale_degree > static_cast<uint32_t>(
                           std::numeric_limits<int32_t>::max()) ||
        ace_level == 0 ||
        ace_level > _phantom_context._data_q_bit_sizes.size()) {
      throw std::runtime_error(
          "Phantom cached complex plaintext scale or level is invalid");
    }
    const int64_t scale_exponent =
        static_cast<int64_t>(scale_degree) *
        static_cast<int64_t>(_phantom_context._scaling_modulus_bits);
    if (scale_exponent > std::numeric_limits<double>::max_exponent - 1) {
      throw std::runtime_error(
          "Phantom cached complex plaintext scale is not finite");
    }
    const double raw_scale =
        std::ldexp(1.0, static_cast<int>(scale_exponent));
    const uint32_t chain_index =
        1 + static_cast<uint32_t>(_phantom_context._data_q_bit_sizes.size()) -
        ace_level;
    const uint64_t constant_id = constant->Id().Value();
    const std::string symbol =
        Sanitize_phantom_identifier(Constant_name_prefix()) + "_cst_" +
        std::to_string(constant_id);
    const std::string payload_sha256 =
        Phantom_sha256(constant->Array_buffer(), payload_bytes);
    const std::string raw_scale_text = Phantom_raw_scale_text(raw_scale);
    const std::string cache_key_sha256 = Phantom_sha256(
        Build_phantom_cache_key_json(constant_id, _phantom_context_sha256,
                                     chain_index, raw_scale_text,
                                     "complex_f64", slot_count));

    for (const auto& emitted : _phantom_constants) {
      const PHANTOM_CONSTANT_DESCRIPTOR& prior = emitted._descriptor;
      if (prior._constant_id == constant_id &&
          prior._chain_index == chain_index &&
          prior._raw_scale_text == raw_scale_text &&
          prior._element_type == "complex_f64" &&
          prior._slot_count == slot_count) {
        if (prior._payload_sha256 != payload_sha256 ||
            prior._ace_level != ace_level ||
            prior._scale_degree != static_cast<int32_t>(scale_degree)) {
          throw std::runtime_error(
              "Phantom plaintext cache key resolves to conflicting "
              "constant metadata");
        }
        return prior._entry_id;
      }
    }

    PHANTOM_CONSTANT_DESCRIPTOR descriptor;
    descriptor._entry_id = static_cast<uint32_t>(_phantom_constants.size());
    descriptor._constant_id = constant_id;
    descriptor._symbol = symbol;
    descriptor._slot_count = slot_count;
    descriptor._ace_level = ace_level;
    descriptor._chain_index = chain_index;
    descriptor._scale_degree = static_cast<int32_t>(scale_degree);
    descriptor._raw_scale = raw_scale;
    descriptor._raw_scale_text = raw_scale_text;
    descriptor._payload_sha256 = payload_sha256;
    descriptor._cache_key_sha256 = cache_key_sha256;
    _phantom_constants.push_back({descriptor, constant});
    return descriptor._entry_id;
  }

  template <typename RETV, typename VISITOR>
  void Emit_cached_dcmplx_encode(VISITOR* visitor, air::base::NODE_PTR dest,
                                 air::base::NODE_PTR node) {
    air::base::NODE_PTR cst = node->Child(0);
    AIR_ASSERT(cst->Opcode() == air::core::OPC_LDC);
    air::base::CONSTANT_PTR cst_val = cst->Const();
    AIR_ASSERT(cst_val != air::base::Null_ptr);

    const uint32_t* num_p_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::NUM_P);
    const uint32_t* level_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::LEVEL);
    bool use_extended_dcmplx =
        Provider() != core::PROVIDER::PHANTOM && num_p_attr != nullptr &&
        *num_p_attr != 0;

    if (Provider() == core::PROVIDER::PHANTOM) {
      const uint32_t entry_id = Register_phantom_constant(node, cst_val);
      _ir2c_util << "{ Load_cached_plain(&";
      Emit_st_var<RETV, VISITOR>(visitor, dest);
      _ir2c_util << ", " << entry_id << "); }";
      return;
    }

    uint32_t node_id = node->Id().Value();
    _ir2c_util << "{ static PLAINTEXT _pre_plain_" << node_id
               << "; static uint32_t _pre_plain_" << node_id
               << "_init = 0; if (!_pre_plain_" << node_id << "_init) {\n";
    _ir2c_util << "#pragma omp critical(_pre_plain_" << node_id << "_lock)\n";
    _ir2c_util << "{ if (!_pre_plain_" << node_id << "_init) { ";
    _ir2c_util << (use_extended_dcmplx
                       ? "Encode_dcmplx_ext(&_pre_plain_"
                       : "Encode_dcmplx(&_pre_plain_");
    _ir2c_util << node_id << ", (DCMPLX*)";
    Emit_buffer_address<RETV, VISITOR>(visitor, cst);
    _ir2c_util << ", ";
    visitor->template Visit<RETV>(node->Child(1));
    if (use_extended_dcmplx) {
      _ir2c_util << ", ";
      if (level_attr != nullptr) {
        _ir2c_util << *level_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(3));
      }
      _ir2c_util << ", " << *num_p_attr;
    } else {
      const uint32_t* scale_attr =
          node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::SCALE);
      _ir2c_util << ", ";
      if (scale_attr != nullptr) {
        _ir2c_util << *scale_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(2));
      }
      _ir2c_util << ", ";
      if (level_attr != nullptr) {
        _ir2c_util << *level_attr;
      } else {
        visitor->template Visit<RETV>(node->Child(3));
      }
    }
    _ir2c_util << "); _pre_plain_" << node_id
               << "_init = 1; } } } ";
    Emit_st_var<RETV, VISITOR>(visitor, dest);
    _ir2c_util << " = _pre_plain_" << node_id << "; }";
  }

  template <typename RETV, typename VISITOR>
  void Emit_runtime_encode(VISITOR* visitor, air::base::NODE_PTR dest,
                           air::base::NODE_PTR node) {
    air::base::NODE_PTR cst      = node->Child(0);
    air::base::TYPE_PTR cst_type = cst->Rtype();
    air::base::TYPE_PTR domain_type;
    const double*       mask_attr = node->Attr<double>(nn::core::ATTR::MASK);
    bool                encoding_mask = (mask_attr != nullptr);
    const uint32_t*     complex_attr =
        node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::ENCODE_DCMPLX);
    bool encoding_dcmplx = (complex_attr != nullptr) && (*complex_attr != 0);
    if (cst_type->Is_ptr()) {
      domain_type = cst_type->Cast_to_ptr()->Domain_type();
    } else if (cst_type->Is_prim()) {
      // Handle primitive types (e.g., INTCONST from Python DSL)
      domain_type = cst_type;
    } else {
      // in case of encoding mask, the constant is a single float value.
      if (encoding_mask) {
        domain_type = cst_type;
      } else {
        AIR_ASSERT(cst_type->Is_array());
        domain_type = cst_type->Cast_to_arr()->Elem_type();
      }
    }
    AIR_ASSERT(domain_type->Is_prim());
    air::base::NODE_PTR level_child = node->Child(3);
    if (encoding_dcmplx) {
      AIR_ASSERT_MSG(!encoding_mask,
                     "Encode_dcmplx and mask-encoding are mutually exclusive");
      const uint32_t* num_p_attr =
          node->Attr<uint32_t>(fhe::core::FHE_ATTR_KIND::NUM_P);
      bool use_extended_dcmplx =
          Provider() != core::PROVIDER::PHANTOM && num_p_attr != nullptr &&
          *num_p_attr != 0;
      _ir2c_util << (use_extended_dcmplx
                         ? "Encode_dcmplx_ext(&"
                         : "Encode_dcmplx(&");
      Emit_st_var<RETV, VISITOR>(visitor, dest);
      _ir2c_util << ", (DCMPLX*)";
      Emit_buffer_address<RETV, VISITOR>(visitor, cst);
      return;
    }

    switch (domain_type->Cast_to_prim()->Encoding()) {
      case air::base::PRIMITIVE_TYPE::FLOAT_32:
        if (this->Provider() == core::PROVIDER::SEAL &&
            level_child->Opcode() == air::core::OPC_INTCONST) {
          _ir2c_util << (encoding_mask ? "Encode_float_mask_cst_lvl(&"
                                       : "Encode_float_cst_lvl(&");
        } else {
          _ir2c_util << (encoding_mask ? "Encode_float_mask(&"
                                       : "Encode_float(&");
        }
        break;
      case air::base::PRIMITIVE_TYPE::FLOAT_64:
      // Handle integer types from Python DSL - treat as double encoding
      case air::base::PRIMITIVE_TYPE::INT_S32:
      case air::base::PRIMITIVE_TYPE::INT_U32:
      case air::base::PRIMITIVE_TYPE::INT_S64:
      case air::base::PRIMITIVE_TYPE::INT_U64:
        if (this->Provider() == core::PROVIDER::SEAL &&
            level_child->Opcode() == air::core::OPC_INTCONST) {
          _ir2c_util << (encoding_mask ? "Encode_double_mask_cst_lvl(&"
                                       : "Encode_double_cst_lvl(&");
        } else {
          _ir2c_util << (encoding_mask ? "Encode_double_mask(&"
                                       : "Encode_double(&");
        }
        break;
      default:
        AIR_ASSERT_MSG(false, "not supported primitive type for encoding");
    }

    Emit_st_var<RETV, VISITOR>(visitor, dest);
    _ir2c_util << ", ";

    // For integer arrays, emit cast to (double*) for Encode_double
    // compatibility
    bool need_cast = (domain_type->Cast_to_prim()->Encoding() ==
                          air::base::PRIMITIVE_TYPE::INT_S32 ||
                      domain_type->Cast_to_prim()->Encoding() ==
                          air::base::PRIMITIVE_TYPE::INT_U32 ||
                      domain_type->Cast_to_prim()->Encoding() ==
                          air::base::PRIMITIVE_TYPE::INT_S64 ||
                      domain_type->Cast_to_prim()->Encoding() ==
                          air::base::PRIMITIVE_TYPE::INT_U64);
    if (need_cast) {
      _ir2c_util << "(double*)";
    }
    Emit_buffer_address<RETV, VISITOR>(visitor, cst);
  }

  template <typename RETV, typename VISITOR>
  void Emit_buffer_address(VISITOR* visitor, air::base::NODE_PTR node) {
    if (node->Opcode() == air::core::LDC && node->Rtype()->Is_array() &&
        visitor->Parent(0)->Opcode() == OPC_ENCODE) {
      air::base::CONSTANT_PTR cst = node->Const();
      AIR_ASSERT(cst->Kind() == air::base::CONSTANT_KIND::ARRAY);
      AIR_ASSERT(cst->Type()->Is_array());
      air::base::CONST_ARRAY_TYPE_PTR arr_ty = cst->Type()->Cast_to_arr();
      AIR_ASSERT(arr_ty->Elem_type()->Is_prim());

      // Check if element type is integer - need cast to (double*)
      air::base::PRIMITIVE_TYPE elem_enc =
          arr_ty->Elem_type()->Cast_to_prim()->Encoding();
      bool need_cast = (elem_enc == air::base::PRIMITIVE_TYPE::INT_S32 ||
                        elem_enc == air::base::PRIMITIVE_TYPE::INT_U32 ||
                        elem_enc == air::base::PRIMITIVE_TYPE::INT_S64 ||
                        elem_enc == air::base::PRIMITIVE_TYPE::INT_U64);

      if (need_cast) {
        _ir2c_util << "(double*)";
      }

      uint32_t dims = arr_ty->Dim();
      if (dims == 1) {
        visitor->template Visit<RETV>(node);
      } else {
        _ir2c_util << "&";
        visitor->template Visit<RETV>(node);
        for (size_t i = 0; i < dims; i++) {
          _ir2c_util << "[0]";
        }
      }
    } else {
      visitor->template Visit<RETV>(node);
    }
  }

  const char* Data_file_uuid() const { return _data_file_uuid.c_str(); }

  core::DATA_ENTRY_TYPE Data_entry_type() const { return _data_entry_type; }

public:
  // return total stride from subscript array
  int64_t Stride(const std::vector<std::pair<int64_t, int64_t> >& subscript) {
    AIR_ASSERT(subscript.size() > 0);
    AIR_ASSERT(subscript.back().second == -1);
    int64_t stride = 1;
    for (int i = 0; i < subscript.size() - 1; ++i) {
      stride *= subscript[i].second;
    }
    AIR_ASSERT(stride >= 1);
    return stride;
  }

  // Parse do_loop children to get constant lb/ub/stride
  bool Parse_do_loop(air::base::NODE_PTR node, int64_t& lb, int64_t& ub,
                     int64_t& stride) {
    if (node->Opcode() != air::core::OPC_DO_LOOP) {
      return false;
    }
    air::base::ADDR_DATUM_ID iv = node->Iv_id();
    if (node->Loop_init()->Opcode() != air::core::OPC_INTCONST) {
      return false;
    }
    lb = node->Loop_init()->Intconst();
    if (node->Compare()->Opcode() != air::core::OPC_LT ||
        node->Compare()->Child(0)->Opcode() != air::core::OPC_LD ||
        node->Compare()->Child(1)->Opcode() != air::core::OPC_INTCONST) {
      return false;
    }
    AIR_ASSERT(node->Compare()->Child(0)->Addr_datum_id() == iv);
    ub = node->Compare()->Child(1)->Intconst();
    if (node->Loop_incr()->Opcode() != air::core::OPC_ADD ||
        node->Loop_incr()->Child(0)->Opcode() != air::core::OPC_LD ||
        node->Loop_incr()->Child(1)->Opcode() != air::core::OPC_INTCONST) {
      return false;
    }
    AIR_ASSERT(node->Loop_incr()->Child(0)->Addr_datum_id() == iv);
    stride = node->Loop_incr()->Child(1)->Intconst();
    return true;
  }

  // Parse compound expression with add/mul to get subscript info
  bool Parse_subscript_expr(
      air::base::NODE_PTR                        node,
      std::vector<std::pair<int64_t, int64_t> >& subscript) {
    if (node->Opcode() == air::core::OPC_LD) {
      subscript.emplace_back(std::make_pair(node->Addr_datum_id().Value(), -1));
      return true;
    } else if (node->Opcode() == air::core::OPC_ADD) {
      if (node->Child(0)->Opcode() == air::core::OPC_MUL) {
        if (node->Child(1)->Opcode() == air::core::OPC_LD) {
          // iv_0 * stride_1 + iv_1
          Parse_subscript_expr(node->Child(1), subscript);
          return Parse_subscript_expr(node->Child(0), subscript);
        }
        // (iv_0 * stride_1 + iv_1) * stride_2 + iv2
        //   --> iv_0 * stride_1 * stride_2 + iv_1 * stride_2 + iv2
        AIR_ASSERT(node->Child(1)->Opcode() == air::core::OPC_ADD);
        AIR_ASSERT(node->Child(0)->Child(1)->Opcode() ==
                   air::core::OPC_INTCONST);

        if (Parse_subscript_expr(node->Child(1), subscript) == false) {
          return false;
        }

        int64_t inner_stride = Stride(subscript);
        AIR_ASSERT(inner_stride != -1 &&
                   node->Child(0)->Child(1)->Intconst() % inner_stride == 0);
        subscript.back().second =
            node->Child(0)->Child(1)->Intconst() / inner_stride;

        if (Parse_subscript_expr(node->Child(0)->Child(0), subscript) ==
            false) {
          return false;
        }

        for (int i = 0; i < subscript.size(); ++i) {
          const air::base::FUNC_SCOPE*    fscope = node->Func_scope();
          const air::base::ADDR_DATUM_PTR datum =
              fscope->Addr_datum(air::base::ADDR_DATUM_ID(subscript[i].first));
          printf("FULL: %d: %s -> %ld\n", i, datum->Name()->Char_str(),
                 subscript[i].second);
        }

#if 0
        if (Parse_subscript_expr(node->Child(0)->Child(0), subscript) == false) {
          return false;
        }
        AIR_ASSERT(subscript.back().second == -1);

        for (int i = 0; i < subscript.size(); ++i) {
          const air::base::FUNC_SCOPE* fscope = node->Func_scope();
          const air::base::ADDR_DATUM_PTR datum = fscope->Addr_datum(air::base::ADDR_DATUM_ID(subscript[i].first));
          printf("OUTER: %d: %s -> %ld\n", i, datum->Name()->Char_str(), subscript[i].second);
        }

        std::vector<std::pair<int64_t, int64_t> > inner;
        if (Parse_subscript_expr(node->Child(1), inner) == false) {
          return false;
        }
        int64_t inner_stride = Stride(inner);
        AIR_ASSERT(inner_stride != -1 &&
                   node->Child(0)->Child(1)->Intconst() % inner_stride == 0);

        for (int i = 0; i < subscript.size(); ++i) {
          printf("INNER: %d: %ld -> %ld\n", i, subscript[i].first, subscript[i].second);
        }

        subscript.back().second = node->Child(0)->Child(1)->Intconst() / inner_stride;
        subscript.insert(subscript.begin(), inner.begin(), inner.end());

        for (int i = 0; i < subscript.size(); ++i) {
          const air::base::FUNC_SCOPE* fscope = node->Func_scope();
          const air::base::ADDR_DATUM_PTR datum = fscope->Addr_datum(air::base::ADDR_DATUM_ID(subscript[i].first));
          printf("FULL: %d: %s -> %ld\n", i, datum->Name()->Char_str(), subscript[i].second);
        }
#endif
        return true;

        // AIR_ASSERT(subscript.back().second != -1 &&
        //            node->Child(0)->Child(1)->Intconst() %
        //            subscript.back().second == 0);
      } else {
        // iv_1 + iv_0 * stride_1
        AIR_ASSERT(node->Child(0)->Opcode() == air::core::OPC_LD);
        AIR_ASSERT(node->Child(1)->Opcode() == air::core::OPC_MUL);
        Parse_subscript_expr(node->Child(0), subscript);
        return Parse_subscript_expr(node->Child(1), subscript);
      }
    } else if (node->Opcode() == air::core::OPC_MUL) {
      // iv * stride
      AIR_ASSERT(node->Child(1)->Opcode() == air::core::OPC_INTCONST);
      AIR_ASSERT(subscript.size() > 0);
      AIR_ASSERT(subscript.back().second == -1);
      subscript.back().second = node->Child(1)->Intconst();
      return Parse_subscript_expr(node->Child(0), subscript);
    }
    return false;
  }

  fhe::core::RT_DATA_WRITER* _rt_data_writer;
  std::string                _data_file_uuid;
  fhe::core::DATA_ENTRY_TYPE _data_entry_type;
  bool                       _ct_encode = false;
  bool                       _need_bts = false;
  bool                       _has_phantom_manifests = false;
  PHANTOM_CONTEXT_DESCRIPTOR _phantom_context;
  PHANTOM_RESOURCE_DESCRIPTOR _phantom_resources;
  std::string                  _phantom_context_sha256;
  std::vector<PHANTOM_EMITTED_CONSTANT> _phantom_constants;
  size_t                       _observed_rotation_batches = 0;
};  // IR2C_CTX

}  // namespace ckks

}  // namespace fhe

#endif  // FHE_CKKS_IR2C_CTX_H
