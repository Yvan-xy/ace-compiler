//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_PLANNING_H
#define NN_VECTOR_TENSOR2VECTOR_PLANNING_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nn/vector/tensor2vector_plan.h"

namespace nn {
namespace vector {

// This file is the AIR-independent provider-neutral planning boundary. None of
// the records below may contain AIR objects, IDs, symbols, type pointers, or
// borrowed payloads.

enum class VECTOR_KERNEL_OPERATION : uint8_t {
  GEMM,
  CONV,
};

enum class VECTOR_KERNEL_REQUESTED_PLAN_KIND : uint8_t {
  AUTO,
  BASELINE_GEMM,
  BASELINE_CONV,
  FAST_GEMM,
  FAST_CONV,
};

enum class VECTOR_KERNEL_PLAN_PROVIDER_KIND : uint8_t {
  CPP,
  PYTHON,
};

enum class VECTOR_KERNEL_IMPLEMENTATION : uint8_t {
  NATIVE,
  DSL,
};

enum class VECTOR_KERNEL_FALLBACK_POLICY : uint8_t {
  ERROR,
  CPP_NATIVE,
};

enum class VECTOR_KERNEL_RUNTIME_PREPARATION_KIND : uint8_t {
  PACKED_VECTOR,
  FLATTEN_PACKED_VECTOR,
  BLOCKING_ROTATIONS,
};

struct VECTOR_KERNEL_TYPED_PAYLOAD {
  const std::string                    _role;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN _type;
  // Canonical row-major little-endian bytes.
  const std::vector<uint8_t> _bytes;
  // Provider-supplied value. The central validator always recomputes it.
  const std::string _content_hash;
};

struct VECTOR_KERNEL_ATTRIBUTE_RECORD {
  const std::string                    _name;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN _type;
  const std::vector<uint8_t>           _bytes;
};

struct VECTOR_KERNEL_OPTION_SNAPSHOT {
  const bool _conv_parallel;
  const bool _mask_fuse;
  const bool _selective_strided_slice;
  const bool _sharding;
  const bool _is_last_operation;
};

struct VECTOR_KERNEL_TARGET_SNAPSHOT {
  const int64_t _num_slots;
  const int64_t _min_slots;
  const int64_t _max_slots;
};

struct VECTOR_KERNEL_PLANNING_REQUEST {
  const VECTOR_KERNEL_OPERATION _operation;
  const std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> _attributes;
  const std::vector<VECTOR_KERNEL_RANKED_TYPE_PLAN> _operand_types;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN _declared_result_type;
  const VECTOR_KERNEL_OPTION_SNAPSHOT _options;
  const VECTOR_KERNEL_TARGET_SNAPSHOT _target;
  const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> _source_constants;
  const VECTOR_KERNEL_REQUESTED_PLAN_KIND _requested_plan_kind;
  // Ordered source-side runtime scalar types. V1 uses one s32 value only for
  // a sharded fast-Conv weight offset.
  const std::vector<air::base::PRIMITIVE_TYPE> _runtime_scalar_types = {};
};

struct VECTOR_KERNEL_RUNTIME_PREPARATION {
  const std::string _role;
  const uint32_t _source_operand;
  const VECTOR_KERNEL_RUNTIME_PREPARATION_KIND _kind;
  const VECTOR_KERNEL_RANKED_TYPE_PLAN _result_type;
  const int64_t _logical_input_size;
  const int64_t _replications;
  const int64_t _blocking_width;
  const std::vector<int32_t> _rotation_candidates;
  const uint32_t _outer_block_depth;
};

struct VECTOR_KERNEL_SCALAR_PREPARATION {
  const std::string _role;
  const uint32_t _source_operand;
  const air::base::PRIMITIVE_TYPE _type;
  const int64_t _scale;
};

// Provider output is untrusted host data. It still owns every byte; the
// validator copies it into PREPARED_VECTOR_KERNEL_PLAN after validation.
struct VECTOR_KERNEL_PROVIDER_RESULT {
  VECTOR_KERNEL_PROVIDER_RESULT(
      VECTOR_KERNEL_PLAN plan,
      std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> constants,
      std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime_preparations,
      std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> scalar_preparations,
      std::string provenance)
      : _plan(std::move(plan)),
        _constants(std::move(constants)),
        _runtime_preparations(std::move(runtime_preparations)),
        _scalar_preparations(std::move(scalar_preparations)),
        _provenance(std::move(provenance)) {}

  VECTOR_KERNEL_PLAN _plan;
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> _constants;
  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> _runtime_preparations;
  std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> _scalar_preparations;
  std::string _provenance;
};

struct VECTOR_KERNEL_PREPARE_RESULT;

class PREPARED_VECTOR_KERNEL_PLAN {
public:
  const VECTOR_KERNEL_PLAN& Plan() const { return _plan; }
  const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD>& Constants() const {
    return _constants;
  }
  const std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION>&
  Runtime_preparations() const {
    return _runtime_preparations;
  }
  const std::vector<VECTOR_KERNEL_SCALAR_PREPARATION>&
  Scalar_preparations() const {
    return _scalar_preparations;
  }
  const std::string& Provenance() const { return _provenance; }
  const std::string& Specialization_key() const {
    return _specialization_key;
  }
  const std::string& Helper_name() const { return _helper_name; }

private:
  PREPARED_VECTOR_KERNEL_PLAN(
      VECTOR_KERNEL_PLAN plan,
      std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> constants,
      std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime_preparations,
      std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> scalar_preparations,
      std::string provenance, std::string specialization_key,
      std::string helper_name)
      : _plan(std::move(plan)),
        _constants(std::move(constants)),
        _runtime_preparations(std::move(runtime_preparations)),
        _scalar_preparations(std::move(scalar_preparations)),
        _provenance(std::move(provenance)),
        _specialization_key(std::move(specialization_key)),
        _helper_name(std::move(helper_name)) {}

  const VECTOR_KERNEL_PLAN _plan;
  const std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> _constants;
  const std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION>
      _runtime_preparations;
  const std::vector<VECTOR_KERNEL_SCALAR_PREPARATION> _scalar_preparations;
  const std::string _provenance;
  const std::string _specialization_key;
  const std::string _helper_name;

  friend VECTOR_KERNEL_PREPARE_RESULT Validate_and_prepare_vector_kernel_plan(
      const VECTOR_KERNEL_PLANNING_REQUEST&,
      const VECTOR_KERNEL_PROVIDER_RESULT&);
};

enum class VECTOR_KERNEL_PLANNING_ERROR : uint8_t {
  NONE,
  INVALID_SELECTION,
  INVALID_REQUEST,
  PROVIDER_UNAVAILABLE,
  PROVIDER_FAILURE,
  INVALID_PROVIDER_RESULT,
  UNSUPPORTED_DSL_RECIPE,
};

struct VECTOR_KERNEL_PROVIDER_CALL_RESULT {
  std::optional<VECTOR_KERNEL_PROVIDER_RESULT> _result;
  std::string _diagnostic;

  static VECTOR_KERNEL_PROVIDER_CALL_RESULT Success(
      VECTOR_KERNEL_PROVIDER_RESULT result) {
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT{std::move(result), {}};
  }
  static VECTOR_KERNEL_PROVIDER_CALL_RESULT Failure(std::string diagnostic) {
    return VECTOR_KERNEL_PROVIDER_CALL_RESULT{std::nullopt,
                                               std::move(diagnostic)};
  }
};

class VECTOR_KERNEL_PLAN_PROVIDER {
public:
  virtual ~VECTOR_KERNEL_PLAN_PROVIDER() = default;
  virtual const char* Name() const = 0;
  virtual VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const = 0;
};

class CPP_VECTOR_KERNEL_PLAN_PROVIDER final
    : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "cpp"; }
  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const override;
};

class VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY {
public:
  bool Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND kind,
                const VECTOR_KERNEL_PLAN_PROVIDER* provider);
  bool Unregister(VECTOR_KERNEL_PLAN_PROVIDER_KIND kind);
  const VECTOR_KERNEL_PLAN_PROVIDER* Lookup(
      VECTOR_KERNEL_PLAN_PROVIDER_KIND kind) const;

private:
  std::array<const VECTOR_KERNEL_PLAN_PROVIDER*, 2> _providers{{nullptr,
                                                                nullptr}};
};

struct VECTOR_KERNEL_SELECTION {
  VECTOR_KERNEL_PLAN_PROVIDER_KIND _plan_provider;
  VECTOR_KERNEL_IMPLEMENTATION _kernel_implementation;
  VECTOR_KERNEL_REQUESTED_PLAN_KIND _plan_kind;
  VECTOR_KERNEL_FALLBACK_POLICY _fallback;
};

struct VECTOR_KERNEL_SELECTION_RESULT {
  std::optional<VECTOR_KERNEL_SELECTION> _selection;
  std::string _diagnostic;
};

struct VECTOR_KERNEL_PREPARE_RESULT {
  std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> _prepared;
  VECTOR_KERNEL_PLANNING_ERROR _error;
  std::string _diagnostic;

  bool Ok() const { return _prepared != nullptr; }
};

struct VECTOR_KERNEL_RESOLUTION_RESULT {
  std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> _prepared;
  VECTOR_KERNEL_PLAN_PROVIDER_KIND _resolved_provider;
  VECTOR_KERNEL_IMPLEMENTATION _resolved_implementation;
  VECTOR_KERNEL_PLANNING_ERROR _error;
  bool _used_fallback;
  std::string _diagnostic;

  bool Ok() const { return _prepared != nullptr; }
};

using VECTOR_KERNEL_DSL_RECIPE_QUERY =
    std::function<bool(VECTOR_KERNEL_PLAN_KIND)>;

const char* Vector_kernel_operation_name(VECTOR_KERNEL_OPERATION operation);
const char* Vector_kernel_requested_plan_kind_name(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind);

VECTOR_KERNEL_SELECTION_RESULT Parse_vector_kernel_selection(
    const std::string& plan_provider, const std::string& kernel_impl,
    const std::string& plan_kind, const std::string& fallback);

VECTOR_KERNEL_PREPARE_RESULT Validate_and_prepare_vector_kernel_plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const VECTOR_KERNEL_PROVIDER_RESULT& provider_result);

VECTOR_KERNEL_RESOLUTION_RESULT Resolve_vector_kernel_plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request,
    const VECTOR_KERNEL_SELECTION& selection,
    const VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY* provider_registry,
    VECTOR_KERNEL_DSL_RECIPE_QUERY has_dsl_recipe = {});

bool Equal_vector_kernel_prepared_semantics(
    const PREPARED_VECTOR_KERNEL_PLAN& lhs,
    const PREPARED_VECTOR_KERNEL_PLAN& rhs);

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_TENSOR2VECTOR_PLANNING_H
