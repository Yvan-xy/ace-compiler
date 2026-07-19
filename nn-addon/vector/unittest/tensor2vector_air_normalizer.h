//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_UNITTEST_TENSOR2VECTOR_AIR_NORMALIZER_H
#define NN_VECTOR_UNITTEST_TENSOR2VECTOR_AIR_NORMALIZER_H

#include <cstddef>
#include <string>
#include <vector>

#include "air/base/container.h"
#include "air/base/st.h"

namespace nn {
namespace vector {
namespace test {

// The native emitter may live inside a larger caller. The view identifies
// only its generated statement interval, semantic inputs, and logical result.
// _end_stmt is exclusive. Normalization synthesizes the same logical return
// boundary used by a destination-owned DSL helper.
struct VECTOR_KERNEL_NATIVE_AIR_VIEW {
  const air::base::FUNC_SCOPE*       _scope;
  std::vector<air::base::NODE_PTR>   _inputs;
  air::base::STMT_PTR                _first_stmt;
  air::base::STMT_PTR                _end_stmt;
  air::base::NODE_PTR                _result;
};

struct VECTOR_KERNEL_AIR_COMPARE_RESULT {
  bool        _equal;
  size_t      _mismatch_offset;
  std::string _message;
  std::string _lhs;
  std::string _rhs;
};

// Compatibility entry point for the frozen M0 whole-function fingerprints.
std::string Normalize_vector_kernel_function(
    const air::base::FUNC_SCOPE& scope);

// Reusable native-versus-helper adapters for Phase-A structural gates.
std::string Normalize_native_vector_kernel(
    const VECTOR_KERNEL_NATIVE_AIR_VIEW& view);
std::string Normalize_vector_kernel_helper(
    const air::base::FUNC_SCOPE& helper_scope);

VECTOR_KERNEL_AIR_COMPARE_RESULT Compare_normalized_vector_kernel_air(
    const std::string& lhs, const std::string& rhs);
VECTOR_KERNEL_AIR_COMPARE_RESULT Compare_native_and_helper_vector_kernel_air(
    const VECTOR_KERNEL_NATIVE_AIR_VIEW& native_view,
    const air::base::FUNC_SCOPE&         helper_scope);

// Validate the bridge independently from the kernel body: exact destination
// entry identity, expected actual-node order and types, caller-owned result
// preg, the attached replacing LDP, one call site, and the helper's single
// terminal RETV.
VECTOR_KERNEL_AIR_COMPARE_RESULT Check_vector_kernel_call_bridge(
    const air::base::FUNC_SCOPE& caller_scope, air::base::STMT_PTR call_stmt,
    const std::vector<air::base::NODE_PTR>& expected_actuals,
    air::base::NODE_PTR replacement,
    const air::base::FUNC_SCOPE& helper_scope);

}  // namespace test
}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_UNITTEST_TENSOR2VECTOR_AIR_NORMALIZER_H
