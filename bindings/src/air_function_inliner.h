//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef ACE_BINDINGS_AIR_FUNCTION_INLINER_H
#define ACE_BINDINGS_AIR_FUNCTION_INLINER_H

#include <cstdint>
#include <string>

#include "air/base/st.h"

namespace ace::bindings {

// Temporary baseline-kernel E2E bridge. This binding-side algorithm is not
// the independent Python inliner planned for M14. M15 replaces its production
// use, and M16 removes it after differential and downstream validation.

struct AIR_FUNCTION_INLINE_RESULT {
  bool        _success         = false;
  bool        _changed         = false;
  uint32_t    _calls_inlined   = 0;
  uint32_t    _helpers_removed = 0;
  std::string _diagnostic;
};

// Inline structurally tagged, same-module, leaf helper calls. The caller owns
// the supplied GLOB_SCOPE and is responsible for transactional clone/swap.
AIR_FUNCTION_INLINE_RESULT Inline_tagged_leaf_helpers(
    air::base::GLOB_SCOPE& glob, const char* call_attribute,
    const char* helper_attribute);

}  // namespace ace::bindings

#endif  // ACE_BINDINGS_AIR_FUNCTION_INLINER_H
