//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef ACE_BINDINGS_AIR_SCOPE_TRANSACTION_H
#define ACE_BINDINGS_AIR_SCOPE_TRANSACTION_H

#include <memory>

#include "air/base/st.h"

namespace ace::bindings {

// Clone global tables, function-local tables, and every function body into an
// independently owned scope. Global and local symbol IDs remain stable; code
// IDs are destination-owned and must be remapped structurally by the caller.
std::unique_ptr<air::base::GLOB_SCOPE> Clone_air_scope_with_code(
    air::base::GLOB_SCOPE& source);

}  // namespace ace::bindings

#endif  // ACE_BINDINGS_AIR_SCOPE_TRANSACTION_H
