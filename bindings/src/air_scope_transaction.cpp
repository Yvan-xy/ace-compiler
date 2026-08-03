//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "air_scope_transaction.h"

#include "air/base/container.h"

namespace ace::bindings {

using namespace air::base;

std::unique_ptr<GLOB_SCOPE> Clone_air_scope_with_code(GLOB_SCOPE& source) {
  auto clone = std::make_unique<GLOB_SCOPE>(source.Id(), true);
  clone->Clone(source);
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = source.Begin_func_scope();
       iter != source.End_func_scope(); ++iter) {
    FUNC_SCOPE& source_function = *iter;
    FUNC_SCOPE& candidate_function =
        clone->New_func_scope(source_function.Owning_func_id());
    candidate_function.Clone(source_function);
    STMT_PTR entry = candidate_function.Container().Clone_stmt_tree(
        source_function.Container().Entry_stmt());
    candidate_function.Set_entry_stmt(entry);
  }
  return clone;
}

}  // namespace ace::bindings
