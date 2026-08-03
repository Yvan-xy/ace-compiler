//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef ACE_BINDINGS_VECTOR_KERNEL_PLAN_PROVIDER_H
#define ACE_BINDINGS_VECTOR_KERNEL_PLAN_PROVIDER_H

#include <pybind11/pybind11.h>

#include "nn/vector/tensor2vector_planning.h"

namespace pyace {

// Synchronous adapter from the provider-neutral C++ planning boundary to a
// Python planning callback.  The callback and every Python transport object
// are scoped to one Vector_driver invocation.
class PYTHON_VECTOR_KERNEL_PLAN_PROVIDER final
    : public nn::vector::VECTOR_KERNEL_PLAN_PROVIDER {
public:
  explicit PYTHON_VECTOR_KERNEL_PLAN_PROVIDER(pybind11::object callback);

  const char* Name() const override { return "python"; }

  nn::vector::VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const nn::vector::VECTOR_KERNEL_PLANNING_REQUEST& request)
      const override;

private:
  pybind11::object _callback;
};

}  // namespace pyace

#endif  // ACE_BINDINGS_VECTOR_KERNEL_PLAN_PROVIDER_H
