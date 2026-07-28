//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_CPP_CONV_PROVIDER_H
#define NN_VECTOR_TENSOR2VECTOR_CPP_CONV_PROVIDER_H

#include "nn/vector/tensor2vector_planning.h"

namespace nn {
namespace vector {

// AIR-independent C++ reference planning for Conv. The request and returned
// package own every byte used by planning and materialization.
VECTOR_KERNEL_PROVIDER_CALL_RESULT
Plan_cpp_vector_kernel_conv(const VECTOR_KERNEL_PLANNING_REQUEST &request);

} // namespace vector
} // namespace nn

#endif // NN_VECTOR_TENSOR2VECTOR_CPP_CONV_PROVIDER_H
