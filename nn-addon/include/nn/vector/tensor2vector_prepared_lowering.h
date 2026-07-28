//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_PREPARED_LOWERING_H
#define NN_VECTOR_TENSOR2VECTOR_PREPARED_LOWERING_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "air/base/node.h"
#include "air/base/st.h"
#include "nn/vector/tensor2vector_planning.h"

namespace nn {
namespace vector {

class TENSOR2VECTOR_CTX;

struct VECTOR_KERNEL_REQUEST_BUILD_RESULT {
  std::optional<VECTOR_KERNEL_PLANNING_REQUEST> _request;
  std::string                                   _diagnostic;

  bool Ok() const { return _request.has_value(); }
};

// Snapshot one source NN Conv/Gemm before any destination AIR mutation. The
// returned request owns all attribute and constant bytes and contains no AIR
// handles, IDs, symbols, or borrowed buffers.
VECTOR_KERNEL_REQUEST_BUILD_RESULT Build_vector_kernel_planning_request(
    TENSOR2VECTOR_CTX& ctx, air::base::NODE_PTR source_node,
    VECTOR_KERNEL_REQUESTED_PLAN_KIND requested_kind);

// Materialize one centrally validated package through the native emitter.
// The ranked input and optional raw sharding scalar must already belong to the
// active Tensor-to-Vector destination. Constants and all remaining AIR are
// created in that same destination.
air::base::NODE_PTR Emit_prepared_vector_kernel_native(
    TENSOR2VECTOR_CTX& ctx,
    const PREPARED_VECTOR_KERNEL_PLAN& prepared,
    air::base::NODE_PTR ranked_input,
    const std::vector<air::base::NODE_PTR>& scalar_actuals,
    const air::base::SPOS& spos);

// Return the only source-side runtime scalar used by the v1 ABI: the raw s32
// sharded-Conv weight-block index. The caller visits/clones the returned node
// only after planning and validation succeed.
air::base::NODE_PTR Find_vector_kernel_source_scalar(
    air::base::NODE_PTR source_node);

// Look up and materialize centrally validated prepared constants. The
// materialized ARRAY constant is owned by destination and contains a private
// host-order copy of the canonical little-endian payload.
const VECTOR_KERNEL_TYPED_PAYLOAD* Find_prepared_vector_kernel_constant(
    const PREPARED_VECTOR_KERNEL_PLAN& prepared, std::string_view role);

air::base::CONSTANT_PTR Materialize_prepared_vector_kernel_constant(
    air::base::GLOB_SCOPE& destination,
    const VECTOR_KERNEL_TYPED_PAYLOAD& payload,
    const air::base::SPOS& spos);

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_TENSOR2VECTOR_PREPARED_LOWERING_H
