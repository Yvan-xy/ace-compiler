//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef NN_VECTOR_TENSOR2VECTOR_DSL_H
#define NN_VECTOR_TENSOR2VECTOR_DSL_H

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "air/base/container.h"
#include "air/base/opcode.h"
#include "air/base/spos.h"
#include "air/base/st.h"

namespace nn {
namespace vector {

class TENSOR2VECTOR_CTX;

// A body builder is invoked only after the helper signature, entry point,
// function scope, entry statement, and body block have been created in the
// active Tensor-to-Vector destination GLOB_SCOPE. It must create every local,
// preg, constant, loop, and expression through those destination objects and
// return the helper-owned result expression. The materializer owns the single
// terminal RETV required by the frozen Phase-A ABI.
using VECTOR_KERNEL_BODY_BUILDER = std::function<air::base::NODE_PTR(
    air::base::FUNC_SCOPE&, air::base::NODE_PTR, const air::base::SPOS&)>;

struct VECTOR_KERNEL_HELPER_SPEC {
  // Stable plan key and deterministic helper name. The M4 planner connection
  // will populate these from VECTOR_KERNEL_PLAN. M1's synthetic fixture uses
  // a deterministic test specialization.
  std::string _specialization_key;
  std::string _helper_name;

  // All types must belong to the destination GLOB_SCOPE supplied to the
  // selector. In particular, source-node type pointers are never valid here.
  std::vector<air::base::TYPE_PTR> _formal_types;
  air::base::TYPE_PTR              _result_type;
  VECTOR_KERNEL_BODY_BUILDER       _build_body;
};

// A selector observes the original NN node but receives already visited
// destination-owned actuals. It returns a fully specialized helper recipe; it
// must not return or capture AIR objects from an independent module.
using VECTOR_KERNEL_LOWERING_SELECTOR =
    std::function<VECTOR_KERNEL_HELPER_SPEC(
        air::base::NODE_PTR, const std::vector<air::base::NODE_PTR>&,
        air::base::GLOB_SCOPE&)>;

// Registration is explicitly per VECTOR_CTX/Vector_driver invocation. This
// avoids process-global callbacks, stale module pointers, and shuffled-test
// order dependencies. Merely setting the legacy skip registry is not a DSL
// lowering registration.
class VECTOR_KERNEL_LOWERING_REGISTRY {
public:
  bool Register(air::base::OPCODE opcode,
                VECTOR_KERNEL_LOWERING_SELECTOR selector);
  bool Unregister(air::base::OPCODE opcode);
  void Clear();

  bool Has(air::base::OPCODE opcode) const;
  const VECTOR_KERNEL_LOWERING_SELECTOR* Lookup(
      air::base::OPCODE opcode) const;

private:
  std::unordered_map<uint32_t, VECTOR_KERNEL_LOWERING_SELECTOR> _selectors;
};

struct VECTOR_KERNEL_LOWERING_RESULT {
  air::base::FUNC_SCOPE* _helper_scope;
  air::base::STMT_PTR    _call_stmt;
  air::base::NODE_PTR    _replacement;
};

// If the current pass has a selector registered for source_node's opcode,
// materialize its helper directly in the active destination GLOB_SCOPE,
// prepend one typed CORE.CALL in the current caller, and return the caller's
// CORE.LDP replacement. Otherwise return an empty result.
std::optional<VECTOR_KERNEL_LOWERING_RESULT>
Try_materialize_registered_vector_kernel(
    TENSOR2VECTOR_CTX& ctx, air::base::NODE_PTR source_node,
    const std::vector<air::base::NODE_PTR>& actuals);

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_TENSOR2VECTOR_DSL_H
