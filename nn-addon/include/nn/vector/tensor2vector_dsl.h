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
#include "nn/vector/tensor2vector_planning.h"

namespace nn {
namespace vector {

class TENSOR2VECTOR_CTX;

// Structural marker carried by destination-owned generated helper calls.
// Generated-helper inliners select this attribute rather than helper names.
inline constexpr char VECTOR_KERNEL_GENERATED_CALL_ATTR[] =
    "ace.vector_kernel.generated_call";
inline constexpr char VECTOR_KERNEL_GENERATED_HELPER_ATTR[] =
    "ace.vector_kernel.generated_helper";

// A body builder is invoked only after the helper signature, entry point,
// function scope, entry statement, and body block have been created in the
// active Tensor-to-Vector destination GLOB_SCOPE. It must create every local,
// preg, constant, loop, and expression through those destination objects and
// return the helper-owned result expression. The materializer owns the single
// terminal RETV required by the frozen Phase-A ABI.
using VECTOR_KERNEL_BODY_BUILDER = std::function<air::base::NODE_PTR(
    air::base::FUNC_SCOPE&, air::base::NODE_PTR, const air::base::SPOS&)>;

struct VECTOR_KERNEL_HELPER_SPEC {
  // Stable plan key and deterministic helper name. Prepared plans populate
  // these from VECTOR_KERNEL_PLAN; synthetic fixtures use deterministic test
  // specializations.
  std::string _specialization_key;
  std::string _helper_name;

  // All types must belong to the destination GLOB_SCOPE supplied to the
  // selector. In particular, source-node type pointers are never valid here.
  std::vector<air::base::TYPE_PTR> _formal_types;
  air::base::TYPE_PTR              _result_type;
  VECTOR_KERNEL_BODY_BUILDER       _build_body;
};

// Destination-owned helper ABI prepared centrally after the provider package
// has been validated. Recipes may observe these types, but must not replace
// them or create AIR while selecting a body builder.
struct VECTOR_KERNEL_DESTINATION_ABI {
  std::vector<air::base::TYPE_PTR> _formal_types;
  air::base::TYPE_PTR              _result_type;
};

// A selector observes the original NN node but receives already visited
// destination-owned actuals. It returns a fully specialized helper recipe; it
// must not return or capture AIR objects from an independent module.
using VECTOR_KERNEL_LOWERING_SELECTOR =
    std::function<VECTOR_KERNEL_HELPER_SPEC(
        air::base::NODE_PTR, const std::vector<air::base::NODE_PTR>&,
        air::base::GLOB_SCOPE&)>;

// A canonical-plan recipe is selected only after the provider result has
// passed central validation. It receives the immutable prepared package and
// destination-owned actuals, and creates no AIR until the materializer invokes
// its returned body builder.
using VECTOR_KERNEL_PLAN_RECIPE =
    std::function<VECTOR_KERNEL_HELPER_SPEC(
        const PREPARED_VECTOR_KERNEL_PLAN&,
        const std::vector<air::base::NODE_PTR>&,
        air::base::GLOB_SCOPE&, const VECTOR_KERNEL_DESTINATION_ABI&)>;

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

  bool Register(VECTOR_KERNEL_PLAN_KIND kind,
                VECTOR_KERNEL_PLAN_RECIPE recipe);
  bool Unregister(VECTOR_KERNEL_PLAN_KIND kind);
  bool Has(VECTOR_KERNEL_PLAN_KIND kind) const;
  const VECTOR_KERNEL_PLAN_RECIPE* Lookup(
      VECTOR_KERNEL_PLAN_KIND kind) const;

  // Materialized helpers are cached per active destination GLOB_SCOPE. These
  // methods are used by the materializer after validating the complete
  // specialization key, deterministic name, and ordered signature.
  air::base::FUNC_SCOPE* Lookup_materialized_helper(
      air::base::GLOB_SCOPE& destination,
      const VECTOR_KERNEL_HELPER_SPEC& spec) const;
  void Remember_materialized_helper(
      air::base::GLOB_SCOPE& destination,
      const VECTOR_KERNEL_HELPER_SPEC& spec,
      air::base::FUNC_SCOPE& helper);
  // A registry may be reused by its owner, but helper IDs and destination
  // addresses are valid only for one Vector_driver invocation.
  void Clear_materialized_helpers();

private:
  struct MATERIALIZED_HELPER {
    std::string        _helper_name;
    air::base::FUNC_ID _helper_func_id;
  };

  using DESTINATION_HELPER_CACHE =
      std::unordered_map<std::string, MATERIALIZED_HELPER>;

  std::unordered_map<uint32_t, VECTOR_KERNEL_LOWERING_SELECTOR> _selectors;
  std::unordered_map<uint8_t, VECTOR_KERNEL_PLAN_RECIPE> _plan_recipes;
  std::unordered_map<air::base::GLOB_SCOPE*, DESTINATION_HELPER_CACHE>
      _materialized_helpers;
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

// Materialize the recipe registered for prepared.Plan()'s canonical kind.
// The prepared specialization key and helper name are authoritative: a recipe
// may leave them empty, but may not replace them. The helper is cached by the
// complete prepared specialization key in the active destination GLOB_SCOPE.
std::optional<VECTOR_KERNEL_LOWERING_RESULT>
Try_materialize_prepared_vector_kernel(
    TENSOR2VECTOR_CTX& ctx,
    const PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const std::vector<air::base::NODE_PTR>& actuals,
    const air::base::SPOS& spos);

}  // namespace vector
}  // namespace nn

#endif  // NN_VECTOR_TENSOR2VECTOR_DSL_H
