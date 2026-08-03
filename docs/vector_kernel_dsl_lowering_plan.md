# Vector-Kernel DSL Lowering Implementation Plan

## Goal

Extend `ace_edsl` so that user-defined `@vector_kernel` functions can implement
the Tensor-to-Vector lowering of `NN.GEMM` and `NN.CONV` while preserving the
native AIR loop structure.

The intended lowering is:

```text
NN.GEMM / NN.CONV
        |
        v
AIR-independent VectorKernelPlanningRequest
        |
        v
selected PlanProvider
  cpp (default/reference) | python (M12 complete)
        |
        v
C++ validation, canonicalization, and owned PreparedVectorKernelPlan
        |
        v
selected kernel implementation
  native | destination-owned @vector_kernel helper
        |
        v
native Vector AIR | CORE.CALL + CORE.LDP in the original caller
        |
        v
Vector middle-op lowering (`Mv2v_opt`)
        |
        v
verified pre-inline Vector AIR
        |
        | Phase C Python AIR pass pipeline for the DSL branch
        v
ordered AIRPassManager at BEFORE_VECTOR2SIHE
  analyses: CallGraphAnalysis, FunctionReachabilityAnalysis
  transforms: FunctionInlinerPass, DeadFunctionEliminationPass
        |
        v
verified call-free Vector AIR
        |
        v
CORE.DO_LOOP + VECTOR.ROLL/SLICE/MUL/ADD + CORE memory operations
        |
        v
Vector -> SIHE -> CKKS -> POLY -> C
```

There are no `VECTOR.GEMM` or `VECTOR.CONV` operators. `@vector_kernel`
describes how an NN operator is decomposed into Vector-domain operations. The
loop, block, scalar-index, and memory structure remains AIR Core IR.

Planning runs at compile time before either kernel implementation is invoked.
The prepared plan is host data whose decisions and constants are emitted into
the body; the object itself is not an AIR formal or traced by `@vector_kernel`.

## Scope

This plan covers:

- preserving the counted loops in `New_gemm_metakernel`,
  `New_conv_metakernel`, `New_gemm_metakernel_fast`, and
  `New_conv_metakernel_fast`;
- materializing each selected DSL lowering as a specialized helper function in
  the active Tensor-to-Vector destination module;
- inserting a typed AIR call whose result replaces the original NN expression;
- producing structurally verified pre-inline Vector AIR for the four kernels;
- implementing, after the completed DSL/planning phases, a generic Python AIR
  analysis/transformation framework and independent inliner/cleanup clients
  that remove generated helper calls before Vector-to-SIHE;
- invoking a DSL lowering during the real Tensor-to-Vector pass;
- adding the missing Core and Vector authoring APIs to `ace_edsl`;
- defining a provider-neutral planning request and prepared-plan package;
- selecting the plan provider, kernel implementation, and forced plan kind
  independently;
- retaining the existing C++ planning, packing, and cost-model algorithms as
  the initial default/reference provider;
- matching the native lowering structurally and end to end;
- completing NN-level Conv/Gemm authoring after the lowering substrate is
  stable.

Initial non-goals:

- implementing the production Python planning, packing, and cost-model
  algorithms before the four kernel implementations are stable;
- allowing a plan provider to return AIR nodes, IDs, pointers, symbols, or
  borrowed buffers;
- silently changing provider or kernel implementation after a planning or
  emission failure;
- inventing composite `VECTOR.GEMM` or `VECTOR.CONV` operators;
- unrolling the native loop structure to avoid implementing Core control flow;
- accepting placeholder arithmetic in place of `VECTOR.ROLL` or
  `VECTOR.SLICE`;
- implementing the Python AIR pass framework or function-inliner pass during
  Phase A;
- allowing generated Vector helper calls to survive into the FHE pipeline
  before the interprocedural FHE call ABI is completed;
- changing the downstream Vector-to-SIHE, SIHE-to-CKKS, or CKKS-to-POLY
  semantics.

## Delivery Phases

### Phase A: Gemm/Conv DSL support (complete through M11)

Phase A covered the DSL and AIR needed to express all four Gemm/Conv
metakernels as destination-owned helper functions. It ended with valid
pre-inline Vector AIR after `Mv2v_opt`.

This phase includes:

- the helper-function ABI and `CORE.CALL + CORE.LDP` bridge;
- Core loops, typed scalar/index arithmetic, typed locals, and typed zero;
- genuine Vector roll, slice, add, multiply, and attributes;
- mutable arrays of vectors needed by the fast kernels;
- the provider-neutral planning contract, C++ reference provider, central
  validator/materializer, and all four DSL kernel bodies;
- complete typed NN-level Conv/Gemm authoring;
- native-versus-DSL comparison of pre-inline Vector AIR.

Phase A implemented neither the production Python planning algorithms nor the
Python AIR pass framework or function inliner, and it did not claim full
Vector-to-SIHE-to-C completion for DSL-generated helper calls.

### Phase B: Python planning provider (complete at M12)

After Phase A became structurally complete, M12 implemented the four planning
algorithms behind the provider-neutral M4 interface in ordinary, separately
testable Python. The Python provider computes host-side plan data only; it does
not run inside `@vector_kernel` and does not build or transform AIR. C++ remains
the owner and authoritative validator/materializer of every returned package.

### Phase C: Generic Python AIR passes and downstream integration (M13 next)

After the provider/emitter matrix is validated, first establish a
domain-neutral Python framework for read-only AIR analyses and transactional AIR
transformations. The framework owns deterministic pass sequencing, analysis
caching and invalidation, structured results and diagnostics, safe structural
bindings, verification, rollback, and a generic pipeline point between
Tensor-to-Vector and Vector-to-SIHE.

Then implement function inlining and dead-function cleanup as ordinary,
independent clients of that framework. Neither pass is part of the Gemm/Conv
kernel implementation or embedded inside the C++ Vector driver. They transform
the same destination AIR `GLOB_SCOPE` produced by Tensor-to-Vector before full
downstream validation and controlled rollout.

## Baseline State at Plan Inception

### What already worked

`range_dynamic` already creates real destination-owned AIR `CORE.DO_LOOP`
statements. The loop body is traced once, stored in a Core block, and executed
at runtime by the generated loop. Nested `range_dynamic` loops are appended to
their enclosing loop body in source order.

Relevant implementation:

- `ace_edsl/edsl/domain_ast_decorators.py:83-221`
- `bindings/src/air_builder_bindings.cpp:1675-1930`

Therefore, the DSL does not need a new loop dialect. Metakernel loops should be
written with `range_dynamic`; `range_constexpr` and ordinary compile-time Python
iteration remain useful only when intentional unrolling is desired.

AIR already represents a value-returning call as a root `CORE.CALL` statement
whose result is written to a caller preg. A `CORE.LDP` of that preg can replace
the original NN expression. This is the call bridge the DSL lowering should
use; the helper body remains a complete function ending in one `CORE.RETV`.

The existing Tensor-to-Vector Python prototype demonstrates the relevant
caller-side ownership mechanism: destination-owned statements can be
prepended through the transform context, and a destination-owned load can
replace the original expression. The new design uses that mechanism to insert
the call rather than transplanting the helper body directly.

Relevant implementation:

- `nn-addon/vector/src/tensor2vector_py_airgen.cxx:25-130`
- `air-infra/include/air/base/transform_ctx.h:39-60`
- `air-infra/base/src/container.cxx:865-878`
- `air-infra/plugin/dsl_pybind11/src/py_airgen.cxx:141-154`

### What did not yet work

At plan inception, the DSL could preserve a simple loop, but it could not
faithfully express or install the four Conv/Gemm metakernels. The baseline gaps
are catalogued below.

## DSL Gap Inventory

### G1. No destination-module Vector-kernel function materialization and call bridge

`ace_edsl` already treats a decorated kernel as a complete function, which is
the right authoring unit for a loop-bearing metakernel. The missing capability
is to materialize that function inside the active Tensor-to-Vector destination
`GLOB_SCOPE` and install a call that replaces an existing `NN.CONV` or
`NN.GEMM` expression.

The helper must not be compiled into an independent DSL module and then cloned
by raw IDs. Its signature, entry point, function scope, constants, locals,
pregs, loop IVs, blocks, and body must all be created in
`ctx.Container()->Glob_scope()`, which is the destination module constructed by
`Lower_to_vector`.

Initial helper ABI:

```text
leaf, nonrecursive @vector_kernel helper
  runtime packed-vector input(s)
  compile-time PreparedVectorKernelPlan input to helper tracing (not a formal)
  same-GLOB_SCOPE weight/bias/mask/rotation constants
  one ranked-vector result
  one final CORE.RETV
```

Shape, layout, loop bounds, rotation candidates, masking policy, and every other
planning decision are frozen in a centrally validated
`PreparedVectorKernelPlan` before helper tracing. Constant weights and bias are
materialized by C++ as same-module `LDC` nodes instead of ordinary array
formals, because current Vector-to-SIHE parameter conversion does not
distinguish plaintext arrays from ciphertext arrays. Deterministic helper
naming and destination-local caching use the canonical immutable plan and
constant content hashes. Provider, kernel-implementation, and provenance
labels are diagnostic metadata only; they do not affect the specialization key
or helper identity.

Required caller bridge:

1. allocate a caller preg compatible with the helper return type;
2. create a root `CORE.CALL` to the destination-owned entry point;
3. populate its arguments and prepend it at the current transform position;
4. return a caller-owned `CORE.LDP` of the result preg as the replacement for
   the original NN expression.

Skipping the native lowering without first installing this helper and call
would leave an invalid mixed-domain pipeline.

Relevant implementation:

- `nn-addon/vector/src/vector.cxx:249-277`
- `air-infra/base/src/container.cxx:865-878`

### G2. No generic Python AIR analysis/transformation pass framework

The existing Python pass surface is not a framework. The tentative
`FunctionInlinerPass` is a thin facade over one binding-side algorithm and its
`PassResult` hard-codes inliner counters. The CKKS rewrite is a free function
with a different dictionary result. Both public pipeline classes special-case
the inliner independently, while the older compile helper has separate native
pass failure behavior. The pybind `PassManager` sequences placeholder pass
names but does not run Python-authored AIR analyses or transformations.

Required capability:

- domain-neutral `AnalysisPass[T]` and `TransformationPass` contracts whose
  immutable configuration objects never own live AIR wrappers;
- a serial, explicitly ordered `AIRPassManager` over one `GLOB_SCOPE` per
  invocation, with no process-global pass registry or automatic reordering;
- a common generic result model containing pass identity, status, `changed`,
  structured diagnostics, pass-specific metrics, AIR revisions, and preserved
  analyses, with no inliner-specific fields;
- an invocation-scoped `PassContext` and `AnalysisManager` that resolve
  declared analysis dependencies, detect cycles, and cache immutable host-data
  results by analysis configuration, IR-unit identity, and module revision;
- conservative invalidation after a changed transformation, explicit
  `PreservedAnalyses`, transitive invalidation of dependent analyses, and no
  cache transfer between modules or pipeline runs;
- read-only deterministic structural views for analyses and transaction-scoped
  mutable views for transformations;
- clone/verify/commit transactions whose failures, exceptions, or verifier
  errors leave committed AIR and the post-required-analysis cache checkpoint
  unchanged;
- a monotonic module revision and generation-checked wrappers so references
  from a committed or rolled-back transaction fail deterministically instead of
  silently crossing AIR ownership;
- deterministic diagnostics and metrics, plus generic pass instrumentation for
  timing, IR dumps, and begin/end callbacks without printing from pass code;
- one named, ordered pipeline extension point after Tensor-to-Vector/`Mv2v_opt`
  and immediately before Vector-to-SIHE, shared by `AcePipeline`, `Pipeline`,
  and `ace_compile`.

The initial scheduler is module-rooted and sequential, with explicit module or
function analysis requests. Analysis results may describe blocks and call
graphs as immutable host data, but M13 does not add block-level scheduling,
parallel execution, fixpoint scheduling, automatic pass ordering, persistent
caches, plugin discovery, raw-pointer mutation, or cross-`GLOB_SCOPE`
transformations. Native C++ lowering phases remain authoritative opaque
transformations. Every opaque native mutation advances the same module
revision/generation and invalidates cached analyses and outstanding child
views; a manager or view may not silently span such a barrier.

The structural binding substrate must expose deterministic function, entry,
formal, local, preg, block, statement, child-node, call, return, symbol,
loop-IV, type, source-position, and attribute views. Checked edit operations
must cover clone/remap, insert, replace, erase, block-parent repair, function
reachability/removal, verification, and atomic commit. Python owns pass policy
and algorithms; C++ owns native memory safety, type/ownership checks, cloning
mechanics, parent-link repair, verification, and scope replacement.

Relevant implementation:

- `ace_edsl/edsl/passes/function_inliner.py:1-81`
- `ace_edsl/edsl/passes/ckks_extended_ops_rewrite.py:15-58`
- `ace_edsl/edsl/pipeline.py:420-530,1040-1110`
- `ace_edsl/edsl/compiler.py:348-369`
- `bindings/src/passmanager_bindings.cpp:100-194`
- `bindings/src/air_builder_bindings.cpp:4843-6309,9130-9275`
- `air-infra/base/src/container.cxx:865-912`
- `air-infra/base/src/st.cxx:1382-1391`

### G3. Nested kernel domain handling is not lexical

Calling a `@vector_kernel` inside another kernel changes a shared
`current_domain`, but incoming `AIRValue` objects retain the domain captured
when they were created. Consequently, an NN-origin operand can still emit
`NN.ADD` or `NN.MUL` inside a Vector lowering.

Required capability:

- push and restore domains lexically for nested kernels;
- re-view or rewrap incoming operands as Vector-domain values;
- preserve the target domain on values returned by loops and conditionals;
- remove the fallback that chooses Vector arithmetic merely because a builder
  happens to expose `new_vec_add` or `new_vec_mul`.

Relevant implementation:

- `ace_edsl/edsl/domain_kernels.py:79-95`
- `ace_edsl/edsl/edsl.py:128-140`
- `ace_edsl/edsl/core/air_value.py:211-304`

### G4. Core scalar/index operations are incomplete

The metakernels use Core scalar arithmetic for loop indices, slice indices, and
rotation amounts. Examples include `cin * kernel_hw + khw` and the baseline
Gemm reduction expression `1 << iv`.

Current problems:

- loop IV values carry no explicit Core/scalar domain;
- domainless `+` and `*` can incorrectly become Vector operations;
- loop IVs and constants are forced to signed 64-bit, while the native
  metakernels use signed 32-bit values;
- there is no normal DSL `SHL` operation;
- casts between index widths are not explicit.

Required capability:

- typed Core scalar values separate from Vector values;
- signed 32-bit constants and loop IVs, or explicit checked casts;
- Core `ADD`, `MUL`, `LT`, and `SHL`;
- type-driven operator dispatch.

### G5. Genuine Vector primitives and metadata are missing

The active binding exposes genuine Vector add and multiply, but it does not
expose the essential metakernel operations:

- dynamic `VECTOR.ROLL`;
- dynamic `VECTOR.SLICE`;
- signed rotation candidate lists in `ATTR::RNUM`;
- correct slice result types;
- output `ATTR::SLOT`;
- the native Vector add result-type rule for differently sized operands.

Required capability:

- bind the existing `VECTOR_GEN` implementations rather than create synthetic
  wrapper nodes;
- accept a runtime/Core scalar shift or slice index;
- attach validated signed `RNUM` candidates, including negative rotations;
- infer or receive the precise result type of each operation.

Relevant implementation:

- `nn-addon/vector/src/vector_gen.cxx:34-218`
- `ace_edsl/edsl/core/vector_ops.py:12-128`
- `bindings/src/air_builder_bindings.cpp:525-572`

### G6. Typed locals and typed zero values are missing

All four native metakernels create vector-typed accumulator locals and
initialize them with a zero of exactly the result type. The DSL currently has
named `new_stid`/`new_ldid`, but local types are inferred from the first store
and `new_zero()` creates only an integer zero.

Required capability:

- explicitly allocate a local with a ranked Vector/Core array type;
- create `CORE.ZERO` for an arbitrary AIR type;
- validate every later store against the local type;
- preserve the local's type through loop-carried loads and stores.

### G7. Mutable arrays of vectors are not real AIR

The fast kernels consume arrays of pre-rotated vectors. `Blocking_rot` allocates
an array-of-vectors, populates it in a loop, and later performs dynamic indexed
loads.

At plan inception, the `to_air` path could materialize a fixed Python list, but
general `new_array` and `new_ist` were wrapper-only operations.
`AIRValue.__setitem__`
therefore appears to store an element but emits no real AIR `IST` statement.

Required capability:

- typed one-dimensional arrays whose element is a vector array type;
- genuine `LDA`, `ARRAY`, `ILD`, and `IST` builders;
- dynamic index load and store;
- correct statement insertion inside nested loops;
- explicit dimensional and element-type validation.

Relevant implementation:

- `bindings/src/air_builder_bindings.cpp:1518-1651`
- `ace_edsl/edsl/core/air_value.py:850-944`
- `nn-addon/vector/src/tensor2vector_util.cxx:865-944`

### G8. Ranked types, constants, and typed attributes are incomplete

The Conv/Gemm lowering uses shaped signed-integer rotation arrays,
two-dimensional packed weights, expanded bias vectors, masks, and typed
attributes.
At plan inception, the constant builder flattened real arrays to one-dimensional `f32`, and
the generic node attribute API supports only one scalar `u32` value.

Required capability:

- structural equality for ranked AIR types;
- preservation of heterogeneous formal shapes;
- explicitly ranked and typed `s32`, `f32`, and other constant arrays;
- correct element/slice result types;
- scalar and vector attributes with signed and unsigned element types;
- constant content hashing for deterministic comparison.

The default C++ provider initially constructs packed weights and expanded bias.
Owned typed host-payload records formed the portable boundary for the
then-planned Python provider. Typed DSL constants are their
materialized/authoring representation for masks, rotation tables, tests, and
eventual NN-level authoring.

### G9. No provider-neutral planner/emitter boundary

The four named metakernel functions are emitters, not complete operator
lowerings. The handlers surrounding them also perform shape validation,
packing, cost modeling, mask policy, dispatch, operand preparation, and
epilogues. Planning and AIR construction are interleaved, so a second provider
cannot currently compute plans without reproducing C++ AIR ownership details.

The frozen v1 `VECTOR_KERNEL_PLAN` is the provider-neutral semantic core. Its
constant descriptors intentionally record roles, types, and hashes rather than
owning payload bytes or operand-preparation recipes. M4 must preserve that
schema and add a companion prepared package instead of putting AIR identities
or language-specific ownership into the plan.

Required capability:

- an immutable, AIR-independent `VectorKernelPlanningRequest` containing
  portable normalized opcode and attribute records, structural type
  descriptors, immutable option/context and target-capability snapshots, owned
  typed source-constant payloads, and the requested plan kind;
- a synchronous provider-call lifetime: language adapters may expose read-only
  views backed by the request only for the duration of the call, and providers
  must not retain or mutate them;
- a per-`Vector_driver` `PlanProvider` that returns host data only and never AIR
  nodes, type pointers, IDs, symbols, or borrowed buffers;
- an immutable `PreparedVectorKernelPlan` containing the canonical
  `VECTOR_KERNEL_PLAN`, owned typed constant payloads keyed by semantic role,
  runtime operand/preparation descriptors, optional scalar/sharding
  descriptors, and diagnostics-only provider provenance;
- canonical normalization and provider-differential comparison of the entire
  semantic prepared package, including every preparation/scalar descriptor and
  payload but excluding diagnostics-only provenance;
- a validation invariant that every AIR-affecting decision is represented in
  the v1 plan or is uniquely derived from and checked against it; any new
  non-derived AIR semantic requires a versioned plan/key schema before use;
- central C++ copying, validation, hash recomputation, canonicalization, and
  destination-`GLOB_SCOPE` materialization for every provider result;
- native whole-kernel and DSL emitters that consume only the same frozen plan
  package and never re-read mutable configuration or redo planning;
- three independent selectors for provider, kernel implementation, and
  automatic or forced plan kind, plus an explicit diagnosed fallback policy;
- destination-local helper deduplication by full canonical key plus compatible
  signature, independent of provider identity.

The existing `VEC:python_dsl` option selects a legacy Python emission path. It
must not be reused for Python plan computation.

### G10. NN-level Conv/Gemm authoring is incomplete

This is distinct from the Vector metakernel emitter.

Current limitations include:

- `NN.CONV` does not preserve its complete attribute schema or infer the output
  type correctly;
- there is no complete bound `NN.GEMM` builder;
- the high-level `gemm` fallback reaches an unimplemented matrix-multiply
  builder;
- Conv/Gemm constants, heterogeneous operand shapes, and result types are not
  reliably preserved.

Required capability:

- typed NN Conv/Gemm builders;
- normalized Conv and Gemm attribute schemas;
- output-shape and result-type inference;
- validation of the supported lowering subset;
- ranked constant operands.

Relevant implementation:

- `ace_edsl/edsl/core/tensor_ops.py:12-141`
- `bindings/src/air_builder_bindings.cpp:446-503`

### G11. No native-versus-DSL structural oracle

The existing Conv-shaped DSL test proves that loop plumbing can reach C, but it
explicitly uses multiply/add placeholders for roll and slice. It is not a
semantic or structural Conv lowering test.

Required capability:

- normalized AIR comparison that ignores generated IDs, symbol spelling, and
  source positions;
- comparison of opcodes, domains, types, loop topology, statement order,
  constants, attributes, masks, and rotation sets;
- provider differential comparison of canonical plan fields, exact prepared
  payload bytes, hashes, specialization keys, and helper names;
- small deterministic runtime comparisons after code generation.

All builds, experiments, benchmarks, and tests must run inside
`ace-compiler-dev`.

### G12. No independent Python AIR function-inliner and cleanup passes

The helper-function design intentionally leaves a `CORE.CALL` in pre-inline
Vector AIR. Eliminating that call is a concrete AIR transformation built on G2,
not the pass framework itself and not part of Gemm/Conv kernel authoring.

The inliner must operate on selected same-module calls and structurally clone
each helper body into its caller while remapping:

- formals to actual values, with each actual evaluated exactly once;
- local address data and pregs;
- `CORE.DO_LOOP` IV symbols and all scalar/index uses;
- nested block-parent statement links;
- the single returned value to the original call-result preg.

Same-module global types and constants must be reused rather than recreated.
Source positions and node attributes such as `RNUM`, `SLOT`, and `MASK` must be
preserved. The return becomes a `CORE.STP` to the original call-result preg so
the caller's existing `CORE.LDP` remains valid.

Generated-helper eligibility is pass configuration or a per-call structural
predicate, not framework policy and not Gemm/Conv name matching. The initial
supported subset is a leaf, nonrecursive, non-variadic helper with one `RETV`
and no escaping local address. The pass must preflight every selected call
before editing so one malformed candidate rolls back the complete
transformation.

Function reachability is a reusable analysis. Removing unreachable helpers is a
separate `DeadFunctionEliminationPass` with an explicit eligibility predicate,
not an inliner side effect. The initial production sequence is therefore:

```text
FunctionInlinerPass
  -> DeadFunctionEliminationPass
       requires FunctionReachabilityAnalysis
```

`CallGraphAnalysis` records direct calls and structural function references in
stable native-ID order. `FunctionReachabilityAnalysis` starts from every
program entry, externally visible/exported function, address-taken function,
and function referenced by an entry descriptor. It handles shared callees and
recursive strongly connected components deterministically. Unknown or indirect
references conservatively retain every possible target (or fail with a stable
diagnostic if no safe target set can be formed); cleanup must never delete a
possibly referenced function.

Allowing calls to remain un-inlined in the FHE pipeline is a later extension.
It requires plaintext-versus-ciphertext formal classification, scalar-formal
support, call-attribute propagation in Vector-to-SIHE, trustworthy helper
multiplication-depth summaries, and enforcement or extension of current
call-graph restrictions.

G2 and G12 intentionally follow the completed Gemm/Conv DSL and Python
planning-provider phases. Phase C starts with G2 at M13; G12 follows as its
first concrete transformation clients at M14, with production activation at
M15.

Relevant implementation:

- `ace_edsl/edsl/passes/function_inliner.py:1-81`
- `ace_edsl/tests/test_function_inliner.py:208-1166`
- `bindings/src/air_function_inliner.h:1-35`
- `bindings/src/air_function_inliner.cpp:1-630`
- `bindings/src/air_builder_bindings.cpp:6230-6309`

## Gap-to-Milestone Closure Matrix

| Gap | Missing capability | Milestone(s) that first fill it | Final validation milestone |
| --- | --- | --- | --- |
| G1 | Destination-module helper materialization and typed call/LDP bridge | M1 | M4/M7 |
| G2 | Generic Python AIR analysis/transformation framework, safe bindings, transactions, and hooks | M13 | M13/M15 |
| G3 | Lexical domain switching and Vector operand views | M2 | M3/M4 |
| G4 | Typed Core scalar/index arithmetic and `SHL` | M2 | M5 |
| G5 | Genuine Vector roll/slice/add/mul and `RNUM`/`SLOT` | M3 | M5/M6 |
| G6 | Typed locals and typed zero | M2 | M5/M6 |
| G7 | Real mutable arrays of vectors and indexed store/load | M8 | M9/M10 |
| G8 | Structural ranked types, constants, and typed attributes | M2/M3; expanded in M8 | M9/M11 |
| G9 | Provider-neutral planning package, providers, plan variants, and orthogonal selection | M0/M4 | M12 |
| G10 | Complete NN-level Conv/Gemm authoring | M11 | M11 |
| G11 | Structural, provider-differential, and runtime oracle | M0 | Provider: M12; structural: every Phase-A milestone; runtime: M15 |
| G12 | Independent Python inliner and dead-function cleanup transformations | M14 | M15/M16; no-inline FHE mode is follow-on work |

Phase-A kernel ports require G1, G3-G6, and the relevant portion of G8. Neither
G2 nor G12 is a prerequisite for writing or structurally validating the DSL
kernels. Fast-kernel work must not begin before G7 is closed. No DSL-generated
helper call may enter Vector-to-SIHE until the M13 framework and M14 Python pass
sequence are complete and their production integration is validated at M15.

## Implementation Milestones

### M0. Freeze the DSL boundary and native structural oracle

M0 froze provider-neutral immutable plan semantics for baseline Gemm, baseline
Conv, fast Gemm, and fast Conv from the existing C++ implementation. The plans
cover prepared-operand and constant descriptors, result type, loop bounds,
rotation candidates, mask and slot policy, reduction parameters, and optional
sharding offset. Under M4, the extracted C++ algorithms become the default/
reference plan provider, and C++ remains the authoritative validator and
destination materializer.

This refinement supersedes only the provider-ownership statement in
`vector_kernel_dsl_m0_contract.md`. The v1 variants, canonical hashes and keys,
helper ABI, and native structural fixtures remain frozen. M4 adds a companion
owned payload/preparation package without retroactively adding a provider
interface to M0.

Freeze the Phase-A helper ABI: leaf and nonrecursive, one or more packed-vector
inputs, same-module constants, a single ranked-vector return, deterministic
specialization identity, and one final `RETV`.

Specify a normalized pre-inline AIR comparator that checks:

- helper signature and call argument/result types;
- domain, opcode, result type, and shape;
- loop bounds, nesting, IV use, and statement order;
- constant type, shape, and content hash;
- scalar and vector attributes;
- ordered slice and rotation behavior.

It may ignore only generated IDs, source positions, and symbol spelling. Record
the future Python AIR-pass handoff point as `Tensor2Vector/Mv2v -> pre-inline
Vector AIR` -> M13 `AIRPassManager` at `BEFORE_VECTOR2SIHE` -> M14
inliner/cleanup sequence -> `Vector2SIHE`, but do not implement the framework
or passes in Phase A.

Acceptance gate:

- M0's effective behavior corresponds to the later `plan_provider=cpp`,
  `kernel_impl=native`, and `plan_kind=auto` tuple, but M0 exposes no such
  controls; normalized native AIR is unchanged;
- every input used by the four native emitters and caller-side epilogues is
  represented in a plan;
- helper ABI and specialization-key fixtures are documented;
- baseline Gemm has a direct test path even though normal dispatch currently
  selects fast Gemm.

Closes: the specification portion of G9 and the structural foundation of G11.

### M1. Add destination-module helper materialization and the call bridge

Add a Tensor-to-Vector callback that selects a registered DSL lowering and
materializes it as a complete helper function in the current destination
`GLOB_SCOPE`. The helper body must already contain Core+Vector AIR and end in a
single `RETV`; it is not revisited by the source-module iteration performed by
`Lower_to_vector`.

Add the caller-side bridge that allocates a compatible preg, prepends a typed
`CORE.CALL`, and returns a caller-owned `CORE.LDP` as the replacement for the
selected NN expression. Do not transplant a function, loop, or symbol from an
independent DSL `GLOB_SCOPE`.

Acceptance gate:

- one synthetic NN operation materializes one destination-owned helper and one
  call, and is replaced exactly once;
- helper signature, entry, constants, function scope, loop IVs, and body all
  belong to the Tensor-to-Vector destination module;
- call target and result preg pass AIR type and scope verification;
- pre-inline AIR contains the expected `CALL + LDP` bridge and complete helper
  `RETV`;
- disabled DSL lowering leaves native behavior unchanged.

M4 feeds this generic registry and materializer only a centrally validated
`PreparedVectorKernelPlan`; recipe selection is by canonical plan kind. Under
M4, with no provider or DSL registry override attached, the default is
`plan_provider=cpp`, `kernel_impl=native`, and `plan_kind=auto`.

Closes: G1. Begins G9's invocation seam.

### M2. Repair DSL domains, types, Core scalars, locals, and loop values

Implement:

- a lexical domain stack and target-domain views of incoming operands;
- domain preservation across loop and conditional results;
- structural equality for ranked types and heterogeneous parameter types;
- typed Core scalar values and type-driven dispatch;
- signed 32-bit loop/index arithmetic and explicit casts;
- Core `ADD`, `MUL`, `LT`, and `SHL`;
- explicitly typed locals and type-checked stores;
- typed `CORE.ZERO`.

Retain the existing unit-step `range_dynamic` loop structure. Arbitrary step and
unroll controls are not required by the native metakernels.

Acceptance gate:

- an NN-origin operand viewed inside `@vector_kernel` emits no NN-domain
  arithmetic;
- `cin * kernel_hw + khw` and `1 << iv` are typed Core arithmetic;
- loop-carried Vector values retain their domain and ranked type;
- typed Vector accumulators can be initialized with a real typed zero;
- a destination-owned nested-loop helper and its call verify as pre-inline
  Vector AIR.

M2 API work may develop in parallel with M1, but M2 closes only after this M1
helper/call integration fixture passes.

Closes: G3, G4, G6, and the foundational type portion of G8.

### M3. Expose genuine Vector primitives and attributes

Bind and wrap the native Vector generators for:

- add and multiply with the native result-type rule;
- dynamic roll with signed `RNUM` candidates;
- dynamic slice with a precise result type;
- required reshape or masking primitives when the planner does not already
  provide a prepared operand;
- typed `SLOT` and other scalar/vector attributes.

Acceptance gate:

- roll accepts a Core scalar shift and records all expected signed candidates;
- slice accepts a Core scalar start and returns the planned Vector shape;
- add/multiply result types match `VECTOR_GEN`;
- generated nodes are genuine AIR nodes, not wrapper placeholders;
- a helper using the primitives verifies after `Mv2v_opt` and before
  Vector-to-SIHE.

M3 primitive work may develop in parallel, but M3 closes only after integration
with the M1 helper materializer and M2 type/domain substrate.

Closes: G5 and the attribute portion of G8.

### M4. Extract the provider-neutral planning contract and connect the C++ provider

Refactor each Conv/Gemm handler into four explicit stages:

1. build an immutable, AIR-independent `VectorKernelPlanningRequest`;
2. invoke a per-`Vector_driver` `PlanProvider`;
3. centrally validate and canonicalize an immutable
   `PreparedVectorKernelPlan` in C++;
4. materialize destination-owned operands/constants and invoke the selected
   kernel implementation.

The request owns portable normalized opcode and attribute records, structural
input/result type descriptors, immutable option/context and target-capability
snapshots, typed source-constant payloads, and the requested plan kind. A
language adapter may expose callback-scoped read-only views of request payloads;
the provider may neither mutate nor retain them. The prepared package owns the
canonical v1 `VECTOR_KERNEL_PLAN`, typed constant payloads keyed by semantic
role, runtime operand/preparation descriptors, and optional scalar/sharding
descriptors. Provider provenance is retained for diagnostics only. Neither
record contains AIR nodes, type pointers, IDs, symbols, or borrowed
provider/Python buffers.

C++ copies every returned payload, recomputes its canonical content hash,
validates schema, operation/kind compatibility, types, shapes, ranges, checked
products, role completeness, every preparation/scalar descriptor, and
cross-field invariants. It verifies that every AIR-affecting decision is
represented by v1 or is its unique canonical derivation; a new semantic requires
a versioned plan and key schema. C++ then creates every AIR object in the active
destination `GLOB_SCOPE`. Validation must complete before helper, call,
constant, or native-emission mutation. Both native whole-kernel and DSL emitters
consume only the same frozen validated package; neither may re-read mutable
configuration, redo planning, or change package ownership.

The production provider in M4 wraps the extracted C++ algorithms and remains
the default/reference provider. M4 also supplies an injected fake provider for
seam and negative testing. A Python provider may be registered through the same
per-pass interface, but its production algorithms are deliberately deferred to
M12.

Freeze and fixture the provider-neutral reference algorithms while extracting
them: exact integer formulas, rounding and checked-overflow behavior, traversal
and tie-break order, dtype and byte-order rules, constant packing, automatic
dispatch, and forced-kind semantics. M12 must implement this specified contract
rather than reverse-engineer platform-dependent C++ behavior.

Add three orthogonal per-pass selectors and one fallback policy:

- `plan_provider=cpp|python`;
- `kernel_impl=native|dsl`;
- `plan_kind=auto|baseline-gemm|baseline-conv|fast-gemm|fast-conv`;
- `fallback=error|cpp-native`.

Throughout this plan, `cpp + native + auto` abbreviates
`plan_provider=cpp`, `kernel_impl=native`, and `plan_kind=auto`; fallback is
stated separately and defaults to `error`.

The defaults are `cpp`, `native`, `auto`, and `error`. `cpp-native` applies only
when the requested provider is unavailable or a valid plan has no supported DSL
recipe. Before destination mutation, it restarts with the C++ provider and
native implementation while preserving the requested `plan_kind`, and emits a
diagnostic. Provider exceptions and malformed or invalid provider results
always fail even when fallback is enabled, and any failure after AIR emission
starts is fatal; planner or emitter bugs cannot be hidden. Do not reuse the
legacy `VEC:python_dsl` emission flag for plan-provider selection.

The DSL registry selects a recipe by canonical plan kind. Cache helpers in the
active destination `GLOB_SCOPE` by the complete specialization key plus helper
signature. Provider, kernel implementation, and provenance do not enter the
key. The validator must prove that every helper-body-affecting package field is
represented in the v1 key or uniquely derived from it; otherwise the schema and
key version must change before materialization. Equal validated plans from C++
and another provider must then share one helper.

Acceptance gate:

- C++-provider/native-emitter AIR is normalized-equal to pre-refactor native
  AIR for all four plan variants;
- the reference formulas, rounding/overflow rules, traversal/tie order,
  packing, byte order, automatic dispatch, and forced-kind behavior are
  specification-backed and frozen by fixtures;
- identical requests produce deterministic normalized values for every prepared
  package descriptor, owned payload byte, hash, specialization key, and helper
  name;
- request and prepared-package records contain no AIR ownership; request views
  are read-only and callback-scoped, and mutating or releasing provider-owned
  output buffers after return cannot change the package;
- central validation rejects missing/extra roles, wrong types or shapes, bad
  hashes, invalid operation/kind pairs, overflow, inconsistent preparation or
  topology, and unsupported plans before destination mutation;
- every forced `plan_kind` reaches its valid C++ plan variant independently of
  automatic dispatch, while invalid forced pairs fail deterministically;
- an injected fake provider feeds both the native emitter and the M1 synthetic
  DSL materialization path through the same validation/materialization seam;
- selecting `python` without a registered provider under `fallback=error`, and
  every provider exception or invalid result under either fallback mode, fails
  clearly before mutation; explicit fallback for a missing provider or
  unsupported DSL recipe is tested separately and emits a diagnostic;
- changing only provider, kernel implementation, or provenance does not change
  specialization identity;
- repeated equal plans in one destination create one helper and multiple calls,
  different plans create different helpers, and a key/signature mismatch fails;
- selected DSL emission replaces exactly one NN operator, and stopping after
  the Vector phase leaves no skipped selected operator;
- the default `cpp + native + auto` path remains unchanged.

Closes the provider-neutral seam and C++ portion of G9 and validates G1. Full
provider parity remains open until M12.

### M5. Port baseline Gemm as a DSL helper

Port `New_gemm_metakernel` first. Both `cpp + native` and `cpp + dsl` consume the
same frozen `PreparedVectorKernelPlan`. The DSL recipe receives it as a
compile-time host input whose decisions and constants it emits; the object is
never an AIR formal. The recipe must not recompute dimensions, packing, or
policy. Baseline Gemm exercises one primary loop, dynamic roll, dynamic slice,
accumulation, optional `SHL` reduction, bias, and mask without the fast
array-of-vector structure.

Acceptance cases:

- a simple case without block reduction or mask;
- a case with `width / height > 1`, block reduction, and mask.

Phase-A acceptance gate:

- `plan_provider=cpp`, `kernel_impl=dsl`, and
  `plan_kind=baseline-gemm` produce a typed call to the specialized helper from
  the same validated package used by the native oracle;
- native and DSL helper AIR have the same loop bounds and nesting;
- slice widths, ordered shifts, `RNUM`, add/multiply counts, bias, and mask
  placement match;
- result type matches, and both paths preserve `ABSENT_NATIVE_BASELINE` by
  emitting no `SLOT`; changing that policy requires a separate intentional
  native-oracle and plan amendment;
- helper and caller verify after `Mv2v_opt`;
- the test intentionally stops before Vector-to-SIHE.

Validates: G1 and G3-G6, G8, G9, and structural G11 for the first real kernel.

### M6. Port baseline Conv as a DSL helper

Port `New_conv_metakernel` using the same substrate. The selected provider
supplies flattened input, im2col weight, expanded bias, rotation data,
dimensions, and stride through the validated prepared package; Phase A uses the
default C++ provider. The DSL recipe consumes all of those fields and payloads
without recomputing them or re-reading configuration.

Acceptance cases:

- a nontrivial kernel and rotation list;
- a shape that selects the baseline Conv path, including
  `channel_out < channel_in` where appropriate.

Phase-A acceptance gate:

- the pre-inline module contains a typed call to the specialized Conv helper;
- the helper preserves the two-level `channel_in x kernel_hw` Core loops;
- `cin * kernel_hw + khw`, `ra[khw]`, weight slices, channel rolls,
  accumulation, and bias match native AIR;
- signed `RNUM`, prepared constant hashes, output type, and `SLOT` match;
- helper and caller verify after `Mv2v_opt`;
- the test intentionally stops before Vector-to-SIHE.

Validates: nested-loop preservation and G1/G3-G6/G8/G9/structural-G11.

### M7. Baseline DSL structural gate

Complete the current baseline-kernel stage without requiring the Python
inliner.

Acceptance gate:

- `plan_provider`, `kernel_impl`, and `plan_kind` are independently selectable,
  including forced baseline Gemm and Conv kinds;
- a missing provider or valid plan with no DSL recipe has tested
  `fallback=error` and explicit diagnosed `fallback=cpp-native` behavior;
  invalid packages, provider exceptions, and post-mutation failures never
  fall back;
- baseline Gemm and Conv produce deterministic, provider-independent helper
  names and plans;
- normalized pre-inline helper bodies and caller bridges match their native
  structural oracles;
- no placeholder or synthetic-only AIR node is present;
- the pipeline can intentionally stop at valid pre-inline Vector AIR;
- no claim is made that generated helper calls can yet enter Vector-to-SIHE.

Final Phase-A baseline validation: G1, G3-G6, G8, G9, and structural G11.

### M8. Add fast-kernel DSL memory and helper substrate

Implement genuine typed mutable arrays of vectors and reusable DSL helpers.
Their constant and operand inputs come from semantic roles in the validated
prepared package rather than from C++-planner-only utilities:

- typed array allocation;
- `LDA`, `ARRAY`, `ILD`, and `IST`;
- dynamic indexed load/store inside loops;
- `blocking_rot`;
- intra-block reduction;
- mask/valid-data clearing;
- cyclic roll;
- collective reduction;
- ranked mask and rotation constants.

Acceptance gate:

- one loop populates an `array<vector>` and another dynamically consumes it;
- all indexed stores are real AIR statements in the correct block;
- element and array types verify;
- helpers produce the expected loop, rotation, and mask topology;
- the smoke helper verifies as pre-inline Vector AIR after `Mv2v_opt`.

Closes: G7 and the fast-kernel portion of G8. Begins fast ownership work in G9.

### M9. Port complete fast Gemm as a DSL helper

Port `New_gemm_metakernel_fast` plus input blocking, caller-side `ps`/`kp`
reductions, bias, mask, and `SLOT` so the helper returns the complete Gemm
result. Move every AIR-affecting epilogue decision behind the same frozen-plan
consumer boundary; the DSL recipe must not call planner utilities.

Acceptance cases:

- `ps = 1` without an extra reduction;
- `ps > 1`;
- `kp > np`;
- mask enabled and disabled where supported.

Phase-A acceptance gate:

- the pre-inline module contains the expected specialized helper and call;
- `gs x bs` loops, block zeroing, dynamic input-array load, packed-weight
  slice, roll, reductions, bias, mask, result type, and `SLOT` match native AIR;
- IRMA constant hashes and `RNUM` sets match;
- helper and caller verify after `Mv2v_opt`;
- the test intentionally stops before Vector-to-SIHE and requires no inliner.

Validates: G1, G3-G9, and structural G11 for fast Gemm.

### M10. Port complete fast Conv and close the Vector-kernel subphase

Port `New_conv_metakernel_fast` last. It is the superset of the required DSL
features: grid/block loops, dynamic input-array access, cyclic masks,
collective reductions, group-dependent behavior, and optional sharding offset.

Acceptance cases:

- ordinary grid/block lowering;
- collective reduction with mask;
- cyclic-roll branch;
- sharded weight offset;
- supported group/depthwise behavior relevant to the cyclic condition.

Phase-A acceptance gate:

- the complete frozen package is consumed without configuration rereads, and
  plan fields and prepared constant hashes match;
- `num_grid x cap_block` loops and statement order match;
- zero placement, slice/index expressions, array loads, roll candidates, cyclic
  masks, collective reduction, bias, output type, and `SLOT` match;
- all four DSL helpers and their callers verify after `Mv2v_opt`;
- all four have normalized pre-inline structural parity with the native oracle;
- fast Conv tests stop before Vector-to-SIHE;
- the Vector-kernel subphase ends without implementing or invoking the Python
  inliner.

Closes the Vector-kernel implementation subphase and validates G1, G3-G8, the
C++-provider portion of G9, and structural G11 for all four kernels. Python
provider parity remains deferred to M12. Phase A continues through NN-level DSL
authoring in M11.

### M11. Complete NN-level Gemm/Conv DSL authoring and close Phase A

Complete the direct NN frontend surface while remaining in the pre-inline
Vector-AIR workflow:

- typed `NN.CONV` and `NN.GEMM` builders;
- normalized Conv attributes: strides, pads, dilation, kernel shape, and group;
- normalized Gemm attributes: alpha, beta, transA, and transB;
- result-shape/type inference and supported-subset validation;
- ranked typed weight and bias constants;
- heterogeneous operand and result types.

Run a compact shape/attribute matrix through NN authoring, the default C++ plan
provider, central validation, and DSL helper materialization, stopping after the
Vector phase.

Phase-A acceptance gate:

- an `@nn_kernel` authors valid Conv and Gemm nodes with complete schemas;
- the matrix exercises `cpp + native` and `cpp + dsl` against the same canonical
  plans, with `cpp + native + auto` remaining the default;
- selected DSL lowering removes every supported NN Conv/Gemm node and produces
  the expected destination-owned helper and call;
- all four kernel variants have normalized pre-inline structural coverage;
- helper signatures, calls, loops, constants, attributes, and result types
  verify after `Mv2v_opt`;
- all current `ace_edsl` DSL tests relevant to NN and Vector authoring pass;
- no test in this milestone requires the Python function inliner or
  Vector-to-SIHE.

Closes: G10 and the Gemm/Conv DSL phase. Provides the final Phase-A
structural gate for G11.

### M12. Implement the Python plan provider and differential planning validation

Begin Phase B only after M11 closes Phase A and all four kernel
implementations supplied by the C++ provider are structurally stable.

Implement an ordinary, separately testable Python planning module and the
per-`Vector_driver` registration/binding that adapts it to M4's `PlanProvider`.
It consumes `VectorKernelPlanningRequest` data and returns only the data needed
to construct a `PreparedVectorKernelPlan`; it never creates, receives, or
transforms AIR objects and does not execute inside `@vector_kernel`.

Implement all four plan variants with specified exact integer arithmetic,
tie-breaking, dtype/shape rules, byte order, and constant-packing behavior.
C++ copies all returned buffers, recomputes canonical hashes, validates and
freezes the package, and remains the production default/reference provider and
authoritative validator/materializer.

Acceptance gate:

- standalone Python tests cover all four variants, boundary and invalid shapes,
  overflow, cost-model ties, mask decisions, sharding, and deterministic
  packing;
- C++ and Python providers produce equal normalized semantic packages, excluding
  diagnostics-only provenance but including canonical plan fields, every
  preparation/scalar descriptor, exact payload bytes and hashes, specialization
  keys, and helper names across the fixture and NN shape/attribute matrices;
- `plan_kind=auto` and every valid forced kind agree across providers, while
  invalid operation/kind pairs fail deterministically;
- `cpp + native`, `cpp + dsl`, `python + native`, and `python + dsl` all consume
  the common validated package for every supported variant and verify after
  `Mv2v_opt` at the pre-inline boundary;
- Python exceptions, missing or extra roles, unsupported dtypes, noncontiguous
  or wrong-endian payloads, bad shapes or hashes, inconsistent preparation
  descriptors, and invalid plans retain actionable diagnostics and cause no
  partial destination mutation;
- Python-owned arrays may be mutated or released after provider return without
  changing the frozen package;
- provider registration is scoped to one driver invocation, repeated and
  shuffled test runs are order-independent, and multiple hash-seed runs are
  deterministic;
- provider identity and provenance never alter helper specialization or
  destination-local deduplication;
- `cpp + native + auto` remains the production default and any fallback remains
  explicit and diagnosed.

Closes the remaining provider-parity portion of G9 and the provider-differential
portion of G11.

### M13. Establish the generic Python AIR analysis/transformation pass framework

Begin Phase C only after M12 closes the Python planning-provider phase.

Create reusable, domain-neutral Python AIR pass infrastructure under
`ace_edsl.edsl.passes`. The framework operates on one live AIR `GLOB_SCOPE`
per invocation and distinguishes read-only analyses from transactional
transformations. Gemm, Conv, Vector-kernel helper selection, and function
inlining must not appear in the framework itself.

Use a clean ownership-oriented namespace:

```text
ace_edsl/edsl/passes/
  framework/
    api.py
    diagnostics.py
    analysis_manager.py
    manager.py
    pipeline_hooks.py
  analyses/
    call_graph.py
    function_reachability.py
  transition/
    tentative_native_inliner.py  # M13 adapter; package removed in M16
  transforms/
    function_inliner.py             # added in M14
    dead_function_elimination.py    # added in M14
```

The package initializers remain side-effect-free. Pass instances are immutable
configuration and do not register themselves globally or retain AIR objects
between invocations. M13 removes the ambiguous root-level
`passes.function_inliner.FunctionInlinerPass` facade: the temporary binding-side
delegate is renamed `TentativeNativeInlinerAdapter`, and the canonical
`FunctionInlinerPass` name is reserved for the independent M14 implementation.

Freeze the public framework contracts before adding mutation bindings:

```text
IRUnitRef(kind="module" | "function", module_id, native_id?)
UnitSelector = CURRENT_MODULE | CURRENT_PASS_UNIT
AnalysisRequest(analysis, unit: UnitSelector | IRUnitRef)
AnalysisPass[T].run(read_only_unit, context) -> AnalysisResult[T]
TransformationPass.required_analyses(current_unit) -> tuple[AnalysisRequest]
TransformationPass.run(edit_unit, context) -> TransformResult
AIRPassManager.run(glob_scope, steps, context) -> PassManagerResult
```

`IRUnitRef` is an immutable host identifier, not an AIR wrapper. M13 supports
explicit module and function analysis requests; block-level scheduling and
automatic per-function pass expansion are follow-on work. Pipeline/pass
configuration normally stores relative `UnitSelector`s because runtime AIR IDs
do not exist yet; the manager resolves them per invocation. Explicit
`IRUnitRef`s are allowed only in an already active invocation. An analysis
declares dependency requests relative to its unit, and a transformation's pure
`required_analyses(current_unit)` method declares its complete requests without
mutating pass configuration.

`AnalysisResult[T]` reports success/failure, an optional immutable host-data
value, diagnostics, and metrics; failure carries no value, and it has no
`changed` field. `TransformResult` reports success/failure, structured
diagnostics, pass-specific metrics, and preserved analyses, but does not claim
whether AIR changed. The manager derives `PassExecutionResult.changed` and its
before/after revisions from transaction state rather than trusting pass code.
`PassManagerResult` retains one ordered execution record per attempted
requested step, an ordered trace of dependency-analysis executions, the IDs of
requested steps left unexecuted by fail-fast, and an aggregate outcome. It does
not invent execution records for passes that never ran. Counts such as
`calls_inlined` and `functions_removed` belong to their concrete execution
records, not any common framework field.

The caller-ordered manager list may contain explicit `AnalysisRequest` values
and transformations. An explicit analysis executes read-only at its declared
position and unit, has `changed=False`, and publishes or reuses its cached
result; an analysis run solely to satisfy a dependency appears in the
dependency trace, not as an invented top-level pass. The manager never reorders
requested steps.

Add immutable structured diagnostics with severity, stable code, message,
optional AIR location, and notes. Passes do not print. The pipeline renders
diagnostics and owns timing, IR-dump, and begin/end instrumentation. Expected
pass failures and unexpected Python/C++ exceptions are attributed to the
current pass and produce deterministic fail-fast results. Wall-clock timings
are instrumentation only and are excluded from deterministic result equality.
Instrumentation is observational: callback and dump failures are caught,
reported with stable instrumentation diagnostics, and never change pass status
or transaction outcome. Begin/end callbacks are attempted in a `finally` pair,
including when either callback or the pass raises.

The invocation-scoped `AnalysisManager`:

- keys cached results by stable analysis identity/configuration, `IRUnitRef`,
  and module revision;
- resolves analysis dependencies on demand while recording their dependency
  graph and rejecting cycles deterministically;
- accepts only immutable Python host data or revision-bound stable descriptors,
  never live native pointers;
- invalidates module and descendant function analyses after a changed module
  transformation, including dependent analyses transitively;
- preserves all analyses for a verified no-op and only explicitly named
  `PreservedAnalyses` after a commit;
- treats every analysis as revision-bound by default. An analysis may declare
  `cross_revision_safe` only when its immutable result is free of AIR wrappers
  and IR identifiers and can remain semantically valid for a changed revision
  of the same module. A transformation must explicitly promise that validity.
  A successfully preserved safe result is re-keyed to the new revision and its
  old key is removed; no result is transferred across modules, and
  revision-bound descriptors/IDs are not rebound in M13;
- treats an attempt to preserve a non-safe result as a pass-contract error
  before commit, rolls back that transformation, and reports the offending
  analysis deterministically;
- requires preservation to be dependency-closed: every dependency of a
  preserved result must itself be safe and preserved. A mixed
  preserved/unpreserved dependency chain is a pre-commit contract error and
  rolls back the transformation;
- retains the cache checkpoint taken after required analyses are resolved and
  immediately before the candidate transaction opens. Thus valid analyses
  computed for the committed module remain cached even if that transformation
  later rolls back;
- never shares cached results across modules, manager invocations, or native
  opaque lowering barriers.

Each `TransformationPass` declares `required_analyses`. The manager resolves
that complete set, including analysis-to-analysis dependencies, against the
committed read-only module before it opens the pass's candidate transaction.
The pass receives frozen results expressed in stable host IDs and uses the
transaction's total source-to-candidate ID map to reach candidate views. The
map covers every ID kind an M13 analysis may return: module/function/entry,
block/statement/node, formal/local/symbol/preg/IV, type, and constant. Global
identities reused unchanged still receive explicit identity mappings, and an
unmapped or wrong-kind ID fails deterministically. Once editing begins,
requesting an undeclared or newly computed analysis fails deterministically;
M13 has no analysis cache over dirty candidate AIR.

Add a neutral module transaction owned by the binding:

```text
committed GLOB_SCOPE
  -> exclusively owned candidate clone
  -> Python pass reads/edits candidate
  -> native AIR verification
  -> atomic commit and revision increment | complete rollback
```

Read-only analysis views expose no edit operations. Mutable function, block,
statement, node, symbol, preg, IV, and type views exist only inside one active
transaction and carry its module/generation token. Commit or rollback
invalidates every candidate view; a later use raises a deterministic stale-view
error. A successful commit keeps only the stable public module wrapper usable,
increments its monotonic revision exactly once, and makes every pre-commit
child view stale. There is no implicit rebinding. A pass cannot edit AIR and
claim a no-op: transaction dirtiness is authoritative.

Atomicity is per transformation, not per manager invocation. If pass A commits
and pass B later fails, A remains committed, B alone rolls back, the manager
stops before pass C, and the final revision/cache describe A's committed AIR.
The failed manager result still prevents the pipeline from entering the next
lowering phase.

The public `GlobScope` wrapper explicitly owns the current native scope. Commit
atomically replaces that scope and refreshes every binding-owned table or
alias, including `types`, `functions`, `lower_ctx`, the active-binding global,
and wrapper caches. Old child wrappers are invalidated before the retired scope
is released; old types cannot remain readable as foreign-scope objects, and
repeated commits must not retain retired scopes.

Move generic clone/swap support needed by transactions into neutral AIR binding
ownership rather than delegating through the tentative inliner. Expose
deterministic, checked structural operations for:

- functions in native-ID order; entries, formals, locals, and pregs in
  native-ID order; blocks and statements in lexical order; and children by
  operand index;
- opcode, result type, source position, attributes, call target and arguments,
  result preg, return value, symbols, loop IVs, and block parents;
- destination-owned local/preg creation and cloning with explicit
  formal/local/preg/IV maps;
- insert-before/after, replace, erase, nested-block parent repair, checked
  function removal, verification, commit, and rollback.

Python owns analysis and transformation algorithms. C++ owns memory safety,
scope/type checks, low-level cloning mechanics, parent-link repair, verifier
execution, and atomic scope replacement. Do not build this framework on the
placeholder `bindings/src/passmanager_bindings.cpp`; sequence Python passes over
the real `air_builder.GlobScope`.

Add one generic `PipelinePoint.BEFORE_VECTOR2SIHE` extension point. It runs
after Tensor-to-Vector and its `Mv2v_opt` work only when execution proceeds to
Vector-to-SIHE. A pipeline stopping at `TENSOR2VECTOR` remains a pre-inline
inspection target, while a pipeline starting from `nn::vector` runs the hook
before Vector-to-SIHE.

Freeze one immutable, invocation-scoped registration/result surface:

```text
PassPipelineConfig[PipelinePoint] -> tuple[AnalysisRequest | TransformationPass]
HookExecutionResult -> ordered manager-segment and native-barrier records
PipelineResult.pass_results[PipelinePoint] -> HookExecutionResult
CompileResult.pass_results[PipelinePoint] -> HookExecutionResult
```

`AcePipeline` and `Pipeline` accept the same optional `PassPipelineConfig`;
`ace_compile` receives it through `CompilerOptions`. Configuration is copied
per invocation, duplicate pass-instance IDs are rejected before execution, and
there is no global registry. All three runners call one shared hook runner,
return the same result schema, and `ace_compile` no longer swallows failures.
The default Python pass tuple is empty and preserves existing native behavior.

Until M15, the hook runner appends `TentativeNativeInlinerAdapter` after the
caller's Python manager segment when the existing DSL configuration requires
it, then ends the hook before Vector-to-SIHE. The adapter is a transition-only
`NativeBarrierStep`, not a `TransformationPass` and never a manager record. It
has its own ordered barrier record with success, changed, revisions,
diagnostics, and metrics. A barrier attempt always closes the manager segment,
advances the wrapper generation, and invalidates child views and analysis
caches; module revision advances only when the adapter reports a verified AIR
change. Adapter failure is diagnosed, leaves committed AIR/revision unchanged,
and stops the pipeline. Migration of unrelated CKKS or legacy callbacks is
follow-on work.

Implement M13 in this order:

1. freeze pass identity, results, diagnostics, revision, invalidation,
   transaction, and stale-view contracts with pure-Python contract tests;
2. implement the manager, analysis dependencies/cache, preservation, aggregate
   results, deterministic sequencing, and fail-fast behavior with test doubles;
3. add deterministic read-only AIR views and the transaction/edit binding;
4. implement generic `CallGraphAnalysis` and
   `FunctionReachabilityAnalysis` as the first real analysis clients;
5. add the shared pipeline hook, instrumentation, explicitly named transitional
   native adapter, and maintained integration tests for all three runners.

Keep responsibility visible in the test layout:

```text
test_air_pass_framework.py    # pure manager/result/analysis contracts
test_air_pass_bindings.py     # traversal, transactions, ownership, stale views
test_air_pass_pipeline.py     # hook ordering and all three public runners
test_tentative_native_inliner.py # M13 adapter/barrier tests; removed in M16
test_function_inliner.py      # independent Python algorithm, created in M14
test_dead_function_cleanup.py # reachability/cleanup composition, added in M14
```

M13 moves the existing facade/oracle tests under the explicitly tentative test
name and keeps only adapter, barrier, and existing regression coverage there.
M14 creates the permanent independent-inliner suite. Expansion of the full
provider/kernel downstream matrix remains exclusively M15 work.

Crash-prone malformed native AIR cases remain subprocess-isolated. Framework
tests use small hand-built AIR fixtures instead of ONNX wherever full compiler
integration is not the behavior under test.

Acceptance gate:

- two distinct analyses exercise dependency resolution, one-computation cache
  reuse, transitive invalidation, safe-result re-keying, dependency-closed
  preservation, rollback of a mixed preserved/unpreserved chain, and
  deterministic cycle diagnostics;
- result-schema tests prove immutability; analysis failure has no value;
  analyses cannot report `changed`; transforms cannot supply `changed` or
  revisions; manager records derive them; explicit-analysis cache hits remain
  top-level records while dependency-only executions remain in their ordered
  trace; fail-fast records only attempted steps and names every unexecuted step;
  and captured stdout/stderr stays empty;
- cache keys isolate analysis configurations, modules, revisions, manager
  invocations, pipeline instances, and native barriers. The same analysis on
  two function `IRUnitRef`s in one revision produces distinct entries, and a
  changed module invalidates both function entries and their dependents; no pass
  list, context, result, or cache leaks between compilations;
- requesting an undeclared analysis after a transformation transaction opens
  fails before mutation, while declared frozen analysis IDs remap explicitly to
  the candidate;
- relative module/pass-unit selectors resolve independently in two invocations,
  while an `IRUnitRef` from one invocation is rejected by another;
- call-graph and reachability fixtures cover multiple program roots, transitive
  and shared callees, unreachable functions, recursive SCCs, stable ordering,
  exported/address-taken/entry-descriptor roots, and conservative indirect
  references;
- an analysis cannot acquire edit operations; an attempted mutation fails
  without changing AIR, revision, or cached results;
- a no-op transformation leaves AIR, native pointer, revision, child views, and
  cached analyses unchanged;
- a small domain-neutral Python transformation traverses and edits structural
  AIR, commits verified AIR once, advances the revision once, and invalidates
  only the analyses required by contract;
- the binding capability matrix round-trips every attribute's byte payload,
  element type, and count plus source positions; proves formal/local/preg/IV
  remaps and the total source-to-candidate ID map; checks native-ID/lexical/
  operand-index traversal order for functions, entries, formals, locals, pregs,
  blocks, statements, and children; exercises call target/arguments/result-preg,
  return, symbol, IV, and parent views; covers insert-before, insert-after,
  replace, and erase; and proves nested parent repair, same-module global
  type/constant reuse, foreign-scope rejection, and complete function
  scope/entry/symbol removal;
- missing analyses, duplicate pass-instance IDs, invalid result types, pass or
  analysis exceptions, native exceptions, and verifier failures produce
  pass-attributed deterministic diagnostics; later passes do not run;
- every failed or exceptional transformation leaves AIR exactly as it stood
  before that pass, with its native pointer and revision unchanged. Its cache
  returns to the post-required-analysis/pre-candidate checkpoint, and candidate
  views cannot escape rollback;
- a three-pass fixture proves per-transformation atomicity: A commits, B fails
  and rolls back only B, C does not run, and the final AIR/revision/cache retain
  A exactly once;
- every pre-commit child view and every committed or rolled-back candidate view
  fails deterministically as stale; only the public `GlobScope` wrapper remains
  usable after commit;
- repeated commits refresh all binding aliases/caches, reject access through
  retired-scope types and children, and release rather than retain old scopes;
- adapter-specific changed, no-op, and failure tests prove manager closure,
  barrier-record ordering, generation/view/cache invalidation, conditional
  revision advance, failure propagation, and the absence of any manager
  `TransformationPass` record for the tentative adapter;
- multiple passes execute strictly in caller-declared order; aggregate
  `changed`, diagnostics, and metrics preserve per-pass identity and order;
- begin/end callbacks are balanced and ordered on success and every failure
  path, including callback exceptions; callback/dump failures are non-mutating
  instrumentation diagnostics, per-pass IR dumps and execution records are
  stable, and timings are excluded from equality checks;
- pass order, analysis results, normalized AIR, metrics, and diagnostics are
  stable across shuffled tests and multiple fixed `PYTHONHASHSEED` values;
- the empty hook is behavior-preserving, and a test pass runs exactly once at
  `BEFORE_VECTOR2SIHE` through `AcePipeline`, `Pipeline`, and `ace_compile`;
- all three runners accept isolated `PassPipelineConfig` instances, reject
  duplicate IDs identically, and expose equal ordered `HookExecutionResult`
  data through `PipelineResult` or `CompileResult`;
- stopping at Tensor-to-Vector does not run the hook, starting at
  `nn::vector` does, and a hook failure is diagnosed and prevents
  Vector-to-SIHE in every runner;
- framework modules contain no Gemm-, Conv-, Vector-, generated-helper-, or
  inliner-specific policy;
- all framework, binding, ownership, cache, determinism, and pipeline-hook tests
  are included in the maintained quick gate;
- the explicitly named tentative adapter preserves current pre-inline matrices
  and `cpp + native + auto` behavior without exporting a second
  `FunctionInlinerPass` class.

M13 does not implement the Python inliner algorithm, automatically schedule or
parallelize passes, persist analysis caches, add plugin discovery, replace the
native C++ pass manager, permit arbitrary raw-pointer or cross-module mutation,
make generated calls optional downstream, run the full provider/kernel
downstream matrix, or remove the tentative inliner.

Closes: G2. Provides the neutral substrate required by G12.

### M14. Implement the independent Python AIR inliner as a framework client

Begin only after M13 freezes the framework, binding, transaction, analysis, and
pipeline-point contracts.

Implement `FunctionInlinerPass` and `DeadFunctionEliminationPass` under
`ace_edsl.edsl.passes.transforms`. Both implement the M13
`TransformationPass` contract and use only M13 read-only views, edit
transactions, analyses, diagnostics, and generic results. Neither pass may call
`GlobScope.inline_generated_vector_kernel_helpers()`, embed a binding-side
inliner algorithm, import `TentativeNativeInlinerAdapter`, parse AIR dumps, or
use cross-`GLOB_SCOPE` replacement.

The intended direct composition is:

```text
AIRPassManager(
  FunctionInlinerPass(policy="always", predicate=...),
  DeadFunctionEliminationPass(predicate=...),
)
```

The inliner discovers candidate calls in deterministic structural order and
invokes its immutable structural predicate once per candidate. It preflights
the complete selected set before editing. For each selected same-module leaf
helper it:

- materializes every nontrivial actual exactly once;
- maps formals to those materialized actuals;
- allocates distinct caller-owned locals, pregs, and loop IVs per call site;
- clones statements and nested blocks with explicit maps and repaired parents;
- reuses same-module global types and constants;
- preserves source positions and complete attribute payloads, including
  `RNUM`, `SLOT`, and `MASK`;
- replaces the single `RETV` with `CORE.STP` to the original call-result
  preg, preserving the caller's existing `CORE.LDP`;
- removes only the selected call statement.

The initial supported subset remains tagged, same-module, nonrecursive,
non-variadic leaf helpers with one `RETV` and no escaping local address.
Unsupported control flow, ownership, ABI, result, recursion, or return shapes
fail the complete inliner transaction before any selected call is changed.

The inliner reports `calls_inlined` only in its pass-specific metrics and
preserves no call-graph or reachability analysis after a change.
`DeadFunctionEliminationPass` declares the M13
`FunctionReachabilityAnalysis` as required, so the manager recomputes it on the
committed post-inline module before opening cleanup's transaction. Cleanup
removes only unreachable functions accepted by its own eligibility predicate
and reports `functions_removed` separately. Helpers still reachable through an
unselected call, program entries, and unrelated dead functions remain.

M14 owns direct pass and manager validation plus differential comparison with
the tentative native oracle. It does not yet make the Python passes the
production pipeline default; that activation and downstream matrix belong to
M15.

Split validation into a permanent Python correctness suite and a transition-only
live differential suite. The permanent suite uses hand-built AIR and frozen
normalized expected fixtures and survives M16. The live suite invokes the
tentative adapter only on the capability overlap it actually supports and is
removed with that adapter in M16.

Acceptance gate:

- an implementation-independence test runs against a structural-only spy that
  exposes exactly the approved M13 traversal/edit primitives and no coarse
  inliner operation; a static import-boundary check rejects delegation to the
  tentative adapter or binding-side inliner, proving Python performs discovery,
  traversal, preflight, and remapping;
- pass selection uses per-call structural metadata, not Gemm/Conv names or one
  hard-coded descriptor;
- scalar, nested-loop, mutable-array, constant-array, shared-helper, and
  multiple-call-site fixtures inline with correct ownership, distinct locals,
  repaired block parents, preserved attributes, and single evaluation of
  actuals;
- positive fixtures prove same-module global type/constant identity is reused,
  source positions survive cloning, every attribute byte/type/count is exact,
  and `RETV` becomes `STP` to the original result preg while the caller's
  existing `LDP` remains;
- predicate false leaves a call untouched, predicate exceptions and invalid
  policies fail without mutation, and the `always` policy removes every
  supported selected call;
- missing or cross-module callees, malformed metadata, arity/type/result
  mismatches, recursion, non-leaf helpers, zero/multiple/nested returns,
  unsupported pointer/array/preg/IV ownership, and invalid control flow all
  produce stable diagnostics;
- a batch containing one valid and one invalid selected call fails before
  either call is changed;
- inliner invalidation forces reachability recomputation before cleanup;
  cleanup removes each eligible unreachable helper once and no other function;
- standalone cleanup fixtures over non-Vector functions cover eligible and
  ineligible dead functions, program/export/address-taken roots, a helper shared
  by selected and unselected calls, predicate false, and predicate exceptions;
  retained functions and failure rollback are exact;
- a second identical manager run is a verified no-op with zero inliner/cleanup
  counts and no revision change;
- normalized post-inline AIR, selection order, metrics, and diagnostics are
  deterministic across hash seeds and shuffled test order;
- Python and tentative-native explicit oracle runs select the same calls,
  preserve single evaluation and ownership, remove the same eligible helpers,
  and produce normalized-equal post-pass AIR only for native-supported overlap
  fixtures (loops and the four Vector kernels). Scalar-formal coverage remains
  Python-only because the tentative implementation rejects it;
- post-pass AIR verifies, contains no selected generated-helper call or eligible
  unreachable helper, and is structurally ready for Vector-to-SIHE;
- the complete M13 framework, ownership, invalidation, transaction, and hook
  suite remains green, and the M14 tests are in the maintained quick gate.

Closes: the implementation portion of G12.

### M15. Integrate the M14 Python pass sequence and validate downstream lowering

Register the ordered M14 `FunctionInlinerPass ->
DeadFunctionEliminationPass` sequence at the M13
`PipelinePoint.BEFORE_VECTOR2SIHE` hook for DSL-generated helpers. Keep
inlining mandatory with `policy="always"` for the initial end-to-end pipeline.
Native paths have no default inliner pass. The tentative binding-side
implementation remains available only through an explicit differential-test
adapter and is never a second production selection.

Run the NN shape/attribute matrix from M11 through the complete
pipeline for all four provider/kernel-implementation combinations. Native paths
bypass the inliner; DSL paths must pass through it.

Acceptance gate:

- all four DSL kernels contain calls before the M14 pass sequence; a successful
  mandatory run selects and removes every structurally tagged generated-helper
  call, configures cleanup to accept every generated helper, and removes every
  unreachable generated helper;
- an unsupported tagged call fails the pass sequence with a structured
  diagnostic before Vector-to-SIHE rather than surviving under `always`;
- `AcePipeline`, `Pipeline`, and `ace_compile` invoke the exact ordered sequence
  once, expose the same ordered execution records, never select the tentative
  adapter in production, and stop before Vector-to-SIHE on pass failure;
- the real inliner-plus-cleanup sequence runs through each runner on a
  representative DSL fixture, not only through a hook test double;
- C++ and Python provider results retain M12 equality of the normalized semantic
  package, payloads, keys, and names through both native and DSL downstream paths;
- Vector-to-SIHE rotation sets match native lowering;
- Vector-to-SIHE, SIHE-to-CKKS, CKKS-to-POLY, and Poly-to-C complete;
- generated C compiles;
- deterministic native-versus-DSL outputs agree within tolerance;
- the complete maintained quick gate and existing `ace_edsl`, NN, Vector, and
  pipeline tests pass;
- DSL loop/op/rotation counts do not exceed the native oracle without an
  explicitly approved semantic or optimization change;
- a pass failure reports its pass identity/diagnostics, is fatal after AIR
  mutation, and cannot continue downstream or trigger fallback;
- explicit, diagnosed `fallback=cpp-native` remains available until another
  `plan_provider`/`kernel_impl` tuple is intentionally promoted, but only for
  the two M4 cases: an unavailable requested provider or a valid plan with no
  supported DSL recipe. It is tested separately from pass failures, while
  `cpp + native + auto` remains the default.

Provides runtime closure for G11 and final production validation of G2, G9,
G10, and G12.

### M16. Retire the tentative inliner after Python-inliner validation

Begin M16 only after the independent Python inliner implemented in M14 passes
its unit and negative tests and the M15 downstream matrix passes with the Python
pass sequence as the only selected production inliner. The tentative
binding-side inliner may remain available through M13-M15 only as a transition
aid and differential oracle; it is not a second production implementation.

Before removing the tentative path, compare it with the Python pass on the
supported generated-helper fixtures. Both paths must select the same tagged
calls, preserve single evaluation and symbol ownership, remove the same dead
helpers, produce normalized-equal post-inline AIR, and retain the same
downstream rotation sets and deterministic runtime results. Confirm that the
M14 permanent frozen fixtures match the final live-oracle results, then remove
the transition-only live-oracle tests along with the adapter.

Then remove the tentative inliner implementation and all delegation to it,
including:

- the binding-side generated-helper inlining algorithm and its build-system
  registration;
- the `GlobScope.inline_generated_vector_kernel_helpers()` entry point and any
  Python wrapper or pipeline branch that delegates inlining to it;
- `TentativeNativeInlinerAdapter`, transition-only fallback, gating,
  diagnostics, and live differential tests that exist solely for the tentative
  path;
- implementation support used only by the tentative algorithm and left unused
  by the M13 framework or M14 Python passes.

Retain the structural generated-helper/call metadata, the generic safe AIR
traversal and mutation bindings required by M13, the Python
`FunctionInlinerPass`, `FunctionReachabilityAnalysis`,
`DeadFunctionEliminationPass`, generic hook, transactional verification, and
their tests. If a generic primitive is currently colocated with the tentative
implementation but is required by M13 or M14, move it to neutral AIR binding
ownership rather than deleting or duplicating it.

Acceptance gate:

- the complete M13 framework and permanent M14 pass unit, ownership,
  malformed-input, frozen-fixture, and transactional-failure suites pass before
  and after removal;
- the transition-only live differential suite passes immediately before
  removal and is then absent; no retained test requires the deleted oracle;
- a transition-only pre-removal downstream comparison runs the adapter and
  Python paths separately on their supported overlap and proves equal rotation
  sets, generated C behavior, and deterministic runtime results without making
  the adapter a production selection;
- the complete M15 provider/kernel matrix and downstream end-to-end suite passes
  with the tentative entry point absent;
- normalized post-inline AIR, rotation sets, generated C, and deterministic
  outputs remain unchanged for the supported DSL kernels;
- no production or test path calls, imports, links, or conditionally falls back
  to the tentative inliner;
- repository search finds no remaining tentative-inliner entry point,
  implementation, build registration, or delegation shim;
- a clean bindings rebuild from an empty build output, CMake registration audit,
  and extension symbol/string inspection prove no stale object or exported
  tentative-inliner symbol survives;
- `cpp + native + auto` remains unchanged, and DSL paths use only the M14
  Python pass sequence before Vector-to-SIHE.

M16 is cleanup of the transition implementation only. It does not add
preserved-call lowering, optional inlining policies, or another inliner
algorithm. Completes the production cleanup portion of G12.

## Parallel Work and Merge Order

### Historical Phase-A merge order

After M0 froze the plan schema, helper ABI, and structural comparison rules,
these DSL tracks could proceed in parallel:

1. M1 destination-module materialization and call bridge;
2. M2 Core/type/domain substrate and M3 Vector primitives;
3. native fixtures and pre-inline AIR normalization;
4. M4 provider-neutral seam and C++ reference-provider extraction, provided the
   M0 schema is stable.

M1-M3 implementation work may proceed in parallel, but M2 and M3 close only
after their M1-based integration fixtures pass. They merge with M4 before a real
kernel port begins. Baseline Gemm lands first as the integration pilot, followed
by baseline Conv and the M7 baseline structural gate. Fast-plan fixtures may be
prepared during M5-M7. Fast Gemm merges before fast Conv because fast Conv adds
cyclic masks, collective reductions, group handling, and sharding.

Required Phase-A order:

```text
M0 -> (M1 || M2/M3 || structural test harness || M4 preparation)
   -> M4 -> M5 -> M6 -> M7 -> M8 -> M9 -> M10 -> M11
```

### Phase-B/Phase-C integration order

Do not make the production Python planning algorithms, generic Python AIR pass
framework, or Python inliner a prerequisite for M5-M11. Start them only after
the four DSL helper bodies, NN-level authoring, and all pre-inline structural
tests are stable. First add the provider implementation and differential tests,
then establish the generic framework, implement its concrete passes, and
validate production integration:

```text
M11 -> M12 (Python plan provider)
    -> M13 (generic Python AIR pass framework)
    -> M14 (Python inliner + dead-function cleanup)
    -> M15 (production integration and downstream validation)
    -> M16 (retire tentative inliner)
```

The M12 Python provider is ordinary compile-time planning code behind the M4
data boundary, not an AIR pass. M13 owns the generic framework, safe structural
bindings, analyses, transactions, results, and pipeline point. M14 implements
the inliner and dead-function cleanup as ordinary framework clients. M15 owns
their production selection, full-pipeline validation, and controlled rollout.
M16 removes the tentative binding-side implementation only after the Python
passes have independently satisfied those gates.

## Follow-on: Make Inlining Truly Optional

The function-first representation deliberately permits a future pipeline in
which selected helpers remain as calls. That mode is not part of the Full
Project Definition of Done because current downstream call handling is
incomplete.

Before `preserve-generated` may enter Vector-to-SIHE in production, add and
validate:

- explicit plaintext, ciphertext, and scalar parameter classification;
- correct conversion of every helper formal and return type;
- propagation of call attributes through Vector-to-SIHE;
- analysis and attachment of each helper's `MUL_DEPTH` summary;
- single-result ABI verification and a documented policy for nested calls;
- CKKS scale/noise and bootstrap-management parity against the inlined form;
- Poly/C code-generation and runtime parity for preserved calls.

Only after those gates pass should the policy expose production `always`,
`never`, or `auto` inlining. Until then, function generation is canonical but
production lowering always inlines generated Vector helpers.

## Phase-A Exit Criteria (satisfied by M11)

Phase A was complete when:

- M4 exposes the AIR-independent request, per-pass provider, centrally validated
  owned prepared package, and default/reference C++ provider;
- `plan_provider`, `kernel_impl`, and `plan_kind` are orthogonal, with
  `cpp + native + auto` as the default;
- invalid provider results fail before destination mutation, fallback is never
  silent, and the validator proves the v1/key derivation invariant before equal
  canonical plans receive provider-independent helper identity;
- each selected `@vector_kernel` lowering is materialized as a complete,
  destination-module helper with a deterministic specialization identity;
- the original NN expression is replaced by a valid typed `CORE.CALL` plus
  `CORE.LDP` bridge;
- baseline and fast Gemm/Conv preserve their native Core loop structures;
- helper bodies use genuine Core and Vector AIR nodes only;
- dynamic rotations carry the expected signed `RNUM` candidates;
- result types, constants, masks, reductions, bias placement, and `SLOT` match
  native pre-inline AIR;
- no supported NN Conv/Gemm remains when `kernel_impl=dsl` is selected;
- all four helpers and callers verify after `Mv2v_opt`;
- all four DSL implementations pass normalized pre-inline structural comparison;
- typed NN-level Conv/Gemm builders author complete, valid nodes and reach the
  expected pre-inline helper/call representation;
- tests stop before Vector-to-SIHE and do not depend on an inliner;
- native lowering remains unchanged and `fallback=cpp-native` is explicit and
  diagnosed.

Implementing the production Python plan provider, generic Python AIR pass
framework, or Python AIR function-inliner pass was explicitly not required to
close Phase A.

## Full Project Definition of Done

The remaining project is complete when:

- the Python plan provider implements all four variants behind the M4 data-only
  boundary;
- C++ and Python planning produce equal normalized semantic packages, excluding
  diagnostics-only provenance but including every descriptor, payload byte and
  hash, specialization key, and helper name for supported cases;
- all four provider/kernel-implementation combinations pass the pre-inline and
  downstream matrices;
- malformed provider output is centrally rejected without partial AIR mutation,
  and provider identity never changes helper deduplication;
- the generic Python AIR framework provides deterministic ordered execution,
  structured results and diagnostics, invocation-scoped analysis caching and
  invalidation, verified transactions, revision/stale-view ownership, safe
  structural AIR bindings, and a shared `BEFORE_VECTOR2SIHE` pipeline point;
- independent Python `FunctionInlinerPass` and
  `DeadFunctionEliminationPass` transformations use that framework, with
  reachability supplied by an ordinary cached analysis;
- the ordered inliner/cleanup sequence runs after Tensor-to-Vector/`Mv2v_opt`
  and before Vector-to-SIHE;
- the tentative binding-side inliner and every delegation or fallback to it are
  removed after Python-inliner correctness and downstream parity are confirmed;
- no generated helper call or unreachable generated helper function reaches
  Vector-to-SIHE under the initial production policy;
- all four kernels complete downstream lowering and C compilation;
- deterministic runtime comparisons pass;
- the NN Conv/Gemm programs authored in Phase A/M11 complete the full
  lowering pipeline;
- native-versus-DSL structural, rotation-set, and runtime gates all pass;
- `cpp + native + auto` remains the default/reference path until another
  `plan_provider`/`kernel_impl` tuple is intentionally promoted, and fallback
  remains explicit and diagnosed.
