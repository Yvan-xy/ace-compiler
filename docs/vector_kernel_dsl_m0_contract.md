# Vector-Kernel Phase-A M0 Contract

This document freezes the data and comparison boundary for Phase-A
Tensor-to-Vector DSL lowering. It supplements
`vector_kernel_dsl_lowering_plan.md`; it does not enable DSL dispatch or alter
native lowering.

## Plan ownership and variants

M0 froze provider-neutral plan semantics from the existing C++ implementation.
Under M4 during Phase A, the existing C++ cost-model, packing, constant
preparation, mask-selection, and other planning algorithms serve as the default
and reference provider. C++ remains authoritative for validating and
canonicalizing provider results and for materializing destination AIR. A
selected provider may use mutable working storage, but its result must freeze
into exactly one immutable variant from `nn/vector/tensor2vector_plan.h` before
either the native or DSL emitter runs.

This amendment supersedes only the original provider-ownership statement. The
v1 plan variants, canonical hashes and keys, helper ABI, and structural fixtures
remain frozen; M4 must add a companion owned data package rather than
retroactively adding a provider interface to M0.

| Variant | AIR-affecting semantics described by the plan |
| --- | --- |
| `BASELINE_GEMM_PLAN` | Prepared diagonal weight and bias, semantic input/result types, `height`, `width`, input duplication, primary and block-reduction loops, ordered slices and rotations, mask decision, and the native slot policy. |
| `BASELINE_CONV_PLAN` | Flattened input, prepared im2col weight, expanded bias, scaled s32 rotation table, effective channel/output/kernel/stride dimensions, duplication policy, nested loop bounds, ordered slices and rotations, result type, and slot policy. |
| `FAST_GEMM_PLAN` | Original and padded matrix dimensions, IRMA constant, blocking/alignment and input replication, `bs/pb/ps/sf/gs` decisions, nested loops, slices and rotations, both optional reductions, bias/mask epilogue, result type, and slot policy. |
| `FAST_CONV_PLAN` | Effective dimensions/group/stride, prepared weight/bias and auxiliary mask/rotation constants, all former `SHARD_MAP_PARAMS` values, blocking placement, nested loops, normal or cyclic rotations, selected collective-reduction topology, optional sharding offset, result type, and slot policy. |

`VECTOR_KERNEL_COMMON_PLAN` stores ordered runtime formals, constant
descriptors, structural loop/slice/rotation/reduction descriptors, and result,
mask, and slot policies. A constant descriptor uses a semantic role and a
structural type plus canonical content hash; it never uses an AIR ID, generated
symbol, or pointer. The v1 plan intentionally describes constants without
owning their payload bytes or operand-preparation recipes.

M4 must introduce a companion `PreparedVectorKernelPlan` that owns the
canonical v1 plan, typed constant payloads keyed by semantic role, runtime
operand/preparation descriptors, and optional scalar/sharding descriptors. It
contains no AIR handles or borrowed provider/Python buffers. Provider
provenance is diagnostic only. C++ must copy and validate this data, recompute
hashes, and materialize it in the active destination `GLOB_SCOPE` before an
emitter runs.

The entire semantic package participates in normalized provider comparison;
diagnostics-only provenance does not. Every helper-body-affecting companion
descriptor must be uniquely derived from and validated against v1. Introducing
any new AIR semantic requires a versioned plan and specialization-key schema.

The current native baseline Gemm path emits no `SLOT` attribute. M0 records
that fact as `ABSENT_NATIVE_BASELINE`; M5 must initially preserve the absence in
both native and DSL emission. Changing that policy requires a separate
native-oracle and plan amendment, so planner extraction cannot hide an unrelated
native semantic change.

### Reference C++ provider input coverage audit

The schema records resolved AIR decisions rather than mutable configuration
flags. The four current native emitters and their caller epilogues map into it
as follows:

| Native source input | Frozen plan ownership |
| --- | --- |
| Baseline Gemm packed input, diagonal weight, bias, `height`, `width`, duplication, `need_mask`, and current missing `SLOT` | Common input/constant/result/mask/slot records plus `BASELINE_GEMM_PLAN` dimensions, loop/slice/rotation/reduction records, and duplication count |
| Baseline Conv packed input, im2col weight, expanded bias, scaled `ra`, effective `cin/cout/oh/ow/kernel_hw/stride`, duplication, and caller `SLOT` | Common input/constants/result/slot plus `BASELINE_CONV_PLAN` dimensions, nested loops, affine slice, ordered rotations, and duplication count |
| Fast Gemm blocked input, IRMA weight, `np/kp/bs/gs`, blocking alignment/replication, and caller bias/reduction/mask/`SLOT` | Common input/constants/result/mask/slot plus all original/padded dimensions and `bs/pb/ps/sf/gs`, loops, slices, rotations, reductions, and replication fields in `FAST_GEMM_PLAN` |
| Fast Conv blocked input, prepared weight/bias/`ra`, every `SHARD_MAP_PARAMS` value, effective dimensions/group/stride, cyclic decision, collective-reduction/mask epilogue, caller `SLOT`, and optional runtime weight offset | Common input/constants/result/mask/slot plus `FAST_CONV_PLAN` effective dimensions, full shard-map values, blocking placement, cyclic flag, selected reduction records, and optional typed/scaled sharding offset |

When M4 planner extraction lands, every mutable input capable of changing the
topology must enter the immutable planning request. A change that selects a
different topology must produce a different frozen plan package, and a native
or DSL plan-consuming emitter must not re-read that option after planning.

## Helper ABI

A generated helper is leaf and nonrecursive. Its formal order is:

1. one or more packed ranked-vector inputs, in source operand order;
2. for sharded fast Conv only, one Core `s32` weight-slice offset.

Weight, bias, rotation, and mask data are C++-materialized same-`GLOB_SCOPE`
constants referenced by `CORE.LDC`; they are not ordinary formals. For every
provider, C++ first copies and validates the owned typed payload and recomputes
its hash. A provider never returns an `LDC`, AIR node, symbol, or constant ID.
The helper owns the complete selected lowering, including blocking, reductions,
bias, masking, and `SLOT` semantics. It returns one ranked vector through
exactly one terminal `CORE.RETV`.

The optional scalar is required because the sharded weight offset is computed
from runtime sharding loop IVs. Its scale is immutable plan data, but its value
cannot be baked into a constant or specialization key.

The helper name is derived from the canonical specialization key:

```text
__ace_vkernel_<variant>_<sha256(canonical-key)>
```

The variant token is a C-safe spelling: `baseline_gemm`, `baseline_conv`,
`fast_gemm`, or `fast_conv`. The digest is the full 64-character lowercase
SHA-256 encoding. A cache entry is local to the active destination
`GLOB_SCOPE` and is valid only when both the canonical key and helper signature
match. Plan-provider identity, kernel implementation, and provenance are not
part of the key or name. When materialized as DSL helpers in the same
destination, canonically equal C++ and Python provider results must deduplicate
to the same helper; provenance belongs only in diagnostics and any separate
plan-computation cache.

## Constant and specialization keys

`Build_vector_kernel_specialization_key` emits the versioned canonical key
used by fixtures and future helper caching. Vector order is significant for
formals, constants, loops, slices, rotations, and reductions. Changing any
AIR-affecting type, shape, constant digest, loop bound/nesting, slice index,
rotation order, reduction, mask, slot, or offset policy must change the key.
M4 validation must reject a companion descriptor that changes helper AIR but is
neither represented in nor uniquely derived from this key; accepting such a
descriptor requires a versioned schema/key update.

Constant hashes use this lowercase form:

```text
sha256:<64 lowercase hexadecimal digits>
```

The SHA-256 preimage is the following binary-safe header immediately followed
by the row-major payload bytes (the newline after `payload:` is part of the
header):

```text
vector-kernel-constant:v1
element=<primitive mnemonic>
rank=<unsigned decimal rank>
shape=<comma-separated signed-decimal dimensions>
payload:
```

The raw payload begins immediately after the final newline shown above.

For every provider package, M4 requires C++ to recompute this hash from the
declared primitive type and shape plus the copied row-major payload bytes. A
provider-supplied digest is checked but never trusted. This portable byte
representation is also the Python-provider ABI. Unsupported or ambiguous
representations are rejected before AIR mutation.

Integer elements use fixed-width two's-complement little-endian bytes.
Floating and complex components use their exact fixed-width IEEE bit patterns
in little-endian order; signed zero and NaN payloads are therefore preserved.
The M0 implementation accepts integer, boolean, `f32`, `f64`, `c32`, and `c64`
primitive arrays. AIR's host-padded `f80`/`f128` and corresponding complex
representations are rejected until a portable payload representation is
defined. External-file offsets, AIR constant IDs, host pointers, and symbol
names are not part of the preimage.

The canonical plan key itself is readable and lossless. `std::hash`, pointer
values, generated IDs, and implementation-defined object bytes are forbidden
for either the key or helper name.

## Normalized pre-inline AIR comparison

The structural oracle operates on AIR objects in C++, not parsed dump text. It
compares two views:

- the native statements and caller epilogue emitted for one NN operation;
- the complete DSL helper body for the same frozen plan.

The DSL caller bridge is checked separately against the helper ABI: one typed
`CORE.CALL`, compatible actual/formal types and result preg, and the replacing
caller-owned `CORE.LDP`.

Once M12 adds the Python provider, provider differential comparison is a
separate host-data oracle. Before either emitter is compared, C++ and Python
results for the same request must have equal normalized semantic packages,
excluding diagnostics-only provenance but including every preparation/scalar
descriptor, exact owned payload bytes and hashes, specialization key, and helper
name. Matching native/DSL AIR cannot substitute for package equivalence.

Normalization assigns formals, locals, pregs, and loop IVs ordinal identities
by structural declaration/traversal order. It then compares:

- helper signature and call argument/result types;
- recursive primitive/ranked types and shapes;
- domain, opcode, result type, operand order, and def-use identity;
- block and statement order;
- loop init, condition, increment, nesting, and IV use;
- constant kind, structural type, shape, and canonical content hash;
- typed scalar/vector attribute names, element types, values, and order;
- exact ordered slice behavior and signed `RNUM` candidates.

Only generated AIR IDs, source positions, and symbol spelling are ignored.
`CORE.COMMENT` remains a statement in the normalized stream and its text is
compared; changing that policy requires an explicit plan amendment. Both input
modules must pass `GLOB_SCOPE::Verify_ir()` before comparison.

M0 behavior corresponds to the later `plan_provider=cpp`,
`kernel_impl=native`, and `plan_kind=auto` tuple; M0 itself exposed no provider
interface or selection controls. A native baseline-Gemm oracle must call public
`TENSOR2VECTOR_UTIL::New_gemm_metakernel` directly in a controlled destination
function rather than changing the production hard-selected fast-Gemm dispatch.
The required cases are `[height,width]=[4,4]` without reduction/mask and
`[2,8]` with block reduction and mask.

The M0 acceptance fixtures are:

- `test_tensor2vector_plan.cxx`, which freezes all four immutable plan keys,
  canonical constant hashing, helper naming, and helper ABI;
- `test_tensor2vector_native_air_oracle.cxx`, which directly calls the public
  baseline-Gemm emitter for both required cases, verifies the resulting AIR, and
  freezes ID/name/source-independent structural fingerprints.

The reusable native-versus-DSL normalizer now lives in
`nn-addon/vector/unittest/tensor2vector_air_normalizer.{h,cxx}`. Its native
adapter accepts an exact generated statement interval, semantic inputs, and a
logical result; its helper adapter removes only the function-entry wrapper and
the required terminal `RETV`. Both become the same canonical kernel view. A
separate bridge validator checks exact destination entry identity, ordered
expected-actual identity and formal types, the caller result preg, one attached
replacing `LDP`, one attached call site, and the helper's single terminal
`RETV`.

`test_tensor2vector_dsl_materialization.cxx` exercises the normalizer through
a synthetic native region and a destination-owned helper built with different
IDs, names, and source positions. Constant-content, kernel operand-order,
bridge actual-order, wrong-preg, detached-call, and detached-`LDP` changes are
required to fail. The frozen M0 whole-function fingerprints use the same object
walker, so extraction of the reusable harness cannot silently weaken M0.

## M1 destination helper and call seam

M1 registration is per `VECTOR_CTX`/`Vector_driver` invocation. A registered
selector sees the source NN node and already visited destination-owned actuals,
then returns a specialized helper recipe whose signature types all belong to
the active destination `GLOB_SCOPE`. It must not retain source or independent
module AIR objects.

The materializer creates the signature, function, entry, function scope,
formals, constants, locals, pregs, loops, blocks, and body in that destination.
It enforces a leaf body and appends the ABI's sole terminal `CORE.RETV`. The
caller bridge checks every actual/formal pair, allocates the result preg in the
caller, prepends exactly one `CORE.CALL`, and returns the caller-owned
`CORE.LDP` of that preg.

The synthetic M1 selector is connected ahead of the legacy skip-only path for
`NN.ADD`, proving the invocation seam through the public `Vector_driver` and
through `Mv2v_opt`. M4 will connect the same generic registry/materializer to
a centrally validated `PreparedVectorKernelPlan` and select a helper recipe by
canonical plan kind. Under M4, provider and DSL registries are scoped to one
`VECTOR_CTX`/`Vector_driver` invocation. With no registry or option override,
the default is `plan_provider=cpp`, `kernel_impl=native`, and
`plan_kind=auto`.

The future phase handoff remains:

```text
Tensor2Vector/Mv2v -> pre-inline Vector AIR -> Python inliner -> Vector2SIHE
```

This Python AIR inliner is distinct from the later Python `PlanProvider`, which
computes host-side plan data and never transforms AIR. M0, M1, and the rest of
Phase A do not implement or invoke either one.
