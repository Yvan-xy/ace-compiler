# Compiler-visible CKKS bootstrap linear transform

## Decision

The DSL should describe each collapsed FFT stage as a semantic linear
transform. It should not require the application author to choose a BSGS giant
step or manually request ModUp/ModDown hoisting. The giant-step override added
for this work is an experiment and debug control; the eventual compiler cost
model owns that choice.

The transform must lower to compiler-visible HPOLY/LPOLY operations. It must
not lower to an opaque runtime FFT-stage or bootstrap call. The intended
schedule is the same extended-basis strategy used by RTL bootstrap, expressed
as IR so later target-specific optimization remains possible:

1. Precompute the input `c1` once.
2. Produce baby-step rotations in QP without an immediate ModDown.
3. Multiply and accumulate the transform rows in QP.
4. Rotate and accumulate outer groups while keeping `c0` extended; ModDown an
   outer `c1` only when its key switch requires a Q input.
5. Perform the two final component ModDowns at the stage boundary.
6. Keep the existing DSL rescale outside the transform operation.

The first schema should use two children: the input ciphertext and one
interleaved-f64 constant containing row-major, BSGS-arranged coefficient rows.
It should carry versioned attributes for slots, term count, input and output
rotation arrays, scale degree, plaintext level, number of P primes, and encode
cache behavior. A later schema can carry logical diagonals and let the compiler
choose and arrange the BSGS plan.

## P0: BSGS-only experiment

The experiment used the ResNet profile (`N=65536`, 32,768 slots, runtime
data-Q=31/P=11, first-call target level 15, CT-encoded constants) and timed one
direct generated `bootstrap_full` invocation. Setup, key generation,
encryption, decoding, and teardown were outside the timing boundary.

| giant step | batched nonzero rotations | standalone outer rotations | median of 3 | change from g=8 | ModDown | rotation keys |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 38 | 42 | 67.8290 s | baseline | 230 | 40 |
| 16 | 78 | 18 | 65.2503 s | -3.8% | 262 | 48 |
| 32 | 154 | 6 | 76.3210 s | +12.5% | 390 | 80 |

All nine runs passed the first-16-slot identity check with maximum error below
0.00063. The three wall-time samples were:

- g=8: 67.82899, 69.25968, 64.58043 seconds
- g=16: 65.25029, 65.85874, 64.46924 seconds
- g=32: 76.32100, 77.16340, 74.04210 seconds

This rejects “minimize visible `FHE_ROTATE` calls” as the primary optimization
objective. Current `Rotate_batch_ciph` shares one precompute but performs each
output in Q and ModDowns it separately. Increasing the giant step therefore
trades expensive standalone wrappers for more batched Q-basis rotations,
plaintext multiplies, NTTs, ModDowns, rotation keys, and key memory. g=16 is a
small useful stopgap, not the mechanism that closes the RTL gap.

The local `tmp_cpu_bootstrap_once_*` scratch directories retain the complete
counter reports from these runs.

## P1: extended-arithmetic prerequisite

LPOLY previously did not dispatch the explicit `ADD_EXT`, `SUB_EXT`, and
`MUL_EXT` opcodes. The branch now lowers them through both supported paths:

- inline Q and P RNS loops using `HW_MODADD`, `HW_MODSUB`, and `HW_MODMUL`;
- generated `Rns_add_ext`, `Rns_sub_ext`, and `Rns_mul_ext` helpers.

Focused smoke tests assert that the extended opcodes disappear, their modular
operations are present, and a P-basis loop is emitted. The complete POLY unit
suite passes.

## P2-P4: compiler-visible QP schedule and exact CPU result

The branch now lowers all six collapsed FFT stages through
`CKKS.LINEAR_TRANSFORM`.  The lowering mirrors the RTL extended-basis
schedule, retains whole-QP multiply/add/rotate operations at the runtime
boundary, accumulates rows with `Mac_poly`, and runs independent baby
rotations as structured POLY parallel sections.  EvalMod weighted sums mark
their scalar products as lazy and rescale the accumulated result once, as
`Eval_linear_wsum_mutable` does in rtlib.

The two full-packed EvalMod branches are also independent.  Experimental
CKKS/POLY section markers preserve both branches as ordinary compiler-visible
operations while emitting the same two-way OpenMP scheduling used by RTL.
Compiler-generated scratch variables are section-scoped; sharing the old
function-wide `_pgen_*` buffers caused a reproducible data race and
segmentation fault.

All measurements below use the same one-round harness boundary and exact
ResNet profile described in P0.  The median values use three correct runs.

| implementation step | median CPU bootstrap | change from prior step |
| --- | ---: | ---: |
| DSL LT before grouped weighted-sum rescale | 46.511 s | - |
| DSL LT + grouped weighted-sum rescale | 41.144 s | -11.5% |
| DSL LT + grouped rescale + parallel EvalMod | **30.068 s** | **-26.9%** |
| native RTL bootstrap, exact-profile harness | **25.798 s** | target |

The three final DSL samples are 30.018855, 30.067511, and 31.266572 seconds.
Their initial first-16-slot maximum identity errors are 0.000592386,
0.000591266, and 0.000609722, all well below the 0.02 harness tolerance.  The
final harness revision checks every one of the 32,768 logical slots. Relative
to the 46.710-second historical DSL profile, 30.068 seconds is a 35.6%
reduction.
At this standalone checkpoint, the DSL overhead was 4.270 seconds, or 16.5%,
over the 25.798-second RTL median.  The complete ResNet experiment below
supersedes that checkpoint: generated-temporary lifetime management materially
changes both warm bootstrap performance and whether a 21-bootstrap process can
finish.

The final schedule also passes outside the standalone harness.  A generated
ResNet20 executable completed its real first bootstrap at target level 15 in
**28.476 seconds** and stopped only at the requested post-bootstrap diagnostic
boundary.  Its decoded first eight values differ from the existing native-RTL
first-bootstrap reference by less than 0.000037, and the all-slot extrema have
the same indices.  The generated legacy ANT context contains 48 rotation
resources, including conjugation automorphism 131071.  This integration found
and fixed a compiler/runtime ABI gap: provider-neutral analysis records
conjugation separately, while legacy `CKKS_PARAMS` must materialize it in the
rotation-key list as `2N-1`.  Phantom manifests remain provider-neutral and
continue to carry an explicit conjugation requirement.

A representative phase probe explains the remaining difference:

| phase | generated DSL | native RTL |
| --- | ---: | ---: |
| CoeffToSlot | 11.433 s | 10.220 s |
| dual EvalMod / ApproxMod | 11.198 s | 8.671 s |
| SlotToCoeff | 6.287 s | 6.163 s |
| DSL split + recombine + post-scale | 1.100 s | included in RTL outer overhead |

The standalone operation counts were already close for the operations targeted by
generic ModUp/ModDown hoisting: generated DSL has 100 ModDown, 155
DecompModUp, and 108 rescale calls; native RTL has 98, 150, and 110.  Generic
hoisting may still improve locality or a small number of calls, but it is no
longer capable of explaining the earlier 4.27-second checkpoint difference.

## P5: generated ownership and full ResNet20 E2E

The first full-network attempt exposed a problem that a one-call benchmark
cannot see.  Each generated bootstrap allocated 24 rotation-precompute arrays
without releasing them, and QP intermediate polynomials remained live well
past their last use.  The process reached about 61.2 GiB RSS at bootstrap 3
and was killed.

The LT lowering now expresses ownership explicitly.  It emits balanced
`Alloc_polys`/`Free_polys` pairs for the shared baby-step and outer-rotation
precomputes and frees materialized QP rotation/key-switch intermediates at
their schedule-defined last use.  The generated ResNet bootstrap body has 24
precompute allocations and 24 matching releases.  RSS fell to about 36.4 GiB
at bootstrap 3 and stayed in the 36.4--37.0 GiB range through bootstrap 21.  A
generation-time guard rejects an LT bootstrap with missing or unbalanced
precompute releases.

The generated and native executables were then built against the same current
installed ANT runtime and run on the same image (`expected=3`).  Both produced
the correct class and nearly identical logits.

| full one-image ResNet20 metric | generated DSL LT | current native RTL | DSL change |
| --- | ---: | ---: | ---: |
| 21 bootstrap wrappers | 431.672 s | 502.078 s | **-14.0%** |
| main graph | 620.659 s | 692.308 s | **-10.3%** |
| process wall time | 641.632 s | 717.412 s | **-10.6%** |
| instrumented `FHE_ROTATE` wrappers | 1166 | 1166 | equal |
| precomputes | 525 | 1197 | -56.1% |
| dot products | 4065 | 6162 | -34.0% |
| plaintext polynomial multiplies | 14043 | 19502 | -28.0% |
| NTTs | 297024 | 311548 | -4.7% |

This is the strict current A/B result: the original 37% deficit is closed, and
the compiler-visible schedule is 14.0% faster in aggregate bootstrap time than
the current native implementation in this workload.  Relative to the
historical generated DSL measurement of 980.959 seconds for 21 bootstraps, the
new generated path uses 431.672 seconds, a 56.0% reduction.  The equal 1166
wrapper count also confirms that eliminating the old 882-call difference was
necessary.  The remaining advantage is consistent with tighter scheduling and
less generic runtime-helper work.  `PRECOMP`, `DOT_PROD`, and `MULP` are
path-sensitive wrapper counters: generated HPOLY may inline equivalent work,
so their reductions must not be read as exact mathematical operation
eliminations.

The generated path has 5458 ModDowns versus native's 5416 and 4800
DecompModUps versus native's 4701.  It is faster despite those slightly higher
counts.  This is direct evidence that the colleague's generic ModUp/ModDown
passes are now incremental follow-up optimizations, not the primary mechanism
that closed the gap.

An additional all-slot standalone validation passed all 32,768 slots in
30.929 seconds with maximum identity error 0.0006234.  That cold, isolated
number should not be compared directly with the warmed, mixed-level ResNet
per-call average of 20.553 seconds.

The new paths remain opt-in while broader integration tests run:

- `ACE_BOOTSTRAP_LINEAR_TRANSFORM=1`
- `ACE_BOOTSTRAP_PARALLEL_EVAL_MOD=1`
- `--poly-lowering linear_transform` (selected automatically by the bootstrap
  example when the semantic transform is enabled)

The earlier standalone artifact is
`/app/tmp_cpu_bts_lt_wsum_dualpar4_g16_ct1`; the exact RTL baseline is
`/app/tmp_cpu_bts_native_rtl_g16`.  The integrated first-bootstrap transcript
is `/app/tmp_dsl_bts_resnet20/a_dsl_bts.log`.  The final full-network generated
and native transcripts are respectively
`/app/tmp_dsl_bts_resnet20/a_dsl_bts_lifetime_full.log` and
`/app/tmp_dsl_bts_resnet20/native_current_full.log`; their timing counters are
the adjacent `rt_timing_lt_lifetime_full.txt` and
`rt_timing_native_current_full.txt` files.

## Staged implementation

1. Add a non-library-call `CKKS.LINEAR_TRANSFORM` opcode, builder API, analyzer,
   scale, statistics, and verifier contracts. Reject it if it reaches ordinary
   CKKS-to-C or Phantom code generation unlowered.
2. Lower a synthetic 2-by-2 transform directly to HPOLY and assert one shared
   precompute, no baby-step ModDown, QP multiply/add operations, required outer
   `c1` ModDowns, and two final ModDowns.
3. Replace exactly one collapsed FFT stage behind a feature flag and compare it
   numerically with the current primitive implementation.
4. Enable all six stages, run the one-round CPU harness, then the ResNet E2E.
5. Evaluate the colleague's generic ModDown/ModUp passes on the now-visible
   HPOLY. Compare them with the explicit transform schedule before relying on
   either pass for correctness or performance.
