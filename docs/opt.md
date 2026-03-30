# Primitive Bootstrap Optimization Plan

## Goal

Speed up the real decomposition-based DSL bootstrap without cheating.

Non-goals:

- do not lower primitive mode back to `CKKS.bootstrap`
- do not call `Eval_bootstrap_ciph(...)` as the implementation
- do not regress the resnet integration correctness that is currently working

The optimization target is the generated primitive bootstrap used by:

- [bootstrap_full.py](/home/dyf/code/ace-compiler/ace_edsl/examples/bootstrap_full.py)
- [a_dsl_bts.sh](/home/dyf/code/ace-compiler/a_dsl_bts.sh)

## Current Performance Snapshot

### First-Bootstrap Timing

Measured in `ace-compiler-dev` using the first-bootstrap-only path.

- Runtime rtlib bootstrap:
  - about `28.241s`
- DSL primitive bootstrap:
  - about `203.140s`

So the first bootstrap alone is about `7.2x` slower than rtlib.

### End-to-End One-Image Resnet Run

Measured in `ace-compiler-dev` on one image with the primitive DSL bootstrap integrated into `resnet20_cifar10`.

- successful run wall time:
  - about `73m15s`
- executed bootstrap count:
  - `21`
- observed per-bootstrap elapsed times:
  - roughly `192s` to `203s` each

### Generated Bootstrap Size

From [bootstrap_full.c](/home/dyf/code/ace-compiler/ace_edsl/examples/output/bootstrap_full.c):

- size:
  - about `334,608,195` bytes
- line count:
  - about `3,129,878`

That size alone is a warning sign:

- compile time is high
- instruction/cache locality will be poor
- runtime overhead from fully expanded helper calls will be high

## Profiling Constraints

The container now has `perf` installed, but it still cannot be used for real
sampling here:

- `perf stat` / `perf record` fail with:
  - `No permission to enable task-clock event`

The usable profiler in this environment is `gprofng`.

## Lightweight Profiling Findings

Heavy `gprof` instrumentation of the full DSL bootstrap body is possible in
principle, but compiling the profiled generated body is itself very expensive
because the body is hundreds of MB. So the current profiling data is based on:

- first-bootstrap wall-clock timings
- generated-C operation census
- raw AIR op counts
- direct inspection of the generated helper structure
- `gprofng` sampling where possible

### Generated C Operation Counts

From [bootstrap_full.c](/home/dyf/code/ace-compiler/ace_edsl/examples/output/bootstrap_full.c):

- `Rotate(`: `374`
- `Encode_dcmplx_ext(`: `378`
- `Init_ciph_up_scale_plain(`: `474`
- `Init_ciph_down_scale(`: `508`
- `Init_ciph_same_scale(`: `538`
- `Relinearize(`: `36`
- `Hw_modmul(`: `1092`
- `Hw_modadd(`: `1105`

### Raw AIR Operation Counts

From [bootstrap_full_raw.air](/home/dyf/code/ace-compiler/ace_edsl/examples/output/bootstrap_full_raw.air):

- `CKKS.rotate`: `372`
- `CKKS.mul`: `510`
- `CKKS.add`: `546`
- `CKKS.sub`: `13`
- `CKKS.modswitch`: `10`
- `CKKS.raise_mod`: `1`
- `CKKS.mul_mono`: `2`
- `CKKS.conjugate`: `1`
- `CKKS.encode`: `506`

### Collapsed-FFT Plaintext Level Structure

For the current `65536`/`32768` full-packed case, the generated complex plaintext encodes are concentrated at:

- encode-side levels:
  - `31`, `30`, `29`
- decode-side levels:
  - `19`, `18`, `17`

That now matches the rtlib `Rotate_precomp(...)` formulas.

## gprofng Results

### Full First-Run Process Profile

Profiler:

- `gprofng collect app -p hi`

Important caveat:

- this captures the full DSL process around the first bootstrap path
- so it includes one-time runtime setup such as rotation/switching key
  generation, not just the steady-state bootstrap body

Top functions by sampled CPU time:

- `Forward_transform`: about `35.6%` exclusive
- `Mul_poly`: about `22.2%` inclusive
- `Sub_poly`: about `20.1%` inclusive
- `Add_poly`: about `4.75%` exclusive
- `Transform_values_to_dcrt.isra.0`: about `4.38%`
- `Encode_impl`: about `9.78%` inclusive
- `Sample_uniform`: about `16.9%` inclusive
- `blake2b_compress`: about `11.5%` exclusive

Most important call-tree finding:

- the sampled run is dominated by setup under:
  - `Generate_rot_maps._omp_fn.0`
  - `Generate_rot_key`
  - `Generate_switching_key`

So the full-process profile says:

1. first-run setup is expensive because the DSL decomposition requires many
   rotation keys
2. once inside the heavy math, the cost is dominated by NTT/FFT and polynomial
   arithmetic, not plaintext ownership

### Gated First-Bootstrap Profiling Attempt

I also tried a signal-gated `gprofng` run that started sampling on:

- `[dsl_bts] begin call=1`

and stopped on:

- `[dsl_bts] end   call=1`

This produced an experiment with zero attributed CPU time, so it was not
useful for function-level attribution.

Practical takeaway:

- the trustworthy profile data we have today is the full-process `gprofng`
  sample plus the direct first-bootstrap wall-clock runs
- that profile is still enough to rule out `Copy_plain` as the main hotspot
  and to show that NTT/polynomial kernels and first-run key generation are the
  dominant costs

## Main Reasons The DSL Version Is Slow

### 1. Repeated Plaintext Encoding Inside Each Bootstrap Call

This is the largest expected cost.

The generated DSL bootstrap currently performs about `378` `Encode_dcmplx_ext(...)` calls per bootstrap invocation.

Rtlib does not do that per bootstrap call. Rtlib computes and stores those diagonal plaintexts once in bootstrap precompute and then reuses them.

Practical consequence:

- the DSL version repeatedly pays FFT/encoding/preparation cost for the same diagonal data
- rtlib amortizes that cost across all bootstrap calls

### 2. Hundreds of Standalone Rotations

The generated DSL bootstrap issues about `374` `Rotate(...)` calls per bootstrap.

Each one is expensive:

- automorphism lookup
- key-switch work
- modulus handling / mod-down path

Rtlib’s bootstrap uses structured precompute and hoisted scheduling inside the transform implementation instead of fully materializing every rotation/plain pair as a standalone high-level step.

### 3. Very Heavy Scale-Management Plumbing

Per bootstrap call, the DSL-generated code also emits:

- `474` up-scale-plain initializations
- `508` down-scale initializations
- `538` same-scale initializations

That means the decomposition is not only doing the math itself, it is also paying a large amount of helper/setup overhead around the math.

### 4. Fully Expanded Code Instead of Structured Stage Code

The generated bootstrap body is fully expanded into millions of lines of helper calls.

This hurts:

- compile time
- icache locality
- branch predictor friendliness
- ability to hoist common subexpressions / repeated runtime setup

Rtlib is much more compact and structured.

## Concrete Optimization Plan

The order below is intentional. Each step is listed in descending expected impact.

### Phase 1: Cache Encoded Diagonal Plaintexts

Status:

- implemented
- correctness restored after fixing a separate raise-mod level mismatch in the
  demo path

What changed:

1. Added an `encode_cache` attr on cacheable CKKS encodes.
2. Marked diagonal complex plaintext encodes in the primitive collapsed-FFT
   path as cacheable.
3. Emitted lazy static cached plaintexts in CKKS `ir2c`, followed by
   `Copy_plain(...)` into the local generated temporary so ownership/freeing
   still works.
4. Added `Copy_plain(...)` to rtlib as a deep-copy helper for plaintexts.

Implementation notes:

- current generated primitive bootstrap now emits cached static plaintext
  blocks plus `Copy_plain(...)` reuse instead of rebuilding every diagonal
  plaintext object each time
- in the demo-sized generated C, the old repeated encode pattern became:
  - `282` `static PLAINTEXT _pre_plain_*`
  - `282` `Copy_plain(...)`
- a transient failure during bring-up was not caused by caching itself:
  - the demo primitive bootstrap was asking `Raise_mod` for one level above the
    available Q chain in the shared-lib harness
  - fixed by restoring a consistent interpretation of
    `ACE_BOOTSTRAP_MUL_LEVEL` in the demo path:
    - `raise_mod(_bootstrap_mul_level())`
    - collapsed-FFT planning uses `level_0 = mul_level`

Plan:

1. Identify all repeated diagonal plaintexts used by:
   - `coeffs_to_slots_primitive`
   - `slots_to_coeffs_primitive`
2. Move them out of per-call execution.
3. Replace per-call `Encode_dcmplx_ext(...)` with one-time initialization plus reuse.

Possible implementations:

- generated static bootstrap-init function plus cached plaintext storage
- or emitted per-stage cached arrays initialized lazily on first use

Success criteria:

- `Encode_dcmplx_ext(...)` count in generated body drops dramatically
- first-bootstrap runtime decreases materially

Expected impact:

- very high

Current result:

- targeted bootstrap checks:
  - `test_python_and_c_api_results_match`: passes
  - `test_z_inline_and_rtlib_results_match`: passes
- integrated first-bootstrap resnet probe:
  - before Phase 1: about `203.140s`
  - after Phase 1: about `188.650s`

So the current Phase 1 improvement on the first bootstrap is about `14.49s`
or roughly `7.1%`.

### Phase 1 Follow-up: Borrowed Precomputed Plaintexts

Status:

- implemented
- correctness preserved
- no additional speedup observed in the first-bootstrap probe

What changed:

- replaced `Copy_plain(...)` reuse with borrowed/static plaintext aliasing in
  generated CKKS C
- taught the poly `MFREE_PASS` to stop freeing cache-backed plaintext temps

Validation:

- generated C now shows:
  - `Copy_plain(`: `0`
  - direct cached aliases (`= _pre_plain_...`): yes
  - no `Free_data(&_pgen_tmp_..._poly)` for cache-backed plaintext temps
- bootstrap correctness checks still pass

Measured first-bootstrap result:

- borrowed-cache path:
  - about `192.475s`

Interpretation:

- this is effectively noise relative to the earlier `188.650s` run
- avoiding deep `Copy_plain(...)` did not produce a meaningful wall-time win

Likely reason:

- the current generated decomposition still creates many distinct cached encode
  nodes, and each is used only once per bootstrap body
- so we are not getting strong reuse inside a single bootstrap call
- removing the deep copy fixed ownership and makes the model more rtlib-like,
  but it does not address the larger structural costs:
  - hundreds of unique diagonal encodes
  - hundreds of rotations
  - large scale-helper overhead
  - heavy NTT/polynomial arithmetic in `Forward_transform`, `Mul_poly`,
    `Sub_poly`, and `Add_poly`
  - expensive first-run rotation/switching key generation

### Phase 2: Hoist Rotations Inside Each Collapsed-FFT Stage

Status:

- implemented as two steps:
  1. stage-level grouping by rotation index
  2. rtlib-style baby-step/giant-step (BSGS) stage evaluation

What changed:

1. Inspected the actual collapsed-FFT stage plans and confirmed that some
   stages contained duplicate rotations.
2. Grouped each stage by rotation index.
3. Summed all diagonals sharing the same rotation before plaintext encoding.
4. Emitted one rotate + one plaintext encode + one ciphertext/plain multiply
   per unique rotation in that stage.

Important observation:

- this is better than just caching rotated ciphertexts
- for duplicated rotations, we now reduce all of:
  - `Rotate(...)`
  - `Encode_dcmplx_ext(...)`
  - `Init_ciph_up_scale_plain(...)`
  - downstream add/sub plumbing

Concrete duplicate pattern found:

- for `32768` slots:
  - one encoding stage had `63` terms but only `32` unique rotations
  - one decoding stage had `63` terms but only `32` unique rotations

Generated-code impact on the `65536` resnet bootstrap body:

- raw AIR:
  - after stage grouping:
    - `CKKS.rotate`: `372 -> 310`
    - `CKKS.mul`: `510 -> 448`
    - `CKKS.add`: `546 -> 484`
  - after BSGS stage evaluation:
    - `CKKS.rotate`: `372 -> 112`
    - `CKKS.mul`: `510 -> 510`
    - `CKKS.add`: `546 -> 546`
- generated C:
  - after stage grouping:
    - `Rotate(`: `374 -> 240`
    - `Encode_dcmplx_ext(`: `378 -> 244`
    - `Init_ciph_up_scale_plain(`: `474 -> 340`
    - `Init_ciph_down_scale(`: `508 -> 374`
    - `Init_ciph_same_scale(`: `538 -> 404`
  - after BSGS stage evaluation:
    - `Rotate(`: `374 -> 92`
    - `Encode_dcmplx_ext(`: `378 -> 282`
    - `Init_ciph_up_scale_plain(`: `474 -> 378`
    - `Init_ciph_down_scale(`: `508 -> 412`
    - `Init_ciph_same_scale(`: `538 -> 442`
- generated body size:
  - after stage grouping:
    - about `334.7 MB -> 321.3 MB`
  - after BSGS stage evaluation:
    - about `334.7 MB -> 334.6 MB`
- generated line count:
  - after stage grouping:
    - about `3.13M -> 2.62M`
  - after BSGS stage evaluation:
    - about `3.13M -> 3.13M`

Validation:

- `tests/test_resnet_bootstrap_utils.py`: passes
- `tests/test_bootstrap_full.py -k "python_and_c_api_results_match or z_inline_and_rtlib_results_match"`: passes

Measured first-bootstrap result:

- before any phase-2 work:
  - about `192.475s` on the latest pre-grouped borrowed-cache path
- after stage-level grouping:
  - about `159.926s`
- after BSGS stage evaluation:
  - about `102.728s`

So the phase-2 work improved the first bootstrap in two steps:

- stage grouping:
  - about `32.55s` faster
  - about `16.9%` better than the pre-grouped borrowed-cache path
- BSGS stage evaluation:
  - about `57.20s` faster than the grouped-only path
  - about `35.8%` better than the grouped-only path

Overall versus the `192.475s` borrowed-cache baseline:

- about `89.75s` faster
- about `46.6%` improvement

Interpretation:

- stage grouping helped by reducing duplicate direct stage terms
- BSGS helped much more because it finally reduced the number of distinct
  runtime rotations toward the rtlib-style schedule, which directly attacks
  both setup/keygen pressure and hot-path rotation work

### Phase 2 Rotation-Set Check Against rtlib

After the BSGS rewrite, I compared the generated DSL bootstrap rotation set
directly against the rtlib `Find_rot_indices(...)` formula for the same
`32768`-slot, `level_budget=3` case.

Result:

- generated unique bootstrap rotations: `53`
- rtlib theoretical bootstrap rotations: `53`
- extra rotations: `0`
- missing rotations: `0`

So at this point the DSL bootstrap no longer has an oversized bootstrap
rotation-key set relative to rtlib for the full-packed `32768` case.

Consequence:

- there is no remaining “free” setup win available from trimming the bootstrap
  rotation-key set further at this layer
- the remaining first-run setup cost now comes from:
  - the unavoidable bootstrap rotation set itself
  - model rotations already required by resnet
  - runtime key generation implementation cost

### Runtime Bootstrap Precompute Skip

Observation:

- ANT `Prepare_context()` was still calling:
  - `Bootstrap_precom(default_slots)`
- that means the DSL-integrated resnet binary was paying rtlib bootstrap
  setup/keygen cost even though it never called `Eval_bootstrap_ciph(...)`

Fix:

- added a runtime env guard:
  - `RTLIB_DISABLE_BOOTSTRAP_PRECOM=1`
- wired:
  - [a_dsl_bts.sh](/home/dyf/code/ace-compiler/a_dsl_bts.sh)
  - [verify_resnet_first.sh](/home/dyf/code/ace-compiler/verify_resnet_first.sh)
  to set that env var only for the DSL path

Measured first-bootstrap result on the current BSGS path:

- before skipping unused runtime bootstrap precompute:
  - about `102.728s`
- after skipping it:
  - about `102.099s`

Interpretation:

- this is only about `0.63s` faster, so the startup-only rtlib bootstrap
  precompute was not the dominant remaining cost in the one-round DSL path
- after the BSGS rewrite, the first-bootstrap runtime is now dominated much
  more by the bootstrap body itself than by the unused rtlib bootstrap setup

### Blocked Attempt: Offline Encoded Plaintext Buffers

Goal:

- eliminate runtime `Encode_ext_at_level` for bootstrap diagonals by storing
  pre-encoded plaintext buffers in the data file (`DE_PLAINTEXT`) and loading
  them at runtime

Implementation attempt:

- added `DE_PLAINTEXT` emission support for cacheable complex bootstrap encodes
- routed generated bootstrap diagonals to `Pt_from_msg(...)`
- enabled the path behind `ACE_BOOTSTRAP_CT_ENCODE`

What happened:

- this path is **blocked** for the `65536` bootstrap/resnet case
- even standalone encoder tests crash at `65536`:
  - `Encode_dcmplx_ext(...)` under encode-only context segfaults
  - the same happens in a full ANT runtime-context standalone test
- smaller `16384` standalone encode tests succeed

Observed behavior:

- the crash occurs in:
  - `Forward_transform`
  - called from `Encode_impl`
  - called from `Encode_ext_at_level`
- so the issue is in the ANT runtime encode path for this high-degree case,
  not in the DSL transform logic itself

Current status:

- the offline-plaintext path is left implemented but disabled by default
- bootstrap demo/default path stays on the working BSGS runtime-encode route

Practical conclusion:

- this is still the right direction according to profiling
- but it is blocked by a runtime-side encoder bug at `65536`
- the next optimization work should stay on the working path unless we decide
  to debug/fix the ANT `Encode_dcmplx_ext` crash directly

Plan:

1. Inspect `_apply_collapsed_fft_transform(...)` in
   [bootstrap_decomposition.py](/home/dyf/code/ace-compiler/ace_edsl/edsl/core/bootstrap_decomposition.py)
2. Group terms by rotation index.
3. Compute each rotated ciphertext once per stage.
4. Reuse the rotated ciphertext across all matching diagonal multiplies.

Why:

- many stage terms reuse the same rotated ciphertext shape
- current code rotates first, then immediately multiplies by one diagonal plain, then repeats

Success criteria:

- fewer live `Rotate(...)` invocations in generated C
- fewer repeated key-switch operations at runtime

Expected impact:

- high

### Phase 3: Emit Structured Stage Loops Instead of Fully Expanded Helper Sequences

Status:

- not implemented yet

Plan:

1. Stop emitting one giant straight-line helper sequence for each transform stage.
2. Emit looped/generated stage code over:
   - stage index
   - diagonal index
   - rotation index
3. Keep the same semantics but reduce C size and repeated helper boilerplate.

Why:

- the current body is too large
- compile time is excessive
- code locality is poor

Success criteria:

- generated C file size drops significantly
- compile time drops significantly
- runtime may improve from better locality even before deeper algorithmic changes

Expected impact:

- medium to high

### Phase 4: Reuse Plaintext/Temporary Objects More Aggressively

Status:

- not implemented yet

Plan:

1. Audit temporary plaintext and ciphertext lifetimes in generated code.
2. Reduce repeated init/free churn where safe.
3. Reuse stage-local buffers instead of allocating equivalent temporary shapes repeatedly.

Targets:

- `Init_ciph_up_scale_plain`
- `Init_ciph_down_scale`
- `Init_ciph_same_scale`

Success criteria:

- fewer helper init/free calls in generated code

Expected impact:

- medium

### Phase 5: Tighten EvalMod Lowering

Status:

- partially aligned semantically, not yet optimized

Plan:

1. Revisit the PS Chebyshev lowering with performance, not just correctness, in mind.
2. Check whether some current `mod_switch` or add-tree choices are semantically right but operationally suboptimal.
3. Compare the generated sequence more directly to rtlib `chebyshev_impl.c`.

Important:

- do not remove the level-alignment fixes that restored correctness
- optimize only after preserving current validated behavior

Expected impact:

- medium

### Phase 6: Resnet Integration Cleanup

Status:

- integration works
- helper scripts still do substantial glue work

Plan:

1. Move more of the shell-script rewrite logic into proper compiler/runtime metadata.
2. Emit needed rotation-key metadata directly instead of discovering it from generated C.
3. Reduce full-file rewrite cost in `a_dsl_bts.sh`.

Expected impact:

- small to medium on runtime
- medium on developer iteration speed

## Validation Plan For Each Optimization

Every optimization step should be validated in this order:

1. Fast regressions
   - [test_resnet_bootstrap_utils.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_resnet_bootstrap_utils.py)
   - [test_bootstrap_stage_ops.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_stage_ops.py)
   - primitive raw-AIR regression in [test_bootstrap_full.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_full.py)

2. First-bootstrap-only runtime check
   - `ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
   - check:
     - requested output level still matches
     - no runtime abort
     - first 8-slot sample remains close

3. Short comparison harness
   - [verify_resnet_first.sh](/home/dyf/code/ace-compiler/verify_resnet_first.sh)
   - compare final logits against baseline

4. Full one-image run
   - [a_dsl_bts.sh](/home/dyf/code/ace-compiler/a_dsl_bts.sh)
   - confirm:
     - all bootstrap calls still complete
     - final classification still matches

## Guardrails

These must remain true throughout optimization:

- primitive mode must stay a real decomposition
- primitive raw AIR must not contain `CKKS.bootstrap`
- generated primitive C must not just call `Eval_bootstrap_ciph(...)`
- resnet callsite target levels must still be honored
- no debug-only diagnostics should remain in committed runtime headers

## Recommended Next Optimization

Start with **Phase 1: cache encoded diagonal plaintexts**.

Reason:

- it is the clearest difference from rtlib
- it targets one of the largest visible costs (`Encode_dcmplx_ext(...)`)
- it should improve both runtime and code size

After that, do **Phase 2: hoist rotations inside each collapsed-FFT stage**.
