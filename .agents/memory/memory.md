# Long-term Memory

## Test Bring-up

- The repo's non-DSL bootstrap bring-up currently revolves around two native paths:
  - `fhe-cmplr/test/cgo25_ae.py` for compile+link+run artifact-eval flow
  - `fhe-cmplr/test/test_bts_rescale.sh` for direct bootstrap/rescale regression checks
- `cgo25_ae.py` assumes a Linux/container environment:
  - `/app` working tree
  - installed compiler trees like `ace_cmplr` / `ace_cmplr_omp`
  - GNU `time -f`
  - Python `psutil`
  - CIFAR dataset directories
- The current macOS checkout is missing the required ONNX models, CIFAR dataset, compiler install artifacts, and Python `psutil`.
- A repo-level `Dockerfile` now exists to provide the expected Linux/GNU environment and Python dependencies.
- The host `./dataset` directory should be mounted into the container at `/dataset`.
- The persistent container `ace-compiler-dev` has a working compiler build/install path:
  - build dir: `/app/build`
  - install prefix: `/usr/local`
  - main binary: `/usr/local/bin/fhe_cmplr`
- The same `/app/build` tree can be configured as a full test-enabled tree; in the current state `ctest -N` reports 76 tests.

## Bootstrap Context

- For `resnet20_cifar10` bootstrap regression without DSL, `test_bts_rescale.sh` is the narrowest backend-native test:
  - compiles `resnet20_cifar10.onnx`
  - uses CKKS options `mxbl=16:mbc=2:sbm:tsbp`
  - compares produced `MIN` level lines against `resnet20_cifar10.log`
- That script also expects external artifacts downloaded via `ossutil64` from the internal OSS path `oss://antsys-fhe/cti/ir/sihe/`.
- Private git fetches inside the container can be bypassed without source edits by using:
  - local `FetchContent` source overrides for `onnx` and `jsoncpp`
  - git `insteadOf` rewrites from private `code.alipay.com` URLs to public upstream mirrors for `jsoncpp`, `uthash`, `BLAKE2`, `googletest`, and `benchmark`
- Enabling rtlib unit tests exposed two missing standard-library includes:
  - `fhe-cmplr/rtlib/ant/unittest/ut_ckks_perf.cxx` needs `<iomanip>`
  - `fhe-cmplr/rtlib/ant/unittest/ut_ksw_opt.cxx` needs `<iomanip>`
- The generated dataset executable path `a.sh` uses in the container has two independent failure modes:
  - runtime abort if the generated `.inc` hardcodes an external `.msg` file path that does not exist
  - link failure if `/usr/local/rtlib/lib/libFHErt_ant.a` is stale relative to current source
- For `resnet20_cifar10_pre.onnx.inc`, `Get_rt_data_info()` currently hardcodes:
  - `/app/release/resnet20_cifar10_pre.onnx.omp.msg`
- That `.msg` file is the external runtime data/weights file consumed by `Pt_from_msg(...)`; if absent, runtime fails in `Rt_data_open`.
- `BUILD_WITH_OPENMP=ON` was already enabled in the working container build; OpenMP was not the cause of either the runtime abort or the `Init_ciph3_same_scale` link error.
- A real rtlib source bug existed in `fhe-cmplr/rtlib/ant/ckks/src/cipher.c`:
  - `Init_ciph3_same_scale(...)` had been placed as a nested function inside `Init_ciph3_same_scale_ciph3(...)`
  - as a result, the archive exported only `Init_ciph3_same_scale_ciph3`
  - fixing the missing brace and rebuilding the ant rtlib restored the global symbol and fixed the link error
- For native `a.sh` bring-up, rebuilding `/app/build/rtlib/build/ant/libFHErt_ant.a` is not enough by itself; the installed archive at `/usr/local/rtlib/lib/libFHErt_ant.a` must also be refreshed because the script links against the installed path.
- `ace_edsl/build.sh` should not assume a repo-local install prefix under `ace_cmplr/`.
- The working approach in `ace-compiler-dev` is:
  - treat the compiler as already installed
  - resolve `ACE_INSTALL_DIR` to `/usr/local`
  - build only bindings/python packaging on top of that install
- `bindings/CMakeLists.txt` also needs the same install-prefix behavior; otherwise the shell script and CMake layer drift apart.
- Keep docs aligned with that behavior; build-facing docs should describe an installed prefix such as `/usr/local`, not `${ACE_COMPILER_DIR}/ace_cmplr`.
- For the `encoder.c:562 invalid scaling factor for encode` regression on `resnet20_cifar10_pre.c`:
  - the generated C currently has zero `Rescale_ciph(...)` calls
  - it contains many scalar encodes of the form `Encode_float(..., Sc_degree(&cipher), Level(&cipher))`
  - that pattern is a strong indicator of runtime failure when ciphertext scale degree grows faster than remaining level budget
- Testing the older `Rescale_ana` + `Handle_rescale` + `Rescale_expr` bundle from `rescale-fix.txt` against the current compiler did not materially change that emitted code pattern for `resnet20`:
  - regenerated C still had zero `Rescale_ciph(...)`
  - sampled scalar encode lines remained structurally identical
- Conclusion from that experiment: the old rescale patch is not sufficient, by itself, to fix the observed `resnet20` runtime encode failure.
- The effective fix was to revert the newer cap/skip-rescale path introduced around:
  - `scale_manager.h`
  - `scale_manager.cxx`
  - `ckks2poly.h`
  - `ckks2hpoly.h`
  - `poly_ir_gen.cxx`
- After rebuilding/installing the compiler and regenerating `resnet20_cifar10_pre.c`, the output changed from:
  - `Init_ciph_down_scale(...)` count = 0
  to:
  - `Init_ciph_down_scale(...)` count = 188
- Running updated `a.sh` in `ace-compiler-dev` after regeneration no longer reproduced the previous early runtime error:
  - `/app/fhe-cmplr/rtlib/ant/ckks/src/encoder.c:562: invalid scaling factor for encode`
  during the observed startup / early execution window.
- ANT CKKS runtime bootstrap lives primarily in:
  - `fhe-cmplr/rtlib/ant/include/ckks/bootstrap.h`
  - `fhe-cmplr/rtlib/ant/ckks/src/bootstrap.c`
  - staged wrappers in `fhe-cmplr/rtlib/ant/ckks/src/cipher.c`
- The runtime bootstrap is organized as:
  - `Bootstrap_setup`: precompute transform matrices / collapsed-FFT plaintexts and per-slot bootstrap params
  - `Bootstrap_keygen`: derive only the rotation/conjugation keys actually needed by the precomputed transform
  - `Eval_bootstrap`: modulus raise, coeffs-to-slots, approximate mod reduction, slots-to-coeffs, final cleanup
- The approximate mod-reduction stage uses runtime-selected Chebyshev coefficients plus double-angle iterations; the selected polynomial family depends on secret-distribution assumptions and hamming weight, with optional even-polynomial mode from `RTLIB_BTS_EVEN_POLY`.
- The low-budget bootstrap path `(enc_budget == 1 && dec_budget == 1)` currently routes through `Linear_transform`, but `Linear_transform()` in `bootstrap.c` is still unimplemented, so the normal collapsed-FFT path is the one that is actually viable today.
- Current primitive-bootstrap debugging rule:
  - treat any lowering of the primitive DSL path to `CKKS.bootstrap` / `Eval_bootstrap_ciph(...)` as invalid/cheating for this project goal
  - the real decomposition path must remain visible in raw AIR and final generated C

## Server Bring-up

- Server host used: `yifan@10.28.27.58`
- Server repo path: `/media/newhd/yifan/ace-compiler`
- Prepared server-side `third_party/` with public clones for `jsoncpp`, `googletest`, `benchmark`, `uthash`, and `BLAKE2`.
- Built the server Docker image successfully from the repo `Dockerfile` with tag `ace-compiler-dev:latest`.
- Built inside the running server container `ace-compiler-dev`:
  - build dir: `/app/build`
  - generator: Ninja
  - install prefix: default `/usr/local`
  - main binary: `/usr/local/bin/fhe_cmplr`
  - `ctest -N` count: 76
- The older server checkout required extra compatibility fixes:
  - `onnx2air.cxx` needed `ParseFromString` instead of `ParseFromIstream`
  - `ut_ckks_perf.cxx` and `ut_ksw_opt.cxx` needed `<iomanip>`
  - container needed `pybind11` and `nlohmann-json3-dev`
- The newer local repo `Dockerfile` already had `pybind11`; it now also installs `nlohmann-json3-dev`.

## ACE EDSL Context

- The active Python DSL work lives in repo-root `ace_edsl/`; `air-infra/plugin/dsl_pybind11` is deprecated and should be ignored for new DSL work.
- `ace_edsl` is AIR-first:
  - it traces Python execution directly into AIR via `ace_bindings`
  - it reuses `base_dsl` mainly for AST preprocessing and surrounding infrastructure
  - MLIR paths in `base_dsl` are intentionally stubbed or bypassed
- The key Python entrypoints are:
  - `ace_edsl/edsl/edsl.py` for DSL lifecycle and AIR generation
  - `ace_edsl/edsl/domain_kernels.py` for call-time decorator dispatch
  - `ace_edsl/edsl/core/air_value.py` for operator-overload IR emission
  - `ace_edsl/edsl/pipeline.py` for AIR-to-C lowering orchestration
  - `ace_edsl/edsl/lowering_registry.py` for skip-op / Python-lowering integration
- `ace_edsl` depends on shared pybind modules built from `bindings/` into `ace_bindings/`:
  - `air_builder`
  - `nn_addon`
  - `fhe_cmplr`
  - `passmanager`
- In the current `/media/newhd/yifan/ace-compiler` checkout, `ace_bindings` shared objects are not present, so importing `ace_edsl.edsl` currently fails with missing `air_builder` bindings.
- Known `ace_edsl` limitations visible in source/tests:
  - dynamic loop bounds are not fully implemented
  - non-unit loop steps fall back to Python execution
  - dynamic control flow can be represented in AIR, but the current `vector2sihe` FHE lowering path does not support it end-to-end
  - some AIRValue reverse/less-common operators remain unimplemented
- `ace_edsl` has an important CKKS constant path in `AceEDSL.generate_air()`:
  - Python scalar/array values passed to `CkksPlaintext`-annotated params are encoded inside the kernel body
  - they are not treated as formal AIR function parameters
  - array constants support complex-valued elements
- `AIRValue.FLAT_IR_MODE` is enabled by default:
  - operation results are stored to named temporaries and reloaded on demand
  - this keeps traced AIR in a flatter SSA-like form and avoids nested-expression IR blowup
- For future work, treat the Python DSL and shared bindings as one feature surface; many meaningful changes cross that seam.
- On the current host checkout, `ace_edsl` import readiness can fail in two independent ways:
  - missing built shared objects in `ace_bindings/`
  - missing Python dependency `typing_extensions`
- The main DSL bootstrap codegen entrypoint is `ace_edsl/examples/bootstrap_full.py:run_demo()`:
  - it traces the `@ckks_kernel` bootstrap into AIR
  - runs `AcePipeline` from `start_domain="fhe::ckks"`
  - writes `ace_edsl/examples/output/bootstrap_full.c` plus stage AIR dumps
- `ACE_BOOTSTRAP_IMPL` selects the bootstrap style emitted by the DSL example:
  - `primitive` for decomposed stage-op bootstrap
  - `rtlib` for direct CKKS bootstrap op emission closest to the ANT runtime baseline
  - `evalmod` for the older explicit EvalMod demo path
- Primitive bootstrap bring-up status:
  - removing implicit CKKS rescale from `AIRValue.__mul__` was enough to make primitive bootstrap codegen-stable
  - primitive bootstrap now reaches `ckks_driver`, `poly_driver`, and `poly2c`, and emits `bootstrap_full.c`
- Primitive `CoeffToSlot` / `SlotToCoeff` now use direct `U0` / `conj(U0^T)` diagonal transforms in `bootstrap_decomposition.py`, not just butterfly skeletons
- Complex primitive plaintext diagonals required backend support changes:
  - `ir2c_ctx.h` must route `encode_dcmplx` nodes through runtime `Encode_dcmplx(...)`
  - the float32-only offline encode fast path must be bypassed for complex encodes
- The primitive bootstrap shared-lib comparison harness in `test_bootstrap_full.py` needed separate fixes:
  - `/usr/local` install-prefix discovery
  - autogenerated wrapper source for `Get_encode_scheme` / `Get_decode_scheme` / `Main_graph`
  - compile generated C and wrapper as C, not C++
- Another backend bug existed for primitive encode level propagation:
  - generated primitive `Encode_dcmplx(...)` calls were initially emitted with level `0`
  - fixing `scale_manager.cxx` to rewrite encode level children from the surrounding ciphertext changed generated calls to use `Level(&cipher_tmp)` instead
- The deeper primitive-scaling issue is not solved by transform-level fixes alone:
  - after extending the bindings to support explicit encode levels and preserving `24` / `12` for primitive transform encodes, generated `bootstrap_full.c` now emits those fixed levels correctly
  - the primitive shared-lib comparison still skips on `invalid scaling factor for encode`
  - conclusion: the remaining scaling problem is now beyond the transform diagonal encodes and likely sits in scalar/mask encodes during EvalMod and/or missing rtlib-style level-alignment behavior
- As of March 29, 2026 (UTC), the next high-value debugging target is:
  - inspect generated primitive `Encode_double_mask(...)` and related scalar/plaintext encode sites
  - compare their level/scale behavior directly against rtlib EvalMod helpers before changing rescale placement again
- Current remaining primitive-bootstrap blocker:
  - the primitive-vs-rtlib shared-lib comparison no longer fails at harness/linkage
  - it now reaches runtime and still skips on a real scaling failure:
    - `invalid scaling factor for encode`
  - so the remaining work is runtime scaling correctness in the primitive bootstrap itself
- A new integration script `a_dsl_bts.sh` exists at repo root:
  - builds a resnet20 dataset binary that overrides `Eval_bootstrap_ciph(...)` with a shim forwarding into generated primitive `bootstrap_full`
  - current result: link succeeds and inference starts, but runtime aborts with `rns_poly.c:312: Level of rescale opnd is too small`
- March 29, 2026 follow-up fixes:
  - removing `x.raise_mod(2)` from primitive `coeffs_to_slots_primitive()` was the key fix for the primitive scale explosion; that op was not equivalent to rtlib bootstrap's modulus-raise step and had been injecting an extra scale degree at the very start
  - `Post_decode_scheme()` in `fhe-cmplr/rtlib/ant/ckks/src/rtlib.c` had a real buffer-overrun bug for `desc->_count == 0`: it copied all decoded slots into the output buffer instead of capping to the requested output tensor length; fixing that made the primitive shared-lib harness stable
  - after those fixes, primitive generated bootstrap now reaches `Handle_output`, decodes cleanly, and `test_python_and_c_api_results_match` passes
  - the primitive output metadata at the shared-lib boundary is now sane: level 11, `sf_degree` 1, default scale
  - to make primitive semantics identical to rtlib, the DSL path was switched to compose the three rtlib stage ops directly:
    - coeffs-to-slots stage op
    - eval-mod stage op
    - slots-to-coeffs stage op
    via direct AIR emitters that bypass primitive stage lowering in `air_value.py`
  - after that switch, `tests/test_bootstrap_full.py` passes in primitive mode and `test_z_inline_and_rtlib_results_match` is green
  - `a_dsl_bts.sh` originally started segfaulting because the generated bootstrap helper now lowered to `Eval_bootstrap_ciph(...)`, and the script macro-replaced that symbol globally, creating recursion through the shim; fixing the script to macro-replace only the resnet translation unit, and compiling the generated bootstrap body as C, removed that crash
  - current `a_dsl_bts.sh` status: generated-bootstrap resnet links, launches, and runs past the old crash point; no immediate segfault observed
  - `a_dsl_bts.sh` shim now logs per-bootstrap progress and timing
  - measured in `ace-compiler-dev` on the one-image workload:
    - DSL bootstrap call 1 took about `43.9s`
    - DSL bootstrap call 2 took about `42.4s`
    - baseline `a.sh` is also a multi-minute job in this container and did not finish before manual termination at about `5m56s`
  - so the current follow-up area is runtime/performance characterization and full completion, not early bootstrap correctness or immediate crash handling
- Later March 29, 2026 decomposition fixes:
  - added explicit `CKKS.modswitch` handling in the compiler scale manager (`fhe-cmplr/ckks/include/scale_manager.h`)
  - restored rtlib-style Paterson-Stockmeyer baby-step level alignment in `ace_edsl/edsl/core/bootstrap_decomposition.py`
  - found and fixed a real mismatch in primitive collapsed-FFT stage plaintext levels:
    - `_collapsed_fft_stage_plan(...)` now matches rtlib `Rotate_precomp(...)`
    - encode-side plaintext levels use `level_enc + 1 + s`
    - decode-side plaintext levels use `level_dec + level_budget - s`
  - primitive bootstrap now raises to the full tower count in `ace_edsl/examples/bootstrap_full.py` with `ct.raise_mod(_bootstrap_mul_level() + 1)`
  - `a_dsl_bts.sh` and `verify_resnet_first.sh` shims now only modswitch downward to satisfy smaller requested `level_after_bts`; they do not try to repair levels upward after bootstrap
  - after those fixes, early live resnet bootstrap callsites now satisfy the requested target levels:
    - call 1: `15 -> 15`
    - call 2: `14 -> 14`
    - call 3: `16 -> 16`
    - call 4: `14 -> 14`
    - call 5: `16 -> 16`
  - the previous abort after call 3 (`rns_poly.c:312: Level of rescale opnd is too small`) is no longer reproduced in that early region
  - first-bootstrap sampled message quality stayed close to baseline when only the stage-level fix was applied (`~3e-05` max complex error on 8 sampled slots), but worsened somewhat after the later full-tower-raise change (`~2.26e-03` max complex error on the same 8-slot sample); full final-logit validation after the latest changes is still pending
  - later the same day, a full one-image `a_dsl_bts.sh` run completed successfully in `ace-compiler-dev`
  - actual executed bootstrap count in that successful run was `21`
  - the live run continued to satisfy requested target levels beyond the early region, including repeated `15`, `14`, `16`, `13`, and `6` targets
  - final outcome:
    - predicted label matched expected label `3`
    - `[RESULT] infer 1 images, pass 1 1.000, fail 0 0.000.`
    - total wall time about `73m15s`
  - `a_dsl_bts.sh` now writes future full logs to `/app/tmp_dsl_bts_resnet20/a_dsl_bts.log`
  - the latest successful PTY result was appended there as a reconstructed result block
  - a new optimization note now exists at `/home/dyf/code/ace-compiler/docs/opt.md`
  - current lightweight profiling picture recorded there:
    - rtlib first bootstrap about `28.241s`
    - DSL first bootstrap about `203.140s`
    - generated bootstrap body about `334 MB` / `3.13M` lines
    - helper counts dominated by:
      - `Rotate(`: `374`
      - `Encode_dcmplx_ext(`: `378`
      - `Init_ciph_up_scale_plain(`: `474`
      - `Init_ciph_down_scale(`: `508`
      - `Init_ciph_same_scale(`: `538`
  - main performance conclusion:
    - the DSL path is slow because it fully materializes the bootstrap algorithm, repeatedly re-encodes diagonal plaintexts, executes hundreds of standalone rotations, and pays a lot of helper/setup overhead that rtlib amortizes through precompute and specialized runtime code
- recommended optimization order in `docs/opt.md`:
  1. cache encoded diagonal plaintexts
  2. hoist/reuse rotations inside collapsed-FFT stages
- Later March 30, 2026 cleanup / integration work:
  - removed the remaining `resnet_bootstrap_utils.py` text-patching path used
    for resnet bootstrap integration
  - the generated bootstrap body now keeps its own context provider and
    `emit-body` renames it to `Get_extra_context_params()`
  - ANT runtime `Prepare_context()` now optionally merges
    `Get_extra_context_params()` with the program's main `Get_context_params()`
    before key generation
  - zero-rotate fast path moved into poly2c-generated `Rotate` helper code
    instead of Python source rewriting
  - the cleaner rotate-key fix is in compiler IR metadata / analysis:
    - `bindings/src/air_builder_bindings.cpp` now attaches `nn::core::ATTR::RNUM`
      to CKKS rotate nodes and preserves it when rewriting `mul_mono`
    - `fhe-cmplr/include/fhe/core/ctx_param_ana.h` now analyzes rotate keys for
      `rotate`, `mul_mono`, and `conjugate`
    - CKKS-only traced bootstrap flow now runs `CTX_PARAM_ANA`
  - the temporary IR2C-side rotate-key workaround was removed after the cleaner
    metadata/analyzer path worked
  - validation after the clean fix:
    - `python3 -m pytest -q ace_edsl/tests/test_resnet_bootstrap_utils.py`
      passes in `ace-compiler-dev`
    - generated bootstrap `Get_context_params()` now emits `42` bootstrap rotate
      keys directly
    - integrated `ACE_BOOTSTRAP_CT_ENCODE=1 ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
      passes with first bootstrap about `65.243s`
    3. reduce fully expanded straight-line code into more structured stage code
  - Phase 1 is now implemented:
    - added `encode_cache` attr support in lowering/codegen
    - primitive diagonal complex plaintext encodes now lower to lazy static cached plaintexts plus `Copy_plain(...)`
    - added rtlib `Copy_plain(...)` to deep-copy cached plaintexts into generated local temporaries safely
  - during Phase 1 validation, found a separate demo/shared-lib correctness issue:
    - primitive demo `Raise_mod`/level planning had an off-by-one mismatch against the runtime Q chain
    - fixed by making the demo path use:
      - `raise_mod(_bootstrap_mul_level())`
      - `level_0 = mul_level` in primitive collapsed-FFT planning
  - after the Phase 1 fixes:
    - targeted bootstrap tests passed again:
      - `test_python_and_c_api_results_match`
      - `test_z_inline_and_rtlib_results_match`
    - first-bootstrap resnet probe via `ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh` still succeeded
    - first bootstrap time improved from about `203.140s` to about `188.650s`
    - that is about a `14.49s` / `7.1%` improvement on the first bootstrap
  - follow-up ownership optimization:
    - replaced deep `Copy_plain(...)` with borrowed/static plaintext aliasing in generated CKKS C
    - patched the poly `MFREE_PASS` so cache-backed plaintext temps are not freed
    - generated C now has:
      - `Copy_plain(`: `0`
      - direct `_pre_plain_*` alias assignments
      - no `Free_data(&_pgen_tmp_..._poly)` for cache-backed plaintext temps
    - correctness checks still pass
    - but first-bootstrap resnet timing was about `192.475s`, so there was no additional meaningful speedup
    - likely reason: many encoded diagonal plaintext nodes are still unique and only used once per bootstrap body
  - profiling follow-up:
    - `perf` is installed but still unusable in the container because perf events are blocked
    - used `gprofng` instead
    - full first-run sampled profile is dominated by setup:
      - `Generate_rot_maps._omp_fn.0`
      - `Generate_rot_key`
      - `Generate_switching_key`
    - hottest sampled functions in that profile include:
      - `Forward_transform` (~35.6% exclusive)
      - `Mul_poly` (~22.2% inclusive)
      - `Sub_poly` (~20.1% inclusive)
      - `Add_poly` (~4.75% exclusive)
      - `Transform_values_to_dcrt.isra.0` (~4.38%)
      - `Encode_impl` (~9.78% inclusive)
      - `Sample_uniform` (~16.9% inclusive)
      - `blake2b_compress` (~11.5% exclusive)
    - attempted a signal-gated `gprofng` run to isolate only the first bootstrap window, but the resulting experiment had zero attributed CPU time
    - current profiling conclusion:
      - `Copy_plain` is not the main hotspot anymore
      - the real costs are first-run key generation plus NTT/polynomial kernels and the fully expanded bootstrap structure
  - Phase 2 optimization:
    - implemented stage-level grouping by rotation index in collapsed-FFT transforms
    - for duplicated rotations, diagonals are now summed before encoding so one unique rotation only pays one rotate/encode/mul chain
    - codegen impact on the `65536` bootstrap body:
      - raw AIR:
        - `CKKS.rotate`: `372 -> 310`
        - `CKKS.mul`: `510 -> 448`
        - `CKKS.add`: `546 -> 484`
      - generated C:
        - `Rotate(`: `374 -> 240`
        - `Encode_dcmplx_ext(`: `378 -> 244`
        - `Init_ciph_up_scale_plain(`: `474 -> 340`
        - `Init_ciph_down_scale(`: `508 -> 374`
        - `Init_ciph_same_scale(`: `538 -> 404`
      - body size: about `334.7 MB -> 321.3 MB`
      - line count: about `3.13M -> 2.62M`
    - correctness checks still pass
    - first-bootstrap resnet timing improved from about `192.475s` to about `159.926s`
    - that is about a `32.55s` / `16.9%` improvement relative to the previous borrowed-cache baseline
  - deeper phase-2 follow-up:
    - checked for cross-stage duplicate diagonal plaintexts after grouping and found none
    - added a regression test to lock that in
    - switched the collapsed-FFT stage evaluation to a true rtlib-style baby-step/giant-step form
    - after BSGS:
      - raw AIR `CKKS.rotate`: `372 -> 112`
      - generated C `Rotate(`: `374 -> 92`
      - generated C `Encode_dcmplx_ext(`: `378 -> 282`
      - bootstrap correctness checks still pass
      - first-bootstrap resnet timing dropped further to about `102.728s`
    - compared with the earlier borrowed-cache baseline (`192.475s`), the current first-bootstrap path is about `46.6%` faster
    - compared the generated DSL bootstrap rotation set directly against the
      rtlib `Find_rot_indices(...)` formula for the same `32768`/`level_budget=3`
      case
    - result after BSGS:
      - generated unique bootstrap rotations: `53`
      - rtlib theoretical bootstrap rotations: `53`
      - extra rotations: `0`
      - missing rotations: `0`
    - conclusion:
      - there is no remaining bootstrap-rotation-key excess to trim at this layer
      - remaining first-run setup cost now comes from unavoidable bootstrap
        rotations, model rotations, and key generation cost itself
  - runtime setup follow-up:
    - ANT `Prepare_context()` was still calling `Bootstrap_precom(default_slots)`
      even for the DSL-integrated resnet binary
    - added env guard `RTLIB_DISABLE_BOOTSTRAP_PRECOM=1`
    - wired that into `a_dsl_bts.sh` and `verify_resnet_first.sh` for the DSL path
    - measured first-bootstrap result on the current BSGS path:
      - before skip: about `102.728s`
      - after skip: about `102.099s`
    - conclusion:
      - only about `0.63s` improvement
      - so unused rtlib bootstrap precompute was not the dominant remaining cost
  - attempted the next profiler-aligned optimization:
    - emit cacheable complex bootstrap diagonals as `DE_PLAINTEXT` data-file
      entries and load them via `Pt_from_msg(...)` instead of runtime
      `Encode_dcmplx_ext(...)`
  - this path is blocked for the `65536` resnet case:
    - `generate-demo` segfaulted during offline plaintext generation
    - standalone tests showed the same thing:
      - `Encode_dcmplx_ext(...)` at `65536` segfaults in encode-only context
      - `Encode_dcmplx_ext(...)` at `65536` also segfaults in a full ANT
        runtime-context standalone test
      - the same standalone encode path works at `16384`
    - gdb backtrace goes through:
      - `Forward_transform`
      - `Ftt_fwd`
      - `Conv_poly2ntt_inplace`
      - `Encode_impl`
      - `Encode_ext_at_level`
  - conclusion:
    - the offline-plaintext direction is still profiler-aligned, but currently
      blocked by a runtime-side ANT encoder bug at `65536`
    - default path is kept on the working BSGS runtime-encode route by leaving
      `ACE_BOOTSTRAP_CT_ENCODE` disabled by default
- Session-level skills currently available to Codex in this environment:
  - `imagegen`
  - `openai-docs`
  - `plugin-creator`
  - `skill-creator`
  - `skill-installer`
- Critical project constraint learned on March 29, 2026:
  - lowering the primitive/bootstrap DSL path to `CKKS.bootstrap` / `Eval_bootstrap_ciph(...)` is not an acceptable end state; it is considered cheating for this DSL work
  - matching rtlib by delegating to rtlib bootstrap does not count as implementing the decomposition DSL
  - the required invariant going forward:
    - `ACE_BOOTSTRAP_IMPL=primitive` must not emit `CKKS.bootstrap` in raw AIR
    - the final generated C for the real decomposition path must not just call `Eval_bootstrap_ciph(...)`
  - if that invariant is violated, tests or resnet correctness results do not validate the DSL implementation goal; they only validate the wrapper path


- 2026-03-30 follow-up on ct_encode bootstrap integration:
  - fixed the real compiler/runtime ownership issue for offline bootstrap plaintexts
    at codegen/runtime level instead of relying on script-side free stripping
  - root cause chain:
    - invalid frees of borrowed plaintext temps loaded from `Pt_from_msg(...)`
    - a regression reintroduced `Copy_plain(...)` and removed mfree suppression
    - the correct compiler-side fix was to stop freeing cached encode temps again
  - ct_encode integration debugging found multiple issues and fixes:
    - compiler-side encode context / `num_p` alignment fixes for `Encode_dcmplx_ext`
    - `Max_plain_buffer_length()` had to account for extended plaintexts
    - CRT transform/reconstruct had a real bug: it used full active `P` count instead
      of the polynomial's actual `p_cnt`; fixed in `crt.c` / `rns_poly.c`
    - resnet integration could not reuse global `Pt_mgr` safely for bootstrap
      plaintexts; added a dedicated bootstrap plaintext loader path in the DSL shim
  - important ownership/runtime fixes:
    - `bootstrap_full.c` no longer emits invalid `Free_data(&_pgen_tmp_..._poly)`
      for borrowed bootstrap plaintext temps
    - custom bootstrap plaintext loader now keeps all bootstrap plaintext entries
      resident for the call so borrowed temps are not overwritten early
  - cleanup pass:
    - removed debug instrumentation again
    - moved duplicated DSL shim emission out of shell scripts and into
      `ace_edsl/examples/resnet_bootstrap_utils.py` (`emit-shim`)
    - `a_dsl_bts.sh` and `verify_resnet_first.sh` now call the helper instead of
      embedding duplicated large C blocks
  - validated sequentially in `ace-compiler-dev`:
    - `ACE_BOOTSTRAP_CT_ENCODE=1 python3 -m pytest -q tests/test_bootstrap_full.py -k "python_and_c_api_results_match or z_inline_and_rtlib_results_match"`
      -> `2 passed`
    - `ACE_BOOTSTRAP_CT_ENCODE=1 ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
      -> succeeds
      -> `out_level=15`, `out_sfdeg=1`
      -> first-bootstrap elapsed about `92s`
  - note: do not run the bootstrap pytest path and `a_dsl_bts.sh` concurrently,
    because they both regenerate shared output files under `ace_edsl/examples/output/`


- cleanup after ct_encode bootstrap fix:
  - removed legacy `evalmod` mode from `ace_edsl/examples/bootstrap_full.py`
    and simplified the example to the two maintained modes:
    - `primitive`
    - `rtlib`
  - removed obsolete evalmod-only expectations from
    `ace_edsl/tests/bootstrap_test_main.c`
  - moved duplicated DSL shim C emission out of shell scripts and into
    `ace_edsl/examples/resnet_bootstrap_utils.py` via `emit-shim`
  - `a_dsl_bts.sh` and `verify_resnet_first.sh` now reuse the shared helper
    instead of embedding large duplicated C blocks
  - ct_encode ownership/runtime final shape:
    - compiler/codegen no longer emits invalid frees for borrowed bootstrap
      plaintext temps
    - private bootstrap plaintext loader caches all bootstrap plaintext entries
      for the call, avoiding overwrite of borrowed entries
    - dedicated bootstrap plaintext loader is used only for the resnet shim path
      and does not switch the global resnet `Pt_mgr`
  - validated sequentially in `ace-compiler-dev` after cleanup:
    - `python3 -m pytest -q tests/test_bootstrap_full.py -k "python_and_c_api_results_match or z_inline_and_rtlib_results_match"`
      -> `2 passed`
    - `ACE_BOOTSTRAP_CT_ENCODE=1 python3 -m pytest -q tests/test_bootstrap_full.py -k "python_and_c_api_results_match or z_inline_and_rtlib_results_match"`
      -> `2 passed`
    - `ACE_BOOTSTRAP_CT_ENCODE=1 ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
      -> succeeds
      -> first bootstrap remains correct at about `92s`


- 2026-03-30 follow-up on rotate-lowering optimization direction:
  - reverted a temporary `CKKS2POLY::Expand_rotate` experiment because that
    helper path is shared with the rtlib bootstrap baseline and should not be
    changed for DSL-only optimization work
  - inspected the saved known-good first-bootstrap `gprofng` capture for the
    optimized DSL path:
    - `tmp_profile_dsl_first_bsgs.er`
  - dominant first-call hotspot remains shared setup, not DSL helper bodies:
    - `Generate_rot_maps._omp_fn.0`
    - `Generate_rot_key`
    - `Generate_switching_key`
  - within that capture, the generated DSL helper bodies are comparatively
    small:
    - `dsl_bts_Rotate`
    - `dsl_bts_Relinearize`
    - `Encode_dcmplx_ext`
    - `Rescale`
  - conclusion:
    - do not optimize `Expand_rotate` for the current first-bootstrap metric
    - the next useful DSL-only target requires a warmed second-call profile that
      excludes first-run key generation noise
- 2026-03-30 warm standalone bootstrap profiling:
  - extended `ace_edsl/tests/bootstrap_test_main.c` with env-controlled warm
    runs:
    - `ACE_BOOTSTRAP_WARMUP_RUNS`
    - `ACE_BOOTSTRAP_MEASURE_RUNS`
    - `ACE_BOOTSTRAP_SKIP_VALIDATE`
  - built a warm standalone harness in `ace-compiler-dev` against the preserved
    runtime-encode primitive bootstrap source
    `ace_edsl/examples/output/bootstrap_full_link.c.inlev1.c`
  - observed steady-state timings:
    - warmup call about `25.0s`
    - subsequent measured calls about `24.2s` to `25.5s`
  - captured `gprofng` experiment:
    - `/app/tmp_bootstrap_warm/warm_bootstrap.er`
  - warm profile still shows heavy NTT/poly kernels:
    - `Forward_transform`
    - `Mul_poly`
    - `Sub_poly`
    - `Add_poly`
  - within `bootstrap_full`, the largest steady-state helper cost is still
    `Rotate`, followed by `Rescale`, `Encode_dcmplx_ext`, and `Relinearize`
  - implication:
    - do not optimize the shared rotate helper implementation
    - the next DSL-only optimization target should be reducing the number of
      emitted `Rotate` / `Rescale` / `Relinearize` calls in the decomposition
      schedule, with `Rotate` pressure the first thing to investigate
- 2026-03-30 collapsed-FFT BSGS retune:
  - updated `docs/opt.md` to reflect the current known-good performance picture
    and the warmed-profile findings
  - changed the DSL collapsed-FFT stage planner in
    `ace_edsl/edsl/core/bootstrap_decomposition.py` to choose `g` by
    minimizing emitted high-level rotation count instead of copying the rtlib
    `Rotate_precomp(...)` heuristic directly
  - for the active `65536` / `32768` full-packed case, each stage now uses:
    - before: `g=16, b=4`
    - after: `g=8, b=8`
  - preliminary effect:
    - collapsed-FFT stage rotation proxy `108 -> 84`
    - traced raw AIR `CKKS.rotate: 112 -> 88`
    - locally generated C `Rotate(: 114 -> 90`
  - initial bring-up exposed a poly2c bug:
    - `Handle_dot_prod()` emitted malformed `Dot_prod(...)` calls for store
      destinations because it only handled preg results correctly
    - fixed in `fhe-cmplr/include/fhe/poly/ir2c_handler.h`
  - after that fix:
    - `generate-demo` completed successfully again in `ace-compiler-dev`
    - integrated `ct_encode` probe
      `ACE_BOOTSTRAP_CT_ENCODE=1 ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
      succeeded
    - output contract remained correct:
      - `out_level=15`
      - `out_sfdeg=1`
      - `out_slots=32768`
    - first-bootstrap elapsed improved from the prior validated `ct_encode`
      baseline:
      - before: `92.088s`
      - after: `85.313s`
      - delta: `6.775s` / `7.36%`
- 2026-03-30 fresh post-retune profile and follow-up:
  - captured a new current-path `gprofng` experiment after the BSGS retune/fix:
    - `/app/tmp_profile_dsl_first_current.er`
  - first-call process time was still dominated by shared setup:
    - `Generate_rot_maps._omp_fn.0`
    - `Generate_rot_key`
    - `Generate_switching_key`
  - inside `dsl_bootstrap_full`, helper ordering had shifted to:
    - `Rescale`
    - `dsl_bts_Rotate`
    - `dsl_bts_Relinearize`
  - current generated artifact counts matched that diagnosis:
    - `CKKS.rescale`: `508`
    - `Rotate(`: `90`
    - `Pt_from_msg(`: `378`
  - checked collapsed-FFT term structure:
    - `189` encode terms and `189` decode terms
    - zero all-zero terms
    - only `3` trivial `+/-1` diagonal terms per direction
  - implication at that point:
    - trivial diagonal skipping alone would not move the needle
    - the next DSL-only target should be reducing scale-management and
      plaintext-multiply pressure in `_apply_collapsed_fft_transform()`
- 2026-03-30 lazy-rescale transform accumulation:
  - added a DSL-only `skip_auto_rescale` attribute for selected `CKKS.mul`
    nodes via:
    - `fhe-cmplr/include/fhe/core/lower_ctx.h`
    - `bindings/src/air_builder_bindings.cpp`
    - `fhe-cmplr/ckks/include/scale_manager.h`
    - `fhe-cmplr/ckks/src/scale_manager.cxx`
  - changed
    `ace_edsl/edsl/core/bootstrap_decomposition.py::_apply_collapsed_fft_transform`
    to accumulate ct-plain products across each inner sum and issue one explicit
    `rescale()` afterward
  - integrated validation:
    - `ACE_BOOTSTRAP_CT_ENCODE=1 ACE_STOP_AFTER_FIRST_BTS=1 bash a_dsl_bts.sh`
      succeeded
    - output contract remained correct:
      - `out_level=15`
      - `out_sfdeg=1`
      - `out_slots=32768`
    - first-bootstrap elapsed improved:
      - before: `85.313s`
      - after: `68.232s`
      - delta: `17.081s` / `20.02%`
  - generated artifact effect:
    - `CKKS.rescale: 508 -> 178`
    - generated C `Rescale(: 1016 -> 356`
    - generated C `Init_ciph_down_scale(: 508 -> 178`
  - fresh profile after that change:
    - `/app/tmp_profile_dsl_first_current2.er`
    - one-call bootstrap in that run: `66.771s`
    - inside `dsl_bootstrap_full`, helper ordering is now:
      - `dsl_bts_Rotate` first
      - `Rescale` second
      - `dsl_bts_Relinearize` third
  - implication:
    - after the scale-management win, the next remaining DSL-only target
      shifted back to `dsl_bts_Rotate`
- 2026-03-30 failed rotate-precomp follow-up:
  - tried a DSL-only alternate rotate lowering/helper path for bootstrap
    transform rotates, aiming to use a separate `Precomp + Dot_prod` route only
    for annotated bootstrap rotations
  - result was a severe regression:
    - raw AIR `CKKS.rotate: 88 -> 418`
    - integrated one-call `ct_encode` bootstrap regressed to `180.067s`
  - reverted that entire attempt immediately
  - rebuilt compiler/bindings and revalidated the restored path:
    - integrated one-call `ct_encode` bootstrap returned to a fast working
      state at `66.045s` on the latest rerun
  - takeaway:
    - a useful rotate optimization likely needs an explicit bootstrap-local
      hoisting/cache design; a naive alternate rotate lowering can easily
      increase IR/helper pressure and lose badly
