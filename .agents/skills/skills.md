- Bootstrap profiling / integration notes:
  - For `dsl-bts`, compare against rtlib using both `MAIN_GRAPH` and the sum of
    explicit bootstrap-call timings. The non-bootstrap residual can otherwise
    hide where the real gap is.
  - `ACE_BOOTSTRAP_STAGE_PROBE=1` is useful to separate:
    - `coeff_to_slots`
    - `split`
    - `dual_evalmod`
    - `recombine`
    - `slots_to_coeffs`
    - `post_scale`
  - When adding runtime-specialized behavior to the integrated DSL bootstrap
    path, prefer rewriting the emitted bootstrap body/shim in
    `ace_edsl/examples/resnet_bootstrap_utils.py` if the deeper compiler stack
    still assumes compile-time constants.
  - For target-aware bootstrap raise levels in the integrated path, use the
    main program `Get_context_params()` as the source of the active Q-chain;
    `Get_extra_context_params()` is not sufficient for that decision.
  - In `_apply_collapsed_fft_transform()`, the safer and faster rescale
    structure is one `rescale()` per collapsed-FFT stage, not one per
    baby-step inner sum.
  - The current DSL collapsed-FFT planner differences are not the main
    remaining cause of the rtlib gap:
    - forcing rtlib's default BSGS (`g=16, b=4` at `slots=32768`) regressed the
      real first-bootstrap probe from about `56.14s` to about `62.43s`
    - disabling the current stage compaction also regressed the same probe to
      about `63.02s`
    - so the current DSL `g=8, b=8` retune and limited compaction are helping
      the heavier generic lowering; the remaining gap is in rotate/decomp
      implementation structure, not these planner choices
  - Nested OpenMP inside the integrated resnet `dsl-bts` harness is ineffective
    for `EvalMod` branch parallelism because `Run_main_graph()` already runs
    inside an outer `#pragma omp parallel for`, while libgomp defaults to
    `OMP_NESTED=FALSE` and `OMP_MAX_ACTIVE_LEVELS=1`.
  - Direct runtime helper substitution is not automatically a win:
    - replacing the generated rotate helper path with runtime `Rotate_ciph`
      and the generated relin helper with runtime `Relin` regressed the real
      first-bootstrap probe from about `56.14s` to about `58.09s`
    - isolating runtime `Relin` alone still regressed the same probe to about
      `57.44s`
    - conclusion: the remaining gap is not fixed by swapping individual helper
      implementations one-for-one; the missing optimization is stage-level
      reuse across many rotates sharing one source ciphertext
  - A first-class grouped rotate batch is viable and useful:
    - represent it as one CKKS op returning an array of ciphertexts, with the
      rotation list carried in `nn::core::ATTR::RNUM`
    - use normal array indexing (`ILD`) on the result in the DSL
    - emit a generic runtime helper `Rotate_batch_ciph(...)` that shares one
      `Alloc_precomp(Get_c1(ciph))` across the batch and calls `Fast_rotate(...)`
    - in the collapsed-FFT transform, this reduced emitted rotate-helper call
      sites from `82` to `44` and improved the real first-bootstrap probe from
      about `56.14s` to about `53.33s`
  - For array-of-ciphertext results such as `rotate_batch`, `poly2c_mfree`
    must treat `st x = ild(array(batch, i))` as ownership transfer of the
    extracted element:
    - do not pre-mark the extracted ciphertext temp as freed
    - suppress only the backing array container free
    - otherwise the generated C body misses `Free_data(...)` calls for the
      extracted ciphertext temporaries and leaks memory
  - Multi-image `dsl-bts` runs need configurable image-level parallelism:
    - one DSL bootstrap image can already consume roughly `15 GB RSS` within
      about 20 seconds and roughly `30 GB RSS` within about 40 seconds
    - the old unconditional image-level `#pragma omp parallel for` in
      `resnet_cifar.main.inc` is therefore unsafe for `0..N` DSL bootstrap
      runs
    - use a generic runtime env control such as `ACE_IMAGE_PARALLELISM`
      instead of hardcoding launcher-only behavior
    - `a_dsl_bts.sh` should default multi-image runs to image parallelism `1`
      unless the user explicitly overrides it
  - When adding a new CKKS opcode, always audit every opcode-indexed table,
    not just the enum and lowering paths.
    A confirmed regression happened after adding `CKKS.ROTATE_BATCH`:
    - `fhe-cmplr/include/fhe/ckks/opcode_def.inc` gained `ROTATE_BATCH`
    - `fhe-cmplr/ckks/src/ckks_cost_model.cxx` still indexed latency tables
      by `opc.Operator()` without a matching new slot
    - result: `fhe_cmplr ... -CKKS:...:sbm ...` crashed in `RESBM`
      (`Operation_cost -> MIN_CUT_REGION::Scc_cut_op_cost`)
    - minimal safe fix: add a `ROTATE_BATCH` slot in both CKKS cost vectors,
      reusing `ROTATE` latency until a real batch cost model exists
  - If `fhe_cmplr` crashes only with `:sbm`, check `RESBM` and cost-model
    alignment first before blaming ONNX import. A quick discriminator:
    - `-CKKS:...:sbm` crashes
    - `-CKKS:...` succeeds
    - `-CKKS:...:mbc=2` succeeds
