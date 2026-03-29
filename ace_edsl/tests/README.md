# Tests

This directory contains the `ace_edsl` regression tests and a few small native
helpers used by those tests.

The tests are not all at the same level. Some are fast unit-style checks on the
Python DSL and AIR generation, while others build generated C and run it
through the ANT runtime.

## Main Groups

### Bootstrap Tests

- [test_bootstrap_full.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_full.py)
  - End-to-end test for [bootstrap_full.py](/home/dyf/code/ace-compiler/ace_edsl/examples/bootstrap_full.py).
  - Checks that the primitive bootstrap demo:
    - traces successfully
    - emits AIR and C code
    - does not cheat by lowering back to `CKKS.bootstrap` in primitive mode
    - can be built and executed through the ANT runtime harness
  - Also contains the inline-vs-rtlib comparison path and the shared-library
    harness used to compare generated bootstrap output against references.

- [test_bootstrap_stage_ops.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_stage_ops.py)
  - Focused checks for the three first-class bootstrap stage ops:
    - `bootstrap_coeffs_to_slots`
    - `bootstrap_eval_mod`
    - `bootstrap_slots_to_coeffs`
  - Verifies both modes:
    - lowering to runtime stage calls
    - explicit primitive lowering to decomposed CKKS ops

- [test_resnet_bootstrap_utils.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_resnet_bootstrap_utils.py)
  - Regression tests for the resnet integration helper script code in
    [resnet_bootstrap_utils.py](/home/dyf/code/ace-compiler/ace_edsl/examples/resnet_bootstrap_utils.py).
  - Covers bugs we hit during bring-up:
    - wrong collapsed-FFT plaintext levels
    - missing rotation-key patching
    - missing zero-rotation fast path
    - body extraction / symbol renaming mistakes

### CKKS Arithmetic / Encoding Tests

- [test_cipher_cipher.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_cipher_cipher.py)
  - Ciphertext-ciphertext arithmetic coverage.

- [test_cipher_plain.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_cipher_plain.py)
  - Ciphertext-plaintext arithmetic coverage.

- [test_cipher_plain_array.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_cipher_plain_array.py)
  - Plain array constant handling and broadcast-style cases.

- [test_cipher_scalar.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_cipher_scalar.py)
  - Ciphertext-scalar interaction cases.

- [test_ckks_complex_encode.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_ckks_complex_encode.py)
  - Regression coverage for complex plaintext encoding / lowering.

- [test_ckks_extended_rewrite.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_ckks_extended_rewrite.py)
  - Checks the CKKS extended-op rewrite path.

- [test_scalar_param.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_scalar_param.py)
  - Scalar parameter plumbing through DSL tracing and lowering.

### Control Flow / Lowering Tests

- [test_ckks_loop.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_ckks_loop.py)
  - CKKS loop tracing / lowering checks.

- [test_sihe_if_lowering.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_sihe_if_lowering.py)
  - `if` lowering tests for the SIHE path.

- [test_vector_if_lowering.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_vector_if_lowering.py)
  - Vector-domain conditional lowering checks.

- [test_vector_lowering.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_vector_lowering.py)
  - General vector-domain lowering coverage.

- [test_selective_lowering.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_selective_lowering.py)
  - Checks mixed lowering / skip-lowering behavior.

- [test_pipeline.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_pipeline.py)
  - General pipeline orchestration tests.

### DSL / Kernel Front-End Tests

- [test_domain_kernels.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_domain_kernels.py)
  - Decorator dispatch and domain-kernel behavior.

- [test_simple_kernel.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_simple_kernel.py)
  - Small sanity checks for basic kernel tracing and codegen.

### NN / Example Lowering Tests

- [test_nn_simple_lowering.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_nn_simple_lowering.py)
  - Simple NN lowering cases.

- [test_nn_conv_c_codegen.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_nn_conv_c_codegen.py)
  - Convolution lowering and generated C checks.

## Native Helper Files

- [bootstrap_test_main.c](/home/dyf/code/ace-compiler/ace_edsl/tests/bootstrap_test_main.c)
  - Small harness main used by bootstrap C-code execution tests.

- [ant_bootstrap_smoke.cxx](/home/dyf/code/ace-compiler/ace_edsl/tests/ant_bootstrap_smoke.cxx)
  - Native smoke helper that exercises the ANT runtime bootstrap path.

- [ckks_e2e_utils.py](/home/dyf/code/ace-compiler/ace_edsl/tests/ckks_e2e_utils.py)
  - Shared utilities for CKKS end-to-end test setup.

- [rtlib_coeff_collapse_dump.c](/home/dyf/code/ace-compiler/ace_edsl/tests/rtlib_coeff_collapse_dump.c)
  - Native helper used by the bootstrap precompute comparison tooling.

## Which Tests Matter Most For Bootstrap Work

If you are changing the bootstrap DSL, the highest-signal tests are:

1. [test_bootstrap_stage_ops.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_stage_ops.py)
2. [test_bootstrap_full.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_bootstrap_full.py)
3. [test_resnet_bootstrap_utils.py](/home/dyf/code/ace-compiler/ace_edsl/tests/test_resnet_bootstrap_utils.py)

Those are the tests most likely to catch regressions we already hit during
bootstrap bring-up:

- primitive path accidentally lowering back to runtime bootstrap
- wrong plaintext levels in collapsed FFT stages
- broken resnet integration glue
- missing rotation-key patching
- bad symbol/body rewriting for linking generated bootstrap into resnet
