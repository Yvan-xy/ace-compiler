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
- For future work, treat the Python DSL and shared bindings as one feature surface; many meaningful changes cross that seam.
