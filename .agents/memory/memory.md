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
