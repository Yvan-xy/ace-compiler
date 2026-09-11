# Preliminary DSL CPU / OpenFHE bootstrap comparison

Measured results: [RESULTS.md](RESULTS.md). Detailed stage timings, `perf`
hotspots, thread-scaling and allocator diagnostics: [PROFILE.md](PROFILE.md).

This standalone benchmark compares the generated primitive DSL bootstrap on ANT
with OpenFHE v1.5.1's conventional `EvalBootstrap`. It does not run ResNet or GPU
code. Source generation, constant encoding/loading, context/key setup, encryption,
input copying, decryption, and validation are outside the measured warm calls.
Output rescaling/modulus dropping is inside the timer.

The fixed profile is N=65536, 32768 real slots, a weight-192 ternary secret,
31 Q primes near `[2^60, 2^56, ..., 2^56]`, 11 approximately 60-bit P primes,
three hybrid decomposition digits, transform budgets `{3,3}`, BSGS dimension 16,
and scale `2^56`. Both use the identical degree-44, K=28 coefficient table and
three double-angle iterations. The runner checks the table against the pinned
OpenFHE source before building.

Both inputs have two Q primes and scale degree 1. Both bootstrap paths raise to
31 Q primes. Outputs are normalized to 14 Q primes and scale degree 1. The input
fills every slot with `0.0625 * ((i % 17) - 8)`; all slots, including imaginary
error, must pass maximum absolute error <= 0.02. Actual error is reported.
OpenFHE uses `FIXEDMANUAL` and an explicit correction factor of 4, matching the
DSL's q0/scale ratio of 16 without OpenFHE's extra default attenuation.

Run each executable sequentially with one unmeasured warm-up followed by three
measured calls on copies of the same original depleted ciphertext. The default
is 16 OpenMP threads, binding to cores, with dynamic teams and nested parallelism
disabled. Each library retains its own scheduling and primitive implementation.
The summary uses medians and rejects missing, failed, or mismatched samples.

## Build and run

Dependencies: GCC/G++, CMake, GMP, Python with NumPy and `typing_extensions`,
working ACE Python bindings, and an installed ANT/compiler prefix. Build OpenFHE
with the same CPU compiler, Release, 64-bit native integers, OpenMP enabled, and
without optional hardware backends. For example, from the repository root:

```sh
git clone --depth 1 --branch v1.5.1 https://github.com/openfheorg/openfhe-development.git tmp_openfhe_cpu_compare/openfhe-src
cmake -S tmp_openfhe_cpu_compare/openfhe-src -B tmp_openfhe_cpu_compare/openfhe-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/tmp_openfhe_cpu_compare/openfhe-install" \
  -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
  -DWITH_OPENMP=ON -DWITH_NATIVEOPT=OFF
cmake --build tmp_openfhe_cpu_compare/openfhe-build -j 8
cmake --install tmp_openfhe_cpu_compare/openfhe-build

python3 ace_edsl/benchmarks/cpu_bootstrap/run.py \
  --ace-prefix /path/to/ace/install \
  --bindings-dir /path/containing/ace_bindings \
  --openfhe-prefix tmp_openfhe_cpu_compare/openfhe-install \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src
```

The pinned OpenFHE commit is `1306d14f8c26bb6150d3e6ad54f28dfe1007689e`.
Use `--threads`, `--repetitions`, or `--work-dir` to change execution settings.
`--build-only` prepares the artifacts; `--skip-build` reruns them after verifying
their recorded hashes. Budget roughly 20 GiB RAM per executable and several GiB
of disk for the generated plaintext constants, in addition to build artifacts.

The CPU NTT experiment adds `--decomp-ntt-threads`: 0 keeps the legacy two-branch
mode, 1 selects the new serial entry, and values above 1 set an internal NTT
limb budget while disabling the demo's outer EvalMod sections. It requires an
updated compiler, binding and ANT runtime. `--ntt-probe` observes actual workers
for correctness smoke runs only; omit it when measuring performance. See the
[step-two report](../../../docs/cpu_rns_parallel_scheduling_step2.md).

For the formal three-policy comparison, `compare_ntt.py` reuses the step-two
generated sources/constants, checks their recorded hashes, and recompiles all
executables without the NTT observer. It first measures legacy, serial and limb
policies at 16 and 1 threads, then runs separate stage-timing diagnostics at 16
threads. Each group has one warm-up and three measured invocations:

```sh
python3 ace_edsl/benchmarks/cpu_bootstrap/compare_ntt.py \
  --ace-prefix tmp_cpu_ntt_step2/install --artifacts tmp_cpu_ntt_step2
```

The default output directory is `tmp_cpu_ntt_step3/`, containing `build.json`,
`settings.json`, `cpu.txt`, per-group logs and `summary.json`. The summary is only
written after all groups validate; `progress.json` contains partial results.
Primary timings exclude diagnostic hooks and perf sampling. Resource snapshots
are taken outside the timer; reported RSS is the process peak, including setup
and cached keys/constants. See `--help` for output and repetition controls.
The [formal result](../../../docs/cpu_rns_parallel_scheduling_step3.md) retains
the legacy default: the current limb experiment improves on its serial control
but does not outperform the existing two-branch schedule.

A separate full EvalMod experiment is available through `--evalmod-schedule`:
0 keeps Legacy, 1 runs sequential branches with operator tasks (A), and 2 runs
both branches with operator tasks sharing one team (B). It requires rebuilt
compiler opcode metadata, bindings and the new runtime. It conflicts with
`--decomp-ntt-threads` and `--ntt-probe`. `ACE_EVALMOD_DIAGNOSTICS=1` reports region
coverage for correctness runs; keep it unset/0 for timings. See the
[full EvalMod results](../../../docs/cpu_evalmod_parallel_results.md) for measured
latency improvements and the higher CPU cost. Default remains 0.

Results and build logs are under `tmp_openfhe_cpu_compare/benchmark/` by default:
`summary.json`, `dsl_cpu.log`, `openfhe.log`, `build.json`, `run.json`, and `cpu.txt`.
For result-validation unit tests:

```sh
python3 -m unittest discover -s ace_edsl/benchmarks/cpu_bootstrap -p test_results.py -v
```

## Continuing on another server

The branch is `explore/cpu-rns-parallel-scheduling`. Source, tests, plans and
reports are versioned. The latest EvalMod measurements and coverage logs are
preserved in [results/evalmod-2026-09-08](results/evalmod-2026-09-08/README.md).
The `tmp_*` directories, installed libraries, Python environments, OpenFHE clone,
generated C and large encoded constants are local artifacts; regenerate them.

1. Build/install this checkout's ACE compiler and ANT runtime with OpenMP enabled
   (`BUILD_WITH_OPENMP=ON`), using the repository's
   [build instructions](../../../fhe-cmplr/doc/BUILD.md) and your server's
   dependency configuration. Reconfigure existing CMake builds so they discover
   `evalmod_exec.c`. Rebuild `FHEckks`, `FHEpoly`, `FHErt_ant`,
   `FHErt_ant_encode` and `FHErt_common`, including installed runtime headers.
2. Rebuild the [Python bindings](../../../bindings/README.md) against that same
   install prefix and your chosen Python interpreter. In particular, the CKKS
   opcode registry (`ckks_opcode.cxx`), POLY registry and `air_builder` must agree
   on the new region attribute. Use the parent of the resulting `ace_bindings`
   package as `--bindings-dir`. The original experiment used Python 3.12 and an
   isolated incremental build with an ONNX loader stub; that local stub is not
   part of this change, and a full repository/ONNX build was not validated here.
3. Build the pinned OpenFHE release as above. With NumPy, SciPy,
   `typing_extensions` and the updated bindings available, generate fresh
   artifacts for each policy. For an EvalMod scheduling comparison only, run
   the following from the repository root (replace the two installation paths):

```sh
for mode in 0 1 2; do
  python3 ace_edsl/benchmarks/cpu_bootstrap/run.py \
    --ace-prefix /path/to/updated/ace/install \
    --bindings-dir /path/containing/ace_bindings \
    --openfhe-prefix tmp_openfhe_cpu_compare/openfhe-install \
    --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
    --work-dir "tmp_evalmod_server/bts${mode}" \
    --evalmod-schedule "$mode" --build-only || break
  OMP_NUM_THREADS=16 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
    OMP_MAX_ACTIVE_LEVELS=1 ACE_EVALMOD_DIAGNOSTICS=0 \
    "tmp_evalmod_server/bts${mode}/ant_bench" 3 \
    > "tmp_evalmod_server/bts${mode}/run.log" 2>&1 || break
done
```

Each executable checks all slots and reports one warm-up plus three timed
samples. Keep allocator/preload settings identical across policies, and run
sequentially. For a fresh OpenFHE comparison, omit `--build-only` and let
`run.py` execute and validate both implementations. On the new machine, first
remeasure Legacy (0) and candidate B (2); previous server timings are historical
evidence. A (1) is the sequential-branch control. Default remains 0.

For independent stage timings, `profile.py --baseline tmp_evalmod_server/bts2
--work-dir tmp_evalmod_server/profile2 --ace-prefix /path/to/updated/ace/install
--openfhe-source tmp_openfhe_cpu_compare/openfhe-src --threads 16` instruments
copies of the generated C and OpenFHE source. Keep those timings separate from
the uninstrumented results. Do not use `--skip-build` with historical archived
`build.json` files: their absolute paths and hashes describe the original server.

## Implementation details and limits

- `ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL=31` describes the compiler's Q count, but
  `ACE_CT_ENCODE_DEPTH=30` describes the encoder's multiplicative depth. They
  must refer to the same 31-prime chain. Using 31 for both creates incompatible
  pre-encoded constants.
- The standalone DSL executable has no application weight file. Generated
  `Free_data` calls are changed to `Free_poly_data` so owned temporaries are
  released even without ANT's global application plaintext manager. Cached
  transform plaintexts remain available across warm calls.
- Prime counts, approximate sizes, scales, key weight, and bootstrap algorithm
  settings match. Prime values and encryption randomness are library-native.
  ANT uses triangular error sampling and balances the signs of its weight-192
  secret; OpenFHE uses its native Gaussian error sampler (sigma 3.19) and native
  sparse-secret sampler. This is a preliminary implementation comparison, not
  a claim of identical cryptographic distributions or security. Both contexts
  disable automatic security-level selection to preserve the fixed profile.
- Accuracy is measured on one deterministic full-packed input and one key per
  executable. Passing the shared tolerance does not establish equal precision
  or a bootstrap failure probability. No parameter sweep or tuning is performed.
