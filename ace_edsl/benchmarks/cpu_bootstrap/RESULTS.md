# CPU bootstrap preliminary result — 2026-09-06

On this machine, the DSL CPU bootstrap took **22.675 s** versus **17.983 s** for
OpenFHE v1.5.1: DSL latency was **26.1% higher** (OpenFHE/DSL time = 0.7931).

| Implementation | Measured samples (seconds) | Median (seconds) | Maximum all-slot absolute error |
| --- | --- | ---: | ---: |
| DSL CPU on ANT | 22.522519, 22.674619, 22.852099 | 22.674619 | 4.4661e-4 |
| OpenFHE v1.5.1 | 17.694539, 17.983117, 18.142078 | 17.983117 | 4.5938e-5 |

Each executable used one unmeasured warm-up, then three calls on fresh copies of
its original depleted input. All eight outputs, including warm-ups, passed the
32768-slot complex absolute-error check and the common 0.02 tolerance. Setup,
key generation, encryption, input copying, decryption, and validation were not
timed. Final output normalization was timed. Runs were sequential.

## Matched workload

- N=65536; 32768 real slots; fixed-weight ternary secret with weight 192.
- Degree-44 EvalMod, identical coefficient table, K=28, three double-angle steps.
- CoeffToSlot/SlotToCoeff budgets `{3,3}`; BSGS dimension 16.
- Full context and active raised chain: 31 Q primes, near 2^60 followed by thirty
  primes near 2^56; 11 P primes near 2^60; three hybrid decomposition digits.
- Input: two Q primes, scale degree 1, scale 2^56.
- Normalized output: 14 Q primes, scale degree 1, scale 2^56.
- OpenFHE `FIXEDMANUAL`, correction factor 4, one bootstrap iteration.
- 16 OpenMP threads, dynamic teams off, core binding, no nested parallelism.

Raw DSL output had 16 Q primes and scale degree 1. Raw OpenFHE output had 16 Q
primes and scale degree 2. Both were normalized inside the timing interval.
The compiler's context Q count was 31, while the offline encoder's
multiplicative-depth argument was 30; these describe the same chain.

## Environment and reproduction

The host is a KVM VM exposing 16 vCPUs identified as AMD Ryzen 9 5900X, with
approximately 62 GiB RAM. Benchmark C/C++ code was compiled with GCC/G++ 13.3,
`-O3 -DNDEBUG` and OpenMP. OpenFHE was built in Release with OpenMP enabled and
`WITH_NATIVEOPT=OFF`. The existing installed ANT runtime and ACE bindings were
reused; their hashes are recorded in `build.json`.

OpenFHE commit: `1306d14f8c26bb6150d3e6ad54f28dfe1007689e` (v1.5.1).
From the repository root, the command used after building was:

```sh
tmp_openfhe_cpu_compare/venv/bin/python ace_edsl/benchmarks/cpu_bootstrap/run.py \
  --ace-prefix tmp_cpu_bts_openfhe_sparse_k28/install-current \
  --bindings-dir tmp_cpu_bts_openfhe_sparse_k28 \
  --openfhe-prefix tmp_openfhe_cpu_compare/openfhe-install \
  --openfhe-source tmp_openfhe_cpu_compare/openfhe-src \
  --skip-build
```

Omit `--skip-build` to regenerate and rebuild. See [README.md](README.md) for
dependency preparation and the benchmark contract.

Raw records: `tmp_openfhe_cpu_compare/benchmark/{dsl_cpu.log,openfhe.log}`.
Summary: `tmp_openfhe_cpu_compare/benchmark/summary.json`.
Build hashes, generation settings, CPU details and OpenMP settings:
`tmp_openfhe_cpu_compare/benchmark/{build.json,cpu.txt,run.json}`.

Additional validation passed: 15 existing `test_resnet_bootstrap_utils` tests
and four benchmark result-validation tests (warm-up exclusion, mismatched output
state, missing samples, and invalid timing/error/scale).

This is one preliminary workload, not a general performance or equal-precision
claim. Each library retains its native prime generation and random samplers:
ANT uses triangular errors and sign-balanced weight-192 secrets; OpenFHE uses
Gaussian errors (sigma 3.19) and its native sparse-secret sampler. The common
algorithm settings and ciphertext resource budgets match; the cryptographic
distributions are not identical. Neither context claims an automatically
validated security level. The earlier one-Q-input, 29-active-Q DSL timings are
not included in this comparison.
