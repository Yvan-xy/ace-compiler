# EvalMod experiment evidence, 2026-09-08

Historical results from the 16-vCPU KVM server, using the fixed degree-44/K=28
profile. See the [results report](../../../../../docs/cpu_evalmod_parallel_results.md)
for interpretation, limitations and build details. No new OpenFHE timing was
collected in this experiment.

- `bts0`, `bts1`, `bts2` correspond to Legacy, A and B.
- `run.txt`: 16-thread primary measurements; `single.txt`: one-thread control.
- `phase.txt`: separate stage instrumentation, also at 16 threads.
- `coverage.txt`: separate diagnostic calls for A/B, excluded from primary timings.
- `build.json`: original generation settings and file hashes. Absolute paths are
  historical provenance, not portable build inputs. `ace_revision` records the
  base commit before these experimental changes were committed; it does not
  identify the complete measured source tree by itself. Recorded file hashes
  identify the captured build inputs. Generated binaries/constants are omitted.
- `final-results.json`: all 40 validated calls, timed samples and worst error;
  `summary16.json`: primary latency/CPU/RSS medians; `phases.json`: stage medians.
- `unit-regression.txt` and `python-tests.txt`: original focused test records.

Logs and JSON files were copied verbatim from `tmp_evalmod_full/` when preparing
the commits. The local overlay build and batch scripts are not required for a
fresh run: use the benchmark [README](../../README.md#continuing-on-another-server).
The scripts' local install prefixes, ONNX stub and compiler outputs are not
included in this archive.
