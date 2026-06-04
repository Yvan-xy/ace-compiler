# Agents Rules

This is the correct workspace for the FHE compiler and EDSL bootstrap work:
`/home/dyf/code/ace-compiler`.

For the subagent team, role definitions, and performance workflow, read
[team.md](./team.md).

## Highest-Priority Rules

- All experiments must run inside the Docker container `ace-compiler-dev`.
  Experiments include builds, tests, benchmarks, profiling, generated-code
  runs, and runtime validation. Host-side source inspection is allowed.
- Hardcoding is forbidden for DSL bootstrap performance work. Do not bake FHE
  parameters, runtime metadata, dataset paths, or dispatch choices into
  implementation code to make a benchmark pass if the rtlib-bts is not doing the same thing.
- The target is primitive DSL bootstrap performance at parity with the RTLIB
  bootstrap baseline (`rtlib-bts`) or faster, with functional equivalence.
- Do not cheat: primitive mode must not lower back to `CKKS.bootstrap`, and the
  primitive implementation must not call `Eval_bootstrap_ciph(...)`.
- Treat `ACE_BOOTSTRAP_STAGE_PROBE=1` as diagnostic only. Final performance
  claims must compare against `rtlib-bts` and account for setup/precompute
  placement differences.
- Watch memory ownership when changing `rotate_batch`, batch ciphertext
  extraction, generated C array outputs, or poly C free logic.
- Be careful with move and delete operations. Prefer moving files to
  `~/.trash` instead of deleting them outright.
- If something fails repeatedly, stop and ask for help rather than retrying the
  same broken path.

## Project Context

- Optimization plan and current measurements: `docs/opt.md`
- Primitive bootstrap demo/generator: `ace_edsl/examples/bootstrap_full.py`
- ResNet DSL bootstrap integration run: `a_dsl_bts.sh`
- First-bootstrap comparison harness: `verify_resnet_first.sh`
- Bootstrap decomposition: `ace_edsl/edsl/core/bootstrap_decomposition.py`
- Bootstrap math/constants: `ace_edsl/edsl/core/bootstrap_math.py`
- EDSL value/operator layer: `ace_edsl/edsl/core/air_value.py`
- Python/C++ bindings: `bindings/src/air_builder_bindings.cpp`
- Poly generated-C free handling: `fhe-cmplr/include/fhe/poly/poly2c_mfree.h`
- Native FHE compiler/runtime: `fhe-cmplr/`

## Docker Rule

Run all build/test/profile/benchmark commands from inside the container:

```bash
docker exec -it ace-compiler-dev bash
cd /app
```

Examples:

```bash
python3 -m pytest -q ace_edsl/tests/test_resnet_bootstrap_utils.py
python3 -m pytest -q ace_edsl/tests/test_bootstrap_stage_ops.py
python3 -m pytest -q ace_edsl/tests/test_bootstrap_full.py
ACE_STOP_AFTER_FIRST_BTS=1 ACE_BOOTSTRAP_CT_ENCODE=1 bash a_dsl_bts.sh
bash verify_resnet_first.sh
```

Keep multi-image DSL runs conservative, usually `ACE_IMAGE_PARALLELISM=1`,
unless the experiment is explicitly about memory behavior.

## Development Workflow

1. Read existing context in `docs/opt.md` and relevant source.
2. Profile or inspect the current behavior inside `ace-compiler-dev`.
3. State the profile-backed assumption.
4. Plan a scoped fix and explain how it avoids hardcoding.
5. Run a small experiment inside `ace-compiler-dev`.
6. If it works, implement on a new branch.
7. Run functional and performance validation against `rtlib-bts`.
8. Commit only after correctness and performance both hold.

## Memory And Skills

- `.agents/memory/` contains project memory. Search it before repeating an
  investigation.
- When a meaningful task is complete, record work done, takeaways, and lessons
  learned in `.agents/memory/YYYYMMDD.md`.
- `.agents/skills/` contains local skills. Keep important new build/test
  knowledge there when it will be reused.

## Working Style

- Treat the repo as shared and possibly dirty. Never revert user changes unless
  explicitly asked.
- Prefer editing existing files over creating new files, unless docs or new
  files are explicitly requested.
- Do only what is asked.
- Keep responses concise and technical.
- Do not leave trailing spaces.
- Do not use emojis unless explicitly requested.
