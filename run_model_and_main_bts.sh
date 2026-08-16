#!/usr/bin/env bash
# Copyright (c) Ant Group Co., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

SUITE_NAME="${BTS_SUITE_NAME:-LeNet}"
MODEL_CHOICES="${BTS_MODEL_CHOICES:-lenet lenet_x2}"
DEFAULT_WORK_DIR="${BTS_DEFAULT_WORK_DIR:-tmp_lenet_bts}"
ENTRYPOINT="${BTS_ENTRYPOINT:-$(basename "${BASH_SOURCE[0]}")}"

usage() {
  cat <<EOF
Build and run the configured model_and_main ${SUITE_NAME} models with RTL
and/or generated DSL bootstrap.

Usage:
  ${ENTRYPOINT} [options]

Options:
  --model MODEL          both (default), or one of: ${MODEL_CHOICES}
  --bootstrap MODE       both (default), dsl, or rtl
  --work-dir DIR         Build/result directory (default: ${DEFAULT_WORK_DIR})
  --oracle-tolerance X   Maximum absolute error from clear output (default: 0.05)
  --parity-tolerance X   Maximum DSL-vs-RTL absolute delta (default: 0.002)
  --build-only           Build selected binaries without running them
  --rebuild              Regenerate and recompile cached artifacts
  -h, --help             Show this help

Examples:
  # Both models, both bootstrap implementations.
  ./${ENTRYPOINT}

  # Only one configured model, still comparing DSL with RTL.
  ./${ENTRYPOINT} --model ${MODEL_CHOICES##* }

  # Only generated DSL bootstrap for one configured model.
  ./${ENTRYPOINT} --model ${MODEL_CHOICES%% *} --bootstrap dsl

Run this script inside ace-compiler-dev, for example:
  docker exec ace-compiler-dev bash -lc 'cd /app && ./${ENTRYPOINT} --model ${MODEL_CHOICES%% *}'
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
MODEL_DIR="${REPO_ROOT}/model_and_main"
MODEL_SELECTION="both"
BOOTSTRAP_SELECTION="both"
WORK_DIR="${REPO_ROOT}/${DEFAULT_WORK_DIR}"
ORACLE_TOLERANCE="0.05"
PARITY_TOLERANCE="0.002"
BUILD_ONLY=0
REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
      MODEL_SELECTION="$2"
      shift 2
      ;;
    --bootstrap)
      [[ $# -ge 2 ]] || { echo "--bootstrap requires a value" >&2; exit 2; }
      BOOTSTRAP_SELECTION="$2"
      shift 2
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || { echo "--work-dir requires a value" >&2; exit 2; }
      WORK_DIR="$2"
      shift 2
      ;;
    --oracle-tolerance)
      [[ $# -ge 2 ]] || { echo "--oracle-tolerance requires a value" >&2; exit 2; }
      ORACLE_TOLERANCE="$2"
      shift 2
      ;;
    --parity-tolerance)
      [[ $# -ge 2 ]] || { echo "--parity-tolerance requires a value" >&2; exit 2; }
      PARITY_TOLERANCE="$2"
      shift 2
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

read -r -a AVAILABLE_MODELS <<<"${MODEL_CHOICES}"
case "${MODEL_SELECTION}" in
  both)
    MODELS=("${AVAILABLE_MODELS[@]}")
    ;;
  *)
    model_is_valid=0
    for available_model in "${AVAILABLE_MODELS[@]}"; do
      if [[ "${MODEL_SELECTION}" == "${available_model}" ]]; then
        model_is_valid=1
        break
      fi
    done
    if [[ "${model_is_valid}" -eq 0 ]]; then
      echo "invalid --model value: ${MODEL_SELECTION}" >&2
      echo "configured models: ${MODEL_CHOICES}" >&2
      exit 2
    fi
    MODELS=("${MODEL_SELECTION}")
    ;;
esac

case "${BOOTSTRAP_SELECTION}" in
  both) BOOTSTRAPS=(rtl dsl) ;;
  rtl|dsl) BOOTSTRAPS=("${BOOTSTRAP_SELECTION}") ;;
  *) echo "invalid --bootstrap value: ${BOOTSTRAP_SELECTION}" >&2; exit 2 ;;
esac

python3 - "${ORACLE_TOLERANCE}" "${PARITY_TOLERANCE}" <<'PY'
import math
import sys

for label, raw in zip(("oracle", "parity"), sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid {label} tolerance: {raw}") from exc
    if not math.isfinite(value) or value < 0:
        raise SystemExit(f"invalid {label} tolerance: {raw}")
PY

if [[ ! -d /usr/local/rtlib/include/ant || ! -f /usr/local/rtlib/lib/libFHErt_ant.a ]]; then
  echo "ANT rtlib is unavailable; run this script inside ace-compiler-dev" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/build/driver/fhe_cmplr" ]]; then
  FHE_CMPLR="${REPO_ROOT}/build/driver/fhe_cmplr"
else
  FHE_CMPLR="/usr/local/bin/fhe_cmplr"
fi
if [[ ! -x "${FHE_CMPLR}" ]]; then
  echo "missing ACE compiler: ${FHE_CMPLR}" >&2
  exit 1
fi

BOOTSTRAP_UTILS="${REPO_ROOT}/ace_edsl/examples/resnet_bootstrap_utils.py"
BOOTSTRAP_OUTPUT_DIR="${REPO_ROOT}/ace_edsl/examples/output"
BOOTSTRAP_GENERATED_C="${BOOTSTRAP_OUTPUT_DIR}/bootstrap_full.c"
BOOTSTRAP_RAW_AIR="${BOOTSTRAP_OUTPUT_DIR}/bootstrap_full_raw.air"
BOOTSTRAP_DATA="${BOOTSTRAP_OUTPUT_DIR}/bootstrap_full_data.msg"
CAPTURE_SOURCE="${REPO_ROOT}/ace_edsl/tests/dnn_output_capture.c"

COMMON_C_FLAGS=(
  -DRTLIB_SUPPORT_LINUX
  -I /usr/local/include
  -I /usr/local/rtlib/include
  -I /usr/local/rtlib/include/ant
  -O3 -DNDEBUG -std=gnu11 -fopenmp
)
LINK_LIBS=(
  /usr/local/rtlib/lib/libFHErt_ant.a
  /usr/local/rtlib/lib/libFHErt_common.a
  /usr/local/lib/libAIRutil.a
  -lgmp -lm -lgomp
)

mkdir -p "${WORK_DIR}"

needs_rebuild() {
  local output="$1"
  shift
  if [[ "${REBUILD}" -ne 0 || ! -e "${output}" ]]; then
    return 0
  fi
  local input
  for input in "$@"; do
    if [[ "${input}" -nt "${output}" ]]; then
      return 0
    fi
  done
  return 1
}

contains_mode() {
  local wanted="$1"
  local mode
  for mode in "${BOOTSTRAPS[@]}"; do
    [[ "${mode}" == "${wanted}" ]] && return 0
  done
  return 1
}

read_graph_context() {
  local graph_c="$1"
  python3 - "${graph_c}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r"CKKS_PARAMS\* Get_context_params\(\).*?LIB_ANT,\s*(\d+),\s*\d+,\s*(\d+),",
    text,
    re.DOTALL,
)
if match is None:
    raise SystemExit(f"cannot read CKKS parameters from {sys.argv[1]}")
print(match.group(1), match.group(2))
PY
}

dsl_context_dir() {
  local graph_c="$1"
  local poly_degree mul_depth
  read -r poly_degree mul_depth < <(read_graph_context "${graph_c}")
  printf '%s/dsl_bootstrap/N%s_Q%s\n' \
    "${WORK_DIR}" "${poly_degree}" "$((mul_depth + 1))"
}

prepare_dsl_bootstrap() {
  local model="$1"
  local graph_c="${WORK_DIR}/${model}/${model}.graph.c"
  local dsl_dir
  dsl_dir="$(dsl_context_dir "${graph_c}")"
  local cached_c="${dsl_dir}/bootstrap_full.generated.c"
  local cached_air="${dsl_dir}/bootstrap_full_raw.air"
  local cached_data="${dsl_dir}/bootstrap_full_data.msg"
  local body_c="${dsl_dir}/bootstrap_full_body.c"
  local shim_c="${dsl_dir}/dsl_bootstrap_shim.c"
  local body_obj="${dsl_dir}/bootstrap_full_body.o"
  local shim_obj="${dsl_dir}/dsl_bootstrap_shim.o"
  mkdir -p "${dsl_dir}"

  read -r poly_degree mul_depth < <(read_graph_context "${graph_c}")

  local reusable=1
  [[ -f "${cached_c}" ]] || reusable=0
  [[ -f "${cached_air}" ]] || reusable=0
  [[ -s "${cached_data}" ]] || reusable=0
  if [[ "${reusable}" -ne 0 ]]; then
    [[ "$(grep -c 'CKKS.linear_transform' "${cached_air}" || true)" -eq 6 ]] || reusable=0
    grep -q "LIB_ANT, ${poly_degree}, 0, ${mul_depth}," "${cached_c}" || reusable=0
  fi

  if [[ "${REBUILD}" -ne 0 || "${reusable}" -eq 0 ]]; then
    echo "==> Generating ${model} DSL bootstrap (N=${poly_degree}, Q=$((mul_depth + 1)))"
    (
      cd "${REPO_ROOT}/ace_edsl/examples"
      PYTHONPATH="${REPO_ROOT}/ace_edsl:${REPO_ROOT}" \
      ACE_BOOTSTRAP_LINEAR_TRANSFORM=1 \
      ACE_BOOTSTRAP_PARALLEL_EVAL_MOD=1 \
      ACE_BOOTSTRAP_CONTEXT_MUL_LEVEL="$((mul_depth + 1))" \
      ACE_BOOTSTRAP_CT_ENCODE=1 \
      ACE_CT_ENCODE_DEPTH="${mul_depth}" \
      python3 "${BOOTSTRAP_UTILS}" generate-demo \
        --impl primitive \
        --poly-degree "${poly_degree}" \
        --mul-level "${mul_depth}"
    )
    cp "${BOOTSTRAP_GENERATED_C}" "${cached_c}"
    cp "${BOOTSTRAP_RAW_AIR}" "${cached_air}"
    cp --reflink=auto "${BOOTSTRAP_DATA}" "${cached_data}"
  else
    echo "==> Reusing compatible ${model} DSL bootstrap (use --rebuild to refresh)"
  fi

  [[ "$(grep -c 'CKKS.linear_transform' "${cached_air}" || true)" -eq 6 ]] || {
    echo "expected six CKKS.linear_transform operations in ${cached_air}" >&2
    exit 1
  }
  grep -q "LIB_ANT, ${poly_degree}, 0, ${mul_depth}," "${cached_c}" || {
    echo "generated DSL bootstrap context does not match ${model}" >&2
    exit 1
  }
  if grep -q 'Eval_bootstrap_ciph(' "${cached_c}"; then
    echo "generated DSL body unexpectedly calls the RTL bootstrap" >&2
    exit 1
  fi

  if needs_rebuild "${body_c}" "${cached_c}" "${BOOTSTRAP_UTILS}"; then
    python3 "${BOOTSTRAP_UTILS}" emit-body \
      --bootstrap-c "${cached_c}" \
      --output "${body_c}"
    # Keep each cached body bound to its own offline-encoded plaintext file;
    # generate-demo uses a shared output location which the next model replaces.
    sed -i "s#${BOOTSTRAP_DATA}#${cached_data}#g" "${body_c}"
  fi
  if needs_rebuild "${shim_c}" "${BOOTSTRAP_UTILS}"; then
    python3 "${BOOTSTRAP_UTILS}" emit-shim --output "${shim_c}"
  fi
  if needs_rebuild "${body_obj}" "${body_c}"; then
    echo "==> Compiling generated DSL bootstrap"
    gcc -c "${body_c}" "${COMMON_C_FLAGS[@]}" -o "${body_obj}"
  fi
  if needs_rebuild "${shim_obj}" "${shim_c}"; then
    gcc -c "${shim_c}" "${COMMON_C_FLAGS[@]}" -o "${shim_obj}"
  fi
}

compile_graph() {
  local model="$1"
  local model_dir="${WORK_DIR}/${model}"
  local onnx="${MODEL_DIR}/${model}.onnx"
  local graph_c="${model_dir}/${model}.graph.c"
  local weight="${model_dir}/${model}.weight"
  local compile_log="${model_dir}/compile.log"
  local profile_file="${model_dir}/compile.profile"
  local compile_profile="auto-ckks-context-v1"
  mkdir -p "${model_dir}"

  [[ -f "${onnx}" ]] || { echo "missing model: ${onnx}" >&2; exit 1; }
  [[ -f "${MODEL_DIR}/${model}.c" ]] || {
    echo "missing driver: ${MODEL_DIR}/${model}.c" >&2
    exit 1
  }

  if needs_rebuild "${graph_c}" "${onnx}" "${FHE_CMPLR}" ||
     [[ ! -s "${weight}" ]] || [[ ! -f "${profile_file}" ]] ||
     [[ "$(<"${profile_file}")" != "${compile_profile}" ]]; then
    echo "==> Compiling ${model}.onnx with its planned CKKS context"
    mapfile -t ace_options < <(
      PYTHONPATH="${REPO_ROOT}/fhe-cmplr/test" \
      python3 - "${model}" <<'PY'
import sys
from ace_util import get_ace_option, get_test_name

for option in get_ace_option(get_test_name(sys.argv[1]), "cgo25", "ant", None, True, False):
    print(option)
PY
    )
    (
      cd "${model_dir}"
      "${FHE_CMPLR}" "${onnx}" "${ace_options[@]}" \
        "-P2C:df=${weight}" -o "${graph_c}"
    ) >"${compile_log}" 2>&1 || {
      tail -n 80 "${compile_log}" >&2
      exit 1
    }
    printf '%s\n' "${compile_profile}" >"${profile_file}"
  else
    echo "==> Reusing compiled ${model} graph"
  fi

  if ! grep -q 'Eval_bootstrap_ciph(' "${graph_c}"; then
    echo "${model} graph contains no bootstrap calls" >&2
    exit 1
  fi
}

build_model_mode() {
  local model="$1"
  local mode="$2"
  local model_dir="${WORK_DIR}/${model}"
  local graph_c="${model_dir}/${model}.graph.c"
  local graph_obj="${model_dir}/${model}.${mode}.graph.o"
  local driver_obj="${model_dir}/${model}.driver.o"
  local capture_obj="${model_dir}/dnn_output_capture.o"
  local binary="${model_dir}/${model}.${mode}.ace"

  if needs_rebuild "${driver_obj}" "${MODEL_DIR}/${model}.c" "${BASH_SOURCE[0]}"; then
    gcc -c -DHandle_output=Capture_dnn_output \
      -DPrint_tensor=Suppress_dnn_tensor \
      "${MODEL_DIR}/${model}.c" "${COMMON_C_FLAGS[@]}" -o "${driver_obj}"
  fi
  if needs_rebuild "${capture_obj}" "${CAPTURE_SOURCE}"; then
    gcc -c "${CAPTURE_SOURCE}" "${COMMON_C_FLAGS[@]}" -o "${capture_obj}"
  fi

  if [[ "${mode}" == "dsl" ]]; then
    if needs_rebuild "${graph_obj}" "${graph_c}"; then
      gcc -c -DEval_bootstrap_ciph=Eval_bootstrap_ciph_dsl \
        "${graph_c}" "${COMMON_C_FLAGS[@]}" -o "${graph_obj}"
    fi
    local dsl_dir
    dsl_dir="$(dsl_context_dir "${graph_c}")"
    local body_obj="${dsl_dir}/bootstrap_full_body.o"
    local shim_obj="${dsl_dir}/dsl_bootstrap_shim.o"
    if needs_rebuild "${binary}" "${graph_obj}" "${driver_obj}" "${capture_obj}" \
        "${body_obj}" "${shim_obj}"; then
      c++ "${graph_obj}" "${driver_obj}" "${capture_obj}" \
        "${body_obj}" "${shim_obj}" "${LINK_LIBS[@]}" -o "${binary}"
    fi
  else
    if needs_rebuild "${graph_obj}" "${graph_c}"; then
      gcc -c "${graph_c}" "${COMMON_C_FLAGS[@]}" -o "${graph_obj}"
    fi
    if needs_rebuild "${binary}" "${graph_obj}" "${driver_obj}" "${capture_obj}"; then
      c++ "${graph_obj}" "${driver_obj}" "${capture_obj}" \
        "${LINK_LIBS[@]}" -o "${binary}"
    fi
  fi
  echo "built ${binary}"
}

run_model_mode() {
  local model="$1"
  local mode="$2"
  local model_dir="${WORK_DIR}/${model}"
  local binary="${model_dir}/${model}.${mode}.ace"
  local log="${model_dir}/${mode}.log"
  local timing="${model_dir}/${mode}.rtlib-timing.txt"
  local status_file="${model_dir}/${mode}.status"
  local -a run_env=("RTLIB_TIMING_OUTPUT=${timing}")
  if [[ "${mode}" == "dsl" ]]; then
    run_env+=(RTLIB_DISABLE_BOOTSTRAP_PRECOM=1)
  fi

  echo "==> Running ${model} with ${mode^^} bootstrap"
  set +e
  (
    cd "${model_dir}"
    /usr/bin/time -f 'ACE_DNN_WALL_SECONDS=%e' \
      env "${run_env[@]}" "${binary}"
  ) 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "${status}" >"${status_file}"
  if [[ "${status}" -ne 0 ]]; then
    echo "${model}/${mode} exited with status ${status}" >&2
  fi
}

analyze_model() {
  local model="$1"
  local model_dir="${WORK_DIR}/${model}"
  local summary="${model_dir}/summary.json"
  local -a analyzer_args=(
    "${model}" "${MODEL_DIR}/${model}.c" "${summary}"
    "${ORACLE_TOLERANCE}" "${PARITY_TOLERANCE}"
  )
  local mode
  for mode in "${BOOTSTRAPS[@]}"; do
    analyzer_args+=("${mode}" "${model_dir}/${mode}.log" "${model_dir}/${mode}.status")
  done

  python3 - "${analyzer_args[@]}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

model, driver_path, summary_path, oracle_raw, parity_raw, *runs = sys.argv[1:]
oracle_tolerance = float(oracle_raw)
parity_tolerance = float(parity_raw)

driver = Path(driver_path).read_text(encoding="utf-8")
match = re.search(r"double\s+Expected_data\[\]\s*=\s*\{([^}]*)\}", driver)
if match is None:
    raise SystemExit(f"cannot find Expected_data in {driver_path}")
expected = [float(value) for value in match.group(1).split(",") if value.strip()]

report = {
    "model": model,
    "oracle_tolerance": oracle_tolerance,
    "parity_tolerance": parity_tolerance,
    "expected": expected,
    "runs": {},
}
passed = True
for mode, log_path, status_path in zip(runs[0::3], runs[1::3], runs[2::3]):
    status = int(Path(status_path).read_text(encoding="utf-8").strip())
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    wall_match = re.search(r"^ACE_DNN_WALL_SECONDS=([^\s]+)$", text, re.MULTILINE)
    wall_seconds = float(wall_match.group(1)) if wall_match else None
    values = {}
    for index, value in re.findall(
        r"^ACE_DNN_OUTPUT index=(\d+) value=([^\s]+)$", text, re.MULTILINE
    ):
        values[int(index)] = float(value)
    ordered = [values[index] for index in range(len(expected)) if index in values]
    finite = len(ordered) == len(expected) and all(math.isfinite(x) for x in ordered)
    if finite:
        differences = [abs(a - b) for a, b in zip(ordered, expected)]
        max_abs = max(differences)
        rmse = math.sqrt(sum(x * x for x in differences) / len(differences))
        argmax = max(range(len(ordered)), key=ordered.__getitem__)
        expected_argmax = max(range(len(expected)), key=expected.__getitem__)
    else:
        max_abs = math.inf
        rmse = math.inf
        argmax = None
        expected_argmax = max(range(len(expected)), key=expected.__getitem__)
    mode_passed = (
        status == 0
        and finite
        and max_abs <= oracle_tolerance
        and argmax == expected_argmax
    )
    passed = passed and mode_passed
    report["runs"][mode] = {
        "exit_status": status,
        "output": ordered,
        "finite": finite,
        "max_abs_vs_clear": max_abs if math.isfinite(max_abs) else None,
        "rmse_vs_clear": rmse if math.isfinite(rmse) else None,
        "argmax": argmax,
        "expected_argmax": expected_argmax,
        "wall_seconds": wall_seconds,
        "passed": mode_passed,
    }
    print(
        f"ACE_DNN_RESULT model={model} mode={mode} "
        f"max_abs_vs_clear={max_abs:.9g} rmse_vs_clear={rmse:.9g} "
        f"argmax={argmax} expected_argmax={expected_argmax} "
        f"wall_seconds={wall_seconds} "
        f"verdict={'PASS' if mode_passed else 'FAIL'}"
    )

if "dsl" in report["runs"] and "rtl" in report["runs"]:
    dsl = report["runs"]["dsl"]["output"]
    rtl = report["runs"]["rtl"]["output"]
    if len(dsl) == len(expected) and len(rtl) == len(expected):
        differences = [abs(a - b) for a, b in zip(dsl, rtl)]
        max_abs = max(differences)
        rmse = math.sqrt(sum(x * x for x in differences) / len(differences))
        dsl_argmax = max(range(len(dsl)), key=dsl.__getitem__)
        rtl_argmax = max(range(len(rtl)), key=rtl.__getitem__)
        parity_passed = (
            math.isfinite(max_abs)
            and max_abs <= parity_tolerance
            and dsl_argmax == rtl_argmax
        )
    else:
        max_abs = math.inf
        rmse = math.inf
        dsl_argmax = None
        rtl_argmax = None
        parity_passed = False
    passed = passed and parity_passed
    report["dsl_rtl_parity"] = {
        "max_abs": max_abs if math.isfinite(max_abs) else None,
        "rmse": rmse if math.isfinite(rmse) else None,
        "dsl_argmax": dsl_argmax,
        "rtl_argmax": rtl_argmax,
        "passed": parity_passed,
    }
    print(
        f"ACE_DNN_PARITY model={model} max_abs={max_abs:.9g} rmse={rmse:.9g} "
        f"dsl_argmax={dsl_argmax} rtl_argmax={rtl_argmax} "
        f"verdict={'PASS' if parity_passed else 'FAIL'}"
    )

report["passed"] = passed
Path(summary_path).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(f"ACE_DNN_VERDICT model={model} verdict={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if passed else 1)
PY
}

for model in "${MODELS[@]}"; do
  compile_graph "${model}"
  if contains_mode dsl; then
    prepare_dsl_bootstrap "${model}"
  fi
  for mode in "${BOOTSTRAPS[@]}"; do
    build_model_mode "${model}" "${mode}"
  done
done

if [[ "${BUILD_ONLY}" -ne 0 ]]; then
  echo "Build-only run complete: ${WORK_DIR}"
  exit 0
fi

for model in "${MODELS[@]}"; do
  for mode in "${BOOTSTRAPS[@]}"; do
    run_model_mode "${model}" "${mode}"
  done
done

overall_status=0
for model in "${MODELS[@]}"; do
  if ! analyze_model "${model}"; then
    overall_status=1
  fi
done

echo "${SUITE_NAME} bootstrap results: ${WORK_DIR}"
exit "${overall_status}"
