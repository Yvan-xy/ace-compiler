#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_PHANTOM_COMMIT="d722091d1a7fa7893abd8f949cd5f6d569c7f6ee"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_DIR="${ACE_PHANTOM_RUN_CNN_SOURCE_DIR:-/home/dyf/code/phantom-ant}"
BUILD_DIR=""
QUALIFICATION_DIR=""
GENERATED_DIR=""
GENERATE=false
BUILD_ONLY=false
MODE="both"
START_IMAGE=0
END_IMAGE=0
BUILD_JOBS="${ACE_PHANTOM_BUILD_JOBS:-2}"

usage() {
  cat <<'EOF'
Usage:
  tools/phantom_gpu/run_cnn_dsl_bootstrap.sh \
    --qualification-dir DIR [options]
  tools/phantom_gpu/run_cnn_dsl_bootstrap.sh --generate [options]
  tools/phantom_gpu/run_cnn_dsl_bootstrap.sh --generated-dir DIR [options]

Build and optionally run the handwritten Phantom ResNet-20/CIFAR-10 program
in a same-profile comparison: generated ACE DSL bootstrap, native Phantom Slim,
or both.  Both modes use the same audited ACE Q32/P11 context and constants.

Artifact input (choose exactly one):
  --qualification-dir DIR  Q32 compile_only run root or bootstrap_qualification dir
  --generate               Reserved; rejected until real-output qualification exists
  --generated-dir DIR      Development-only generated Q32 directory; not qualified

Options:
  --mode MODE              dsl, native, or both (default: both)
  --build-only             Build and inspect the executable, but do not run it
  --start-image N          First CIFAR-10 image, inclusive (default: 0)
  --end-image N            Last CIFAR-10 image, inclusive (default: 0)
  --image-range A:B        Set both inclusive image bounds
  --phantom-dir DIR        d722 Phantom working repository
  --build-dir DIR          Build directory (default: PHANTOM_DIR/build-ace-resnet-q32)
  --jobs N                 Parallel build jobs (default: ACE_PHANTOM_BUILD_JOBS or 2)
  -h, --help               Show this help

The only accepted helper profile is N=65536, full 32768-slot packing, Q32/P11,
q0=60, sf=56, hw=192, input level 1, and q-part count 3. Q26 and Q31 helpers
are deliberately rejected.  This driver never compares against Phantom's
default Q23/q0=51/sf=46 executable.
EOF
}

die() {
  echo "run_cnn_dsl_bootstrap: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qualification-dir)
      [[ $# -ge 2 ]] || die "--qualification-dir requires a value"
      QUALIFICATION_DIR="$2"
      shift 2
      ;;
    --generate)
      GENERATE=true
      shift
      ;;
    --generated-dir)
      [[ $# -ge 2 ]] || die "--generated-dir requires a value"
      GENERATED_DIR="$2"
      shift 2
      ;;
    --build-only)
      BUILD_ONLY=true
      shift
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires dsl, native, or both"
      MODE="$2"
      shift 2
      ;;
    --start-image)
      [[ $# -ge 2 ]] || die "--start-image requires a value"
      START_IMAGE="$2"
      shift 2
      ;;
    --end-image)
      [[ $# -ge 2 ]] || die "--end-image requires a value"
      END_IMAGE="$2"
      shift 2
      ;;
    --image-range)
      [[ $# -ge 2 ]] || die "--image-range requires A:B"
      [[ "$2" =~ ^([0-9]+):([0-9]+)$ ]] || die "invalid image range: $2"
      START_IMAGE="${BASH_REMATCH[1]}"
      END_IMAGE="${BASH_REMATCH[2]}"
      shift 2
      ;;
    --phantom-dir)
      [[ $# -ge 2 ]] || die "--phantom-dir requires a value"
      PHANTOM_DIR="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir requires a value"
      BUILD_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a value"
      BUILD_JOBS="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

artifact_input_count=0
[[ "${GENERATE}" == true ]] && ((artifact_input_count += 1))
[[ -n "${QUALIFICATION_DIR}" ]] && ((artifact_input_count += 1))
[[ -n "${GENERATED_DIR}" ]] && ((artifact_input_count += 1))
((artifact_input_count == 1)) ||
  die "choose exactly one of --generate, --qualification-dir, or --generated-dir"
if [[ "${GENERATE}" == true ]]; then
  die "--generate is unavailable for ResNet until compile_only has a real-output clear-imag qualification profile; use an audited --generated-dir"
fi
[[ "${START_IMAGE}" =~ ^[0-9]+$ ]] || die "--start-image must be an integer"
[[ "${END_IMAGE}" =~ ^[0-9]+$ ]] || die "--end-image must be an integer"
((START_IMAGE <= END_IMAGE)) || die "start image exceeds end image"
((END_IMAGE < 10000)) || die "image range must be within CIFAR-10 test images 0..9999"
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
case "${MODE}" in
  dsl | native | both) ;;
  *) die "--mode must be dsl, native, or both (got: ${MODE})" ;;
esac
BUILD_DSL=false
BUILD_NATIVE=false
if [[ "${MODE}" == dsl || "${MODE}" == both ]]; then
  BUILD_DSL=true
fi
if [[ "${MODE}" == native || "${MODE}" == both ]]; then
  BUILD_NATIVE=true
fi

require_command cmake
require_command c++
require_command git
require_command ninja
require_command nm
require_command python3
require_command rg
require_command sha256sum

[[ -d "${PHANTOM_DIR}" ]] || die "Phantom source directory does not exist: ${PHANTOM_DIR}"
PHANTOM_DIR="$(cd -- "${PHANTOM_DIR}" && pwd -P)"
if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${PHANTOM_DIR}/build-ace-resnet-q32"
fi
mkdir -p -- "${BUILD_DIR}"
BUILD_DIR="$(cd -- "${BUILD_DIR}" && pwd -P)"
case "${BUILD_DIR}" in
  / | "${REPO_ROOT}" | "${PHANTOM_DIR}")
    die "refusing unsafe build directory: ${BUILD_DIR}"
    ;;
esac
[[ "$(dirname -- "${BUILD_DIR}")" == "${PHANTOM_DIR}" ]] ||
  die "--build-dir must be an immediate child of PHANTOM_DIR so run_cnn's ../../ data paths remain valid"

PHANTOM_HEAD="$(git -C "${PHANTOM_DIR}" rev-parse HEAD 2>/dev/null)"
[[ "${PHANTOM_HEAD}" == "${EXPECTED_PHANTOM_COMMIT}" ]] ||
  die "Phantom HEAD ${PHANTOM_HEAD} is not ${EXPECTED_PHANTOM_COMMIT}"
git -C "${PHANTOM_DIR}" cat-file -e "${EXPECTED_PHANTOM_COMMIT}^{commit}" 2>/dev/null ||
  die "qualified Phantom commit is not available in ${PHANTOM_DIR}"

# Record both worktrees before any build system writes. Dirty trees are allowed;
# these records make it explicit that the driver never resets user changes.
git -C "${REPO_ROOT}" status --short >"${BUILD_DIR}/ace-status-before.txt"
git -C "${PHANTOM_DIR}" status --short >"${BUILD_DIR}/phantom-status-before.txt" 2>"${BUILD_DIR}/phantom-status-before.stderr.txt" ||
  die "could not record Phantom worktree status"
printf '%s\n' "${PHANTOM_HEAD}" >"${BUILD_DIR}/phantom-head.txt"

RUN_CNN_SOURCE="${PHANTOM_DIR}/examples/run_cnn.cu"
BRIDGE_SOURCE="${SCRIPT_DIR}/harness/run_cnn_dsl_bootstrap_bridge.cu"
IO_METADATA_SOURCE="${SCRIPT_DIR}/harness/run_cnn_phantom_io_metadata.cc"
SUMMARY_SOURCE="${SCRIPT_DIR}/summarize_run_cnn_e2e.py"
require_file "${RUN_CNN_SOURCE}"
require_file "${IO_METADATA_SOURCE}"
require_file "${SUMMARY_SOURCE}"
if [[ "${BUILD_DSL}" == true ]]; then
  require_file "${BRIDGE_SOURCE}"
  rg -q 'ACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP' "${RUN_CNN_SOURCE}" ||
    die "run_cnn.cu lacks the opt-in DSL bootstrap path"
fi
if [[ "${BUILD_NATIVE}" == true ]]; then
  rg -q 'ACE_PHANTOM_RUN_CNN_MATCHED_NATIVE_SLIM' "${RUN_CNN_SOURCE}" ||
    die "run_cnn.cu lacks the opt-in matched native-Slim path"
  rg -q 'set_slim_relu[[:space:]]*\([[:space:]]*true[[:space:]]*\)' \
    "${RUN_CNN_SOURCE}" ||
    die "matched native-Slim path does not enable its fused ReLU"
  rg -q 'slim_bootstrap[[:space:]]*\(' "${RUN_CNN_SOURCE}" ||
    die "matched native-Slim path does not call Slim directly"
fi
rg -q 'Phantom_borrow_runtime' "${RUN_CNN_SOURCE}" ||
  die "run_cnn.cu does not borrow the ACE singleton runtime"
rg -q 'Prepare_context[[:space:]]*\(' "${RUN_CNN_SOURCE}" ||
  die "run_cnn.cu does not prepare the ACE singleton context"
rg -q 'Finalize_context[[:space:]]*\(' "${RUN_CNN_SOURCE}" ||
  die "run_cnn.cu does not finalize the ACE singleton context"

# A name-only borrow is insufficient: ciphertexts crossing the generated/native
# boundary must have one parameter identity.  Prove from the selected source
# that the CNN evaluator is assembled in the Phantom ABI's exact constructor
# order from the ACE-owned context, keys, and encoder.  Its extra CNN rotation
# key is intentionally separate and is derived from that same secret/context.
python3 - "${RUN_CNN_SOURCE}" <<'PY'
import re
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
source = re.sub(r"//[^\n]*", "", source)
compact = re.sub(r"\s+", "", source)

required = (
    "constPHANTOM_BORROWED_RUNTIMEruntime=Phantom_borrow_runtime();",
    "runtime._secret_key->create_galois_keys_from_steps("
    "*runtime._context,cnn_rotation_basis,",
    "CKKSEvaluatorckks_evaluator("
    "runtime._context,runtime._public_key,runtime._secret_key,"
    "runtime._encoder,runtime._relin_key,&cnn_galois_keys,scale);",
    "voidace_matched_batch_norm(",
    "ckks_evaluator.encoder.encode(offset,value.scale(),encoded_offset);",
    "ckks_evaluator.evaluator.mod_switch_to_inplace("
    "encoded_offset,value.chain_index());",
    "ckks_evaluator.evaluator.sub_plain_inplace(value,encoded_offset);",
)
missing = [contract for contract in required if contract not in compact]
if missing:
    raise SystemExit(
        "run_cnn.cu does not construct its evaluator wholly from the borrowed "
        "ACE provider objects in the required order; missing: " + repr(missing)
    )
if compact.count("ace_matched_batch_norm(cnn,cnn,bn_bias[stage]") != 3:
    raise SystemExit(
        "run_cnn.cu does not route all three matched CIFAR-10 "
        "batch-normalization sites through same-level plaintext subtraction"
    )
PY

# The handwritten CNN silently accepts failed image/label reads, so validate
# every consumed dataset/weight token before spending GPU time.  Record the
# exact inputs in the build evidence for later checksum closure.
[[ -d "${PHANTOM_DIR}/result" && -w "${PHANTOM_DIR}/result" ]] ||
  die "Phantom result directory is missing or not writable"
python3 - "${PHANTOM_DIR}" "${BUILD_DIR}/resnet20-data-preflight.json" \
  "${START_IMAGE}" "${END_IMAGE}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
start_image = int(sys.argv[3])
end_image = int(sys.argv[4])

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def numeric_tokens(path, cast):
    if not path.is_file():
        raise SystemExit(f"missing ResNet input: {path}")
    values = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            for token in line.split():
                try:
                    value = cast(token)
                except ValueError as error:
                    raise SystemExit(
                        f"invalid numeric token in {path}:{line_number}: {token!r}"
                    ) from error
                if isinstance(value, float) and not math.isfinite(value):
                    raise SystemExit(f"nonfinite value in {path}:{line_number}")
                values.append(value)
    return values

values_path = root / "testFile" / "test_values.txt"
labels_path = root / "testFile" / "test_label.txt"
values = numeric_tokens(values_path, float)
labels = numeric_tokens(labels_path, int)
pixels_per_image = 32 * 32 * 3
if len(values) % pixels_per_image:
    raise SystemExit("CIFAR-10 value count is not divisible by 3072")
image_count = len(values) // pixels_per_image
if len(labels) != image_count:
    raise SystemExit(
        f"CIFAR-10 values contain {image_count} images but labels contain {len(labels)}"
    )
if any(label < 0 or label > 9 for label in labels):
    raise SystemExit("CIFAR-10 labels must all be in 0..9")
if end_image >= image_count:
    raise SystemExit(
        f"requested image {end_image}, but local CIFAR-10 fixture has only "
        f"images 0..{image_count - 1}"
    )

weights_root = root / "pretrained_parameters" / "resnet20_new"
expected = {
    "conv1_weight.txt": 3 * 16 * 3 * 3,
    "bn1_bias.txt": 16,
    "bn1_running_mean.txt": 16,
    "bn1_running_var.txt": 16,
    "bn1_weight.txt": 16,
    "linear_weight.txt": 10 * 64,
    "linear_bias.txt": 10,
}
for block, channels in ((1, 16), (2, 32), (3, 64)):
    for unit in range(3):
        if block == 1 or (block == 2 and unit == 0):
            input_channels = 16
        elif block == 2 or (block == 3 and unit == 0):
            input_channels = 32
        else:
            input_channels = 64
        expected[f"layer{block}_{unit}_conv1_weight.txt"] = (
            3 * 3 * input_channels * channels
        )
        expected[f"layer{block}_{unit}_conv2_weight.txt"] = (
            3 * 3 * channels * channels
        )
        for field in (
            "bn1_bias", "bn1_running_mean", "bn1_running_var", "bn1_weight",
            "bn2_bias", "bn2_running_mean", "bn2_running_var", "bn2_weight",
        ):
            expected[f"layer{block}_{unit}_{field}.txt"] = channels

weight_records = []
for name, expected_count in sorted(expected.items()):
    path = weights_root / name
    observed = numeric_tokens(path, float)
    if len(observed) != expected_count:
        raise SystemExit(
            f"weight {path} has {len(observed)} values; expected {expected_count}"
        )
    weight_records.append({
        "path": str(path.relative_to(root)),
        "sha256": digest(path),
        "value_count": len(observed),
    })

record = {
    "schema_version": "ace.phantom.resnet20-data-preflight/1.0.0",
    "status": "pass",
    "dataset": {
        "image_count": image_count,
        "pixels_per_image": pixels_per_image,
        "requested_range": [start_image, end_image],
        "requested_labels": labels[start_image : end_image + 1],
        "values_sha256": digest(values_path),
        "labels_sha256": digest(labels_path),
    },
    "weights": weight_records,
}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(
    f"verified CIFAR-10 fixture ({image_count} images) and "
    f"{len(weight_records)} ResNet-20 parameter files"
)
PY

if [[ "${GENERATE}" == true ]]; then
  if [[ "${ACE_PHANTOM_SOURCE_MODE:-git}" != snapshot ]]; then
    die "--generate requires the audited compile_only snapshot environment (ACE_PHANTOM_SOURCE_MODE=snapshot)"
  fi
  "${SCRIPT_DIR}/compile_only.sh" \
    --gate bootstrap \
    --poly-degree 65536 \
    --vector-capacity 32768 \
    --mul-level 32 \
    --input-level 1 \
    --security-level 0 \
    --scaling-factor-bits 56 \
    --first-prime-bits 60 \
    --hamming-weight 192 \
    --q-part-count 3 \
    --encode-transform-budget 3 \
    --decode-transform-budget 3 \
    --ciphertext-constant-encoding disabled \
    --packing full \
    --post-multiply-real -1.0 \
    --post-multiply-imag 0.0 \
    --post-multiply-scale-degree 0 \
    --post-rotation-step 8 \
    --fixture-id generated-bootstrap-full-packed-n65536-resnet-q32 \
    --fixture-seed 7640891576956012809 \
    --inside-margin 0.125 \
    --provider-clear-threshold 0.01 \
    --gpu-native-threshold 0.02 \
    --gpu-generated-threshold 0.02 \
    --repeat-threshold 0.000001 \
    --host-oracle-timeout-seconds 3600
  STATE_ROOT="${ACE_PHANTOM_STATE_ROOT:-${REPO_ROOT}/build/phantom_gpu}"
  LATEST_RECORD="${STATE_ROOT}/compile_only_results/latest-success-bootstrap.json"
  require_file "${LATEST_RECORD}"
  QUALIFICATION_DIR="$(
    python3 - "${LATEST_RECORD}" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("status") != "pass" or record.get("gate") != "bootstrap":
    raise SystemExit("latest bootstrap qualification did not pass")
print(record["run_root"])
PY
  )"
fi

STRICT_QUALIFICATION=true
ARTIFACT_PROVENANCE="compile-only-qualified"
if [[ -n "${GENERATED_DIR}" ]]; then
  STRICT_QUALIFICATION=false
  ARTIFACT_PROVENANCE="development-generated-unqualified"
  [[ -d "${GENERATED_DIR}" ]] ||
    die "generated directory does not exist: ${GENERATED_DIR}"
  GENERATED_DIR="$(cd -- "${GENERATED_DIR}" && pwd -P)"
  if [[ -d "${GENERATED_DIR}/bootstrap_qualification" ]]; then
    BOOTSTRAP_DIR="${GENERATED_DIR}/bootstrap_qualification"
  elif [[ "$(basename -- "${GENERATED_DIR}")" == bootstrap_qualification ]]; then
    BOOTSTRAP_DIR="${GENERATED_DIR}"
  else
    die "--generated-dir must name bootstrap_qualification or its parent"
  fi
  QUALIFICATION_ROOT=""
  printf '%s\n' \
    "WARNING: --generated-dir is development-only generated output." \
    "It has not passed compile_only host qualification and must not be reported as qualified." \
    >&2
else
  [[ -d "${QUALIFICATION_DIR}" ]] ||
    die "qualification directory does not exist: ${QUALIFICATION_DIR}"
  QUALIFICATION_DIR="$(cd -- "${QUALIFICATION_DIR}" && pwd -P)"
  if [[ -d "${QUALIFICATION_DIR}/bootstrap_qualification" ]]; then
    QUALIFICATION_ROOT="${QUALIFICATION_DIR}"
    BOOTSTRAP_DIR="${QUALIFICATION_DIR}/bootstrap_qualification"
  elif [[ "$(basename -- "${QUALIFICATION_DIR}")" == bootstrap_qualification ]]; then
    QUALIFICATION_ROOT="$(cd -- "${QUALIFICATION_DIR}/.." && pwd -P)"
    BOOTSTRAP_DIR="${QUALIFICATION_DIR}"
  else
    die "qualification path must be a compile_only run root or its bootstrap_qualification directory"
  fi
fi

for required in \
  "${BOOTSTRAP_DIR}/bootstrap_qualification.cu" \
  "${BOOTSTRAP_DIR}/compiler_context_manifest.json" \
  "${BOOTSTRAP_DIR}/compiler_resource_manifest.json" \
  "${BOOTSTRAP_DIR}/compiler_constant_manifest.json" \
  "${BOOTSTRAP_DIR}/generation.json"; do
  require_file "${required}"
done

if [[ "${STRICT_QUALIFICATION}" == true ]]; then
  for required in \
    "${QUALIFICATION_ROOT}/SHA256SUMS" \
    "${QUALIFICATION_ROOT}/artifact_manifest.json" \
    "${QUALIFICATION_ROOT}/manifest.json" \
    "${QUALIFICATION_ROOT}/qualification_invocation.json"; do
    require_file "${required}"
  done
  echo "Verifying the complete qualification checksum closure..."
  (
    cd -- "${QUALIFICATION_ROOT}"
    sha256sum --check --strict SHA256SUMS
  ) >"${BUILD_DIR}/qualification-checksums.txt"
  ARTIFACT_CHECKSUM_RECORD="${BUILD_DIR}/qualification-checksums.txt"
else
  for required in \
    "${BOOTSTRAP_DIR}/compiler_invocation.json" \
    "${BOOTSTRAP_DIR}/bootstrap_raw.air" \
    "${BOOTSTRAP_DIR}/bootstrap_post_ckks.air" \
    "${BOOTSTRAP_DIR}/bootstrap_post_operations.air" \
    "${BOOTSTRAP_DIR}/bootstrap_semantics.json" \
    "${BOOTSTRAP_DIR}/post_operations_attestation.json"; do
    require_file "${required}"
  done
  (
    cd -- "${BOOTSTRAP_DIR}"
    sha256sum -- \
      bootstrap_qualification.cu \
      bootstrap_raw.air bootstrap_post_ckks.air bootstrap_post_operations.air \
      bootstrap_semantics.json compiler_context_manifest.json \
      compiler_resource_manifest.json compiler_constant_manifest.json \
      compiler_invocation.json generation.json post_operations_attestation.json
  ) >"${BUILD_DIR}/development-generated-sha256sums.txt"
  ARTIFACT_CHECKSUM_RECORD="${BUILD_DIR}/development-generated-sha256sums.txt"
fi

python3 - "${BOOTSTRAP_DIR}" "${QUALIFICATION_ROOT}" \
  "${EXPECTED_PHANTOM_COMMIT}" "${STRICT_QUALIFICATION}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

qdir = Path(sys.argv[1])
root = Path(sys.argv[2]) if sys.argv[2] else None
expected_phantom = sys.argv[3]
strict_qualification = sys.argv[4] == "true"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def require(condition, message):
    if not condition:
        raise SystemExit(f"Q32 qualification rejected: {message}")

context_path = qdir / "compiler_context_manifest.json"
resource_path = qdir / "compiler_resource_manifest.json"
constant_path = qdir / "compiler_constant_manifest.json"
generation_path = qdir / "generation.json"
source_path = qdir / "bootstrap_qualification.cu"
artifact_path = root / "artifact_manifest.json" if root else None
manifest_path = root / "manifest.json" if root else None
invocation_path = root / "qualification_invocation.json" if root else None

context = load(context_path)
resource = load(resource_path)
constants = load(constant_path)
generation = load(generation_path)
artifact = load(artifact_path) if strict_qualification else None
manifest = load(manifest_path) if strict_qualification else None
invocation = load(invocation_path) if strict_qualification else None

data_q = context.get("data_q_bit_sizes", [])
special_p = context.get("special_p_bit_sizes", [])
if len(data_q) in (26, 31):
    raise SystemExit(
        f"Q{len(data_q)} helper rejected: handwritten ReLU requires Q32 -> Q17"
    )
require(data_q == [60] + [56] * 31, "data-Q chain is not Q32 with q0=60/sf=56")
require(special_p == [60] * 11, "special-prime chain is not P11")
require(context.get("polynomial_degree") == 65536, "polynomial degree is not 65536")
require(context.get("logical_slot_capacity") == 32768, "slot capacity is not 32768")
require(context.get("packing") == "full", "packing is not full")
require(context.get("input_level") == 1, "input level is not 1")
require(context.get("q_part_count") == 3, "q-part count is not 3")
require(context.get("hamming_weight") == 192, "hamming weight is not 192")
require(context.get("security_level") == 0, "security level is not 0")
require(context.get("first_modulus_bits") == 60, "q0 is not 60 bits")
require(context.get("scaling_modulus_bits") == 56, "scaling modulus is not 56 bits")
require(context.get("resource_schema_version") == 3, "resource schema is not version 3")

parameters = generation.get("compiler_parameters", {})
expected_parameters = {
    "ciphertext_constant_encoding": "disabled",
    "clear_imag": True,
    "decode_transform_budget": 3,
    "encode_transform_budget": 3,
    "first_prime_bits": 60,
    "hamming_weight": 192,
    "input_level": 1,
    "mul_level": 32,
    "packing": "full",
    "poly_degree": 65536,
    "post_multiply_imag": 0.0,
    "post_multiply_real": -1.0,
    "post_multiply_scale_degree": 0,
    "post_rotation_step": 8,
    "q_part_count": 3,
    "scaling_factor_bits": 56,
    "security_level": 0,
    "vector_capacity": 32768,
}
require(parameters == expected_parameters, "generation profile differs from the fixed ResNet Q32 profile")
bootstrap_parameters = generation.get("bootstrap_parameters", {})
require(
    bootstrap_parameters == {
        "clear_imag": True,
        "ct_encode": False,
        "dec_budget": 3,
        "enc_budget": 3,
        "q_parts": 3,
    },
    "bootstrap decomposition parameters differ",
)
require(generation.get("status") == "pass", "generation status is not pass")
require(generation.get("constant_count", 0) > 0, "generated helper has no constants")

require(resource.get("schema_version") == 3, "resource manifest schema differs")
for flag in (
    "complex_plaintext",
    "conjugation_key",
    "raise_mod",
    "relinearization_key",
    "rotate_batch",
):
    require(resource.get(flag) is True, f"resource requirement {flag} is absent")
require(resource.get("native_bootstrap_precompute") is False, "helper requests native bootstrap")

context_sha = digest(context_path)
resource_sha = digest(resource_path)
constant_sha = digest(constant_path)
source_sha = digest(source_path)
require(constants.get("context_manifest_sha256") == context_sha, "constant/context digest binding differs")
entries = constants.get("constants", [])
require(entries and all(entry.get("entry_id") == index for index, entry in enumerate(entries)),
        "constant entry IDs are not dense and ordered")
require(len(entries) == generation.get("constant_count"),
        "generation constant count differs from the manifest")

def verify_generation_ref(record, label):
    path = qdir / record.get("path", "")
    require(path.is_file(), f"generation binding {label} names a missing file")
    require(record.get("bytes") == path.stat().st_size,
            f"generation binding {label} has the wrong byte count")
    require(record.get("sha256") == digest(path),
            f"generation binding {label} has the wrong digest")

verify_generation_ref(generation.get("source", {}), "source")
for label, record in generation.get("manifests", {}).items():
    verify_generation_ref(record, f"manifest/{label}")
verify_generation_ref(
    generation.get("normalized_compiler_invocation", {}), "compiler invocation"
)
verify_generation_ref(generation.get("bootstrap_semantics", {}), "semantics")
verify_generation_ref(generation.get("post_operations_air", {}), "post operations AIR")
verify_generation_ref(
    generation.get("post_operations_attestation", {}), "post operations attestation"
)
for label, record in generation.get("air", {}).items():
    verify_generation_ref(record, f"AIR/{label}")

if strict_qualification:
    require(artifact.get("status") == "bound", "artifact manifest is not bound")
    require(artifact.get("phantom_commit") == expected_phantom, "artifact has the wrong Phantom commit")
    require(artifact.get("source_mode") == "snapshot", "artifact was not made from audited snapshots")
    files = artifact.get("files", {})
    expected_files = {
        "bootstrap_qualification/compiler_context_manifest.json": context_sha,
        "bootstrap_qualification/compiler_resource_manifest.json": resource_sha,
        "bootstrap_qualification/compiler_constant_manifest.json": constant_sha,
        "bootstrap_qualification/bootstrap_qualification.cu": source_sha,
    }
    for name, expected in expected_files.items():
        require(files.get(name) == expected, f"artifact digest binding differs for {name}")
    require(artifact.get("compiler_context_manifest_sha256") == context_sha,
            "artifact/context digest binding differs")
    require(artifact.get("compiler_resource_manifest_sha256") == resource_sha,
            "artifact/resource digest binding differs")
    require(artifact.get("compiler_constant_manifest_sha256") == constant_sha,
            "artifact/constant digest binding differs")

    require(manifest.get("status") == "pass", "qualification manifest did not pass")
    require(manifest.get("phantom_commit") == expected_phantom, "qualification manifest has the wrong Phantom commit")
    require(manifest.get("artifact_manifest_sha256") == digest(artifact_path),
            "qualification/artifact digest binding differs")
    require(manifest.get("evidence_sha256_manifest_sha256") == digest(root / "SHA256SUMS"),
            "qualification checksum-manifest binding differs")

    argv = invocation.get("argv", [])
    require(argv and argv[0] == "tools/phantom_gpu/compile_only.sh", "qualification invocation is not compile_only")
    require(len(argv[1:]) % 2 == 0, "qualification invocation is malformed")
    arguments = dict(zip(argv[1::2], argv[2::2]))
    require(arguments.get("--gate") == "bootstrap", "qualification gate is not bootstrap")
    require(arguments.get("--mul-level") == "32", "qualification was not invoked with Q32")
    require(arguments.get("--poly-degree") == "65536", "qualification invocation has the wrong N")
    require(arguments.get("--vector-capacity") == "32768", "qualification invocation has the wrong slots")
else:
    compiler_invocation = load(qdir / "compiler_invocation.json")
    require(compiler_invocation.get("status") == "pass",
            "development compiler invocation did not pass")
    require(compiler_invocation.get("options") == expected_parameters,
            "development compiler invocation profile differs")
    require(
        compiler_invocation.get("normalized_argv", [])[-1:] == ["--clear-imag"],
        "development compiler invocation is not explicitly clear-imag",
    )
    semantics = load(qdir / "bootstrap_semantics.json")
    expanded = semantics.get("expanded_bootstrap", {})
    require(expanded.get("clear_imag") is True,
            "development semantics do not require clear-imag")
    restoration = semantics.get("restoration_attestation", {})
    projection = restoration.get("real_projection", {})
    require(
        restoration.get("kind") ==
        "terminal-real-projection-and-ciphertext-self-add-chain"
        and projection.get("kind") == "terminal-conjugate-real-projection"
        and projection.get("projected_component") == "real"
        and projection.get("caller_proof_required") is True,
        "development semantics do not attest the terminal real projection",
    )
    require(generation.get("generated_program_executed") is False,
            "development generation unexpectedly claims execution")

print(
    ("verified qualified" if strict_qualification else "verified UNQUALIFIED generated") +
    " Q32/P11 helper: "
    f"context={context_sha} resource={resource_sha} constants={constant_sha} source={source_sha}"
)
PY

HELPER_SOURCE="${BOOTSTRAP_DIR}/bootstrap_qualification.cu"
printf 'artifact_provenance=%s\nbootstrap_dir=%s\n' \
  "${ARTIFACT_PROVENANCE}" "${BOOTSTRAP_DIR}" >"${BUILD_DIR}/artifact-provenance.txt"
rg -q 'Raise_mod\([^;]*,[[:space:]]*32\);' "${HELPER_SOURCE}" ||
  die "generated helper does not raise to Q32"
if rg -q 'Raise_mod\([^;]*,[[:space:]]*(26|31)\);' "${HELPER_SOURCE}"; then
  die "generated helper contains a forbidden Q26/Q31 raise target"
fi

if [[ -n "${CUDACXX:-}" ]]; then
  if [[ "${CUDACXX}" == */* ]]; then
    NVCC="${CUDACXX}"
  else
    NVCC="$(command -v -- "${CUDACXX}" 2>/dev/null || true)"
  fi
  [[ -n "${NVCC}" && -x "${NVCC}" ]] || die "CUDACXX is not executable: ${CUDACXX}"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
else
  die "nvcc is required to build the SM80 executable"
fi
CUDA_ROOT="$(cd -- "$(dirname -- "${NVCC}")/.." && pwd -P)"
require_file "${CUDA_ROOT}/include/cuda_runtime.h"
require_file "${CUDA_ROOT}/lib64/libcudadevrt.a"

RUNTIME_BUILD="${BUILD_DIR}/ace-runtime"
OBJECT_DIR="${BUILD_DIR}/objects"
BIN_DIR="${BUILD_DIR}/bin"
mkdir -p -- "${RUNTIME_BUILD}" "${OBJECT_DIR}" "${BIN_DIR}"
COMMAND_LOG="${BUILD_DIR}/commands.txt"
: >"${COMMAND_LOG}"

run_logged() {
  printf '%q ' "$@" >>"${COMMAND_LOG}"
  printf '\n' >>"${COMMAND_LOG}"
  "$@"
}

PACKAGE_INCLUDES="${REPO_ROOT}/air-infra/include|${REPO_ROOT}/nn-addon/include|${REPO_ROOT}/fhe-cmplr/include|${REPO_ROOT}/fhe-cmplr/rtlib/include"
run_logged cmake \
  -S "${REPO_ROOT}/fhe-cmplr/rtlib" \
  -B "${RUNTIME_BUILD}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER="${NVCC}" \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DPACKAGE_BASE_DIR="${REPO_ROOT}/air-infra" \
  -DPACKAGE_INC_DIR_TF="${PACKAGE_INCLUDES}" \
  -DBUILD_STATIC=ON \
  -DBUILD_UNITTEST=OFF \
  -DBUILD_BENCH=OFF \
  -DRTLIB_BUILD_TEST=OFF \
  -DRTLIB_BUILD_EXAMPLE=OFF \
  -DRTLIB_INSTALL_APP=OFF \
  -DRTLIB_CODE_CHECK=OFF \
  -DRTLIB_ENABLE_CUDA=ON \
  -DRTLIB_ENABLE_PHANTOM=ON \
  -DRTLIB_ENABLE_SEAL=OFF \
  -DRTLIB_ENABLE_SEAL_BTS=OFF \
  -DRTLIB_ENABLE_OPENFHE=OFF \
  -DBUILD_WITH_OPENMP=OFF \
  -DPHANTOM_SOURCE_DIR="${PHANTOM_DIR}" \
  -DPHANTOM_GIT_TAG="${EXPECTED_PHANTOM_COMMIT}" \
  -DPHANTOM_SOURCE_SNAPSHOT=OFF
run_logged cmake --build "${RUNTIME_BUILD}" --target FHErt_phantom --parallel "${BUILD_JOBS}"

EXTERNAL_SOURCE="${RUNTIME_BUILD}/external/src/phantom_external"
EXTERNAL_BUILD="${RUNTIME_BUILD}/external/src/phantom_external-build"
[[ -d "${EXTERNAL_SOURCE}/.git" ]] || die "ACE runtime did not create the pinned Phantom clone"
EXTERNAL_HEAD="$(git -C "${EXTERNAL_SOURCE}" rev-parse HEAD)"
[[ "${EXTERNAL_HEAD}" == "${EXPECTED_PHANTOM_COMMIT}" ]] ||
  die "ACE runtime external Phantom is ${EXTERNAL_HEAD}, expected ${EXPECTED_PHANTOM_COMMIT}; use a fresh --build-dir"
[[ -z "$(git -C "${EXTERNAL_SOURCE}" status --porcelain --untracked-files=all)" ]] ||
  die "ACE runtime external Phantom clone is dirty"
run_logged cmake --build "${EXTERNAL_BUILD}" \
  --target phantom_ordinary phantom_native_bts_oracle \
  --parallel "${BUILD_JOBS}"

ADAPTER_ARCHIVE="${RUNTIME_BUILD}/phantom/libFHErt_phantom.a"
COMMON_ARCHIVE="${RUNTIME_BUILD}/common/libFHErt_common.a"
ORDINARY_ARCHIVE="${EXTERNAL_BUILD}/lib/libphantom_ordinary.a"
NATIVE_ARCHIVE="${EXTERNAL_BUILD}/lib/libphantom_native_bts_oracle.a"
for archive in "${ADAPTER_ARCHIVE}" "${COMMON_ARCHIVE}" "${ORDINARY_ARCHIVE}" "${NATIVE_ARCHIVE}"; do
  require_file "${archive}"
done
[[ "$(basename -- "${ORDINARY_ARCHIVE}")" == libphantom_ordinary.a ]] ||
  die "ordinary provider archive has an unexpected name"
[[ "$(basename -- "${NATIVE_ARCHIVE}")" == libphantom_native_bts_oracle.a ]] ||
  die "native provider archive has an unexpected name"

python3 - "${ORDINARY_ARCHIVE}" "${NATIVE_ARCHIVE}" <<'PY'
import subprocess
import sys

ordinary = set(subprocess.check_output(["ar", "t", sys.argv[1]], text=True).splitlines())
native = set(subprocess.check_output(["ar", "t", sys.argv[2]], text=True).splitlines())
overlap = sorted(ordinary & native)
if overlap:
    raise SystemExit("ordinary/native provider archives overlap: " + ", ".join(overlap))
PY

NVCC_COMMON=(
  -std=c++17
  -arch=sm_80
  -rdc=true
  -O3
  -DNDEBUG
  -I"${REPO_ROOT}/fhe-cmplr/rtlib/include"
  -I"${EXTERNAL_SOURCE}/include"
)

HELPER_OBJECT="${OBJECT_DIR}/bootstrap_qualification.o"
BRIDGE_OBJECT="${OBJECT_DIR}/run_cnn_dsl_bootstrap_bridge.o"
IO_METADATA_OBJECT="${OBJECT_DIR}/run_cnn_phantom_io_metadata.o"
DSL_RUN_CNN_OBJECT="${OBJECT_DIR}/run_cnn.dsl.o"
NATIVE_RUN_CNN_OBJECT="${OBJECT_DIR}/run_cnn.matched-native-slim.o"
DSL_DEVICE_LINK_OBJECT="${OBJECT_DIR}/run_cnn_dsl_bootstrap.dlink.o"
NATIVE_DEVICE_LINK_OBJECT="${OBJECT_DIR}/run_cnn_matched_native_slim.dlink.o"
DSL_BINARY="${BIN_DIR}/run_cnn_dsl_bootstrap"
NATIVE_BINARY="${BIN_DIR}/run_cnn_matched_native_slim"
BUILT_BINARIES=()

run_logged "${NVCC}" "${NVCC_COMMON[@]}" -dc \
  "${HELPER_SOURCE}" -o "${HELPER_OBJECT}"
run_logged c++ -std=c++17 -O3 -DNDEBUG \
  -I"${REPO_ROOT}/fhe-cmplr/rtlib/include" -c \
  "${IO_METADATA_SOURCE}" -o "${IO_METADATA_OBJECT}"

if [[ "${BUILD_DSL}" == true ]]; then
  run_logged "${NVCC}" "${NVCC_COMMON[@]}" -dc \
    "${BRIDGE_SOURCE}" -o "${BRIDGE_OBJECT}"
  run_logged "${NVCC}" "${NVCC_COMMON[@]}" \
    -DACE_PHANTOM_RUN_CNN_DSL_BOOTSTRAP=1 -Xcompiler=-fopenmp -dc \
    "${RUN_CNN_SOURCE}" -o "${DSL_RUN_CNN_OBJECT}"

  run_logged "${NVCC}" -std=c++17 -arch=sm_80 -rdc=true -dlink \
    "${HELPER_OBJECT}" "${BRIDGE_OBJECT}" "${DSL_RUN_CNN_OBJECT}" \
    "${ADAPTER_ARCHIVE}" "${NATIVE_ARCHIVE}" "${ORDINARY_ARCHIVE}" \
    "${COMMON_ARCHIVE}" \
    -L"${CUDA_ROOT}/lib64" -lcudadevrt \
    -o "${DSL_DEVICE_LINK_OBJECT}"

  run_logged c++ -std=c++17 -O3 -DNDEBUG \
    "${HELPER_OBJECT}" "${BRIDGE_OBJECT}" "${IO_METADATA_OBJECT}" \
    "${DSL_RUN_CNN_OBJECT}" \
    "${DSL_DEVICE_LINK_OBJECT}" \
    -Wl,--start-group \
    "${ADAPTER_ARCHIVE}" "${NATIVE_ARCHIVE}" "${ORDINARY_ARCHIVE}" \
    "${COMMON_ARCHIVE}" \
    -lntl -lgmpxx -lgmp \
    -Wl,--end-group \
    -L"${CUDA_ROOT}/lib64" -Wl,-rpath,"${CUDA_ROOT}/lib64" \
    -lcudadevrt -lcudart -pthread -fopenmp -ldl -lrt -lm \
    -o "${DSL_BINARY}"

  nm -C --defined-only "${DSL_BINARY}" >"${BUILD_DIR}/linked-symbols-dsl.txt"
  rg -q 'bootstrap_full\(PhantomCiphertext, PhantomCiphertext\)' \
    "${BUILD_DIR}/linked-symbols-dsl.txt" ||
    die "DSL executable does not define the generated bootstrap"
  rg -q 'Ace_phantom_run_cnn_dsl_bootstrap' \
    "${BUILD_DIR}/linked-symbols-dsl.txt" ||
    die "DSL executable does not define the native/DSL bridge"
  rg -q 'Phantom_borrow_runtime' "${BUILD_DIR}/linked-symbols-dsl.txt" ||
    die "DSL executable does not contain the borrowed-runtime adapter API"
  for callback in Get_input_count Get_output_count Get_encode_scheme Get_decode_scheme; do
    rg -q " ${callback}$" "${BUILD_DIR}/linked-symbols-dsl.txt" ||
      die "DSL executable does not define ${callback}"
  done
  BUILT_BINARIES+=("${DSL_BINARY}")
fi

if [[ "${BUILD_NATIVE}" == true ]]; then
  run_logged "${NVCC}" "${NVCC_COMMON[@]}" \
    -DACE_PHANTOM_RUN_CNN_MATCHED_NATIVE_SLIM=1 -Xcompiler=-fopenmp -dc \
    "${RUN_CNN_SOURCE}" -o "${NATIVE_RUN_CNN_OBJECT}"

  # The generated object is linked here on purpose: it is the sole owner of
  # the audited context/resource/constant manifests used by Prepare_context.
  # There is no bridge object and this mode's run_cnn path calls Slim directly.
  run_logged "${NVCC}" -std=c++17 -arch=sm_80 -rdc=true -dlink \
    "${HELPER_OBJECT}" "${NATIVE_RUN_CNN_OBJECT}" \
    "${ADAPTER_ARCHIVE}" "${NATIVE_ARCHIVE}" "${ORDINARY_ARCHIVE}" \
    "${COMMON_ARCHIVE}" \
    -L"${CUDA_ROOT}/lib64" -lcudadevrt \
    -o "${NATIVE_DEVICE_LINK_OBJECT}"

  run_logged c++ -std=c++17 -O3 -DNDEBUG \
    "${HELPER_OBJECT}" "${IO_METADATA_OBJECT}" "${NATIVE_RUN_CNN_OBJECT}" \
    "${NATIVE_DEVICE_LINK_OBJECT}" \
    -Wl,--start-group \
    "${ADAPTER_ARCHIVE}" "${NATIVE_ARCHIVE}" "${ORDINARY_ARCHIVE}" \
    "${COMMON_ARCHIVE}" \
    -lntl -lgmpxx -lgmp \
    -Wl,--end-group \
    -L"${CUDA_ROOT}/lib64" -Wl,-rpath,"${CUDA_ROOT}/lib64" \
    -lcudadevrt -lcudart -pthread -fopenmp -ldl -lrt -lm \
    -o "${NATIVE_BINARY}"

  nm -C --defined-only "${NATIVE_BINARY}" >"${BUILD_DIR}/linked-symbols-native.txt"
  rg -q 'bootstrap_full\(PhantomCiphertext, PhantomCiphertext\)' \
    "${BUILD_DIR}/linked-symbols-native.txt" ||
    die "matched-native executable does not carry the shared generated manifest/constants object"
  rg -q 'Phantom_borrow_runtime' "${BUILD_DIR}/linked-symbols-native.txt" ||
    die "matched-native executable does not contain the borrowed-runtime adapter API"
  rg -q 'Bootstrapper::slim_bootstrap' "${BUILD_DIR}/linked-symbols-native.txt" ||
    die "matched-native executable does not contain Phantom Slim"
  for callback in Get_input_count Get_output_count Get_encode_scheme Get_decode_scheme; do
    rg -q " ${callback}$" "${BUILD_DIR}/linked-symbols-native.txt" ||
      die "matched-native executable does not define ${callback}"
  done
  if rg -q 'Ace_phantom_run_cnn_dsl_bootstrap' \
    "${BUILD_DIR}/linked-symbols-native.txt"; then
    die "matched-native executable unexpectedly defines the DSL bridge"
  fi
  BUILT_BINARIES+=("${NATIVE_BINARY}")
fi

CHECKSUM_INPUTS=(
  "${HELPER_SOURCE}"
  "${RUN_CNN_SOURCE}"
  "${IO_METADATA_SOURCE}"
  "${SUMMARY_SOURCE}"
  "${BUILD_DIR}/artifact-provenance.txt"
  "${BUILD_DIR}/resnet20-data-preflight.json"
  "${ARTIFACT_CHECKSUM_RECORD}"
  "${ADAPTER_ARCHIVE}"
  "${COMMON_ARCHIVE}"
  "${ORDINARY_ARCHIVE}"
  "${NATIVE_ARCHIVE}"
  "${BUILT_BINARIES[@]}"
)
if [[ "${BUILD_DSL}" == true ]]; then
  CHECKSUM_INPUTS+=("${BRIDGE_SOURCE}")
fi
sha256sum "${CHECKSUM_INPUTS[@]}" >"${BUILD_DIR}/build-sha256sums.txt"

printf 'Built %s\n' "${BUILT_BINARIES[@]}"
echo "Artifact provenance: ${ARTIFACT_PROVENANCE}"
if [[ "${STRICT_QUALIFICATION}" == true ]]; then
  echo "Qualification: ${QUALIFICATION_ROOT}"
else
  echo "UNQUALIFIED generated helper: ${BOOTSTRAP_DIR}"
fi
echo "Phantom source: ${PHANTOM_DIR} @ ${PHANTOM_HEAD}"
if [[ "${BUILD_ONLY}" == true ]]; then
  git -C "${REPO_ROOT}" status --short >"${BUILD_DIR}/ace-status-after.txt"
  git -C "${PHANTOM_DIR}" status --short >"${BUILD_DIR}/phantom-status-after.txt" \
    2>"${BUILD_DIR}/phantom-status-after.stderr.txt" || true
  exit 0
fi

require_command nvidia-smi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | rg -q 'A100'; then
  die "execution requires an NVIDIA A100; use --build-only on a non-A100 host"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_ROOT="${BUILD_DIR}/runs/${RUN_ID}"
RUN_BUILD_EVIDENCE="${RUN_ROOT}/build-evidence"
mkdir -p -- "${RUN_BUILD_EVIDENCE}"
nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,memory.free \
  --format=csv,noheader >"${RUN_BUILD_EVIDENCE}/gpu-environment.txt"
BUILD_EVIDENCE_FILES=(
  artifact-provenance.txt
  build-sha256sums.txt
  commands.txt
  ace-status-before.txt
  phantom-status-before.txt
  phantom-head.txt
  resnet20-data-preflight.json
)
[[ -f "${BUILD_DIR}/qualification-checksums.txt" ]] &&
  BUILD_EVIDENCE_FILES+=(qualification-checksums.txt)
[[ -f "${BUILD_DIR}/development-generated-sha256sums.txt" ]] &&
  BUILD_EVIDENCE_FILES+=(development-generated-sha256sums.txt)
[[ -f "${BUILD_DIR}/linked-symbols-dsl.txt" ]] &&
  BUILD_EVIDENCE_FILES+=(linked-symbols-dsl.txt)
[[ -f "${BUILD_DIR}/linked-symbols-native.txt" ]] &&
  BUILD_EVIDENCE_FILES+=(linked-symbols-native.txt)
for evidence_file in "${BUILD_EVIDENCE_FILES[@]}"; do
  cp -- "${BUILD_DIR}/${evidence_file}" "${RUN_BUILD_EVIDENCE}/"
done
(
  cd -- "${RUN_BUILD_EVIDENCE}"
  sha256sum -- * >SHA256SUMS
)

run_and_capture() {
  local label="$1"
  local binary="$2"
  local capture_dir="${RUN_ROOT}/${label}"
  local run_log="${capture_dir}/run.log"
  local run_marker="${capture_dir}/run-start.marker"
  local -a pipeline_status=()

  mkdir -p -- "${capture_dir}"
  : >"${run_marker}"
  echo "Running ${label} ResNet-20/CIFAR-10 images ${START_IMAGE}..${END_IMAGE} from ${BIN_DIR}"
  set +e
  (
    cd -- "${BIN_DIR}"
    "${binary}" 20 10 "${START_IMAGE}" "${END_IMAGE}"
  ) 2>&1 | tee "${run_log}"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  printf 'program_exit=%s\ntee_exit=%s\n' \
    "${pipeline_status[0]}" "${pipeline_status[1]}" >"${capture_dir}/run-status.txt"
  if ((pipeline_status[0] != 0 || pipeline_status[1] != 0)); then
    (
      cd -- "${capture_dir}"
      sha256sum -- run.log run-status.txt >SHA256SUMS
    )
    echo "${label} execution failed; retained checksum-closed log at ${capture_dir}" >&2
    return 1
  fi

  local image_id
  local result_file
  for ((image_id = START_IMAGE; image_id <= END_IMAGE; ++image_id)); do
    result_file="${PHANTOM_DIR}/result/resnet20_cifar10_image${image_id}.txt"
    if [[ ! -f "${result_file}" || ! "${result_file}" -nt "${run_marker}" ]]; then
      echo "run did not freshly write ${result_file}" \
        >"${capture_dir}/validation-error.txt"
      (
        cd -- "${capture_dir}"
        sha256sum -- run.log run-status.txt validation-error.txt >SHA256SUMS
      )
      return 1
    fi
    cp -- "${result_file}" "${capture_dir}/"
  done
  result_file="${PHANTOM_DIR}/result/resnet20_cifar10_label_${START_IMAGE}_${END_IMAGE}"
  if [[ ! -f "${result_file}" || ! "${result_file}" -nt "${run_marker}" ]]; then
    echo "run did not freshly write ${result_file}" \
      >"${capture_dir}/validation-error.txt"
    (
      cd -- "${capture_dir}"
      sha256sum -- run.log run-status.txt validation-error.txt >SHA256SUMS
    )
    return 1
  fi
  cp -- "${result_file}" "${capture_dir}/"

  summary_args=(
    python3 "${SUMMARY_SOURCE}"
    --mode "${label}"
    --start-image "${START_IMAGE}"
    --end-image "${END_IMAGE}"
    --run-log "${run_log}"
    --result-dir "${capture_dir}"
    --output "${capture_dir}/summary.json"
  )
  if ((START_IMAGE == 0 && END_IMAGE == 0)); then
    summary_args+=(--require-correct)
  fi
  if ! "${summary_args[@]}" 2>"${capture_dir}/validation-error.txt"; then
    (
      cd -- "${capture_dir}"
      failed_files=(run.log run-status.txt validation-error.txt)
      failed_files+=("resnet20_cifar10_label_${START_IMAGE}_${END_IMAGE}")
      for ((image_id = START_IMAGE; image_id <= END_IMAGE; ++image_id)); do
        failed_files+=("resnet20_cifar10_image${image_id}.txt")
      done
      sha256sum -- "${failed_files[@]}" >SHA256SUMS
    )
    return 1
  fi
  [[ ! -s "${capture_dir}/validation-error.txt" ]] || return 1
  rm -- "${capture_dir}/validation-error.txt"
  (
    cd -- "${capture_dir}"
    checksum_files=(
      "run.log"
      "run-status.txt"
      "summary.json"
      "resnet20_cifar10_label_${START_IMAGE}_${END_IMAGE}"
    )
    for ((image_id = START_IMAGE; image_id <= END_IMAGE; ++image_id)); do
      checksum_files+=("resnet20_cifar10_image${image_id}.txt")
    done
    sha256sum -- "${checksum_files[@]}" >SHA256SUMS
  )
}

RUN_FAILED=false
if [[ "${BUILD_DSL}" == true ]]; then
  if ! run_and_capture dsl "${DSL_BINARY}"; then
    RUN_FAILED=true
  fi
fi
if [[ "${BUILD_NATIVE}" == true ]]; then
  if ! run_and_capture matched-native-slim "${NATIVE_BINARY}"; then
    RUN_FAILED=true
  fi
fi

if [[ "${BUILD_DSL}" == true && "${BUILD_NATIVE}" == true &&
  -f "${RUN_ROOT}/dsl/summary.json" &&
  -f "${RUN_ROOT}/matched-native-slim/summary.json" ]]; then
  if ! python3 - "${RUN_ROOT}" 2>"${RUN_ROOT}/comparison-error.txt" <<'PY'; then
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
dsl = json.loads((root / "dsl" / "summary.json").read_text(encoding="utf-8"))
native = json.loads(
    (root / "matched-native-slim" / "summary.json").read_text(encoding="utf-8")
)
if dsl["image_range"] != native["image_range"]:
    raise SystemExit("DSL/native image ranges differ")
dsl_results = [(row["clear_label"], row["inferred_label"]) for row in dsl["results"]]
native_results = [
    (row["clear_label"], row["inferred_label"]) for row in native["results"]
]
if dsl_results != native_results:
    raise SystemExit("DSL and matched Native Slim classification results differ")
comparison = {
    "schema_version": "ace.phantom.resnet20-e2e-comparison/1.0.0",
    "status": "pass",
    "image_range": dsl["image_range"],
    "classification_results_equal": True,
    "dsl_bootstrap_time_ms": dsl["bootstrap_time_ms"],
    "matched_native_slim_bootstrap_time_ms": native["bootstrap_time_ms"],
}
(root / "comparison-summary.json").write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
    RUN_FAILED=true
  else
    rm -- "${RUN_ROOT}/comparison-error.txt"
  fi
fi

(
  cd -- "${RUN_ROOT}"
  root_files=()
  [[ -f dsl/SHA256SUMS ]] && root_files+=(dsl/SHA256SUMS)
  [[ -f matched-native-slim/SHA256SUMS ]] &&
    root_files+=(matched-native-slim/SHA256SUMS)
  [[ -f comparison-summary.json ]] && root_files+=(comparison-summary.json)
  [[ -f comparison-error.txt ]] && root_files+=(comparison-error.txt)
  root_files+=(build-evidence/SHA256SUMS)
  sha256sum -- "${root_files[@]}" >SHA256SUMS
)

git -C "${REPO_ROOT}" status --short >"${BUILD_DIR}/ace-status-after.txt"
git -C "${PHANTOM_DIR}" status --short >"${BUILD_DIR}/phantom-status-after.txt" \
  2>"${BUILD_DIR}/phantom-status-after.stderr.txt" || true
echo "Completed mode=${MODE}; checksum-closed results are under ${RUN_ROOT}"
if [[ "${RUN_FAILED}" == true ]]; then
  die "one or more selected ResNet executions failed; see ${RUN_ROOT}"
fi
