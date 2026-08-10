#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"

usage() {
  echo "usage: $0 [--mode full|generated-bootstrap-correctness] --ace-commit COMMIT --phantom-commit COMMIT [--ordinary-run-root DIR --retained-run-root DIR] --bootstrap-run-root DIR OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
PACKAGE_MODE="full"
ORDINARY_RUN_ROOT=""
RETAINED_RUN_ROOT=""
BOOTSTRAP_RUN_ROOT=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || usage
      PACKAGE_MODE="$2"
      shift 2
      ;;
    --ace-commit)
      [[ $# -ge 2 ]] || usage
      ACE_COMMIT="$2"
      shift 2
      ;;
    --phantom-commit)
      [[ $# -ge 2 ]] || usage
      PHANTOM_COMMIT="$2"
      shift 2
      ;;
    --ordinary-run-root)
      [[ $# -ge 2 ]] || usage
      ORDINARY_RUN_ROOT="$2"
      shift 2
      ;;
    --retained-run-root)
      [[ $# -ge 2 ]] || usage
      RETAINED_RUN_ROOT="$2"
      shift 2
      ;;
    --bootstrap-run-root)
      [[ $# -ge 2 ]] || usage
      BOOTSTRAP_RUN_ROOT="$2"
      shift 2
      ;;
    --)
      shift
      [[ $# -eq 1 && -z "${OUTPUT_ARGUMENT}" ]] || usage
      OUTPUT_ARGUMENT="$1"
      shift
      ;;
    -*) usage ;;
    *)
      [[ -z "${OUTPUT_ARGUMENT}" ]] || usage
      OUTPUT_ARGUMENT="$1"
      shift
      ;;
  esac
done
[[ "${PACKAGE_MODE}" == "full" ||
   "${PACKAGE_MODE}" == "generated-bootstrap-correctness" ]] || usage
[[ -n "${ACE_COMMIT}" && -n "${PHANTOM_COMMIT}" &&
   -n "${BOOTSTRAP_RUN_ROOT}" && -n "${OUTPUT_ARGUMENT}" ]] || usage
if [[ "${PACKAGE_MODE}" == "full" ]]; then
  [[ -n "${ORDINARY_RUN_ROOT}" && -n "${RETAINED_RUN_ROOT}" ]] || usage
fi
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commits must be full lowercase 40-character object IDs" >&2
  exit 1
fi
git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"
SELECTED_DEPENDENCY_LOCK="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
)"
SELECTED_PHANTOM_COMMIT="$(
  sed -n 's/^PHANTOM_COMMIT=//p' <<<"${SELECTED_DEPENDENCY_LOCK}"
)"
if [[ ! "${SELECTED_PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      "${SELECTED_PHANTOM_COMMIT}" != "${PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit differs from the selected ACE dependency lock" >&2
  exit 1
fi

package_generated_bootstrap_correctness() {
  local run_root qualification output archive_tools ace_archive phantom_archive
  run_root="$(realpath -- "${BOOTSTRAP_RUN_ROOT}")"
  qualification="${run_root}/bootstrap_qualification"
  output="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
  if [[ -e "${output}" ]]; then
    echo "output already exists: ${output}" >&2
    return 1
  fi
  local -a required=(
    "${run_root}/SHA256SUMS"
    "${run_root}/ace_source_manifest.json"
    "${run_root}/phantom_source_manifest.json"
    "${run_root}/artifact_manifest.json"
    "${run_root}/manifest.json"
    "${run_root}/qualification_invocation.json"
    "${run_root}/bootstrap_generation_invocation.json"
    "${qualification}/compiler_invocation.json"
    "${qualification}/bootstrap_raw.air"
    "${qualification}/bootstrap_post_ckks.air"
    "${qualification}/bootstrap_post_operations.air"
    "${qualification}/compiler_context_manifest.json"
    "${qualification}/compiler_resource_manifest.json"
    "${qualification}/compiler_constant_manifest.json"
    "${qualification}/bootstrap_semantics.json"
    "${qualification}/post_operations_attestation.json"
    "${qualification}/bootstrap_correctness_fixture.json"
    "${qualification}/generation.json"
    "${qualification}/source-audit.json"
    "${qualification}/qualification.json"
    "${qualification}/cuda_elf.txt"
    "${qualification}/native_ant_reference.json"
    "${qualification}/native_ant_reference.bin"
    "${qualification}/generated_ant_reference.json"
    "${qualification}/generated_ant_reference.bin"
    "${qualification}/host_oracle_replay.json"
    "${qualification}/generated_bootstrap_phantom_correctness.cu"
    "${qualification}/bootstrap_qualification.cu"
    "${qualification}/bootstrap_qualification_ant.cxx"
  )
  local path
  for path in "${required[@]}"; do
    [[ -s "${path}" ]] || {
      echo "missing generated-bootstrap correctness input: ${path}" >&2
      return 1
    }
  done
  python3 - "${run_root}/artifact_manifest.json" \
    "${qualification}/qualification.json" \
    "${run_root}/qualification_invocation.json" \
    "${run_root}/bootstrap_generation_invocation.json" \
    "${qualification}/generation.json" \
    "${qualification}/bootstrap_semantics.json" \
    "${qualification}/bootstrap_correctness_fixture.json" \
    "${qualification}/source-audit.json" \
    "${qualification}/cuda_elf.txt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    artifact,
    qualification,
    qualification_invocation,
    generation_invocation,
    generation,
    semantics,
    fixture,
    source_audit,
) = [
    json.loads(Path(value).read_text(encoding="utf-8")) for value in sys.argv[1:9]
]
if artifact.get("schema_version") != "ace.phantom.bootstrap-artifacts/3.0.0":
    raise SystemExit("correctness artifact manifest schema is unsupported")
if qualification.get("schema_version") != "ace.phantom.bootstrap-host-qualification/2.0.0":
    raise SystemExit("correctness host qualification schema is unsupported")
if (
    generation.get("schema_version")
    != "ace.phantom.bootstrap-generation/4.0.0"
    or generation.get("status") != "pass"
    or semantics.get("schema_version")
    != "ace.phantom.generated-bootstrap.semantics/2.0.0"
    or semantics.get("status") != "pass"
    or fixture.get("schema_version")
    != "ace.phantom.bootstrap-correctness-fixture/2.0.0"
    or source_audit.get("schema_version")
    != "ace.phantom.bootstrap-generated-artifact-audit/4.0.0"
    or source_audit.get("status") != "pass"
):
    raise SystemExit("correctness generated authority schema is unsupported")
try:
    constant_count = source_audit["counts"]["constants"]
    qualification_closure = source_audit["qualification_closure"]
    body_closure = qualification_closure["terminal_body_closure"]
    phantom_body = body_closure["phantom"]
    ant_body = body_closure["generated_dsl_ant"]
    transform_semantics = semantics["identity_domain_attestation"][
        "normalization"
    ]["compiler_transform_payload_semantics"]
    transform_stage_groups = [
        transform_semantics[direction]["stages"]
        for direction in ("coefficients_to_slots", "slots_to_coefficients")
    ]
    expected_transform_stage_count = sum(
        len(stages) for stages in transform_stage_groups
    )
    generated_post_ckks_sha256 = generation["air"]["post_ckks"]["sha256"]
except (KeyError, TypeError) as error:
    raise SystemExit("correctness terminal-body closure is incomplete") from error
if (
    any(not isinstance(stages, list) or not stages for stages in transform_stage_groups)
    or expected_transform_stage_count <= 0
    or generated_post_ckks_sha256 != artifact.get("post_ckks_air_sha256")
    or generated_post_ckks_sha256
    != source_audit.get("inputs", {}).get("post_ckks_air", {}).get("sha256")
    or qualification_closure.get("canonical_post_ckks_air_sha256")
    != generated_post_ckks_sha256
    or qualification_closure.get("identity_domain_attested") is not True
    or qualification_closure.get("post_operations_attested") is not True
    or qualification_closure.get("post_ckks_transform_roles_attested") is not True
    or qualification_closure.get("post_ckks_evalmod_polynomial_attested") is not True
    or isinstance(
        qualification_closure.get("post_ckks_transform_stage_count"), bool
    )
    or not isinstance(
        qualification_closure.get("post_ckks_transform_stage_count"), int
    )
    or qualification_closure["post_ckks_transform_stage_count"] <= 0
    or qualification_closure["post_ckks_transform_stage_count"]
    != expected_transform_stage_count
    or set(body_closure) != {"phantom", "generated_dsl_ant"}
    or not isinstance(phantom_body, dict)
    or set(phantom_body)
    != {
        "reachable_value_count",
        "transform_constant_count",
        "reachable_required_stage_count",
        "scalar_encode_count",
        "scalar_encode_payload_sha256",
    }
    or not isinstance(ant_body, dict)
    or set(ant_body)
    != {
        "returned_dependency_count",
        "transform_constant_count",
        "reachable_required_stage_count",
        "scalar_encode_count",
        "scalar_encode_payload_sha256",
    }
    or phantom_body["transform_constant_count"] != constant_count
    or ant_body["transform_constant_count"] != constant_count
    or phantom_body["reachable_required_stage_count"] != 4
    or ant_body["reachable_required_stage_count"] != 4
    or phantom_body["scalar_encode_count"] != ant_body["scalar_encode_count"]
    or phantom_body["scalar_encode_payload_sha256"]
    != ant_body["scalar_encode_payload_sha256"]
    or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            constant_count,
            phantom_body["reachable_value_count"],
            phantom_body["reachable_required_stage_count"],
            phantom_body["scalar_encode_count"],
            ant_body["returned_dependency_count"],
            ant_body["reachable_required_stage_count"],
            ant_body["scalar_encode_count"],
        )
    )
    or not isinstance(phantom_body["scalar_encode_payload_sha256"], str)
    or len(phantom_body["scalar_encode_payload_sha256"]) != 64
    or any(
        character not in "0123456789abcdef"
        for character in phantom_body["scalar_encode_payload_sha256"]
    )
):
    raise SystemExit("correctness terminal-body closure is invalid")


def checked_argv(record, schema, label):
    argv = record.get("argv")
    if (
        record.get("schema_version") != schema
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value for value in argv)
    ):
        raise SystemExit(f"correctness {label} invocation is invalid")
    normalized = json.dumps(
        argv, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != record.get(
        "normalized_argv_sha256"
    ):
        raise SystemExit(f"correctness {label} invocation hash is invalid")
    return argv


qualification_argv = checked_argv(
    qualification_invocation,
    "ace.phantom.qualification-invocation/1.0.0",
    "qualification",
)
generation_argv = checked_argv(
    generation_invocation,
    "ace.phantom.bootstrap-qualification-invocation/1.0.0",
    "generation",
)
if qualification_argv[0] != "tools/phantom_gpu/compile_only.sh":
    raise SystemExit("correctness qualification tool is invalid")
if generation_argv[:2] != [
    "tools/phantom_gpu/generate_bootstrap_qualification.py",
    "bootstrap_qualification",
]:
    raise SystemExit("correctness generation tool or destination is invalid")


def unique_options(arguments, label):
    if len(arguments) % 2:
        raise SystemExit(f"correctness {label} invocation has an unpaired option")
    options = dict(zip(arguments[0::2], arguments[1::2]))
    if len(options) != len(arguments) // 2:
        raise SystemExit(f"correctness {label} invocation repeats an option")
    return options


qualification_options = unique_options(qualification_argv[1:], "qualification")
generation_options = unique_options(generation_argv[2:], "generation")
compiler_option_names = {
    "--poly-degree", "--vector-capacity", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight", "--q-part-count", "--encode-transform-budget",
    "--decode-transform-budget", "--ciphertext-constant-encoding", "--packing",
    "--post-multiply-real", "--post-multiply-imag",
    "--post-multiply-scale-degree", "--post-rotation-step",
}
semantic_policy_names = {"--identity-error-threshold"}
if set(generation_options) != compiler_option_names | semantic_policy_names:
    raise SystemExit("correctness generation invocation is not fully explicit")
if {
    key: value for key, value in qualification_options.items()
    if key in compiler_option_names
} != {
    key: value for key, value in generation_options.items()
    if key in compiler_option_names
}:
    raise SystemExit("correctness qualification and generation options differ")
if generation_options["--identity-error-threshold"] != qualification_options.get(
    "--provider-clear-threshold"
):
    raise SystemExit("correctness identity threshold differs from qualification")
if "sm_80" not in Path(sys.argv[9]).read_text(encoding="utf-8"):
    raise SystemExit("correctness CUDA inventory lacks sm_80")
PY
  (
    cd "${run_root}"
    sha256sum -c SHA256SUMS
  )
  mkdir -p "${output}"
  chmod 0755 "${output}"
  archive_tools="$(mktemp -d)"
  trap 'rm -rf -- "${archive_tools}"' RETURN
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/source_archive.py" \
    >"${archive_tools}/source_archive.py"
  ace_archive="ace-source-${ACE_COMMIT}.tar.gz"
  phantom_archive="phantom-source-${PHANTOM_COMMIT}.tar.gz"
  python3 "${archive_tools}/source_archive.py" create \
    --repo "${REPO_ROOT}" --commit "${ACE_COMMIT}" --kind ace \
    --output "${output}/${ace_archive}" \
    --manifest "${output}/ace-source.manifest.json" \
    >"${output}/ace-source.audit.json"
  python3 "${archive_tools}/source_archive.py" create \
    --repo "${PHANTOM_REPO}" --commit "${PHANTOM_COMMIT}" --kind phantom \
    --output "${output}/${phantom_archive}" \
    --manifest "${output}/phantom-source.manifest.json" \
    >"${output}/phantom-source.audit.json"
  cmp "${run_root}/ace_source_manifest.json" \
    "${output}/ace-source.manifest.json"
  cmp "${run_root}/phantom_source_manifest.json" \
    "${output}/phantom-source.manifest.json"
  python3 - "${output}/ace-source.manifest.json" \
    "${output}/phantom-source.manifest.json" \
    "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import json
from pathlib import Path
import sys

ace, phantom = map(Path, sys.argv[1:3])
ace_commit, phantom_commit = sys.argv[3:5]
if json.loads(ace.read_text(encoding="utf-8")).get("commit") != ace_commit:
    raise SystemExit("ACE source manifest commit differs from the request")
if json.loads(phantom.read_text(encoding="utf-8")).get("commit") != phantom_commit:
    raise SystemExit("Phantom source manifest commit differs from the request")
PY
  while IFS= read -r relative; do
    git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" \
      >"${output}/${relative##*/}"
  done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/run_build_and_health.sh
tools/phantom_gpu/bootstrap_domain_attestation.py
tools/phantom_gpu/bootstrap_correctness.py
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
  chmod 0755 "${output}/source_archive.py" \
    "${output}/phase_helpers.sh" "${output}/bootstrap_environment.sh" \
    "${output}/run_build_and_health.sh" \
    "${output}/bootstrap_domain_attestation.py" \
    "${output}/bootstrap_correctness.py"
  local -a copies=(
    "artifact_manifest.json:correctness-artifact-manifest.json"
    "manifest.json:correctness-run-manifest.json"
    "qualification_invocation.json:correctness-qualification-invocation.json"
    "bootstrap_generation_invocation.json:correctness-generation-invocation.json"
    "bootstrap_qualification/compiler_invocation.json:correctness-compiler-invocation.json"
    "bootstrap_qualification/bootstrap_raw.air:correctness-raw.air"
    "bootstrap_qualification/bootstrap_post_ckks.air:correctness-post-ckks.air"
    "bootstrap_qualification/bootstrap_post_operations.air:correctness-post-operations.air"
    "bootstrap_qualification/compiler_context_manifest.json:correctness-context-manifest.json"
    "bootstrap_qualification/compiler_resource_manifest.json:correctness-resource-manifest.json"
    "bootstrap_qualification/compiler_constant_manifest.json:correctness-constant-manifest.json"
    "bootstrap_qualification/bootstrap_semantics.json:correctness-bootstrap-semantics.json"
    "bootstrap_qualification/post_operations_attestation.json:correctness-post-operation-attestation.json"
    "bootstrap_qualification/bootstrap_correctness_fixture.json:correctness-fixture.json"
    "bootstrap_qualification/generation.json:correctness-generation.json"
    "bootstrap_qualification/source-audit.json:correctness-source-audit.json"
    "bootstrap_qualification/qualification.json:correctness-host-qualification.json"
    "bootstrap_qualification/cuda_elf.txt:correctness-cuda-elf.txt"
    "bootstrap_qualification/native_ant_reference.json:correctness-native-ant.json"
    "bootstrap_qualification/native_ant_reference.bin:correctness-native-ant.bin"
    "bootstrap_qualification/generated_ant_reference.json:correctness-generated-ant.json"
    "bootstrap_qualification/generated_ant_reference.bin:correctness-generated-ant.bin"
    "bootstrap_qualification/host_oracle_replay.json:correctness-compiler-host-replay.json"
    "bootstrap_qualification/generated_bootstrap_phantom_correctness.cu:correctness-gpu-harness.cu"
    "bootstrap_qualification/bootstrap_qualification.cu:correctness-generated-phantom.cu"
    "bootstrap_qualification/bootstrap_qualification_ant.cxx:correctness-generated-ant.cxx"
  )
  local entry source_name destination_name
  for entry in "${copies[@]}"; do
    source_name="${entry%%:*}"
    destination_name="${entry#*:}"
    cp -- "${run_root}/${source_name}" "${output}/${destination_name}"
  done
  python3 -B "${output}/bootstrap_correctness.py" verify-host \
    --fixture "${output}/correctness-fixture.json" \
    --ace-source-manifest "${output}/ace-source.manifest.json" \
    --phantom-source-manifest "${output}/phantom-source.manifest.json" \
    --compiler-invocation "${output}/correctness-compiler-invocation.json" \
    --raw-air "${output}/correctness-raw.air" \
    --post-ckks-air "${output}/correctness-post-ckks.air" \
    --context-manifest "${output}/correctness-context-manifest.json" \
    --resource-manifest "${output}/correctness-resource-manifest.json" \
    --constant-manifest "${output}/correctness-constant-manifest.json" \
    --bootstrap-semantics "${output}/correctness-bootstrap-semantics.json" \
    --post-operations-air "${output}/correctness-post-operations.air" \
    --post-operation-attestation \
      "${output}/correctness-post-operation-attestation.json" \
    --native-record "${output}/correctness-native-ant.json" \
    --native-values "${output}/correctness-native-ant.bin" \
    --generated-record "${output}/correctness-generated-ant.json" \
    --generated-values "${output}/correctness-generated-ant.bin" \
    --output "${output}/correctness-host-replay.json"
  python3 - "${output}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
entries = sorted(root.iterdir())
non_regular = [
    path.name for path in entries if path.is_symlink() or not path.is_file()
]
if non_regular:
    raise SystemExit(
        "correctness package contains a non-regular top-level entry: "
        + ", ".join(non_regular)
    )
files = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in entries
    if path.name not in {"payload.json", "SHA256SUMS"}
}
record = {
    "schema_version": "ace.phantom.generated-bootstrap-correctness-payload/1.0.0",
    "status": "pass",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "source_snapshots": {
        "ace_manifest_sha256": hashlib.sha256(
            (root / "ace-source.manifest.json").read_bytes()
        ).hexdigest(),
        "phantom_manifest_sha256": hashlib.sha256(
            (root / "phantom-source.manifest.json").read_bytes()
        ).hexdigest(),
    },
    "host_oracles_replayed": True,
    "files": files,
}
(root / "payload.json").write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  (
    cd "${output}"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
      LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
  )
  trap - RETURN
  rm -rf -- "${archive_tools}"
}

if [[ "${PACKAGE_MODE}" == "generated-bootstrap-correctness" ]]; then
  expected_entrypoint_sha256="$(
    git -C "${REPO_ROOT}" show \
      "${ACE_COMMIT}:tools/phantom_gpu/package_runpod_sources.sh" |
      sha256sum | awk '{print $1}'
  )"
  observed_entrypoint_sha256="$(
    sha256sum "${REPO_ROOT}/tools/phantom_gpu/package_runpod_sources.sh" |
      awk '{print $1}'
  )"
  if [[ "${observed_entrypoint_sha256}" != "${expected_entrypoint_sha256}" ]]; then
    echo "correctness packaging entrypoint differs from the selected ACE commit" >&2
    exit 1
  fi
  package_generated_bootstrap_correctness
  exit 0
fi

ORDINARY_RUN_ROOT="$(realpath -- "${ORDINARY_RUN_ROOT}")"
RETAINED_RUN_ROOT="$(realpath -- "${RETAINED_RUN_ROOT}")"
BOOTSTRAP_RUN_ROOT="$(realpath -- "${BOOTSTRAP_RUN_ROOT}")"
FROZEN_RUN_MANIFEST="${ORDINARY_RUN_ROOT}/manifest.json"
FROZEN_ACE_SOURCE_MANIFEST="${ORDINARY_RUN_ROOT}/ace_source_manifest.json"
FROZEN_PHANTOM_SOURCE_MANIFEST="${ORDINARY_RUN_ROOT}/phantom_source_manifest.json"
FROZEN_ARTIFACT_MANIFEST="${ORDINARY_RUN_ROOT}/artifact_manifest.json"
FROZEN_QUALIFICATION_INVOCATION="${ORDINARY_RUN_ROOT}/qualification_invocation.json"
FROZEN_COMPILER_INVOCATION="${ORDINARY_RUN_ROOT}/compiler_invocation.json"
FROZEN_CONTEXT="${ORDINARY_RUN_ROOT}/ckks2c/compiler_context_manifest.json"
FROZEN_RESOURCES="${ORDINARY_RUN_ROOT}/ckks2c/compiler_resource_manifest.json"
FROZEN_CONSTANTS="${ORDINARY_RUN_ROOT}/ckks2c/compiler_constant_manifest.json"
FROZEN_POST_CKKS_AIR="${ORDINARY_RUN_ROOT}/ckks2c/ordinary_ckks_post_ckks.air"
FROZEN_FIXTURE="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_v1.json"
FROZEN_CPU_REFERENCE="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_cpu_reference.json"
FROZEN_CPU_VALUES="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_cpu_values.bin"
FROZEN_ANT_VERIFICATION="${ORDINARY_RUN_ROOT}/ordinary_ckks/ordinary_ckks_ant_verification.json"
FROZEN_HOST_QUALIFICATION="${ORDINARY_RUN_ROOT}/ordinary_ckks/host-qualification.json"
for required in \
  "${ORDINARY_RUN_ROOT}/SHA256SUMS" \
  "${FROZEN_RUN_MANIFEST}" "${FROZEN_ARTIFACT_MANIFEST}" \
  "${FROZEN_ACE_SOURCE_MANIFEST}" "${FROZEN_PHANTOM_SOURCE_MANIFEST}" \
  "${FROZEN_QUALIFICATION_INVOCATION}" "${FROZEN_COMPILER_INVOCATION}" \
  "${FROZEN_CONTEXT}" "${FROZEN_RESOURCES}" "${FROZEN_CONSTANTS}" \
  "${FROZEN_POST_CKKS_AIR}" "${FROZEN_FIXTURE}" \
  "${FROZEN_CPU_REFERENCE}" "${FROZEN_CPU_VALUES}" \
  "${FROZEN_ANT_VERIFICATION}" "${FROZEN_HOST_QUALIFICATION}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing frozen ordinary-CKKS evidence: ${required}" >&2
    exit 1
  fi
done
python3 - "${ORDINARY_RUN_ROOT}/SHA256SUMS" <<'PY'
from pathlib import Path, PurePosixPath
import re
import sys

sums = Path(sys.argv[1])
for line in sums.read_text(encoding="utf-8").splitlines():
    fields = line.split(maxsplit=1)
    if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise SystemExit("frozen SHA256SUMS contains a malformed entry")
    name = fields[1].removeprefix("*").removeprefix("./")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise SystemExit("frozen SHA256SUMS contains an unsafe path")
PY
(
  cd "${ORDINARY_RUN_ROOT}"
  sha256sum -c SHA256SUMS
)
EXPECTED_ANT_SOURCE_SHA256="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/harness/ordinary_ckks_ant_oracle.cxx" |
    sha256sum |
    awk '{print $1}'
)"
EXPECTED_RUNNER_SOURCE_SHA256="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/harness/ordinary_ckks_gpu_runner.cu" |
    sha256sum |
    awk '{print $1}'
)"
EXPECTED_HEALTH_SOURCE_SHA256="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/harness/native_phantom_health.cu" |
    sha256sum |
    awk '{print $1}'
)"
python3 - "${FROZEN_QUALIFICATION_INVOCATION}" \
  "${FROZEN_COMPILER_INVOCATION}" "${FROZEN_RUN_MANIFEST}" \
  "${FROZEN_ARTIFACT_MANIFEST}" "${FROZEN_CONTEXT}" \
  "${FROZEN_RESOURCES}" "${FROZEN_CONSTANTS}" \
  "${FROZEN_POST_CKKS_AIR}" "${FROZEN_FIXTURE}" \
  "${FROZEN_CPU_REFERENCE}" "${FROZEN_CPU_VALUES}" \
  "${FROZEN_ANT_VERIFICATION}" "${FROZEN_HOST_QUALIFICATION}" \
  "${ORDINARY_RUN_ROOT}/SHA256SUMS" "${ACE_COMMIT}" \
  "${PHANTOM_COMMIT}" "${EXPECTED_ANT_SOURCE_SHA256}" \
  "${EXPECTED_RUNNER_SOURCE_SHA256}" \
  "${EXPECTED_HEALTH_SOURCE_SHA256}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

(
    qualification_invocation_path, compiler_invocation_path,
    run_manifest_path, artifact_manifest_path, context_path, resources_path,
    constants_path, post_ckks_air_path, fixture_path, cpu_reference_path,
    cpu_values_path,
    ant_verification_path, host_qualification_path, sums_path,
) = map(Path, sys.argv[1:15])
(
    ace_commit, phantom_commit, expected_ant_source_sha,
    expected_runner_source_sha, expected_health_source_sha,
) = sys.argv[15:20]

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in frozen evidence: {key}")
        value[key] = item
    return value

def load(path):
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

listed_paths = []
listed_hashes = {}
for line in sums_path.read_text(encoding="utf-8").splitlines():
    fields = line.split(maxsplit=1)
    if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise SystemExit("frozen SHA256SUMS contains a malformed entry")
    relative = fields[1].removeprefix("*").removeprefix("./")
    listed_paths.append(relative)
    listed_hashes[relative] = fields[0]
actual_paths = sorted(
    str(path.relative_to(sums_path.parent))
    for path in sums_path.parent.rglob("*")
    if path.is_file()
    and str(path.relative_to(sums_path.parent))
    not in {"manifest.json", "SHA256SUMS"}
)
if sorted(listed_paths) != actual_paths or len(set(listed_paths)) != len(listed_paths):
    raise SystemExit("frozen SHA256SUMS does not enumerate the complete run root")

def load_invocation(path, label):
    invocation = load(path)
    if set(invocation) != {"schema_version", "argv", "normalized_argv_sha256"}:
        raise SystemExit(f"frozen {label} invocation has an invalid shape")
    expected_schema = f"ace.phantom.{label}-invocation/1.0.0"
    if invocation["schema_version"] != expected_schema:
        raise SystemExit(f"frozen {label} invocation has an unsupported schema")
    argv = invocation["argv"]
    if not isinstance(argv, list) or not all(isinstance(value, str) and value for value in argv):
        raise SystemExit(f"frozen {label} invocation argv must be a string array")
    normalized = json.dumps(
        argv, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if hashlib.sha256(normalized).hexdigest() != invocation["normalized_argv_sha256"]:
        raise SystemExit(f"frozen {label} invocation normalized hash mismatch")
    return invocation, argv

qualification_invocation, qualification_argv = load_invocation(
    qualification_invocation_path, "qualification"
)
if qualification_argv[0] != "tools/phantom_gpu/compile_only.sh":
    raise SystemExit("frozen qualification invocation has an unexpected executable")
arguments = qualification_argv[1:]
if len(arguments) % 2:
    raise SystemExit("frozen compiler invocation has an argument without a value")
pairs = dict(zip(arguments[0::2], arguments[1::2]))
expected_options = {
    "--gate", "--poly-degree", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight",
}
if len(pairs) != len(arguments) // 2 or set(pairs) != expected_options:
    raise SystemExit("frozen compiler invocation options are incomplete or duplicated")
if pairs["--gate"] != "ordinary":
    raise SystemExit("frozen compiler invocation is not the ordinary qualification")
for option in expected_options - {"--gate"}:
    if re.fullmatch(r"[0-9]+", pairs[option]) is None:
        raise SystemExit(f"frozen compiler invocation {option} is not an integer")

compiler_invocation, compiler_argv = load_invocation(
    compiler_invocation_path, "compiler"
)
if compiler_argv[0] != "tools/phantom_gpu/generate_ckks2c_probe.py":
    raise SystemExit("frozen compiler invocation has an unexpected executable")
compiler_arguments = compiler_argv[1:]
if len(compiler_arguments) % 2:
    raise SystemExit("frozen compiler invocation has an argument without a value")
compiler_pairs = dict(zip(compiler_arguments[0::2], compiler_arguments[1::2]))
expected_compiler_options = {
    "--output", "--post-ckks-air", "--context-manifest",
    "--resource-manifest", "--poly-degree", "--mul-level",
    "--input-level", "--security-level", "--scaling-factor-bits",
    "--first-prime-bits", "--hamming-weight",
}
if len(compiler_pairs) != len(compiler_arguments) // 2 or set(compiler_pairs) != expected_compiler_options:
    raise SystemExit("frozen compiler invocation options are incomplete or duplicated")
expected_paths = {
    "--output": "ckks2c/add_mul_rotate.cu",
    "--post-ckks-air": "ckks2c/ordinary_ckks_post_ckks.air",
    "--context-manifest": "ckks2c/compiler_context_manifest.json",
    "--resource-manifest": "ckks2c/compiler_resource_manifest.json",
}
if any(compiler_pairs[key] != value for key, value in expected_paths.items()):
    raise SystemExit("frozen compiler invocation contains non-portable artifact paths")
for option in expected_compiler_options - expected_paths.keys():
    if re.fullmatch(r"[0-9]+", compiler_pairs[option]) is None:
        raise SystemExit(f"frozen compiler invocation {option} is not an integer")
qualification_to_compiler = {
    "--poly-degree": "--poly-degree",
    "--mul-level": "--mul-level",
    "--input-level": "--input-level",
    "--security-level": "--security-level",
    "--scaling-factor-bits": "--scaling-factor-bits",
    "--first-prime-bits": "--first-prime-bits",
    "--hamming-weight": "--hamming-weight",
}
if any(
    pairs[qualification_option] != compiler_pairs[compiler_option]
    for qualification_option, compiler_option in qualification_to_compiler.items()
):
    raise SystemExit("qualification and compiler invocation parameters disagree")

context = load(context_path)
required_context = {
    "schema_version", "packing", "polynomial_degree",
    "logical_slot_capacity", "data_q_bit_sizes", "special_p_bit_sizes",
    "input_level", "q_part_count", "hamming_weight", "security_level",
    "first_modulus_bits", "scaling_modulus_bits",
    "resource_schema_version",
}
if not isinstance(context, dict) or set(context) != required_context:
    raise SystemExit("frozen compiler context manifest has an invalid shape")
degree = context["polynomial_degree"]
data_q = context["data_q_bit_sizes"]
if (
    context["schema_version"] != 1
    or context["resource_schema_version"] != 3
    or context["packing"] != "full"
    or isinstance(degree, bool)
    or not isinstance(degree, int)
    or degree < 2
    or degree & (degree - 1)
    or context["logical_slot_capacity"] != degree // 2
    or not isinstance(data_q, list)
    or not data_q
    or not isinstance(context["special_p_bit_sizes"], list)
    or not context["special_p_bit_sizes"]
):
    raise SystemExit("frozen compiler context manifest is structurally invalid")
expected_context_options = {
    "--poly-degree": degree,
    "--mul-level": len(data_q),
    "--input-level": context["input_level"],
    "--security-level": context["security_level"],
    "--scaling-factor-bits": context["scaling_modulus_bits"],
    "--first-prime-bits": context["first_modulus_bits"],
    "--hamming-weight": context["hamming_weight"],
}
if any(
    int(compiler_pairs[option]) != value
    for option, value in expected_context_options.items()
):
    raise SystemExit("compiler invocation disagrees with the emitted context manifest")
if any(
    isinstance(bits, bool)
    or not isinstance(bits, int)
    or bits != (
        context["first_modulus_bits"]
        if index == 0 else context["scaling_modulus_bits"]
    )
    for index, bits in enumerate(data_q)
):
    raise SystemExit("frozen data-Q list disagrees with compiler prime policy")
if any(
    isinstance(bits, bool) or not isinstance(bits, int) or bits <= 0
    for bits in context["special_p_bit_sizes"]
):
    raise SystemExit("frozen special-P list is invalid")

resources = load(resources_path)
required_resources = {
    "schema_version", "context_schema_version", "relinearization_key",
    "rotation_steps", "conjugation_key", "rotate_batch",
    "rotation_batches", "raise_mod", "monomial_powers",
    "complex_plaintext", "native_bootstrap_precompute",
}
if (
    set(resources) != required_resources
    or resources["schema_version"] != 3
    or resources["context_schema_version"] != 1
    or resources["complex_plaintext"] is not False
    or resources["native_bootstrap_precompute"] is not False
):
    raise SystemExit("frozen ordinary resource manifest is not schema-v3 key-safe")

context_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
constants = load(constants_path)
if constants != {
    "constants": [],
    "context_manifest_sha256": context_sha,
    "context_schema_version": 1,
    "resource_schema_version": 3,
    "schema_version": 1,
}:
    raise SystemExit("frozen ordinary constant manifest is not the derived empty manifest")

run_manifest = load(run_manifest_path)
if (run_manifest.get("status"), run_manifest.get("gate")) != ("pass", "ordinary"):
    raise SystemExit("frozen evidence is not a passing ordinary qualification")
if run_manifest.get("ace_commit") != ace_commit:
    raise SystemExit("frozen evidence ACE producer does not match the requested commit")
if run_manifest.get("phantom_commit") != phantom_commit:
    raise SystemExit("frozen evidence Phantom producer does not match the requested commit")
if run_manifest.get("ace_worktree_dirty") is not False:
    raise SystemExit("frozen qualification source snapshot is not clean")
if run_manifest.get("source_mode") != "snapshot":
    raise SystemExit("frozen qualification was not produced from audited snapshots")
ace_source_manifest_path = sums_path.parent / "ace_source_manifest.json"
phantom_source_manifest_path = sums_path.parent / "phantom_source_manifest.json"
ace_source_manifest = load(ace_source_manifest_path)
phantom_source_manifest = load(phantom_source_manifest_path)
if (
    ace_source_manifest.get("kind") != "ace"
    or ace_source_manifest.get("commit") != ace_commit
    or phantom_source_manifest.get("kind") != "phantom"
    or phantom_source_manifest.get("commit") != phantom_commit
):
    raise SystemExit("frozen qualification source manifests have stale identities")
if (
    run_manifest.get("ace_tracked_source_manifest")
    != "ace_source_manifest.json"
    or run_manifest.get("ace_tracked_source_manifest_sha256")
    != hashlib.sha256(ace_source_manifest_path.read_bytes()).hexdigest()
    or run_manifest.get("phantom_source_manifest")
    != "phantom_source_manifest.json"
    or run_manifest.get("phantom_source_manifest_sha256")
    != hashlib.sha256(phantom_source_manifest_path.read_bytes()).hexdigest()
):
    raise SystemExit("run manifest does not bind the audited source snapshots")
sums_sha = hashlib.sha256(sums_path.read_bytes()).hexdigest()
if run_manifest.get("evidence_sha256_manifest_sha256") != sums_sha:
    raise SystemExit("frozen evidence checksum manifest is not bound by the run manifest")
artifact_manifest_sha = hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest()
if run_manifest.get("artifact_manifest_sha256") != artifact_manifest_sha:
    raise SystemExit("frozen artifact manifest is not bound by the run manifest")

resources_sha = hashlib.sha256(resources_path.read_bytes()).hexdigest()
constants_sha = hashlib.sha256(constants_path.read_bytes()).hexdigest()
post_ckks_air_sha = hashlib.sha256(post_ckks_air_path.read_bytes()).hexdigest()
fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
cpu_reference_sha = hashlib.sha256(cpu_reference_path.read_bytes()).hexdigest()
cpu_values_sha = hashlib.sha256(cpu_values_path.read_bytes()).hexdigest()
host = load(host_qualification_path)
qualification_invocation_sha = hashlib.sha256(
    qualification_invocation_path.read_bytes()
).hexdigest()
compiler_invocation_sha = hashlib.sha256(
    compiler_invocation_path.read_bytes()
).hexdigest()
host_qualification_sha = hashlib.sha256(
    host_qualification_path.read_bytes()
).hexdigest()
expected_run = {
    "compiler_context_manifest_sha256": context_sha,
    "compiler_resource_manifest_sha256": resources_sha,
    "compiler_constant_manifest_sha256": constants_sha,
    "qualification_invocation_sha256": qualification_invocation_sha,
    "normalized_qualification_argv_sha256": qualification_invocation[
        "normalized_argv_sha256"
    ],
    "compiler_invocation_sha256": compiler_invocation_sha,
    "normalized_compiler_command_sha256": compiler_invocation[
        "normalized_argv_sha256"
    ],
    "post_ckks_air_sha256": post_ckks_air_sha,
    "host_qualification_sha256": host_qualification_sha,
    "ordinary_fixture_sha256": fixture_sha,
    "ordinary_cpu_reference_sha256": cpu_reference_sha,
    "ordinary_cpu_values_sha256": cpu_values_sha,
}
if any(run_manifest.get(key) != value for key, value in expected_run.items()):
    raise SystemExit("run manifest does not bind all qualification artifacts")
expected_host = {
    "compiler_context_manifest_sha256": context_sha,
    "compiler_resource_manifest_sha256": resources_sha,
    "compiler_constant_manifest_sha256": constants_sha,
    "keyless_constant_manifest_sha256": listed_hashes[
        "ordinary_ckks/ordinary_ckks_keyless_constants.json"
    ],
    "fixture_sha256": fixture_sha,
    "cpu_reference_sha256": cpu_reference_sha,
    "cpu_values_sha256": cpu_values_sha,
    "ant_source_sha256": expected_ant_source_sha,
    "runner_source_sha256": expected_runner_source_sha,
    "native_health_source_sha256": expected_health_source_sha,
    "generated_source_sha256": listed_hashes["ckks2c/add_mul_rotate.cu"],
    "keyless_source_sha256": listed_hashes[
        "ordinary_ckks/ordinary_ckks_keyless_probe.cu"
    ],
    "runner_sha256": listed_hashes[
        "ordinary_ckks/ordinary_ckks_gpu_runner_sm80"
    ],
    "keyless_runner_sha256": listed_hashes[
        "ordinary_ckks/ordinary_ckks_keyless_runner_sm80"
    ],
    "ant_binary_sha256": listed_hashes[
        "ordinary_ckks/ordinary_ckks_ant_oracle"
    ],
    "ace_commit": ace_commit,
    "phantom_commit": phantom_commit,
    "qualification_invocation_sha256": qualification_invocation_sha,
    "normalized_qualification_argv_sha256": qualification_invocation[
        "normalized_argv_sha256"
    ],
    "compiler_invocation_sha256": compiler_invocation_sha,
    "normalized_compiler_command_sha256": compiler_invocation[
        "normalized_argv_sha256"
    ],
    "post_ckks_air_sha256": post_ckks_air_sha,
    "ace_source_manifest_sha256": hashlib.sha256(
        ace_source_manifest_path.read_bytes()
    ).hexdigest(),
    "phantom_source_manifest_sha256": hashlib.sha256(
        phantom_source_manifest_path.read_bytes()
    ).hexdigest(),
}
if host.get("status") != "pass" or any(host.get(key) != value for key, value in expected_host.items()):
    raise SystemExit("host qualification hashes do not bind the frozen artifacts")
if (
    host.get("ant_executable_was_run") is not True
    or host.get("gpu_executables_were_run") is not False
    or host.get("compute_sanitizer_was_run") is not False
):
    raise SystemExit("host qualification execution claims are inconsistent")

cpu_reference = load(cpu_reference_path)
expected_producer = f"ace-{ace_commit}-phantom-{phantom_commit}-ant"
if cpu_reference.get("ant", {}).get("producer") != expected_producer:
    raise SystemExit(
        "frozen CPU/ANT producer does not match the requested commits: "
        f"expected {expected_producer!r}"
    )
ant_verification = load(ant_verification_path)
if ant_verification.get("status") != "pass":
    raise SystemExit("frozen ANT verification did not pass")

fixture = load(fixture_path)
bindings = fixture.get("qualification_bindings", {})
seed = bindings.get("deterministic_seed")
if (
    set(bindings) != {
        "status",
        "deterministic_seed",
        "deterministic_generator",
        "normalized_compiler_command_sha256",
        "post_ckks_air_sha256",
    }
    or bindings.get("status") != "bound"
    or isinstance(seed, bool)
    or not isinstance(seed, int)
    or seed < 0
    or seed > (1 << 64) - 1
    or bindings.get("deterministic_generator")
    != "splitmix64-float53-complex-v1"
    or bindings.get("normalized_compiler_command_sha256")
    != compiler_invocation["normalized_argv_sha256"]
    or bindings.get("post_ckks_air_sha256") != post_ckks_air_sha
):
    raise SystemExit("frozen fixture is not bound to the invocation and post-CKKS AIR")
if fixture.get("compiler_context_manifest") != {"sha256": context_sha}:
    raise SystemExit("frozen fixture is not bound to the compiler context manifest")

artifact = load(artifact_manifest_path)
if artifact.get("schema_version") != "ace.phantom.ordinary-artifacts/1.1.0" or artifact.get("status") != "bound":
    raise SystemExit("frozen artifact manifest has an invalid schema or status")
if artifact.get("source_mode") != "snapshot":
    raise SystemExit("frozen artifact manifest was not produced from snapshots")
if artifact.get("source_manifests") != {
    "ace_sha256": hashlib.sha256(ace_source_manifest_path.read_bytes()).hexdigest(),
    "phantom_sha256": hashlib.sha256(phantom_source_manifest_path.read_bytes()).hexdigest(),
}:
    raise SystemExit("frozen artifact manifest source snapshot binding is stale")
if (artifact.get("ace_commit"), artifact.get("phantom_commit")) != (ace_commit, phantom_commit):
    raise SystemExit("frozen artifact manifest source identities are stale")
if artifact.get("normalized_qualification_argv_sha256") != qualification_invocation["normalized_argv_sha256"]:
    raise SystemExit("frozen artifact manifest qualification invocation binding is stale")
if artifact.get("normalized_compiler_command_sha256") != compiler_invocation["normalized_argv_sha256"]:
    raise SystemExit("frozen artifact manifest invocation binding is stale")
if artifact.get("post_ckks_air_sha256") != post_ckks_air_sha:
    raise SystemExit("frozen artifact manifest post-CKKS AIR binding is stale")
if artifact.get("compiler_constant_manifest_sha256") != constants_sha:
    raise SystemExit("frozen artifact manifest constant binding is stale")
expected_harness_sources = {
    "tools/phantom_gpu/harness/native_phantom_health.cu":
        expected_health_source_sha,
    "tools/phantom_gpu/harness/ordinary_ckks_ant_oracle.cxx":
        expected_ant_source_sha,
    "tools/phantom_gpu/harness/ordinary_ckks_gpu_runner.cu":
        expected_runner_source_sha,
}
if artifact.get("harness_sources") != expected_harness_sources:
    raise SystemExit("frozen artifact manifest harness source bindings are stale")
expected_phantom_producer = expected_producer.removesuffix("-ant")
if artifact.get("phantom_producer") != expected_phantom_producer or artifact.get("ant_producer") != expected_producer:
    raise SystemExit("frozen artifact manifest producer identities are stale")
artifact_files = artifact.get("files", {})
expected_files = {
    path: digest for path, digest in listed_hashes.items()
    if path != "artifact_manifest.json"
}
if artifact_files != expected_files:
    raise SystemExit("frozen artifact relationship hashes are incomplete or stale")
expected_production_sources = {
    path: expected_files[path]
    for path in (
        "ckks2c/add_mul_rotate.cu",
        "ckks2c/ordinary_runtime_symbols.cu",
        "ordinary_ckks/ordinary_ckks_keyless_probe.cu",
    )
}
expected_host_binaries = {
    path: expected_files[path]
    for path in (
        "ckks2c/add_mul_rotate_sm80",
        "ckks2c/native_phantom_health_sm80",
        "ordinary_ckks/ordinary_ckks_gpu_runner_sm80",
        "ordinary_ckks/ordinary_ckks_keyless_runner_sm80",
        "ordinary_ckks/ordinary_ckks_ant_oracle",
    )
}
if artifact.get("production_sources") != expected_production_sources:
    raise SystemExit("frozen artifact manifest production source bindings are stale")
if artifact.get("host_binaries") != expected_host_binaries:
    raise SystemExit("frozen artifact manifest host binary bindings are stale")
for audit_path in (
    "ckks2c/archive-member-audit.json",
    "ckks2c/source_audit.json",
    "ordinary_ckks/fixture-validation.json",
    "ordinary_ckks/static-rejection-coverage.json",
    "ordinary_ckks/keyless-source-audit.json",
):
    audit = load(sums_path.parent / audit_path)
    if audit.get("status") != "pass":
        raise SystemExit(f"frozen qualification audit did not pass: {audit_path}")
archive_audit = load(sums_path.parent / "ckks2c/archive-member-audit.json")
expected_inventories = {
    Path(path).name: expected_files[path]
    for path in (
        "ckks2c/adapter_archive_members.txt",
        "ckks2c/provider_archive_members.txt",
        "ckks2c/common_archive_members.txt",
    )
}
observed_inventories = archive_audit.get("inventories")
if (
    archive_audit.get("schema_version")
    != "ace.phantom.production-archive-members/1.0.0"
    or archive_audit.get("forbidden_member_count") != 0
    or archive_audit.get("forbidden_members") != []
    or not isinstance(observed_inventories, list)
    or {
        item.get("path"): item.get("sha256")
        for item in observed_inventories
        if isinstance(item, dict) and item.get("member_count", 0) > 0
    } != expected_inventories
):
    raise SystemExit("frozen production archive member audit is incomplete or stale")
ckks_qualification = load(sums_path.parent / "ckks2c/qualification.json")
if (
    ckks_qualification.get("status") != "pass"
    or ckks_qualification.get("gate") != "ckks2c"
    or ckks_qualification.get("architecture") != "sm_80"
    or ckks_qualification.get("archive_member_inventory_audit") != "pass"
    or ckks_qualification.get("production_archive_contains_native_bootstrap")
    is not False
    or ckks_qualification.get("generated_source_contains_native_bootstrap")
    is not False
    or ckks_qualification.get("source_sha256")
    != expected_files["ckks2c/add_mul_rotate.cu"]
    or ckks_qualification.get("binary_sha256")
    != expected_files["ckks2c/add_mul_rotate_sm80"]
    or ckks_qualification.get("health_binary_sha256")
    != expected_files["ckks2c/native_phantom_health_sm80"]
    or ckks_qualification.get("compiler_constant_manifest_sha256")
    != constants_sha
):
    raise SystemExit("frozen CKKS2C qualification bindings are incomplete or stale")
for archive_field in (
    "adapter_archive_sha256",
    "provider_archive_sha256",
    "common_archive_sha256",
):
    if re.fullmatch(r"[0-9a-f]{64}", ckks_qualification.get(archive_field, "")) is None:
        raise SystemExit(f"frozen CKKS2C qualification lacks {archive_field}")
PY

BOOTSTRAP_RUN_MANIFEST="${BOOTSTRAP_RUN_ROOT}/manifest.json"
BOOTSTRAP_ARTIFACT_MANIFEST="${BOOTSTRAP_RUN_ROOT}/artifact_manifest.json"
BOOTSTRAP_QUALIFICATION_INVOCATION="${BOOTSTRAP_RUN_ROOT}/qualification_invocation.json"
BOOTSTRAP_GENERATION_INVOCATION="${BOOTSTRAP_RUN_ROOT}/bootstrap_generation_invocation.json"
BOOTSTRAP_CONTEXT="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/compiler_context_manifest.json"
BOOTSTRAP_RESOURCES="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/compiler_resource_manifest.json"
BOOTSTRAP_CONSTANTS="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/compiler_constant_manifest.json"
BOOTSTRAP_GENERATION="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/generation.json"
BOOTSTRAP_SOURCE_AUDIT="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/source-audit.json"
BOOTSTRAP_HOST_QUALIFICATION="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/qualification.json"
BOOTSTRAP_HARNESS="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/bootstrap_phantom_constants.cu"
BOOTSTRAP_RAW_AIR="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/bootstrap_raw.air"
BOOTSTRAP_POST_CKKS_AIR="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/bootstrap_post_ckks.air"
BOOTSTRAP_GENERATED_SOURCE="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/bootstrap_qualification.cu"
BOOTSTRAP_HOST_BINARY="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/bootstrap_phantom_constants_sm80"
BOOTSTRAP_LINK_COMMANDS="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/link-commands.txt"
BOOTSTRAP_CUDA_ELF="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/cuda_elf.txt"
BOOTSTRAP_CUDA_RESOURCES="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/cuda_resources.txt"
BOOTSTRAP_FILE_REPORT="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/file.txt"
BOOTSTRAP_READELF="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/readelf.txt"
BOOTSTRAP_LINKED_SYMBOLS="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/linked_binary_symbols.txt"
BOOTSTRAP_SYMBOL_CLOSURE="${BOOTSTRAP_RUN_ROOT}/bootstrap_qualification/symbol-closure.json"
for required in \
  "${BOOTSTRAP_RUN_ROOT}/SHA256SUMS" \
  "${BOOTSTRAP_RUN_MANIFEST}" "${BOOTSTRAP_ARTIFACT_MANIFEST}" \
  "${BOOTSTRAP_QUALIFICATION_INVOCATION}" \
  "${BOOTSTRAP_GENERATION_INVOCATION}" \
  "${BOOTSTRAP_CONTEXT}" "${BOOTSTRAP_RESOURCES}" \
  "${BOOTSTRAP_CONSTANTS}" "${BOOTSTRAP_GENERATION}" \
  "${BOOTSTRAP_SOURCE_AUDIT}" "${BOOTSTRAP_HOST_QUALIFICATION}" \
  "${BOOTSTRAP_HARNESS}" "${BOOTSTRAP_RAW_AIR}" \
  "${BOOTSTRAP_POST_CKKS_AIR}" "${BOOTSTRAP_GENERATED_SOURCE}" \
  "${BOOTSTRAP_HOST_BINARY}" "${BOOTSTRAP_LINK_COMMANDS}" \
  "${BOOTSTRAP_CUDA_ELF}" "${BOOTSTRAP_CUDA_RESOURCES}" \
  "${BOOTSTRAP_FILE_REPORT}" "${BOOTSTRAP_READELF}" \
  "${BOOTSTRAP_LINKED_SYMBOLS}" "${BOOTSTRAP_SYMBOL_CLOSURE}" \
  "${BOOTSTRAP_RUN_ROOT}/ace_source_manifest.json" \
  "${BOOTSTRAP_RUN_ROOT}/phantom_source_manifest.json"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing frozen bootstrap qualification evidence: ${required}" >&2
    exit 1
  fi
done
python3 - "${BOOTSTRAP_RUN_ROOT}/SHA256SUMS" <<'PY'
from pathlib import Path, PurePosixPath
import re
import sys

sums = Path(sys.argv[1])
for line in sums.read_text(encoding="utf-8").splitlines():
    fields = line.split(maxsplit=1)
    if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
        raise SystemExit("bootstrap SHA256SUMS contains a malformed entry")
    name = fields[1].removeprefix("*").removeprefix("./")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise SystemExit("bootstrap SHA256SUMS contains an unsafe path")
PY
(
  cd "${BOOTSTRAP_RUN_ROOT}"
  sha256sum -c SHA256SUMS
)
EXPECTED_BOOTSTRAP_HARNESS_SHA256="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/harness/bootstrap_phantom_constants.cu" |
    sha256sum | awk '{print $1}'
)"
python3 - "${BOOTSTRAP_RUN_ROOT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" \
  "${EXPECTED_BOOTSTRAP_HARNESS_SHA256}" \
  "${FROZEN_ACE_SOURCE_MANIFEST}" "${FROZEN_PHANTOM_SOURCE_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
ace_commit, phantom_commit, expected_harness_sha = sys.argv[2:5]
ordinary_ace_source, ordinary_phantom_source = map(Path, sys.argv[5:7])

def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"duplicate JSON key in bootstrap evidence: {key}")
        value[key] = item
    return value

def load(relative):
    path = root / relative
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )

def digest(relative):
    return hashlib.sha256((root / relative).read_bytes()).hexdigest()

def normalized_invocation(relative, schema):
    invocation = load(relative)
    if set(invocation) != {"schema_version", "argv", "normalized_argv_sha256"}:
        raise SystemExit(f"bootstrap invocation has an invalid shape: {relative}")
    if invocation["schema_version"] != schema:
        raise SystemExit(f"bootstrap invocation has an invalid schema: {relative}")
    argv = invocation["argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        raise SystemExit(f"bootstrap invocation argv is invalid: {relative}")
    encoded = json.dumps(
        argv, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != invocation["normalized_argv_sha256"]:
        raise SystemExit(f"bootstrap invocation normalized hash mismatch: {relative}")
    return invocation, argv

listed = {}
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    checksum, name = line.split(maxsplit=1)
    relative = name.removeprefix("*").removeprefix("./")
    if relative in listed:
        raise SystemExit("bootstrap SHA256SUMS contains a duplicate path")
    listed[relative] = checksum
actual = sorted(
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file()
    and str(path.relative_to(root)) not in {"manifest.json", "SHA256SUMS"}
)
if sorted(listed) != actual:
    raise SystemExit("bootstrap SHA256SUMS does not enumerate the complete run root")

qualification_invocation, qualification_argv = normalized_invocation(
    "qualification_invocation.json",
    "ace.phantom.qualification-invocation/1.0.0",
)
if qualification_argv[0] != "tools/phantom_gpu/compile_only.sh":
    raise SystemExit("bootstrap qualification invocation executable is invalid")
qualification_arguments = qualification_argv[1:]
if len(qualification_arguments) % 2:
    raise SystemExit("bootstrap qualification invocation has an unpaired argument")
qualification_options = dict(
    zip(qualification_arguments[0::2], qualification_arguments[1::2])
)
if (
    qualification_options.get("--gate") != "bootstrap"
    or len(qualification_options) != len(qualification_arguments) // 2
):
    raise SystemExit("bootstrap qualification invocation is not explicit and unique")

generation_invocation, generation_argv = normalized_invocation(
    "bootstrap_generation_invocation.json",
    "ace.phantom.bootstrap-qualification-invocation/1.0.0",
)
if generation_argv[:2] != [
    "tools/phantom_gpu/generate_bootstrap_qualification.py",
    "bootstrap_qualification",
]:
    raise SystemExit("bootstrap generation invocation executable/output is invalid")
generation_arguments = generation_argv[2:]
if len(generation_arguments) % 2:
    raise SystemExit("bootstrap generation invocation has an unpaired argument")
generation_options = dict(zip(generation_arguments[0::2], generation_arguments[1::2]))
if len(generation_options) != len(generation_arguments) // 2:
    raise SystemExit("bootstrap generation invocation options are not unique")
compiler_option_names = {
    "--poly-degree", "--vector-capacity", "--mul-level", "--input-level",
    "--security-level", "--scaling-factor-bits", "--first-prime-bits",
    "--hamming-weight", "--q-part-count", "--encode-transform-budget",
    "--decode-transform-budget", "--ciphertext-constant-encoding", "--packing",
    "--post-multiply-real", "--post-multiply-imag",
    "--post-multiply-scale-degree", "--post-rotation-step",
}
if set(generation_options) != compiler_option_names:
    raise SystemExit("bootstrap generation invocation omits an explicit option")
qualification_compiler_options = {
    key: value for key, value in qualification_options.items()
    if key in compiler_option_names
}
if qualification_compiler_options != generation_options:
    raise SystemExit("qualification and generation compiler options differ")

context = load("bootstrap_qualification/compiler_context_manifest.json")
if (
    context.get("schema_version") != 1
    or context.get("resource_schema_version") != 3
    or context.get("packing") != generation_options["--packing"]
    or context.get("polynomial_degree") != int(generation_options["--poly-degree"])
    or context.get("logical_slot_capacity") != int(generation_options["--vector-capacity"])
    or len(context.get("data_q_bit_sizes", [])) != int(generation_options["--mul-level"])
    or context.get("input_level") != int(generation_options["--input-level"])
    or context.get("security_level") != int(generation_options["--security-level"])
    or context.get("scaling_modulus_bits") != int(generation_options["--scaling-factor-bits"])
    or context.get("first_modulus_bits") != int(generation_options["--first-prime-bits"])
    or context.get("hamming_weight") != int(generation_options["--hamming-weight"])
    or context.get("q_part_count") != int(generation_options["--q-part-count"])
):
    raise SystemExit("bootstrap context manifest is not the exact schema-v3 qualification context")
resources = load("bootstrap_qualification/compiler_resource_manifest.json")
required_true = {
    "relinearization_key", "conjugation_key", "rotate_batch", "raise_mod",
    "complex_plaintext",
}
if (
    resources.get("schema_version") != 3
    or resources.get("context_schema_version") != 1
    or any(resources.get(field) is not True for field in required_true)
    or resources.get("native_bootstrap_precompute") is not False
    or not resources.get("rotation_steps")
    or not resources.get("rotation_batches")
    or not resources.get("monomial_powers")
):
    raise SystemExit("bootstrap resource manifest is not the full primitive qualification set")
context_sha = digest("bootstrap_qualification/compiler_context_manifest.json")
constants = load("bootstrap_qualification/compiler_constant_manifest.json")
if (
    constants.get("schema_version") != 1
    or constants.get("context_schema_version") != 1
    or constants.get("resource_schema_version") != 3
    or constants.get("context_manifest_sha256") != context_sha
    or not constants.get("constants")
):
    raise SystemExit("bootstrap constant manifest is empty or not context-bound")
for relative in (
    "bootstrap_qualification/compiler_context_manifest.json",
    "bootstrap_qualification/compiler_resource_manifest.json",
    "bootstrap_qualification/compiler_constant_manifest.json",
):
    if (root / relative).read_bytes().endswith(b"\n"):
        raise SystemExit("bootstrap compiler manifests are not canonical no-newline JSON")

generation = load("bootstrap_qualification/generation.json")
compiler_invocation = load("bootstrap_qualification/compiler_invocation.json")
expected_compiler_parameters = compiler_invocation.get("options")
if not isinstance(expected_compiler_parameters, dict):
    raise SystemExit("bootstrap compiler invocation lacks typed options")
generation_manifest_paths = {
    "context": "bootstrap_qualification/compiler_context_manifest.json",
    "resource": "bootstrap_qualification/compiler_resource_manifest.json",
    "constant": "bootstrap_qualification/compiler_constant_manifest.json",
}
expected_generation_keys = {
    "schema_version", "status", "qualification_scope", "compiler_parameters",
    "bootstrap_parameters", "stages_completed", "source", "air", "manifests",
    "constant_count", "rotation_count", "rotation_batch_count", "monomial_count",
    "native_bootstrap_precompute", "generated_program_executed",
}
generated_source_path = root / "bootstrap_qualification/bootstrap_qualification.cu"
if (
    set(generation) != expected_generation_keys
    or generation.get("schema_version") != "ace.phantom.bootstrap-generation/3.0.0"
    or generation.get("status") != "pass"
    or generation.get("qualification_scope")
       != "full-generated-bootstrap-compile-only"
    or generation.get("generated_program_executed") is not False
    or generation.get("native_bootstrap_precompute") is not False
    or generation.get("compiler_parameters") != expected_compiler_parameters
    or generation.get("bootstrap_parameters") != {
        "q_parts": expected_compiler_parameters["q_part_count"],
        "enc_budget": expected_compiler_parameters["encode_transform_budget"],
        "dec_budget": expected_compiler_parameters["decode_transform_budget"],
        "ct_encode": expected_compiler_parameters["ciphertext_constant_encoding"] == "enabled",
    }
    or generation.get("stages_completed") != ["ckks_driver", "ckks2c"]
    or generation.get("source") != {
        "path": "bootstrap_qualification.cu",
        "sha256": digest("bootstrap_qualification/bootstrap_qualification.cu"),
        "bytes": generated_source_path.stat().st_size,
    }
    or generation.get("constant_count") != len(constants["constants"])
    or generation.get("rotation_count") != len(resources["rotation_steps"])
    or generation.get("rotation_batch_count") != len(resources["rotation_batches"])
    or generation.get("monomial_count") != len(resources["monomial_powers"])
):
    raise SystemExit("bootstrap generation record is incomplete or stale")
generation_air_paths = {
    "raw": "bootstrap_qualification/bootstrap_raw.air",
    "post_ckks": "bootstrap_qualification/bootstrap_post_ckks.air",
}
if set(generation.get("air", {})) != set(generation_air_paths):
    raise SystemExit("bootstrap generation AIR binding has an invalid shape")
for name, relative in generation_air_paths.items():
    entry = generation.get("air", {}).get(name, {})
    if (
        entry.get("path") != Path(relative).name
        or entry.get("sha256") != digest(relative)
        or entry.get("bytes") != (root / relative).stat().st_size
    ):
        raise SystemExit(f"bootstrap generation record does not bind {name} AIR")
if set(generation.get("manifests", {})) != set(generation_manifest_paths):
    raise SystemExit("bootstrap generation manifest binding has an invalid shape")
for name, relative in generation_manifest_paths.items():
    entry = generation.get("manifests", {}).get(name, {})
    if (
        entry.get("path") != Path(relative).name
        or entry.get("sha256") != digest(relative)
        or entry.get("bytes") != (root / relative).stat().st_size
    ):
        raise SystemExit(f"bootstrap generation record does not bind {name}")

audit = load("bootstrap_qualification/source-audit.json")
audit_inputs = {
    "raw_air": "bootstrap_qualification/bootstrap_raw.air",
    "post_ckks_air": "bootstrap_qualification/bootstrap_post_ckks.air",
    "context_manifest": "bootstrap_qualification/compiler_context_manifest.json",
    "resource_manifest": "bootstrap_qualification/compiler_resource_manifest.json",
    "constant_manifest": "bootstrap_qualification/compiler_constant_manifest.json",
    "source": "bootstrap_qualification/bootstrap_qualification.cu",
}
if not isinstance(audit.get("inputs"), dict) or set(audit["inputs"]) != set(audit_inputs):
    raise SystemExit("bootstrap generated-artifact audit inputs have an invalid shape")
required_opcodes = (
    "ckks.conjugate", "ckks.rotate_batch", "ckks.raise_mod", "ckks.mul_mono",
)
required_calls = (
    "Conjugate_ciph", "Rotate_batch_ciph", "Raise_mod", "Mul_mono_ciph",
)
air_records = {}
for stage, relative in (
    ("raw", audit_inputs["raw_air"]),
    ("post_ckks", audit_inputs["post_ckks_air"]),
):
    air_text = (root / relative).read_text(encoding="utf-8")
    if re.search(
        r"\bckks\.bootstrap(?:\b|[._])|"
        r"\b(?:bootstrap_coeffs_to_slots|bootstrap_eval_mod|"
        r"bootstrap_slots_to_coeffs)\b|"
        r"(?:\bfhe::(?:poly|hpoly|lpoly)\b|\b(?:poly|hpoly|lpoly)\.[A-Za-z_])",
        air_text,
        re.IGNORECASE,
    ):
        raise SystemExit(f"bootstrap {stage} AIR contains a forbidden operation")
    lowered_air = air_text.lower()
    opcode_counts = {name: lowered_air.count(name) for name in required_opcodes}
    if any(count <= 0 for count in opcode_counts.values()):
        raise SystemExit(f"bootstrap {stage} AIR lacks a required primitive opcode")
    air_records[stage] = {
        "required_opcode_counts": opcode_counts,
        "forbidden_matches": [],
    }
generated_source = (root / audit_inputs["source"]).read_text(encoding="utf-8")
source_counts = {
    name: len(re.findall(r"\b" + re.escape(name) + r"\s*\(", generated_source))
    for name in required_calls
}
if any(count <= 0 for count in source_counts.values()):
    raise SystemExit("bootstrap generated source lacks a required primitive call")
if re.search(
    r"Bootstrapper|Phantom_bootstrap|bootstrap_3|\bBootstrap\s*\(|"
    r"\bEval_bootstrap[A-Za-z0-9_]*\s*\(|"
    r"\b(?:bootstrap_coeffs_to_slots|CoeffToSlots?|bootstrap_eval_mod|EvalMod|"
    r"bootstrap_slots_to_coeffs|SlotToCoeffs?)\b",
    generated_source,
):
    raise SystemExit("bootstrap generated source contains a forbidden operation")
expected_audit_keys = {
    "schema_version", "status", "inputs", "errors", "air", "source",
    "forbidden_matches", "forbidden_native_bts_matches", "counts",
}
if (
    set(audit) != expected_audit_keys
    or audit.get("schema_version")
       != "ace.phantom.bootstrap-generated-artifact-audit/2.0.0"
    or audit.get("status") != "pass"
    or audit.get("errors") != []
    or audit.get("air") != air_records
    or audit.get("source") != {
        "required_call_counts": source_counts, "forbidden_matches": [],
    }
    or audit.get("forbidden_matches") != {
        "raw_air": [], "post_ckks_air": [], "source": [],
    }
    or audit.get("forbidden_native_bts_matches") != []
    or audit.get("counts") != {
        "constants": len(constants["constants"]),
        "monomial_powers": len(resources["monomial_powers"]),
        "rotation_batches": len(resources["rotation_batches"]),
        "rotation_steps": len(resources["rotation_steps"]),
    }
):
    raise SystemExit("bootstrap generated-artifact audit did not pass")
for name, relative in audit_inputs.items():
    entry = audit["inputs"].get(name, {})
    if entry != {
        "path": Path(relative).name,
        "sha256": digest(relative),
        "size_bytes": (root / relative).stat().st_size,
    }:
        raise SystemExit(f"bootstrap source audit does not bind {name}")

qualification = load("bootstrap_qualification/qualification.json")
expected_qualification = {
    "status": "pass",
    "gate": "bootstrap",
    "architecture": "sm_80",
    "phantom_commit": phantom_commit,
    "generated_source_sha256": digest(
        "bootstrap_qualification/bootstrap_qualification.cu"
    ),
    "raw_air_sha256": digest("bootstrap_qualification/bootstrap_raw.air"),
    "post_ckks_air_sha256": digest(
        "bootstrap_qualification/bootstrap_post_ckks.air"
    ),
    "compiler_context_manifest_sha256": context_sha,
    "compiler_resource_manifest_sha256": digest(
        "bootstrap_qualification/compiler_resource_manifest.json"
    ),
    "compiler_constant_manifest_sha256": digest(
        "bootstrap_qualification/compiler_constant_manifest.json"
    ),
    "generation_record_sha256": digest(
        "bootstrap_qualification/generation.json"
    ),
    "generated_artifact_audit_sha256": digest(
        "bootstrap_qualification/source-audit.json"
    ),
    "linked_binary_sha256": digest(
        "bootstrap_qualification/bootstrap_phantom_constants_sm80"
    ),
    "harness_source_sha256": digest(
        "bootstrap_qualification/bootstrap_phantom_constants.cu"
    ),
    "symbol_closure_sha256": digest(
        "bootstrap_qualification/symbol-closure.json"
    ),
    "io_helper_closure_sha256": digest(
        "bootstrap_qualification/io-helper-closure.json"
    ),
    "archive_member_audit_sha256": digest(
        "bootstrap_qualification/archive-member-audit.json"
    ),
}
if any(qualification.get(key) != value for key, value in expected_qualification.items()):
    raise SystemExit("bootstrap host qualification hashes are incomplete or stale")
if (
    qualification.get("generated_source_contains_native_bootstrap") is not False
    or qualification.get("production_archive_contains_native_bootstrap") is not False
    or qualification.get("primitive_only_provider_archive") is not True
    or qualification.get("executable_was_run") is not False
    or qualification.get("context_contract") != {
        "polynomial_degree": context["polynomial_degree"],
        "mul_level": len(context["data_q_bit_sizes"]),
        "input_level": context["input_level"],
        "security_level": context["security_level"],
        "scaling_factor_bits": context["scaling_modulus_bits"],
        "first_prime_bits": context["first_modulus_bits"],
        "hamming_weight": context["hamming_weight"],
    }
):
    raise SystemExit("bootstrap host qualification claims are inconsistent")
for archive_field in (
    "adapter_archive_sha256", "provider_archive_sha256", "common_archive_sha256"
):
    if re.fullmatch(r"[0-9a-f]{64}", qualification.get(archive_field, "")) is None:
        raise SystemExit(f"bootstrap host qualification lacks {archive_field}")

artifact = load("artifact_manifest.json")
artifact_files = artifact.get("files", {})
expected_artifact_files = {
    relative: checksum
    for relative, checksum in listed.items()
    if relative != "artifact_manifest.json"
}
if (
    artifact.get("schema_version")
       != "ace.phantom.bootstrap-artifacts/3.0.0"
    or artifact.get("status") != "bound"
    or artifact.get("source_mode") != "snapshot"
    or artifact.get("ace_commit") != ace_commit
    or artifact.get("phantom_commit") != phantom_commit
    or artifact.get("normalized_qualification_argv_sha256")
       != qualification_invocation["normalized_argv_sha256"]
    or artifact.get("normalized_generation_argv_sha256")
       != generation_invocation["normalized_argv_sha256"]
    or artifact.get("raw_air_sha256")
       != digest("bootstrap_qualification/bootstrap_raw.air")
    or artifact.get("post_ckks_air_sha256")
       != digest("bootstrap_qualification/bootstrap_post_ckks.air")
    or artifact.get("compiler_context_manifest_sha256") != context_sha
    or artifact.get("compiler_resource_manifest_sha256")
       != digest("bootstrap_qualification/compiler_resource_manifest.json")
    or artifact.get("compiler_constant_manifest_sha256")
       != digest("bootstrap_qualification/compiler_constant_manifest.json")
    or artifact.get("generation_record_sha256")
       != digest("bootstrap_qualification/generation.json")
    or artifact.get("generated_artifact_audit_sha256")
       != digest("bootstrap_qualification/source-audit.json")
    or artifact.get("linked_binary_sha256")
       != digest("bootstrap_qualification/bootstrap_phantom_constants_sm80")
    or artifact.get("harness_source_sha256") != expected_harness_sha
    or artifact_files != expected_artifact_files
):
    raise SystemExit("bootstrap artifact manifest is incomplete or stale")

symbol_closure = load("bootstrap_qualification/symbol-closure.json")
required_symbols = symbol_closure.get("required_symbols")
expected_required_symbols = [
    "bootstrap_full",
    "Conjugate_ciph",
    "Rotate_batch_ciph",
    "Raise_mod",
    "Mul_mono_ciph",
    "Phantom_conjugate",
    "Phantom_rotate_batch",
    "Phantom_raise_mod",
    "Phantom_mul_mono",
    "complex_conjugate_inplace",
    "rotate_batch",
    "raise_modulus",
    "multiply_by_monomial",
    "Get_phantom_context_manifest",
    "Get_phantom_resource_manifest",
    "Get_phantom_constant_manifest",
    "Load_cached_plain",
    "main",
]
linked_symbols = (
    root / "bootstrap_qualification/linked_binary_symbols.txt"
).read_text(encoding="utf-8")
forbidden_symbol_pattern = re.compile(
    r"Bootstrapper|\bBootstrap\b|Phantom_bootstrap|Eval_bootstrap|bootstrap_3|"
    r"bootstrap_(?:coeffs_to_slots|eval_mod|slots_to_coeffs)|"
    r"cnn_phantom|conv_eval|FHErt_(?:ant|poly)|fhe::(?:ant|poly)|"
    r"CoeffToSlot|SlotToCoeff|EvalMod|Native.*precom|precom.*Native|"
    r"Bootstrap.*stage|stage.*Bootstrap",
    re.IGNORECASE,
)
if (
    symbol_closure.get("schema_version")
       != "ace.phantom.bootstrap-symbol-closure/2.0.0"
    or symbol_closure.get("status") != "pass"
    or required_symbols != expected_required_symbols
    or symbol_closure.get("missing_symbols") != []
    or symbol_closure.get("native_bootstrap_symbol_count") != 0
    or symbol_closure.get("linked_binary_symbols_sha256")
       != digest("bootstrap_qualification/linked_binary_symbols.txt")
    or any(
        re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(symbol) + r"(?![A-Za-z0-9_])",
            linked_symbols,
        ) is None
        for symbol in expected_required_symbols
    )
    or forbidden_symbol_pattern.search(linked_symbols) is not None
):
    raise SystemExit("bootstrap semantic symbol closure is incomplete or stale")
if "sm_80" not in (
    root / "bootstrap_qualification/cuda_elf.txt"
).read_text(encoding="utf-8"):
    raise SystemExit("bootstrap CUDA image inventory lacks sm_80")
link_commands = (
    root / "bootstrap_qualification/link-commands.txt"
).read_text(encoding="utf-8")
for token in ("-arch=sm_80", "-dlink", "bootstrap_phantom_constants_sm80"):
    if token not in link_commands:
        raise SystemExit(f"bootstrap link command record lacks {token}")

run_manifest = load("manifest.json")
if (
    run_manifest.get("status") != "pass"
    or run_manifest.get("gate") != "bootstrap"
    or run_manifest.get("source_mode") != "snapshot"
    or run_manifest.get("ace_commit") != ace_commit
    or run_manifest.get("phantom_commit") != phantom_commit
    or run_manifest.get("ace_worktree_dirty") is not False
    or run_manifest.get("qualification_invocation_sha256")
       != digest("qualification_invocation.json")
    or run_manifest.get("normalized_qualification_argv_sha256")
       != qualification_invocation["normalized_argv_sha256"]
    or run_manifest.get("artifact_manifest_sha256") != digest("artifact_manifest.json")
    or run_manifest.get("evidence_sha256_manifest_sha256") != digest("SHA256SUMS")
    or run_manifest.get("compiler_context_manifest_sha256") != context_sha
    or run_manifest.get("compiler_resource_manifest_sha256")
       != digest("bootstrap_qualification/compiler_resource_manifest.json")
    or run_manifest.get("compiler_constant_manifest_sha256")
       != digest("bootstrap_qualification/compiler_constant_manifest.json")
    or run_manifest.get("bootstrap_raw_air_sha256")
       != digest("bootstrap_qualification/bootstrap_raw.air")
    or run_manifest.get("bootstrap_post_ckks_air_sha256")
       != digest("bootstrap_qualification/bootstrap_post_ckks.air")
    or run_manifest.get("bootstrap_generation_record_sha256")
       != digest("bootstrap_qualification/generation.json")
    or run_manifest.get("bootstrap_generated_artifact_audit_sha256")
       != digest("bootstrap_qualification/source-audit.json")
    or run_manifest.get("bootstrap_linked_binary_sha256")
       != digest("bootstrap_qualification/bootstrap_phantom_constants_sm80")
    or run_manifest.get("bootstrap_harness_source_sha256") != expected_harness_sha
    or run_manifest.get("bootstrap_host_qualification_sha256")
       != digest("bootstrap_qualification/qualification.json")
):
    raise SystemExit("bootstrap run manifest is incomplete or stale")

ace_source = root / "ace_source_manifest.json"
phantom_source = root / "phantom_source_manifest.json"
if (
    ace_source.read_bytes() != ordinary_ace_source.read_bytes()
    or phantom_source.read_bytes() != ordinary_phantom_source.read_bytes()
    or run_manifest.get("ace_tracked_source_manifest_sha256")
       != hashlib.sha256(ace_source.read_bytes()).hexdigest()
    or run_manifest.get("phantom_source_manifest_sha256")
       != hashlib.sha256(phantom_source.read_bytes()).hexdigest()
):
    raise SystemExit("bootstrap source snapshot manifests differ from ordinary evidence")
if digest("bootstrap_qualification/bootstrap_phantom_constants.cu") != expected_harness_sha:
    raise SystemExit("bootstrap harness source differs from the selected ACE commit")
PY

while IFS= read -r retained_evidence_dependency; do
  expected_dependency_sha256="$(
    git -C "${REPO_ROOT}" show \
      "${ACE_COMMIT}:${retained_evidence_dependency}" |
      sha256sum | awk '{print $1}'
  )"
  actual_dependency_sha256="$(
    sha256sum "${REPO_ROOT}/${retained_evidence_dependency}" | awk '{print $1}'
  )"
  if [[ "${actual_dependency_sha256}" != "${expected_dependency_sha256}" ]]; then
    echo "retained evidence dependency differs from the selected ACE commit: ${retained_evidence_dependency}" >&2
    exit 1
  fi
done <<'FILES'
tools/phantom_gpu/package_runpod_sources.sh
tools/phantom_gpu/retained_runpod_evidence.py
tools/phantom_gpu/compare_retained_ckks_results.py
tools/phantom_gpu/generate_retained_ckks_fixtures.py
FILES
python3 "${SCRIPT_DIR}/retained_runpod_evidence.py" validate-frozen \
  --root "${RETAINED_RUN_ROOT}" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}"

DEPENDENCY_LOCK="$(
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
)"
lock_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" <<<"${DEPENDENCY_LOCK}")"
  if [[ -z "${value}" || "${value}" == *$'\n'* ]]; then
    echo "selected ACE commit has an invalid ${key} dependency lock" >&2
    exit 1
  fi
  printf '%s\n' "${value}"
}
LOCKED_PHANTOM_COMMIT="$(lock_value PHANTOM_COMMIT)"
if [[ "${PHANTOM_COMMIT}" != "${LOCKED_PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit does not match the selected ACE commit lock" >&2
  echo "requested: ${PHANTOM_COMMIT}" >&2
  echo "locked:    ${LOCKED_PHANTOM_COMMIT}" >&2
  exit 1
fi

OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
if [[ -e "${OUTPUT}" ]]; then
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}"
chmod 0700 "${OUTPUT}"

ARCHIVE_TOOL_DIRECTORY="$(mktemp -d)"
cleanup_archive_tool() {
  rm -f -- "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"
  rmdir -- "${ARCHIVE_TOOL_DIRECTORY}"
}
trap cleanup_archive_tool EXIT
git -C "${REPO_ROOT}" show \
  "${ACE_COMMIT}:tools/phantom_gpu/source_archive.py" \
  >"${ARCHIVE_TOOL_DIRECTORY}/source_archive.py"

ACE_ARCHIVE="ace-source-${ACE_COMMIT}.tar.gz"
PHANTOM_ARCHIVE="phantom-source-${PHANTOM_COMMIT}.tar.gz"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${REPO_ROOT}" \
  --commit "${ACE_COMMIT}" \
  --kind ace \
  --output "${OUTPUT}/${ACE_ARCHIVE}" \
  --manifest "${OUTPUT}/ace-source.manifest.json" \
  >"${OUTPUT}/ace-source.audit.json"
python3 "${ARCHIVE_TOOL_DIRECTORY}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" \
  --commit "${PHANTOM_COMMIT}" \
  --kind phantom \
  --output "${OUTPUT}/${PHANTOM_ARCHIVE}" \
  --manifest "${OUTPUT}/phantom-source.manifest.json" \
  >"${OUTPUT}/phantom-source.audit.json"
cmp "${FROZEN_ACE_SOURCE_MANIFEST}" "${OUTPUT}/ace-source.manifest.json"
cmp "${FROZEN_PHANTOM_SOURCE_MANIFEST}" \
  "${OUTPUT}/phantom-source.manifest.json"
cmp "${RETAINED_RUN_ROOT}/source/ace_source_manifest.json" \
  "${OUTPUT}/ace-source.manifest.json"
cmp "${RETAINED_RUN_ROOT}/source/phantom_source_manifest.json" \
  "${OUTPUT}/phantom-source.manifest.json"

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" >"${OUTPUT}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/run_build_and_health.sh
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 \
  "${OUTPUT}/source_archive.py" \
  "${OUTPUT}/phase_helpers.sh" \
  "${OUTPUT}/bootstrap_environment.sh" \
  "${OUTPUT}/run_build_and_health.sh"

cp "${FROZEN_CONTEXT}" "${OUTPUT}/ordinary-context-manifest.json"
cp "${FROZEN_QUALIFICATION_INVOCATION}" \
  "${OUTPUT}/qualification-invocation.json"
cp "${FROZEN_COMPILER_INVOCATION}" "${OUTPUT}/compiler-invocation.json"
cp "${FROZEN_RESOURCES}" "${OUTPUT}/ordinary-resource-manifest.json"
cp "${FROZEN_CONSTANTS}" "${OUTPUT}/ordinary-constant-manifest.json"
cp "${FROZEN_POST_CKKS_AIR}" "${OUTPUT}/ordinary-post-ckks.air"
cp "${FROZEN_ARTIFACT_MANIFEST}" "${OUTPUT}/ordinary-artifact-manifest.json"
cp "${FROZEN_RUN_MANIFEST}" "${OUTPUT}/ordinary-run-manifest.json"
cp "${FROZEN_HOST_QUALIFICATION}" \
  "${OUTPUT}/ordinary-host-qualification.json"
cp "${FROZEN_FIXTURE}" "${OUTPUT}/ordinary-fixture.json"
cp "${FROZEN_CPU_REFERENCE}" "${OUTPUT}/ordinary-cpu-reference.json"
cp "${FROZEN_CPU_VALUES}" "${OUTPUT}/ordinary-cpu-values.bin"
cp "${FROZEN_ANT_VERIFICATION}" "${OUTPUT}/ordinary-ant-verification.json"
cp "${BOOTSTRAP_QUALIFICATION_INVOCATION}" \
  "${OUTPUT}/bootstrap-qualification-invocation.json"
cp "${BOOTSTRAP_GENERATION_INVOCATION}" \
  "${OUTPUT}/bootstrap-generation-invocation.json"
cp "${BOOTSTRAP_ARTIFACT_MANIFEST}" \
  "${OUTPUT}/bootstrap-artifact-manifest.json"
cp "${BOOTSTRAP_RUN_MANIFEST}" "${OUTPUT}/bootstrap-run-manifest.json"
cp "${BOOTSTRAP_CONTEXT}" "${OUTPUT}/bootstrap-context-manifest.json"
cp "${BOOTSTRAP_RESOURCES}" "${OUTPUT}/bootstrap-resource-manifest.json"
cp "${BOOTSTRAP_CONSTANTS}" "${OUTPUT}/bootstrap-constant-manifest.json"
cp "${BOOTSTRAP_GENERATION}" "${OUTPUT}/bootstrap-generation.json"
cp "${BOOTSTRAP_SOURCE_AUDIT}" "${OUTPUT}/bootstrap-source-audit.json"
cp "${BOOTSTRAP_HOST_QUALIFICATION}" \
  "${OUTPUT}/bootstrap-host-qualification.json"
cp "${BOOTSTRAP_RAW_AIR}" "${OUTPUT}/bootstrap-raw.air"
cp "${BOOTSTRAP_POST_CKKS_AIR}" "${OUTPUT}/bootstrap-post-ckks.air"
cp "${BOOTSTRAP_GENERATED_SOURCE}" "${OUTPUT}/bootstrap-generated.cu"
cp "${BOOTSTRAP_HOST_BINARY}" "${OUTPUT}/bootstrap-host-linked-sm80"
cp "${BOOTSTRAP_LINK_COMMANDS}" "${OUTPUT}/bootstrap-link-commands.txt"
cp "${BOOTSTRAP_CUDA_ELF}" "${OUTPUT}/bootstrap-cuda-elf.txt"
cp "${BOOTSTRAP_CUDA_RESOURCES}" "${OUTPUT}/bootstrap-cuda-resources.txt"
cp "${BOOTSTRAP_FILE_REPORT}" "${OUTPUT}/bootstrap-file-report.txt"
cp "${BOOTSTRAP_READELF}" "${OUTPUT}/bootstrap-readelf.txt"
cp "${BOOTSTRAP_LINKED_SYMBOLS}" "${OUTPUT}/bootstrap-linked-symbols.txt"
cp "${BOOTSTRAP_SYMBOL_CLOSURE}" "${OUTPUT}/bootstrap-symbol-closure.json"
python3 "${SCRIPT_DIR}/retained_runpod_evidence.py" export-frozen \
  --root "${RETAINED_RUN_ROOT}" \
  --output "${OUTPUT}" \
  --ace-commit "${ACE_COMMIT}" \
  --phantom-commit "${PHANTOM_COMMIT}"

(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

python3 - "${OUTPUT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
payload = {
    "schema_version": "2.0.0",
    "contents": (
        "audited-source-snapshots-and-frozen-provider-neutral-references-"
        "with-exact-bootstrap-compile-handoff-and-regeneration-attestations"
    ),
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "source_snapshots": {
        "ace_manifest_sha256": hashlib.sha256(
            (output / "ace-source.manifest.json").read_bytes()
        ).hexdigest(),
        "phantom_manifest_sha256": hashlib.sha256(
            (output / "phantom-source.manifest.json").read_bytes()
        ).hexdigest(),
    },
    "qualification_invocation": {
        "sha256": hashlib.sha256(
            (output / "qualification-invocation.json").read_bytes()
        ).hexdigest(),
        "normalized_argv_sha256": json.loads(
            (output / "qualification-invocation.json").read_text(encoding="utf-8")
        )["normalized_argv_sha256"],
    },
    "compiler_invocation": {
        "sha256": hashlib.sha256(
            (output / "compiler-invocation.json").read_bytes()
        ).hexdigest(),
        "normalized_argv_sha256": json.loads(
            (output / "compiler-invocation.json").read_text(encoding="utf-8")
        )["normalized_argv_sha256"],
    },
    "frozen_ordinary_reference": {
        "context_manifest_sha256": hashlib.sha256(
            (output / "ordinary-context-manifest.json").read_bytes()
        ).hexdigest(),
        "resource_manifest_sha256": hashlib.sha256(
            (output / "ordinary-resource-manifest.json").read_bytes()
        ).hexdigest(),
        "constant_manifest_sha256": hashlib.sha256(
            (output / "ordinary-constant-manifest.json").read_bytes()
        ).hexdigest(),
        "post_ckks_air_sha256": hashlib.sha256(
            (output / "ordinary-post-ckks.air").read_bytes()
        ).hexdigest(),
        "artifact_manifest_sha256": hashlib.sha256(
            (output / "ordinary-artifact-manifest.json").read_bytes()
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            (output / "ordinary-run-manifest.json").read_bytes()
        ).hexdigest(),
        "host_qualification_sha256": hashlib.sha256(
            (output / "ordinary-host-qualification.json").read_bytes()
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            (output / "ordinary-fixture.json").read_bytes()
        ).hexdigest(),
        "cpu_reference_sha256": hashlib.sha256(
            (output / "ordinary-cpu-reference.json").read_bytes()
        ).hexdigest(),
        "cpu_values_sha256": hashlib.sha256(
            (output / "ordinary-cpu-values.bin").read_bytes()
        ).hexdigest(),
        "ant_verification_sha256": hashlib.sha256(
            (output / "ordinary-ant-verification.json").read_bytes()
        ).hexdigest(),
    },
    "bootstrap_qualification": {
        "qualification_invocation_sha256": hashlib.sha256(
            (output / "bootstrap-qualification-invocation.json").read_bytes()
        ).hexdigest(),
        "normalized_qualification_argv_sha256": json.loads(
            (output / "bootstrap-qualification-invocation.json").read_text(
                encoding="utf-8"
            )
        )["normalized_argv_sha256"],
        "generation_invocation_sha256": hashlib.sha256(
            (output / "bootstrap-generation-invocation.json").read_bytes()
        ).hexdigest(),
        "normalized_generation_argv_sha256": json.loads(
            (output / "bootstrap-generation-invocation.json").read_text(
                encoding="utf-8"
            )
        )["normalized_argv_sha256"],
        "artifact_manifest_sha256": hashlib.sha256(
            (output / "bootstrap-artifact-manifest.json").read_bytes()
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            (output / "bootstrap-run-manifest.json").read_bytes()
        ).hexdigest(),
        "context_manifest_sha256": hashlib.sha256(
            (output / "bootstrap-context-manifest.json").read_bytes()
        ).hexdigest(),
        "resource_manifest_sha256": hashlib.sha256(
            (output / "bootstrap-resource-manifest.json").read_bytes()
        ).hexdigest(),
        "constant_manifest_sha256": hashlib.sha256(
            (output / "bootstrap-constant-manifest.json").read_bytes()
        ).hexdigest(),
        "generation_record_sha256": hashlib.sha256(
            (output / "bootstrap-generation.json").read_bytes()
        ).hexdigest(),
        "source_audit_sha256": hashlib.sha256(
            (output / "bootstrap-source-audit.json").read_bytes()
        ).hexdigest(),
        "host_qualification_sha256": hashlib.sha256(
            (output / "bootstrap-host-qualification.json").read_bytes()
        ).hexdigest(),
        "raw_air_sha256": hashlib.sha256(
            (output / "bootstrap-raw.air").read_bytes()
        ).hexdigest(),
        "post_ckks_air_sha256": hashlib.sha256(
            (output / "bootstrap-post-ckks.air").read_bytes()
        ).hexdigest(),
        "generated_source_sha256": hashlib.sha256(
            (output / "bootstrap-generated.cu").read_bytes()
        ).hexdigest(),
        "host_linked_binary_sha256": hashlib.sha256(
            (output / "bootstrap-host-linked-sm80").read_bytes()
        ).hexdigest(),
        "harness_source_sha256": json.loads(
            (output / "bootstrap-artifact-manifest.json").read_text(
                encoding="utf-8"
            )
        )["harness_source_sha256"],
        "link_commands_sha256": hashlib.sha256(
            (output / "bootstrap-link-commands.txt").read_bytes()
        ).hexdigest(),
        "cuda_elf_sha256": hashlib.sha256(
            (output / "bootstrap-cuda-elf.txt").read_bytes()
        ).hexdigest(),
        "cuda_resources_sha256": hashlib.sha256(
            (output / "bootstrap-cuda-resources.txt").read_bytes()
        ).hexdigest(),
        "file_report_sha256": hashlib.sha256(
            (output / "bootstrap-file-report.txt").read_bytes()
        ).hexdigest(),
        "readelf_sha256": hashlib.sha256(
            (output / "bootstrap-readelf.txt").read_bytes()
        ).hexdigest(),
        "linked_symbols_sha256": hashlib.sha256(
            (output / "bootstrap-linked-symbols.txt").read_bytes()
        ).hexdigest(),
        "symbol_closure_sha256": hashlib.sha256(
            (output / "bootstrap-symbol-closure.json").read_bytes()
        ).hexdigest(),
        "architecture": "sm_80",
        "packaged_compile_handoff": True,
        "linked_binary_byte_identity_required": False,
    },
    "frozen_retained_reference": json.loads(
        (output / "retained-frozen-export.json").read_text(encoding="utf-8")
    ),
    "files": sorted(path.name for path in output.iterdir() if path.is_file()),
}
(output / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
(
  cd "${OUTPUT}"
  sha256sum payload.json >>SHA256SUMS
  sha256sum -c SHA256SUMS
)
echo "audited RunPod source payload: ${OUTPUT}"
