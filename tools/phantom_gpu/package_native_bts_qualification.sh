#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PHANTOM_REPO="${ACE_PHANTOM_REPO:-/home/dyf/code/phantom-ant}"

usage() {
  echo "usage: $0 --ace-commit COMMIT --phantom-commit COMMIT --baseline-package DIR --baseline-runpod-archive FILE --baseline-host-results DIR OUTPUT_DIRECTORY" >&2
  exit 2
}

ACE_COMMIT=""
PHANTOM_COMMIT=""
BASELINE_PACKAGE=""
BASELINE_RUNPOD_ARCHIVE=""
BASELINE_HOST_RESULTS=""
OUTPUT_ARGUMENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ace-commit) ACE_COMMIT="$2"; shift 2 ;;
    --phantom-commit) PHANTOM_COMMIT="$2"; shift 2 ;;
    --baseline-package) BASELINE_PACKAGE="$2"; shift 2 ;;
    --baseline-runpod-archive) BASELINE_RUNPOD_ARCHIVE="$2"; shift 2 ;;
    --baseline-host-results) BASELINE_HOST_RESULTS="$2"; shift 2 ;;
    -*) usage ;;
    *) [[ -z "${OUTPUT_ARGUMENT}" ]] || usage
       OUTPUT_ARGUMENT="$1"; shift ;;
  esac
done
if [[ ! "${ACE_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      ! "${PHANTOM_COMMIT}" =~ ^[0-9a-f]{40}$ ||
      -z "${BASELINE_PACKAGE}" || -z "${BASELINE_RUNPOD_ARCHIVE}" ||
      -z "${BASELINE_HOST_RESULTS}" || -z "${OUTPUT_ARGUMENT}" ]]; then
  usage
fi

git -C "${REPO_ROOT}" cat-file -e "${ACE_COMMIT}^{commit}"
git -C "${PHANTOM_REPO}" cat-file -e "${PHANTOM_COMMIT}^{commit}"
selected_phantom="$({
  git -C "${REPO_ROOT}" show \
    "${ACE_COMMIT}:tools/phantom_gpu/configs/dependencies.env"
} | sed -n 's/^PHANTOM_COMMIT=//p')"
if [[ "${selected_phantom}" != "${PHANTOM_COMMIT}" ]]; then
  echo "requested Phantom commit differs from the selected ACE dependency lock" >&2
  exit 1
fi
for relative in \
  tools/phantom_gpu/package_native_bts_qualification.sh \
  tools/phantom_gpu/run_build_and_health.sh \
  tools/phantom_gpu/native_bts_pipeline.sh \
  tools/phantom_gpu/native_bts_correctness.py \
  tools/phantom_gpu/harness/native_phantom_bts_correctness.cu; do
  expected="$(git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" | sha256sum | awk '{print $1}')"
  observed="$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')"
  [[ "${expected}" == "${observed}" ]] || {
    echo "packaging input differs from selected ACE commit: ${relative}" >&2
    exit 1
  }
done

BASELINE_PACKAGE="$(realpath -- "${BASELINE_PACKAGE}")"
BASELINE_RUNPOD_ARCHIVE="$(realpath -- "${BASELINE_RUNPOD_ARCHIVE}")"
BASELINE_HOST_RESULTS="$(realpath -- "${BASELINE_HOST_RESULTS}")"
OUTPUT="$(realpath -m -- "${OUTPUT_ARGUMENT}")"
[[ ! -e "${OUTPUT}" ]] || {
  echo "output already exists: ${OUTPUT}" >&2
  exit 1
}

(
  cd "${BASELINE_PACKAGE}"
  sha256sum -c SHA256SUMS
) >/dev/null
(
  cd "$(dirname -- "${BASELINE_RUNPOD_ARCHIVE}")"
  sha256sum -c "$(basename -- "${BASELINE_RUNPOD_ARCHIVE}").sha256"
) >/dev/null
(
  cd "${BASELINE_HOST_RESULTS}"
  sha256sum -c SHA256SUMS
) >/dev/null

temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT
python3 - "${BASELINE_RUNPOD_ARCHIVE}" "${temporary}/remote" <<'PY'
import hashlib
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

archive, output = map(Path, sys.argv[1:])
wanted = {
    "results/SHA256SUMS",
    "results/generated-phantom-correctness.json",
    "results/generated-phantom-correctness.bin",
    "results/generated-bootstrap-three-way-comparison.json",
    "results/correctness-sanitizer.json",
}
output.mkdir()
with tarfile.open(archive, "r:gz") as bundle:
    members = {member.name: member for member in bundle.getmembers()}
    if not wanted <= members.keys():
        raise SystemExit("closed RunPod archive lacks an M6 correctness artifact")
    sums_stream = bundle.extractfile(members["results/SHA256SUMS"])
    if sums_stream is None:
        raise SystemExit("cannot read closed RunPod checksum manifest")
    listed = {}
    for line in sums_stream.read().decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit("closed RunPod checksum manifest is malformed")
        name = match.group(2).removeprefix("./")
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or name in listed:
            raise SystemExit("closed RunPod checksum path is unsafe")
        listed[name] = match.group(1)
    for name in sorted(wanted - {"results/SHA256SUMS"}):
        member = members[name]
        if not member.isfile():
            raise SystemExit(f"closed RunPod member is not regular: {name}")
        stream = bundle.extractfile(member)
        if stream is None:
            raise SystemExit(f"cannot read closed RunPod member: {name}")
        contents = stream.read()
        relative = name.removeprefix("results/")
        if listed.get(relative) != hashlib.sha256(contents).hexdigest():
            raise SystemExit(f"closed RunPod checksum mismatch: {name}")
        (output / relative).write_bytes(contents)
PY

mkdir "${OUTPUT}"
python3 "${SCRIPT_DIR}/source_archive.py" create \
  --repo "${REPO_ROOT}" --commit "${ACE_COMMIT}" --kind ace \
  --output "${OUTPUT}/ace-source-${ACE_COMMIT}.tar.gz" \
  --manifest "${OUTPUT}/ace-source.manifest.json" \
  >"${OUTPUT}/ace-source.audit.json"
python3 "${SCRIPT_DIR}/source_archive.py" create \
  --repo "${PHANTOM_REPO}" --commit "${PHANTOM_COMMIT}" --kind phantom \
  --output "${OUTPUT}/phantom-source-${PHANTOM_COMMIT}.tar.gz" \
  --manifest "${OUTPUT}/phantom-source.manifest.json" \
  >"${OUTPUT}/phantom-source.audit.json"

while IFS= read -r relative; do
  git -C "${REPO_ROOT}" show "${ACE_COMMIT}:${relative}" \
    >"${OUTPUT}/${relative##*/}"
done <<'FILES'
tools/phantom_gpu/source_archive.py
tools/phantom_gpu/phase_helpers.sh
tools/phantom_gpu/bootstrap_environment.sh
tools/phantom_gpu/run_build_and_health.sh
tools/phantom_gpu/native_bts_pipeline.sh
tools/phantom_gpu/bootstrap_domain_attestation.py
tools/phantom_gpu/bootstrap_correctness.py
tools/phantom_gpu/native_bts_correctness.py
tools/phantom_gpu/harness/native_phantom_bts_correctness.cu
tools/phantom_gpu/configs/apt-packages.lock
tools/phantom_gpu/configs/python-requirements-hashed.lock
tools/phantom_gpu/configs/base-files.sha256
tools/phantom_gpu/configs/dependencies.env
tools/phantom_gpu/configs/toolchain.env
FILES
chmod 0755 "${OUTPUT}"/*.sh "${OUTPUT}"/*.py

declare -a baseline_copies=(
  "ace-source.manifest.json:baseline-ace-source.manifest.json"
  "phantom-source.manifest.json:baseline-phantom-source.manifest.json"
  "correctness-fixture.json:baseline-fixture.json"
  "correctness-compiler-invocation.json:baseline-compiler-invocation.json"
  "correctness-raw.air:baseline-raw.air"
  "correctness-post-ckks.air:baseline-post-ckks.air"
  "correctness-context-manifest.json:baseline-context-manifest.json"
  "correctness-resource-manifest.json:baseline-resource-manifest.json"
  "correctness-constant-manifest.json:baseline-constant-manifest.json"
  "correctness-bootstrap-semantics.json:baseline-bootstrap-semantics.json"
  "correctness-post-operations.air:baseline-post-operations.air"
  "correctness-post-operation-attestation.json:baseline-post-operation-attestation.json"
)
for entry in "${baseline_copies[@]}"; do
  cp -- "${BASELINE_PACKAGE}/${entry%%:*}" "${OUTPUT}/${entry#*:}"
done
cp -- "${temporary}/remote/generated-phantom-correctness.json" \
  "${OUTPUT}/baseline-generated-phantom.json"
cp -- "${temporary}/remote/generated-phantom-correctness.bin" \
  "${OUTPUT}/baseline-generated-phantom.bin"
cp -- "${temporary}/remote/generated-bootstrap-three-way-comparison.json" \
  "${OUTPUT}/baseline-comparison.json"
cp -- "${temporary}/remote/correctness-sanitizer.json" \
  "${OUTPUT}/baseline-sanitizer.json"

qualification="${BASELINE_HOST_RESULTS}/qualification/bootstrap_qualification"
for entry in \
  qualification.json symbol-closure.json archive-member-audit.json \
  native_bootstrap_symbols.txt provider_archive_symbols.txt \
  adapter_archive_symbols.txt common_archive_symbols.txt linked_binary_symbols.txt \
  provider_archive_members.txt adapter_archive_members.txt common_archive_members.txt; do
  cp -- "${qualification}/${entry}" "${OUTPUT}/baseline-${entry}"
done

python3 - "${REPO_ROOT}" "${ACE_COMMIT}" \
  "${OUTPUT}/baseline-ace-source.manifest.json" \
  "${OUTPUT}/production-source-identity.json" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1])
current = sys.argv[2]
baseline = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["commit"]
result = subprocess.run(
    ["git", "-C", str(repo), "diff", "--name-only", baseline, current, "--"],
    check=True, capture_output=True, text=True)
changed = [line for line in result.stdout.splitlines() if line]
allowed = {
    "tools/phantom_gpu/harness/native_phantom_bts_correctness.cu",
    "tools/phantom_gpu/native_bts_correctness.py",
    "tools/phantom_gpu/native_bts_pipeline.sh",
    "tools/phantom_gpu/package_native_bts_qualification.sh",
    "tools/phantom_gpu/run_build_and_health.sh",
    "tools/phantom_gpu/runpod_transfer.sh",
}
if not changed or set(changed) - allowed:
    raise SystemExit(
        "M7P source revision changes production code or has an incomplete delta: "
        + ", ".join(changed))
record = {
    "schema_version": "ace.phantom.native-bts-production-source-identity/1.0.0",
    "status": "pass",
    "baseline_ace_commit": baseline,
    "qualification_ace_commit": current,
    "changed_paths": changed,
    "production_runtime_source_changed": False,
    "generated_dsl_source_changed": False,
}
Path(sys.argv[4]).write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

baseline_args=(
  --fixture "${OUTPUT}/baseline-fixture.json"
  --ace-source-manifest "${OUTPUT}/baseline-ace-source.manifest.json"
  --phantom-source-manifest "${OUTPUT}/baseline-phantom-source.manifest.json"
  --compiler-invocation "${OUTPUT}/baseline-compiler-invocation.json"
  --raw-air "${OUTPUT}/baseline-raw.air"
  --post-ckks-air "${OUTPUT}/baseline-post-ckks.air"
  --context-manifest "${OUTPUT}/baseline-context-manifest.json"
  --resource-manifest "${OUTPUT}/baseline-resource-manifest.json"
  --constant-manifest "${OUTPUT}/baseline-constant-manifest.json"
  --bootstrap-semantics "${OUTPUT}/baseline-bootstrap-semantics.json"
  --post-operations-air "${OUTPUT}/baseline-post-operations.air"
  --post-operation-attestation \
    "${OUTPUT}/baseline-post-operation-attestation.json"
  --baseline-gpu-record "${OUTPUT}/baseline-generated-phantom.json"
  --baseline-gpu-values "${OUTPUT}/baseline-generated-phantom.bin"
  --baseline-comparison "${OUTPUT}/baseline-comparison.json"
  --baseline-sanitizer "${OUTPUT}/baseline-sanitizer.json"
)
python3 -B "${OUTPUT}/native_bts_correctness.py" prepare \
  "${baseline_args[@]}" \
  --current-ace-source-manifest "${OUTPUT}/ace-source.manifest.json" \
  --current-phantom-source-manifest "${OUTPUT}/phantom-source.manifest.json" \
  --clear-record-output "${OUTPUT}/native-clear-inputs.json" \
  --clear-values-output "${OUTPUT}/native-clear-inputs.bin" \
  --output "${OUTPUT}/native-compatibility.json"

python3 - "${OUTPUT}" "${ACE_COMMIT}" "${PHANTOM_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.iterdir())
    if path.is_file() and path.name not in {"payload.json", "SHA256SUMS"}
}
payload = {
    "schema_version": "ace.phantom.native-bts-correctness-payload/1.0.0",
    "status": "pass",
    "ace_commit": sys.argv[2],
    "phantom_commit": sys.argv[3],
    "source_snapshots": {
        "ace_manifest_sha256": hashlib.sha256(
            (root / "ace-source.manifest.json").read_bytes()).hexdigest(),
        "phantom_manifest_sha256": hashlib.sha256(
            (root / "phantom-source.manifest.json").read_bytes()).hexdigest(),
    },
    "closed_m6": {
        "generated_record_sha256": files["baseline-generated-phantom.json"],
        "generated_values_sha256": files["baseline-generated-phantom.bin"],
        "comparison_sha256": files["baseline-comparison.json"],
        "sanitizer_sha256": files["baseline-sanitizer.json"],
    },
    "compatibility_sha256": files["native-compatibility.json"],
    "files": files,
}
(root / "payload.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
(
  cd "${OUTPUT}"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\0' |
    LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
  sha256sum -c SHA256SUMS >/dev/null
)
trap - EXIT
rm -rf -- "${temporary}"
echo "native BTS correctness payload: ${OUTPUT}"
