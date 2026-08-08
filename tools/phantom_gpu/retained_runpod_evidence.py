#!/usr/bin/env python3
"""Validate and bind retained CKKS source-only RunPod evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

import compare_retained_ckks_results as retained_comparator  # noqa: E402
from generate_retained_ckks_fixtures import RetainedFixtureError  # noqa: E402


HOST_SCHEMA = "ace.phantom.retained_ckks.host-qualification/1.0.0"
BUILD_SCHEMA = "ace.phantom.retained_ckks.build-attestation/1.0.0"
RUN_SCHEMA = "ace.phantom.retained_ckks.run-attestation/2.0.0"
PROVIDER_SCHEMA = "ace.phantom.retained_ckks.provider-attestation/2.0.0"


# Only frozen oracle/reference records and their identity receipts are
# exported.  In particular, no executable, object, archive, generated CUDA
# source, generated C++ source, or build log crosses this boundary.
FROZEN_FILES: dict[str, tuple[str, str]] = {
    "host_manifest": ("manifest.json", "retained-host-manifest.json"),
    "artifact_manifest": (
        "artifact_manifest.json",
        "retained-host-artifact-manifest.json",
    ),
    "build_attestation": (
        "build_attestation.json",
        "retained-host-build-attestation.json",
    ),
    "context_manifest": (
        "inputs/compiler_context_manifest.json",
        "retained-context-manifest.json",
    ),
    "generation_fixture": (
        "inputs/retained_ckks_fixture_template.json",
        "retained-generation-fixture.json",
    ),
    "fixture": (
        "inputs/retained_ckks_fixture.json",
        "retained-fixture.json",
    ),
    "emitted_context_manifest": (
        "outputs/compiler_context_manifest.json",
        "retained-emitted-context-manifest.json",
    ),
    "resource_manifest": (
        "outputs/compiler_resource_manifest.json",
        "retained-resource-manifest.json",
    ),
    "ant_post_ckks_air": (
        "outputs/retained_ckks_ant_post.air",
        "retained-ant-post-ckks.air",
    ),
    "phantom_post_ckks_air": (
        "outputs/retained_ckks_phantom_post.air",
        "retained-phantom-post-ckks.air",
    ),
    "production_post_ckks_air": (
        "outputs/retained_ckks_production_post.air",
        "retained-production-post-ckks.air",
    ),
    "production_rotation_batches": (
        "outputs/retained_ckks_production_rnums.json",
        "retained-production-rotation-batches.json",
    ),
    "generation_attestation": (
        "outputs/retained_ckks_generation.json",
        "retained-generation-attestation.json",
    ),
    "analytic_json": (
        "outputs/retained_ckks_analytic_reference.json",
        "retained-analytic-reference.json",
    ),
    "analytic_binary": (
        "outputs/retained_ckks_analytic_values.bin",
        "retained-analytic-values.bin",
    ),
    "exact_source_json": (
        "outputs/retained_ckks_exact_source.json",
        "retained-exact-source.json",
    ),
    "exact_source_binary": (
        "outputs/retained_ckks_exact_source.bin",
        "retained-exact-source.bin",
    ),
    "ant_reference_json": (
        "outputs/retained_ckks_cpu_reference.json",
        "retained-ant-reference.json",
    ),
    "ant_reference_binary": (
        "outputs/retained_ckks_cpu_values.bin",
        "retained-ant-values.bin",
    ),
}


REPLAY_EXACT_KEYS = tuple(
    key
    for key in FROZEN_FILES
    if key
    not in {
        "host_manifest",
        "artifact_manifest",
        "build_attestation",
        "ant_reference_json",
        "ant_reference_binary",
    }
)


class EvidenceError(ValueError):
    """A frozen or regenerated evidence relationship is invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvidenceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON constant: {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON record is not an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def require_digest(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        rf"[0-9a-f]{{{length}}}", value
    ) is None:
        raise EvidenceError(f"{label} is not a lowercase {length}-digit digest")
    return value


def read_complete_sums(root: Path) -> dict[str, str]:
    sums_path = root / "SHA256SUMS"
    listed: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvidenceError(f"cannot read retained SHA256SUMS: {error}") from error
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64}) ([ *])(.+)", line)
        if match is None:
            raise EvidenceError("retained SHA256SUMS contains a malformed entry")
        name = match.group(3).removeprefix("./")
        logical = PurePosixPath(name)
        if (
            not name
            or logical.is_absolute()
            or ".." in logical.parts
            or str(logical) != name
            or name in {"manifest.json", "SHA256SUMS"}
            or name in listed
        ):
            raise EvidenceError("retained SHA256SUMS contains an unsafe entry")
        listed[name] = match.group(1)
    actual = {
        path.relative_to(root).as_posix(): sha256_path(path)
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "SHA256SUMS"}
    }
    if listed != actual:
        raise EvidenceError(
            "retained SHA256SUMS is incomplete or an evidence file hash differs"
        )
    return listed


def validate_source_manifest(path: Path, kind: str, commit: str) -> str:
    value = load_json(path)
    if (
        value.get("schema_version") != "1.0.0"
        or value.get("kind") != kind
        or value.get("source_method") != "git-commit-object-archive"
        or value.get("commit") != commit
        or not isinstance(value.get("members"), list)
    ):
        raise EvidenceError(f"retained {kind} source manifest identity differs")
    return sha256_path(path)


def validate_root(root: Path, ace_commit: str, phantom_commit: str) -> dict[str, Any]:
    require_digest(ace_commit, 40, "ACE commit")
    require_digest(phantom_commit, 40, "Phantom commit")
    if not root.is_dir():
        raise EvidenceError(f"retained host evidence root is absent: {root}")
    listed = read_complete_sums(root)
    for source_name, _payload_name in FROZEN_FILES.values():
        path = root / source_name
        if not path.is_file() or path.stat().st_size == 0:
            raise EvidenceError(f"retained host evidence is absent: {source_name}")
        if source_name not in {"manifest.json"} and source_name not in listed:
            raise EvidenceError(f"retained host evidence is not checksummed: {source_name}")

    ace_source_path = root / "source/ace_source_manifest.json"
    phantom_source_path = root / "source/phantom_source_manifest.json"
    ace_source_sha = validate_source_manifest(ace_source_path, "ace", ace_commit)
    phantom_source_sha = validate_source_manifest(
        phantom_source_path, "phantom", phantom_commit
    )
    host = load_json(root / "manifest.json")
    expected_host = {
        "schema_version": HOST_SCHEMA,
        "status": "pass",
        "gate": "retained_ckks",
        "source_mode": "snapshot",
        "fixture_lifecycle": "checked-bound",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_worktree_dirty": False,
        "host_ant_oracle_was_run": True,
        "gpu_executables_were_run": False,
    }
    for field, expected in expected_host.items():
        if host.get(field) != expected:
            raise EvidenceError(f"retained host manifest {field} differs")
    if host.get("ace_source_manifest_sha256") != ace_source_sha:
        raise EvidenceError("retained host manifest ACE source binding differs")
    if host.get("phantom_source_manifest_sha256") != phantom_source_sha:
        raise EvidenceError("retained host manifest Phantom source binding differs")

    context_path = root / FROZEN_FILES["context_manifest"][0]
    fixture_path = root / FROZEN_FILES["fixture"][0]
    resource_path = root / FROZEN_FILES["resource_manifest"][0]
    generation_path = root / FROZEN_FILES["generation_attestation"][0]
    build_path = root / FROZEN_FILES["build_attestation"][0]
    artifact_path = root / FROZEN_FILES["artifact_manifest"][0]
    expected_host_hashes = {
        "compiler_context_manifest_sha256": sha256_path(context_path),
        "compiler_resource_manifest_sha256": sha256_path(resource_path),
        "fixture_sha256": sha256_path(fixture_path),
        "cpu_reference_sha256": sha256_path(
            root / FROZEN_FILES["ant_reference_json"][0]
        ),
        "cpu_values_sha256": sha256_path(
            root / FROZEN_FILES["ant_reference_binary"][0]
        ),
        "build_attestation_sha256": sha256_path(build_path),
        "artifact_manifest_sha256": sha256_path(artifact_path),
        "evidence_sha256_manifest_sha256": sha256_path(root / "SHA256SUMS"),
    }
    for field, expected in expected_host_hashes.items():
        if host.get(field) != expected:
            raise EvidenceError(f"retained host manifest {field} is stale")

    generation = load_json(generation_path)
    if (
        generation.get("ace_commit") != ace_commit
        or generation.get("fixture_sha256")
        != sha256_path(root / FROZEN_FILES["generation_fixture"][0])
        or generation.get("input_context_manifest_sha256")
        != sha256_path(context_path)
        or generation.get("emitted_context_manifest_sha256")
        != sha256_path(root / FROZEN_FILES["emitted_context_manifest"][0])
        or generation.get("resource_manifest_sha256") != sha256_path(resource_path)
        or generation.get("ant_post_ckks_air_sha256")
        != sha256_path(root / FROZEN_FILES["ant_post_ckks_air"][0])
        or generation.get("phantom_post_ckks_air_sha256")
        != sha256_path(root / FROZEN_FILES["phantom_post_ckks_air"][0])
    ):
        raise EvidenceError("retained generation attestation bindings differ")
    if (root / FROZEN_FILES["ant_post_ckks_air"][0]).read_bytes() != (
        root / FROZEN_FILES["phantom_post_ckks_air"][0]
    ).read_bytes():
        raise EvidenceError("retained ANT and Phantom post-CKKS AIR differ")
    if load_json(context_path) != load_json(
        root / FROZEN_FILES["emitted_context_manifest"][0]
    ):
        raise EvidenceError("retained emitted context differs from its input")
    try:
        (
            validated_generation,
            normalized_compiler_command_sha256,
            _compiler_options,
        ) = retained_comparator._load_invocation(generation_path)
        retained_comparator.verify_invocation_context(
            validated_generation,
            context_path,
            root / FROZEN_FILES["phantom_post_ckks_air"][0],
            root / FROZEN_FILES["generation_fixture"][0],
        )
    except (retained_comparator.ComparisonError, RetainedFixtureError) as error:
        raise EvidenceError(
            f"retained compiler invocation validation failed: {error}"
        ) from error

    build = load_json(build_path)
    expected_build = {
        "schema_version": BUILD_SCHEMA,
        "status": "pass",
        "architecture": "sm_80",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "source_mode": "snapshot",
        "ace_source_manifest_sha256": ace_source_sha,
        "phantom_source_manifest_sha256": phantom_source_sha,
        "compiler_context_manifest_sha256": sha256_path(context_path),
        "compiler_resource_manifest_sha256": sha256_path(resource_path),
        "fixture_sha256": sha256_path(fixture_path),
        "compiler_invocation_sha256": sha256_path(generation_path),
        "host_ant_oracle_was_run": True,
        "gpu_executables_were_run": False,
    }
    for field, expected in expected_build.items():
        if build.get(field) != expected:
            raise EvidenceError(f"retained build attestation {field} differs")
    for field in (
        "archive_inspection",
        "undefined_symbol_inspection",
        "cubin_architecture_inspection",
        "host_tests",
    ):
        if build.get(field) != "pass":
            raise EvidenceError(f"retained build attestation {field} did not pass")

    artifact = load_json(artifact_path)
    if (
        artifact.get("schema_version")
        != "ace.phantom.retained_ckks.artifact-manifest/1.0.0"
        or artifact.get("status") != "bound"
        or artifact.get("fixture_lifecycle") != "checked-bound"
        or artifact.get("ace_commit") != ace_commit
        or artifact.get("phantom_commit") != phantom_commit
        or artifact.get("ace_source_manifest_sha256") != ace_source_sha
        or artifact.get("phantom_source_manifest_sha256") != phantom_source_sha
        or artifact.get("compiler_context_manifest_sha256")
        != sha256_path(context_path)
        or artifact.get("compiler_resource_manifest_sha256")
        != sha256_path(resource_path)
        or artifact.get("normalized_compiler_command_sha256")
        != normalized_compiler_command_sha256
        or artifact.get("post_ckks_air_sha256")
        != sha256_path(root / FROZEN_FILES["phantom_post_ckks_air"][0])
        or artifact.get("production_post_ckks_air_sha256")
        != sha256_path(root / FROZEN_FILES["production_post_ckks_air"][0])
        or artifact.get("fixture_sha256") != sha256_path(fixture_path)
        or artifact.get("build_attestation_sha256") != sha256_path(build_path)
    ):
        raise EvidenceError("retained artifact manifest binding differs")
    artifact_files = artifact.get("files")
    if not isinstance(artifact_files, dict):
        raise EvidenceError("retained artifact manifest has no file map")
    expected_artifact_files = {
        name: digest
        for name, digest in listed.items()
        if name != "artifact_manifest.json"
    }
    if artifact_files != expected_artifact_files:
        raise EvidenceError(
            "retained artifact manifest is not an exhaustive pre-manifest file map"
        )
    for source_name, _payload_name in FROZEN_FILES.values():
        if source_name in {"manifest.json", "artifact_manifest.json"}:
            continue
        if artifact_files.get(source_name) != sha256_path(root / source_name):
            raise EvidenceError(
                f"retained artifact manifest does not bind {source_name}"
            )
    fixture = load_json(fixture_path)
    try:
        retained_comparator.verify_bindings(
            fixture,
            context_path,
            generation_path,
            root / FROZEN_FILES["phantom_post_ckks_air"][0],
            root / FROZEN_FILES["production_post_ckks_air"][0],
        )
    except (retained_comparator.ComparisonError, RetainedFixtureError) as error:
        raise EvidenceError(
            f"retained fixture/compiler binding validation failed: {error}"
        ) from error
    audit_paths = {
        "production_archive": root
        / "build/inspection/production-archive-audit.json",
        "generated_source": root
        / "build/inspection/generated-source-audit.json",
    }
    for label, path in audit_paths.items():
        relative = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or relative not in listed
            or artifact_files.get(relative) != sha256_path(path)
        ):
            raise EvidenceError(f"retained {label} audit is not artifact-bound")
    archive_audit = load_json(audit_paths["production_archive"])
    archive_inventories = archive_audit.get("inventories")
    if (
        set(archive_audit)
        != {
            "schema_version",
            "status",
            "forbidden_entry_count",
            "inventories",
        }
        or archive_audit.get("schema_version")
        != "ace.phantom.retained_ckks.production-archive-audit/1.0.0"
        or archive_audit.get("status") != "pass"
        or archive_audit.get("forbidden_entry_count") != 0
        or not isinstance(archive_inventories, list)
        or len(archive_inventories) != 6
        or any(
            not isinstance(item, dict) or item.get("forbidden_entries") != []
            for item in archive_inventories
        )
    ):
        raise EvidenceError("retained production archive audit did not pass")
    expected_archive_inventories = [
        ("member", f"{name}-archive-members.txt")
        for name in ("adapter", "provider", "common")
    ] + [
        ("defined_symbol", f"{name}-archive-symbols.txt")
        for name in ("adapter", "provider", "common")
    ]
    for item, (expected_kind, expected_name) in zip(
        archive_inventories, expected_archive_inventories, strict=True
    ):
        inventory_path = root / "build/inspection" / expected_name
        inventory_relative = inventory_path.relative_to(root).as_posix()
        if (
            not inventory_path.is_file()
            or artifact_files.get(inventory_relative)
            != sha256_path(inventory_path)
        ):
            raise EvidenceError(
                f"retained production archive inventory is not bound: {expected_name}"
            )
        lines = inventory_path.read_text(encoding="utf-8").splitlines()
        if (
            set(item)
            != {
                "kind",
                "path",
                "sha256",
                "entry_count",
                "forbidden_entries",
            }
            or item.get("kind") != expected_kind
            or item.get("path") != expected_name
            or item.get("sha256") != sha256_path(inventory_path)
            or item.get("entry_count") != len(lines)
            or not lines
            or any(not line for line in lines)
        ):
            raise EvidenceError(
                f"retained production archive inventory differs: {expected_name}"
            )
    generated_audit = load_json(audit_paths["generated_source"])
    required_calls = [
        "Conjugate_ciph",
        "Rotate_batch_ciph",
        "Raise_mod",
        "Mul_mono_ciph",
    ]
    expected_batches = [
        fixture["rotate_batch_steps"],
        *fixture["production_rotation_batches"],
        fixture["rotate_batch_steps"],
    ]
    call_counts = generated_audit.get("call_counts")
    argument_order = generated_audit.get("argument_order")
    if (
        set(generated_audit)
        != {
            "schema_version",
            "status",
            "source_sha256",
            "required_calls",
            "first_distinct_calls",
            "call_counts",
            "argument_order",
            "rotation_array_emission",
            "expected_rotation_batches",
            "observed_rotation_batches",
            "forbidden_matches",
        }
        or generated_audit.get("schema_version")
        != "ace.phantom.retained_ckks.generated-source-audit/1.0.0"
        or generated_audit.get("status") != "pass"
        or generated_audit.get("source_sha256")
        != sha256_path(root / "outputs/retained_ckks_phantom.cu")
        or generated_audit.get("required_calls") != required_calls
        or generated_audit.get("first_distinct_calls")
        != required_calls
        or not isinstance(call_counts, dict)
        or set(call_counts) != set(required_calls)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in call_counts.values()
        )
        or not isinstance(argument_order, dict)
        or argument_order != {call: True for call in required_calls}
        or generated_audit.get("expected_rotation_batches") != expected_batches
        or generated_audit.get("observed_rotation_batches") != expected_batches
        or generated_audit.get("forbidden_matches") != []
        or generated_audit.get("rotation_array_emission")
        != "ckks-owned-static-int32"
    ):
        raise EvidenceError("retained generated source audit did not pass")

    required_bound_paths = {
        "outputs/retained_ckks_keyless.cu",
        "outputs/retained_ckks_keyless_context.json",
        "outputs/retained_ckks_keyless_resources.json",
        "outputs/retained_ckks_keyless_post.air",
        "build/retained_ckks_conjugation_keyless_sm80",
        "build/retained_ckks_rotation_keyless_sm80",
        "build/retained_ckks_native_primitives_sm80",
        "build/gtest-source-attestation.json",
        "build/inspection/conjugation-keyless-undefined-symbols.txt",
        "build/inspection/conjugation-keyless-cuda-elf.txt",
        "build/inspection/conjugation-keyless-forbidden-undefined.txt",
        "build/inspection/rotation-keyless-undefined-symbols.txt",
        "build/inspection/rotation-keyless-cuda-elf.txt",
        "build/inspection/rotation-keyless-forbidden-undefined.txt",
        "build/inspection/native-primitives-undefined-symbols.txt",
        "build/inspection/native-primitives-defined-symbols.txt",
        "build/inspection/native-primitives-cuda-elf.txt",
        "build/inspection/native-primitives-forbidden-symbols.txt",
        "build/inspection/phantom-undefined-symbols.txt",
        "build/inspection/phantom-defined-symbols.txt",
        "build/inspection/phantom-cuda-elf.txt",
        "build/inspection/forbidden-retained-undefined-symbols.txt",
        "build/inspection/forbidden-phantom-binary-symbols.txt",
    }
    missing_bound = sorted(required_bound_paths - artifact_files.keys())
    if missing_bound:
        raise EvidenceError(
            "retained qualification artifacts are not bound: "
            + ", ".join(missing_bound)
        )
    for path in (
        "build/inspection/forbidden-retained-undefined-symbols.txt",
        "build/inspection/forbidden-phantom-binary-symbols.txt",
        "build/inspection/conjugation-keyless-forbidden-undefined.txt",
        "build/inspection/rotation-keyless-forbidden-undefined.txt",
        "build/inspection/native-primitives-forbidden-symbols.txt",
    ):
        if (root / path).read_bytes():
            raise EvidenceError(f"retained forbidden-symbol report is nonempty: {path}")
    for path in (
        "build/inspection/phantom-cuda-elf.txt",
        "build/inspection/conjugation-keyless-cuda-elf.txt",
        "build/inspection/rotation-keyless-cuda-elf.txt",
        "build/inspection/native-primitives-cuda-elf.txt",
    ):
        if "sm_80" not in (root / path).read_text(encoding="utf-8"):
            raise EvidenceError(f"retained CUDA inspection lacks sm_80: {path}")
    keyless = load_json(root / "outputs/retained_ckks_keyless_resources.json")
    expected_keyless = {
        "schema_version": 2,
        "context_schema_version": 1,
        "relinearization_key": False,
        "rotation_steps": [],
        "conjugation_key": False,
        "rotate_batch": False,
        "rotation_batches": [],
        "raise_mod": False,
        "monomial_powers": [],
    }
    if keyless != expected_keyless:
        raise EvidenceError("retained compiler-emitted keyless resources differ")
    gtest_attestation = load_json(root / "build/gtest-source-attestation.json")
    if (
        set(gtest_attestation)
        != {
            "schema_version",
            "status",
            "source_method",
            "commit",
            "tree",
            "tracked_inventory_sha256",
            "ace_gtest_archive_sha256",
            "ace_dependency_module_sha256",
            "fetchcontent_source_override",
            "fetchcontent_fully_disconnected",
        }
        or gtest_attestation.get("schema_version")
        != "ace.phantom.retained_ckks.gtest-source/1.0.0"
        or gtest_attestation.get("status") != "pass"
        or gtest_attestation.get("source_method")
        != "ace-pinned-external-project-source-reuse"
        or gtest_attestation.get("fetchcontent_source_override") is not True
        or gtest_attestation.get("fetchcontent_fully_disconnected") is not True
    ):
        raise EvidenceError("retained googletest source attestation differs")
    for field, length in (
        ("commit", 40),
        ("tree", 40),
        ("tracked_inventory_sha256", 64),
        ("ace_gtest_archive_sha256", 64),
        ("ace_dependency_module_sha256", 64),
    ):
        require_digest(
            gtest_attestation.get(field), length, f"retained googletest {field}"
        )

    fixture_bindings = fixture.get("qualification_bindings")
    if (
        not isinstance(fixture_bindings, dict)
        or fixture_bindings.get("status") != "bound"
    ):
        raise EvidenceError("retained checked fixture binding is not bound")
    if fixture_path.read_bytes() != (
        root / FROZEN_FILES["generation_fixture"][0]
    ).read_bytes():
        raise EvidenceError(
            "retained checked fixture differs from the generation fixture"
        )
    if fixture_bindings.get("compiler_context_manifest_sha256") != sha256_path(
        context_path
    ):
        raise EvidenceError("retained fixture context binding differs")
    return {
        "schema_version": "ace.phantom.retained_ckks.frozen-validation/1.0.0",
        "status": "pass",
        "ace_commit": ace_commit,
        "phantom_commit": phantom_commit,
        "ace_source_manifest_sha256": ace_source_sha,
        "phantom_source_manifest_sha256": phantom_source_sha,
        "host_manifest_sha256": sha256_path(root / "manifest.json"),
        "artifact_manifest_sha256": sha256_path(artifact_path),
        "build_attestation_sha256": sha256_path(build_path),
        "context_manifest_sha256": sha256_path(context_path),
        "resource_manifest_sha256": sha256_path(resource_path),
        "fixture_sha256": sha256_path(fixture_path),
        "generation_attestation_sha256": sha256_path(generation_path),
        "sha256_manifest_sha256": sha256_path(root / "SHA256SUMS"),
    }


def export_frozen(arguments: argparse.Namespace) -> dict[str, Any]:
    validation = validate_root(arguments.root, arguments.ace_commit, arguments.phantom_commit)
    if not arguments.output.is_dir():
        raise EvidenceError("frozen retained export destination must already exist")
    exported: dict[str, str] = {}
    for key, (source_name, payload_name) in FROZEN_FILES.items():
        destination = arguments.output / payload_name
        if destination.exists():
            raise EvidenceError(f"retained export destination exists: {destination}")
        shutil.copyfile(arguments.root / source_name, destination)
        exported[payload_name] = sha256_path(destination)
    record = {
        **validation,
        "schema_version": "ace.phantom.retained_ckks.frozen-export/1.0.0",
        "contents": "provider-neutral-references-and-attestations-no-build-output",
        "files": dict(sorted(exported.items())),
    }
    write_json(arguments.output / "retained-frozen-export.json", record)
    return record


def ant_semantic_summary(
    *,
    fixture_path: Path,
    context_path: Path,
    generation_path: Path,
    post_ckks_air_path: Path,
    production_post_ckks_air_path: Path,
    analytic_json_path: Path,
    analytic_binary_path: Path,
    ant_json_path: Path,
    ant_binary_path: Path,
    build_attestation_path: Path,
    label: str,
) -> dict[str, Any]:
    """Validate one randomized ANT run without treating its bytes as stable."""

    try:
        fixture = retained_comparator.load_json(fixture_path)
        resolved = retained_comparator.verify_bindings(
            fixture,
            context_path,
            generation_path,
            post_ckks_air_path,
            production_post_ckks_air_path,
        )
        fixture_sha256 = retained_comparator.sha256_path(fixture_path)
        context_sha256 = retained_comparator.sha256_path(context_path)
        inputs, analytic_records = retained_comparator.load_analytic(
            analytic_json_path,
            analytic_binary_path,
            fixture_sha256,
            fixture,
            resolved,
        )
        expected_order = retained_comparator.expected_provider_order(fixture)
        analytic_order = [record["case_id"] for record in analytic_records]
        if analytic_order != expected_order[: len(analytic_order)]:
            raise EvidenceError(f"{label} analytic case prefix differs")
        build = load_json(build_attestation_path)
        executable_hashes = build.get("executables")
        if not isinstance(executable_hashes, dict):
            raise EvidenceError(f"{label} build executable hashes are absent")
        identifiers = {
            "ace_commit": build.get("ace_commit"),
            "phantom_commit": build.get("phantom_commit"),
            "executable_sha256": executable_hashes.get("ant_oracle"),
        }
        records, first_data_chain_index = retained_comparator.load_provider(
            ant_json_path,
            ant_binary_path,
            "ant",
            fixture_sha256,
            context_sha256,
            fixture["qualification_bindings"],
            expected_order,
            resolved,
            identifiers,
        )
    except (retained_comparator.ComparisonError, RetainedFixtureError) as error:
        raise EvidenceError(f"{label} ANT contract validation failed: {error}") from error

    by_id = {record["case_id"]: record for record in records}
    analytic_by_id = {record["case_id"]: record for record in analytic_records}
    tolerances = fixture["tolerances"]

    def metric_status(actual: list[complex], expected: list[complex], composite: bool) -> str:
        result = retained_comparator.metric(
            actual,
            expected,
            absolute_tolerance=tolerances[
                "composite_absolute" if composite else "primitive_absolute"
            ],
            relative_tolerance=tolerances[
                "composite_relative" if composite else "primitive_relative"
            ],
            hard_maximum=tolerances["hard_maximum_absolute"],
            relative_floor=tolerances["relative_metric_floor"],
        )
        return result["status"]

    analytic_status = {
        case_id: metric_status(
            by_id[case_id]["decoded_values"],
            analytic_by_id[case_id]["decoded_values"],
            case_id.startswith("composite."),
        )
        for case_id in analytic_order
    }
    if set(analytic_status.values()) != {"pass"}:
        raise EvidenceError(f"{label} ANT differs from the analytic oracle")

    source = inputs["bounded_nonperiodic"]
    zero_id = "rotate_batch.bounded_nonperiodic.output_1.step_0"
    duplicate_a = "rotate_batch.bounded_nonperiodic.output_0.step_5"
    duplicate_b = "rotate_batch.bounded_nonperiodic.output_3.step_5"
    metamorphic = {
        "conjugate_twice_identity": metric_status(
            by_id["conjugate_twice.bounded_nonperiodic"]["decoded_values"],
            source,
            False,
        )
        == "pass",
        "raise_mod_identity": metric_status(
            by_id["raise_mod.bounded_nonperiodic"]["decoded_values"],
            source,
            False,
        )
        == "pass",
        "zero_step_identity": metric_status(
            by_id[zero_id]["decoded_values"], source, False
        )
        == "pass",
        "duplicate_steps_equal": (
            by_id[duplicate_a]["decoded_values"]
            == by_id[duplicate_b]["decoded_values"]
        ),
        "monomial_zero_identity": metric_status(
            by_id["mul_mono.0.bounded_nonperiodic"]["decoded_values"],
            source,
            False,
        )
        == "pass",
        "monomial_degree_negation": metric_status(
            by_id["mul_mono.N.bounded_nonperiodic"]["decoded_values"],
            [-value for value in source],
            False,
        )
        == "pass",
        "monomial_inverse_composition": metric_status(
            by_id["mul_mono.inverse_composition.bounded_nonperiodic"][
                "decoded_values"
            ],
            source,
            False,
        )
        == "pass",
    }
    if not all(metamorphic.values()):
        raise EvidenceError(f"{label} ANT metamorphic checks failed")

    # Random encryption/key generation changes ciphertext-derived hashes,
    # ownership tokens, decoded bytes, and small numerical errors.  The replay
    # identity is therefore this closed semantic summary, never those bytes.
    return {
        "schema_version": "ace.phantom.retained_ckks.ant-semantic-summary/1.0.0",
        "status": "pass",
        "fixture_sha256": fixture_sha256,
        "context_manifest_sha256": context_sha256,
        "first_data_chain_index": first_data_chain_index,
        "record_order": [record["case_id"] for record in records],
        "operations": [record["operation"] for record in records],
        "provider_independent_metadata": [
            retained_comparator.provider_independent_metadata(record["metadata"])
            for record in records
        ],
        "decoded_projections": [record["decoded_projection"] for record in records],
        "analytic_status": analytic_status,
        "metamorphic_checks": metamorphic,
    }


def verify_replay(arguments: argparse.Namespace) -> dict[str, Any]:
    validation = validate_root(
        arguments.regenerated_root, arguments.ace_commit, arguments.phantom_commit
    )
    export_record = load_json(arguments.frozen / "retained-frozen-export.json")
    if (
        export_record.get("status") != "pass"
        or export_record.get("ace_commit") != arguments.ace_commit
        or export_record.get("phantom_commit") != arguments.phantom_commit
    ):
        raise EvidenceError("frozen retained export identity differs")
    exported_files = export_record.get("files")
    expected_export_names = {payload_name for _, payload_name in FROZEN_FILES.values()}
    if not isinstance(exported_files, dict) or set(exported_files) != expected_export_names:
        raise EvidenceError("frozen retained export file inventory differs")
    for payload_name, expected_sha256 in exported_files.items():
        require_digest(expected_sha256, 64, f"frozen retained {payload_name}")
        if sha256_path(arguments.frozen / payload_name) != expected_sha256:
            raise EvidenceError(f"frozen retained export hash differs: {payload_name}")
    compared: dict[str, str] = {}
    for key in REPLAY_EXACT_KEYS:
        source_name, payload_name = FROZEN_FILES[key]
        frozen = arguments.frozen / payload_name
        regenerated = arguments.regenerated_root / source_name
        if not frozen.is_file() or frozen.read_bytes() != regenerated.read_bytes():
            raise EvidenceError(f"regenerated retained artifact differs: {key}")
        compared[key] = sha256_path(frozen)
    frozen_ant_summary = ant_semantic_summary(
        fixture_path=arguments.frozen / FROZEN_FILES["fixture"][1],
        context_path=arguments.frozen / FROZEN_FILES["context_manifest"][1],
        generation_path=arguments.frozen
        / FROZEN_FILES["generation_attestation"][1],
        post_ckks_air_path=arguments.frozen
        / FROZEN_FILES["phantom_post_ckks_air"][1],
        production_post_ckks_air_path=arguments.frozen
        / FROZEN_FILES["production_post_ckks_air"][1],
        analytic_json_path=arguments.frozen / FROZEN_FILES["analytic_json"][1],
        analytic_binary_path=arguments.frozen
        / FROZEN_FILES["analytic_binary"][1],
        ant_json_path=arguments.frozen / FROZEN_FILES["ant_reference_json"][1],
        ant_binary_path=arguments.frozen
        / FROZEN_FILES["ant_reference_binary"][1],
        build_attestation_path=arguments.frozen
        / FROZEN_FILES["build_attestation"][1],
        label="frozen",
    )
    regenerated_ant_summary = ant_semantic_summary(
        fixture_path=arguments.regenerated_root / FROZEN_FILES["fixture"][0],
        context_path=arguments.regenerated_root
        / FROZEN_FILES["context_manifest"][0],
        generation_path=arguments.regenerated_root
        / FROZEN_FILES["generation_attestation"][0],
        post_ckks_air_path=arguments.regenerated_root
        / FROZEN_FILES["phantom_post_ckks_air"][0],
        production_post_ckks_air_path=arguments.regenerated_root
        / FROZEN_FILES["production_post_ckks_air"][0],
        analytic_json_path=arguments.regenerated_root
        / FROZEN_FILES["analytic_json"][0],
        analytic_binary_path=arguments.regenerated_root
        / FROZEN_FILES["analytic_binary"][0],
        ant_json_path=arguments.regenerated_root
        / FROZEN_FILES["ant_reference_json"][0],
        ant_binary_path=arguments.regenerated_root
        / FROZEN_FILES["ant_reference_binary"][0],
        build_attestation_path=arguments.regenerated_root
        / FROZEN_FILES["build_attestation"][0],
        label="regenerated",
    )
    if frozen_ant_summary != regenerated_ant_summary:
        raise EvidenceError("regenerated retained ANT semantic summary differs")
    return {
        **validation,
        "schema_version": "ace.phantom.retained_ckks.local-replay/1.0.0",
        "status": "pass",
        "exact_artifact_count": len(REPLAY_EXACT_KEYS),
        "provider_neutral_ant_reference_matches": True,
        "ant_replay": {
            "status": "pass",
            "comparison": "semantic-summary-only-no-decoded-byte-comparison",
            "summary": frozen_ant_summary,
            "summary_sha256": sha256_bytes(
                json.dumps(
                    frozen_ant_summary,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        },
        "artifacts": dict(sorted(compared.items())),
    }


def write_run_attestations(arguments: argparse.Namespace) -> dict[str, Any]:
    root = arguments.retained_root
    frozen = arguments.frozen
    results = arguments.result_dir
    if arguments.expected_gpu not in {
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
    }:
        raise EvidenceError("retained run requires one exact supported A100")
    paths = {
        "ace_source": root / "source/ace_source_manifest.json",
        "phantom_source": root / "source/phantom_source_manifest.json",
        "generation": root / "outputs/retained_ckks_generation.json",
        "frozen_build": frozen / FROZEN_FILES["build_attestation"][1],
        "remote_build": root / "build_attestation.json",
        "fixture": root / "inputs/retained_ckks_fixture.json",
        "context": root / "inputs/compiler_context_manifest.json",
        "resource": root / "outputs/compiler_resource_manifest.json",
        "phantom_executable": root / "build/retained_ckks_phantom_sm80",
        "ant_json": frozen / FROZEN_FILES["ant_reference_json"][1],
        "ant_binary": frozen / FROZEN_FILES["ant_reference_binary"][1],
        "phantom_json": results / "retained_ckks_gpu_results.json",
        "phantom_binary": results / "retained_ckks_gpu_values.bin",
        "exact_json": results / "retained_ckks_exact_observed.json",
        "exact_binary": results / "retained_ckks_exact_observed.bin",
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise EvidenceError(f"retained run artifact is absent: {label}")
    ace_source = load_json(paths["ace_source"])
    phantom_source = load_json(paths["phantom_source"])
    frozen_build = load_json(paths["frozen_build"])
    remote_build = load_json(paths["remote_build"])
    frozen_export = load_json(frozen / "retained-frozen-export.json")
    frozen_files = frozen_export.get("files")
    if not isinstance(frozen_files, dict):
        raise EvidenceError("frozen retained export file inventory is absent")
    for key in ("build_attestation", "ant_reference_json", "ant_reference_binary"):
        payload_name = FROZEN_FILES[key][1]
        if frozen_files.get(payload_name) != sha256_path(frozen / payload_name):
            raise EvidenceError(f"frozen retained export binding differs: {payload_name}")

    expected_identity = {
        "ace_commit": ace_source.get("commit"),
        "phantom_commit": phantom_source.get("commit"),
        "ace_source_manifest_sha256": sha256_path(paths["ace_source"]),
        "phantom_source_manifest_sha256": sha256_path(paths["phantom_source"]),
        "compiler_context_manifest_sha256": sha256_path(paths["context"]),
        "compiler_resource_manifest_sha256": sha256_path(paths["resource"]),
        "fixture_sha256": sha256_path(paths["fixture"]),
        "compiler_invocation_sha256": sha256_path(paths["generation"]),
    }
    for label, build in (("frozen", frozen_build), ("remote", remote_build)):
        if build.get("schema_version") != BUILD_SCHEMA or build.get("status") != "pass":
            raise EvidenceError(f"{label} retained build attestation did not pass")
        for field, expected in expected_identity.items():
            if build.get(field) != expected:
                raise EvidenceError(
                    f"{label} retained build attestation {field} differs"
                )
        if not isinstance(build.get("executables"), dict):
            raise EvidenceError(f"{label} retained build executables are absent")

    ant_result = load_json(paths["ant_json"])
    phantom_result = load_json(paths["phantom_json"])
    ant_identifiers = ant_result.get("identifiers")
    phantom_identifiers = phantom_result.get("identifiers")
    if not isinstance(ant_identifiers, dict) or not isinstance(
        phantom_identifiers, dict
    ):
        raise EvidenceError("retained provider result identifiers are absent")
    expected_identifiers = {
        "ant": {
            "ace_commit": ace_source.get("commit"),
            "phantom_commit": phantom_source.get("commit"),
            "executable_sha256": frozen_build["executables"].get("ant_oracle"),
        },
        "phantom": {
            "ace_commit": ace_source.get("commit"),
            "phantom_commit": phantom_source.get("commit"),
            "executable_sha256": sha256_path(paths["phantom_executable"]),
        },
    }
    if ant_identifiers != expected_identifiers["ant"]:
        raise EvidenceError(
            "frozen retained ANT result identity differs from its attested producer"
        )
    if remote_build["executables"].get("phantom_sm80") != sha256_path(
        paths["phantom_executable"]
    ):
        raise EvidenceError("remote retained Phantom executable identity differs")
    if phantom_identifiers != expected_identifiers["phantom"]:
        raise EvidenceError(
            "retained Phantom result identity differs from its executable"
        )
    run_path = results / "retained_ckks_run_attestation.json"
    provider_path = results / "retained_ckks_provider_attestation.json"
    if run_path.exists() or provider_path.exists():
        raise EvidenceError("retained run attestation output already exists")
    run = {
        "schema_version": RUN_SCHEMA,
        "status": "pass",
        "ace_commit": ace_source["commit"],
        "phantom_commit": phantom_source["commit"],
        "ace_source_manifest_sha256": sha256_path(paths["ace_source"]),
        "phantom_source_manifest_sha256": sha256_path(paths["phantom_source"]),
        "generation_attestation_sha256": sha256_path(paths["generation"]),
        "frozen_build_attestation_sha256": sha256_path(paths["frozen_build"]),
        "remote_build_attestation_sha256": sha256_path(paths["remote_build"]),
        "fixture_sha256": sha256_path(paths["fixture"]),
        "compiler_context_manifest_sha256": sha256_path(paths["context"]),
        "compiler_resource_manifest_sha256": sha256_path(paths["resource"]),
        "executables": {
            "ant_oracle": expected_identifiers["ant"]["executable_sha256"],
            "phantom_sm80": sha256_path(paths["phantom_executable"]),
        },
        "provider_results": {
            "ant": {
                "json_sha256": sha256_path(paths["ant_json"]),
                "binary_sha256": sha256_path(paths["ant_binary"]),
            },
            "phantom": {
                "json_sha256": sha256_path(paths["phantom_json"]),
                "binary_sha256": sha256_path(paths["phantom_binary"]),
            },
        },
        "exact_observed": {
            "json_sha256": sha256_path(paths["exact_json"]),
            "binary_sha256": sha256_path(paths["exact_binary"]),
        },
        "gpu": {"device_count": 1, "device_name": arguments.expected_gpu},
    }
    write_json(run_path, run)
    provider = {
        "schema_version": PROVIDER_SCHEMA,
        "fixture_sha256": run["fixture_sha256"],
        "context_manifest_sha256": run["compiler_context_manifest_sha256"],
        "compiler_invocation_sha256": run["generation_attestation_sha256"],
        "frozen_build_attestation_sha256": run[
            "frozen_build_attestation_sha256"
        ],
        "remote_build_attestation_sha256": run[
            "remote_build_attestation_sha256"
        ],
        "run_attestation_sha256": sha256_path(run_path),
        "providers": [
            {
                "provider": "ant",
                "identifiers": expected_identifiers["ant"],
                "result_json_sha256": sha256_path(paths["ant_json"]),
                "result_binary_sha256": sha256_path(paths["ant_binary"]),
            },
            {
                "provider": "phantom",
                "identifiers": expected_identifiers["phantom"],
                "result_json_sha256": sha256_path(paths["phantom_json"]),
                "result_binary_sha256": sha256_path(paths["phantom_binary"]),
            },
        ],
        "exact_observed": {
            "provider": "phantom",
            "executable_sha256": expected_identifiers["phantom"][
                "executable_sha256"
            ],
            "result_json_sha256": sha256_path(paths["exact_json"]),
            "result_binary_sha256": sha256_path(paths["exact_binary"]),
        },
    }
    write_json(provider_path, provider)
    return {
        "schema_version": "ace.phantom.retained_ckks.run-evidence/1.0.0",
        "status": "pass",
        "run_attestation_sha256": sha256_path(run_path),
        "provider_attestation_sha256": sha256_path(provider_path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-frozen")
    validate.add_argument("--root", required=True, type=Path)
    validate.add_argument("--ace-commit", required=True)
    validate.add_argument("--phantom-commit", required=True)
    export = commands.add_parser("export-frozen")
    export.add_argument("--root", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--ace-commit", required=True)
    export.add_argument("--phantom-commit", required=True)
    replay = commands.add_parser("verify-replay")
    replay.add_argument("--frozen", required=True, type=Path)
    replay.add_argument("--regenerated-root", required=True, type=Path)
    replay.add_argument("--ace-commit", required=True)
    replay.add_argument("--phantom-commit", required=True)
    attest = commands.add_parser("write-run-attestations")
    attest.add_argument("--frozen", required=True, type=Path)
    attest.add_argument("--retained-root", required=True, type=Path)
    attest.add_argument("--result-dir", required=True, type=Path)
    attest.add_argument("--expected-gpu", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.command == "validate-frozen":
            result = validate_root(
                arguments.root, arguments.ace_commit, arguments.phantom_commit
            )
        elif arguments.command == "export-frozen":
            result = export_frozen(arguments)
        elif arguments.command == "verify-replay":
            result = verify_replay(arguments)
        else:
            result = write_run_attestations(arguments)
    except EvidenceError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
