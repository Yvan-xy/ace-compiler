"""Static contracts for the source-only retained CKKS A100 pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re


TOOLS = Path(__file__).resolve().parents[1]
REPOSITORY = TOOLS.parents[1]


def _evidence_module():
    path = TOOLS / "retained_runpod_evidence.py"
    spec = importlib.util.spec_from_file_location("retained_runpod_evidence", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_export_contains_no_build_output() -> None:
    module = _evidence_module()
    payload_names = {
        payload_name for _source_name, payload_name in module.FROZEN_FILES.values()
    }
    assert payload_names
    assert not any(
        name.endswith((".cu", ".cxx", ".cc", ".o", ".a", ".so"))
        for name in payload_names
    )
    assert not any("executable" in name or "binary" in name for name in payload_names)
    assert {
        "retained-context-manifest.json",
        "retained-resource-manifest.json",
        "retained-fixture.json",
        "retained-analytic-reference.json",
        "retained-ant-reference.json",
        "retained-exact-source.json",
    }.issubset(payload_names)


def test_combined_pipeline_runs_ordinary_gpu_before_retained_build() -> None:
    source = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    sequence = source[source.index("phase payload_verification") :]
    ordinary = sequence.index(
        "phase ordinary_gpu_qualification run_ordinary_gpu_qualification"
    )
    retained_host = sequence.index(
        "phase retained_host_qualification run_retained_host_qualification"
    )
    retained_gpu = sequence.index(
        "phase retained_gpu_qualification run_retained_gpu_qualification"
    )
    assert ordinary < retained_host < retained_gpu
    assert "write-run-attestations" in source
    assert source.index("write-run-attestations") < source.index(
        "compare_retained_ckks_results.py"
    )


def test_retained_pipeline_replays_frozen_provider_neutral_evidence() -> None:
    package = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    local = (TOOLS / "run_local_reproduction.sh").read_text(encoding="utf-8")
    runner = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "--retained-run-root" in package
    assert "export-frozen" in package
    assert "without-build-output" in package
    assert "--retained-run-root" in local
    assert "verify-replay" in runner
    assert "provider_neutral_ant_reference_matches" in runner
    assert "ACE_RETAINED_CKKS_PROVISIONAL_BINDING=0" in runner
    assert 'local ant_json="${INPUT}/retained-ant-reference.json"' in runner
    assert 'local ant_bin="${INPUT}/retained-ant-values.bin"' in runner
    assert "--frozen-build-attestation" in runner
    assert "--remote-build-attestation" in runner
    assert "--ant-executable" not in runner
    assert "semantic-summary-only-no-decoded-byte-comparison" in runner


def test_packaging_checks_every_retained_evidence_dependency() -> None:
    package = (TOOLS / "package_runpod_sources.sh").read_text(encoding="utf-8")
    for dependency in (
        "package_runpod_sources.sh",
        "retained_runpod_evidence.py",
        "compare_retained_ckks_results.py",
        "generate_retained_ckks_fixtures.py",
    ):
        assert dependency in package
    assert "retained evidence dependency differs from the selected ACE commit" in package


def test_keyless_rejections_use_compiler_emitted_translation_unit() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    harness = (
        TOOLS / "harness/retained_ckks_phantom_conformance.cu"
    ).read_text(encoding="utf-8")
    fixture = json.loads(
        (TOOLS / "fixtures/retained_ckks_v1.json").read_text(encoding="utf-8")
    )
    assert "generate_ckks2c_probe.py" in host
    assert "--resource-mode keyless" in host
    assert "retained_ckks_keyless_resources.json" in host
    assert "ACE_REJECTION_ONLY" in harness
    rejections = {item["id"]: item for item in fixture["runtime_rejections"]}
    assert rejections["conjugate_missing_key"]["manifest"] == "keyless-conjugation"
    assert rejections["rotate_batch_missing_nonzero_key"] == {
        "id": "rotate_batch_missing_nonzero_key",
        "diagnostic": "ROTATE_BATCH_RESOURCE",
        "manifest": "keyless-rotation",
    }


def test_formal_host_gate_requires_checked_bound_fixture() -> None:
    source = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    assert "checked-bound" in source
    assert "provisional-bind" in source
    assert "ACE_RETAINED_CKKS_PROVISIONAL_BINDING" in source
    assert "checked retained fixture must be bound" in source
    evidence = (TOOLS / "retained_runpod_evidence.py").read_text(
        encoding="utf-8"
    )
    assert 'artifact.get("fixture_lifecycle") != "checked-bound"' in evidence
    assert 'get("status") != "bound"' in evidence
    assert "checked fixture differs from the generation fixture" in evidence


def test_formal_host_gate_audits_production_closure_and_native_test() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    runner = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    assert "production-archive-audit.json" in host
    assert "generated-source-audit.json" in host
    assert "adapter-archive-members.txt" in host
    assert "adapter-archive-symbols.txt" in host
    assert "native-primitives-forbidden-symbols.txt" in host
    assert "ckks-owned-static-int32" in host
    assert "first_distinct_calls" in host
    assert "PHANTOM_BUILD_TESTS=ON" in host
    assert "ckks_retained_primitives" in host
    assert "retained_ckks_native_primitives_sm80" in host
    sequence = runner[runner.index("phase payload_verification") :]
    assert sequence.index(
        "phase ordinary_gpu_qualification run_ordinary_gpu_qualification"
    ) < sequence.index(
        "phase retained_gpu_qualification run_retained_gpu_qualification"
    )
    retained_function = runner[
        runner.index("run_retained_gpu_qualification() {") :
        runner.index("\nrun_native_health() {")
    ]
    assert retained_function.index(
        'timeout 900 "${native_test}" --context-manifest "${emitted_context}"'
    ) < retained_function.index('timeout 900 "${runner}" conformance')


def test_host_evidence_checksum_excludes_only_root_receipts(tmp_path: Path) -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    assert (
        "find . -type f ! -path ./manifest.json ! -path ./SHA256SUMS -print0"
        in host
    )
    assert "! -name manifest.json" not in host
    assert "! -name SHA256SUMS" not in host

    nested = tmp_path / "inputs/manifest.json"
    nested.parent.mkdir()
    nested.write_text("nested evidence\n", encoding="utf-8")
    payload = tmp_path / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("root receipt\n", encoding="utf-8")
    lines = []
    for path in (nested, payload):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(tmp_path).as_posix()}\n")
    (tmp_path / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    assert _evidence_module().read_complete_sums(tmp_path) == {
        "inputs/manifest.json": hashlib.sha256(nested.read_bytes()).hexdigest(),
        "payload.txt": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }


def test_retained_harness_consumes_fixture_owned_batch_contract() -> None:
    harness = (
        TOOLS / "harness/retained_ckks_phantom_conformance.cu"
    ).read_text(encoding="utf-8")
    assert "std::vector<std::int32_t>({5, 0, -7, 5})" not in harness
    assert "iterations == 100" not in harness
    assert 'fixture.at("rotate_batch_steps")' in harness
    assert 'fixture.at("ownership").at("iterations")' in harness
    assert 'fixture.at("ownership").at("free_order")' in harness


def test_native_test_reuses_audited_ace_googletest_source() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    assert "FETCHCONTENT_SOURCE_DIR_GOOGLETEST" in host
    assert "FETCHCONTENT_FULLY_DISCONNECTED=ON" in host
    assert "ace-pinned-external-project-source-reuse" in host
    assert "gtest-source-attestation.json" in host
    evidence = (TOOLS / "retained_runpod_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "required_bound_paths" in evidence
    assert "exhaustive pre-manifest file map" in evidence


def test_ant_oracle_uses_audited_installed_dependency_headers() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    assert 'ant_install_include="${INSTALL_ROOT}/rtlib/include/ant"' in host
    assert '[[ -r "${ant_install_include}/uthash.h" ]]' in host
    assert '"-I${ant_install_include}"' in host


def test_ant_runtime_header_declares_generated_callbacks_with_c_abi() -> None:
    ant_header = (
        REPOSITORY / "fhe-cmplr/rtlib/include/rt_ant/rt_ant.h"
    ).read_text(encoding="utf-8")
    common_api = (
        REPOSITORY / "fhe-cmplr/rtlib/include/common/rt_api.h"
    ).read_text(encoding="utf-8")
    assert '#include "common/rt_api.h"' in ant_header
    assert 'extern "C" {' in common_api
    assert "CKKS_PARAMS* Get_context_params();" in common_api
    assert "RT_DATA_INFO* Get_rt_data_info();" in common_api


def test_retained_harnesses_share_the_sha256_constant_table() -> None:
    def constants(relative: str) -> list[str]:
        source = (TOOLS / relative).read_text(encoding="utf-8")
        table = source[
            source.index(
                "static constexpr std::array<std::uint32_t, 64> constants"
            ) : source.index("std::array<std::uint32_t, 8> state")
        ]
        return re.findall(r"0x[0-9a-f]+U", table)

    ant = constants("harness/retained_ckks_ant_oracle.cxx")
    phantom = constants("harness/retained_ckks_phantom_conformance.cu")
    assert len(ant) == 64
    assert ant == phantom
    assert "0x4ed8aa4aU" in ant


def test_ant_oracle_provisions_only_its_internal_conjugation_sentinel() -> None:
    harness = (
        TOOLS / "harness/retained_ckks_ant_oracle.cxx"
    ).read_text(encoding="utf-8")
    provision = harness[
        harness.index("void ProvisionAntConjugationKey(") : harness.index(
            "\nvoid VerifyPrimeChain("
        )
    ]
    assert 'resources.at("conjugation_key").get<bool>()' in provision
    assert "2U * degree - 1U" in provision
    assert "Insert_rot_map(key_generator, ant_conjugation_index)" in provision
    assert harness.index("Prepare_context();") < harness.index(
        "ProvisionAntConjugationKey(context, resources);"
    )
    assert "ant_conjugation_index" not in harness[
        harness.index("void ValidateManifest(") : harness.index(
            "void ProvisionAntConjugationKey("
        )
    ]


def test_ant_rotation_runtime_normalizes_steps_to_compiler_key_ids() -> None:
    runtime = (
        REPOSITORY / "fhe-cmplr/rtlib/ant/ckks/src/cipher.c"
    ).read_text(encoding="utf-8")
    batch = runtime[
        runtime.index("static int32_t Normalize_rotation(") : runtime.index(
            "\nCIPHER Conjugate_ciph("
        )
    ]
    assert "Get_ciph_slots(ciph)" in batch
    assert "normalized > (int64_t)(slots / 2U)" in batch
    assert (
        "const int32_t normalized = Normalize_rotation(ciph, rot_idx);" in batch
    )
    assert "if (normalized == 0)" in batch
    assert "if (res != ciph) Copy_ciphertext(res, ciph);" in batch
    assert "Normalize_rotation(ciph, rot_idx[idx])" in batch
    assert batch.index("Normalize_rotation(ciph, rot_idx[idx])") < batch.index(
        "if (rot == 0)"
    )


def test_executable_symbol_audits_do_not_match_the_binary_path() -> None:
    host = (TOOLS / "run_retained_ckks_correctness.sh").read_text(
        encoding="utf-8"
    )
    inspection = host[
        host.index('nm -C --undefined-only "${PHANTOM_BINARY}"') : host.index(
            "\nwrite_attestations() {"
        )
    ]
    assert "nm -A -C" not in inspection
    for command in (
        'nm -C --undefined-only "${PHANTOM_BINARY}"',
        'nm -C --defined-only "${PHANTOM_BINARY}"',
        'nm -C --undefined-only "${ANT_BINARY}"',
        'nm -C --undefined-only "${PHANTOM_NATIVE_TEST_BINARY}"',
        'nm -C --defined-only "${PHANTOM_NATIVE_TEST_BINARY}"',
        'nm -C --undefined-only "${keyless_binary}"',
    ):
        assert command in inspection
    assert (
        "Conjugate_ciph|Rotate_batch_ciph|Raise_mod|Mul_mono_ciph|"
        "retained_ckks_"
    ) in inspection
    archive_loop = host[
        host.index("inspect_build() {") : host.index(
            'nm -C --undefined-only "${PHANTOM_BINARY}"'
        )
    ]
    # Archive inputs still need member provenance in their symbol receipts.
    assert 'nm -A -C --defined-only "${archive}"' in archive_loop


def test_a100_gate_executes_fixture_owned_adapter_aliases() -> None:
    runner = (TOOLS / "run_build_and_health.sh").read_text(encoding="utf-8")
    retained_function = runner[
        runner.index("run_retained_gpu_qualification() {") :
        runner.index("\nrun_native_health() {")
    ]
    assert '"${runner}" aliases "${fixture}" "${context}"' in retained_function
    assert 'fixture[0].monomial_powers' in retained_function
    assert '"conjugate.in_place"' in retained_function
    assert "metadata_matches_out_of_place" in retained_function
    assert "decoded_values_match_out_of_place" in retained_function
    assert "residues_match_out_of_place" in retained_function
    assert "retained_ckks_adapter_aliases.json" in runner
