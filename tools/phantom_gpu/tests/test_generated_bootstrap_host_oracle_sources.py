from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_HEADER = ROOT / "fhe-cmplr/rtlib/ant/include/ckks/bootstrap.h"
RUNTIME_SOURCE = ROOT / "fhe-cmplr/rtlib/ant/ckks/src/bootstrap.c"
COMMON = ROOT / "tools/phantom_gpu/harness/generated_bootstrap_ant_common.h"
NATIVE = (
    ROOT
    / "tools/phantom_gpu/harness/generated_bootstrap_native_ant_oracle.cxx"
)
GENERATED = (
    ROOT / "tools/phantom_gpu/harness/generated_bootstrap_dsl_ant_oracle.cxx"
)
COMPILE_ONLY = ROOT / "tools/phantom_gpu/compile_only.sh"
HOST_FREEZE = ROOT / "tools/phantom_gpu/freeze_bootstrap_host_evidence.sh"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_runtime_attestation_distinguishes_the_identity_copy_path() -> None:
    header = source(RUNTIME_HEADER)
    implementation = source(RUNTIME_SOURCE)
    for field in (
        "invocation_count",
        "early_copy_return_count",
        "full_execution_count",
        "coeffs_to_slots_entry_count",
        "coeffs_to_slots_completion_count",
        "eval_mod_entry_count",
        "eval_mod_completion_count",
        "slots_to_coeffs_entry_count",
        "slots_to_coeffs_completion_count",
        "full_completion_count",
    ):
        assert field in header
        assert field in implementation
    assert "Reset_bootstrap_execution_attestation" in header
    assert "Get_bootstrap_execution_attestation" in header
    early_gate = implementation.index("// just return the original ciphertext")
    early_count = implementation.index("early_copy_return_count", early_gate)
    early_return = implementation.index("return res;", early_count)
    full_count = implementation.index("full_execution_count", early_return)
    assert early_gate < early_count < early_return < full_count


def test_shared_contract_matches_the_canonical_fixture_and_value_file() -> None:
    shared = source(COMMON)
    assert "ace.phantom.bootstrap-correctness-fixture/1.0.0" in shared
    assert "ace.bootstrap-correctness.complex_float64le/1.0.0" in shared
    assert "'A', 'C', 'E', 'B',\n                                                      'S', 'C', '0', '1'" in shared
    for binding in (
        "ace_source_manifest_sha256",
        "phantom_source_manifest_sha256",
        "compiler_invocation_sha256",
        "raw_air_sha256",
        "post_ckks_air_sha256",
        "context_manifest_sha256",
        "resource_manifest_sha256",
        "constant_manifest_sha256",
        "bootstrap_semantics_sha256",
        "post_operations_air_sha256",
        "post_operation_attestation_sha256",
    ):
        assert binding in shared
    for option in (
        "poly_degree",
        "vector_capacity",
        "mul_level",
        "input_level",
        "encode_transform_budget",
        "decode_transform_budget",
        "ciphertext_constant_encoding",
        "post_multiply_real",
        "post_multiply_imag",
        "post_multiply_scale_degree",
        "post_rotation_step",
    ):
        assert option in shared
    assert "provider_clear_maximum_absolute" in shared
    assert "kRequiredMaximumError = 1.0e-2" in shared
    assert "post-operation rotation is not canonical and nonzero" in shared
    assert "post-operation rotation is absent from compiler resources" in shared
    assert "bytes.size() == 128U" in shared
    assert "CaseManifestSha256" in shared
    for field in (
        "data_q_requested_bit_sizes",
        "special_p_requested_bit_sizes",
        "phantom_context_manifest_sha256",
        "physical_prime_identity_authority",
        "schedule_authority",
    ):
        assert field in shared
    assert "independent-non-authoritative-for-gpu" in shared


def test_native_oracle_uses_explicit_budgets_and_requires_full_execution() -> None:
    native = source(NATIVE)
    assert "Bootstrap_setup(bootstrap_context, budgets, dimensions, inputs.slots)" in native
    assert "UI32_VALUE_AT(budgets, 0) = inputs.encode_budget" in native
    assert "UI32_VALUE_AT(budgets, 1) = inputs.decode_budget" in native
    assert "Bootstrap_keygen(bootstrap_context, inputs.slots)" in native
    assert "Bootstrap_precom(" not in native
    assert "Eval_bootstrap_ciph(result, source, 0U, inputs.slots)" in native
    assert "Reset_bootstrap_execution_attestation()" in native
    assert "value.early_copy_return_count == 0U" in native
    assert "value.full_completion_count == 1U" in native
    assert "Decode(result, inputs.slots)" in native
    assert 'constexpr char kProvider[] = "native-ant"' in native
    assert '"context_attestation", context_attestation' in native


def test_generated_oracle_calls_only_the_linked_decomposition() -> None:
    generated = source(GENERATED)
    assert "bootstrap_full(*source, *encrypted_zero)" in generated
    assert "EncryptComplex(zero_values, inputs.input_level)" in generated
    assert "ACE_POST_CKKS_AIR_SHA256" in generated
    assert "ACE_GENERATED_DSL_ANT_SOURCE_SHA256" in generated
    assert "Eval_bootstrap_ciph(" not in generated
    assert re.search(r"\bBootstrap\s*\(", generated) is None
    assert 'constexpr char kProvider[] = "generated-ant"' in generated
    assert "Decode(&result, inputs.slots)" in generated
    assert '"context_attestation", context_attestation' in generated
    io_block = generated[generated.index('extern "C" {') : generated.index(
        "\n}\n\nnamespace {", generated.index('extern "C" {')
    )]
    for helper in (
        "Get_input_count",
        "Get_output_count",
        "Get_encode_scheme",
        "Get_decode_scheme",
    ):
        assert len(re.findall(rf"\b{helper}\s*\(", io_block)) == 1


def test_owned_paths_do_not_embed_milestone_tokens() -> None:
    token = re.compile(r"(?i)(?:^|[^a-z])m[0-9]+(?:[^a-z0-9]|$)")
    for path in (COMMON, NATIVE, GENERATED):
        assert token.search(path.name) is None
        assert token.search(source(path)) is None


def test_compile_gate_builds_and_runs_both_host_oracles_before_gpu_use() -> None:
    script = source(COMPILE_ONLY)
    assert 'if gate == "bootstrap":' in script
    assert 'if gate in {"bootstrap", "all"}:' not in script
    for option in (
        "--vector-capacity",
        "--q-part-count",
        "--encode-transform-budget",
        "--decode-transform-budget",
        "--ciphertext-constant-encoding",
        "--packing",
        "--post-multiply-real",
        "--post-multiply-imag",
        "--post-multiply-scale-degree",
        "--post-rotation-step",
        "--fixture-id",
        "--fixture-seed",
        "--inside-margin",
        "--provider-clear-threshold",
        "--gpu-native-threshold",
        "--gpu-generated-threshold",
        "--repeat-threshold",
        "--host-oracle-timeout-seconds",
    ):
        assert option in script
    native_run = script.index(
        'RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 timeout "${HOST_ORACLE_TIMEOUT_SECONDS}"'
    )
    generated_run = script.index(
        'RTLIB_DISABLE_BOOTSTRAP_PRECOM=1 timeout "${HOST_ORACLE_TIMEOUT_SECONDS}"',
        native_run + 1,
    )
    host_replay = script.index("bootstrap_correctness.py\" verify-host", generated_run)
    gpu_build = script.index('local correctness_compile_arguments=', generated_run)
    assert native_run < generated_run < host_replay < gpu_build
    assert "bootstrap_correctness_fixture.json" in script
    assert "native_ant_reference.json" in script
    assert "generated_ant_reference.json" in script
    assert "host_oracle_replay.json" in script
    assert 'echo "native ANT oracle failed with exit ${native_oracle_exit}"' in script
    assert (
        'echo "generated DSL/ANT oracle failed with exit '
        '${generated_oracle_exit}"' in script
    )
    assert 'cat -- "${BOOTSTRAP_RESULTS}/native-ant.stderr.txt" >&2' in script
    assert 'cat -- "${BOOTSTRAP_RESULTS}/generated-ant.stderr.txt" >&2' in script
    assert "generated_bootstrap_phantom_correctness_sm80" in script
    assert "ace.phantom.bootstrap-artifacts/3.0.0" in script
    assert "ace.phantom.bootstrap-host-qualification/2.0.0" in script
    assert "ace.phantom.bootstrap-generated-artifact-audit/3.0.0" in script
    for audit_option in (
        "--ant-source",
        "--generation-record",
        "--bootstrap-semantics",
        "--post-operations-air",
        "--post-operation-attestation",
    ):
        assert audit_option in script
    for suite in (
        "test_generated_bootstrap_invocation.py",
        "test_bootstrap_correctness.py",
        "test_generated_bootstrap_host_oracle_sources.py",
        "test_generated_bootstrap_correctness_lifecycle.py",
        "test_bootstrap_host_freeze_wrapper.py",
    ):
        assert suite in script


def test_host_freeze_forwards_only_explicit_qualification_arguments() -> None:
    script = source(HOST_FREEZE)
    assert '"arguments": arguments' in script
    assert "ace.phantom.bootstrap-host-freeze-payload/2.0.0" in script
    assert '"${QUALIFICATION_ARGS[@]}"' in script
    assert '"parameters": {' not in script
    assert '--gate bootstrap "${QUALIFICATION_ARGS[@]}"' in script
    assert "qualification rotation is not canonical signed modulo capacity" in script


def test_phantom_runtime_accepts_attested_raw_scale_one_plaintext() -> None:
    runtime = source(
        ROOT / "fhe-cmplr/rtlib/phantom/src/phantom_lib.cu"
    )
    assert "degree < 0.0" in runtime
    assert "must be a nonnegative integer" in runtime
    assert "ScaleForDegree(integral_degree" in runtime
    assert "QueryPlainScaleDegree" in runtime
