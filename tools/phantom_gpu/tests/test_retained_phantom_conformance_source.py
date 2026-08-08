from __future__ import annotations

import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = (
    REPO_ROOT
    / "tools"
    / "phantom_gpu"
    / "harness"
    / "retained_ckks_phantom_conformance.cu"
)
FIXTURE = REPO_ROOT / "tools" / "phantom_gpu" / "fixtures" / "retained_ckks_v1.json"


def _source() -> str:
    return HARNESS.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", source)
    assert match is not None, f"missing function {name}"
    opening = source.find("{", match.start())
    depth = 1
    position = opening + 1
    while position < len(source) and depth:
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
        position += 1
    assert depth == 0, f"unterminated function {name}"
    return source[opening + 1 : position - 1]


def test_decoded_path_uses_stable_runtime_surface() -> None:
    source = _source()
    body = _function_body(source, "RunDecoded")
    for call in (
        "Conjugate_ciph",
        "Rotate_batch_ciph",
        "Raise_mod",
        "Mul_mono_ciph",
    ):
        assert re.search(rf"\b{call}\s*\(", body)
    for direct in (
        "raise_modulus",
        "multiply_by_monomial",
        "export_ciphertext_coefficients",
        "copy_ciphertext_to_coefficient_form",
        "copy_ciphertext_to_ntt_form",
    ):
        assert not re.search(rf"\b{direct}\s*\(", body)


def test_exact_path_imports_and_observes_both_representations() -> None:
    source = _source()
    body = _function_body(source, "RunExact")
    assert "ImportCoefficientCipher" in body
    assert "raise_modulus(" in body
    assert "multiply_by_monomial(" in body
    assert "copy_ciphertext_to_ntt_form(" in body
    assert body.count("export_ciphertext_coefficients(") >= 6
    assert "NTT and coefficient paths disagree" in body
    assert "EXACT_NTT_RAISE" in body
    assert "mutated its NTT source" in body
    assert "NTT and coefficient-form exact residues disagree" in body
    assert "ReadI64Blob" in body
    assert "ReduceSignedSource" in body
    assert 'exact_reference.at("ordered_data_q_moduli")' not in body
    assert 'specification.at("source")' not in body
    raise_branch_start = body.index("if (index == 0)")
    raise_branch_end = body.index("continue;", raise_branch_start)
    raise_branch = body[raise_branch_start:raise_branch_end]
    assert raise_branch.count("raise_modulus(") == 2
    assert "copy_ciphertext_to_ntt_form(" in raise_branch
    assert "raise_modulus(" not in body[raise_branch_end + len("continue;") :]
    for wrapper in (
        "Conjugate_ciph",
        "Rotate_batch_ciph",
        "Raise_mod",
        "Mul_mono_ciph",
    ):
        assert not re.search(rf"\b{wrapper}\s*\(", body)


def test_case_matrix_order_and_strict_artifact_schemas_are_literal() -> None:
    source = _source()
    expected_decoded = (
        "conjugate.bounded_nonperiodic",
        "conjugate_twice.bounded_nonperiodic",
        "rotate_batch.bounded_nonperiodic",
        "raise_mod.bounded_nonperiodic",
        "mul_mono.inverse_composition.bounded_nonperiodic",
        "composite.bounded_nonperiodic",
    )
    positions = [source.index(case_id) for case_id in expected_decoded]
    assert positions == sorted(positions)
    for label in (
        "N_over_2",
        "N",
        "3N_over_2",
        "2N_minus_1",
        "2N_plus_1",
    ):
        assert label in source
    assert "ace.phantom.retained_ckks.provider-result/3.0.0" in source
    assert "ace.phantom.retained_ckks.exact-observed/2.0.0" in source
    compact = re.sub(r"\s+", "", source)
    assert "'A','C','E','R','C','K','0','1'" in compact
    assert "'A','C','E','R','N','S','0','1'" in compact
    assert '"component,modulus,coefficient"' in source
    assert "ace.phantom.retained_ckks.exact-source/2.0.0" in source
    assert "exact_source_json_sha256" in source
    assert "exact_source_binary_sha256" in source
    assert "AppendDoubleLe" in source
    assert "AppendU64Le" in source


def test_batch_contract_and_ownership_stress_are_explicit() -> None:
    source = _source()
    assert "Json({5, 0, -7, 5})" in source
    assert 'fixture.at("production_rotation_batches")' in source
    assert "RequireIndependentBatch" in source
    assert "mutating one output changed a sibling output" in source
    assert "retained_batches.push_back" in source
    assert 'iterations == 100' in source
    assert 'fixture.at("ownership").at("free_order")' in source
    assert "OwnershipToken" in source


def test_decoded_composite_invokes_the_generated_interface() -> None:
    source = _source()
    body = _function_body(source, "RunDecoded")
    assert '#include "retained_ckks_generated_interface.h"' in source
    assert "CIPHERTEXT result = retained_ckks_composite(*source);" in body


def test_source_preservation_hashes_exact_device_residues() -> None:
    source = _source()
    body = _function_body(source, "CipherResiduesSha256")
    assert "cipher->size()" in body
    assert "cipher->coeff_modulus_size()" in body
    assert "cipher->poly_modulus_degree()" in body
    assert "cudaMemcpyDeviceToHost" in body
    assert "PackU64(residues)" in body
    snapshot = _function_body(source, "Snapshot")
    assert "CipherResiduesSha256(cipher)" in snapshot
    assert '"source_values_sha256_before", before._residues_sha256' in source
    assert '"source_values_sha256_after", after._residues_sha256' in source


def test_raised_decodes_use_a_verified_temporary_q0_projection() -> None:
    source = _source()
    projection = _function_body(source, "DecodeStrictQ0")
    assert "Copy_ciph(&projected, result)" in projection
    assert "while (Active_q_count(&projected) > 1)" in projection
    assert "Mod_switch(&projected, &projected)" in projection
    assert "Q0Tower(&projected) == full_q0" in projection
    assert "CipherResiduesSha256(result) == full_residues" in projection
    assert "result->data() == full_buffer" in projection
    assert '"strict_q0_prefix_drop"' in source
    body = _function_body(source, "RunDecoded")
    assert body.count("nullptr, true") == 2


def test_rejections_publish_stable_identifiers_and_tokens() -> None:
    source = _source()
    expected = {
        "conjugate_missing_key": "CONJUGATE_KEY_MISSING",
        "rotate_batch_missing_nonzero_key": "RESOURCE_ROTATE_BATCH",
        "rotate_batch_source_overlap": "ROTATE_BATCH_ALIAS",
        "raise_alias": "RAISE_MOD_ALIAS",
        "raise_size": "RAISE_MOD_SIZE",
        "raise_chain": "RAISE_MOD_SOURCE_LEVEL",
        "raise_target": "RAISE_MOD_TARGET",
        "raise_malformed_metadata": "RAISE_MOD_SOURCE",
        "mul_mono_undeclared": "MUL_MONO_RESOURCE",
    }
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed = {
        item["id"]: (item["diagnostic"], item["manifest"])
        for item in fixture["runtime_rejections"]
    }
    assert {key: value[0] for key, value in observed.items()} == expected
    assert observed["conjugate_missing_key"][1] == "keyless-conjugation"
    assert observed["rotate_batch_missing_nonzero_key"][1] == "keyless-rotation"
    assert all(
        manifest == "production"
        for rejection_id, (_diagnostic, manifest) in observed.items()
        if "missing" not in rejection_id
    )
    assert 'fixture.at("runtime_rejections")' in source
    assert "ACE_RETAINED_EXPECT_DIAGNOSTIC[" in source
    assert "ACE_REJECTION_MANIFEST_ID" in source
    assert "expected_manifest == ACE_REJECTION_MANIFEST_ID" in source
    assert "PHANTOM_RESOURCE_CONJUGATION_KEY" in source
    assert "missing_nonzero_key" in source
    assert "REJECTION_INITIALIZATION" in source
    assert "REJECTION_RETURNED" in source
    assert "raise_invalid_form" not in source
    assert 'Require(source->is_ntt_form(), "REJECTION_SETUP"' in source
    assert "source->set_ntt_form(false)" not in source
    assert "source->set_parms_id(phantom::parms_id_zero)" in source


def test_context_and_resources_have_one_generated_authority() -> None:
    source = _source()
    assert "Get_phantom_context_manifest()" in source
    assert "Get_phantom_resource_manifest()" in source
    assert "MakeExactContext" in source
    exact_context = _function_body(source, "MakeExactContext")
    for field in (
        "_poly_degree",
        "_data_q_count",
        "_data_q_bit_sizes",
        "_special_p_count",
        "_special_p_bit_sizes",
        "_hamming_weight",
        "_logical_slots",
    ):
        assert field in exact_context
    assert "CoeffModulus::Create" in exact_context
    assert "ordered_data_q_moduli" in source
    authenticated = _function_body(source, "AuthenticateContext")
    assert "ParseJson(bytes, path)" in authenticated
    assert 'require_ordered_bits("data_q_bit_sizes"' in authenticated
    assert 'require_ordered_bits("special_p_bit_sizes"' in authenticated
    assert "Sha256(bytes)" in authenticated
    assert "CONTEXT_FILE_MISMATCH" in authenticated
    assert "JSON_DUPLICATE_KEY" in source
    assert source.count("AuthenticateContext(argv[3])") == 3
    assert source.count('{"first_data_chain_index", first_data_chain_index}') == 2
    for profile_literal in ("16384", "8192", "192"):
        assert re.search(rf"\b{profile_literal}\b", source) is None


def test_harness_has_no_refresh_or_lowering_dependencies() -> None:
    source = _source()
    forbidden = (
        r'#\s*include\s*[<"][^">]*(?:boot|bootstrap)[^">]*[">]',
        r'#\s*include\s*[<"][^">]*(?:^|/)(?:ant|poly)(?:/|_)[^">]*[">]',
        r"\b(?:Bootstrapper|Phantom_bootstrap|bootstrap_3)\b",
        r"\b(?:CoeffToSlots|SlotToCoeffs|EvalMod)\b",
        r"\b(?:Raise_mod_poly|Hw_)\w*\s*\(",
    )
    for pattern in forbidden:
        assert re.search(pattern, source, re.IGNORECASE) is None
    assert re.search(r"\bm\d+(?:_\w+)?\b", source, re.IGNORECASE) is None
