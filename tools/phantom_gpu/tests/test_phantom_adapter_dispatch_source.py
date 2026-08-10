from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER = REPO_ROOT / "fhe-cmplr" / "rtlib" / "phantom" / "src" / "phantom_lib.cu"


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\bvoid\s+{name}\s*\([^)]*\)\s*\{{", source)
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


def test_rotate_batch_matches_raw_resource_then_normalizes_provider_steps() -> None:
    body = _function_body(ADAPTER.read_text(encoding="utf-8"), "RotateBatch")

    raw_match = body.index("std::equal(steps, steps + count")
    normalization = body.index(
        "NormalizeRotation(steps[index], _logical_slots)"
    )
    provider_call = body.index("rotate_batch(*_context")

    assert raw_match < normalization < provider_call
    assert "provider_steps.reserve(count)" in body
    assert "rotate_batch(*_context, *source, provider_steps" in body


def test_monomial_alias_uses_provider_in_place_entry_point() -> None:
    body = _function_body(
        ADAPTER.read_text(encoding="utf-8"), "MultiplyMonomial"
    )

    assert "if (result == source)" in body
    assert "multiply_by_monomial_inplace(*_context, *result, power)" in body
    assert "multiply_by_monomial(*_context, *source, power, *result)" in body


def test_single_element_mask_preserves_scalar_broadcast_semantics() -> None:
    body = _function_body(ADAPTER.read_text(encoding="utf-8"), "EncodeMask")

    length_selection = body.index(
        "const size_t encoded_len = len == 1 ? _logical_slots : len"
    )
    vector_construction = body.index("std::vector<double> values(encoded_len")
    provider_encoding = body.index("EncodeVector(plain, values")

    assert length_selection < vector_construction < provider_encoding
    assert "std::vector<double> values(len" not in body


def test_provider_outputs_are_live_before_metadata_validation() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    for function, provider, result_check in (
        ("Rescale", "RESCALE_PROVIDER", "RESCALE_RESULT"),
        ("ModSwitch", "MODSWITCH_PROVIDER", "MODSWITCH_RESULT"),
        ("Conjugate", "CONJUGATE_PROVIDER", "CONJUGATE_RESULT"),
        ("MultiplyMonomial", "MUL_MONO_PROVIDER", "MUL_MONO_RESULT"),
    ):
        body = _function_body(source, function)
        provider_call = body.index(f'ProviderCall("{provider}"')
        live_transition = body.index("MarkCipher(result, ObjectState::kLive)")
        metadata_validation = body.index(f'ActiveQ(result, "{result_check}")')
        assert provider_call < live_transition < metadata_validation


def test_add_sub_uses_logical_scale_degree_and_normalizes_destinations() -> None:
    source = ADAPTER.read_text(encoding="utf-8")

    cipher_validator = _function_body(source, "ValidateAddSub")
    assert "QueryScaleDegree(left, diagnostic)" in cipher_validator
    assert "QueryScaleDegree(right, diagnostic)" in cipher_validator
    assert "left_scale_degree != right_scale_degree" in cipher_validator
    assert "left->scale() != right->scale()" not in cipher_validator

    plain_validator = _function_body(source, "ValidateCipherPlainAddSub")
    assert "QueryScaleDegree(left, diagnostic)" in plain_validator
    assert "QueryPlainScaleDegree(right, diagnostic)" in plain_validator
    assert "left_scale_degree != right_scale_degree" in plain_validator
    assert "left->scale() != right->scale()" not in plain_validator

    add_cipher = _function_body(source, "AddCipher")
    assert add_cipher.index("const double output_scale = left->scale()") < (
        add_cipher.index('ProviderCall("ADD_PROVIDER"')
    )
    assert add_cipher.index('ProviderCall("ADD_PROVIDER"') < add_cipher.index(
        "result->scale() = output_scale"
    )

    sub_cipher = _function_body(source, "SubCipher")
    provider = sub_cipher.index('ProviderCall("SUB_PROVIDER"')
    restore = sub_cipher.index("result->scale() = output_scale")
    assert sub_cipher.count("result->scale() = right->scale()") == 2
    assert sub_cipher.count("result->scale() = left->scale()") == 1
    assert provider < sub_cipher.index("result->scale() = right->scale()")
    assert provider < sub_cipher.index("result->scale() = left->scale()")
    first_normalize = sub_cipher.index("result->scale() = right->scale()")
    first_provider_call = sub_cipher.index("sub_inplace(")
    assert first_normalize < first_provider_call
    rhs_normalize = sub_cipher.index("result->scale() = left->scale()")
    rhs_provider_call = sub_cipher.index("sub_inplace(", first_provider_call + 1)
    assert rhs_normalize < rhs_provider_call
    distinct_normalize = sub_cipher.rindex("result->scale() = right->scale()")
    distinct_provider_call = sub_cipher.rindex("sub_inplace(")
    assert distinct_normalize < distinct_provider_call
    assert sub_cipher.rindex("sub_inplace(") < restore
    assert "left->scale() =" not in sub_cipher
    assert "right->scale() =" not in sub_cipher
    assert "left->set_scale(" not in sub_cipher
    assert "right->set_scale(" not in sub_cipher

    add_plain = _function_body(source, "AddPlain")
    assert add_plain.index('ProviderCall("ADD_PLAIN_PROVIDER"') < (
        add_plain.index("result->scale() = output_scale")
    )

    sub_plain = _function_body(source, "SubPlain")
    provider = sub_plain.index('ProviderCall("SUB_PLAIN_PROVIDER"')
    normalize = sub_plain.index("result->scale() = right->scale()")
    provider_call = sub_plain.index("sub_plain_inplace(")
    restore = sub_plain.index("result->scale() = output_scale")
    assert provider < normalize < provider_call < restore
    assert "right->scale() =" not in sub_plain
    assert "right->set_scale(" not in sub_plain


def test_cipher_validation_rejects_foreign_parameter_identity() -> None:
    body = _function_body(
        ADAPTER.read_text(encoding="utf-8"), "ValidateCipher"
    )
    assert (
        "cipher->parms_id() != context_data.parms().parms_id()" in body
    )
