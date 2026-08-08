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
