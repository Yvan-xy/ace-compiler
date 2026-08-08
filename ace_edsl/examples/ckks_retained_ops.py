#!/usr/bin/env python3
"""Emit focused CKKS AIR micrographs for the retained backend operations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from ace_edsl.edsl import AceEDSL, CkksCiphertext, ckks_kernel


EDGE_ROTATION_BATCH = [5, 0, -7, 5]
GENERATED_FUNCTIONS = (
    "retained_ckks_conformance",
)


class ContextManifestError(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextManifestError(f"duplicate context-manifest key: {key}")
        result[key] = value
    return result


def load_context_dimensions(path: Path) -> tuple[int, int, int]:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContextManifestError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContextManifestError(f"cannot read context manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ContextManifestError("context manifest must be an object")
    degree = manifest.get("polynomial_degree")
    slots = manifest.get("logical_slot_capacity")
    data_q = manifest.get("data_q_bit_sizes")
    if (
        isinstance(degree, bool)
        or not isinstance(degree, int)
        or degree < 2
        or degree & (degree - 1)
    ):
        raise ContextManifestError("polynomial_degree must be a power of two")
    if slots != degree // 2:
        raise ContextManifestError("logical_slot_capacity must equal N/2")
    if not isinstance(data_q, list) or not data_q:
        raise ContextManifestError("data_q_bit_sizes must be a nonempty array")
    return degree, slots, len(data_q)


def _load_retained_fixture(path: Path) -> dict[str, Any]:
    try:
        fixture = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContextManifestError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContextManifestError(f"cannot read retained fixture: {error}") from error
    if (
        not isinstance(fixture, dict)
        or fixture.get("schema_version")
        != "ace.phantom.retained_ckks.fixture/1.0.0"
    ):
        raise ContextManifestError("retained fixture schema is unsupported")
    return fixture


def load_production_rotation_batches(path: Path) -> tuple[tuple[int, ...], ...]:
    fixture = _load_retained_fixture(path)
    batches = fixture.get("production_rotation_batches")
    if not isinstance(batches, list) or not batches:
        raise ContextManifestError("retained fixture has no production rotation batches")
    result: list[tuple[int, ...]] = []
    for batch_index, batch in enumerate(batches):
        if (
            not isinstance(batch, list)
            or not batch
            or any(isinstance(step, bool) or not isinstance(step, int) for step in batch)
        ):
            raise ContextManifestError(
                f"production rotation batch {batch_index} is malformed"
            )
        result.append(tuple(batch))
    return tuple(result)


def load_normalized_monomial_powers(path: Path, degree: int) -> tuple[int, ...]:
    fixture = _load_retained_fixture(path)
    symbolic = fixture.get("monomial_powers")
    expected = ("0", "N/2", "N", "3N/2", "2N-1", "2N+1")
    if symbolic != list(expected):
        raise ContextManifestError("retained monomial power matrix changed")
    raw = (0, degree // 2, degree, 3 * degree // 2, 2 * degree - 1, 2 * degree + 1)
    normalized = tuple(power % (2 * degree) for power in raw)
    if len(set(normalized)) != len(normalized):
        raise ContextManifestError("normalized monomial power matrix has duplicates")
    return normalized


@dataclass(frozen=True)
class RetainedCkksMicrographs:
    polynomial_degree: int
    logical_slots: int
    full_data_q_count: int
    production_rotation_batches: tuple[tuple[int, ...], ...]
    normalized_monomial_powers: tuple[int, ...]
    conjugate: Callable[..., Any]
    rotate_batch: Callable[..., Any]
    raise_mod: Callable[..., Any]
    mul_mono: Callable[..., Any]
    composite: Callable[..., Any]
    conformance: Callable[..., Any]


def _emit_rotation_batches(ct: Any, batches: tuple[tuple[int, ...], ...]) -> list[Any]:
    """Trace constant batches without turning Python indices into AIR indices."""

    return [ct.rotate_batch(list(batch)) for batch in batches]


def _combine_first_batch_outputs(outputs: list[Any]) -> Any:
    """Keep every emitted batch live through terminal source generation."""

    accumulator = outputs[0][0]
    for output in outputs[1:]:
        accumulator = accumulator + output[0]
    return accumulator


def create_retained_ckks_micrographs(
    context_manifest: Path,
    retained_fixture: Path,
) -> RetainedCkksMicrographs:
    """Create graphs whose only context constants come from the manifest."""

    degree, slots, full_data_q_count = load_context_dimensions(context_manifest)
    production_rotation_batches = load_production_rotation_batches(retained_fixture)
    normalized_monomial_powers = load_normalized_monomial_powers(
        retained_fixture, degree
    )

    @ckks_kernel
    def retained_ckks_conjugate(ct: CkksCiphertext) -> CkksCiphertext:
        return ct.conjugate()

    @ckks_kernel
    def retained_ckks_rotate_batches(ct: CkksCiphertext) -> CkksCiphertext:
        outputs = _emit_rotation_batches(
            ct, (tuple(EDGE_ROTATION_BATCH),) + production_rotation_batches
        )
        return _combine_first_batch_outputs(outputs)

    @ckks_kernel
    def retained_ckks_raise_mod(ct: CkksCiphertext) -> CkksCiphertext:
        return ct.raise_mod(full_data_q_count)

    @ckks_kernel
    def retained_ckks_mul_monomials(ct: CkksCiphertext) -> CkksCiphertext:
        outputs = [ct.mul_mono(power) for power in normalized_monomial_powers]
        accumulator = outputs[0]
        for output in outputs[1:]:
            accumulator = accumulator + output
        return accumulator

    @ckks_kernel
    def retained_ckks_composite(ct: CkksCiphertext) -> CkksCiphertext:
        raised = ct.raise_mod(full_data_q_count)
        multiplied = raised.mul_mono(degree // 2)
        conjugated = multiplied.conjugate()
        outputs = conjugated.rotate_batch(EDGE_ROTATION_BATCH)
        return (outputs[0] + outputs[1]) + (outputs[2] + outputs[3])

    @ckks_kernel
    def retained_ckks_conformance(ct: CkksCiphertext) -> CkksCiphertext:
        """Executable matrix from one bottom-Q input to one ciphertext."""

        raised = ct.raise_mod(full_data_q_count)
        monomials = [
            raised.mul_mono(power) for power in normalized_monomial_powers
        ]
        multiplied = monomials[normalized_monomial_powers.index(degree // 2)]
        conjugated = multiplied.conjugate()
        batches = _emit_rotation_batches(
            conjugated,
            (tuple(EDGE_ROTATION_BATCH),) + production_rotation_batches,
        )
        accumulator = _combine_first_batch_outputs(batches)
        for monomial in monomials:
            accumulator = accumulator + monomial
        return accumulator

    return RetainedCkksMicrographs(
        polynomial_degree=degree,
        logical_slots=slots,
        full_data_q_count=full_data_q_count,
        production_rotation_batches=production_rotation_batches,
        normalized_monomial_powers=normalized_monomial_powers,
        conjugate=retained_ckks_conjugate,
        rotate_batch=retained_ckks_rotate_batches,
        raise_mod=retained_ckks_raise_mod,
        mul_mono=retained_ckks_mul_monomials,
        composite=retained_ckks_composite,
        conformance=retained_ckks_conformance,
    )


def declare_retained_ckks_conformance_module(
    context_manifest: Path,
    retained_fixture: Path,
) -> RetainedCkksMicrographs:
    """Declare every retained conformance entry point in one AIR module."""

    AceEDSL._get_dsl.cache_clear()
    graphs = create_retained_ckks_micrographs(context_manifest, retained_fixture)
    graphs.conformance(
        CkksCiphertext(
            shape=(graphs.polynomial_degree,), name="retained_conformance_input"
        )
    )
    return graphs


def emit_air(
    context_manifest: Path,
    retained_fixture: Path,
    graph_name: str,
    output: Path,
) -> None:
    if graph_name == "conformance":
        declare_retained_ckks_conformance_module(context_manifest, retained_fixture)
    else:
        AceEDSL._get_dsl.cache_clear()
        graphs = create_retained_ckks_micrographs(context_manifest, retained_fixture)
        graph = getattr(graphs, graph_name)
        graph(CkksCiphertext(shape=(graphs.polynomial_degree,), name="input_ct"))
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise RuntimeError("micrograph did not emit AIR")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(module.dump(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument(
        "--graph",
        required=True,
        choices=(
            "conjugate",
            "rotate_batch",
            "raise_mod",
            "mul_mono",
            "composite",
            "conformance",
        ),
    )
    parser.add_argument("--output-air", required=True, type=Path)
    arguments = parser.parse_args()
    emit_air(
        arguments.context_manifest,
        arguments.fixture,
        arguments.graph,
        arguments.output_air,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
