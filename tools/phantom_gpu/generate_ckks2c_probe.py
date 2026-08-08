#!/usr/bin/env python3
"""Generate a small Phantom CUDA translation unit through the real CKKS2C path."""

from __future__ import annotations

import argparse
from pathlib import Path

from ace_edsl.edsl import (
    AceEDSL,
    AcePipeline,
    CkksCiphertext,
    ckks_kernel,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-manifest", required=True, type=Path)
    parser.add_argument("--resource-manifest", required=True, type=Path)
    parser.add_argument("--post-ckks-air", type=Path)
    parser.add_argument("--poly-degree", required=True, type=int)
    parser.add_argument("--mul-level", required=True, type=int)
    parser.add_argument("--input-level", required=True, type=int)
    parser.add_argument("--security-level", required=True, type=int)
    parser.add_argument("--scaling-factor-bits", required=True, type=int)
    parser.add_argument("--first-prime-bits", required=True, type=int)
    parser.add_argument("--hamming-weight", required=True, type=int)
    parser.add_argument(
        "--resource-mode", choices=("ordinary", "keyless"), default="ordinary"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.output.suffix != ".cu":
        raise SystemExit("output must use the .cu suffix")

    for manifest_path in (
        arguments.context_manifest,
        arguments.resource_manifest,
    ):
        if manifest_path.suffix != ".json":
            raise SystemExit("manifest outputs must use the .json suffix")

    for output_path in (
        arguments.output,
        arguments.context_manifest,
        arguments.resource_manifest,
        arguments.post_ckks_air,
    ):
        if output_path is None:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fhe_parameters = {
        "poly_degree": arguments.poly_degree,
        "mul_level": arguments.mul_level,
        "input_level": arguments.input_level,
        "security_level": arguments.security_level,
        "scaling_factor_bits": arguments.scaling_factor_bits,
        "first_prime_bits": arguments.first_prime_bits,
        "hamming_weight": arguments.hamming_weight,
    }

    AceEDSL._get_dsl.cache_clear()

    shape = (fhe_parameters["poly_degree"],)
    left = CkksCiphertext(shape=shape, name="left")
    if arguments.resource_mode == "ordinary":
        @ckks_kernel
        def arithmetic_probe(
            left_value: CkksCiphertext, right_value: CkksCiphertext
        ) -> CkksCiphertext:
            return ((left_value * right_value) + left_value.rotate(3)) + (
                right_value.rotate(-3)
            )

        right = CkksCiphertext(shape=shape, name="right")
        arithmetic_probe(left, right)
    else:
        @ckks_kernel
        def keyless_probe(left_value: CkksCiphertext) -> CkksCiphertext:
            return left_value.rotate(0)

        keyless_probe(left)

    module = AceEDSL._get_dsl().current_air_module
    pipeline = AcePipeline(module).configure_fhe(
        **fhe_parameters,
        data_file="",
        provider="phantom",
        codegen_ir="ckks",
        context_manifest_file=str(arguments.context_manifest),
        resource_manifest_file=str(arguments.resource_manifest),
    )
    result = pipeline.run(
        start_domain="fhe::ckks", dump_stages=True, verbose=False
    )
    if not result.success:
        raise SystemExit(f"CKKS2C generation failed: {result.error}")
    if result.stages_completed != ["ckks_driver", "ckks2c"]:
        raise SystemExit(
            f"unexpected terminal stages: {result.stages_completed}"
        )
    source = result.c_code or ""
    if not source:
        raise SystemExit("CKKS2C returned empty source")

    if arguments.post_ckks_air is not None:
        if arguments.post_ckks_air.suffix != ".air":
            raise SystemExit("post-CKKS AIR output must use the .air suffix")
        post_ckks_air = result.air_dumps.get("ckks_driver", "")
        if not post_ckks_air:
            raise SystemExit("CKKS2C returned no post-CKKS AIR dump")
        post_ckks_air = canonicalize_qualification_air(
            post_ckks_air, Path(__file__)
        )
        arguments.post_ckks_air.write_text(post_ckks_air, encoding="utf-8")
    arguments.output.write_text(source, encoding="utf-8")
    print(f"generated {arguments.output} ({len(source.encode('utf-8'))} bytes)")
    return 0


def canonicalize_qualification_air(
    air_dump: str, physical_source_path: Path
) -> str:
    """Replace the probe's checkout path with its stable repository path."""
    logical_source = "tools/phantom_gpu/generate_ckks2c_probe.py"
    physical_source = physical_source_path.resolve().as_posix()
    physical_size = len(physical_source.encode("utf-8"))
    logical_size = len(logical_source.encode("utf-8"))
    physical_entry = (
        f'"{physical_source}" length(0x{physical_size:x})'
    )
    logical_entry = f'"{logical_source}" length(0x{logical_size:x})'
    if air_dump.count(physical_entry) != 1:
        raise SystemExit(
            "post-CKKS AIR must contain exactly one physical probe source entry"
        )

    header_prefix = "STRING TABLE ("
    header_suffix = " Bytes)\n"
    if not air_dump.startswith(header_prefix):
        raise SystemExit("post-CKKS AIR is missing its string-table header")
    header_end = air_dump.find(header_suffix, len(header_prefix))
    if header_end < 0:
        raise SystemExit("post-CKKS AIR has a malformed string-table header")
    size_text = air_dump[len(header_prefix):header_end]
    if not size_text.isdecimal():
        raise SystemExit("post-CKKS AIR string-table size is not decimal")
    canonical_size = int(size_text) - physical_size + logical_size
    if canonical_size < 0:
        raise SystemExit("post-CKKS AIR string-table size is inconsistent")

    canonical = (
        f"{header_prefix}{canonical_size}{header_suffix}"
        + air_dump[header_end + len(header_suffix):]
    )
    canonical = canonical.replace(physical_entry, logical_entry, 1)
    if physical_source in canonical:
        raise SystemExit("physical probe source path remains in post-CKKS AIR")
    if canonical.count(logical_entry) != 1:
        raise SystemExit(
            "canonical post-CKKS AIR must contain exactly one logical probe "
            "source entry"
        )
    for line in canonical.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("STR["):
            continue
        value_start = stripped.find('"')
        value_end = stripped.rfind('" length(')
        if value_start < 0 or value_end <= value_start:
            raise SystemExit("post-CKKS AIR has a malformed string entry")
        if stripped[value_start + 1:value_end].startswith("/"):
            raise SystemExit(
                "absolute source path remains in canonical post-CKKS AIR"
            )
    return canonical


if __name__ == "__main__":
    raise SystemExit(main())
