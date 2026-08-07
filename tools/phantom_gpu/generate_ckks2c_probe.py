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
    ):
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

    arguments.output.write_text(source, encoding="utf-8")
    print(f"generated {arguments.output} ({len(source.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
