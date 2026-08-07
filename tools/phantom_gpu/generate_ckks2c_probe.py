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
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.output.suffix != ".cu":
        raise SystemExit("output must use the .cu suffix")

    AceEDSL._get_dsl.cache_clear()

    @ckks_kernel
    def arithmetic_probe(
        left: CkksCiphertext, right: CkksCiphertext
    ) -> CkksCiphertext:
        return (left * right) + left.rotate(3)

    left = CkksCiphertext(shape=(16384,), name="left")
    right = CkksCiphertext(shape=(16384,), name="right")
    arithmetic_probe(left, right)

    module = AceEDSL._get_dsl().current_air_module
    pipeline = AcePipeline(module).configure_fhe(
        poly_degree=16384,
        mul_level=4,
        security_level=0,
        scaling_factor_bits=56,
        first_prime_bits=60,
        hamming_weight=192,
        data_file="",
        provider="phantom",
        codegen_ir="ckks",
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

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(source, encoding="utf-8")
    print(f"generated {arguments.output} ({len(source.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
