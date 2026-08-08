#!/usr/bin/env python3
"""Generate canonical default-bootstrap post-CKKS AIR and extract only RNUMs.

This traces and lowers the checked-in default EDSL bootstrap through the CKKS
driver.  It never enters POLY/POLY2C or CKKS2C and never executes bootstrap
cryptography.  The resulting artifact exists solely to bind the reviewed
production rotation batches in the retained qualification fixture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "ace_edsl" / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ace_edsl.edsl import AceEDSL, AcePipeline, CkksCiphertext  # noqa: E402
import bootstrap_full as bootstrap_module  # noqa: E402
from generate_retained_ckks_fixtures import (  # noqa: E402
    extract_production_rotation_batches,
)
from retained_air_tools import (  # noqa: E402
    canonicalize_checkout_paths,
    require_clean_tracked_sources,
    sha256_text,
    write_json,
)


def _reject_environment_overrides() -> None:
    overrides = sorted(name for name in os.environ if name.startswith("ACE_BOOTSTRAP_"))
    if overrides:
        raise SystemExit(
            "default production AIR forbids ACE_BOOTSTRAP_* overrides: "
            + ", ".join(overrides)
        )


def generate(output_air: Path, output_record: Path) -> None:
    _reject_environment_overrides()
    ace_commit = require_clean_tracked_sources(
        REPOSITORY,
        (
            "ace_edsl/examples/bootstrap_full.py",
            "ace_edsl/examples/bootstrap_ant_constants.py",
            "ace_edsl/edsl",
            "tools/phantom_gpu/generate_retained_production_air.py",
            "tools/phantom_gpu/generate_retained_ckks_fixtures.py",
            "tools/phantom_gpu/retained_air_tools.py",
        ),
    )
    AceEDSL._get_dsl.cache_clear()
    config = bootstrap_module._bootstrap_trace_config()
    ct = CkksCiphertext(shape=(config.poly_degree,), name="input_ct")
    zero = CkksCiphertext(shape=(config.poly_degree,), name="zero_ct")
    arguments = (
        [ct, zero, 1.0]
        + list(bootstrap_module.G_COEFFICIENTS_UNIFORM_HW_192)
        + list(config.double_angle_scalars)
        + [config.post_scale]
    )
    bootstrap_module.bootstrap_full(*arguments)
    module = AceEDSL._get_dsl().current_air_module
    if module is None:
        raise SystemExit("default bootstrap trace emitted no AIR")
    pipeline = AcePipeline(module).configure_fhe(
        poly_degree=config.poly_degree,
        mul_level=config.mul_level,
        input_level=bootstrap_module._bootstrap_input_level(),
        security_level=0,
        scaling_factor_bits=config.scaling_factor_bits,
        first_prime_bits=config.first_prime_bits,
        hamming_weight=config.hamming_weight,
        data_file="",
        provider="ant",
        codegen_ir="poly",
    )
    result = pipeline.run_ckks_driver()
    if not result.get("success"):
        raise SystemExit(f"CKKS driver failed: {result.get('message', result)}")
    canonical = canonicalize_checkout_paths(module.dump(), REPOSITORY)
    output_air.parent.mkdir(parents=True, exist_ok=True)
    output_air.write_text(canonical, encoding="utf-8")
    batches = extract_production_rotation_batches(output_air)
    write_json(
        output_record,
        {
            "schema_version": "ace.retained_ckks.production-rnums/1.0.0",
            "generator": "tools/phantom_gpu/generate_retained_production_air.py",
            "ace_commit": ace_commit,
            "purpose": "extract-default-bootstrap-rotate-batch-rnums-only",
            "terminal_driver": None,
            "post_ckks_air_sha256": sha256_text(canonical),
            "production_rotation_batches": batches,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-air", required=True, type=Path)
    parser.add_argument("--output-record", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.output_air, arguments.output_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
