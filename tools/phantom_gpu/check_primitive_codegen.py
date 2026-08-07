#!/usr/bin/env python3
"""Audit a generated Phantom primitive CUDA translation unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


REQUIRED = (
    '#include "rt_phantom/rt_phantom.h"',
    "LIB_PHANTOM",
    "Add_ciph(",
    "Mul_ciph(",
    "Rotate_ciph(",
)

FORBIDDEN = {
    "ANT runtime header": re.compile(r"rt_ant/"),
    "SEAL runtime header": re.compile(r"rt_seal/"),
    "ANT provider": re.compile(r"\bLIB_ANT\b"),
    "SEAL provider": re.compile(r"\bLIB_SEAL\b"),
    "hardware-level call": re.compile(r"\bHw_[A-Za-z0-9_]*\s*\("),
    "POLY-level call": re.compile(r"\bPoly_[A-Za-z0-9_]*\s*\("),
    "opaque bootstrap call": re.compile(r"\bBootstrap\s*\("),
    "bootstrap evaluator": re.compile(r"Eval_bootstrap"),
    "bootstrap coefficient stage": re.compile(r"bootstrap_coeffs_to_slots"),
    "bootstrap evaluation stage": re.compile(r"bootstrap_eval_mod"),
    "bootstrap slot stage": re.compile(r"bootstrap_slots_to_coeffs"),
    "direct Phantom implementation": re.compile(r"\bphantom::"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_bytes = arguments.source.read_bytes()
    source = source_bytes.decode("utf-8")
    missing = [token for token in REQUIRED if token not in source]
    forbidden = [
        label for label, pattern in FORBIDDEN.items() if pattern.search(source)
    ]
    report = {
        "status": "pass" if not missing and not forbidden else "fail",
        "source": str(arguments.source),
        "size_bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "required_tokens": list(REQUIRED),
        "missing_required_tokens": missing,
        "forbidden_matches": forbidden,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
