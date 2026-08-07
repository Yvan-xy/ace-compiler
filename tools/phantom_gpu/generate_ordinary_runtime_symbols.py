#!/usr/bin/env python3
"""Generate the ordinary Phantom runtime symbol and alias conformance unit."""

from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = (
    SCRIPT_DIR / "harness" / "ordinary_ckks_runtime_symbols.cu.in"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser.parse_args()


def generate(output: Path, template: Path = DEFAULT_TEMPLATE) -> None:
    if output.suffix != ".cu":
        raise ValueError("output must use the .cu suffix")
    if template.suffixes[-2:] != [".cu", ".in"]:
        raise ValueError("template must use the .cu.in suffix")
    source = template.read_text(encoding="utf-8")
    if not source.endswith("\n"):
        raise ValueError("template must end with a newline")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()
    generate(arguments.output, arguments.template)
    print(f"generated {arguments.output} ({arguments.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
