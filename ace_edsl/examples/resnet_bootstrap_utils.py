#!/usr/bin/env python3
"""Utilities for integrating generated primitive bootstrap into resnet runs."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path


ROTATE_INIT = "  Init_ciph_same_scale(&_pgen_rot_res_2, &ciph_0, 0);\n"
ROTATE_FAST_PATH = (
    ROTATE_INIT
    + "  if (rot_idx_1 == 0) {\n"
    + "    Copy_ciphertext(&_pgen_rot_res_2, &ciph_0);\n"
    + "    RTLIB_TM_END(20, rtm);\n"
    + "    return _pgen_rot_res_2;\n"
    + "  }\n"
)
CTX_PAT = re.compile(
    r"(static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*"
    r"LIB_ANT\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*"
    r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*)"
    r"(\d+)\s*,\s*\n\s*\{\s*([^}]*)\s*\}",
    re.S,
)


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _bootstrap_module():
    script_dir = _script_dir()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import bootstrap_full

    return bootstrap_full


def _with_env(**updates):
    class _EnvCtx:
        def __enter__(self):
            self._old = {key: os.environ.get(key) for key in updates}
            for key, value in updates.items():
                os.environ[key] = str(value)

        def __exit__(self, exc_type, exc, tb):
            for key, value in self._old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return _EnvCtx()


def _extract_rotation_indices(bootstrap_c: str) -> list[int]:
    rot_idxs: set[int] = set()
    for pattern in (
        r"\bRotate\s*\([^,]+,\s*(-?\d+)\s*\)",
        r"\bRotate_ciph\s*\([^,]+,\s*[^,]+,\s*(-?\d+)\s*\)",
    ):
        for match in re.finditer(pattern, bootstrap_c):
            rot_idxs.add(int(match.group(1)))

    deg_match = re.search(
        r"static\s+CKKS_PARAMS\s+parm\s*=\s*\{\s*LIB_ANT\s*,\s*(\d+)",
        bootstrap_c,
        re.S,
    )
    if deg_match and "Conjugate_ciph(" in bootstrap_c:
        ring_degree = int(deg_match.group(1))
        rot_idxs.add(2 * ring_degree - 1)

    return sorted(v for v in rot_idxs if v != 0)


def _patch_zero_rotate_fast_path(source: str) -> str:
    return source.replace(ROTATE_INIT, ROTATE_FAST_PATH, 1)


def generate_demo(args: argparse.Namespace) -> int:
    with _with_env(
        ACE_BOOTSTRAP_IMPL=args.impl,
        ACE_BOOTSTRAP_POLY_DEGREE=args.poly_degree,
        ACE_BOOTSTRAP_MUL_LEVEL=args.mul_level,
    ):
        bootstrap_full = _bootstrap_module()
        importlib.reload(bootstrap_full)
        ok = bootstrap_full.run_demo()
    return 0 if ok else 1


def patch_resnet_context(args: argparse.Namespace) -> int:
    bootstrap_path = Path(args.bootstrap_c)
    resnet_inc_path = Path(args.resnet_inc)

    bootstrap_c = bootstrap_path.read_text(encoding="utf-8")
    resnet_inc = resnet_inc_path.read_text(encoding="utf-8")

    match = CTX_PAT.search(resnet_inc)
    if match is None:
        raise SystemExit("failed to locate CKKS_PARAMS in resnet include")

    existing = []
    for token in re.split(r"[,\s]+", match.group(3).strip()):
        if token:
            existing.append(int(token))

    all_rot_idxs = sorted(set(existing) | set(_extract_rotation_indices(bootstrap_c)))
    rot_list = ", ".join(str(v) for v in all_rot_idxs)
    patched = CTX_PAT.sub(
        lambda mm: f"{mm.group(1)}{len(all_rot_idxs)}, \n    {{ {rot_list} }}",
        resnet_inc,
        count=1,
    )
    patched = _patch_zero_rotate_fast_path(patched)
    resnet_inc_path.write_text(patched, encoding="utf-8")
    return 0


def emit_body(args: argparse.Namespace) -> int:
    src_path = Path(args.bootstrap_c)
    dst_path = Path(args.output)

    skip = False
    brace_depth = 0
    out_lines: list[str] = []

    def starts_wrapper(line: str) -> bool:
        return (
            line.startswith("CKKS_PARAMS* Get_context_params()")
            or line.startswith("RT_DATA_INFO* Get_rt_data_info()")
        )

    with src_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not skip and starts_wrapper(line):
                skip = True
                brace_depth = line.count("{") - line.count("}")
                continue
            if skip:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    skip = False
                continue
            line = re.sub(r"\bbootstrap_full\b", args.entry_name, line)
            line = re.sub(r"\bRotate\b", args.rotate_name, line)
            line = re.sub(r"\bRelinearize\b", args.relin_name, line)
            line = re.sub(r"\b(_cst_\d+)\b", rf"{args.const_prefix}\1", line)
            out_lines.append(line)

    body = _patch_zero_rotate_fast_path("".join(out_lines))
    dst_path.write_text(body, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    generate = subparsers.add_parser("generate-demo")
    generate.add_argument("--impl", default="primitive")
    generate.add_argument("--poly-degree", type=int, required=True)
    generate.add_argument("--mul-level", type=int, required=True)
    generate.set_defaults(func=generate_demo)

    patch = subparsers.add_parser("patch-resnet-context")
    patch.add_argument("--bootstrap-c", required=True)
    patch.add_argument("--resnet-inc", required=True)
    patch.set_defaults(func=patch_resnet_context)

    emit = subparsers.add_parser("emit-body")
    emit.add_argument("--bootstrap-c", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--entry-name", default="dsl_bootstrap_full")
    emit.add_argument("--rotate-name", default="dsl_bts_Rotate")
    emit.add_argument("--relin-name", default="dsl_bts_Relinearize")
    emit.add_argument("--const-prefix", default="dsl_bts")
    emit.set_defaults(func=emit_body)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
