import json
import os
import subprocess
import sys
from pathlib import Path


def _setup_sys_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(repo_root, ".."))
    for path in (repo_root, parent_root):
        if path not in sys.path:
            sys.path.insert(0, path)


_setup_sys_path()

from ace_edsl.edsl.core.bootstrap_decomposition import (  # noqa: E402
    get_bootstrap_precompute_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
OUT_DIR = Path(__file__).resolve().parent / "output"
HELPER_SRC = TESTS_DIR / "rtlib_coeff_collapse_dump.c"
HELPER_BIN = OUT_DIR / "rtlib_coeff_collapse_dump.bin"


def _default_slots() -> int:
    raw = os.environ.get("ACE_BOOTSTRAP_POLY_DEGREE", "").strip()
    if raw:
        try:
            degree = int(raw)
            if degree > 0:
                return degree // 2
        except ValueError:
            pass
    return 8192


def _build_helper():
    cmd = [
        "gcc",
        "-O2",
        "-std=gnu11",
        "-I",
        "/usr/local/rtlib/include",
        "-I",
        "/usr/local/rtlib/include/ant",
        str(HELPER_SRC),
        "/usr/local/rtlib/lib/libFHErt_ant.a",
        "/usr/local/rtlib/lib/libFHErt_common.a",
        "-lgmp",
        "-lm",
        "-lgomp",
        "-o",
        str(HELPER_BIN),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _load_rtlib_summary(slots: int, level_budget: int):
    _build_helper()
    result = subprocess.run(
        [str(HELPER_BIN), str(slots), str(level_budget)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return json.loads(result.stdout)


def _compare_section(name: str, lhs, rhs):
    mismatches = []
    if lhs["rows"] and rhs["rows"] and len(lhs["rows"]) != len(rhs["rows"]):
        mismatches.append(f"{name}: row_count {len(lhs['rows'])} != {len(rhs['rows'])}")
        return mismatches
    for lrow, rrow in zip(lhs["rows"], rhs["rows"]):
        if lrow["digest"] != rrow["digest"]:
            mismatches.append(
                f"{name}: row {lrow['row']} digest {lrow['digest']} != {rrow['digest']}"
            )
    return mismatches


def compare_summaries(dsl, rtlib):
    mismatches = []
    if dsl["slots"] != rtlib["slots"]:
        mismatches.append(f"slots {dsl['slots']} != {rtlib['slots']}")
    if dsl["level_budget"] != rtlib["level_budget"]:
        mismatches.append(
            f"level_budget {dsl['level_budget']} != {rtlib['level_budget']}"
        )
    if dsl["encode_params"] != rtlib["encode_params"]:
        mismatches.append("encode_params mismatch")
    if dsl["decode_params"] != rtlib["decode_params"]:
        mismatches.append("decode_params mismatch")
    for lstage, rstage in zip(
        dsl["encode_coeff_collapse"], rtlib["encode_coeff_collapse"]
    ):
        mismatches.extend(_compare_section(f"encode stage {lstage['stage']}", lstage, rstage))
    for lstage, rstage in zip(
        dsl["decode_coeff_collapse"], rtlib["decode_coeff_collapse"]
    ):
        mismatches.extend(_compare_section(f"decode stage {lstage['stage']}", lstage, rstage))
    return mismatches


def main(argv):
    slots = int(argv[1]) if len(argv) > 1 else _default_slots()
    level_budget = int(argv[2]) if len(argv) > 2 else 3

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dsl = get_bootstrap_precompute_summary(slots, level_budget)
    rtlib = _load_rtlib_summary(slots, level_budget)

    dsl_path = OUT_DIR / f"bootstrap_precompute_dsl_{slots}.json"
    rtlib_path = OUT_DIR / f"bootstrap_precompute_rtlib_{slots}.json"
    cmp_path = OUT_DIR / f"bootstrap_precompute_compare_{slots}.json"
    dsl_path.write_text(json.dumps(dsl, indent=2), encoding="utf-8")
    rtlib_path.write_text(json.dumps(rtlib, indent=2), encoding="utf-8")

    mismatches = compare_summaries(dsl, rtlib)
    cmp = {
        "slots": slots,
        "level_budget": level_budget,
        "dsl": str(dsl_path),
        "rtlib": str(rtlib_path),
        "match": not mismatches,
        "mismatches": mismatches,
    }
    cmp_path.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(cmp_path)
    print("MATCH" if not mismatches else "MISMATCH")
    if mismatches:
        for item in mismatches[:20]:
            print(item)


if __name__ == "__main__":
    main(sys.argv)
