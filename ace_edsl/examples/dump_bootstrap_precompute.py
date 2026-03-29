import json
import os
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


def main(argv):
    slots = int(argv[1]) if len(argv) > 1 else _default_slots()
    level_budget = int(argv[2]) if len(argv) > 2 else 3

    summary = get_bootstrap_precompute_summary(slots, level_budget)
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bootstrap_precompute_dsl_{slots}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main(sys.argv)
