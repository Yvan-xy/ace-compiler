from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
PHANTOM_ROOT = REPO_ROOT / "fhe-cmplr" / "rtlib" / "phantom"
PHANTOM_CMAKE = PHANTOM_ROOT / "CMakeLists.txt"
LEGACY_EXAMPLES = {
    "eg_rtphantom_add.inc": ("CKKS_PARAMS", "Get_context_params"),
    "eg_rtphantom_boot.inc": (
        "CKKS_PARAMS",
        "Get_context_params",
        "Bootstrap(",
    ),
    "relu.inc": ("CKKS_PARAMS", "Get_context_params"),
}


def _active_cmake(source: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


def test_phantom_cmake_has_no_active_legacy_example_wiring() -> None:
    cmake = _active_cmake(PHANTOM_CMAKE.read_text(encoding="utf-8"))

    assert re.search(r"\bfile\s*\(\s*GLOB(?:_RECURSE)?[^\n]*example", cmake) is None
    assert re.search(r"\badd_executable\s*\(", cmake) is None
    assert re.search(r"\badd_custom_target\s*\([^\n]*example", cmake) is None
    assert re.search(r"\binstall\s*\([^\n]*DESTINATION\s+example\b", cmake) is None
    assert "FHERT_PHANTOM_EXAMPLE_SRC_FILES" not in cmake
    assert "FHERT_PHANTOM_EGAPP" not in cmake
    assert "fhert_phantom_example" not in cmake
    assert re.search(
        r"\binstall\s*\(\s*TARGETS\s+\$\{FHERT_PHANTOM_UTAPP\}\s+"
        r"RUNTIME\s+DESTINATION\s+unittest\s*\)",
        cmake,
    ) is not None


def test_stale_legacy_includes_are_preserved_but_not_referenced_by_cmake() -> None:
    cmake = _active_cmake(PHANTOM_CMAKE.read_text(encoding="utf-8"))

    for filename, legacy_tokens in LEGACY_EXAMPLES.items():
        source_path = PHANTOM_ROOT / "example" / filename
        source = source_path.read_text(encoding="utf-8")

        assert filename not in cmake
        for token in legacy_tokens:
            assert token in source
