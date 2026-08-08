from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = (
    REPO_ROOT
    / "tools"
    / "phantom_gpu"
    / "generate_ordinary_runtime_symbols.py"
)
GPU_RUNNER_PATH = (
    REPO_ROOT
    / "tools"
    / "phantom_gpu"
    / "harness"
    / "ordinary_ckks_gpu_runner.cu"
)

BEGIN_MARKER = "// ORDINARY_RUNTIME_CALLS_BEGIN"
END_MARKER = "// ORDINARY_RUNTIME_CALLS_END"
PRESERVATION_BEGIN_MARKER = "// EXACT_SOURCE_PRESERVATION_BEGIN"
PRESERVATION_END_MARKER = "// EXACT_SOURCE_PRESERVATION_END"


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: tuple[str, ...]


EXPECTED_CALLS = (
    ParsedCall("Degree", ()),
    ParsedCall("Encode_float", ("&plain_float", "float_values", "2", "1.0", "1")),
    ParsedCall("Encode_double", ("&plain_double", "double_values", "2", "1.0", "1")),
    ParsedCall("Encode_dcmplx", ("&plain_complex", "complex_values", "2", "1.0", "1")),
    ParsedCall("Encode_float_mask", ("&plain_float_mask", "0.5F", "slots", "1.0", "1")),
    ParsedCall("Encode_double_mask", ("&plain_double_mask", "-0.75", "slots", "1.0", "1")),
    ParsedCall("Add_ciph", ("&destination", "&left", "&right")),
    ParsedCall("Add_ciph", ("&left", "&left", "&right")),
    ParsedCall("Add_ciph", ("&right", "&left", "&right")),
    ParsedCall("Sub_ciph", ("&destination", "&left", "&right")),
    ParsedCall("Sub_ciph", ("&left", "&left", "&right")),
    ParsedCall("Sub_ciph", ("&right", "&left", "&right")),
    ParsedCall("Mul_ciph", ("&product", "&left", "&right")),
    ParsedCall("Mul_ciph", ("&left", "&left", "&right")),
    ParsedCall("Mul_ciph", ("&right", "&left", "&right")),
    ParsedCall("Add_plain", ("&destination", "&left", "&plain_float")),
    ParsedCall("Add_plain", ("&left", "&left", "&plain_float")),
    ParsedCall("Sub_plain", ("&destination", "&left", "&plain_double")),
    ParsedCall("Sub_plain", ("&left", "&left", "&plain_double")),
    ParsedCall("Mul_plain", ("&destination", "&left", "&plain_complex")),
    ParsedCall("Mul_plain", ("&left", "&left", "&plain_complex")),
    ParsedCall("Add_scalar", ("&destination", "&left", "1.25")),
    ParsedCall("Add_scalar", ("&left", "&left", "1.25")),
    ParsedCall("Sub_scalar", ("&destination", "&left", "-0.5")),
    ParsedCall("Sub_scalar", ("&left", "&left", "-0.5")),
    ParsedCall("Mul_scalar", ("&destination", "&left", "2.0")),
    ParsedCall("Mul_scalar", ("&left", "&left", "2.0")),
    ParsedCall("Relin", ("&destination", "&product")),
    ParsedCall("Relin", ("&product", "&product")),
    ParsedCall("Rescale_ciph", ("&destination", "&left")),
    ParsedCall("Rescale_ciph", ("&left", "&left")),
    ParsedCall("Mod_switch", ("&destination", "&left")),
    ParsedCall("Mod_switch", ("&left", "&left")),
    ParsedCall("Rotate_ciph", ("&destination", "&left", "1")),
    ParsedCall("Rotate_ciph", ("&left", "&left", "1")),
    ParsedCall("Rotate_ciph", ("&destination", "&left", "-1")),
    ParsedCall("Rotate_ciph", ("&left", "&left", "-1")),
    ParsedCall("Rotate_ciph", ("&destination", "&left", "0")),
    ParsedCall("Rotate_ciph", ("&left", "&left", "0")),
    ParsedCall("Copy_ciph", ("&destination", "&left")),
    ParsedCall("Copy_ciph", ("&left", "&left")),
    ParsedCall("Zero_ciph", ("&destination",)),
    ParsedCall("Sc_degree", ("&left",)),
    ParsedCall("Level", ("&left",)),
    ParsedCall("Active_q_count", ("&left",)),
    ParsedCall("Chain_index", ("&left",)),
    ParsedCall("Raw_scale", ("&left",)),
    ParsedCall("Get_ciph_slots", ("&left",)),
    ParsedCall("Get_ciph_size", ("&left",)),
    ParsedCall("Is_ciph_ntt", ("&left",)),
    ParsedCall("Get_plain_level", ("&plain_float",)),
    ParsedCall("Get_plain_active_q_count", ("&plain_float",)),
    ParsedCall("Get_plain_chain_index", ("&plain_float",)),
    ParsedCall("Get_plain_raw_scale", ("&plain_float",)),
    ParsedCall("Get_plain_scale_degree", ("&plain_float",)),
    ParsedCall("Get_plain_slots", ("&plain_float",)),
    ParsedCall("Is_plain_ntt", ("&plain_float",)),
    ParsedCall("Set_output_data", ('"ordinary_output"', "0", "&destination")),
    ParsedCall("Free_ciph", ("&left",)),
    ParsedCall("Free_ciph", ("&right",)),
    ParsedCall("Free_ciph", ("&destination",)),
    ParsedCall("Free_ciph", ("&product",)),
    ParsedCall("Free_plain", ("&plain_float",)),
    ParsedCall("Free_plain", ("&plain_double",)),
    ParsedCall("Free_plain", ("&plain_complex",)),
    ParsedCall("Free_plain", ("&plain_float_mask",)),
    ParsedCall("Free_plain", ("&plain_double_mask",)),
    ParsedCall("Free_ciph_array", ("cipher_array", "2")),
)

REQUIRED_CALL_NAMES = frozenset(call.name for call in EXPECTED_CALLS) | {
    "Get_input_data"
}
ALLOWED_CALL_NAMES = REQUIRED_CALL_NAMES
ALLOWED_SOURCE_NAMES = ALLOWED_CALL_NAMES | {
    "Ordinary_runtime_symbol_probe",
}

EXPECTED_INPUT_CALLS = Counter(
    {
        ParsedCall("Get_input_data", ('"ordinary_input"', "0")): 1,
        ParsedCall("Get_input_data", ('"ordinary_input"', "1")): 1,
    }
)

FORBIDDEN_PATTERNS = {
    "retained operation": re.compile(
        r"\b(?:Raise_mod|Conjugate_ciph|Rotate_batch_ciph|Mul_mono_ciph)\b"
    ),
    "refresh operation": re.compile(r"(?:bootstrap|Need_bts)", re.IGNORECASE),
    "transform stage": re.compile(
        r"(?:coeffs?_to_slots|slots_to_coeffs?|CoeffToSlots|SlotToCoeffs|EvalMod|"
        r"\bstage[A-Za-z0-9_]*\s*\()",
        re.IGNORECASE,
    ),
    "lower-level provider": re.compile(
        r"(?:\bLIB_ANT\b|\brt_ant\b|\bANT\b|\bPOLY\b|"
        r"\b(?:ant|poly)[A-Za-z0-9_]*\s*\(|\bHw_)",
        re.IGNORECASE,
    ),
    "direct provider call": re.compile(r"\bPhantom_[A-Za-z0-9_]*\s*\("),
    "direct provider header": re.compile(r"#\s*include\s*[<\"]phantom\.h[>\"]"),
    "legacy context source": re.compile(
        r"(?:Get_context_params|Get_phantom_ordinary_features|CKKS_PARAMS|"
        r"\bprofile::)"
    ),
}

INTERNAL_LABEL_PATTERN = re.compile(
    r"\bm\d+(?:_[A-Za-z0-9_]+)?\b", re.IGNORECASE
)


def _mask_comments(source: str) -> str:
    result = list(source)
    index = 0
    while index < len(result):
        if source.startswith("//", index):
            end = source.find("\n", index)
            end = len(source) if end == -1 else end
            result[index:end] = " " * (end - index)
            index = end
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end == -1:
                raise AssertionError("unterminated block comment")
            end += 2
            for position in range(index, end):
                if result[position] != "\n":
                    result[position] = " "
            index = end
        else:
            index += 1
    return "".join(result)


def _split_arguments(arguments: str) -> tuple[str, ...]:
    if not arguments.strip():
        return ()
    pieces: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(arguments):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            pieces.append(arguments[start:index])
            start = index + 1
    pieces.append(arguments[start:])
    return tuple(re.sub(r"\s+", "", piece) for piece in pieces)


def _parse_calls(source: str) -> tuple[ParsedCall, ...]:
    masked = _mask_comments(source)
    calls: list[ParsedCall] = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", masked):
        name = match.group(1)
        open_parenthesis = masked.find("(", match.start())
        depth = 1
        index = open_parenthesis + 1
        in_string = False
        escaped = False
        while index < len(masked) and depth:
            character = masked[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            index += 1
        if depth:
            raise AssertionError(f"unterminated call to {name}")
        calls.append(
            ParsedCall(
                name,
                _split_arguments(masked[open_parenthesis + 1 : index - 1]),
            )
        )
    return tuple(calls)


def _call_region(source: str) -> str:
    assert source.count(BEGIN_MARKER) == 1
    assert source.count(END_MARKER) == 1
    start = source.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = source.index(END_MARKER)
    assert start < end
    return source[start:end]


def audit_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    region = _call_region(source)
    region_calls = Counter(_parse_calls(region))
    input_prefix = source[: source.index(BEGIN_MARKER)]
    input_calls = Counter(
        call for call in _parse_calls(input_prefix) if call.name == "Get_input_data"
    )
    observed_names = frozenset(call.name for call in region_calls) | frozenset(
        call.name for call in input_calls
    )
    source_names = frozenset(call.name for call in _parse_calls(source))

    assert observed_names <= ALLOWED_CALL_NAMES, sorted(
        observed_names - ALLOWED_CALL_NAMES
    )
    assert REQUIRED_CALL_NAMES <= observed_names, sorted(
        REQUIRED_CALL_NAMES - observed_names
    )
    assert region_calls == Counter(EXPECTED_CALLS)
    assert input_calls == EXPECTED_INPUT_CALLS
    assert source_names <= ALLOWED_SOURCE_NAMES, sorted(
        source_names - ALLOWED_SOURCE_NAMES
    )

    assert re.search(
        r"const\s+std::size_t\s+slots\s*=\s*degree\s*/\s*2\s*;", region
    )
    forbidden = [
        label for label, pattern in FORBIDDEN_PATTERNS.items() if pattern.search(source)
    ]
    assert not forbidden, forbidden
    assert not INTERNAL_LABEL_PATTERN.search(path.name)
    assert not INTERNAL_LABEL_PATTERN.search(source)


def audit_gpu_runner_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert source.count(PRESERVATION_BEGIN_MARKER) == 1
    assert source.count(PRESERVATION_END_MARKER) == 1
    start = source.index(PRESERVATION_BEGIN_MARKER) + len(
        PRESERVATION_BEGIN_MARKER
    )
    end = source.index(PRESERVATION_END_MARKER)
    assert start < end
    preservation = source[start:end]

    required_exact_contract = (
        "CheckedElementProduct",
        "std::numeric_limits<std::size_t>::max() / left",
        "cudaDeviceSynchronize()",
        "cudaMemcpy(",
        "cudaMemcpyDeviceToHost",
        "CipherMetadata(value)",
        "PlainMetadata(value)",
        "ExactDoubleBits(value->scale())",
        "value->parms_id()",
        "value->GetNoiseScaleDeg()",
        "value->is_asymmetric()",
        "before._coefficients",
        "if (destination != left)",
        "if (destination != right)",
        "if (destination != source)",
        "RequirePlainPreserved(right_before, right",
    )
    for required in required_exact_contract:
        assert required in preservation, required
    for forbidden in ("cudaMemcpyAsync", "MaximumError", "tolerance", "fabs"):
        assert forbidden not in preservation, forbidden

    guarded_calls = {
        "InvokeCipherBinaryPreservingOperands": (
            "Add_ciph",
            "Sub_ciph",
            "Mul_ciph",
        ),
        "InvokeCipherPlainPreservingOperands": (
            "Add_plain",
            "Sub_plain",
            "Mul_plain",
        ),
        "InvokeCipherUnaryPreservingSource": (
            "Add_scalar",
            "Sub_scalar",
            "Mul_scalar",
            "Copy_ciph",
            "Mod_switch",
            "Relin",
            "Rescale_ciph",
            "Rotate_ciph",
        ),
    }
    for guard, calls in guarded_calls.items():
        for call in calls:
            pattern = re.compile(
                rf"\b{guard}\s*\(.{{0,320}}?\[&\]\s*\{{\s*{call}\s*\(",
                re.DOTALL,
            )
            assert pattern.search(source), f"{call} is not guarded by {guard}"


def test_ordinary_runtime_source_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "ordinary_runtime_symbols.cu"
    subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output", str(source_path)],
        check=True,
    )
    audit_source(source_path)


def test_gpu_runner_exact_source_preservation_contract() -> None:
    audit_gpu_runner_source(GPU_RUNNER_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--gpu-runner-source", type=Path, default=GPU_RUNNER_PATH)
    arguments = parser.parse_args()
    audit_source(arguments.source)
    audit_gpu_runner_source(arguments.gpu_runner_source)
    print(
        "ordinary runtime source audits passed: "
        f"{arguments.source}, {arguments.gpu_runner_source}"
    )
