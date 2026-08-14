from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"
sys.path.insert(0, str(TOOLS_ROOT))

import evalmod_scale_diagnostic as diagnostic  # noqa: E402


EXPECTED_ANT_Q_N65536 = [
    1152921504606584833,
    72057594069123073,
    72057594004111361,
    72057594065453057,
    72057594006863873,
    72057594063093761,
    72057594011189249,
    72057594062831617,
    72057594020626433,
    72057594062438401,
    72057594021150721,
    72057594061651969,
    72057594023903233,
    72057594058899457,
    72057594027704321,
    72057594058375169,
    72057594029015041,
    72057594057195521,
    72057594030981121,
    72057594047889409,
    72057594034913281,
    72057594042646529,
    72057594035306497,
    72057594040680449,
    72057594036879361,
    72057594038321153,
]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return diagnostic.build_report()


def test_ant_balanced_n65536_q_is_exact() -> None:
    config = diagnostic.DiagnosticConfig()
    values = diagnostic.ant_balanced_q(config)
    assert values == EXPECTED_ANT_Q_N65536
    assert len(values) == len(set(values)) == 26
    assert all(value % (2 * config.poly_degree) == 1 for value in values)
    assert all(diagnostic._is_prime(value) for value in values)

    # ANT calls this a 56-bit prime because 2^56 is its nearest power of two;
    # ordinary integer bit_length() is not the manifest size convention.
    assert values[1].bit_length() == 57
    assert values[1] - 2**56 < 2**57 - values[1]


def test_current_model_reproduces_n65536_failure(
    report: dict[str, object],
) -> None:
    models = report["models"]
    current = models["phantom_current"]["summary"]
    final = current["final_evalmod"]

    assert final["raw_scale"] == 72057600340394256.0
    assert final["dropped_q_count"] == 12
    assert final["value"] == pytest.approx(7.309101529573425e-8, rel=0.0, abs=1.0e-22)

    assert current["final_chebyshev"]["value"] == pytest.approx(
        0.7794736814947817, rel=0.0, abs=1.0e-15
    )
    assert [entry["value"] for entry in current["double_angle"]] == pytest.approx(
        [0.5835396761407632, 0.28209484421248976, 7.309101529573425e-8],
        rel=0.0,
        abs=1.0e-15,
    )


def test_first_value_divergence_is_t2_nominal_scalar_add(
    report: dict[str, object],
) -> None:
    models = report["models"]
    current = models["phantom_current"]["summary"]["t2_first_scalar"]
    corrected = models["phantom_corrected"]["summary"]["t2_first_scalar"]
    oracle = models["ant_nominal_oracle"]["summary"]["t2_first_scalar"]

    assert current["pre_scalar"]["value"] == 0.0
    assert current["pre_scalar"]["raw_scale"] == 72057594058244096.0
    assert current["encoded_scale"] == 72057594037927936.0
    assert current["post_scalar"]["value"] == pytest.approx(
        -0.9999999997180566, rel=0.0, abs=1.0e-16
    )
    assert corrected["encoded_scale"] == corrected["pre_scalar"]["raw_scale"]
    assert corrected["post_scalar"]["value"] == -1.0
    assert oracle["post_scalar"]["value"] == -1.0

    first = report["diagnosis"]["first_material_value_divergence"]
    assert first["label"] == "baby.T2.add_minus_one.add"
    assert first["operation"] == "scalar_add"
    assert (
        report["comparisons"]["current_vs_ant_nominal"]["first_raw_scale_divergence"][
            "label"
        ]
        == "evalmod.input"
    )


def test_corrected_and_nominal_models_cover_requested_boundaries(
    report: dict[str, object],
) -> None:
    models = report["models"]
    corrected = models["phantom_corrected"]["summary"]
    oracle = models["ant_nominal_oracle"]["summary"]

    assert corrected["final_evalmod"]["raw_scale"] == 72057593154764800.0
    assert corrected["final_evalmod"]["value"] == pytest.approx(
        -2.516430203103326e-9, rel=0.0, abs=1.0e-22
    )
    assert oracle["final_evalmod"]["raw_scale"] == 72057594037927936.0
    assert abs(oracle["final_evalmod"]["value"]) < 1.0e-12

    labels = {entry["label"] for entry in models["phantom_current"]["checkpoints"]}
    assert {f"baby.T{index}" for index in range(1, 9)} <= labels
    assert {
        "ps.root.q.combine",
        "ps.root.s.combine",
        "ps.root.combine",
        "chebyshev.final",
        "double_angle.1",
        "double_angle.2",
        "double_angle.3",
    } <= labels

    boundary_operations = {
        entry["operation"] for entry in report["boundary_comparison"]
    }
    assert boundary_operations == {
        "scalar_encode",
        "scalar_add",
        "rescale",
        "mod_switch",
    }
    for model in models.values():
        summary = model["summary"]
        assert summary["scalar_encode_count"] == 16
        assert summary["rescale_count"] == 65
        assert summary["mod_switch_count"] == 49


def test_cli_emits_deterministic_canonical_json() -> None:
    command = [
        sys.executable,
        str(TOOLS_ROOT / "evalmod_scale_diagnostic.py"),
        "--compact",
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    parsed = json.loads(first.stdout)
    assert parsed["schema_version"] == diagnostic.SCHEMA_VERSION
    assert parsed["status"] == "pass"
