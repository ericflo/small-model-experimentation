from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from selector import (  # noqa: E402
    PRIMARY_NAME,
    deployable_predictions,
    oracle_mismatch_predictions,
    paired_bootstrap_lower,
)


ALIASES = ["a", "b", "c"]


def _task(real, shuffled, gold="a"):
    return {"real": real, "shuffled": shuffled, "gold": gold}


def test_primary_is_mean_probability_delta_not_mean_probability():
    grouped = {
        "t": _task(
            [[0.45, 0.40, 0.15], [0.45, 0.40, 0.15], [0.45, 0.40, 0.15]],
            [[0.44, 0.10, 0.46], [0.44, 0.10, 0.46], [0.44, 0.10, 0.46]],
        )
    }
    predictions = deployable_predictions(grouped, ALIASES)["t"]
    assert predictions[PRIMARY_NAME] == "b"
    assert predictions["mean_real_probability"] == "a"


def test_deployable_predictions_ignore_hidden_label_mutation():
    grouped = {
        "t": _task(
            [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]],
            [[0.2, 0.5, 0.3], [0.1, 0.4, 0.5], [0.1, 0.3, 0.6]],
            gold="a",
        )
    }
    mutated = copy.deepcopy(grouped)
    mutated["t"]["gold"] = "c"
    assert deployable_predictions(grouped, ALIASES) == deployable_predictions(
        mutated, ALIASES
    )


def test_majority_tie_breaks_by_mean_probability_then_alias_order():
    grouped = {
        "t": _task(
            [[0.55, 0.35, 0.10], [0.20, 0.60, 0.20], [0.20, 0.25, 0.55]],
            [[0.34, 0.33, 0.33]] * 3,
        )
    }
    predictions = deployable_predictions(grouped, ALIASES)["t"]
    assert predictions["majority_real"] == "b"


def test_oracle_mismatch_cycles_only_within_gold_stratum():
    grouped = {
        "a0": _task([[0.8, 0.1, 0.1]] * 3, [[0.7, 0.2, 0.1]] * 3, "a"),
        "a1": _task([[0.8, 0.1, 0.1]] * 3, [[0.6, 0.3, 0.1]] * 3, "a"),
        "b0": _task([[0.1, 0.8, 0.1]] * 3, [[0.2, 0.7, 0.1]] * 3, "b"),
        "b1": _task([[0.1, 0.8, 0.1]] * 3, [[0.3, 0.6, 0.1]] * 3, "b"),
    }
    _predictions, donors = oracle_mismatch_predictions(grouped, ALIASES)
    assert donors == {"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"}


def test_bootstrap_is_deterministic():
    differences = [1.0, 0.0, -1.0, 1.0, 0.0]
    first = paired_bootstrap_lower(differences, seed=7, resamples=1000)
    second = paired_bootstrap_lower(differences, seed=7, resamples=1000)
    assert first == second


def test_confirmation_artifacts_are_absent_at_design_boundary():
    confirmation = EXP / "data" / "confirmation"
    assert not (confirmation / "real.jsonl").exists()
    assert not (confirmation / "shuffled.jsonl").exists()


def test_confirmation_runner_fails_closed_before_loading_rows():
    result = subprocess.run(
        [
            sys.executable,
            str(EXP / "scripts" / "run.py"),
            "--stage",
            "confirmation",
        ],
        cwd=EXP.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "confirmation requires qualification and boundary receipts" in result.stderr
