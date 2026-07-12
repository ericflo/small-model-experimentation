from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stats import paired_bootstrap_mean_ci  # noqa: E402


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    differences = [1.0] * 8 + [0.0] * 2
    first = paired_bootstrap_mean_ci(differences, resamples=1000, seed=9)
    second = paired_bootstrap_mean_ci(differences, resamples=1000, seed=9)
    assert first == second
    assert first["mean"] == 0.8
    assert 0.0 < first["lower"] <= first["mean"] <= first["upper"] <= 1.0
