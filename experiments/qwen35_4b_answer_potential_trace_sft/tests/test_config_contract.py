from __future__ import annotations

import unittest
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


class ConfigContractTests(unittest.TestCase):
    def test_g0_threshold_keys_match_the_analyzer_contract(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        self.assertEqual(
            set(config["scorer_gate"]),
            {
                "within_task_auroc_min",
                "top1_uplift_min",
                "bootstrap_resamples",
                "bootstrap_seed",
                "kendall_tau_min",
                "premention_fraction_min",
            },
        )
