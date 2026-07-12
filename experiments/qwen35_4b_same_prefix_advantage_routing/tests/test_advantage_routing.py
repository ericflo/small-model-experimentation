from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from advantage_routing import analyze_route_blocks, select_teacher  # noqa: E402


def row(
    index: int,
    block: int,
    teacher: str,
    *,
    audit_advantage: float = 0.01,
    alternate_advantage: float | None = None,
) -> dict:
    alternate_advantage = (
        audit_advantage if alternate_advantage is None else alternate_advantage
    )
    if teacher == "quick":
        selection = {"quick": [0.6] * 4, "deep": [0.4] * 4, "student": [0.5] * 4}
        audit = {
            "quick": [0.5 + audit_advantage] * 4,
            "deep": [0.5 + audit_advantage - alternate_advantage] * 4,
            "student": [0.5] * 4,
        }
    else:
        selection = {"quick": [0.4] * 4, "deep": [0.6] * 4, "student": [0.5] * 4}
        audit = {
            "quick": [0.5 + audit_advantage - alternate_advantage] * 4,
            "deep": [0.5 + audit_advantage] * 4,
            "student": [0.5] * 4,
        }
    return {
        "state_id": f"b{block}-{teacher}-{index}",
        "block": block,
        "family": f"family-{index % 2}",
        "kind": "atom" if index % 2 == 0 else "episode",
        "level": 1 + index % 3,
        "selection": selection,
        "audit": audit,
    }


def analyze(rows: list[dict], minimum: int = 2) -> dict:
    return analyze_route_blocks(
        rows,
        selection_branches=4,
        audit_branches=4,
        minimum_per_teacher_per_block=minimum,
        bootstrap_samples=300,
        confidence=0.95,
        bootstrap_seed=19,
    )


class AdvantageRoutingTests(unittest.TestCase):
    def test_strict_winner_routes_and_ties_abstain(self):
        self.assertEqual(
            select_teacher({"quick": [0.5001], "deep": [0.5], "student": [0.5]}),
            "quick",
        )
        self.assertIsNone(
            select_teacher({"quick": [0.5], "deep": [0.5], "student": [0.4]})
        )
        self.assertIsNone(
            select_teacher({"quick": [0.4], "deep": [0.3], "student": [0.4]})
        )

    def test_arbitrarily_small_replicated_positive_effect_can_pass(self):
        rows = []
        for block in (0, 1):
            rows.extend(row(index, block, "quick", audit_advantage=1e-6) for index in range(4))
            rows.extend(row(index, block, "deep", audit_advantage=1e-6) for index in range(4))
        result = analyze(rows)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["downstream_authorization"], "locality_pilot")

    def test_selection_winners_curse_cannot_pass_zero_audit(self):
        rows = []
        for block in (0, 1):
            rows.extend(row(index, block, "quick", audit_advantage=0.0) for index in range(4))
            rows.extend(row(index, block, "deep", audit_advantage=0.0) for index in range(4))
        result = analyze(rows)
        self.assertFalse(result["gate"]["passed"])
        self.assertEqual(result["downstream_authorization"], "stop_before_mopd")

    def test_one_teacher_only_fails_composition_support(self):
        rows = [row(index, block, "deep") for block in (0, 1) for index in range(8)]
        result = analyze(rows)
        self.assertFalse(result["gate"]["passed"])
        self.assertFalse(result["by_teacher"]["quick"]["support_passed"])

    def test_negative_second_block_fails_replication(self):
        rows = []
        for block in (0, 1):
            advantage = 0.02 if block == 0 else -0.01
            rows.extend(row(index, block, "quick", audit_advantage=advantage) for index in range(4))
            rows.extend(row(index, block, "deep", audit_advantage=advantage) for index in range(4))
        result = analyze(rows)
        self.assertFalse(result["gate"]["passed"])
        self.assertFalse(
            result["by_teacher"]["quick"]["by_contrast"]["student"][
                "all_block_means_positive"
            ]
        )

    def test_branch_count_mismatch_fails_closed(self):
        invalid = row(0, 0, "quick")
        invalid["audit"]["quick"] = [1.0]
        with self.assertRaisesRegex(ValueError, "audit branch count"):
            analyze([invalid, row(1, 1, "quick")], minimum=1)


if __name__ == "__main__":
    unittest.main()

