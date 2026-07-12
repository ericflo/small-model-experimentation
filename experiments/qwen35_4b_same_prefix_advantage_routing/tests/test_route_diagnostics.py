from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_route_diagnostics import (  # noqa: E402
    _cell_macro,
    _cross_block_group_router,
    _pearson,
    _split_label_stability,
    _strict_teacher_choice,
)


class RouteDiagnosticTests(unittest.TestCase):
    def test_pearson_detects_linear_and_constant_cases(self):
        self.assertAlmostEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(_pearson([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertIsNone(_pearson([1, 1, 1], [2, 3, 4]))

    def test_cell_macro_does_not_micro_weight_dense_cell(self):
        rows = [
            {"family": "a", "kind": "atom", "level": 1, "value": 1.0},
            {"family": "a", "kind": "atom", "level": 1, "value": 1.0},
            {"family": "b", "kind": "atom", "level": 1, "value": 0.0},
        ]
        self.assertEqual(_cell_macro(rows, lambda row: row["value"]), 0.5)

    def test_strict_teacher_choice_abstains_for_student_or_tie(self):
        self.assertEqual(
            _strict_teacher_choice({"quick": 0.3, "deep": 0.2, "student": 0.1}),
            "quick",
        )
        self.assertIsNone(
            _strict_teacher_choice({"quick": 0.3, "deep": 0.3, "student": 0.1})
        )
        self.assertIsNone(
            _strict_teacher_choice({"quick": 0.1, "deep": 0.2, "student": 0.3})
        )

    def test_cross_block_group_router_never_fits_on_evaluation_rows(self):
        def row(block, state, selection, audit):
            return {
                "block": block,
                "state_id": state,
                "family": "family",
                "kind": "atom",
                "level": 1,
                "selection_means": selection,
                "audit_means": audit,
            }

        rows = [
            row(
                0,
                "fit-quick",
                {"quick": 2.0, "deep": 0.0, "student": 1.0},
                {"quick": 2.0, "deep": 0.0, "student": 1.0},
            ),
            row(
                1,
                "fit-deep",
                {"quick": 0.0, "deep": 2.0, "student": 1.0},
                {"quick": 0.0, "deep": 2.0, "student": 1.0},
            ),
        ]
        result = _cross_block_group_router(
            rows, ("family", "kind"), require_half_agreement=True
        )
        forward = result["directions"]["fit_0_evaluate_1"]
        reverse = result["directions"]["fit_1_evaluate_0"]
        self.assertEqual(forward["teacher_counts"], {"quick": 1})
        self.assertEqual(forward["contrasts"]["student"]["state_mean"], -1.0)
        self.assertEqual(reverse["teacher_counts"], {"deep": 1})
        self.assertEqual(reverse["contrasts"]["student"]["state_mean"], -1.0)

    def test_split_label_stability_counts_abstention_and_teacher_switch(self):
        rows = [
            {
                "state_id": "same",
                "selection": {"quick": [1.0], "deep": [0.0], "student": [0.0]},
                "audit": {"quick": [1.0], "deep": [0.0], "student": [0.0]},
            },
            {
                "state_id": "switch",
                "selection": {"quick": [1.0], "deep": [0.0], "student": [0.0]},
                "audit": {"quick": [0.0], "deep": [1.0], "student": [0.0]},
            },
            {
                "state_id": "abstain",
                "selection": {"quick": [0.0], "deep": [0.0], "student": [0.0]},
                "audit": {"quick": [0.0], "deep": [0.0], "student": [0.0]},
            },
        ]
        result = _split_label_stability(rows)
        self.assertAlmostEqual(result["exact_three_way_agreement"], 2 / 3)
        self.assertEqual(
            result["confusion_first_half_to_second_half"]["quick"]["deep"], 1
        )
        self.assertEqual(result["by_teacher"]["quick"]["first_half_routes"], 2)
        self.assertEqual(
            result["by_teacher"]["quick"][
                "first_half_precision_for_same_second_half_teacher"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
