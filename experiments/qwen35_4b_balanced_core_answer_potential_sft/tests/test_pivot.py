from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pivot import choose_pivot, natural_checkpoint_indices  # noqa: E402


class FakeTokenizer:
    IDS = {"\n": [198], "\n\n": [271], ".": [13], "?": [30], "!": [0], ";": [26], "。": [1710], "！": [9991], "？": [9992]}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return self.IDS[text]


class PivotTests(unittest.TestCase):
    def test_checkpoints_are_natural_spread_and_terminal(self) -> None:
        tokens = []
        for index in range(100):
            tokens.extend([1000 + index, 13])
        got = natural_checkpoint_indices(FakeTokenizer(), tokens, max_checkpoints=8)
        self.assertEqual(len(got), 8)
        self.assertEqual(got[-1], len(tokens))
        self.assertTrue(all(tokens[index - 1] == 13 for index in got))

    def test_pivot_precedes_largest_jump_or_falls_back(self) -> None:
        rows = [
            {"token_index": 100, "joint_gain_per_answer_token": 0.01},
            {"token_index": 200, "joint_gain_per_answer_token": 0.40},
            {"token_index": 300, "joint_gain_per_answer_token": 0.35},
        ]
        got = choose_pivot(rows, minimum_positive_jump=0.05, fallback_fraction=0.5, full_length=300)
        self.assertEqual(got["pivot_token_index"], 100)
        flat = [dict(row, joint_gain_per_answer_token=0.01) for row in rows]
        got = choose_pivot(flat, minimum_positive_jump=0.05, fallback_fraction=0.5, full_length=300)
        self.assertEqual(got["pivot_token_index"], 100)


if __name__ == "__main__":
    unittest.main()
