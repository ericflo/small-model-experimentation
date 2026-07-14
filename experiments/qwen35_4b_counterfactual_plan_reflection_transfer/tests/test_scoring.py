from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import scoring as S  # noqa: E402


class ScoringTests(unittest.TestCase):
    def test_strict_parser_accepts_only_exact_three_output_array(self) -> None:
        self.assertEqual(S.parse_answer("</think>\nANSWER: [[1],\"x\",[2,3]]"), [[1], "x", [2, 3]])
        for bad in (
            "Here: ANSWER: [1,2,3]",
            "ANSWER: [1,2]",
            "ANSWER: [1,2,3] trailing",
            "ANSWER: nope",
        ):
            with self.subTest(bad=bad):
                self.assertIsNone(S.parse_answer(bad))

    def test_exact_periodic_tail_requires_repeats_and_total_length(self) -> None:
        self.assertTrue(S.exact_periodic_tail([1, 2, 3] * 20, 4, 16, 60))
        self.assertFalse(S.exact_periodic_tail(list(range(60)), 4, 16, 60))
        self.assertFalse(S.exact_periodic_tail([1, 2, 3] * 10, 4, 16, 60))

    def test_scoring_uses_candidate_prefixes_and_contacts(self) -> None:
        outputs = []
        for index in range(4):
            outputs.append(
                {
                    "text": "</think>\n\nANSWER: [1,2,3]" if index == 2 else "bad",
                    "truncated": index == 3,
                    "n_answer_tokens": 3,
                    "retained_thinking_token_ids": [1, 2, 3] * 20 if index == 3 else [],
                    "n_sampled_tokens": 10,
                    "n_injected_tokens": 2,
                }
            )
        scored = S.score_generation_rows(
            [{"id": "t1", "outputs": outputs}],
            [{"id": "t1", "family": "list", "depth": 3, "split": "qualification", "answers": [1, 2, 3]}],
            arm="frozen_action",
            candidate_counts=(1, 4),
            answer_max_tokens=128,
            loop_detector={"min_repeats": 4, "max_period_tokens": 16, "min_total_repeated_tokens": 60},
        )[0]
        self.assertEqual(scored["coverage_at_1"], 0.0)
        self.assertEqual(scored["coverage_at_4"], 1.0)
        self.assertEqual(scored["strict_parse_rate"], 0.25)
        self.assertEqual(scored["answer_limit_contact"], 0.25)
        self.assertEqual(scored["periodic_loop_contact"], 0.25)
        self.assertEqual(scored["logical_tokens"], 48)


if __name__ == "__main__":
    unittest.main()
