from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from interface_analysis import (  # noqa: E402
    answer_cap_contact,
    calibration_qualifies,
    choose_interface,
    thinking_cap_contact,
)


GATE = {
    "rows": 48,
    "suffix_rows": 24,
    "direct_rows": 24,
    "exact_echo_successes_min": 44,
    "parse_successes_min": 44,
    "answer_cap_contacts_max": 2,
    "each_arity_exact_successes_min": 22,
    "each_arity_parse_successes_min": 22,
    "each_arity_answer_cap_contacts_max": 1,
}


def metrics(*, exact: int = 44, parsed: int = 44, caps: int = 2) -> dict:
    return {
        "rows": 48,
        "exact_echo_successes": exact,
        "parse_successes": parsed,
        "answer_cap_contacts": caps,
        "by_arity": {
            "2": {
                "rows": 24,
                "exact_echo_successes": 22,
                "parse_successes": 22,
                "answer_cap_contacts": 1,
            },
            "3": {
                "rows": 24,
                "exact_echo_successes": 22,
                "parse_successes": 22,
                "answer_cap_contacts": 1,
            },
        },
    }


class InterfaceAnalysisTests(unittest.TestCase):
    def test_integer_gate_boundaries_are_exact(self) -> None:
        self.assertTrue(calibration_qualifies(metrics(), GATE))
        self.assertFalse(calibration_qualifies(metrics(exact=43), GATE))
        self.assertFalse(calibration_qualifies(metrics(parsed=43), GATE))
        self.assertFalse(calibration_qualifies(metrics(caps=3), GATE))
        value = metrics()
        value["by_arity"]["2"]["exact_echo_successes"] = 21
        self.assertFalse(calibration_qualifies(value, GATE))

    def test_winner_is_fixed_priority_not_best_metric(self) -> None:
        priority = ["a", "b", "c", "d"]
        values = {arm: metrics() for arm in priority}
        values["b"]["exact_echo_successes"] = 48
        decision = choose_interface(values, priority=priority, gate=GATE)
        self.assertEqual(decision["winner"], "a")
        self.assertFalse(decision["selection_uses_metric_ranking"])
        values["a"] = metrics(exact=43)
        self.assertEqual(
            choose_interface(values, priority=priority, gate=GATE)["winner"], "b"
        )

    def test_no_qualifier_is_terminal(self) -> None:
        priority = ["a", "b", "c", "d"]
        values = {arm: metrics(exact=43) for arm in priority}
        decision = choose_interface(values, priority=priority, gate=GATE)
        self.assertIsNone(decision["winner"])
        self.assertEqual(decision["decision"], "NO_VALID_RESIDUAL_ANSWER_SEAM")

    def test_cap_contacts_include_exact_cap_and_length_finish(self) -> None:
        self.assertTrue(
            answer_cap_contact(
                {"n_answer_tokens": 24, "finish_reason": "stop"}, 24
            )
        )
        self.assertTrue(
            answer_cap_contact(
                {"n_answer_tokens": 3, "finish_reason": "length"}, 24
            )
        )
        self.assertFalse(
            answer_cap_contact(
                {"n_answer_tokens": 23, "finish_reason": "stop"}, 24
            )
        )

    def test_no_think_answer_length_is_not_a_thought_cap(self) -> None:
        self.assertFalse(
            thinking_cap_contact(
                {
                    "n_thinking_tokens": 0,
                    "stage1_finish_reason": "length",
                    "seed_domain_stage1": "answer",
                },
                512,
            )
        )
        self.assertTrue(
            thinking_cap_contact(
                {
                    "n_thinking_tokens": 512,
                    "stage1_finish_reason": "length",
                    "seed_domain_stage1": "thought",
                },
                512,
            )
        )


if __name__ == "__main__":
    unittest.main()
