from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from protocol import parse_program, score_echo, select_visible  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ProtocolTests(unittest.TestCase):
    def test_exact_program_grammar_rejects_whitespace_and_embedded_lines(self) -> None:
        accepted = parse_program("PROGRAM: A | X", arity=2)
        self.assertTrue(accepted["parsed"])
        rejected = (
            " PROGRAM: A | X",
            "PROGRAM:A | X",
            "PROGRAM: A|X",
            "PROGRAM: A | X ",
            "PROGRAM: A\n| X",
            "explanation\nPROGRAM: A | X",
        )
        for text in rejected:
            with self.subTest(text=text):
                self.assertFalse(parse_program(text, arity=2)["parsed"])

    def test_thinking_boundary_and_terminal_are_exact(self) -> None:
        value = "thought</think>\n\nPROGRAM: A | B<|im_end|>"
        self.assertTrue(parse_program(value, arity=2)["parsed"])
        self.assertTrue(
            parse_program(value, arity=2, thinking_expected=True)["parsed"]
        )
        self.assertFalse(
            parse_program("thought</think>\nPROGRAM: A | B", arity=2)["parsed"]
        )
        adversarial = (
            "junk</think>\n\nPROGRAM: A | B",
            "thought</think>\n\njunk</think>\n\nPROGRAM: A | B",
        )
        for text in adversarial:
            with self.subTest(text=text):
                self.assertFalse(
                    parse_program(
                        text, arity=2, thinking_expected=False
                    )["parsed"]
                )
        self.assertFalse(
            parse_program(
                adversarial[1], arity=2, thinking_expected=True
            )["parsed"]
        )
        self.assertFalse(
            parse_program(
                "PROGRAM: A | B", arity=2, thinking_expected=True
            )["parsed"]
        )

    def test_exact_echo_is_stricly_separate_from_parse(self) -> None:
        value = score_echo(
            "PROGRAM: A | B", expected="PROGRAM: B | A", arity=2
        )
        self.assertTrue(value["parsed"])
        self.assertFalse(value["exact_echo"])

    def test_calibration_aliases_are_balanced_in_every_position(self) -> None:
        rows = read_jsonl(EXP / "data/procedural/calibration_public.jsonl")
        for arity in (2, 3):
            subset = [row for row in rows if row["arity"] == arity]
            self.assertEqual(len(subset), 24)
            for position in range(arity):
                self.assertEqual(
                    sorted(row["expected_aliases"][position] for row in subset),
                    list("ABCDEFGHIJKLMNOPQRSTUVWX"),
                )

    def test_visible_selector_uses_no_hidden_field(self) -> None:
        task = read_jsonl(EXP / "data/procedural/mechanics_public.jsonl")[0]
        candidates = [
            {
                "candidate_id": "bad",
                "candidate": None,
                "text": "PROGRAM: A | A | A",
            }
        ]
        first = select_visible(task, candidates)
        mutated = {**task, "not_hidden": [{"input": [999], "output": [0]}]}
        with self.assertRaisesRegex(ValueError, "schema"):
            select_visible(mutated, candidates)
        second = select_visible(task, candidates)
        self.assertEqual(first, second)
        echoed_close = select_visible(
            task,
            [
                {
                    "candidate_id": "close",
                    "candidate": None,
                    "text": "junk</think>\n\nPROGRAM: A | A | A",
                }
            ],
            thinking_expected=False,
        )
        self.assertFalse(echoed_close["scored"][0]["parsed"])
        self.assertEqual(
            echoed_close["scored"][0]["error"],
            "unexpected thinking answer boundary",
        )


if __name__ == "__main__":
    unittest.main()
