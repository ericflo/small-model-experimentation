from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import protocol  # noqa: E402
import task_data as td  # noqa: E402


def _rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (EXP / "data" / "procedural" / name).read_text().splitlines()
    ]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        public = _rows("mechanics_public.jsonl")
        gold = {row["task_id"]: row for row in _rows("mechanics_gold.jsonl")}
        audit = {row["task_id"]: row for row in _rows("mechanics_audit.jsonl")}
        cls.tasks = [(row, gold[row["task_id"]], audit[row["task_id"]]) for row in public]

    def test_strict_alias_parsers(self) -> None:
        self.assertTrue(protocol.parse_program("PROGRAM: A | X", arity=2)["parsed"])
        self.assertTrue(protocol.parse_program("PROGRAM: A | B | C", arity=3)["parsed"])
        for value in ("A | B", "PROGRAM: A", "PROGRAM: A | B\nextra", "PROGRAM: A | Z"):
            self.assertFalse(protocol.parse_program(value, arity=2)["parsed"])

    def test_prompts_are_public_and_derangement_breaks_values(self) -> None:
        public, _gold, audit = self.tasks[0]
        live = audit["public_live"][0]
        candidate = td.operation_from_record(live["operation"])
        clean = protocol.suffix_prompt(public, candidate=candidate, representation="materialized")
        shuffled = protocol.suffix_prompt(public, candidate=candidate, representation="shuffled")
        self.assertNotEqual(clean, shuffled)
        self.assertNotIn("hidden", clean.lower())
        self.assertNotIn("probe", clean.lower())
        outputs = [row["output"] for row in public["visible"]]
        permutation = protocol.target_derangement(outputs, salt="test")
        self.assertTrue(all(outputs[index] != outputs[source] for index, source in enumerate(permutation)))

    def test_actual_target_is_selectable_without_hidden_labels(self) -> None:
        for public, gold, _audit in self.tasks[:4]:
            target = tuple(td.operation_from_record(value) for value in gold["target_pipeline"])
            candidate = {
                "candidate_id": "target",
                "candidate": target[0],
                "text": f"PROGRAM: {td.alias_program(target[1:])}",
            }
            selected = protocol.select_visible(public, [candidate])
            self.assertFalse(selected["abstained"])
            self.assertEqual(selected["selected_candidate_id"], "target")

    def test_viability_orientation_is_explicit_and_balanced_in_data(self) -> None:
        orientations = {"A": 0, "B": 0}
        for public, _gold, _audit in self.tasks:
            orientations[public["viability_live_alias"]] += 1
            mapping = protocol.viability_mapping(public)
            self.assertEqual(mapping[public["viability_live_alias"]], "LIVE")
        self.assertEqual(orientations, {"A": 12, "B": 12})


if __name__ == "__main__":
    unittest.main()
