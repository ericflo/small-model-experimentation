from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "gen_state_table_curriculum.py"
SPEC = importlib.util.spec_from_file_location("state_table_curriculum_generator", SCRIPT)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class StateTableCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = generator.generate()

    def test_default_mix_depths_and_surfaces_are_frozen(self) -> None:
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(
            Counter(row["kind"] for row in self.rows),
            Counter({f"u_state_table_{stage}": 20 for stage in generator.STAGES}),
        )
        self.assertEqual(len({row["task_id"] for row in self.rows}), 80)
        self.assertGreaterEqual(len({row["surface"] for row in self.rows}), 5)
        for stage in ("execute", "repair", "commit"):
            rows = [row for row in self.rows if row["_audit"]["stage"] == stage]
            self.assertEqual(Counter(row["depth"] for row in rows), Counter({2: 5, 3: 5, 4: 5, 5: 5}))
        scores = [row for row in self.rows if row["_audit"]["stage"] == "score"]
        self.assertEqual(Counter(row["depth"] for row in scores), Counter({2: 7, 3: 7, 4: 6}))
        self.assertEqual(
            Counter(row["_audit"]["correct_index"] for row in scores),
            Counter({0: 7, 1: 7, 2: 6}),
        )

    def test_generation_is_byte_deterministic(self) -> None:
        first = generator.render_rows(generator.generate())
        second = generator.render_rows(generator.generate())
        self.assertEqual(first, second)
        self.assertEqual(
            generator.sha256_bytes(first),
            "a7b453afa0d2273b7008a96a8460086b62ae7004fa7aa4557493728cd87e88bb",
        )
        self.assertEqual(
            (EXP / "data" / "state_table_curriculum_source.jsonl").read_bytes(), first
        )

    def test_every_answer_recomputes_from_executable_audit(self) -> None:
        for row in self.rows:
            with self.subTest(task_id=row["task_id"]):
                generator.validate_row(row)
                self.assertEqual(row["_audit"]["expected"], row["answer"].removeprefix("ANSWER: "))

    def test_score_rows_simulate_all_hypotheses_on_all_probes(self) -> None:
        rows = [row for row in self.rows if row["_audit"]["stage"] == "score"]
        for row in rows:
            with self.subTest(task_id=row["task_id"]):
                audit = row["_audit"]
                self.assertEqual(len(audit["hypotheses"]), 3)
                self.assertTrue(all(len(predictions) == 5 for predictions in audit["predictions"]))
                self.assertEqual(audit["scores"][audit["correct_index"]], 5)
                self.assertEqual(sum(score == 5 for score in audit["scores"]), 1)
                self.assertTrue(all(0 < score < 5 for index, score in enumerate(audit["scores"]) if index != audit["correct_index"]))
                for hypothesis in (1, 2, 3):
                    self.assertIn(f"H{hypothesis}:", row["think"])
                self.assertEqual(row["think"].count("| expected "), 15)

    def test_repair_rows_have_one_first_consequential_error(self) -> None:
        rows = [row for row in self.rows if row["_audit"]["stage"] == "repair"]
        observed_steps = set()
        for row in rows:
            with self.subTest(task_id=row["task_id"]):
                audit = row["_audit"]
                states = audit["states"]
                draft = audit["draft_states"]
                first = next(index for index, pair in enumerate(zip(states, draft)) if pair[0] != pair[1])
                self.assertEqual(first, audit["first_error_step"])
                self.assertNotEqual(states[-1], draft[-1])
                self.assertEqual(row["think"].count("FIRST MISMATCH; REPAIR"), 1)
                observed_steps.add(first)
        self.assertEqual(observed_steps, {1, 2, 3, 4, 5})

    def test_execution_transitions_change_and_commit_rows_are_short(self) -> None:
        for row in self.rows:
            stage = row["_audit"]["stage"]
            if stage in {"execute", "repair", "commit"}:
                states = row["_audit"]["states"]
                self.assertTrue(all(left != right for left, right in zip(states, states[1:])))
            if stage == "commit":
                self.assertLessEqual(len(row["think"].split()), 25)

    def test_interface_is_natural_language_bounded_and_answer_only(self) -> None:
        banned = (
            "LEDGER", "FIT_SECOND", "REJECT_FIRST", "APPLY_FIRST", "EXECUTE_PAIR",
            "ADVANCE_", "SWAP_", "SET_", "ROTATE_IF_",
        )
        for row in self.rows:
            with self.subTest(task_id=row["task_id"]):
                visible = row["messages"][0]["content"] + "\n" + row["think"]
                self.assertFalse(any(value in visible for value in banned))
                self.assertTrue(row["think"].endswith("COMMIT ANSWER ONLY."))
                self.assertLessEqual(len(row["think"].split()), 900)
                self.assertTrue(row["answer"].startswith("ANSWER: "))
                self.assertNotIn("\n", row["answer"])
                self.assertTrue(row["_audit"]["truth_valid"])


if __name__ == "__main__":
    unittest.main()
