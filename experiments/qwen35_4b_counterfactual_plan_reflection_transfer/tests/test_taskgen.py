from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import taskgen as T  # noqa: E402


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.counts = {"train": 4, "qualification": 3, "confirmation": 3}
        cls.corpus = T.build_corpus(cls.counts, 73_301)

    def test_splits_are_composition_and_behavior_disjoint(self) -> None:
        receipt = T.validate_corpus(self.corpus, self.counts)
        self.assertEqual(receipt["cross_split_composition_collisions"], 0)
        self.assertEqual(receipt["cross_split_behavior_collisions"], 0)

    def test_targets_reexecute_exactly(self) -> None:
        families = {family.name: family for family in T.FAMILIES}
        for rows in self.corpus.values():
            for row in rows:
                actual = [
                    T.execute(families[row["family"]], tuple(row["target_ops"]), query)
                    for query in row["queries"]
                ]
                self.assertEqual(actual, row["answers"])

    def test_reflection_never_contains_exact_answer(self) -> None:
        for rows in self.corpus.values():
            for row in rows:
                self.assertNotIn(row["target_answer"], row["target_plan"])
                self.assertNotIn(T._state_text(row["answers"]), row["target_plan"])

    def test_shuffled_arm_is_within_family_derangement(self) -> None:
        arms = T.build_reflection_arms(self.corpus["train"], 73_319)
        for correct, shuffled in zip(arms["reflection_correct"], arms["reflection_shuffled"]):
            self.assertEqual(correct["task_id"], shuffled["task_id"])
            self.assertEqual(correct["family"], shuffled["family"])
            self.assertNotEqual(correct["target_ops"], shuffled["target_ops"])
            self.assertNotEqual(correct["target_plan"], shuffled["target_plan"])
            self.assertEqual(correct["common_messages"], shuffled["common_messages"])


if __name__ == "__main__":
    unittest.main()
