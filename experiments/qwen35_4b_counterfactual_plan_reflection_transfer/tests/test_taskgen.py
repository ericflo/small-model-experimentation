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
            # Immutable task truth is preserved; only the supervised label changes.
            self.assertEqual(correct["target_ops"], shuffled["target_ops"])
            self.assertEqual(correct["target_plan"], shuffled["target_plan"])
            self.assertNotEqual(correct["supervision_ops"], shuffled["supervision_ops"])
            self.assertNotEqual(correct["supervision_plan"], shuffled["supervision_plan"])
            self.assertEqual(correct["common_messages"], shuffled["common_messages"])
            family = next(family for family in T.FAMILIES if family.name == correct["family"])
            inputs = [row["input"] for row in correct["examples"]] + correct["queries"]
            outputs = [row["output"] for row in correct["examples"]] + correct["answers"]
            self.assertFalse(
                T._matches(family, tuple(shuffled["supervision_ops"]), inputs, outputs)
            )

    def test_explosive_candidates_are_skipped(self) -> None:
        with self.assertRaisesRegex(ValueError, "state explosion"):
            T.execute(T.LIST, ("square", "square", "square"), [6, 4, 3, 2])
        # A neighboring corpus still constructs instead of leaking the exception.
        corpus = T.build_corpus({"train": 8}, 91_337)
        self.assertEqual(len(corpus["train"]), 8 * len(T.FAMILIES))

    def test_proposed_full_splits_are_feasible_and_position_complete(self) -> None:
        counts = {"train": 72, "calibration": 24, "qualification": 48, "confirmation": 48}
        corpus = T.build_corpus(counts, 73_301)
        receipt = T.validate_corpus(corpus, counts)
        self.assertEqual(receipt["unique_behavior_signatures"], 576)
        for rows in corpus.values():
            for family in T.FAMILIES:
                family_rows = [row for row in rows if row["family"] == family.name]
                expected = {primitive.name for primitive in family.primitives}
                for position in range(3):
                    self.assertEqual(
                        {row["target_ops"][position] for row in family_rows}, expected
                    )
                self.assertTrue(
                    all(row["visible_shallow_candidate_count"] == 0 for row in family_rows)
                )
                self.assertTrue(
                    all(row["visible_depth3_candidate_count"] == 1 for row in family_rows)
                )

    def test_retention_depths_are_real_and_identifiable(self) -> None:
        retention = T.build_retention_corpus(count_per_family_per_depth=8, seed=73_337)
        receipt = T.validate_retention_corpus(retention, count_per_family_per_depth=8)
        self.assertEqual(receipt["tasks"], 48)
        self.assertEqual(receipt["depth_1"], 24)
        self.assertEqual(receipt["depth_2"], 24)
        main_signatures = {
            (row["family"], row["behavior_signature_sha256"])
            for rows in self.corpus.values()
            for row in rows
        }
        retention_signatures = {
            (row["family"], row["behavior_signature_sha256"]) for row in retention
        }
        self.assertTrue(main_signatures.isdisjoint(retention_signatures))


if __name__ == "__main__":
    unittest.main()
