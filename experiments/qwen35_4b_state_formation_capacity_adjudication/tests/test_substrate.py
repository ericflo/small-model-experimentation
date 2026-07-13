from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.substrate import FAMILIES, generate_example, trajectory_targets, verify_example  # noqa: E402


class SubstrateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.yaml")
        cls.architecture = cls.config["architecture"]
        cls.substrate = cls.config["substrate"]

    def make(self, seed: int, family: str, depth: int, *, query_kind: str = "node") -> dict:
        return generate_example(
            seed=seed,
            split="test",
            family=family,
            template="ledger",
            depth=depth,
            node_count=self.substrate["node_count"],
            checksum_modulus=self.substrate["checksum_modulus"],
            num_choices=self.substrate["num_choices"],
            state_token=self.architecture["state_token"],
            state_slots=self.architecture["state_slots"],
            max_attempts=self.substrate["max_generation_attempts"],
            query_kind=query_kind,
        )

    def test_generation_is_deterministic_and_verifiable_for_every_family(self) -> None:
        for index, family in enumerate(FAMILIES):
            with self.subTest(family=family):
                first = self.make(100 + index, family, 7)
                second = self.make(100 + index, family, 7)
                self.assertEqual(first, second)
                verify_example(first, self.architecture["state_token"], self.architecture["state_slots"])
                self.assertEqual(len(first["trajectory"]), 8)
                query_values = [state[first["query_kind"]] for state in first["trajectory"]]
                self.assertNotIn(query_values[-1], query_values[:-1])

    def test_query_is_causally_after_all_state_slots(self) -> None:
        row = self.make(201, "phase_branch", 4)
        prompt = row["prompt"]
        self.assertEqual(
            prompt.count(self.architecture["state_token"]),
            self.architecture["state_slots"],
        )
        self.assertLess(prompt.rindex(self.architecture["state_token"]), prompt.index("Query:"))

    def test_targets_halt_after_semantic_depth(self) -> None:
        row = self.make(202, "checksum_branch", 2, query_kind="checksum")
        targets = trajectory_targets(row, 6)
        for name in ("node", "phase", "checksum"):
            self.assertEqual(targets[name][1:], [targets[name][-1]] * 5)


if __name__ == "__main__":
    unittest.main()
