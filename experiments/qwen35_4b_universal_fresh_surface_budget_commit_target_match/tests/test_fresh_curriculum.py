import json
import random
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as original  # noqa: E402
import gen_fresh_curriculum as fresh  # noqa: E402


class FreshSurfaceTests(unittest.TestCase):
    def test_surface_pools_disjoint_from_original(self) -> None:
        fresh_tokens = {item for pool in fresh.SURFACE_POOLS.values() for item in pool}
        original_tokens = {item for pool in original.SURFACE_POOLS.values() for item in pool}
        self.assertFalse(fresh_tokens & original_tokens)
        self.assertFalse(set(fresh.SEPARATORS) & set(original.SEPARATORS))
        self.assertFalse(set(fresh.ATTRIBUTES) & set(original.ATTRIBUTES))

    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = fresh.generate_curriculum(fresh.SMOKE_MIX, 12345)
        summary = fresh.validate_generated(rows)
        fresh.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 2 * len(fresh.SKILLS))
        self.assertIn("u_budget", summary["kinds"])

    def test_budget_hit_answer_is_first_satisfier_within_allowance(self) -> None:
        rng = random.Random(7)
        for _ in range(60):
            row = fresh.budget_lesson(rng)
            audit = row["_audit"]
            prompt = row["messages"][0]["content"]
            lines = [line.strip()[2:] for line in prompt.splitlines() if line.strip().startswith("- ")]
            records = []
            for line in lines:
                identifier, _, rest = line.partition(":")
                records.append((identifier.strip(), int(rest.strip().rsplit(" ", 1)[-1])))
            threshold = int(prompt.split("is at least ")[1].split(" ")[0])
            allowance = audit["allowance"]
            scanned = records[:allowance]
            first_hit = next((identifier for identifier, value in scanned if value >= threshold), None)
            expected = first_hit if first_hit is not None else "BUDGET"
            self.assertEqual(row["answer"], f"ANSWER: {expected}")
            if audit["outcome"] == "exhaust":
                self.assertGreaterEqual(records[allowance][1], threshold)  # decoy past cutoff
                self.assertTrue(all(value < threshold for _, value in scanned))
            else:
                self.assertEqual(audit["hit_position"] is not None, True)
                self.assertLessEqual(audit["hit_position"], allowance)

    def test_frozen_corpora_regenerate_and_arm_b_is_subset_plus_budget(self) -> None:
        designed = fresh.generate_curriculum(fresh.ARM_D_MIX, 77116)
        arm_d_lines = {
            json.dumps(fresh.public_row(row), sort_keys=True, ensure_ascii=False)
            for row in designed
        }
        arm_b_rows = [
            json.loads(line)
            for line in (EXP / "data" / "sft_fresh_budget160.jsonl").read_text().splitlines()
        ]
        designed_part = [row for row in arm_b_rows if row["kind"] != "u_budget"]
        budget_part = [row for row in arm_b_rows if row["kind"] == "u_budget"]
        self.assertEqual(len(designed_part), 120)
        self.assertEqual(len(budget_part), 40)
        for row in designed_part:
            self.assertIn(json.dumps(row, sort_keys=True, ensure_ascii=False), arm_d_lines)

    def test_lesson_logic_unchanged_from_original(self) -> None:
        # The thirteen constructors must stay behaviorally identical up to surface
        # vocabulary: pin the op algebra and a rendered execute lesson's structure.
        self.assertEqual(fresh.OP_KINDS, original.OP_KINDS)
        row = fresh.execute_lesson(random.Random(3))
        self.assertTrue(row["answer"].startswith("ANSWER: "))
        self.assertIn("Step 1", row["think"])


if __name__ == "__main__":
    unittest.main()
