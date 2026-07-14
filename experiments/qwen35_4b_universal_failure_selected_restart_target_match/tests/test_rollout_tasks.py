from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_rollout_tasks.py"
SPEC = importlib.util.spec_from_file_location("restart_rollout_tasks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RolloutTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = MODULE.generate_source()

    def test_generation_is_deterministic_fresh_and_balanced(self) -> None:
        self.assertEqual(MODULE.jsonl_bytes(self.rows), MODULE.jsonl_bytes(MODULE.generate_source()))
        summary = MODULE.validate(self.rows)
        self.assertEqual(len(self.rows), 624)
        self.assertEqual(
            Counter(row["selection_skill"] for row in self.rows),
            Counter({skill: 48 for skill in MODULE.curriculum.SKILLS}),
        )
        self.assertEqual(summary["rows"], 624)

    def test_runner_input_excludes_every_oracle_field(self) -> None:
        rendered = MODULE.jsonl_bytes(MODULE.runner_rows(self.rows))
        public = [json.loads(line) for line in rendered.decode().splitlines()]
        self.assertEqual(len(public), 624)
        for row in public:
            self.assertEqual(set(row), {"id", "messages", "meta"})
            self.assertEqual(
                set(row["meta"]),
                {"kind", "skill", "surface", "level", "construction_seed"},
            )
        for forbidden in (b'"answer"', b'"think"', b'"_audit"', b'"truth_valid"'):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
