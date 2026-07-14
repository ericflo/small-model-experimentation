from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_rollout_tasks.py"
SPEC = importlib.util.spec_from_file_location("on_policy_rollout_tasks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RolloutTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = MODULE.generate()

    def test_generation_is_byte_deterministic_and_balanced(self) -> None:
        self.assertEqual(MODULE.render_source(self.rows), MODULE.render_source(MODULE.generate()))
        self.assertEqual(len(self.rows), 288)
        self.assertEqual(
            Counter(row["failure_class"] for row in self.rows),
            Counter({name: 48 for name in MODULE.FAILURE_CLASSES}),
        )

    def test_runner_input_has_only_messages_id_and_public_routing_metadata(self) -> None:
        rendered = MODULE.render_runner_input(self.rows).decode().splitlines()
        public = [json.loads(line) for line in rendered]
        self.assertEqual(len(public), len(self.rows))
        for row in public:
            self.assertEqual(set(row), {"id", "messages", "meta"})
            self.assertNotIn("oracle_think", row)
            self.assertNotIn("answer", row)
            self.assertEqual(
                set(row["meta"]), {"failure_class", "surface", "level"}
            )

    def test_declaration_cycle_is_never_a_listed_operation(self) -> None:
        rows = [
            row for row in self.rows if row["failure_class"] == "declaration_operation"
        ]
        self.assertEqual(len(rows), 48)
        for row in rows:
            prompt = row["messages"][0]["content"]
            procedure = prompt.partition("Cycle order:")[0]
            self.assertNotIn("advance every item", procedure)
            self.assertTrue(
                row["_audit"]["source_audit"]["cycle_is_reference_only"]
            )

    def test_commit_tasks_expose_verified_work_but_keep_hidden_oracle_separate(self) -> None:
        rows = [
            row for row in self.rows if row["failure_class"] == "commit_serialization"
        ]
        self.assertEqual(len(rows), 48)
        for row in rows:
            prompt = row["messages"][0]["content"]
            self.assertIn("Verified scratch work:", prompt)
            self.assertTrue(
                row["_audit"]["source_audit"]["immediate_commit_required"]
            )
            self.assertNotEqual(row["oracle_think"], prompt)


if __name__ == "__main__":
    unittest.main()
