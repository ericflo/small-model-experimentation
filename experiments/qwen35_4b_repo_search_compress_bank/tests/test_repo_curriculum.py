from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402


class RepoCurriculumTests(unittest.TestCase):
    def test_all_families_start_broken_and_oracle_passes(self):
        families = (*repo_tasks.TRAIN_FAMILIES, *repo_tasks.TRANSFER_FAMILIES)
        for task in repo_tasks.make_tasks(families, 2, seed=73001, split="test"):
            env = repo_tasks.RepoEnv(task)
            try:
                self.assertFalse(env.visible_pass())
                self.assertFalse(env.hidden_pass())
                env.apply_oracle()
                self.assertTrue(env.visible_pass())
                self.assertTrue(env.hidden_pass())
            finally:
                env.close()

    def test_parser_ignores_json_inside_think_block(self):
        raw = '{"tool":"submit"}\n</think>\n\n{"tool":"read","path":"src/x.py"}'
        action, status = repo_agent.parse_action(raw)
        self.assertEqual(status, "ok")
        self.assertEqual(action, {"tool": "read", "path": "src/x.py"})

    def test_oracle_trace_compresses_replays_and_balances_without_leakage(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 73002, "test")[0]
        env = repo_tasks.RepoEnv(task)
        steps = []
        try:
            for turn, patch in enumerate(task.oracle_patches, 1):
                action = {"tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new}
                before = env.workspace_digest()
                observation, _, _ = repo_agent.execute_action(env, action)
                after = env.workspace_digest()
                steps.append({
                    "turn": turn,
                    "action": action,
                    "observation": observation,
                    "before_digest": before,
                    "after_digest": after,
                })
            trajectory = {
                "task_id": task.task_id,
                "trajectory": 0,
                "workspace_success": env.hidden_pass(),
                "sampled_tokens": 100,
                "turns": len(steps),
                "steps": steps,
            }
        finally:
            env.close()
        built = bank.build_banks([task], [trajectory])
        self.assertFalse(built["uncovered_task_ids"])
        self.assertEqual({row["operator"] for row in built["compact_rows"]}, set(bank.OPERATORS))
        masses = built["operator_balance"]["loss_mass"]
        self.assertEqual(len({round(value, 8) for value in masses.values()}), 1)
        self.assertTrue(built["replay_receipts"][0]["hidden_pass"])
        bank.assert_firewall_clean(built, [task])
        rendered = json.dumps(built)
        self.assertNotIn(task.hidden_test, rendered)

    def test_repo_path_controls_and_visible_test_immutability(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 73003, "test")[0]
        env = repo_tasks.RepoEnv(task)
        try:
            self.assertTrue(env.read("../../etc/passwd").startswith("ERROR"))
            self.assertTrue(env.patch("tests/test_visible.py", "x", "y").startswith("ERROR"))
            self.assertTrue(env.patch("../outside.py", "x", "y").startswith("ERROR"))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
