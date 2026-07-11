#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402


class RepoTaskTests(unittest.TestCase):
    def test_every_family_has_failing_bug_and_passing_oracle(self):
        tasks = repo_tasks.make_tasks(list(repo_tasks.BUILDERS), 1, 1234)
        self.assertEqual(len(tasks), len(repo_tasks.BUILDERS))
        for task in tasks:
            env = repo_tasks.RepoEnv(task)
            try:
                self.assertTrue(env.initial_visible_fails(), task.family)
                self.assertFalse(env.hidden_pass(), task.family)
                env.apply_oracle()
                self.assertTrue(env.run_visible().startswith("PASS"), task.family)
                self.assertTrue(env.hidden_pass(), task.family)
            finally:
                env.close()

    def test_patch_rejects_test_and_escape_paths(self):
        task = repo_tasks.make_tasks(["window_rollup"], 1, 2)[0]
        env = repo_tasks.RepoEnv(task)
        try:
            self.assertTrue(env.patch("tests/test_visible.py", "x", "y").startswith("ERROR"))
            self.assertTrue(env.patch("../escape.py", "x", "y").startswith("ERROR"))
        finally:
            env.close()

    def test_tool_json_parser_uses_answer_region(self):
        action, status = repo_agent.parse_action(
            '<think>{"tool":"bad"}</think>\n{"tool":"read","path":"src/a.py"}')
        self.assertEqual(status, "ok")
        self.assertEqual(action, {"tool": "read", "path": "src/a.py"})
        action, status = repo_agent.parse_action("no tool")
        self.assertIsNone(action)
        self.assertEqual(status, "no_json_tool_call")


if __name__ == "__main__":
    unittest.main()
