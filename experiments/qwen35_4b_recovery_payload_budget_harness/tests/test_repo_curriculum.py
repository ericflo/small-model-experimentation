from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402
from vllm_runner import EngineConfig  # noqa: E402


class CharacterTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **_kwargs):
        return "PROMPT:" + json.dumps(messages, sort_keys=True)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(text)))}


class RepoCurriculumTests(unittest.TestCase):
    def test_all_families_have_real_unresolved_partial_states(self):
        families = (*repo_tasks.TRAIN_FAMILIES, *repo_tasks.TRANSFER_FAMILIES)
        for task in repo_tasks.make_tasks(families, 2, seed=84501, split="test"):
            for state, expected in (("initial", (False, False)),
                                    ("partial", (False, False)),
                                    ("oracle", (True, True))):
                env = repo_tasks.RepoEnv(task)
                try:
                    if state == "partial":
                        env.apply_partial()
                    elif state == "oracle":
                        env.apply_oracle()
                    observed = (env.visible_pass(), env.hidden_pass())
                    self.assertEqual(observed, expected, (task.task_id, state))
                finally:
                    env.close()

    def test_runtime_caches_are_not_on_public_tool_surface(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 84502, "test")[0]
        env = repo_tasks.RepoEnv(task)
        try:
            env.run_visible()
            self.assertNotIn("__pycache__", env.tree())
            self.assertIsInstance(env.search("def "), str)
            self.assertNotIn(".pyc", env.search("def "))
        finally:
            env.close()

    def test_parser_ignores_json_inside_think_block(self):
        raw = '{"tool":"submit"}\n</think>\n\n{"tool":"read","path":"src/x.py"}'
        action, status = repo_agent.parse_action(raw)
        self.assertEqual(status, "ok")
        self.assertEqual(action, {"tool": "read", "path": "src/x.py"})

    def test_controlled_scenarios_are_public_failures_and_scaffolded(self):
        tasks = repo_tasks.make_tasks(repo_tasks.TRAIN_FAMILIES, 1, 84503, "test")
        for task in tasks:
            rejected = repo_agent.Episode(task, 0, scenario="rejected_patch", scaffold=True)
            try:
                self.assertTrue(rejected.prefix_steps[-1]["observation"].startswith("ERROR"))
                self.assertEqual(
                    rejected.prefix_steps[-1]["before_digest"],
                    rejected.prefix_steps[-1]["after_digest"],
                )
                self.assertIn("RECOVERY RULE:", rejected.messages[-1]["content"])
            finally:
                rejected.env.close()
            failed = repo_agent.Episode(task, 0, scenario="failed_test", scaffold=True)
            try:
                self.assertTrue(failed.prefix_steps[-1]["observation"].startswith("FAIL"))
                self.assertFalse(failed.env.hidden_pass())
                self.assertIn("RECOVERY RULE:", failed.messages[-1]["content"])
            finally:
                failed.env.close()

    def test_banks_replay_and_balance_every_conditional_stratum(self):
        tasks = repo_tasks.make_tasks(repo_tasks.TRAIN_FAMILIES[:2], 2, 84504, "test")
        built = bank.build_banks(tasks, trajectories=None)
        self.assertFalse(built["uncovered_task_ids"])
        for key in ("happy_action_rows", "recovery_action_rows", "recovery_reason_rows"):
            rows = built[key]
            self.assertEqual(len(rows), len(tasks) * len(bank.TRANSITIONS))
            self.assertEqual(set(row["transition"] for row in rows), set(bank.TRANSITIONS))
        action = built["recovery_action_rows"]
        reason = built["recovery_reason_rows"]
        for left, right in zip(action, reason):
            self.assertEqual(left["messages"], right["messages"])
            self.assertEqual(left["think"], right["think"])
            self.assertEqual(left["answer"], right["answer"])
            self.assertEqual(left["prefix_actions"], right["prefix_actions"])
        by_transition = {row["transition"]: row for row in action if row["task_id"] == tasks[0].task_id}
        self.assertTrue(
            by_transition["rejected_patch_to_changed_patch"]["state_receipt"]["observations"][-1]
            .startswith("ERROR")
        )
        self.assertTrue(
            by_transition["failed_test_to_diagnose"]["state_receipt"]["observations"][-1]
            .startswith("FAIL")
        )
        self.assertTrue(by_transition["passed_test_to_commit"]["state_receipt"]["visible_pass"])

        tokenizer = CharacterTokenizer()
        probe = bank.calibrate_transition_loss_mass(
            action, tokenizer, target_operator_action_mass=1.0,
            plan_mass_fraction=0.0, max_length=20000,
        )
        target = sum(probe["raw_action_tokens_by_transition"].values())
        for rows, plan_fraction in (
            (built["happy_action_rows"], 0.0),
            (action, 0.0),
            (reason, 0.05),
        ):
            receipt = bank.calibrate_transition_loss_mass(
                rows, tokenizer, target_operator_action_mass=target,
                plan_mass_fraction=plan_fraction, max_length=20000,
            )
            self.assertEqual(
                len({round(value, 7) for value in receipt["weighted_action_mass_by_operator"].values()}),
                1,
            )
            for transitions in receipt["operator_transition_strata"].values():
                masses = [receipt["weighted_action_mass_by_transition"][name] for name in transitions]
                self.assertEqual(len({round(value, 7) for value in masses}), 1)
        bank.assert_firewall_clean(built, tasks)
        self.assertNotIn(tasks[0].hidden_test, json.dumps(built))

    def test_invalid_action_steps_do_not_break_patch_extraction(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 84505, "test")[0]
        env = repo_tasks.RepoEnv(task)
        steps = [{"turn": 1, "action": None, "before_digest": env.workspace_digest(),
                  "after_digest": env.workspace_digest()}]
        try:
            patch = task.oracle_patches[0]
            action = {"tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new}
            before = env.workspace_digest()
            observation, _, _ = repo_agent.execute_action(env, action)
            steps.append({"turn": 2, "action": action, "observation": observation,
                          "before_digest": before, "after_digest": env.workspace_digest()})
            trajectory = {"task_id": task.task_id, "trajectory": 0,
                          "workspace_success": env.visible_pass() and env.hidden_pass(),
                          "sampled_tokens": 5, "turns": 2, "steps": steps}
        finally:
            env.close()
        built = bank.build_banks([task], [trajectory])
        self.assertEqual(len(built["replay_receipts"]), 1)

    def test_repo_path_controls_and_visible_test_immutability(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 84506, "test")[0]
        env = repo_tasks.RepoEnv(task)
        try:
            self.assertTrue(env.read("../../etc/passwd").startswith("ERROR"))
            self.assertTrue(env.patch("tests/test_visible.py", "x", "y").startswith("ERROR"))
            self.assertTrue(env.patch("../outside.py", "x", "y").startswith("ERROR"))
        finally:
            env.close()

    def test_local_checkpoint_requires_exact_qwen35_4b_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text", "vocab_size": 248320,
                    "hidden_size": 2560, "num_hidden_layers": 32,
                },
            }
            (root / "config.json").write_text(json.dumps(good), encoding="utf-8")
            EngineConfig(model_override=root).validate()
            good["text_config"]["hidden_size"] = 4096
            (root / "config.json").write_text(json.dumps(good), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a merged Qwen/Qwen3.5-4B"):
                EngineConfig(model_override=root).validate()


if __name__ == "__main__":
    unittest.main()
