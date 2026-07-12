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

    def test_invalid_action_steps_do_not_break_patch_extraction(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 73004, "test")[0]
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
                          "workspace_success": env.hidden_pass(), "sampled_tokens": 5,
                          "turns": 2, "steps": steps}
        finally:
            env.close()
        built = bank.build_banks([task], [trajectory])
        self.assertEqual(len(built["replay_receipts"]), 1)

    def test_hidden_only_pass_is_not_counted_as_repository_success(self):
        tasks = repo_tasks.make_tasks(repo_tasks.TRAIN_FAMILIES, 1, 73100, "harvest")
        task = next(task for task in tasks if task.family == "stable_merge")
        key = "id" if "`id`" in task.issue else "code"
        wrong = f'''"""Stable field-aware collection merge."""

def merge_unique(groups):
    seen = set()
    merged = []
    for group in groups:
        for item in group:
            marker = item["{key}"]
            if marker in seen:
                for field, value in item.items():
                    if field not in merged[-1] or merged[-1][field] is None:
                        merged[-1][field] = value
            else:
                seen.add(marker)
                merged.append(dict(item))
    return merged
'''
        action = {"tool": "patch", "path": "src/merge.py",
                  "old": task.files["src/merge.py"], "new": wrong}
        episode = repo_agent.Episode(task, 0)
        episode.consume({"text": f'</think>\n\n{json.dumps(action)}',
                         "n_sampled_tokens": 1, "thinking_closed": True})
        result = episode.finish()
        self.assertFalse(result["final_visible_pass"])
        self.assertTrue(result["final_hidden_pass"])
        self.assertFalse(result["workspace_success"])

    def test_repo_path_controls_and_visible_test_immutability(self):
        task = repo_tasks.make_tasks([repo_tasks.TRAIN_FAMILIES[0]], 1, 73003, "test")[0]
        env = repo_tasks.RepoEnv(task)
        try:
            self.assertTrue(env.read("../../etc/passwd").startswith("ERROR"))
            self.assertTrue(env.patch("tests/test_visible.py", "x", "y").startswith("ERROR"))
            self.assertTrue(env.patch("../outside.py", "x", "y").startswith("ERROR"))
        finally:
            env.close()

    def test_exact_token_mass_calibration_balances_long_patch_actions(self):
        class FakeTokenizer:
            eos_token = "<eos>"

            def apply_chat_template(self, messages, **_kwargs):
                return "P" * (10 + len(messages))

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return {"input_ids": list(range(len(text)))}

        rows = []
        lengths = {"INSPECT": 10, "PATCH": 100, "VERIFY": 5, "COMMIT": 3}
        for operator, length in lengths.items():
            rows.append({"id": operator, "operator": operator,
                         "messages": [{"role": "user", "content": "x"}],
                         "think": "plan", "answer": "a" * length})
        receipt = bank.calibrate_token_loss_mass(
            rows, FakeTokenizer(), repo_multiplier=4.0, max_length=1000
        )
        self.assertEqual(
            len({round(value, 8) for value in receipt["weighted_action_token_mass"].values()}), 1
        )
        self.assertEqual(
            len({round(value, 8) for value in receipt["weighted_plan_token_mass"].values()}), 1
        )
        self.assertLess(rows[1]["row_weight"], rows[3]["row_weight"])

    def test_local_checkpoint_requires_exact_qwen35_4b_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 248320,
                    "hidden_size": 2560,
                    "num_hidden_layers": 32,
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
