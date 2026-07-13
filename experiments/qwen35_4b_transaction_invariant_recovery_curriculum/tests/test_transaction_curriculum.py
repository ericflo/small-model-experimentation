from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"transaction_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **_kwargs):
        return "PROMPT:" + json.dumps(messages, sort_keys=True)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(text)))}


class TransactionCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_only_registered_model_and_exact_parent_hashes(self):
        self.assertEqual(self.cfg["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(
            self.cfg["model"]["revision"],
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(len(self.cfg["model"]["start_weight_sha256"]), 64)
        self.assertEqual(len(self.cfg["model"]["anchor_weight_sha256"]), 64)
        self.assertEqual(
            self.cfg["training"]["arms"], ["transaction_replay", "replay_only"]
        )

    def test_transaction_families_are_disjoint_and_executable(self):
        train = tuple(self.cfg["families"]["transaction_train"])
        transfer = tuple(self.cfg["families"]["transaction_transfer"])
        self.assertEqual(train, repo_tasks.TRANSACTION_TRAIN_FAMILIES)
        self.assertEqual(transfer, repo_tasks.TRANSACTION_TRANSFER_FAMILIES)
        self.assertFalse(set(train) & set(transfer))
        for task in repo_tasks.make_tasks(train + transfer, 2, 86902, "unit"):
            for state, expected in (
                ("initial", (False, False)),
                ("partial", (False, False)),
                ("oracle", (True, True)),
            ):
                env = repo_tasks.RepoEnv(task)
                try:
                    if state == "partial":
                        env.apply_partial()
                    elif state == "oracle":
                        env.apply_oracle()
                    self.assertEqual(
                        (env.visible_pass(), env.hidden_pass()), expected,
                        (task.task_id, state),
                    )
                finally:
                    env.close()

    def test_transaction_rows_cover_every_conditional_transition(self):
        tasks = repo_tasks.make_tasks(
            repo_tasks.TRANSACTION_TRAIN_FAMILIES[:2], 2, 86903, "unit_bank"
        )
        built = bank.build_banks(tasks, trajectories=None)
        rows = built["recovery_action_rows"]
        self.assertEqual(len(rows), len(tasks) * len(bank.TRANSITIONS))
        by_task = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row)
        for block in by_task.values():
            self.assertEqual(Counter(row["transition"] for row in block), Counter(bank.TRANSITIONS))
            self.assertTrue(all(row["think_weight"] == 0.0 for row in block))
        receipt = bank.calibrate_transition_loss_mass(
            rows, CharacterTokenizer(), target_operator_action_mass=38248.0,
            plan_mass_fraction=0.0, max_length=20000,
        )
        self.assertEqual(
            {round(value, 7) for value in receipt["weighted_action_mass_by_operator"].values()},
            {38248.0},
        )
        bank.assert_firewall_clean(rows, tasks)
        self.assertNotIn(tasks[0].hidden_test, json.dumps(rows))

    def test_training_batches_are_complete_transition_supercycles(self):
        training = load_script("train")
        rows = []
        for task_index in range(5):
            for transition in bank.TRANSITIONS:
                rows.append({
                    "task_id": f"task-{task_index}", "row_id": f"task-{task_index}-{transition}",
                    "transition": transition, "operator": bank.TRANSITION_OPERATOR[transition],
                    "family": "synthetic", "input_ids": [1], "attention_mask": [1],
                    "labels": [1], "loss_weights": [1.0], "answer_mask": [1.0],
                })
        ordered, receipt = training.make_batches(rows, 4, 7, 43)
        self.assertTrue(receipt["complete_transition_supercycle"])
        self.assertTrue(receipt["every_optimizer_step_contains_all_transitions"])
        self.assertEqual(receipt["whole_task_padding_duplicates"], 3)
        self.assertEqual(len(ordered), 8 * 7)

    def test_locality_block_is_fresh_and_includes_varentropy(self):
        payload = json.loads((EXP / "data" / "locality_contexts.json").read_text())
        current = {row["content_sha256"] for row in payload["contexts"]}
        self.assertEqual(len(current), 48)
        prior = set()
        for path in (ROOT / "experiments").glob("*/data/*.json"):
            if EXP in path.parents:
                continue
            try:
                other = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows = other.get("contexts", []) if isinstance(other, dict) else []
            prior.update(
                row.get("content_sha256") for row in rows
                if isinstance(row, dict) and row.get("content_sha256")
            )
        self.assertFalse(current & prior)
        locality_source = (EXP / "scripts" / "audit_locality.py").read_text()
        self.assertIn("varentropy", locality_source)
        self.assertIn("entropy_retained", locality_source)


if __name__ == "__main__":
    unittest.main()
