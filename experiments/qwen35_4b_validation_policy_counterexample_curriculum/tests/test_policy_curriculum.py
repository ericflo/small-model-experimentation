from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"policy_{name}", path)
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


class PolicyCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_only_registered_model_exact_hashes_and_fixed_arms(self):
        self.assertEqual(self.cfg["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(
            self.cfg["model"]["revision"],
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(
            self.cfg["model"]["start_weight_sha256"],
            "1cf5fbca317808d6d00225f5cd533c82c7e1602b2b2e5e2da8f4307b01941ba3",
        )
        self.assertEqual(
            self.cfg["training"]["arms"],
            ["policy_counterexample", "extra_transaction"],
        )
        self.assertEqual(
            self.cfg["bank"]["injected_transition"], "diagnosis_to_changed_patch"
        )

    def test_policy_families_are_disjoint_structured_and_executable(self):
        train = tuple(self.cfg["families"]["policy_train"])
        transfer = tuple(self.cfg["families"]["policy_transfer"])
        self.assertEqual(train, repo_tasks.POLICY_TRAIN_FAMILIES)
        self.assertEqual(transfer, repo_tasks.POLICY_TRANSFER_FAMILIES)
        self.assertFalse(set(train) & set(transfer))
        shapes = set()
        for task in repo_tasks.make_tasks(train + transfer, 2, 87602, "unit"):
            source = next(text for path, text in task.files.items() if path.startswith("src/") and path not in ("src/__init__.py", "src/constants.py", "src/models.py"))
            if "for item in" in source and '["resource"]' in source:
                shapes.add("record")
            elif "for item in" in source:
                shapes.add("bundle")
            elif "for name, quantity in" in source:
                shapes.add("tuple")
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
        self.assertEqual(shapes, {"bundle", "record", "tuple"})

    def test_repository_content_is_unique_and_cross_block_disjoint(self):
        train = repo_tasks.POLICY_TRAIN_FAMILIES
        transfer = repo_tasks.POLICY_TRANSFER_FAMILIES
        blocks = {
            "bank": repo_tasks.make_tasks(train, 4, 87110, "policy_bank"),
            "calibration": repo_tasks.make_tasks(train, 4, 87200, "policy_calibration"),
            "dev": repo_tasks.make_tasks(transfer, 8, 87300, "policy_dev"),
            "confirm": repo_tasks.make_tasks(transfer, 8, 87400, "policy_confirm"),
        }
        digests = {
            name: [repo_tasks.content_digest(task) for task in tasks]
            for name, tasks in blocks.items()
        }
        for name, values in digests.items():
            self.assertEqual(len(values), len(set(values)), name)
        self.assertFalse(set(digests["bank"]) & set(digests["calibration"]))
        self.assertFalse(set(digests["dev"]) & set(digests["confirm"]))

    def test_counterexample_row_is_exact_near_correct_revision(self):
        task = repo_tasks.make_tasks(
            repo_tasks.POLICY_TRAIN_FAMILIES[:1], 1, 87603, "unit_bank"
        )[0]
        built = bank.build_banks([task], trajectories=None)
        rows = built["recovery_action_rows"]
        by_transition = {row["transition"]: row for row in rows}
        revision = by_transition["diagnosis_to_changed_patch"]
        self.assertIn(
            "negative",
            "\n".join(message["content"] for message in revision["messages"]).lower(),
        )
        self.assertEqual(revision["operator"], "PATCH")
        partial = task.partial_patches[0]
        replay = bank.replay_patch_set(task, [
            {"tool": "patch", "path": partial.path, "old": partial.old, "new": partial.new},
            revision["target_action"],
        ])
        self.assertTrue(replay["visible"] and replay["hidden"])
        self.assertEqual(
            Counter(row["transition"] for row in rows), Counter(bank.TRANSITIONS)
        )
        self.assertTrue(all(row["think_weight"] == 0.0 for row in rows))
        bank.assert_firewall_clean(rows, [task])

    def test_transition_mass_and_complete_training_supercycles(self):
        tasks = repo_tasks.make_tasks(repo_tasks.POLICY_TRAIN_FAMILIES[:2], 2, 87604, "unit")
        rows = bank.build_banks(tasks, trajectories=None)["recovery_action_rows"]
        receipt = bank.calibrate_transition_loss_mass(
            rows,
            CharacterTokenizer(),
            target_operator_action_mass=38248.0,
            plan_mass_fraction=0.0,
            max_length=20000,
        )
        self.assertEqual(
            {round(value, 7) for value in receipt["weighted_action_mass_by_operator"].values()},
            {38248.0},
        )
        training = load_script("train")
        encoded = []
        for task_index in range(5):
            for transition in bank.TRANSITIONS:
                encoded.append({
                    "task_id": f"task-{task_index}",
                    "row_id": f"task-{task_index}-{transition}",
                    "transition": transition,
                    "operator": bank.TRANSITION_OPERATOR[transition],
                    "family": "synthetic",
                    "input_ids": [1],
                    "attention_mask": [1],
                    "labels": [1],
                    "loss_weights": [1.0],
                    "answer_mask": [1.0],
                })
        ordered, batching = training.make_batches(encoded, 4, 7, 47)
        self.assertTrue(batching["every_optimizer_step_contains_all_transitions"])
        self.assertEqual(batching["whole_task_padding_duplicates"], 3)
        self.assertEqual(len(ordered), 8 * 7)

    def test_manifests_are_cross_process_hash_seed_stable(self):
        snippet = (
            "import json,sys;"
            f"sys.path.insert(0,{str(EXP / 'src')!r});"
            "import repo_tasks;"
            "t=repo_tasks.make_tasks(repo_tasks.POLICY_TRANSFER_FAMILIES,2,87300,'stable');"
            "print(repo_tasks.manifest_digest(t))"
        )
        outputs = []
        for seed in ("0", "271828"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
            outputs.append(subprocess.check_output([sys.executable, "-c", snippet], env=env, text=True).strip())
        self.assertEqual(outputs[0], outputs[1])

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

    def test_official_children_freeze_hash_seed_and_menagerie_seeds_are_fresh(self):
        runner = load_script("run")
        with patch.object(
            runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as mocked:
            runner.command(["synthetic-command"])
        self.assertEqual(mocked.call_args.kwargs["env"]["PYTHONHASHSEED"], "0")
        self.assertEqual(mocked.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        bench = load_script("bench")
        frozen = set(self.cfg["menagerie"]["paired_seeds"].values())
        self.assertEqual(len(frozen), 2)
        self.assertFalse(frozen & bench.used_seeds())


if __name__ == "__main__":
    unittest.main()
