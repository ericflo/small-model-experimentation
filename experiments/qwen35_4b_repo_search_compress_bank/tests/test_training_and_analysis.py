from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"repo_scb_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = load_script("train")
analyze_repo = load_script("analyze_repo")
run = load_script("run")
summarize_primary = load_script("summarize_primary")


class TrainingAndAnalysisTests(unittest.TestCase):
    def test_orchestrator_accepts_only_explicit_expected_gate_codes(self):
        command = ["gate"]
        with mock.patch.object(
            run.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(command, 4),
        ):
            self.assertEqual(run.run_command(command, allowed_returncodes=(0, 4)), 4)
            with self.assertRaises(subprocess.CalledProcessError):
                run.run_command(command)

    def test_transition_diagnostics_distinguish_repair_from_rigid_retry(self):
        trajectories = [
            {
                "operator_sequence": ["INSPECT", "PATCH", "VERIFY", "PATCH"],
                "steps": [
                    {"observation": "source"},
                    {"observation": "PATCH_OK"},
                    {"observation": "FAIL(exit=1)"},
                    {"observation": "PATCH_OK"},
                ],
            },
            {
                "operator_sequence": ["INSPECT", "PATCH", "INSPECT", "PATCH"],
                "steps": [
                    {"observation": "source"},
                    {"observation": "ERROR: old text matched 0 times", "action": {"tool": "patch"}},
                    {"observation": "source"},
                    {"observation": "ERROR: old text matched 0 times", "action": {"tool": "patch"}},
                ],
            },
        ]
        result = summarize_primary.transition_diagnostics(trajectories)
        self.assertEqual(result["after_failed_test"]["next_patch_rate"], 1.0)
        self.assertEqual(result["tasks_repeating_identical_failed_patch"], 1)

    def test_checkpointed_cross_entropy_matches_dense_loss_and_gradient(self):
        generator = torch.Generator().manual_seed(11)
        logits_dense = torch.randn(2, 5, 7, generator=generator, requires_grad=True)
        logits_chunked = logits_dense.detach().clone().requires_grad_(True)
        labels = torch.tensor([[1, 2, 3, 4, -100], [0, 6, 2, 1, 5]])
        weights = torch.tensor([[1.0, 0.2, -0.3, 0.0, 0.0], [1.0, 2.0, 0.5, 1.0, 0.1]])
        active = (labels != -100).float()
        dense_losses = torch.nn.functional.cross_entropy(
            logits_dense.reshape(-1, 7), labels.reshape(-1).clamp(min=0), reduction="none"
        ).view_as(labels)
        dense = (dense_losses * weights * active).sum()
        chunked = train.checkpointed_weighted_cross_entropy(
            logits_chunked, labels, weights * active, chunk_positions=3
        )
        dense.backward()
        chunked.backward()
        self.assertTrue(torch.allclose(dense, chunked, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(logits_dense.grad, logits_chunked.grad, atol=1e-6, rtol=1e-6))

    def test_batches_keep_apex_and_four_operator_repository_rows_separate(self):
        apex = [
            {"source": "apex", "input_ids": list(range(10 + index)), "source_code": 0}
            for index in range(5)
        ]
        repo = [
            {"source": "repo", "input_ids": list(range(20)), "source_code": 1,
             "task_id": "task-1", "operator": operator}
            for operator in train.OPERATORS
        ]
        ordered, receipt = train.make_batches(
            apex + repo, batch_size=4, gradient_accumulation_steps=4, seed=42
        )
        chunks = [ordered[index:index + 4] for index in range(0, len(ordered), 4)]
        self.assertTrue(all(len({row["source"] for row in chunk}) == 1 for chunk in chunks))
        repo_chunk = next(chunk for chunk in chunks if chunk[0]["source"] == "repo")
        self.assertEqual({row["operator"] for row in repo_chunk}, set(train.OPERATORS))
        self.assertEqual(receipt["apex_padding_duplicates"], 11)

    def test_two_row_microbatches_preserve_effective_batch_and_task_blocks(self):
        apex = [
            {"source": "apex", "input_ids": list(range(10)), "source_code": 0}
            for _ in range(13)
        ]
        repo = [
            {"source": "repo", "input_ids": list(range(20)), "source_code": 1,
             "task_id": "task-1", "operator": operator}
            for operator in train.OPERATORS
        ]
        ordered, receipt = train.make_batches(
            apex + repo, batch_size=2, gradient_accumulation_steps=8, seed=42
        )
        chunks = [ordered[index:index + 2] for index in range(0, len(ordered), 2)]
        self.assertTrue(all(len({row["source"] for row in chunk}) == 1 for chunk in chunks))
        self.assertEqual(receipt["effective_batch_size"], 16)
        self.assertEqual(receipt["apex_padding_duplicates"], 3)
        self.assertEqual(receipt["repository_microbatches_per_epoch"], 2)

    def test_paired_bootstrap_uses_taskwise_differences(self):
        left = {f"t{i}": {"success": value} for i, value in enumerate([1, 1, 0, 1])}
        right = {f"t{i}": {"success": value} for i, value in enumerate([0, 1, 0, 0])}
        result = analyze_repo.paired_delta(left, right, seed=7)
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 0)

    def test_locality_contexts_are_frozen_and_unique(self):
        payload = json.loads((EXP / "data" / "locality_contexts.json").read_text())
        self.assertEqual(payload["count"], 48)
        self.assertEqual(len({row["id"] for row in payload["contexts"]}), 48)
        self.assertEqual(len({row["content_sha256"] for row in payload["contexts"]}), 48)


if __name__ == "__main__":
    unittest.main()
