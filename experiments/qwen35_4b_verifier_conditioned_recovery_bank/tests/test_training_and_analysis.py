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
    spec = importlib.util.spec_from_file_location(f"repo_vcrb_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = load_script("train")
analyze = load_script("analyze_primary")
select = load_script("select_candidate")
evaluation = load_script("eval_repo_agent")
uncertainty = load_script("audit_transition_uncertainty")
run = load_script("run")


class TrainingAndAnalysisTests(unittest.TestCase):
    def test_orchestrator_accepts_only_explicit_expected_gate_codes(self):
        command = ["gate"]
        with mock.patch.object(
            run.subprocess, "run", return_value=subprocess.CompletedProcess(command, 4)
        ):
            self.assertEqual(run.run_command(command, allowed_returncodes=(0, 4)), 4)
            with self.assertRaises(subprocess.CalledProcessError):
                run.run_command(command)

    def test_checkpointed_cross_entropy_matches_dense_loss_and_gradient(self):
        generator = torch.Generator().manual_seed(11)
        logits_dense = torch.randn(2, 5, 7, generator=generator, requires_grad=True)
        logits_chunked = logits_dense.detach().clone().requires_grad_(True)
        labels = torch.tensor([[1, 2, 3, 4, -100], [0, 6, 2, 1, 5]])
        weights = torch.tensor([[1.0, 0.2, -0.3, 0.0, 0.0],
                                [1.0, 2.0, 0.5, 1.0, 0.1]])
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
        self.assertTrue(torch.allclose(
            logits_dense.grad, logits_chunked.grad, atol=1e-6, rtol=1e-6
        ))

    def test_batches_pad_whole_tasks_and_balance_every_optimizer_step(self):
        rows = []
        for task_index in range(5):
            for transition in train.TRANSITIONS:
                rows.append({
                    "input_ids": list(range(20)),
                    "task_id": f"task-{task_index}",
                    "row_id": f"task-{task_index}-{transition}",
                    "family": "family",
                    "operator": "PATCH",
                    "transition": transition,
                })
        ordered, receipt = train.make_batches(
            rows, batch_size=4, gradient_accumulation_steps=7, seed=42
        )
        self.assertEqual(receipt["original_tasks"], 5)
        self.assertEqual(receipt["whole_task_padding_duplicates"], 3)
        self.assertEqual(receipt["effective_tasks_per_epoch"], 8)
        microbatches = [ordered[index:index + 4] for index in range(0, len(ordered), 4)]
        self.assertTrue(all(len({row["transition"] for row in batch}) == 1
                            for batch in microbatches))
        for index in range(0, len(microbatches), 7):
            cycle = microbatches[index:index + 7]
            self.assertEqual(
                {batch[0]["transition"] for batch in cycle}, set(train.TRANSITIONS)
            )

    def test_candidate_score_prefers_success_then_worst_scenario_then_transition(self):
        def payload(overall, rejected, failed, transition):
            return {"aggregate": {
                "success": overall,
                "per_scenario": {
                    "rejected_patch": {
                        "success": rejected, "immediate_transition_rate": transition,
                    },
                    "failed_test": {
                        "success": failed, "immediate_transition_rate": transition,
                        "changed_patch_within_two": transition,
                    },
                },
            }}
        self.assertGreater(
            select.score(payload(0.5, 0.4, 0.6, 0.2)),
            select.score(payload(0.49, 0.49, 0.49, 1.0)),
        )
        self.assertGreater(
            select.score(payload(0.5, 0.45, 0.55, 0.1)),
            select.score(payload(0.5, 0.4, 0.6, 1.0)),
        )

    def test_paired_bootstrap_uses_casewise_differences(self):
        left = {f"t{i}": {"success": value} for i, value in enumerate([1, 1, 0, 1])}
        right = {f"t{i}": {"success": value} for i, value in enumerate([0, 1, 0, 0])}
        result = analyze.paired_delta(left, right, seed=7)
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 0)

    def test_evaluation_aggregates_union_success_by_task_scenario(self):
        base = {
            "family": "f", "scenario": "failed_test", "verified_after_final_patch": False,
            "commit_after_pass": False, "submitted": False, "sampled_tokens": 10,
            "turns": 2, "invalid_actions": 0, "rejected_patch_changed_immediately": False,
            "failed_test_diagnose_or_revise_immediately": True,
            "failed_test_changed_patch_within_two": False,
        }
        rows = [
            {**base, "case_id": "a::failed_test", "task_id": "a", "workspace_success": False},
            {**base, "case_id": "a::failed_test", "task_id": "a", "workspace_success": True,
             "verified_after_final_patch": True, "commit_after_pass": True,
             "submitted": True, "failed_test_changed_patch_within_two": True},
        ]
        result = evaluation.aggregate(rows, "sample_more")
        self.assertEqual(result["n_cases"], 1)
        self.assertEqual(result["success"], 1.0)
        self.assertEqual(result["per_scenario"]["failed_test"]["changed_patch_within_two"], 1.0)

    def test_entropy_and_varentropy_are_finite(self):
        metrics = uncertainty.distribution_metrics(
            torch.tensor([0.0, 1.0, -1.0, 0.5]), target=1
        )
        self.assertGreater(metrics["entropy_nats"], 0)
        self.assertGreater(metrics["varentropy_nats2"], 0)
        self.assertEqual(metrics["target_rank"], 1)

    def test_locality_contexts_are_frozen_and_unique(self):
        payload = json.loads((EXP / "data" / "locality_contexts.json").read_text())
        self.assertEqual(payload["seed"], 85000)
        self.assertEqual(payload["count"], 48)
        self.assertEqual(len({row["id"] for row in payload["contexts"]}), 48)
        self.assertEqual(len({row["content_sha256"] for row in payload["contexts"]}), 48)


if __name__ == "__main__":
    unittest.main()
