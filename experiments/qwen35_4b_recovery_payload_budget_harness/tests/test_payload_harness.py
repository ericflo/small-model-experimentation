from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"payload_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluation = load_script("eval_repo_agent")
locality = load_script("audit_locality")


def output(text: str, *, answer_tokens: int = 16) -> dict:
    return {
        "text": text,
        "n_thinking_tokens": 8,
        "n_answer_tokens": answer_tokens,
        "n_sampled_tokens": 8 + answer_tokens,
        "thinking_closed": True,
        "forced_close": False,
    }


def action_output(action: dict) -> dict:
    return output("</think>\n" + json.dumps(action, separators=(",", ":")))


class PayloadHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        cls.task = repo_tasks.make_tasks(("retry_backoff",), 1, 85320, "test")[0]

    def test_budget_and_candidate_are_frozen(self) -> None:
        self.assertEqual(self.cfg["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(self.cfg["evaluation"]["think_budget"], 512)
        self.assertEqual(self.cfg["evaluation"]["answer_max_tokens"], 512)
        self.assertIn("reason_mix_018", self.cfg["model"]["candidate"])
        per_call = 1024
        self.assertEqual(self.cfg["evaluation"]["recovery"]["deep_turns"] * per_call, 6144)
        self.assertEqual(
            self.cfg["evaluation"]["recovery"]["sample_more_trajectories"]
            * self.cfg["evaluation"]["recovery"]["sample_more_turns_each"]
            * per_call,
            6144,
        )

    def test_inspect_then_changed_patch_is_valid_recovery(self) -> None:
        episode = repo_agent.Episode(self.task, 0, scenario="rejected_patch")
        patch = self.task.oracle_patches[0]
        episode.consume(action_output({"tool": "read", "path": patch.path}))
        episode.consume(action_output({
            "tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new,
        }))
        result = episode.finish()
        self.assertFalse(result["rejected_patch_changed_immediately"])
        self.assertTrue(result["rejected_patch_changed_within_two"])
        self.assertTrue(result["rejected_patch_valid_changed_within_two"])
        self.assertEqual(result["rejected_patch_first_two_operators"], ["INSPECT", "PATCH"])

    def test_invalid_then_patch_does_not_satisfy_valid_transition(self) -> None:
        episode = repo_agent.Episode(self.task, 1, scenario="rejected_patch")
        patch = self.task.oracle_patches[0]
        episode.consume(output("</think>\n{\"tool\":\"patch\"", answer_tokens=512))
        episode.consume(action_output({
            "tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new,
        }))
        result = episode.finish()
        self.assertTrue(result["rejected_patch_changed_within_two"])
        self.assertFalse(result["rejected_patch_valid_changed_within_two"])
        aggregate = evaluation.aggregate([result], "deep", answer_max_tokens=512)
        self.assertEqual(aggregate["invalid_action_rate_per_turn"], 0.5)
        self.assertEqual(aggregate["answer_cap_hit_rate_per_turn"], 0.5)
        self.assertEqual(aggregate["invalid_answer_cap_hit_fraction"], 1.0)
        self.assertEqual(
            aggregate["per_scenario"]["rejected_patch"]["valid_changed_patch_within_two"],
            0.0,
        )

    def test_fresh_locality_is_disjoint_and_uncertainty_finite(self) -> None:
        current = json.loads((EXP / "data" / "locality_contexts.json").read_text())
        current_hashes = {row["content_sha256"] for row in current["contexts"]}
        self.assertEqual(len(current_hashes), 48)
        for name in ("locality_screen.json", "locality_confirm.json"):
            prior = json.loads(
                (ROOT / "experiments/qwen35_4b_recovery_reason_locality_interpolation/data" / name).read_text()
            )
            self.assertFalse(current_hashes & {row["content_sha256"] for row in prior["contexts"]})
        entropy, varentropy = locality.uncertainty(
            __import__("torch").tensor([0.3, -0.7, 1.0])
        )
        self.assertGreater(entropy, 0.0)
        self.assertGreaterEqual(varentropy, 0.0)


if __name__ == "__main__":
    unittest.main()
