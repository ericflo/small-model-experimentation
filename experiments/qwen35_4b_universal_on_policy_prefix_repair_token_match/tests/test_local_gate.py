from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    path = EXP / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load_module("prefix_local_gate", "check_local.py")
DESIGN = load_module("prefix_local_design", "gen_local_gate.py")
EVAL = load_module("prefix_local_eval", "eval_local_vllm.py")
MERGE = load_module("prefix_trained_merge", "merge_trained_arm.py")


def make_rows(label: str, *, correct: int, target_correct: int) -> list[dict]:
    kinds = [
        "u_execute",
        "u_execute",
        "u_induct",
        "u_induct",
        "u_probe",
        "u_probe",
        "u_route",
        "u_route",
        *["u_count"] * 18,
    ]
    target_indexes = set((0, 2, 4)[:target_correct])
    remaining = correct - target_correct
    other_indexes = set(range(6, 6 + remaining))
    return [
        {
            "adapter": label,
            "kind": kind,
            "parsed": "OK",
            "correct": index in target_indexes or index in other_indexes,
            "cap_contact": False,
        }
        for index, kind in enumerate(kinds)
    ]


class LocalDesignAndGateTests(unittest.TestCase):
    def test_fresh_design_has_two_rows_per_skill_and_hidden_free_input(self) -> None:
        source, runner = DESIGN.build_rows()
        self.assertEqual(len(source), 26)
        self.assertEqual(len(runner), 26)
        self.assertEqual({row["kind"] for row in source}, {f"u_{x}" for x in DESIGN.curriculum.SKILLS})
        self.assertTrue(all(row["task_id"].startswith("local88009_") for row in source))
        self.assertTrue(all(set(row) == {"id", "messages", "meta"} for row in runner))
        self.assertNotIn("answer", runner[0])
        self.assertNotIn("think", runner[0])

    def test_boundary_absolute_gate_passes(self) -> None:
        payload = {
            "seed": 88009,
            "rows_per_arm": 26,
            "rows": make_rows(GATE.CANDIDATE, correct=17, target_correct=3),
        }
        result = GATE.absolute_gate(payload, GATE.CANDIDATE)
        self.assertTrue(result["passes"])
        self.assertTrue(all(result["checks"].values()))

    def test_candidate_must_strictly_beat_both_controls_total_and_target(self) -> None:
        payload = {
            "seed": 88009,
            "rows_per_arm": 26,
            "rows": [
                *make_rows(GATE.PARENT, correct=16, target_correct=2),
                *make_rows(GATE.CONTROL, correct=16, target_correct=2),
                *make_rows(GATE.CANDIDATE, correct=17, target_correct=3),
            ],
        }
        self.assertTrue(GATE.evaluate_promotion(payload)["eligible"])
        for row in payload["rows"]:
            if row["adapter"] == GATE.CONTROL and not row["correct"]:
                row["correct"] = True
                break
        self.assertEqual(GATE.evaluate_promotion(payload)["eligible"], [])

    def test_merge_surface_contains_only_frozen_trained_arms(self) -> None:
        self.assertEqual(
            set(MERGE.FROZEN_ADAPTERS),
            {"replay_after_close", "prefix_repair_after_close"},
        )

    def test_local_commands_share_vllm_protocol_and_never_use_runtime_lora(self) -> None:
        commands = {
            label: EVAL.command_for(label, EVAL.arm_paths(label)) for label in EVAL.LABELS
        }
        normalized = []
        for label, command in commands.items():
            self.assertIn("vllm_runner.py", Path(command[2]).name)
            self.assertIn("--model-override", command)
            self.assertNotIn("--adapter", command)
            self.assertEqual(command[command.index("--seed") + 1], "88009")
            self.assertEqual(command[command.index("--max-tokens") + 1], "1024")
            normalized.append(
                [
                    "<MODEL>" if value == str(EVAL.MERGED[label]) else value
                    for value in command
                    if "seed88009_" not in value
                ]
            )
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])

    def test_answer_parser_uses_last_exact_answer_line(self) -> None:
        self.assertEqual(EVAL.parse_answer("x\nANSWER: first\ny\nANSWER: final\n"), "final")
        self.assertIsNone(EVAL.parse_answer("no exact answer"))


if __name__ == "__main__":
    unittest.main()
