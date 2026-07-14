from __future__ import annotations

import importlib.util
import tempfile
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


GATE = load_module("restart_local_gate", "check_local.py")
DESIGN = load_module("restart_local_design", "gen_local_gate.py")
EVAL = load_module("restart_local_eval", "eval_local_vllm.py")
MERGE = load_module("restart_trained_merge", "merge_trained_arm.py")


def make_rows(label: str, *, correct: int, target_correct: int) -> list[dict]:
    kinds = [kind for kind in sorted(GATE.EXPECTED_KINDS) for _ in range(2)]
    target_indexes = [
        kinds.index("u_execute"),
        kinds.index("u_induct"),
        kinds.index("u_probe"),
    ][:target_correct]
    other_indexes = [
        index
        for index, kind in enumerate(kinds)
        if kind not in GATE.TARGET_KINDS and index not in target_indexes
    ][: correct - target_correct]
    correct_indexes = set(target_indexes + other_indexes)
    return [
        {
            "adapter": label,
            "task_id": f"task-{index:02d}",
            "kind": kind,
            "parsed": "OK",
            "correct": index in correct_indexes,
            "cap_contact": False,
        }
        for index, kind in enumerate(kinds)
    ]


def passing_payload() -> dict:
    return {
        "seed": GATE.SEED,
        "rows_per_arm": GATE.ROWS,
        "labels": list(GATE.ARMS),
        "rows": [
            *make_rows(GATE.PARENT, correct=16, target_correct=2),
            *make_rows(GATE.CONTROL, correct=16, target_correct=2),
            *make_rows(GATE.CANDIDATE, correct=17, target_correct=3),
        ],
    }


class LocalDesignAndGateTests(unittest.TestCase):
    def test_fresh_design_has_two_rows_per_skill_and_hidden_free_input(self) -> None:
        source, runner = DESIGN.build_rows()
        self.assertEqual(len(source), 26)
        self.assertEqual(len(runner), 26)
        self.assertEqual(
            {row["kind"] for row in source},
            {f"u_{name}" for name in DESIGN.curriculum.SKILLS},
        )
        self.assertTrue(
            all(row["task_id"].startswith("local88010_") for row in source)
        )
        self.assertTrue(
            all(set(row) == {"id", "messages", "meta"} for row in runner)
        )
        self.assertNotIn("answer", runner[0])
        self.assertNotIn("think", runner[0])

    def test_candidate_must_pass_absolute_and_strict_relative_gates(self) -> None:
        payload = passing_payload()
        result = GATE.evaluate_promotion(payload)
        self.assertEqual(result["eligible"], [GATE.CANDIDATE])
        self.assertTrue(result["gates"][GATE.CANDIDATE]["passes"])
        for row in payload["rows"]:
            if row["adapter"] == GATE.CONTROL and not row["correct"]:
                row["correct"] = True
                break
        self.assertEqual(GATE.evaluate_promotion(payload)["eligible"], [])

    def test_gate_rejects_extra_duplicate_or_cross_arm_drift(self) -> None:
        payload = passing_payload()
        payload["rows"].append(dict(payload["rows"][0]))
        with self.assertRaisesRegex(ValueError, "exactly 78"):
            GATE.evaluate_promotion(payload)

        payload = passing_payload()
        payload["rows"][1]["task_id"] = payload["rows"][0]["task_id"]
        with self.assertRaisesRegex(ValueError, "schema changed"):
            GATE.evaluate_promotion(payload)

        payload = passing_payload()
        control = next(
            row
            for row in payload["rows"]
            if row["adapter"] == GATE.CONTROL and row["task_id"] == "task-00"
        )
        control["kind"] = "u_verify"
        with self.assertRaises(ValueError):
            GATE.evaluate_promotion(payload)

    def test_merge_surface_contains_only_frozen_trained_arms(self) -> None:
        self.assertEqual(
            set(MERGE.FROZEN_ADAPTERS),
            {"replay_control", "counterfactual_restart_candidate"},
        )

    def test_merged_tree_manifest_rejects_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MERGE.MERGED_FILE_NAMES:
                (root / name).write_bytes(name.encode())
            manifest = MERGE.merged_tree_manifest(root)
            self.assertEqual({row["name"] for row in manifest}, MERGE.MERGED_FILE_NAMES)
            self.assertEqual(len(MERGE.tree_manifest_sha256(manifest)), 64)
            (root / "unexpected.txt").write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                MERGE.merged_tree_manifest(root)

    def test_local_commands_share_vllm_protocol_and_forbid_runtime_lora(self) -> None:
        commands = {
            label: EVAL.command_for(label, EVAL.arm_paths(label))
            for label in EVAL.LABELS
        }
        normalized = []
        for label, command in commands.items():
            self.assertEqual(Path(command[2]).name, "vllm_runner.py")
            self.assertIn("--model-override", command)
            self.assertNotIn("--adapter", command)
            self.assertEqual(command[command.index("--seed") + 1], "88010")
            self.assertEqual(command[command.index("--max-tokens") + 1], "1024")
            normalized.append(
                [
                    "<MODEL>" if value == str(EVAL.MERGED[label]) else value
                    for value in command
                    if "seed88010_" not in value
                ]
            )
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])

    def test_answer_parser_uses_last_exact_answer_line(self) -> None:
        self.assertEqual(
            EVAL.parse_answer("x\nANSWER: first\ny\nANSWER: final\n"), "final"
        )
        self.assertIsNone(EVAL.parse_answer("no exact answer"))


if __name__ == "__main__":
    unittest.main()
