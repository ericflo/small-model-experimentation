from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import mechanics_stage as stage  # noqa: E402
import mechanics_protocol as mechanics  # noqa: E402
from calibration_stage import load_calibration_inputs  # noqa: E402
from transactions import read_canonical  # noqa: E402


class MechanicsStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = read_canonical(EXP / "runs/calibration/decision.json")

    def test_selected_sampling_is_no_think_tokenizer_winner_with_frozen_seeds(self) -> None:
        plan = stage.mechanics_sampling_plan(self.decision, self.inputs)
        self.assertEqual(tuple(plan), stage.MECHANICS_INVOCATION_ORDER)
        self.assertEqual(plan["transport"]["run_seed"], 2026140706)
        self.assertEqual(plan["suffix_materialized"]["run_seed"], 2026140702)
        self.assertEqual(plan["direct"]["run_seed"], 2026140703)
        for value in plan.values():
            self.assertEqual(value["thinking"], "off")
            self.assertEqual(value["answer_prefix"], "PROGRAM:")
            self.assertEqual(value["max_tokens"], 24)
            self.assertTrue(value["paired_answer_seed"])

    def test_transport_scoring_accepts_terminal_token_but_rejects_extra_content(self) -> None:
        rows = []
        for index in range(24):
            arity = 2 if index % 2 == 0 else 3
            expected = "PROGRAM: A | B" + (" | C" if arity == 3 else "")
            rows.append(
                {
                    "id": f"row-{index:02d}",
                    "meta": {
                        "task_id": f"task-{index:02d}",
                        "arity": arity,
                        "expected": expected,
                    },
                    "outputs": [
                        {
                            "text": expected + "<|im_end|>",
                            "answer_cap_contact": False,
                        }
                    ],
                }
            )
        metrics = stage._score_transport_rows(rows)
        self.assertEqual(metrics["exact_echo_successes"], 24)
        self.assertEqual(metrics["parse_successes"], 24)
        forged = json.loads(json.dumps(rows))
        forged[0]["outputs"][0]["text"] = "PROGRAM: A | B\n<|im_end|>"
        bad = stage._score_transport_rows(forged)
        self.assertEqual(bad["exact_echo_successes"], 23)
        self.assertEqual(bad["parse_successes"], 23)

    def test_selector_tie_break_is_canonical_program_hash_not_row_order(self) -> None:
        task = {
            "task_id": "tie-task",
            "depth": 3,
            "visible": [{"input": [1, 2], "output": [1, 2]}],
            "unlabeled_probe_inputs": [[-2, 1]],
        }
        candidates = [
            {
                "candidate_id": "row-z",
                "candidate": None,
                "text": "PROGRAM: B | B | B<|im_end|>",
            },
            {
                "candidate_id": "row-a",
                "candidate": None,
                "text": "PROGRAM: D | D | D<|im_end|>",
            },
        ]
        forward = mechanics.select_visible(task, candidates, thinking_expected=False)
        reverse = mechanics.select_visible(
            task, list(reversed(candidates)), thinking_expected=False
        )
        expected = min(
            (("B", "B", "B"), ("D", "D", "D")),
            key=lambda aliases: (mechanics._selector_hash("tie-task", aliases), aliases),
        )
        self.assertEqual(forward["selected_program_id"], reverse["selected_program_id"])
        self.assertEqual(
            forward["selected_program_id"], mechanics.canonical_program_id(expected)
        )

    def test_hidden_scoring_keeps_selector_primary_and_coverage_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            public_path = Path(directory) / "public.jsonl"
            program = (("reverse", None), ("reverse", None), ("reverse", None))
            visible_tasks = {}
            gold = []
            public = []
            for index in range(24):
                task_id = f"task-{index:02d}"
                selection = {
                    "selected_candidate_id": "candidate",
                    "scored": [
                        {"candidate_id": "candidate", "full_program": program}
                    ],
                }
                visible_tasks[task_id] = {
                    "selections": {
                        arm: selection
                        for arm in (
                            "materialized",
                            "name_only",
                            "shuffled",
                            "direct_sampled",
                            "direct_logical",
                        )
                    }
                }
                gold.append(
                    {
                        "task_id": task_id,
                        "hidden": [{"input": [1, 2, 3], "output": [3, 2, 1]}],
                    }
                )
                public.append(
                    {
                        "task_id": task_id,
                        "depth": 3,
                        "visible": [{"input": [1, 2, 3], "output": [3, 2, 1]}],
                        "unlabeled_probe_inputs": [[4, 5, 6]],
                    }
                )
            public_path.write_text(
                "".join(json.dumps(row) + "\n" for row in public)
            )
            result = stage.score_hidden(
                visible={
                    "schema_version": 1,
                    "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                    "winner": stage.SELECTED_INTERFACE,
                    "generation_abi_pass": True,
                    "generation_metrics": {},
                    "generation_authentication": {},
                    "selector_uses_hidden": False,
                    "tasks": visible_tasks,
                    "transaction_chain": {},
                    "public_sha256": hashlib.sha256(
                        public_path.read_bytes()
                    ).hexdigest(),
                    "hidden_files_read": [],
                    "benchmark_files_read": [],
                },
                gold_rows=gold,
                gold_receipt={
                    "algorithm": "AES-256-GCM",
                    "aad_utf8": (
                        "tokenizer-eos-answer-commit-factorial-v1/mechanics-gold-v1"
                    ),
                    "ciphertext_sha256": "b" * 64,
                    "plaintext_sha256": "a" * 64,
                    "local_key_sha256": "c" * 64,
                    "hidden_files_read": ["ciphertext", "key"],
                },
                public_path=public_path,
                config=self.inputs.config,
                program_inventory=(program,),
            )
            self.assertEqual(result["primary_selected_accuracy"]["materialized"], 1.0)
            self.assertEqual(
                result["oracle_proposal_coverage_diagnostic"]["materialized"], 1.0
            )
            self.assertEqual(
                result["decision"],
                "TOKENIZER_EOS_MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL",
            )
            self.assertEqual(
                result["report_only_exhaustive_cpu_ceiling"]["coverage"], 1.0
            )
            self.assertEqual(result["hidden_files_read"], ["ciphertext", "key"])


if __name__ == "__main__":
    unittest.main()
