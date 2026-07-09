from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("macro_analyze", SCRIPT)
assert SPEC and SPEC.loader
analyze = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyze
SPEC.loader.exec_module(analyze)


PRIMITIVES = {"ADD1": "increment", "CONST_ONE": "replace by one", "NOOP": "identity"}


def fake_execute(program, value):
    current = value
    for token in program:
        if token == "ADD1":
            current += 1
        elif token == "CONST_ONE":
            current = 1
        elif token == "NOOP":
            pass
        else:
            raise ValueError(f"unknown token {token}")
    return current


def fake_expand(program, macros):
    expanded = []
    for token in program:
        expanded.extend(macros.get(token, (token,)))
    if any(token not in PRIMITIVES for token in expanded):
        raise ValueError("unknown expanded token")
    return tuple(expanded)


def patched_domain():
    stack = ExitStack()
    stack.enter_context(mock.patch.object(analyze.domain, "PRIMITIVES", PRIMITIVES))
    stack.enter_context(
        mock.patch.object(analyze.domain, "execute_program", fake_execute, create=True)
    )
    stack.enter_context(
        mock.patch.object(analyze.domain, "expand_program", fake_expand, create=True)
    )
    return stack


def output(sample_index: int, text: str, *, sampled: int = 2, forced=False, truncated=False):
    token_ids = [10, 11]
    return {
        "sample_index": sample_index,
        "text": text,
        "token_ids": token_ids,
        "n_stage1_prompt_tokens": 5,
        "n_stage2_prompt_tokens": 0,
        "n_sampled_tokens": sampled,
        "n_injected_tokens": 0,
        "n_completion_tokens": len(token_ids),
        "n_thinking_tokens": 0,
        "n_answer_tokens": len(token_ids),
        "n_terminal_tokens_trimmed": 0,
        "forced_close": forced,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
    }


def summary(rows):
    requests = len(rows)
    completions = sum(len(row["outputs"]) for row in rows)
    unique = sum(row["n_prompt_tokens"] for row in rows)
    stage1 = sum(
        item["n_stage1_prompt_tokens"] for row in rows for item in row["outputs"]
    )
    stage2 = sum(
        item["n_stage2_prompt_tokens"] for row in rows for item in row["outputs"]
    )
    return {
        "model": analyze.harness.REQUIRED_MODEL_ID,
        "model_revision": analyze.harness.MODEL_REVISION,
        "counts": {
            "requests": requests,
            "completions": completions,
            "unique_input_prompt_tokens": unique,
            "stage1_logical_prompt_tokens": stage1,
            "stage2_logical_prompt_tokens": stage2,
            "logical_model_input_tokens": stage1 + stage2,
            "sampled_tokens": sum(
                item["n_sampled_tokens"] for row in rows for item in row["outputs"]
            ),
            "injected_tokens": 0,
        },
    }


def normalized_task(task_id="t0", split="reuse"):
    return {
        "id": task_id,
        "split": split,
        "program": ("ADD1", "NOOP"),
        "min_depth": 2,
        "visible": [{"input": 0, "output": 1}],
        "hidden": [{"input": 2, "output": 3}],
        "probe": [{"input": 4, "output": 5}],
        "paired_task_id": None,
        "motif_names": ["INC_ID"],
        "program_signature": f"sig-{task_id}",
    }


class CandidateSelectionTests(unittest.TestCase):
    def test_hidden_labels_never_break_visible_ties(self):
        task = normalized_task()
        library = {
            "id": "lib-mined",
            "provenance": "test",
            "macros": [
                {
                    "token": "M0",
                    "expansion": ("ADD1", "NOOP"),
                    "support": 4,
                    "length": 2,
                    "source_name": None,
                }
            ],
            "draw_seed": None,
        }
        rows = [
            {
                "id": "t0::mined",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "mined",
                    "library_id": "lib-mined",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(0, "PROGRAM: CONST_ONE | NOOP"),
                    output(1, "PROGRAM: M0"),
                    output(2, "This is not a program."),
                ],
            }
        ]
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="mined",
                rows=rows,
                summary=summary(rows),
                tasks={"t0": task},
                library=library,
            )
        result = by_task["t0"]
        self.assertEqual(result["selected_sample_index"], 0)
        self.assertFalse(result["selected_hidden_pass"])
        self.assertTrue(result["oracle_hidden_pass"])
        self.assertFalse(result["selected_macro_used"])
        self.assertTrue(result["candidates"][1]["macro_used"])
        self.assertFalse(result["candidates"][2]["parsed"])
        metrics = analyze.summarize_task_rows(by_task)
        self.assertAlmostEqual(metrics["parse_rate"], 2 / 3)
        self.assertEqual(metrics["false_visible_pass_count"], 1)

    def test_token_match_uses_first_no_smaller_base_prefix(self):
        def candidate(index, tokens, visible, hidden):
            return {
                "sample_index": index,
                "valid": True,
                "visible_score": visible,
                "sampled_tokens": tokens,
                "hidden_pass": hidden,
            }

        base = {
            "task_id": "t",
            "split": "reuse",
            "unique_prompt_tokens": 5,
            "candidates": [
                candidate(0, 6, 1.0, False),
                candidate(1, 6, 0.0, True),
                candidate(2, 6, 1.0, True),
                candidate(3, 6, 1.0, True),
            ],
        }
        treatment = {
            "unique_prompt_tokens": 5,
            "candidates": [candidate(0, 15, 1.0, True)],
        }
        matched = analyze.token_matched_base_prefix(base, treatment)
        self.assertTrue(matched["matched"])
        self.assertEqual(matched["prefix_k"], 3)
        # Earliest visible tie, not hidden correctness, remains selected.
        self.assertEqual(matched["base_prefix_selected_sample_index"], 0)
        self.assertFalse(matched["base_prefix_selected_hidden_pass"])


class StatisticsTests(unittest.TestCase):
    def test_paired_and_hierarchical_bootstraps_are_deterministic(self):
        paired = analyze.paired_bootstrap([1, 1, 1], [0, 0, 0], repetitions=50, seed=9)
        self.assertEqual(paired["point_delta"], 1.0)
        self.assertEqual(paired["ci95"], [1.0, 1.0])
        hierarchical = analyze.hierarchical_random_bootstrap(
            {"a": 1.0, "b": 1.0},
            {
                "random_0": {"a": 0.0, "b": 0.0},
                "random_1": {"a": 0.0, "b": 0.0},
            },
            repetitions=50,
            seed=4,
        )
        self.assertEqual(hierarchical["point_delta"], 1.0)
        self.assertEqual(hierarchical["ci95"], [1.0, 1.0])

    def test_machine_verdict_requires_and_can_clear_all_conjunctive_gates(self):
        def task_row(task_id, split, correct, macro):
            candidate = {
                "sample_index": 0,
                "parsed": True,
                "valid": True,
                "visible_score": 1.0,
                "visible_pass": True,
                "hidden_pass": correct,
                "macro_used": macro,
                "macro_tokens": ["M0"] if macro else [],
                "surface_depth": 1,
                "expanded_depth": 2,
                "sampled_tokens": 2,
                "injected_tokens": 0,
                "thinking_tokens": 1,
                "answer_tokens": 1,
                "stage1_logical_prompt_tokens": 5,
                "stage2_logical_prompt_tokens": 0,
                "forced_close": False,
                "truncated": False,
                "selected": True,
            }
            return {
                "task_id": task_id,
                "split": split,
                "arm": "placeholder",
                "library_id": "placeholder",
                "target_program": ["ADD1", "NOOP"],
                "target_min_depth": 2,
                "motif_names": ["INC_ID"] if split == "reuse" else [],
                "unique_prompt_tokens": 3,
                "n_samples": 1,
                "candidates": [candidate],
                "selected_sample_index": 0,
                "abstained": False,
                "selected_hidden_pass": correct,
                "selected_visible_pass": True,
                "selected_macro_used": macro,
                "selected_surface_depth": 1,
                "selected_expanded_depth": 2,
                "oracle_hidden_pass": correct,
            }

        task_splits = {**{f"r{i}": "reuse" for i in range(6)}, **{f"n{i}": "no_reuse" for i in range(3)}}
        arm_names = [
            "base",
            "mined",
            "mined_hint",
            "designed_ceiling",
            "random_0",
            "random_1",
            "random_2",
            "random_3",
            "random_4",
        ]
        arms = {}
        for arm in arm_names:
            rows = {}
            for task_id, split in task_splits.items():
                correct = split == "reuse" and arm in {"mined", "designed_ceiling"}
                macro = arm not in {"base", "mined_hint"}
                row = task_row(task_id, split, correct, macro)
                row["arm"] = arm
                row["library_id"] = f"lib-{arm}"
                rows[task_id] = row
            arms[arm] = rows
        summaries = {arm: analyze.summarize_arm(rows) for arm, rows in arms.items()}
        macro = {
            "token": "M0",
            "expansion": ("ADD1", "NOOP"),
            "support": 7,
            "length": 2,
            "source_name": None,
        }
        libraries = {
            arm: {
                "id": f"lib-{arm}",
                "provenance": "fixture",
                "macros": [] if arm == "base" else [dict(macro)],
                "draw_seed": None,
            }
            for arm in arm_names
        }
        decision = {
            "primary_min_delta": 0.10,
            "callable_vs_hint_min_delta": 0.05,
            "mined_vs_random_min_delta": 0.05,
            "forced_close_contingency": 0.50,
            "contingency_thinking_budget": 1536,
            "contingency_tasks": 20,
        }
        with tempfile.TemporaryDirectory() as temporary:
            verdict = analyze.build_full_verdict(
                exp=Path(temporary),
                arms=arms,
                summaries=summaries,
                libraries=libraries,
                decision=decision,
                expected_arms=arm_names,
                repetitions=50,
                seed=11,
            )
        self.assertTrue(verdict["system_benefit"])
        self.assertTrue(verdict["callable_chunking"])
        self.assertTrue(verdict["learned_recurrence"])
        self.assertTrue(verdict["claimable_complete_callable_abstraction"])
        self.assertEqual(verdict["status"], "complete_callable_abstraction_supported")


class SmokeVerdictTests(unittest.TestCase):
    @staticmethod
    def candidate(
        index: int,
        *,
        hidden: bool = False,
        macro: bool = False,
        truncated: bool = False,
    ):
        return {
            "sample_index": index,
            "parsed": not truncated,
            "valid": not truncated,
            "macro_used": macro,
            "hidden_pass": hidden,
            "truncated": truncated,
        }

    @staticmethod
    def task(split: str, candidates):
        return {"split": split, "candidates": list(candidates)}

    @staticmethod
    def decision():
        return {
            "smoke_matched_k": 12,
            "smoke_min_parse_rate": 0.50,
            "smoke_min_macro_candidates": 2,
            "smoke_max_answer_truncation": 0.05,
        }

    @staticmethod
    def summaries():
        return {"base": {"all": {}}, "designed_ceiling": {"all": {}}}

    def test_smoke_gate_uses_matched_prefix_instead_of_larger_base_k(self):
        arms = {"base": {}, "designed_ceiling": {}}
        for task_index in range(2):
            base_candidates = [self.candidate(index) for index in range(24)]
            # These completions would make the unmatched K=24 base look better and
            # badly truncated. Neither may enter the registered K=12 comparison.
            base_candidates[20] = self.candidate(20, hidden=task_index == 0)
            for index in range(12, 24):
                if index != 20:
                    base_candidates[index] = self.candidate(index, truncated=True)
            designed_candidates = [
                self.candidate(index, macro=index == 0) for index in range(12)
            ]
            task_id = f"reuse-{task_index}"
            arms["base"][task_id] = self.task("smoke_reuse", base_candidates)
            arms["designed_ceiling"][task_id] = self.task(
                "smoke_reuse", designed_candidates
            )

        verdict = analyze.build_smoke_verdict(
            arms, self.summaries(), self.decision()
        )
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["metrics"]["matched_k"], 12)
        self.assertEqual(
            verdict["metrics"]["base_oracle_hidden_coverage_reuse"], 0.0
        )
        self.assertEqual(verdict["metrics"]["overall_truncation_rate"], 0.0)
        self.assertEqual(
            verdict["metrics"]["designed_valid_macro_using_reuse_tasks"], 2
        )

    def test_smoke_gate_does_not_pool_no_reuse_or_macro_use(self):
        arms = {"base": {}, "designed_ceiling": {}}
        for split in ("smoke_reuse", "smoke_no_reuse"):
            for task_index in range(2):
                task_id = f"{split}-{task_index}"
                base_candidates = [
                    self.candidate(index, hidden=split == "smoke_reuse" and index == 0)
                    for index in range(12)
                ]
                designed_candidates = [
                    self.candidate(
                        index,
                        hidden=split == "smoke_no_reuse" and index == 0,
                        macro=split == "smoke_no_reuse" and index == 0,
                    )
                    for index in range(12)
                ]
                arms["base"][task_id] = self.task(split, base_candidates)
                arms["designed_ceiling"][task_id] = self.task(
                    split, designed_candidates
                )

        verdict = analyze.build_smoke_verdict(
            arms, self.summaries(), self.decision()
        )
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["gates"]["designed_oracle_not_below_base"])
        self.assertFalse(
            verdict["gates"]["designed_valid_macro_reuse_tasks"]
        )
        self.assertEqual(
            verdict["metrics"]["base_oracle_hidden_coverage_reuse"], 1.0
        )
        self.assertEqual(
            verdict["metrics"]["designed_oracle_hidden_coverage_reuse"], 0.0
        )
        self.assertEqual(
            verdict["metrics"]["base_oracle_hidden_coverage_no_reuse"], 0.0
        )
        self.assertEqual(
            verdict["metrics"]["designed_oracle_hidden_coverage_no_reuse"], 1.0
        )
        self.assertEqual(
            verdict["metrics"]["designed_valid_macro_using_reuse_tasks"], 0
        )


class ArtifactIntegrationTests(unittest.TestCase):
    def test_smoke_artifacts_analyze_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            exp = Path(temporary)
            (exp / "configs").mkdir()
            (exp / "data").mkdir()
            (exp / "src").mkdir()
            (exp / "runs" / "smoke").mkdir(parents=True)
            (exp / "analysis").mkdir()
            runner_bytes = b"# fixture vLLM runner identity\n"
            (exp / "src" / "vllm_runner.py").write_bytes(runner_bytes)
            (exp / "configs" / "default.yaml").write_text(
                """inference:\n  arms: [base, designed_ceiling]\ndecision:\n  bootstrap_repetitions: 20\n  bootstrap_seed: 7\n  smoke_matched_k: 1\n  smoke_min_parse_rate: 0.50\n  smoke_min_macro_candidates: 2\n  smoke_max_answer_truncation: 0.05\n""",
                encoding="utf-8",
            )
            tasks = []
            for index in range(2):
                task = normalized_task(f"s{index}", split="smoke_reuse")
                tasks.append(
                    {
                        **task,
                        "program": list(task["program"]),
                    }
                )
            (exp / "data" / "tasks.json").write_text(
                json.dumps({"schema_version": 1, "dataset_manifest": {}, "tasks": tasks}),
                encoding="utf-8",
            )
            macro = {
                "token": "M0",
                "expansion": ["ADD1", "NOOP"],
                "support": 5,
                "length": 2,
            }
            libraries = {
                "base": {"id": "lib-base", "provenance": "none", "macros": []},
                "designed_ceiling": {
                    "id": "lib-designed",
                    "provenance": "generator",
                    "macros": [macro],
                },
            }
            (exp / "data" / "libraries.json").write_text(
                json.dumps({"schema_version": 1, "libraries": libraries}), encoding="utf-8"
            )
            for arm, program, library_id in (
                ("base", "PROGRAM: CONST_ONE | NOOP", "lib-base"),
                ("designed_ceiling", "PROGRAM: M0", "lib-designed"),
            ):
                rows = []
                for index in range(2):
                    rows.append(
                        {
                            "id": f"s{index}::{arm}",
                            "meta": {
                                "task_id": f"s{index}",
                                "split": "smoke_reuse",
                                "arm": arm,
                                "library_id": library_id,
                                "max_surface_calls": 5,
                                "max_expanded_primitive_depth": 5,
                            },
                            "n_prompt_tokens": 5,
                            "outputs": [output(0, program)],
                        }
                    )
                (exp / "runs" / "smoke" / f"{arm}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                (exp / "runs" / "smoke" / f"{arm}.meta.json").write_text(
                    json.dumps(
                        {
                            **summary(rows),
                            "schema_version": analyze.local_vllm.RUNNER_SCHEMA_VERSION,
                            "runner_sha256": hashlib.sha256(runner_bytes).hexdigest(),
                            "engine": {"fixture": True},
                            "sampling": {"thinking": "budget"},
                        }
                    ),
                    encoding="utf-8",
                )
            with patched_domain():
                result = analyze.analyze_experiment(
                    exp=exp, run="smoke", bootstrap_repetitions=20, write=True
                )
            self.assertTrue(result["smoke_gate"]["pass"])
            self.assertTrue((exp / "analysis" / "smoke_per_task.csv").exists())
            self.assertTrue((exp / "analysis" / "smoke_per_task.json").exists())
            machine = json.loads((exp / "analysis" / "smoke_verdict.json").read_text())
            self.assertTrue(machine["smoke_gate"]["pass"])


if __name__ == "__main__":
    unittest.main()
