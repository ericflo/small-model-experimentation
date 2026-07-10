from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
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


def output(
    sample_index: int,
    text: str,
    *,
    sampled: int = 2,
    forced: bool = False,
    truncated: bool = False,
    thinking_tokens: int = 0,
    answer_tokens: int | None = None,
    stage1_finish_reason: str = "stop",
    stage1_token_ids=None,
    retained_thinking_token_ids=None,
):
    token_ids = [10, 11]
    result = {
        "sample_index": sample_index,
        "text": text,
        "token_ids": token_ids,
        "n_stage1_prompt_tokens": 5,
        "n_stage2_prompt_tokens": 0,
        "n_sampled_tokens": sampled,
        "n_injected_tokens": 0,
        "n_completion_tokens": len(token_ids),
        "n_thinking_tokens": thinking_tokens,
        "n_answer_tokens": len(token_ids) if answer_tokens is None else answer_tokens,
        "n_terminal_tokens_trimmed": 0,
        "forced_close": forced,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "stop",
        "stage1_finish_reason": stage1_finish_reason,
    }
    if stage1_token_ids is not None:
        result["stage1_token_ids"] = stage1_token_ids
    if retained_thinking_token_ids is not None:
        result["retained_thinking_token_ids"] = retained_thinking_token_ids
    return result


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
        "sampling": {
            "thinking": "budget",
            "thinking_budget": 4,
            "answer_max_tokens": 512,
        },
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
    @staticmethod
    def fixture_library():
        return {
            "id": "lib-base",
            "provenance": "test",
            "macros": [],
            "draw_seed": None,
        }

    def test_budget_cap_contact_does_not_confuse_answer_restart_with_reasoning_cap(self):
        task = normalized_task()
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(0, "PROGRAM: ADD1 | NOOP", forced=True),
                    output(
                        1,
                        "PROGRAM: ADD1 | NOOP",
                        stage1_finish_reason="length",
                    ),
                    output(2, "PROGRAM: ADD1 | NOOP", thinking_tokens=4),
                    output(3, "PROGRAM: ADD1 | NOOP", truncated=True),
                    output(4, "PROGRAM: ADD1 | NOOP"),
                ],
            }
        ]
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=summary(rows),
                tasks={"t0": task},
                library=self.fixture_library(),
            )

        candidates = by_task["t0"]["candidates"]
        self.assertEqual(
            [candidate["cap_contact"] for candidate in candidates],
            [True, False, True, False, False],
        )
        self.assertTrue(candidates[1]["answer_restart_after_natural_close"])
        self.assertTrue(candidates[2]["thinking_tokens_at_budget"])
        self.assertTrue(candidates[3]["answer_truncated"])
        metrics = analyze.summarize_task_rows(by_task)
        self.assertEqual(metrics["cap_contact_rate"], 2 / 5)
        self.assertEqual(metrics["answer_restart_after_natural_close_rate"], 1 / 5)
        self.assertEqual(metrics["answer_truncation_rate"], 1 / 5)
        self.assertEqual(metrics["adequate_completion_rate"], 2 / 5)
        self.assertFalse(metrics["budget_adequacy"])
        relaxed = analyze.summarize_task_rows(
            by_task,
            max_cap_contact=0.7,
            max_answer_truncation=0.25,
        )
        self.assertTrue(relaxed["budget_adequacy"])

    def test_amendment9_analyzer_parity_for_all_five_boundary_cases(self):
        task = normalized_task()
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(
                        0,
                        "PROGRAM: ADD1 | NOOP",
                        forced=True,
                        thinking_tokens=4,
                        stage1_finish_reason="length",
                    ),
                    output(
                        1,
                        "PROGRAM: ADD1 | NOOP",
                        forced=True,
                        thinking_tokens=1,
                        stage1_finish_reason="stop",
                    ),
                    output(
                        2,
                        "PROGRAM: ADD1 | NOOP",
                        thinking_tokens=3,
                        stage1_finish_reason="length",
                    ),
                    output(
                        3,
                        "PROGRAM: ADD1 | NOOP",
                        thinking_tokens=1,
                        answer_tokens=32,
                        stage1_finish_reason="length",
                    ),
                    output(
                        4,
                        "PROGRAM: ADD1 | NOOP",
                        thinking_tokens=1,
                        answer_tokens=512,
                        stage1_finish_reason="length",
                    ),
                ],
            }
        ]
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=summary(rows),
                tasks={"t0": task},
                library=self.fixture_library(),
            )

        candidates = by_task["t0"]["candidates"]
        self.assertEqual(
            [row["forced_intervention"] for row in candidates],
            [True, True, False, False, False],
        )
        self.assertEqual(
            [row["reasoning_boundary_contact"] for row in candidates],
            [True, False, True, False, False],
        )
        self.assertEqual(
            [row["stage1_length_finish"] for row in candidates],
            [True, False, True, True, True],
        )
        self.assertEqual(
            [row["answer_restart_after_natural_close"] for row in candidates],
            [False, False, False, True, True],
        )
        self.assertEqual(
            [row["cap_contact"] for row in candidates],
            [True, True, True, False, False],
        )
        self.assertEqual(
            [row["answer_limit_contact"] for row in candidates],
            [False, False, False, False, True],
        )
        metrics = analyze.summarize_task_rows(by_task)
        self.assertEqual(metrics["forced_intervention_rate"], 2 / 5)
        self.assertEqual(metrics["reasoning_boundary_contact_rate"], 2 / 5)
        self.assertEqual(metrics["stage1_length_finish_rate"], 4 / 5)
        self.assertEqual(metrics["answer_restart_after_natural_close_rate"], 2 / 5)
        self.assertEqual(metrics["answer_limit_contact_rate"], 1 / 5)

    def test_natural_stop_at_answer_ceiling_is_limit_contact(self):
        task = normalized_task()
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(
                        0,
                        "PROGRAM: ADD1 | NOOP",
                        answer_tokens=512,
                    ),
                    output(
                        1,
                        "PROGRAM: ADD1 | NOOP",
                        answer_tokens=511,
                    ),
                ],
            }
        ]
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=summary(rows),
                tasks={"t0": task},
                library=self.fixture_library(),
            )

        at_limit, below_limit = by_task["t0"]["candidates"]
        self.assertFalse(at_limit["truncated"])
        self.assertTrue(at_limit["answer_limit_contact"])
        self.assertTrue(at_limit["answer_truncated"])
        self.assertTrue(at_limit["answer_tokens_at_limit"])
        self.assertFalse(below_limit["truncated"])
        self.assertFalse(below_limit["answer_limit_contact"])
        self.assertFalse(below_limit["answer_truncated"])
        self.assertFalse(below_limit["answer_tokens_at_limit"])
        metrics = analyze.summarize_task_rows(by_task)
        self.assertEqual(metrics["answer_truncation_rate"], 0.5)
        self.assertEqual(metrics["answer_limit_contact_rate"], 0.5)
        self.assertFalse(metrics["budget_adequacy"])

    @staticmethod
    def loop_decision():
        return {
            "loop_override_min_budget": 32768,
            "loop_tail_tokens": 8192,
            "loop_max_period_tokens": 2048,
            "loop_min_match_rate": 0.99,
        }

    def test_perfect_periodic_cap_contact_is_classified_and_resolved(self):
        task = normalized_task()
        periodic_ids = [index % 17 for index in range(32768)]
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(
                        0,
                        "PROGRAM: ADD1 | NOOP",
                        forced=True,
                        thinking_tokens=32768,
                        # Retained thinking must take precedence over the raw stage.
                        stage1_token_ids=list(range(32768)),
                        retained_thinking_token_ids=periodic_ids,
                    ),
                    output(1, "PROGRAM: ADD1 | NOOP"),
                    output(2, "PROGRAM: ADD1 | NOOP"),
                    output(3, "PROGRAM: ADD1 | NOOP"),
                ],
            }
        ]
        run_summary = summary(rows)
        run_summary["sampling"]["thinking_budget"] = 32768
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=run_summary,
                tasks={"t0": task},
                library=self.fixture_library(),
                decision=self.loop_decision(),
            )

        candidate = by_task["t0"]["candidates"][0]
        self.assertTrue(candidate["cap_contact"])
        self.assertTrue(candidate["periodic_loop"])
        self.assertFalse(candidate["unresolved_cap_contact"])
        self.assertEqual(candidate["loop_period_tokens"], 17)
        self.assertEqual(candidate["loop_tail_tokens"], 8192)
        self.assertEqual(candidate["loop_match_rate"], 1.0)
        self.assertEqual(
            candidate["loop_token_source"], "retained_thinking_token_ids"
        )
        metrics = analyze.summarize_task_rows(by_task)
        self.assertEqual(metrics["cap_contact_rate"], 0.25)
        self.assertEqual(metrics["periodic_loop_rate"], 0.25)
        self.assertEqual(metrics["unresolved_cap_contact_rate"], 0.0)
        self.assertTrue(metrics["budget_adequacy"])

    def test_nonperiodic_cap_contact_remains_unresolved(self):
        task = normalized_task()
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(
                        0,
                        "PROGRAM: ADD1 | NOOP",
                        forced=True,
                        thinking_tokens=32768,
                        stage1_token_ids=list(range(32768)),
                    ),
                    output(1, "PROGRAM: ADD1 | NOOP"),
                    output(2, "PROGRAM: ADD1 | NOOP"),
                    output(3, "PROGRAM: ADD1 | NOOP"),
                ],
            }
        ]
        run_summary = summary(rows)
        run_summary["sampling"]["thinking_budget"] = 32768
        with patched_domain():
            by_task = analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=run_summary,
                tasks={"t0": task},
                library=self.fixture_library(),
                decision=self.loop_decision(),
            )

        candidate = by_task["t0"]["candidates"][0]
        self.assertTrue(candidate["cap_contact"])
        self.assertFalse(candidate["periodic_loop"])
        self.assertTrue(candidate["unresolved_cap_contact"])
        self.assertEqual(candidate["loop_token_source"], "stage1_token_ids")
        metrics = analyze.summarize_task_rows(by_task)
        self.assertEqual(metrics["cap_contact_rate"], 0.25)
        self.assertEqual(metrics["periodic_loop_rate"], 0.0)
        self.assertEqual(metrics["unresolved_cap_contact_rate"], 0.25)
        self.assertFalse(metrics["budget_adequacy"])

    def test_budget_validation_rejects_thinking_beyond_registered_cap(self):
        task = normalized_task()
        rows = [
            {
                "id": "t0::base",
                "meta": {
                    "task_id": "t0",
                    "split": "reuse",
                    "arm": "base",
                    "library_id": "lib-base",
                    "max_surface_calls": 5,
                    "max_expanded_primitive_depth": 5,
                },
                "n_prompt_tokens": 5,
                "outputs": [
                    output(0, "PROGRAM: ADD1 | NOOP", thinking_tokens=5),
                ],
            }
        ]
        with patched_domain(), self.assertRaisesRegex(
            ValueError, "thinking tokens exceed registered budget"
        ):
            analyze.analyze_arm_rows(
                arm="base",
                rows=rows,
                summary=summary(rows),
                tasks={"t0": task},
                library=self.fixture_library(),
            )

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
                "answer_truncated": False,
                "cap_contact": False,
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
            "scored_max_cap_contact": 0.05,
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
        self.assertTrue(verdict["budget_adequacy"]["all_confirmatory_arms_adequate"])

        arms["random_4"]["r0"]["candidates"][0]["cap_contact"] = True
        summaries = {arm: analyze.summarize_arm(rows) for arm, rows in arms.items()}
        with tempfile.TemporaryDirectory() as temporary:
            unresolved = analyze.build_full_verdict(
                exp=Path(temporary),
                arms=arms,
                summaries=summaries,
                libraries=libraries,
                decision=decision,
                expected_arms=arm_names,
                repetitions=50,
                seed=11,
            )
        self.assertTrue(unresolved["complete_callable_abstraction"])
        self.assertTrue(unresolved["budget_unresolved"])
        self.assertFalse(unresolved["claimable_complete_callable_abstraction"])
        self.assertEqual(unresolved["status"], "budget_unresolved")
        self.assertEqual(unresolved["budget_adequacy"]["offending_arms"], ["random_4"])

        arms["random_4"]["r0"]["candidates"][0]["cap_contact"] = False
        arms["random_3"]["r0"]["candidates"][0]["truncated"] = True
        arms["random_3"]["r0"]["candidates"][0]["answer_truncated"] = True
        summaries = {arm: analyze.summarize_arm(rows) for arm, rows in arms.items()}
        with tempfile.TemporaryDirectory() as temporary:
            answer_unresolved = analyze.build_full_verdict(
                exp=Path(temporary),
                arms=arms,
                summaries=summaries,
                libraries=libraries,
                decision=decision,
                expected_arms=arm_names,
                repetitions=50,
                seed=11,
            )
        self.assertEqual(
            answer_unresolved["budget_adequacy"]["offending_arms"], ["random_3"]
        )
        self.assertEqual(answer_unresolved["status"], "budget_unresolved")


class SmokeVerdictTests(unittest.TestCase):
    @staticmethod
    def candidate(
        index: int,
        *,
        hidden: bool = False,
        macro: bool = False,
        truncated: bool = False,
        answer_limit_contact: bool = False,
        cap_contact: bool = False,
    ):
        return {
            "sample_index": index,
            "parsed": not truncated,
            "valid": not truncated,
            "macro_used": macro,
            "hidden_pass": hidden,
            "truncated": truncated,
            "answer_truncated": truncated or answer_limit_contact,
            "answer_limit_contact": truncated or answer_limit_contact,
            "cap_contact": cap_contact,
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
            "scored_max_cap_contact": 0.05,
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
        self.assertTrue(verdict["metrics"]["budget_adequacy"])

    def test_smoke_gate_rejects_cap_contact_and_answer_truncation(self):
        arms = {"base": {}, "designed_ceiling": {}}
        for task_index in range(2):
            task_id = f"reuse-{task_index}"
            base = [self.candidate(index) for index in range(12)]
            designed = [
                self.candidate(index, macro=index == 0) for index in range(12)
            ]
            arms["base"][task_id] = self.task("smoke_reuse", base)
            arms["designed_ceiling"][task_id] = self.task(
                "smoke_reuse", designed
            )
        # Three of 48 matched completions is 6.25%, beyond the registered 5%.
        arms["base"]["reuse-0"]["candidates"][0]["cap_contact"] = True
        arms["base"]["reuse-0"]["candidates"][1]["cap_contact"] = True
        arms["base"]["reuse-0"]["candidates"][2]["cap_contact"] = True
        for index in range(3):
            candidate = arms["designed_ceiling"]["reuse-0"]["candidates"][index]
            candidate["truncated"] = True
            candidate["answer_truncated"] = True

        verdict = analyze.build_smoke_verdict(
            arms, self.summaries(), self.decision()
        )

        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["gates"]["scored_cap_contact"])
        self.assertFalse(verdict["gates"]["scored_answer_truncation"])
        self.assertFalse(verdict["metrics"]["budget_adequacy"])
        self.assertAlmostEqual(verdict["metrics"]["overall_cap_contact_rate"], 3 / 48)
        self.assertAlmostEqual(
            verdict["metrics"]["overall_answer_truncation_rate"], 3 / 48
        )

    def test_smoke_gate_uses_answer_limit_contact_even_without_length_finish(self):
        arms = {"base": {}, "designed_ceiling": {}}
        for task_index in range(2):
            task_id = f"reuse-{task_index}"
            base = [self.candidate(index) for index in range(12)]
            designed = [
                self.candidate(
                    index,
                    macro=index == 0,
                    answer_limit_contact=task_index == 0 and index < 3,
                )
                for index in range(12)
            ]
            arms["base"][task_id] = self.task("smoke_reuse", base)
            arms["designed_ceiling"][task_id] = self.task(
                "smoke_reuse", designed
            )

        verdict = analyze.build_smoke_verdict(
            arms, self.summaries(), self.decision()
        )

        contacts = arms["designed_ceiling"]["reuse-0"]["candidates"][:3]
        self.assertTrue(all(not candidate["truncated"] for candidate in contacts))
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["gates"]["answer_truncation"])
        self.assertFalse(verdict["gates"]["scored_answer_truncation"])
        self.assertAlmostEqual(
            verdict["metrics"]["overall_answer_limit_contact_rate"], 3 / 48
        )
        self.assertAlmostEqual(
            verdict["metrics"]["designed_answer_limit_contact_rate"], 3 / 24
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
            (exp / "scripts").mkdir()
            (exp / "analysis").mkdir()
            source_exp = SCRIPT.parents[1]
            for relative in (
                "scripts/analyze.py",
                "scripts/run.py",
                "src/macro_domain.py",
                "src/model_harness.py",
                "src/scientific_artifacts.py",
                "src/vllm_runner.py",
            ):
                destination = exp / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_exp / relative, destination)
            external = exp / "external"
            (exp / "configs" / "default.yaml").write_text(
                f"""scientific_smoke:\n  external_root: {external}\ninference:\n  arms: [base, designed_ceiling]\n  smoke_arms: [base, designed_ceiling]\ndecision:\n  bootstrap_repetitions: 20\n  bootstrap_seed: 7\n  smoke_matched_k: 1\n  smoke_min_parse_rate: 0.50\n  smoke_min_macro_candidates: 2\n  smoke_max_answer_truncation: 0.05\n  scored_max_cap_contact: 0.05\n""",
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
            (exp / "data" / "demonstrations.json").write_text(
                json.dumps({"schema_version": 1, "demonstrations": []}) + "\n",
                encoding="utf-8",
            )
            for arm, program, library_id in (
                ("base", "PROGRAM: CONST_ONE | NOOP", "lib-base"),
                ("designed_ceiling", "PROGRAM: M0", "lib-designed"),
            ):
                rows = []
                prompt_sha256 = hashlib.sha256(f"prompt:{arm}".encode()).hexdigest()
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
                            "prompt_sha256": prompt_sha256,
                            "n_prompt_tokens": 5,
                            "outputs": [output(0, program)],
                        }
                    )
                prefix = f"smoke_tiers/think_4/{arm}"
                paths = analyze.scientific_store.bundle_paths(external, prefix)
                preflight = {
                    "schema_version": 1,
                    "pass": True,
                    "max_model_len": 65536,
                    "generation_reserve_tokens": 518,
                    "n_records": 2,
                    "min_prompt_tokens": 5,
                    "max_prompt_tokens": 5,
                    "max_prompt_plus_reserve_tokens": 523,
                    "records": [
                        {
                            "id": f"s{index}::{arm}",
                            "input_record_sha256": hashlib.sha256(
                                f"input:{index}:{arm}".encode()
                            ).hexdigest(),
                            "rendered_prompt_sha256": prompt_sha256,
                            "prompt_tokens": 5,
                            "prompt_plus_reserve_tokens": 523,
                        }
                        for index in range(2)
                    ],
                }
                analyze.scientific_store.write_preflight_only(external, prefix, preflight)
                paths.rows.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                metadata = {
                    **summary(rows),
                    "schema_version": analyze.local_vllm.RUNNER_SCHEMA_VERSION,
                    "runner_sha256": analyze.scientific_store.RUNNER_SHA256,
                    "engine": {"fixture": True, "max_model_len": 65536},
                    "sampling": {
                        "thinking": "budget",
                        "thinking_budget": 4,
                        "answer_max_tokens": 512,
                        "n": 1,
                    },
                }
                paths.metadata.write_text(
                    json.dumps(
                        metadata
                    ),
                    encoding="utf-8",
                )
                analyze.scientific_store.commit_receipt(
                    external,
                    prefix,
                    role="complete_matrix_arm",
                    tier_mode="complete_k12_matrix",
                    thinking_budget=4,
                    arm=arm,
                    k=1,
                    expected_identity={
                        "model": analyze.harness.REQUIRED_MODEL_ID,
                        "model_revision": analyze.harness.MODEL_REVISION,
                        "runner_sha256": analyze.scientific_store.RUNNER_SHA256,
                        "sampling": metadata["sampling"],
                        "engine": metadata["engine"],
                    },
                )
            selection_path = exp / "analysis" / "smoke_budget_selection.json"
            selection_path.write_text(
                json.dumps({"pass": True, "selected_thinking_budget": 4}) + "\n",
                encoding="utf-8",
            )
            protocol = analyze.scientific_store.build_protocol_binding(exp)
            catalog = analyze.scientific_store.build_catalog(
                external,
                protocol_binding=protocol,
                selection_file=selection_path,
                selected_budget=4,
                selected_entries={
                    "base": "matrix/think_4/base",
                    "designed_ceiling": "matrix/think_4/designed_ceiling",
                },
            )
            analyze.scientific_store.write_catalog(
                exp / "analysis" / "scientific_smoke_artifact_catalog.json", catalog
            )
            with patched_domain():
                result = analyze.analyze_experiment(
                    exp=exp, run="smoke", bootstrap_repetitions=20, write=True
                )
            self.assertTrue(result["smoke_gate"]["pass"])
            self.assertTrue((exp / "analysis" / "smoke_per_task.csv").exists())
            self.assertTrue((exp / "analysis" / "smoke_per_task.json").exists())
            compact_text = (exp / "analysis" / "smoke_per_task.json").read_text()
            self.assertNotIn("raw_text", compact_text)
            self.assertNotIn("token_ids", compact_text)
            machine = json.loads((exp / "analysis" / "smoke_verdict.json").read_text())
            self.assertTrue(machine["smoke_gate"]["pass"])

            # Later train-only Qwen additions are deliberately outside the
            # base/designed smoke protocol binding.
            libraries_payload = json.loads(
                (exp / "data" / "libraries.json").read_text(encoding="utf-8")
            )
            libraries_payload["libraries"]["qwen_ranked"] = {
                "id": "qwen-later",
                "provenance": "post-smoke",
                "macros": [],
            }
            (exp / "data" / "libraries.json").write_text(
                json.dumps(libraries_payload), encoding="utf-8"
            )
            with patched_domain():
                replay = analyze.analyze_experiment(
                    exp=exp, run="smoke", bootstrap_repetitions=20, write=False
                )
            self.assertTrue(replay["smoke_gate"]["pass"])

            # Hidden labels are inside tasks.json and must invalidate the
            # catalog before a single model row is read or parsed.
            tasks_payload = json.loads(
                (exp / "data" / "tasks.json").read_text(encoding="utf-8")
            )
            tasks_payload["tasks"][0]["hidden"][0]["input"] += 1
            tasks_payload["tasks"][0]["hidden"][0]["output"] += 1
            (exp / "data" / "tasks.json").write_text(
                json.dumps(tasks_payload), encoding="utf-8"
            )
            with patched_domain(), mock.patch.object(
                analyze, "_read_jsonl", side_effect=AssertionError("rows read before catalog")
            ) as read_rows:
                with self.assertRaisesRegex(
                    analyze.scientific_store.ScientificArtifactError, "catalog differs"
                ):
                    analyze.analyze_experiment(
                        exp=exp, run="smoke", bootstrap_repetitions=20, write=False
                    )
            read_rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()
