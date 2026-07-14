from __future__ import annotations

import sys
import copy
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import scoring as S  # noqa: E402


class ScoringTests(unittest.TestCase):
    @staticmethod
    def output(text: str, tokens: int) -> dict:
        return {
            "text": text,
            "token_ids": [7] * tokens,
            "stage1_token_ids": [7] * tokens,
            "stage2_token_ids": [],
            "injected_token_ids": [],
            "n_sampled_tokens": tokens,
            "n_injected_tokens": 0,
            "n_completion_tokens": tokens,
        }

    @staticmethod
    def ordinary_generation() -> tuple[list[dict], dict]:
        output = {
            "sample_index": 0,
            "text": "synthetic",
            "token_ids": [1, 2],
            "stage1_token_ids": [1, 2, S.HF_MODEL_EOS_TOKEN_ID],
            "stage2_token_ids": [],
            "injected_token_ids": [],
            "n_thinking_tokens": 0,
            "n_answer_tokens": 2,
            "n_sampled_tokens": 3,
            "n_injected_tokens": 0,
            "n_completion_tokens": 2,
            "n_terminal_tokens_trimmed": 1,
            "n_stage1_prompt_tokens": 5,
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": False,
            "forced_close": False,
            "truncated": False,
        }
        generated = [{
            "id": "t1",
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "n_prompt_tokens": 5,
            "outputs": [output],
        }]
        metadata = {
            "sampling": {"thinking": "off", "shuffle_thinking": False},
            "termination": {
                "hf_model_eos_token_id": S.HF_MODEL_EOS_TOKEN_ID,
                "vllm_tokenizer_eos_ignored": 248046,
            },
            "think_token_ids": {
                "open": 248068,
                "close": S.THINK_CLOSE_TOKEN_ID,
                "forced_close_sequence": [S.THINK_CLOSE_TOKEN_ID, 198, 198],
                "thinking_prompt_suffix": [1],
                "no_thinking_prompt_suffix": [2],
            },
            "counts": {
                "requests": 1,
                "completions": 1,
                "unique_input_prompt_tokens": 5,
                "stage1_logical_prompt_tokens": 5,
                "stage2_logical_prompt_tokens": 0,
                "logical_model_input_tokens": 5,
                "sampled_tokens": 3,
                "injected_tokens": 0,
            },
            "timing": {
                "model_load_seconds": 1.0,
                "generation_seconds": 2.0,
                "sampled_tokens_per_second": 1.5,
            },
        }
        return generated, metadata

    def test_strict_parser_accepts_only_exact_three_output_array(self) -> None:
        self.assertEqual(S.parse_answer("</think>\nANSWER: [[1],\"x\",[2,3]]"), [[1], "x", [2, 3]])
        for bad in (
            "Here: ANSWER: [1,2,3]",
            "ANSWER: [1,2]",
            "ANSWER: [1,2,3] trailing",
            "ANSWER: nope",
        ):
            with self.subTest(bad=bad):
                self.assertIsNone(S.parse_answer(bad))

    def test_exact_periodic_tail_requires_repeats_and_total_length(self) -> None:
        self.assertTrue(S.exact_periodic_tail([1, 2, 3] * 20, 4, 16, 60))
        self.assertFalse(S.exact_periodic_tail(list(range(60)), 4, 16, 60))
        self.assertFalse(S.exact_periodic_tail([1, 2, 3] * 10, 4, 16, 60))

    def test_scoring_uses_candidate_prefixes_and_contacts(self) -> None:
        outputs = []
        for index in range(4):
            outputs.append(
                {
                    "text": "</think>\n\nANSWER: [1,2,3]" if index == 2 else "bad",
                    "token_ids": [4] * 12,
                    "stage1_token_ids": [4] * 10,
                    "stage2_token_ids": [],
                    "injected_token_ids": [5, 6],
                    "truncated": index == 3,
                    "n_answer_tokens": 3,
                    "retained_thinking_token_ids": [1, 2, 3] * 20 if index == 3 else [],
                    "n_sampled_tokens": 10,
                    "n_injected_tokens": 2,
                    "n_completion_tokens": 12,
                }
            )
        scored = S.score_generation_rows(
            [{"id": "t1", "outputs": outputs}],
            [{"id": "t1", "family": "list", "depth": 3, "split": "qualification", "answers": [1, 2, 3]}],
            arm="frozen_action",
            candidate_counts=(1, 4),
            answer_max_tokens=128,
            loop_detector={"min_repeats": 4, "max_period_tokens": 16, "min_total_repeated_tokens": 60},
        )[0]
        self.assertEqual(scored["coverage_at_1"], 0.0)
        self.assertEqual(scored["coverage_at_4"], 1.0)
        self.assertEqual(scored["strict_parse_rate"], 0.25)
        self.assertEqual(scored["answer_limit_contact"], 0.25)
        self.assertEqual(scored["periodic_loop_contact"], 0.25)
        self.assertEqual(scored["logical_tokens"], 48)

    def test_generation_counts_are_reconstructed_from_raw_arrays(self) -> None:
        generated, metadata = self.ordinary_generation()
        self.assertEqual(S.validate_generation_counters(generated, metadata)["sampled_tokens"], 3)
        forged = copy.deepcopy(metadata)
        forged["counts"]["sampled_tokens"] = 1_000_000_000
        with self.assertRaisesRegex(ValueError, "raw token reconstruction"):
            S.validate_generation_counters(generated, forged)
        forged_rows = copy.deepcopy(generated)
        forged_rows[0]["n_prompt_tokens"] = 1_000_000_000
        forged = copy.deepcopy(metadata)
        forged["counts"].update(
            unique_input_prompt_tokens=1_000_000_000,
            stage1_logical_prompt_tokens=1_000_000_000,
            logical_model_input_tokens=1_000_000_000,
        )
        with self.assertRaisesRegex(ValueError, "differs from raw token arrays"):
            S.validate_generation_counters(forged_rows, forged)

    def test_counter_schema_rejects_booleans_and_nonfinite_timing(self) -> None:
        generated, metadata = self.ordinary_generation()
        forged_rows = copy.deepcopy(generated)
        forged_rows[0]["outputs"][0]["n_sampled_tokens"] = True
        with self.assertRaisesRegex(ValueError, "exact integer"):
            S.validate_generation_counters(forged_rows, metadata)
        forged_metadata = copy.deepcopy(metadata)
        forged_metadata["timing"]["generation_seconds"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            S.validate_generation_counters(generated, forged_metadata)
        forged_metadata = copy.deepcopy(metadata)
        forged_metadata["counts"]["requests"] = True
        with self.assertRaisesRegex(ValueError, "raw token reconstruction"):
            S.validate_generation_counters(generated, forged_metadata)

    def test_budget_continuation_reconstructs_forced_close_and_stage2_prompt(self) -> None:
        close = [S.THINK_CLOSE_TOKEN_ID, 198, 198]
        output = {
            "sample_index": 0,
            "text": "synthetic",
            "token_ids": [11, 12] + close + [21],
            "stage1_token_ids": [11, 12],
            "retained_thinking_token_ids": [11, 12],
            "injected_token_ids": close,
            "stage2_token_ids": [21, S.HF_MODEL_EOS_TOKEN_ID],
            "n_thinking_tokens": 2,
            "n_answer_tokens": 1,
            "n_sampled_tokens": 4,
            "n_injected_tokens": 3,
            "n_completion_tokens": 6,
            "n_terminal_tokens_trimmed": 1,
            "n_stage1_prompt_tokens": 5,
            "n_stage2_prompt_tokens": 10,
            "thinking_closed": True,
            "forced_close": True,
            "truncated": False,
        }
        generated = [{
            "id": "t1",
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "n_prompt_tokens": 5,
            "outputs": [output],
        }]
        _, metadata = self.ordinary_generation()
        metadata["sampling"]["thinking"] = "budget"
        metadata["counts"].update(
            stage2_logical_prompt_tokens=10,
            logical_model_input_tokens=15,
            sampled_tokens=4,
            injected_tokens=3,
        )
        metadata["timing"]["sampled_tokens_per_second"] = 2.0
        S.validate_generation_counters(generated, metadata)
        forged = copy.deepcopy(generated)
        forged[0]["outputs"][0]["n_stage2_prompt_tokens"] = 9
        with self.assertRaisesRegex(ValueError, "reconstructed prompt"):
            S.validate_generation_counters(forged, metadata)

    def test_literal_diagnostic_uses_shortest_base_prefix_reaching_token_spend(self) -> None:
        answer = "ANSWER: [1,2,3]"
        reflection = [{"id": "t1", "outputs": [self.output("PLAN: x", 5)] * 4}]
        action = [
            {
                "id": f"t1::literal::{index}",
                "meta": {"parent_task_id": "t1", "sample_index": index},
                "outputs": [self.output(answer if index == 3 else "bad", 10)],
            }
            for index in range(4)
        ]
        base = [
            {
                "id": "t1",
                "outputs": [self.output("bad", 25), self.output(answer, 25), self.output("bad", 25)],
            }
        ]
        labels = [{"id": "t1", "family": "list", "split": "qualification", "answers": [1, 2, 3]}]
        scored = S.score_literal_reflection_diagnostic(
            reflection, action, base, labels, literal_candidate_count=4
        )[0]
        self.assertEqual(scored["literal_logical_tokens"], 60)
        self.assertEqual(scored["matched_base_candidates"], 3)
        self.assertEqual(scored["literal_coverage"], 1.0)
        self.assertEqual(scored["matched_base_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
