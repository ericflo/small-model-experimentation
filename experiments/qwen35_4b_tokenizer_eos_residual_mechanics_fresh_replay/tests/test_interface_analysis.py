from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import interface_analysis as analysis  # noqa: E402

RUN_SEED = 17


class FakeTokenizer:
    @staticmethod
    def decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
        if skip_special_tokens:
            raise AssertionError("special tokens must be preserved")
        return "/".join(str(value) for value in token_ids)


def grammar_receipt(row_ids: list[str]) -> dict:
    inventories = {}
    for condition in ("freeform", "program_slot"):
        inventories[condition] = {}
        prefix = [] if condition == "freeform" else [90, 91]
        for arity in (2, 3):
            sequences = [prefix + [arity, index] for index in range(24**arity)]
            inventories[condition][str(arity)] = {
                "rows": len(sequences),
                "token_id_sequences": sequences,
                "token_id_sequences_sha256": analysis.canonical_sha256(sequences),
            }
    expected = {}
    for index, row_id in enumerate(row_ids):
        arity = 2 if index < 24 else 3
        local = index if index < 24 else index - 24
        expected[row_id] = {
            "freeform": [arity, local],
            "program_slot": [90, 91, arity, local],
        }
    return {
        "think_token_ids": {"forced_close_sequence": [248069, 271]},
        "program_slot_prefix_token_ids": [90, 91],
        "grammar_inventories": inventories,
        "calibration_expected_token_ids": expected,
        "calibration_prompt_token_ids": {
            row_id: {
                "no_think": {
                    "token_ids": [7, index],
                    "prompt_text_sha256": "0" * 64,
                },
                "think512": {
                    "token_ids": [7, index],
                    "prompt_text_sha256": "0" * 64,
                },
            }
            for index, row_id in enumerate(row_ids)
        },
    }


def output(
    *,
    raw: list[int],
    boundary: str,
    pair_index: int,
    pair_offset: int,
) -> dict:
    stop = analysis.BOUNDARY_STOP_IDS[boundary]
    seed = analysis._stable_seed(RUN_SEED, f"row-{pair_index:02d}", 0, "answer")
    return {
        "sample_index": 0,
        "pair_index": pair_index,
        "pair_offset": pair_offset,
        "batch_position": pair_index * 2 + pair_offset,
        "pair_adjacent": True,
        "boundary": boundary,
        "registered_stop_token_id": stop,
        "stage1_parent_seed": seed,
        "seed_stage1": seed,
        "seed_stage2": None,
        "answer_seed": seed,
        "seed_domain_stage1": "answer",
        "seed_domain_stage2": None,
        "text": FakeTokenizer.decode(raw, skip_special_tokens=False),
        "token_ids": list(raw),
        "raw_answer_token_ids": list(raw),
        "stage1_token_ids": list(raw),
        "retained_thinking_token_ids": [],
        "answer_prefix_token_ids": [],
        "injected_token_ids": [],
        "stage2_token_ids": [],
        "n_thinking_tokens": 0,
        "n_answer_tokens": len(raw),
        "n_sampled_tokens": len(raw),
        "n_injected_tokens": 0,
        "n_completion_tokens": len(raw),
        "n_terminal_tokens_trimmed": 0,
        "n_tokens_discarded_after_close": 0,
        "n_stage1_prompt_tokens": 2,
        "n_stage2_prompt_tokens": 0,
        "thinking_closed": False,
        "forced_close": False,
        "finish_reason": "stop",
        "stop_reason": stop,
        "stage1_finish_reason": "stop",
        "stage1_stop_reason": stop,
        "stage2_finish_reason": None,
        "stage2_stop_reason": None,
        "truncated": False,
        "answer_cap_contact": False,
        "stage1_cumulative_logprob": -1.0,
        "stage2_cumulative_logprob": None,
        "sampled_cumulative_logprob": -1.0,
        "stage1_logprobs": None,
        "stage2_logprobs": None,
    }


def refresh_counts(rows: list[dict], metadata: dict) -> None:
    outputs = [item for row in rows for item in row["outputs"]]
    stage1 = sum(item["n_stage1_prompt_tokens"] for item in outputs)
    stage2 = sum(item["n_stage2_prompt_tokens"] for item in outputs)
    sampled = sum(item["n_sampled_tokens"] for item in outputs)
    physical_sampled = sum(len(item["raw_answer_token_ids"]) for item in outputs)
    logical_prompt = stage1 + stage2
    injected = sum(item["n_injected_tokens"] for item in outputs)
    metadata["counts"] = {
        "requests": len(rows),
        "completions": len(outputs),
        "unique_input_prompt_tokens": sum(row["n_prompt_tokens"] for row in rows),
        "stage1_logical_prompt_tokens": stage1,
        "stage2_logical_prompt_tokens": stage2,
        "logical_model_input_tokens": logical_prompt,
        "logical_prompt_tokens": logical_prompt,
        "physical_prompt_tokens": logical_prompt,
        "reused_prompt_tokens": 0,
        "sampled_tokens": sampled,
        "physical_sampled_tokens": physical_sampled,
        "reused_sampled_tokens": sampled - physical_sampled,
        "logical_model_tokens": logical_prompt + sampled,
        "physical_model_tokens": logical_prompt + physical_sampled,
        "reused_model_tokens": sampled - physical_sampled,
        "injected_tokens": injected,
    }


def bundle() -> tuple[list[dict], dict, dict]:
    rows = []
    for index in range(48):
        arity = 2 if index < 24 else 3
        local = index if index < 24 else index - 24
        prompt_ids = [7, index]
        prompt_hash = hashlib.sha256(
            b"".join(value.to_bytes(4, "big") for value in prompt_ids)
        ).hexdigest()
        tokenizer_raw = [arity, local, analysis.TOKENIZER_EOS_ID]
        hf_raw = [
            arity,
            local,
            analysis.TOKENIZER_EOS_ID,
            198,
            analysis.HF_MODEL_EOS_ID,
        ]
        rows.append(
            {
                "id": f"row-{index:02d}",
                "meta": {"task_id": f"task-{index:02d}", "arity": arity},
                "prompt_sha256": "0" * 64,
                "effective_prompt_sha256": prompt_hash,
                "effective_prompt_token_ids": prompt_ids,
                "n_prompt_tokens": 2,
                "n_original_prompt_tokens": 2,
                "n_effective_prompt_tokens": 2,
                "prompt_channel": "off",
                "answer_prefix_token_ids": [],
                "prompt_logprobs": None,
                "outputs": [
                    output(
                        raw=tokenizer_raw,
                        boundary="tokenizer_eos",
                        pair_index=index,
                        pair_offset=0,
                    ),
                    output(
                        raw=hf_raw,
                        boundary="hf_model_eos",
                        pair_index=index,
                        pair_offset=1,
                    ),
                ],
            }
        )
    metadata = {
        "schema_version": 6,
        "generation_mode": "answer_boundary_pairs",
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "think_token_ids": {"forced_close_sequence": [248069, 271]},
        "termination": {
            "hf_model_eos_token_id": 248044,
            "vllm_tokenizer_eos_ignored": 248046,
        },
        "sampling": {
            "thinking": "off",
            "thinking_budget": None,
            "n": 1,
            "answer_max_tokens": 24,
            "run_seed": RUN_SEED,
            "paired_answer_seed": True,
        },
        "boundary_pairing": {
            "boundary_order": ["tokenizer_eos", "hf_model_eos"],
            "registered_stop_token_ids": [248046, 248044],
            "pairs": 48,
            "requests": 96,
            "adjacent_in_single_generate_call": True,
            "identical_prompt_ids_within_pair": True,
            "identical_answer_seed_within_pair": True,
            "raw_answer_tokens_preserved": True,
            "answer_cap_tokens": 24,
        },
    }
    refresh_counts(rows, metadata)
    return rows, metadata, grammar_receipt([row["id"] for row in rows])


def set_raw(
    output_row: dict,
    raw: list[int],
    *,
    finish_reason: str = "stop",
    stop_reason: int | None = None,
) -> None:
    output_row["raw_answer_token_ids"] = list(raw)
    output_row["stage1_token_ids"] = list(raw)
    output_row["token_ids"] = list(raw)
    output_row["text"] = FakeTokenizer.decode(raw, skip_special_tokens=False)
    output_row["n_answer_tokens"] = len(raw)
    output_row["n_sampled_tokens"] = len(raw)
    output_row["n_completion_tokens"] = len(raw)
    output_row["finish_reason"] = finish_reason
    output_row["stop_reason"] = stop_reason
    output_row["stage1_finish_reason"] = finish_reason
    output_row["stage1_stop_reason"] = stop_reason
    output_row["truncated"] = finish_reason == "length"
    output_row["answer_cap_contact"] = len(raw) >= 24 or finish_reason == "length"


class InterfaceAnalysisTests(unittest.TestCase):
    def score(self, rows: list[dict], metadata: dict, receipt: dict) -> dict:
        return analysis.authenticate_and_score_bundle(
            rows,
            metadata,
            prefix_condition="freeform",
            thinking_policy="no_think",
            grammar_receipt=receipt,
            tokenizer=FakeTokenizer(),
        )

    def test_tokenizer_boundary_qualifies_while_hf_content_does_not(self) -> None:
        rows, metadata, receipt = bundle()
        result = self.score(rows, metadata, receipt)
        self.assertEqual(result["pair_authentication"], "PASS")
        self.assertEqual(
            result["cells"]["tokenizer_eos"]["exact_echo_successes"], 48
        )
        self.assertEqual(result["cells"]["hf_model_eos"]["parse_successes"], 0)
        self.assertEqual(result["cells"]["tokenizer_eos"]["by_arity"]["2"]["rows"], 24)

    def test_well_formed_early_stop_scores_false_without_auth_failure(self) -> None:
        rows, metadata, receipt = bundle()
        set_raw(rows[0]["outputs"][0], [2, 248046], stop_reason=248046)
        set_raw(rows[0]["outputs"][1], [2, 248046, 248044], stop_reason=248044)
        refresh_counts(rows, metadata)
        result = self.score(rows, metadata, receipt)
        self.assertEqual(result["pair_authentication"], "PASS")
        self.assertFalse(result["cells"]["tokenizer_eos"]["scored"][0]["parsed"])

    def test_exact_cap_length_is_content_and_authenticates(self) -> None:
        rows, metadata, receipt = bundle()
        content = list(range(1000, 1024))
        sequences = receipt["grammar_inventories"]["freeform"]["2"]["token_id_sequences"]
        sequences[0] = content
        receipt["grammar_inventories"]["freeform"]["2"][
            "token_id_sequences_sha256"
        ] = analysis.canonical_sha256(sequences)
        receipt["calibration_expected_token_ids"]["row-00"]["freeform"] = content
        for value in rows[0]["outputs"]:
            set_raw(value, content, finish_reason="length", stop_reason=None)
        refresh_counts(rows, metadata)
        result = self.score(rows, metadata, receipt)
        scored = result["cells"]["tokenizer_eos"]["scored"][0]
        self.assertTrue(scored["exact_echo"])
        self.assertTrue(scored["answer_cap_contact"])
        self.assertEqual(scored["precommit_token_ids"], content)

    def test_malformed_terminal_classes_fail_closed(self) -> None:
        cases = []
        rows, metadata, receipt = bundle()
        set_raw(rows[0]["outputs"][0], [2, 0], stop_reason=248046)
        refresh_counts(rows, metadata)
        cases.append((rows, metadata, receipt, "malformed_registered_stop"))
        rows, metadata, receipt = bundle()
        set_raw(rows[0]["outputs"][0], [2, 0], finish_reason="length")
        refresh_counts(rows, metadata)
        cases.append((rows, metadata, receipt, "short_output_relabeled_length"))
        rows, metadata, receipt = bundle()
        set_raw(rows[0]["outputs"][0], [2, 248046, 248046], stop_reason=248046)
        refresh_counts(rows, metadata)
        cases.append((rows, metadata, receipt, "malformed_registered_stop"))
        rows, metadata, receipt = bundle()
        rows[0]["outputs"][0]["stop_reason"] = 248044
        rows[0]["outputs"][0]["stage1_stop_reason"] = 248044
        cases.append((rows, metadata, receipt, "wrong stop/finish reason"))
        for rows, metadata, receipt, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                analysis.BoundaryAuthenticationError, message
            ):
                self.score(rows, metadata, receipt)

    def test_pair_prefix_prompt_seed_text_and_cost_mutations_fail(self) -> None:
        mutations = []
        rows, metadata, receipt = bundle()
        set_raw(rows[0]["outputs"][1], [9, 0, 248044], stop_reason=248044)
        refresh_counts(rows, metadata)
        mutations.append((rows, metadata, receipt, "sampled_prefix_divergence"))
        rows, metadata, receipt = bundle()
        rows[0]["effective_prompt_token_ids"][0] = 8
        mutations.append((rows, metadata, receipt, "prompt changed"))
        rows, metadata, receipt = bundle()
        rows[0]["outputs"][1]["answer_seed"] += 1
        mutations.append((rows, metadata, receipt, "output geometry changed"))
        rows, metadata, receipt = bundle()
        rows[0]["outputs"][0]["text"] = "mutated"
        mutations.append((rows, metadata, receipt, "text changed"))
        rows, metadata, receipt = bundle()
        metadata["counts"]["physical_sampled_tokens"] += 1
        mutations.append((rows, metadata, receipt, "cost summary changed"))
        for rows, metadata, receipt, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                analysis.BoundaryAuthenticationError, message
            ):
                self.score(rows, metadata, receipt)

    def test_coordinated_seed_cost_and_close_rewrites_fail(self) -> None:
        mutations = []
        rows, metadata, receipt = bundle()
        for value in rows[0]["outputs"]:
            value["stage1_parent_seed"] += 1
            value["seed_stage1"] += 1
            value["answer_seed"] += 1
        mutations.append((rows, metadata, receipt, "output geometry changed"))

        rows, metadata, receipt = bundle()
        for value in rows[0]["outputs"]:
            value["n_stage1_prompt_tokens"] += 1
        refresh_counts(rows, metadata)
        mutations.append((rows, metadata, receipt, "answer geometry changed"))

        rows, metadata, receipt = bundle()
        for value in rows[0]["outputs"]:
            value["n_sampled_tokens"] += 1
            value["n_completion_tokens"] += 1
        refresh_counts(rows, metadata)
        mutations.append((rows, metadata, receipt, "answer geometry changed"))

        rows, metadata, receipt = bundle()
        metadata["think_token_ids"]["forced_close_sequence"] = [1, 2]
        mutations.append((rows, metadata, receipt, "forced-close token receipt"))

        for rows, metadata, receipt, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                analysis.BoundaryAuthenticationError, message
            ):
                self.score(rows, metadata, receipt)

    def test_selection_priority_hf_only_neither_and_dual_invariant(self) -> None:
        gate = {
            "rows": 48,
            "exact_echo_successes_min": 44,
            "parse_successes_min": 44,
            "answer_cap_contacts_max": 2,
            "each_arity_rows": 24,
            "each_arity_exact_successes_min": 22,
            "each_arity_parse_successes_min": 22,
            "each_arity_answer_cap_contacts_max": 1,
        }

        def metric(success: bool) -> dict:
            count = 48 if success else 0
            per = 24 if success else 0
            return {
                "rows": 48,
                "exact_echo_successes": count,
                "parse_successes": count,
                "answer_cap_contacts": 0,
                "by_arity": {
                    str(arity): {
                        "rows": 24,
                        "exact_echo_successes": per,
                        "parse_successes": per,
                        "answer_cap_contacts": 0,
                    }
                    for arity in (2, 3)
                },
            }

        cells = {
            f"{boundary}_{condition}": metric(False)
            for boundary in analysis.BOUNDARY_ORDER
            for condition in analysis.PAIR_CONDITIONS
        }
        cells["tokenizer_eos_think512_freeform"] = metric(True)
        self.assertEqual(
            analysis.choose_interface(cells, gate)["winner"],
            "tokenizer_eos_think512_freeform",
        )
        cells = {key: metric(False) for key in cells}
        cells["hf_model_eos_no_think_freeform"] = metric(True)
        self.assertEqual(
            analysis.choose_interface(cells, gate)["decision"],
            "HF_ONLY_CONTROL_QUALIFIES_TOKENIZER_FAIL",
        )
        cells["hf_model_eos_no_think_freeform"] = metric(False)
        self.assertEqual(
            analysis.choose_interface(cells, gate)["decision"],
            "NO_VALID_TOKENIZER_EOS_ANSWER_SEAM",
        )
        cells["tokenizer_eos_no_think_freeform"] = metric(True)
        cells["hf_model_eos_no_think_freeform"] = metric(True)
        self.assertEqual(
            analysis.choose_interface(cells, gate)["decision"],
            "SCORING_INVARIANT_VIOLATION",
        )


if __name__ == "__main__":
    unittest.main()
