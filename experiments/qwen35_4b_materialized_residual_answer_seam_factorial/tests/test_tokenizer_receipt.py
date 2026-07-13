from __future__ import annotations

import json
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


class TokenizerReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(
            (EXP / "runs/tokenizer/receipt_v3.json").read_text()
        )

    def test_exact_qwen_termination_and_slot_ids(self) -> None:
        self.assertEqual(
            self.receipt["termination"],
            {
                "hf_model_eos_token_id": 248044,
                "tokenizer_eos_token_id": 248046,
                "tokenizer_eos_token": "<|im_end|>",
            },
        )
        self.assertEqual(
            self.receipt["think_token_ids"]["forced_close_sequence"],
            [248069, 271],
        )
        self.assertEqual(
            self.receipt["answer_prefix"],
            {
                "text": "PROGRAM:",
                "token_ids": [78041, 25],
                "close_plus_prefix_decode_exact": True,
            },
        )

    def test_every_canonical_line_composes_and_fits(self) -> None:
        canonical = self.receipt["canonical_answers"]
        self.assertEqual(canonical["rows"], 14_400)
        self.assertEqual(canonical["arity_rows"], {"2": 576, "3": 13_824})
        self.assertEqual(canonical["prefix_tail_token_compositional_failures"], 0)
        self.assertTrue(canonical["all_fit_sampled_answer_cap_24_with_terminal_slack"])

    def test_rendered_parent_overlap_and_context_fit_are_zero_safe(self) -> None:
        rendered = self.receipt["rendered_prompt_inventory"]
        self.assertEqual(rendered["parent_unique"], 1_984)
        self.assertEqual(
            rendered["parent_overlap"],
            {"think_base": 0, "no_think_base": 0, "no_think_program_slot": 0},
        )
        self.assertLessEqual(
            max(self.receipt["registered_max_context_tokens"].values()),
            self.receipt["max_model_len"],
        )

    def test_receipt_is_model_free_and_has_empty_forbidden_reads(self) -> None:
        self.assertEqual(self.receipt["schema_version"], 3)
        self.assertEqual(
            self.receipt["stage"],
            "real_tokenizer_shared_thought_physical_compute_receipt",
        )
        self.assertEqual(
            self.receipt["decision"], "TOKENIZER_AND_RENDERED_FRESHNESS_PASS"
        )
        self.assertFalse(self.receipt["model_loaded"])
        self.assertEqual(self.receipt["model_calls"], 0)
        self.assertEqual(self.receipt["sampled_model_outputs"], 0)
        for key in (
            "hidden_files_read",
            "qualification_files_read",
            "confirmation_files_read",
            "benchmark_files_read",
        ):
            self.assertEqual(self.receipt[key], [])


if __name__ == "__main__":
    unittest.main()
