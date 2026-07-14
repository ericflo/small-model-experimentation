from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
RECEIPT = EXP / "runs/tokenizer/receipt.json"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class TokenizerReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(RECEIPT.read_text())

    def test_receipt_is_model_free_and_uses_pinned_termination(self) -> None:
        value = self.receipt
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(
            value["decision"], "TOKENIZER_GRAMMAR_PROMPT_FRESHNESS_PASS"
        )
        self.assertEqual(value["model"], "Qwen/Qwen3.5-4B")
        self.assertEqual(
            value["revision"],
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(
            value["termination"],
            {
                "hf_model_eos_token_id": 248044,
                "tokenizer_eos_token": "<|im_end|>",
                "tokenizer_eos_token_id": 248046,
            },
        )
        self.assertIs(value["model_loaded"], False)
        self.assertEqual(value["model_calls"], 0)
        self.assertEqual(value["sampled_model_outputs"], 0)
        for field in (
            "hidden_files_read",
            "qualification_files_read",
            "confirmation_files_read",
            "benchmark_files_read",
        ):
            self.assertEqual(value[field], [])

    def test_every_prefix_and_arity_inventory_is_explicit_and_hash_bound(self) -> None:
        inventories = self.receipt["grammar_inventories"]
        prefix = self.receipt["program_slot_prefix_token_ids"]
        self.assertTrue(prefix)
        for arity in (2, 3):
            free = inventories["freeform"][str(arity)]
            slot = inventories["program_slot"][str(arity)]
            expected = 24**arity
            self.assertEqual(free["rows"], expected)
            self.assertEqual(slot["rows"], expected)
            self.assertEqual(
                free["token_id_sequences_sha256"],
                canonical_sha256(free["token_id_sequences"]),
            )
            self.assertEqual(
                slot["token_id_sequences_sha256"],
                canonical_sha256(slot["token_id_sequences"]),
            )
            self.assertEqual(
                slot["sampled_remainder_token_id_sequences_sha256"],
                canonical_sha256(slot["sampled_remainder_token_id_sequences"]),
            )
            self.assertEqual(
                len({tuple(row) for row in free["token_id_sequences"]}), expected
            )
            self.assertEqual(
                len({tuple(row) for row in slot["token_id_sequences"]}), expected
            )
            self.assertTrue(
                all(
                    composed == prefix + remainder
                    for composed, remainder in zip(
                        slot["token_id_sequences"],
                        slot["sampled_remainder_token_id_sequences"],
                        strict=True,
                    )
                )
            )
            self.assertEqual(
                free["semantic_lines_sha256"], slot["semantic_lines_sha256"]
            )
            self.assertLessEqual(
                free["max_sampled_tokens_including_terminal"], 24
            )
            self.assertLessEqual(
                slot["max_sampled_tokens_including_terminal"], 24
            )

    def test_calibration_expected_and_prompt_registries_are_complete(self) -> None:
        expected = self.receipt["calibration_expected_token_ids"]
        sampled = self.receipt["calibration_expected_sampled_token_ids"]
        prompts = self.receipt["calibration_prompt_token_ids"]
        self.assertEqual(len(expected), 48)
        self.assertEqual(set(expected), set(sampled))
        self.assertEqual(set(expected), set(prompts))
        prefix = self.receipt["program_slot_prefix_token_ids"]
        grammar = self.receipt["grammar_inventories"]
        free_sets = {
            arity: {tuple(row) for row in grammar["freeform"][str(arity)]["token_id_sequences"]}
            for arity in (2, 3)
        }
        slot_sets = {
            arity: {tuple(row) for row in grammar["program_slot"][str(arity)]["token_id_sequences"]}
            for arity in (2, 3)
        }
        arity_counts = {2: 0, 3: 0}
        for row_id in sorted(expected):
            free = expected[row_id]["freeform"]
            slot = expected[row_id]["program_slot"]
            arity = 2 if tuple(free) in free_sets[2] else 3
            self.assertIn(tuple(free), free_sets[arity])
            self.assertIn(tuple(slot), slot_sets[arity])
            self.assertEqual(slot, prefix + sampled[row_id]["program_slot"])
            self.assertEqual(free, sampled[row_id]["freeform"])
            self.assertEqual(set(prompts[row_id]), {"no_think", "think512"})
            for policy in ("no_think", "think512"):
                self.assertTrue(prompts[row_id][policy]["token_ids"])
                self.assertEqual(len(prompts[row_id][policy]["prompt_text_sha256"]), 64)
            arity_counts[arity] += 1
        self.assertEqual(arity_counts, {2: 24, 3: 24})

    def test_prompt_freshness_and_context_receipts_are_strict(self) -> None:
        inventory = self.receipt["rendered_prompt_inventory"]
        self.assertEqual(
            inventory["predecessor_overlap"],
            {
                "no_think_base": 0,
                "no_think_program_slot": 0,
                "thinking_base": 0,
            },
        )
        self.assertLessEqual(
            max(self.receipt["registered_max_context_tokens"].values()),
            self.receipt["max_model_len"],
        )
        self.assertFalse(
            any("benchmarks/" in path for path in self.receipt["read_receipt"])
        )
        self.assertFalse(
            any("mechanics_gold" in path for path in self.receipt["read_receipt"])
        )


if __name__ == "__main__":
    unittest.main()
