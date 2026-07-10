from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
SPEC = importlib.util.spec_from_file_location("partial_structure_run_search", EXP / "scripts" / "run_search.py")
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def output(index: int, sampled: int, stage1_prompt: int = 10, stage2_prompt: int = 20):
    return {
        "sample_index": index,
        "n_sampled_tokens": sampled,
        "n_stage1_prompt_tokens": stage1_prompt,
        "n_stage2_prompt_tokens": stage2_prompt,
        "n_thinking_tokens": max(0, sampled - 1),
        "n_injected_tokens": 2,
    }


class DirectPrefixAccountingTests(unittest.TestCase):
    def test_prefix_is_deterministic_maximal_and_reports_next_cost(self):
        rows = [output(2, 7), output(0, 6), output(1, 5)]
        kept, spent, next_cost, exhausted = R._prefix_under_cap(rows, 12, "sampled_tokens")
        self.assertEqual([row["sample_index"] for row in kept], [0, 1])
        self.assertEqual(spent, 11)
        self.assertEqual(next_cost, 7)
        self.assertFalse(exhausted)

    def test_pool_exhaustion_is_explicit(self):
        kept, spent, next_cost, exhausted = R._prefix_under_cap(
            [output(0, 6), output(1, 5)], 20, "sampled_tokens"
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(spent, 11)
        self.assertIsNone(next_cost)
        self.assertTrue(exhausted)

    def test_total_token_basis_and_resource_vector_include_both_prefills(self):
        rows = [output(0, 6, 10, 20), output(1, 5, 11, 21)]
        self.assertEqual(R._direct_cost(rows[0], "total_model_tokens"), 36)
        accounting = R._direct_accounting(rows)
        self.assertEqual(accounting["requests"], 4)
        self.assertEqual(accounting["prefill_tokens"], 62)
        self.assertEqual(accounting["sampled_tokens"], 11)
        self.assertEqual(accounting["total_model_tokens"], 73)

    def test_naturally_closed_output_charges_only_one_request(self):
        row = output(0, 17, 10, 0)
        accounting = R._direct_accounting([row])
        self.assertEqual(accounting["requests"], 1)
        self.assertEqual(accounting["completions"], 1)


if __name__ == "__main__":
    unittest.main()
