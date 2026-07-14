from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "src" / "protocol.py"
SPEC = importlib.util.spec_from_file_location("commit_protocol", PROTOCOL_PATH)
assert SPEC and SPEC.loader
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


class AnswerCommitProtocolTests(unittest.TestCase):
    def test_frozen_smoke_cases(self) -> None:
        cases = protocol.smoke_cases()
        protocol.validate_smoke_cases(cases)
        self.assertTrue(cases["tokenizer_clean"].strict_exact)
        self.assertFalse(cases["hf_boundary_control"].strict_exact)

    def test_first_stop_is_terminal_and_unique(self) -> None:
        result = protocol.evaluate_answer_commit(
            [10, 248046, 11, 248046],
            stop_token_id=248046,
            expected_token_ids=[10],
            policy="tokenizer_eos_answer_stage",
        )
        self.assertFalse(result.valid_trace)
        self.assertEqual(result.failure, "tokens_after_first_registered_stop")

    def test_every_precommit_token_is_grammar_content(self) -> None:
        result = protocol.evaluate_answer_commit(
            [10, 11, 198, 248046],
            stop_token_id=248046,
            expected_token_ids=[10, 11],
            policy="tokenizer_eos_answer_stage",
        )
        self.assertTrue(result.valid_trace)
        self.assertFalse(result.strict_exact)
        self.assertEqual(result.content_token_ids, (10, 11, 198))

    def test_all_pair_gate_accepts_tokenizer_stop_prefix(self) -> None:
        result = protocol.authenticate_boundary_pair(
            [10, 11, 248046],
            [10, 11, 248046, 198, 248044],
            tokenizer_event="stop",
            hf_event="stop",
            cap=24,
        )
        self.assertTrue(result.valid_pair)
        self.assertEqual(result.compared_tokens, 3)

    def test_all_pair_gate_accepts_early_hf_event(self) -> None:
        result = protocol.authenticate_boundary_pair(
            [10, 248044, 11, 248046],
            [10, 248044],
            tokenizer_event="stop",
            hf_event="stop",
            cap=24,
        )
        self.assertTrue(result.valid_pair)
        self.assertEqual(result.compared_tokens, 2)

    def test_all_pair_gate_authenticates_cap_pairs(self) -> None:
        result = protocol.authenticate_boundary_pair(
            [10, 11, 12],
            [10, 11, 12],
            tokenizer_event="length",
            hf_event="length",
            cap=3,
        )
        self.assertTrue(result.valid_pair)

    def test_all_pair_gate_rejects_prefix_divergence(self) -> None:
        result = protocol.authenticate_boundary_pair(
            [10, 11, 248046],
            [10, 12, 248044],
            tokenizer_event="stop",
            hf_event="stop",
            cap=24,
        )
        self.assertFalse(result.valid_pair)
        self.assertEqual(result.failure, "sampled_prefix_divergence")

    def test_all_pair_gate_rejects_short_length_claim(self) -> None:
        result = protocol.authenticate_boundary_pair(
            [10, 11],
            [10, 11],
            tokenizer_event="length",
            hf_event="length",
            cap=3,
        )
        self.assertFalse(result.valid_pair)
        self.assertEqual(result.failure, "tokenizer_short_output_relabeled_length")

    def test_terminal_stop_on_token_24_is_a_cap_contact(self) -> None:
        sampled = [10] * 23 + [protocol.TOKENIZER_EOS_ID]
        self.assertTrue(
            protocol.is_answer_cap_contact(sampled, finish_reason="stop", cap=24)
        )

    def test_short_stop_is_not_a_cap_contact(self) -> None:
        self.assertFalse(
            protocol.is_answer_cap_contact(
                [10, protocol.TOKENIZER_EOS_ID], finish_reason="stop", cap=24
            )
        )

    def test_suffix_binding_uses_semantic_candidate_first_operation(self) -> None:
        left = protocol.canonical_full_program(
            ["B", "C"], candidate_first_alias="A"
        )
        right = protocol.canonical_full_program(
            ["B", "C"], candidate_first_alias="D"
        )
        self.assertEqual(left, ("A", "B", "C"))
        self.assertNotEqual(left, right)

    def test_suffix_and_direct_share_one_canonical_proposal_type(self) -> None:
        suffix = protocol.canonical_full_program(
            ["B", "C"], candidate_first_alias="A"
        )
        direct = protocol.canonical_full_program(["A", "B", "C"])
        self.assertEqual(suffix, direct)

    def test_canonical_program_rejects_row_identity_instead_of_alias(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside A-X"):
            protocol.canonical_full_program(
                ["B", "C"], candidate_first_alias="candidate-row-7"
            )

    def test_unique_final_early_stop_authenticates_then_scores_content(self) -> None:
        content = protocol.content_for_terminal_event(
            [10, protocol.TOKENIZER_EOS_ID],
            registered_stop_token_id=protocol.TOKENIZER_EOS_ID,
            event="stop",
            cap=24,
        )
        self.assertEqual(content, (10,))

    def test_exact_cap_length_retains_every_token_as_content(self) -> None:
        sampled = list(range(24))
        content = protocol.content_for_terminal_event(
            sampled,
            registered_stop_token_id=protocol.TOKENIZER_EOS_ID,
            event="length",
            cap=24,
        )
        self.assertEqual(content, tuple(sampled))

    def test_claimed_stop_without_registered_stop_is_auth_failure(self) -> None:
        with self.assertRaisesRegex(ValueError, "malformed_registered_stop"):
            protocol.content_for_terminal_event(
                [10, 11],
                registered_stop_token_id=protocol.TOKENIZER_EOS_ID,
                event="stop",
                cap=24,
            )

    def test_canonical_program_id_endpoints(self) -> None:
        self.assertEqual(protocol.canonical_program_id(["A", "A", "A"]), 0)
        self.assertEqual(protocol.canonical_program_id(["X", "X", "X"]), 13823)
        self.assertEqual(
            protocol.canonical_program_id_bytes(["X", "X", "X"]),
            (13823).to_bytes(2, "big"),
        )

    def test_all_canonical_program_ids_are_injective(self) -> None:
        ids = {
            protocol.canonical_program_id((first, second, third))
            for first in protocol.OPERATION_ALIASES
            for second in protocol.OPERATION_ALIASES
            for third in protocol.OPERATION_ALIASES
        }
        byte_ids = {
            protocol.canonical_program_id_bytes((first, second, third))
            for first in protocol.OPERATION_ALIASES
            for second in protocol.OPERATION_ALIASES
            for third in protocol.OPERATION_ALIASES
        }
        self.assertEqual(len(ids), 13824)
        self.assertEqual(len(byte_ids), 13824)


if __name__ == "__main__":
    unittest.main()
