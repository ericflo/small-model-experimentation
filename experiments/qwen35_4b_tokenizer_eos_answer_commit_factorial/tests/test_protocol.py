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


if __name__ == "__main__":
    unittest.main()
