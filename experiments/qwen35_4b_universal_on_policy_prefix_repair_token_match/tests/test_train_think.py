from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_think.py"
SPEC = importlib.util.spec_from_file_location("state_table_trainer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    eos_token = "<EOS>"
    eos_token_id = 248046

    def apply_chat_template(self, *unused_args, **unused_kwargs) -> str:
        return "<PROMPT>"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}

    def convert_tokens_to_ids(self, token: str) -> int:
        self.assert_token = token
        return 248069

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token) for token in token_ids)


class OrdinaryWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.row = {
            "messages": [{"role": "user", "content": "do it"}],
            "kind": "u_state_table_score",
            "think": "reason",
            "answer": "ANSWER: result",
            "row_weight": 1.0,
        }

    def encode(self, **overrides):
        arguments = {
            "rec": self.row,
            "tok": self.tokenizer,
            "max_length": 4096,
            "w_think": 0.2,
            "w_close": 0.2,
        }
        arguments.update(overrides)
        return MODULE.encode_row(**arguments)

    def test_span_weights_are_prompt_zero_think_and_close_point_two_answer_one(self) -> None:
        encoded = self.encode()
        self.assertIsNotNone(encoded)
        prompt_end = len("<PROMPT>")
        think_end = len("<PROMPT>reason\n")
        close_end = len("<PROMPT>reason\n</think>\n\n")
        weights = encoded["loss_weights"]
        self.assertTrue(all(value == 0.0 for value in weights[:prompt_end]))
        self.assertTrue(all(value == 0.2 for value in weights[prompt_end:think_end]))
        self.assertTrue(all(value == 0.2 for value in weights[think_end:close_end]))
        self.assertTrue(all(value == 1.0 for value in weights[close_end:]))

    def test_forced_close_row_suppresses_thought_but_keeps_ordinary_close(self) -> None:
        encoded = self.encode(rec=dict(self.row, kind="atom_fc"))
        self.assertIsNotNone(encoded)
        prompt_end = len("<PROMPT>")
        think_end = len("<PROMPT>reason\n")
        close_end = len("<PROMPT>reason\n</think>\n\n")
        weights = encoded["loss_weights"]
        self.assertTrue(all(value == 0.0 for value in weights[:think_end]))
        self.assertTrue(all(value == 0.2 for value in weights[think_end:close_end]))
        self.assertTrue(all(value == 1.0 for value in weights[close_end:]))

    def test_negative_row_never_pushes_down_thought_or_close(self) -> None:
        encoded = self.encode(rec=dict(self.row, row_weight=-1.0))
        self.assertIsNotNone(encoded)
        answer_start = len("<PROMPT>reason\n</think>\n\n")
        self.assertTrue(all(value == 0.0 for value in encoded["loss_weights"][:answer_start]))
        self.assertTrue(all(value == -1.0 for value in encoded["loss_weights"][answer_start:]))

    def test_overlength_row_is_rejected_instead_of_truncated(self) -> None:
        full = self.encode()
        self.assertIsNotNone(full)
        self.assertIsNone(self.encode(max_length=len(full["input_ids"]) - 1))

    def test_on_policy_parent_prefix_is_exact_and_fully_masked(self) -> None:
        prefix = "parent mistake\n"
        rec = dict(
            self.row,
            assistant_prefix_token_ids=[ord(char) for char in prefix],
            assistant_prefix_text=prefix,
            prefix_loss_masked=True,
        )
        encoded = self.encode(rec=rec)
        self.assertIsNotNone(encoded)
        target_start = len("<PROMPT>") + len(prefix)
        think_end = target_start + len("reason\n")
        close_end = think_end + len("</think>\n\n")
        weights = encoded["loss_weights"]
        self.assertTrue(all(value == 0.0 for value in weights[:target_start]))
        self.assertTrue(all(value == 0.2 for value in weights[target_start:think_end]))
        self.assertTrue(all(value == 0.2 for value in weights[think_end:close_end]))
        self.assertTrue(all(value == 1.0 for value in weights[close_end:]))
        self.assertTrue(all(value == -100 for value in encoded["labels"][:target_start]))

    def test_on_policy_prefix_rejects_unmasked_or_closed_context(self) -> None:
        base = dict(
            self.row,
            assistant_prefix_token_ids=[ord("x")],
            assistant_prefix_text="x",
        )
        self.assertIsNone(self.encode(rec=base))
        self.assertIsNone(
            self.encode(
                rec=dict(
                    base,
                    prefix_loss_masked=True,
                    assistant_prefix_token_ids=[248069],
                    assistant_prefix_text="",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
