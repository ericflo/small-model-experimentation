from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_think_close.py"
SPEC = importlib.util.spec_from_file_location("close_weight_trainer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    eos_token = "<EOS>"

    def apply_chat_template(self, *unused_args, **unused_kwargs) -> str:
        return "<PROMPT>"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


class CloseWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.row = {
            "messages": [{"role": "user", "content": "do it"}],
            "kind": "u_execute",
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
        encoded = MODULE.encode_row(**arguments)
        self.assertIsNotNone(encoded)
        return encoded

    def test_target_treatment_changes_only_autonomous_close_weights(self) -> None:
        standard = self.encode()
        treatment = self.encode(
            target_close_kinds=frozenset({"u_execute", "u_induct"}),
            target_w_close=1.0,
        )

        self.assertEqual(standard["input_ids"], treatment["input_ids"])
        self.assertEqual(standard["attention_mask"], treatment["attention_mask"])
        self.assertEqual(standard["labels"], treatment["labels"])
        changed = [
            index for index, (left, right) in enumerate(
                zip(standard["loss_weights"], treatment["loss_weights"], strict=True)
            )
            if left != right
        ]
        close = "</think>\n\n"
        close_start = len("<PROMPT>reason\n")
        self.assertEqual(changed, list(range(close_start, close_start + len(close))))
        self.assertTrue(all(standard["loss_weights"][index] == 0.2 for index in changed))
        self.assertTrue(all(treatment["loss_weights"][index] == 1.0 for index in changed))

    def test_non_target_replay_row_is_unchanged(self) -> None:
        replay = dict(self.row, kind="episode")
        standard = self.encode(rec=replay)
        treatment = self.encode(
            rec=replay,
            target_close_kinds=frozenset({"u_execute", "u_induct"}),
            target_w_close=1.0,
        )
        self.assertEqual(standard, treatment)

    def test_negative_row_never_pushes_down_think_or_close(self) -> None:
        negative = dict(self.row, row_weight=-1.0)
        encoded = self.encode(
            rec=negative,
            target_close_kinds=frozenset({"u_execute"}),
            target_w_close=1.0,
        )
        answer_start = len("<PROMPT>reason\n</think>\n\n")
        self.assertTrue(all(weight == 0.0 for weight in encoded["loss_weights"][:answer_start]))
        self.assertTrue(all(weight == -1.0 for weight in encoded["loss_weights"][answer_start:]))


if __name__ == "__main__":
    unittest.main()
