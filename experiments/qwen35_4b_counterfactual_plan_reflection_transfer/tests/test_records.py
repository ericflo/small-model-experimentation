from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import records as R  # noqa: E402
import taskgen as T  # noqa: E402


class CharacterTokenizer:
    eos_token = "<EOS>"
    eos_token_id = 248046

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.last_messages = messages
        if kwargs != {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
        }:
            raise AssertionError(f"unexpected template arguments: {kwargs}")
        content = "|".join(row["content"] for row in messages)
        return f"<PROMPT>{content}<think>\n"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict:
        if add_special_tokens is not False:
            raise AssertionError("special tokens must be disabled")
        return {"input_ids": [ord(char) for char in text]}


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        corpus = T.build_corpus({"train": 6}, 73_301)
        cls.arms, cls.receipt = R.build_training_records(
            corpus["train"], shuffle_seed=73_319, schedule_seed=73_323
        )

    def test_all_arms_share_task_and_optimizer_schedule(self) -> None:
        expected = [row["task_id"] for row in self.arms["reflection_correct"]]
        self.assertEqual(self.receipt["rows_per_arm"], 18)
        self.assertEqual(self.receipt["optimizer_groups_per_epoch"], 1)
        for arm in R.TRAINING_ARMS:
            self.assertEqual([row["task_id"] for row in self.arms[arm]], expected)

    def test_causal_arms_are_context_identical_and_step_target_matched(self) -> None:
        correct = self.arms["reflection_correct"]
        shuffled = self.arms["reflection_shuffled"]
        self.assertEqual(
            [row["messages"] for row in correct], [row["messages"] for row in shuffled]
        )
        self.assertEqual(
            sorted((row["think"], row["answer"]) for row in correct),
            sorted((row["think"], row["answer"]) for row in shuffled),
        )
        for left, right in zip(correct, shuffled):
            self.assertEqual(left["truth_ops"], right["truth_ops"])
            self.assertNotEqual(left["supervision_ops"], right["supervision_ops"])

    def test_auxiliary_arm_changes_only_framing_and_keeps_correct_target(self) -> None:
        reflection = self.arms["reflection_correct"]
        auxiliary = self.arms["auxiliary_plan_label_correct"]
        for left, right in zip(reflection, auxiliary):
            self.assertEqual(left["messages"][:-1], right["messages"][:-1])
            self.assertNotEqual(left["messages"][-1], right["messages"][-1])
            self.assertEqual(left["think"], right["think"])
            self.assertEqual(left["answer"], right["answer"])
        # Character equality is a model-free guard; tokenizer equality is a
        # separate fail-closed receipt before training.
        self.assertEqual(len(T.REFLECTION_QUESTION), len(R.AUXILIARY_QUESTION))

    def test_direct_positive_control_trains_the_actual_answer_branch(self) -> None:
        for row in self.arms["direct_plan_answer_positive_control"]:
            self.assertEqual(row["messages"][-1]["content"], T.ACTION_QUESTION)
            self.assertTrue(row["answer"].startswith("ANSWER: "))
            self.assertTrue(row["noncomparable_positive_control"])

    def test_exact_final_assistant_mask_and_token_parity(self) -> None:
        tokenizer = CharacterTokenizer()
        encoded = {
            arm: [
                R.encode_training_record(row, tokenizer, 10_000, 0.2, 0.2)
                for row in rows
            ]
            for arm, rows in self.arms.items()
        }
        parity = R.validate_tokenized_parity(self.arms, encoded)
        self.assertEqual(
            parity["totals"]["reflection_correct"],
            parity["totals"]["reflection_shuffled"],
        )
        for row in encoded["reflection_correct"]:
            prompt_tokens = row["prompt_tokens"]
            self.assertTrue(all(label == -100 for label in row["labels"][:prompt_tokens]))
            self.assertTrue(all(label != -100 for label in row["labels"][prompt_tokens:]))
            self.assertTrue(all(weight == 0.0 for weight in row["loss_weights"][:prompt_tokens]))

    def test_overlength_fails_instead_of_truncating(self) -> None:
        tokenizer = CharacterTokenizer()
        record = self.arms["reflection_correct"][0]
        full = R.encode_training_record(record, tokenizer, 10_000, 0.2, 0.2)
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            R.encode_training_record(
                record, tokenizer, len(full["input_ids"]) - 1, 0.2, 0.2
            )


if __name__ == "__main__":
    unittest.main()
