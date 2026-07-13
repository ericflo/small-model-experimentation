from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from training_units import (  # noqa: E402
    fit_prompt_around_completion,
    make_sparse_sample,
    offpolicy_prompt_and_completion,
    prompt_and_student_completion,
)


class DummyTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "rendered"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8]}


class TrainingUnitTests(unittest.TestCase):
    def test_overlength_episode_preserves_completion_and_left_truncates_prompt(self):
        class LongPromptTokenizer(DummyTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(range(3000))}

        unit = {
            "state_id": "episode-overlength",
            "family": "loomfix",
            "kind": "episode",
            "level": 5,
            "role": "route_control",
            "primary_teacher": "deep",
            "state": {
                "kind": "episode",
                "messages": [{"role": "user", "content": "x"}],
                "selected_student_turn": {
                    "token_ids": list(range(10_000, 10_203)),
                    "n_thinking_tokens": 203,
                    "injected_token_ids": [],
                },
            },
        }
        sample = make_sparse_sample(
            unit,
            LongPromptTokenizer(),
            max_positions=256,
            max_length=3072,
        )
        self.assertEqual(sample["completion_ids"].tolist(), list(range(10_000, 10_203)))
        self.assertEqual(sample["prompt_ids"].tolist(), list(range(131, 3000)))
        self.assertEqual(sample["positions"].tolist(), list(range(203)))
        self.assertEqual(sample["meta"]["original_prompt_tokens"], 3000)
        self.assertEqual(sample["meta"]["prompt_tokens_truncated"], 131)
        self.assertEqual(sample["meta"]["input_tokens"], 3072)

    def test_completion_must_leave_one_causal_prompt_token(self):
        with self.assertRaisesRegex(ValueError, "leaves no prompt budget"):
            fit_prompt_around_completion(
                [1],
                list(range(3072)),
                max_length=3072,
                state_id="too-long-completion",
            )

    def test_atom_masks_only_injected_close(self):
        unit = {
            "state_id": "a",
            "state": {
                "kind": "atom",
                "exact_prompt_token_ids": [1, 2],
                "student_suffix_ids": [3, 4, 99, 5],
                "prefix_length": 2,
                "student_output": {
                    "n_thinking_tokens": 4,
                    "injected_token_ids": [99],
                },
            },
        }
        prompt, completion, active = prompt_and_student_completion(unit, DummyTokenizer())
        self.assertEqual(prompt, [1, 2])
        self.assertEqual(completion, [3, 4, 99, 5])
        self.assertEqual(active, [0, 1, 3])

    def test_episode_uses_exact_visible_messages_and_masks_close(self):
        unit = {
            "state_id": "e",
            "state": {
                "kind": "episode",
                "messages": [{"role": "user", "content": "x"}],
                "selected_student_turn": {
                    "token_ids": [10, 99, 11],
                    "n_thinking_tokens": 1,
                    "injected_token_ids": [99],
                },
            },
        }
        prompt, completion, active = prompt_and_student_completion(unit, DummyTokenizer())
        self.assertEqual(prompt, [7, 8])
        self.assertEqual(completion, [10, 99, 11])
        self.assertEqual(active, [0, 2])

    def test_offpolicy_target_masks_unique_injected_close(self):
        unit = {
            "state_id": "o",
            "role": "capability",
            "state": {"kind": "atom", "exact_prompt_token_ids": [1, 2]},
            "offpolicy_target": {
                "completion_ids": [5, 99, 6],
                "injected_token_ids": [99],
            },
        }
        prompt, completion, active = offpolicy_prompt_and_completion(
            unit, DummyTokenizer()
        )
        self.assertEqual(prompt, [1, 2])
        self.assertEqual(completion, [5, 99, 6])
        self.assertEqual(active, [0, 2])

    def test_offpolicy_target_rejects_ambiguous_injected_close(self):
        unit = {
            "state_id": "o2",
            "role": "capability",
            "state": {"kind": "atom", "exact_prompt_token_ids": [1]},
            "offpolicy_target": {
                "completion_ids": [99, 3, 99],
                "injected_token_ids": [99],
            },
        }
        with self.assertRaisesRegex(ValueError, "unique"):
            offpolicy_prompt_and_completion(unit, DummyTokenizer())


if __name__ == "__main__":
    unittest.main()
