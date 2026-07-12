from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from training_units import (  # noqa: E402
    offpolicy_prompt_and_completion,
    prompt_and_student_completion,
)


class DummyTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "rendered"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 8]}


class TrainingUnitTests(unittest.TestCase):
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
