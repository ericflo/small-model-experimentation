from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym.families import load as load_family  # noqa: E402
from state_replay import (  # noqa: E402
    build_atom_state,
    build_episode_state,
    replay_episode_state,
    select_balanced_states,
)


class StateReplayTests(unittest.TestCase):
    def _failed_episode_row(self) -> dict:
        family = load_family("kilnrite")
        episode = family.Episode(12345, 2)
        system = episode.system_prompt()
        initial = episode.initial_observation()
        turns = []
        for turn_index in range(2):
            action = "NOT A VALID ACTION"
            observation, done = episode.step(action)
            turns.append(
                {
                    "turn": turn_index,
                    "action": action,
                    "action_ok": bool(episode.last_action_ok),
                    "observation": observation,
                    "token_ids": [10, 11, 12],
                    "thinking_closed": True,
                    "forced_close": False,
                    "finish_reason": "stop",
                    "truncated": False,
                    "n_thinking_tokens": 1,
                    "n_answer_tokens": 2,
                    "n_sampled_tokens": 3,
                    "injected_token_ids": [],
                }
            )
            if done:
                break
        return {
            "rid": "synthetic-failed-episode",
            "family": "kilnrite",
            "level": 2,
            "ep_seed": 12345,
            "score": float(episode.score()),
            "system_prompt": system,
            "initial_observation": initial,
            "turns": turns,
        }

    def test_episode_state_reconstructs_exact_visible_history(self):
        state = build_episode_state(
            self._failed_episode_row(), block=0, failure_ceiling=0.999999
        )
        self.assertIsNotNone(state)
        self.assertEqual(state["selected_turn_index"], 0)
        self.assertEqual(state["selection_reason"], "first_invalid_action")
        episode, messages = replay_episode_state(state)
        self.assertEqual(messages, state["messages"])
        self.assertEqual(episode.score(), 0.0)

    def test_episode_replay_rejects_tampered_observation(self):
        row = self._failed_episode_row()
        # Make the second turn the selected turn so turn zero enters past_turns.
        row["turns"][0]["action_ok"] = True
        # Rebuild with an actually valid first action to get a replayable state.
        family = load_family("kilnrite")
        episode = family.Episode(12345, 2)
        policy = family.OraclePolicy(episode)
        action = policy.act([episode.initial_observation()])
        observation, _ = episode.step(action)
        row["turns"][0].update(
            {"action": action, "action_ok": True, "observation": observation}
        )
        state = build_episode_state(row, block=0, failure_ceiling=0.999999)
        tampered = copy.deepcopy(state)
        tampered["past_turns"][0]["observation"] += " tampered"
        with self.assertRaisesRegex(ValueError, "observation replay mismatch"):
            replay_episode_state(tampered)

    def test_atom_prefix_is_exact_and_leaves_a_suffix(self):
        close = 99
        row = {
            "id": "atom-1",
            "family": "caravan",
            "level": 1,
            "prompt": "p",
            "gold": 3,
            "outputs": [
                {
                    "score": 0.0,
                    "token_ids": [1, 2, 3, 4, close, 5, 6],
                    "n_thinking_tokens": 4,
                    "n_answer_tokens": 2,
                    "n_sampled_tokens": 7,
                    "thinking_closed": True,
                    "forced_close": False,
                    "finish_reason": "stop",
                    "truncated": False,
                    "injected_token_ids": [],
                }
            ],
        }
        state = build_atom_state(
            row,
            block=0,
            prompt_token_ids=[20, 21],
            think_close_token_id=close,
            prefix_fraction=0.5,
            prefix_min_tokens=1,
            prefix_max_tokens=128,
            failure_ceiling=0.999999,
        )
        self.assertEqual(state["student_prefix_ids"], [1, 2])
        self.assertEqual(state["exact_prompt_token_ids"], [20, 21, 1, 2])
        self.assertEqual(state["student_suffix_ids"], [3, 4, close, 5, 6])

    def test_balanced_selection_fails_when_a_kind_is_short(self):
        candidates = [
            {"state_id": f"a-{i}", "kind": "atom", "family": "f", "level": 1}
            for i in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "eligible balanced episode"):
            select_balanced_states(candidates, atom_count=2, episode_count=1)


if __name__ == "__main__":
    unittest.main()
