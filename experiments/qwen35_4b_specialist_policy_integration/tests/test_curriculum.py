from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from curriculum import (  # noqa: E402
    ALL_PROCESS_FAMILIES,
    classify_action,
    expert_decision,
    semantic_group_diagnostics,
)
from gym.families import load as load_family  # noqa: E402
from io_utils import domain_families, load_config, training_seed  # noqa: E402
from rollout import collect_expert_demonstrations  # noqa: E402
from eval_proxy import _compact_logprob_receipts  # noqa: E402
from train_sequence_grpo import _policy_sample, _shuffle_group_advantages  # noqa: E402


class _FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, *args, **kwargs):
        return "PROMPT<think>\n"

    def __call__(self, text, add_special_tokens=False):
        if text == "PROMPT<think>\n":
            return {"input_ids": [1, 2, 3]}
        raise AssertionError(text)

    def convert_tokens_to_ids(self, token):
        assert token == "</think>"
        return 9


def _expert_rollout(family_name: str, seed: int, level: int, inject_bad: bool = False):
    family = load_family(family_name)
    episode = family.Episode(seed, level)
    messages = [
        {"role": "system", "content": episode.system_prompt()},
        {"role": "user", "content": episode.initial_observation()},
    ]
    if inject_bad:
        observation, done = episode.step("XYZZY 42 GARBAGE")
        assert not episode.last_action_ok
        messages.extend(
            [
                {"role": "assistant", "content": "XYZZY 42 GARBAGE"},
                {"role": "user", "content": observation},
            ]
        )
        if done:
            return episode
    for _ in range(episode.max_turns):
        decision = expert_decision(family_name, episode, messages)
        assert len(decision.thought.split()) <= 120
        observation, done = episode.step(decision.action)
        assert episode.last_action_ok, (family_name, level, decision.action, observation)
        messages.extend(
            [
                {"role": "assistant", "content": decision.action},
                {"role": "user", "content": observation},
            ]
        )
        if done:
            break
    return episode


class CurriculumTests(unittest.TestCase):
    def test_frozen_domains_partition_training_split_and_seed_list(self):
        config, _ = load_config(EXP / "configs" / "default.yaml")
        resolved = [domain_families(config, name) for name in ("discover", "control", "tools", "compose")]
        flattened = [family for group in resolved for family in group]
        self.assertEqual(flattened, config["split"]["train_families"])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([training_seed(config, index) for index in range(3)], [42, 43, 44])

    def test_state_aware_experts_solve_all_families_and_levels(self):
        for family_name in ALL_PROCESS_FAMILIES:
            family = load_family(family_name)
            for level in family.LEVELS:
                with self.subTest(family=family_name, level=level):
                    episode = _expert_rollout(
                        family_name, seed=81000 + level, level=level
                    )
                    self.assertTrue(math.isclose(episode.score(), 1.0))

    def test_experts_recover_after_one_malformed_action_where_slack_exists(self):
        # Level 2 generators reserve recovery slack in every family used for
        # the live-state curriculum.  This is the DAgger-specific contract,
        # not merely untouched-oracle replay.
        for offset, family_name in enumerate(ALL_PROCESS_FAMILIES):
            with self.subTest(family=family_name):
                episode = _expert_rollout(
                    family_name,
                    seed=82000 + offset,
                    level=2,
                    inject_bad=True,
                )
                # Spindle permanently scores first-try accuracy; a malformed
                # first attempt is finish-recoverable but not score-recoverable.
                if family_name == "spindle":
                    self.assertTrue(episode._done)
                    self.assertGreater(episode.score(), 0.0)
                else:
                    self.assertTrue(math.isclose(episode.score(), 1.0))

    def test_action_classifier_and_semantic_uncertainty(self):
        self.assertEqual(classify_action("PROBE za-ke"), "PROBE")
        self.assertEqual(classify_action("PATCH 2 ADD A 1"), "REVISE")
        self.assertEqual(classify_action("RUN"), "VERIFY")
        self.assertEqual(classify_action("ANSWER: 7"), "COMMIT")
        self.assertEqual(classify_action("CALL make(x)"), "TOOL")
        self.assertEqual(classify_action(""), "INVALID")

        rows = [
            {"score": 1.0, "turns": [{"action": "RUN"}]},
            {"score": 0.0, "turns": [{"action": "PATCH 1 ZERO A"}]},
            {"score": 0.0, "turns": [{"action": "PATCH 1 ZERO A"}]},
            {"score": 1.0, "turns": [{"action": "RUN"}]},
        ]
        stats = semantic_group_diagnostics(rows)
        self.assertEqual(stats["operator_counts"], {"REVISE": 2, "VERIFY": 2})
        self.assertGreater(stats["operator_entropy"], 0.6)
        self.assertTrue(math.isclose(stats["mean_score"], 0.5))
        self.assertTrue(math.isclose(stats["outcome_variance"], 0.25))
        self.assertFalse(stats["constant_outcome"])

    def test_expert_demonstration_rows_keep_visible_context_separate(self):
        trajectories, rows = collect_expert_demonstrations(
            [("loomfix", 2, 83001), ("ferrier", 2, 83002)]
        )
        self.assertEqual(len(trajectories), 2)
        self.assertGreater(len(rows), 2)
        for row in rows:
            self.assertEqual(row["messages"][-1]["role"], "user")
            self.assertNotIn("oracle", "\n".join(m["content"].lower() for m in row["messages"]))
            self.assertIn("OBSERVE:", row["think"])
            self.assertNotIn("\n", row["answer"].strip())

    def test_policy_mask_excludes_harness_injected_close(self):
        trajectory = {
            "rid": "x",
            "family": "ferrier",
            "level": 2,
            "episode_key": "x",
            "advantage": 1.0,
            "turns": [1, 2],
        }
        turn = {
            "turn": 0,
            "messages_before": [{"role": "user", "content": "x"}],
            "policy": {
                "n_stage1_prompt_tokens": 3,
                "token_ids": [10, 11, 9, 12, 20, 21],
                "forced_close": True,
                "retained_thinking_token_ids": [10, 11],
                "injected_token_ids": [9, 12],
            },
        }
        sample = _policy_sample(_FakeTokenizer(), trajectory, turn, 32, 0.2)
        self.assertEqual(sample["completion_weights"], [0.2, 0.2, 0.0, 0.0, 1.0, 1.0])

    def test_eval_compacts_top20_payload_to_entropy_sufficient_statistics(self):
        rows = [
            {
                "spec": {"hidden": "never persist in eval rows"},
                "turns": [
                    {
                        "expert": {"answer": "hidden label"},
                        "messages_before": [{"role": "user", "content": "visible"}],
                        "policy": {
                            "stage1_logprobs": [
                                {
                                    "1": {"logprob": math.log(0.6)},
                                    "2": {"logprob": math.log(0.3)},
                                }
                            ],
                            "stage2_logprobs": None,
                        }
                    }
                ]
            }
        ]
        _compact_logprob_receipts(rows)
        policy = rows[0]["turns"][0]["policy"]
        self.assertNotIn("spec", rows[0])
        self.assertNotIn("expert", rows[0]["turns"][0])
        self.assertNotIn("messages_before", rows[0]["turns"][0])
        self.assertNotIn("stage1_logprobs", policy)
        self.assertNotIn("stage2_logprobs", policy)
        self.assertEqual(policy["reported_top20_tail_lumped_entropy_positions"], 1)
        self.assertGreater(policy["reported_top20_tail_lumped_entropy_sum"], 0.8)

    def test_shuffled_control_preserves_advantage_vectors(self):
        trajectories = []
        for group, values in (("a", [-1.0, 1.0]), ("b", [-2.0, 2.0]), ("c", [-3.0, 3.0])):
            for rollout, value in enumerate(values):
                trajectories.append(
                    {
                        "episode_key": group,
                        "family": "ferrier",
                        "level": 2,
                        "rollout": rollout,
                        "advantage": value,
                    }
                )
        before = sorted(abs(row["advantage"]) for row in trajectories)
        before_by_group = {
            group: [row["advantage"] for row in trajectories if row["episode_key"] == group]
            for group in ("a", "b", "c")
        }
        digest = _shuffle_group_advantages(trajectories, 43)
        after = sorted(abs(row["advantage"]) for row in trajectories)
        self.assertEqual(before, after)
        for group in before_by_group:
            moved = [
                row["advantage"]
                for row in trajectories
                if row["episode_key"] == group
            ]
            self.assertNotEqual(moved, before_by_group[group])
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
