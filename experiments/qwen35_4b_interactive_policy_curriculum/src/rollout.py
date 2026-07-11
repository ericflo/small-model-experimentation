"""Live lockstep episode rollout with DAgger labels and policy-token receipts."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Iterable

from curriculum import expert_decision, semantic_group_diagnostics
from gym import base
from gym.families import load as load_family
from vllm_runner import SamplingConfig, VLLMRunner


POLICY_OUTPUT_FIELDS = (
    "sample_index",
    "stage1_parent_seed",
    "seed_stage1",
    "seed_stage2",
    "text",
    "token_ids",
    "stage1_token_ids",
    "retained_thinking_token_ids",
    "injected_token_ids",
    "stage2_token_ids",
    "n_thinking_tokens",
    "n_answer_tokens",
    "n_sampled_tokens",
    "n_injected_tokens",
    "n_completion_tokens",
    "n_terminal_tokens_trimmed",
    "n_stage1_prompt_tokens",
    "n_stage2_prompt_tokens",
    "thinking_closed",
    "forced_close",
    "finish_reason",
    "stop_reason",
    "stage1_finish_reason",
    "stage1_stop_reason",
    "truncated",
    "stage1_cumulative_logprob",
    "stage2_cumulative_logprob",
    "sampled_cumulative_logprob",
)


def policy_output_receipt(output: dict[str, Any]) -> dict[str, Any]:
    """Retain tokens needed to reconstruct sampled-policy loss exactly."""
    return {field: copy.deepcopy(output.get(field)) for field in POLICY_OUTPUT_FIELDS}


def sampling_config(
    *,
    think_budget: int,
    answer_max_tokens: int,
    run_seed: int,
    greedy: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> SamplingConfig:
    return SamplingConfig(
        thinking="budget",
        thinking_budget=int(think_budget),
        n=1,
        answer_max_tokens=int(answer_max_tokens),
        greedy=bool(greedy),
        temperature=None if greedy else temperature,
        top_p=None if greedy else top_p,
        top_k=None if greedy else top_k,
        run_seed=int(run_seed),
    )


def _new_rollout(family_name: str, level: int, ep_seed: int, rollout_index: int):
    family = load_family(family_name)
    episode = family.Episode(ep_seed, level)
    return {
        "rid": f"{family_name}-L{level}-e{ep_seed}-r{rollout_index}",
        "episode_key": f"{family_name}-L{level}-e{ep_seed}",
        "family": family_name,
        "level": int(level),
        "ep_seed": int(ep_seed),
        "rollout": int(rollout_index),
        "episode": episode,
        "messages": [
            {"role": "system", "content": episode.system_prompt()},
            {"role": "user", "content": episode.initial_observation()},
        ],
        "turns": [],
        "done": False,
    }


def collect_policy_episodes(
    runner: VLLMRunner,
    specs: Iterable[tuple[str, int, int]],
    *,
    rollouts_per_episode: int,
    think_budget: int,
    answer_max_tokens: int,
    run_seed: int,
    greedy: bool,
    rollout_offset: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Roll out a policy and label every pre-action live state with an expert.

    The environment is mutated only by the sampled model action.  The expert
    label is computed immediately before that action against the same current
    state and visible message transcript.
    """
    k = 1 if greedy else int(rollouts_per_episode)
    rollouts = [
        _new_rollout(family_name, level, ep_seed, rollout_index)
        for family_name, level, ep_seed in specs
        for local_index in range(k)
        for rollout_index in [int(rollout_offset) + local_index]
    ]
    sampling = sampling_config(
        think_budget=think_budget,
        answer_max_tokens=answer_max_tokens,
        run_seed=run_seed,
        greedy=greedy,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    max_horizon = max((r["episode"].max_turns for r in rollouts), default=0)
    for turn_index in range(max_horizon):
        active = [
            rollout
            for rollout in rollouts
            if not rollout["done"] and turn_index < rollout["episode"].max_turns
        ]
        if not active:
            break
        records = []
        decisions = []
        for rollout in active:
            messages_before = copy.deepcopy(rollout["messages"])
            decision = expert_decision(
                rollout["family"], rollout["episode"], messages_before
            )
            decisions.append((decision, messages_before))
            records.append(
                {
                    "id": f"{rollout['rid']}-t{turn_index}",
                    "messages": messages_before,
                }
            )
        rows, _ = runner.generate(records, sampling)
        for rollout, row, (decision, messages_before) in zip(active, rows, decisions):
            output = row["outputs"][0]
            action = base.extract_action(output["text"])
            observation, done = rollout["episode"].step(action)
            receipt = policy_output_receipt(output)
            rollout["turns"].append(
                {
                    "turn": turn_index,
                    "messages_before": messages_before,
                    "expert": decision.to_dict(),
                    "action": action,
                    "action_ok": bool(
                        getattr(rollout["episode"], "last_action_ok", True)
                    ),
                    "observation": observation,
                    "policy": receipt,
                }
            )
            rollout["messages"].append({"role": "assistant", "content": action})
            rollout["messages"].append({"role": "user", "content": observation})
            rollout["done"] = bool(done)
        if progress:
            live = sum(not rollout["done"] for rollout in rollouts)
            print(
                f"[interactive-rollout] turn={turn_index} active={len(active)} live={live}",
                flush=True,
            )

    results: list[dict[str, Any]] = []
    for rollout in rollouts:
        episode = rollout["episode"]
        turns = rollout["turns"]
        results.append(
            {
                "rid": rollout["rid"],
                "episode_key": rollout["episode_key"],
                "family": rollout["family"],
                "level": rollout["level"],
                "ep_seed": rollout["ep_seed"],
                "rollout": rollout["rollout"],
                "spec": copy.deepcopy(episode.spec),
                "system_prompt": rollout["messages"][0]["content"],
                "initial_observation": rollout["messages"][1]["content"],
                "turns": turns,
                "done": bool(rollout["done"]),
                "score": float(episode.score()),
                "n_turns": len(turns),
                "max_turns": int(episode.max_turns),
                "action_valid_rate": (
                    sum(bool(turn["action_ok"]) for turn in turns) / len(turns)
                    if turns
                    else 0.0
                ),
                "natural_close_rate": (
                    sum(not bool(turn["policy"]["forced_close"]) for turn in turns)
                    / len(turns)
                    if turns
                    else 0.0
                ),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result["episode_key"]].append(result)
    group_rows = []
    for key in sorted(groups):
        members = groups[key]
        diagnostic = semantic_group_diagnostics(members)
        diagnostic.update(
            {
                "episode_key": key,
                "family": members[0]["family"],
                "level": members[0]["level"],
                "ep_seed": members[0]["ep_seed"],
            }
        )
        group_rows.append(diagnostic)
    summary = summarize_trajectories(results, group_rows)
    return results, summary


def collect_expert_demonstrations(
    specs: Iterable[tuple[str, int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive state-aware experts on untouched episodes without a model call."""
    trajectories: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for family_name, level, ep_seed in specs:
        family = load_family(family_name)
        episode = family.Episode(ep_seed, level)
        messages = [
            {"role": "system", "content": episode.system_prompt()},
            {"role": "user", "content": episode.initial_observation()},
        ]
        turns = []
        for turn_index in range(episode.max_turns):
            messages_before = copy.deepcopy(messages)
            decision = expert_decision(family_name, episode, messages_before)
            observation, done = episode.step(decision.action)
            if not getattr(episode, "last_action_ok", True):
                raise RuntimeError(
                    f"expert emitted invalid action at {family_name}/L{level}/{ep_seed}: "
                    f"{decision.action!r} -> {observation!r}"
                )
            row = dagger_row(
                family=family_name,
                level=level,
                ep_seed=ep_seed,
                turn=turn_index,
                messages=messages_before,
                decision=decision.to_dict(),
                source="expert_demo",
                source_score=1.0,
            )
            rows.append(row)
            turns.append(
                {
                    "turn": turn_index,
                    "messages_before": messages_before,
                    "expert": decision.to_dict(),
                    "action": decision.action,
                    "action_ok": True,
                    "observation": observation,
                }
            )
            messages.extend(
                [
                    {"role": "assistant", "content": decision.action},
                    {"role": "user", "content": observation},
                ]
            )
            if done:
                break
        score = float(episode.score())
        if score < 0.999:
            raise RuntimeError(
                f"expert demonstration failed {family_name}/L{level}/{ep_seed}: {score}"
            )
        trajectories.append(
            {
                "rid": f"expert-{family_name}-L{level}-e{ep_seed}",
                "episode_key": f"{family_name}-L{level}-e{ep_seed}",
                "family": family_name,
                "level": level,
                "ep_seed": ep_seed,
                "rollout": 0,
                "turns": turns,
                "done": True,
                "score": score,
                "n_turns": len(turns),
                "max_turns": episode.max_turns,
            }
        )
    return trajectories, rows


def dagger_row(
    *,
    family: str,
    level: int,
    ep_seed: int,
    turn: int,
    messages: list[dict[str, str]],
    decision: dict[str, Any],
    source: str,
    source_score: float,
    episode_key: str | None = None,
    rollout: int | None = None,
) -> dict[str, Any]:
    return {
        "id": (
            f"dagger-{source}-{family}-L{level}-e{ep_seed}"
            f"-r{rollout if rollout is not None else 0}-t{turn}"
        ),
        "family": family,
        "level": int(level),
        "kind": "dagger_expert_demo" if source == "expert_demo" else "dagger_visited",
        "messages": copy.deepcopy(messages),
        "think": str(decision["thought"]),
        "answer": str(decision["action"]),
        "operator": str(decision["operator"]),
        "source": source,
        "source_score": float(source_score),
        "episode_key": episode_key or f"{family}-L{level}-e{ep_seed}",
        "ep_seed": int(ep_seed),
        "rollout": rollout,
        "turn": int(turn),
    }


def visited_dagger_rows(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for turn in trajectory.get("turns", []):
            rows.append(
                dagger_row(
                    family=trajectory["family"],
                    level=trajectory["level"],
                    ep_seed=trajectory["ep_seed"],
                    turn=turn["turn"],
                    messages=turn["messages_before"],
                    decision=turn["expert"],
                    source="policy_visited",
                    source_score=trajectory["score"],
                    episode_key=trajectory["episode_key"],
                    rollout=trajectory["rollout"],
                )
            )
    return rows


def summarize_trajectories(
    trajectories: list[dict[str, Any]],
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_family[row["family"]].append(row)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
        n = len(rows)
        return {
            "n": n,
            "mean_score": sum(float(row["score"]) for row in rows) / n if n else 0.0,
            "exact_rate": sum(float(row["score"]) >= 0.999 for row in rows) / n if n else 0.0,
            "mean_turns": sum(int(row["n_turns"]) for row in rows) / n if n else 0.0,
            "action_valid_rate": (
                sum(float(row.get("action_valid_rate", 1.0)) for row in rows) / n if n else 0.0
            ),
            "natural_close_rate": (
                sum(float(row.get("natural_close_rate", 1.0)) for row in rows) / n if n else 0.0
            ),
        }

    family_metrics = {name: metrics(rows) for name, rows in sorted(by_family.items())}
    macro = (
        sum(float(row["mean_score"]) for row in family_metrics.values())
        / len(family_metrics)
        if family_metrics
        else 0.0
    )
    return {
        "overall": metrics(trajectories),
        "family_macro_score": macro,
        "by_family": family_metrics,
        "groups": groups or [],
        "n_nonconstant_groups": sum(
            not bool(group.get("constant_outcome", True)) for group in (groups or [])
        ),
    }
