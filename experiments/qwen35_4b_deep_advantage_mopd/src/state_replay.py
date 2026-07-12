"""Build and exactly replay verifier-backed student states."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from gym.families import load as load_family


def state_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{state_digest(payload)[:24]}"


def build_atom_state(
    row: Mapping[str, Any],
    *,
    block: int,
    prompt_token_ids: Sequence[int],
    think_close_token_id: int,
    prefix_fraction: float,
    prefix_min_tokens: int,
    prefix_max_tokens: int,
    failure_ceiling: float,
    require_failure: bool = True,
) -> dict[str, Any] | None:
    """Convert one failed autonomous atom completion into an exact prefix state."""
    outputs = row.get("outputs") or []
    if len(outputs) != 1:
        raise ValueError("atom state construction requires exactly one student completion")
    output = outputs[0]
    score = float(output["score"])
    if not math.isfinite(score):
        raise ValueError("atom score must be finite")
    is_failure = score < failure_ceiling
    if require_failure and not is_failure:
        return None
    if not require_failure and is_failure:
        return None
    token_ids = [int(value) for value in output["token_ids"]]
    if think_close_token_id in token_ids:
        thought_length = token_ids.index(think_close_token_id)
    else:
        thought_length = int(output.get("n_thinking_tokens", len(token_ids)))
    if thought_length < 2:
        return None
    proposed = math.floor(thought_length * float(prefix_fraction))
    prefix_length = max(int(prefix_min_tokens), proposed)
    prefix_length = min(int(prefix_max_tokens), prefix_length, thought_length - 1)
    if prefix_length < 1:
        return None
    prefix_ids = token_ids[:prefix_length]
    payload = {
        "block": int(block),
        "family": str(row["family"]),
        "kind": "atom",
        "level": int(row["level"]),
        "source_id": str(row["id"]),
        "prompt": str(row["prompt"]),
        "gold": row["gold"],
        "answer_domain": row.get("answer_domain"),
        "student_terminal_score": score,
        "source_outcome": "failure" if is_failure else "success",
        "student_full_token_ids": token_ids,
        "student_prefix_ids": prefix_ids,
        "student_suffix_ids": token_ids[prefix_length:],
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "exact_prompt_token_ids": [int(value) for value in prompt_token_ids] + prefix_ids,
        "prefix_length": prefix_length,
        "thought_length": thought_length,
        "student_output": {
            key: output.get(key)
            for key in (
                "thinking_closed", "forced_close", "finish_reason", "truncated",
                "n_thinking_tokens", "n_answer_tokens", "n_sampled_tokens",
                "injected_token_ids",
            )
        },
    }
    payload["state_id"] = _state_id("atom", payload)
    return payload


def episode_messages(
    system_prompt: str,
    initial_observation: str,
    past_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": str(initial_observation)},
    ]
    for turn in past_turns:
        messages.append({"role": "assistant", "content": str(turn["action"])})
        messages.append({"role": "user", "content": str(turn["observation"])})
    return messages


def replay_episode_state(state: Mapping[str, Any]):
    """Reconstruct a state and assert every visible transition is identical."""
    family = load_family(str(state["family"]))
    episode = family.Episode(int(state["ep_seed"]), int(state["level"]))
    system = episode.system_prompt()
    initial = episode.initial_observation()
    if system != state["system_prompt"]:
        raise ValueError(f"{state.get('state_id')}: system prompt replay mismatch")
    if initial != state["initial_observation"]:
        raise ValueError(f"{state.get('state_id')}: initial observation replay mismatch")
    done = False
    for expected_index, turn in enumerate(state["past_turns"]):
        if done:
            raise ValueError(f"{state.get('state_id')}: replay reached terminal too early")
        if int(turn["turn"]) != expected_index:
            raise ValueError(f"{state.get('state_id')}: noncontiguous replay turn index")
        observation, done = episode.step(str(turn["action"]))
        if observation != turn["observation"]:
            raise ValueError(
                f"{state.get('state_id')}: observation replay mismatch at {expected_index}"
            )
        if bool(getattr(episode, "last_action_ok", True)) != bool(turn["action_ok"]):
            raise ValueError(
                f"{state.get('state_id')}: action_ok replay mismatch at {expected_index}"
            )
    if done:
        raise ValueError(f"{state.get('state_id')}: selected state is already terminal")
    messages = episode_messages(system, initial, state["past_turns"])
    if messages != state["messages"]:
        raise ValueError(f"{state.get('state_id')}: visible message replay mismatch")
    return episode, messages


def build_episode_state(
    row: Mapping[str, Any],
    *,
    block: int,
    failure_ceiling: float,
    require_failure: bool = True,
) -> dict[str, Any] | None:
    """Select the state before the first invalid action, else the final action."""
    score = float(row["score"])
    if not math.isfinite(score):
        raise ValueError("episode score must be finite")
    is_failure = score < failure_ceiling
    if require_failure and not is_failure:
        return None
    if not require_failure and is_failure:
        return None
    turns = list(row.get("turns") or [])
    if not turns:
        return None
    selected_index = (
        next(
            (index for index, turn in enumerate(turns) if not bool(turn["action_ok"])),
            len(turns) - 1,
        )
        if is_failure
        else len(turns) - 1
    )
    selected = turns[selected_index]
    past_turns = [
        {
            "turn": int(turn["turn"]),
            "action": str(turn["action"]),
            "action_ok": bool(turn["action_ok"]),
            "observation": str(turn["observation"]),
        }
        for turn in turns[:selected_index]
    ]
    messages = episode_messages(
        str(row["system_prompt"]), str(row["initial_observation"]), past_turns
    )
    payload = {
        "block": int(block),
        "family": str(row["family"]),
        "kind": "episode",
        "level": int(row["level"]),
        "ep_seed": int(row["ep_seed"]),
        "source_id": str(row["rid"]),
        "student_terminal_score": score,
        "source_outcome": "failure" if is_failure else "success",
        "system_prompt": str(row["system_prompt"]),
        "initial_observation": str(row["initial_observation"]),
        "past_turns": past_turns,
        "messages": messages,
        "selected_turn_index": selected_index,
        "selected_student_turn": {
            key: selected.get(key)
            for key in (
                "turn", "action", "action_ok", "observation", "token_ids",
                "thinking_closed", "forced_close", "finish_reason", "truncated",
                "n_thinking_tokens", "n_answer_tokens", "n_sampled_tokens",
                "injected_token_ids",
            )
        },
        "selection_reason": (
            "first_invalid_action"
            if not bool(selected["action_ok"])
            else (
                "final_action_after_terminal_failure"
                if is_failure
                else "successful_final_action_anchor"
            )
        ),
    }
    payload["state_id"] = _state_id("episode", payload)
    replay_episode_state(payload)
    return payload


def select_balanced_states(
    candidates: Sequence[dict[str, Any]],
    *,
    atom_count: int,
    episode_count: int,
) -> list[dict[str, Any]]:
    """Deterministic cell-round-robin selection without teacher information."""
    selected: list[dict[str, Any]] = []
    for kind, required in (("atom", atom_count), ("episode", episode_count)):
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in candidates:
            if row["kind"] != kind:
                continue
            buckets.setdefault((str(row["family"]), int(row["level"])), []).append(row)
        for values in buckets.values():
            values.sort(key=lambda value: value["state_id"])
        keys = sorted(key for key, values in buckets.items() if values)
        cursors = {key: 0 for key in keys}
        chosen: list[dict[str, Any]] = []
        while len(chosen) < required:
            progressed = False
            for key in keys:
                cursor = cursors[key]
                if cursor >= len(buckets[key]):
                    continue
                chosen.append(buckets[key][cursor])
                cursors[key] += 1
                progressed = True
                if len(chosen) == required:
                    break
            if not progressed:
                raise ValueError(
                    f"only {len(chosen)} eligible balanced {kind} states for {required}"
                )
        selected.extend(chosen)
    identities = [row["state_id"] for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("balanced state selection produced duplicate IDs")
    return sorted(selected, key=lambda row: row["state_id"])
