"""Exact token/mask construction shared by dense and off-policy controls."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def fit_prompt_around_completion(
    prompt: list[int],
    completion: list[int],
    *,
    max_length: int,
    state_id: str,
) -> tuple[list[int], int]:
    """Preserve the exact completion and trim only the oldest prompt tokens.

    Sparse MOPD and the off-policy control supervise positions in the generated
    completion.  Right truncation would silently delete those registered target
    positions, so the fixed sequence budget is realized by left-truncating the
    prompt.  At least one prompt token is retained for the first causal target.
    """

    limit = int(max_length)
    if limit < 2:
        raise ValueError(f"max_length must be at least 2, got {limit}")
    prompt_budget = limit - len(completion)
    if prompt_budget < 1:
        raise ValueError(
            f"completion leaves no prompt budget for {state_id}: "
            f"{len(completion)} completion tokens at max_length {limit}"
        )
    truncated = max(0, len(prompt) - prompt_budget)
    return prompt[truncated:], truncated


def prompt_and_student_completion(unit: Mapping[str, Any], tokenizer) -> tuple[list[int], list[int], list[int]]:
    state = unit["state"]
    if state["kind"] == "atom":
        prompt = [int(value) for value in state["exact_prompt_token_ids"]]
        completion = [int(value) for value in state["student_suffix_ids"]]
        output = state["student_output"]
        injected = [int(value) for value in output.get("injected_token_ids") or []]
        injection_start = int(output.get("n_thinking_tokens") or 0) - int(
            state["prefix_length"]
        )
    elif state["kind"] == "episode":
        rendered = tokenizer.apply_chat_template(
            state["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = [
            int(value)
            for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ]
        output = state["selected_student_turn"]
        completion = [int(value) for value in output["token_ids"]]
        injected = [int(value) for value in output.get("injected_token_ids") or []]
        injection_start = int(output.get("n_thinking_tokens") or 0)
    else:
        raise ValueError(f"unknown state kind: {state['kind']!r}")
    if not prompt or not completion:
        raise ValueError(f"empty prompt/completion for {unit['state_id']}")
    active = list(range(len(completion)))
    if injected:
        if injection_start < 0 or injection_start + len(injected) > len(completion):
            raise ValueError(f"injected-token span out of range for {unit['state_id']}")
        if completion[injection_start:injection_start + len(injected)] != injected:
            raise ValueError(f"injected-token identity mismatch for {unit['state_id']}")
        blocked = set(range(injection_start, injection_start + len(injected)))
        active = [position for position in active if position not in blocked]
    if not active:
        raise ValueError(f"no natural policy positions for {unit['state_id']}")
    return prompt, completion, active


def make_sparse_sample(
    unit: Mapping[str, Any],
    tokenizer,
    *,
    max_positions: int,
    max_length: int,
) -> dict[str, Any]:
    prompt, completion, active = prompt_and_student_completion(unit, tokenizer)
    positions = active[-int(max_positions):]
    original_prompt_tokens = len(prompt)
    prompt, prompt_tokens_truncated = fit_prompt_around_completion(
        prompt,
        completion,
        max_length=max_length,
        state_id=str(unit["state_id"]),
    )
    return {
        "id": str(unit["state_id"]),
        "meta": {
            "family": str(unit["family"]),
            "kind": str(unit["kind"]),
            "level": int(unit["level"]),
            "role": str(unit["role"]),
            "primary_teacher": str(unit["primary_teacher"]),
            "observed_route": unit.get("observed_route"),
            "matched_primary_state_id": unit.get("matched_primary_state_id"),
            "match_tier": unit.get("match_tier"),
            "original_prompt_tokens": original_prompt_tokens,
            "prompt_tokens_truncated": prompt_tokens_truncated,
            "input_tokens": len(prompt) + len(completion),
        },
        "prompt_ids": torch.tensor(prompt, dtype=torch.int32),
        "completion_ids": torch.tensor(completion, dtype=torch.int32),
        "positions": torch.tensor(positions, dtype=torch.int32),
        "targets": {},
    }


def offpolicy_prompt_and_completion(unit: Mapping[str, Any], tokenizer) -> tuple[list[int], list[int], list[int]]:
    if unit.get("role") != "capability" or not unit.get("offpolicy_target"):
        raise ValueError("off-policy targets exist only for capability units")
    state = unit["state"]
    if state["kind"] == "atom":
        prompt = [int(value) for value in state["exact_prompt_token_ids"]]
    else:
        rendered = tokenizer.apply_chat_template(
            state["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt = [
            int(value)
            for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ]
    target = unit["offpolicy_target"]
    completion = [int(value) for value in target["completion_ids"]]
    injected = [int(value) for value in target.get("injected_token_ids") or []]
    active = list(range(len(completion)))
    if injected:
        # The runner injects the close sequence after retained thought. Find
        # the unique exact subsequence instead of trusting decoded text.
        starts = [
            index
            for index in range(len(completion) - len(injected) + 1)
            if completion[index:index + len(injected)] == injected
        ]
        if len(starts) != 1:
            raise ValueError(f"cannot locate unique off-policy injection for {unit['state_id']}")
        blocked = set(range(starts[0], starts[0] + len(injected)))
        active = [position for position in active if position not in blocked]
    if not prompt or not completion or not active:
        raise ValueError(f"invalid off-policy unit {unit['state_id']}")
    return prompt, completion, active
