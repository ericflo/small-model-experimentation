"""Strict action-answer parsing, exact scoring, and censoring diagnostics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any


ANSWER = re.compile(r"\s*ANSWER:\s*(\[.*\])\s*", re.DOTALL)
HF_MODEL_EOS_TOKEN_ID = 248044
THINK_CLOSE_TOKEN_ID = 248069


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _token_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a token-ID list")
    return [_exact_int(item, f"{label} item") for item in value]


def _trim_hf_eos(token_ids: list[int]) -> list[int]:
    return (
        token_ids[: token_ids.index(HF_MODEL_EOS_TOKEN_ID)]
        if HF_MODEL_EOS_TOKEN_ID in token_ids
        else list(token_ids)
    )


def _require_counter(output: dict[str, Any], key: str, expected: int) -> None:
    if _exact_int(output.get(key), key) != expected:
        raise ValueError(f"{key} differs from raw token arrays")


def reconstruct_output_token_counts(output: dict[str, Any]) -> dict[str, int]:
    """Recompute backend sampling and injection spend from raw token arrays."""
    stage1 = _token_ids(output.get("stage1_token_ids"), "stage1_token_ids")
    stage2 = _token_ids(output.get("stage2_token_ids"), "stage2_token_ids")
    injected = _token_ids(output.get("injected_token_ids"), "injected_token_ids")
    final = _token_ids(output.get("token_ids"), "token_ids")
    sampled = len(stage1) + len(stage2)
    _require_counter(output, "n_sampled_tokens", sampled)
    _require_counter(output, "n_injected_tokens", len(injected))
    _require_counter(output, "n_completion_tokens", len(final))
    return {
        "sampled_tokens": sampled,
        "injected_tokens": len(injected),
        "completion_tokens": len(final),
        "logical_tokens": sampled + len(injected),
    }


def validate_generation_counters(
    generated: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, int]:
    """Reconstruct every generation counter from raw, EOS-aware token arrays."""
    sampling = metadata.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("generation metadata lacks a sampling dictionary")
    thinking = sampling.get("thinking")
    if thinking not in {"off", "natural", "budget"}:
        raise ValueError("generation thinking mode is invalid")
    shuffle_thinking = sampling.get("shuffle_thinking")
    if type(shuffle_thinking) is not bool:
        raise ValueError("shuffle_thinking must be an exact boolean")
    termination = metadata.get("termination")
    if termination != {
        "hf_model_eos_token_id": HF_MODEL_EOS_TOKEN_ID,
        "vllm_tokenizer_eos_ignored": 248046,
    }:
        raise ValueError("generation termination IDs differ from the pinned runner")
    think_ids = metadata.get("think_token_ids")
    if (
        not isinstance(think_ids, dict)
        or set(think_ids)
        != {
            "open",
            "close",
            "forced_close_sequence",
            "thinking_prompt_suffix",
            "no_thinking_prompt_suffix",
        }
        or think_ids.get("open") != 248068
        or think_ids.get("close") != THINK_CLOSE_TOKEN_ID
    ):
        raise ValueError("generation think-token metadata differs from the pinned runner")
    close_ids = _token_ids(think_ids.get("forced_close_sequence"), "forced_close_sequence")
    if not close_ids or close_ids[0] != THINK_CLOSE_TOKEN_ID:
        raise ValueError("forced-close sequence does not begin with </think>")
    _token_ids(think_ids.get("thinking_prompt_suffix"), "thinking_prompt_suffix")
    _token_ids(think_ids.get("no_thinking_prompt_suffix"), "no_thinking_prompt_suffix")

    totals = {
        "requests": len(generated),
        "completions": 0,
        "unique_input_prompt_tokens": 0,
        "stage1_logical_prompt_tokens": 0,
        "stage2_logical_prompt_tokens": 0,
        "logical_model_input_tokens": 0,
        "sampled_tokens": 0,
        "injected_tokens": 0,
    }
    seen_ids: set[str] = set()
    for row in generated:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("generation row has an invalid ID")
        if row["id"] in seen_ids:
            raise ValueError("generation rows contain duplicate IDs")
        seen_ids.add(row["id"])
        prompt_ids = _token_ids(row.get("prompt_token_ids"), "prompt_token_ids")
        if not prompt_ids:
            raise ValueError("prompt_token_ids must not be empty")
        prompt_tokens = len(prompt_ids)
        _require_counter(row, "n_prompt_tokens", prompt_tokens)
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("generation row has no outputs")
        totals["unique_input_prompt_tokens"] += prompt_tokens
        totals["completions"] += len(outputs)
        seen_samples: set[int] = set()
        for output in outputs:
            if not isinstance(output, dict):
                raise ValueError("generation output is not a dictionary")
            sample_index = _exact_int(output.get("sample_index"), "sample_index")
            if sample_index in seen_samples:
                raise ValueError("generation row has duplicate sample indexes")
            seen_samples.add(sample_index)
            if type(output.get("truncated")) is not bool:
                raise ValueError("truncated must be an exact boolean")
            if type(output.get("thinking_closed")) is not bool:
                raise ValueError("thinking_closed must be an exact boolean")
            if type(output.get("forced_close")) is not bool:
                raise ValueError("forced_close must be an exact boolean")

            reconstructed = reconstruct_output_token_counts(output)
            stage1 = _token_ids(output["stage1_token_ids"], "stage1_token_ids")
            stage2 = _token_ids(output["stage2_token_ids"], "stage2_token_ids")
            injected = _token_ids(output["injected_token_ids"], "injected_token_ids")
            final = _token_ids(output["token_ids"], "token_ids")
            trimmed_stage1 = _trim_hf_eos(stage1)
            trimmed_stage2 = _trim_hf_eos(stage2)
            stage1_prompt = _exact_int(
                output.get("n_stage1_prompt_tokens"), "n_stage1_prompt_tokens", minimum=1
            )
            stage2_prompt = _exact_int(
                output.get("n_stage2_prompt_tokens"), "n_stage2_prompt_tokens"
            )
            if stage1_prompt != prompt_tokens:
                raise ValueError("stage-1 prompt count differs from raw row prompt count")

            continuation = bool(stage2 or injected)
            if continuation:
                if thinking != "budget":
                    raise ValueError("two-stage continuation occurred outside budget mode")
                retained = _token_ids(
                    output.get("retained_thinking_token_ids"),
                    "retained_thinking_token_ids",
                )
                close_index = (
                    trimmed_stage1.index(THINK_CLOSE_TOKEN_ID)
                    if THINK_CLOSE_TOKEN_ID in trimmed_stage1
                    else None
                )
                expected_retained = (
                    trimmed_stage1[:close_index]
                    if close_index is not None
                    else trimmed_stage1
                )
                if shuffle_thinking:
                    retained_matches = Counter(retained) == Counter(expected_retained)
                else:
                    retained_matches = retained == expected_retained
                if not retained_matches:
                    raise ValueError("retained thinking differs from raw stage-1 tokens")
                if injected != close_ids:
                    raise ValueError("injected tokens differ from the pinned forced-close sequence")
                if final != retained + injected + trimmed_stage2:
                    raise ValueError("final tokens differ from retained/injected/stage-2 arrays")
                if output["forced_close"] != (close_index is None):
                    raise ValueError("forced-close flag differs from raw stage-1 tokens")
                if output["thinking_closed"] is not True:
                    raise ValueError("two-stage budget output is not marked thinking-closed")
                expected_stage2_prompt = prompt_tokens + len(retained) + len(injected)
                if stage2_prompt != expected_stage2_prompt:
                    raise ValueError("stage-2 prompt count differs from reconstructed prompt")
                n_thinking = len(retained)
                n_answer = len(trimmed_stage2)
            else:
                if stage2 or injected:
                    raise AssertionError("unreachable continuation classification")
                if final != trimmed_stage1:
                    raise ValueError("ordinary final tokens differ from trimmed stage-1 tokens")
                if stage2_prompt != 0:
                    raise ValueError("ordinary output has a nonzero stage-2 prompt count")
                if output["forced_close"] is not False:
                    raise ValueError("ordinary output is incorrectly marked forced-close")
                close_index = (
                    final.index(THINK_CLOSE_TOKEN_ID)
                    if THINK_CLOSE_TOKEN_ID in final
                    else None
                )
                if thinking == "off":
                    n_thinking, n_answer = 0, len(final)
                elif close_index is None:
                    n_thinking, n_answer = len(final), 0
                else:
                    n_thinking = close_index
                    n_answer = len(final) - close_index - 1
                if output["thinking_closed"] != (close_index is not None):
                    raise ValueError("thinking-closed flag differs from raw tokens")

            _require_counter(output, "n_thinking_tokens", n_thinking)
            _require_counter(output, "n_answer_tokens", n_answer)
            _require_counter(
                output,
                "n_terminal_tokens_trimmed",
                len(stage1) - len(trimmed_stage1) + len(stage2) - len(trimmed_stage2),
            )
            totals["stage1_logical_prompt_tokens"] += stage1_prompt
            totals["stage2_logical_prompt_tokens"] += stage2_prompt
            totals["sampled_tokens"] += reconstructed["sampled_tokens"]
            totals["injected_tokens"] += reconstructed["injected_tokens"]

    totals["logical_model_input_tokens"] = (
        totals["stage1_logical_prompt_tokens"]
        + totals["stage2_logical_prompt_tokens"]
    )
    observed_counts = metadata.get("counts")
    if (
        not isinstance(observed_counts, dict)
        or set(observed_counts) != set(totals)
        or any(type(observed_counts[key]) is not int for key in totals)
        or any(observed_counts[key] < 0 for key in totals)
        or observed_counts["requests"] < 1
        or observed_counts["completions"] < 1
        or observed_counts["unique_input_prompt_tokens"] < 1
        or observed_counts["stage1_logical_prompt_tokens"] < 1
        or observed_counts["logical_model_input_tokens"] < 1
        or observed_counts["sampled_tokens"] < 1
        or observed_counts != totals
    ):
        raise ValueError("generation metadata counts differ from raw token reconstruction")
    timing = metadata.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "model_load_seconds",
        "generation_seconds",
        "sampled_tokens_per_second",
    }:
        raise ValueError("generation timing schema differs from the runner")
    for key in ("model_load_seconds", "generation_seconds", "sampled_tokens_per_second"):
        value = timing[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"generation {key} must be finite and positive")
    expected_rate = totals["sampled_tokens"] / timing["generation_seconds"]
    if timing["sampled_tokens_per_second"] != expected_rate:
        raise ValueError("sampled-token rate differs from raw counts and elapsed time")
    return totals


def parse_answer(text: str) -> list[Any] | None:
    visible = text.rsplit("</think>", 1)[-1]
    match = ANSWER.fullmatch(visible)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or len(value) != 3:
        return None
    return value


def exact_periodic_tail(
    token_ids: list[int], min_repeats: int, max_period: int, min_total_repeated: int
) -> bool:
    """Detect an exact periodic suffix without using decoded lexical content."""
    if len(token_ids) < min_total_repeated:
        return False
    for period in range(1, min(max_period, len(token_ids) // min_repeats) + 1):
        pattern = token_ids[-period:]
        repeats = 1
        cursor = len(token_ids) - 2 * period
        while cursor >= 0 and token_ids[cursor:cursor + period] == pattern:
            repeats += 1
            cursor -= period
        if repeats >= min_repeats and repeats * period >= min_total_repeated:
            return True
    return False


def score_generation_rows(
    generated: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    arm: str,
    candidate_counts: tuple[int, ...],
    answer_max_tokens: int,
    loop_detector: dict[str, int],
) -> list[dict[str, Any]]:
    label_by_id = {row["id"]: row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("duplicate label IDs")
    generated_ids = [row["id"] for row in generated]
    if len(set(generated_ids)) != len(generated_ids):
        raise ValueError("duplicate generated IDs")
    if set(generated_ids) != set(label_by_id):
        raise ValueError("generated/label task IDs differ")
    maximum = max(candidate_counts)
    scored = []
    for row in generated:
        label = label_by_id[row["id"]]
        outputs = row["outputs"]
        if len(outputs) < maximum:
            raise ValueError(f"{row['id']} has fewer than {maximum} candidates")
        candidates = []
        for output in outputs[:maximum]:
            parsed = parse_answer(str(output["text"]))
            reconstructed = reconstruct_output_token_counts(output)
            candidates.append(
                {
                    "parsed": parsed is not None,
                    "correct": parsed == label["answers"],
                    "answer_limit": bool(output.get("truncated"))
                    or int(output["n_answer_tokens"]) >= answer_max_tokens,
                    "periodic_loop": exact_periodic_tail(
                        list(output.get("retained_thinking_token_ids", [])),
                        min_repeats=int(loop_detector["min_repeats"]),
                        max_period=int(loop_detector["max_period_tokens"]),
                        min_total_repeated=int(loop_detector["min_total_repeated_tokens"]),
                    ),
                    "logical_tokens": reconstructed["logical_tokens"],
                }
            )
        scored_row = {
            "schema_version": 1,
            "task_id": row["id"],
            "family": label["family"],
            "depth": int(label["depth"]),
            "split": label["split"],
            "arm": arm,
        }
        for count in candidate_counts:
            prefix = candidates[:count]
            scored_row[f"coverage_at_{count}"] = float(any(item["correct"] for item in prefix))
            scored_row[f"candidate_accuracy_at_{count}"] = sum(
                item["correct"] for item in prefix
            ) / count
        primary = candidates[:maximum]
        scored_row.update(
            {
                "strict_parse_rate": sum(item["parsed"] for item in primary) / maximum,
                "answer_limit_contact": sum(item["answer_limit"] for item in primary) / maximum,
                "periodic_loop_contact": sum(item["periodic_loop"] for item in primary) / maximum,
                "logical_tokens": sum(item["logical_tokens"] for item in primary),
            }
        )
        scored.append(scored_row)
    return scored


def score_literal_reflection_diagnostic(
    reflection_generated: list[dict[str, Any]],
    action_generated: list[dict[str, Any]],
    base_generated: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    literal_candidate_count: int,
) -> list[dict[str, Any]]:
    """Compare literal plan-then-action against a token-matched base prefix."""
    label_by_id = {row["id"]: row for row in labels}
    reflection_by_id = {row["id"]: row for row in reflection_generated}
    base_by_id = {row["id"]: row for row in base_generated}
    action_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in action_generated:
        parent = str(row["meta"]["parent_task_id"])
        action_by_parent.setdefault(parent, []).append(row)
    if not (
        set(label_by_id) == set(reflection_by_id) == set(base_by_id) == set(action_by_parent)
    ):
        raise ValueError("literal/base/label task IDs differ")

    def logical(output: dict[str, Any]) -> int:
        return reconstruct_output_token_counts(output)["logical_tokens"]

    scored = []
    for task_id in sorted(label_by_id):
        reflection_outputs = reflection_by_id[task_id]["outputs"]
        action_rows = sorted(
            action_by_parent[task_id], key=lambda row: int(row["meta"]["sample_index"])
        )
        if len(reflection_outputs) != literal_candidate_count or len(action_rows) != literal_candidate_count:
            raise ValueError("literal candidate count differs from preregistration")
        if any(len(row["outputs"]) != 1 for row in action_rows):
            raise ValueError("literal action continuation must use n=1 per plan")
        literal_outputs = [row["outputs"][0] for row in action_rows]
        literal_tokens = sum(logical(output) for output in reflection_outputs) + sum(
            logical(output) for output in literal_outputs
        )
        literal_coverage = any(
            parse_answer(str(output["text"])) == label_by_id[task_id]["answers"]
            for output in literal_outputs
        )
        base_outputs = base_by_id[task_id]["outputs"]
        cumulative = 0
        matched_prefix = []
        for output in base_outputs:
            matched_prefix.append(output)
            cumulative += logical(output)
            if cumulative >= literal_tokens:
                break
        if cumulative < literal_tokens:
            raise ValueError("base reserve does not reach literal logical-token spend")
        base_coverage = any(
            parse_answer(str(output["text"])) == label_by_id[task_id]["answers"]
            for output in matched_prefix
        )
        scored.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "family": label_by_id[task_id]["family"],
                "split": label_by_id[task_id]["split"],
                "literal_candidates": literal_candidate_count,
                "literal_logical_tokens": literal_tokens,
                "literal_coverage": float(literal_coverage),
                "matched_base_candidates": len(matched_prefix),
                "matched_base_logical_tokens": cumulative,
                "matched_base_coverage": float(base_coverage),
                "literal_minus_matched_base": float(literal_coverage) - float(base_coverage),
            }
        )
    return scored
