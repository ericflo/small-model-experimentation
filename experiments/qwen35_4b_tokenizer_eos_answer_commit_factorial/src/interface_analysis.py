"""Token-native authentication, scoring, and selection for EOS boundary pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from protocol import (
    HF_MODEL_EOS_ID,
    TOKENIZER_EOS_ID,
    authenticate_boundary_pair,
    content_for_terminal_event,
    is_answer_cap_contact,
)


BOUNDARY_ORDER = ("tokenizer_eos", "hf_model_eos")
BOUNDARY_STOP_IDS = {
    "tokenizer_eos": TOKENIZER_EOS_ID,
    "hf_model_eos": HF_MODEL_EOS_ID,
}
TOKENIZER_PRIORITY = (
    "tokenizer_eos_no_think_program_slot",
    "tokenizer_eos_no_think_freeform",
    "tokenizer_eos_think512_program_slot",
    "tokenizer_eos_think512_freeform",
)
PAIR_CONDITIONS = (
    "no_think_freeform",
    "no_think_program_slot",
    "think512_freeform",
    "think512_program_slot",
)


class BoundaryAuthenticationError(RuntimeError):
    """A generated receipt is not the frozen paired-boundary transaction."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise BoundaryAuthenticationError(f"invalid {label}")
    return value


def _terminal_event(output: dict[str, Any], stop_token_id: int) -> str:
    finish = output.get("finish_reason")
    reason = output.get("stop_reason")
    if finish == "stop":
        if reason != stop_token_id:
            raise BoundaryAuthenticationError("wrong stop/finish reason")
        return "stop"
    if finish == "length":
        if reason is not None:
            raise BoundaryAuthenticationError("length event carried a stop reason")
        return "length"
    raise BoundaryAuthenticationError("unknown stop/finish reason")


def _grammar_set(
    grammar_receipt: dict[str, Any], prefix_condition: str, arity: int
) -> set[tuple[int, ...]]:
    try:
        entry = grammar_receipt["grammar_inventories"][prefix_condition][str(arity)]
        sequences = entry["token_id_sequences"]
    except (KeyError, TypeError) as error:
        raise BoundaryAuthenticationError("grammar receipt schema changed") from error
    expected_rows = 24**arity
    if (
        not isinstance(sequences, list)
        or len(sequences) != expected_rows
        or entry.get("rows") != expected_rows
        or entry.get("token_id_sequences_sha256") != canonical_sha256(sequences)
    ):
        raise BoundaryAuthenticationError("grammar inventory receipt changed")
    result: set[tuple[int, ...]] = set()
    for sequence in sequences:
        if not isinstance(sequence, list):
            raise BoundaryAuthenticationError("grammar token sequence is not a list")
        token_ids = tuple(_integer(value, "grammar token ID") for value in sequence)
        result.add(token_ids)
    if len(result) != expected_rows:
        raise BoundaryAuthenticationError("grammar inventory token IDs collide")
    return result


def _expected_ids(
    grammar_receipt: dict[str, Any], row_id: str, prefix_condition: str
) -> tuple[int, ...]:
    try:
        values = grammar_receipt["calibration_expected_token_ids"][row_id][
            prefix_condition
        ]
    except (KeyError, TypeError) as error:
        raise BoundaryAuthenticationError("known-answer token registry changed") from error
    if not isinstance(values, list):
        raise BoundaryAuthenticationError("known-answer token IDs are not a list")
    return tuple(_integer(value, "known-answer token ID") for value in values)


def _prefix_ids(
    grammar_receipt: dict[str, Any], prefix_condition: str
) -> tuple[int, ...]:
    if prefix_condition == "freeform":
        expected: Any = []
    elif prefix_condition == "program_slot":
        expected = grammar_receipt.get("program_slot_prefix_token_ids")
    else:
        raise BoundaryAuthenticationError("unknown prefix condition")
    if not isinstance(expected, list):
        raise BoundaryAuthenticationError("program-slot prefix receipt changed")
    return tuple(_integer(value, "answer-prefix token ID") for value in expected)


def _authenticate_metadata(
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    thinking_policy: str,
    cap: int,
) -> None:
    expected_mode = (
        "shared_thought_boundary_pairs"
        if thinking_policy == "think512"
        else "answer_boundary_pairs"
    )
    pairing = metadata.get("boundary_pairing")
    if (
        metadata.get("schema_version") != 6
        or metadata.get("generation_mode") != expected_mode
        or metadata.get("model") != "Qwen/Qwen3.5-4B"
        or metadata.get("model_revision")
        != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or not isinstance(pairing, dict)
        or pairing.get("boundary_order") != list(BOUNDARY_ORDER)
        or pairing.get("registered_stop_token_ids")
        != [TOKENIZER_EOS_ID, HF_MODEL_EOS_ID]
        or pairing.get("pairs") != len(rows)
        or pairing.get("requests") != len(rows) * 2
        or pairing.get("adjacent_in_single_generate_call") is not True
        or pairing.get("identical_prompt_ids_within_pair") is not True
        or pairing.get("identical_answer_seed_within_pair") is not True
        or pairing.get("raw_answer_tokens_preserved") is not True
        or pairing.get("answer_cap_tokens") != cap
    ):
        raise BoundaryAuthenticationError("boundary-pair runner metadata changed")
    if thinking_policy == "think512":
        if (
            not isinstance(metadata.get("thought_source_sha256"), str)
            or metadata.get("thought_source_generation_mode")
            != "shared_thought_prefixes"
        ):
            raise BoundaryAuthenticationError("shared-thought receipt binding changed")
    elif "thought_source_sha256" in metadata:
        raise BoundaryAuthenticationError("no-think receipt claimed a thought source")

    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        raise BoundaryAuthenticationError("runner cost summary is absent")
    unique_prompt = sum(_integer(row.get("n_prompt_tokens"), "prompt tokens") for row in rows)
    outputs = [output for row in rows for output in row.get("outputs", [])]
    if len(outputs) != len(rows) * 2:
        raise BoundaryAuthenticationError("boundary completion count changed")
    stage1_prompt = sum(
        _integer(output.get("n_stage1_prompt_tokens"), "stage-1 prompt tokens")
        for output in outputs
    )
    stage2_prompt = sum(
        _integer(output.get("n_stage2_prompt_tokens"), "stage-2 prompt tokens")
        for output in outputs
    )
    sampled = sum(
        _integer(output.get("n_sampled_tokens"), "sampled tokens")
        for output in outputs
    )
    injected = sum(
        _integer(output.get("n_injected_tokens"), "injected tokens")
        for output in outputs
    )
    physical_sampled = sum(
        len(output.get("raw_answer_token_ids", [])) for output in outputs
    )
    logical_prompt = stage1_prompt + stage2_prompt
    physical_prompt = stage2_prompt if thinking_policy == "think512" else logical_prompt
    expected_counts = {
        "requests": len(rows),
        "completions": len(rows) * 2,
        "unique_input_prompt_tokens": unique_prompt,
        "stage1_logical_prompt_tokens": stage1_prompt,
        "stage2_logical_prompt_tokens": stage2_prompt,
        "logical_model_input_tokens": logical_prompt,
        "logical_prompt_tokens": logical_prompt,
        "physical_prompt_tokens": physical_prompt,
        "reused_prompt_tokens": logical_prompt - physical_prompt,
        "sampled_tokens": sampled,
        "physical_sampled_tokens": physical_sampled,
        "reused_sampled_tokens": sampled - physical_sampled,
        "logical_model_tokens": logical_prompt + sampled,
        "physical_model_tokens": physical_prompt + physical_sampled,
        "reused_model_tokens": logical_prompt
        + sampled
        - physical_prompt
        - physical_sampled,
        "injected_tokens": injected,
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise BoundaryAuthenticationError("runner token-cost summary changed")


def _authenticate_output_shape(
    output: dict[str, Any],
    *,
    row: dict[str, Any],
    pair_index: int,
    pair_offset: int,
    boundary: str,
    thinking_policy: str,
    prefix_ids: tuple[int, ...],
    close_ids: tuple[int, ...],
    cap: int,
    tokenizer: Any,
) -> tuple[str, tuple[int, ...], bool]:
    stop_id = BOUNDARY_STOP_IDS[boundary]
    raw = output.get("raw_answer_token_ids")
    if not isinstance(raw, list):
        raise BoundaryAuthenticationError("raw answer token IDs are absent")
    raw_ids = tuple(_integer(value, "raw answer token ID") for value in raw)
    event = _terminal_event(output, stop_id)
    try:
        content = content_for_terminal_event(
            raw_ids,
            registered_stop_token_id=stop_id,
            event=event,
            cap=cap,
        )
    except ValueError as error:
        raise BoundaryAuthenticationError(str(error)) from error
    if (
        output.get("sample_index") != 0
        or output.get("pair_index") != pair_index
        or output.get("pair_offset") != pair_offset
        or output.get("batch_position") != pair_index * 2 + pair_offset
        or output.get("pair_adjacent") is not True
        or output.get("boundary") != boundary
        or output.get("registered_stop_token_id") != stop_id
        or output.get("answer_prefix_token_ids") != list(prefix_ids)
        or output.get("n_answer_tokens") != len(raw_ids)
        or output.get("answer_cap_contact")
        is not is_answer_cap_contact(raw_ids, finish_reason=event, cap=cap)
        or output.get("truncated") is not (event == "length")
    ):
        raise BoundaryAuthenticationError("boundary output geometry changed")
    retained = output.get("retained_thinking_token_ids")
    token_ids = output.get("token_ids")
    if not isinstance(retained, list) or not isinstance(token_ids, list):
        raise BoundaryAuthenticationError("completion token inventory changed")
    retained_ids = tuple(_integer(value, "retained thought token ID") for value in retained)
    full_ids = tuple(_integer(value, "completion token ID") for value in token_ids)
    if thinking_policy == "think512":
        expected_full = retained_ids + close_ids + prefix_ids + raw_ids
        if (
            output.get("stage2_token_ids") != list(raw_ids)
            or output.get("n_thinking_tokens") != len(retained_ids)
            or output.get("thinking_closed") is not True
            or output.get("forced_close") is not True
            or output.get("seed_domain_stage1") != "thought"
            or output.get("seed_domain_stage2") != "answer"
        ):
            raise BoundaryAuthenticationError("shared-thought answer geometry changed")
    else:
        expected_full = prefix_ids + raw_ids
        if (
            retained_ids
            or output.get("stage1_token_ids") != list(raw_ids)
            or output.get("stage2_token_ids") != []
            or output.get("n_thinking_tokens") != 0
            or output.get("thinking_closed") is not False
            or output.get("forced_close") is not False
            or output.get("seed_domain_stage1") != "answer"
            or output.get("seed_domain_stage2") is not None
        ):
            raise BoundaryAuthenticationError("no-think answer geometry changed")
    if full_ids != expected_full:
        raise BoundaryAuthenticationError("completion token composition changed")
    try:
        decoded = tokenizer.decode(list(full_ids), skip_special_tokens=False)
    except Exception as error:  # pragma: no cover - real tokenizer surface
        raise BoundaryAuthenticationError("tokenizer decode failed") from error
    if output.get("text") != decoded:
        raise BoundaryAuthenticationError("decoded completion text changed")
    return event, content, is_answer_cap_contact(raw_ids, finish_reason=event, cap=cap)


def authenticate_and_score_bundle(
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    prefix_condition: str,
    thinking_policy: str,
    grammar_receipt: dict[str, Any],
    tokenizer: Any,
    cap: int = 24,
) -> dict[str, Any]:
    """Authenticate all pairs before returning per-boundary token-native metrics."""
    if thinking_policy not in {"no_think", "think512"}:
        raise BoundaryAuthenticationError("unknown thinking policy")
    if len(rows) != 48:
        raise BoundaryAuthenticationError("calibration row denominator changed")
    _authenticate_metadata(rows, metadata, thinking_policy=thinking_policy, cap=cap)
    prefix_ids = _prefix_ids(grammar_receipt, prefix_condition)
    close = metadata.get("think_token_ids", {}).get("forced_close_sequence")
    if not isinstance(close, list):
        raise BoundaryAuthenticationError("forced-close token receipt changed")
    close_ids = tuple(_integer(value, "forced-close token ID") for value in close)
    grammar = {arity: _grammar_set(grammar_receipt, prefix_condition, arity) for arity in (2, 3)}
    seen: set[str] = set()
    scored = {boundary: [] for boundary in BOUNDARY_ORDER}
    for pair_index, row in enumerate(rows):
        row_id = row.get("id")
        meta = row.get("meta")
        outputs = row.get("outputs")
        effective_ids = row.get("effective_prompt_token_ids")
        if (
            not isinstance(row_id, str)
            or row_id in seen
            or not isinstance(meta, dict)
            or not isinstance(outputs, list)
            or len(outputs) != 2
            or not isinstance(effective_ids, list)
            or row.get("answer_prefix_token_ids") != list(prefix_ids)
        ):
            raise BoundaryAuthenticationError("paired row identity/prompt changed")
        seen.add(row_id)
        arity = _integer(meta.get("arity"), "arity", minimum=2)
        if arity not in {2, 3}:
            raise BoundaryAuthenticationError("row arity changed")
        expected = _expected_ids(grammar_receipt, row_id, prefix_condition)
        terminal: dict[str, tuple[str, tuple[int, ...], bool]] = {}
        for pair_offset, boundary in enumerate(BOUNDARY_ORDER):
            terminal[boundary] = _authenticate_output_shape(
                outputs[pair_offset],
                row=row,
                pair_index=pair_index,
                pair_offset=pair_offset,
                boundary=boundary,
                thinking_policy=thinking_policy,
                prefix_ids=prefix_ids,
                close_ids=close_ids,
                cap=cap,
                tokenizer=tokenizer,
            )
        tokenizer_output, hf_output = outputs
        common_fields = (
            "stage1_parent_seed",
            "answer_seed",
            "retained_thinking_token_ids",
            "answer_prefix_token_ids",
            "injected_token_ids",
            "n_thinking_tokens",
            "n_injected_tokens",
            "n_stage1_prompt_tokens",
            "n_stage2_prompt_tokens",
            "thinking_closed",
            "forced_close",
            "seed_domain_stage1",
            "seed_domain_stage2",
        )
        if any(tokenizer_output.get(field) != hf_output.get(field) for field in common_fields):
            raise BoundaryAuthenticationError("paired prompt/seed/thought fields diverged")
        if tokenizer_output.get("answer_seed") is None:
            raise BoundaryAuthenticationError("paired answer seed is absent")
        if thinking_policy == "think512":
            thought_fields = (
                "seed_stage1",
                "seed_stage2",
                "stage1_token_ids",
                "stage1_finish_reason",
                "stage1_stop_reason",
                "stage1_cumulative_logprob",
                "stage1_logprobs",
                "n_tokens_discarded_after_close",
            )
            if any(
                tokenizer_output.get(field) != hf_output.get(field)
                for field in thought_fields
            ):
                raise BoundaryAuthenticationError("shared-thought source diverged")
        else:
            if (
                tokenizer_output.get("seed_stage1")
                != hf_output.get("seed_stage1")
                or tokenizer_output.get("seed_stage2") is not None
                or hf_output.get("seed_stage2") is not None
            ):
                raise BoundaryAuthenticationError("no-think answer seeds diverged")
        try:
            prompt_entry = grammar_receipt["calibration_prompt_token_ids"][row_id][
                thinking_policy
            ]
            base_prompt_ids = prompt_entry["token_ids"]
            prompt_text_sha256 = prompt_entry["prompt_text_sha256"]
        except (KeyError, TypeError) as error:
            raise BoundaryAuthenticationError(
                "calibration prompt-token registry changed"
            ) from error
        if not isinstance(base_prompt_ids, list):
            raise BoundaryAuthenticationError("registered base prompt is not token IDs")
        registered_base = tuple(
            _integer(value, "registered prompt token ID") for value in base_prompt_ids
        )
        retained_for_prompt = tuple(tokenizer_output["retained_thinking_token_ids"])
        expected_effective = (
            registered_base
            + (retained_for_prompt + close_ids if thinking_policy == "think512" else ())
            + prefix_ids
        )
        effective = tuple(
            _integer(value, "effective prompt token ID") for value in effective_ids
        )
        effective_hash = hashlib.sha256(
            b"".join(value.to_bytes(4, "big") for value in effective)
        ).hexdigest()
        if (
            effective != expected_effective
            or row.get("prompt_sha256") != prompt_text_sha256
            or row.get("n_prompt_tokens") != len(registered_base)
            or row.get("n_original_prompt_tokens") != len(registered_base)
            or row.get("n_effective_prompt_tokens") != len(effective)
            or row.get("effective_prompt_sha256") != effective_hash
            or row.get("prompt_channel")
            != ("thinking" if thinking_policy == "think512" else "off")
        ):
            raise BoundaryAuthenticationError("paired registered prompt changed")
        pair = authenticate_boundary_pair(
            tokenizer_output["raw_answer_token_ids"],
            hf_output["raw_answer_token_ids"],
            tokenizer_event=terminal["tokenizer_eos"][0],
            hf_event=terminal["hf_model_eos"][0],
            cap=cap,
        )
        if not pair.valid_pair:
            raise BoundaryAuthenticationError(
                f"boundary pair {row_id} failed: {pair.failure}"
            )
        for boundary in BOUNDARY_ORDER:
            event, sampled_content, cap_contact = terminal[boundary]
            composed = prefix_ids + sampled_content
            scored[boundary].append(
                {
                    "id": row_id,
                    "task_id": meta.get("task_id"),
                    "arity": arity,
                    "event": event,
                    "precommit_token_ids": list(sampled_content),
                    "composed_token_ids": list(composed),
                    "parsed": composed in grammar[arity],
                    "exact_echo": composed == expected,
                    "answer_cap_contact": cap_contact,
                    "compared_pair_tokens": pair.compared_tokens,
                }
            )
    result: dict[str, Any] = {
        "pair_authentication": "PASS",
        "pairs": len(rows),
        "condition": f"{thinking_policy}_{prefix_condition}",
        "cells": {},
    }
    for boundary in BOUNDARY_ORDER:
        values = scored[boundary]
        by_arity = {}
        for arity in (2, 3):
            subset = [row for row in values if row["arity"] == arity]
            by_arity[str(arity)] = {
                "rows": len(subset),
                "exact_echo_successes": sum(row["exact_echo"] for row in subset),
                "parse_successes": sum(row["parsed"] for row in subset),
                "answer_cap_contacts": sum(row["answer_cap_contact"] for row in subset),
            }
        result["cells"][boundary] = {
            "rows": len(values),
            "exact_echo_successes": sum(row["exact_echo"] for row in values),
            "parse_successes": sum(row["parsed"] for row in values),
            "answer_cap_contacts": sum(row["answer_cap_contact"] for row in values),
            "arity_counts": dict(sorted(Counter(row["arity"] for row in values).items())),
            "by_arity": by_arity,
            "scored": values,
        }
    return result


def calibration_qualifies(metrics: dict[str, Any], gate: dict[str, int]) -> bool:
    if metrics.get("rows") != gate["rows"]:
        raise ValueError("calibration row denominator changed")
    return bool(
        metrics["exact_echo_successes"] >= gate["exact_echo_successes_min"]
        and metrics["parse_successes"] >= gate["parse_successes_min"]
        and metrics["answer_cap_contacts"] <= gate["answer_cap_contacts_max"]
        and all(
            metrics["by_arity"][str(arity)]["rows"] == gate["each_arity_rows"]
            and metrics["by_arity"][str(arity)]["exact_echo_successes"]
            >= gate["each_arity_exact_successes_min"]
            and metrics["by_arity"][str(arity)]["parse_successes"]
            >= gate["each_arity_parse_successes_min"]
            and metrics["by_arity"][str(arity)]["answer_cap_contacts"]
            <= gate["each_arity_answer_cap_contacts_max"]
            for arity in (2, 3)
        )
    )


def choose_interface(
    metrics_by_cell: dict[str, dict[str, Any]], gate: dict[str, int]
) -> dict[str, Any]:
    expected_cells = {
        f"{boundary}_{condition}"
        for boundary in BOUNDARY_ORDER
        for condition in PAIR_CONDITIONS
    }
    if set(metrics_by_cell) != expected_cells:
        raise ValueError("interface cell inventory changed")
    qualification = {
        cell: calibration_qualifies(metrics_by_cell[cell], gate)
        for cell in sorted(expected_cells)
    }
    for condition in PAIR_CONDITIONS:
        if (
            qualification[f"tokenizer_eos_{condition}"]
            and qualification[f"hf_model_eos_{condition}"]
        ):
            return {
                "decision": "SCORING_INVARIANT_VIOLATION",
                "winner": None,
                "matched_hf_control": None,
                "qualification": qualification,
                "fixed_tokenizer_priority": list(TOKENIZER_PRIORITY),
            }
    winner = next((cell for cell in TOKENIZER_PRIORITY if qualification[cell]), None)
    if winner is not None:
        condition = winner.removeprefix("tokenizer_eos_")
        return {
            "decision": "TOKENIZER_EOS_ONLY_INTERFACE_QUALIFIED",
            "winner": winner,
            "matched_hf_control": f"hf_model_eos_{condition}",
            "qualification": qualification,
            "fixed_tokenizer_priority": list(TOKENIZER_PRIORITY),
        }
    hf_qualifies = any(
        qualification[f"hf_model_eos_{condition}"] for condition in PAIR_CONDITIONS
    )
    return {
        "decision": (
            "HF_ONLY_CONTROL_QUALIFIES_TOKENIZER_FAIL"
            if hf_qualifies
            else "NO_VALID_TOKENIZER_EOS_ANSWER_SEAM"
        ),
        "winner": None,
        "matched_hf_control": None,
        "qualification": qualification,
        "fixed_tokenizer_priority": list(TOKENIZER_PRIORITY),
    }
