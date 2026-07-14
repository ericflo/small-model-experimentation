"""Winner-bound tokenizer-EOS generation and authentication for mechanics."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

from vllm_runner import (
    MODEL_ID,
    MODEL_REVISION,
    RUNNER_SCHEMA_VERSION,
    TOKENIZER_EOS_TOKEN_ID,
    SamplingConfig,
    _jsonable_logprobs,
    _stable_seed,
)


SELECTED_INTERFACE = "tokenizer_eos_no_think_program_slot"
GENERATION_MODE = "winner_bound_tokenizer_eos_no_think"
ANSWER_CAP = 24


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token_id).to_bytes(4, "big") for token_id in token_ids)
    ).hexdigest()


def _ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise RuntimeError(f"invalid token IDs in {label}")
    return list(value)


def _validate_sampling(sampling: SamplingConfig) -> None:
    sampling.validate()
    if (
        sampling.thinking != "off"
        or sampling.n != 1
        or sampling.max_tokens != ANSWER_CAP
        or sampling.answer_max_tokens != ANSWER_CAP
        or sampling.answer_prefix != "PROGRAM:"
        or sampling.force_answer_seam
        or not sampling.paired_answer_seed
    ):
        raise RuntimeError("mechanics sampling escaped the frozen winner")


def _terminal_event(raw_ids: Sequence[int], finish: Any, stop_reason: Any) -> str:
    if finish == "stop":
        if (
            stop_reason != TOKENIZER_EOS_TOKEN_ID
            or not raw_ids
            or len(raw_ids) > ANSWER_CAP
            or raw_ids[-1] != TOKENIZER_EOS_TOKEN_ID
            or TOKENIZER_EOS_TOKEN_ID in raw_ids[:-1]
        ):
            raise RuntimeError("tokenizer-EOS stop geometry changed")
        return "stop"
    if finish == "length":
        if (
            stop_reason is not None
            or len(raw_ids) != ANSWER_CAP
            or TOKENIZER_EOS_TOKEN_ID in raw_ids
        ):
            raise RuntimeError("tokenizer-EOS length geometry changed")
        return "length"
    raise RuntimeError("tokenizer-EOS finish reason changed")


def generate_selected_interface(
    runner: Any,
    records: Sequence[dict[str, Any]],
    sampling: SamplingConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one completion per row at the frozen tokenizer-EOS boundary.

    The calibrated runner already exposes an audited per-request stop-token
    primitive.  This mechanics-only wrapper fixes that primitive to tokenizer
    EOS without modifying the calibration-locked runner bytes.
    """

    _validate_sampling(sampling)
    prepared = runner.prepare(records, "off", sampling.allow_custom_prompts)
    if not prepared:
        raise ValueError("input is empty")
    prefix_ids = list(
        runner.tokenizer.encode(sampling.answer_prefix, add_special_tokens=False)
    )
    if not prefix_ids:
        raise RuntimeError("winner prefix tokenized to no tokens")
    runner._check_context(prepared, sampling, prefix_ids)
    effective_prompt_ids = [
        list(record.prompt_token_ids) + prefix_ids for record in prepared
    ]
    prompts = [
        {"prompt_token_ids": list(token_ids)} for token_ids in effective_prompt_ids
    ]
    seeds = [
        _stable_seed(sampling.run_seed, record.record_id, 0, "answer")
        for record in prepared
    ]
    params = [
        runner._params(
            sampling,
            max_tokens=ANSWER_CAP,
            seed=seed,
            n=1,
            stop_token_id=TOKENIZER_EOS_TOKEN_ID,
        )
        for seed in seeds
    ]
    started = time.perf_counter()
    generated = runner.llm.generate(
        prompts,
        params,
        use_tqdm=False,
        lora_request=runner.lora_request,
    )
    elapsed = time.perf_counter() - started
    if len(generated) != len(prepared):
        raise RuntimeError("winner-bound request geometry changed")

    rows: list[dict[str, Any]] = []
    for prepared_row, request_output, seed, effective in zip(
        prepared, generated, seeds, effective_prompt_ids, strict=True
    ):
        if (
            len(request_output.outputs) != 1
            or type(request_output.outputs[0].index) is not int
            or request_output.outputs[0].index != 0
        ):
            raise RuntimeError("winner-bound completion geometry changed")
        completion = request_output.outputs[0]
        raw_ids = _ids(list(completion.token_ids), label="generated answer")
        event = _terminal_event(
            raw_ids, completion.finish_reason, completion.stop_reason
        )
        final_ids = prefix_ids + raw_ids
        output = {
            "sample_index": 0,
            "stage1_parent_seed": seed,
            "seed_stage1": seed,
            "seed_stage2": None,
            "seed_domain_stage1": "answer",
            "seed_domain_stage2": None,
            "text": runner.tokenizer.decode(final_ids, skip_special_tokens=False),
            "token_ids": final_ids,
            "raw_answer_token_ids": raw_ids,
            "stage1_token_ids": raw_ids,
            "answer_prefix_token_ids": prefix_ids,
            "injected_token_ids": prefix_ids,
            "stage2_token_ids": [],
            "n_thinking_tokens": 0,
            "n_answer_tokens": len(raw_ids),
            "n_sampled_tokens": len(raw_ids),
            "n_injected_tokens": len(prefix_ids),
            "n_completion_tokens": len(final_ids),
            "n_terminal_tokens_trimmed": 0,
            "n_stage1_prompt_tokens": len(effective),
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": False,
            "forced_close": False,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "stage1_finish_reason": completion.finish_reason,
            "stage1_stop_reason": completion.stop_reason,
            "truncated": completion.finish_reason == "length",
            "answer_cap_contact": (
                len(raw_ids) >= ANSWER_CAP or completion.finish_reason == "length"
            ),
            "registered_stop_token_id": TOKENIZER_EOS_TOKEN_ID,
            "terminal_event": event,
            "stage1_cumulative_logprob": completion.cumulative_logprob,
            "stage2_cumulative_logprob": None,
            "sampled_cumulative_logprob": completion.cumulative_logprob,
            "stage1_logprobs": _jsonable_logprobs(completion.logprobs),
            "stage2_logprobs": None,
        }
        rows.append(
            {
                "id": prepared_row.record_id,
                "meta": prepared_row.meta,
                "prompt_sha256": hashlib.sha256(
                    prepared_row.prompt_text.encode("utf-8")
                ).hexdigest(),
                "effective_prompt_sha256": _token_ids_sha256(effective),
                "n_prompt_tokens": len(effective),
                "n_original_prompt_tokens": len(prepared_row.prompt_token_ids),
                "prompt_channel": prepared_row.prompt_channel,
                "answer_prefix_token_ids": prefix_ids,
                "prompt_logprobs": None,
                "outputs": [output],
            }
        )
    return rows, runner._generation_summary(
        rows,
        sampling,
        elapsed,
        generation_mode=GENERATION_MODE,
        extra={
            "selected_interface": SELECTED_INTERFACE,
            "registered_answer_stop_token_id": TOKENIZER_EOS_TOKEN_ID,
            "terminal_token_preserved_in_raw_and_text": True,
        },
    )


def authenticate_selected_interface_bundle(
    *,
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    sampling: SamplingConfig,
    tokenizer: Any,
    tokenizer_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate exact prompts, seeds, tokens, terminal events, and costs."""

    _validate_sampling(sampling)
    prefix_ids = list(
        tokenizer.encode(sampling.answer_prefix, add_special_tokens=False)
    )
    if (
        prefix_ids != tokenizer_receipt.get("program_slot_prefix_token_ids")
        or tokenizer_receipt.get("termination", {}).get("tokenizer_eos_token_id")
        != TOKENIZER_EOS_TOKEN_ID
    ):
        raise RuntimeError("winner-bound tokenizer receipt changed")
    rows = bundle.get("rows")
    metadata = bundle.get("runner_metadata")
    if (
        not isinstance(rows, list)
        or not isinstance(metadata, dict)
        or metadata.get("schema_version") != RUNNER_SCHEMA_VERSION
        or metadata.get("generation_mode") != GENERATION_MODE
        or metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("selected_interface") != SELECTED_INTERFACE
        or metadata.get("registered_answer_stop_token_id")
        != TOKENIZER_EOS_TOKEN_ID
        or metadata.get("terminal_token_preserved_in_raw_and_text") is not True
        or metadata.get("sampling") != dataclasses.asdict(sampling)
        or [row.get("id") for row in rows] != [record.get("id") for record in records]
        or [row.get("meta") for row in rows]
        != [record.get("meta") for record in records]
    ):
        raise RuntimeError("winner-bound batch identity changed")

    outputs: list[Mapping[str, Any]] = []
    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise RuntimeError("winner-bound chat template returned non-text")
        original_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
        effective = original_ids + prefix_ids
        expected_row = {
            "id": record["id"],
            "meta": record.get("meta"),
            "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "effective_prompt_sha256": _token_ids_sha256(effective),
            "n_prompt_tokens": len(effective),
            "n_original_prompt_tokens": len(original_ids),
            "prompt_channel": "off",
            "answer_prefix_token_ids": prefix_ids,
            "prompt_logprobs": None,
        }
        changed = {
            key for key, value in expected_row.items() if row.get(key) != value
        }
        if changed:
            raise RuntimeError(
                f"winner-bound prompt row {index} changed: {sorted(changed)}"
            )
        row_outputs = row.get("outputs")
        if not isinstance(row_outputs, list) or len(row_outputs) != 1:
            raise RuntimeError("winner-bound output geometry changed")
        output = row_outputs[0]
        if not isinstance(output, Mapping):
            raise RuntimeError("winner-bound output schema changed")
        if type(output.get("answer_cap_contact")) is not bool or any(
            type(output.get(field)) is not int or output[field] < 0
            for field in (
                "sample_index",
                "stage1_parent_seed",
                "seed_stage1",
                "n_thinking_tokens",
                "n_answer_tokens",
                "n_sampled_tokens",
                "n_injected_tokens",
                "n_completion_tokens",
                "n_terminal_tokens_trimmed",
                "n_stage1_prompt_tokens",
                "n_stage2_prompt_tokens",
                "registered_stop_token_id",
            )
        ):
            raise RuntimeError("winner-bound typed output fields changed")
        raw_ids = _ids(output.get("raw_answer_token_ids"), label="raw answer")
        if output.get("stage1_token_ids") != raw_ids:
            raise RuntimeError("winner-bound raw/stage-one IDs diverged")
        event = _terminal_event(
            raw_ids, output.get("finish_reason"), output.get("stop_reason")
        )
        final_ids = prefix_ids + raw_ids
        seed = _stable_seed(sampling.run_seed, record["id"], 0, "answer")
        expected_output = {
            "sample_index": 0,
            "stage1_parent_seed": seed,
            "seed_stage1": seed,
            "seed_stage2": None,
            "seed_domain_stage1": "answer",
            "seed_domain_stage2": None,
            "text": tokenizer.decode(final_ids, skip_special_tokens=False),
            "token_ids": final_ids,
            "answer_prefix_token_ids": prefix_ids,
            "injected_token_ids": prefix_ids,
            "stage2_token_ids": [],
            "n_thinking_tokens": 0,
            "n_answer_tokens": len(raw_ids),
            "n_sampled_tokens": len(raw_ids),
            "n_injected_tokens": len(prefix_ids),
            "n_completion_tokens": len(final_ids),
            "n_terminal_tokens_trimmed": 0,
            "n_stage1_prompt_tokens": len(effective),
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": False,
            "forced_close": False,
            "stage1_finish_reason": output.get("finish_reason"),
            "stage1_stop_reason": output.get("stop_reason"),
            "truncated": event == "length",
            "answer_cap_contact": len(raw_ids) >= ANSWER_CAP or event == "length",
            "registered_stop_token_id": TOKENIZER_EOS_TOKEN_ID,
            "terminal_event": event,
        }
        changed_output = {
            key
            for key, value in expected_output.items()
            if output.get(key) != value
        }
        if changed_output:
            raise RuntimeError(
                f"winner-bound output {index} changed: {sorted(changed_output)}"
            )
        outputs.append(output)

    sampled = sum(int(output["n_sampled_tokens"]) for output in outputs)
    prompt_tokens = sum(int(output["n_stage1_prompt_tokens"]) for output in outputs)
    injected = sum(int(output["n_injected_tokens"]) for output in outputs)
    expected_counts = {
        "requests": len(rows),
        "completions": len(rows),
        "unique_input_prompt_tokens": sum(int(row["n_prompt_tokens"]) for row in rows),
        "stage1_logical_prompt_tokens": prompt_tokens,
        "stage2_logical_prompt_tokens": 0,
        "logical_model_input_tokens": prompt_tokens,
        "logical_prompt_tokens": prompt_tokens,
        "physical_prompt_tokens": prompt_tokens,
        "reused_prompt_tokens": 0,
        "sampled_tokens": sampled,
        "physical_sampled_tokens": sampled,
        "reused_sampled_tokens": 0,
        "logical_model_tokens": prompt_tokens + sampled,
        "physical_model_tokens": prompt_tokens + sampled,
        "reused_model_tokens": 0,
        "injected_tokens": injected,
    }
    if metadata.get("counts") != expected_counts:
        raise RuntimeError("winner-bound model-token accounting changed")
    return {
        "rows": len(rows),
        "selected_interface": SELECTED_INTERFACE,
        "registered_answer_stop_token_id": TOKENIZER_EOS_TOKEN_ID,
        "prompt_token_text_seed_coherence": True,
        "terminal_geometry_authenticated": True,
        "model_token_accounting": expected_counts,
    }
