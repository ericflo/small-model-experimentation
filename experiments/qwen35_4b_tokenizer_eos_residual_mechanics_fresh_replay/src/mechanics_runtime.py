"""Winner-bound tokenizer-EOS generation and authentication for mechanics."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import vllm_runner as runner_module
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
ROW_KEYS = {
    "id",
    "meta",
    "prompt_sha256",
    "effective_prompt_sha256",
    "n_prompt_tokens",
    "n_original_prompt_tokens",
    "prompt_channel",
    "answer_prefix_token_ids",
    "prompt_logprobs",
    "outputs",
}
OUTPUT_KEYS = {
    "sample_index",
    "stage1_parent_seed",
    "seed_stage1",
    "seed_stage2",
    "seed_domain_stage1",
    "seed_domain_stage2",
    "text",
    "token_ids",
    "raw_answer_token_ids",
    "stage1_token_ids",
    "answer_prefix_token_ids",
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
    "answer_cap_contact",
    "registered_stop_token_id",
    "terminal_event",
    "stage1_cumulative_logprob",
    "stage2_cumulative_logprob",
    "sampled_cumulative_logprob",
    "stage1_logprobs",
    "stage2_logprobs",
}
METADATA_KEYS = {
    "schema_version",
    "generation_mode",
    "model",
    "model_revision",
    "runner_sha256",
    "engine",
    "engine_args",
    "resolved_cudagraph",
    "resolved_logprobs_mode",
    "sampling",
    "resolved_sampling",
    "adapter",
    "think_token_ids",
    "termination",
    "rng_isolation",
    "counts",
    "timing",
    "runtime",
    "selected_interface",
    "registered_answer_stop_token_id",
    "terminal_token_preserved_in_raw_and_text",
}


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        b"".join(int(token_id).to_bytes(4, "big") for token_id in token_ids)
    ).hexdigest()


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def _typed_equal(observed: Any, expected: Any) -> bool:
    observed = _json_value(observed)
    expected = _json_value(expected)
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _typed_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _typed_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
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
        or sampling.thinking_budget is not None
        or type(sampling.n) is not int
        or sampling.n != 1
        or type(sampling.max_tokens) is not int
        or sampling.max_tokens != ANSWER_CAP
        or type(sampling.answer_max_tokens) is not int
        or sampling.answer_max_tokens != ANSWER_CAP
        or sampling.answer_prefix != "PROGRAM:"
        or sampling.greedy is not False
        or sampling.shuffle_thinking is not False
        or sampling.allow_custom_prompts is not False
        or sampling.force_answer_seam is not False
        or sampling.paired_answer_seed is not True
        or type(sampling.run_seed) is not int
        or sampling.run_seed < 0
        or type(sampling.top_k) is not int
        or any(
            type(value) is not float
            for value in (
                sampling.temperature,
                sampling.top_p,
                sampling.min_p,
                sampling.presence_penalty,
                sampling.frequency_penalty,
                sampling.repetition_penalty,
            )
        )
        or sampling.logprobs is not None
        or sampling.prompt_logprobs is not None
        or tuple(sampling.logprob_token_ids) != ()
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
            # Per-token logprobs are not requested by the qualified interface,
            # so its cumulative value cannot be independently recomputed.
            # Preserve no unauthenticated likelihood diagnostic.
            "stage1_cumulative_logprob": None,
            "stage2_cumulative_logprob": None,
            "sampled_cumulative_logprob": None,
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


def _authenticate_engine_metadata(
    *,
    metadata: Mapping[str, Any],
    sampling: SamplingConfig,
    tokenizer_receipt: Mapping[str, Any],
    engine_receipt: Mapping[str, Any],
) -> None:
    think = tokenizer_receipt.get("think_token_ids")
    termination = tokenizer_receipt.get("termination")
    if (
        not isinstance(think, Mapping)
        or not isinstance(termination, Mapping)
        or not isinstance(think.get("open"), list)
        or len(think["open"]) != 1
        or not isinstance(think.get("close"), list)
        or len(think["close"]) != 1
    ):
        raise RuntimeError("winner-bound tokenizer engine receipt changed")
    expected_think = {
        "open": think["open"][0],
        "close": think["close"][0],
        "forced_close_sequence": think.get("forced_close_sequence"),
        "thinking_prompt_suffix": think.get("thinking_prompt_suffix"),
        "no_thinking_prompt_suffix": think.get("no_thinking_prompt_suffix"),
    }
    expected_termination = {
        "hf_model_eos_token_id": termination.get("hf_model_eos_token_id"),
        "vllm_tokenizer_eos_ignored": termination.get("tokenizer_eos_token_id"),
    }
    runner_sha = hashlib.sha256(Path(runner_module.__file__).resolve().read_bytes()).hexdigest()
    runtime = metadata.get("runtime")
    preflight_runtime = engine_receipt.get("runtime")
    if (
        set(metadata) != METADATA_KEYS
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != RUNNER_SCHEMA_VERSION
        or metadata.get("generation_mode") != GENERATION_MODE
        or metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256") != runner_sha
        or engine_receipt.get("runner_sha256") != runner_sha
        or not _typed_equal(metadata.get("engine"), engine_receipt.get("engine"))
        or _canonical_sha256(metadata.get("engine_args"))
        != engine_receipt.get("engine_args_sha256")
        or not _typed_equal(
            metadata.get("resolved_cudagraph"),
            engine_receipt.get("resolved_cudagraph"),
        )
        or metadata.get("resolved_logprobs_mode")
        != engine_receipt.get("resolved_logprobs_mode")
        or not _typed_equal(metadata.get("sampling"), dataclasses.asdict(sampling))
        or not _typed_equal(
            metadata.get("resolved_sampling"), sampling.resolved_sampling()
        )
        or not _typed_equal(metadata.get("adapter"), engine_receipt.get("adapter"))
        or not _typed_equal(
            metadata.get("rng_isolation"), engine_receipt.get("rng_isolation")
        )
        or not _typed_equal(metadata.get("think_token_ids"), expected_think)
        or not _typed_equal(metadata.get("termination"), expected_termination)
        or metadata.get("selected_interface") != SELECTED_INTERFACE
        or type(metadata.get("registered_answer_stop_token_id")) is not int
        or metadata["registered_answer_stop_token_id"] != TOKENIZER_EOS_TOKEN_ID
        or metadata.get("terminal_token_preserved_in_raw_and_text") is not True
        or not isinstance(runtime, Mapping)
        or not isinstance(preflight_runtime, Mapping)
        or set(runtime) != set(preflight_runtime)
        or runtime.get("git_dirty") is not True
        or preflight_runtime.get("git_dirty") is not False
        or any(
            not _typed_equal(runtime.get(field), value)
            for field, value in preflight_runtime.items()
            if field != "git_dirty"
        )
    ):
        raise RuntimeError("winner-bound engine metadata changed")


def authenticate_selected_interface_bundle(
    *,
    records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    sampling: SamplingConfig,
    tokenizer: Any,
    tokenizer_receipt: Mapping[str, Any],
    engine_receipt: Mapping[str, Any],
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
        or len(rows) != len(records)
        or any(not isinstance(row, Mapping) or set(row) != ROW_KEYS for row in rows)
        or [row.get("id") for row in rows] != [record.get("id") for record in records]
        or any(
            not _typed_equal(row.get("meta"), record.get("meta"))
            for row, record in zip(rows, records, strict=True)
        )
    ):
        raise RuntimeError("winner-bound batch identity changed")
    _authenticate_engine_metadata(
        metadata=metadata,
        sampling=sampling,
        tokenizer_receipt=tokenizer_receipt,
        engine_receipt=engine_receipt,
    )

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
            key
            for key, value in expected_row.items()
            if not _typed_equal(row.get(key), value)
        }
        if changed:
            raise RuntimeError(
                f"winner-bound prompt row {index} changed: {sorted(changed)}"
            )
        row_outputs = row.get("outputs")
        if not isinstance(row_outputs, list) or len(row_outputs) != 1:
            raise RuntimeError("winner-bound output geometry changed")
        output = row_outputs[0]
        if not isinstance(output, Mapping) or set(output) != OUTPUT_KEYS:
            raise RuntimeError("winner-bound output schema changed")
        if (
            any(
                type(output.get(field)) is not bool
                for field in (
                    "thinking_closed",
                    "forced_close",
                    "truncated",
                    "answer_cap_contact",
                )
            )
            or any(
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
            )
        ):
            raise RuntimeError("winner-bound typed output fields changed")
        raw_ids = _ids(output.get("raw_answer_token_ids"), label="raw answer")
        if (
            _ids(output.get("stage1_token_ids"), label="stage-one answer") != raw_ids
            or _ids(output.get("token_ids"), label="completion")
            != prefix_ids + raw_ids
            or _ids(output.get("answer_prefix_token_ids"), label="answer prefix")
            != prefix_ids
            or _ids(output.get("injected_token_ids"), label="injected prefix")
            != prefix_ids
            or _ids(output.get("stage2_token_ids"), label="stage-two answer") != []
        ):
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
            "stage1_logprobs": None,
            "stage2_logprobs": None,
            "stage1_cumulative_logprob": None,
            "stage2_cumulative_logprob": None,
            "sampled_cumulative_logprob": None,
        }
        changed_output = {
            key
            for key, value in expected_output.items()
            if not _typed_equal(output.get(key), value)
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
    if not _typed_equal(metadata.get("counts"), expected_counts):
        raise RuntimeError("winner-bound model-token accounting changed")
    timing = metadata.get("timing")
    if not isinstance(timing, Mapping) or set(timing) != {
        "model_load_seconds",
        "generation_seconds",
        "sampled_tokens_per_second",
    }:
        raise RuntimeError("winner-bound timing schema changed")
    model_load = timing["model_load_seconds"]
    generation = timing["generation_seconds"]
    throughput = timing["sampled_tokens_per_second"]
    if (
        type(model_load) is not float
        or not math.isfinite(model_load)
        or model_load < 0.0
        or type(generation) is not float
        or not math.isfinite(generation)
        or generation <= 0.0
        or type(throughput) is not float
        or not math.isfinite(throughput)
        or throughput != sampled / generation
    ):
        raise RuntimeError("winner-bound timing fields changed")
    return {
        "rows": len(rows),
        "selected_interface": SELECTED_INTERFACE,
        "registered_answer_stop_token_id": TOKENIZER_EOS_TOKEN_ID,
        "prompt_token_text_seed_coherence": True,
        "terminal_geometry_authenticated": True,
        "exact_typed_schema_authenticated": True,
        "engine_preflight_authenticated": True,
        "unrequested_logprob_diagnostics_omitted": True,
        "model_token_accounting": expected_counts,
    }
