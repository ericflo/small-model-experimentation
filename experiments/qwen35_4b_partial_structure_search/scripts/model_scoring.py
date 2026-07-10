#!/usr/bin/env python3
"""vLLM-only model scores for partial-structure calibration and search.

The module deliberately keeps task construction out of the model layer.  A
caller supplies a visible task description, visible examples, a candidate
operation prefix, and the number of open slots.  Hidden examples must never be
passed here.

All probability readouts are *targeted next-token* log probabilities from the
experiment-local :mod:`src.vllm_runner`.  In particular, the thinking judge is
two model requests, not a text parser:

1. sample a bounded thinking continuation;
2. retain exactly the tokens before ``</think>`` (or the whole non-EOS sample
   when the budget forces a close), append the exact token prefix
   ``</think>\n\nAnswer: ``, and request one token while asking vLLM for the
   selected A/B log probabilities.

The returned score is the label probability renormalized over the audited
label set.  It is not the label's full-vocabulary probability.  Every row and
batch records logical prefill tokens, sampled tokens, and model requests so a
search policy cannot hide repeated judge prefills.
"""

from __future__ import annotations

import dataclasses
import json
import math
import string
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = EXPERIMENT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vllm_runner import SamplingConfig, _stable_seed  # noqa: E402


BINARY_LABELS = ("A", "B")
NEXT_OPERATION_LABELS = tuple(string.ascii_uppercase[:16])
ANSWER_PREFIX = "Answer: "
THINKING_ANSWER_SUFFIX = "</think>\n\nAnswer: "

VIABILITY_SYSTEM_PROMPT = (
    "You judge unfinished symbolic programs. Decide whether the shown operation "
    "prefix can still be completed, within the stated number of open slots, into "
    "a program consistent with every visible example. Use only the information "
    "shown in the prompt."
)

NEXT_OPERATION_SYSTEM_PROMPT = (
    "You rank the next operation of an unfinished symbolic program. Choose the "
    "operation most likely to occur next in at least one completion consistent "
    "with every visible example and the stated number of open slots. Use only "
    "the information shown in the prompt."
)


@dataclasses.dataclass(frozen=True)
class RequestAccounting:
    """Logical model work; injected tokens are already included in prefill."""

    requests: int = 0
    completions: int = 0
    prefill_tokens: int = 0
    sampled_tokens: int = 0
    injected_prompt_tokens: int = 0
    retained_thinking_tokens: int = 0
    discarded_sampled_tokens: int = 0

    @property
    def total_model_tokens(self) -> int:
        return self.prefill_tokens + self.sampled_tokens

    def __add__(self, other: "RequestAccounting") -> "RequestAccounting":
        if not isinstance(other, RequestAccounting):
            return NotImplemented
        return RequestAccounting(
            requests=self.requests + other.requests,
            completions=self.completions + other.completions,
            prefill_tokens=self.prefill_tokens + other.prefill_tokens,
            sampled_tokens=self.sampled_tokens + other.sampled_tokens,
            injected_prompt_tokens=(
                self.injected_prompt_tokens + other.injected_prompt_tokens
            ),
            retained_thinking_tokens=(
                self.retained_thinking_tokens + other.retained_thinking_tokens
            ),
            discarded_sampled_tokens=(
                self.discarded_sampled_tokens + other.discarded_sampled_tokens
            ),
        )

    def as_dict(self) -> dict[str, int]:
        value = dataclasses.asdict(self)
        value["total_model_tokens"] = self.total_model_tokens
        return value


def _sum_accounting(values: Sequence[RequestAccounting]) -> RequestAccounting:
    total = RequestAccounting()
    for value in values:
        total += value
    return total


def _normalize_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in normalized:
        raise ValueError(f"{name} must not contain NUL")
    return normalized


def _stable_visible_examples(visible_examples: Any) -> str:
    """Render only the examples explicitly supplied through the visible channel."""
    if visible_examples is None:
        return "(none supplied)"
    if isinstance(visible_examples, str):
        return _normalize_text(visible_examples, "visible_examples")
    try:
        rendered = json.dumps(
            visible_examples,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("visible_examples must be JSON-serializable") from exc
    return rendered


def _normalize_prefix(prefix: Sequence[str]) -> tuple[str, ...]:
    if isinstance(prefix, (str, bytes)) or not isinstance(prefix, Sequence):
        raise TypeError("prefix must be a sequence of operation strings")
    return tuple(
        _normalize_text(operation, f"prefix[{index}]")
        for index, operation in enumerate(prefix)
    )


def _normalize_remaining_slots(remaining_slots: Any) -> int:
    if isinstance(remaining_slots, bool) or not isinstance(remaining_slots, int):
        raise TypeError("remaining_slots must be an integer")
    if remaining_slots < 0:
        raise ValueError("remaining_slots must be non-negative")
    return remaining_slots


def _format_prefix(prefix: Sequence[str]) -> str:
    operations = _normalize_prefix(prefix)
    if not operations:
        return "(empty prefix)"
    return "\n".join(f"{index + 1}. {operation}" for index, operation in enumerate(operations))


def build_binary_viability_messages(
    task: str,
    prefix: Sequence[str],
    remaining_slots: int,
    *,
    visible_examples: Any = None,
) -> list[dict[str, str]]:
    """Build the frozen A=viable/B=not-viable semantic judge prompt."""
    task_text = _normalize_text(task, "task")
    slots = _normalize_remaining_slots(remaining_slots)
    user = (
        "VISIBLE TASK\n"
        f"{task_text}\n\n"
        "VISIBLE EXAMPLES\n"
        f"{_stable_visible_examples(visible_examples)}\n\n"
        "CANDIDATE OPERATION PREFIX (execution order)\n"
        f"{_format_prefix(prefix)}\n\n"
        "OPEN SLOTS\n"
        f"{slots}\n\n"
        "Decide whether at least one legal completion of exactly the open slots "
        "is consistent with every visible example.\n"
        "A = viable: at least one such completion exists.\n"
        "B = not viable: no such completion exists.\n"
        "Return one letter only."
    )
    return [
        {"role": "system", "content": VIABILITY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_next_operation_messages(
    task: str,
    prefix: Sequence[str],
    remaining_slots: int,
    choices: Sequence[str],
    *,
    visible_examples: Any = None,
) -> list[dict[str, str]]:
    """Build the frozen 16-way next-operation letter-ranking prompt."""
    task_text = _normalize_text(task, "task")
    slots = _normalize_remaining_slots(remaining_slots)
    if slots < 1:
        raise ValueError("next-operation ranking requires at least one open slot")
    if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
        raise TypeError("choices must be a sequence of exactly 16 operation strings")
    normalized_choices = tuple(
        _normalize_text(choice, f"choices[{index}]")
        for index, choice in enumerate(choices)
    )
    if len(normalized_choices) != len(NEXT_OPERATION_LABELS):
        raise ValueError("next-operation ranking requires exactly 16 choices")
    if len(set(normalized_choices)) != len(normalized_choices):
        raise ValueError("next-operation choices must be unique")
    menu = "\n".join(
        f"{label} = {choice}"
        for label, choice in zip(NEXT_OPERATION_LABELS, normalized_choices)
    )
    user = (
        "VISIBLE TASK\n"
        f"{task_text}\n\n"
        "VISIBLE EXAMPLES\n"
        f"{_stable_visible_examples(visible_examples)}\n\n"
        "CANDIDATE OPERATION PREFIX (execution order)\n"
        f"{_format_prefix(prefix)}\n\n"
        "OPEN SLOTS (including the next operation)\n"
        f"{slots}\n\n"
        "NEXT-OPERATION MENU\n"
        f"{menu}\n\n"
        "Choose the operation most likely to be next in at least one legal "
        "completion consistent with every visible example. Return one letter only."
    )
    return [
        {"role": "system", "content": NEXT_OPERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def single_token_label_ids(
    tokenizer: Any, labels: Sequence[str]
) -> dict[str, int]:
    """Resolve label IDs from the loaded tokenizer and fail closed on drift."""
    if isinstance(labels, (str, bytes)) or not labels:
        raise ValueError("labels must be a non-empty sequence")
    result: dict[str, int] = {}
    for index, label in enumerate(labels):
        label = _normalize_text(label, f"labels[{index}]")
        if label in result:
            raise ValueError(f"duplicate label: {label!r}")
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise RuntimeError(
                f"label {label!r} must be exactly one token; got {token_ids!r}"
            )
        result[label] = int(token_ids[0])
    if len(set(result.values())) != len(result):
        raise RuntimeError(f"distinct labels resolved to duplicate token IDs: {result!r}")
    return result


def _coerce_logprob(value: Any) -> float:
    if hasattr(value, "logprob"):
        value = value.logprob
    elif isinstance(value, Mapping) and "logprob" in value:
        value = value["logprob"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"invalid logprob value: {value!r}")
    result = float(value)
    if math.isnan(result) or result == math.inf:
        raise ValueError(f"invalid logprob value: {result!r}")
    return result


def _first_logprob_step(source: Any) -> Mapping[Any, Any]:
    """Accept a raw vLLM completion or the runner's JSON-normalized output."""
    if hasattr(source, "logprobs"):
        source = source.logprobs
    elif isinstance(source, Mapping):
        for key in ("stage1_logprobs", "logprobs"):
            if key in source:
                source = source[key]
                break
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if not source:
            raise ValueError("completion has no token logprob steps")
        source = source[0]
    if not isinstance(source, Mapping):
        raise TypeError("first token logprobs must be a mapping")
    return source


def extract_target_logprobs(
    source: Any, label_token_ids: Mapping[str, int]
) -> dict[str, float]:
    """Extract every requested label from the first generated-token logprobs."""
    step = _first_logprob_step(source)
    extracted: dict[str, float] = {}
    missing: list[tuple[str, int]] = []
    for label, token_id in label_token_ids.items():
        if token_id in step:
            value = step[token_id]
        elif str(token_id) in step:
            value = step[str(token_id)]
        else:
            missing.append((label, token_id))
            continue
        extracted[label] = _coerce_logprob(value)
    if missing:
        raise ValueError(f"targeted logprobs missing labels/token IDs: {missing!r}")
    return extracted


def normalize_label_logprobs(logprobs: Mapping[str, float]) -> dict[str, float]:
    """Stable softmax over an audited label set, retaining caller order."""
    if not logprobs:
        raise ValueError("cannot normalize an empty logprob mapping")
    values = {label: _coerce_logprob(value) for label, value in logprobs.items()}
    finite = [value for value in values.values() if value != -math.inf]
    if not finite:
        raise ValueError("all targeted labels have negative-infinite logprob")
    maximum = max(finite)
    weights = {
        label: (0.0 if value == -math.inf else math.exp(value - maximum))
        for label, value in values.items()
    }
    denominator = math.fsum(weights.values())
    if not denominator or not math.isfinite(denominator):
        raise ValueError("targeted label normalization failed")
    return {label: weight / denominator for label, weight in weights.items()}


def _record_value(record: Mapping[str, Any], names: Sequence[str], what: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"record is missing {what}; accepted fields: {', '.join(names)}")


def _record_id(record: Mapping[str, Any]) -> str:
    value = _record_value(record, ("id", "record_id"), "id")
    return _normalize_text(str(value), "record id")


def _record_task(record: Mapping[str, Any]) -> str:
    return _record_value(
        record,
        ("task", "task_text", "specification", "instruction"),
        "visible task text",
    )


def _record_prefix(record: Mapping[str, Any]) -> Sequence[str]:
    return _record_value(
        record,
        ("prefix", "candidate_prefix", "partial_structure", "partial_skeleton"),
        "candidate prefix",
    )


def _record_remaining_slots(record: Mapping[str, Any]) -> int:
    return _record_value(
        record,
        ("remaining_slots", "remaining_steps", "slots_remaining"),
        "remaining slots",
    )


def _record_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "task",
        "task_text",
        "specification",
        "instruction",
        "prefix",
        "candidate_prefix",
        "partial_structure",
        "partial_skeleton",
        "visible_examples",
        "choices",
        "operations",
        "next_operations",
    }
    return {key: value for key, value in record.items() if key not in excluded}


def _validate_unique_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    ids = [_record_id(record) for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record IDs must be unique within a scoring batch")
    return ids


def _raw_logprobs(step: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for token_id, value in _first_logprob_step(step).items():
        if hasattr(value, "logprob"):
            result[str(token_id)] = {
                "logprob": float(value.logprob),
                "rank": getattr(value, "rank", None),
                "decoded_token": getattr(value, "decoded_token", None),
            }
        elif isinstance(value, Mapping):
            result[str(token_id)] = dict(value)
        else:
            result[str(token_id)] = {"logprob": float(value)}
    return result


class ModelScorer:
    """Batched selected-token scorers around an initialized ``VLLMRunner``."""

    def __init__(self, runner: Any):
        self.runner = runner
        self.binary_label_ids = single_token_label_ids(
            runner.tokenizer, BINARY_LABELS
        )
        self.next_operation_label_ids = single_token_label_ids(
            runner.tokenizer, NEXT_OPERATION_LABELS
        )
        self.answer_prefix_ids = self._encode_exact(ANSWER_PREFIX)
        self.thinking_answer_suffix_ids = self._encode_exact(
            THINKING_ANSWER_SUFFIX
        )
        if (
            not self.thinking_answer_suffix_ids
            or self.thinking_answer_suffix_ids[0] != runner.think_close_id
        ):
            raise RuntimeError(
                "thinking score suffix no longer begins with the runner's </think> token"
            )

    def _encode_exact(self, text: str) -> list[int]:
        token_ids = self.runner.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"scoring prefix tokenized empty: {text!r}")
        return [int(token_id) for token_id in token_ids]

    def _render_chat_ids(
        self, messages: Sequence[Mapping[str, str]], *, thinking: bool
    ) -> tuple[str, list[int]]:
        rendered = self.runner._render_messages(  # noqa: SLF001 - local runner API
            list(messages), enable_thinking=thinking
        )
        token_ids = self._encode_exact(rendered)
        expected = "thinking" if thinking else "off"
        prompt_channel = self.runner._prompt_channel(token_ids)  # noqa: SLF001
        if prompt_channel != expected:
            raise RuntimeError(
                f"rendered judge prompt used {prompt_channel!r}, expected {expected!r}"
            )
        return rendered, token_ids

    def _check_context(self, prompt_ids: Sequence[Sequence[int]], reserve: int) -> None:
        maximum = int(self.runner.config.max_model_len)
        too_long = [len(ids) + reserve for ids in prompt_ids if len(ids) + reserve > maximum]
        if too_long:
            raise ValueError(
                f"judge prompt + reserve exceeds max_model_len={maximum}; "
                f"first offending length={too_long[0]}"
            )

    def _vllm_generate(
        self,
        prompt_ids: Sequence[Sequence[int]],
        params: Sequence[Any],
    ) -> list[Any]:
        outputs = self.runner.llm.generate(
            [{"prompt_token_ids": list(ids)} for ids in prompt_ids],
            list(params),
            use_tqdm=False,
            lora_request=getattr(self.runner, "lora_request", None),
        )
        outputs = list(outputs)
        if len(outputs) != len(prompt_ids):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} requests for {len(prompt_ids)} prompts"
            )
        return outputs

    @staticmethod
    def _one_completion(request_output: Any) -> Any:
        outputs = getattr(request_output, "outputs", None)
        if outputs is None or len(outputs) != 1:
            raise RuntimeError("scoring requests must return exactly one completion")
        return outputs[0]

    def _targeted_scores(
        self,
        ids: Sequence[str],
        prompt_ids: Sequence[Sequence[int]],
        label_token_ids: Mapping[str, int],
        *,
        run_seed: int,
        stage: str,
        injected_prompt_tokens: Sequence[int],
    ) -> tuple[list[dict[str, Any]], list[RequestAccounting]]:
        if len(ids) != len(prompt_ids) or len(ids) != len(injected_prompt_tokens):
            raise ValueError("targeted score batch fields have inconsistent lengths")
        if len(label_token_ids) > 20:
            raise ValueError("the local vLLM runner supports at most 20 targeted labels")
        sampling = SamplingConfig(
            thinking="off",
            n=1,
            max_tokens=1,
            answer_max_tokens=1,
            greedy=True,
            run_seed=run_seed,
            logprobs=len(label_token_ids),
            logprob_token_ids=tuple(label_token_ids.values()),
            allow_custom_prompts=True,
        )
        sampling.validate()
        self._check_context(prompt_ids, 1)
        seeds = [
            _stable_seed(run_seed, record_id, 0, stage) for record_id in ids
        ]
        params = [
            self.runner._params(  # noqa: SLF001 - selected-token runner API
                sampling,
                max_tokens=1,
                seed=seed,
                n=1,
            )
            for seed in seeds
        ]
        request_outputs = self._vllm_generate(prompt_ids, params)
        rows: list[dict[str, Any]] = []
        accounts: list[RequestAccounting] = []
        for record_id, exact_prompt, injected, seed, request_output in zip(
            ids, prompt_ids, injected_prompt_tokens, seeds, request_outputs
        ):
            completion = self._one_completion(request_output)
            token_ids = [int(token_id) for token_id in completion.token_ids]
            if not token_ids:
                raise RuntimeError(f"targeted scoring returned no token for {record_id!r}")
            label_logprobs = extract_target_logprobs(completion, label_token_ids)
            probabilities = normalize_label_logprobs(label_logprobs)
            predicted_label = max(probabilities, key=probabilities.__getitem__)
            accounting = RequestAccounting(
                requests=1,
                completions=1,
                prefill_tokens=len(exact_prompt),
                sampled_tokens=len(token_ids),
                injected_prompt_tokens=int(injected),
            )
            accounts.append(accounting)
            rows.append(
                {
                    "id": record_id,
                    "score": probabilities[predicted_label],
                    "predicted_label": predicted_label,
                    "label_logprobs": label_logprobs,
                    "label_probabilities": probabilities,
                    "forced_close": False,
                    "request_count": accounting.requests,
                    "prefill_tokens": accounting.prefill_tokens,
                    "sampled_tokens": accounting.sampled_tokens,
                    "accounting": accounting.as_dict(),
                    "raw": {
                        "prompt_token_ids": list(exact_prompt),
                        "sampled_token_ids": token_ids,
                        "sampled_text": self.runner._decode(token_ids),  # noqa: SLF001
                        "first_token_logprobs": _raw_logprobs(completion),
                        "finish_reason": getattr(completion, "finish_reason", None),
                        "stop_reason": getattr(completion, "stop_reason", None),
                        "sampled_cumulative_logprob": getattr(
                            completion, "cumulative_logprob", None
                        ),
                        "seed": seed,
                        "sampled_label_is_audited": token_ids[0]
                        in label_token_ids.values(),
                    },
                }
            )
        return rows, accounts

    def score_no_think_viability(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        run_seed: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return no-think P(viable), normalized over A/B."""
        records = list(records)
        if not records:
            raise ValueError("scoring input must not be empty")
        ids = _validate_unique_ids(records)
        message_sets = [
            build_binary_viability_messages(
                _record_task(record),
                _record_prefix(record),
                _record_remaining_slots(record),
                visible_examples=record.get("visible_examples"),
            )
            for record in records
        ]
        rendered = [
            self._render_chat_ids(messages, thinking=False)
            for messages in message_sets
        ]
        prompt_ids = [ids_ + self.answer_prefix_ids for _, ids_ in rendered]
        score_rows, accounts = self._targeted_scores(
            ids,
            prompt_ids,
            self.binary_label_ids,
            run_seed=run_seed,
            stage="nothink_viability_label",
            injected_prompt_tokens=[len(self.answer_prefix_ids)] * len(ids),
        )
        rows: list[dict[str, Any]] = []
        for record, row, (prompt_text, _) in zip(records, score_rows, rendered):
            row["score"] = row["label_probabilities"]["A"]
            row["p_viable"] = row["score"]
            row["meta"] = _record_meta(record)
            row["raw"]["rendered_chat_prompt"] = prompt_text
            rows.append(row)
        total = _sum_accounting(accounts)
        return rows, {
            "method": "no_think_p_viable",
            "label_semantics": {"A": "viable", "B": "not_viable"},
            "label_token_ids": dict(self.binary_label_ids),
            "run_seed": run_seed,
            "vllm_batch_calls": 1,
            "accounting": total.as_dict(),
        }

    def score_thinking_viability(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        thinking_budget: int,
        run_seed: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return two-pass thinking P(viable), normalized over A/B."""
        if (
            isinstance(thinking_budget, bool)
            or not isinstance(thinking_budget, int)
            or thinking_budget < 1
        ):
            raise ValueError("thinking_budget must be a positive integer")
        records = list(records)
        if not records:
            raise ValueError("scoring input must not be empty")
        ids = _validate_unique_ids(records)
        message_sets = [
            build_binary_viability_messages(
                _record_task(record),
                _record_prefix(record),
                _record_remaining_slots(record),
                visible_examples=record.get("visible_examples"),
            )
            for record in records
        ]
        rendered = [
            self._render_chat_ids(messages, thinking=True)
            for messages in message_sets
        ]
        base_prompt_ids = [token_ids for _, token_ids in rendered]
        thinking_sampling = SamplingConfig(
            thinking="natural",
            n=1,
            max_tokens=thinking_budget,
            answer_max_tokens=1,
            run_seed=run_seed,
        )
        thinking_sampling.validate()
        self._check_context(base_prompt_ids, thinking_budget)
        thinking_seeds = [
            _stable_seed(run_seed, record_id, 0, "viability_thinking")
            for record_id in ids
        ]
        thinking_params = [
            self.runner._params(  # noqa: SLF001
                thinking_sampling,
                max_tokens=thinking_budget,
                seed=seed,
                n=1,
            )
            for seed in thinking_seeds
        ]
        thinking_outputs = self._vllm_generate(base_prompt_ids, thinking_params)

        retained_by_row: list[list[int]] = []
        phase1_raw_by_row: list[list[int]] = []
        phase1_trimmed_by_row: list[list[int]] = []
        forced_by_row: list[bool] = []
        phase1_completions: list[Any] = []
        score_prompt_ids: list[list[int]] = []
        phase1_accounts: list[RequestAccounting] = []
        for base_ids, request_output in zip(base_prompt_ids, thinking_outputs):
            completion = self._one_completion(request_output)
            phase1_completions.append(completion)
            sampled = [int(token_id) for token_id in completion.token_ids]
            trimmed = list(self.runner._trim_hf_eos(sampled))  # noqa: SLF001
            if self.runner.think_close_id in trimmed:
                close_index = trimmed.index(self.runner.think_close_id)
                retained = trimmed[:close_index]
                forced = False
            else:
                retained = trimmed
                forced = True
            phase1_raw_by_row.append(sampled)
            phase1_trimmed_by_row.append(trimmed)
            retained_by_row.append(retained)
            forced_by_row.append(forced)
            score_prompt_ids.append(
                list(base_ids) + retained + self.thinking_answer_suffix_ids
            )
            phase1_accounts.append(
                RequestAccounting(
                    requests=1,
                    completions=1,
                    prefill_tokens=len(base_ids),
                    sampled_tokens=len(sampled),
                    retained_thinking_tokens=len(retained),
                    discarded_sampled_tokens=len(sampled) - len(retained),
                )
            )

        score_rows, phase2_accounts = self._targeted_scores(
            ids,
            score_prompt_ids,
            self.binary_label_ids,
            run_seed=run_seed,
            stage="thinking_viability_label",
            injected_prompt_tokens=[len(self.thinking_answer_suffix_ids)] * len(ids),
        )
        rows: list[dict[str, Any]] = []
        combined_accounts: list[RequestAccounting] = []
        for (
            record,
            row,
            rendered_prompt,
            base_ids,
            phase1_completion,
            sampled,
            trimmed,
            retained,
            forced,
            phase1_seed,
            phase1_account,
            phase2_account,
        ) in zip(
            records,
            score_rows,
            (text for text, _ in rendered),
            base_prompt_ids,
            phase1_completions,
            phase1_raw_by_row,
            phase1_trimmed_by_row,
            retained_by_row,
            forced_by_row,
            thinking_seeds,
            phase1_accounts,
            phase2_accounts,
        ):
            accounting = phase1_account + phase2_account
            combined_accounts.append(accounting)
            row["score"] = row["label_probabilities"]["A"]
            row["p_viable"] = row["score"]
            row["forced_close"] = forced
            row["request_count"] = accounting.requests
            row["prefill_tokens"] = accounting.prefill_tokens
            row["sampled_tokens"] = accounting.sampled_tokens
            row["accounting"] = accounting.as_dict()
            row["meta"] = _record_meta(record)
            row["raw"].update(
                {
                    "rendered_chat_prompt": rendered_prompt,
                    "phase1_prompt_token_ids": list(base_ids),
                    "phase1_sampled_token_ids": sampled,
                    "phase1_trimmed_token_ids": trimmed,
                    "retained_thinking_token_ids": retained,
                    "retained_thinking_text": self.runner._decode(retained),  # noqa: SLF001
                    "phase1_finish_reason": getattr(
                        phase1_completion, "finish_reason", None
                    ),
                    "phase1_stop_reason": getattr(
                        phase1_completion, "stop_reason", None
                    ),
                    "phase1_cumulative_logprob": getattr(
                        phase1_completion, "cumulative_logprob", None
                    ),
                    "phase1_seed": phase1_seed,
                }
            )
            rows.append(row)
        total = _sum_accounting(combined_accounts)
        return rows, {
            "method": "thinking_p_viable",
            "thinking_budget": thinking_budget,
            "label_semantics": {"A": "viable", "B": "not_viable"},
            "label_token_ids": dict(self.binary_label_ids),
            "run_seed": run_seed,
            "thinking_answer_suffix_token_ids": list(
                self.thinking_answer_suffix_ids
            ),
            "vllm_batch_calls": 2,
            "accounting": total.as_dict(),
        }

    def score_next_operation_likelihood(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        run_seed: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return the normalized A--P next-operation likelihood distribution."""
        records = list(records)
        if not records:
            raise ValueError("scoring input must not be empty")
        ids = _validate_unique_ids(records)
        choices_by_row = [
            tuple(
                _record_value(
                    record,
                    ("choices", "operations", "next_operations"),
                    "next-operation choices",
                )
            )
            for record in records
        ]
        message_sets = [
            build_next_operation_messages(
                _record_task(record),
                _record_prefix(record),
                _record_remaining_slots(record),
                choices,
                visible_examples=record.get("visible_examples"),
            )
            for record, choices in zip(records, choices_by_row)
        ]
        rendered = [
            self._render_chat_ids(messages, thinking=False)
            for messages in message_sets
        ]
        prompt_ids = [ids_ + self.answer_prefix_ids for _, ids_ in rendered]
        score_rows, accounts = self._targeted_scores(
            ids,
            prompt_ids,
            self.next_operation_label_ids,
            run_seed=run_seed,
            stage="next_operation_label",
            injected_prompt_tokens=[len(self.answer_prefix_ids)] * len(ids),
        )
        rows: list[dict[str, Any]] = []
        for record, choices, row, (prompt_text, _) in zip(
            records, choices_by_row, score_rows, rendered
        ):
            by_choice = {
                choice: row["label_probabilities"][label]
                for label, choice in zip(NEXT_OPERATION_LABELS, choices)
            }
            predicted_index = NEXT_OPERATION_LABELS.index(row["predicted_label"])
            row["score"] = row["label_probabilities"][row["predicted_label"]]
            row["predicted_choice"] = choices[predicted_index]
            row["choice_probabilities"] = by_choice
            row["ranked_choices"] = sorted(
                by_choice, key=by_choice.__getitem__, reverse=True
            )
            row["meta"] = _record_meta(record)
            row["raw"]["rendered_chat_prompt"] = prompt_text
            rows.append(row)
        total = _sum_accounting(accounts)
        return rows, {
            "method": "next_operation_likelihood",
            "label_token_ids": dict(self.next_operation_label_ids),
            "run_seed": run_seed,
            "vllm_batch_calls": 1,
            "accounting": total.as_dict(),
        }


def score_thinking_ptrue(
    runner: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    thinking_budget: int,
    run_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Functional wrapper for ``ModelScorer.score_thinking_viability``."""
    return ModelScorer(runner).score_thinking_viability(
        records, thinking_budget=thinking_budget, run_seed=run_seed
    )


def score_nothink_ptrue(
    runner: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    run_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Functional wrapper for ``ModelScorer.score_no_think_viability``."""
    return ModelScorer(runner).score_no_think_viability(records, run_seed=run_seed)


def score_next_op_likelihood(
    runner: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    run_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Functional wrapper for ``ModelScorer.score_next_operation_likelihood``."""
    return ModelScorer(runner).score_next_operation_likelihood(
        records, run_seed=run_seed
    )
