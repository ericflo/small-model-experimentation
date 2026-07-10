"""Auditable vLLM operations for thought harvest, answer scoring, and rollout.

This module deliberately uses the experiment-local runner for model loading
and prompt rendering, then calls its in-process vLLM engine for the two
measurements the generic runner does not expose: stop-at-``</think>`` harvest
and observed-prompt-token likelihood.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gym import base
from gym.families import load as load_train_family
from io_utils import sha256_file
from vllm_runner import (
    MODEL_ID,
    MODEL_REVISION,
    EngineConfig,
    SamplingConfig,
    VLLMRunner,
    _stable_seed,
)

ANSWER_BOUNDARY = "</think>\n\nANSWER: "
FORMAT_VARIANT_BOUNDARY = "</think>\nANSWER: "


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _loop_diagnostic(token_ids: Sequence[int]) -> dict[str, Any]:
    if len(token_ids) < 9:
        return {"max_trigram_count": 1, "loop_flag": False}
    counts = Counter(tuple(token_ids[index : index + 3]) for index in range(len(token_ids) - 2))
    maximum = max(counts.values(), default=1)
    return {
        "max_trigram_count": maximum,
        "loop_flag": maximum >= 8,
    }


def _extract_observed_logprob(entry: Any, token_id: int) -> float:
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"prompt logprob entry is not a mapping: {type(entry).__name__}")
    value = entry.get(token_id)
    if value is None:
        value = entry.get(str(token_id))
    if value is None:
        raise RuntimeError(
            f"observed token {token_id} missing from prompt-logprob entry with keys "
            f"{list(entry)[:8]}"
        )
    logprob = float(value.logprob if hasattr(value, "logprob") else value["logprob"])
    if not math.isfinite(logprob):
        raise RuntimeError(f"nonfinite observed-token logprob: {logprob}")
    return logprob


class AnswerPotentialModel:
    """One loaded Qwen3.5-4B vLLM engine with exact token accounting."""

    def __init__(self, engine_config: EngineConfig):
        self.runner = VLLMRunner(engine_config)
        self.engine_config = engine_config
        self._thinking_prompt_cache: dict[str, tuple[str, list[int]]] = {}
        self.answer_boundary_ids = self.runner.tokenizer.encode(
            ANSWER_BOUNDARY, add_special_tokens=False
        )
        self.format_variant_boundary_ids = self.runner.tokenizer.encode(
            FORMAT_VARIANT_BOUNDARY, add_special_tokens=False
        )
        if not self.answer_boundary_ids or self.answer_boundary_ids[0] != self.runner.think_close_id:
            raise RuntimeError("unexpected answer-boundary tokenization")
        if self.answer_boundary_ids[:2] != self.runner.close_ids:
            raise RuntimeError(
                "registered answer boundary does not start with the runner's exact close sequence"
            )
        self._logical_counts = Counter()

    def close(self) -> None:
        self.runner.close()

    def __enter__(self) -> "AnswerPotentialModel":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def thinking_prompt(self, item: Mapping[str, Any]) -> tuple[str, list[int]]:
        task_id = str(item["id"])
        cached = self._thinking_prompt_cache.get(task_id)
        if cached is not None:
            return cached
        rendered = self.runner._render_messages(  # noqa: SLF001
            [{"role": "user", "content": str(item["prompt"])}], True
        )
        token_ids = self.runner.tokenizer.encode(rendered, add_special_tokens=False)
        if self.runner._prompt_channel(token_ids) != "thinking":  # noqa: SLF001
            raise RuntimeError(f"task {task_id} did not render in the thinking channel")
        result = (rendered, list(token_ids))
        self._thinking_prompt_cache[task_id] = result
        return result

    def capacity_receipt(self, *, largest_reservation: int, logical_sequences: int) -> dict[str, Any]:
        active = min(logical_sequences, self.engine_config.max_num_seqs)
        receipt: dict[str, Any] = {
            "logical_sequences": logical_sequences,
            "active_sequence_cap": active,
            "largest_prompt_plus_generation_tokens": largest_reservation,
            "requested_active_tokens_upper_bound": active * largest_reservation,
            "source": None,
            "passed": None,
        }
        try:
            cache = self.runner.llm.llm_engine.vllm_config.cache_config
            blocks = int(cache.num_gpu_blocks)
            block_size = int(cache.block_size)
            rounded = ((largest_reservation + block_size - 1) // block_size) * block_size
            available = blocks * block_size
            required = active * rounded
            receipt.update(
                source="llm_engine.vllm_config.cache_config",
                gpu_cache_blocks=blocks,
                block_size=block_size,
                rounded_sequence_reservation=rounded,
                live_kv_cache_tokens=available,
                required_live_kv_cache_tokens=required,
                remaining_token_margin=available - required,
                passed=required <= available,
            )
            if required > available:
                raise RuntimeError(
                    f"vLLM live KV capacity does not fit the registered concurrency: "
                    f"required={required}, available={available}"
                )
        except AttributeError:
            # vLLM V1 does not expose the scheduler's cache geometry uniformly.
            # The receipt is explicit about the unavailable audit rather than
            # fabricating a passing value.
            receipt.update(source="unavailable_in_vllm_v1_public_state", passed=None)
        return receipt

    def metadata(self, *, operation: str, started: float, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": 1,
            "operation": operation,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": sha256_file(Path(__file__).with_name("vllm_runner.py")),
            "model_ops_sha256": sha256_file(Path(__file__)),
            "engine": {
                key: str(item) if isinstance(item, Path) else item
                for key, item in asdict(self.engine_config).items()
            },
            "engine_args": {
                key: str(item) if isinstance(item, Path) else item
                for key, item in self.runner.engine_args.items()
            },
            "resolved_cudagraph": self.runner.resolved_cudagraph,
            "logical_counts": dict(sorted(self._logical_counts.items())),
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": self.runner.runtime_metadata(),
        }
        if extra:
            value["extra"] = dict(extra)
        return value

    def generate_thoughts(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        n: int,
        max_tokens: int,
        run_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        chunk_size: int = 128,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sample only the thinking region, stopping at the first close token."""
        from vllm import SamplingParams

        started = time.perf_counter()
        traces: list[dict[str, Any]] = []
        largest = 0
        for item_chunk in _chunks(items, chunk_size):
            prompts: list[dict[str, list[int]]] = []
            params: list[Any] = []
            prompt_ids_by_task: list[list[int]] = []
            for item in item_chunk:
                _, prompt_ids = self.thinking_prompt(item)
                largest = max(largest, len(prompt_ids) + max_tokens)
                prompt_ids_by_task.append(prompt_ids)
                prompts.append({"prompt_token_ids": prompt_ids})
                params.append(
                    SamplingParams(
                        n=n,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=_stable_seed(run_seed, str(item["id"]), -1, "thought_only"),
                        max_tokens=max_tokens,
                        ignore_eos=True,
                        stop_token_ids=[self.runner.think_close_id, self.runner.hf_eos_id],
                        include_stop_str_in_output=True,
                        detokenize=True,
                        skip_special_tokens=False,
                    )
                )
            outputs = self.runner.llm.generate(
                prompts, params, use_tqdm=False, lora_request=self.runner.lora_request
            )
            for item, prompt_ids, request_output in zip(item_chunk, prompt_ids_by_task, outputs):
                seen: set[tuple[int, ...]] = set()
                parent_seed = _stable_seed(run_seed, str(item["id"]), -1, "thought_only")
                for completion in request_output.outputs:
                    sample_index = int(completion.index)
                    sampled = [int(token_id) for token_id in completion.token_ids]
                    trimmed = list(self.runner._trim_hf_eos(sampled))  # noqa: SLF001
                    natural_close = self.runner.think_close_id in trimmed or (
                        getattr(completion, "stop_reason", None) == self.runner.think_close_id
                    )
                    if self.runner.think_close_id in trimmed:
                        retained = trimmed[: trimmed.index(self.runner.think_close_id)]
                    else:
                        retained = trimmed
                    key = tuple(retained)
                    duplicate = key in seen
                    seen.add(key)
                    prior_sum = getattr(completion, "cumulative_logprob", None)
                    loop = _loop_diagnostic(retained)
                    traces.append(
                        {
                            "trace_id": f"{item['id']}::t{sample_index:03d}",
                            "task_id": item["id"],
                            "family": item["family"],
                            "level": item["level"],
                            "sample_index": sample_index,
                            "parent_seed": parent_seed,
                            "effective_seed": parent_seed + sample_index,
                            "token_ids": retained,
                            "text": self.runner._decode(retained),  # noqa: SLF001
                            "n_tokens": len(retained),
                            "n_sampled_tokens": len(sampled),
                            "natural_close": natural_close,
                            "forced_close_required": not natural_close,
                            "finish_reason": getattr(completion, "finish_reason", None),
                            "stop_reason": getattr(completion, "stop_reason", None),
                            "prior_logprob_sum": prior_sum,
                            "prior_logprob_mean": (
                                float(prior_sum) / len(sampled)
                                if prior_sum is not None and sampled
                                else None
                            ),
                            "exact_duplicate_within_task": duplicate,
                            **loop,
                        }
                    )
                    self._logical_counts["thought_sampled_tokens"] += len(sampled)
                    self._logical_counts["thought_prompt_tokens"] += len(prompt_ids)
        receipt = self.metadata(
            operation="thought_only_harvest",
            started=started,
            extra={
                "items": len(items),
                "n_per_item": n,
                "traces": len(traces),
                "sampling": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "run_seed": run_seed,
                    "stop_token_ids": [self.runner.think_close_id, self.runner.hf_eos_id],
                },
                "capacity_fit": self.capacity_receipt(
                    largest_reservation=largest, logical_sequences=len(items) * n
                ),
            },
        )
        return traces, receipt

    def _score_sequence_requests(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        chunk_size: int,
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.v1.outputs import LogprobsTensors

        # vLLM 0.24's prompt_logprobs=0 path computes a full-vocabulary rank
        # for EVERY prompt position.  We need only the observed answer span.
        # Score the same teacher-forced factorization as one targeted
        # next-token request per answer token.  The target prefix is injected;
        # the next sampled token is *not* forced, and logprob_token_ids returns
        # the target token's raw model logprob.
        token_requests: list[dict[str, Any]] = []
        for request in requests:
            answer_start = int(request["answer_start"])
            target_ids = request["full_token_ids"][answer_start:]
            for token_index, token_id in enumerate(target_ids):
                token_requests.append(
                    {
                        "sequence_request_id": request["request_id"],
                        "token_index": token_index,
                        "target_token_id": int(token_id),
                        "prefix_token_ids": request["full_token_ids"][
                            : answer_start + token_index
                        ],
                    }
                )

        sampler = (
            self.runner.llm.llm_engine.model_executor.driver_worker.model_runner.sampler
        )
        had_instance_override = "gather_specific_token_logprobs" in sampler.__dict__
        previous_instance_override = sampler.__dict__.get(
            "gather_specific_token_logprobs"
        )

        def gather_target_without_unused_rank(
            logprobs: Any,
            logprob_token_ids: dict[int, list[int]],
            sampled: Any,
        ) -> Any:
            """Gather sampled+registered target logprobs, assigning dummy rank 1.

            Rank is output metadata only and is not used by this experiment.
            vLLM otherwise invokes batched_count_greater_than over the entire
            248k padded vocabulary merely to report the sampled token's rank.
            """
            import torch

            batch_size = int(logprobs.shape[0])
            if set(logprob_token_ids) != set(range(batch_size)) or any(
                len(values) != 1 for values in logprob_token_ids.values()
            ):
                raise RuntimeError(
                    "targeted scorer requires exactly one registered token per request"
                )
            targets = torch.tensor(
                [logprob_token_ids[index][0] for index in range(batch_size)],
                dtype=torch.int64,
                device=logprobs.device,
            )
            token_ids = torch.stack((sampled.to(torch.int64), targets), dim=1)
            gathered = logprobs.gather(-1, token_ids)
            dummy_ranks = torch.ones(
                batch_size, dtype=torch.int32, device=logprobs.device
            )
            return LogprobsTensors(
                logprob_token_ids=token_ids.to(torch.int32),
                logprobs=gathered,
                selected_token_ranks=dummy_ranks,
            )

        # Instance assignment avoids editing the pinned environment.  Restore
        # it in finally so ordinary runner calls retain stock vLLM behavior.
        sampler.__dict__["gather_specific_token_logprobs"] = (
            gather_target_without_unused_rank
        )

        token_scores: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        try:
            for request_chunk in _chunks(token_requests, chunk_size):
                prompts = [
                    {"prompt_token_ids": request["prefix_token_ids"]}
                    for request in request_chunk
                ]
                params = [
                    SamplingParams(
                        n=1,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        seed=0,
                        max_tokens=1,
                        logprob_token_ids=[request["target_token_id"]],
                        detokenize=False,
                        skip_special_tokens=False,
                    )
                    for request in request_chunk
                ]
                outputs = self.runner.llm.generate(
                    prompts,
                    params,
                    use_tqdm=False,
                    lora_request=self.runner.lora_request,
                )
                for request, request_output in zip(request_chunk, outputs):
                    if len(request_output.outputs) != 1:
                        raise RuntimeError("targeted score request returned !=1 completion")
                    completion = request_output.outputs[0]
                    if not completion.logprobs or len(completion.logprobs) != 1:
                        raise RuntimeError(
                            f"targeted logprobs missing for {request['sequence_request_id']}"
                        )
                    logprob = _extract_observed_logprob(
                        completion.logprobs[0], request["target_token_id"]
                    )
                    token_scores[str(request["sequence_request_id"])].append(
                        (
                            int(request["token_index"]),
                            int(request["target_token_id"]),
                            logprob,
                        )
                    )
                    self._logical_counts["score_prefill_tokens"] += len(
                        request["prefix_token_ids"]
                    )
                    self._logical_counts["score_sampled_tokens"] += len(
                        completion.token_ids
                    )
        finally:
            if had_instance_override:
                sampler.__dict__["gather_specific_token_logprobs"] = (
                    previous_instance_override
                )
            else:
                sampler.__dict__.pop("gather_specific_token_logprobs", None)

        results: list[dict[str, Any]] = []
        for request in requests:
            triples = sorted(token_scores[str(request["request_id"])])
            expected_count = len(request["full_token_ids"]) - int(
                request["answer_start"]
            )
            if [index for index, _, _ in triples] != list(range(expected_count)):
                raise RuntimeError(
                    f"incomplete targeted score for {request['request_id']}: {triples}"
                )
            token_ids = [token_id for _, token_id, _ in triples]
            token_logprobs = [logprob for _, _, logprob in triples]
            results.append(
                {
                    **{
                        key: value
                        for key, value in request.items()
                        if key != "full_token_ids"
                    },
                    "answer_token_ids": token_ids,
                    "answer_token_logprobs": token_logprobs,
                    "ll_sum": sum(token_logprobs),
                    "ll_mean": sum(token_logprobs) / len(token_logprobs),
                    "first_logprob": token_logprobs[0],
                }
            )
        return results

    def score_answer_potential(
        self,
        items: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
        *,
        boundary: str = "canonical",
        include_decoy: bool = True,
        chunk_size: int = 256,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Teacher-force all registered answer variants after every thought."""
        started = time.perf_counter()
        item_by_id = {str(item["id"]): item for item in items}
        if len(item_by_id) != len(items):
            raise ValueError("duplicate task IDs in potential-scoring input")
        unknown = sorted({str(trace["task_id"]) for trace in traces} - set(item_by_id))
        if unknown:
            raise ValueError(f"traces refer to unknown tasks: {unknown[:5]}")
        boundary_ids = (
            self.answer_boundary_ids
            if boundary == "canonical"
            else self.format_variant_boundary_ids
            if boundary == "format_variant"
            else None
        )
        if boundary_ids is None:
            raise ValueError("boundary must be canonical or format_variant")

        conditions: list[dict[str, Any]] = []
        for item in items:
            if not item.get("potential_scorable", True):
                continue
            conditions.append(
                {
                    "trace_id": f"{item['id']}::empty",
                    "task_id": item["id"],
                    "condition": "empty",
                    "token_ids": [],
                }
            )
        conditions.extend(dict(trace) for trace in traces if item_by_id[str(trace["task_id"])].get("potential_scorable", True))

        requests: list[dict[str, Any]] = []
        for condition in conditions:
            item = item_by_id[str(condition["task_id"])]
            _, prompt_ids = self.thinking_prompt(item)
            prefix_ids = prompt_ids + [int(value) for value in condition.get("token_ids", [])] + list(boundary_ids)
            variants = list(item["answer_variants"])
            if not variants:
                raise ValueError(f"scorable task {item['id']} has no answer variants")
            for variant_index, answer in enumerate(variants):
                answer_ids = self.runner.tokenizer.encode(str(answer), add_special_tokens=False)
                if not answer_ids:
                    raise ValueError(f"empty tokenized answer for {item['id']}")
                requests.append(
                    {
                        "request_id": f"{condition['trace_id']}::{boundary}::a{variant_index}",
                        "trace_id": condition["trace_id"],
                        "task_id": item["id"],
                        "condition": condition.get("condition", "real"),
                        "variant_index": variant_index,
                        "answer_text": answer,
                        "canonical_variant": answer == item["canonical_answer"],
                        "answer_start": len(prefix_ids),
                        "full_token_ids": prefix_ids + answer_ids,
                    }
                )
            if include_decoy and item.get("procedural_decoy") is not None:
                decoy = str(item["procedural_decoy"])
                decoy_ids = self.runner.tokenizer.encode(decoy, add_special_tokens=False)
                requests.append(
                    {
                        "request_id": f"{condition['trace_id']}::{boundary}::decoy",
                        "trace_id": condition["trace_id"],
                        "task_id": item["id"],
                        "condition": condition.get("condition", "real"),
                        "variant_index": -1,
                        "answer_text": decoy,
                        "canonical_variant": False,
                        "is_decoy": True,
                        "answer_start": len(prefix_ids),
                        "full_token_ids": prefix_ids + decoy_ids,
                    }
                )
        raw = self._score_sequence_requests(requests, chunk_size=chunk_size)
        by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            by_trace[str(row["trace_id"])].append(row)

        aggregated: list[dict[str, Any]] = []
        for condition in conditions:
            rows = by_trace[str(condition["trace_id"])]
            answer_rows = [row for row in rows if not row.get("is_decoy", False)]
            decoy_rows = [row for row in rows if row.get("is_decoy", False)]
            canonical_rows = [row for row in answer_rows if row["canonical_variant"]]
            if len(canonical_rows) != 1:
                raise RuntimeError(
                    f"expected exactly one canonical variant for {condition['trace_id']}, "
                    f"got {len(canonical_rows)}"
                )
            canonical = canonical_rows[0]
            equivalence_ll = _logsumexp([float(row["ll_sum"]) for row in answer_rows])
            result = {
                **{key: value for key, value in condition.items() if key != "token_ids"},
                "boundary": boundary,
                "equivalence_ll_sum": equivalence_ll,
                "canonical_ll_sum": canonical["ll_sum"],
                "canonical_ll_mean": canonical["ll_mean"],
                "first_answer_logprob": canonical["first_logprob"],
                "canonical_answer_token_ids": canonical["answer_token_ids"],
                "variant_scores": [
                    {
                        "answer_text": row["answer_text"],
                        "answer_token_ids": row["answer_token_ids"],
                        "answer_token_logprobs": row["answer_token_logprobs"],
                        "ll_sum": row["ll_sum"],
                        "ll_mean": row["ll_mean"],
                    }
                    for row in answer_rows
                ],
                "decoy_text": decoy_rows[0]["answer_text"] if decoy_rows else None,
                "decoy_ll_sum": decoy_rows[0]["ll_sum"] if decoy_rows else None,
                "decoy_token_logprobs": (
                    decoy_rows[0]["answer_token_logprobs"] if decoy_rows else None
                ),
                "canonical_decoy_margin": (
                    float(canonical["ll_sum"]) - float(decoy_rows[0]["ll_sum"])
                    if decoy_rows
                    else None
                ),
            }
            aggregated.append(result)

        empty_by_task = {
            str(row["task_id"]): row
            for row in aggregated
            if row.get("condition") == "empty"
        }
        output: list[dict[str, Any]] = []
        trace_by_id = {str(trace["trace_id"]): trace for trace in traces}
        for row in aggregated:
            if row.get("condition") == "empty":
                continue
            baseline = empty_by_task[str(row["task_id"])]
            source = trace_by_id.get(str(row["trace_id"]), {})
            answer_tokens = len(row["canonical_answer_token_ids"])
            output.append(
                {
                    **row,
                    "gain_sum": float(row["equivalence_ll_sum"])
                    - float(baseline["equivalence_ll_sum"]),
                    "gain_per_canonical_answer_token": (
                        float(row["equivalence_ll_sum"])
                        - float(baseline["equivalence_ll_sum"])
                    )
                    / answer_tokens,
                    "empty_equivalence_ll_sum": baseline["equivalence_ll_sum"],
                    "empty_canonical_ll_sum": baseline["canonical_ll_sum"],
                    "n_trace_tokens": source.get("n_tokens", len(source.get("token_ids", []))),
                    "prior_logprob_mean": source.get("prior_logprob_mean"),
                }
            )
        largest = max((len(request["full_token_ids"]) for request in requests), default=0) + 1
        receipt = self.metadata(
            operation=f"answer_potential_{boundary}",
            started=started,
            extra={
                "items": len(items),
                "traces_input": len(traces),
                "traces_scored": len(output),
                "answer_sequences": len(requests),
                "targeted_next_token_requests": sum(
                    len(request["full_token_ids"]) - int(request["answer_start"])
                    for request in requests
                ),
                "readout": "vllm_raw_logprob_token_ids_at_exact_teacher_forced_prefix",
                "sampled_token_rank_bypassed": True,
                "sampled_token_rank_used": False,
                "sampled_tokens_per_score_request": 1,
                "answer_boundary": ANSWER_BOUNDARY if boundary == "canonical" else FORMAT_VARIANT_BOUNDARY,
                "capacity_fit": self.capacity_receipt(
                    largest_reservation=largest, logical_sequences=len(requests)
                ),
            },
        )
        return output, receipt

    def generate_answer_rollouts(
        self,
        items: Sequence[Mapping[str, Any]],
        traces: Sequence[Mapping[str, Any]],
        *,
        r: int,
        max_tokens: int,
        run_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        chunk_size: int = 256,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sample fresh answer continuations from fixed trace-conditioned states."""
        from vllm import SamplingParams

        started = time.perf_counter()
        item_by_id = {str(item["id"]): item for item in items}
        rows: list[dict[str, Any]] = []
        largest = 0
        for trace_chunk in _chunks(traces, chunk_size):
            prompts: list[dict[str, list[int]]] = []
            params: list[Any] = []
            prefixes: list[list[int]] = []
            for trace in trace_chunk:
                item = item_by_id[str(trace["task_id"])]
                _, base_ids = self.thinking_prompt(item)
                prefix = base_ids + [int(value) for value in trace["token_ids"]] + list(self.runner.close_ids)
                prefixes.append(prefix)
                largest = max(largest, len(prefix) + max_tokens)
                prompts.append({"prompt_token_ids": prefix})
                params.append(
                    SamplingParams(
                        n=r,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=_stable_seed(run_seed, str(trace["trace_id"]), -1, "answer_rollout"),
                        max_tokens=max_tokens,
                        ignore_eos=True,
                        stop_token_ids=[self.runner.hf_eos_id],
                        detokenize=True,
                        skip_special_tokens=False,
                    )
                )
            outputs = self.runner.llm.generate(
                prompts, params, use_tqdm=False, lora_request=self.runner.lora_request
            )
            for trace, prefix, request_output in zip(trace_chunk, prefixes, outputs):
                item = item_by_id[str(trace["task_id"])]
                module = load_train_family(str(item["family"]))
                parent_seed = _stable_seed(
                    run_seed, str(trace["trace_id"]), -1, "answer_rollout"
                )
                outcomes = []
                for completion in request_output.outputs:
                    token_ids = list(self.runner._trim_hf_eos(completion.token_ids))  # noqa: SLF001
                    full_text = self.runner._decode(  # noqa: SLF001
                        [int(value) for value in trace["token_ids"]]
                        + list(self.runner.close_ids)
                        + token_ids
                    )
                    score = float(module.score_atom(item, full_text))
                    outcomes.append(
                        {
                            "rollout_index": int(completion.index),
                            "parent_seed": parent_seed,
                            "effective_seed": parent_seed + int(completion.index),
                            "token_ids": token_ids,
                            "text": self.runner._decode(token_ids),  # noqa: SLF001
                            "answer_value": base.extract_answer(full_text),
                            "score": score,
                            "correct": score == 1.0,
                            "finish_reason": getattr(completion, "finish_reason", None),
                            "stop_reason": getattr(completion, "stop_reason", None),
                            "n_sampled_tokens": len(completion.token_ids),
                        }
                    )
                    self._logical_counts["rollout_prompt_tokens"] += len(prefix)
                    self._logical_counts["rollout_sampled_tokens"] += len(completion.token_ids)
                rows.append(
                    {
                        "trace_id": trace["trace_id"],
                        "task_id": trace["task_id"],
                        "family": trace["family"],
                        "level": trace["level"],
                        "r": r,
                        "success_fraction": sum(outcome["correct"] for outcome in outcomes) / r,
                        "any_success": any(outcome["correct"] for outcome in outcomes),
                        "outcomes": outcomes,
                    }
                )
        receipt = self.metadata(
            operation="trace_conditioned_answer_rollouts",
            started=started,
            extra={
                "traces": len(traces),
                "r": r,
                "sampling": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                    "run_seed": run_seed,
                },
                "capacity_fit": self.capacity_receipt(
                    largest_reservation=largest, logical_sequences=len(traces) * r
                ),
            },
        )
        return rows, receipt


def make_token_shuffled_controls(
    traces: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for trace in traces:
        token_ids = [int(value) for value in trace["token_ids"]]
        shuffled = list(token_ids)
        random.Random(
            _stable_seed(seed, str(trace["trace_id"]), -1, "token_shuffle")
        ).shuffle(shuffled)
        controls.append(
            {
                **dict(trace),
                "trace_id": f"{trace['trace_id']}::token_shuffled",
                "source_trace_id": trace["trace_id"],
                "condition": "token_shuffled",
                "token_ids": shuffled,
                "text": "",
            }
        )
    return controls


def make_foreign_controls(traces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cyclic within-family/level foreign traces, greedily length matched."""
    by_stratum: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_stratum[(str(trace["family"]), int(trace["level"]))].append(trace)
    controls: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda row: (str(row["task_id"]), str(row["trace_id"])))
        task_ids = sorted({str(row["task_id"]) for row in group})
        if len(task_ids) < 2:
            continue
        next_task = {task_id: task_ids[(index + 1) % len(task_ids)] for index, task_id in enumerate(task_ids)}
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            by_task[str(row["task_id"])].append(row)
        for trace in group:
            candidates = by_task[next_task[str(trace["task_id"])]]
            source = min(
                candidates,
                key=lambda row: (abs(int(row["n_tokens"]) - int(trace["n_tokens"])), str(row["trace_id"])),
            )
            controls.append(
                {
                    **dict(trace),
                    "trace_id": f"{trace['trace_id']}::foreign",
                    "source_trace_id": source["trace_id"],
                    "condition": "foreign",
                    "token_ids": list(source["token_ids"]),
                    "text": source.get("text", ""),
                    "foreign_source_task_id": source["task_id"],
                }
            )
    return controls


def answer_mention(text: str, answer: str) -> int | None:
    """First case-insensitive verbatim mention with token boundaries."""
    if not answer.strip():
        return None
    escaped = re.escape(answer.strip())
    pattern = re.compile(rf"(?<![\w-]){escaped}(?![\w-])", re.IGNORECASE)
    match = pattern.search(text)
    return match.start() if match else None
