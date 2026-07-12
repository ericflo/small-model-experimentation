"""Memory-bounded full-sequence answer and close-plus-answer likelihood scorer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from vllm_runner import MODEL_ID, MODEL_REVISION

ANSWER_BOUNDARY = "</think>\n\nANSWER: "
FORMAT_VARIANT_BOUNDARY = "</think>\nANSWER: "


class HFAnswerPotentialScorer:
    """Score only registered target positions while processing the full prefix once."""

    def __init__(self, *, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=local_files_only,
        )
        # Qwen3.5-4B is packaged with the multimodal wrapper config even when
        # used text-only.  AutoModelForCausalLM currently maps that outer
        # config to Qwen3_5ForCausalLM and then passes the wrong (outer)
        # config object, which lacks vocab_size.  The image-text auto class
        # selects Qwen3_5ForConditionalGeneration, whose text-only forward is
        # exactly the checkpoint's native path and accepts input_ids without
        # any image inputs.
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        ).to("cuda").eval()
        self.boundary_ids_by_name = {
            "canonical": self.tokenizer.encode(
                ANSWER_BOUNDARY, add_special_tokens=False
            ),
            "format_variant": self.tokenizer.encode(
                FORMAT_VARIANT_BOUNDARY, add_special_tokens=False
            ),
        }
        self.boundary_ids = self.boundary_ids_by_name["canonical"]
        close_id = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(close_id) != 1 or self.boundary_ids[0] != close_id[0]:
            raise RuntimeError("unexpected close/boundary tokenization")
        self._prompt_cache: dict[str, list[int]] = {}
        self._empty_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()

    def prompt_ids(self, item: Mapping[str, Any]) -> list[int]:
        task_id = str(item["id"])
        if task_id not in self._prompt_cache:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": str(item["prompt"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            expected = self.tokenizer.encode("<think>\n", add_special_tokens=False)
            if ids[-len(expected) :] != expected:
                raise RuntimeError(f"unexpected thinking prompt tail for {task_id}")
            self._prompt_cache[task_id] = ids
        return list(self._prompt_cache[task_id])

    def _score_target(
        self,
        *,
        prefix_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> list[float]:
        if not target_ids:
            raise ValueError("target may not be empty")
        full_ids = [*prefix_ids, *target_ids]
        if len(full_ids) > 16_384:
            raise ValueError(f"scoring sequence exceeds model context: {len(full_ids)}")
        # Token t at full position i is predicted by the logit at i-1.  Qwen's
        # logits_to_keep tensor avoids materializing a [sequence,vocab] tensor.
        positions = torch.arange(
            len(prefix_ids) - 1,
            len(full_ids) - 1,
            dtype=torch.long,
            device="cuda",
        )
        inputs = torch.tensor([full_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            logits = self.model(
                input_ids=inputs,
                use_cache=False,
                logits_to_keep=positions,
            ).logits[0].float()
            targets = torch.tensor(target_ids, dtype=torch.long, device="cuda")
            values = logits.log_softmax(dim=-1).gather(1, targets[:, None])[:, 0]
        output = [float(value) for value in values.cpu()]
        if not output or not all(math.isfinite(value) for value in output):
            raise RuntimeError("nonfinite teacher-forced likelihood")
        return output

    def _condition(
        self,
        item: Mapping[str, Any],
        trace_token_ids: Sequence[int],
        *,
        boundary: str,
    ) -> dict[str, Any]:
        prompt = self.prompt_ids(item)
        if boundary not in self.boundary_ids_by_name:
            raise ValueError(f"unknown answer boundary: {boundary}")
        boundary_ids = self.boundary_ids_by_name[boundary]
        answer_ids = self.tokenizer.encode(
            str(item["canonical_answer"]), add_special_tokens=False
        )
        target = [*boundary_ids, *answer_ids]
        values = self._score_target(
            prefix_ids=[*prompt, *[int(value) for value in trace_token_ids]],
            target_ids=target,
        )
        answer_values = values[len(boundary_ids) :]
        return {
            "boundary": boundary,
            "joint_ll_sum": sum(values),
            "answer_ll_sum": sum(answer_values),
            "close_boundary_ll_sum": sum(values[: len(boundary_ids)]),
            "joint_token_logprobs": values,
            "answer_token_logprobs": answer_values,
            "boundary_token_ids": boundary_ids,
            "answer_token_ids": answer_ids,
            "full_sequence_tokens": len(prompt) + len(trace_token_ids) + len(target),
        }

    def empty(
        self, item: Mapping[str, Any], *, boundary: str = "canonical"
    ) -> dict[str, Any]:
        task_id = str(item["id"])
        key = (task_id, boundary)
        if key not in self._empty_cache:
            self._empty_cache[key] = self._condition(item, [], boundary=boundary)
        return self._empty_cache[key]

    def score_trace(
        self,
        item: Mapping[str, Any],
        trace: Mapping[str, Any],
        *,
        boundary: str = "canonical",
    ) -> dict[str, Any]:
        condition = self._condition(
            item, trace["token_ids"], boundary=boundary
        )
        baseline = self.empty(item, boundary=boundary)
        answer_tokens = len(condition["answer_token_ids"])
        return {
            "trace_id": trace["trace_id"],
            "task_id": trace["task_id"],
            "family": trace["family"],
            "level": trace["level"],
            "source_kind": trace.get("source_kind", "independent"),
            "n_trace_tokens": trace["n_tokens"],
            "prior_logprob_mean": trace.get("prior_logprob_mean"),
            **condition,
            "empty_joint_ll_sum": baseline["joint_ll_sum"],
            "empty_answer_ll_sum": baseline["answer_ll_sum"],
            "joint_gain_sum": condition["joint_ll_sum"] - baseline["joint_ll_sum"],
            "answer_gain_sum": condition["answer_ll_sum"] - baseline["answer_ll_sum"],
            "joint_gain_per_answer_token": (
                condition["joint_ll_sum"] - baseline["joint_ll_sum"]
            ) / answer_tokens,
            "answer_gain_per_answer_token": (
                condition["answer_ll_sum"] - baseline["answer_ll_sum"]
            ) / answer_tokens,
        }
