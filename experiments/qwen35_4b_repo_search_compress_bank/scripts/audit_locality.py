#!/usr/bin/env python3
"""Compare apex and compact next-token logits on frozen non-coding contexts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_logits(model_path: Path, contexts: list[dict], max_context_tokens: int) -> tuple[list[torch.Tensor], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    values = []
    lengths = []
    with torch.inference_mode():
        for index, context in enumerate(contexts):
            prompt = tokenizer.apply_chat_template(
                context["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_context_tokens:
                raise SystemExit(f"locality context too long: {context['id']}={len(ids)}")
            tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=tensor, attention_mask=torch.ones_like(tensor),
                logits_to_keep=1, use_cache=False,
            ).logits[0, -1].float().cpu()
            values.append(logits)
            lengths.append(len(ids))
            if (index + 1) % 12 == 0:
                print(f"[locality] {model_path.name}: {index + 1}/{len(contexts)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values, lengths


def entropy(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(-(log_probs.exp() * log_probs).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ceiling", type=float, default=0.15)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    args = parser.parse_args()
    payload = json.loads(args.contexts.read_text())
    contexts = payload["contexts"]
    before, lengths_before = load_logits(args.before_model, contexts, args.max_context_tokens)
    after, lengths_after = load_logits(args.after_model, contexts, args.max_context_tokens)
    if lengths_before != lengths_after:
        raise SystemExit("locality tokenization differs between merged checkpoints")
    rows = []
    for context, left, right in zip(contexts, before, after):
        # Exclude the apex policy's top-20 intended continuations and center raw
        # logits to remove the softmax-invariant additive degree of freedom.
        top = torch.topk(left, k=20).indices
        mask = torch.ones(left.numel(), dtype=torch.bool)
        mask[top] = False
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        drift = float(torch.median((right_centered - left_centered).abs()[mask]).item())
        rows.append({
            "id": context["id"],
            "median_non_target_centered_logit_drift": drift,
            "entropy_before": entropy(left),
            "entropy_after": entropy(right),
        })
    median_drift = statistics.median(
        row["median_non_target_centered_logit_drift"] for row in rows
    )
    entropy_before = statistics.mean(row["entropy_before"] for row in rows)
    entropy_after = statistics.mean(row["entropy_after"] for row in rows)
    finite = all(
        math.isfinite(value)
        for row in rows
        for value in (
            row["median_non_target_centered_logit_drift"],
            row["entropy_before"], row["entropy_after"],
        )
    )
    result = {
        "schema_version": 1,
        "before_model": str(args.before_model.resolve()),
        "after_model": str(args.after_model.resolve()),
        "contexts": str(args.contexts.resolve()),
        "contexts_sha256": sha256_file(args.contexts),
        "n_contexts": len(rows),
        "median_non_target_centered_logit_drift": median_drift,
        "mean_entropy_before": entropy_before,
        "mean_entropy_after": entropy_after,
        "mean_entropy_delta": entropy_after - entropy_before,
        "ceiling": args.ceiling,
        "checks": {
            "finite": finite,
            "context_count": len(rows) == 48,
            "within_drift_ceiling": median_drift <= args.ceiling,
        },
        "rows": rows,
    }
    result["gate"] = {"passed": all(result["checks"].values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
