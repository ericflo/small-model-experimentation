#!/usr/bin/env python3
"""Audit a frozen interpolation ladder against one non-coding locality block."""

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


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    return name, Path(raw_path)


def validate_qwen_checkpoint(path: Path) -> None:
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"locality model is not Qwen/Qwen3.5-4B: {path}")


def load_logits(
    model_path: Path,
    contexts: list[dict],
    max_context_tokens: int,
) -> tuple[list[torch.Tensor], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    values: list[torch.Tensor] = []
    lengths: list[int] = []
    with torch.inference_mode():
        for index, context in enumerate(contexts):
            prompt = tokenizer.apply_chat_template(
                context["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_context_tokens:
                raise SystemExit(f"locality context too long: {context['id']}={len(ids)}")
            tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                logits_to_keep=1,
                use_cache=False,
            ).logits[0, -1].float().cpu()
            values.append(logits)
            lengths.append(len(ids))
            if (index + 1) % 12 == 0:
                print(f"[locality] {model_path.name}: {index + 1}/{len(contexts)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values, lengths


def uncertainty(logits: torch.Tensor) -> tuple[float, float]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    surprisal = -log_probs
    entropy = (probabilities * surprisal).sum()
    varentropy = (probabilities * (surprisal - entropy).square()).sum()
    return float(entropy.item()), float(varentropy.item())


def compare(
    contexts: list[dict],
    before: list[torch.Tensor],
    after: list[torch.Tensor],
    lengths_before: list[int],
    lengths_after: list[int],
    *,
    drift_ceiling: float,
    entropy_delta_min: float,
    expected_contexts: int,
) -> dict:
    if lengths_before != lengths_after:
        raise SystemExit("locality tokenization differs between checkpoints")
    rows = []
    for context, left, right in zip(contexts, before, after):
        top = torch.topk(left, k=20).indices
        mask = torch.ones(left.numel(), dtype=torch.bool)
        mask[top] = False
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        drift = float(torch.median((right_centered - left_centered).abs()[mask]).item())
        entropy_before, varentropy_before = uncertainty(left)
        entropy_after, varentropy_after = uncertainty(right)
        rows.append({
            "id": context["id"],
            "median_non_target_centered_logit_drift": drift,
            "entropy_before": entropy_before,
            "entropy_after": entropy_after,
            "varentropy_before": varentropy_before,
            "varentropy_after": varentropy_after,
        })
    median_drift = statistics.median(
        row["median_non_target_centered_logit_drift"] for row in rows
    )
    mean_entropy_before = statistics.mean(row["entropy_before"] for row in rows)
    mean_entropy_after = statistics.mean(row["entropy_after"] for row in rows)
    mean_varentropy_before = statistics.mean(row["varentropy_before"] for row in rows)
    mean_varentropy_after = statistics.mean(row["varentropy_after"] for row in rows)
    scalars = [
        median_drift,
        mean_entropy_before,
        mean_entropy_after,
        mean_varentropy_before,
        mean_varentropy_after,
    ]
    checks = {
        "finite": all(math.isfinite(value) for value in scalars),
        "context_count": len(rows) == expected_contexts,
        "within_drift_ceiling": median_drift <= drift_ceiling,
        "entropy_retention": mean_entropy_after - mean_entropy_before >= entropy_delta_min,
    }
    return {
        "n_contexts": len(rows),
        "median_non_target_centered_logit_drift": median_drift,
        "mean_entropy_before": mean_entropy_before,
        "mean_entropy_after": mean_entropy_after,
        "mean_entropy_delta": mean_entropy_after - mean_entropy_before,
        "mean_varentropy_before": mean_varentropy_before,
        "mean_varentropy_after": mean_varentropy_after,
        "mean_varentropy_delta": mean_varentropy_after - mean_varentropy_before,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--eligible", action="append", default=[])
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drift-ceiling", type=float, default=0.15)
    parser.add_argument("--entropy-delta-min", type=float, default=-0.05)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--expected-contexts", type=int, default=48)
    parser.add_argument("--limit-contexts", type=int, default=None)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise SystemExit("candidate names must be unique")
    if not set(args.eligible).issubset(candidates):
        raise SystemExit("every eligible name must identify a candidate")
    validate_qwen_checkpoint(args.before_model)
    for path in candidates.values():
        validate_qwen_checkpoint(path)
    payload = json.loads(args.contexts.read_text())
    contexts = payload["contexts"]
    if args.limit_contexts is not None:
        contexts = contexts[: args.limit_contexts]
    before, lengths_before = load_logits(
        args.before_model, contexts, args.max_context_tokens
    )
    results = {}
    for name, path in candidates.items():
        after, lengths_after = load_logits(path, contexts, args.max_context_tokens)
        results[name] = {
            "model": str(path.resolve()),
            **compare(
                contexts,
                before,
                after,
                lengths_before,
                lengths_after,
                drift_ceiling=args.drift_ceiling,
                entropy_delta_min=args.entropy_delta_min,
                expected_contexts=args.expected_contexts,
            ),
        }
    passing_eligible = [
        name for name in args.eligible if results[name]["gate"]["passed"]
    ]
    result = {
        "schema_version": 1,
        "before_model": str(args.before_model.resolve()),
        "contexts": str(args.contexts.resolve()),
        "contexts_sha256": sha256_file(args.contexts),
        "drift_ceiling": args.drift_ceiling,
        "entropy_delta_min": args.entropy_delta_min,
        "eligible_candidates": args.eligible,
        "passing_eligible_candidates": passing_eligible,
        "candidates": results,
        "gate": {"passed": bool(passing_eligible)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        "contexts_sha256": result["contexts_sha256"],
        "passing_eligible_candidates": passing_eligible,
        "gate": result["gate"],
        "candidates": {
            name: {
                key: item[key]
                for key in (
                    "median_non_target_centered_logit_drift",
                    "mean_entropy_delta",
                    "mean_varentropy_delta",
                    "gate",
                )
            }
            for name, item in results.items()
        },
    }
    print(json.dumps(compact, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
