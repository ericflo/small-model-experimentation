#!/usr/bin/env python3
"""Compare frozen-anchor and candidate uncertainty on fresh non-coding contexts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from harness import (  # noqa: E402
    tokenizer_provenance,
    validate_model_execution_lock,
    validate_registered_checkpoint,
    validate_registered_tokenizer_provenance,
)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_logits(
    model_path: Path,
    contexts: list[dict],
    max_context_tokens: int,
) -> tuple[list[torch.Tensor], list[str], list[list[int]], dict[str, object]]:
    try:
        provenance = tokenizer_provenance(model_path)
        merge_receipt = json.loads(
            (model_path / "merge_receipt.json").read_text(encoding="utf-8")
        )
        validate_registered_tokenizer_provenance(
            model_path, merge_receipt, allow_absent=True
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid locality tokenizer at {model_path}: {exc}") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    values = []
    rendered_prompts = []
    token_ids = []
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
            rendered_prompts.append(prompt)
            token_ids.append(ids)
            if (index + 1) % 12 == 0:
                print(f"[locality] {model_path.name}: {index + 1}/{len(contexts)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values, rendered_prompts, token_ids, provenance


def context_value_sha256(
    contexts: list[dict], values: list[object], value_key: str
) -> str:
    payload = [
        {"id": context["id"], value_key: value}
        for context, value in zip(contexts, values, strict=True)
    ]
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def assert_compatible_tokenizations(
    tokenizer_before: dict[str, object],
    tokenizer_after: dict[str, object],
    prompts_before: list[str],
    prompts_after: list[str],
    ids_before: list[list[int]],
    ids_after: list[list[int]],
) -> None:
    if (
        tokenizer_before.get("tokenizer_compatibility_sha256")
        != tokenizer_after.get("tokenizer_compatibility_sha256")
    ):
        raise SystemExit("locality tokenizer compatibility differs between checkpoints")
    if prompts_before != prompts_after:
        raise SystemExit("locality rendered prompts differ between merged checkpoints")
    if ids_before != ids_after:
        raise SystemExit("locality token IDs differ between merged checkpoints")


def uncertainty(logits: torch.Tensor) -> tuple[float, float]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probabilities = log_probs.exp()
    surprisal = -log_probs
    entropy = (probabilities * surprisal).sum()
    varentropy = (probabilities * (surprisal - entropy).square()).sum()
    return float(entropy.item()), float(varentropy.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-lock", type=Path, required=True)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ceiling", type=float, default=0.10)
    parser.add_argument("--entropy-delta-min", type=float, default=-0.05)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    args = parser.parse_args()
    try:
        validate_model_execution_lock(
            EXP, args.design_lock, "scripts/audit_locality.py"
        )
    except ValueError as exc:
        raise SystemExit(f"model execution is not design-locked: {exc}") from exc
    cfg = yaml.safe_load(
        (EXP / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    repository = EXP.parents[1]

    def registered_path(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else repository / path).resolve()

    expected_contexts = registered_path(cfg["locality"]["contexts"])
    if args.contexts.resolve() != expected_contexts:
        raise SystemExit("locality contexts are not the frozen context bank")
    if (
        not math.isclose(
            args.ceiling,
            float(cfg["locality"]["median_non_target_logit_drift_max"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            args.entropy_delta_min,
            float(cfg["locality"]["mean_entropy_delta_min"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or args.max_context_tokens != int(cfg["locality"]["max_context_tokens"])
    ):
        raise SystemExit("locality thresholds differ from the frozen config")
    artifact_root = registered_path(cfg["artifacts"]["root"])
    registered_roles = {
        registered_path(cfg["model"]["locality_anchor"]): "anchor",
        registered_path(cfg["model"]["start_checkpoint"]): "start",
        (artifact_root / "merged" / "evidence_binding").resolve(): "evidence_binding",
        (artifact_root / "merged" / "explicit_redundant").resolve(): "explicit_redundant",
        (artifact_root / "merged" / "shuffled_binding").resolve(): "shuffled_binding",
    }
    before_model = args.before_model.resolve()
    after_model = args.after_model.resolve()
    before_role = registered_roles.get(before_model)
    after_role = registered_roles.get(after_model)
    allowed_pairs = {
        ("anchor", "start"),
        ("anchor", "evidence_binding"),
        ("start", "evidence_binding"),
        ("anchor", "explicit_redundant"),
        ("anchor", "shuffled_binding"),
    }
    if (before_role, after_role) not in allowed_pairs:
        raise SystemExit(
            f"locality checkpoint-role pair is not registered: {before_role}/{after_role}"
        )
    try:
        validate_registered_checkpoint(
            EXP, before_model, cfg, args.design_lock, str(before_role)
        )
        validate_registered_checkpoint(
            EXP, after_model, cfg, args.design_lock, str(after_role)
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"locality checkpoint is not registered: {exc}") from exc
    payload = json.loads(args.contexts.read_text())
    contexts = payload["contexts"]
    if len(contexts) != int(cfg["locality"]["count"]):
        raise SystemExit("locality context count differs from the frozen config")
    before, prompts_before, ids_before, tokenizer_before = load_logits(
        before_model, contexts, args.max_context_tokens
    )
    after, prompts_after, ids_after, tokenizer_after = load_logits(
        after_model, contexts, args.max_context_tokens
    )
    assert_compatible_tokenizations(
        tokenizer_before,
        tokenizer_after,
        prompts_before,
        prompts_after,
        ids_before,
        ids_after,
    )
    prompts_before_sha256 = context_value_sha256(
        contexts, prompts_before, "rendered_prompt"
    )
    prompts_after_sha256 = context_value_sha256(
        contexts, prompts_after, "rendered_prompt"
    )
    ids_before_sha256 = context_value_sha256(contexts, ids_before, "token_ids")
    ids_after_sha256 = context_value_sha256(contexts, ids_after, "token_ids")
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
    entropy_before = statistics.mean(row["entropy_before"] for row in rows)
    entropy_after = statistics.mean(row["entropy_after"] for row in rows)
    varentropy_before = statistics.mean(row["varentropy_before"] for row in rows)
    varentropy_after = statistics.mean(row["varentropy_after"] for row in rows)
    finite = all(
        math.isfinite(value)
        for row in rows
        for value in (
            row["median_non_target_centered_logit_drift"],
            row["entropy_before"], row["entropy_after"],
            row["varentropy_before"], row["varentropy_after"],
        )
    )
    result = {
        "schema_version": 1,
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "before_model": str(before_model),
        "after_model": str(after_model),
        "before_model_weight_sha256": sha256_file(
            before_model / "model.safetensors"
        ),
        "after_model_weight_sha256": sha256_file(
            after_model / "model.safetensors"
        ),
        "before_model_config_sha256": sha256_file(before_model / "config.json"),
        "after_model_config_sha256": sha256_file(after_model / "config.json"),
        "before_model_generation_config_sha256": sha256_file(
            before_model / "generation_config.json"
        ),
        "after_model_generation_config_sha256": sha256_file(
            after_model / "generation_config.json"
        ),
        "before_merge_receipt_sha256": sha256_file(
            before_model / "merge_receipt.json"
        ),
        "after_merge_receipt_sha256": sha256_file(
            after_model / "merge_receipt.json"
        ),
        "before_tokenizer_files": tokenizer_before["tokenizer_files"],
        "before_tokenizer_manifest_sha256": tokenizer_before[
            "tokenizer_manifest_sha256"
        ],
        "before_tokenizer_compatibility_sha256": tokenizer_before[
            "tokenizer_compatibility_sha256"
        ],
        "after_tokenizer_files": tokenizer_after["tokenizer_files"],
        "after_tokenizer_manifest_sha256": tokenizer_after[
            "tokenizer_manifest_sha256"
        ],
        "after_tokenizer_compatibility_sha256": tokenizer_after[
            "tokenizer_compatibility_sha256"
        ],
        "before_rendered_prompts_sha256": prompts_before_sha256,
        "after_rendered_prompts_sha256": prompts_after_sha256,
        "rendered_prompts_equal": True,
        "before_tokenized_contexts_sha256": ids_before_sha256,
        "after_tokenized_contexts_sha256": ids_after_sha256,
        "tokenized_context_ids_equal": True,
        "contexts": str(args.contexts.resolve()),
        "contexts_sha256": sha256_file(args.contexts),
        "n_contexts": len(rows),
        "median_non_target_centered_logit_drift": median_drift,
        "mean_entropy_before": entropy_before,
        "mean_entropy_after": entropy_after,
        "mean_entropy_delta": entropy_after - entropy_before,
        "mean_varentropy_before": varentropy_before,
        "mean_varentropy_after": varentropy_after,
        "mean_varentropy_delta": varentropy_after - varentropy_before,
        "ceiling": args.ceiling,
        "entropy_delta_min": args.entropy_delta_min,
        "max_context_tokens": args.max_context_tokens,
        "checks": {
            "finite": finite,
            "context_count": len(rows) == 48,
            "within_drift_ceiling": median_drift <= args.ceiling,
            "entropy_retained": entropy_after - entropy_before >= args.entropy_delta_min,
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
