#!/usr/bin/env python3
"""Exploratory entropy/varentropy audit at plan and semantic tool pivots.

Uncertainty is diagnostic only: it never selects rows, weights loss, or labels
correctness.  The semantic probe places every tool behind the same
``{\"tool\":\"`` prefix so the measured token actually distinguishes INSPECT,
PATCH, VERIFY, and COMMIT rather than the common opening brace.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def distribution_metrics(
    logits: torch.Tensor, target: int, semantic_tokens: dict[str, int] | None = None
) -> dict:
    values = logits.float()
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    surprisal = -log_probs
    entropy = (probs * surprisal).sum()
    varentropy = (probs * (surprisal - entropy).square()).sum()
    target_logprob = log_probs[target]
    result = {
        "entropy_nats": float(entropy.item()),
        "varentropy_nats2": float(varentropy.item()),
        "target_logprob": float(target_logprob.item()),
        "target_rank": int((values > values[target]).sum().item()) + 1,
        "target_token_id": int(target),
    }
    if semantic_tokens:
        result["semantic_first_token_probability"] = {
            name: float(probs[token_id].item())
            for name, token_id in semantic_tokens.items()
        }
    return result


def seam_inputs(row: dict, tokenizer) -> dict[str, tuple[list[int], int, dict[str, int] | None]]:
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    think = row["think"].strip() + "\n</think>\n\n"
    answer = row["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    plan_ids = tokenizer(prompt + think, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think + answer, add_special_tokens=False)["input_ids"]
    if plan_ids[:len(prompt_ids)] != prompt_ids or full_ids[:len(plan_ids)] != plan_ids:
        raise AssertionError(f"seam token boundary merged for {row['id']}")
    operator_prefix = prompt + think + '{"tool":"'
    operator_prefix_ids = tokenizer(
        operator_prefix, add_special_tokens=False
    )["input_ids"]
    tool = str(row["target_action"]["tool"])
    operator_full_ids = tokenizer(
        operator_prefix + tool, add_special_tokens=False
    )["input_ids"]
    if operator_full_ids[:len(operator_prefix_ids)] != operator_prefix_ids:
        raise AssertionError(f"operator probe boundary merged for {row['id']}")
    semantic_tokens = {}
    for name in ("tree", "read", "search", "patch", "test", "submit"):
        ids = tokenizer(operator_prefix + name, add_special_tokens=False)["input_ids"]
        if ids[:len(operator_prefix_ids)] != operator_prefix_ids:
            raise AssertionError(f"semantic probe boundary merged for {name}")
        semantic_tokens[name] = ids[len(operator_prefix_ids)]
    return {
        "plan": (prompt_ids, plan_ids[len(prompt_ids)], None),
        "semantic_tool": (
            operator_prefix_ids,
            operator_full_ids[len(operator_prefix_ids)],
            semantic_tokens,
        ),
    }


def evaluate_model(model_path: Path, rows: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            item = {
                "id": row["id"],
                "task_id": row["task_id"],
                "family": row["family"],
                "transition": row["transition"],
                "operator": row["operator"],
                "seams": {},
            }
            for seam, (ids, target, semantic_tokens) in seam_inputs(row, tokenizer).items():
                tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
                logits = model(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    logits_to_keep=1,
                    use_cache=False,
                ).logits[0, -1]
                item["seams"][seam] = distribution_metrics(
                    logits, target, semantic_tokens
                )
            results.append(item)
            if (index + 1) % 14 == 0:
                print(f"[uncertainty] {model_path.name}: {index + 1}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize(rows: list[dict]) -> dict:
    result = {}
    by_transition = defaultdict(list)
    for row in rows:
        by_transition[row["transition"]].append(row)
    for transition, members in sorted(by_transition.items()):
        result[transition] = {}
        for seam in ("plan", "semantic_tool"):
            result[transition][seam] = {
                key: statistics.mean(row["seams"][seam][key] for row in members)
                for key in ("entropy_nats", "varentropy_nats2", "target_logprob", "target_rank")
            }
    return result


def semantic_bucket(row: dict) -> str:
    tool = str(row["target_action"]["tool"])
    if tool == "search" and row["transition"] == "ambiguous_source_to_inspect_evidence":
        return "search_discriminator"
    if tool in ("tree", "read", "search"):
        return "read_other"
    if tool == "patch":
        return "patch"
    return "other"


def _paired_summary(
    ids: list[str], before: dict[str, dict], after: dict[str, dict], seam: str
) -> dict:
    keys = ("entropy_nats", "varentropy_nats2", "target_logprob", "target_rank")
    left = {
        key: statistics.mean(before[row_id]["seams"][seam][key] for row_id in ids)
        for key in keys
    }
    right = {
        key: statistics.mean(after[row_id]["seams"][seam][key] for row_id in ids)
        for key in keys
    }
    return {
        "n": len(ids),
        "before": left,
        "after": right,
        "after_minus_before": {key: right[key] - left[key] for key in keys},
    }


def stratified_summary(
    selected: list[dict], before_rows: list[dict], after_rows: list[dict], strata: int
) -> dict:
    """Stratify only after fixed-row evaluation, using baseline uncertainty."""
    if strata < 2:
        raise ValueError("uncertainty strata must be at least two")
    before = {row["id"]: row for row in before_rows}
    after = {row["id"]: row for row in after_rows}
    row_by_id = {row["id"]: row for row in selected}
    if set(before) != set(after) or set(before) != set(row_by_id):
        raise AssertionError("uncertainty row identities differ across checkpoints")
    quartiles = {}
    for seam in ("plan", "semantic_tool"):
        ordered = sorted(
            before,
            key=lambda row_id: (
                before[row_id]["seams"][seam]["entropy_nats"], row_id
            ),
        )
        groups = {index: [] for index in range(strata)}
        for rank, row_id in enumerate(ordered):
            groups[min(strata - 1, rank * strata // len(ordered))].append(row_id)
        quartiles[seam] = {
            f"q{index + 1}": _paired_summary(ids, before, after, seam)
            for index, ids in groups.items()
        }
    bucket_ids = defaultdict(list)
    for row_id, row in row_by_id.items():
        bucket_ids[semantic_bucket(row)].append(row_id)
    expected_buckets = {"search_discriminator", "read_other", "patch", "other"}
    if set(bucket_ids) != expected_buckets or any(not ids for ids in bucket_ids.values()):
        raise AssertionError(f"uncertainty semantic bucket coverage failed: {bucket_ids}")
    buckets = {
        name: {
            seam: _paired_summary(sorted(ids), before, after, seam)
            for seam in ("plan", "semantic_tool")
        }
        for name, ids in sorted(bucket_ids.items())
    }
    return {
        "stratification_rule": (
            "fixed sampled rows; deterministic rank by start-checkpoint entropy then row id; "
            "strata never select rows or affect loss"
        ),
        "baseline_entropy_strata": quartiles,
        "semantic_action_buckets": buckets,
    }


def validate_registered_sample_geometry(
    rows_per_transition: int, strata: int, cfg: dict
) -> None:
    if rows_per_transition != int(cfg["uncertainty"]["rows_per_transition"]):
        raise SystemExit(
            "uncertainty rows-per-transition differs from the frozen config"
        )
    if strata != int(cfg["uncertainty"]["strata"]):
        raise SystemExit("uncertainty strata differs from the frozen config")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--design-lock", type=Path, required=True)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--rows-per-transition", type=int, default=6)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        harness.validate_model_execution_lock(
            EXP, args.design_lock, "scripts/audit_transition_uncertainty.py"
        )
    except ValueError as exc:
        raise SystemExit(f"model execution is not design-locked: {exc}") from exc
    try:
        harness.validate_canonical_config_path(EXP, args.config)
    except ValueError as exc:
        raise SystemExit(f"uncertainty config is not frozen: {exc}") from exc
    cfg = yaml.safe_load(args.config.read_text())
    root = EXP.parents[1]
    validate_registered_sample_geometry(
        args.rows_per_transition, args.strata, cfg
    )
    artifact_root = Path(cfg["artifacts"]["root"])
    artifact_root = (
        artifact_root if artifact_root.is_absolute() else root / artifact_root
    ).resolve()
    registered_bank = artifact_root / "bank" / "evidence_binding.jsonl"
    bank_receipt_path = artifact_root / "bank" / "receipt.json"
    if args.bank.resolve() != registered_bank or not bank_receipt_path.is_file():
        raise SystemExit("uncertainty bank is not the frozen evidence-binding bank")
    bank_receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    if (
        bank_receipt.get("bank_sha256", {}).get("evidence_binding")
        != sha256_file(registered_bank)
    ):
        raise SystemExit("uncertainty bank differs from its registered receipt")
    before_model = args.before_model.resolve()
    after_model = args.after_model.resolve()
    checkpoint_provenance = {}
    for label, checkpoint_role, model_path in (
        ("before", "start", before_model),
        ("after", "evidence_binding", after_model),
    ):
        try:
            checkpoint = harness.validate_registered_checkpoint(
                EXP, model_path, cfg, args.design_lock, checkpoint_role
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid {label} uncertainty checkpoint: {exc}") from exc
        checkpoint_provenance[label] = checkpoint
    all_rows = [json.loads(line) for line in args.bank.read_text().splitlines() if line.strip()]
    by_transition = defaultdict(list)
    for row in sorted(all_rows, key=lambda item: item["id"]):
        by_transition[row["transition"]].append(row)
    selected = [
        row
        for transition in sorted(by_transition)
        for row in by_transition[transition][:args.rows_per_transition]
    ]
    if any(len(rows) < args.rows_per_transition for rows in by_transition.values()):
        raise SystemExit("bank has too few rows for the registered uncertainty sample")
    before_rows = evaluate_model(before_model, selected)
    after_rows = evaluate_model(after_model, selected)
    before = summarize(before_rows)
    after = summarize(after_rows)
    deltas = {}
    for transition in before:
        deltas[transition] = {}
        for seam in before[transition]:
            deltas[transition][seam] = {
                key: after[transition][seam][key] - before[transition][seam][key]
                for key in before[transition][seam]
            }
    stratified = stratified_summary(selected, before_rows, after_rows, args.strata)
    result = {
        "schema_version": 1,
        "status": "exploratory_non_gating",
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(args.config),
        "design_lock_sha256": sha256_file(args.design_lock),
        "before_model": str(before_model),
        "after_model": str(after_model),
        "bank": str(args.bank.resolve()),
        "bank_sha256": sha256_file(args.bank),
        "checkpoint_provenance": checkpoint_provenance,
        "rows_per_transition": args.rows_per_transition,
        "rows": len(selected),
        "strata": args.strata,
        "before": before,
        "after": after,
        "after_minus_before": deltas,
        **stratified,
        "before_rows": before_rows,
        "after_rows": after_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("before_rows", "after_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
