#!/usr/bin/env python3
"""Audit a five-update MOPD pilot for tail-logit drift and entropy collapse."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402
from mopd_loss import sparse_teacher_topk_reverse_kl  # noqa: E402


def _load_model(path: Path):
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    return model


def _probe_logits(model, probes: list[dict]) -> list[torch.Tensor]:
    values = []
    with torch.inference_mode():
        for index, probe in enumerate(probes):
            # To obtain the distribution that predicts completion[position],
            # feed the exact prefix ending immediately before that token.
            ids = probe["prompt_ids"] + probe["completion_ids"][: probe["position"]]
            tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=tensor, attention_mask=torch.ones_like(tensor),
                logits_to_keep=1, use_cache=False,
            ).logits[0, -1].float().cpu()
            values.append(logits)
            if (index + 1) % 10 == 0:
                print(f"[locality] logits {index + 1}/{len(probes)}", flush=True)
    return values


def _entropy(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(-(log_probs.exp() * log_probs).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "locality_pilot.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    cache_receipt_path = args.teacher_cache.with_suffix(
        args.teacher_cache.suffix + ".receipt.json"
    )
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    if cache_receipt.get("cache_sha256") != sha256_file(args.teacher_cache):
        raise SystemExit("locality teacher-cache checksum mismatch")
    training = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    if int(training.get("optimizer_steps", -1)) != 5:
        raise SystemExit("locality pilot must contain exactly five optimizer updates")
    if not training.get("consume_once_verified"):
        raise SystemExit("locality training receipt did not verify consume-once sampling")
    payload = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    samples = {str(sample["id"]): sample for sample in payload["samples"]}
    selected_ids = [str(row["sample_id"]) for row in training.get("logs", [])]
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) - set(samples):
        raise SystemExit("locality training/cache sample identity mismatch")
    probes = []
    for sample_id in selected_ids:
        sample = samples[sample_id]
        active = torch.nonzero(sample["policy_mask"], as_tuple=False).flatten().tolist()
        positions = active[-256:]
        if not positions:
            raise SystemExit(f"locality sample has no active positions: {sample_id}")
        position = positions[len(positions) // 2]
        probes.append({
            "id": sample_id,
            "stratum": str(sample["meta"]["stratum"]),
            "prompt_ids": sample["prompt_ids"].to(dtype=torch.long).tolist(),
            "completion_ids": sample["completion_ids"].to(dtype=torch.long).tolist(),
            "position": int(position),
            "teacher_indices": sample["teacher_indices"][position].to(dtype=torch.long),
            "teacher_log_probs": sample["teacher_log_probs"][position].float(),
        })

    before_model = _load_model(args.before_model)
    before_logits = _probe_logits(before_model, probes)
    del before_model
    gc.collect()
    torch.cuda.empty_cache()
    after_model = _load_model(args.after_model)
    after_logits = _probe_logits(after_model, probes)
    del after_model
    gc.collect()
    torch.cuda.empty_cache()

    rows = []
    for probe, before, after in zip(probes, before_logits, after_logits):
        target = probe["teacher_indices"]
        non_target = torch.ones(before.numel(), dtype=torch.bool)
        non_target[target] = False
        # Center raw logits to remove the softmax-invariant additive degree of
        # freedom before measuring changes outside the teacher's top-k set.
        before_centered = before - before.mean()
        after_centered = after - after.mean()
        drift = float(
            torch.median((after_centered - before_centered).abs()[non_target]).item()
        )
        before_entropy = _entropy(before)
        after_entropy = _entropy(after)
        before_loss = float(sparse_teacher_topk_reverse_kl(
            before.unsqueeze(0), target.unsqueeze(0),
            probe["teacher_log_probs"].unsqueeze(0), reduction="mean",
        ).item())
        after_loss = float(sparse_teacher_topk_reverse_kl(
            after.unsqueeze(0), target.unsqueeze(0),
            probe["teacher_log_probs"].unsqueeze(0), reduction="mean",
        ).item())
        rows.append({
            "id": probe["id"], "stratum": probe["stratum"],
            "position": probe["position"],
            "median_non_target_centered_logit_drift": drift,
            "entropy_before": before_entropy, "entropy_after": after_entropy,
            "target_loss_before": before_loss, "target_loss_after": after_loss,
        })
    finite = all(
        math.isfinite(value)
        for row in rows
        for value in (
            row["median_non_target_centered_logit_drift"], row["entropy_before"],
            row["entropy_after"], row["target_loss_before"], row["target_loss_after"],
        )
    )
    median_drift = statistics.median(
        row["median_non_target_centered_logit_drift"] for row in rows
    )
    entropy_before = statistics.mean(row["entropy_before"] for row in rows)
    entropy_after = statistics.mean(row["entropy_after"] for row in rows)
    relative_entropy_drop = (entropy_before - entropy_after) / max(entropy_before, 1e-12)
    target_loss_before = statistics.mean(row["target_loss_before"] for row in rows)
    target_loss_after = statistics.mean(row["target_loss_after"] for row in rows)
    checks = {
        "finite": finite,
        "five_updates": int(training["optimizer_steps"]) == 5,
        "consume_once": len(selected_ids) == len(set(selected_ids)),
        "median_non_target_logit_drift_within_ceiling": median_drift <= float(
            config["teacher_audit"]["maximum_median_non_target_logit_drift"]
        ),
        "mean_entropy_relative_drop_within_ceiling": relative_entropy_drop <= float(
            config["teacher_audit"]["maximum_entropy_relative_drop"]
        ),
    }
    result = {
        "stage": "five_update_locality_pilot", "config": str(config_path),
        "before_model": str(args.before_model.resolve()),
        "after_model": str(args.after_model.resolve()),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": sha256_file(args.teacher_cache),
        "training_receipt": str(args.training_receipt.resolve()),
        "training_receipt_sha256": sha256_file(args.training_receipt),
        "probes": len(rows),
        "median_non_target_centered_logit_drift": median_drift,
        "mean_entropy_before": entropy_before, "mean_entropy_after": entropy_after,
        "mean_entropy_relative_drop": relative_entropy_drop,
        "mean_target_loss_before": target_loss_before,
        "mean_target_loss_after": target_loss_after,
        "target_loss_delta": target_loss_after - target_loss_before,
        "checks": checks, "rows": rows,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "four_round_mopd" if all(checks.values()) else "stop_before_full_mopd"
        ),
    }
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
