#!/usr/bin/env python3
"""Audit the registered five-update MOPD locality pilot exactly batch-of-one."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402
from mopd_loss import sparse_teacher_topk_reverse_kl  # noqa: E402


def _entropy(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(-(log_probs.exp() * log_probs).sum().item())


def _row_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> dict[str, float]:
    if before.ndim != 1 or after.shape != before.shape:
        raise ValueError("locality logits must be equal one-dimensional vocabularies")
    target = teacher_indices.to(dtype=torch.long)
    if target.ndim != 1 or teacher_log_probs.shape != target.shape:
        raise ValueError("locality teacher target shape mismatch")
    non_target = torch.ones(before.numel(), dtype=torch.bool)
    non_target[target] = False
    if not bool(non_target.any()):
        raise ValueError("teacher top-k exhausts vocabulary")
    # Remove the softmax-invariant global offset before measuring collateral
    # movement outside the selected teacher's top-50 support.
    before_centered = before.float() - before.float().mean()
    after_centered = after.float() - after.float().mean()
    drift = float(
        torch.median((after_centered - before_centered).abs()[non_target]).item()
    )
    return {
        "median_centered_non_target_logit_drift": drift,
        "entropy_before": _entropy(before),
        "entropy_after": _entropy(after),
        "target_loss_before": float(
            sparse_teacher_topk_reverse_kl(
                before.unsqueeze(0), target.unsqueeze(0),
                teacher_log_probs.float().unsqueeze(0), reduction="mean",
            ).item()
        ),
        "target_loss_after": float(
            sparse_teacher_topk_reverse_kl(
                after.unsqueeze(0), target.unsqueeze(0),
                teacher_log_probs.float().unsqueeze(0), reduction="mean",
            ).item()
        ),
    }


def _load_model(path: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    return model


def _probe_logits(model, probes: list[dict]) -> list[torch.Tensor]:
    values = []
    with torch.inference_mode():
        for index, probe in enumerate(probes):
            # The last input position predicts completion[position].
            ids = probe["prompt_ids"] + probe["completion_ids"][: probe["position"]]
            tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
            logits = model(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                logits_to_keep=1,
                use_cache=False,
            ).logits[0, -1].float().cpu()
            values.append(logits)
            if (index + 1) % 5 == 0:
                print(f"[locality] exact logits {index + 1}/{len(probes)}", flush=True)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "locality_pilot.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    cfg = config["locality"]

    cache_receipt_path = args.target_cache.with_suffix(
        args.target_cache.suffix + ".receipt.json"
    )
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    if cache_receipt.get("cache_sha256") != sha256_file(args.target_cache):
        raise SystemExit("locality target-cache checksum mismatch")
    training = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    expected_updates = int(cfg["updates"])
    if int(training.get("requested_updates", -1)) != expected_updates:
        raise SystemExit("locality trainer did not request the frozen update count")
    if int(training.get("completed_updates", -1)) != expected_updates:
        raise SystemExit("locality trainer did not complete the frozen update count")
    if training.get("target_cache_sha256") != sha256_file(args.target_cache):
        raise SystemExit("locality trainer/cache provenance mismatch")
    if Path(training.get("base_model", "")).resolve() != args.before_model.resolve():
        raise SystemExit("locality trainer used the wrong pre-update checkpoint")
    unit_ledger = list(training.get("unit_ledger") or [])
    expected_units = expected_updates * int(config["mopd"]["grad_accum"])
    unit_ids = [str(row["sample_id"]) for row in unit_ledger]
    if (
        len(unit_ledger) != expected_units
        or len(unit_ids) != len(set(unit_ids))
        or not training.get("consume_once_verified")
    ):
        raise SystemExit("locality unit ledger violates consume-once geometry")

    before_merge = json.loads(
        (args.before_model / "merge_receipt.json").read_text(encoding="utf-8")
    )
    after_merge = json.loads(
        (args.after_model / "merge_receipt.json").read_text(encoding="utf-8")
    )
    if Path(after_merge.get("base_model", "")).resolve() != args.before_model.resolve():
        raise SystemExit("locality merged checkpoint has the wrong base")
    if Path(after_merge.get("adapter", "")).resolve() != args.training_receipt.parent.resolve():
        raise SystemExit("locality merged checkpoint has the wrong adapter")
    if after_merge.get("adapter_weights_sha256") != sha256_file(
        args.training_receipt.parent / "adapter_model.safetensors"
    ):
        raise SystemExit("locality merged adapter checksum mismatch")

    cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    samples = {str(sample["id"]): sample for sample in cache["samples"]}
    if set(unit_ids) - set(samples):
        raise SystemExit("locality unit IDs are absent from the exact target cache")
    probes = []
    for ledger_row in unit_ledger:
        sample = samples[str(ledger_row["sample_id"])]
        target_name = str(ledger_row["target"])
        if target_name not in sample["targets"]:
            raise SystemExit(f"missing locality target {target_name} for {sample['id']}")
        positions = sample["positions"].to(dtype=torch.long).tolist()
        position_index = len(positions) // 2
        position = int(positions[position_index])
        target = sample["targets"][target_name]
        probes.append(
            {
                "id": str(sample["id"]),
                "role": str(sample["meta"]["role"]),
                "target": target_name,
                "prompt_ids": sample["prompt_ids"].to(dtype=torch.long).tolist(),
                "completion_ids": sample["completion_ids"].to(dtype=torch.long).tolist(),
                "position": position,
                "teacher_indices": target["indices"][position_index].to(dtype=torch.long),
                "teacher_log_probs": target["log_probs"][position_index].float(),
            }
        )

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
        metrics = _row_metrics(
            before, after, probe["teacher_indices"], probe["teacher_log_probs"]
        )
        rows.append(
            {
                "id": probe["id"],
                "role": probe["role"],
                "target": probe["target"],
                "position": probe["position"],
                **metrics,
            }
        )
    numeric_keys = (
        "median_centered_non_target_logit_drift",
        "entropy_before",
        "entropy_after",
        "target_loss_before",
        "target_loss_after",
    )
    finite = bool(rows) and all(
        math.isfinite(float(row[key])) for row in rows for key in numeric_keys
    )
    mean_row_median_drift = statistics.mean(
        row["median_centered_non_target_logit_drift"] for row in rows
    )
    entropy_before = statistics.mean(row["entropy_before"] for row in rows)
    entropy_after = statistics.mean(row["entropy_after"] for row in rows)
    relative_entropy_drop = (entropy_before - entropy_after) / max(entropy_before, 1e-12)
    target_loss_before = statistics.mean(row["target_loss_before"] for row in rows)
    target_loss_after = statistics.mean(row["target_loss_after"] for row in rows)
    target_counts = {
        target: sum(row["target"] == target for row in rows)
        for target in ("quick", "deep", "soup")
    }
    checks = {
        "finite": finite,
        "five_updates": int(training["completed_updates"]) == expected_updates,
        "consume_once": len(unit_ids) == len(set(unit_ids)) == expected_units,
        "registered_pilot_mixture": target_counts == {"quick": 8, "deep": 7, "soup": 5},
        "training_round_safety_gate": bool(training.get("round_gate", {}).get("passed")),
        "mean_row_median_non_target_drift_within_ceiling": (
            mean_row_median_drift
            <= float(cfg["maximum_median_centered_non_target_logit_drift"])
        ),
        "mean_entropy_relative_drop_within_ceiling": (
            relative_entropy_drop <= float(cfg["maximum_entropy_relative_drop"])
        ),
        "mean_corrected_topk_loss_within_ceiling": (
            target_loss_after <= float(cfg["maximum_mean_corrected_topk_loss"])
            and float(training["mean_corrected_topk_loss"])
            <= float(cfg["maximum_mean_corrected_topk_loss"])
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 2,
        "stage": "five_update_exact_logit_locality_pilot",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "before_model": str(args.before_model.resolve()),
        "before_merge_receipt_sha256": sha256_file(args.before_model / "merge_receipt.json"),
        "before_merge_method": before_merge.get("method"),
        "after_model": str(args.after_model.resolve()),
        "after_merge_receipt_sha256": sha256_file(args.after_model / "merge_receipt.json"),
        "target_cache": str(args.target_cache.resolve()),
        "target_cache_sha256": sha256_file(args.target_cache),
        "training_receipt": str(args.training_receipt.resolve()),
        "training_receipt_sha256": sha256_file(args.training_receipt),
        "probe_rows": len(rows),
        "target_counts": target_counts,
        "mean_row_median_centered_non_target_logit_drift": mean_row_median_drift,
        "mean_entropy_before": entropy_before,
        "mean_entropy_after": entropy_after,
        "mean_entropy_relative_drop": relative_entropy_drop,
        "mean_target_loss_before": target_loss_before,
        "mean_target_loss_after": target_loss_after,
        "target_loss_delta": target_loss_after - target_loss_before,
        "checks": checks,
        "rows": rows,
        "gate": {"passed": passed},
        "downstream_authorization": "four_round_mopd" if passed else "stop_before_full_mopd",
    }
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
