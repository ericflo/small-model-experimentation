#!/usr/bin/env python3
"""Audit targeted and collateral logit movement on the frozen training contexts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

EXP = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def final_logits(model, row: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = torch.tensor(row["context_ids"], dtype=torch.long, device=device).unsqueeze(0)
    hidden = model.get_base_model().model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    return model.get_base_model().lm_head(hidden[:, -1]).float()[0]


def mean(rows: list[dict], key: str) -> float | None:
    return float(np.mean([r[key] for r in rows])) if rows else None


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "objective_hit_rate": mean(rows, "objective_hit"),
        "chosen_gain_mean": mean(rows, "chosen_gain"),
        "rejected_drift_mean": mean(rows, "rejected_drift"),
        "pair_gap_shift_mean": mean(rows, "pair_gap_shift"),
        "median_abs_nontarget_drift_mean": mean(rows, "median_abs_nontarget_drift"),
        "p95_abs_nontarget_drift_mean": mean(rows, "p95_abs_nontarget_drift"),
        "entropy_change_mean": mean(rows, "entropy_change"),
        "varentropy_change_mean": mean(rows, "varentropy_change"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["demote", "uplift", "uplift_shuffled"], required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    row_file = "rows_shuffled_selected.jsonl.gz" if args.arm == "uplift_shuffled" \
        else "rows_real_selected.jsonl.gz"
    rows = load_rows(EXP / "data" / row_file)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], revision=cfg["model"]["revision"], trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    temperature = float(cfg["source_rows"]["harvest_temperature"])
    target_gain = float(cfg["train"]["uplift_gain_logits"])

    records = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with model.disable_adapter():
                ref = final_logits(model, row)
            z = final_logits(model, row)
            rejected = int(row["rejected_id"])
            chosen_ids = torch.tensor(row["chosen_ids"], dtype=torch.long, device=z.device)
            ref_best = int(torch.argmax(ref[chosen_ids]))
            chosen = int(chosen_ids[ref_best])
            chosen_gain = float(z[chosen] - ref[chosen])
            rejected_drift = float(z[rejected] - ref[rejected])
            pair_before = float(ref[chosen] - ref[rejected])
            pair_after = float(z[chosen] - z[rejected])
            target_mask = torch.zeros_like(z, dtype=torch.bool)
            target_mask[chosen_ids] = True
            if args.arm == "demote":
                target_mask[rejected] = True
                hit = pair_after > 0
            else:
                hit = chosen_gain >= target_gain
            drift = (z - ref).abs()[~target_mask]
            logp_ref = torch.log_softmax(ref / temperature, dim=0)
            p_ref = logp_ref.exp(); h_ref = (p_ref * -logp_ref).sum()
            v_ref = (p_ref * ((-logp_ref) - h_ref).square()).sum()
            logp = torch.log_softmax(z / temperature, dim=0)
            p = logp.exp(); h = (p * -logp).sum()
            v = (p * ((-logp) - h).square()).sum()
            records.append({
                "item_id": row["item_id"], "source": row["source"],
                "family": row["family"],
                "entropy_quartile": row["geometry"].get("entropy_quartile"),
                "varentropy_quartile": row["geometry"].get("varentropy_quartile"),
                "objective_hit": bool(hit), "chosen_gain": chosen_gain,
                "rejected_drift": rejected_drift,
                "pair_gap_shift": pair_after - pair_before,
                "median_abs_nontarget_drift": float(torch.median(drift)),
                "p95_abs_nontarget_drift": float(torch.quantile(drift, .95)),
                "entropy_change": float(h - h_ref),
                "varentropy_change": float(v - v_ref),
            })
            if (index + 1) % 50 == 0:
                print(f"[{args.arm}] audited {index + 1}/{len(rows)}", flush=True)

    strata: dict[str, dict[str, dict]] = {"entropy": {}, "varentropy": {}}
    for field, out_name in (("entropy_quartile", "entropy"),
                            ("varentropy_quartile", "varentropy")):
        buckets = defaultdict(list)
        for record in records:
            buckets[str(record[field])].append(record)
        strata[out_name] = {key: summarize(value) for key, value in sorted(buckets.items())}
    out = {"arm": args.arm, "adapter": str(args.adapter),
           "overall": summarize(records), "strata": strata, "records": records}
    dest = EXP / "runs" / f"logit_audit_{args.arm}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
