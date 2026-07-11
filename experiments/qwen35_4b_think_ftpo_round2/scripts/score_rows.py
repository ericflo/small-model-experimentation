#!/usr/bin/env python3
"""Score parent FTPO rows under the frozen base and build matched geometry sets.

This is an exact-logit Transformers pass and must run under the training venv.
Every row is forwarded alone: round 1 measured 0.30--0.44-logit batching drift
on this hybrid recurrent architecture.

  ../../.venv/bin/python scripts/score_rows.py
  ../../.venv/bin/python scripts/score_rows.py --smoke 8
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

EXP = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def final_logits(backbone, head, context_ids: list[int], device) -> torch.Tensor:
    ids = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.ones_like(ids)
    hidden = backbone(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    return head(hidden[:, -1, :]).float()[0]


def score_one(z: torch.Tensor, row: dict, temperature: float) -> dict:
    rejected = int(row["rejected_id"])
    chosen = torch.tensor(row["chosen_ids"], dtype=torch.long, device=z.device)
    zr = z[rejected]
    zc = z[chosen]
    best_idx = int(torch.argmax(zc))
    best_chosen_id = int(chosen[best_idx])
    gap = float(zr - zc[best_idx])
    top_id = int(torch.argmax(z))
    logp = torch.log_softmax(z / temperature, dim=0)
    p = logp.exp()
    surprise = -logp
    entropy = (p * surprise).sum()
    varentropy = (p * (surprise - entropy).square()).sum()
    return {
        "rejected_is_argmax": rejected == top_id,
        "rejected_rank": 1 + int((z > zr).sum()),
        "rejected_minus_best_chosen_logits": gap,
        "best_chosen_id": best_chosen_id,
        "rejected_probability_t06": float(p[rejected]),
        "best_chosen_probability_t06": float(p[best_chosen_id]),
        "top_probability_t06": float(p[top_id]),
        "entropy_t06": float(entropy),
        "varentropy_t06": float(varentropy),
    }


def qualifies(metrics: dict, geom: dict) -> bool:
    checks = [
        (not geom["rejected_must_be_argmax"] or metrics["rejected_is_argmax"]),
        metrics["rejected_minus_best_chosen_logits"]
        >= float(geom["min_rejected_minus_chosen_logits"]),
        metrics["rejected_probability_t06"] >= float(geom["min_rejected_probability"]),
        metrics["entropy_t06"] <= float(geom["max_entropy"]),
        metrics["varentropy_t06"] >= float(geom["min_varentropy"]),
        metrics["best_chosen_probability_t06"]
        >= float(geom["min_chosen_relative_probability"])
        * metrics["top_probability_t06"],
    ]
    return all(checks)


def quantiles(rows: list[dict], key: str) -> dict[str, float]:
    if not rows:
        return {}
    values = np.asarray([r["geometry"][key] for r in rows], dtype=float)
    return {
        name: float(np.quantile(values, q))
        for name, q in (("min", 0), ("p10", .1), ("p25", .25),
                        ("median", .5), ("p75", .75), ("p90", .9), ("max", 1))
    }


def assign_quartiles(rows: list[dict], key: str, n_bins: int) -> None:
    if not rows:
        return
    values = np.asarray([r["geometry"][key] for r in rows], dtype=float)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    inner = edges[1:-1]
    for row, value in zip(rows, values):
        row["geometry"][key.replace("_t06", "") + "_quartile"] = int(
            np.searchsorted(inner, value, side="right") + 1
        )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=0, metavar="N")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    source = cfg["source_rows"]
    geom = cfg["geometry"]
    paths = {name: EXP / rel for name, rel in
             (("real", source["real"]), ("shuffled", source["shuffled"]))}
    pools = {name: load_rows(path) for name, path in paths.items()}
    if args.smoke:
        pools = {name: rows[:args.smoke] for name, rows in pools.items()}

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], revision=cfg["model"]["revision"],
        trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    backbone, head = model.model, model.lm_head
    device = next(model.parameters()).device
    temperature = float(source["harvest_temperature"])

    scored: dict[str, list[dict]] = {}
    with torch.inference_mode():
        for name, rows in pools.items():
            out = []
            for index, row in enumerate(rows):
                enriched = dict(row)
                enriched["geometry"] = score_one(
                    final_logits(backbone, head, row["context_ids"], device),
                    row, temperature)
                enriched["geometry"]["qualified"] = qualifies(enriched["geometry"], geom)
                out.append(enriched)
                if (index + 1) % 100 == 0:
                    print(f"[{name}] scored {index + 1}/{len(rows)}", flush=True)
            scored[name] = out

    qualified = {name: [r for r in rows if r["geometry"]["qualified"]]
                 for name, rows in scored.items()}
    matched_n = min(int(geom["max_rows_per_arm"]), *(len(v) for v in qualified.values()))
    gate_pass = matched_n >= int(geom["min_rows_gate"])
    rng = np.random.default_rng(int(geom["selection_seed"]))
    selected = {}
    for name, rows in qualified.items():
        if len(rows) > matched_n:
            keep = sorted(rng.choice(len(rows), size=matched_n, replace=False).tolist())
            rows = [rows[i] for i in keep]
        assign_quartiles(rows, "entropy_t06", int(geom["entropy_quartiles"]))
        assign_quartiles(rows, "varentropy_t06", int(geom["entropy_quartiles"]))
        selected[name] = rows

    summary = {
        "smoke": bool(args.smoke),
        "source_sha256": {name: digest(path) for name, path in paths.items()},
        "pool_rows": {name: len(rows) for name, rows in pools.items()},
        "qualified_rows": {name: len(rows) for name, rows in qualified.items()},
        "matched_rows": matched_n,
        "min_rows_gate": int(geom["min_rows_gate"]),
        "gate_pass": gate_pass,
        "filter": geom,
        "quantiles": {
            name: {key: quantiles(rows, key) for key in
                   ("rejected_probability_t06", "best_chosen_probability_t06",
                    "rejected_minus_best_chosen_logits", "entropy_t06", "varentropy_t06")}
            for name, rows in scored.items()
        },
        "qualified_top_rejected": {
            name: Counter(r.get("rejected_surface", str(r["rejected_id"])).strip()
                          for r in rows).most_common(15)
            for name, rows in qualified.items()
        },
    }
    dest = EXP / "runs" / ("row_geometry_smoke.json" if args.smoke else "row_geometry.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))
    if not args.smoke and gate_pass:
        write_rows(EXP / "data" / "rows_real_selected.jsonl.gz", selected["real"])
        write_rows(EXP / "data" / "rows_shuffled_selected.jsonl.gz", selected["shuffled"])
    print(json.dumps({k: summary[k] for k in
                      ("pool_rows", "qualified_rows", "matched_rows", "gate_pass")}, indent=2))
    if not args.smoke and not gate_pass:
        raise SystemExit("P0 geometry gate failed; refusing to create training rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
