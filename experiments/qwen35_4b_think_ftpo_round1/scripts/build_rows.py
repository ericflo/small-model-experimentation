#!/usr/bin/env python3
"""Build FTPO training rows from harvest groups (CPU only).

--arm pivot     outcome-conditioned mining (real labels)
--arm shuffled  identical mining after per-prompt label permutation (control)

Applies the preregistered regularization: rejected-token flattening (0.3,
median-anchored power transform, greedy fill with source tiebreak) and
chosen flattening (0.5, p95/floor-50 reference, per-row dedupe then prune).

python3 scripts/build_rows.py --arm pivot
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import pivotmine  # noqa: E402

THINK_CLOSE = 248069


def think_sequence(ids: list[int]) -> list[int]:
    return ids[: ids.index(THINK_CLOSE)] if THINK_CLOSE in ids else list(ids)


def load_groups(harvest_dir: Path):
    for shard in sorted(harvest_dir.glob("groups_slice*.jsonl.gz")):
        with gzip.open(shard, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def normalize_surface(surface: str) -> str:
    s = surface.strip().lower()
    start, end = 0, len(s)
    while start < end and not (s[start].isalnum() or s[start] == "_"):
        start += 1
    while end > start and not (s[end - 1].isalnum() or s[end - 1] == "_"):
        end -= 1
    return s[start:end] or s or surface


def mine_rows(cfg: dict, groups, shuffle_labels: bool, tokenizer) -> tuple[list[dict], Counter]:
    m = cfg["mining"]
    rng = np.random.default_rng(int(m["shuffle_seed"]))
    counts: Counter = Counter()
    rows: list[dict] = []
    for group in groups:
        counts["groups"] += 1
        sequences = [think_sequence(o["stage1_token_ids"]) for o in group["outputs"]]
        successes = [o["score"] >= 1.0 for o in group["outputs"]]
        if shuffle_labels:
            successes = list(rng.permutation(np.array(successes)))
        pivots = pivotmine.mine_pivots(
            sequences, successes,
            min_depth=int(m["pivot_min_depth"]),
            min_branch_rollouts=int(m["pivot_min_branch_rollouts"]),
            min_gap=float(m["pivot_min_gap"]),
            max_nodes=int(m["pivot_max_nodes_per_prompt"]),
        )
        if pivots:
            counts["groups_with_node"] += 1
        for pivot in pivots:
            context_ids = list(group["prompt_token_ids"]) + list(pivot.prefix)
            if len(context_ids) + 1 > int(m["context_cap_tokens"]):
                counts["overlength_context"] += 1
                continue
            rejected_surface = tokenizer.decode([pivot.rejected_id])
            chosen = []
            seen_keys = {normalize_surface(rejected_surface)}
            for tid in pivot.chosen_ids:
                key = normalize_surface(tokenizer.decode([tid]))
                if key in seen_keys:
                    counts["chosen_case_variant_dropped"] += 1
                    continue
                seen_keys.add(key)
                chosen.append(tid)
            if not chosen:
                counts["no_chosen_after_filters"] += 1
                continue
            rows.append({
                "item_id": group["item_id"],
                "source": group["source"],
                "family": group["family"],
                "level": group["level"],
                "context_ids": context_ids,
                "rejected_id": int(pivot.rejected_id),
                "rejected_surface": rejected_surface,
                "chosen_ids": [int(t) for t in chosen],
                "chosen_surfaces": [tokenizer.decode([t]) for t in chosen],
                "depth": pivot.depth,
                "gap": pivot.gap,
                "n_rejected_rollouts": pivot.n_rejected,
                "n_chosen_rollouts": pivot.n_chosen,
            })
            counts["rows"] += 1
    return rows, counts


def flatten_rejected(rows: list[dict], strength: float, target_n: int, seed: int) -> list[dict]:
    """Median-anchored power-transform resampling with a source-share tiebreak."""
    if not rows:
        return rows
    token_counts = Counter(r["rejected_surface"].strip().lower() for r in rows)
    median = float(np.median(list(token_counts.values())))
    weights = {tok: 1.0 if cnt <= median or strength <= 0 else (median / cnt) ** strength
               for tok, cnt in token_counts.items()}
    total = sum(weights[t] * c for t, c in token_counts.items())
    ratios = {tok: weights[tok] * cnt / total for tok, cnt in token_counts.items()}
    targets = {tok: int(round(r * target_n)) for tok, r in ratios.items()}

    pool_share = Counter(r["source"] for r in rows)
    share = {s: c / len(rows) for s, c in pool_share.items()}
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row["rejected_surface"].strip().lower()][row["source"]].append(row)
    rng = np.random.default_rng(seed)
    for tok in buckets:
        for source in buckets[tok]:
            rng.shuffle(buckets[tok][source])

    selected: list[dict] = []
    seen_tok: Counter = Counter()
    seen_src: Counter = Counter()

    def best(respect_cap: bool):
        denom = max(len(selected), 1)
        best_key, best_pick = None, None
        for tok, sources in buckets.items():
            if respect_cap and seen_tok[tok] >= targets.get(tok, 0):
                continue
            tok_gap = ratios.get(tok, 0.0) - seen_tok[tok] / denom
            for source, avail in sources.items():
                if not avail:
                    continue
                src_def = share.get(source, 0.0) - seen_src[source] / denom
                key = (tok_gap, src_def)
                if best_key is None or key > best_key:
                    best_key, best_pick = key, (tok, source)
        return best_pick

    for respect_cap in (True, False):
        while len(selected) < target_n:
            pick = best(respect_cap)
            if pick is None:
                break
            tok, source = pick
            selected.append(buckets[tok][source].pop())
            seen_tok[tok] += 1
            seen_src[source] += 1
    rng.shuffle(selected)
    return selected


def flatten_chosen(rows: list[dict], strength: float, percentile: int,
                   min_ref: int, counts: Counter) -> list[dict]:
    key_counts: Counter = Counter()
    for row in rows:
        for surface in row["chosen_surfaces"]:
            key_counts[normalize_surface(surface)] += 1
    if not key_counts:
        return rows
    values = sorted(key_counts.values())
    ref = max(min_ref, values[int((len(values) - 1) * percentile / 100)])
    targets = {k: (int(round(c * (ref / c) ** strength)) if strength > 0 and c > ref else c)
               for k, c in key_counts.items()}
    current = Counter(key_counts)
    # prune from rows with the most chosen entries first; drop rows only when empty
    for key in sorted(targets, key=lambda k: current[k] - targets[k], reverse=True):
        while current[key] > targets[key]:
            candidates = [r for r in rows
                          if key in [normalize_surface(s) for s in r["chosen_surfaces"]]
                          and len(r["chosen_ids"]) > 1]
            if not candidates:
                break
            row = max(candidates, key=lambda r: len(r["chosen_ids"]))
            idx = [normalize_surface(s) for s in row["chosen_surfaces"]].index(key)
            del row["chosen_ids"][idx]
            del row["chosen_surfaces"][idx]
            current[key] -= 1
            counts["chosen_flattened"] += 1
    kept = [r for r in rows if r["chosen_ids"]]
    counts["rows_dropped_empty_chosen"] += len(rows) - len(kept)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["pivot", "shuffled"], required=True)
    parser.add_argument("--harvest-dir", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    reg = cfg["regularization"]
    harvest_dir = Path(args.harvest_dir) if args.harvest_dir else EXP / "runs" / "harvest"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        trust_remote_code=True)

    rows, counts = mine_rows(cfg, load_groups(harvest_dir),
                             shuffle_labels=(args.arm == "shuffled"),
                             tokenizer=tokenizer)
    pool = len(rows)
    counts["pool_rows"] = pool

    target_n = min(int(reg["max_train_examples"]),
                   int(pool * float(reg["max_train_fraction"])))
    rows = flatten_rejected(rows, float(reg["rejected_strength"]), target_n,
                            seed=int(cfg["train"]["seed"]))
    rows = flatten_chosen(rows, float(reg["chosen_strength"]),
                          int(reg["chosen_ref_percentile"]),
                          int(reg["chosen_ref_min_count"]), counts)
    rows = [r for r in rows if len(r["chosen_ids"]) >= int(reg["min_chosen_tokens"])]
    counts["train_rows"] = len(rows)

    dest = EXP / "data" / f"rows_{args.arm}.jsonl.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    chosen_hist = Counter(len(r["chosen_ids"]) for r in rows)
    top_rejected = Counter(r["rejected_surface"].strip() for r in rows).most_common(15)
    summary = {
        "arm": args.arm,
        "counts": dict(counts),
        "train_rows": len(rows),
        "pool_rows": pool,
        "min_train_rows_gate": int(reg["min_train_rows"]),
        "gate_pass": len(rows) >= int(reg["min_train_rows"]),
        "chosen_per_row_hist": {str(k): v for k, v in sorted(chosen_hist.items())},
        "top_rejected_surfaces": top_rejected,
        "median_context_tokens": int(np.median([len(r["context_ids"]) for r in rows])) if rows else 0,
    }
    (EXP / "runs" / f"build_rows_{args.arm}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("arm", "pool_rows", "train_rows", "gate_pass",
                       "chosen_per_row_hist")}, indent=2))
    print("top rejected:", top_rejected[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
