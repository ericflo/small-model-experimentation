#!/usr/bin/env python3
"""A data-design RECIPE = a fast weighted sample over the cached pool. Weight each
row by kind * level * family-group multipliers from a recipe spec, sample to a
target size, write SFT data. Seconds. The knob the fast search turns."""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
POOL = EXP / "data" / "cached_pool.jsonl"
# weak-axis family groups (the menagerie axes we want to lift)
GROUP = {
    "induction": {"glyphgate"}, "exploration": {"burrowmaze"},
    "repair": {"loomfix", "patchwheel"}, "optimization": {"packhouse", "stallwright"},
    "state": {"caravan", "foundry_ledger", "brinework", "kilnrite"},
    "tools": {"ferrier", "gatepost"}, "abstain": {"runeward"},
}
FAM2GROUP = {f: g for g, fs in GROUP.items() for f in fs}


def load_pool():
    return [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]


def build(pool, recipe, seed=0):
    rng = random.Random(seed)
    kw = recipe.get("kind_w", {}); lw = recipe.get("level_w", {}); gw = recipe.get("group_w", {})
    dflt_k = recipe.get("kind_default", 1.0); dflt_l = recipe.get("level_default", 1.0); dflt_g = recipe.get("group_default", 1.0)
    weighted = []
    for r in pool:
        w = (kw.get(r.get("kind"), dflt_k)
             * (lw.get(r.get("level")) or lw.get(str(r.get("level")), dflt_l))
             * gw.get(FAM2GROUP.get(r.get("family"), "other"), dflt_g))
        if w > 0:
            weighted.append((w, r))
    if not weighted:
        return []
    size = min(recipe.get("size", 1200), len(weighted))
    ws = [w for w, _ in weighted]
    picked = rng.choices(range(len(weighted)), weights=ws, k=size)  # weighted, with replacement
    rows = []
    for i in set(picked):  # dedup the sample; keep count as row_weight boost
        _, r = weighted[i]
        out = {k: r[k] for k in ("family", "level", "kind", "messages", "think", "answer", "n_think_tokens") if k in r}
        out["row_weight"] = round(r.get("row_weight", 1.0), 3)
        rows.append(out)
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recipe", required=True, help="JSON recipe spec")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = load_pool(); rows = build(pool, json.loads(a.recipe), a.seed)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"recipe -> {len(rows)} rows | kinds {dict(Counter(r['kind'] for r in rows).most_common(6))}")


if __name__ == "__main__":
    main()
