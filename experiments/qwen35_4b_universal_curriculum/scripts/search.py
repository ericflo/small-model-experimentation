#!/usr/bin/env python3
"""Curriculum-design SEARCH driver — the doctrine's fast loop made concrete.

Given a list of curriculum configs (skill mix + train hyperparams), for each one:
  1. synthesize the curriculum (gen_curriculum.py --mix ...)          [CPU, seconds]
  2. fast-train a LoRA adapter from base (train_think.py)             [~5-15 min]
  3. eval TRANSFER to the held-out menagerie, paired base-vs-adapter  [~8 min]
     (bench.py --tier quick --backend qwen --think-budget 1024)
  4. log the transfer delta (adapter - base) + per-family deltas.

Each config gets a FRESH eval seed (bench.py forbids reuse). Resumable: configs
already present in runs/search_results.jsonl are skipped. Prints a leaderboard
sorted by transfer delta. Reads configs from scripts/search_configs.json (or --configs).

This is the fitness function for "what lifts all boats the most." It does NOT read
menagerie internals (firewall): bench.py returns aggregate + per-family only.

Run under the repo .venv. One RTX 4090; single-tenant the GPU.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
GF = ROOT / "experiments" / "qwen35_4b_gauntlet_frontier"      # owns bench.py
PY = str(ROOT / ".venv" / "bin" / "python")
GEN = str(EXP / "scripts" / "gen_curriculum.py")
TRAIN = str(GF / "scripts" / "train_think.py")
BENCH = str(GF / "scripts" / "bench.py")
MEN = GF / "runs" / "menagerie"
RESULTS = EXP / "runs" / "search_results.jsonl"
ADAPTERS = ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum" / "adapters"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def done_names() -> set[str]:
    if not RESULTS.exists():
        return set()
    return {json.loads(l)["name"] for l in RESULTS.read_text().splitlines() if l.strip()}


def eval_transfer(name: str, adapter: Path, seed: int, tier: str, budget: int) -> dict:
    run([PY, BENCH, "--tier", tier, "--seed", str(seed), "--backend", "qwen",
         "--adapter", str(adapter), "--arms", "base", "adapter",
         "--think-budget", str(budget), "--note", f"universal-search:{name}"])
    base = json.loads((MEN / f"{tier}_seed{seed}_base.json").read_text())
    adap = json.loads((MEN / f"{tier}_seed{seed}_adapter.json").read_text())
    pfd = {k: round(adap["per_family"][k] - base["per_family"][k], 4) for k in base["per_family"]}
    return {"base_agg": base["aggregate"], "adapter_agg": adap["aggregate"],
            "delta": round(adap["aggregate"] - base["aggregate"], 4), "per_family_delta": pfd,
            "eval_seed": seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=EXP / "scripts" / "search_configs.json")
    ap.add_argument("--tier", default="quick")
    ap.add_argument("--think-budget", type=int, default=1024)
    ap.add_argument("--base-eval-seed", type=int, default=59100,
                    help="config i uses eval seed base-eval-seed + i (fresh per bench.py rule)")
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None, help="run only these config names")
    a = ap.parse_args()

    configs = json.loads(a.configs.read_text())
    if a.only:
        configs = [c for c in configs if c["name"] in a.only]
    skip = done_names()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(configs):
        name = c["name"]
        if name in skip:
            print(f"[search] skip {name} (already in results)", flush=True)
            continue
        print(f"\n===== config {name}: {c} =====", flush=True)
        data = EXP / "data" / f"sft_{name}.jsonl"
        adapter = ADAPTERS / name
        run([PY, GEN, "--mix", c["mix"], "--out", str(data)])
        run([PY, TRAIN, "--train", str(data), "--out", str(adapter),
             "--epochs", str(c.get("epochs", 2.0)), "--rank", str(c.get("rank", 32)),
             "--alpha", str(c.get("alpha", 2 * c.get("rank", 32))),
             "--batch-size", "2", "--grad-accum", "4", "--max-length", "2560",
             "--w-think", str(c.get("w_think", 0.2)), "--seed", str(a.train_seed)])
        res = eval_transfer(name, adapter, a.base_eval_seed + i, a.tier, a.think_budget)
        row = {"name": name, "config": c, **res}
        with RESULTS.open("a", encoding="utf-8") as h:
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[search] {name}: base {res['base_agg']:.4f} -> adapter {res['adapter_agg']:.4f} "
              f"(delta {res['delta']:+.4f})", flush=True)

    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: -r["delta"])
    print("\n=== TRANSFER LEADERBOARD (delta = adapter - base, menagerie aggregate) ===")
    for r in rows:
        print(f"  {r['delta']:+.4f}  {r['name']:<20} (base {r['base_agg']:.3f} -> {r['adapter_agg']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
