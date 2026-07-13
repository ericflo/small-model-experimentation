#!/usr/bin/env python3
"""Fast data-design SEARCH loop. Per recipe: remix the cached pool (seconds) ->
FAST train (low-rank, 1 epoch, ~5min) -> eval the ADAPTER on menagerie quick @1024
via the HF backend (adapter APPLIES in HF; no 9GB merge, no vLLM no-op) -> record
the aggregate. Ranks recipes by how much they lift the menagerie vs a fixed base
baseline. Winners get confirmed at full fidelity (8192 + medium) separately.
Run under .venv."""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]; ROOT = EXP.parents[1]
VP = ROOT / ".venv" / "bin" / "python"
POOL_SCRIPT = EXP / "scripts" / "build_recipe.py"
TRAIN = EXP / "scripts" / "train_think.py"
BENCH = EXP / "scripts" / "bench.py"
ADAPTERS = ROOT / "large_artifacts" / "qwen35_4b_gauntlet_frontier" / "adapters"
LOG = EXP / "runs" / "fast_search_log.jsonl"

# Initial hypothesis grid: what lifts ALL boats? Each is a weighted remix.
RECIPES = {
  "balanced_broad":   {"size":1400,"level_default":1.0,"group_default":1.0},
  "weakaxis_hard":    {"size":1400,"level_w":{"3":1.3,"4":1.7,"5":1.7,"6":1.7},"level_default":0.5,
                       "group_w":{"induction":1.5,"exploration":1.5,"repair":1.5,"optimization":1.5},"group_default":0.7},
  "oracle_heavy":     {"size":1400,"kind_w":{"oracle_trace":2.5,"induction_trace":2.5,"skin_trace":2.0,"oracle_inject":2.0},"kind_default":0.6},
  "selfverified_heavy":{"size":1400,"kind_w":{"atom":2.0,"atom_fc":2.0,"episode":2.0,"epmastery":2.0,"eff_compress":1.5},"kind_default":0.5},
  "episode_horizon":  {"size":1400,"kind_w":{"episode":2.5,"epmastery":2.5},"kind_default":0.6,"level_w":{"3":1.3,"4":1.5,"5":1.5,"6":1.5},"level_default":0.7},
  "recovery_emission":{"size":1400,"kind_w":{"atom_fc":3.0,"eff_compress":2.0,"eff_brevity":1.5,"atom":1.5},"kind_default":0.4},
}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=58100, help="fresh menagerie seed base (one per recipe)")
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--recipes", nargs="*", default=list(RECIPES))
    a = ap.parse_args()
    for i, name in enumerate(a.recipes):
        rec = RECIPES[name]; t0 = time.time()
        data = EXP / "data" / f"recipe_{name}.jsonl"
        adir = ADAPTERS / f"fs_{name}"
        run([VP, POOL_SCRIPT, "--recipe", json.dumps(rec), "--out", data])
        tr = run([VP, TRAIN, "--train", data, "--out", adir, "--epochs", a.epochs,
                  "--rank", a.rank, "--alpha", a.rank*2, "--batch-size", 2, "--grad-accum", 4,
                  "--max-length", 2048, "--w-think", 0.2])
        if not (adir / "adapter_model.safetensors").exists():
            print(f"[fs] {name}: TRAIN FAILED\n{tr.stderr[-800:]}", flush=True); continue
        ev = run([VP, BENCH, "--tier", "quick", "--seed", a.seed+i, "--backend", "qwen",
                  "--adapter", str(adir), "--arms", "adapter", "--think-budget", "1024",
                  "--note", f"fast-search {name}"])
        agg = None
        for line in ev.stdout.splitlines():
            if "aggregate=" in line:
                try: agg = float(line.split("aggregate=")[1].split()[0])
                except Exception: pass
        dt = time.time() - t0
        rec_out = {"recipe": name, "quick1024_agg": agg, "minutes": round(dt/60,1)}
        print(f"[fs] {name}: quick@1024 agg={agg} ({dt/60:.1f} min)", flush=True)
        with LOG.open("a") as h: h.write(json.dumps(rec_out)+"\n")


if __name__ == "__main__":
    main()
