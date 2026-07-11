#!/usr/bin/env python3
"""Menagerie evaluation event wrapper (firewall-compliant).

Invokes the benchmark suite's run.py CLI as a subprocess under the repo .venv
interpreter, reads ONLY aggregate + per_family scores, appends to
runs/menagerie_log.jsonl. Seeds are fresh per event and union-checked against
this experiment's log, the gauntlet's log, and the suite baseline seed 31337.

  python3 scripts/bench.py --tier quick --seed 61001 --arms base pivot \
      --merged pivot=<merged_dir>
  python3 scripts/bench.py --tier quick --seed 61002 --arms base base   # null
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
VENV_PY = REPO / ".venv" / "bin" / "python"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
LOG = EXP / "runs" / "menagerie_log.jsonl"
GAUNTLET_LOG = REPO / ("experiments/qwen35_4b_gauntlet_breadth_round1/"
                       "runs/menagerie_log.jsonl")
BASELINE_SEED = 31337


def used_seeds() -> set[int]:
    seeds = {BASELINE_SEED}
    for log in (LOG, GAUNTLET_LOG):
        if log.exists():
            for line in log.read_text().splitlines():
                if line.strip():
                    seeds.add(int(json.loads(line)["seed"]))
    return seeds


def run_arm(tier: str, seed: int, arm: str, model_id: str | None) -> dict:
    out_path = EXP / "runs" / "menagerie" / f"{tier}_seed{seed}_{arm}_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(VENV_PY), str(MENAGERIE), "--tier", tier, "--seed", str(seed),
           "--out", str(out_path)]
    if model_id:
        cmd += ["--model-id", model_id, "--device", "cuda:0"]
    print(f"[bench] {arm}: {' '.join(cmd)}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=str(MENAGERIE.parent))
    if result.returncode != 0:
        raise SystemExit(f"menagerie run failed for arm {arm}")
    payload = json.loads(out_path.read_text())
    return {
        "aggregate": payload["aggregate"],
        "per_family": {fam: stats["score"] if isinstance(stats, dict) else stats
                       for fam, stats in payload["per_family"].items()},
        "within_budget": payload.get("within_budget"),
        "wall_s": time.time() - started,
        "out": str(out_path.relative_to(EXP)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", required=True, choices=["quick", "medium", "slow", "deep"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arms", nargs="+", required=True,
                        help="e.g. base pivot; 'base base' for a null repeat")
    parser.add_argument("--merged", action="append", default=[],
                        help="arm=merged_checkpoint_dir mappings")
    parser.add_argument("--allow-seed-reuse", action="store_true",
                        help="ONLY for the same-seed null repeat / paired arms")
    args = parser.parse_args()

    merged = dict(pair.split("=", 1) for pair in args.merged)
    for arm in args.arms:
        if arm != "base" and arm not in merged:
            raise SystemExit(f"trained arm {arm!r} needs --merged {arm}=<dir> (C49)")

    if args.seed in used_seeds() and not args.allow_seed_reuse:
        raise SystemExit(f"seed {args.seed} already used (union check incl. "
                         f"gauntlet log + baseline {BASELINE_SEED})")

    event: dict = {"tier": args.tier, "seed": args.seed, "time": time.time(),
                   "arms": {}}
    for arm in args.arms:
        key = arm if arm not in event["arms"] else f"{arm}_repeat"
        event["arms"][key] = run_arm(args.tier, args.seed, key,
                                     merged.get(arm))
        print(f"[bench] {key}: aggregate={event['arms'][key]['aggregate']:.4f} "
              f"({event['arms'][key]['wall_s']:.0f}s)", flush=True)

    arms = list(event["arms"])
    if len(arms) == 2:
        delta = event["arms"][arms[1]]["aggregate"] - event["arms"][arms[0]]["aggregate"]
        event["delta"] = delta
        print(f"[bench] delta {arms[1]} - {arms[0]} = {delta:+.4f}", flush=True)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(event) + "\n")
    print(f"[bench] appended to {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
