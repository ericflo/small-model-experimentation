#!/usr/bin/env python3
"""Menagerie evaluation event: base and/or adapter on ONE fresh seed + tier.

Firewall-compliant wrapper: invokes the benchmark suite's run.py CLI as a
subprocess (under the required repo .venv interpreter), reads ONLY the
aggregate and per_family numbers from the output JSON, and appends them to
runs/menagerie_log.jsonl. Never reuse a seed across evaluation events.

  python3 scripts/bench.py --seed 52001 --tier quick --arms base adapter \
      --adapter large_artifacts/qwen35_4b_gauntlet_breadth_round1/adapters/round1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
VENV_PY = REPO / ".venv" / "bin" / "python"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
LOG = EXP / "runs" / "menagerie_log.jsonl"


def used_seeds() -> set[int]:
    seeds = set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            if line.strip():
                seeds.add(int(json.loads(line)["seed"]))
    return seeds


def run_arm(tier: str, seed: int, out_path: Path, adapter: str | None,
            merged: str | None = None, backend: str | None = None,
            max_batch: int | None = None) -> dict:
    command = [
        str(VENV_PY), str(MENAGERIE),
        "--tier", tier,
        "--seed", str(seed),
        "--out", str(out_path),
    ]
    if backend:
        command += ["--backend", backend]
    if max_batch:
        command += ["--max-batch", str(max_batch)]
    if adapter and merged:
        raise ValueError("pass either adapter or merged, not both")
    if adapter:
        # WARNING (2026-07-10): vLLM runtime LoRA is a verified silent no-op
        # for Qwen3.5-4B; prefer merged checkpoints via --model-id.
        command += ["--adapter", str(Path(adapter).resolve())]
    if merged:
        command += ["--model-id", str(Path(merged).resolve())]
    started = time.perf_counter()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        command, cwd=str(MENAGERIE.parent), capture_output=True, text=True, env=env
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:] + "\n")
        raise RuntimeError(f"menagerie run failed (exit {completed.returncode})")
    payload = json.loads(out_path.read_text())
    per_family = {}
    for family, stats in payload["per_family"].items():
        if isinstance(stats, dict):
            if "score" in stats:
                per_family[family] = float(stats["score"])
            elif "mean" in stats:
                per_family[family] = float(stats["mean"])
            else:
                raise RuntimeError(
                    f"per_family entry for {family!r} has no score/mean field; refusing to guess"
                )
        else:
            per_family[family] = float(stats)
    record = {
        "aggregate": float(payload["aggregate"]),
        "per_family": per_family,
        "within_budget": payload.get("within_budget"),
        "wall_seconds": round(elapsed, 1),
    }
    # Firewall hygiene: rewrite the stored file to aggregate-level fields only
    # so nothing item-level lingers inside the experiment tree.
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="quick")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arms", nargs="+", choices=["base", "adapter", "merged"],
                        default=["base", "merged"])
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--merged", type=str, default=None,
                        help="merged full checkpoint dir (deployed via menagerie --model-id)")
    parser.add_argument("--backend", type=str, default=None,
                        help="menagerie backend override (e.g. qwen for the HF parity oracle; "
                             "never compare across backends)")
    parser.add_argument("--max-batch", type=int, default=None)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--allow-seed-reuse", action="store_true",
                        help="only for re-running a failed arm of the SAME event")
    args = parser.parse_args()

    if "adapter" in args.arms and not args.adapter:
        parser.error("--adapter path required when running the adapter arm")
    if "merged" in args.arms and not args.merged:
        parser.error("--merged path required when running the merged arm")
    if args.seed == 31337:
        parser.error("31337 is the published baseline seed; pick a fresh one")
    if not args.allow_seed_reuse and args.seed in used_seeds():
        parser.error(f"seed {args.seed} already used in a prior evaluation event; pick a fresh one")

    out_dir = EXP / "runs" / "menagerie"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for occurrence, arm in enumerate(args.arms):
        adapter = args.adapter if arm == "adapter" else None
        merged = args.merged if arm == "merged" else None
        arm_label = f"{arm}{occurrence}" if args.arms.count(arm) > 1 else arm
        out_path = out_dir / f"{args.tier}_seed{args.seed}_{arm_label}.json"
        print(f"[bench] {args.tier} seed={args.seed} arm={arm_label} ...", flush=True)
        results[arm_label] = run_arm(args.tier, args.seed, out_path, adapter, merged,
                                     backend=args.backend, max_batch=args.max_batch)
        print(f"[bench] {arm_label}: aggregate={results[arm_label]['aggregate']:.4f} "
              f"({results[arm_label]['wall_seconds']}s)", flush=True)
        record = {
            "tier": args.tier,
            "seed": args.seed,
            "arm": arm_label,
            "adapter": adapter,
            "note": args.note,
            **results[arm_label],
        }
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    labels = list(results.keys())
    if len(labels) == 2:
        first, second = labels
        delta = results[second]["aggregate"] - results[first]["aggregate"]
        per_family_delta = {
            family: round(results[second]["per_family"][family]
                          - results[first]["per_family"][family], 4)
            for family in results[first]["per_family"]
        }
        kind = "NULL-CALIBRATION" if first.startswith("base") and second.startswith("base") else "DELTA"
        print(f"[bench] {kind} aggregate ({second} - {first}): {delta:+.4f}")
        print(json.dumps(per_family_delta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
