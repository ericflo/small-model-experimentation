"""Score a token-cap arm against the pre-registered criteria (Arm 1/2 of the cap ladder).

Reads a pi_episode result JSON and reports every number the protocol pre-registered, so the
keep/kill decision is a table lookup, not an interpretation:

  (a) mean episode secs  <= 250   (baseline 364)
  (b) timeout rate       <= 25%   (baseline 13/32 = 41%)
  (c) single-shot mean   >= 0.55  (baseline 0.606; null is acceptable -- the cap converts
                                   wall-clock into turns, not turns into solves)
  (d) matched best-of-3  >= 0.70  (baseline best-of-3 = 0.727). At k=6 this is bootstrapped:
      mean over all C(6,3) 3-subsets per task of [any of the 3 passed], averaged over tasks --
      the fair same-N comparison the critique demanded instead of best-of-6 vs best-of-3.
  HEADLINE: execution-selected best-of-k >= 0.78 at total wall-clock <= the baseline arm's.
  CANARY KILL: any task that was 3/3 at baseline dropping below 4/6 here.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

CANARIES = ["wordcount", "flatten", "base_convert", "rle", "pretty_bytes", "min_heap"]


def per_task(episodes):
    d = {}
    for e in episodes:
        d.setdefault(e["id"], []).append(e)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="pi_episode result JSON for the cap arm")
    ap.add_argument("--baseline", default=str(OUTD / "reports" / "pi_arm0_reanchor.json"))
    a = ap.parse_args()

    arm = json.load(open(a.arm))
    base = json.load(open(a.baseline)) if Path(a.baseline).exists() else None
    eps = arm["episodes"]
    tasks = per_task(eps)

    mean_secs = sum(e["secs"] for e in eps) / len(eps)
    to_rate = sum(1 for e in eps if e.get("exit") == 124) / len(eps)
    single = sum(sum(1 for e in v if e["reward"] == 1.0) / len(v) for v in tasks.values()) / len(tasks)
    bestofk = sum(1 for v in tasks.values() if any(e["reward"] == 1.0 for e in v)) / len(tasks)

    # matched best-of-3: for each task, average over all 3-subsets of its episodes
    mb3 = []
    for v in tasks.values():
        outcomes = [e["reward"] == 1.0 for e in v]
        subs = list(combinations(outcomes, 3)) if len(outcomes) >= 3 else [tuple(outcomes)]
        mb3.append(sum(any(s) for s in subs) / len(subs))
    matched_b3 = sum(mb3) / len(mb3)

    total_wall = sum(e["secs"] for e in eps)
    base_wall = sum(e["secs"] for e in base["episodes"]) if base else None

    print(f"=== {arm.get('label')} (k={arm.get('k')}, {len(eps)} episodes) ===")
    rows = [
        ("(a) mean episode secs", f"{mean_secs:.0f}", "<= 250", mean_secs <= 250),
        ("(b) timeout rate", f"{100*to_rate:.0f}%", "<= 25%", to_rate <= 0.25),
        ("(c) single-shot mean", f"{single:.3f}", ">= 0.55", single >= 0.55),
        ("(d) matched best-of-3", f"{matched_b3:.3f}", ">= 0.70", matched_b3 >= 0.70),
        (f"HEADLINE best-of-{arm.get('k')}", f"{bestofk:.3f}", ">= 0.78", bestofk >= 0.78),
    ]
    for name, val, crit, ok in rows:
        print(f"  {name:24s} {val:>8s}  (criterion {crit:8s}) {'PASS' if ok else 'FAIL'}")
    if base_wall:
        print(f"  total wall-clock         {total_wall:>7.0f}s  (baseline arm {base_wall:.0f}s) "
              f"{'PASS' if total_wall <= base_wall else 'OVER'}")

    print("\n  canaries (kill if any < 4/6-equivalent, i.e. rate < 0.667):")
    canary_kill = False
    for t in CANARIES:
        v = tasks.get(t, [])
        r = sum(1 for e in v if e["reward"] == 1.0) / max(1, len(v))
        bad = r < 0.667
        canary_kill |= bad
        print(f"    {t:14s} {sum(1 for e in v if e['reward']==1.0)}/{len(v)}  ({r:.2f}) "
              f"{'KILL' if bad else 'ok'}")

    keep = (mean_secs <= 250 and to_rate <= 0.25 and single >= 0.55 and matched_b3 >= 0.70
            and not canary_kill)
    print(f"\n  VERDICT: {'KEEP the cap' if keep else 'REJECT (see failed criteria above)'}"
          f"{' + HEADLINE WIN' if keep and bestofk >= 0.78 else ''}")

    print("\n  per-task detail:")
    for t, v in sorted(tasks.items()):
        n1 = sum(1 for e in v if e["reward"] == 1.0)
        mx = max(e["reward"] for e in v)
        to = sum(1 for e in v if e.get("exit") == 124)
        print(f"    {t:16s} pass {n1}/{len(v)}  maxR {mx:.2f}  timeouts {to}/{len(v)}")


if __name__ == "__main__":
    main()
