#!/usr/bin/env python3
"""Construct MATCHED-SIZE training sets from the shared candidate pool (design-review-hardened; ranking
signal = THINK-P(True) after the attempt-2 pivot -- the no-think judge is within-task chance here).
- exec:        execution-verified (full_pass), per-task most-frequent order, cap 12/task -- C18-identical ceiling.
- conf_strat:  PRIMARY verifier-free arm. Depth-stratified top-score: per-depth quotas proportional to the
               pool's JUDGE-SCORE MASS per depth (candidate-count quotas allocate slots where WRONG candidates
               explode -- the attempt-2 gate failure), top-score within depth, cap 12/task.
- conf_global: ablation. Global top-score -- the naive fully-deployable policy.
- rand:        no-filter floor, sampled PROPORTIONAL TO DRAW FREQUENCY (what "bank with no filter" means),
               via Efraimidis-Spirakis weighted shuffle, same cap.
All arms are TRIMMED to the same size (hard, not a warning -- matched optimizer steps at fixed epochs).
full_pass is read by exec's keep-test and for post-hoc purity REPORTING only elsewhere.
Writes arms_summary{suf}.json including the run/stop GATE stats (pool AUROC, purities, n)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
from harvest_pool import auroc  # noqa: E402


def greedy_take(cands, n_total, cap):
    kept, per_task = [], Counter()
    for x in cands:
        if len(kept) >= n_total:
            break
        if per_task[x["ti"]] >= cap:
            continue
        kept.append(x)
        per_task[x["ti"]] += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-task", type=int, default=12)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score-key", type=str, default="p_true_think",
                    help="verifier-free ranking signal (post-diagnostic: think-P(True); no-think is within-task chance here)")
    args = ap.parse_args()
    suf = "_smoke" if args.smoke else ""

    fam = FAM.FAMILIES["list"]
    tasks = [json.loads(l) for l in (EXP / "data" / f"train_tasks{suf}.jsonl").read_text().splitlines() if l.strip()]
    pool = [json.loads(l) for l in (EXP / "data" / f"pool{suf}.jsonl").read_text().splitlines() if l.strip()]
    prompt_of = {t_i: C.ident_prompt(fam, t) for t_i, t in enumerate(tasks)}

    # exec arm: C18-identical keep rule (most-frequent order, full_pass, cap)
    by_task = {}
    for x in pool:
        by_task.setdefault(x["ti"], []).append(x)
    exec_arm = []
    for ti, xs in sorted(by_task.items()):
        kept = 0
        for x in sorted(xs, key=lambda x: -x["freq"]):
            if kept >= args.cap_per_task:
                break
            if x["full_pass"]:
                exec_arm.append(x)
                kept += 1
    n = len(exec_arm)

    # conf_global: naive global top-P(True) (freq tiebreak), same cap, matched size
    sk = args.score_key
    assert all(sk in x for x in pool), f"pool missing {sk} -- run judge_pool_think.py first"
    conf_global = greedy_take(sorted(pool, key=lambda x: (-x[sk], -x["freq"])), n, args.cap_per_task)

    # conf_strat (PRIMARY): per-depth quotas proportional to pool JUDGE-SCORE MASS (no oracle), top-P(True) within
    pool_by_depth = Counter(x["depth"] for x in pool)
    depths = sorted(pool_by_depth)
    mass = {d: sum(x[sk] for x in pool if x["depth"] == d) for d in depths}  # judge-mass, not candidate count:
    tot_mass = sum(mass.values())                                            # wrong candidates EXPLODE at hard
    quota = {d: int(round(n * mass[d] / tot_mass)) for d in depths}          # depths (attempt-2 lesson)
    drift = n - sum(quota.values())
    if drift:
        quota[max(depths, key=lambda d: mass[d])] += drift
    conf_strat = []
    for d in depths:
        cands = sorted([x for x in pool if x["depth"] == d], key=lambda x: (-x[sk], -x["freq"]))
        conf_strat += greedy_take(cands, quota[d], args.cap_per_task)

    # rand: draw-frequency-weighted shuffle (Efraimidis-Spirakis), same cap, matched size
    rr = random.Random(args.seed)
    shuf = sorted(pool, key=lambda x: rr.random() ** (1.0 / max(1, x["freq"])))
    rand_arm = greedy_take(shuf, n, args.cap_per_task)

    arms = {"exec": exec_arm, "conf_strat": conf_strat, "conf_global": conf_global, "rand": rand_arm}
    n_final = min(len(a) for a in arms.values())
    if n_final < n:
        print(f"NOTE: trimming ALL arms {n} -> {n_final} (matched size is hard, per C23 count-confound)")
        for tag in arms:
            trim = arms[tag][:]
            random.Random(9).shuffle(trim)
            arms[tag] = trim[:n_final]
    if n_final == 0:
        sys.exit("FATAL: empty exec arm -- pool has no verified solutions")

    key = lambda x: (x["ti"], x["code"])  # noqa: E731
    exec_keys, strat_keys = {key(x) for x in arms["exec"]}, {key(x) for x in arms["conf_strat"]}
    ys = [int(x["full_pass"]) for x in pool]
    ss = [x[sk] for x in pool]
    within = []
    for ti, xs in by_task.items():
        yy = [int(x["full_pass"]) for x in xs]
        if 0 < sum(yy) < len(yy):
            a = auroc([x[sk] for x in xs], yy)
            if a is not None:
                within.append(a)

    summary = {"score_key": sk, "gate": {
        "pool_n": len(pool), "pool_pass_rate": round(sum(ys) / max(1, len(ys)), 3),
        "pool_auroc_pooled": (lambda a: round(a, 3) if a is not None else None)(auroc(ss, ys)),
        "pool_auroc_within_mean": round(sum(within) / len(within), 3) if within else None,
        "n_mixed_tasks": len(within), "n_pairs": n_final,
        "score_by_depth": {str(d): round(sum(x[sk] for x in pool if x["depth"] == d)
                                         / max(1, pool_by_depth[d]), 3) for d in depths},
        "quota_conf_strat": {str(d): quota[d] for d in depths},
        "exec_conf_strat_jaccard": round(len(exec_keys & strat_keys) / max(1, len(exec_keys | strat_keys)), 3),
    }}
    for tag, arm in arms.items():
        pairs = [{"prompt": prompt_of[x["ti"]], "code": x["code"], "depth": x["depth"]} for x in arm]
        random.Random(7).shuffle(pairs)
        (EXP / "data" / f"train_{tag}{suf}.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
        summary[tag] = {
            "n_pairs": len(arm),
            "purity_full_pass": round(sum(x["full_pass"] for x in arm) / max(1, len(arm)), 3),
            "visible_pass_rate": round(sum(x["visible_pass"] for x in arm) / max(1, len(arm)), 3),
            "distinct_tasks": len({x["ti"] for x in arm}),
            "depth_mix": dict(Counter(x["depth"] for x in arm)),
            "mean_score": round(sum(x[sk] for x in arm) / max(1, len(arm)), 3),
            "min_score_kept": round(min((x[sk] for x in arm), default=0.0), 3),
            "mean_p_true_nothink": round(sum(x["p_true"] for x in arm) / max(1, len(arm)), 3),
            "mean_freq": round(sum(x["freq"] for x in arm) / max(1, len(arm)), 1),
        }
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / f"arms_summary{suf}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote data/train_{{{','.join(arms)}}}{suf}.jsonl ({n_final} pairs each)")


if __name__ == "__main__":
    main()
