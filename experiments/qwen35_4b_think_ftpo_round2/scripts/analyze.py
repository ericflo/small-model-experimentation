#!/usr/bin/env python3
"""Regenerate preregistered round-2 statistics, gates, and headline table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
ARMS = ("base", "demote", "uplift", "uplift_shuffled")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def paired_delta(a_rows: list[dict], b_rows: list[dict], key: str,
                 seed: int = 4407, n_boot: int = 10000) -> dict:
    a = {r["item_id"] if "item_id" in r else r["task_id"]: float(r[key]) for r in a_rows}
    b = {r["item_id"] if "item_id" in r else r["task_id"]: float(r[key]) for r in b_rows}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no paired ids")
    diff = np.asarray([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {"n": len(diff), "delta": float(diff.mean()),
            "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}


def c49_pass(arm: str) -> bool:
    base = read(EXP / "runs" / "gate_base.json")["outputs"]
    trained = read(EXP / "runs" / f"gate_{arm}.json")["outputs"]
    return any(base[key] != trained.get(key) for key in base)


def relative_ok(base: float, value: float, max_drop: float) -> bool:
    if base == 0:
        return value >= base
    return (base - value) / base <= max_drop


def main() -> int:
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    ev = cfg["eval"]; acfg = cfg["repo_agent"]
    geometry = read(EXP / "runs" / "row_geometry.json")
    audits = {arm: read(EXP / "runs" / f"logit_audit_{arm}.json")
              for arm in ARMS if arm != "base"}
    white = {arm: read(EXP / "runs" / f"whitebox_{arm}_main.json") for arm in ARMS}
    collapse = {arm: read(EXP / "runs" / f"whitebox_{arm}_collapse.json") for arm in ARMS}
    nothink = {arm: read(EXP / "runs" / f"whitebox_{arm}_nothink.json") for arm in ARMS}
    gym = {arm: read(EXP / "runs" / f"eval_gym_{arm}.json") for arm in ARMS}
    repo = {arm: read(EXP / "runs" / f"repo_agent_{arm}_deep.json") for arm in ARMS}
    repo_sample = read(EXP / "runs" / "repo_agent_base_sample_more.json")

    white_deltas = {}
    termination = {}
    for budget in ev["whitebox_budgets"]:
        label = f"think@{budget}"
        white_deltas[label] = {
            arm: paired_delta(white["base"][label + "_rows"],
                              white[arm][label + "_rows"], "score")
            for arm in ARMS if arm != "base"
        }
        white_deltas[label]["uplift_vs_shuffled"] = paired_delta(
            white["uplift_shuffled"][label + "_rows"],
            white["uplift"][label + "_rows"], "score")
        termination[label] = {
            arm: {key: white[arm][label][key] for key in
                  ("natural_close_rate", "answer_limit_rate", "loop_rate",
                   "unresolved_rate", "mean_think_tokens")}
            for arm in ARMS
        }

    repo_rows = {arm: repo[arm]["aggregate"]["tasks"] for arm in ARMS}
    repo_delta = {
        arm: paired_delta(repo_rows["base"], repo_rows[arm], "success")
        for arm in ARMS if arm != "base"
    }
    repo_delta["uplift_vs_shuffled"] = paired_delta(
        repo_rows["uplift_shuffled"], repo_rows["uplift"], "success")
    repo_delta["uplift_vs_sample_more"] = paired_delta(
        repo_sample["aggregate"]["tasks"], repo_rows["uplift"], "success")

    p0 = bool(geometry["gate_pass"])
    p1 = {
        arm: (audits[arm]["overall"]["objective_hit_rate"] >= .35
              and audits[arm]["overall"]["median_abs_nontarget_drift_mean"] <= .10)
        for arm in audits
    }
    p2_by_budget = {}
    for budget in ev["whitebox_budgets"]:
        label = f"think@{budget}"
        base_term = termination[label]["base"]
        up_term = termination[label]["uplift"]
        p2_by_budget[label] = (
            white_deltas[label]["uplift"]["delta"] >= float(ev["whitebox_success_bar"])
            and white_deltas[label]["uplift_vs_shuffled"]["delta"]
            >= float(ev["whitebox_control_separation"])
            and up_term["natural_close_rate"]
            >= base_term["natural_close_rate"] - float(ev["natural_close_max_drop"])
            and up_term["answer_limit_rate"]
            <= base_term["answer_limit_rate"] + float(ev["answer_limit_max_rise"])
        )
    p2 = any(p2_by_budget.values())
    uplift_gain = repo_delta["uplift"]["delta"]
    shuffled_gain = repo_delta["uplift_shuffled"]["delta"]
    p3 = (
        uplift_gain >= float(acfg["uplift_vs_base_bar"])
        and repo_delta["uplift_vs_sample_more"]["delta"]
        >= float(acfg["uplift_vs_sample_more_bar"])
        and shuffled_gain < float(acfg["control_gain_fraction_max"]) * uplift_gain
    )

    guards = {}
    for arm in ARMS:
        if arm == "base":
            continue
        guards[arm] = {
            "c49": c49_pass(arm),
            "gym": gym[arm]["aggregate_all"]
            >= gym["base"]["aggregate_all"] - float(ev["gym_guard_max_drop"]),
            "collapse_greedy": relative_ok(
                collapse["base"]["greedy"]["success"], collapse[arm]["greedy"]["success"],
                float(ev["collapse_guard_max_rel_drop"])),
            "collapse_passk": relative_ok(
                collapse["base"][f"pass@{ev['collapse_guard_pass_k']}"],
                collapse[arm][f"pass@{ev['collapse_guard_pass_k']}"],
                float(ev["collapse_guard_max_rel_drop"])),
            "nothink": nothink[arm]["nothink"]["success"]
            >= nothink["base"]["nothink"]["success"] - float(ev["nothink_guard_max_drop"]),
        }
        guards[arm]["all"] = all(guards[arm].values())
    p4 = guards["uplift"]["all"] and all(guards[a]["c49"] for a in guards)
    menagerie_eligible = (p2 or p3) and p4 and p1["uplift"]

    menagerie = {"eligible": menagerie_eligible, "events": [], "P5": None}
    log = EXP / "runs" / "menagerie_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if "base" in event["arms"] and "uplift" in event["arms"]:
                menagerie["events"].append({"seed": event["seed"], "delta": event["delta"]})
        if len(menagerie["events"]) >= len(cfg["menagerie"]["quick_seeds"]):
            deltas = [e["delta"] for e in menagerie["events"][-2:]]
            menagerie["mean_delta"] = float(np.mean(deltas))
            menagerie["P5"] = (
                menagerie["mean_delta"] >= float(cfg["menagerie"]["positive_mean_bar"])
                and (not cfg["menagerie"]["require_nonnegative_each_seed"]
                     or all(delta >= 0 for delta in deltas)))

    if not p0:
        verdict = "GEOMETRY_FAIL"
    elif menagerie.get("P5"):
        verdict = "POSITIVE"
    elif menagerie_eligible:
        verdict = "CAPABILITY_CANDIDATE"
    else:
        # Generic-harm is deliberately conservative: both uplift labels must
        # hurt similarly at both budgets or on the agent task.
        harms = []
        for budget in ev["whitebox_budgets"]:
            label = f"think@{budget}"
            harms.append(white_deltas[label]["uplift"]["delta"] <= -.02
                         and abs(white_deltas[label]["uplift_vs_shuffled"]["delta"]) <= .02)
        harms.append(uplift_gain <= -.02
                     and abs(repo_delta["uplift_vs_shuffled"]["delta"]) <= .02)
        verdict = "GENERIC_TRAINING_HARM" if any(harms) else "LOW_DOSE_NULL"

    summary = {
        "verdict": verdict, "geometry": geometry, "logit_audits": audits,
        "whitebox": {"deltas": white_deltas, "termination": termination},
        "repo_agent": {
            "aggregates": {arm: repo[arm]["aggregate"] for arm in ARMS},
            "sample_more": repo_sample["aggregate"], "deltas": repo_delta},
        "gym": {arm: {k: gym[arm][k] for k in
                       ("aggregate_all", "aggregate_trained", "aggregate_heldout")}
                for arm in ARMS},
        "guards": guards,
        "gates": {"P0": p0, "P1": p1, "P2": p2,
                  "P2_by_budget": p2_by_budget, "P3": p3, "P4": p4,
                  "menagerie_eligible": menagerie_eligible},
        "menagerie": menagerie,
    }
    analysis = EXP / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Round-2 headline", "", f"**Verdict: {verdict}.**", "",
             "## Whitebox success", "",
             "| budget | base | demote | uplift | shuffled | uplift-base | uplift-shuffled |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for budget in ev["whitebox_budgets"]:
        label = f"think@{budget}"
        lines.append(
            f"| {label} | {white['base'][label]['success']:.3f} | "
            f"{white['demote'][label]['success']:.3f} | {white['uplift'][label]['success']:.3f} | "
            f"{white['uplift_shuffled'][label]['success']:.3f} | "
            f"{white_deltas[label]['uplift']['delta']:+.3f} | "
            f"{white_deltas[label]['uplift_vs_shuffled']['delta']:+.3f} |")
    lines += ["", "## Repository agent", "",
              "| arm | success | patch-correct | submit | mean sampled tokens |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        agg = repo[arm]["aggregate"]
        lines.append(f"| {arm} | {agg['success']:.3f} | {agg['patch_correct']:.3f} | "
                     f"{agg['submit_rate']:.3f} | {agg['mean_sampled_tokens']:.0f} |")
    agg = repo_sample["aggregate"]
    lines.append(f"| base sample-more | {agg['success']:.3f} | {agg['patch_correct']:.3f} | "
                 f"{agg['submit_rate']:.3f} | {agg['mean_sampled_tokens']:.0f} |")
    lines += ["", "## Gates", "", "```json",
              json.dumps(summary["gates"], indent=2), "```", ""]
    (analysis / "headline.md").write_text("\n".join(lines))
    print(json.dumps({"verdict": verdict, **summary["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
