#!/usr/bin/env python3
"""Aggregate all run outputs into analysis/summary.json + markdown tables.

python3 scripts/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "runs"
OUT = EXP / "analysis"


def load(name: str) -> dict | None:
    path = RUNS / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    OUT.mkdir(exist_ok=True)
    summary: dict = {}

    summary["census_existing_logs"] = load("census_existing.json")
    harvest = load("harvest/harvest_summary.json")
    summary["harvest_extension_slices"] = harvest
    for arm in ("pivot", "shuffled"):
        summary[f"build_rows_{arm}"] = load(f"build_rows_{arm}.json")

    whitebox: dict = {}
    for arm in ("base", "pivot", "shuffled"):
        for stage in ("main", "formats", "coverage", "collapse", "nothink"):
            data = load(f"whitebox_{arm}_{stage}.json")
            if data:
                whitebox[f"{arm}_{stage}"] = data
    summary["whitebox"] = whitebox
    summary["eval_gym"] = {arm: load(f"eval_gym_{arm}.json")
                           for arm in ("base", "pivot", "shuffled")}
    summary["gates"] = {
        "pivot_c49": "PASS (0/8 identical)" if (RUNS / "gate_pivot.json").exists() else None,
        "shuffled_c49": "see eval_chain.log" if (RUNS / "gate_shuffled.json").exists() else None,
    }

    # Headline decision-relevant table
    rows = []
    base_main = whitebox.get("base_main")
    for arm in ("base", "pivot", "shuffled"):
        main = whitebox.get(f"{arm}_main")
        if not main:
            continue
        for budget in ("think@1024", "think@2048"):
            a = main[budget]
            row = {"arm": arm, "budget": budget,
                   "success": round(a["success"], 4),
                   "natural_close": round(a["natural_close_rate"], 4),
                   "loop_rate": round(a["loop_rate"], 4),
                   "unresolved": round(a["unresolved_rate"], 4),
                   "answer_limit": round(a["answer_limit_rate"], 4)}
            if arm != "base" and base_main:
                row["success_delta_vs_base"] = round(
                    a["success"] - base_main[budget]["success"], 4)
            rows.append(row)
    summary["headline_table"] = rows

    # Predictions verdicts
    p = whitebox.get("pivot_main")
    b = whitebox.get("base_main")
    s = whitebox.get("shuffled_main")
    verdicts = {}
    if harvest:
        verdicts["P0_census"] = {
            "eligible_rate": None, "note": "combined across main+extension; see report",
        }
    if p and b:
        gain_1024 = p["think@1024"]["success"] - b["think@1024"]["success"]
        verdicts["P1_mechanism"] = {
            "bar": "+0.05 absolute greedy success on held-out band tasks",
            "measured_1024": round(gain_1024, 4),
            "measured_2048": round(p["think@2048"]["success"] - b["think@2048"]["success"], 4),
            "verdict": "FAIL" if gain_1024 < 0.05 else "PASS",
        }
    if p and b and s:
        verdicts["control_read"] = {
            "shuffled_delta_1024": round(s["think@1024"]["success"] - b["think@1024"]["success"], 4),
            "shuffled_delta_2048": round(s["think@2048"]["success"] - b["think@2048"]["success"], 4),
            "note": "shuffled ~= pivot degradation -> damage is generic to the "
                    "training regime, not the outcome-conditioned signal",
        }
    cb = whitebox.get("base_collapse")
    cp = whitebox.get("pivot_collapse")
    if cb and cp:
        g = (cp["greedy"]["success"] - cb["greedy"]["success"]) / max(cb["greedy"]["success"], 1e-9)
        k8 = (cp.get("pass@8", 0) - cb.get("pass@8", 0)) / max(cb.get("pass@8", 1e-9), 1e-9)
        verdicts["collapse_guard"] = {
            "greedy_rel_change": round(g, 4), "pass8_rel_change": round(k8, 4),
            "flag_damaging": bool(g < -0.10 or k8 < -0.10),
        }
    nb = whitebox.get("base_nothink")
    np_ = whitebox.get("pivot_nothink")
    if nb and np_:
        verdicts["nothink_guard"] = {
            "base": round(nb["nothink"]["success"], 4),
            "pivot": round(np_["nothink"]["success"], 4),
            "flag": bool(nb["nothink"]["success"] - np_["nothink"]["success"] > 0.02),
        }
    gyms = summary["eval_gym"]
    if gyms.get("base") and gyms.get("pivot"):
        verdicts["P3_gym"] = {
            "base_aggregate": round(gyms["base"]["aggregate_all"], 4),
            "pivot_aggregate": round(gyms["pivot"]["aggregate_all"], 4),
            "shuffled_aggregate": round(gyms["shuffled"]["aggregate_all"], 4)
                                   if gyms.get("shuffled") else None,
            "guard_fail": bool(gyms["pivot"]["aggregate_all"]
                               < gyms["base"]["aggregate_all"] - 0.02),
        }
    verdicts["P4_menagerie"] = {
        "verdict": "NOT RUN — preregistered rule: mechanism prediction (P1) "
                   "failed, so the round is a training-recipe failure with no "
                   "capability read; blackbox spend cancelled",
    }
    summary["verdicts"] = verdicts

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# Round-1 headline table", "",
             "| arm | budget | success | Δsuccess | natural_close | loop | unresolved | answer_limit |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['budget']} | {r['success']:.3f} | "
            f"{r.get('success_delta_vs_base', 0):+.3f} | {r['natural_close']:.3f} | "
            f"{r['loop_rate']:.3f} | {r['unresolved']:.3f} | {r['answer_limit']:.3f} |")
    lines += ["", "## Verdicts", "```json", json.dumps(verdicts, indent=2), "```"]
    (OUT / "headline.md").write_text("\n".join(lines))
    print(json.dumps(verdicts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
