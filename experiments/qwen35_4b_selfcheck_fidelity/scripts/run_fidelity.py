#!/usr/bin/env python3
"""Score every candidate against BOTH verifiers -- self-written checks and the hidden repo suite --
on the SAME reconstructed artifact, then measure fidelity.

PIPELINE per (task, candidate): fresh checkout copy -> stub the target (recreating the byte-exact
old_text the agent's edit matched) -> replace old_text with the candidate implementation -> run
  (a) the task's fail_to_pass tests from the repo's own suite  = HIDDEN verdict
  (b) the model's self-written check file                      = SELF verdict
Both verdicts come from one artifact, so reconstruction error cannot split label from measurement.

SUITE QUALITY GATES, computed model-free per task before any fidelity number:
  * fails-on-stub: the suite run against the stubbed function must FAIL (else it is vacuous);
  * passes-on-original: run against the repo's true implementation it must PASS (else it contradicts
    actual behavior -- wrong, not strict).
Suites are scored on all candidates regardless, but headline fidelity is reported for gate-passing
suites, with the gate-pass rate itself a primary result (an everyday-task assistant only helps if its
self-checks are usually valid).

METRICS: per-candidate agreement and AUROC of self-check pass-fraction against the hidden verdict
(comparators: the direct judge P(True) AUROC 0.679 and the length heuristic 0.652 measured on these
same trajectories); and SELECTION -- per task, does argmax-self-check pick a hidden-passing candidate
better than random and better than the judge picked?

ACTIVITY note per the new quality gate: this is not a scaffold arm (no agent runs); the only compute
is pytest, capped and spooled via env_util.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
DATA = HERE.parent / "data"

import env_util  # noqa: E402
import stub as stubmod  # noqa: E402

CHECK_FILE = "selfcheck_test.py"


def prepare_candidate(task_row, impl):
    """Fresh copy with the candidate implementation in place. None if reconstruction fails."""
    repo = STORE / task_row["repo"]
    copy = env_util.prepare_copy(repo, prefix=f"fid_{task_row['repo']}_")
    if not stubmod.stub_function(copy, task_row["rel_file"], task_row["qual_name"]):
        shutil.rmtree(copy, ignore_errors=True)
        return None
    target = copy / task_row["rel_file"]
    src = target.read_text(errors="replace")
    if task_row["old_text"] not in src:
        shutil.rmtree(copy, ignore_errors=True)
        return None
    target.write_text(src.replace(task_row["old_text"], impl))
    return copy


def hidden_verdict(copy, repo, ftp):
    files = sorted({nid.split("::")[0] for nid in ftp if "::" in nid})
    cmd = f"python -m pytest {' '.join(files)} -v -rA --tb=line -p no:cacheprovider"
    res = env_util.run_tests(copy, repo, timeout=180, cmd=cmd)
    ok = env_util.passing_ids(res["statuses"])
    return all(nid in ok for nid in ftp), res


def self_verdict(copy, repo, check_src):
    (copy / CHECK_FILE).write_text(check_src)
    cmd = f"python -m pytest {CHECK_FILE} -v -rA --tb=line -p no:cacheprovider"
    res = env_util.run_tests(copy, repo, timeout=120, cmd=cmd)
    n_pass = res["passed"]
    n_total = res["passed"] + res["failed"] + res["errors"]
    return (n_total > 0 and res["failed"] + res["errors"] == 0), \
        (n_pass / n_total if n_total else 0.0), res


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return 0.5
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(DATA / "pairs.jsonl"))
    ap.add_argument("--checks", default=str(DATA / "checks"))
    ap.add_argument("--out", default=str(HERE.parent / "reports" / "fidelity.json"))
    a = ap.parse_args()

    pairs = [json.loads(l) for l in open(a.pairs)]
    by_task = defaultdict(list)
    for p in pairs:
        by_task[p["task_id"]].append(p)

    # ---- suite quality gates ---------------------------------------------------------------------
    gates, rows = {}, []
    for tid, plist in sorted(by_task.items()):
        t = plist[0]
        fn = Path(a.checks) / (tid.replace("/", "-").replace("::", "__") + ".py")
        if not fn.exists():
            gates[tid] = {"has_suite": False}
            continue
        check_src = fn.read_text()
        repo = STORE / t["repo"]
        # gate 1: fails on stub
        copy = prepare_candidate(t, t["old_text"])            # old_text IS the stubbed region
        if copy is None:
            gates[tid] = {"has_suite": True, "recon_failed": True}
            continue
        try:
            stub_pass, _, _ = self_verdict(copy, repo, check_src)
        finally:
            shutil.rmtree(copy, ignore_errors=True)
        # gate 2: passes on the ORIGINAL implementation (an untouched copy)
        orig = env_util.prepare_copy(repo, prefix=f"orig_{t['repo']}_")
        try:
            orig_pass, orig_frac, _ = self_verdict(orig, repo, check_src)
        finally:
            shutil.rmtree(orig, ignore_errors=True)
        gates[tid] = {"has_suite": True, "fails_on_stub": not stub_pass,
                      "passes_on_original": orig_pass, "orig_frac": round(orig_frac, 3),
                      "valid": (not stub_pass) and orig_pass}
        print(f"  gate {tid[:56]:56s} stub-fail={not stub_pass} orig-pass={orig_pass}", flush=True)

    n_valid = sum(1 for g in gates.values() if g.get("valid"))
    print(f"\nSUITE GATES: {n_valid}/{len(gates)} valid "
          f"(fails-on-stub AND passes-on-original)", flush=True)

    # ---- score every candidate on both verifiers -------------------------------------------------
    for tid, plist in sorted(by_task.items()):
        g = gates.get(tid, {})
        if not g.get("has_suite") or g.get("recon_failed"):
            continue
        t = plist[0]
        repo = STORE / t["repo"]
        check_src = (Path(a.checks) / (tid.replace("/", "-").replace("::", "__") + ".py")).read_text()
        for p in plist:
            copy = prepare_candidate(t, p["impl"])
            if copy is None:
                continue
            try:
                hid, _ = hidden_verdict(copy, repo, t["fail_to_pass"])
                sv, sfrac, _ = self_verdict(copy, repo, check_src)
            finally:
                shutil.rmtree(copy, ignore_errors=True)
            rows.append({"task_id": tid, "pair_id": p["pair_id"], "suite_valid": bool(g.get("valid")),
                         "hidden": bool(hid), "self_pass": bool(sv), "self_frac": round(sfrac, 3),
                         "episode_solved": p["episode_solved"], "impl_len": len(p["impl"])})
        done = len({r['task_id'] for r in rows})
        print(f"  scored task {done}/{n_valid if n_valid else len(by_task)}: {tid[:48]}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"gates": gates, "rows": rows}, indent=1))

    # ---- metrics ---------------------------------------------------------------------------------
    def report(sub, label):
        if not sub:
            print(f"  [{label}] no rows", flush=True)
            return {}
        hid = [r["hidden"] for r in sub]
        agree = sum(r["self_pass"] == r["hidden"] for r in sub) / len(sub)
        au = auroc([r["self_frac"] for r in sub], hid)
        lena = auroc([-r["impl_len"] for r in sub], hid)
        base_rate = max(sum(hid), len(hid) - sum(hid)) / len(hid)
        m = {"n": len(sub), "hidden_pass_rate": round(sum(hid) / len(sub), 3),
             "agreement": round(agree, 3), "auroc_self_frac": round(au, 4),
             "auroc_neg_length": round(lena, 4), "majority": round(base_rate, 3)}
        print(f"  [{label}] n={m['n']} agree={m['agreement']} AUROC(self)={m['auroc_self_frac']} "
              f"vs len {m['auroc_neg_length']} vs majority {m['majority']}", flush=True)
        return m

    print("\n=== FIDELITY (self-check vs hidden suite, same artifact) ===", flush=True)
    res = {"gates_valid": n_valid, "gates_total": len(gates),
           "all": report(rows, "all suites"),
           "valid_suites": report([r for r in rows if r["suite_valid"]], "gate-passing suites")}

    # selection: per task with >=2 candidates and >=1 hidden-passing, does argmax self_frac find one?
    sel_hits = sel_n = rand_exp = 0
    for tid, plist in by_task.items():
        sub = [r for r in rows if r["task_id"] == tid and r["suite_valid"]]
        if len(sub) < 2 or not any(r["hidden"] for r in sub):
            continue
        sel_n += 1
        rand_exp += sum(r["hidden"] for r in sub) / len(sub)
        best = max(sub, key=lambda r: (r["self_frac"], -r["impl_len"]))
        sel_hits += bool(best["hidden"])
    if sel_n:
        res["selection"] = {"tasks": sel_n, "selfcheck_pick_rate": round(sel_hits / sel_n, 3),
                            "random_pick_rate": round(rand_exp / sel_n, 3)}
        print(f"  selection over {sel_n} multi-candidate tasks: self-check pick {sel_hits/sel_n:.3f} "
              f"vs random {rand_exp/sel_n:.3f}", flush=True)
    print(f"  comparators from judge_probe (same trajectories): judge P(True) AUROC 0.679, "
          f"length 0.652", flush=True)
    Path(a.out).write_text(json.dumps({"gates": gates, "rows": rows, "metrics": res}, indent=1))
    Path(str(a.out) + ".DONE").write_text(json.dumps(res.get("valid_suites", {})) + "\n")
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
