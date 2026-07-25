"""Two-side-validate stub-a-function tasks across every GREEN repo, in parallel.

A candidate becomes a task only if BOTH sides hold, each measured in a throwaway copy built exactly
as an episode will be built (env_util):
  * UNSTUBBED: the repo's suite passes (established once per repo by fetch_repos.py's green gate);
  * STUBBED:   replacing this one function body makes the suite FAIL.

The second gate is what makes the instrument trustworthy. It rejects functions that no test exercises
(a task the agent cannot be scored on), and — critically — it is the detector for the
editable-install import leak described in env_util: if pytest were importing the original checkout
instead of the episode's copy, stubbing would break nothing and EVERY candidate would be rejected.
So a silent leak shows up as "this repo produced 0 tasks", never as free reward.

Recorded per task: body_lines, n_failing_when_stubbed, and the failing-test node ids (truncated).
Those support difficulty stratification later WITHOUT re-running anything, and the node ids let an
analysis check that a solve actually fixed the tests the stub broke.
"""
import argparse
import json
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
DATA = HERE.parent / "data"

import env_util  # noqa: E402
import stub as stubmod  # noqa: E402


_BASELINES = {}


def _baseline(path, repo_name):
    """Load repo baselines once per worker process (they are large: ~1k node ids per repo)."""
    if not _BASELINES:
        _BASELINES.update(json.load(open(path)))
    return _BASELINES[repo_name]


def validate(job):
    """Stub one candidate in a fresh copy and derive its fail_to_pass set against the baseline.

    fail_to_pass = tests that PASS in the untouched baseline but FAIL once this function is stubbed.
    Intersecting with the baseline is what makes the set trustworthy: a test that was already failing
    for environment reasons is excluded, so it can neither create a phantom task nor cap an episode's
    reward. If that intersection is empty the candidate is rejected — which is also the detector for
    an import-resolution leak (if pytest were importing the ORIGINAL checkout instead of this copy,
    stubbing would break nothing and the repo would yield zero tasks instead of free rewards).
    """
    repo_name, cand, suite_timeout, baselines_path = job
    repo = STORE / repo_name
    base_pass = set(_baseline(baselines_path, repo_name)["pass"])
    copy = env_util.prepare_copy(repo, prefix=f"taskgen_{repo_name}_")
    try:
        if not stubmod.stub_function(copy, cand["rel_file"], cand["qual_name"]):
            return {**cand, "repo": repo_name, "admit": False, "reason": "stub_failed"}
        res = env_util.run_tests(copy, repo, timeout=suite_timeout)
        broken_now = {k for k, v in res["statuses"].items() if v in ("FAILED", "ERROR", "SUBFAIL")}
        ftp = sorted(broken_now & base_pass)
        rec = {**cand, "repo": repo_name, "admit": bool(ftp),
               "failing_ids": ftp[:40],
               "n_fail_to_pass": len(ftp),
               "n_pass_to_pass": len(base_pass) - len(ftp),
               "n_broken_outside_baseline": len(broken_now - base_pass)}
        if not ftp:
            # Distinguish "no test covers this" from "the suite could not run" — the latter is an
            # instrument fault (bad env / timeout / crash), not a property of the candidate.
            rec["reason"] = ("suite_unrunnable"
                             if res["code"] in (124, 137) or not res["statuses"]
                             else "not_test_covered")
        return rec
    except Exception as exc:                                   # pragma: no cover - defensive
        return {**cand, "repo": repo_name, "admit": False,
                "reason": f"exception:{type(exc).__name__}:{str(exc)[:120]}"}
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "repo_manifest.json"))
    ap.add_argument("--baselines", default=str(DATA / "repo_baselines.json"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--suite-timeout", type=int, default=300)
    ap.add_argument("--min-body-lines", type=int, default=3)
    ap.add_argument("--max-body-lines", type=int, default=120)
    ap.add_argument("--max-per-repo", type=int, default=40,
                    help="cap candidates per repo so one large library cannot dominate the set")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(DATA / "tasks_all.json"))
    a = ap.parse_args()

    manifest = json.load(open(a.manifest))["repos"]
    green = [r for r in manifest if r.get("verdict") == "usable"]
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        green = [r for r in green if r["name"] in want]
    if not green:
        raise SystemExit(f"no usable repos in {a.manifest} -- run fetch_repos.py first "
                         f"(verdicts present: {sorted({r.get('verdict') for r in manifest})})")
    print(f"enumerating candidates in {len(green)} usable repos", flush=True)

    jobs, per_repo = [], {}
    for r in green:
        cands = []
        for rel in r["src"]:
            f = STORE / r["name"] / rel
            if not f.exists():
                print(f"  WARN {r['name']}: missing src {rel}", flush=True)
                continue
            cands += stubmod.candidates(f, rel, a.min_body_lines, a.max_body_lines)
        # Deterministic, difficulty-spread selection: sort by body size and take an even stride, so
        # the cap keeps small AND large functions instead of only the alphabetically-first ones.
        cands.sort(key=lambda c: (c["body_lines"], c["rel_file"], c["qual_name"]))
        if len(cands) > a.max_per_repo:
            step = len(cands) / a.max_per_repo
            cands = [cands[int(i * step)] for i in range(a.max_per_repo)]
        per_repo[r["name"]] = len(cands)
        jobs += [(r["name"], c, a.suite_timeout, a.baselines) for c in cands]
        print(f"  {r['name']:20s} {len(cands):3d} candidates", flush=True)

    print(f"\nvalidating {len(jobs)} candidates with {a.jobs} workers "
          f"(each runs the repo's full suite once)", flush=True)
    out_recs, t0 = [], time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(validate, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                out_recs.append(f.result())
            except Exception as exc:
                print(f"  worker error: {type(exc).__name__}: {exc}", flush=True)
            if i % 25 == 0 or i == len(jobs):
                adm = sum(1 for r in out_recs if r["admit"])
                print(f"  {i}/{len(jobs)} validated | admitted {adm} "
                      f"({round(time.time()-t0)}s)", flush=True)
                Path(a.out).write_text(json.dumps({"partial": i < len(jobs), "candidates": out_recs}, indent=1))

    admitted = [r for r in out_recs if r["admit"]]
    print(f"\n=== ADMITTED {len(admitted)}/{len(out_recs)} candidates ===", flush=True)
    from collections import Counter
    by_repo = Counter(r["repo"] for r in admitted)
    reasons = Counter(r.get("reason") for r in out_recs if not r["admit"])
    for name in sorted(per_repo):
        print(f"  {name:20s} {by_repo.get(name,0):3d} tasks / {per_repo[name]:3d} candidates", flush=True)
    print(f"  reject reasons: {dict(reasons)}", flush=True)
    zero = [n for n in per_repo if by_repo.get(n, 0) == 0 and per_repo[n] > 0]
    if zero:
        print(f"  !! ZERO-TASK REPOS (suspect import leak or unrunnable suite): {zero}", flush=True)
    Path(a.out).write_text(json.dumps(
        {"partial": False, "candidates": out_recs,
         "counts": {"admitted": len(admitted), "validated": len(out_recs),
                    "by_repo": dict(by_repo), "reject_reasons": dict(reasons)}}, indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
