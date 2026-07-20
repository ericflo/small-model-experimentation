"""Per-task pass-rate calibration for a served policy — find the RLVR-usable difficulty band.

GRPO needs WITHIN-group variance: the same task passing SOMETIMES and failing sometimes. After the
SFT warm-start, engagement is installed but real-repo stub tasks sit at ~0% and self-contained tasks
at ~100% — neither gives advantage signal. This measures pass rate PER TASK over K samples so we can
select the ~30-70% band to train RLVR on (and see whether such a band exists at all).

Requires the policy under test to be SERVED (vLLM OpenAI server on :1234, model id `base`).
"""
import argparse, json, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
TASKS = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "toolz_tasks.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--step-cap", type=int, default=14)
    ap.add_argument("--n-repo", type=int, default=16, help="how many repo stub tasks to sample")
    ap.add_argument("--n-synth", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "difficulty.json"))
    a = ap.parse_args()

    import loop_repo, loop_raw, synth_scenarios

    jobs = []
    repo_tasks = json.load(open(TASKS))
    repo_tasks = sorted(repo_tasks, key=lambda t: t["body_lines"])[: a.n_repo]
    for t in repo_tasks:
        rel, fn = t["rel_file"], t["func_name"]
        task = {"id": f"repo:{fn}", "repo_dir": str(STORE / t["repo"]),
                "python": str(STORE / t["repo"] / ".venv-test" / "bin" / "python"),
                "rel_file": rel, "func_name": fn, "test_cmd": "python -m pytest -q",
                "prompt": (f"You are in the project root directory (do not cd). The function `{fn}` in `{rel}` "
                           f"has had its body replaced with `raise NotImplementedError`. Implement `{fn}` so the "
                           f"test suite passes. Read its tests, implement it, and run `python -m pytest -q` to "
                           f"verify. Only edit `{rel}`.")}
        for _ in range(a.k):
            jobs.append(("repo", task))
    for sc in synth_scenarios.SCENARIOS[: a.n_synth]:
        for _ in range(a.k):
            jobs.append(("synth", sc))

    print(f"{len(jobs)} episodes ({len(repo_tasks)} repo + {min(a.n_synth, len(synth_scenarios.SCENARIOS))} synth) x k={a.k}", flush=True)

    def one(kind, task):
        try:
            if kind == "repo":
                r = loop_repo.run_repo_episode(task, temperature=1.0, step_cap=a.step_cap)
                return (f"repo:{task['func_name']}", bool(r["passed"]))
            r = loop_raw.with_temp_project(task, lambda d: loop_raw.run_episode(task, d, temperature=1.0, step_cap=a.step_cap))
            return (f"synth:{task['id']}", bool(r["passed"]))
        except Exception as e:
            return (f"{kind}:ERR", None)

    res = defaultdict(list)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, k, t) for k, t in jobs]
        for f in as_completed(futs):
            tid, ok = f.result(); done += 1
            if ok is not None:
                res[tid].append(ok)
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time()-t0)}s)", flush=True)

    rates = {tid: (sum(v) / len(v), len(v)) for tid, v in sorted(res.items())}
    band = {t: r for t, (r, n) in rates.items() if 0.25 <= r <= 0.75}
    print("\n=== per-task pass rate ===", flush=True)
    for tid, (r, n) in sorted(rates.items(), key=lambda x: -x[1][0]):
        mark = "  <-- RLVR BAND" if 0.25 <= r <= 0.75 else ""
        print(f"  {tid:34s} {r:.2f} ({n} samples){mark}", flush=True)
    print(f"\nRLVR-usable band (0.25-0.75): {len(band)}/{len(rates)} tasks", flush=True)
    Path(a.out).write_text(json.dumps({"rates": {k: v[0] for k, v in rates.items()},
                                       "band": sorted(band)}, indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
