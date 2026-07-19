"""Harvest completed+PASSING agentic trajectories for the SFT warm-start (loop discipline).

Two sources, both execution-verified and self-generated (no teacher):
  --mode repo   : stub-a-function tasks in a real OSS repo (matches the RLVR CodingEnv shape)
  --mode synth  : self-contained tasks (base solves these reliably -> fast, high-yield loop data)

Outputs to large_artifacts/ (durable across scratchpad wipes). Keeps full per-turn data
(reasoning + tool calls + tool results) for conversion to multi-turn tool-calling SFT rows.
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
OUTDIR = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility" / "harvest"


def repo_jobs(tasks_json, K):
    import loop_repo  # noqa
    tasks = json.load(open(tasks_json))
    jobs = []
    for t in tasks:
        rel, fn = t["rel_file"], t["func_name"]
        prompt = (f"You are in the project root directory (do not cd anywhere). The function `{fn}` in "
                  f"`{rel}` has had its body replaced with `raise NotImplementedError`. Implement `{fn}` so "
                  f"the test suite passes. Read its docstring and its tests, implement it, and run "
                  f"`python -m pytest -q` to verify. Only edit `{rel}`.")
        task = {"id": f"{t['repo']}:{fn}", "repo_dir": str(STORE / t["repo"]),
                "python": str(STORE / t["repo"] / ".venv-test" / "bin" / "python"),
                "rel_file": rel, "func_name": fn, "test_cmd": "python -m pytest -q", "prompt": prompt,
                "body_lines": t.get("body_lines", 0)}
        for k in range(K):
            jobs.append(task)
    return jobs


def run_repo_job(task, step_cap):
    import loop_repo
    r = loop_repo.run_repo_episode(task, temperature=0.8, step_cap=step_cap)
    return {"src": "repo", "id": task["id"], "func_name": task["func_name"], "rel_file": task["rel_file"],
            "passed": r["passed"], "produced_edit": r.get("produced_edit"),
            "transcript": r["transcript"], "history": r["history"], "n_steps": r["n_steps"]}


def synth_jobs(K):
    import synth_scenarios
    return [sc for sc in synth_scenarios.SCENARIOS for _ in range(K)]


def run_synth_job(sc, step_cap):
    import loop_raw
    r = loop_raw.with_temp_project(sc, lambda d: loop_raw.run_episode(sc, d, temperature=0.7, step_cap=step_cap))
    return {"src": "synth", "id": sc["id"], "func_name": sc["id"], "rel_file": "solution.py",
            "passed": r["passed"], "transcript": r["transcript"], "history": r["history"], "n_steps": r["n_steps"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repo", "synth"], required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--step-cap", type=int, default=14)
    ap.add_argument("--tasks-json", default=str(OUTDIR.parent / "toolz_tasks.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else OUTDIR / f"harvested_{a.mode}.jsonl"

    jobs = repo_jobs(a.tasks_json, a.k) if a.mode == "repo" else synth_jobs(a.k)
    random.seed(0); random.shuffle(jobs)
    print(f"{len(jobs)} episodes (mode={a.mode}), {a.workers} workers -> {out}", flush=True)

    def one(j):
        try:
            return run_repo_job(j, a.step_cap) if a.mode == "repo" else run_synth_job(j, a.step_cap)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    kept = done = 0
    t0 = time.time()
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result(); done += 1
            if "error" not in r and r["passed"]:
                fh.write(json.dumps(r) + "\n"); fh.flush(); kept += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)} done, {kept} passing kept, {round(time.time()-t0)}s", flush=True)
    print(f"DONE: {kept} passing -> {out} ({round(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    main()
