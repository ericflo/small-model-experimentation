"""Real-codebase agentic coding through pi + execution selection (the goal's core deliverable).

The synthetic-scenario line completed: through pi the warm-start deploys at 0.62 single-shot / 0.82
execution-selected, and every holdout task proved solvable. REAL-repo tasks (toolz stub-a-function)
were only ever measured in OUR harness — where they scored ~0% — and our harness is now proven to
undersell this model ~1.7x (0.486 vs 0.810 on identical tasks). This runs them through the
deployment-truth scaffold for the first time: pi drives the model inside a real repo checkout, the
repo's own pytest suite is the verifier, and execution-selected best-of-N is the deploy protocol
confirmed by C63.

FIREWALL: the warm-start's SFT data is 100% self-contained synthetic trajectories; no repo task was
ever trained on. All 67 toolz tasks are evaluation-clean.

Mechanics are reused, not re-invented: CodingEnv.reset() does the checkout copy, AST function-stub,
and venv PATH wiring exactly as the verified harvest pipeline did (stubbing was two-side validated at
task-generation time: every task's tests fail when stubbed); CodingEnv.get_reward() scores the
edited checkout with the repo's pytest (1.0 full pass, partial credit below). pi runs headless with
cwd = the stubbed checkout, so its read/edit/write/bash tools operate on the real files.

Checkpoints every 5 episodes and resumes from its own partial file (same crash discipline as
pi_episode.py — these evals have been killed by two full-system crashes).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"
STORE = ROOT / "large_artifacts" / "_taskrepos"

from coding_env import CodingEnv  # noqa: E402
from pi_episode import PI_BIN, NODE_BIN_DIR  # noqa: E402


def task_prompt(t):
    rel, fn = t["rel_file"], t["func_name"]
    return (f"You are in the project root directory (do not cd). The function `{fn}` in `{rel}` "
            f"has had its body replaced with `raise NotImplementedError`. Implement `{fn}` so the "
            f"test suite passes. Read its tests, implement it, and run `python -m pytest -q` to "
            f"verify. Only edit `{rel}`.")


def run_repo_episode(task, provider, model, timeout):
    env = CodingEnv()
    env.reset(repo_dir=str(STORE / task["repo"]),
              rel_file=task["rel_file"], func_name=task["func_name"],
              python=str(STORE / task["repo"] / ".venv-test" / "bin" / "python"),
              test_cmd="python -m pytest -q")
    try:
        cmd = [PI_BIN, "--provider", provider, "--model", model, "-p", task_prompt(task),
               "--mode", "json", "--no-session"]
        # pi inherits this env: CodingEnv wired the repo's real test venv onto PATH, so the model's
        # `python -m pytest` resolves inside the checkout exactly as the verified pipeline runs it.
        penv = {**env.env, "PATH": NODE_BIN_DIR + os.pathsep + env.env.get("PATH", "")}
        t0 = time.time()
        code = 0
        try:
            subprocess.run(cmd, cwd=str(env.project), env=penv, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            code = 124
        reward = env.get_reward()
        edited = "raise NotImplementedError  # TODO: implement" not in (
            env.project / task["rel_file"]).read_text(errors="replace")
        return {"id": task["func_name"], "reward": reward, "edited": edited, "exit": code,
                "secs": round(time.time() - t0, 1)}
    finally:
        env._cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="kiln-proxy")
    ap.add_argument("--model", default="qwen35-4b-pi8k")
    ap.add_argument("--n-tasks", type=int, default=16, help="smallest-body tasks first (as calibrated)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--tasks-file", default=str(OUTD / "toolz_tasks.json"))
    ap.add_argument("--out", default=str(OUTD / "reports" / "pi_repo_eval.json"))
    ap.add_argument("--label", default="repo_warmstart")
    a = ap.parse_args()

    tasks = sorted(json.load(open(a.tasks_file)), key=lambda t: t["body_lines"])[: a.n_tasks]

    prior = []
    if Path(a.out).exists():
        try:
            prev = json.loads(Path(a.out).read_text())
            if prev.get("partial") and prev.get("label") == a.label:
                prior = prev.get("episodes", [])
        except Exception:
            pass
    done = {}
    for r in prior:
        done[r["id"]] = done.get(r["id"], 0) + 1
    jobs = [t for t in tasks for i in range(a.k) if i >= done.get(t["func_name"], 0)]
    if prior:
        print(f"RESUME: {len(prior)} episodes recovered; {len(jobs)} remaining", flush=True)
    print(f"pi REAL-REPO eval: {len(jobs)} episodes = {len(tasks)} toolz tasks x k={a.k} "
          f"| provider={a.provider} model={a.model}", flush=True)

    def summarize(recs):
        per = {}
        for r in recs:
            per.setdefault(r["id"], []).append(r)
        rates = {t: sum(1 for e in v if e["reward"] == 1.0) / len(v) for t, v in per.items()}
        return per, rates, sum(rates.values()) / max(1, len(rates))

    def save(recs, partial):
        per, rates, mean_pass = summarize(recs)
        sel = sum(1 for v in per.values() if any(e["reward"] == 1.0 for e in v)) / max(1, len(per))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"label": a.label, "provider": a.provider, "model": a.model, "k": a.k,
             "partial": partial, "n_episodes": len(recs), "rates": rates,
             "mean_pass": mean_pass, "selected": sel, "episodes": recs}, indent=1))

    recs = list(prior)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_repo_episode, t, a.provider, a.model, a.timeout) for t in jobs]
        t0 = time.time()
        for i, f in enumerate(as_completed(futs), 1):
            try:
                recs.append(f.result())
            except Exception as exc:
                print(f"  episode error: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} ({round(time.time()-t0)}s)", flush=True)
                save(recs, partial=True)

    per, rates, mean_pass = summarize(recs)
    print("\n=== pi REAL-REPO per-task pass rate ===", flush=True)
    for t, v in sorted(rates.items(), key=lambda x: -x[1]):
        mr = sum(e["reward"] for e in per[t]) / len(per[t])
        print(f"  {t:26s} pass {v:.2f}  meanR {mr:.2f}  (n={len(per[t])})", flush=True)
    sel = sum(1 for v in per.values() if any(e["reward"] == 1.0 for e in v)) / max(1, len(per))
    print(f"\nMEAN pass (single-shot): {mean_pass:.3f} over {len(rates)} tasks", flush=True)
    print(f"EXECUTION-SELECTED best-of-{a.k}: {sel:.3f}", flush=True)
    print(f"engaged (edited): {sum(r['edited'] for r in recs)}/{len(recs)}", flush=True)
    save(recs, partial=False)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
