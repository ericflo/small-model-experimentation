"""Run real-repo stub-a-function tasks through pi-coding-agent and score them per-test.

DEPLOYMENT TRUTH. Everything measured in a bespoke tool loop is a claim about that loop until it is
reproduced here: three separate "the 4B cannot do X" conclusions in this program reversed when
measured through pi (real-repo ~0.00 -> 0.70; synthetic 0.486 -> 0.810; seven "absent" tasks that were
a dict-default bug). pi is the actual scaffold, with its own system prompt, tool schemas, multi-turn
loop, and truncation behaviour.

TWO DIFFERENCES FROM THE PRIOR RUNNER (experiments/qwen35_4b_agentic_rlvr_feasibility):

1. Output is SPOOLED TO DISK, never captured with `stdout=PIPE`. `pi --mode json` emits one JSON event
   per line including full file contents and tool results; a looping episode (96% of hard-task
   failures are timeouts) streams that for the whole 600 s wall, and with PIPE every byte accumulates
   in THIS process's RSS. That is the mechanism that grew a driver to ~31 GB of anonymous memory and
   killed the WSL VM (docs/wsl_stability.md) — and it is almost certainly what killed it during the
   earlier "sustained GPU load" runs too, since those used the same PIPE capture.
   The spool is not thrown away: it IS the trajectory, kept for the Line-1 harvest.

2. Scoring is per-test (env_util.score_task) against the task's `fail_to_pass` set and the repo's
   `pass_to_pass` baseline, so pre-existing environment-dependent failures cannot cap an episode's
   reward, and "fix the test instead of the code" cannot raise it.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
OUTD = ROOT / "large_artifacts" / "qwen35_4b_realrepo_agentic_instrument"
DATA = HERE.parent / "data"

import env_util  # noqa: E402
import stub as stubmod  # noqa: E402

PI_BIN = os.environ.get("PI_BIN", "/home/ericflo/.nvm/versions/node/v24.16.0/bin/pi")
NODE_BIN_DIR = str(Path(PI_BIN).parent)
MAX_TRAJ_BYTES = 32 * 1024 * 1024      # keep a trajectory only if it is a sane size


def task_prompt(task, test_hint):
    """Instructions for the agent. Deliberately close to the prior runner's wording so numbers stay
    comparable, plus a pointer at the relevant test file (a real developer would not run a 700-test
    suite to check one function, and on the slower repos the full suite dominates the episode)."""
    rel, qual = task["rel_file"], task["qual_name"]
    return (f"You are in the project root directory (do not cd). The function `{qual}` in `{rel}` "
            f"has had its body replaced with `raise NotImplementedError`. Implement `{qual}` so the "
            f"test suite passes. Read its tests, implement it, and run `{test_hint}` to verify. "
            f"Only edit `{rel}`.")


def run_episode(task, baselines, provider, model, timeout, traj_dir, idx):
    repo_name = task["repo"]
    repo = STORE / repo_name
    base = baselines[repo_name]
    ftp = task["fail_to_pass"]
    ptp = sorted(set(base["pass"]) - set(ftp))

    copy = env_util.prepare_copy(repo, prefix=f"pi_{repo_name}_")
    # task_id contains '/' (from rel_file) and '::', so it must be flattened before use as a filename;
    # unsanitised it silently creates nested directories instead of one trajectory file per episode.
    safe_id = task["task_id"].replace("/", "-").replace("::", "__").replace(" ", "_")
    spool = Path(traj_dir) / f"{safe_id}__{idx}.jsonl"
    spool.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not stubmod.stub_function(copy, task["rel_file"], task["qual_name"]):
            return {"task_id": task["task_id"], "repo": repo_name, "error": "stub_failed",
                    "reward": 0.0, "solved": False}
        test_hint = task.get("test_cmd_hint") or "python -m pytest -q"
        cmd = [PI_BIN, "--provider", provider, "--model", model, "-p",
               task_prompt(task, test_hint), "--mode", "json", "--no-session"]
        # pi inherits the repo's test venv on PATH (so the model's own `pytest` resolves inside the
        # checkout) and PYTHONPATH pointing at the COPY (so imports cannot leak to the original).
        penv = {**env_util.build_env(copy, repo),
                "PATH": NODE_BIN_DIR + os.pathsep + env_util.build_env(copy, repo)["PATH"]}
        t0 = time.time()
        code, tail, total, _ = env_util.run_spooled(
            " ".join(_shquote(c) for c in cmd), cwd=copy, env=penv, timeout=timeout,
            spool_path=str(spool), max_out_bytes=MAX_TRAJ_BYTES)
        secs = round(time.time() - t0, 1)

        # engagement gate: did the agent actually implement the function?
        edited = not stubmod.is_stubbed(copy, task["rel_file"])
        if not edited:
            score = {"reward": 0.0, "ftp_passed": 0, "ftp_total": len(ftp), "solved": False,
                     "regressed": [], "n_regressed": 0, "unrunnable": False}
            res = {"code": None}
        else:
            res = env_util.run_tests(copy, repo, timeout=max(120, timeout // 3))
            score = env_util.score_task(res["statuses"], ftp, ptp)
        return {"task_id": task["task_id"], "repo": repo_name, "edited": edited,
                "pi_exit": code, "secs": secs, "traj_bytes": total,
                "traj": str(spool.relative_to(OUTD)) if spool.exists() else None,
                "test_code": res.get("code"), **score}
    except Exception as exc:                                   # pragma: no cover - defensive
        return {"task_id": task["task_id"], "repo": repo_name, "reward": 0.0, "solved": False,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def _shquote(s):
    import shlex
    return shlex.quote(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="kiln-local")
    ap.add_argument("--model", default="qwen35-4b-pi8k",
                    help="pi model entry whose maxTokens fits the served context; pi sends maxTokens "
                         "as max_completion_tokens on EVERY call, so a 32768 entry dies once the "
                         "conversation passes ~8k in a 40960 window")
    ap.add_argument("--tasks", default=str(DATA / "tasks_split.json"))
    ap.add_argument("--baselines", default=str(DATA / "repo_baselines.json"))
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="FIREWALL: 'test' repos are eval-only and must never be harvested/trained on")
    ap.add_argument("--k", type=int, default=3, help="rollouts per task (execution-selected best-of-k)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="cap number of tasks (0 = all)")
    ap.add_argument("--label", required=True, help="arm name, e.g. base / warmstart")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    split = json.load(open(a.tasks))
    tasks = split[a.split]
    if a.limit:
        tasks = tasks[: a.limit]
    baselines = json.load(open(a.baselines))
    out = Path(a.out or (OUTD / "reports" / f"pi_{a.label}_{a.split}.json"))
    traj_dir = OUTD / "trajectories" / a.label
    out.parent.mkdir(parents=True, exist_ok=True)

    # RESUME: these runs are hours long and this box has died 8 times. Never lose finished episodes.
    prior = []
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            if prev.get("label") == a.label and prev.get("split") == a.split:
                prior = prev.get("episodes", [])
        except Exception:
            pass
    done = {}
    for r in prior:
        done[r["task_id"]] = done.get(r["task_id"], 0) + 1
    jobs = [(t, i) for t in tasks for i in range(a.k) if i >= done.get(t["task_id"], 0)]
    if prior:
        print(f"RESUME: {len(prior)} episodes recovered; {len(jobs)} remaining", flush=True)
    print(f"pi REAL-REPO eval [{a.label}] split={a.split}: {len(jobs)} episodes "
          f"= {len(tasks)} tasks x k={a.k} | provider={a.provider} model={a.model}", flush=True)

    def summarize(recs):
        per = {}
        for r in recs:
            per.setdefault(r["task_id"], []).append(r)
        single = sum(sum(1 for e in v if e.get("solved")) / len(v) for v in per.values()) / max(1, len(per))
        sel = sum(1 for v in per.values() if any(e.get("solved") for e in v)) / max(1, len(per))
        meanr = sum(sum(e.get("reward", 0) for e in v) / len(v) for v in per.values()) / max(1, len(per))
        return per, single, sel, meanr

    def save(recs, partial):
        per, single, sel, meanr = summarize(recs)
        out.write_text(json.dumps(
            {"label": a.label, "split": a.split, "provider": a.provider, "model": a.model,
             "k": a.k, "partial": partial, "n_episodes": len(recs), "n_tasks": len(per),
             "single_shot": round(single, 4), "selected_best_of_k": round(sel, 4),
             "mean_reward": round(meanr, 4),
             "per_task": {t: {"solved_any": any(e.get("solved") for e in v),
                              "rate": sum(1 for e in v if e.get("solved")) / len(v),
                              "mean_reward": round(sum(e.get("reward", 0) for e in v) / len(v), 3)}
                          for t, v in per.items()},
             "episodes": recs}, indent=1))

    recs, t0 = list(prior), time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_episode, t, baselines, a.provider, a.model, a.timeout, traj_dir, i)
                for t, i in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                recs.append(f.result())
            except Exception as exc:
                print(f"  episode error: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            if i % 4 == 0 or i == len(jobs):
                _, single, sel, meanr = summarize(recs)
                print(f"  {i}/{len(jobs)} ({round(time.time()-t0)}s) "
                      f"single={single:.3f} selected={sel:.3f} meanR={meanr:.3f}", flush=True)
                save(recs, partial=(i < len(jobs)))

    per, single, sel, meanr = summarize(recs)
    print(f"\n=== [{a.label}] {a.split} split: {len(per)} tasks ===", flush=True)
    for t, v in sorted(per.items(), key=lambda x: -sum(1 for e in x[1] if e.get("solved"))):
        rate = sum(1 for e in v if e.get("solved")) / len(v)
        mr = sum(e.get("reward", 0) for e in v) / len(v)
        print(f"  {t:44s} solved {rate:.2f}  meanR {mr:.2f}  (n={len(v)})", flush=True)
    print(f"\nSINGLE-SHOT: {single:.4f}", flush=True)
    print(f"EXECUTION-SELECTED best-of-{a.k}: {sel:.4f}", flush=True)
    print(f"MEAN REWARD (partial credit): {meanr:.4f}", flush=True)
    print(f"engaged (edited): {sum(1 for r in recs if r.get('edited'))}/{len(recs)}", flush=True)
    save(recs, partial=False)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
