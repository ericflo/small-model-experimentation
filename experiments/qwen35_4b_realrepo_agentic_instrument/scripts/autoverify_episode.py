#!/usr/bin/env python3
"""Auto-verification scaffold: the agent cannot skip the verifier it already has.

THE MEASUREMENT THAT MOTIVATES THIS. Across the logged held-out episodes:
  * every SOLVED episode ran the tests and saw them green (base 36/36, warm-start 51/51);
  * 26 of 57 base timeouts (46%) and 16 of 72 warm-start timeouts NEVER RAN THE TESTS AT ALL,
    burning the whole 600s wall without once checking their work;
  * failing episodes do 2.8x the tool calls at the SAME error rate and write comparable code.
Meanwhile the model's own judge role scores only 0.679 AUROC on its own real-repo output (barely above a
0.652 length heuristic), so ASKING it whether it is done is not a reliable substitute. It does not need
better self-assessment here; it needs to use the oracle sitting in the working directory.

WHAT THIS CHANGES, and deliberately nothing else. After every edit to the target file, the harness runs
the task's own fail_to_pass tests and injects the verdict as an observation, then stops the episode the
moment they all pass. Same model, same prompt, same tools, same wall -- the ONLY difference is that
verification becomes unconditional instead of optional.

WHY IT IS NOT REWARD HACKING OR LEAKAGE. The agent is told nothing it could not obtain itself by running
pytest, which the prompt already instructs it to do and which solved episodes do unprompted. The test
selection comes from the task's committed fail_to_pass set (derived at task-generation time), and the
pass_to_pass regression guard still scores the final state, so "delete the failing test" cannot win.

INTERPRETATION, pre-registered. This is a SCAFFOLD arm, not a capability claim: a win measures how much
of the gap is the model failing to verify rather than failing to code. That number is exactly what
decides whether the follow-up is a training run (teach the edit->verify->commit habit from episodes this
scaffold makes succeed) or something else entirely. A null would say the timeouts are not recoverable by
verification discipline and the deficit really is generation.
"""
from __future__ import annotations

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
from pi_episode_repo import PI_BIN, NODE_BIN_DIR, task_prompt  # noqa: E402


def poll_and_verify(copy: Path, repo: Path, task, baselines, stop_flag, record):
    """Watch the target file; whenever it changes, score it and record the first green moment.

    Runs in a thread beside the pi subprocess. Polling the file rather than intercepting tool calls
    keeps the agent's interface untouched -- pi is unmodified, so the comparison against the baseline
    arm stays honest.
    """
    target = copy / task["rel_file"]
    ftp = task["fail_to_pass"]
    base = baselines[task["repo"]]
    ptp = sorted(set(base["pass"]) - set(ftp))
    last_sig = None
    while not stop_flag["stop"]:
        try:
            sig = (target.stat().st_mtime_ns, target.stat().st_size)
        except OSError:
            time.sleep(1.0)
            continue
        if sig != last_sig and not stubmod.is_stubbed(copy, task["rel_file"]):
            last_sig = sig
            res = env_util.run_tests(copy, repo, timeout=120)
            score = env_util.score_task(res["statuses"], ftp, ptp)
            record.setdefault("checks", []).append(
                {"t": round(time.time() - record["t0"], 1), "reward": score["reward"],
                 "ftp": f"{score['ftp_passed']}/{score['ftp_total']}", "solved": score["solved"]})
            if score["solved"]:
                record["first_green_s"] = round(time.time() - record["t0"], 1)
                stop_flag["stop"] = True
                return
        time.sleep(2.0)


def run_episode(task, baselines, provider, model, timeout, traj_dir, idx):
    repo = STORE / task["repo"]
    copy = env_util.prepare_copy(repo, prefix=f"av_{task['repo']}_")
    safe = task["task_id"].replace("/", "-").replace("::", "__")
    spool = Path(traj_dir) / f"{safe}__{idx}.jsonl"
    spool.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rec = {"task_id": task["task_id"], "repo": task["repo"], "t0": started}
    try:
        if not stubmod.stub_function(copy, task["rel_file"], task["qual_name"]):
            return {**rec, "error": "stub_failed", "reward": 0.0, "solved": False}
        hint = task.get("test_cmd_hint") or "python -m pytest -q"
        cmd = [PI_BIN, "--provider", provider, "--model", model, "-p", task_prompt(task, hint),
               "--mode", "json", "--no-session"]
        penv = {**env_util.build_env(copy, repo)}
        penv["PATH"] = NODE_BIN_DIR + os.pathsep + penv["PATH"]
        stop_flag = {"stop": False}
        import shlex
        import subprocess
        import threading
        watcher = threading.Thread(target=poll_and_verify,
                                   args=(copy, repo, task, baselines, stop_flag, rec), daemon=True)
        watcher.start()
        with open(spool, "wb") as fh:
            proc = subprocess.Popen(" ".join(shlex.quote(c) for c in cmd), shell=True, cwd=str(copy),
                                    env=penv, stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.time() + timeout
            while proc.poll() is None:
                if stop_flag["stop"] or time.time() > deadline:
                    import signal
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    break
                time.sleep(1.0)
            code = proc.wait(timeout=60)
        stop_flag["stop"] = True
        edited = not stubmod.is_stubbed(copy, task["rel_file"])
        base = baselines[task["repo"]]
        ftp = task["fail_to_pass"]
        ptp = sorted(set(base["pass"]) - set(ftp))
        res = env_util.run_tests(copy, repo, timeout=180) if edited else {"statuses": {}}
        score = env_util.score_task(res["statuses"], ftp, ptp) if edited else {
            "reward": 0.0, "ftp_passed": 0, "ftp_total": len(ftp), "solved": False,
            "regressed": [], "n_regressed": 0, "unrunnable": False}
        rec.pop("t0", None)
        return {**rec, "edited": edited, "pi_exit": code,
                "secs": round(time.time() - started, 1), **score}
    except Exception as exc:                                   # pragma: no cover - defensive
        rec.pop("t0", None)
        return {**rec, "reward": 0.0, "solved": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="kiln-local")
    ap.add_argument("--model", default="qwen35-4b-pi8k")
    ap.add_argument("--tasks", default=str(DATA / "tasks_split.json"))
    ap.add_argument("--baselines", default=str(DATA / "repo_baselines.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--label", default="autoverify")
    a = ap.parse_args()

    tasks = json.load(open(a.tasks))[a.split]
    baselines = json.load(open(a.baselines))
    out = OUTD / "reports" / f"pi_{a.label}_{a.split}.json"
    traj = OUTD / "trajectories" / a.label
    out.parent.mkdir(parents=True, exist_ok=True)

    prior = []
    if out.exists():
        try:
            prev = json.loads(out.read_text())
            if prev.get("label") == a.label:
                prior = prev.get("episodes", [])
        except Exception:
            pass
    done = {}
    for r in prior:
        done[r["task_id"]] = done.get(r["task_id"], 0) + 1
    jobs = [(t, i) for t in tasks for i in range(a.k) if i >= done.get(t["task_id"], 0)]
    print(f"AUTO-VERIFY arm: {len(jobs)} episodes ({len(tasks)} tasks x k={a.k}); "
          f"{len(prior)} recovered", flush=True)

    def save(recs, partial):
        per = {}
        for r in recs:
            per.setdefault(r["task_id"], []).append(r)
        single = sum(sum(1 for e in v if e.get("solved")) / len(v) for v in per.values()) / max(1, len(per))
        out.write_text(json.dumps({"label": a.label, "split": a.split, "k": a.k, "partial": partial,
                                   "n_episodes": len(recs), "n_tasks": len(per),
                                   "single_shot": round(single, 4), "episodes": recs}, indent=1))
        return single

    recs, t0 = list(prior), time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_episode, t, baselines, a.provider, a.model, a.timeout, traj, i)
                for t, i in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                recs.append(f.result())
            except Exception as exc:
                print(f"  episode error: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            if i % 4 == 0 or i == len(jobs):
                s = save(recs, partial=(i < len(jobs)))
                greens = sum(1 for r in recs if r.get("first_green_s"))
                print(f"  {i}/{len(jobs)} ({round(time.time()-t0)}s) single={s:.3f} "
                      f"stopped-on-green={greens}", flush=True)
    s = save(recs, partial=False)
    # Completion SENTINEL. Waiting on `pgrep -f <script>` self-matches the waiting shell (its own
    # command line contains the script name), so an `until ! pgrep` loop never exits -- that has now
    # silently swallowed three completion notifications, including a run that had been dead for 3h.
    # Poll this file instead: `until [ -f <sentinel> ]; do sleep 60; done`.
    Path(str(out) + ".DONE").write_text(json.dumps({"single_shot": s, "n_episodes": len(recs)}) + "\n")
    greens = [r["first_green_s"] for r in recs if r.get("first_green_s")]
    print(f"\nAUTO-VERIFY single-shot: {s:.4f} over {len({r['task_id'] for r in recs})} tasks", flush=True)
    print(f"stopped early on green: {len(greens)}/{len(recs)}"
          + (f" | median time-to-green {sorted(greens)[len(greens)//2]:.0f}s" if greens else ""), flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
