"""Build DPO preference pairs that penalize the loop-forever pathology (task #62).

The pre-registered bet from the RFT negative result: execution-filtered SFT reinforces successes but
has NO negative gradient, so it cannot suppress the timeouts that are the actual deployment failure
(pi-RFT looped to the 600s wall on 28/33 holdout episodes). DPO supplies that missing negative
gradient directly, needing no logprobs (which pi does not surface).

CONSTRUCTION — maximally contrastive, both sides REAL and execution-verified:
  chosen   = a warm-start trajectory that SOLVED the task and STOPPED  (reward 1.0, ends on a
             content-only assistant turn). Reused from the existing harvest -- these are the same
             policy's own verified successes.
  rejected = a trajectory that LOOPED and TIMED OUT on the SAME task (exit 124), harvested fresh from
             a loop-forever policy served on :8420 (the pi-RFT composite is a timeout factory: 28/33).
Both come from our own Qwen3.5-4B derivatives; no external teacher. The prompt (pi's system prompt +
the task) is byte-identical across episodes of a task, which is exactly what DPO requires.

FIREWALL: chosen tasks are intersected with the TRAIN split; any holdout task is dropped. Rejected is
harvested only on --split-key train. The eval never sees a trained-on scenario.

MEMORY: a timed-out trajectory is the LONGEST possible sequence (it looped to the wall). Left whole it
would OOM DPO's two-sequence forward, so rejected is truncated to the first --max-rej-turns assistant
turns, always ending ON an assistant turn (a dangling tool result is not a valid completion). The
truncated span still exhibits the pathology: at turn N it is STILL calling tools instead of stopping.
"""
import argparse, json, os, shutil, subprocess, sys, socket, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

from pi_harvest import ensure_providers, rows_from_log  # reuse the proven harvest infra
from pi_episode import PI_BIN, NODE_BIN_DIR, score_dir


def _user_text(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def _msg_text(msg):
    """All rendered text of a message: content + tool-call arguments (where a written file shows up)."""
    parts = [_user_text(msg)]
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, dict):
            parts.append(" ".join(str(v) for v in args.values()))
        else:
            parts.append(str(args))
    return "\n".join(parts)


def build_task_fingerprints(pool):
    """The scenario PROMPT is identical across tasks (the spec lives in test_solution.py). Identify a
    trajectory's task by the test-file content it read into a tool result. Use a distinctive slice so
    truncated reads still match."""
    fp = {}
    for sid, s in pool.items():
        test = (s.get("files") or {}).get("test_solution.py", "")
        sig = test.strip()[:400]
        if len(sig) >= 40:
            fp[sid] = sig
    return fp


def recover_task_id(msgs, fingerprints):
    """Match on the test-file content the model read (appears verbatim in a tool result)."""
    blob = "\n".join(_msg_text(m) for m in msgs)
    hits = [sid for sid, sig in fingerprints.items() if sig in blob]
    # if several scenarios share a common test prefix, prefer the longest (most specific) signature
    if not hits:
        return None
    return max(hits, key=lambda sid: len(fingerprints[sid]))


def _assistant_turns(msgs):
    return sum(1 for m in msgs if m.get("role") == "assistant")


# --- SYNTHETIC decision-point negative -------------------------------------------------------
# pi_rft's timeout is NOT a clean multi-turn loop we can harvest -- it is a single non-terminating
# generation (measured: 92 stream deltas, 1 message_end, content frozen at "let me read the test
# file" while <think> ran forever). The proxy logs on stream COMPLETION, so a run that never
# completes logs nothing -> the harvest yields 0-turn rejected. The pathology is "won't STOP once the
# task is solved". The sharpest, most faithful negative for that is built at the decision point:
#   prompt  = the solved trajectory UP TO the final tool result (tests just passed)
#   chosen  = the real terminal turn (the model stops and answers)   [reward-verified success]
#   rejected= one more tool call instead of stopping                 [the non-termination behaviour]
# The only thing that differs is stop-vs-continue, so DPO's gradient is exactly "terminate when done".
_REDUNDANT = [
    ("bash", {"command": "python3 test_solution.py"}, "Let me run the tests one more time to be sure."),
    ("read", {"path": "solution.py"}, "Let me re-read my solution before finishing."),
    ("bash", {"command": "cat solution.py"}, "Let me double-check the file contents again."),
    ("read", {"path": "test_solution.py"}, "Let me look at the tests once more to be thorough."),
]


def synthetic_rejected(idx):
    name, args, reason = _REDUNDANT[idx % len(_REDUNDANT)]
    return [{"role": "assistant", "content": "", "reasoning_content": reason,
             "tool_calls": [{"type": "function", "id": f"rej_{idx}",
                             "function": {"name": name, "arguments": args}}]}]


def build_synthetic_pairs(chosen_by_task, tools):
    """One decision-point pair per clean-stop passing trajectory (no harvest / server needed)."""
    pairs, i = [], 0
    for task in sorted(chosen_by_task):
        for msgs in chosen_by_task[task]:
            last = msgs[-1]
            if last.get("role") != "assistant" or last.get("tool_calls"):
                continue   # need a clean terminal stop as the chosen
            prompt = msgs[:-1]
            if not prompt or prompt[-1].get("role") not in ("tool", "user"):
                continue
            pairs.append({"prompt": prompt, "chosen": [last], "rejected": synthetic_rejected(i),
                          "tools": tools, "task": task, "chosen_turns": 1, "rejected_turns": 1,
                          "rejected_exit": -1})   # -1 = synthetic (not a harvested timeout)
            i += 1
    return pairs


def _split_prompt_completion(msgs):
    """prompt = [system, user]; completion = the rest (assistant/tool turns)."""
    # find the first user turn; everything up to & including it is the shared prompt
    u = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    if u is None:
        return None, None
    return msgs[: u + 1], msgs[u + 1 :]


def _truncate_completion(completion, max_turns):
    """Keep the first `max_turns` assistant turns; end ON an assistant turn (drop a trailing tool)."""
    out, seen = [], 0
    for m in completion:
        out.append(m)
        if m.get("role") == "assistant":
            seen += 1
            if seen >= max_turns:
                break
    while out and out[-1].get("role") != "assistant":
        out.pop()
    return out


# ---- harvest the served (loop-forever) policy for REAL rejected trajectories -------------------
def worker(scenarios, provider, model, port, logpath, timeout, results, lock):
    env_base = {**os.environ, "PATH": NODE_BIN_DIR + os.pathsep + os.environ.get("PATH", "")}
    import tempfile
    for sc in scenarios:
        project = Path(tempfile.mkdtemp(prefix="pidpo_"))
        try:
            for rel, content in sc["files"].items():
                p = project / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            start = os.path.getsize(logpath) if os.path.exists(logpath) else 0
            cmd = [PI_BIN, "--provider", provider, "--model", model, "-p", sc["prompt"],
                   "--mode", "json", "--no-session"]
            code = 0
            try:
                subprocess.run(cmd, cwd=str(project), env=env_base, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                code = 124
            reward = score_dir(project, sc["check"])
            lines = []
            if os.path.exists(logpath):
                with open(logpath) as fh:
                    fh.seek(start)
                    for ln in fh:
                        ln = ln.strip()
                        if ln:
                            try:
                                lines.append(json.loads(ln))
                            except Exception:
                                pass
            row = rows_from_log(lines)
            with lock:
                results.append({"id": sc["id"], "reward": reward, "exit": code,
                                "turns": _assistant_turns(row["messages"]) if row else 0, "row": row})
                done = len(results)
            print(f"  [{done}] {sc['id']:22s} reward {reward:.2f} exit {code} "
                  f"turns {results[-1]['turns']}", flush=True)
        finally:
            shutil.rmtree(project, ignore_errors=True)


def harvest_rejected(train_tasks, pool, k, workers, base_port, upstream, model, timeout):
    scenarios = [pool[i] for i in train_tasks if i in pool]
    jobs = [s for s in scenarios for _ in range(k)]
    print(f"rejected harvest: {len(jobs)} episodes = {len(scenarios)} tasks x k={k} "
          f"({workers} workers, timeout {timeout}s)", flush=True)
    for i in range(workers):
        with socket.socket() as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind(("127.0.0.1", base_port + i))
            except OSError:
                raise SystemExit(f"port {base_port + i} busy -- kill stale pi_proxy (pgrep -af pi_proxy.py)")
    providers = ensure_providers(workers, base_port, model)
    logdir = Path(OUTD / "logs"); logdir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for i in range(workers):
        lp = logdir / f"pi_dpo_proxy_{i}.jsonl"; lp.write_text(""); logs.append(str(lp))
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE / "pi_proxy.py"), "--port", str(base_port + i),
             "--upstream", upstream, "--log", str(lp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(3)
    shards = [jobs[i::workers] for i in range(workers)]
    results, lock, threads = [], threading.Lock(), []
    try:
        for i in range(workers):
            t = threading.Thread(target=worker, args=(shards[i], providers[i], model,
                                                       base_port + i, logs[i], timeout, results, lock))
            t.start(); threads.append(t)
        for t in threads:
            t.join()
    finally:
        for p in procs:
            p.terminate()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen-rows", default=str(OUTD / "pi_sft_rows.jsonl"),
                    help="warm-start's execution-verified PASSES (reused as chosen)")
    ap.add_argument("--split", default=str(OUTD / "pi_split.json"),
                    help="pi-native split -- the RFT baseline (0.606) and eval both use THIS split")
    ap.add_argument("--out", default=str(OUTD / "pi_dpo_pairs.jsonl"))
    ap.add_argument("--k", type=int, default=4, help="rejected episodes per train task")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=8451)
    ap.add_argument("--upstream", default="http://127.0.0.1:8420",
                    help="serve the LOOP-FOREVER policy (pi_rft composite) here for real timeouts")
    ap.add_argument("--model", default="qwen35-4b-pi8k")
    ap.add_argument("--timeout", type=int, default=300,
                    help="a 300s loop is unambiguously a timeout; shorter than eval's 600 to save budget")
    ap.add_argument("--max-rej-turns", type=int, default=8, help="truncate rejected to bound DPO memory")
    ap.add_argument("--pairs-per-task", type=int, default=3)
    ap.add_argument("--reharvest", action="store_true",
                    help="force a fresh rejected harvest even if a cache exists")
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "harvest"],
                    help="synthetic = decision-point negatives from chosen (no server); "
                         "harvest = real pi_rft timeouts via proxy (fragile: pi_rft never completes "
                         "a stream so the proxy captures nothing -- kept for reference)")
    a = ap.parse_args()

    import synth_scenarios, mined_scenarios
    pool = {s["id"]: s for s in list(synth_scenarios.SCENARIOS) + list(mined_scenarios.SCENARIOS)}
    split = json.load(open(a.split))
    train, holdout = set(split["train"]), set(split["holdout"])

    # --- chosen: reuse warm-start passes, recover task id from the READ test file, firewall to train ---
    fingerprints = build_task_fingerprints(pool)
    chosen_by_task = {}
    dropped_holdout = unmatched = 0
    for l in open(a.chosen_rows):
        if not l.strip():
            continue
        r = json.loads(l)
        msgs = r["messages"]
        uid = recover_task_id(msgs, fingerprints)
        if uid is None:
            unmatched += 1
            continue
        if uid in holdout:
            dropped_holdout += 1
            continue
        if uid not in train:
            continue
        chosen_by_task.setdefault(uid, []).append(msgs)
    if unmatched:
        print(f"NOTE: {unmatched} chosen rows could not be matched to a task (no test-file read found)", flush=True)
    if dropped_holdout:
        print(f"FIREWALL: dropped {dropped_holdout} chosen rows whose task is in the holdout split", flush=True)
    print(f"chosen: {sum(len(v) for v in chosen_by_task.values())} passes over "
          f"{len(chosen_by_task)} train tasks", flush=True)

    # tool schemas: pi's real read/edit/write/bash, taken verbatim from a chosen row
    example_tools = None
    for l in open(a.chosen_rows):
        if l.strip():
            example_tools = json.loads(l).get("tools") or []
            if example_tools:
                break

    if a.mode == "synthetic":
        pairs = build_synthetic_pairs(chosen_by_task, example_tools)
        with open(a.out, "w") as fh:
            for p in pairs:
                fh.write(json.dumps(p) + "\n")
        by_task = {}
        for p in pairs:
            by_task[p["task"]] = by_task.get(p["task"], 0) + 1
        print(f"\n=== {len(pairs)} SYNTHETIC decision-point pairs over {len(by_task)} tasks "
              f"-> {a.out} ===", flush=True)
        for t, c in sorted(by_task.items()):
            print(f"  {t:22s} {c}", flush=True)
        Path(str(a.out) + ".meta.json").write_text(json.dumps(
            {"mode": "synthetic", "pairs": len(pairs), "tasks": len(by_task),
             "per_task": by_task, "chosen_tasks": sorted(chosen_by_task)}, indent=1))
        return

    # --- rejected: harvest the served loop-forever policy for real timeouts on the same train tasks ---
    cache = Path(str(a.out) + ".rejected_cache.jsonl")
    want_tasks = sorted(set(chosen_by_task) & train)   # only tasks we can actually pair
    if cache.exists() and not a.reharvest:
        results = [json.loads(l) for l in open(cache) if l.strip()]
        print(f"reusing cached rejected harvest ({len(results)} episodes) -- pass --reharvest to redo", flush=True)
    else:
        results = harvest_rejected(want_tasks, pool, a.k, a.workers, a.base_port,
                                   a.upstream, a.model, a.timeout)
        with open(cache, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        print(f"cached rejected harvest -> {cache}", flush=True)

    rejected_by_task = {}
    for r in results:
        if r["row"] is None or r["reward"] == 1.0:
            continue   # a pass is not a rejected; keep only real failures/timeouts
        rejected_by_task.setdefault(r["id"], []).append(r)

    # --- pair per task: shortest passes (cleanest stop signal) x worst failures (timeouts first) ---
    def tools_of(msgs_row_or_none, fallback):
        return fallback
    # tool schemas: take from a chosen row (they carry pi's real read/edit/write/bash schemas)
    example_tools = None
    for l in open(a.chosen_rows):
        if l.strip():
            example_tools = json.loads(l).get("tools") or []
            if example_tools:
                break

    pairs, per_task_counts = [], {}
    for task in sorted(want_tasks):
        chs = sorted(chosen_by_task.get(task, []), key=_assistant_turns)          # short first
        rejs = rejected_by_task.get(task, [])
        # worst rejected first: timeouts (exit 124), then most turns (most looping)
        rejs = sorted(rejs, key=lambda r: (r["exit"] != 124, -r["turns"]))
        if not chs or not rejs:
            continue
        n = min(a.pairs_per_task, len(chs), len(rejs))
        for i in range(n):
            ch_msgs = chs[i]
            rej_row = rejs[i]
            p_ch, comp_ch = _split_prompt_completion(ch_msgs)
            p_rej, comp_rej = _split_prompt_completion(rej_row["row"]["messages"])
            if not comp_ch or not comp_rej:
                continue
            comp_rej = _truncate_completion(comp_rej, a.max_rej_turns)
            if not comp_rej:
                continue
            # DPO needs prompt shared; use the chosen's [system,user] (byte-identical task prompt)
            pairs.append({"prompt": p_ch, "chosen": comp_ch, "rejected": comp_rej,
                          "tools": example_tools, "task": task,
                          "chosen_turns": _assistant_turns(comp_ch),
                          "rejected_turns": _assistant_turns(comp_rej),
                          "rejected_exit": rej_row["exit"]})
        per_task_counts[task] = n

    with open(a.out, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    n_timeout = sum(1 for p in pairs if p["rejected_exit"] == 124)
    print(f"\n=== {len(pairs)} DPO pairs over {len(per_task_counts)} tasks "
          f"({n_timeout} rejected are real timeouts) -> {a.out} ===", flush=True)
    for t, c in sorted(per_task_counts.items()):
        print(f"  {t:22s} {c} pairs", flush=True)
    Path(str(a.out) + ".meta.json").write_text(json.dumps(
        {"pairs": len(pairs), "tasks": len(per_task_counts), "timeout_rejected": n_timeout,
         "chosen_tasks": sorted(chosen_by_task), "per_task": per_task_counts}, indent=1))


if __name__ == "__main__":
    main()
