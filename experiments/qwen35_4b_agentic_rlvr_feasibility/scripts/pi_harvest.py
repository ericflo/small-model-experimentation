"""Harvest pi-coding-agent's own execution-verified successful trajectories into SFT rows.

This is the executable form of "drive RLVR from execution reward through pi-coding-agent rollouts".
GRPO cannot consume pi rollouts -- pi ships no logprobs and TRL's GRPO requires `per_token_logps` --
so policy improvement runs as execution-filtered fitting instead:

    pi generates an episode  ->  tests decide pass/fail  ->  keep ONLY passing episodes
    ->  fit those trajectories  ->  merge  ->  repeat

Every training signal is self-generated and execution-verified, and every rollout comes from the
real deployment scaffold, so what gets installed is behaviour that works in pi -- not behaviour that
works in our harness. (Measured gap: the same policy scores 0.810 in pi vs 0.486 in ours.)

Prompt fidelity: rows are built from what the PROXY logged, not from pi's --mode json events,
because the events omit pi's system prompt and tool schemas. Training on a reconstructed context the
model never actually sees is how harness mismatches get baked into weights.

Concurrency: each worker gets its OWN proxy port and log file, and runs its episodes sequentially,
so exchanges never interleave between episodes -- an episode's rows are exactly the log lines
written between its start and end offsets. No per-request tagging needed.

FIREWALL: --split-key defaults to `train`; the held-out tasks must never be harvested.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

from coding_env import SynthEnv  # noqa: E402
from pi_episode import PI_BIN, NODE_BIN_DIR, score_dir  # noqa: E402


def ensure_providers(n, base_port, upstream_model):
    """Add one pi provider per worker, each pointing at that worker's proxy port (additive)."""
    cfg = Path.home() / ".pi/agent/models.json"
    d = json.loads(cfg.read_text())
    changed = False
    for i in range(n):
        name = f"kiln-harvest{i}"
        if name in d["providers"]:
            continue
        d["providers"][name] = {
            "api": "openai-completions", "apiKey": "dummy",
            "baseUrl": f"http://127.0.0.1:{base_port + i}/v1",
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [{"contextWindow": 40960, "id": upstream_model, "input": ["text"],
                        "maxTokens": 8192, "name": f"harvest worker {i}"}],
        }
        changed = True
    if changed:
        bak = cfg.with_suffix(".json.bak-harvest")
        if not bak.exists():
            shutil.copy(cfg, bak)
        cfg.write_text(json.dumps(d, indent=2))
    return [f"kiln-harvest{i}" for i in range(n)]


def _normalize(msg):
    """Coerce a logged OpenAI message into what the Qwen training template expects.

    pi sends tool_call arguments as JSON STRINGS (the OpenAI wire format), but the Qwen chat
    template calls .items() on `arguments`, so a string silently renders wrong or raises. Only the
    final completion was being converted; the REPLAYED assistant turns in `messages` -- i.e. most of
    the trajectory -- still carried strings. Convert every one.
    """
    m = dict(msg)
    out = []
    for tc in (m.get("tool_calls") or []):
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        fn["arguments"] = args if isinstance(args, dict) else {}
        tc["function"] = fn
        out.append(tc)
    if out:
        m["tool_calls"] = out
    return m


def rows_from_log(lines):
    """Turn one episode's logged exchanges into ONE conversational SFT row.

    The last exchange carries the longest message list (pi resends the whole conversation each turn),
    so it already contains every prior user/assistant/tool message. We take that as the conversation
    prefix and append the final assistant completion.
    """
    if not lines:
        return None
    last = max(lines, key=lambda r: len(r.get("messages") or []))
    msgs = [_normalize(m) for m in (last.get("messages") or [])]
    comp = last.get("completion") or {}
    assistant = {"role": "assistant", "content": comp.get("content") or ""}
    if comp.get("reasoning_content"):
        assistant["reasoning_content"] = comp["reasoning_content"]
    if comp.get("tool_calls"):
        tcs = []
        for c in comp["tool_calls"]:
            args = c.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            # Qwen's template calls .items() on arguments -> must be a dict, never a JSON string
            tcs.append({"type": "function", "function": {"name": c.get("name", ""),
                                                         "arguments": args if isinstance(args, dict) else {}}})
        assistant["tool_calls"] = tcs
    msgs.append(assistant)
    return {"messages": msgs, "tools": last.get("tools") or []}


def worker(idx, scenarios, provider, model, port, logpath, timeout, results, lock):
    env_base = {**os.environ, "PATH": NODE_BIN_DIR + os.pathsep + os.environ.get("PATH", "")}
    for sc in scenarios:
        project = Path(tempfile.mkdtemp(prefix="piharv_"))
        try:
            for rel, content in sc["files"].items():
                p = project / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            start = os.path.getsize(logpath) if os.path.exists(logpath) else 0
            cmd = [PI_BIN, "--provider", provider, "--model", model, "-p", sc["prompt"],
                   "--mode", "json", "--no-session"]
            try:
                subprocess.run(cmd, cwd=str(project), env=env_base, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                pass
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
            with lock:
                results.append({"id": sc["id"], "reward": reward, "exchanges": len(lines),
                                "row": rows_from_log(lines) if reward == 1.0 else None})
                done = len(results)
                kept = sum(1 for r in results if r["row"])
            print(f"  [{done}] {sc['id']:22s} reward {reward:.2f} "
                  f"({len(lines)} exchanges) | kept {kept}", flush=True)
        finally:
            shutil.rmtree(project, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="episodes per task")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=8431)
    ap.add_argument("--upstream", default="http://127.0.0.1:8420")
    ap.add_argument("--model", default="qwen35-4b-pi8k")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--split", default=str(OUTD / "adapters" / "rlvr_band_split.json"))
    ap.add_argument("--split-key", default="train", choices=["train", "holdout"])
    ap.add_argument("--tasks", default=None, help="explicit comma-separated ids (overrides --split)")
    ap.add_argument("--out", default=str(OUTD / "pi_sft_rows.jsonl"))
    a = ap.parse_args()

    import synth_scenarios, mined_scenarios
    pool = {s["id"]: s for s in list(synth_scenarios.SCENARIOS) + list(mined_scenarios.SCENARIOS)}
    if a.tasks:
        want = [t.strip() for t in a.tasks.split(",") if t.strip()]
    else:
        want = json.load(open(a.split))[a.split_key]
        if a.split_key == "holdout":
            print("REFUSING: harvesting the holdout split would break the train/test firewall")
            return
    scenarios = [pool[i] for i in want if i in pool]
    jobs = [s for s in scenarios for _ in range(a.k)]
    print(f"pi harvest: {len(jobs)} episodes = {len(scenarios)} tasks x k={a.k} "
          f"| {a.workers} workers (split={a.split_key})", flush=True)

    providers = ensure_providers(a.workers, a.base_port, a.model)
    logdir = Path(OUTD / "logs"); logdir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for i in range(a.workers):
        lp = logdir / f"pi_proxy_{i}.jsonl"
        lp.write_text("")
        logs.append(str(lp))
        procs.append(subprocess.Popen(
            [sys.executable, str(HERE / "pi_proxy.py"), "--port", str(a.base_port + i),
             "--upstream", a.upstream, "--log", str(lp)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(3)

    shards = [jobs[i::a.workers] for i in range(a.workers)]
    results, lock, threads = [], threading.Lock(), []
    t0 = time.time()
    try:
        for i in range(a.workers):
            t = threading.Thread(target=worker, args=(i, shards[i], providers[i], a.model,
                                                      a.base_port + i, logs[i], a.timeout,
                                                      results, lock))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        for p in procs:
            p.terminate()

    rows = [r["row"] for r in results if r["row"]]
    passed = sum(1 for r in results if r["reward"] == 1.0)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n=== pi harvest done in {round(time.time()-t0)}s ===", flush=True)
    print(f"episodes {len(results)} | PASSED {passed} ({100*passed/max(1,len(results)):.0f}%) "
          f"| SFT rows written {len(rows)}", flush=True)
    Path(str(a.out) + ".meta.json").write_text(json.dumps(
        {"episodes": len(results), "passed": passed, "rows": len(rows),
         "per_task": {t: [r["reward"] for r in results if r["id"] == t] for t in {r["id"] for r in results}}},
        indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
