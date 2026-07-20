"""Run a scenario through pi-coding-agent itself, then score it with execution reward.

This is the DEPLOYMENT-TRUTH harness. Everything else in this experiment drives the model through
our own tool loop (TRL's for training, calibrate_trl's for calibration). Those measure the model in
OUR harness. pi-coding-agent is the actual scaffold the model is meant to run in, and it differs in
every way that has already bitten us this session: its own system prompt, its own tool schemas and
names, its own multi-turn loop, its own truncation and retry behaviour. A number measured in our
harness is a claim about our harness until it is reproduced here.

pi is a Node CLI driven headlessly:
    pi --provider <p> --model <m> -p "<task>" --mode json --no-session
run with cwd set to the project directory, so pi's built-in file/bash tools operate on the real
files. We then score the resulting directory with the SAME execution reward the trainer uses, so the
number is comparable to the RLVR reward and to calibrate_trl's pass rate.

Provider wiring (from ~/.pi/agent/models.json, recorded here because the earlier probe never
codified it): provider `kiln-local`, api `openai-completions`, baseUrl http://localhost:8420/v1,
apiKey `dummy`, model id `Qwen3.5-4B`. So serve the policy under test as:
    vllm serve <merged-composite> --served-model-name Qwen3.5-4B --enforce-eager \
        --max-model-len 16384 --port 8420
and the existing pi config works unchanged -- no pi-side edits needed to swap policies.

Verified against a mock endpoint (no GPU needed) before trusting it against the real policy:
  - pi reaches the configured baseUrl and runs fully headless with cwd = the project dir
  - pi REQUIRES SSE: a plain-JSON response yields "Stream ended without finish_reason"
  - on that error pi auto-retries 3x (2s, 4s backoff) before giving up, so transport failures
    present as long hangs rather than immediate errors -- budget --timeout accordingly
  - `--mode json` emits one JSON event per line (session/agent_start/turn_start/message_*/
    turn_end{toolResults}/agent_end), so trajectories are harvestable straight from stdout
vLLM's OpenAI server speaks SSE natively, so the real run needs no shim.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

from coding_env import SynthEnv  # noqa: E402

PI_BIN = os.environ.get("PI_BIN", "/home/ericflo/.nvm/versions/node/v24.16.0/bin/pi")
NODE_BIN_DIR = str(Path(PI_BIN).parent)


def score_dir(project: Path, check: str) -> float:
    """Score a finished project directory with the trainer's execution reward."""
    env = SynthEnv()
    env.project = project
    env.test_cmd = check
    env.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env._task = {}
    return env.get_reward()


def run_pi_episode(scenario, provider, model, timeout, extra_args=()):
    """Seed a scenario into a temp dir, let pi drive, return (reward, meta)."""
    project = Path(tempfile.mkdtemp(prefix="pienv_"))
    try:
        for rel, content in scenario["files"].items():
            p = project / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        cmd = [PI_BIN, "--provider", provider, "--model", model, "-p", scenario["prompt"],
               "--mode", "json", "--no-session", *extra_args]
        env = {**os.environ, "PATH": NODE_BIN_DIR + os.pathsep + os.environ.get("PATH", "")}
        t0 = time.time()
        try:
            # stdin MUST be closed: pi is interactive by default and inherits the parent's stdin.
            r = subprocess.run(cmd, cwd=str(project), env=env, text=True, timeout=timeout,
                               stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            out, code = r.stdout, r.returncode
        except subprocess.TimeoutExpired as exc:
            out, code = (exc.stdout or "") if isinstance(exc.stdout, str) else "", 124
        reward = score_dir(project, scenario["check"])
        edited = "NotImplementedError" not in (project / "solution.py").read_text(errors="replace")
        return reward, {"id": scenario["id"], "reward": reward, "edited": edited, "exit": code,
                        "secs": round(time.time() - t0, 1), "out_tail": out[-400:]}
    finally:
        shutil.rmtree(project, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="kiln-local")
    ap.add_argument("--model", default="qwen35-4b-pi8k",
                    help="pi model entry whose maxTokens fits the served context: pi sends "
                         "maxTokens as max_completion_tokens on EVERY call, so the stock 32768 "
                         "entry dies once the conversation passes ~8k in a 40960 window")
    ap.add_argument("--k", type=int, default=3, help="samples per task")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900,
                    help="pi AUTO-RETRIES 3x with 2s/4s backoff on stream errors, so a dead or "
                         "non-SSE endpoint burns ~3 full attempts before exiting -- keep this "
                         "generous or a transport problem looks like a slow episode")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated ids, or a *_split.json from rlvr_band.py (uses --split-key)")
    ap.add_argument("--split-key", default="holdout", choices=["holdout", "train"])
    ap.add_argument("--out", default=str(OUTD / "pi_eval.json"))
    ap.add_argument("--label", default="policy", help="name for this arm in the output")
    a = ap.parse_args()

    import synth_scenarios, mined_scenarios
    scenarios = list(synth_scenarios.SCENARIOS) + list(mined_scenarios.SCENARIOS)
    if a.tasks:
        if a.tasks.endswith(".json"):
            want = set(json.load(open(a.tasks))[a.split_key])
            print(f"FIREWALL: restricting to the {a.split_key} split -> {sorted(want)}", flush=True)
        else:
            want = {t.strip() for t in a.tasks.split(",") if t.strip()}
        scenarios = [s for s in scenarios if s["id"] in want]
    if not scenarios:
        print("no matching tasks"); return

    jobs = [s for s in scenarios for _ in range(a.k)]
    print(f"pi-coding-agent eval: {len(jobs)} episodes = {len(scenarios)} tasks x k={a.k} "
          f"| provider={a.provider} model={a.model}", flush=True)

    recs, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_pi_episode, s, a.provider, a.model, a.timeout) for s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                _, meta = f.result()
                recs.append(meta)
            except Exception as exc:
                print(f"  episode error: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            if i % 5 == 0:
                print(f"  {i}/{len(jobs)} ({round(time.time()-t0)}s)", flush=True)

    from collections import defaultdict
    per = defaultdict(list)
    for r in recs:
        per[r["id"]].append(r["reward"])
    rates = {t: sum(1 for x in v if x == 1.0) / len(v) for t, v in per.items()}
    mean_pass = sum(rates.values()) / max(1, len(rates))
    print("\n=== pi-coding-agent per-task pass rate ===", flush=True)
    for t, v in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {t:22s} pass {v:.2f}  meanR {sum(per[t])/len(per[t]):.2f}  (n={len(per[t])})", flush=True)
    print(f"\nMEAN pass rate: {mean_pass:.3f} over {len(rates)} tasks", flush=True)
    print(f"engaged (edited): {sum(r['edited'] for r in recs)}/{len(recs)}", flush=True)
    Path(a.out).write_text(json.dumps(
        {"label": a.label, "provider": a.provider, "model": a.model, "k": a.k,
         "rates": rates, "mean_pass": mean_pass, "episodes": recs}, indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
