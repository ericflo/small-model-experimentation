"""Per-task pass rate for a SERVED policy, using TRL's OWN template and tool-call parser.

Why this exists (the mistake it prevents): the previous calibrator drove episodes through a
hand-rolled XML tool-call loop and reported a healthy 0.25-0.75 band. GRPO then scored 0.0 on those
same tasks, every rollout, for hours -- which read as "the band is wrong" but was a HARNESS
MISMATCH. The hand-rolled parser silently ignored the spurious `content=None` the policy emits on
every call, while TRL passed it straight through to the tool and 86% of calls died on TypeError.
A calibrator whose harness differs from the trainer's harness measures a policy that will never run.

So this renders prompts with TRL's chat template, extracts tool calls with TRL's regex schema
(`qwen3_5_schema`), and executes them against the SAME SynthEnv the trainer uses. Generation-only:
no trainer resident, so vLLM gets the whole card and episodes run at full length (the 8k context
that truncation analysis showed is required -- at 3072, 100% of episodes were clipped before
write_file and the task was physically unsolvable).

Requires the policy served on :1234 as model id `base`, e.g.
  vllm serve <merged> --served-model-name base --enforce-eager --max-model-len 16384
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

from coding_env import SynthEnv  # noqa: E402

SYSTEM = ("You are an expert Python coding agent. Think step by step, then use the tools to inspect "
          "files, write code, and run the tests until they pass. You MUST edit the target file with "
          "write_file and MUST run the tests with run_bash to verify before finishing. Do not give up "
          "after only reading files. Keep iterating until the tests pass.")

# TRL's qwen3_5_schema tool-call extraction, applied verbatim so parsing matches the trainer exactly.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.S)
FN_RE = re.compile(r"<function=([^\n>]+)>")
ARG_RE = re.compile(r"<parameter=(?P<key>[^>\n]+)>\n(?P<value>.*?)\n</parameter>", re.S)


def parse_tool_calls(text):
    """Extract [(name, kwargs)] exactly as TRL's regex schema would."""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        m = FN_RE.search(block)
        if not m:
            continue
        args = {}
        for a in ARG_RE.finditer(block):
            v = a.group("value")
            try:                       # TRL parses values as JSON with allow_non_json
                v = json.loads(v)
            except Exception:
                pass
            args[a.group("key")] = v
        calls.append((m.group(1), args))
    return calls


def run_episode(tok, template, tools, scenario, url, step_cap, max_tokens, temperature):
    env = SynthEnv()
    task = env.reset(scenario_id=scenario["id"], files=json.dumps(scenario["files"]),
                     check=scenario["check"], prompt=[], prompt_text=scenario["prompt"])
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Complete the coding task using the available tools.\n" + task}]
    try:
        for _ in range(step_cap):
            prompt = tok.apply_chat_template(msgs, tools=tools, chat_template=template,
                                             tokenize=False, add_generation_prompt=True)
            r = requests.post(f"{url}/v1/completions", timeout=600, json={
                "model": "base", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": temperature, "stop": ["<|im_end|>"]})
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
            calls = parse_tool_calls(text)
            reasoning = re.search(r"<think>([\s\S]*?)</think>", text)
            content = TOOL_CALL_RE.sub("", re.sub(r"<think>[\s\S]*?</think>", "", text)).strip()
            am = {"role": "assistant", "content": content}
            if reasoning:
                am["reasoning_content"] = reasoning.group(1).strip()
            if calls:
                am["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}}
                                    for n, a in calls]
            msgs.append(am)
            if not calls:
                break                                  # no tool call -> the agent is done
            for name, args in calls:
                try:
                    out = str(getattr(env, name)(**args))
                except Exception as exc:
                    out = f"{{'error': \"{type(exc).__name__}: {exc}\"}}"
                msgs.append({"role": "tool", "content": out[:8000]})
        reward = env.get_reward()
    finally:
        env._cleanup()
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="samples per task (pass rate resolution)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--step-cap", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--url", default="http://127.0.0.1:1234")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="tokenizer source")
    ap.add_argument("--out", default=str(OUTD / "difficulty_trl.json"))
    a = ap.parse_args()

    import trl
    from transformers import AutoTokenizer
    from transformers.utils import get_json_schema
    import synth_scenarios, mined_scenarios

    tok = AutoTokenizer.from_pretrained(a.model)
    template = (Path(trl.__file__).parent / "chat_templates" / "qwen3_5_think.jinja").read_text()
    probe = SynthEnv()
    tools = [get_json_schema(getattr(probe, n)) for n in ("read_file", "write_file", "list_dir", "run_bash")]

    scenarios = list(synth_scenarios.SCENARIOS) + list(mined_scenarios.SCENARIOS)
    jobs = [s for s in scenarios for _ in range(a.k)]
    print(f"{len(jobs)} episodes = {len(scenarios)} tasks x k={a.k} | TRL template + TRL tool parser", flush=True)

    res, rewards = defaultdict(list), defaultdict(list)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_episode, tok, template, tools, s, a.url, a.step_cap,
                          a.max_tokens, a.temperature): s["id"] for s in jobs}
        for f in as_completed(futs):
            tid = futs[f]
            done += 1
            try:
                rw = f.result()
                res[tid].append(rw == 1.0)
                rewards[tid].append(rw)
            except Exception as exc:
                print(f"  [{tid}] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)} ({round(time.time()-t0)}s)", flush=True)

    rates = {t: (sum(v) / len(v), len(v)) for t, v in res.items()}
    band = sorted(t for t, (r, _) in rates.items() if 0.20 <= r <= 0.80)
    print("\n=== per-task pass rate (TRL-faithful) ===", flush=True)
    for t, (r, n) in sorted(rates.items(), key=lambda x: -x[1][0]):
        mean_r = sum(rewards[t]) / max(1, len(rewards[t]))
        print(f"  {t:38s} pass {r:.2f}  meanR {mean_r:.2f}  (n={n}){'  <-- BAND' if t in band else ''}", flush=True)
    solved = [t for t, (r, _) in rates.items() if r > 0]
    print(f"\ntasks ever solved : {len(solved)}/{len(rates)}", flush=True)
    print(f"RLVR band (0.2-0.8): {len(band)}/{len(rates)}", flush=True)
    Path(a.out).write_text(json.dumps(
        {"rates": {k: v[0] for k, v in rates.items()},
         "mean_reward": {k: sum(v) / max(1, len(v)) for k, v in rewards.items()},
         "band": band}, indent=1))
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
