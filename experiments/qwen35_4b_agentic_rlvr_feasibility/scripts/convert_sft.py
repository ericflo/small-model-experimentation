"""Convert harvested PASSING trajectories -> multi-turn tool-calling SFT rows (TRL conversational).

Assistant turns carry `reasoning_content` (the base's OWN harvested reasoning — C60: harvest, never
author) + `tool_calls`; tool results are role 'tool' (masked at train time). The tool schema is taken
from CodingEnv's methods so it EXACTLY matches the RLVR env; loop_raw's `run` is renamed `run_bash`.
Only keeps trajectories that actually completed the loop (contain write_file AND run_bash).

Output row: {"messages": [...], "tools": [...]}
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
from transformers.utils import get_json_schema
import coding_env

TOOL_METHODS = ["read_file", "write_file", "list_dir", "run_bash"]
TOOLS = [get_json_schema(getattr(coding_env.CodingEnv, m)) for m in TOOL_METHODS]
RENAME = {"run": "run_bash"}

SYSTEM = ("You are an expert Python coding agent working in a project directory. Think step by step, "
          "then use the tools to inspect files, write code, and run the tests until they pass. You MUST "
          "edit the target file with write_file and MUST run the tests with run_bash to verify before "
          "finishing. Do not give up after only reading files — always make your implementation and test "
          "it. Keep iterating until the tests pass.")


def convert_episode(ep):
    transcript = ep.get("transcript") or []
    reasonings = [t.get("reasoning", "") for t in transcript]
    hist = ep.get("history") or []
    user = next((m for m in hist if m["role"] == "user"), None)
    if user is None:
        return None
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user["content"]}]
    ai = 0
    for m in hist:
        if m["role"] == "assistant":
            calls = []
            for tc in (m.get("tool_calls") or []):
                fn = tc["function"]
                calls.append({"type": "function",
                              "function": {"name": RENAME.get(fn["name"], fn["name"]), "arguments": fn["arguments"]}})
            am = {"role": "assistant",
                  "reasoning_content": reasonings[ai] if ai < len(reasonings) else "",
                  "content": m.get("content", "") or ""}
            if calls:
                am["tool_calls"] = calls
            msgs.append(am)
            ai += 1
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "content": m["content"]})
    names = [tc["function"]["name"] for mm in msgs if mm["role"] == "assistant" for tc in mm.get("tool_calls", [])]
    if "write_file" not in names or "run_bash" not in names:
        return None  # not a completed loop
    return {"messages": msgs, "tools": TOOLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="harvested_*.jsonl files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = kept = 0
    with open(a.out, "w") as fh:
        for src in a.inputs:
            p = Path(src)
            if not p.exists():
                print(f"  (missing {src})"); continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                n += 1
                row = convert_episode(json.loads(line))
                if row:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n"); kept += 1
    print(f"converted {kept}/{n} episodes -> {a.out}")
    print(f"tools: {[t['function']['name'] for t in TOOLS]}")


if __name__ == "__main__":
    main()
