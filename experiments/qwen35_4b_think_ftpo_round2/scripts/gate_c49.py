#!/usr/bin/env python3
"""C49 adapter-application gate: on-vs-off greedy behavioral diff.

vLLM runtime LoRA silently no-ops on Qwen3.5-4B (claim C49), so installs are
merged composite checkpoints. This gate proves a merged checkpoint actually
differs behaviorally from base: identical greedy outputs on the probe set =
FAIL (the merge did not apply, or the adapter is a no-op).

One engine per invocation (VRAM discipline):
  ../../.venv-vllm/bin/python scripts/gate_c49.py --model base --out runs/gate_base.json
  ../../.venv-vllm/bin/python scripts/gate_c49.py --model <merged_dir> --out runs/gate_pivot.json
  python3 scripts/gate_c49.py --compare runs/gate_base.json runs/gate_pivot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

PROBE_PROMPTS = [
    "List three prime numbers greater than 40 and justify each briefly.",
    "A ledger shows credits of 120, 45, and 80, and debits of 60 and 95. What is the net balance?",
    "Write a Python function `dedupe(xs)` that removes duplicates while preserving order.",
    "You have a 3-liter and a 5-liter jug. Measure exactly 4 liters. Give the steps.",
    "Which is larger: 7^5 or 5^7? Show the comparison.",
    "Rewrite 'the quick brown fox jumps over the lazy dog' with every word reversed.",
    "A state machine has states A->B on x, B->C on y, C->A on z. Starting at A, where does xyzxy end?",
    "Sum the digits of 987654321 and state whether the result is divisible by 9.",
]


def generate(model: str, out: Path) -> int:
    import harness
    import yaml
    from vllm_runner import SamplingConfig

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    runner = harness.make_runner(
        cfg["engine"], model_override=None if model == "base" else model)
    records = [{"id": f"probe-{i}", "messages": [{"role": "user", "content": p}]}
               for i, p in enumerate(PROBE_PROMPTS)]
    sampling = SamplingConfig(thinking="budget", thinking_budget=512, n=1,
                              answer_max_tokens=256, greedy=True, run_seed=99)
    rows, _ = runner.generate(records, sampling)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model,
        "outputs": {row["id"]: row["outputs"][0]["text"] for row in rows},
    }, indent=2))
    print(f"wrote {out}")
    return 0


def compare(path_a: Path, path_b: Path) -> int:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    same = sum(1 for k in a["outputs"] if a["outputs"][k] == b["outputs"].get(k))
    total = len(a["outputs"])
    print(f"identical outputs: {same}/{total} ({a['model']} vs {b['model']})")
    if same == total:
        print("C49 GATE FAIL: merged model is behaviorally identical to base")
        return 1
    print("C49 GATE PASS: merged model differs behaviorally")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="'base' or merged checkpoint dir")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.model or not args.out:
        parser.error("--model and --out required unless --compare")
    return generate(args.model, args.out)


if __name__ == "__main__":
    sys.exit(main())
