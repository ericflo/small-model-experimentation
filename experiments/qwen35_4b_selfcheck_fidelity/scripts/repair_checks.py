#!/usr/bin/env python3
"""One ORACLE-FREE repair round for generated check suites.

V1's failure taxonomy (reports/fidelity.json): of 60 suites, 35 never emitted code because thinking hit
the 6144 cap (generation truncation -- the same artifact class this repo has now hit four times), 19
parsed but CRASH on import/setup (orig_frac exactly 0.0), and only ONE asserted genuinely wrong
behavior. So the deficits are budget and scaffolding, not check-writing ability -- and scaffolding
errors are detectable WITHOUT any reference implementation: run the suite against the STUBBED checkout
and read pytest's own output.

The repair signal is deployment-honest by construction. Feedback comes only from (a) collection/import
errors and (b) the vacuousness signal "your tests PASSED against a function that raises
NotImplementedError -- they cannot be exercising it". The ORIGINAL implementation is never consulted
here; it remains a quality GATE in run_fidelity.py, never a repair signal, because repairing against
the reference would leak ground-truth behavior into the suite -- the thing an everyday-task assistant
cannot have.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
DATA = HERE.parent / "data"

import env_util  # noqa: E402
import stub as stubmod  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)

REPAIR_PROMPT = """Your pytest file below is meant to test `{qual_name}` from package `{package}` \
against its documented contract, but running it against a STUB implementation (the function body is \
just `raise NotImplementedError`) gave the result shown. {problem}

Your file:
```python
{suite}
```

Pytest output against the stub:
```
{output}
```

Fix the file. Rules unchanged: 5-10 `test_*` functions asserting only documented behavior; import from \
the installed package; self-contained. Against the stub, the tests SHOULD FAIL with \
NotImplementedError -- they must not error during collection/import, and they must not pass. \
Output ONLY the corrected file in one ```python block."""


def stub_result(spec_row, suite_src):
    """Run a suite against a stubbed checkout. Returns (verdict, pytest tail)."""
    repo = STORE / spec_row["repo"]
    copy = env_util.prepare_copy(repo, prefix=f"rep_{spec_row['repo']}_")
    try:
        if not stubmod.stub_function(copy, spec_row["rel_file"], spec_row["qual_name"]):
            return "recon_failed", ""
        (copy / "selfcheck_test.py").write_text(suite_src)
        res = env_util.run_tests(copy, repo, timeout=120,
                                 cmd="python -m pytest selfcheck_test.py -v -rA --tb=short -p no:cacheprovider")
        n_run = res["passed"] + res["failed"] + res["errors"]
        if n_run == 0 or res["errors"] > 0 and res["passed"] + res["failed"] == 0:
            return "crash", res["out"][-1800:]
        if res["failed"] + res["errors"] == 0:
            return "vacuous", res["out"][-1200:]
        return "ok", ""
    finally:
        shutil.rmtree(copy, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default=str(DATA / "checks_v2"))
    ap.add_argument("--max-new", type=int, default=16384)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()

    specs = {s["task_id"]: s for s in (json.loads(l) for l in open(DATA / "specs.jsonl"))}
    outd = Path(a.checks)
    todo = []
    for tid, s in specs.items():
        fn = outd / (tid.replace("/", "-").replace("::", "__") + ".py")
        if not fn.exists():
            continue
        verdict, out = stub_result(s, fn.read_text())
        print(f"  stub-run {tid[:56]:56s} {verdict}", flush=True)
        if verdict in ("crash", "vacuous"):
            problem = ("It ERRORED before the tests could run (import/collection/setup problem)."
                       if verdict == "crash" else
                       "Every test PASSED against the stub, so the tests are not exercising the function at all.")
            todo.append((s, fn, problem, out))
    print(f"\n{len(todo)} suites need repair", flush=True)
    if not todo:
        (outd / ".REPAIRED").write_text("{}\n")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()
    fixed = {}
    for i in range(0, len(todo), a.bs):
        sub = todo[i:i + a.bs]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": REPAIR_PROMPT.format(
                qual_name=s["qual_name"], package=s["package"], problem=prob,
                suite=fn.read_text()[:5000], output=out)}],
            tokenize=False, add_generation_prompt=True) for s, fn, prob, out in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for (s, fn, _, _), row in zip(sub, gen):
            txt = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            m = FENCE.findall(txt.split("</think>")[-1]) or FENCE.findall(txt)
            if m:
                fn.write_text(m[-1])
                v2, _ = stub_result(s, m[-1])
                fixed[s["task_id"]] = v2
                print(f"  repaired {s['task_id'][:52]:52s} -> {v2}", flush=True)
            else:
                fixed[s["task_id"]] = "unparsed"
        (outd / ".REPAIRED").write_text(json.dumps(fixed, indent=1))
    ok = sum(1 for v in fixed.values() if v == "ok")
    print(f"\nrepair round: {ok}/{len(fixed)} now clean against the stub", flush=True)


if __name__ == "__main__":
    main()
