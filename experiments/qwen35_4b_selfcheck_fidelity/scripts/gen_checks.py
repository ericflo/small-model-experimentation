#!/usr/bin/env python3
"""Generate an executable check suite for each task from its SPEC ALONE (the firewall).

The model sees: package name, file path, qualified function name, and the function's signature +
docstring. It never sees candidate implementations, the repo's own tests, or any other source. That is
the information an everyday-task assistant would have -- a description of what "done" means -- so
fidelity measured under this firewall is the number that decides whether self-written verification can
gate search on tasks that ship no test suite.

Thinking is ON (repo default) with a generous budget; the check file is parsed from the last fenced
python block after the think close. Suites are written to data/checks/<task_id>.py and quality-gated
later by run_fidelity.py against two anchors it can compute without any model: the suite must FAIL on
the STUB (a suite that passes on `raise NotImplementedError` is vacuous) and PASS on the ORIGINAL
implementation (a suite inconsistent with true behavior is wrong, not strict).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.S)

PROMPT = """You are writing an independent acceptance test for a Python function, from its documented \
contract only. You have NOT seen any implementation.

Package: `{package}` (already importable). The function `{qual_name}` lives in `{rel_file}`.

Its signature and docstring:
```python
{spec}
```

Write a standalone pytest file that checks whether an arbitrary implementation of `{qual_name}` \
honors this contract. Rules:
- 5 to 10 focused test functions named `test_*`, each asserting one documented behavior (typical \
cases, documented edge cases, documented error behavior).
- Derive expectations ONLY from the signature and docstring above plus the standard behavior its \
wording implies. Do not guess undocumented internals.
- Import the function from the installed package (e.g. `from {package} import ...` or import the \
module and access the attribute; for methods, construct the object as the docstring implies).
- The file must be self-contained: no fixtures from the package's own test suite, no network, no files.
- Output ONLY the complete file in a single ```python code block."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=str(DATA / "specs.jsonl"))
    ap.add_argument("--out-dir", default=str(DATA / "checks"))
    ap.add_argument("--max-new", type=int, default=6144)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()

    specs = [json.loads(l) for l in open(a.specs)]
    outd = Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (outd / (s["task_id"].replace("/", "-").replace("::", "__") + ".py")).exists()]
    print(f"{len(todo)}/{len(specs)} suites to generate", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model.eval()

    meta = {}
    for i in range(0, len(todo), a.bs):
        sub = todo[i:i + a.bs]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(**s)}],
            tokenize=False, add_generation_prompt=True) for s in sub]   # thinking ON (default)
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for s, row in zip(sub, gen):
            txt = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            body = txt.split("</think>")[-1]
            m = FENCE.findall(body) or FENCE.findall(txt)
            fn = s["task_id"].replace("/", "-").replace("::", "__") + ".py"
            n_tok = len(tok(txt, add_special_tokens=False).input_ids)
            if m:
                (outd / fn).write_text(m[-1])
                meta[s["task_id"]] = {"gen_tokens": n_tok, "parsed": True,
                                      "n_tests": m[-1].count("def test_")}
            else:
                meta[s["task_id"]] = {"gen_tokens": n_tok, "parsed": False}
            print(f"  {s['task_id'][:60]:60s} {'ok' if m else 'NO-FENCE'} "
                  f"({meta[s['task_id']].get('n_tests','-')} tests, {n_tok} tok)", flush=True)
        (DATA / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    unparsed = sum(1 for v in meta.values() if not v.get("parsed"))
    truncated = sum(1 for v in meta.values() if v.get("gen_tokens", 0) >= a.max_new - 8)
    print(f"\ndone: {len(meta)} generated, {unparsed} unparsed, {truncated} at the {a.max_new} cap", flush=True)
    (outd / ".DONE").write_text(json.dumps({"n": len(meta), "unparsed": unparsed}) + "\n")


if __name__ == "__main__":
    main()
