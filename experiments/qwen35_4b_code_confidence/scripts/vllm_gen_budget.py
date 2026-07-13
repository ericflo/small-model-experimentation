#!/usr/bin/env python3
"""vLLM port of gen_budget.py — think-mode MBPP candidate pools at several think
budgets, MUCH faster than the HF backend. Replicates the pipeline token-for-token:
two-phase budget forcing (generate up to B think tokens; if </think> was not
emitted, force it via "</think>\\n\\n" and regenerate the answer), the same
execution scoring (full_pass), and the same single-token P(True) judge
(softmax of logit_A vs logit_B after "Answer: ", A_ID=32/B_ID=33). Run under
.venv-vllm.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
# vLLM's JIT (FlashInfer/inductor) needs `ninja`, which lives in the vllm venv's
# bin; put it on PATH so the engine can compile without a caller-set env.
_VENV_BIN = EXP.parents[1] / ".venv-vllm" / "bin"
if _VENV_BIN.is_dir():
    os.environ["PATH"] = f"{_VENV_BIN}:" + os.environ.get("PATH", "")
sys.path.insert(0, str(EXP))
from src.coverage_utils import sampling_prompt, candidate_from_completion, mbpp_record  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
THINK_CLOSE = 248069
A_ID, B_ID = 32, 33
OUT = EXP.parents[1] / "experiments" / "qwen35_4b_confidence_policy" / "runs"


def load_records(count, offset, visible_tests=1, timeout_s=5.0):
    import ast as _ast
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp")["test"]
    recs = []
    for raw in list(ds)[offset: offset + count * 2]:
        tl = raw.get("test_list")
        if isinstance(tl, str):
            try: tl = _ast.literal_eval(tl)
            except Exception: continue
        imports = raw.get("test_imports")
        if isinstance(imports, str):
            try: imports = _ast.literal_eval(imports)
            except Exception: imports = []
        text = raw.get("prompt") or raw.get("text")
        if not text or not tl:
            continue
        rec = mbpp_record({"task_id": raw["task_id"], "text": text, "test_list": tl,
                           "test_setup_code": "\n".join(imports or []),
                           "code": raw.get("code", "")}, "heldout", visible_tests, timeout_s)
        if rec is not None:
            recs.append(rec)
        if len(recs) >= count:
            break
    return recs


def judge_prompt(tok, task_text, code):
    sys_m = "You are a strict Python code reviewer. Decide if a candidate solution solves the task."
    user = (f"Task:\n{task_text}\n\nCandidate solution:\n```python\n{code}\n```\n\n"
            f"Does this solution correctly solve the task (pass all valid tests)? "
            f"Answer with a single letter: A = correct, B = incorrect.")
    msgs = [{"role": "system", "content": sys_m}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_pool(llm, tok, records, prompts, budget, k, answer_max):
    eos = tok.eos_token_id
    close_ids = tok("</think>\n\n", add_special_tokens=False).input_ids
    ans_ids = tok("Answer: ", add_special_tokens=False).input_ids
    prompt_ids = [tok(pr, add_special_tokens=False).input_ids for pr in prompts]
    # After "Answer: " the model emits the SPACE-PREFIXED " A"/" B" (ids 357/417),
    # not the bare "A"/"B" (32/33); accept either variant, whichever the model puts
    # in its top logprobs, so the P(True) readout is not a degenerate 0.5.
    a_ids = {tok(" A", add_special_tokens=False).input_ids[-1], A_ID}
    b_ids = {tok(" B", add_special_tokens=False).input_ids[-1], B_ID}

    def p_true_from(lp):
        la = max((lp[t].logprob for t in a_ids if t in lp), default=-20.0)
        lb = max((lp[t].logprob for t in b_ids if t in lp), default=-20.0)
        return math.exp(la) / (math.exp(la) + math.exp(lb))

    # one greedy + k sampled draws per task, flattened
    jobs = []  # (task_index, is_greedy)
    for ri in range(len(records)):
        jobs.append((ri, True))
        for _ in range(k):
            jobs.append((ri, False))

    def sp(is_greedy, max_tokens, stop_ids, logprobs=None):
        if is_greedy:
            return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                  stop_token_ids=stop_ids, logprobs=logprobs)
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens,
                              stop_token_ids=stop_ids, logprobs=logprobs)

    # phase 1: thinking up to budget, stop at </think>
    p1_prompts = [TokensPrompt(prompt_token_ids=prompt_ids[ri]) for ri, _ in jobs]
    p1_params = [sp(g, budget, [THINK_CLOSE, eos]) for _, g in jobs]
    o1 = llm.generate(p1_prompts, p1_params)
    thinking, forced = [], []
    for out in o1:
        toks = list(out.outputs[0].token_ids)
        closed = out.outputs[0].finish_reason == "stop" and (not toks or toks[-1] != eos)
        # strip a trailing stop/eos token if present
        if toks and toks[-1] in (THINK_CLOSE, eos):
            toks = toks[:-1]
        thinking.append(toks)
        forced.append(not closed)

    # phase 2: force close, generate the answer
    p2_prompts = [TokensPrompt(prompt_token_ids=prompt_ids[ri] + thinking[j] + close_ids)
                  for j, (ri, _) in enumerate(jobs)]
    p2_params = [sp(g, answer_max, [eos]) for _, g in jobs]
    o2 = llm.generate(p2_prompts, p2_params)

    # assemble candidates
    rows = [{"task_id": records[ri]["task_id"], "cands": []} for ri in range(len(records))]
    cand_meta = []  # (task_index, cand_index, task_text) for the judge
    for j, (ri, is_greedy) in enumerate(jobs):
        ans_toks = [t for t in o2[j].outputs[0].token_ids if t != eos]
        full_ids = thinking[j] + close_ids + ans_toks
        text = tok.decode(full_ids, skip_special_tokens=True)
        cand = candidate_from_completion(text, records[ri],
                                         source=("greedy" if is_greedy else "s"),
                                         order=len(rows[ri]["cands"]))
        entry = {"tag": "greedy" if is_greedy else f"s{len(rows[ri]['cands'])}",
                 "full_pass": bool(cand["full_pass"]),
                 "visible_all_pass": bool(cand.get("visible_all_pass", False)),
                 "behavior_signature": cand.get("behavior_signature", ""),
                 "parse_ok": cand["parse_status"] == "parsed",
                 "code": cand["code"], "code_len": len(cand["code"]),
                 "n_think": len(thinking[j]), "forced": bool(forced[j])}
        rows[ri]["cands"].append(entry)
        if entry["parse_ok"] and entry["code"]:
            cand_meta.append((ri, len(rows[ri]["cands"]) - 1, records[ri]["task_text"], entry["code"]))

    # judge: P(True) = softmax(logit_A, logit_B) after "Answer: "
    jprompts = [TokensPrompt(prompt_token_ids=tok(judge_prompt(tok, tt, code),
                                                  add_special_tokens=False).input_ids + ans_ids)
                for (_, _, tt, code) in cand_meta]
    jparams = [SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)] * len(jprompts)
    oj = llm.generate(jprompts, jparams)
    for (ri, ci, _, _), out in zip(cand_meta, oj):
        rows[ri]["cands"][ci]["p_true"] = round(p_true_from(out.outputs[0].logprobs[0]), 4)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--answer-max", type=int, default=420)
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 2048])
    ap.add_argument("--suffix", type=str, default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records(a.n, a.offset)
    print(f"[vgb] {len(records)} MBPP records; budgets {a.budgets}", flush=True)
    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=8192, enforce_eager=False,
              trust_remote_code=True)
    tok = llm.get_tokenizer()
    prompts = [sampling_prompt(r, tok) for r in records]
    for b in a.budgets:
        print(f"[vgb] === budget {b} ===", flush=True)
        rows = build_pool(llm, tok, records, prompts, b, a.k, a.answer_max)
        outp = OUT / f"pool_think_b{b}{a.suffix}.json"
        json.dump(rows, open(outp, "w"))
        nall = sum(len(r["cands"]) for r in rows)
        npass = sum(c["full_pass"] for r in rows for c in r["cands"])
        mth = sum(c["n_think"] for r in rows for c in r["cands"]) / max(1, nall)
        print(f"[vgb] wrote {outp} | pass-rate {npass/nall:.3f} | mean n_think {mth:.0f}", flush=True)


if __name__ == "__main__":
    main()
