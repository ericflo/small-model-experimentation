#!/usr/bin/env python3
"""Agentic-domain arbitration of the confidence policy (C57): does confidence-
gated selection/allocation generalize from static code/reasoning benchmarks to
the multi-step GYM atoms? Samples K think-mode candidates per gym atom on vLLM,
scores each with the gym's own machine verifier (family.score_atom), and reads a
verifier-free confidence = mean logprob of the ANSWER-region tokens (C40's
P(answer), the natural signal for terse reasoning answers). Writes a pool with
per-candidate {score, answer_value, conf, n_think} for the frontier/adaptive
analysis. My own gym (not firewalled). Run under .venv-vllm.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
_VENV_BIN = EXP.parents[1] / ".venv-vllm" / "bin"
if _VENV_BIN.is_dir():
    os.environ["PATH"] = f"{_VENV_BIN}:" + os.environ.get("PATH", "")
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402
from gym.families import load as load_family  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
THINK_CLOSE = 248069
OUT = EXP / "runs" / "gym_confidence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["glyphgate", "burrowmaze", "loomfix", "packhouse"])
    ap.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--n", type=int, default=20, help="atoms per (family,level)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--answer-max", type=int, default=96)
    ap.add_argument("--seed", type=int, default=90211)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    for fam in a.families:
        F = load_family(fam)
        for lvl in a.levels:
            if lvl in F.LEVELS:
                for it in F.gen_atoms(a.seed, lvl, a.n):
                    items.append((fam, it))
    print(f"[gc] {len(items)} atoms from {a.families} L{a.levels}", flush=True)

    llm = LLM(model=MODEL_ID, revision=REVISION, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=8192,
              enforce_eager=False, trust_remote_code=True)
    tok = llm.get_tokenizer()
    eos = tok.eos_token_id
    close_ids = tok("</think>\n\n", add_special_tokens=False).input_ids

    def render(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)

    prompt_ids = [tok(render(it["prompt"]), add_special_tokens=False).input_ids for _, it in items]
    jobs = [(i, g) for i in range(len(items)) for g in ([True] + [False] * a.k)]

    def sp(greedy, mx, stop, logprobs=None):
        if greedy:
            return SamplingParams(temperature=0.0, max_tokens=mx, stop_token_ids=stop, logprobs=logprobs)
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=mx,
                              stop_token_ids=stop, logprobs=logprobs)

    # phase 1: thinking
    o1 = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[i]) for i, _ in jobs],
                      [sp(g, a.budget, [THINK_CLOSE, eos]) for _, g in jobs])
    thinking = []
    for out in o1:
        t = list(out.outputs[0].token_ids)
        if t and t[-1] in (THINK_CLOSE, eos):
            t = t[:-1]
        thinking.append(t)
    # phase 2: answer, with per-token logprobs -> mean answer logprob = confidence
    o2 = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids[i] + thinking[j] + close_ids)
                       for j, (i, _) in enumerate(jobs)],
                      [sp(g, a.answer_max, [eos], logprobs=1) for _, g in jobs])

    rows = [{"family": items[i][0], "id": items[i][1]["id"], "level": items[i][1]["level"],
             "cands": []} for i in range(len(items))]
    for j, (i, is_greedy) in enumerate(jobs):
        ans_toks = list(o2[j].outputs[0].token_ids)
        lps = o2[j].outputs[0].logprobs or []
        # mean logprob over emitted answer tokens (exclude a trailing eos)
        vals = []
        for pos, tid in enumerate(ans_toks):
            if tid == eos:
                continue
            if pos < len(lps) and lps[pos] and tid in lps[pos]:
                vals.append(lps[pos][tid].logprob)
        conf = statistics.mean(vals) if vals else -99.0
        text = tok.decode(thinking[j] + close_ids + [t for t in ans_toks if t != eos], skip_special_tokens=True)
        fam, it = items[i]
        rows[i]["cands"].append({
            "tag": "greedy" if is_greedy else f"s{len(rows[i]['cands'])}",
            "score": float(load_family(fam).score_atom(it, text)),
            "answer_value": base.extract_answer(text),
            "conf": round(conf, 4), "n_think": len(thinking[j])})

    outp = OUT / f"pool_b{a.budget}.json"
    outp.write_text(json.dumps(rows, indent=0) + "\n")
    allc = [c for r in rows for c in r["cands"]]
    print(f"[gc] wrote {outp} | {len(rows)} atoms x {a.k+1} cands | "
          f"per-cand score {statistics.mean(c['score'] for c in allc):.3f} | "
          f"solvable@k {statistics.mean(any(c['score']>=1.0 for c in r['cands']) for r in rows):.3f}", flush=True)


if __name__ == "__main__":
    main()
