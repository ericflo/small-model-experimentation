#!/usr/bin/env python3
"""Beat sample-more with the model's own uncertainty. For a MIX of problems (easy execute / coverage-limited
familiar-induce / capability-limited novel-induce) on the C40 successor task, collect: a cheap GREEDY pass
(answer + P(answer) = the C40 confidence signal, one forward pass) and k SAMPLED answers (each with its own
P(answer)). Downstream (analyze.py) simulates policies at matched forward-pass budget: uniform+random/self-
consistency/oracle-select vs confidence-guided ALLOCATION and ABSTENTION. The thesis: P(answer) distinguishes
COVERAGE-limited (sampling raises pass@k) from CAPABILITY-limited (sampling futile) problems, so you spend samples
where they pay off and abstain where they don't."""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import succ_family as SF  # noqa: E402
DIGIT_IDS = [15 + d for d in range(10)]
MIX = {"execute": ("familiar", True), "familiar_induce": ("familiar", False), "novel_induce": ("novel", False)}


def parse_ans(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"[Aa]nswer:\s*\**\s*(\d)", text)
    return m[-1] if m else (re.findall(r"\d", text)[-1] if re.findall(r"\d", text) else "")


@torch.no_grad()
def digit_probs(p, prefixes, batch_size=64):
    out = [None] * len(prefixes); pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0, len(order), batch_size):
        sub = order[s:s + batch_size]; seqs = [prefixes[i] for i in sub]; ml = max(len(x) for x in seqs)
        ids = torch.tensor([[pad] * (ml - len(x)) + x for x in seqs], device=p.device)
        o = p.model(input_ids=ids, attention_mask=(ids != pad).long(), logits_to_keep=1)
        pr = torch.softmax(o.logits[:, -1, DIGIT_IDS].float(), dim=1).cpu().tolist()
        for i, v in zip(sub, pr): out[i] = v
    return out


def p_of_answer(p, prompt, gen_text):
    mk = gen_text.rfind("Answer:")
    reason = gen_text[:mk] if mk >= 0 else gen_text
    return p._ids(prompt + reason + "Answer: ")


def gen_and_score(p, prompts, greedy):
    """Generate (greedy or sampled) then read each output's P(answer). Returns list of (answer, p_answer)."""
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=greedy, answer_max=220, batch_size=64)
    texts = [p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=True) for pr, g in zip(prompts, gg)]
    answers = [parse_ans(t) for t in texts]
    prefixes = [p_of_answer(p, pr, t) for pr, t in zip(prompts, texts)]
    dprobs = digit_probs(p, prefixes)
    res = []
    for a, dp in zip(answers, dprobs):
        pa = dp[int(a)] if a.isdigit() else 0.0
        res.append((a, round(pa, 4)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=700)
    args = ap.parse_args()
    import gen_lib as GL
    p = GL.Probe()
    records = []
    for cond, (kind, stated) in MIX.items():
        tasks = [SF.gen_task(kind, args.seed * 100000 + i) for i in range(args.n)]
        prompts = [p.prompt(SF.render(t, rule_stated=stated), enable_thinking=False) for t in tasks]
        greedy = gen_and_score(p, prompts, greedy=True)
        # k sampled draws: replicate prompts, generate+score, regroup
        rep = [pr for pr in prompts for _ in range(args.k)]
        srep = gen_and_score(p, rep, greedy=False)
        for i, t in enumerate(tasks):
            samples = srep[i * args.k:(i + 1) * args.k]
            records.append({"cond": cond, "true": t["answer"],
                            "greedy_ans": greedy[i][0], "greedy_p": greedy[i][1],
                            "greedy_correct": int(greedy[i][0] == t["answer"]),
                            "samples": [{"a": a, "p": pa, "c": int(a == t["answer"])} for a, pa in samples]})
        rc = [r for r in records if r["cond"] == cond]
        g_acc = sum(r["greedy_correct"] for r in rc) / len(rc)
        passk = sum(any(s["c"] for s in r["samples"]) for r in rc) / len(rc)
        print(f"[cgc] {cond}: greedy_acc={g_acc:.2f} pass@{args.k}={passk:.2f} mean_greedy_p={sum(r['greedy_p'] for r in rc)/len(rc):.2f}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump({"k": args.k, "n": args.n, "records": records}, open(EXP / "runs" / "sampling_records.json", "w"))
    print(f"[cgc] wrote {len(records)} problem records (k={args.k})", flush=True)


if __name__ == "__main__":
    main()
