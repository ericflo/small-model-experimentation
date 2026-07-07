#!/usr/bin/env python3
"""Does the model KNOW when it will fail? Review-hardened. Verified competence spectrum (C39) on a format-equalized
single-value task. Two NON-DEGENERATE logit confidence signals (verbalized 0-100 is a constant 100 -- degenerate):
  IMPLICIT  P(answer): the model's probability on the digit it actually emits (softmax over the 10 digit tokens at
            the 'Answer: ' position, one forward pass) + entropy + top-2 margin.
  EXPLICIT  P(True): Kadavath-style self-verification -- 'is your answer correct? A/B', read P(A).
Conditions: familiar_execute (anchor ~1.0), familiar_induce (HEADLINE cell, intermediate acc, surface-matched),
reversal_induce (DISSOCIATION: scrambled-LOOKING but easy), novel_induce (chance). Per-item records + surface
features (for the external baseline). Headline = WITHIN familiar_induce, do the model's signals predict per-item
correctness beyond surface? No-think (answer channel; think triggers code-mode -- C39)."""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import succ_family as SF  # noqa: E402
DIGIT_IDS = [15 + d for d in range(10)]  # tokens '0'..'9' (verified)

CONDS = {
    "familiar_execute": ("familiar", True),
    "familiar_induce":  ("familiar", False),
    "reversal_induce":  ("reversal", False),
    "novel_induce":     ("novel", False),
}


def parse_ans(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"[Aa]nswer:\s*\**\s*(\d)", text)
    return m[-1] if m else (re.findall(r"\d", text)[-1] if re.findall(r"\d", text) else "")


@torch.no_grad()
def digit_probs(p, prefixes, batch_size=32):
    """prefixes: token-id lists ending in 'Answer: '. Returns per-prefix softmax over the 10 digit tokens."""
    out = [None] * len(prefixes); pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0, len(order), batch_size):
        sub = order[s:s + batch_size]; seqs = [prefixes[i] for i in sub]
        ml = max(len(x) for x in seqs)
        ids = torch.tensor([[pad] * (ml - len(x)) + x for x in seqs], device=p.device)
        attn = (ids != pad).long()
        o = p.model(input_ids=ids, attention_mask=attn, logits_to_keep=1)
        pr = torch.softmax(o.logits[:, -1, DIGIT_IDS].float(), dim=1).cpu().tolist()
        for i, v in zip(sub, pr): out[i] = v
    return out


def ptrue_prompt(t, ans, stated):
    body = SF.render(t, rule_stated=stated).rsplit("\n", 1)[0]
    return (f"{body}\n\nA student proposes this answer: {ans}\n"
            f"Is the student's answer correct? A = correct, B = incorrect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=500)
    args = ap.parse_args()
    import gen_lib as GL
    p = GL.Probe()
    records = []
    for cond, (kind, stated) in CONDS.items():
        tasks = [SF.gen_task(kind, args.seed * 100000 + i) for i in range(args.n)]
        prompts = [p.prompt(SF.render(t, rule_stated=stated), enable_thinking=False) for t in tasks]
        gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=260, batch_size=48)
        answers, conf_prefixes = [], []
        for t, pr, g in zip(tasks, prompts, gg):
            gtext = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=True)
            ans = parse_ans(gtext); answers.append(ans)
            mk = gtext.rfind("Answer:")
            reason = gtext[:mk] if mk >= 0 else gtext
            conf_prefixes.append(p._ids(pr + reason + "Answer: "))
        dprobs = digit_probs(p, conf_prefixes)
        # explicit P(True) self-verification
        ptrue = p._judge_logit([p._ids(p.prompt(ptrue_prompt(t, a or "0", stated), enable_thinking=False) + "Answer: ")
                                for t, a in zip(tasks, answers)])
        for t, a, dp, pt in zip(tasks, answers, dprobs, ptrue):
            ai = int(a) if a.isdigit() else None
            srt = sorted(dp, reverse=True)
            records.append({"cond": cond, "answer": a, "true": t["answer"], "natural_succ": t["natural_succ"],
                            "correct": int(a == t["answer"]),
                            "p_answer": round(dp[ai], 4) if ai is not None else 0.0,
                            "entropy": round(-sum(x * math.log(max(x, 1e-9)) for x in dp), 4),
                            "margin": round(srt[0] - srt[1], 4), "p_true": round(pt, 4), "feats": t["feats"]})
        rc = [r for r in records if r["cond"] == cond]
        acc = sum(r["correct"] for r in rc) / len(rc)
        print(f"[meta] {cond}: acc={acc:.2f} mean_p_answer={sum(r['p_answer'] for r in rc)/len(rc):.2f} "
              f"mean_p_true={sum(r['p_true'] for r in rc)/len(rc):.2f}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(records, open(EXP / "runs" / "metacog_records.json", "w"), indent=1)
    print(f"[meta] wrote {len(records)} records", flush=True)


if __name__ == "__main__":
    main()
