#!/usr/bin/env python3
"""HF glyphgate eval (base or +adapter, no merge): think-mode generate on held-out
glyphgate atoms L1-L6, parse ANSWER, score with the gym verifier. Measures whether
a curriculum installs composed-rule induction (L4-L6)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "src"))
from gym.families import glyphgate as G  # noqa
from gym import base as B  # noqa
MID = "Qwen/Qwen3.5-4B"; REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"; TC = 248069


@torch.no_grad()
def gen(model, tok, prompts, budget, ansmax=64, bs=16):
    close = tok("</think>\n\n", add_special_tokens=False).input_ids
    texts = [None]*len(prompts)
    for s in range(0, len(prompts), bs):
        sub = list(range(s, min(s+bs, len(prompts))))
        enc = tok([prompts[i] for i in sub], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g1 = model.generate(**enc, max_new_tokens=budget, do_sample=False, pad_token_id=tok.pad_token_id,
                            eos_token_id=[TC, tok.eos_token_id])
        conts = []
        for r, i in enumerate(sub):
            think = g1[r][enc.input_ids.shape[1]:].tolist()
            think = think[:think.index(TC)] if TC in think else [t for t in think if t != tok.eos_token_id]
            conts.append(enc.input_ids[r].tolist()[enc.attention_mask[r].sum().item()*0:] if False else None)
            texts[i] = ("__PRE__", enc.input_ids[r][enc.attention_mask[r].bool()].tolist(), think)
    # phase 2: force close + answer
    out = [None]*len(prompts)
    for s in range(0, len(prompts), bs):
        sub = list(range(s, min(s+bs, len(prompts))))
        seqs = [texts[i][1] + texts[i][2] + close for i in sub]
        ml = max(len(x) for x in seqs); pad = tok.pad_token_id
        ids = torch.tensor([[pad]*(ml-len(x))+x for x in seqs], device="cuda")
        g2 = model.generate(input_ids=ids, attention_mask=(ids != pad).long(), max_new_tokens=ansmax,
                            do_sample=False, pad_token_id=pad, eos_token_id=tok.eos_token_id)
        for r, i in enumerate(sub):
            ans = g2[r][ids.shape[1]:].tolist(); ans = [t for t in ans if t != tok.eos_token_id]
            full = tok.decode(texts[i][2] + close + ans, skip_special_tokens=True)
            out[i] = full
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=15); ap.add_argument("--budget", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=90501); a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID, revision=REV, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MID, revision=REV, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    if a.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, a.adapter).eval()
    print(f"[gg] adapter={a.adapter}")
    for lvl in G.LEVELS:
        items = G.gen_atoms(a.seed, lvl, a.n)
        prompts = [tok.apply_chat_template([{"role":"user","content":it["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for it in items]
        outs = gen(model, tok, prompts, a.budget)
        acc = sum(G.score_atom(it, o) for it, o in zip(items, outs))/len(items)
        print(f"  L{lvl}: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
