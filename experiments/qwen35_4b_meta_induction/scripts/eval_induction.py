#!/usr/bin/env python3
"""Eval induction/execution (base or +LoRA). INDUCE mode = forced 'Answer: ' digit read (one forward pass, argmax
over the 10 digit tokens) -- fast, deterministic, fair for base and answer-only-SFT. EXECUTE mode = generate WITH
reasoning and parse (the C39-0.97 ceiling: can the base APPLY a stated rule? -- gate for interpreting induction)."""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_gen as EG
MODEL_ID = "Qwen/Qwen3.5-4B"; EXP = Path(__file__).resolve().parents[1]
DIGIT_IDS = [15 + d for d in range(10)]; ANS = re.compile(r"[Aa]nswer:\s*\**\s*(\d)")


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="cuda",
                                                 dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def chat(tok, user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def induce_acc(model, tok, recs, render, batch_size=64):
    """Forced 'Answer: ' -> argmax digit."""
    correct = 0; got = []
    for s in range(0, len(recs), batch_size):
        sub = recs[s:s + batch_size]
        prompts = [chat(tok, render(r["ep"])) + "Answer: " for r in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc).logits[:, -1, DIGIT_IDS]
        pred = lg.argmax(-1).cpu().tolist()
        for r, d in zip(sub, pred): got.append(str(d)); correct += int(str(d) == r["ep"]["answer"])
    return correct / len(recs), Counter(got)


@torch.no_grad()
def execute_acc(model, tok, recs, batch_size=48, max_new=256):
    correct = 0
    for s in range(0, len(recs), batch_size):
        sub = recs[s:s + batch_size]
        prompts = [chat(tok, EG.render_execute(r["ep"])) for r in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, seq in zip(sub, out):
            txt = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = ANS.findall(txt); correct += int((m[-1] if m else "") == r["ep"]["answer"])
    return correct / len(recs)


@torch.no_grad()
def induce_gen_acc(model, tok, recs, batch_size=32, max_new=400):
    """Generate from the INDUCE prompt (let the model reason), parse Answer -- for reasoning-SFT models."""
    correct = 0
    for s in range(0, len(recs), batch_size):
        sub = recs[s:s + batch_size]
        prompts = [chat(tok, r["ep"]["_prompt"]) for r in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, seq in zip(sub, out):
            txt = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = ANS.findall(txt); correct += int((m[-1] if m else "") == r["ep"]["answer"])
    return correct / len(recs)


@torch.no_grad()
def strategy_acc(model, tok, recs, batch_size=32, max_new=320):
    correct = 0
    for s in range(0, len(recs), batch_size):
        sub = recs[s:s + batch_size]
        prompts = [chat(tok, EG.render_strategy(r["ep"])) for r in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, seq in zip(sub, out):
            txt = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = ANS.findall(txt); correct += int((m[-1] if m else "") == r["ep"]["answer"])
    return correct / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--mode", choices=["induce", "execute", "induce_gen", "strategy"], default="induce")
    ap.add_argument("--tag", default="base"); ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()][:a.limit]
    recs = []
    for r in rows:
        ep = EG.gen_episode(r["family"], r["seed"])   # deterministic full episode (examples + query)
        ep["_prompt"] = r["prompt"]
        recs.append({"ep": ep})
    model, tok = load(a.adapter)
    if a.mode == "induce":
        render = lambda ep: ep["_prompt"]
        acc, dist = induce_acc(model, tok, recs, render)
        print(f"[eval] {a.tag} INDUCE {a.data.name}: acc={acc:.3f} (n={len(recs)}, chance 0.1) | top={dist.most_common(3)}", flush=True)
        res = {"tag": a.tag, "mode": "induce", "data": a.data.name, "acc": round(acc, 3), "n": len(recs), "dist": dict(dist)}
    elif a.mode == "induce_gen":
        acc = induce_gen_acc(model, tok, recs)
        print(f"[eval] {a.tag} INDUCE_GEN {a.data.name}: acc={acc:.3f} (n={len(recs)}, chance 0.1)", flush=True)
        res = {"tag": a.tag, "mode": "induce_gen", "data": a.data.name, "acc": round(acc, 3), "n": len(recs)}
    elif a.mode == "strategy":
        acc = strategy_acc(model, tok, recs)
        print(f"[eval] {a.tag} STRATEGY {a.data.name}: acc={acc:.3f} (n={len(recs)}, chance 0.1)", flush=True)
        res = {"tag": a.tag, "mode": "strategy", "data": a.data.name, "acc": round(acc, 3), "n": len(recs)}
    else:
        acc = execute_acc(model, tok, recs)
        print(f"[eval] {a.tag} EXECUTE {a.data.name}: acc={acc:.3f} (n={len(recs)})", flush=True)
        res = {"tag": a.tag, "mode": "execute", "data": a.data.name, "acc": round(acc, 3), "n": len(recs)}
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(res, open(EXP / "runs" / f"eval_{a.tag}_{a.mode}_{a.data.stem}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
