#!/usr/bin/env python3
"""Is the C44 serial-compute wall TOKEN-bound or COMPUTE-bound? C44: the 4B
induces held-out rules at ~0.01 in ONE forward pass but ~1.0 via chain-of-thought
GENERATION. Every serial-compute claim is argued in TOKEN space. This probes
compute WITHOUT tokens: append the model's own last-layer hidden state (final
position) back as the next input embedding and run again -- N extra forward
passes over a 'latent thought', emitting NO tokens -- then read the forced-Answer
digit. If accuracy climbs with N, the wall is compute-DEPTH (addressable by latent
recurrence); if flat at ~0.01, it is genuinely token-bound. Held-out affine
induction (out-of-family). Run under .venv."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import episode_gen as EG  # noqa
MODEL_ID = "Qwen/Qwen3.5-4B"; REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DIGIT_IDS = [15 + d for d in range(10)]


def chat(tok, u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def induce_recur(model, tok, recs, N, norm_match, bs=32):
    emb_layer = model.get_input_embeddings()
    correct = 0
    for s in range(0, len(recs), bs):
        sub = recs[s:s+bs]
        prompts = [chat(tok, r["ep"]["prompt"]) + "Answer: " for r in sub]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        emb = emb_layer(enc.input_ids); mask = enc.attention_mask
        # reference input-embedding norm (for optional scale matching)
        ref = emb.norm(dim=-1, keepdim=True).mean()
        for _ in range(N):
            out = model(inputs_embeds=emb, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]                       # last-layer, final position
            if norm_match:
                h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6) * ref
            emb = torch.cat([emb, h[:, None, :]], dim=1)
            mask = torch.cat([mask, mask.new_ones(mask.size(0), 1)], dim=1)
        logits = model(inputs_embeds=emb, attention_mask=mask).logits[:, -1, DIGIT_IDS]
        pred = logits.argmax(-1).tolist()
        for r, d in zip(sub, pred):
            correct += int(str(d) == r["ep"]["answer"])
    return correct / len(recs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=str(EXP / "data" / "test_affine.jsonl"))
    ap.add_argument("--Ns", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    a = ap.parse_args()
    eps = [json.loads(l) for l in open(a.data)][:a.n]
    recs = [{"ep": e} for e in eps]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, trust_remote_code=True,
                                                 dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"[latrec] {len(recs)} held-out affine induction episodes", flush=True)
    print(f"{'N':>4} {'raw-hidden':>11} {'norm-matched':>13}")
    out = {}
    for N in a.Ns:
        raw = induce_recur(model, tok, recs, N, norm_match=False)
        nm = induce_recur(model, tok, recs, N, norm_match=True) if N > 0 else raw
        out[N] = {"raw": round(raw, 4), "norm": round(nm, 4)}
        print(f"{N:>4} {raw:>11.4f} {nm:>13.4f}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "latent_recurrence.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote runs/latent_recurrence.json")


if __name__ == "__main__":
    main()
