#!/usr/bin/env python3
"""Token vs compute vs CONTENT: decompose the C44 induction wall on ONE substrate.
Four ways to spend test-time compute before the forced-Answer read, on held-out
affine induction:
  (0) forced single pass (base);
  (L) LATENT recurrence -- N hidden-state feedback passes, no tokens (C59);
  (F) FILLER tokens -- append N content-free '.' tokens then force Answer (compute
      via real token positions but no reasoning content);
  (C) real CoT -- let the model GENERATE freely, then parse Answer (the C44 ceiling).
If only (C) climbs, the wall needs reasoning CONTENT, not mere compute/tokens.
Run under .venv."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "scripts"))
MODEL_ID="Qwen/Qwen3.5-4B"; REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DIGIT_IDS=[15+d for d in range(10)]; ANS=re.compile(r"[Aa]nswer:\s*\**\s*(\d)")


def chat(tok,u): return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True,enable_thinking=False)


@torch.no_grad()
def forced(model,tok,eps,filler=0,latent=0,bs=32):
    emb_layer=model.get_input_embeddings(); ok=0
    dot = tok(".",add_special_tokens=False).input_ids
    for s in range(0,len(eps),bs):
        sub=eps[s:s+bs]
        pr=[chat(tok,e["prompt"])+("."*filler if filler else "")+"Answer: " for e in sub]
        enc=tok(pr,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        emb=emb_layer(enc.input_ids); mask=enc.attention_mask
        ref=emb.norm(dim=-1,keepdim=True).mean()
        for _ in range(latent):
            out=model(inputs_embeds=emb,attention_mask=mask,output_hidden_states=True)
            h=out.hidden_states[-1][:,-1,:]; h=h/h.norm(dim=-1,keepdim=True).clamp_min(1e-6)*ref
            emb=torch.cat([emb,h[:,None,:]],1); mask=torch.cat([mask,mask.new_ones(mask.size(0),1)],1)
        lg=model(inputs_embeds=emb,attention_mask=mask).logits[:,-1,DIGIT_IDS]
        for e,d in zip(sub,lg.argmax(-1).tolist()): ok+=int(str(d)==e["answer"])
    return ok/len(eps)


@torch.no_grad()
def real_cot(model,tok,eps,bs=24,max_new=768):
    ok=0
    for s in range(0,len(eps),bs):
        sub=eps[s:s+bs]
        enc=tok([chat(tok,e["prompt"]) for e in sub],return_tensors="pt",padding=True,add_special_tokens=False).to("cuda")
        g=model.generate(**enc,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id)
        for e,row in zip(sub,g):
            txt=tok.decode(row[enc.input_ids.shape[1]:],skip_special_tokens=True)
            m=ANS.findall(txt); ok+=int((m[-1] if m else "")==e["answer"])
    return ok/len(eps)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=150)
    ap.add_argument("--data",default=str(EXP/"data"/"test_affine.jsonl")); a=ap.parse_args()
    eps=[json.loads(l) for l in open(a.data)][:a.n]
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=True); tok.padding_side="left"
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=True,dtype=torch.bfloat16,device_map="cuda").eval()
    print(f"[decomp] {len(eps)} held-out affine induction episodes")
    r={}
    r["forced_N0"]=forced(model,tok,eps)
    r["latent_N8"]=forced(model,tok,eps,latent=8)
    r["filler_N8"]=forced(model,tok,eps,filler=8)
    r["filler_N32"]=forced(model,tok,eps,filler=32)
    r["real_cot"]=real_cot(model,tok,eps)
    for k,v in r.items(): print(f"  {k:>12}: {v:.4f}")
    (EXP/"runs").mkdir(exist_ok=True); (EXP/"runs"/"compute_decomp.json").write_text(json.dumps(r,indent=2)+"\n")
    print("wrote runs/compute_decomp.json")


if __name__=="__main__": main()
