#!/usr/bin/env python3
"""Adversarial verification of the block-12:16 looping effect (acc 0.085 -> 0.240 at n=200).

The headline is extraordinary -- it matches C59's real-chain-of-thought number (0.235) with ZERO
generated tokens -- so it gets attacked before it gets believed. Four ways it could be fake, each with
its own arm here:

1. LABEL-PRIOR EXPLOITATION. The forced read is an argmax over 10 digit tokens and the held-out
   answers are NOT uniform: the most common class is 0.18 of the set. A model degraded into predicting
   one digit therefore "beats" a 0.085 baseline for free. Killed by BALANCED ACCURACY (mean per-class
   recall), which a constant predictor scores 0.10 on by construction, plus the predicted-digit
   distribution and its top-1 share.

2. POSITIONAL COINCIDENCE. If any 4-layer loop anywhere helps, the story is "extra compute", not
   "these layers". Every 4-layer block on the hybrid period is swept: 4:8 ... 28:32.

3. IT IS NOT THE SPECIFIC LAYERS. Insert a copy of a DIFFERENT 4-layer block at the same stack
   position (run 12:16 once, then a copy of 4:8) -- same added depth and parameter count, different
   computation. If that helps equally, the effect is generic depth, not the looped block's function.

4. SAMPLE SIZE. n=200 has SE ~0.02 at these rates; the winner is re-measured on all 400 held-out
   episodes, and on the disjoint second half alone as an out-of-sample replication.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import recur
from recur import DIGIT_IDS, LoopedLayers, chat

EXP = Path(__file__).resolve().parents[1]


class InsertBlock(LoopedLayers):
    """Insert a copy of layers[c:d] after position b instead of re-running layers[a:b].

    Control 3: identical added depth and parameters, different computation.
    """

    def __init__(self, model, a, b, c, d):
        super().__init__(model, a, b, 2, 1.0)
        self.c, self.d = c, d

    def __enter__(self):
        layers = list(self.orig_layers)
        new_layers = layers[:self.b] + layers[self.c:self.d] + layers[self.b:]
        self.inner.layers = nn.ModuleList(new_layers)
        if self.orig_types:
            t = self.orig_types
            self.cfg.layer_types = t[:self.b] + t[self.c:self.d] + t[self.b:]
        self.cfg.num_hidden_layers = len(new_layers)
        return self


@torch.no_grad()
def predict(model, tok, episodes, ctx_factory, bs=16):
    """Return (predictions, golds) under a given stack modification."""
    preds, golds = [], []
    ctx = ctx_factory()
    with ctx:
        for s in range(0, len(episodes), bs):
            sub = episodes[s:s + bs]
            prompts = [chat(tok, e["prompt"]) + "Answer: " for e in sub]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            logits = model(**enc, use_cache=False).logits[:, -1, DIGIT_IDS]
            preds += [str(d) for d in logits.argmax(-1).tolist()]
            golds += [e["answer"] for e in sub]
    return preds, golds


def metrics(preds, golds):
    acc = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
    # Balanced accuracy = mean per-class recall. A constant predictor scores exactly 1/10 here, so any
    # value materially above 0.10 cannot be produced by exploiting the skewed answer prior.
    recalls = []
    for cls in {*golds}:
        idx = [i for i, g in enumerate(golds) if g == cls]
        recalls.append(sum(preds[i] == cls for i in idx) / len(idx))
    dist = Counter(preds)
    return {"acc": round(acc, 4), "balanced_acc": round(sum(recalls) / len(recalls), 4),
            "n": len(golds), "distinct_preds": len(dist),
            "top1_pred": dist.most_common(1)[0][0], "top1_share": round(dist.most_common(1)[0][1] / len(preds), 3),
            "pred_dist": dict(sorted(dist.items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(EXP / "data" / "heldout_shift.jsonl"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=str(EXP / "reports" / "verify_results.json"))
    args = ap.parse_args()

    episodes = [json.loads(l) for l in open(args.data)][: args.n]
    gold_dist = Counter(e["answer"] for e in episodes)
    majority = max(gold_dist.values()) / len(episodes)
    print(f"n={len(episodes)} | gold distribution {dict(sorted(gold_dist.items()))}", flush=True)
    print(f"MAJORITY-CLASS RATE (what a constant predictor scores) = {majority:.3f}", flush=True)

    tok = AutoTokenizer.from_pretrained(recur.MODEL_ID, revision=recur.REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(recur.MODEL_ID, revision=recur.REV,
                                                 dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager")  # footgun-ok: the PUBLISHED forced-read numbers (0.085/0.245/0.278) were measured under eager; sdpa shifts them ~0.005-0.010 via bf16 argmax flips, so switching would break comparability with committed results. New work should use sdpa.
    model.eval()

    arms = [("baseline", lambda: LoopedLayers(model, 0, 0, 1))]
    for a in (4, 8, 12, 16, 20, 24, 28):                       # control 2: every 4-layer block
        arms.append((f"loop_{a}:{a+4}_k2", lambda a=a: LoopedLayers(model, a, a + 4, 2, 1.0)))
    arms.append(("insert_4:8_after_16", lambda: InsertBlock(model, 12, 16, 4, 8)))    # control 3
    arms.append(("insert_20:24_after_16", lambda: InsertBlock(model, 12, 16, 20, 24)))

    results = {"n": len(episodes), "majority_class_rate": round(majority, 4),
               "gold_dist": dict(sorted(gold_dist.items())),
               "c59_reference": recur.C59_REFERENCE, "arms": {}}
    for name, factory in arms:
        preds, golds = predict(model, tok, episodes, factory)
        m = metrics(preds, golds)
        results["arms"][name] = m
        flag = ""
        if m["balanced_acc"] <= 0.11:
            flag = "  <-- at/below constant-predictor balanced accuracy"
        print(f"  {name:24s} acc {m['acc']:.3f}  balanced {m['balanced_acc']:.3f}  "
              f"distinct {m['distinct_preds']:2d}  top1 '{m['top1_pred']}' {m['top1_share']:.2f}{flag}",
              flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))

    # Out-of-sample replication of the winner on the disjoint second half.
    win = max((k for k in results["arms"] if k.startswith("loop_")),
              key=lambda k: results["arms"][k]["balanced_acc"])
    half = len(episodes) // 2
    a = int(win.split("_")[1].split(":")[0])
    for tag, sub in (("first_half", episodes[:half]), ("second_half", episodes[half:])):
        for nm, fac in (("baseline", lambda: LoopedLayers(model, 0, 0, 1)),
                        (win, lambda: LoopedLayers(model, a, a + 4, 2, 1.0))):
            p, g = predict(model, tok, sub, fac)
            results.setdefault("split_half", {})[f"{tag}/{nm}"] = metrics(p, g)
            print(f"  [{tag}] {nm:20s} acc {results['split_half'][f'{tag}/{nm}']['acc']:.3f} "
                  f"balanced {results['split_half'][f'{tag}/{nm}']['balanced_acc']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
