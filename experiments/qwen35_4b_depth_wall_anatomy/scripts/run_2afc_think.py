#!/usr/bin/env python3
"""P12: thinking-mode 2AFC — does deliberate simulate-and-compare rescue hypothesis discrimination?"""
import json, random, sys, time
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP/"src"))
import gen_tasks as G, decompose_lib as D
from run_probes import steps_of, decoy_pipeline, pipe_str  # reuse

def main():
    allt=[json.loads(l) for l in (EXP/"data"/"grid_tasks.jsonl").read_text().splitlines()]
    CELLS=[(2,0),(2,2),(3,0),(3,2),(4,0),(4,2)]
    tasks=[]
    for (d,k) in CELLS: tasks += [t for t in allt if t["depth"]==d and t["n_destr"]==k][:20]
    import gen_lib as GL
    p=GL.Probe(); rng=random.Random(4242)  # SAME seed as no-think 2AFC -> same decoys/order
    prompts, truth = [], []
    for t in tasks:
        true_s, dec_s = pipe_str(steps_of(t)), pipe_str(decoy_pipeline(t, rng))
        a_is_true = rng.random() < 0.5
        pa, pb = (true_s, dec_s) if a_is_true else (dec_s, true_s)
        ex="\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"][:6])
        user=(f"Examples of `transform`:\n{ex}\n\nWhich pipeline produces exactly this behaviour?\n"
              f"A) {pa}\nB) {pb}\n\nWork through the examples step by step, then answer with the single letter A or B.")
        prompts.append(p.prompt(user, enable_thinking=True)); truth.append(a_is_true)
    t0=time.time()
    gens=p.gen_sequences(prompts, think=True, budget=512, greedy=True, batch_size=24)
    correct=[]
    for pr,g,a_true,t in zip(prompts,gens,truth,tasks):
        txt=p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
        ans=txt.split("</think>")[-1] if "</think>" in txt else txt
        picked_a=None
        for ch in ans:
            if ch in "AB": picked_a=(ch=="A"); break
        correct.append({"task_id":t["task_id"],"depth":t["depth"],"n_destr":t["n_destr"],
                        "correct": (picked_a==a_true) if picked_a is not None else False,
                        "parsed": picked_a is not None})
    (EXP/"data"/"twoafc_think_records.jsonl").write_text("\n".join(json.dumps(r) for r in correct)+"\n")
    from collections import defaultdict
    by=defaultdict(lambda:[0,0])
    for r in correct:
        c=by[(r["depth"],r["n_destr"])]; c[0]+=1; c[1]+=int(r["correct"])
    print(f"=== THINKING 2AFC (greedy, budget 512) [{time.time()-t0:.0f}s] ===")
    for (d,k),(n,s) in sorted(by.items()): print(f"  d{d}k{k}: {s/n:.2f} (n={n})")
    print(f"  overall: {sum(r['correct'] for r in correct)/len(correct):.2f}; parse rate {sum(r['parsed'] for r in correct)/len(correct):.2f}")
main()
