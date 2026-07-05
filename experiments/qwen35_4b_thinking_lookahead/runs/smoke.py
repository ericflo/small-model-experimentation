import sys, json, time
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
import gen_lib as GL, decompose as D, think_rank as TR
tasks=[json.loads(l) for l in open("data/eval_frozen_d3.jsonl")][:10]
probe=GL.Probe(); print("loaded", flush=True)
for budget in (0, 512, 1024):
    t0=time.time(); hit=0
    for t in tasks:
        states,target=TR.states_at_step(t,1)
        sc,think=TR.think_then_rank(probe, D.propose_prompt(states,target), budget)
        if max(range(len(sc)), key=lambda i:sc[i])==TR.gt_ops(t)[0]: hit+=1
    print(f"[budget={budget}] STEP-1 top1 {hit/len(tasks):.3f} ({hit}/{len(tasks)}) | {time.time()-t0:.0f}s", flush=True)
print("SMOKEDONE", flush=True)
