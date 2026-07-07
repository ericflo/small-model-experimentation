#!/usr/bin/env python3
"""Retrieval vs induction, execution-SAFE single-value substrate. 'Advance k steps in a cyclic order', query an
UNSEEN digit (generalization). Crux: FAMILIAR order (natural 0..9) vs NOVEL order (stated random) x {few-shot
(INDUCE k) vs rule-stated (EXECUTE given k)}. If familiar-fewshot >> novel-fewshot while the rule-stated controls
are comparable, the model RETRIEVES a familiar rule but cannot INDUCE/apply a novel (even fully-stated) one =>
ICL = retrieval, not induction. Robust extraction (model reasons + tends to code-mode)."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import succ_family as SF  # noqa: E402


def extract(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    m = re.findall(r"[Aa]nswer:\s*\**\s*(\d)", text)
    if m: return m[-1]
    ds = re.findall(r"\d", text)               # fallback: last standalone digit
    return ds[-1] if ds else ""


def run(kind, mode, n, seed, p, GL):
    # no-think: the model reasons in PROSE (not code-mode, which think-mode triggers on 'apply-a-rule' tasks)
    tasks = [SF.gen_task(kind, seed * 100000 + i) for i in range(n)]
    prompts = [p.prompt(SF.render(t, rule_stated=(mode == "stated")), enable_thinking=False) for t in tasks]
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=300, batch_size=48)
    ok = fin = codemode = 0
    for t, pr, g in zip(tasks, prompts, gg):
        txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=True)
        fin += int("</think>" in txt); codemode += int("```" in txt or "def " in txt)
        ok += int(extract(txt) == t["answer"])
    return {"kind": kind, "mode": mode, "n": n, "acc": round(ok / n, 3),
            "finished": round(fin / n, 3), "codemode_rate": round(codemode / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--conditions", nargs="+", default=["familiar_fewshot", "novel_fewshot", "familiar_stated", "novel_stated"])
    args = ap.parse_args()
    import gen_lib as GL
    p = GL.Probe()
    out = {}
    for cond in args.conditions:
        kind, m = cond.split("_"); mode = "stated" if m == "stated" else "fewshot"
        r = run(kind, mode, args.n, args.seed, p, GL)
        out[cond] = r
        print(f"[succ] {cond}: acc={r['acc']:.2f} (finished {r['finished']:.2f}, codemode {r['codemode_rate']:.2f})", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(out, open(EXP / "runs" / "succ_crux.json", "w"), indent=1)
    print("[succ] wrote runs/succ_crux.json", flush=True)


if __name__ == "__main__":
    main()
