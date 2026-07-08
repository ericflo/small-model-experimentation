#!/usr/bin/env python3
"""General enumerate-and-verify CoT: infer a hidden affine rule by trying each candidate multiplier a in {1,3,7,9},
deriving b from example 1, verifying on example 2, keeping the a that fits -- then apply. A GENUINELY GENERAL
induction procedure (works for any a). Train on families {a=1,3,9}, hold out a=7: if reasoning-SFT transfers to
a=7, general induction-via-reasoning is installed; if not, it learned only the taught families."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_gen as EG
EXP = Path(__file__).resolve().parents[1]
CANDS = [1, 3, 7, 9]


def cot(ep):
    """Return the enumerate-verify CoT, or None if example1+example2 do not UNIQUELY pin a among the candidates."""
    order = ep["order"]; pos = {d: i for i, d in enumerate(order)}
    exs = [(pos[x], pos[y]) for x, y in ep["examples"]]
    (x1, y1), (x2, y2) = ep["examples"][0], ep["examples"][1]
    p1, q1, p2, q2 = pos[x1], pos[y1], pos[x2], pos[y2]
    # a candidate a fits iff b=(q1-a*p1)%10 maps ALL examples; require exactly one such a AND that ex2 alone
    # already selects it (so the shown 2-step derivation is valid)
    fit_all = [a for a in CANDS if all(((a * p + (q1 - a * p1)) % 10) == q for p, q in exs)]
    fit_ex2 = [a for a in CANDS if ((a * p2 + (q1 - a * p1)) % 10) == q2]
    if len(fit_all) != 1 or len(fit_ex2) != 1 or fit_all[0] != fit_ex2[0]:
        return None
    a = fit_all[0]; b = (q1 - a * p1) % 10
    lines = [f"The rule maps each digit's position pos to (a*pos + b) mod 10 for some a in {{1,3,7,9}}. "
             f"Use example 1 ({x1}->{y1}: pos {p1}->{q1}) to get b for each a, then verify on example 2 "
             f"({x2}->{y2}: pos {p2}->{q2})."]
    for c in CANDS:
        bc = (q1 - c * p1) % 10; chk = (c * p2 + bc) % 10
        lines.append(f"a={c}: b=({q1}-{c}*{p1}) mod 10={bc}; check {c}*{p2}+{bc}={c*p2+bc}, mod 10={chk} vs {q2} "
                     f"-> {'MATCH' if chk == q2 else 'no'}.")
    q = ep["query"]; p = pos[q]; r = (a * p + b) % 10; ans = order[r]
    assert ans == ep["answer"]
    lines.append(f"Only a={a} fits, with b={b}. Apply to {q} (position {p}): ({a}*{p}+{b}) mod 10={r}, the digit "
                 f"at position {r} is {ans}.\nAnswer: {ans}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per-fam", type=int, default=1400); a = ap.parse_args()
    train_fams = ["a1", "a3", "a9"]
    with open(EXP / "data" / "train_general.jsonl", "w") as f:
        for fam in train_fams:
            made = seed = 0
            while made < a.n_per_fam and seed < a.n_per_fam * 6:
                ep = EG.gen_episode(fam, hash((fam, seed)) % (2**31)); seed += 1
                t = cot(ep)
                if t is None: continue
                f.write(json.dumps({"prompt": EG.render(ep), "target": t, "answer": ep["answer"], "fam": fam}) + "\n")
                made += 1
    # held-out family (a7) + in-family held-out
    for fam, tag, rng0 in [("a7", "heldfam_a7", 8_000_000), ("a1", "infam_a1", 8_100_000),
                           ("a3", "infam_a3", 8_200_000), ("a9", "infam_a9", 8_300_000)]:
        with open(EXP / "data" / f"gen_{tag}.jsonl", "w") as f:
            for s in range(rng0, rng0 + 200):
                ep = EG.gen_episode(fam, s)
                f.write(json.dumps({"prompt": EG.render(ep), "answer": ep["answer"], "family": fam, "seed": s}) + "\n")
    print("wrote train_general.jsonl (3 fams x", a.n_per_fam, ") + heldfam_a7 + in-family sets")
    print("--- sample general CoT ---"); print(cot(EG.gen_episode("a3", 5))[:600])
