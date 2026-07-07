#!/usr/bin/env python3
"""Generate meta-induction datasets. Train on random SHIFT episodes; hold out disjoint SHIFT episodes (in-family)
and AFFINE episodes (out-of-family). Random orders/params -> nothing to memorize."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_gen as EG

EXP = Path(__file__).resolve().parents[1]


def dump(path, family, seeds):
    with open(path, "w") as f:
        for s in seeds:
            ep = EG.gen_episode(family, s)
            f.write(json.dumps({"prompt": EG.render(ep), "answer": ep["answer"], "family": family,
                                "order": "".join(ep["order"]), "a": ep["a"], "b": ep["b"], "seed": s}) + "\n")
    print("wrote", path.name, len(seeds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-train", type=int, default=4000); a = ap.parse_args()
    (EXP / "data").mkdir(exist_ok=True)
    dump(EXP / "data" / "train_shift.jsonl", "shift", range(a.n_train))                 # train (in-family)
    dump(EXP / "data" / "heldout_shift.jsonl", "shift", range(5_000_000, 5_000_400))    # held-out in-family
    dump(EXP / "data" / "test_affine.jsonl", "affine", range(7_000_000, 7_000_400))     # out-of-family
