#!/usr/bin/env python3
"""Figure for the activation-steering result: naming accuracy vs steering coefficient, per depth, for
steer-true / steer-wrong / steer-random vs the no-steer baseline. The flat, overlapping curves are the
finding (decodability != steerability)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def series(nmdict, cond, coefs):
    return [nmdict[f"{cond}@{c:g}"]["naming_acc"] for c in coefs]


def main():
    d = json.load(open(EXP / "runs" / "steer_results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, depth in zip(axes, ["1", "2"]):
        r = d["per_depth"][depth]
        nmd = r["naming"]
        coefs = sorted({float(k.split("@")[1]) for k in nmd if not k.startswith("baseline")})
        base = r["baseline"]
        ax.axhline(base, ls="--", color="#888", lw=1.5, label="baseline (no steer)")
        ax.plot(coefs, series(nmd, "steer_true", coefs), "o-", color="#16a34a", lw=2, label="steer → TRUE first-op")
        ax.plot(coefs, series(nmd, "steer_wrong", coefs), "s-", color="#e34948", lw=2, label="steer → wrong op")
        ax.plot(coefs, series(nmd, "steer_random", coefs), "^-", color="#94a3b8", lw=2, label="steer → random")
        ax.set_title(f"depth {depth} (probe first-op {'0.99' if depth=='1' else '0.42'}, layer {r['layer']})")
        ax.set_xlabel("steering coefficient"); ax.set_ylabel("first-op naming accuracy")
        ax.set_ylim(0, max(0.5, base + 0.2)); ax.grid(alpha=0.25)
        if depth == "1":
            ax.legend(fontsize=8)
    fig.suptitle("Adding the decodable 'first-op' direction does NOT change behavior — decodability ≠ steerability",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "steering.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/steering.png")


if __name__ == "__main__":
    main()
