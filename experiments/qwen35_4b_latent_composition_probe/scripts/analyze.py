#!/usr/bin/env python3
"""Figure + native-chart data for the latent-composition probe: representation (linear probe) vs expression
(behavior) by depth, and the per-layer probe profile."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def main():
    d = json.load(open(EXP / "runs" / "probe_results.json"))
    pd = {int(k): v for k, v in d["per_depth"].items()}
    depths = sorted(pd)

    probe = [pd[x]["probe_first_op_best"] for x in depths]
    shuf = [pd[x]["probe_first_op_shuffled"] for x in depths]
    name = [pd[x]["behavioral_name_first_op"] for x in depths]
    ident = [pd[x]["behavioral_ident_pass1"] for x in depths]
    chance = [pd[x]["chance_first_op"] for x in depths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = range(len(depths))
    w = 0.2
    ax1.bar([i - 1.5 * w for i in x], probe, w, label="linear probe (representation)", color="#4a3aa7")
    ax1.bar([i - 0.5 * w for i in x], name, w, label="model names 1st op (expression)", color="#eda100")
    ax1.bar([i + 0.5 * w for i in x], ident, w, label="model generates it (ident@1)", color="#1baf7a")
    ax1.bar([i + 1.5 * w for i in x], shuf, w, label="shuffled-label floor", color="#c3c2b7")
    ax1.plot([i for i in x], chance, "k_", ms=22, mew=2, label="chance")
    ax1.set_xticks(list(x)); ax1.set_xticklabels([f"depth {dd}" for dd in depths])
    ax1.set_ylabel("first-op accuracy"); ax1.set_ylim(0, 1.05)
    ax1.set_title("Representation ≫ expression at shallow depth; both thin at the wall")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")

    colors = {1: "#2a78d6", 2: "#1baf7a", 3: "#e34948"}
    for dd in depths:
        prof = pd[dd]["layer_profile"]
        ax2.plot(range(len(prof)), prof, "-", color=colors.get(dd, "#333"), lw=2, label=f"depth {dd}")
    ax2.axhline(pd[depths[-1]]["chance_first_op"], ls=":", color="gray", lw=1, label="chance")
    ax2.set_xlabel("layer (0 = embedding)"); ax2.set_ylabel("first-op probe accuracy")
    ax2.set_ylim(0, 1.05); ax2.set_title("Where the first op is computed (layer profile)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

    fig.suptitle("Inside the generation wall: is the composition latent or absent?", y=1.02, fontsize=13)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "latent_probe.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/latent_probe.png")

    # native chart data (for experiment_viz.json)
    print("\n--- native chart values ---")
    print("categories:", [f"depth {dd}" for dd in depths])
    print("probe:", probe, "| name:", name, "| ident:", ident, "| shuffled:", shuf)
    for dd in depths:
        print(f"depth {dd} layer_profile:", [round(v, 3) for v in pd[dd]["layer_profile"]])


if __name__ == "__main__":
    main()
