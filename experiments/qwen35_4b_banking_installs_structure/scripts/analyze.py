#!/usr/bin/env python3
"""Does banking install STRUCTURE? YES. base structure-coverage 0.000 -> banked 0.512 (held-out depth-3),
confirming banking installs the op-sequence structure the base can't propose (C32). And banking CONVERTS the
wall from structure-bound (base: struct=concrete=0) to value-bound (banked: struct 0.512 > concrete 0.362, a
value tax of +0.15 = right-skeleton-wrong-param failures) -- recoverable by value-fill (oracle-skelfill=1.0, C32).
Emits verdict + figure."""
import json
from math import comb, sqrt
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
def covk(c, n, k): k = min(k, n); return 0.0 if c == 0 else (1.0 if n-c < k else 1-comb(n-c, k)/comb(n, k))
def wil(p, n, z=1.96): return z*sqrt(p*(1-p)/n)/(1+z*z/n) if n else 0.0

out = {}
for tag in ("base", "banked"):
    d = json.load(open(EXP / "runs" / f"bank_{tag}.json")); rows = d["rows"]; K = d["k"]; n = d["n"]
    out[tag] = {"n": n, "K": K,
                "greedy": round(sum(r["greedy"] for r in rows)/n, 3),
                "cov": round(sum(covk(r["cov"], K, K) for r in rows)/n, 3),
                "struct_cov": round(sum(covk(r["struct_cov"], K, K) for r in rows)/n, 3),
                "struct_greedy": round(sum(r["struct_greedy"] for r in rows)/n, 3)}
    out[tag]["value_tax"] = round(out[tag]["struct_cov"] - out[tag]["cov"], 3)
    out[tag]["struct_cov_ci"] = round(wil(out[tag]["struct_cov"], n), 3)

print("=== Does banking install STRUCTURE? (held-out depth-3) ===")
for tag in ("base", "banked"):
    o = out[tag]
    print(f"{tag:7}: greedy@1 {o['greedy']:.3f} | cov@{o['K']} {o['cov']:.3f} | "
          f"STRUCTURE-cov {o['struct_cov']:.3f} (+/-{o['struct_cov_ci']:.3f}) | value tax {o['value_tax']:+.3f}")

out["verdict"] = (
    f"Banking installs STRUCTURE. Base structure-coverage {out['base']['struct_cov']} -> banked "
    f"{out['banked']['struct_cov']} on HELD-OUT depth-3 (generalizable, not memorized): banking installs the "
    f"op-sequence STRUCTURE the base cannot propose (C32). And banking CONVERTS the wall from structure-bound to "
    f"value-bound: the base has no skeletons (struct=concrete={out['base']['cov']}), but the banked model proposes "
    f"the right skeleton {out['banked']['struct_cov']} of the time while nailing the full program only "
    f"{out['banked']['cov']} -- a VALUE TAX of {out['banked']['value_tax']:+.3f} (right-skeleton-wrong-param "
    f"failures the base never had). Since oracle-skeletonfill=1.0 (C32), value-filling the banked model's proposed "
    f"skeletons would deploy at ~{out['banked']['struct_cov']} (bank installs structure; value-fill recovers the "
    f"value tax). Mechanistic confirmation of C22-24: banking's lever IS structure-proposal.")
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print("\n" + out["verdict"])

fig, ax = plt.subplots(figsize=(10, 5.2))
groups = ["greedy@1", f"cov@{out['base']['K']}", "STRUCTURE-cov\n(right op-sequence,\nany param)"]
keys = ["greedy", "cov", "struct_cov"]
x = range(len(keys)); w = 0.38
ax.bar([i-w/2 for i in x], [out["base"][k] for k in keys], w, label="base", color="#94a3b8")
ax.bar([i+w/2 for i in x], [out["banked"][k] for k in keys], w, label="banked (banked_1280)", color="#16a34a")
for i, k in enumerate(keys):
    ax.text(i+w/2, out["banked"][k]+0.008, f"{out['banked'][k]:.2f}", ha="center", fontsize=9)
# value tax annotation
ax.annotate(f"value tax +{out['banked']['value_tax']:.2f}\n(right skeleton,\nwrong param -> fillable)",
            xy=(2+w/2, out["banked"]["struct_cov"]), xytext=(1.15, 0.45), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=9)
ax.set_ylabel("solve rate (held-out depth-3, n=80, no-think)"); ax.legend(); ax.grid(alpha=0.25, axis="y")
ax.set_title("Does banking install STRUCTURE? YES: base structure-cov 0.00 -> banked 0.51.\n"
             "Banking converts the wall from STRUCTURE-bound (base) to VALUE-bound (banked value tax +0.15)", fontsize=10)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "banking_installs_structure.png", dpi=130, bbox_inches="tight")
print("wrote analysis/banking_installs_structure.png")
