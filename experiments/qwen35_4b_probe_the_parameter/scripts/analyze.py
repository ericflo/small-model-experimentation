#!/usr/bin/env python3
"""Is the parameter latent? Combines (A) decodability -- op-TYPE is model-latent (probe > external I/O) but the
PARAMETER is surface-readable (external I/O matches the probe), and (B) deployability -- on param-first-op tasks
the full op deploys (oracle_full >> oracle_type) but the model probe barely delivers and the CHEAP surface
pipeline delivers as well or better (surface_full >= probe_full). Emits verdict + figure."""
import json
from math import comb
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
dec = json.load(open(EXP / "runs" / "decode_results.json"))
full = json.load(open(EXP / "runs" / "full_results.json")); K = full["K"]; T = full["per_task"]
def cov(c, n, k): k = min(k, n); return 0.0 if c == 0 else (1.0 if n-c < k else 1-comb(n-c, k)/comb(n, k))
ARMS = ["nohint", "oracle_type", "oracle_full", "probe_full", "surface_full", "wrong_param"]

print("=== (A) DECODABILITY: model probe vs external-I/O surface baseline ===")
for d in ("2", "3"):
    x = dec[d]
    print(f"depth {d}: TYPE probe {x['probe_type']} vs surface {x['ext_io_type']} | "
          f"CONCRETE probe {x['probe_concrete']} vs surface {x['ext_io_concrete']} | "
          f"PARAM|type probe {x['probe_param|type']} vs surface {x['ext_param|type']} (chance {x['param|type_chance']})")

def rate(rows, arm, metric):
    n = len(rows)
    if not n: return float("nan")
    if metric == "greedy": return round(sum(r[f"{arm}_greedy"] for r in rows)/n, 3)
    return round(sum(cov(r[f"{arm}_ncorrect"], K, K) for r in rows)/n, 3)

print("\n=== (B) DEPLOYABILITY by is_param (greedy@1 | cov@{}) ===".format(K))
out = {"decode": dec, "deploy": {}, "probe_conc_acc": full["probe_conc_acc"], "surf_conc_acc": full["surf_conc_acc"]}
for grp, sel in [("param-first-op", lambda r: r["is_param"]), ("non-param", lambda r: not r["is_param"])]:
    rows = [r for r in T if sel(r)]
    print(f"--- {grp} (n={len(rows)}) ---")
    out["deploy"][grp] = {"n": len(rows)}
    for a in ARMS:
        g, c = rate(rows, a, "greedy"), rate(rows, a, "cov")
        out["deploy"][grp][a] = {"greedy": g, "cov": c}
        print(f"  {a:12} greedy {g:.3f} | cov {c:.3f}")

# two-term check: probe_full ~ oracle_full on conc-correct, ~ no-hint on conc-incorrect (param tasks)
pt = [r for r in T if r["is_param"]]
cc = [r for r in pt if r["conc_correct"]]; ci = [r for r in pt if not r["conc_correct"]]
out["two_term"] = {"conc_acc_paramtasks": round(len(cc)/max(1, len(pt)), 3),
                   "probe_on_concCorrect": rate(cc, "probe_full", "greedy"), "oracle_on_concCorrect": rate(cc, "oracle_full", "greedy"),
                   "probe_on_concWrong": rate(ci, "probe_full", "greedy"), "nohint_on_concWrong": rate(ci, "nohint", "greedy")}
print(f"\ntwo-term: probe_full on conc-CORRECT {out['two_term']['probe_on_concCorrect']} (oracle {out['two_term']['oracle_on_concCorrect']}) | "
      f"on conc-WRONG {out['two_term']['probe_on_concWrong']} (nohint {out['two_term']['nohint_on_concWrong']})")

p = out["deploy"]["param-first-op"]
out["verdict"] = (
    f"Is the parameter latent? NO -- it is SURFACE-READABLE, not model-computed. The op-TYPE is model-latent "
    f"(depth-2 probe {dec['2']['probe_type']} > external-I/O {dec['2']['ext_io_type']}), but the PARAMETER given type is "
    f"decoded as well by a trivial I/O classifier as by the model residual (probe {dec['2']['probe_param|type']} vs "
    f"surface {dec['2']['ext_param|type']}, chance {dec['2']['param|type_chance']}). DEPLOYABILITY on param-first-op tasks: "
    f"the param IS the deployable bottleneck (oracle_full {p['oracle_full']['greedy']} >> oracle_type {p['oracle_type']['greedy']}), "
    f"but the model probe barely delivers (probe_full {p['probe_full']['greedy']}) and the CHEAP SURFACE pipeline delivers "
    f"as well or better (surface_full {p['surface_full']['greedy']}) -- you do not need the model for the param. wrong_param "
    f"{p['wrong_param']['greedy']} (content-causal). Sharp localization of C30: the forward pass COMPUTES the op-type "
    f"(latent, elicitable) but only READS the parameter off surface I/O (no privileged model knowledge to elicit).")
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print("\n" + out["verdict"])

# figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# (A) decodability depth-2: probe vs surface for type, concrete, param|type
cats = ["op-TYPE", "CONCRETE (op+param)", "PARAM | type"]
pr = [dec["2"]["probe_type"], dec["2"]["probe_concrete"], dec["2"]["probe_param|type"]]
su = [dec["2"]["ext_io_type"], dec["2"]["ext_io_concrete"], dec["2"]["ext_param|type"]]
x = range(len(cats))
ax1.bar([i-0.2 for i in x], pr, 0.38, label="model probe (residual)", color="#2563eb")
ax1.bar([i+0.2 for i in x], su, 0.38, label="external I/O classifier (no 4B)", color="#f59e0b")
ax1.axhline(dec["2"]["param|type_chance"], ls=":", color="#888", label="param|type chance")
ax1.set_xticks(list(x)); ax1.set_xticklabels(cats, fontsize=9); ax1.set_ylabel("decode accuracy (depth-2, fsig-disjoint)")
ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")
ax1.set_title("Decodability: op-TYPE is model-latent (probe>surface);\nPARAMETER is surface-readable (surface>=probe)")
# (B) deployability on param tasks
cols = {"nohint": "#94a3b8", "oracle_type": "#60a5fa", "oracle_full": "#16a34a", "probe_full": "#a855f7", "surface_full": "#f59e0b", "wrong_param": "#ef4444"}
g = [p[a]["greedy"] for a in ARMS]; c = [p[a]["cov"] for a in ARMS]
x2 = range(len(ARMS))
ax2.bar([i-0.2 for i in x2], c, 0.38, color=[cols[a] for a in ARMS], alpha=0.5, label=f"coverage@{K}")
ax2.bar([i+0.2 for i in x2], g, 0.38, color=[cols[a] for a in ARMS], label="greedy@1")
ax2.set_xticks(list(x2)); ax2.set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=8)
ax2.set_ylabel("solve rate on PARAM-first-op tasks (no-think)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.25, axis="y")
ax2.set_title("Deployability (param tasks): full op deploys (oracle_full>>oracle_type),\nbut cheap surface >= model probe")
fig.suptitle("Is the parameter latent? The op-TYPE is model-computed (elicitable); the PARAMETER is just read off surface I/O", fontsize=11, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "probe_the_parameter.png", dpi=130, bbox_inches="tight")
print("wrote analysis/probe_the_parameter.png")
