#!/usr/bin/env python3
"""Update C56 with the definitive clean exploration-only ceiling (n=6). Idempotent."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
c56 = next(c for c in d["claims"] if c["id"] == "C56")

marker = "CLEAN EXPLORATION-ONLY"
if marker not in c56["summary"]:
    c56["summary"] += (
        "  " + marker + " (burrowmaze + replay, no glyphgate; the combined install's "
        "+0.190 was a lower bound dragged by the net-negative induction traces): gym "
        "burrowmaze MEAN +0.200 at 8192 (L5 0.67->1.00, L6 0.33->0.80), and the "
        "MENAGERIE medium retain-delta rises to +0.261 +- 0.048 (n=6; merged mean 0.600 "
        "-- tight at 0.575-0.616 -- vs base mean 0.339), quick +0.199 (n=2, sd ~0.001). "
        "The merged medium ABSOLUTE 0.60 is the best measured (base ~0.34). This is the "
        "definitive exploration ceiling: the strongest single-4B install, yet still "
        "below +0.32 on medium and far below on quick, so the conjunction stays "
        "unreachable -- exploration lifts medium (episodes) but cannot lift atoms-only "
        "quick, and no single install does both (tier-Pareto C54)."
    )

c56["implication"] = (
    "Install-value compression at fair budget (C55) is AXIS-STRUCTURED: executable "
    "procedures (exploration) install durably and transfer -- the clean "
    "exploration-only install is the strongest single-4B install measured (menagerie "
    "medium +0.261 +- 0.048 n=6, merged medium absolute 0.60 vs base 0.34) -- while the "
    "non-serial inductive leap (composed induction) is walled and un-installable by "
    "trace-SFT (and even hurts). No install flavor clears the +0.32 conjunction at fair "
    "budget: medium tops out ~+0.26 (exploration) and quick ~+0.20 (efficiency; "
    "exploration cannot lift atoms-only quick), and the two tiers need different levers "
    "with no single model doing both (tier-Pareto C54). The gauntlet's positive core "
    "result is that EXPLORATION is a genuinely installable, budget-robust capability; "
    "its negative core result is that the +0.32-on-both target was a budget-starvation "
    "artifact (C55) whose residual is the serial-compute induction wall (C39/C44/C48)."
)

# The exploration-only next-test is now done; replace it with the forward-looking ones.
c56["next_tests"] = [
    "Do the OTHER executable-procedure weak axes install like exploration? Repeat the isolate-and-measure recipe for program repair (loomfix/patchwheel) and constrained optimization (packhouse/stallwright) at 8192.",
    "Skin-transfer probe: burrowmaze is SKINNABLE -- does the exploration lift survive fresh pseudo-vocab (procedure, not surface)?",
    "Tier-router deployment: exploration-merged for medium/episode workloads + efficiency-merged for quick/atom workloads -- the only remaining path to strong deltas on BOTH tiers, since no single model does both.",
    "Push the exploration install harder (more burrowmaze hard-level traces, expert-iteration on burrowmaze successes) to test whether medium can be pushed decisively past +0.32 while quick is served by a different model.",
]

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C56 updated with exploration-only ceiling.")
