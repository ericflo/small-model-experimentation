#!/usr/bin/env python3
"""Fix C54's stale title and append C55 (budget-compression law). Idempotent."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
claims = d["claims"]

# --- Fix C54: its title still says "+0.345 DECISIVELY clear MEDIUM"; that was
# corrected to +0.305 (favorable noise at n=3; pooled n=22). Make the title honest.
c54 = next(c for c in claims if c["id"] == "C54")
c54["title"] = (
    "TIER-PARETO FRONTIER (corrected): novel serial-compute mechanisms "
    "(length-penalized compression advantage + skin-shuffle) lift the MEDIUM "
    "menagerie tier to the +0.32 line but do NOT decisively clear it — the "
    "early +0.345 read was favorable noise (n=3); pooled n=22 = +0.305 ± 0.010. "
    "No single Qwen3.5-4B model clears quick AND medium together by any method "
    "(training, capacity, data-interpolation, weight-space soup, expert iteration, "
    "tier-router, episode-mastery, or oracle-injection). Refined by C55: at the "
    "maxed 8192 budget the medium delta compresses further as the base catches up."
)

# --- Append C55 if absent.
if not any(c["id"] == "C55" for c in claims):
    claims.append({
        "id": "C55",
        "title": (
            "BUDGET-COMPRESSION LAW: maxing the menagerie think budget (all tiers "
            "→ 8192, uncapped `huge` tier, max_model_len 65536) reveals the "
            "gym-installed advantage was PARTLY compensation for a budget-starved "
            "base. A deployment-time compute-response study first confirms the "
            "medium wall is SERIAL-COMPUTE (merged absolute medium score rises "
            "monotonically 0.337→0.436→0.518 at think budget 1024/2048/4096); "
            "then at the new canonical 8192 budget BASE leaps (quick 0.11→0.46, "
            "medium 0.13→0.36) and the merged-vs-base DELTA compresses from "
            "+0.33/+0.31 to +0.21/+0.15. The install still yields the best ABSOLUTE "
            "capability yet (merged 0.666 quick / 0.506 medium) but its MARGINAL "
            "value over a fairly-resourced base is ~+0.15–+0.21, not +0.32."
        ),
        "status": "Promising",
        "programs": [
            "agentic_breadth_installation",
            "posttraining_and_adaptation",
            "test_time_reasoning_budget",
        ],
        "summary": (
            "Experiment qwen35_4b_gauntlet_frontier, budget-response phase. Two "
            "moves. (1) COMPUTE-RESPONSE STUDY (bench.py --think-budget, paired "
            "base-vs-merged on medium items at escalating budgets, fresh seeds): "
            "the merged medium ABSOLUTE score rises monotonically with the deployed "
            "think budget — 0.337 (n=2) @1024, 0.436 (n=6) @2048, 0.518 (n=2) @4096 "
            "— directly confirming AT DEPLOYMENT (not merely inferred from training, "
            "as in C54) that the medium wall is a C44 serial-compute limit: the "
            "procedure is in the weights and more tokens execute more of it. The "
            "merged-minus-base DELTA, however, only rises +0.269→+0.309→+0.301 and "
            "PLATEAUS ~+0.31 because base also converts budget into gains. (2) "
            "BENCHMARK REDEFINITION (owner-directed, one-off; applied via a "
            "context-shielded subagent to keep menagerie internals firewalled): all "
            "named tiers → think_budget 8192, a fully-uncapped `huge` tier (65536 == "
            "max_model_len), and max_model_len 16384→65536 (verified to init on the "
            "RTX 4090 at ~22.5 GB, no OOM); tiers stay ordered by coverage/wall-clock "
            "(quick<medium<slow<deep<huge). At the NEW canonical 8192 budget (paired "
            "base-vs-merged, n=2/tier, tight): quick base 0.455 / merged 0.666 / "
            "delta +0.211; medium base 0.360 / merged 0.506 / delta +0.146. Base "
            "leapt from the old budget-starved ~0.11/~0.13, so the delta compressed "
            "to ~10× its own n=2 spread below the old +0.31–+0.33. The gym install "
            "was thus partly compensating for base not being allowed to think."
        ),
        "evidence": [{"kind": "experiment", "id": "qwen35_4b_gauntlet_frontier"}],
        "implication": (
            "The +0.32 quick-AND-medium conjunction was defined against a "
            "budget-STARVED base and is the wrong target: give base fair serial "
            "compute and it recovers most of the gap. Report ABSOLUTE capability "
            "(merged 0.666 quick / 0.506 medium at 8192 — the best measured), not "
            "delta-over-a-crippled-base. To beat a fairly-resourced base by a LARGE "
            "margin, install what base CANNOT do even with 8192 tokens — the "
            "induction / hypothesize-verify walls (C43/C44/C48) — not efficiency or "
            "procedure knowledge the base rediscovers once it can think. Old "
            "baselines (0.112/0.146/0.138) are superseded; re-baseline at 8192."
        ),
        "next_tests": [
            "Tighten quick@8192 and medium@8192 to n>=6 (currently n=2; sd tight ~0.01-0.02 but thin).",
            "Regenerate full baselines at 8192 (seed 31337) for slow/deep; produce a first `huge`-tier baseline (uncapped, ~18 h wall).",
            "Does an install TARGETING the induction/hypothesize-verify walls (what base cannot do even at 8192) retain a large delta at maxed budget, unlike the efficiency/procedure install that compresses?",
            "Sweep merged-vs-base delta at budget 8192->16384->uncapped(`huge`): does it compress to ~0 (base fully catches up) or stabilize at a floor?",
        ],
        "avoid": [
            "Do not compare any post-2026-07-12 menagerie run against the OLD baselines (0.112/0.146/0.138) — they were measured at budget-starved settings and are superseded; re-baseline at 8192.",
            "Do not read the gym install's old-budget +0.32 as its true value — only ~+0.15–+0.21 survives a fairly-resourced base; the rest was budget-starvation compensation.",
            "Do not chase quick-AND-medium >+0.32 at maxed budget: base now scores ~0.36–0.46, so a +0.32 medium delta needs merged ~0.68–0.78 — beyond the 4B execution frontier (C44).",
            "Do not read a shrinking delta as a failed install: the merged ABSOLUTE score is the best measured (0.666/0.506); the delta shrinks because BASE improved, not because merged regressed.",
        ],
    })

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C54 title corrected; C55", "present" if any(c["id"] == "C55" for c in claims) else "MISSING")
print("total claims:", len(claims))
