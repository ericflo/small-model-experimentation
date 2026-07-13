#!/usr/bin/env python3
"""Append C56 (axis-structured install compression). Idempotent."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
claims = d["claims"]

if not any(c["id"] == "C56" for c in claims):
    claims.append({
        "id": "C56",
        "title": (
            "AXIS-STRUCTURED INSTALL COMPRESSION: at the maxed 8192 menagerie "
            "budget the two weakest axes DISSOCIATE — EXPLORATION is installable "
            "and transfers (gym burrowmaze mean +0.167 at 8192, L6 0.33->0.67; "
            "menagerie medium retain-delta +0.190 > the efficiency install's "
            "+0.146) while composed-rule INDUCTION is NOT (gym glyphgate L4-L6 "
            "stay ~0.0 before and after; trace-SFT even DEGRADES the easy induction "
            "the base could already do, L2 0.93->0.53). No single-4B install flavor "
            "clears the +0.32 conjunction at fair budget; decomposed by axis, the "
            "residual IS the executor-vs-inducer wall (C39/C44/C48), a serial-compute "
            "property of the fixed model, not a data or method gap. Answers C55's "
            "open next-test."
        ),
        "status": "Promising",
        "programs": [
            "agentic_breadth_installation",
            "posttraining_and_adaptation",
            "test_time_reasoning_budget",
        ],
        "summary": (
            "Experiment qwen35_4b_gauntlet_frontier, induction/exploration phase "
            "(the goal's untried weak-axis prescription, greenlit after C55). "
            "MAXED-BUDGET DIAGNOSTIC (gym glyphgate, active induction, greedy@1, "
            "tb=8192): base does single-rule induction (L1-L3 1.00/0.93/0.80) but is "
            "at a hard 0.0 floor on composed-rule induction (L4-L6), and the broad "
            "apex install HURTS induction (L2 0.93->0.47). FOCUSED INSTALL "
            "(data/sft_induction.jsonl: 860 glyphgate+burrowmaze oracle "
            "hypothesize-verify traces weighted to L4-L6 + 900 broad replay; "
            "co-trained from base, emission-seam recipe, adapter induction1). GYM "
            "GATE at 8192: glyphgate MEAN -0.056 (L2 0.93->0.53, L4-L6 ~0.0->~0.0) "
            "— composed induction NOT installable, trace-SFT trains at 1.0 but does "
            "not deploy and degrades easy induction; burrowmaze MEAN +0.167 (L3 "
            "0.87->1.00, L4 0.73->1.00, L5 0.67->0.93, L6 0.33->0.67) — exploration "
            "IS installable with durable lifts at every hard level, base unsaturated. "
            "MENAGERIE TRANSFER (paired base-vs-induction1, n=2/tier, tb=8192): quick "
            "+0.183, medium +0.190 — the exploration gain transfers to the held-out "
            "benchmark and beats the efficiency apex install on medium (+0.190 vs "
            "+0.146; medium carries the multi-turn episodes), despite the combined "
            "install also carrying the net-negative glyphgate traces."
        ),
        "evidence": [{"kind": "experiment", "id": "qwen35_4b_gauntlet_frontier"}],
        "implication": (
            "Install-value compression at fair budget (C55) is AXIS-STRUCTURED, not "
            "uniform: executable procedures (exploration = search + spatial memory) "
            "are installable and transfer to the held-out benchmark, retaining a "
            "real delta at maxed budget; the non-serial inductive leap (composed-rule "
            "induction) is walled and cannot be installed by trace-SFT (trains 1.0, "
            "deploys ~0.0, and even hurts the easy induction the model had). To "
            "improve the fixed 4B at fair budget, target executable procedures, not "
            "induction. The clean exploration-only install (drop the net-negative "
            "glyphgate traces) is the obvious optimization to maximize the medium "
            "retain-delta."
        ),
        "next_tests": [
            "Clean EXPLORATION-ONLY install (burrowmaze + replay, no glyphgate): does dropping the net-negative induction traces push the menagerie medium retain-delta above the combined +0.190?",
            "Tighten the transfer deltas to n>=6 (currently n=2, tight but thin).",
            "Test the other executable-procedure weak axes (program repair loomfix/patchwheel, constrained optimization packhouse/stallwright) for installability + retain at 8192 — do all PROCEDURE axes install while only INDUCTION is walled?",
            "Skin-transfer probe: exploration installs on burrowmaze (SKINNABLE) — does the lift survive fresh pseudo-vocab, i.e. is it the procedure not the surface?",
        ],
        "avoid": [
            "Do not try to install composed-rule induction via oracle-trace SFT: it trains at 1.0 but deploys ~0.0 (C44 serial-compute) and DEGRADES the single-rule induction the base already does.",
            "Do not include glyphgate (induction) traces in an exploration install — they are net-negative on their own axis and drag the combined install.",
            "Do not expect ANY single-4B install flavor to clear +0.32 at the fair 8192 budget: base is ~0.36-0.46, so it would need merged ~0.68-0.78 (beyond the C44 frontier).",
            "Do not read the exploration lift as budget-compensation: base was NOT saturated at 8192 on burrowmaze (L6 0.33), so the install adds capability rather than compensating starvation.",
        ],
    })

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C56", "present" if any(c["id"] == "C56" for c in claims) else "MISSING", "| total", len(claims))
