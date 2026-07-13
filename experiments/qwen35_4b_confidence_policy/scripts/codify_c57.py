#!/usr/bin/env python3
"""Append C57 (compute-optimal confidence policy). Idempotent."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
claims = d["claims"]

if not any(c["id"] == "C57" for c in claims):
    claims.append({
        "id": "C57",
        "title": (
            "COMPUTE-OPTIMAL CONFIDENCE POLICY (select + abstain + escalate): the "
            "confidence arc's owed deployable capstone, built on the fixed Qwen3.5-4B "
            "for MBPP. (1) single-token P(True) confidence-select is the best "
            "VERIFIER-FREE selector at every matched-compute k (k=9: 0.762 vs "
            "majority-vote 0.742 vs mean-logprob 0.725 vs greedy 0.701); (2) "
            "max-P(True) abstention yields a clean risk-coverage curve (100% cov "
            "0.757 -> 68% 0.866 -> 43% 0.902); (3) NOVEL: for the low-confidence "
            "(abstained) tasks, escalating to a higher THINK BUDGET beats spending "
            "the same compute on more breadth -- consistently across abstain "
            "fractions (hardest-20%: escalate 0.458 vs matched-breadth 0.304 vs base "
            "0.250; and depth pure-2048 dominates breadth pure-256 on the whole "
            "accuracy-vs-tokens frontier). The C40/C41 confidence signal correctly "
            "aims the C44/C55 serial-compute lever."
        ),
        "status": "Promising",
        "programs": [
            "test_time_reasoning_budget",
            "evidence_conditioned_selection",
            "reliability_and_safety",
        ],
        "summary": (
            "Experiment qwen35_4b_confidence_policy. Fuses the individually-"
            "demonstrated confidence levers (C41 confidence-select > majority; C46 "
            "single-token P(True) > mean-logprob; C40 abstention) into ONE "
            "verifier-free deploy policy and, for the first time, plots them against "
            "sample-more at MATCHED compute -- the frontier C41/C46 explicitly owed "
            "but no experiment built. PART 1 (post-hoc, zero new inference, on the "
            "cached 244-task MBPP pool of qwen35_4b_code_confidence, 9 candidates/"
            "task with full_pass/p_true/mean_logprob/behavior_signature): "
            "confidence-select (argmax p_true) is the best verifier-free selector at "
            "every k (k=9 conf 0.762 > majority 0.742 > logprob 0.725 > greedy "
            "0.701; execution-line 0.840 when a visible test exists; oracle 0.848). "
            "conf+abstain risk-coverage is clean and monotone (max-p_true threshold: "
            "cov 1.00->0.757, 0.81->0.824, 0.68->0.866, 0.43->0.902). PART 2 (new "
            "think-mode generation, 120 MBPP tasks x k=6 at two think budgets, HF "
            "backend): the escalation arm. The 4B self-limits to ~100-140 think "
            "tokens on MBPP so the budget only bites on the hard/forced tasks -- "
            "exactly the abstained set. At MATCHED token-compute with conf-select: "
            "pure-2048 dominates pure-256 on the whole frontier (~750 tok: 0.630 vs "
            "0.618; ~1250 tok: 0.639 vs 0.625), and SELECTIVE escalation of the "
            "abstained tasks beats matched-compute breadth at every abstain fraction "
            "(hardest 20%: base 0.250 -> escalate 0.458 vs breadth 0.304; 30%: 0.361 "
            "-> 0.500 vs 0.417; 40%: 0.354 -> 0.479 vs 0.416; 50%: 0.400 -> 0.467 vs "
            "0.423). Effects are on n=24-60 abstained subsets (directionally "
            "consistent, not yet tightly powered)."
        ),
        "evidence": [
            {"kind": "experiment", "id": "qwen35_4b_confidence_policy"},
            {"kind": "experiment", "id": "qwen35_4b_code_confidence"},
        ],
        "implication": (
            "A deployable, verifier-free test-time policy for the fixed 4B: sample k, "
            "pick argmax single-token P(True), abstain below a max-P(True) threshold, "
            "and ESCALATE the abstained tasks to a higher think budget rather than to "
            "more samples. This is the concrete fusion of the two arcs -- the "
            "metacognition/confidence readout (C40/C41/C46) identifies WHICH tasks "
            "are hard, and the serial-compute lever (C44/C55) is the right way to "
            "spend extra compute on them (depth > breadth on the hard tail). Read a "
            "concentrated single-token logit for selection AND for allocation."
        ),
        "next_tests": [
            "Power up the escalation result: n>=400 MBPP tasks and a bootstrap CI on escalate-minus-breadth per abstain fraction (current n=24-60 is directionally consistent but not tightly powered).",
            "Menagerie arbitration: does the fitted select+abstain+escalate policy generalize as a deployed win on the held-out benchmark (quick/medium, base vs policy at matched compute)?",
            "Escalate DEPTH-LADDER: 256 -> 1024 -> 4096 on the abstained tail -- is there a monotone depth benefit or a knee, and does re-abstention after escalation help?",
            "Does the win hold on a genuinely think-budget-bound task family (harder reasoning where the 4B does NOT self-limit at ~100 tokens), where the escalation lever should be even larger?",
        ],
        "avoid": [
            "Do not use mean-logprob for selection -- single-token P(True) dominates it at every k (C46 replicated here).",
            "Do not spend escalation compute on more breadth for low-confidence tasks: matched-compute depth (higher think budget) beats breadth on the abstained tail.",
            "Do not over-read the escalation point estimates: they are on n=24-60 abstained subsets; the DIRECTION (escalate > breadth > base at all fractions) is the robust claim, the magnitudes need more n.",
        ],
    })

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C57", "present" if any(c["id"] == "C57" for c in claims) else "MISSING", "| total", len(claims))
