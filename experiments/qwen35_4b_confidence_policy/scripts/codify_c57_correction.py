#!/usr/bin/env python3
"""Correct C57 after the powered-up n=400 vLLM re-run reversed the escalation
sub-result. Idempotent (keyed on the corrected title)."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
c57 = next(c for c in d["claims"] if c["id"] == "C57")

c57["title"] = (
    "COMPUTE-OPTIMAL CONFIDENCE POLICY (corrected, powered-up): on the fixed "
    "Qwen3.5-4B for MBPP, single-token P(True) confidence-SELECT is the best "
    "verifier-free selector at every matched-compute k (k=9: 0.762 > majority "
    "0.742 > mean-logprob 0.725; per-candidate AUROC 0.77), max-P(True) ABSTENTION "
    "gives a clean risk-coverage curve (solvability AUROC 0.72), and DEPTH (a "
    "higher think budget) modestly beats BREADTH on the overall accuracy-vs-tokens "
    "frontier (pure-2048 0.593 > pure-256 0.581). BUT selectively ESCALATING the "
    "abstained tail to depth does NOT beat matched-compute breadth: at the "
    "powered-up n=400 all four abstain-fraction deltas are +0.004..+0.022 with 95% "
    "bootstrap CIs spanning 0 — the initial n=24-60 escalation win (+0.15) was a "
    "small-sample artifact, caught by the claim's own pre-registered power-up."
)
c57["status"] = "Promising"
c57["summary"] = (
    "Experiment qwen35_4b_confidence_policy. PART 1 (post-hoc, cached 244-task MBPP "
    "pool, HF-judge p_true): confidence-select (argmax p_true) is the best "
    "verifier-free selector at every k (k=9 conf 0.762 > majority 0.742 > logprob "
    "0.725 > greedy 0.701; exec-line 0.840, oracle 0.848); conf+abstain "
    "risk-coverage clean and monotone (cov 1.00->0.757, 0.68->0.866, 0.43->0.902). "
    "PART 2 (the escalation arm; regenerated on vLLM ~10x faster than the first HF "
    "pass, n=120 -> powered up to n=400 x k=6 at budgets 256 vs 2048; a vLLM judge "
    "readout bug was fixed — the model emits the SPACE-PREFIXED ' A'/' B' after "
    "'Answer: ', ids 357/417, not the bare 32/33). The vLLM p_true is a strong "
    "signal (per-cand AUROC 0.756-0.769, solvability AUROC 0.715-0.727, matching "
    "the HF pool). At MATCHED token-compute with conf-select: pure-2048 modestly "
    "dominates pure-256 on the whole frontier (high-k 0.593 vs 0.581), REPLICATING. "
    "But selectively escalating the abstained (bottom-by-max-p_true) tail to budget "
    "2048 does NOT beat matched-compute extra breadth at 256: esc-minus-breadth = "
    "+0.022 (20%, CI [-0.044,+0.085]), +0.004 (30%, [-0.045,+0.057]), +0.017 (40%, "
    "[-0.024,+0.057]), +0.006 (50%, [-0.028,+0.039]) — every 95% CI includes 0. The "
    "earlier n=24-60 (HF) escalation win (hardest-20% 0.458 vs 0.304) did NOT "
    "replicate; it was a small-sample artifact. Likely mechanism for the null: MBPP "
    "nearly SATURATES the budget (the 4B self-limits to ~108-172 think tokens even "
    "at budget 2048), so the serial-compute lever is too weak to differentiate here."
)
c57["implication"] = (
    "The robust deployable core is SELECT + ABSTAIN, both verifier-free from one "
    "concentrated logit: sample k, pick argmax single-token P(True), abstain below "
    "a max-P(True) threshold. For spending extra compute, DEPTH (higher think "
    "budget) is modestly better than BREADTH uniformly across the frontier, but "
    "there is NO measured benefit to selectively escalating the low-confidence tail "
    "over just doing more breadth on it — on MBPP, which nearly saturates the think "
    "budget. Whether the selective-escalation lever matters on a genuinely "
    "budget-BOUND task family (where the 4B does not self-limit) is the open "
    "question. Methodological lesson: the n=24-60 subset win was pre-registered as "
    "under-powered and REVERSED on the n=400 re-run — power up abstained-subset "
    "effects before claiming them."
)
c57["next_tests"] = [
    "Test the escalation lever on a genuinely budget-BOUND task family (harder multi-step reasoning where the 4B does NOT self-limit at ~100-170 think tokens) — MBPP saturates the budget, which likely explains the null here.",
    "Menagerie arbitration of the SELECT+ABSTAIN policy: does verifier-free P(True) selection + abstention beat majority-vote at matched compute on the held-out benchmark?",
    "Depth ladder 256->1024->4096->8192 on the abstained tail with n>=400 + CIs: confirm whether depth ever beats breadth selectively at a budget where the model actually keeps thinking.",
]
c57["avoid"] = [
    "Do not claim the select-abstain-ESCALATE policy: the escalation (selective depth on the abstained tail) is a NULL result at n=400 (all 95% CIs span 0) — only SELECT + ABSTAIN and a modest OVERALL depth>breadth survive.",
    "Do not read the original n=24-60 escalation win (+0.15) as real — it did not replicate at n=400; it was a small-sample artifact.",
    "Do not use the bare 'A'/'B' token ids (32/33) for a vLLM P(True) judge — the model emits the space-prefixed ' A'/' B' (357/417) after 'Answer: '; reading 32/33 yields a degenerate 0.5.",
    "Do not generalize the escalation null beyond budget-saturated tasks: MBPP self-limits at ~100-170 think tokens, so it under-stresses the serial-compute lever.",
]

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C57 corrected (escalation -> null at n=400; select+abstain survive).")
