#!/usr/bin/env python3
"""Fold the HumanEval generalization bound into C57. Idempotent."""
import json
from pathlib import Path

LEDGER = Path("knowledge/claims/claim_ledger.json")
d = json.loads(LEDGER.read_text())
c57 = next(c for c in d["claims"] if c["id"] == "C57")

marker = "HumanEval"
if marker not in c57["summary"]:
    c57["summary"] += (
        "  GENERALIZATION (HumanEval, cached 68-task pool, same schema): the "
        "confidence-SELECT advantage is DIFFICULTY-DEPENDENT and does NOT transfer "
        "to easy HumanEval — base pass is 0.91 (67/68 solvable), and there "
        "confidence-select TIES majority-vote (k=9 both 0.941; majority is slightly "
        "ahead at several k). On MBPP (base pass ~0.53) conf-select clearly beat "
        "majority (0.762 vs 0.742). So single-token P(True) selection helps only "
        "when the task is hard enough that selection matters; when the model is "
        "already ~0.9 accurate, self-consistency catches up. Abstention still works "
        "on both (HumanEval risk-coverage clean, though only 1 unsolvable task). "
        "P(True) > mean-logprob holds on both."
    )

if "difficulty-dependent" not in c57["title"]:
    c57["title"] = c57["title"].replace(
        "single-token P(True) confidence-SELECT is the best verifier-free selector "
        "at every matched-compute k (k=9: 0.762 > majority 0.742 > mean-logprob 0.725; "
        "per-candidate AUROC 0.77)",
        "single-token P(True) confidence-SELECT is the best verifier-free selector on "
        "MODERATE-difficulty MBPP (k=9: 0.762 > majority 0.742 > mean-logprob 0.725; "
        "per-cand AUROC 0.77) but only DIFFICULTY-DEPENDENTLY so — it ties majority-vote "
        "on easy HumanEval (base pass 0.91, both 0.941)")

# refresh next_tests / avoid with the generalization bound
if not any("difficulty" in t for t in c57["next_tests"]):
    c57["next_tests"].insert(0,
        "Map the difficulty-dependence of conf-select-vs-majority: at what base pass rate does the P(True)-select advantage appear/vanish? (MBPP 0.53 -> advantage; HumanEval 0.91 -> tie.)")
c57["avoid"].append(
    "Do not claim conf-select beats majority-vote in general: it is difficulty-dependent — an MBPP-scale win (base ~0.53) that VANISHES to a tie on easy HumanEval (base 0.91), where self-consistency catches up.")

LEDGER.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
print("C57 updated with HumanEval difficulty-dependence bound.")
