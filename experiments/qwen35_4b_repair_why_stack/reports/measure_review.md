# Measure Review

Transfer measurement of the repair_why_stack composite (corpus-union of
self_repair + why_comment) vs base.

Results (base -> stack):
- HumanEval: 125/164 -> 126/164, +1 problem (paired 11 base-only / 12
  stack-only, heavy churn, McNemar p=1.000).
- MBPP: 113/200 -> 110/200, -3 problems.
- Agentic (duet-eval gen4, n=35): 8/35 -> 7/35.

FROZEN RULE: NULL (no >=3-problem gain; not RETENTION_FAIL — MBPP -3
= -0.015 within the 0.02 tolerance, HumanEval +1).

HONEST READ: MIXTURE DILUTION. The corpus-union stack DILUTED BOTH
component effects — why_comment's +5 HumanEval collapsed to +1, and
self_repair's 10/35 agentic collapsed to 7/35 (at/below base). Training
ONE adapter on both curricula at half-concentration splits capacity and
washes out each effect — the same mixture-dilution law the menagerie
program found (one kind per dose at full concentration; mixing installs
less). The complementary effects DO NOT combine via corpus-mixing.
CONSEQUENCE: the correct way to combine two full-concentration
complementary fine-tunes is WEIGHT-SPACE (task vectors: base +
(self_repair-base) + (why_comment-base), each delta at full magnitude),
tested next as a cheap no-training follow-on.

**Verdict:** `PASS_MEASURE`.
