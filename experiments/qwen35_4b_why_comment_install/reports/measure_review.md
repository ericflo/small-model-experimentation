# Measure Review

Transfer measurement of the why_comment composite vs base (shared
fitness harness, greedy pass@1; comments are INERT to the execution
grader, so this is the clean unconfounded test of the WHY hypothesis)
plus the agentic confirm.

Results (base -> why_comment):
- HumanEval: 0.7622 (125/164) -> 0.7927 (130/164), +5 problems
  (paired: 11 why_comment-only vs 6 base-only; McNemar p=0.332).
- MBPP: 0.5650 (113/200) -> 0.5550 (111/200), -2 problems.
- Agentic (duet-eval gen4, n=35): 8/35 -> 8/35, symmetric 3v3 discordant
  (McNemar p=1.000).

FROZEN TIGHTENED RULE: INSTALLED_CODING fires (HumanEval +5 well above
the >=3 threshold; MBPP within the 0.02 retention tolerance).

HONEST READ: TARGET-SPECIFIC WEAK POSITIVE. why_comment gives the
program's BIGGEST single-function gain (+5 HumanEval, on the clean
inert-to-grading test — genuinely suggestive that teaching WHY improves
per-function code quality) but is FLAT on the agentic multi-step real
target (symmetric reshuffle). Not individually significant (HE p=0.332).
Recipe note: the epoch-1 recipe undertrained this high-entropy target
(loss 6.3); retrained at 4 epochs (converged ~0.05) before measurement.

CROSS-BET FINDING (the actionable one): the two positive bets are
COMPLEMENTARY and target-specific — why_comment (WHY reasoning) helps
single-function correctness (HumanEval +5, agentic flat); self_repair
(loop behavior) helps the agentic loop (agentic +2, HumanEval +3). This
motivates STACKING self_repair + why_comment: if the gains are
complementary the combined effect may clear significance and confirm
both ingredients.

**Verdict:** `PASS_MEASURE`.
