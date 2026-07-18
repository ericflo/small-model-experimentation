# Measure Review

Transfer measurement of the self_repair composite vs base (shared
fitness harness, greedy pass@1) plus the agentic confirm.

Results (base -> self_repair):
- HumanEval: 0.7622 (125/164) -> 0.7805 (128/164), +3 problems.
- MBPP: 0.5650 (113/200) -> 0.5550 (111/200), -2 problems.
- Agentic (duet-eval gen4, n=35, PRIMARY real target): 8/35 -> 10/35,
  discordant 1 base-only / 3 self_repair-only (asymmetric-positive).

FROZEN TIGHTENED RULE: INSTALLED_CODING fires (HumanEval +3 meets the
>=3-problem threshold; MBPP within the 0.02 retention tolerance).

HONEST READ: WEAK POSITIVE, first non-null. Unlike bet #1 (execution-
tracing: symmetric 5v5 reshuffle, exactly flat), self_repair is
directionally positive on the two most relevant signals (HE +3, agentic
+2) with an ASYMMETRIC agentic discordant (3 self_repair-only vs 1
base-only) — adding, not just reshuffling. BUT underpowered: no delta is
individually significant (agentic McNemar p~0.63, only 4 discordant;
+3/164 marginal near ceiling). So: promising direction, NOT a confirmed
win. Finding: LOOP-BEHAVIOR curricula (repair) outperform PASSIVE-SKILL
curricula (tracing) for coding — consistent with the agentic-gap-is-
behavior hypothesis. self_repair is a candidate INGREDIENT to stack with
the WHY-family bets and to confirm with a larger agentic N.

**Verdict:** `PASS_MEASURE`.
