# Pre-registration: do banking and thinking STACK?

Logged 2026-07-05, before the banked+think data (base cells and banked+no-think measured; design reviewed).
Two levers move different capabilities: C25 -- BANKING installs planning (banked_1280 step-1 next-op ranking
0.138 vs base 0.013); C26 -- THINKING amplifies recognition not planning (base step-1 stays ~chance across
budgets, but step-3 recognition 0.275 -> 0.600). Do they compose?

## Method
- 2x2: {base, banked_1280} x {no-think, think} on per-step next-op top-1 RANKING (think->RANK, channel-matched
  to C25/C26, parse-immune). n=40 frozen true-depth-3 held-out (same set/harness).
- All four cells re-measured IN-RUN where possible: banked+no-think = this run's B=0 (confirmed reproduces C25:
  step-1 0.175 on n=40); base cells reused from C26 (same harness/held-out/n).
- Budgets B in {0, 1024, 2048}. Steps 1 (planning, goal 3 away, clean), 2, 3 (recognition, goal 1 away).

## Predictions (locked)
- **P1 (stacking on PLANNING, step-1):** does thinking lift the BANKED model's step-1 beyond banking-alone
  (0.175)? Hypothesis: NO material lift -- thinking doesn't add planning even after banking (consistent with
  C26). If banked+think step-1 >> banked+no-think -> thinking DOES amplify banked planning (a real stack).
- **P2 (stacking on RECOGNITION, step-3):** banked+think step-3 should be highest of all four cells (both
  levers help recognition); test whether it exceeds banked+no-think (0.525) and base+think (0.600).
- **P3 (interaction):** are the levers ADDITIVE on their own axes (banking owns step-1, thinking owns step-3,
  no interference), or does one cap/redundant-with the other? Decision by the 2x2 pattern.

## Honest limits
Closed-set ranking easier than generation; n=40, one seed/budget. banked+no-think re-measured in-run; base
cells cited from C26 (same setup). End-to-end thinking-guided search is token-expensive; if run, it is a
token-costed footnote, not the headline (per the review).
