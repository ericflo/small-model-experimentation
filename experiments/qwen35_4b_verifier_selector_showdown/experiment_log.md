# Qwen3.5-4B Verifier vs Visible Selector Showdown Experiment Log

## Scaffold

Matched-cost, deployable follow-up to the generator-verifier gap (C10): is the thinking-verifier worth its
cost as a selector vs the cheap visible test? Fully offline -- reuses the generator-verifier candidate pool
(k=8, P(A) signals + full_pass) and adds a visible-test (first-assert) label per candidate.

## Results (see reports/report.md)

Deployable selectors on the k=8 pool: pass@1 0.771, visible-only 0.850, no-think verifier 0.800, thinking
verifier 0.860, visible+no-think verifier 0.870, visible+thinking verifier 0.870, oracle 0.890. C2 false-pass
rate 6.6%. The thinking verifier is Pareto-dominated (0.860 at ~5x cost); no-think ties it in the combination.
Best deployable = visible + no-think verifier (0.870, ~free), closing 83% of the pass@1->oracle gap. Tempers
C10: thinking-verification's edge only matters in verifier-only settings; when a visible test exists, use it
+ a free no-think verifier.
