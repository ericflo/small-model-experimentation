# Qwen3.5-4B Counterfactual Order-Support Selector Experiment Log

## 2026-07-13 — Intake and adversarial design

- Created a distinct follow-up rather than extending the result-bearing parent.
- Copied only the parent's 113-task qualification real/shuffled slot rows and
  verified exact source hashes. Confirmation artifacts remain absent.
- Froze one primary rule, five strong deployable baselines, reverse and
  task-mismatch controls, paired task uncertainty, breadth gates, and the
  matched-compute limitation before calculating any derived selector accuracy.

## 2026-07-13 — Terminal qualification negative

- Ran the single frozen primary rule after pushed boundary `89a4949e`.
- Decision `NO_ORDER_SUPPORT_SELECTOR`: 43/113 candidate successes (38.05%).
- Candidate beat first by +10.62pp (lower +4.42pp) and majority by +8.85pp
  (lower +2.65pp), but missed the registered all-comparator rule.
- Gains over mean probability, max confidence, and minimum entropy were
  +5.31pp, +2.65pp, and +1.77pp; all paired lower bounds crossed zero.
- The oracle-balanced task-mismatched shuffle scored 44/113 versus candidate
  43/113, defeating task-specific counterfactual attribution.
- Breadth passed at 11 predicted and 10 successful aliases; reverse delta was
  only 8/113. These cannot rescue the failed point/uncertainty gates.
- Rerun hashes were byte-identical: summary `9fc1af5c...`, task predictions
  `03d2e4ff...`.
- Confirmation files remain absent. No fresh K=3-versus-K=6 GPU experiment is
  authorized.
