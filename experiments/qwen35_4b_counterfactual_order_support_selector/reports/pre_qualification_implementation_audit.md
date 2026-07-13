# Pre-Qualification Implementation Audit

Completed after the design review and outcome-blind smoke, before computing any
derived selector accuracy.

## Verdict

**Qualification is authorized after this implementation boundary is committed
and pushed.** Confirmation remains absent and fatal-unavailable.

## Data and firewall assertions

1. Qualification real/shuffle SHA-256 values exactly match the frozen parent
   artifacts.
2. Exactly 113 tasks and 339 rows per arm are present.
3. Every task has exactly trace indices 0, 1, and 2 in both arms.
4. Real and shuffle keys pair exactly by task and trace.
5. Every shuffle source-thought hash equals its ordered-thought hash.
6. All 339 shuffle rows certify exact token-multiset equality.
7. Paired thought lengths match and all rows are finite.
8. Both arms use one identical 12-token alias order.
9. Every named probability vector has exactly the 12 public aliases, is finite,
   nonnegative, and sums to one within `2e-4`.
10. Stored choices are probability-maximal; exact-logit ties may use the
    parent's recorded argmax rather than an invented tie order.
11. Confirmation real/shuffle files do not exist in the new experiment.
12. Smoke reports `confirmation_opened=false`, `model_loaded=false`, and
    `outcome_metrics_computed=false`.
13. The confirmation runner fails before loading rows without a passing
    qualification and committed boundary receipt.
14. The runner never reads source artifacts from the parent directory.
15. Expected confirmation hashes are frozen in config before qualification.

## Selector assertions

16. The primary prediction is exactly argmax of the across-trace mean raw
    probability difference, with public alias order for exact ties.
17. The primary consumes only real/shuffled probability matrices.
18. Label mutation leaves primary and all five deployable baselines unchanged.
19. Mean ordered probability is a distinct mandatory soft-ensemble baseline.
20. Majority uses recorded ordered argmax choices, then mean-probability and
    public-order tie breaks.
21. Max-confidence and minimum-entropy select only from ordered paths.
22. First trace is fixed at trace index zero.
23. Reverse delta uses the smallest, not negated/refit, primary delta.
24. Task mismatch cycles deterministically within correct-alias strata and is
    explicitly named oracle-balanced in every output.
25. Gold never moves from mismatch construction/grading into deployable output.
26. Candidate-in-real-choice-pool is diagnostic and not a gate.

## Statistics and gates

27. Accuracy is task exact; no row, trace, pair, or alias pseudo-replication is
    used.
28. Every paired difference is computed on the same 113 task IDs.
29. Bootstrap uses 10,000 task resamples, deterministic stable name seeds, and
    the one-sided fifth percentile.
30. The primary must beat all five deployable baselines and mismatch; no
    best-looking comparator is selected.
31. Every comparison separately requires +3pp and a lower bound above zero.
32. Candidate accuracy, chosen breadth, successful breadth, and reverse-control
    separation are conjunctive.
33. Baseline/control reachability is evaluated before the scientific gates; a
    comparator above 0.67 makes the fixed 0.70 ceiling plus 3pp infeasible.
34. Qualification cannot pool with confirmation or authorize a claim.
35. Secondary score transforms are not implemented, preventing rescue.
36. Confirmation code checks the qualification bytes and current
    config/selector/runner hashes against an ancestor commit, and requires its
    boundary receipt to be committed at current HEAD.

## Smoke and tests

- Seven tests pass, including formula separation from mean probability, hidden-
  label mutation invariance, majority tie behavior, within-label mismatch
  cycling, deterministic bootstrap, absent confirmation, and fail-closed
  confirmation invocation.
- Smoke passes all 339 paired controls and records implementation hashes:
  config `a4a0e3ff...`, selector `f1493c88...`, runner `ce53ccc3...`.

An implementation issue was caught before outcomes: source probability vectors
are named mappings, not positional lists. A second audit caught exact bf16/logit
ties where the stored argmax is probability-maximal but not the lowest public
index. Parsing now uses the frozen public alias order for vector construction
and the parent's stored choices only for choice-based baselines. Neither issue
opened or summarized correctness.
