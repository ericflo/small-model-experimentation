# Pre-Selection Adversarial Implementation Audit

Completed after outcome-blind model smoke and before any scientific row. No task
correctness, chosen alias, trace text, or comparison was visible. This checks
that the powered runner enforces the 60-point design rather than merely
describing it.

## Verdict

Proceed after committing/pushing this audit. The implementation uses one fixed
cap, 113 task-bootstrap units, exact row/control contracts, task/alias diversity
gates, and a hash-locked equally powered confirmation. J stages remain fatal-
unavailable. No rule changed after model behavior was observed.

## Assertions

1. Runtime design boundary passes ancestor, README, preregistration,
   adversarial-review, semantic-config, and lens hashes.
2. Runtime data receipt passes exact hashes for 322 rows across four splits and
   zero overlap with five named parents.
3. Runtime power receipt passes parent-analysis hash, required/planned N=113,
   both seam-stage counts, and planned power >=0.80.
4. Config contains only cap `[1024]`; CPU smoke rejects any cap ladder change.
5. Qualification and confirmation each contain 113 tasks and exactly three
   traces/task, so task macro and pooled accuracy coincide without unequal
   weighting.
6. Qualification exact cardinality is 339 traces, 339 real slots, 339 shuffled
   slots, 339 free-form controls, and 113 no-thought slots; code derives and
   compares these counts before any scientific summary.
7. The same derived cardinality check applies independently to confirmation.
8. Native and free-form cached calls require one full prefill followed only by
   length-one forwards; any failing row aborts.
9. Real, shuffled, and no-thought decisions call one identical cache-free slot
   function with the same suffix, alias IDs, dtype, model, and argmax.
10. Shuffled control hashes source/permuted sequences, requires identical length
    and sorted token multiset, and records moved-position rate.
11. Every actually evaluated shuffled/no-thought row must be finite; malformed
    pre-cap paths remain denominator failures rather than numeric-control aborts.
12. Full-vocabulary top token, alias mass, and alias probabilities are computed
    from the same logits as constrained choice; there is no replay drift.
13. `slot_metrics` groups real/shuffled rows by task ID, computes task means,
    and bootstraps the list of 113 task differences—not 339 traces.
14. Bootstrap uses 10,000 stable stage/cap-specific resamples and the frozen 5%
    one-sided lower quantile. The gate uses strict `> 0`, so an exact zero fails.
15. No-thought comparison is one deterministic row/task and remains a +3pp
    point gate; its task lower bound is clearly named diagnostic.
16. Mixed tasks count a task only when its three real traces contain at least one
    correct and one incorrect choice.
17. Correct-alias support counts distinct gold alias strings among correct real
    rows; repeated rows cannot inflate the eight-alias gate.
18. Chosen-alias support counts distinct real argmax strings and independently
    requires eight.
19. Full-vocabulary top-is-alias and mean alias-mass thresholds are both inside
    `seam_gate`, not report-only diagnostics.
20. Close-only parse/success/answer-cap metrics are absent from `seam_gate` and
    cannot rescue or invalidate the matched slot comparison.
21. Gate reachability records 68--237 feasible successes and 28 mixed tasks
    among 339 traces; target/choice support maxima are also checked against the
    11/12 available aliases.
22. Qualification tests exactly one cap and labels a pass
    `POWERED_COMMIT_SLOT_SEAM_QUALIFIED`; it cannot select another budget.
23. Confirmation verifies all five qualification raw-file hashes, reads only
    the untouched split, applies identical thresholds, and cannot pool stages.
24. Model smoke result is schema-checked for `outcomes_recorded=false` and
    `correctness_recorded=false`; the receipt stores neither chosen alias nor
    trace text.
25. Scientific rows are buffered until complete contracts pass. Progress output
    contains only completed trace counts.
26. Value, control-calibration, and causal-confirmation commands raise a fatal
    unavailable error and cannot emit placeholder evidence.
27. Seven CPU tests cover exact-depth/freshness, gate reachability, strict
    bootstrap/diversity failures, prefix boundaries, separated controls, exact
    shuffle, semantic-config sensitivity, and parent/config power matching.
28. Repository validation, syntax, links, text, charts, brief, dates, and site
    rendering passed before the scientific run.

## Accepted limitations

- The power approximation inherits a 16-task parent variance estimate and may
  be optimistic; the nonparametric gate is decisive.
- The run is intentionally large and non-resumable at scientific-summary level;
  interruption cannot create partial evidence.
- Shuffled thought tests order versus identical token bag, not all coherent
  control counterfactuals.
- Fixed syntax and alias masking define constrained choice, not free-form
  capability.

None permits a retry, threshold change, cap increase, decoder change, or pooled
rescue inside this experiment.
