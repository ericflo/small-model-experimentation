# Adversarial Design Review

## Question

Does the C46 P(True) code-confidence result replicate on HumanEval in a
verifier-free setting?

## Main Threats

1. **This could be MBPP-specific.** HumanEval gives a second code substrate, but
   its public examples are not uniform. The primary endpoint therefore uses all
   164 tasks with `--visible-tests 0` and compares only verifier-free selectors.
2. **Public probes could turn the task into execution selection.** Public-output
   majority and visible-test execution are excluded from the main run. They are
   reported only in a separate 68-task diagnostic where one parseable doctest is
   available.
3. **The signal could be verbosity.** Code length is included as the surface
   baseline, and the headline AUROC is within-problem on mixed problems only.
4. **Mean-logprob could be advantaged by shorter/easier completions.** P(True)
   and mean-logprob are compared on the same candidate set with paired bootstrap
   over tasks.
5. **Judge OOM could bias the completed candidate set.** The harness writes the
   checkpoint and `_logprob.json` before judging, then uses
   `--judge-batch-size 1` for the long HumanEval prompts.

## Required Controls

- Report random-pick expectation and oracle pass@8 to show available headroom.
- Report P(True) versus mean-logprob with paired bootstrap.
- Report within-problem AUROC versus length on mixed problems only.
- Keep the public-probe subset clearly separate from the all-task no-probe
  endpoint.
- Preserve negative or ceiling-limited comparisons instead of folding them into
  the headline.

## Verdict

The design answers the intended scope: HumanEval verifier-free selection. A
positive result supports C46's cross-benchmark law for P(True); it does not claim
that confidence beats execution where public tests are available.
