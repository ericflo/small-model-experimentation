# Idea Intake

## Program Fit

- Programs: `evidence_conditioned_selection` (primary),
  `test_time_reasoning_budget`, and `interpretability_and_diagnostics`.
- Existing or new program: existing.
- Closest program scorecards reviewed: all three named programs in
  `knowledge/program_scorecards.md`.
- Related future queue item: `thinking_content_vs_compute_control`; this is not
  a duplicate because coherent content is established and the new question is
  whether its counterfactual contribution selects answers.

## Prior Evidence

- Anchor 1: `qwen35_4b_commit_slot_semantic_power_replication` proves ordered
  thought beats an exact token-multiset shuffle twice, but its learned J-value
  readout fails.
- Anchor 2: `qwen35_4b_thinking_content_vs_compute` finds coherent content, not
  filler or shuffled tokens, carries the behavioral gain elsewhere.
- Anchor 3: `qwen35_4b_confidence_guided_compute` shows a label-free logit
  selector can beat majority when its confidence signal is calibrated.
- Closest near-duplicate: `qwen35_4b_commit_slot_semantic_power_replication`.
  It used ordered-minus-shuffled probability only as a gold-answer aggregate;
  it never formed a deployable answer selector from the full delta vector.

## Novelty Claim

The unresolved question is whether the per-alias forward effect of coherent
token order is useful without knowing the answer: choose the alias with largest
mean `P(alias|ordered)-P(alias|same tokens shuffled)` across paths.

## Mechanism

The replicated seam says semantic order shifts probability toward correct
aliases on average. Subtracting the identical-token shuffle should cancel alias
priors, token presence, length, and syntax while retaining the ordered semantic
contribution. If raw probabilities, confidence selection, majority, or a
correct-alias-balanced task-mismatched shuffle are as good, the causal group
effect is not an actionable per-task selector.

## Control Plan

- Baseline: first path, majority with soft tie-break, mean ordered probability,
  max-confidence path, and minimum-entropy path on the same three traces.
- Mechanism-falsifying control: subtract shuffled probabilities from another
  task with the same gold alias. Gold is used only to balance this diagnostic;
  it never enters the candidate prediction.
- Shift or robustness check: an independently collected 113-task confirmation
  can open only if qualification passes after code and rules are committed.
- Hidden-label boundary: every deployable prediction consumes only probability
  vectors and public alias order. Unit tests mutate labels and require identical
  predictions. Labels enter only grading and the oracle-balanced control.

## Evidence Output

- Program evidence update: all three named programs if a terminal result changes
  routing.
- Claim ledger or synthesis update: synthesis only; retrospective analysis
  cannot create a capability claim.
- Reusable artifact: pure counterfactual selection and paired task bootstrap
  with hidden-label invariance tests.
- Stop or branch condition: failure retires exact-shuffle support as a selector.
  A two-stage retrospective pass licenses only a fresh K=3 candidate versus K=6
  actual-forward-token sample-more experiment.

## Decision

- Run experiment: yes, as a cheap gated secondary analysis before new GPU work.
- Create program: no.
- Write synthesis only: no; the deployable transform is genuinely untested.
- Defer: fresh task generation/GPU work until the retrospective gate passes.
