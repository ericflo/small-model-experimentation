# Preregistration: public-verifier recovery branch tournament

Frozen before any Qwen generation on the prospective families.

## Claim under test

A bounded portfolio of the locality-safe recovery action and λ=.18 policies can
convert their complementary proposal coverage into a deployable recovery gain
using only a final visible-test bit, and can beat spending the same reserved
model-token compute on two complete trajectories from either source alone.

This experiment can establish a capability-producing harness. It cannot
establish a single-checkpoint gain or Menagerie improvement.

## Immutable inputs

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Frozen C54 apex, recovery action-only, and λ=.18 checkpoints; hashes in
  `configs/default.yaml`.
- Frozen retrospective source files and hashes in the config. They select the
  rule and may not be treated as prospective evidence.
- Prospective dev: four new families × ten tasks × two controlled states at
  seed 85000 (80 cases).
- Prospective confirm: identical size, seed 85100, no adaptation.

## Selector and public boundary

Run action-only and λ=.18 independently from the byte-identical controlled
state for at most six calls each. After each branch ends, the host runs only the
committed visible suite on its final workspace. Select action iff
`action.final_visible_pass == true` and
`candidate.final_visible_pass != true`; select candidate otherwise.

The chosen-arm function accepts only those two booleans. Tests enforce that
changing quarantined hidden-success fields cannot change the arm. Branch
evaluation files contain hidden booleans for scoring, but they are never passed
to the chosen-arm function. Family, scenario, model confidence,
entropy/varentropy, sampled-token count, commit behavior, and hidden outcome are
not selector inputs.

## Compute contract

All calls use the copied vLLM 0.24 runner, 512 thinking tokens, 512 answer
tokens, and identical engine geometry.

- Mixed tournament: 2 policies × 1 trajectory × 6 calls × 1,024 = 12,288
  reserved tokens/case.
- Candidate sample-more: 2 stochastic trajectories × 6 calls × 1,024 = 12,288.
- Action sample-more: same.
- Each greedy source is also reported at 6,144 reserved tokens/case.

Sample-more is scored pass-if-either using hidden evaluation, an oracle-generous
coverage upper bound. The public tournament must still exceed it by 3pp. Record
actual sampled tokens, answer-cap hits, invalids, and turns for every arm.

## Controls and order

For each prospective block, generate base deep, candidate deep, action deep,
candidate sample-more, and action sample-more before selector scoring. Compute
the deterministic mixed-policy hidden union only to test gate feasibility. If
that ceiling cannot clear the best single and each sample-more arm by 3pp, stop
before applying the public selector. No threshold or rule changes follow.

The exact random-choice expectation is the casewise mean of the two source
success bits; a frozen hash-random selector is retained diagnostically but does
not set a noisy gate.

## Prospective gate

The public selector passes a block iff all hold:

- success ≥ best source +0.03;
- success ≥ candidate pass-if-either sample-more +0.03;
- success ≥ action pass-if-either sample-more +0.03;
- success ≥ exact expected random source choice +0.03;
- paired case-bootstrap 95% lower bounds versus the best source and stronger
  sample-more arm are each ≥0 (10,000 resamples, frozen seeds);
- selected success captures ≥0.85 of the action/candidate hidden-success union;
- selected rejected-patch valid changed-within-two ≥0.80;
- selected failed-test changed-within-two ≥0.70;
- selected invalid-action and answer-cap rates are each ≤ candidate +0.02;
- at least three of four family deltas versus the best source are nonnegative;
  no family delta is below −0.10.

Prospective confirm repeats every control and threshold unchanged. Dev failure
stops confirm. Confirm failure stops winner banking. Passing both authorizes a
new experiment to mine verifier-selected winners and balance them at
state→action transitions; it does not authorize Menagerie here.

## Stop labels

- `RETROSPECTIVE_RULE_INVALID`
- `PROSPECTIVE_DEV_INFEASIBLE`
- `PROSPECTIVE_DEV_FAIL`
- `PROSPECTIVE_CONFIRM_INFEASIBLE`
- `PROSPECTIVE_CONFIRM_FAIL`
- `PUBLIC_TOURNAMENT_POSITIVE`

Every stop and control is preserved. Prospective family code, thresholds, and
selector are not changed inside this result-bearing directory.
