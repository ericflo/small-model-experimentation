# Public-verifier recovery branch tournament report

## Summary

**Status: `PROSPECTIVE_DEV_INFEASIBLE`; stopped before selector scoring.** The
old policy complementarity did not transfer to four new families, and the mixed
union could not beat equal-compute same-policy sampling.

## Research program fit

The predecessor repaired the agent interface and found stable complementary
coverage, but no globally superior checkpoint. This experiment tests whether
the complementarity is deployably capturable before spending on winner-trace
curriculum compression.

## Method

Action-only and λ=.18 each receive one greedy six-call recovery branch. Action
replaces candidate only when action's final workspace passes visible tests and
candidate's does not. Candidate is the fixed default. Two complete stochastic
trajectories from each single policy receive the same 12,288-token reservation
and are scored with the generous pass-if-either hidden coverage upper bound.

Four new procedural families and two seeds provide 80 dev plus 80 confirm
cases. Controls and union feasibility precede selector scoring on each block.

## Results

Base scored 49/80 (61.25%). Candidate and action-only each scored 59/80
(73.75%). Two complete candidate trajectories still scored 59/80, while two
action trajectories scored 60/80 (75.0%). The deterministic candidate/action
union was also 60/80. Its preregistered ceiling needed to exceed each comparator
by 3pp, so all feasibility checks failed and the selector was not applied.

Paired source outcomes were 58 both-correct, one candidate-only, one
action-only, and 20 both-wrong. Every shared failure belonged to
`atomic_reservations`. Both greedy policies nevertheless changed code within
two turns in 100% of rejected-patch and failed-test cases, localizing the miss to
semantic conjunction rather than process control. Across deterministic and
stochastic action runs, only one of 40 atomic trajectories solved.

## Controls

C54 apex context, each source policy, two candidate trajectories, two action
trajectories, exact random-policy expectation, deterministic random choice, and
action-default public selection.

## Oracle versus deployable evidence

The hidden union and pass-if-either sample-more are oracle evaluation ceilings.
The primary tournament arm selects before hidden tests using only public visible
passes. No benchmark source or result is accessed.

## Interpretation

Visible-test arbitration is useful only when source policies propose different
successful workspaces. On the new tasks, their diversity collapsed: the same
transactional conjunction defeated both. The next capability producer must
shift proposals with executable supervision for atomic validate-copy-commit
patterns, not add branches or tune a selector.

## Next experiment

Build diverse transactional repository families, obtain executable tool-found
solutions the incumbent almost never proposes, and train an action-seam
curriculum mixed with the existing conditional recovery replay. Gate locality,
ordinary recovery retention, unseen transactional transfer, and sample-more
before Menagerie.

## Artifact manifest

See `artifact_manifest.yaml`.
