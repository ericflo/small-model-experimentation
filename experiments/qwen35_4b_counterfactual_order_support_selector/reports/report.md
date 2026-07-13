# Qwen3.5-4B Counterfactual Order-Support Selector Report

## Summary

Terminal `NO_ORDER_SUPPORT_SELECTOR`. Confirmation artifacts and the fresh
matched-compute successor remain sealed.

## Research Program Fit

## Method

For each task and alias, the frozen primary averaged three ordered probabilities
minus their three exact-token-shuffle probabilities, then selected the largest
delta. It used no label. See the preregistration and 15-point design review.

## Results

| method | correct / 113 | accuracy | primary minus method | one-sided paired lower |
| --- | ---: | ---: | ---: | ---: |
| primary order delta | 43 | 0.3805 | — | — |
| first trace | 31 | 0.2743 | +0.1062 | +0.0442 |
| majority | 33 | 0.2920 | +0.0885 | +0.0265 |
| mean ordered probability | 37 | 0.3274 | +0.0531 | -0.0088 |
| max-confidence trace | 40 | 0.3540 | +0.0265 | -0.0265 |
| minimum-entropy trace | 41 | 0.3628 | +0.0177 | -0.0354 |
| oracle-balanced mismatch | 44 | 0.3894 | -0.0088 | -0.0708 |
| reverse delta | 8 | 0.0708 | +0.3097 | diagnostic |

The candidate passed its 15%--70% range, reachability, 11-alias prediction
breadth, 10-alias success breadth, and reverse-control gap. It failed mandatory
all-comparator point gain and uncertainty. Against minimum entropy it won nine
tasks, lost seven, and tied 97. Against max confidence it won nine, lost six,
and tied 98. These tiny net advantages cannot support a selector.

The candidate selected an alias absent from all three ordered argmax choices on
27/113 tasks and was correct on eight, so the vector can recover weak common
support that voting discards. This is a useful mechanism clue, not a passed
system.

## Controls

The reverse delta collapsed to 8/113, confirming the sign contains information.
But subtracting another task's shuffled distribution from the same correct-alias
stratum reached 44/113, versus 43/113 for the exact matched shuffle. Because the
mismatch is oracle-balanced, its absolute score is not deployable; nevertheless,
the primary also independently failed three direct deployable comparator gates.

## Oracle Versus Deployable Evidence

The primary and five standard baselines are label-free. The task-mismatch
control uses gold only to preserve alias strata and is explicitly oracle-only.
No confirmation or hidden label entered the candidate. This retrospective
qualification is not matched compute: the three shuffle prefills would have to
be charged against additional ordered samples.

## Interpretation

The causal group effect does not become a robust per-task selector through raw
probability subtraction. It beats majority decisively but fails to improve on
simple confidence/entropy with adequate effect and uncertainty, and exact task
matching is not load-bearing against the oracle-balanced mismatch. Retire the
registered transform without log-ratio, residualization, or subset tuning.

## Next Experiments

Do not open confirmation or a fresh compute-matched selector run. Redirect from
commit-logit ranking to a method that changes the proposal or continuation
distribution—where Jacobian counterfactuals could create new branches rather
than score existing ones.

## Artifact Manifest

See `artifact_manifest.yaml`.
