# Exploratory Route Diagnostics

## Status

These diagnostics explain the completed split-branch result. They are
descriptive, were written after the preregistered gate fired, and neither
change that gate nor authorize MOPD. The machine-readable source is
`analysis/route_diagnostics.json`; the analysis is reproducible with
`scripts/analyze_route_diagnostics.py`.

## What Failed

The branch scorer could estimate each policy's absolute return reasonably
well. Across the two blocks, selection-to-audit Pearson correlations were
`0.857/0.835` for quick, `0.854/0.792` for deep, and `0.843/0.791` for the
student; mean absolute errors were `0.093`--`0.117`. The failure appeared only
after conditioning on the noisy three-way maximum.

| Selected route | Block | Selection margin vs student | Audit mean vs student | Audit dominator precision |
| --- | ---: | ---: | ---: | ---: |
| quick | 0 | +0.334 | +0.176 | 12/29 = 0.414 |
| quick | 1 | +0.319 | -0.019 | 6/26 = 0.231 |
| deep | 0 | +0.267 | +0.118 | 11/22 = 0.500 |
| deep | 1 | +0.404 | +0.059 | 12/34 = 0.353 |

Here, “audit dominator” means the selected teacher beat both alternatives on
the disjoint audit branches for that exact state. This is not a new gate; it
shows the label noise hidden by a positive conditional mean. Quick block 1 is
the decisive winner's-curse case: the apparent `+0.319` selection advantage
became `-0.019` on audit, with only 23.1% of selected states retaining quick as
the audit winner.

Independent halves also selected the same quick state only 12/29 times in
block 0 and 6/26 in block 1 (Jaccard `0.267` and `0.122`). Deep agreement was
11/22 and 12/34 (Jaccard `0.282` and `0.261`). Overall three-way agreement
looks less alarming (`0.719`, `0.641`) only because both halves abstained on
most states.

## Why A Larger Fixed Margin Is Not The Repair

The original route deliberately required only strict positive advantage. A
post-result threshold sensitivity confirms that adding the earlier arbitrary
`+0.10` convention would not have rescued quick block 1:

| Minimum observed selection margin | Retained quick states | Audit mean vs student | Audit dominator precision |
| ---: | ---: | ---: | ---: |
| 0.00 | 26 | -0.0194 | 0.231 |
| 0.10 | 24 | -0.0259 | 0.208 |
| 0.25 | 22 | -0.0089 | 0.227 |
| 0.50 | 6 | +0.0540 | 0.167 |

The apparent effect was already large. The problem was conditional estimation,
not a too-permissive effect-size floor. The `0.50` row has only six states and
worse winner precision; it is an exploratory tail, not a candidate gate.

## Scope And Route Structure

Without routing, neither teacher dominated the soup across blocks. Quick's
unconditional audit delta versus the student was `+0.0100` then `-0.0097`;
deep's was `+0.0063` then `-0.0196`. The positive deep result is therefore a
conditional advantage, not an endpoint effect.

Routing was also concentrated in atom states: 101/288 atoms (35.1%) versus
10/96 episodes (10.4%). Selected exact cells replicated weakly across blocks:
quick cell Jaccard was `0.235` and deep was `0.294`. The evidence does not yet
support a claim that the route composes multi-step episode policies.

## Post-Result Cross-Block Sensitivity

As a diagnostic only, coarse group routes were fitted on one block and scored
on the other. A group routed only when its selection and audit halves agreed on
the same strict teacher winner. Four granularities were inspected; reporting
all four avoids selecting the favorable one after seeing outcomes.

| Grouping | Fit 0, evaluate 1: n / cell-macro vs student / alternate | Fit 1, evaluate 0: n / cell-macro vs student / alternate |
| --- | --- | --- |
| exact family-kind-level cell | 39 / +0.0000 / +0.0845 | 25 / +0.0514 / +0.0051 |
| family-kind | 70 / +0.0459 / +0.0444 | 17 / +0.1063 / +0.0250 |
| family | 94 / +0.0332 / +0.0269 | 12 / +0.0879 / -0.0183 |
| kind-level | 70 / -0.0040 / +0.0540 | 55 / +0.0214 / +0.0075 |

This does not discover a qualified replacement router. Exact cells were flat
against the student in one direction; family and kind-level each failed a
contrast; and the superficially positive family-kind rule selected only deep
in the reverse direction (17 states), so it did not replicate two-teacher
support. Each block is also reused once for fitting and once for evaluation,
and these granularities were inspected post-result. A third untouched block
would be required before treating any coarse rule as evidence.

## Scientific Read

The experiment validates the state-prefix unit but rejects four-branch
three-way argmax as a sufficiently stable labeler for both teachers. The
strongest surviving signal is deep's replicated conditional advantage. MOPD
itself remains untested.

The next two tests should be separated:

1. Test the update kernel in a new deep-only experiment on fresh states. Freeze
   the already validated selection/audit rule, require deep to requalify, then
   ask whether deep-routed MOPD can improve the soup while preserving its quick
   behavior. This is the shortest path to learning whether verifier-backed
   local advantage can be installed at all.
2. Before another two-teacher integration attempt, replace statewise argmax
   labels with cross-fitted direct advantage estimates
   `E[R_teacher - R_student | state]`. Fit on acquisition blocks, use
   uncertainty-aware/sequential branch allocation, freeze the predictor, and
   require both teachers to replicate on a third untouched block. If quick
   still fails, retire it as a complementary teacher rather than tune a
   margin.

Any future integrated checkpoint still owes the original terminal comparison:
both source teachers, the soup, a visible router, matched controls, and
matched-compute sample-more on sealed evaluation.
