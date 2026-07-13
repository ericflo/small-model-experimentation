# Preregistration

Status: draft pending adversarial design review. No model/GPU stage is
authorized.

## Decision order

1. Construct fresh, disjoint calibration and mechanics tasks and prove zero
   parent identity/prompt/seed overlap.
2. Run the three interface policies only on known-answer echo/calibration rows.
3. An interface qualifies only if exact echo >=0.90, parse >=0.90, cap contact
   <=0.05, and all transaction/token/EOS checks pass.
4. Freeze the qualifying interface using a deterministic rule specified before
   outcomes. If none qualifies, stop `NO_VALID_RESIDUAL_ANSWER_SEAM`.
5. Only then open disjoint mechanics and compare materialized, name-only,
   shuffled, and taskwise matched-compute direct arms on the same backend.

## Non-rescue rules

No cap increase, parser change, arm deletion, threshold change, backend mix,
task reuse, or downstream peeking is allowed after interface outcomes. The
cheap one-token viability/top-four branch from the parent is terminal and is
not part of this experiment.
