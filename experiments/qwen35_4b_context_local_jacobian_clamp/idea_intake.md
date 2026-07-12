# Idea Intake: Context-Local Jacobian Clamp

## Rough idea

Repair the causal mismatch exposed by `qwen35_4b_jacobian_value_transport`.
Instead of swapping an averaged coordinate at the final prediction position,
fit concept pullbacks at the earlier token that names a selected key, clamp its
coordinates to a clean counterfactual donor trajectory, and ask whether a later
arbitrary lookup consequence changes.

## Program fit

Primary program: `interpretability_and_diagnostics`. The experiment distinguishes
three properties—decodability, local writability, and downstream transport—and
uses the distinction to decide whether native-thought value work is warranted.
It becomes relevant to `structured_execution_and_compilers` and
`test_time_reasoning_budget` only after a nontrivial consequence gate passes.

## Closest near-duplicate

`qwen35_4b_jacobian_value_transport` is the direct parent. It fitted an averaged
24-token targeted J lens and applied source/target pair swaps at the final prompt
position. A layer-24 swap redirected 75% of direct concept reports, but the
mapped consequence remained 0% at all layers. Its own audit identified three
unresolved variants: patch the represented antecedent position, use a true fixed
clamp across a layer band, and norm-match controls by realized delta.

Related anchors found by `make related` are C20 (ActAdd inert), C30
(decode-to-prompt works), C52 (token-local labels do not imply update locality),
and `qwen35_4b_context_composition` (an explicit procedure can make an installed
module usable).

## Material delta

1. The intervention site moves from the answer position to the selected-key
   token that causally precedes a shared suffix.
2. Directions are fitted as context-local pullbacks from a future concept report
   to that selected-key position.
3. Every patched layer is set to a fixed clean donor coordinate vector; no
   coordinate is swapped twice.
4. The random control is orthogonal to the full J dictionary and exactly matches
   the realized J delta norm for every item and layer.
5. Full-activation donor transport selects the band without seeing J outcomes.
6. Target-digit gradients are diagnostic-only and cannot leak the answer into
   the intervention.

## Decision value

- Full donor fails: retire this prompt position/interface and locate a causal
  state site before more J work.
- Full donor passes but J clamp fails: the J dictionary is not a semantic state
  basis; pivot to donor-delta decomposition or learned context-gated subspaces.
- J clamp passes direct only: preserve the token-motor interpretation.
- J clamp changes the mapped digit specifically: proceed to a new thought-prefix
  experiment with the same consequence firewall, then seek a non-oracle policy.

## Contamination and model boundary

All tables are generated inside this experiment from public concept names and
random one-digit assignments. No benchmark directory is read. The only model is
the pinned `Qwen/Qwen3.5-4B`.
