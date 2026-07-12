# Idea Intake: Jacobian Transport Control Replication

## Trigger

`qwen35_4b_context_local_jacobian_clamp` produced an otherwise perfect causal
result: all-24 J changed direct key and mapped digit on 48/48 fresh mappings,
pair J reached 47/48 consequences, wrong-donor J produced its own consequence on
48/48, and logit/random controls produced 0/48. Its frozen verdict was correctly
`INVALID_CONTROL` because one random consequence row missed the 1e-5 realized
norm tolerance by 1.55e-6, and bf16 rounding introduced up to 5.7% realized
projection into the J span.

## Closest near-duplicates

1. `qwen35_4b_context_local_jacobian_clamp`: direct parent and exact mechanism.
2. `qwen35_4b_jacobian_value_transport`: late answer-position parent, where
   direct writing did not transport.

## Material delta

- Freeze the parent lens and band; no refitting, layer selection, or coefficient
  search.
- Use entirely new mapping tuples and source/target/wrong assignments.
- Put quantization control feasibility on a separate calibration split.
- Optimize random requested deltas using only current residual geometry until
  the **realized post-bf16** delta meets both norm and span-projection thresholds.
- Use two independent random arms and retain wrong-donor J as the stronger
  same-subspace label-specificity control.
- Open confirmation only if every calibration layer/control passes.

## Decision value

- Calibration fails: stop as `CONTROL_UNREACHABLE`; do not reopen outcome data.
- Calibration passes, J does not replicate: parent was unstable or mapping-
  specific; do not enter thoughts.
- J replicates but random/wrong specificity fails: reject J semantics.
- Every gate passes: establish context-local semantic transport as a replicated
  oracle mechanism and design a separate thought-state/non-oracle experiment.

## Boundary

Fresh prompt-local lookup tables only; no benchmark content. Only the pinned 4B.
No training, adapter, target digit gradient, or deployable capability claim.
