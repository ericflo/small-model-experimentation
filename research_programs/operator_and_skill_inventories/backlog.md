# Backlog

## Next Experiments

- **Exactly-one-macro slot-conditioned sweep (conditional follow-up, new experiment):** use fresh,
  contamination-controlled procedural data and give every treated prompt exactly one explicit
  macro slot. Run mined, support-matched random, and designed libraries under identical slot
  positions, task schedules, vLLM budgets, and sampling schedules. Add a literal-chunk hint arm
  that exposes the same primitive expansion in the same slot but does not permit a callable alias.
  Compare against a matched-compute base arm with the expanded primitive slots and enough sampling
  to meet or exceed treatment token cost. The long-context follow-up has now cleared its
  plan-given K=4 record-level gate, so run this only if its adequately provisioned induction
  comparison shows alias-selection ambiguity that the slot intervention can isolate. Create it
  under a fresh experiment id; do not extend either verified-macro directory.
- Scale operator banks while measuring search cost, target coverage, and selected accuracy.
- Train top-k shortlisters with held-out primitive families.
- Add active disambiguation for type-colliding operators.
- Compare human-designed, mined, and model-discovered inventory entries.
- Define a portable operator-card schema with examples, invariants, aliases, and failure modes.

## Required Controls

- Closed-vocabulary baseline.
- Full-search oracle.
- Random shortlister.
- Held-out operator/family splits.
- Literal-chunk, non-callable hint matched to each callable composite.
- Matched-compute sample-more base on the same vLLM backend.
- Identical macro-slot schedules across mined, random, and designed libraries.

## Stop Conditions

Do not call inventory expansion progress if coverage rises but deployable selected accuracy falls
without a clear selection plan. Do not interpret a cap-bound alias-emission run as evidence about
macro quality. The long-context plan-given record-level gate is now established; induction must
still pass its own termination gate and beat both its literal-chunk hint and matched
random/computation controls.
