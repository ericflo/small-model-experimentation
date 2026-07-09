# Backlog

## Next Experiments

- **Exactly-one-macro slot-conditioned sweep (next priority, new experiment):** use fresh,
  contamination-controlled procedural data and give every treated prompt exactly one explicit
  macro slot. Run mined, support-matched random, and designed libraries under identical slot
  positions, task schedules, vLLM budgets, and sampling schedules. Add a literal-chunk hint arm
  that exposes the same primitive expansion in the same slot but does not permit a callable alias.
  Compare against a matched-compute base arm with the expanded primitive slots and enough sampling
  to meet or exceed treatment token cost. This isolates *which composite is useful* from the
  stopped prototype's failures to decide whether, where, and how often to emit aliases. Create it
  under a fresh experiment id; do not extend the stopped verified-macro directory.
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
without a clear selection plan. Do not interpret another free-form alias-emission failure as
evidence about macro quality: first establish the slot-conditioned interface ceiling, then require
the selected library to beat both its literal-chunk hint and matched random/computation controls.
