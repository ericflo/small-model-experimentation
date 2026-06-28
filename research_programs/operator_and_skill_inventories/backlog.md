# Backlog

## Next Experiments

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

## Stop Conditions

Do not call inventory expansion progress if coverage rises but deployable selected accuracy falls without a clear selection plan.
