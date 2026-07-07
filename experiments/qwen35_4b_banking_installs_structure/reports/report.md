# Does banking install STRUCTURE? Yes — base structure-coverage 0.00 → banked 0.51, converting the wall from structure-bound to value-bound

## Motivation
C32 established that the compositional wall is a STRUCTURE-proposal problem: the base can't propose the depth-3
op-sequence (failures are wrong-skeleton), while values are trivially searchable once structure is known
(oracle-skeletonfill = 1.0). C22–24 showed banking crosses depth-3. So banking must install the STRUCTURE the
base lacks. This tests it directly.

## Method
Run C32's **format-immune structure-coverage** (does the model program's *behavior* match the true op-type
skeleton with *any* params?) on the **base** vs the **banked_1280** model, on held-out depth-3 tasks (banked_1280's
frozen eval, disjoint from its training), min-depth-verified. no-think, greedy@1 + cov@8.

## Result (held-out depth-3, n=80)
| model | greedy@1 | cov@8 | STRUCTURE-cov@8 | value tax (struct−concrete) |
|---|---|---|---|---|
| base | 0.000 | 0.000 | **0.000** | +0.000 |
| banked_1280 | 0.200 | 0.362 | **0.512** (±0.105) | **+0.150** |

- **Banking installs STRUCTURE.** Base structure-coverage 0.000 → banked **0.512** on *held-out* depth-3.
  Banking installs **generalizable** op-sequence structure (held-out tasks ⇒ not memorized skeletons) — the exact
  capability C32 showed the base lacks.
- **Banking converts the wall from structure-bound to value-bound.** The base has no skeletons at all
  (struct = concrete = 0). The banked model proposes the *right skeleton* 51% of the time but nails the *full
  concrete program* only 36% — a **value tax of +0.15** (right-skeleton-wrong-param failures the base never had,
  because it had no skeletons).
- **The value tax is fillable.** Since oracle-skeletonfill = 1.0 (C32: values trivially searchable given
  structure), value-filling the banked model's proposed skeletons would deploy at ~**0.512** vs 0.362 for the
  banked model alone — a concrete recipe: *bank installs structure; cheap value-search recovers the value tax.*

## Implication
Mechanistic confirmation and closure of the C32 loop: the compositional wall is structure-proposal (base can't),
and **banking's entire lever is installing that structure** (0 → 0.51 held-out). Once structure is present, the
residual is a small, fillable value gap. This unifies the arc — C22–24 (banking crosses depth-3), C32 (the wall
is structure), C31 (values surface-readable/searchable): **banking = structure-installation.** It also explains
why value-side interventions (C31 param-hint, C29 DPO) never moved the *base* wall — the base's problem is
structure, not values; only after banking installs structure does a (fillable) value gap even exist.

## Honest scope
- "Structure" = op-type sequence on the list DSL; single banked model (banked_1280, C24) and one held-out set
  (n=80). The +0.15 value tax has a wide CI at n=80. The bank+fill deploy (~0.512) is inferred from
  structure-coverage + C32's oracle-skeletonfill = 1.0, not run end-to-end here.

## Artifact Manifest
See `reports/artifact_manifest.yaml`. Uses the external banked_1280 adapter (C24, in scratchpad).
