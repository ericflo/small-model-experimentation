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

## Addendum (phase 2): end-to-end bank + value-fill DEPLOY — and the brute-force control that flips the conclusion

C33 inferred bank+value-fill "would deploy at ~0.51". Phase 2 runs it end-to-end (no oracle) and — following the
review — adds the decisive **brute-force-all-4096 deploy control**. Recipe: the banked model emits Python, so
recover its proposed STRUCTURE from behavior (run each of k=8 samples, infer which op-type skeleton it implements),
value-fill against the true outputs, and deploy via **execution-consensus** (plurality output-vector on 16 fresh
probe inputs — leakage-free, C17-legal). Coverage is tautological (≥ struct-cov), so **DEPLOY is the headline**.

### Result (held-out depth-3, n=80)
| recipe | deploy | coverage |
|---|---|---|
| banked model alone (greedy@1, forward pass) | 0.200 | (cov@8 0.325) |
| **bank-fill** (model's structure + value-fill + select) | 0.463 ±0.10 | 0.475 |
| **brute-fill** (all-4096 structure + value-fill + select) | **0.975 ±0.03** | 1.000 |

- **C33 confirmed:** bank-fill deploys at 0.463 ≈ the banked model's structure-coverage (0.475) — the value-fill
  recovers the value tax exactly on the tasks where the model proposes the structure.
- **But the model's structure is DOMINATED by brute-force search.** Brute-force structure enumeration + value-fill
  + execution-consensus deploys at **0.975** — near-solving depth-3 *without the model*. After the 8-visible filter
  only ~2 skeletons survive per task (the DSL isn't value-fungible, C32), and consensus picks the right one 97.5%
  of the time (visible-overfit 2.7%).
- **Using the model's structure is WORSE than ignoring it:** bank-fill (0.463) is *capped* at the model's
  structure-coverage because the banked model proposes the right skeleton only ~48% of the time, while brute
  enumeration always contains it.

### Implication
With the interpreter available (mission-legal, free selection per C17), **free structure-SEARCH dominates the
model at deploy** (0.975 vs 0.46 vs 0.20). Banking's installed structure (C33: 0.20 → 0.475) is a **forward-pass
asset** — it matters only if you must deploy in a single pass with no interpreter; the moment you can
search+execute, brute-force structure enumeration + value-fill + consensus near-solves the wall and the model is
unnecessary. **Scope:** brute-force wins *because* the depth-3 structure space (4096) is enumerable; the model's
structure-pruning would only become a deployable lever when the space is too large to brute-force (larger
op-inventory or deeper compositions).
