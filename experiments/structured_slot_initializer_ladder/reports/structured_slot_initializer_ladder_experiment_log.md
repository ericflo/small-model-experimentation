# Structured Slot Initializer Ladder Experiment Log

## Objective

Test whether a structured initializer can place the sparse modular support

```text
B = A + d (mod p), with A unknown
```

into weighted slots before an exact recurrent transition executes the program.
Each slot stores one distribution over `A`, one distribution over `B`, and one
mixture weight. The experiment varies only the initializer while keeping the
transition update exact by default.

## Primary Questions

1. Can a generic prompt-conditioned initializer learn the support reliably?
2. Does hard-wiring the cyclic relation `B=A+d` leave only slot coverage to
   learn?
3. Does a free `B` initializer fail relative to the cyclic-relation initializer?
4. Do coverage, relation, and slot-purity regularizers improve strict initial
   belief mass?
5. Does any learned initializer remain reliable at larger modulus and held-out
   program length?

## Metrics

- `init_belief_target_mass`: probability assigned to the exact initial
  `(A,B)` support before any recurrent transition steps.
- `init_slot_relation_accuracy`: fraction of slots whose strongest `A` and
  `B` values satisfy `B=A+d`.
- `init_slot_unique_a_frac`: fraction of the target residue set represented by
  distinct slot-level `A` argmaxes.
- `decoder_belief_target_mass`: probability assigned to the exact final
  `(A,B)` support after executing `K` recurrent steps.
- `decoder_query_target_mass`: probability assigned to the exact final query
  support after projecting the decoded belief.
- `mean_slot_purity`: mean product of the strongest `A` and `B` probabilities
  per slot.

The strict headline metric is `decoder_belief_target_mass` at the first
evaluated `K >= L`, with `init_belief_target_mass` as the mechanistic
diagnostic.

## Artifact Layout

- Code and lightweight outputs:
  `experiments/structured_slot_initializer_ladder/`
- Checkpoints:
  `large_artifacts/structured_slot_initializer_ladder/checkpoints/`
- Run outputs:
  `experiments/structured_slot_initializer_ladder/runs/<variant>/`
- Analysis outputs:
  `experiments/structured_slot_initializer_ladder/analysis/`

## Planned Sequence

1. Smoke tests on modulus 7 to validate all initializer modes and diagnostics.
2. Pilot runs on modulus 11 to compare generic, factorized, and structured
   regularized initializers.
3. Main runs on modulus 31 using the most informative pilot variants.
4. Scale check on modulus 97 if a learned structured initializer is promising.
5. Generate tables, figures, checkpoint manifest, standalone report, HTML
   report, and final audit.

## Variant Plan

- `oracle`: exact initializer ceiling.
- `generic_mlp`: prompt-conditioned MLP over delta and slot id.
- `factorized_cyclic`: learned slot-to-residue logits for `A`, with `B`
  produced by the exact cyclic shift `A+d`.
- `factorized_free_b`: learned slot-to-residue logits for `A`, with a free
  delta-conditioned MLP for `B`.
- `sinkhorn_cyclic`: learned slot-to-residue logits normalized with Sinkhorn
  iterations before applying the exact cyclic shift `A+d`.
- `indexed_cyclic`: deterministic indexed cyclic initializer control.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/structured_slot_initializer_ladder/src/`
- `experiments/structured_slot_initializer_ladder/reports/`
- `experiments/structured_slot_initializer_ladder/runs/`
- `experiments/structured_slot_initializer_ladder/analysis/figures/`
- `large_artifacts/structured_slot_initializer_ladder/checkpoints/`

Implemented the initializer ladder harness:

- Exact modular task generator and exact recurrent transition.
- Oracle, generic MLP, factorized cyclic, factorized free-B, and indexed cyclic
  initializers.
- Full-prefix belief and final-query supervision.
- Optional initializer coverage, relation, weight-uniformity, and slot-entropy
  regularizers.
- Initializer-specific diagnostics for initial support mass, relation accuracy,
  and unique slot coverage.
- External checkpoint writing and analysis script.

Next action: run source compilation and modulus-7 smoke tests.

## 2026-06-21 Smoke Tests

Source compilation passed for both experiment scripts.

Ran modulus-7 smoke variants with evaluation lengths 2 and 3:

| Variant | Init mode | Supervision | Steps | L=3 query | L=3 belief | Initial belief | Relation acc | Unique A slots |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `smoke_oracle_ceiling` | oracle | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `smoke_indexed_cyclic` | indexed cyclic | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `smoke_generic_mlp` | generic MLP | full belief | 500 | 93.3% | 84.7% | 85.4% | 90.8% | 88.0% |
| `smoke_generic_mlp_final_query` | generic MLP | final query | 500 | 89.9% | 76.1% | 77.9% | 89.8% | 79.6% |
| `smoke_factorized_cyclic_plain` | factorized cyclic | full belief | 500 | 85.4% | 64.4% | 69.3% | 100.0% | 71.4% |
| `smoke_factorized_cyclic_reg` | factorized cyclic | full belief | 500 | 85.7% | 65.1% | 70.3% | 100.0% | 71.4% |
| `smoke_factorized_cyclic_overlap` | factorized cyclic | full belief | 900 | 92.7% | 83.0% | 85.3% | 100.0% | 85.7% |
| `smoke_factorized_free_b_reg` | factorized free-B | full belief | 500 | 83.8% | 62.5% | 67.0% | 91.1% | 85.7% |

Smoke interpretation:

- The oracle and indexed cyclic controls are exact, validating generation,
  exact transition, K-indexing, and decoding.
- Generic MLP initialization is a strong small-modulus baseline when trained
  with full prefix-belief supervision.
- Final-query supervision learns less initial support than full belief.
- Factorized cyclic initialization learns the relation `B=A+d` immediately by
  construction, but without an explicit overlap penalty it duplicates slots and
  leaves part of the residue set uncovered.
- Adding slot-overlap regularization fixes most of that failure mode and brings
  the factorized cyclic initializer close to the generic MLP baseline.
- The free-B factorized ablation is weaker, suggesting that the cyclic relation
  constraint is useful but not sufficient by itself.

Pilot decision:

- Run p=11 pilots for oracle, generic full-belief, generic final-query,
  factorized cyclic with overlap, and factorized free-B with relation/coverage
  regularization.
- Use training lengths 1-6 and held-out evaluation lengths 3, 6, 9, and 12.

## 2026-06-21 Pilot Sweep

Ran five modulus-11 pilot variants with training lengths 1-6 and evaluation
lengths 3, 6, 9, and 12:

| Variant | Init mode | Supervision | Steps | L=12 query | L=12 belief | Initial belief | Relation acc | Unique A slots | A overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pilot_oracle_ceiling` | oracle | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 |
| `pilot_factorized_cyclic_overlap` | factorized cyclic | full belief | 1400 | 99.3% | 98.9% | 99.5% | 100.0% | 100.0% | 0.000 |
| `pilot_factorized_free_b_reg` | factorized free-B | full belief | 1100 | 77.9% | 69.0% | 76.3% | 92.5% | 81.8% | 0.039 |
| `pilot_generic_mlp_final_query` | generic MLP | final query | 1100 | 71.5% | 60.8% | 68.3% | 89.2% | 69.6% | 0.085 |
| `pilot_generic_mlp_full_belief` | generic MLP | full belief | 1100 | 66.9% | 55.0% | 59.2% | 79.9% | 64.2% | 0.142 |

Pilot interpretation:

- The exact ceiling is again perfect, so the task is not lossy under exact
  transition and sufficient initialization.
- The factorized cyclic initializer with overlap regularization is nearly
  exact through held-out length 12.
- The decisive diagnostic is slot coverage: the best structured initializer
  reaches 100% unique `A` slots and zero measured `A` overlap, while every
  learned baseline leaves duplicated or diffuse slot support.
- Freeing `B` hurts despite relation regularization, so the cyclic relation is
  not merely a weak auxiliary target; it is the critical structural constraint.
- Generic final-query supervision beats generic full-belief supervision at this
  scale, but both are far below the structured cyclic initializer.

Main decision:

- Run modulus-31 main variants for oracle, generic full-belief, generic
  final-query, factorized free-B, and factorized cyclic with overlap.
- Use training lengths 1-8 and held-out evaluation lengths 4, 8, 12, 16, and
  24.

## 2026-06-21 Main Sweep

Ran five modulus-31 main variants with training lengths 1-8 and evaluation
lengths 4, 8, 12, 16, and 24:

| Variant | Init mode | Supervision | Steps | L=24 query | L=24 belief | Initial belief | Relation acc | Unique A slots | A overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `main_oracle_ceiling` | oracle | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 |
| `main_factorized_cyclic_overlap` | factorized cyclic | full belief | 2400 | 93.6% | 92.2% | 90.0% | 100.0% | 90.3% | 0.007 |
| `main_factorized_free_b_reg` | factorized free-B | full belief | 1600 | 78.9% | 73.3% | 74.2% | 89.0% | 83.9% | 0.015 |
| `main_generic_mlp_full_belief` | generic MLP | full belief | 1400 | 62.8% | 55.5% | 46.4% | 84.5% | 50.4% | 0.119 |
| `main_generic_mlp_final_query` | generic MLP | final query | 1400 | 30.5% | 16.7% | 29.4% | 52.6% | 48.2% | 0.047 |

Main interpretation:

- Structured cyclic initialization scales far better than the generic
  initializers at modulus 31.
- The free-B ablation is substantially better than generic initialization but
  still well below the exact cyclic relation initializer.
- The remaining gap for `main_factorized_cyclic_overlap` is visibly an
  assignment gap, not a relation gap: relation accuracy is 100%, but unique
  `A` slot coverage is only 90.3%.
- The next targeted iteration is a Sinkhorn-normalized cyclic initializer, which
  turns slot coverage from a soft overlap penalty into a structural constraint.

## 2026-06-21 Sinkhorn Iteration

Added a `sinkhorn_cyclic` initializer:

- It learns slot-to-residue logits.
- Sinkhorn normalization makes the slot/residue assignment approximately
  doubly stochastic.
- The exact cyclic shift then sets `B=A+d`.

Ran validation rows:

| Variant | Modulus | Steps | Eval length | Query | Belief | Initial belief | Relation acc | Unique A slots | A overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `smoke_sinkhorn_cyclic` | 7 | 500 | 3 | 98.4% | 96.0% | 96.9% | 100.0% | 100.0% | 0.005 |
| `pilot_sinkhorn_cyclic` | 11 | 800 | 12 | 97.8% | 96.7% | 98.5% | 100.0% | 100.0% | 0.001 |
| `main_sinkhorn_cyclic` | 31 | 2400 | 24 | 98.6% | 98.1% | 99.7% | 100.0% | 100.0% | 0.000 |

Sinkhorn interpretation:

- The Sinkhorn initializer fixes the p=31 assignment gap observed in the
  overlap-regularized factorized initializer.
- At p=31 and held-out length 24, strict belief improves from 92.2% to 98.1%.
- The diagnostic metrics show complete slot coverage: 100% relation accuracy,
  100% unique `A` slots, and zero measured `A` overlap.

## 2026-06-21 Modulus-97 Initializer Scale Check

Ran an initializer-only scale check at modulus 97. This check evaluates `K=0`
support formation only; it does not run full p=97 program execution.

| Variant | Modulus | Steps | Initial query | Initial belief | Relation acc | Unique A slots | A overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_oracle_init_only` | 97 | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.000 |
| `scale_sinkhorn_cyclic_init_only` | 97 | 1200 | 99.7% | 98.7% | 100.0% | 100.0% | 0.000 |

Scale interpretation:

- The Sinkhorn assignment mechanism scales to a 97-residue support in the
  isolated initializer setting.
- Because this row is initializer-only, the p=31 full-program result remains
  the main behavioral result.

Next action: generate the standalone report, HTML artifact, checkpoint
manifest, and final audit.

## 2026-06-21 Final Audit

Report artifacts generated:

- `reports/structured_slot_initializer_ladder_paper.md`
- `reports/structured_slot_initializer_ladder_paper.html`

Consistency checks:

- Source compilation passed for `src/structured_slot_initializer_ladder_experiment.py`
  and `src/analyze_structured_slot_initializer_ladder.py`.
- The paper and HTML report are self-contained and contain no references to
  external experiment artifacts or source scripts.
- The experiment directory contains no `.pt`, `.pth`, or `.ckpt` files.
- Checkpoints are stored under
  `large_artifacts/structured_slot_initializer_ladder/checkpoints/`.
- `checkpoint_manifest.csv` contains 18 rows; every listed checkpoint exists
  and matches the recorded byte count.
- The HTML report references 3 local figure files, and all referenced figures
  exist.
- The analysis directory contains 24 generated figure files.
- The experiment directory is 7.4M; the external artifact directory is 2.0M.
