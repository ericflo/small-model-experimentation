# Cyclic Transition Ladder Experiment Log

## Objective

Test whether modular arithmetic inductive bias is the missing ingredient for
learning recurrent belief-state transitions in an oracle-initialized slot
memory.

Each example starts from:

```text
B = A + d (mod p), with A unknown
```

Programs apply arithmetic updates and observation filters over hidden registers
`A` and `B`. The model receives the initial support in slots, then must update
that support recurrently. The experiment varies only the transition
architecture.

## Primary Questions

1. Can a generic per-slot MLP learn the transition at small modulus but fail at
   larger modulus?
2. Do residue/Fourier features improve the generic MLP transition?
3. Does a cyclic candidate mixer learn scale-stable modular transitions?
4. Does a learned router over exact equivariant primitives provide an upper
   learnable-dispatch control?
5. How close do learned transition variants get to the exact transition
   ceiling?

## Metrics

- `decoder_query_target_mass`: probability assigned to the exact final query
  support after projecting the decoded slot belief.
- `decoder_belief_target_mass`: probability assigned to the exact final
  `(A,B)` support.
- `decoder_belief_top1_on_support`: whether the highest-probability pair is in
  the exact final support.
- `mean_slot_purity`: mean product of the strongest `A` and `B` probabilities
  per slot.
- `mean_route_entropy`: entropy of the learned transition routing distribution
  where applicable.
- `mean_route_accuracy`: whether the highest-probability transition route
  matches the true operation where applicable.

The strict headline metric is `decoder_belief_target_mass`; query mass is the
task-level readout metric.

## Artifact Layout

- Code and lightweight outputs: `experiments/cyclic_transition_ladder/`
- Checkpoints: `large_artifacts/cyclic_transition_ladder/checkpoints/`
- Run outputs: `experiments/cyclic_transition_ladder/runs/<variant>/`
- Analysis outputs: `experiments/cyclic_transition_ladder/analysis/`

## Planned Sequence

1. Smoke tests on modulus 7 to validate exact, MLP, Fourier MLP, cyclic mixer,
   and primitive-router transitions.
2. Pilot runs on modulus 11 to compare transition learnability under moderate
   scale.
3. Main runs on modulus 31 for the full ladder.
4. Optional scale run on modulus 97 if the main sweep identifies a promising
   learned transition.
5. Generate tables, figures, checkpoint manifest, standalone report, and HTML
   report.

## Variant Plan

- `exact_ceiling`: exact recurrent transition, no trainable parameters.
- `mlp`: generic MLP transition from expected slot embeddings.
- `fourier_mlp`: generic MLP transition with cyclic residue features.
- `cyclic_mixer`: learned gates over equivariant candidate updates.
- `primitive_router`: learned router over exact equivariant primitive updates.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/cyclic_transition_ladder/src/`
- `experiments/cyclic_transition_ladder/reports/`
- `experiments/cyclic_transition_ladder/runs/`
- `experiments/cyclic_transition_ladder/analysis/figures/`
- `large_artifacts/cyclic_transition_ladder/checkpoints/`

Next action: implement the transition ladder and run smoke tests.

## 2026-06-21 Smoke Tests

Implemented the training and evaluation harness:

- Oracle slot initialization.
- Exact, generic MLP, Fourier-feature MLP, cyclic-mixer, and primitive-router
  transitions.
- Full prefix-belief supervision.
- Decoded query and belief metrics.
- Slot purity, route entropy, and route-accuracy diagnostics.
- External checkpoint writing.
- Analysis script and figures.

Ran five modulus-7 smoke variants with evaluation lengths 2 and 3:

| Variant | Transition | Steps | L=2 query | L=2 belief | L=3 query | L=3 belief | L=3 route acc |
|---|---|---:|---:|---:|---:|---:|---:|
| `smoke_exact_ceiling` | exact | 0 | 100.0% | 100.0% | 100.0% | 100.0% | n/a |
| `smoke_mlp` | MLP | 500 | 83.9% | 57.0% | 70.3% | 45.3% | n/a |
| `smoke_fourier_mlp` | Fourier MLP | 500 | 73.0% | 36.5% | 56.8% | 24.5% | n/a |
| `smoke_cyclic_mixer` | cyclic mixer | 500 | 100.0% | 100.0% | 100.0% | 100.0% | 89.6% |
| `smoke_primitive_router` | primitive router | 500 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Smoke interpretation:

- The exact ceiling confirms that generation, decoding, and K-indexed
  evaluation are wired correctly.
- The generic MLP learns partially on the small task.
- Fourier features alone do not improve the generic MLP in this short smoke
  run.
- Both structured routed variants reach exact held-out smoke performance. The
  primitive router learns exact operation dispatch; the cyclic mixer reaches
  exact decoded belief even though its three gate families are not all
  perfectly one-hot.

Pilot decision:

- Run all five transition modes at modulus 11.
- Keep oracle initialization fixed.
- Use training lengths 1-6 and held-out evaluation lengths 3, 6, 9, and 12.
- Use the pilot to decide which learned transition modes deserve full
  modulus-31 budgets.

## 2026-06-21 Pilot Sweep

Ran five modulus-11 pilot variants with training lengths 1-6 and evaluation
lengths 3, 6, 9, and 12:

| Variant | Transition | Steps | L=3 query | L=6 query | L=9 query | L=12 query | L=12 belief | L=12 route acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pilot_exact_ceiling` | exact | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a |
| `pilot_mlp` | MLP | 900 | 73.7% | 55.1% | 44.8% | 38.2% | 16.4% | n/a |
| `pilot_fourier_mlp` | Fourier MLP | 900 | 60.6% | 38.4% | 28.4% | 23.1% | 5.1% | n/a |
| `pilot_cyclic_mixer` | cyclic mixer | 600 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 91.2% |
| `pilot_primitive_router` | primitive router | 500 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Pilot interpretation:

- The exact ceiling remains perfect through held-out length 12.
- The generic MLP learns a useful but incomplete transition at modulus 11.
- Fourier residue features alone are weaker than the generic MLP.
- The cyclic mixer exactly preserves the target belief through held-out length
  12 after training only up to length 6.
- The primitive router also stays exact and learns the direct operation route.
- The cyclic mixer reaches exact decoded belief with route accuracy below
  100%, indicating that the candidate families have some redundant routes.

Main decision:

- Run the full five-variant ladder at modulus 31.
- Use training lengths 1-8 and evaluation lengths 4, 8, 12, 16, and 24.
- Keep exact and primitive-router rows as ceilings/controls.
- Keep MLP and Fourier MLP rows as negative baselines even though the pilot is
  weak, because they define the value of adding cyclic transition structure.

## 2026-06-21 Main Sweep

Ran five modulus-31 main variants with training lengths 1-8 and evaluation
lengths 4, 8, 12, 16, and 24:

| Variant | Transition | Steps | L=4 query | L=8 query | L=12 query | L=16 query | L=24 query | L=24 belief | L=24 route acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `main_exact_ceiling` | exact | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a |
| `main_mlp` | MLP | 1200 | 66.4% | 50.3% | 39.1% | 31.4% | 22.6% | 5.9% | n/a |
| `main_fourier_mlp` | Fourier MLP | 1200 | 60.2% | 43.9% | 33.1% | 26.1% | 18.4% | 3.9% | n/a |
| `main_cyclic_mixer` | cyclic mixer | 800 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 94.3% |
| `main_primitive_router` | primitive router | 600 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Main interpretation:

- The exact ceiling remains perfect at modulus 31.
- The generic MLP fails to learn the exact recurrent transition at scale. At
  held-out length 24 it reaches only 22.6% query mass and 5.9% strict belief
  mass.
- Fourier residue features are not enough. They underperform the generic MLP,
  reaching only 18.4% query mass and 3.9% belief mass at length 24.
- The cyclic mixer reaches exact decoded belief at every evaluated length,
  including held-out length 24.
- The primitive router also reaches exact decoded belief and exact operation
  route accuracy.
- The cyclic mixer route accuracy is below 100% because its A, B, and weight
  gate families contain redundant correct routes; exact decoded belief is the
  stricter behavioral metric.

Scale decision:

- Run a small modulus-97 scale check for the exact ceiling, cyclic mixer, and
  primitive router.
- Do not run modulus-97 MLP baselines because both dense baselines already
  fail at modulus 31 and modulus-97 dense pair decoding is substantially more
  expensive.

## 2026-06-21 Modulus-97 Scale Check

Ran three modulus-97 variants with training lengths 1-8, evaluation lengths 4,
8, 12, 16, and 24, and 128 examples per evaluated length:

| Variant | Transition | Steps | L=4 query | L=8 query | L=12 query | L=16 query | L=24 query | L=24 belief | L=24 route acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_exact_ceiling` | exact | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a |
| `scale_cyclic_mixer` | cyclic mixer | 300 | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% | 99.8% | 100.0% |
| `scale_primitive_router` | primitive router | 300 | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% | 100.0% |

Scale interpretation:

- The structured transition variants transfer to a much larger residue space.
- The cyclic mixer and primitive router remain essentially exact through
  length 24 after training only on lengths up to 8.
- The small drop below 100% at length 24 is in probability mass, not top-level
  route selection; route accuracy is 100% for both learned scale variants.

Next action: write the standalone report and HTML artifact, generate the
checkpoint manifest, and run the final audit.

## 2026-06-21 Final Audit

Report artifacts generated:

- `reports/cyclic_transition_ladder_paper.md`
- `reports/cyclic_transition_ladder_paper.html`

Consistency checks:

- Source compilation passed for `src/cyclic_transition_ladder_experiment.py`
  and `src/analyze_cyclic_transition_ladder.py`.
- The experiment directory is 6.0M and contains no `.pt`, `.pth`, or `.ckpt`
  files.
- External artifacts are stored under
  `large_artifacts/cyclic_transition_ladder/` and total 3.8M.
- `checkpoint_manifest.csv` contains 14 rows; every listed checkpoint exists
  and matches the recorded byte count.
- The HTML report references 3 local figure files, and all referenced figures
  exist.
- There are 18 run directories, and each contains `metrics_final.csv` and
  `results.json`.
- The standalone paper markdown and HTML were checked for backward-looking
  references to external experiment context; no matches were found.
- No `__pycache__` directories remain under the experiment directory.

Final status:

- The experiment has its own subdirectory with source, runs, analysis, reports,
  checkpoint manifest, and a progress log.
- Large model checkpoints are separated from the experiment directory and
  indexed by manifest.
- The write-up is standalone and does not rely on any external experiment
  context.
