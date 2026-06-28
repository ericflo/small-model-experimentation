# Learned Sparse Slot Executor Experiment Log

## Objective

Test whether a neural recurrent executor can learn to represent and update a
sparse slot memory for modular belief-state execution.

Each example starts from:

```text
B = A + d (mod p), with A unknown
```

Programs apply arithmetic updates and observation filters over hidden registers
`A` and `B`. The model maintains `S` slots. Each slot represents a distribution
over one candidate `A` value, one candidate `B` value, and a slot weight. The
decoded belief distribution is the weighted mixture of slot-local `(A,B)`
distributions.

## Primary Questions

1. With oracle slot initialization, can a learned transition module recover the
   arithmetic and filter updates?
2. With exact transitions, can a learned initializer populate a support memory
   from the initial relation?
3. Can the fully learned slot executor combine learned initialization and
   learned transitions?
4. Does final-query supervision alone induce the same slot machinery, or is
   prefix belief supervision required?

## Metrics

- `decoder_query_target_mass`: probability assigned to the exact final query
  support after projecting the decoded slot belief.
- `decoder_belief_target_mass`: probability assigned to the exact final
  `(A,B)` support.
- `decoder_belief_top1_on_support`: whether the highest-probability pair is in
  the exact final support.
- `mean_slot_purity`: mean product of the strongest `A` and `B` probabilities
  per slot.
- `mean_weight_entropy`: entropy of the slot-weight distribution.

The headline metric is `decoder_query_target_mass`, with
`decoder_belief_target_mass` as the stricter execution metric.

## Artifact Layout

- Code and lightweight outputs: `experiments/learned_sparse_slot_executor/`
- Checkpoints: `large_artifacts/learned_sparse_slot_executor/checkpoints/`
- Run outputs: `experiments/learned_sparse_slot_executor/runs/<variant>/`
- Analysis outputs: `experiments/learned_sparse_slot_executor/analysis/`

## Planned Sequence

1. Smoke tests on modulus 7 to validate oracle initialization, learned
   initialization, exact transition, and neural transition paths.
2. Pilot runs on modulus 11 to compare staged learning conditions.
3. Main runs on modulus 31 for the most informative staged variants.
4. Generate tables, figures, checkpoint manifest, standalone report, and HTML
   report.

## Variant Plan

- `smoke_oracle_init_exact_transition`: ceiling path for output and metrics.
- `smoke_learned_init_exact_transition`: learned slot initialization with exact
  recurrent updates.
- `smoke_oracle_init_neural_transition`: oracle slots with learned recurrent
  updates.
- `smoke_learned_init_neural_transition`: fully learned slot executor.
- `pilot_*`: modulus-11 versions of the staged variants that pass smoke.
- `main_*`: modulus-31 runs for the highest-signal staged variants.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/learned_sparse_slot_executor/src/`
- `experiments/learned_sparse_slot_executor/reports/`
- `experiments/learned_sparse_slot_executor/runs/`
- `experiments/learned_sparse_slot_executor/analysis/figures/`
- `large_artifacts/learned_sparse_slot_executor/checkpoints/`

CUDA is available on the current machine. Checkpoints will be kept outside the
experiment directory.

Next action: implement the trainable slot executor and run smoke tests.

## 2026-06-21 Smoke Tests

Implemented the training and evaluation harness:

- Oracle or learned slot initialization.
- Exact or neural recurrent transitions.
- Full prefix-belief supervision or final-query supervision.
- Decoded query and belief metrics.
- Slot purity and weight-entropy diagnostics.
- External checkpoint writing.
- Analysis script and figures.

Ran five modulus-7 smoke variants:

| Variant | Init | Transition | Supervision | Steps | L=2 query | L=2 belief | L=3 query | L=3 belief |
|---|---|---|---|---:|---:|---:|---:|---:|
| `smoke_oracle_init_exact_transition` | oracle | exact | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% |
| `smoke_learned_init_exact_transition` | learned | exact | full belief | 200 | 90.0% | 75.7% | 87.5% | 75.8% |
| `smoke_oracle_init_neural_transition` | oracle | neural | full belief | 500 | 93.9% | 84.4% | 90.8% | 82.2% |
| `smoke_learned_init_neural_transition` | learned | neural | full belief | 700 | 90.5% | 72.8% | 83.6% | 68.3% |
| `smoke_learned_init_exact_final_query` | learned | exact | final query | 700 | 91.2% | 77.3% | 88.0% | 76.8% |

Smoke interpretation:

- The ceiling path is exact, confirming the decoder, target generation, and
  K-indexed evaluation are correct.
- Learned initialization with exact transitions recovers much of the support
  state but is not exact after a short run.
- Learned neural transitions with oracle slots are stronger than the fully
  learned path, so transition learning and initialization learning should stay
  separated in the pilot.
- Final-query supervision is enough to recover substantial belief mass on the
  small task when transitions are exact.

Pilot decision:

- Run modulus-11 staged variants for learned initialization, learned
  transition, full end-to-end learning, and final-query-only initialization.
- Use longer budgets than smoke, but keep the sweep scoped enough to inspect
  results before any modulus-31 run.

## 2026-06-21 Pilot Sweep

Ran four modulus-11 pilot variants with evaluation lengths 3, 6, 9, and 12:

| Variant | Init | Transition | Supervision | Steps | L=3 query | L=6 query | L=9 query | L=12 query | L=12 belief |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `pilot_learned_init_exact_full_belief` | learned | exact | full belief | 1400 | 84.1% | 75.3% | 72.3% | 70.2% | 58.1% |
| `pilot_oracle_init_neural_full_belief` | oracle | neural | full belief | 900 | 98.4% | 97.6% | 97.2% | 97.1% | 95.5% |
| `pilot_learned_init_neural_full_belief` | learned | neural | full belief | 1100 | 84.2% | 72.4% | 64.9% | 57.8% | 39.3% |
| `pilot_learned_init_exact_final_query` | learned | exact | final query | 900 | 89.3% | 82.7% | 80.1% | 78.9% | 70.2% |

Pilot interpretation:

- Learned transition rules are highly learnable when the slot support is
  initialized correctly. The oracle-initialized neural transition reached
  97.1% query mass and 95.5% belief mass at held-out length 12.
- Learned initialization is the harder subproblem. Even with exact transitions,
  the full-belief initializer plateaued near 58.1% belief mass at length 12.
- Final-query supervision trained the initializer better than full prefix
  belief supervision in this pilot, reaching 70.2% belief mass at length 12.
- The fully learned model did not compose the two subskills well at this scale;
  it fell to 57.8% query and 39.3% belief at length 12.

Main sweep decision:

- Include an exact oracle ceiling at modulus 31.
- Scale the successful learned-transition condition:
  `oracle init + neural transition + full belief`.
- Scale the strongest learned-initialization condition:
  `learned init + exact transition + final query`.
- Do not spend a full modulus-31 budget on the fully learned condition unless
  the two staged main runs leave enough time and evidence requires it.

## 2026-06-21 Main Sweep

Ran three modulus-31 main variants with evaluation lengths 4, 8, 12, 16, and
24:

| Variant | Init | Transition | Supervision | Steps | L=4 query | L=8 query | L=12 query | L=16 query | L=24 query | L=24 belief |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `main_oracle_init_exact_ceiling` | oracle | exact | full belief | 0 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `main_oracle_init_neural_full_belief` | oracle | neural | full belief | 1200 | 74.1% | 60.3% | 49.2% | 41.2% | 30.7% | 12.3% |
| `main_learned_init_exact_final_query` | learned | exact | final query | 900 | 84.8% | 74.6% | 66.4% | 59.3% | 49.1% | 36.5% |

Strict belief mass:

| Variant | L=4 belief | L=8 belief | L=12 belief | L=16 belief | L=24 belief |
|---|---:|---:|---:|---:|---:|
| `main_oracle_init_exact_ceiling` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `main_oracle_init_neural_full_belief` | 34.4% | 27.5% | 21.3% | 17.2% | 12.3% |
| `main_learned_init_exact_final_query` | 60.0% | 54.5% | 49.0% | 44.2% | 36.5% |

Main interpretation:

- The exact oracle ceiling remains perfect at modulus 31, so the slot format
  and evaluation path can represent the target state exactly.
- Neural transitions learned well at modulus 11 but did not scale to modulus
  31 with this generic MLP transition. Even with oracle slot initialization,
  held-out length-24 belief mass reached only 12.3%.
- Learned initialization with exact transitions and final-query supervision
  scaled better than the neural transition at modulus 31, reaching 49.1% query
  mass and 36.5% belief mass at length 24.
- Neither learned component recovered the exact sparse support machine at
  modulus 31. The main bottleneck is now learnability and inductive bias, not
  representational sufficiency.

Analysis artifacts generated:

- `analysis/all_metrics_long.csv`
- `analysis/all_metrics_query_mean.csv`
- `analysis/first_k_ge_l_summary.csv`
- `analysis/train_log.csv`
- `analysis/summary.md`
- `analysis/figures/`

## 2026-06-21 Final Audit

Report artifacts generated:

- `reports/learned_sparse_slot_paper.md`
- `reports/learned_sparse_slot_paper.html`

Consistency checks:

- Source compilation passed for `src/learned_sparse_slot_experiment.py` and
  `src/analyze_learned_sparse_slot.py`.
- The experiment directory is 3.5M and contains no `.pt`, `.pth`, or `.ckpt`
  files.
- External artifacts are stored under
  `large_artifacts/learned_sparse_slot_executor/` and total 2.9M.
- `checkpoint_manifest.csv` contains 10 rows; every listed checkpoint exists
  and matches the recorded byte count.
- The HTML report references 5 local figure files, and all referenced figures
  exist.
- The standalone paper markdown and HTML were checked for backward-looking
  phrases such as "previous", "prior", "original", "follow-up", "earlier",
  "older", "past experiment", and "another experiment"; no matches were found.

Final status:

- The experiment has its own subdirectory with source, runs, analysis, reports,
  and a progress log.
- Large model checkpoints are separated from the experiment directory and
  indexed by manifest.
- The write-up is standalone and does not rely on any earlier experiment.
