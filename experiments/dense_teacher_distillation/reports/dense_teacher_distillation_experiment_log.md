# Dense Teacher Distillation Experiment Log

## Objective

Test whether a recurrent executor with only a fixed-width dense hidden state can represent exact belief-state computation when every prefix state is supervised by an exact teacher distribution.

The task uses modular programs over two registers `(A,B)`. Programs combine arithmetic updates with observation filters. The teacher computes the exact belief distribution over all `(A,B)` pairs after each prefix. The student receives the same symbolic program and initial relation, keeps a dense hidden vector, and is trained to decode the teacher belief at each recurrent step.

## Primary Questions

1. Does full prefix-belief distillation make the dense recurrent executor solve the task, or does it still fail despite high-bandwidth supervision?
2. Does increasing dense state capacity improve exact belief decoding and decoder-projected query accuracy?
3. Does a residual transition cell improve over a GRU cell when the target is exact belief-state execution?
4. Does a low-rank decoder expose a decoder bottleneck by improving or degrading projected query accuracy relative to an MLP decoder?

## Metrics

- `decoder_belief_target_mass`: probability assigned by the student belief decoder to the exact target support.
- `decoder_query_target_mass`: query probability obtained by projecting the decoded pair distribution and measuring mass on the exact query support.
- `probe_belief_target_mass`: probability assigned by a separately trained frozen-state probe to the exact target support.
- `query_target_mass`: direct query-head mass, treated as diagnostic when the query-head loss is disabled.

The headline metric is `decoder_query_target_mass`, with `decoder_belief_target_mass` as the stricter state-execution metric.

## Artifact Layout

- Code and lightweight outputs: `experiments/dense_teacher_distillation/`
- Checkpoints: `large_artifacts/dense_teacher_distillation/checkpoints/`
- Run outputs: `experiments/dense_teacher_distillation/runs/<variant>/`
- Analysis outputs: `experiments/dense_teacher_distillation/analysis/`

## Planned Sequence

1. Smoke test on modulus 7 with tiny capacity and short chains to validate the harness.
2. Pilot on modulus 11 across state sizes and transition/decoder variants.
3. Main bottleneck sweep on modulus 31 with longer held-out chains.
4. Generate tables, figures, checkpoint manifest, standalone report, and HTML report.

## Variant Plan

- `smoke_gru_mlp_d32`: minimal GRU with MLP belief decoder.
- `smoke_residual_mlp_d32`: minimal residual transition with MLP belief decoder.
- `smoke_gru_lowrank_d32_r4`: minimal GRU with low-rank pair decoder.
- `pilot_gru_mlp_d128`: baseline dense state at moderate capacity.
- `pilot_gru_mlp_d256`: capacity scaling check.
- `pilot_residual_mlp_d256`: transition structure check.
- `pilot_gru_lowrank_d256_r16`: decoder structure check.
- `main_gru_mlp_d256`: main baseline.
- `main_gru_mlp_d512`: main capacity scaling check.
- `main_residual_mlp_d512`: main transition check.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/dense_teacher_distillation/src/`
- `experiments/dense_teacher_distillation/reports/`
- `experiments/dense_teacher_distillation/runs/`
- `experiments/dense_teacher_distillation/analysis/figures/`
- `large_artifacts/dense_teacher_distillation/checkpoints/`

Implemented the first harness changes:

- Teacher-belief distillation is the default and only supervision mode.
- Added decoder-projected query metrics from decoded pair distributions.
- Added `variant_name`, `transition`, `decoder_type`, and `decoder_rank` controls.
- Added GRU and residual transition choices.
- Added MLP and low-rank belief decoder choices.
- Checkpoints default to the external `large_artifacts` tree.

Next action: run smoke tests and fix any harness issues before starting the pilot sweep.

## 2026-06-21 Smoke Tests

Ran three two-step CUDA smoke tests at modulus 7:

| Variant | Status | Notes |
|---|---|---|
| `smoke_gru_mlp_d32` | passed | GRU transition and MLP decoder path writes full metrics and checkpoint. |
| `smoke_residual_mlp_d32` | passed | Residual transition path writes full metrics and checkpoint. |
| `smoke_gru_lowrank_d32_r4` | passed | Low-rank decoder path writes full metrics and checkpoint. |

Smoke artifacts:

- Run directories: `experiments/dense_teacher_distillation/runs/smoke_*`
- Checkpoints: `large_artifacts/dense_teacher_distillation/checkpoints/smoke_*`

The metric CSVs include variant metadata, decoder belief metrics, probe belief metrics, and decoder-projected query metrics. The external checkpoint layout is working.

Next action: run a modulus-11 pilot sweep to decide which architecture and capacity settings deserve a larger modulus-31 run.

## 2026-06-21 Pilot Sweep

Ran four modulus-11 pilot variants with train lengths up to 6 and evaluation lengths 3, 6, 9, and 12:

| Variant | L=3 decoder-query | L=6 decoder-query | L=9 decoder-query | L=12 decoder-query | Decision |
|---|---:|---:|---:|---:|---|
| `pilot_gru_mlp_d128` | 90.5% | 61.9% | 44.8% | 35.2% | Keep as capacity baseline only. |
| `pilot_gru_mlp_d256` | 94.7% | 71.3% | 53.1% | 41.4% | Carry forward as main baseline. |
| `pilot_residual_mlp_d256` | 94.8% | 71.6% | 54.7% | 42.6% | Carry forward at larger width. |
| `pilot_gru_lowrank_d256_r16` | 78.8% | 50.3% | 36.4% | 28.5% | Drop from main sweep. |

Strict decoded-belief mass showed the same ordering. The low-rank decoder also produced weaker frozen-probe results, so the constrained decoder appears to alter the learned state rather than merely reducing the final readout.

Analysis artifacts generated:

- `analysis/all_metrics_long.csv`
- `analysis/final_metrics_query_mean.csv`
- `analysis/threshold_summary.csv`
- `analysis/summary.md`
- `analysis/figures/`

Main sweep decision:

- Use MLP belief decoders.
- Test state capacity at modulus 31.
- Include a residual-transition run because it was slightly better on longer held-out chains.
- Evaluate lengths beyond the training range so the key outcome is length transfer, not only short-chain fit.

## 2026-06-21 Main Sweep

Ran three modulus-31 main variants with train lengths up to 8 and evaluation lengths 4, 8, 12, 16, and 24:

| Variant | L=4 decoder-query | L=8 decoder-query | L=12 decoder-query | L=16 decoder-query | L=24 decoder-query |
|---|---:|---:|---:|---:|---:|
| `main_gru_mlp_d256` | 69.9% | 36.1% | 23.7% | 16.8% | 11.7% |
| `main_gru_mlp_d512` | 78.1% | 49.2% | 35.4% | 26.3% | 18.7% |
| `main_residual_mlp_d512` | 80.5% | 52.1% | 36.5% | 27.9% | 20.5% |

Strict decoded-belief mass at K >= L:

| Variant | L=4 belief | L=8 belief | L=12 belief | L=16 belief | L=24 belief |
|---|---:|---:|---:|---:|---:|
| `main_gru_mlp_d256` | 32.6% | 9.0% | 4.0% | 2.5% | 1.6% |
| `main_gru_mlp_d512` | 45.7% | 19.3% | 9.9% | 6.8% | 4.3% |
| `main_residual_mlp_d512` | 50.1% | 21.9% | 11.5% | 7.8% | 5.1% |

Main interpretation:

- Width matters. Moving from d256 to d512 substantially improved both decoder-projected query mass and strict decoded-belief mass.
- The residual transition was consistently, but modestly, better than the GRU at the same d512 width.
- Full teacher-belief supervision did not make the dense state exact on modulus 31. The best run reached strong query support mass for short programs but only 21.9% strict belief mass at train-length 8 and 5.1% at held-out length 24.
- Frozen probes were close to the trained decoder on fixed eval rows, so the bottleneck is mostly the learned dense state and transition, not only the final decoder.

Next action: write the standalone report and HTML artifact, then verify the directory layout and references.
