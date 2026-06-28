# Sparse Support Memory Executor Experiment Log

## Objective

Test whether modular filter-program execution becomes exact when the runtime
state is an explicit sparse memory of candidate `(A,B)` support states.

Each example starts from the relation:

```text
B = A + d (mod p), with A unknown
```

A program then applies arithmetic updates and observation filters. The executor
keeps at most `S` support slots. Each slot stores one concrete `(A,B)` pair and
a weight. Arithmetic updates move every active slot. Observation filters delete
inconsistent slots. The output distribution is the normalized weighted support
set represented by the active slots.

## Primary Questions

1. Is a slot budget equal to the initial support size, `S=p`, sufficient for
   exact execution across long programs?
2. How sharply does performance degrade when `S<p`?
3. Does the K-threshold remain clean: weak before `K=L`, exact or near-exact
   once the recurrent budget reaches the program length?
4. Does the same result hold at larger modulus when the support memory scales
   with the modulus?

## Metrics

- `decoder_belief_target_mass`: probability assigned by the slot distribution
  to the exact final pair-support.
- `decoder_query_target_mass`: query probability after projecting the slot
  distribution to the requested query.
- `decoder_belief_top1_on_support`: whether the highest-probability pair is in
  the exact final support.
- `empty_slot_rate`: fraction of examples whose compressed support lost all
  surviving particles and had to fall back to a uniform distribution.
- `mean_active_slots`: mean active slots after the selected recurrent budget.

The headline metric is `decoder_query_target_mass`, with
`decoder_belief_target_mass` as the stricter state-execution metric.

## Artifact Layout

- Code and lightweight outputs: `experiments/sparse_support_memory_executor/`
- Large artifacts: `large_artifacts/sparse_support_memory_executor/checkpoints/`
- Run outputs: `experiments/sparse_support_memory_executor/runs/<variant>/`
- Analysis outputs: `experiments/sparse_support_memory_executor/analysis/`

## Planned Sequence

1. Smoke test on modulus 7 to validate the sparse executor and output files.
2. Pilot capacity sweep on modulus 11.
3. Main capacity sweep on modulus 31.
4. Scale check on modulus 97.
5. Generate tables, figures, checkpoint manifest, standalone report, and HTML
   report.

## Variant Plan

- `smoke_mod7_s{2,4,7}`: tiny validation sweep.
- `pilot_mod11_s{4,8,11}`: small-modulus capacity sweep.
- `main_mod31_s{4,8,16,31}`: main sparse-slot capacity sweep.
- `scale_mod97_s{16,32,64,97}`: larger-modulus capacity sweep.

## 2026-06-21 Setup

Created the standalone experiment directory:

- `experiments/sparse_support_memory_executor/src/`
- `experiments/sparse_support_memory_executor/reports/`
- `experiments/sparse_support_memory_executor/runs/`
- `experiments/sparse_support_memory_executor/analysis/figures/`
- `large_artifacts/sparse_support_memory_executor/checkpoints/`

Next action: implement the sparse support-memory evaluator and analysis script,
then run smoke tests before launching the pilot and main sweeps.

## 2026-06-21 Smoke Tests

Ran three modulus-7 smoke variants with evaluation lengths 2 and 3:

| Variant | Slot capacity | L=2 decoder-query | L=2 belief | L=3 decoder-query | L=3 belief | Empty rate at K=L |
|---|---:|---:|---:|---:|---:|---:|
| `smoke_mod7_s2` | 2 | 75.9% | 68.5% | 70.2% | 61.4% | 32.8-40.2% |
| `smoke_mod7_s4` | 4 | 95.9% | 95.1% | 94.1% | 92.8% | 5.1-7.4% |
| `smoke_mod7_s7` | 7 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |

Smoke interpretation:

- The harness writes `metrics_final.csv`, `results.json`, and analysis artifacts.
- The exact-capacity case, `S=p`, is exactly correct once `K` reaches the
  program length.
- Undersized slot memories degrade smoothly, and the `empty_slot_rate` field
  exposes when the compressed support has lost every surviving state.

Next action: run the modulus-11 pilot sweep with `S=4,8,11`.

## 2026-06-21 Pilot Sweep

Ran three modulus-11 pilot variants with evaluation lengths 3, 6, 9, and 12:

| Variant | Slot capacity | L=3 decoder-query | L=6 decoder-query | L=9 decoder-query | L=12 decoder-query | L=12 empty rate |
|---|---:|---:|---:|---:|---:|---:|
| `pilot_mod11_s4` | 4 | 87.7% | 76.5% | 66.8% | 60.5% | 44.5% |
| `pilot_mod11_s8` | 8 | 91.8% | 86.9% | 81.9% | 80.6% | 22.0% |
| `pilot_mod11_s11` | 11 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |

Pilot interpretation:

- `S=p` again gives exact execution at all tested lengths once `K>=L`.
- Sub-capacity memories retain useful query signal, but they lose exact support
  mass because the initial relation has more live states than the memory can
  store.
- Empty-support fallbacks increase with length for undersized memories, making
  slot loss directly measurable rather than hidden in the query score.

Next action: run the main modulus-31 capacity sweep with `S=4,8,16,31`.

## 2026-06-21 Main Sweep

Ran four modulus-31 variants with evaluation lengths 4, 8, 12, 16, and 24:

| Variant | Slot capacity | L=4 decoder-query | L=8 decoder-query | L=12 decoder-query | L=16 decoder-query | L=24 decoder-query | L=24 empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `main_mod31_s4` | 4 | 70.0% | 55.4% | 42.5% | 32.1% | 23.4% | 80.3% |
| `main_mod31_s8` | 8 | 93.7% | 80.0% | 66.7% | 56.3% | 41.9% | 60.5% |
| `main_mod31_s16` | 16 | 98.5% | 93.1% | 86.9% | 78.2% | 68.6% | 32.5% |
| `main_mod31_s31` | 31 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |

Strict decoded-belief mass followed the same ordering:

| Variant | L=4 belief | L=8 belief | L=12 belief | L=16 belief | L=24 belief |
|---|---:|---:|---:|---:|---:|
| `main_mod31_s4` | 65.3% | 50.6% | 37.8% | 27.7% | 19.9% |
| `main_mod31_s8` | 93.1% | 78.9% | 65.0% | 54.3% | 39.5% |
| `main_mod31_s16` | 98.5% | 92.9% | 86.4% | 77.4% | 67.6% |
| `main_mod31_s31` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Main interpretation:

- Exact symbolic execution appears at the predicted capacity threshold:
  `S=p`.
- The degradation below `S=p` is not mysterious decoder error; it tracks
  support loss directly through `empty_slot_rate`.
- A half-size support memory, `S=16`, preserves much of the query mass but is
  still materially below exact on long programs.

Next action: run a larger modulus-97 scale check with `S=16,32,64,97`.

## 2026-06-21 Scale Check

Ran four modulus-97 variants with evaluation lengths 4, 8, 12, 16, and 24:

| Variant | Slot capacity | L=4 decoder-query | L=8 decoder-query | L=12 decoder-query | L=16 decoder-query | L=24 decoder-query | L=24 empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scale_mod97_s16` | 16 | 94.9% | 83.5% | 71.4% | 61.8% | 42.3% | 58.7% |
| `scale_mod97_s32` | 32 | 97.9% | 92.6% | 86.8% | 79.4% | 66.6% | 33.8% |
| `scale_mod97_s64` | 64 | 99.7% | 98.1% | 96.1% | 93.1% | 86.5% | 13.7% |
| `scale_mod97_s97` | 97 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% |

Scale interpretation:

- The `S=p` threshold replicated at a much larger pair space.
- Sub-capacity memories can remain highly useful when `S` is large, but exact
  belief execution still requires enough slots to cover the initial relation.
- At length 24, `S=64` preserves 86.5% query mass but still loses support in
  13.7% of examples; `S=97` removes that failure mode.

Analysis artifacts generated:

- `analysis/all_metrics_long.csv`
- `analysis/all_metrics_query_mean.csv`
- `analysis/first_k_ge_l_summary.csv`
- `analysis/summary.md`
- `analysis/figures/`

Next action: write the standalone report and HTML artifact, then audit the
directory layout and checkpoint manifest.

## 2026-06-21 Final Audit

Completed the standalone write-up and layout audit.

Created report artifacts:

- `reports/sparse_support_memory_paper.md`
- `reports/sparse_support_memory_paper.html`

Audit results:

- 14 run directories are present, and each has `metrics_final.csv` plus
  `results.json`.
- The HTML report references four generated figures, and all four image paths
  resolve.
- The experiment directory is lightweight, about 4 MB.
- No trainable checkpoints were produced by this deterministic sparse-support
  sweep; `large_artifacts/sparse_support_memory_executor/` is intentionally
  empty and `checkpoint_manifest.csv` contains only the header.
- The standalone report does not refer to any older or source experiment.

Final status: complete.
