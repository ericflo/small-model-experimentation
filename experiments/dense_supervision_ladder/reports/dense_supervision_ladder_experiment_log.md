# Dense Supervision Ladder Experiment Log

## Objective

Test which level of supervision is sufficient for a dense recurrent hidden state to learn sequential modular belief execution.

The model keeps a fixed-width dense vector. It does not store an explicit categorical table as its recurrent state. The experiment compares a ladder of increasingly informative executor training objectives:

1. `sampled_final`: one sampled final query label.
2. `soft_final_query`: exact final query distribution.
3. `prefix_query`: exact query distributions at every executed prefix.
4. `sparse_belief`: prefix query supervision plus exact belief distillation on a sampled subset of prefix states.
5. `full_belief`: prefix query supervision plus exact belief distillation on every executed prefix state.

Exact final query distributions and exact final belief distributions are used for evaluation. A frozen post-hoc belief probe is trained after executor training for all variants. Variants with belief distillation also have a trained belief decoder, which is evaluated separately from the post-hoc probe.

## Hypothesis

If the dense recurrent architecture can host the belief algorithm but the sampled final-answer objective is too weak, then stronger supervision should lift the same architecture substantially:

1. `sampled_final` should show the weakest K-threshold.
2. `soft_final_query` should improve final query calibration but may not produce a strong belief state.
3. `prefix_query` should improve recurrent execution because every internal step receives query-level pressure.
4. `sparse_belief` should show whether occasional exact state targets are enough to stabilize the dense representation.
5. `full_belief` should reveal whether the architecture can represent the belief state at all.

If `full_belief` fails, the dense recurrent substrate is likely the bottleneck. If `full_belief` succeeds but weaker variants fail, the objective is the bottleneck.

## Task

Initial belief:

```text
B = A + d (mod p), with A unknown
```

Programs contain modular arithmetic updates over `A` and `B`, plus observation filters of the form `A % m = r` or `B % m = r`. Observation residues are sampled from the current support so the target support is never empty.

Each example samples one final query type:

- `A`
- `B`
- `A+B mod p`
- `A-B mod p`

The supervision ladder changes what target information is used during training. Evaluation always measures exact final query support mass, exact final query top-1 support membership, post-hoc probe belief support mass, and trained belief-decoder support mass.

## Planned Sequence

1. Implement ladder losses and trained belief decoder.
2. Smoke test at tiny modulus.
3. Run a small-modulus pilot across all ladder levels.
4. Analyze the pilot and tune run budget if needed.
5. Run the scaled modulus-31 ladder.
6. Aggregate metrics, generate figures, and write a standalone report.

## Implementation Notes

Implemented:

- Dense recurrent executor with query head and trained belief decoder.
- Supervision ladder choices: `sampled_final`, `soft_final_query`, `prefix_query`, `sparse_belief`, `full_belief`.
- Exact query-distribution projection from pair distributions.
- Prefix query loss over executed prefix states.
- Sparse/full belief distillation losses over executed prefix states.
- Final evaluation rows with query metrics, post-hoc probe belief metrics, and trained decoder belief metrics.

## Smoke Test: Modulus 7 Ladder

Runs:

- `../runs/smoke_sampled_final_mod7`
- `../runs/smoke_soft_final_query_mod7`
- `../runs/smoke_prefix_query_mod7`
- `../runs/smoke_sparse_belief_mod7`
- `../runs/smoke_full_belief_mod7`

Command pattern:

```bash
python experiments/dense_supervision_ladder/src/dense_supervision_ladder_experiment.py --mode dense --supervision <level> --modulus 7 --observe_mod 3 --observe_prob 0.4 --train_max_len 3 --eval_lengths 2,3 --eval_k 0,1,2,3 --train_steps 2 --batch_size 16 --eval_batch_size 16 --eval_examples 16 --probe_steps 2 --probe_batch_size 16 --state_dim 32 --instr_dim 16 --log_every 1 --probe_log_every 1 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_supervision_ladder/runs/smoke_<level>_mod7 --checkpoint_dir large_artifacts/dense_supervision_ladder/checkpoints/smoke_<level>_mod7
```

Status: complete.

Result: all five supervision levels compile, train, probe, evaluate, write CSV/JSON outputs, and save checkpoints.

Bug found and fixed: the first prefix-query smoke showed `qloss=0.0000`. The projection helper used `out[mask].scatter_add_`, which mutates an indexed copy rather than the output tensor. The helper now scatters into a temporary tensor and assigns it back to `out[mask]`. After the fix, `prefix_query`, `sparse_belief`, and `full_belief` all report nonzero query losses.

Interpretation: the full ladder path is functional. The smoke runs are intentionally too short to test learning.

## Pilot: Modulus 11 Ladder

Runs:

- `../runs/pilot_sampled_final_mod11`
- `../runs/pilot_soft_final_query_mod11`
- `../runs/pilot_prefix_query_mod11`
- `../runs/pilot_sparse_belief_mod11`
- `../runs/pilot_full_belief_mod11`

Command pattern:

```bash
python experiments/dense_supervision_ladder/src/dense_supervision_ladder_experiment.py --mode dense --supervision <level> --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --train_steps 1200 --batch_size 256 --eval_batch_size 256 --eval_examples 256 --probe_steps 600 --probe_batch_size 256 --state_dim 256 --instr_dim 128 --log_every 200 --probe_log_every 100 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_supervision_ladder/runs/pilot_<level>_mod11 --checkpoint_dir large_artifacts/dense_supervision_ladder/checkpoints/pilot_<level>_mod11
```

Status: complete.

Aggregated final-query metrics at `K=L`, averaged across the four query types:

| supervision | L=3 qmass | L=6 qmass | L=9 qmass | L=12 qmass | L=12 probe belief | L=12 decoder belief |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_final` | 69.1% | 48.1% | 35.8% | 28.9% | 9.9% | 1.7% |
| `soft_final_query` | 81.7% | 60.5% | 46.8% | 39.2% | 20.1% | 1.7% |
| `prefix_query` | 90.2% | 70.9% | 56.9% | 45.3% | 26.4% | 1.7% |
| `sparse_belief` | 86.8% | 62.8% | 47.8% | 38.8% | 20.6% | 17.5% |
| `full_belief` | 90.9% | 71.3% | 57.7% | 45.7% | 30.2% | 28.9% |

Interpretation:

- `sampled_final` learns a weak recurrent threshold but leaves little recoverable belief information at longer lengths.
- `soft_final_query` improves both final query mass and post-hoc belief readability, showing that exact final distributions are a better objective than sampled labels.
- `prefix_query` produces a large gain, implying that per-step query supervision is a high-leverage training signal for this dense recurrent substrate.
- `sparse_belief` trains a nontrivial decoder, but the sparse distillation objective underperforms prefix-query on final query mass at this pilot budget.
- `full_belief` matches or slightly exceeds prefix-query on final query mass and also trains a useful belief decoder, making it the strongest pilot variant.

Decision for scaled run: run the complete five-level ladder at modulus 31. The scaled run should preserve the full ladder, because the pilot shows useful separation across all adjacent supervision levels. Use a moderate budget rather than a single long run so the final result can compare the whole ladder under one matched protocol.

## Main Run: Modulus 31 Ladder

Runs:

- `../runs/main_sampled_final_mod31`
- `../runs/main_soft_final_query_mod31`
- `../runs/main_prefix_query_mod31`
- `../runs/main_sparse_belief_mod31`
- `../runs/main_full_belief_mod31`

Command pattern:

```bash
python experiments/dense_supervision_ladder/src/dense_supervision_ladder_experiment.py --mode dense --supervision <level> --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --train_steps 2000 --batch_size 256 --eval_batch_size 256 --eval_examples 256 --probe_steps 800 --probe_batch_size 256 --state_dim 256 --instr_dim 128 --log_every 250 --probe_log_every 200 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_supervision_ladder/runs/main_<level>_mod31 --checkpoint_dir large_artifacts/dense_supervision_ladder/checkpoints/main_<level>_mod31
```

Status: complete.

Aggregated final-query metrics at `K=L`, averaged across the four query types:

| supervision | L=4 qmass | L=8 qmass | L=12 qmass | L=16 qmass | L=24 qmass | L=24 probe belief | L=24 decoder belief |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sampled_final` | 51.0% | 28.2% | 20.5% | 13.6% | 9.4% | 1.2% | 0.2% |
| `soft_final_query` | 62.0% | 39.4% | 30.2% | 22.2% | 16.1% | 3.2% | 0.2% |
| `prefix_query` | 72.8% | 46.3% | 34.1% | 25.8% | 18.5% | 4.3% | 0.2% |
| `sparse_belief` | 66.7% | 42.4% | 31.5% | 22.8% | 16.2% | 3.2% | 3.0% |
| `full_belief` | 74.1% | 49.7% | 35.8% | 27.3% | 19.4% | 4.8% | 4.6% |

Interpretation:

- The ladder separates cleanly at modulus 31. Stronger supervision increases query mass and belief-state readability under the same architecture and training budget.
- `sampled_final` is a weak objective for this task: it learns a K-dependent signal, but its long-chain accuracy and belief readability remain low.
- `soft_final_query` is a large improvement over sampled labels, so exact final distributions matter.
- `prefix_query` is the largest jump in query performance, showing that per-step query supervision is highly useful for training recurrent execution.
- `sparse_belief` trains a decoder above chance but underperforms `prefix_query` on query mass; sampled state distillation is not enough to justify its optimization cost in this configuration.
- `full_belief` is best at every evaluated length and trains a decoder whose belief mass tracks the post-hoc probe, so the dense recurrent state can represent useful belief information when the objective directly asks for it.

Decision: no extra extension run is needed for the primary claim. The matched five-level ladder already identifies the main bottleneck as supervision strength rather than the dense recurrent substrate alone. A longer `full_belief` run could improve absolute scores, but it would not replace the matched ladder as the central evidence.

## Analysis Artifacts

Generated:

- `../analysis/all_metrics_long.csv`
- `../analysis/final_metrics_query_mean.csv`
- `../analysis/mod11_ladder_threshold_summary.csv`
- `../analysis/mod31_ladder_threshold_summary.csv`
- `../analysis/mod31_ladder_per_query_threshold_summary.csv`
- `../analysis/summary.md`
- `../analysis/figures/mod31_ladder_query_mass_at_k_ge_l.png`
- `../analysis/figures/mod31_ladder_probe_belief_mass_at_k_ge_l.png`
- `../analysis/figures/mod31_ladder_decoder_belief_mass_at_k_ge_l.png`
- `../analysis/figures/mod31_full_belief_query_mass_heatmap.png`
- `../analysis/figures/mod31_full_belief_decoder_mass_heatmap.png`
- `../analysis/figures/mod31_query_mass_by_k_<supervision>.png`

Status: complete.
