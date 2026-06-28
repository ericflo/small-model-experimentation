# Dense Latent Query Executor Experiment Log

## Objective

Test whether a latent recurrent runtime with a fixed-width dense hidden state can learn arithmetic and observation-filter execution when each training example provides only one sampled final query value.

The model is not given an explicit categorical belief vector over `(A, B)` pairs as its state. It keeps a dense vector, executes one instruction per recurrent step, and predicts a final query value. Exact final query distributions and exact final belief distributions are used for evaluation. A frozen post-hoc belief probe is trained after executor training to measure how much exact belief-state information is recoverable from the dense hidden state.

## Hypothesis

If sampled-answer supervision can induce a reusable dense latent executor, then:

1. Direct query performance should improve sharply when `K >= L`.
2. The threshold should generalize to held-out lengths longer than training.
3. A probe trained only on training-length hidden states should recover nontrivial exact belief information on held-out lengths.
4. A static dense compiler should not reproduce the same length-generalizing K threshold.
5. An order-destroying recurrent control should be weaker, showing that the dense state is using sequential execution rather than only bag-of-instruction statistics.

## Task

Initial belief:

```text
B = A + d (mod p), with A unknown
```

Programs contain:

- arithmetic updates over `A` and `B`
- observation filters of the form `A % m = r`
- observation filters of the form `B % m = r`

Observation residues are sampled from the current support, so the target support is never empty.

Each example samples one final query type and one answer value from the exact final query distribution:

- `A`
- `B`
- `A+B mod p`
- `A-B mod p`

The executor is trained with one-label cross-entropy on the sampled answer. Probe training happens only after the executor is frozen.

## Models

### Dense Recurrent Executor

The primary model encodes the initial relation in a dense vector and updates that vector with a GRU-style recurrent cell, one program instruction per internal step. The query head maps the dense state to logits over the query value for each query type.

### Dense Static Compiler Control

The static control receives the whole program and initial relation, pools a Transformer encoder representation, and predicts query logits in one pass. It has no recurrent execution axis.

### Shuffled Dense Recurrent Control

The shuffled control uses the same dense recurrent architecture but consumes a deterministic sorted version of the instructions rather than the original sequence. It preserves some instruction content while destroying program order.

### Frozen Belief Probe

After executor training, the recurrent model is frozen. A separate MLP probe maps dense hidden states to distributions over `(A,B)` pairs. The probe is trained on hidden states from training-length prefixes and evaluated on held-out lengths and K values.

## Planned Sequence

1. Smoke test at tiny modulus.
2. Pilot dense recurrent run at small modulus.
3. Probe the pilot model.
4. Run matched static and shuffled controls at small modulus.
5. Run the main dense recurrent experiment at modulus 31.
6. Run matched scaled controls.
7. Aggregate metrics, generate figures, and write a standalone report.

## Smoke Test

Run: `../runs/smoke_dense_mod7`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode dense --modulus 7 --observe_mod 3 --observe_prob 0.4 --train_max_len 3 --eval_lengths 2,3 --eval_k 0,1,2,3 --train_steps 2 --batch_size 16 --eval_batch_size 16 --eval_examples 32 --probe_steps 2 --probe_batch_size 16 --state_dim 64 --instr_dim 32 --log_every 1 --probe_log_every 1 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/smoke_dense_mod7 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/smoke_dense_mod7
```

Status: complete.

Result: the script compiled, trained the dense executor from sampled query labels, froze it, trained a belief probe, evaluated exact query and probe-belief metrics, wrote metrics, and saved the checkpoint under `../../../large_artifacts/dense_latent_query_executor/checkpoints/smoke_dense_mod7`.

Interpretation: the full dense-state training/probing/evaluation path is functional. The run is intentionally too short to test learning.

## Pilot 1: Dense Recurrent Executor, Modulus 11

Run: `../runs/pilot_dense_mod11`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode dense --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --train_steps 3000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --probe_steps 1500 --probe_batch_size 512 --state_dim 256 --instr_dim 128 --log_every 200 --probe_log_every 150 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/pilot_dense_mod11 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/pilot_dense_mod11
```

Status: complete.

Result at final evaluation:

- `L=3`: at `K=3`, query target mass reached 83.6-91.5%; probe belief target mass reached 79.6-83.9%.
- `L=6`: at `K=6`, query target mass reached 62.1-77.4%; probe belief target mass reached 56.6-61.3%.
- `L=9`: at `K=9`, query target mass reached 46.7-60.5%; probe belief target mass reached 37.9-40.3%.
- `L=12`: at `K=12`, query target mass reached 40.8-53.4%; probe belief target mass reached 30.6-32.8%.

Interpretation: the dense recurrent model learned a real recurrent execution signal from sampled query labels. The K threshold is visible, and a frozen probe can recover substantial belief information from the dense hidden state. However, the result is not near-exact, and held-out lengths degrade sharply. This is a meaningful positive pilot but not a saturated architecture.

## Control 1: Dense Static Compiler, Modulus 11

Run: `../runs/control_static_mod11`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode static --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --train_steps 3000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --probe_steps 1500 --probe_batch_size 512 --state_dim 256 --instr_dim 128 --heads 4 --compiler_layers 2 --log_every 200 --probe_log_every 150 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/control_static_mod11 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/control_static_mod11
```

Status: complete.

Result at final evaluation:

- `L=3`: query target mass reached 69.3-78.8%; probe belief target mass reached 50.3-57.5%.
- `L=6`: query target mass reached 39.3-52.7%; probe belief target mass reached 23.6-25.6%.
- `L=9`: query target mass reached 22.9-27.7%; probe belief target mass reached 4.1-5.1%.
- `L=12`: query target mass reached 17.3-21.1%; probe belief target mass reached 2.2-2.8%.

Interpretation: the static compiler learns some short-program sampled-query signal but does not preserve the recurrent model's held-out length behavior. Its probed belief state collapses on lengths beyond training.

## Control 2: Shuffled Dense Recurrent Executor, Modulus 11

Run: `../runs/control_shuffled_mod11`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode shuffled --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --train_steps 3000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --probe_steps 1500 --probe_batch_size 512 --state_dim 256 --instr_dim 128 --log_every 200 --probe_log_every 150 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/control_shuffled_mod11 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/control_shuffled_mod11
```

Status: complete.

Result at final evaluation:

- `L=3`: at sufficient `K`, query target mass reached 71.8-84.2%; probe belief target mass reached 46.9-50.3%.
- `L=6`: at sufficient `K`, query target mass reached 39.5-45.7%; probe belief target mass reached 16.7-17.6%.
- `L=9`: at sufficient `K`, query target mass reached 24.0-28.2%; probe belief target mass reached 5.4-6.1%.
- `L=12`: at sufficient `K`, query target mass reached 18.7-20.9%; probe belief target mass reached 2.7-3.3%.

Interpretation: preserving instruction content while destroying order removes most of the dense recurrent model's held-out behavior. This supports the interpretation that the ordered dense model is learning a sequential update, not only a bag-of-instructions heuristic.

## Main Run: Dense Recurrent Executor, Modulus 31

Run: `../runs/main_dense_mod31`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode dense --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --train_steps 5000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --probe_steps 2000 --probe_batch_size 512 --state_dim 256 --instr_dim 128 --log_every 250 --probe_log_every 200 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/main_dense_mod31 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/main_dense_mod31
```

Status: complete.

Result at final evaluation:

- `L=4`: at `K=4`, query target mass averaged 70.2%; query top-1 on support averaged 79.1%; probe belief target mass averaged 41.6%.
- `L=8`: at `K=8`, query target mass averaged 49.3%; query top-1 on support averaged 59.3%; probe belief target mass averaged 18.8%.
- `L=12`: at `K=12`, query target mass averaged 36.1%; query top-1 on support averaged 44.6%; probe belief target mass averaged 12.0%.
- `L=16`: at `K=16`, query target mass averaged 29.6%; query top-1 on support averaged 37.1%; probe belief target mass averaged 8.8%.
- `L=24`: at `K=24`, query target mass averaged 21.5%; query top-1 on support averaged 28.8%; probe belief target mass averaged 5.7%.

Best query mass before `K>=L` was 44.9%, 21.8%, 14.9%, 10.8%, and 6.0% for lengths 4, 8, 12, 16, and 24. Best probe belief mass before `K>=L` was 11.3%, 2.5%, 1.4%, 0.8%, and 0.3%.

Interpretation: the dense recurrent model shows a real execution-threshold signature, but it is weak at the scaled modulus. The state contains recoverable belief information and improves sharply when `K` reaches `L`, yet the dense hidden state does not form a near-exact belief executor under sampled-query supervision alone. This is a partial positive result with a clear scaling limitation.

## Scaled Control: Dense Static Compiler, Modulus 31

Run: `../runs/control_static_mod31`

Command:

```bash
python experiments/dense_latent_query_executor/src/dense_latent_query_executor_experiment.py --mode static --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --train_steps 5000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --probe_steps 2000 --probe_batch_size 512 --state_dim 256 --instr_dim 128 --heads 4 --compiler_layers 2 --log_every 250 --probe_log_every 200 --lr 0.001 --probe_lr 0.001 --output_dir experiments/dense_latent_query_executor/runs/control_static_mod31 --checkpoint_dir large_artifacts/dense_latent_query_executor/checkpoints/control_static_mod31
```

Status: complete.

Result at final evaluation:

- `L=4`: static query target mass averaged 50.4%; query top-1 on support averaged 54.8%; probe belief target mass averaged 13.0%.
- `L=8`: static query target mass averaged 26.9%; query top-1 on support averaged 29.1%; probe belief target mass averaged 3.7%.
- `L=12`: static query target mass averaged 13.6%; query top-1 on support averaged 11.8%; probe belief target mass averaged 0.8%.
- `L=16`: static query target mass averaged 9.0%; query top-1 on support averaged 9.5%; probe belief target mass averaged 0.4%.
- `L=24`: static query target mass averaged 5.5%; query top-1 on support averaged 5.6%; probe belief target mass averaged 0.2%.

Interpretation: the one-pass static compiler learns some short-program query signal but does not match the dense recurrent model's length-generalizing threshold. The gap is especially clear in the probe-belief metric: even where static query mass is nontrivial, recoverable belief mass remains low and collapses on held-out lengths.

## Final Analysis Pass

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/dense_latent_query_executor/src/analyze_dense_latent_query_executor.py
```

Status: complete.

Generated:

- `../analysis/all_metrics_long.csv`
- `../analysis/final_metrics_long.csv`
- `../analysis/final_metrics_query_mean.csv`
- `../analysis/mod31_threshold_summary.csv`
- `../analysis/mod31_per_query_threshold_summary.csv`
- `../analysis/mod31_control_summary.csv`
- `../analysis/summary.md`
- `../analysis/figures/`
