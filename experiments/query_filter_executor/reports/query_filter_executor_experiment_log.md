# Query Filter Executor Experiment Log

## Objective

Test whether a latent recurrent runtime can learn to execute arithmetic and observation-filter programs when it is trained only from final query distributions, not from dense belief-state supervision.

The hidden state is still evaluated as a full belief distribution over `(A, B)` pairs, but the training loss only asks the model to answer one query about the final state:

- `A`
- `B`
- `A+B mod p`
- `A-B mod p`

## Hypothesis

If final-query supervision is sufficient to induce a reusable latent executor, then:

1. Query accuracy and query target mass should be low when `K < L` and high when `K >= L`.
2. The threshold should generalize to held-out lengths longer than training.
3. The latent belief state should become coherent even though it is not directly supervised.
4. A marginal recurrent control should be weaker, especially on relational queries.
5. A static one-shot compiler should not reproduce the same length-generalizing K threshold.

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

Each example samples one final query type. The query target is the exact distribution of the queried value under the final belief state.

## Models

### Joint Query Filter

The primary model stores a categorical distribution over all `(A,B)` pairs. At each recurrent step, it applies the next learned arithmetic transition or learned observation likelihood, then normalizes the belief state. Training loss is computed only after projecting the final pair distribution into the sampled query distribution.

### Marginal Query Filter Control

The marginal control stores separate distributions over `A` and `B`. It can answer some marginal queries but cannot exactly preserve pairwise correlations.

### Static Query Compiler Control

The static control receives the initial relation and whole program, then predicts a final pair distribution in one pass. It is trained through the same final query loss.

## Planned Sequence

1. Smoke test at tiny modulus.
2. Pilot joint query filter at small modulus.
3. Matched marginal and static controls at small modulus.
4. Main scaled run at modulus 31.
5. Matched scaled controls.
6. Aggregate metrics, generate figures, and write a standalone report.

## Smoke Test

Run: `../runs/smoke_joint_mod7`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode joint --modulus 7 --observe_mod 3 --observe_prob 0.4 --train_max_len 3 --eval_lengths 2,3 --eval_k 0,1,2,3 --eval_query_types all --train_steps 2 --batch_size 16 --eval_batch_size 16 --eval_examples 32 --log_every 1 --eval_every 1 --lr 0.01 --output_dir experiments/query_filter_executor/runs/smoke_joint_mod7 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/smoke_joint_mod7
```

Result: the script compiled, trained, evaluated all query types, wrote metrics, and saved checkpoints under `../../../large_artifacts/query_filter_executor/checkpoints/smoke_joint_mod7`.

Interpretation: query targets, query projection, hidden-belief audit metrics, evaluation aggregation, and separated checkpoint writing are functional. The run is intentionally too short to test learning.

## Pilot 1: Joint Query Filter, Modulus 11

Run: `../runs/pilot_joint_mod11`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode joint --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 800 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 50 --eval_every 400 --lr 0.03 --output_dir experiments/query_filter_executor/runs/pilot_joint_mod11 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/pilot_joint_mod11
```

Result at step 800:

- `L=3`: at `K=3`, all query types reached 100% top-1-on-support, 99.6-99.7% query target mass, and 99.2% hidden belief target mass.
- `L=6`: at `K=6`, all query types reached 100% top-1-on-support, 99.1-99.4% query target mass, and 98.6-98.7% hidden belief target mass.
- `L=9`: at `K=9`, all query types reached 100% top-1-on-support, 98.7-99.1% query target mass, and 98.3% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 98.5-98.9% query target mass, and 98.0-98.1% hidden belief target mass.

For `K<L`, query target mass stayed far lower and hidden belief mass stayed low. This is a positive result: final-query supervision induced a coherent latent belief state.

## Control 1: Marginal Query Filter, Modulus 11

Run: `../runs/control_marginal_mod11`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode marginal --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 800 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 800 --lr 0.03 --output_dir experiments/query_filter_executor/runs/control_marginal_mod11 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/control_marginal_mod11
```

Result at step 800:

- `L=3`: at `K=3`, query target mass reached 47.3-66.0%; hidden belief mass was 10.5-11.3%.
- `L=6`: at `K=6`, query target mass reached 35.6-52.0%; hidden belief mass was 11.2-11.5%.
- `L=9`: at `K=9`, query target mass reached 28.2-43.3%; hidden belief mass was 10.1-11.3%.
- `L=12`: at `K=12`, query target mass reached 23.6-38.3%; hidden belief mass was 9.3-10.7%.

Interpretation: the marginal model learns some query signal but does not recover a coherent joint belief state.

## Control 2: Static Query Compiler, Modulus 11

Run: `../runs/control_static_mod11`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode static --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 800 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 800 --lr 0.001 --dim 128 --heads 4 --compiler_layers 2 --output_dir experiments/query_filter_executor/runs/control_static_mod11 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/control_static_mod11
```

Result at step 800:

- `L=3`: query target mass reached 52.6-65.9%; hidden belief mass was 16.0-18.5%.
- `L=6`: query target mass reached 33.3-45.5%; hidden belief mass was 8.5-9.3%.
- `L=9`: query target mass reached 22.0-28.6%; hidden belief mass was 3.5-3.7%.
- `L=12`: query target mass reached 17.2-19.1%; hidden belief mass was 2.1-2.5%.

Interpretation: the static compiler learns short-length signal but does not length-generalize and does not recover a coherent final belief state.

## Main Run: Joint Query Filter, Modulus 31

Run: `../runs/main_joint_mod31`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode joint --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 50 --eval_every 500 --lr 0.03 --output_dir experiments/query_filter_executor/runs/main_joint_mod31 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/main_joint_mod31
```

Status: complete.

Interim result at step 500:

- `L=4`: at `K=4`, all query types reached 100% top-1-on-support, 96.4-97.9% query target mass, and 93.5-93.6% hidden belief target mass.
- `L=8`: at `K=8`, all query types reached 100% top-1-on-support, 92.3-94.6% query target mass, and 88.8-89.0% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 88.7-91.4% query target mass, and 85.1-85.9% hidden belief target mass.
- `L=16`: at `K=16`, all query types reached 100% top-1-on-support, 86.5-89.2% query target mass, and 83.2-83.9% hidden belief target mass.
- `L=24`: at `K=24`, all query types reached 100% top-1-on-support, 83.8-86.9% query target mass, and 81.1-81.8% hidden belief target mass.

For `K<L`, query and hidden-belief mass stayed much lower. This interim result already shows length-generalizing recurrent execution under query-only supervision.

Final result at step 1000:

- `L=4`: at `K=4`, all query types reached 100% top-1-on-support, 99.1-99.4% query target mass, and 98.3% hidden belief target mass.
- `L=8`: at `K=8`, all query types reached 100% top-1-on-support, 97.9-98.5% query target mass, and 97.0% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 96.8-97.8% query target mass, and 96.0-96.2% hidden belief target mass.
- `L=16`: at `K=16`, all query types reached 100% top-1-on-support, 96.4-97.1% query target mass, and 95.4-95.6% hidden belief target mass.
- `L=24`: at `K=24`, all query types reached 100% top-1-on-support, 95.7-96.6% query target mass, and 94.9-95.1% hidden belief target mass.

For `K<L`, the final run preserved the sharp threshold: long programs remained near baseline until the recurrent budget reached the program length.

## Scaled Control: Marginal Query Filter, Modulus 31

Run: `../runs/control_marginal_mod31`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode marginal --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1000 --lr 0.03 --output_dir experiments/query_filter_executor/runs/control_marginal_mod31 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/control_marginal_mod31
```

Status: complete.

Result at step 1000:

- `L=4`: at `K=4`, query target mass reached 37.0-57.7%; hidden belief mass was 4.0-4.5%.
- `L=8`: at `K=8`, query target mass reached 26.2-39.4%; hidden belief mass was 3.7-4.1%.
- `L=12`: at `K=12`, query target mass reached 17.0-27.9%; hidden belief mass was 2.9-3.2%.
- `L=16`: at `K=16`, query target mass reached 12.5-21.1%; hidden belief mass was 2.3-2.5%.
- `L=24`: at `K=24`, query target mass reached 8.5-15.7%; hidden belief mass was 1.6-2.1%.

Interpretation: the marginal control answers some easy marginal signal but does not recover the joint state and does not approach the joint recurrent model on held-out lengths.

## Scaled Control: Static Query Compiler, Modulus 31

Run: `../runs/control_static_mod31`

Command:

```bash
python experiments/query_filter_executor/src/query_filter_executor_experiment.py --mode static --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1000 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1000 --lr 0.001 --dim 128 --heads 4 --compiler_layers 2 --output_dir experiments/query_filter_executor/runs/control_static_mod31 --checkpoint_dir large_artifacts/query_filter_executor/checkpoints/control_static_mod31
```

Status: complete.

Result at step 1000:

- `L=4`: query target mass reached 38.1-53.9%; hidden belief mass was 6.0-6.7%.
- `L=8`: query target mass reached 23.2-32.9%; hidden belief mass was 2.4-2.9%.
- `L=12`: query target mass reached 11.7-14.7%; hidden belief mass was 0.7-0.8%.
- `L=16`: query target mass reached 8.1-9.1%; hidden belief mass was 0.3-0.4%.
- `L=24`: query target mass reached 4.9-5.4%; hidden belief mass was 0.2%.

Interpretation: the static compiler learns some short-length signal but collapses on longer programs and does not recover the latent belief state.
