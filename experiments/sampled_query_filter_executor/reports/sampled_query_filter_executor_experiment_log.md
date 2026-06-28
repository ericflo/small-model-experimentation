# Sampled Query Filter Executor Experiment Log

## Objective

Test whether a latent recurrent runtime can learn arithmetic and observation-filter execution when each training example provides only one sampled final query value, not the full final query distribution and not the full belief state.

The hidden state is evaluated against the exact final belief distribution over `(A, B)` pairs, but the training loss only observes a single sampled value for one query:

- `A`
- `B`
- `A+B mod p`
- `A-B mod p`

## Hypothesis

If sampled-value supervision is sufficient to induce a reusable latent executor, then:

1. Query performance should improve sharply when `K >= L`.
2. The threshold should generalize to held-out lengths longer than training.
3. Hidden belief mass should rise even though the model never sees full belief targets.
4. A marginal recurrent control should be weaker because it cannot represent pairwise correlations.
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

Each example samples one final query type and one answer value from the exact final query distribution. The model is trained with ordinary one-label cross-entropy on that sampled value. Evaluation still uses exact query distributions and exact hidden belief states.

## Models

### Joint Sampled-Query Filter

The primary model stores a categorical distribution over all `(A,B)` pairs. At each recurrent step, it applies the next learned arithmetic transition or learned observation likelihood, then normalizes the belief state. Training loss is computed only after projecting the final pair distribution into the sampled query distribution and scoring the sampled answer value.

### Marginal Sampled-Query Control

The marginal control stores separate distributions over `A` and `B`. It can learn some marginal answer signal but cannot exactly preserve pairwise correlations.

### Static Sampled-Query Compiler Control

The static control receives the initial relation and whole program, then predicts a final pair distribution in one pass. It is trained through the same sampled-value query loss.

## Planned Sequence

1. Smoke test at tiny modulus.
2. Pilot joint sampled-query filter at small modulus.
3. Matched marginal and static controls at small modulus.
4. Main scaled run at modulus 31.
5. Matched scaled controls.
6. Aggregate metrics, generate figures, and write a standalone report.

## Smoke Test

Run: `../runs/smoke_joint_mod7`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode joint --modulus 7 --observe_mod 3 --observe_prob 0.4 --train_max_len 3 --eval_lengths 2,3 --eval_k 0,1,2,3 --eval_query_types all --train_steps 2 --batch_size 16 --eval_batch_size 16 --eval_examples 32 --log_every 1 --eval_every 1 --lr 0.01 --output_dir experiments/sampled_query_filter_executor/runs/smoke_joint_mod7 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/smoke_joint_mod7
```

Status: complete.

Result: the script compiled, trained with sampled query labels, evaluated exact query and hidden-belief metrics, wrote metrics, and saved checkpoints under `../../../large_artifacts/sampled_query_filter_executor/checkpoints/smoke_joint_mod7`.

Interpretation: sampled-value targets, exact query evaluation, hidden-belief audit metrics, evaluation aggregation, and separated checkpoint writing are functional. The run is intentionally too short to test learning.

## Pilot 1: Joint Sampled-Query Filter, Modulus 11

Run: `../runs/pilot_joint_mod11`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode joint --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 1200 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 50 --eval_every 600 --lr 0.03 --output_dir experiments/sampled_query_filter_executor/runs/pilot_joint_mod11 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/pilot_joint_mod11
```

Status: complete.

Interim result at step 600:

- `L=3`: at `K=3`, all query types reached 100% top-1-on-support, 98.8-99.2% query target mass, and 97.6-97.8% hidden belief target mass.
- `L=6`: at `K=6`, all query types reached 100% top-1-on-support, 97.4-98.1% query target mass, and 96.0-96.2% hidden belief target mass.
- `L=9`: at `K=9`, all query types reached 100% top-1-on-support, 96.1-97.2% query target mass, and 94.9-95.0% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 95.4-96.5% query target mass, and 94.3-94.4% hidden belief target mass.

For `K<L`, query target mass and hidden belief mass were much lower. This interim result shows that one sampled answer value per example is enough to induce a coherent latent belief state at small modulus.

Final result at step 1200:

- `L=3`: at `K=3`, all query types reached 100% top-1-on-support, 99.6-99.7% query target mass, and 99.3% hidden belief target mass.
- `L=6`: at `K=6`, all query types reached 100% top-1-on-support, 99.2-99.4% query target mass, and 98.7-98.9% hidden belief target mass.
- `L=9`: at `K=9`, all query types reached 100% top-1-on-support, 98.8-99.1% query target mass, and 98.4-98.5% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 98.6-99.0% query target mass, and 98.2-98.3% hidden belief target mass.

Interpretation: sampled-value supervision is sufficient to train the joint recurrent executor at small modulus, including held-out lengths.

## Control 1: Marginal Sampled-Query Filter, Modulus 11

Run: `../runs/control_marginal_mod11`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode marginal --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 1200 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1200 --lr 0.03 --output_dir experiments/sampled_query_filter_executor/runs/control_marginal_mod11 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/control_marginal_mod11
```

Status: complete.

Result at step 1200:

- `L=3`: at sufficient `K`, query target mass reached 43.6-66.2%; hidden belief mass was 10.3-11.5%.
- `L=6`: at sufficient `K`, query target mass reached 35.3-52.0%; hidden belief mass was 11.2-11.9%.
- `L=9`: at sufficient `K`, query target mass reached 28.1-44.2%; hidden belief mass was 10.2-11.1%.
- `L=12`: at sufficient `K`, query target mass reached 23.0-37.6%; hidden belief mass was 9.3-10.5%.

Interpretation: the marginal model learns some sampled-answer signal but does not recover a coherent joint belief state.

## Control 2: Static Sampled-Query Compiler, Modulus 11

Run: `../runs/control_static_mod11`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode static --modulus 11 --observe_mod 4 --observe_prob 0.3 --train_max_len 6 --eval_lengths 3,6,9,12 --eval_k 0,1,2,3,6,9,12 --eval_query_types all --train_steps 1200 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1200 --lr 0.001 --dim 128 --heads 4 --compiler_layers 2 --output_dir experiments/sampled_query_filter_executor/runs/control_static_mod11 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/control_static_mod11
```

Status: complete.

Result at step 1200:

- `L=3`: query target mass reached 48.0-66.6%; hidden belief mass was 15.1-17.2%.
- `L=6`: query target mass reached 32.7-45.3%; hidden belief mass was 8.2-9.4%.
- `L=9`: query target mass reached 21.2-28.1%; hidden belief mass was 3.1-3.8%.
- `L=12`: query target mass reached 16.4-18.1%; hidden belief mass was 2.0-2.4%.

Interpretation: the static compiler learns short-length signal but does not length-generalize and does not recover a coherent hidden belief state.

## Main Run: Joint Sampled-Query Filter, Modulus 31

Run: `../runs/main_joint_mod31`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode joint --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1500 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 50 --eval_every 750 --lr 0.03 --output_dir experiments/sampled_query_filter_executor/runs/main_joint_mod31 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/main_joint_mod31
```

Status: complete.

Interim result at step 750:

- `L=4`: at `K=4`, all query types reached 100% top-1-on-support, 95.0-97.2% query target mass, and 91.2-91.3% hidden belief target mass.
- `L=8`: at `K=8`, all query types reached 100% top-1-on-support, 89.7-92.6% query target mass, and 84.6-85.4% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 85.1-89.2% query target mass, and 80.9-81.4% hidden belief target mass.
- `L=16`: at `K=16`, all query types reached 100% top-1-on-support, 82.1-85.8% query target mass, and 78.3-79.0% hidden belief target mass.
- `L=24`: at `K=24`, all query types reached 100% top-1-on-support, 78.9-83.8% query target mass, and 75.5-76.9% hidden belief target mass.

Final result at step 1500:

- `L=4`: at `K=4`, all query types reached 100% top-1-on-support, 98.8-99.3% query target mass, and 97.8-97.9% hidden belief target mass.
- `L=8`: at `K=8`, all query types reached 100% top-1-on-support, 97.3-98.2% query target mass, and 96.1-96.3% hidden belief target mass.
- `L=12`: at `K=12`, all query types reached 100% top-1-on-support, 96.2-97.1% query target mass, and 95.0-95.1% hidden belief target mass.
- `L=16`: at `K=16`, all query types reached 100% top-1-on-support, 95.2-96.4% query target mass, and 94.2-94.5% hidden belief target mass.
- `L=24`: at `K=24`, all query types reached 100% top-1-on-support, 94.7-95.6% query target mass, and 93.6-93.8% hidden belief target mass.

For `K<L`, the same examples remained close to baseline. At `L=24`, the best pre-threshold query target mass stayed below 10% for all four query types, while `K=24` recovered the exact sampled-query answer support and most of the full hidden belief mass.

## Scaled Control: Marginal Sampled-Query Filter, Modulus 31

Run: `../runs/control_marginal_mod31`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode marginal --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1500 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1500 --lr 0.03 --output_dir experiments/sampled_query_filter_executor/runs/control_marginal_mod31 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/control_marginal_mod31
```

Status: complete.

Result at step 1500:

- `L=4`: at sufficient `K`, query target mass reached 36.2-56.2%; hidden belief mass was 3.8-4.3%.
- `L=8`: at sufficient `K`, query target mass reached 23.9-36.8%; hidden belief mass was 3.3-3.5%.
- `L=12`: at sufficient `K`, query target mass reached 16.4-29.0%; hidden belief mass was 2.8-3.0%.
- `L=16`: at sufficient `K`, query target mass reached 11.7-21.5%; hidden belief mass was 2.2-2.6%.
- `L=24`: at sufficient `K`, query target mass reached 8.5-15.0%; hidden belief mass was 1.6%.

Interpretation: the marginal recurrent control learns some sampled-query signal but does not recover the joint hidden state and does not approach the joint recurrent model on held-out lengths.

## Scaled Control: Static Sampled-Query Compiler, Modulus 31

Run: `../runs/control_static_mod31`

Command:

```bash
python experiments/sampled_query_filter_executor/src/sampled_query_filter_executor_experiment.py --mode static --modulus 31 --observe_mod 5 --observe_prob 0.3 --train_max_len 8 --eval_lengths 4,8,12,16,24 --eval_k 0,1,2,4,8,12,16,24 --eval_query_types all --train_steps 1500 --batch_size 512 --eval_batch_size 512 --eval_examples 512 --log_every 100 --eval_every 1500 --lr 0.001 --dim 128 --heads 4 --compiler_layers 2 --output_dir experiments/sampled_query_filter_executor/runs/control_static_mod31 --checkpoint_dir large_artifacts/sampled_query_filter_executor/checkpoints/control_static_mod31
```

Status: complete.

Result at step 1500:

- `L=4`: query target mass reached 36.7-54.0%; hidden belief mass was 4.8-5.9%.
- `L=8`: query target mass reached 20.8-29.7%; hidden belief mass was 2.1-2.3%.
- `L=12`: query target mass reached 11.5-15.0%; hidden belief mass was 0.6-0.7%.
- `L=16`: query target mass reached 7.6-9.2%; hidden belief mass was 0.3-0.4%.
- `L=24`: query target mass reached 5.0-5.4%; hidden belief mass was 0.2%.

Interpretation: the static compiler learns some short-length sampled-query signal but does not length-generalize and does not recover the hidden joint belief state.
