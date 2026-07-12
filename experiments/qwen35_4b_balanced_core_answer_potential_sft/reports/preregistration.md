# Preregistration: Balanced-Core Answer-Potential SFT

## Freeze And Evidence Boundary

This protocol is frozen after the parent experiment's calibration and 331-task runtime evidence but before:

- importing those shards into this experiment's external index;
- generating the remaining 29 tasks;
- scoring any training trace;
- sampling any training-pool R1 answer rollout;
- constructing or training any SFT arm; or
- running any sealed evaluation task through base or an adapter.

The parent calibration values are declared design inputs, never new confirmatory outcomes. This experiment's
claim begins at the prospectively frozen SFT comparison.

## Claim Under Test

Within a balanced pool of complete self-generated thoughts, canonical-answer potential identifies SFT
targets that install more fresh task capability than length-matched random thoughts, R1-successful thoughts,
the shortest thoughts, or task-shuffled potential-selected thoughts.

Answer potential is the original treatment. Joint close-boundary-plus-answer potential is a co-primary
deployment-seam treatment. Both must be reported; neither can substitute for the other after outcomes.

## Population

Training is exactly 360 tasks: `caravan`, `foundry_ledger`, and `runeward`, levels 1--3, 40 tasks per cell.
These are complete blocks already first in the parent's frozen train ordering. The choice follows saved
runtime position, not correctness or score.

Each task receives exactly N=64 independent samples before any mechanical top-up. Planned main-pool size is
23,040. Pivot branches are prohibited. The 331 inherited task shards are accepted only if the source index
digest, expected counts, and every per-shard receipt pass. Imported raw shards are not rewritten.

## Termination And Eligibility

Sampling remains temperature 1.0, top-p 0.95, top-k 20 on the pinned vLLM backend. The model has 12,288
tokens to close naturally; a non-loop contact receives one exact-prefix continuation of at most 2,048.
An eligible trace:

1. emitted the model's own `</think>`;
2. is not an exact periodic loop;
3. has finite captured trace prior; and
4. produces an exact SFT record of at most 16,000 tokens.

Incomplete traces are never force-closed into the bank. A task with fewer than two eligible N=64 traces
receives up to four N=16 top-up batches from registered disjoint seeds. Remaining-deficient tasks are excluded
from every trace arm symmetrically.

## Scoring Instrument

The vLLM targeted readout scores the canonical boundary and answer token by token at the exact
teacher-forced prefix. No token is forced or vocabulary-masked at the sampled position; only unused rank
metadata is bypassed. It returns HF-compatible answer, boundary, and joint log-probability fields.

Before bulk train scoring, 32 fixed natural calibration traces are scored by both vLLM and Transformers bf16.
For trace and empty conditions, all of the following maximum absolute mean-token discrepancies must be at
most 0.15 nats/token:

- canonical-answer likelihood;
- close-boundary-plus-answer likelihood;
- answer gain; and
- joint gain.

Failure is an instrument block. Repair requires a new code commit and rerunning parity before bulk scoring;
the threshold cannot be relaxed. The calibration format-variant pass (tau-b 0.841) licenses canonical-only
train scoring but is not reused as an outcome label.

## Selection

All scores and R1 rollouts are computed for every eligible independent trace before selection. Within task:

- `answer_potential`: best answer gain, then maximum normalized-trigram distance among top-12 candidates
  within 0.25 nats/answer-token of best;
- `joint_potential`: the same rule using joint gain;
- `random_natural`: without replacement, closest in length to each answer-treatment trace, with stable seeded
  tie breaks;
- `success_rft`: the same length matching restricted to traces with one correct fresh answer rollout;
- `shortest_natural`: the two smallest eligible full traces, stable trace-ID ties; and
- `potential_shuffle`: one-to-one reassignment of the exact answer-treatment trace multiset to different
  tasks within family/level while minimizing total length mismatch.

Every trace remains complete. There is no prefix pruning or after-the-fact compression.

## Training

The six seed-42 arms in config are mandatory. Each target task contributes two rows. The success arm may have
fewer unique rows and is deterministically oversampled to match row count; its unique task/row coverage must
be reported. All other arms must match target row count exactly.

QLoRA is rank 32/alpha 64/dropout 0.05 for two epochs at learning rate 2e-4, batch 1, accumulation 16, and
max length 16,000. Prompt loss is 0, full thought loss 0.5, and close/answer loss 1.0. Any overflow is a hard
failure, not truncation. Record rows, optimizer steps, thought/supervised/forward tokens, elapsed time, peak
memory, package lock digest, and adapter hashes.

Each adapter is merged into the full Qwen3.5 composite. A deterministic same-prompt base/base repeat must be
identical and every merged arm must differ from base on at least one token sequence before scientific eval.
Runtime LoRA is prohibited.

## Mandatory Stage A

Evaluate base and all six merged seed-42 arms greedily with natural thinking at 12,288 tokens on:

- core IID: 180 tasks (three core families x levels 1--3 x 20);
- core hard: 60 tasks (three core families x level 4 x 20); and
- held family: 60 tasks (brinework/spindle x levels 1--3 x first 10 by frozen source order).

Core-IID exact-answer accuracy is primary. Other mandatory metrics: paired 10,000-resample intervals,
family macro, parse, natural close, parse-conditional accuracy, thought-length distribution, and actual
prompt-plus-sampled tokens.

For each treatment in `{answer_potential, joint_potential}`, select the strongest core-IID baseline among
`{random_natural, success_rft, shortest_natural}` only after all are evaluated. The Stage-B trigger requires:

```text
treatment - strongest_baseline >= 0.03
paired bootstrap 95% lower bound > 0
treatment - potential_shuffle > 0
parse delta >= -0.02
family-macro delta >= -0.02
```

Before applying an absolute gate, write a reachability receipt from its hard [0,1] range and observed
baseline. An unreachable gate fails closed and cannot be lowered in place.

## Conditional Stage B

If no treatment triggers, stop. If one or both trigger:

1. evaluate all arms greedily on full inherited IID540, hard180, held-family180, and rendering80 splits;
2. run k=8 on core IID for base, each triggered treatment, its strongest baseline, shortest-natural, and
   potential-shuffle (deduplicated union);
3. compare base pass@{1,2,4,8} using actual forward tokens; and
4. train seed 43 for each triggered treatment and its strongest baseline, then evaluate core IID180.

The full-result positive retains the Stage-A thresholds on the full IID comparison and no >0.02 held-family
or rendering regression. Replication requires a same-sign seed-43 delta and pooled paired lower bound >0.
A mission positive additionally beats a base sample-more point at no more actual inference forward tokens.

## Scope And Verdicts

The registered verdict labels are in the README. Calibration cannot produce a positive. Stage A can support
only a three-family core verdict. No result may be presented as the unfinished parent's nine-family/pivot
test, and no compression claim is allowed from shortest winning alone.

## Amendments

Only pre-outcome operational amendments may be appended below. Candidate scores, rollout labels, SFT
outcomes, or evaluation outputs must not be observed before such an amendment is committed.

### 2026-07-12 — Retire Batch-Sensitive vLLM Train Scoring

The registered 32-row cross-backend gate ran after the 360-task raw harvest and before any training-pool
score, R1 rollout, selection, SFT, or evaluation. It failed closed: maximum registered discrepancy was
0.692447 nats/token-equivalent versus the unchanged 0.15 maximum. The failed receipt is preserved at
`runs/scorer_parity_joint_32.json`.

Diagnosis was limited to the instrument receipt. Answer gain remained within 0.147865 and joint likelihood
within 0.054477 mean nats/target-token, but task-diverse long-prefix batching exposed Qwen3.5's known
batch-sensitive logits. Joint-gain discrepancy reached 0.692447 after boundary-token differences accumulated
and the parent metric normalized by a one-token canonical answer; an empty-answer condition also reached
0.156281. The threshold is not relaxed, the rows are not replaced, and the failed candidate backend is not
used for any bulk score.

Operational amendment: all training-pool canonical answer and joint scores use the existing Transformers
bf16 SDPA single-context reference scorer, one trace per forward, uniformly for every task and arm. This is
the reference side of the failed gate and the scorer already used for the parent's complete calibration.
Because there is no backend comparison after this change, no cross-backend parity claim is made. The
canonical-only format decision, score definitions, eligibility, selectors, arms, thresholds, and staged
evaluation remain unchanged. Expected cost is about 2.5 GPU-hours from the parent's measured reference
throughput, within the balanced funnel.
