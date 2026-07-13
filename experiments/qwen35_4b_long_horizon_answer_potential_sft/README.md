# Qwen3.5-4B Long-Horizon Answer-Potential SFT

**Status:** finished

## Status

Design frozen before scientific GPU work. This follow-up is intentionally not a larger replay of C51:
it removes the operative 512-token cutoff, requires naturally closed thoughts, scales the pool to about
95,000 candidate traces, banks complete traces before any compression, and runs the SFT comparison even
if calibration shows only modest predictive signal.

The original run is now paused after 331/1,080 train tasks because its measured wall time exceeded the
available budget. Its design and partial artifacts remain intact; no terminal SFT claim is made here. The
checksum-preserving, prospectively frozen completion is
[`qwen35_4b_balanced_core_answer_potential_sft`](../qwen35_4b_balanced_core_answer_potential_sft/README.md).

The immutable protocol is in [`reports/preregistration.md`](reports/preregistration.md), and the
pre-run adversarial review is in [`reports/design_review.md`](reports/design_review.md).

## Research Program

- Programs: `posttraining_and_adaptation`, `evidence_conditioned_selection`, and
  `test_time_reasoning_budget`.
- Closest near-duplicate: `qwen35_4b_answer_potential_trace_sft` / C51. It sampled 2,048 thoughts at
  512 tokens, force-closed 99.37% of them, found real-but-modest answer-potential signal, and stopped
  before training.
- Other anchors: C28 (own successful thoughts can be inert rationalizations), C50 (the answer-emission
  seam and weighted loss are load-bearing), C44/C45 (serial reasoning can carry installed skill), and
  C24 (training diversity matters more than repeated gradient exposure).

## Question

If Qwen3.5-4B is allowed to finish long reasoning naturally, does the same model's likelihood of a
known correct answer identify complete reasoning strategies that are better SFT targets than random
natural thoughts, binary successful-answer rejection sampling, answer-only SFT, and task-shuffled
thoughts?

The stronger mechanistic question is whether a deployment-matched joint score for
`close-token + ANSWER + correct answer` improves on answer-only potential. The original answer-only
score remains a primary treatment; the joint score is the predeclared seam repair.

## Why This Is A New Experiment

C51 did not train anything. Its fresh-continuation labels showed positive top-choice deltas over
random and shortest traces, and real thoughts beat shuffled and foreign controls. Its dominant fact was
instead interface censorship: nearly every thought was still running at 512 tokens. This experiment
changes four load-bearing variables together and therefore gets a new directory and fresh tasks:

1. natural termination with a 12,288-token allowance and continuation of rare non-loop contacts;
2. about 95,000 candidate traces rather than 2,048;
3. complete-trace banking with no brevity objective; and
4. an actual controlled SFT/evaluation matrix, with no predictive-effect gate before training.

## Model And Firewall

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Fresh procedural atom tasks copied locally from the firewall-clean gauntlet gym.
- No content under `benchmarks/` is read, imported, scored into training data, or used to tune this
  experiment.
- Reference answers and verifiers are oracle-side curation/evaluation instruments only. They are never
  included in prompts at deployment.
- Nine finite-answer training families are used. Stallwright's combinatorial answer-rendering family is
  excluded from potential curation and retained as a transfer-only evaluation family. Brinework and
  spindle remain fully held-out families.

## Frozen Fresh Splits

| split | construction | items | purpose |
| --- | --- | ---: | --- |
| termination pilot | 9 families x L1-L3 x 1 | 27 | capacity and natural-close mechanics only |
| calibration | 9 families x L1-L3 x 5 | 135 | selector diagnostics and threshold freeze |
| harvest/train | 9 families x L1-L3 x 40 | 1,080 | candidate pool and SFT rows |
| IID evaluation | 9 families x L1-L3 x 20 | 540 | primary capability evaluation |
| harder evaluation | 9 families x L4 x 20 | 180 | difficulty transfer |
| held-family evaluation | 2 families x L1-L3 x 30 | 180 | family transfer |
| rendering-held evaluation | stallwright x L1-L4 x 20 | 80 | transfer to an uncurated answer interface |

IDs, prompts, prompt digests, and family-generator seeds must be disjoint across every split before any
model call.

## Long-Horizon Sampling

The experiment has no 512-token reasoning cutoff. Thought sampling stops only when the model emits its
own `</think>` token or reaches the 12,288-token context-safety allowance. A non-loop allowance contact
is resumed once from its exact prefix for up to 2,048 additional tokens; periodic loops are recorded and
excluded rather than laundered into complete reasoning. The finite safety envelope is required by the
16,384-token model context and is not used as a forced-close training intervention.

- independent sampling: temperature 1.0, top-p 0.95, top-k 20;
- calibration: `135 x 64 = 8,640` thoughts;
- harvest: `1,080 x 64 = 69,120` independent thoughts;
- pivot branches: `1,080 x 16 = 17,280` suffix resamples;
- combined planned candidate pool: `95,040` thoughts before deterministic exclusions;
- sampled-token log-probability is captured for the trace-prior diagnostic;
- every raw shard, seed, context contact, loop diagnostic, and sampled token is counted.

Natural closure is a mechanical validity diagnostic, not an effectiveness veto. The full SFT matrix runs
whenever each training task has at least two trainable natural traces after the continuation path. If that
minimum is missed, the runner completes additional independent natural samples for only the deficient
tasks; it does not lower an AUROC bar or force-close traces.

## Scores

For prompt `x`, complete natural thought `z`, and canonical answer `y*`:

```text
answer_gain(z) = log p(y* | x, z, close, ANSWER) - log p(y* | x, empty, close, ANSWER)

joint_gain(z)  = log p(close, ANSWER, y* | x, z)
                 - log p(close, ANSWER, y* | x, empty)
```

`answer_gain` is the original idea under a deployment-valid natural-close pool. `joint_gain` makes the
close/commit seam part of the event rather than teacher-forcing it away. Both are computed uniformly by
a Transformers bf16 SDPA teacher-forced scorer because full-sequence logits are the required internal
measurement; 32 held smoke rows must agree with the exact vLLM targeted readout within 0.15 mean
nats/token before bulk scoring. All generative comparisons remain on vLLM.

Calibration obtains four fresh answer continuations per trace. Harvest obtains one per trace, which both
validates ranking and constructs the binary-success RFT baseline. Calibration reports within-task AUROC,
top-k success curves, length/prior baselines, pre-answer-mention checkpoints, family heterogeneity, and
answer-versus-joint score disagreement. These results freeze selector tolerances but do not cancel SFT.

## Potential-Guided Pivot Branching

For the highest-scoring independent natural trace per training task, score at most eight natural
sentence/newline checkpoints. Choose the boundary immediately before the largest positive joint-gain
jump; if no positive jump exceeds 0.05 nats per canonical-answer token, use the natural boundary nearest
half the trace. Preserve that prefix and sample 16 fresh suffixes to natural close.

This is suffix resampling, never arbitrary token editing. Independent-only `N=64` results remain a
complete nested baseline, and every preserved-prefix prefill plus sampled suffix token is counted. The
branch pool is included in selection regardless of whether it wins; branch source and adoption are
reported so a null cannot disappear.

## Quality, Diversity, Then Full-Trace Banking

Selection is within task and does not reward brevity:

1. retain natural, non-loop, finite-score, trainable-length traces;
2. rank separately by answer gain and joint gain;
3. retain the top 12 candidates per score;
4. choose the top trace, then the most structurally distant trace within 0.25 nats per answer token of
   the top, using identifier/number-normalized token-trigram Jaccard distance; and
5. keep both complete traces exactly. No prefix compression or shortest-near-best rule is applied.

This yields up to 2,160 full-trace rows per potential arm. Compression is explicitly deferred: if
full-trace banking works, a separate follow-up may re-harvest from the trained model and test iterative
compression without contaminating this first causal comparison.

## SFT Arms

All adapters start from the same pinned base, use the same canonical answers, and use seed 42.

| arm | trace target | question |
| --- | --- | --- |
| `empty` | empty thought | does answer learning alone explain gains? |
| `random_natural` | same-task natural trace nearest treatment length | do long thought tokens/style explain gains? |
| `success_rft` | same-task R1-successful natural trace | does ordinary binary rejection sampling suffice? |
| `answer_potential` | diverse top answer-gain full traces | original treatment |
| `joint_potential` | diverse top close-plus-answer-gain full traces | deployment-seam repair |
| `potential_shuffle` | answer-potential traces reassigned within family/level/length | is task-specific reasoning content causal? |

Potential-versus-success comparisons include both the full training sets and the exact common-task
intersection. Success-RFT may have fewer unique tasks; it is oversampled only to match optimizer steps,
and its unique coverage is reported rather than hidden.

QLoRA: rank 32, alpha 64, dropout 0.05, two epochs, learning rate 2e-4, batch 1 x gradient accumulation
16, maximum sequence length 16,000. Prompt loss is 0, full thought loss is 0.5, and close/answer loss is
1.0. The larger thought weight deliberately tests strategy banking; C50's answer-seam weighting remains.
Trace arms are matched on rows, selected-task quotas, optimizer steps, and length as closely as their
definitions permit. Actual supervised and forward tokens are reported.

Adapters remain external. Each is merged into a full composite checkpoint, then must produce a real
on-versus-off behavioral difference before evaluation; vLLM runtime LoRA is prohibited by C49.

## Evaluation

Every base and merged-trained arm generates naturally on vLLM with an allowance of 12,288 tokens; no
evaluation arm is force-closed at 512. Primary: fresh IID greedy exact-answer accuracy and paired
`answer_potential - random_natural`, `answer_potential - success_rft`, and
`answer_potential - potential_shuffle` deltas.

Secondary:

- joint-potential comparisons;
- natural-close, parse, and parse-conditional accuracy;
- mean/median/p95 thinking tokens;
- family macro, L4, held-family, and stallwright transfer;
- k=8 sampled coverage, majority selection, diversity, and oracle pass@8 ceiling;
- base sample-more accuracy versus actual forward tokens; and
- branch adoption and independent-N scaling at N in {8, 16, 32, 64}.

The primary seed-42 matrix always completes. If either potential arm beats the strongest matched trace
baseline by at least 0.03 IID with paired 95% lower confidence bound above zero and no >0.02 parse/family
regression, that treatment and the strongest baseline are replicated at training seed 43.

## Decision Rules

- **Scorer informative / banking negative:** scores predict outcomes, but potential SFT does not beat
  random natural and success-RFT controls.
- **Full-trace banking positive:** a potential arm beats both controls by at least 0.03 IID, paired 95%
  lower bound above zero, beats shuffled content, and preserves parse/family macro within 0.02.
- **Replicated banking positive:** the seed-43 contrast has the same sign and the pooled paired interval
  excludes zero.
- **Mission positive:** replicated banking positive and the trained greedy point beats base sample-more
  at some matched actual-forward-token point.
- **Compression licensed:** full-trace banking positive. No compression claim is made in this experiment.

There is deliberately no pre-SFT AUROC/effect-size stop. A terminal claim about the overarching idea is
made only after the complete seed-42 SFT matrix and fresh evaluation.

## Run

The restartable staged interface is:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_long_horizon_answer_potential_sft/scripts/run.py --stage smoke
.venv-vllm/bin/python experiments/qwen35_4b_long_horizon_answer_potential_sft/scripts/run.py --stage full
```

Granular stages include `pilot`, `calibration-generate`, `scorer-parity`, `calibration-score`,
`calibration-rollouts`, `harvest-generate`, `harvest-score`, `pivot-plan`, `branch-generate`,
`branch-score`, `train-rollouts`, `select`, `train`, `merge`, `deployment-probe`, both evaluation modes,
analysis, and conditional replication. Sharding, external paths, and exact commands are registered in the
artifact manifest and experiment log.

## Results

### Termination and training-envelope gates

The registered 27-task termination pilot is complete (108 traces). Every trace exceeded the old
512-token cutoff; median thought length was 4,636 tokens and p95/max was 14,336. The model closed
naturally on 96/108 traces (88.9%), with zero exact periodic loops. Thirteen traces reached the initial
12,288-token allowance; one then closed during the exact-prefix continuation. The 12 still-open traces
were all loomfix and remain ineligible rather than being force-closed. Correctness was not inspected in
this operational pilot.

The exact-token QLoRA path was also validated before dataset construction. Ordinary 3--4k rows train in
4.7 seconds per two-example optimizer step. A deliberately worst-case 14,687-token row initially exposed
a quadratic SDPA backward workspace, then passed untruncated in 29.1 seconds at 15.0 GiB peak after the
training-only full-attention kernel was moved to xFormers and >8k rows received explicit layer/loss
checkpointing. The six-arm scientific matrix has not yet reported a result. The complete calibration
harvest is now banked: 8,640 independent traces over 135 tasks (N=64), totaling 45,728,102 sampled
thought tokens. Of these, 7,814 closed naturally and 27 exact periodic loops were detected; loops and
every unresolved allowance contact remain mechanically ineligible. The preregistered 32-row HF/vLLM
canonical-answer likelihood parity gate passed with a maximum difference of 0.000448 nats per answer
token (threshold 0.15). Calibration answer rollouts and full-prefix scoring are in progress.

Loomfix is the important termination stress case: its 960 calibration traces consumed 12,676,528
sampled tokens, but only 204 (21.3%) closed naturally within 12,288 plus the single exact-prefix 2,048
continuation. This is preserved as evidence that even the enlarged protocol does not cover every search
horizon; it does not veto the complete SFT matrix, and unresolved traces are never force-closed or used
as training examples.

Calibration scoring and R=4 answer rollouts are also complete for all 7,814 eligible traces (31,256
answer rollouts). Answer gain has task-macro AUROC 0.597 and its top-1 trace averages 22.46% rollout
success versus 15.63% for seeded random (+6.84 percentage points). Joint close+answer gain has AUROC
0.678 and top-1 success 21.88% (+6.25 points). Canonical versus one-newline score rankings are stable
(task-macro Kendall tau-b 0.841). These are encouraging selector diagnostics, not a banking result:
negative length (AUROC 0.690; top-1 26.56%) and the sampled-trace prior (AUROC 0.700; top-1 22.46%) are
strong controls. Per the preregistration, none of these diagnostics gates the 1,080-task harvest or the
complete six-arm SFT matrix.

No scientific result yet. This file records the complete pre-run plan; later results are added above this
boundary without rewriting the frozen preregistration.

## Artifacts

- `idea_intake.md`: novelty and near-duplicate decision
- `reports/preregistration.md`: immutable protocol and decision rules
- `reports/design_review.md`: adversarial pre-run review
- `configs/default.yaml`: frozen counts, seeds, model, and recipes
- `reports/artifact_manifest.yaml`: sharded pools, adapters, merged checkpoints, and checksums
- `runs/`: compact receipts and scientific summaries retained in git
- external root: `/workspace/large_artifacts/qwen35_4b_long_horizon_answer_potential_sft`
