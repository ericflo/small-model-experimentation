# Qwen3.5-4B Answer-Potential Trace SFT

## Status

Design frozen before any GPU-scale run. The full pre-registration is in
[`reports/preregistration.md`](reports/preregistration.md), and the adversarial design review is in
[`reports/design_review.md`](reports/design_review.md). The first repository commit containing this
directory is the immutable pre-run design boundary.

## Research Program

- Programs: `posttraining_and_adaptation`, `test_time_reasoning_budget`,
  `evidence_conditioned_selection`
- Program question: can a dense, answer-conditioned signal identify self-generated thinking that is
  worth banking, where binary correct-answer rejection sampling selected inert rationalizations?
- Closest duplicate: `qwen35_4b_bank_the_thoughts` Phase 2 (C28), which selected the model's own
  thoughts only after a sampled answer happened to be correct and found no coverage gain over
  answer-only SFT.
- Other anchors: C9 (coherent thinking content is load-bearing), C46/C47 (within-task probability
  readouts and pooled-score failure), and C50 (breadth plus answer-seam-weighted SFT transfers).

## Question

Does teacher-forced likelihood of a known canonical answer, measured *after a sampled thought but
before sampling an answer*, identify concise and diverse thoughts that improve held-out deployable
accuracy after SFT more than binary successful-answer rejection sampling?

This is RL-free but not oracle-free. Reference answers are used only during training-data curation;
they are unavailable to every deployed selector and appear only in evaluation graders after the
split is frozen.

## Hypothesis

For prompt `x`, pre-answer thought `z`, and canonical answer `y*`, sample `z ~ p(z|x)` and score

```text
gain(z) = log p(y* | x, z) - log p(y* | x, empty_thought).
```

Because thoughts are already sampled from the model prior, reweighting them by `p(y*|x,z)` is an
importance-sampling approximation to the posterior over thoughts conditioned on the correct answer.
Unlike a one-rollout binary filter, the score marginalizes answer-emission luck. The treatment should
therefore enrich for thoughts that place the model in a genuinely better pre-answer state.

The claim is false if answer gain cannot rank fresh trace-conditioned answer rollouts within a task,
if it merely selects short/format-priming/answer-copying traces, or if the selected traces do not beat
length-matched random and binary-success trace SFT on fresh tasks.

## Novelty Boundary

This is not a claim that future-token likelihood or reasoning-potential search is globally new. It is
the repository's first controlled test of the following combination on the fixed model:

1. sample only the visible `<think>...</think>` region without showing the answer;
2. use the same model's teacher-forced canonical-answer likelihood as a dense oracle-side trace score;
3. validate that cheap score against fresh continuation success before training;
4. select quality first, then structural diversity, then the shortest near-best representative;
5. optionally branch at measured potential drops rather than perturbing arbitrary tokens; and
6. bank the resulting trace through matched, answer-seam-weighted QLoRA SFT.

## Substrate And Firewall

- Model: **only** `Qwen/Qwen3.5-4B`, pinned to repository revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Substrate: a self-contained copy of the procedurally generated atom families from
  `qwen35_4b_gauntlet_breadth_round1`; no episode tasks in this first test.
- Training families: the ten original trained atom families.
- Family-transfer evaluation: the two original held-out atom families, never used in calibration,
  harvesting, threshold selection, or training.
- No content under `benchmarks/` is imported, inspected, or used for training.
- All generator seeds are split-disjoint. Generated item IDs and canonical prompt digests must be
  disjoint across calibration, train, IID evaluation, held-family evaluation, and harder-level
  evaluation before any model call.

The atom-only choice is deliberate: short canonical `ANSWER:` targets make answer likelihood a clean
measurement. Multi-turn action likelihood is a separate follow-up if this mechanism works.

## Frozen Splits

| split | families / levels | items | use |
| --- | --- | ---: | --- |
| calibration | training families, L1-L2 | 64 | scorer validation and threshold freeze |
| harvest/train | 10 training families x L1-L2 x 30 | 600 | candidate pool and SFT construction |
| IID evaluation | 10 training families x L1-L2 x 20 | 400 | primary held-out accuracy |
| held-family evaluation | 2 untouched families x L1-L2 x 25 | 100 | family transfer |
| harder-level evaluation | 10 training families x L3 x 10 | 100 | difficulty shift |

Counts may shrink only for deterministic generator rejection or an explicit pre-result amendment.
They may not be changed after observing treatment metrics.

## Sampling Protocol

Calibration uses `N=32` independent thoughts per prompt. The full pool uses `N=128` with nested
first-16/32/64/128 analyses from the *single* N=128 artifact; different vLLM batch shapes are never
treated as paired continuations.

- temperature `1.0`
- top-p `0.95`
- top-k `20`
- maximum sampled thinking tokens `512`
- stop at the first `</think>` token; do not sample an answer in the treatment harvest
- explicit fixed shards, request seeds, prompt order, vLLM scheduler, capacity-fit receipt, and
  sampled-token accounting

Natural-close rate is a launch diagnostic. Forced-close thoughts may be scored, but they are not
supervised as complete reasoning unless a sentence-boundary compression step explicitly turns them
into a registered early-close target.

## Answer-Potential Score

Render the exact thinking-channel prefix plus the sampled thought, the exact `</think>\n\nANSWER: `
boundary, and the canonical answer content. Use vLLM prompt log-probabilities to gather only the
observed answer-token log-probabilities. Exclude the fixed boundary and terminal token.

For each trace store:

- summed and mean canonical-answer log-likelihood;
- gain against the same prompt with an empty thought;
- first answer-content-token log-probability;
- correct-versus-procedural-decoy margin when the family admits unambiguous decoys;
- sampled trace log-probability, length, closure, duplicate, and loop diagnostics;
- whether and where the canonical answer first appears verbatim in the thought; and
- scores at natural sentence/newline checkpoints for compression and branching.

Raw scores are never pooled across tasks. Every calibration statistic is task-macro or within-task.
Multiple valid answer renderings are scored as an equivalence set or the item is excluded from the
confirmatory scorer gate.

## Gate G0: Cheap Scorer Validation

On the 64 calibration prompts, generate eight fresh short answer continuations from every thought
using seeds not used for thought generation. These continuation outcomes approximate the expensive
rollout potential and are not used in the likelihood score.

Proceed to the N=128 harvest only if all conditions hold:

1. task-macro within-task AUROC of answer gain against fresh rollout correctness is at least `0.65`;
2. top-one-by-gain rollout success exceeds random-trace and shortest-trace selection by at least
   `0.10`, with paired task-bootstrap lower confidence bound above zero for both comparisons;
3. answer gain beats trace length and trace prior-likelihood as within-task rankers;
4. real thoughts beat length-matched token-shuffled and foreign-task controls;
5. answer-format perturbations retain rank correlation `Kendall tau >= 0.80`; and
6. at least 75% of selected traces show positive gain *before* their first verbatim answer mention,
   or contain no verbatim answer mention.

If G0 fails, stop before SFT. The negative result is that canonical-answer likelihood is not a valid
trace-value interface under this protocol.

## Trace Selection

Quality, diversity, and brevity are lexicographic rather than collapsed into one tuned scalar:

1. discard malformed loops, exact duplicates, nonfinite scores, and traces below a frozen prior-
   likelihood floor;
2. retain the high-answer-gain set using the threshold frozen on calibration;
3. normalize identifiers, numbers, and whitespace, then use token-trigram Jaccard distance and
   deterministic farthest-first selection to retain distinct trace structures;
4. within each selected cluster, choose the shortest trace within the frozen near-best gain
   tolerance; and
5. cap at two traces per item and apply level/kind round-robin plus per-family and global-template
   caps, targeting 900-1,200 rows across at least eight families.

For prefix compression, score natural boundaries and choose the earliest boundary within the frozen
tolerance of the trace's maximum score. Never cut inside a token or unfinished sentence.

## Conditional Pivot/Branch Arm

The perturbation arm resamples suffixes; it never edits arbitrary tokens in-place. For each seed
trace, preserve prefixes through positive potential jumps and branch immediately before the first
material score drop or long plateau. Compare:

- `independent_128`; and
- `independent_64_plus_branch_64`.

Include every repeated-prefix prefill and sampled suffix token in matched-compute accounting. The
branch pool is eligible for SFT only if it improves top-selected fresh-rollout success over
independent sampling with a paired lower confidence bound above zero and does not reduce task or
family coverage. Otherwise it remains a negative pool-level ablation.

## SFT Arms

All arms use identical prompts, canonical answers, row counts, family/level quotas, optimizer steps,
and base initialization. Only the trace-selection rule changes.

| arm | thought target | purpose |
| --- | --- | --- |
| `empty` | empty thinking region | channel-matched answer-only floor |
| `random_length` | random same-task trace nearest treatment length | controls extra tokens/style |
| `success_rft` | trace whose one sampled continuation was correct | binary rejection-SFT / C28 baseline |
| `potential` | diverse compressed high-answer-gain trace | primary treatment |
| `potential_shuffle` | selected trace reassigned within family/level/length stratum | content-causality falsifier |

The applied potential arm may cover prompts with no successful sampled continuation. A matched-task
intersection analysis separates better trace quality from broader task coverage.

Training format:

```text
prompt -> <think> selected thought </think>\n\nANSWER: canonical answer
```

QLoRA recipe: rank 32, alpha 64, dropout 0.05, two epochs, learning rate `2e-4`, prompt loss 0,
thought-token loss 0.2, and close/answer loss 1.0. Truncated recovery contexts receive thought loss 0.
The answer is retained in every arm because C50 found the answer-emission seam load-bearing; the
experiment tests which *thought context* should be learned, not whether to suppress answer learning.

Screen every arm at training seed 42. If `potential` beats the strongest baseline by at least 0.03
on the frozen IID screen without a parse-rate loss, replicate `potential` and that baseline at seed
43 before making a positive claim. Otherwise the training verdict is negative/inconclusive and no
selective replication is launched.

## Evaluation

Run all base and trained arms through one inference backend and identical prompt renderer. Because
vLLM 0.24 runtime LoRA silently no-ops for this model (C49), merge each adapter into the composite
checkpoint and require a real greedy on-versus-off behavioral-difference gate before accepting its
evaluation.

Primary:

- fresh IID greedy exact-answer accuracy at think budget 512;
- paired `potential - success_rft` and `potential - random_length` deltas;
- actual sampled thinking tokens and total forward-token accounting.

Secondary:

- parse rate and parse-conditional accuracy;
- no-think off-diagonal deployment;
- coverage/pass@8 and unique-answer diversity;
- held-family and harder-level macro accuracy;
- selected-trace length, natural-close rate, answer-copy rate, and generated-trace adoption;
- base sample-more curves using deployable majority/confidence selection; and
- oracle pass@k only as a clearly labeled ceiling.

The repository mission gate is stricter than an adapter delta: the potential-trained curve must beat
the base sample-more curve at at least one matched actual-forward-token point without losing greedy
accuracy or family-macro coverage. One-time training and curation cost is reported separately as an
amortization curve.

## Decision Rules

- **Scorer negative:** G0 fails; do not harvest or train.
- **Selector positive / training negative:** G0 passes but `potential` does not beat both matched
  trace controls after SFT; retain the scorer as a measurement result only.
- **Local SFT positive:** `potential` beats `success_rft` by at least 0.05 IID absolute accuracy,
  paired 95% confidence lower bound above zero, and beats `random_length` and
  `potential_shuffle`, with no parse or family-macro regression larger than 0.02.
- **Efficiency positive:** in addition, median thought length is at most 70% of `success_rft`, or
  treatment strictly Pareto-dominates it in accuracy versus sampled tokens.
- **Mission positive:** local SFT positive plus a win over matched-forward-token base sample-more.
- **Transfer positive:** held-family delta is positive with paired lower confidence bound above zero;
  otherwise any win is explicitly substrate/family-local.

## Run

CPU and tiny GPU smoke:

```bash
python experiments/qwen35_4b_answer_potential_trace_sft/scripts/run.py --smoke
```

Calibration gate:

```bash
python experiments/qwen35_4b_answer_potential_trace_sft/scripts/run.py --stage calibrate
```

Full gated pipeline:

```bash
python experiments/qwen35_4b_answer_potential_trace_sft/scripts/run.py --stage full
```

The orchestrator must refuse `--stage full` without a committed design-boundary receipt and a
passing G0 artifact unless an explicit diagnostic-only override is recorded.

## Results

No scientific result yet. This README records the pre-run design and will be updated additively after
the gated run; the frozen design remains available in `reports/preregistration.md` and git history.

## Interpretation

The experiment is designed so every terminal outcome compounds knowledge: scorer failure rejects the
interface cheaply; scorer success plus SFT failure separates measurement from learnability; and a
controlled SFT win would establish answer-potential posterior mining as a better thought-bank builder
than binary rejection sampling for the fixed 4B.

## Knowledgebase Update

After the terminal result, update all owning program evidence/backlogs, C28/C50 or a new claim as
warranted, shared synthesis if strategy changes, the practitioner brief, and a native chart spec.

## Artifacts

- `idea_intake.md`: novelty and routing decision
- `reports/preregistration.md`: immutable design specification
- `reports/design_review.md`: pre-run adversarial review
- `configs/default.yaml`: frozen counts, seeds, thresholds, and recipes
- `src/vllm_runner.py`: pinned common inference backend
- `reports/artifact_manifest.yaml`: external adapters/checkpoints and regeneration commands
- `runs/`: compact receipts, gates, summaries, and scored rows
- `analysis/`: derived tables and plots
