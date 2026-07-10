# Preregistration: Long-Horizon Answer-Potential SFT

## Freeze Boundary

This document, the experiment README, configuration, idea intake, and design review are committed before
any scientific GPU call. A later receipt records that commit and hashes these files. Result-bearing
thresholds may not be changed after calibration output is observed.

## Claim Under Test

Among many complete natural thoughts sampled by Qwen3.5-4B for a task with a known training-time answer,
the thoughts that make the correct answer most likely are better SFT targets than random natural thoughts
or thoughts retained because one sampled answer happened to be correct.

The claim is specifically about self-generated reasoning strategies under long natural termination. It is
not a claim that hidden reference answers are available at deployment.

## Changes From C51

- 512 forced-close -> 12,288 natural-close allowance plus one 2,048-token continuation for non-loop
  contacts.
- 2,048 thoughts -> 95,040 planned candidates.
- no training -> complete six-arm seed-42 SFT matrix.
- shortest-near-best -> two full traces selected by quality then structural diversity.
- answer-only likelihood alone -> answer-only primary plus joint close-and-answer secondary.
- effectiveness stop at calibration -> no effectiveness stop before SFT.

## Population And Splits

Fresh seeds and exact counts are frozen in `configs/default.yaml`. Nine finite-answer families contribute
135 calibration tasks, 1,080 training tasks, 540 IID tasks, and 180 L4 tasks. Brinework/spindle contribute
180 held-family tasks. Stallwright contributes 80 rendering-held transfer tasks but no curated trace.

The split builder must fail on any cross-split overlap in ID, prompt, prompt digest, or family-generator
seed. Training-stage imports must not load held-family registries.

## Sampling And Termination

Independent thoughts use temperature 1.0, top-p 0.95, top-k 20, exact stable request seeds, and vLLM.
The model stops its own thought with `</think>`. A length contact is not silently treated as a complete
trace. Non-loop contacts receive one continuation from the exact token prefix; exact periodic loops and
still-unclosed continuations remain negative yield and are excluded from SFT.

The termination pilot can change only operational geometry: concurrency may decrease to fit live KV/Mamba
capacity, and the registered continuation path may be exercised. It may not reduce N, shorten the allowance,
force-close primary traces, or inspect correctness when choosing geometry.

## Candidate Counts

- calibration independent: 8,640;
- training independent: 69,120;
- training pivot suffixes: 17,280;
- planned total: 95,040 before deterministic exclusions.

If a training task has fewer than two natural, non-loop, trainable traces, the runner samples an additional
batch of 16 independent thoughts for that task until it has two or has received four top-up batches. A task
still deficient after four batches is retained in answer-only analysis but excluded symmetrically from all
trace arms. This is a mechanical data-availability rule, not an outcome gate.

## Likelihood Readouts

`answer_gain` and `joint_gain` are defined exactly in the README. The full-sequence Transformers scorer is
used uniformly for both readouts. Before bulk scoring, 32 smoke traces are rescored by vLLM's exact targeted
next-token readout; maximum absolute mean discrepancy must be <=0.15 nats/token. Failure is an instrument
block requiring a pre-result implementation repair, not evidence against the hypothesis.

Trace generation requests sampled-token log-probabilities. The runner must abort the first scientific shard
if cumulative trace prior is absent or nonfinite on any nonempty smoke trace.

## Calibration

Each of 8,640 calibration traces receives four fresh short answer rollouts from disjoint seeds. Report,
without gating SFT:

- task-macro within-task AUROC against rollout correctness for answer gain, joint gain, negative length,
  and trace-prior mean log-probability;
- top-{1,2,4,8} selected rollout success versus seeded random, shortest, and prior;
- natural-close, loop, answer-mention, parse, and family diagnostics;
- score stability to an equivalent answer format; and
- answer-gain versus joint-gain rank correlation.

Selector near-best and branching tolerances remain the values in config regardless of these outcomes.

## Branching

Checkpoint and pivot selection is deterministic and answer-score-based, never verifier-outcome-based. The
root is the highest joint-gain independent trace. Up to eight natural boundaries are scored; the pivot rule
is frozen in the README. Branch suffix seeds are disjoint from independent and rollout seeds. Independent
N=64 remains a complete control pool.

## Dataset Construction

Every SFT record contains exact prompt tokenization, exact sampled thought token IDs, exact registered close
and answer boundary, canonical answer, source trace ID, source kind, length, and score. Potential selection
uses no length reward. Random and shuffled controls are matched to selected lengths within task or
family/level strata. No benchmark row can enter.

Success-RFT uses exactly one fresh rollout per candidate. Tasks with no success are not relabeled; unique
task coverage and a common-task intersection are mandatory outputs.

## Training

The six seed-42 adapters and recipe are frozen in config. The full sampled trace is supervised at weight
0.5; close/boundary/answer tokens at 1.0; prompt at 0. Training must fail rather than truncate a selected
target. Rows, effective optimizer updates, supervised tokens, skipped rows, elapsed time, peak memory, and
adapter hashes are recorded.

Adapters and merged checkpoints live outside git. Each merged checkpoint must pass a same-prompt greedy
behavioral-difference probe against base before scientific evaluation.

## Evaluation And Statistics

All generative arms use vLLM natural thinking and the same sampling parameters. IID greedy accuracy is the
primary metric. Paired task bootstrap uses 10,000 resamples. The complete seed-42 matrix runs before any
verdict. Training seed 43 is launched only for a potential treatment and its strongest trace baseline when
the frozen +0.03/CI/noninferiority trigger is met; the initial hypothesis has already been fully tested at
that point.

Report full and common-task training comparisons, family macro, L4, held families, stallwright, parse,
termination, length, pass@8, majority@8, oracle pass@8, actual inference tokens, curation tokens, and training
tokens.

## Verdicts

Verdicts are the README decision rules. No positive claim is allowed from calibration alone. No negative
claim about full-trace banking is allowed without the completed seed-42 training/evaluation matrix. A
positive compression follow-up is licensed only by full-trace banking positive.

## Amendments

Only dated, pre-result operational amendments may be added below. They must state whether any scientific
output was observed and preserve the original text above.
