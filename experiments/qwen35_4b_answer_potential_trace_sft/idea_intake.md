# Idea Intake

## Program Fit

- Programs: `posttraining_and_adaptation`, `test_time_reasoning_budget`,
  `evidence_conditioned_selection`
- Existing or new program: existing programs; no new charter is needed.
- Closest program scorecard reviewed: `knowledge/program_scorecards.md`, especially Posttraining And
  Adaptation and Evidence-Conditioned Selection.
- Related future queue item: `supervision_causality_ablation` is adjacent but broader; this is a
  concrete trace-selection mechanism rather than a general supervision-source tournament.

## Prior Evidence

- Anchor 1: C28 / `qwen35_4b_bank_the_thoughts` -- binary-success-selected own thoughts did not beat
  answer-only banking; concise explicit plans did.
- Anchor 2: C9 / `qwen35_4b_thinking_content_vs_compute` and its budget follow-up -- coherent thought
  content, not filler or shuffled content, causes the thinking gain.
- Anchor 3: C46/C47 -- probability readouts must be validated within task; pooled scores can be pure
  difficulty, and a good completed-candidate ranker need not be a good training filter.
- Anchor 4: C50 / `qwen35_4b_gauntlet_breadth_round1` -- broad procedural SFT transfers only after
  canonicalizing answers, representing recovery states, and weighting thought 0.2 / answer 1.0.
- Closest duplicate: `qwen35_4b_bank_the_thoughts` Phase 2.

## Novelty Claim

The unresolved uncertainty is whether C28 failed because the model's thoughts were intrinsically
useless, or because a one-sample correct-answer filter is a noisy selector that admits lucky answer
emissions paired with inert rationalizations. This experiment replaces that binary event with the
same model's dense teacher-forced likelihood of the canonical answer before answer sampling, validates
the score against fresh continuations, and banks quality/diversity/brevity-constrained traces.

## Mechanism

Thoughts are latent variables. Sampling thoughts from `p(z|x)` and weighting them by
`p(y*|x,z)` approximates the answer-conditioned posterior over thoughts without policy-gradient RL.
The explanation is false if likelihood gain cannot rank independent continuation correctness, is
explained by length/format/answer copying, or selects data that does not beat binary RFT and
length-matched random traces after matched SFT.

## Control Plan

- Baselines: empty-thought answer SFT, same-task length-matched random trace SFT, binary successful-
  continuation rejection SFT, independent N=128 sampling.
- Mechanism-falsifying controls: potential-selected traces shuffled within family/level/length,
  token-shuffled and foreign thoughts during scorer calibration, pre-answer-mention gain, and
  matched-task intersection analysis.
- Shift checks: fresh IID tasks, never-trained families, harder level 3, no-think deployment, and
  pass@8 diversity.
- Hidden-label boundary: canonical answers are oracle-side curation labels only. Eval references are
  not rendered into prompts and no `benchmarks/` content is read or trained on.
- Compute control: report actual prompt/prefill plus sampled tokens; branch-prefix recomputation and
  scoring passes count. Compare against base sample-more at matched forward tokens.

## Evidence Output

- Program evidence update: all three owning program evidence/backlog files after the terminal result.
- Claim ledger or synthesis update: revise C28's scope or add a new claim only if the gated run
  supports it; preserve a negative scorer/SFT result otherwise.
- Reusable artifacts: thought-only vLLM path, teacher-forced answer-span scorer, within-task gate,
  deterministic quality-diversity-brevity selector, and matched SFT builder.
- Stop or branch condition: hard stop at G0 scorer failure; pivot/branch SFT only if its pool-level
  matched-compute gate passes; replication only after the frozen seed-42 screen threshold.

## Decision

- Run experiment: yes, with the pre-registered staged gates.
- Create program: no.
- Write synthesis only: no; C28 does not test answer-potential selection.
- Defer: multi-turn action targets and deep-frontier composition are follow-ups, not this experiment.
