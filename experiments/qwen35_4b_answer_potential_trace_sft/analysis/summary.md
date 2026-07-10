# Terminal Analysis Summary

## Verdict

`SCORER_NEGATIVE`: the preregistered G0 gate failed with 3/8 stored criteria passing. The full-stage
guard refused the N=128 harvest and SFT, so there is no trained-arm result.

## Short Read

Teacher-forced canonical-answer gain after a sampled thought contained real task-relevant information:
it beat token-shuffled and foreign thoughts, retained its ranking under an answer-format perturbation,
and selected traces whose fresh rollout success was modestly above random and shortest selection. It
was not actionable enough to bank. Task-macro within-task AUROC was 0.617 versus the frozen 0.65 gate,
and top-one uplifts of +0.073 and +0.058 missed the required +0.10.

The largest mismatch was at the close/commit seam. Of 2,048 thoughts, 2,035 hit the 512-token cap and
only 13 closed naturally. Forced-close answer continuations parsed 13.2% of the time, but parsed answers
were 86.9% correct. The score measured a useful answer state after an injected boundary, not a state
the model reliably reached and expressed autonomously.

## Durable Lessons

- Real-versus-shuffled and real-versus-foreign controls are necessary but not sufficient evidence for
  a trace selector.
- Within-task prediction and top-choice effect size must be gated before SFT; positive confidence
  intervals alone do not make a selector useful.
- Natural-close and parse-rate gates should precede large thought harvests.
- A trace-prior baseline declared in the design must be captured during generation and asserted before
  the first scientific shard. This run missed it and failed that criterion closed.
- Any follow-up must materially change the measured event—jointly score closure plus answer, or first
  establish adequate naturally closing coverage—not merely increase samples or retune thresholds.

Machine-readable summaries: [`g0_summary.json`](g0_summary.json),
[`g0_metrics.csv`](g0_metrics.csv), [`family_summary.csv`](family_summary.csv), and
[`compute_summary.json`](compute_summary.json).
