# Backlog

## Next Experiments

- Treat any C51 follow-up as a new close/commit-scoring experiment, not a larger rejection-sampling run;
  require autonomous termination and parse gates before training.
- Treat any C52 follow-up as a locality experiment first: compare lower-dose
  positive-only uplift with a context-gated intervention, and stop before
  downstream training/evaluation if exact-logit non-target drift exceeds 0.10.
- Compare DPO, SFT, process distillation, and DAgger on one shared candidate/evidence substrate.
- Add adapter-free reproducibility manifests for every trained run.
- Measure catastrophic narrowing: does an update improve one substrate while hurting direct baselines?
- Train on hard negatives and evaluate whether coverage or selection improves.
- Distill process labels only when labels are deployable or clearly marked oracle-only.
- C45 follow-up: compositional-grammar reasoning-SFT. Teach a serial hypothesize-and-verify search over condition x action rules, with mixed execute examples to prevent forgetting. Evaluate held-out combinations, held-out productions, and held-out composition-depth separately; require execute-ceiling gates, token-budget/truncation checks, and branch-coverage/value-fill sufficiency checks before interpreting induction failures.

## Required Controls

- Frozen model.
- Shuffled labels or shuffled traces.
- Same-token-budget sampling.
- Held-out task and family evaluation.

## Stop Conditions

Do not retain trained adapters in git. Do not claim a posttraining method works if it beats only a weak baseline and fails frozen or shuffled controls.

Do not spend SFT compute on a trace score that has not cleared a preregistered within-task outcome gate and
practical top-selection margin. C51 is stopped before training under its current answer-only potential event.
