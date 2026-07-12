# Backlog

## Next Experiments

- Completed cross-program qualification negative:
  `qwen35_4b_pareto_policy_integration` replaced the arbitrary absolute bar
  with replicated paired `delta > 0`, then found that C54's labels did not
  transport to the clean distillation proxy. `blend` was negative on quick in
  both blocks; `apex` was positive on deep but failed retention. No MOPD update
  ran.
- Completed route-qualification negative:
  `qwen35_4b_same_prefix_advantage_routing` scored 384 fresh soup states with
  disjoint selection/audit branches. Deep independently passed, and the
  combined router passed, but quick reversed from `+0.2009` to `-0.0253`
  versus the soup across blocks. Four-branch statewise argmax is stopped as a
  two-teacher labeler; MOPD itself remains untested.
- Highest-value new experiment: requalify the frozen deep selection/audit rule
  on fresh states, then test deep-only verifier-backed MOPD from the 40/60
  soup. Require locality before escalation and still compare the final single
  checkpoint with quick, deep, soup, visible routing, matched controls, and
  sample-more. This isolates the validated teacher signal from the failed
  two-teacher estimator.
- A later two-teacher successor must estimate each direct
  `teacher - student` advantage with cross-fitting and uncertainty-aware branch
  allocation, freeze the predictor, and confirm both teachers on a third
  untouched block. Do not retry statewise three-way argmax or repair it with a
  posthoc fixed margin.
- Stopped cross-program test: `qwen35_4b_specialist_policy_integration` reached
  a design-feasibility negative before any specialist or integration update.
  Its sole tools core scored 0.994 at baseline, so the mandatory `+0.10` gain
  required 1.094 on a score capped at 1.0. A new run must use a harder,
  independently calibrated tools/provenance core and prove every arm's
  theoretical headroom before best-of-k or training; do not amend this result.
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

Do not spend posttraining compute on a mandatory arm whose frozen absolute
gain criterion exceeds its score ceiling. Endpoint-average headroom does not
establish per-arm feasibility.

Do not distill from a teacher merely because it won an external aggregate
instrument. Require positive same-prefix continuation advantage on the actual
training-state distribution, replicated on a disjoint block.
