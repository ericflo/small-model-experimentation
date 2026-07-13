# Backlog

## Next Experiments

- For training-time policy routing, replace four-branch three-way argmax with
  cross-fitted direct `teacher - student` advantage estimates. Freeze the
  predictor before a third block, allocate extra branches by uncertainty, and
  report route support, precision, abstention, both block signs, and pooled
  bounds. Do not promote a pooled-only or posthoc-margin route.
- Only as a new preregistered experiment, compare joint close-plus-answer likelihood against C51's
  answer-only potential after first passing natural-close and autonomous-parse gates; retain within-task,
  length, prior, shuffled, and foreign controls.
- Test listwise sibling selection only after enriching partial states with feasible-domain and residual
  evidence; gate it against random, surface, no-think, and task-shuffled controls before any search run.
- Compare visible-only stability/simplicity selectors on exact solver pools to close the observed 60/60
  coverage versus 56/60 selected gap without model confidence.
- Train visible-only selectors on candidate pools with explicit false-pass labels held out by family.
- Compare public-test augmentation, generated counterexamples, consensus, and code/verifier reranking on the same pool.
- Retire raw ordered-minus-exact-shuffle probability as a commit-logit selector:
  it beat majority but not confidence/entropy with uncertainty, and task-matched
  shuffle was no better than an oracle-balanced mismatched shuffle. Any
  successor must change the continuation/proposal state, not retune this score.
- Require supplied-hypothesis write control before any latent proposal branch.
  Centered additive J at a last-thought token fails this prerequisite; test an
  explicit semantic anchor plus donor-coordinate clamp only as a new experiment.
- Build an abstaining selector benchmark that reports precision, recall, and coverage separately.
- Stress selectors under intentionally adversarial visible examples.
- Convert oracle ceiling reports into deployable-gap scorecards.

## Required Controls

- First-visible or shortest-visible baseline.
- Hidden oracle ceiling clearly labeled as non-deployable.
- Random or shuffled candidate ordering.
- Family-held-out evaluation.

## Stop Conditions

Do not continue selector variants that improve selected accuracy only by silently reducing commit rate. Precision, recall, and abstention must be visible.

Do not promote partial-state confidence from pooled AUROC: it must clear within-task discrimination and the
deployed recall@beam gate. Type-only independent P(viable) is stopped until the state changes materially.

Do not retry C51 by increasing N or retuning answer-gain thresholds. At 99.37% cap contact, a follow-up
must change the measured event or termination interface before another harvest is justified.
