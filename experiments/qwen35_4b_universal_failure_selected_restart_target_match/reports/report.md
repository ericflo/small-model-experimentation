# Report

## Current status

The model-free design, one authenticated parent rollout, and frozen failure selection
are complete. No stream exposure match, training event, local evaluation, or
benchmark event has run.

The active hypothesis is that task-level on-policy failure selection can help when
the supervised example restarts cleanly before the error and target exposure is
matched exactly to replay. See `preregistration.md` and `design_review.md` for the
frozen contract.

## Evidence

- 624 fresh truth-audited tasks, balanced 48 per universal skill.
- Zero prompt overlap with predecessor collection/local sources and prior reserved
  local seeds.
- Explicit merged replay-parent authentication.
- Unit-tested removal of the parent's failed prefix from trainable rows.
- Design receipt SHA-256: `e861cd647c5a39df893366b948a39fc2bf67ac08e1b1fe704a69032597ffae24`.
- Parent event: 624/624 completions, 304,013 sampled tokens, 879.9 tok/s, 394.96
  seconds. Rollout receipt SHA-256: `1d35c63a70d53d8803666cb8c30f4d0efffd884c7f6ab04adceaf8b05442b381`.
- Selection: 602 eligible, 228 hard failures, 52 selected (four per skill), with 40
  hard and 12 budget-only rows. All 52 are full oracle restarts from the original
  prompt and zero contain a parent prefix.

## Next authorized event

Self-contained replay copying, token measurement, deterministic exact three-axis
exposure feasibility, and a second adversarial compute review after this selection
checkpoint is committed, rebased, pushed to `main`, and both workflows are green.
