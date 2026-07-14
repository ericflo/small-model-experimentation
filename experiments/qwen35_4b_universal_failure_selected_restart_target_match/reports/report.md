# Report

## Current status

The model-free design is complete. No parent rollout, training event, local
evaluation, or benchmark event has run.

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

## Next authorized event

One parent rollout at seed 66,114 after this checkpoint is committed, rebased, pushed
to `main`, and both required GitHub workflows are green.
