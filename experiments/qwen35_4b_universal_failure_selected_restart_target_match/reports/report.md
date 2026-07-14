# Report

## Current status

The model-free design, one authenticated parent rollout, frozen failure selection,
exact-exposure stream freeze, and replay-control training are complete. No candidate
training, merge, local evaluation, or benchmark event has run.

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
- Exact final-arm equality: 320 rows, 297,731 forward tokens, 126,796 loss-bearing
  target tokens, absolute loss mass 27,632.8, zero skips, and 200 byte-identical
  aligned replay rows per arm.
- Stream hashes: control `7a8d4566...b5078`, candidate `28deb20e...3190`, manifest
  `7ba55045...91de1`, and independent token receipt `52a761ef...170`.
- Candidate minus control is zero on the three preregistered axes, answer targets,
  close targets, and parent prefixes. Its 16,414-token target-span difference is
  zero-weight forced-close composition and is disclosed in `compute_review.md`.
- Replay control: 320/320 rows, zero skips, 40/40 steps, train loss 0.3873, complete
  169,903,320-byte adapter. Receipt/log/adapter hashes are `3a9cc1ea...6d49`,
  `3bedc341...f25`, and `5840757d...b1c`.

## Next authorized event

After the control receipt and log are committed, rebased, pushed to `main`, and both
workflows are green, train only the counterfactual-restart candidate. Any merge or
capability evaluation requires a later separately frozen checkpoint.
