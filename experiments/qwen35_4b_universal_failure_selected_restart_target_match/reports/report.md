# Report

## Current status

The model-free design, one authenticated parent rollout, frozen failure selection,
exact-exposure stream freeze, both paired training events, and the separately
reviewed fresh-local protocol, and replay-control explicit merge are complete. No
candidate merge, local result, or benchmark event has run.

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
- Restart candidate: 320/320 rows, zero skips, 40/40 steps, train loss 0.5838,
  complete 169,903,320-byte adapter. Receipt/log/adapter hashes are
  `6aa5c3f1...9871`, `c8572c88...202a`, and `2072c5c8...39bc`. Its receipt binds the
  published control prerequisite and independent original-parent warm start.
- Fresh local protocol: seed 88,010; 26 new tasks, two per all 13 skills; hidden-free
  runner input; identical explicit-composite vLLM geometry across unchanged parent,
  replay control, and candidate; complete model-tree authentication; strict absolute
  and two-control-relative promotion rules. Source/input/design hashes are
  `7b69473b...975f`, `6efefc92...15e2`, and `124bbf99...2db5`. This is design
  evidence only.
- Replay-control composite: 128/128 nonzero merged modules at scale 2; exact
  seven-file tree hash `d1a8336d...6027`; merged weight `e48ed4a0...ae17`;
  run/external receipts `751a0152...f72f` / `bcb0060e...53e2`. This authenticates
  deployment and is not a capability result.

## Next authorized event

After the replay-control merge receipt/log are committed, rebased, pushed to `main`,
and both workflows are green, run only the candidate explicit merge. Local
generation requires both published merge receipts; aggregate access remains sealed.
