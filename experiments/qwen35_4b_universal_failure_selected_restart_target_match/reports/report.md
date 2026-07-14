# Report

## Current status

The model-free design, one authenticated parent rollout, frozen failure selection,
exact-exposure stream freeze, both paired training events, and the separately
reviewed fresh-local protocol, and both current-arm explicit merges are complete. No
benchmark event ran. The fresh local event is a terminal negative and the aggregate
gateway remains sealed.

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
- Restart-candidate composite: 128/128 nonzero merged modules at scale 2; exact
  seven-file tree hash `9f64dc55...4a1b`; merged weight `d704af19...49a9`;
  run/external receipts `2956fa41...8ea7` / `97edeb08...6df6`. This likewise
  authenticates deployment only.
- Fresh local result: parent/replay/candidate scored 17/16/15 correct, parsed
  21/22/25, and contacted the cap 5/4/1 times. Execute+induct+probe subtotals were
  2/2/0 of six. Candidate passed parse/cap/route-abstention mechanics but missed the
  accuracy floor, was 0/2 on each target kind, and lost every strict comparison with
  both controls. Local/promotion hashes are `39fe68b9...de9e` /
  `4c381fbd...6759`; promotion is empty and aggregate seed 78,140 remains sealed.

## Interpretation

Removing the parent's failed trajectory and exactly matching forward tokens,
loss-bearing targets, loss mass, update count, and shared replay rows did not rescue
on-policy failure selection. The candidate learned a shorter, more parseable
termination policy—mean sampled tokens fell to 414 from 448 for parent and 436 for
replay—but correctness fell, and the two probe wins present in both controls were
erased. Clean oracle restarts were therefore in-distribution enough to shape emission
but not to transfer the required execute/induct decisions.

This rejects the complete balanced 52-restart curriculum at this dose, not every
on-policy objective. The next result-separated mechanism should use verified
successful sibling trajectories sampled from the same model on greedy-failure tasks,
so supervision stays within policy support while retaining fresh procedural truth,
exact-exposure replay, and the unchanged local gate.

## Terminal disposition

No further event is authorized in this experiment. Preserve the negative and move
any successor to a fresh directory with fresh collection, training, local, and
conditional aggregate seeds.
