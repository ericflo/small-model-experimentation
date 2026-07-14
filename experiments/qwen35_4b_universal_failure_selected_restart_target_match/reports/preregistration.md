# Preregistration

## Frozen identities

- Experiment: `qwen35_4b_universal_failure_selected_restart_target_match`.
- Only model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: published `replay_after_close` explicit composite from
  `qwen35_4b_universal_on_policy_prefix_repair_token_match`.
  Tracked/external/full-weight hashes are `bc78f332...d550`,
  `aa763255...45a3`, and `7ab4c419...36e2e`; the full weight is 9,078,620,536 bytes.
- Construction/rollout/selection/training/local/conditional-aggregate seeds:
  `77114/66114/55114/48/88010/78140`.

## Frozen collection substrate

- 624 fresh executable procedural tasks: 48 each for induct, execute, select, trace,
  verify, count, repair, optimize, abstain, state, order, probe, and route.
- Truth-bearing source SHA-256: `81edc9eaee6cd7f48d1dfd5917a7e7ae364a2ad7ee0e0c3cb81d3f38202de304`.
- Model-facing input SHA-256: `253826897972493a09a10fed96f8a5d5d12f4e98c890bae0d7eabfc642750f5b`.
  It contains only `id`, `messages`, and public metadata; think, answer, and truth
  audit fields are absent.
- Manifest SHA-256: `3a6c4e61c0cf7cdb2f8f52e0c0cdf90487c2c3f8f36dd0f7b902051f953ef937`.
- Canonical-message overlap is zero against 314 predecessor collection/local
  messages and 260 regenerated messages from reserved local seeds 88,000–88,009.
- Fresh local seed 88,010 is reserved but not materialized.

## Frozen parent event

All 624 prompts run exactly once in one experiment-local vLLM event over the explicit
merged parent: natural thinking, greedy decoding, `n=1`, seed 66,114, 1,024 output
tokens, 4,096 context, 16 maximum sequences, 8,192 batched tokens, and CUDA graph
sizes 1/2/4/8/16. Runtime LoRA is forbidden. Runner input, runner bytes, parent
receipts, Git HEAD, clean pushed `main`, sampling, environment, and token counts must
be bound into the output receipt.

## Frozen failure selection

- Eligibility is the union of cap contact/unclosed thinking, missing answer, wrong
  exact answer, and more than 128 thinking tokens.
- Cap, missing, and wrong-answer cases are hard failures and rank before correct but
  over-budget cases. Within a tier, longer thinking ranks first and SHA-256 of
  `55114:skill:task_id` breaks ties.
- Exactly four rows per each of the 13 skills are selected. There is no borrowing,
  replacement, threshold adjustment, or outcome-aware class weighting.
- If any skill exposes fewer than four eligible rows, preserve the complete inventory,
  emit `STOP_INSUFFICIENT_FAILURES`, create no training source, and stop.
- Every selected row uses the original user message plus the existing executable
  oracle think and answer. No parent token, text prefix, hidden answer, or rollout
  transcript is inserted into its training context.

## Prospective paired compute

Training is not authorized by this preregistration. If all quotas pass, a second
model-free checkpoint must copy the clean replay source into this result directory,
tokenize the observed restarts, and either construct or fail to construct:

- 320 rows per arm;
- 200 byte-identical, position-aligned shared replay rows;
- candidate variable block: 52 clean restarts plus 68 replay fillers;
- control variable block: 120 disjoint replay rows;
- exact equality on total forward tokens, nonzero target tokens, and absolute loss
  mass, with zero skipped rows;
- one epoch, batch size one, gradient accumulation eight, 40 updates, LR `1e-5`,
  thought/close weights `0.2/0.2`, and training seed 48;
- both arms warm-started independently from the same published replay adapter.

An exact solver failure is `STOP_EXPOSURE_MATCH_INFEASIBLE`; do not pad, duplicate,
truncate, mask, or alter oracle targets post hoc. A second adversarial compute review
must be committed and green before the control arm is exposed.

## Promotion

Fresh local seed 88,010 will retain the established 26-row, two-per-skill protocol and
same-vLLM merged-composite geometry, but it is not yet materialized. The sole
candidate must pass all absolute parse/accuracy/cap and execute/induct/probe gates,
then strictly beat both unchanged parent and replay control on total correct and on
the combined six target rows. Ties fail.

Only a local pass may open one aggregate-only event at seed 78,140. It must strictly
improve the aggregate and every reported public family score, then survive a
result-separated higher-tier confirmation and matched-compute sample-more. No broad
event can supply training or selection data.

## Mandatory checkpoint order

1. Publish this design on `main`; fetch/rebase, rerun smoke and `make check`, push,
   and verify both required workflows green.
2. Run only `collect-parent`. Preserve all output, then repeat the complete publish
   gate.
3. Run only model-free `mine-restarts`. Preserve a quota pass or stop, then repeat
   the complete publish gate.
4. If quotas pass, freeze the self-contained exact-exposure streams and second review,
   publish them, and only then train the replay control.
5. Give every later training, merge, and evaluation event its own checked, rebased,
   pushed, two-workflow-green checkpoint.

Rebase conflicts require regenerating derived artifacts from the combined source tree
and rerunning all checks. No dirty, unpushed, or non-`main` expensive stage is valid.

## 2026-07-14 — Frozen post-collection receipt

The single parent event completed from pushed-green commit `1744e753` without
recovery or rerun. It wrote 624/624 completions and 304,013 sampled tokens at 879.9
tok/s. Rollout/metadata/log/receipt hashes are `4bf15134...1099f`,
`b43b3a0...1206d`, `668e9b70...369ff`, and `1d35c63a...2b381`.
`benchmark_data_read=false`; local and aggregate seeds remain sealed.

This amendment records the preregistered event only. It does not change selection
rules or authorize training. After this receipt is committed, rebased, pushed, and
green in both workflows, the next authorized stage is model-free `mine-restarts`.

## 2026-07-14 — Frozen post-selection receipt

The unchanged model-free selector found 602 eligible rows, including 228 hard
correctness/cap failures, and cleared every fixed skill quota. It selected 52 rows,
four per skill: 40 hard failures and 12 correct but over-budget rows. The budget-only
rows are confined to skills with fewer than four hard failures: abstain, count, route,
and select. Inventory/restart/selection/summary hashes are `c19d3de7...66240`,
`022b1ea4...d951f`, `567d6b02...b662`, and `2e8a2192...e28ddf`.

Every restart is reconstructed from executable source truth at the original prompt;
no parent prefix is present. This result authorizes no training. The next checkpoint
must copy replay self-contained, tokenize all sources, solve exact forward/target/loss
mass equality without modifying targets, and pass the second adversarial review.

## 2026-07-14 — Frozen exact-exposure receipt

The replay copy and predecessor partition hashes are `25a9595f...f0c2` and
`abf8b505...0966f`. Exact trainer encoding plus an integral solver produced the
preregistered 200-shared / 52-restart / 68-filler / 120-control layout. Source-token,
stream-manifest, control, candidate, and final independent-receipt hashes are
`ac9b9c8a...0bd6`, `7ba55045...91de1`, `7a8d4566...b5078`,
`28deb20e...3190`, and `52a761ef...170`.

Both arms encode 320 rows and zero skips at exactly 297,731 forward tokens, 126,796
nonzero target tokens, and absolute loss mass 27,632.8. They retain exactly 200
byte-identical position-aligned rows. No target was modified, duplicated, truncated,
or post-hoc masked. The second review verdict is `PASS_CONTROL_TRAINING`; after this
freeze is published and green, it authorizes only the replay control. Candidate
training still requires a published-green control receipt.
