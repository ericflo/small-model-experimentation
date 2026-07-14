# Experiment Log

## 2026-07-14 — Model-free design checkpoint

- Created a result-separated successor to the terminal long-prefix repair negative.
- Chose the published stronger `replay_after_close` merged composite as parent.
- Frozen seeds: construction 77,114; parent rollout 66,114; selection 55,114;
  training 48; local 88,010; conditional aggregate 78,140.
- Materialized 624 fresh truth-audited tasks, 48 per all 13 universal skills.
- Source/input/manifest hashes: `81edc9ea...de304`, `25382689...0f5b`,
  `3a6c4e61...ef937`.
- Verified zero canonical-message overlap against predecessor sources and prior local
  seeds 88,000–88,009. Local seed 88,010 remains unmaterialized.
- Froze task-level failure selection plus clean restart from the original prompt;
  parent prefixes are excluded from training context.
- Froze four rows per skill and a 128-thinking-token deployment budget. Hard
  correctness/cap failures rank before budget-only cases.
- Froze exact future equality on forward tokens, loss-bearing target tokens, and
  absolute loss mass. Training remains unauthorized pending observed quotas and a
  second compute review.
- Adversarial review verdict: `PASS_PARENT_ROLLOUT` only.
- No model call or benchmark read occurred.

## 2026-07-14 — Authenticated parent rollout

- Launched only after design commit `1744e753` was pushed to `main` and GitHub runs
  `29359316770` / `29359317072` both succeeded.
- The explicit merged `replay_after_close` parent produced 624/624 greedy natural-
  thinking completions in one vLLM event at seed 66,114.
- Sampled 304,013 tokens at 879.9 tok/s; wrapper wall time was 394.96 seconds.
- Rollout/metadata/log/receipt hashes: `4bf15134...1099f`, `b43b3a0...1206d`,
  `668e9b70...369ff`, `1d35c63a...2b381`.
- Preflight was clean pushed `main`; runtime dirty state is explained solely by the
  collector opening its durable log before runner metadata sampling.
- Recovery was unused and generation was not rerun. No benchmark data was read and
  aggregate seed 78,140 remains sealed.
- Failure mining was not run or inspected in this checkpoint.

## 2026-07-14 — Frozen model-free failure selection

- Ran only after collection commit `fd08c7fe` was pushed to `main` and GitHub runs
  `29360147608` / `29360147678` both succeeded.
- The preregistered selector found 602 eligible rows and 228 hard failures. All 13
  four-row quotas passed; total eligibility ranged from 40 to 48 per skill.
- Selected exactly 52 rows, four per skill. Forty are hard failures; 12 are correct
  but over the 128-thinking-token budget. Count had zero hard failures, route/select
  one each, and abstain two, so those four skills supplied all 12 budget-only rows.
- Selected overlapping reasons: 29 cap contacts, 26 missing answers, 13 wrong
  answers, and 51 over-budget flags.
- Inventory/restart/selection/summary hashes: `c19d3de7...66240`,
  `022b1ea4...d951f`, `567d6b02...b662`, `2e8a2192...e28ddf`.
- All 52 trainable candidates begin at the original prompt; zero parent-prefix rows
  exist. Training remains unauthorized pending exact three-axis exposure matching
  and a second adversarial compute review.
- No model call or benchmark read occurred during selection; aggregate seed remains
  sealed.

## 2026-07-14 — Exact-exposure feasibility and second review

- Ran only after selection commit `f0d08544` was pushed to `main` and GitHub runs
  `29360679439` / `29360679657` both succeeded.
- Copied the predecessor's 2,240-row replay source and inherited partition manifest
  into the experiment. Their hashes are `25a9595f...f0c2` and
  `abf8b505...0966f`; the training encoder remains byte-identical at
  `0cfb126f...2cc4`.
- Measured every replay and restart row with the exact Qwen tokenizer and actual
  trainer encoder. Source-token receipt hash: `ac9b9c8a...0bd6`.
- SciPy 1.18 HiGHS found an exact integral partition in 4.43 seconds and 801 nodes:
  200 shared replay rows, 52 clean restarts plus 68 replay fillers, and 120 disjoint
  replay-control rows. Solver gap was zero.
- Final stream/manifest hashes are replay `7a8d4566...b5078`, candidate
  `28deb20e...3190`, and manifest `7ba55045...91de1`.
- Independent final-file encoding confirmed 320 rows, 297,731 forward tokens,
  126,796 nonzero target tokens, absolute loss mass 27,632.8, and zero skips in each
  arm. Exactly 200 rows are byte-identical at aligned positions. Final receipt hash:
  `52a761ef...170`.
- Candidate minus control is zero on every registered axis, answer targets, close
  targets, and parent prefixes. It has 16,414 more total zero/nonzero thinking-span
  tokens and 16,414 fewer masked context tokens because the variable replay blocks
  contain different forced-close composition; this is disclosed in the review.
- Second adversarial verdict: `PASS_CONTROL_TRAINING`. It authorizes only the replay
  control after this checkpoint is committed, rebased, pushed, and green in both
  workflows. Candidate training must wait for a separately published control.
- No model call or benchmark read occurred; local seed 88,010 is unmaterialized and
  aggregate seed 78,140 remains sealed.

## 2026-07-14 — Authenticated replay-control training

- Launched only after exact-exposure commit `821d50d4` was pushed directly to
  `main` and GitHub runs `29362464655` / `29362464584` both succeeded.
- The wrapper authenticated a clean pushed `main`, the frozen 320-row replay stream,
  exact token receipt, and original replay-parent adapter before opening outputs.
- Trained exactly 320/320 rows with zero skips and 40/40 optimizer steps at the
  frozen seed and hyperparameters. Trainer runtime was 297.3 seconds, wrapper runtime
  318.70 seconds, and final train loss was 0.3873.
- Receipt/log/config/weight hashes are `3a9cc1ea...6d49`,
  `3bedc341...f25`, `dce1095c...f8f6`, and `5840757d...b1c`. The complete external
  adapter is 169,903,320 bytes.
- The receipt records `benchmark_data_read=false` and a sealed aggregate seed. No
  merge, capability evaluation, or benchmark event occurred.
- Candidate training remains blocked until these tracked results are committed,
  rebased, pushed to `main`, and both workflows are green.
