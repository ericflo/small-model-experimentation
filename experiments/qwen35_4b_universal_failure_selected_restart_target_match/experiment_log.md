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
