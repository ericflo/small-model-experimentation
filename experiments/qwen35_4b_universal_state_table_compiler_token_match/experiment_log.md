# Natural-Language State-Table Universal Curriculum Experiment Log

## 2026-07-14 — Intake

- Opened only after the staged-search negative was preserved in commit `d68f0042`,
  pushed to `main`, and both GitHub workflows passed.
- Ran related-work discovery and named
  `qwen35_4b_universal_search_scaffold_token_match` as the closest near-duplicate.
- Selected the existing `agentic_breadth_installation` program and the authenticated
  `close_xi` parent; the failed scaffold adapter will not be inherited.
- Reserved fresh construction/training/local/conditional aggregate seeds
  77112/46/88008/78138.
- Authorized CPU feasibility and adversarial design review only. No GPU, merge, local
  capability, or benchmark event ran.

Next: publish and CI-verify this intake, then implement and adversarially review the
smallest truth-audited generator and exact-token control before any training.

## 2026-07-14 — CPU feasibility and design freeze

- Began only after intake commit `a9689c52` was pushed to `main` and both GitHub
  workflows passed.
- Deterministically generated 80 rows at construction seed 77112: 20 each execute,
  score, repair, and commit; source SHA-256 is `a7b453af...e88bb`.
- Recomputed every transition, answer, hypothesis prediction/score, and first repair
  error. Correct hypothesis position is balanced 7/7/6 and every false hypothesis
  matches 1–4 of five probes.
- Materialized 320-row replay and candidate arms at exactly 286,814 forward tokens,
  zero skips, 40 planned updates, and 200 byte-identical aligned replay positions.
- Froze replay/candidate hashes `2727e29a...a2b5` / `8e1b8fdc...1355`; token/design
  receipt hashes are `163e40a6...f0b8` / `0bac3340...ef837`.
- Proved all absolute local gates reachable and added fail-closed strict wins over both
  controls overall and on execute/induct/probe. Aggregate seed 78138 stays conditional.
- Adversarial review returned `PASS_EXPENSIVE_RUN`. The harness permits one expensive
  stage per clean, incrementally committed checkpoint. Frozen smoke passes 48 tests.
- No model/GPU, local capability, merge, or benchmark event ran.

Next: publish and CI-verify this design checkpoint, then train only the active replay
control.

## 2026-07-14 — Replay control training

- Began only after design-freeze commit `956a307a` was pushed to `main` and GitHub
  Validate Repository run `29336429602` and Publish Research Site run `29336429945`
  both completed successfully.
- Fetched and fast-forwarded the clean worktree to current `origin/main` commit
  `e21aa1bb` before starting; the incoming change was unrelated to this experiment.
- Re-ran the frozen generator, exact-token materialization, design receipt, and all 48
  experiment tests before loading the authenticated `close_xi` warm start.
- Trained `replay_after_close` for the frozen 320 rows, one epoch, 40 optimizer steps,
  seed 46, and ordinary thought/close weights 0.2. All 320 rows encoded and zero
  skipped.
- Completed in 294.1 wrapper wall seconds with final train loss 0.4226. This loss is
  operational evidence only and is not a capability comparison.
- Preserved receipt/log hashes `b05dc72e...e99a` / `5f4d1fe3...60ba`. The external
  169,903,320-byte adapter weights/config hashes are `83a741e4...409a` /
  `13838f2e...843` and independently match the receipt.
- No candidate training, local generation, merge, or benchmark event ran.

Next: publish and CI-verify this replay-control checkpoint, then train only the frozen
candidate.
