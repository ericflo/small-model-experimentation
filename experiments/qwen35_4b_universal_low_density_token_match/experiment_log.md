# Low-Density Token-Matched Universal Curriculum Experiment Log

## 2026-07-13 — intake, design freeze, and smoke

- Created as the result-separated successor to
  `qwen35_4b_universal_replay_anchor`; the closest near-duplicate is named in
  `idea_intake.md`.
- Copied the parent's checksum-pinned 800-row designed source and 2,240-row replay
  source. No benchmark content was read or copied.
- Froze a common 1,440-row replay core and two 40-row designed halves covering all
  13 skills. Deterministic local-swap selection found two disjoint 40-row replay
  blocks with exactly the same token sums: 16,732 and 16,543.
- Materialized three position-aligned 1,520-row arms. Each has exactly 1,429,053
  forward tokens, zero skipped rows, and 190 optimizer steps.
- Froze training seed 43, local seed 88,004, and aggregate-only quick@1,024 seed
  78,134 before training.
- Adversarial design review passed with the explicit limitation that target-token
  composition differs by mechanism even though forward compute is exact.
- Non-GPU smoke and six dose/local-gate tests passed. No training or new benchmark
  event has run.
