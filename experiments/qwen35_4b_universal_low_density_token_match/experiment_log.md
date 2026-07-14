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

## 2026-07-13 — replay-only exact-token control training

- Ran `scripts/run.py --stage train-control` from design commit `740f30a3`.
- The wrapper reauthenticated the parent replay-refresh adapter, the checked-in
  `replay_repeat` bytes, and the zero-skip token receipt before launching training.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,380.519
  wall seconds. Final training loss was 0.4069.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `bb4f0f8d35ce51e59fb06e8fc835ef043ac8960a5c178e6a511ec75c0a622a07`.
  Adapter config SHA-256:
  `8a89c0cc0ec7d7d9db475479a916c51c8b442c46b354991aa2eb27ab91017f17`.
- Durable receipt: `runs/training/replay_repeat.json`, SHA-256
  `e6f041e8b77dee0c80a625a7191a8609fed914238f3017e68ecb8f9517d6be5c`.
  Full log: `runs/training/replay_repeat.log`, SHA-256
  `1083db586b0278745930eadb6088ca9c4b64f72d77d188714d33d5c340e76f1a`.
- No local or benchmark evaluation was performed at this checkpoint.
