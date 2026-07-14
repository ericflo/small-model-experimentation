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

## 2026-07-13 — 40-row designed arm training

- Ran `scripts/run.py --stage train-d40` from replay-control checkpoint
  `668366b5` after both of that checkpoint's GitHub workflows passed.
- Reauthenticated the same parent adapter and token receipt. The position-aligned
  stream differed from `replay_repeat` in exactly 40 rows and retained the exact
  1,429,053-forward-token exposure.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,362.717
  wall seconds. Final training loss was 0.5128.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `b4ca4c0187797f57ae3259f7de1817be34aad927583c0a8728786c56b40ac4a9`.
  Adapter config SHA-256:
  `d70536028419d199d5ca4a273ad8af18a4819d3a1e40898cec2f74625eb1a964`.
- Durable receipt: `runs/training/designed40.json`, SHA-256
  `820e6df4aec64639e3fb1639a799ba94645c2297810950bc3e5c95291586773f`.
  Normalized full log: `runs/training/designed40.log`, SHA-256
  `9dbb302ee6072b24abd39803ea85b80cab37099e400a51eb3c442ff5876470e8`.
- No local or benchmark evaluation was performed at this checkpoint.

## 2026-07-13 — 80-row designed arm training

- Ran `scripts/run.py --stage train-d80` from 40-row checkpoint `83d9f078` after
  that checkpoint was rebased, fully checked, and pushed to `main`.
- Reauthenticated the same parent adapter and token receipt. The position-aligned
  stream differed from `replay_repeat` in exactly 80 rows and retained the exact
  1,429,053-forward-token exposure.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,334.550
  wall seconds. Final training loss was 0.5864.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `ba82457d127c63662b5b86b4a2e1d94ed18014651b59aefd6512690eef1dabc4`.
  Adapter config SHA-256:
  `6f3710860c78a559d3592d4a43389f0347b171d1f9bfc083737d448e2fe04520`.
- Durable receipt: `runs/training/designed80.json`, SHA-256
  `8f1ab673b7570452474639d351ac8e5d756fc574dce607c221d84edacd3838b5`.
  Normalized full log: `runs/training/designed80.log`, SHA-256
  `23685837e14702657bcb941d34166e9997ed19d5c249f81fe30a3a6c553a9af1`.
- All three arms are now trained. No local or benchmark evaluation was performed
  at this checkpoint.

## 2026-07-13 — fresh local gate negative

- Ran `scripts/run.py --stage local` from pushed checkpoint `f30098ce` after all
  three training arms and their receipts were independently published.
- Evaluated the inherited replay-refresh anchor and all three arms on the same 26
  fresh procedural cases at seed 88,004, greedy decoding, and a 1,024-token cap.
- `replay_repeat` scored 0.500 accuracy, 0.538 parse, and 13 cap contacts;
  `designed40` scored 0.500, 0.538, and 12; `designed80` scored 0.538, 0.615,
  and 10. The inherited anchor scored 0.538, 0.577, and 11.
- Every candidate passed the feasible-route abstention check but failed the frozen
  accuracy ≥0.65, parse ≥0.90, and cap-contact ≤2 checks. The promotion
  receipt therefore contains `eligible: []` and exited nonzero as designed.
- Full local receipt: `runs/local/seed88004.json`, SHA-256
  `5a012f15225d93cca00e7ea974472e3319a66e39b071b54418d323750631fcbb`.
  Promotion receipt: `runs/local/seed88004_promotion.json`, SHA-256
  `fc90df3c56c861819c2bc6f8776fd659d6d37484c4357a3320bc6598288a1da7`.
- The merge stage did not run and benchmark seed 78,134 remains unconsumed.
