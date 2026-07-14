# Mid-Density Token-Matched Universal Curriculum Experiment Log

## 2026-07-13 — intake, feasibility revision, and design freeze

- Created as the result-separated successor to the exact-token 0/40/80 local
  negative; the closest near-duplicate is named in `idea_intake.md`.
- Copied the authenticated 800-row designed and 2,240-row replay sources. No
  benchmark content was read or copied.
- Rejected the initially proposed 320-row arm before freeze: proportional designed
  rows were collectively shorter than the shortest row-matched replay selection.
- Froze a representative 0/160/240 ladder with a 1,280-row common replay core and
  three all-skill 80-row blocks. Each designed/replay block matches forward tokens
  exactly at 33,613, 34,091, and 33,015.
- Materialized three position-aligned 1,520-row arms. Each has exactly 1,405,510
  forward tokens, zero skipped rows, and 190 optimizer steps.
- Froze training seed 43, local seed 88,005, and conditional aggregate-only
  quick@1,024 seed 78,135 before training.
- Adversarial design review passed after the 320-row feasibility revision. No model
  training or new evaluation event has run.

## 2026-07-13 — replay-only exact-token control training

- Ran `scripts/run.py --stage train-control` from design commit `49e42f0b`.
- The wrapper reauthenticated the replay-refresh parent, checked-in
  `replay_repeat` bytes, and zero-skip token receipt before launching training.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,396.686
  wall seconds. Final training loss was 0.4199.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `7db84c6313fbf479ec7d08334cfa41a1b4883c95a7a0215eb40b2059e55d2ac5`.
  Adapter config SHA-256:
  `12943a035cdbe3cbded978903911fad6e135dddeb1872a1298371269ade3cd4f`.
- Durable receipt: `runs/training/replay_repeat.json`, SHA-256
  `950d30cd3a69ebe1957729e6da099913096f7eb2a2443585c4e25bed6c053eed`.
  Normalized full log: `runs/training/replay_repeat.log`, SHA-256
  `e8a72ef7fde6cdb7d3f642ffeb5e93755a8783f2c1bfe9feaf6f2b7c7a6c83f8`.
- No local or benchmark evaluation was performed at this checkpoint.

## 2026-07-13 — 160-row exact-token arm training

- Ran `scripts/run.py --stage train-d160` from published replay-control commit
  `0ad7ca07`.
- The wrapper reauthenticated the replay-refresh parent, checked-in `designed160`
  bytes, and zero-skip token receipt before launching training.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,390.190
  wall seconds. Final training loss was 0.6606.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `f05c13ae66e19bbd29abbd2b62ae3c1a577642efefbdba435879012bf4494654`.
  Adapter config SHA-256:
  `0cd3ca7c710e48c264fd1d4c019c304ec0b9e5b13098b89a5b4aa0b171191e58`.
- Durable receipt: `runs/training/designed160.json`, SHA-256
  `485e3a76a8ef45d92df0a60dbcc338d5c1d4ddfc53fbb4e83acb10af7e75d258`.
  Normalized full log: `runs/training/designed160.log`, SHA-256
  `34ce943073908585c933991161f84a076a734b98cc8a23c0fc953dde1db995f3`.
- No local or benchmark evaluation was performed at this checkpoint.

## 2026-07-13 — 240-row exact-token arm training

- Ran `scripts/run.py --stage train-d240` from published 160-row checkpoint
  `71d6e641`.
- The wrapper reauthenticated the replay-refresh parent, checked-in `designed240`
  bytes, and zero-skip token receipt before launching training.
- Completed 190/190 optimizer steps over 1,520 rows with zero skips in 1,373.185
  wall seconds. Final training loss was 0.7284.
- Adapter weights: 169,903,320 bytes,
  SHA-256 `9b159156a7fd59f259454427371e7eb6f72dc2bb1f4e51d6cb6c9dc169af0116`.
  Adapter config SHA-256:
  `66fb435d789ce43a741a04cdbceca1f3c133f157310e2751cbead79d7e51531c`.
- Durable receipt: `runs/training/designed240.json`, SHA-256
  `8bc2f0528f75766c0954ec55455cf9873e4396324c55707f8b1a617af707d82d`.
  Normalized full log: `runs/training/designed240.log`, SHA-256
  `3c2e0692742ac13f3712a0825d86fb728ef561eca22c3a551b1d8923aca77838`.
- All three exact-token arms are now trained. No local or benchmark evaluation was
  performed at this checkpoint.

## 2026-07-13 — fresh local gate negative

- Ran `scripts/run.py --stage local` from published all-arms checkpoint `68667bea`.
- Consumed the single registered experiment-owned seed 88,005 event across the
  inherited anchor and all three arms at greedy decode and 1,024 generated tokens.
- Anchor and `replay_repeat` each scored 17/26 accuracy, 18/26 parse, and 9 cap
  contacts. `designed160` scored 19/26, 23/26, and 3; `designed240` scored 17/26,
  22/26, and 5. All candidates had zero feasible-route abstentions.
- Every arm passed accuracy ≥0.65 and the route check. Every arm failed parse ≥0.90
  and cap contacts ≤2. The 160-row arm missed each remaining bar by one case.
- Full receipt: `runs/local/seed88005.json`, SHA-256
  `ca1a33612ba7dcd430c05a90ee3953358b1137b345101ec78e829e64137bffb3`.
  Promotion receipt: `runs/local/seed88005_promotion.json`, SHA-256
  `3bdad3e9e7536972e7d6178484d02098503d5c77a7a14cb530f46af0173ea41a`.
- Gate receipt hashes: replay
  `5e2861186d11a454695a47b9e806391d58950fd12d1dfacc3b8fc7c6a8f39279`,
  160 rows `ee70d0fd88400a4909b4e0b7b45537f2bfe4697ed47521126a56f971bcba03b0`,
  and 240 rows `7fca078a2050586f91d0eaa233932f9991704315175a98fa5a48b3830082dbd8`.
- Eligible list was empty. No merge or benchmark ran; seed 78,135 remains sealed.
