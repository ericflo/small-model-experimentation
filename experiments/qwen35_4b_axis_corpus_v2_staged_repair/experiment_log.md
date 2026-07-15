# Axis Corpus V2 Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened from the three-event failure forensics (1,296 graded completions):
  v1 trace-repair failed as an ASSERTED search the model improvised; v1
  protocol was redundant dose; hygiene's residual is co-located injections;
  21 correct answers were whitespace-rejected by the exact-match grader.
- Wrote `scripts/gen_axis_v2.py`: staged repair lessons with DEMONSTRATED
  bounded search (rejected op-TYPE candidates, two checkpoint rules,
  immediate commits) on the best-measured formalism; hygiene co-location
  oversampled; explore unchanged. Corpus balance verified (31/40 injected,
  17 co-located; 25/30 early bugs).
- Reserved fresh construction/slot/training/gate/aggregate seeds
  `77118/55120/54/88017/78147`; the corrected detectability bar and the
  documented answer normalization govern the gate; the kill rule for the
  trace-repair axis is frozen in the intake and README.
- No model, GPU, training, local, or benchmark event has run.

## 2026-07-15 — Model-free pipeline run (freeze → measure → materialize → validate → gate → design)

- Adapted the full stack-trial pipeline (build/measure/materialize/validate/
  train/merge/gate/eval/benchmark/harness) from the stack predecessor plus the
  re-adjudication's corrected detectability promotion logic, retargeted to the
  `axis_on_replay` parent and the five v2 kinds; every fail-closed convention
  kept (hash pins, `--check`, TODO-PIN fail-closed, encoder binding, merge
  self-pin in the gate receipt, shared `finalize_promotion` writer, full
  benchmark CLI, weight recomputation).
- Froze the v2 corpus at construction seed 77118:
  `data/sft_axis_v2.jsonl` sha256 `28d9be20180b017e64eab4749d79eb659089b2bcc12985efbb753f4a66479e79`
  (160 rows; bugfind 30 / bugmend 25 / retrace 25 / explore 40 / hygiene 40;
  balance: 25/30 early bugs, 31/40 injected, 17 co-located), manifest
  `ac333ee1da03e3445cc46cafc863b40988dbea4104c9e5e37c0e406d4eb38cbe`;
  replay blend copied byte-identically (`25a9595f…abf0c2`).
- Measured exact spans (`source_token_lengths.json`
  `5a88c4ea9b0999ce35cbbc552c0dfff67d4d4f398e323117d56c484a61633765`); the
  three-axis MILP solved optimally in 2.8 s: both 240-row variable blocks at
  forward 134,708 / nonzero 58,660 / mass×5 68,780; arm totals 1,357,677 /
  566,115 / 621,987; 1,280 position-aligned shared rows; zero skips. Streams:
  `replay_repeat3.jsonl` `793603eb…d7d19b6`, `axis_v2.jsonl` `1da40090…da27a01`;
  stream manifest `156649d0…fb769d2`; independent token receipt
  `a4721dc5…156b78` (all deltas 0 on the match axes).
- Froze the seed-88017 gate: 154 rows (50 axis holdout + 104 retention),
  source `13ea8441…409b5`, runner input `989b889a…ea561`, design receipt
  `23c1b899…cce2d6`; the receipt documents `answer_normalization`
  (whitespace collapse + strip, then no spaces adjacent to '>' or ';',
  citing the 21 measured whitespace rejections in the re-adjudication
  forensics). Zero overlap against the v2 corpus, the replay blend, both
  training streams, regenerated construction rows, prior local seeds
  88000–88016, and all four predecessor gates (88013/88014/88015/88016).
- Design receipt `data/design_receipt.json`
  `d41ad8cfd6a63322d8070fa23d27f304a0cd7fb77a342056644405a73a864d87`;
  every builder `--check` regenerates byte-identically; 53 unit tests green
  (per-kind answer re-derivation incl. exhaustive-grammar bugfind/bugmend,
  five-kind corrected gate, normalization contract); `run.py --smoke` green.
- Stream pins filled in `train_trial.py` (receipt sha + exposure constants +
  arm stream hashes). Left fail-closed on purpose: `PUBLISHED_ARM_HASHES`
  (both arms) and the eval's `EXPECTED_TREE_SHA256` pins — filled only after
  training/merges publish. No model was loaded; no GPU stage ran.

## 2026-07-15 — Authenticated control training

- `train-control` ran only after freeze commit `516c7b33` matched `origin/main`
  with both workflows green and a clean worktree.
- `replay_repeat3` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed.

## 2026-07-15 — Authenticated candidate training

- `train-candidate` ran only after control checkpoint `a919238c` matched
  `origin/main` with both workflows green and a clean worktree.
- `axis_v2` trained 1,520/1,520 rows with 0 skipped over 190 updates;
  receipt/log published and pinned fail-closed. Merges are next.
