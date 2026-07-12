# Qwen3.5-4B Balanced-Core Answer-Potential SFT Experiment Log

## 2026-07-12 — Resource-Constrained Fork

The parent long-horizon experiment was paused after 331/1,080 train tasks because observed throughput made
the remaining nine-family schedule incompatible with the user's time budget. No saved shard was lost:
21,184 traces and 97,883,041 sampled thought tokens are present behind atomic SHA-256 receipts.

The user selected the balanced-core funnel. Repository lifecycle rules require a new experiment because
calibration results are already visible. This fork declares those observations, selects the already-leading
three complete family blocks, adds shortest-natural as the strongest calibration control, replaces slow HF
train scoring with a broader parity-gated exact vLLM readout, drops pivot branches, and makes full evaluation
conditional.

At this boundary the GPU is idle. No remaining harvest task, training score, R1 train rollout, SFT update,
or evaluation generation has run under this experiment.

## 2026-07-12 — Immutable Design Anchor

- Prospective README, preregistration, adversarial review, full restartable harness, frozen data, and 40
  passing CPU tests were committed at original `c847615f` before any experiment GPU call.
- The configured guard now points to that commit and its three exact file digests. It fails before model
  load if ancestry or content identity changes.
- After the anchor, the README's first relative link was moved below the generated summary paragraph so the
  repository catalog resolves it from the correct directory. This is a navigation-only repair; the guard
  still verifies that byte-exact prospective design.

## 2026-07-12 — Concurrent-Main Rebase

- Rebasing over three concurrent site commits changed the design anchor from original `c847615f` to
  `cb3d64e3`. All three frozen-file SHA-256 values remained identical.
- The configured ancestry pointer was re-anchored to the rebased commit without changing any design text,
  threshold, split, arm, or code. No experiment GPU call had run.

## 2026-07-12 — Balanced Harvest And Scorer Instrument Stop

- Imported all 331 parent shards at the frozen index digest, then completed exactly 29 Runeward-L3 tasks.
  Final pool: 360 tasks, 23,040 traces, 108,759,239 sampled thought tokens, 22,681 natural closes, four exact
  loops, finite priors on every trace, and no task requiring a top-up.
- The registered task-diverse 32-row joint parity gate then failed closed at 0.692447 > 0.15. No training
  score, R1 train rollout, selection, adapter update, or evaluation had run.
- Inspection of the instrument receipt showed the known long-prefix batch-sensitivity boundary: answer gain
  max 0.147865, joint-likelihood mean-token max 0.054477, empty-answer max 0.156281, and parent-normalized
  joint-gain max 0.692447. No threshold or row was changed.
- Added the pre-outcome amendment to retire vLLM bulk likelihoods and use the single-context Transformers bf16
  reference uniformly. The failed receipt remains evidence; all generation and later evaluation stay on vLLM.
