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
  passing CPU tests were committed at `c847615f` before any experiment GPU call.
- The configured guard now points to that commit and its three exact file digests. It fails before model
  load if ancestry or content identity changes.
