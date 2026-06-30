# Qwen3.5-4B Thinking Separability Probe Experiment Log

## Scaffold

Third experiment of `test_time_reasoning_budget`. Tests whether native thinking makes the model's
own correctness more linearly decodable from its activations — the interpretability complement to
the shuffle-control puzzle (C9) and an internal-signal angle on the C2 selection bottleneck.

## Design / method notes

- Generation reuses the s1-style budget forcing + shuffled-thinking from the scaling experiment, but
  `src/probe_lib.py` returns the FULL clean token sequence (prompt + thinking + </think> + answer) so
  a forward pass can read the answer-token hidden state.
- The sibling experiment did not persist raw thinking tokens (only extracted code), so generations are
  re-run here to obtain the activations under each condition.
- **Right padding** for the activation forward (the sibling generation uses left padding): the
  qwen3_5 linear-attention recurrence is order-sensitive, so left padding would feed pad tokens into
  the recurrence before real tokens; right padding keeps the last-real-token state clean. Signal =
  per-layer hidden state at each sequence's last real token (33 states: embeddings + 32 layers).
- Probe = per-layer standardized logistic regression, GroupKFold **by task_id** (prevents
  task-identity leakage), out-of-fold AUC for full-test pass; bootstrap CI by resampling tasks;
  shuffled-label control must give ~0.5.
- Verification runs in a separate torch-free process (fork-safe sandbox), as in the sibling sweep.

## Smoke

4 tasks x {no_think, think_1024} x k=2 validated extract -> activations (8, 33, 2560) -> verify
(no_think 1.0 full, think_1024 0.75 full). Probe needs the full run's label variance.

## Run notes

- First full run crashed after no_think with `CUDA driver error: device not ready` at
  `torch.stack(o.hidden_states)` — stacking the full [B, 33, ~1000, 2560] tensor doubled a ~3GB
  allocation on the long thinking sequences (no_think survived only because its sequences are ~8×
  shorter). Fix: slice each layer's last-token vector BEFORE stacking (the giant tensor never
  exists) + activation batch 8. Re-run completed all 5 conditions (~1.8h).

## Results (see reports/report.md)

Behavioral verify (full-pass): no_think 0.769, think_512 0.851, shuffle_512 0.791, think_1024 0.848,
shuffle_1024 0.830 (real > shuffle at both budgets, replicating the sweep).

Probe (best-layer AUC predicting full-pass from the answer-token activation): no_think 0.642,
think_512 0.708, shuffle_512 0.733, think_1024 0.720, shuffle_1024 0.755; shuffled-label control
~0.50. Three findings: (1) correctness is moderately decodable (AUC 0.64–0.76); (2) thinking raises
decodability at every layer; (3) shuffled thinking matches/exceeds real thinking → the gain is NOT
coherent reasoning, falsifying the pre-registered hypothesis and converging with C9 at the
representational level. Deployable spinoff: probe vs C2 false-passes (visible-passer AUC) ~0.60–0.68
under thinking, ~chance under no-think.
