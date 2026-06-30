# Qwen3.5-4B Thinking Content vs Compute Experiment Log

## Scaffold

Fourth experiment of `test_time_reasoning_budget`. Adds the foreign-task-thinking control the
separability report flagged as the decisive next test (remove relevance/token-presence, not just order).

## Design / method notes

- Ladder no_think -> foreign -> shuffle -> real at a fixed budget (512). Real thinking is generated
  ONCE (`gen_real` captures the thinking-region tokens); shuffle permutes those tokens, foreign uses a
  cyclically-shifted other task's thinking tokens (same sample slot); both regenerate ONLY the answer
  (`gen_answer`) from the modified prefix. So all conditions share one thinking-token multiset and
  matched thinking length — only relevance (shuffle vs foreign) and order (real vs shuffle) vary.
- Behavioral full-pass + per-layer answer-token separability (reusing the separability experiment's
  right-padded activation extraction + GroupKFold-by-task logistic probe + shuffled-label control).
- Attribution (behavioral full-pass): compute+scaffold = foreign - no_think; relevance = shuffle -
  foreign; order = real - shuffle.

## Smoke

4 tasks x k=2, budget 512: ladder generated + activations (8,33,2560) + verify ran. Tiny-n hint
(NOT evidence): foreign full-pass 0.000 vs shuffle/real 1.000, no_think 0.875 — suggesting relevance
may matter a lot (foreign possibly below no_think). Needs the full run.

## Results (see reports/report.md)

Behavioral ladder (full-pass): no_think 0.764, foreign 0.043, shuffle 0.739, real 0.859. Foreign
collapses (spot-check: task `remove_Occ` fed a matrix-sort thought -> emits `sort_matrix`, i.e. solves
the WRONG problem). Decomposition: irrelevant-content (foreign-no_think) -0.721; relevance
(shuffle-foreign) +0.696; coherent order (real-shuffle) +0.120. So the model uses thinking as content,
and the efficient-budget gain is coherent reasoning over relevant content — CORRECTING the earlier
"mostly compute/scaffold" claim. Separability noisy (no_think 0.682, shuffle 0.636, real 0.676,
overlapping CIs; foreign 0.994 is a 34/800-imbalance artifact). Robust result is behavioral.

Probe.py fix logged: the copied analysis/probe.py initially pointed ACTS at the sibling
separability experiment's large_artifacts dir (loaded wrong no_think, crashed on acts_foreign);
fixed to this experiment's dir.

Remaining arm: filler/pause-token (contentless) to isolate pure compute (foreign adds misleading
content, not contentless compute).
