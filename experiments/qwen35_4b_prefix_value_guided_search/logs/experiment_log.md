# qwen35_4b_prefix_value_guided_search

## Purpose

This standalone experiment tests whether partial code prefixes expose useful search states. The first gate is an oracle ceiling: if hidden-test oracle prefix selection cannot beat ordinary full-code sampling at matched completion budget, then training a prefix value model is not justified.

## Protocol

1. Implement MBPP execution, full-code sampling, prefix proposal, prefix completion, and prefix-search metrics.
2. Run a smoke pass on a tiny held-out slice.
3. Run a pilot oracle-prefix ceiling against matched full-code sampling.
4. Only train a prefix value selector if the oracle ceiling clears.
5. Generate a report with charts and a gate decision.

## Running Notes

- 2026-06-26: Created fresh standalone experiment and large-artifact directories.
- 2026-06-26: Smoke run on 2 held-out test tasks completed. A summary bug initially reported zeros because per-task metrics were nested; fixed the aggregator and rebuilt the smoke summary.
- 2026-06-26: Pilot run on 12 held-out test tasks completed with matched completion budget: 8 full samples versus 4 prefixes x 2 completions. Full sampling, prefix union, and oracle-selected prefix all covered 9/12. Lexical-selected prefix covered 8/12; random-selected prefix covered 7/12.
- 2026-06-26: Gate decision: strict oracle-prefix coverage gate failed because prefix search did not improve coverage over full sampling at matched completion count. No value model was trained.
