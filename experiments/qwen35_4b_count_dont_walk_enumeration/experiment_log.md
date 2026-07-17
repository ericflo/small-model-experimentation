# Count-Don't-Walk Enumeration Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-16 — model-free construction frozen (lifecycle 27)

- Cloned the entire enumerative-repair reference cell (lifecycle 26):
  scripts, tests, lineage package, frozen vLLM runner, the two-direction
  menders consequences with the 0.50 fidelity precondition, the six-slot
  normalized-pin hardened benchmark runner. Machine simulators, legality
  bounding, canonical order, K_CYCLE, and uniqueness invariants stay
  byte-equivalent (`gen_feedloop_curriculum.py` byte-identical to the
  menders source; `train_think.py`/`merge_adapter.py`/lineage trainers
  byte-identical to their proven sources).
- The ONE designed delta (evidence:
  `experiments/qwen35_4b_enumerative_repair_protocol/analysis/truncation_forensics.md`):
  the expression pedagogy. Think targets are a fixed-shape five-line
  compact computation (count → k+1 → rendered-range lookup with explicit
  offset subtraction → action-list slot skipping the written action →
  emit), constant token cost in k, under a frozen think length budget
  (five-line shape + caps per row in the generator; REAL tokenizer bound
  <= 120 tokens fail-closed in `measure_source_tokens.py` — measured max
  105, mean 95.8 over the frozen corpus). The order statement gains the
  rendered per-step candidate counts, verified against the exhaustive
  enumeration exactly. New non-gating gate reading: `expression_cost`
  (per-arm think-token-length distribution + truncation count on the
  axis rows).
- Corpus frozen: `data/sft_count_walk.jsonl` sha `21e6f5cb…`, 160 rows
  single-kind `u_count_walk` at construction seed 77,191; manifest
  `4343251285…`; zero row overlap against 83 pinned predecessor sources
  including the reference cell's corpus/streams/gates.
- Exposure frozen: exact zero-delta three-axis MILP at namespace seed
  55,171 (HiGHS optimal, 0.4 s) — 1,438,010 forward / 564,379 nonzero /
  621,239 mass×5 per arm, 1,280 aligned core rows, zero skips;
  independent validation receipt `baaa454d…`.
- Gate frozen at seeds 88,056 + 88,057/88,058/88,059 (sealed aggregate
  78,163); `gen_local_gate.py --check` and `check_design.py --check`
  byte-stable across repeated runs; `rebuild_clean_chain.py
  --verify-inputs` green (stage 7 = `count_walk`, training seed 85).
- Seeds verified grep-fresh in seed contexts; ONE next-free substitution
  recorded: training seed 84 is taken (task seed of
  `qwen35_4b_hypothesize_verify_wall`), so 85.
- Note: during design freeze, the `train_think_stage12.py` sha pin in
  `check_design.py` was found altered by one hex character
  (`…8805910c…`) relative to the byte-identical vendored copy's true
  sha256 (`…8805510c…`, matching the lineage manifest, the provenance
  receipts, and lifecycle 22's committed original); check_design
  correctly failed closed, and the pin was corrected to the verified
  file hash.
- Tests: 178 green (`unittest discover`), including the new
  think-budget/constant-shape suite, the rendered-range-vs-enumeration
  suite, and the expression-cost gate-reading suite. `run.py --smoke`
  green end to end. Boundary drills refuse (train/merge/local/benchmark
  stages abort on a dirty tree and on unfilled TODO-PINs).
