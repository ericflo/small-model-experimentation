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

## 2026-07-16 — Pre-GPU adversarial review (three lenses): NO MAJOR; amendments recorded

Three independent review lenses ran against the frozen design at
ddde0e37 before any GPU stage: (1) preregistration/consequence-rule
soundness, (2) treatment-content correctness — an independent per-row
re-derivation of all 200 model-facing rows from prompt text alone,
0 errors, byte-exact — and contamination (0 banned-vocab hits, 0
overlap across 83 pinned sources, no renamed-structure leak), (3)
pins/reproducibility — every pin live-verified non-vacuous, the
normalized-hash mutation probes all held (slot canonicalization exactly
6 slots; injection/relocation/duplication all fail closed), seeds
grep-fresh, stage ordering and merge refusals confirmed. Verdict: no
MAJOR finding on any lens. Actions taken from the minors, pre-GPU:

- REVIEW AMENDMENT (preregistration): added frozen consequence 4 —
  candidate menders > 0 with ANY control > 0 is DESCRIPTIVE ONLY, no
  mechanism claim; matches the code's existing frozen_interpretation =
  None branch; interpretation-only, no code change, no pin churn.
- ERRATA (design receipt, seed 85): the receipt sentence "zero
  seed-context hits anywhere in the repo" for training seed 85 is
  overstated by its own evidentiary standard: `"seed": 85` appears as a
  per-row data field in
  experiments/qwen35_4b_meta_induction/data/train_shift.jsonl (one
  row), the same class of hit the receipt cites as secondary takenness
  evidence for 84. Not a collision (per-row generation field in an
  unrelated cell vs a torch training seed; program precedent excludes
  per-row fields — the reference cell recorded-and-excluded the same
  class for its seed 83). The receipt is NOT rewritten; this errata is
  the record. Seed 85 stands.
- ACCEPTED INHERITED LIMITATIONS (logged, not fixed here; carried
  byte-identically from the reference cell): (a) rebuild_clean_chain
  verify_inputs unconditionally requires the sibling zero-root cell's
  receipts to exist even though all information is in-cell — a scoping
  fix belongs in a future cell's template; (b) eval_local_vllm.py is
  deferred-pinned only (drift before the local stage is caught by
  clean-pushed-main + post-hoc sha, not a design-time pin); (c)
  authenticate_local_promotion verifies the promotion receipt by
  sha/pointer without recomputing canonical_next_counts — a hand-edited
  receipt could flip verdict 3 to 2 (both non-success verdicts; the raw
  local receipt is committed and sha-pinned so tampering is
  deterministically detectable post-hoc); (d) benchmark ledger: a crash
  between summary write and closed-append wedges (recovery: delete
  summary.json, --resume regenerates byte-identically without new
  gateway calls); no file lock against concurrent invocations
  (single-operator environment; double-consumption detectable as 2
  ledger rows).
- READOUT PRIORS (from the content lens, to carry into interpretation):
  the locate arithmetic is exercised only at canonical indices <= 11
  and never targets steps 4-5 (train/gate symmetric; family episodes
  needing deeper indices are extrapolation); 98.75% of training rows
  teach proposing a candidate that does NOT repair (the intended
  propose-and-let-the-trial-judge discipline, but a strong prior
  against "answer = the fix").

## 2026-07-16 — Sealed event 78163: MECHANISM_ANSWER; cell closed

- Local gate (88056-88059): count_walk PROMOTED (strict totals over
  both controls; retention in-band) but the cell's central bet was
  refuted at the same gate: think tokens still at the 1,024 cap
  (median) with 25/40 truncations, fidelity 7/40 = 0.175 << 0.50, and
  the replay control drew 5/40 by itself off the rendered-ranges
  prompt delta. Short thinking does not install at a 160-row dose;
  verbose enumeration is the model's own preference, not a
  walk-pedagogy artifact.
- Sealed medium event at 78163: menders 0.1 with ALL controls at 0.0
  -> frozen_interpretation = MECHANISM_ANSWER (positive precedence
  branch; the failed fidelity precondition scoped only zero readings).
  First candidate-vs-all-controls menders movement in program history.
  Aggregate: candidate 0.3312 tops all arms (replay 0.3298, parent
  0.2950, base 0.0753); goal gate vs base 9 wins / 1 tie / 0 losses.
- Honest scope, recorded at closure: n=1 episode in one sealed seed;
  the reference cell's untreated replay control drew menders 0.1 on a
  different seed; the mechanism that converted is NOT the taught
  compact expression (refuted above) — the dose installed something
  else (candidate-specific: menders +0.1, mirage +0.4, stockade +0.21
  vs replay, all single-seed). Per the program's confirmation doctrine
  the funded successor is an EVAL-ONLY multi-seed confirmation on the
  same committed composites; no capability claim until it replicates.
