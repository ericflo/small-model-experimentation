# Zero-Root Lineage Rebuild Experiment Log

## 2026-07-16 — Model-free design freeze

- Opened as the map-completion's strongest remaining bet and the owner's
  provenance question made measurable: the six documented stages
  replayed from a zero root (the undocumented C53-era blend adapter
  omitted fail-closed as the treatment), merged, and measured once at
  medium beside the original at sealed seed 78,159.
- The lineage package copied byte-identically; the stage plan test-pinned
  against the manifest with the chain rewired to the zero root; the
  hardened single-seed runner carries the receipt-pinned ledger and the
  discovery-pinned implementation signature. 84 tests green; smoke green.

## 2026-07-16 — Review fix: normalized-hash pin for run_benchmark.py

- The adversarial review confirmed a MAJOR, mutation-verified live: the
  original substring-contract mechanism pinned 23 constants/def-lines
  but ZERO call sites — deleting `require_todo_pins_filled()`,
  `require_verdict(...)`, the `require_clean_pushed_main` block,
  `append_ledger(opened_record())`, the `ledger_plan` call, or
  neutralizing `require_zero_root_provenance()` left `gen --check`
  green. A drifted runner could have consumed sealed seed 78159 with no
  verdict gate, no pin refusal, and no ledger.
- Fix (the stronger option): a NORMALIZED-HASH code pin. A deterministic
  regex canonicalizes exactly the three TODO-pin VALUE slots — the two
  `    "zero_root_hygiene_explore": None,`-shaped dict entries and the
  `ZERO_ROOT_MERGE_RECEIPT_SHA256 = None` constant, each also matching
  the post-fill quoted 64-hex — to the fixed placeholder
  `__ZERO_ROOT_TODO_PIN__` (fail-closed on any slot-count mismatch), and
  the sha256 of the canonicalized bytes is frozen as
  `RUN_BENCHMARK_NORMALIZED_SHA256` = `a2d87408efe346a9…` in
  `gen_design_receipt.py` and in the receipt
  (`run_benchmark_normalized_pin` block: digest + the full normalization
  rule). Every byte outside the three value slots is byte-frozen pre-
  and post-fill; six call-site substring contracts remain as
  belt-and-braces diagnostics only. Runner and generator docstrings
  updated to state the mechanism exactly.
- Verified: receipt regenerated (sha `7b822095…`), `--check` twice
  byte-identical; LIVE drills — deleting the `require_todo_pins_filled()`
  call and the `require_verdict(...)` call each made `--check` refuse
  (exit 2, normalized-hash mismatch), and simulating the post-merge pin
  fill with real 64-hex values PASSED `--check` byte-identically; 13 new
  unit tests (96 total, green) cover None-vs-filled hash equality, all
  six reviewer mutations as regressions, one-byte drift anywhere, and
  fail-closed slot drift; smoke green; stage refusals intact.

## 2026-07-16 — Review: the pin mechanism hardened; rebuild authorized

- The runner lens mutation-drilled the deferred substring contracts and
  proved them vacuous for control flow (deleting the verdict gate and
  pin refusal left every check green). Replaced with the normalized-hash
  pin: three canonicalized fill slots, everything else byte-frozen, the
  six mutations now regression tests. Fidelity lens clean. 96 tests
  green; PASS_EXPENSIVE_RUN and PASS_REBUILD granted.
