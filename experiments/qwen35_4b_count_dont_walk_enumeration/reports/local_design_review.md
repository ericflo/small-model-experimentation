# Local Gate Adversarial Review

Backed by the three-lens pre-GPU adversarial review recorded in
experiment_log.md (2026-07-16, commit 1492fbea), which live-verified
the merge refusals, the eval boundary pins, the promotion logic, and
the screen-seed freshness before any GPU stage ran.

## Merge authorization

- Both arms trained clean on the zero-root parent under the frozen
  recipe (1,520 rows each, zero skips; control loss 1.255, candidate
  loss 1.319 — the clean chain's known loss-level property, never
  capability evidence). Published pins filled fail-closed for both arms
  (control at bd237ab1, candidate in this checkpoint), each
  authenticated against the frozen training contract.
- The merge uses the cell's own copied merger (cb9af8b4…) with
  --base-model at the authenticated zero_root composite (tree
  recomputed against 414f5829… pre-merge); no runtime-LoRA path exists;
  the merger refuses without receipts (probed by the pins lens).

**Verdict:** `PASS_CONTROL_MERGE`.

## Local event authorization

- Merge receipts published and self-pinned for both arms; the eval's
  trained-tree pins are filled fail-closed and the boundary
  re-authenticates the zero-root parent and both merges before each run.
- The event: three arms × four frozen oracle-free inputs (count-walk
  axis holdout 88,056 with the per-row fidelity decomposition and the
  non-gating expression_cost reading; retention screens
  88,057/88,058/88,059) in sequential authenticated engine runs;
  promotion logic (single-kind strict totals vs BOTH controls) and
  pooled_k3 bands are the frozen preregistration's, unit-pinned; the
  fidelity readout is recorded-never-gating; no seed can be opened by
  this event.

**Verdict:** `PASS_LOCAL_EVENT`.
