# Local Gate and Merge Adversarial Review

- Gate: 144 frozen rows at seed 88,015 (40 axis holdout, 104 retention), zero
  canonical-message overlap against the inherited corpus, the replay pool, both
  stream files, regenerated construction rows, fifteen prior local seeds, and
  both predecessors' frozen gate files; the model-facing input is oracle-free.
- Promotion: the predecessor's verified logic verbatim with the new labels —
  relative axis wins (total plus 3-of-4 kinds, ties fail), retention
  non-inferiority bands, no absolute per-kind floors; eleven synthetic-receipt
  unit tests including recovery-writer schema parity.
- Merges: one explicit composite per arm via the pinned external merger with
  fingerprint and 128-module checks; the design receipt pins the merge harness
  hash; the eval refuses to run until both merged-tree pins are filled and
  re-authenticates design, model tree, and git state around every engine run.
- The benchmark stage passes the full frozen medium-tier CLI, requires a clean
  pushed main with the promotion receipt committed, and recomputes every arm's
  weights before consuming seed 78,145.

**Verdict:** `PASS_CONTROL_MERGE`.
