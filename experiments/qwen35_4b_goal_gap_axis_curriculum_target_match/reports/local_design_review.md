# Local Gate and Merge Adversarial Review

Scope: the frozen two-instrument gate, the merge plan, and the evaluation path.

- Gate: 144 frozen rows (40 axis holdout at 10 per kind from the unseen gate
  seed; 104 retention rows at 8 per original skill), zero canonical-message
  overlap against the treatment corpus, the replay pool, both stream files, the
  regenerated axis construction rows, fourteen prior local seeds, and the
  predecessor's frozen gate; the model-facing input is oracle-free.
- Promotion: relative axis wins (total plus 3-of-4 kinds, ties fail) with
  retention non-inferiority bands and no absolute per-kind floors; the logic is
  unit-tested on eleven synthetic receipts including schema parity of the
  recovery writer, and both promotion writers are one shared function.
- Merges: one explicit composite per arm via the pinned external merger with
  fingerprint and 128-module checks; the design receipt pins the merge harness
  hash (the predecessor's contract bug, fixed); the eval refuses to run until
  both merged-tree pins are filled and re-authenticates design, model tree, and
  git state around every engine run.
- Eval geometry equals the preregistration exactly; three sequential
  single-tenant engine runs over the frozen 144-row input; the parent arm is the
  externally pinned `designed_fresh` composite.
- The benchmark stage passes the full frozen CLI, requires a clean pushed main
  with the promotion receipt committed, and recomputes every arm's weights
  against frozen pins or committed merge receipts before consuming seed 78,144.

**Verdict:** `PASS_CONTROL_MERGE`.
