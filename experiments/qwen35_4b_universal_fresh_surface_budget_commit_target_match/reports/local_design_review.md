# Local Gate and Merge Adversarial Review

Scope: the frozen 104-task local gate, the merge plan, and the evaluation path that
consumes them.

- Gate tasks: 8 per each of 13 skills from the byte-identical original-surface
  generator at fresh seed 88,013; zero canonical message overlap against all
  training streams, both fresh corpora, the replay pool, and regenerated local
  seeds 88,000–88,012; the model-facing input carries `{id, messages, meta}` only.
- Bars are exact ×4 translations of the predecessor's frozen integer bars and are
  binomially never weaker at the boundary; strict-win definitions, the
  single-winner tiebreak, and the BUDGET-counts-as-route-abstention rule match the
  preregistration and are unit-tested (7 synthetic-receipt tests).
- Because training renders only fresh surfaces and the gate renders only original
  surfaces, candidates take the gate as a surface-transfer test while the parent
  retains same-surface familiarity: the comparison is conservative against the
  candidates.
- Merges: one explicit composite per arm via the pinned external merger
  (`cb9af8b4...`), fingerprint-validated (qwen3_5 / 2560 / 32 / 248,320), 128/128
  nonzero modules required, receipts emitted per arm; the eval refuses to run
  until each merged tree hash pin is filled and re-authenticates design, model
  tree, and git state before and after every engine run.
- Eval geometry equals the preregistration exactly: greedy, natural thinking,
  1,024-token cap, max-model-len 4,096, gpu-mem 0.90, 16 sequences, 8,192 batched
  tokens, CUDA graphs 1/2/4/8/16, four sequential single-tenant engine runs.
- The promotion receipt schema is now identical between its two writers, and the
  benchmark stage authenticates weights and the committed promotion receipt before
  consuming the sealed seed (review fixes 1–5).

**Verdict:** `PASS_CONTROL_MERGE`.
