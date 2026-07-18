# Compute Review

Scoped to the single training event (LoRA r32/a64 on base_reserialized
from the 1008-row STACK corpus = union of the two verified positive
curricula). No new generation — both source corpora were already
adversarially reviewed and execution-verified in their own cells; this
cell combines them.

- Corpus CORRECT and provenance-clean: union 2462c93e… = 504 self_repair
  (source 920cb228…, verified) + 504 why_comment (source 040be350…,
  verified), deterministically interleaved (seed 93570; two rebuilds
  byte-identical; I independently re-verified both source shas, the
  1008/504+504 split, the interleave, and the union sha). Contamination
  on the union: 0 banned names, 0 distinctive shared 7-grams (code-only,
  matching both parents).
- Vehicle: fresh r32/a64 adapter via vendored train_think.py (e0eca2a2…)
  on base_reserialized (tree 26d8ee48…, weights b654e033… confirmed),
  epochs 4 (the stack contains why_comment rows which need 4 epochs — the
  epoch-1 recipe undertrained them to loss 6.3), lr 1e-5, batch 1,
  grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, seed 93571.
  Base authenticated fail-closed.
- HYPOTHESIS: the two prior positives are complementary/target-specific
  (why_comment -> HumanEval +5, self_repair -> agentic +2). If the stack
  captures BOTH, the individual weak signals are confirmed real; if flat,
  they were likely noise. Tightened rule (>=3-problem gain); agentic the
  primary follow-on.

**Verdict:** `PASS_CONTROL_TRAINING`.
