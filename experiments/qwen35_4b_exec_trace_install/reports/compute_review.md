# Compute Review

Scoped to the single training event (LoRA r32/a64 on the base_reserialized
composite from the 400-row execution-trace curriculum). Backed by the
two-lens adversarial workflow recorded in experiment_log.md (curriculum
correctness+contamination; pipeline+consequence+standalone) which found
ZERO MAJOR and zero minor findings, plus an independent 30-row
re-execution of the corpus at freeze (30/30 FINAL answers match).

- Training signal is CORRECT and provenance-clean: 400 random-program
  trace rows (corpus 7c5b77ea…) triple-verified by the generator (primary
  interpreter + independent CPython settrace re-execution + safety caps);
  self-generated + execution-verified (no teacher model). Contamination
  audit: 0 banned HumanEval/MBPP function names over all 400 rows, 0
  distinctive shared 7-grams with the benchmark solutions. Max 835 tokens
  < 4096 → zero truncation, trainer's skipped_rows==0 holds.
- Vehicle: fresh rank-32/alpha-64 adapter via the vendored train_think.py
  (sha e0eca2a2…, byte-identical) with --model-path on the base_reserialized
  composite (tree 26d8ee48…, weights b654e033… — recomputed and confirmed
  matching pre-training), epochs 1, lr 1e-5, batch 1, grad-accum 8,
  max-length 4096, w_think 0.2, w_close 0.2, training seed 90211. Base
  authenticated fail-closed; pins fail closed on None.
- The KNOWN RISK (pure-trace SFT catastrophically forgetting code
  generation) is measured, not assumed: the frozen consequence rule's
  RETENTION_FAIL branch triggers on any >0.02 HumanEval/MBPP regression,
  and a --mix-retention switch is prepared if the pure-trace probe tanks
  retention. The retention probe (HumanEval after merge) runs before any
  claim.

**Verdict:** `PASS_CONTROL_TRAINING`.
