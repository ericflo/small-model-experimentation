# Compute Review

Scoped to the single training event (LoRA r32/a64 on base_reserialized
from the 504-row self-repair curriculum). Backed by the two-lens
adversarial workflow (repair-correctness+contamination; pipeline+
consequence+standalone) which found ZERO MAJOR and zero minor findings,
plus an independent 25-row re-execution at freeze (25/25: buggy code
fails its tests, corrected code passes).

- Training signal CORRECT and provenance-clean: 504 mutation-injected
  self-repair rows (corpus 920cb228…) triple-verified (buggy fails >=1
  test with a wrong value, corrected passes all, they differ); the
  failure output in each prompt is the REAL execution output.
  Self-generated + execution-verified (no teacher). Contamination: 0
  banned HumanEval/MBPP names over 504 rows, 0 distinctive shared
  7-grams. Max ~489 tokens < 4096 -> zero truncation.
- Vehicle: fresh r32/a64 adapter via vendored train_think.py (sha
  e0eca2a2…) with --model-path on base_reserialized (tree 26d8ee48…,
  weights b654e033… — confirmed), epochs 1, lr 1e-5, batch 1,
  grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, seed 91331.
  Base authenticated fail-closed.
- TIGHTENED transfer rule (fixes bet #1's leniency): INSTALLED_CODING
  requires a >=3-problem gain (not 1); the agentic eval is the primary
  real target (manual follow-on). Forgetting guard: the repair task ends
  by emitting CODE, so it does not bias away from generation; retention
  measured regardless.

**Verdict:** `PASS_CONTROL_TRAINING`.
