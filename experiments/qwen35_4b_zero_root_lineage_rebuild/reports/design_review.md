# Adversarial Design Review

The zero-root rebuild: the owner's provenance question made measurable.
Two focused lenses (stage-replay fidelity against the copied manifest;
the runner, consequence partition, and seed safety) with adversarial
verification.

- The fidelity lens returned no confirmed defects: every stage's dataset,
  seed, trainer variant, and full hyperparameter set equals the
  manifest's recorded recipe (including stage 1's missing w_close and
  stage 3's targeted overrides, checked against the trainer sources'
  defaults); stage 1 is genuinely fresh; stages 2–6 chain from the
  zero-root outputs; the root omission fails closed; original hashes are
  contrast-only.
- ONE MAJOR confirmed in the runner's pin mechanism, mutation-verified
  live by the reviewer (deleting the verdict-gate and TODO-pin-refusal
  call sites left every automated check green): the deferred substring
  contracts pinned constants, not control flow. Fixed with a
  normalized-hash pin — the three TODO-pin value slots canonicalize to
  placeholders under a fail-closed slot-count rule and every other byte
  of run_benchmark.py is frozen pre- and post-fill
  (RUN_BENCHMARK_NORMALIZED_SHA256 a2d87408…). The reviewer's exact
  mutations are now regression tests (all six change the hash and refuse
  --check); a simulated post-merge fill passes with the receipt
  byte-identical.
- Post-fix state: 96 tests green; smoke green; receipt 7b822095…
  --check byte-identical twice; stage-refusal drills intact; seed 78,159
  audit fresh (a planted probe was refused live).

**Verdict:** `PASS_EXPENSIVE_RUN`.
