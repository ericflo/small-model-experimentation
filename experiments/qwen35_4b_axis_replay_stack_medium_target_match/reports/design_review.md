# Adversarial Design Review

A focused delta review (the pipeline and corpus are inherited from two
predecessors that passed full multi-lens reviews; the corpus's 200/200
independent answer re-derivation carries over byte-identically).

- Parent pins: every occurrence enumerated and recomputed against disk,
  including a full 9 GB tree re-hash; the merged-weights pin cross-checks the
  committed seed-78,144 event summary; zero stale predecessor-era pins remain.
- Corpus inheritance: byte-identical to the donor; the inheritance
  authenticator re-derives the corpus from the copied generator and
  byte-compares.
- Medium-tier wiring: tier/think-budget/seed frozen and deviation-rejected; the
  gateway supports the tier; four historical medium@tb1024 events ran 121–215 s,
  well inside the 300 s budget; the "8 of 92" power statement reproduces exactly
  from the public gauntlet logs.
- Gate: seed 88,015 consistent across all six scripts; the overlap receipt
  covers both predecessor gate files, the corpus, the replay pool, regenerated
  construction rows, fifteen prior local seeds, and both stream files; both
  promotion writers are one shared function with a parity test.
- Seeds 55,119 / 53 / 88,015 / 78,145 verified fresh repo-wide.
- All TODO-pins verified fail-closed by direct invocation; the full run battery
  (compile sweep, six --check regenerations, smoke, 41 tests) exits zero.

Minor notes: one dead-code guard in the benchmark CLI; the smoke path
intentionally skips the None-pin abort while every consequential path enforces
it; the inherited generator docstring narrates predecessor history by design.

**Verdict:** `PASS_EXPENSIVE_RUN`.
