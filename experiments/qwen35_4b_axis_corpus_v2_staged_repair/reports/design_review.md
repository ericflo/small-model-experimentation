# Adversarial Design Review

A focused delta review over the reviewed predecessor pipeline; the scientific
core (the v2 generator) received a full independent truth audit.

- Truth audit: an independent verifier with no generator import re-derived all
  210 answers (160 training + 50 gate axis rows) from prompt text alone —
  exhaustive single-step-fix searches for bugfind/bugmend (including an
  out-of-vocabulary append probe against grammar-bound ambiguity), trace
  re-execution for retrace, walk enumeration for explore, decoy checks for
  hygiene — and machine-matched EVERY narrated think line (states, rejected
  candidates, commits) against re-execution. Zero wrong answers; zero
  narration inconsistencies; the v1 asserted-search flaw is structurally gone.
- Shortcuts: nothing at or above 0.8 within-kind on training or holdout.
- Normalization: one frozen implementation applied to both sides of every
  comparison for every arm and both instruments, documented in the receipt,
  unit-tested including the digit/word-merge concern, with per-row
  pre-normalization grades preserved for the preregistered effect reading.
- Pins: parent tree/weights/adapter recomputed from disk (9.08 GB re-hash);
  no stale predecessor pins; seeds 77118/55120/54/88017/78147 fresh.
- Findings and remediations, applied before this freeze:
  - MAJOR: the bugfind gate holdout drew only early bug positions (training
    early-bias leaked into gate construction plus an unlucky draw). FIX: gate
    bugfind rows are now position-stratified five/five across program halves
    with the bias disabled; the corpus itself regenerates byte-identically.
  - MINOR: the kill-rule win flags are now recorded unconditionally in the
    promotion receipt.
  - MINOR (accepted): the generator retains unused v1 machinery and one stale
    section header; removal would churn the pinned generator hash for zero
    behavioral change.
  - MINOR (noted): hygiene gate rows drew 10/10 injected (7/10 co-located,
    which correctly kills the v1 injected-record anti-shortcut); the
    no-injection case is untested at this seed and is recorded as an
    interpretation limit, not a bar change.

**Verdict:** `PASS_EXPENSIVE_RUN`.
