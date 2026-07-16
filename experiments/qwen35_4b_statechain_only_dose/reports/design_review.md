# Adversarial Design Review

Lifecycle 15's funded successor: the proven statechain install alone.
Four independent lenses (preregistration-versus-implementation; corpus
validity with hand-simulation of the two NEW formalisms; gate statistics
and promotion logic; stage safety and the sealed seed) ran with
adversarial verification of non-minor findings.

- ZERO blockers and zero majors. Two lenses returned clean; two returned
  one MINOR each, both fixed before freeze:
  1. The gated `parsed` retention band's input was not schema-validated —
     a missing key or non-string value silently defaulted instead of
     aborting. `validate_receipt_layout` now requires the key explicitly
     (None or string) on every row; three new tests pin missing-key,
     non-string, and legal-None behavior; the local design receipt was
     regenerated after the code change (all eight gate task files
     byte-identical; `--check` twice).
  2. The preregistration overstated the hidden-updates floor (≥5) versus
     the enforced generator contract (≥3; the shipped corpus measures ≥5
     everywhere) — the frozen text now states both honestly.
- Lens results otherwise: the two new formalisms (peatstove, muletrack)
  hand-simulate correctly from prompt text alone with genuine
  multi-update state dependence and verified-wrong distractors; every
  parameterized element carries a rendered legality clause with the
  bounded-parameter probe honest; contamination and overlap audits
  reproduce (29 pinned sources, zero hits); promotion logic matches the
  frozen text exactly (strict totals, no kind logic, pooled bands at
  −15/+9/−9 on sums with the replay band independent of the parent — the
  lifecycle-15 failure mode is encoded as a test); the stage DAG is
  fail-closed at every verdict with live refusal drills; the benchmark
  runner carries the hardened seed-boundary pattern; seeds
  77140/55131/67/88033–88036/78154 all grep-fresh with zero
  substitutions.
- Post-fix state: 89 tests green; smoke green end-to-end; corpus
  ab6c7845…; exposure exact zero-delta (1,368,815 / 574,630 / 628,314
  per arm).

**Verdict:** `PASS_EXPENSIVE_RUN`.
