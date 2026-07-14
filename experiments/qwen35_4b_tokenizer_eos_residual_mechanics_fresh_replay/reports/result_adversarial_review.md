# Adversarial result interpretation review

**Verdict:** `PASS_SCIENCE_INTERPRETATION`

**Date:** 2026-07-14

**Reviewed visible commit:**
`5bce02313a19f6fb94868843e44af628b639ad0a`

**Reviewed hidden-result SHA-256:**
`71090626bcea3f8fb0ef1d3b802f2535107bfba6419a0056c3fae7bb5f1245a6`

The independent reviewer inspected the preregistered design and aggregate
transport, visible, and hidden receipts. It did not inspect a raw generated
bundle, hidden key/ciphertext content, benchmark, GPU, or model output beyond
those aggregate receipts.

## Findings

- The selected interface was a healthy protocol instrument: calibration was
  48/48, transport was 24/24 exact and parse with 12/12 per arity, mechanics
  parse was 98.78--99.83%, and every arm had zero cap contacts. This proves
  syntax, termination, and prompt plumbing, not semantic comprehension.
- All five primary arms had the correct 24-task denominator and scored 0/24
  selected success and 0/24 oracle proposal coverage. There was no correct
  proposal for a different selector to recover.
- Sampled-token matching used 16--17 direct candidates per task and logical-
  token matching used 30--34; neither direct pool exhausted.
- Exhaustive enumeration found 88 visible-consistent programs across the 24
  tasks, and every one was hidden-correct. All 24 tasks therefore had a valid
  program inside the registered 13,824-program class; the visible filter did
  not exclude a valid-but-generalizing solution.
- File-byte SHA-256 `c64dd163...c965ed` and canonical-object digest
  `a62bd73e...04a03` are consistent but must remain distinctly labeled.

## Interpretation boundary

Selector-only follow-up is unsupported. Straightforward larger sampling is not
mathematically disproven: the non-primary 96-sample direct diagnostic found a
visible-consistent program on one task. But the experiment produced no
materialization-specific signal that warrants more brute-force allocation.
Retire this exact prompt/interface/selector mechanism. Do not generalize the
negative to every possible materialization or qualitatively different causal
intervention.

## Access accounting

- Model calls: 0
- GPU calls: 0
- Raw generated bundles read: none
- Hidden key/ciphertext read: none
- Benchmarks read: none
- Other models accessed: none
- Repository edits by reviewer: none
