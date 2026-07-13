# Qwen3.5-4B Early Text Hypothesis Forking Experiment Log

## 2026-07-13 — Scaffold

Created from synchronized `origin/main` as the deployable successor to the
terminal-invalid late semantic-anchor experiment. The initial design freezes
early systematic, early duplicate, late systematic, and matched sample-more
arms; no model has been loaded.

## 2026-07-13 — Adversarial redesign before GPU use

Two independent reviews rejected the initial type-only bank. The frozen design
now supplies all 24 bound operations, exhaustively audits the 24² grammar,
requires visible-equivalence of every public-data fit, uses a strict Python AST
answer ABI, and adds independent-prefix equal-total and equal-post late arms.
Duplicate, exact-scaffold placebo, neutral/plain matched-sampling, and CPU
exhaustive controls are mandatory. Gold-mutation, resource-matching, composed-
map, and token-stitching audits were promoted to pre-GPU gates. No model outcome
was observed before these changes.

## 2026-07-13 — Refreshed CPU smoke passes

Regenerated the complete 48/96 split after bound-operation hardening. The smoke
exhausted all 576 programs per task, found zero readable-ancestor behavior
collisions, verified 24 distinct consequences in each of four diagnostics,
serialized 144 unique composed branch maps with balanced gold slots, and froze
the pre-grade mutation/resource firewall. The experiment-local test suite passed
31 tests and 33 parameterized subtests. `model_loaded=false`,
`outcomes_loaded=false`, and all model stages remain fail closed.

## 2026-07-13 — Pre-model mechanics amendment

Implementation-level adversarial review found that the unspecified four-case
program ceiling happened to cover only parameter-free first operations. Before
any model construction, generation, or outcome, the design was amended to
eight cases: four parameter-free plus `add_k(-2)`, `mul_k(3)`, `take_k(3)`, and
`rotate_k(2)`. The ceiling now requires `.50` visible pass overall and within
the parameterized stratum, strict `.90` parse, and at most `.05` cap contact.
It is explicitly non-causal reachability evidence. The amendment also freezes
per-context adherence gates, exact terminal-token matching for padded controls,
authenticated receipt-last generation, and a conservative live KV
no-preemption gate. No threshold was relaxed.
