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

## 2026-07-13 — Adversarial implementation audit passes

Three independent code reviews found and closed raw-authentication, lock,
terminal-padding, context-aggregation, parameter-coverage, live-capacity, and
crash-resume defects before model construction. Prepared prompts are now
independently rebuilt; raw text must equal decoded token IDs; natural and
forced continuation accounting, seeds, exact engine/sampling settings, and the
complete pinned package set are authenticated. Full package parity is checked
before tokenizer or engine construction and again against generation metadata.
Receipt-last transactions permit verification-only finalization but never
resample a started-only invocation. The deterministic preparation froze four
96-row diagnostic arms and eight program cases with receipt SHA-256
`2d6b668a6d43e1bd657124c3645d85ea9996d9aaaea8f81225b97472a2f5b292`.
All 39 tests plus 33 parameterized subtests pass, refreshed smoke remains
`CPU_SMOKE_PASS`, and `model_loaded=false`, `outcomes_loaded=false`.

## 2026-07-13 — Implementation lock frozen

The separately published implementation commit is
`a7bd9fbe093b3f02b3ebdecd5ab533b816e133b3`. The receipt-last lock binds that
commit, the original design, amendment `af9c8431`, and the exact 20-file
critical allowlist. Lock SHA-256 is
`12c298aef3ba9cc83bd4d1cdadc304aa0daa7d71bde4be3fce81514fdf4b3148` and
records `model_calls_before_lock=0`. Live mechanics remains unrun at this
boundary.

## 2026-07-13 — Authenticated mechanics stops before qualification

The first locked live call passed engine preflight and wrote immutable complete
receipts for systematic, deranged, duplicate, and placebo (96 rows each) and
the eight-case program ceiling. Independent forensic review recomputed all
request/raw/metadata/receipt hashes and found no partial or mutable transaction.
The authentication receipt is `MECHANICS_AUTHENTICATION_PASS`.

Early bound text exerted broad direct control: systematic executed the injected
registered operation on 84/96 rows, deranged executed that registered operation
on 0/96 but its own supplied operation on 84/96, and all 24 operations and four
contexts had support. Duplicate registered execution was 4/96 and placebo 0/96.
The preregistered adherence subgate passed.

The formal outcome is nevertheless terminal `INVALID_INTERFACE_PARSE`.
Systematic/deranged answer-cap contacts were 9/96 and 10/96 against a maximum of
0.05; duplicate/placebo parse was 70/96 and 51/96 and cap contact was 22/96 and
39/96. Independently, the noncausal full-program ceiling reached only 3/8
visible passes versus the required 4/8, with its parameterized stratum at 2/4.
Qualification and confirmation remain sealed.

The automatic same-process analysis initially false-aborted after engine
teardown because `setuptools/_vendor` introduced a duplicate `packaging 26.0`
distribution after generation. Every generation sidecar had already recorded
the real locked `26.2` package set. The documented fresh-process analysis-only
stage authenticated and scored existing receipts without a model call or
resample. This operational footgun is now documented in `docs/vllm_inference.md`.
