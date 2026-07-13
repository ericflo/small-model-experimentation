# Adversarial Mechanics Result Review

**Review date:** 2026-07-13

**Scope:** authenticated mechanics only

**Model calls during review:** zero

**Verdict:** terminal `INVALID_INTERFACE_PARSE`; qualification remains sealed

This review was performed after the immutable live transactions completed and
the frozen analysis-only stage emitted the mechanics summary. Three independent
read-only audits covered transaction integrity, the package-inventory anomaly,
and scientific/gate interpretation. The result-bearing implementation was not
changed.

## Authentication audit

All five receipt-last invocations were complete before engine teardown:

- systematic, deranged, duplicate, and placebo: 96 rows each;
- noncausal program ceiling: eight rows; and
- total authenticated model rows: 392.

The audit independently recomputed every STARTED, prepared-request, raw,
metadata, and COMPLETE hash. It matched the implementation lock
`12c298aef3ba9cc83bd4d1cdadc304aa0daa7d71bde4be3fce81514fdf4b3148`,
live-preflight hash
`8510c7a449a179174c588562b2cf65136f88202d40eb10e45bc02ee2a06a9416`,
runner hash
`50248e42323a832a931781840eac4bc3817c00ed6955dd74d914ac96e2062e41`,
and pinned Qwen3.5-4B revision. No symlink, partial transaction, ordering change,
seed change, prompt-token mismatch, decoded-text mismatch, or package/engine
sidecar mismatch was found.

The fresh-process frozen analyzer produced `MECHANICS_AUTHENTICATION_PASS`.
Its receipt SHA-256 is
`921699b45e585e4990defb3c81fd95334c52d71f953446c537d21a3f640bcac7`;
the summary and every scored-file hash bind to it. Resampling is neither needed
nor allowed.

## Independent gate recomputation

| Arm | Parse | Parse gate | Cap contact | Cap gate |
| --- | ---: | ---: | ---: | ---: |
| systematic | 87/96 = 0.9062 | pass | 9/96 = 0.0938 | fail |
| deranged | 87/96 = 0.9062 | pass | 10/96 = 0.1042 | fail |
| duplicate | 70/96 = 0.7292 | fail | 22/96 = 0.2292 | fail |
| placebo | 51/96 = 0.5312 | fail | 39/96 = 0.4062 | fail |
| program ceiling | 8/8 = 1.0000 | pass | 0/8 = 0.0000 | pass |

Every diagnostic arm exceeds the preregistered 0.05 cap-contact ceiling, and
duplicate/placebo also miss the 0.90 parse floor. The formal interface failure
therefore does not rest on one borderline row or one control.

The adherence component independently passes:

- systematic supplied and registered execution: 84/96 = 0.875;
- deranged supplied execution: 84/96 = 0.875;
- deranged registered execution: 0/96;
- systematic-minus-deranged registered effect: +0.875;
- systematic context counts: 21, 22, 20, and 21 of 24;
- deranged-supplied context counts: 21, 22, 22, and 19 of 24;
- all four contextwise gates pass;
- systematic success covers all 24 operations and all four contexts; and
- duplicate registered execution is 4/96, placebo 0/96.

The eight-row program ceiling independently fails overall:

- strict parse: 8/8;
- visible pass: 3/8 = 0.375, below 0.50;
- parameterized visible pass: 2/4 = 0.50, exactly at its stratum gate; and
- answer-cap contact: 0/8.

The three successful programs began with `sort_desc`, `mul_k(3)`, and
`take_k(3)`. Independent 24² public-data enumeration reproduced every
pass/fail. Five of eight generated programs retained the supplied first
operation, but only three completed a visible-correct second operation. The
summary correctly exposes `hypothesis_adherence_valid=true`,
`correct_hypothesis_ceiling_valid=false`, and the preregistered decision
priority `INVALID_INTERFACE_PARSE`.

## Package teardown anomaly

The original process false-aborted only after every COMPLETE receipt existed.
The lock and real venv both contained `packaging 26.2`, as did every generation
sidecar. During vLLM extension setup, `setuptools` appended
`setuptools/_vendor` to `sys.path`; that directory contains a second vendored
`packaging-26.0.dist-info`. The analyzer enumerated distributions into a
last-write-wins dictionary and selected the vendored copy.

This was not environment drift. The runner had intentionally snapshotted the
real package set before vLLM import, and `importlib.metadata.version("packaging")`
continued to resolve 26.2. The already-frozen `--stage analyze` entry point in a
fresh process reauthenticated existing rows without constructing a model. The
locked implementation remains unchanged; the reusable template now resolves
versions through normal distribution lookup and has a vendored-duplicate
regression test.

## Adversarial interpretation

The narrow component finding is real but limited: in four simple one-operation
contexts, changing length-matched early bound text changed which operation the
model executed. Both correct and independently deranged injections achieved
84/96 execution of their own supplied operation, while deranged injections
produced 0/96 registered-target outputs. This is preregistered descriptive
evidence of text-conditioned local routing.

It is not a mechanics pass or deployable capability:

- The diagnostic explicitly commands the model to execute the supplied
  operation. Success can be ordinary instruction following, not hypothesis
  evaluation or trust.
- The placebo is semantically underdetermined: it supplies `unknown` where the
  prompt promises an operation. Its long deliberation and interface failure do
  not provide a clean semantic comparison with systematic.
- Duplicate padding contains repeated neutral filler. Its failures can combine
  candidate identity, padding semantics, and operation-specific seed effects;
  they do not isolate diversity alone.
- The program ceiling has no candidate-blind causal comparator. Its three
  successes receive no causal credit.
- Retaining a correct first operation was insufficient in two more cases, and
  three other cases revised it away. Local routing did not reliably become
  composition.
- No matched-sampling, late-timing, qualification, confirmation, training, or
  benchmark result exists.

The result therefore supports neither internal certainty nor J-space
transport, full-program proposal shaping, installed capability, autonomous
solving, or a matched-compute gain.

## Disposition and successor requirements

Do not rerun, reparse, increase the budget, relax a threshold, or open the
sealed qualification. A distinct successor may test hypothesis-conditioned
residualization:

1. enumerate first-operation hypotheses on fresh procedural tasks;
2. deterministically materialize each candidate's intermediate outputs using
   public inputs only;
3. ask the same Qwen3.5-4B model through a short fixed DSL ABI to infer only the
   residual operation or operations;
4. select with a frozen visible-only verifier;
5. retain direct generation, candidate-blind full-program, duplicate, placebo,
   late-conditioning, and neutral/plain matched-sampling controls;
6. preserve the 0.90 parse and 0.05 cap gates rather than moving them; and
7. include fresh exact-depth-three tasks so a positive is not merely another
   trivial depth-two exhaustive-search result.

That design targets the observed bottleneck: early text can route a local
operation, but the model did not reliably preserve and compose it while finding
the rest of the program.
