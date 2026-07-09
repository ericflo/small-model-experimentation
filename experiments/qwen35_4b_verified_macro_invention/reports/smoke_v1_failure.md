# Smoke v1 failure: interface unusable, macro hypothesis unresolved

Date: 2026-07-09. Status: preserved failed smoke; not confirmatory evidence.

## Scope

Smoke v1 was the first model-facing check of the verified-macro harness. Every proposal and
solver completion used the experiment-local vLLM runner with the pinned
`Qwen/Qwen3.5-4B` revision. The construction corpus, libraries, and full evaluation had already
passed their CPU gates, but no full task was generated or inspected.

The smoke was designed to answer a narrower question: can the solver return strict programs at
an acceptable rate, and can it use a generator-known macro library at all? It failed that
interface gate. It therefore says nothing about whether mined macros improve full-task
performance.

## Registered gate result

The registered parse and truncation pool contains the matched `base` and `designed_ceiling`
smoke arms.

| Check | Smoke v1 result | Gate | Result |
| --- | ---: | ---: | --- |
| Pooled solver parse rate | 0.5972 | at least 0.50 | pass |
| Base parse rate | 0.6111 | at least 0.50 | pass |
| Designed-ceiling parse rate | 0.5694 | at least 0.50 | pass |
| Pooled answer-truncation rate | 0.40046 | below 0.05 | **fail** |
| Valid designed candidates using a macro | 0 | at least 2 | **fail** |
| Designed hidden oracle coverage | 0/12 = 0.0000 | not below base | **fail** |
| Base hidden oracle coverage | 1/12 = 0.0833 | comparison | — |

The sole base oracle solve was on the no-reuse half of the smoke set. Across all solver arms,
all 1,440 samples force-closed their thinking stage and 607 answer stages hit their cap. The
607/1,440 all-arm diagnostic is broader than the registered 0.40046 matched-arm truncation
metric; the two numbers must not be substituted for one another.

These failures have a coherent interface interpretation. At think@192, the model exhausted the
thinking allowance on every sample, then frequently continued analysis in the answer channel
until the 128-token answer cap. Parsed programs sometimes survived, but no valid designed-arm
candidate called a macro. Because the designed ceiling could not expose even one hidden solve,
the smoke cannot distinguish bad abstractions from a model that never learned to operate the
surface.

## Proposal-stage diagnostic

The original whole-answer proposal parser accepted 0/16 samples. It rejected an entire sample
when the answer contained prose, too many macro lines, or any malformed line. After the smoke
had already failed, a line-local audit recovered 18 train-supported, behaviorally unique
candidate expansions from the same raw outputs.

That reparse is **exploratory only**. It does not retroactively populate `qwen_ranked`, does not
change the v1 verdict, and is not evidence that Qwen discovered a useful library. It establishes
only that the v1 all-or-nothing parser discarded syntactically usable lines, motivating the
predeclared line-local extraction rule in the v2 proposal interface.

## Decision

The original preregistration explicitly routed a failed interface gate to a repair on a fresh
smoke set. Smoke v1 and its exact code, config, data, raw vLLM sidecars, and analyses are retained
under the versioned v1 archive. The repair is frozen separately in
[`preregistration_amendment_1.md`](preregistration_amendment_1.md).

Full generation was not run. The verified-macro hypothesis remains unresolved.
