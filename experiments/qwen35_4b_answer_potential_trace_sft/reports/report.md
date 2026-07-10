# Qwen3.5-4B Answer-Potential Trace SFT Report

## Status

**Terminal verdict: `SCORER_NEGATIVE`.** The preregistered G0 gate failed. In accordance with the
frozen decision rule, the full N=128 harvest, pivot/branch arm, trace selection, QLoRA training, and
adapter evaluation were not run. The full-stage command was exercised and wrote
[`../runs/full_refusal.json`](../runs/full_refusal.json) before refusing to proceed.

This is a result about the scorer prerequisite, not a direct SFT comparison. It rules out banking
traces selected by this exact answer-potential protocol; it does not establish that every possible
answer-conditioned trace score or trace-SFT design fails.

## Design Boundary

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- The complete plan, gates, controls, and split policy were frozen in commit `3441dd23` before any
  GPU-scale call.
- Substrate: fresh, split-disjoint procedural atom tasks copied into the experiment; no benchmark
  content was read or used.
- Calibration: 64 prompts, 32 thoughts per prompt, temperature 1.0, top-p 0.95, top-k 20, and a
  512-token thinking cap. Six prompts whose answer equivalence class was combinatorial were excluded
  from confirmatory potential scoring under the frozen equivalence-set exclusion rule.
- Outcome label: eight fresh short answer continuations per scored thought, generated with disjoint
  seeds on the same vLLM backend.
- The dated pre-result scoring amendment in
  [`preregistration.md`](preregistration.md) replaced vLLM's pathological full-vocabulary
  `prompt_logprobs` path with mathematically identical targeted next-token reads. HF bf16 SDPA parity
  passed at a 0.060-nat/token maximum discrepancy versus the frozen 0.15 tolerance.

## Primary Gate

Calibration produced 2,048 thoughts. The 58 scorable prompts contributed 1,856 scored thoughts and
14,848 fresh rollout outcomes. The six numbered G0 conditions expand to eight stored booleans because
the two selector comparisons and two corruption controls are evaluated independently.

| diagnostic | observed | frozen requirement | result |
| --- | ---: | ---: | :---: |
| task-macro within-task answer-gain AUROC | 0.6167 on 45 mixed tasks | >= 0.65 | fail |
| top-one gain minus seeded-random success | +0.0733, CI [0.0172, 0.1379] | >= +0.10 and CI lower > 0 | fail |
| top-one gain minus shortest success | +0.0582, CI [0.0108, 0.1142] | >= +0.10 and CI lower > 0 | fail |
| gain beats negative length and trace prior | 0.6167 vs 0.5004 length; prior unavailable | strictly beats both | fail closed |
| real minus token-shuffled mean gain | +0.5554 nats, CI [0.2014, 0.9781] | CI lower > 0 | pass |
| real minus foreign-task mean gain | +4.7906 nats, CI [3.5132, 6.1641] | CI lower > 0 | pass |
| task-macro format-variant rank stability | Kendall tau 0.8301 | >= 0.80 | pass |
| positive gain before first answer mention | 0.5690 | >= 0.75 | fail |

Overall: **3/8 stored criteria passed; G0 failed.** Full numerical output and artifact checksums are
preserved in [`../runs/calibration/g0.json`](../runs/calibration/g0.json), with a compact table in
[`../analysis/g0_metrics.csv`](../analysis/g0_metrics.csv).

## What The Signal Did And Did Not Do

The top trace by answer gain had fresh rollout success 0.2026, compared with 0.1293 for the seeded
random trace and 0.1444 for the shortest trace. Both paired bootstrap lower bounds were positive.
Thus the score has modest within-task ranking value; the failure is not equivalent to chance. The
preregistered gate required a practically useful +0.10 uplift, however, and observed uplifts were
only +0.0733 and +0.0582.

The mechanism controls also show real trace dependence. A real thought improved canonical-answer
gain over a length-matched token shuffle by 0.555 nats task-macro, and over a foreign-task thought by
4.791 nats; both confidence intervals excluded zero. Changing the answer rendering preserved rankings
at Kendall tau 0.830. A positive canonical-versus-procedural-decoy margin occurred for 91.3% of scored
thoughts. The scorer is therefore sensitive to task-relevant thought content and robust to the tested
format perturbation, but that information did not translate into sufficient autonomous-answer lift.

The trace-prior ranker could not be evaluated because the thought-generation request did not capture
sample log-probabilities. The implementation writes JSON strictly and represents the missing metric as
`null`; the criterion fails closed. This instrumentation miss is material and recorded, but it cannot
rescue the gate: four other independent conditions failed.

## The Dominant Failure: An Unreachable Answer State

| rollout/termination diagnostic | value |
| --- | ---: |
| naturally closed thoughts | 13 / 2,048 (0.63%) |
| 512-token cap contacts | 2,035 / 2,048 (99.37%) |
| mean / median thought tokens | 511.2 / 512 |
| fresh answer rollout accuracy | 1,705 / 14,848 (11.48%) |
| fresh answer parse rate | 1,962 / 14,848 (13.21%) |
| accuracy conditional on parsing | 1,705 / 1,962 (86.90%) |

The scorer teacher-forced `y*` after an injected `</think>\n\nANSWER: ` seam. Almost every sampled
thought was still running when that seam was injected. The continuation test, by contrast, required
the model to recover from the forced close and emit a parseable terse answer. The extremely low parse
rate and high parse-conditional accuracy locate the main loss at this interface: when the model
committed in the expected form it was usually right, but it rarely entered that form after a cap-bound
thought.

This explains how answer potential can pass the corruption and format tests while failing selection:
it measures whether the reference answer is locally compatible with a counterfactual forced answer
state, not whether the model will autonomously terminate and express that answer. It also aligns with
C50's independent finding that the answer-emission seam can dominate deployable behavior.

The pre-answer-mention diagnostic adds a second caution. Only 33/58 selected traces showed positive
gain before their first verbatim answer mention or never mentioned it; the required count was at least
44/58. Ten selected traces never mentioned the answer, while only 23 of the remaining 48 passed before
mention. Some high gain therefore arrives too late to rule out answer-copy or answer-rehearsal effects.

## Heterogeneity

The scorer was not uniform across task families. Task-macro AUROC exceeded 0.65 for caravan (0.653),
foundry_ledger (0.710), and gatepost (0.764). Glyphgate's nominal 0.955 came from only one mixed task and
is not stable evidence. Ferrier, kilnrite, loomfix, and runeward were near chance. Parse rates ranged
from 0.5% to 56.2%, so family-level discrimination is entangled with whether the answer interface works
for that family. The full table is [`../analysis/family_summary.csv`](../analysis/family_summary.csv).

## Compute And Provenance

- Counted logical tokens: 32,805,906.
- Thought sampling: 490,784 prompt plus 1,046,911 sampled tokens.
- Targeted scoring: 19,474,315 repeated-prefix prefill plus 26,702 one-token reads.
- Rollouts: 11,110,544 prompt plus 656,650 sampled tokens.
- Sum of timed GPU operations: 3,128.8 seconds (52.15 minutes); this is an operation-time sum, not a
  claim about wall-clock exclusivity.

Every GPU artifact was written atomically before reduction. The first reduction then exposed a
configuration-key typo (`premember` versus `premention`). No inference was regenerated: the added
CPU-only `--stage analyze-g0` path verified and reduced the saved artifacts. Strict JSON serialization
also caught and removed a non-standard `NaN`, replacing the unavailable trace-prior metric with `null`.
These recovery details and the design receipt remain part of the committed provenance.

## Learned Lessons And Decision

1. **Dense oracle-side signal is not enough.** Answer potential carried real trace-specific
   information, yet its ranking lift missed the actionable bar. Selector validation must precede
   expensive curation or SFT.
2. **Deployment matching is part of score validity.** Scoring a teacher-forced answer after a forced
   boundary can validate a counterfactual state the model almost never reaches. Termination and parsing
   need to be part of the measured event.
3. **Do not answer a closure failure with larger `N`.** At 99.37% cap contact, N=128 would mostly buy
   four times as many unfinished traces. The matched-compute sample-more baseline would remain the bar,
   and no evidence licenses that expense.
4. **Capture every declared baseline at generation time.** Trace-prior log-probability was a frozen
   comparator but was not recorded. Future harnesses should assert required fields before the first
   scientific shard, not merely fail closed during reduction.
5. **Preserve effect-size gates.** The positive confidence intervals could have invited a post-hoc
   success story. The preregistered +0.10 threshold correctly separated detectable signal from a method
   worth training on.
6. **The next useful question is the close/commit seam, not a selector retune.** A fresh experiment
   could compare joint likelihood of `</think>\n\nANSWER: y*` against answer-only potential, or first
   create adequate naturally early-closing coverage and then repeat the within-task gate. It should
   retain shuffled/foreign/length/prior controls and treat autonomous parseability as confirmatory.

Decision: stop this experiment at G0. No pivot branching, diversity optimization, SFT, adapter, or
held-out capability claim is authorized. The complete terminal summary is
[`../analysis/summary.md`](../analysis/summary.md).

## Smoke Evidence

- Frozen procedural split construction produced 64 calibration, 600 train, 400 IID, 100 held-family,
  and 100 hard items with zero ID, prompt, digest, or generator-seed overlap.
- Twenty-five CPU tests pass across the firewall, verifier-equivalent answers, controls, statistics,
  and vLLM geometry.
- The corrected real-model smoke produced four finite trace scores and eight fresh answer
  continuations with exact registered CUDA-graph geometry and a passing live-KV capacity receipt.
- HF bf16 SDPA versus vLLM bf16 targeted likelihood differed by at most 0.060 nats per answer token,
  below the pre-calibration plumbing tolerance of 0.15. HF rows are diagnostic only.
- The 64-token smoke traces all contacted the thought cap; their zero rollout score is intentionally
  non-scientific and does not enter G0.

These smoke rows were plumbing checks only and were not included in any scientific statistic above.
