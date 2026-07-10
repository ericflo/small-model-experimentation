# Adversarial design review: verified macro invention long-context rerun

Date: 2026-07-09, before any follow-up model call. Verdict: **sound after the mandatory
anti-censoring and immutability controls below**. The preregistration and harness must encode these
controls before GPU generation.

## 1. A rerun after failure can become outcome shopping

The direct parent already stopped after multiple interface attempts. Reopening it in place or
changing prompts until a pass would erase the meaning of its stop rule.

**Resolution:** create this separate material follow-up and leave the parent artifacts unchanged.
Change only the inference envelope and a broader train-only interface gate. Freeze the full ladder,
headroom criteria, interface threshold, scientific prompts, data, libraries, K values, and
confirmatory thresholds before generation. A clean failure at the registered maximum ends this
follow-up; it does not authorize another rung or prompt repair.

## 2. The parent observation was censored, not negative evidence

The 768-token plan-given attempt hit the thinking cap in 16/16 completions and the 128-token answer
cap in 12/16. Treating exact-record failure as semantic failure under those conditions would confuse
resource exhaustion with capability.

**Resolution:** explicitly reclassify the parent budgeted interface evidence as setup censoring.
The new run selects among `[2048, 4096, 8192, 16384, 32768]`, uses a 512-token answer stage, and
requires below-5% cap/truncation before a semantic gate is eligible. Ladder exhaustion is
setup-inconclusive, never a negative verified-macro result.

The parent's no-thinking attempt is still useful evidence about that distinct transcription mode,
but it cannot establish what budgeted induction can do. This follow-up does not overwrite or pool
that result.

## 3. Budget selection can peek at quality

Choosing the first rung whose answers look good would tune test-time compute to the desired result.
Even parser success or exact macro coverage is outcome information.

**Resolution:** calibrate on four deterministic train-only plan-given records at n=16 and expose
only runner metadata to the selector. The smallest eligible rung must have thinking-cap and
answer-truncation rates each below 5%, nearest-rank p99 thinking tokens no greater than 75% of its
allowance, and p99 answer tokens no greater than 256. Generated text, parse status, exactness,
macro use, and correctness remain sealed until selection is final. Unit tests must fail if
correctness fields enter the budget decision.

## 4. Marginal cap rates do not demonstrate headroom

A 4.7% cap rate technically passes a 5% gate while still showing that the ceiling is close; a
distribution shift from four calibration records to science tasks could immediately recensor the
run.

**Resolution:** the calibration adds two metadata-only headroom gates: p99 thinking usage at most
75% of B and p99 answer usage at most 50% of A. Heldout/scored stages retain the simple below-5%
censoring gates, with automatic whole-stage escalation if violated. Record zero-cap/zero-truncation
rates and maximum usage as diagnostics.

## 5. Interface calibration could leak the evaluation

Plan-given examples derived from smoke/full motifs could effectively demonstrate the answer surface
or select a convenient macro pattern before evaluation.

**Resolution:** both calibration and heldout interface records come only from verified construction
programs. The four calibration records and 16 gate records are mutually disjoint and contain no
smoke/full I/O or complete target. The gate may test literal alias replacement only; it cannot rank
libraries, select scientific tasks, or change thresholds.

## 6. Four old records were too narrow

The parent's task-independent gate used only four records. A few repeated successes or failures
could be record-specific rather than evidence that the surface generally works.

**Resolution:** retain four records only for metadata calibration at n=16. The independent semantic
gate uses 16 records at n=4 and requires exact macro-using optimal-surface coverage on at least
12/16 distinct records. Sample counts cannot substitute for record breadth. Strict literal
expansion, not raw parse or alias presence, decides success.

## 7. Larger budgets can change behavior, not just remove truncation

Reasoning budgets are not inert: Qwen3.5-4B can overthink, and a much larger maximum may alter its
candidate distribution.

**Resolution:** describe the ladder as a generation ceiling, not free accuracy optimization. Select
it from termination/headroom metadata rather than correctness. Apply the exact same settled rung to
every treatment and matched-compute baseline, preserve actual token counts, and interpret results as
the performance of that registered inference protocol. No claim about an optimal thinking budget is
allowed.

## 8. A 65k engine limit could create a second hidden context cap

Increasing the thinking allowance while retaining the template's 16,384-token engine limit would
fail before generation or silently force a smaller reserve. The long proposal prompt is especially
at risk.

**Resolution:** use `max_model_len=65536` and require exact tokenized preflight of prompt + B + close
sequence + 512 answer tokens before every request. Record rendered prompt length, reserve, engine
limit, and pinned model revision. Context overflow, OOM, or runner failure is infrastructure failure;
the harness may reduce concurrency but may not reduce the registered token allowances or omit arms.

## 9. Per-arm escalation would confound the comparison

If only a verbose or difficult treatment arm receives a larger budget, the treatment changes
compute along with representation. If lower-rung rows are retained for easy tasks, the result
becomes an adaptive mixture.

**Resolution:** censoring at or above 5% in a proposal batch reruns that complete batch. Censoring in
smoke reruns the whole base/designed matrix. Censoring pooled or within any full arm reruns the
entire full arm-by-task-by-sample matrix at the next rung. Preserve lower rungs only as diagnostics;
never pool, backfill, cherry-pick, or carry completed arms forward. Every published contrast uses
one complete rung.

## 10. Whole-matrix escalation is expensive

The conservative rule can consume far more GPU time, especially if censoring is discovered late in
the full matrix.

**Resolution:** use the 64-completion calibration and 64-completion heldout gate before fresh smoke,
then the 24-record base/designed smoke before full generation. Generate each stage as continuously
batched vLLM requests and inspect censoring automatically before parsing task results. The compute
cost is an accepted protection against a cheaper but uninterpretable mixed-budget result.

## 11. The answer cap can remain the binding limit

Expanding only thinking from 768 to 32k would repeat the parent error if the answer remains capped at
128. Conversely, silently switching to tolerant prose recovery would make the parser easier rather
than the envelope larger.

**Resolution:** fix the answer allowance at 512 for every rung and arm, measure second-stage length
termination separately, and include it in every escalation rule. Keep the strict one-line solver
parser and literal macro expansion unchanged. Do not recover programs from arbitrary prose.

## 12. Freshness claims require byte-level provenance

Copying an old harness without recording the exact source revision risks accidental task
regeneration or using a later modified library. It is also easy to forget that previous v1 smoke
was seen while v2 smoke was not.

**Resolution:** record parent commit
`1c8c5bbb81d2a67618891597205ceb2f40f498d8`, all copied file hashes, the full subset hash, and the
v2-smoke subset hash in `data/source_provenance.json`. Preparation must compare bytes and reject
drift. Do not copy parent run outputs into this result. The v2 tasks are fresh relative to all parent
model calls; v1 tasks remain historical only.

## 13. The original macro mechanism still has strong confounds

`mined > base` alone could come from highlighted subsequences, a larger inventory, shorter syntax,
or lucky library content rather than learned recurrence.

**Resolution:** retain every original mechanism control and threshold:

- base K=24 with a no-smaller-token prefix comparison;
- `mined_hint` with identical information but no callable aliases;
- five count/length/support-matched random libraries;
- a separate matched ensemble for `qwen_ranked` if needed;
- a generator-known designed ceiling;
- paired reuse/no-reuse slices;
- visible-only selection and hidden-only final grading.

The complete callable-abstraction verdict remains conjunctive: mined must beat base by at least
+0.10 with a positive paired lower bound, beat hint and matched random by at least +0.05 with
positive lower bounds, survive compute matching, show macro-mediated treatment-only successes, and
concentrate its effect on reuse.

## 14. Qwen proposal is ranking inside a closed language

The legal length-2/3 train-supported macro space is enumerable. Calling any selected entries
"invention" without exclusivity evidence would overstate the model's contribution.

**Resolution:** retain the name `qwen_ranked` and exact verifier. Require eight unique supported
entries, a +0.05 positive-interval advantage over its matched ensemble, performance within -0.05 of
mined, and at least two uniquely enabled correct selections. Otherwise report proposal failure or
ranking-only recovery.

## 15. Hidden labels and post-gate inspection can still leak

Even with frozen data, inspecting smoke/full hidden failures and then changing the budget, prompt,
parser, task set, or library would invalidate the result.

**Resolution:** budget changes are automated only by registered cap/truncation metadata and always
follow the fixed ladder. Hidden and probe labels enter only the committed analyzer. A clean
interface or smoke failure stops; no qualitative inspection authorizes a repair on these tasks.
Any future interface redesign, including the backlog's exactly-one-macro slot treatment, requires a
new experiment.

## Registered review verdict

Proceed only after tests demonstrate source-hash immutability, metadata-only rung selection, exact
context preflight, strict literal interface scoring, and whole-matrix no-pooling escalation. Under
those protections, the rerun can distinguish a cap-bound roadblock from a clean interface or macro
result without erasing the parent's history.
