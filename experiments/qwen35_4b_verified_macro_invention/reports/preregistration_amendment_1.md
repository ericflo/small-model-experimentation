# Preregistration amendment 1: fresh interface-repair smoke

Date frozen: 2026-07-09, after smoke v1 and before any v2 GPU call.

This amendment activates the failure branch already registered in
[`preregistration.md`](preregistration.md): repair the solver interface on a new smoke set, then
refreeze before full generation. It does not replace or retroactively edit the original
preregistration. Smoke v1 remains a failed interface diagnostic, documented in
[`smoke_v1_failure.md`](smoke_v1_failure.md).

## Information allowed into the repair

The repair may use:

- raw and derived smoke v1 outputs;
- the failed train-only macro-proposal outputs;
- CPU-only construction and leakage diagnostics; and
- a new, non-scored, train-only plan-given interface probe.

No full-evaluation model output exists, and no full hidden label, task-specific solution, or
evaluation-derived macro may influence this amendment.

The post-failure line-local reparse of v1 proposal outputs, which found 18 behaviorally unique
train-supported candidates after the strict whole-answer parser accepted 0/16 samples, is an
**exploratory diagnostic only**. It motivates the v2 extraction rule but cannot be counted as a
v1 result or used to backfill the v1 Qwen arm.

## Frozen and unchanged

The following remain byte-for-byte frozen across the repair:

- the 800-program construction corpus and 150-program proposal view;
- the latent-motif grammar and primitive semantics;
- the mined, highlighted-only, five random-placebo, and designed-ceiling libraries;
- the 80 reuse and 40 paired no-reuse full tasks, their visible/hidden/probe examples, depth
  proofs, and hashes;
- the full arm set and its sampling K values, visible-only selector, analyzer, outcome metrics,
  matched-compute rule, bootstrap procedure, confirmatory thresholds, and contingency branch;
- the hidden-label boundary; and
- the pinned model, vLLM backend, runner, model revision, and no-backend-mixing rule.

The Qwen-ranked library remains downstream of the repaired train-only proposal stage and retains
the original verification, support, uniqueness, ranking, no-padding, and eight-entry gates.

## Fresh v2 smoke set

The v2 smoke seed is `20260710`. Its record ids are
`smoke-v2-reuse-NNN` and `smoke-v2-no-reuse-NNN`. Before a GPU call, the manifest must prove that
the v2 programs and frozen behavioral signatures are disjoint from the construction corpus,
smoke v1, and the full evaluation. The v2 tasks may not replace, reorder, or otherwise alter any
full task.

The v1 state is preserved under:

- `configs/smoke_v1.yaml`;
- `data/smoke_v1_frozen/`;
- `runs/proposal_v1_failed/`;
- `runs/smoke_v1_failed/`;
- `analysis/smoke_v1_failed/`; and
- `archive/smoke_v1_source/`.

V2 continues to use the standard `runs/smoke/` and `analysis/smoke_*` paths.

## Solver interface repair

Both v2 solver arms use vLLM budgeted thinking with a 768-token thinking budget, matching the
already registered full-run budget, and the unchanged 128-token answer cap. Temperature, top-p,
top-k, parent seed policy, engine configuration, and exact token accounting remain unchanged.

Every solver prompt receives the same surface-first procedure:

1. infer an expanded primitive program that fits every visible pair;
2. rewrite it with any supplied exact aliases to minimize surface calls;
3. mentally check the rewritten expansion against all visible pairs; and
4. emit exactly one `PROGRAM: TOKEN | TOKEN | ...` line.

The shared instruction includes one inventory-neutral alias example, such as: if an abstract
alias `MX` expands to `A | B`, then the expanded plan `A | B | C` should be rendered as the
shorter surface `MX | C`. The placeholders are not legal task tokens and convey only how exact
aliases are called. The wording is identical in base and designed prompts; when an arm supplies
no callable alias, step 2 is a no-op.

Solver grading remains strict. After the standard answer-channel boundary handling, the parser
accepts exactly one non-empty `PROGRAM` line, only tokens from that arm's supplied inventory, and
no prose, salvage, or line-local recovery. Every accepted alias is literally expanded and checked
by the committed interpreter.

## Train-only plan-given probe

Before the scored v2 smoke, a frozen train-only probe supplies a known primitive plan and asks the
same interface to render the shortest legal surface under its supplied inventory. It tests only
formatting, alias calling, and exact expansion; it does not ask the model to infer a program from
evaluation I/O. It is non-scored, cannot contribute to any hypothesis metric, and may not use
smoke or full tasks. Failure stops the scored smoke for mechanical repair; passing it is not
scientific evidence.

## Proposal interface repair

The v2 proposal prompt contains the same frozen verified train programs but uses their program
token sequences only; train I/O examples are removed as irrelevant bulk. It requests compact
macro lines and retains the original 16 vLLM samples and closed two-/three-primitive language.

Proposal parsing is deliberately line-local because each macro line is an independent proposed
record. Within each completion, scan lines in output order, validate each `MACRO:` line
independently, retain only syntactically valid, train-supported, behaviorally non-duplicate
expansions, and stop after the first eight accepted lines. Prose and malformed lines are recorded
but do not invalidate a different valid line. The already frozen cross-sample ranking and exact
verification rules then construct `qwen_ranked`; the library is never padded. This relaxation
applies only to proposal records, not solver programs.

## V2 scored smoke and gate

The scored repair smoke runs only `base` and `designed_ceiling`, through the same copied vLLM
runner, at matched K=12. The registered go/no-go comparison is the fresh **reuse** slice; the
no-reuse records remain a descriptive interface check and disjointness control.

Full generation remains prohibited unless all of the following hold on v2:

- pooled, base, and designed-ceiling solver parse rates are each at least 0.50;
- pooled answer truncation is below 0.05;
- the designed-ceiling hidden oracle coverage is not below base at matched K=12; and
- valid macro-using designed candidates occur on at least two **distinct reuse tasks**.

The last condition strengthens v1's candidate-count gate so repeated samples from one task cannot
certify that the interface generalizes. Smoke metrics are go/no-go diagnostics only, not evidence
for the macro hypothesis.

## Full-run decision rules

If v2 fails, stop; do not tune on the full set. If it passes, run the originally registered full
arm set and analyze it under the original preregistration without changing any confirmatory
contrast or threshold. In particular, the primary `mined - base`, callable-versus-hint,
matched-random, reuse-specificity, macro-carry, Qwen-specific, matched-compute, and force-close
contingency rules are unchanged.
