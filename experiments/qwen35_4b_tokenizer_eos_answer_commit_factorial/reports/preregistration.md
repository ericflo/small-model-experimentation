# Preregistration

## Status and scope

Prospective and model-free as of 2026-07-14. No task construction, calibration
sample, mechanics sample, or protected label has been opened for this
experiment. Any amendment after model output must be a new experiment.

## Scientific question

Does first tokenizer EOS provide a strict answer-stage commit boundary on fresh
known-answer rows, and—conditional on independent qualification—does the
resulting interface expose a materialized-residual capability gain?

## Fixed model/backend

- `Qwen/Qwen3.5-4B`
- revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- bf16, pinned vLLM, identical engine and runner within all paired comparisons
- no adapter during interface or mechanics evaluation

No other model may generate, judge, label, or teach.

## Fresh substrate and firewalls

Construct a new namespace of exact-depth-three procedural list transforms.
Exclude behaviorally shallower functions and all parent task/function/request/
seed/prompt identities. Freeze 48 calibration tasks and a disjoint mechanics
split before any model call. Calibration may open only known-answer echo rows.
Mechanics files remain sealed behind a committed-green calibration decision and
winner-bound lock. Hidden mechanics labels remain sealed behind committed-green
visible selection. Never read `benchmarks/`.

## Calibration factorial

Cross:

1. answer stop boundary: first tokenizer EOS 248046 versus HF model EOS 248044;
2. thought policy: no-think versus 512-token thought then forced close; and
3. answer prefix: freeform versus literal `PROGRAM:` token prefix.

The tokenizer-EOS policy applies only to answer stages. Thought generation uses
the unchanged parent policy. Pair task order, run seed, and answer seed across
boundaries; authenticate matching sampled prefixes up to first 248046 whenever
both traces reach it.

Strict output is exactly `PROGRAM: <alias> | <alias>` with no whitespace or
token before the registered commit beyond the grammar. The commit token itself
is boundary metadata and is trimmed token-natively. No decode/re-tokenize path
may define exactness.

## Interface gates and selection

For one tokenizer-EOS arm to qualify it must have:

- at least 44/48 strict parses;
- at least 44/48 exact echoes;
- no more than 2/48 answer-cap contacts;
- within each 24-row arity block, at least 22 parses, at least 22 exact echoes,
  and no more than one answer-cap contact; and
- zero authenticated malformed-stop receipt violations.

Select the first qualifier in this fixed order:

1. tokenizer EOS, no-think, `PROGRAM:`;
2. tokenizer EOS, no-think, freeform;
3. tokenizer EOS, think@512, `PROGRAM:`;
4. tokenizer EOS, think@512, freeform.

Observed metric ranking cannot change the order. HF-EOS cells are causal
controls and cannot become the selected successor interface. The two no-think
prefix cells are paired conditions, not replications.

If no tokenizer-EOS arm qualifies, emit `NO_VALID_TOKENIZER_EOS_ANSWER_SEAM`,
open no mechanics file, and retire this branch.

## Required termination controls

Before a live lock, model-free tests must reject:

- missing registered stop;
- an early registered stop followed by additional sampled tokens;
- repeated/interior plus terminal registered stops;
- wrong stop reason or finish reason;
- answer cap overflow or a short answer relabeled length;
- extra newline, close, chat marker, or other byte before commit;
- prompt/seed/token/text/cost/summary mutations; and
- a tokenizer stop applied to the thought stage.

Live receipts must bind exact stop-token configuration, first-stop sampled IDs,
finish/stop reason, pre-commit content, physical/logical token cost, prompt,
seed, runner, backend, model, and revision.

## Conditional mechanics

If calibration qualifies, publish its complete authenticated result and wait
for both workflows to pass. Mint a second lock binding the selected arm and
fresh mechanics bytes. Run a selected-interface transport gate before bulk
mechanics. If transport fails, stop.

Mechanics must compare materialized residual states with name-only, deterministic
shuffled materialized state/target, candidate-blind direct sampling, exhaustive
CPU ceiling, and direct prefixes matched taskwise by both sampled and logical
model-token cost on the same vLLM backend. Freeze visible-only selection before
opening hidden labels. Oracle proposal coverage is report-only.

## Capability claim gate

No interface result is a capability result. A deployable gain requires the
visible-only selector to beat the frozen Qwen3.5-4B baseline, every structured
control, and both matched-compute direct baselines on contamination-free hidden
tasks with the preregistered task-level inference. It must rule out extra-token,
format, backend, answer leakage, and oracle-selection explanations.
