# Preregistration: Early Text Hypothesis Forking

Frozen after adversarial review and before any model-bearing call. Later prompt,
parser, arm, candidate, threshold, temperature, layer, or budget changes require
a fresh experiment directory.

The pre-model implementation review exposed underspecified mechanics details.
Before any model construction or output, these were frozen in
[`preregistration_amendment_1.md`](preregistration_amendment_1.md). That
amendment withdraws the underspecified Stage 1 contract at `c064d7a4`, becomes
the new design boundary when committed and pushed, and controls Stage 1 where
it is more specific than this document. No original threshold is relaxed.

## Hypothesis and falsifier

Primary hypothesis: placing a bound first-operation hypothesis at the start of
thinking changes Qwen3.5-4B's full-program proposal distribution enough that a
single visible-only selector beats the identical hypothesis inserted late and
ordinary sampling matched on both sampled and logical model tokens.

The hypothesis is falsified for this interface if valid mechanics passes but
early forking misses either registered proposal-coverage or deployed-selection
margins against any mandatory comparator. Interface, adherence, and oracle-
ceiling failures receive separate terminal outcomes.

## Bound DSL and task construction

The frozen inventory has exactly 24 concrete operations:

- `reverse`, `sort_asc`, `sort_desc`, `abs_all`, `square`, `negate`,
  `running_sum`, and `adjacent_diff`;
- `add_k` for `k in {-3,-2,-1,1,2,3}`;
- `mul_k` for `k in {-2,2,3}`;
- `take_k` for `k in {1,2,3,4}`; and
- `rotate_k` for `k in {1,2,3}`.

`negate` is present in every candidate bank but is not generated as the true
first step because common algebraic reorderings make unique attribution
unstable. Gold first steps are balanced over the other 23 bound operations.

Each task has exactly two operations, 8 visible examples, 8 hidden examples,
and 16 task-seeded unlabeled probe inputs. Before model use, exhaustive 24²
enumeration must establish:

1. no one-operation program fits every visible example;
2. every visible-consistent two-operation program begins with the same bound
   operation; and
3. every visible-consistent program agrees on every hidden and unlabeled probe
   input.

The complete visible-consistent set is stored in gold for audit. Qualification
and confirmation behaviors are globally unique, disjoint from readable prior
procedural artifacts, and never pooled. Benchmark contents remain unread.

## Output ABI

The final answer is raw code or one fenced code block of exactly this shape:

```python
def transform(xs):
    xs = reverse(xs)
    xs = add_k(xs, -2)
    return xs
```

The AST parser accepts one `transform(xs)` function, exactly two assignments to
`xs` calling allowed helpers with legal literal parameters, and one `return
xs`. It rejects imports, attributes, control flow, arbitrary expressions,
defaults, decorators, annotations, extra statements, and multiple code blocks.
The canonical operation tuple is interpreted; generated Python is never run.

## Injection and frozen arms

The exact candidate string is appended as token IDs inside open `<think>`:

```text
Hypothesis fork — provisional; test it against every example and revise if contradicted.
Concrete first operation: add_k(-2)
```

Early and late arms must have identical injection-token hashes. Branch
permutations are independently task-randomized with domain-separated seeds;
cyclic shifts are forbidden.

- `early_concrete_24`: one distinct bound operation per independent trajectory,
  then 1,024 thought tokens.
- `late_equal_total_24`: one independent candidate-blind 512-token prefix per
  branch, candidate injection, then 512 thought tokens.
- `late_equal_post_24`: the same construction with 1,024 post-injection thought
  tokens. This intentionally overmatches early's post-hypothesis allowance.
- `early_duplicate_24`: 24 independent branches carrying one task-hash-selected
  bound operation.
- `early_placebo_24`: 24 independent branches carrying exact-scaffold neutral
  text with all injected tokens charged.
- `neutral_sample_more_master`: 48 independent 1,024-token samples with the
  neutral scaffold.
- `plain_sample_more_master`: 48 independent 1,024-token samples without the
  fork scaffold.
- `cpu_exhaustive`: deterministic enumeration of all 576 programs using visible
  examples only. It is a scope/tool control, never an oracle model selector.

All model arms use the pinned vLLM backend and identical decode distribution
(`temperature=.6`, `top_p=.95`, `top_k=20`). Prefixes are exact token IDs; late
continuations may not decode and retokenize cached text. The same force-close
and answer continuation policy applies everywhere.

## Visible-only selector

Before any gold file is opened, each arm writes prompts, token hashes, raw
outputs, canonical parse rows, resource receipts, and selected candidate IDs.
The selector then:

1. strictly parses and canonical-deduplicates candidates;
2. retains canonical programs with exact agreement on all visible examples;
3. groups them by the 16-output unlabeled-probe vector;
4. chooses the group with most distinct canonical support; and
5. breaks remaining group/program ties by a frozen task-seeded canonical hash.

No visible passer means abstention. Invalid outputs and abstentions are wrong.
Every arm uses this exact selector. Replacing hidden examples, the gold first
operation, or the target program with arbitrary values must leave prompts,
resource matching, canonical candidates, and selected programs byte-identical.

## Resource matching

The neutral and plain master-pool orders are frozen before grading. For each,
derive and retain:

- the largest cumulative prefix not exceeding early's sampled-token cost;
- the first cumulative prefix meeting/exceeding that cost;
- the corresponding under/over prefixes for total logical model-token cost;
  and
- the full K=24 and full master-pool references.

Mandatory capability comparisons use the conservative first-over-budget point
for both sampled and logical costs; under-budget points are descriptive. Count
prompt prefills, all independent blind prefixes, every resumed-prefix re-prefill,
injection tokens, sampled thought, forced close, answer continuation, failed and
duplicate completions, and any model-based verification call. Pool exhaustion
invalidates the stage. Pool membership is fixed before outcome grading.

## Stage 0: model-free smoke

All must pass before a live model call:

- exact 24-operation inventory, split counts, deterministic regeneration, and
  ancestor behavior disjointness;
- exhaustive depth/unique-bound-first/equivalent-visible-fit checks;
- strict AST round trips and malicious-code rejection;
- selector, branch-plan, and resource-plan invariance to gold mutations;
- independent balanced branch positions and distinct serialized
  `slot -> bound operation -> fixed-panel behavior` compositions;
- exact public/gold separation and no benchmark/model/outcome read;
- candidate injection construction and early/late token-identity unit tests;
  and
- mechanics, qualification, and confirmation fail closed.

## Stage 1: label-free mechanics

Four public five-element diagnostic lists are crossed with all 24 supplied
bound operations: 96 rows in each of systematic, length-matched independently
deranged, duplicate, and placebo arms. The model must compute the resulting
list and emit exactly one unrestricted `RESULT: [...]` line. No qualification
gold, task pipeline, or hidden example is loaded.

All mechanics gates must pass:

- complete runner/resource receipts and pinned model/revision/backend;
- unrestricted parse at least .90 in every primary arm;
- answer-limit contact at most .05;
- systematic supplied-operation execution at least .75, deranged supplied-
  operation execution at least .60, and systematic-registered minus deranged-
  registered execution at least +.35, both globally and separately in all
  four contexts;
- systematic candidate adherence at least .60 globally and in every context;
- successful results spanning at least 20 of 24 bound operations and all four
  contexts;
- duplicate and placebo registered-target rates at most .20; and
- on eight stratified full-program mechanics cases, strict parse is at least
  .90, cap contact is at most .05, visible pass is at least .50 overall, and
  visible pass is at least .50 within four parameterized-first-operation cases.

The full-program arm is a non-causal reachability ceiling only. The live KV
preflight deliberately requires block-rounded capacity for the frozen maximum
active width, preventing scheduler preemption/recomputation from contaminating
resource comparisons.

Fail-specific decisions are `INVALID_INTERFACE_PARSE`,
`NO_HYPOTHESIS_ADHERENCE`, and `NO_CORRECT_HYPOTHESIS_CEILING`. Any failure is
terminal for this interface and keeps qualification sealed.

Any prepared-artifact authentication, implementation-lock, transaction,
package-environment, CUDA-graph, or live-KV preflight failure yields no
scientific verdict and also keeps qualification sealed.

## Stage 2: qualification

Run 48 tasks in two frozen 24-task blocks. Gold opens only after all generation,
selection, and resource artifacts are immutable. Early selected hidden-all
accuracy must be between .15 and .85 so the registered contrasts are reachable.

All gates apply globally, with paired task bootstrap inference and Holm
correction over mandatory comparators:

- every primary arm parse rate at least .90 and early no more than .02 below
  the strongest control;
- `early_concrete_24` selected accuracy at least +.05 over
  `late_equal_total_24` and nonnegative versus `late_equal_post_24`;
- early selected accuracy at least +.10 over duplicate, placebo, and every
  mandatory neutral/plain sampled-token- and logical-token-overmatched pool;
- every required one-sided paired 95% lower bound strictly above zero after the
  registered family correction;
- early oracle hidden coverage at least +.10 over the strongest matched
  sample-more pool, with paired lower bound above zero;
- visible selector capture at least .90 of early oracle coverage;
- both 24-task blocks have positive early gain;
- successes span at least 8 operation types, include parameterized and
  parameter-free cases, and no one type supplies more than .25 of aggregate
  gain; and
- exact resources with no master-pool exhaustion.

Coverage without selection is `PROPOSAL_SHIFT_SELECTOR_FAIL`; selection without
coverage is `NO_PROPOSAL_SHIFT`; early failing the timing contrasts is
`EARLY_EQUALS_LATE`; and a valid miss versus ordinary sampling is
`NO_MATCHED_SAMPLING_GAIN`. A complete pass is
`EARLY_HYPOTHESIS_FORKING_QUALIFIED`.

## Stage 3: confirmation

Run the identical frozen protocol on 96 untouched tasks only after a complete
qualification pass. Apply the same gates and comparator family; subdivision is
descriptive. Qualification and confirmation never pool. A complete replicated
pass is `EARLY_HYPOTHESIS_FORKING_CONFIRMED`.

## Interpretation boundary

The strongest positive interpretation is externally stratified,
verifier-assisted text elicitation on an exhaustively searchable depth-two DSL.
It is not J-space transport, internal confidence measurement, consciousness,
weight installation, verifier-free capability, or superiority to exhaustive
symbolic search. Depth expansion or training requires a fresh preregistered
successor.

## Prohibited rescues

- no outcome-informed prompt/parser/threshold/candidate/budget/temperature fix;
- no hidden-informed selection, regeneration, stopping, or tie-breaking;
- no benchmark source reads or benchmark-derived training data;
- no other model, teacher, judge, or distillation source;
- no confirmation after a failed qualification gate; and
- no in-place higher-depth, training, or new-substrate follow-up.
