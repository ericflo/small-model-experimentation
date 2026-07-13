# Preregistration v2: answer-seam factorial and fresh mechanics pilot

Status: model-free construction and implementation complete; live calibration
remains sealed pending a fresh adversarial PASS over final committed hashes,
then a separately committed/pushed implementation lock and green CI.

The immutable scaffold-v1 receipt proposed three policies. Adversarial review
correctly found that design causally incomplete. This append-only v2 replaces
it for all future execution with a full 2x2 interface calibration. The v1
receipt remains preserved at SHA-256 `fe03944be...`.

## Question and claim boundary

The interface stage asks only whether a predeclared emission policy reliably
produces an exact program-shaped event. Echo success does not demonstrate
reasoning, termination, certainty, or latent capability. The mechanics stage
asks whether one fixed, separately calibrated deployment policy makes
materialized residual generation outperform name-only, token-preserving
shuffled, and taskwise matched-compute direct sampling on fresh tasks.

This 24-task mechanics study is a large-effect pilot. Exact tests and bootstrap
intervals are report-only. No confirmatory capability claim is licensed by this
experiment alone.

## Frozen 2x2 interface

All cells receive byte-identical user messages, use unconstrained sampling,
`n=1`, temperature 0.6, top-p 0.95, top-k 20, and a 24-sampled-token answer
cap. There is no grammar, token mask, logit bias, constrained alias choice, or
teacher-forced answer token.

| Arm | Reasoning policy | Answer syntax prefill |
|---|---|---|
| `think512_freeform` | Qwen thinking chat channel; sample stage one up to 512 tokens; always retain only tokens before the first natural close, then inject the registered close and resample the answer | none |
| `think512_program_slot` | identical stage-one policy and seed domain | literal `PROGRAM:` token IDs |
| `no_think_freeform` | Qwen no-thinking chat channel; zero sampled thought tokens | none |
| `no_think_program_slot` | identical no-thinking policy and answer seed | literal `PROGRAM:` token IDs appended to the prompt before generation |

The two think512 cells consume one append-only shared-thought transaction: the
exact stage-one sampled and retained token IDs, prompt identity, row order,
seeds, finish state, runner hash, and Qwen revision are authenticated before
either continuation. They do not independently resample stage one. Even if
stage one naturally emits a close and reaches EOS, the harness
discards the sampled answer, retains only pre-close thought tokens, injects
`</think>\n\n`, optionally injects `PROGRAM:`, and samples the registered answer
tail. It never decodes and retokenizes a sampled thought. Logical matched-cost
accounting charges each arm for the shared thought it requires; separate
physical/reused counters disclose that the causal pair reused one physical
sample.

All four cells use a stable `answer` seed domain with canonical request IDs.
Because Ada/vLLM samples are not batch-invariant, paired numeric seeds are an
identity aid rather than a claim of common random numbers. Invocation order,
row order, `n=1`, scheduler mode, and batch geometry are implementation-locked.
Calibration invokes exactly one 48-row batch in this order:
`calibration_thoughts`, `think512_freeform`, `think512_program_slot`,
`no_think_freeform`, `no_think_program_slot`.

The program-slot cells use the same sampled-tail cap as freeform. The injected
syntax therefore supplies extra logical context but no answer identity; those
tokens are recorded separately and included in logical-token accounting. This
choice tests a deployable syntax seam, not equal full-line token capacity.

Stopping must preserve Qwen's asymmetric contract: ignore tokenizer EOS
`248046`, stop only on model EOS `248044`, retain semantic `248046`, and trim
only terminal `248044`. A cap contact is `n_answer_tokens >= 24` or answer
finish reason `length`. Thought-cap contact is recorded separately and never
conflated with no-think answer length.

## Exact answer event

The only accepted strings after the registered answer boundary are:

- `PROGRAM: A | B` shape for arity two; or
- `PROGRAM: A | B | C` shape for arity three.

Aliases vary from A through X. Spaces, newlines, explanations, alternate
punctuation, or extra text fail the parser. A single exact terminal special
token may follow. Parse success and byte-exact echo of the supplied line are
distinct metrics.

## Fresh construction

Construction seed is `2026072800`. It creates exactly 48 calibration and 24
mechanics tasks under namespace
`materialized-residual-answer-seam-factorial-v1`, with eight visible, eight
hidden, and sixteen unlabeled-probe inputs each. Every public instance, target
function, target triple, target suffix, and input row is unique across splits.
No visible relation has a depth-zero/one/two solution. Publicly live first
operations are exhaustively labeled over every one of 576 two-operation
suffixes.

Each 24-task balance block has 8/8/4/4 tasks with exactly 1/2/3/4 publicly live
first operations. The constructor authenticates and excludes all 264 public
instances from the closest fresh parent and proves zero task-ID, request-ID,
seed-key, and user-prompt overlap. Rendered-token and derived-seed overlap
remain mandatory implementation-review gates.

## Calibration rows and integer gates

There is exactly one request per calibration task: 24 arity-two and 24
arity-three mechanics-length prompts. The known answer is appended solely for
echo. Each alias A-X appears exactly once in every answer position, covering
the entire parameter-free and parameterized operation bank.

An arm qualifies only if all conditions hold:

- at least 44/48 exact echoes;
- at least 44/48 parses;
- at most 2/48 answer-cap contacts;
- within each arity, at least 22/24 exact echoes, at least 22/24 parses, and at
  most 1/24 answer-cap contact; and
- transaction, tokenizer, EOS, seed, prompt, and inventory authentication all
  pass.

The winner is not the best observed metric. Among all qualifiers, select the
first in this frozen least-departure order:

1. `think512_freeform`
2. `think512_program_slot`
3. `no_think_freeform`
4. `no_think_program_slot`

If none qualifies, stop `NO_VALID_RESIDUAL_ANSWER_SEAM`. No mechanics file may
be opened and no mechanics call may run.

## Calibration-to-mechanics firewall

All construction, prompts, parsers, selectors, metrics, thresholds, direct
master-pool rows, tests, and transaction code must be committed before the
first calibration call. A calibration implementation lock must be committed,
pushed, ancestral to `origin/main`, and green in both workflows. Calibration
may read only its request table, config, source hashes, environment receipt,
and lock.

The calibration winner receipt must then be committed and pushed. A distinct
mechanics lock binds that receipt and authorizes only the selected interface.
Mechanics begins with 24 disjoint public-visible transport echoes, alternating
12 suffix and 12 direct shapes. It requires at least 22/24 exact echoes and
parses, at most one cap contact, and at least 11/12 exact echoes and parses in
each arity. Failure stops `SELECTED_INTERFACE_DID_NOT_TRANSPORT`.

After generation, the visible-only selector and resource-match plans are
written durably and committed before the hidden file can be opened. Hidden
outcomes may score already frozen selected IDs and oracle proposal coverage;
they may not change a selection, prefix length, or prompt.

## Mechanics generation

The selected policy is used unchanged for every mechanics arm on the same vLLM
backend:

- materialized candidate-state-to-target relations, all 24 first operations;
- name-only original visible relations, all 24 first operations;
- task-hash target-deranged materialized relations, all 24 first operations;
- a 96-row-per-task candidate-blind direct master pool.

The three suffix arms share exact request IDs, order, paired thought/answer
seeds, candidate order, caps, and batch geometry. Invocation order is transport,
direct master pool, materialized, name-only, shuffled. Direct rows and order
are frozen before any live output and cannot be extended.

For each task, the materialized all-24 cost freezes two mandatory conservative
first-over prefixes of the already generated direct master pool: sampled-token
and logical-model-token matching. Logical tokens include every stage-one and
stage-two prompt/prefill, sampled token, forced close, and slot prefix in its
actual model-input position. Pool exhaustion fails the experiment.

## Estimands and decision

The deployable primary is hidden accuracy of a candidate selected before the
hidden file opens by `visible-probe-consensus-v1`. The selector filters programs
that execute on every visible row, clusters unique programs by outputs on
unlabeled probes, chooses the largest consensus cluster, then uses frozen hash
ties. It may abstain.

Oracle proposal coverage—whether any generated program is hidden-correct—is a
diagnostic, not a deployable result. The exhaustive CPU ceiling is also
diagnostic.

`MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_PASS` requires all of:

- parse >=0.90 and answer-cap contact <=0.05 in every generation arm;
- materialized selected accuracy >=0.25 and at least 6/24 selected successes;
- selected-accuracy gains >=0.125 versus name-only, shuffled, sampled-token
  matched direct, and logical-token matched direct;
- materialized oracle coverage >=0.35;
- oracle-coverage gains >=0.125 versus the same four controls; and
- at least eight distinct first-operation aliases among hidden-correct
  materialized proposals.

If the generation ABI gates fail after transport, label
`MECHANICS_INTERFACE_NONTRANSPORT`, not a materialized-residual null. Otherwise
failure is `MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL`.

## Non-rescue rules

No task count, prompt, arm, priority, parser, cap, temperature, threshold,
direct-pool ceiling, invocation order, selector, backend, or seed may change
after calibration begins. No cap increase, parser relaxation, alternate answer
extraction, extra direct samples, cheap-ranker revival, threshold tuning, or
outcome-conditioned rerun is allowed. All negative results and transaction
incidents are preserved.
