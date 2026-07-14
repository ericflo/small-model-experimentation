# Qwen3.5-4B Materialized Residual Answer-Seam Factorial Report

## Summary

The registered calibration terminated at `NO_VALID_RESIDUAL_ANSWER_SEAM`, so
mechanics never opened. All 240 outputs in five durable transactions
authenticated, but every arm scored 0/48 strict parses and exact echoes.

This is a clean interface negative, not a residual-mechanics result. A frozen
post-decision diagnostic found that both no-think arms became 48/48 exact under
the frozen parser after removal of only tokenizer EOS `<|im_end|>` plus newline.
Thinking arms retained additional close-boundary failures. The runner
intentionally waited for the later HF model EOS, so the registered parser
correctly rejected the extra terminal bytes. This licenses a fresh answer-stage
tokenizer-EOS experiment; it does not change this experiment's decision.

## Research Program Fit

The closest parent cleanly falsified its cheap viability ranker but could not
adjudicate materialized residual generation because every thought hit cap and
the answer ABI failed. This experiment tested that prerequisite on separate
known-answer tasks before reopening composition. It neither revives the failed
ranker nor treats echo as a capability result.

## Method

Fresh construction uses the same 24-operation list DSL but a new namespace and
seeds. Exact common-panel functions exclude depth <=2 behavior. Every accepted
task exhaustively enumerates 24 first operations x 576 two-operation suffixes,
giving exact public-live labels. The 72 tasks are split 48 calibration / 24
mechanics with unique public functions, target triples, target suffixes, and
input rows.

Calibration crosses think@512/no-think with freeform/`PROGRAM:` prefill. It uses
48 mechanics-length echo rows balanced across A-X in every answer position.
All aliases remain autonomously sampled. The fixed winner is the first gate
passer in preregistered least-departure order.

Mechanics would have generated all 24 residual siblings under materialized,
name-only, and shuffled evidence plus a frozen 96-sample direct pool per task.
The calibration gate failed, so none of those prepared mechanics requests or
protected labels were opened.

## Results

Construction summary SHA-256:
`b39e0ad1ccf49503eb48353eac118500432953f32ad27ae2acc1448ed99f622d`.

- 48 calibration and 24 mechanics tasks;
- 24/24/12/12 single/double/triple/quad public-live strata overall;
- 72 unique public-instance fingerprints;
- zero overlap with 264 authenticated parent public instances;
- 4,104 prepared rows and 2,952 unique canonical request IDs;
- three suffix controls share exact IDs/order;
- zero parent request-ID, seed-key, or user-prompt overlap;
- every A-X alias appears once in each calibration answer position;
- real-tokenizer receipt `61ff7292...` authenticates `PROGRAM:` `[78041,25]`,
  close `[248069,271]`, all 14,400 canonical answer compositions, 1,396-token
  worst-case context, and zero overlap with 1,984 parent rendered prompts;
- a reviewed implementation lock with zero prior model calls/outputs and empty
  protected-read receipts; and
- 82/82 model-free tests before execution, including append-only transaction
  and crash-recovery mutations.

After five exact-hash adversarial reviews, two independent archive audits
returned `PASS_RELEASE_LIVE_CALLS`. The lock-bearing commit passed both GitHub
workflows before calibration. The live run then authenticated:

- five registered invocations and 240/240 sampled outputs;
- exact shared thought-token pairing across the two thinking continuations;
- exact answer-seed pairing and registered prefix assignment; and
- empty benchmark, mechanics, qualification, confirmation, and hidden reads.

The frozen gate metrics were:

| arm | strict parse | exact echo | thought cap | answer cap |
|---|---:|---:|---:|---:|
| think@512, freeform | 0/48 | 0/48 | 48/48 | 18/48 |
| think@512, `PROGRAM:` | 0/48 | 0/48 | 48/48 | 0/48 |
| no-think, freeform | 0/48 | 0/48 | 0/48 | 0/48 |
| no-think, `PROGRAM:` | 0/48 | 0/48 | 0/48 | 0/48 |

No arm approached the registered >=44/48 exact/parse gates. The fixed winner is
therefore null and `NO_VALID_RESIDUAL_ANSWER_SEAM` is terminal.

### Post-decision terminal-boundary diagnostic

This diagnostic is report-only and cannot change qualification. It first
removes only an exact final decoded `<|im_end|>\n`, then reruns the frozen full-
string parser. A separate expected-tail diagnostic takes the final segment
after a thinking close and can therefore hide an additional `</think>`; it is
reported only to locate the requested answer, not as exact output. The sampled
terminal sequence is tokenizer EOS 248046, newline 198, and registered HF model
EOS 248044.

| arm | frozen parser after suffix-only removal | expected-tail match | extra `</think>` |
|---|---:|---:|---:|
| think@512, freeform | 24/48 | 29/48 | 5/48 |
| think@512, `PROGRAM:` | 38/48 | 48/48 | 10/48 |
| no-think, freeform | 48/48 | 48/48 | 0/48 |
| no-think, `PROGRAM:` | 48/48 | 48/48 | 0/48 |

Thus tokenizer-EOS stopping is a complete post hoc explanation only for the two
no-think cells. The thinking cells also have extra close boundaries; freeform
thinking adds 18 answer-cap contacts. A fresh successor must treat thinking as
a control rather than assume that one stop-token change repairs it.

## Controls

The model boundary is exact `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Prepared arms freeze backend,
requests, ordering, seeds, caps, and direct-pool ceiling. Forbidden benchmark,
mechanics, qualification, confirmation, and hidden-read receipts are empty.
These are application-level receipts, not OS syscall telemetry. The
post-decision diagnostic was computed only from already opened calibration
outputs.

## Oracle Versus Deployable Evidence

No mechanics outcome exists. The deployable selector, hidden proposal coverage,
and exhaustive CPU ceiling all remain unopened because the interface gate
failed.

## Interpretation

The strict HF-model-EOS answer ABI is invalid for this workload, and residual
generation remains unadjudicated. The result should not be summarized as “the
model could not copy a program”: both no-think arms became 48/48 exact after
removal of one terminal suffix. Nor should the expected-tail diagnostic for
thinking be called full-string exactness, or any diagnostic be called a
successful interface after the fact: parser, stopping policy, cap accounting,
and transaction semantics were registered around the later HF EOS.

The narrow causal hypothesis for the successor is now concrete. Treat
tokenizer EOS as an explicit answer-stage deployment commit event, stop at its
first sampled occurrence, remove only that registered terminal token, and
require every preceding byte to pass the same exact grammar. Fresh tasks,
record IDs, and seeds must be used. Controls must include the current HF-EOS
boundary, freeform versus literal `PROGRAM:` prefix, think versus no-think,
early/interior/missing terminators, extra pre-commit bytes, exact stop/token/cost
authentication, fresh transport, and the same mechanics firewall. The paired
no-think cells share rows/seeds and are not independent replications. Only a
newly qualified interface may reopen residual mechanics.

## Remaining Work

This experiment is complete and terminal. Publish the authenticated negative
and create a separate fresh tokenizer-EOS commit-boundary successor. Do not run
the existing mechanics lock, transport, generation, visible, or hidden stages.

## Artifact Manifest

All construction, calibration transactions, decision, and derived diagnostic
are tracked. The manifest records their controlling hashes. There are no
external or omitted artifacts.
