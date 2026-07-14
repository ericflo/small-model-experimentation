# Adversarial Design Review

## Scope

This review covers the model-free design freeze for
`qwen35_4b_universal_state_table_compiler_token_match`. It inspected only
experiment-owned sources, inherited replay artifacts, public benchmark contract
metadata, and executable receipts. It made zero model calls and read no benchmark
source, item, transcript, private output, or score detail.

## Question and closest duplicate

The closest near-duplicate is
`qwen35_4b_universal_search_scaffold_token_match`. That experiment separately
supervised canonical apply/fit/reject/execute/search stages but tied active replay at
16/26, trailed its parent at 18/26, reached 0/2 execute and 0/2 probe, and stayed
sealed from broad evaluation. The present design is not another dose or close-weight
variant: it removes canonical operation codes and tests a different deployment
interface—variable-depth natural-language state execution, independent scoring of
explicit hypotheses, repair of a first bad transition, and a deliberately short
answer commit.

## Generator truth and shortcut audit

- Construction seed 77,112 deterministically emits 80 rows: 20 each execute, score,
  repair, and commit. Source SHA-256 is `a7b453af...e88bb`.
- Execute, repair, and commit cover depths 2–5 with five rows at every depth. Score
  covers depths 2/3/4 as 7/7/6. Six surface classes occur.
- Every answer, transition, repair location, hypothesis prediction, and score is
  recomputed from the executable operation representation. The source check rejects
  any mismatch.
- Every score row evaluates all three hypotheses on all five probes. The correct
  position is balanced 7/7/6; each false hypothesis matches between one and four
  probes, so label position, perfect-versus-zero score, and early-stop shortcuts do
  not solve the task.
- Every repair has one first consequential wrong state and an incorrect draft final
  state. Every true execution transition changes state.
- The predecessor's distinctive canonical vocabulary is forbidden and absent. The
  prompt uses ordinary operation descriptions with two wording variants. The fresh
  local seed is not materialized or inspected during design.

Read: the construction truth is strong. It does not prove the model will internalize
the state table when the local prompt does not request one; that interface transfer is
the intended falsifiable uncertainty.

## Control and compute audit

Both arms independently start from the authenticated `close_xi` parent (weights
`16e9dc75...c179`, config `de953bd5...7ff`). The failed scaffold adapter is not
inherited. Each arm contains 320 rows, exactly 286,814 encoded forward tokens, zero
skips, 40 effective-batch-8 optimizer steps, and 200 byte-identical replay rows in the
same positions.

The active control's variable block is 120 replay rows / 87,454 tokens. The candidate
replaces that block with 80 curriculum rows / 48,806 tokens plus 40 disjoint replay
rows / 38,648 tokens. Candidate minus control target-token deltas are +1,196 prompt,
-1,955 thought, 0 close, and +759 answer, summing to zero forward-token difference.
Thus total compute is exact, while target composition is intentionally not identical.
Score lessons consume 25,683 of the curriculum's 48,806 tokens; any positive result
belongs to the four-stage package and cannot isolate a component.

## Gate reachability and promotion audit

The 26-case local event has a nonempty pass region. Accuracy ≥0.65 requires 17/26;
parse ≥0.90 requires 24/26; cap contacts ≤2 and route abstentions ≤1 are attainable.
Execute, induct, and probe each have two rows and require at least 1/2. The candidate
must additionally beat both parent and replay strictly in total correct answers and
strictly over the six execute/induct/probe rows. A perfect control makes the relevant
strict gate impossible and therefore causes a fail-closed stop; the bar will not be
lowered after observation.

Only `state_table_after_close` can promote. Failure produces an empty promotion
receipt and seals aggregate seed 78,138. A local pass merely opens one aggregate-only
quick event; broad promotion then requires a positive aggregate, strict improvement
on all ten public families versus base, aggregate at least the strong blend, and
aggregate strictly above replay refresh, the immediate parent, and the active replay
continuation. Even that pilot is not a universal claim: it must be replicated at the
higher tier and beat matched-compute sample-more in a fresh result-separated
confirmation experiment.

## Contamination, backend, and operational audit

- Training data is fresh procedural synthesis plus the inherited clean replay pool.
  No `benchmarks/` path is imported or read.
- The broad runner invokes only `scripts/run_benchmark_aggregate.py`, receives only
  aggregate/per-family scores, and keeps all six paired arms on `qwen_vllm` with the
  same quick tier, 1,024-token budget, and seed.
- Local comparison uses one Transformers process for parent, replay, and candidate;
  no local score is compared to a vLLM score.
- The harness permits exactly one expensive stage per invocation. It requires a clean
  worktree, and every predecessor training/local/merge receipt must already be
  committed byte-for-byte at `HEAD`. The prescribed cadence is commit, fetch/rebase,
  push `main`, and verify both workflows before the next stage.
- The machine-readable design receipt
  `data/design_receipt.json` (`0bac3340...ef837`) binds truth, source/stream hashes,
  exact compute, reachable gates, benchmark firewall, and checkpoint policy.

## Remaining adversarial risks

1. Twenty commit rows expose an already verified final state and may teach copying
   rather than computation. Sixty other rows require execution, scoring, or repair;
   the commit block is an explicit emission intervention. No component-level causal
   claim is allowed.
2. State-table formatting may become a surface habit that does not appear under
   ordinary local prompts. The absence of table vocabulary at evaluation is a
   deliberate transfer test, not something to repair after the event.
3. One local seed and two cases per kind are noisy. Strict control-relative admission
   reduces false promotion but can reject a useful model. A rejection is preserved;
   the seed and thresholds cannot be tuned.
4. The authenticated parent was selected from prior evidence and may carry selection
   bias. Same-parent replay and explicit parent evaluation are mandatory, and broad
   claims require independent confirmation.
5. Equal forward tokens do not equalize gradient content. That content difference is
   the intervention; train loss is not a capability comparison.

No objection makes the experiment redundant, contaminated, structurally impossible,
or causally uninterpretable under its registered package-level claim.

**Verdict:** `PASS_EXPENSIVE_RUN`.

This verdict authorizes only the one-stage-at-a-time commands in `scripts/run.py`, and
only after this exact design checkpoint is committed, rebased, pushed to `main`, and
both GitHub workflows pass. It does not authorize benchmark access before an
authenticated local promotion receipt.
