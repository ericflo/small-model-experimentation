# Adversarial Design Review

## Scope and verdict boundary

This review covers only model-free task construction, freshness, merged-parent
authentication, one rollout event, deterministic failure selection, and the clean
restart schema. It inspected repository-owned procedural artifacts and published
receipts. It made no model call and read no benchmark source, item, transcript, or
detailed result.

Actual failure availability and token exposure do not exist yet. This review cannot
authorize stream construction, training, local evaluation, or benchmark access.

## Closest duplicate and causal change

The closest duplicate is
`qwen35_4b_universal_on_policy_prefix_repair_token_match`. Its real parent failures
were a valid selection substrate, but the candidate still conditioned on 47,123
masked wrong-prefix tokens and received 33,421 fewer target tokens than replay. The
result was worse overall and 0/6 on execute/induct/probe.

The present design changes two causal facts at once, prospectively and explicitly:

1. parent behavior selects tasks but never enters training context; and
2. replay and candidate must match both total forward compute and supervised target
   exposure.

The package-level question is therefore whether balanced failure-selected clean
recomputation beats same-parent replay. It does not isolate task selection from
context removal, and a positive result may not be described as proof of either alone.

## Truth, breadth, and freshness audit

- Construction seed 77,114 deterministically emits 624 rows, 48 for each registered
  universal skill. The executable-truth source hash is `81edc9ea...de304`.
- Each source row carries a nonempty solution, one-line answer, and a truth-valid
  audit. The model-facing file hash is `25382689...0f5b` and contains only id,
  messages, and routing metadata.
- All 624 task ids and canonical messages are unique. Message overlap is zero against
  predecessor collection/local sources and regenerated local seeds 88,000–88,009.
- Selection is balanced over all 13 skills rather than six hand-named failure classes.
  This improves breadth but does not estimate natural deployment frequency.
- The local seed and aggregate seed are only numbers in this review; no corresponding
  prompt or benchmark content has been materialized.

The generator's oracle solutions are concise relative to the 1,024-token deployment
cap but are not claimed minimal. They are permitted because the mechanism is a full
restart from the prompt, not a first-error patch.

## Parent and backend audit

The parent is the stronger published replay control from the predecessor, not the
weaker `close_xi` parent. Its tracked merge receipt binds 128/128 applied nonzero LoRA
modules and weight SHA-256 `7ab4c419...36e2e`. The full local composite has the
expected 9,078,620,536-byte weight and external receipt.

The scaffolded current vLLM template was extended only with the already-tested
explicit-composite override: it validates the exact Qwen3.5-4B architecture
fingerprint, makes runtime adapter and model override mutually exclusive, loads the
local tokenizer/config, records a null hub revision, and preserves the template's
sampling and provenance behavior. Unit tests reject wrong architectures and mixed
runtime-LoRA/override use.

Collection is one event for all 624 tasks under the frozen greedy natural-thinking
geometry. Splitting or rerunning only bad/missing rows would change RNG/batch behavior
and is forbidden. Infrastructure failure preserves logs and requires an explicit
recovery audit; it never silently resamples.

## Selection and shortcut audit

The selector uses only exact executable answers and public deployment diagnostics.
Hard failures are cap contact, missing answer, or wrong answer. Correct answers with
more than 128 thinking tokens are bounded-compute policy failures. The threshold is
frozen before observation and aligns with the intended short restart interface; it
is not an accuracy claim.

Four eligible rows are required independently in each skill. Hard failures rank
before budget-only failures, then longer thinking and a seeded hash determine order.
This avoids outcome-aware quota borrowing, but it can select a tail distribution and
overrepresent verbosity. That selection bias is the intervention, not an estimate of
task prevalence.

The emitted row is reconstructed from the procedural source, not from parent text.
Tests prove that messages, oracle think, and oracle answer are preserved while
`assistant_prefix_token_ids` is absent and
`parent_prefix_in_training_context=false`. Parent output is represented only by a
SHA-256 and diagnostic provenance fields outside the model-facing target.

## Compute-control audit

The planned 320-row layout retains 200 identical aligned replay positions. Candidate
and control variable blocks have equal cardinality: 52 restarts plus 68 replay versus
120 replay. Equal cardinality and updates are insufficient, so three exact exposure
axes are mandatory after tokenization:

- total forward tokens;
- loss-bearing/nonzero target tokens; and
- absolute loss mass under the frozen thought/close/answer weights.

No padding, target masking, oracle rewriting, truncation, or duplication is allowed to
manufacture equality. Failure of the deterministic solver is a preserved feasibility
negative. A second review must inspect the concrete selected indices, zero-skip
receipt, span composition, optimizer geometry, and copied replay lineage before any
training arm runs.

## Contamination and checkpoint audit

- Construction and selection use only fresh procedural tasks and experiment-owned
  receipts. No benchmark module or path is imported by the harness.
- Aggregate seed 78,140 is sealed. Fresh local seed 88,010 is not materialized.
- Each stage requires a clean `main` whose HEAD equals `origin/main`, and the preceding
  receipt must be byte-identical to the object committed at HEAD.
- Repository policy adds `make check`, fetch/rebase, post-rebase smoke/check, push,
  and both GitHub workflow checks between every expensive event.
- Quota failure, solver infeasibility, and local failure are results, not permission
  to change seeds, thresholds, or quotas in this directory.

## Remaining risks

1. A 128-token budget can select correct but verbose behavior, so the package mixes
   accuracy correction with bounded-compute compression. Results must report the
   hard-versus-budget-only composition.
2. Full oracle restarts may act like ordinary synthetic SFT. The same-parent replay
   control and exact target exposure are necessary but cannot decompose all semantic
   differences.
3. Four examples per skill is balanced but small. A local negative can be noisy; no
   post-result dose increase is allowed in this directory.
4. One rollout seed can make quota availability stochastic. The fixed 48-task pool
   and fail-closed rule trade power for interpretability.
5. Exact exposure does not equal exact gradient difficulty. Capability gates, not
   training loss, decide promotion.

No current objection makes the parent rollout contaminated, structurally impossible,
or uninterpretable at the package level.

**Verdict:** `PASS_PARENT_ROLLOUT`.

This verdict authorizes exactly one `collect-parent` stage from the committed,
pushed, CI-green design checkpoint. It does not authorize mining before collection is
published, or any training, evaluation, merge, benchmark, or aggregate access.

## Post-selection composition note

The frozen selector later cleared every quota. As anticipated in Remaining Risk 1,
the 128-token policy budget is broad: 598/624 rows exceed it. Hard failures still rank
first and account for 40/52 selected rows; the other 12 are budget-only rows in
abstain, count, route, and select, whose hard-failure pools were smaller than four.
This does not revise the verdict or selection. The second compute review must preserve
and report that mixed accuracy-plus-bounded-compute causal unit.
