# Idea Intake: partial-structure recognition-guided search

## Program Fit

- Program: `structured_execution_and_compilers` (primary), `evidence_conditioned_selection` (secondary).
- Existing or new program: existing programs; no registry change.
- Closest program scorecard reviewed: Structured Execution and Evidence-Conditioned Selection.
- Related future queue item: none. The direct provenance is C48 report, Next Experiments item 4.

## Prior Evidence

- Anchor 1: `qwen35_4b_latent_decomposition` (C25), which ranked the next concrete operation at depth 3.
- Anchor 2: `qwen35_4b_structure_search_scaling` (C35), which measured brute structure search through depth 4
  but only projected depth 5.
- Anchor 3: `qwen35_4b_verifier_free_banking` (C47), where thinking P(True) separated completed computational
  candidates but no-think P(True) was within-task chance.
- Further anchors: `qwen35_4b_thinking_lookahead` (C26), `qwen_prefix_state_process_verifier`,
  `qwen_semantic_prefix_value_model`, and `qwen35_4b_hypothesize_verify_wall` (C48).
- Closest duplicate or near-duplicate: `qwen_semantic_prefix_value_model`. It trained a separate legacy
  semantic-reachability value head and obtained AUC 0.853 without a search gain. The present test uses only
  frozen Qwen3.5-4B's own thinking judgment, a contamination-free list DSL, exact alternative-completion
  labels, actionable frontier-retention gates, and matched-compute depth-5 search.

## Novelty Claim

This is the first test of frozen Qwen3.5-4B's own thinking-based, visible-only semantic viability judgment
over externally supplied operation-type prefixes at exact depth 5, gated on within-parent path retention and
compared with matched model-token and interpreter-work search baselines.

It is recognition, not proposal: the question is “can this supplied partial structure still work?”, not
“what is the canonical next operation?” It is also not a canonical-prefix classifier: every prefix of every
semantically successful completion is positive.

## Mechanism

Thinking may allow the model to mentally test whether some parameters and remaining operations could reconcile
the visible examples. Ranking children by that judgment could preserve a rare live branch without enumerating
the full `16^5` space. The explanation is false if a task-shuffled prompt scores equally, if scores only track
prefix length/op identity, if they do not retain live siblings, or if score shuffling performs the same search.

## Control Plan

- Baseline: direct matched-compute type-skeleton sampling plus identical value fill, visible filtering, and
  visible-only selection.
- Mechanism-falsifying control: shuffle thinking scores among siblings after paying the same judgment cost.
- Proposal control: C25-style no-think next-operation letter likelihood over the same 16-type menu.
- Serial-compute control: no-think P(True).
- Surface control: cross-fitted features using only prefix length, op identities/arities, and visible-shape
  summaries.
- Shift check: calibrate at exact depth 4, evaluate search on disjoint exact depth 5, and report two frozen
  depth-5 shards.
- Hidden-label boundary: prompts are rendered from a whitelist and byte-audited after all oracle fields are
  deleted. Hidden and label probes are never used for search termination or selection.

## Evidence Output

- Program evidence update: whether partial semantic recognition is a useful bridge between weak proposal and
  strong complete-candidate verification.
- Claim/synthesis update: narrow C35's untested depth-5 extrapolation; add a new claim only after the pending
  re-grade and only if the result is durable.
- Reusable artifact: exact exhaustion-audited min-depth generator, semantic live-prefix oracle, vLLM two-pass
  A/B scorer, and resource-accounted search harness.
- Stop/branch condition: stop at failed oracle or recognition gate. On a search win, create a separate fresh
  banking experiment with disjoint harvest and evaluation data.

## Decision

- Run experiment: yes, selected by the user after a portfolio review.
- Create program: no.
- Write synthesis only: no; the seam is empirically unresolved.
- Defer: banking until a search win.
