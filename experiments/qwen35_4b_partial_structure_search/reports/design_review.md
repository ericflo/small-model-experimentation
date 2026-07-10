# Adversarial Design Review: partial-structure recognition-guided search

Verdict before implementation: **sound with mandatory gates**. The full search is not authorized until both
the model-free oracle-state gate and the recognition-actionability gate pass.

## Lens 1: novelty and mechanism

### Risks found

1. C25/C26 already test next-operation ranking, including thinking. Calling another next-op ranker “partial
   recognition” would be a duplicate.
2. `qwen_prefix_state_process_verifier` learned canonical-prefix consistency (AUC 0.937) for only modest search
   gains. Canonical-prefix labels would repeat that weakness.
3. `qwen_semantic_prefix_value_model` learned broad semantic reachability (AUC 0.853) but did not improve
   decoding. AUROC alone is not evidence of a useful controller.

### Required fixes adopted

- A node is a parameter-free operation-TYPE prefix. The judge receives it externally and predicts whether any
  shared parameterization and suffix can finish it. It does not propose the next concrete operation and sees
  no materialized intermediate state.
- A prefix is live iff it prefixes at least one fully enumerated semantic solver, including noncanonical
  factorizations. Canonical target membership is never the label.
- Store completion counts/densities as well as binary live labels. Gate on sibling live recall and whole-path
  survival, not pooled AUROC.
- Score-shuffled thinking is mandatory: equal paid reasoning with destroyed sibling alignment directly tests
  whether score content, rather than compute or beam structure, causes a gain.

## Lens 2: task validity and leakage

### Risks found

1. The inherited `min_depth_leq` stops after 60,000 seen states and returns `False`; at depth 5 that can silently
   relabel cap exhaustion as “true depth.” The inherited generator also verifies only through depth 3.
2. Oracle probes or target pipelines could leak through generic task serialization.
3. Visible-pass programs may fail hidden examples, and stopping at the first visible passer would create an
   arm-dependent overfit artifact.

### Required fixes adopted

- Implement an exact, exhaustion-audited depth<=4 search with no seen cap. Every task stores its receipt.
- Use disjoint visible, oracle-label-probe, and final-hidden example sets. Exact min-depth and semantic labels
  use all oracle data; model prompts use visible examples only.
- Render prompts from a whitelist rather than serializing task dictionaries. Unit-test that deleting target,
  probes, hidden data, labels, and counts leaves prompt bytes unchanged.
- Every arm collects all visible-passers under the same fill cap and uses the same visible-only consensus
  selector. Hidden examples grade only after selection.
- Deduplicate splits by behavior on a frozen probe bank, not target string alone.

## Lens 3: oracle target usefulness

### Risk found

Existential reachability may be so dense that even a perfect binary judge does not meaningfully compress the
tree. The previous semantic-prefix experiment observed raw positive densities above one half.

### Gate adopted

Before GPU scoring, enumerate all successful full skeletons, mark every live prefix, and report by slot:

- live prefix and live-child density;
- completion-count distribution;
- random recall@4;
- oracle path survival;
- full leaves, retained leaves, parameter fills, primitive applications, and wall time.

Proceed only if oracle-live beam retains a hidden solver on at least 90% of development tasks and completes at
least 10x fewer skeletons than full enumeration. A failure means the search state is wrong, regardless of how
interesting a judge might appear.

## Lens 4: calibration and statistics

### Risks found

1. Pooled AUROC can be a pure task-difficulty signal; C47 pooled/no-think looked useful while within-task AUROC
   was chance.
2. Random negatives make an easy classifier but not an on-policy search controller.
3. Selecting a judge budget or beam from primary depth-5 results would invalidate the comparison.

### Required fixes adopted

- Calibration groups are all 16 siblings of live, hard-negative, uniform-frontier, likelihood-frontier, and
  one-edit parents. Report task-macro and prefix-depth-stratified AUROC.
- Primary action metrics are best-live rank, live recall@4 within viable parents, and simulated complete-path
  survival. Task-shuffled prompts and surface features are canaries.
- Freeze thinking budget 256 and beam width 4 before primary search. No primary-set tuning.
- Use task-cluster bootstrap intervals and paired differences. Recognition gate: AUROC >=0.65 with lower bound
  >0.50, >=0.05 paired lift over the strongest non-oracle baseline, and recall@4 lift >=0.10 with lower bound
  >0.
- A calibration win without retention improvement is reported as “readable but non-actionable” and stops.

## Lens 5: backend and compute parity

### Risks found

1. C47/C26 use Transformers generation paths. Mixing their scorer with vLLM direct sampling would invalidate
   matched arms.
2. Repeated judge prompts incur large prefill cost that sampled-token totals hide.
3. Depth 5 was projected, not measured; it may be CPU-manageable rather than “intractable.”

### Required fixes adopted

- Every model arm uses the pinned local vLLM runner. Derive A/B and A–P token IDs dynamically and require one
  token each.
- Thinking P(True) uses two passes: retain the thought, append the exact `</think>\n\nAnswer: ` token prefix,
  then request targeted A/B logprobs. Assert token round-trip and record both passes.
- Freeze prompt order, batch shape, seeds, scheduler, and concurrency. Store runner sidecars.
- Report a resource vector: logical prefill tokens, sampled reasoning tokens, terminal readout tokens, requests,
  prefix nodes, completed skeletons, parameter fills, visible executions, primitive applications, GPU/CPU
  seconds, and wall time.
- Measure full depth-5 brute wall time. Never describe it as intractable merely because `16^5` is large.
- Direct sample-more uses one frozen 512-sample pool and two deterministic sample-index prefixes: one matched
  to recognition's sampled tokens and one matched to its total logical model tokens (prefill plus sampled).
  Pool exhaustion is fail-closed. Both use the identical parser, filler, executor, selector, and fill cap.

## Pre-registered verdicts

- **G0 wrong state:** oracle-state gate fails; stop before model claims.
- **G1 unreadable:** oracle passes, recognition calibration fails; partial viability is not readable here.
- **G2 non-actionable:** AUROC passes but live-child/path retention fails; do not run primary search.
- **G3 contribution without frontier advance:** search beats score shuffle but not matched sample-more or
  proposal likelihood.
- **G4 frontier advance:** thinking-guided hidden selected success is >=0.10 above both sampled-token- and
  total-token-matched sample-more with paired CI lower bound >0, beats shuffle and next-op likelihood, and
  agrees against both direct arms across both frozen shards.

Only G4 permits a separately scaffolded Recognize -> Search -> Bank follow-up.
