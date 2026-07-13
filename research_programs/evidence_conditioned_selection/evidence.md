# Evidence

## Seed Experiments

- [qwen35_4b_retrieval_adapt_verify_scale](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [qwen35_4b_foofah_selective_program_fallback](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [qwen35_4b_foofah_program_ensemble_consensus](../../experiments/qwen35_4b_foofah_program_ensemble_consensus/reports/report.md)
- [qwen35_4b_independent_retrieval_consensus](../../experiments/qwen35_4b_independent_retrieval_consensus/reports/final_report.md)

## Key Result

- [qwen35_4b_partial_structure_search](../../experiments/qwen35_4b_partial_structure_search/reports/report.md)
  adds a hard negative boundary to confidence-based control. On 7,200 unfinished type-prefix children,
  thinking P(viable) had pooled AUROC 0.557 but task-macro AUROC 0.506 and recall@4 0.251; task-shuffled visible
  evidence was no worse. No-think was modestly stronger (0.556 AUROC, 0.303 recall) but still below the frozen
  actionability threshold. Thus C10/C46-style confidence selection does not transfer automatically from
  completed candidates to existential partial reachability. Search controllers must be evaluated within task
  and at the actual retained beam, and should expose residual evidence rather than only symbolic prefix names.

- [qwen35_4b_generator_verifier_gap](../../experiments/qwen35_4b_generator_verifier_gap/reports/report.md)
  (claim C10): the C2 wall is **plumbing, not capability**. A frozen 4B's black-box self-verification is
  weak/yes-biased with no-think (AUROC 0.77) but strong with thinking (AUROC 0.93); its own zero-training,
  deployable thinking-verifier selects best-of-8 to close **75%** of the pass@1(0.771)→oracle(0.890) gap
  (no-think 24%). Foreign-solution reject rate 1.00. So the model CAN tell its own good solutions from bad —
  once it thinks — and selection has real headroom.

- [qwen35_4b_verifier_selector_showdown](../../experiments/qwen35_4b_verifier_selector_showdown/reports/report.md)
  (matched-cost follow-up): on one k=8 pool, the thinking-verifier is **Pareto-dominated** when a cheap visible
  test exists — standalone 0.860 ≈ visible-only 0.850 at ~5× cost; the deployable sweet spot is **visible +
  free no-think verifier (0.870)**, 83% of the pass@1→oracle gap. Expensive thinking-verification only earns
  its cost in verifier-only settings. So the C2 wall is fixable with *cheap* plumbing.

- [qwen35_4b_code_confidence](../../experiments/qwen35_4b_code_confidence/reports/report.md)
  (claim C46, MBPP leg): in the verifier-free code regime, single-token P(True) is the selector to beat. MBPP
  P(True) selection 0.762 beats public-output majority 0.721 and random 0.696. When a visible test exists,
  execute it first (MBPP visible-test execution 0.816), then use confidence for abstain/route or no-test
  settings.

- [qwen35_4b_humaneval_code_confidence](../../experiments/qwen35_4b_humaneval_code_confidence/reports/report.md)
  (claim C46, HumanEval replication): in the strict no-public-probe setting, P(True) selection 0.835 beats
  mean-logprob 0.787 and random 0.766, with oracle pass@8 0.872. This is the clean no-verifier selection
  replication.

- [qwen35_4b_answer_potential_trace_sft](../../experiments/qwen35_4b_answer_potential_trace_sft/reports/report.md)
  (claim C51): same-model canonical-answer likelihood after a sampled thought is **real but not
  actionable** under the tested forced-close protocol. Within-task AUROC was 0.617 (gate 0.65), and
  top-one success beat random/shortest by +0.073/+0.058 but missed the frozen +0.10 bar. Real thoughts
  beat token-shuffled and foreign controls and format ranks were stable, yet 99.37% of thoughts hit the
  cap and autonomous answers parsed only 13.2%. G0 failed and correctly stopped before SFT.

- [qwen35_4b_same_prefix_advantage_routing](../../experiments/qwen35_4b_same_prefix_advantage_routing/reports/route_diagnostics.md)
  adds a training-time selection boundary. Three policies' absolute
  continuation estimates replicated well across disjoint four-branch halves
  (`r=0.79`--`0.86`), but conditioning on the statewise maximum was unstable:
  only 6/26 block-1 quick selections remained quick winners on audit, and its
  apparent `+0.319` margin over the student became `-0.019`. Thresholds of
  `+0.10` and `+0.25` did not repair the sign. Selection quality must be
  measured after conditioning, on disjoint outcomes, rather than inferred
  from component-score reliability.

- [qwen35_4b_counterfactual_order_support_selector](../../experiments/qwen35_4b_counterfactual_order_support_selector/reports/report.md)
  adds a forward-counterfactual boundary. The label-free mean per-alias ordered-
  minus-exact-shuffle probability reached 43/113 (0.381), beating first trace
  31/113 and majority 33/113 with positive paired lower bounds. But it was only
  +2 tasks over minimum entropy and +3 over max confidence, with lower bounds
  -0.035/-0.027, and an oracle-balanced task-mismatched shuffle reached 44/113.
  Thus the replicated coherent-content group effect contains weak selection
  information but raw subtraction is not a robust, task-specific selector.
  Confirmation and the K=3-versus-K=6 matched-compute successor stayed sealed.

## Current Read

The biggest strategic gap is selection under deployable evidence — and C10/C46 say that gap is *fixable* with
cheap plumbing: execute visible evidence when it exists, then use a free no-think P(True) readout for
verifier-free selection, abstention, or routing. Thinking-verification is reserved for verifier-only settings
where its added cost beats the no-think readout. Future selection work
should (a) benchmark against the thinking-verifier before building trained selectors, (b) treat native
thinking as a verification lever (not only generation), and (c) still report oracle coverage only as a
diagnostic with the deployable decision rule as the main object. Top follow-up: wire the thinking-verifier
into a controller (vs/with the visible test) and measure the deployable accuracy-vs-token Pareto.

The partial-structure result narrows that optimism: confidence is useful when correctness is readable from a
completed candidate, but an existential unfinished-state judgment can collapse to task difficulty. Any
controller follow-up must report task-macro discrimination, sibling recall at the deployed beam, a task-
shuffled evidence canary, and prefill-inclusive compute—not just pooled AUROC.

C51 adds a second boundary: a score may contain trace-specific information and still condition on an
unreachable deployment state. Oracle-side trace selectors must predict fresh autonomous outcomes within
task, clear a practical top-choice effect-size gate, and include termination/parseability in validation.
Do not scale a dense score because its corruption controls pass; first prove the scored seam is deployable.

The same-prefix result adds winner conditioning to that checklist. Reliable
component scores do not imply a reliable argmax label when differences are
small and the selected tail is reused as the curriculum. For policy routing,
estimate direct advantages with cross-fitting, expose abstention and per-route
support, and retain independent block signs; a positive pooled router average
cannot certify every named teacher.

The order-support negative adds a group-to-instance warning. A perturbation can
causally improve correctness on average and its signed probability vector can
beat hard voting, yet still fail against cheap confidence/entropy or a relevance
control. Before spending matched compute, require the counterfactual readout to
add task-specific information beyond ordinary probability geometry.

The subsequent J-branch mechanics negative reinforces the proposal/value split:
moving before commit is not enough when the intervention cannot write its own
supplied hypothesis. All numeric controls passed, but additive J target selection
was exactly chance. Proposal-shifting systems need a label-free write/coverage
gate before selector or matched-compute evaluation.
