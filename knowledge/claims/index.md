# Claim Index

Generated from `knowledge/claims/claim_ledger.json`. Edit the ledger, not this file.

- Claims: 12

## Status Counts

| Status | Claims |
| --- | ---: |
| Confirmed | 5 |
| Negative | 1 |
| Open | 1 |
| Promising | 5 |

## Program Counts

| Program | Claims |
| --- | ---: |
| `active_evidence_acquisition` | 1 |
| `algorithmic_memory_and_retrieval` | 1 |
| `benchmark_generalization` | 1 |
| `collective_experimentation_infrastructure` | 2 |
| `evidence_conditioned_selection` | 3 |
| `interpretability_and_diagnostics` | 1 |
| `operator_and_skill_inventories` | 1 |
| `posttraining_and_adaptation` | 3 |
| `process_control_and_tool_use` | 3 |
| `reliability_and_safety` | 4 |
| `structured_execution_and_compilers` | 3 |
| `test_time_reasoning_budget` | 2 |

## C1: Structured intermediates improve small-model reliability

- Status: `Confirmed`
- Programs: `structured_execution_and_compilers`, `reliability_and_safety`
- Summary: Executable, typed, latent, or otherwise structured intermediates repeatedly create checkable surfaces that direct final-answer prompting lacks.
- Implication: Future direct-answer tasks should include a structured-output or executable-evidence baseline before making broad claims.

### Evidence

- [`qwen_structural_latent_compiler_expansion`](../../experiments/qwen_structural_latent_compiler_expansion/reports/structural_latent_compiler_expansion_report.md)
- [`qwen35_4b_foofah_selective_program_fallback`](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [`qwen35_4b_operator_inventory_search_pilot`](../../experiments/qwen35_4b_operator_inventory_search_pilot/reports/qwen35_4b_operator_inventory_search_pilot_report.md)

### Next Tests

- Compare direct text, typed bytecode, latent slots, and stateful executors on one shared suite.
- Ablate supervision source separately from representation.

### Avoid

- Another isolated structured-output win without direct-output and corrupted-intermediate controls.

## C2: Candidate coverage does not imply deployable accuracy

- Status: `Confirmed`
- Programs: `evidence_conditioned_selection`, `reliability_and_safety`
- Summary: Candidate pools can contain hidden-correct answers while visible evidence still selects or commits incorrectly.
- Implication: Candidate-pool work should report oracle coverage, deployable selection, false-pass rate, precision, recall, and abstention separately.

### Evidence

- [`qwen35_4b_retrieval_adapt_verify_scale`](../../experiments/qwen35_4b_retrieval_adapt_verify_scale/reports/final_report.md)
- [`qwen35_4b_foofah_selective_program_fallback`](../../experiments/qwen35_4b_foofah_selective_program_fallback/reports/report.md)
- [`qwen35_4b_independent_retrieval_consensus`](../../experiments/qwen35_4b_independent_retrieval_consensus/reports/final_report.md)

### Next Tests

- Build visible-only selector benchmarks with family-held-out candidate pools.
- Convert oracle-ceiling reports into deployable-gap scorecards.

### Avoid

- Claiming selection progress from hidden-oracle coverage alone.

## C3: Plain prompt memory is not enough

- Status: `Negative`
- Programs: `algorithmic_memory_and_retrieval`, `operator_and_skill_inventories`
- Summary: Naive retrieved skill cards are a weak memory mechanism unless retrieval contributes candidates, tests, constraints, or verifier inputs.
- Implication: Memory experiments should include random, mismatched, and constraint/test/candidate variants rather than only prompt-context variants.

### Evidence

- [`qwen_verified_skill_memory_rag`](../../experiments/qwen_verified_skill_memory_rag/reports/qwen_verified_skill_memory_rag_report.md)
- [`qwen35_4b_verified_algorithm_retrieval_adaptation`](../../experiments/qwen35_4b_verified_algorithm_retrieval_adaptation/reports/final_report.md)

### Next Tests

- Compare retrieved examples, retrieved algorithms, retrieved tests, and retrieved failure cases on one task family.

### Avoid

- Treating retrieval as useful because it looks semantically relevant but is never verified.

## C4: Active evidence must be coupled to downstream decisions

- Status: `Promising`
- Programs: `active_evidence_acquisition`, `process_control_and_tool_use`, `evidence_conditioned_selection`
- Summary: Actively selected examples and probes can help, but the acquisition objective must optimize the final selector, verifier, or controller decision.
- Implication: Evidence-acquisition work should report downstream decision lift per evidence budget, not only probe informativeness.

### Evidence

- [`qwen_active_example_acquisition`](../../experiments/qwen_active_example_acquisition/reports/qwen_active_example_acquisition_report.md)
- [`qwen35_4b_active_counterexample_trace_selection`](../../experiments/qwen35_4b_active_counterexample_trace_selection/reports/qwen35_4b_active_counterexample_trace_selection_report.md)
- [`qwen35_4b_learned_active_trace_policy`](../../experiments/qwen35_4b_learned_active_trace_policy/reports/qwen35_4b_learned_active_trace_policy_report.md)

### Next Tests

- Compare active probe policies by selector lift under identical budgets.
- Separate expected-output-free probes from oracle-only diagnostics.

### Avoid

- Optimizing examples or probes without showing commit, repair, or selection improvement.

## C5: Adaptation must beat strong frozen alternatives

- Status: `Open`
- Programs: `posttraining_and_adaptation`, `process_control_and_tool_use`
- Summary: LoRA, DPO, distillation, GRPO, and DAgger are promising, but their value is unclear unless compared against frozen sampling, verification, retrieval, and tool-control alternatives.
- Implication: Post-training runs should include frozen inference-time alternatives and explicit deployability boundaries.

### Evidence

- [`qwen35_4b_constrained_coverage_dpo`](../../experiments/qwen35_4b_constrained_coverage_dpo/reports/final_report.md)
- [`qwen35_4b_live_tool_dagger`](../../experiments/qwen35_4b_live_tool_dagger/reports/report.md)
- [`qwen35_4b_oracle_process_grpo`](../../experiments/qwen35_4b_oracle_process_grpo/reports/qwen35_4b_oracle_process_grpo_report.md)

### Next Tests

- Run one update method against frozen inference-time alternatives on the same selector or compiler benchmark.

### Avoid

- Adapter wins whose labels are hidden-only or whose artifacts are not auditable.

## C6: Controls are the difference between a result and a story

- Status: `Confirmed`
- Programs: `benchmark_generalization`, `interpretability_and_diagnostics`, `reliability_and_safety`
- Summary: The strongest imported lessons come from experiments that separate mechanism from confounders through corrupted, shuffled, random, frozen, length-held-out, or family-held-out controls.
- Implication: Future claims should name the mechanism-falsifying control before running expensive work.

### Evidence

- [`factor_recombination_ladder`](../../experiments/factor_recombination_ladder/reports/factor_recombination_ladder_report.md)
- [`feature_factorized_rule_diversity`](../../experiments/feature_factorized_rule_diversity/reports/feature_factorized_rule_diversity_report.md)
- [`qwen_structural_compiler_attribution_ablation`](../../experiments/qwen_structural_compiler_attribution_ablation/reports/structural_compiler_attribution_ablation_report.md)

### Next Tests

- Build a shared generalization and control suite used across compiler, selector, memory, and adaptation work.

### Avoid

- Treating train or IID lift as evidence for generalization.

## C7: Reliability requires explicit hidden-label and artifact boundaries

- Status: `Confirmed`
- Programs: `reliability_and_safety`, `collective_experimentation_infrastructure`
- Summary: The repository should preserve small reproducible artifacts while excluding trained adapters and labeling oracle-only evidence separately from deployable evidence.
- Implication: Reports and scaffolds must separate deployable evidence from hidden-oracle evaluation, and trained adapter directories must remain out of git.

### Evidence

- [`qwen35_4b_reliability_exec_opsd_audit`](../../experiments/qwen35_4b_reliability_exec_opsd_audit/reports/final_report.md)
- [`qwen35_4b_real_sample_verify_commit`](../../experiments/qwen35_4b_real_sample_verify_commit/reports/qwen35_4b_real_sample_verify_commit_report.md)
- [`.gitignore`](../../.gitignore)
- [`scripts/validate_repository.py`](../../scripts/validate_repository.py)

### Next Tests

- Add stronger hidden-label boundary and artifact-manifest checks as new experiments are created.

### Avoid

- Checking in trained adapters or letting oracle-only metrics become headline deployment claims.

## C8: The repository itself is a research instrument

- Status: `Confirmed`
- Programs: `collective_experimentation_infrastructure`
- Summary: Program charters, generated indexes, claim ledgers, scorecards, intake notes, templates, validation, and CI are part of the experiment system because they determine whether future work compounds.
- Implication: New experiments should update program evidence and shared claims when they change what future work should believe.

### Evidence

- [`research_programs/registry.yaml`](../../research_programs/registry.yaml)
- [`knowledge/program_scorecards.md`](../../knowledge/program_scorecards.md)
- [`docs/quality_gates.md`](../../docs/quality_gates.md)
- [`.github/workflows/validate.yml`](../../.github/workflows/validate.yml)

### Next Tests

- Measure whether scorecards and intake notes reduce duplicate proposals and improve citation of prior evidence.

### Avoid

- Treating process as complete when it no longer improves research memory.

## C9: Native thinking's accuracy gain is coherent reasoning content at every budget (unused lever)

- Status: `Promising`
- Programs: `test_time_reasoning_budget`, `process_control_and_tool_use`
- Summary: Enabling Qwen3.5-4B's native thinking (disabled corpus-wide) raises deployable MBPP greedy pass@1 +15pp (0.76->0.91); greedy accuracy is non-monotonic (optimum ~1024, then decline), though sampled full-pass is roughly flat across budgets. A visible-test budget controller is an efficiency (not accuracy) win, bounded by C2 false-passes. NATURE OF THE GAIN (decomposed via filler/shuffle/foreign controls across budgets 512/1024/2048): the model USES thinking as CONTENT -- splicing a different task's thinking (foreign) collapses accuracy to ~3-4% because the model follows it to the wrong problem; pure forward compute (contentless '.' filler) ~= no_think at every budget; relevant-but-scrambled thinking (shuffle) ~= no_think; and coherent reasoning (real) is the ENTIRE gain (real-shuffle = +0.105/+0.108/+0.150 at 512/1024/2048 -- it GROWS with budget). So the thinking accuracy gain is coherent reasoning CONTENT at every budget, not compute, scaffold, or token-presence. The earlier 'mostly compute/scaffold, not reasoning' read was wrong -- it only appeared through a greedy-metric lens and a shuffle-protocol artifact in the scaling run (its '2048 shuffle~=real'). Separately, at the REPRESENTATIONAL level a linear probe on the answer-token activation finds correctness moderately decodable and thinking raises decodability, but shuffle>=real there (noisy, overlapping CIs) -- i.e. coherence improves ACCURACY without clearly improving internal correctness-decodability.
- Implication: Re-baseline CoT-substitute results against a fair, budgeted native-thinking baseline. Native thinking's accuracy benefit is genuine reasoning content the model uses (credit it as such) -- but judge 'is it reasoning?' with content controls (filler + foreign), not a single greedy number, which misled here.

### Evidence

- [`qwen35_4b_thinking_budget_scaling`](../../experiments/qwen35_4b_thinking_budget_scaling/reports/report.md)
- [`qwen35_4b_thinking_budget_controller`](../../experiments/qwen35_4b_thinking_budget_controller/reports/report.md)
- [`qwen35_4b_thinking_separability_probe`](../../experiments/qwen35_4b_thinking_separability_probe/reports/report.md)
- [`qwen35_4b_thinking_content_vs_compute`](../../experiments/qwen35_4b_thinking_content_vs_compute/reports/report.md)
- [`qwen35_4b_overthinking_content_ladder`](../../experiments/qwen35_4b_overthinking_content_ladder/reports/report.md)
- [`qwen_python_shaped_silent_executor`](../../experiments/qwen_python_shaped_silent_executor/reports/qwen_python_shaped_silent_executor_report.md)

### Next Tests

- High-budget (1024/2048) ladder to confirm the coherence advantage shrinks under overthinking.
- Contamination-controlled / harder substrate where the no-think baseline is weaker (more headroom).
- A learned controller using internal/uncertainty signals vs the visible-test C2 wall.

### Avoid

- Claiming thinking gains are 'reasoning' without a content control that removes token-presence.
- Pinning an exact optimal budget from single-seed n=100 gaps.

## C10: The C2 selection wall is plumbing not capability: a thinking-verifier closes most of it

- Status: `Promising`
- Programs: `evidence_conditioned_selection`, `test_time_reasoning_budget`
- Summary: For a frozen Qwen3.5-4B on MBPP, intrinsic black-box self-verification (judge a candidate correct/incorrect from the A/B logit, no execution/hidden tests) is WEAK with no-think (balanced-acc 0.627, AUROC 0.773, heavy yes-bias: says 'correct' 0.91 vs base pass 0.771) but STRONG with thinking (balanced-acc 0.827, AUROC 0.926). The model's own thinking-verifier, zero-training and fully deployable (no execution), selects best-of-8 at 0.860, closing 75% of the pass@1(0.771)->oracle(0.890) gap (no-think closes only 24%); foreign-solution reject rate is 1.00. So the C2 coverage-vs-selection wall is a plumbing/evidence problem, not a verification-CAPABILITY limit -- the selection program has real headroom and the lever is thinking-augmented self-verification (cheaper/stronger than the trained selectors the corpus favored). This inverts C9: thinking helps VERIFICATION (+0.20 balanced-acc) at least as much as GENERATION -- its deepest value may be helping the model recognize correct answers, not just produce them. MATCHED-COST CAVEAT (selector showdown): when a cheap visible test exists, the thinking-verifier is Pareto-DOMINATED -- standalone 0.860 barely beats visible-only 0.850 at ~5x token cost, and combined the no-think verifier ties it (both 0.870). The deployable sweet spot is visible + FREE no-think verifier (0.870, closes 83% of the pass@1->oracle gap). So thinking-verification's expensive edge is real only in verifier-only settings (no cheap execution signal); the C2 wall is fixable, but with cheap plumbing (visible test + no-think verifier), not expensive thinking-verification.
- Implication: The C2 selection wall is fixable with CHEAP plumbing: a visible/execution signal + a free no-think self-verifier. Reserve expensive thinking-verification for verifier-only settings. Evaluate native thinking as a verification lever, and always compare selector signals at matched token cost.

### Evidence

- [`qwen35_4b_generator_verifier_gap`](../../experiments/qwen35_4b_generator_verifier_gap/reports/report.md)
- [`qwen35_4b_verifier_selector_showdown`](../../experiments/qwen35_4b_verifier_selector_showdown/reports/report.md)

### Next Tests

- Wire the thinking-verifier as the controller signal vs/with the visible test; deployable accuracy-vs-token Pareto.
- Verify a think-generated (harder-negative) candidate pool, and on a contamination-controlled substrate.
- Iterated generate->self-verify->revise self-correction loop driven by the strong thinking-verifier.

### Avoid

- Reporting no-think self-verification as strong -- it is yes-biased (0.91) and weak (AUROC 0.77).
- Treating verifier-selected accuracy as the oracle -- it is deployable but below the pass@k ceiling.

## C11: Self-training on verified self-solutions banks capability; test-time execution feedback does not (contamination-free substrate)

- Status: `Promising`
- Programs: `structured_execution_and_compilers`, `posttraining_and_adaptation`
- Summary: On a FRESH, procedurally-generated, CONTAMINATION-FREE program-synthesis substrate (random depth-D compositions of list primitives, graded by held-out execution; reference oracle solves 100%), built to test elicitation of the FIXED Qwen3.5-4B (no teacher, no scaling), two contrasting levers: (M2, NEGATIVE) a neurosymbolic multi-turn REPL loop (draft -> execute -> real feedback -> refine) does NOT beat matched-compute independent sampling -- repl_real 0.287 @ ~3.9 gens sits on/below the sample_more curve which reaches 0.338 @ 5; and the execution-feedback CONTENT adds only +0.024 over a PAIRED no-feedback control (within noise). Where the model cannot sample a solution (depth >=3, all arms 0.0) feedback adds nothing. So test-time self-correction does not push past the frozen model's sampling distribution. (M3, POSITIVE) QLoRA-SFT on the 4B's OWN 189 execution-verified (prompt->code) solutions (no teacher) banks capability into deployable single-shot: held-out fresh think-greedy@1 0.224 -> 0.319 (+0.095, ~2.2 SE over N=210, +42% relative), pass@5 rises 0.310 -> 0.371 (NO diversity collapse), confirmed on two fresh seeds and broad across depths 1-3. Notably this self-training WORKS here but REGRESSED on contaminated MBPP (qwen35_4b_verifier_guided_self_improvement), implicating substrate/contamination -- not the method -- in that earlier failure. REPLICATED: a second, independently-trained adapter (fresh training data, seed 505) reproduced the held-out gain (+0.103, matching the original +0.111), so the banking effect is robust to the training data, not a single-adapter artifact. EXPERT ITERATION (M4): scaled into a 3-round flywheel (solve pool with current model -> accumulate verified pairs -> retrain from base -> eval), the banking gain COMPOUNDS monotonically 0.267 -> 0.356 -> 0.385 -> 0.393 (+0.126, +47%) with pass@5 rising throughout (no collapse) and each round harvesting more verified data (107->144->162 of 360 solved; 147->287 pairs). But it shows clear DIMINISHING RETURNS (+.089,+.029,+.008, plateauing) and is COVERAGE-BOUNDED: it lifts depths 1-2 but never cracks the depth-3 frontier the model can't sample. Self-training widens the deployable footprint of the model's own distribution; it does not extend the distribution's frontier.
- Implication: For small-model capability: don't try to read the frozen weights more cleverly at test time (execution-feedback self-correction doesn't beat sampling) -- BANK sampling-accessible capability into the weights by self-training on what the model can already verify. And evaluate self-training on CONTAMINATION-CONTROLLED substrates: MBPP-based self-improvement results are confounded, and a clean structured substrate is what lets honest self-improvement show up. On-mission: no teacher, no scaling.

### Evidence

- [`qwen35_4b_neurosymbolic_repl_substrate`](../../experiments/qwen35_4b_neurosymbolic_repl_substrate/reports/report.md)

### Next Tests

- The flywheel plateaus by round 3 and can't crack depth-3 -- what extends the FRONTIER (richer primitives, curriculum, or a genuinely new elicitation signal) without a teacher?
- Does the banked single-shot capability transfer to a DIFFERENT clean substrate, or only within-family?
- Re-run the corpus's failed MBPP self-improvement with contamination controls to confirm the substrate hypothesis.

### Avoid

- Reading M2 as 'feedback is useless' -- it is specifically that feedback does not beat matched-compute sampling for THIS 4B on THIS substrate.
- Still one substrate + one model; generalization across substrates/models is untested.

## C12: The fixed 4B's compositional frontier extends without a teacher via tool-augmented search + banking

- Status: `Promising`
- Programs: `structured_execution_and_compilers`, `posttraining_and_adaptation`
- Summary: On the contamination-free depth-graded program-synthesis substrate, C11/M4 left the depth-3 frontier uncrackable by self-training (coverage-bounded). A decompose-and-compose search (the 4B ranks the next primitive via a letter-logit read of current-state->target; the interpreter executes it to materialize the intermediate state; recurse/backtrack over 23 primitives) CRACKS it: hidden-generalizing depth-3 solve rate monolithic 0.125 -> decompose 0.40-0.43 (3.4x). BUT against the brute-force bar (23 primitives -> blind enumeration already solves depth-3), the model's GUIDANCE buys EFFICIENCY not COVERAGE: guided solves with ~2.5x fewer interpreter calls and wins the low-budget regime, but PLATEAUS (planner-wall -- where its ranking misses it never recovers) while brute-force enumeration matches/beats it at high budget (d2 guided 0.575 vs brute 0.875; d3 converge ~0.40). So the frontier crack is the composition-structure + interpreter, not the model's planning -- the wall RELOCATES to the planner. Crucially, BANKING the search-found solutions (QLoRA-SFT on 327 (prompt->code) traces, no teacher) EXTENDS the frontier into the weights: monolithic held-out pass@5 0.125->0.237 (+0.112, ~2.6 SE), depth-3 pass@5 0.025->0.100 (4x), greedy@1 0.075->0.125 (+0.05, ~1.5 SE), no-think one-shot 0->0.062. This is the exact bound M4 (confined to the sampling distribution) could not break: search+interpreter harvested solutions OUTSIDE the sampling support and banking pulled them INTO it. REPLICATED: a second harvest seed reproduces banking (greedy 0.125 identical; pass@5 0.263; d3 pass@5 0.175). RETRO-AUDIT CORRECTION (behavioral min-depth): the generator did not exclude shallower-equivalent compositions -- 40% of nominal depth-3 tasks are behaviorally depth<=2. Re-sliced, decompose solved 16/16 collapsed but only 4/24 (17%) TRUE depth-3 (monolithic true-d3 = 0/24, and 0 corpus-wide); the banking eval is ~30% collapsed at d3 (mixed-population caveat). The frontier extension is real (17% > 0) but far more modest than nominal numbers; every prior 'depth-3' figure in the arc was inflated by this artifact. Verified-depth follow-up: depth-wall anatomy experiment.
- Implication: To extend a fixed small model's frontier without a teacher: use tool-augmented search (composition + an interpreter, both allowed calculators/algorithms) to harvest execution-verified solutions OUTSIDE its sampling support, then self-train on them. The model's own planning helps search EFFICIENCY but not COVERAGE -- the search structure + interpreter carry the crack, so don't credit the model with out-searching brute force.

### Evidence

- [`qwen35_4b_decompose_compose_frontier`](../../experiments/qwen35_4b_decompose_compose_frontier/reports/report.md)

### Next Tests

- Iterate as a frontier flywheel (harvest -> bank -> re-harvest with the improved model): does the extended frontier compound or re-saturate?
- Larger primitive vocabulary where brute-force enumeration explodes -- there the model's guidance must carry the search; does it?
- Deeper depths (4-5): does banked composition generalize to compositions deeper than any trained?

### Avoid

- Claiming the MODEL out-searches brute force -- on coverage it does not; guidance is an efficiency win.
- Overstating banking -- absolutes stay low; greedy@1 +0.05 is ~1.5 SE (pass@5 +0.112 is ~2.6 SE), one seed n=80.
- Using nominal composition depth as difficulty without a behavioral min-depth check -- 40% of random depth-3 compositions are shallower-equivalent.
